//===- SpatialTilingPass.cpp - Inter-core distribution --------*- C++ -*-===//
//
// Tiles linalg ops and distributes across cores using scf.for.
// Uses CostModel.evaluateSpatialSplit() to pick the best split dimension.
//
// Key correctness requirement:
//   For conv2d, tiling the output H dimension requires extracting an
//   input slice with halo (extra rows for the kernel). The TilingInterface
//   handles this automatically when tile sizes are specified in the
//   iteration domain coordinate system (not the output tensor rank).
//
//   conv2d_nchw_fchw has 7 iteration dims: (N, F, OH, OW, C, KH, KW)
//   - Parallel:  N(0), F(1), OH(2), OW(3)
//   - Reduction: C(4), KH(5), KW(6)
//   Tile sizes must be length 7, with 0 for untiled dims.
//
//===----------------------------------------------------------------------===//

#include "npu/Transforms/Passes.h"
#include "npu/Analysis/CostModel.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

namespace npu {
#define GEN_PASS_DEF_NPUSPATIALTILING
#include "npu/Transforms/Passes.h.inc"
} // namespace npu

using namespace mlir;
using namespace npu;

namespace {

/// Map an output-tensor dimension index to the corresponding iteration
/// domain dimension for a linalg op.  For matmul (M,N,K) the output
/// is (M,N) → iter dims 0,1.  For conv2d_nchw_fchw the output is
/// (N,F,OH,OW) → iter dims 0,1,2,3.  For generic ops the mapping is
/// usually identity.
static unsigned outputDimToIterDim(linalg::LinalgOp op, unsigned outDim) {
  // For most ops, output dims map 1:1 to the first output-rank iteration dims.
  // This is correct for matmul, conv2d named ops, and most generics.
  (void)op;
  return outDim;
}

struct NPUSpatialTilingPass
    : public npu::impl::NPUSpatialTilingBase<NPUSpatialTilingPass> {
  using NPUSpatialTilingBase::NPUSpatialTilingBase;

  void runOnOperation() override {
    auto funcOp = getOperation();

    Operation *module = funcOp->getParentOfType<ModuleOp>();
    if (!module) module = funcOp;
    CostModelAnalysis analysis(module);
    const CostModel &costModel = analysis.getCostModel();

    int64_t cores = numCores;

    SmallVector<linalg::LinalgOp> ops;
    funcOp.walk([&](linalg::LinalgOp op) {
      if (!op->getParentOfType<scf::ForOp>() &&
          !op->getParentOfType<scf::ForallOp>()) {
        if (isa<linalg::MatmulOp, linalg::Conv2DNchwFchwOp,
                linalg::GenericOp>(op.getOperation()))
          ops.push_back(op);
      }
    });

    for (auto op : ops) {
      if (op->getNumResults() == 0)
        continue;
      auto outType = dyn_cast<ShapedType>(op->getResult(0).getType());
      if (!outType || !outType.hasStaticShape())
        continue;

      SmallVector<utils::IteratorType> iterTypes = op.getIteratorTypesArray();
      unsigned numLoops = iterTypes.size();

      // --- Search: evaluate each parallel output dimension ---
      unsigned bestOutDim = 0;
      int64_t bestCost = std::numeric_limits<int64_t>::max();

      for (unsigned outDim = 0;
           outDim < static_cast<unsigned>(outType.getRank()); ++outDim) {
        unsigned iterDim = outputDimToIterDim(op, outDim);
        if (iterDim >= numLoops)
          continue;
        if (iterTypes[iterDim] == utils::IteratorType::reduction)
          continue;

        ScheduleCost cost = costModel.evaluateSpatialSplit(op, outDim, cores);
        if (cost.totalCycles < bestCost) {
          bestCost = cost.totalCycles;
          bestOutDim = outDim;
        }
      }

      unsigned bestIterDim = outputDimToIterDim(op, bestOutDim);
      int64_t dimSize = outType.getDimSize(bestOutDim);
      int64_t tileSize = (dimSize + cores - 1) / cores;

      // Build tile sizes in iteration domain coordinates (length = numLoops)
      SmallVector<OpFoldResult> tileSizes(numLoops,
                                           OpBuilder(op).getIndexAttr(0));
      tileSizes[bestIterDim] = OpBuilder(op).getIndexAttr(tileSize);

      auto tilingOptions = scf::SCFTilingOptions().setTileSizes(tileSizes);

      IRRewriter rewriter(op->getContext());
      rewriter.setInsertionPoint(op);
      FailureOr<scf::SCFTilingResult> tilingResult =
          scf::tileUsingSCF(rewriter,
                            cast<TilingInterface>(op.getOperation()),
                            tilingOptions);
      if (succeeded(tilingResult)) {
        rewriter.replaceOp(op, tilingResult->replacements);
      }
    }
  }
};

} // namespace
