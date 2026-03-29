//===- SpatialTilingPass.cpp - Inter-core distribution --------*- C++ -*-===//
//
// Tiles linalg ops and distributes across cores.
// Uses OperatorTilingSpec to understand per-op split semantics:
//   - Which dimensions are parallel vs reduction
//   - Which inputs are split vs shared vs need halo
//   - Data reuse patterns for DMA cost estimation
//
// Uses CostModel to evaluate the cost of each candidate split dimension,
// considering the spec's reuse ratio and shared-data DMA savings.
//
//===----------------------------------------------------------------------===//

#include "npu/Transforms/Passes.h"
#include "npu/Analysis/CostModel.h"
#include "npu/Analysis/OperatorTilingSpec.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"

namespace npu {
#define GEN_PASS_DEF_NPUSPATIALTILING
#include "npu/Transforms/Passes.h.inc"
} // namespace npu

using namespace mlir;
using namespace npu;

namespace {

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
        if (isa<linalg::MatmulOp, linalg::BatchMatmulOp,
                linalg::Conv2DNchwFchwOp, linalg::TransposeOp,
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

      // Get per-operator tiling spec
      const OperatorTilingSpec *spec = getTilingSpec(op);

      // --- Pick the best split dimension ---
      unsigned bestIterDim = 0;
      int64_t bestCost = std::numeric_limits<int64_t>::max();

      if (spec) {
        // Use OperatorTilingSpec + detailed cost model
        for (const auto &dimSpec : spec->splitDims) {
          if (!dimSpec.isParallel)
            continue; // skip reduction dims for spatial tiling
          if (dimSpec.iterDim >= numLoops)
            continue;

          int64_t dimSize = 0;
          if (dimSpec.outputDim < static_cast<unsigned>(outType.getRank()))
            dimSize = outType.getDimSize(dimSpec.outputDim);
          if (dimSize <= 1)
            continue;

          // Precise cost considering per-operand slice/shared/halo semantics
          auto splitCost = costModel.evaluateSpatialSplitDetailed(
              op, dimSpec, cores);

          op->emitRemark("spatial split candidate: iter_dim=")
              << dimSpec.iterDim
              << " out_dim=" << dimSpec.outputDim
              << " per_core_dma=" << splitCost.perCoreDmaInBytes << "B"
              << " shared=" << splitCost.sharedDataBytes << "B"
              << " split=" << splitCost.splitDataBytes << "B"
              << " halo=" << splitCost.haloOverheadBytes << "B"
              << " reduce=" << splitCost.needsReduce
              << " total=" << splitCost.totalCycles;

          if (splitCost.totalCycles < bestCost) {
            bestCost = splitCost.totalCycles;
            bestIterDim = dimSpec.iterDim;
          }
        }
      } else {
        // Fallback: use output tensor rank heuristic
        for (unsigned outDim = 0;
             outDim < static_cast<unsigned>(outType.getRank()); ++outDim) {
          if (outDim >= numLoops)
            continue;
          if (iterTypes[outDim] == utils::IteratorType::reduction)
            continue;

          ScheduleCost cost = costModel.evaluateSpatialSplit(op, outDim, cores);
          if (cost.totalCycles < bestCost) {
            bestCost = cost.totalCycles;
            bestIterDim = outDim;
          }
        }
      }

      // Find the output dim corresponding to bestIterDim
      unsigned bestOutDim = bestIterDim; // default
      if (spec) {
        for (const auto &dimSpec : spec->splitDims) {
          if (dimSpec.iterDim == bestIterDim) {
            if (dimSpec.outputDim < static_cast<unsigned>(outType.getRank()))
              bestOutDim = dimSpec.outputDim;
            break;
          }
        }
      }

      int64_t dimSize = outType.getDimSize(bestOutDim);
      int64_t tileSize = (dimSize + cores - 1) / cores;

      // Build tile sizes in iteration domain coordinates
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

        // Peel the remainder iteration to eliminate dynamic shapes.
        // scf.for 0 to 80 step 27 → scf.for 0 to 54 step 27 (static)
        //                           + peeled iteration at 54:80 (size 26, static)
        for (auto loop : tilingResult->loops) {
          if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation())) {
            scf::ForOp partialIteration;
            (void)scf::peelForLoopAndSimplifyBounds(rewriter, forOp,
                                                     partialIteration);
          }
        }
      }
    }
  }
};

} // namespace
