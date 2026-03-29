//===- TemporalTilingPass.cpp - Intra-core tiling for SRAM ----*- C++ -*-===//
//
// Further tiles each core's work so the working set fits in SRAM.
// Uses CostModel.bestTileConfig() to search for optimal tile sizes
// considering DMA/compute overlap with double buffering.
//
// Tile sizes are specified in iteration domain coordinates:
//   matmul:              (M, N, K)  — 3 dims
//   conv2d_nchw_fchw:   (N, F, OH, OW, C, KH, KW) — 7 dims
//   generic:             matches iterator_types count
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
#define GEN_PASS_DEF_NPUTEMPORALTILING
#include "npu/Transforms/Passes.h.inc"
} // namespace npu

using namespace mlir;
using namespace npu;

namespace {

struct NPUTemporalTilingPass
    : public npu::impl::NPUTemporalTilingBase<NPUTemporalTilingPass> {
  using NPUTemporalTilingBase::NPUTemporalTilingBase;

  void runOnOperation() override {
    auto funcOp = getOperation();

    Operation *module = funcOp->getParentOfType<ModuleOp>();
    if (!module) module = funcOp;
    CostModelAnalysis analysis(module);
    const CostModel &costModel = analysis.getCostModel();

    SmallVector<linalg::LinalgOp> targets;
    funcOp.walk([&](linalg::LinalgOp op) {
      if (isa<linalg::MatmulOp, linalg::Conv2DNchwFchwOp,
              linalg::GenericOp>(op.getOperation()))
        targets.push_back(op);
    });

    for (auto op : targets) {
      TileConfig config = costModel.bestTileConfig(op);
      if (config.tileSizes.empty() || !config.cost.isValid())
        continue;
      if (config.cost.numTiles <= 1)
        continue;

      if (op->getNumResults() == 0)
        continue;
      auto outType = dyn_cast<ShapedType>(op->getResult(0).getType());
      if (!outType)
        continue;

      SmallVector<utils::IteratorType> iterTypes = op.getIteratorTypesArray();
      unsigned numLoops = iterTypes.size();

      // Cost model returns tile sizes in output-tensor coordinates.
      // Map them into iteration domain coordinates (pad to numLoops with 0).
      SmallVector<OpFoldResult> tileSizes(numLoops,
                                           OpBuilder(op).getIndexAttr(0));
      for (size_t i = 0; i < config.tileSizes.size() && i < numLoops; ++i) {
        tileSizes[i] = OpBuilder(op).getIndexAttr(config.tileSizes[i]);
      }

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
