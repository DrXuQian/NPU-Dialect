//===- FusionPass.cpp - Cost-model driven linalg fusion -------*- C++ -*-===//
//
// Fuses producer->consumer linalg pairs when CostModel says it's profitable.
// Uses elementwise fusion for generic->generic pairs, and tile-and-fuse
// for matmul/conv->generic pairs (tile the consumer, fuse the producer).
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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace npu {
#define GEN_PASS_DEF_NPUFUSION
#include "npu/Transforms/Passes.h.inc"
} // namespace npu

using namespace mlir;
using namespace npu;

namespace {

struct NPUFusionPass : public npu::impl::NPUFusionBase<NPUFusionPass> {

  void runOnOperation() override {
    auto funcOp = getOperation();

    // Build cost model from module-level target attribute
    Operation *module = funcOp->getParentOfType<ModuleOp>();
    if (!module) module = funcOp;
    CostModelAnalysis analysis(module);
    const CostModel &costModel = analysis.getCostModel();

    // Phase 1: Tile-and-fuse for non-generic producer -> generic consumer
    // (e.g. matmul -> relu). Tile the consumer, then fuse the matmul
    // producer into the tile loop.
    {
      SmallVector<linalg::LinalgOp> candidates;
      funcOp.walk([&](linalg::GenericOp consumer) {
        for (OpOperand &operand : consumer->getOpOperands()) {
          auto producer =
              operand.get().getDefiningOp<linalg::LinalgOp>();
          if (!producer)
            continue;
          if (!producer->hasOneUse())
            continue;
          if (isa<linalg::GenericOp>(producer.getOperation()))
            continue;

          FusionDecision decision =
              costModel.isFusionProfitable(producer, consumer);
          if (!decision.shouldFuse)
            continue;

          // Check tile alignment before adding to candidates.
          SmallVector<Operation *> pair = {producer.getOperation(),
                                           consumer.getOperation()};
          auto alignment = costModel.checkTileAlignment(pair);
          if (!alignment.aligned &&
              alignment.alignmentOverheadBytes >
                  costModel.target().sramPerCore / 4)
            continue;

          candidates.push_back(
              cast<linalg::LinalgOp>(consumer.getOperation()));
          return;
        }
      });

      for (auto op : candidates) {
        // Get best tile config from cost model
        TileConfig config = costModel.bestTileConfig(op);
        if (config.tileSizes.empty() || !config.cost.isValid())
          continue;
        if (config.cost.numTiles <= 1)
          continue;

        auto outType = dyn_cast<ShapedType>(op->getResult(0).getType());
        if (!outType)
          continue;

        // Build tile sizes matching the iteration domain
        SmallVector<OpFoldResult> tileSizes;
        OpBuilder builder(op);
        for (size_t i = 0; i < config.tileSizes.size() &&
                            i < static_cast<size_t>(outType.getRank()); ++i) {
          tileSizes.push_back(builder.getIndexAttr(config.tileSizes[i]));
        }
        while (tileSizes.size() < static_cast<size_t>(outType.getRank()))
          tileSizes.push_back(builder.getIndexAttr(0));

        // Tile the consumer
        auto tilingOptions =
            scf::SCFTilingOptions().setTileSizes(tileSizes);
        IRRewriter rewriter(op->getContext());
        rewriter.setInsertionPoint(op);
        FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
            rewriter, cast<TilingInterface>(op.getOperation()),
            tilingOptions);
        if (failed(tilingResult))
          continue;

        // Fuse non-generic producers into the tile loop.
        // The generated slices include extract_slice ops that read from
        // the producer results. We fuse each such producer.
        SmallVector<LoopLikeOpInterface> loops(
            tilingResult->loops.begin(), tilingResult->loops.end());
        for (auto *sliceOp : tilingResult->generatedSlices) {
          auto extractSlice =
              dyn_cast<tensor::ExtractSliceOp>(sliceOp);
          if (!extractSlice)
            continue;
          Value source = extractSlice.getSource();
          // Only fuse linalg producers that are not generic
          // (we want to fuse matmul/conv into the loop)
          auto producerLinalgOp =
              source.getDefiningOp<linalg::LinalgOp>();
          if (!producerLinalgOp)
            continue;
          if (isa<linalg::GenericOp>(producerLinalgOp.getOperation()))
            continue;

          std::optional<scf::SCFFuseProducerOfSliceResult> fuseResult =
              scf::tileAndFuseProducerOfSlice(
                  rewriter, extractSlice, loops);
          (void)fuseResult;
        }

        rewriter.replaceOp(op, tilingResult->replacements);
      }
    }

    // Phase 2: Elementwise fusion for generic->generic pairs
    // (existing path using fuseElementwiseOps)
    {
      bool changed = true;
      while (changed) {
        changed = false;
        funcOp.walk([&](linalg::LinalgOp consumer) {
          if (changed)
            return; // one fusion per iteration for simplicity

          for (OpOperand &operand : consumer->getOpOperands()) {
            auto producer =
                operand.get().getDefiningOp<linalg::LinalgOp>();
            if (!producer)
              continue;

            // Single-use check: only fuse if producer has exactly one consumer
            if (!producer->hasOneUse())
              continue;

            // Ask cost model
            FusionDecision decision =
                costModel.isFusionProfitable(producer, consumer);
            if (!decision.shouldFuse)
              continue;

            // Check tile alignment before fusing.
            SmallVector<Operation *> pair = {producer.getOperation(),
                                             consumer.getOperation()};
            auto alignment = costModel.checkTileAlignment(pair);
            if (!alignment.aligned &&
                alignment.alignmentOverheadBytes >
                    costModel.target().sramPerCore / 4)
              continue;

            // fuseElementwiseOps requires both producer and consumer to be
            // GenericOp.
            auto genericProducer =
                dyn_cast<linalg::GenericOp>(producer.getOperation());
            auto genericConsumer =
                dyn_cast<linalg::GenericOp>(consumer.getOperation());
            if (!genericProducer || !genericConsumer)
              continue;

            // Use MLIR's linalg fusion utilities
            IRRewriter rewriter(consumer->getContext());
            rewriter.setInsertionPoint(consumer);
            FailureOr<linalg::ElementwiseOpFusionResult> fusionResult =
                linalg::fuseElementwiseOps(rewriter, &operand);
            if (succeeded(fusionResult)) {
              consumer->replaceAllUsesWith(
                  fusionResult->fusedOp->getResults());
              consumer->erase();
              if (producer->use_empty())
                producer->erase();
              changed = true;
              return; // restart walk
            }
          }
        });
      }
    }
  }
};

} // namespace
