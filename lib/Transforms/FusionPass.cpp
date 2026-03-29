//===- FusionPass.cpp - Greedy vertical fusion for NPU --------*- C++ -*-===//
//
// Implements the greedy baseline vertical fusion algorithm:
//   1. Sort ops by compute cost (most expensive first)
//   2. Greedily expand each anchor op by adding profitable neighbors
//   3. Apply pairwise fusion transformations for each group
//
// Uses evaluateSubgraph() for cost-based decisions and the existing
// tile-and-fuse / elementwise fusion infrastructure for transformations.
//
//===----------------------------------------------------------------------===//

#include "npu/Transforms/Passes.h"
#include "npu/Analysis/CostModel.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include <algorithm>

namespace npu {
#define GEN_PASS_DEF_NPUFUSION
#include "npu/Transforms/Passes.h.inc"
} // namespace npu

using namespace mlir;
using namespace npu;

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

static constexpr unsigned kMaxFusionLength = 6;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Check if an op is a "real" compute op worth considering for fusion.
/// Skips fill ops which are trivially handled by the transformation.
static bool isComputeOp(Operation *op) {
  if (isa<linalg::FillOp>(op))
    return false;
  return isa<linalg::LinalgOp>(op);
}

/// Check if an op is a matrix-unit op (matmul or conv).
static bool isMatrixOp(Operation *op) {
  return isa<linalg::MatmulOp>(op) || isa<linalg::Conv2DNchwFchwOp>(op);
}

/// Get single-use linalg consumers of an op (ops that use exactly one result
/// of `op`, and that result has a single use).
static SmallVector<Operation *>
getSingleUseConsumers(Operation *op,
                      const llvm::DenseSet<Operation *> &linalgSet) {
  SmallVector<Operation *> consumers;
  for (Value result : op->getResults()) {
    if (!result.hasOneUse())
      continue;
    Operation *user = *result.getUsers().begin();
    // Walk up to find the actual linalg consumer (may be through
    // tensor.empty/fill chain).
    if (linalgSet.contains(user) && isComputeOp(user))
      consumers.push_back(user);
  }
  return consumers;
}

/// Get single-use linalg producers of an op (ops whose result flows into
/// `op` as an operand, and that result has a single use).
static SmallVector<Operation *>
getSingleUseProducers(Operation *op,
                      const llvm::DenseSet<Operation *> &linalgSet) {
  SmallVector<Operation *> producers;
  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp || !linalgSet.contains(defOp))
      continue;
    if (!isComputeOp(defOp))
      continue;
    if (!defOp->hasOneUse())
      continue;
    producers.push_back(defOp);
  }
  return producers;
}

/// Check basic shape compatibility for fusion: the connecting tensor
/// between producer and consumer must have static shapes.
static bool areShapesCompatible(Operation *producer, Operation *consumer) {
  for (Value result : producer->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (user == consumer) {
        auto shapedType = dyn_cast<ShapedType>(result.getType());
        if (!shapedType || !shapedType.hasStaticShape())
          return false;
        return true;
      }
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Fusion group: ordered list of ops to fuse together
//===----------------------------------------------------------------------===//

struct FusionGroup {
  SmallVector<Operation *> ops;

  bool contains(Operation *op) const {
    return llvm::is_contained(ops, op);
  }
};

//===----------------------------------------------------------------------===//
// Greedy fusion algorithm
//===----------------------------------------------------------------------===//

/// Build fusion groups using the greedy algorithm:
///   1. Sort ops by compute cost (descending)
///   2. For each unvisited anchor, expand forward then backward
///   3. Accept a neighbor if fused cost < unfused cost
static SmallVector<FusionGroup>
buildFusionGroups(func::FuncOp funcOp, const CostModel &costModel) {
  // Step 1: Collect all linalg ops.
  SmallVector<Operation *> allLinalgOps;
  funcOp.walk([&](linalg::LinalgOp op) {
    if (isComputeOp(op.getOperation()))
      allLinalgOps.push_back(op.getOperation());
  });

  if (allLinalgOps.empty())
    return {};

  llvm::DenseSet<Operation *> linalgSet(allLinalgOps.begin(),
                                         allLinalgOps.end());

  // Step 2: Sort by compute cost (most expensive first).
  SmallVector<std::pair<int64_t, Operation *>> costPairs;
  for (Operation *op : allLinalgOps) {
    int64_t cost = costModel.computeCycles(op);
    costPairs.push_back({cost, op});
  }
  llvm::sort(costPairs, [](const auto &a, const auto &b) {
    return a.first > b.first; // descending
  });

  // Step 3: Greedy expansion.
  llvm::DenseSet<Operation *> visited;
  SmallVector<FusionGroup> groups;

  for (auto &[anchorCost, anchorOp] : costPairs) {
    if (visited.contains(anchorOp))
      continue;

    FusionGroup group;
    group.ops.push_back(anchorOp);

    // --- Forward expansion (consumers) ---
    while (group.ops.size() < kMaxFusionLength) {
      Operation *lastOp = group.ops.back();
      SmallVector<Operation *> candidates =
          getSingleUseConsumers(lastOp, linalgSet);

      Operation *bestCandidate = nullptr;
      int64_t bestBenefit = 0;

      for (Operation *candidate : candidates) {
        if (visited.contains(candidate) || group.contains(candidate))
          continue;

        // Basic shape compatibility check.
        if (!areShapesCompatible(lastOp, candidate))
          continue;

        // Check tile alignment.
        SmallVector<Operation *> pair = {lastOp, candidate};
        auto alignment = costModel.checkTileAlignment(pair);
        if (!alignment.aligned &&
            alignment.alignmentOverheadBytes >
                costModel.target().sramPerCore / 4)
          continue;

        // Evaluate cost: fused group+candidate vs group + candidate separately.
        SmallVector<Operation *> trial(group.ops.begin(), group.ops.end());
        trial.push_back(candidate);

        int64_t fusedCost =
            costModel.evaluateSubgraph(trial).pipelinedCycles;

        int64_t groupCost =
            costModel.evaluateSubgraph(group.ops).pipelinedCycles;
        SmallVector<Operation *> single = {candidate};
        int64_t candidateCost =
            costModel.evaluateSubgraph(single).pipelinedCycles;
        int64_t unfusedCost = groupCost + candidateCost;

        int64_t benefit = unfusedCost - fusedCost;
        if (benefit > bestBenefit) {
          bestCandidate = candidate;
          bestBenefit = benefit;
        }
      }

      if (bestCandidate)
        group.ops.push_back(bestCandidate);
      else
        break;
    }

    // --- Backward expansion (producers) ---
    while (group.ops.size() < kMaxFusionLength) {
      Operation *firstOp = group.ops.front();
      SmallVector<Operation *> candidates =
          getSingleUseProducers(firstOp, linalgSet);

      Operation *bestCandidate = nullptr;
      int64_t bestBenefit = 0;

      for (Operation *candidate : candidates) {
        if (visited.contains(candidate) || group.contains(candidate))
          continue;

        // Basic shape compatibility check.
        if (!areShapesCompatible(candidate, firstOp))
          continue;

        // Check tile alignment.
        SmallVector<Operation *> pair = {candidate, firstOp};
        auto alignment = costModel.checkTileAlignment(pair);
        if (!alignment.aligned &&
            alignment.alignmentOverheadBytes >
                costModel.target().sramPerCore / 4)
          continue;

        // Evaluate cost: candidate+group fused vs separate.
        SmallVector<Operation *> trial;
        trial.push_back(candidate);
        trial.append(group.ops.begin(), group.ops.end());

        int64_t fusedCost =
            costModel.evaluateSubgraph(trial).pipelinedCycles;

        int64_t groupCost =
            costModel.evaluateSubgraph(group.ops).pipelinedCycles;
        SmallVector<Operation *> single = {candidate};
        int64_t candidateCost =
            costModel.evaluateSubgraph(single).pipelinedCycles;
        int64_t unfusedCost = groupCost + candidateCost;

        int64_t benefit = unfusedCost - fusedCost;
        if (benefit > bestBenefit) {
          bestCandidate = candidate;
          bestBenefit = benefit;
        }
      }

      if (bestCandidate)
        group.ops.insert(group.ops.begin(), bestCandidate);
      else
        break;
    }

    // Mark all ops in the group as visited.
    for (Operation *op : group.ops)
      visited.insert(op);

    // Only record groups with more than 1 op.
    if (group.ops.size() > 1)
      groups.push_back(std::move(group));
  }

  return groups;
}

//===----------------------------------------------------------------------===//
// Fusion transformations
//===----------------------------------------------------------------------===//

/// Apply tile-and-fuse: tile the consumer, then fuse the non-generic producer
/// into the tile loop. Used for matmul/conv -> generic patterns.
static bool applyTileAndFuse(linalg::LinalgOp consumer,
                             const CostModel &costModel) {
  TileConfig config = costModel.bestTileConfig(consumer);
  if (config.tileSizes.empty() || !config.cost.isValid())
    return false;
  if (config.cost.numTiles <= 1)
    return false;

  auto outType = dyn_cast<ShapedType>(consumer->getResult(0).getType());
  if (!outType)
    return false;

  // Build tile sizes matching the output rank.
  SmallVector<OpFoldResult> tileSizes;
  OpBuilder builder(consumer);
  for (size_t i = 0; i < config.tileSizes.size() &&
                      i < static_cast<size_t>(outType.getRank());
       ++i) {
    tileSizes.push_back(builder.getIndexAttr(config.tileSizes[i]));
  }
  while (tileSizes.size() < static_cast<size_t>(outType.getRank()))
    tileSizes.push_back(builder.getIndexAttr(0));

  // Tile the consumer.
  auto tilingOptions = scf::SCFTilingOptions().setTileSizes(tileSizes);
  IRRewriter rewriter(consumer->getContext());
  rewriter.setInsertionPoint(consumer);
  FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(consumer.getOperation()), tilingOptions);
  if (failed(tilingResult))
    return false;

  // Fuse non-generic producers into the tile loop.
  SmallVector<LoopLikeOpInterface> loops(tilingResult->loops.begin(),
                                         tilingResult->loops.end());
  for (auto *sliceOp : tilingResult->generatedSlices) {
    auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(sliceOp);
    if (!extractSlice)
      continue;
    Value source = extractSlice.getSource();
    auto producerLinalgOp = source.getDefiningOp<linalg::LinalgOp>();
    if (!producerLinalgOp)
      continue;
    if (isa<linalg::GenericOp>(producerLinalgOp.getOperation()))
      continue;

    std::optional<scf::SCFFuseProducerOfSliceResult> fuseResult =
        scf::tileAndFuseProducerOfSlice(rewriter, extractSlice, loops);
    (void)fuseResult;
  }

  rewriter.replaceOp(consumer, tilingResult->replacements);
  return true;
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct NPUFusionPass : public npu::impl::NPUFusionBase<NPUFusionPass> {

  void runOnOperation() override {
    auto funcOp = getOperation();

    // Build cost model from module-level target attribute.
    Operation *module = funcOp->getParentOfType<ModuleOp>();
    if (!module)
      module = funcOp;
    CostModelAnalysis analysis(module);
    const CostModel &costModel = analysis.getCostModel();

    // ================================================================
    // Phase 1: Build fusion groups using the greedy algorithm.
    // ================================================================
    SmallVector<FusionGroup> groups = buildFusionGroups(funcOp, costModel);

    // ================================================================
    // Phase 2: Apply pairwise fusion transformations.
    //
    // For each group, we apply fusions pairwise along the chain.
    // The OutlineFusedGroupsPass (which runs after) handles multi-op
    // outlining into kernel functions.
    //
    // Strategy:
    //   - If pair is (matmul/conv, generic): tile-and-fuse
    //   - If pair is (generic, generic): elementwise fusion
    //   - Otherwise: skip (let OutlineFusedGroupsPass handle it)
    // ================================================================

    // We process groups in order. For each group, walk the chain of ops
    // and fuse adjacent pairs. We work from the end of the chain backward
    // for tile-and-fuse (tile consumer, fuse producer) and forward for
    // elementwise fusion.

    // Collect tile-and-fuse candidates from groups.
    // These are (consumer) ops that have a non-generic producer in
    // the same group.
    SmallVector<linalg::LinalgOp> tileAndFuseCandidates;
    // Collect elementwise pairs: (producer, consumer) both generic.
    SmallVector<std::pair<linalg::GenericOp, linalg::GenericOp>>
        elementwisePairs;

    for (auto &group : groups) {
      for (size_t i = 0; i + 1 < group.ops.size(); ++i) {
        Operation *producer = group.ops[i];
        Operation *consumer = group.ops[i + 1];

        // Check if there's a direct producer->consumer connection.
        bool connected = false;
        for (Value result : producer->getResults()) {
          for (Operation *user : result.getUsers()) {
            if (user == consumer) {
              connected = true;
              break;
            }
          }
          if (connected)
            break;
        }
        if (!connected)
          continue;

        auto genericProducer =
            dyn_cast<linalg::GenericOp>(producer);
        auto genericConsumer =
            dyn_cast<linalg::GenericOp>(consumer);

        if (isMatrixOp(producer) && genericConsumer) {
          // matmul/conv -> generic: tile-and-fuse
          tileAndFuseCandidates.push_back(
              cast<linalg::LinalgOp>(consumer));
        } else if (genericProducer && genericConsumer) {
          // generic -> generic: elementwise fusion
          elementwisePairs.push_back({genericProducer, genericConsumer});
        }
        // Other patterns (e.g., generic -> matmul) are not fused here;
        // they'll be handled by OutlineFusedGroupsPass.
      }
    }

    // Apply tile-and-fuse transformations.
    for (auto consumer : tileAndFuseCandidates) {
      // Verify the op is still valid (not erased by a prior fusion).
      if (consumer->getBlock() == nullptr)
        continue;
      applyTileAndFuse(consumer, costModel);
    }

    // Apply elementwise fusions.
    // We iterate in a fixed-point loop because one fusion may enable
    // another (the fused op inherits operands from both producer and
    // consumer).
    {
      bool changed = true;
      while (changed) {
        changed = false;
        funcOp.walk([&](linalg::GenericOp consumer) {
          if (changed)
            return; // one fusion per iteration for stability

          for (OpOperand &operand : consumer->getOpOperands()) {
            auto producer =
                operand.get().getDefiningOp<linalg::GenericOp>();
            if (!producer)
              continue;
            if (!producer->hasOneUse())
              continue;

            // Use the cost model to check profitability.
            SmallVector<Operation *> pair = {producer.getOperation(),
                                             consumer.getOperation()};
            FusionDecision decision =
                costModel.isSubgraphFusionProfitable(pair);
            if (!decision.shouldFuse)
              continue;

            // Apply elementwise fusion.
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
