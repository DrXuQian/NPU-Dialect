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
#include "npu/Analysis/OperatorTilingSpec.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include <limits>

namespace npu {
#define GEN_PASS_DEF_NPUTEMPORALTILING
#include "npu/Transforms/Passes.h.inc"
} // namespace npu

using namespace mlir;
using namespace npu;

namespace {

/// Extract iteration domain dimension sizes for the given linalg op.
/// Returns a vector of size numLoops with the extent of each iteration dim.
static SmallVector<int64_t> getIterDomainSizes(linalg::LinalgOp op) {
  SmallVector<int64_t> sizes;
  // For linalg ops, we can use the output shape + indexing maps to determine
  // iteration domain sizes. As a practical shortcut, use the static loop
  // ranges which linalg provides.
  SmallVector<utils::IteratorType> iterTypes = op.getIteratorTypesArray();
  unsigned numLoops = iterTypes.size();
  sizes.resize(numLoops, 1);

  // Use the shapes of the operands projected through indexing maps.
  auto indexingMaps = op.getIndexingMapsArray();
  for (unsigned mapIdx = 0; mapIdx < indexingMaps.size(); ++mapIdx) {
    AffineMap map = indexingMaps[mapIdx];
    Value operand;
    if (mapIdx < op.getNumDpsInputs())
      operand = op.getDpsInputOperand(mapIdx)->get();
    else
      operand = op.getDpsInitOperand(mapIdx - op.getNumDpsInputs())->get();
    auto shapedType = dyn_cast<ShapedType>(operand.getType());
    if (!shapedType)
      continue;
    for (unsigned resIdx = 0; resIdx < map.getNumResults(); ++resIdx) {
      auto expr = map.getResult(resIdx);
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        unsigned pos = dimExpr.getPosition();
        if (pos < numLoops) {
          int64_t dimSize = shapedType.getDimSize(resIdx);
          if (dimSize > 0)
            sizes[pos] = std::max(sizes[pos], dimSize);
        }
      }
    }
  }
  return sizes;
}

/// Spec-driven tile search: use the OperatorTilingSpec's preferredTemporalDims
/// to determine which iteration domain dimensions to tile, generate candidates,
/// and pick the combination with lowest cost.
static TileConfig specDrivenTileSearch(linalg::LinalgOp op,
                                       const OperatorTilingSpec &spec,
                                       const CostModel &costModel) {
  static constexpr int64_t kInf = std::numeric_limits<int64_t>::max() / 2;
  TileConfig best;
  best.cost = {0, 0, kInf, 0, 0, 0};

  if (op->getNumResults() == 0)
    return best;
  auto outType = dyn_cast<ShapedType>(op->getResult(0).getType());
  if (!outType)
    return best;

  SmallVector<utils::IteratorType> iterTypes = op.getIteratorTypesArray();
  unsigned numLoops = iterTypes.size();

  // Get iteration domain sizes.
  SmallVector<int64_t> iterSizes = getIterDomainSizes(op);

  // Build candidate lists for each preferred temporal dim.
  // Each entry: (iterDim, candidates).
  SmallVector<std::pair<unsigned, SmallVector<int64_t>>> dimCandidates;
  for (unsigned prefIdx : spec.preferredTemporalDims) {
    if (prefIdx >= spec.splitDims.size())
      continue;
    unsigned iterDim = spec.splitDims[prefIdx].iterDim;
    if (iterDim >= numLoops)
      continue;
    int64_t dimSize = iterSizes[iterDim];
    if (dimSize <= 0)
      continue;
    auto cands = CostModel::tileCandidates(dimSize);
    dimCandidates.push_back({iterDim, std::move(cands)});
  }

  if (dimCandidates.empty())
    return best;

  int64_t dtypeBytes =
      std::max<int64_t>(1, outType.getElementTypeBitWidth() / 8);

  // For matmul: evaluate using evaluateMatmulTile
  if (auto matmul = dyn_cast<linalg::MatmulOp>(op.getOperation())) {
    auto lhsType = cast<ShapedType>(matmul.getInputs()[0].getType());
    int64_t M = outType.getDimSize(0);
    int64_t N = outType.getDimSize(1);
    int64_t K = lhsType.getDimSize(1);

    // Build a mapping from iterDim -> index into dimCandidates
    // iterDim 0 = M, 1 = N, 2 = K for matmul
    auto getCandidatesForIterDim = [&](unsigned iterDim,
                                       int64_t fullSize) -> SmallVector<int64_t> {
      for (auto &[dim, cands] : dimCandidates) {
        if (dim == iterDim)
          return cands;
      }
      // Not a preferred dim → use full size (no tiling on this dim)
      return {fullSize};
    };

    auto mCands = getCandidatesForIterDim(0, M);
    auto nCands = getCandidatesForIterDim(1, N);
    auto kCands = getCandidatesForIterDim(2, K);

    for (int64_t tm : mCands) {
      for (int64_t tn : nCands) {
        for (int64_t tk : kCands) {
          auto cost = costModel.evaluateMatmulTile(tm, tn, tk, M, N, K,
                                                   dtypeBytes);
          if (cost.totalCycles < best.cost.totalCycles) {
            // Store tile sizes in iteration domain coordinates
            best.tileSizes.assign(numLoops, 0);
            best.tileSizes[0] = tm;
            best.tileSizes[1] = tn;
            best.tileSizes[2] = tk;
            best.cost = cost;
          }
        }
      }
    }
    return best;
  }

  // For conv2d: evaluate using evaluateConv2dTile
  if (auto conv = dyn_cast<linalg::Conv2DNchwFchwOp>(op.getOperation())) {
    auto filterType = cast<ShapedType>(conv.getInputs()[1].getType());
    auto inputType = cast<ShapedType>(conv.getInputs()[0].getType());
    int64_t N_ = outType.getDimSize(0);
    int64_t Co = outType.getDimSize(1);
    int64_t Ho = outType.getDimSize(2);
    int64_t Wo = outType.getDimSize(3);
    int64_t Ci = inputType.getDimSize(1);
    int64_t Kh = filterType.getDimSize(2);
    int64_t Kw = filterType.getDimSize(3);
    int64_t stride[] = {1, 1};

    // Conv2d iteration domain: (d0=N, d1=Co, d2=Ho, d3=Wo, d4=Ci, d5=Kh, d6=Kw)
    auto getCandidatesForIterDim = [&](unsigned iterDim,
                                       int64_t fullSize) -> SmallVector<int64_t> {
      for (auto &[dim, cands] : dimCandidates) {
        if (dim == iterDim)
          return cands;
      }
      return {fullSize};
    };

    auto nCands = getCandidatesForIterDim(0, N_);
    auto coCands = getCandidatesForIterDim(1, Co);
    auto hCands = getCandidatesForIterDim(2, Ho);
    auto wCands = getCandidatesForIterDim(3, Wo);

    for (int64_t tn : nCands) {
      for (int64_t tco : coCands) {
        for (int64_t th : hCands) {
          for (int64_t tw : wCands) {
            auto cost = costModel.evaluateConv2dTile(tn, tco, th, tw,
                                                     N_, Co, Ho, Wo, Ci,
                                                     Kh, Kw, stride,
                                                     dtypeBytes);
            if (cost.totalCycles < best.cost.totalCycles) {
              // Store in iteration domain coordinates.
              // Conv2d iteration domain has 7 dims; we tile the first 4.
              best.tileSizes.assign(numLoops, 0);
              best.tileSizes[0] = tn;
              best.tileSizes[1] = tco;
              best.tileSizes[2] = th;
              best.tileSizes[3] = tw;
              best.cost = cost;
            }
          }
        }
      }
    }
    return best;
  }

  // For generic/elementwise ops: tile along the preferred dims
  if (isa<linalg::GenericOp>(op.getOperation())) {
    // Simple 1D tiling along the first preferred dim (same approach as
    // bestTileConfig but using the spec's preferred dimension).
    if (!dimCandidates.empty()) {
      unsigned tileIterDim = dimCandidates[0].first;
      int64_t dimSize = iterSizes[tileIterDim];
      int64_t elemsPerRow = 1;
      for (unsigned i = 0; i < static_cast<unsigned>(outType.getRank()); ++i) {
        if (i != tileIterDim)
          elemsPerRow *= outType.getDimSize(i);
      }
      int64_t bytesPerRow = elemsPerRow * dtypeBytes * 3;

      for (int64_t td : dimCandidates[0].second) {
        int64_t sram = td * bytesPerRow * 2;
        if (sram > costModel.target().sramPerCore)
          continue;
        int64_t numTiles = (dimSize + td - 1) / td;
        int64_t tileCompute = std::max<int64_t>(
            1, (td * elemsPerRow) / costModel.target().dspThroughput);
        int64_t tileDma = costModel.dmaCycles(td * bytesPerRow);
        int64_t total;
        if (numTiles <= 1)
          total = tileDma + tileCompute + tileDma;
        else {
          int64_t steady = std::max(tileDma, tileCompute);
          total = tileDma + (numTiles - 1) * steady + tileCompute + tileDma;
        }
        if (total < best.cost.totalCycles) {
          best.tileSizes.assign(numLoops, 0);
          best.tileSizes[tileIterDim] = td;
          best.cost = {numTiles * tileCompute, numTiles * tileDma * 2,
                       total, sram, numTiles,
                       numTiles * td * bytesPerRow * 2};
        }
      }
    }
    return best;
  }

  return best;
}

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
      if (isa<linalg::MatmulOp, linalg::BatchMatmulOp,
              linalg::Conv2DNchwFchwOp, linalg::TransposeOp,
              linalg::GenericOp>(op.getOperation()))
        targets.push_back(op);
    });

    for (auto op : targets) {
      TileConfig config;
      config.cost = {0, 0, std::numeric_limits<int64_t>::max() / 2, 0, 0, 0};

      // Try spec-driven tiling search first.
      const OperatorTilingSpec *spec = getTilingSpec(op);
      if (spec && !spec->preferredTemporalDims.empty()) {
        config = specDrivenTileSearch(op, *spec, costModel);
      }

      // Fallback to existing bestTileConfig if spec is unavailable or
      // spec-driven search didn't find a valid config.
      if (config.tileSizes.empty() || !config.cost.isValid())
        config = costModel.bestTileConfig(op);

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

      // Map tile sizes into iteration domain coordinates (pad to numLoops
      // with 0). The spec-driven path already returns iteration-domain
      // coordinates; the fallback bestTileConfig returns output coordinates
      // that we map positionally.
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

        // Peel remainder iterations to keep all shapes static.
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
