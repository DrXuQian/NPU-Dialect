//===- OperatorTilingSpec.cpp - Per-operator tiling specifications -*- C++ -*-===//
//
// Defines how each operator type splits inputs/outputs when tiled.
//
//===----------------------------------------------------------------------===//

#include "npu/Analysis/OperatorTilingSpec.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir;
using namespace npu;

//===----------------------------------------------------------------------===//
// matmul C[M,N] = A[M,K] * B[K,N]
// Iteration domain: (d0=M, d1=N, d2=K)
// Indexing maps: A(d0,d2), B(d2,d1), C(d0,d1)
//===----------------------------------------------------------------------===//

OperatorTilingSpec npu::getMatmulTilingSpec() {
  OperatorTilingSpec spec;
  spec.opName = "linalg.matmul";

  // --- Split M (dim 0): A splits rows, B shared, C splits rows ---
  {
    SplitDimSpec s;
    s.outputDim = 0;
    s.iterDim = 0;
    s.isParallel = true;
    s.inputRules = {
        {/*operandIdx=*/0, SliceKind::Split, /*sliceDim=*/0},  // A[tile_m, K]
        {/*operandIdx=*/1, SliceKind::Shared, /*sliceDim=*/0}, // B[K, N] full
    };
    s.outputRules = {
        {/*operandIdx=*/0, SliceKind::Split, /*sliceDim=*/0},  // C[tile_m, N]
    };
    // B is fully shared → high reuse
    s.reuseRatio = 2.0;
    spec.splitDims.push_back(s);
  }

  // --- Split N (dim 1): A shared, B splits cols, C splits cols ---
  {
    SplitDimSpec s;
    s.outputDim = 1;
    s.iterDim = 1;
    s.isParallel = true;
    s.inputRules = {
        {0, SliceKind::Shared, 0},  // A[M, K] full
        {1, SliceKind::Split, 1},   // B[K, tile_n]
    };
    s.outputRules = {
        {0, SliceKind::Split, 1},   // C[M, tile_n]
    };
    s.reuseRatio = 2.0;
    spec.splitDims.push_back(s);
  }

  // --- Split K (dim 2): reduction — both A and B split, output needs reduce ---
  {
    SplitDimSpec s;
    s.outputDim = UINT_MAX; // K is not an output dimension
    s.iterDim = 2;
    s.isParallel = false; // REDUCTION
    s.inputRules = {
        {0, SliceKind::Split, 1},   // A[M, tile_k]
        {1, SliceKind::Split, 0},   // B[tile_k, N]
    };
    s.outputRules = {
        {0, SliceKind::Reduce, 0},  // C[M, N] partial → needs accumulation
    };
    s.reuseRatio = 1.0; // no reuse benefit — both inputs split
    spec.splitDims.push_back(s);
  }

  spec.preferredSpatialDim = 0; // prefer split M for inter-core
  spec.preferredTemporalDims = {0, 1, 2}; // M, N, K

  return spec;
}

//===----------------------------------------------------------------------===//
// batch_matmul C[B,M,N] = A[B,M,K] * B[B,K,N]
// Iteration domain: (d0=B, d1=M, d2=N, d3=K)
// Indexing maps: A(d0,d1,d3), B(d0,d3,d2), C(d0,d1,d2)
//===----------------------------------------------------------------------===//

OperatorTilingSpec npu::getBatchMatmulTilingSpec() {
  OperatorTilingSpec spec;
  spec.opName = "linalg.batch_matmul";

  // --- Split B (batch, dim 0): parallel, all operands split on dim 0 ---
  {
    SplitDimSpec s;
    s.outputDim = 0;
    s.iterDim = 0;
    s.isParallel = true;
    s.inputRules = {
        {/*operandIdx=*/0, SliceKind::Split, /*sliceDim=*/0},  // A[tile_b, M, K]
        {/*operandIdx=*/1, SliceKind::Split, /*sliceDim=*/0},  // B[tile_b, K, N]
    };
    s.outputRules = {
        {/*operandIdx=*/0, SliceKind::Split, /*sliceDim=*/0},  // C[tile_b, M, N]
    };
    s.reuseRatio = 1.0; // both inputs split on batch
    spec.splitDims.push_back(s);
  }

  // --- Split M (dim 1): parallel, A splits rows, B shared, C splits rows ---
  {
    SplitDimSpec s;
    s.outputDim = 1;
    s.iterDim = 1;
    s.isParallel = true;
    s.inputRules = {
        {0, SliceKind::Split, 1},   // A[B, tile_m, K]
        {1, SliceKind::Shared, 0},  // B[B, K, N] full (shared across M tiles)
    };
    s.outputRules = {
        {0, SliceKind::Split, 1},   // C[B, tile_m, N]
    };
    s.reuseRatio = 2.0; // B is fully shared
    spec.splitDims.push_back(s);
  }

  // --- Split N (dim 2): parallel, A shared, B splits cols, C splits cols ---
  {
    SplitDimSpec s;
    s.outputDim = 2;
    s.iterDim = 2;
    s.isParallel = true;
    s.inputRules = {
        {0, SliceKind::Shared, 0},  // A[B, M, K] full (shared across N tiles)
        {1, SliceKind::Split, 2},   // B[B, K, tile_n]
    };
    s.outputRules = {
        {0, SliceKind::Split, 2},   // C[B, M, tile_n]
    };
    s.reuseRatio = 2.0; // A is fully shared
    spec.splitDims.push_back(s);
  }

  // --- Split K (dim 3): reduction ---
  {
    SplitDimSpec s;
    s.outputDim = UINT_MAX; // K is not an output dimension
    s.iterDim = 3;
    s.isParallel = false; // REDUCTION
    s.inputRules = {
        {0, SliceKind::Split, 2},   // A[B, M, tile_k]
        {1, SliceKind::Split, 1},   // B[B, tile_k, N]
    };
    s.outputRules = {
        {0, SliceKind::Reduce, 0},  // C[B, M, N] partial → needs accumulation
    };
    s.reuseRatio = 1.0;
    spec.splitDims.push_back(s);
  }

  spec.preferredSpatialDim = 1; // prefer split M for inter-core
  spec.preferredTemporalDims = {0, 1, 2, 3}; // B, M, N, K

  return spec;
}

//===----------------------------------------------------------------------===//
// conv2d_nchw_fchw  out[N,Co,Ho,Wo] = in[N,Ci,Hi,Wi] * filt[Co,Ci,Kh,Kw]
// Iteration domain: (d0=N, d1=F/Co, d2=OH, d3=OW, d4=C/Ci, d5=KH, d6=KW)
// Indexing maps: in(d0, d4, d2+d5, d3+d6), filt(d1, d4, d5, d6), out(d0,d1,d2,d3)
//===----------------------------------------------------------------------===//

OperatorTilingSpec npu::getConv2dNchwTilingSpec(
    int64_t Kh, int64_t Kw, int64_t strideH, int64_t strideW) {
  OperatorTilingSpec spec;
  spec.opName = "linalg.conv_2d_nchw_fchw";

  // --- Split N (batch, iter dim 0): parallel, input splits batch ---
  {
    SplitDimSpec s;
    s.outputDim = 0; // N in output
    s.iterDim = 0;
    s.isParallel = true;
    s.inputRules = {
        {0, SliceKind::Split, 0},   // input[tile_n, Ci, Hi, Wi]
        {1, SliceKind::Shared, 0},  // filter full
    };
    s.outputRules = {{0, SliceKind::Split, 0}};
    s.reuseRatio = 3.0; // filter fully shared
    spec.splitDims.push_back(s);
  }

  // --- Split Co (output channels, iter dim 1): parallel, filter splits ---
  {
    SplitDimSpec s;
    s.outputDim = 1; // Co in output
    s.iterDim = 1;
    s.isParallel = true;
    s.inputRules = {
        {0, SliceKind::Shared, 0},  // input full
        {1, SliceKind::Split, 0},   // filter[tile_co, Ci, Kh, Kw]
    };
    s.outputRules = {{0, SliceKind::Split, 1}};
    s.reuseRatio = 2.0; // input fully shared
    spec.splitDims.push_back(s);
  }

  // --- Split Ho (output height, iter dim 2): parallel, input needs HALO ---
  {
    SplitDimSpec s;
    s.outputDim = 2; // Ho in output
    s.iterDim = 2;
    s.isParallel = true;
    s.inputRules = {
        {0, SliceKind::Halo, 2, /*haloLow=*/0, /*haloHigh=*/Kh - 1},
        // input H slice: [oh*stride : oh*stride + tile_h*stride + Kh - 1]
        // haloHigh = Kh-1 extra rows beyond the tile boundary
        {1, SliceKind::Shared, 0},  // filter full
    };
    s.outputRules = {{0, SliceKind::Split, 2}};
    s.reuseRatio = 1.5;
    spec.splitDims.push_back(s);
  }

  // --- Split Wo (output width, iter dim 3): parallel, input needs HALO ---
  {
    SplitDimSpec s;
    s.outputDim = 3; // Wo in output
    s.iterDim = 3;
    s.isParallel = true;
    s.inputRules = {
        {0, SliceKind::Halo, 3, 0, Kw - 1},
        {1, SliceKind::Shared, 0},
    };
    s.outputRules = {{0, SliceKind::Split, 3}};
    s.reuseRatio = 1.5;
    spec.splitDims.push_back(s);
  }

  // --- Split Ci (input channels, iter dim 4): REDUCTION ---
  {
    SplitDimSpec s;
    s.outputDim = UINT_MAX; // Ci is not an output dimension
    s.iterDim = 4;
    s.isParallel = false; // REDUCTION
    s.inputRules = {
        {0, SliceKind::Split, 1},   // input[N, tile_ci, Hi, Wi]
        {1, SliceKind::Split, 1},   // filter[Co, tile_ci, Kh, Kw]
    };
    s.outputRules = {{0, SliceKind::Reduce, 0}};
    s.reuseRatio = 1.0;
    spec.splitDims.push_back(s);
  }

  // Spatial: prefer splitting Co (each core handles subset of output channels)
  spec.preferredSpatialDim = 1;
  // Temporal: tile Ho, Wo, then Co
  spec.preferredTemporalDims = {2, 3, 1};

  return spec;
}

//===----------------------------------------------------------------------===//
// Elementwise (linalg.generic with all parallel iterators)
//===----------------------------------------------------------------------===//

OperatorTilingSpec npu::getElementwiseTilingSpec(unsigned rank) {
  OperatorTilingSpec spec;
  spec.opName = "linalg.generic (elementwise)";

  for (unsigned dim = 0; dim < rank; ++dim) {
    SplitDimSpec s;
    s.outputDim = dim;
    s.iterDim = dim;
    s.isParallel = true;
    s.inputRules = {
        {0, SliceKind::Split, dim},
    };
    s.outputRules = {
        {0, SliceKind::Split, dim},
    };
    s.reuseRatio = 1.0;
    spec.splitDims.push_back(s);
  }

  spec.preferredSpatialDim = 0;
  spec.preferredTemporalDims = {0};

  return spec;
}

//===----------------------------------------------------------------------===//
// Generic op with mixed parallel/reduction iterators
// (e.g., softmax, layernorm, reduce_sum)
//===----------------------------------------------------------------------===//

OperatorTilingSpec npu::getGenericTilingSpec(
    llvm::ArrayRef<mlir::utils::IteratorType> iterTypes,
    unsigned outputRank) {
  OperatorTilingSpec spec;

  // Classify the generic op by its iterator pattern
  unsigned numParallel = 0, numReduction = 0;
  for (auto t : iterTypes) {
    if (t == utils::IteratorType::parallel)
      numParallel++;
    else
      numReduction++;
  }

  if (numReduction == 0) {
    spec.opName = "linalg.generic (elementwise)";
  } else if (numParallel > 0 && numReduction > 0) {
    spec.opName = "linalg.generic (mixed par/red)";
  } else {
    spec.opName = "linalg.generic (full reduction)";
  }

  // Build split dim specs from iterator types
  unsigned outDimIdx = 0;
  for (unsigned dim = 0; dim < iterTypes.size(); ++dim) {
    SplitDimSpec s;
    s.iterDim = dim;
    s.isParallel = (iterTypes[dim] == utils::IteratorType::parallel);

    if (s.isParallel && outDimIdx < outputRank) {
      s.outputDim = outDimIdx;
      outDimIdx++;
      // Parallel dim: all operands split
      s.inputRules = {{0, SliceKind::Split, s.outputDim}};
      s.outputRules = {{0, SliceKind::Split, s.outputDim}};
      s.reuseRatio = 1.0;
    } else {
      // Reduction dim: output doesn't have this dim, inputs are split,
      // output needs Reduce (partial sums accumulated)
      s.outputDim = UINT_MAX; // not an output dimension
      s.inputRules = {{0, SliceKind::Split, dim < outputRank ? dim : 0}};
      s.outputRules = {{0, SliceKind::Reduce, 0}};
      s.reuseRatio = 1.0;
    }

    spec.splitDims.push_back(s);
  }

  // Preferred dims: only parallel dims are safe for spatial tiling
  // For temporal tiling: parallel dims first, then reduction dims
  for (unsigned dim = 0; dim < iterTypes.size(); ++dim) {
    if (iterTypes[dim] == utils::IteratorType::parallel) {
      if (spec.preferredSpatialDim == -1)
        spec.preferredSpatialDim = dim;
      spec.preferredTemporalDims.push_back(dim);
    }
  }
  // Reduction dims go last in temporal search
  for (unsigned dim = 0; dim < iterTypes.size(); ++dim) {
    if (iterTypes[dim] == utils::IteratorType::reduction)
      spec.preferredTemporalDims.push_back(dim);
  }

  return spec;
}

//===----------------------------------------------------------------------===//
// linalg.transpose
// Tiling output dim i means tiling input dim perm[i].
// All dimensions are parallel (permutation is a bijection).
//===----------------------------------------------------------------------===//

OperatorTilingSpec npu::getTransposeTilingSpec(
    llvm::ArrayRef<int64_t> permutation, unsigned rank) {
  OperatorTilingSpec spec;
  spec.opName = "linalg.transpose";

  for (unsigned dim = 0; dim < rank; ++dim) {
    SplitDimSpec s;
    s.outputDim = dim;
    s.iterDim = dim;
    s.isParallel = true;

    // Tiling output dim `dim` means tiling input dim `perm[dim]`.
    unsigned inputDim = static_cast<unsigned>(permutation[dim]);
    s.inputRules = {
        {/*operandIdx=*/0, SliceKind::Split, /*sliceDim=*/inputDim},
    };
    s.outputRules = {
        {/*operandIdx=*/0, SliceKind::Split, /*sliceDim=*/dim},
    };
    // Transpose has no data reuse — each element is read and written exactly once.
    s.reuseRatio = 1.0;
    spec.splitDims.push_back(s);
  }

  // Spatial tiling: prefer the largest output dimension.
  // We don't know the actual dimension sizes here, so we default to dim 0.
  // The caller (SpatialTilingPass) will evaluate all parallel dims via cost model
  // and pick the largest.
  spec.preferredSpatialDim = 0;

  // Temporal tiling: transpose should ideally be at fusion boundaries,
  // so we avoid splitting. Set empty preferred temporal dims to signal
  // "don't tile temporally".
  spec.preferredTemporalDims = {};

  return spec;
}

//===----------------------------------------------------------------------===//
// Reshape boundary check
//===----------------------------------------------------------------------===//

bool npu::isReshapeBoundaryOp(Operation *op) {
  return isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(op);
}

//===----------------------------------------------------------------------===//
// Generic fallback: build from indexing_maps
//===----------------------------------------------------------------------===//

OperatorTilingSpec npu::buildSpecFromIndexingMaps(Operation *op) {
  OperatorTilingSpec spec;
  spec.opName = op->getName().getStringRef();

  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    return spec;
  }

  auto iterTypes = linalgOp.getIteratorTypesArray();
  auto indexingMaps = linalgOp.getIndexingMapsArray();

  if (indexingMaps.empty() || linalgOp.getNumDpsInits() == 0)
    return spec;

  // The last indexing map is for the output (init/outs)
  AffineMap outputMap = indexingMaps.back();

  for (unsigned iterDim = 0; iterDim < iterTypes.size(); ++iterDim) {
    SplitDimSpec s;
    s.iterDim = iterDim;
    s.isParallel = (iterTypes[iterDim] == utils::IteratorType::parallel);

    // Find which output dimension this iter dim maps to
    s.outputDim = UINT_MAX;
    for (unsigned outDim = 0; outDim < outputMap.getNumResults(); ++outDim) {
      auto expr = outputMap.getResult(outDim);
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        if (dimExpr.getPosition() == iterDim) {
          s.outputDim = outDim;
          break;
        }
      }
    }

    // For each input operand, determine how it's affected
    unsigned numInputs = linalgOp.getNumDpsInputs();
    for (unsigned opIdx = 0; opIdx < numInputs; ++opIdx) {
      if (opIdx >= indexingMaps.size())
        break;
      AffineMap inputMap = indexingMaps[opIdx];

      // Check if this iter dim appears in the input's indexing map
      bool found = false;
      for (unsigned inDim = 0; inDim < inputMap.getNumResults(); ++inDim) {
        auto expr = inputMap.getResult(inDim);
        // Simple case: direct dimension reference
        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
          if (dimExpr.getPosition() == iterDim) {
            s.inputRules.push_back({opIdx, SliceKind::Split, inDim});
            found = true;
            break;
          }
        }
        // Affine expression with offset (e.g., d2 + d5 for conv halo)
        // This indicates a Halo pattern
        if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
          if (binExpr.getKind() == AffineExprKind::Add) {
            auto lhs = dyn_cast<AffineDimExpr>(binExpr.getLHS());
            if (lhs && lhs.getPosition() == iterDim) {
              // d_iter + d_kernel → Halo pattern
              // haloHigh = range of the kernel dimension
              s.inputRules.push_back(
                  {opIdx, SliceKind::Halo, inDim, 0, /*haloHigh=*/0});
              found = true;
              break;
            }
          }
        }
      }
      if (!found) {
        // This iter dim doesn't appear in this input → input is Shared
        s.inputRules.push_back({opIdx, SliceKind::Shared, 0});
      }
    }

    // Output rule
    if (s.outputDim != UINT_MAX) {
      s.outputRules.push_back(
          {0, s.isParallel ? SliceKind::Split : SliceKind::Reduce, s.outputDim});
    }

    s.reuseRatio = 1.0;
    // Count shared inputs for reuse estimate
    for (auto &rule : s.inputRules) {
      if (rule.kind == SliceKind::Shared)
        s.reuseRatio += 0.5;
    }

    spec.splitDims.push_back(s);
  }

  return spec;
}

//===----------------------------------------------------------------------===//
// Registry lookup
//===----------------------------------------------------------------------===//

const OperatorTilingSpec *npu::getTilingSpec(Operation *op) {
  // Static specs for known ops (cached)
  static OperatorTilingSpec matmulSpec = getMatmulTilingSpec();
  static OperatorTilingSpec batchMatmulSpec = getBatchMatmulTilingSpec();

  if (isa<linalg::MatmulOp>(op))
    return &matmulSpec;

  if (isa<linalg::BatchMatmulOp>(op))
    return &batchMatmulSpec;

  // Conv2d needs runtime kernel sizes, so build on the fly
  // (could cache by {Kh, Kw, strideH, strideW} key)
  if (isa<linalg::Conv2DNchwFchwOp>(op)) {
    auto filterType = cast<ShapedType>(op->getOperand(1).getType());
    int64_t Kh = filterType.getDimSize(2);
    int64_t Kw = filterType.getDimSize(3);
    // TODO: extract strides from op attributes
    static OperatorTilingSpec convSpec = getConv2dNchwTilingSpec(Kh, Kw, 1, 1);
    return &convSpec;
  }

  // Transpose: build spec from permutation attribute
  if (auto transposeOp = dyn_cast<linalg::TransposeOp>(op)) {
    auto perm = transposeOp.getPermutation();
    unsigned rank = perm.size();
    // Cache by (rank, permutation hash)
    thread_local llvm::DenseMap<uint64_t, OperatorTilingSpec> transposeSpecCache;
    uint64_t key = rank;
    for (unsigned i = 0; i < perm.size() && i < 8; ++i)
      key = key * 31 + static_cast<uint64_t>(perm[i]);
    auto it = transposeSpecCache.find(key);
    if (it == transposeSpecCache.end()) {
      auto spec = getTransposeTilingSpec(perm, rank);
      it = transposeSpecCache.try_emplace(key, std::move(spec)).first;
    }
    return &it->second;
  }

  // Generic: build spec from iterator_types
  // - Pure elementwise (all parallel): relu, sigmoid, add, etc.
  // - Mixed parallel+reduction: softmax, layernorm, reduce_sum, etc.
  // - The spec correctly marks reduction dims as non-splittable for spatial tiling
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    if (genericOp.getNumResults() > 0) {
      auto iterTypes = genericOp.getIteratorTypesArray();
      auto outType = cast<ShapedType>(genericOp.getResult(0).getType());
      // Cache by (numIters, numParallel) signature — good enough for most cases
      // Use thread_local to avoid static init issues
      thread_local llvm::DenseMap<uint64_t, OperatorTilingSpec> genericSpecCache;
      uint64_t key = (uint64_t(iterTypes.size()) << 32) | outType.getRank();
      // Also encode parallel/reduction pattern
      for (unsigned i = 0; i < iterTypes.size() && i < 8; ++i) {
        if (iterTypes[i] == utils::IteratorType::reduction)
          key |= (1ULL << (16 + i));
      }
      auto it = genericSpecCache.find(key);
      if (it == genericSpecCache.end()) {
        auto spec = getGenericTilingSpec(iterTypes, outType.getRank());
        it = genericSpecCache.try_emplace(key, std::move(spec)).first;
      }
      return &it->second;
    }
  }

  // Fallback: try building from indexing maps
  return nullptr;
}
