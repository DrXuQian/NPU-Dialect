//===- OperatorTilingSpec.h - Per-operator tiling specifications -*- C++ -*-===//
//
// Each operator type defines:
//   1. Which dimensions are splittable (parallel vs reduction)
//   2. For each split dimension, how inputs/outputs are sliced
//   3. Whether output needs reduction (e.g., split on K for matmul)
//   4. Data reuse pattern (which inputs are shared vs split)
//
// This is the "operator registry" that cost model and tiling passes query.
//
// Examples:
//   matmul C[M,N] = A[M,K] * B[K,N]:
//     split M → A[tile_m, K], B[K, N] (B shared), C[tile_m, N]
//     split N → A[M, K] (A shared), B[K, tile_n], C[M, tile_n]
//     split K → A[M, tile_k], B[tile_k, N], C[M, N] (needs reduce!)
//
//   conv2d out[N,Co,Ho,Wo] = input[N,Ci,Hi,Wi] * filter[Co,Ci,Kh,Kw]:
//     split N  → input[tile_n,...], filter(shared), out[tile_n,...]
//     split Co → input(shared), filter[tile_co,...], out[..tile_co..]
//     split Ho → input[..,oh*s:oh*s+tile_h+kh-1,..] (HALO), filter(shared)
//     split Ci → needs reduce across partial sums!
//
//===----------------------------------------------------------------------===//

#ifndef NPU_ANALYSIS_OPERATORTILINGSPEC_H
#define NPU_ANALYSIS_OPERATORTILINGSPEC_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace npu {

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

/// How an operand is affected when the output is split along a dimension.
enum class SliceKind {
  Split,     // Operand is sliced on a corresponding dimension
  Shared,    // Operand is NOT sliced (broadcast / fully replicated)
  Halo,      // Operand is sliced with extra overlap (conv kernel window)
  Reduce,    // Split produces partial results that need accumulation
};

//===----------------------------------------------------------------------===//
// OperandSliceRule: how one operand maps to a split dimension
//===----------------------------------------------------------------------===//

struct OperandSliceRule {
  unsigned operandIdx;       // which operand (0-based)
  SliceKind kind;
  unsigned sliceDim;         // which dim of the operand to slice (for Split/Halo)
  int64_t haloLow = 0;      // extra elements before tile start (for Halo)
  int64_t haloHigh = 0;     // extra elements after tile end (for Halo)
};

//===----------------------------------------------------------------------===//
// SplitDimSpec: complete splitting spec for one output dimension
//===----------------------------------------------------------------------===//

struct SplitDimSpec {
  unsigned outputDim;        // which output dimension we're splitting
  unsigned iterDim;          // corresponding iteration domain dimension
  bool isParallel;           // true = parallel, false = reduction (needs reduce)

  /// How each input operand is affected by this split.
  llvm::SmallVector<OperandSliceRule> inputRules;

  /// How each output operand is affected (usually just Split on outputDim).
  llvm::SmallVector<OperandSliceRule> outputRules;

  /// Estimated data reuse ratio: higher = more reuse = less DMA per compute.
  /// For matmul split-M: B is shared across tiles → reuse = K*N bytes shared.
  double reuseRatio = 1.0;
};

//===----------------------------------------------------------------------===//
// OperatorTilingSpec: complete tiling specification for one op type
//===----------------------------------------------------------------------===//

struct OperatorTilingSpec {
  llvm::StringRef opName;    // e.g., "linalg.matmul"

  /// All possible split dimensions with their rules.
  llvm::SmallVector<SplitDimSpec> splitDims;

  /// Preferred split dimension for spatial tiling (inter-core).
  /// -1 means "let cost model decide".
  int preferredSpatialDim = -1;

  /// Preferred split dimensions for temporal tiling (intra-core).
  /// Order matters: first = outermost loop.
  llvm::SmallVector<unsigned> preferredTemporalDims;
};

//===----------------------------------------------------------------------===//
// Registry: get spec for a given operation
//===----------------------------------------------------------------------===//

/// Get the tiling spec for a linalg op. Returns nullptr if unknown.
const OperatorTilingSpec *getTilingSpec(mlir::Operation *op);

/// Build specs from linalg op's indexing_maps (generic fallback).
OperatorTilingSpec buildSpecFromIndexingMaps(mlir::Operation *op);

//===----------------------------------------------------------------------===//
// Built-in specs for common operators
//===----------------------------------------------------------------------===//

/// matmul C[M,N] = A[M,K] * B[K,N]
OperatorTilingSpec getMatmulTilingSpec();

/// conv2d_nchw out[N,Co,Ho,Wo] = in[N,Ci,Hi,Wi] * filt[Co,Ci,Kh,Kw]
OperatorTilingSpec getConv2dNchwTilingSpec(int64_t Kh, int64_t Kw,
                                            int64_t strideH, int64_t strideW);

/// elementwise (generic with all parallel iterators)
OperatorTilingSpec getElementwiseTilingSpec(unsigned rank);

/// generic op with mixed parallel/reduction iterators (softmax, layernorm, etc.)
OperatorTilingSpec getGenericTilingSpec(
    llvm::ArrayRef<mlir::utils::IteratorType> iterTypes,
    unsigned outputRank);

} // namespace npu

#endif // NPU_ANALYSIS_OPERATORTILINGSPEC_H
