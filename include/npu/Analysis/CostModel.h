//===- CostModel.h - Unified NPU cost model -------------------*- C++ -*-===//
//
// Single cost model shared by ALL compiler passes.
// Each pass defines its own search space; the cost model evaluates candidates.
//
// Integration patterns:
//   Query           (LLVM TTI style)  — computeCycles(), dmaCycles()
//   Profitability   (XLA style)       — isFusionProfitable()
//   Evaluate+Select (TVM/Halide)      — evaluateTileConfig(), evaluateSpatialSplit()
//
//===----------------------------------------------------------------------===//

#ifndef NPU_ANALYSIS_COSTMODEL_H
#define NPU_ANALYSIS_COSTMODEL_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <cstdint>
#include <vector>

namespace npu {

// Forward declarations from OperatorTilingSpec.h
struct SplitDimSpec;
struct OperatorTilingSpec;

//===----------------------------------------------------------------------===//
// Hardware target descriptor
//===----------------------------------------------------------------------===//

/// Inter-core interconnect type.
enum class InterconnectKind {
  NoC,        // Direct tile-to-tile network (low latency, on-chip)
  SharedL1,   // Shared L1 cache (hardware coherence, ≈free)
  SharedSRAM, // Shared addressable SRAM across cores
  DRAMOnly,   // No fast inter-core path; must go through DRAM
};

struct HWTarget {
  int64_t numCores = 4;
  int64_t sramPerCore = 256 * 1024; // bytes (core-private)
  int64_t dramBandwidth = 32;       // GB/s
  int64_t sramBandwidth = 256;      // GB/s
  int64_t matrixThroughput = 4096;  // MACs/cycle
  int64_t dspThroughput = 256;      // ops/cycle
  int64_t dmaChannels = 2;
  int64_t interCoreSyncCost = 100;  // cycles
  double clockGHz = 1.0;

  // Inter-core interconnect
  InterconnectKind interconnect = InterconnectKind::NoC;
  int64_t nocBandwidth = 128;       // GB/s (for NoC)
  int64_t nocLatencyCycles = 10;    // per-hop latency
  int64_t sharedL1Size = 0;         // bytes (for SharedL1/SharedSRAM)

  double bytesPerCycle() const { return dramBandwidth / clockGHz; }
  double nocBytesPerCycle() const { return nocBandwidth / clockGHz; }

  /// Cost of inter-core data transfer (spatial tiling boundary).
  /// Returns cycles to move `bytes` between two cores.
  int64_t interCoreTransferCycles(int64_t bytes) const {
    switch (interconnect) {
    case InterconnectKind::NoC:
      return nocLatencyCycles +
             std::max<int64_t>(1, static_cast<int64_t>(bytes / nocBytesPerCycle()));
    case InterconnectKind::SharedL1:
    case InterconnectKind::SharedSRAM:
      return 1; // essentially free (hardware-managed)
    case InterconnectKind::DRAMOnly:
      // Must go through DRAM: write + read
      return 2 * std::max<int64_t>(1, static_cast<int64_t>(bytes / bytesPerCycle()));
    }
    return 0;
  }
};

//===----------------------------------------------------------------------===//
// Cost structures
//===----------------------------------------------------------------------===//

struct ScheduleCost {
  int64_t computeCycles = 0;
  int64_t dmaCycles = 0;
  int64_t totalCycles = 0;   // after overlap (double-buffering) modeling
  int64_t sramPeakBytes = 0;
  int64_t numTiles = 0;
  int64_t totalDMABytes = 0;

  bool isValid() const { return totalCycles < INT64_MAX / 2; }
};

/// Roofline-based cost: the ground truth cost model.
/// For each op/subgraph, determines if it's compute-bound or memory-bound.
struct RooflineCost {
  double flops = 0;                // total floating-point operations
  double memBytes = 0;             // total DRAM bytes accessed (external only)
  double arithmeticIntensity = 0;  // flops / memBytes (FLOP/byte)
  double timeSec = 0;              // estimated wall time
  bool isMemoryBound = true;       // true if bottleneck is memory BW
  double peakAttainable = 0;       // attainable FLOP/s under roofline
  double efficiency = 0;           // actual / peak (0..1)
};

struct FusionDecision {
  bool shouldFuse = false;
  int64_t benefitCycles = 0; // positive = fusing is cheaper
};

struct TileConfig {
  llvm::SmallVector<int64_t> tileSizes; // per-dimension tile sizes
  ScheduleCost cost;
};

//===----------------------------------------------------------------------===//
// CostModel — the single shared instance across all passes
//===----------------------------------------------------------------------===//

class CostModel {
public:
  explicit CostModel(const HWTarget &target) : target_(target) {}

  const HWTarget &target() const { return target_; }

  // ================================================================
  // Level 0: Hardware primitive queries
  // ================================================================

  /// Exact MAC count for a linalg op given output/input shapes.
  int64_t opMACs(mlir::Operation *op) const;

  /// Compute cycles for executing one op.
  int64_t computeCycles(mlir::Operation *op) const;

  /// DMA transfer cycles for a given byte count.
  int64_t dmaCycles(int64_t bytes) const;

  /// Byte size of a shaped type (tensor or memref).
  static int64_t tensorBytes(mlir::ShapedType type);

  // ================================================================
  // Roofline model: ground truth cost evaluation
  // ================================================================

  /// Evaluate a single op using roofline model.
  /// Determines if it's compute-bound or memory-bound on the target hardware.
  RooflineCost evaluateRoofline(mlir::Operation *op) const;

  /// Evaluate a fused subgraph using roofline model.
  /// Intermediates stay in SRAM → reduced memBytes → higher arithmetic intensity.
  RooflineCost evaluateRoofline(llvm::ArrayRef<mlir::Operation *> ops) const;

  /// Evaluate roofline from raw FLOPs and memory bytes (for lowered IR).
  RooflineCost evaluateRooflineRaw(double flops, double memBytes,
                                    bool useMatrixUnit = true) const;

  // ================================================================
  // Level 1: Tile configuration evaluation
  // ================================================================

  /// Evaluate a matmul tiling: (tm, tn, tk) over full (M, N, K).
  ScheduleCost evaluateMatmulTile(int64_t tm, int64_t tn, int64_t tk,
                                  int64_t M, int64_t N, int64_t K,
                                  int64_t dtypeBytes,
                                  int64_t extraIntermediateBytes = 0) const;

  /// Evaluate a conv2d tiling.
  ScheduleCost evaluateConv2dTile(int64_t tn, int64_t tco, int64_t th,
                                  int64_t tw, int64_t N, int64_t Co,
                                  int64_t Ho, int64_t Wo, int64_t Ci,
                                  int64_t Kh, int64_t Kw,
                                  llvm::ArrayRef<int64_t> stride,
                                  int64_t dtypeBytes,
                                  int64_t extraOutputShaped = 0) const;

  // ================================================================
  // Level 2: Pass decision APIs
  // ================================================================

  /// Find the best temporal tile configuration for a linalg op.
  /// Used by: temporal tiling pass, and internally by fusion/spatial for
  /// downstream cost estimation.
  TileConfig bestTileConfig(mlir::Operation *op) const;

  /// Decide whether fusing producer into consumer is profitable.
  /// Compares: unfused_cost + intermediate_DMA  vs  fused_cost.
  FusionDecision isFusionProfitable(mlir::Operation *producer,
                                    mlir::Operation *consumer) const;

  /// Evaluate splitting a workload along a given dimension.
  /// Returns the worst-case single-core cost (determines wall time).
  ScheduleCost evaluateSpatialSplit(mlir::Operation *op, unsigned splitDim,
                                    int64_t numChunks) const;

  /// Evaluate spatial split using OperatorTilingSpec — precisely accounts for:
  ///   - Per-core DMA for Split operands (sliced data)
  ///   - Per-core DMA for Shared operands (full data, replicated to each core)
  ///   - Per-core DMA for Halo operands (sliced + overlap)
  ///   - Output: no concat needed (insert_slice handles it)
  ///   - Reduction: needs cross-core accumulation (penalty)
  struct SpatialSplitCost {
    int64_t perCoreDmaInBytes = 0;  // total bytes each core loads
    int64_t perCoreComputeCycles = 0;
    int64_t sharedDataBytes = 0;    // bytes replicated to every core
    int64_t splitDataBytes = 0;     // bytes unique to each core
    int64_t haloOverheadBytes = 0;  // extra bytes due to halo overlap
    bool needsReduce = false;       // true if splitting a reduction dim
    int64_t totalCycles = 0;        // overall estimate
  };
  SpatialSplitCost evaluateSpatialSplitDetailed(
      mlir::Operation *op, const struct SplitDimSpec &dimSpec,
      int64_t numChunks) const;

  /// Compare retile vs spill strategy.
  enum class SpillStrategy { Retile, Spill };
  SpillStrategy evaluateSpillStrategy(int64_t tileWorkingSet,
                                      int64_t intermediateBytes) const;

  // ================================================================
  // Level 3: Subgraph (fused kernel) pipeline evaluation
  // ================================================================

  /// Cost breakdown for a fused subgraph's three-engine pipeline.
  struct SubgraphCost {
    int64_t dmaInCycles = 0;     // DMA engine: load inputs from DRAM
    int64_t matrixCycles = 0;    // Matrix engine: matmul/conv compute
    int64_t dspCycles = 0;       // DSP engine: elementwise compute
    int64_t dmaOutCycles = 0;    // DMA engine: store outputs to DRAM
    int64_t intermediateDMA = 0; // DMA saved by keeping intermediates in SRAM
    int64_t pipelinedCycles = 0; // Total with 3-engine overlap
    int64_t unpipelinedCycles = 0; // Total without overlap (sequential)
    int64_t numOps = 0;
  };

  /// Evaluate the pipeline cost of a fused subgraph (list of ops).
  /// Models the NPU's three independent engines overlapping:
  ///   DMA_in → Matrix → DSP → DMA_out
  /// Intermediates between Matrix→DSP stay in SRAM (no DMA roundtrip).
  SubgraphCost evaluateSubgraph(llvm::ArrayRef<mlir::Operation *> ops) const;

  /// Evaluate whether a set of ops should be fused into one subgraph.
  /// Returns (should_fuse, benefit = unfused_cost - fused_cost).
  FusionDecision isSubgraphFusionProfitable(
      llvm::ArrayRef<mlir::Operation *> ops) const;

  // ================================================================
  // Post-hoc: cycle estimation for already-lowered IR
  // ================================================================

  /// Estimate cycles for a npu.dma_copy / npu.dma_start op.
  int64_t estimateDMAOp(mlir::Operation *op) const;

  /// Estimate cycles for a npu.compute region.
  int64_t estimateComputeOp(mlir::Operation *op) const;

private:
  HWTarget target_;

  /// Generate tile-size candidates by halving a dimension.
  static llvm::SmallVector<int64_t> tileCandidates(int64_t dim);
};

//===----------------------------------------------------------------------===//
// MLIR Analysis wrapper — so passes can request it via getAnalysis<>()
//===----------------------------------------------------------------------===//

class CostModelAnalysis {
public:
  CostModelAnalysis(mlir::Operation *op);

  const CostModel &getCostModel() const { return *costModel_; }

private:
  std::unique_ptr<CostModel> costModel_;
};

} // namespace npu

#endif // NPU_ANALYSIS_COSTMODEL_H
