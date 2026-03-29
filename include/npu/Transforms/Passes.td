//===- Passes.td - NPU transformation passes ----------------*- tablegen -*-===//
#ifndef NPU_TRANSFORMS_PASSES_TD
#define NPU_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Pass 1: Fusion (Linalg level)
//===----------------------------------------------------------------------===//

def NPUFusion : Pass<"npu-fusion", "mlir::func::FuncOp"> {
  let summary = "Cost-model driven vertical fusion of linalg ops";
  let description = [{
    Fuses producer-consumer linalg op pairs to eliminate intermediate DRAM
    round-trips. Uses CostModel.isFusionProfitable() to decide each fusion.

    Search space: for each producer->consumer edge with single consumer,
    compare fused vs unfused cost.
  }];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Pass 1b: Outline fused subgraphs into functions
//===----------------------------------------------------------------------===//

def NPUOutlineFusedGroups : Pass<"npu-outline-fused-groups", "mlir::ModuleOp"> {
  let summary = "Outline fusible op subgraphs into separate func.func";
  let description = [{
    Identifies connected subgraphs of fusible linalg ops (using the cost model),
    outlines each subgraph into its own `func.func`, and replaces the original
    ops with `func.call`.

    Each outlined function represents a "kernel" — a unit of work that the NPU
    executes with its three engines (DMA, matrix, DSP) in a pipelined fashion.
    The cost model evaluates the full subgraph pipeline cost, not just
    individual op costs.

    Example:
      Before:  %mm = linalg.matmul ...  /  %relu = linalg.generic(relu) ...
      After:   func.func @fused_matmul_generic(%A, %B, ...) { matmul; relu; }
               %result = func.call @fused_matmul_generic(...)
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Pass 2: Spatial Tiling (inter-core distribution)
//===----------------------------------------------------------------------===//

def NPUSpatialTiling : Pass<"npu-spatial-tiling", "mlir::func::FuncOp"> {
  let summary = "Distribute work across NPU cores via spatial tiling";
  let description = [{
    Partitions linalg ops across cores by tiling along a parallel dimension.
    Uses CostModel.evaluateSpatialSplit() to pick the best split dimension.

    Search space: for each parallel (non-reduction) dimension of the output,
    evaluate the per-core cost of splitting along that dimension.
  }];
  let options = [
    Option<"numCores", "num-cores", "int64_t", "4",
           "Number of NPU cores">,
  ];
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

//===----------------------------------------------------------------------===//
// Pass 3: Temporal Tiling (intra-core, fit SRAM)
//===----------------------------------------------------------------------===//

def NPUTemporalTiling : Pass<"npu-temporal-tiling", "mlir::func::FuncOp"> {
  let summary = "Tile per-core work to fit in SRAM with double buffering";
  let description = [{
    Further tiles each core's work so that the working set fits in SRAM.
    Uses CostModel.bestTileConfig() to search over tile-size candidates,
    optimizing for total cycles including DMA/compute overlap.

    Search space: for each dimension, halving candidates down to 16.
    Evaluation: num_iterations * max(DMA_in, compute) pipeline model.
  }];
  let options = [
    Option<"sramSize", "sram-size", "int64_t", "262144",
           "SRAM size per core in bytes">,
  ];
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

//===----------------------------------------------------------------------===//
// Pass 4: Buffer allocation + DMA insertion
//===----------------------------------------------------------------------===//

def NPUBufferize : Pass<"npu-bufferize", "mlir::ModuleOp"> {
  let summary = "Allocate SRAM buffers and insert explicit DMA ops";
  let description = [{
    Converts tensor operations to memref with explicit SRAM allocation
    (npu.alloc_sram) and DMA transfers (npu.dma_copy / npu.dma_start).
    Uses One-Shot Module Bufferize, then replaces tile-local allocs with
    npu.alloc_sram and inserts npu.dma_copy for DRAM<->SRAM transfers.
  }];
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::bufferization::BufferizationDialect",
    "npu::NPUDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Pass 5: Spill handling
//===----------------------------------------------------------------------===//

def NPUSpillHandling : Pass<"npu-spill-handling", "mlir::func::FuncOp"> {
  let summary = "Handle SRAM overflow via retiling or spilling to DRAM";
  let description = [{
    If any tile's working set exceeds SRAM after allocation:
    1. Re-tile with smaller tile sizes (CostModel.evaluateSpillStrategy)
    2. Spill intermediates to DRAM with explicit DMA
  }];
  let dependentDialects = ["npu::NPUDialect"];
}

#endif // NPU_TRANSFORMS_PASSES_TD
