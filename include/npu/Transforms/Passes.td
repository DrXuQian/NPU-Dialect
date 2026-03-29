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
    Option<"numCores", "num-cores", "int64_t", "6",
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
    Option<"sramSize", "sram-size", "int64_t", "2097152",
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
// Pass 5: SRAM Allocation (address planning + spill-via-DMA)
//===----------------------------------------------------------------------===//

def NPUSRAMAllocation : Pass<"npu-sram-allocation", "mlir::func::FuncOp"> {
  let summary = "Plan SRAM addresses and insert DMA for spill/reload";
  let description = [{
    Unified SRAM address planning for each kernel function.

    Invariant: SRAM is empty at kernel entry and exit.

    Address layout (dual-end):
      [0 ── inputs → ]  [ ← intermediates → ]  [ ← outputs ── SRAM_SIZE]
    Inputs grow from address 0 upward, outputs from SRAM_SIZE downward.
    Intermediates fill the gap in the middle.

    When total working set exceeds SRAM:
      - Evict the largest dead-at-this-point tensor to DDR:
        insert `npu.dma_copy` (S2D) after last use
      - Reload before next use:
        insert `npu.dma_copy` (D2S) before consumer
      - A DDR shadow buffer (`memref.alloc`) is created for each evicted tensor

    Each outlined `npu.fused_kernel` function is planned independently,
    so SRAM allocation can be parallelized across kernels.
  }];
  let options = [
    Option<"sramSize", "sram-size", "int64_t", "2097152",
           "SRAM size per core in bytes">,
  ];
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "npu::NPUDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Pass 5b: DMA Scheduling (lower sync DMA to async ping/pong + prefetch)
//===----------------------------------------------------------------------===//

def NPUBarrierAlloc : Pass<"npu-barrier-alloc", "mlir::func::FuncOp"> {
  let summary = "Allocate hardware barrier resources to DMA/compute subgraphs";
  let description = [{
    Assigns barrier_group IDs to ops based on cost model timeline analysis.

    Design:
    1. Build timeline: group ops into subgraphs (DMA_in → compute → DMA_out)
    2. Allocate barrier IDs sequentially: subgraph 0 → barrier 0,
       subgraph 1 → barrier 1, ...
    3. When barrier pool exhausted → CUT (start new generation)
    4. Double buffer: generation n+2 reuses generation n's barriers
       (n+2 cannot start until n is fully complete)

    No explicit wait/signal ops — barrier is an attribute on each op,
    hardware manages the synchronization:

      npu.dma_copy ... {barrier_group = 0 : i32}
      linalg.matmul ... {barrier_group = 0 : i32}
      npu.dma_copy ... {barrier_group = 0 : i32}
      // subgraph 1
      npu.dma_copy ... {barrier_group = 1 : i32}
      ...
      // CUT — pool exhausted, new generation
      npu.dma_copy ... {barrier_group = 0 : i32, generation = 1 : i32}
      // generation 1, barrier 0 → must wait for generation 0, barrier 0 to finish

    The op ordering in IR IS the execution schedule.
    Within a barrier_group, the hardware enforces DMA→compute→DMA ordering.
    Between barrier_groups, ops on different engines can overlap.
  }];
  let options = [
    Option<"barrierPoolSize", "barrier-pool-size", "int64_t", "4",
           "Number of hardware barrier resources">,
  ];
  let dependentDialects = ["npu::NPUDialect"];
}

//===----------------------------------------------------------------------===//
// Pass 2b: Core Mapping (make spatial core assignment explicit)
//===----------------------------------------------------------------------===//

def NPUCoreMapping : Pass<"npu-core-mapping", "mlir::func::FuncOp"> {
  let summary = "Map spatial tile iterations to NPU cores";
  let description = [{
    Wraps the body of spatial tiling loops (outermost scf.for) with
    npu.core_execute regions, assigning each tile iteration to a core.

    For scf.for loops generated by spatial tiling, the induction variable
    divided by the step gives the core ID. This pass makes the core
    assignment explicit in the IR.
  }];
  let dependentDialects = ["npu::NPUDialect", "mlir::scf::SCFDialect"];
}

//===----------------------------------------------------------------------===//
// Pass 6 (analysis): Cost evaluation of lowered IR
//===----------------------------------------------------------------------===//

def NPUCostEvaluate : Pass<"npu-cost-evaluate", "mlir::func::FuncOp"> {
  let summary = "Evaluate execution cost of IR using roofline model";
  let description = [{
    Walks the function body and computes cost metrics:
    - Per-op: FLOPs, memory bytes, arithmetic intensity, roofline time
    - Per-kernel: 3-engine pipeline cost with DMA/matrix/DSP overlap
    - Overall: total time, peak SRAM, bottleneck analysis

    Emits cost breakdown as JSON to stderr (parseable by harness).
    Does NOT modify the IR.
  }];
  let dependentDialects = [
    "npu::NPUDialect",
    "mlir::scf::SCFDialect",
  ];
}

#endif // NPU_TRANSFORMS_PASSES_TD
