//===- Pipeline.cpp - Register the npu-pipeline ---------------------------===//

#include "npu/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

void npu::registerNPUPipeline() {
  PassPipelineRegistration<>(
      "npu-pipeline",
      "Full NPU compilation pipeline",
      [](OpPassManager &pm) {
        // Pass 1: Cost-model driven fusion (on FuncOp)
        pm.addNestedPass<func::FuncOp>(npu::createNPUFusion());
        // Pass 1b: Outline fused subgraphs into kernel functions (ModuleOp)
        pm.addPass(npu::createNPUOutlineFusedGroups());
        // Pass 2: Spatial tiling — distribute across cores (FuncOp)
        pm.addNestedPass<func::FuncOp>(npu::createNPUSpatialTiling());
        // Pass 3: Temporal tiling — fit SRAM with double buffering (FuncOp)
        pm.addNestedPass<func::FuncOp>(npu::createNPUTemporalTiling());
        // Pass 4: Bufferize — tensor → memref + npu.alloc_sram + npu.dma_copy
        pm.addPass(npu::createNPUBufferize());
        // Pass 5: SRAM allocation — address planning + spill via DMA (FuncOp)
        //   Each kernel assumes SRAM empty at entry/exit.
        //   Inputs from low end, outputs from high end.
        //   Overflow → evict to DDR + DMA reload.
        pm.addNestedPass<func::FuncOp>(npu::createNPUSRAMAllocation());
      });
}
