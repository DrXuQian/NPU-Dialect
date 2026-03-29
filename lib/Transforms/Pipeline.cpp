//===- Pipeline.cpp - Register the npu-pipeline ---------------------------===//

#include "npu/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

void npu::registerNPUPipeline() {
  PassPipelineRegistration<>(
      "npu-pipeline",
      "Full NPU compilation pipeline (all static shapes)",
      [](OpPassManager &pm) {
        // Pass 1: Cost-model driven greedy fusion
        pm.addNestedPass<func::FuncOp>(npu::createNPUFusion());
        // Pass 1b: Outline fused subgraphs into kernel functions
        pm.addPass(npu::createNPUOutlineFusedGroups());
        // Pass 2: Spatial tiling (with loop peeling for static shapes)
        pm.addNestedPass<func::FuncOp>(npu::createNPUSpatialTiling());
        // Pass 2b: Core mapping
        pm.addNestedPass<func::FuncOp>(npu::createNPUCoreMapping());
        // Pass 3: Temporal tiling (spec-driven, with loop peeling)
        pm.addNestedPass<func::FuncOp>(npu::createNPUTemporalTiling());
        // Fold affine constants from peeling → all shapes static
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
        // Pass 4: Bufferize (tensor → memref + npu.alloc_sram + npu.dma_copy)
        pm.addPass(npu::createNPUBufferize());
        // Pass 5: SRAM allocation (dual-end, spill via DMA, double buffer, prefetch)
        pm.addNestedPass<func::FuncOp>(npu::createNPUSRAMAllocation());
        // Pass 5b: Allocate hardware barrier resources
        pm.addNestedPass<func::FuncOp>(npu::createNPUBarrierAlloc());
        // Pass 6: Cost evaluation (roofline, does not modify IR)
        pm.addNestedPass<func::FuncOp>(npu::createNPUCostEvaluate());
      });
}
