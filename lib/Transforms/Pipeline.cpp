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
      "Full NPU compilation pipeline",
      [](OpPassManager &pm) {
        // Pass 1: Cost-model driven fusion
        pm.addNestedPass<func::FuncOp>(npu::createNPUFusion());
        // Pass 1b: Outline fused subgraphs into kernel functions
        pm.addPass(npu::createNPUOutlineFusedGroups());
        // Pass 2: Spatial tiling (with loop peeling for static shapes)
        pm.addNestedPass<func::FuncOp>(npu::createNPUSpatialTiling());
        // Pass 2b: Core mapping
        pm.addNestedPass<func::FuncOp>(npu::createNPUCoreMapping());
        // Pass 3: Temporal tiling (with loop peeling)
        pm.addNestedPass<func::FuncOp>(npu::createNPUTemporalTiling());
        // Canonicalize + CSE: fold affine constants from peeling,
        // ensure all shapes are static before bufferize.
        pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
        pm.addNestedPass<func::FuncOp>(createCSEPass());
        // Pass 4: Bufferize (tensor → memref + npu.alloc_sram + npu.dma_copy)
        pm.addPass(npu::createNPUBufferize());
        // Pass 5: SRAM allocation
        pm.addNestedPass<func::FuncOp>(npu::createNPUSRAMAllocation());
        // Pass 6: Cost evaluation
        pm.addNestedPass<func::FuncOp>(npu::createNPUCostEvaluate());
      });
}
