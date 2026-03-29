//===- Pipeline.cpp - Register the npu-pipeline ---------------------------===//

#include "npu/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

void npu::registerNPUPipeline() {
  PassPipelineRegistration<>(
      "npu-pipeline",
      "Full NPU compilation pipeline: fusion -> spatial -> temporal -> bufferize -> spill",
      [](OpPassManager &pm) {
        // Passes 1-3: operate on FuncOp
        pm.addNestedPass<func::FuncOp>(npu::createNPUFusion());
        // Pass 1b: outline fused subgraphs (operates on ModuleOp)
        pm.addPass(npu::createNPUOutlineFusedGroups());
        pm.addNestedPass<func::FuncOp>(npu::createNPUSpatialTiling());
        pm.addNestedPass<func::FuncOp>(npu::createNPUTemporalTiling());
        // Pass 4: bufferize operates on ModuleOp
        pm.addPass(npu::createNPUBufferize());
        // Pass 5: spill handling on FuncOp (post-bufferize)
        pm.addNestedPass<func::FuncOp>(npu::createNPUSpillHandling());
      });
}
