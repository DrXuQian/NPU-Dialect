//===- npu-opt.cpp - NPU MLIR optimizer driver ----------------------------===//
//
// Main entry point for the NPU compiler. Registers all NPU-specific passes
// and the NPU dialect, then delegates to the MLIR opt infrastructure.
//
// Usage:
//   npu-opt input.mlir --npu-fusion --npu-spatial-tiling --npu-temporal-tiling
//   npu-opt input.mlir --npu-pipeline  (runs all passes in sequence)
//
//===----------------------------------------------------------------------===//

#include "npu/Dialect/NPU/NPUDialect.h"
#include "npu/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register standard dialects
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::tensor::TensorDialect>();

  // Register external interface implementations (TilingInterface for linalg)
  mlir::linalg::registerTilingInterfaceExternalModels(registry);

  // Register BufferizableOpInterface external models for all dialects
  // that participate in bufferization.
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);

  // Register NPU dialect
  registry.insert<npu::NPUDialect>();

  // Register NPU passes
  npu::registerNPUPasses();
  npu::registerNPUPipeline();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NPU MLIR Compiler\n", registry));
}
