// RUN: %npu-opt %s --npu-spatial-tiling | FileCheck %s

// Test: batch_matmul spatial tiling should tile along a parallel
// dimension and produce an scf.for loop with extract_slice.

func.func @batch_matmul(%A: tensor<4x128x256xf16>, %B: tensor<4x256x64xf16>) -> tensor<4x128x64xf16> {
  %cst = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<4x128x64xf16>
  %fill = linalg.fill ins(%cst : f16) outs(%init : tensor<4x128x64xf16>) -> tensor<4x128x64xf16>
  %result = linalg.batch_matmul ins(%A, %B : tensor<4x128x256xf16>, tensor<4x256x64xf16>)
                                outs(%fill : tensor<4x128x64xf16>) -> tensor<4x128x64xf16>
  return %result : tensor<4x128x64xf16>
}

// Batch matmul should be recognized and tiled spatially.
// CHECK-LABEL: func @batch_matmul
// CHECK:       scf.for
// CHECK:         tensor.extract_slice
// CHECK:         linalg.batch_matmul
// CHECK:         tensor.insert_slice
