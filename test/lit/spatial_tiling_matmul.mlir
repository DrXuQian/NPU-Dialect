// RUN: %npu-opt %s --npu-spatial-tiling | FileCheck %s

// Test: spatial tiling of matmul produces an scf.for loop with
// tensor.extract_slice / tensor.insert_slice around the matmul.

func.func @spatial_matmul(%A: tensor<128x256xf16>, %B: tensor<256x64xf16>) -> tensor<128x64xf16> {
  %cst = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<128x64xf16>
  %fill = linalg.fill ins(%cst : f16) outs(%init : tensor<128x64xf16>) -> tensor<128x64xf16>
  %result = linalg.matmul ins(%A, %B : tensor<128x256xf16>, tensor<256x64xf16>)
                          outs(%fill : tensor<128x64xf16>) -> tensor<128x64xf16>
  return %result : tensor<128x64xf16>
}

// CHECK-LABEL: func @spatial_matmul
// CHECK:       scf.for
// CHECK:         tensor.extract_slice
// CHECK:         linalg.matmul
// CHECK:         tensor.insert_slice
// CHECK:       scf.yield
