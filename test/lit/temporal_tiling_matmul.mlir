// RUN: %npu-opt %s --npu-temporal-tiling | FileCheck %s

// Test: temporal tiling of matmul produces tiled loops that fit
// the working set into SRAM. The matmul should appear inside
// one or more scf.for loops with tensor.extract_slice.

func.func @temporal_matmul(%A: tensor<256x512xf16>, %B: tensor<512x256xf16>) -> tensor<256x256xf16> {
  %cst = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<256x256xf16>
  %fill = linalg.fill ins(%cst : f16) outs(%init : tensor<256x256xf16>) -> tensor<256x256xf16>
  %result = linalg.matmul ins(%A, %B : tensor<256x512xf16>, tensor<512x256xf16>)
                          outs(%fill : tensor<256x256xf16>) -> tensor<256x256xf16>
  return %result : tensor<256x256xf16>
}

// Temporal tiling should create at least one scf.for with extract_slice/insert_slice.
// CHECK-LABEL: func @temporal_matmul
// CHECK:       scf.for
// CHECK:         tensor.extract_slice
// CHECK:         linalg.matmul
// CHECK:         tensor.insert_slice
