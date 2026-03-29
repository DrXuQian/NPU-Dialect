// RUN: %npu-opt %s --npu-spatial-tiling | %mlir-opt --canonicalize --cse | FileCheck %s

// Test: after spatial tiling + canonicalize + cse, all tensor shapes
// should be fully static (no dynamic dimensions marked with '?').

func.func @static_matmul(%A: tensor<128x256xf16>, %B: tensor<256x64xf16>) -> tensor<128x64xf16> {
  %cst = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<128x64xf16>
  %fill = linalg.fill ins(%cst : f16) outs(%init : tensor<128x64xf16>) -> tensor<128x64xf16>
  %result = linalg.matmul ins(%A, %B : tensor<128x256xf16>, tensor<256x64xf16>)
                          outs(%fill : tensor<128x64xf16>) -> tensor<128x64xf16>
  return %result : tensor<128x64xf16>
}

// After canonicalize+cse, all tensor dimensions should be static constants.
// No dynamic tensor types (tensor<?x...> or tensor<...x?x...>) should remain.
// CHECK-LABEL: func @static_matmul
// CHECK-NOT: tensor<{{[^>]*}}?{{[^>]*}}>
// CHECK: return
