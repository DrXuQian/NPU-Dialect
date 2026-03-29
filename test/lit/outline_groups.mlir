// RUN: %npu-opt %s --npu-fusion --npu-outline-fused-groups | FileCheck %s

// Test: fusion + outlining produces a private fused kernel function
// with the npu.fused_kernel attribute, and a func.call at the use site.

func.func @matmul_relu(%A: tensor<128x256xf16>, %B: tensor<256x512xf16>) -> tensor<128x512xf16> {
  %cst = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<128x512xf16>
  %fill = linalg.fill ins(%cst : f16) outs(%init : tensor<128x512xf16>) -> tensor<128x512xf16>
  %matmul = linalg.matmul ins(%A, %B : tensor<128x256xf16>, tensor<256x512xf16>)
                          outs(%fill : tensor<128x512xf16>) -> tensor<128x512xf16>

  %empty = tensor.empty() : tensor<128x512xf16>
  %relu = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%matmul : tensor<128x512xf16>)
    outs(%empty : tensor<128x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %zero = arith.constant 0.0 : f16
    %cmp = arith.maximumf %in, %zero : f16
    linalg.yield %cmp : f16
  } -> tensor<128x512xf16>

  return %relu : tensor<128x512xf16>
}

// The outlined fused kernel function should be private and have the attribute.
// CHECK: func.func private @fused_
// CHECK-SAME: npu.fused_kernel
// The original function should call the fused kernel.
// CHECK: func.func @matmul_relu
// CHECK:   func.call @fused_
