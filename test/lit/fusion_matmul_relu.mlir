// RUN: %npu-opt %s --npu-fusion | FileCheck %s

// Test: matmul -> relu is fused via tile-and-fuse into a single scf.for loop
// containing both linalg.matmul and the relu linalg.generic.

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

// After fusion: both matmul and relu generic should be inside a scf.for loop.
// CHECK-LABEL: func @matmul_relu
// CHECK:       scf.for
// CHECK:         linalg.matmul
// CHECK:         linalg.generic
// CHECK:       scf.yield
