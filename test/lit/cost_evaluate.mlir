// RUN: %npu-opt %s --npu-cost-evaluate 2>&1 | FileCheck %s

// Test: cost evaluation pass emits remarks with cycle/byte estimates.
// The matmul should produce cost remarks with compute and DMA information.

func.func @cost_matmul(%A: tensor<128x256xf16>, %B: tensor<256x512xf16>) -> tensor<128x512xf16> {
  %cst = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<128x512xf16>
  %fill = linalg.fill ins(%cst : f16) outs(%init : tensor<128x512xf16>) -> tensor<128x512xf16>
  %result = linalg.matmul ins(%A, %B : tensor<128x256xf16>, tensor<256x512xf16>)
                          outs(%fill : tensor<128x512xf16>) -> tensor<128x512xf16>
  return %result : tensor<128x512xf16>
}

// The cost evaluate pass should emit a roofline remark with FLOP and byte counts.
// CHECK: remark: roofline:
// CHECK-SAME: FLOP
// CHECK-SAME: bytes
