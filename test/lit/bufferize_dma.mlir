// RUN: %npu-opt %s --npu-spatial-tiling --npu-bufferize | FileCheck %s

// Test: after bufferization, SRAM alloc and DMA copies should be inserted
// for memref values used inside scf.for loops.

func.func @bufferize_matmul(%A: tensor<128x256xf16>, %B: tensor<256x64xf16>) -> tensor<128x64xf16> {
  %cst = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<128x64xf16>
  %fill = linalg.fill ins(%cst : f16) outs(%init : tensor<128x64xf16>) -> tensor<128x64xf16>
  %result = linalg.matmul ins(%A, %B : tensor<128x256xf16>, tensor<256x64xf16>)
                          outs(%fill : tensor<128x64xf16>) -> tensor<128x64xf16>
  return %result : tensor<128x64xf16>
}

// After bufferize, we should see npu.alloc_sram and npu.dma_copy ops
// inside the loop body for DRAM->SRAM transfers.
// CHECK-LABEL: func @bufferize_matmul
// CHECK:       npu.alloc_sram
// CHECK:       npu.dma_copy
// CHECK-SAME:  dram_to_sram
