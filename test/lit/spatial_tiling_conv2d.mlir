// RUN: %npu-opt %s --npu-spatial-tiling | FileCheck %s

// Test: conv2d spatial tiling splits on output channels (Co),
// producing extract_slice over the filter and output tensors.
// The input tensor (pre-padded 58x58) should appear as a full slice
// since it is shared across Co tiles.

func.func @conv2d_spatial(
    %input: tensor<1x64x56x56xf16>,
    %filter: tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16> {
  %cst = arith.constant 0.0 : f16
  %padded = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>

  %init = tensor.empty() : tensor<1x64x56x56xf16>
  %fill = linalg.fill ins(%cst : f16) outs(%init : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%padded, %filter : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>

  return %conv : tensor<1x64x56x56xf16>
}

// Spatial tiling should produce an scf.for over output channels.
// The input is fully shared (1x64x58x58), filter is sliced on Co dim.
// CHECK-LABEL: func @conv2d_spatial
// CHECK:       scf.for
// CHECK:         tensor.extract_slice {{.*}} tensor<1x64x58x58xf16>
// CHECK:         linalg.conv_2d_nchw_fchw
// CHECK:         tensor.insert_slice
