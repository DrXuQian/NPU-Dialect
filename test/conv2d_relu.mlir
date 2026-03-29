// Test: conv2d -> relu with padding, spatial tiling handles halo regions
//
// Conv2d 3x3 stride=1 on 56x56 input → needs pad=1 for same-size output.
// linalg.conv_2d_nchw_fchw does not have built-in padding,
// so the input must be pre-padded to 58x58.

func.func @conv2d_relu(
    %input: tensor<1x64x56x56xf16>,
    %filter: tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16> {
  %cst = arith.constant 0.0 : f16

  // Pad input: 56x56 → 58x58 (pad 1 on each side of H and W)
  %padded = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>

  // Conv2d on padded input → 56x56 output
  %init = tensor.empty() : tensor<1x64x56x56xf16>
  %fill = linalg.fill ins(%cst : f16) outs(%init : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%padded, %filter : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>

  // relu
  %empty = tensor.empty() : tensor<1x64x56x56xf16>
  %relu = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv : tensor<1x64x56x56xf16>)
    outs(%empty : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %zero = arith.constant 0.0 : f16
    %max = arith.maximumf %in, %zero : f16
    linalg.yield %max : f16
  } -> tensor<1x64x56x56xf16>

  return %relu : tensor<1x64x56x56xf16>
}
