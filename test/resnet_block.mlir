// Test: ResNet basic block (conv -> relu -> conv -> residual add)

func.func @resnet_block(
    %input: tensor<1x64x56x56xf16>,
    %w1: tensor<64x64x3x3xf16>,
    %w2: tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16> {
  %cst = arith.constant 0.0 : f16

  // Conv1 + ReLU
  %init1 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv1 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %w1 : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>)
    outs(%fill1 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>

  %empty1 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1 : tensor<1x64x56x56xf16>)
    outs(%empty1 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // Conv2
  %init2 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init2 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv2 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1, %w2 : tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>)
    outs(%fill2 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>

  // Residual add
  %empty2 = tensor.empty() : tensor<1x64x56x56xf16>
  %add = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv2, %input : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty2 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>

  return %add : tensor<1x64x56x56xf16>
}
