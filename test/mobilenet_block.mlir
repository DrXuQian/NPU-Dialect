// Test: MobileNetV2 inverted residual block (simplified)
// Input: [1, 32, 112, 112] f16
// 1x1 conv 32->192 (expand), relu
// 3x3 conv 192->192 stride 1 pad 1 (depthwise approximation), relu
// 1x1 conv 192->32 (project)
// residual add with input

func.func @mobilenet_block(
    %input: tensor<1x32x112x112xf16>,
    %w_expand: tensor<192x32x1x1xf16>,
    %w_dw: tensor<192x192x3x3xf16>,
    %w_project: tensor<32x192x1x1xf16>) -> tensor<1x32x112x112xf16> {
  %cst = arith.constant 0.0 : f16

  // 1x1 conv expand: 32->192, no padding
  // Input [1,32,112,112], filter [192,32,1,1] -> output [1,192,112,112]
  %init1 = tensor.empty() : tensor<1x192x112x112xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x192x112x112xf16>) -> tensor<1x192x112x112xf16>
  %conv1 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %w_expand : tensor<1x32x112x112xf16>, tensor<192x32x1x1xf16>)
    outs(%fill1 : tensor<1x192x112x112xf16>) -> tensor<1x192x112x112xf16>

  // ReLU after expand conv
  %empty1 = tensor.empty() : tensor<1x192x112x112xf16>
  %relu1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1 : tensor<1x192x112x112xf16>)
    outs(%empty1 : tensor<1x192x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x112x112xf16>

  // Pad for 3x3 conv: 112->114
  %padded2 = tensor.pad %relu1 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x192x112x112xf16> to tensor<1x192x114x114xf16>

  // 3x3 conv depthwise (modeled as standard conv): 192->192, stride 1
  // Padded input [1,192,114,114], filter [192,192,3,3] -> output (114-3)/1+1=112, [1,192,112,112]
  %init2 = tensor.empty() : tensor<1x192x112x112xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init2 : tensor<1x192x112x112xf16>) -> tensor<1x192x112x112xf16>
  %conv2 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%padded2, %w_dw : tensor<1x192x114x114xf16>, tensor<192x192x3x3xf16>)
    outs(%fill2 : tensor<1x192x112x112xf16>) -> tensor<1x192x112x112xf16>

  // ReLU after depthwise conv
  %empty2 = tensor.empty() : tensor<1x192x112x112xf16>
  %relu2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv2 : tensor<1x192x112x112xf16>)
    outs(%empty2 : tensor<1x192x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x112x112xf16>

  // 1x1 conv project: 192->32, no padding
  // Input [1,192,112,112], filter [32,192,1,1] -> output [1,32,112,112]
  %init3 = tensor.empty() : tensor<1x32x112x112xf16>
  %fill3 = linalg.fill ins(%cst : f16) outs(%init3 : tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu2, %w_project : tensor<1x192x112x112xf16>, tensor<32x192x1x1xf16>)
    outs(%fill3 : tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16>

  // Residual add: conv3 + input
  %empty3 = tensor.empty() : tensor<1x32x112x112xf16>
  %add = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3, %input : tensor<1x32x112x112xf16>, tensor<1x32x112x112xf16>)
    outs(%empty3 : tensor<1x32x112x112xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x32x112x112xf16>

  return %add : tensor<1x32x112x112xf16>
}
