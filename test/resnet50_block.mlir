// Test: ResNet-50 bottleneck block
// Input: [1, 256, 56, 56] f16
// 1x1 conv 256->64, relu, 3x3 conv 64->64 pad 1, relu, 1x1 conv 64->256, residual add, relu

func.func @resnet50_block(
    %input: tensor<1x256x56x56xf16>,
    %w1: tensor<64x256x1x1xf16>,
    %w2: tensor<64x64x3x3xf16>,
    %w3: tensor<256x64x1x1xf16>) -> tensor<1x256x56x56xf16> {
  %cst = arith.constant 0.0 : f16

  // 1x1 conv: 256->64, no padding needed
  // Input [1,256,56,56], filter [64,256,1,1] -> output [1,64,56,56]
  %init1 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv1 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %w1 : tensor<1x256x56x56xf16>, tensor<64x256x1x1xf16>)
    outs(%fill1 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>

  // ReLU after conv1
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

  // Pad for 3x3 conv: 56->58
  %padded2 = tensor.pad %relu1 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>

  // 3x3 conv: 64->64, stride 1
  // Padded input [1,64,58,58], filter [64,64,3,3] -> output (58-3)/1+1=56, [1,64,56,56]
  %init2 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init2 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv2 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%padded2, %w2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill2 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>

  // ReLU after conv2
  %empty2 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv2 : tensor<1x64x56x56xf16>)
    outs(%empty2 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // 1x1 conv: 64->256, no padding needed
  // Input [1,64,56,56], filter [256,64,1,1] -> output [1,256,56,56]
  %init3 = tensor.empty() : tensor<1x256x56x56xf16>
  %fill3 = linalg.fill ins(%cst : f16) outs(%init3 : tensor<1x256x56x56xf16>) -> tensor<1x256x56x56xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu2, %w3 : tensor<1x64x56x56xf16>, tensor<256x64x1x1xf16>)
    outs(%fill3 : tensor<1x256x56x56xf16>) -> tensor<1x256x56x56xf16>

  // Residual add: conv3 + input
  %empty3 = tensor.empty() : tensor<1x256x56x56xf16>
  %add = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3, %input : tensor<1x256x56x56xf16>, tensor<1x256x56x56xf16>)
    outs(%empty3 : tensor<1x256x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x56x56xf16>

  // Final ReLU
  %empty4 = tensor.empty() : tensor<1x256x56x56xf16>
  %relu3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add : tensor<1x256x56x56xf16>)
    outs(%empty4 : tensor<1x256x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x56x56xf16>

  return %relu3 : tensor<1x256x56x56xf16>
}
