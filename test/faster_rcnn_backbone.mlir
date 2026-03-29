// Test: Faster R-CNN backbone block (ResNet stage transition with stride-2 downsampling)
// Input: [1, 512, 28, 28] f16
// Main path: 1x1 conv 512->256, relu, 3x3 conv 256->256 stride 2 pad 1 (pad 28->30, output 14x14), relu, 1x1 conv 256->1024
// Shortcut: 1x1 conv 512->1024 stride 2 (output 14x14)
// Add main + shortcut, relu

func.func @faster_rcnn_backbone(
    %input: tensor<1x512x28x28xf16>,
    %w1: tensor<256x512x1x1xf16>,
    %w2: tensor<256x256x3x3xf16>,
    %w3: tensor<1024x256x1x1xf16>,
    %w_shortcut: tensor<1024x512x1x1xf16>) -> tensor<1x1024x14x14xf16> {
  %cst = arith.constant 0.0 : f16

  // === Main path ===

  // 1x1 conv: 512->256, stride 1, no padding
  // Input [1,512,28,28], filter [256,512,1,1] -> output [1,256,28,28]
  %init1 = tensor.empty() : tensor<1x256x28x28xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %conv1 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %w1 : tensor<1x512x28x28xf16>, tensor<256x512x1x1xf16>)
    outs(%fill1 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>

  // ReLU after conv1
  %empty1 = tensor.empty() : tensor<1x256x28x28xf16>
  %relu1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1 : tensor<1x256x28x28xf16>)
    outs(%empty1 : tensor<1x256x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x28x28xf16>

  // Pad for 3x3 conv stride 2: 28->30
  %padded2 = tensor.pad %relu1 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x28x28xf16> to tensor<1x256x30x30xf16>

  // 3x3 conv: 256->256, stride 2
  // Padded input [1,256,30,30], filter [256,256,3,3] -> output (30-3)/2+1=14, [1,256,14,14]
  %init2 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init2 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv2 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%padded2, %w2 : tensor<1x256x30x30xf16>, tensor<256x256x3x3xf16>)
    outs(%fill2 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>

  // ReLU after conv2
  %empty2 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv2 : tensor<1x256x14x14xf16>)
    outs(%empty2 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // 1x1 conv: 256->1024, stride 1, no padding
  // Input [1,256,14,14], filter [1024,256,1,1] -> output [1,1024,14,14]
  %init3 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill3 = linalg.fill ins(%cst : f16) outs(%init3 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu2, %w3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill3 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>

  // === Shortcut path ===

  // 1x1 conv: 512->1024, stride 2, no padding needed for 1x1 kernel
  // Input [1,512,28,28], filter [1024,512,1,1] -> output (28-1)/2+1=14, [1,1024,14,14]
  %init_sc = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill_sc = linalg.fill ins(%cst : f16) outs(%init_sc : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %shortcut = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%input, %w_shortcut : tensor<1x512x28x28xf16>, tensor<1024x512x1x1xf16>)
    outs(%fill_sc : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>

  // Add: main path + shortcut
  %empty3 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3, %shortcut : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty3 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>

  // Final ReLU
  %empty4 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add : tensor<1x1024x14x14xf16>)
    outs(%empty4 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  return %relu3 : tensor<1x1024x14x14xf16>
}
