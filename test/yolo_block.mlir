// Test: YOLO detection head block
// Input: [1, 256, 52, 52] f16
// 1x1 conv 256->128, relu
// 3x3 conv 128->256 stride 1 pad 1 (pad 52->54), relu
// 1x1 conv 256->255 (detection output: 85*3 anchors)

func.func @yolo_block(
    %input: tensor<1x256x52x52xf16>,
    %w1: tensor<128x256x1x1xf16>,
    %w2: tensor<256x128x3x3xf16>,
    %w3: tensor<255x256x1x1xf16>) -> tensor<1x255x52x52xf16> {
  %cst = arith.constant 0.0 : f16

  // 1x1 conv: 256->128, no padding
  // Input [1,256,52,52], filter [128,256,1,1] -> output [1,128,52,52]
  %init1 = tensor.empty() : tensor<1x128x52x52xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x128x52x52xf16>) -> tensor<1x128x52x52xf16>
  %conv1 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %w1 : tensor<1x256x52x52xf16>, tensor<128x256x1x1xf16>)
    outs(%fill1 : tensor<1x128x52x52xf16>) -> tensor<1x128x52x52xf16>

  // ReLU after conv1
  %empty1 = tensor.empty() : tensor<1x128x52x52xf16>
  %relu1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1 : tensor<1x128x52x52xf16>)
    outs(%empty1 : tensor<1x128x52x52xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x52x52xf16>

  // Pad for 3x3 conv: 52->54
  %padded2 = tensor.pad %relu1 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x52x52xf16> to tensor<1x128x54x54xf16>

  // 3x3 conv: 128->256, stride 1
  // Padded input [1,128,54,54], filter [256,128,3,3] -> output (54-3)/1+1=52, [1,256,52,52]
  %init2 = tensor.empty() : tensor<1x256x52x52xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init2 : tensor<1x256x52x52xf16>) -> tensor<1x256x52x52xf16>
  %conv2 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%padded2, %w2 : tensor<1x128x54x54xf16>, tensor<256x128x3x3xf16>)
    outs(%fill2 : tensor<1x256x52x52xf16>) -> tensor<1x256x52x52xf16>

  // ReLU after conv2
  %empty2 = tensor.empty() : tensor<1x256x52x52xf16>
  %relu2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv2 : tensor<1x256x52x52xf16>)
    outs(%empty2 : tensor<1x256x52x52xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x52x52xf16>

  // 1x1 conv: 256->255 (detection output), no padding
  // Input [1,256,52,52], filter [255,256,1,1] -> output [1,255,52,52]
  %init3 = tensor.empty() : tensor<1x255x52x52xf16>
  %fill3 = linalg.fill ins(%cst : f16) outs(%init3 : tensor<1x255x52x52xf16>) -> tensor<1x255x52x52xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu2, %w3 : tensor<1x256x52x52xf16>, tensor<255x256x1x1xf16>)
    outs(%fill3 : tensor<1x255x52x52xf16>) -> tensor<1x255x52x52xf16>

  return %conv3 : tensor<1x255x52x52xf16>
}
