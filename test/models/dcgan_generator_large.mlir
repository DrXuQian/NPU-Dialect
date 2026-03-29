func.func @dcgan_generator_large(
    %input: tensor<1x200x4x4xf16>,
    %w0: tensor<1024x200x3x3xf16>,
    %w1: tensor<512x1024x3x3xf16>,
    %w2: tensor<256x512x3x3xf16>,
    %w3: tensor<128x256x3x3xf16>,
    %w4: tensor<64x128x3x3xf16>,
    %w5: tensor<3x64x3x3xf16>) -> tensor<1x3x4x4xf16> {
  %cst = arith.constant 0.0 : f16

  // conv0: 3x3 s1 p1 200->1024 4x4
  %pad0 = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x200x4x4xf16> to tensor<1x200x6x6xf16>
  %init1 = tensor.empty() : tensor<1x1024x4x4xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x1024x4x4xf16>) -> tensor<1x1024x4x4xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad0, %w0 : tensor<1x200x6x6xf16>, tensor<1024x200x3x3xf16>)
    outs(%fill2 : tensor<1x1024x4x4xf16>) -> tensor<1x1024x4x4xf16>
  %empty4 = tensor.empty() : tensor<1x1024x4x4xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x1024x4x4xf16>)
    outs(%empty4 : tensor<1x1024x4x4xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x4x4xf16>

  // conv1: 3x3 s1 p1 1024->512 4x4
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1024x4x4xf16> to tensor<1x1024x6x6xf16>
  %init7 = tensor.empty() : tensor<1x512x4x4xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x512x4x4xf16>) -> tensor<1x512x4x4xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad6, %w1 : tensor<1x1024x6x6xf16>, tensor<512x1024x3x3xf16>)
    outs(%fill8 : tensor<1x512x4x4xf16>) -> tensor<1x512x4x4xf16>
  %empty10 = tensor.empty() : tensor<1x512x4x4xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x512x4x4xf16>)
    outs(%empty10 : tensor<1x512x4x4xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x4x4xf16>

  // conv2: 3x3 s1 p1 512->256 4x4
  %pad12 = tensor.pad %relu11 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x4x4xf16> to tensor<1x512x6x6xf16>
  %init13 = tensor.empty() : tensor<1x256x4x4xf16>
  %fill14 = linalg.fill ins(%cst : f16) outs(%init13 : tensor<1x256x4x4xf16>) -> tensor<1x256x4x4xf16>
  %conv15 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad12, %w2 : tensor<1x512x6x6xf16>, tensor<256x512x3x3xf16>)
    outs(%fill14 : tensor<1x256x4x4xf16>) -> tensor<1x256x4x4xf16>
  %empty16 = tensor.empty() : tensor<1x256x4x4xf16>
  %relu17 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv15 : tensor<1x256x4x4xf16>)
    outs(%empty16 : tensor<1x256x4x4xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x4x4xf16>

  // conv3: 3x3 s1 p1 256->128 4x4
  %pad18 = tensor.pad %relu17 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x4x4xf16> to tensor<1x256x6x6xf16>
  %init19 = tensor.empty() : tensor<1x128x4x4xf16>
  %fill20 = linalg.fill ins(%cst : f16) outs(%init19 : tensor<1x128x4x4xf16>) -> tensor<1x128x4x4xf16>
  %conv21 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad18, %w3 : tensor<1x256x6x6xf16>, tensor<128x256x3x3xf16>)
    outs(%fill20 : tensor<1x128x4x4xf16>) -> tensor<1x128x4x4xf16>
  %empty22 = tensor.empty() : tensor<1x128x4x4xf16>
  %relu23 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv21 : tensor<1x128x4x4xf16>)
    outs(%empty22 : tensor<1x128x4x4xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x4x4xf16>

  // conv4: 3x3 s1 p1 128->64 4x4
  %pad24 = tensor.pad %relu23 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x4x4xf16> to tensor<1x128x6x6xf16>
  %init25 = tensor.empty() : tensor<1x64x4x4xf16>
  %fill26 = linalg.fill ins(%cst : f16) outs(%init25 : tensor<1x64x4x4xf16>) -> tensor<1x64x4x4xf16>
  %conv27 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad24, %w4 : tensor<1x128x6x6xf16>, tensor<64x128x3x3xf16>)
    outs(%fill26 : tensor<1x64x4x4xf16>) -> tensor<1x64x4x4xf16>
  %empty28 = tensor.empty() : tensor<1x64x4x4xf16>
  %relu29 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv27 : tensor<1x64x4x4xf16>)
    outs(%empty28 : tensor<1x64x4x4xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x4x4xf16>

  // conv5: 3x3 s1 p1 64->3 4x4
  %pad30 = tensor.pad %relu29 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x4x4xf16> to tensor<1x64x6x6xf16>
  %init31 = tensor.empty() : tensor<1x3x4x4xf16>
  %fill32 = linalg.fill ins(%cst : f16) outs(%init31 : tensor<1x3x4x4xf16>) -> tensor<1x3x4x4xf16>
  %conv33 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad30, %w5 : tensor<1x64x6x6xf16>, tensor<3x64x3x3xf16>)
    outs(%fill32 : tensor<1x3x4x4xf16>) -> tensor<1x3x4x4xf16>

  return %conv33 : tensor<1x3x4x4xf16>
}
