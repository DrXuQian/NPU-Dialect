func.func @yolov3(
    %input: tensor<1x3x416x416xf16>,
    %w0: tensor<32x3x3x3xf16>,
    %w1: tensor<64x32x3x3xf16>,
    %w2: tensor<128x64x3x3xf16>,
    %w3: tensor<256x128x3x3xf16>,
    %w4: tensor<512x256x3x3xf16>,
    %w5: tensor<1024x512x3x3xf16>,
    %w6: tensor<512x1024x1x1xf16>,
    %w7: tensor<1024x512x3x3xf16>,
    %w8: tensor<512x1024x1x1xf16>,
    %w_det: tensor<255x512x1x1xf16>) -> tensor<1x255x13x13xf16> {
  %cst = arith.constant 0.0 : f16

  // conv0: 3x3 s1 3->32 416x416
  %pad0 = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x416x416xf16> to tensor<1x3x418x418xf16>
  %init1 = tensor.empty() : tensor<1x32x416x416xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x32x416x416xf16>) -> tensor<1x32x416x416xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad0, %w0 : tensor<1x3x418x418xf16>, tensor<32x3x3x3xf16>)
    outs(%fill2 : tensor<1x32x416x416xf16>) -> tensor<1x32x416x416xf16>
  %empty4 = tensor.empty() : tensor<1x32x416x416xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x32x416x416xf16>)
    outs(%empty4 : tensor<1x32x416x416xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x416x416xf16>

  // conv1: 3x3 s2 32->64 416x416
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x32x416x416xf16> to tensor<1x32x418x418xf16>
  %init7 = tensor.empty() : tensor<1x64x208x208xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x64x208x208xf16>) -> tensor<1x64x208x208xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad6, %w1 : tensor<1x32x418x418xf16>, tensor<64x32x3x3xf16>)
    outs(%fill8 : tensor<1x64x208x208xf16>) -> tensor<1x64x208x208xf16>
  %empty10 = tensor.empty() : tensor<1x64x208x208xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x64x208x208xf16>)
    outs(%empty10 : tensor<1x64x208x208xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x208x208xf16>

  // conv2: 3x3 s2 64->128 208x208
  %pad12 = tensor.pad %relu11 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x208x208xf16> to tensor<1x64x210x210xf16>
  %init13 = tensor.empty() : tensor<1x128x104x104xf16>
  %fill14 = linalg.fill ins(%cst : f16) outs(%init13 : tensor<1x128x104x104xf16>) -> tensor<1x128x104x104xf16>
  %conv15 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad12, %w2 : tensor<1x64x210x210xf16>, tensor<128x64x3x3xf16>)
    outs(%fill14 : tensor<1x128x104x104xf16>) -> tensor<1x128x104x104xf16>
  %empty16 = tensor.empty() : tensor<1x128x104x104xf16>
  %relu17 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv15 : tensor<1x128x104x104xf16>)
    outs(%empty16 : tensor<1x128x104x104xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x104x104xf16>

  // conv3: 3x3 s2 128->256 104x104
  %pad18 = tensor.pad %relu17 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x104x104xf16> to tensor<1x128x106x106xf16>
  %init19 = tensor.empty() : tensor<1x256x52x52xf16>
  %fill20 = linalg.fill ins(%cst : f16) outs(%init19 : tensor<1x256x52x52xf16>) -> tensor<1x256x52x52xf16>
  %conv21 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad18, %w3 : tensor<1x128x106x106xf16>, tensor<256x128x3x3xf16>)
    outs(%fill20 : tensor<1x256x52x52xf16>) -> tensor<1x256x52x52xf16>
  %empty22 = tensor.empty() : tensor<1x256x52x52xf16>
  %relu23 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv21 : tensor<1x256x52x52xf16>)
    outs(%empty22 : tensor<1x256x52x52xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x52x52xf16>

  // conv4: 3x3 s2 256->512 52x52
  %pad24 = tensor.pad %relu23 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x52x52xf16> to tensor<1x256x54x54xf16>
  %init25 = tensor.empty() : tensor<1x512x26x26xf16>
  %fill26 = linalg.fill ins(%cst : f16) outs(%init25 : tensor<1x512x26x26xf16>) -> tensor<1x512x26x26xf16>
  %conv27 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad24, %w4 : tensor<1x256x54x54xf16>, tensor<512x256x3x3xf16>)
    outs(%fill26 : tensor<1x512x26x26xf16>) -> tensor<1x512x26x26xf16>
  %empty28 = tensor.empty() : tensor<1x512x26x26xf16>
  %relu29 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv27 : tensor<1x512x26x26xf16>)
    outs(%empty28 : tensor<1x512x26x26xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x26x26xf16>

  // conv5: 3x3 s2 512->1024 26x26
  %pad30 = tensor.pad %relu29 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x26x26xf16> to tensor<1x512x28x28xf16>
  %init31 = tensor.empty() : tensor<1x1024x13x13xf16>
  %fill32 = linalg.fill ins(%cst : f16) outs(%init31 : tensor<1x1024x13x13xf16>) -> tensor<1x1024x13x13xf16>
  %conv33 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad30, %w5 : tensor<1x512x28x28xf16>, tensor<1024x512x3x3xf16>)
    outs(%fill32 : tensor<1x1024x13x13xf16>) -> tensor<1x1024x13x13xf16>
  %empty34 = tensor.empty() : tensor<1x1024x13x13xf16>
  %relu35 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv33 : tensor<1x1024x13x13xf16>)
    outs(%empty34 : tensor<1x1024x13x13xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x13x13xf16>

  // conv6: 1x1 s1 1024->512 13x13
  %init36 = tensor.empty() : tensor<1x512x13x13xf16>
  %fill37 = linalg.fill ins(%cst : f16) outs(%init36 : tensor<1x512x13x13xf16>) -> tensor<1x512x13x13xf16>
  %conv38 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu35, %w6 : tensor<1x1024x13x13xf16>, tensor<512x1024x1x1xf16>)
    outs(%fill37 : tensor<1x512x13x13xf16>) -> tensor<1x512x13x13xf16>
  %empty39 = tensor.empty() : tensor<1x512x13x13xf16>
  %relu40 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv38 : tensor<1x512x13x13xf16>)
    outs(%empty39 : tensor<1x512x13x13xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x13x13xf16>

  // conv7: 3x3 s1 512->1024 13x13
  %pad41 = tensor.pad %relu40 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x13x13xf16> to tensor<1x512x15x15xf16>
  %init42 = tensor.empty() : tensor<1x1024x13x13xf16>
  %fill43 = linalg.fill ins(%cst : f16) outs(%init42 : tensor<1x1024x13x13xf16>) -> tensor<1x1024x13x13xf16>
  %conv44 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad41, %w7 : tensor<1x512x15x15xf16>, tensor<1024x512x3x3xf16>)
    outs(%fill43 : tensor<1x1024x13x13xf16>) -> tensor<1x1024x13x13xf16>
  %empty45 = tensor.empty() : tensor<1x1024x13x13xf16>
  %relu46 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv44 : tensor<1x1024x13x13xf16>)
    outs(%empty45 : tensor<1x1024x13x13xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x13x13xf16>

  // conv8: 1x1 s1 1024->512 13x13
  %init47 = tensor.empty() : tensor<1x512x13x13xf16>
  %fill48 = linalg.fill ins(%cst : f16) outs(%init47 : tensor<1x512x13x13xf16>) -> tensor<1x512x13x13xf16>
  %conv49 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu46, %w8 : tensor<1x1024x13x13xf16>, tensor<512x1024x1x1xf16>)
    outs(%fill48 : tensor<1x512x13x13xf16>) -> tensor<1x512x13x13xf16>
  %empty50 = tensor.empty() : tensor<1x512x13x13xf16>
  %relu51 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv49 : tensor<1x512x13x13xf16>)
    outs(%empty50 : tensor<1x512x13x13xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x13x13xf16>

  // Detection head: 1x1 512->255
  %init52 = tensor.empty() : tensor<1x255x13x13xf16>
  %fill53 = linalg.fill ins(%cst : f16) outs(%init52 : tensor<1x255x13x13xf16>) -> tensor<1x255x13x13xf16>
  %conv54 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu51, %w_det : tensor<1x512x13x13xf16>, tensor<255x512x1x1xf16>)
    outs(%fill53 : tensor<1x255x13x13xf16>) -> tensor<1x255x13x13xf16>
  return %conv54 : tensor<1x255x13x13xf16>
}
