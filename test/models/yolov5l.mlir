func.func @yolov5l(
    %input: tensor<1x3x416x416xf16>,
    %w0: tensor<64x3x3x3xf16>,
    %w1: tensor<128x64x3x3xf16>,
    %w2: tensor<128x128x3x3xf16>,
    %w3: tensor<128x128x3x3xf16>,
    %w4: tensor<128x128x3x3xf16>,
    %w5: tensor<256x128x3x3xf16>,
    %w6: tensor<256x256x3x3xf16>,
    %w7: tensor<256x256x3x3xf16>,
    %w8: tensor<256x256x3x3xf16>,
    %w9: tensor<256x256x3x3xf16>,
    %w10: tensor<512x256x3x3xf16>,
    %w11: tensor<512x512x3x3xf16>,
    %w12: tensor<512x512x3x3xf16>,
    %w13: tensor<512x512x3x3xf16>,
    %w14: tensor<1024x512x3x3xf16>,
    %w15: tensor<1024x1024x3x3xf16>,
    %w_det: tensor<255x1024x1x1xf16>) -> tensor<1x255x26x26xf16> {
  %cst = arith.constant 0.0 : f16

  // conv0: 3x3 s1 3->64 416x416
  %pad0 = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x416x416xf16> to tensor<1x3x418x418xf16>
  %init1 = tensor.empty() : tensor<1x64x416x416xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x64x416x416xf16>) -> tensor<1x64x416x416xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad0, %w0 : tensor<1x3x418x418xf16>, tensor<64x3x3x3xf16>)
    outs(%fill2 : tensor<1x64x416x416xf16>) -> tensor<1x64x416x416xf16>
  %empty4 = tensor.empty() : tensor<1x64x416x416xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x64x416x416xf16>)
    outs(%empty4 : tensor<1x64x416x416xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x416x416xf16>

  // conv1: 3x3 s2 64->128 416x416
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x416x416xf16> to tensor<1x64x418x418xf16>
  %init7 = tensor.empty() : tensor<1x128x208x208xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x128x208x208xf16>) -> tensor<1x128x208x208xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad6, %w1 : tensor<1x64x418x418xf16>, tensor<128x64x3x3xf16>)
    outs(%fill8 : tensor<1x128x208x208xf16>) -> tensor<1x128x208x208xf16>
  %empty10 = tensor.empty() : tensor<1x128x208x208xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x128x208x208xf16>)
    outs(%empty10 : tensor<1x128x208x208xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x208x208xf16>

  // conv2: 3x3 s1 128->128 208x208
  %pad12 = tensor.pad %relu11 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x208x208xf16> to tensor<1x128x210x210xf16>
  %init13 = tensor.empty() : tensor<1x128x208x208xf16>
  %fill14 = linalg.fill ins(%cst : f16) outs(%init13 : tensor<1x128x208x208xf16>) -> tensor<1x128x208x208xf16>
  %conv15 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad12, %w2 : tensor<1x128x210x210xf16>, tensor<128x128x3x3xf16>)
    outs(%fill14 : tensor<1x128x208x208xf16>) -> tensor<1x128x208x208xf16>
  %empty16 = tensor.empty() : tensor<1x128x208x208xf16>
  %relu17 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv15 : tensor<1x128x208x208xf16>)
    outs(%empty16 : tensor<1x128x208x208xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x208x208xf16>

  // conv3: 3x3 s1 128->128 208x208
  %pad18 = tensor.pad %relu17 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x208x208xf16> to tensor<1x128x210x210xf16>
  %init19 = tensor.empty() : tensor<1x128x208x208xf16>
  %fill20 = linalg.fill ins(%cst : f16) outs(%init19 : tensor<1x128x208x208xf16>) -> tensor<1x128x208x208xf16>
  %conv21 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad18, %w3 : tensor<1x128x210x210xf16>, tensor<128x128x3x3xf16>)
    outs(%fill20 : tensor<1x128x208x208xf16>) -> tensor<1x128x208x208xf16>
  %empty22 = tensor.empty() : tensor<1x128x208x208xf16>
  %relu23 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv21 : tensor<1x128x208x208xf16>)
    outs(%empty22 : tensor<1x128x208x208xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x208x208xf16>

  // conv4: 3x3 s1 128->128 208x208
  %pad24 = tensor.pad %relu23 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x208x208xf16> to tensor<1x128x210x210xf16>
  %init25 = tensor.empty() : tensor<1x128x208x208xf16>
  %fill26 = linalg.fill ins(%cst : f16) outs(%init25 : tensor<1x128x208x208xf16>) -> tensor<1x128x208x208xf16>
  %conv27 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad24, %w4 : tensor<1x128x210x210xf16>, tensor<128x128x3x3xf16>)
    outs(%fill26 : tensor<1x128x208x208xf16>) -> tensor<1x128x208x208xf16>
  %empty28 = tensor.empty() : tensor<1x128x208x208xf16>
  %relu29 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv27 : tensor<1x128x208x208xf16>)
    outs(%empty28 : tensor<1x128x208x208xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x208x208xf16>

  // conv5: 3x3 s2 128->256 208x208
  %pad30 = tensor.pad %relu29 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x208x208xf16> to tensor<1x128x210x210xf16>
  %init31 = tensor.empty() : tensor<1x256x104x104xf16>
  %fill32 = linalg.fill ins(%cst : f16) outs(%init31 : tensor<1x256x104x104xf16>) -> tensor<1x256x104x104xf16>
  %conv33 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad30, %w5 : tensor<1x128x210x210xf16>, tensor<256x128x3x3xf16>)
    outs(%fill32 : tensor<1x256x104x104xf16>) -> tensor<1x256x104x104xf16>
  %empty34 = tensor.empty() : tensor<1x256x104x104xf16>
  %relu35 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv33 : tensor<1x256x104x104xf16>)
    outs(%empty34 : tensor<1x256x104x104xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x104x104xf16>

  // conv6: 3x3 s1 256->256 104x104
  %pad36 = tensor.pad %relu35 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x104x104xf16> to tensor<1x256x106x106xf16>
  %init37 = tensor.empty() : tensor<1x256x104x104xf16>
  %fill38 = linalg.fill ins(%cst : f16) outs(%init37 : tensor<1x256x104x104xf16>) -> tensor<1x256x104x104xf16>
  %conv39 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad36, %w6 : tensor<1x256x106x106xf16>, tensor<256x256x3x3xf16>)
    outs(%fill38 : tensor<1x256x104x104xf16>) -> tensor<1x256x104x104xf16>
  %empty40 = tensor.empty() : tensor<1x256x104x104xf16>
  %relu41 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv39 : tensor<1x256x104x104xf16>)
    outs(%empty40 : tensor<1x256x104x104xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x104x104xf16>

  // conv7: 3x3 s1 256->256 104x104
  %pad42 = tensor.pad %relu41 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x104x104xf16> to tensor<1x256x106x106xf16>
  %init43 = tensor.empty() : tensor<1x256x104x104xf16>
  %fill44 = linalg.fill ins(%cst : f16) outs(%init43 : tensor<1x256x104x104xf16>) -> tensor<1x256x104x104xf16>
  %conv45 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad42, %w7 : tensor<1x256x106x106xf16>, tensor<256x256x3x3xf16>)
    outs(%fill44 : tensor<1x256x104x104xf16>) -> tensor<1x256x104x104xf16>
  %empty46 = tensor.empty() : tensor<1x256x104x104xf16>
  %relu47 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv45 : tensor<1x256x104x104xf16>)
    outs(%empty46 : tensor<1x256x104x104xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x104x104xf16>

  // conv8: 3x3 s1 256->256 104x104
  %pad48 = tensor.pad %relu47 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x104x104xf16> to tensor<1x256x106x106xf16>
  %init49 = tensor.empty() : tensor<1x256x104x104xf16>
  %fill50 = linalg.fill ins(%cst : f16) outs(%init49 : tensor<1x256x104x104xf16>) -> tensor<1x256x104x104xf16>
  %conv51 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad48, %w8 : tensor<1x256x106x106xf16>, tensor<256x256x3x3xf16>)
    outs(%fill50 : tensor<1x256x104x104xf16>) -> tensor<1x256x104x104xf16>
  %empty52 = tensor.empty() : tensor<1x256x104x104xf16>
  %relu53 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv51 : tensor<1x256x104x104xf16>)
    outs(%empty52 : tensor<1x256x104x104xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x104x104xf16>

  // conv9: 3x3 s1 256->256 104x104
  %pad54 = tensor.pad %relu53 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x104x104xf16> to tensor<1x256x106x106xf16>
  %init55 = tensor.empty() : tensor<1x256x104x104xf16>
  %fill56 = linalg.fill ins(%cst : f16) outs(%init55 : tensor<1x256x104x104xf16>) -> tensor<1x256x104x104xf16>
  %conv57 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad54, %w9 : tensor<1x256x106x106xf16>, tensor<256x256x3x3xf16>)
    outs(%fill56 : tensor<1x256x104x104xf16>) -> tensor<1x256x104x104xf16>
  %empty58 = tensor.empty() : tensor<1x256x104x104xf16>
  %relu59 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv57 : tensor<1x256x104x104xf16>)
    outs(%empty58 : tensor<1x256x104x104xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x104x104xf16>

  // conv10: 3x3 s2 256->512 104x104
  %pad60 = tensor.pad %relu59 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x104x104xf16> to tensor<1x256x106x106xf16>
  %init61 = tensor.empty() : tensor<1x512x52x52xf16>
  %fill62 = linalg.fill ins(%cst : f16) outs(%init61 : tensor<1x512x52x52xf16>) -> tensor<1x512x52x52xf16>
  %conv63 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad60, %w10 : tensor<1x256x106x106xf16>, tensor<512x256x3x3xf16>)
    outs(%fill62 : tensor<1x512x52x52xf16>) -> tensor<1x512x52x52xf16>
  %empty64 = tensor.empty() : tensor<1x512x52x52xf16>
  %relu65 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv63 : tensor<1x512x52x52xf16>)
    outs(%empty64 : tensor<1x512x52x52xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x52x52xf16>

  // conv11: 3x3 s1 512->512 52x52
  %pad66 = tensor.pad %relu65 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x52x52xf16> to tensor<1x512x54x54xf16>
  %init67 = tensor.empty() : tensor<1x512x52x52xf16>
  %fill68 = linalg.fill ins(%cst : f16) outs(%init67 : tensor<1x512x52x52xf16>) -> tensor<1x512x52x52xf16>
  %conv69 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad66, %w11 : tensor<1x512x54x54xf16>, tensor<512x512x3x3xf16>)
    outs(%fill68 : tensor<1x512x52x52xf16>) -> tensor<1x512x52x52xf16>
  %empty70 = tensor.empty() : tensor<1x512x52x52xf16>
  %relu71 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv69 : tensor<1x512x52x52xf16>)
    outs(%empty70 : tensor<1x512x52x52xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x52x52xf16>

  // conv12: 3x3 s1 512->512 52x52
  %pad72 = tensor.pad %relu71 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x52x52xf16> to tensor<1x512x54x54xf16>
  %init73 = tensor.empty() : tensor<1x512x52x52xf16>
  %fill74 = linalg.fill ins(%cst : f16) outs(%init73 : tensor<1x512x52x52xf16>) -> tensor<1x512x52x52xf16>
  %conv75 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad72, %w12 : tensor<1x512x54x54xf16>, tensor<512x512x3x3xf16>)
    outs(%fill74 : tensor<1x512x52x52xf16>) -> tensor<1x512x52x52xf16>
  %empty76 = tensor.empty() : tensor<1x512x52x52xf16>
  %relu77 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv75 : tensor<1x512x52x52xf16>)
    outs(%empty76 : tensor<1x512x52x52xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x52x52xf16>

  // conv13: 3x3 s1 512->512 52x52
  %pad78 = tensor.pad %relu77 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x52x52xf16> to tensor<1x512x54x54xf16>
  %init79 = tensor.empty() : tensor<1x512x52x52xf16>
  %fill80 = linalg.fill ins(%cst : f16) outs(%init79 : tensor<1x512x52x52xf16>) -> tensor<1x512x52x52xf16>
  %conv81 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad78, %w13 : tensor<1x512x54x54xf16>, tensor<512x512x3x3xf16>)
    outs(%fill80 : tensor<1x512x52x52xf16>) -> tensor<1x512x52x52xf16>
  %empty82 = tensor.empty() : tensor<1x512x52x52xf16>
  %relu83 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv81 : tensor<1x512x52x52xf16>)
    outs(%empty82 : tensor<1x512x52x52xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x52x52xf16>

  // conv14: 3x3 s2 512->1024 52x52
  %pad84 = tensor.pad %relu83 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x52x52xf16> to tensor<1x512x54x54xf16>
  %init85 = tensor.empty() : tensor<1x1024x26x26xf16>
  %fill86 = linalg.fill ins(%cst : f16) outs(%init85 : tensor<1x1024x26x26xf16>) -> tensor<1x1024x26x26xf16>
  %conv87 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad84, %w14 : tensor<1x512x54x54xf16>, tensor<1024x512x3x3xf16>)
    outs(%fill86 : tensor<1x1024x26x26xf16>) -> tensor<1x1024x26x26xf16>
  %empty88 = tensor.empty() : tensor<1x1024x26x26xf16>
  %relu89 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv87 : tensor<1x1024x26x26xf16>)
    outs(%empty88 : tensor<1x1024x26x26xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x26x26xf16>

  // conv15: 3x3 s1 1024->1024 26x26
  %pad90 = tensor.pad %relu89 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1024x26x26xf16> to tensor<1x1024x28x28xf16>
  %init91 = tensor.empty() : tensor<1x1024x26x26xf16>
  %fill92 = linalg.fill ins(%cst : f16) outs(%init91 : tensor<1x1024x26x26xf16>) -> tensor<1x1024x26x26xf16>
  %conv93 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad90, %w15 : tensor<1x1024x28x28xf16>, tensor<1024x1024x3x3xf16>)
    outs(%fill92 : tensor<1x1024x26x26xf16>) -> tensor<1x1024x26x26xf16>
  %empty94 = tensor.empty() : tensor<1x1024x26x26xf16>
  %relu95 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv93 : tensor<1x1024x26x26xf16>)
    outs(%empty94 : tensor<1x1024x26x26xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x26x26xf16>

  // Detection head: 1x1 1024->255
  %init96 = tensor.empty() : tensor<1x255x26x26xf16>
  %fill97 = linalg.fill ins(%cst : f16) outs(%init96 : tensor<1x255x26x26xf16>) -> tensor<1x255x26x26xf16>
  %conv98 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu95, %w_det : tensor<1x1024x26x26xf16>, tensor<255x1024x1x1xf16>)
    outs(%fill97 : tensor<1x255x26x26xf16>) -> tensor<1x255x26x26xf16>
  return %conv98 : tensor<1x255x26x26xf16>
}
