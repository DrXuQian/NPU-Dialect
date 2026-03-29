func.func @squeezenet1_0(
    %input: tensor<1x3x224x224xf16>,
    %w_stem: tensor<96x3x7x7xf16>,
    %w_fire0_sq: tensor<16x96x1x1xf16>,
    %w_fire0_e1: tensor<64x16x1x1xf16>,
    %w_fire0_e3: tensor<64x64x3x3xf16>,
    %w_fire1_sq: tensor<16x64x1x1xf16>,
    %w_fire1_e1: tensor<64x16x1x1xf16>,
    %w_fire1_e3: tensor<64x64x3x3xf16>,
    %w_fire2_sq: tensor<32x64x1x1xf16>,
    %w_fire2_e1: tensor<128x32x1x1xf16>,
    %w_fire2_e3: tensor<128x128x3x3xf16>,
    %w_fire3_sq: tensor<32x128x1x1xf16>,
    %w_fire3_e1: tensor<128x32x1x1xf16>,
    %w_fire3_e3: tensor<128x128x3x3xf16>,
    %w_fire4_sq: tensor<48x128x1x1xf16>,
    %w_fire4_e1: tensor<192x48x1x1xf16>,
    %w_fire4_e3: tensor<192x192x3x3xf16>,
    %w_fire5_sq: tensor<48x192x1x1xf16>,
    %w_fire5_e1: tensor<192x48x1x1xf16>,
    %w_fire5_e3: tensor<192x192x3x3xf16>,
    %w_fire6_sq: tensor<64x192x1x1xf16>,
    %w_fire6_e1: tensor<256x64x1x1xf16>,
    %w_fire6_e3: tensor<256x256x3x3xf16>,
    %w_fire7_sq: tensor<64x256x1x1xf16>,
    %w_fire7_e1: tensor<256x64x1x1xf16>,
    %w_fire7_e3: tensor<256x256x3x3xf16>,
    %w_final: tensor<1000x256x1x1xf16>) -> tensor<1x1000x112x112xf16> {
  %cst = arith.constant 0.0 : f16

  // Stem: 7x7 stride 2, 3->96
  %pad0 = tensor.pad %input low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x224x224xf16> to tensor<1x3x230x230xf16>
  %init1 = tensor.empty() : tensor<1x96x112x112xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x96x112x112xf16>) -> tensor<1x96x112x112xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad0, %w_stem : tensor<1x3x230x230xf16>, tensor<96x3x7x7xf16>)
    outs(%fill2 : tensor<1x96x112x112xf16>) -> tensor<1x96x112x112xf16>
  %empty4 = tensor.empty() : tensor<1x96x112x112xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x96x112x112xf16>)
    outs(%empty4 : tensor<1x96x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x96x112x112xf16>

  // Fire0: squeeze=16, expand=64
  %init6 = tensor.empty() : tensor<1x16x112x112xf16>
  %fill7 = linalg.fill ins(%cst : f16) outs(%init6 : tensor<1x16x112x112xf16>) -> tensor<1x16x112x112xf16>
  %conv8 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu5, %w_fire0_sq : tensor<1x96x112x112xf16>, tensor<16x96x1x1xf16>)
    outs(%fill7 : tensor<1x16x112x112xf16>) -> tensor<1x16x112x112xf16>
  %empty9 = tensor.empty() : tensor<1x16x112x112xf16>
  %relu10 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv8 : tensor<1x16x112x112xf16>)
    outs(%empty9 : tensor<1x16x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x16x112x112xf16>
  %init11 = tensor.empty() : tensor<1x64x112x112xf16>
  %fill12 = linalg.fill ins(%cst : f16) outs(%init11 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %conv13 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu10, %w_fire0_e1 : tensor<1x16x112x112xf16>, tensor<64x16x1x1xf16>)
    outs(%fill12 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %empty14 = tensor.empty() : tensor<1x64x112x112xf16>
  %relu15 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv13 : tensor<1x64x112x112xf16>)
    outs(%empty14 : tensor<1x64x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x112x112xf16>
  %pad16 = tensor.pad %relu15 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x112x112xf16> to tensor<1x64x114x114xf16>
  %init17 = tensor.empty() : tensor<1x64x112x112xf16>
  %fill18 = linalg.fill ins(%cst : f16) outs(%init17 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %conv19 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad16, %w_fire0_e3 : tensor<1x64x114x114xf16>, tensor<64x64x3x3xf16>)
    outs(%fill18 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %empty20 = tensor.empty() : tensor<1x64x112x112xf16>
  %relu21 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv19 : tensor<1x64x112x112xf16>)
    outs(%empty20 : tensor<1x64x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x112x112xf16>

  // Fire1: squeeze=16, expand=64
  %init22 = tensor.empty() : tensor<1x16x112x112xf16>
  %fill23 = linalg.fill ins(%cst : f16) outs(%init22 : tensor<1x16x112x112xf16>) -> tensor<1x16x112x112xf16>
  %conv24 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu21, %w_fire1_sq : tensor<1x64x112x112xf16>, tensor<16x64x1x1xf16>)
    outs(%fill23 : tensor<1x16x112x112xf16>) -> tensor<1x16x112x112xf16>
  %empty25 = tensor.empty() : tensor<1x16x112x112xf16>
  %relu26 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv24 : tensor<1x16x112x112xf16>)
    outs(%empty25 : tensor<1x16x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x16x112x112xf16>
  %init27 = tensor.empty() : tensor<1x64x112x112xf16>
  %fill28 = linalg.fill ins(%cst : f16) outs(%init27 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %conv29 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu26, %w_fire1_e1 : tensor<1x16x112x112xf16>, tensor<64x16x1x1xf16>)
    outs(%fill28 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %empty30 = tensor.empty() : tensor<1x64x112x112xf16>
  %relu31 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv29 : tensor<1x64x112x112xf16>)
    outs(%empty30 : tensor<1x64x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x112x112xf16>
  %pad32 = tensor.pad %relu31 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x112x112xf16> to tensor<1x64x114x114xf16>
  %init33 = tensor.empty() : tensor<1x64x112x112xf16>
  %fill34 = linalg.fill ins(%cst : f16) outs(%init33 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %conv35 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad32, %w_fire1_e3 : tensor<1x64x114x114xf16>, tensor<64x64x3x3xf16>)
    outs(%fill34 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %empty36 = tensor.empty() : tensor<1x64x112x112xf16>
  %relu37 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv35 : tensor<1x64x112x112xf16>)
    outs(%empty36 : tensor<1x64x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x112x112xf16>

  // Fire2: squeeze=32, expand=128
  %init38 = tensor.empty() : tensor<1x32x112x112xf16>
  %fill39 = linalg.fill ins(%cst : f16) outs(%init38 : tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16>
  %conv40 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu37, %w_fire2_sq : tensor<1x64x112x112xf16>, tensor<32x64x1x1xf16>)
    outs(%fill39 : tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16>
  %empty41 = tensor.empty() : tensor<1x32x112x112xf16>
  %relu42 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv40 : tensor<1x32x112x112xf16>)
    outs(%empty41 : tensor<1x32x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x112x112xf16>
  %init43 = tensor.empty() : tensor<1x128x112x112xf16>
  %fill44 = linalg.fill ins(%cst : f16) outs(%init43 : tensor<1x128x112x112xf16>) -> tensor<1x128x112x112xf16>
  %conv45 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu42, %w_fire2_e1 : tensor<1x32x112x112xf16>, tensor<128x32x1x1xf16>)
    outs(%fill44 : tensor<1x128x112x112xf16>) -> tensor<1x128x112x112xf16>
  %empty46 = tensor.empty() : tensor<1x128x112x112xf16>
  %relu47 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv45 : tensor<1x128x112x112xf16>)
    outs(%empty46 : tensor<1x128x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x112x112xf16>
  %pad48 = tensor.pad %relu47 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x112x112xf16> to tensor<1x128x114x114xf16>
  %init49 = tensor.empty() : tensor<1x128x112x112xf16>
  %fill50 = linalg.fill ins(%cst : f16) outs(%init49 : tensor<1x128x112x112xf16>) -> tensor<1x128x112x112xf16>
  %conv51 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad48, %w_fire2_e3 : tensor<1x128x114x114xf16>, tensor<128x128x3x3xf16>)
    outs(%fill50 : tensor<1x128x112x112xf16>) -> tensor<1x128x112x112xf16>
  %empty52 = tensor.empty() : tensor<1x128x112x112xf16>
  %relu53 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv51 : tensor<1x128x112x112xf16>)
    outs(%empty52 : tensor<1x128x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x112x112xf16>

  // Fire3: squeeze=32, expand=128
  %init54 = tensor.empty() : tensor<1x32x112x112xf16>
  %fill55 = linalg.fill ins(%cst : f16) outs(%init54 : tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16>
  %conv56 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu53, %w_fire3_sq : tensor<1x128x112x112xf16>, tensor<32x128x1x1xf16>)
    outs(%fill55 : tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16>
  %empty57 = tensor.empty() : tensor<1x32x112x112xf16>
  %relu58 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv56 : tensor<1x32x112x112xf16>)
    outs(%empty57 : tensor<1x32x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x112x112xf16>
  %init59 = tensor.empty() : tensor<1x128x112x112xf16>
  %fill60 = linalg.fill ins(%cst : f16) outs(%init59 : tensor<1x128x112x112xf16>) -> tensor<1x128x112x112xf16>
  %conv61 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu58, %w_fire3_e1 : tensor<1x32x112x112xf16>, tensor<128x32x1x1xf16>)
    outs(%fill60 : tensor<1x128x112x112xf16>) -> tensor<1x128x112x112xf16>
  %empty62 = tensor.empty() : tensor<1x128x112x112xf16>
  %relu63 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv61 : tensor<1x128x112x112xf16>)
    outs(%empty62 : tensor<1x128x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x112x112xf16>
  %pad64 = tensor.pad %relu63 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x112x112xf16> to tensor<1x128x114x114xf16>
  %init65 = tensor.empty() : tensor<1x128x112x112xf16>
  %fill66 = linalg.fill ins(%cst : f16) outs(%init65 : tensor<1x128x112x112xf16>) -> tensor<1x128x112x112xf16>
  %conv67 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad64, %w_fire3_e3 : tensor<1x128x114x114xf16>, tensor<128x128x3x3xf16>)
    outs(%fill66 : tensor<1x128x112x112xf16>) -> tensor<1x128x112x112xf16>
  %empty68 = tensor.empty() : tensor<1x128x112x112xf16>
  %relu69 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv67 : tensor<1x128x112x112xf16>)
    outs(%empty68 : tensor<1x128x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x112x112xf16>

  // Fire4: squeeze=48, expand=192
  %init70 = tensor.empty() : tensor<1x48x112x112xf16>
  %fill71 = linalg.fill ins(%cst : f16) outs(%init70 : tensor<1x48x112x112xf16>) -> tensor<1x48x112x112xf16>
  %conv72 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu69, %w_fire4_sq : tensor<1x128x112x112xf16>, tensor<48x128x1x1xf16>)
    outs(%fill71 : tensor<1x48x112x112xf16>) -> tensor<1x48x112x112xf16>
  %empty73 = tensor.empty() : tensor<1x48x112x112xf16>
  %relu74 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv72 : tensor<1x48x112x112xf16>)
    outs(%empty73 : tensor<1x48x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x48x112x112xf16>
  %init75 = tensor.empty() : tensor<1x192x112x112xf16>
  %fill76 = linalg.fill ins(%cst : f16) outs(%init75 : tensor<1x192x112x112xf16>) -> tensor<1x192x112x112xf16>
  %conv77 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu74, %w_fire4_e1 : tensor<1x48x112x112xf16>, tensor<192x48x1x1xf16>)
    outs(%fill76 : tensor<1x192x112x112xf16>) -> tensor<1x192x112x112xf16>
  %empty78 = tensor.empty() : tensor<1x192x112x112xf16>
  %relu79 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv77 : tensor<1x192x112x112xf16>)
    outs(%empty78 : tensor<1x192x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x112x112xf16>
  %pad80 = tensor.pad %relu79 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x192x112x112xf16> to tensor<1x192x114x114xf16>
  %init81 = tensor.empty() : tensor<1x192x112x112xf16>
  %fill82 = linalg.fill ins(%cst : f16) outs(%init81 : tensor<1x192x112x112xf16>) -> tensor<1x192x112x112xf16>
  %conv83 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad80, %w_fire4_e3 : tensor<1x192x114x114xf16>, tensor<192x192x3x3xf16>)
    outs(%fill82 : tensor<1x192x112x112xf16>) -> tensor<1x192x112x112xf16>
  %empty84 = tensor.empty() : tensor<1x192x112x112xf16>
  %relu85 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv83 : tensor<1x192x112x112xf16>)
    outs(%empty84 : tensor<1x192x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x112x112xf16>

  // Fire5: squeeze=48, expand=192
  %init86 = tensor.empty() : tensor<1x48x112x112xf16>
  %fill87 = linalg.fill ins(%cst : f16) outs(%init86 : tensor<1x48x112x112xf16>) -> tensor<1x48x112x112xf16>
  %conv88 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu85, %w_fire5_sq : tensor<1x192x112x112xf16>, tensor<48x192x1x1xf16>)
    outs(%fill87 : tensor<1x48x112x112xf16>) -> tensor<1x48x112x112xf16>
  %empty89 = tensor.empty() : tensor<1x48x112x112xf16>
  %relu90 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv88 : tensor<1x48x112x112xf16>)
    outs(%empty89 : tensor<1x48x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x48x112x112xf16>
  %init91 = tensor.empty() : tensor<1x192x112x112xf16>
  %fill92 = linalg.fill ins(%cst : f16) outs(%init91 : tensor<1x192x112x112xf16>) -> tensor<1x192x112x112xf16>
  %conv93 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu90, %w_fire5_e1 : tensor<1x48x112x112xf16>, tensor<192x48x1x1xf16>)
    outs(%fill92 : tensor<1x192x112x112xf16>) -> tensor<1x192x112x112xf16>
  %empty94 = tensor.empty() : tensor<1x192x112x112xf16>
  %relu95 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv93 : tensor<1x192x112x112xf16>)
    outs(%empty94 : tensor<1x192x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x112x112xf16>
  %pad96 = tensor.pad %relu95 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x192x112x112xf16> to tensor<1x192x114x114xf16>
  %init97 = tensor.empty() : tensor<1x192x112x112xf16>
  %fill98 = linalg.fill ins(%cst : f16) outs(%init97 : tensor<1x192x112x112xf16>) -> tensor<1x192x112x112xf16>
  %conv99 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad96, %w_fire5_e3 : tensor<1x192x114x114xf16>, tensor<192x192x3x3xf16>)
    outs(%fill98 : tensor<1x192x112x112xf16>) -> tensor<1x192x112x112xf16>
  %empty100 = tensor.empty() : tensor<1x192x112x112xf16>
  %relu101 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv99 : tensor<1x192x112x112xf16>)
    outs(%empty100 : tensor<1x192x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x112x112xf16>

  // Fire6: squeeze=64, expand=256
  %init102 = tensor.empty() : tensor<1x64x112x112xf16>
  %fill103 = linalg.fill ins(%cst : f16) outs(%init102 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %conv104 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu101, %w_fire6_sq : tensor<1x192x112x112xf16>, tensor<64x192x1x1xf16>)
    outs(%fill103 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %empty105 = tensor.empty() : tensor<1x64x112x112xf16>
  %relu106 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv104 : tensor<1x64x112x112xf16>)
    outs(%empty105 : tensor<1x64x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x112x112xf16>
  %init107 = tensor.empty() : tensor<1x256x112x112xf16>
  %fill108 = linalg.fill ins(%cst : f16) outs(%init107 : tensor<1x256x112x112xf16>) -> tensor<1x256x112x112xf16>
  %conv109 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu106, %w_fire6_e1 : tensor<1x64x112x112xf16>, tensor<256x64x1x1xf16>)
    outs(%fill108 : tensor<1x256x112x112xf16>) -> tensor<1x256x112x112xf16>
  %empty110 = tensor.empty() : tensor<1x256x112x112xf16>
  %relu111 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv109 : tensor<1x256x112x112xf16>)
    outs(%empty110 : tensor<1x256x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x112x112xf16>
  %pad112 = tensor.pad %relu111 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x112x112xf16> to tensor<1x256x114x114xf16>
  %init113 = tensor.empty() : tensor<1x256x112x112xf16>
  %fill114 = linalg.fill ins(%cst : f16) outs(%init113 : tensor<1x256x112x112xf16>) -> tensor<1x256x112x112xf16>
  %conv115 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad112, %w_fire6_e3 : tensor<1x256x114x114xf16>, tensor<256x256x3x3xf16>)
    outs(%fill114 : tensor<1x256x112x112xf16>) -> tensor<1x256x112x112xf16>
  %empty116 = tensor.empty() : tensor<1x256x112x112xf16>
  %relu117 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv115 : tensor<1x256x112x112xf16>)
    outs(%empty116 : tensor<1x256x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x112x112xf16>

  // Fire7: squeeze=64, expand=256
  %init118 = tensor.empty() : tensor<1x64x112x112xf16>
  %fill119 = linalg.fill ins(%cst : f16) outs(%init118 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %conv120 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu117, %w_fire7_sq : tensor<1x256x112x112xf16>, tensor<64x256x1x1xf16>)
    outs(%fill119 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %empty121 = tensor.empty() : tensor<1x64x112x112xf16>
  %relu122 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv120 : tensor<1x64x112x112xf16>)
    outs(%empty121 : tensor<1x64x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x112x112xf16>
  %init123 = tensor.empty() : tensor<1x256x112x112xf16>
  %fill124 = linalg.fill ins(%cst : f16) outs(%init123 : tensor<1x256x112x112xf16>) -> tensor<1x256x112x112xf16>
  %conv125 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu122, %w_fire7_e1 : tensor<1x64x112x112xf16>, tensor<256x64x1x1xf16>)
    outs(%fill124 : tensor<1x256x112x112xf16>) -> tensor<1x256x112x112xf16>
  %empty126 = tensor.empty() : tensor<1x256x112x112xf16>
  %relu127 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv125 : tensor<1x256x112x112xf16>)
    outs(%empty126 : tensor<1x256x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x112x112xf16>
  %pad128 = tensor.pad %relu127 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x112x112xf16> to tensor<1x256x114x114xf16>
  %init129 = tensor.empty() : tensor<1x256x112x112xf16>
  %fill130 = linalg.fill ins(%cst : f16) outs(%init129 : tensor<1x256x112x112xf16>) -> tensor<1x256x112x112xf16>
  %conv131 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad128, %w_fire7_e3 : tensor<1x256x114x114xf16>, tensor<256x256x3x3xf16>)
    outs(%fill130 : tensor<1x256x112x112xf16>) -> tensor<1x256x112x112xf16>
  %empty132 = tensor.empty() : tensor<1x256x112x112xf16>
  %relu133 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv131 : tensor<1x256x112x112xf16>)
    outs(%empty132 : tensor<1x256x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x112x112xf16>

  // Final 1x1 conv: 256->1000
  %init134 = tensor.empty() : tensor<1x1000x112x112xf16>
  %fill135 = linalg.fill ins(%cst : f16) outs(%init134 : tensor<1x1000x112x112xf16>) -> tensor<1x1000x112x112xf16>
  %conv136 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu133, %w_final : tensor<1x256x112x112xf16>, tensor<1000x256x1x1xf16>)
    outs(%fill135 : tensor<1x1000x112x112xf16>) -> tensor<1x1000x112x112xf16>
  return %conv136 : tensor<1x1000x112x112xf16>
}
