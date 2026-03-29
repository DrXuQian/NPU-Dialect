func.func @vgg19(
    %input: tensor<1x3x224x224xf16>,
    %w_b0_c0: tensor<64x3x3x3xf16>,
    %w_b0_c1: tensor<64x64x3x3xf16>,
    %w_b0_pool: tensor<64x64x3x3xf16>,
    %w_b1_c0: tensor<128x64x3x3xf16>,
    %w_b1_c1: tensor<128x128x3x3xf16>,
    %w_b1_pool: tensor<128x128x3x3xf16>,
    %w_b2_c0: tensor<256x128x3x3xf16>,
    %w_b2_c1: tensor<256x256x3x3xf16>,
    %w_b2_c2: tensor<256x256x3x3xf16>,
    %w_b2_c3: tensor<256x256x3x3xf16>,
    %w_b2_pool: tensor<256x256x3x3xf16>,
    %w_b3_c0: tensor<512x256x3x3xf16>,
    %w_b3_c1: tensor<512x512x3x3xf16>,
    %w_b3_c2: tensor<512x512x3x3xf16>,
    %w_b3_c3: tensor<512x512x3x3xf16>,
    %w_b3_pool: tensor<512x512x3x3xf16>,
    %w_b4_c0: tensor<512x512x3x3xf16>,
    %w_b4_c1: tensor<512x512x3x3xf16>,
    %w_b4_c2: tensor<512x512x3x3xf16>,
    %w_b4_c3: tensor<512x512x3x3xf16>,
    %w_b4_pool: tensor<512x512x3x3xf16>,
    %w_fc1: tensor<4096x512x7x7xf16>,
    %w_fc2: tensor<4096x4096x1x1xf16>,
    %w_fc3: tensor<1000x4096x1x1xf16>) -> tensor<1x1000x1x1xf16> {
  %cst = arith.constant 0.0 : f16

  // VGG block 0: 2 convs, 64 channels
  %pad0 = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x224x224xf16> to tensor<1x3x226x226xf16>
  %init1 = tensor.empty() : tensor<1x64x224x224xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x64x224x224xf16>) -> tensor<1x64x224x224xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad0, %w_b0_c0 : tensor<1x3x226x226xf16>, tensor<64x3x3x3xf16>)
    outs(%fill2 : tensor<1x64x224x224xf16>) -> tensor<1x64x224x224xf16>
  %empty4 = tensor.empty() : tensor<1x64x224x224xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x64x224x224xf16>)
    outs(%empty4 : tensor<1x64x224x224xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x224x224xf16>
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x224x224xf16> to tensor<1x64x226x226xf16>
  %init7 = tensor.empty() : tensor<1x64x224x224xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x64x224x224xf16>) -> tensor<1x64x224x224xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad6, %w_b0_c1 : tensor<1x64x226x226xf16>, tensor<64x64x3x3xf16>)
    outs(%fill8 : tensor<1x64x224x224xf16>) -> tensor<1x64x224x224xf16>
  %empty10 = tensor.empty() : tensor<1x64x224x224xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x64x224x224xf16>)
    outs(%empty10 : tensor<1x64x224x224xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x224x224xf16>
  // Pool: 224->112
  %pad12 = tensor.pad %relu11 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x224x224xf16> to tensor<1x64x226x226xf16>
  %init13 = tensor.empty() : tensor<1x64x112x112xf16>
  %fill14 = linalg.fill ins(%cst : f16) outs(%init13 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %conv15 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad12, %w_b0_pool : tensor<1x64x226x226xf16>, tensor<64x64x3x3xf16>)
    outs(%fill14 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %empty16 = tensor.empty() : tensor<1x64x112x112xf16>
  %relu17 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv15 : tensor<1x64x112x112xf16>)
    outs(%empty16 : tensor<1x64x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x112x112xf16>

  // VGG block 1: 2 convs, 128 channels
  %pad18 = tensor.pad %relu17 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x112x112xf16> to tensor<1x64x114x114xf16>
  %init19 = tensor.empty() : tensor<1x128x112x112xf16>
  %fill20 = linalg.fill ins(%cst : f16) outs(%init19 : tensor<1x128x112x112xf16>) -> tensor<1x128x112x112xf16>
  %conv21 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad18, %w_b1_c0 : tensor<1x64x114x114xf16>, tensor<128x64x3x3xf16>)
    outs(%fill20 : tensor<1x128x112x112xf16>) -> tensor<1x128x112x112xf16>
  %empty22 = tensor.empty() : tensor<1x128x112x112xf16>
  %relu23 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv21 : tensor<1x128x112x112xf16>)
    outs(%empty22 : tensor<1x128x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x112x112xf16>
  %pad24 = tensor.pad %relu23 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x112x112xf16> to tensor<1x128x114x114xf16>
  %init25 = tensor.empty() : tensor<1x128x112x112xf16>
  %fill26 = linalg.fill ins(%cst : f16) outs(%init25 : tensor<1x128x112x112xf16>) -> tensor<1x128x112x112xf16>
  %conv27 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad24, %w_b1_c1 : tensor<1x128x114x114xf16>, tensor<128x128x3x3xf16>)
    outs(%fill26 : tensor<1x128x112x112xf16>) -> tensor<1x128x112x112xf16>
  %empty28 = tensor.empty() : tensor<1x128x112x112xf16>
  %relu29 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv27 : tensor<1x128x112x112xf16>)
    outs(%empty28 : tensor<1x128x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x112x112xf16>
  // Pool: 112->56
  %pad30 = tensor.pad %relu29 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x112x112xf16> to tensor<1x128x114x114xf16>
  %init31 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill32 = linalg.fill ins(%cst : f16) outs(%init31 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv33 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad30, %w_b1_pool : tensor<1x128x114x114xf16>, tensor<128x128x3x3xf16>)
    outs(%fill32 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %empty34 = tensor.empty() : tensor<1x128x56x56xf16>
  %relu35 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv33 : tensor<1x128x56x56xf16>)
    outs(%empty34 : tensor<1x128x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x56x56xf16>

  // VGG block 2: 4 convs, 256 channels
  %pad36 = tensor.pad %relu35 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x56x56xf16> to tensor<1x128x58x58xf16>
  %init37 = tensor.empty() : tensor<1x256x56x56xf16>
  %fill38 = linalg.fill ins(%cst : f16) outs(%init37 : tensor<1x256x56x56xf16>) -> tensor<1x256x56x56xf16>
  %conv39 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad36, %w_b2_c0 : tensor<1x128x58x58xf16>, tensor<256x128x3x3xf16>)
    outs(%fill38 : tensor<1x256x56x56xf16>) -> tensor<1x256x56x56xf16>
  %empty40 = tensor.empty() : tensor<1x256x56x56xf16>
  %relu41 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv39 : tensor<1x256x56x56xf16>)
    outs(%empty40 : tensor<1x256x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x56x56xf16>
  %pad42 = tensor.pad %relu41 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x56x56xf16> to tensor<1x256x58x58xf16>
  %init43 = tensor.empty() : tensor<1x256x56x56xf16>
  %fill44 = linalg.fill ins(%cst : f16) outs(%init43 : tensor<1x256x56x56xf16>) -> tensor<1x256x56x56xf16>
  %conv45 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad42, %w_b2_c1 : tensor<1x256x58x58xf16>, tensor<256x256x3x3xf16>)
    outs(%fill44 : tensor<1x256x56x56xf16>) -> tensor<1x256x56x56xf16>
  %empty46 = tensor.empty() : tensor<1x256x56x56xf16>
  %relu47 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv45 : tensor<1x256x56x56xf16>)
    outs(%empty46 : tensor<1x256x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x56x56xf16>
  %pad48 = tensor.pad %relu47 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x56x56xf16> to tensor<1x256x58x58xf16>
  %init49 = tensor.empty() : tensor<1x256x56x56xf16>
  %fill50 = linalg.fill ins(%cst : f16) outs(%init49 : tensor<1x256x56x56xf16>) -> tensor<1x256x56x56xf16>
  %conv51 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad48, %w_b2_c2 : tensor<1x256x58x58xf16>, tensor<256x256x3x3xf16>)
    outs(%fill50 : tensor<1x256x56x56xf16>) -> tensor<1x256x56x56xf16>
  %empty52 = tensor.empty() : tensor<1x256x56x56xf16>
  %relu53 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv51 : tensor<1x256x56x56xf16>)
    outs(%empty52 : tensor<1x256x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x56x56xf16>
  %pad54 = tensor.pad %relu53 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x56x56xf16> to tensor<1x256x58x58xf16>
  %init55 = tensor.empty() : tensor<1x256x56x56xf16>
  %fill56 = linalg.fill ins(%cst : f16) outs(%init55 : tensor<1x256x56x56xf16>) -> tensor<1x256x56x56xf16>
  %conv57 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad54, %w_b2_c3 : tensor<1x256x58x58xf16>, tensor<256x256x3x3xf16>)
    outs(%fill56 : tensor<1x256x56x56xf16>) -> tensor<1x256x56x56xf16>
  %empty58 = tensor.empty() : tensor<1x256x56x56xf16>
  %relu59 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv57 : tensor<1x256x56x56xf16>)
    outs(%empty58 : tensor<1x256x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x56x56xf16>
  // Pool: 56->28
  %pad60 = tensor.pad %relu59 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x56x56xf16> to tensor<1x256x58x58xf16>
  %init61 = tensor.empty() : tensor<1x256x28x28xf16>
  %fill62 = linalg.fill ins(%cst : f16) outs(%init61 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %conv63 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad60, %w_b2_pool : tensor<1x256x58x58xf16>, tensor<256x256x3x3xf16>)
    outs(%fill62 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %empty64 = tensor.empty() : tensor<1x256x28x28xf16>
  %relu65 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv63 : tensor<1x256x28x28xf16>)
    outs(%empty64 : tensor<1x256x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x28x28xf16>

  // VGG block 3: 4 convs, 512 channels
  %pad66 = tensor.pad %relu65 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x28x28xf16> to tensor<1x256x30x30xf16>
  %init67 = tensor.empty() : tensor<1x512x28x28xf16>
  %fill68 = linalg.fill ins(%cst : f16) outs(%init67 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %conv69 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad66, %w_b3_c0 : tensor<1x256x30x30xf16>, tensor<512x256x3x3xf16>)
    outs(%fill68 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %empty70 = tensor.empty() : tensor<1x512x28x28xf16>
  %relu71 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv69 : tensor<1x512x28x28xf16>)
    outs(%empty70 : tensor<1x512x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x28x28xf16>
  %pad72 = tensor.pad %relu71 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x28x28xf16> to tensor<1x512x30x30xf16>
  %init73 = tensor.empty() : tensor<1x512x28x28xf16>
  %fill74 = linalg.fill ins(%cst : f16) outs(%init73 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %conv75 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad72, %w_b3_c1 : tensor<1x512x30x30xf16>, tensor<512x512x3x3xf16>)
    outs(%fill74 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %empty76 = tensor.empty() : tensor<1x512x28x28xf16>
  %relu77 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv75 : tensor<1x512x28x28xf16>)
    outs(%empty76 : tensor<1x512x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x28x28xf16>
  %pad78 = tensor.pad %relu77 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x28x28xf16> to tensor<1x512x30x30xf16>
  %init79 = tensor.empty() : tensor<1x512x28x28xf16>
  %fill80 = linalg.fill ins(%cst : f16) outs(%init79 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %conv81 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad78, %w_b3_c2 : tensor<1x512x30x30xf16>, tensor<512x512x3x3xf16>)
    outs(%fill80 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %empty82 = tensor.empty() : tensor<1x512x28x28xf16>
  %relu83 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv81 : tensor<1x512x28x28xf16>)
    outs(%empty82 : tensor<1x512x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x28x28xf16>
  %pad84 = tensor.pad %relu83 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x28x28xf16> to tensor<1x512x30x30xf16>
  %init85 = tensor.empty() : tensor<1x512x28x28xf16>
  %fill86 = linalg.fill ins(%cst : f16) outs(%init85 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %conv87 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad84, %w_b3_c3 : tensor<1x512x30x30xf16>, tensor<512x512x3x3xf16>)
    outs(%fill86 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %empty88 = tensor.empty() : tensor<1x512x28x28xf16>
  %relu89 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv87 : tensor<1x512x28x28xf16>)
    outs(%empty88 : tensor<1x512x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x28x28xf16>
  // Pool: 28->14
  %pad90 = tensor.pad %relu89 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x28x28xf16> to tensor<1x512x30x30xf16>
  %init91 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill92 = linalg.fill ins(%cst : f16) outs(%init91 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv93 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad90, %w_b3_pool : tensor<1x512x30x30xf16>, tensor<512x512x3x3xf16>)
    outs(%fill92 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty94 = tensor.empty() : tensor<1x512x14x14xf16>
  %relu95 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv93 : tensor<1x512x14x14xf16>)
    outs(%empty94 : tensor<1x512x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x14x14xf16>

  // VGG block 4: 4 convs, 512 channels
  %pad96 = tensor.pad %relu95 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x16x16xf16>
  %init97 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill98 = linalg.fill ins(%cst : f16) outs(%init97 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv99 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad96, %w_b4_c0 : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>)
    outs(%fill98 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty100 = tensor.empty() : tensor<1x512x14x14xf16>
  %relu101 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv99 : tensor<1x512x14x14xf16>)
    outs(%empty100 : tensor<1x512x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x14x14xf16>
  %pad102 = tensor.pad %relu101 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x16x16xf16>
  %init103 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill104 = linalg.fill ins(%cst : f16) outs(%init103 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv105 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad102, %w_b4_c1 : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>)
    outs(%fill104 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty106 = tensor.empty() : tensor<1x512x14x14xf16>
  %relu107 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv105 : tensor<1x512x14x14xf16>)
    outs(%empty106 : tensor<1x512x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x14x14xf16>
  %pad108 = tensor.pad %relu107 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x16x16xf16>
  %init109 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill110 = linalg.fill ins(%cst : f16) outs(%init109 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv111 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad108, %w_b4_c2 : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>)
    outs(%fill110 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty112 = tensor.empty() : tensor<1x512x14x14xf16>
  %relu113 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv111 : tensor<1x512x14x14xf16>)
    outs(%empty112 : tensor<1x512x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x14x14xf16>
  %pad114 = tensor.pad %relu113 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x16x16xf16>
  %init115 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill116 = linalg.fill ins(%cst : f16) outs(%init115 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv117 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad114, %w_b4_c3 : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>)
    outs(%fill116 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty118 = tensor.empty() : tensor<1x512x14x14xf16>
  %relu119 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv117 : tensor<1x512x14x14xf16>)
    outs(%empty118 : tensor<1x512x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x14x14xf16>
  // Pool: 14->7
  %pad120 = tensor.pad %relu119 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x16x16xf16>
  %init121 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill122 = linalg.fill ins(%cst : f16) outs(%init121 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv123 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad120, %w_b4_pool : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>)
    outs(%fill122 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty124 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu125 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv123 : tensor<1x512x7x7xf16>)
    outs(%empty124 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // FC1 as 7x7 conv -> 4096
  %init126 = tensor.empty() : tensor<1x4096x1x1xf16>
  %fill127 = linalg.fill ins(%cst : f16) outs(%init126 : tensor<1x4096x1x1xf16>) -> tensor<1x4096x1x1xf16>
  %fc1128 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu125, %w_fc1 : tensor<1x512x7x7xf16>, tensor<4096x512x7x7xf16>)
    outs(%fill127 : tensor<1x4096x1x1xf16>) -> tensor<1x4096x1x1xf16>
  %empty129 = tensor.empty() : tensor<1x4096x1x1xf16>
  %relu130 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%fc1128 : tensor<1x4096x1x1xf16>)
    outs(%empty129 : tensor<1x4096x1x1xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x4096x1x1xf16>

  // FC2 as 1x1 conv -> 4096
  %init131 = tensor.empty() : tensor<1x4096x1x1xf16>
  %fill132 = linalg.fill ins(%cst : f16) outs(%init131 : tensor<1x4096x1x1xf16>) -> tensor<1x4096x1x1xf16>
  %conv133 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu130, %w_fc2 : tensor<1x4096x1x1xf16>, tensor<4096x4096x1x1xf16>)
    outs(%fill132 : tensor<1x4096x1x1xf16>) -> tensor<1x4096x1x1xf16>
  %empty134 = tensor.empty() : tensor<1x4096x1x1xf16>
  %relu135 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv133 : tensor<1x4096x1x1xf16>)
    outs(%empty134 : tensor<1x4096x1x1xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x4096x1x1xf16>

  // FC3 as 1x1 conv -> 1000
  %init136 = tensor.empty() : tensor<1x1000x1x1xf16>
  %fill137 = linalg.fill ins(%cst : f16) outs(%init136 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  %conv138 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu135, %w_fc3 : tensor<1x4096x1x1xf16>, tensor<1000x4096x1x1xf16>)
    outs(%fill137 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  return %conv138 : tensor<1x1000x1x1xf16>
}
