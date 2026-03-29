func.func @mobilenetv3_large(
    %input: tensor<1x3x224x224xf16>,
    %w_stem: tensor<16x3x3x3xf16>,
    %w_blk0_dw: tensor<16x16x3x3xf16>,
    %w_blk0_proj: tensor<16x16x1x1xf16>,
    %w_blk1_exp: tensor<64x16x1x1xf16>,
    %w_blk1_dw: tensor<64x64x3x3xf16>,
    %w_blk1_proj: tensor<24x64x1x1xf16>,
    %w_blk2_exp: tensor<72x24x1x1xf16>,
    %w_blk2_dw: tensor<72x72x3x3xf16>,
    %w_blk2_proj: tensor<24x72x1x1xf16>,
    %w_blk3_exp: tensor<72x24x1x1xf16>,
    %w_blk3_dw: tensor<72x72x3x3xf16>,
    %w_blk3_proj: tensor<40x72x1x1xf16>,
    %w_blk4_exp: tensor<120x40x1x1xf16>,
    %w_blk4_dw: tensor<120x120x3x3xf16>,
    %w_blk4_proj: tensor<40x120x1x1xf16>,
    %w_blk5_exp: tensor<120x40x1x1xf16>,
    %w_blk5_dw: tensor<120x120x3x3xf16>,
    %w_blk5_proj: tensor<40x120x1x1xf16>,
    %w_blk6_exp: tensor<240x40x1x1xf16>,
    %w_blk6_dw: tensor<240x240x3x3xf16>,
    %w_blk6_proj: tensor<80x240x1x1xf16>,
    %w_blk7_exp: tensor<160x80x1x1xf16>,
    %w_blk7_dw: tensor<160x160x3x3xf16>,
    %w_blk7_proj: tensor<80x160x1x1xf16>,
    %w_blk8_exp: tensor<160x80x1x1xf16>,
    %w_blk8_dw: tensor<160x160x3x3xf16>,
    %w_blk8_proj: tensor<80x160x1x1xf16>,
    %w_blk9_exp: tensor<480x80x1x1xf16>,
    %w_blk9_dw: tensor<480x480x3x3xf16>,
    %w_blk9_proj: tensor<112x480x1x1xf16>,
    %w_blk10_exp: tensor<672x112x1x1xf16>,
    %w_blk10_dw: tensor<672x672x3x3xf16>,
    %w_blk10_proj: tensor<112x672x1x1xf16>,
    %w_blk11_exp: tensor<672x112x1x1xf16>,
    %w_blk11_dw: tensor<672x672x3x3xf16>,
    %w_blk11_proj: tensor<160x672x1x1xf16>,
    %w_blk12_exp: tensor<960x160x1x1xf16>,
    %w_blk12_dw: tensor<960x960x3x3xf16>,
    %w_blk12_proj: tensor<160x960x1x1xf16>,
    %w_blk13_exp: tensor<960x160x1x1xf16>,
    %w_blk13_dw: tensor<960x960x3x3xf16>,
    %w_blk13_proj: tensor<160x960x1x1xf16>,
    %w_final: tensor<960x160x1x1xf16>,
    %w_fc: tensor<1000x960x1x1xf16>) -> tensor<1x1000x7x7xf16> {
  %cst = arith.constant 0.0 : f16

  // Stem: 3x3 stride 2, 3->16
  %pad0 = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x224x224xf16> to tensor<1x3x226x226xf16>
  %init1 = tensor.empty() : tensor<1x16x112x112xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x16x112x112xf16>) -> tensor<1x16x112x112xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad0, %w_stem : tensor<1x3x226x226xf16>, tensor<16x3x3x3xf16>)
    outs(%fill2 : tensor<1x16x112x112xf16>) -> tensor<1x16x112x112xf16>
  %empty4 = tensor.empty() : tensor<1x16x112x112xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x16x112x112xf16>)
    outs(%empty4 : tensor<1x16x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x16x112x112xf16>

  // IRB 0: 16->16 mid=16 s=1 112x112
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x16x112x112xf16> to tensor<1x16x114x114xf16>
  %init7 = tensor.empty() : tensor<1x16x112x112xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x16x112x112xf16>) -> tensor<1x16x112x112xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad6, %w_blk0_dw : tensor<1x16x114x114xf16>, tensor<16x16x3x3xf16>)
    outs(%fill8 : tensor<1x16x112x112xf16>) -> tensor<1x16x112x112xf16>
  %empty10 = tensor.empty() : tensor<1x16x112x112xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x16x112x112xf16>)
    outs(%empty10 : tensor<1x16x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x16x112x112xf16>
  %init12 = tensor.empty() : tensor<1x16x112x112xf16>
  %fill13 = linalg.fill ins(%cst : f16) outs(%init12 : tensor<1x16x112x112xf16>) -> tensor<1x16x112x112xf16>
  %conv14 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu11, %w_blk0_proj : tensor<1x16x112x112xf16>, tensor<16x16x1x1xf16>)
    outs(%fill13 : tensor<1x16x112x112xf16>) -> tensor<1x16x112x112xf16>
  %empty15 = tensor.empty() : tensor<1x16x112x112xf16>
  %add16 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv14, %relu5 : tensor<1x16x112x112xf16>, tensor<1x16x112x112xf16>)
    outs(%empty15 : tensor<1x16x112x112xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x16x112x112xf16>

  // IRB 1: 16->24 mid=64 s=2 112x112
  %init17 = tensor.empty() : tensor<1x64x112x112xf16>
  %fill18 = linalg.fill ins(%cst : f16) outs(%init17 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %conv19 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add16, %w_blk1_exp : tensor<1x16x112x112xf16>, tensor<64x16x1x1xf16>)
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
  %pad22 = tensor.pad %relu21 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x112x112xf16> to tensor<1x64x114x114xf16>
  %init23 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill24 = linalg.fill ins(%cst : f16) outs(%init23 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv25 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad22, %w_blk1_dw : tensor<1x64x114x114xf16>, tensor<64x64x3x3xf16>)
    outs(%fill24 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty26 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu27 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv25 : tensor<1x64x56x56xf16>)
    outs(%empty26 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %init28 = tensor.empty() : tensor<1x24x56x56xf16>
  %fill29 = linalg.fill ins(%cst : f16) outs(%init28 : tensor<1x24x56x56xf16>) -> tensor<1x24x56x56xf16>
  %conv30 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu27, %w_blk1_proj : tensor<1x64x56x56xf16>, tensor<24x64x1x1xf16>)
    outs(%fill29 : tensor<1x24x56x56xf16>) -> tensor<1x24x56x56xf16>

  // IRB 2: 24->24 mid=72 s=1 56x56
  %init31 = tensor.empty() : tensor<1x72x56x56xf16>
  %fill32 = linalg.fill ins(%cst : f16) outs(%init31 : tensor<1x72x56x56xf16>) -> tensor<1x72x56x56xf16>
  %conv33 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv30, %w_blk2_exp : tensor<1x24x56x56xf16>, tensor<72x24x1x1xf16>)
    outs(%fill32 : tensor<1x72x56x56xf16>) -> tensor<1x72x56x56xf16>
  %empty34 = tensor.empty() : tensor<1x72x56x56xf16>
  %relu35 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv33 : tensor<1x72x56x56xf16>)
    outs(%empty34 : tensor<1x72x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x56x56xf16>
  %pad36 = tensor.pad %relu35 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x72x56x56xf16> to tensor<1x72x58x58xf16>
  %init37 = tensor.empty() : tensor<1x72x56x56xf16>
  %fill38 = linalg.fill ins(%cst : f16) outs(%init37 : tensor<1x72x56x56xf16>) -> tensor<1x72x56x56xf16>
  %conv39 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad36, %w_blk2_dw : tensor<1x72x58x58xf16>, tensor<72x72x3x3xf16>)
    outs(%fill38 : tensor<1x72x56x56xf16>) -> tensor<1x72x56x56xf16>
  %empty40 = tensor.empty() : tensor<1x72x56x56xf16>
  %relu41 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv39 : tensor<1x72x56x56xf16>)
    outs(%empty40 : tensor<1x72x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x56x56xf16>
  %init42 = tensor.empty() : tensor<1x24x56x56xf16>
  %fill43 = linalg.fill ins(%cst : f16) outs(%init42 : tensor<1x24x56x56xf16>) -> tensor<1x24x56x56xf16>
  %conv44 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu41, %w_blk2_proj : tensor<1x72x56x56xf16>, tensor<24x72x1x1xf16>)
    outs(%fill43 : tensor<1x24x56x56xf16>) -> tensor<1x24x56x56xf16>
  %empty45 = tensor.empty() : tensor<1x24x56x56xf16>
  %add46 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv44, %conv30 : tensor<1x24x56x56xf16>, tensor<1x24x56x56xf16>)
    outs(%empty45 : tensor<1x24x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x24x56x56xf16>

  // IRB 3: 24->40 mid=72 s=2 56x56
  %init47 = tensor.empty() : tensor<1x72x56x56xf16>
  %fill48 = linalg.fill ins(%cst : f16) outs(%init47 : tensor<1x72x56x56xf16>) -> tensor<1x72x56x56xf16>
  %conv49 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add46, %w_blk3_exp : tensor<1x24x56x56xf16>, tensor<72x24x1x1xf16>)
    outs(%fill48 : tensor<1x72x56x56xf16>) -> tensor<1x72x56x56xf16>
  %empty50 = tensor.empty() : tensor<1x72x56x56xf16>
  %relu51 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv49 : tensor<1x72x56x56xf16>)
    outs(%empty50 : tensor<1x72x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x56x56xf16>
  %pad52 = tensor.pad %relu51 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x72x56x56xf16> to tensor<1x72x58x58xf16>
  %init53 = tensor.empty() : tensor<1x72x28x28xf16>
  %fill54 = linalg.fill ins(%cst : f16) outs(%init53 : tensor<1x72x28x28xf16>) -> tensor<1x72x28x28xf16>
  %conv55 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad52, %w_blk3_dw : tensor<1x72x58x58xf16>, tensor<72x72x3x3xf16>)
    outs(%fill54 : tensor<1x72x28x28xf16>) -> tensor<1x72x28x28xf16>
  %empty56 = tensor.empty() : tensor<1x72x28x28xf16>
  %relu57 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv55 : tensor<1x72x28x28xf16>)
    outs(%empty56 : tensor<1x72x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x28x28xf16>
  %init58 = tensor.empty() : tensor<1x40x28x28xf16>
  %fill59 = linalg.fill ins(%cst : f16) outs(%init58 : tensor<1x40x28x28xf16>) -> tensor<1x40x28x28xf16>
  %conv60 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu57, %w_blk3_proj : tensor<1x72x28x28xf16>, tensor<40x72x1x1xf16>)
    outs(%fill59 : tensor<1x40x28x28xf16>) -> tensor<1x40x28x28xf16>

  // IRB 4: 40->40 mid=120 s=1 28x28
  %init61 = tensor.empty() : tensor<1x120x28x28xf16>
  %fill62 = linalg.fill ins(%cst : f16) outs(%init61 : tensor<1x120x28x28xf16>) -> tensor<1x120x28x28xf16>
  %conv63 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv60, %w_blk4_exp : tensor<1x40x28x28xf16>, tensor<120x40x1x1xf16>)
    outs(%fill62 : tensor<1x120x28x28xf16>) -> tensor<1x120x28x28xf16>
  %empty64 = tensor.empty() : tensor<1x120x28x28xf16>
  %relu65 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv63 : tensor<1x120x28x28xf16>)
    outs(%empty64 : tensor<1x120x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x120x28x28xf16>
  %pad66 = tensor.pad %relu65 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x120x28x28xf16> to tensor<1x120x30x30xf16>
  %init67 = tensor.empty() : tensor<1x120x28x28xf16>
  %fill68 = linalg.fill ins(%cst : f16) outs(%init67 : tensor<1x120x28x28xf16>) -> tensor<1x120x28x28xf16>
  %conv69 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad66, %w_blk4_dw : tensor<1x120x30x30xf16>, tensor<120x120x3x3xf16>)
    outs(%fill68 : tensor<1x120x28x28xf16>) -> tensor<1x120x28x28xf16>
  %empty70 = tensor.empty() : tensor<1x120x28x28xf16>
  %relu71 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv69 : tensor<1x120x28x28xf16>)
    outs(%empty70 : tensor<1x120x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x120x28x28xf16>
  %init72 = tensor.empty() : tensor<1x40x28x28xf16>
  %fill73 = linalg.fill ins(%cst : f16) outs(%init72 : tensor<1x40x28x28xf16>) -> tensor<1x40x28x28xf16>
  %conv74 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu71, %w_blk4_proj : tensor<1x120x28x28xf16>, tensor<40x120x1x1xf16>)
    outs(%fill73 : tensor<1x40x28x28xf16>) -> tensor<1x40x28x28xf16>
  %empty75 = tensor.empty() : tensor<1x40x28x28xf16>
  %add76 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv74, %conv60 : tensor<1x40x28x28xf16>, tensor<1x40x28x28xf16>)
    outs(%empty75 : tensor<1x40x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x40x28x28xf16>

  // IRB 5: 40->40 mid=120 s=1 28x28
  %init77 = tensor.empty() : tensor<1x120x28x28xf16>
  %fill78 = linalg.fill ins(%cst : f16) outs(%init77 : tensor<1x120x28x28xf16>) -> tensor<1x120x28x28xf16>
  %conv79 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add76, %w_blk5_exp : tensor<1x40x28x28xf16>, tensor<120x40x1x1xf16>)
    outs(%fill78 : tensor<1x120x28x28xf16>) -> tensor<1x120x28x28xf16>
  %empty80 = tensor.empty() : tensor<1x120x28x28xf16>
  %relu81 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv79 : tensor<1x120x28x28xf16>)
    outs(%empty80 : tensor<1x120x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x120x28x28xf16>
  %pad82 = tensor.pad %relu81 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x120x28x28xf16> to tensor<1x120x30x30xf16>
  %init83 = tensor.empty() : tensor<1x120x28x28xf16>
  %fill84 = linalg.fill ins(%cst : f16) outs(%init83 : tensor<1x120x28x28xf16>) -> tensor<1x120x28x28xf16>
  %conv85 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad82, %w_blk5_dw : tensor<1x120x30x30xf16>, tensor<120x120x3x3xf16>)
    outs(%fill84 : tensor<1x120x28x28xf16>) -> tensor<1x120x28x28xf16>
  %empty86 = tensor.empty() : tensor<1x120x28x28xf16>
  %relu87 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv85 : tensor<1x120x28x28xf16>)
    outs(%empty86 : tensor<1x120x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x120x28x28xf16>
  %init88 = tensor.empty() : tensor<1x40x28x28xf16>
  %fill89 = linalg.fill ins(%cst : f16) outs(%init88 : tensor<1x40x28x28xf16>) -> tensor<1x40x28x28xf16>
  %conv90 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu87, %w_blk5_proj : tensor<1x120x28x28xf16>, tensor<40x120x1x1xf16>)
    outs(%fill89 : tensor<1x40x28x28xf16>) -> tensor<1x40x28x28xf16>
  %empty91 = tensor.empty() : tensor<1x40x28x28xf16>
  %add92 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv90, %add76 : tensor<1x40x28x28xf16>, tensor<1x40x28x28xf16>)
    outs(%empty91 : tensor<1x40x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x40x28x28xf16>

  // IRB 6: 40->80 mid=240 s=2 28x28
  %init93 = tensor.empty() : tensor<1x240x28x28xf16>
  %fill94 = linalg.fill ins(%cst : f16) outs(%init93 : tensor<1x240x28x28xf16>) -> tensor<1x240x28x28xf16>
  %conv95 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add92, %w_blk6_exp : tensor<1x40x28x28xf16>, tensor<240x40x1x1xf16>)
    outs(%fill94 : tensor<1x240x28x28xf16>) -> tensor<1x240x28x28xf16>
  %empty96 = tensor.empty() : tensor<1x240x28x28xf16>
  %relu97 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv95 : tensor<1x240x28x28xf16>)
    outs(%empty96 : tensor<1x240x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x240x28x28xf16>
  %pad98 = tensor.pad %relu97 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x240x28x28xf16> to tensor<1x240x30x30xf16>
  %init99 = tensor.empty() : tensor<1x240x14x14xf16>
  %fill100 = linalg.fill ins(%cst : f16) outs(%init99 : tensor<1x240x14x14xf16>) -> tensor<1x240x14x14xf16>
  %conv101 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad98, %w_blk6_dw : tensor<1x240x30x30xf16>, tensor<240x240x3x3xf16>)
    outs(%fill100 : tensor<1x240x14x14xf16>) -> tensor<1x240x14x14xf16>
  %empty102 = tensor.empty() : tensor<1x240x14x14xf16>
  %relu103 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv101 : tensor<1x240x14x14xf16>)
    outs(%empty102 : tensor<1x240x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x240x14x14xf16>
  %init104 = tensor.empty() : tensor<1x80x14x14xf16>
  %fill105 = linalg.fill ins(%cst : f16) outs(%init104 : tensor<1x80x14x14xf16>) -> tensor<1x80x14x14xf16>
  %conv106 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu103, %w_blk6_proj : tensor<1x240x14x14xf16>, tensor<80x240x1x1xf16>)
    outs(%fill105 : tensor<1x80x14x14xf16>) -> tensor<1x80x14x14xf16>

  // IRB 7: 80->80 mid=160 s=1 14x14
  %init107 = tensor.empty() : tensor<1x160x14x14xf16>
  %fill108 = linalg.fill ins(%cst : f16) outs(%init107 : tensor<1x160x14x14xf16>) -> tensor<1x160x14x14xf16>
  %conv109 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv106, %w_blk7_exp : tensor<1x80x14x14xf16>, tensor<160x80x1x1xf16>)
    outs(%fill108 : tensor<1x160x14x14xf16>) -> tensor<1x160x14x14xf16>
  %empty110 = tensor.empty() : tensor<1x160x14x14xf16>
  %relu111 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv109 : tensor<1x160x14x14xf16>)
    outs(%empty110 : tensor<1x160x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x160x14x14xf16>
  %pad112 = tensor.pad %relu111 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x160x14x14xf16> to tensor<1x160x16x16xf16>
  %init113 = tensor.empty() : tensor<1x160x14x14xf16>
  %fill114 = linalg.fill ins(%cst : f16) outs(%init113 : tensor<1x160x14x14xf16>) -> tensor<1x160x14x14xf16>
  %conv115 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad112, %w_blk7_dw : tensor<1x160x16x16xf16>, tensor<160x160x3x3xf16>)
    outs(%fill114 : tensor<1x160x14x14xf16>) -> tensor<1x160x14x14xf16>
  %empty116 = tensor.empty() : tensor<1x160x14x14xf16>
  %relu117 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv115 : tensor<1x160x14x14xf16>)
    outs(%empty116 : tensor<1x160x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x160x14x14xf16>
  %init118 = tensor.empty() : tensor<1x80x14x14xf16>
  %fill119 = linalg.fill ins(%cst : f16) outs(%init118 : tensor<1x80x14x14xf16>) -> tensor<1x80x14x14xf16>
  %conv120 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu117, %w_blk7_proj : tensor<1x160x14x14xf16>, tensor<80x160x1x1xf16>)
    outs(%fill119 : tensor<1x80x14x14xf16>) -> tensor<1x80x14x14xf16>
  %empty121 = tensor.empty() : tensor<1x80x14x14xf16>
  %add122 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv120, %conv106 : tensor<1x80x14x14xf16>, tensor<1x80x14x14xf16>)
    outs(%empty121 : tensor<1x80x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x80x14x14xf16>

  // IRB 8: 80->80 mid=160 s=1 14x14
  %init123 = tensor.empty() : tensor<1x160x14x14xf16>
  %fill124 = linalg.fill ins(%cst : f16) outs(%init123 : tensor<1x160x14x14xf16>) -> tensor<1x160x14x14xf16>
  %conv125 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add122, %w_blk8_exp : tensor<1x80x14x14xf16>, tensor<160x80x1x1xf16>)
    outs(%fill124 : tensor<1x160x14x14xf16>) -> tensor<1x160x14x14xf16>
  %empty126 = tensor.empty() : tensor<1x160x14x14xf16>
  %relu127 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv125 : tensor<1x160x14x14xf16>)
    outs(%empty126 : tensor<1x160x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x160x14x14xf16>
  %pad128 = tensor.pad %relu127 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x160x14x14xf16> to tensor<1x160x16x16xf16>
  %init129 = tensor.empty() : tensor<1x160x14x14xf16>
  %fill130 = linalg.fill ins(%cst : f16) outs(%init129 : tensor<1x160x14x14xf16>) -> tensor<1x160x14x14xf16>
  %conv131 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad128, %w_blk8_dw : tensor<1x160x16x16xf16>, tensor<160x160x3x3xf16>)
    outs(%fill130 : tensor<1x160x14x14xf16>) -> tensor<1x160x14x14xf16>
  %empty132 = tensor.empty() : tensor<1x160x14x14xf16>
  %relu133 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv131 : tensor<1x160x14x14xf16>)
    outs(%empty132 : tensor<1x160x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x160x14x14xf16>
  %init134 = tensor.empty() : tensor<1x80x14x14xf16>
  %fill135 = linalg.fill ins(%cst : f16) outs(%init134 : tensor<1x80x14x14xf16>) -> tensor<1x80x14x14xf16>
  %conv136 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu133, %w_blk8_proj : tensor<1x160x14x14xf16>, tensor<80x160x1x1xf16>)
    outs(%fill135 : tensor<1x80x14x14xf16>) -> tensor<1x80x14x14xf16>
  %empty137 = tensor.empty() : tensor<1x80x14x14xf16>
  %add138 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv136, %add122 : tensor<1x80x14x14xf16>, tensor<1x80x14x14xf16>)
    outs(%empty137 : tensor<1x80x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x80x14x14xf16>

  // IRB 9: 80->112 mid=480 s=1 14x14
  %init139 = tensor.empty() : tensor<1x480x14x14xf16>
  %fill140 = linalg.fill ins(%cst : f16) outs(%init139 : tensor<1x480x14x14xf16>) -> tensor<1x480x14x14xf16>
  %conv141 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add138, %w_blk9_exp : tensor<1x80x14x14xf16>, tensor<480x80x1x1xf16>)
    outs(%fill140 : tensor<1x480x14x14xf16>) -> tensor<1x480x14x14xf16>
  %empty142 = tensor.empty() : tensor<1x480x14x14xf16>
  %relu143 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv141 : tensor<1x480x14x14xf16>)
    outs(%empty142 : tensor<1x480x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x480x14x14xf16>
  %pad144 = tensor.pad %relu143 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x480x14x14xf16> to tensor<1x480x16x16xf16>
  %init145 = tensor.empty() : tensor<1x480x14x14xf16>
  %fill146 = linalg.fill ins(%cst : f16) outs(%init145 : tensor<1x480x14x14xf16>) -> tensor<1x480x14x14xf16>
  %conv147 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad144, %w_blk9_dw : tensor<1x480x16x16xf16>, tensor<480x480x3x3xf16>)
    outs(%fill146 : tensor<1x480x14x14xf16>) -> tensor<1x480x14x14xf16>
  %empty148 = tensor.empty() : tensor<1x480x14x14xf16>
  %relu149 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv147 : tensor<1x480x14x14xf16>)
    outs(%empty148 : tensor<1x480x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x480x14x14xf16>
  %init150 = tensor.empty() : tensor<1x112x14x14xf16>
  %fill151 = linalg.fill ins(%cst : f16) outs(%init150 : tensor<1x112x14x14xf16>) -> tensor<1x112x14x14xf16>
  %conv152 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu149, %w_blk9_proj : tensor<1x480x14x14xf16>, tensor<112x480x1x1xf16>)
    outs(%fill151 : tensor<1x112x14x14xf16>) -> tensor<1x112x14x14xf16>

  // IRB 10: 112->112 mid=672 s=1 14x14
  %init153 = tensor.empty() : tensor<1x672x14x14xf16>
  %fill154 = linalg.fill ins(%cst : f16) outs(%init153 : tensor<1x672x14x14xf16>) -> tensor<1x672x14x14xf16>
  %conv155 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv152, %w_blk10_exp : tensor<1x112x14x14xf16>, tensor<672x112x1x1xf16>)
    outs(%fill154 : tensor<1x672x14x14xf16>) -> tensor<1x672x14x14xf16>
  %empty156 = tensor.empty() : tensor<1x672x14x14xf16>
  %relu157 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv155 : tensor<1x672x14x14xf16>)
    outs(%empty156 : tensor<1x672x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x14x14xf16>
  %pad158 = tensor.pad %relu157 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x672x14x14xf16> to tensor<1x672x16x16xf16>
  %init159 = tensor.empty() : tensor<1x672x14x14xf16>
  %fill160 = linalg.fill ins(%cst : f16) outs(%init159 : tensor<1x672x14x14xf16>) -> tensor<1x672x14x14xf16>
  %conv161 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad158, %w_blk10_dw : tensor<1x672x16x16xf16>, tensor<672x672x3x3xf16>)
    outs(%fill160 : tensor<1x672x14x14xf16>) -> tensor<1x672x14x14xf16>
  %empty162 = tensor.empty() : tensor<1x672x14x14xf16>
  %relu163 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv161 : tensor<1x672x14x14xf16>)
    outs(%empty162 : tensor<1x672x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x14x14xf16>
  %init164 = tensor.empty() : tensor<1x112x14x14xf16>
  %fill165 = linalg.fill ins(%cst : f16) outs(%init164 : tensor<1x112x14x14xf16>) -> tensor<1x112x14x14xf16>
  %conv166 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu163, %w_blk10_proj : tensor<1x672x14x14xf16>, tensor<112x672x1x1xf16>)
    outs(%fill165 : tensor<1x112x14x14xf16>) -> tensor<1x112x14x14xf16>
  %empty167 = tensor.empty() : tensor<1x112x14x14xf16>
  %add168 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv166, %conv152 : tensor<1x112x14x14xf16>, tensor<1x112x14x14xf16>)
    outs(%empty167 : tensor<1x112x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x112x14x14xf16>

  // IRB 11: 112->160 mid=672 s=2 14x14
  %init169 = tensor.empty() : tensor<1x672x14x14xf16>
  %fill170 = linalg.fill ins(%cst : f16) outs(%init169 : tensor<1x672x14x14xf16>) -> tensor<1x672x14x14xf16>
  %conv171 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add168, %w_blk11_exp : tensor<1x112x14x14xf16>, tensor<672x112x1x1xf16>)
    outs(%fill170 : tensor<1x672x14x14xf16>) -> tensor<1x672x14x14xf16>
  %empty172 = tensor.empty() : tensor<1x672x14x14xf16>
  %relu173 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv171 : tensor<1x672x14x14xf16>)
    outs(%empty172 : tensor<1x672x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x14x14xf16>
  %pad174 = tensor.pad %relu173 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x672x14x14xf16> to tensor<1x672x16x16xf16>
  %init175 = tensor.empty() : tensor<1x672x7x7xf16>
  %fill176 = linalg.fill ins(%cst : f16) outs(%init175 : tensor<1x672x7x7xf16>) -> tensor<1x672x7x7xf16>
  %conv177 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad174, %w_blk11_dw : tensor<1x672x16x16xf16>, tensor<672x672x3x3xf16>)
    outs(%fill176 : tensor<1x672x7x7xf16>) -> tensor<1x672x7x7xf16>
  %empty178 = tensor.empty() : tensor<1x672x7x7xf16>
  %relu179 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv177 : tensor<1x672x7x7xf16>)
    outs(%empty178 : tensor<1x672x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x7x7xf16>
  %init180 = tensor.empty() : tensor<1x160x7x7xf16>
  %fill181 = linalg.fill ins(%cst : f16) outs(%init180 : tensor<1x160x7x7xf16>) -> tensor<1x160x7x7xf16>
  %conv182 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu179, %w_blk11_proj : tensor<1x672x7x7xf16>, tensor<160x672x1x1xf16>)
    outs(%fill181 : tensor<1x160x7x7xf16>) -> tensor<1x160x7x7xf16>

  // IRB 12: 160->160 mid=960 s=1 7x7
  %init183 = tensor.empty() : tensor<1x960x7x7xf16>
  %fill184 = linalg.fill ins(%cst : f16) outs(%init183 : tensor<1x960x7x7xf16>) -> tensor<1x960x7x7xf16>
  %conv185 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv182, %w_blk12_exp : tensor<1x160x7x7xf16>, tensor<960x160x1x1xf16>)
    outs(%fill184 : tensor<1x960x7x7xf16>) -> tensor<1x960x7x7xf16>
  %empty186 = tensor.empty() : tensor<1x960x7x7xf16>
  %relu187 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv185 : tensor<1x960x7x7xf16>)
    outs(%empty186 : tensor<1x960x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x960x7x7xf16>
  %pad188 = tensor.pad %relu187 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x960x7x7xf16> to tensor<1x960x9x9xf16>
  %init189 = tensor.empty() : tensor<1x960x7x7xf16>
  %fill190 = linalg.fill ins(%cst : f16) outs(%init189 : tensor<1x960x7x7xf16>) -> tensor<1x960x7x7xf16>
  %conv191 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad188, %w_blk12_dw : tensor<1x960x9x9xf16>, tensor<960x960x3x3xf16>)
    outs(%fill190 : tensor<1x960x7x7xf16>) -> tensor<1x960x7x7xf16>
  %empty192 = tensor.empty() : tensor<1x960x7x7xf16>
  %relu193 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv191 : tensor<1x960x7x7xf16>)
    outs(%empty192 : tensor<1x960x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x960x7x7xf16>
  %init194 = tensor.empty() : tensor<1x160x7x7xf16>
  %fill195 = linalg.fill ins(%cst : f16) outs(%init194 : tensor<1x160x7x7xf16>) -> tensor<1x160x7x7xf16>
  %conv196 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu193, %w_blk12_proj : tensor<1x960x7x7xf16>, tensor<160x960x1x1xf16>)
    outs(%fill195 : tensor<1x160x7x7xf16>) -> tensor<1x160x7x7xf16>
  %empty197 = tensor.empty() : tensor<1x160x7x7xf16>
  %add198 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv196, %conv182 : tensor<1x160x7x7xf16>, tensor<1x160x7x7xf16>)
    outs(%empty197 : tensor<1x160x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x160x7x7xf16>

  // IRB 13: 160->160 mid=960 s=1 7x7
  %init199 = tensor.empty() : tensor<1x960x7x7xf16>
  %fill200 = linalg.fill ins(%cst : f16) outs(%init199 : tensor<1x960x7x7xf16>) -> tensor<1x960x7x7xf16>
  %conv201 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add198, %w_blk13_exp : tensor<1x160x7x7xf16>, tensor<960x160x1x1xf16>)
    outs(%fill200 : tensor<1x960x7x7xf16>) -> tensor<1x960x7x7xf16>
  %empty202 = tensor.empty() : tensor<1x960x7x7xf16>
  %relu203 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv201 : tensor<1x960x7x7xf16>)
    outs(%empty202 : tensor<1x960x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x960x7x7xf16>
  %pad204 = tensor.pad %relu203 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x960x7x7xf16> to tensor<1x960x9x9xf16>
  %init205 = tensor.empty() : tensor<1x960x7x7xf16>
  %fill206 = linalg.fill ins(%cst : f16) outs(%init205 : tensor<1x960x7x7xf16>) -> tensor<1x960x7x7xf16>
  %conv207 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad204, %w_blk13_dw : tensor<1x960x9x9xf16>, tensor<960x960x3x3xf16>)
    outs(%fill206 : tensor<1x960x7x7xf16>) -> tensor<1x960x7x7xf16>
  %empty208 = tensor.empty() : tensor<1x960x7x7xf16>
  %relu209 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv207 : tensor<1x960x7x7xf16>)
    outs(%empty208 : tensor<1x960x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x960x7x7xf16>
  %init210 = tensor.empty() : tensor<1x160x7x7xf16>
  %fill211 = linalg.fill ins(%cst : f16) outs(%init210 : tensor<1x160x7x7xf16>) -> tensor<1x160x7x7xf16>
  %conv212 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu209, %w_blk13_proj : tensor<1x960x7x7xf16>, tensor<160x960x1x1xf16>)
    outs(%fill211 : tensor<1x160x7x7xf16>) -> tensor<1x160x7x7xf16>
  %empty213 = tensor.empty() : tensor<1x160x7x7xf16>
  %add214 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv212, %add198 : tensor<1x160x7x7xf16>, tensor<1x160x7x7xf16>)
    outs(%empty213 : tensor<1x160x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x160x7x7xf16>

  // Final 1x1: 160->960
  %init215 = tensor.empty() : tensor<1x960x7x7xf16>
  %fill216 = linalg.fill ins(%cst : f16) outs(%init215 : tensor<1x960x7x7xf16>) -> tensor<1x960x7x7xf16>
  %conv217 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add214, %w_final : tensor<1x160x7x7xf16>, tensor<960x160x1x1xf16>)
    outs(%fill216 : tensor<1x960x7x7xf16>) -> tensor<1x960x7x7xf16>
  %empty218 = tensor.empty() : tensor<1x960x7x7xf16>
  %relu219 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv217 : tensor<1x960x7x7xf16>)
    outs(%empty218 : tensor<1x960x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x960x7x7xf16>

  // FC: 1x1 960->1000
  %init220 = tensor.empty() : tensor<1x1000x7x7xf16>
  %fill221 = linalg.fill ins(%cst : f16) outs(%init220 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  %conv222 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu219, %w_fc : tensor<1x960x7x7xf16>, tensor<1000x960x1x1xf16>)
    outs(%fill221 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  return %conv222 : tensor<1x1000x7x7xf16>
}
