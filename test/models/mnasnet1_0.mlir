func.func @mnasnet1_0(
    %input: tensor<1x3x224x224xf16>,
    %w_stem: tensor<32x3x3x3xf16>,
    %w_blk0_dw: tensor<32x32x3x3xf16>,
    %w_blk0_proj: tensor<16x32x1x1xf16>,
    %w_blk1_exp: tensor<48x16x1x1xf16>,
    %w_blk1_dw: tensor<48x48x3x3xf16>,
    %w_blk1_proj: tensor<24x48x1x1xf16>,
    %w_blk2_exp: tensor<72x24x1x1xf16>,
    %w_blk2_dw: tensor<72x72x3x3xf16>,
    %w_blk2_proj: tensor<24x72x1x1xf16>,
    %w_blk3_exp: tensor<72x24x1x1xf16>,
    %w_blk3_dw: tensor<72x72x3x3xf16>,
    %w_blk3_proj: tensor<24x72x1x1xf16>,
    %w_blk4_exp: tensor<72x24x1x1xf16>,
    %w_blk4_dw: tensor<72x72x3x3xf16>,
    %w_blk4_proj: tensor<40x72x1x1xf16>,
    %w_blk5_exp: tensor<120x40x1x1xf16>,
    %w_blk5_dw: tensor<120x120x3x3xf16>,
    %w_blk5_proj: tensor<40x120x1x1xf16>,
    %w_blk6_exp: tensor<120x40x1x1xf16>,
    %w_blk6_dw: tensor<120x120x3x3xf16>,
    %w_blk6_proj: tensor<40x120x1x1xf16>,
    %w_blk7_exp: tensor<240x40x1x1xf16>,
    %w_blk7_dw: tensor<240x240x3x3xf16>,
    %w_blk7_proj: tensor<80x240x1x1xf16>,
    %w_blk8_exp: tensor<480x80x1x1xf16>,
    %w_blk8_dw: tensor<480x480x3x3xf16>,
    %w_blk8_proj: tensor<80x480x1x1xf16>,
    %w_blk9_exp: tensor<480x80x1x1xf16>,
    %w_blk9_dw: tensor<480x480x3x3xf16>,
    %w_blk9_proj: tensor<80x480x1x1xf16>,
    %w_blk10_exp: tensor<480x80x1x1xf16>,
    %w_blk10_dw: tensor<480x480x3x3xf16>,
    %w_blk10_proj: tensor<96x480x1x1xf16>,
    %w_blk11_exp: tensor<576x96x1x1xf16>,
    %w_blk11_dw: tensor<576x576x3x3xf16>,
    %w_blk11_proj: tensor<96x576x1x1xf16>,
    %w_blk12_exp: tensor<576x96x1x1xf16>,
    %w_blk12_dw: tensor<576x576x3x3xf16>,
    %w_blk12_proj: tensor<192x576x1x1xf16>,
    %w_blk13_exp: tensor<1152x192x1x1xf16>,
    %w_blk13_dw: tensor<1152x1152x3x3xf16>,
    %w_blk13_proj: tensor<192x1152x1x1xf16>,
    %w_blk14_exp: tensor<1152x192x1x1xf16>,
    %w_blk14_dw: tensor<1152x1152x3x3xf16>,
    %w_blk14_proj: tensor<192x1152x1x1xf16>,
    %w_blk15_exp: tensor<1152x192x1x1xf16>,
    %w_blk15_dw: tensor<1152x1152x3x3xf16>,
    %w_blk15_proj: tensor<192x1152x1x1xf16>,
    %w_blk16_exp: tensor<1152x192x1x1xf16>,
    %w_blk16_dw: tensor<1152x1152x3x3xf16>,
    %w_blk16_proj: tensor<320x1152x1x1xf16>,
    %w_final: tensor<1280x320x1x1xf16>,
    %w_fc: tensor<1000x1280x1x1xf16>) -> tensor<1x1000x7x7xf16> {
  %cst = arith.constant 0.0 : f16

  // Stem: 3x3 stride 2, 3->32
  %pad0 = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x224x224xf16> to tensor<1x3x226x226xf16>
  %init1 = tensor.empty() : tensor<1x32x112x112xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad0, %w_stem : tensor<1x3x226x226xf16>, tensor<32x3x3x3xf16>)
    outs(%fill2 : tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16>
  %empty4 = tensor.empty() : tensor<1x32x112x112xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x32x112x112xf16>)
    outs(%empty4 : tensor<1x32x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x112x112xf16>

  // IRB 0: 32->16 mid=32 s=1 112x112
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x32x112x112xf16> to tensor<1x32x114x114xf16>
  %init7 = tensor.empty() : tensor<1x32x112x112xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad6, %w_blk0_dw : tensor<1x32x114x114xf16>, tensor<32x32x3x3xf16>)
    outs(%fill8 : tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16>
  %empty10 = tensor.empty() : tensor<1x32x112x112xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x32x112x112xf16>)
    outs(%empty10 : tensor<1x32x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x112x112xf16>
  %init12 = tensor.empty() : tensor<1x16x112x112xf16>
  %fill13 = linalg.fill ins(%cst : f16) outs(%init12 : tensor<1x16x112x112xf16>) -> tensor<1x16x112x112xf16>
  %conv14 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu11, %w_blk0_proj : tensor<1x32x112x112xf16>, tensor<16x32x1x1xf16>)
    outs(%fill13 : tensor<1x16x112x112xf16>) -> tensor<1x16x112x112xf16>

  // IRB 1: 16->24 mid=48 s=2 112x112
  %init15 = tensor.empty() : tensor<1x48x112x112xf16>
  %fill16 = linalg.fill ins(%cst : f16) outs(%init15 : tensor<1x48x112x112xf16>) -> tensor<1x48x112x112xf16>
  %conv17 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv14, %w_blk1_exp : tensor<1x16x112x112xf16>, tensor<48x16x1x1xf16>)
    outs(%fill16 : tensor<1x48x112x112xf16>) -> tensor<1x48x112x112xf16>
  %empty18 = tensor.empty() : tensor<1x48x112x112xf16>
  %relu19 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv17 : tensor<1x48x112x112xf16>)
    outs(%empty18 : tensor<1x48x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x48x112x112xf16>
  %pad20 = tensor.pad %relu19 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x48x112x112xf16> to tensor<1x48x114x114xf16>
  %init21 = tensor.empty() : tensor<1x48x56x56xf16>
  %fill22 = linalg.fill ins(%cst : f16) outs(%init21 : tensor<1x48x56x56xf16>) -> tensor<1x48x56x56xf16>
  %conv23 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad20, %w_blk1_dw : tensor<1x48x114x114xf16>, tensor<48x48x3x3xf16>)
    outs(%fill22 : tensor<1x48x56x56xf16>) -> tensor<1x48x56x56xf16>
  %empty24 = tensor.empty() : tensor<1x48x56x56xf16>
  %relu25 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv23 : tensor<1x48x56x56xf16>)
    outs(%empty24 : tensor<1x48x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x48x56x56xf16>
  %init26 = tensor.empty() : tensor<1x24x56x56xf16>
  %fill27 = linalg.fill ins(%cst : f16) outs(%init26 : tensor<1x24x56x56xf16>) -> tensor<1x24x56x56xf16>
  %conv28 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu25, %w_blk1_proj : tensor<1x48x56x56xf16>, tensor<24x48x1x1xf16>)
    outs(%fill27 : tensor<1x24x56x56xf16>) -> tensor<1x24x56x56xf16>

  // IRB 2: 24->24 mid=72 s=1 56x56
  %init29 = tensor.empty() : tensor<1x72x56x56xf16>
  %fill30 = linalg.fill ins(%cst : f16) outs(%init29 : tensor<1x72x56x56xf16>) -> tensor<1x72x56x56xf16>
  %conv31 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv28, %w_blk2_exp : tensor<1x24x56x56xf16>, tensor<72x24x1x1xf16>)
    outs(%fill30 : tensor<1x72x56x56xf16>) -> tensor<1x72x56x56xf16>
  %empty32 = tensor.empty() : tensor<1x72x56x56xf16>
  %relu33 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv31 : tensor<1x72x56x56xf16>)
    outs(%empty32 : tensor<1x72x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x56x56xf16>
  %pad34 = tensor.pad %relu33 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x72x56x56xf16> to tensor<1x72x58x58xf16>
  %init35 = tensor.empty() : tensor<1x72x56x56xf16>
  %fill36 = linalg.fill ins(%cst : f16) outs(%init35 : tensor<1x72x56x56xf16>) -> tensor<1x72x56x56xf16>
  %conv37 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad34, %w_blk2_dw : tensor<1x72x58x58xf16>, tensor<72x72x3x3xf16>)
    outs(%fill36 : tensor<1x72x56x56xf16>) -> tensor<1x72x56x56xf16>
  %empty38 = tensor.empty() : tensor<1x72x56x56xf16>
  %relu39 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv37 : tensor<1x72x56x56xf16>)
    outs(%empty38 : tensor<1x72x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x56x56xf16>
  %init40 = tensor.empty() : tensor<1x24x56x56xf16>
  %fill41 = linalg.fill ins(%cst : f16) outs(%init40 : tensor<1x24x56x56xf16>) -> tensor<1x24x56x56xf16>
  %conv42 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu39, %w_blk2_proj : tensor<1x72x56x56xf16>, tensor<24x72x1x1xf16>)
    outs(%fill41 : tensor<1x24x56x56xf16>) -> tensor<1x24x56x56xf16>
  %empty43 = tensor.empty() : tensor<1x24x56x56xf16>
  %add44 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv42, %conv28 : tensor<1x24x56x56xf16>, tensor<1x24x56x56xf16>)
    outs(%empty43 : tensor<1x24x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x24x56x56xf16>

  // IRB 3: 24->24 mid=72 s=1 56x56
  %init45 = tensor.empty() : tensor<1x72x56x56xf16>
  %fill46 = linalg.fill ins(%cst : f16) outs(%init45 : tensor<1x72x56x56xf16>) -> tensor<1x72x56x56xf16>
  %conv47 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add44, %w_blk3_exp : tensor<1x24x56x56xf16>, tensor<72x24x1x1xf16>)
    outs(%fill46 : tensor<1x72x56x56xf16>) -> tensor<1x72x56x56xf16>
  %empty48 = tensor.empty() : tensor<1x72x56x56xf16>
  %relu49 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv47 : tensor<1x72x56x56xf16>)
    outs(%empty48 : tensor<1x72x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x56x56xf16>
  %pad50 = tensor.pad %relu49 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x72x56x56xf16> to tensor<1x72x58x58xf16>
  %init51 = tensor.empty() : tensor<1x72x56x56xf16>
  %fill52 = linalg.fill ins(%cst : f16) outs(%init51 : tensor<1x72x56x56xf16>) -> tensor<1x72x56x56xf16>
  %conv53 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad50, %w_blk3_dw : tensor<1x72x58x58xf16>, tensor<72x72x3x3xf16>)
    outs(%fill52 : tensor<1x72x56x56xf16>) -> tensor<1x72x56x56xf16>
  %empty54 = tensor.empty() : tensor<1x72x56x56xf16>
  %relu55 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv53 : tensor<1x72x56x56xf16>)
    outs(%empty54 : tensor<1x72x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x56x56xf16>
  %init56 = tensor.empty() : tensor<1x24x56x56xf16>
  %fill57 = linalg.fill ins(%cst : f16) outs(%init56 : tensor<1x24x56x56xf16>) -> tensor<1x24x56x56xf16>
  %conv58 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu55, %w_blk3_proj : tensor<1x72x56x56xf16>, tensor<24x72x1x1xf16>)
    outs(%fill57 : tensor<1x24x56x56xf16>) -> tensor<1x24x56x56xf16>
  %empty59 = tensor.empty() : tensor<1x24x56x56xf16>
  %add60 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv58, %add44 : tensor<1x24x56x56xf16>, tensor<1x24x56x56xf16>)
    outs(%empty59 : tensor<1x24x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x24x56x56xf16>

  // IRB 4: 24->40 mid=72 s=2 56x56
  %init61 = tensor.empty() : tensor<1x72x56x56xf16>
  %fill62 = linalg.fill ins(%cst : f16) outs(%init61 : tensor<1x72x56x56xf16>) -> tensor<1x72x56x56xf16>
  %conv63 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add60, %w_blk4_exp : tensor<1x24x56x56xf16>, tensor<72x24x1x1xf16>)
    outs(%fill62 : tensor<1x72x56x56xf16>) -> tensor<1x72x56x56xf16>
  %empty64 = tensor.empty() : tensor<1x72x56x56xf16>
  %relu65 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv63 : tensor<1x72x56x56xf16>)
    outs(%empty64 : tensor<1x72x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x56x56xf16>
  %pad66 = tensor.pad %relu65 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x72x56x56xf16> to tensor<1x72x58x58xf16>
  %init67 = tensor.empty() : tensor<1x72x28x28xf16>
  %fill68 = linalg.fill ins(%cst : f16) outs(%init67 : tensor<1x72x28x28xf16>) -> tensor<1x72x28x28xf16>
  %conv69 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad66, %w_blk4_dw : tensor<1x72x58x58xf16>, tensor<72x72x3x3xf16>)
    outs(%fill68 : tensor<1x72x28x28xf16>) -> tensor<1x72x28x28xf16>
  %empty70 = tensor.empty() : tensor<1x72x28x28xf16>
  %relu71 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv69 : tensor<1x72x28x28xf16>)
    outs(%empty70 : tensor<1x72x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x28x28xf16>
  %init72 = tensor.empty() : tensor<1x40x28x28xf16>
  %fill73 = linalg.fill ins(%cst : f16) outs(%init72 : tensor<1x40x28x28xf16>) -> tensor<1x40x28x28xf16>
  %conv74 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu71, %w_blk4_proj : tensor<1x72x28x28xf16>, tensor<40x72x1x1xf16>)
    outs(%fill73 : tensor<1x40x28x28xf16>) -> tensor<1x40x28x28xf16>

  // IRB 5: 40->40 mid=120 s=1 28x28
  %init75 = tensor.empty() : tensor<1x120x28x28xf16>
  %fill76 = linalg.fill ins(%cst : f16) outs(%init75 : tensor<1x120x28x28xf16>) -> tensor<1x120x28x28xf16>
  %conv77 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv74, %w_blk5_exp : tensor<1x40x28x28xf16>, tensor<120x40x1x1xf16>)
    outs(%fill76 : tensor<1x120x28x28xf16>) -> tensor<1x120x28x28xf16>
  %empty78 = tensor.empty() : tensor<1x120x28x28xf16>
  %relu79 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv77 : tensor<1x120x28x28xf16>)
    outs(%empty78 : tensor<1x120x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x120x28x28xf16>
  %pad80 = tensor.pad %relu79 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x120x28x28xf16> to tensor<1x120x30x30xf16>
  %init81 = tensor.empty() : tensor<1x120x28x28xf16>
  %fill82 = linalg.fill ins(%cst : f16) outs(%init81 : tensor<1x120x28x28xf16>) -> tensor<1x120x28x28xf16>
  %conv83 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad80, %w_blk5_dw : tensor<1x120x30x30xf16>, tensor<120x120x3x3xf16>)
    outs(%fill82 : tensor<1x120x28x28xf16>) -> tensor<1x120x28x28xf16>
  %empty84 = tensor.empty() : tensor<1x120x28x28xf16>
  %relu85 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv83 : tensor<1x120x28x28xf16>)
    outs(%empty84 : tensor<1x120x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x120x28x28xf16>
  %init86 = tensor.empty() : tensor<1x40x28x28xf16>
  %fill87 = linalg.fill ins(%cst : f16) outs(%init86 : tensor<1x40x28x28xf16>) -> tensor<1x40x28x28xf16>
  %conv88 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu85, %w_blk5_proj : tensor<1x120x28x28xf16>, tensor<40x120x1x1xf16>)
    outs(%fill87 : tensor<1x40x28x28xf16>) -> tensor<1x40x28x28xf16>
  %empty89 = tensor.empty() : tensor<1x40x28x28xf16>
  %add90 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv88, %conv74 : tensor<1x40x28x28xf16>, tensor<1x40x28x28xf16>)
    outs(%empty89 : tensor<1x40x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x40x28x28xf16>

  // IRB 6: 40->40 mid=120 s=1 28x28
  %init91 = tensor.empty() : tensor<1x120x28x28xf16>
  %fill92 = linalg.fill ins(%cst : f16) outs(%init91 : tensor<1x120x28x28xf16>) -> tensor<1x120x28x28xf16>
  %conv93 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add90, %w_blk6_exp : tensor<1x40x28x28xf16>, tensor<120x40x1x1xf16>)
    outs(%fill92 : tensor<1x120x28x28xf16>) -> tensor<1x120x28x28xf16>
  %empty94 = tensor.empty() : tensor<1x120x28x28xf16>
  %relu95 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv93 : tensor<1x120x28x28xf16>)
    outs(%empty94 : tensor<1x120x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x120x28x28xf16>
  %pad96 = tensor.pad %relu95 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x120x28x28xf16> to tensor<1x120x30x30xf16>
  %init97 = tensor.empty() : tensor<1x120x28x28xf16>
  %fill98 = linalg.fill ins(%cst : f16) outs(%init97 : tensor<1x120x28x28xf16>) -> tensor<1x120x28x28xf16>
  %conv99 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad96, %w_blk6_dw : tensor<1x120x30x30xf16>, tensor<120x120x3x3xf16>)
    outs(%fill98 : tensor<1x120x28x28xf16>) -> tensor<1x120x28x28xf16>
  %empty100 = tensor.empty() : tensor<1x120x28x28xf16>
  %relu101 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv99 : tensor<1x120x28x28xf16>)
    outs(%empty100 : tensor<1x120x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x120x28x28xf16>
  %init102 = tensor.empty() : tensor<1x40x28x28xf16>
  %fill103 = linalg.fill ins(%cst : f16) outs(%init102 : tensor<1x40x28x28xf16>) -> tensor<1x40x28x28xf16>
  %conv104 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu101, %w_blk6_proj : tensor<1x120x28x28xf16>, tensor<40x120x1x1xf16>)
    outs(%fill103 : tensor<1x40x28x28xf16>) -> tensor<1x40x28x28xf16>
  %empty105 = tensor.empty() : tensor<1x40x28x28xf16>
  %add106 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv104, %add90 : tensor<1x40x28x28xf16>, tensor<1x40x28x28xf16>)
    outs(%empty105 : tensor<1x40x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x40x28x28xf16>

  // IRB 7: 40->80 mid=240 s=2 28x28
  %init107 = tensor.empty() : tensor<1x240x28x28xf16>
  %fill108 = linalg.fill ins(%cst : f16) outs(%init107 : tensor<1x240x28x28xf16>) -> tensor<1x240x28x28xf16>
  %conv109 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add106, %w_blk7_exp : tensor<1x40x28x28xf16>, tensor<240x40x1x1xf16>)
    outs(%fill108 : tensor<1x240x28x28xf16>) -> tensor<1x240x28x28xf16>
  %empty110 = tensor.empty() : tensor<1x240x28x28xf16>
  %relu111 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv109 : tensor<1x240x28x28xf16>)
    outs(%empty110 : tensor<1x240x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x240x28x28xf16>
  %pad112 = tensor.pad %relu111 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x240x28x28xf16> to tensor<1x240x30x30xf16>
  %init113 = tensor.empty() : tensor<1x240x14x14xf16>
  %fill114 = linalg.fill ins(%cst : f16) outs(%init113 : tensor<1x240x14x14xf16>) -> tensor<1x240x14x14xf16>
  %conv115 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad112, %w_blk7_dw : tensor<1x240x30x30xf16>, tensor<240x240x3x3xf16>)
    outs(%fill114 : tensor<1x240x14x14xf16>) -> tensor<1x240x14x14xf16>
  %empty116 = tensor.empty() : tensor<1x240x14x14xf16>
  %relu117 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv115 : tensor<1x240x14x14xf16>)
    outs(%empty116 : tensor<1x240x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x240x14x14xf16>
  %init118 = tensor.empty() : tensor<1x80x14x14xf16>
  %fill119 = linalg.fill ins(%cst : f16) outs(%init118 : tensor<1x80x14x14xf16>) -> tensor<1x80x14x14xf16>
  %conv120 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu117, %w_blk7_proj : tensor<1x240x14x14xf16>, tensor<80x240x1x1xf16>)
    outs(%fill119 : tensor<1x80x14x14xf16>) -> tensor<1x80x14x14xf16>

  // IRB 8: 80->80 mid=480 s=1 14x14
  %init121 = tensor.empty() : tensor<1x480x14x14xf16>
  %fill122 = linalg.fill ins(%cst : f16) outs(%init121 : tensor<1x480x14x14xf16>) -> tensor<1x480x14x14xf16>
  %conv123 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv120, %w_blk8_exp : tensor<1x80x14x14xf16>, tensor<480x80x1x1xf16>)
    outs(%fill122 : tensor<1x480x14x14xf16>) -> tensor<1x480x14x14xf16>
  %empty124 = tensor.empty() : tensor<1x480x14x14xf16>
  %relu125 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv123 : tensor<1x480x14x14xf16>)
    outs(%empty124 : tensor<1x480x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x480x14x14xf16>
  %pad126 = tensor.pad %relu125 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x480x14x14xf16> to tensor<1x480x16x16xf16>
  %init127 = tensor.empty() : tensor<1x480x14x14xf16>
  %fill128 = linalg.fill ins(%cst : f16) outs(%init127 : tensor<1x480x14x14xf16>) -> tensor<1x480x14x14xf16>
  %conv129 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad126, %w_blk8_dw : tensor<1x480x16x16xf16>, tensor<480x480x3x3xf16>)
    outs(%fill128 : tensor<1x480x14x14xf16>) -> tensor<1x480x14x14xf16>
  %empty130 = tensor.empty() : tensor<1x480x14x14xf16>
  %relu131 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv129 : tensor<1x480x14x14xf16>)
    outs(%empty130 : tensor<1x480x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x480x14x14xf16>
  %init132 = tensor.empty() : tensor<1x80x14x14xf16>
  %fill133 = linalg.fill ins(%cst : f16) outs(%init132 : tensor<1x80x14x14xf16>) -> tensor<1x80x14x14xf16>
  %conv134 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu131, %w_blk8_proj : tensor<1x480x14x14xf16>, tensor<80x480x1x1xf16>)
    outs(%fill133 : tensor<1x80x14x14xf16>) -> tensor<1x80x14x14xf16>
  %empty135 = tensor.empty() : tensor<1x80x14x14xf16>
  %add136 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv134, %conv120 : tensor<1x80x14x14xf16>, tensor<1x80x14x14xf16>)
    outs(%empty135 : tensor<1x80x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x80x14x14xf16>

  // IRB 9: 80->80 mid=480 s=1 14x14
  %init137 = tensor.empty() : tensor<1x480x14x14xf16>
  %fill138 = linalg.fill ins(%cst : f16) outs(%init137 : tensor<1x480x14x14xf16>) -> tensor<1x480x14x14xf16>
  %conv139 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add136, %w_blk9_exp : tensor<1x80x14x14xf16>, tensor<480x80x1x1xf16>)
    outs(%fill138 : tensor<1x480x14x14xf16>) -> tensor<1x480x14x14xf16>
  %empty140 = tensor.empty() : tensor<1x480x14x14xf16>
  %relu141 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv139 : tensor<1x480x14x14xf16>)
    outs(%empty140 : tensor<1x480x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x480x14x14xf16>
  %pad142 = tensor.pad %relu141 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x480x14x14xf16> to tensor<1x480x16x16xf16>
  %init143 = tensor.empty() : tensor<1x480x14x14xf16>
  %fill144 = linalg.fill ins(%cst : f16) outs(%init143 : tensor<1x480x14x14xf16>) -> tensor<1x480x14x14xf16>
  %conv145 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad142, %w_blk9_dw : tensor<1x480x16x16xf16>, tensor<480x480x3x3xf16>)
    outs(%fill144 : tensor<1x480x14x14xf16>) -> tensor<1x480x14x14xf16>
  %empty146 = tensor.empty() : tensor<1x480x14x14xf16>
  %relu147 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv145 : tensor<1x480x14x14xf16>)
    outs(%empty146 : tensor<1x480x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x480x14x14xf16>
  %init148 = tensor.empty() : tensor<1x80x14x14xf16>
  %fill149 = linalg.fill ins(%cst : f16) outs(%init148 : tensor<1x80x14x14xf16>) -> tensor<1x80x14x14xf16>
  %conv150 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu147, %w_blk9_proj : tensor<1x480x14x14xf16>, tensor<80x480x1x1xf16>)
    outs(%fill149 : tensor<1x80x14x14xf16>) -> tensor<1x80x14x14xf16>
  %empty151 = tensor.empty() : tensor<1x80x14x14xf16>
  %add152 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv150, %add136 : tensor<1x80x14x14xf16>, tensor<1x80x14x14xf16>)
    outs(%empty151 : tensor<1x80x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x80x14x14xf16>

  // IRB 10: 80->96 mid=480 s=1 14x14
  %init153 = tensor.empty() : tensor<1x480x14x14xf16>
  %fill154 = linalg.fill ins(%cst : f16) outs(%init153 : tensor<1x480x14x14xf16>) -> tensor<1x480x14x14xf16>
  %conv155 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add152, %w_blk10_exp : tensor<1x80x14x14xf16>, tensor<480x80x1x1xf16>)
    outs(%fill154 : tensor<1x480x14x14xf16>) -> tensor<1x480x14x14xf16>
  %empty156 = tensor.empty() : tensor<1x480x14x14xf16>
  %relu157 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv155 : tensor<1x480x14x14xf16>)
    outs(%empty156 : tensor<1x480x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x480x14x14xf16>
  %pad158 = tensor.pad %relu157 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x480x14x14xf16> to tensor<1x480x16x16xf16>
  %init159 = tensor.empty() : tensor<1x480x14x14xf16>
  %fill160 = linalg.fill ins(%cst : f16) outs(%init159 : tensor<1x480x14x14xf16>) -> tensor<1x480x14x14xf16>
  %conv161 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad158, %w_blk10_dw : tensor<1x480x16x16xf16>, tensor<480x480x3x3xf16>)
    outs(%fill160 : tensor<1x480x14x14xf16>) -> tensor<1x480x14x14xf16>
  %empty162 = tensor.empty() : tensor<1x480x14x14xf16>
  %relu163 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv161 : tensor<1x480x14x14xf16>)
    outs(%empty162 : tensor<1x480x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x480x14x14xf16>
  %init164 = tensor.empty() : tensor<1x96x14x14xf16>
  %fill165 = linalg.fill ins(%cst : f16) outs(%init164 : tensor<1x96x14x14xf16>) -> tensor<1x96x14x14xf16>
  %conv166 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu163, %w_blk10_proj : tensor<1x480x14x14xf16>, tensor<96x480x1x1xf16>)
    outs(%fill165 : tensor<1x96x14x14xf16>) -> tensor<1x96x14x14xf16>

  // IRB 11: 96->96 mid=576 s=1 14x14
  %init167 = tensor.empty() : tensor<1x576x14x14xf16>
  %fill168 = linalg.fill ins(%cst : f16) outs(%init167 : tensor<1x576x14x14xf16>) -> tensor<1x576x14x14xf16>
  %conv169 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv166, %w_blk11_exp : tensor<1x96x14x14xf16>, tensor<576x96x1x1xf16>)
    outs(%fill168 : tensor<1x576x14x14xf16>) -> tensor<1x576x14x14xf16>
  %empty170 = tensor.empty() : tensor<1x576x14x14xf16>
  %relu171 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv169 : tensor<1x576x14x14xf16>)
    outs(%empty170 : tensor<1x576x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x14x14xf16>
  %pad172 = tensor.pad %relu171 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x576x14x14xf16> to tensor<1x576x16x16xf16>
  %init173 = tensor.empty() : tensor<1x576x14x14xf16>
  %fill174 = linalg.fill ins(%cst : f16) outs(%init173 : tensor<1x576x14x14xf16>) -> tensor<1x576x14x14xf16>
  %conv175 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad172, %w_blk11_dw : tensor<1x576x16x16xf16>, tensor<576x576x3x3xf16>)
    outs(%fill174 : tensor<1x576x14x14xf16>) -> tensor<1x576x14x14xf16>
  %empty176 = tensor.empty() : tensor<1x576x14x14xf16>
  %relu177 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv175 : tensor<1x576x14x14xf16>)
    outs(%empty176 : tensor<1x576x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x14x14xf16>
  %init178 = tensor.empty() : tensor<1x96x14x14xf16>
  %fill179 = linalg.fill ins(%cst : f16) outs(%init178 : tensor<1x96x14x14xf16>) -> tensor<1x96x14x14xf16>
  %conv180 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu177, %w_blk11_proj : tensor<1x576x14x14xf16>, tensor<96x576x1x1xf16>)
    outs(%fill179 : tensor<1x96x14x14xf16>) -> tensor<1x96x14x14xf16>
  %empty181 = tensor.empty() : tensor<1x96x14x14xf16>
  %add182 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv180, %conv166 : tensor<1x96x14x14xf16>, tensor<1x96x14x14xf16>)
    outs(%empty181 : tensor<1x96x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x96x14x14xf16>

  // IRB 12: 96->192 mid=576 s=2 14x14
  %init183 = tensor.empty() : tensor<1x576x14x14xf16>
  %fill184 = linalg.fill ins(%cst : f16) outs(%init183 : tensor<1x576x14x14xf16>) -> tensor<1x576x14x14xf16>
  %conv185 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add182, %w_blk12_exp : tensor<1x96x14x14xf16>, tensor<576x96x1x1xf16>)
    outs(%fill184 : tensor<1x576x14x14xf16>) -> tensor<1x576x14x14xf16>
  %empty186 = tensor.empty() : tensor<1x576x14x14xf16>
  %relu187 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv185 : tensor<1x576x14x14xf16>)
    outs(%empty186 : tensor<1x576x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x14x14xf16>
  %pad188 = tensor.pad %relu187 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x576x14x14xf16> to tensor<1x576x16x16xf16>
  %init189 = tensor.empty() : tensor<1x576x7x7xf16>
  %fill190 = linalg.fill ins(%cst : f16) outs(%init189 : tensor<1x576x7x7xf16>) -> tensor<1x576x7x7xf16>
  %conv191 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad188, %w_blk12_dw : tensor<1x576x16x16xf16>, tensor<576x576x3x3xf16>)
    outs(%fill190 : tensor<1x576x7x7xf16>) -> tensor<1x576x7x7xf16>
  %empty192 = tensor.empty() : tensor<1x576x7x7xf16>
  %relu193 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv191 : tensor<1x576x7x7xf16>)
    outs(%empty192 : tensor<1x576x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x7x7xf16>
  %init194 = tensor.empty() : tensor<1x192x7x7xf16>
  %fill195 = linalg.fill ins(%cst : f16) outs(%init194 : tensor<1x192x7x7xf16>) -> tensor<1x192x7x7xf16>
  %conv196 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu193, %w_blk12_proj : tensor<1x576x7x7xf16>, tensor<192x576x1x1xf16>)
    outs(%fill195 : tensor<1x192x7x7xf16>) -> tensor<1x192x7x7xf16>

  // IRB 13: 192->192 mid=1152 s=1 7x7
  %init197 = tensor.empty() : tensor<1x1152x7x7xf16>
  %fill198 = linalg.fill ins(%cst : f16) outs(%init197 : tensor<1x1152x7x7xf16>) -> tensor<1x1152x7x7xf16>
  %conv199 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv196, %w_blk13_exp : tensor<1x192x7x7xf16>, tensor<1152x192x1x1xf16>)
    outs(%fill198 : tensor<1x1152x7x7xf16>) -> tensor<1x1152x7x7xf16>
  %empty200 = tensor.empty() : tensor<1x1152x7x7xf16>
  %relu201 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv199 : tensor<1x1152x7x7xf16>)
    outs(%empty200 : tensor<1x1152x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1152x7x7xf16>
  %pad202 = tensor.pad %relu201 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1152x7x7xf16> to tensor<1x1152x9x9xf16>
  %init203 = tensor.empty() : tensor<1x1152x7x7xf16>
  %fill204 = linalg.fill ins(%cst : f16) outs(%init203 : tensor<1x1152x7x7xf16>) -> tensor<1x1152x7x7xf16>
  %conv205 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad202, %w_blk13_dw : tensor<1x1152x9x9xf16>, tensor<1152x1152x3x3xf16>)
    outs(%fill204 : tensor<1x1152x7x7xf16>) -> tensor<1x1152x7x7xf16>
  %empty206 = tensor.empty() : tensor<1x1152x7x7xf16>
  %relu207 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv205 : tensor<1x1152x7x7xf16>)
    outs(%empty206 : tensor<1x1152x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1152x7x7xf16>
  %init208 = tensor.empty() : tensor<1x192x7x7xf16>
  %fill209 = linalg.fill ins(%cst : f16) outs(%init208 : tensor<1x192x7x7xf16>) -> tensor<1x192x7x7xf16>
  %conv210 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu207, %w_blk13_proj : tensor<1x1152x7x7xf16>, tensor<192x1152x1x1xf16>)
    outs(%fill209 : tensor<1x192x7x7xf16>) -> tensor<1x192x7x7xf16>
  %empty211 = tensor.empty() : tensor<1x192x7x7xf16>
  %add212 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv210, %conv196 : tensor<1x192x7x7xf16>, tensor<1x192x7x7xf16>)
    outs(%empty211 : tensor<1x192x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x192x7x7xf16>

  // IRB 14: 192->192 mid=1152 s=1 7x7
  %init213 = tensor.empty() : tensor<1x1152x7x7xf16>
  %fill214 = linalg.fill ins(%cst : f16) outs(%init213 : tensor<1x1152x7x7xf16>) -> tensor<1x1152x7x7xf16>
  %conv215 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add212, %w_blk14_exp : tensor<1x192x7x7xf16>, tensor<1152x192x1x1xf16>)
    outs(%fill214 : tensor<1x1152x7x7xf16>) -> tensor<1x1152x7x7xf16>
  %empty216 = tensor.empty() : tensor<1x1152x7x7xf16>
  %relu217 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv215 : tensor<1x1152x7x7xf16>)
    outs(%empty216 : tensor<1x1152x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1152x7x7xf16>
  %pad218 = tensor.pad %relu217 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1152x7x7xf16> to tensor<1x1152x9x9xf16>
  %init219 = tensor.empty() : tensor<1x1152x7x7xf16>
  %fill220 = linalg.fill ins(%cst : f16) outs(%init219 : tensor<1x1152x7x7xf16>) -> tensor<1x1152x7x7xf16>
  %conv221 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad218, %w_blk14_dw : tensor<1x1152x9x9xf16>, tensor<1152x1152x3x3xf16>)
    outs(%fill220 : tensor<1x1152x7x7xf16>) -> tensor<1x1152x7x7xf16>
  %empty222 = tensor.empty() : tensor<1x1152x7x7xf16>
  %relu223 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv221 : tensor<1x1152x7x7xf16>)
    outs(%empty222 : tensor<1x1152x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1152x7x7xf16>
  %init224 = tensor.empty() : tensor<1x192x7x7xf16>
  %fill225 = linalg.fill ins(%cst : f16) outs(%init224 : tensor<1x192x7x7xf16>) -> tensor<1x192x7x7xf16>
  %conv226 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu223, %w_blk14_proj : tensor<1x1152x7x7xf16>, tensor<192x1152x1x1xf16>)
    outs(%fill225 : tensor<1x192x7x7xf16>) -> tensor<1x192x7x7xf16>
  %empty227 = tensor.empty() : tensor<1x192x7x7xf16>
  %add228 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv226, %add212 : tensor<1x192x7x7xf16>, tensor<1x192x7x7xf16>)
    outs(%empty227 : tensor<1x192x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x192x7x7xf16>

  // IRB 15: 192->192 mid=1152 s=1 7x7
  %init229 = tensor.empty() : tensor<1x1152x7x7xf16>
  %fill230 = linalg.fill ins(%cst : f16) outs(%init229 : tensor<1x1152x7x7xf16>) -> tensor<1x1152x7x7xf16>
  %conv231 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add228, %w_blk15_exp : tensor<1x192x7x7xf16>, tensor<1152x192x1x1xf16>)
    outs(%fill230 : tensor<1x1152x7x7xf16>) -> tensor<1x1152x7x7xf16>
  %empty232 = tensor.empty() : tensor<1x1152x7x7xf16>
  %relu233 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv231 : tensor<1x1152x7x7xf16>)
    outs(%empty232 : tensor<1x1152x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1152x7x7xf16>
  %pad234 = tensor.pad %relu233 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1152x7x7xf16> to tensor<1x1152x9x9xf16>
  %init235 = tensor.empty() : tensor<1x1152x7x7xf16>
  %fill236 = linalg.fill ins(%cst : f16) outs(%init235 : tensor<1x1152x7x7xf16>) -> tensor<1x1152x7x7xf16>
  %conv237 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad234, %w_blk15_dw : tensor<1x1152x9x9xf16>, tensor<1152x1152x3x3xf16>)
    outs(%fill236 : tensor<1x1152x7x7xf16>) -> tensor<1x1152x7x7xf16>
  %empty238 = tensor.empty() : tensor<1x1152x7x7xf16>
  %relu239 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv237 : tensor<1x1152x7x7xf16>)
    outs(%empty238 : tensor<1x1152x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1152x7x7xf16>
  %init240 = tensor.empty() : tensor<1x192x7x7xf16>
  %fill241 = linalg.fill ins(%cst : f16) outs(%init240 : tensor<1x192x7x7xf16>) -> tensor<1x192x7x7xf16>
  %conv242 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu239, %w_blk15_proj : tensor<1x1152x7x7xf16>, tensor<192x1152x1x1xf16>)
    outs(%fill241 : tensor<1x192x7x7xf16>) -> tensor<1x192x7x7xf16>
  %empty243 = tensor.empty() : tensor<1x192x7x7xf16>
  %add244 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv242, %add228 : tensor<1x192x7x7xf16>, tensor<1x192x7x7xf16>)
    outs(%empty243 : tensor<1x192x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x192x7x7xf16>

  // IRB 16: 192->320 mid=1152 s=1 7x7
  %init245 = tensor.empty() : tensor<1x1152x7x7xf16>
  %fill246 = linalg.fill ins(%cst : f16) outs(%init245 : tensor<1x1152x7x7xf16>) -> tensor<1x1152x7x7xf16>
  %conv247 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add244, %w_blk16_exp : tensor<1x192x7x7xf16>, tensor<1152x192x1x1xf16>)
    outs(%fill246 : tensor<1x1152x7x7xf16>) -> tensor<1x1152x7x7xf16>
  %empty248 = tensor.empty() : tensor<1x1152x7x7xf16>
  %relu249 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv247 : tensor<1x1152x7x7xf16>)
    outs(%empty248 : tensor<1x1152x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1152x7x7xf16>
  %pad250 = tensor.pad %relu249 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1152x7x7xf16> to tensor<1x1152x9x9xf16>
  %init251 = tensor.empty() : tensor<1x1152x7x7xf16>
  %fill252 = linalg.fill ins(%cst : f16) outs(%init251 : tensor<1x1152x7x7xf16>) -> tensor<1x1152x7x7xf16>
  %conv253 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad250, %w_blk16_dw : tensor<1x1152x9x9xf16>, tensor<1152x1152x3x3xf16>)
    outs(%fill252 : tensor<1x1152x7x7xf16>) -> tensor<1x1152x7x7xf16>
  %empty254 = tensor.empty() : tensor<1x1152x7x7xf16>
  %relu255 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv253 : tensor<1x1152x7x7xf16>)
    outs(%empty254 : tensor<1x1152x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1152x7x7xf16>
  %init256 = tensor.empty() : tensor<1x320x7x7xf16>
  %fill257 = linalg.fill ins(%cst : f16) outs(%init256 : tensor<1x320x7x7xf16>) -> tensor<1x320x7x7xf16>
  %conv258 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu255, %w_blk16_proj : tensor<1x1152x7x7xf16>, tensor<320x1152x1x1xf16>)
    outs(%fill257 : tensor<1x320x7x7xf16>) -> tensor<1x320x7x7xf16>

  // Final 1x1: 320->1280
  %init259 = tensor.empty() : tensor<1x1280x7x7xf16>
  %fill260 = linalg.fill ins(%cst : f16) outs(%init259 : tensor<1x1280x7x7xf16>) -> tensor<1x1280x7x7xf16>
  %conv261 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv258, %w_final : tensor<1x320x7x7xf16>, tensor<1280x320x1x1xf16>)
    outs(%fill260 : tensor<1x1280x7x7xf16>) -> tensor<1x1280x7x7xf16>
  %empty262 = tensor.empty() : tensor<1x1280x7x7xf16>
  %relu263 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv261 : tensor<1x1280x7x7xf16>)
    outs(%empty262 : tensor<1x1280x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1280x7x7xf16>

  // FC: 1x1 1280->1000
  %init264 = tensor.empty() : tensor<1x1000x7x7xf16>
  %fill265 = linalg.fill ins(%cst : f16) outs(%init264 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  %conv266 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu263, %w_fc : tensor<1x1280x7x7xf16>, tensor<1000x1280x1x1xf16>)
    outs(%fill265 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  return %conv266 : tensor<1x1000x7x7xf16>
}
