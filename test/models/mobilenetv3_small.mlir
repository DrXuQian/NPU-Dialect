func.func @mobilenetv3_small(
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
    %w_blk5_proj: tensor<48x120x1x1xf16>,
    %w_blk6_exp: tensor<144x48x1x1xf16>,
    %w_blk6_dw: tensor<144x144x3x3xf16>,
    %w_blk6_proj: tensor<48x144x1x1xf16>,
    %w_blk7_exp: tensor<288x48x1x1xf16>,
    %w_blk7_dw: tensor<288x288x3x3xf16>,
    %w_blk7_proj: tensor<96x288x1x1xf16>,
    %w_blk8_exp: tensor<576x96x1x1xf16>,
    %w_blk8_dw: tensor<576x576x3x3xf16>,
    %w_blk8_proj: tensor<96x576x1x1xf16>,
    %w_blk9_exp: tensor<576x96x1x1xf16>,
    %w_blk9_dw: tensor<576x576x3x3xf16>,
    %w_blk9_proj: tensor<96x576x1x1xf16>,
    %w_final: tensor<576x96x1x1xf16>,
    %w_fc: tensor<1000x576x1x1xf16>) -> tensor<1x1000x7x7xf16> {
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

  // IRB 0: 16->16 mid=16 s=2 112x112
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x16x112x112xf16> to tensor<1x16x114x114xf16>
  %init7 = tensor.empty() : tensor<1x16x56x56xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x16x56x56xf16>) -> tensor<1x16x56x56xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad6, %w_blk0_dw : tensor<1x16x114x114xf16>, tensor<16x16x3x3xf16>)
    outs(%fill8 : tensor<1x16x56x56xf16>) -> tensor<1x16x56x56xf16>
  %empty10 = tensor.empty() : tensor<1x16x56x56xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x16x56x56xf16>)
    outs(%empty10 : tensor<1x16x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x16x56x56xf16>
  %init12 = tensor.empty() : tensor<1x16x56x56xf16>
  %fill13 = linalg.fill ins(%cst : f16) outs(%init12 : tensor<1x16x56x56xf16>) -> tensor<1x16x56x56xf16>
  %conv14 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu11, %w_blk0_proj : tensor<1x16x56x56xf16>, tensor<16x16x1x1xf16>)
    outs(%fill13 : tensor<1x16x56x56xf16>) -> tensor<1x16x56x56xf16>

  // IRB 1: 16->24 mid=64 s=2 56x56
  %init15 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill16 = linalg.fill ins(%cst : f16) outs(%init15 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv17 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv14, %w_blk1_exp : tensor<1x16x56x56xf16>, tensor<64x16x1x1xf16>)
    outs(%fill16 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty18 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu19 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv17 : tensor<1x64x56x56xf16>)
    outs(%empty18 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad20 = tensor.pad %relu19 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init21 = tensor.empty() : tensor<1x64x28x28xf16>
  %fill22 = linalg.fill ins(%cst : f16) outs(%init21 : tensor<1x64x28x28xf16>) -> tensor<1x64x28x28xf16>
  %conv23 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad20, %w_blk1_dw : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill22 : tensor<1x64x28x28xf16>) -> tensor<1x64x28x28xf16>
  %empty24 = tensor.empty() : tensor<1x64x28x28xf16>
  %relu25 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv23 : tensor<1x64x28x28xf16>)
    outs(%empty24 : tensor<1x64x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x28x28xf16>
  %init26 = tensor.empty() : tensor<1x24x28x28xf16>
  %fill27 = linalg.fill ins(%cst : f16) outs(%init26 : tensor<1x24x28x28xf16>) -> tensor<1x24x28x28xf16>
  %conv28 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu25, %w_blk1_proj : tensor<1x64x28x28xf16>, tensor<24x64x1x1xf16>)
    outs(%fill27 : tensor<1x24x28x28xf16>) -> tensor<1x24x28x28xf16>

  // IRB 2: 24->24 mid=72 s=1 28x28
  %init29 = tensor.empty() : tensor<1x72x28x28xf16>
  %fill30 = linalg.fill ins(%cst : f16) outs(%init29 : tensor<1x72x28x28xf16>) -> tensor<1x72x28x28xf16>
  %conv31 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv28, %w_blk2_exp : tensor<1x24x28x28xf16>, tensor<72x24x1x1xf16>)
    outs(%fill30 : tensor<1x72x28x28xf16>) -> tensor<1x72x28x28xf16>
  %empty32 = tensor.empty() : tensor<1x72x28x28xf16>
  %relu33 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv31 : tensor<1x72x28x28xf16>)
    outs(%empty32 : tensor<1x72x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x28x28xf16>
  %pad34 = tensor.pad %relu33 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x72x28x28xf16> to tensor<1x72x30x30xf16>
  %init35 = tensor.empty() : tensor<1x72x28x28xf16>
  %fill36 = linalg.fill ins(%cst : f16) outs(%init35 : tensor<1x72x28x28xf16>) -> tensor<1x72x28x28xf16>
  %conv37 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad34, %w_blk2_dw : tensor<1x72x30x30xf16>, tensor<72x72x3x3xf16>)
    outs(%fill36 : tensor<1x72x28x28xf16>) -> tensor<1x72x28x28xf16>
  %empty38 = tensor.empty() : tensor<1x72x28x28xf16>
  %relu39 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv37 : tensor<1x72x28x28xf16>)
    outs(%empty38 : tensor<1x72x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x28x28xf16>
  %init40 = tensor.empty() : tensor<1x24x28x28xf16>
  %fill41 = linalg.fill ins(%cst : f16) outs(%init40 : tensor<1x24x28x28xf16>) -> tensor<1x24x28x28xf16>
  %conv42 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu39, %w_blk2_proj : tensor<1x72x28x28xf16>, tensor<24x72x1x1xf16>)
    outs(%fill41 : tensor<1x24x28x28xf16>) -> tensor<1x24x28x28xf16>
  %empty43 = tensor.empty() : tensor<1x24x28x28xf16>
  %add44 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv42, %conv28 : tensor<1x24x28x28xf16>, tensor<1x24x28x28xf16>)
    outs(%empty43 : tensor<1x24x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x24x28x28xf16>

  // IRB 3: 24->40 mid=72 s=2 28x28
  %init45 = tensor.empty() : tensor<1x72x28x28xf16>
  %fill46 = linalg.fill ins(%cst : f16) outs(%init45 : tensor<1x72x28x28xf16>) -> tensor<1x72x28x28xf16>
  %conv47 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add44, %w_blk3_exp : tensor<1x24x28x28xf16>, tensor<72x24x1x1xf16>)
    outs(%fill46 : tensor<1x72x28x28xf16>) -> tensor<1x72x28x28xf16>
  %empty48 = tensor.empty() : tensor<1x72x28x28xf16>
  %relu49 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv47 : tensor<1x72x28x28xf16>)
    outs(%empty48 : tensor<1x72x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x28x28xf16>
  %pad50 = tensor.pad %relu49 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x72x28x28xf16> to tensor<1x72x30x30xf16>
  %init51 = tensor.empty() : tensor<1x72x14x14xf16>
  %fill52 = linalg.fill ins(%cst : f16) outs(%init51 : tensor<1x72x14x14xf16>) -> tensor<1x72x14x14xf16>
  %conv53 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad50, %w_blk3_dw : tensor<1x72x30x30xf16>, tensor<72x72x3x3xf16>)
    outs(%fill52 : tensor<1x72x14x14xf16>) -> tensor<1x72x14x14xf16>
  %empty54 = tensor.empty() : tensor<1x72x14x14xf16>
  %relu55 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv53 : tensor<1x72x14x14xf16>)
    outs(%empty54 : tensor<1x72x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x14x14xf16>
  %init56 = tensor.empty() : tensor<1x40x14x14xf16>
  %fill57 = linalg.fill ins(%cst : f16) outs(%init56 : tensor<1x40x14x14xf16>) -> tensor<1x40x14x14xf16>
  %conv58 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu55, %w_blk3_proj : tensor<1x72x14x14xf16>, tensor<40x72x1x1xf16>)
    outs(%fill57 : tensor<1x40x14x14xf16>) -> tensor<1x40x14x14xf16>

  // IRB 4: 40->40 mid=120 s=1 14x14
  %init59 = tensor.empty() : tensor<1x120x14x14xf16>
  %fill60 = linalg.fill ins(%cst : f16) outs(%init59 : tensor<1x120x14x14xf16>) -> tensor<1x120x14x14xf16>
  %conv61 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv58, %w_blk4_exp : tensor<1x40x14x14xf16>, tensor<120x40x1x1xf16>)
    outs(%fill60 : tensor<1x120x14x14xf16>) -> tensor<1x120x14x14xf16>
  %empty62 = tensor.empty() : tensor<1x120x14x14xf16>
  %relu63 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv61 : tensor<1x120x14x14xf16>)
    outs(%empty62 : tensor<1x120x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x120x14x14xf16>
  %pad64 = tensor.pad %relu63 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x120x14x14xf16> to tensor<1x120x16x16xf16>
  %init65 = tensor.empty() : tensor<1x120x14x14xf16>
  %fill66 = linalg.fill ins(%cst : f16) outs(%init65 : tensor<1x120x14x14xf16>) -> tensor<1x120x14x14xf16>
  %conv67 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad64, %w_blk4_dw : tensor<1x120x16x16xf16>, tensor<120x120x3x3xf16>)
    outs(%fill66 : tensor<1x120x14x14xf16>) -> tensor<1x120x14x14xf16>
  %empty68 = tensor.empty() : tensor<1x120x14x14xf16>
  %relu69 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv67 : tensor<1x120x14x14xf16>)
    outs(%empty68 : tensor<1x120x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x120x14x14xf16>
  %init70 = tensor.empty() : tensor<1x40x14x14xf16>
  %fill71 = linalg.fill ins(%cst : f16) outs(%init70 : tensor<1x40x14x14xf16>) -> tensor<1x40x14x14xf16>
  %conv72 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu69, %w_blk4_proj : tensor<1x120x14x14xf16>, tensor<40x120x1x1xf16>)
    outs(%fill71 : tensor<1x40x14x14xf16>) -> tensor<1x40x14x14xf16>
  %empty73 = tensor.empty() : tensor<1x40x14x14xf16>
  %add74 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv72, %conv58 : tensor<1x40x14x14xf16>, tensor<1x40x14x14xf16>)
    outs(%empty73 : tensor<1x40x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x40x14x14xf16>

  // IRB 5: 40->48 mid=120 s=1 14x14
  %init75 = tensor.empty() : tensor<1x120x14x14xf16>
  %fill76 = linalg.fill ins(%cst : f16) outs(%init75 : tensor<1x120x14x14xf16>) -> tensor<1x120x14x14xf16>
  %conv77 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add74, %w_blk5_exp : tensor<1x40x14x14xf16>, tensor<120x40x1x1xf16>)
    outs(%fill76 : tensor<1x120x14x14xf16>) -> tensor<1x120x14x14xf16>
  %empty78 = tensor.empty() : tensor<1x120x14x14xf16>
  %relu79 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv77 : tensor<1x120x14x14xf16>)
    outs(%empty78 : tensor<1x120x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x120x14x14xf16>
  %pad80 = tensor.pad %relu79 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x120x14x14xf16> to tensor<1x120x16x16xf16>
  %init81 = tensor.empty() : tensor<1x120x14x14xf16>
  %fill82 = linalg.fill ins(%cst : f16) outs(%init81 : tensor<1x120x14x14xf16>) -> tensor<1x120x14x14xf16>
  %conv83 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad80, %w_blk5_dw : tensor<1x120x16x16xf16>, tensor<120x120x3x3xf16>)
    outs(%fill82 : tensor<1x120x14x14xf16>) -> tensor<1x120x14x14xf16>
  %empty84 = tensor.empty() : tensor<1x120x14x14xf16>
  %relu85 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv83 : tensor<1x120x14x14xf16>)
    outs(%empty84 : tensor<1x120x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x120x14x14xf16>
  %init86 = tensor.empty() : tensor<1x48x14x14xf16>
  %fill87 = linalg.fill ins(%cst : f16) outs(%init86 : tensor<1x48x14x14xf16>) -> tensor<1x48x14x14xf16>
  %conv88 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu85, %w_blk5_proj : tensor<1x120x14x14xf16>, tensor<48x120x1x1xf16>)
    outs(%fill87 : tensor<1x48x14x14xf16>) -> tensor<1x48x14x14xf16>

  // IRB 6: 48->48 mid=144 s=1 14x14
  %init89 = tensor.empty() : tensor<1x144x14x14xf16>
  %fill90 = linalg.fill ins(%cst : f16) outs(%init89 : tensor<1x144x14x14xf16>) -> tensor<1x144x14x14xf16>
  %conv91 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv88, %w_blk6_exp : tensor<1x48x14x14xf16>, tensor<144x48x1x1xf16>)
    outs(%fill90 : tensor<1x144x14x14xf16>) -> tensor<1x144x14x14xf16>
  %empty92 = tensor.empty() : tensor<1x144x14x14xf16>
  %relu93 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv91 : tensor<1x144x14x14xf16>)
    outs(%empty92 : tensor<1x144x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x144x14x14xf16>
  %pad94 = tensor.pad %relu93 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x144x14x14xf16> to tensor<1x144x16x16xf16>
  %init95 = tensor.empty() : tensor<1x144x14x14xf16>
  %fill96 = linalg.fill ins(%cst : f16) outs(%init95 : tensor<1x144x14x14xf16>) -> tensor<1x144x14x14xf16>
  %conv97 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad94, %w_blk6_dw : tensor<1x144x16x16xf16>, tensor<144x144x3x3xf16>)
    outs(%fill96 : tensor<1x144x14x14xf16>) -> tensor<1x144x14x14xf16>
  %empty98 = tensor.empty() : tensor<1x144x14x14xf16>
  %relu99 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv97 : tensor<1x144x14x14xf16>)
    outs(%empty98 : tensor<1x144x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x144x14x14xf16>
  %init100 = tensor.empty() : tensor<1x48x14x14xf16>
  %fill101 = linalg.fill ins(%cst : f16) outs(%init100 : tensor<1x48x14x14xf16>) -> tensor<1x48x14x14xf16>
  %conv102 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu99, %w_blk6_proj : tensor<1x144x14x14xf16>, tensor<48x144x1x1xf16>)
    outs(%fill101 : tensor<1x48x14x14xf16>) -> tensor<1x48x14x14xf16>
  %empty103 = tensor.empty() : tensor<1x48x14x14xf16>
  %add104 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv102, %conv88 : tensor<1x48x14x14xf16>, tensor<1x48x14x14xf16>)
    outs(%empty103 : tensor<1x48x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x48x14x14xf16>

  // IRB 7: 48->96 mid=288 s=2 14x14
  %init105 = tensor.empty() : tensor<1x288x14x14xf16>
  %fill106 = linalg.fill ins(%cst : f16) outs(%init105 : tensor<1x288x14x14xf16>) -> tensor<1x288x14x14xf16>
  %conv107 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add104, %w_blk7_exp : tensor<1x48x14x14xf16>, tensor<288x48x1x1xf16>)
    outs(%fill106 : tensor<1x288x14x14xf16>) -> tensor<1x288x14x14xf16>
  %empty108 = tensor.empty() : tensor<1x288x14x14xf16>
  %relu109 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv107 : tensor<1x288x14x14xf16>)
    outs(%empty108 : tensor<1x288x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x288x14x14xf16>
  %pad110 = tensor.pad %relu109 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x288x14x14xf16> to tensor<1x288x16x16xf16>
  %init111 = tensor.empty() : tensor<1x288x7x7xf16>
  %fill112 = linalg.fill ins(%cst : f16) outs(%init111 : tensor<1x288x7x7xf16>) -> tensor<1x288x7x7xf16>
  %conv113 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad110, %w_blk7_dw : tensor<1x288x16x16xf16>, tensor<288x288x3x3xf16>)
    outs(%fill112 : tensor<1x288x7x7xf16>) -> tensor<1x288x7x7xf16>
  %empty114 = tensor.empty() : tensor<1x288x7x7xf16>
  %relu115 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv113 : tensor<1x288x7x7xf16>)
    outs(%empty114 : tensor<1x288x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x288x7x7xf16>
  %init116 = tensor.empty() : tensor<1x96x7x7xf16>
  %fill117 = linalg.fill ins(%cst : f16) outs(%init116 : tensor<1x96x7x7xf16>) -> tensor<1x96x7x7xf16>
  %conv118 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu115, %w_blk7_proj : tensor<1x288x7x7xf16>, tensor<96x288x1x1xf16>)
    outs(%fill117 : tensor<1x96x7x7xf16>) -> tensor<1x96x7x7xf16>

  // IRB 8: 96->96 mid=576 s=1 7x7
  %init119 = tensor.empty() : tensor<1x576x7x7xf16>
  %fill120 = linalg.fill ins(%cst : f16) outs(%init119 : tensor<1x576x7x7xf16>) -> tensor<1x576x7x7xf16>
  %conv121 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv118, %w_blk8_exp : tensor<1x96x7x7xf16>, tensor<576x96x1x1xf16>)
    outs(%fill120 : tensor<1x576x7x7xf16>) -> tensor<1x576x7x7xf16>
  %empty122 = tensor.empty() : tensor<1x576x7x7xf16>
  %relu123 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv121 : tensor<1x576x7x7xf16>)
    outs(%empty122 : tensor<1x576x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x7x7xf16>
  %pad124 = tensor.pad %relu123 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x576x7x7xf16> to tensor<1x576x9x9xf16>
  %init125 = tensor.empty() : tensor<1x576x7x7xf16>
  %fill126 = linalg.fill ins(%cst : f16) outs(%init125 : tensor<1x576x7x7xf16>) -> tensor<1x576x7x7xf16>
  %conv127 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad124, %w_blk8_dw : tensor<1x576x9x9xf16>, tensor<576x576x3x3xf16>)
    outs(%fill126 : tensor<1x576x7x7xf16>) -> tensor<1x576x7x7xf16>
  %empty128 = tensor.empty() : tensor<1x576x7x7xf16>
  %relu129 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv127 : tensor<1x576x7x7xf16>)
    outs(%empty128 : tensor<1x576x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x7x7xf16>
  %init130 = tensor.empty() : tensor<1x96x7x7xf16>
  %fill131 = linalg.fill ins(%cst : f16) outs(%init130 : tensor<1x96x7x7xf16>) -> tensor<1x96x7x7xf16>
  %conv132 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu129, %w_blk8_proj : tensor<1x576x7x7xf16>, tensor<96x576x1x1xf16>)
    outs(%fill131 : tensor<1x96x7x7xf16>) -> tensor<1x96x7x7xf16>
  %empty133 = tensor.empty() : tensor<1x96x7x7xf16>
  %add134 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv132, %conv118 : tensor<1x96x7x7xf16>, tensor<1x96x7x7xf16>)
    outs(%empty133 : tensor<1x96x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x96x7x7xf16>

  // IRB 9: 96->96 mid=576 s=1 7x7
  %init135 = tensor.empty() : tensor<1x576x7x7xf16>
  %fill136 = linalg.fill ins(%cst : f16) outs(%init135 : tensor<1x576x7x7xf16>) -> tensor<1x576x7x7xf16>
  %conv137 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add134, %w_blk9_exp : tensor<1x96x7x7xf16>, tensor<576x96x1x1xf16>)
    outs(%fill136 : tensor<1x576x7x7xf16>) -> tensor<1x576x7x7xf16>
  %empty138 = tensor.empty() : tensor<1x576x7x7xf16>
  %relu139 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv137 : tensor<1x576x7x7xf16>)
    outs(%empty138 : tensor<1x576x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x7x7xf16>
  %pad140 = tensor.pad %relu139 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x576x7x7xf16> to tensor<1x576x9x9xf16>
  %init141 = tensor.empty() : tensor<1x576x7x7xf16>
  %fill142 = linalg.fill ins(%cst : f16) outs(%init141 : tensor<1x576x7x7xf16>) -> tensor<1x576x7x7xf16>
  %conv143 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad140, %w_blk9_dw : tensor<1x576x9x9xf16>, tensor<576x576x3x3xf16>)
    outs(%fill142 : tensor<1x576x7x7xf16>) -> tensor<1x576x7x7xf16>
  %empty144 = tensor.empty() : tensor<1x576x7x7xf16>
  %relu145 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv143 : tensor<1x576x7x7xf16>)
    outs(%empty144 : tensor<1x576x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x7x7xf16>
  %init146 = tensor.empty() : tensor<1x96x7x7xf16>
  %fill147 = linalg.fill ins(%cst : f16) outs(%init146 : tensor<1x96x7x7xf16>) -> tensor<1x96x7x7xf16>
  %conv148 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu145, %w_blk9_proj : tensor<1x576x7x7xf16>, tensor<96x576x1x1xf16>)
    outs(%fill147 : tensor<1x96x7x7xf16>) -> tensor<1x96x7x7xf16>
  %empty149 = tensor.empty() : tensor<1x96x7x7xf16>
  %add150 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv148, %add134 : tensor<1x96x7x7xf16>, tensor<1x96x7x7xf16>)
    outs(%empty149 : tensor<1x96x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x96x7x7xf16>

  // Final 1x1: 96->576
  %init151 = tensor.empty() : tensor<1x576x7x7xf16>
  %fill152 = linalg.fill ins(%cst : f16) outs(%init151 : tensor<1x576x7x7xf16>) -> tensor<1x576x7x7xf16>
  %conv153 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add150, %w_final : tensor<1x96x7x7xf16>, tensor<576x96x1x1xf16>)
    outs(%fill152 : tensor<1x576x7x7xf16>) -> tensor<1x576x7x7xf16>
  %empty154 = tensor.empty() : tensor<1x576x7x7xf16>
  %relu155 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv153 : tensor<1x576x7x7xf16>)
    outs(%empty154 : tensor<1x576x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x7x7xf16>

  // FC: 1x1 576->1000
  %init156 = tensor.empty() : tensor<1x1000x7x7xf16>
  %fill157 = linalg.fill ins(%cst : f16) outs(%init156 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  %conv158 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu155, %w_fc : tensor<1x576x7x7xf16>, tensor<1000x576x1x1xf16>)
    outs(%fill157 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  return %conv158 : tensor<1x1000x7x7xf16>
}
