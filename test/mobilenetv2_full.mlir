func.func @mobilenetv2(
    %input: tensor<1x3x224x224xf16>,
    %w_conv0: tensor<32x3x3x3xf16>,
    %w_blk0_dw: tensor<32x32x3x3xf16>,
    %w_blk0_proj: tensor<16x32x1x1xf16>,
    %w_blk1_exp: tensor<96x16x1x1xf16>,
    %w_blk1_dw: tensor<96x96x3x3xf16>,
    %w_blk1_proj: tensor<24x96x1x1xf16>,
    %w_blk2_exp: tensor<144x24x1x1xf16>,
    %w_blk2_dw: tensor<144x144x3x3xf16>,
    %w_blk2_proj: tensor<24x144x1x1xf16>,
    %w_blk3_exp: tensor<144x24x1x1xf16>,
    %w_blk3_dw: tensor<144x144x3x3xf16>,
    %w_blk3_proj: tensor<32x144x1x1xf16>,
    %w_blk4_exp: tensor<192x32x1x1xf16>,
    %w_blk4_dw: tensor<192x192x3x3xf16>,
    %w_blk4_proj: tensor<32x192x1x1xf16>,
    %w_blk5_exp: tensor<192x32x1x1xf16>,
    %w_blk5_dw: tensor<192x192x3x3xf16>,
    %w_blk5_proj: tensor<32x192x1x1xf16>,
    %w_final: tensor<1280x32x1x1xf16>,
    %w_fc: tensor<1000x1280x1x1xf16>) -> tensor<1x1000x28x28xf16> {
  %cst = arith.constant 0.0 : f16

  // Initial conv: 3x3 stride 2, 3->32, 224->112
  %pad0 = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x224x224xf16> to tensor<1x3x226x226xf16>
  %init1 = tensor.empty() : tensor<1x32x112x112xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad0, %w_conv0 : tensor<1x3x226x226xf16>, tensor<32x3x3x3xf16>)
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

  // Inverted residual block 0: 32->16, mid=32, stride=1, 112x112
  // 3x3 conv
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
  // 1x1 project conv (no relu)
  %init12 = tensor.empty() : tensor<1x16x112x112xf16>
  %fill13 = linalg.fill ins(%cst : f16) outs(%init12 : tensor<1x16x112x112xf16>) -> tensor<1x16x112x112xf16>
  %conv14 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu11, %w_blk0_proj : tensor<1x32x112x112xf16>, tensor<16x32x1x1xf16>)
    outs(%fill13 : tensor<1x16x112x112xf16>) -> tensor<1x16x112x112xf16>

  // Inverted residual block 1: 16->24, mid=96, stride=2, 112x112
  // 1x1 expand conv
  %init15 = tensor.empty() : tensor<1x96x112x112xf16>
  %fill16 = linalg.fill ins(%cst : f16) outs(%init15 : tensor<1x96x112x112xf16>) -> tensor<1x96x112x112xf16>
  %conv17 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv14, %w_blk1_exp : tensor<1x16x112x112xf16>, tensor<96x16x1x1xf16>)
    outs(%fill16 : tensor<1x96x112x112xf16>) -> tensor<1x96x112x112xf16>
  %empty18 = tensor.empty() : tensor<1x96x112x112xf16>
  %relu19 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv17 : tensor<1x96x112x112xf16>)
    outs(%empty18 : tensor<1x96x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x96x112x112xf16>
  // 3x3 conv
  %pad20 = tensor.pad %relu19 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x96x112x112xf16> to tensor<1x96x114x114xf16>
  %init21 = tensor.empty() : tensor<1x96x56x56xf16>
  %fill22 = linalg.fill ins(%cst : f16) outs(%init21 : tensor<1x96x56x56xf16>) -> tensor<1x96x56x56xf16>
  %conv23 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad20, %w_blk1_dw : tensor<1x96x114x114xf16>, tensor<96x96x3x3xf16>)
    outs(%fill22 : tensor<1x96x56x56xf16>) -> tensor<1x96x56x56xf16>
  %empty24 = tensor.empty() : tensor<1x96x56x56xf16>
  %relu25 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv23 : tensor<1x96x56x56xf16>)
    outs(%empty24 : tensor<1x96x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x96x56x56xf16>
  // 1x1 project conv (no relu)
  %init26 = tensor.empty() : tensor<1x24x56x56xf16>
  %fill27 = linalg.fill ins(%cst : f16) outs(%init26 : tensor<1x24x56x56xf16>) -> tensor<1x24x56x56xf16>
  %conv28 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu25, %w_blk1_proj : tensor<1x96x56x56xf16>, tensor<24x96x1x1xf16>)
    outs(%fill27 : tensor<1x24x56x56xf16>) -> tensor<1x24x56x56xf16>

  // Inverted residual block 2: 24->24, mid=144, stride=1, 56x56
  // 1x1 expand conv
  %init29 = tensor.empty() : tensor<1x144x56x56xf16>
  %fill30 = linalg.fill ins(%cst : f16) outs(%init29 : tensor<1x144x56x56xf16>) -> tensor<1x144x56x56xf16>
  %conv31 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv28, %w_blk2_exp : tensor<1x24x56x56xf16>, tensor<144x24x1x1xf16>)
    outs(%fill30 : tensor<1x144x56x56xf16>) -> tensor<1x144x56x56xf16>
  %empty32 = tensor.empty() : tensor<1x144x56x56xf16>
  %relu33 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv31 : tensor<1x144x56x56xf16>)
    outs(%empty32 : tensor<1x144x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x144x56x56xf16>
  // 3x3 conv
  %pad34 = tensor.pad %relu33 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x144x56x56xf16> to tensor<1x144x58x58xf16>
  %init35 = tensor.empty() : tensor<1x144x56x56xf16>
  %fill36 = linalg.fill ins(%cst : f16) outs(%init35 : tensor<1x144x56x56xf16>) -> tensor<1x144x56x56xf16>
  %conv37 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad34, %w_blk2_dw : tensor<1x144x58x58xf16>, tensor<144x144x3x3xf16>)
    outs(%fill36 : tensor<1x144x56x56xf16>) -> tensor<1x144x56x56xf16>
  %empty38 = tensor.empty() : tensor<1x144x56x56xf16>
  %relu39 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv37 : tensor<1x144x56x56xf16>)
    outs(%empty38 : tensor<1x144x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x144x56x56xf16>
  // 1x1 project conv (no relu)
  %init40 = tensor.empty() : tensor<1x24x56x56xf16>
  %fill41 = linalg.fill ins(%cst : f16) outs(%init40 : tensor<1x24x56x56xf16>) -> tensor<1x24x56x56xf16>
  %conv42 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu39, %w_blk2_proj : tensor<1x144x56x56xf16>, tensor<24x144x1x1xf16>)
    outs(%fill41 : tensor<1x24x56x56xf16>) -> tensor<1x24x56x56xf16>
  // Residual add
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

  // Inverted residual block 3: 24->32, mid=144, stride=2, 56x56
  // 1x1 expand conv
  %init45 = tensor.empty() : tensor<1x144x56x56xf16>
  %fill46 = linalg.fill ins(%cst : f16) outs(%init45 : tensor<1x144x56x56xf16>) -> tensor<1x144x56x56xf16>
  %conv47 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add44, %w_blk3_exp : tensor<1x24x56x56xf16>, tensor<144x24x1x1xf16>)
    outs(%fill46 : tensor<1x144x56x56xf16>) -> tensor<1x144x56x56xf16>
  %empty48 = tensor.empty() : tensor<1x144x56x56xf16>
  %relu49 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv47 : tensor<1x144x56x56xf16>)
    outs(%empty48 : tensor<1x144x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x144x56x56xf16>
  // 3x3 conv
  %pad50 = tensor.pad %relu49 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x144x56x56xf16> to tensor<1x144x58x58xf16>
  %init51 = tensor.empty() : tensor<1x144x28x28xf16>
  %fill52 = linalg.fill ins(%cst : f16) outs(%init51 : tensor<1x144x28x28xf16>) -> tensor<1x144x28x28xf16>
  %conv53 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad50, %w_blk3_dw : tensor<1x144x58x58xf16>, tensor<144x144x3x3xf16>)
    outs(%fill52 : tensor<1x144x28x28xf16>) -> tensor<1x144x28x28xf16>
  %empty54 = tensor.empty() : tensor<1x144x28x28xf16>
  %relu55 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv53 : tensor<1x144x28x28xf16>)
    outs(%empty54 : tensor<1x144x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x144x28x28xf16>
  // 1x1 project conv (no relu)
  %init56 = tensor.empty() : tensor<1x32x28x28xf16>
  %fill57 = linalg.fill ins(%cst : f16) outs(%init56 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %conv58 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu55, %w_blk3_proj : tensor<1x144x28x28xf16>, tensor<32x144x1x1xf16>)
    outs(%fill57 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>

  // Inverted residual block 4: 32->32, mid=192, stride=1, 28x28
  // 1x1 expand conv
  %init59 = tensor.empty() : tensor<1x192x28x28xf16>
  %fill60 = linalg.fill ins(%cst : f16) outs(%init59 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %conv61 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv58, %w_blk4_exp : tensor<1x32x28x28xf16>, tensor<192x32x1x1xf16>)
    outs(%fill60 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %empty62 = tensor.empty() : tensor<1x192x28x28xf16>
  %relu63 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv61 : tensor<1x192x28x28xf16>)
    outs(%empty62 : tensor<1x192x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x28x28xf16>
  // 3x3 conv
  %pad64 = tensor.pad %relu63 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x192x28x28xf16> to tensor<1x192x30x30xf16>
  %init65 = tensor.empty() : tensor<1x192x28x28xf16>
  %fill66 = linalg.fill ins(%cst : f16) outs(%init65 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %conv67 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad64, %w_blk4_dw : tensor<1x192x30x30xf16>, tensor<192x192x3x3xf16>)
    outs(%fill66 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %empty68 = tensor.empty() : tensor<1x192x28x28xf16>
  %relu69 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv67 : tensor<1x192x28x28xf16>)
    outs(%empty68 : tensor<1x192x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x28x28xf16>
  // 1x1 project conv (no relu)
  %init70 = tensor.empty() : tensor<1x32x28x28xf16>
  %fill71 = linalg.fill ins(%cst : f16) outs(%init70 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %conv72 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu69, %w_blk4_proj : tensor<1x192x28x28xf16>, tensor<32x192x1x1xf16>)
    outs(%fill71 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  // Residual add
  %empty73 = tensor.empty() : tensor<1x32x28x28xf16>
  %add74 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv72, %conv58 : tensor<1x32x28x28xf16>, tensor<1x32x28x28xf16>)
    outs(%empty73 : tensor<1x32x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x32x28x28xf16>

  // Inverted residual block 5: 32->32, mid=192, stride=1, 28x28
  // 1x1 expand conv
  %init75 = tensor.empty() : tensor<1x192x28x28xf16>
  %fill76 = linalg.fill ins(%cst : f16) outs(%init75 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %conv77 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add74, %w_blk5_exp : tensor<1x32x28x28xf16>, tensor<192x32x1x1xf16>)
    outs(%fill76 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %empty78 = tensor.empty() : tensor<1x192x28x28xf16>
  %relu79 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv77 : tensor<1x192x28x28xf16>)
    outs(%empty78 : tensor<1x192x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x28x28xf16>
  // 3x3 conv
  %pad80 = tensor.pad %relu79 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x192x28x28xf16> to tensor<1x192x30x30xf16>
  %init81 = tensor.empty() : tensor<1x192x28x28xf16>
  %fill82 = linalg.fill ins(%cst : f16) outs(%init81 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %conv83 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad80, %w_blk5_dw : tensor<1x192x30x30xf16>, tensor<192x192x3x3xf16>)
    outs(%fill82 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %empty84 = tensor.empty() : tensor<1x192x28x28xf16>
  %relu85 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv83 : tensor<1x192x28x28xf16>)
    outs(%empty84 : tensor<1x192x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x28x28xf16>
  // 1x1 project conv (no relu)
  %init86 = tensor.empty() : tensor<1x32x28x28xf16>
  %fill87 = linalg.fill ins(%cst : f16) outs(%init86 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %conv88 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu85, %w_blk5_proj : tensor<1x192x28x28xf16>, tensor<32x192x1x1xf16>)
    outs(%fill87 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  // Residual add
  %empty89 = tensor.empty() : tensor<1x32x28x28xf16>
  %add90 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv88, %add74 : tensor<1x32x28x28xf16>, tensor<1x32x28x28xf16>)
    outs(%empty89 : tensor<1x32x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x32x28x28xf16>

  // Final 1x1 conv: 32->1280, 28x28
  %init91 = tensor.empty() : tensor<1x1280x28x28xf16>
  %fill92 = linalg.fill ins(%cst : f16) outs(%init91 : tensor<1x1280x28x28xf16>) -> tensor<1x1280x28x28xf16>
  %conv93 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add90, %w_final : tensor<1x32x28x28xf16>, tensor<1280x32x1x1xf16>)
    outs(%fill92 : tensor<1x1280x28x28xf16>) -> tensor<1x1280x28x28xf16>
  %empty94 = tensor.empty() : tensor<1x1280x28x28xf16>
  %relu95 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv93 : tensor<1x1280x28x28xf16>)
    outs(%empty94 : tensor<1x1280x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1280x28x28xf16>

  // FC as 1x1 conv: 1280->1000, 28x28
  %init96 = tensor.empty() : tensor<1x1000x28x28xf16>
  %fill97 = linalg.fill ins(%cst : f16) outs(%init96 : tensor<1x1000x28x28xf16>) -> tensor<1x1000x28x28xf16>
  %conv98 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu95, %w_fc : tensor<1x1280x28x28xf16>, tensor<1000x1280x1x1xf16>)
    outs(%fill97 : tensor<1x1000x28x28xf16>) -> tensor<1x1000x28x28xf16>
  return %conv98 : tensor<1x1000x28x28xf16>
}
