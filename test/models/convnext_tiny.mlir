func.func @convnext_tiny(
    %input: tensor<1x3x224x224xf16>,
    %w_stem: tensor<96x3x4x4xf16>,
    %w_s0_b0_dw: tensor<96x96x7x7xf16>,
    %w_s0_b0_pw1: tensor<384x96x1x1xf16>,
    %w_s0_b0_pw2: tensor<96x384x1x1xf16>,
    %w_s0_b1_dw: tensor<96x96x7x7xf16>,
    %w_s0_b1_pw1: tensor<384x96x1x1xf16>,
    %w_s0_b1_pw2: tensor<96x384x1x1xf16>,
    %w_s0_b2_dw: tensor<96x96x7x7xf16>,
    %w_s0_b2_pw1: tensor<384x96x1x1xf16>,
    %w_s0_b2_pw2: tensor<96x384x1x1xf16>,
    %w_ds1: tensor<192x96x3x3xf16>,
    %w_s1_b0_dw: tensor<192x192x7x7xf16>,
    %w_s1_b0_pw1: tensor<768x192x1x1xf16>,
    %w_s1_b0_pw2: tensor<192x768x1x1xf16>,
    %w_s1_b1_dw: tensor<192x192x7x7xf16>,
    %w_s1_b1_pw1: tensor<768x192x1x1xf16>,
    %w_s1_b1_pw2: tensor<192x768x1x1xf16>,
    %w_s1_b2_dw: tensor<192x192x7x7xf16>,
    %w_s1_b2_pw1: tensor<768x192x1x1xf16>,
    %w_s1_b2_pw2: tensor<192x768x1x1xf16>,
    %w_ds2: tensor<384x192x3x3xf16>,
    %w_s2_b0_dw: tensor<384x384x7x7xf16>,
    %w_s2_b0_pw1: tensor<1536x384x1x1xf16>,
    %w_s2_b0_pw2: tensor<384x1536x1x1xf16>,
    %w_s2_b1_dw: tensor<384x384x7x7xf16>,
    %w_s2_b1_pw1: tensor<1536x384x1x1xf16>,
    %w_s2_b1_pw2: tensor<384x1536x1x1xf16>,
    %w_s2_b2_dw: tensor<384x384x7x7xf16>,
    %w_s2_b2_pw1: tensor<1536x384x1x1xf16>,
    %w_s2_b2_pw2: tensor<384x1536x1x1xf16>,
    %w_s2_b3_dw: tensor<384x384x7x7xf16>,
    %w_s2_b3_pw1: tensor<1536x384x1x1xf16>,
    %w_s2_b3_pw2: tensor<384x1536x1x1xf16>,
    %w_s2_b4_dw: tensor<384x384x7x7xf16>,
    %w_s2_b4_pw1: tensor<1536x384x1x1xf16>,
    %w_s2_b4_pw2: tensor<384x1536x1x1xf16>,
    %w_s2_b5_dw: tensor<384x384x7x7xf16>,
    %w_s2_b5_pw1: tensor<1536x384x1x1xf16>,
    %w_s2_b5_pw2: tensor<384x1536x1x1xf16>,
    %w_s2_b6_dw: tensor<384x384x7x7xf16>,
    %w_s2_b6_pw1: tensor<1536x384x1x1xf16>,
    %w_s2_b6_pw2: tensor<384x1536x1x1xf16>,
    %w_s2_b7_dw: tensor<384x384x7x7xf16>,
    %w_s2_b7_pw1: tensor<1536x384x1x1xf16>,
    %w_s2_b7_pw2: tensor<384x1536x1x1xf16>,
    %w_s2_b8_dw: tensor<384x384x7x7xf16>,
    %w_s2_b8_pw1: tensor<1536x384x1x1xf16>,
    %w_s2_b8_pw2: tensor<384x1536x1x1xf16>,
    %w_ds3: tensor<768x384x3x3xf16>,
    %w_s3_b0_dw: tensor<768x768x7x7xf16>,
    %w_s3_b0_pw1: tensor<3072x768x1x1xf16>,
    %w_s3_b0_pw2: tensor<768x3072x1x1xf16>,
    %w_s3_b1_dw: tensor<768x768x7x7xf16>,
    %w_s3_b1_pw1: tensor<3072x768x1x1xf16>,
    %w_s3_b1_pw2: tensor<768x3072x1x1xf16>,
    %w_s3_b2_dw: tensor<768x768x7x7xf16>,
    %w_s3_b2_pw1: tensor<3072x768x1x1xf16>,
    %w_s3_b2_pw2: tensor<768x3072x1x1xf16>,
    %w_fc: tensor<1000x768x1x1xf16>) -> tensor<1x1000x7x7xf16> {
  %cst = arith.constant 0.0 : f16

  // Stem: 4x4 stride 4, 3->96
  %init0 = tensor.empty() : tensor<1x96x56x56xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init0 : tensor<1x96x56x56xf16>) -> tensor<1x96x56x56xf16>
  %conv2 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<4> : tensor<2xi64>
  } ins(%input, %w_stem : tensor<1x3x224x224xf16>, tensor<96x3x4x4xf16>)
    outs(%fill1 : tensor<1x96x56x56xf16>) -> tensor<1x96x56x56xf16>

  // === ConvNeXt Stage 0: dim=96 ===
  %pad3 = tensor.pad %conv2 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x96x56x56xf16> to tensor<1x96x62x62xf16>
  %init4 = tensor.empty() : tensor<1x96x56x56xf16>
  %fill5 = linalg.fill ins(%cst : f16) outs(%init4 : tensor<1x96x56x56xf16>) -> tensor<1x96x56x56xf16>
  %conv6 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad3, %w_s0_b0_dw : tensor<1x96x62x62xf16>, tensor<96x96x7x7xf16>)
    outs(%fill5 : tensor<1x96x56x56xf16>) -> tensor<1x96x56x56xf16>
  %init7 = tensor.empty() : tensor<1x384x56x56xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x384x56x56xf16>) -> tensor<1x384x56x56xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv6, %w_s0_b0_pw1 : tensor<1x96x56x56xf16>, tensor<384x96x1x1xf16>)
    outs(%fill8 : tensor<1x384x56x56xf16>) -> tensor<1x384x56x56xf16>
  %empty10 = tensor.empty() : tensor<1x384x56x56xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x384x56x56xf16>)
    outs(%empty10 : tensor<1x384x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x384x56x56xf16>
  %init12 = tensor.empty() : tensor<1x96x56x56xf16>
  %fill13 = linalg.fill ins(%cst : f16) outs(%init12 : tensor<1x96x56x56xf16>) -> tensor<1x96x56x56xf16>
  %conv14 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu11, %w_s0_b0_pw2 : tensor<1x384x56x56xf16>, tensor<96x384x1x1xf16>)
    outs(%fill13 : tensor<1x96x56x56xf16>) -> tensor<1x96x56x56xf16>
  %empty15 = tensor.empty() : tensor<1x96x56x56xf16>
  %add16 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv14, %conv2 : tensor<1x96x56x56xf16>, tensor<1x96x56x56xf16>)
    outs(%empty15 : tensor<1x96x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x96x56x56xf16>
  %pad17 = tensor.pad %add16 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x96x56x56xf16> to tensor<1x96x62x62xf16>
  %init18 = tensor.empty() : tensor<1x96x56x56xf16>
  %fill19 = linalg.fill ins(%cst : f16) outs(%init18 : tensor<1x96x56x56xf16>) -> tensor<1x96x56x56xf16>
  %conv20 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad17, %w_s0_b1_dw : tensor<1x96x62x62xf16>, tensor<96x96x7x7xf16>)
    outs(%fill19 : tensor<1x96x56x56xf16>) -> tensor<1x96x56x56xf16>
  %init21 = tensor.empty() : tensor<1x384x56x56xf16>
  %fill22 = linalg.fill ins(%cst : f16) outs(%init21 : tensor<1x384x56x56xf16>) -> tensor<1x384x56x56xf16>
  %conv23 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv20, %w_s0_b1_pw1 : tensor<1x96x56x56xf16>, tensor<384x96x1x1xf16>)
    outs(%fill22 : tensor<1x384x56x56xf16>) -> tensor<1x384x56x56xf16>
  %empty24 = tensor.empty() : tensor<1x384x56x56xf16>
  %relu25 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv23 : tensor<1x384x56x56xf16>)
    outs(%empty24 : tensor<1x384x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x384x56x56xf16>
  %init26 = tensor.empty() : tensor<1x96x56x56xf16>
  %fill27 = linalg.fill ins(%cst : f16) outs(%init26 : tensor<1x96x56x56xf16>) -> tensor<1x96x56x56xf16>
  %conv28 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu25, %w_s0_b1_pw2 : tensor<1x384x56x56xf16>, tensor<96x384x1x1xf16>)
    outs(%fill27 : tensor<1x96x56x56xf16>) -> tensor<1x96x56x56xf16>
  %empty29 = tensor.empty() : tensor<1x96x56x56xf16>
  %add30 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv28, %add16 : tensor<1x96x56x56xf16>, tensor<1x96x56x56xf16>)
    outs(%empty29 : tensor<1x96x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x96x56x56xf16>
  %pad31 = tensor.pad %add30 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x96x56x56xf16> to tensor<1x96x62x62xf16>
  %init32 = tensor.empty() : tensor<1x96x56x56xf16>
  %fill33 = linalg.fill ins(%cst : f16) outs(%init32 : tensor<1x96x56x56xf16>) -> tensor<1x96x56x56xf16>
  %conv34 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad31, %w_s0_b2_dw : tensor<1x96x62x62xf16>, tensor<96x96x7x7xf16>)
    outs(%fill33 : tensor<1x96x56x56xf16>) -> tensor<1x96x56x56xf16>
  %init35 = tensor.empty() : tensor<1x384x56x56xf16>
  %fill36 = linalg.fill ins(%cst : f16) outs(%init35 : tensor<1x384x56x56xf16>) -> tensor<1x384x56x56xf16>
  %conv37 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv34, %w_s0_b2_pw1 : tensor<1x96x56x56xf16>, tensor<384x96x1x1xf16>)
    outs(%fill36 : tensor<1x384x56x56xf16>) -> tensor<1x384x56x56xf16>
  %empty38 = tensor.empty() : tensor<1x384x56x56xf16>
  %relu39 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv37 : tensor<1x384x56x56xf16>)
    outs(%empty38 : tensor<1x384x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x384x56x56xf16>
  %init40 = tensor.empty() : tensor<1x96x56x56xf16>
  %fill41 = linalg.fill ins(%cst : f16) outs(%init40 : tensor<1x96x56x56xf16>) -> tensor<1x96x56x56xf16>
  %conv42 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu39, %w_s0_b2_pw2 : tensor<1x384x56x56xf16>, tensor<96x384x1x1xf16>)
    outs(%fill41 : tensor<1x96x56x56xf16>) -> tensor<1x96x56x56xf16>
  %empty43 = tensor.empty() : tensor<1x96x56x56xf16>
  %add44 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv42, %add30 : tensor<1x96x56x56xf16>, tensor<1x96x56x56xf16>)
    outs(%empty43 : tensor<1x96x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x96x56x56xf16>

  // Downsample: 96->192, stride 2
  %pad45 = tensor.pad %add44 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x96x56x56xf16> to tensor<1x96x58x58xf16>
  %init46 = tensor.empty() : tensor<1x192x28x28xf16>
  %fill47 = linalg.fill ins(%cst : f16) outs(%init46 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %conv48 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad45, %w_ds1 : tensor<1x96x58x58xf16>, tensor<192x96x3x3xf16>)
    outs(%fill47 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>

  // === ConvNeXt Stage 1: dim=192 ===
  %pad49 = tensor.pad %conv48 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x192x28x28xf16> to tensor<1x192x34x34xf16>
  %init50 = tensor.empty() : tensor<1x192x28x28xf16>
  %fill51 = linalg.fill ins(%cst : f16) outs(%init50 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %conv52 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad49, %w_s1_b0_dw : tensor<1x192x34x34xf16>, tensor<192x192x7x7xf16>)
    outs(%fill51 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %init53 = tensor.empty() : tensor<1x768x28x28xf16>
  %fill54 = linalg.fill ins(%cst : f16) outs(%init53 : tensor<1x768x28x28xf16>) -> tensor<1x768x28x28xf16>
  %conv55 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv52, %w_s1_b0_pw1 : tensor<1x192x28x28xf16>, tensor<768x192x1x1xf16>)
    outs(%fill54 : tensor<1x768x28x28xf16>) -> tensor<1x768x28x28xf16>
  %empty56 = tensor.empty() : tensor<1x768x28x28xf16>
  %relu57 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv55 : tensor<1x768x28x28xf16>)
    outs(%empty56 : tensor<1x768x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x28x28xf16>
  %init58 = tensor.empty() : tensor<1x192x28x28xf16>
  %fill59 = linalg.fill ins(%cst : f16) outs(%init58 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %conv60 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu57, %w_s1_b0_pw2 : tensor<1x768x28x28xf16>, tensor<192x768x1x1xf16>)
    outs(%fill59 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %empty61 = tensor.empty() : tensor<1x192x28x28xf16>
  %add62 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv60, %conv48 : tensor<1x192x28x28xf16>, tensor<1x192x28x28xf16>)
    outs(%empty61 : tensor<1x192x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x192x28x28xf16>
  %pad63 = tensor.pad %add62 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x192x28x28xf16> to tensor<1x192x34x34xf16>
  %init64 = tensor.empty() : tensor<1x192x28x28xf16>
  %fill65 = linalg.fill ins(%cst : f16) outs(%init64 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %conv66 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad63, %w_s1_b1_dw : tensor<1x192x34x34xf16>, tensor<192x192x7x7xf16>)
    outs(%fill65 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %init67 = tensor.empty() : tensor<1x768x28x28xf16>
  %fill68 = linalg.fill ins(%cst : f16) outs(%init67 : tensor<1x768x28x28xf16>) -> tensor<1x768x28x28xf16>
  %conv69 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv66, %w_s1_b1_pw1 : tensor<1x192x28x28xf16>, tensor<768x192x1x1xf16>)
    outs(%fill68 : tensor<1x768x28x28xf16>) -> tensor<1x768x28x28xf16>
  %empty70 = tensor.empty() : tensor<1x768x28x28xf16>
  %relu71 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv69 : tensor<1x768x28x28xf16>)
    outs(%empty70 : tensor<1x768x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x28x28xf16>
  %init72 = tensor.empty() : tensor<1x192x28x28xf16>
  %fill73 = linalg.fill ins(%cst : f16) outs(%init72 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %conv74 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu71, %w_s1_b1_pw2 : tensor<1x768x28x28xf16>, tensor<192x768x1x1xf16>)
    outs(%fill73 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %empty75 = tensor.empty() : tensor<1x192x28x28xf16>
  %add76 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv74, %add62 : tensor<1x192x28x28xf16>, tensor<1x192x28x28xf16>)
    outs(%empty75 : tensor<1x192x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x192x28x28xf16>
  %pad77 = tensor.pad %add76 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x192x28x28xf16> to tensor<1x192x34x34xf16>
  %init78 = tensor.empty() : tensor<1x192x28x28xf16>
  %fill79 = linalg.fill ins(%cst : f16) outs(%init78 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %conv80 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad77, %w_s1_b2_dw : tensor<1x192x34x34xf16>, tensor<192x192x7x7xf16>)
    outs(%fill79 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %init81 = tensor.empty() : tensor<1x768x28x28xf16>
  %fill82 = linalg.fill ins(%cst : f16) outs(%init81 : tensor<1x768x28x28xf16>) -> tensor<1x768x28x28xf16>
  %conv83 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv80, %w_s1_b2_pw1 : tensor<1x192x28x28xf16>, tensor<768x192x1x1xf16>)
    outs(%fill82 : tensor<1x768x28x28xf16>) -> tensor<1x768x28x28xf16>
  %empty84 = tensor.empty() : tensor<1x768x28x28xf16>
  %relu85 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv83 : tensor<1x768x28x28xf16>)
    outs(%empty84 : tensor<1x768x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x28x28xf16>
  %init86 = tensor.empty() : tensor<1x192x28x28xf16>
  %fill87 = linalg.fill ins(%cst : f16) outs(%init86 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %conv88 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu85, %w_s1_b2_pw2 : tensor<1x768x28x28xf16>, tensor<192x768x1x1xf16>)
    outs(%fill87 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %empty89 = tensor.empty() : tensor<1x192x28x28xf16>
  %add90 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv88, %add76 : tensor<1x192x28x28xf16>, tensor<1x192x28x28xf16>)
    outs(%empty89 : tensor<1x192x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x192x28x28xf16>

  // Downsample: 192->384, stride 2
  %pad91 = tensor.pad %add90 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x192x28x28xf16> to tensor<1x192x30x30xf16>
  %init92 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill93 = linalg.fill ins(%cst : f16) outs(%init92 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv94 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad91, %w_ds2 : tensor<1x192x30x30xf16>, tensor<384x192x3x3xf16>)
    outs(%fill93 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>

  // === ConvNeXt Stage 2: dim=384 ===
  %pad95 = tensor.pad %conv94 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x384x14x14xf16> to tensor<1x384x20x20xf16>
  %init96 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill97 = linalg.fill ins(%cst : f16) outs(%init96 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv98 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad95, %w_s2_b0_dw : tensor<1x384x20x20xf16>, tensor<384x384x7x7xf16>)
    outs(%fill97 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %init99 = tensor.empty() : tensor<1x1536x14x14xf16>
  %fill100 = linalg.fill ins(%cst : f16) outs(%init99 : tensor<1x1536x14x14xf16>) -> tensor<1x1536x14x14xf16>
  %conv101 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv98, %w_s2_b0_pw1 : tensor<1x384x14x14xf16>, tensor<1536x384x1x1xf16>)
    outs(%fill100 : tensor<1x1536x14x14xf16>) -> tensor<1x1536x14x14xf16>
  %empty102 = tensor.empty() : tensor<1x1536x14x14xf16>
  %relu103 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv101 : tensor<1x1536x14x14xf16>)
    outs(%empty102 : tensor<1x1536x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1536x14x14xf16>
  %init104 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill105 = linalg.fill ins(%cst : f16) outs(%init104 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv106 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu103, %w_s2_b0_pw2 : tensor<1x1536x14x14xf16>, tensor<384x1536x1x1xf16>)
    outs(%fill105 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %empty107 = tensor.empty() : tensor<1x384x14x14xf16>
  %add108 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv106, %conv94 : tensor<1x384x14x14xf16>, tensor<1x384x14x14xf16>)
    outs(%empty107 : tensor<1x384x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x384x14x14xf16>
  %pad109 = tensor.pad %add108 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x384x14x14xf16> to tensor<1x384x20x20xf16>
  %init110 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill111 = linalg.fill ins(%cst : f16) outs(%init110 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv112 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad109, %w_s2_b1_dw : tensor<1x384x20x20xf16>, tensor<384x384x7x7xf16>)
    outs(%fill111 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %init113 = tensor.empty() : tensor<1x1536x14x14xf16>
  %fill114 = linalg.fill ins(%cst : f16) outs(%init113 : tensor<1x1536x14x14xf16>) -> tensor<1x1536x14x14xf16>
  %conv115 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv112, %w_s2_b1_pw1 : tensor<1x384x14x14xf16>, tensor<1536x384x1x1xf16>)
    outs(%fill114 : tensor<1x1536x14x14xf16>) -> tensor<1x1536x14x14xf16>
  %empty116 = tensor.empty() : tensor<1x1536x14x14xf16>
  %relu117 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv115 : tensor<1x1536x14x14xf16>)
    outs(%empty116 : tensor<1x1536x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1536x14x14xf16>
  %init118 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill119 = linalg.fill ins(%cst : f16) outs(%init118 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv120 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu117, %w_s2_b1_pw2 : tensor<1x1536x14x14xf16>, tensor<384x1536x1x1xf16>)
    outs(%fill119 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %empty121 = tensor.empty() : tensor<1x384x14x14xf16>
  %add122 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv120, %add108 : tensor<1x384x14x14xf16>, tensor<1x384x14x14xf16>)
    outs(%empty121 : tensor<1x384x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x384x14x14xf16>
  %pad123 = tensor.pad %add122 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x384x14x14xf16> to tensor<1x384x20x20xf16>
  %init124 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill125 = linalg.fill ins(%cst : f16) outs(%init124 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv126 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad123, %w_s2_b2_dw : tensor<1x384x20x20xf16>, tensor<384x384x7x7xf16>)
    outs(%fill125 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %init127 = tensor.empty() : tensor<1x1536x14x14xf16>
  %fill128 = linalg.fill ins(%cst : f16) outs(%init127 : tensor<1x1536x14x14xf16>) -> tensor<1x1536x14x14xf16>
  %conv129 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv126, %w_s2_b2_pw1 : tensor<1x384x14x14xf16>, tensor<1536x384x1x1xf16>)
    outs(%fill128 : tensor<1x1536x14x14xf16>) -> tensor<1x1536x14x14xf16>
  %empty130 = tensor.empty() : tensor<1x1536x14x14xf16>
  %relu131 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv129 : tensor<1x1536x14x14xf16>)
    outs(%empty130 : tensor<1x1536x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1536x14x14xf16>
  %init132 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill133 = linalg.fill ins(%cst : f16) outs(%init132 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv134 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu131, %w_s2_b2_pw2 : tensor<1x1536x14x14xf16>, tensor<384x1536x1x1xf16>)
    outs(%fill133 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %empty135 = tensor.empty() : tensor<1x384x14x14xf16>
  %add136 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv134, %add122 : tensor<1x384x14x14xf16>, tensor<1x384x14x14xf16>)
    outs(%empty135 : tensor<1x384x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x384x14x14xf16>
  %pad137 = tensor.pad %add136 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x384x14x14xf16> to tensor<1x384x20x20xf16>
  %init138 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill139 = linalg.fill ins(%cst : f16) outs(%init138 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv140 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad137, %w_s2_b3_dw : tensor<1x384x20x20xf16>, tensor<384x384x7x7xf16>)
    outs(%fill139 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %init141 = tensor.empty() : tensor<1x1536x14x14xf16>
  %fill142 = linalg.fill ins(%cst : f16) outs(%init141 : tensor<1x1536x14x14xf16>) -> tensor<1x1536x14x14xf16>
  %conv143 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv140, %w_s2_b3_pw1 : tensor<1x384x14x14xf16>, tensor<1536x384x1x1xf16>)
    outs(%fill142 : tensor<1x1536x14x14xf16>) -> tensor<1x1536x14x14xf16>
  %empty144 = tensor.empty() : tensor<1x1536x14x14xf16>
  %relu145 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv143 : tensor<1x1536x14x14xf16>)
    outs(%empty144 : tensor<1x1536x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1536x14x14xf16>
  %init146 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill147 = linalg.fill ins(%cst : f16) outs(%init146 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv148 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu145, %w_s2_b3_pw2 : tensor<1x1536x14x14xf16>, tensor<384x1536x1x1xf16>)
    outs(%fill147 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %empty149 = tensor.empty() : tensor<1x384x14x14xf16>
  %add150 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv148, %add136 : tensor<1x384x14x14xf16>, tensor<1x384x14x14xf16>)
    outs(%empty149 : tensor<1x384x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x384x14x14xf16>
  %pad151 = tensor.pad %add150 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x384x14x14xf16> to tensor<1x384x20x20xf16>
  %init152 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill153 = linalg.fill ins(%cst : f16) outs(%init152 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv154 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad151, %w_s2_b4_dw : tensor<1x384x20x20xf16>, tensor<384x384x7x7xf16>)
    outs(%fill153 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %init155 = tensor.empty() : tensor<1x1536x14x14xf16>
  %fill156 = linalg.fill ins(%cst : f16) outs(%init155 : tensor<1x1536x14x14xf16>) -> tensor<1x1536x14x14xf16>
  %conv157 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv154, %w_s2_b4_pw1 : tensor<1x384x14x14xf16>, tensor<1536x384x1x1xf16>)
    outs(%fill156 : tensor<1x1536x14x14xf16>) -> tensor<1x1536x14x14xf16>
  %empty158 = tensor.empty() : tensor<1x1536x14x14xf16>
  %relu159 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv157 : tensor<1x1536x14x14xf16>)
    outs(%empty158 : tensor<1x1536x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1536x14x14xf16>
  %init160 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill161 = linalg.fill ins(%cst : f16) outs(%init160 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv162 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu159, %w_s2_b4_pw2 : tensor<1x1536x14x14xf16>, tensor<384x1536x1x1xf16>)
    outs(%fill161 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %empty163 = tensor.empty() : tensor<1x384x14x14xf16>
  %add164 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv162, %add150 : tensor<1x384x14x14xf16>, tensor<1x384x14x14xf16>)
    outs(%empty163 : tensor<1x384x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x384x14x14xf16>
  %pad165 = tensor.pad %add164 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x384x14x14xf16> to tensor<1x384x20x20xf16>
  %init166 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill167 = linalg.fill ins(%cst : f16) outs(%init166 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv168 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad165, %w_s2_b5_dw : tensor<1x384x20x20xf16>, tensor<384x384x7x7xf16>)
    outs(%fill167 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %init169 = tensor.empty() : tensor<1x1536x14x14xf16>
  %fill170 = linalg.fill ins(%cst : f16) outs(%init169 : tensor<1x1536x14x14xf16>) -> tensor<1x1536x14x14xf16>
  %conv171 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv168, %w_s2_b5_pw1 : tensor<1x384x14x14xf16>, tensor<1536x384x1x1xf16>)
    outs(%fill170 : tensor<1x1536x14x14xf16>) -> tensor<1x1536x14x14xf16>
  %empty172 = tensor.empty() : tensor<1x1536x14x14xf16>
  %relu173 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv171 : tensor<1x1536x14x14xf16>)
    outs(%empty172 : tensor<1x1536x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1536x14x14xf16>
  %init174 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill175 = linalg.fill ins(%cst : f16) outs(%init174 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv176 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu173, %w_s2_b5_pw2 : tensor<1x1536x14x14xf16>, tensor<384x1536x1x1xf16>)
    outs(%fill175 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %empty177 = tensor.empty() : tensor<1x384x14x14xf16>
  %add178 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv176, %add164 : tensor<1x384x14x14xf16>, tensor<1x384x14x14xf16>)
    outs(%empty177 : tensor<1x384x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x384x14x14xf16>
  %pad179 = tensor.pad %add178 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x384x14x14xf16> to tensor<1x384x20x20xf16>
  %init180 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill181 = linalg.fill ins(%cst : f16) outs(%init180 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv182 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad179, %w_s2_b6_dw : tensor<1x384x20x20xf16>, tensor<384x384x7x7xf16>)
    outs(%fill181 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %init183 = tensor.empty() : tensor<1x1536x14x14xf16>
  %fill184 = linalg.fill ins(%cst : f16) outs(%init183 : tensor<1x1536x14x14xf16>) -> tensor<1x1536x14x14xf16>
  %conv185 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv182, %w_s2_b6_pw1 : tensor<1x384x14x14xf16>, tensor<1536x384x1x1xf16>)
    outs(%fill184 : tensor<1x1536x14x14xf16>) -> tensor<1x1536x14x14xf16>
  %empty186 = tensor.empty() : tensor<1x1536x14x14xf16>
  %relu187 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv185 : tensor<1x1536x14x14xf16>)
    outs(%empty186 : tensor<1x1536x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1536x14x14xf16>
  %init188 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill189 = linalg.fill ins(%cst : f16) outs(%init188 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv190 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu187, %w_s2_b6_pw2 : tensor<1x1536x14x14xf16>, tensor<384x1536x1x1xf16>)
    outs(%fill189 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %empty191 = tensor.empty() : tensor<1x384x14x14xf16>
  %add192 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv190, %add178 : tensor<1x384x14x14xf16>, tensor<1x384x14x14xf16>)
    outs(%empty191 : tensor<1x384x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x384x14x14xf16>
  %pad193 = tensor.pad %add192 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x384x14x14xf16> to tensor<1x384x20x20xf16>
  %init194 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill195 = linalg.fill ins(%cst : f16) outs(%init194 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv196 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad193, %w_s2_b7_dw : tensor<1x384x20x20xf16>, tensor<384x384x7x7xf16>)
    outs(%fill195 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %init197 = tensor.empty() : tensor<1x1536x14x14xf16>
  %fill198 = linalg.fill ins(%cst : f16) outs(%init197 : tensor<1x1536x14x14xf16>) -> tensor<1x1536x14x14xf16>
  %conv199 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv196, %w_s2_b7_pw1 : tensor<1x384x14x14xf16>, tensor<1536x384x1x1xf16>)
    outs(%fill198 : tensor<1x1536x14x14xf16>) -> tensor<1x1536x14x14xf16>
  %empty200 = tensor.empty() : tensor<1x1536x14x14xf16>
  %relu201 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv199 : tensor<1x1536x14x14xf16>)
    outs(%empty200 : tensor<1x1536x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1536x14x14xf16>
  %init202 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill203 = linalg.fill ins(%cst : f16) outs(%init202 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv204 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu201, %w_s2_b7_pw2 : tensor<1x1536x14x14xf16>, tensor<384x1536x1x1xf16>)
    outs(%fill203 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %empty205 = tensor.empty() : tensor<1x384x14x14xf16>
  %add206 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv204, %add192 : tensor<1x384x14x14xf16>, tensor<1x384x14x14xf16>)
    outs(%empty205 : tensor<1x384x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x384x14x14xf16>
  %pad207 = tensor.pad %add206 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x384x14x14xf16> to tensor<1x384x20x20xf16>
  %init208 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill209 = linalg.fill ins(%cst : f16) outs(%init208 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv210 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad207, %w_s2_b8_dw : tensor<1x384x20x20xf16>, tensor<384x384x7x7xf16>)
    outs(%fill209 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %init211 = tensor.empty() : tensor<1x1536x14x14xf16>
  %fill212 = linalg.fill ins(%cst : f16) outs(%init211 : tensor<1x1536x14x14xf16>) -> tensor<1x1536x14x14xf16>
  %conv213 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv210, %w_s2_b8_pw1 : tensor<1x384x14x14xf16>, tensor<1536x384x1x1xf16>)
    outs(%fill212 : tensor<1x1536x14x14xf16>) -> tensor<1x1536x14x14xf16>
  %empty214 = tensor.empty() : tensor<1x1536x14x14xf16>
  %relu215 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv213 : tensor<1x1536x14x14xf16>)
    outs(%empty214 : tensor<1x1536x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1536x14x14xf16>
  %init216 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill217 = linalg.fill ins(%cst : f16) outs(%init216 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv218 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu215, %w_s2_b8_pw2 : tensor<1x1536x14x14xf16>, tensor<384x1536x1x1xf16>)
    outs(%fill217 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %empty219 = tensor.empty() : tensor<1x384x14x14xf16>
  %add220 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv218, %add206 : tensor<1x384x14x14xf16>, tensor<1x384x14x14xf16>)
    outs(%empty219 : tensor<1x384x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x384x14x14xf16>

  // Downsample: 384->768, stride 2
  %pad221 = tensor.pad %add220 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x384x14x14xf16> to tensor<1x384x16x16xf16>
  %init222 = tensor.empty() : tensor<1x768x7x7xf16>
  %fill223 = linalg.fill ins(%cst : f16) outs(%init222 : tensor<1x768x7x7xf16>) -> tensor<1x768x7x7xf16>
  %conv224 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad221, %w_ds3 : tensor<1x384x16x16xf16>, tensor<768x384x3x3xf16>)
    outs(%fill223 : tensor<1x768x7x7xf16>) -> tensor<1x768x7x7xf16>

  // === ConvNeXt Stage 3: dim=768 ===
  %pad225 = tensor.pad %conv224 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x768x7x7xf16> to tensor<1x768x13x13xf16>
  %init226 = tensor.empty() : tensor<1x768x7x7xf16>
  %fill227 = linalg.fill ins(%cst : f16) outs(%init226 : tensor<1x768x7x7xf16>) -> tensor<1x768x7x7xf16>
  %conv228 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad225, %w_s3_b0_dw : tensor<1x768x13x13xf16>, tensor<768x768x7x7xf16>)
    outs(%fill227 : tensor<1x768x7x7xf16>) -> tensor<1x768x7x7xf16>
  %init229 = tensor.empty() : tensor<1x3072x7x7xf16>
  %fill230 = linalg.fill ins(%cst : f16) outs(%init229 : tensor<1x3072x7x7xf16>) -> tensor<1x3072x7x7xf16>
  %conv231 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv228, %w_s3_b0_pw1 : tensor<1x768x7x7xf16>, tensor<3072x768x1x1xf16>)
    outs(%fill230 : tensor<1x3072x7x7xf16>) -> tensor<1x3072x7x7xf16>
  %empty232 = tensor.empty() : tensor<1x3072x7x7xf16>
  %relu233 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv231 : tensor<1x3072x7x7xf16>)
    outs(%empty232 : tensor<1x3072x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x3072x7x7xf16>
  %init234 = tensor.empty() : tensor<1x768x7x7xf16>
  %fill235 = linalg.fill ins(%cst : f16) outs(%init234 : tensor<1x768x7x7xf16>) -> tensor<1x768x7x7xf16>
  %conv236 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu233, %w_s3_b0_pw2 : tensor<1x3072x7x7xf16>, tensor<768x3072x1x1xf16>)
    outs(%fill235 : tensor<1x768x7x7xf16>) -> tensor<1x768x7x7xf16>
  %empty237 = tensor.empty() : tensor<1x768x7x7xf16>
  %add238 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv236, %conv224 : tensor<1x768x7x7xf16>, tensor<1x768x7x7xf16>)
    outs(%empty237 : tensor<1x768x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x768x7x7xf16>
  %pad239 = tensor.pad %add238 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x768x7x7xf16> to tensor<1x768x13x13xf16>
  %init240 = tensor.empty() : tensor<1x768x7x7xf16>
  %fill241 = linalg.fill ins(%cst : f16) outs(%init240 : tensor<1x768x7x7xf16>) -> tensor<1x768x7x7xf16>
  %conv242 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad239, %w_s3_b1_dw : tensor<1x768x13x13xf16>, tensor<768x768x7x7xf16>)
    outs(%fill241 : tensor<1x768x7x7xf16>) -> tensor<1x768x7x7xf16>
  %init243 = tensor.empty() : tensor<1x3072x7x7xf16>
  %fill244 = linalg.fill ins(%cst : f16) outs(%init243 : tensor<1x3072x7x7xf16>) -> tensor<1x3072x7x7xf16>
  %conv245 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv242, %w_s3_b1_pw1 : tensor<1x768x7x7xf16>, tensor<3072x768x1x1xf16>)
    outs(%fill244 : tensor<1x3072x7x7xf16>) -> tensor<1x3072x7x7xf16>
  %empty246 = tensor.empty() : tensor<1x3072x7x7xf16>
  %relu247 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv245 : tensor<1x3072x7x7xf16>)
    outs(%empty246 : tensor<1x3072x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x3072x7x7xf16>
  %init248 = tensor.empty() : tensor<1x768x7x7xf16>
  %fill249 = linalg.fill ins(%cst : f16) outs(%init248 : tensor<1x768x7x7xf16>) -> tensor<1x768x7x7xf16>
  %conv250 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu247, %w_s3_b1_pw2 : tensor<1x3072x7x7xf16>, tensor<768x3072x1x1xf16>)
    outs(%fill249 : tensor<1x768x7x7xf16>) -> tensor<1x768x7x7xf16>
  %empty251 = tensor.empty() : tensor<1x768x7x7xf16>
  %add252 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv250, %add238 : tensor<1x768x7x7xf16>, tensor<1x768x7x7xf16>)
    outs(%empty251 : tensor<1x768x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x768x7x7xf16>
  %pad253 = tensor.pad %add252 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x768x7x7xf16> to tensor<1x768x13x13xf16>
  %init254 = tensor.empty() : tensor<1x768x7x7xf16>
  %fill255 = linalg.fill ins(%cst : f16) outs(%init254 : tensor<1x768x7x7xf16>) -> tensor<1x768x7x7xf16>
  %conv256 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad253, %w_s3_b2_dw : tensor<1x768x13x13xf16>, tensor<768x768x7x7xf16>)
    outs(%fill255 : tensor<1x768x7x7xf16>) -> tensor<1x768x7x7xf16>
  %init257 = tensor.empty() : tensor<1x3072x7x7xf16>
  %fill258 = linalg.fill ins(%cst : f16) outs(%init257 : tensor<1x3072x7x7xf16>) -> tensor<1x3072x7x7xf16>
  %conv259 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv256, %w_s3_b2_pw1 : tensor<1x768x7x7xf16>, tensor<3072x768x1x1xf16>)
    outs(%fill258 : tensor<1x3072x7x7xf16>) -> tensor<1x3072x7x7xf16>
  %empty260 = tensor.empty() : tensor<1x3072x7x7xf16>
  %relu261 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv259 : tensor<1x3072x7x7xf16>)
    outs(%empty260 : tensor<1x3072x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x3072x7x7xf16>
  %init262 = tensor.empty() : tensor<1x768x7x7xf16>
  %fill263 = linalg.fill ins(%cst : f16) outs(%init262 : tensor<1x768x7x7xf16>) -> tensor<1x768x7x7xf16>
  %conv264 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu261, %w_s3_b2_pw2 : tensor<1x3072x7x7xf16>, tensor<768x3072x1x1xf16>)
    outs(%fill263 : tensor<1x768x7x7xf16>) -> tensor<1x768x7x7xf16>
  %empty265 = tensor.empty() : tensor<1x768x7x7xf16>
  %add266 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv264, %add252 : tensor<1x768x7x7xf16>, tensor<1x768x7x7xf16>)
    outs(%empty265 : tensor<1x768x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x768x7x7xf16>

  // FC: 1x1 768->1000
  %init267 = tensor.empty() : tensor<1x1000x7x7xf16>
  %fill268 = linalg.fill ins(%cst : f16) outs(%init267 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  %conv269 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add266, %w_fc : tensor<1x768x7x7xf16>, tensor<1000x768x1x1xf16>)
    outs(%fill268 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  return %conv269 : tensor<1x1000x7x7xf16>
}
