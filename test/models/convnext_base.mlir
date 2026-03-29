func.func @convnext_base(
    %input: tensor<1x3x224x224xf16>,
    %w_stem: tensor<128x3x4x4xf16>,
    %w_s0_b0_dw: tensor<128x128x7x7xf16>,
    %w_s0_b0_pw1: tensor<512x128x1x1xf16>,
    %w_s0_b0_pw2: tensor<128x512x1x1xf16>,
    %w_s0_b1_dw: tensor<128x128x7x7xf16>,
    %w_s0_b1_pw1: tensor<512x128x1x1xf16>,
    %w_s0_b1_pw2: tensor<128x512x1x1xf16>,
    %w_s0_b2_dw: tensor<128x128x7x7xf16>,
    %w_s0_b2_pw1: tensor<512x128x1x1xf16>,
    %w_s0_b2_pw2: tensor<128x512x1x1xf16>,
    %w_ds1: tensor<256x128x3x3xf16>,
    %w_s1_b0_dw: tensor<256x256x7x7xf16>,
    %w_s1_b0_pw1: tensor<1024x256x1x1xf16>,
    %w_s1_b0_pw2: tensor<256x1024x1x1xf16>,
    %w_s1_b1_dw: tensor<256x256x7x7xf16>,
    %w_s1_b1_pw1: tensor<1024x256x1x1xf16>,
    %w_s1_b1_pw2: tensor<256x1024x1x1xf16>,
    %w_s1_b2_dw: tensor<256x256x7x7xf16>,
    %w_s1_b2_pw1: tensor<1024x256x1x1xf16>,
    %w_s1_b2_pw2: tensor<256x1024x1x1xf16>,
    %w_ds2: tensor<512x256x3x3xf16>,
    %w_s2_b0_dw: tensor<512x512x7x7xf16>,
    %w_s2_b0_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b0_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b1_dw: tensor<512x512x7x7xf16>,
    %w_s2_b1_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b1_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b2_dw: tensor<512x512x7x7xf16>,
    %w_s2_b2_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b2_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b3_dw: tensor<512x512x7x7xf16>,
    %w_s2_b3_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b3_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b4_dw: tensor<512x512x7x7xf16>,
    %w_s2_b4_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b4_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b5_dw: tensor<512x512x7x7xf16>,
    %w_s2_b5_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b5_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b6_dw: tensor<512x512x7x7xf16>,
    %w_s2_b6_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b6_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b7_dw: tensor<512x512x7x7xf16>,
    %w_s2_b7_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b7_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b8_dw: tensor<512x512x7x7xf16>,
    %w_s2_b8_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b8_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b9_dw: tensor<512x512x7x7xf16>,
    %w_s2_b9_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b9_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b10_dw: tensor<512x512x7x7xf16>,
    %w_s2_b10_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b10_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b11_dw: tensor<512x512x7x7xf16>,
    %w_s2_b11_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b11_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b12_dw: tensor<512x512x7x7xf16>,
    %w_s2_b12_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b12_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b13_dw: tensor<512x512x7x7xf16>,
    %w_s2_b13_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b13_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b14_dw: tensor<512x512x7x7xf16>,
    %w_s2_b14_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b14_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b15_dw: tensor<512x512x7x7xf16>,
    %w_s2_b15_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b15_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b16_dw: tensor<512x512x7x7xf16>,
    %w_s2_b16_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b16_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b17_dw: tensor<512x512x7x7xf16>,
    %w_s2_b17_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b17_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b18_dw: tensor<512x512x7x7xf16>,
    %w_s2_b18_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b18_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b19_dw: tensor<512x512x7x7xf16>,
    %w_s2_b19_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b19_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b20_dw: tensor<512x512x7x7xf16>,
    %w_s2_b20_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b20_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b21_dw: tensor<512x512x7x7xf16>,
    %w_s2_b21_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b21_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b22_dw: tensor<512x512x7x7xf16>,
    %w_s2_b22_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b22_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b23_dw: tensor<512x512x7x7xf16>,
    %w_s2_b23_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b23_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b24_dw: tensor<512x512x7x7xf16>,
    %w_s2_b24_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b24_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b25_dw: tensor<512x512x7x7xf16>,
    %w_s2_b25_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b25_pw2: tensor<512x2048x1x1xf16>,
    %w_s2_b26_dw: tensor<512x512x7x7xf16>,
    %w_s2_b26_pw1: tensor<2048x512x1x1xf16>,
    %w_s2_b26_pw2: tensor<512x2048x1x1xf16>,
    %w_ds3: tensor<1024x512x3x3xf16>,
    %w_s3_b0_dw: tensor<1024x1024x7x7xf16>,
    %w_s3_b0_pw1: tensor<4096x1024x1x1xf16>,
    %w_s3_b0_pw2: tensor<1024x4096x1x1xf16>,
    %w_s3_b1_dw: tensor<1024x1024x7x7xf16>,
    %w_s3_b1_pw1: tensor<4096x1024x1x1xf16>,
    %w_s3_b1_pw2: tensor<1024x4096x1x1xf16>,
    %w_s3_b2_dw: tensor<1024x1024x7x7xf16>,
    %w_s3_b2_pw1: tensor<4096x1024x1x1xf16>,
    %w_s3_b2_pw2: tensor<1024x4096x1x1xf16>,
    %w_fc: tensor<1000x1024x1x1xf16>) -> tensor<1x1000x7x7xf16> {
  %cst = arith.constant 0.0 : f16

  // Stem: 4x4 stride 4, 3->128
  %init0 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init0 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv2 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<4> : tensor<2xi64>
  } ins(%input, %w_stem : tensor<1x3x224x224xf16>, tensor<128x3x4x4xf16>)
    outs(%fill1 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>

  // === ConvNeXt Stage 0: dim=128 ===
  %pad3 = tensor.pad %conv2 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x56x56xf16> to tensor<1x128x62x62xf16>
  %init4 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill5 = linalg.fill ins(%cst : f16) outs(%init4 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv6 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad3, %w_s0_b0_dw : tensor<1x128x62x62xf16>, tensor<128x128x7x7xf16>)
    outs(%fill5 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %init7 = tensor.empty() : tensor<1x512x56x56xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x512x56x56xf16>) -> tensor<1x512x56x56xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv6, %w_s0_b0_pw1 : tensor<1x128x56x56xf16>, tensor<512x128x1x1xf16>)
    outs(%fill8 : tensor<1x512x56x56xf16>) -> tensor<1x512x56x56xf16>
  %empty10 = tensor.empty() : tensor<1x512x56x56xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x512x56x56xf16>)
    outs(%empty10 : tensor<1x512x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x56x56xf16>
  %init12 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill13 = linalg.fill ins(%cst : f16) outs(%init12 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv14 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu11, %w_s0_b0_pw2 : tensor<1x512x56x56xf16>, tensor<128x512x1x1xf16>)
    outs(%fill13 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %empty15 = tensor.empty() : tensor<1x128x56x56xf16>
  %add16 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv14, %conv2 : tensor<1x128x56x56xf16>, tensor<1x128x56x56xf16>)
    outs(%empty15 : tensor<1x128x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x56x56xf16>
  %pad17 = tensor.pad %add16 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x56x56xf16> to tensor<1x128x62x62xf16>
  %init18 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill19 = linalg.fill ins(%cst : f16) outs(%init18 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv20 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad17, %w_s0_b1_dw : tensor<1x128x62x62xf16>, tensor<128x128x7x7xf16>)
    outs(%fill19 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %init21 = tensor.empty() : tensor<1x512x56x56xf16>
  %fill22 = linalg.fill ins(%cst : f16) outs(%init21 : tensor<1x512x56x56xf16>) -> tensor<1x512x56x56xf16>
  %conv23 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv20, %w_s0_b1_pw1 : tensor<1x128x56x56xf16>, tensor<512x128x1x1xf16>)
    outs(%fill22 : tensor<1x512x56x56xf16>) -> tensor<1x512x56x56xf16>
  %empty24 = tensor.empty() : tensor<1x512x56x56xf16>
  %relu25 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv23 : tensor<1x512x56x56xf16>)
    outs(%empty24 : tensor<1x512x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x56x56xf16>
  %init26 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill27 = linalg.fill ins(%cst : f16) outs(%init26 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv28 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu25, %w_s0_b1_pw2 : tensor<1x512x56x56xf16>, tensor<128x512x1x1xf16>)
    outs(%fill27 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %empty29 = tensor.empty() : tensor<1x128x56x56xf16>
  %add30 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv28, %add16 : tensor<1x128x56x56xf16>, tensor<1x128x56x56xf16>)
    outs(%empty29 : tensor<1x128x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x56x56xf16>
  %pad31 = tensor.pad %add30 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x56x56xf16> to tensor<1x128x62x62xf16>
  %init32 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill33 = linalg.fill ins(%cst : f16) outs(%init32 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv34 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad31, %w_s0_b2_dw : tensor<1x128x62x62xf16>, tensor<128x128x7x7xf16>)
    outs(%fill33 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %init35 = tensor.empty() : tensor<1x512x56x56xf16>
  %fill36 = linalg.fill ins(%cst : f16) outs(%init35 : tensor<1x512x56x56xf16>) -> tensor<1x512x56x56xf16>
  %conv37 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv34, %w_s0_b2_pw1 : tensor<1x128x56x56xf16>, tensor<512x128x1x1xf16>)
    outs(%fill36 : tensor<1x512x56x56xf16>) -> tensor<1x512x56x56xf16>
  %empty38 = tensor.empty() : tensor<1x512x56x56xf16>
  %relu39 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv37 : tensor<1x512x56x56xf16>)
    outs(%empty38 : tensor<1x512x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x56x56xf16>
  %init40 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill41 = linalg.fill ins(%cst : f16) outs(%init40 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv42 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu39, %w_s0_b2_pw2 : tensor<1x512x56x56xf16>, tensor<128x512x1x1xf16>)
    outs(%fill41 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %empty43 = tensor.empty() : tensor<1x128x56x56xf16>
  %add44 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv42, %add30 : tensor<1x128x56x56xf16>, tensor<1x128x56x56xf16>)
    outs(%empty43 : tensor<1x128x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x56x56xf16>

  // Downsample: 128->256, stride 2
  %pad45 = tensor.pad %add44 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x56x56xf16> to tensor<1x128x58x58xf16>
  %init46 = tensor.empty() : tensor<1x256x28x28xf16>
  %fill47 = linalg.fill ins(%cst : f16) outs(%init46 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %conv48 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad45, %w_ds1 : tensor<1x128x58x58xf16>, tensor<256x128x3x3xf16>)
    outs(%fill47 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>

  // === ConvNeXt Stage 1: dim=256 ===
  %pad49 = tensor.pad %conv48 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x28x28xf16> to tensor<1x256x34x34xf16>
  %init50 = tensor.empty() : tensor<1x256x28x28xf16>
  %fill51 = linalg.fill ins(%cst : f16) outs(%init50 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %conv52 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad49, %w_s1_b0_dw : tensor<1x256x34x34xf16>, tensor<256x256x7x7xf16>)
    outs(%fill51 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %init53 = tensor.empty() : tensor<1x1024x28x28xf16>
  %fill54 = linalg.fill ins(%cst : f16) outs(%init53 : tensor<1x1024x28x28xf16>) -> tensor<1x1024x28x28xf16>
  %conv55 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv52, %w_s1_b0_pw1 : tensor<1x256x28x28xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill54 : tensor<1x1024x28x28xf16>) -> tensor<1x1024x28x28xf16>
  %empty56 = tensor.empty() : tensor<1x1024x28x28xf16>
  %relu57 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv55 : tensor<1x1024x28x28xf16>)
    outs(%empty56 : tensor<1x1024x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x28x28xf16>
  %init58 = tensor.empty() : tensor<1x256x28x28xf16>
  %fill59 = linalg.fill ins(%cst : f16) outs(%init58 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %conv60 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu57, %w_s1_b0_pw2 : tensor<1x1024x28x28xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill59 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %empty61 = tensor.empty() : tensor<1x256x28x28xf16>
  %add62 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv60, %conv48 : tensor<1x256x28x28xf16>, tensor<1x256x28x28xf16>)
    outs(%empty61 : tensor<1x256x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x28x28xf16>
  %pad63 = tensor.pad %add62 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x28x28xf16> to tensor<1x256x34x34xf16>
  %init64 = tensor.empty() : tensor<1x256x28x28xf16>
  %fill65 = linalg.fill ins(%cst : f16) outs(%init64 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %conv66 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad63, %w_s1_b1_dw : tensor<1x256x34x34xf16>, tensor<256x256x7x7xf16>)
    outs(%fill65 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %init67 = tensor.empty() : tensor<1x1024x28x28xf16>
  %fill68 = linalg.fill ins(%cst : f16) outs(%init67 : tensor<1x1024x28x28xf16>) -> tensor<1x1024x28x28xf16>
  %conv69 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv66, %w_s1_b1_pw1 : tensor<1x256x28x28xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill68 : tensor<1x1024x28x28xf16>) -> tensor<1x1024x28x28xf16>
  %empty70 = tensor.empty() : tensor<1x1024x28x28xf16>
  %relu71 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv69 : tensor<1x1024x28x28xf16>)
    outs(%empty70 : tensor<1x1024x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x28x28xf16>
  %init72 = tensor.empty() : tensor<1x256x28x28xf16>
  %fill73 = linalg.fill ins(%cst : f16) outs(%init72 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %conv74 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu71, %w_s1_b1_pw2 : tensor<1x1024x28x28xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill73 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %empty75 = tensor.empty() : tensor<1x256x28x28xf16>
  %add76 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv74, %add62 : tensor<1x256x28x28xf16>, tensor<1x256x28x28xf16>)
    outs(%empty75 : tensor<1x256x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x28x28xf16>
  %pad77 = tensor.pad %add76 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x28x28xf16> to tensor<1x256x34x34xf16>
  %init78 = tensor.empty() : tensor<1x256x28x28xf16>
  %fill79 = linalg.fill ins(%cst : f16) outs(%init78 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %conv80 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad77, %w_s1_b2_dw : tensor<1x256x34x34xf16>, tensor<256x256x7x7xf16>)
    outs(%fill79 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %init81 = tensor.empty() : tensor<1x1024x28x28xf16>
  %fill82 = linalg.fill ins(%cst : f16) outs(%init81 : tensor<1x1024x28x28xf16>) -> tensor<1x1024x28x28xf16>
  %conv83 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv80, %w_s1_b2_pw1 : tensor<1x256x28x28xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill82 : tensor<1x1024x28x28xf16>) -> tensor<1x1024x28x28xf16>
  %empty84 = tensor.empty() : tensor<1x1024x28x28xf16>
  %relu85 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv83 : tensor<1x1024x28x28xf16>)
    outs(%empty84 : tensor<1x1024x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x28x28xf16>
  %init86 = tensor.empty() : tensor<1x256x28x28xf16>
  %fill87 = linalg.fill ins(%cst : f16) outs(%init86 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %conv88 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu85, %w_s1_b2_pw2 : tensor<1x1024x28x28xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill87 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %empty89 = tensor.empty() : tensor<1x256x28x28xf16>
  %add90 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv88, %add76 : tensor<1x256x28x28xf16>, tensor<1x256x28x28xf16>)
    outs(%empty89 : tensor<1x256x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x28x28xf16>

  // Downsample: 256->512, stride 2
  %pad91 = tensor.pad %add90 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x28x28xf16> to tensor<1x256x30x30xf16>
  %init92 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill93 = linalg.fill ins(%cst : f16) outs(%init92 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv94 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad91, %w_ds2 : tensor<1x256x30x30xf16>, tensor<512x256x3x3xf16>)
    outs(%fill93 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>

  // === ConvNeXt Stage 2: dim=512 ===
  %pad95 = tensor.pad %conv94 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init96 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill97 = linalg.fill ins(%cst : f16) outs(%init96 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv98 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad95, %w_s2_b0_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill97 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init99 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill100 = linalg.fill ins(%cst : f16) outs(%init99 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv101 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv98, %w_s2_b0_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill100 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty102 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu103 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv101 : tensor<1x2048x14x14xf16>)
    outs(%empty102 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init104 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill105 = linalg.fill ins(%cst : f16) outs(%init104 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv106 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu103, %w_s2_b0_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill105 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty107 = tensor.empty() : tensor<1x512x14x14xf16>
  %add108 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv106, %conv94 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty107 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad109 = tensor.pad %add108 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init110 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill111 = linalg.fill ins(%cst : f16) outs(%init110 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv112 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad109, %w_s2_b1_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill111 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init113 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill114 = linalg.fill ins(%cst : f16) outs(%init113 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv115 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv112, %w_s2_b1_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill114 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty116 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu117 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv115 : tensor<1x2048x14x14xf16>)
    outs(%empty116 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init118 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill119 = linalg.fill ins(%cst : f16) outs(%init118 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv120 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu117, %w_s2_b1_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill119 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty121 = tensor.empty() : tensor<1x512x14x14xf16>
  %add122 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv120, %add108 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty121 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad123 = tensor.pad %add122 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init124 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill125 = linalg.fill ins(%cst : f16) outs(%init124 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv126 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad123, %w_s2_b2_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill125 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init127 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill128 = linalg.fill ins(%cst : f16) outs(%init127 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv129 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv126, %w_s2_b2_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill128 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty130 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu131 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv129 : tensor<1x2048x14x14xf16>)
    outs(%empty130 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init132 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill133 = linalg.fill ins(%cst : f16) outs(%init132 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv134 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu131, %w_s2_b2_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill133 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty135 = tensor.empty() : tensor<1x512x14x14xf16>
  %add136 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv134, %add122 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty135 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad137 = tensor.pad %add136 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init138 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill139 = linalg.fill ins(%cst : f16) outs(%init138 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv140 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad137, %w_s2_b3_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill139 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init141 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill142 = linalg.fill ins(%cst : f16) outs(%init141 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv143 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv140, %w_s2_b3_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill142 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty144 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu145 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv143 : tensor<1x2048x14x14xf16>)
    outs(%empty144 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init146 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill147 = linalg.fill ins(%cst : f16) outs(%init146 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv148 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu145, %w_s2_b3_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill147 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty149 = tensor.empty() : tensor<1x512x14x14xf16>
  %add150 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv148, %add136 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty149 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad151 = tensor.pad %add150 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init152 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill153 = linalg.fill ins(%cst : f16) outs(%init152 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv154 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad151, %w_s2_b4_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill153 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init155 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill156 = linalg.fill ins(%cst : f16) outs(%init155 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv157 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv154, %w_s2_b4_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill156 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty158 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu159 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv157 : tensor<1x2048x14x14xf16>)
    outs(%empty158 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init160 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill161 = linalg.fill ins(%cst : f16) outs(%init160 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv162 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu159, %w_s2_b4_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill161 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty163 = tensor.empty() : tensor<1x512x14x14xf16>
  %add164 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv162, %add150 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty163 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad165 = tensor.pad %add164 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init166 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill167 = linalg.fill ins(%cst : f16) outs(%init166 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv168 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad165, %w_s2_b5_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill167 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init169 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill170 = linalg.fill ins(%cst : f16) outs(%init169 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv171 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv168, %w_s2_b5_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill170 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty172 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu173 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv171 : tensor<1x2048x14x14xf16>)
    outs(%empty172 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init174 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill175 = linalg.fill ins(%cst : f16) outs(%init174 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv176 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu173, %w_s2_b5_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill175 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty177 = tensor.empty() : tensor<1x512x14x14xf16>
  %add178 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv176, %add164 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty177 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad179 = tensor.pad %add178 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init180 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill181 = linalg.fill ins(%cst : f16) outs(%init180 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv182 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad179, %w_s2_b6_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill181 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init183 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill184 = linalg.fill ins(%cst : f16) outs(%init183 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv185 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv182, %w_s2_b6_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill184 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty186 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu187 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv185 : tensor<1x2048x14x14xf16>)
    outs(%empty186 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init188 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill189 = linalg.fill ins(%cst : f16) outs(%init188 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv190 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu187, %w_s2_b6_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill189 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty191 = tensor.empty() : tensor<1x512x14x14xf16>
  %add192 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv190, %add178 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty191 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad193 = tensor.pad %add192 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init194 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill195 = linalg.fill ins(%cst : f16) outs(%init194 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv196 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad193, %w_s2_b7_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill195 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init197 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill198 = linalg.fill ins(%cst : f16) outs(%init197 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv199 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv196, %w_s2_b7_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill198 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty200 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu201 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv199 : tensor<1x2048x14x14xf16>)
    outs(%empty200 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init202 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill203 = linalg.fill ins(%cst : f16) outs(%init202 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv204 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu201, %w_s2_b7_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill203 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty205 = tensor.empty() : tensor<1x512x14x14xf16>
  %add206 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv204, %add192 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty205 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad207 = tensor.pad %add206 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init208 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill209 = linalg.fill ins(%cst : f16) outs(%init208 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv210 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad207, %w_s2_b8_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill209 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init211 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill212 = linalg.fill ins(%cst : f16) outs(%init211 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv213 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv210, %w_s2_b8_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill212 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty214 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu215 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv213 : tensor<1x2048x14x14xf16>)
    outs(%empty214 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init216 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill217 = linalg.fill ins(%cst : f16) outs(%init216 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv218 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu215, %w_s2_b8_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill217 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty219 = tensor.empty() : tensor<1x512x14x14xf16>
  %add220 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv218, %add206 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty219 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad221 = tensor.pad %add220 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init222 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill223 = linalg.fill ins(%cst : f16) outs(%init222 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv224 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad221, %w_s2_b9_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill223 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init225 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill226 = linalg.fill ins(%cst : f16) outs(%init225 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv227 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv224, %w_s2_b9_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill226 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty228 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu229 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv227 : tensor<1x2048x14x14xf16>)
    outs(%empty228 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init230 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill231 = linalg.fill ins(%cst : f16) outs(%init230 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv232 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu229, %w_s2_b9_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill231 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty233 = tensor.empty() : tensor<1x512x14x14xf16>
  %add234 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv232, %add220 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty233 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad235 = tensor.pad %add234 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init236 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill237 = linalg.fill ins(%cst : f16) outs(%init236 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv238 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad235, %w_s2_b10_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill237 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init239 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill240 = linalg.fill ins(%cst : f16) outs(%init239 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv241 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv238, %w_s2_b10_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill240 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty242 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu243 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv241 : tensor<1x2048x14x14xf16>)
    outs(%empty242 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init244 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill245 = linalg.fill ins(%cst : f16) outs(%init244 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv246 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu243, %w_s2_b10_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill245 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty247 = tensor.empty() : tensor<1x512x14x14xf16>
  %add248 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv246, %add234 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty247 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad249 = tensor.pad %add248 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init250 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill251 = linalg.fill ins(%cst : f16) outs(%init250 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv252 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad249, %w_s2_b11_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill251 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init253 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill254 = linalg.fill ins(%cst : f16) outs(%init253 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv255 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv252, %w_s2_b11_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill254 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty256 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu257 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv255 : tensor<1x2048x14x14xf16>)
    outs(%empty256 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init258 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill259 = linalg.fill ins(%cst : f16) outs(%init258 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv260 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu257, %w_s2_b11_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill259 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty261 = tensor.empty() : tensor<1x512x14x14xf16>
  %add262 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv260, %add248 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty261 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad263 = tensor.pad %add262 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init264 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill265 = linalg.fill ins(%cst : f16) outs(%init264 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv266 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad263, %w_s2_b12_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill265 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init267 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill268 = linalg.fill ins(%cst : f16) outs(%init267 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv269 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv266, %w_s2_b12_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill268 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty270 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu271 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv269 : tensor<1x2048x14x14xf16>)
    outs(%empty270 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init272 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill273 = linalg.fill ins(%cst : f16) outs(%init272 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv274 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu271, %w_s2_b12_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill273 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty275 = tensor.empty() : tensor<1x512x14x14xf16>
  %add276 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv274, %add262 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty275 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad277 = tensor.pad %add276 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init278 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill279 = linalg.fill ins(%cst : f16) outs(%init278 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv280 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad277, %w_s2_b13_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill279 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init281 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill282 = linalg.fill ins(%cst : f16) outs(%init281 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv283 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv280, %w_s2_b13_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill282 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty284 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu285 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv283 : tensor<1x2048x14x14xf16>)
    outs(%empty284 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init286 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill287 = linalg.fill ins(%cst : f16) outs(%init286 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv288 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu285, %w_s2_b13_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill287 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty289 = tensor.empty() : tensor<1x512x14x14xf16>
  %add290 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv288, %add276 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty289 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad291 = tensor.pad %add290 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init292 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill293 = linalg.fill ins(%cst : f16) outs(%init292 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv294 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad291, %w_s2_b14_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill293 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init295 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill296 = linalg.fill ins(%cst : f16) outs(%init295 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv297 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv294, %w_s2_b14_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill296 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty298 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu299 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv297 : tensor<1x2048x14x14xf16>)
    outs(%empty298 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init300 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill301 = linalg.fill ins(%cst : f16) outs(%init300 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv302 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu299, %w_s2_b14_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill301 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty303 = tensor.empty() : tensor<1x512x14x14xf16>
  %add304 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv302, %add290 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty303 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad305 = tensor.pad %add304 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init306 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill307 = linalg.fill ins(%cst : f16) outs(%init306 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv308 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad305, %w_s2_b15_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill307 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init309 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill310 = linalg.fill ins(%cst : f16) outs(%init309 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv311 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv308, %w_s2_b15_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill310 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty312 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu313 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv311 : tensor<1x2048x14x14xf16>)
    outs(%empty312 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init314 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill315 = linalg.fill ins(%cst : f16) outs(%init314 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv316 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu313, %w_s2_b15_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill315 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty317 = tensor.empty() : tensor<1x512x14x14xf16>
  %add318 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv316, %add304 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty317 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad319 = tensor.pad %add318 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init320 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill321 = linalg.fill ins(%cst : f16) outs(%init320 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv322 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad319, %w_s2_b16_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill321 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init323 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill324 = linalg.fill ins(%cst : f16) outs(%init323 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv325 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv322, %w_s2_b16_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill324 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty326 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu327 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv325 : tensor<1x2048x14x14xf16>)
    outs(%empty326 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init328 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill329 = linalg.fill ins(%cst : f16) outs(%init328 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv330 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu327, %w_s2_b16_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill329 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty331 = tensor.empty() : tensor<1x512x14x14xf16>
  %add332 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv330, %add318 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty331 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad333 = tensor.pad %add332 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init334 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill335 = linalg.fill ins(%cst : f16) outs(%init334 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv336 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad333, %w_s2_b17_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill335 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init337 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill338 = linalg.fill ins(%cst : f16) outs(%init337 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv339 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv336, %w_s2_b17_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill338 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty340 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu341 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv339 : tensor<1x2048x14x14xf16>)
    outs(%empty340 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init342 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill343 = linalg.fill ins(%cst : f16) outs(%init342 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv344 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu341, %w_s2_b17_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill343 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty345 = tensor.empty() : tensor<1x512x14x14xf16>
  %add346 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv344, %add332 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty345 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad347 = tensor.pad %add346 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init348 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill349 = linalg.fill ins(%cst : f16) outs(%init348 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv350 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad347, %w_s2_b18_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill349 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init351 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill352 = linalg.fill ins(%cst : f16) outs(%init351 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv353 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv350, %w_s2_b18_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill352 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty354 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu355 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv353 : tensor<1x2048x14x14xf16>)
    outs(%empty354 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init356 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill357 = linalg.fill ins(%cst : f16) outs(%init356 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv358 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu355, %w_s2_b18_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill357 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty359 = tensor.empty() : tensor<1x512x14x14xf16>
  %add360 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv358, %add346 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty359 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad361 = tensor.pad %add360 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init362 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill363 = linalg.fill ins(%cst : f16) outs(%init362 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv364 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad361, %w_s2_b19_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill363 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init365 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill366 = linalg.fill ins(%cst : f16) outs(%init365 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv367 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv364, %w_s2_b19_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill366 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty368 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu369 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv367 : tensor<1x2048x14x14xf16>)
    outs(%empty368 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init370 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill371 = linalg.fill ins(%cst : f16) outs(%init370 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv372 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu369, %w_s2_b19_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill371 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty373 = tensor.empty() : tensor<1x512x14x14xf16>
  %add374 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv372, %add360 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty373 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad375 = tensor.pad %add374 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init376 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill377 = linalg.fill ins(%cst : f16) outs(%init376 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv378 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad375, %w_s2_b20_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill377 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init379 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill380 = linalg.fill ins(%cst : f16) outs(%init379 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv381 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv378, %w_s2_b20_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill380 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty382 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu383 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv381 : tensor<1x2048x14x14xf16>)
    outs(%empty382 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init384 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill385 = linalg.fill ins(%cst : f16) outs(%init384 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv386 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu383, %w_s2_b20_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill385 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty387 = tensor.empty() : tensor<1x512x14x14xf16>
  %add388 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv386, %add374 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty387 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad389 = tensor.pad %add388 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init390 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill391 = linalg.fill ins(%cst : f16) outs(%init390 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv392 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad389, %w_s2_b21_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill391 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init393 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill394 = linalg.fill ins(%cst : f16) outs(%init393 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv395 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv392, %w_s2_b21_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill394 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty396 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu397 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv395 : tensor<1x2048x14x14xf16>)
    outs(%empty396 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init398 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill399 = linalg.fill ins(%cst : f16) outs(%init398 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv400 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu397, %w_s2_b21_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill399 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty401 = tensor.empty() : tensor<1x512x14x14xf16>
  %add402 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv400, %add388 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty401 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad403 = tensor.pad %add402 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init404 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill405 = linalg.fill ins(%cst : f16) outs(%init404 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv406 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad403, %w_s2_b22_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill405 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init407 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill408 = linalg.fill ins(%cst : f16) outs(%init407 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv409 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv406, %w_s2_b22_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill408 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty410 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu411 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv409 : tensor<1x2048x14x14xf16>)
    outs(%empty410 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init412 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill413 = linalg.fill ins(%cst : f16) outs(%init412 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv414 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu411, %w_s2_b22_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill413 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty415 = tensor.empty() : tensor<1x512x14x14xf16>
  %add416 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv414, %add402 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty415 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad417 = tensor.pad %add416 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init418 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill419 = linalg.fill ins(%cst : f16) outs(%init418 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv420 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad417, %w_s2_b23_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill419 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init421 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill422 = linalg.fill ins(%cst : f16) outs(%init421 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv423 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv420, %w_s2_b23_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill422 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty424 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu425 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv423 : tensor<1x2048x14x14xf16>)
    outs(%empty424 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init426 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill427 = linalg.fill ins(%cst : f16) outs(%init426 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv428 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu425, %w_s2_b23_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill427 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty429 = tensor.empty() : tensor<1x512x14x14xf16>
  %add430 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv428, %add416 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty429 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad431 = tensor.pad %add430 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init432 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill433 = linalg.fill ins(%cst : f16) outs(%init432 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv434 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad431, %w_s2_b24_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill433 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init435 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill436 = linalg.fill ins(%cst : f16) outs(%init435 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv437 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv434, %w_s2_b24_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill436 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty438 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu439 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv437 : tensor<1x2048x14x14xf16>)
    outs(%empty438 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init440 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill441 = linalg.fill ins(%cst : f16) outs(%init440 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv442 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu439, %w_s2_b24_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill441 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty443 = tensor.empty() : tensor<1x512x14x14xf16>
  %add444 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv442, %add430 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty443 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad445 = tensor.pad %add444 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init446 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill447 = linalg.fill ins(%cst : f16) outs(%init446 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv448 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad445, %w_s2_b25_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill447 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init449 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill450 = linalg.fill ins(%cst : f16) outs(%init449 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv451 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv448, %w_s2_b25_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill450 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty452 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu453 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv451 : tensor<1x2048x14x14xf16>)
    outs(%empty452 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init454 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill455 = linalg.fill ins(%cst : f16) outs(%init454 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv456 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu453, %w_s2_b25_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill455 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty457 = tensor.empty() : tensor<1x512x14x14xf16>
  %add458 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv456, %add444 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty457 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>
  %pad459 = tensor.pad %add458 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x20x20xf16>
  %init460 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill461 = linalg.fill ins(%cst : f16) outs(%init460 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv462 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad459, %w_s2_b26_dw : tensor<1x512x20x20xf16>, tensor<512x512x7x7xf16>)
    outs(%fill461 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %init463 = tensor.empty() : tensor<1x2048x14x14xf16>
  %fill464 = linalg.fill ins(%cst : f16) outs(%init463 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %conv465 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv462, %w_s2_b26_pw1 : tensor<1x512x14x14xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill464 : tensor<1x2048x14x14xf16>) -> tensor<1x2048x14x14xf16>
  %empty466 = tensor.empty() : tensor<1x2048x14x14xf16>
  %relu467 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv465 : tensor<1x2048x14x14xf16>)
    outs(%empty466 : tensor<1x2048x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x14x14xf16>
  %init468 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill469 = linalg.fill ins(%cst : f16) outs(%init468 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv470 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu467, %w_s2_b26_pw2 : tensor<1x2048x14x14xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill469 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty471 = tensor.empty() : tensor<1x512x14x14xf16>
  %add472 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv470, %add458 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty471 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>

  // Downsample: 512->1024, stride 2
  %pad473 = tensor.pad %add472 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x16x16xf16>
  %init474 = tensor.empty() : tensor<1x1024x7x7xf16>
  %fill475 = linalg.fill ins(%cst : f16) outs(%init474 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %conv476 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad473, %w_ds3 : tensor<1x512x16x16xf16>, tensor<1024x512x3x3xf16>)
    outs(%fill475 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>

  // === ConvNeXt Stage 3: dim=1024 ===
  %pad477 = tensor.pad %conv476 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1024x7x7xf16> to tensor<1x1024x13x13xf16>
  %init478 = tensor.empty() : tensor<1x1024x7x7xf16>
  %fill479 = linalg.fill ins(%cst : f16) outs(%init478 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %conv480 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad477, %w_s3_b0_dw : tensor<1x1024x13x13xf16>, tensor<1024x1024x7x7xf16>)
    outs(%fill479 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %init481 = tensor.empty() : tensor<1x4096x7x7xf16>
  %fill482 = linalg.fill ins(%cst : f16) outs(%init481 : tensor<1x4096x7x7xf16>) -> tensor<1x4096x7x7xf16>
  %conv483 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv480, %w_s3_b0_pw1 : tensor<1x1024x7x7xf16>, tensor<4096x1024x1x1xf16>)
    outs(%fill482 : tensor<1x4096x7x7xf16>) -> tensor<1x4096x7x7xf16>
  %empty484 = tensor.empty() : tensor<1x4096x7x7xf16>
  %relu485 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv483 : tensor<1x4096x7x7xf16>)
    outs(%empty484 : tensor<1x4096x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x4096x7x7xf16>
  %init486 = tensor.empty() : tensor<1x1024x7x7xf16>
  %fill487 = linalg.fill ins(%cst : f16) outs(%init486 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %conv488 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu485, %w_s3_b0_pw2 : tensor<1x4096x7x7xf16>, tensor<1024x4096x1x1xf16>)
    outs(%fill487 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %empty489 = tensor.empty() : tensor<1x1024x7x7xf16>
  %add490 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv488, %conv476 : tensor<1x1024x7x7xf16>, tensor<1x1024x7x7xf16>)
    outs(%empty489 : tensor<1x1024x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x7x7xf16>
  %pad491 = tensor.pad %add490 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1024x7x7xf16> to tensor<1x1024x13x13xf16>
  %init492 = tensor.empty() : tensor<1x1024x7x7xf16>
  %fill493 = linalg.fill ins(%cst : f16) outs(%init492 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %conv494 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad491, %w_s3_b1_dw : tensor<1x1024x13x13xf16>, tensor<1024x1024x7x7xf16>)
    outs(%fill493 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %init495 = tensor.empty() : tensor<1x4096x7x7xf16>
  %fill496 = linalg.fill ins(%cst : f16) outs(%init495 : tensor<1x4096x7x7xf16>) -> tensor<1x4096x7x7xf16>
  %conv497 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv494, %w_s3_b1_pw1 : tensor<1x1024x7x7xf16>, tensor<4096x1024x1x1xf16>)
    outs(%fill496 : tensor<1x4096x7x7xf16>) -> tensor<1x4096x7x7xf16>
  %empty498 = tensor.empty() : tensor<1x4096x7x7xf16>
  %relu499 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv497 : tensor<1x4096x7x7xf16>)
    outs(%empty498 : tensor<1x4096x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x4096x7x7xf16>
  %init500 = tensor.empty() : tensor<1x1024x7x7xf16>
  %fill501 = linalg.fill ins(%cst : f16) outs(%init500 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %conv502 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu499, %w_s3_b1_pw2 : tensor<1x4096x7x7xf16>, tensor<1024x4096x1x1xf16>)
    outs(%fill501 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %empty503 = tensor.empty() : tensor<1x1024x7x7xf16>
  %add504 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv502, %add490 : tensor<1x1024x7x7xf16>, tensor<1x1024x7x7xf16>)
    outs(%empty503 : tensor<1x1024x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x7x7xf16>
  %pad505 = tensor.pad %add504 low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1024x7x7xf16> to tensor<1x1024x13x13xf16>
  %init506 = tensor.empty() : tensor<1x1024x7x7xf16>
  %fill507 = linalg.fill ins(%cst : f16) outs(%init506 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %conv508 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad505, %w_s3_b2_dw : tensor<1x1024x13x13xf16>, tensor<1024x1024x7x7xf16>)
    outs(%fill507 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %init509 = tensor.empty() : tensor<1x4096x7x7xf16>
  %fill510 = linalg.fill ins(%cst : f16) outs(%init509 : tensor<1x4096x7x7xf16>) -> tensor<1x4096x7x7xf16>
  %conv511 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv508, %w_s3_b2_pw1 : tensor<1x1024x7x7xf16>, tensor<4096x1024x1x1xf16>)
    outs(%fill510 : tensor<1x4096x7x7xf16>) -> tensor<1x4096x7x7xf16>
  %empty512 = tensor.empty() : tensor<1x4096x7x7xf16>
  %relu513 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv511 : tensor<1x4096x7x7xf16>)
    outs(%empty512 : tensor<1x4096x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x4096x7x7xf16>
  %init514 = tensor.empty() : tensor<1x1024x7x7xf16>
  %fill515 = linalg.fill ins(%cst : f16) outs(%init514 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %conv516 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu513, %w_s3_b2_pw2 : tensor<1x4096x7x7xf16>, tensor<1024x4096x1x1xf16>)
    outs(%fill515 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %empty517 = tensor.empty() : tensor<1x1024x7x7xf16>
  %add518 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv516, %add504 : tensor<1x1024x7x7xf16>, tensor<1x1024x7x7xf16>)
    outs(%empty517 : tensor<1x1024x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x7x7xf16>

  // FC: 1x1 1024->1000
  %init519 = tensor.empty() : tensor<1x1000x7x7xf16>
  %fill520 = linalg.fill ins(%cst : f16) outs(%init519 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  %conv521 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add518, %w_fc : tensor<1x1024x7x7xf16>, tensor<1000x1024x1x1xf16>)
    outs(%fill520 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  return %conv521 : tensor<1x1000x7x7xf16>
}
