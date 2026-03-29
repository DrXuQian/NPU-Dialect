func.func @resnet_basic_18_18_18_18(
    %input: tensor<1x3x224x224xf16>,
    %w_conv1: tensor<64x3x7x7xf16>,
    %w_pool: tensor<64x64x3x3xf16>,
    %w_s0_b0_c1: tensor<64x64x3x3xf16>,
    %w_s0_b0_c2: tensor<64x64x3x3xf16>,
    %w_s0_b1_c1: tensor<64x64x3x3xf16>,
    %w_s0_b1_c2: tensor<64x64x3x3xf16>,
    %w_s0_b2_c1: tensor<64x64x3x3xf16>,
    %w_s0_b2_c2: tensor<64x64x3x3xf16>,
    %w_s0_b3_c1: tensor<64x64x3x3xf16>,
    %w_s0_b3_c2: tensor<64x64x3x3xf16>,
    %w_s0_b4_c1: tensor<64x64x3x3xf16>,
    %w_s0_b4_c2: tensor<64x64x3x3xf16>,
    %w_s0_b5_c1: tensor<64x64x3x3xf16>,
    %w_s0_b5_c2: tensor<64x64x3x3xf16>,
    %w_s0_b6_c1: tensor<64x64x3x3xf16>,
    %w_s0_b6_c2: tensor<64x64x3x3xf16>,
    %w_s0_b7_c1: tensor<64x64x3x3xf16>,
    %w_s0_b7_c2: tensor<64x64x3x3xf16>,
    %w_s0_b8_c1: tensor<64x64x3x3xf16>,
    %w_s0_b8_c2: tensor<64x64x3x3xf16>,
    %w_s0_b9_c1: tensor<64x64x3x3xf16>,
    %w_s0_b9_c2: tensor<64x64x3x3xf16>,
    %w_s0_b10_c1: tensor<64x64x3x3xf16>,
    %w_s0_b10_c2: tensor<64x64x3x3xf16>,
    %w_s0_b11_c1: tensor<64x64x3x3xf16>,
    %w_s0_b11_c2: tensor<64x64x3x3xf16>,
    %w_s0_b12_c1: tensor<64x64x3x3xf16>,
    %w_s0_b12_c2: tensor<64x64x3x3xf16>,
    %w_s0_b13_c1: tensor<64x64x3x3xf16>,
    %w_s0_b13_c2: tensor<64x64x3x3xf16>,
    %w_s0_b14_c1: tensor<64x64x3x3xf16>,
    %w_s0_b14_c2: tensor<64x64x3x3xf16>,
    %w_s0_b15_c1: tensor<64x64x3x3xf16>,
    %w_s0_b15_c2: tensor<64x64x3x3xf16>,
    %w_s0_b16_c1: tensor<64x64x3x3xf16>,
    %w_s0_b16_c2: tensor<64x64x3x3xf16>,
    %w_s0_b17_c1: tensor<64x64x3x3xf16>,
    %w_s0_b17_c2: tensor<64x64x3x3xf16>,
    %w_s1_b0_c1: tensor<128x64x3x3xf16>,
    %w_s1_b0_c2: tensor<128x128x3x3xf16>,
    %w_s1_b0_sc: tensor<128x64x1x1xf16>,
    %w_s1_b1_c1: tensor<128x128x3x3xf16>,
    %w_s1_b1_c2: tensor<128x128x3x3xf16>,
    %w_s1_b2_c1: tensor<128x128x3x3xf16>,
    %w_s1_b2_c2: tensor<128x128x3x3xf16>,
    %w_s1_b3_c1: tensor<128x128x3x3xf16>,
    %w_s1_b3_c2: tensor<128x128x3x3xf16>,
    %w_s1_b4_c1: tensor<128x128x3x3xf16>,
    %w_s1_b4_c2: tensor<128x128x3x3xf16>,
    %w_s1_b5_c1: tensor<128x128x3x3xf16>,
    %w_s1_b5_c2: tensor<128x128x3x3xf16>,
    %w_s1_b6_c1: tensor<128x128x3x3xf16>,
    %w_s1_b6_c2: tensor<128x128x3x3xf16>,
    %w_s1_b7_c1: tensor<128x128x3x3xf16>,
    %w_s1_b7_c2: tensor<128x128x3x3xf16>,
    %w_s1_b8_c1: tensor<128x128x3x3xf16>,
    %w_s1_b8_c2: tensor<128x128x3x3xf16>,
    %w_s1_b9_c1: tensor<128x128x3x3xf16>,
    %w_s1_b9_c2: tensor<128x128x3x3xf16>,
    %w_s1_b10_c1: tensor<128x128x3x3xf16>,
    %w_s1_b10_c2: tensor<128x128x3x3xf16>,
    %w_s1_b11_c1: tensor<128x128x3x3xf16>,
    %w_s1_b11_c2: tensor<128x128x3x3xf16>,
    %w_s1_b12_c1: tensor<128x128x3x3xf16>,
    %w_s1_b12_c2: tensor<128x128x3x3xf16>,
    %w_s1_b13_c1: tensor<128x128x3x3xf16>,
    %w_s1_b13_c2: tensor<128x128x3x3xf16>,
    %w_s1_b14_c1: tensor<128x128x3x3xf16>,
    %w_s1_b14_c2: tensor<128x128x3x3xf16>,
    %w_s1_b15_c1: tensor<128x128x3x3xf16>,
    %w_s1_b15_c2: tensor<128x128x3x3xf16>,
    %w_s1_b16_c1: tensor<128x128x3x3xf16>,
    %w_s1_b16_c2: tensor<128x128x3x3xf16>,
    %w_s1_b17_c1: tensor<128x128x3x3xf16>,
    %w_s1_b17_c2: tensor<128x128x3x3xf16>,
    %w_s2_b0_c1: tensor<256x128x3x3xf16>,
    %w_s2_b0_c2: tensor<256x256x3x3xf16>,
    %w_s2_b0_sc: tensor<256x128x1x1xf16>,
    %w_s2_b1_c1: tensor<256x256x3x3xf16>,
    %w_s2_b1_c2: tensor<256x256x3x3xf16>,
    %w_s2_b2_c1: tensor<256x256x3x3xf16>,
    %w_s2_b2_c2: tensor<256x256x3x3xf16>,
    %w_s2_b3_c1: tensor<256x256x3x3xf16>,
    %w_s2_b3_c2: tensor<256x256x3x3xf16>,
    %w_s2_b4_c1: tensor<256x256x3x3xf16>,
    %w_s2_b4_c2: tensor<256x256x3x3xf16>,
    %w_s2_b5_c1: tensor<256x256x3x3xf16>,
    %w_s2_b5_c2: tensor<256x256x3x3xf16>,
    %w_s2_b6_c1: tensor<256x256x3x3xf16>,
    %w_s2_b6_c2: tensor<256x256x3x3xf16>,
    %w_s2_b7_c1: tensor<256x256x3x3xf16>,
    %w_s2_b7_c2: tensor<256x256x3x3xf16>,
    %w_s2_b8_c1: tensor<256x256x3x3xf16>,
    %w_s2_b8_c2: tensor<256x256x3x3xf16>,
    %w_s2_b9_c1: tensor<256x256x3x3xf16>,
    %w_s2_b9_c2: tensor<256x256x3x3xf16>,
    %w_s2_b10_c1: tensor<256x256x3x3xf16>,
    %w_s2_b10_c2: tensor<256x256x3x3xf16>,
    %w_s2_b11_c1: tensor<256x256x3x3xf16>,
    %w_s2_b11_c2: tensor<256x256x3x3xf16>,
    %w_s2_b12_c1: tensor<256x256x3x3xf16>,
    %w_s2_b12_c2: tensor<256x256x3x3xf16>,
    %w_s2_b13_c1: tensor<256x256x3x3xf16>,
    %w_s2_b13_c2: tensor<256x256x3x3xf16>,
    %w_s2_b14_c1: tensor<256x256x3x3xf16>,
    %w_s2_b14_c2: tensor<256x256x3x3xf16>,
    %w_s2_b15_c1: tensor<256x256x3x3xf16>,
    %w_s2_b15_c2: tensor<256x256x3x3xf16>,
    %w_s2_b16_c1: tensor<256x256x3x3xf16>,
    %w_s2_b16_c2: tensor<256x256x3x3xf16>,
    %w_s2_b17_c1: tensor<256x256x3x3xf16>,
    %w_s2_b17_c2: tensor<256x256x3x3xf16>,
    %w_s3_b0_c1: tensor<512x256x3x3xf16>,
    %w_s3_b0_c2: tensor<512x512x3x3xf16>,
    %w_s3_b0_sc: tensor<512x256x1x1xf16>,
    %w_s3_b1_c1: tensor<512x512x3x3xf16>,
    %w_s3_b1_c2: tensor<512x512x3x3xf16>,
    %w_s3_b2_c1: tensor<512x512x3x3xf16>,
    %w_s3_b2_c2: tensor<512x512x3x3xf16>,
    %w_s3_b3_c1: tensor<512x512x3x3xf16>,
    %w_s3_b3_c2: tensor<512x512x3x3xf16>,
    %w_s3_b4_c1: tensor<512x512x3x3xf16>,
    %w_s3_b4_c2: tensor<512x512x3x3xf16>,
    %w_s3_b5_c1: tensor<512x512x3x3xf16>,
    %w_s3_b5_c2: tensor<512x512x3x3xf16>,
    %w_s3_b6_c1: tensor<512x512x3x3xf16>,
    %w_s3_b6_c2: tensor<512x512x3x3xf16>,
    %w_s3_b7_c1: tensor<512x512x3x3xf16>,
    %w_s3_b7_c2: tensor<512x512x3x3xf16>,
    %w_s3_b8_c1: tensor<512x512x3x3xf16>,
    %w_s3_b8_c2: tensor<512x512x3x3xf16>,
    %w_s3_b9_c1: tensor<512x512x3x3xf16>,
    %w_s3_b9_c2: tensor<512x512x3x3xf16>,
    %w_s3_b10_c1: tensor<512x512x3x3xf16>,
    %w_s3_b10_c2: tensor<512x512x3x3xf16>,
    %w_s3_b11_c1: tensor<512x512x3x3xf16>,
    %w_s3_b11_c2: tensor<512x512x3x3xf16>,
    %w_s3_b12_c1: tensor<512x512x3x3xf16>,
    %w_s3_b12_c2: tensor<512x512x3x3xf16>,
    %w_s3_b13_c1: tensor<512x512x3x3xf16>,
    %w_s3_b13_c2: tensor<512x512x3x3xf16>,
    %w_s3_b14_c1: tensor<512x512x3x3xf16>,
    %w_s3_b14_c2: tensor<512x512x3x3xf16>,
    %w_s3_b15_c1: tensor<512x512x3x3xf16>,
    %w_s3_b15_c2: tensor<512x512x3x3xf16>,
    %w_s3_b16_c1: tensor<512x512x3x3xf16>,
    %w_s3_b16_c2: tensor<512x512x3x3xf16>,
    %w_s3_b17_c1: tensor<512x512x3x3xf16>,
    %w_s3_b17_c2: tensor<512x512x3x3xf16>,
    %w_fc: tensor<1000x512x7x7xf16>) -> tensor<1x1000x1x1xf16> {
  %cst = arith.constant 0.0 : f16

  // conv1: 7x7 stride 2, 3->64, 224->112
  %pad0 = tensor.pad %input low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x224x224xf16> to tensor<1x3x230x230xf16>
  %init1 = tensor.empty() : tensor<1x64x112x112xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad0, %w_conv1 : tensor<1x3x230x230xf16>, tensor<64x3x7x7xf16>)
    outs(%fill2 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %empty4 = tensor.empty() : tensor<1x64x112x112xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x64x112x112xf16>)
    outs(%empty4 : tensor<1x64x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x112x112xf16>

  // stride-2 3x3 conv simulating maxpool: 112->56
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x112x112xf16> to tensor<1x64x114x114xf16>
  %init7 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad6, %w_pool : tensor<1x64x114x114xf16>, tensor<64x64x3x3xf16>)
    outs(%fill8 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty10 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x64x56x56xf16>)
    outs(%empty10 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // === Stage 0 ===
  // BasicBlock 64->64 56x56 stride=1
  %pad12 = tensor.pad %relu11 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init13 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill14 = linalg.fill ins(%cst : f16) outs(%init13 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv15 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad12, %w_s0_b0_c1 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill14 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty16 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu17 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv15 : tensor<1x64x56x56xf16>)
    outs(%empty16 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad18 = tensor.pad %relu17 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init19 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill20 = linalg.fill ins(%cst : f16) outs(%init19 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv21 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad18, %w_s0_b0_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill20 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty22 = tensor.empty() : tensor<1x64x56x56xf16>
  %add23 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv21, %relu11 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty22 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>
  %empty24 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu25 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add23 : tensor<1x64x56x56xf16>)
    outs(%empty24 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // BasicBlock 64->64 56x56 stride=1
  %pad26 = tensor.pad %relu25 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init27 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill28 = linalg.fill ins(%cst : f16) outs(%init27 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv29 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad26, %w_s0_b1_c1 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill28 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty30 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu31 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv29 : tensor<1x64x56x56xf16>)
    outs(%empty30 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad32 = tensor.pad %relu31 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init33 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill34 = linalg.fill ins(%cst : f16) outs(%init33 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv35 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad32, %w_s0_b1_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill34 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty36 = tensor.empty() : tensor<1x64x56x56xf16>
  %add37 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv35, %relu25 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty36 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>
  %empty38 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu39 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add37 : tensor<1x64x56x56xf16>)
    outs(%empty38 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // BasicBlock 64->64 56x56 stride=1
  %pad40 = tensor.pad %relu39 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init41 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill42 = linalg.fill ins(%cst : f16) outs(%init41 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv43 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad40, %w_s0_b2_c1 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill42 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty44 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu45 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv43 : tensor<1x64x56x56xf16>)
    outs(%empty44 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad46 = tensor.pad %relu45 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init47 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill48 = linalg.fill ins(%cst : f16) outs(%init47 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv49 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad46, %w_s0_b2_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill48 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty50 = tensor.empty() : tensor<1x64x56x56xf16>
  %add51 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv49, %relu39 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty50 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>
  %empty52 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu53 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add51 : tensor<1x64x56x56xf16>)
    outs(%empty52 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // BasicBlock 64->64 56x56 stride=1
  %pad54 = tensor.pad %relu53 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init55 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill56 = linalg.fill ins(%cst : f16) outs(%init55 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv57 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad54, %w_s0_b3_c1 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill56 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty58 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu59 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv57 : tensor<1x64x56x56xf16>)
    outs(%empty58 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad60 = tensor.pad %relu59 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init61 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill62 = linalg.fill ins(%cst : f16) outs(%init61 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv63 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad60, %w_s0_b3_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill62 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty64 = tensor.empty() : tensor<1x64x56x56xf16>
  %add65 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv63, %relu53 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty64 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>
  %empty66 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu67 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add65 : tensor<1x64x56x56xf16>)
    outs(%empty66 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // BasicBlock 64->64 56x56 stride=1
  %pad68 = tensor.pad %relu67 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init69 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill70 = linalg.fill ins(%cst : f16) outs(%init69 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv71 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad68, %w_s0_b4_c1 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill70 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty72 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu73 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv71 : tensor<1x64x56x56xf16>)
    outs(%empty72 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad74 = tensor.pad %relu73 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init75 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill76 = linalg.fill ins(%cst : f16) outs(%init75 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv77 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad74, %w_s0_b4_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill76 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty78 = tensor.empty() : tensor<1x64x56x56xf16>
  %add79 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv77, %relu67 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty78 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>
  %empty80 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu81 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add79 : tensor<1x64x56x56xf16>)
    outs(%empty80 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // BasicBlock 64->64 56x56 stride=1
  %pad82 = tensor.pad %relu81 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init83 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill84 = linalg.fill ins(%cst : f16) outs(%init83 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv85 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad82, %w_s0_b5_c1 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill84 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty86 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu87 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv85 : tensor<1x64x56x56xf16>)
    outs(%empty86 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad88 = tensor.pad %relu87 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init89 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill90 = linalg.fill ins(%cst : f16) outs(%init89 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv91 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad88, %w_s0_b5_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill90 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty92 = tensor.empty() : tensor<1x64x56x56xf16>
  %add93 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv91, %relu81 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty92 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>
  %empty94 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu95 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add93 : tensor<1x64x56x56xf16>)
    outs(%empty94 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // BasicBlock 64->64 56x56 stride=1
  %pad96 = tensor.pad %relu95 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init97 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill98 = linalg.fill ins(%cst : f16) outs(%init97 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv99 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad96, %w_s0_b6_c1 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill98 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty100 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu101 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv99 : tensor<1x64x56x56xf16>)
    outs(%empty100 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad102 = tensor.pad %relu101 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init103 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill104 = linalg.fill ins(%cst : f16) outs(%init103 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv105 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad102, %w_s0_b6_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill104 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty106 = tensor.empty() : tensor<1x64x56x56xf16>
  %add107 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv105, %relu95 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty106 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>
  %empty108 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu109 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add107 : tensor<1x64x56x56xf16>)
    outs(%empty108 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // BasicBlock 64->64 56x56 stride=1
  %pad110 = tensor.pad %relu109 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init111 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill112 = linalg.fill ins(%cst : f16) outs(%init111 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv113 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad110, %w_s0_b7_c1 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill112 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty114 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu115 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv113 : tensor<1x64x56x56xf16>)
    outs(%empty114 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad116 = tensor.pad %relu115 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init117 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill118 = linalg.fill ins(%cst : f16) outs(%init117 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv119 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad116, %w_s0_b7_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill118 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty120 = tensor.empty() : tensor<1x64x56x56xf16>
  %add121 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv119, %relu109 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty120 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>
  %empty122 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu123 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add121 : tensor<1x64x56x56xf16>)
    outs(%empty122 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // BasicBlock 64->64 56x56 stride=1
  %pad124 = tensor.pad %relu123 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init125 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill126 = linalg.fill ins(%cst : f16) outs(%init125 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv127 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad124, %w_s0_b8_c1 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill126 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty128 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu129 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv127 : tensor<1x64x56x56xf16>)
    outs(%empty128 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad130 = tensor.pad %relu129 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init131 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill132 = linalg.fill ins(%cst : f16) outs(%init131 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv133 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad130, %w_s0_b8_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill132 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty134 = tensor.empty() : tensor<1x64x56x56xf16>
  %add135 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv133, %relu123 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty134 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>
  %empty136 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu137 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add135 : tensor<1x64x56x56xf16>)
    outs(%empty136 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // BasicBlock 64->64 56x56 stride=1
  %pad138 = tensor.pad %relu137 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init139 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill140 = linalg.fill ins(%cst : f16) outs(%init139 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv141 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad138, %w_s0_b9_c1 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill140 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty142 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu143 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv141 : tensor<1x64x56x56xf16>)
    outs(%empty142 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad144 = tensor.pad %relu143 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init145 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill146 = linalg.fill ins(%cst : f16) outs(%init145 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv147 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad144, %w_s0_b9_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill146 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty148 = tensor.empty() : tensor<1x64x56x56xf16>
  %add149 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv147, %relu137 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty148 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>
  %empty150 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu151 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add149 : tensor<1x64x56x56xf16>)
    outs(%empty150 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // BasicBlock 64->64 56x56 stride=1
  %pad152 = tensor.pad %relu151 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init153 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill154 = linalg.fill ins(%cst : f16) outs(%init153 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv155 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad152, %w_s0_b10_c1 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill154 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty156 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu157 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv155 : tensor<1x64x56x56xf16>)
    outs(%empty156 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad158 = tensor.pad %relu157 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init159 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill160 = linalg.fill ins(%cst : f16) outs(%init159 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv161 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad158, %w_s0_b10_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill160 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty162 = tensor.empty() : tensor<1x64x56x56xf16>
  %add163 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv161, %relu151 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty162 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>
  %empty164 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu165 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add163 : tensor<1x64x56x56xf16>)
    outs(%empty164 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // BasicBlock 64->64 56x56 stride=1
  %pad166 = tensor.pad %relu165 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init167 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill168 = linalg.fill ins(%cst : f16) outs(%init167 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv169 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad166, %w_s0_b11_c1 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill168 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty170 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu171 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv169 : tensor<1x64x56x56xf16>)
    outs(%empty170 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad172 = tensor.pad %relu171 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init173 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill174 = linalg.fill ins(%cst : f16) outs(%init173 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv175 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad172, %w_s0_b11_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill174 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty176 = tensor.empty() : tensor<1x64x56x56xf16>
  %add177 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv175, %relu165 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty176 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>
  %empty178 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu179 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add177 : tensor<1x64x56x56xf16>)
    outs(%empty178 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // BasicBlock 64->64 56x56 stride=1
  %pad180 = tensor.pad %relu179 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init181 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill182 = linalg.fill ins(%cst : f16) outs(%init181 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv183 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad180, %w_s0_b12_c1 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill182 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty184 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu185 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv183 : tensor<1x64x56x56xf16>)
    outs(%empty184 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad186 = tensor.pad %relu185 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init187 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill188 = linalg.fill ins(%cst : f16) outs(%init187 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv189 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad186, %w_s0_b12_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill188 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty190 = tensor.empty() : tensor<1x64x56x56xf16>
  %add191 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv189, %relu179 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty190 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>
  %empty192 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu193 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add191 : tensor<1x64x56x56xf16>)
    outs(%empty192 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // BasicBlock 64->64 56x56 stride=1
  %pad194 = tensor.pad %relu193 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init195 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill196 = linalg.fill ins(%cst : f16) outs(%init195 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv197 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad194, %w_s0_b13_c1 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill196 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty198 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu199 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv197 : tensor<1x64x56x56xf16>)
    outs(%empty198 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad200 = tensor.pad %relu199 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init201 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill202 = linalg.fill ins(%cst : f16) outs(%init201 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv203 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad200, %w_s0_b13_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill202 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty204 = tensor.empty() : tensor<1x64x56x56xf16>
  %add205 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv203, %relu193 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty204 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>
  %empty206 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu207 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add205 : tensor<1x64x56x56xf16>)
    outs(%empty206 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // BasicBlock 64->64 56x56 stride=1
  %pad208 = tensor.pad %relu207 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init209 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill210 = linalg.fill ins(%cst : f16) outs(%init209 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv211 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad208, %w_s0_b14_c1 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill210 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty212 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu213 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv211 : tensor<1x64x56x56xf16>)
    outs(%empty212 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad214 = tensor.pad %relu213 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init215 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill216 = linalg.fill ins(%cst : f16) outs(%init215 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv217 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad214, %w_s0_b14_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill216 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty218 = tensor.empty() : tensor<1x64x56x56xf16>
  %add219 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv217, %relu207 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty218 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>
  %empty220 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu221 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add219 : tensor<1x64x56x56xf16>)
    outs(%empty220 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // BasicBlock 64->64 56x56 stride=1
  %pad222 = tensor.pad %relu221 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init223 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill224 = linalg.fill ins(%cst : f16) outs(%init223 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv225 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad222, %w_s0_b15_c1 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill224 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty226 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu227 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv225 : tensor<1x64x56x56xf16>)
    outs(%empty226 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad228 = tensor.pad %relu227 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init229 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill230 = linalg.fill ins(%cst : f16) outs(%init229 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv231 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad228, %w_s0_b15_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill230 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty232 = tensor.empty() : tensor<1x64x56x56xf16>
  %add233 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv231, %relu221 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty232 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>
  %empty234 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu235 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add233 : tensor<1x64x56x56xf16>)
    outs(%empty234 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // BasicBlock 64->64 56x56 stride=1
  %pad236 = tensor.pad %relu235 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init237 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill238 = linalg.fill ins(%cst : f16) outs(%init237 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv239 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad236, %w_s0_b16_c1 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill238 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty240 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu241 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv239 : tensor<1x64x56x56xf16>)
    outs(%empty240 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad242 = tensor.pad %relu241 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init243 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill244 = linalg.fill ins(%cst : f16) outs(%init243 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv245 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad242, %w_s0_b16_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill244 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty246 = tensor.empty() : tensor<1x64x56x56xf16>
  %add247 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv245, %relu235 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty246 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>
  %empty248 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu249 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add247 : tensor<1x64x56x56xf16>)
    outs(%empty248 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // BasicBlock 64->64 56x56 stride=1
  %pad250 = tensor.pad %relu249 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init251 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill252 = linalg.fill ins(%cst : f16) outs(%init251 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv253 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad250, %w_s0_b17_c1 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill252 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty254 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu255 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv253 : tensor<1x64x56x56xf16>)
    outs(%empty254 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad256 = tensor.pad %relu255 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init257 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill258 = linalg.fill ins(%cst : f16) outs(%init257 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv259 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad256, %w_s0_b17_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill258 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty260 = tensor.empty() : tensor<1x64x56x56xf16>
  %add261 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv259, %relu249 : tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>)
    outs(%empty260 : tensor<1x64x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x64x56x56xf16>
  %empty262 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu263 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add261 : tensor<1x64x56x56xf16>)
    outs(%empty262 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // === Stage 1 ===
  // BasicBlock 64->128 56x56 stride=2
  %pad264 = tensor.pad %relu263 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init265 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill266 = linalg.fill ins(%cst : f16) outs(%init265 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv267 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad264, %w_s1_b0_c1 : tensor<1x64x58x58xf16>, tensor<128x64x3x3xf16>)
    outs(%fill266 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty268 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu269 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv267 : tensor<1x128x28x28xf16>)
    outs(%empty268 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad270 = tensor.pad %relu269 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init271 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill272 = linalg.fill ins(%cst : f16) outs(%init271 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv273 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad270, %w_s1_b0_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill272 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  // 1x1 shortcut conv
  %init274 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill275 = linalg.fill ins(%cst : f16) outs(%init274 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv276 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu263, %w_s1_b0_sc : tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>)
    outs(%fill275 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty277 = tensor.empty() : tensor<1x128x28x28xf16>
  %add278 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv273, %conv276 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty277 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty279 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu280 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add278 : tensor<1x128x28x28xf16>)
    outs(%empty279 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad281 = tensor.pad %relu280 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init282 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill283 = linalg.fill ins(%cst : f16) outs(%init282 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv284 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad281, %w_s1_b1_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill283 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty285 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu286 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv284 : tensor<1x128x28x28xf16>)
    outs(%empty285 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad287 = tensor.pad %relu286 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init288 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill289 = linalg.fill ins(%cst : f16) outs(%init288 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv290 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad287, %w_s1_b1_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill289 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty291 = tensor.empty() : tensor<1x128x28x28xf16>
  %add292 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv290, %relu280 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty291 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty293 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu294 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add292 : tensor<1x128x28x28xf16>)
    outs(%empty293 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad295 = tensor.pad %relu294 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init296 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill297 = linalg.fill ins(%cst : f16) outs(%init296 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv298 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad295, %w_s1_b2_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill297 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty299 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu300 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv298 : tensor<1x128x28x28xf16>)
    outs(%empty299 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad301 = tensor.pad %relu300 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init302 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill303 = linalg.fill ins(%cst : f16) outs(%init302 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv304 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad301, %w_s1_b2_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill303 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty305 = tensor.empty() : tensor<1x128x28x28xf16>
  %add306 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv304, %relu294 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty305 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty307 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu308 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add306 : tensor<1x128x28x28xf16>)
    outs(%empty307 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad309 = tensor.pad %relu308 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init310 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill311 = linalg.fill ins(%cst : f16) outs(%init310 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv312 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad309, %w_s1_b3_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill311 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty313 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu314 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv312 : tensor<1x128x28x28xf16>)
    outs(%empty313 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad315 = tensor.pad %relu314 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init316 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill317 = linalg.fill ins(%cst : f16) outs(%init316 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv318 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad315, %w_s1_b3_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill317 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty319 = tensor.empty() : tensor<1x128x28x28xf16>
  %add320 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv318, %relu308 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty319 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty321 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu322 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add320 : tensor<1x128x28x28xf16>)
    outs(%empty321 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad323 = tensor.pad %relu322 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init324 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill325 = linalg.fill ins(%cst : f16) outs(%init324 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv326 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad323, %w_s1_b4_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill325 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty327 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu328 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv326 : tensor<1x128x28x28xf16>)
    outs(%empty327 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad329 = tensor.pad %relu328 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init330 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill331 = linalg.fill ins(%cst : f16) outs(%init330 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv332 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad329, %w_s1_b4_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill331 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty333 = tensor.empty() : tensor<1x128x28x28xf16>
  %add334 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv332, %relu322 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty333 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty335 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu336 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add334 : tensor<1x128x28x28xf16>)
    outs(%empty335 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad337 = tensor.pad %relu336 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init338 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill339 = linalg.fill ins(%cst : f16) outs(%init338 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv340 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad337, %w_s1_b5_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill339 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty341 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu342 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv340 : tensor<1x128x28x28xf16>)
    outs(%empty341 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad343 = tensor.pad %relu342 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init344 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill345 = linalg.fill ins(%cst : f16) outs(%init344 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv346 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad343, %w_s1_b5_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill345 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty347 = tensor.empty() : tensor<1x128x28x28xf16>
  %add348 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv346, %relu336 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty347 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty349 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu350 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add348 : tensor<1x128x28x28xf16>)
    outs(%empty349 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad351 = tensor.pad %relu350 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init352 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill353 = linalg.fill ins(%cst : f16) outs(%init352 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv354 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad351, %w_s1_b6_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill353 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty355 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu356 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv354 : tensor<1x128x28x28xf16>)
    outs(%empty355 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad357 = tensor.pad %relu356 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init358 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill359 = linalg.fill ins(%cst : f16) outs(%init358 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv360 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad357, %w_s1_b6_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill359 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty361 = tensor.empty() : tensor<1x128x28x28xf16>
  %add362 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv360, %relu350 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty361 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty363 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu364 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add362 : tensor<1x128x28x28xf16>)
    outs(%empty363 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad365 = tensor.pad %relu364 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init366 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill367 = linalg.fill ins(%cst : f16) outs(%init366 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv368 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad365, %w_s1_b7_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill367 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty369 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu370 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv368 : tensor<1x128x28x28xf16>)
    outs(%empty369 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad371 = tensor.pad %relu370 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init372 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill373 = linalg.fill ins(%cst : f16) outs(%init372 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv374 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad371, %w_s1_b7_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill373 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty375 = tensor.empty() : tensor<1x128x28x28xf16>
  %add376 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv374, %relu364 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty375 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty377 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu378 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add376 : tensor<1x128x28x28xf16>)
    outs(%empty377 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad379 = tensor.pad %relu378 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init380 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill381 = linalg.fill ins(%cst : f16) outs(%init380 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv382 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad379, %w_s1_b8_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill381 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty383 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu384 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv382 : tensor<1x128x28x28xf16>)
    outs(%empty383 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad385 = tensor.pad %relu384 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init386 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill387 = linalg.fill ins(%cst : f16) outs(%init386 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv388 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad385, %w_s1_b8_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill387 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty389 = tensor.empty() : tensor<1x128x28x28xf16>
  %add390 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv388, %relu378 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty389 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty391 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu392 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add390 : tensor<1x128x28x28xf16>)
    outs(%empty391 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad393 = tensor.pad %relu392 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init394 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill395 = linalg.fill ins(%cst : f16) outs(%init394 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv396 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad393, %w_s1_b9_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill395 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty397 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu398 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv396 : tensor<1x128x28x28xf16>)
    outs(%empty397 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad399 = tensor.pad %relu398 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init400 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill401 = linalg.fill ins(%cst : f16) outs(%init400 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv402 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad399, %w_s1_b9_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill401 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty403 = tensor.empty() : tensor<1x128x28x28xf16>
  %add404 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv402, %relu392 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty403 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty405 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu406 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add404 : tensor<1x128x28x28xf16>)
    outs(%empty405 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad407 = tensor.pad %relu406 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init408 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill409 = linalg.fill ins(%cst : f16) outs(%init408 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv410 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad407, %w_s1_b10_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill409 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty411 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu412 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv410 : tensor<1x128x28x28xf16>)
    outs(%empty411 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad413 = tensor.pad %relu412 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init414 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill415 = linalg.fill ins(%cst : f16) outs(%init414 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv416 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad413, %w_s1_b10_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill415 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty417 = tensor.empty() : tensor<1x128x28x28xf16>
  %add418 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv416, %relu406 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty417 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty419 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu420 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add418 : tensor<1x128x28x28xf16>)
    outs(%empty419 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad421 = tensor.pad %relu420 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init422 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill423 = linalg.fill ins(%cst : f16) outs(%init422 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv424 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad421, %w_s1_b11_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill423 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty425 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu426 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv424 : tensor<1x128x28x28xf16>)
    outs(%empty425 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad427 = tensor.pad %relu426 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init428 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill429 = linalg.fill ins(%cst : f16) outs(%init428 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv430 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad427, %w_s1_b11_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill429 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty431 = tensor.empty() : tensor<1x128x28x28xf16>
  %add432 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv430, %relu420 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty431 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty433 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu434 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add432 : tensor<1x128x28x28xf16>)
    outs(%empty433 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad435 = tensor.pad %relu434 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init436 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill437 = linalg.fill ins(%cst : f16) outs(%init436 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv438 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad435, %w_s1_b12_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill437 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty439 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu440 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv438 : tensor<1x128x28x28xf16>)
    outs(%empty439 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad441 = tensor.pad %relu440 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init442 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill443 = linalg.fill ins(%cst : f16) outs(%init442 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv444 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad441, %w_s1_b12_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill443 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty445 = tensor.empty() : tensor<1x128x28x28xf16>
  %add446 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv444, %relu434 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty445 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty447 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu448 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add446 : tensor<1x128x28x28xf16>)
    outs(%empty447 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad449 = tensor.pad %relu448 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init450 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill451 = linalg.fill ins(%cst : f16) outs(%init450 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv452 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad449, %w_s1_b13_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill451 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty453 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu454 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv452 : tensor<1x128x28x28xf16>)
    outs(%empty453 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad455 = tensor.pad %relu454 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init456 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill457 = linalg.fill ins(%cst : f16) outs(%init456 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv458 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad455, %w_s1_b13_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill457 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty459 = tensor.empty() : tensor<1x128x28x28xf16>
  %add460 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv458, %relu448 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty459 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty461 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu462 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add460 : tensor<1x128x28x28xf16>)
    outs(%empty461 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad463 = tensor.pad %relu462 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init464 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill465 = linalg.fill ins(%cst : f16) outs(%init464 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv466 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad463, %w_s1_b14_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill465 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty467 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu468 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv466 : tensor<1x128x28x28xf16>)
    outs(%empty467 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad469 = tensor.pad %relu468 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init470 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill471 = linalg.fill ins(%cst : f16) outs(%init470 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv472 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad469, %w_s1_b14_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill471 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty473 = tensor.empty() : tensor<1x128x28x28xf16>
  %add474 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv472, %relu462 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty473 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty475 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu476 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add474 : tensor<1x128x28x28xf16>)
    outs(%empty475 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad477 = tensor.pad %relu476 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init478 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill479 = linalg.fill ins(%cst : f16) outs(%init478 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv480 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad477, %w_s1_b15_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill479 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty481 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu482 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv480 : tensor<1x128x28x28xf16>)
    outs(%empty481 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad483 = tensor.pad %relu482 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init484 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill485 = linalg.fill ins(%cst : f16) outs(%init484 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv486 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad483, %w_s1_b15_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill485 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty487 = tensor.empty() : tensor<1x128x28x28xf16>
  %add488 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv486, %relu476 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty487 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty489 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu490 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add488 : tensor<1x128x28x28xf16>)
    outs(%empty489 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad491 = tensor.pad %relu490 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init492 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill493 = linalg.fill ins(%cst : f16) outs(%init492 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv494 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad491, %w_s1_b16_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill493 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty495 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu496 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv494 : tensor<1x128x28x28xf16>)
    outs(%empty495 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad497 = tensor.pad %relu496 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init498 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill499 = linalg.fill ins(%cst : f16) outs(%init498 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv500 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad497, %w_s1_b16_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill499 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty501 = tensor.empty() : tensor<1x128x28x28xf16>
  %add502 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv500, %relu490 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty501 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty503 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu504 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add502 : tensor<1x128x28x28xf16>)
    outs(%empty503 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad505 = tensor.pad %relu504 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init506 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill507 = linalg.fill ins(%cst : f16) outs(%init506 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv508 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad505, %w_s1_b17_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill507 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty509 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu510 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv508 : tensor<1x128x28x28xf16>)
    outs(%empty509 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad511 = tensor.pad %relu510 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init512 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill513 = linalg.fill ins(%cst : f16) outs(%init512 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv514 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad511, %w_s1_b17_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill513 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty515 = tensor.empty() : tensor<1x128x28x28xf16>
  %add516 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv514, %relu504 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty515 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty517 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu518 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add516 : tensor<1x128x28x28xf16>)
    outs(%empty517 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // === Stage 2 ===
  // BasicBlock 128->256 28x28 stride=2
  %pad519 = tensor.pad %relu518 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init520 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill521 = linalg.fill ins(%cst : f16) outs(%init520 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv522 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad519, %w_s2_b0_c1 : tensor<1x128x30x30xf16>, tensor<256x128x3x3xf16>)
    outs(%fill521 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty523 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu524 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv522 : tensor<1x256x14x14xf16>)
    outs(%empty523 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad525 = tensor.pad %relu524 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init526 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill527 = linalg.fill ins(%cst : f16) outs(%init526 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv528 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad525, %w_s2_b0_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill527 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  // 1x1 shortcut conv
  %init529 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill530 = linalg.fill ins(%cst : f16) outs(%init529 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv531 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu518, %w_s2_b0_sc : tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>)
    outs(%fill530 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty532 = tensor.empty() : tensor<1x256x14x14xf16>
  %add533 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv528, %conv531 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty532 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty534 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu535 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add533 : tensor<1x256x14x14xf16>)
    outs(%empty534 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad536 = tensor.pad %relu535 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init537 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill538 = linalg.fill ins(%cst : f16) outs(%init537 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv539 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad536, %w_s2_b1_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill538 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty540 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu541 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv539 : tensor<1x256x14x14xf16>)
    outs(%empty540 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad542 = tensor.pad %relu541 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init543 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill544 = linalg.fill ins(%cst : f16) outs(%init543 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv545 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad542, %w_s2_b1_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill544 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty546 = tensor.empty() : tensor<1x256x14x14xf16>
  %add547 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv545, %relu535 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty546 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty548 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu549 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add547 : tensor<1x256x14x14xf16>)
    outs(%empty548 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad550 = tensor.pad %relu549 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init551 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill552 = linalg.fill ins(%cst : f16) outs(%init551 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv553 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad550, %w_s2_b2_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill552 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty554 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu555 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv553 : tensor<1x256x14x14xf16>)
    outs(%empty554 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad556 = tensor.pad %relu555 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init557 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill558 = linalg.fill ins(%cst : f16) outs(%init557 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv559 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad556, %w_s2_b2_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill558 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty560 = tensor.empty() : tensor<1x256x14x14xf16>
  %add561 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv559, %relu549 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty560 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty562 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu563 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add561 : tensor<1x256x14x14xf16>)
    outs(%empty562 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad564 = tensor.pad %relu563 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init565 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill566 = linalg.fill ins(%cst : f16) outs(%init565 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv567 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad564, %w_s2_b3_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill566 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty568 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu569 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv567 : tensor<1x256x14x14xf16>)
    outs(%empty568 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad570 = tensor.pad %relu569 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init571 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill572 = linalg.fill ins(%cst : f16) outs(%init571 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv573 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad570, %w_s2_b3_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill572 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty574 = tensor.empty() : tensor<1x256x14x14xf16>
  %add575 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv573, %relu563 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty574 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty576 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu577 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add575 : tensor<1x256x14x14xf16>)
    outs(%empty576 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad578 = tensor.pad %relu577 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init579 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill580 = linalg.fill ins(%cst : f16) outs(%init579 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv581 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad578, %w_s2_b4_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill580 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty582 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu583 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv581 : tensor<1x256x14x14xf16>)
    outs(%empty582 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad584 = tensor.pad %relu583 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init585 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill586 = linalg.fill ins(%cst : f16) outs(%init585 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv587 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad584, %w_s2_b4_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill586 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty588 = tensor.empty() : tensor<1x256x14x14xf16>
  %add589 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv587, %relu577 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty588 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty590 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu591 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add589 : tensor<1x256x14x14xf16>)
    outs(%empty590 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad592 = tensor.pad %relu591 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init593 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill594 = linalg.fill ins(%cst : f16) outs(%init593 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv595 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad592, %w_s2_b5_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill594 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty596 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu597 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv595 : tensor<1x256x14x14xf16>)
    outs(%empty596 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad598 = tensor.pad %relu597 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init599 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill600 = linalg.fill ins(%cst : f16) outs(%init599 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv601 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad598, %w_s2_b5_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill600 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty602 = tensor.empty() : tensor<1x256x14x14xf16>
  %add603 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv601, %relu591 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty602 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty604 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu605 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add603 : tensor<1x256x14x14xf16>)
    outs(%empty604 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad606 = tensor.pad %relu605 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init607 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill608 = linalg.fill ins(%cst : f16) outs(%init607 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv609 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad606, %w_s2_b6_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill608 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty610 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu611 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv609 : tensor<1x256x14x14xf16>)
    outs(%empty610 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad612 = tensor.pad %relu611 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init613 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill614 = linalg.fill ins(%cst : f16) outs(%init613 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv615 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad612, %w_s2_b6_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill614 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty616 = tensor.empty() : tensor<1x256x14x14xf16>
  %add617 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv615, %relu605 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty616 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty618 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu619 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add617 : tensor<1x256x14x14xf16>)
    outs(%empty618 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad620 = tensor.pad %relu619 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init621 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill622 = linalg.fill ins(%cst : f16) outs(%init621 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv623 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad620, %w_s2_b7_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill622 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty624 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu625 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv623 : tensor<1x256x14x14xf16>)
    outs(%empty624 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad626 = tensor.pad %relu625 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init627 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill628 = linalg.fill ins(%cst : f16) outs(%init627 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv629 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad626, %w_s2_b7_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill628 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty630 = tensor.empty() : tensor<1x256x14x14xf16>
  %add631 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv629, %relu619 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty630 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty632 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu633 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add631 : tensor<1x256x14x14xf16>)
    outs(%empty632 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad634 = tensor.pad %relu633 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init635 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill636 = linalg.fill ins(%cst : f16) outs(%init635 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv637 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad634, %w_s2_b8_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill636 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty638 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu639 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv637 : tensor<1x256x14x14xf16>)
    outs(%empty638 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad640 = tensor.pad %relu639 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init641 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill642 = linalg.fill ins(%cst : f16) outs(%init641 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv643 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad640, %w_s2_b8_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill642 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty644 = tensor.empty() : tensor<1x256x14x14xf16>
  %add645 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv643, %relu633 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty644 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty646 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu647 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add645 : tensor<1x256x14x14xf16>)
    outs(%empty646 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad648 = tensor.pad %relu647 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init649 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill650 = linalg.fill ins(%cst : f16) outs(%init649 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv651 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad648, %w_s2_b9_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill650 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty652 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu653 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv651 : tensor<1x256x14x14xf16>)
    outs(%empty652 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad654 = tensor.pad %relu653 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init655 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill656 = linalg.fill ins(%cst : f16) outs(%init655 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv657 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad654, %w_s2_b9_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill656 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty658 = tensor.empty() : tensor<1x256x14x14xf16>
  %add659 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv657, %relu647 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty658 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty660 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu661 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add659 : tensor<1x256x14x14xf16>)
    outs(%empty660 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad662 = tensor.pad %relu661 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init663 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill664 = linalg.fill ins(%cst : f16) outs(%init663 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv665 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad662, %w_s2_b10_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill664 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty666 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu667 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv665 : tensor<1x256x14x14xf16>)
    outs(%empty666 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad668 = tensor.pad %relu667 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init669 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill670 = linalg.fill ins(%cst : f16) outs(%init669 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv671 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad668, %w_s2_b10_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill670 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty672 = tensor.empty() : tensor<1x256x14x14xf16>
  %add673 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv671, %relu661 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty672 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty674 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu675 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add673 : tensor<1x256x14x14xf16>)
    outs(%empty674 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad676 = tensor.pad %relu675 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init677 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill678 = linalg.fill ins(%cst : f16) outs(%init677 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv679 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad676, %w_s2_b11_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill678 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty680 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu681 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv679 : tensor<1x256x14x14xf16>)
    outs(%empty680 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad682 = tensor.pad %relu681 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init683 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill684 = linalg.fill ins(%cst : f16) outs(%init683 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv685 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad682, %w_s2_b11_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill684 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty686 = tensor.empty() : tensor<1x256x14x14xf16>
  %add687 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv685, %relu675 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty686 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty688 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu689 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add687 : tensor<1x256x14x14xf16>)
    outs(%empty688 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad690 = tensor.pad %relu689 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init691 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill692 = linalg.fill ins(%cst : f16) outs(%init691 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv693 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad690, %w_s2_b12_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill692 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty694 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu695 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv693 : tensor<1x256x14x14xf16>)
    outs(%empty694 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad696 = tensor.pad %relu695 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init697 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill698 = linalg.fill ins(%cst : f16) outs(%init697 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv699 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad696, %w_s2_b12_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill698 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty700 = tensor.empty() : tensor<1x256x14x14xf16>
  %add701 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv699, %relu689 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty700 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty702 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu703 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add701 : tensor<1x256x14x14xf16>)
    outs(%empty702 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad704 = tensor.pad %relu703 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init705 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill706 = linalg.fill ins(%cst : f16) outs(%init705 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv707 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad704, %w_s2_b13_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill706 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty708 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu709 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv707 : tensor<1x256x14x14xf16>)
    outs(%empty708 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad710 = tensor.pad %relu709 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init711 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill712 = linalg.fill ins(%cst : f16) outs(%init711 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv713 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad710, %w_s2_b13_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill712 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty714 = tensor.empty() : tensor<1x256x14x14xf16>
  %add715 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv713, %relu703 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty714 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty716 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu717 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add715 : tensor<1x256x14x14xf16>)
    outs(%empty716 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad718 = tensor.pad %relu717 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init719 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill720 = linalg.fill ins(%cst : f16) outs(%init719 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv721 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad718, %w_s2_b14_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill720 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty722 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu723 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv721 : tensor<1x256x14x14xf16>)
    outs(%empty722 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad724 = tensor.pad %relu723 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init725 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill726 = linalg.fill ins(%cst : f16) outs(%init725 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv727 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad724, %w_s2_b14_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill726 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty728 = tensor.empty() : tensor<1x256x14x14xf16>
  %add729 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv727, %relu717 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty728 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty730 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu731 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add729 : tensor<1x256x14x14xf16>)
    outs(%empty730 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad732 = tensor.pad %relu731 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init733 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill734 = linalg.fill ins(%cst : f16) outs(%init733 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv735 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad732, %w_s2_b15_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill734 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty736 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu737 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv735 : tensor<1x256x14x14xf16>)
    outs(%empty736 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad738 = tensor.pad %relu737 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init739 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill740 = linalg.fill ins(%cst : f16) outs(%init739 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv741 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad738, %w_s2_b15_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill740 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty742 = tensor.empty() : tensor<1x256x14x14xf16>
  %add743 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv741, %relu731 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty742 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty744 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu745 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add743 : tensor<1x256x14x14xf16>)
    outs(%empty744 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad746 = tensor.pad %relu745 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init747 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill748 = linalg.fill ins(%cst : f16) outs(%init747 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv749 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad746, %w_s2_b16_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill748 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty750 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu751 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv749 : tensor<1x256x14x14xf16>)
    outs(%empty750 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad752 = tensor.pad %relu751 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init753 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill754 = linalg.fill ins(%cst : f16) outs(%init753 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv755 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad752, %w_s2_b16_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill754 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty756 = tensor.empty() : tensor<1x256x14x14xf16>
  %add757 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv755, %relu745 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty756 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty758 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu759 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add757 : tensor<1x256x14x14xf16>)
    outs(%empty758 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad760 = tensor.pad %relu759 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init761 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill762 = linalg.fill ins(%cst : f16) outs(%init761 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv763 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad760, %w_s2_b17_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill762 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty764 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu765 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv763 : tensor<1x256x14x14xf16>)
    outs(%empty764 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad766 = tensor.pad %relu765 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init767 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill768 = linalg.fill ins(%cst : f16) outs(%init767 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv769 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad766, %w_s2_b17_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill768 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty770 = tensor.empty() : tensor<1x256x14x14xf16>
  %add771 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv769, %relu759 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty770 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty772 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu773 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add771 : tensor<1x256x14x14xf16>)
    outs(%empty772 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // === Stage 3 ===
  // BasicBlock 256->512 14x14 stride=2
  %pad774 = tensor.pad %relu773 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init775 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill776 = linalg.fill ins(%cst : f16) outs(%init775 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv777 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad774, %w_s3_b0_c1 : tensor<1x256x16x16xf16>, tensor<512x256x3x3xf16>)
    outs(%fill776 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty778 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu779 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv777 : tensor<1x512x7x7xf16>)
    outs(%empty778 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad780 = tensor.pad %relu779 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init781 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill782 = linalg.fill ins(%cst : f16) outs(%init781 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv783 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad780, %w_s3_b0_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill782 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  // 1x1 shortcut conv
  %init784 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill785 = linalg.fill ins(%cst : f16) outs(%init784 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv786 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu773, %w_s3_b0_sc : tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>)
    outs(%fill785 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty787 = tensor.empty() : tensor<1x512x7x7xf16>
  %add788 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv783, %conv786 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty787 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty789 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu790 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add788 : tensor<1x512x7x7xf16>)
    outs(%empty789 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad791 = tensor.pad %relu790 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init792 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill793 = linalg.fill ins(%cst : f16) outs(%init792 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv794 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad791, %w_s3_b1_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill793 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty795 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu796 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv794 : tensor<1x512x7x7xf16>)
    outs(%empty795 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad797 = tensor.pad %relu796 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init798 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill799 = linalg.fill ins(%cst : f16) outs(%init798 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv800 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad797, %w_s3_b1_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill799 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty801 = tensor.empty() : tensor<1x512x7x7xf16>
  %add802 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv800, %relu790 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty801 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty803 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu804 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add802 : tensor<1x512x7x7xf16>)
    outs(%empty803 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad805 = tensor.pad %relu804 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init806 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill807 = linalg.fill ins(%cst : f16) outs(%init806 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv808 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad805, %w_s3_b2_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill807 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty809 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu810 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv808 : tensor<1x512x7x7xf16>)
    outs(%empty809 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad811 = tensor.pad %relu810 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init812 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill813 = linalg.fill ins(%cst : f16) outs(%init812 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv814 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad811, %w_s3_b2_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill813 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty815 = tensor.empty() : tensor<1x512x7x7xf16>
  %add816 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv814, %relu804 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty815 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty817 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu818 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add816 : tensor<1x512x7x7xf16>)
    outs(%empty817 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad819 = tensor.pad %relu818 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init820 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill821 = linalg.fill ins(%cst : f16) outs(%init820 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv822 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad819, %w_s3_b3_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill821 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty823 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu824 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv822 : tensor<1x512x7x7xf16>)
    outs(%empty823 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad825 = tensor.pad %relu824 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init826 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill827 = linalg.fill ins(%cst : f16) outs(%init826 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv828 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad825, %w_s3_b3_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill827 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty829 = tensor.empty() : tensor<1x512x7x7xf16>
  %add830 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv828, %relu818 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty829 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty831 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu832 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add830 : tensor<1x512x7x7xf16>)
    outs(%empty831 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad833 = tensor.pad %relu832 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init834 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill835 = linalg.fill ins(%cst : f16) outs(%init834 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv836 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad833, %w_s3_b4_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill835 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty837 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu838 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv836 : tensor<1x512x7x7xf16>)
    outs(%empty837 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad839 = tensor.pad %relu838 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init840 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill841 = linalg.fill ins(%cst : f16) outs(%init840 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv842 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad839, %w_s3_b4_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill841 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty843 = tensor.empty() : tensor<1x512x7x7xf16>
  %add844 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv842, %relu832 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty843 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty845 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu846 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add844 : tensor<1x512x7x7xf16>)
    outs(%empty845 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad847 = tensor.pad %relu846 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init848 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill849 = linalg.fill ins(%cst : f16) outs(%init848 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv850 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad847, %w_s3_b5_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill849 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty851 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu852 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv850 : tensor<1x512x7x7xf16>)
    outs(%empty851 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad853 = tensor.pad %relu852 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init854 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill855 = linalg.fill ins(%cst : f16) outs(%init854 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv856 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad853, %w_s3_b5_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill855 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty857 = tensor.empty() : tensor<1x512x7x7xf16>
  %add858 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv856, %relu846 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty857 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty859 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu860 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add858 : tensor<1x512x7x7xf16>)
    outs(%empty859 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad861 = tensor.pad %relu860 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init862 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill863 = linalg.fill ins(%cst : f16) outs(%init862 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv864 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad861, %w_s3_b6_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill863 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty865 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu866 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv864 : tensor<1x512x7x7xf16>)
    outs(%empty865 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad867 = tensor.pad %relu866 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init868 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill869 = linalg.fill ins(%cst : f16) outs(%init868 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv870 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad867, %w_s3_b6_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill869 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty871 = tensor.empty() : tensor<1x512x7x7xf16>
  %add872 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv870, %relu860 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty871 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty873 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu874 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add872 : tensor<1x512x7x7xf16>)
    outs(%empty873 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad875 = tensor.pad %relu874 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init876 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill877 = linalg.fill ins(%cst : f16) outs(%init876 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv878 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad875, %w_s3_b7_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill877 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty879 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu880 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv878 : tensor<1x512x7x7xf16>)
    outs(%empty879 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad881 = tensor.pad %relu880 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init882 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill883 = linalg.fill ins(%cst : f16) outs(%init882 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv884 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad881, %w_s3_b7_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill883 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty885 = tensor.empty() : tensor<1x512x7x7xf16>
  %add886 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv884, %relu874 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty885 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty887 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu888 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add886 : tensor<1x512x7x7xf16>)
    outs(%empty887 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad889 = tensor.pad %relu888 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init890 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill891 = linalg.fill ins(%cst : f16) outs(%init890 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv892 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad889, %w_s3_b8_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill891 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty893 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu894 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv892 : tensor<1x512x7x7xf16>)
    outs(%empty893 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad895 = tensor.pad %relu894 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init896 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill897 = linalg.fill ins(%cst : f16) outs(%init896 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv898 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad895, %w_s3_b8_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill897 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty899 = tensor.empty() : tensor<1x512x7x7xf16>
  %add900 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv898, %relu888 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty899 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty901 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu902 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add900 : tensor<1x512x7x7xf16>)
    outs(%empty901 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad903 = tensor.pad %relu902 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init904 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill905 = linalg.fill ins(%cst : f16) outs(%init904 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv906 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad903, %w_s3_b9_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill905 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty907 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu908 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv906 : tensor<1x512x7x7xf16>)
    outs(%empty907 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad909 = tensor.pad %relu908 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init910 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill911 = linalg.fill ins(%cst : f16) outs(%init910 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv912 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad909, %w_s3_b9_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill911 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty913 = tensor.empty() : tensor<1x512x7x7xf16>
  %add914 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv912, %relu902 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty913 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty915 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu916 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add914 : tensor<1x512x7x7xf16>)
    outs(%empty915 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad917 = tensor.pad %relu916 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init918 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill919 = linalg.fill ins(%cst : f16) outs(%init918 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv920 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad917, %w_s3_b10_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill919 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty921 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu922 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv920 : tensor<1x512x7x7xf16>)
    outs(%empty921 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad923 = tensor.pad %relu922 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init924 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill925 = linalg.fill ins(%cst : f16) outs(%init924 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv926 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad923, %w_s3_b10_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill925 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty927 = tensor.empty() : tensor<1x512x7x7xf16>
  %add928 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv926, %relu916 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty927 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty929 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu930 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add928 : tensor<1x512x7x7xf16>)
    outs(%empty929 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad931 = tensor.pad %relu930 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init932 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill933 = linalg.fill ins(%cst : f16) outs(%init932 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv934 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad931, %w_s3_b11_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill933 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty935 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu936 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv934 : tensor<1x512x7x7xf16>)
    outs(%empty935 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad937 = tensor.pad %relu936 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init938 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill939 = linalg.fill ins(%cst : f16) outs(%init938 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv940 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad937, %w_s3_b11_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill939 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty941 = tensor.empty() : tensor<1x512x7x7xf16>
  %add942 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv940, %relu930 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty941 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty943 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu944 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add942 : tensor<1x512x7x7xf16>)
    outs(%empty943 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad945 = tensor.pad %relu944 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init946 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill947 = linalg.fill ins(%cst : f16) outs(%init946 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv948 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad945, %w_s3_b12_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill947 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty949 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu950 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv948 : tensor<1x512x7x7xf16>)
    outs(%empty949 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad951 = tensor.pad %relu950 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init952 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill953 = linalg.fill ins(%cst : f16) outs(%init952 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv954 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad951, %w_s3_b12_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill953 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty955 = tensor.empty() : tensor<1x512x7x7xf16>
  %add956 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv954, %relu944 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty955 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty957 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu958 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add956 : tensor<1x512x7x7xf16>)
    outs(%empty957 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad959 = tensor.pad %relu958 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init960 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill961 = linalg.fill ins(%cst : f16) outs(%init960 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv962 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad959, %w_s3_b13_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill961 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty963 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu964 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv962 : tensor<1x512x7x7xf16>)
    outs(%empty963 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad965 = tensor.pad %relu964 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init966 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill967 = linalg.fill ins(%cst : f16) outs(%init966 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv968 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad965, %w_s3_b13_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill967 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty969 = tensor.empty() : tensor<1x512x7x7xf16>
  %add970 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv968, %relu958 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty969 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty971 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu972 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add970 : tensor<1x512x7x7xf16>)
    outs(%empty971 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad973 = tensor.pad %relu972 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init974 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill975 = linalg.fill ins(%cst : f16) outs(%init974 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv976 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad973, %w_s3_b14_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill975 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty977 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu978 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv976 : tensor<1x512x7x7xf16>)
    outs(%empty977 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad979 = tensor.pad %relu978 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init980 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill981 = linalg.fill ins(%cst : f16) outs(%init980 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv982 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad979, %w_s3_b14_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill981 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty983 = tensor.empty() : tensor<1x512x7x7xf16>
  %add984 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv982, %relu972 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty983 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty985 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu986 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add984 : tensor<1x512x7x7xf16>)
    outs(%empty985 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad987 = tensor.pad %relu986 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init988 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill989 = linalg.fill ins(%cst : f16) outs(%init988 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv990 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad987, %w_s3_b15_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill989 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty991 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu992 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv990 : tensor<1x512x7x7xf16>)
    outs(%empty991 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad993 = tensor.pad %relu992 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init994 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill995 = linalg.fill ins(%cst : f16) outs(%init994 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv996 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad993, %w_s3_b15_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill995 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty997 = tensor.empty() : tensor<1x512x7x7xf16>
  %add998 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv996, %relu986 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty997 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty999 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu1000 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add998 : tensor<1x512x7x7xf16>)
    outs(%empty999 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad1001 = tensor.pad %relu1000 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init1002 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill1003 = linalg.fill ins(%cst : f16) outs(%init1002 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv1004 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1001, %w_s3_b16_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill1003 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty1005 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu1006 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1004 : tensor<1x512x7x7xf16>)
    outs(%empty1005 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad1007 = tensor.pad %relu1006 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init1008 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill1009 = linalg.fill ins(%cst : f16) outs(%init1008 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv1010 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1007, %w_s3_b16_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill1009 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty1011 = tensor.empty() : tensor<1x512x7x7xf16>
  %add1012 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1010, %relu1000 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty1011 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty1013 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu1014 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add1012 : tensor<1x512x7x7xf16>)
    outs(%empty1013 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad1015 = tensor.pad %relu1014 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init1016 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill1017 = linalg.fill ins(%cst : f16) outs(%init1016 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv1018 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1015, %w_s3_b17_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill1017 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty1019 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu1020 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1018 : tensor<1x512x7x7xf16>)
    outs(%empty1019 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad1021 = tensor.pad %relu1020 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init1022 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill1023 = linalg.fill ins(%cst : f16) outs(%init1022 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv1024 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1021, %w_s3_b17_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill1023 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty1025 = tensor.empty() : tensor<1x512x7x7xf16>
  %add1026 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1024, %relu1014 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty1025 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty1027 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu1028 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add1026 : tensor<1x512x7x7xf16>)
    outs(%empty1027 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // FC as 7x7 conv: 512->1000
  %fc_init1029 = tensor.empty() : tensor<1x1000x1x1xf16>
  %fc_fill1030 = linalg.fill ins(%cst : f16) outs(%fc_init1029 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  %fc1031 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1028, %w_fc : tensor<1x512x7x7xf16>, tensor<1000x512x7x7xf16>)
    outs(%fc_fill1030 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  return %fc1031 : tensor<1x1000x1x1xf16>
}
