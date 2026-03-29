func.func @resnet_bottleneck_3_8_36_3(
    %input: tensor<1x3x224x224xf16>,
    %w_conv1: tensor<64x3x7x7xf16>,
    %w_pool: tensor<64x64x3x3xf16>,
    %w_s0_b0_c1: tensor<64x64x1x1xf16>,
    %w_s0_b0_c2: tensor<64x64x3x3xf16>,
    %w_s0_b0_c3: tensor<256x64x1x1xf16>,
    %w_s0_b0_sc: tensor<256x64x1x1xf16>,
    %w_s0_b1_c1: tensor<64x256x1x1xf16>,
    %w_s0_b1_c2: tensor<64x64x3x3xf16>,
    %w_s0_b1_c3: tensor<256x64x1x1xf16>,
    %w_s0_b2_c1: tensor<64x256x1x1xf16>,
    %w_s0_b2_c2: tensor<64x64x3x3xf16>,
    %w_s0_b2_c3: tensor<256x64x1x1xf16>,
    %w_s1_b0_c1: tensor<128x256x1x1xf16>,
    %w_s1_b0_c2: tensor<128x128x3x3xf16>,
    %w_s1_b0_c3: tensor<512x128x1x1xf16>,
    %w_s1_b0_sc: tensor<512x256x1x1xf16>,
    %w_s1_b1_c1: tensor<128x512x1x1xf16>,
    %w_s1_b1_c2: tensor<128x128x3x3xf16>,
    %w_s1_b1_c3: tensor<512x128x1x1xf16>,
    %w_s1_b2_c1: tensor<128x512x1x1xf16>,
    %w_s1_b2_c2: tensor<128x128x3x3xf16>,
    %w_s1_b2_c3: tensor<512x128x1x1xf16>,
    %w_s1_b3_c1: tensor<128x512x1x1xf16>,
    %w_s1_b3_c2: tensor<128x128x3x3xf16>,
    %w_s1_b3_c3: tensor<512x128x1x1xf16>,
    %w_s1_b4_c1: tensor<128x512x1x1xf16>,
    %w_s1_b4_c2: tensor<128x128x3x3xf16>,
    %w_s1_b4_c3: tensor<512x128x1x1xf16>,
    %w_s1_b5_c1: tensor<128x512x1x1xf16>,
    %w_s1_b5_c2: tensor<128x128x3x3xf16>,
    %w_s1_b5_c3: tensor<512x128x1x1xf16>,
    %w_s1_b6_c1: tensor<128x512x1x1xf16>,
    %w_s1_b6_c2: tensor<128x128x3x3xf16>,
    %w_s1_b6_c3: tensor<512x128x1x1xf16>,
    %w_s1_b7_c1: tensor<128x512x1x1xf16>,
    %w_s1_b7_c2: tensor<128x128x3x3xf16>,
    %w_s1_b7_c3: tensor<512x128x1x1xf16>,
    %w_s2_b0_c1: tensor<256x512x1x1xf16>,
    %w_s2_b0_c2: tensor<256x256x3x3xf16>,
    %w_s2_b0_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b0_sc: tensor<1024x512x1x1xf16>,
    %w_s2_b1_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b1_c2: tensor<256x256x3x3xf16>,
    %w_s2_b1_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b2_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b2_c2: tensor<256x256x3x3xf16>,
    %w_s2_b2_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b3_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b3_c2: tensor<256x256x3x3xf16>,
    %w_s2_b3_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b4_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b4_c2: tensor<256x256x3x3xf16>,
    %w_s2_b4_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b5_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b5_c2: tensor<256x256x3x3xf16>,
    %w_s2_b5_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b6_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b6_c2: tensor<256x256x3x3xf16>,
    %w_s2_b6_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b7_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b7_c2: tensor<256x256x3x3xf16>,
    %w_s2_b7_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b8_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b8_c2: tensor<256x256x3x3xf16>,
    %w_s2_b8_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b9_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b9_c2: tensor<256x256x3x3xf16>,
    %w_s2_b9_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b10_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b10_c2: tensor<256x256x3x3xf16>,
    %w_s2_b10_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b11_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b11_c2: tensor<256x256x3x3xf16>,
    %w_s2_b11_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b12_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b12_c2: tensor<256x256x3x3xf16>,
    %w_s2_b12_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b13_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b13_c2: tensor<256x256x3x3xf16>,
    %w_s2_b13_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b14_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b14_c2: tensor<256x256x3x3xf16>,
    %w_s2_b14_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b15_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b15_c2: tensor<256x256x3x3xf16>,
    %w_s2_b15_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b16_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b16_c2: tensor<256x256x3x3xf16>,
    %w_s2_b16_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b17_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b17_c2: tensor<256x256x3x3xf16>,
    %w_s2_b17_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b18_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b18_c2: tensor<256x256x3x3xf16>,
    %w_s2_b18_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b19_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b19_c2: tensor<256x256x3x3xf16>,
    %w_s2_b19_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b20_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b20_c2: tensor<256x256x3x3xf16>,
    %w_s2_b20_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b21_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b21_c2: tensor<256x256x3x3xf16>,
    %w_s2_b21_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b22_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b22_c2: tensor<256x256x3x3xf16>,
    %w_s2_b22_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b23_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b23_c2: tensor<256x256x3x3xf16>,
    %w_s2_b23_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b24_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b24_c2: tensor<256x256x3x3xf16>,
    %w_s2_b24_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b25_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b25_c2: tensor<256x256x3x3xf16>,
    %w_s2_b25_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b26_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b26_c2: tensor<256x256x3x3xf16>,
    %w_s2_b26_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b27_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b27_c2: tensor<256x256x3x3xf16>,
    %w_s2_b27_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b28_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b28_c2: tensor<256x256x3x3xf16>,
    %w_s2_b28_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b29_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b29_c2: tensor<256x256x3x3xf16>,
    %w_s2_b29_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b30_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b30_c2: tensor<256x256x3x3xf16>,
    %w_s2_b30_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b31_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b31_c2: tensor<256x256x3x3xf16>,
    %w_s2_b31_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b32_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b32_c2: tensor<256x256x3x3xf16>,
    %w_s2_b32_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b33_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b33_c2: tensor<256x256x3x3xf16>,
    %w_s2_b33_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b34_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b34_c2: tensor<256x256x3x3xf16>,
    %w_s2_b34_c3: tensor<1024x256x1x1xf16>,
    %w_s2_b35_c1: tensor<256x1024x1x1xf16>,
    %w_s2_b35_c2: tensor<256x256x3x3xf16>,
    %w_s2_b35_c3: tensor<1024x256x1x1xf16>,
    %w_s3_b0_c1: tensor<512x1024x1x1xf16>,
    %w_s3_b0_c2: tensor<512x512x3x3xf16>,
    %w_s3_b0_c3: tensor<2048x512x1x1xf16>,
    %w_s3_b0_sc: tensor<2048x1024x1x1xf16>,
    %w_s3_b1_c1: tensor<512x2048x1x1xf16>,
    %w_s3_b1_c2: tensor<512x512x3x3xf16>,
    %w_s3_b1_c3: tensor<2048x512x1x1xf16>,
    %w_s3_b2_c1: tensor<512x2048x1x1xf16>,
    %w_s3_b2_c2: tensor<512x512x3x3xf16>,
    %w_s3_b2_c3: tensor<2048x512x1x1xf16>,
    %w_fc: tensor<1000x2048x7x7xf16>) -> tensor<1x1000x1x1xf16> {
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
  // Bottleneck 64->256 mid=64 56x56 stride=1
  %init12 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill13 = linalg.fill ins(%cst : f16) outs(%init12 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv14 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu11, %w_s0_b0_c1 : tensor<1x64x56x56xf16>, tensor<64x64x1x1xf16>)
    outs(%fill13 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty15 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu16 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv14 : tensor<1x64x56x56xf16>)
    outs(%empty15 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad17 = tensor.pad %relu16 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init18 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill19 = linalg.fill ins(%cst : f16) outs(%init18 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv20 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad17, %w_s0_b0_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill19 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty21 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu22 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv20 : tensor<1x64x56x56xf16>)
    outs(%empty21 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %init23 = tensor.empty() : tensor<1x256x56x56xf16>
  %fill24 = linalg.fill ins(%cst : f16) outs(%init23 : tensor<1x256x56x56xf16>) -> tensor<1x256x56x56xf16>
  %conv25 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu22, %w_s0_b0_c3 : tensor<1x64x56x56xf16>, tensor<256x64x1x1xf16>)
    outs(%fill24 : tensor<1x256x56x56xf16>) -> tensor<1x256x56x56xf16>
  // 1x1 shortcut conv
  %init26 = tensor.empty() : tensor<1x256x56x56xf16>
  %fill27 = linalg.fill ins(%cst : f16) outs(%init26 : tensor<1x256x56x56xf16>) -> tensor<1x256x56x56xf16>
  %conv28 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu11, %w_s0_b0_sc : tensor<1x64x56x56xf16>, tensor<256x64x1x1xf16>)
    outs(%fill27 : tensor<1x256x56x56xf16>) -> tensor<1x256x56x56xf16>
  %empty29 = tensor.empty() : tensor<1x256x56x56xf16>
  %add30 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv25, %conv28 : tensor<1x256x56x56xf16>, tensor<1x256x56x56xf16>)
    outs(%empty29 : tensor<1x256x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x56x56xf16>
  %empty31 = tensor.empty() : tensor<1x256x56x56xf16>
  %relu32 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add30 : tensor<1x256x56x56xf16>)
    outs(%empty31 : tensor<1x256x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x56x56xf16>

  // Bottleneck 256->256 mid=64 56x56 stride=1
  %init33 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill34 = linalg.fill ins(%cst : f16) outs(%init33 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv35 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu32, %w_s0_b1_c1 : tensor<1x256x56x56xf16>, tensor<64x256x1x1xf16>)
    outs(%fill34 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty36 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu37 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv35 : tensor<1x64x56x56xf16>)
    outs(%empty36 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad38 = tensor.pad %relu37 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init39 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill40 = linalg.fill ins(%cst : f16) outs(%init39 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv41 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad38, %w_s0_b1_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill40 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty42 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu43 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv41 : tensor<1x64x56x56xf16>)
    outs(%empty42 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %init44 = tensor.empty() : tensor<1x256x56x56xf16>
  %fill45 = linalg.fill ins(%cst : f16) outs(%init44 : tensor<1x256x56x56xf16>) -> tensor<1x256x56x56xf16>
  %conv46 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu43, %w_s0_b1_c3 : tensor<1x64x56x56xf16>, tensor<256x64x1x1xf16>)
    outs(%fill45 : tensor<1x256x56x56xf16>) -> tensor<1x256x56x56xf16>
  %empty47 = tensor.empty() : tensor<1x256x56x56xf16>
  %add48 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv46, %relu32 : tensor<1x256x56x56xf16>, tensor<1x256x56x56xf16>)
    outs(%empty47 : tensor<1x256x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x56x56xf16>
  %empty49 = tensor.empty() : tensor<1x256x56x56xf16>
  %relu50 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add48 : tensor<1x256x56x56xf16>)
    outs(%empty49 : tensor<1x256x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x56x56xf16>

  // Bottleneck 256->256 mid=64 56x56 stride=1
  %init51 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill52 = linalg.fill ins(%cst : f16) outs(%init51 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv53 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu50, %w_s0_b2_c1 : tensor<1x256x56x56xf16>, tensor<64x256x1x1xf16>)
    outs(%fill52 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty54 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu55 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv53 : tensor<1x64x56x56xf16>)
    outs(%empty54 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %pad56 = tensor.pad %relu55 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init57 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill58 = linalg.fill ins(%cst : f16) outs(%init57 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv59 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad56, %w_s0_b2_c2 : tensor<1x64x58x58xf16>, tensor<64x64x3x3xf16>)
    outs(%fill58 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty60 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu61 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv59 : tensor<1x64x56x56xf16>)
    outs(%empty60 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %init62 = tensor.empty() : tensor<1x256x56x56xf16>
  %fill63 = linalg.fill ins(%cst : f16) outs(%init62 : tensor<1x256x56x56xf16>) -> tensor<1x256x56x56xf16>
  %conv64 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu61, %w_s0_b2_c3 : tensor<1x64x56x56xf16>, tensor<256x64x1x1xf16>)
    outs(%fill63 : tensor<1x256x56x56xf16>) -> tensor<1x256x56x56xf16>
  %empty65 = tensor.empty() : tensor<1x256x56x56xf16>
  %add66 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv64, %relu50 : tensor<1x256x56x56xf16>, tensor<1x256x56x56xf16>)
    outs(%empty65 : tensor<1x256x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x56x56xf16>
  %empty67 = tensor.empty() : tensor<1x256x56x56xf16>
  %relu68 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add66 : tensor<1x256x56x56xf16>)
    outs(%empty67 : tensor<1x256x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x56x56xf16>

  // === Stage 1 ===
  // Bottleneck 256->512 mid=128 56x56 stride=2
  %init69 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill70 = linalg.fill ins(%cst : f16) outs(%init69 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv71 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu68, %w_s1_b0_c1 : tensor<1x256x56x56xf16>, tensor<128x256x1x1xf16>)
    outs(%fill70 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %empty72 = tensor.empty() : tensor<1x128x56x56xf16>
  %relu73 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv71 : tensor<1x128x56x56xf16>)
    outs(%empty72 : tensor<1x128x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x56x56xf16>
  %pad74 = tensor.pad %relu73 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x56x56xf16> to tensor<1x128x58x58xf16>
  %init75 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill76 = linalg.fill ins(%cst : f16) outs(%init75 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv77 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad74, %w_s1_b0_c2 : tensor<1x128x58x58xf16>, tensor<128x128x3x3xf16>)
    outs(%fill76 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty78 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu79 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv77 : tensor<1x128x28x28xf16>)
    outs(%empty78 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %init80 = tensor.empty() : tensor<1x512x28x28xf16>
  %fill81 = linalg.fill ins(%cst : f16) outs(%init80 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %conv82 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu79, %w_s1_b0_c3 : tensor<1x128x28x28xf16>, tensor<512x128x1x1xf16>)
    outs(%fill81 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  // 1x1 shortcut conv
  %init83 = tensor.empty() : tensor<1x512x28x28xf16>
  %fill84 = linalg.fill ins(%cst : f16) outs(%init83 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %conv85 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu68, %w_s1_b0_sc : tensor<1x256x56x56xf16>, tensor<512x256x1x1xf16>)
    outs(%fill84 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %empty86 = tensor.empty() : tensor<1x512x28x28xf16>
  %add87 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv82, %conv85 : tensor<1x512x28x28xf16>, tensor<1x512x28x28xf16>)
    outs(%empty86 : tensor<1x512x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x28x28xf16>
  %empty88 = tensor.empty() : tensor<1x512x28x28xf16>
  %relu89 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add87 : tensor<1x512x28x28xf16>)
    outs(%empty88 : tensor<1x512x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x28x28xf16>

  // Bottleneck 512->512 mid=128 28x28 stride=1
  %init90 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill91 = linalg.fill ins(%cst : f16) outs(%init90 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv92 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu89, %w_s1_b1_c1 : tensor<1x512x28x28xf16>, tensor<128x512x1x1xf16>)
    outs(%fill91 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty93 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu94 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv92 : tensor<1x128x28x28xf16>)
    outs(%empty93 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad95 = tensor.pad %relu94 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init96 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill97 = linalg.fill ins(%cst : f16) outs(%init96 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv98 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad95, %w_s1_b1_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill97 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty99 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu100 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv98 : tensor<1x128x28x28xf16>)
    outs(%empty99 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %init101 = tensor.empty() : tensor<1x512x28x28xf16>
  %fill102 = linalg.fill ins(%cst : f16) outs(%init101 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %conv103 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu100, %w_s1_b1_c3 : tensor<1x128x28x28xf16>, tensor<512x128x1x1xf16>)
    outs(%fill102 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %empty104 = tensor.empty() : tensor<1x512x28x28xf16>
  %add105 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv103, %relu89 : tensor<1x512x28x28xf16>, tensor<1x512x28x28xf16>)
    outs(%empty104 : tensor<1x512x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x28x28xf16>
  %empty106 = tensor.empty() : tensor<1x512x28x28xf16>
  %relu107 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add105 : tensor<1x512x28x28xf16>)
    outs(%empty106 : tensor<1x512x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x28x28xf16>

  // Bottleneck 512->512 mid=128 28x28 stride=1
  %init108 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill109 = linalg.fill ins(%cst : f16) outs(%init108 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv110 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu107, %w_s1_b2_c1 : tensor<1x512x28x28xf16>, tensor<128x512x1x1xf16>)
    outs(%fill109 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty111 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu112 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv110 : tensor<1x128x28x28xf16>)
    outs(%empty111 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad113 = tensor.pad %relu112 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init114 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill115 = linalg.fill ins(%cst : f16) outs(%init114 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv116 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad113, %w_s1_b2_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill115 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty117 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu118 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv116 : tensor<1x128x28x28xf16>)
    outs(%empty117 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %init119 = tensor.empty() : tensor<1x512x28x28xf16>
  %fill120 = linalg.fill ins(%cst : f16) outs(%init119 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %conv121 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu118, %w_s1_b2_c3 : tensor<1x128x28x28xf16>, tensor<512x128x1x1xf16>)
    outs(%fill120 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %empty122 = tensor.empty() : tensor<1x512x28x28xf16>
  %add123 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv121, %relu107 : tensor<1x512x28x28xf16>, tensor<1x512x28x28xf16>)
    outs(%empty122 : tensor<1x512x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x28x28xf16>
  %empty124 = tensor.empty() : tensor<1x512x28x28xf16>
  %relu125 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add123 : tensor<1x512x28x28xf16>)
    outs(%empty124 : tensor<1x512x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x28x28xf16>

  // Bottleneck 512->512 mid=128 28x28 stride=1
  %init126 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill127 = linalg.fill ins(%cst : f16) outs(%init126 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv128 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu125, %w_s1_b3_c1 : tensor<1x512x28x28xf16>, tensor<128x512x1x1xf16>)
    outs(%fill127 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty129 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu130 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv128 : tensor<1x128x28x28xf16>)
    outs(%empty129 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad131 = tensor.pad %relu130 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init132 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill133 = linalg.fill ins(%cst : f16) outs(%init132 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv134 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad131, %w_s1_b3_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill133 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty135 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu136 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv134 : tensor<1x128x28x28xf16>)
    outs(%empty135 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %init137 = tensor.empty() : tensor<1x512x28x28xf16>
  %fill138 = linalg.fill ins(%cst : f16) outs(%init137 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %conv139 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu136, %w_s1_b3_c3 : tensor<1x128x28x28xf16>, tensor<512x128x1x1xf16>)
    outs(%fill138 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %empty140 = tensor.empty() : tensor<1x512x28x28xf16>
  %add141 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv139, %relu125 : tensor<1x512x28x28xf16>, tensor<1x512x28x28xf16>)
    outs(%empty140 : tensor<1x512x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x28x28xf16>
  %empty142 = tensor.empty() : tensor<1x512x28x28xf16>
  %relu143 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add141 : tensor<1x512x28x28xf16>)
    outs(%empty142 : tensor<1x512x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x28x28xf16>

  // Bottleneck 512->512 mid=128 28x28 stride=1
  %init144 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill145 = linalg.fill ins(%cst : f16) outs(%init144 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv146 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu143, %w_s1_b4_c1 : tensor<1x512x28x28xf16>, tensor<128x512x1x1xf16>)
    outs(%fill145 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty147 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu148 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv146 : tensor<1x128x28x28xf16>)
    outs(%empty147 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad149 = tensor.pad %relu148 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init150 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill151 = linalg.fill ins(%cst : f16) outs(%init150 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv152 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad149, %w_s1_b4_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill151 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty153 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu154 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv152 : tensor<1x128x28x28xf16>)
    outs(%empty153 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %init155 = tensor.empty() : tensor<1x512x28x28xf16>
  %fill156 = linalg.fill ins(%cst : f16) outs(%init155 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %conv157 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu154, %w_s1_b4_c3 : tensor<1x128x28x28xf16>, tensor<512x128x1x1xf16>)
    outs(%fill156 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %empty158 = tensor.empty() : tensor<1x512x28x28xf16>
  %add159 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv157, %relu143 : tensor<1x512x28x28xf16>, tensor<1x512x28x28xf16>)
    outs(%empty158 : tensor<1x512x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x28x28xf16>
  %empty160 = tensor.empty() : tensor<1x512x28x28xf16>
  %relu161 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add159 : tensor<1x512x28x28xf16>)
    outs(%empty160 : tensor<1x512x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x28x28xf16>

  // Bottleneck 512->512 mid=128 28x28 stride=1
  %init162 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill163 = linalg.fill ins(%cst : f16) outs(%init162 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv164 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu161, %w_s1_b5_c1 : tensor<1x512x28x28xf16>, tensor<128x512x1x1xf16>)
    outs(%fill163 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty165 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu166 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv164 : tensor<1x128x28x28xf16>)
    outs(%empty165 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad167 = tensor.pad %relu166 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init168 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill169 = linalg.fill ins(%cst : f16) outs(%init168 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv170 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad167, %w_s1_b5_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill169 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty171 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu172 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv170 : tensor<1x128x28x28xf16>)
    outs(%empty171 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %init173 = tensor.empty() : tensor<1x512x28x28xf16>
  %fill174 = linalg.fill ins(%cst : f16) outs(%init173 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %conv175 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu172, %w_s1_b5_c3 : tensor<1x128x28x28xf16>, tensor<512x128x1x1xf16>)
    outs(%fill174 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %empty176 = tensor.empty() : tensor<1x512x28x28xf16>
  %add177 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv175, %relu161 : tensor<1x512x28x28xf16>, tensor<1x512x28x28xf16>)
    outs(%empty176 : tensor<1x512x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x28x28xf16>
  %empty178 = tensor.empty() : tensor<1x512x28x28xf16>
  %relu179 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add177 : tensor<1x512x28x28xf16>)
    outs(%empty178 : tensor<1x512x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x28x28xf16>

  // Bottleneck 512->512 mid=128 28x28 stride=1
  %init180 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill181 = linalg.fill ins(%cst : f16) outs(%init180 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv182 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu179, %w_s1_b6_c1 : tensor<1x512x28x28xf16>, tensor<128x512x1x1xf16>)
    outs(%fill181 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty183 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu184 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv182 : tensor<1x128x28x28xf16>)
    outs(%empty183 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad185 = tensor.pad %relu184 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init186 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill187 = linalg.fill ins(%cst : f16) outs(%init186 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv188 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad185, %w_s1_b6_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill187 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty189 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu190 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv188 : tensor<1x128x28x28xf16>)
    outs(%empty189 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %init191 = tensor.empty() : tensor<1x512x28x28xf16>
  %fill192 = linalg.fill ins(%cst : f16) outs(%init191 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %conv193 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu190, %w_s1_b6_c3 : tensor<1x128x28x28xf16>, tensor<512x128x1x1xf16>)
    outs(%fill192 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %empty194 = tensor.empty() : tensor<1x512x28x28xf16>
  %add195 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv193, %relu179 : tensor<1x512x28x28xf16>, tensor<1x512x28x28xf16>)
    outs(%empty194 : tensor<1x512x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x28x28xf16>
  %empty196 = tensor.empty() : tensor<1x512x28x28xf16>
  %relu197 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add195 : tensor<1x512x28x28xf16>)
    outs(%empty196 : tensor<1x512x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x28x28xf16>

  // Bottleneck 512->512 mid=128 28x28 stride=1
  %init198 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill199 = linalg.fill ins(%cst : f16) outs(%init198 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv200 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu197, %w_s1_b7_c1 : tensor<1x512x28x28xf16>, tensor<128x512x1x1xf16>)
    outs(%fill199 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty201 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu202 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv200 : tensor<1x128x28x28xf16>)
    outs(%empty201 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad203 = tensor.pad %relu202 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init204 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill205 = linalg.fill ins(%cst : f16) outs(%init204 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv206 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad203, %w_s1_b7_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill205 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty207 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu208 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv206 : tensor<1x128x28x28xf16>)
    outs(%empty207 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %init209 = tensor.empty() : tensor<1x512x28x28xf16>
  %fill210 = linalg.fill ins(%cst : f16) outs(%init209 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %conv211 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu208, %w_s1_b7_c3 : tensor<1x128x28x28xf16>, tensor<512x128x1x1xf16>)
    outs(%fill210 : tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
  %empty212 = tensor.empty() : tensor<1x512x28x28xf16>
  %add213 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv211, %relu197 : tensor<1x512x28x28xf16>, tensor<1x512x28x28xf16>)
    outs(%empty212 : tensor<1x512x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x28x28xf16>
  %empty214 = tensor.empty() : tensor<1x512x28x28xf16>
  %relu215 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add213 : tensor<1x512x28x28xf16>)
    outs(%empty214 : tensor<1x512x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x28x28xf16>

  // === Stage 2 ===
  // Bottleneck 512->1024 mid=256 28x28 stride=2
  %init216 = tensor.empty() : tensor<1x256x28x28xf16>
  %fill217 = linalg.fill ins(%cst : f16) outs(%init216 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %conv218 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu215, %w_s2_b0_c1 : tensor<1x512x28x28xf16>, tensor<256x512x1x1xf16>)
    outs(%fill217 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %empty219 = tensor.empty() : tensor<1x256x28x28xf16>
  %relu220 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv218 : tensor<1x256x28x28xf16>)
    outs(%empty219 : tensor<1x256x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x28x28xf16>
  %pad221 = tensor.pad %relu220 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x28x28xf16> to tensor<1x256x30x30xf16>
  %init222 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill223 = linalg.fill ins(%cst : f16) outs(%init222 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv224 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad221, %w_s2_b0_c2 : tensor<1x256x30x30xf16>, tensor<256x256x3x3xf16>)
    outs(%fill223 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty225 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu226 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv224 : tensor<1x256x14x14xf16>)
    outs(%empty225 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init227 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill228 = linalg.fill ins(%cst : f16) outs(%init227 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv229 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu226, %w_s2_b0_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill228 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  // 1x1 shortcut conv
  %init230 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill231 = linalg.fill ins(%cst : f16) outs(%init230 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv232 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu215, %w_s2_b0_sc : tensor<1x512x28x28xf16>, tensor<1024x512x1x1xf16>)
    outs(%fill231 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty233 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add234 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv229, %conv232 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty233 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty235 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu236 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add234 : tensor<1x1024x14x14xf16>)
    outs(%empty235 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init237 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill238 = linalg.fill ins(%cst : f16) outs(%init237 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv239 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu236, %w_s2_b1_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill238 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty240 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu241 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv239 : tensor<1x256x14x14xf16>)
    outs(%empty240 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad242 = tensor.pad %relu241 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init243 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill244 = linalg.fill ins(%cst : f16) outs(%init243 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv245 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad242, %w_s2_b1_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill244 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty246 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu247 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv245 : tensor<1x256x14x14xf16>)
    outs(%empty246 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init248 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill249 = linalg.fill ins(%cst : f16) outs(%init248 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv250 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu247, %w_s2_b1_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill249 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty251 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add252 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv250, %relu236 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty251 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty253 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu254 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add252 : tensor<1x1024x14x14xf16>)
    outs(%empty253 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init255 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill256 = linalg.fill ins(%cst : f16) outs(%init255 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv257 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu254, %w_s2_b2_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill256 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty258 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu259 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv257 : tensor<1x256x14x14xf16>)
    outs(%empty258 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad260 = tensor.pad %relu259 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init261 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill262 = linalg.fill ins(%cst : f16) outs(%init261 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv263 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad260, %w_s2_b2_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill262 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty264 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu265 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv263 : tensor<1x256x14x14xf16>)
    outs(%empty264 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init266 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill267 = linalg.fill ins(%cst : f16) outs(%init266 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv268 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu265, %w_s2_b2_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill267 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty269 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add270 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv268, %relu254 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty269 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty271 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu272 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add270 : tensor<1x1024x14x14xf16>)
    outs(%empty271 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init273 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill274 = linalg.fill ins(%cst : f16) outs(%init273 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv275 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu272, %w_s2_b3_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill274 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty276 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu277 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv275 : tensor<1x256x14x14xf16>)
    outs(%empty276 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad278 = tensor.pad %relu277 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init279 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill280 = linalg.fill ins(%cst : f16) outs(%init279 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv281 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad278, %w_s2_b3_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill280 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty282 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu283 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv281 : tensor<1x256x14x14xf16>)
    outs(%empty282 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init284 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill285 = linalg.fill ins(%cst : f16) outs(%init284 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv286 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu283, %w_s2_b3_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill285 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty287 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add288 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv286, %relu272 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty287 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty289 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu290 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add288 : tensor<1x1024x14x14xf16>)
    outs(%empty289 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init291 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill292 = linalg.fill ins(%cst : f16) outs(%init291 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv293 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu290, %w_s2_b4_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill292 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty294 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu295 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv293 : tensor<1x256x14x14xf16>)
    outs(%empty294 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad296 = tensor.pad %relu295 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init297 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill298 = linalg.fill ins(%cst : f16) outs(%init297 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv299 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad296, %w_s2_b4_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill298 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty300 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu301 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv299 : tensor<1x256x14x14xf16>)
    outs(%empty300 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init302 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill303 = linalg.fill ins(%cst : f16) outs(%init302 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv304 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu301, %w_s2_b4_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill303 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty305 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add306 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv304, %relu290 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty305 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty307 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu308 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add306 : tensor<1x1024x14x14xf16>)
    outs(%empty307 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init309 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill310 = linalg.fill ins(%cst : f16) outs(%init309 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv311 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu308, %w_s2_b5_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill310 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty312 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu313 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv311 : tensor<1x256x14x14xf16>)
    outs(%empty312 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad314 = tensor.pad %relu313 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init315 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill316 = linalg.fill ins(%cst : f16) outs(%init315 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv317 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad314, %w_s2_b5_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill316 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty318 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu319 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv317 : tensor<1x256x14x14xf16>)
    outs(%empty318 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init320 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill321 = linalg.fill ins(%cst : f16) outs(%init320 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv322 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu319, %w_s2_b5_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill321 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty323 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add324 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv322, %relu308 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty323 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty325 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu326 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add324 : tensor<1x1024x14x14xf16>)
    outs(%empty325 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init327 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill328 = linalg.fill ins(%cst : f16) outs(%init327 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv329 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu326, %w_s2_b6_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill328 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty330 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu331 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv329 : tensor<1x256x14x14xf16>)
    outs(%empty330 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad332 = tensor.pad %relu331 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init333 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill334 = linalg.fill ins(%cst : f16) outs(%init333 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv335 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad332, %w_s2_b6_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill334 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty336 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu337 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv335 : tensor<1x256x14x14xf16>)
    outs(%empty336 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init338 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill339 = linalg.fill ins(%cst : f16) outs(%init338 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv340 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu337, %w_s2_b6_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill339 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty341 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add342 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv340, %relu326 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty341 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty343 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu344 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add342 : tensor<1x1024x14x14xf16>)
    outs(%empty343 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init345 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill346 = linalg.fill ins(%cst : f16) outs(%init345 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv347 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu344, %w_s2_b7_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill346 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty348 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu349 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv347 : tensor<1x256x14x14xf16>)
    outs(%empty348 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad350 = tensor.pad %relu349 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init351 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill352 = linalg.fill ins(%cst : f16) outs(%init351 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv353 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad350, %w_s2_b7_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill352 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty354 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu355 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv353 : tensor<1x256x14x14xf16>)
    outs(%empty354 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init356 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill357 = linalg.fill ins(%cst : f16) outs(%init356 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv358 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu355, %w_s2_b7_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill357 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty359 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add360 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv358, %relu344 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty359 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty361 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu362 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add360 : tensor<1x1024x14x14xf16>)
    outs(%empty361 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init363 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill364 = linalg.fill ins(%cst : f16) outs(%init363 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv365 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu362, %w_s2_b8_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill364 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty366 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu367 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv365 : tensor<1x256x14x14xf16>)
    outs(%empty366 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad368 = tensor.pad %relu367 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init369 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill370 = linalg.fill ins(%cst : f16) outs(%init369 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv371 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad368, %w_s2_b8_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill370 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty372 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu373 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv371 : tensor<1x256x14x14xf16>)
    outs(%empty372 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init374 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill375 = linalg.fill ins(%cst : f16) outs(%init374 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv376 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu373, %w_s2_b8_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill375 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty377 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add378 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv376, %relu362 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty377 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty379 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu380 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add378 : tensor<1x1024x14x14xf16>)
    outs(%empty379 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init381 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill382 = linalg.fill ins(%cst : f16) outs(%init381 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv383 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu380, %w_s2_b9_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill382 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty384 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu385 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv383 : tensor<1x256x14x14xf16>)
    outs(%empty384 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad386 = tensor.pad %relu385 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init387 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill388 = linalg.fill ins(%cst : f16) outs(%init387 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv389 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad386, %w_s2_b9_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill388 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty390 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu391 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv389 : tensor<1x256x14x14xf16>)
    outs(%empty390 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init392 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill393 = linalg.fill ins(%cst : f16) outs(%init392 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv394 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu391, %w_s2_b9_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill393 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty395 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add396 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv394, %relu380 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty395 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty397 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu398 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add396 : tensor<1x1024x14x14xf16>)
    outs(%empty397 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init399 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill400 = linalg.fill ins(%cst : f16) outs(%init399 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv401 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu398, %w_s2_b10_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill400 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty402 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu403 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv401 : tensor<1x256x14x14xf16>)
    outs(%empty402 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad404 = tensor.pad %relu403 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init405 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill406 = linalg.fill ins(%cst : f16) outs(%init405 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv407 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad404, %w_s2_b10_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill406 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty408 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu409 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv407 : tensor<1x256x14x14xf16>)
    outs(%empty408 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init410 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill411 = linalg.fill ins(%cst : f16) outs(%init410 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv412 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu409, %w_s2_b10_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill411 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty413 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add414 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv412, %relu398 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty413 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty415 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu416 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add414 : tensor<1x1024x14x14xf16>)
    outs(%empty415 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init417 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill418 = linalg.fill ins(%cst : f16) outs(%init417 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv419 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu416, %w_s2_b11_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill418 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty420 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu421 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv419 : tensor<1x256x14x14xf16>)
    outs(%empty420 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad422 = tensor.pad %relu421 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init423 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill424 = linalg.fill ins(%cst : f16) outs(%init423 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv425 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad422, %w_s2_b11_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill424 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty426 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu427 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv425 : tensor<1x256x14x14xf16>)
    outs(%empty426 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init428 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill429 = linalg.fill ins(%cst : f16) outs(%init428 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv430 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu427, %w_s2_b11_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill429 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty431 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add432 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv430, %relu416 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty431 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty433 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu434 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add432 : tensor<1x1024x14x14xf16>)
    outs(%empty433 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init435 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill436 = linalg.fill ins(%cst : f16) outs(%init435 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv437 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu434, %w_s2_b12_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill436 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty438 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu439 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv437 : tensor<1x256x14x14xf16>)
    outs(%empty438 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad440 = tensor.pad %relu439 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init441 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill442 = linalg.fill ins(%cst : f16) outs(%init441 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv443 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad440, %w_s2_b12_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill442 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty444 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu445 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv443 : tensor<1x256x14x14xf16>)
    outs(%empty444 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init446 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill447 = linalg.fill ins(%cst : f16) outs(%init446 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv448 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu445, %w_s2_b12_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill447 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty449 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add450 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv448, %relu434 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty449 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty451 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu452 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add450 : tensor<1x1024x14x14xf16>)
    outs(%empty451 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init453 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill454 = linalg.fill ins(%cst : f16) outs(%init453 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv455 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu452, %w_s2_b13_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill454 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty456 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu457 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv455 : tensor<1x256x14x14xf16>)
    outs(%empty456 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad458 = tensor.pad %relu457 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init459 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill460 = linalg.fill ins(%cst : f16) outs(%init459 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv461 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad458, %w_s2_b13_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill460 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty462 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu463 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv461 : tensor<1x256x14x14xf16>)
    outs(%empty462 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init464 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill465 = linalg.fill ins(%cst : f16) outs(%init464 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv466 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu463, %w_s2_b13_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill465 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty467 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add468 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv466, %relu452 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty467 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty469 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu470 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add468 : tensor<1x1024x14x14xf16>)
    outs(%empty469 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init471 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill472 = linalg.fill ins(%cst : f16) outs(%init471 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv473 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu470, %w_s2_b14_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill472 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty474 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu475 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv473 : tensor<1x256x14x14xf16>)
    outs(%empty474 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad476 = tensor.pad %relu475 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init477 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill478 = linalg.fill ins(%cst : f16) outs(%init477 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv479 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad476, %w_s2_b14_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill478 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty480 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu481 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv479 : tensor<1x256x14x14xf16>)
    outs(%empty480 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init482 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill483 = linalg.fill ins(%cst : f16) outs(%init482 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv484 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu481, %w_s2_b14_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill483 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty485 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add486 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv484, %relu470 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty485 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty487 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu488 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add486 : tensor<1x1024x14x14xf16>)
    outs(%empty487 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init489 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill490 = linalg.fill ins(%cst : f16) outs(%init489 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv491 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu488, %w_s2_b15_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill490 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty492 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu493 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv491 : tensor<1x256x14x14xf16>)
    outs(%empty492 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad494 = tensor.pad %relu493 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init495 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill496 = linalg.fill ins(%cst : f16) outs(%init495 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv497 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad494, %w_s2_b15_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill496 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty498 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu499 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv497 : tensor<1x256x14x14xf16>)
    outs(%empty498 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init500 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill501 = linalg.fill ins(%cst : f16) outs(%init500 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv502 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu499, %w_s2_b15_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill501 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty503 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add504 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv502, %relu488 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty503 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty505 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu506 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add504 : tensor<1x1024x14x14xf16>)
    outs(%empty505 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init507 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill508 = linalg.fill ins(%cst : f16) outs(%init507 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv509 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu506, %w_s2_b16_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill508 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty510 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu511 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv509 : tensor<1x256x14x14xf16>)
    outs(%empty510 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad512 = tensor.pad %relu511 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init513 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill514 = linalg.fill ins(%cst : f16) outs(%init513 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv515 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad512, %w_s2_b16_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill514 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty516 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu517 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv515 : tensor<1x256x14x14xf16>)
    outs(%empty516 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init518 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill519 = linalg.fill ins(%cst : f16) outs(%init518 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv520 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu517, %w_s2_b16_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill519 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty521 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add522 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv520, %relu506 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty521 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty523 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu524 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add522 : tensor<1x1024x14x14xf16>)
    outs(%empty523 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init525 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill526 = linalg.fill ins(%cst : f16) outs(%init525 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv527 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu524, %w_s2_b17_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill526 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty528 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu529 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv527 : tensor<1x256x14x14xf16>)
    outs(%empty528 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad530 = tensor.pad %relu529 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init531 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill532 = linalg.fill ins(%cst : f16) outs(%init531 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv533 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad530, %w_s2_b17_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill532 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty534 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu535 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv533 : tensor<1x256x14x14xf16>)
    outs(%empty534 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init536 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill537 = linalg.fill ins(%cst : f16) outs(%init536 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv538 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu535, %w_s2_b17_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill537 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty539 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add540 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv538, %relu524 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty539 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty541 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu542 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add540 : tensor<1x1024x14x14xf16>)
    outs(%empty541 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init543 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill544 = linalg.fill ins(%cst : f16) outs(%init543 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv545 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu542, %w_s2_b18_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill544 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty546 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu547 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv545 : tensor<1x256x14x14xf16>)
    outs(%empty546 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad548 = tensor.pad %relu547 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init549 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill550 = linalg.fill ins(%cst : f16) outs(%init549 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv551 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad548, %w_s2_b18_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill550 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty552 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu553 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv551 : tensor<1x256x14x14xf16>)
    outs(%empty552 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init554 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill555 = linalg.fill ins(%cst : f16) outs(%init554 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv556 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu553, %w_s2_b18_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill555 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty557 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add558 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv556, %relu542 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty557 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty559 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu560 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add558 : tensor<1x1024x14x14xf16>)
    outs(%empty559 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init561 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill562 = linalg.fill ins(%cst : f16) outs(%init561 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv563 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu560, %w_s2_b19_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill562 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty564 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu565 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv563 : tensor<1x256x14x14xf16>)
    outs(%empty564 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad566 = tensor.pad %relu565 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init567 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill568 = linalg.fill ins(%cst : f16) outs(%init567 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv569 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad566, %w_s2_b19_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill568 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty570 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu571 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv569 : tensor<1x256x14x14xf16>)
    outs(%empty570 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init572 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill573 = linalg.fill ins(%cst : f16) outs(%init572 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv574 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu571, %w_s2_b19_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill573 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty575 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add576 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv574, %relu560 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty575 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty577 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu578 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add576 : tensor<1x1024x14x14xf16>)
    outs(%empty577 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init579 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill580 = linalg.fill ins(%cst : f16) outs(%init579 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv581 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu578, %w_s2_b20_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad584, %w_s2_b20_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill586 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty588 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu589 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv587 : tensor<1x256x14x14xf16>)
    outs(%empty588 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init590 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill591 = linalg.fill ins(%cst : f16) outs(%init590 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv592 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu589, %w_s2_b20_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill591 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty593 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add594 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv592, %relu578 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty593 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty595 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu596 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add594 : tensor<1x1024x14x14xf16>)
    outs(%empty595 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init597 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill598 = linalg.fill ins(%cst : f16) outs(%init597 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv599 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu596, %w_s2_b21_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill598 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty600 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu601 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv599 : tensor<1x256x14x14xf16>)
    outs(%empty600 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad602 = tensor.pad %relu601 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init603 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill604 = linalg.fill ins(%cst : f16) outs(%init603 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv605 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad602, %w_s2_b21_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill604 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty606 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu607 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv605 : tensor<1x256x14x14xf16>)
    outs(%empty606 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init608 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill609 = linalg.fill ins(%cst : f16) outs(%init608 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv610 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu607, %w_s2_b21_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill609 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty611 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add612 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv610, %relu596 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty611 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty613 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu614 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add612 : tensor<1x1024x14x14xf16>)
    outs(%empty613 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init615 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill616 = linalg.fill ins(%cst : f16) outs(%init615 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv617 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu614, %w_s2_b22_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill616 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty618 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu619 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv617 : tensor<1x256x14x14xf16>)
    outs(%empty618 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad620 = tensor.pad %relu619 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init621 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill622 = linalg.fill ins(%cst : f16) outs(%init621 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv623 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad620, %w_s2_b22_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  %init626 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill627 = linalg.fill ins(%cst : f16) outs(%init626 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv628 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu625, %w_s2_b22_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill627 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty629 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add630 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv628, %relu614 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty629 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty631 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu632 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add630 : tensor<1x1024x14x14xf16>)
    outs(%empty631 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init633 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill634 = linalg.fill ins(%cst : f16) outs(%init633 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv635 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu632, %w_s2_b23_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill634 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty636 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu637 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv635 : tensor<1x256x14x14xf16>)
    outs(%empty636 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad638 = tensor.pad %relu637 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init639 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill640 = linalg.fill ins(%cst : f16) outs(%init639 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv641 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad638, %w_s2_b23_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill640 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty642 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu643 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv641 : tensor<1x256x14x14xf16>)
    outs(%empty642 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init644 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill645 = linalg.fill ins(%cst : f16) outs(%init644 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv646 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu643, %w_s2_b23_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill645 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty647 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add648 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv646, %relu632 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty647 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty649 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu650 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add648 : tensor<1x1024x14x14xf16>)
    outs(%empty649 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init651 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill652 = linalg.fill ins(%cst : f16) outs(%init651 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv653 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu650, %w_s2_b24_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill652 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty654 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu655 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv653 : tensor<1x256x14x14xf16>)
    outs(%empty654 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad656 = tensor.pad %relu655 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init657 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill658 = linalg.fill ins(%cst : f16) outs(%init657 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv659 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad656, %w_s2_b24_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill658 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty660 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu661 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv659 : tensor<1x256x14x14xf16>)
    outs(%empty660 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init662 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill663 = linalg.fill ins(%cst : f16) outs(%init662 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv664 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu661, %w_s2_b24_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill663 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty665 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add666 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv664, %relu650 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty665 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty667 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu668 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add666 : tensor<1x1024x14x14xf16>)
    outs(%empty667 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init669 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill670 = linalg.fill ins(%cst : f16) outs(%init669 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv671 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu668, %w_s2_b25_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill670 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty672 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu673 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv671 : tensor<1x256x14x14xf16>)
    outs(%empty672 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad674 = tensor.pad %relu673 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init675 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill676 = linalg.fill ins(%cst : f16) outs(%init675 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv677 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad674, %w_s2_b25_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill676 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty678 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu679 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv677 : tensor<1x256x14x14xf16>)
    outs(%empty678 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init680 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill681 = linalg.fill ins(%cst : f16) outs(%init680 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv682 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu679, %w_s2_b25_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill681 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty683 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add684 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv682, %relu668 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty683 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty685 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu686 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add684 : tensor<1x1024x14x14xf16>)
    outs(%empty685 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init687 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill688 = linalg.fill ins(%cst : f16) outs(%init687 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv689 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu686, %w_s2_b26_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill688 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty690 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu691 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv689 : tensor<1x256x14x14xf16>)
    outs(%empty690 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad692 = tensor.pad %relu691 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init693 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill694 = linalg.fill ins(%cst : f16) outs(%init693 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv695 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad692, %w_s2_b26_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill694 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty696 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu697 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv695 : tensor<1x256x14x14xf16>)
    outs(%empty696 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init698 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill699 = linalg.fill ins(%cst : f16) outs(%init698 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv700 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu697, %w_s2_b26_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill699 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty701 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add702 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv700, %relu686 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty701 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty703 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu704 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add702 : tensor<1x1024x14x14xf16>)
    outs(%empty703 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init705 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill706 = linalg.fill ins(%cst : f16) outs(%init705 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv707 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu704, %w_s2_b27_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad710, %w_s2_b27_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill712 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty714 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu715 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv713 : tensor<1x256x14x14xf16>)
    outs(%empty714 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init716 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill717 = linalg.fill ins(%cst : f16) outs(%init716 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv718 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu715, %w_s2_b27_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill717 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty719 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add720 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv718, %relu704 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty719 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty721 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu722 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add720 : tensor<1x1024x14x14xf16>)
    outs(%empty721 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init723 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill724 = linalg.fill ins(%cst : f16) outs(%init723 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv725 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu722, %w_s2_b28_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill724 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty726 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu727 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv725 : tensor<1x256x14x14xf16>)
    outs(%empty726 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad728 = tensor.pad %relu727 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init729 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill730 = linalg.fill ins(%cst : f16) outs(%init729 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv731 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad728, %w_s2_b28_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill730 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty732 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu733 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv731 : tensor<1x256x14x14xf16>)
    outs(%empty732 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init734 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill735 = linalg.fill ins(%cst : f16) outs(%init734 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv736 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu733, %w_s2_b28_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill735 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty737 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add738 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv736, %relu722 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty737 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty739 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu740 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add738 : tensor<1x1024x14x14xf16>)
    outs(%empty739 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init741 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill742 = linalg.fill ins(%cst : f16) outs(%init741 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv743 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu740, %w_s2_b29_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill742 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty744 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu745 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv743 : tensor<1x256x14x14xf16>)
    outs(%empty744 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad746 = tensor.pad %relu745 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init747 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill748 = linalg.fill ins(%cst : f16) outs(%init747 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv749 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad746, %w_s2_b29_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  %init752 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill753 = linalg.fill ins(%cst : f16) outs(%init752 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv754 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu751, %w_s2_b29_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill753 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty755 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add756 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv754, %relu740 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty755 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty757 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu758 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add756 : tensor<1x1024x14x14xf16>)
    outs(%empty757 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init759 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill760 = linalg.fill ins(%cst : f16) outs(%init759 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv761 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu758, %w_s2_b30_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill760 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty762 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu763 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv761 : tensor<1x256x14x14xf16>)
    outs(%empty762 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad764 = tensor.pad %relu763 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init765 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill766 = linalg.fill ins(%cst : f16) outs(%init765 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv767 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad764, %w_s2_b30_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill766 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty768 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu769 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv767 : tensor<1x256x14x14xf16>)
    outs(%empty768 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init770 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill771 = linalg.fill ins(%cst : f16) outs(%init770 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv772 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu769, %w_s2_b30_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill771 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty773 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add774 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv772, %relu758 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty773 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty775 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu776 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add774 : tensor<1x1024x14x14xf16>)
    outs(%empty775 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init777 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill778 = linalg.fill ins(%cst : f16) outs(%init777 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv779 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu776, %w_s2_b31_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill778 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty780 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu781 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv779 : tensor<1x256x14x14xf16>)
    outs(%empty780 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad782 = tensor.pad %relu781 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init783 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill784 = linalg.fill ins(%cst : f16) outs(%init783 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv785 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad782, %w_s2_b31_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill784 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty786 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu787 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv785 : tensor<1x256x14x14xf16>)
    outs(%empty786 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init788 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill789 = linalg.fill ins(%cst : f16) outs(%init788 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv790 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu787, %w_s2_b31_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill789 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty791 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add792 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv790, %relu776 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty791 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty793 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu794 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add792 : tensor<1x1024x14x14xf16>)
    outs(%empty793 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init795 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill796 = linalg.fill ins(%cst : f16) outs(%init795 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv797 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu794, %w_s2_b32_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill796 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty798 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu799 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv797 : tensor<1x256x14x14xf16>)
    outs(%empty798 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad800 = tensor.pad %relu799 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init801 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill802 = linalg.fill ins(%cst : f16) outs(%init801 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv803 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad800, %w_s2_b32_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill802 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty804 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu805 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv803 : tensor<1x256x14x14xf16>)
    outs(%empty804 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init806 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill807 = linalg.fill ins(%cst : f16) outs(%init806 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv808 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu805, %w_s2_b32_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill807 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty809 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add810 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv808, %relu794 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty809 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty811 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu812 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add810 : tensor<1x1024x14x14xf16>)
    outs(%empty811 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init813 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill814 = linalg.fill ins(%cst : f16) outs(%init813 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv815 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu812, %w_s2_b33_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill814 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty816 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu817 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv815 : tensor<1x256x14x14xf16>)
    outs(%empty816 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad818 = tensor.pad %relu817 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init819 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill820 = linalg.fill ins(%cst : f16) outs(%init819 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv821 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad818, %w_s2_b33_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill820 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty822 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu823 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv821 : tensor<1x256x14x14xf16>)
    outs(%empty822 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init824 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill825 = linalg.fill ins(%cst : f16) outs(%init824 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv826 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu823, %w_s2_b33_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill825 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty827 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add828 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv826, %relu812 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty827 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty829 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu830 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add828 : tensor<1x1024x14x14xf16>)
    outs(%empty829 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init831 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill832 = linalg.fill ins(%cst : f16) outs(%init831 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv833 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu830, %w_s2_b34_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill832 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty834 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu835 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv833 : tensor<1x256x14x14xf16>)
    outs(%empty834 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad836 = tensor.pad %relu835 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init837 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill838 = linalg.fill ins(%cst : f16) outs(%init837 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv839 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad836, %w_s2_b34_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill838 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty840 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu841 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv839 : tensor<1x256x14x14xf16>)
    outs(%empty840 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init842 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill843 = linalg.fill ins(%cst : f16) outs(%init842 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv844 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu841, %w_s2_b34_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill843 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty845 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add846 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv844, %relu830 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty845 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty847 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu848 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add846 : tensor<1x1024x14x14xf16>)
    outs(%empty847 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init849 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill850 = linalg.fill ins(%cst : f16) outs(%init849 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv851 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu848, %w_s2_b35_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill850 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty852 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu853 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv851 : tensor<1x256x14x14xf16>)
    outs(%empty852 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad854 = tensor.pad %relu853 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init855 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill856 = linalg.fill ins(%cst : f16) outs(%init855 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv857 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad854, %w_s2_b35_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill856 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty858 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu859 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv857 : tensor<1x256x14x14xf16>)
    outs(%empty858 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init860 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill861 = linalg.fill ins(%cst : f16) outs(%init860 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv862 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu859, %w_s2_b35_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill861 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty863 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add864 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv862, %relu848 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty863 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty865 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu866 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add864 : tensor<1x1024x14x14xf16>)
    outs(%empty865 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // === Stage 3 ===
  // Bottleneck 1024->2048 mid=512 14x14 stride=2
  %init867 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill868 = linalg.fill ins(%cst : f16) outs(%init867 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv869 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu866, %w_s3_b0_c1 : tensor<1x1024x14x14xf16>, tensor<512x1024x1x1xf16>)
    outs(%fill868 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty870 = tensor.empty() : tensor<1x512x14x14xf16>
  %relu871 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv869 : tensor<1x512x14x14xf16>)
    outs(%empty870 : tensor<1x512x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x14x14xf16>
  %pad872 = tensor.pad %relu871 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x16x16xf16>
  %init873 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill874 = linalg.fill ins(%cst : f16) outs(%init873 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv875 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad872, %w_s3_b0_c2 : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>)
    outs(%fill874 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty876 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu877 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv875 : tensor<1x512x7x7xf16>)
    outs(%empty876 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %init878 = tensor.empty() : tensor<1x2048x7x7xf16>
  %fill879 = linalg.fill ins(%cst : f16) outs(%init878 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %conv880 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu877, %w_s3_b0_c3 : tensor<1x512x7x7xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill879 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  // 1x1 shortcut conv
  %init881 = tensor.empty() : tensor<1x2048x7x7xf16>
  %fill882 = linalg.fill ins(%cst : f16) outs(%init881 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %conv883 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu866, %w_s3_b0_sc : tensor<1x1024x14x14xf16>, tensor<2048x1024x1x1xf16>)
    outs(%fill882 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %empty884 = tensor.empty() : tensor<1x2048x7x7xf16>
  %add885 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv880, %conv883 : tensor<1x2048x7x7xf16>, tensor<1x2048x7x7xf16>)
    outs(%empty884 : tensor<1x2048x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x2048x7x7xf16>
  %empty886 = tensor.empty() : tensor<1x2048x7x7xf16>
  %relu887 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add885 : tensor<1x2048x7x7xf16>)
    outs(%empty886 : tensor<1x2048x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x7x7xf16>

  // Bottleneck 2048->2048 mid=512 7x7 stride=1
  %init888 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill889 = linalg.fill ins(%cst : f16) outs(%init888 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv890 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu887, %w_s3_b1_c1 : tensor<1x2048x7x7xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill889 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty891 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu892 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv890 : tensor<1x512x7x7xf16>)
    outs(%empty891 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad893 = tensor.pad %relu892 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init894 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill895 = linalg.fill ins(%cst : f16) outs(%init894 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv896 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad893, %w_s3_b1_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill895 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty897 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu898 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv896 : tensor<1x512x7x7xf16>)
    outs(%empty897 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %init899 = tensor.empty() : tensor<1x2048x7x7xf16>
  %fill900 = linalg.fill ins(%cst : f16) outs(%init899 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %conv901 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu898, %w_s3_b1_c3 : tensor<1x512x7x7xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill900 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %empty902 = tensor.empty() : tensor<1x2048x7x7xf16>
  %add903 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv901, %relu887 : tensor<1x2048x7x7xf16>, tensor<1x2048x7x7xf16>)
    outs(%empty902 : tensor<1x2048x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x2048x7x7xf16>
  %empty904 = tensor.empty() : tensor<1x2048x7x7xf16>
  %relu905 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add903 : tensor<1x2048x7x7xf16>)
    outs(%empty904 : tensor<1x2048x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x7x7xf16>

  // Bottleneck 2048->2048 mid=512 7x7 stride=1
  %init906 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill907 = linalg.fill ins(%cst : f16) outs(%init906 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv908 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu905, %w_s3_b2_c1 : tensor<1x2048x7x7xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill907 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty909 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu910 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv908 : tensor<1x512x7x7xf16>)
    outs(%empty909 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad911 = tensor.pad %relu910 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init912 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill913 = linalg.fill ins(%cst : f16) outs(%init912 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv914 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad911, %w_s3_b2_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill913 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty915 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu916 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv914 : tensor<1x512x7x7xf16>)
    outs(%empty915 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %init917 = tensor.empty() : tensor<1x2048x7x7xf16>
  %fill918 = linalg.fill ins(%cst : f16) outs(%init917 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %conv919 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu916, %w_s3_b2_c3 : tensor<1x512x7x7xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill918 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %empty920 = tensor.empty() : tensor<1x2048x7x7xf16>
  %add921 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv919, %relu905 : tensor<1x2048x7x7xf16>, tensor<1x2048x7x7xf16>)
    outs(%empty920 : tensor<1x2048x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x2048x7x7xf16>
  %empty922 = tensor.empty() : tensor<1x2048x7x7xf16>
  %relu923 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add921 : tensor<1x2048x7x7xf16>)
    outs(%empty922 : tensor<1x2048x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x7x7xf16>

  // FC as 7x7 conv: 2048->1000
  %fc_init924 = tensor.empty() : tensor<1x1000x1x1xf16>
  %fc_fill925 = linalg.fill ins(%cst : f16) outs(%fc_init924 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  %fc926 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu923, %w_fc : tensor<1x2048x7x7xf16>, tensor<1000x2048x7x7xf16>)
    outs(%fc_fill925 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  return %fc926 : tensor<1x1000x1x1xf16>
}
