func.func @resnet_bottleneck_3_4_23_3(
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

  // === Stage 2 ===
  // Bottleneck 512->1024 mid=256 28x28 stride=2
  %init144 = tensor.empty() : tensor<1x256x28x28xf16>
  %fill145 = linalg.fill ins(%cst : f16) outs(%init144 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %conv146 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu143, %w_s2_b0_c1 : tensor<1x512x28x28xf16>, tensor<256x512x1x1xf16>)
    outs(%fill145 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %empty147 = tensor.empty() : tensor<1x256x28x28xf16>
  %relu148 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv146 : tensor<1x256x28x28xf16>)
    outs(%empty147 : tensor<1x256x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x28x28xf16>
  %pad149 = tensor.pad %relu148 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x28x28xf16> to tensor<1x256x30x30xf16>
  %init150 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill151 = linalg.fill ins(%cst : f16) outs(%init150 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv152 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad149, %w_s2_b0_c2 : tensor<1x256x30x30xf16>, tensor<256x256x3x3xf16>)
    outs(%fill151 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty153 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu154 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv152 : tensor<1x256x14x14xf16>)
    outs(%empty153 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init155 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill156 = linalg.fill ins(%cst : f16) outs(%init155 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv157 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu154, %w_s2_b0_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill156 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  // 1x1 shortcut conv
  %init158 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill159 = linalg.fill ins(%cst : f16) outs(%init158 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv160 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu143, %w_s2_b0_sc : tensor<1x512x28x28xf16>, tensor<1024x512x1x1xf16>)
    outs(%fill159 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty161 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add162 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv157, %conv160 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty161 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty163 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu164 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add162 : tensor<1x1024x14x14xf16>)
    outs(%empty163 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init165 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill166 = linalg.fill ins(%cst : f16) outs(%init165 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv167 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu164, %w_s2_b1_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill166 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty168 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu169 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv167 : tensor<1x256x14x14xf16>)
    outs(%empty168 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad170 = tensor.pad %relu169 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init171 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill172 = linalg.fill ins(%cst : f16) outs(%init171 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv173 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad170, %w_s2_b1_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill172 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty174 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu175 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv173 : tensor<1x256x14x14xf16>)
    outs(%empty174 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init176 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill177 = linalg.fill ins(%cst : f16) outs(%init176 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv178 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu175, %w_s2_b1_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill177 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty179 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add180 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv178, %relu164 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty179 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty181 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu182 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add180 : tensor<1x1024x14x14xf16>)
    outs(%empty181 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init183 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill184 = linalg.fill ins(%cst : f16) outs(%init183 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv185 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu182, %w_s2_b2_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill184 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty186 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu187 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv185 : tensor<1x256x14x14xf16>)
    outs(%empty186 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad188 = tensor.pad %relu187 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init189 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill190 = linalg.fill ins(%cst : f16) outs(%init189 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv191 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad188, %w_s2_b2_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill190 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty192 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu193 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv191 : tensor<1x256x14x14xf16>)
    outs(%empty192 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init194 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill195 = linalg.fill ins(%cst : f16) outs(%init194 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv196 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu193, %w_s2_b2_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill195 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty197 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add198 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv196, %relu182 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty197 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty199 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu200 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add198 : tensor<1x1024x14x14xf16>)
    outs(%empty199 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init201 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill202 = linalg.fill ins(%cst : f16) outs(%init201 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv203 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu200, %w_s2_b3_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill202 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty204 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu205 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv203 : tensor<1x256x14x14xf16>)
    outs(%empty204 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad206 = tensor.pad %relu205 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init207 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill208 = linalg.fill ins(%cst : f16) outs(%init207 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv209 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad206, %w_s2_b3_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill208 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty210 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu211 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv209 : tensor<1x256x14x14xf16>)
    outs(%empty210 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init212 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill213 = linalg.fill ins(%cst : f16) outs(%init212 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv214 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu211, %w_s2_b3_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill213 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty215 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add216 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv214, %relu200 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
    outs(%empty215 : tensor<1x1024x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x14x14xf16>
  %empty217 = tensor.empty() : tensor<1x1024x14x14xf16>
  %relu218 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add216 : tensor<1x1024x14x14xf16>)
    outs(%empty217 : tensor<1x1024x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x14x14xf16>

  // Bottleneck 1024->1024 mid=256 14x14 stride=1
  %init219 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill220 = linalg.fill ins(%cst : f16) outs(%init219 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv221 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu218, %w_s2_b4_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill220 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty222 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu223 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv221 : tensor<1x256x14x14xf16>)
    outs(%empty222 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad224 = tensor.pad %relu223 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init225 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill226 = linalg.fill ins(%cst : f16) outs(%init225 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv227 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad224, %w_s2_b4_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill226 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty228 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu229 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv227 : tensor<1x256x14x14xf16>)
    outs(%empty228 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init230 = tensor.empty() : tensor<1x1024x14x14xf16>
  %fill231 = linalg.fill ins(%cst : f16) outs(%init230 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %conv232 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu229, %w_s2_b4_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill231 : tensor<1x1024x14x14xf16>) -> tensor<1x1024x14x14xf16>
  %empty233 = tensor.empty() : tensor<1x1024x14x14xf16>
  %add234 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv232, %relu218 : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16>)
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
  } ins(%relu236, %w_s2_b5_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad242, %w_s2_b5_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%relu247, %w_s2_b5_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
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
  } ins(%relu254, %w_s2_b6_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad260, %w_s2_b6_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%relu265, %w_s2_b6_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
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
  } ins(%relu272, %w_s2_b7_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad278, %w_s2_b7_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%relu283, %w_s2_b7_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
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
  } ins(%relu290, %w_s2_b8_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad296, %w_s2_b8_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%relu301, %w_s2_b8_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
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
  } ins(%relu308, %w_s2_b9_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad314, %w_s2_b9_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%relu319, %w_s2_b9_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
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
  } ins(%relu326, %w_s2_b10_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad332, %w_s2_b10_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%relu337, %w_s2_b10_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
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
  } ins(%relu344, %w_s2_b11_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad350, %w_s2_b11_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%relu355, %w_s2_b11_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
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
  } ins(%relu362, %w_s2_b12_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad368, %w_s2_b12_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%relu373, %w_s2_b12_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
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
  } ins(%relu380, %w_s2_b13_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad386, %w_s2_b13_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%relu391, %w_s2_b13_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
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
  } ins(%relu398, %w_s2_b14_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad404, %w_s2_b14_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%relu409, %w_s2_b14_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
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
  } ins(%relu416, %w_s2_b15_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad422, %w_s2_b15_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%relu427, %w_s2_b15_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
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
  } ins(%relu434, %w_s2_b16_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad440, %w_s2_b16_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%relu445, %w_s2_b16_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
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
  } ins(%relu452, %w_s2_b17_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad458, %w_s2_b17_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%relu463, %w_s2_b17_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
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
  } ins(%relu470, %w_s2_b18_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad476, %w_s2_b18_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%relu481, %w_s2_b18_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
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
  } ins(%relu488, %w_s2_b19_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad494, %w_s2_b19_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%relu499, %w_s2_b19_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
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
  } ins(%relu506, %w_s2_b20_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad512, %w_s2_b20_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%relu517, %w_s2_b20_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
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
  } ins(%relu524, %w_s2_b21_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad530, %w_s2_b21_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%relu535, %w_s2_b21_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
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
  } ins(%relu542, %w_s2_b22_c1 : tensor<1x1024x14x14xf16>, tensor<256x1024x1x1xf16>)
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
  } ins(%pad548, %w_s2_b22_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%relu553, %w_s2_b22_c3 : tensor<1x256x14x14xf16>, tensor<1024x256x1x1xf16>)
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

  // === Stage 3 ===
  // Bottleneck 1024->2048 mid=512 14x14 stride=2
  %init561 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill562 = linalg.fill ins(%cst : f16) outs(%init561 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv563 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu560, %w_s3_b0_c1 : tensor<1x1024x14x14xf16>, tensor<512x1024x1x1xf16>)
    outs(%fill562 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty564 = tensor.empty() : tensor<1x512x14x14xf16>
  %relu565 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv563 : tensor<1x512x14x14xf16>)
    outs(%empty564 : tensor<1x512x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x14x14xf16>
  %pad566 = tensor.pad %relu565 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x16x16xf16>
  %init567 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill568 = linalg.fill ins(%cst : f16) outs(%init567 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv569 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad566, %w_s3_b0_c2 : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>)
    outs(%fill568 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty570 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu571 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv569 : tensor<1x512x7x7xf16>)
    outs(%empty570 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %init572 = tensor.empty() : tensor<1x2048x7x7xf16>
  %fill573 = linalg.fill ins(%cst : f16) outs(%init572 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %conv574 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu571, %w_s3_b0_c3 : tensor<1x512x7x7xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill573 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  // 1x1 shortcut conv
  %init575 = tensor.empty() : tensor<1x2048x7x7xf16>
  %fill576 = linalg.fill ins(%cst : f16) outs(%init575 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %conv577 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu560, %w_s3_b0_sc : tensor<1x1024x14x14xf16>, tensor<2048x1024x1x1xf16>)
    outs(%fill576 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %empty578 = tensor.empty() : tensor<1x2048x7x7xf16>
  %add579 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv574, %conv577 : tensor<1x2048x7x7xf16>, tensor<1x2048x7x7xf16>)
    outs(%empty578 : tensor<1x2048x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x2048x7x7xf16>
  %empty580 = tensor.empty() : tensor<1x2048x7x7xf16>
  %relu581 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add579 : tensor<1x2048x7x7xf16>)
    outs(%empty580 : tensor<1x2048x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x7x7xf16>

  // Bottleneck 2048->2048 mid=512 7x7 stride=1
  %init582 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill583 = linalg.fill ins(%cst : f16) outs(%init582 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv584 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu581, %w_s3_b1_c1 : tensor<1x2048x7x7xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill583 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty585 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu586 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv584 : tensor<1x512x7x7xf16>)
    outs(%empty585 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad587 = tensor.pad %relu586 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init588 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill589 = linalg.fill ins(%cst : f16) outs(%init588 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv590 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad587, %w_s3_b1_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill589 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty591 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu592 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv590 : tensor<1x512x7x7xf16>)
    outs(%empty591 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %init593 = tensor.empty() : tensor<1x2048x7x7xf16>
  %fill594 = linalg.fill ins(%cst : f16) outs(%init593 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %conv595 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu592, %w_s3_b1_c3 : tensor<1x512x7x7xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill594 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %empty596 = tensor.empty() : tensor<1x2048x7x7xf16>
  %add597 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv595, %relu581 : tensor<1x2048x7x7xf16>, tensor<1x2048x7x7xf16>)
    outs(%empty596 : tensor<1x2048x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x2048x7x7xf16>
  %empty598 = tensor.empty() : tensor<1x2048x7x7xf16>
  %relu599 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add597 : tensor<1x2048x7x7xf16>)
    outs(%empty598 : tensor<1x2048x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x7x7xf16>

  // Bottleneck 2048->2048 mid=512 7x7 stride=1
  %init600 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill601 = linalg.fill ins(%cst : f16) outs(%init600 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv602 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu599, %w_s3_b2_c1 : tensor<1x2048x7x7xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill601 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty603 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu604 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv602 : tensor<1x512x7x7xf16>)
    outs(%empty603 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad605 = tensor.pad %relu604 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init606 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill607 = linalg.fill ins(%cst : f16) outs(%init606 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv608 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad605, %w_s3_b2_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill607 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty609 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu610 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv608 : tensor<1x512x7x7xf16>)
    outs(%empty609 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %init611 = tensor.empty() : tensor<1x2048x7x7xf16>
  %fill612 = linalg.fill ins(%cst : f16) outs(%init611 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %conv613 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu610, %w_s3_b2_c3 : tensor<1x512x7x7xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill612 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %empty614 = tensor.empty() : tensor<1x2048x7x7xf16>
  %add615 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv613, %relu599 : tensor<1x2048x7x7xf16>, tensor<1x2048x7x7xf16>)
    outs(%empty614 : tensor<1x2048x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x2048x7x7xf16>
  %empty616 = tensor.empty() : tensor<1x2048x7x7xf16>
  %relu617 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add615 : tensor<1x2048x7x7xf16>)
    outs(%empty616 : tensor<1x2048x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x7x7xf16>

  // FC as 7x7 conv: 2048->1000
  %fc_init618 = tensor.empty() : tensor<1x1000x1x1xf16>
  %fc_fill619 = linalg.fill ins(%cst : f16) outs(%fc_init618 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  %fc620 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu617, %w_fc : tensor<1x2048x7x7xf16>, tensor<1000x2048x7x7xf16>)
    outs(%fc_fill619 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  return %fc620 : tensor<1x1000x1x1xf16>
}
