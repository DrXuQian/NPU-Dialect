func.func @resnet_bottleneck_3_4_6_3(
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

  // === Stage 3 ===
  // Bottleneck 1024->2048 mid=512 14x14 stride=2
  %init255 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill256 = linalg.fill ins(%cst : f16) outs(%init255 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv257 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu254, %w_s3_b0_c1 : tensor<1x1024x14x14xf16>, tensor<512x1024x1x1xf16>)
    outs(%fill256 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty258 = tensor.empty() : tensor<1x512x14x14xf16>
  %relu259 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv257 : tensor<1x512x14x14xf16>)
    outs(%empty258 : tensor<1x512x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x14x14xf16>
  %pad260 = tensor.pad %relu259 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x16x16xf16>
  %init261 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill262 = linalg.fill ins(%cst : f16) outs(%init261 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv263 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad260, %w_s3_b0_c2 : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>)
    outs(%fill262 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty264 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu265 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv263 : tensor<1x512x7x7xf16>)
    outs(%empty264 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %init266 = tensor.empty() : tensor<1x2048x7x7xf16>
  %fill267 = linalg.fill ins(%cst : f16) outs(%init266 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %conv268 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu265, %w_s3_b0_c3 : tensor<1x512x7x7xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill267 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  // 1x1 shortcut conv
  %init269 = tensor.empty() : tensor<1x2048x7x7xf16>
  %fill270 = linalg.fill ins(%cst : f16) outs(%init269 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %conv271 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu254, %w_s3_b0_sc : tensor<1x1024x14x14xf16>, tensor<2048x1024x1x1xf16>)
    outs(%fill270 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %empty272 = tensor.empty() : tensor<1x2048x7x7xf16>
  %add273 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv268, %conv271 : tensor<1x2048x7x7xf16>, tensor<1x2048x7x7xf16>)
    outs(%empty272 : tensor<1x2048x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x2048x7x7xf16>
  %empty274 = tensor.empty() : tensor<1x2048x7x7xf16>
  %relu275 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add273 : tensor<1x2048x7x7xf16>)
    outs(%empty274 : tensor<1x2048x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x7x7xf16>

  // Bottleneck 2048->2048 mid=512 7x7 stride=1
  %init276 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill277 = linalg.fill ins(%cst : f16) outs(%init276 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv278 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu275, %w_s3_b1_c1 : tensor<1x2048x7x7xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill277 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty279 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu280 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv278 : tensor<1x512x7x7xf16>)
    outs(%empty279 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad281 = tensor.pad %relu280 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init282 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill283 = linalg.fill ins(%cst : f16) outs(%init282 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv284 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad281, %w_s3_b1_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill283 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty285 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu286 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv284 : tensor<1x512x7x7xf16>)
    outs(%empty285 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %init287 = tensor.empty() : tensor<1x2048x7x7xf16>
  %fill288 = linalg.fill ins(%cst : f16) outs(%init287 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %conv289 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu286, %w_s3_b1_c3 : tensor<1x512x7x7xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill288 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %empty290 = tensor.empty() : tensor<1x2048x7x7xf16>
  %add291 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv289, %relu275 : tensor<1x2048x7x7xf16>, tensor<1x2048x7x7xf16>)
    outs(%empty290 : tensor<1x2048x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x2048x7x7xf16>
  %empty292 = tensor.empty() : tensor<1x2048x7x7xf16>
  %relu293 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add291 : tensor<1x2048x7x7xf16>)
    outs(%empty292 : tensor<1x2048x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x7x7xf16>

  // Bottleneck 2048->2048 mid=512 7x7 stride=1
  %init294 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill295 = linalg.fill ins(%cst : f16) outs(%init294 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv296 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu293, %w_s3_b2_c1 : tensor<1x2048x7x7xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill295 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty297 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu298 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv296 : tensor<1x512x7x7xf16>)
    outs(%empty297 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad299 = tensor.pad %relu298 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init300 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill301 = linalg.fill ins(%cst : f16) outs(%init300 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv302 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad299, %w_s3_b2_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill301 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty303 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu304 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv302 : tensor<1x512x7x7xf16>)
    outs(%empty303 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %init305 = tensor.empty() : tensor<1x2048x7x7xf16>
  %fill306 = linalg.fill ins(%cst : f16) outs(%init305 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %conv307 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu304, %w_s3_b2_c3 : tensor<1x512x7x7xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill306 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x7x7xf16>
  %empty308 = tensor.empty() : tensor<1x2048x7x7xf16>
  %add309 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv307, %relu293 : tensor<1x2048x7x7xf16>, tensor<1x2048x7x7xf16>)
    outs(%empty308 : tensor<1x2048x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x2048x7x7xf16>
  %empty310 = tensor.empty() : tensor<1x2048x7x7xf16>
  %relu311 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add309 : tensor<1x2048x7x7xf16>)
    outs(%empty310 : tensor<1x2048x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x7x7xf16>

  // FC as 7x7 conv: 2048->1000
  %fc_init312 = tensor.empty() : tensor<1x1000x1x1xf16>
  %fc_fill313 = linalg.fill ins(%cst : f16) outs(%fc_init312 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  %fc314 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu311, %w_fc : tensor<1x2048x7x7xf16>, tensor<1000x2048x7x7xf16>)
    outs(%fc_fill313 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  return %fc314 : tensor<1x1000x1x1xf16>
}
