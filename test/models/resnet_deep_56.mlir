func.func @resnet_basic_9_9_9_9(
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

  // === Stage 1 ===
  // BasicBlock 64->128 56x56 stride=2
  %pad138 = tensor.pad %relu137 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init139 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill140 = linalg.fill ins(%cst : f16) outs(%init139 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv141 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad138, %w_s1_b0_c1 : tensor<1x64x58x58xf16>, tensor<128x64x3x3xf16>)
    outs(%fill140 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty142 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu143 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv141 : tensor<1x128x28x28xf16>)
    outs(%empty142 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad144 = tensor.pad %relu143 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init145 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill146 = linalg.fill ins(%cst : f16) outs(%init145 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv147 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad144, %w_s1_b0_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill146 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  // 1x1 shortcut conv
  %init148 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill149 = linalg.fill ins(%cst : f16) outs(%init148 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv150 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu137, %w_s1_b0_sc : tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>)
    outs(%fill149 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty151 = tensor.empty() : tensor<1x128x28x28xf16>
  %add152 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv147, %conv150 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty151 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty153 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu154 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add152 : tensor<1x128x28x28xf16>)
    outs(%empty153 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad155 = tensor.pad %relu154 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init156 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill157 = linalg.fill ins(%cst : f16) outs(%init156 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv158 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad155, %w_s1_b1_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill157 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty159 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu160 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv158 : tensor<1x128x28x28xf16>)
    outs(%empty159 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad161 = tensor.pad %relu160 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init162 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill163 = linalg.fill ins(%cst : f16) outs(%init162 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv164 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad161, %w_s1_b1_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill163 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty165 = tensor.empty() : tensor<1x128x28x28xf16>
  %add166 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv164, %relu154 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty165 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty167 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu168 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add166 : tensor<1x128x28x28xf16>)
    outs(%empty167 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad169 = tensor.pad %relu168 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init170 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill171 = linalg.fill ins(%cst : f16) outs(%init170 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv172 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad169, %w_s1_b2_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill171 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty173 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu174 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv172 : tensor<1x128x28x28xf16>)
    outs(%empty173 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad175 = tensor.pad %relu174 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init176 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill177 = linalg.fill ins(%cst : f16) outs(%init176 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv178 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad175, %w_s1_b2_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill177 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty179 = tensor.empty() : tensor<1x128x28x28xf16>
  %add180 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv178, %relu168 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty179 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty181 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu182 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add180 : tensor<1x128x28x28xf16>)
    outs(%empty181 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad183 = tensor.pad %relu182 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init184 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill185 = linalg.fill ins(%cst : f16) outs(%init184 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv186 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad183, %w_s1_b3_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill185 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty187 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu188 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv186 : tensor<1x128x28x28xf16>)
    outs(%empty187 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad189 = tensor.pad %relu188 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init190 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill191 = linalg.fill ins(%cst : f16) outs(%init190 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv192 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad189, %w_s1_b3_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill191 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty193 = tensor.empty() : tensor<1x128x28x28xf16>
  %add194 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv192, %relu182 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty193 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty195 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu196 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add194 : tensor<1x128x28x28xf16>)
    outs(%empty195 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad197 = tensor.pad %relu196 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init198 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill199 = linalg.fill ins(%cst : f16) outs(%init198 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv200 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad197, %w_s1_b4_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
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
  } ins(%pad203, %w_s1_b4_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill205 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty207 = tensor.empty() : tensor<1x128x28x28xf16>
  %add208 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv206, %relu196 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty207 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty209 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu210 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add208 : tensor<1x128x28x28xf16>)
    outs(%empty209 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad211 = tensor.pad %relu210 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init212 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill213 = linalg.fill ins(%cst : f16) outs(%init212 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv214 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad211, %w_s1_b5_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill213 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty215 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu216 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv214 : tensor<1x128x28x28xf16>)
    outs(%empty215 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad217 = tensor.pad %relu216 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init218 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill219 = linalg.fill ins(%cst : f16) outs(%init218 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv220 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad217, %w_s1_b5_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill219 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty221 = tensor.empty() : tensor<1x128x28x28xf16>
  %add222 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv220, %relu210 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty221 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty223 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu224 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add222 : tensor<1x128x28x28xf16>)
    outs(%empty223 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad225 = tensor.pad %relu224 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init226 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill227 = linalg.fill ins(%cst : f16) outs(%init226 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv228 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad225, %w_s1_b6_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill227 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty229 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu230 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv228 : tensor<1x128x28x28xf16>)
    outs(%empty229 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad231 = tensor.pad %relu230 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init232 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill233 = linalg.fill ins(%cst : f16) outs(%init232 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv234 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad231, %w_s1_b6_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill233 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty235 = tensor.empty() : tensor<1x128x28x28xf16>
  %add236 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv234, %relu224 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty235 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty237 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu238 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add236 : tensor<1x128x28x28xf16>)
    outs(%empty237 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad239 = tensor.pad %relu238 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init240 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill241 = linalg.fill ins(%cst : f16) outs(%init240 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv242 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad239, %w_s1_b7_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill241 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty243 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu244 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv242 : tensor<1x128x28x28xf16>)
    outs(%empty243 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad245 = tensor.pad %relu244 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init246 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill247 = linalg.fill ins(%cst : f16) outs(%init246 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv248 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad245, %w_s1_b7_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill247 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty249 = tensor.empty() : tensor<1x128x28x28xf16>
  %add250 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv248, %relu238 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty249 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty251 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu252 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add250 : tensor<1x128x28x28xf16>)
    outs(%empty251 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad253 = tensor.pad %relu252 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init254 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill255 = linalg.fill ins(%cst : f16) outs(%init254 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv256 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad253, %w_s1_b8_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill255 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty257 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu258 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv256 : tensor<1x128x28x28xf16>)
    outs(%empty257 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad259 = tensor.pad %relu258 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init260 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill261 = linalg.fill ins(%cst : f16) outs(%init260 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv262 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad259, %w_s1_b8_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill261 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty263 = tensor.empty() : tensor<1x128x28x28xf16>
  %add264 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv262, %relu252 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty263 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty265 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu266 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add264 : tensor<1x128x28x28xf16>)
    outs(%empty265 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // === Stage 2 ===
  // BasicBlock 128->256 28x28 stride=2
  %pad267 = tensor.pad %relu266 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init268 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill269 = linalg.fill ins(%cst : f16) outs(%init268 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv270 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad267, %w_s2_b0_c1 : tensor<1x128x30x30xf16>, tensor<256x128x3x3xf16>)
    outs(%fill269 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty271 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu272 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv270 : tensor<1x256x14x14xf16>)
    outs(%empty271 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad273 = tensor.pad %relu272 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init274 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill275 = linalg.fill ins(%cst : f16) outs(%init274 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv276 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad273, %w_s2_b0_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill275 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  // 1x1 shortcut conv
  %init277 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill278 = linalg.fill ins(%cst : f16) outs(%init277 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv279 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu266, %w_s2_b0_sc : tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>)
    outs(%fill278 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty280 = tensor.empty() : tensor<1x256x14x14xf16>
  %add281 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv276, %conv279 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty280 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty282 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu283 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add281 : tensor<1x256x14x14xf16>)
    outs(%empty282 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad284 = tensor.pad %relu283 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init285 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill286 = linalg.fill ins(%cst : f16) outs(%init285 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv287 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad284, %w_s2_b1_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill286 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty288 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu289 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv287 : tensor<1x256x14x14xf16>)
    outs(%empty288 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad290 = tensor.pad %relu289 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init291 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill292 = linalg.fill ins(%cst : f16) outs(%init291 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv293 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad290, %w_s2_b1_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill292 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty294 = tensor.empty() : tensor<1x256x14x14xf16>
  %add295 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv293, %relu283 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty294 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty296 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu297 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add295 : tensor<1x256x14x14xf16>)
    outs(%empty296 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad298 = tensor.pad %relu297 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init299 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill300 = linalg.fill ins(%cst : f16) outs(%init299 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv301 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad298, %w_s2_b2_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill300 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty302 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu303 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv301 : tensor<1x256x14x14xf16>)
    outs(%empty302 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad304 = tensor.pad %relu303 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init305 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill306 = linalg.fill ins(%cst : f16) outs(%init305 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv307 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad304, %w_s2_b2_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill306 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty308 = tensor.empty() : tensor<1x256x14x14xf16>
  %add309 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv307, %relu297 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty308 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty310 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu311 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add309 : tensor<1x256x14x14xf16>)
    outs(%empty310 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad312 = tensor.pad %relu311 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init313 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill314 = linalg.fill ins(%cst : f16) outs(%init313 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv315 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad312, %w_s2_b3_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill314 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty316 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu317 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv315 : tensor<1x256x14x14xf16>)
    outs(%empty316 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad318 = tensor.pad %relu317 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init319 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill320 = linalg.fill ins(%cst : f16) outs(%init319 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv321 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad318, %w_s2_b3_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill320 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty322 = tensor.empty() : tensor<1x256x14x14xf16>
  %add323 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv321, %relu311 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty322 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty324 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu325 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add323 : tensor<1x256x14x14xf16>)
    outs(%empty324 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad326 = tensor.pad %relu325 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init327 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill328 = linalg.fill ins(%cst : f16) outs(%init327 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv329 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad326, %w_s2_b4_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%pad332, %w_s2_b4_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill334 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty336 = tensor.empty() : tensor<1x256x14x14xf16>
  %add337 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv335, %relu325 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty336 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty338 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu339 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add337 : tensor<1x256x14x14xf16>)
    outs(%empty338 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad340 = tensor.pad %relu339 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init341 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill342 = linalg.fill ins(%cst : f16) outs(%init341 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv343 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad340, %w_s2_b5_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill342 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty344 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu345 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv343 : tensor<1x256x14x14xf16>)
    outs(%empty344 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad346 = tensor.pad %relu345 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init347 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill348 = linalg.fill ins(%cst : f16) outs(%init347 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv349 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad346, %w_s2_b5_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill348 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty350 = tensor.empty() : tensor<1x256x14x14xf16>
  %add351 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv349, %relu339 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty350 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty352 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu353 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add351 : tensor<1x256x14x14xf16>)
    outs(%empty352 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad354 = tensor.pad %relu353 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init355 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill356 = linalg.fill ins(%cst : f16) outs(%init355 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv357 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad354, %w_s2_b6_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill356 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty358 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu359 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv357 : tensor<1x256x14x14xf16>)
    outs(%empty358 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad360 = tensor.pad %relu359 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init361 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill362 = linalg.fill ins(%cst : f16) outs(%init361 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv363 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad360, %w_s2_b6_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill362 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty364 = tensor.empty() : tensor<1x256x14x14xf16>
  %add365 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv363, %relu353 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty364 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty366 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu367 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add365 : tensor<1x256x14x14xf16>)
    outs(%empty366 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad368 = tensor.pad %relu367 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init369 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill370 = linalg.fill ins(%cst : f16) outs(%init369 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv371 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad368, %w_s2_b7_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  %pad374 = tensor.pad %relu373 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init375 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill376 = linalg.fill ins(%cst : f16) outs(%init375 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv377 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad374, %w_s2_b7_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill376 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty378 = tensor.empty() : tensor<1x256x14x14xf16>
  %add379 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv377, %relu367 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty378 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty380 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu381 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add379 : tensor<1x256x14x14xf16>)
    outs(%empty380 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad382 = tensor.pad %relu381 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init383 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill384 = linalg.fill ins(%cst : f16) outs(%init383 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv385 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad382, %w_s2_b8_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill384 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty386 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu387 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv385 : tensor<1x256x14x14xf16>)
    outs(%empty386 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad388 = tensor.pad %relu387 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init389 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill390 = linalg.fill ins(%cst : f16) outs(%init389 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv391 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad388, %w_s2_b8_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill390 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty392 = tensor.empty() : tensor<1x256x14x14xf16>
  %add393 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv391, %relu381 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty392 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty394 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu395 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add393 : tensor<1x256x14x14xf16>)
    outs(%empty394 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // === Stage 3 ===
  // BasicBlock 256->512 14x14 stride=2
  %pad396 = tensor.pad %relu395 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init397 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill398 = linalg.fill ins(%cst : f16) outs(%init397 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv399 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad396, %w_s3_b0_c1 : tensor<1x256x16x16xf16>, tensor<512x256x3x3xf16>)
    outs(%fill398 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty400 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu401 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv399 : tensor<1x512x7x7xf16>)
    outs(%empty400 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad402 = tensor.pad %relu401 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init403 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill404 = linalg.fill ins(%cst : f16) outs(%init403 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv405 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad402, %w_s3_b0_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill404 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  // 1x1 shortcut conv
  %init406 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill407 = linalg.fill ins(%cst : f16) outs(%init406 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv408 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu395, %w_s3_b0_sc : tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>)
    outs(%fill407 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty409 = tensor.empty() : tensor<1x512x7x7xf16>
  %add410 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv405, %conv408 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty409 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty411 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu412 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add410 : tensor<1x512x7x7xf16>)
    outs(%empty411 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad413 = tensor.pad %relu412 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init414 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill415 = linalg.fill ins(%cst : f16) outs(%init414 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv416 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad413, %w_s3_b1_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill415 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty417 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu418 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv416 : tensor<1x512x7x7xf16>)
    outs(%empty417 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad419 = tensor.pad %relu418 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init420 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill421 = linalg.fill ins(%cst : f16) outs(%init420 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv422 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad419, %w_s3_b1_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill421 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty423 = tensor.empty() : tensor<1x512x7x7xf16>
  %add424 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv422, %relu412 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty423 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty425 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu426 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add424 : tensor<1x512x7x7xf16>)
    outs(%empty425 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad427 = tensor.pad %relu426 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init428 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill429 = linalg.fill ins(%cst : f16) outs(%init428 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv430 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad427, %w_s3_b2_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill429 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty431 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu432 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv430 : tensor<1x512x7x7xf16>)
    outs(%empty431 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad433 = tensor.pad %relu432 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init434 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill435 = linalg.fill ins(%cst : f16) outs(%init434 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv436 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad433, %w_s3_b2_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill435 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty437 = tensor.empty() : tensor<1x512x7x7xf16>
  %add438 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv436, %relu426 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty437 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty439 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu440 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add438 : tensor<1x512x7x7xf16>)
    outs(%empty439 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad441 = tensor.pad %relu440 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init442 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill443 = linalg.fill ins(%cst : f16) outs(%init442 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv444 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad441, %w_s3_b3_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill443 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty445 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu446 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv444 : tensor<1x512x7x7xf16>)
    outs(%empty445 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad447 = tensor.pad %relu446 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init448 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill449 = linalg.fill ins(%cst : f16) outs(%init448 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv450 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad447, %w_s3_b3_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill449 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty451 = tensor.empty() : tensor<1x512x7x7xf16>
  %add452 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv450, %relu440 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty451 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty453 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu454 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add452 : tensor<1x512x7x7xf16>)
    outs(%empty453 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad455 = tensor.pad %relu454 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init456 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill457 = linalg.fill ins(%cst : f16) outs(%init456 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv458 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad455, %w_s3_b4_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill457 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty459 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu460 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv458 : tensor<1x512x7x7xf16>)
    outs(%empty459 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad461 = tensor.pad %relu460 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init462 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill463 = linalg.fill ins(%cst : f16) outs(%init462 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv464 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad461, %w_s3_b4_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill463 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty465 = tensor.empty() : tensor<1x512x7x7xf16>
  %add466 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv464, %relu454 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty465 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty467 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu468 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add466 : tensor<1x512x7x7xf16>)
    outs(%empty467 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad469 = tensor.pad %relu468 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init470 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill471 = linalg.fill ins(%cst : f16) outs(%init470 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv472 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad469, %w_s3_b5_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill471 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty473 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu474 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv472 : tensor<1x512x7x7xf16>)
    outs(%empty473 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad475 = tensor.pad %relu474 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init476 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill477 = linalg.fill ins(%cst : f16) outs(%init476 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv478 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad475, %w_s3_b5_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill477 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty479 = tensor.empty() : tensor<1x512x7x7xf16>
  %add480 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv478, %relu468 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty479 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty481 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu482 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add480 : tensor<1x512x7x7xf16>)
    outs(%empty481 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad483 = tensor.pad %relu482 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init484 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill485 = linalg.fill ins(%cst : f16) outs(%init484 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv486 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad483, %w_s3_b6_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill485 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty487 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu488 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv486 : tensor<1x512x7x7xf16>)
    outs(%empty487 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad489 = tensor.pad %relu488 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init490 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill491 = linalg.fill ins(%cst : f16) outs(%init490 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv492 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad489, %w_s3_b6_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill491 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty493 = tensor.empty() : tensor<1x512x7x7xf16>
  %add494 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv492, %relu482 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty493 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty495 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu496 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add494 : tensor<1x512x7x7xf16>)
    outs(%empty495 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad497 = tensor.pad %relu496 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init498 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill499 = linalg.fill ins(%cst : f16) outs(%init498 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv500 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad497, %w_s3_b7_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill499 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty501 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu502 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv500 : tensor<1x512x7x7xf16>)
    outs(%empty501 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad503 = tensor.pad %relu502 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init504 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill505 = linalg.fill ins(%cst : f16) outs(%init504 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv506 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad503, %w_s3_b7_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill505 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty507 = tensor.empty() : tensor<1x512x7x7xf16>
  %add508 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv506, %relu496 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty507 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty509 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu510 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add508 : tensor<1x512x7x7xf16>)
    outs(%empty509 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad511 = tensor.pad %relu510 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init512 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill513 = linalg.fill ins(%cst : f16) outs(%init512 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv514 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad511, %w_s3_b8_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill513 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty515 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu516 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv514 : tensor<1x512x7x7xf16>)
    outs(%empty515 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad517 = tensor.pad %relu516 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init518 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill519 = linalg.fill ins(%cst : f16) outs(%init518 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv520 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad517, %w_s3_b8_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill519 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty521 = tensor.empty() : tensor<1x512x7x7xf16>
  %add522 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv520, %relu510 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty521 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty523 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu524 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add522 : tensor<1x512x7x7xf16>)
    outs(%empty523 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // FC as 7x7 conv: 512->1000
  %fc_init525 = tensor.empty() : tensor<1x1000x1x1xf16>
  %fc_fill526 = linalg.fill ins(%cst : f16) outs(%fc_init525 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  %fc527 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu524, %w_fc : tensor<1x512x7x7xf16>, tensor<1000x512x7x7xf16>)
    outs(%fc_fill526 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  return %fc527 : tensor<1x1000x1x1xf16>
}
