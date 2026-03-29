func.func @resnet_basic_7_7_7_7(
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

  // === Stage 1 ===
  // BasicBlock 64->128 56x56 stride=2
  %pad110 = tensor.pad %relu109 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init111 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill112 = linalg.fill ins(%cst : f16) outs(%init111 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv113 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad110, %w_s1_b0_c1 : tensor<1x64x58x58xf16>, tensor<128x64x3x3xf16>)
    outs(%fill112 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty114 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu115 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv113 : tensor<1x128x28x28xf16>)
    outs(%empty114 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad116 = tensor.pad %relu115 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init117 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill118 = linalg.fill ins(%cst : f16) outs(%init117 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv119 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad116, %w_s1_b0_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill118 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  // 1x1 shortcut conv
  %init120 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill121 = linalg.fill ins(%cst : f16) outs(%init120 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv122 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu109, %w_s1_b0_sc : tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>)
    outs(%fill121 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty123 = tensor.empty() : tensor<1x128x28x28xf16>
  %add124 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv119, %conv122 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty123 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty125 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu126 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add124 : tensor<1x128x28x28xf16>)
    outs(%empty125 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad127 = tensor.pad %relu126 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init128 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill129 = linalg.fill ins(%cst : f16) outs(%init128 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv130 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad127, %w_s1_b1_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill129 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty131 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu132 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv130 : tensor<1x128x28x28xf16>)
    outs(%empty131 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad133 = tensor.pad %relu132 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init134 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill135 = linalg.fill ins(%cst : f16) outs(%init134 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv136 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad133, %w_s1_b1_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill135 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty137 = tensor.empty() : tensor<1x128x28x28xf16>
  %add138 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv136, %relu126 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty137 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty139 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu140 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add138 : tensor<1x128x28x28xf16>)
    outs(%empty139 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad141 = tensor.pad %relu140 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init142 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill143 = linalg.fill ins(%cst : f16) outs(%init142 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv144 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad141, %w_s1_b2_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill143 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty145 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu146 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv144 : tensor<1x128x28x28xf16>)
    outs(%empty145 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad147 = tensor.pad %relu146 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init148 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill149 = linalg.fill ins(%cst : f16) outs(%init148 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv150 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad147, %w_s1_b2_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill149 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty151 = tensor.empty() : tensor<1x128x28x28xf16>
  %add152 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv150, %relu140 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
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
  } ins(%pad155, %w_s1_b3_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
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
  } ins(%pad161, %w_s1_b3_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
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
  } ins(%pad169, %w_s1_b4_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
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
  } ins(%pad175, %w_s1_b4_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
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
  } ins(%pad183, %w_s1_b5_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
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
  } ins(%pad189, %w_s1_b5_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
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
  } ins(%pad197, %w_s1_b6_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
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
  } ins(%pad203, %w_s1_b6_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
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

  // === Stage 2 ===
  // BasicBlock 128->256 28x28 stride=2
  %pad211 = tensor.pad %relu210 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init212 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill213 = linalg.fill ins(%cst : f16) outs(%init212 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv214 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad211, %w_s2_b0_c1 : tensor<1x128x30x30xf16>, tensor<256x128x3x3xf16>)
    outs(%fill213 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty215 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu216 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv214 : tensor<1x256x14x14xf16>)
    outs(%empty215 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad217 = tensor.pad %relu216 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init218 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill219 = linalg.fill ins(%cst : f16) outs(%init218 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv220 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad217, %w_s2_b0_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill219 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  // 1x1 shortcut conv
  %init221 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill222 = linalg.fill ins(%cst : f16) outs(%init221 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv223 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu210, %w_s2_b0_sc : tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>)
    outs(%fill222 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty224 = tensor.empty() : tensor<1x256x14x14xf16>
  %add225 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv220, %conv223 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty224 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty226 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu227 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add225 : tensor<1x256x14x14xf16>)
    outs(%empty226 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad228 = tensor.pad %relu227 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init229 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill230 = linalg.fill ins(%cst : f16) outs(%init229 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv231 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad228, %w_s2_b1_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill230 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty232 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu233 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv231 : tensor<1x256x14x14xf16>)
    outs(%empty232 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad234 = tensor.pad %relu233 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init235 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill236 = linalg.fill ins(%cst : f16) outs(%init235 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv237 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad234, %w_s2_b1_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill236 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty238 = tensor.empty() : tensor<1x256x14x14xf16>
  %add239 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv237, %relu227 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty238 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty240 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu241 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add239 : tensor<1x256x14x14xf16>)
    outs(%empty240 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad242 = tensor.pad %relu241 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init243 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill244 = linalg.fill ins(%cst : f16) outs(%init243 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv245 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad242, %w_s2_b2_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  %pad248 = tensor.pad %relu247 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init249 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill250 = linalg.fill ins(%cst : f16) outs(%init249 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv251 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad248, %w_s2_b2_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill250 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty252 = tensor.empty() : tensor<1x256x14x14xf16>
  %add253 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv251, %relu241 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty252 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty254 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu255 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add253 : tensor<1x256x14x14xf16>)
    outs(%empty254 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad256 = tensor.pad %relu255 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init257 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill258 = linalg.fill ins(%cst : f16) outs(%init257 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv259 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad256, %w_s2_b3_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill258 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty260 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu261 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv259 : tensor<1x256x14x14xf16>)
    outs(%empty260 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad262 = tensor.pad %relu261 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init263 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill264 = linalg.fill ins(%cst : f16) outs(%init263 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv265 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad262, %w_s2_b3_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill264 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty266 = tensor.empty() : tensor<1x256x14x14xf16>
  %add267 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv265, %relu255 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty266 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty268 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu269 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add267 : tensor<1x256x14x14xf16>)
    outs(%empty268 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad270 = tensor.pad %relu269 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init271 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill272 = linalg.fill ins(%cst : f16) outs(%init271 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv273 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad270, %w_s2_b4_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill272 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty274 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu275 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv273 : tensor<1x256x14x14xf16>)
    outs(%empty274 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad276 = tensor.pad %relu275 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init277 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill278 = linalg.fill ins(%cst : f16) outs(%init277 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv279 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad276, %w_s2_b4_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill278 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty280 = tensor.empty() : tensor<1x256x14x14xf16>
  %add281 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv279, %relu269 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
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
  } ins(%pad284, %w_s2_b5_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%pad290, %w_s2_b5_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%pad298, %w_s2_b6_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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
  } ins(%pad304, %w_s2_b6_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
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

  // === Stage 3 ===
  // BasicBlock 256->512 14x14 stride=2
  %pad312 = tensor.pad %relu311 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init313 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill314 = linalg.fill ins(%cst : f16) outs(%init313 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv315 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad312, %w_s3_b0_c1 : tensor<1x256x16x16xf16>, tensor<512x256x3x3xf16>)
    outs(%fill314 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty316 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu317 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv315 : tensor<1x512x7x7xf16>)
    outs(%empty316 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad318 = tensor.pad %relu317 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init319 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill320 = linalg.fill ins(%cst : f16) outs(%init319 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv321 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad318, %w_s3_b0_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill320 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  // 1x1 shortcut conv
  %init322 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill323 = linalg.fill ins(%cst : f16) outs(%init322 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv324 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu311, %w_s3_b0_sc : tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>)
    outs(%fill323 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty325 = tensor.empty() : tensor<1x512x7x7xf16>
  %add326 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv321, %conv324 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty325 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty327 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu328 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add326 : tensor<1x512x7x7xf16>)
    outs(%empty327 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad329 = tensor.pad %relu328 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init330 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill331 = linalg.fill ins(%cst : f16) outs(%init330 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv332 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad329, %w_s3_b1_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill331 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty333 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu334 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv332 : tensor<1x512x7x7xf16>)
    outs(%empty333 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad335 = tensor.pad %relu334 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init336 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill337 = linalg.fill ins(%cst : f16) outs(%init336 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv338 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad335, %w_s3_b1_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill337 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty339 = tensor.empty() : tensor<1x512x7x7xf16>
  %add340 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv338, %relu328 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty339 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty341 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu342 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add340 : tensor<1x512x7x7xf16>)
    outs(%empty341 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad343 = tensor.pad %relu342 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init344 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill345 = linalg.fill ins(%cst : f16) outs(%init344 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv346 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad343, %w_s3_b2_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill345 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty347 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu348 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv346 : tensor<1x512x7x7xf16>)
    outs(%empty347 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad349 = tensor.pad %relu348 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init350 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill351 = linalg.fill ins(%cst : f16) outs(%init350 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv352 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad349, %w_s3_b2_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill351 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty353 = tensor.empty() : tensor<1x512x7x7xf16>
  %add354 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv352, %relu342 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty353 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty355 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu356 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add354 : tensor<1x512x7x7xf16>)
    outs(%empty355 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad357 = tensor.pad %relu356 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init358 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill359 = linalg.fill ins(%cst : f16) outs(%init358 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv360 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad357, %w_s3_b3_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill359 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty361 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu362 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv360 : tensor<1x512x7x7xf16>)
    outs(%empty361 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad363 = tensor.pad %relu362 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init364 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill365 = linalg.fill ins(%cst : f16) outs(%init364 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv366 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad363, %w_s3_b3_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill365 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty367 = tensor.empty() : tensor<1x512x7x7xf16>
  %add368 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv366, %relu356 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty367 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty369 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu370 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add368 : tensor<1x512x7x7xf16>)
    outs(%empty369 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad371 = tensor.pad %relu370 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init372 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill373 = linalg.fill ins(%cst : f16) outs(%init372 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv374 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad371, %w_s3_b4_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill373 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty375 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu376 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv374 : tensor<1x512x7x7xf16>)
    outs(%empty375 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad377 = tensor.pad %relu376 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init378 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill379 = linalg.fill ins(%cst : f16) outs(%init378 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv380 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad377, %w_s3_b4_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill379 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty381 = tensor.empty() : tensor<1x512x7x7xf16>
  %add382 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv380, %relu370 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty381 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty383 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu384 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add382 : tensor<1x512x7x7xf16>)
    outs(%empty383 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad385 = tensor.pad %relu384 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init386 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill387 = linalg.fill ins(%cst : f16) outs(%init386 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv388 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad385, %w_s3_b5_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill387 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty389 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu390 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv388 : tensor<1x512x7x7xf16>)
    outs(%empty389 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad391 = tensor.pad %relu390 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init392 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill393 = linalg.fill ins(%cst : f16) outs(%init392 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv394 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad391, %w_s3_b5_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill393 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty395 = tensor.empty() : tensor<1x512x7x7xf16>
  %add396 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv394, %relu384 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty395 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty397 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu398 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add396 : tensor<1x512x7x7xf16>)
    outs(%empty397 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad399 = tensor.pad %relu398 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init400 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill401 = linalg.fill ins(%cst : f16) outs(%init400 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv402 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad399, %w_s3_b6_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill401 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty403 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu404 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv402 : tensor<1x512x7x7xf16>)
    outs(%empty403 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad405 = tensor.pad %relu404 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init406 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill407 = linalg.fill ins(%cst : f16) outs(%init406 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv408 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad405, %w_s3_b6_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill407 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty409 = tensor.empty() : tensor<1x512x7x7xf16>
  %add410 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv408, %relu398 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
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

  // FC as 7x7 conv: 512->1000
  %fc_init413 = tensor.empty() : tensor<1x1000x1x1xf16>
  %fc_fill414 = linalg.fill ins(%cst : f16) outs(%fc_init413 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  %fc415 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu412, %w_fc : tensor<1x512x7x7xf16>, tensor<1000x512x7x7xf16>)
    outs(%fc_fill414 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  return %fc415 : tensor<1x1000x1x1xf16>
}
