func.func @resnet_basic_3_4_6_3(
    %input: tensor<1x3x224x224xf16>,
    %w_conv1: tensor<64x3x7x7xf16>,
    %w_pool: tensor<64x64x3x3xf16>,
    %w_s0_b0_c1: tensor<64x64x3x3xf16>,
    %w_s0_b0_c2: tensor<64x64x3x3xf16>,
    %w_s0_b1_c1: tensor<64x64x3x3xf16>,
    %w_s0_b1_c2: tensor<64x64x3x3xf16>,
    %w_s0_b2_c1: tensor<64x64x3x3xf16>,
    %w_s0_b2_c2: tensor<64x64x3x3xf16>,
    %w_s1_b0_c1: tensor<128x64x3x3xf16>,
    %w_s1_b0_c2: tensor<128x128x3x3xf16>,
    %w_s1_b0_sc: tensor<128x64x1x1xf16>,
    %w_s1_b1_c1: tensor<128x128x3x3xf16>,
    %w_s1_b1_c2: tensor<128x128x3x3xf16>,
    %w_s1_b2_c1: tensor<128x128x3x3xf16>,
    %w_s1_b2_c2: tensor<128x128x3x3xf16>,
    %w_s1_b3_c1: tensor<128x128x3x3xf16>,
    %w_s1_b3_c2: tensor<128x128x3x3xf16>,
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
    %w_s3_b0_c1: tensor<512x256x3x3xf16>,
    %w_s3_b0_c2: tensor<512x512x3x3xf16>,
    %w_s3_b0_sc: tensor<512x256x1x1xf16>,
    %w_s3_b1_c1: tensor<512x512x3x3xf16>,
    %w_s3_b1_c2: tensor<512x512x3x3xf16>,
    %w_s3_b2_c1: tensor<512x512x3x3xf16>,
    %w_s3_b2_c2: tensor<512x512x3x3xf16>,
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

  // === Stage 1 ===
  // BasicBlock 64->128 56x56 stride=2
  %pad54 = tensor.pad %relu53 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init55 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill56 = linalg.fill ins(%cst : f16) outs(%init55 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv57 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad54, %w_s1_b0_c1 : tensor<1x64x58x58xf16>, tensor<128x64x3x3xf16>)
    outs(%fill56 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty58 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu59 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv57 : tensor<1x128x28x28xf16>)
    outs(%empty58 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad60 = tensor.pad %relu59 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init61 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill62 = linalg.fill ins(%cst : f16) outs(%init61 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv63 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad60, %w_s1_b0_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill62 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  // 1x1 shortcut conv
  %init64 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill65 = linalg.fill ins(%cst : f16) outs(%init64 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv66 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu53, %w_s1_b0_sc : tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>)
    outs(%fill65 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty67 = tensor.empty() : tensor<1x128x28x28xf16>
  %add68 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv63, %conv66 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty67 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty69 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu70 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add68 : tensor<1x128x28x28xf16>)
    outs(%empty69 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad71 = tensor.pad %relu70 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init72 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill73 = linalg.fill ins(%cst : f16) outs(%init72 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv74 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad71, %w_s1_b1_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill73 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty75 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu76 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv74 : tensor<1x128x28x28xf16>)
    outs(%empty75 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad77 = tensor.pad %relu76 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init78 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill79 = linalg.fill ins(%cst : f16) outs(%init78 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv80 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad77, %w_s1_b1_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill79 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty81 = tensor.empty() : tensor<1x128x28x28xf16>
  %add82 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv80, %relu70 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty81 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty83 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu84 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add82 : tensor<1x128x28x28xf16>)
    outs(%empty83 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad85 = tensor.pad %relu84 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init86 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill87 = linalg.fill ins(%cst : f16) outs(%init86 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv88 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad85, %w_s1_b2_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill87 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty89 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu90 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv88 : tensor<1x128x28x28xf16>)
    outs(%empty89 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad91 = tensor.pad %relu90 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init92 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill93 = linalg.fill ins(%cst : f16) outs(%init92 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv94 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad91, %w_s1_b2_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill93 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty95 = tensor.empty() : tensor<1x128x28x28xf16>
  %add96 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv94, %relu84 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty95 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty97 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu98 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add96 : tensor<1x128x28x28xf16>)
    outs(%empty97 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad99 = tensor.pad %relu98 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init100 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill101 = linalg.fill ins(%cst : f16) outs(%init100 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv102 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad99, %w_s1_b3_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill101 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty103 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu104 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv102 : tensor<1x128x28x28xf16>)
    outs(%empty103 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad105 = tensor.pad %relu104 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init106 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill107 = linalg.fill ins(%cst : f16) outs(%init106 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv108 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad105, %w_s1_b3_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill107 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty109 = tensor.empty() : tensor<1x128x28x28xf16>
  %add110 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv108, %relu98 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty109 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty111 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu112 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add110 : tensor<1x128x28x28xf16>)
    outs(%empty111 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // === Stage 2 ===
  // BasicBlock 128->256 28x28 stride=2
  %pad113 = tensor.pad %relu112 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init114 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill115 = linalg.fill ins(%cst : f16) outs(%init114 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv116 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad113, %w_s2_b0_c1 : tensor<1x128x30x30xf16>, tensor<256x128x3x3xf16>)
    outs(%fill115 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty117 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu118 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv116 : tensor<1x256x14x14xf16>)
    outs(%empty117 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad119 = tensor.pad %relu118 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init120 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill121 = linalg.fill ins(%cst : f16) outs(%init120 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv122 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad119, %w_s2_b0_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill121 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  // 1x1 shortcut conv
  %init123 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill124 = linalg.fill ins(%cst : f16) outs(%init123 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv125 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu112, %w_s2_b0_sc : tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>)
    outs(%fill124 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty126 = tensor.empty() : tensor<1x256x14x14xf16>
  %add127 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv122, %conv125 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty126 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty128 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu129 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add127 : tensor<1x256x14x14xf16>)
    outs(%empty128 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad130 = tensor.pad %relu129 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init131 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill132 = linalg.fill ins(%cst : f16) outs(%init131 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv133 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad130, %w_s2_b1_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill132 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty134 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu135 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv133 : tensor<1x256x14x14xf16>)
    outs(%empty134 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad136 = tensor.pad %relu135 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init137 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill138 = linalg.fill ins(%cst : f16) outs(%init137 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv139 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad136, %w_s2_b1_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill138 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty140 = tensor.empty() : tensor<1x256x14x14xf16>
  %add141 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv139, %relu129 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty140 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty142 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu143 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add141 : tensor<1x256x14x14xf16>)
    outs(%empty142 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad144 = tensor.pad %relu143 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init145 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill146 = linalg.fill ins(%cst : f16) outs(%init145 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv147 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad144, %w_s2_b2_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill146 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty148 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu149 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv147 : tensor<1x256x14x14xf16>)
    outs(%empty148 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad150 = tensor.pad %relu149 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init151 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill152 = linalg.fill ins(%cst : f16) outs(%init151 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv153 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad150, %w_s2_b2_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill152 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty154 = tensor.empty() : tensor<1x256x14x14xf16>
  %add155 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv153, %relu143 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty154 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty156 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu157 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add155 : tensor<1x256x14x14xf16>)
    outs(%empty156 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad158 = tensor.pad %relu157 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init159 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill160 = linalg.fill ins(%cst : f16) outs(%init159 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv161 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad158, %w_s2_b3_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill160 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty162 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu163 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv161 : tensor<1x256x14x14xf16>)
    outs(%empty162 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad164 = tensor.pad %relu163 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init165 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill166 = linalg.fill ins(%cst : f16) outs(%init165 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv167 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad164, %w_s2_b3_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill166 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty168 = tensor.empty() : tensor<1x256x14x14xf16>
  %add169 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv167, %relu157 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty168 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty170 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu171 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add169 : tensor<1x256x14x14xf16>)
    outs(%empty170 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad172 = tensor.pad %relu171 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init173 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill174 = linalg.fill ins(%cst : f16) outs(%init173 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv175 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad172, %w_s2_b4_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill174 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty176 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu177 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv175 : tensor<1x256x14x14xf16>)
    outs(%empty176 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad178 = tensor.pad %relu177 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init179 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill180 = linalg.fill ins(%cst : f16) outs(%init179 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv181 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad178, %w_s2_b4_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill180 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty182 = tensor.empty() : tensor<1x256x14x14xf16>
  %add183 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv181, %relu171 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty182 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty184 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu185 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add183 : tensor<1x256x14x14xf16>)
    outs(%empty184 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad186 = tensor.pad %relu185 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init187 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill188 = linalg.fill ins(%cst : f16) outs(%init187 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv189 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad186, %w_s2_b5_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill188 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty190 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu191 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv189 : tensor<1x256x14x14xf16>)
    outs(%empty190 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad192 = tensor.pad %relu191 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init193 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill194 = linalg.fill ins(%cst : f16) outs(%init193 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv195 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad192, %w_s2_b5_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill194 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty196 = tensor.empty() : tensor<1x256x14x14xf16>
  %add197 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv195, %relu185 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty196 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty198 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu199 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add197 : tensor<1x256x14x14xf16>)
    outs(%empty198 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // === Stage 3 ===
  // BasicBlock 256->512 14x14 stride=2
  %pad200 = tensor.pad %relu199 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init201 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill202 = linalg.fill ins(%cst : f16) outs(%init201 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv203 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad200, %w_s3_b0_c1 : tensor<1x256x16x16xf16>, tensor<512x256x3x3xf16>)
    outs(%fill202 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty204 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu205 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv203 : tensor<1x512x7x7xf16>)
    outs(%empty204 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad206 = tensor.pad %relu205 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init207 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill208 = linalg.fill ins(%cst : f16) outs(%init207 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv209 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad206, %w_s3_b0_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill208 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  // 1x1 shortcut conv
  %init210 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill211 = linalg.fill ins(%cst : f16) outs(%init210 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv212 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu199, %w_s3_b0_sc : tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>)
    outs(%fill211 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty213 = tensor.empty() : tensor<1x512x7x7xf16>
  %add214 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv209, %conv212 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty213 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty215 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu216 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add214 : tensor<1x512x7x7xf16>)
    outs(%empty215 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad217 = tensor.pad %relu216 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init218 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill219 = linalg.fill ins(%cst : f16) outs(%init218 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv220 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad217, %w_s3_b1_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill219 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty221 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu222 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv220 : tensor<1x512x7x7xf16>)
    outs(%empty221 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad223 = tensor.pad %relu222 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init224 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill225 = linalg.fill ins(%cst : f16) outs(%init224 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv226 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad223, %w_s3_b1_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill225 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty227 = tensor.empty() : tensor<1x512x7x7xf16>
  %add228 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv226, %relu216 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty227 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty229 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu230 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add228 : tensor<1x512x7x7xf16>)
    outs(%empty229 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad231 = tensor.pad %relu230 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init232 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill233 = linalg.fill ins(%cst : f16) outs(%init232 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv234 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad231, %w_s3_b2_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill233 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty235 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu236 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv234 : tensor<1x512x7x7xf16>)
    outs(%empty235 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad237 = tensor.pad %relu236 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init238 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill239 = linalg.fill ins(%cst : f16) outs(%init238 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv240 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad237, %w_s3_b2_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill239 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty241 = tensor.empty() : tensor<1x512x7x7xf16>
  %add242 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv240, %relu230 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty241 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty243 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu244 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add242 : tensor<1x512x7x7xf16>)
    outs(%empty243 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // FC as 7x7 conv: 512->1000
  %fc_init245 = tensor.empty() : tensor<1x1000x1x1xf16>
  %fc_fill246 = linalg.fill ins(%cst : f16) outs(%fc_init245 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  %fc247 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu244, %w_fc : tensor<1x512x7x7xf16>, tensor<1000x512x7x7xf16>)
    outs(%fc_fill246 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  return %fc247 : tensor<1x1000x1x1xf16>
}
