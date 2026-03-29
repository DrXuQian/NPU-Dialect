func.func @resnet_basic_2_2_2_2(
    %input: tensor<1x3x224x224xf16>,
    %w_conv1: tensor<64x3x7x7xf16>,
    %w_pool: tensor<64x64x3x3xf16>,
    %w_s0_b0_c1: tensor<64x64x3x3xf16>,
    %w_s0_b0_c2: tensor<64x64x3x3xf16>,
    %w_s0_b1_c1: tensor<64x64x3x3xf16>,
    %w_s0_b1_c2: tensor<64x64x3x3xf16>,
    %w_s1_b0_c1: tensor<128x64x3x3xf16>,
    %w_s1_b0_c2: tensor<128x128x3x3xf16>,
    %w_s1_b0_sc: tensor<128x64x1x1xf16>,
    %w_s1_b1_c1: tensor<128x128x3x3xf16>,
    %w_s1_b1_c2: tensor<128x128x3x3xf16>,
    %w_s2_b0_c1: tensor<256x128x3x3xf16>,
    %w_s2_b0_c2: tensor<256x256x3x3xf16>,
    %w_s2_b0_sc: tensor<256x128x1x1xf16>,
    %w_s2_b1_c1: tensor<256x256x3x3xf16>,
    %w_s2_b1_c2: tensor<256x256x3x3xf16>,
    %w_s3_b0_c1: tensor<512x256x3x3xf16>,
    %w_s3_b0_c2: tensor<512x512x3x3xf16>,
    %w_s3_b0_sc: tensor<512x256x1x1xf16>,
    %w_s3_b1_c1: tensor<512x512x3x3xf16>,
    %w_s3_b1_c2: tensor<512x512x3x3xf16>,
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

  // === Stage 1 ===
  // BasicBlock 64->128 56x56 stride=2
  %pad40 = tensor.pad %relu39 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init41 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill42 = linalg.fill ins(%cst : f16) outs(%init41 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv43 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad40, %w_s1_b0_c1 : tensor<1x64x58x58xf16>, tensor<128x64x3x3xf16>)
    outs(%fill42 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty44 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu45 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv43 : tensor<1x128x28x28xf16>)
    outs(%empty44 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad46 = tensor.pad %relu45 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init47 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill48 = linalg.fill ins(%cst : f16) outs(%init47 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv49 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad46, %w_s1_b0_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill48 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  // 1x1 shortcut conv
  %init50 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill51 = linalg.fill ins(%cst : f16) outs(%init50 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv52 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu39, %w_s1_b0_sc : tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>)
    outs(%fill51 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty53 = tensor.empty() : tensor<1x128x28x28xf16>
  %add54 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv49, %conv52 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
    outs(%empty53 : tensor<1x128x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x28x28xf16>
  %empty55 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu56 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add54 : tensor<1x128x28x28xf16>)
    outs(%empty55 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>

  // BasicBlock 128->128 28x28 stride=1
  %pad57 = tensor.pad %relu56 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init58 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill59 = linalg.fill ins(%cst : f16) outs(%init58 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv60 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad57, %w_s1_b1_c1 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill59 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty61 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu62 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv60 : tensor<1x128x28x28xf16>)
    outs(%empty61 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad63 = tensor.pad %relu62 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init64 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill65 = linalg.fill ins(%cst : f16) outs(%init64 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv66 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad63, %w_s1_b1_c2 : tensor<1x128x30x30xf16>, tensor<128x128x3x3xf16>)
    outs(%fill65 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty67 = tensor.empty() : tensor<1x128x28x28xf16>
  %add68 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv66, %relu56 : tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>)
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

  // === Stage 2 ===
  // BasicBlock 128->256 28x28 stride=2
  %pad71 = tensor.pad %relu70 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init72 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill73 = linalg.fill ins(%cst : f16) outs(%init72 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv74 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad71, %w_s2_b0_c1 : tensor<1x128x30x30xf16>, tensor<256x128x3x3xf16>)
    outs(%fill73 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty75 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu76 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv74 : tensor<1x256x14x14xf16>)
    outs(%empty75 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad77 = tensor.pad %relu76 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init78 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill79 = linalg.fill ins(%cst : f16) outs(%init78 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv80 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad77, %w_s2_b0_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill79 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  // 1x1 shortcut conv
  %init81 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill82 = linalg.fill ins(%cst : f16) outs(%init81 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv83 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu70, %w_s2_b0_sc : tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>)
    outs(%fill82 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty84 = tensor.empty() : tensor<1x256x14x14xf16>
  %add85 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv80, %conv83 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty84 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty86 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu87 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add85 : tensor<1x256x14x14xf16>)
    outs(%empty86 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // BasicBlock 256->256 14x14 stride=1
  %pad88 = tensor.pad %relu87 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init89 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill90 = linalg.fill ins(%cst : f16) outs(%init89 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv91 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad88, %w_s2_b1_c1 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill90 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty92 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu93 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv91 : tensor<1x256x14x14xf16>)
    outs(%empty92 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %pad94 = tensor.pad %relu93 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init95 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill96 = linalg.fill ins(%cst : f16) outs(%init95 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv97 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad94, %w_s2_b1_c2 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill96 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty98 = tensor.empty() : tensor<1x256x14x14xf16>
  %add99 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv97, %relu87 : tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>)
    outs(%empty98 : tensor<1x256x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x14x14xf16>
  %empty100 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu101 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add99 : tensor<1x256x14x14xf16>)
    outs(%empty100 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // === Stage 3 ===
  // BasicBlock 256->512 14x14 stride=2
  %pad102 = tensor.pad %relu101 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init103 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill104 = linalg.fill ins(%cst : f16) outs(%init103 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv105 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad102, %w_s3_b0_c1 : tensor<1x256x16x16xf16>, tensor<512x256x3x3xf16>)
    outs(%fill104 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty106 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu107 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv105 : tensor<1x512x7x7xf16>)
    outs(%empty106 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad108 = tensor.pad %relu107 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init109 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill110 = linalg.fill ins(%cst : f16) outs(%init109 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv111 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad108, %w_s3_b0_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill110 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  // 1x1 shortcut conv
  %init112 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill113 = linalg.fill ins(%cst : f16) outs(%init112 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv114 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%relu101, %w_s3_b0_sc : tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>)
    outs(%fill113 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty115 = tensor.empty() : tensor<1x512x7x7xf16>
  %add116 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv111, %conv114 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty115 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty117 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu118 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add116 : tensor<1x512x7x7xf16>)
    outs(%empty117 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // BasicBlock 512->512 7x7 stride=1
  %pad119 = tensor.pad %relu118 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init120 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill121 = linalg.fill ins(%cst : f16) outs(%init120 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv122 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad119, %w_s3_b1_c1 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill121 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty123 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu124 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv122 : tensor<1x512x7x7xf16>)
    outs(%empty123 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>
  %pad125 = tensor.pad %relu124 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x7x7xf16> to tensor<1x512x9x9xf16>
  %init126 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill127 = linalg.fill ins(%cst : f16) outs(%init126 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv128 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad125, %w_s3_b1_c2 : tensor<1x512x9x9xf16>, tensor<512x512x3x3xf16>)
    outs(%fill127 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %empty129 = tensor.empty() : tensor<1x512x7x7xf16>
  %add130 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv128, %relu118 : tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>)
    outs(%empty129 : tensor<1x512x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x7x7xf16>
  %empty131 = tensor.empty() : tensor<1x512x7x7xf16>
  %relu132 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add130 : tensor<1x512x7x7xf16>)
    outs(%empty131 : tensor<1x512x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x7x7xf16>

  // FC as 7x7 conv: 512->1000
  %fc_init133 = tensor.empty() : tensor<1x1000x1x1xf16>
  %fc_fill134 = linalg.fill ins(%cst : f16) outs(%fc_init133 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  %fc135 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu132, %w_fc : tensor<1x512x7x7xf16>, tensor<1000x512x7x7xf16>)
    outs(%fc_fill134 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  return %fc135 : tensor<1x1000x1x1xf16>
}
