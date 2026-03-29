func.func @mobilenetv1(
    %input: tensor<1x3x224x224xf16>,
    %w_stem: tensor<32x3x3x3xf16>,
    %w_blk0_dw: tensor<32x32x3x3xf16>,
    %w_blk0_proj: tensor<64x32x1x1xf16>,
    %w_blk1_dw: tensor<64x64x3x3xf16>,
    %w_blk1_proj: tensor<128x64x1x1xf16>,
    %w_blk2_dw: tensor<128x128x3x3xf16>,
    %w_blk2_proj: tensor<128x128x1x1xf16>,
    %w_blk3_dw: tensor<128x128x3x3xf16>,
    %w_blk3_proj: tensor<256x128x1x1xf16>,
    %w_blk4_dw: tensor<256x256x3x3xf16>,
    %w_blk4_proj: tensor<256x256x1x1xf16>,
    %w_blk5_dw: tensor<256x256x3x3xf16>,
    %w_blk5_proj: tensor<512x256x1x1xf16>,
    %w_blk6_dw: tensor<512x512x3x3xf16>,
    %w_blk6_proj: tensor<512x512x1x1xf16>,
    %w_blk7_dw: tensor<512x512x3x3xf16>,
    %w_blk7_proj: tensor<512x512x1x1xf16>,
    %w_blk8_dw: tensor<512x512x3x3xf16>,
    %w_blk8_proj: tensor<512x512x1x1xf16>,
    %w_blk9_dw: tensor<512x512x3x3xf16>,
    %w_blk9_proj: tensor<512x512x1x1xf16>,
    %w_blk10_dw: tensor<512x512x3x3xf16>,
    %w_blk10_proj: tensor<512x512x1x1xf16>,
    %w_blk11_dw: tensor<512x512x3x3xf16>,
    %w_blk11_proj: tensor<1024x512x1x1xf16>,
    %w_blk12_dw: tensor<1024x1024x3x3xf16>,
    %w_blk12_proj: tensor<1024x1024x1x1xf16>,
    %w_final: tensor<1280x1024x1x1xf16>,
    %w_fc: tensor<1000x1280x1x1xf16>) -> tensor<1x1000x7x7xf16> {
  %cst = arith.constant 0.0 : f16

  // Stem: 3x3 stride 2, 3->32
  %pad0 = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x224x224xf16> to tensor<1x3x226x226xf16>
  %init1 = tensor.empty() : tensor<1x32x112x112xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad0, %w_stem : tensor<1x3x226x226xf16>, tensor<32x3x3x3xf16>)
    outs(%fill2 : tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16>
  %empty4 = tensor.empty() : tensor<1x32x112x112xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x32x112x112xf16>)
    outs(%empty4 : tensor<1x32x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x112x112xf16>

  // IRB 0: 32->64 mid=32 s=1 112x112
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x32x112x112xf16> to tensor<1x32x114x114xf16>
  %init7 = tensor.empty() : tensor<1x32x112x112xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad6, %w_blk0_dw : tensor<1x32x114x114xf16>, tensor<32x32x3x3xf16>)
    outs(%fill8 : tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16>
  %empty10 = tensor.empty() : tensor<1x32x112x112xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x32x112x112xf16>)
    outs(%empty10 : tensor<1x32x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x112x112xf16>
  %init12 = tensor.empty() : tensor<1x64x112x112xf16>
  %fill13 = linalg.fill ins(%cst : f16) outs(%init12 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %conv14 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu11, %w_blk0_proj : tensor<1x32x112x112xf16>, tensor<64x32x1x1xf16>)
    outs(%fill13 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>

  // IRB 1: 64->128 mid=64 s=2 112x112
  %pad15 = tensor.pad %conv14 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x112x112xf16> to tensor<1x64x114x114xf16>
  %init16 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill17 = linalg.fill ins(%cst : f16) outs(%init16 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv18 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad15, %w_blk1_dw : tensor<1x64x114x114xf16>, tensor<64x64x3x3xf16>)
    outs(%fill17 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty19 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu20 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv18 : tensor<1x64x56x56xf16>)
    outs(%empty19 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>
  %init21 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill22 = linalg.fill ins(%cst : f16) outs(%init21 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv23 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu20, %w_blk1_proj : tensor<1x64x56x56xf16>, tensor<128x64x1x1xf16>)
    outs(%fill22 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>

  // IRB 2: 128->128 mid=128 s=1 56x56
  %pad24 = tensor.pad %conv23 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x56x56xf16> to tensor<1x128x58x58xf16>
  %init25 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill26 = linalg.fill ins(%cst : f16) outs(%init25 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv27 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad24, %w_blk2_dw : tensor<1x128x58x58xf16>, tensor<128x128x3x3xf16>)
    outs(%fill26 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %empty28 = tensor.empty() : tensor<1x128x56x56xf16>
  %relu29 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv27 : tensor<1x128x56x56xf16>)
    outs(%empty28 : tensor<1x128x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x56x56xf16>
  %init30 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill31 = linalg.fill ins(%cst : f16) outs(%init30 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv32 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu29, %w_blk2_proj : tensor<1x128x56x56xf16>, tensor<128x128x1x1xf16>)
    outs(%fill31 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %empty33 = tensor.empty() : tensor<1x128x56x56xf16>
  %add34 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv32, %conv23 : tensor<1x128x56x56xf16>, tensor<1x128x56x56xf16>)
    outs(%empty33 : tensor<1x128x56x56xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x128x56x56xf16>

  // IRB 3: 128->256 mid=128 s=2 56x56
  %pad35 = tensor.pad %add34 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x56x56xf16> to tensor<1x128x58x58xf16>
  %init36 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill37 = linalg.fill ins(%cst : f16) outs(%init36 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv38 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad35, %w_blk3_dw : tensor<1x128x58x58xf16>, tensor<128x128x3x3xf16>)
    outs(%fill37 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty39 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu40 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv38 : tensor<1x128x28x28xf16>)
    outs(%empty39 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %init41 = tensor.empty() : tensor<1x256x28x28xf16>
  %fill42 = linalg.fill ins(%cst : f16) outs(%init41 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %conv43 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu40, %w_blk3_proj : tensor<1x128x28x28xf16>, tensor<256x128x1x1xf16>)
    outs(%fill42 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>

  // IRB 4: 256->256 mid=256 s=1 28x28
  %pad44 = tensor.pad %conv43 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x28x28xf16> to tensor<1x256x30x30xf16>
  %init45 = tensor.empty() : tensor<1x256x28x28xf16>
  %fill46 = linalg.fill ins(%cst : f16) outs(%init45 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %conv47 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad44, %w_blk4_dw : tensor<1x256x30x30xf16>, tensor<256x256x3x3xf16>)
    outs(%fill46 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %empty48 = tensor.empty() : tensor<1x256x28x28xf16>
  %relu49 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv47 : tensor<1x256x28x28xf16>)
    outs(%empty48 : tensor<1x256x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x28x28xf16>
  %init50 = tensor.empty() : tensor<1x256x28x28xf16>
  %fill51 = linalg.fill ins(%cst : f16) outs(%init50 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %conv52 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu49, %w_blk4_proj : tensor<1x256x28x28xf16>, tensor<256x256x1x1xf16>)
    outs(%fill51 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %empty53 = tensor.empty() : tensor<1x256x28x28xf16>
  %add54 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv52, %conv43 : tensor<1x256x28x28xf16>, tensor<1x256x28x28xf16>)
    outs(%empty53 : tensor<1x256x28x28xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x256x28x28xf16>

  // IRB 5: 256->512 mid=256 s=2 28x28
  %pad55 = tensor.pad %add54 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x28x28xf16> to tensor<1x256x30x30xf16>
  %init56 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill57 = linalg.fill ins(%cst : f16) outs(%init56 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv58 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad55, %w_blk5_dw : tensor<1x256x30x30xf16>, tensor<256x256x3x3xf16>)
    outs(%fill57 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty59 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu60 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv58 : tensor<1x256x14x14xf16>)
    outs(%empty59 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>
  %init61 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill62 = linalg.fill ins(%cst : f16) outs(%init61 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv63 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu60, %w_blk5_proj : tensor<1x256x14x14xf16>, tensor<512x256x1x1xf16>)
    outs(%fill62 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>

  // IRB 6: 512->512 mid=512 s=1 14x14
  %pad64 = tensor.pad %conv63 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x16x16xf16>
  %init65 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill66 = linalg.fill ins(%cst : f16) outs(%init65 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv67 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad64, %w_blk6_dw : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>)
    outs(%fill66 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty68 = tensor.empty() : tensor<1x512x14x14xf16>
  %relu69 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv67 : tensor<1x512x14x14xf16>)
    outs(%empty68 : tensor<1x512x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x14x14xf16>
  %init70 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill71 = linalg.fill ins(%cst : f16) outs(%init70 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv72 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu69, %w_blk6_proj : tensor<1x512x14x14xf16>, tensor<512x512x1x1xf16>)
    outs(%fill71 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty73 = tensor.empty() : tensor<1x512x14x14xf16>
  %add74 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv72, %conv63 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty73 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>

  // IRB 7: 512->512 mid=512 s=1 14x14
  %pad75 = tensor.pad %add74 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x16x16xf16>
  %init76 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill77 = linalg.fill ins(%cst : f16) outs(%init76 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv78 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad75, %w_blk7_dw : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>)
    outs(%fill77 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty79 = tensor.empty() : tensor<1x512x14x14xf16>
  %relu80 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv78 : tensor<1x512x14x14xf16>)
    outs(%empty79 : tensor<1x512x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x14x14xf16>
  %init81 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill82 = linalg.fill ins(%cst : f16) outs(%init81 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv83 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu80, %w_blk7_proj : tensor<1x512x14x14xf16>, tensor<512x512x1x1xf16>)
    outs(%fill82 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty84 = tensor.empty() : tensor<1x512x14x14xf16>
  %add85 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv83, %add74 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty84 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>

  // IRB 8: 512->512 mid=512 s=1 14x14
  %pad86 = tensor.pad %add85 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x16x16xf16>
  %init87 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill88 = linalg.fill ins(%cst : f16) outs(%init87 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv89 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad86, %w_blk8_dw : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>)
    outs(%fill88 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty90 = tensor.empty() : tensor<1x512x14x14xf16>
  %relu91 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv89 : tensor<1x512x14x14xf16>)
    outs(%empty90 : tensor<1x512x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x14x14xf16>
  %init92 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill93 = linalg.fill ins(%cst : f16) outs(%init92 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv94 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu91, %w_blk8_proj : tensor<1x512x14x14xf16>, tensor<512x512x1x1xf16>)
    outs(%fill93 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty95 = tensor.empty() : tensor<1x512x14x14xf16>
  %add96 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv94, %add85 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty95 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>

  // IRB 9: 512->512 mid=512 s=1 14x14
  %pad97 = tensor.pad %add96 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x16x16xf16>
  %init98 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill99 = linalg.fill ins(%cst : f16) outs(%init98 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv100 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad97, %w_blk9_dw : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>)
    outs(%fill99 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty101 = tensor.empty() : tensor<1x512x14x14xf16>
  %relu102 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv100 : tensor<1x512x14x14xf16>)
    outs(%empty101 : tensor<1x512x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x14x14xf16>
  %init103 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill104 = linalg.fill ins(%cst : f16) outs(%init103 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv105 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu102, %w_blk9_proj : tensor<1x512x14x14xf16>, tensor<512x512x1x1xf16>)
    outs(%fill104 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty106 = tensor.empty() : tensor<1x512x14x14xf16>
  %add107 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv105, %add96 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty106 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>

  // IRB 10: 512->512 mid=512 s=1 14x14
  %pad108 = tensor.pad %add107 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x16x16xf16>
  %init109 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill110 = linalg.fill ins(%cst : f16) outs(%init109 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv111 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad108, %w_blk10_dw : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>)
    outs(%fill110 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty112 = tensor.empty() : tensor<1x512x14x14xf16>
  %relu113 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv111 : tensor<1x512x14x14xf16>)
    outs(%empty112 : tensor<1x512x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x14x14xf16>
  %init114 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill115 = linalg.fill ins(%cst : f16) outs(%init114 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv116 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu113, %w_blk10_proj : tensor<1x512x14x14xf16>, tensor<512x512x1x1xf16>)
    outs(%fill115 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty117 = tensor.empty() : tensor<1x512x14x14xf16>
  %add118 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv116, %add107 : tensor<1x512x14x14xf16>, tensor<1x512x14x14xf16>)
    outs(%empty117 : tensor<1x512x14x14xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x512x14x14xf16>

  // IRB 11: 512->1024 mid=512 s=2 14x14
  %pad119 = tensor.pad %add118 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x16x16xf16>
  %init120 = tensor.empty() : tensor<1x512x7x7xf16>
  %fill121 = linalg.fill ins(%cst : f16) outs(%init120 : tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
  %conv122 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad119, %w_blk11_dw : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>)
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
  %init125 = tensor.empty() : tensor<1x1024x7x7xf16>
  %fill126 = linalg.fill ins(%cst : f16) outs(%init125 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %conv127 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu124, %w_blk11_proj : tensor<1x512x7x7xf16>, tensor<1024x512x1x1xf16>)
    outs(%fill126 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>

  // IRB 12: 1024->1024 mid=1024 s=1 7x7
  %pad128 = tensor.pad %conv127 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1024x7x7xf16> to tensor<1x1024x9x9xf16>
  %init129 = tensor.empty() : tensor<1x1024x7x7xf16>
  %fill130 = linalg.fill ins(%cst : f16) outs(%init129 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %conv131 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad128, %w_blk12_dw : tensor<1x1024x9x9xf16>, tensor<1024x1024x3x3xf16>)
    outs(%fill130 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %empty132 = tensor.empty() : tensor<1x1024x7x7xf16>
  %relu133 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv131 : tensor<1x1024x7x7xf16>)
    outs(%empty132 : tensor<1x1024x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x7x7xf16>
  %init134 = tensor.empty() : tensor<1x1024x7x7xf16>
  %fill135 = linalg.fill ins(%cst : f16) outs(%init134 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %conv136 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu133, %w_blk12_proj : tensor<1x1024x7x7xf16>, tensor<1024x1024x1x1xf16>)
    outs(%fill135 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %empty137 = tensor.empty() : tensor<1x1024x7x7xf16>
  %add138 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv136, %conv127 : tensor<1x1024x7x7xf16>, tensor<1x1024x7x7xf16>)
    outs(%empty137 : tensor<1x1024x7x7xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1x1024x7x7xf16>

  // Final 1x1: 1024->1280
  %init139 = tensor.empty() : tensor<1x1280x7x7xf16>
  %fill140 = linalg.fill ins(%cst : f16) outs(%init139 : tensor<1x1280x7x7xf16>) -> tensor<1x1280x7x7xf16>
  %conv141 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%add138, %w_final : tensor<1x1024x7x7xf16>, tensor<1280x1024x1x1xf16>)
    outs(%fill140 : tensor<1x1280x7x7xf16>) -> tensor<1x1280x7x7xf16>
  %empty142 = tensor.empty() : tensor<1x1280x7x7xf16>
  %relu143 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv141 : tensor<1x1280x7x7xf16>)
    outs(%empty142 : tensor<1x1280x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1280x7x7xf16>

  // FC: 1x1 1280->1000
  %init144 = tensor.empty() : tensor<1x1000x7x7xf16>
  %fill145 = linalg.fill ins(%cst : f16) outs(%init144 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  %conv146 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu143, %w_fc : tensor<1x1280x7x7xf16>, tensor<1000x1280x1x1xf16>)
    outs(%fill145 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  return %conv146 : tensor<1x1000x7x7xf16>
}
