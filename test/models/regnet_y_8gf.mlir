func.func @regnet_y_8gf(
    %input: tensor<1x3x224x224xf16>,
    %w0: tensor<32x3x3x3xf16>,
    %w1: tensor<80x32x3x3xf16>,
    %w2: tensor<80x80x3x3xf16>,
    %w3: tensor<240x80x3x3xf16>,
    %w4: tensor<240x240x3x3xf16>,
    %w5: tensor<240x240x3x3xf16>,
    %w6: tensor<240x240x3x3xf16>,
    %w7: tensor<240x240x3x3xf16>,
    %w8: tensor<240x240x3x3xf16>,
    %w9: tensor<240x240x3x3xf16>,
    %w10: tensor<560x240x3x3xf16>,
    %w11: tensor<560x560x3x3xf16>,
    %w12: tensor<560x560x3x3xf16>,
    %w13: tensor<560x560x3x3xf16>,
    %w14: tensor<560x560x3x3xf16>,
    %w15: tensor<560x560x3x3xf16>,
    %w16: tensor<560x560x3x3xf16>,
    %w17: tensor<560x560x3x3xf16>,
    %w18: tensor<560x560x3x3xf16>,
    %w19: tensor<560x560x3x3xf16>,
    %w20: tensor<560x560x3x3xf16>,
    %w21: tensor<560x560x3x3xf16>,
    %w22: tensor<560x560x3x3xf16>,
    %w23: tensor<560x560x3x3xf16>,
    %w24: tensor<560x560x3x3xf16>,
    %w25: tensor<560x560x3x3xf16>,
    %w26: tensor<560x560x3x3xf16>,
    %w27: tensor<1360x560x3x3xf16>,
    %w_fc: tensor<1000x1360x1x1xf16>) -> tensor<1x1000x7x7xf16> {
  %cst = arith.constant 0.0 : f16

  // conv0: 3x3 s2 p1 3->32 224x224
  %pad0 = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x224x224xf16> to tensor<1x3x226x226xf16>
  %init1 = tensor.empty() : tensor<1x32x112x112xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad0, %w0 : tensor<1x3x226x226xf16>, tensor<32x3x3x3xf16>)
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

  // conv1: 3x3 s2 p1 32->80 112x112
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x32x112x112xf16> to tensor<1x32x114x114xf16>
  %init7 = tensor.empty() : tensor<1x80x56x56xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x80x56x56xf16>) -> tensor<1x80x56x56xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad6, %w1 : tensor<1x32x114x114xf16>, tensor<80x32x3x3xf16>)
    outs(%fill8 : tensor<1x80x56x56xf16>) -> tensor<1x80x56x56xf16>
  %empty10 = tensor.empty() : tensor<1x80x56x56xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x80x56x56xf16>)
    outs(%empty10 : tensor<1x80x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x80x56x56xf16>

  // conv2: 3x3 s1 p1 80->80 56x56
  %pad12 = tensor.pad %relu11 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x80x56x56xf16> to tensor<1x80x58x58xf16>
  %init13 = tensor.empty() : tensor<1x80x56x56xf16>
  %fill14 = linalg.fill ins(%cst : f16) outs(%init13 : tensor<1x80x56x56xf16>) -> tensor<1x80x56x56xf16>
  %conv15 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad12, %w2 : tensor<1x80x58x58xf16>, tensor<80x80x3x3xf16>)
    outs(%fill14 : tensor<1x80x56x56xf16>) -> tensor<1x80x56x56xf16>
  %empty16 = tensor.empty() : tensor<1x80x56x56xf16>
  %relu17 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv15 : tensor<1x80x56x56xf16>)
    outs(%empty16 : tensor<1x80x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x80x56x56xf16>

  // conv3: 3x3 s2 p1 80->240 56x56
  %pad18 = tensor.pad %relu17 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x80x56x56xf16> to tensor<1x80x58x58xf16>
  %init19 = tensor.empty() : tensor<1x240x28x28xf16>
  %fill20 = linalg.fill ins(%cst : f16) outs(%init19 : tensor<1x240x28x28xf16>) -> tensor<1x240x28x28xf16>
  %conv21 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad18, %w3 : tensor<1x80x58x58xf16>, tensor<240x80x3x3xf16>)
    outs(%fill20 : tensor<1x240x28x28xf16>) -> tensor<1x240x28x28xf16>
  %empty22 = tensor.empty() : tensor<1x240x28x28xf16>
  %relu23 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv21 : tensor<1x240x28x28xf16>)
    outs(%empty22 : tensor<1x240x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x240x28x28xf16>

  // conv4: 3x3 s1 p1 240->240 28x28
  %pad24 = tensor.pad %relu23 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x240x28x28xf16> to tensor<1x240x30x30xf16>
  %init25 = tensor.empty() : tensor<1x240x28x28xf16>
  %fill26 = linalg.fill ins(%cst : f16) outs(%init25 : tensor<1x240x28x28xf16>) -> tensor<1x240x28x28xf16>
  %conv27 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad24, %w4 : tensor<1x240x30x30xf16>, tensor<240x240x3x3xf16>)
    outs(%fill26 : tensor<1x240x28x28xf16>) -> tensor<1x240x28x28xf16>
  %empty28 = tensor.empty() : tensor<1x240x28x28xf16>
  %relu29 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv27 : tensor<1x240x28x28xf16>)
    outs(%empty28 : tensor<1x240x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x240x28x28xf16>

  // conv5: 3x3 s1 p1 240->240 28x28
  %pad30 = tensor.pad %relu29 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x240x28x28xf16> to tensor<1x240x30x30xf16>
  %init31 = tensor.empty() : tensor<1x240x28x28xf16>
  %fill32 = linalg.fill ins(%cst : f16) outs(%init31 : tensor<1x240x28x28xf16>) -> tensor<1x240x28x28xf16>
  %conv33 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad30, %w5 : tensor<1x240x30x30xf16>, tensor<240x240x3x3xf16>)
    outs(%fill32 : tensor<1x240x28x28xf16>) -> tensor<1x240x28x28xf16>
  %empty34 = tensor.empty() : tensor<1x240x28x28xf16>
  %relu35 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv33 : tensor<1x240x28x28xf16>)
    outs(%empty34 : tensor<1x240x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x240x28x28xf16>

  // conv6: 3x3 s1 p1 240->240 28x28
  %pad36 = tensor.pad %relu35 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x240x28x28xf16> to tensor<1x240x30x30xf16>
  %init37 = tensor.empty() : tensor<1x240x28x28xf16>
  %fill38 = linalg.fill ins(%cst : f16) outs(%init37 : tensor<1x240x28x28xf16>) -> tensor<1x240x28x28xf16>
  %conv39 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad36, %w6 : tensor<1x240x30x30xf16>, tensor<240x240x3x3xf16>)
    outs(%fill38 : tensor<1x240x28x28xf16>) -> tensor<1x240x28x28xf16>
  %empty40 = tensor.empty() : tensor<1x240x28x28xf16>
  %relu41 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv39 : tensor<1x240x28x28xf16>)
    outs(%empty40 : tensor<1x240x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x240x28x28xf16>

  // conv7: 3x3 s1 p1 240->240 28x28
  %pad42 = tensor.pad %relu41 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x240x28x28xf16> to tensor<1x240x30x30xf16>
  %init43 = tensor.empty() : tensor<1x240x28x28xf16>
  %fill44 = linalg.fill ins(%cst : f16) outs(%init43 : tensor<1x240x28x28xf16>) -> tensor<1x240x28x28xf16>
  %conv45 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad42, %w7 : tensor<1x240x30x30xf16>, tensor<240x240x3x3xf16>)
    outs(%fill44 : tensor<1x240x28x28xf16>) -> tensor<1x240x28x28xf16>
  %empty46 = tensor.empty() : tensor<1x240x28x28xf16>
  %relu47 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv45 : tensor<1x240x28x28xf16>)
    outs(%empty46 : tensor<1x240x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x240x28x28xf16>

  // conv8: 3x3 s1 p1 240->240 28x28
  %pad48 = tensor.pad %relu47 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x240x28x28xf16> to tensor<1x240x30x30xf16>
  %init49 = tensor.empty() : tensor<1x240x28x28xf16>
  %fill50 = linalg.fill ins(%cst : f16) outs(%init49 : tensor<1x240x28x28xf16>) -> tensor<1x240x28x28xf16>
  %conv51 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad48, %w8 : tensor<1x240x30x30xf16>, tensor<240x240x3x3xf16>)
    outs(%fill50 : tensor<1x240x28x28xf16>) -> tensor<1x240x28x28xf16>
  %empty52 = tensor.empty() : tensor<1x240x28x28xf16>
  %relu53 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv51 : tensor<1x240x28x28xf16>)
    outs(%empty52 : tensor<1x240x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x240x28x28xf16>

  // conv9: 3x3 s1 p1 240->240 28x28
  %pad54 = tensor.pad %relu53 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x240x28x28xf16> to tensor<1x240x30x30xf16>
  %init55 = tensor.empty() : tensor<1x240x28x28xf16>
  %fill56 = linalg.fill ins(%cst : f16) outs(%init55 : tensor<1x240x28x28xf16>) -> tensor<1x240x28x28xf16>
  %conv57 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad54, %w9 : tensor<1x240x30x30xf16>, tensor<240x240x3x3xf16>)
    outs(%fill56 : tensor<1x240x28x28xf16>) -> tensor<1x240x28x28xf16>
  %empty58 = tensor.empty() : tensor<1x240x28x28xf16>
  %relu59 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv57 : tensor<1x240x28x28xf16>)
    outs(%empty58 : tensor<1x240x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x240x28x28xf16>

  // conv10: 3x3 s2 p1 240->560 28x28
  %pad60 = tensor.pad %relu59 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x240x28x28xf16> to tensor<1x240x30x30xf16>
  %init61 = tensor.empty() : tensor<1x560x14x14xf16>
  %fill62 = linalg.fill ins(%cst : f16) outs(%init61 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %conv63 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad60, %w10 : tensor<1x240x30x30xf16>, tensor<560x240x3x3xf16>)
    outs(%fill62 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %empty64 = tensor.empty() : tensor<1x560x14x14xf16>
  %relu65 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv63 : tensor<1x560x14x14xf16>)
    outs(%empty64 : tensor<1x560x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x560x14x14xf16>

  // conv11: 3x3 s1 p1 560->560 14x14
  %pad66 = tensor.pad %relu65 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x560x14x14xf16> to tensor<1x560x16x16xf16>
  %init67 = tensor.empty() : tensor<1x560x14x14xf16>
  %fill68 = linalg.fill ins(%cst : f16) outs(%init67 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %conv69 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad66, %w11 : tensor<1x560x16x16xf16>, tensor<560x560x3x3xf16>)
    outs(%fill68 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %empty70 = tensor.empty() : tensor<1x560x14x14xf16>
  %relu71 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv69 : tensor<1x560x14x14xf16>)
    outs(%empty70 : tensor<1x560x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x560x14x14xf16>

  // conv12: 3x3 s1 p1 560->560 14x14
  %pad72 = tensor.pad %relu71 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x560x14x14xf16> to tensor<1x560x16x16xf16>
  %init73 = tensor.empty() : tensor<1x560x14x14xf16>
  %fill74 = linalg.fill ins(%cst : f16) outs(%init73 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %conv75 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad72, %w12 : tensor<1x560x16x16xf16>, tensor<560x560x3x3xf16>)
    outs(%fill74 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %empty76 = tensor.empty() : tensor<1x560x14x14xf16>
  %relu77 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv75 : tensor<1x560x14x14xf16>)
    outs(%empty76 : tensor<1x560x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x560x14x14xf16>

  // conv13: 3x3 s1 p1 560->560 14x14
  %pad78 = tensor.pad %relu77 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x560x14x14xf16> to tensor<1x560x16x16xf16>
  %init79 = tensor.empty() : tensor<1x560x14x14xf16>
  %fill80 = linalg.fill ins(%cst : f16) outs(%init79 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %conv81 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad78, %w13 : tensor<1x560x16x16xf16>, tensor<560x560x3x3xf16>)
    outs(%fill80 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %empty82 = tensor.empty() : tensor<1x560x14x14xf16>
  %relu83 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv81 : tensor<1x560x14x14xf16>)
    outs(%empty82 : tensor<1x560x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x560x14x14xf16>

  // conv14: 3x3 s1 p1 560->560 14x14
  %pad84 = tensor.pad %relu83 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x560x14x14xf16> to tensor<1x560x16x16xf16>
  %init85 = tensor.empty() : tensor<1x560x14x14xf16>
  %fill86 = linalg.fill ins(%cst : f16) outs(%init85 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %conv87 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad84, %w14 : tensor<1x560x16x16xf16>, tensor<560x560x3x3xf16>)
    outs(%fill86 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %empty88 = tensor.empty() : tensor<1x560x14x14xf16>
  %relu89 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv87 : tensor<1x560x14x14xf16>)
    outs(%empty88 : tensor<1x560x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x560x14x14xf16>

  // conv15: 3x3 s1 p1 560->560 14x14
  %pad90 = tensor.pad %relu89 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x560x14x14xf16> to tensor<1x560x16x16xf16>
  %init91 = tensor.empty() : tensor<1x560x14x14xf16>
  %fill92 = linalg.fill ins(%cst : f16) outs(%init91 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %conv93 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad90, %w15 : tensor<1x560x16x16xf16>, tensor<560x560x3x3xf16>)
    outs(%fill92 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %empty94 = tensor.empty() : tensor<1x560x14x14xf16>
  %relu95 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv93 : tensor<1x560x14x14xf16>)
    outs(%empty94 : tensor<1x560x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x560x14x14xf16>

  // conv16: 3x3 s1 p1 560->560 14x14
  %pad96 = tensor.pad %relu95 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x560x14x14xf16> to tensor<1x560x16x16xf16>
  %init97 = tensor.empty() : tensor<1x560x14x14xf16>
  %fill98 = linalg.fill ins(%cst : f16) outs(%init97 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %conv99 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad96, %w16 : tensor<1x560x16x16xf16>, tensor<560x560x3x3xf16>)
    outs(%fill98 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %empty100 = tensor.empty() : tensor<1x560x14x14xf16>
  %relu101 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv99 : tensor<1x560x14x14xf16>)
    outs(%empty100 : tensor<1x560x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x560x14x14xf16>

  // conv17: 3x3 s1 p1 560->560 14x14
  %pad102 = tensor.pad %relu101 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x560x14x14xf16> to tensor<1x560x16x16xf16>
  %init103 = tensor.empty() : tensor<1x560x14x14xf16>
  %fill104 = linalg.fill ins(%cst : f16) outs(%init103 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %conv105 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad102, %w17 : tensor<1x560x16x16xf16>, tensor<560x560x3x3xf16>)
    outs(%fill104 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %empty106 = tensor.empty() : tensor<1x560x14x14xf16>
  %relu107 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv105 : tensor<1x560x14x14xf16>)
    outs(%empty106 : tensor<1x560x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x560x14x14xf16>

  // conv18: 3x3 s1 p1 560->560 14x14
  %pad108 = tensor.pad %relu107 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x560x14x14xf16> to tensor<1x560x16x16xf16>
  %init109 = tensor.empty() : tensor<1x560x14x14xf16>
  %fill110 = linalg.fill ins(%cst : f16) outs(%init109 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %conv111 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad108, %w18 : tensor<1x560x16x16xf16>, tensor<560x560x3x3xf16>)
    outs(%fill110 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %empty112 = tensor.empty() : tensor<1x560x14x14xf16>
  %relu113 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv111 : tensor<1x560x14x14xf16>)
    outs(%empty112 : tensor<1x560x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x560x14x14xf16>

  // conv19: 3x3 s1 p1 560->560 14x14
  %pad114 = tensor.pad %relu113 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x560x14x14xf16> to tensor<1x560x16x16xf16>
  %init115 = tensor.empty() : tensor<1x560x14x14xf16>
  %fill116 = linalg.fill ins(%cst : f16) outs(%init115 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %conv117 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad114, %w19 : tensor<1x560x16x16xf16>, tensor<560x560x3x3xf16>)
    outs(%fill116 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %empty118 = tensor.empty() : tensor<1x560x14x14xf16>
  %relu119 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv117 : tensor<1x560x14x14xf16>)
    outs(%empty118 : tensor<1x560x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x560x14x14xf16>

  // conv20: 3x3 s1 p1 560->560 14x14
  %pad120 = tensor.pad %relu119 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x560x14x14xf16> to tensor<1x560x16x16xf16>
  %init121 = tensor.empty() : tensor<1x560x14x14xf16>
  %fill122 = linalg.fill ins(%cst : f16) outs(%init121 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %conv123 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad120, %w20 : tensor<1x560x16x16xf16>, tensor<560x560x3x3xf16>)
    outs(%fill122 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %empty124 = tensor.empty() : tensor<1x560x14x14xf16>
  %relu125 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv123 : tensor<1x560x14x14xf16>)
    outs(%empty124 : tensor<1x560x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x560x14x14xf16>

  // conv21: 3x3 s1 p1 560->560 14x14
  %pad126 = tensor.pad %relu125 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x560x14x14xf16> to tensor<1x560x16x16xf16>
  %init127 = tensor.empty() : tensor<1x560x14x14xf16>
  %fill128 = linalg.fill ins(%cst : f16) outs(%init127 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %conv129 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad126, %w21 : tensor<1x560x16x16xf16>, tensor<560x560x3x3xf16>)
    outs(%fill128 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %empty130 = tensor.empty() : tensor<1x560x14x14xf16>
  %relu131 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv129 : tensor<1x560x14x14xf16>)
    outs(%empty130 : tensor<1x560x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x560x14x14xf16>

  // conv22: 3x3 s1 p1 560->560 14x14
  %pad132 = tensor.pad %relu131 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x560x14x14xf16> to tensor<1x560x16x16xf16>
  %init133 = tensor.empty() : tensor<1x560x14x14xf16>
  %fill134 = linalg.fill ins(%cst : f16) outs(%init133 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %conv135 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad132, %w22 : tensor<1x560x16x16xf16>, tensor<560x560x3x3xf16>)
    outs(%fill134 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %empty136 = tensor.empty() : tensor<1x560x14x14xf16>
  %relu137 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv135 : tensor<1x560x14x14xf16>)
    outs(%empty136 : tensor<1x560x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x560x14x14xf16>

  // conv23: 3x3 s1 p1 560->560 14x14
  %pad138 = tensor.pad %relu137 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x560x14x14xf16> to tensor<1x560x16x16xf16>
  %init139 = tensor.empty() : tensor<1x560x14x14xf16>
  %fill140 = linalg.fill ins(%cst : f16) outs(%init139 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %conv141 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad138, %w23 : tensor<1x560x16x16xf16>, tensor<560x560x3x3xf16>)
    outs(%fill140 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %empty142 = tensor.empty() : tensor<1x560x14x14xf16>
  %relu143 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv141 : tensor<1x560x14x14xf16>)
    outs(%empty142 : tensor<1x560x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x560x14x14xf16>

  // conv24: 3x3 s1 p1 560->560 14x14
  %pad144 = tensor.pad %relu143 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x560x14x14xf16> to tensor<1x560x16x16xf16>
  %init145 = tensor.empty() : tensor<1x560x14x14xf16>
  %fill146 = linalg.fill ins(%cst : f16) outs(%init145 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %conv147 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad144, %w24 : tensor<1x560x16x16xf16>, tensor<560x560x3x3xf16>)
    outs(%fill146 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %empty148 = tensor.empty() : tensor<1x560x14x14xf16>
  %relu149 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv147 : tensor<1x560x14x14xf16>)
    outs(%empty148 : tensor<1x560x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x560x14x14xf16>

  // conv25: 3x3 s1 p1 560->560 14x14
  %pad150 = tensor.pad %relu149 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x560x14x14xf16> to tensor<1x560x16x16xf16>
  %init151 = tensor.empty() : tensor<1x560x14x14xf16>
  %fill152 = linalg.fill ins(%cst : f16) outs(%init151 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %conv153 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad150, %w25 : tensor<1x560x16x16xf16>, tensor<560x560x3x3xf16>)
    outs(%fill152 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %empty154 = tensor.empty() : tensor<1x560x14x14xf16>
  %relu155 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv153 : tensor<1x560x14x14xf16>)
    outs(%empty154 : tensor<1x560x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x560x14x14xf16>

  // conv26: 3x3 s1 p1 560->560 14x14
  %pad156 = tensor.pad %relu155 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x560x14x14xf16> to tensor<1x560x16x16xf16>
  %init157 = tensor.empty() : tensor<1x560x14x14xf16>
  %fill158 = linalg.fill ins(%cst : f16) outs(%init157 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %conv159 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad156, %w26 : tensor<1x560x16x16xf16>, tensor<560x560x3x3xf16>)
    outs(%fill158 : tensor<1x560x14x14xf16>) -> tensor<1x560x14x14xf16>
  %empty160 = tensor.empty() : tensor<1x560x14x14xf16>
  %relu161 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv159 : tensor<1x560x14x14xf16>)
    outs(%empty160 : tensor<1x560x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x560x14x14xf16>

  // conv27: 3x3 s2 p1 560->1360 14x14
  %pad162 = tensor.pad %relu161 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x560x14x14xf16> to tensor<1x560x16x16xf16>
  %init163 = tensor.empty() : tensor<1x1360x7x7xf16>
  %fill164 = linalg.fill ins(%cst : f16) outs(%init163 : tensor<1x1360x7x7xf16>) -> tensor<1x1360x7x7xf16>
  %conv165 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad162, %w27 : tensor<1x560x16x16xf16>, tensor<1360x560x3x3xf16>)
    outs(%fill164 : tensor<1x1360x7x7xf16>) -> tensor<1x1360x7x7xf16>
  %empty166 = tensor.empty() : tensor<1x1360x7x7xf16>
  %relu167 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv165 : tensor<1x1360x7x7xf16>)
    outs(%empty166 : tensor<1x1360x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1360x7x7xf16>

  // FC as 1x1 conv: 1360->1000
  %init168 = tensor.empty() : tensor<1x1000x7x7xf16>
  %fill169 = linalg.fill ins(%cst : f16) outs(%init168 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  %conv170 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu167, %w_fc : tensor<1x1360x7x7xf16>, tensor<1000x1360x1x1xf16>)
    outs(%fill169 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  return %conv170 : tensor<1x1000x7x7xf16>
}
