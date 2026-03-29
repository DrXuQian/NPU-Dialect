func.func @efficientnet_b4(
    %input: tensor<1x3x380x380xf16>,
    %w0: tensor<44x3x3x3xf16>,
    %w1: tensor<44x44x3x3xf16>,
    %w2: tensor<22x44x1x1xf16>,
    %w3: tensor<22x22x3x3xf16>,
    %w4: tensor<22x22x1x1xf16>,
    %w5: tensor<132x22x1x1xf16>,
    %w6: tensor<132x132x3x3xf16>,
    %w7: tensor<33x132x1x1xf16>,
    %w8: tensor<198x33x1x1xf16>,
    %w9: tensor<198x198x3x3xf16>,
    %w10: tensor<33x198x1x1xf16>,
    %w11: tensor<198x33x1x1xf16>,
    %w12: tensor<198x198x3x3xf16>,
    %w13: tensor<33x198x1x1xf16>,
    %w14: tensor<198x33x1x1xf16>,
    %w15: tensor<198x198x3x3xf16>,
    %w16: tensor<33x198x1x1xf16>,
    %w17: tensor<198x33x1x1xf16>,
    %w18: tensor<198x198x3x3xf16>,
    %w19: tensor<56x198x1x1xf16>,
    %w20: tensor<336x56x1x1xf16>,
    %w21: tensor<336x336x3x3xf16>,
    %w22: tensor<56x336x1x1xf16>,
    %w23: tensor<336x56x1x1xf16>,
    %w24: tensor<336x336x3x3xf16>,
    %w25: tensor<56x336x1x1xf16>,
    %w26: tensor<336x56x1x1xf16>,
    %w27: tensor<336x336x3x3xf16>,
    %w28: tensor<56x336x1x1xf16>,
    %w29: tensor<336x56x1x1xf16>,
    %w30: tensor<336x336x3x3xf16>,
    %w31: tensor<112x336x1x1xf16>,
    %w32: tensor<672x112x1x1xf16>,
    %w33: tensor<672x672x3x3xf16>,
    %w34: tensor<112x672x1x1xf16>,
    %w35: tensor<672x112x1x1xf16>,
    %w36: tensor<672x672x3x3xf16>,
    %w37: tensor<112x672x1x1xf16>,
    %w38: tensor<672x112x1x1xf16>,
    %w39: tensor<672x672x3x3xf16>,
    %w40: tensor<112x672x1x1xf16>,
    %w41: tensor<672x112x1x1xf16>,
    %w42: tensor<672x672x3x3xf16>,
    %w43: tensor<112x672x1x1xf16>,
    %w44: tensor<672x112x1x1xf16>,
    %w45: tensor<672x672x3x3xf16>,
    %w46: tensor<156x672x1x1xf16>,
    %w47: tensor<936x156x1x1xf16>,
    %w48: tensor<936x936x3x3xf16>,
    %w49: tensor<156x936x1x1xf16>,
    %w50: tensor<936x156x1x1xf16>,
    %w51: tensor<936x936x3x3xf16>,
    %w52: tensor<156x936x1x1xf16>,
    %w53: tensor<936x156x1x1xf16>,
    %w54: tensor<936x936x3x3xf16>,
    %w55: tensor<156x936x1x1xf16>,
    %w56: tensor<936x156x1x1xf16>,
    %w57: tensor<936x936x3x3xf16>,
    %w58: tensor<156x936x1x1xf16>,
    %w59: tensor<936x156x1x1xf16>,
    %w60: tensor<936x936x3x3xf16>,
    %w61: tensor<268x936x1x1xf16>,
    %w62: tensor<1608x268x1x1xf16>,
    %w63: tensor<1608x1608x3x3xf16>,
    %w64: tensor<268x1608x1x1xf16>,
    %w65: tensor<1608x268x1x1xf16>,
    %w66: tensor<1608x1608x3x3xf16>,
    %w67: tensor<268x1608x1x1xf16>,
    %w68: tensor<1608x268x1x1xf16>,
    %w69: tensor<1608x1608x3x3xf16>,
    %w70: tensor<268x1608x1x1xf16>,
    %w71: tensor<1608x268x1x1xf16>,
    %w72: tensor<1608x1608x3x3xf16>,
    %w73: tensor<268x1608x1x1xf16>,
    %w74: tensor<1608x268x1x1xf16>,
    %w75: tensor<1608x1608x3x3xf16>,
    %w76: tensor<268x1608x1x1xf16>,
    %w77: tensor<1608x268x1x1xf16>,
    %w78: tensor<1608x1608x3x3xf16>,
    %w79: tensor<268x1608x1x1xf16>,
    %w80: tensor<1608x268x1x1xf16>,
    %w81: tensor<1608x1608x3x3xf16>,
    %w82: tensor<448x1608x1x1xf16>,
    %w83: tensor<2688x448x1x1xf16>,
    %w84: tensor<2688x2688x3x3xf16>,
    %w85: tensor<448x2688x1x1xf16>,
    %w86: tensor<1792x448x1x1xf16>,
    %w_fc: tensor<1000x1792x1x1xf16>) -> tensor<1x1000x12x12xf16> {
  %cst = arith.constant 0.0 : f16

  // conv0: 3x3 s2 p1 3->44 380x380
  %pad0 = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x380x380xf16> to tensor<1x3x382x382xf16>
  %init1 = tensor.empty() : tensor<1x44x190x190xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x44x190x190xf16>) -> tensor<1x44x190x190xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad0, %w0 : tensor<1x3x382x382xf16>, tensor<44x3x3x3xf16>)
    outs(%fill2 : tensor<1x44x190x190xf16>) -> tensor<1x44x190x190xf16>
  %empty4 = tensor.empty() : tensor<1x44x190x190xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x44x190x190xf16>)
    outs(%empty4 : tensor<1x44x190x190xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x44x190x190xf16>

  // conv1: 3x3 s1 p1 44->44 190x190
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x44x190x190xf16> to tensor<1x44x192x192xf16>
  %init7 = tensor.empty() : tensor<1x44x190x190xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x44x190x190xf16>) -> tensor<1x44x190x190xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad6, %w1 : tensor<1x44x192x192xf16>, tensor<44x44x3x3xf16>)
    outs(%fill8 : tensor<1x44x190x190xf16>) -> tensor<1x44x190x190xf16>
  %empty10 = tensor.empty() : tensor<1x44x190x190xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x44x190x190xf16>)
    outs(%empty10 : tensor<1x44x190x190xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x44x190x190xf16>

  // conv2: 1x1 s1 p0 44->22 190x190
  %init12 = tensor.empty() : tensor<1x22x190x190xf16>
  %fill13 = linalg.fill ins(%cst : f16) outs(%init12 : tensor<1x22x190x190xf16>) -> tensor<1x22x190x190xf16>
  %conv14 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu11, %w2 : tensor<1x44x190x190xf16>, tensor<22x44x1x1xf16>)
    outs(%fill13 : tensor<1x22x190x190xf16>) -> tensor<1x22x190x190xf16>
  %empty15 = tensor.empty() : tensor<1x22x190x190xf16>
  %relu16 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv14 : tensor<1x22x190x190xf16>)
    outs(%empty15 : tensor<1x22x190x190xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x22x190x190xf16>

  // conv3: 3x3 s1 p1 22->22 190x190
  %pad17 = tensor.pad %relu16 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x22x190x190xf16> to tensor<1x22x192x192xf16>
  %init18 = tensor.empty() : tensor<1x22x190x190xf16>
  %fill19 = linalg.fill ins(%cst : f16) outs(%init18 : tensor<1x22x190x190xf16>) -> tensor<1x22x190x190xf16>
  %conv20 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad17, %w3 : tensor<1x22x192x192xf16>, tensor<22x22x3x3xf16>)
    outs(%fill19 : tensor<1x22x190x190xf16>) -> tensor<1x22x190x190xf16>
  %empty21 = tensor.empty() : tensor<1x22x190x190xf16>
  %relu22 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv20 : tensor<1x22x190x190xf16>)
    outs(%empty21 : tensor<1x22x190x190xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x22x190x190xf16>

  // conv4: 1x1 s1 p0 22->22 190x190
  %init23 = tensor.empty() : tensor<1x22x190x190xf16>
  %fill24 = linalg.fill ins(%cst : f16) outs(%init23 : tensor<1x22x190x190xf16>) -> tensor<1x22x190x190xf16>
  %conv25 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu22, %w4 : tensor<1x22x190x190xf16>, tensor<22x22x1x1xf16>)
    outs(%fill24 : tensor<1x22x190x190xf16>) -> tensor<1x22x190x190xf16>
  %empty26 = tensor.empty() : tensor<1x22x190x190xf16>
  %relu27 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv25 : tensor<1x22x190x190xf16>)
    outs(%empty26 : tensor<1x22x190x190xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x22x190x190xf16>

  // conv5: 1x1 s1 p0 22->132 190x190
  %init28 = tensor.empty() : tensor<1x132x190x190xf16>
  %fill29 = linalg.fill ins(%cst : f16) outs(%init28 : tensor<1x132x190x190xf16>) -> tensor<1x132x190x190xf16>
  %conv30 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu27, %w5 : tensor<1x22x190x190xf16>, tensor<132x22x1x1xf16>)
    outs(%fill29 : tensor<1x132x190x190xf16>) -> tensor<1x132x190x190xf16>
  %empty31 = tensor.empty() : tensor<1x132x190x190xf16>
  %relu32 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv30 : tensor<1x132x190x190xf16>)
    outs(%empty31 : tensor<1x132x190x190xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x132x190x190xf16>

  // conv6: 3x3 s2 p1 132->132 190x190
  %pad33 = tensor.pad %relu32 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x132x190x190xf16> to tensor<1x132x192x192xf16>
  %init34 = tensor.empty() : tensor<1x132x95x95xf16>
  %fill35 = linalg.fill ins(%cst : f16) outs(%init34 : tensor<1x132x95x95xf16>) -> tensor<1x132x95x95xf16>
  %conv36 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad33, %w6 : tensor<1x132x192x192xf16>, tensor<132x132x3x3xf16>)
    outs(%fill35 : tensor<1x132x95x95xf16>) -> tensor<1x132x95x95xf16>
  %empty37 = tensor.empty() : tensor<1x132x95x95xf16>
  %relu38 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv36 : tensor<1x132x95x95xf16>)
    outs(%empty37 : tensor<1x132x95x95xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x132x95x95xf16>

  // conv7: 1x1 s1 p0 132->33 95x95
  %init39 = tensor.empty() : tensor<1x33x95x95xf16>
  %fill40 = linalg.fill ins(%cst : f16) outs(%init39 : tensor<1x33x95x95xf16>) -> tensor<1x33x95x95xf16>
  %conv41 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu38, %w7 : tensor<1x132x95x95xf16>, tensor<33x132x1x1xf16>)
    outs(%fill40 : tensor<1x33x95x95xf16>) -> tensor<1x33x95x95xf16>
  %empty42 = tensor.empty() : tensor<1x33x95x95xf16>
  %relu43 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv41 : tensor<1x33x95x95xf16>)
    outs(%empty42 : tensor<1x33x95x95xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x33x95x95xf16>

  // conv8: 1x1 s1 p0 33->198 95x95
  %init44 = tensor.empty() : tensor<1x198x95x95xf16>
  %fill45 = linalg.fill ins(%cst : f16) outs(%init44 : tensor<1x198x95x95xf16>) -> tensor<1x198x95x95xf16>
  %conv46 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu43, %w8 : tensor<1x33x95x95xf16>, tensor<198x33x1x1xf16>)
    outs(%fill45 : tensor<1x198x95x95xf16>) -> tensor<1x198x95x95xf16>
  %empty47 = tensor.empty() : tensor<1x198x95x95xf16>
  %relu48 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv46 : tensor<1x198x95x95xf16>)
    outs(%empty47 : tensor<1x198x95x95xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x198x95x95xf16>

  // conv9: 3x3 s1 p1 198->198 95x95
  %pad49 = tensor.pad %relu48 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x198x95x95xf16> to tensor<1x198x97x97xf16>
  %init50 = tensor.empty() : tensor<1x198x95x95xf16>
  %fill51 = linalg.fill ins(%cst : f16) outs(%init50 : tensor<1x198x95x95xf16>) -> tensor<1x198x95x95xf16>
  %conv52 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad49, %w9 : tensor<1x198x97x97xf16>, tensor<198x198x3x3xf16>)
    outs(%fill51 : tensor<1x198x95x95xf16>) -> tensor<1x198x95x95xf16>
  %empty53 = tensor.empty() : tensor<1x198x95x95xf16>
  %relu54 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv52 : tensor<1x198x95x95xf16>)
    outs(%empty53 : tensor<1x198x95x95xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x198x95x95xf16>

  // conv10: 1x1 s1 p0 198->33 95x95
  %init55 = tensor.empty() : tensor<1x33x95x95xf16>
  %fill56 = linalg.fill ins(%cst : f16) outs(%init55 : tensor<1x33x95x95xf16>) -> tensor<1x33x95x95xf16>
  %conv57 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu54, %w10 : tensor<1x198x95x95xf16>, tensor<33x198x1x1xf16>)
    outs(%fill56 : tensor<1x33x95x95xf16>) -> tensor<1x33x95x95xf16>
  %empty58 = tensor.empty() : tensor<1x33x95x95xf16>
  %relu59 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv57 : tensor<1x33x95x95xf16>)
    outs(%empty58 : tensor<1x33x95x95xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x33x95x95xf16>

  // conv11: 1x1 s1 p0 33->198 95x95
  %init60 = tensor.empty() : tensor<1x198x95x95xf16>
  %fill61 = linalg.fill ins(%cst : f16) outs(%init60 : tensor<1x198x95x95xf16>) -> tensor<1x198x95x95xf16>
  %conv62 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu59, %w11 : tensor<1x33x95x95xf16>, tensor<198x33x1x1xf16>)
    outs(%fill61 : tensor<1x198x95x95xf16>) -> tensor<1x198x95x95xf16>
  %empty63 = tensor.empty() : tensor<1x198x95x95xf16>
  %relu64 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv62 : tensor<1x198x95x95xf16>)
    outs(%empty63 : tensor<1x198x95x95xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x198x95x95xf16>

  // conv12: 3x3 s1 p1 198->198 95x95
  %pad65 = tensor.pad %relu64 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x198x95x95xf16> to tensor<1x198x97x97xf16>
  %init66 = tensor.empty() : tensor<1x198x95x95xf16>
  %fill67 = linalg.fill ins(%cst : f16) outs(%init66 : tensor<1x198x95x95xf16>) -> tensor<1x198x95x95xf16>
  %conv68 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad65, %w12 : tensor<1x198x97x97xf16>, tensor<198x198x3x3xf16>)
    outs(%fill67 : tensor<1x198x95x95xf16>) -> tensor<1x198x95x95xf16>
  %empty69 = tensor.empty() : tensor<1x198x95x95xf16>
  %relu70 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv68 : tensor<1x198x95x95xf16>)
    outs(%empty69 : tensor<1x198x95x95xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x198x95x95xf16>

  // conv13: 1x1 s1 p0 198->33 95x95
  %init71 = tensor.empty() : tensor<1x33x95x95xf16>
  %fill72 = linalg.fill ins(%cst : f16) outs(%init71 : tensor<1x33x95x95xf16>) -> tensor<1x33x95x95xf16>
  %conv73 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu70, %w13 : tensor<1x198x95x95xf16>, tensor<33x198x1x1xf16>)
    outs(%fill72 : tensor<1x33x95x95xf16>) -> tensor<1x33x95x95xf16>
  %empty74 = tensor.empty() : tensor<1x33x95x95xf16>
  %relu75 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv73 : tensor<1x33x95x95xf16>)
    outs(%empty74 : tensor<1x33x95x95xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x33x95x95xf16>

  // conv14: 1x1 s1 p0 33->198 95x95
  %init76 = tensor.empty() : tensor<1x198x95x95xf16>
  %fill77 = linalg.fill ins(%cst : f16) outs(%init76 : tensor<1x198x95x95xf16>) -> tensor<1x198x95x95xf16>
  %conv78 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu75, %w14 : tensor<1x33x95x95xf16>, tensor<198x33x1x1xf16>)
    outs(%fill77 : tensor<1x198x95x95xf16>) -> tensor<1x198x95x95xf16>
  %empty79 = tensor.empty() : tensor<1x198x95x95xf16>
  %relu80 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv78 : tensor<1x198x95x95xf16>)
    outs(%empty79 : tensor<1x198x95x95xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x198x95x95xf16>

  // conv15: 3x3 s1 p1 198->198 95x95
  %pad81 = tensor.pad %relu80 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x198x95x95xf16> to tensor<1x198x97x97xf16>
  %init82 = tensor.empty() : tensor<1x198x95x95xf16>
  %fill83 = linalg.fill ins(%cst : f16) outs(%init82 : tensor<1x198x95x95xf16>) -> tensor<1x198x95x95xf16>
  %conv84 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad81, %w15 : tensor<1x198x97x97xf16>, tensor<198x198x3x3xf16>)
    outs(%fill83 : tensor<1x198x95x95xf16>) -> tensor<1x198x95x95xf16>
  %empty85 = tensor.empty() : tensor<1x198x95x95xf16>
  %relu86 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv84 : tensor<1x198x95x95xf16>)
    outs(%empty85 : tensor<1x198x95x95xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x198x95x95xf16>

  // conv16: 1x1 s1 p0 198->33 95x95
  %init87 = tensor.empty() : tensor<1x33x95x95xf16>
  %fill88 = linalg.fill ins(%cst : f16) outs(%init87 : tensor<1x33x95x95xf16>) -> tensor<1x33x95x95xf16>
  %conv89 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu86, %w16 : tensor<1x198x95x95xf16>, tensor<33x198x1x1xf16>)
    outs(%fill88 : tensor<1x33x95x95xf16>) -> tensor<1x33x95x95xf16>
  %empty90 = tensor.empty() : tensor<1x33x95x95xf16>
  %relu91 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv89 : tensor<1x33x95x95xf16>)
    outs(%empty90 : tensor<1x33x95x95xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x33x95x95xf16>

  // conv17: 1x1 s1 p0 33->198 95x95
  %init92 = tensor.empty() : tensor<1x198x95x95xf16>
  %fill93 = linalg.fill ins(%cst : f16) outs(%init92 : tensor<1x198x95x95xf16>) -> tensor<1x198x95x95xf16>
  %conv94 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu91, %w17 : tensor<1x33x95x95xf16>, tensor<198x33x1x1xf16>)
    outs(%fill93 : tensor<1x198x95x95xf16>) -> tensor<1x198x95x95xf16>
  %empty95 = tensor.empty() : tensor<1x198x95x95xf16>
  %relu96 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv94 : tensor<1x198x95x95xf16>)
    outs(%empty95 : tensor<1x198x95x95xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x198x95x95xf16>

  // conv18: 3x3 s2 p1 198->198 95x95
  %pad97 = tensor.pad %relu96 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x198x95x95xf16> to tensor<1x198x97x97xf16>
  %init98 = tensor.empty() : tensor<1x198x48x48xf16>
  %fill99 = linalg.fill ins(%cst : f16) outs(%init98 : tensor<1x198x48x48xf16>) -> tensor<1x198x48x48xf16>
  %conv100 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad97, %w18 : tensor<1x198x97x97xf16>, tensor<198x198x3x3xf16>)
    outs(%fill99 : tensor<1x198x48x48xf16>) -> tensor<1x198x48x48xf16>
  %empty101 = tensor.empty() : tensor<1x198x48x48xf16>
  %relu102 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv100 : tensor<1x198x48x48xf16>)
    outs(%empty101 : tensor<1x198x48x48xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x198x48x48xf16>

  // conv19: 1x1 s1 p0 198->56 48x48
  %init103 = tensor.empty() : tensor<1x56x48x48xf16>
  %fill104 = linalg.fill ins(%cst : f16) outs(%init103 : tensor<1x56x48x48xf16>) -> tensor<1x56x48x48xf16>
  %conv105 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu102, %w19 : tensor<1x198x48x48xf16>, tensor<56x198x1x1xf16>)
    outs(%fill104 : tensor<1x56x48x48xf16>) -> tensor<1x56x48x48xf16>
  %empty106 = tensor.empty() : tensor<1x56x48x48xf16>
  %relu107 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv105 : tensor<1x56x48x48xf16>)
    outs(%empty106 : tensor<1x56x48x48xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x56x48x48xf16>

  // conv20: 1x1 s1 p0 56->336 48x48
  %init108 = tensor.empty() : tensor<1x336x48x48xf16>
  %fill109 = linalg.fill ins(%cst : f16) outs(%init108 : tensor<1x336x48x48xf16>) -> tensor<1x336x48x48xf16>
  %conv110 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu107, %w20 : tensor<1x56x48x48xf16>, tensor<336x56x1x1xf16>)
    outs(%fill109 : tensor<1x336x48x48xf16>) -> tensor<1x336x48x48xf16>
  %empty111 = tensor.empty() : tensor<1x336x48x48xf16>
  %relu112 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv110 : tensor<1x336x48x48xf16>)
    outs(%empty111 : tensor<1x336x48x48xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x336x48x48xf16>

  // conv21: 3x3 s1 p1 336->336 48x48
  %pad113 = tensor.pad %relu112 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x336x48x48xf16> to tensor<1x336x50x50xf16>
  %init114 = tensor.empty() : tensor<1x336x48x48xf16>
  %fill115 = linalg.fill ins(%cst : f16) outs(%init114 : tensor<1x336x48x48xf16>) -> tensor<1x336x48x48xf16>
  %conv116 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad113, %w21 : tensor<1x336x50x50xf16>, tensor<336x336x3x3xf16>)
    outs(%fill115 : tensor<1x336x48x48xf16>) -> tensor<1x336x48x48xf16>
  %empty117 = tensor.empty() : tensor<1x336x48x48xf16>
  %relu118 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv116 : tensor<1x336x48x48xf16>)
    outs(%empty117 : tensor<1x336x48x48xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x336x48x48xf16>

  // conv22: 1x1 s1 p0 336->56 48x48
  %init119 = tensor.empty() : tensor<1x56x48x48xf16>
  %fill120 = linalg.fill ins(%cst : f16) outs(%init119 : tensor<1x56x48x48xf16>) -> tensor<1x56x48x48xf16>
  %conv121 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu118, %w22 : tensor<1x336x48x48xf16>, tensor<56x336x1x1xf16>)
    outs(%fill120 : tensor<1x56x48x48xf16>) -> tensor<1x56x48x48xf16>
  %empty122 = tensor.empty() : tensor<1x56x48x48xf16>
  %relu123 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv121 : tensor<1x56x48x48xf16>)
    outs(%empty122 : tensor<1x56x48x48xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x56x48x48xf16>

  // conv23: 1x1 s1 p0 56->336 48x48
  %init124 = tensor.empty() : tensor<1x336x48x48xf16>
  %fill125 = linalg.fill ins(%cst : f16) outs(%init124 : tensor<1x336x48x48xf16>) -> tensor<1x336x48x48xf16>
  %conv126 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu123, %w23 : tensor<1x56x48x48xf16>, tensor<336x56x1x1xf16>)
    outs(%fill125 : tensor<1x336x48x48xf16>) -> tensor<1x336x48x48xf16>
  %empty127 = tensor.empty() : tensor<1x336x48x48xf16>
  %relu128 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv126 : tensor<1x336x48x48xf16>)
    outs(%empty127 : tensor<1x336x48x48xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x336x48x48xf16>

  // conv24: 3x3 s1 p1 336->336 48x48
  %pad129 = tensor.pad %relu128 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x336x48x48xf16> to tensor<1x336x50x50xf16>
  %init130 = tensor.empty() : tensor<1x336x48x48xf16>
  %fill131 = linalg.fill ins(%cst : f16) outs(%init130 : tensor<1x336x48x48xf16>) -> tensor<1x336x48x48xf16>
  %conv132 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad129, %w24 : tensor<1x336x50x50xf16>, tensor<336x336x3x3xf16>)
    outs(%fill131 : tensor<1x336x48x48xf16>) -> tensor<1x336x48x48xf16>
  %empty133 = tensor.empty() : tensor<1x336x48x48xf16>
  %relu134 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv132 : tensor<1x336x48x48xf16>)
    outs(%empty133 : tensor<1x336x48x48xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x336x48x48xf16>

  // conv25: 1x1 s1 p0 336->56 48x48
  %init135 = tensor.empty() : tensor<1x56x48x48xf16>
  %fill136 = linalg.fill ins(%cst : f16) outs(%init135 : tensor<1x56x48x48xf16>) -> tensor<1x56x48x48xf16>
  %conv137 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu134, %w25 : tensor<1x336x48x48xf16>, tensor<56x336x1x1xf16>)
    outs(%fill136 : tensor<1x56x48x48xf16>) -> tensor<1x56x48x48xf16>
  %empty138 = tensor.empty() : tensor<1x56x48x48xf16>
  %relu139 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv137 : tensor<1x56x48x48xf16>)
    outs(%empty138 : tensor<1x56x48x48xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x56x48x48xf16>

  // conv26: 1x1 s1 p0 56->336 48x48
  %init140 = tensor.empty() : tensor<1x336x48x48xf16>
  %fill141 = linalg.fill ins(%cst : f16) outs(%init140 : tensor<1x336x48x48xf16>) -> tensor<1x336x48x48xf16>
  %conv142 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu139, %w26 : tensor<1x56x48x48xf16>, tensor<336x56x1x1xf16>)
    outs(%fill141 : tensor<1x336x48x48xf16>) -> tensor<1x336x48x48xf16>
  %empty143 = tensor.empty() : tensor<1x336x48x48xf16>
  %relu144 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv142 : tensor<1x336x48x48xf16>)
    outs(%empty143 : tensor<1x336x48x48xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x336x48x48xf16>

  // conv27: 3x3 s1 p1 336->336 48x48
  %pad145 = tensor.pad %relu144 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x336x48x48xf16> to tensor<1x336x50x50xf16>
  %init146 = tensor.empty() : tensor<1x336x48x48xf16>
  %fill147 = linalg.fill ins(%cst : f16) outs(%init146 : tensor<1x336x48x48xf16>) -> tensor<1x336x48x48xf16>
  %conv148 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad145, %w27 : tensor<1x336x50x50xf16>, tensor<336x336x3x3xf16>)
    outs(%fill147 : tensor<1x336x48x48xf16>) -> tensor<1x336x48x48xf16>
  %empty149 = tensor.empty() : tensor<1x336x48x48xf16>
  %relu150 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv148 : tensor<1x336x48x48xf16>)
    outs(%empty149 : tensor<1x336x48x48xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x336x48x48xf16>

  // conv28: 1x1 s1 p0 336->56 48x48
  %init151 = tensor.empty() : tensor<1x56x48x48xf16>
  %fill152 = linalg.fill ins(%cst : f16) outs(%init151 : tensor<1x56x48x48xf16>) -> tensor<1x56x48x48xf16>
  %conv153 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu150, %w28 : tensor<1x336x48x48xf16>, tensor<56x336x1x1xf16>)
    outs(%fill152 : tensor<1x56x48x48xf16>) -> tensor<1x56x48x48xf16>
  %empty154 = tensor.empty() : tensor<1x56x48x48xf16>
  %relu155 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv153 : tensor<1x56x48x48xf16>)
    outs(%empty154 : tensor<1x56x48x48xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x56x48x48xf16>

  // conv29: 1x1 s1 p0 56->336 48x48
  %init156 = tensor.empty() : tensor<1x336x48x48xf16>
  %fill157 = linalg.fill ins(%cst : f16) outs(%init156 : tensor<1x336x48x48xf16>) -> tensor<1x336x48x48xf16>
  %conv158 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu155, %w29 : tensor<1x56x48x48xf16>, tensor<336x56x1x1xf16>)
    outs(%fill157 : tensor<1x336x48x48xf16>) -> tensor<1x336x48x48xf16>
  %empty159 = tensor.empty() : tensor<1x336x48x48xf16>
  %relu160 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv158 : tensor<1x336x48x48xf16>)
    outs(%empty159 : tensor<1x336x48x48xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x336x48x48xf16>

  // conv30: 3x3 s2 p1 336->336 48x48
  %pad161 = tensor.pad %relu160 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x336x48x48xf16> to tensor<1x336x50x50xf16>
  %init162 = tensor.empty() : tensor<1x336x24x24xf16>
  %fill163 = linalg.fill ins(%cst : f16) outs(%init162 : tensor<1x336x24x24xf16>) -> tensor<1x336x24x24xf16>
  %conv164 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad161, %w30 : tensor<1x336x50x50xf16>, tensor<336x336x3x3xf16>)
    outs(%fill163 : tensor<1x336x24x24xf16>) -> tensor<1x336x24x24xf16>
  %empty165 = tensor.empty() : tensor<1x336x24x24xf16>
  %relu166 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv164 : tensor<1x336x24x24xf16>)
    outs(%empty165 : tensor<1x336x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x336x24x24xf16>

  // conv31: 1x1 s1 p0 336->112 24x24
  %init167 = tensor.empty() : tensor<1x112x24x24xf16>
  %fill168 = linalg.fill ins(%cst : f16) outs(%init167 : tensor<1x112x24x24xf16>) -> tensor<1x112x24x24xf16>
  %conv169 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu166, %w31 : tensor<1x336x24x24xf16>, tensor<112x336x1x1xf16>)
    outs(%fill168 : tensor<1x112x24x24xf16>) -> tensor<1x112x24x24xf16>
  %empty170 = tensor.empty() : tensor<1x112x24x24xf16>
  %relu171 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv169 : tensor<1x112x24x24xf16>)
    outs(%empty170 : tensor<1x112x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x112x24x24xf16>

  // conv32: 1x1 s1 p0 112->672 24x24
  %init172 = tensor.empty() : tensor<1x672x24x24xf16>
  %fill173 = linalg.fill ins(%cst : f16) outs(%init172 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %conv174 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu171, %w32 : tensor<1x112x24x24xf16>, tensor<672x112x1x1xf16>)
    outs(%fill173 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %empty175 = tensor.empty() : tensor<1x672x24x24xf16>
  %relu176 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv174 : tensor<1x672x24x24xf16>)
    outs(%empty175 : tensor<1x672x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x24x24xf16>

  // conv33: 3x3 s1 p1 672->672 24x24
  %pad177 = tensor.pad %relu176 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x672x24x24xf16> to tensor<1x672x26x26xf16>
  %init178 = tensor.empty() : tensor<1x672x24x24xf16>
  %fill179 = linalg.fill ins(%cst : f16) outs(%init178 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %conv180 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad177, %w33 : tensor<1x672x26x26xf16>, tensor<672x672x3x3xf16>)
    outs(%fill179 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %empty181 = tensor.empty() : tensor<1x672x24x24xf16>
  %relu182 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv180 : tensor<1x672x24x24xf16>)
    outs(%empty181 : tensor<1x672x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x24x24xf16>

  // conv34: 1x1 s1 p0 672->112 24x24
  %init183 = tensor.empty() : tensor<1x112x24x24xf16>
  %fill184 = linalg.fill ins(%cst : f16) outs(%init183 : tensor<1x112x24x24xf16>) -> tensor<1x112x24x24xf16>
  %conv185 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu182, %w34 : tensor<1x672x24x24xf16>, tensor<112x672x1x1xf16>)
    outs(%fill184 : tensor<1x112x24x24xf16>) -> tensor<1x112x24x24xf16>
  %empty186 = tensor.empty() : tensor<1x112x24x24xf16>
  %relu187 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv185 : tensor<1x112x24x24xf16>)
    outs(%empty186 : tensor<1x112x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x112x24x24xf16>

  // conv35: 1x1 s1 p0 112->672 24x24
  %init188 = tensor.empty() : tensor<1x672x24x24xf16>
  %fill189 = linalg.fill ins(%cst : f16) outs(%init188 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %conv190 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu187, %w35 : tensor<1x112x24x24xf16>, tensor<672x112x1x1xf16>)
    outs(%fill189 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %empty191 = tensor.empty() : tensor<1x672x24x24xf16>
  %relu192 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv190 : tensor<1x672x24x24xf16>)
    outs(%empty191 : tensor<1x672x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x24x24xf16>

  // conv36: 3x3 s1 p1 672->672 24x24
  %pad193 = tensor.pad %relu192 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x672x24x24xf16> to tensor<1x672x26x26xf16>
  %init194 = tensor.empty() : tensor<1x672x24x24xf16>
  %fill195 = linalg.fill ins(%cst : f16) outs(%init194 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %conv196 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad193, %w36 : tensor<1x672x26x26xf16>, tensor<672x672x3x3xf16>)
    outs(%fill195 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %empty197 = tensor.empty() : tensor<1x672x24x24xf16>
  %relu198 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv196 : tensor<1x672x24x24xf16>)
    outs(%empty197 : tensor<1x672x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x24x24xf16>

  // conv37: 1x1 s1 p0 672->112 24x24
  %init199 = tensor.empty() : tensor<1x112x24x24xf16>
  %fill200 = linalg.fill ins(%cst : f16) outs(%init199 : tensor<1x112x24x24xf16>) -> tensor<1x112x24x24xf16>
  %conv201 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu198, %w37 : tensor<1x672x24x24xf16>, tensor<112x672x1x1xf16>)
    outs(%fill200 : tensor<1x112x24x24xf16>) -> tensor<1x112x24x24xf16>
  %empty202 = tensor.empty() : tensor<1x112x24x24xf16>
  %relu203 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv201 : tensor<1x112x24x24xf16>)
    outs(%empty202 : tensor<1x112x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x112x24x24xf16>

  // conv38: 1x1 s1 p0 112->672 24x24
  %init204 = tensor.empty() : tensor<1x672x24x24xf16>
  %fill205 = linalg.fill ins(%cst : f16) outs(%init204 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %conv206 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu203, %w38 : tensor<1x112x24x24xf16>, tensor<672x112x1x1xf16>)
    outs(%fill205 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %empty207 = tensor.empty() : tensor<1x672x24x24xf16>
  %relu208 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv206 : tensor<1x672x24x24xf16>)
    outs(%empty207 : tensor<1x672x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x24x24xf16>

  // conv39: 3x3 s1 p1 672->672 24x24
  %pad209 = tensor.pad %relu208 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x672x24x24xf16> to tensor<1x672x26x26xf16>
  %init210 = tensor.empty() : tensor<1x672x24x24xf16>
  %fill211 = linalg.fill ins(%cst : f16) outs(%init210 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %conv212 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad209, %w39 : tensor<1x672x26x26xf16>, tensor<672x672x3x3xf16>)
    outs(%fill211 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %empty213 = tensor.empty() : tensor<1x672x24x24xf16>
  %relu214 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv212 : tensor<1x672x24x24xf16>)
    outs(%empty213 : tensor<1x672x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x24x24xf16>

  // conv40: 1x1 s1 p0 672->112 24x24
  %init215 = tensor.empty() : tensor<1x112x24x24xf16>
  %fill216 = linalg.fill ins(%cst : f16) outs(%init215 : tensor<1x112x24x24xf16>) -> tensor<1x112x24x24xf16>
  %conv217 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu214, %w40 : tensor<1x672x24x24xf16>, tensor<112x672x1x1xf16>)
    outs(%fill216 : tensor<1x112x24x24xf16>) -> tensor<1x112x24x24xf16>
  %empty218 = tensor.empty() : tensor<1x112x24x24xf16>
  %relu219 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv217 : tensor<1x112x24x24xf16>)
    outs(%empty218 : tensor<1x112x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x112x24x24xf16>

  // conv41: 1x1 s1 p0 112->672 24x24
  %init220 = tensor.empty() : tensor<1x672x24x24xf16>
  %fill221 = linalg.fill ins(%cst : f16) outs(%init220 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %conv222 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu219, %w41 : tensor<1x112x24x24xf16>, tensor<672x112x1x1xf16>)
    outs(%fill221 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %empty223 = tensor.empty() : tensor<1x672x24x24xf16>
  %relu224 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv222 : tensor<1x672x24x24xf16>)
    outs(%empty223 : tensor<1x672x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x24x24xf16>

  // conv42: 3x3 s1 p1 672->672 24x24
  %pad225 = tensor.pad %relu224 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x672x24x24xf16> to tensor<1x672x26x26xf16>
  %init226 = tensor.empty() : tensor<1x672x24x24xf16>
  %fill227 = linalg.fill ins(%cst : f16) outs(%init226 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %conv228 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad225, %w42 : tensor<1x672x26x26xf16>, tensor<672x672x3x3xf16>)
    outs(%fill227 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %empty229 = tensor.empty() : tensor<1x672x24x24xf16>
  %relu230 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv228 : tensor<1x672x24x24xf16>)
    outs(%empty229 : tensor<1x672x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x24x24xf16>

  // conv43: 1x1 s1 p0 672->112 24x24
  %init231 = tensor.empty() : tensor<1x112x24x24xf16>
  %fill232 = linalg.fill ins(%cst : f16) outs(%init231 : tensor<1x112x24x24xf16>) -> tensor<1x112x24x24xf16>
  %conv233 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu230, %w43 : tensor<1x672x24x24xf16>, tensor<112x672x1x1xf16>)
    outs(%fill232 : tensor<1x112x24x24xf16>) -> tensor<1x112x24x24xf16>
  %empty234 = tensor.empty() : tensor<1x112x24x24xf16>
  %relu235 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv233 : tensor<1x112x24x24xf16>)
    outs(%empty234 : tensor<1x112x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x112x24x24xf16>

  // conv44: 1x1 s1 p0 112->672 24x24
  %init236 = tensor.empty() : tensor<1x672x24x24xf16>
  %fill237 = linalg.fill ins(%cst : f16) outs(%init236 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %conv238 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu235, %w44 : tensor<1x112x24x24xf16>, tensor<672x112x1x1xf16>)
    outs(%fill237 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %empty239 = tensor.empty() : tensor<1x672x24x24xf16>
  %relu240 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv238 : tensor<1x672x24x24xf16>)
    outs(%empty239 : tensor<1x672x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x24x24xf16>

  // conv45: 3x3 s1 p1 672->672 24x24
  %pad241 = tensor.pad %relu240 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x672x24x24xf16> to tensor<1x672x26x26xf16>
  %init242 = tensor.empty() : tensor<1x672x24x24xf16>
  %fill243 = linalg.fill ins(%cst : f16) outs(%init242 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %conv244 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad241, %w45 : tensor<1x672x26x26xf16>, tensor<672x672x3x3xf16>)
    outs(%fill243 : tensor<1x672x24x24xf16>) -> tensor<1x672x24x24xf16>
  %empty245 = tensor.empty() : tensor<1x672x24x24xf16>
  %relu246 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv244 : tensor<1x672x24x24xf16>)
    outs(%empty245 : tensor<1x672x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x24x24xf16>

  // conv46: 1x1 s1 p0 672->156 24x24
  %init247 = tensor.empty() : tensor<1x156x24x24xf16>
  %fill248 = linalg.fill ins(%cst : f16) outs(%init247 : tensor<1x156x24x24xf16>) -> tensor<1x156x24x24xf16>
  %conv249 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu246, %w46 : tensor<1x672x24x24xf16>, tensor<156x672x1x1xf16>)
    outs(%fill248 : tensor<1x156x24x24xf16>) -> tensor<1x156x24x24xf16>
  %empty250 = tensor.empty() : tensor<1x156x24x24xf16>
  %relu251 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv249 : tensor<1x156x24x24xf16>)
    outs(%empty250 : tensor<1x156x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x156x24x24xf16>

  // conv47: 1x1 s1 p0 156->936 24x24
  %init252 = tensor.empty() : tensor<1x936x24x24xf16>
  %fill253 = linalg.fill ins(%cst : f16) outs(%init252 : tensor<1x936x24x24xf16>) -> tensor<1x936x24x24xf16>
  %conv254 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu251, %w47 : tensor<1x156x24x24xf16>, tensor<936x156x1x1xf16>)
    outs(%fill253 : tensor<1x936x24x24xf16>) -> tensor<1x936x24x24xf16>
  %empty255 = tensor.empty() : tensor<1x936x24x24xf16>
  %relu256 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv254 : tensor<1x936x24x24xf16>)
    outs(%empty255 : tensor<1x936x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x936x24x24xf16>

  // conv48: 3x3 s1 p1 936->936 24x24
  %pad257 = tensor.pad %relu256 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x936x24x24xf16> to tensor<1x936x26x26xf16>
  %init258 = tensor.empty() : tensor<1x936x24x24xf16>
  %fill259 = linalg.fill ins(%cst : f16) outs(%init258 : tensor<1x936x24x24xf16>) -> tensor<1x936x24x24xf16>
  %conv260 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad257, %w48 : tensor<1x936x26x26xf16>, tensor<936x936x3x3xf16>)
    outs(%fill259 : tensor<1x936x24x24xf16>) -> tensor<1x936x24x24xf16>
  %empty261 = tensor.empty() : tensor<1x936x24x24xf16>
  %relu262 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv260 : tensor<1x936x24x24xf16>)
    outs(%empty261 : tensor<1x936x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x936x24x24xf16>

  // conv49: 1x1 s1 p0 936->156 24x24
  %init263 = tensor.empty() : tensor<1x156x24x24xf16>
  %fill264 = linalg.fill ins(%cst : f16) outs(%init263 : tensor<1x156x24x24xf16>) -> tensor<1x156x24x24xf16>
  %conv265 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu262, %w49 : tensor<1x936x24x24xf16>, tensor<156x936x1x1xf16>)
    outs(%fill264 : tensor<1x156x24x24xf16>) -> tensor<1x156x24x24xf16>
  %empty266 = tensor.empty() : tensor<1x156x24x24xf16>
  %relu267 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv265 : tensor<1x156x24x24xf16>)
    outs(%empty266 : tensor<1x156x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x156x24x24xf16>

  // conv50: 1x1 s1 p0 156->936 24x24
  %init268 = tensor.empty() : tensor<1x936x24x24xf16>
  %fill269 = linalg.fill ins(%cst : f16) outs(%init268 : tensor<1x936x24x24xf16>) -> tensor<1x936x24x24xf16>
  %conv270 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu267, %w50 : tensor<1x156x24x24xf16>, tensor<936x156x1x1xf16>)
    outs(%fill269 : tensor<1x936x24x24xf16>) -> tensor<1x936x24x24xf16>
  %empty271 = tensor.empty() : tensor<1x936x24x24xf16>
  %relu272 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv270 : tensor<1x936x24x24xf16>)
    outs(%empty271 : tensor<1x936x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x936x24x24xf16>

  // conv51: 3x3 s1 p1 936->936 24x24
  %pad273 = tensor.pad %relu272 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x936x24x24xf16> to tensor<1x936x26x26xf16>
  %init274 = tensor.empty() : tensor<1x936x24x24xf16>
  %fill275 = linalg.fill ins(%cst : f16) outs(%init274 : tensor<1x936x24x24xf16>) -> tensor<1x936x24x24xf16>
  %conv276 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad273, %w51 : tensor<1x936x26x26xf16>, tensor<936x936x3x3xf16>)
    outs(%fill275 : tensor<1x936x24x24xf16>) -> tensor<1x936x24x24xf16>
  %empty277 = tensor.empty() : tensor<1x936x24x24xf16>
  %relu278 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv276 : tensor<1x936x24x24xf16>)
    outs(%empty277 : tensor<1x936x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x936x24x24xf16>

  // conv52: 1x1 s1 p0 936->156 24x24
  %init279 = tensor.empty() : tensor<1x156x24x24xf16>
  %fill280 = linalg.fill ins(%cst : f16) outs(%init279 : tensor<1x156x24x24xf16>) -> tensor<1x156x24x24xf16>
  %conv281 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu278, %w52 : tensor<1x936x24x24xf16>, tensor<156x936x1x1xf16>)
    outs(%fill280 : tensor<1x156x24x24xf16>) -> tensor<1x156x24x24xf16>
  %empty282 = tensor.empty() : tensor<1x156x24x24xf16>
  %relu283 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv281 : tensor<1x156x24x24xf16>)
    outs(%empty282 : tensor<1x156x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x156x24x24xf16>

  // conv53: 1x1 s1 p0 156->936 24x24
  %init284 = tensor.empty() : tensor<1x936x24x24xf16>
  %fill285 = linalg.fill ins(%cst : f16) outs(%init284 : tensor<1x936x24x24xf16>) -> tensor<1x936x24x24xf16>
  %conv286 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu283, %w53 : tensor<1x156x24x24xf16>, tensor<936x156x1x1xf16>)
    outs(%fill285 : tensor<1x936x24x24xf16>) -> tensor<1x936x24x24xf16>
  %empty287 = tensor.empty() : tensor<1x936x24x24xf16>
  %relu288 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv286 : tensor<1x936x24x24xf16>)
    outs(%empty287 : tensor<1x936x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x936x24x24xf16>

  // conv54: 3x3 s1 p1 936->936 24x24
  %pad289 = tensor.pad %relu288 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x936x24x24xf16> to tensor<1x936x26x26xf16>
  %init290 = tensor.empty() : tensor<1x936x24x24xf16>
  %fill291 = linalg.fill ins(%cst : f16) outs(%init290 : tensor<1x936x24x24xf16>) -> tensor<1x936x24x24xf16>
  %conv292 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad289, %w54 : tensor<1x936x26x26xf16>, tensor<936x936x3x3xf16>)
    outs(%fill291 : tensor<1x936x24x24xf16>) -> tensor<1x936x24x24xf16>
  %empty293 = tensor.empty() : tensor<1x936x24x24xf16>
  %relu294 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv292 : tensor<1x936x24x24xf16>)
    outs(%empty293 : tensor<1x936x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x936x24x24xf16>

  // conv55: 1x1 s1 p0 936->156 24x24
  %init295 = tensor.empty() : tensor<1x156x24x24xf16>
  %fill296 = linalg.fill ins(%cst : f16) outs(%init295 : tensor<1x156x24x24xf16>) -> tensor<1x156x24x24xf16>
  %conv297 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu294, %w55 : tensor<1x936x24x24xf16>, tensor<156x936x1x1xf16>)
    outs(%fill296 : tensor<1x156x24x24xf16>) -> tensor<1x156x24x24xf16>
  %empty298 = tensor.empty() : tensor<1x156x24x24xf16>
  %relu299 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv297 : tensor<1x156x24x24xf16>)
    outs(%empty298 : tensor<1x156x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x156x24x24xf16>

  // conv56: 1x1 s1 p0 156->936 24x24
  %init300 = tensor.empty() : tensor<1x936x24x24xf16>
  %fill301 = linalg.fill ins(%cst : f16) outs(%init300 : tensor<1x936x24x24xf16>) -> tensor<1x936x24x24xf16>
  %conv302 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu299, %w56 : tensor<1x156x24x24xf16>, tensor<936x156x1x1xf16>)
    outs(%fill301 : tensor<1x936x24x24xf16>) -> tensor<1x936x24x24xf16>
  %empty303 = tensor.empty() : tensor<1x936x24x24xf16>
  %relu304 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv302 : tensor<1x936x24x24xf16>)
    outs(%empty303 : tensor<1x936x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x936x24x24xf16>

  // conv57: 3x3 s1 p1 936->936 24x24
  %pad305 = tensor.pad %relu304 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x936x24x24xf16> to tensor<1x936x26x26xf16>
  %init306 = tensor.empty() : tensor<1x936x24x24xf16>
  %fill307 = linalg.fill ins(%cst : f16) outs(%init306 : tensor<1x936x24x24xf16>) -> tensor<1x936x24x24xf16>
  %conv308 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad305, %w57 : tensor<1x936x26x26xf16>, tensor<936x936x3x3xf16>)
    outs(%fill307 : tensor<1x936x24x24xf16>) -> tensor<1x936x24x24xf16>
  %empty309 = tensor.empty() : tensor<1x936x24x24xf16>
  %relu310 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv308 : tensor<1x936x24x24xf16>)
    outs(%empty309 : tensor<1x936x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x936x24x24xf16>

  // conv58: 1x1 s1 p0 936->156 24x24
  %init311 = tensor.empty() : tensor<1x156x24x24xf16>
  %fill312 = linalg.fill ins(%cst : f16) outs(%init311 : tensor<1x156x24x24xf16>) -> tensor<1x156x24x24xf16>
  %conv313 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu310, %w58 : tensor<1x936x24x24xf16>, tensor<156x936x1x1xf16>)
    outs(%fill312 : tensor<1x156x24x24xf16>) -> tensor<1x156x24x24xf16>
  %empty314 = tensor.empty() : tensor<1x156x24x24xf16>
  %relu315 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv313 : tensor<1x156x24x24xf16>)
    outs(%empty314 : tensor<1x156x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x156x24x24xf16>

  // conv59: 1x1 s1 p0 156->936 24x24
  %init316 = tensor.empty() : tensor<1x936x24x24xf16>
  %fill317 = linalg.fill ins(%cst : f16) outs(%init316 : tensor<1x936x24x24xf16>) -> tensor<1x936x24x24xf16>
  %conv318 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu315, %w59 : tensor<1x156x24x24xf16>, tensor<936x156x1x1xf16>)
    outs(%fill317 : tensor<1x936x24x24xf16>) -> tensor<1x936x24x24xf16>
  %empty319 = tensor.empty() : tensor<1x936x24x24xf16>
  %relu320 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv318 : tensor<1x936x24x24xf16>)
    outs(%empty319 : tensor<1x936x24x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x936x24x24xf16>

  // conv60: 3x3 s2 p1 936->936 24x24
  %pad321 = tensor.pad %relu320 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x936x24x24xf16> to tensor<1x936x26x26xf16>
  %init322 = tensor.empty() : tensor<1x936x12x12xf16>
  %fill323 = linalg.fill ins(%cst : f16) outs(%init322 : tensor<1x936x12x12xf16>) -> tensor<1x936x12x12xf16>
  %conv324 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad321, %w60 : tensor<1x936x26x26xf16>, tensor<936x936x3x3xf16>)
    outs(%fill323 : tensor<1x936x12x12xf16>) -> tensor<1x936x12x12xf16>
  %empty325 = tensor.empty() : tensor<1x936x12x12xf16>
  %relu326 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv324 : tensor<1x936x12x12xf16>)
    outs(%empty325 : tensor<1x936x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x936x12x12xf16>

  // conv61: 1x1 s1 p0 936->268 12x12
  %init327 = tensor.empty() : tensor<1x268x12x12xf16>
  %fill328 = linalg.fill ins(%cst : f16) outs(%init327 : tensor<1x268x12x12xf16>) -> tensor<1x268x12x12xf16>
  %conv329 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu326, %w61 : tensor<1x936x12x12xf16>, tensor<268x936x1x1xf16>)
    outs(%fill328 : tensor<1x268x12x12xf16>) -> tensor<1x268x12x12xf16>
  %empty330 = tensor.empty() : tensor<1x268x12x12xf16>
  %relu331 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv329 : tensor<1x268x12x12xf16>)
    outs(%empty330 : tensor<1x268x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x268x12x12xf16>

  // conv62: 1x1 s1 p0 268->1608 12x12
  %init332 = tensor.empty() : tensor<1x1608x12x12xf16>
  %fill333 = linalg.fill ins(%cst : f16) outs(%init332 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %conv334 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu331, %w62 : tensor<1x268x12x12xf16>, tensor<1608x268x1x1xf16>)
    outs(%fill333 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %empty335 = tensor.empty() : tensor<1x1608x12x12xf16>
  %relu336 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv334 : tensor<1x1608x12x12xf16>)
    outs(%empty335 : tensor<1x1608x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1608x12x12xf16>

  // conv63: 3x3 s1 p1 1608->1608 12x12
  %pad337 = tensor.pad %relu336 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1608x12x12xf16> to tensor<1x1608x14x14xf16>
  %init338 = tensor.empty() : tensor<1x1608x12x12xf16>
  %fill339 = linalg.fill ins(%cst : f16) outs(%init338 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %conv340 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad337, %w63 : tensor<1x1608x14x14xf16>, tensor<1608x1608x3x3xf16>)
    outs(%fill339 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %empty341 = tensor.empty() : tensor<1x1608x12x12xf16>
  %relu342 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv340 : tensor<1x1608x12x12xf16>)
    outs(%empty341 : tensor<1x1608x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1608x12x12xf16>

  // conv64: 1x1 s1 p0 1608->268 12x12
  %init343 = tensor.empty() : tensor<1x268x12x12xf16>
  %fill344 = linalg.fill ins(%cst : f16) outs(%init343 : tensor<1x268x12x12xf16>) -> tensor<1x268x12x12xf16>
  %conv345 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu342, %w64 : tensor<1x1608x12x12xf16>, tensor<268x1608x1x1xf16>)
    outs(%fill344 : tensor<1x268x12x12xf16>) -> tensor<1x268x12x12xf16>
  %empty346 = tensor.empty() : tensor<1x268x12x12xf16>
  %relu347 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv345 : tensor<1x268x12x12xf16>)
    outs(%empty346 : tensor<1x268x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x268x12x12xf16>

  // conv65: 1x1 s1 p0 268->1608 12x12
  %init348 = tensor.empty() : tensor<1x1608x12x12xf16>
  %fill349 = linalg.fill ins(%cst : f16) outs(%init348 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %conv350 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu347, %w65 : tensor<1x268x12x12xf16>, tensor<1608x268x1x1xf16>)
    outs(%fill349 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %empty351 = tensor.empty() : tensor<1x1608x12x12xf16>
  %relu352 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv350 : tensor<1x1608x12x12xf16>)
    outs(%empty351 : tensor<1x1608x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1608x12x12xf16>

  // conv66: 3x3 s1 p1 1608->1608 12x12
  %pad353 = tensor.pad %relu352 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1608x12x12xf16> to tensor<1x1608x14x14xf16>
  %init354 = tensor.empty() : tensor<1x1608x12x12xf16>
  %fill355 = linalg.fill ins(%cst : f16) outs(%init354 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %conv356 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad353, %w66 : tensor<1x1608x14x14xf16>, tensor<1608x1608x3x3xf16>)
    outs(%fill355 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %empty357 = tensor.empty() : tensor<1x1608x12x12xf16>
  %relu358 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv356 : tensor<1x1608x12x12xf16>)
    outs(%empty357 : tensor<1x1608x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1608x12x12xf16>

  // conv67: 1x1 s1 p0 1608->268 12x12
  %init359 = tensor.empty() : tensor<1x268x12x12xf16>
  %fill360 = linalg.fill ins(%cst : f16) outs(%init359 : tensor<1x268x12x12xf16>) -> tensor<1x268x12x12xf16>
  %conv361 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu358, %w67 : tensor<1x1608x12x12xf16>, tensor<268x1608x1x1xf16>)
    outs(%fill360 : tensor<1x268x12x12xf16>) -> tensor<1x268x12x12xf16>
  %empty362 = tensor.empty() : tensor<1x268x12x12xf16>
  %relu363 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv361 : tensor<1x268x12x12xf16>)
    outs(%empty362 : tensor<1x268x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x268x12x12xf16>

  // conv68: 1x1 s1 p0 268->1608 12x12
  %init364 = tensor.empty() : tensor<1x1608x12x12xf16>
  %fill365 = linalg.fill ins(%cst : f16) outs(%init364 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %conv366 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu363, %w68 : tensor<1x268x12x12xf16>, tensor<1608x268x1x1xf16>)
    outs(%fill365 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %empty367 = tensor.empty() : tensor<1x1608x12x12xf16>
  %relu368 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv366 : tensor<1x1608x12x12xf16>)
    outs(%empty367 : tensor<1x1608x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1608x12x12xf16>

  // conv69: 3x3 s1 p1 1608->1608 12x12
  %pad369 = tensor.pad %relu368 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1608x12x12xf16> to tensor<1x1608x14x14xf16>
  %init370 = tensor.empty() : tensor<1x1608x12x12xf16>
  %fill371 = linalg.fill ins(%cst : f16) outs(%init370 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %conv372 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad369, %w69 : tensor<1x1608x14x14xf16>, tensor<1608x1608x3x3xf16>)
    outs(%fill371 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %empty373 = tensor.empty() : tensor<1x1608x12x12xf16>
  %relu374 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv372 : tensor<1x1608x12x12xf16>)
    outs(%empty373 : tensor<1x1608x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1608x12x12xf16>

  // conv70: 1x1 s1 p0 1608->268 12x12
  %init375 = tensor.empty() : tensor<1x268x12x12xf16>
  %fill376 = linalg.fill ins(%cst : f16) outs(%init375 : tensor<1x268x12x12xf16>) -> tensor<1x268x12x12xf16>
  %conv377 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu374, %w70 : tensor<1x1608x12x12xf16>, tensor<268x1608x1x1xf16>)
    outs(%fill376 : tensor<1x268x12x12xf16>) -> tensor<1x268x12x12xf16>
  %empty378 = tensor.empty() : tensor<1x268x12x12xf16>
  %relu379 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv377 : tensor<1x268x12x12xf16>)
    outs(%empty378 : tensor<1x268x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x268x12x12xf16>

  // conv71: 1x1 s1 p0 268->1608 12x12
  %init380 = tensor.empty() : tensor<1x1608x12x12xf16>
  %fill381 = linalg.fill ins(%cst : f16) outs(%init380 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %conv382 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu379, %w71 : tensor<1x268x12x12xf16>, tensor<1608x268x1x1xf16>)
    outs(%fill381 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %empty383 = tensor.empty() : tensor<1x1608x12x12xf16>
  %relu384 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv382 : tensor<1x1608x12x12xf16>)
    outs(%empty383 : tensor<1x1608x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1608x12x12xf16>

  // conv72: 3x3 s1 p1 1608->1608 12x12
  %pad385 = tensor.pad %relu384 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1608x12x12xf16> to tensor<1x1608x14x14xf16>
  %init386 = tensor.empty() : tensor<1x1608x12x12xf16>
  %fill387 = linalg.fill ins(%cst : f16) outs(%init386 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %conv388 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad385, %w72 : tensor<1x1608x14x14xf16>, tensor<1608x1608x3x3xf16>)
    outs(%fill387 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %empty389 = tensor.empty() : tensor<1x1608x12x12xf16>
  %relu390 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv388 : tensor<1x1608x12x12xf16>)
    outs(%empty389 : tensor<1x1608x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1608x12x12xf16>

  // conv73: 1x1 s1 p0 1608->268 12x12
  %init391 = tensor.empty() : tensor<1x268x12x12xf16>
  %fill392 = linalg.fill ins(%cst : f16) outs(%init391 : tensor<1x268x12x12xf16>) -> tensor<1x268x12x12xf16>
  %conv393 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu390, %w73 : tensor<1x1608x12x12xf16>, tensor<268x1608x1x1xf16>)
    outs(%fill392 : tensor<1x268x12x12xf16>) -> tensor<1x268x12x12xf16>
  %empty394 = tensor.empty() : tensor<1x268x12x12xf16>
  %relu395 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv393 : tensor<1x268x12x12xf16>)
    outs(%empty394 : tensor<1x268x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x268x12x12xf16>

  // conv74: 1x1 s1 p0 268->1608 12x12
  %init396 = tensor.empty() : tensor<1x1608x12x12xf16>
  %fill397 = linalg.fill ins(%cst : f16) outs(%init396 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %conv398 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu395, %w74 : tensor<1x268x12x12xf16>, tensor<1608x268x1x1xf16>)
    outs(%fill397 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %empty399 = tensor.empty() : tensor<1x1608x12x12xf16>
  %relu400 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv398 : tensor<1x1608x12x12xf16>)
    outs(%empty399 : tensor<1x1608x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1608x12x12xf16>

  // conv75: 3x3 s1 p1 1608->1608 12x12
  %pad401 = tensor.pad %relu400 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1608x12x12xf16> to tensor<1x1608x14x14xf16>
  %init402 = tensor.empty() : tensor<1x1608x12x12xf16>
  %fill403 = linalg.fill ins(%cst : f16) outs(%init402 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %conv404 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad401, %w75 : tensor<1x1608x14x14xf16>, tensor<1608x1608x3x3xf16>)
    outs(%fill403 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %empty405 = tensor.empty() : tensor<1x1608x12x12xf16>
  %relu406 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv404 : tensor<1x1608x12x12xf16>)
    outs(%empty405 : tensor<1x1608x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1608x12x12xf16>

  // conv76: 1x1 s1 p0 1608->268 12x12
  %init407 = tensor.empty() : tensor<1x268x12x12xf16>
  %fill408 = linalg.fill ins(%cst : f16) outs(%init407 : tensor<1x268x12x12xf16>) -> tensor<1x268x12x12xf16>
  %conv409 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu406, %w76 : tensor<1x1608x12x12xf16>, tensor<268x1608x1x1xf16>)
    outs(%fill408 : tensor<1x268x12x12xf16>) -> tensor<1x268x12x12xf16>
  %empty410 = tensor.empty() : tensor<1x268x12x12xf16>
  %relu411 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv409 : tensor<1x268x12x12xf16>)
    outs(%empty410 : tensor<1x268x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x268x12x12xf16>

  // conv77: 1x1 s1 p0 268->1608 12x12
  %init412 = tensor.empty() : tensor<1x1608x12x12xf16>
  %fill413 = linalg.fill ins(%cst : f16) outs(%init412 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %conv414 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu411, %w77 : tensor<1x268x12x12xf16>, tensor<1608x268x1x1xf16>)
    outs(%fill413 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %empty415 = tensor.empty() : tensor<1x1608x12x12xf16>
  %relu416 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv414 : tensor<1x1608x12x12xf16>)
    outs(%empty415 : tensor<1x1608x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1608x12x12xf16>

  // conv78: 3x3 s1 p1 1608->1608 12x12
  %pad417 = tensor.pad %relu416 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1608x12x12xf16> to tensor<1x1608x14x14xf16>
  %init418 = tensor.empty() : tensor<1x1608x12x12xf16>
  %fill419 = linalg.fill ins(%cst : f16) outs(%init418 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %conv420 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad417, %w78 : tensor<1x1608x14x14xf16>, tensor<1608x1608x3x3xf16>)
    outs(%fill419 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %empty421 = tensor.empty() : tensor<1x1608x12x12xf16>
  %relu422 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv420 : tensor<1x1608x12x12xf16>)
    outs(%empty421 : tensor<1x1608x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1608x12x12xf16>

  // conv79: 1x1 s1 p0 1608->268 12x12
  %init423 = tensor.empty() : tensor<1x268x12x12xf16>
  %fill424 = linalg.fill ins(%cst : f16) outs(%init423 : tensor<1x268x12x12xf16>) -> tensor<1x268x12x12xf16>
  %conv425 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu422, %w79 : tensor<1x1608x12x12xf16>, tensor<268x1608x1x1xf16>)
    outs(%fill424 : tensor<1x268x12x12xf16>) -> tensor<1x268x12x12xf16>
  %empty426 = tensor.empty() : tensor<1x268x12x12xf16>
  %relu427 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv425 : tensor<1x268x12x12xf16>)
    outs(%empty426 : tensor<1x268x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x268x12x12xf16>

  // conv80: 1x1 s1 p0 268->1608 12x12
  %init428 = tensor.empty() : tensor<1x1608x12x12xf16>
  %fill429 = linalg.fill ins(%cst : f16) outs(%init428 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %conv430 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu427, %w80 : tensor<1x268x12x12xf16>, tensor<1608x268x1x1xf16>)
    outs(%fill429 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %empty431 = tensor.empty() : tensor<1x1608x12x12xf16>
  %relu432 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv430 : tensor<1x1608x12x12xf16>)
    outs(%empty431 : tensor<1x1608x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1608x12x12xf16>

  // conv81: 3x3 s1 p1 1608->1608 12x12
  %pad433 = tensor.pad %relu432 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1608x12x12xf16> to tensor<1x1608x14x14xf16>
  %init434 = tensor.empty() : tensor<1x1608x12x12xf16>
  %fill435 = linalg.fill ins(%cst : f16) outs(%init434 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %conv436 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad433, %w81 : tensor<1x1608x14x14xf16>, tensor<1608x1608x3x3xf16>)
    outs(%fill435 : tensor<1x1608x12x12xf16>) -> tensor<1x1608x12x12xf16>
  %empty437 = tensor.empty() : tensor<1x1608x12x12xf16>
  %relu438 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv436 : tensor<1x1608x12x12xf16>)
    outs(%empty437 : tensor<1x1608x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1608x12x12xf16>

  // conv82: 1x1 s1 p0 1608->448 12x12
  %init439 = tensor.empty() : tensor<1x448x12x12xf16>
  %fill440 = linalg.fill ins(%cst : f16) outs(%init439 : tensor<1x448x12x12xf16>) -> tensor<1x448x12x12xf16>
  %conv441 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu438, %w82 : tensor<1x1608x12x12xf16>, tensor<448x1608x1x1xf16>)
    outs(%fill440 : tensor<1x448x12x12xf16>) -> tensor<1x448x12x12xf16>
  %empty442 = tensor.empty() : tensor<1x448x12x12xf16>
  %relu443 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv441 : tensor<1x448x12x12xf16>)
    outs(%empty442 : tensor<1x448x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x448x12x12xf16>

  // conv83: 1x1 s1 p0 448->2688 12x12
  %init444 = tensor.empty() : tensor<1x2688x12x12xf16>
  %fill445 = linalg.fill ins(%cst : f16) outs(%init444 : tensor<1x2688x12x12xf16>) -> tensor<1x2688x12x12xf16>
  %conv446 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu443, %w83 : tensor<1x448x12x12xf16>, tensor<2688x448x1x1xf16>)
    outs(%fill445 : tensor<1x2688x12x12xf16>) -> tensor<1x2688x12x12xf16>
  %empty447 = tensor.empty() : tensor<1x2688x12x12xf16>
  %relu448 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv446 : tensor<1x2688x12x12xf16>)
    outs(%empty447 : tensor<1x2688x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2688x12x12xf16>

  // conv84: 3x3 s1 p1 2688->2688 12x12
  %pad449 = tensor.pad %relu448 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x2688x12x12xf16> to tensor<1x2688x14x14xf16>
  %init450 = tensor.empty() : tensor<1x2688x12x12xf16>
  %fill451 = linalg.fill ins(%cst : f16) outs(%init450 : tensor<1x2688x12x12xf16>) -> tensor<1x2688x12x12xf16>
  %conv452 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad449, %w84 : tensor<1x2688x14x14xf16>, tensor<2688x2688x3x3xf16>)
    outs(%fill451 : tensor<1x2688x12x12xf16>) -> tensor<1x2688x12x12xf16>
  %empty453 = tensor.empty() : tensor<1x2688x12x12xf16>
  %relu454 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv452 : tensor<1x2688x12x12xf16>)
    outs(%empty453 : tensor<1x2688x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2688x12x12xf16>

  // conv85: 1x1 s1 p0 2688->448 12x12
  %init455 = tensor.empty() : tensor<1x448x12x12xf16>
  %fill456 = linalg.fill ins(%cst : f16) outs(%init455 : tensor<1x448x12x12xf16>) -> tensor<1x448x12x12xf16>
  %conv457 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu454, %w85 : tensor<1x2688x12x12xf16>, tensor<448x2688x1x1xf16>)
    outs(%fill456 : tensor<1x448x12x12xf16>) -> tensor<1x448x12x12xf16>
  %empty458 = tensor.empty() : tensor<1x448x12x12xf16>
  %relu459 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv457 : tensor<1x448x12x12xf16>)
    outs(%empty458 : tensor<1x448x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x448x12x12xf16>

  // conv86: 1x1 s1 p0 448->1792 12x12
  %init460 = tensor.empty() : tensor<1x1792x12x12xf16>
  %fill461 = linalg.fill ins(%cst : f16) outs(%init460 : tensor<1x1792x12x12xf16>) -> tensor<1x1792x12x12xf16>
  %conv462 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu459, %w86 : tensor<1x448x12x12xf16>, tensor<1792x448x1x1xf16>)
    outs(%fill461 : tensor<1x1792x12x12xf16>) -> tensor<1x1792x12x12xf16>
  %empty463 = tensor.empty() : tensor<1x1792x12x12xf16>
  %relu464 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv462 : tensor<1x1792x12x12xf16>)
    outs(%empty463 : tensor<1x1792x12x12xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1792x12x12xf16>

  // FC as 1x1 conv: 1792->1000
  %init465 = tensor.empty() : tensor<1x1000x12x12xf16>
  %fill466 = linalg.fill ins(%cst : f16) outs(%init465 : tensor<1x1000x12x12xf16>) -> tensor<1x1000x12x12xf16>
  %conv467 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu464, %w_fc : tensor<1x1792x12x12xf16>, tensor<1000x1792x1x1xf16>)
    outs(%fill466 : tensor<1x1000x12x12xf16>) -> tensor<1x1000x12x12xf16>
  return %conv467 : tensor<1x1000x12x12xf16>
}
