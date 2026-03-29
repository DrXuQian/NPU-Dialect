func.func @efficientnet_b6(
    %input: tensor<1x3x528x528xf16>,
    %w0: tensor<57x3x3x3xf16>,
    %w1: tensor<57x57x3x3xf16>,
    %w2: tensor<28x57x1x1xf16>,
    %w3: tensor<28x28x3x3xf16>,
    %w4: tensor<28x28x1x1xf16>,
    %w5: tensor<28x28x3x3xf16>,
    %w6: tensor<28x28x1x1xf16>,
    %w7: tensor<168x28x1x1xf16>,
    %w8: tensor<168x168x3x3xf16>,
    %w9: tensor<43x168x1x1xf16>,
    %w10: tensor<258x43x1x1xf16>,
    %w11: tensor<258x258x3x3xf16>,
    %w12: tensor<43x258x1x1xf16>,
    %w13: tensor<258x43x1x1xf16>,
    %w14: tensor<258x258x3x3xf16>,
    %w15: tensor<43x258x1x1xf16>,
    %w16: tensor<258x43x1x1xf16>,
    %w17: tensor<258x258x3x3xf16>,
    %w18: tensor<43x258x1x1xf16>,
    %w19: tensor<258x43x1x1xf16>,
    %w20: tensor<258x258x3x3xf16>,
    %w21: tensor<43x258x1x1xf16>,
    %w22: tensor<258x43x1x1xf16>,
    %w23: tensor<258x258x3x3xf16>,
    %w24: tensor<72x258x1x1xf16>,
    %w25: tensor<432x72x1x1xf16>,
    %w26: tensor<432x432x3x3xf16>,
    %w27: tensor<72x432x1x1xf16>,
    %w28: tensor<432x72x1x1xf16>,
    %w29: tensor<432x432x3x3xf16>,
    %w30: tensor<72x432x1x1xf16>,
    %w31: tensor<432x72x1x1xf16>,
    %w32: tensor<432x432x3x3xf16>,
    %w33: tensor<72x432x1x1xf16>,
    %w34: tensor<432x72x1x1xf16>,
    %w35: tensor<432x432x3x3xf16>,
    %w36: tensor<72x432x1x1xf16>,
    %w37: tensor<432x72x1x1xf16>,
    %w38: tensor<432x432x3x3xf16>,
    %w39: tensor<144x432x1x1xf16>,
    %w40: tensor<864x144x1x1xf16>,
    %w41: tensor<864x864x3x3xf16>,
    %w42: tensor<144x864x1x1xf16>,
    %w43: tensor<864x144x1x1xf16>,
    %w44: tensor<864x864x3x3xf16>,
    %w45: tensor<144x864x1x1xf16>,
    %w46: tensor<864x144x1x1xf16>,
    %w47: tensor<864x864x3x3xf16>,
    %w48: tensor<144x864x1x1xf16>,
    %w49: tensor<864x144x1x1xf16>,
    %w50: tensor<864x864x3x3xf16>,
    %w51: tensor<144x864x1x1xf16>,
    %w52: tensor<864x144x1x1xf16>,
    %w53: tensor<864x864x3x3xf16>,
    %w54: tensor<144x864x1x1xf16>,
    %w55: tensor<864x144x1x1xf16>,
    %w56: tensor<864x864x3x3xf16>,
    %w57: tensor<144x864x1x1xf16>,
    %w58: tensor<864x144x1x1xf16>,
    %w59: tensor<864x864x3x3xf16>,
    %w60: tensor<144x864x1x1xf16>,
    %w61: tensor<864x144x1x1xf16>,
    %w62: tensor<864x864x3x3xf16>,
    %w63: tensor<201x864x1x1xf16>,
    %w64: tensor<1206x201x1x1xf16>,
    %w65: tensor<1206x1206x3x3xf16>,
    %w66: tensor<201x1206x1x1xf16>,
    %w67: tensor<1206x201x1x1xf16>,
    %w68: tensor<1206x1206x3x3xf16>,
    %w69: tensor<201x1206x1x1xf16>,
    %w70: tensor<1206x201x1x1xf16>,
    %w71: tensor<1206x1206x3x3xf16>,
    %w72: tensor<201x1206x1x1xf16>,
    %w73: tensor<1206x201x1x1xf16>,
    %w74: tensor<1206x1206x3x3xf16>,
    %w75: tensor<201x1206x1x1xf16>,
    %w76: tensor<1206x201x1x1xf16>,
    %w77: tensor<1206x1206x3x3xf16>,
    %w78: tensor<201x1206x1x1xf16>,
    %w79: tensor<1206x201x1x1xf16>,
    %w80: tensor<1206x1206x3x3xf16>,
    %w81: tensor<201x1206x1x1xf16>,
    %w82: tensor<1206x201x1x1xf16>,
    %w83: tensor<1206x1206x3x3xf16>,
    %w84: tensor<201x1206x1x1xf16>,
    %w85: tensor<1206x201x1x1xf16>,
    %w86: tensor<1206x1206x3x3xf16>,
    %w87: tensor<345x1206x1x1xf16>,
    %w88: tensor<2070x345x1x1xf16>,
    %w89: tensor<2070x2070x3x3xf16>,
    %w90: tensor<345x2070x1x1xf16>,
    %w91: tensor<2070x345x1x1xf16>,
    %w92: tensor<2070x2070x3x3xf16>,
    %w93: tensor<345x2070x1x1xf16>,
    %w94: tensor<2070x345x1x1xf16>,
    %w95: tensor<2070x2070x3x3xf16>,
    %w96: tensor<345x2070x1x1xf16>,
    %w97: tensor<2070x345x1x1xf16>,
    %w98: tensor<2070x2070x3x3xf16>,
    %w99: tensor<345x2070x1x1xf16>,
    %w100: tensor<2070x345x1x1xf16>,
    %w101: tensor<2070x2070x3x3xf16>,
    %w102: tensor<345x2070x1x1xf16>,
    %w103: tensor<2070x345x1x1xf16>,
    %w104: tensor<2070x2070x3x3xf16>,
    %w105: tensor<345x2070x1x1xf16>,
    %w106: tensor<2070x345x1x1xf16>,
    %w107: tensor<2070x2070x3x3xf16>,
    %w108: tensor<345x2070x1x1xf16>,
    %w109: tensor<2070x345x1x1xf16>,
    %w110: tensor<2070x2070x3x3xf16>,
    %w111: tensor<345x2070x1x1xf16>,
    %w112: tensor<2070x345x1x1xf16>,
    %w113: tensor<2070x2070x3x3xf16>,
    %w114: tensor<345x2070x1x1xf16>,
    %w115: tensor<2070x345x1x1xf16>,
    %w116: tensor<2070x2070x3x3xf16>,
    %w117: tensor<576x2070x1x1xf16>,
    %w118: tensor<3456x576x1x1xf16>,
    %w119: tensor<3456x3456x3x3xf16>,
    %w120: tensor<576x3456x1x1xf16>,
    %w121: tensor<3456x576x1x1xf16>,
    %w122: tensor<3456x3456x3x3xf16>,
    %w123: tensor<576x3456x1x1xf16>,
    %w124: tensor<2304x576x1x1xf16>,
    %w_fc: tensor<1000x2304x1x1xf16>) -> tensor<1x1000x17x17xf16> {
  %cst = arith.constant 0.0 : f16

  // conv0: 3x3 s2 p1 3->57 528x528
  %pad0 = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x528x528xf16> to tensor<1x3x530x530xf16>
  %init1 = tensor.empty() : tensor<1x57x264x264xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x57x264x264xf16>) -> tensor<1x57x264x264xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad0, %w0 : tensor<1x3x530x530xf16>, tensor<57x3x3x3xf16>)
    outs(%fill2 : tensor<1x57x264x264xf16>) -> tensor<1x57x264x264xf16>
  %empty4 = tensor.empty() : tensor<1x57x264x264xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x57x264x264xf16>)
    outs(%empty4 : tensor<1x57x264x264xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x57x264x264xf16>

  // conv1: 3x3 s1 p1 57->57 264x264
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x57x264x264xf16> to tensor<1x57x266x266xf16>
  %init7 = tensor.empty() : tensor<1x57x264x264xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x57x264x264xf16>) -> tensor<1x57x264x264xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad6, %w1 : tensor<1x57x266x266xf16>, tensor<57x57x3x3xf16>)
    outs(%fill8 : tensor<1x57x264x264xf16>) -> tensor<1x57x264x264xf16>
  %empty10 = tensor.empty() : tensor<1x57x264x264xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x57x264x264xf16>)
    outs(%empty10 : tensor<1x57x264x264xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x57x264x264xf16>

  // conv2: 1x1 s1 p0 57->28 264x264
  %init12 = tensor.empty() : tensor<1x28x264x264xf16>
  %fill13 = linalg.fill ins(%cst : f16) outs(%init12 : tensor<1x28x264x264xf16>) -> tensor<1x28x264x264xf16>
  %conv14 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu11, %w2 : tensor<1x57x264x264xf16>, tensor<28x57x1x1xf16>)
    outs(%fill13 : tensor<1x28x264x264xf16>) -> tensor<1x28x264x264xf16>
  %empty15 = tensor.empty() : tensor<1x28x264x264xf16>
  %relu16 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv14 : tensor<1x28x264x264xf16>)
    outs(%empty15 : tensor<1x28x264x264xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x28x264x264xf16>

  // conv3: 3x3 s1 p1 28->28 264x264
  %pad17 = tensor.pad %relu16 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x28x264x264xf16> to tensor<1x28x266x266xf16>
  %init18 = tensor.empty() : tensor<1x28x264x264xf16>
  %fill19 = linalg.fill ins(%cst : f16) outs(%init18 : tensor<1x28x264x264xf16>) -> tensor<1x28x264x264xf16>
  %conv20 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad17, %w3 : tensor<1x28x266x266xf16>, tensor<28x28x3x3xf16>)
    outs(%fill19 : tensor<1x28x264x264xf16>) -> tensor<1x28x264x264xf16>
  %empty21 = tensor.empty() : tensor<1x28x264x264xf16>
  %relu22 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv20 : tensor<1x28x264x264xf16>)
    outs(%empty21 : tensor<1x28x264x264xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x28x264x264xf16>

  // conv4: 1x1 s1 p0 28->28 264x264
  %init23 = tensor.empty() : tensor<1x28x264x264xf16>
  %fill24 = linalg.fill ins(%cst : f16) outs(%init23 : tensor<1x28x264x264xf16>) -> tensor<1x28x264x264xf16>
  %conv25 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu22, %w4 : tensor<1x28x264x264xf16>, tensor<28x28x1x1xf16>)
    outs(%fill24 : tensor<1x28x264x264xf16>) -> tensor<1x28x264x264xf16>
  %empty26 = tensor.empty() : tensor<1x28x264x264xf16>
  %relu27 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv25 : tensor<1x28x264x264xf16>)
    outs(%empty26 : tensor<1x28x264x264xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x28x264x264xf16>

  // conv5: 3x3 s1 p1 28->28 264x264
  %pad28 = tensor.pad %relu27 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x28x264x264xf16> to tensor<1x28x266x266xf16>
  %init29 = tensor.empty() : tensor<1x28x264x264xf16>
  %fill30 = linalg.fill ins(%cst : f16) outs(%init29 : tensor<1x28x264x264xf16>) -> tensor<1x28x264x264xf16>
  %conv31 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad28, %w5 : tensor<1x28x266x266xf16>, tensor<28x28x3x3xf16>)
    outs(%fill30 : tensor<1x28x264x264xf16>) -> tensor<1x28x264x264xf16>
  %empty32 = tensor.empty() : tensor<1x28x264x264xf16>
  %relu33 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv31 : tensor<1x28x264x264xf16>)
    outs(%empty32 : tensor<1x28x264x264xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x28x264x264xf16>

  // conv6: 1x1 s1 p0 28->28 264x264
  %init34 = tensor.empty() : tensor<1x28x264x264xf16>
  %fill35 = linalg.fill ins(%cst : f16) outs(%init34 : tensor<1x28x264x264xf16>) -> tensor<1x28x264x264xf16>
  %conv36 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu33, %w6 : tensor<1x28x264x264xf16>, tensor<28x28x1x1xf16>)
    outs(%fill35 : tensor<1x28x264x264xf16>) -> tensor<1x28x264x264xf16>
  %empty37 = tensor.empty() : tensor<1x28x264x264xf16>
  %relu38 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv36 : tensor<1x28x264x264xf16>)
    outs(%empty37 : tensor<1x28x264x264xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x28x264x264xf16>

  // conv7: 1x1 s1 p0 28->168 264x264
  %init39 = tensor.empty() : tensor<1x168x264x264xf16>
  %fill40 = linalg.fill ins(%cst : f16) outs(%init39 : tensor<1x168x264x264xf16>) -> tensor<1x168x264x264xf16>
  %conv41 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu38, %w7 : tensor<1x28x264x264xf16>, tensor<168x28x1x1xf16>)
    outs(%fill40 : tensor<1x168x264x264xf16>) -> tensor<1x168x264x264xf16>
  %empty42 = tensor.empty() : tensor<1x168x264x264xf16>
  %relu43 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv41 : tensor<1x168x264x264xf16>)
    outs(%empty42 : tensor<1x168x264x264xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x168x264x264xf16>

  // conv8: 3x3 s2 p1 168->168 264x264
  %pad44 = tensor.pad %relu43 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x168x264x264xf16> to tensor<1x168x266x266xf16>
  %init45 = tensor.empty() : tensor<1x168x132x132xf16>
  %fill46 = linalg.fill ins(%cst : f16) outs(%init45 : tensor<1x168x132x132xf16>) -> tensor<1x168x132x132xf16>
  %conv47 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad44, %w8 : tensor<1x168x266x266xf16>, tensor<168x168x3x3xf16>)
    outs(%fill46 : tensor<1x168x132x132xf16>) -> tensor<1x168x132x132xf16>
  %empty48 = tensor.empty() : tensor<1x168x132x132xf16>
  %relu49 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv47 : tensor<1x168x132x132xf16>)
    outs(%empty48 : tensor<1x168x132x132xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x168x132x132xf16>

  // conv9: 1x1 s1 p0 168->43 132x132
  %init50 = tensor.empty() : tensor<1x43x132x132xf16>
  %fill51 = linalg.fill ins(%cst : f16) outs(%init50 : tensor<1x43x132x132xf16>) -> tensor<1x43x132x132xf16>
  %conv52 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu49, %w9 : tensor<1x168x132x132xf16>, tensor<43x168x1x1xf16>)
    outs(%fill51 : tensor<1x43x132x132xf16>) -> tensor<1x43x132x132xf16>
  %empty53 = tensor.empty() : tensor<1x43x132x132xf16>
  %relu54 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv52 : tensor<1x43x132x132xf16>)
    outs(%empty53 : tensor<1x43x132x132xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x43x132x132xf16>

  // conv10: 1x1 s1 p0 43->258 132x132
  %init55 = tensor.empty() : tensor<1x258x132x132xf16>
  %fill56 = linalg.fill ins(%cst : f16) outs(%init55 : tensor<1x258x132x132xf16>) -> tensor<1x258x132x132xf16>
  %conv57 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu54, %w10 : tensor<1x43x132x132xf16>, tensor<258x43x1x1xf16>)
    outs(%fill56 : tensor<1x258x132x132xf16>) -> tensor<1x258x132x132xf16>
  %empty58 = tensor.empty() : tensor<1x258x132x132xf16>
  %relu59 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv57 : tensor<1x258x132x132xf16>)
    outs(%empty58 : tensor<1x258x132x132xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x258x132x132xf16>

  // conv11: 3x3 s1 p1 258->258 132x132
  %pad60 = tensor.pad %relu59 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x258x132x132xf16> to tensor<1x258x134x134xf16>
  %init61 = tensor.empty() : tensor<1x258x132x132xf16>
  %fill62 = linalg.fill ins(%cst : f16) outs(%init61 : tensor<1x258x132x132xf16>) -> tensor<1x258x132x132xf16>
  %conv63 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad60, %w11 : tensor<1x258x134x134xf16>, tensor<258x258x3x3xf16>)
    outs(%fill62 : tensor<1x258x132x132xf16>) -> tensor<1x258x132x132xf16>
  %empty64 = tensor.empty() : tensor<1x258x132x132xf16>
  %relu65 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv63 : tensor<1x258x132x132xf16>)
    outs(%empty64 : tensor<1x258x132x132xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x258x132x132xf16>

  // conv12: 1x1 s1 p0 258->43 132x132
  %init66 = tensor.empty() : tensor<1x43x132x132xf16>
  %fill67 = linalg.fill ins(%cst : f16) outs(%init66 : tensor<1x43x132x132xf16>) -> tensor<1x43x132x132xf16>
  %conv68 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu65, %w12 : tensor<1x258x132x132xf16>, tensor<43x258x1x1xf16>)
    outs(%fill67 : tensor<1x43x132x132xf16>) -> tensor<1x43x132x132xf16>
  %empty69 = tensor.empty() : tensor<1x43x132x132xf16>
  %relu70 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv68 : tensor<1x43x132x132xf16>)
    outs(%empty69 : tensor<1x43x132x132xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x43x132x132xf16>

  // conv13: 1x1 s1 p0 43->258 132x132
  %init71 = tensor.empty() : tensor<1x258x132x132xf16>
  %fill72 = linalg.fill ins(%cst : f16) outs(%init71 : tensor<1x258x132x132xf16>) -> tensor<1x258x132x132xf16>
  %conv73 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu70, %w13 : tensor<1x43x132x132xf16>, tensor<258x43x1x1xf16>)
    outs(%fill72 : tensor<1x258x132x132xf16>) -> tensor<1x258x132x132xf16>
  %empty74 = tensor.empty() : tensor<1x258x132x132xf16>
  %relu75 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv73 : tensor<1x258x132x132xf16>)
    outs(%empty74 : tensor<1x258x132x132xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x258x132x132xf16>

  // conv14: 3x3 s1 p1 258->258 132x132
  %pad76 = tensor.pad %relu75 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x258x132x132xf16> to tensor<1x258x134x134xf16>
  %init77 = tensor.empty() : tensor<1x258x132x132xf16>
  %fill78 = linalg.fill ins(%cst : f16) outs(%init77 : tensor<1x258x132x132xf16>) -> tensor<1x258x132x132xf16>
  %conv79 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad76, %w14 : tensor<1x258x134x134xf16>, tensor<258x258x3x3xf16>)
    outs(%fill78 : tensor<1x258x132x132xf16>) -> tensor<1x258x132x132xf16>
  %empty80 = tensor.empty() : tensor<1x258x132x132xf16>
  %relu81 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv79 : tensor<1x258x132x132xf16>)
    outs(%empty80 : tensor<1x258x132x132xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x258x132x132xf16>

  // conv15: 1x1 s1 p0 258->43 132x132
  %init82 = tensor.empty() : tensor<1x43x132x132xf16>
  %fill83 = linalg.fill ins(%cst : f16) outs(%init82 : tensor<1x43x132x132xf16>) -> tensor<1x43x132x132xf16>
  %conv84 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu81, %w15 : tensor<1x258x132x132xf16>, tensor<43x258x1x1xf16>)
    outs(%fill83 : tensor<1x43x132x132xf16>) -> tensor<1x43x132x132xf16>
  %empty85 = tensor.empty() : tensor<1x43x132x132xf16>
  %relu86 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv84 : tensor<1x43x132x132xf16>)
    outs(%empty85 : tensor<1x43x132x132xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x43x132x132xf16>

  // conv16: 1x1 s1 p0 43->258 132x132
  %init87 = tensor.empty() : tensor<1x258x132x132xf16>
  %fill88 = linalg.fill ins(%cst : f16) outs(%init87 : tensor<1x258x132x132xf16>) -> tensor<1x258x132x132xf16>
  %conv89 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu86, %w16 : tensor<1x43x132x132xf16>, tensor<258x43x1x1xf16>)
    outs(%fill88 : tensor<1x258x132x132xf16>) -> tensor<1x258x132x132xf16>
  %empty90 = tensor.empty() : tensor<1x258x132x132xf16>
  %relu91 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv89 : tensor<1x258x132x132xf16>)
    outs(%empty90 : tensor<1x258x132x132xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x258x132x132xf16>

  // conv17: 3x3 s1 p1 258->258 132x132
  %pad92 = tensor.pad %relu91 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x258x132x132xf16> to tensor<1x258x134x134xf16>
  %init93 = tensor.empty() : tensor<1x258x132x132xf16>
  %fill94 = linalg.fill ins(%cst : f16) outs(%init93 : tensor<1x258x132x132xf16>) -> tensor<1x258x132x132xf16>
  %conv95 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad92, %w17 : tensor<1x258x134x134xf16>, tensor<258x258x3x3xf16>)
    outs(%fill94 : tensor<1x258x132x132xf16>) -> tensor<1x258x132x132xf16>
  %empty96 = tensor.empty() : tensor<1x258x132x132xf16>
  %relu97 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv95 : tensor<1x258x132x132xf16>)
    outs(%empty96 : tensor<1x258x132x132xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x258x132x132xf16>

  // conv18: 1x1 s1 p0 258->43 132x132
  %init98 = tensor.empty() : tensor<1x43x132x132xf16>
  %fill99 = linalg.fill ins(%cst : f16) outs(%init98 : tensor<1x43x132x132xf16>) -> tensor<1x43x132x132xf16>
  %conv100 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu97, %w18 : tensor<1x258x132x132xf16>, tensor<43x258x1x1xf16>)
    outs(%fill99 : tensor<1x43x132x132xf16>) -> tensor<1x43x132x132xf16>
  %empty101 = tensor.empty() : tensor<1x43x132x132xf16>
  %relu102 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv100 : tensor<1x43x132x132xf16>)
    outs(%empty101 : tensor<1x43x132x132xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x43x132x132xf16>

  // conv19: 1x1 s1 p0 43->258 132x132
  %init103 = tensor.empty() : tensor<1x258x132x132xf16>
  %fill104 = linalg.fill ins(%cst : f16) outs(%init103 : tensor<1x258x132x132xf16>) -> tensor<1x258x132x132xf16>
  %conv105 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu102, %w19 : tensor<1x43x132x132xf16>, tensor<258x43x1x1xf16>)
    outs(%fill104 : tensor<1x258x132x132xf16>) -> tensor<1x258x132x132xf16>
  %empty106 = tensor.empty() : tensor<1x258x132x132xf16>
  %relu107 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv105 : tensor<1x258x132x132xf16>)
    outs(%empty106 : tensor<1x258x132x132xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x258x132x132xf16>

  // conv20: 3x3 s1 p1 258->258 132x132
  %pad108 = tensor.pad %relu107 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x258x132x132xf16> to tensor<1x258x134x134xf16>
  %init109 = tensor.empty() : tensor<1x258x132x132xf16>
  %fill110 = linalg.fill ins(%cst : f16) outs(%init109 : tensor<1x258x132x132xf16>) -> tensor<1x258x132x132xf16>
  %conv111 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad108, %w20 : tensor<1x258x134x134xf16>, tensor<258x258x3x3xf16>)
    outs(%fill110 : tensor<1x258x132x132xf16>) -> tensor<1x258x132x132xf16>
  %empty112 = tensor.empty() : tensor<1x258x132x132xf16>
  %relu113 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv111 : tensor<1x258x132x132xf16>)
    outs(%empty112 : tensor<1x258x132x132xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x258x132x132xf16>

  // conv21: 1x1 s1 p0 258->43 132x132
  %init114 = tensor.empty() : tensor<1x43x132x132xf16>
  %fill115 = linalg.fill ins(%cst : f16) outs(%init114 : tensor<1x43x132x132xf16>) -> tensor<1x43x132x132xf16>
  %conv116 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu113, %w21 : tensor<1x258x132x132xf16>, tensor<43x258x1x1xf16>)
    outs(%fill115 : tensor<1x43x132x132xf16>) -> tensor<1x43x132x132xf16>
  %empty117 = tensor.empty() : tensor<1x43x132x132xf16>
  %relu118 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv116 : tensor<1x43x132x132xf16>)
    outs(%empty117 : tensor<1x43x132x132xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x43x132x132xf16>

  // conv22: 1x1 s1 p0 43->258 132x132
  %init119 = tensor.empty() : tensor<1x258x132x132xf16>
  %fill120 = linalg.fill ins(%cst : f16) outs(%init119 : tensor<1x258x132x132xf16>) -> tensor<1x258x132x132xf16>
  %conv121 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu118, %w22 : tensor<1x43x132x132xf16>, tensor<258x43x1x1xf16>)
    outs(%fill120 : tensor<1x258x132x132xf16>) -> tensor<1x258x132x132xf16>
  %empty122 = tensor.empty() : tensor<1x258x132x132xf16>
  %relu123 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv121 : tensor<1x258x132x132xf16>)
    outs(%empty122 : tensor<1x258x132x132xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x258x132x132xf16>

  // conv23: 3x3 s2 p1 258->258 132x132
  %pad124 = tensor.pad %relu123 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x258x132x132xf16> to tensor<1x258x134x134xf16>
  %init125 = tensor.empty() : tensor<1x258x66x66xf16>
  %fill126 = linalg.fill ins(%cst : f16) outs(%init125 : tensor<1x258x66x66xf16>) -> tensor<1x258x66x66xf16>
  %conv127 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad124, %w23 : tensor<1x258x134x134xf16>, tensor<258x258x3x3xf16>)
    outs(%fill126 : tensor<1x258x66x66xf16>) -> tensor<1x258x66x66xf16>
  %empty128 = tensor.empty() : tensor<1x258x66x66xf16>
  %relu129 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv127 : tensor<1x258x66x66xf16>)
    outs(%empty128 : tensor<1x258x66x66xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x258x66x66xf16>

  // conv24: 1x1 s1 p0 258->72 66x66
  %init130 = tensor.empty() : tensor<1x72x66x66xf16>
  %fill131 = linalg.fill ins(%cst : f16) outs(%init130 : tensor<1x72x66x66xf16>) -> tensor<1x72x66x66xf16>
  %conv132 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu129, %w24 : tensor<1x258x66x66xf16>, tensor<72x258x1x1xf16>)
    outs(%fill131 : tensor<1x72x66x66xf16>) -> tensor<1x72x66x66xf16>
  %empty133 = tensor.empty() : tensor<1x72x66x66xf16>
  %relu134 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv132 : tensor<1x72x66x66xf16>)
    outs(%empty133 : tensor<1x72x66x66xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x66x66xf16>

  // conv25: 1x1 s1 p0 72->432 66x66
  %init135 = tensor.empty() : tensor<1x432x66x66xf16>
  %fill136 = linalg.fill ins(%cst : f16) outs(%init135 : tensor<1x432x66x66xf16>) -> tensor<1x432x66x66xf16>
  %conv137 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu134, %w25 : tensor<1x72x66x66xf16>, tensor<432x72x1x1xf16>)
    outs(%fill136 : tensor<1x432x66x66xf16>) -> tensor<1x432x66x66xf16>
  %empty138 = tensor.empty() : tensor<1x432x66x66xf16>
  %relu139 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv137 : tensor<1x432x66x66xf16>)
    outs(%empty138 : tensor<1x432x66x66xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x432x66x66xf16>

  // conv26: 3x3 s1 p1 432->432 66x66
  %pad140 = tensor.pad %relu139 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x432x66x66xf16> to tensor<1x432x68x68xf16>
  %init141 = tensor.empty() : tensor<1x432x66x66xf16>
  %fill142 = linalg.fill ins(%cst : f16) outs(%init141 : tensor<1x432x66x66xf16>) -> tensor<1x432x66x66xf16>
  %conv143 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad140, %w26 : tensor<1x432x68x68xf16>, tensor<432x432x3x3xf16>)
    outs(%fill142 : tensor<1x432x66x66xf16>) -> tensor<1x432x66x66xf16>
  %empty144 = tensor.empty() : tensor<1x432x66x66xf16>
  %relu145 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv143 : tensor<1x432x66x66xf16>)
    outs(%empty144 : tensor<1x432x66x66xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x432x66x66xf16>

  // conv27: 1x1 s1 p0 432->72 66x66
  %init146 = tensor.empty() : tensor<1x72x66x66xf16>
  %fill147 = linalg.fill ins(%cst : f16) outs(%init146 : tensor<1x72x66x66xf16>) -> tensor<1x72x66x66xf16>
  %conv148 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu145, %w27 : tensor<1x432x66x66xf16>, tensor<72x432x1x1xf16>)
    outs(%fill147 : tensor<1x72x66x66xf16>) -> tensor<1x72x66x66xf16>
  %empty149 = tensor.empty() : tensor<1x72x66x66xf16>
  %relu150 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv148 : tensor<1x72x66x66xf16>)
    outs(%empty149 : tensor<1x72x66x66xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x66x66xf16>

  // conv28: 1x1 s1 p0 72->432 66x66
  %init151 = tensor.empty() : tensor<1x432x66x66xf16>
  %fill152 = linalg.fill ins(%cst : f16) outs(%init151 : tensor<1x432x66x66xf16>) -> tensor<1x432x66x66xf16>
  %conv153 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu150, %w28 : tensor<1x72x66x66xf16>, tensor<432x72x1x1xf16>)
    outs(%fill152 : tensor<1x432x66x66xf16>) -> tensor<1x432x66x66xf16>
  %empty154 = tensor.empty() : tensor<1x432x66x66xf16>
  %relu155 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv153 : tensor<1x432x66x66xf16>)
    outs(%empty154 : tensor<1x432x66x66xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x432x66x66xf16>

  // conv29: 3x3 s1 p1 432->432 66x66
  %pad156 = tensor.pad %relu155 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x432x66x66xf16> to tensor<1x432x68x68xf16>
  %init157 = tensor.empty() : tensor<1x432x66x66xf16>
  %fill158 = linalg.fill ins(%cst : f16) outs(%init157 : tensor<1x432x66x66xf16>) -> tensor<1x432x66x66xf16>
  %conv159 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad156, %w29 : tensor<1x432x68x68xf16>, tensor<432x432x3x3xf16>)
    outs(%fill158 : tensor<1x432x66x66xf16>) -> tensor<1x432x66x66xf16>
  %empty160 = tensor.empty() : tensor<1x432x66x66xf16>
  %relu161 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv159 : tensor<1x432x66x66xf16>)
    outs(%empty160 : tensor<1x432x66x66xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x432x66x66xf16>

  // conv30: 1x1 s1 p0 432->72 66x66
  %init162 = tensor.empty() : tensor<1x72x66x66xf16>
  %fill163 = linalg.fill ins(%cst : f16) outs(%init162 : tensor<1x72x66x66xf16>) -> tensor<1x72x66x66xf16>
  %conv164 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu161, %w30 : tensor<1x432x66x66xf16>, tensor<72x432x1x1xf16>)
    outs(%fill163 : tensor<1x72x66x66xf16>) -> tensor<1x72x66x66xf16>
  %empty165 = tensor.empty() : tensor<1x72x66x66xf16>
  %relu166 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv164 : tensor<1x72x66x66xf16>)
    outs(%empty165 : tensor<1x72x66x66xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x66x66xf16>

  // conv31: 1x1 s1 p0 72->432 66x66
  %init167 = tensor.empty() : tensor<1x432x66x66xf16>
  %fill168 = linalg.fill ins(%cst : f16) outs(%init167 : tensor<1x432x66x66xf16>) -> tensor<1x432x66x66xf16>
  %conv169 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu166, %w31 : tensor<1x72x66x66xf16>, tensor<432x72x1x1xf16>)
    outs(%fill168 : tensor<1x432x66x66xf16>) -> tensor<1x432x66x66xf16>
  %empty170 = tensor.empty() : tensor<1x432x66x66xf16>
  %relu171 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv169 : tensor<1x432x66x66xf16>)
    outs(%empty170 : tensor<1x432x66x66xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x432x66x66xf16>

  // conv32: 3x3 s1 p1 432->432 66x66
  %pad172 = tensor.pad %relu171 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x432x66x66xf16> to tensor<1x432x68x68xf16>
  %init173 = tensor.empty() : tensor<1x432x66x66xf16>
  %fill174 = linalg.fill ins(%cst : f16) outs(%init173 : tensor<1x432x66x66xf16>) -> tensor<1x432x66x66xf16>
  %conv175 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad172, %w32 : tensor<1x432x68x68xf16>, tensor<432x432x3x3xf16>)
    outs(%fill174 : tensor<1x432x66x66xf16>) -> tensor<1x432x66x66xf16>
  %empty176 = tensor.empty() : tensor<1x432x66x66xf16>
  %relu177 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv175 : tensor<1x432x66x66xf16>)
    outs(%empty176 : tensor<1x432x66x66xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x432x66x66xf16>

  // conv33: 1x1 s1 p0 432->72 66x66
  %init178 = tensor.empty() : tensor<1x72x66x66xf16>
  %fill179 = linalg.fill ins(%cst : f16) outs(%init178 : tensor<1x72x66x66xf16>) -> tensor<1x72x66x66xf16>
  %conv180 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu177, %w33 : tensor<1x432x66x66xf16>, tensor<72x432x1x1xf16>)
    outs(%fill179 : tensor<1x72x66x66xf16>) -> tensor<1x72x66x66xf16>
  %empty181 = tensor.empty() : tensor<1x72x66x66xf16>
  %relu182 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv180 : tensor<1x72x66x66xf16>)
    outs(%empty181 : tensor<1x72x66x66xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x66x66xf16>

  // conv34: 1x1 s1 p0 72->432 66x66
  %init183 = tensor.empty() : tensor<1x432x66x66xf16>
  %fill184 = linalg.fill ins(%cst : f16) outs(%init183 : tensor<1x432x66x66xf16>) -> tensor<1x432x66x66xf16>
  %conv185 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu182, %w34 : tensor<1x72x66x66xf16>, tensor<432x72x1x1xf16>)
    outs(%fill184 : tensor<1x432x66x66xf16>) -> tensor<1x432x66x66xf16>
  %empty186 = tensor.empty() : tensor<1x432x66x66xf16>
  %relu187 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv185 : tensor<1x432x66x66xf16>)
    outs(%empty186 : tensor<1x432x66x66xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x432x66x66xf16>

  // conv35: 3x3 s1 p1 432->432 66x66
  %pad188 = tensor.pad %relu187 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x432x66x66xf16> to tensor<1x432x68x68xf16>
  %init189 = tensor.empty() : tensor<1x432x66x66xf16>
  %fill190 = linalg.fill ins(%cst : f16) outs(%init189 : tensor<1x432x66x66xf16>) -> tensor<1x432x66x66xf16>
  %conv191 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad188, %w35 : tensor<1x432x68x68xf16>, tensor<432x432x3x3xf16>)
    outs(%fill190 : tensor<1x432x66x66xf16>) -> tensor<1x432x66x66xf16>
  %empty192 = tensor.empty() : tensor<1x432x66x66xf16>
  %relu193 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv191 : tensor<1x432x66x66xf16>)
    outs(%empty192 : tensor<1x432x66x66xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x432x66x66xf16>

  // conv36: 1x1 s1 p0 432->72 66x66
  %init194 = tensor.empty() : tensor<1x72x66x66xf16>
  %fill195 = linalg.fill ins(%cst : f16) outs(%init194 : tensor<1x72x66x66xf16>) -> tensor<1x72x66x66xf16>
  %conv196 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu193, %w36 : tensor<1x432x66x66xf16>, tensor<72x432x1x1xf16>)
    outs(%fill195 : tensor<1x72x66x66xf16>) -> tensor<1x72x66x66xf16>
  %empty197 = tensor.empty() : tensor<1x72x66x66xf16>
  %relu198 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv196 : tensor<1x72x66x66xf16>)
    outs(%empty197 : tensor<1x72x66x66xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x72x66x66xf16>

  // conv37: 1x1 s1 p0 72->432 66x66
  %init199 = tensor.empty() : tensor<1x432x66x66xf16>
  %fill200 = linalg.fill ins(%cst : f16) outs(%init199 : tensor<1x432x66x66xf16>) -> tensor<1x432x66x66xf16>
  %conv201 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu198, %w37 : tensor<1x72x66x66xf16>, tensor<432x72x1x1xf16>)
    outs(%fill200 : tensor<1x432x66x66xf16>) -> tensor<1x432x66x66xf16>
  %empty202 = tensor.empty() : tensor<1x432x66x66xf16>
  %relu203 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv201 : tensor<1x432x66x66xf16>)
    outs(%empty202 : tensor<1x432x66x66xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x432x66x66xf16>

  // conv38: 3x3 s2 p1 432->432 66x66
  %pad204 = tensor.pad %relu203 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x432x66x66xf16> to tensor<1x432x68x68xf16>
  %init205 = tensor.empty() : tensor<1x432x33x33xf16>
  %fill206 = linalg.fill ins(%cst : f16) outs(%init205 : tensor<1x432x33x33xf16>) -> tensor<1x432x33x33xf16>
  %conv207 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad204, %w38 : tensor<1x432x68x68xf16>, tensor<432x432x3x3xf16>)
    outs(%fill206 : tensor<1x432x33x33xf16>) -> tensor<1x432x33x33xf16>
  %empty208 = tensor.empty() : tensor<1x432x33x33xf16>
  %relu209 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv207 : tensor<1x432x33x33xf16>)
    outs(%empty208 : tensor<1x432x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x432x33x33xf16>

  // conv39: 1x1 s1 p0 432->144 33x33
  %init210 = tensor.empty() : tensor<1x144x33x33xf16>
  %fill211 = linalg.fill ins(%cst : f16) outs(%init210 : tensor<1x144x33x33xf16>) -> tensor<1x144x33x33xf16>
  %conv212 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu209, %w39 : tensor<1x432x33x33xf16>, tensor<144x432x1x1xf16>)
    outs(%fill211 : tensor<1x144x33x33xf16>) -> tensor<1x144x33x33xf16>
  %empty213 = tensor.empty() : tensor<1x144x33x33xf16>
  %relu214 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv212 : tensor<1x144x33x33xf16>)
    outs(%empty213 : tensor<1x144x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x144x33x33xf16>

  // conv40: 1x1 s1 p0 144->864 33x33
  %init215 = tensor.empty() : tensor<1x864x33x33xf16>
  %fill216 = linalg.fill ins(%cst : f16) outs(%init215 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %conv217 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu214, %w40 : tensor<1x144x33x33xf16>, tensor<864x144x1x1xf16>)
    outs(%fill216 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %empty218 = tensor.empty() : tensor<1x864x33x33xf16>
  %relu219 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv217 : tensor<1x864x33x33xf16>)
    outs(%empty218 : tensor<1x864x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x864x33x33xf16>

  // conv41: 3x3 s1 p1 864->864 33x33
  %pad220 = tensor.pad %relu219 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x864x33x33xf16> to tensor<1x864x35x35xf16>
  %init221 = tensor.empty() : tensor<1x864x33x33xf16>
  %fill222 = linalg.fill ins(%cst : f16) outs(%init221 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %conv223 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad220, %w41 : tensor<1x864x35x35xf16>, tensor<864x864x3x3xf16>)
    outs(%fill222 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %empty224 = tensor.empty() : tensor<1x864x33x33xf16>
  %relu225 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv223 : tensor<1x864x33x33xf16>)
    outs(%empty224 : tensor<1x864x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x864x33x33xf16>

  // conv42: 1x1 s1 p0 864->144 33x33
  %init226 = tensor.empty() : tensor<1x144x33x33xf16>
  %fill227 = linalg.fill ins(%cst : f16) outs(%init226 : tensor<1x144x33x33xf16>) -> tensor<1x144x33x33xf16>
  %conv228 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu225, %w42 : tensor<1x864x33x33xf16>, tensor<144x864x1x1xf16>)
    outs(%fill227 : tensor<1x144x33x33xf16>) -> tensor<1x144x33x33xf16>
  %empty229 = tensor.empty() : tensor<1x144x33x33xf16>
  %relu230 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv228 : tensor<1x144x33x33xf16>)
    outs(%empty229 : tensor<1x144x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x144x33x33xf16>

  // conv43: 1x1 s1 p0 144->864 33x33
  %init231 = tensor.empty() : tensor<1x864x33x33xf16>
  %fill232 = linalg.fill ins(%cst : f16) outs(%init231 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %conv233 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu230, %w43 : tensor<1x144x33x33xf16>, tensor<864x144x1x1xf16>)
    outs(%fill232 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %empty234 = tensor.empty() : tensor<1x864x33x33xf16>
  %relu235 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv233 : tensor<1x864x33x33xf16>)
    outs(%empty234 : tensor<1x864x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x864x33x33xf16>

  // conv44: 3x3 s1 p1 864->864 33x33
  %pad236 = tensor.pad %relu235 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x864x33x33xf16> to tensor<1x864x35x35xf16>
  %init237 = tensor.empty() : tensor<1x864x33x33xf16>
  %fill238 = linalg.fill ins(%cst : f16) outs(%init237 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %conv239 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad236, %w44 : tensor<1x864x35x35xf16>, tensor<864x864x3x3xf16>)
    outs(%fill238 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %empty240 = tensor.empty() : tensor<1x864x33x33xf16>
  %relu241 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv239 : tensor<1x864x33x33xf16>)
    outs(%empty240 : tensor<1x864x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x864x33x33xf16>

  // conv45: 1x1 s1 p0 864->144 33x33
  %init242 = tensor.empty() : tensor<1x144x33x33xf16>
  %fill243 = linalg.fill ins(%cst : f16) outs(%init242 : tensor<1x144x33x33xf16>) -> tensor<1x144x33x33xf16>
  %conv244 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu241, %w45 : tensor<1x864x33x33xf16>, tensor<144x864x1x1xf16>)
    outs(%fill243 : tensor<1x144x33x33xf16>) -> tensor<1x144x33x33xf16>
  %empty245 = tensor.empty() : tensor<1x144x33x33xf16>
  %relu246 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv244 : tensor<1x144x33x33xf16>)
    outs(%empty245 : tensor<1x144x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x144x33x33xf16>

  // conv46: 1x1 s1 p0 144->864 33x33
  %init247 = tensor.empty() : tensor<1x864x33x33xf16>
  %fill248 = linalg.fill ins(%cst : f16) outs(%init247 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %conv249 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu246, %w46 : tensor<1x144x33x33xf16>, tensor<864x144x1x1xf16>)
    outs(%fill248 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %empty250 = tensor.empty() : tensor<1x864x33x33xf16>
  %relu251 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv249 : tensor<1x864x33x33xf16>)
    outs(%empty250 : tensor<1x864x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x864x33x33xf16>

  // conv47: 3x3 s1 p1 864->864 33x33
  %pad252 = tensor.pad %relu251 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x864x33x33xf16> to tensor<1x864x35x35xf16>
  %init253 = tensor.empty() : tensor<1x864x33x33xf16>
  %fill254 = linalg.fill ins(%cst : f16) outs(%init253 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %conv255 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad252, %w47 : tensor<1x864x35x35xf16>, tensor<864x864x3x3xf16>)
    outs(%fill254 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %empty256 = tensor.empty() : tensor<1x864x33x33xf16>
  %relu257 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv255 : tensor<1x864x33x33xf16>)
    outs(%empty256 : tensor<1x864x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x864x33x33xf16>

  // conv48: 1x1 s1 p0 864->144 33x33
  %init258 = tensor.empty() : tensor<1x144x33x33xf16>
  %fill259 = linalg.fill ins(%cst : f16) outs(%init258 : tensor<1x144x33x33xf16>) -> tensor<1x144x33x33xf16>
  %conv260 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu257, %w48 : tensor<1x864x33x33xf16>, tensor<144x864x1x1xf16>)
    outs(%fill259 : tensor<1x144x33x33xf16>) -> tensor<1x144x33x33xf16>
  %empty261 = tensor.empty() : tensor<1x144x33x33xf16>
  %relu262 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv260 : tensor<1x144x33x33xf16>)
    outs(%empty261 : tensor<1x144x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x144x33x33xf16>

  // conv49: 1x1 s1 p0 144->864 33x33
  %init263 = tensor.empty() : tensor<1x864x33x33xf16>
  %fill264 = linalg.fill ins(%cst : f16) outs(%init263 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %conv265 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu262, %w49 : tensor<1x144x33x33xf16>, tensor<864x144x1x1xf16>)
    outs(%fill264 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %empty266 = tensor.empty() : tensor<1x864x33x33xf16>
  %relu267 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv265 : tensor<1x864x33x33xf16>)
    outs(%empty266 : tensor<1x864x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x864x33x33xf16>

  // conv50: 3x3 s1 p1 864->864 33x33
  %pad268 = tensor.pad %relu267 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x864x33x33xf16> to tensor<1x864x35x35xf16>
  %init269 = tensor.empty() : tensor<1x864x33x33xf16>
  %fill270 = linalg.fill ins(%cst : f16) outs(%init269 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %conv271 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad268, %w50 : tensor<1x864x35x35xf16>, tensor<864x864x3x3xf16>)
    outs(%fill270 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %empty272 = tensor.empty() : tensor<1x864x33x33xf16>
  %relu273 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv271 : tensor<1x864x33x33xf16>)
    outs(%empty272 : tensor<1x864x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x864x33x33xf16>

  // conv51: 1x1 s1 p0 864->144 33x33
  %init274 = tensor.empty() : tensor<1x144x33x33xf16>
  %fill275 = linalg.fill ins(%cst : f16) outs(%init274 : tensor<1x144x33x33xf16>) -> tensor<1x144x33x33xf16>
  %conv276 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu273, %w51 : tensor<1x864x33x33xf16>, tensor<144x864x1x1xf16>)
    outs(%fill275 : tensor<1x144x33x33xf16>) -> tensor<1x144x33x33xf16>
  %empty277 = tensor.empty() : tensor<1x144x33x33xf16>
  %relu278 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv276 : tensor<1x144x33x33xf16>)
    outs(%empty277 : tensor<1x144x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x144x33x33xf16>

  // conv52: 1x1 s1 p0 144->864 33x33
  %init279 = tensor.empty() : tensor<1x864x33x33xf16>
  %fill280 = linalg.fill ins(%cst : f16) outs(%init279 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %conv281 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu278, %w52 : tensor<1x144x33x33xf16>, tensor<864x144x1x1xf16>)
    outs(%fill280 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %empty282 = tensor.empty() : tensor<1x864x33x33xf16>
  %relu283 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv281 : tensor<1x864x33x33xf16>)
    outs(%empty282 : tensor<1x864x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x864x33x33xf16>

  // conv53: 3x3 s1 p1 864->864 33x33
  %pad284 = tensor.pad %relu283 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x864x33x33xf16> to tensor<1x864x35x35xf16>
  %init285 = tensor.empty() : tensor<1x864x33x33xf16>
  %fill286 = linalg.fill ins(%cst : f16) outs(%init285 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %conv287 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad284, %w53 : tensor<1x864x35x35xf16>, tensor<864x864x3x3xf16>)
    outs(%fill286 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %empty288 = tensor.empty() : tensor<1x864x33x33xf16>
  %relu289 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv287 : tensor<1x864x33x33xf16>)
    outs(%empty288 : tensor<1x864x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x864x33x33xf16>

  // conv54: 1x1 s1 p0 864->144 33x33
  %init290 = tensor.empty() : tensor<1x144x33x33xf16>
  %fill291 = linalg.fill ins(%cst : f16) outs(%init290 : tensor<1x144x33x33xf16>) -> tensor<1x144x33x33xf16>
  %conv292 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu289, %w54 : tensor<1x864x33x33xf16>, tensor<144x864x1x1xf16>)
    outs(%fill291 : tensor<1x144x33x33xf16>) -> tensor<1x144x33x33xf16>
  %empty293 = tensor.empty() : tensor<1x144x33x33xf16>
  %relu294 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv292 : tensor<1x144x33x33xf16>)
    outs(%empty293 : tensor<1x144x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x144x33x33xf16>

  // conv55: 1x1 s1 p0 144->864 33x33
  %init295 = tensor.empty() : tensor<1x864x33x33xf16>
  %fill296 = linalg.fill ins(%cst : f16) outs(%init295 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %conv297 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu294, %w55 : tensor<1x144x33x33xf16>, tensor<864x144x1x1xf16>)
    outs(%fill296 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %empty298 = tensor.empty() : tensor<1x864x33x33xf16>
  %relu299 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv297 : tensor<1x864x33x33xf16>)
    outs(%empty298 : tensor<1x864x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x864x33x33xf16>

  // conv56: 3x3 s1 p1 864->864 33x33
  %pad300 = tensor.pad %relu299 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x864x33x33xf16> to tensor<1x864x35x35xf16>
  %init301 = tensor.empty() : tensor<1x864x33x33xf16>
  %fill302 = linalg.fill ins(%cst : f16) outs(%init301 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %conv303 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad300, %w56 : tensor<1x864x35x35xf16>, tensor<864x864x3x3xf16>)
    outs(%fill302 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %empty304 = tensor.empty() : tensor<1x864x33x33xf16>
  %relu305 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv303 : tensor<1x864x33x33xf16>)
    outs(%empty304 : tensor<1x864x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x864x33x33xf16>

  // conv57: 1x1 s1 p0 864->144 33x33
  %init306 = tensor.empty() : tensor<1x144x33x33xf16>
  %fill307 = linalg.fill ins(%cst : f16) outs(%init306 : tensor<1x144x33x33xf16>) -> tensor<1x144x33x33xf16>
  %conv308 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu305, %w57 : tensor<1x864x33x33xf16>, tensor<144x864x1x1xf16>)
    outs(%fill307 : tensor<1x144x33x33xf16>) -> tensor<1x144x33x33xf16>
  %empty309 = tensor.empty() : tensor<1x144x33x33xf16>
  %relu310 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv308 : tensor<1x144x33x33xf16>)
    outs(%empty309 : tensor<1x144x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x144x33x33xf16>

  // conv58: 1x1 s1 p0 144->864 33x33
  %init311 = tensor.empty() : tensor<1x864x33x33xf16>
  %fill312 = linalg.fill ins(%cst : f16) outs(%init311 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %conv313 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu310, %w58 : tensor<1x144x33x33xf16>, tensor<864x144x1x1xf16>)
    outs(%fill312 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %empty314 = tensor.empty() : tensor<1x864x33x33xf16>
  %relu315 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv313 : tensor<1x864x33x33xf16>)
    outs(%empty314 : tensor<1x864x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x864x33x33xf16>

  // conv59: 3x3 s1 p1 864->864 33x33
  %pad316 = tensor.pad %relu315 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x864x33x33xf16> to tensor<1x864x35x35xf16>
  %init317 = tensor.empty() : tensor<1x864x33x33xf16>
  %fill318 = linalg.fill ins(%cst : f16) outs(%init317 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %conv319 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad316, %w59 : tensor<1x864x35x35xf16>, tensor<864x864x3x3xf16>)
    outs(%fill318 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %empty320 = tensor.empty() : tensor<1x864x33x33xf16>
  %relu321 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv319 : tensor<1x864x33x33xf16>)
    outs(%empty320 : tensor<1x864x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x864x33x33xf16>

  // conv60: 1x1 s1 p0 864->144 33x33
  %init322 = tensor.empty() : tensor<1x144x33x33xf16>
  %fill323 = linalg.fill ins(%cst : f16) outs(%init322 : tensor<1x144x33x33xf16>) -> tensor<1x144x33x33xf16>
  %conv324 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu321, %w60 : tensor<1x864x33x33xf16>, tensor<144x864x1x1xf16>)
    outs(%fill323 : tensor<1x144x33x33xf16>) -> tensor<1x144x33x33xf16>
  %empty325 = tensor.empty() : tensor<1x144x33x33xf16>
  %relu326 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv324 : tensor<1x144x33x33xf16>)
    outs(%empty325 : tensor<1x144x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x144x33x33xf16>

  // conv61: 1x1 s1 p0 144->864 33x33
  %init327 = tensor.empty() : tensor<1x864x33x33xf16>
  %fill328 = linalg.fill ins(%cst : f16) outs(%init327 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %conv329 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu326, %w61 : tensor<1x144x33x33xf16>, tensor<864x144x1x1xf16>)
    outs(%fill328 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %empty330 = tensor.empty() : tensor<1x864x33x33xf16>
  %relu331 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv329 : tensor<1x864x33x33xf16>)
    outs(%empty330 : tensor<1x864x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x864x33x33xf16>

  // conv62: 3x3 s1 p1 864->864 33x33
  %pad332 = tensor.pad %relu331 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x864x33x33xf16> to tensor<1x864x35x35xf16>
  %init333 = tensor.empty() : tensor<1x864x33x33xf16>
  %fill334 = linalg.fill ins(%cst : f16) outs(%init333 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %conv335 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad332, %w62 : tensor<1x864x35x35xf16>, tensor<864x864x3x3xf16>)
    outs(%fill334 : tensor<1x864x33x33xf16>) -> tensor<1x864x33x33xf16>
  %empty336 = tensor.empty() : tensor<1x864x33x33xf16>
  %relu337 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv335 : tensor<1x864x33x33xf16>)
    outs(%empty336 : tensor<1x864x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x864x33x33xf16>

  // conv63: 1x1 s1 p0 864->201 33x33
  %init338 = tensor.empty() : tensor<1x201x33x33xf16>
  %fill339 = linalg.fill ins(%cst : f16) outs(%init338 : tensor<1x201x33x33xf16>) -> tensor<1x201x33x33xf16>
  %conv340 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu337, %w63 : tensor<1x864x33x33xf16>, tensor<201x864x1x1xf16>)
    outs(%fill339 : tensor<1x201x33x33xf16>) -> tensor<1x201x33x33xf16>
  %empty341 = tensor.empty() : tensor<1x201x33x33xf16>
  %relu342 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv340 : tensor<1x201x33x33xf16>)
    outs(%empty341 : tensor<1x201x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x201x33x33xf16>

  // conv64: 1x1 s1 p0 201->1206 33x33
  %init343 = tensor.empty() : tensor<1x1206x33x33xf16>
  %fill344 = linalg.fill ins(%cst : f16) outs(%init343 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %conv345 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu342, %w64 : tensor<1x201x33x33xf16>, tensor<1206x201x1x1xf16>)
    outs(%fill344 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %empty346 = tensor.empty() : tensor<1x1206x33x33xf16>
  %relu347 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv345 : tensor<1x1206x33x33xf16>)
    outs(%empty346 : tensor<1x1206x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1206x33x33xf16>

  // conv65: 3x3 s1 p1 1206->1206 33x33
  %pad348 = tensor.pad %relu347 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1206x33x33xf16> to tensor<1x1206x35x35xf16>
  %init349 = tensor.empty() : tensor<1x1206x33x33xf16>
  %fill350 = linalg.fill ins(%cst : f16) outs(%init349 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %conv351 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad348, %w65 : tensor<1x1206x35x35xf16>, tensor<1206x1206x3x3xf16>)
    outs(%fill350 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %empty352 = tensor.empty() : tensor<1x1206x33x33xf16>
  %relu353 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv351 : tensor<1x1206x33x33xf16>)
    outs(%empty352 : tensor<1x1206x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1206x33x33xf16>

  // conv66: 1x1 s1 p0 1206->201 33x33
  %init354 = tensor.empty() : tensor<1x201x33x33xf16>
  %fill355 = linalg.fill ins(%cst : f16) outs(%init354 : tensor<1x201x33x33xf16>) -> tensor<1x201x33x33xf16>
  %conv356 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu353, %w66 : tensor<1x1206x33x33xf16>, tensor<201x1206x1x1xf16>)
    outs(%fill355 : tensor<1x201x33x33xf16>) -> tensor<1x201x33x33xf16>
  %empty357 = tensor.empty() : tensor<1x201x33x33xf16>
  %relu358 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv356 : tensor<1x201x33x33xf16>)
    outs(%empty357 : tensor<1x201x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x201x33x33xf16>

  // conv67: 1x1 s1 p0 201->1206 33x33
  %init359 = tensor.empty() : tensor<1x1206x33x33xf16>
  %fill360 = linalg.fill ins(%cst : f16) outs(%init359 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %conv361 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu358, %w67 : tensor<1x201x33x33xf16>, tensor<1206x201x1x1xf16>)
    outs(%fill360 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %empty362 = tensor.empty() : tensor<1x1206x33x33xf16>
  %relu363 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv361 : tensor<1x1206x33x33xf16>)
    outs(%empty362 : tensor<1x1206x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1206x33x33xf16>

  // conv68: 3x3 s1 p1 1206->1206 33x33
  %pad364 = tensor.pad %relu363 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1206x33x33xf16> to tensor<1x1206x35x35xf16>
  %init365 = tensor.empty() : tensor<1x1206x33x33xf16>
  %fill366 = linalg.fill ins(%cst : f16) outs(%init365 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %conv367 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad364, %w68 : tensor<1x1206x35x35xf16>, tensor<1206x1206x3x3xf16>)
    outs(%fill366 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %empty368 = tensor.empty() : tensor<1x1206x33x33xf16>
  %relu369 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv367 : tensor<1x1206x33x33xf16>)
    outs(%empty368 : tensor<1x1206x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1206x33x33xf16>

  // conv69: 1x1 s1 p0 1206->201 33x33
  %init370 = tensor.empty() : tensor<1x201x33x33xf16>
  %fill371 = linalg.fill ins(%cst : f16) outs(%init370 : tensor<1x201x33x33xf16>) -> tensor<1x201x33x33xf16>
  %conv372 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu369, %w69 : tensor<1x1206x33x33xf16>, tensor<201x1206x1x1xf16>)
    outs(%fill371 : tensor<1x201x33x33xf16>) -> tensor<1x201x33x33xf16>
  %empty373 = tensor.empty() : tensor<1x201x33x33xf16>
  %relu374 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv372 : tensor<1x201x33x33xf16>)
    outs(%empty373 : tensor<1x201x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x201x33x33xf16>

  // conv70: 1x1 s1 p0 201->1206 33x33
  %init375 = tensor.empty() : tensor<1x1206x33x33xf16>
  %fill376 = linalg.fill ins(%cst : f16) outs(%init375 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %conv377 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu374, %w70 : tensor<1x201x33x33xf16>, tensor<1206x201x1x1xf16>)
    outs(%fill376 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %empty378 = tensor.empty() : tensor<1x1206x33x33xf16>
  %relu379 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv377 : tensor<1x1206x33x33xf16>)
    outs(%empty378 : tensor<1x1206x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1206x33x33xf16>

  // conv71: 3x3 s1 p1 1206->1206 33x33
  %pad380 = tensor.pad %relu379 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1206x33x33xf16> to tensor<1x1206x35x35xf16>
  %init381 = tensor.empty() : tensor<1x1206x33x33xf16>
  %fill382 = linalg.fill ins(%cst : f16) outs(%init381 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %conv383 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad380, %w71 : tensor<1x1206x35x35xf16>, tensor<1206x1206x3x3xf16>)
    outs(%fill382 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %empty384 = tensor.empty() : tensor<1x1206x33x33xf16>
  %relu385 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv383 : tensor<1x1206x33x33xf16>)
    outs(%empty384 : tensor<1x1206x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1206x33x33xf16>

  // conv72: 1x1 s1 p0 1206->201 33x33
  %init386 = tensor.empty() : tensor<1x201x33x33xf16>
  %fill387 = linalg.fill ins(%cst : f16) outs(%init386 : tensor<1x201x33x33xf16>) -> tensor<1x201x33x33xf16>
  %conv388 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu385, %w72 : tensor<1x1206x33x33xf16>, tensor<201x1206x1x1xf16>)
    outs(%fill387 : tensor<1x201x33x33xf16>) -> tensor<1x201x33x33xf16>
  %empty389 = tensor.empty() : tensor<1x201x33x33xf16>
  %relu390 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv388 : tensor<1x201x33x33xf16>)
    outs(%empty389 : tensor<1x201x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x201x33x33xf16>

  // conv73: 1x1 s1 p0 201->1206 33x33
  %init391 = tensor.empty() : tensor<1x1206x33x33xf16>
  %fill392 = linalg.fill ins(%cst : f16) outs(%init391 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %conv393 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu390, %w73 : tensor<1x201x33x33xf16>, tensor<1206x201x1x1xf16>)
    outs(%fill392 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %empty394 = tensor.empty() : tensor<1x1206x33x33xf16>
  %relu395 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv393 : tensor<1x1206x33x33xf16>)
    outs(%empty394 : tensor<1x1206x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1206x33x33xf16>

  // conv74: 3x3 s1 p1 1206->1206 33x33
  %pad396 = tensor.pad %relu395 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1206x33x33xf16> to tensor<1x1206x35x35xf16>
  %init397 = tensor.empty() : tensor<1x1206x33x33xf16>
  %fill398 = linalg.fill ins(%cst : f16) outs(%init397 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %conv399 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad396, %w74 : tensor<1x1206x35x35xf16>, tensor<1206x1206x3x3xf16>)
    outs(%fill398 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %empty400 = tensor.empty() : tensor<1x1206x33x33xf16>
  %relu401 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv399 : tensor<1x1206x33x33xf16>)
    outs(%empty400 : tensor<1x1206x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1206x33x33xf16>

  // conv75: 1x1 s1 p0 1206->201 33x33
  %init402 = tensor.empty() : tensor<1x201x33x33xf16>
  %fill403 = linalg.fill ins(%cst : f16) outs(%init402 : tensor<1x201x33x33xf16>) -> tensor<1x201x33x33xf16>
  %conv404 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu401, %w75 : tensor<1x1206x33x33xf16>, tensor<201x1206x1x1xf16>)
    outs(%fill403 : tensor<1x201x33x33xf16>) -> tensor<1x201x33x33xf16>
  %empty405 = tensor.empty() : tensor<1x201x33x33xf16>
  %relu406 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv404 : tensor<1x201x33x33xf16>)
    outs(%empty405 : tensor<1x201x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x201x33x33xf16>

  // conv76: 1x1 s1 p0 201->1206 33x33
  %init407 = tensor.empty() : tensor<1x1206x33x33xf16>
  %fill408 = linalg.fill ins(%cst : f16) outs(%init407 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %conv409 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu406, %w76 : tensor<1x201x33x33xf16>, tensor<1206x201x1x1xf16>)
    outs(%fill408 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %empty410 = tensor.empty() : tensor<1x1206x33x33xf16>
  %relu411 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv409 : tensor<1x1206x33x33xf16>)
    outs(%empty410 : tensor<1x1206x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1206x33x33xf16>

  // conv77: 3x3 s1 p1 1206->1206 33x33
  %pad412 = tensor.pad %relu411 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1206x33x33xf16> to tensor<1x1206x35x35xf16>
  %init413 = tensor.empty() : tensor<1x1206x33x33xf16>
  %fill414 = linalg.fill ins(%cst : f16) outs(%init413 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %conv415 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad412, %w77 : tensor<1x1206x35x35xf16>, tensor<1206x1206x3x3xf16>)
    outs(%fill414 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %empty416 = tensor.empty() : tensor<1x1206x33x33xf16>
  %relu417 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv415 : tensor<1x1206x33x33xf16>)
    outs(%empty416 : tensor<1x1206x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1206x33x33xf16>

  // conv78: 1x1 s1 p0 1206->201 33x33
  %init418 = tensor.empty() : tensor<1x201x33x33xf16>
  %fill419 = linalg.fill ins(%cst : f16) outs(%init418 : tensor<1x201x33x33xf16>) -> tensor<1x201x33x33xf16>
  %conv420 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu417, %w78 : tensor<1x1206x33x33xf16>, tensor<201x1206x1x1xf16>)
    outs(%fill419 : tensor<1x201x33x33xf16>) -> tensor<1x201x33x33xf16>
  %empty421 = tensor.empty() : tensor<1x201x33x33xf16>
  %relu422 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv420 : tensor<1x201x33x33xf16>)
    outs(%empty421 : tensor<1x201x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x201x33x33xf16>

  // conv79: 1x1 s1 p0 201->1206 33x33
  %init423 = tensor.empty() : tensor<1x1206x33x33xf16>
  %fill424 = linalg.fill ins(%cst : f16) outs(%init423 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %conv425 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu422, %w79 : tensor<1x201x33x33xf16>, tensor<1206x201x1x1xf16>)
    outs(%fill424 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %empty426 = tensor.empty() : tensor<1x1206x33x33xf16>
  %relu427 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv425 : tensor<1x1206x33x33xf16>)
    outs(%empty426 : tensor<1x1206x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1206x33x33xf16>

  // conv80: 3x3 s1 p1 1206->1206 33x33
  %pad428 = tensor.pad %relu427 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1206x33x33xf16> to tensor<1x1206x35x35xf16>
  %init429 = tensor.empty() : tensor<1x1206x33x33xf16>
  %fill430 = linalg.fill ins(%cst : f16) outs(%init429 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %conv431 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad428, %w80 : tensor<1x1206x35x35xf16>, tensor<1206x1206x3x3xf16>)
    outs(%fill430 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %empty432 = tensor.empty() : tensor<1x1206x33x33xf16>
  %relu433 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv431 : tensor<1x1206x33x33xf16>)
    outs(%empty432 : tensor<1x1206x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1206x33x33xf16>

  // conv81: 1x1 s1 p0 1206->201 33x33
  %init434 = tensor.empty() : tensor<1x201x33x33xf16>
  %fill435 = linalg.fill ins(%cst : f16) outs(%init434 : tensor<1x201x33x33xf16>) -> tensor<1x201x33x33xf16>
  %conv436 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu433, %w81 : tensor<1x1206x33x33xf16>, tensor<201x1206x1x1xf16>)
    outs(%fill435 : tensor<1x201x33x33xf16>) -> tensor<1x201x33x33xf16>
  %empty437 = tensor.empty() : tensor<1x201x33x33xf16>
  %relu438 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv436 : tensor<1x201x33x33xf16>)
    outs(%empty437 : tensor<1x201x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x201x33x33xf16>

  // conv82: 1x1 s1 p0 201->1206 33x33
  %init439 = tensor.empty() : tensor<1x1206x33x33xf16>
  %fill440 = linalg.fill ins(%cst : f16) outs(%init439 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %conv441 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu438, %w82 : tensor<1x201x33x33xf16>, tensor<1206x201x1x1xf16>)
    outs(%fill440 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %empty442 = tensor.empty() : tensor<1x1206x33x33xf16>
  %relu443 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv441 : tensor<1x1206x33x33xf16>)
    outs(%empty442 : tensor<1x1206x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1206x33x33xf16>

  // conv83: 3x3 s1 p1 1206->1206 33x33
  %pad444 = tensor.pad %relu443 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1206x33x33xf16> to tensor<1x1206x35x35xf16>
  %init445 = tensor.empty() : tensor<1x1206x33x33xf16>
  %fill446 = linalg.fill ins(%cst : f16) outs(%init445 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %conv447 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad444, %w83 : tensor<1x1206x35x35xf16>, tensor<1206x1206x3x3xf16>)
    outs(%fill446 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %empty448 = tensor.empty() : tensor<1x1206x33x33xf16>
  %relu449 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv447 : tensor<1x1206x33x33xf16>)
    outs(%empty448 : tensor<1x1206x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1206x33x33xf16>

  // conv84: 1x1 s1 p0 1206->201 33x33
  %init450 = tensor.empty() : tensor<1x201x33x33xf16>
  %fill451 = linalg.fill ins(%cst : f16) outs(%init450 : tensor<1x201x33x33xf16>) -> tensor<1x201x33x33xf16>
  %conv452 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu449, %w84 : tensor<1x1206x33x33xf16>, tensor<201x1206x1x1xf16>)
    outs(%fill451 : tensor<1x201x33x33xf16>) -> tensor<1x201x33x33xf16>
  %empty453 = tensor.empty() : tensor<1x201x33x33xf16>
  %relu454 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv452 : tensor<1x201x33x33xf16>)
    outs(%empty453 : tensor<1x201x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x201x33x33xf16>

  // conv85: 1x1 s1 p0 201->1206 33x33
  %init455 = tensor.empty() : tensor<1x1206x33x33xf16>
  %fill456 = linalg.fill ins(%cst : f16) outs(%init455 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %conv457 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu454, %w85 : tensor<1x201x33x33xf16>, tensor<1206x201x1x1xf16>)
    outs(%fill456 : tensor<1x1206x33x33xf16>) -> tensor<1x1206x33x33xf16>
  %empty458 = tensor.empty() : tensor<1x1206x33x33xf16>
  %relu459 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv457 : tensor<1x1206x33x33xf16>)
    outs(%empty458 : tensor<1x1206x33x33xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1206x33x33xf16>

  // conv86: 3x3 s2 p1 1206->1206 33x33
  %pad460 = tensor.pad %relu459 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1206x33x33xf16> to tensor<1x1206x35x35xf16>
  %init461 = tensor.empty() : tensor<1x1206x17x17xf16>
  %fill462 = linalg.fill ins(%cst : f16) outs(%init461 : tensor<1x1206x17x17xf16>) -> tensor<1x1206x17x17xf16>
  %conv463 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad460, %w86 : tensor<1x1206x35x35xf16>, tensor<1206x1206x3x3xf16>)
    outs(%fill462 : tensor<1x1206x17x17xf16>) -> tensor<1x1206x17x17xf16>
  %empty464 = tensor.empty() : tensor<1x1206x17x17xf16>
  %relu465 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv463 : tensor<1x1206x17x17xf16>)
    outs(%empty464 : tensor<1x1206x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1206x17x17xf16>

  // conv87: 1x1 s1 p0 1206->345 17x17
  %init466 = tensor.empty() : tensor<1x345x17x17xf16>
  %fill467 = linalg.fill ins(%cst : f16) outs(%init466 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %conv468 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu465, %w87 : tensor<1x1206x17x17xf16>, tensor<345x1206x1x1xf16>)
    outs(%fill467 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %empty469 = tensor.empty() : tensor<1x345x17x17xf16>
  %relu470 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv468 : tensor<1x345x17x17xf16>)
    outs(%empty469 : tensor<1x345x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x345x17x17xf16>

  // conv88: 1x1 s1 p0 345->2070 17x17
  %init471 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill472 = linalg.fill ins(%cst : f16) outs(%init471 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv473 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu470, %w88 : tensor<1x345x17x17xf16>, tensor<2070x345x1x1xf16>)
    outs(%fill472 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty474 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu475 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv473 : tensor<1x2070x17x17xf16>)
    outs(%empty474 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv89: 3x3 s1 p1 2070->2070 17x17
  %pad476 = tensor.pad %relu475 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x2070x17x17xf16> to tensor<1x2070x19x19xf16>
  %init477 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill478 = linalg.fill ins(%cst : f16) outs(%init477 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv479 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad476, %w89 : tensor<1x2070x19x19xf16>, tensor<2070x2070x3x3xf16>)
    outs(%fill478 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty480 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu481 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv479 : tensor<1x2070x17x17xf16>)
    outs(%empty480 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv90: 1x1 s1 p0 2070->345 17x17
  %init482 = tensor.empty() : tensor<1x345x17x17xf16>
  %fill483 = linalg.fill ins(%cst : f16) outs(%init482 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %conv484 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu481, %w90 : tensor<1x2070x17x17xf16>, tensor<345x2070x1x1xf16>)
    outs(%fill483 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %empty485 = tensor.empty() : tensor<1x345x17x17xf16>
  %relu486 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv484 : tensor<1x345x17x17xf16>)
    outs(%empty485 : tensor<1x345x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x345x17x17xf16>

  // conv91: 1x1 s1 p0 345->2070 17x17
  %init487 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill488 = linalg.fill ins(%cst : f16) outs(%init487 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv489 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu486, %w91 : tensor<1x345x17x17xf16>, tensor<2070x345x1x1xf16>)
    outs(%fill488 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty490 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu491 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv489 : tensor<1x2070x17x17xf16>)
    outs(%empty490 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv92: 3x3 s1 p1 2070->2070 17x17
  %pad492 = tensor.pad %relu491 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x2070x17x17xf16> to tensor<1x2070x19x19xf16>
  %init493 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill494 = linalg.fill ins(%cst : f16) outs(%init493 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv495 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad492, %w92 : tensor<1x2070x19x19xf16>, tensor<2070x2070x3x3xf16>)
    outs(%fill494 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty496 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu497 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv495 : tensor<1x2070x17x17xf16>)
    outs(%empty496 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv93: 1x1 s1 p0 2070->345 17x17
  %init498 = tensor.empty() : tensor<1x345x17x17xf16>
  %fill499 = linalg.fill ins(%cst : f16) outs(%init498 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %conv500 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu497, %w93 : tensor<1x2070x17x17xf16>, tensor<345x2070x1x1xf16>)
    outs(%fill499 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %empty501 = tensor.empty() : tensor<1x345x17x17xf16>
  %relu502 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv500 : tensor<1x345x17x17xf16>)
    outs(%empty501 : tensor<1x345x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x345x17x17xf16>

  // conv94: 1x1 s1 p0 345->2070 17x17
  %init503 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill504 = linalg.fill ins(%cst : f16) outs(%init503 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv505 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu502, %w94 : tensor<1x345x17x17xf16>, tensor<2070x345x1x1xf16>)
    outs(%fill504 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty506 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu507 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv505 : tensor<1x2070x17x17xf16>)
    outs(%empty506 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv95: 3x3 s1 p1 2070->2070 17x17
  %pad508 = tensor.pad %relu507 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x2070x17x17xf16> to tensor<1x2070x19x19xf16>
  %init509 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill510 = linalg.fill ins(%cst : f16) outs(%init509 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv511 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad508, %w95 : tensor<1x2070x19x19xf16>, tensor<2070x2070x3x3xf16>)
    outs(%fill510 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty512 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu513 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv511 : tensor<1x2070x17x17xf16>)
    outs(%empty512 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv96: 1x1 s1 p0 2070->345 17x17
  %init514 = tensor.empty() : tensor<1x345x17x17xf16>
  %fill515 = linalg.fill ins(%cst : f16) outs(%init514 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %conv516 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu513, %w96 : tensor<1x2070x17x17xf16>, tensor<345x2070x1x1xf16>)
    outs(%fill515 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %empty517 = tensor.empty() : tensor<1x345x17x17xf16>
  %relu518 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv516 : tensor<1x345x17x17xf16>)
    outs(%empty517 : tensor<1x345x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x345x17x17xf16>

  // conv97: 1x1 s1 p0 345->2070 17x17
  %init519 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill520 = linalg.fill ins(%cst : f16) outs(%init519 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv521 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu518, %w97 : tensor<1x345x17x17xf16>, tensor<2070x345x1x1xf16>)
    outs(%fill520 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty522 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu523 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv521 : tensor<1x2070x17x17xf16>)
    outs(%empty522 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv98: 3x3 s1 p1 2070->2070 17x17
  %pad524 = tensor.pad %relu523 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x2070x17x17xf16> to tensor<1x2070x19x19xf16>
  %init525 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill526 = linalg.fill ins(%cst : f16) outs(%init525 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv527 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad524, %w98 : tensor<1x2070x19x19xf16>, tensor<2070x2070x3x3xf16>)
    outs(%fill526 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty528 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu529 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv527 : tensor<1x2070x17x17xf16>)
    outs(%empty528 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv99: 1x1 s1 p0 2070->345 17x17
  %init530 = tensor.empty() : tensor<1x345x17x17xf16>
  %fill531 = linalg.fill ins(%cst : f16) outs(%init530 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %conv532 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu529, %w99 : tensor<1x2070x17x17xf16>, tensor<345x2070x1x1xf16>)
    outs(%fill531 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %empty533 = tensor.empty() : tensor<1x345x17x17xf16>
  %relu534 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv532 : tensor<1x345x17x17xf16>)
    outs(%empty533 : tensor<1x345x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x345x17x17xf16>

  // conv100: 1x1 s1 p0 345->2070 17x17
  %init535 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill536 = linalg.fill ins(%cst : f16) outs(%init535 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv537 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu534, %w100 : tensor<1x345x17x17xf16>, tensor<2070x345x1x1xf16>)
    outs(%fill536 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty538 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu539 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv537 : tensor<1x2070x17x17xf16>)
    outs(%empty538 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv101: 3x3 s1 p1 2070->2070 17x17
  %pad540 = tensor.pad %relu539 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x2070x17x17xf16> to tensor<1x2070x19x19xf16>
  %init541 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill542 = linalg.fill ins(%cst : f16) outs(%init541 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv543 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad540, %w101 : tensor<1x2070x19x19xf16>, tensor<2070x2070x3x3xf16>)
    outs(%fill542 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty544 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu545 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv543 : tensor<1x2070x17x17xf16>)
    outs(%empty544 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv102: 1x1 s1 p0 2070->345 17x17
  %init546 = tensor.empty() : tensor<1x345x17x17xf16>
  %fill547 = linalg.fill ins(%cst : f16) outs(%init546 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %conv548 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu545, %w102 : tensor<1x2070x17x17xf16>, tensor<345x2070x1x1xf16>)
    outs(%fill547 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %empty549 = tensor.empty() : tensor<1x345x17x17xf16>
  %relu550 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv548 : tensor<1x345x17x17xf16>)
    outs(%empty549 : tensor<1x345x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x345x17x17xf16>

  // conv103: 1x1 s1 p0 345->2070 17x17
  %init551 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill552 = linalg.fill ins(%cst : f16) outs(%init551 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv553 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu550, %w103 : tensor<1x345x17x17xf16>, tensor<2070x345x1x1xf16>)
    outs(%fill552 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty554 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu555 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv553 : tensor<1x2070x17x17xf16>)
    outs(%empty554 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv104: 3x3 s1 p1 2070->2070 17x17
  %pad556 = tensor.pad %relu555 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x2070x17x17xf16> to tensor<1x2070x19x19xf16>
  %init557 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill558 = linalg.fill ins(%cst : f16) outs(%init557 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv559 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad556, %w104 : tensor<1x2070x19x19xf16>, tensor<2070x2070x3x3xf16>)
    outs(%fill558 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty560 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu561 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv559 : tensor<1x2070x17x17xf16>)
    outs(%empty560 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv105: 1x1 s1 p0 2070->345 17x17
  %init562 = tensor.empty() : tensor<1x345x17x17xf16>
  %fill563 = linalg.fill ins(%cst : f16) outs(%init562 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %conv564 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu561, %w105 : tensor<1x2070x17x17xf16>, tensor<345x2070x1x1xf16>)
    outs(%fill563 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %empty565 = tensor.empty() : tensor<1x345x17x17xf16>
  %relu566 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv564 : tensor<1x345x17x17xf16>)
    outs(%empty565 : tensor<1x345x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x345x17x17xf16>

  // conv106: 1x1 s1 p0 345->2070 17x17
  %init567 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill568 = linalg.fill ins(%cst : f16) outs(%init567 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv569 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu566, %w106 : tensor<1x345x17x17xf16>, tensor<2070x345x1x1xf16>)
    outs(%fill568 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty570 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu571 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv569 : tensor<1x2070x17x17xf16>)
    outs(%empty570 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv107: 3x3 s1 p1 2070->2070 17x17
  %pad572 = tensor.pad %relu571 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x2070x17x17xf16> to tensor<1x2070x19x19xf16>
  %init573 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill574 = linalg.fill ins(%cst : f16) outs(%init573 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv575 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad572, %w107 : tensor<1x2070x19x19xf16>, tensor<2070x2070x3x3xf16>)
    outs(%fill574 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty576 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu577 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv575 : tensor<1x2070x17x17xf16>)
    outs(%empty576 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv108: 1x1 s1 p0 2070->345 17x17
  %init578 = tensor.empty() : tensor<1x345x17x17xf16>
  %fill579 = linalg.fill ins(%cst : f16) outs(%init578 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %conv580 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu577, %w108 : tensor<1x2070x17x17xf16>, tensor<345x2070x1x1xf16>)
    outs(%fill579 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %empty581 = tensor.empty() : tensor<1x345x17x17xf16>
  %relu582 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv580 : tensor<1x345x17x17xf16>)
    outs(%empty581 : tensor<1x345x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x345x17x17xf16>

  // conv109: 1x1 s1 p0 345->2070 17x17
  %init583 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill584 = linalg.fill ins(%cst : f16) outs(%init583 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv585 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu582, %w109 : tensor<1x345x17x17xf16>, tensor<2070x345x1x1xf16>)
    outs(%fill584 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty586 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu587 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv585 : tensor<1x2070x17x17xf16>)
    outs(%empty586 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv110: 3x3 s1 p1 2070->2070 17x17
  %pad588 = tensor.pad %relu587 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x2070x17x17xf16> to tensor<1x2070x19x19xf16>
  %init589 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill590 = linalg.fill ins(%cst : f16) outs(%init589 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv591 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad588, %w110 : tensor<1x2070x19x19xf16>, tensor<2070x2070x3x3xf16>)
    outs(%fill590 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty592 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu593 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv591 : tensor<1x2070x17x17xf16>)
    outs(%empty592 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv111: 1x1 s1 p0 2070->345 17x17
  %init594 = tensor.empty() : tensor<1x345x17x17xf16>
  %fill595 = linalg.fill ins(%cst : f16) outs(%init594 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %conv596 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu593, %w111 : tensor<1x2070x17x17xf16>, tensor<345x2070x1x1xf16>)
    outs(%fill595 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %empty597 = tensor.empty() : tensor<1x345x17x17xf16>
  %relu598 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv596 : tensor<1x345x17x17xf16>)
    outs(%empty597 : tensor<1x345x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x345x17x17xf16>

  // conv112: 1x1 s1 p0 345->2070 17x17
  %init599 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill600 = linalg.fill ins(%cst : f16) outs(%init599 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv601 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu598, %w112 : tensor<1x345x17x17xf16>, tensor<2070x345x1x1xf16>)
    outs(%fill600 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty602 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu603 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv601 : tensor<1x2070x17x17xf16>)
    outs(%empty602 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv113: 3x3 s1 p1 2070->2070 17x17
  %pad604 = tensor.pad %relu603 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x2070x17x17xf16> to tensor<1x2070x19x19xf16>
  %init605 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill606 = linalg.fill ins(%cst : f16) outs(%init605 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv607 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad604, %w113 : tensor<1x2070x19x19xf16>, tensor<2070x2070x3x3xf16>)
    outs(%fill606 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty608 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu609 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv607 : tensor<1x2070x17x17xf16>)
    outs(%empty608 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv114: 1x1 s1 p0 2070->345 17x17
  %init610 = tensor.empty() : tensor<1x345x17x17xf16>
  %fill611 = linalg.fill ins(%cst : f16) outs(%init610 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %conv612 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu609, %w114 : tensor<1x2070x17x17xf16>, tensor<345x2070x1x1xf16>)
    outs(%fill611 : tensor<1x345x17x17xf16>) -> tensor<1x345x17x17xf16>
  %empty613 = tensor.empty() : tensor<1x345x17x17xf16>
  %relu614 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv612 : tensor<1x345x17x17xf16>)
    outs(%empty613 : tensor<1x345x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x345x17x17xf16>

  // conv115: 1x1 s1 p0 345->2070 17x17
  %init615 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill616 = linalg.fill ins(%cst : f16) outs(%init615 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv617 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu614, %w115 : tensor<1x345x17x17xf16>, tensor<2070x345x1x1xf16>)
    outs(%fill616 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty618 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu619 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv617 : tensor<1x2070x17x17xf16>)
    outs(%empty618 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv116: 3x3 s1 p1 2070->2070 17x17
  %pad620 = tensor.pad %relu619 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x2070x17x17xf16> to tensor<1x2070x19x19xf16>
  %init621 = tensor.empty() : tensor<1x2070x17x17xf16>
  %fill622 = linalg.fill ins(%cst : f16) outs(%init621 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %conv623 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad620, %w116 : tensor<1x2070x19x19xf16>, tensor<2070x2070x3x3xf16>)
    outs(%fill622 : tensor<1x2070x17x17xf16>) -> tensor<1x2070x17x17xf16>
  %empty624 = tensor.empty() : tensor<1x2070x17x17xf16>
  %relu625 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv623 : tensor<1x2070x17x17xf16>)
    outs(%empty624 : tensor<1x2070x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2070x17x17xf16>

  // conv117: 1x1 s1 p0 2070->576 17x17
  %init626 = tensor.empty() : tensor<1x576x17x17xf16>
  %fill627 = linalg.fill ins(%cst : f16) outs(%init626 : tensor<1x576x17x17xf16>) -> tensor<1x576x17x17xf16>
  %conv628 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu625, %w117 : tensor<1x2070x17x17xf16>, tensor<576x2070x1x1xf16>)
    outs(%fill627 : tensor<1x576x17x17xf16>) -> tensor<1x576x17x17xf16>
  %empty629 = tensor.empty() : tensor<1x576x17x17xf16>
  %relu630 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv628 : tensor<1x576x17x17xf16>)
    outs(%empty629 : tensor<1x576x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x17x17xf16>

  // conv118: 1x1 s1 p0 576->3456 17x17
  %init631 = tensor.empty() : tensor<1x3456x17x17xf16>
  %fill632 = linalg.fill ins(%cst : f16) outs(%init631 : tensor<1x3456x17x17xf16>) -> tensor<1x3456x17x17xf16>
  %conv633 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu630, %w118 : tensor<1x576x17x17xf16>, tensor<3456x576x1x1xf16>)
    outs(%fill632 : tensor<1x3456x17x17xf16>) -> tensor<1x3456x17x17xf16>
  %empty634 = tensor.empty() : tensor<1x3456x17x17xf16>
  %relu635 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv633 : tensor<1x3456x17x17xf16>)
    outs(%empty634 : tensor<1x3456x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x3456x17x17xf16>

  // conv119: 3x3 s1 p1 3456->3456 17x17
  %pad636 = tensor.pad %relu635 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3456x17x17xf16> to tensor<1x3456x19x19xf16>
  %init637 = tensor.empty() : tensor<1x3456x17x17xf16>
  %fill638 = linalg.fill ins(%cst : f16) outs(%init637 : tensor<1x3456x17x17xf16>) -> tensor<1x3456x17x17xf16>
  %conv639 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad636, %w119 : tensor<1x3456x19x19xf16>, tensor<3456x3456x3x3xf16>)
    outs(%fill638 : tensor<1x3456x17x17xf16>) -> tensor<1x3456x17x17xf16>
  %empty640 = tensor.empty() : tensor<1x3456x17x17xf16>
  %relu641 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv639 : tensor<1x3456x17x17xf16>)
    outs(%empty640 : tensor<1x3456x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x3456x17x17xf16>

  // conv120: 1x1 s1 p0 3456->576 17x17
  %init642 = tensor.empty() : tensor<1x576x17x17xf16>
  %fill643 = linalg.fill ins(%cst : f16) outs(%init642 : tensor<1x576x17x17xf16>) -> tensor<1x576x17x17xf16>
  %conv644 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu641, %w120 : tensor<1x3456x17x17xf16>, tensor<576x3456x1x1xf16>)
    outs(%fill643 : tensor<1x576x17x17xf16>) -> tensor<1x576x17x17xf16>
  %empty645 = tensor.empty() : tensor<1x576x17x17xf16>
  %relu646 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv644 : tensor<1x576x17x17xf16>)
    outs(%empty645 : tensor<1x576x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x17x17xf16>

  // conv121: 1x1 s1 p0 576->3456 17x17
  %init647 = tensor.empty() : tensor<1x3456x17x17xf16>
  %fill648 = linalg.fill ins(%cst : f16) outs(%init647 : tensor<1x3456x17x17xf16>) -> tensor<1x3456x17x17xf16>
  %conv649 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu646, %w121 : tensor<1x576x17x17xf16>, tensor<3456x576x1x1xf16>)
    outs(%fill648 : tensor<1x3456x17x17xf16>) -> tensor<1x3456x17x17xf16>
  %empty650 = tensor.empty() : tensor<1x3456x17x17xf16>
  %relu651 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv649 : tensor<1x3456x17x17xf16>)
    outs(%empty650 : tensor<1x3456x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x3456x17x17xf16>

  // conv122: 3x3 s1 p1 3456->3456 17x17
  %pad652 = tensor.pad %relu651 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3456x17x17xf16> to tensor<1x3456x19x19xf16>
  %init653 = tensor.empty() : tensor<1x3456x17x17xf16>
  %fill654 = linalg.fill ins(%cst : f16) outs(%init653 : tensor<1x3456x17x17xf16>) -> tensor<1x3456x17x17xf16>
  %conv655 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad652, %w122 : tensor<1x3456x19x19xf16>, tensor<3456x3456x3x3xf16>)
    outs(%fill654 : tensor<1x3456x17x17xf16>) -> tensor<1x3456x17x17xf16>
  %empty656 = tensor.empty() : tensor<1x3456x17x17xf16>
  %relu657 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv655 : tensor<1x3456x17x17xf16>)
    outs(%empty656 : tensor<1x3456x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x3456x17x17xf16>

  // conv123: 1x1 s1 p0 3456->576 17x17
  %init658 = tensor.empty() : tensor<1x576x17x17xf16>
  %fill659 = linalg.fill ins(%cst : f16) outs(%init658 : tensor<1x576x17x17xf16>) -> tensor<1x576x17x17xf16>
  %conv660 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu657, %w123 : tensor<1x3456x17x17xf16>, tensor<576x3456x1x1xf16>)
    outs(%fill659 : tensor<1x576x17x17xf16>) -> tensor<1x576x17x17xf16>
  %empty661 = tensor.empty() : tensor<1x576x17x17xf16>
  %relu662 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv660 : tensor<1x576x17x17xf16>)
    outs(%empty661 : tensor<1x576x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x17x17xf16>

  // conv124: 1x1 s1 p0 576->2304 17x17
  %init663 = tensor.empty() : tensor<1x2304x17x17xf16>
  %fill664 = linalg.fill ins(%cst : f16) outs(%init663 : tensor<1x2304x17x17xf16>) -> tensor<1x2304x17x17xf16>
  %conv665 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu662, %w124 : tensor<1x576x17x17xf16>, tensor<2304x576x1x1xf16>)
    outs(%fill664 : tensor<1x2304x17x17xf16>) -> tensor<1x2304x17x17xf16>
  %empty666 = tensor.empty() : tensor<1x2304x17x17xf16>
  %relu667 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv665 : tensor<1x2304x17x17xf16>)
    outs(%empty666 : tensor<1x2304x17x17xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2304x17x17xf16>

  // FC as 1x1 conv: 2304->1000
  %init668 = tensor.empty() : tensor<1x1000x17x17xf16>
  %fill669 = linalg.fill ins(%cst : f16) outs(%init668 : tensor<1x1000x17x17xf16>) -> tensor<1x1000x17x17xf16>
  %conv670 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu667, %w_fc : tensor<1x2304x17x17xf16>, tensor<1000x2304x1x1xf16>)
    outs(%fill669 : tensor<1x1000x17x17xf16>) -> tensor<1x1000x17x17xf16>
  return %conv670 : tensor<1x1000x17x17xf16>
}
