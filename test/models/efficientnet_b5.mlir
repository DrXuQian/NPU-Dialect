func.func @efficientnet_b5(
    %input: tensor<1x3x456x456xf16>,
    %w0: tensor<51x3x3x3xf16>,
    %w1: tensor<51x51x3x3xf16>,
    %w2: tensor<25x51x1x1xf16>,
    %w3: tensor<25x25x3x3xf16>,
    %w4: tensor<25x25x1x1xf16>,
    %w5: tensor<150x25x1x1xf16>,
    %w6: tensor<150x150x3x3xf16>,
    %w7: tensor<38x150x1x1xf16>,
    %w8: tensor<228x38x1x1xf16>,
    %w9: tensor<228x228x3x3xf16>,
    %w10: tensor<38x228x1x1xf16>,
    %w11: tensor<228x38x1x1xf16>,
    %w12: tensor<228x228x3x3xf16>,
    %w13: tensor<38x228x1x1xf16>,
    %w14: tensor<228x38x1x1xf16>,
    %w15: tensor<228x228x3x3xf16>,
    %w16: tensor<38x228x1x1xf16>,
    %w17: tensor<228x38x1x1xf16>,
    %w18: tensor<228x228x3x3xf16>,
    %w19: tensor<64x228x1x1xf16>,
    %w20: tensor<384x64x1x1xf16>,
    %w21: tensor<384x384x3x3xf16>,
    %w22: tensor<64x384x1x1xf16>,
    %w23: tensor<384x64x1x1xf16>,
    %w24: tensor<384x384x3x3xf16>,
    %w25: tensor<64x384x1x1xf16>,
    %w26: tensor<384x64x1x1xf16>,
    %w27: tensor<384x384x3x3xf16>,
    %w28: tensor<64x384x1x1xf16>,
    %w29: tensor<384x64x1x1xf16>,
    %w30: tensor<384x384x3x3xf16>,
    %w31: tensor<128x384x1x1xf16>,
    %w32: tensor<768x128x1x1xf16>,
    %w33: tensor<768x768x3x3xf16>,
    %w34: tensor<128x768x1x1xf16>,
    %w35: tensor<768x128x1x1xf16>,
    %w36: tensor<768x768x3x3xf16>,
    %w37: tensor<128x768x1x1xf16>,
    %w38: tensor<768x128x1x1xf16>,
    %w39: tensor<768x768x3x3xf16>,
    %w40: tensor<128x768x1x1xf16>,
    %w41: tensor<768x128x1x1xf16>,
    %w42: tensor<768x768x3x3xf16>,
    %w43: tensor<128x768x1x1xf16>,
    %w44: tensor<768x128x1x1xf16>,
    %w45: tensor<768x768x3x3xf16>,
    %w46: tensor<128x768x1x1xf16>,
    %w47: tensor<768x128x1x1xf16>,
    %w48: tensor<768x768x3x3xf16>,
    %w49: tensor<128x768x1x1xf16>,
    %w50: tensor<768x128x1x1xf16>,
    %w51: tensor<768x768x3x3xf16>,
    %w52: tensor<179x768x1x1xf16>,
    %w53: tensor<1074x179x1x1xf16>,
    %w54: tensor<1074x1074x3x3xf16>,
    %w55: tensor<179x1074x1x1xf16>,
    %w56: tensor<1074x179x1x1xf16>,
    %w57: tensor<1074x1074x3x3xf16>,
    %w58: tensor<179x1074x1x1xf16>,
    %w59: tensor<1074x179x1x1xf16>,
    %w60: tensor<1074x1074x3x3xf16>,
    %w61: tensor<179x1074x1x1xf16>,
    %w62: tensor<1074x179x1x1xf16>,
    %w63: tensor<1074x1074x3x3xf16>,
    %w64: tensor<179x1074x1x1xf16>,
    %w65: tensor<1074x179x1x1xf16>,
    %w66: tensor<1074x1074x3x3xf16>,
    %w67: tensor<179x1074x1x1xf16>,
    %w68: tensor<1074x179x1x1xf16>,
    %w69: tensor<1074x1074x3x3xf16>,
    %w70: tensor<179x1074x1x1xf16>,
    %w71: tensor<1074x179x1x1xf16>,
    %w72: tensor<1074x1074x3x3xf16>,
    %w73: tensor<307x1074x1x1xf16>,
    %w74: tensor<1842x307x1x1xf16>,
    %w75: tensor<1842x1842x3x3xf16>,
    %w76: tensor<307x1842x1x1xf16>,
    %w77: tensor<1842x307x1x1xf16>,
    %w78: tensor<1842x1842x3x3xf16>,
    %w79: tensor<307x1842x1x1xf16>,
    %w80: tensor<1842x307x1x1xf16>,
    %w81: tensor<1842x1842x3x3xf16>,
    %w82: tensor<307x1842x1x1xf16>,
    %w83: tensor<1842x307x1x1xf16>,
    %w84: tensor<1842x1842x3x3xf16>,
    %w85: tensor<307x1842x1x1xf16>,
    %w86: tensor<1842x307x1x1xf16>,
    %w87: tensor<1842x1842x3x3xf16>,
    %w88: tensor<307x1842x1x1xf16>,
    %w89: tensor<1842x307x1x1xf16>,
    %w90: tensor<1842x1842x3x3xf16>,
    %w91: tensor<307x1842x1x1xf16>,
    %w92: tensor<1842x307x1x1xf16>,
    %w93: tensor<1842x1842x3x3xf16>,
    %w94: tensor<307x1842x1x1xf16>,
    %w95: tensor<1842x307x1x1xf16>,
    %w96: tensor<1842x1842x3x3xf16>,
    %w97: tensor<307x1842x1x1xf16>,
    %w98: tensor<1842x307x1x1xf16>,
    %w99: tensor<1842x1842x3x3xf16>,
    %w100: tensor<512x1842x1x1xf16>,
    %w101: tensor<3072x512x1x1xf16>,
    %w102: tensor<3072x3072x3x3xf16>,
    %w103: tensor<512x3072x1x1xf16>,
    %w104: tensor<2048x512x1x1xf16>,
    %w_fc: tensor<1000x2048x1x1xf16>) -> tensor<1x1000x15x15xf16> {
  %cst = arith.constant 0.0 : f16

  // conv0: 3x3 s2 p1 3->51 456x456
  %pad0 = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x456x456xf16> to tensor<1x3x458x458xf16>
  %init1 = tensor.empty() : tensor<1x51x228x228xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x51x228x228xf16>) -> tensor<1x51x228x228xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad0, %w0 : tensor<1x3x458x458xf16>, tensor<51x3x3x3xf16>)
    outs(%fill2 : tensor<1x51x228x228xf16>) -> tensor<1x51x228x228xf16>
  %empty4 = tensor.empty() : tensor<1x51x228x228xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x51x228x228xf16>)
    outs(%empty4 : tensor<1x51x228x228xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x51x228x228xf16>

  // conv1: 3x3 s1 p1 51->51 228x228
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x51x228x228xf16> to tensor<1x51x230x230xf16>
  %init7 = tensor.empty() : tensor<1x51x228x228xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x51x228x228xf16>) -> tensor<1x51x228x228xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad6, %w1 : tensor<1x51x230x230xf16>, tensor<51x51x3x3xf16>)
    outs(%fill8 : tensor<1x51x228x228xf16>) -> tensor<1x51x228x228xf16>
  %empty10 = tensor.empty() : tensor<1x51x228x228xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x51x228x228xf16>)
    outs(%empty10 : tensor<1x51x228x228xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x51x228x228xf16>

  // conv2: 1x1 s1 p0 51->25 228x228
  %init12 = tensor.empty() : tensor<1x25x228x228xf16>
  %fill13 = linalg.fill ins(%cst : f16) outs(%init12 : tensor<1x25x228x228xf16>) -> tensor<1x25x228x228xf16>
  %conv14 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu11, %w2 : tensor<1x51x228x228xf16>, tensor<25x51x1x1xf16>)
    outs(%fill13 : tensor<1x25x228x228xf16>) -> tensor<1x25x228x228xf16>
  %empty15 = tensor.empty() : tensor<1x25x228x228xf16>
  %relu16 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv14 : tensor<1x25x228x228xf16>)
    outs(%empty15 : tensor<1x25x228x228xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x25x228x228xf16>

  // conv3: 3x3 s1 p1 25->25 228x228
  %pad17 = tensor.pad %relu16 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x25x228x228xf16> to tensor<1x25x230x230xf16>
  %init18 = tensor.empty() : tensor<1x25x228x228xf16>
  %fill19 = linalg.fill ins(%cst : f16) outs(%init18 : tensor<1x25x228x228xf16>) -> tensor<1x25x228x228xf16>
  %conv20 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad17, %w3 : tensor<1x25x230x230xf16>, tensor<25x25x3x3xf16>)
    outs(%fill19 : tensor<1x25x228x228xf16>) -> tensor<1x25x228x228xf16>
  %empty21 = tensor.empty() : tensor<1x25x228x228xf16>
  %relu22 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv20 : tensor<1x25x228x228xf16>)
    outs(%empty21 : tensor<1x25x228x228xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x25x228x228xf16>

  // conv4: 1x1 s1 p0 25->25 228x228
  %init23 = tensor.empty() : tensor<1x25x228x228xf16>
  %fill24 = linalg.fill ins(%cst : f16) outs(%init23 : tensor<1x25x228x228xf16>) -> tensor<1x25x228x228xf16>
  %conv25 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu22, %w4 : tensor<1x25x228x228xf16>, tensor<25x25x1x1xf16>)
    outs(%fill24 : tensor<1x25x228x228xf16>) -> tensor<1x25x228x228xf16>
  %empty26 = tensor.empty() : tensor<1x25x228x228xf16>
  %relu27 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv25 : tensor<1x25x228x228xf16>)
    outs(%empty26 : tensor<1x25x228x228xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x25x228x228xf16>

  // conv5: 1x1 s1 p0 25->150 228x228
  %init28 = tensor.empty() : tensor<1x150x228x228xf16>
  %fill29 = linalg.fill ins(%cst : f16) outs(%init28 : tensor<1x150x228x228xf16>) -> tensor<1x150x228x228xf16>
  %conv30 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu27, %w5 : tensor<1x25x228x228xf16>, tensor<150x25x1x1xf16>)
    outs(%fill29 : tensor<1x150x228x228xf16>) -> tensor<1x150x228x228xf16>
  %empty31 = tensor.empty() : tensor<1x150x228x228xf16>
  %relu32 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv30 : tensor<1x150x228x228xf16>)
    outs(%empty31 : tensor<1x150x228x228xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x150x228x228xf16>

  // conv6: 3x3 s2 p1 150->150 228x228
  %pad33 = tensor.pad %relu32 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x150x228x228xf16> to tensor<1x150x230x230xf16>
  %init34 = tensor.empty() : tensor<1x150x114x114xf16>
  %fill35 = linalg.fill ins(%cst : f16) outs(%init34 : tensor<1x150x114x114xf16>) -> tensor<1x150x114x114xf16>
  %conv36 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad33, %w6 : tensor<1x150x230x230xf16>, tensor<150x150x3x3xf16>)
    outs(%fill35 : tensor<1x150x114x114xf16>) -> tensor<1x150x114x114xf16>
  %empty37 = tensor.empty() : tensor<1x150x114x114xf16>
  %relu38 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv36 : tensor<1x150x114x114xf16>)
    outs(%empty37 : tensor<1x150x114x114xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x150x114x114xf16>

  // conv7: 1x1 s1 p0 150->38 114x114
  %init39 = tensor.empty() : tensor<1x38x114x114xf16>
  %fill40 = linalg.fill ins(%cst : f16) outs(%init39 : tensor<1x38x114x114xf16>) -> tensor<1x38x114x114xf16>
  %conv41 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu38, %w7 : tensor<1x150x114x114xf16>, tensor<38x150x1x1xf16>)
    outs(%fill40 : tensor<1x38x114x114xf16>) -> tensor<1x38x114x114xf16>
  %empty42 = tensor.empty() : tensor<1x38x114x114xf16>
  %relu43 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv41 : tensor<1x38x114x114xf16>)
    outs(%empty42 : tensor<1x38x114x114xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x38x114x114xf16>

  // conv8: 1x1 s1 p0 38->228 114x114
  %init44 = tensor.empty() : tensor<1x228x114x114xf16>
  %fill45 = linalg.fill ins(%cst : f16) outs(%init44 : tensor<1x228x114x114xf16>) -> tensor<1x228x114x114xf16>
  %conv46 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu43, %w8 : tensor<1x38x114x114xf16>, tensor<228x38x1x1xf16>)
    outs(%fill45 : tensor<1x228x114x114xf16>) -> tensor<1x228x114x114xf16>
  %empty47 = tensor.empty() : tensor<1x228x114x114xf16>
  %relu48 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv46 : tensor<1x228x114x114xf16>)
    outs(%empty47 : tensor<1x228x114x114xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x228x114x114xf16>

  // conv9: 3x3 s1 p1 228->228 114x114
  %pad49 = tensor.pad %relu48 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x228x114x114xf16> to tensor<1x228x116x116xf16>
  %init50 = tensor.empty() : tensor<1x228x114x114xf16>
  %fill51 = linalg.fill ins(%cst : f16) outs(%init50 : tensor<1x228x114x114xf16>) -> tensor<1x228x114x114xf16>
  %conv52 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad49, %w9 : tensor<1x228x116x116xf16>, tensor<228x228x3x3xf16>)
    outs(%fill51 : tensor<1x228x114x114xf16>) -> tensor<1x228x114x114xf16>
  %empty53 = tensor.empty() : tensor<1x228x114x114xf16>
  %relu54 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv52 : tensor<1x228x114x114xf16>)
    outs(%empty53 : tensor<1x228x114x114xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x228x114x114xf16>

  // conv10: 1x1 s1 p0 228->38 114x114
  %init55 = tensor.empty() : tensor<1x38x114x114xf16>
  %fill56 = linalg.fill ins(%cst : f16) outs(%init55 : tensor<1x38x114x114xf16>) -> tensor<1x38x114x114xf16>
  %conv57 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu54, %w10 : tensor<1x228x114x114xf16>, tensor<38x228x1x1xf16>)
    outs(%fill56 : tensor<1x38x114x114xf16>) -> tensor<1x38x114x114xf16>
  %empty58 = tensor.empty() : tensor<1x38x114x114xf16>
  %relu59 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv57 : tensor<1x38x114x114xf16>)
    outs(%empty58 : tensor<1x38x114x114xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x38x114x114xf16>

  // conv11: 1x1 s1 p0 38->228 114x114
  %init60 = tensor.empty() : tensor<1x228x114x114xf16>
  %fill61 = linalg.fill ins(%cst : f16) outs(%init60 : tensor<1x228x114x114xf16>) -> tensor<1x228x114x114xf16>
  %conv62 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu59, %w11 : tensor<1x38x114x114xf16>, tensor<228x38x1x1xf16>)
    outs(%fill61 : tensor<1x228x114x114xf16>) -> tensor<1x228x114x114xf16>
  %empty63 = tensor.empty() : tensor<1x228x114x114xf16>
  %relu64 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv62 : tensor<1x228x114x114xf16>)
    outs(%empty63 : tensor<1x228x114x114xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x228x114x114xf16>

  // conv12: 3x3 s1 p1 228->228 114x114
  %pad65 = tensor.pad %relu64 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x228x114x114xf16> to tensor<1x228x116x116xf16>
  %init66 = tensor.empty() : tensor<1x228x114x114xf16>
  %fill67 = linalg.fill ins(%cst : f16) outs(%init66 : tensor<1x228x114x114xf16>) -> tensor<1x228x114x114xf16>
  %conv68 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad65, %w12 : tensor<1x228x116x116xf16>, tensor<228x228x3x3xf16>)
    outs(%fill67 : tensor<1x228x114x114xf16>) -> tensor<1x228x114x114xf16>
  %empty69 = tensor.empty() : tensor<1x228x114x114xf16>
  %relu70 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv68 : tensor<1x228x114x114xf16>)
    outs(%empty69 : tensor<1x228x114x114xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x228x114x114xf16>

  // conv13: 1x1 s1 p0 228->38 114x114
  %init71 = tensor.empty() : tensor<1x38x114x114xf16>
  %fill72 = linalg.fill ins(%cst : f16) outs(%init71 : tensor<1x38x114x114xf16>) -> tensor<1x38x114x114xf16>
  %conv73 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu70, %w13 : tensor<1x228x114x114xf16>, tensor<38x228x1x1xf16>)
    outs(%fill72 : tensor<1x38x114x114xf16>) -> tensor<1x38x114x114xf16>
  %empty74 = tensor.empty() : tensor<1x38x114x114xf16>
  %relu75 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv73 : tensor<1x38x114x114xf16>)
    outs(%empty74 : tensor<1x38x114x114xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x38x114x114xf16>

  // conv14: 1x1 s1 p0 38->228 114x114
  %init76 = tensor.empty() : tensor<1x228x114x114xf16>
  %fill77 = linalg.fill ins(%cst : f16) outs(%init76 : tensor<1x228x114x114xf16>) -> tensor<1x228x114x114xf16>
  %conv78 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu75, %w14 : tensor<1x38x114x114xf16>, tensor<228x38x1x1xf16>)
    outs(%fill77 : tensor<1x228x114x114xf16>) -> tensor<1x228x114x114xf16>
  %empty79 = tensor.empty() : tensor<1x228x114x114xf16>
  %relu80 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv78 : tensor<1x228x114x114xf16>)
    outs(%empty79 : tensor<1x228x114x114xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x228x114x114xf16>

  // conv15: 3x3 s1 p1 228->228 114x114
  %pad81 = tensor.pad %relu80 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x228x114x114xf16> to tensor<1x228x116x116xf16>
  %init82 = tensor.empty() : tensor<1x228x114x114xf16>
  %fill83 = linalg.fill ins(%cst : f16) outs(%init82 : tensor<1x228x114x114xf16>) -> tensor<1x228x114x114xf16>
  %conv84 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad81, %w15 : tensor<1x228x116x116xf16>, tensor<228x228x3x3xf16>)
    outs(%fill83 : tensor<1x228x114x114xf16>) -> tensor<1x228x114x114xf16>
  %empty85 = tensor.empty() : tensor<1x228x114x114xf16>
  %relu86 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv84 : tensor<1x228x114x114xf16>)
    outs(%empty85 : tensor<1x228x114x114xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x228x114x114xf16>

  // conv16: 1x1 s1 p0 228->38 114x114
  %init87 = tensor.empty() : tensor<1x38x114x114xf16>
  %fill88 = linalg.fill ins(%cst : f16) outs(%init87 : tensor<1x38x114x114xf16>) -> tensor<1x38x114x114xf16>
  %conv89 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu86, %w16 : tensor<1x228x114x114xf16>, tensor<38x228x1x1xf16>)
    outs(%fill88 : tensor<1x38x114x114xf16>) -> tensor<1x38x114x114xf16>
  %empty90 = tensor.empty() : tensor<1x38x114x114xf16>
  %relu91 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv89 : tensor<1x38x114x114xf16>)
    outs(%empty90 : tensor<1x38x114x114xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x38x114x114xf16>

  // conv17: 1x1 s1 p0 38->228 114x114
  %init92 = tensor.empty() : tensor<1x228x114x114xf16>
  %fill93 = linalg.fill ins(%cst : f16) outs(%init92 : tensor<1x228x114x114xf16>) -> tensor<1x228x114x114xf16>
  %conv94 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu91, %w17 : tensor<1x38x114x114xf16>, tensor<228x38x1x1xf16>)
    outs(%fill93 : tensor<1x228x114x114xf16>) -> tensor<1x228x114x114xf16>
  %empty95 = tensor.empty() : tensor<1x228x114x114xf16>
  %relu96 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv94 : tensor<1x228x114x114xf16>)
    outs(%empty95 : tensor<1x228x114x114xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x228x114x114xf16>

  // conv18: 3x3 s2 p1 228->228 114x114
  %pad97 = tensor.pad %relu96 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x228x114x114xf16> to tensor<1x228x116x116xf16>
  %init98 = tensor.empty() : tensor<1x228x57x57xf16>
  %fill99 = linalg.fill ins(%cst : f16) outs(%init98 : tensor<1x228x57x57xf16>) -> tensor<1x228x57x57xf16>
  %conv100 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad97, %w18 : tensor<1x228x116x116xf16>, tensor<228x228x3x3xf16>)
    outs(%fill99 : tensor<1x228x57x57xf16>) -> tensor<1x228x57x57xf16>
  %empty101 = tensor.empty() : tensor<1x228x57x57xf16>
  %relu102 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv100 : tensor<1x228x57x57xf16>)
    outs(%empty101 : tensor<1x228x57x57xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x228x57x57xf16>

  // conv19: 1x1 s1 p0 228->64 57x57
  %init103 = tensor.empty() : tensor<1x64x57x57xf16>
  %fill104 = linalg.fill ins(%cst : f16) outs(%init103 : tensor<1x64x57x57xf16>) -> tensor<1x64x57x57xf16>
  %conv105 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu102, %w19 : tensor<1x228x57x57xf16>, tensor<64x228x1x1xf16>)
    outs(%fill104 : tensor<1x64x57x57xf16>) -> tensor<1x64x57x57xf16>
  %empty106 = tensor.empty() : tensor<1x64x57x57xf16>
  %relu107 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv105 : tensor<1x64x57x57xf16>)
    outs(%empty106 : tensor<1x64x57x57xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x57x57xf16>

  // conv20: 1x1 s1 p0 64->384 57x57
  %init108 = tensor.empty() : tensor<1x384x57x57xf16>
  %fill109 = linalg.fill ins(%cst : f16) outs(%init108 : tensor<1x384x57x57xf16>) -> tensor<1x384x57x57xf16>
  %conv110 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu107, %w20 : tensor<1x64x57x57xf16>, tensor<384x64x1x1xf16>)
    outs(%fill109 : tensor<1x384x57x57xf16>) -> tensor<1x384x57x57xf16>
  %empty111 = tensor.empty() : tensor<1x384x57x57xf16>
  %relu112 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv110 : tensor<1x384x57x57xf16>)
    outs(%empty111 : tensor<1x384x57x57xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x384x57x57xf16>

  // conv21: 3x3 s1 p1 384->384 57x57
  %pad113 = tensor.pad %relu112 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x384x57x57xf16> to tensor<1x384x59x59xf16>
  %init114 = tensor.empty() : tensor<1x384x57x57xf16>
  %fill115 = linalg.fill ins(%cst : f16) outs(%init114 : tensor<1x384x57x57xf16>) -> tensor<1x384x57x57xf16>
  %conv116 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad113, %w21 : tensor<1x384x59x59xf16>, tensor<384x384x3x3xf16>)
    outs(%fill115 : tensor<1x384x57x57xf16>) -> tensor<1x384x57x57xf16>
  %empty117 = tensor.empty() : tensor<1x384x57x57xf16>
  %relu118 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv116 : tensor<1x384x57x57xf16>)
    outs(%empty117 : tensor<1x384x57x57xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x384x57x57xf16>

  // conv22: 1x1 s1 p0 384->64 57x57
  %init119 = tensor.empty() : tensor<1x64x57x57xf16>
  %fill120 = linalg.fill ins(%cst : f16) outs(%init119 : tensor<1x64x57x57xf16>) -> tensor<1x64x57x57xf16>
  %conv121 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu118, %w22 : tensor<1x384x57x57xf16>, tensor<64x384x1x1xf16>)
    outs(%fill120 : tensor<1x64x57x57xf16>) -> tensor<1x64x57x57xf16>
  %empty122 = tensor.empty() : tensor<1x64x57x57xf16>
  %relu123 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv121 : tensor<1x64x57x57xf16>)
    outs(%empty122 : tensor<1x64x57x57xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x57x57xf16>

  // conv23: 1x1 s1 p0 64->384 57x57
  %init124 = tensor.empty() : tensor<1x384x57x57xf16>
  %fill125 = linalg.fill ins(%cst : f16) outs(%init124 : tensor<1x384x57x57xf16>) -> tensor<1x384x57x57xf16>
  %conv126 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu123, %w23 : tensor<1x64x57x57xf16>, tensor<384x64x1x1xf16>)
    outs(%fill125 : tensor<1x384x57x57xf16>) -> tensor<1x384x57x57xf16>
  %empty127 = tensor.empty() : tensor<1x384x57x57xf16>
  %relu128 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv126 : tensor<1x384x57x57xf16>)
    outs(%empty127 : tensor<1x384x57x57xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x384x57x57xf16>

  // conv24: 3x3 s1 p1 384->384 57x57
  %pad129 = tensor.pad %relu128 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x384x57x57xf16> to tensor<1x384x59x59xf16>
  %init130 = tensor.empty() : tensor<1x384x57x57xf16>
  %fill131 = linalg.fill ins(%cst : f16) outs(%init130 : tensor<1x384x57x57xf16>) -> tensor<1x384x57x57xf16>
  %conv132 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad129, %w24 : tensor<1x384x59x59xf16>, tensor<384x384x3x3xf16>)
    outs(%fill131 : tensor<1x384x57x57xf16>) -> tensor<1x384x57x57xf16>
  %empty133 = tensor.empty() : tensor<1x384x57x57xf16>
  %relu134 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv132 : tensor<1x384x57x57xf16>)
    outs(%empty133 : tensor<1x384x57x57xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x384x57x57xf16>

  // conv25: 1x1 s1 p0 384->64 57x57
  %init135 = tensor.empty() : tensor<1x64x57x57xf16>
  %fill136 = linalg.fill ins(%cst : f16) outs(%init135 : tensor<1x64x57x57xf16>) -> tensor<1x64x57x57xf16>
  %conv137 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu134, %w25 : tensor<1x384x57x57xf16>, tensor<64x384x1x1xf16>)
    outs(%fill136 : tensor<1x64x57x57xf16>) -> tensor<1x64x57x57xf16>
  %empty138 = tensor.empty() : tensor<1x64x57x57xf16>
  %relu139 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv137 : tensor<1x64x57x57xf16>)
    outs(%empty138 : tensor<1x64x57x57xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x57x57xf16>

  // conv26: 1x1 s1 p0 64->384 57x57
  %init140 = tensor.empty() : tensor<1x384x57x57xf16>
  %fill141 = linalg.fill ins(%cst : f16) outs(%init140 : tensor<1x384x57x57xf16>) -> tensor<1x384x57x57xf16>
  %conv142 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu139, %w26 : tensor<1x64x57x57xf16>, tensor<384x64x1x1xf16>)
    outs(%fill141 : tensor<1x384x57x57xf16>) -> tensor<1x384x57x57xf16>
  %empty143 = tensor.empty() : tensor<1x384x57x57xf16>
  %relu144 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv142 : tensor<1x384x57x57xf16>)
    outs(%empty143 : tensor<1x384x57x57xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x384x57x57xf16>

  // conv27: 3x3 s1 p1 384->384 57x57
  %pad145 = tensor.pad %relu144 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x384x57x57xf16> to tensor<1x384x59x59xf16>
  %init146 = tensor.empty() : tensor<1x384x57x57xf16>
  %fill147 = linalg.fill ins(%cst : f16) outs(%init146 : tensor<1x384x57x57xf16>) -> tensor<1x384x57x57xf16>
  %conv148 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad145, %w27 : tensor<1x384x59x59xf16>, tensor<384x384x3x3xf16>)
    outs(%fill147 : tensor<1x384x57x57xf16>) -> tensor<1x384x57x57xf16>
  %empty149 = tensor.empty() : tensor<1x384x57x57xf16>
  %relu150 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv148 : tensor<1x384x57x57xf16>)
    outs(%empty149 : tensor<1x384x57x57xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x384x57x57xf16>

  // conv28: 1x1 s1 p0 384->64 57x57
  %init151 = tensor.empty() : tensor<1x64x57x57xf16>
  %fill152 = linalg.fill ins(%cst : f16) outs(%init151 : tensor<1x64x57x57xf16>) -> tensor<1x64x57x57xf16>
  %conv153 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu150, %w28 : tensor<1x384x57x57xf16>, tensor<64x384x1x1xf16>)
    outs(%fill152 : tensor<1x64x57x57xf16>) -> tensor<1x64x57x57xf16>
  %empty154 = tensor.empty() : tensor<1x64x57x57xf16>
  %relu155 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv153 : tensor<1x64x57x57xf16>)
    outs(%empty154 : tensor<1x64x57x57xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x57x57xf16>

  // conv29: 1x1 s1 p0 64->384 57x57
  %init156 = tensor.empty() : tensor<1x384x57x57xf16>
  %fill157 = linalg.fill ins(%cst : f16) outs(%init156 : tensor<1x384x57x57xf16>) -> tensor<1x384x57x57xf16>
  %conv158 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu155, %w29 : tensor<1x64x57x57xf16>, tensor<384x64x1x1xf16>)
    outs(%fill157 : tensor<1x384x57x57xf16>) -> tensor<1x384x57x57xf16>
  %empty159 = tensor.empty() : tensor<1x384x57x57xf16>
  %relu160 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv158 : tensor<1x384x57x57xf16>)
    outs(%empty159 : tensor<1x384x57x57xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x384x57x57xf16>

  // conv30: 3x3 s2 p1 384->384 57x57
  %pad161 = tensor.pad %relu160 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x384x57x57xf16> to tensor<1x384x59x59xf16>
  %init162 = tensor.empty() : tensor<1x384x29x29xf16>
  %fill163 = linalg.fill ins(%cst : f16) outs(%init162 : tensor<1x384x29x29xf16>) -> tensor<1x384x29x29xf16>
  %conv164 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad161, %w30 : tensor<1x384x59x59xf16>, tensor<384x384x3x3xf16>)
    outs(%fill163 : tensor<1x384x29x29xf16>) -> tensor<1x384x29x29xf16>
  %empty165 = tensor.empty() : tensor<1x384x29x29xf16>
  %relu166 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv164 : tensor<1x384x29x29xf16>)
    outs(%empty165 : tensor<1x384x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x384x29x29xf16>

  // conv31: 1x1 s1 p0 384->128 29x29
  %init167 = tensor.empty() : tensor<1x128x29x29xf16>
  %fill168 = linalg.fill ins(%cst : f16) outs(%init167 : tensor<1x128x29x29xf16>) -> tensor<1x128x29x29xf16>
  %conv169 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu166, %w31 : tensor<1x384x29x29xf16>, tensor<128x384x1x1xf16>)
    outs(%fill168 : tensor<1x128x29x29xf16>) -> tensor<1x128x29x29xf16>
  %empty170 = tensor.empty() : tensor<1x128x29x29xf16>
  %relu171 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv169 : tensor<1x128x29x29xf16>)
    outs(%empty170 : tensor<1x128x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x29x29xf16>

  // conv32: 1x1 s1 p0 128->768 29x29
  %init172 = tensor.empty() : tensor<1x768x29x29xf16>
  %fill173 = linalg.fill ins(%cst : f16) outs(%init172 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %conv174 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu171, %w32 : tensor<1x128x29x29xf16>, tensor<768x128x1x1xf16>)
    outs(%fill173 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %empty175 = tensor.empty() : tensor<1x768x29x29xf16>
  %relu176 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv174 : tensor<1x768x29x29xf16>)
    outs(%empty175 : tensor<1x768x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x29x29xf16>

  // conv33: 3x3 s1 p1 768->768 29x29
  %pad177 = tensor.pad %relu176 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x768x29x29xf16> to tensor<1x768x31x31xf16>
  %init178 = tensor.empty() : tensor<1x768x29x29xf16>
  %fill179 = linalg.fill ins(%cst : f16) outs(%init178 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %conv180 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad177, %w33 : tensor<1x768x31x31xf16>, tensor<768x768x3x3xf16>)
    outs(%fill179 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %empty181 = tensor.empty() : tensor<1x768x29x29xf16>
  %relu182 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv180 : tensor<1x768x29x29xf16>)
    outs(%empty181 : tensor<1x768x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x29x29xf16>

  // conv34: 1x1 s1 p0 768->128 29x29
  %init183 = tensor.empty() : tensor<1x128x29x29xf16>
  %fill184 = linalg.fill ins(%cst : f16) outs(%init183 : tensor<1x128x29x29xf16>) -> tensor<1x128x29x29xf16>
  %conv185 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu182, %w34 : tensor<1x768x29x29xf16>, tensor<128x768x1x1xf16>)
    outs(%fill184 : tensor<1x128x29x29xf16>) -> tensor<1x128x29x29xf16>
  %empty186 = tensor.empty() : tensor<1x128x29x29xf16>
  %relu187 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv185 : tensor<1x128x29x29xf16>)
    outs(%empty186 : tensor<1x128x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x29x29xf16>

  // conv35: 1x1 s1 p0 128->768 29x29
  %init188 = tensor.empty() : tensor<1x768x29x29xf16>
  %fill189 = linalg.fill ins(%cst : f16) outs(%init188 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %conv190 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu187, %w35 : tensor<1x128x29x29xf16>, tensor<768x128x1x1xf16>)
    outs(%fill189 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %empty191 = tensor.empty() : tensor<1x768x29x29xf16>
  %relu192 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv190 : tensor<1x768x29x29xf16>)
    outs(%empty191 : tensor<1x768x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x29x29xf16>

  // conv36: 3x3 s1 p1 768->768 29x29
  %pad193 = tensor.pad %relu192 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x768x29x29xf16> to tensor<1x768x31x31xf16>
  %init194 = tensor.empty() : tensor<1x768x29x29xf16>
  %fill195 = linalg.fill ins(%cst : f16) outs(%init194 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %conv196 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad193, %w36 : tensor<1x768x31x31xf16>, tensor<768x768x3x3xf16>)
    outs(%fill195 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %empty197 = tensor.empty() : tensor<1x768x29x29xf16>
  %relu198 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv196 : tensor<1x768x29x29xf16>)
    outs(%empty197 : tensor<1x768x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x29x29xf16>

  // conv37: 1x1 s1 p0 768->128 29x29
  %init199 = tensor.empty() : tensor<1x128x29x29xf16>
  %fill200 = linalg.fill ins(%cst : f16) outs(%init199 : tensor<1x128x29x29xf16>) -> tensor<1x128x29x29xf16>
  %conv201 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu198, %w37 : tensor<1x768x29x29xf16>, tensor<128x768x1x1xf16>)
    outs(%fill200 : tensor<1x128x29x29xf16>) -> tensor<1x128x29x29xf16>
  %empty202 = tensor.empty() : tensor<1x128x29x29xf16>
  %relu203 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv201 : tensor<1x128x29x29xf16>)
    outs(%empty202 : tensor<1x128x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x29x29xf16>

  // conv38: 1x1 s1 p0 128->768 29x29
  %init204 = tensor.empty() : tensor<1x768x29x29xf16>
  %fill205 = linalg.fill ins(%cst : f16) outs(%init204 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %conv206 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu203, %w38 : tensor<1x128x29x29xf16>, tensor<768x128x1x1xf16>)
    outs(%fill205 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %empty207 = tensor.empty() : tensor<1x768x29x29xf16>
  %relu208 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv206 : tensor<1x768x29x29xf16>)
    outs(%empty207 : tensor<1x768x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x29x29xf16>

  // conv39: 3x3 s1 p1 768->768 29x29
  %pad209 = tensor.pad %relu208 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x768x29x29xf16> to tensor<1x768x31x31xf16>
  %init210 = tensor.empty() : tensor<1x768x29x29xf16>
  %fill211 = linalg.fill ins(%cst : f16) outs(%init210 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %conv212 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad209, %w39 : tensor<1x768x31x31xf16>, tensor<768x768x3x3xf16>)
    outs(%fill211 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %empty213 = tensor.empty() : tensor<1x768x29x29xf16>
  %relu214 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv212 : tensor<1x768x29x29xf16>)
    outs(%empty213 : tensor<1x768x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x29x29xf16>

  // conv40: 1x1 s1 p0 768->128 29x29
  %init215 = tensor.empty() : tensor<1x128x29x29xf16>
  %fill216 = linalg.fill ins(%cst : f16) outs(%init215 : tensor<1x128x29x29xf16>) -> tensor<1x128x29x29xf16>
  %conv217 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu214, %w40 : tensor<1x768x29x29xf16>, tensor<128x768x1x1xf16>)
    outs(%fill216 : tensor<1x128x29x29xf16>) -> tensor<1x128x29x29xf16>
  %empty218 = tensor.empty() : tensor<1x128x29x29xf16>
  %relu219 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv217 : tensor<1x128x29x29xf16>)
    outs(%empty218 : tensor<1x128x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x29x29xf16>

  // conv41: 1x1 s1 p0 128->768 29x29
  %init220 = tensor.empty() : tensor<1x768x29x29xf16>
  %fill221 = linalg.fill ins(%cst : f16) outs(%init220 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %conv222 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu219, %w41 : tensor<1x128x29x29xf16>, tensor<768x128x1x1xf16>)
    outs(%fill221 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %empty223 = tensor.empty() : tensor<1x768x29x29xf16>
  %relu224 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv222 : tensor<1x768x29x29xf16>)
    outs(%empty223 : tensor<1x768x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x29x29xf16>

  // conv42: 3x3 s1 p1 768->768 29x29
  %pad225 = tensor.pad %relu224 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x768x29x29xf16> to tensor<1x768x31x31xf16>
  %init226 = tensor.empty() : tensor<1x768x29x29xf16>
  %fill227 = linalg.fill ins(%cst : f16) outs(%init226 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %conv228 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad225, %w42 : tensor<1x768x31x31xf16>, tensor<768x768x3x3xf16>)
    outs(%fill227 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %empty229 = tensor.empty() : tensor<1x768x29x29xf16>
  %relu230 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv228 : tensor<1x768x29x29xf16>)
    outs(%empty229 : tensor<1x768x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x29x29xf16>

  // conv43: 1x1 s1 p0 768->128 29x29
  %init231 = tensor.empty() : tensor<1x128x29x29xf16>
  %fill232 = linalg.fill ins(%cst : f16) outs(%init231 : tensor<1x128x29x29xf16>) -> tensor<1x128x29x29xf16>
  %conv233 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu230, %w43 : tensor<1x768x29x29xf16>, tensor<128x768x1x1xf16>)
    outs(%fill232 : tensor<1x128x29x29xf16>) -> tensor<1x128x29x29xf16>
  %empty234 = tensor.empty() : tensor<1x128x29x29xf16>
  %relu235 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv233 : tensor<1x128x29x29xf16>)
    outs(%empty234 : tensor<1x128x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x29x29xf16>

  // conv44: 1x1 s1 p0 128->768 29x29
  %init236 = tensor.empty() : tensor<1x768x29x29xf16>
  %fill237 = linalg.fill ins(%cst : f16) outs(%init236 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %conv238 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu235, %w44 : tensor<1x128x29x29xf16>, tensor<768x128x1x1xf16>)
    outs(%fill237 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %empty239 = tensor.empty() : tensor<1x768x29x29xf16>
  %relu240 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv238 : tensor<1x768x29x29xf16>)
    outs(%empty239 : tensor<1x768x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x29x29xf16>

  // conv45: 3x3 s1 p1 768->768 29x29
  %pad241 = tensor.pad %relu240 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x768x29x29xf16> to tensor<1x768x31x31xf16>
  %init242 = tensor.empty() : tensor<1x768x29x29xf16>
  %fill243 = linalg.fill ins(%cst : f16) outs(%init242 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %conv244 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad241, %w45 : tensor<1x768x31x31xf16>, tensor<768x768x3x3xf16>)
    outs(%fill243 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %empty245 = tensor.empty() : tensor<1x768x29x29xf16>
  %relu246 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv244 : tensor<1x768x29x29xf16>)
    outs(%empty245 : tensor<1x768x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x29x29xf16>

  // conv46: 1x1 s1 p0 768->128 29x29
  %init247 = tensor.empty() : tensor<1x128x29x29xf16>
  %fill248 = linalg.fill ins(%cst : f16) outs(%init247 : tensor<1x128x29x29xf16>) -> tensor<1x128x29x29xf16>
  %conv249 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu246, %w46 : tensor<1x768x29x29xf16>, tensor<128x768x1x1xf16>)
    outs(%fill248 : tensor<1x128x29x29xf16>) -> tensor<1x128x29x29xf16>
  %empty250 = tensor.empty() : tensor<1x128x29x29xf16>
  %relu251 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv249 : tensor<1x128x29x29xf16>)
    outs(%empty250 : tensor<1x128x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x29x29xf16>

  // conv47: 1x1 s1 p0 128->768 29x29
  %init252 = tensor.empty() : tensor<1x768x29x29xf16>
  %fill253 = linalg.fill ins(%cst : f16) outs(%init252 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %conv254 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu251, %w47 : tensor<1x128x29x29xf16>, tensor<768x128x1x1xf16>)
    outs(%fill253 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %empty255 = tensor.empty() : tensor<1x768x29x29xf16>
  %relu256 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv254 : tensor<1x768x29x29xf16>)
    outs(%empty255 : tensor<1x768x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x29x29xf16>

  // conv48: 3x3 s1 p1 768->768 29x29
  %pad257 = tensor.pad %relu256 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x768x29x29xf16> to tensor<1x768x31x31xf16>
  %init258 = tensor.empty() : tensor<1x768x29x29xf16>
  %fill259 = linalg.fill ins(%cst : f16) outs(%init258 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %conv260 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad257, %w48 : tensor<1x768x31x31xf16>, tensor<768x768x3x3xf16>)
    outs(%fill259 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %empty261 = tensor.empty() : tensor<1x768x29x29xf16>
  %relu262 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv260 : tensor<1x768x29x29xf16>)
    outs(%empty261 : tensor<1x768x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x29x29xf16>

  // conv49: 1x1 s1 p0 768->128 29x29
  %init263 = tensor.empty() : tensor<1x128x29x29xf16>
  %fill264 = linalg.fill ins(%cst : f16) outs(%init263 : tensor<1x128x29x29xf16>) -> tensor<1x128x29x29xf16>
  %conv265 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu262, %w49 : tensor<1x768x29x29xf16>, tensor<128x768x1x1xf16>)
    outs(%fill264 : tensor<1x128x29x29xf16>) -> tensor<1x128x29x29xf16>
  %empty266 = tensor.empty() : tensor<1x128x29x29xf16>
  %relu267 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv265 : tensor<1x128x29x29xf16>)
    outs(%empty266 : tensor<1x128x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x29x29xf16>

  // conv50: 1x1 s1 p0 128->768 29x29
  %init268 = tensor.empty() : tensor<1x768x29x29xf16>
  %fill269 = linalg.fill ins(%cst : f16) outs(%init268 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %conv270 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu267, %w50 : tensor<1x128x29x29xf16>, tensor<768x128x1x1xf16>)
    outs(%fill269 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %empty271 = tensor.empty() : tensor<1x768x29x29xf16>
  %relu272 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv270 : tensor<1x768x29x29xf16>)
    outs(%empty271 : tensor<1x768x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x29x29xf16>

  // conv51: 3x3 s1 p1 768->768 29x29
  %pad273 = tensor.pad %relu272 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x768x29x29xf16> to tensor<1x768x31x31xf16>
  %init274 = tensor.empty() : tensor<1x768x29x29xf16>
  %fill275 = linalg.fill ins(%cst : f16) outs(%init274 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %conv276 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad273, %w51 : tensor<1x768x31x31xf16>, tensor<768x768x3x3xf16>)
    outs(%fill275 : tensor<1x768x29x29xf16>) -> tensor<1x768x29x29xf16>
  %empty277 = tensor.empty() : tensor<1x768x29x29xf16>
  %relu278 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv276 : tensor<1x768x29x29xf16>)
    outs(%empty277 : tensor<1x768x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x29x29xf16>

  // conv52: 1x1 s1 p0 768->179 29x29
  %init279 = tensor.empty() : tensor<1x179x29x29xf16>
  %fill280 = linalg.fill ins(%cst : f16) outs(%init279 : tensor<1x179x29x29xf16>) -> tensor<1x179x29x29xf16>
  %conv281 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu278, %w52 : tensor<1x768x29x29xf16>, tensor<179x768x1x1xf16>)
    outs(%fill280 : tensor<1x179x29x29xf16>) -> tensor<1x179x29x29xf16>
  %empty282 = tensor.empty() : tensor<1x179x29x29xf16>
  %relu283 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv281 : tensor<1x179x29x29xf16>)
    outs(%empty282 : tensor<1x179x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x179x29x29xf16>

  // conv53: 1x1 s1 p0 179->1074 29x29
  %init284 = tensor.empty() : tensor<1x1074x29x29xf16>
  %fill285 = linalg.fill ins(%cst : f16) outs(%init284 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %conv286 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu283, %w53 : tensor<1x179x29x29xf16>, tensor<1074x179x1x1xf16>)
    outs(%fill285 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %empty287 = tensor.empty() : tensor<1x1074x29x29xf16>
  %relu288 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv286 : tensor<1x1074x29x29xf16>)
    outs(%empty287 : tensor<1x1074x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1074x29x29xf16>

  // conv54: 3x3 s1 p1 1074->1074 29x29
  %pad289 = tensor.pad %relu288 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1074x29x29xf16> to tensor<1x1074x31x31xf16>
  %init290 = tensor.empty() : tensor<1x1074x29x29xf16>
  %fill291 = linalg.fill ins(%cst : f16) outs(%init290 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %conv292 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad289, %w54 : tensor<1x1074x31x31xf16>, tensor<1074x1074x3x3xf16>)
    outs(%fill291 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %empty293 = tensor.empty() : tensor<1x1074x29x29xf16>
  %relu294 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv292 : tensor<1x1074x29x29xf16>)
    outs(%empty293 : tensor<1x1074x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1074x29x29xf16>

  // conv55: 1x1 s1 p0 1074->179 29x29
  %init295 = tensor.empty() : tensor<1x179x29x29xf16>
  %fill296 = linalg.fill ins(%cst : f16) outs(%init295 : tensor<1x179x29x29xf16>) -> tensor<1x179x29x29xf16>
  %conv297 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu294, %w55 : tensor<1x1074x29x29xf16>, tensor<179x1074x1x1xf16>)
    outs(%fill296 : tensor<1x179x29x29xf16>) -> tensor<1x179x29x29xf16>
  %empty298 = tensor.empty() : tensor<1x179x29x29xf16>
  %relu299 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv297 : tensor<1x179x29x29xf16>)
    outs(%empty298 : tensor<1x179x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x179x29x29xf16>

  // conv56: 1x1 s1 p0 179->1074 29x29
  %init300 = tensor.empty() : tensor<1x1074x29x29xf16>
  %fill301 = linalg.fill ins(%cst : f16) outs(%init300 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %conv302 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu299, %w56 : tensor<1x179x29x29xf16>, tensor<1074x179x1x1xf16>)
    outs(%fill301 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %empty303 = tensor.empty() : tensor<1x1074x29x29xf16>
  %relu304 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv302 : tensor<1x1074x29x29xf16>)
    outs(%empty303 : tensor<1x1074x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1074x29x29xf16>

  // conv57: 3x3 s1 p1 1074->1074 29x29
  %pad305 = tensor.pad %relu304 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1074x29x29xf16> to tensor<1x1074x31x31xf16>
  %init306 = tensor.empty() : tensor<1x1074x29x29xf16>
  %fill307 = linalg.fill ins(%cst : f16) outs(%init306 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %conv308 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad305, %w57 : tensor<1x1074x31x31xf16>, tensor<1074x1074x3x3xf16>)
    outs(%fill307 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %empty309 = tensor.empty() : tensor<1x1074x29x29xf16>
  %relu310 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv308 : tensor<1x1074x29x29xf16>)
    outs(%empty309 : tensor<1x1074x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1074x29x29xf16>

  // conv58: 1x1 s1 p0 1074->179 29x29
  %init311 = tensor.empty() : tensor<1x179x29x29xf16>
  %fill312 = linalg.fill ins(%cst : f16) outs(%init311 : tensor<1x179x29x29xf16>) -> tensor<1x179x29x29xf16>
  %conv313 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu310, %w58 : tensor<1x1074x29x29xf16>, tensor<179x1074x1x1xf16>)
    outs(%fill312 : tensor<1x179x29x29xf16>) -> tensor<1x179x29x29xf16>
  %empty314 = tensor.empty() : tensor<1x179x29x29xf16>
  %relu315 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv313 : tensor<1x179x29x29xf16>)
    outs(%empty314 : tensor<1x179x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x179x29x29xf16>

  // conv59: 1x1 s1 p0 179->1074 29x29
  %init316 = tensor.empty() : tensor<1x1074x29x29xf16>
  %fill317 = linalg.fill ins(%cst : f16) outs(%init316 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %conv318 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu315, %w59 : tensor<1x179x29x29xf16>, tensor<1074x179x1x1xf16>)
    outs(%fill317 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %empty319 = tensor.empty() : tensor<1x1074x29x29xf16>
  %relu320 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv318 : tensor<1x1074x29x29xf16>)
    outs(%empty319 : tensor<1x1074x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1074x29x29xf16>

  // conv60: 3x3 s1 p1 1074->1074 29x29
  %pad321 = tensor.pad %relu320 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1074x29x29xf16> to tensor<1x1074x31x31xf16>
  %init322 = tensor.empty() : tensor<1x1074x29x29xf16>
  %fill323 = linalg.fill ins(%cst : f16) outs(%init322 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %conv324 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad321, %w60 : tensor<1x1074x31x31xf16>, tensor<1074x1074x3x3xf16>)
    outs(%fill323 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %empty325 = tensor.empty() : tensor<1x1074x29x29xf16>
  %relu326 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv324 : tensor<1x1074x29x29xf16>)
    outs(%empty325 : tensor<1x1074x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1074x29x29xf16>

  // conv61: 1x1 s1 p0 1074->179 29x29
  %init327 = tensor.empty() : tensor<1x179x29x29xf16>
  %fill328 = linalg.fill ins(%cst : f16) outs(%init327 : tensor<1x179x29x29xf16>) -> tensor<1x179x29x29xf16>
  %conv329 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu326, %w61 : tensor<1x1074x29x29xf16>, tensor<179x1074x1x1xf16>)
    outs(%fill328 : tensor<1x179x29x29xf16>) -> tensor<1x179x29x29xf16>
  %empty330 = tensor.empty() : tensor<1x179x29x29xf16>
  %relu331 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv329 : tensor<1x179x29x29xf16>)
    outs(%empty330 : tensor<1x179x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x179x29x29xf16>

  // conv62: 1x1 s1 p0 179->1074 29x29
  %init332 = tensor.empty() : tensor<1x1074x29x29xf16>
  %fill333 = linalg.fill ins(%cst : f16) outs(%init332 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %conv334 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu331, %w62 : tensor<1x179x29x29xf16>, tensor<1074x179x1x1xf16>)
    outs(%fill333 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %empty335 = tensor.empty() : tensor<1x1074x29x29xf16>
  %relu336 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv334 : tensor<1x1074x29x29xf16>)
    outs(%empty335 : tensor<1x1074x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1074x29x29xf16>

  // conv63: 3x3 s1 p1 1074->1074 29x29
  %pad337 = tensor.pad %relu336 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1074x29x29xf16> to tensor<1x1074x31x31xf16>
  %init338 = tensor.empty() : tensor<1x1074x29x29xf16>
  %fill339 = linalg.fill ins(%cst : f16) outs(%init338 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %conv340 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad337, %w63 : tensor<1x1074x31x31xf16>, tensor<1074x1074x3x3xf16>)
    outs(%fill339 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %empty341 = tensor.empty() : tensor<1x1074x29x29xf16>
  %relu342 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv340 : tensor<1x1074x29x29xf16>)
    outs(%empty341 : tensor<1x1074x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1074x29x29xf16>

  // conv64: 1x1 s1 p0 1074->179 29x29
  %init343 = tensor.empty() : tensor<1x179x29x29xf16>
  %fill344 = linalg.fill ins(%cst : f16) outs(%init343 : tensor<1x179x29x29xf16>) -> tensor<1x179x29x29xf16>
  %conv345 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu342, %w64 : tensor<1x1074x29x29xf16>, tensor<179x1074x1x1xf16>)
    outs(%fill344 : tensor<1x179x29x29xf16>) -> tensor<1x179x29x29xf16>
  %empty346 = tensor.empty() : tensor<1x179x29x29xf16>
  %relu347 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv345 : tensor<1x179x29x29xf16>)
    outs(%empty346 : tensor<1x179x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x179x29x29xf16>

  // conv65: 1x1 s1 p0 179->1074 29x29
  %init348 = tensor.empty() : tensor<1x1074x29x29xf16>
  %fill349 = linalg.fill ins(%cst : f16) outs(%init348 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %conv350 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu347, %w65 : tensor<1x179x29x29xf16>, tensor<1074x179x1x1xf16>)
    outs(%fill349 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %empty351 = tensor.empty() : tensor<1x1074x29x29xf16>
  %relu352 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv350 : tensor<1x1074x29x29xf16>)
    outs(%empty351 : tensor<1x1074x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1074x29x29xf16>

  // conv66: 3x3 s1 p1 1074->1074 29x29
  %pad353 = tensor.pad %relu352 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1074x29x29xf16> to tensor<1x1074x31x31xf16>
  %init354 = tensor.empty() : tensor<1x1074x29x29xf16>
  %fill355 = linalg.fill ins(%cst : f16) outs(%init354 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %conv356 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad353, %w66 : tensor<1x1074x31x31xf16>, tensor<1074x1074x3x3xf16>)
    outs(%fill355 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %empty357 = tensor.empty() : tensor<1x1074x29x29xf16>
  %relu358 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv356 : tensor<1x1074x29x29xf16>)
    outs(%empty357 : tensor<1x1074x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1074x29x29xf16>

  // conv67: 1x1 s1 p0 1074->179 29x29
  %init359 = tensor.empty() : tensor<1x179x29x29xf16>
  %fill360 = linalg.fill ins(%cst : f16) outs(%init359 : tensor<1x179x29x29xf16>) -> tensor<1x179x29x29xf16>
  %conv361 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu358, %w67 : tensor<1x1074x29x29xf16>, tensor<179x1074x1x1xf16>)
    outs(%fill360 : tensor<1x179x29x29xf16>) -> tensor<1x179x29x29xf16>
  %empty362 = tensor.empty() : tensor<1x179x29x29xf16>
  %relu363 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv361 : tensor<1x179x29x29xf16>)
    outs(%empty362 : tensor<1x179x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x179x29x29xf16>

  // conv68: 1x1 s1 p0 179->1074 29x29
  %init364 = tensor.empty() : tensor<1x1074x29x29xf16>
  %fill365 = linalg.fill ins(%cst : f16) outs(%init364 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %conv366 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu363, %w68 : tensor<1x179x29x29xf16>, tensor<1074x179x1x1xf16>)
    outs(%fill365 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %empty367 = tensor.empty() : tensor<1x1074x29x29xf16>
  %relu368 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv366 : tensor<1x1074x29x29xf16>)
    outs(%empty367 : tensor<1x1074x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1074x29x29xf16>

  // conv69: 3x3 s1 p1 1074->1074 29x29
  %pad369 = tensor.pad %relu368 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1074x29x29xf16> to tensor<1x1074x31x31xf16>
  %init370 = tensor.empty() : tensor<1x1074x29x29xf16>
  %fill371 = linalg.fill ins(%cst : f16) outs(%init370 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %conv372 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad369, %w69 : tensor<1x1074x31x31xf16>, tensor<1074x1074x3x3xf16>)
    outs(%fill371 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %empty373 = tensor.empty() : tensor<1x1074x29x29xf16>
  %relu374 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv372 : tensor<1x1074x29x29xf16>)
    outs(%empty373 : tensor<1x1074x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1074x29x29xf16>

  // conv70: 1x1 s1 p0 1074->179 29x29
  %init375 = tensor.empty() : tensor<1x179x29x29xf16>
  %fill376 = linalg.fill ins(%cst : f16) outs(%init375 : tensor<1x179x29x29xf16>) -> tensor<1x179x29x29xf16>
  %conv377 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu374, %w70 : tensor<1x1074x29x29xf16>, tensor<179x1074x1x1xf16>)
    outs(%fill376 : tensor<1x179x29x29xf16>) -> tensor<1x179x29x29xf16>
  %empty378 = tensor.empty() : tensor<1x179x29x29xf16>
  %relu379 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv377 : tensor<1x179x29x29xf16>)
    outs(%empty378 : tensor<1x179x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x179x29x29xf16>

  // conv71: 1x1 s1 p0 179->1074 29x29
  %init380 = tensor.empty() : tensor<1x1074x29x29xf16>
  %fill381 = linalg.fill ins(%cst : f16) outs(%init380 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %conv382 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu379, %w71 : tensor<1x179x29x29xf16>, tensor<1074x179x1x1xf16>)
    outs(%fill381 : tensor<1x1074x29x29xf16>) -> tensor<1x1074x29x29xf16>
  %empty383 = tensor.empty() : tensor<1x1074x29x29xf16>
  %relu384 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv382 : tensor<1x1074x29x29xf16>)
    outs(%empty383 : tensor<1x1074x29x29xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1074x29x29xf16>

  // conv72: 3x3 s2 p1 1074->1074 29x29
  %pad385 = tensor.pad %relu384 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1074x29x29xf16> to tensor<1x1074x31x31xf16>
  %init386 = tensor.empty() : tensor<1x1074x15x15xf16>
  %fill387 = linalg.fill ins(%cst : f16) outs(%init386 : tensor<1x1074x15x15xf16>) -> tensor<1x1074x15x15xf16>
  %conv388 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad385, %w72 : tensor<1x1074x31x31xf16>, tensor<1074x1074x3x3xf16>)
    outs(%fill387 : tensor<1x1074x15x15xf16>) -> tensor<1x1074x15x15xf16>
  %empty389 = tensor.empty() : tensor<1x1074x15x15xf16>
  %relu390 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv388 : tensor<1x1074x15x15xf16>)
    outs(%empty389 : tensor<1x1074x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1074x15x15xf16>

  // conv73: 1x1 s1 p0 1074->307 15x15
  %init391 = tensor.empty() : tensor<1x307x15x15xf16>
  %fill392 = linalg.fill ins(%cst : f16) outs(%init391 : tensor<1x307x15x15xf16>) -> tensor<1x307x15x15xf16>
  %conv393 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu390, %w73 : tensor<1x1074x15x15xf16>, tensor<307x1074x1x1xf16>)
    outs(%fill392 : tensor<1x307x15x15xf16>) -> tensor<1x307x15x15xf16>
  %empty394 = tensor.empty() : tensor<1x307x15x15xf16>
  %relu395 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv393 : tensor<1x307x15x15xf16>)
    outs(%empty394 : tensor<1x307x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x307x15x15xf16>

  // conv74: 1x1 s1 p0 307->1842 15x15
  %init396 = tensor.empty() : tensor<1x1842x15x15xf16>
  %fill397 = linalg.fill ins(%cst : f16) outs(%init396 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %conv398 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu395, %w74 : tensor<1x307x15x15xf16>, tensor<1842x307x1x1xf16>)
    outs(%fill397 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %empty399 = tensor.empty() : tensor<1x1842x15x15xf16>
  %relu400 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv398 : tensor<1x1842x15x15xf16>)
    outs(%empty399 : tensor<1x1842x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1842x15x15xf16>

  // conv75: 3x3 s1 p1 1842->1842 15x15
  %pad401 = tensor.pad %relu400 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1842x15x15xf16> to tensor<1x1842x17x17xf16>
  %init402 = tensor.empty() : tensor<1x1842x15x15xf16>
  %fill403 = linalg.fill ins(%cst : f16) outs(%init402 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %conv404 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad401, %w75 : tensor<1x1842x17x17xf16>, tensor<1842x1842x3x3xf16>)
    outs(%fill403 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %empty405 = tensor.empty() : tensor<1x1842x15x15xf16>
  %relu406 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv404 : tensor<1x1842x15x15xf16>)
    outs(%empty405 : tensor<1x1842x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1842x15x15xf16>

  // conv76: 1x1 s1 p0 1842->307 15x15
  %init407 = tensor.empty() : tensor<1x307x15x15xf16>
  %fill408 = linalg.fill ins(%cst : f16) outs(%init407 : tensor<1x307x15x15xf16>) -> tensor<1x307x15x15xf16>
  %conv409 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu406, %w76 : tensor<1x1842x15x15xf16>, tensor<307x1842x1x1xf16>)
    outs(%fill408 : tensor<1x307x15x15xf16>) -> tensor<1x307x15x15xf16>
  %empty410 = tensor.empty() : tensor<1x307x15x15xf16>
  %relu411 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv409 : tensor<1x307x15x15xf16>)
    outs(%empty410 : tensor<1x307x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x307x15x15xf16>

  // conv77: 1x1 s1 p0 307->1842 15x15
  %init412 = tensor.empty() : tensor<1x1842x15x15xf16>
  %fill413 = linalg.fill ins(%cst : f16) outs(%init412 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %conv414 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu411, %w77 : tensor<1x307x15x15xf16>, tensor<1842x307x1x1xf16>)
    outs(%fill413 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %empty415 = tensor.empty() : tensor<1x1842x15x15xf16>
  %relu416 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv414 : tensor<1x1842x15x15xf16>)
    outs(%empty415 : tensor<1x1842x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1842x15x15xf16>

  // conv78: 3x3 s1 p1 1842->1842 15x15
  %pad417 = tensor.pad %relu416 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1842x15x15xf16> to tensor<1x1842x17x17xf16>
  %init418 = tensor.empty() : tensor<1x1842x15x15xf16>
  %fill419 = linalg.fill ins(%cst : f16) outs(%init418 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %conv420 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad417, %w78 : tensor<1x1842x17x17xf16>, tensor<1842x1842x3x3xf16>)
    outs(%fill419 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %empty421 = tensor.empty() : tensor<1x1842x15x15xf16>
  %relu422 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv420 : tensor<1x1842x15x15xf16>)
    outs(%empty421 : tensor<1x1842x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1842x15x15xf16>

  // conv79: 1x1 s1 p0 1842->307 15x15
  %init423 = tensor.empty() : tensor<1x307x15x15xf16>
  %fill424 = linalg.fill ins(%cst : f16) outs(%init423 : tensor<1x307x15x15xf16>) -> tensor<1x307x15x15xf16>
  %conv425 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu422, %w79 : tensor<1x1842x15x15xf16>, tensor<307x1842x1x1xf16>)
    outs(%fill424 : tensor<1x307x15x15xf16>) -> tensor<1x307x15x15xf16>
  %empty426 = tensor.empty() : tensor<1x307x15x15xf16>
  %relu427 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv425 : tensor<1x307x15x15xf16>)
    outs(%empty426 : tensor<1x307x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x307x15x15xf16>

  // conv80: 1x1 s1 p0 307->1842 15x15
  %init428 = tensor.empty() : tensor<1x1842x15x15xf16>
  %fill429 = linalg.fill ins(%cst : f16) outs(%init428 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %conv430 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu427, %w80 : tensor<1x307x15x15xf16>, tensor<1842x307x1x1xf16>)
    outs(%fill429 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %empty431 = tensor.empty() : tensor<1x1842x15x15xf16>
  %relu432 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv430 : tensor<1x1842x15x15xf16>)
    outs(%empty431 : tensor<1x1842x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1842x15x15xf16>

  // conv81: 3x3 s1 p1 1842->1842 15x15
  %pad433 = tensor.pad %relu432 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1842x15x15xf16> to tensor<1x1842x17x17xf16>
  %init434 = tensor.empty() : tensor<1x1842x15x15xf16>
  %fill435 = linalg.fill ins(%cst : f16) outs(%init434 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %conv436 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad433, %w81 : tensor<1x1842x17x17xf16>, tensor<1842x1842x3x3xf16>)
    outs(%fill435 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %empty437 = tensor.empty() : tensor<1x1842x15x15xf16>
  %relu438 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv436 : tensor<1x1842x15x15xf16>)
    outs(%empty437 : tensor<1x1842x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1842x15x15xf16>

  // conv82: 1x1 s1 p0 1842->307 15x15
  %init439 = tensor.empty() : tensor<1x307x15x15xf16>
  %fill440 = linalg.fill ins(%cst : f16) outs(%init439 : tensor<1x307x15x15xf16>) -> tensor<1x307x15x15xf16>
  %conv441 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu438, %w82 : tensor<1x1842x15x15xf16>, tensor<307x1842x1x1xf16>)
    outs(%fill440 : tensor<1x307x15x15xf16>) -> tensor<1x307x15x15xf16>
  %empty442 = tensor.empty() : tensor<1x307x15x15xf16>
  %relu443 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv441 : tensor<1x307x15x15xf16>)
    outs(%empty442 : tensor<1x307x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x307x15x15xf16>

  // conv83: 1x1 s1 p0 307->1842 15x15
  %init444 = tensor.empty() : tensor<1x1842x15x15xf16>
  %fill445 = linalg.fill ins(%cst : f16) outs(%init444 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %conv446 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu443, %w83 : tensor<1x307x15x15xf16>, tensor<1842x307x1x1xf16>)
    outs(%fill445 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %empty447 = tensor.empty() : tensor<1x1842x15x15xf16>
  %relu448 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv446 : tensor<1x1842x15x15xf16>)
    outs(%empty447 : tensor<1x1842x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1842x15x15xf16>

  // conv84: 3x3 s1 p1 1842->1842 15x15
  %pad449 = tensor.pad %relu448 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1842x15x15xf16> to tensor<1x1842x17x17xf16>
  %init450 = tensor.empty() : tensor<1x1842x15x15xf16>
  %fill451 = linalg.fill ins(%cst : f16) outs(%init450 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %conv452 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad449, %w84 : tensor<1x1842x17x17xf16>, tensor<1842x1842x3x3xf16>)
    outs(%fill451 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %empty453 = tensor.empty() : tensor<1x1842x15x15xf16>
  %relu454 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv452 : tensor<1x1842x15x15xf16>)
    outs(%empty453 : tensor<1x1842x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1842x15x15xf16>

  // conv85: 1x1 s1 p0 1842->307 15x15
  %init455 = tensor.empty() : tensor<1x307x15x15xf16>
  %fill456 = linalg.fill ins(%cst : f16) outs(%init455 : tensor<1x307x15x15xf16>) -> tensor<1x307x15x15xf16>
  %conv457 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu454, %w85 : tensor<1x1842x15x15xf16>, tensor<307x1842x1x1xf16>)
    outs(%fill456 : tensor<1x307x15x15xf16>) -> tensor<1x307x15x15xf16>
  %empty458 = tensor.empty() : tensor<1x307x15x15xf16>
  %relu459 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv457 : tensor<1x307x15x15xf16>)
    outs(%empty458 : tensor<1x307x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x307x15x15xf16>

  // conv86: 1x1 s1 p0 307->1842 15x15
  %init460 = tensor.empty() : tensor<1x1842x15x15xf16>
  %fill461 = linalg.fill ins(%cst : f16) outs(%init460 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %conv462 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu459, %w86 : tensor<1x307x15x15xf16>, tensor<1842x307x1x1xf16>)
    outs(%fill461 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %empty463 = tensor.empty() : tensor<1x1842x15x15xf16>
  %relu464 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv462 : tensor<1x1842x15x15xf16>)
    outs(%empty463 : tensor<1x1842x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1842x15x15xf16>

  // conv87: 3x3 s1 p1 1842->1842 15x15
  %pad465 = tensor.pad %relu464 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1842x15x15xf16> to tensor<1x1842x17x17xf16>
  %init466 = tensor.empty() : tensor<1x1842x15x15xf16>
  %fill467 = linalg.fill ins(%cst : f16) outs(%init466 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %conv468 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad465, %w87 : tensor<1x1842x17x17xf16>, tensor<1842x1842x3x3xf16>)
    outs(%fill467 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %empty469 = tensor.empty() : tensor<1x1842x15x15xf16>
  %relu470 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv468 : tensor<1x1842x15x15xf16>)
    outs(%empty469 : tensor<1x1842x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1842x15x15xf16>

  // conv88: 1x1 s1 p0 1842->307 15x15
  %init471 = tensor.empty() : tensor<1x307x15x15xf16>
  %fill472 = linalg.fill ins(%cst : f16) outs(%init471 : tensor<1x307x15x15xf16>) -> tensor<1x307x15x15xf16>
  %conv473 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu470, %w88 : tensor<1x1842x15x15xf16>, tensor<307x1842x1x1xf16>)
    outs(%fill472 : tensor<1x307x15x15xf16>) -> tensor<1x307x15x15xf16>
  %empty474 = tensor.empty() : tensor<1x307x15x15xf16>
  %relu475 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv473 : tensor<1x307x15x15xf16>)
    outs(%empty474 : tensor<1x307x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x307x15x15xf16>

  // conv89: 1x1 s1 p0 307->1842 15x15
  %init476 = tensor.empty() : tensor<1x1842x15x15xf16>
  %fill477 = linalg.fill ins(%cst : f16) outs(%init476 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %conv478 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu475, %w89 : tensor<1x307x15x15xf16>, tensor<1842x307x1x1xf16>)
    outs(%fill477 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %empty479 = tensor.empty() : tensor<1x1842x15x15xf16>
  %relu480 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv478 : tensor<1x1842x15x15xf16>)
    outs(%empty479 : tensor<1x1842x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1842x15x15xf16>

  // conv90: 3x3 s1 p1 1842->1842 15x15
  %pad481 = tensor.pad %relu480 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1842x15x15xf16> to tensor<1x1842x17x17xf16>
  %init482 = tensor.empty() : tensor<1x1842x15x15xf16>
  %fill483 = linalg.fill ins(%cst : f16) outs(%init482 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %conv484 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad481, %w90 : tensor<1x1842x17x17xf16>, tensor<1842x1842x3x3xf16>)
    outs(%fill483 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %empty485 = tensor.empty() : tensor<1x1842x15x15xf16>
  %relu486 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv484 : tensor<1x1842x15x15xf16>)
    outs(%empty485 : tensor<1x1842x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1842x15x15xf16>

  // conv91: 1x1 s1 p0 1842->307 15x15
  %init487 = tensor.empty() : tensor<1x307x15x15xf16>
  %fill488 = linalg.fill ins(%cst : f16) outs(%init487 : tensor<1x307x15x15xf16>) -> tensor<1x307x15x15xf16>
  %conv489 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu486, %w91 : tensor<1x1842x15x15xf16>, tensor<307x1842x1x1xf16>)
    outs(%fill488 : tensor<1x307x15x15xf16>) -> tensor<1x307x15x15xf16>
  %empty490 = tensor.empty() : tensor<1x307x15x15xf16>
  %relu491 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv489 : tensor<1x307x15x15xf16>)
    outs(%empty490 : tensor<1x307x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x307x15x15xf16>

  // conv92: 1x1 s1 p0 307->1842 15x15
  %init492 = tensor.empty() : tensor<1x1842x15x15xf16>
  %fill493 = linalg.fill ins(%cst : f16) outs(%init492 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %conv494 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu491, %w92 : tensor<1x307x15x15xf16>, tensor<1842x307x1x1xf16>)
    outs(%fill493 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %empty495 = tensor.empty() : tensor<1x1842x15x15xf16>
  %relu496 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv494 : tensor<1x1842x15x15xf16>)
    outs(%empty495 : tensor<1x1842x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1842x15x15xf16>

  // conv93: 3x3 s1 p1 1842->1842 15x15
  %pad497 = tensor.pad %relu496 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1842x15x15xf16> to tensor<1x1842x17x17xf16>
  %init498 = tensor.empty() : tensor<1x1842x15x15xf16>
  %fill499 = linalg.fill ins(%cst : f16) outs(%init498 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %conv500 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad497, %w93 : tensor<1x1842x17x17xf16>, tensor<1842x1842x3x3xf16>)
    outs(%fill499 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %empty501 = tensor.empty() : tensor<1x1842x15x15xf16>
  %relu502 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv500 : tensor<1x1842x15x15xf16>)
    outs(%empty501 : tensor<1x1842x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1842x15x15xf16>

  // conv94: 1x1 s1 p0 1842->307 15x15
  %init503 = tensor.empty() : tensor<1x307x15x15xf16>
  %fill504 = linalg.fill ins(%cst : f16) outs(%init503 : tensor<1x307x15x15xf16>) -> tensor<1x307x15x15xf16>
  %conv505 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu502, %w94 : tensor<1x1842x15x15xf16>, tensor<307x1842x1x1xf16>)
    outs(%fill504 : tensor<1x307x15x15xf16>) -> tensor<1x307x15x15xf16>
  %empty506 = tensor.empty() : tensor<1x307x15x15xf16>
  %relu507 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv505 : tensor<1x307x15x15xf16>)
    outs(%empty506 : tensor<1x307x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x307x15x15xf16>

  // conv95: 1x1 s1 p0 307->1842 15x15
  %init508 = tensor.empty() : tensor<1x1842x15x15xf16>
  %fill509 = linalg.fill ins(%cst : f16) outs(%init508 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %conv510 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu507, %w95 : tensor<1x307x15x15xf16>, tensor<1842x307x1x1xf16>)
    outs(%fill509 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %empty511 = tensor.empty() : tensor<1x1842x15x15xf16>
  %relu512 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv510 : tensor<1x1842x15x15xf16>)
    outs(%empty511 : tensor<1x1842x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1842x15x15xf16>

  // conv96: 3x3 s1 p1 1842->1842 15x15
  %pad513 = tensor.pad %relu512 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1842x15x15xf16> to tensor<1x1842x17x17xf16>
  %init514 = tensor.empty() : tensor<1x1842x15x15xf16>
  %fill515 = linalg.fill ins(%cst : f16) outs(%init514 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %conv516 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad513, %w96 : tensor<1x1842x17x17xf16>, tensor<1842x1842x3x3xf16>)
    outs(%fill515 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %empty517 = tensor.empty() : tensor<1x1842x15x15xf16>
  %relu518 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv516 : tensor<1x1842x15x15xf16>)
    outs(%empty517 : tensor<1x1842x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1842x15x15xf16>

  // conv97: 1x1 s1 p0 1842->307 15x15
  %init519 = tensor.empty() : tensor<1x307x15x15xf16>
  %fill520 = linalg.fill ins(%cst : f16) outs(%init519 : tensor<1x307x15x15xf16>) -> tensor<1x307x15x15xf16>
  %conv521 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu518, %w97 : tensor<1x1842x15x15xf16>, tensor<307x1842x1x1xf16>)
    outs(%fill520 : tensor<1x307x15x15xf16>) -> tensor<1x307x15x15xf16>
  %empty522 = tensor.empty() : tensor<1x307x15x15xf16>
  %relu523 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv521 : tensor<1x307x15x15xf16>)
    outs(%empty522 : tensor<1x307x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x307x15x15xf16>

  // conv98: 1x1 s1 p0 307->1842 15x15
  %init524 = tensor.empty() : tensor<1x1842x15x15xf16>
  %fill525 = linalg.fill ins(%cst : f16) outs(%init524 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %conv526 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu523, %w98 : tensor<1x307x15x15xf16>, tensor<1842x307x1x1xf16>)
    outs(%fill525 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %empty527 = tensor.empty() : tensor<1x1842x15x15xf16>
  %relu528 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv526 : tensor<1x1842x15x15xf16>)
    outs(%empty527 : tensor<1x1842x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1842x15x15xf16>

  // conv99: 3x3 s1 p1 1842->1842 15x15
  %pad529 = tensor.pad %relu528 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1842x15x15xf16> to tensor<1x1842x17x17xf16>
  %init530 = tensor.empty() : tensor<1x1842x15x15xf16>
  %fill531 = linalg.fill ins(%cst : f16) outs(%init530 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %conv532 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad529, %w99 : tensor<1x1842x17x17xf16>, tensor<1842x1842x3x3xf16>)
    outs(%fill531 : tensor<1x1842x15x15xf16>) -> tensor<1x1842x15x15xf16>
  %empty533 = tensor.empty() : tensor<1x1842x15x15xf16>
  %relu534 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv532 : tensor<1x1842x15x15xf16>)
    outs(%empty533 : tensor<1x1842x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1842x15x15xf16>

  // conv100: 1x1 s1 p0 1842->512 15x15
  %init535 = tensor.empty() : tensor<1x512x15x15xf16>
  %fill536 = linalg.fill ins(%cst : f16) outs(%init535 : tensor<1x512x15x15xf16>) -> tensor<1x512x15x15xf16>
  %conv537 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu534, %w100 : tensor<1x1842x15x15xf16>, tensor<512x1842x1x1xf16>)
    outs(%fill536 : tensor<1x512x15x15xf16>) -> tensor<1x512x15x15xf16>
  %empty538 = tensor.empty() : tensor<1x512x15x15xf16>
  %relu539 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv537 : tensor<1x512x15x15xf16>)
    outs(%empty538 : tensor<1x512x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x15x15xf16>

  // conv101: 1x1 s1 p0 512->3072 15x15
  %init540 = tensor.empty() : tensor<1x3072x15x15xf16>
  %fill541 = linalg.fill ins(%cst : f16) outs(%init540 : tensor<1x3072x15x15xf16>) -> tensor<1x3072x15x15xf16>
  %conv542 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu539, %w101 : tensor<1x512x15x15xf16>, tensor<3072x512x1x1xf16>)
    outs(%fill541 : tensor<1x3072x15x15xf16>) -> tensor<1x3072x15x15xf16>
  %empty543 = tensor.empty() : tensor<1x3072x15x15xf16>
  %relu544 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv542 : tensor<1x3072x15x15xf16>)
    outs(%empty543 : tensor<1x3072x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x3072x15x15xf16>

  // conv102: 3x3 s1 p1 3072->3072 15x15
  %pad545 = tensor.pad %relu544 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3072x15x15xf16> to tensor<1x3072x17x17xf16>
  %init546 = tensor.empty() : tensor<1x3072x15x15xf16>
  %fill547 = linalg.fill ins(%cst : f16) outs(%init546 : tensor<1x3072x15x15xf16>) -> tensor<1x3072x15x15xf16>
  %conv548 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad545, %w102 : tensor<1x3072x17x17xf16>, tensor<3072x3072x3x3xf16>)
    outs(%fill547 : tensor<1x3072x15x15xf16>) -> tensor<1x3072x15x15xf16>
  %empty549 = tensor.empty() : tensor<1x3072x15x15xf16>
  %relu550 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv548 : tensor<1x3072x15x15xf16>)
    outs(%empty549 : tensor<1x3072x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x3072x15x15xf16>

  // conv103: 1x1 s1 p0 3072->512 15x15
  %init551 = tensor.empty() : tensor<1x512x15x15xf16>
  %fill552 = linalg.fill ins(%cst : f16) outs(%init551 : tensor<1x512x15x15xf16>) -> tensor<1x512x15x15xf16>
  %conv553 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu550, %w103 : tensor<1x3072x15x15xf16>, tensor<512x3072x1x1xf16>)
    outs(%fill552 : tensor<1x512x15x15xf16>) -> tensor<1x512x15x15xf16>
  %empty554 = tensor.empty() : tensor<1x512x15x15xf16>
  %relu555 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv553 : tensor<1x512x15x15xf16>)
    outs(%empty554 : tensor<1x512x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x15x15xf16>

  // conv104: 1x1 s1 p0 512->2048 15x15
  %init556 = tensor.empty() : tensor<1x2048x15x15xf16>
  %fill557 = linalg.fill ins(%cst : f16) outs(%init556 : tensor<1x2048x15x15xf16>) -> tensor<1x2048x15x15xf16>
  %conv558 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu555, %w104 : tensor<1x512x15x15xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill557 : tensor<1x2048x15x15xf16>) -> tensor<1x2048x15x15xf16>
  %empty559 = tensor.empty() : tensor<1x2048x15x15xf16>
  %relu560 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv558 : tensor<1x2048x15x15xf16>)
    outs(%empty559 : tensor<1x2048x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x15x15xf16>

  // FC as 1x1 conv: 2048->1000
  %init561 = tensor.empty() : tensor<1x1000x15x15xf16>
  %fill562 = linalg.fill ins(%cst : f16) outs(%init561 : tensor<1x1000x15x15xf16>) -> tensor<1x1000x15x15xf16>
  %conv563 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu560, %w_fc : tensor<1x2048x15x15xf16>, tensor<1000x2048x1x1xf16>)
    outs(%fill562 : tensor<1x1000x15x15xf16>) -> tensor<1x1000x15x15xf16>
  return %conv563 : tensor<1x1000x15x15xf16>
}
