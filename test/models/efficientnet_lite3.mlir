func.func @efficientnet_lite3(
    %input: tensor<1x3x280x280xf16>,
    %w0: tensor<38x3x3x3xf16>,
    %w1: tensor<38x38x3x3xf16>,
    %w2: tensor<19x38x1x1xf16>,
    %w3: tensor<114x19x1x1xf16>,
    %w4: tensor<114x114x3x3xf16>,
    %w5: tensor<28x114x1x1xf16>,
    %w6: tensor<168x28x1x1xf16>,
    %w7: tensor<168x168x3x3xf16>,
    %w8: tensor<28x168x1x1xf16>,
    %w9: tensor<168x28x1x1xf16>,
    %w10: tensor<168x168x3x3xf16>,
    %w11: tensor<28x168x1x1xf16>,
    %w12: tensor<168x28x1x1xf16>,
    %w13: tensor<168x168x3x3xf16>,
    %w14: tensor<48x168x1x1xf16>,
    %w15: tensor<288x48x1x1xf16>,
    %w16: tensor<288x288x3x3xf16>,
    %w17: tensor<48x288x1x1xf16>,
    %w18: tensor<288x48x1x1xf16>,
    %w19: tensor<288x288x3x3xf16>,
    %w20: tensor<48x288x1x1xf16>,
    %w21: tensor<288x48x1x1xf16>,
    %w22: tensor<288x288x3x3xf16>,
    %w23: tensor<96x288x1x1xf16>,
    %w24: tensor<576x96x1x1xf16>,
    %w25: tensor<576x576x3x3xf16>,
    %w26: tensor<96x576x1x1xf16>,
    %w27: tensor<576x96x1x1xf16>,
    %w28: tensor<576x576x3x3xf16>,
    %w29: tensor<96x576x1x1xf16>,
    %w30: tensor<576x96x1x1xf16>,
    %w31: tensor<576x576x3x3xf16>,
    %w32: tensor<96x576x1x1xf16>,
    %w33: tensor<576x96x1x1xf16>,
    %w34: tensor<576x576x3x3xf16>,
    %w35: tensor<134x576x1x1xf16>,
    %w36: tensor<804x134x1x1xf16>,
    %w37: tensor<804x804x3x3xf16>,
    %w38: tensor<134x804x1x1xf16>,
    %w39: tensor<804x134x1x1xf16>,
    %w40: tensor<804x804x3x3xf16>,
    %w41: tensor<134x804x1x1xf16>,
    %w42: tensor<804x134x1x1xf16>,
    %w43: tensor<804x804x3x3xf16>,
    %w44: tensor<134x804x1x1xf16>,
    %w45: tensor<804x134x1x1xf16>,
    %w46: tensor<804x804x3x3xf16>,
    %w47: tensor<230x804x1x1xf16>,
    %w48: tensor<1380x230x1x1xf16>,
    %w49: tensor<1380x1380x3x3xf16>,
    %w50: tensor<230x1380x1x1xf16>,
    %w51: tensor<1380x230x1x1xf16>,
    %w52: tensor<1380x1380x3x3xf16>,
    %w53: tensor<230x1380x1x1xf16>,
    %w54: tensor<1380x230x1x1xf16>,
    %w55: tensor<1380x1380x3x3xf16>,
    %w56: tensor<230x1380x1x1xf16>,
    %w57: tensor<1380x230x1x1xf16>,
    %w58: tensor<1380x1380x3x3xf16>,
    %w59: tensor<230x1380x1x1xf16>,
    %w60: tensor<1380x230x1x1xf16>,
    %w61: tensor<1380x1380x3x3xf16>,
    %w62: tensor<230x1380x1x1xf16>,
    %w63: tensor<1380x230x1x1xf16>,
    %w64: tensor<1380x1380x3x3xf16>,
    %w65: tensor<384x1380x1x1xf16>,
    %w66: tensor<1536x384x1x1xf16>,
    %w_fc: tensor<1000x1536x1x1xf16>) -> tensor<1x1000x9x9xf16> {
  %cst = arith.constant 0.0 : f16

  // conv0: 3x3 s2 p1 3->38 280x280
  %pad0 = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x280x280xf16> to tensor<1x3x282x282xf16>
  %init1 = tensor.empty() : tensor<1x38x140x140xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x38x140x140xf16>) -> tensor<1x38x140x140xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad0, %w0 : tensor<1x3x282x282xf16>, tensor<38x3x3x3xf16>)
    outs(%fill2 : tensor<1x38x140x140xf16>) -> tensor<1x38x140x140xf16>
  %empty4 = tensor.empty() : tensor<1x38x140x140xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x38x140x140xf16>)
    outs(%empty4 : tensor<1x38x140x140xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x38x140x140xf16>

  // conv1: 3x3 s1 p1 38->38 140x140
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x38x140x140xf16> to tensor<1x38x142x142xf16>
  %init7 = tensor.empty() : tensor<1x38x140x140xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x38x140x140xf16>) -> tensor<1x38x140x140xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad6, %w1 : tensor<1x38x142x142xf16>, tensor<38x38x3x3xf16>)
    outs(%fill8 : tensor<1x38x140x140xf16>) -> tensor<1x38x140x140xf16>
  %empty10 = tensor.empty() : tensor<1x38x140x140xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x38x140x140xf16>)
    outs(%empty10 : tensor<1x38x140x140xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x38x140x140xf16>

  // conv2: 1x1 s1 p0 38->19 140x140
  %init12 = tensor.empty() : tensor<1x19x140x140xf16>
  %fill13 = linalg.fill ins(%cst : f16) outs(%init12 : tensor<1x19x140x140xf16>) -> tensor<1x19x140x140xf16>
  %conv14 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu11, %w2 : tensor<1x38x140x140xf16>, tensor<19x38x1x1xf16>)
    outs(%fill13 : tensor<1x19x140x140xf16>) -> tensor<1x19x140x140xf16>
  %empty15 = tensor.empty() : tensor<1x19x140x140xf16>
  %relu16 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv14 : tensor<1x19x140x140xf16>)
    outs(%empty15 : tensor<1x19x140x140xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x19x140x140xf16>

  // conv3: 1x1 s1 p0 19->114 140x140
  %init17 = tensor.empty() : tensor<1x114x140x140xf16>
  %fill18 = linalg.fill ins(%cst : f16) outs(%init17 : tensor<1x114x140x140xf16>) -> tensor<1x114x140x140xf16>
  %conv19 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu16, %w3 : tensor<1x19x140x140xf16>, tensor<114x19x1x1xf16>)
    outs(%fill18 : tensor<1x114x140x140xf16>) -> tensor<1x114x140x140xf16>
  %empty20 = tensor.empty() : tensor<1x114x140x140xf16>
  %relu21 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv19 : tensor<1x114x140x140xf16>)
    outs(%empty20 : tensor<1x114x140x140xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x114x140x140xf16>

  // conv4: 3x3 s2 p1 114->114 140x140
  %pad22 = tensor.pad %relu21 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x114x140x140xf16> to tensor<1x114x142x142xf16>
  %init23 = tensor.empty() : tensor<1x114x70x70xf16>
  %fill24 = linalg.fill ins(%cst : f16) outs(%init23 : tensor<1x114x70x70xf16>) -> tensor<1x114x70x70xf16>
  %conv25 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad22, %w4 : tensor<1x114x142x142xf16>, tensor<114x114x3x3xf16>)
    outs(%fill24 : tensor<1x114x70x70xf16>) -> tensor<1x114x70x70xf16>
  %empty26 = tensor.empty() : tensor<1x114x70x70xf16>
  %relu27 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv25 : tensor<1x114x70x70xf16>)
    outs(%empty26 : tensor<1x114x70x70xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x114x70x70xf16>

  // conv5: 1x1 s1 p0 114->28 70x70
  %init28 = tensor.empty() : tensor<1x28x70x70xf16>
  %fill29 = linalg.fill ins(%cst : f16) outs(%init28 : tensor<1x28x70x70xf16>) -> tensor<1x28x70x70xf16>
  %conv30 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu27, %w5 : tensor<1x114x70x70xf16>, tensor<28x114x1x1xf16>)
    outs(%fill29 : tensor<1x28x70x70xf16>) -> tensor<1x28x70x70xf16>
  %empty31 = tensor.empty() : tensor<1x28x70x70xf16>
  %relu32 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv30 : tensor<1x28x70x70xf16>)
    outs(%empty31 : tensor<1x28x70x70xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x28x70x70xf16>

  // conv6: 1x1 s1 p0 28->168 70x70
  %init33 = tensor.empty() : tensor<1x168x70x70xf16>
  %fill34 = linalg.fill ins(%cst : f16) outs(%init33 : tensor<1x168x70x70xf16>) -> tensor<1x168x70x70xf16>
  %conv35 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu32, %w6 : tensor<1x28x70x70xf16>, tensor<168x28x1x1xf16>)
    outs(%fill34 : tensor<1x168x70x70xf16>) -> tensor<1x168x70x70xf16>
  %empty36 = tensor.empty() : tensor<1x168x70x70xf16>
  %relu37 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv35 : tensor<1x168x70x70xf16>)
    outs(%empty36 : tensor<1x168x70x70xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x168x70x70xf16>

  // conv7: 3x3 s1 p1 168->168 70x70
  %pad38 = tensor.pad %relu37 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x168x70x70xf16> to tensor<1x168x72x72xf16>
  %init39 = tensor.empty() : tensor<1x168x70x70xf16>
  %fill40 = linalg.fill ins(%cst : f16) outs(%init39 : tensor<1x168x70x70xf16>) -> tensor<1x168x70x70xf16>
  %conv41 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad38, %w7 : tensor<1x168x72x72xf16>, tensor<168x168x3x3xf16>)
    outs(%fill40 : tensor<1x168x70x70xf16>) -> tensor<1x168x70x70xf16>
  %empty42 = tensor.empty() : tensor<1x168x70x70xf16>
  %relu43 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv41 : tensor<1x168x70x70xf16>)
    outs(%empty42 : tensor<1x168x70x70xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x168x70x70xf16>

  // conv8: 1x1 s1 p0 168->28 70x70
  %init44 = tensor.empty() : tensor<1x28x70x70xf16>
  %fill45 = linalg.fill ins(%cst : f16) outs(%init44 : tensor<1x28x70x70xf16>) -> tensor<1x28x70x70xf16>
  %conv46 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu43, %w8 : tensor<1x168x70x70xf16>, tensor<28x168x1x1xf16>)
    outs(%fill45 : tensor<1x28x70x70xf16>) -> tensor<1x28x70x70xf16>
  %empty47 = tensor.empty() : tensor<1x28x70x70xf16>
  %relu48 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv46 : tensor<1x28x70x70xf16>)
    outs(%empty47 : tensor<1x28x70x70xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x28x70x70xf16>

  // conv9: 1x1 s1 p0 28->168 70x70
  %init49 = tensor.empty() : tensor<1x168x70x70xf16>
  %fill50 = linalg.fill ins(%cst : f16) outs(%init49 : tensor<1x168x70x70xf16>) -> tensor<1x168x70x70xf16>
  %conv51 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu48, %w9 : tensor<1x28x70x70xf16>, tensor<168x28x1x1xf16>)
    outs(%fill50 : tensor<1x168x70x70xf16>) -> tensor<1x168x70x70xf16>
  %empty52 = tensor.empty() : tensor<1x168x70x70xf16>
  %relu53 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv51 : tensor<1x168x70x70xf16>)
    outs(%empty52 : tensor<1x168x70x70xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x168x70x70xf16>

  // conv10: 3x3 s1 p1 168->168 70x70
  %pad54 = tensor.pad %relu53 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x168x70x70xf16> to tensor<1x168x72x72xf16>
  %init55 = tensor.empty() : tensor<1x168x70x70xf16>
  %fill56 = linalg.fill ins(%cst : f16) outs(%init55 : tensor<1x168x70x70xf16>) -> tensor<1x168x70x70xf16>
  %conv57 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad54, %w10 : tensor<1x168x72x72xf16>, tensor<168x168x3x3xf16>)
    outs(%fill56 : tensor<1x168x70x70xf16>) -> tensor<1x168x70x70xf16>
  %empty58 = tensor.empty() : tensor<1x168x70x70xf16>
  %relu59 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv57 : tensor<1x168x70x70xf16>)
    outs(%empty58 : tensor<1x168x70x70xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x168x70x70xf16>

  // conv11: 1x1 s1 p0 168->28 70x70
  %init60 = tensor.empty() : tensor<1x28x70x70xf16>
  %fill61 = linalg.fill ins(%cst : f16) outs(%init60 : tensor<1x28x70x70xf16>) -> tensor<1x28x70x70xf16>
  %conv62 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu59, %w11 : tensor<1x168x70x70xf16>, tensor<28x168x1x1xf16>)
    outs(%fill61 : tensor<1x28x70x70xf16>) -> tensor<1x28x70x70xf16>
  %empty63 = tensor.empty() : tensor<1x28x70x70xf16>
  %relu64 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv62 : tensor<1x28x70x70xf16>)
    outs(%empty63 : tensor<1x28x70x70xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x28x70x70xf16>

  // conv12: 1x1 s1 p0 28->168 70x70
  %init65 = tensor.empty() : tensor<1x168x70x70xf16>
  %fill66 = linalg.fill ins(%cst : f16) outs(%init65 : tensor<1x168x70x70xf16>) -> tensor<1x168x70x70xf16>
  %conv67 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu64, %w12 : tensor<1x28x70x70xf16>, tensor<168x28x1x1xf16>)
    outs(%fill66 : tensor<1x168x70x70xf16>) -> tensor<1x168x70x70xf16>
  %empty68 = tensor.empty() : tensor<1x168x70x70xf16>
  %relu69 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv67 : tensor<1x168x70x70xf16>)
    outs(%empty68 : tensor<1x168x70x70xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x168x70x70xf16>

  // conv13: 3x3 s2 p1 168->168 70x70
  %pad70 = tensor.pad %relu69 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x168x70x70xf16> to tensor<1x168x72x72xf16>
  %init71 = tensor.empty() : tensor<1x168x35x35xf16>
  %fill72 = linalg.fill ins(%cst : f16) outs(%init71 : tensor<1x168x35x35xf16>) -> tensor<1x168x35x35xf16>
  %conv73 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad70, %w13 : tensor<1x168x72x72xf16>, tensor<168x168x3x3xf16>)
    outs(%fill72 : tensor<1x168x35x35xf16>) -> tensor<1x168x35x35xf16>
  %empty74 = tensor.empty() : tensor<1x168x35x35xf16>
  %relu75 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv73 : tensor<1x168x35x35xf16>)
    outs(%empty74 : tensor<1x168x35x35xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x168x35x35xf16>

  // conv14: 1x1 s1 p0 168->48 35x35
  %init76 = tensor.empty() : tensor<1x48x35x35xf16>
  %fill77 = linalg.fill ins(%cst : f16) outs(%init76 : tensor<1x48x35x35xf16>) -> tensor<1x48x35x35xf16>
  %conv78 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu75, %w14 : tensor<1x168x35x35xf16>, tensor<48x168x1x1xf16>)
    outs(%fill77 : tensor<1x48x35x35xf16>) -> tensor<1x48x35x35xf16>
  %empty79 = tensor.empty() : tensor<1x48x35x35xf16>
  %relu80 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv78 : tensor<1x48x35x35xf16>)
    outs(%empty79 : tensor<1x48x35x35xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x48x35x35xf16>

  // conv15: 1x1 s1 p0 48->288 35x35
  %init81 = tensor.empty() : tensor<1x288x35x35xf16>
  %fill82 = linalg.fill ins(%cst : f16) outs(%init81 : tensor<1x288x35x35xf16>) -> tensor<1x288x35x35xf16>
  %conv83 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu80, %w15 : tensor<1x48x35x35xf16>, tensor<288x48x1x1xf16>)
    outs(%fill82 : tensor<1x288x35x35xf16>) -> tensor<1x288x35x35xf16>
  %empty84 = tensor.empty() : tensor<1x288x35x35xf16>
  %relu85 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv83 : tensor<1x288x35x35xf16>)
    outs(%empty84 : tensor<1x288x35x35xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x288x35x35xf16>

  // conv16: 3x3 s1 p1 288->288 35x35
  %pad86 = tensor.pad %relu85 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x288x35x35xf16> to tensor<1x288x37x37xf16>
  %init87 = tensor.empty() : tensor<1x288x35x35xf16>
  %fill88 = linalg.fill ins(%cst : f16) outs(%init87 : tensor<1x288x35x35xf16>) -> tensor<1x288x35x35xf16>
  %conv89 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad86, %w16 : tensor<1x288x37x37xf16>, tensor<288x288x3x3xf16>)
    outs(%fill88 : tensor<1x288x35x35xf16>) -> tensor<1x288x35x35xf16>
  %empty90 = tensor.empty() : tensor<1x288x35x35xf16>
  %relu91 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv89 : tensor<1x288x35x35xf16>)
    outs(%empty90 : tensor<1x288x35x35xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x288x35x35xf16>

  // conv17: 1x1 s1 p0 288->48 35x35
  %init92 = tensor.empty() : tensor<1x48x35x35xf16>
  %fill93 = linalg.fill ins(%cst : f16) outs(%init92 : tensor<1x48x35x35xf16>) -> tensor<1x48x35x35xf16>
  %conv94 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu91, %w17 : tensor<1x288x35x35xf16>, tensor<48x288x1x1xf16>)
    outs(%fill93 : tensor<1x48x35x35xf16>) -> tensor<1x48x35x35xf16>
  %empty95 = tensor.empty() : tensor<1x48x35x35xf16>
  %relu96 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv94 : tensor<1x48x35x35xf16>)
    outs(%empty95 : tensor<1x48x35x35xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x48x35x35xf16>

  // conv18: 1x1 s1 p0 48->288 35x35
  %init97 = tensor.empty() : tensor<1x288x35x35xf16>
  %fill98 = linalg.fill ins(%cst : f16) outs(%init97 : tensor<1x288x35x35xf16>) -> tensor<1x288x35x35xf16>
  %conv99 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu96, %w18 : tensor<1x48x35x35xf16>, tensor<288x48x1x1xf16>)
    outs(%fill98 : tensor<1x288x35x35xf16>) -> tensor<1x288x35x35xf16>
  %empty100 = tensor.empty() : tensor<1x288x35x35xf16>
  %relu101 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv99 : tensor<1x288x35x35xf16>)
    outs(%empty100 : tensor<1x288x35x35xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x288x35x35xf16>

  // conv19: 3x3 s1 p1 288->288 35x35
  %pad102 = tensor.pad %relu101 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x288x35x35xf16> to tensor<1x288x37x37xf16>
  %init103 = tensor.empty() : tensor<1x288x35x35xf16>
  %fill104 = linalg.fill ins(%cst : f16) outs(%init103 : tensor<1x288x35x35xf16>) -> tensor<1x288x35x35xf16>
  %conv105 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad102, %w19 : tensor<1x288x37x37xf16>, tensor<288x288x3x3xf16>)
    outs(%fill104 : tensor<1x288x35x35xf16>) -> tensor<1x288x35x35xf16>
  %empty106 = tensor.empty() : tensor<1x288x35x35xf16>
  %relu107 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv105 : tensor<1x288x35x35xf16>)
    outs(%empty106 : tensor<1x288x35x35xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x288x35x35xf16>

  // conv20: 1x1 s1 p0 288->48 35x35
  %init108 = tensor.empty() : tensor<1x48x35x35xf16>
  %fill109 = linalg.fill ins(%cst : f16) outs(%init108 : tensor<1x48x35x35xf16>) -> tensor<1x48x35x35xf16>
  %conv110 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu107, %w20 : tensor<1x288x35x35xf16>, tensor<48x288x1x1xf16>)
    outs(%fill109 : tensor<1x48x35x35xf16>) -> tensor<1x48x35x35xf16>
  %empty111 = tensor.empty() : tensor<1x48x35x35xf16>
  %relu112 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv110 : tensor<1x48x35x35xf16>)
    outs(%empty111 : tensor<1x48x35x35xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x48x35x35xf16>

  // conv21: 1x1 s1 p0 48->288 35x35
  %init113 = tensor.empty() : tensor<1x288x35x35xf16>
  %fill114 = linalg.fill ins(%cst : f16) outs(%init113 : tensor<1x288x35x35xf16>) -> tensor<1x288x35x35xf16>
  %conv115 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu112, %w21 : tensor<1x48x35x35xf16>, tensor<288x48x1x1xf16>)
    outs(%fill114 : tensor<1x288x35x35xf16>) -> tensor<1x288x35x35xf16>
  %empty116 = tensor.empty() : tensor<1x288x35x35xf16>
  %relu117 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv115 : tensor<1x288x35x35xf16>)
    outs(%empty116 : tensor<1x288x35x35xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x288x35x35xf16>

  // conv22: 3x3 s2 p1 288->288 35x35
  %pad118 = tensor.pad %relu117 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x288x35x35xf16> to tensor<1x288x37x37xf16>
  %init119 = tensor.empty() : tensor<1x288x18x18xf16>
  %fill120 = linalg.fill ins(%cst : f16) outs(%init119 : tensor<1x288x18x18xf16>) -> tensor<1x288x18x18xf16>
  %conv121 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad118, %w22 : tensor<1x288x37x37xf16>, tensor<288x288x3x3xf16>)
    outs(%fill120 : tensor<1x288x18x18xf16>) -> tensor<1x288x18x18xf16>
  %empty122 = tensor.empty() : tensor<1x288x18x18xf16>
  %relu123 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv121 : tensor<1x288x18x18xf16>)
    outs(%empty122 : tensor<1x288x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x288x18x18xf16>

  // conv23: 1x1 s1 p0 288->96 18x18
  %init124 = tensor.empty() : tensor<1x96x18x18xf16>
  %fill125 = linalg.fill ins(%cst : f16) outs(%init124 : tensor<1x96x18x18xf16>) -> tensor<1x96x18x18xf16>
  %conv126 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu123, %w23 : tensor<1x288x18x18xf16>, tensor<96x288x1x1xf16>)
    outs(%fill125 : tensor<1x96x18x18xf16>) -> tensor<1x96x18x18xf16>
  %empty127 = tensor.empty() : tensor<1x96x18x18xf16>
  %relu128 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv126 : tensor<1x96x18x18xf16>)
    outs(%empty127 : tensor<1x96x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x96x18x18xf16>

  // conv24: 1x1 s1 p0 96->576 18x18
  %init129 = tensor.empty() : tensor<1x576x18x18xf16>
  %fill130 = linalg.fill ins(%cst : f16) outs(%init129 : tensor<1x576x18x18xf16>) -> tensor<1x576x18x18xf16>
  %conv131 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu128, %w24 : tensor<1x96x18x18xf16>, tensor<576x96x1x1xf16>)
    outs(%fill130 : tensor<1x576x18x18xf16>) -> tensor<1x576x18x18xf16>
  %empty132 = tensor.empty() : tensor<1x576x18x18xf16>
  %relu133 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv131 : tensor<1x576x18x18xf16>)
    outs(%empty132 : tensor<1x576x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x18x18xf16>

  // conv25: 3x3 s1 p1 576->576 18x18
  %pad134 = tensor.pad %relu133 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x576x18x18xf16> to tensor<1x576x20x20xf16>
  %init135 = tensor.empty() : tensor<1x576x18x18xf16>
  %fill136 = linalg.fill ins(%cst : f16) outs(%init135 : tensor<1x576x18x18xf16>) -> tensor<1x576x18x18xf16>
  %conv137 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad134, %w25 : tensor<1x576x20x20xf16>, tensor<576x576x3x3xf16>)
    outs(%fill136 : tensor<1x576x18x18xf16>) -> tensor<1x576x18x18xf16>
  %empty138 = tensor.empty() : tensor<1x576x18x18xf16>
  %relu139 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv137 : tensor<1x576x18x18xf16>)
    outs(%empty138 : tensor<1x576x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x18x18xf16>

  // conv26: 1x1 s1 p0 576->96 18x18
  %init140 = tensor.empty() : tensor<1x96x18x18xf16>
  %fill141 = linalg.fill ins(%cst : f16) outs(%init140 : tensor<1x96x18x18xf16>) -> tensor<1x96x18x18xf16>
  %conv142 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu139, %w26 : tensor<1x576x18x18xf16>, tensor<96x576x1x1xf16>)
    outs(%fill141 : tensor<1x96x18x18xf16>) -> tensor<1x96x18x18xf16>
  %empty143 = tensor.empty() : tensor<1x96x18x18xf16>
  %relu144 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv142 : tensor<1x96x18x18xf16>)
    outs(%empty143 : tensor<1x96x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x96x18x18xf16>

  // conv27: 1x1 s1 p0 96->576 18x18
  %init145 = tensor.empty() : tensor<1x576x18x18xf16>
  %fill146 = linalg.fill ins(%cst : f16) outs(%init145 : tensor<1x576x18x18xf16>) -> tensor<1x576x18x18xf16>
  %conv147 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu144, %w27 : tensor<1x96x18x18xf16>, tensor<576x96x1x1xf16>)
    outs(%fill146 : tensor<1x576x18x18xf16>) -> tensor<1x576x18x18xf16>
  %empty148 = tensor.empty() : tensor<1x576x18x18xf16>
  %relu149 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv147 : tensor<1x576x18x18xf16>)
    outs(%empty148 : tensor<1x576x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x18x18xf16>

  // conv28: 3x3 s1 p1 576->576 18x18
  %pad150 = tensor.pad %relu149 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x576x18x18xf16> to tensor<1x576x20x20xf16>
  %init151 = tensor.empty() : tensor<1x576x18x18xf16>
  %fill152 = linalg.fill ins(%cst : f16) outs(%init151 : tensor<1x576x18x18xf16>) -> tensor<1x576x18x18xf16>
  %conv153 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad150, %w28 : tensor<1x576x20x20xf16>, tensor<576x576x3x3xf16>)
    outs(%fill152 : tensor<1x576x18x18xf16>) -> tensor<1x576x18x18xf16>
  %empty154 = tensor.empty() : tensor<1x576x18x18xf16>
  %relu155 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv153 : tensor<1x576x18x18xf16>)
    outs(%empty154 : tensor<1x576x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x18x18xf16>

  // conv29: 1x1 s1 p0 576->96 18x18
  %init156 = tensor.empty() : tensor<1x96x18x18xf16>
  %fill157 = linalg.fill ins(%cst : f16) outs(%init156 : tensor<1x96x18x18xf16>) -> tensor<1x96x18x18xf16>
  %conv158 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu155, %w29 : tensor<1x576x18x18xf16>, tensor<96x576x1x1xf16>)
    outs(%fill157 : tensor<1x96x18x18xf16>) -> tensor<1x96x18x18xf16>
  %empty159 = tensor.empty() : tensor<1x96x18x18xf16>
  %relu160 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv158 : tensor<1x96x18x18xf16>)
    outs(%empty159 : tensor<1x96x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x96x18x18xf16>

  // conv30: 1x1 s1 p0 96->576 18x18
  %init161 = tensor.empty() : tensor<1x576x18x18xf16>
  %fill162 = linalg.fill ins(%cst : f16) outs(%init161 : tensor<1x576x18x18xf16>) -> tensor<1x576x18x18xf16>
  %conv163 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu160, %w30 : tensor<1x96x18x18xf16>, tensor<576x96x1x1xf16>)
    outs(%fill162 : tensor<1x576x18x18xf16>) -> tensor<1x576x18x18xf16>
  %empty164 = tensor.empty() : tensor<1x576x18x18xf16>
  %relu165 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv163 : tensor<1x576x18x18xf16>)
    outs(%empty164 : tensor<1x576x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x18x18xf16>

  // conv31: 3x3 s1 p1 576->576 18x18
  %pad166 = tensor.pad %relu165 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x576x18x18xf16> to tensor<1x576x20x20xf16>
  %init167 = tensor.empty() : tensor<1x576x18x18xf16>
  %fill168 = linalg.fill ins(%cst : f16) outs(%init167 : tensor<1x576x18x18xf16>) -> tensor<1x576x18x18xf16>
  %conv169 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad166, %w31 : tensor<1x576x20x20xf16>, tensor<576x576x3x3xf16>)
    outs(%fill168 : tensor<1x576x18x18xf16>) -> tensor<1x576x18x18xf16>
  %empty170 = tensor.empty() : tensor<1x576x18x18xf16>
  %relu171 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv169 : tensor<1x576x18x18xf16>)
    outs(%empty170 : tensor<1x576x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x18x18xf16>

  // conv32: 1x1 s1 p0 576->96 18x18
  %init172 = tensor.empty() : tensor<1x96x18x18xf16>
  %fill173 = linalg.fill ins(%cst : f16) outs(%init172 : tensor<1x96x18x18xf16>) -> tensor<1x96x18x18xf16>
  %conv174 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu171, %w32 : tensor<1x576x18x18xf16>, tensor<96x576x1x1xf16>)
    outs(%fill173 : tensor<1x96x18x18xf16>) -> tensor<1x96x18x18xf16>
  %empty175 = tensor.empty() : tensor<1x96x18x18xf16>
  %relu176 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv174 : tensor<1x96x18x18xf16>)
    outs(%empty175 : tensor<1x96x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x96x18x18xf16>

  // conv33: 1x1 s1 p0 96->576 18x18
  %init177 = tensor.empty() : tensor<1x576x18x18xf16>
  %fill178 = linalg.fill ins(%cst : f16) outs(%init177 : tensor<1x576x18x18xf16>) -> tensor<1x576x18x18xf16>
  %conv179 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu176, %w33 : tensor<1x96x18x18xf16>, tensor<576x96x1x1xf16>)
    outs(%fill178 : tensor<1x576x18x18xf16>) -> tensor<1x576x18x18xf16>
  %empty180 = tensor.empty() : tensor<1x576x18x18xf16>
  %relu181 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv179 : tensor<1x576x18x18xf16>)
    outs(%empty180 : tensor<1x576x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x18x18xf16>

  // conv34: 3x3 s1 p1 576->576 18x18
  %pad182 = tensor.pad %relu181 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x576x18x18xf16> to tensor<1x576x20x20xf16>
  %init183 = tensor.empty() : tensor<1x576x18x18xf16>
  %fill184 = linalg.fill ins(%cst : f16) outs(%init183 : tensor<1x576x18x18xf16>) -> tensor<1x576x18x18xf16>
  %conv185 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad182, %w34 : tensor<1x576x20x20xf16>, tensor<576x576x3x3xf16>)
    outs(%fill184 : tensor<1x576x18x18xf16>) -> tensor<1x576x18x18xf16>
  %empty186 = tensor.empty() : tensor<1x576x18x18xf16>
  %relu187 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv185 : tensor<1x576x18x18xf16>)
    outs(%empty186 : tensor<1x576x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x576x18x18xf16>

  // conv35: 1x1 s1 p0 576->134 18x18
  %init188 = tensor.empty() : tensor<1x134x18x18xf16>
  %fill189 = linalg.fill ins(%cst : f16) outs(%init188 : tensor<1x134x18x18xf16>) -> tensor<1x134x18x18xf16>
  %conv190 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu187, %w35 : tensor<1x576x18x18xf16>, tensor<134x576x1x1xf16>)
    outs(%fill189 : tensor<1x134x18x18xf16>) -> tensor<1x134x18x18xf16>
  %empty191 = tensor.empty() : tensor<1x134x18x18xf16>
  %relu192 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv190 : tensor<1x134x18x18xf16>)
    outs(%empty191 : tensor<1x134x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x134x18x18xf16>

  // conv36: 1x1 s1 p0 134->804 18x18
  %init193 = tensor.empty() : tensor<1x804x18x18xf16>
  %fill194 = linalg.fill ins(%cst : f16) outs(%init193 : tensor<1x804x18x18xf16>) -> tensor<1x804x18x18xf16>
  %conv195 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu192, %w36 : tensor<1x134x18x18xf16>, tensor<804x134x1x1xf16>)
    outs(%fill194 : tensor<1x804x18x18xf16>) -> tensor<1x804x18x18xf16>
  %empty196 = tensor.empty() : tensor<1x804x18x18xf16>
  %relu197 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv195 : tensor<1x804x18x18xf16>)
    outs(%empty196 : tensor<1x804x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x804x18x18xf16>

  // conv37: 3x3 s1 p1 804->804 18x18
  %pad198 = tensor.pad %relu197 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x804x18x18xf16> to tensor<1x804x20x20xf16>
  %init199 = tensor.empty() : tensor<1x804x18x18xf16>
  %fill200 = linalg.fill ins(%cst : f16) outs(%init199 : tensor<1x804x18x18xf16>) -> tensor<1x804x18x18xf16>
  %conv201 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad198, %w37 : tensor<1x804x20x20xf16>, tensor<804x804x3x3xf16>)
    outs(%fill200 : tensor<1x804x18x18xf16>) -> tensor<1x804x18x18xf16>
  %empty202 = tensor.empty() : tensor<1x804x18x18xf16>
  %relu203 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv201 : tensor<1x804x18x18xf16>)
    outs(%empty202 : tensor<1x804x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x804x18x18xf16>

  // conv38: 1x1 s1 p0 804->134 18x18
  %init204 = tensor.empty() : tensor<1x134x18x18xf16>
  %fill205 = linalg.fill ins(%cst : f16) outs(%init204 : tensor<1x134x18x18xf16>) -> tensor<1x134x18x18xf16>
  %conv206 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu203, %w38 : tensor<1x804x18x18xf16>, tensor<134x804x1x1xf16>)
    outs(%fill205 : tensor<1x134x18x18xf16>) -> tensor<1x134x18x18xf16>
  %empty207 = tensor.empty() : tensor<1x134x18x18xf16>
  %relu208 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv206 : tensor<1x134x18x18xf16>)
    outs(%empty207 : tensor<1x134x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x134x18x18xf16>

  // conv39: 1x1 s1 p0 134->804 18x18
  %init209 = tensor.empty() : tensor<1x804x18x18xf16>
  %fill210 = linalg.fill ins(%cst : f16) outs(%init209 : tensor<1x804x18x18xf16>) -> tensor<1x804x18x18xf16>
  %conv211 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu208, %w39 : tensor<1x134x18x18xf16>, tensor<804x134x1x1xf16>)
    outs(%fill210 : tensor<1x804x18x18xf16>) -> tensor<1x804x18x18xf16>
  %empty212 = tensor.empty() : tensor<1x804x18x18xf16>
  %relu213 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv211 : tensor<1x804x18x18xf16>)
    outs(%empty212 : tensor<1x804x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x804x18x18xf16>

  // conv40: 3x3 s1 p1 804->804 18x18
  %pad214 = tensor.pad %relu213 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x804x18x18xf16> to tensor<1x804x20x20xf16>
  %init215 = tensor.empty() : tensor<1x804x18x18xf16>
  %fill216 = linalg.fill ins(%cst : f16) outs(%init215 : tensor<1x804x18x18xf16>) -> tensor<1x804x18x18xf16>
  %conv217 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad214, %w40 : tensor<1x804x20x20xf16>, tensor<804x804x3x3xf16>)
    outs(%fill216 : tensor<1x804x18x18xf16>) -> tensor<1x804x18x18xf16>
  %empty218 = tensor.empty() : tensor<1x804x18x18xf16>
  %relu219 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv217 : tensor<1x804x18x18xf16>)
    outs(%empty218 : tensor<1x804x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x804x18x18xf16>

  // conv41: 1x1 s1 p0 804->134 18x18
  %init220 = tensor.empty() : tensor<1x134x18x18xf16>
  %fill221 = linalg.fill ins(%cst : f16) outs(%init220 : tensor<1x134x18x18xf16>) -> tensor<1x134x18x18xf16>
  %conv222 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu219, %w41 : tensor<1x804x18x18xf16>, tensor<134x804x1x1xf16>)
    outs(%fill221 : tensor<1x134x18x18xf16>) -> tensor<1x134x18x18xf16>
  %empty223 = tensor.empty() : tensor<1x134x18x18xf16>
  %relu224 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv222 : tensor<1x134x18x18xf16>)
    outs(%empty223 : tensor<1x134x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x134x18x18xf16>

  // conv42: 1x1 s1 p0 134->804 18x18
  %init225 = tensor.empty() : tensor<1x804x18x18xf16>
  %fill226 = linalg.fill ins(%cst : f16) outs(%init225 : tensor<1x804x18x18xf16>) -> tensor<1x804x18x18xf16>
  %conv227 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu224, %w42 : tensor<1x134x18x18xf16>, tensor<804x134x1x1xf16>)
    outs(%fill226 : tensor<1x804x18x18xf16>) -> tensor<1x804x18x18xf16>
  %empty228 = tensor.empty() : tensor<1x804x18x18xf16>
  %relu229 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv227 : tensor<1x804x18x18xf16>)
    outs(%empty228 : tensor<1x804x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x804x18x18xf16>

  // conv43: 3x3 s1 p1 804->804 18x18
  %pad230 = tensor.pad %relu229 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x804x18x18xf16> to tensor<1x804x20x20xf16>
  %init231 = tensor.empty() : tensor<1x804x18x18xf16>
  %fill232 = linalg.fill ins(%cst : f16) outs(%init231 : tensor<1x804x18x18xf16>) -> tensor<1x804x18x18xf16>
  %conv233 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad230, %w43 : tensor<1x804x20x20xf16>, tensor<804x804x3x3xf16>)
    outs(%fill232 : tensor<1x804x18x18xf16>) -> tensor<1x804x18x18xf16>
  %empty234 = tensor.empty() : tensor<1x804x18x18xf16>
  %relu235 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv233 : tensor<1x804x18x18xf16>)
    outs(%empty234 : tensor<1x804x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x804x18x18xf16>

  // conv44: 1x1 s1 p0 804->134 18x18
  %init236 = tensor.empty() : tensor<1x134x18x18xf16>
  %fill237 = linalg.fill ins(%cst : f16) outs(%init236 : tensor<1x134x18x18xf16>) -> tensor<1x134x18x18xf16>
  %conv238 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu235, %w44 : tensor<1x804x18x18xf16>, tensor<134x804x1x1xf16>)
    outs(%fill237 : tensor<1x134x18x18xf16>) -> tensor<1x134x18x18xf16>
  %empty239 = tensor.empty() : tensor<1x134x18x18xf16>
  %relu240 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv238 : tensor<1x134x18x18xf16>)
    outs(%empty239 : tensor<1x134x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x134x18x18xf16>

  // conv45: 1x1 s1 p0 134->804 18x18
  %init241 = tensor.empty() : tensor<1x804x18x18xf16>
  %fill242 = linalg.fill ins(%cst : f16) outs(%init241 : tensor<1x804x18x18xf16>) -> tensor<1x804x18x18xf16>
  %conv243 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu240, %w45 : tensor<1x134x18x18xf16>, tensor<804x134x1x1xf16>)
    outs(%fill242 : tensor<1x804x18x18xf16>) -> tensor<1x804x18x18xf16>
  %empty244 = tensor.empty() : tensor<1x804x18x18xf16>
  %relu245 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv243 : tensor<1x804x18x18xf16>)
    outs(%empty244 : tensor<1x804x18x18xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x804x18x18xf16>

  // conv46: 3x3 s2 p1 804->804 18x18
  %pad246 = tensor.pad %relu245 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x804x18x18xf16> to tensor<1x804x20x20xf16>
  %init247 = tensor.empty() : tensor<1x804x9x9xf16>
  %fill248 = linalg.fill ins(%cst : f16) outs(%init247 : tensor<1x804x9x9xf16>) -> tensor<1x804x9x9xf16>
  %conv249 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad246, %w46 : tensor<1x804x20x20xf16>, tensor<804x804x3x3xf16>)
    outs(%fill248 : tensor<1x804x9x9xf16>) -> tensor<1x804x9x9xf16>
  %empty250 = tensor.empty() : tensor<1x804x9x9xf16>
  %relu251 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv249 : tensor<1x804x9x9xf16>)
    outs(%empty250 : tensor<1x804x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x804x9x9xf16>

  // conv47: 1x1 s1 p0 804->230 9x9
  %init252 = tensor.empty() : tensor<1x230x9x9xf16>
  %fill253 = linalg.fill ins(%cst : f16) outs(%init252 : tensor<1x230x9x9xf16>) -> tensor<1x230x9x9xf16>
  %conv254 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu251, %w47 : tensor<1x804x9x9xf16>, tensor<230x804x1x1xf16>)
    outs(%fill253 : tensor<1x230x9x9xf16>) -> tensor<1x230x9x9xf16>
  %empty255 = tensor.empty() : tensor<1x230x9x9xf16>
  %relu256 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv254 : tensor<1x230x9x9xf16>)
    outs(%empty255 : tensor<1x230x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x230x9x9xf16>

  // conv48: 1x1 s1 p0 230->1380 9x9
  %init257 = tensor.empty() : tensor<1x1380x9x9xf16>
  %fill258 = linalg.fill ins(%cst : f16) outs(%init257 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %conv259 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu256, %w48 : tensor<1x230x9x9xf16>, tensor<1380x230x1x1xf16>)
    outs(%fill258 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %empty260 = tensor.empty() : tensor<1x1380x9x9xf16>
  %relu261 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv259 : tensor<1x1380x9x9xf16>)
    outs(%empty260 : tensor<1x1380x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1380x9x9xf16>

  // conv49: 3x3 s1 p1 1380->1380 9x9
  %pad262 = tensor.pad %relu261 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1380x9x9xf16> to tensor<1x1380x11x11xf16>
  %init263 = tensor.empty() : tensor<1x1380x9x9xf16>
  %fill264 = linalg.fill ins(%cst : f16) outs(%init263 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %conv265 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad262, %w49 : tensor<1x1380x11x11xf16>, tensor<1380x1380x3x3xf16>)
    outs(%fill264 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %empty266 = tensor.empty() : tensor<1x1380x9x9xf16>
  %relu267 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv265 : tensor<1x1380x9x9xf16>)
    outs(%empty266 : tensor<1x1380x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1380x9x9xf16>

  // conv50: 1x1 s1 p0 1380->230 9x9
  %init268 = tensor.empty() : tensor<1x230x9x9xf16>
  %fill269 = linalg.fill ins(%cst : f16) outs(%init268 : tensor<1x230x9x9xf16>) -> tensor<1x230x9x9xf16>
  %conv270 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu267, %w50 : tensor<1x1380x9x9xf16>, tensor<230x1380x1x1xf16>)
    outs(%fill269 : tensor<1x230x9x9xf16>) -> tensor<1x230x9x9xf16>
  %empty271 = tensor.empty() : tensor<1x230x9x9xf16>
  %relu272 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv270 : tensor<1x230x9x9xf16>)
    outs(%empty271 : tensor<1x230x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x230x9x9xf16>

  // conv51: 1x1 s1 p0 230->1380 9x9
  %init273 = tensor.empty() : tensor<1x1380x9x9xf16>
  %fill274 = linalg.fill ins(%cst : f16) outs(%init273 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %conv275 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu272, %w51 : tensor<1x230x9x9xf16>, tensor<1380x230x1x1xf16>)
    outs(%fill274 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %empty276 = tensor.empty() : tensor<1x1380x9x9xf16>
  %relu277 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv275 : tensor<1x1380x9x9xf16>)
    outs(%empty276 : tensor<1x1380x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1380x9x9xf16>

  // conv52: 3x3 s1 p1 1380->1380 9x9
  %pad278 = tensor.pad %relu277 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1380x9x9xf16> to tensor<1x1380x11x11xf16>
  %init279 = tensor.empty() : tensor<1x1380x9x9xf16>
  %fill280 = linalg.fill ins(%cst : f16) outs(%init279 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %conv281 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad278, %w52 : tensor<1x1380x11x11xf16>, tensor<1380x1380x3x3xf16>)
    outs(%fill280 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %empty282 = tensor.empty() : tensor<1x1380x9x9xf16>
  %relu283 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv281 : tensor<1x1380x9x9xf16>)
    outs(%empty282 : tensor<1x1380x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1380x9x9xf16>

  // conv53: 1x1 s1 p0 1380->230 9x9
  %init284 = tensor.empty() : tensor<1x230x9x9xf16>
  %fill285 = linalg.fill ins(%cst : f16) outs(%init284 : tensor<1x230x9x9xf16>) -> tensor<1x230x9x9xf16>
  %conv286 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu283, %w53 : tensor<1x1380x9x9xf16>, tensor<230x1380x1x1xf16>)
    outs(%fill285 : tensor<1x230x9x9xf16>) -> tensor<1x230x9x9xf16>
  %empty287 = tensor.empty() : tensor<1x230x9x9xf16>
  %relu288 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv286 : tensor<1x230x9x9xf16>)
    outs(%empty287 : tensor<1x230x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x230x9x9xf16>

  // conv54: 1x1 s1 p0 230->1380 9x9
  %init289 = tensor.empty() : tensor<1x1380x9x9xf16>
  %fill290 = linalg.fill ins(%cst : f16) outs(%init289 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %conv291 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu288, %w54 : tensor<1x230x9x9xf16>, tensor<1380x230x1x1xf16>)
    outs(%fill290 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %empty292 = tensor.empty() : tensor<1x1380x9x9xf16>
  %relu293 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv291 : tensor<1x1380x9x9xf16>)
    outs(%empty292 : tensor<1x1380x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1380x9x9xf16>

  // conv55: 3x3 s1 p1 1380->1380 9x9
  %pad294 = tensor.pad %relu293 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1380x9x9xf16> to tensor<1x1380x11x11xf16>
  %init295 = tensor.empty() : tensor<1x1380x9x9xf16>
  %fill296 = linalg.fill ins(%cst : f16) outs(%init295 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %conv297 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad294, %w55 : tensor<1x1380x11x11xf16>, tensor<1380x1380x3x3xf16>)
    outs(%fill296 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %empty298 = tensor.empty() : tensor<1x1380x9x9xf16>
  %relu299 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv297 : tensor<1x1380x9x9xf16>)
    outs(%empty298 : tensor<1x1380x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1380x9x9xf16>

  // conv56: 1x1 s1 p0 1380->230 9x9
  %init300 = tensor.empty() : tensor<1x230x9x9xf16>
  %fill301 = linalg.fill ins(%cst : f16) outs(%init300 : tensor<1x230x9x9xf16>) -> tensor<1x230x9x9xf16>
  %conv302 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu299, %w56 : tensor<1x1380x9x9xf16>, tensor<230x1380x1x1xf16>)
    outs(%fill301 : tensor<1x230x9x9xf16>) -> tensor<1x230x9x9xf16>
  %empty303 = tensor.empty() : tensor<1x230x9x9xf16>
  %relu304 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv302 : tensor<1x230x9x9xf16>)
    outs(%empty303 : tensor<1x230x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x230x9x9xf16>

  // conv57: 1x1 s1 p0 230->1380 9x9
  %init305 = tensor.empty() : tensor<1x1380x9x9xf16>
  %fill306 = linalg.fill ins(%cst : f16) outs(%init305 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %conv307 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu304, %w57 : tensor<1x230x9x9xf16>, tensor<1380x230x1x1xf16>)
    outs(%fill306 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %empty308 = tensor.empty() : tensor<1x1380x9x9xf16>
  %relu309 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv307 : tensor<1x1380x9x9xf16>)
    outs(%empty308 : tensor<1x1380x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1380x9x9xf16>

  // conv58: 3x3 s1 p1 1380->1380 9x9
  %pad310 = tensor.pad %relu309 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1380x9x9xf16> to tensor<1x1380x11x11xf16>
  %init311 = tensor.empty() : tensor<1x1380x9x9xf16>
  %fill312 = linalg.fill ins(%cst : f16) outs(%init311 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %conv313 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad310, %w58 : tensor<1x1380x11x11xf16>, tensor<1380x1380x3x3xf16>)
    outs(%fill312 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %empty314 = tensor.empty() : tensor<1x1380x9x9xf16>
  %relu315 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv313 : tensor<1x1380x9x9xf16>)
    outs(%empty314 : tensor<1x1380x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1380x9x9xf16>

  // conv59: 1x1 s1 p0 1380->230 9x9
  %init316 = tensor.empty() : tensor<1x230x9x9xf16>
  %fill317 = linalg.fill ins(%cst : f16) outs(%init316 : tensor<1x230x9x9xf16>) -> tensor<1x230x9x9xf16>
  %conv318 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu315, %w59 : tensor<1x1380x9x9xf16>, tensor<230x1380x1x1xf16>)
    outs(%fill317 : tensor<1x230x9x9xf16>) -> tensor<1x230x9x9xf16>
  %empty319 = tensor.empty() : tensor<1x230x9x9xf16>
  %relu320 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv318 : tensor<1x230x9x9xf16>)
    outs(%empty319 : tensor<1x230x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x230x9x9xf16>

  // conv60: 1x1 s1 p0 230->1380 9x9
  %init321 = tensor.empty() : tensor<1x1380x9x9xf16>
  %fill322 = linalg.fill ins(%cst : f16) outs(%init321 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %conv323 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu320, %w60 : tensor<1x230x9x9xf16>, tensor<1380x230x1x1xf16>)
    outs(%fill322 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %empty324 = tensor.empty() : tensor<1x1380x9x9xf16>
  %relu325 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv323 : tensor<1x1380x9x9xf16>)
    outs(%empty324 : tensor<1x1380x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1380x9x9xf16>

  // conv61: 3x3 s1 p1 1380->1380 9x9
  %pad326 = tensor.pad %relu325 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1380x9x9xf16> to tensor<1x1380x11x11xf16>
  %init327 = tensor.empty() : tensor<1x1380x9x9xf16>
  %fill328 = linalg.fill ins(%cst : f16) outs(%init327 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %conv329 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad326, %w61 : tensor<1x1380x11x11xf16>, tensor<1380x1380x3x3xf16>)
    outs(%fill328 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %empty330 = tensor.empty() : tensor<1x1380x9x9xf16>
  %relu331 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv329 : tensor<1x1380x9x9xf16>)
    outs(%empty330 : tensor<1x1380x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1380x9x9xf16>

  // conv62: 1x1 s1 p0 1380->230 9x9
  %init332 = tensor.empty() : tensor<1x230x9x9xf16>
  %fill333 = linalg.fill ins(%cst : f16) outs(%init332 : tensor<1x230x9x9xf16>) -> tensor<1x230x9x9xf16>
  %conv334 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu331, %w62 : tensor<1x1380x9x9xf16>, tensor<230x1380x1x1xf16>)
    outs(%fill333 : tensor<1x230x9x9xf16>) -> tensor<1x230x9x9xf16>
  %empty335 = tensor.empty() : tensor<1x230x9x9xf16>
  %relu336 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv334 : tensor<1x230x9x9xf16>)
    outs(%empty335 : tensor<1x230x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x230x9x9xf16>

  // conv63: 1x1 s1 p0 230->1380 9x9
  %init337 = tensor.empty() : tensor<1x1380x9x9xf16>
  %fill338 = linalg.fill ins(%cst : f16) outs(%init337 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %conv339 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu336, %w63 : tensor<1x230x9x9xf16>, tensor<1380x230x1x1xf16>)
    outs(%fill338 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %empty340 = tensor.empty() : tensor<1x1380x9x9xf16>
  %relu341 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv339 : tensor<1x1380x9x9xf16>)
    outs(%empty340 : tensor<1x1380x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1380x9x9xf16>

  // conv64: 3x3 s1 p1 1380->1380 9x9
  %pad342 = tensor.pad %relu341 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1380x9x9xf16> to tensor<1x1380x11x11xf16>
  %init343 = tensor.empty() : tensor<1x1380x9x9xf16>
  %fill344 = linalg.fill ins(%cst : f16) outs(%init343 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %conv345 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad342, %w64 : tensor<1x1380x11x11xf16>, tensor<1380x1380x3x3xf16>)
    outs(%fill344 : tensor<1x1380x9x9xf16>) -> tensor<1x1380x9x9xf16>
  %empty346 = tensor.empty() : tensor<1x1380x9x9xf16>
  %relu347 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv345 : tensor<1x1380x9x9xf16>)
    outs(%empty346 : tensor<1x1380x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1380x9x9xf16>

  // conv65: 1x1 s1 p0 1380->384 9x9
  %init348 = tensor.empty() : tensor<1x384x9x9xf16>
  %fill349 = linalg.fill ins(%cst : f16) outs(%init348 : tensor<1x384x9x9xf16>) -> tensor<1x384x9x9xf16>
  %conv350 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu347, %w65 : tensor<1x1380x9x9xf16>, tensor<384x1380x1x1xf16>)
    outs(%fill349 : tensor<1x384x9x9xf16>) -> tensor<1x384x9x9xf16>
  %empty351 = tensor.empty() : tensor<1x384x9x9xf16>
  %relu352 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv350 : tensor<1x384x9x9xf16>)
    outs(%empty351 : tensor<1x384x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x384x9x9xf16>

  // conv66: 1x1 s1 p0 384->1536 9x9
  %init353 = tensor.empty() : tensor<1x1536x9x9xf16>
  %fill354 = linalg.fill ins(%cst : f16) outs(%init353 : tensor<1x1536x9x9xf16>) -> tensor<1x1536x9x9xf16>
  %conv355 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu352, %w66 : tensor<1x384x9x9xf16>, tensor<1536x384x1x1xf16>)
    outs(%fill354 : tensor<1x1536x9x9xf16>) -> tensor<1x1536x9x9xf16>
  %empty356 = tensor.empty() : tensor<1x1536x9x9xf16>
  %relu357 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv355 : tensor<1x1536x9x9xf16>)
    outs(%empty356 : tensor<1x1536x9x9xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1536x9x9xf16>

  // FC as 1x1 conv: 1536->1000
  %init358 = tensor.empty() : tensor<1x1000x9x9xf16>
  %fill359 = linalg.fill ins(%cst : f16) outs(%init358 : tensor<1x1000x9x9xf16>) -> tensor<1x1000x9x9xf16>
  %conv360 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu357, %w_fc : tensor<1x1536x9x9xf16>, tensor<1000x1536x1x1xf16>)
    outs(%fill359 : tensor<1x1000x9x9xf16>) -> tensor<1x1000x9x9xf16>
  return %conv360 : tensor<1x1000x9x9xf16>
}
