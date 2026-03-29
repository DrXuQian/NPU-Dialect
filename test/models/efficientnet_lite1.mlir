func.func @efficientnet_lite1(
    %input: tensor<1x3x240x240xf16>,
    %w0: tensor<32x3x3x3xf16>,
    %w1: tensor<32x32x3x3xf16>,
    %w2: tensor<16x32x1x1xf16>,
    %w3: tensor<96x16x1x1xf16>,
    %w4: tensor<96x96x3x3xf16>,
    %w5: tensor<24x96x1x1xf16>,
    %w6: tensor<144x24x1x1xf16>,
    %w7: tensor<144x144x3x3xf16>,
    %w8: tensor<24x144x1x1xf16>,
    %w9: tensor<144x24x1x1xf16>,
    %w10: tensor<144x144x3x3xf16>,
    %w11: tensor<40x144x1x1xf16>,
    %w12: tensor<240x40x1x1xf16>,
    %w13: tensor<240x240x3x3xf16>,
    %w14: tensor<40x240x1x1xf16>,
    %w15: tensor<240x40x1x1xf16>,
    %w16: tensor<240x240x3x3xf16>,
    %w17: tensor<80x240x1x1xf16>,
    %w18: tensor<480x80x1x1xf16>,
    %w19: tensor<480x480x3x3xf16>,
    %w20: tensor<80x480x1x1xf16>,
    %w21: tensor<480x80x1x1xf16>,
    %w22: tensor<480x480x3x3xf16>,
    %w23: tensor<80x480x1x1xf16>,
    %w24: tensor<480x80x1x1xf16>,
    %w25: tensor<480x480x3x3xf16>,
    %w26: tensor<112x480x1x1xf16>,
    %w27: tensor<672x112x1x1xf16>,
    %w28: tensor<672x672x3x3xf16>,
    %w29: tensor<112x672x1x1xf16>,
    %w30: tensor<672x112x1x1xf16>,
    %w31: tensor<672x672x3x3xf16>,
    %w32: tensor<112x672x1x1xf16>,
    %w33: tensor<672x112x1x1xf16>,
    %w34: tensor<672x672x3x3xf16>,
    %w35: tensor<192x672x1x1xf16>,
    %w36: tensor<1152x192x1x1xf16>,
    %w37: tensor<1152x1152x3x3xf16>,
    %w38: tensor<192x1152x1x1xf16>,
    %w39: tensor<1152x192x1x1xf16>,
    %w40: tensor<1152x1152x3x3xf16>,
    %w41: tensor<192x1152x1x1xf16>,
    %w42: tensor<1152x192x1x1xf16>,
    %w43: tensor<1152x1152x3x3xf16>,
    %w44: tensor<192x1152x1x1xf16>,
    %w45: tensor<1152x192x1x1xf16>,
    %w46: tensor<1152x1152x3x3xf16>,
    %w47: tensor<320x1152x1x1xf16>,
    %w48: tensor<1280x320x1x1xf16>,
    %w_fc: tensor<1000x1280x1x1xf16>) -> tensor<1x1000x8x8xf16> {
  %cst = arith.constant 0.0 : f16

  // conv0: 3x3 s2 p1 3->32 240x240
  %pad0 = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x240x240xf16> to tensor<1x3x242x242xf16>
  %init1 = tensor.empty() : tensor<1x32x120x120xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x32x120x120xf16>) -> tensor<1x32x120x120xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad0, %w0 : tensor<1x3x242x242xf16>, tensor<32x3x3x3xf16>)
    outs(%fill2 : tensor<1x32x120x120xf16>) -> tensor<1x32x120x120xf16>
  %empty4 = tensor.empty() : tensor<1x32x120x120xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x32x120x120xf16>)
    outs(%empty4 : tensor<1x32x120x120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x120x120xf16>

  // conv1: 3x3 s1 p1 32->32 120x120
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x32x120x120xf16> to tensor<1x32x122x122xf16>
  %init7 = tensor.empty() : tensor<1x32x120x120xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x32x120x120xf16>) -> tensor<1x32x120x120xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad6, %w1 : tensor<1x32x122x122xf16>, tensor<32x32x3x3xf16>)
    outs(%fill8 : tensor<1x32x120x120xf16>) -> tensor<1x32x120x120xf16>
  %empty10 = tensor.empty() : tensor<1x32x120x120xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x32x120x120xf16>)
    outs(%empty10 : tensor<1x32x120x120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x120x120xf16>

  // conv2: 1x1 s1 p0 32->16 120x120
  %init12 = tensor.empty() : tensor<1x16x120x120xf16>
  %fill13 = linalg.fill ins(%cst : f16) outs(%init12 : tensor<1x16x120x120xf16>) -> tensor<1x16x120x120xf16>
  %conv14 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu11, %w2 : tensor<1x32x120x120xf16>, tensor<16x32x1x1xf16>)
    outs(%fill13 : tensor<1x16x120x120xf16>) -> tensor<1x16x120x120xf16>
  %empty15 = tensor.empty() : tensor<1x16x120x120xf16>
  %relu16 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv14 : tensor<1x16x120x120xf16>)
    outs(%empty15 : tensor<1x16x120x120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x16x120x120xf16>

  // conv3: 1x1 s1 p0 16->96 120x120
  %init17 = tensor.empty() : tensor<1x96x120x120xf16>
  %fill18 = linalg.fill ins(%cst : f16) outs(%init17 : tensor<1x96x120x120xf16>) -> tensor<1x96x120x120xf16>
  %conv19 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu16, %w3 : tensor<1x16x120x120xf16>, tensor<96x16x1x1xf16>)
    outs(%fill18 : tensor<1x96x120x120xf16>) -> tensor<1x96x120x120xf16>
  %empty20 = tensor.empty() : tensor<1x96x120x120xf16>
  %relu21 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv19 : tensor<1x96x120x120xf16>)
    outs(%empty20 : tensor<1x96x120x120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x96x120x120xf16>

  // conv4: 3x3 s2 p1 96->96 120x120
  %pad22 = tensor.pad %relu21 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x96x120x120xf16> to tensor<1x96x122x122xf16>
  %init23 = tensor.empty() : tensor<1x96x60x60xf16>
  %fill24 = linalg.fill ins(%cst : f16) outs(%init23 : tensor<1x96x60x60xf16>) -> tensor<1x96x60x60xf16>
  %conv25 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad22, %w4 : tensor<1x96x122x122xf16>, tensor<96x96x3x3xf16>)
    outs(%fill24 : tensor<1x96x60x60xf16>) -> tensor<1x96x60x60xf16>
  %empty26 = tensor.empty() : tensor<1x96x60x60xf16>
  %relu27 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv25 : tensor<1x96x60x60xf16>)
    outs(%empty26 : tensor<1x96x60x60xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x96x60x60xf16>

  // conv5: 1x1 s1 p0 96->24 60x60
  %init28 = tensor.empty() : tensor<1x24x60x60xf16>
  %fill29 = linalg.fill ins(%cst : f16) outs(%init28 : tensor<1x24x60x60xf16>) -> tensor<1x24x60x60xf16>
  %conv30 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu27, %w5 : tensor<1x96x60x60xf16>, tensor<24x96x1x1xf16>)
    outs(%fill29 : tensor<1x24x60x60xf16>) -> tensor<1x24x60x60xf16>
  %empty31 = tensor.empty() : tensor<1x24x60x60xf16>
  %relu32 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv30 : tensor<1x24x60x60xf16>)
    outs(%empty31 : tensor<1x24x60x60xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x24x60x60xf16>

  // conv6: 1x1 s1 p0 24->144 60x60
  %init33 = tensor.empty() : tensor<1x144x60x60xf16>
  %fill34 = linalg.fill ins(%cst : f16) outs(%init33 : tensor<1x144x60x60xf16>) -> tensor<1x144x60x60xf16>
  %conv35 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu32, %w6 : tensor<1x24x60x60xf16>, tensor<144x24x1x1xf16>)
    outs(%fill34 : tensor<1x144x60x60xf16>) -> tensor<1x144x60x60xf16>
  %empty36 = tensor.empty() : tensor<1x144x60x60xf16>
  %relu37 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv35 : tensor<1x144x60x60xf16>)
    outs(%empty36 : tensor<1x144x60x60xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x144x60x60xf16>

  // conv7: 3x3 s1 p1 144->144 60x60
  %pad38 = tensor.pad %relu37 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x144x60x60xf16> to tensor<1x144x62x62xf16>
  %init39 = tensor.empty() : tensor<1x144x60x60xf16>
  %fill40 = linalg.fill ins(%cst : f16) outs(%init39 : tensor<1x144x60x60xf16>) -> tensor<1x144x60x60xf16>
  %conv41 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad38, %w7 : tensor<1x144x62x62xf16>, tensor<144x144x3x3xf16>)
    outs(%fill40 : tensor<1x144x60x60xf16>) -> tensor<1x144x60x60xf16>
  %empty42 = tensor.empty() : tensor<1x144x60x60xf16>
  %relu43 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv41 : tensor<1x144x60x60xf16>)
    outs(%empty42 : tensor<1x144x60x60xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x144x60x60xf16>

  // conv8: 1x1 s1 p0 144->24 60x60
  %init44 = tensor.empty() : tensor<1x24x60x60xf16>
  %fill45 = linalg.fill ins(%cst : f16) outs(%init44 : tensor<1x24x60x60xf16>) -> tensor<1x24x60x60xf16>
  %conv46 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu43, %w8 : tensor<1x144x60x60xf16>, tensor<24x144x1x1xf16>)
    outs(%fill45 : tensor<1x24x60x60xf16>) -> tensor<1x24x60x60xf16>
  %empty47 = tensor.empty() : tensor<1x24x60x60xf16>
  %relu48 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv46 : tensor<1x24x60x60xf16>)
    outs(%empty47 : tensor<1x24x60x60xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x24x60x60xf16>

  // conv9: 1x1 s1 p0 24->144 60x60
  %init49 = tensor.empty() : tensor<1x144x60x60xf16>
  %fill50 = linalg.fill ins(%cst : f16) outs(%init49 : tensor<1x144x60x60xf16>) -> tensor<1x144x60x60xf16>
  %conv51 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu48, %w9 : tensor<1x24x60x60xf16>, tensor<144x24x1x1xf16>)
    outs(%fill50 : tensor<1x144x60x60xf16>) -> tensor<1x144x60x60xf16>
  %empty52 = tensor.empty() : tensor<1x144x60x60xf16>
  %relu53 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv51 : tensor<1x144x60x60xf16>)
    outs(%empty52 : tensor<1x144x60x60xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x144x60x60xf16>

  // conv10: 3x3 s2 p1 144->144 60x60
  %pad54 = tensor.pad %relu53 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x144x60x60xf16> to tensor<1x144x62x62xf16>
  %init55 = tensor.empty() : tensor<1x144x30x30xf16>
  %fill56 = linalg.fill ins(%cst : f16) outs(%init55 : tensor<1x144x30x30xf16>) -> tensor<1x144x30x30xf16>
  %conv57 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad54, %w10 : tensor<1x144x62x62xf16>, tensor<144x144x3x3xf16>)
    outs(%fill56 : tensor<1x144x30x30xf16>) -> tensor<1x144x30x30xf16>
  %empty58 = tensor.empty() : tensor<1x144x30x30xf16>
  %relu59 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv57 : tensor<1x144x30x30xf16>)
    outs(%empty58 : tensor<1x144x30x30xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x144x30x30xf16>

  // conv11: 1x1 s1 p0 144->40 30x30
  %init60 = tensor.empty() : tensor<1x40x30x30xf16>
  %fill61 = linalg.fill ins(%cst : f16) outs(%init60 : tensor<1x40x30x30xf16>) -> tensor<1x40x30x30xf16>
  %conv62 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu59, %w11 : tensor<1x144x30x30xf16>, tensor<40x144x1x1xf16>)
    outs(%fill61 : tensor<1x40x30x30xf16>) -> tensor<1x40x30x30xf16>
  %empty63 = tensor.empty() : tensor<1x40x30x30xf16>
  %relu64 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv62 : tensor<1x40x30x30xf16>)
    outs(%empty63 : tensor<1x40x30x30xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x40x30x30xf16>

  // conv12: 1x1 s1 p0 40->240 30x30
  %init65 = tensor.empty() : tensor<1x240x30x30xf16>
  %fill66 = linalg.fill ins(%cst : f16) outs(%init65 : tensor<1x240x30x30xf16>) -> tensor<1x240x30x30xf16>
  %conv67 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu64, %w12 : tensor<1x40x30x30xf16>, tensor<240x40x1x1xf16>)
    outs(%fill66 : tensor<1x240x30x30xf16>) -> tensor<1x240x30x30xf16>
  %empty68 = tensor.empty() : tensor<1x240x30x30xf16>
  %relu69 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv67 : tensor<1x240x30x30xf16>)
    outs(%empty68 : tensor<1x240x30x30xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x240x30x30xf16>

  // conv13: 3x3 s1 p1 240->240 30x30
  %pad70 = tensor.pad %relu69 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x240x30x30xf16> to tensor<1x240x32x32xf16>
  %init71 = tensor.empty() : tensor<1x240x30x30xf16>
  %fill72 = linalg.fill ins(%cst : f16) outs(%init71 : tensor<1x240x30x30xf16>) -> tensor<1x240x30x30xf16>
  %conv73 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad70, %w13 : tensor<1x240x32x32xf16>, tensor<240x240x3x3xf16>)
    outs(%fill72 : tensor<1x240x30x30xf16>) -> tensor<1x240x30x30xf16>
  %empty74 = tensor.empty() : tensor<1x240x30x30xf16>
  %relu75 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv73 : tensor<1x240x30x30xf16>)
    outs(%empty74 : tensor<1x240x30x30xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x240x30x30xf16>

  // conv14: 1x1 s1 p0 240->40 30x30
  %init76 = tensor.empty() : tensor<1x40x30x30xf16>
  %fill77 = linalg.fill ins(%cst : f16) outs(%init76 : tensor<1x40x30x30xf16>) -> tensor<1x40x30x30xf16>
  %conv78 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu75, %w14 : tensor<1x240x30x30xf16>, tensor<40x240x1x1xf16>)
    outs(%fill77 : tensor<1x40x30x30xf16>) -> tensor<1x40x30x30xf16>
  %empty79 = tensor.empty() : tensor<1x40x30x30xf16>
  %relu80 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv78 : tensor<1x40x30x30xf16>)
    outs(%empty79 : tensor<1x40x30x30xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x40x30x30xf16>

  // conv15: 1x1 s1 p0 40->240 30x30
  %init81 = tensor.empty() : tensor<1x240x30x30xf16>
  %fill82 = linalg.fill ins(%cst : f16) outs(%init81 : tensor<1x240x30x30xf16>) -> tensor<1x240x30x30xf16>
  %conv83 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu80, %w15 : tensor<1x40x30x30xf16>, tensor<240x40x1x1xf16>)
    outs(%fill82 : tensor<1x240x30x30xf16>) -> tensor<1x240x30x30xf16>
  %empty84 = tensor.empty() : tensor<1x240x30x30xf16>
  %relu85 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv83 : tensor<1x240x30x30xf16>)
    outs(%empty84 : tensor<1x240x30x30xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x240x30x30xf16>

  // conv16: 3x3 s2 p1 240->240 30x30
  %pad86 = tensor.pad %relu85 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x240x30x30xf16> to tensor<1x240x32x32xf16>
  %init87 = tensor.empty() : tensor<1x240x15x15xf16>
  %fill88 = linalg.fill ins(%cst : f16) outs(%init87 : tensor<1x240x15x15xf16>) -> tensor<1x240x15x15xf16>
  %conv89 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad86, %w16 : tensor<1x240x32x32xf16>, tensor<240x240x3x3xf16>)
    outs(%fill88 : tensor<1x240x15x15xf16>) -> tensor<1x240x15x15xf16>
  %empty90 = tensor.empty() : tensor<1x240x15x15xf16>
  %relu91 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv89 : tensor<1x240x15x15xf16>)
    outs(%empty90 : tensor<1x240x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x240x15x15xf16>

  // conv17: 1x1 s1 p0 240->80 15x15
  %init92 = tensor.empty() : tensor<1x80x15x15xf16>
  %fill93 = linalg.fill ins(%cst : f16) outs(%init92 : tensor<1x80x15x15xf16>) -> tensor<1x80x15x15xf16>
  %conv94 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu91, %w17 : tensor<1x240x15x15xf16>, tensor<80x240x1x1xf16>)
    outs(%fill93 : tensor<1x80x15x15xf16>) -> tensor<1x80x15x15xf16>
  %empty95 = tensor.empty() : tensor<1x80x15x15xf16>
  %relu96 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv94 : tensor<1x80x15x15xf16>)
    outs(%empty95 : tensor<1x80x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x80x15x15xf16>

  // conv18: 1x1 s1 p0 80->480 15x15
  %init97 = tensor.empty() : tensor<1x480x15x15xf16>
  %fill98 = linalg.fill ins(%cst : f16) outs(%init97 : tensor<1x480x15x15xf16>) -> tensor<1x480x15x15xf16>
  %conv99 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu96, %w18 : tensor<1x80x15x15xf16>, tensor<480x80x1x1xf16>)
    outs(%fill98 : tensor<1x480x15x15xf16>) -> tensor<1x480x15x15xf16>
  %empty100 = tensor.empty() : tensor<1x480x15x15xf16>
  %relu101 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv99 : tensor<1x480x15x15xf16>)
    outs(%empty100 : tensor<1x480x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x480x15x15xf16>

  // conv19: 3x3 s1 p1 480->480 15x15
  %pad102 = tensor.pad %relu101 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x480x15x15xf16> to tensor<1x480x17x17xf16>
  %init103 = tensor.empty() : tensor<1x480x15x15xf16>
  %fill104 = linalg.fill ins(%cst : f16) outs(%init103 : tensor<1x480x15x15xf16>) -> tensor<1x480x15x15xf16>
  %conv105 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad102, %w19 : tensor<1x480x17x17xf16>, tensor<480x480x3x3xf16>)
    outs(%fill104 : tensor<1x480x15x15xf16>) -> tensor<1x480x15x15xf16>
  %empty106 = tensor.empty() : tensor<1x480x15x15xf16>
  %relu107 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv105 : tensor<1x480x15x15xf16>)
    outs(%empty106 : tensor<1x480x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x480x15x15xf16>

  // conv20: 1x1 s1 p0 480->80 15x15
  %init108 = tensor.empty() : tensor<1x80x15x15xf16>
  %fill109 = linalg.fill ins(%cst : f16) outs(%init108 : tensor<1x80x15x15xf16>) -> tensor<1x80x15x15xf16>
  %conv110 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu107, %w20 : tensor<1x480x15x15xf16>, tensor<80x480x1x1xf16>)
    outs(%fill109 : tensor<1x80x15x15xf16>) -> tensor<1x80x15x15xf16>
  %empty111 = tensor.empty() : tensor<1x80x15x15xf16>
  %relu112 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv110 : tensor<1x80x15x15xf16>)
    outs(%empty111 : tensor<1x80x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x80x15x15xf16>

  // conv21: 1x1 s1 p0 80->480 15x15
  %init113 = tensor.empty() : tensor<1x480x15x15xf16>
  %fill114 = linalg.fill ins(%cst : f16) outs(%init113 : tensor<1x480x15x15xf16>) -> tensor<1x480x15x15xf16>
  %conv115 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu112, %w21 : tensor<1x80x15x15xf16>, tensor<480x80x1x1xf16>)
    outs(%fill114 : tensor<1x480x15x15xf16>) -> tensor<1x480x15x15xf16>
  %empty116 = tensor.empty() : tensor<1x480x15x15xf16>
  %relu117 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv115 : tensor<1x480x15x15xf16>)
    outs(%empty116 : tensor<1x480x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x480x15x15xf16>

  // conv22: 3x3 s1 p1 480->480 15x15
  %pad118 = tensor.pad %relu117 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x480x15x15xf16> to tensor<1x480x17x17xf16>
  %init119 = tensor.empty() : tensor<1x480x15x15xf16>
  %fill120 = linalg.fill ins(%cst : f16) outs(%init119 : tensor<1x480x15x15xf16>) -> tensor<1x480x15x15xf16>
  %conv121 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad118, %w22 : tensor<1x480x17x17xf16>, tensor<480x480x3x3xf16>)
    outs(%fill120 : tensor<1x480x15x15xf16>) -> tensor<1x480x15x15xf16>
  %empty122 = tensor.empty() : tensor<1x480x15x15xf16>
  %relu123 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv121 : tensor<1x480x15x15xf16>)
    outs(%empty122 : tensor<1x480x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x480x15x15xf16>

  // conv23: 1x1 s1 p0 480->80 15x15
  %init124 = tensor.empty() : tensor<1x80x15x15xf16>
  %fill125 = linalg.fill ins(%cst : f16) outs(%init124 : tensor<1x80x15x15xf16>) -> tensor<1x80x15x15xf16>
  %conv126 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu123, %w23 : tensor<1x480x15x15xf16>, tensor<80x480x1x1xf16>)
    outs(%fill125 : tensor<1x80x15x15xf16>) -> tensor<1x80x15x15xf16>
  %empty127 = tensor.empty() : tensor<1x80x15x15xf16>
  %relu128 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv126 : tensor<1x80x15x15xf16>)
    outs(%empty127 : tensor<1x80x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x80x15x15xf16>

  // conv24: 1x1 s1 p0 80->480 15x15
  %init129 = tensor.empty() : tensor<1x480x15x15xf16>
  %fill130 = linalg.fill ins(%cst : f16) outs(%init129 : tensor<1x480x15x15xf16>) -> tensor<1x480x15x15xf16>
  %conv131 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu128, %w24 : tensor<1x80x15x15xf16>, tensor<480x80x1x1xf16>)
    outs(%fill130 : tensor<1x480x15x15xf16>) -> tensor<1x480x15x15xf16>
  %empty132 = tensor.empty() : tensor<1x480x15x15xf16>
  %relu133 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv131 : tensor<1x480x15x15xf16>)
    outs(%empty132 : tensor<1x480x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x480x15x15xf16>

  // conv25: 3x3 s1 p1 480->480 15x15
  %pad134 = tensor.pad %relu133 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x480x15x15xf16> to tensor<1x480x17x17xf16>
  %init135 = tensor.empty() : tensor<1x480x15x15xf16>
  %fill136 = linalg.fill ins(%cst : f16) outs(%init135 : tensor<1x480x15x15xf16>) -> tensor<1x480x15x15xf16>
  %conv137 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad134, %w25 : tensor<1x480x17x17xf16>, tensor<480x480x3x3xf16>)
    outs(%fill136 : tensor<1x480x15x15xf16>) -> tensor<1x480x15x15xf16>
  %empty138 = tensor.empty() : tensor<1x480x15x15xf16>
  %relu139 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv137 : tensor<1x480x15x15xf16>)
    outs(%empty138 : tensor<1x480x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x480x15x15xf16>

  // conv26: 1x1 s1 p0 480->112 15x15
  %init140 = tensor.empty() : tensor<1x112x15x15xf16>
  %fill141 = linalg.fill ins(%cst : f16) outs(%init140 : tensor<1x112x15x15xf16>) -> tensor<1x112x15x15xf16>
  %conv142 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu139, %w26 : tensor<1x480x15x15xf16>, tensor<112x480x1x1xf16>)
    outs(%fill141 : tensor<1x112x15x15xf16>) -> tensor<1x112x15x15xf16>
  %empty143 = tensor.empty() : tensor<1x112x15x15xf16>
  %relu144 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv142 : tensor<1x112x15x15xf16>)
    outs(%empty143 : tensor<1x112x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x112x15x15xf16>

  // conv27: 1x1 s1 p0 112->672 15x15
  %init145 = tensor.empty() : tensor<1x672x15x15xf16>
  %fill146 = linalg.fill ins(%cst : f16) outs(%init145 : tensor<1x672x15x15xf16>) -> tensor<1x672x15x15xf16>
  %conv147 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu144, %w27 : tensor<1x112x15x15xf16>, tensor<672x112x1x1xf16>)
    outs(%fill146 : tensor<1x672x15x15xf16>) -> tensor<1x672x15x15xf16>
  %empty148 = tensor.empty() : tensor<1x672x15x15xf16>
  %relu149 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv147 : tensor<1x672x15x15xf16>)
    outs(%empty148 : tensor<1x672x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x15x15xf16>

  // conv28: 3x3 s1 p1 672->672 15x15
  %pad150 = tensor.pad %relu149 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x672x15x15xf16> to tensor<1x672x17x17xf16>
  %init151 = tensor.empty() : tensor<1x672x15x15xf16>
  %fill152 = linalg.fill ins(%cst : f16) outs(%init151 : tensor<1x672x15x15xf16>) -> tensor<1x672x15x15xf16>
  %conv153 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad150, %w28 : tensor<1x672x17x17xf16>, tensor<672x672x3x3xf16>)
    outs(%fill152 : tensor<1x672x15x15xf16>) -> tensor<1x672x15x15xf16>
  %empty154 = tensor.empty() : tensor<1x672x15x15xf16>
  %relu155 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv153 : tensor<1x672x15x15xf16>)
    outs(%empty154 : tensor<1x672x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x15x15xf16>

  // conv29: 1x1 s1 p0 672->112 15x15
  %init156 = tensor.empty() : tensor<1x112x15x15xf16>
  %fill157 = linalg.fill ins(%cst : f16) outs(%init156 : tensor<1x112x15x15xf16>) -> tensor<1x112x15x15xf16>
  %conv158 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu155, %w29 : tensor<1x672x15x15xf16>, tensor<112x672x1x1xf16>)
    outs(%fill157 : tensor<1x112x15x15xf16>) -> tensor<1x112x15x15xf16>
  %empty159 = tensor.empty() : tensor<1x112x15x15xf16>
  %relu160 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv158 : tensor<1x112x15x15xf16>)
    outs(%empty159 : tensor<1x112x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x112x15x15xf16>

  // conv30: 1x1 s1 p0 112->672 15x15
  %init161 = tensor.empty() : tensor<1x672x15x15xf16>
  %fill162 = linalg.fill ins(%cst : f16) outs(%init161 : tensor<1x672x15x15xf16>) -> tensor<1x672x15x15xf16>
  %conv163 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu160, %w30 : tensor<1x112x15x15xf16>, tensor<672x112x1x1xf16>)
    outs(%fill162 : tensor<1x672x15x15xf16>) -> tensor<1x672x15x15xf16>
  %empty164 = tensor.empty() : tensor<1x672x15x15xf16>
  %relu165 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv163 : tensor<1x672x15x15xf16>)
    outs(%empty164 : tensor<1x672x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x15x15xf16>

  // conv31: 3x3 s1 p1 672->672 15x15
  %pad166 = tensor.pad %relu165 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x672x15x15xf16> to tensor<1x672x17x17xf16>
  %init167 = tensor.empty() : tensor<1x672x15x15xf16>
  %fill168 = linalg.fill ins(%cst : f16) outs(%init167 : tensor<1x672x15x15xf16>) -> tensor<1x672x15x15xf16>
  %conv169 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad166, %w31 : tensor<1x672x17x17xf16>, tensor<672x672x3x3xf16>)
    outs(%fill168 : tensor<1x672x15x15xf16>) -> tensor<1x672x15x15xf16>
  %empty170 = tensor.empty() : tensor<1x672x15x15xf16>
  %relu171 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv169 : tensor<1x672x15x15xf16>)
    outs(%empty170 : tensor<1x672x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x15x15xf16>

  // conv32: 1x1 s1 p0 672->112 15x15
  %init172 = tensor.empty() : tensor<1x112x15x15xf16>
  %fill173 = linalg.fill ins(%cst : f16) outs(%init172 : tensor<1x112x15x15xf16>) -> tensor<1x112x15x15xf16>
  %conv174 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu171, %w32 : tensor<1x672x15x15xf16>, tensor<112x672x1x1xf16>)
    outs(%fill173 : tensor<1x112x15x15xf16>) -> tensor<1x112x15x15xf16>
  %empty175 = tensor.empty() : tensor<1x112x15x15xf16>
  %relu176 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv174 : tensor<1x112x15x15xf16>)
    outs(%empty175 : tensor<1x112x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x112x15x15xf16>

  // conv33: 1x1 s1 p0 112->672 15x15
  %init177 = tensor.empty() : tensor<1x672x15x15xf16>
  %fill178 = linalg.fill ins(%cst : f16) outs(%init177 : tensor<1x672x15x15xf16>) -> tensor<1x672x15x15xf16>
  %conv179 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu176, %w33 : tensor<1x112x15x15xf16>, tensor<672x112x1x1xf16>)
    outs(%fill178 : tensor<1x672x15x15xf16>) -> tensor<1x672x15x15xf16>
  %empty180 = tensor.empty() : tensor<1x672x15x15xf16>
  %relu181 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv179 : tensor<1x672x15x15xf16>)
    outs(%empty180 : tensor<1x672x15x15xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x15x15xf16>

  // conv34: 3x3 s2 p1 672->672 15x15
  %pad182 = tensor.pad %relu181 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x672x15x15xf16> to tensor<1x672x17x17xf16>
  %init183 = tensor.empty() : tensor<1x672x8x8xf16>
  %fill184 = linalg.fill ins(%cst : f16) outs(%init183 : tensor<1x672x8x8xf16>) -> tensor<1x672x8x8xf16>
  %conv185 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad182, %w34 : tensor<1x672x17x17xf16>, tensor<672x672x3x3xf16>)
    outs(%fill184 : tensor<1x672x8x8xf16>) -> tensor<1x672x8x8xf16>
  %empty186 = tensor.empty() : tensor<1x672x8x8xf16>
  %relu187 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv185 : tensor<1x672x8x8xf16>)
    outs(%empty186 : tensor<1x672x8x8xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x672x8x8xf16>

  // conv35: 1x1 s1 p0 672->192 8x8
  %init188 = tensor.empty() : tensor<1x192x8x8xf16>
  %fill189 = linalg.fill ins(%cst : f16) outs(%init188 : tensor<1x192x8x8xf16>) -> tensor<1x192x8x8xf16>
  %conv190 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu187, %w35 : tensor<1x672x8x8xf16>, tensor<192x672x1x1xf16>)
    outs(%fill189 : tensor<1x192x8x8xf16>) -> tensor<1x192x8x8xf16>
  %empty191 = tensor.empty() : tensor<1x192x8x8xf16>
  %relu192 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv190 : tensor<1x192x8x8xf16>)
    outs(%empty191 : tensor<1x192x8x8xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x8x8xf16>

  // conv36: 1x1 s1 p0 192->1152 8x8
  %init193 = tensor.empty() : tensor<1x1152x8x8xf16>
  %fill194 = linalg.fill ins(%cst : f16) outs(%init193 : tensor<1x1152x8x8xf16>) -> tensor<1x1152x8x8xf16>
  %conv195 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu192, %w36 : tensor<1x192x8x8xf16>, tensor<1152x192x1x1xf16>)
    outs(%fill194 : tensor<1x1152x8x8xf16>) -> tensor<1x1152x8x8xf16>
  %empty196 = tensor.empty() : tensor<1x1152x8x8xf16>
  %relu197 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv195 : tensor<1x1152x8x8xf16>)
    outs(%empty196 : tensor<1x1152x8x8xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1152x8x8xf16>

  // conv37: 3x3 s1 p1 1152->1152 8x8
  %pad198 = tensor.pad %relu197 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1152x8x8xf16> to tensor<1x1152x10x10xf16>
  %init199 = tensor.empty() : tensor<1x1152x8x8xf16>
  %fill200 = linalg.fill ins(%cst : f16) outs(%init199 : tensor<1x1152x8x8xf16>) -> tensor<1x1152x8x8xf16>
  %conv201 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad198, %w37 : tensor<1x1152x10x10xf16>, tensor<1152x1152x3x3xf16>)
    outs(%fill200 : tensor<1x1152x8x8xf16>) -> tensor<1x1152x8x8xf16>
  %empty202 = tensor.empty() : tensor<1x1152x8x8xf16>
  %relu203 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv201 : tensor<1x1152x8x8xf16>)
    outs(%empty202 : tensor<1x1152x8x8xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1152x8x8xf16>

  // conv38: 1x1 s1 p0 1152->192 8x8
  %init204 = tensor.empty() : tensor<1x192x8x8xf16>
  %fill205 = linalg.fill ins(%cst : f16) outs(%init204 : tensor<1x192x8x8xf16>) -> tensor<1x192x8x8xf16>
  %conv206 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu203, %w38 : tensor<1x1152x8x8xf16>, tensor<192x1152x1x1xf16>)
    outs(%fill205 : tensor<1x192x8x8xf16>) -> tensor<1x192x8x8xf16>
  %empty207 = tensor.empty() : tensor<1x192x8x8xf16>
  %relu208 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv206 : tensor<1x192x8x8xf16>)
    outs(%empty207 : tensor<1x192x8x8xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x8x8xf16>

  // conv39: 1x1 s1 p0 192->1152 8x8
  %init209 = tensor.empty() : tensor<1x1152x8x8xf16>
  %fill210 = linalg.fill ins(%cst : f16) outs(%init209 : tensor<1x1152x8x8xf16>) -> tensor<1x1152x8x8xf16>
  %conv211 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu208, %w39 : tensor<1x192x8x8xf16>, tensor<1152x192x1x1xf16>)
    outs(%fill210 : tensor<1x1152x8x8xf16>) -> tensor<1x1152x8x8xf16>
  %empty212 = tensor.empty() : tensor<1x1152x8x8xf16>
  %relu213 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv211 : tensor<1x1152x8x8xf16>)
    outs(%empty212 : tensor<1x1152x8x8xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1152x8x8xf16>

  // conv40: 3x3 s1 p1 1152->1152 8x8
  %pad214 = tensor.pad %relu213 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1152x8x8xf16> to tensor<1x1152x10x10xf16>
  %init215 = tensor.empty() : tensor<1x1152x8x8xf16>
  %fill216 = linalg.fill ins(%cst : f16) outs(%init215 : tensor<1x1152x8x8xf16>) -> tensor<1x1152x8x8xf16>
  %conv217 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad214, %w40 : tensor<1x1152x10x10xf16>, tensor<1152x1152x3x3xf16>)
    outs(%fill216 : tensor<1x1152x8x8xf16>) -> tensor<1x1152x8x8xf16>
  %empty218 = tensor.empty() : tensor<1x1152x8x8xf16>
  %relu219 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv217 : tensor<1x1152x8x8xf16>)
    outs(%empty218 : tensor<1x1152x8x8xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1152x8x8xf16>

  // conv41: 1x1 s1 p0 1152->192 8x8
  %init220 = tensor.empty() : tensor<1x192x8x8xf16>
  %fill221 = linalg.fill ins(%cst : f16) outs(%init220 : tensor<1x192x8x8xf16>) -> tensor<1x192x8x8xf16>
  %conv222 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu219, %w41 : tensor<1x1152x8x8xf16>, tensor<192x1152x1x1xf16>)
    outs(%fill221 : tensor<1x192x8x8xf16>) -> tensor<1x192x8x8xf16>
  %empty223 = tensor.empty() : tensor<1x192x8x8xf16>
  %relu224 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv222 : tensor<1x192x8x8xf16>)
    outs(%empty223 : tensor<1x192x8x8xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x8x8xf16>

  // conv42: 1x1 s1 p0 192->1152 8x8
  %init225 = tensor.empty() : tensor<1x1152x8x8xf16>
  %fill226 = linalg.fill ins(%cst : f16) outs(%init225 : tensor<1x1152x8x8xf16>) -> tensor<1x1152x8x8xf16>
  %conv227 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu224, %w42 : tensor<1x192x8x8xf16>, tensor<1152x192x1x1xf16>)
    outs(%fill226 : tensor<1x1152x8x8xf16>) -> tensor<1x1152x8x8xf16>
  %empty228 = tensor.empty() : tensor<1x1152x8x8xf16>
  %relu229 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv227 : tensor<1x1152x8x8xf16>)
    outs(%empty228 : tensor<1x1152x8x8xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1152x8x8xf16>

  // conv43: 3x3 s1 p1 1152->1152 8x8
  %pad230 = tensor.pad %relu229 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1152x8x8xf16> to tensor<1x1152x10x10xf16>
  %init231 = tensor.empty() : tensor<1x1152x8x8xf16>
  %fill232 = linalg.fill ins(%cst : f16) outs(%init231 : tensor<1x1152x8x8xf16>) -> tensor<1x1152x8x8xf16>
  %conv233 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad230, %w43 : tensor<1x1152x10x10xf16>, tensor<1152x1152x3x3xf16>)
    outs(%fill232 : tensor<1x1152x8x8xf16>) -> tensor<1x1152x8x8xf16>
  %empty234 = tensor.empty() : tensor<1x1152x8x8xf16>
  %relu235 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv233 : tensor<1x1152x8x8xf16>)
    outs(%empty234 : tensor<1x1152x8x8xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1152x8x8xf16>

  // conv44: 1x1 s1 p0 1152->192 8x8
  %init236 = tensor.empty() : tensor<1x192x8x8xf16>
  %fill237 = linalg.fill ins(%cst : f16) outs(%init236 : tensor<1x192x8x8xf16>) -> tensor<1x192x8x8xf16>
  %conv238 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu235, %w44 : tensor<1x1152x8x8xf16>, tensor<192x1152x1x1xf16>)
    outs(%fill237 : tensor<1x192x8x8xf16>) -> tensor<1x192x8x8xf16>
  %empty239 = tensor.empty() : tensor<1x192x8x8xf16>
  %relu240 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv238 : tensor<1x192x8x8xf16>)
    outs(%empty239 : tensor<1x192x8x8xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x8x8xf16>

  // conv45: 1x1 s1 p0 192->1152 8x8
  %init241 = tensor.empty() : tensor<1x1152x8x8xf16>
  %fill242 = linalg.fill ins(%cst : f16) outs(%init241 : tensor<1x1152x8x8xf16>) -> tensor<1x1152x8x8xf16>
  %conv243 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu240, %w45 : tensor<1x192x8x8xf16>, tensor<1152x192x1x1xf16>)
    outs(%fill242 : tensor<1x1152x8x8xf16>) -> tensor<1x1152x8x8xf16>
  %empty244 = tensor.empty() : tensor<1x1152x8x8xf16>
  %relu245 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv243 : tensor<1x1152x8x8xf16>)
    outs(%empty244 : tensor<1x1152x8x8xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1152x8x8xf16>

  // conv46: 3x3 s1 p1 1152->1152 8x8
  %pad246 = tensor.pad %relu245 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1152x8x8xf16> to tensor<1x1152x10x10xf16>
  %init247 = tensor.empty() : tensor<1x1152x8x8xf16>
  %fill248 = linalg.fill ins(%cst : f16) outs(%init247 : tensor<1x1152x8x8xf16>) -> tensor<1x1152x8x8xf16>
  %conv249 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad246, %w46 : tensor<1x1152x10x10xf16>, tensor<1152x1152x3x3xf16>)
    outs(%fill248 : tensor<1x1152x8x8xf16>) -> tensor<1x1152x8x8xf16>
  %empty250 = tensor.empty() : tensor<1x1152x8x8xf16>
  %relu251 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv249 : tensor<1x1152x8x8xf16>)
    outs(%empty250 : tensor<1x1152x8x8xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1152x8x8xf16>

  // conv47: 1x1 s1 p0 1152->320 8x8
  %init252 = tensor.empty() : tensor<1x320x8x8xf16>
  %fill253 = linalg.fill ins(%cst : f16) outs(%init252 : tensor<1x320x8x8xf16>) -> tensor<1x320x8x8xf16>
  %conv254 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu251, %w47 : tensor<1x1152x8x8xf16>, tensor<320x1152x1x1xf16>)
    outs(%fill253 : tensor<1x320x8x8xf16>) -> tensor<1x320x8x8xf16>
  %empty255 = tensor.empty() : tensor<1x320x8x8xf16>
  %relu256 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv254 : tensor<1x320x8x8xf16>)
    outs(%empty255 : tensor<1x320x8x8xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x320x8x8xf16>

  // conv48: 1x1 s1 p0 320->1280 8x8
  %init257 = tensor.empty() : tensor<1x1280x8x8xf16>
  %fill258 = linalg.fill ins(%cst : f16) outs(%init257 : tensor<1x1280x8x8xf16>) -> tensor<1x1280x8x8xf16>
  %conv259 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu256, %w48 : tensor<1x320x8x8xf16>, tensor<1280x320x1x1xf16>)
    outs(%fill258 : tensor<1x1280x8x8xf16>) -> tensor<1x1280x8x8xf16>
  %empty260 = tensor.empty() : tensor<1x1280x8x8xf16>
  %relu261 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv259 : tensor<1x1280x8x8xf16>)
    outs(%empty260 : tensor<1x1280x8x8xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1280x8x8xf16>

  // FC as 1x1 conv: 1280->1000
  %init262 = tensor.empty() : tensor<1x1000x8x8xf16>
  %fill263 = linalg.fill ins(%cst : f16) outs(%init262 : tensor<1x1000x8x8xf16>) -> tensor<1x1000x8x8xf16>
  %conv264 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu261, %w_fc : tensor<1x1280x8x8xf16>, tensor<1000x1280x1x1xf16>)
    outs(%fill263 : tensor<1x1000x8x8xf16>) -> tensor<1x1000x8x8xf16>
  return %conv264 : tensor<1x1000x8x8xf16>
}
