func.func @densenet121(
    %input: tensor<1x3x224x224xf16>,
    %w_stem: tensor<64x3x7x7xf16>,
    %w_pool: tensor<32x64x3x3xf16>,
    %w_d0_l0_bn: tensor<128x32x1x1xf16>,
    %w_d0_l0_conv: tensor<32x128x3x3xf16>,
    %w_d0_l1_bn: tensor<128x32x1x1xf16>,
    %w_d0_l1_conv: tensor<32x128x3x3xf16>,
    %w_d0_l2_bn: tensor<128x32x1x1xf16>,
    %w_d0_l2_conv: tensor<32x128x3x3xf16>,
    %w_d0_l3_bn: tensor<128x32x1x1xf16>,
    %w_d0_l3_conv: tensor<32x128x3x3xf16>,
    %w_d0_l4_bn: tensor<128x32x1x1xf16>,
    %w_d0_l4_conv: tensor<32x128x3x3xf16>,
    %w_d0_l5_bn: tensor<128x32x1x1xf16>,
    %w_d0_l5_conv: tensor<32x128x3x3xf16>,
    %w_t0_conv: tensor<32x32x1x1xf16>,
    %w_t0_pool: tensor<32x32x3x3xf16>,
    %w_d1_l0_bn: tensor<128x32x1x1xf16>,
    %w_d1_l0_conv: tensor<32x128x3x3xf16>,
    %w_d1_l1_bn: tensor<128x32x1x1xf16>,
    %w_d1_l1_conv: tensor<32x128x3x3xf16>,
    %w_d1_l2_bn: tensor<128x32x1x1xf16>,
    %w_d1_l2_conv: tensor<32x128x3x3xf16>,
    %w_d1_l3_bn: tensor<128x32x1x1xf16>,
    %w_d1_l3_conv: tensor<32x128x3x3xf16>,
    %w_d1_l4_bn: tensor<128x32x1x1xf16>,
    %w_d1_l4_conv: tensor<32x128x3x3xf16>,
    %w_d1_l5_bn: tensor<128x32x1x1xf16>,
    %w_d1_l5_conv: tensor<32x128x3x3xf16>,
    %w_d1_l6_bn: tensor<128x32x1x1xf16>,
    %w_d1_l6_conv: tensor<32x128x3x3xf16>,
    %w_d1_l7_bn: tensor<128x32x1x1xf16>,
    %w_d1_l7_conv: tensor<32x128x3x3xf16>,
    %w_d1_l8_bn: tensor<128x32x1x1xf16>,
    %w_d1_l8_conv: tensor<32x128x3x3xf16>,
    %w_d1_l9_bn: tensor<128x32x1x1xf16>,
    %w_d1_l9_conv: tensor<32x128x3x3xf16>,
    %w_d1_l10_bn: tensor<128x32x1x1xf16>,
    %w_d1_l10_conv: tensor<32x128x3x3xf16>,
    %w_d1_l11_bn: tensor<128x32x1x1xf16>,
    %w_d1_l11_conv: tensor<32x128x3x3xf16>,
    %w_t1_conv: tensor<32x32x1x1xf16>,
    %w_t1_pool: tensor<32x32x3x3xf16>,
    %w_d2_l0_bn: tensor<128x32x1x1xf16>,
    %w_d2_l0_conv: tensor<32x128x3x3xf16>,
    %w_d2_l1_bn: tensor<128x32x1x1xf16>,
    %w_d2_l1_conv: tensor<32x128x3x3xf16>,
    %w_d2_l2_bn: tensor<128x32x1x1xf16>,
    %w_d2_l2_conv: tensor<32x128x3x3xf16>,
    %w_d2_l3_bn: tensor<128x32x1x1xf16>,
    %w_d2_l3_conv: tensor<32x128x3x3xf16>,
    %w_d2_l4_bn: tensor<128x32x1x1xf16>,
    %w_d2_l4_conv: tensor<32x128x3x3xf16>,
    %w_d2_l5_bn: tensor<128x32x1x1xf16>,
    %w_d2_l5_conv: tensor<32x128x3x3xf16>,
    %w_d2_l6_bn: tensor<128x32x1x1xf16>,
    %w_d2_l6_conv: tensor<32x128x3x3xf16>,
    %w_d2_l7_bn: tensor<128x32x1x1xf16>,
    %w_d2_l7_conv: tensor<32x128x3x3xf16>,
    %w_d2_l8_bn: tensor<128x32x1x1xf16>,
    %w_d2_l8_conv: tensor<32x128x3x3xf16>,
    %w_d2_l9_bn: tensor<128x32x1x1xf16>,
    %w_d2_l9_conv: tensor<32x128x3x3xf16>,
    %w_d2_l10_bn: tensor<128x32x1x1xf16>,
    %w_d2_l10_conv: tensor<32x128x3x3xf16>,
    %w_d2_l11_bn: tensor<128x32x1x1xf16>,
    %w_d2_l11_conv: tensor<32x128x3x3xf16>,
    %w_d2_l12_bn: tensor<128x32x1x1xf16>,
    %w_d2_l12_conv: tensor<32x128x3x3xf16>,
    %w_d2_l13_bn: tensor<128x32x1x1xf16>,
    %w_d2_l13_conv: tensor<32x128x3x3xf16>,
    %w_d2_l14_bn: tensor<128x32x1x1xf16>,
    %w_d2_l14_conv: tensor<32x128x3x3xf16>,
    %w_d2_l15_bn: tensor<128x32x1x1xf16>,
    %w_d2_l15_conv: tensor<32x128x3x3xf16>,
    %w_d2_l16_bn: tensor<128x32x1x1xf16>,
    %w_d2_l16_conv: tensor<32x128x3x3xf16>,
    %w_d2_l17_bn: tensor<128x32x1x1xf16>,
    %w_d2_l17_conv: tensor<32x128x3x3xf16>,
    %w_d2_l18_bn: tensor<128x32x1x1xf16>,
    %w_d2_l18_conv: tensor<32x128x3x3xf16>,
    %w_d2_l19_bn: tensor<128x32x1x1xf16>,
    %w_d2_l19_conv: tensor<32x128x3x3xf16>,
    %w_d2_l20_bn: tensor<128x32x1x1xf16>,
    %w_d2_l20_conv: tensor<32x128x3x3xf16>,
    %w_d2_l21_bn: tensor<128x32x1x1xf16>,
    %w_d2_l21_conv: tensor<32x128x3x3xf16>,
    %w_d2_l22_bn: tensor<128x32x1x1xf16>,
    %w_d2_l22_conv: tensor<32x128x3x3xf16>,
    %w_d2_l23_bn: tensor<128x32x1x1xf16>,
    %w_d2_l23_conv: tensor<32x128x3x3xf16>,
    %w_t2_conv: tensor<32x32x1x1xf16>,
    %w_t2_pool: tensor<32x32x3x3xf16>,
    %w_d3_l0_bn: tensor<128x32x1x1xf16>,
    %w_d3_l0_conv: tensor<32x128x3x3xf16>,
    %w_d3_l1_bn: tensor<128x32x1x1xf16>,
    %w_d3_l1_conv: tensor<32x128x3x3xf16>,
    %w_d3_l2_bn: tensor<128x32x1x1xf16>,
    %w_d3_l2_conv: tensor<32x128x3x3xf16>,
    %w_d3_l3_bn: tensor<128x32x1x1xf16>,
    %w_d3_l3_conv: tensor<32x128x3x3xf16>,
    %w_d3_l4_bn: tensor<128x32x1x1xf16>,
    %w_d3_l4_conv: tensor<32x128x3x3xf16>,
    %w_d3_l5_bn: tensor<128x32x1x1xf16>,
    %w_d3_l5_conv: tensor<32x128x3x3xf16>,
    %w_d3_l6_bn: tensor<128x32x1x1xf16>,
    %w_d3_l6_conv: tensor<32x128x3x3xf16>,
    %w_d3_l7_bn: tensor<128x32x1x1xf16>,
    %w_d3_l7_conv: tensor<32x128x3x3xf16>,
    %w_d3_l8_bn: tensor<128x32x1x1xf16>,
    %w_d3_l8_conv: tensor<32x128x3x3xf16>,
    %w_d3_l9_bn: tensor<128x32x1x1xf16>,
    %w_d3_l9_conv: tensor<32x128x3x3xf16>,
    %w_d3_l10_bn: tensor<128x32x1x1xf16>,
    %w_d3_l10_conv: tensor<32x128x3x3xf16>,
    %w_d3_l11_bn: tensor<128x32x1x1xf16>,
    %w_d3_l11_conv: tensor<32x128x3x3xf16>,
    %w_d3_l12_bn: tensor<128x32x1x1xf16>,
    %w_d3_l12_conv: tensor<32x128x3x3xf16>,
    %w_d3_l13_bn: tensor<128x32x1x1xf16>,
    %w_d3_l13_conv: tensor<32x128x3x3xf16>,
    %w_d3_l14_bn: tensor<128x32x1x1xf16>,
    %w_d3_l14_conv: tensor<32x128x3x3xf16>,
    %w_d3_l15_bn: tensor<128x32x1x1xf16>,
    %w_d3_l15_conv: tensor<32x128x3x3xf16>,
    %w_fc: tensor<1000x32x1x1xf16>) -> tensor<1x1000x7x7xf16> {
  %cst = arith.constant 0.0 : f16

  // Stem: 7x7 conv stride 2, 3->64
  %pad0 = tensor.pad %input low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x224x224xf16> to tensor<1x3x230x230xf16>
  %init1 = tensor.empty() : tensor<1x64x112x112xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad0, %w_stem : tensor<1x3x230x230xf16>, tensor<64x3x7x7xf16>)
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

  // Pool: stride-2 3x3 conv, 64->32
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x112x112xf16> to tensor<1x64x114x114xf16>
  %init7 = tensor.empty() : tensor<1x32x56x56xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x32x56x56xf16>) -> tensor<1x32x56x56xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad6, %w_pool : tensor<1x64x114x114xf16>, tensor<32x64x3x3xf16>)
    outs(%fill8 : tensor<1x32x56x56xf16>) -> tensor<1x32x56x56xf16>
  %empty10 = tensor.empty() : tensor<1x32x56x56xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x32x56x56xf16>)
    outs(%empty10 : tensor<1x32x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x56x56xf16>

  // === Dense Block 0: 6 layers ===
  %init12 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill13 = linalg.fill ins(%cst : f16) outs(%init12 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv14 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu11, %w_d0_l0_bn : tensor<1x32x56x56xf16>, tensor<128x32x1x1xf16>)
    outs(%fill13 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %empty15 = tensor.empty() : tensor<1x128x56x56xf16>
  %relu16 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv14 : tensor<1x128x56x56xf16>)
    outs(%empty15 : tensor<1x128x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x56x56xf16>
  %pad17 = tensor.pad %relu16 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x56x56xf16> to tensor<1x128x58x58xf16>
  %init18 = tensor.empty() : tensor<1x32x56x56xf16>
  %fill19 = linalg.fill ins(%cst : f16) outs(%init18 : tensor<1x32x56x56xf16>) -> tensor<1x32x56x56xf16>
  %conv20 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad17, %w_d0_l0_conv : tensor<1x128x58x58xf16>, tensor<32x128x3x3xf16>)
    outs(%fill19 : tensor<1x32x56x56xf16>) -> tensor<1x32x56x56xf16>
  %empty21 = tensor.empty() : tensor<1x32x56x56xf16>
  %relu22 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv20 : tensor<1x32x56x56xf16>)
    outs(%empty21 : tensor<1x32x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x56x56xf16>
  %init23 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill24 = linalg.fill ins(%cst : f16) outs(%init23 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv25 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu22, %w_d0_l1_bn : tensor<1x32x56x56xf16>, tensor<128x32x1x1xf16>)
    outs(%fill24 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %empty26 = tensor.empty() : tensor<1x128x56x56xf16>
  %relu27 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv25 : tensor<1x128x56x56xf16>)
    outs(%empty26 : tensor<1x128x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x56x56xf16>
  %pad28 = tensor.pad %relu27 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x56x56xf16> to tensor<1x128x58x58xf16>
  %init29 = tensor.empty() : tensor<1x32x56x56xf16>
  %fill30 = linalg.fill ins(%cst : f16) outs(%init29 : tensor<1x32x56x56xf16>) -> tensor<1x32x56x56xf16>
  %conv31 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad28, %w_d0_l1_conv : tensor<1x128x58x58xf16>, tensor<32x128x3x3xf16>)
    outs(%fill30 : tensor<1x32x56x56xf16>) -> tensor<1x32x56x56xf16>
  %empty32 = tensor.empty() : tensor<1x32x56x56xf16>
  %relu33 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv31 : tensor<1x32x56x56xf16>)
    outs(%empty32 : tensor<1x32x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x56x56xf16>
  %init34 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill35 = linalg.fill ins(%cst : f16) outs(%init34 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv36 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu33, %w_d0_l2_bn : tensor<1x32x56x56xf16>, tensor<128x32x1x1xf16>)
    outs(%fill35 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %empty37 = tensor.empty() : tensor<1x128x56x56xf16>
  %relu38 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv36 : tensor<1x128x56x56xf16>)
    outs(%empty37 : tensor<1x128x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x56x56xf16>
  %pad39 = tensor.pad %relu38 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x56x56xf16> to tensor<1x128x58x58xf16>
  %init40 = tensor.empty() : tensor<1x32x56x56xf16>
  %fill41 = linalg.fill ins(%cst : f16) outs(%init40 : tensor<1x32x56x56xf16>) -> tensor<1x32x56x56xf16>
  %conv42 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad39, %w_d0_l2_conv : tensor<1x128x58x58xf16>, tensor<32x128x3x3xf16>)
    outs(%fill41 : tensor<1x32x56x56xf16>) -> tensor<1x32x56x56xf16>
  %empty43 = tensor.empty() : tensor<1x32x56x56xf16>
  %relu44 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv42 : tensor<1x32x56x56xf16>)
    outs(%empty43 : tensor<1x32x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x56x56xf16>
  %init45 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill46 = linalg.fill ins(%cst : f16) outs(%init45 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv47 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu44, %w_d0_l3_bn : tensor<1x32x56x56xf16>, tensor<128x32x1x1xf16>)
    outs(%fill46 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %empty48 = tensor.empty() : tensor<1x128x56x56xf16>
  %relu49 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv47 : tensor<1x128x56x56xf16>)
    outs(%empty48 : tensor<1x128x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x56x56xf16>
  %pad50 = tensor.pad %relu49 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x56x56xf16> to tensor<1x128x58x58xf16>
  %init51 = tensor.empty() : tensor<1x32x56x56xf16>
  %fill52 = linalg.fill ins(%cst : f16) outs(%init51 : tensor<1x32x56x56xf16>) -> tensor<1x32x56x56xf16>
  %conv53 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad50, %w_d0_l3_conv : tensor<1x128x58x58xf16>, tensor<32x128x3x3xf16>)
    outs(%fill52 : tensor<1x32x56x56xf16>) -> tensor<1x32x56x56xf16>
  %empty54 = tensor.empty() : tensor<1x32x56x56xf16>
  %relu55 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv53 : tensor<1x32x56x56xf16>)
    outs(%empty54 : tensor<1x32x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x56x56xf16>
  %init56 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill57 = linalg.fill ins(%cst : f16) outs(%init56 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv58 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu55, %w_d0_l4_bn : tensor<1x32x56x56xf16>, tensor<128x32x1x1xf16>)
    outs(%fill57 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %empty59 = tensor.empty() : tensor<1x128x56x56xf16>
  %relu60 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv58 : tensor<1x128x56x56xf16>)
    outs(%empty59 : tensor<1x128x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x56x56xf16>
  %pad61 = tensor.pad %relu60 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x56x56xf16> to tensor<1x128x58x58xf16>
  %init62 = tensor.empty() : tensor<1x32x56x56xf16>
  %fill63 = linalg.fill ins(%cst : f16) outs(%init62 : tensor<1x32x56x56xf16>) -> tensor<1x32x56x56xf16>
  %conv64 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad61, %w_d0_l4_conv : tensor<1x128x58x58xf16>, tensor<32x128x3x3xf16>)
    outs(%fill63 : tensor<1x32x56x56xf16>) -> tensor<1x32x56x56xf16>
  %empty65 = tensor.empty() : tensor<1x32x56x56xf16>
  %relu66 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv64 : tensor<1x32x56x56xf16>)
    outs(%empty65 : tensor<1x32x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x56x56xf16>
  %init67 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill68 = linalg.fill ins(%cst : f16) outs(%init67 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv69 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu66, %w_d0_l5_bn : tensor<1x32x56x56xf16>, tensor<128x32x1x1xf16>)
    outs(%fill68 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %empty70 = tensor.empty() : tensor<1x128x56x56xf16>
  %relu71 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv69 : tensor<1x128x56x56xf16>)
    outs(%empty70 : tensor<1x128x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x56x56xf16>
  %pad72 = tensor.pad %relu71 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x56x56xf16> to tensor<1x128x58x58xf16>
  %init73 = tensor.empty() : tensor<1x32x56x56xf16>
  %fill74 = linalg.fill ins(%cst : f16) outs(%init73 : tensor<1x32x56x56xf16>) -> tensor<1x32x56x56xf16>
  %conv75 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad72, %w_d0_l5_conv : tensor<1x128x58x58xf16>, tensor<32x128x3x3xf16>)
    outs(%fill74 : tensor<1x32x56x56xf16>) -> tensor<1x32x56x56xf16>
  %empty76 = tensor.empty() : tensor<1x32x56x56xf16>
  %relu77 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv75 : tensor<1x32x56x56xf16>)
    outs(%empty76 : tensor<1x32x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x56x56xf16>

  // Transition 0: 1x1 conv + stride-2 pool
  %init78 = tensor.empty() : tensor<1x32x56x56xf16>
  %fill79 = linalg.fill ins(%cst : f16) outs(%init78 : tensor<1x32x56x56xf16>) -> tensor<1x32x56x56xf16>
  %conv80 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu77, %w_t0_conv : tensor<1x32x56x56xf16>, tensor<32x32x1x1xf16>)
    outs(%fill79 : tensor<1x32x56x56xf16>) -> tensor<1x32x56x56xf16>
  %empty81 = tensor.empty() : tensor<1x32x56x56xf16>
  %relu82 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv80 : tensor<1x32x56x56xf16>)
    outs(%empty81 : tensor<1x32x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x56x56xf16>
  %pad83 = tensor.pad %relu82 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x32x56x56xf16> to tensor<1x32x58x58xf16>
  %init84 = tensor.empty() : tensor<1x32x28x28xf16>
  %fill85 = linalg.fill ins(%cst : f16) outs(%init84 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %conv86 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad83, %w_t0_pool : tensor<1x32x58x58xf16>, tensor<32x32x3x3xf16>)
    outs(%fill85 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %empty87 = tensor.empty() : tensor<1x32x28x28xf16>
  %relu88 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv86 : tensor<1x32x28x28xf16>)
    outs(%empty87 : tensor<1x32x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x28x28xf16>

  // === Dense Block 1: 12 layers ===
  %init89 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill90 = linalg.fill ins(%cst : f16) outs(%init89 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv91 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu88, %w_d1_l0_bn : tensor<1x32x28x28xf16>, tensor<128x32x1x1xf16>)
    outs(%fill90 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty92 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu93 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv91 : tensor<1x128x28x28xf16>)
    outs(%empty92 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad94 = tensor.pad %relu93 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init95 = tensor.empty() : tensor<1x32x28x28xf16>
  %fill96 = linalg.fill ins(%cst : f16) outs(%init95 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %conv97 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad94, %w_d1_l0_conv : tensor<1x128x30x30xf16>, tensor<32x128x3x3xf16>)
    outs(%fill96 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %empty98 = tensor.empty() : tensor<1x32x28x28xf16>
  %relu99 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv97 : tensor<1x32x28x28xf16>)
    outs(%empty98 : tensor<1x32x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x28x28xf16>
  %init100 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill101 = linalg.fill ins(%cst : f16) outs(%init100 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv102 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu99, %w_d1_l1_bn : tensor<1x32x28x28xf16>, tensor<128x32x1x1xf16>)
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
  %init106 = tensor.empty() : tensor<1x32x28x28xf16>
  %fill107 = linalg.fill ins(%cst : f16) outs(%init106 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %conv108 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad105, %w_d1_l1_conv : tensor<1x128x30x30xf16>, tensor<32x128x3x3xf16>)
    outs(%fill107 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %empty109 = tensor.empty() : tensor<1x32x28x28xf16>
  %relu110 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv108 : tensor<1x32x28x28xf16>)
    outs(%empty109 : tensor<1x32x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x28x28xf16>
  %init111 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill112 = linalg.fill ins(%cst : f16) outs(%init111 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv113 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu110, %w_d1_l2_bn : tensor<1x32x28x28xf16>, tensor<128x32x1x1xf16>)
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
  %init117 = tensor.empty() : tensor<1x32x28x28xf16>
  %fill118 = linalg.fill ins(%cst : f16) outs(%init117 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %conv119 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad116, %w_d1_l2_conv : tensor<1x128x30x30xf16>, tensor<32x128x3x3xf16>)
    outs(%fill118 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %empty120 = tensor.empty() : tensor<1x32x28x28xf16>
  %relu121 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv119 : tensor<1x32x28x28xf16>)
    outs(%empty120 : tensor<1x32x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x28x28xf16>
  %init122 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill123 = linalg.fill ins(%cst : f16) outs(%init122 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv124 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu121, %w_d1_l3_bn : tensor<1x32x28x28xf16>, tensor<128x32x1x1xf16>)
    outs(%fill123 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty125 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu126 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv124 : tensor<1x128x28x28xf16>)
    outs(%empty125 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad127 = tensor.pad %relu126 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init128 = tensor.empty() : tensor<1x32x28x28xf16>
  %fill129 = linalg.fill ins(%cst : f16) outs(%init128 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %conv130 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad127, %w_d1_l3_conv : tensor<1x128x30x30xf16>, tensor<32x128x3x3xf16>)
    outs(%fill129 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %empty131 = tensor.empty() : tensor<1x32x28x28xf16>
  %relu132 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv130 : tensor<1x32x28x28xf16>)
    outs(%empty131 : tensor<1x32x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x28x28xf16>
  %init133 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill134 = linalg.fill ins(%cst : f16) outs(%init133 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv135 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu132, %w_d1_l4_bn : tensor<1x32x28x28xf16>, tensor<128x32x1x1xf16>)
    outs(%fill134 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty136 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu137 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv135 : tensor<1x128x28x28xf16>)
    outs(%empty136 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad138 = tensor.pad %relu137 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init139 = tensor.empty() : tensor<1x32x28x28xf16>
  %fill140 = linalg.fill ins(%cst : f16) outs(%init139 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %conv141 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad138, %w_d1_l4_conv : tensor<1x128x30x30xf16>, tensor<32x128x3x3xf16>)
    outs(%fill140 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %empty142 = tensor.empty() : tensor<1x32x28x28xf16>
  %relu143 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv141 : tensor<1x32x28x28xf16>)
    outs(%empty142 : tensor<1x32x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x28x28xf16>
  %init144 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill145 = linalg.fill ins(%cst : f16) outs(%init144 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv146 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu143, %w_d1_l5_bn : tensor<1x32x28x28xf16>, tensor<128x32x1x1xf16>)
    outs(%fill145 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty147 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu148 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv146 : tensor<1x128x28x28xf16>)
    outs(%empty147 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad149 = tensor.pad %relu148 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init150 = tensor.empty() : tensor<1x32x28x28xf16>
  %fill151 = linalg.fill ins(%cst : f16) outs(%init150 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %conv152 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad149, %w_d1_l5_conv : tensor<1x128x30x30xf16>, tensor<32x128x3x3xf16>)
    outs(%fill151 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %empty153 = tensor.empty() : tensor<1x32x28x28xf16>
  %relu154 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv152 : tensor<1x32x28x28xf16>)
    outs(%empty153 : tensor<1x32x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x28x28xf16>
  %init155 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill156 = linalg.fill ins(%cst : f16) outs(%init155 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv157 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu154, %w_d1_l6_bn : tensor<1x32x28x28xf16>, tensor<128x32x1x1xf16>)
    outs(%fill156 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty158 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu159 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv157 : tensor<1x128x28x28xf16>)
    outs(%empty158 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad160 = tensor.pad %relu159 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init161 = tensor.empty() : tensor<1x32x28x28xf16>
  %fill162 = linalg.fill ins(%cst : f16) outs(%init161 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %conv163 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad160, %w_d1_l6_conv : tensor<1x128x30x30xf16>, tensor<32x128x3x3xf16>)
    outs(%fill162 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %empty164 = tensor.empty() : tensor<1x32x28x28xf16>
  %relu165 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv163 : tensor<1x32x28x28xf16>)
    outs(%empty164 : tensor<1x32x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x28x28xf16>
  %init166 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill167 = linalg.fill ins(%cst : f16) outs(%init166 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv168 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu165, %w_d1_l7_bn : tensor<1x32x28x28xf16>, tensor<128x32x1x1xf16>)
    outs(%fill167 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty169 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu170 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv168 : tensor<1x128x28x28xf16>)
    outs(%empty169 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad171 = tensor.pad %relu170 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init172 = tensor.empty() : tensor<1x32x28x28xf16>
  %fill173 = linalg.fill ins(%cst : f16) outs(%init172 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %conv174 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad171, %w_d1_l7_conv : tensor<1x128x30x30xf16>, tensor<32x128x3x3xf16>)
    outs(%fill173 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %empty175 = tensor.empty() : tensor<1x32x28x28xf16>
  %relu176 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv174 : tensor<1x32x28x28xf16>)
    outs(%empty175 : tensor<1x32x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x28x28xf16>
  %init177 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill178 = linalg.fill ins(%cst : f16) outs(%init177 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv179 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu176, %w_d1_l8_bn : tensor<1x32x28x28xf16>, tensor<128x32x1x1xf16>)
    outs(%fill178 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty180 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu181 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv179 : tensor<1x128x28x28xf16>)
    outs(%empty180 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad182 = tensor.pad %relu181 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init183 = tensor.empty() : tensor<1x32x28x28xf16>
  %fill184 = linalg.fill ins(%cst : f16) outs(%init183 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %conv185 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad182, %w_d1_l8_conv : tensor<1x128x30x30xf16>, tensor<32x128x3x3xf16>)
    outs(%fill184 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %empty186 = tensor.empty() : tensor<1x32x28x28xf16>
  %relu187 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv185 : tensor<1x32x28x28xf16>)
    outs(%empty186 : tensor<1x32x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x28x28xf16>
  %init188 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill189 = linalg.fill ins(%cst : f16) outs(%init188 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv190 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu187, %w_d1_l9_bn : tensor<1x32x28x28xf16>, tensor<128x32x1x1xf16>)
    outs(%fill189 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty191 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu192 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv190 : tensor<1x128x28x28xf16>)
    outs(%empty191 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad193 = tensor.pad %relu192 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init194 = tensor.empty() : tensor<1x32x28x28xf16>
  %fill195 = linalg.fill ins(%cst : f16) outs(%init194 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %conv196 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad193, %w_d1_l9_conv : tensor<1x128x30x30xf16>, tensor<32x128x3x3xf16>)
    outs(%fill195 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %empty197 = tensor.empty() : tensor<1x32x28x28xf16>
  %relu198 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv196 : tensor<1x32x28x28xf16>)
    outs(%empty197 : tensor<1x32x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x28x28xf16>
  %init199 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill200 = linalg.fill ins(%cst : f16) outs(%init199 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv201 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu198, %w_d1_l10_bn : tensor<1x32x28x28xf16>, tensor<128x32x1x1xf16>)
    outs(%fill200 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty202 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu203 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv201 : tensor<1x128x28x28xf16>)
    outs(%empty202 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad204 = tensor.pad %relu203 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init205 = tensor.empty() : tensor<1x32x28x28xf16>
  %fill206 = linalg.fill ins(%cst : f16) outs(%init205 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %conv207 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad204, %w_d1_l10_conv : tensor<1x128x30x30xf16>, tensor<32x128x3x3xf16>)
    outs(%fill206 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %empty208 = tensor.empty() : tensor<1x32x28x28xf16>
  %relu209 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv207 : tensor<1x32x28x28xf16>)
    outs(%empty208 : tensor<1x32x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x28x28xf16>
  %init210 = tensor.empty() : tensor<1x128x28x28xf16>
  %fill211 = linalg.fill ins(%cst : f16) outs(%init210 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %conv212 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu209, %w_d1_l11_bn : tensor<1x32x28x28xf16>, tensor<128x32x1x1xf16>)
    outs(%fill211 : tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
  %empty213 = tensor.empty() : tensor<1x128x28x28xf16>
  %relu214 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv212 : tensor<1x128x28x28xf16>)
    outs(%empty213 : tensor<1x128x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x28x28xf16>
  %pad215 = tensor.pad %relu214 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x28x28xf16> to tensor<1x128x30x30xf16>
  %init216 = tensor.empty() : tensor<1x32x28x28xf16>
  %fill217 = linalg.fill ins(%cst : f16) outs(%init216 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %conv218 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad215, %w_d1_l11_conv : tensor<1x128x30x30xf16>, tensor<32x128x3x3xf16>)
    outs(%fill217 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %empty219 = tensor.empty() : tensor<1x32x28x28xf16>
  %relu220 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv218 : tensor<1x32x28x28xf16>)
    outs(%empty219 : tensor<1x32x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x28x28xf16>

  // Transition 1: 1x1 conv + stride-2 pool
  %init221 = tensor.empty() : tensor<1x32x28x28xf16>
  %fill222 = linalg.fill ins(%cst : f16) outs(%init221 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %conv223 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu220, %w_t1_conv : tensor<1x32x28x28xf16>, tensor<32x32x1x1xf16>)
    outs(%fill222 : tensor<1x32x28x28xf16>) -> tensor<1x32x28x28xf16>
  %empty224 = tensor.empty() : tensor<1x32x28x28xf16>
  %relu225 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv223 : tensor<1x32x28x28xf16>)
    outs(%empty224 : tensor<1x32x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x28x28xf16>
  %pad226 = tensor.pad %relu225 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x32x28x28xf16> to tensor<1x32x30x30xf16>
  %init227 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill228 = linalg.fill ins(%cst : f16) outs(%init227 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv229 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad226, %w_t1_pool : tensor<1x32x30x30xf16>, tensor<32x32x3x3xf16>)
    outs(%fill228 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty230 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu231 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv229 : tensor<1x32x14x14xf16>)
    outs(%empty230 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>

  // === Dense Block 2: 24 layers ===
  %init232 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill233 = linalg.fill ins(%cst : f16) outs(%init232 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv234 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu231, %w_d2_l0_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill233 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty235 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu236 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv234 : tensor<1x128x14x14xf16>)
    outs(%empty235 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad237 = tensor.pad %relu236 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init238 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill239 = linalg.fill ins(%cst : f16) outs(%init238 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv240 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad237, %w_d2_l0_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill239 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty241 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu242 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv240 : tensor<1x32x14x14xf16>)
    outs(%empty241 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init243 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill244 = linalg.fill ins(%cst : f16) outs(%init243 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv245 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu242, %w_d2_l1_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill244 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty246 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu247 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv245 : tensor<1x128x14x14xf16>)
    outs(%empty246 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad248 = tensor.pad %relu247 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init249 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill250 = linalg.fill ins(%cst : f16) outs(%init249 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv251 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad248, %w_d2_l1_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill250 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty252 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu253 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv251 : tensor<1x32x14x14xf16>)
    outs(%empty252 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init254 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill255 = linalg.fill ins(%cst : f16) outs(%init254 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv256 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu253, %w_d2_l2_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill255 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty257 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu258 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv256 : tensor<1x128x14x14xf16>)
    outs(%empty257 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad259 = tensor.pad %relu258 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init260 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill261 = linalg.fill ins(%cst : f16) outs(%init260 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv262 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad259, %w_d2_l2_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill261 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty263 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu264 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv262 : tensor<1x32x14x14xf16>)
    outs(%empty263 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init265 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill266 = linalg.fill ins(%cst : f16) outs(%init265 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv267 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu264, %w_d2_l3_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill266 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty268 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu269 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv267 : tensor<1x128x14x14xf16>)
    outs(%empty268 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad270 = tensor.pad %relu269 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init271 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill272 = linalg.fill ins(%cst : f16) outs(%init271 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv273 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad270, %w_d2_l3_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill272 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty274 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu275 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv273 : tensor<1x32x14x14xf16>)
    outs(%empty274 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init276 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill277 = linalg.fill ins(%cst : f16) outs(%init276 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv278 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu275, %w_d2_l4_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill277 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty279 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu280 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv278 : tensor<1x128x14x14xf16>)
    outs(%empty279 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad281 = tensor.pad %relu280 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init282 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill283 = linalg.fill ins(%cst : f16) outs(%init282 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv284 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad281, %w_d2_l4_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill283 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty285 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu286 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv284 : tensor<1x32x14x14xf16>)
    outs(%empty285 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init287 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill288 = linalg.fill ins(%cst : f16) outs(%init287 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv289 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu286, %w_d2_l5_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill288 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty290 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu291 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv289 : tensor<1x128x14x14xf16>)
    outs(%empty290 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad292 = tensor.pad %relu291 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init293 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill294 = linalg.fill ins(%cst : f16) outs(%init293 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv295 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad292, %w_d2_l5_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill294 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty296 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu297 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv295 : tensor<1x32x14x14xf16>)
    outs(%empty296 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init298 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill299 = linalg.fill ins(%cst : f16) outs(%init298 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv300 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu297, %w_d2_l6_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill299 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty301 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu302 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv300 : tensor<1x128x14x14xf16>)
    outs(%empty301 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad303 = tensor.pad %relu302 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init304 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill305 = linalg.fill ins(%cst : f16) outs(%init304 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv306 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad303, %w_d2_l6_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill305 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty307 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu308 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv306 : tensor<1x32x14x14xf16>)
    outs(%empty307 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init309 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill310 = linalg.fill ins(%cst : f16) outs(%init309 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv311 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu308, %w_d2_l7_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill310 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty312 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu313 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv311 : tensor<1x128x14x14xf16>)
    outs(%empty312 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad314 = tensor.pad %relu313 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init315 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill316 = linalg.fill ins(%cst : f16) outs(%init315 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv317 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad314, %w_d2_l7_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill316 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty318 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu319 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv317 : tensor<1x32x14x14xf16>)
    outs(%empty318 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init320 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill321 = linalg.fill ins(%cst : f16) outs(%init320 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv322 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu319, %w_d2_l8_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill321 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty323 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu324 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv322 : tensor<1x128x14x14xf16>)
    outs(%empty323 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad325 = tensor.pad %relu324 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init326 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill327 = linalg.fill ins(%cst : f16) outs(%init326 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv328 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad325, %w_d2_l8_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill327 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty329 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu330 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv328 : tensor<1x32x14x14xf16>)
    outs(%empty329 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init331 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill332 = linalg.fill ins(%cst : f16) outs(%init331 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv333 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu330, %w_d2_l9_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill332 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty334 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu335 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv333 : tensor<1x128x14x14xf16>)
    outs(%empty334 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad336 = tensor.pad %relu335 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init337 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill338 = linalg.fill ins(%cst : f16) outs(%init337 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv339 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad336, %w_d2_l9_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill338 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty340 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu341 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv339 : tensor<1x32x14x14xf16>)
    outs(%empty340 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init342 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill343 = linalg.fill ins(%cst : f16) outs(%init342 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv344 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu341, %w_d2_l10_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill343 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty345 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu346 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv344 : tensor<1x128x14x14xf16>)
    outs(%empty345 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad347 = tensor.pad %relu346 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init348 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill349 = linalg.fill ins(%cst : f16) outs(%init348 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv350 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad347, %w_d2_l10_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill349 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty351 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu352 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv350 : tensor<1x32x14x14xf16>)
    outs(%empty351 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init353 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill354 = linalg.fill ins(%cst : f16) outs(%init353 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv355 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu352, %w_d2_l11_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill354 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty356 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu357 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv355 : tensor<1x128x14x14xf16>)
    outs(%empty356 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad358 = tensor.pad %relu357 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init359 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill360 = linalg.fill ins(%cst : f16) outs(%init359 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv361 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad358, %w_d2_l11_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill360 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty362 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu363 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv361 : tensor<1x32x14x14xf16>)
    outs(%empty362 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init364 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill365 = linalg.fill ins(%cst : f16) outs(%init364 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv366 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu363, %w_d2_l12_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill365 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty367 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu368 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv366 : tensor<1x128x14x14xf16>)
    outs(%empty367 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad369 = tensor.pad %relu368 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init370 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill371 = linalg.fill ins(%cst : f16) outs(%init370 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv372 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad369, %w_d2_l12_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill371 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty373 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu374 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv372 : tensor<1x32x14x14xf16>)
    outs(%empty373 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init375 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill376 = linalg.fill ins(%cst : f16) outs(%init375 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv377 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu374, %w_d2_l13_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill376 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty378 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu379 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv377 : tensor<1x128x14x14xf16>)
    outs(%empty378 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad380 = tensor.pad %relu379 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init381 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill382 = linalg.fill ins(%cst : f16) outs(%init381 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv383 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad380, %w_d2_l13_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill382 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty384 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu385 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv383 : tensor<1x32x14x14xf16>)
    outs(%empty384 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init386 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill387 = linalg.fill ins(%cst : f16) outs(%init386 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv388 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu385, %w_d2_l14_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill387 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty389 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu390 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv388 : tensor<1x128x14x14xf16>)
    outs(%empty389 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad391 = tensor.pad %relu390 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init392 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill393 = linalg.fill ins(%cst : f16) outs(%init392 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv394 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad391, %w_d2_l14_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill393 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty395 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu396 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv394 : tensor<1x32x14x14xf16>)
    outs(%empty395 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init397 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill398 = linalg.fill ins(%cst : f16) outs(%init397 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv399 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu396, %w_d2_l15_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill398 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty400 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu401 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv399 : tensor<1x128x14x14xf16>)
    outs(%empty400 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad402 = tensor.pad %relu401 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init403 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill404 = linalg.fill ins(%cst : f16) outs(%init403 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv405 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad402, %w_d2_l15_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill404 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty406 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu407 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv405 : tensor<1x32x14x14xf16>)
    outs(%empty406 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init408 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill409 = linalg.fill ins(%cst : f16) outs(%init408 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv410 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu407, %w_d2_l16_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill409 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty411 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu412 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv410 : tensor<1x128x14x14xf16>)
    outs(%empty411 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad413 = tensor.pad %relu412 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init414 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill415 = linalg.fill ins(%cst : f16) outs(%init414 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv416 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad413, %w_d2_l16_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill415 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty417 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu418 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv416 : tensor<1x32x14x14xf16>)
    outs(%empty417 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init419 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill420 = linalg.fill ins(%cst : f16) outs(%init419 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv421 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu418, %w_d2_l17_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill420 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty422 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu423 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv421 : tensor<1x128x14x14xf16>)
    outs(%empty422 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad424 = tensor.pad %relu423 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init425 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill426 = linalg.fill ins(%cst : f16) outs(%init425 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv427 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad424, %w_d2_l17_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill426 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty428 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu429 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv427 : tensor<1x32x14x14xf16>)
    outs(%empty428 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init430 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill431 = linalg.fill ins(%cst : f16) outs(%init430 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv432 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu429, %w_d2_l18_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill431 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty433 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu434 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv432 : tensor<1x128x14x14xf16>)
    outs(%empty433 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad435 = tensor.pad %relu434 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init436 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill437 = linalg.fill ins(%cst : f16) outs(%init436 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv438 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad435, %w_d2_l18_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill437 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty439 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu440 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv438 : tensor<1x32x14x14xf16>)
    outs(%empty439 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init441 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill442 = linalg.fill ins(%cst : f16) outs(%init441 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv443 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu440, %w_d2_l19_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill442 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty444 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu445 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv443 : tensor<1x128x14x14xf16>)
    outs(%empty444 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad446 = tensor.pad %relu445 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init447 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill448 = linalg.fill ins(%cst : f16) outs(%init447 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv449 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad446, %w_d2_l19_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill448 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty450 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu451 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv449 : tensor<1x32x14x14xf16>)
    outs(%empty450 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init452 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill453 = linalg.fill ins(%cst : f16) outs(%init452 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv454 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu451, %w_d2_l20_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill453 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty455 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu456 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv454 : tensor<1x128x14x14xf16>)
    outs(%empty455 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad457 = tensor.pad %relu456 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init458 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill459 = linalg.fill ins(%cst : f16) outs(%init458 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv460 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad457, %w_d2_l20_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill459 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty461 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu462 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv460 : tensor<1x32x14x14xf16>)
    outs(%empty461 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init463 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill464 = linalg.fill ins(%cst : f16) outs(%init463 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv465 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu462, %w_d2_l21_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill464 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty466 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu467 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv465 : tensor<1x128x14x14xf16>)
    outs(%empty466 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad468 = tensor.pad %relu467 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init469 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill470 = linalg.fill ins(%cst : f16) outs(%init469 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv471 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad468, %w_d2_l21_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill470 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty472 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu473 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv471 : tensor<1x32x14x14xf16>)
    outs(%empty472 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init474 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill475 = linalg.fill ins(%cst : f16) outs(%init474 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv476 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu473, %w_d2_l22_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill475 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty477 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu478 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv476 : tensor<1x128x14x14xf16>)
    outs(%empty477 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad479 = tensor.pad %relu478 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init480 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill481 = linalg.fill ins(%cst : f16) outs(%init480 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv482 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad479, %w_d2_l22_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill481 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty483 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu484 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv482 : tensor<1x32x14x14xf16>)
    outs(%empty483 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init485 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill486 = linalg.fill ins(%cst : f16) outs(%init485 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv487 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu484, %w_d2_l23_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill486 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty488 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu489 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv487 : tensor<1x128x14x14xf16>)
    outs(%empty488 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad490 = tensor.pad %relu489 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init491 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill492 = linalg.fill ins(%cst : f16) outs(%init491 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv493 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad490, %w_d2_l23_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill492 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty494 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu495 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv493 : tensor<1x32x14x14xf16>)
    outs(%empty494 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>

  // Transition 2: 1x1 conv + stride-2 pool
  %init496 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill497 = linalg.fill ins(%cst : f16) outs(%init496 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv498 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu495, %w_t2_conv : tensor<1x32x14x14xf16>, tensor<32x32x1x1xf16>)
    outs(%fill497 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty499 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu500 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv498 : tensor<1x32x14x14xf16>)
    outs(%empty499 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %pad501 = tensor.pad %relu500 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x32x14x14xf16> to tensor<1x32x16x16xf16>
  %init502 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill503 = linalg.fill ins(%cst : f16) outs(%init502 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv504 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad501, %w_t2_pool : tensor<1x32x16x16xf16>, tensor<32x32x3x3xf16>)
    outs(%fill503 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty505 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu506 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv504 : tensor<1x32x7x7xf16>)
    outs(%empty505 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>

  // === Dense Block 3: 16 layers ===
  %init507 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill508 = linalg.fill ins(%cst : f16) outs(%init507 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv509 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu506, %w_d3_l0_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill508 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty510 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu511 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv509 : tensor<1x128x7x7xf16>)
    outs(%empty510 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad512 = tensor.pad %relu511 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init513 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill514 = linalg.fill ins(%cst : f16) outs(%init513 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv515 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad512, %w_d3_l0_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill514 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty516 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu517 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv515 : tensor<1x32x7x7xf16>)
    outs(%empty516 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init518 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill519 = linalg.fill ins(%cst : f16) outs(%init518 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv520 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu517, %w_d3_l1_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill519 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty521 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu522 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv520 : tensor<1x128x7x7xf16>)
    outs(%empty521 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad523 = tensor.pad %relu522 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init524 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill525 = linalg.fill ins(%cst : f16) outs(%init524 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv526 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad523, %w_d3_l1_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill525 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty527 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu528 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv526 : tensor<1x32x7x7xf16>)
    outs(%empty527 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init529 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill530 = linalg.fill ins(%cst : f16) outs(%init529 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv531 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu528, %w_d3_l2_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill530 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty532 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu533 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv531 : tensor<1x128x7x7xf16>)
    outs(%empty532 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad534 = tensor.pad %relu533 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init535 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill536 = linalg.fill ins(%cst : f16) outs(%init535 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv537 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad534, %w_d3_l2_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill536 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty538 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu539 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv537 : tensor<1x32x7x7xf16>)
    outs(%empty538 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init540 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill541 = linalg.fill ins(%cst : f16) outs(%init540 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv542 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu539, %w_d3_l3_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill541 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty543 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu544 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv542 : tensor<1x128x7x7xf16>)
    outs(%empty543 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad545 = tensor.pad %relu544 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init546 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill547 = linalg.fill ins(%cst : f16) outs(%init546 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv548 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad545, %w_d3_l3_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill547 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty549 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu550 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv548 : tensor<1x32x7x7xf16>)
    outs(%empty549 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init551 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill552 = linalg.fill ins(%cst : f16) outs(%init551 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv553 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu550, %w_d3_l4_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill552 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty554 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu555 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv553 : tensor<1x128x7x7xf16>)
    outs(%empty554 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad556 = tensor.pad %relu555 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init557 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill558 = linalg.fill ins(%cst : f16) outs(%init557 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv559 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad556, %w_d3_l4_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill558 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty560 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu561 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv559 : tensor<1x32x7x7xf16>)
    outs(%empty560 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init562 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill563 = linalg.fill ins(%cst : f16) outs(%init562 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv564 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu561, %w_d3_l5_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill563 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty565 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu566 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv564 : tensor<1x128x7x7xf16>)
    outs(%empty565 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad567 = tensor.pad %relu566 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init568 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill569 = linalg.fill ins(%cst : f16) outs(%init568 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv570 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad567, %w_d3_l5_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill569 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty571 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu572 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv570 : tensor<1x32x7x7xf16>)
    outs(%empty571 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init573 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill574 = linalg.fill ins(%cst : f16) outs(%init573 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv575 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu572, %w_d3_l6_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill574 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty576 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu577 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv575 : tensor<1x128x7x7xf16>)
    outs(%empty576 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad578 = tensor.pad %relu577 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init579 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill580 = linalg.fill ins(%cst : f16) outs(%init579 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv581 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad578, %w_d3_l6_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill580 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty582 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu583 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv581 : tensor<1x32x7x7xf16>)
    outs(%empty582 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init584 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill585 = linalg.fill ins(%cst : f16) outs(%init584 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv586 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu583, %w_d3_l7_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill585 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty587 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu588 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv586 : tensor<1x128x7x7xf16>)
    outs(%empty587 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad589 = tensor.pad %relu588 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init590 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill591 = linalg.fill ins(%cst : f16) outs(%init590 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv592 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad589, %w_d3_l7_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill591 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty593 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu594 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv592 : tensor<1x32x7x7xf16>)
    outs(%empty593 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init595 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill596 = linalg.fill ins(%cst : f16) outs(%init595 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv597 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu594, %w_d3_l8_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill596 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty598 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu599 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv597 : tensor<1x128x7x7xf16>)
    outs(%empty598 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad600 = tensor.pad %relu599 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init601 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill602 = linalg.fill ins(%cst : f16) outs(%init601 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv603 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad600, %w_d3_l8_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill602 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty604 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu605 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv603 : tensor<1x32x7x7xf16>)
    outs(%empty604 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init606 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill607 = linalg.fill ins(%cst : f16) outs(%init606 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv608 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu605, %w_d3_l9_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill607 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty609 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu610 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv608 : tensor<1x128x7x7xf16>)
    outs(%empty609 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad611 = tensor.pad %relu610 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init612 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill613 = linalg.fill ins(%cst : f16) outs(%init612 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv614 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad611, %w_d3_l9_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill613 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty615 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu616 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv614 : tensor<1x32x7x7xf16>)
    outs(%empty615 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init617 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill618 = linalg.fill ins(%cst : f16) outs(%init617 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv619 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu616, %w_d3_l10_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill618 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty620 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu621 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv619 : tensor<1x128x7x7xf16>)
    outs(%empty620 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad622 = tensor.pad %relu621 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init623 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill624 = linalg.fill ins(%cst : f16) outs(%init623 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv625 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad622, %w_d3_l10_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill624 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty626 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu627 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv625 : tensor<1x32x7x7xf16>)
    outs(%empty626 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init628 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill629 = linalg.fill ins(%cst : f16) outs(%init628 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv630 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu627, %w_d3_l11_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill629 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty631 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu632 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv630 : tensor<1x128x7x7xf16>)
    outs(%empty631 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad633 = tensor.pad %relu632 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init634 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill635 = linalg.fill ins(%cst : f16) outs(%init634 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv636 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad633, %w_d3_l11_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill635 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty637 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu638 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv636 : tensor<1x32x7x7xf16>)
    outs(%empty637 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init639 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill640 = linalg.fill ins(%cst : f16) outs(%init639 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv641 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu638, %w_d3_l12_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill640 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty642 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu643 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv641 : tensor<1x128x7x7xf16>)
    outs(%empty642 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad644 = tensor.pad %relu643 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init645 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill646 = linalg.fill ins(%cst : f16) outs(%init645 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv647 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad644, %w_d3_l12_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill646 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty648 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu649 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv647 : tensor<1x32x7x7xf16>)
    outs(%empty648 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init650 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill651 = linalg.fill ins(%cst : f16) outs(%init650 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv652 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu649, %w_d3_l13_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill651 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty653 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu654 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv652 : tensor<1x128x7x7xf16>)
    outs(%empty653 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad655 = tensor.pad %relu654 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init656 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill657 = linalg.fill ins(%cst : f16) outs(%init656 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv658 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad655, %w_d3_l13_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill657 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty659 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu660 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv658 : tensor<1x32x7x7xf16>)
    outs(%empty659 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init661 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill662 = linalg.fill ins(%cst : f16) outs(%init661 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv663 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu660, %w_d3_l14_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill662 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty664 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu665 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv663 : tensor<1x128x7x7xf16>)
    outs(%empty664 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad666 = tensor.pad %relu665 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init667 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill668 = linalg.fill ins(%cst : f16) outs(%init667 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv669 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad666, %w_d3_l14_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill668 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty670 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu671 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv669 : tensor<1x32x7x7xf16>)
    outs(%empty670 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init672 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill673 = linalg.fill ins(%cst : f16) outs(%init672 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv674 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu671, %w_d3_l15_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill673 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty675 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu676 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv674 : tensor<1x128x7x7xf16>)
    outs(%empty675 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad677 = tensor.pad %relu676 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init678 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill679 = linalg.fill ins(%cst : f16) outs(%init678 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv680 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad677, %w_d3_l15_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill679 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty681 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu682 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv680 : tensor<1x32x7x7xf16>)
    outs(%empty681 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>

  // FC: 1x1 conv 32->1000
  %init683 = tensor.empty() : tensor<1x1000x7x7xf16>
  %fill684 = linalg.fill ins(%cst : f16) outs(%init683 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  %conv685 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu682, %w_fc : tensor<1x32x7x7xf16>, tensor<1000x32x1x1xf16>)
    outs(%fill684 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  return %conv685 : tensor<1x1000x7x7xf16>
}
