func.func @densenet261(
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
    %w_d2_l24_bn: tensor<128x32x1x1xf16>,
    %w_d2_l24_conv: tensor<32x128x3x3xf16>,
    %w_d2_l25_bn: tensor<128x32x1x1xf16>,
    %w_d2_l25_conv: tensor<32x128x3x3xf16>,
    %w_d2_l26_bn: tensor<128x32x1x1xf16>,
    %w_d2_l26_conv: tensor<32x128x3x3xf16>,
    %w_d2_l27_bn: tensor<128x32x1x1xf16>,
    %w_d2_l27_conv: tensor<32x128x3x3xf16>,
    %w_d2_l28_bn: tensor<128x32x1x1xf16>,
    %w_d2_l28_conv: tensor<32x128x3x3xf16>,
    %w_d2_l29_bn: tensor<128x32x1x1xf16>,
    %w_d2_l29_conv: tensor<32x128x3x3xf16>,
    %w_d2_l30_bn: tensor<128x32x1x1xf16>,
    %w_d2_l30_conv: tensor<32x128x3x3xf16>,
    %w_d2_l31_bn: tensor<128x32x1x1xf16>,
    %w_d2_l31_conv: tensor<32x128x3x3xf16>,
    %w_d2_l32_bn: tensor<128x32x1x1xf16>,
    %w_d2_l32_conv: tensor<32x128x3x3xf16>,
    %w_d2_l33_bn: tensor<128x32x1x1xf16>,
    %w_d2_l33_conv: tensor<32x128x3x3xf16>,
    %w_d2_l34_bn: tensor<128x32x1x1xf16>,
    %w_d2_l34_conv: tensor<32x128x3x3xf16>,
    %w_d2_l35_bn: tensor<128x32x1x1xf16>,
    %w_d2_l35_conv: tensor<32x128x3x3xf16>,
    %w_d2_l36_bn: tensor<128x32x1x1xf16>,
    %w_d2_l36_conv: tensor<32x128x3x3xf16>,
    %w_d2_l37_bn: tensor<128x32x1x1xf16>,
    %w_d2_l37_conv: tensor<32x128x3x3xf16>,
    %w_d2_l38_bn: tensor<128x32x1x1xf16>,
    %w_d2_l38_conv: tensor<32x128x3x3xf16>,
    %w_d2_l39_bn: tensor<128x32x1x1xf16>,
    %w_d2_l39_conv: tensor<32x128x3x3xf16>,
    %w_d2_l40_bn: tensor<128x32x1x1xf16>,
    %w_d2_l40_conv: tensor<32x128x3x3xf16>,
    %w_d2_l41_bn: tensor<128x32x1x1xf16>,
    %w_d2_l41_conv: tensor<32x128x3x3xf16>,
    %w_d2_l42_bn: tensor<128x32x1x1xf16>,
    %w_d2_l42_conv: tensor<32x128x3x3xf16>,
    %w_d2_l43_bn: tensor<128x32x1x1xf16>,
    %w_d2_l43_conv: tensor<32x128x3x3xf16>,
    %w_d2_l44_bn: tensor<128x32x1x1xf16>,
    %w_d2_l44_conv: tensor<32x128x3x3xf16>,
    %w_d2_l45_bn: tensor<128x32x1x1xf16>,
    %w_d2_l45_conv: tensor<32x128x3x3xf16>,
    %w_d2_l46_bn: tensor<128x32x1x1xf16>,
    %w_d2_l46_conv: tensor<32x128x3x3xf16>,
    %w_d2_l47_bn: tensor<128x32x1x1xf16>,
    %w_d2_l47_conv: tensor<32x128x3x3xf16>,
    %w_d2_l48_bn: tensor<128x32x1x1xf16>,
    %w_d2_l48_conv: tensor<32x128x3x3xf16>,
    %w_d2_l49_bn: tensor<128x32x1x1xf16>,
    %w_d2_l49_conv: tensor<32x128x3x3xf16>,
    %w_d2_l50_bn: tensor<128x32x1x1xf16>,
    %w_d2_l50_conv: tensor<32x128x3x3xf16>,
    %w_d2_l51_bn: tensor<128x32x1x1xf16>,
    %w_d2_l51_conv: tensor<32x128x3x3xf16>,
    %w_d2_l52_bn: tensor<128x32x1x1xf16>,
    %w_d2_l52_conv: tensor<32x128x3x3xf16>,
    %w_d2_l53_bn: tensor<128x32x1x1xf16>,
    %w_d2_l53_conv: tensor<32x128x3x3xf16>,
    %w_d2_l54_bn: tensor<128x32x1x1xf16>,
    %w_d2_l54_conv: tensor<32x128x3x3xf16>,
    %w_d2_l55_bn: tensor<128x32x1x1xf16>,
    %w_d2_l55_conv: tensor<32x128x3x3xf16>,
    %w_d2_l56_bn: tensor<128x32x1x1xf16>,
    %w_d2_l56_conv: tensor<32x128x3x3xf16>,
    %w_d2_l57_bn: tensor<128x32x1x1xf16>,
    %w_d2_l57_conv: tensor<32x128x3x3xf16>,
    %w_d2_l58_bn: tensor<128x32x1x1xf16>,
    %w_d2_l58_conv: tensor<32x128x3x3xf16>,
    %w_d2_l59_bn: tensor<128x32x1x1xf16>,
    %w_d2_l59_conv: tensor<32x128x3x3xf16>,
    %w_d2_l60_bn: tensor<128x32x1x1xf16>,
    %w_d2_l60_conv: tensor<32x128x3x3xf16>,
    %w_d2_l61_bn: tensor<128x32x1x1xf16>,
    %w_d2_l61_conv: tensor<32x128x3x3xf16>,
    %w_d2_l62_bn: tensor<128x32x1x1xf16>,
    %w_d2_l62_conv: tensor<32x128x3x3xf16>,
    %w_d2_l63_bn: tensor<128x32x1x1xf16>,
    %w_d2_l63_conv: tensor<32x128x3x3xf16>,
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
    %w_d3_l16_bn: tensor<128x32x1x1xf16>,
    %w_d3_l16_conv: tensor<32x128x3x3xf16>,
    %w_d3_l17_bn: tensor<128x32x1x1xf16>,
    %w_d3_l17_conv: tensor<32x128x3x3xf16>,
    %w_d3_l18_bn: tensor<128x32x1x1xf16>,
    %w_d3_l18_conv: tensor<32x128x3x3xf16>,
    %w_d3_l19_bn: tensor<128x32x1x1xf16>,
    %w_d3_l19_conv: tensor<32x128x3x3xf16>,
    %w_d3_l20_bn: tensor<128x32x1x1xf16>,
    %w_d3_l20_conv: tensor<32x128x3x3xf16>,
    %w_d3_l21_bn: tensor<128x32x1x1xf16>,
    %w_d3_l21_conv: tensor<32x128x3x3xf16>,
    %w_d3_l22_bn: tensor<128x32x1x1xf16>,
    %w_d3_l22_conv: tensor<32x128x3x3xf16>,
    %w_d3_l23_bn: tensor<128x32x1x1xf16>,
    %w_d3_l23_conv: tensor<32x128x3x3xf16>,
    %w_d3_l24_bn: tensor<128x32x1x1xf16>,
    %w_d3_l24_conv: tensor<32x128x3x3xf16>,
    %w_d3_l25_bn: tensor<128x32x1x1xf16>,
    %w_d3_l25_conv: tensor<32x128x3x3xf16>,
    %w_d3_l26_bn: tensor<128x32x1x1xf16>,
    %w_d3_l26_conv: tensor<32x128x3x3xf16>,
    %w_d3_l27_bn: tensor<128x32x1x1xf16>,
    %w_d3_l27_conv: tensor<32x128x3x3xf16>,
    %w_d3_l28_bn: tensor<128x32x1x1xf16>,
    %w_d3_l28_conv: tensor<32x128x3x3xf16>,
    %w_d3_l29_bn: tensor<128x32x1x1xf16>,
    %w_d3_l29_conv: tensor<32x128x3x3xf16>,
    %w_d3_l30_bn: tensor<128x32x1x1xf16>,
    %w_d3_l30_conv: tensor<32x128x3x3xf16>,
    %w_d3_l31_bn: tensor<128x32x1x1xf16>,
    %w_d3_l31_conv: tensor<32x128x3x3xf16>,
    %w_d3_l32_bn: tensor<128x32x1x1xf16>,
    %w_d3_l32_conv: tensor<32x128x3x3xf16>,
    %w_d3_l33_bn: tensor<128x32x1x1xf16>,
    %w_d3_l33_conv: tensor<32x128x3x3xf16>,
    %w_d3_l34_bn: tensor<128x32x1x1xf16>,
    %w_d3_l34_conv: tensor<32x128x3x3xf16>,
    %w_d3_l35_bn: tensor<128x32x1x1xf16>,
    %w_d3_l35_conv: tensor<32x128x3x3xf16>,
    %w_d3_l36_bn: tensor<128x32x1x1xf16>,
    %w_d3_l36_conv: tensor<32x128x3x3xf16>,
    %w_d3_l37_bn: tensor<128x32x1x1xf16>,
    %w_d3_l37_conv: tensor<32x128x3x3xf16>,
    %w_d3_l38_bn: tensor<128x32x1x1xf16>,
    %w_d3_l38_conv: tensor<32x128x3x3xf16>,
    %w_d3_l39_bn: tensor<128x32x1x1xf16>,
    %w_d3_l39_conv: tensor<32x128x3x3xf16>,
    %w_d3_l40_bn: tensor<128x32x1x1xf16>,
    %w_d3_l40_conv: tensor<32x128x3x3xf16>,
    %w_d3_l41_bn: tensor<128x32x1x1xf16>,
    %w_d3_l41_conv: tensor<32x128x3x3xf16>,
    %w_d3_l42_bn: tensor<128x32x1x1xf16>,
    %w_d3_l42_conv: tensor<32x128x3x3xf16>,
    %w_d3_l43_bn: tensor<128x32x1x1xf16>,
    %w_d3_l43_conv: tensor<32x128x3x3xf16>,
    %w_d3_l44_bn: tensor<128x32x1x1xf16>,
    %w_d3_l44_conv: tensor<32x128x3x3xf16>,
    %w_d3_l45_bn: tensor<128x32x1x1xf16>,
    %w_d3_l45_conv: tensor<32x128x3x3xf16>,
    %w_d3_l46_bn: tensor<128x32x1x1xf16>,
    %w_d3_l46_conv: tensor<32x128x3x3xf16>,
    %w_d3_l47_bn: tensor<128x32x1x1xf16>,
    %w_d3_l47_conv: tensor<32x128x3x3xf16>,
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

  // === Dense Block 2: 64 layers ===
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
  %init496 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill497 = linalg.fill ins(%cst : f16) outs(%init496 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv498 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu495, %w_d2_l24_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill497 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty499 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu500 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv498 : tensor<1x128x14x14xf16>)
    outs(%empty499 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad501 = tensor.pad %relu500 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init502 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill503 = linalg.fill ins(%cst : f16) outs(%init502 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv504 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad501, %w_d2_l24_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill503 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty505 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu506 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv504 : tensor<1x32x14x14xf16>)
    outs(%empty505 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init507 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill508 = linalg.fill ins(%cst : f16) outs(%init507 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv509 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu506, %w_d2_l25_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill508 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty510 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu511 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv509 : tensor<1x128x14x14xf16>)
    outs(%empty510 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad512 = tensor.pad %relu511 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init513 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill514 = linalg.fill ins(%cst : f16) outs(%init513 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv515 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad512, %w_d2_l25_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill514 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty516 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu517 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv515 : tensor<1x32x14x14xf16>)
    outs(%empty516 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init518 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill519 = linalg.fill ins(%cst : f16) outs(%init518 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv520 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu517, %w_d2_l26_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill519 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty521 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu522 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv520 : tensor<1x128x14x14xf16>)
    outs(%empty521 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad523 = tensor.pad %relu522 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init524 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill525 = linalg.fill ins(%cst : f16) outs(%init524 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv526 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad523, %w_d2_l26_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill525 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty527 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu528 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv526 : tensor<1x32x14x14xf16>)
    outs(%empty527 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init529 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill530 = linalg.fill ins(%cst : f16) outs(%init529 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv531 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu528, %w_d2_l27_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill530 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty532 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu533 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv531 : tensor<1x128x14x14xf16>)
    outs(%empty532 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad534 = tensor.pad %relu533 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init535 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill536 = linalg.fill ins(%cst : f16) outs(%init535 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv537 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad534, %w_d2_l27_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill536 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty538 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu539 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv537 : tensor<1x32x14x14xf16>)
    outs(%empty538 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init540 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill541 = linalg.fill ins(%cst : f16) outs(%init540 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv542 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu539, %w_d2_l28_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill541 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty543 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu544 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv542 : tensor<1x128x14x14xf16>)
    outs(%empty543 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad545 = tensor.pad %relu544 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init546 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill547 = linalg.fill ins(%cst : f16) outs(%init546 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv548 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad545, %w_d2_l28_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill547 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty549 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu550 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv548 : tensor<1x32x14x14xf16>)
    outs(%empty549 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init551 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill552 = linalg.fill ins(%cst : f16) outs(%init551 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv553 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu550, %w_d2_l29_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill552 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty554 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu555 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv553 : tensor<1x128x14x14xf16>)
    outs(%empty554 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad556 = tensor.pad %relu555 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init557 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill558 = linalg.fill ins(%cst : f16) outs(%init557 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv559 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad556, %w_d2_l29_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill558 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty560 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu561 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv559 : tensor<1x32x14x14xf16>)
    outs(%empty560 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init562 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill563 = linalg.fill ins(%cst : f16) outs(%init562 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv564 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu561, %w_d2_l30_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill563 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty565 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu566 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv564 : tensor<1x128x14x14xf16>)
    outs(%empty565 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad567 = tensor.pad %relu566 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init568 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill569 = linalg.fill ins(%cst : f16) outs(%init568 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv570 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad567, %w_d2_l30_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill569 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty571 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu572 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv570 : tensor<1x32x14x14xf16>)
    outs(%empty571 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init573 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill574 = linalg.fill ins(%cst : f16) outs(%init573 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv575 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu572, %w_d2_l31_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill574 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty576 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu577 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv575 : tensor<1x128x14x14xf16>)
    outs(%empty576 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad578 = tensor.pad %relu577 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init579 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill580 = linalg.fill ins(%cst : f16) outs(%init579 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv581 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad578, %w_d2_l31_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill580 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty582 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu583 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv581 : tensor<1x32x14x14xf16>)
    outs(%empty582 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init584 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill585 = linalg.fill ins(%cst : f16) outs(%init584 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv586 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu583, %w_d2_l32_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill585 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty587 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu588 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv586 : tensor<1x128x14x14xf16>)
    outs(%empty587 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad589 = tensor.pad %relu588 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init590 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill591 = linalg.fill ins(%cst : f16) outs(%init590 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv592 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad589, %w_d2_l32_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill591 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty593 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu594 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv592 : tensor<1x32x14x14xf16>)
    outs(%empty593 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init595 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill596 = linalg.fill ins(%cst : f16) outs(%init595 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv597 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu594, %w_d2_l33_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill596 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty598 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu599 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv597 : tensor<1x128x14x14xf16>)
    outs(%empty598 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad600 = tensor.pad %relu599 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init601 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill602 = linalg.fill ins(%cst : f16) outs(%init601 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv603 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad600, %w_d2_l33_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill602 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty604 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu605 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv603 : tensor<1x32x14x14xf16>)
    outs(%empty604 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init606 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill607 = linalg.fill ins(%cst : f16) outs(%init606 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv608 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu605, %w_d2_l34_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill607 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty609 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu610 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv608 : tensor<1x128x14x14xf16>)
    outs(%empty609 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad611 = tensor.pad %relu610 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init612 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill613 = linalg.fill ins(%cst : f16) outs(%init612 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv614 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad611, %w_d2_l34_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill613 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty615 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu616 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv614 : tensor<1x32x14x14xf16>)
    outs(%empty615 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init617 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill618 = linalg.fill ins(%cst : f16) outs(%init617 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv619 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu616, %w_d2_l35_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill618 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty620 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu621 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv619 : tensor<1x128x14x14xf16>)
    outs(%empty620 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad622 = tensor.pad %relu621 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init623 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill624 = linalg.fill ins(%cst : f16) outs(%init623 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv625 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad622, %w_d2_l35_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill624 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty626 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu627 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv625 : tensor<1x32x14x14xf16>)
    outs(%empty626 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init628 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill629 = linalg.fill ins(%cst : f16) outs(%init628 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv630 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu627, %w_d2_l36_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill629 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty631 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu632 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv630 : tensor<1x128x14x14xf16>)
    outs(%empty631 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad633 = tensor.pad %relu632 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init634 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill635 = linalg.fill ins(%cst : f16) outs(%init634 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv636 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad633, %w_d2_l36_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill635 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty637 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu638 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv636 : tensor<1x32x14x14xf16>)
    outs(%empty637 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init639 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill640 = linalg.fill ins(%cst : f16) outs(%init639 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv641 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu638, %w_d2_l37_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill640 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty642 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu643 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv641 : tensor<1x128x14x14xf16>)
    outs(%empty642 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad644 = tensor.pad %relu643 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init645 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill646 = linalg.fill ins(%cst : f16) outs(%init645 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv647 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad644, %w_d2_l37_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill646 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty648 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu649 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv647 : tensor<1x32x14x14xf16>)
    outs(%empty648 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init650 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill651 = linalg.fill ins(%cst : f16) outs(%init650 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv652 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu649, %w_d2_l38_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill651 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty653 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu654 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv652 : tensor<1x128x14x14xf16>)
    outs(%empty653 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad655 = tensor.pad %relu654 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init656 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill657 = linalg.fill ins(%cst : f16) outs(%init656 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv658 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad655, %w_d2_l38_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill657 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty659 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu660 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv658 : tensor<1x32x14x14xf16>)
    outs(%empty659 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init661 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill662 = linalg.fill ins(%cst : f16) outs(%init661 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv663 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu660, %w_d2_l39_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill662 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty664 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu665 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv663 : tensor<1x128x14x14xf16>)
    outs(%empty664 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad666 = tensor.pad %relu665 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init667 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill668 = linalg.fill ins(%cst : f16) outs(%init667 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv669 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad666, %w_d2_l39_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill668 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty670 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu671 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv669 : tensor<1x32x14x14xf16>)
    outs(%empty670 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init672 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill673 = linalg.fill ins(%cst : f16) outs(%init672 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv674 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu671, %w_d2_l40_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill673 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty675 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu676 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv674 : tensor<1x128x14x14xf16>)
    outs(%empty675 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad677 = tensor.pad %relu676 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init678 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill679 = linalg.fill ins(%cst : f16) outs(%init678 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv680 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad677, %w_d2_l40_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill679 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty681 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu682 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv680 : tensor<1x32x14x14xf16>)
    outs(%empty681 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init683 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill684 = linalg.fill ins(%cst : f16) outs(%init683 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv685 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu682, %w_d2_l41_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill684 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty686 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu687 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv685 : tensor<1x128x14x14xf16>)
    outs(%empty686 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad688 = tensor.pad %relu687 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init689 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill690 = linalg.fill ins(%cst : f16) outs(%init689 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv691 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad688, %w_d2_l41_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill690 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty692 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu693 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv691 : tensor<1x32x14x14xf16>)
    outs(%empty692 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init694 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill695 = linalg.fill ins(%cst : f16) outs(%init694 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv696 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu693, %w_d2_l42_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill695 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty697 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu698 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv696 : tensor<1x128x14x14xf16>)
    outs(%empty697 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad699 = tensor.pad %relu698 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init700 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill701 = linalg.fill ins(%cst : f16) outs(%init700 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv702 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad699, %w_d2_l42_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill701 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty703 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu704 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv702 : tensor<1x32x14x14xf16>)
    outs(%empty703 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init705 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill706 = linalg.fill ins(%cst : f16) outs(%init705 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv707 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu704, %w_d2_l43_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill706 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty708 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu709 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv707 : tensor<1x128x14x14xf16>)
    outs(%empty708 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad710 = tensor.pad %relu709 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init711 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill712 = linalg.fill ins(%cst : f16) outs(%init711 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv713 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad710, %w_d2_l43_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill712 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty714 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu715 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv713 : tensor<1x32x14x14xf16>)
    outs(%empty714 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init716 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill717 = linalg.fill ins(%cst : f16) outs(%init716 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv718 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu715, %w_d2_l44_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill717 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty719 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu720 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv718 : tensor<1x128x14x14xf16>)
    outs(%empty719 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad721 = tensor.pad %relu720 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init722 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill723 = linalg.fill ins(%cst : f16) outs(%init722 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv724 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad721, %w_d2_l44_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill723 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty725 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu726 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv724 : tensor<1x32x14x14xf16>)
    outs(%empty725 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init727 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill728 = linalg.fill ins(%cst : f16) outs(%init727 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv729 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu726, %w_d2_l45_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill728 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty730 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu731 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv729 : tensor<1x128x14x14xf16>)
    outs(%empty730 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad732 = tensor.pad %relu731 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init733 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill734 = linalg.fill ins(%cst : f16) outs(%init733 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv735 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad732, %w_d2_l45_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill734 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty736 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu737 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv735 : tensor<1x32x14x14xf16>)
    outs(%empty736 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init738 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill739 = linalg.fill ins(%cst : f16) outs(%init738 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv740 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu737, %w_d2_l46_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill739 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty741 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu742 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv740 : tensor<1x128x14x14xf16>)
    outs(%empty741 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad743 = tensor.pad %relu742 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init744 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill745 = linalg.fill ins(%cst : f16) outs(%init744 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv746 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad743, %w_d2_l46_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill745 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty747 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu748 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv746 : tensor<1x32x14x14xf16>)
    outs(%empty747 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init749 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill750 = linalg.fill ins(%cst : f16) outs(%init749 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv751 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu748, %w_d2_l47_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill750 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty752 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu753 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv751 : tensor<1x128x14x14xf16>)
    outs(%empty752 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad754 = tensor.pad %relu753 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init755 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill756 = linalg.fill ins(%cst : f16) outs(%init755 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv757 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad754, %w_d2_l47_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill756 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty758 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu759 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv757 : tensor<1x32x14x14xf16>)
    outs(%empty758 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init760 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill761 = linalg.fill ins(%cst : f16) outs(%init760 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv762 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu759, %w_d2_l48_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill761 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty763 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu764 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv762 : tensor<1x128x14x14xf16>)
    outs(%empty763 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad765 = tensor.pad %relu764 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init766 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill767 = linalg.fill ins(%cst : f16) outs(%init766 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv768 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad765, %w_d2_l48_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill767 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty769 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu770 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv768 : tensor<1x32x14x14xf16>)
    outs(%empty769 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init771 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill772 = linalg.fill ins(%cst : f16) outs(%init771 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv773 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu770, %w_d2_l49_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill772 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty774 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu775 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv773 : tensor<1x128x14x14xf16>)
    outs(%empty774 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad776 = tensor.pad %relu775 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init777 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill778 = linalg.fill ins(%cst : f16) outs(%init777 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv779 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad776, %w_d2_l49_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill778 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty780 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu781 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv779 : tensor<1x32x14x14xf16>)
    outs(%empty780 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init782 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill783 = linalg.fill ins(%cst : f16) outs(%init782 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv784 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu781, %w_d2_l50_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill783 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty785 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu786 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv784 : tensor<1x128x14x14xf16>)
    outs(%empty785 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad787 = tensor.pad %relu786 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init788 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill789 = linalg.fill ins(%cst : f16) outs(%init788 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv790 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad787, %w_d2_l50_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill789 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty791 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu792 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv790 : tensor<1x32x14x14xf16>)
    outs(%empty791 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init793 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill794 = linalg.fill ins(%cst : f16) outs(%init793 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv795 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu792, %w_d2_l51_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill794 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty796 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu797 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv795 : tensor<1x128x14x14xf16>)
    outs(%empty796 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad798 = tensor.pad %relu797 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init799 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill800 = linalg.fill ins(%cst : f16) outs(%init799 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv801 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad798, %w_d2_l51_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill800 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty802 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu803 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv801 : tensor<1x32x14x14xf16>)
    outs(%empty802 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init804 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill805 = linalg.fill ins(%cst : f16) outs(%init804 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv806 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu803, %w_d2_l52_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill805 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty807 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu808 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv806 : tensor<1x128x14x14xf16>)
    outs(%empty807 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad809 = tensor.pad %relu808 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init810 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill811 = linalg.fill ins(%cst : f16) outs(%init810 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv812 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad809, %w_d2_l52_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill811 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty813 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu814 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv812 : tensor<1x32x14x14xf16>)
    outs(%empty813 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init815 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill816 = linalg.fill ins(%cst : f16) outs(%init815 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv817 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu814, %w_d2_l53_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill816 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty818 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu819 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv817 : tensor<1x128x14x14xf16>)
    outs(%empty818 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad820 = tensor.pad %relu819 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init821 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill822 = linalg.fill ins(%cst : f16) outs(%init821 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv823 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad820, %w_d2_l53_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill822 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty824 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu825 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv823 : tensor<1x32x14x14xf16>)
    outs(%empty824 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init826 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill827 = linalg.fill ins(%cst : f16) outs(%init826 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv828 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu825, %w_d2_l54_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill827 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty829 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu830 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv828 : tensor<1x128x14x14xf16>)
    outs(%empty829 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad831 = tensor.pad %relu830 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init832 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill833 = linalg.fill ins(%cst : f16) outs(%init832 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv834 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad831, %w_d2_l54_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill833 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty835 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu836 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv834 : tensor<1x32x14x14xf16>)
    outs(%empty835 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init837 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill838 = linalg.fill ins(%cst : f16) outs(%init837 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv839 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu836, %w_d2_l55_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill838 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty840 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu841 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv839 : tensor<1x128x14x14xf16>)
    outs(%empty840 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad842 = tensor.pad %relu841 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init843 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill844 = linalg.fill ins(%cst : f16) outs(%init843 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv845 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad842, %w_d2_l55_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill844 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty846 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu847 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv845 : tensor<1x32x14x14xf16>)
    outs(%empty846 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init848 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill849 = linalg.fill ins(%cst : f16) outs(%init848 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv850 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu847, %w_d2_l56_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill849 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty851 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu852 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv850 : tensor<1x128x14x14xf16>)
    outs(%empty851 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad853 = tensor.pad %relu852 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init854 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill855 = linalg.fill ins(%cst : f16) outs(%init854 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv856 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad853, %w_d2_l56_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill855 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty857 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu858 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv856 : tensor<1x32x14x14xf16>)
    outs(%empty857 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init859 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill860 = linalg.fill ins(%cst : f16) outs(%init859 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv861 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu858, %w_d2_l57_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill860 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty862 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu863 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv861 : tensor<1x128x14x14xf16>)
    outs(%empty862 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad864 = tensor.pad %relu863 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init865 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill866 = linalg.fill ins(%cst : f16) outs(%init865 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv867 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad864, %w_d2_l57_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill866 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty868 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu869 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv867 : tensor<1x32x14x14xf16>)
    outs(%empty868 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init870 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill871 = linalg.fill ins(%cst : f16) outs(%init870 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv872 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu869, %w_d2_l58_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill871 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty873 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu874 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv872 : tensor<1x128x14x14xf16>)
    outs(%empty873 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad875 = tensor.pad %relu874 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init876 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill877 = linalg.fill ins(%cst : f16) outs(%init876 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv878 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad875, %w_d2_l58_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill877 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty879 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu880 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv878 : tensor<1x32x14x14xf16>)
    outs(%empty879 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init881 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill882 = linalg.fill ins(%cst : f16) outs(%init881 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv883 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu880, %w_d2_l59_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill882 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty884 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu885 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv883 : tensor<1x128x14x14xf16>)
    outs(%empty884 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad886 = tensor.pad %relu885 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init887 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill888 = linalg.fill ins(%cst : f16) outs(%init887 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv889 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad886, %w_d2_l59_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill888 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty890 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu891 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv889 : tensor<1x32x14x14xf16>)
    outs(%empty890 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init892 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill893 = linalg.fill ins(%cst : f16) outs(%init892 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv894 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu891, %w_d2_l60_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill893 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty895 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu896 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv894 : tensor<1x128x14x14xf16>)
    outs(%empty895 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad897 = tensor.pad %relu896 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init898 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill899 = linalg.fill ins(%cst : f16) outs(%init898 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv900 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad897, %w_d2_l60_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill899 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty901 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu902 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv900 : tensor<1x32x14x14xf16>)
    outs(%empty901 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init903 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill904 = linalg.fill ins(%cst : f16) outs(%init903 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv905 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu902, %w_d2_l61_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill904 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty906 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu907 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv905 : tensor<1x128x14x14xf16>)
    outs(%empty906 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad908 = tensor.pad %relu907 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init909 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill910 = linalg.fill ins(%cst : f16) outs(%init909 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv911 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad908, %w_d2_l61_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill910 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty912 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu913 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv911 : tensor<1x32x14x14xf16>)
    outs(%empty912 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init914 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill915 = linalg.fill ins(%cst : f16) outs(%init914 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv916 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu913, %w_d2_l62_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill915 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty917 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu918 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv916 : tensor<1x128x14x14xf16>)
    outs(%empty917 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad919 = tensor.pad %relu918 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init920 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill921 = linalg.fill ins(%cst : f16) outs(%init920 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv922 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad919, %w_d2_l62_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill921 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty923 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu924 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv922 : tensor<1x32x14x14xf16>)
    outs(%empty923 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %init925 = tensor.empty() : tensor<1x128x14x14xf16>
  %fill926 = linalg.fill ins(%cst : f16) outs(%init925 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %conv927 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu924, %w_d2_l63_bn : tensor<1x32x14x14xf16>, tensor<128x32x1x1xf16>)
    outs(%fill926 : tensor<1x128x14x14xf16>) -> tensor<1x128x14x14xf16>
  %empty928 = tensor.empty() : tensor<1x128x14x14xf16>
  %relu929 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv927 : tensor<1x128x14x14xf16>)
    outs(%empty928 : tensor<1x128x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x14x14xf16>
  %pad930 = tensor.pad %relu929 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x14x14xf16> to tensor<1x128x16x16xf16>
  %init931 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill932 = linalg.fill ins(%cst : f16) outs(%init931 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv933 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad930, %w_d2_l63_conv : tensor<1x128x16x16xf16>, tensor<32x128x3x3xf16>)
    outs(%fill932 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty934 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu935 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv933 : tensor<1x32x14x14xf16>)
    outs(%empty934 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>

  // Transition 2: 1x1 conv + stride-2 pool
  %init936 = tensor.empty() : tensor<1x32x14x14xf16>
  %fill937 = linalg.fill ins(%cst : f16) outs(%init936 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %conv938 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu935, %w_t2_conv : tensor<1x32x14x14xf16>, tensor<32x32x1x1xf16>)
    outs(%fill937 : tensor<1x32x14x14xf16>) -> tensor<1x32x14x14xf16>
  %empty939 = tensor.empty() : tensor<1x32x14x14xf16>
  %relu940 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv938 : tensor<1x32x14x14xf16>)
    outs(%empty939 : tensor<1x32x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x14x14xf16>
  %pad941 = tensor.pad %relu940 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x32x14x14xf16> to tensor<1x32x16x16xf16>
  %init942 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill943 = linalg.fill ins(%cst : f16) outs(%init942 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv944 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad941, %w_t2_pool : tensor<1x32x16x16xf16>, tensor<32x32x3x3xf16>)
    outs(%fill943 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty945 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu946 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv944 : tensor<1x32x7x7xf16>)
    outs(%empty945 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>

  // === Dense Block 3: 48 layers ===
  %init947 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill948 = linalg.fill ins(%cst : f16) outs(%init947 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv949 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu946, %w_d3_l0_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill948 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty950 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu951 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv949 : tensor<1x128x7x7xf16>)
    outs(%empty950 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad952 = tensor.pad %relu951 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init953 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill954 = linalg.fill ins(%cst : f16) outs(%init953 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv955 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad952, %w_d3_l0_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill954 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty956 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu957 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv955 : tensor<1x32x7x7xf16>)
    outs(%empty956 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init958 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill959 = linalg.fill ins(%cst : f16) outs(%init958 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv960 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu957, %w_d3_l1_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill959 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty961 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu962 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv960 : tensor<1x128x7x7xf16>)
    outs(%empty961 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad963 = tensor.pad %relu962 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init964 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill965 = linalg.fill ins(%cst : f16) outs(%init964 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv966 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad963, %w_d3_l1_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill965 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty967 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu968 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv966 : tensor<1x32x7x7xf16>)
    outs(%empty967 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init969 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill970 = linalg.fill ins(%cst : f16) outs(%init969 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv971 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu968, %w_d3_l2_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill970 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty972 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu973 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv971 : tensor<1x128x7x7xf16>)
    outs(%empty972 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad974 = tensor.pad %relu973 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init975 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill976 = linalg.fill ins(%cst : f16) outs(%init975 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv977 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad974, %w_d3_l2_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill976 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty978 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu979 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv977 : tensor<1x32x7x7xf16>)
    outs(%empty978 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init980 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill981 = linalg.fill ins(%cst : f16) outs(%init980 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv982 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu979, %w_d3_l3_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill981 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty983 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu984 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv982 : tensor<1x128x7x7xf16>)
    outs(%empty983 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad985 = tensor.pad %relu984 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init986 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill987 = linalg.fill ins(%cst : f16) outs(%init986 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv988 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad985, %w_d3_l3_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill987 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty989 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu990 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv988 : tensor<1x32x7x7xf16>)
    outs(%empty989 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init991 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill992 = linalg.fill ins(%cst : f16) outs(%init991 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv993 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu990, %w_d3_l4_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill992 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty994 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu995 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv993 : tensor<1x128x7x7xf16>)
    outs(%empty994 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad996 = tensor.pad %relu995 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init997 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill998 = linalg.fill ins(%cst : f16) outs(%init997 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv999 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad996, %w_d3_l4_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill998 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1000 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1001 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv999 : tensor<1x32x7x7xf16>)
    outs(%empty1000 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1002 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1003 = linalg.fill ins(%cst : f16) outs(%init1002 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1004 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1001, %w_d3_l5_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1003 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1005 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1006 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1004 : tensor<1x128x7x7xf16>)
    outs(%empty1005 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1007 = tensor.pad %relu1006 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1008 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1009 = linalg.fill ins(%cst : f16) outs(%init1008 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1010 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1007, %w_d3_l5_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1009 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1011 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1012 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1010 : tensor<1x32x7x7xf16>)
    outs(%empty1011 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1013 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1014 = linalg.fill ins(%cst : f16) outs(%init1013 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1015 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1012, %w_d3_l6_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1014 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1016 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1017 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1015 : tensor<1x128x7x7xf16>)
    outs(%empty1016 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1018 = tensor.pad %relu1017 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1019 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1020 = linalg.fill ins(%cst : f16) outs(%init1019 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1021 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1018, %w_d3_l6_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1020 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1022 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1023 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1021 : tensor<1x32x7x7xf16>)
    outs(%empty1022 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1024 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1025 = linalg.fill ins(%cst : f16) outs(%init1024 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1026 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1023, %w_d3_l7_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1025 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1027 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1028 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1026 : tensor<1x128x7x7xf16>)
    outs(%empty1027 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1029 = tensor.pad %relu1028 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1030 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1031 = linalg.fill ins(%cst : f16) outs(%init1030 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1032 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1029, %w_d3_l7_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1031 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1033 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1034 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1032 : tensor<1x32x7x7xf16>)
    outs(%empty1033 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1035 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1036 = linalg.fill ins(%cst : f16) outs(%init1035 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1037 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1034, %w_d3_l8_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1036 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1038 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1039 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1037 : tensor<1x128x7x7xf16>)
    outs(%empty1038 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1040 = tensor.pad %relu1039 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1041 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1042 = linalg.fill ins(%cst : f16) outs(%init1041 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1043 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1040, %w_d3_l8_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1042 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1044 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1045 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1043 : tensor<1x32x7x7xf16>)
    outs(%empty1044 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1046 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1047 = linalg.fill ins(%cst : f16) outs(%init1046 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1048 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1045, %w_d3_l9_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1047 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1049 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1050 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1048 : tensor<1x128x7x7xf16>)
    outs(%empty1049 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1051 = tensor.pad %relu1050 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1052 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1053 = linalg.fill ins(%cst : f16) outs(%init1052 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1054 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1051, %w_d3_l9_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1053 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1055 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1056 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1054 : tensor<1x32x7x7xf16>)
    outs(%empty1055 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1057 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1058 = linalg.fill ins(%cst : f16) outs(%init1057 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1059 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1056, %w_d3_l10_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1058 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1060 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1061 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1059 : tensor<1x128x7x7xf16>)
    outs(%empty1060 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1062 = tensor.pad %relu1061 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1063 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1064 = linalg.fill ins(%cst : f16) outs(%init1063 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1065 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1062, %w_d3_l10_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1064 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1066 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1067 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1065 : tensor<1x32x7x7xf16>)
    outs(%empty1066 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1068 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1069 = linalg.fill ins(%cst : f16) outs(%init1068 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1070 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1067, %w_d3_l11_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1069 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1071 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1072 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1070 : tensor<1x128x7x7xf16>)
    outs(%empty1071 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1073 = tensor.pad %relu1072 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1074 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1075 = linalg.fill ins(%cst : f16) outs(%init1074 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1076 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1073, %w_d3_l11_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1075 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1077 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1078 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1076 : tensor<1x32x7x7xf16>)
    outs(%empty1077 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1079 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1080 = linalg.fill ins(%cst : f16) outs(%init1079 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1081 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1078, %w_d3_l12_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1080 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1082 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1083 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1081 : tensor<1x128x7x7xf16>)
    outs(%empty1082 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1084 = tensor.pad %relu1083 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1085 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1086 = linalg.fill ins(%cst : f16) outs(%init1085 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1087 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1084, %w_d3_l12_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1086 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1088 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1089 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1087 : tensor<1x32x7x7xf16>)
    outs(%empty1088 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1090 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1091 = linalg.fill ins(%cst : f16) outs(%init1090 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1092 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1089, %w_d3_l13_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1091 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1093 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1094 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1092 : tensor<1x128x7x7xf16>)
    outs(%empty1093 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1095 = tensor.pad %relu1094 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1096 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1097 = linalg.fill ins(%cst : f16) outs(%init1096 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1098 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1095, %w_d3_l13_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1097 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1099 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1100 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1098 : tensor<1x32x7x7xf16>)
    outs(%empty1099 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1101 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1102 = linalg.fill ins(%cst : f16) outs(%init1101 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1103 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1100, %w_d3_l14_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1102 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1104 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1105 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1103 : tensor<1x128x7x7xf16>)
    outs(%empty1104 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1106 = tensor.pad %relu1105 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1107 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1108 = linalg.fill ins(%cst : f16) outs(%init1107 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1109 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1106, %w_d3_l14_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1108 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1110 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1111 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1109 : tensor<1x32x7x7xf16>)
    outs(%empty1110 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1112 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1113 = linalg.fill ins(%cst : f16) outs(%init1112 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1114 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1111, %w_d3_l15_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1113 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1115 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1116 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1114 : tensor<1x128x7x7xf16>)
    outs(%empty1115 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1117 = tensor.pad %relu1116 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1118 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1119 = linalg.fill ins(%cst : f16) outs(%init1118 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1120 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1117, %w_d3_l15_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1119 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1121 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1122 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1120 : tensor<1x32x7x7xf16>)
    outs(%empty1121 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1123 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1124 = linalg.fill ins(%cst : f16) outs(%init1123 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1125 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1122, %w_d3_l16_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1124 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1126 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1127 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1125 : tensor<1x128x7x7xf16>)
    outs(%empty1126 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1128 = tensor.pad %relu1127 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1129 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1130 = linalg.fill ins(%cst : f16) outs(%init1129 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1131 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1128, %w_d3_l16_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1130 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1132 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1133 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1131 : tensor<1x32x7x7xf16>)
    outs(%empty1132 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1134 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1135 = linalg.fill ins(%cst : f16) outs(%init1134 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1136 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1133, %w_d3_l17_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1135 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1137 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1138 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1136 : tensor<1x128x7x7xf16>)
    outs(%empty1137 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1139 = tensor.pad %relu1138 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1140 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1141 = linalg.fill ins(%cst : f16) outs(%init1140 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1142 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1139, %w_d3_l17_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1141 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1143 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1144 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1142 : tensor<1x32x7x7xf16>)
    outs(%empty1143 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1145 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1146 = linalg.fill ins(%cst : f16) outs(%init1145 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1147 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1144, %w_d3_l18_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1146 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1148 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1149 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1147 : tensor<1x128x7x7xf16>)
    outs(%empty1148 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1150 = tensor.pad %relu1149 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1151 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1152 = linalg.fill ins(%cst : f16) outs(%init1151 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1153 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1150, %w_d3_l18_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1152 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1154 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1155 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1153 : tensor<1x32x7x7xf16>)
    outs(%empty1154 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1156 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1157 = linalg.fill ins(%cst : f16) outs(%init1156 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1158 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1155, %w_d3_l19_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1157 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1159 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1160 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1158 : tensor<1x128x7x7xf16>)
    outs(%empty1159 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1161 = tensor.pad %relu1160 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1162 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1163 = linalg.fill ins(%cst : f16) outs(%init1162 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1164 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1161, %w_d3_l19_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1163 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1165 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1166 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1164 : tensor<1x32x7x7xf16>)
    outs(%empty1165 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1167 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1168 = linalg.fill ins(%cst : f16) outs(%init1167 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1169 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1166, %w_d3_l20_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1168 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1170 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1171 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1169 : tensor<1x128x7x7xf16>)
    outs(%empty1170 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1172 = tensor.pad %relu1171 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1173 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1174 = linalg.fill ins(%cst : f16) outs(%init1173 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1175 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1172, %w_d3_l20_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1174 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1176 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1177 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1175 : tensor<1x32x7x7xf16>)
    outs(%empty1176 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1178 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1179 = linalg.fill ins(%cst : f16) outs(%init1178 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1180 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1177, %w_d3_l21_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1179 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1181 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1182 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1180 : tensor<1x128x7x7xf16>)
    outs(%empty1181 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1183 = tensor.pad %relu1182 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1184 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1185 = linalg.fill ins(%cst : f16) outs(%init1184 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1186 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1183, %w_d3_l21_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1185 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1187 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1188 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1186 : tensor<1x32x7x7xf16>)
    outs(%empty1187 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1189 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1190 = linalg.fill ins(%cst : f16) outs(%init1189 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1191 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1188, %w_d3_l22_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1190 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1192 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1193 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1191 : tensor<1x128x7x7xf16>)
    outs(%empty1192 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1194 = tensor.pad %relu1193 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1195 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1196 = linalg.fill ins(%cst : f16) outs(%init1195 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1197 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1194, %w_d3_l22_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1196 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1198 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1199 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1197 : tensor<1x32x7x7xf16>)
    outs(%empty1198 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1200 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1201 = linalg.fill ins(%cst : f16) outs(%init1200 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1202 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1199, %w_d3_l23_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1201 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1203 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1204 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1202 : tensor<1x128x7x7xf16>)
    outs(%empty1203 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1205 = tensor.pad %relu1204 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1206 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1207 = linalg.fill ins(%cst : f16) outs(%init1206 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1208 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1205, %w_d3_l23_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1207 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1209 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1210 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1208 : tensor<1x32x7x7xf16>)
    outs(%empty1209 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1211 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1212 = linalg.fill ins(%cst : f16) outs(%init1211 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1213 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1210, %w_d3_l24_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1212 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1214 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1215 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1213 : tensor<1x128x7x7xf16>)
    outs(%empty1214 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1216 = tensor.pad %relu1215 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1217 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1218 = linalg.fill ins(%cst : f16) outs(%init1217 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1219 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1216, %w_d3_l24_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1218 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1220 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1221 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1219 : tensor<1x32x7x7xf16>)
    outs(%empty1220 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1222 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1223 = linalg.fill ins(%cst : f16) outs(%init1222 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1224 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1221, %w_d3_l25_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1223 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1225 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1226 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1224 : tensor<1x128x7x7xf16>)
    outs(%empty1225 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1227 = tensor.pad %relu1226 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1228 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1229 = linalg.fill ins(%cst : f16) outs(%init1228 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1230 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1227, %w_d3_l25_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1229 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1231 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1232 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1230 : tensor<1x32x7x7xf16>)
    outs(%empty1231 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1233 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1234 = linalg.fill ins(%cst : f16) outs(%init1233 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1235 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1232, %w_d3_l26_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1234 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1236 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1237 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1235 : tensor<1x128x7x7xf16>)
    outs(%empty1236 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1238 = tensor.pad %relu1237 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1239 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1240 = linalg.fill ins(%cst : f16) outs(%init1239 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1241 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1238, %w_d3_l26_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1240 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1242 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1243 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1241 : tensor<1x32x7x7xf16>)
    outs(%empty1242 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1244 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1245 = linalg.fill ins(%cst : f16) outs(%init1244 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1246 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1243, %w_d3_l27_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1245 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1247 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1248 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1246 : tensor<1x128x7x7xf16>)
    outs(%empty1247 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1249 = tensor.pad %relu1248 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1250 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1251 = linalg.fill ins(%cst : f16) outs(%init1250 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1252 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1249, %w_d3_l27_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1251 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1253 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1254 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1252 : tensor<1x32x7x7xf16>)
    outs(%empty1253 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1255 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1256 = linalg.fill ins(%cst : f16) outs(%init1255 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1257 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1254, %w_d3_l28_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1256 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1258 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1259 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1257 : tensor<1x128x7x7xf16>)
    outs(%empty1258 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1260 = tensor.pad %relu1259 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1261 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1262 = linalg.fill ins(%cst : f16) outs(%init1261 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1263 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1260, %w_d3_l28_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1262 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1264 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1265 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1263 : tensor<1x32x7x7xf16>)
    outs(%empty1264 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1266 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1267 = linalg.fill ins(%cst : f16) outs(%init1266 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1268 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1265, %w_d3_l29_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1267 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1269 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1270 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1268 : tensor<1x128x7x7xf16>)
    outs(%empty1269 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1271 = tensor.pad %relu1270 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1272 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1273 = linalg.fill ins(%cst : f16) outs(%init1272 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1274 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1271, %w_d3_l29_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1273 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1275 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1276 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1274 : tensor<1x32x7x7xf16>)
    outs(%empty1275 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1277 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1278 = linalg.fill ins(%cst : f16) outs(%init1277 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1279 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1276, %w_d3_l30_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1278 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1280 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1281 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1279 : tensor<1x128x7x7xf16>)
    outs(%empty1280 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1282 = tensor.pad %relu1281 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1283 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1284 = linalg.fill ins(%cst : f16) outs(%init1283 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1285 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1282, %w_d3_l30_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1284 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1286 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1287 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1285 : tensor<1x32x7x7xf16>)
    outs(%empty1286 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1288 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1289 = linalg.fill ins(%cst : f16) outs(%init1288 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1290 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1287, %w_d3_l31_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1289 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1291 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1292 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1290 : tensor<1x128x7x7xf16>)
    outs(%empty1291 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1293 = tensor.pad %relu1292 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1294 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1295 = linalg.fill ins(%cst : f16) outs(%init1294 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1296 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1293, %w_d3_l31_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1295 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1297 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1298 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1296 : tensor<1x32x7x7xf16>)
    outs(%empty1297 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1299 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1300 = linalg.fill ins(%cst : f16) outs(%init1299 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1301 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1298, %w_d3_l32_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1300 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1302 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1303 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1301 : tensor<1x128x7x7xf16>)
    outs(%empty1302 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1304 = tensor.pad %relu1303 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1305 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1306 = linalg.fill ins(%cst : f16) outs(%init1305 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1307 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1304, %w_d3_l32_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1306 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1308 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1309 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1307 : tensor<1x32x7x7xf16>)
    outs(%empty1308 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1310 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1311 = linalg.fill ins(%cst : f16) outs(%init1310 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1312 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1309, %w_d3_l33_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1311 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1313 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1314 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1312 : tensor<1x128x7x7xf16>)
    outs(%empty1313 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1315 = tensor.pad %relu1314 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1316 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1317 = linalg.fill ins(%cst : f16) outs(%init1316 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1318 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1315, %w_d3_l33_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1317 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1319 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1320 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1318 : tensor<1x32x7x7xf16>)
    outs(%empty1319 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1321 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1322 = linalg.fill ins(%cst : f16) outs(%init1321 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1323 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1320, %w_d3_l34_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1322 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1324 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1325 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1323 : tensor<1x128x7x7xf16>)
    outs(%empty1324 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1326 = tensor.pad %relu1325 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1327 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1328 = linalg.fill ins(%cst : f16) outs(%init1327 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1329 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1326, %w_d3_l34_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1328 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1330 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1331 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1329 : tensor<1x32x7x7xf16>)
    outs(%empty1330 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1332 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1333 = linalg.fill ins(%cst : f16) outs(%init1332 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1334 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1331, %w_d3_l35_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1333 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1335 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1336 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1334 : tensor<1x128x7x7xf16>)
    outs(%empty1335 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1337 = tensor.pad %relu1336 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1338 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1339 = linalg.fill ins(%cst : f16) outs(%init1338 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1340 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1337, %w_d3_l35_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1339 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1341 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1342 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1340 : tensor<1x32x7x7xf16>)
    outs(%empty1341 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1343 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1344 = linalg.fill ins(%cst : f16) outs(%init1343 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1345 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1342, %w_d3_l36_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1344 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1346 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1347 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1345 : tensor<1x128x7x7xf16>)
    outs(%empty1346 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1348 = tensor.pad %relu1347 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1349 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1350 = linalg.fill ins(%cst : f16) outs(%init1349 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1351 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1348, %w_d3_l36_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1350 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1352 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1353 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1351 : tensor<1x32x7x7xf16>)
    outs(%empty1352 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1354 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1355 = linalg.fill ins(%cst : f16) outs(%init1354 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1356 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1353, %w_d3_l37_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1355 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1357 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1358 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1356 : tensor<1x128x7x7xf16>)
    outs(%empty1357 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1359 = tensor.pad %relu1358 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1360 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1361 = linalg.fill ins(%cst : f16) outs(%init1360 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1362 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1359, %w_d3_l37_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1361 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1363 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1364 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1362 : tensor<1x32x7x7xf16>)
    outs(%empty1363 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1365 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1366 = linalg.fill ins(%cst : f16) outs(%init1365 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1367 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1364, %w_d3_l38_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1366 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1368 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1369 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1367 : tensor<1x128x7x7xf16>)
    outs(%empty1368 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1370 = tensor.pad %relu1369 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1371 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1372 = linalg.fill ins(%cst : f16) outs(%init1371 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1373 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1370, %w_d3_l38_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1372 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1374 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1375 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1373 : tensor<1x32x7x7xf16>)
    outs(%empty1374 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1376 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1377 = linalg.fill ins(%cst : f16) outs(%init1376 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1378 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1375, %w_d3_l39_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1377 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1379 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1380 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1378 : tensor<1x128x7x7xf16>)
    outs(%empty1379 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1381 = tensor.pad %relu1380 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1382 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1383 = linalg.fill ins(%cst : f16) outs(%init1382 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1384 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1381, %w_d3_l39_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1383 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1385 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1386 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1384 : tensor<1x32x7x7xf16>)
    outs(%empty1385 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1387 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1388 = linalg.fill ins(%cst : f16) outs(%init1387 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1389 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1386, %w_d3_l40_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1388 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1390 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1391 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1389 : tensor<1x128x7x7xf16>)
    outs(%empty1390 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1392 = tensor.pad %relu1391 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1393 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1394 = linalg.fill ins(%cst : f16) outs(%init1393 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1395 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1392, %w_d3_l40_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1394 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1396 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1397 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1395 : tensor<1x32x7x7xf16>)
    outs(%empty1396 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1398 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1399 = linalg.fill ins(%cst : f16) outs(%init1398 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1400 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1397, %w_d3_l41_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1399 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1401 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1402 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1400 : tensor<1x128x7x7xf16>)
    outs(%empty1401 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1403 = tensor.pad %relu1402 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1404 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1405 = linalg.fill ins(%cst : f16) outs(%init1404 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1406 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1403, %w_d3_l41_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1405 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1407 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1408 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1406 : tensor<1x32x7x7xf16>)
    outs(%empty1407 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1409 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1410 = linalg.fill ins(%cst : f16) outs(%init1409 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1411 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1408, %w_d3_l42_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1410 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1412 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1413 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1411 : tensor<1x128x7x7xf16>)
    outs(%empty1412 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1414 = tensor.pad %relu1413 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1415 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1416 = linalg.fill ins(%cst : f16) outs(%init1415 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1417 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1414, %w_d3_l42_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1416 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1418 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1419 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1417 : tensor<1x32x7x7xf16>)
    outs(%empty1418 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1420 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1421 = linalg.fill ins(%cst : f16) outs(%init1420 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1422 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1419, %w_d3_l43_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1421 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1423 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1424 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1422 : tensor<1x128x7x7xf16>)
    outs(%empty1423 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1425 = tensor.pad %relu1424 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1426 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1427 = linalg.fill ins(%cst : f16) outs(%init1426 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1428 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1425, %w_d3_l43_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1427 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1429 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1430 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1428 : tensor<1x32x7x7xf16>)
    outs(%empty1429 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1431 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1432 = linalg.fill ins(%cst : f16) outs(%init1431 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1433 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1430, %w_d3_l44_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1432 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1434 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1435 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1433 : tensor<1x128x7x7xf16>)
    outs(%empty1434 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1436 = tensor.pad %relu1435 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1437 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1438 = linalg.fill ins(%cst : f16) outs(%init1437 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1439 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1436, %w_d3_l44_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1438 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1440 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1441 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1439 : tensor<1x32x7x7xf16>)
    outs(%empty1440 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1442 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1443 = linalg.fill ins(%cst : f16) outs(%init1442 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1444 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1441, %w_d3_l45_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1443 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1445 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1446 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1444 : tensor<1x128x7x7xf16>)
    outs(%empty1445 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1447 = tensor.pad %relu1446 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1448 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1449 = linalg.fill ins(%cst : f16) outs(%init1448 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1450 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1447, %w_d3_l45_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1449 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1451 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1452 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1450 : tensor<1x32x7x7xf16>)
    outs(%empty1451 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1453 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1454 = linalg.fill ins(%cst : f16) outs(%init1453 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1455 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1452, %w_d3_l46_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1454 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1456 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1457 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1455 : tensor<1x128x7x7xf16>)
    outs(%empty1456 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1458 = tensor.pad %relu1457 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1459 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1460 = linalg.fill ins(%cst : f16) outs(%init1459 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1461 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1458, %w_d3_l46_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1460 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1462 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1463 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1461 : tensor<1x32x7x7xf16>)
    outs(%empty1462 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>
  %init1464 = tensor.empty() : tensor<1x128x7x7xf16>
  %fill1465 = linalg.fill ins(%cst : f16) outs(%init1464 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %conv1466 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1463, %w_d3_l47_bn : tensor<1x32x7x7xf16>, tensor<128x32x1x1xf16>)
    outs(%fill1465 : tensor<1x128x7x7xf16>) -> tensor<1x128x7x7xf16>
  %empty1467 = tensor.empty() : tensor<1x128x7x7xf16>
  %relu1468 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1466 : tensor<1x128x7x7xf16>)
    outs(%empty1467 : tensor<1x128x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x7x7xf16>
  %pad1469 = tensor.pad %relu1468 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x7x7xf16> to tensor<1x128x9x9xf16>
  %init1470 = tensor.empty() : tensor<1x32x7x7xf16>
  %fill1471 = linalg.fill ins(%cst : f16) outs(%init1470 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %conv1472 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad1469, %w_d3_l47_conv : tensor<1x128x9x9xf16>, tensor<32x128x3x3xf16>)
    outs(%fill1471 : tensor<1x32x7x7xf16>) -> tensor<1x32x7x7xf16>
  %empty1473 = tensor.empty() : tensor<1x32x7x7xf16>
  %relu1474 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv1472 : tensor<1x32x7x7xf16>)
    outs(%empty1473 : tensor<1x32x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x7x7xf16>

  // FC: 1x1 conv 32->1000
  %init1475 = tensor.empty() : tensor<1x1000x7x7xf16>
  %fill1476 = linalg.fill ins(%cst : f16) outs(%init1475 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  %conv1477 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu1474, %w_fc : tensor<1x32x7x7xf16>, tensor<1000x32x1x1xf16>)
    outs(%fill1476 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  return %conv1477 : tensor<1x1000x7x7xf16>
}
