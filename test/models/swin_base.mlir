func.func @swin_base(
    %input: tensor<1x3x224x224xf16>,
    %w_patch: tensor<128x3x4x4xf16>,
    %w_s0_l0_q: tensor<128x128xf16>,
    %w_s0_l0_k: tensor<128x128xf16>,
    %w_s0_l0_v: tensor<128x128xf16>,
    %w_s0_l0_kt: tensor<128x3136xf16>,
    %w_s0_l0_o: tensor<128x128xf16>,
    %w_s0_l0_ff_up: tensor<128x512xf16>,
    %w_s0_l0_ff_dn: tensor<512x128xf16>,
    %w_s0_l1_q: tensor<128x128xf16>,
    %w_s0_l1_k: tensor<128x128xf16>,
    %w_s0_l1_v: tensor<128x128xf16>,
    %w_s0_l1_kt: tensor<128x3136xf16>,
    %w_s0_l1_o: tensor<128x128xf16>,
    %w_s0_l1_ff_up: tensor<128x512xf16>,
    %w_s0_l1_ff_dn: tensor<512x128xf16>,
    %w_merge0: tensor<128x256xf16>,
    %w_s1_l0_q: tensor<256x256xf16>,
    %w_s1_l0_k: tensor<256x256xf16>,
    %w_s1_l0_v: tensor<256x256xf16>,
    %w_s1_l0_kt: tensor<256x784xf16>,
    %w_s1_l0_o: tensor<256x256xf16>,
    %w_s1_l0_ff_up: tensor<256x1024xf16>,
    %w_s1_l0_ff_dn: tensor<1024x256xf16>,
    %w_s1_l1_q: tensor<256x256xf16>,
    %w_s1_l1_k: tensor<256x256xf16>,
    %w_s1_l1_v: tensor<256x256xf16>,
    %w_s1_l1_kt: tensor<256x784xf16>,
    %w_s1_l1_o: tensor<256x256xf16>,
    %w_s1_l1_ff_up: tensor<256x1024xf16>,
    %w_s1_l1_ff_dn: tensor<1024x256xf16>,
    %w_merge1: tensor<256x512xf16>,
    %w_s2_l0_q: tensor<512x512xf16>,
    %w_s2_l0_k: tensor<512x512xf16>,
    %w_s2_l0_v: tensor<512x512xf16>,
    %w_s2_l0_kt: tensor<512x196xf16>,
    %w_s2_l0_o: tensor<512x512xf16>,
    %w_s2_l0_ff_up: tensor<512x2048xf16>,
    %w_s2_l0_ff_dn: tensor<2048x512xf16>,
    %w_s2_l1_q: tensor<512x512xf16>,
    %w_s2_l1_k: tensor<512x512xf16>,
    %w_s2_l1_v: tensor<512x512xf16>,
    %w_s2_l1_kt: tensor<512x196xf16>,
    %w_s2_l1_o: tensor<512x512xf16>,
    %w_s2_l1_ff_up: tensor<512x2048xf16>,
    %w_s2_l1_ff_dn: tensor<2048x512xf16>,
    %w_s2_l2_q: tensor<512x512xf16>,
    %w_s2_l2_k: tensor<512x512xf16>,
    %w_s2_l2_v: tensor<512x512xf16>,
    %w_s2_l2_kt: tensor<512x196xf16>,
    %w_s2_l2_o: tensor<512x512xf16>,
    %w_s2_l2_ff_up: tensor<512x2048xf16>,
    %w_s2_l2_ff_dn: tensor<2048x512xf16>,
    %w_s2_l3_q: tensor<512x512xf16>,
    %w_s2_l3_k: tensor<512x512xf16>,
    %w_s2_l3_v: tensor<512x512xf16>,
    %w_s2_l3_kt: tensor<512x196xf16>,
    %w_s2_l3_o: tensor<512x512xf16>,
    %w_s2_l3_ff_up: tensor<512x2048xf16>,
    %w_s2_l3_ff_dn: tensor<2048x512xf16>,
    %w_s2_l4_q: tensor<512x512xf16>,
    %w_s2_l4_k: tensor<512x512xf16>,
    %w_s2_l4_v: tensor<512x512xf16>,
    %w_s2_l4_kt: tensor<512x196xf16>,
    %w_s2_l4_o: tensor<512x512xf16>,
    %w_s2_l4_ff_up: tensor<512x2048xf16>,
    %w_s2_l4_ff_dn: tensor<2048x512xf16>,
    %w_s2_l5_q: tensor<512x512xf16>,
    %w_s2_l5_k: tensor<512x512xf16>,
    %w_s2_l5_v: tensor<512x512xf16>,
    %w_s2_l5_kt: tensor<512x196xf16>,
    %w_s2_l5_o: tensor<512x512xf16>,
    %w_s2_l5_ff_up: tensor<512x2048xf16>,
    %w_s2_l5_ff_dn: tensor<2048x512xf16>,
    %w_s2_l6_q: tensor<512x512xf16>,
    %w_s2_l6_k: tensor<512x512xf16>,
    %w_s2_l6_v: tensor<512x512xf16>,
    %w_s2_l6_kt: tensor<512x196xf16>,
    %w_s2_l6_o: tensor<512x512xf16>,
    %w_s2_l6_ff_up: tensor<512x2048xf16>,
    %w_s2_l6_ff_dn: tensor<2048x512xf16>,
    %w_s2_l7_q: tensor<512x512xf16>,
    %w_s2_l7_k: tensor<512x512xf16>,
    %w_s2_l7_v: tensor<512x512xf16>,
    %w_s2_l7_kt: tensor<512x196xf16>,
    %w_s2_l7_o: tensor<512x512xf16>,
    %w_s2_l7_ff_up: tensor<512x2048xf16>,
    %w_s2_l7_ff_dn: tensor<2048x512xf16>,
    %w_s2_l8_q: tensor<512x512xf16>,
    %w_s2_l8_k: tensor<512x512xf16>,
    %w_s2_l8_v: tensor<512x512xf16>,
    %w_s2_l8_kt: tensor<512x196xf16>,
    %w_s2_l8_o: tensor<512x512xf16>,
    %w_s2_l8_ff_up: tensor<512x2048xf16>,
    %w_s2_l8_ff_dn: tensor<2048x512xf16>,
    %w_s2_l9_q: tensor<512x512xf16>,
    %w_s2_l9_k: tensor<512x512xf16>,
    %w_s2_l9_v: tensor<512x512xf16>,
    %w_s2_l9_kt: tensor<512x196xf16>,
    %w_s2_l9_o: tensor<512x512xf16>,
    %w_s2_l9_ff_up: tensor<512x2048xf16>,
    %w_s2_l9_ff_dn: tensor<2048x512xf16>,
    %w_s2_l10_q: tensor<512x512xf16>,
    %w_s2_l10_k: tensor<512x512xf16>,
    %w_s2_l10_v: tensor<512x512xf16>,
    %w_s2_l10_kt: tensor<512x196xf16>,
    %w_s2_l10_o: tensor<512x512xf16>,
    %w_s2_l10_ff_up: tensor<512x2048xf16>,
    %w_s2_l10_ff_dn: tensor<2048x512xf16>,
    %w_s2_l11_q: tensor<512x512xf16>,
    %w_s2_l11_k: tensor<512x512xf16>,
    %w_s2_l11_v: tensor<512x512xf16>,
    %w_s2_l11_kt: tensor<512x196xf16>,
    %w_s2_l11_o: tensor<512x512xf16>,
    %w_s2_l11_ff_up: tensor<512x2048xf16>,
    %w_s2_l11_ff_dn: tensor<2048x512xf16>,
    %w_s2_l12_q: tensor<512x512xf16>,
    %w_s2_l12_k: tensor<512x512xf16>,
    %w_s2_l12_v: tensor<512x512xf16>,
    %w_s2_l12_kt: tensor<512x196xf16>,
    %w_s2_l12_o: tensor<512x512xf16>,
    %w_s2_l12_ff_up: tensor<512x2048xf16>,
    %w_s2_l12_ff_dn: tensor<2048x512xf16>,
    %w_s2_l13_q: tensor<512x512xf16>,
    %w_s2_l13_k: tensor<512x512xf16>,
    %w_s2_l13_v: tensor<512x512xf16>,
    %w_s2_l13_kt: tensor<512x196xf16>,
    %w_s2_l13_o: tensor<512x512xf16>,
    %w_s2_l13_ff_up: tensor<512x2048xf16>,
    %w_s2_l13_ff_dn: tensor<2048x512xf16>,
    %w_s2_l14_q: tensor<512x512xf16>,
    %w_s2_l14_k: tensor<512x512xf16>,
    %w_s2_l14_v: tensor<512x512xf16>,
    %w_s2_l14_kt: tensor<512x196xf16>,
    %w_s2_l14_o: tensor<512x512xf16>,
    %w_s2_l14_ff_up: tensor<512x2048xf16>,
    %w_s2_l14_ff_dn: tensor<2048x512xf16>,
    %w_s2_l15_q: tensor<512x512xf16>,
    %w_s2_l15_k: tensor<512x512xf16>,
    %w_s2_l15_v: tensor<512x512xf16>,
    %w_s2_l15_kt: tensor<512x196xf16>,
    %w_s2_l15_o: tensor<512x512xf16>,
    %w_s2_l15_ff_up: tensor<512x2048xf16>,
    %w_s2_l15_ff_dn: tensor<2048x512xf16>,
    %w_s2_l16_q: tensor<512x512xf16>,
    %w_s2_l16_k: tensor<512x512xf16>,
    %w_s2_l16_v: tensor<512x512xf16>,
    %w_s2_l16_kt: tensor<512x196xf16>,
    %w_s2_l16_o: tensor<512x512xf16>,
    %w_s2_l16_ff_up: tensor<512x2048xf16>,
    %w_s2_l16_ff_dn: tensor<2048x512xf16>,
    %w_s2_l17_q: tensor<512x512xf16>,
    %w_s2_l17_k: tensor<512x512xf16>,
    %w_s2_l17_v: tensor<512x512xf16>,
    %w_s2_l17_kt: tensor<512x196xf16>,
    %w_s2_l17_o: tensor<512x512xf16>,
    %w_s2_l17_ff_up: tensor<512x2048xf16>,
    %w_s2_l17_ff_dn: tensor<2048x512xf16>,
    %w_merge2: tensor<512x1024xf16>,
    %w_s3_l0_q: tensor<1024x1024xf16>,
    %w_s3_l0_k: tensor<1024x1024xf16>,
    %w_s3_l0_v: tensor<1024x1024xf16>,
    %w_s3_l0_kt: tensor<1024x49xf16>,
    %w_s3_l0_o: tensor<1024x1024xf16>,
    %w_s3_l0_ff_up: tensor<1024x4096xf16>,
    %w_s3_l0_ff_dn: tensor<4096x1024xf16>,
    %w_s3_l1_q: tensor<1024x1024xf16>,
    %w_s3_l1_k: tensor<1024x1024xf16>,
    %w_s3_l1_v: tensor<1024x1024xf16>,
    %w_s3_l1_kt: tensor<1024x49xf16>,
    %w_s3_l1_o: tensor<1024x1024xf16>,
    %w_s3_l1_ff_up: tensor<1024x4096xf16>,
    %w_s3_l1_ff_dn: tensor<4096x1024xf16>,
    %w_head: tensor<1024x1000xf16>) -> tensor<49x1000xf16> {
  %cst = arith.constant 0.0 : f16

  // Patch embedding: 4x4 conv
  %init0 = tensor.empty() : tensor<1x128x56x56xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init0 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>
  %conv2 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<4> : tensor<2xi64>
  } ins(%input, %w_patch : tensor<1x3x224x224xf16>, tensor<128x3x4x4xf16>)
    outs(%fill1 : tensor<1x128x56x56xf16>) -> tensor<1x128x56x56xf16>

  // Reshape to [3136, 128]
  %seq3 = tensor.empty() : tensor<3136x128xf16>
  %fill4 = linalg.fill ins(%cst : f16) outs(%seq3 : tensor<3136x128xf16>) -> tensor<3136x128xf16>

  // === Swin Stage 0: seq=3136, dim=128 ===
  %init5 = tensor.empty() : tensor<3136x128xf16>
  %fill6 = linalg.fill ins(%cst : f16) outs(%init5 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %mm7 = linalg.matmul ins(%fill4, %w_s0_l0_q : tensor<3136x128xf16>, tensor<128x128xf16>)
                          outs(%fill6 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %init8 = tensor.empty() : tensor<3136x128xf16>
  %fill9 = linalg.fill ins(%cst : f16) outs(%init8 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %mm10 = linalg.matmul ins(%fill4, %w_s0_l0_k : tensor<3136x128xf16>, tensor<128x128xf16>)
                          outs(%fill9 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %init11 = tensor.empty() : tensor<3136x128xf16>
  %fill12 = linalg.fill ins(%cst : f16) outs(%init11 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %mm13 = linalg.matmul ins(%fill4, %w_s0_l0_v : tensor<3136x128xf16>, tensor<128x128xf16>)
                          outs(%fill12 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %init14 = tensor.empty() : tensor<3136x3136xf16>
  %fill15 = linalg.fill ins(%cst : f16) outs(%init14 : tensor<3136x3136xf16>) -> tensor<3136x3136xf16>
  %mm16 = linalg.matmul ins(%mm7, %w_s0_l0_kt : tensor<3136x128xf16>, tensor<128x3136xf16>)
                          outs(%fill15 : tensor<3136x3136xf16>) -> tensor<3136x3136xf16>
  %empty17 = tensor.empty() : tensor<3136x3136xf16>
  %relu18 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm16 : tensor<3136x3136xf16>)
    outs(%empty17 : tensor<3136x3136xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<3136x3136xf16>
  %init19 = tensor.empty() : tensor<3136x128xf16>
  %fill20 = linalg.fill ins(%cst : f16) outs(%init19 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %mm21 = linalg.matmul ins(%relu18, %mm13 : tensor<3136x3136xf16>, tensor<3136x128xf16>)
                          outs(%fill20 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %init22 = tensor.empty() : tensor<3136x128xf16>
  %fill23 = linalg.fill ins(%cst : f16) outs(%init22 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %mm24 = linalg.matmul ins(%mm21, %w_s0_l0_o : tensor<3136x128xf16>, tensor<128x128xf16>)
                          outs(%fill23 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %empty25 = tensor.empty() : tensor<3136x128xf16>
  %add26 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm24, %fill4 : tensor<3136x128xf16>, tensor<3136x128xf16>)
    outs(%empty25 : tensor<3136x128xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<3136x128xf16>
  %init27 = tensor.empty() : tensor<3136x512xf16>
  %fill28 = linalg.fill ins(%cst : f16) outs(%init27 : tensor<3136x512xf16>) -> tensor<3136x512xf16>
  %mm29 = linalg.matmul ins(%add26, %w_s0_l0_ff_up : tensor<3136x128xf16>, tensor<128x512xf16>)
                          outs(%fill28 : tensor<3136x512xf16>) -> tensor<3136x512xf16>
  %empty30 = tensor.empty() : tensor<3136x512xf16>
  %relu31 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm29 : tensor<3136x512xf16>)
    outs(%empty30 : tensor<3136x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<3136x512xf16>
  %init32 = tensor.empty() : tensor<3136x128xf16>
  %fill33 = linalg.fill ins(%cst : f16) outs(%init32 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %mm34 = linalg.matmul ins(%relu31, %w_s0_l0_ff_dn : tensor<3136x512xf16>, tensor<512x128xf16>)
                          outs(%fill33 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %empty35 = tensor.empty() : tensor<3136x128xf16>
  %add36 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm34, %add26 : tensor<3136x128xf16>, tensor<3136x128xf16>)
    outs(%empty35 : tensor<3136x128xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<3136x128xf16>
  %init37 = tensor.empty() : tensor<3136x128xf16>
  %fill38 = linalg.fill ins(%cst : f16) outs(%init37 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %mm39 = linalg.matmul ins(%add36, %w_s0_l1_q : tensor<3136x128xf16>, tensor<128x128xf16>)
                          outs(%fill38 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %init40 = tensor.empty() : tensor<3136x128xf16>
  %fill41 = linalg.fill ins(%cst : f16) outs(%init40 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %mm42 = linalg.matmul ins(%add36, %w_s0_l1_k : tensor<3136x128xf16>, tensor<128x128xf16>)
                          outs(%fill41 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %init43 = tensor.empty() : tensor<3136x128xf16>
  %fill44 = linalg.fill ins(%cst : f16) outs(%init43 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %mm45 = linalg.matmul ins(%add36, %w_s0_l1_v : tensor<3136x128xf16>, tensor<128x128xf16>)
                          outs(%fill44 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %init46 = tensor.empty() : tensor<3136x3136xf16>
  %fill47 = linalg.fill ins(%cst : f16) outs(%init46 : tensor<3136x3136xf16>) -> tensor<3136x3136xf16>
  %mm48 = linalg.matmul ins(%mm39, %w_s0_l1_kt : tensor<3136x128xf16>, tensor<128x3136xf16>)
                          outs(%fill47 : tensor<3136x3136xf16>) -> tensor<3136x3136xf16>
  %empty49 = tensor.empty() : tensor<3136x3136xf16>
  %relu50 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm48 : tensor<3136x3136xf16>)
    outs(%empty49 : tensor<3136x3136xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<3136x3136xf16>
  %init51 = tensor.empty() : tensor<3136x128xf16>
  %fill52 = linalg.fill ins(%cst : f16) outs(%init51 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %mm53 = linalg.matmul ins(%relu50, %mm45 : tensor<3136x3136xf16>, tensor<3136x128xf16>)
                          outs(%fill52 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %init54 = tensor.empty() : tensor<3136x128xf16>
  %fill55 = linalg.fill ins(%cst : f16) outs(%init54 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %mm56 = linalg.matmul ins(%mm53, %w_s0_l1_o : tensor<3136x128xf16>, tensor<128x128xf16>)
                          outs(%fill55 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %empty57 = tensor.empty() : tensor<3136x128xf16>
  %add58 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm56, %add36 : tensor<3136x128xf16>, tensor<3136x128xf16>)
    outs(%empty57 : tensor<3136x128xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<3136x128xf16>
  %init59 = tensor.empty() : tensor<3136x512xf16>
  %fill60 = linalg.fill ins(%cst : f16) outs(%init59 : tensor<3136x512xf16>) -> tensor<3136x512xf16>
  %mm61 = linalg.matmul ins(%add58, %w_s0_l1_ff_up : tensor<3136x128xf16>, tensor<128x512xf16>)
                          outs(%fill60 : tensor<3136x512xf16>) -> tensor<3136x512xf16>
  %empty62 = tensor.empty() : tensor<3136x512xf16>
  %relu63 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm61 : tensor<3136x512xf16>)
    outs(%empty62 : tensor<3136x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<3136x512xf16>
  %init64 = tensor.empty() : tensor<3136x128xf16>
  %fill65 = linalg.fill ins(%cst : f16) outs(%init64 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %mm66 = linalg.matmul ins(%relu63, %w_s0_l1_ff_dn : tensor<3136x512xf16>, tensor<512x128xf16>)
                          outs(%fill65 : tensor<3136x128xf16>) -> tensor<3136x128xf16>
  %empty67 = tensor.empty() : tensor<3136x128xf16>
  %add68 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm66, %add58 : tensor<3136x128xf16>, tensor<3136x128xf16>)
    outs(%empty67 : tensor<3136x128xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<3136x128xf16>

  // Patch merging: [3136,128] -> [784,256]
  %init69 = tensor.empty() : tensor<3136x256xf16>
  %fill70 = linalg.fill ins(%cst : f16) outs(%init69 : tensor<3136x256xf16>) -> tensor<3136x256xf16>
  %mm71 = linalg.matmul ins(%add68, %w_merge0 : tensor<3136x128xf16>, tensor<128x256xf16>)
                          outs(%fill70 : tensor<3136x256xf16>) -> tensor<3136x256xf16>
  %merge_reshape72 = tensor.empty() : tensor<784x256xf16>
  %fill73 = linalg.fill ins(%cst : f16) outs(%merge_reshape72 : tensor<784x256xf16>) -> tensor<784x256xf16>

  // === Swin Stage 1: seq=784, dim=256 ===
  %init74 = tensor.empty() : tensor<784x256xf16>
  %fill75 = linalg.fill ins(%cst : f16) outs(%init74 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %mm76 = linalg.matmul ins(%fill73, %w_s1_l0_q : tensor<784x256xf16>, tensor<256x256xf16>)
                          outs(%fill75 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %init77 = tensor.empty() : tensor<784x256xf16>
  %fill78 = linalg.fill ins(%cst : f16) outs(%init77 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %mm79 = linalg.matmul ins(%fill73, %w_s1_l0_k : tensor<784x256xf16>, tensor<256x256xf16>)
                          outs(%fill78 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %init80 = tensor.empty() : tensor<784x256xf16>
  %fill81 = linalg.fill ins(%cst : f16) outs(%init80 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %mm82 = linalg.matmul ins(%fill73, %w_s1_l0_v : tensor<784x256xf16>, tensor<256x256xf16>)
                          outs(%fill81 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %init83 = tensor.empty() : tensor<784x784xf16>
  %fill84 = linalg.fill ins(%cst : f16) outs(%init83 : tensor<784x784xf16>) -> tensor<784x784xf16>
  %mm85 = linalg.matmul ins(%mm76, %w_s1_l0_kt : tensor<784x256xf16>, tensor<256x784xf16>)
                          outs(%fill84 : tensor<784x784xf16>) -> tensor<784x784xf16>
  %empty86 = tensor.empty() : tensor<784x784xf16>
  %relu87 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm85 : tensor<784x784xf16>)
    outs(%empty86 : tensor<784x784xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<784x784xf16>
  %init88 = tensor.empty() : tensor<784x256xf16>
  %fill89 = linalg.fill ins(%cst : f16) outs(%init88 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %mm90 = linalg.matmul ins(%relu87, %mm82 : tensor<784x784xf16>, tensor<784x256xf16>)
                          outs(%fill89 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %init91 = tensor.empty() : tensor<784x256xf16>
  %fill92 = linalg.fill ins(%cst : f16) outs(%init91 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %mm93 = linalg.matmul ins(%mm90, %w_s1_l0_o : tensor<784x256xf16>, tensor<256x256xf16>)
                          outs(%fill92 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %empty94 = tensor.empty() : tensor<784x256xf16>
  %add95 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm93, %fill73 : tensor<784x256xf16>, tensor<784x256xf16>)
    outs(%empty94 : tensor<784x256xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<784x256xf16>
  %init96 = tensor.empty() : tensor<784x1024xf16>
  %fill97 = linalg.fill ins(%cst : f16) outs(%init96 : tensor<784x1024xf16>) -> tensor<784x1024xf16>
  %mm98 = linalg.matmul ins(%add95, %w_s1_l0_ff_up : tensor<784x256xf16>, tensor<256x1024xf16>)
                          outs(%fill97 : tensor<784x1024xf16>) -> tensor<784x1024xf16>
  %empty99 = tensor.empty() : tensor<784x1024xf16>
  %relu100 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm98 : tensor<784x1024xf16>)
    outs(%empty99 : tensor<784x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<784x1024xf16>
  %init101 = tensor.empty() : tensor<784x256xf16>
  %fill102 = linalg.fill ins(%cst : f16) outs(%init101 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %mm103 = linalg.matmul ins(%relu100, %w_s1_l0_ff_dn : tensor<784x1024xf16>, tensor<1024x256xf16>)
                          outs(%fill102 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %empty104 = tensor.empty() : tensor<784x256xf16>
  %add105 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm103, %add95 : tensor<784x256xf16>, tensor<784x256xf16>)
    outs(%empty104 : tensor<784x256xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<784x256xf16>
  %init106 = tensor.empty() : tensor<784x256xf16>
  %fill107 = linalg.fill ins(%cst : f16) outs(%init106 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %mm108 = linalg.matmul ins(%add105, %w_s1_l1_q : tensor<784x256xf16>, tensor<256x256xf16>)
                          outs(%fill107 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %init109 = tensor.empty() : tensor<784x256xf16>
  %fill110 = linalg.fill ins(%cst : f16) outs(%init109 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %mm111 = linalg.matmul ins(%add105, %w_s1_l1_k : tensor<784x256xf16>, tensor<256x256xf16>)
                          outs(%fill110 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %init112 = tensor.empty() : tensor<784x256xf16>
  %fill113 = linalg.fill ins(%cst : f16) outs(%init112 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %mm114 = linalg.matmul ins(%add105, %w_s1_l1_v : tensor<784x256xf16>, tensor<256x256xf16>)
                          outs(%fill113 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %init115 = tensor.empty() : tensor<784x784xf16>
  %fill116 = linalg.fill ins(%cst : f16) outs(%init115 : tensor<784x784xf16>) -> tensor<784x784xf16>
  %mm117 = linalg.matmul ins(%mm108, %w_s1_l1_kt : tensor<784x256xf16>, tensor<256x784xf16>)
                          outs(%fill116 : tensor<784x784xf16>) -> tensor<784x784xf16>
  %empty118 = tensor.empty() : tensor<784x784xf16>
  %relu119 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm117 : tensor<784x784xf16>)
    outs(%empty118 : tensor<784x784xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<784x784xf16>
  %init120 = tensor.empty() : tensor<784x256xf16>
  %fill121 = linalg.fill ins(%cst : f16) outs(%init120 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %mm122 = linalg.matmul ins(%relu119, %mm114 : tensor<784x784xf16>, tensor<784x256xf16>)
                          outs(%fill121 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %init123 = tensor.empty() : tensor<784x256xf16>
  %fill124 = linalg.fill ins(%cst : f16) outs(%init123 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %mm125 = linalg.matmul ins(%mm122, %w_s1_l1_o : tensor<784x256xf16>, tensor<256x256xf16>)
                          outs(%fill124 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %empty126 = tensor.empty() : tensor<784x256xf16>
  %add127 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm125, %add105 : tensor<784x256xf16>, tensor<784x256xf16>)
    outs(%empty126 : tensor<784x256xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<784x256xf16>
  %init128 = tensor.empty() : tensor<784x1024xf16>
  %fill129 = linalg.fill ins(%cst : f16) outs(%init128 : tensor<784x1024xf16>) -> tensor<784x1024xf16>
  %mm130 = linalg.matmul ins(%add127, %w_s1_l1_ff_up : tensor<784x256xf16>, tensor<256x1024xf16>)
                          outs(%fill129 : tensor<784x1024xf16>) -> tensor<784x1024xf16>
  %empty131 = tensor.empty() : tensor<784x1024xf16>
  %relu132 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm130 : tensor<784x1024xf16>)
    outs(%empty131 : tensor<784x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<784x1024xf16>
  %init133 = tensor.empty() : tensor<784x256xf16>
  %fill134 = linalg.fill ins(%cst : f16) outs(%init133 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %mm135 = linalg.matmul ins(%relu132, %w_s1_l1_ff_dn : tensor<784x1024xf16>, tensor<1024x256xf16>)
                          outs(%fill134 : tensor<784x256xf16>) -> tensor<784x256xf16>
  %empty136 = tensor.empty() : tensor<784x256xf16>
  %add137 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm135, %add127 : tensor<784x256xf16>, tensor<784x256xf16>)
    outs(%empty136 : tensor<784x256xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<784x256xf16>

  // Patch merging: [784,256] -> [196,512]
  %init138 = tensor.empty() : tensor<784x512xf16>
  %fill139 = linalg.fill ins(%cst : f16) outs(%init138 : tensor<784x512xf16>) -> tensor<784x512xf16>
  %mm140 = linalg.matmul ins(%add137, %w_merge1 : tensor<784x256xf16>, tensor<256x512xf16>)
                          outs(%fill139 : tensor<784x512xf16>) -> tensor<784x512xf16>
  %merge_reshape141 = tensor.empty() : tensor<196x512xf16>
  %fill142 = linalg.fill ins(%cst : f16) outs(%merge_reshape141 : tensor<196x512xf16>) -> tensor<196x512xf16>

  // === Swin Stage 2: seq=196, dim=512 ===
  %init143 = tensor.empty() : tensor<196x512xf16>
  %fill144 = linalg.fill ins(%cst : f16) outs(%init143 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm145 = linalg.matmul ins(%fill142, %w_s2_l0_q : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill144 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init146 = tensor.empty() : tensor<196x512xf16>
  %fill147 = linalg.fill ins(%cst : f16) outs(%init146 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm148 = linalg.matmul ins(%fill142, %w_s2_l0_k : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill147 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init149 = tensor.empty() : tensor<196x512xf16>
  %fill150 = linalg.fill ins(%cst : f16) outs(%init149 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm151 = linalg.matmul ins(%fill142, %w_s2_l0_v : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill150 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init152 = tensor.empty() : tensor<196x196xf16>
  %fill153 = linalg.fill ins(%cst : f16) outs(%init152 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm154 = linalg.matmul ins(%mm145, %w_s2_l0_kt : tensor<196x512xf16>, tensor<512x196xf16>)
                          outs(%fill153 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty155 = tensor.empty() : tensor<196x196xf16>
  %relu156 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm154 : tensor<196x196xf16>)
    outs(%empty155 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init157 = tensor.empty() : tensor<196x512xf16>
  %fill158 = linalg.fill ins(%cst : f16) outs(%init157 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm159 = linalg.matmul ins(%relu156, %mm151 : tensor<196x196xf16>, tensor<196x512xf16>)
                          outs(%fill158 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init160 = tensor.empty() : tensor<196x512xf16>
  %fill161 = linalg.fill ins(%cst : f16) outs(%init160 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm162 = linalg.matmul ins(%mm159, %w_s2_l0_o : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill161 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty163 = tensor.empty() : tensor<196x512xf16>
  %add164 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm162, %fill142 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty163 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init165 = tensor.empty() : tensor<196x2048xf16>
  %fill166 = linalg.fill ins(%cst : f16) outs(%init165 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %mm167 = linalg.matmul ins(%add164, %w_s2_l0_ff_up : tensor<196x512xf16>, tensor<512x2048xf16>)
                          outs(%fill166 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %empty168 = tensor.empty() : tensor<196x2048xf16>
  %relu169 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm167 : tensor<196x2048xf16>)
    outs(%empty168 : tensor<196x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x2048xf16>
  %init170 = tensor.empty() : tensor<196x512xf16>
  %fill171 = linalg.fill ins(%cst : f16) outs(%init170 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm172 = linalg.matmul ins(%relu169, %w_s2_l0_ff_dn : tensor<196x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill171 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty173 = tensor.empty() : tensor<196x512xf16>
  %add174 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm172, %add164 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty173 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init175 = tensor.empty() : tensor<196x512xf16>
  %fill176 = linalg.fill ins(%cst : f16) outs(%init175 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm177 = linalg.matmul ins(%add174, %w_s2_l1_q : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill176 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init178 = tensor.empty() : tensor<196x512xf16>
  %fill179 = linalg.fill ins(%cst : f16) outs(%init178 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm180 = linalg.matmul ins(%add174, %w_s2_l1_k : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill179 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init181 = tensor.empty() : tensor<196x512xf16>
  %fill182 = linalg.fill ins(%cst : f16) outs(%init181 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm183 = linalg.matmul ins(%add174, %w_s2_l1_v : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill182 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init184 = tensor.empty() : tensor<196x196xf16>
  %fill185 = linalg.fill ins(%cst : f16) outs(%init184 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm186 = linalg.matmul ins(%mm177, %w_s2_l1_kt : tensor<196x512xf16>, tensor<512x196xf16>)
                          outs(%fill185 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty187 = tensor.empty() : tensor<196x196xf16>
  %relu188 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm186 : tensor<196x196xf16>)
    outs(%empty187 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init189 = tensor.empty() : tensor<196x512xf16>
  %fill190 = linalg.fill ins(%cst : f16) outs(%init189 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm191 = linalg.matmul ins(%relu188, %mm183 : tensor<196x196xf16>, tensor<196x512xf16>)
                          outs(%fill190 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init192 = tensor.empty() : tensor<196x512xf16>
  %fill193 = linalg.fill ins(%cst : f16) outs(%init192 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm194 = linalg.matmul ins(%mm191, %w_s2_l1_o : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill193 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty195 = tensor.empty() : tensor<196x512xf16>
  %add196 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm194, %add174 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty195 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init197 = tensor.empty() : tensor<196x2048xf16>
  %fill198 = linalg.fill ins(%cst : f16) outs(%init197 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %mm199 = linalg.matmul ins(%add196, %w_s2_l1_ff_up : tensor<196x512xf16>, tensor<512x2048xf16>)
                          outs(%fill198 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %empty200 = tensor.empty() : tensor<196x2048xf16>
  %relu201 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm199 : tensor<196x2048xf16>)
    outs(%empty200 : tensor<196x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x2048xf16>
  %init202 = tensor.empty() : tensor<196x512xf16>
  %fill203 = linalg.fill ins(%cst : f16) outs(%init202 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm204 = linalg.matmul ins(%relu201, %w_s2_l1_ff_dn : tensor<196x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill203 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty205 = tensor.empty() : tensor<196x512xf16>
  %add206 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm204, %add196 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty205 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init207 = tensor.empty() : tensor<196x512xf16>
  %fill208 = linalg.fill ins(%cst : f16) outs(%init207 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm209 = linalg.matmul ins(%add206, %w_s2_l2_q : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill208 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init210 = tensor.empty() : tensor<196x512xf16>
  %fill211 = linalg.fill ins(%cst : f16) outs(%init210 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm212 = linalg.matmul ins(%add206, %w_s2_l2_k : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill211 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init213 = tensor.empty() : tensor<196x512xf16>
  %fill214 = linalg.fill ins(%cst : f16) outs(%init213 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm215 = linalg.matmul ins(%add206, %w_s2_l2_v : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill214 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init216 = tensor.empty() : tensor<196x196xf16>
  %fill217 = linalg.fill ins(%cst : f16) outs(%init216 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm218 = linalg.matmul ins(%mm209, %w_s2_l2_kt : tensor<196x512xf16>, tensor<512x196xf16>)
                          outs(%fill217 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty219 = tensor.empty() : tensor<196x196xf16>
  %relu220 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm218 : tensor<196x196xf16>)
    outs(%empty219 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init221 = tensor.empty() : tensor<196x512xf16>
  %fill222 = linalg.fill ins(%cst : f16) outs(%init221 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm223 = linalg.matmul ins(%relu220, %mm215 : tensor<196x196xf16>, tensor<196x512xf16>)
                          outs(%fill222 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init224 = tensor.empty() : tensor<196x512xf16>
  %fill225 = linalg.fill ins(%cst : f16) outs(%init224 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm226 = linalg.matmul ins(%mm223, %w_s2_l2_o : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill225 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty227 = tensor.empty() : tensor<196x512xf16>
  %add228 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm226, %add206 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty227 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init229 = tensor.empty() : tensor<196x2048xf16>
  %fill230 = linalg.fill ins(%cst : f16) outs(%init229 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %mm231 = linalg.matmul ins(%add228, %w_s2_l2_ff_up : tensor<196x512xf16>, tensor<512x2048xf16>)
                          outs(%fill230 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %empty232 = tensor.empty() : tensor<196x2048xf16>
  %relu233 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm231 : tensor<196x2048xf16>)
    outs(%empty232 : tensor<196x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x2048xf16>
  %init234 = tensor.empty() : tensor<196x512xf16>
  %fill235 = linalg.fill ins(%cst : f16) outs(%init234 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm236 = linalg.matmul ins(%relu233, %w_s2_l2_ff_dn : tensor<196x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill235 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty237 = tensor.empty() : tensor<196x512xf16>
  %add238 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm236, %add228 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty237 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init239 = tensor.empty() : tensor<196x512xf16>
  %fill240 = linalg.fill ins(%cst : f16) outs(%init239 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm241 = linalg.matmul ins(%add238, %w_s2_l3_q : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill240 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init242 = tensor.empty() : tensor<196x512xf16>
  %fill243 = linalg.fill ins(%cst : f16) outs(%init242 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm244 = linalg.matmul ins(%add238, %w_s2_l3_k : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill243 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init245 = tensor.empty() : tensor<196x512xf16>
  %fill246 = linalg.fill ins(%cst : f16) outs(%init245 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm247 = linalg.matmul ins(%add238, %w_s2_l3_v : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill246 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init248 = tensor.empty() : tensor<196x196xf16>
  %fill249 = linalg.fill ins(%cst : f16) outs(%init248 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm250 = linalg.matmul ins(%mm241, %w_s2_l3_kt : tensor<196x512xf16>, tensor<512x196xf16>)
                          outs(%fill249 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty251 = tensor.empty() : tensor<196x196xf16>
  %relu252 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm250 : tensor<196x196xf16>)
    outs(%empty251 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init253 = tensor.empty() : tensor<196x512xf16>
  %fill254 = linalg.fill ins(%cst : f16) outs(%init253 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm255 = linalg.matmul ins(%relu252, %mm247 : tensor<196x196xf16>, tensor<196x512xf16>)
                          outs(%fill254 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init256 = tensor.empty() : tensor<196x512xf16>
  %fill257 = linalg.fill ins(%cst : f16) outs(%init256 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm258 = linalg.matmul ins(%mm255, %w_s2_l3_o : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill257 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty259 = tensor.empty() : tensor<196x512xf16>
  %add260 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm258, %add238 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty259 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init261 = tensor.empty() : tensor<196x2048xf16>
  %fill262 = linalg.fill ins(%cst : f16) outs(%init261 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %mm263 = linalg.matmul ins(%add260, %w_s2_l3_ff_up : tensor<196x512xf16>, tensor<512x2048xf16>)
                          outs(%fill262 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %empty264 = tensor.empty() : tensor<196x2048xf16>
  %relu265 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm263 : tensor<196x2048xf16>)
    outs(%empty264 : tensor<196x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x2048xf16>
  %init266 = tensor.empty() : tensor<196x512xf16>
  %fill267 = linalg.fill ins(%cst : f16) outs(%init266 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm268 = linalg.matmul ins(%relu265, %w_s2_l3_ff_dn : tensor<196x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill267 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty269 = tensor.empty() : tensor<196x512xf16>
  %add270 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm268, %add260 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty269 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init271 = tensor.empty() : tensor<196x512xf16>
  %fill272 = linalg.fill ins(%cst : f16) outs(%init271 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm273 = linalg.matmul ins(%add270, %w_s2_l4_q : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill272 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init274 = tensor.empty() : tensor<196x512xf16>
  %fill275 = linalg.fill ins(%cst : f16) outs(%init274 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm276 = linalg.matmul ins(%add270, %w_s2_l4_k : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill275 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init277 = tensor.empty() : tensor<196x512xf16>
  %fill278 = linalg.fill ins(%cst : f16) outs(%init277 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm279 = linalg.matmul ins(%add270, %w_s2_l4_v : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill278 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init280 = tensor.empty() : tensor<196x196xf16>
  %fill281 = linalg.fill ins(%cst : f16) outs(%init280 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm282 = linalg.matmul ins(%mm273, %w_s2_l4_kt : tensor<196x512xf16>, tensor<512x196xf16>)
                          outs(%fill281 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty283 = tensor.empty() : tensor<196x196xf16>
  %relu284 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm282 : tensor<196x196xf16>)
    outs(%empty283 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init285 = tensor.empty() : tensor<196x512xf16>
  %fill286 = linalg.fill ins(%cst : f16) outs(%init285 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm287 = linalg.matmul ins(%relu284, %mm279 : tensor<196x196xf16>, tensor<196x512xf16>)
                          outs(%fill286 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init288 = tensor.empty() : tensor<196x512xf16>
  %fill289 = linalg.fill ins(%cst : f16) outs(%init288 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm290 = linalg.matmul ins(%mm287, %w_s2_l4_o : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill289 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty291 = tensor.empty() : tensor<196x512xf16>
  %add292 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm290, %add270 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty291 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init293 = tensor.empty() : tensor<196x2048xf16>
  %fill294 = linalg.fill ins(%cst : f16) outs(%init293 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %mm295 = linalg.matmul ins(%add292, %w_s2_l4_ff_up : tensor<196x512xf16>, tensor<512x2048xf16>)
                          outs(%fill294 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %empty296 = tensor.empty() : tensor<196x2048xf16>
  %relu297 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm295 : tensor<196x2048xf16>)
    outs(%empty296 : tensor<196x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x2048xf16>
  %init298 = tensor.empty() : tensor<196x512xf16>
  %fill299 = linalg.fill ins(%cst : f16) outs(%init298 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm300 = linalg.matmul ins(%relu297, %w_s2_l4_ff_dn : tensor<196x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill299 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty301 = tensor.empty() : tensor<196x512xf16>
  %add302 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm300, %add292 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty301 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init303 = tensor.empty() : tensor<196x512xf16>
  %fill304 = linalg.fill ins(%cst : f16) outs(%init303 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm305 = linalg.matmul ins(%add302, %w_s2_l5_q : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill304 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init306 = tensor.empty() : tensor<196x512xf16>
  %fill307 = linalg.fill ins(%cst : f16) outs(%init306 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm308 = linalg.matmul ins(%add302, %w_s2_l5_k : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill307 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init309 = tensor.empty() : tensor<196x512xf16>
  %fill310 = linalg.fill ins(%cst : f16) outs(%init309 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm311 = linalg.matmul ins(%add302, %w_s2_l5_v : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill310 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init312 = tensor.empty() : tensor<196x196xf16>
  %fill313 = linalg.fill ins(%cst : f16) outs(%init312 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm314 = linalg.matmul ins(%mm305, %w_s2_l5_kt : tensor<196x512xf16>, tensor<512x196xf16>)
                          outs(%fill313 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty315 = tensor.empty() : tensor<196x196xf16>
  %relu316 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm314 : tensor<196x196xf16>)
    outs(%empty315 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init317 = tensor.empty() : tensor<196x512xf16>
  %fill318 = linalg.fill ins(%cst : f16) outs(%init317 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm319 = linalg.matmul ins(%relu316, %mm311 : tensor<196x196xf16>, tensor<196x512xf16>)
                          outs(%fill318 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init320 = tensor.empty() : tensor<196x512xf16>
  %fill321 = linalg.fill ins(%cst : f16) outs(%init320 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm322 = linalg.matmul ins(%mm319, %w_s2_l5_o : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill321 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty323 = tensor.empty() : tensor<196x512xf16>
  %add324 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm322, %add302 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty323 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init325 = tensor.empty() : tensor<196x2048xf16>
  %fill326 = linalg.fill ins(%cst : f16) outs(%init325 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %mm327 = linalg.matmul ins(%add324, %w_s2_l5_ff_up : tensor<196x512xf16>, tensor<512x2048xf16>)
                          outs(%fill326 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %empty328 = tensor.empty() : tensor<196x2048xf16>
  %relu329 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm327 : tensor<196x2048xf16>)
    outs(%empty328 : tensor<196x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x2048xf16>
  %init330 = tensor.empty() : tensor<196x512xf16>
  %fill331 = linalg.fill ins(%cst : f16) outs(%init330 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm332 = linalg.matmul ins(%relu329, %w_s2_l5_ff_dn : tensor<196x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill331 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty333 = tensor.empty() : tensor<196x512xf16>
  %add334 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm332, %add324 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty333 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init335 = tensor.empty() : tensor<196x512xf16>
  %fill336 = linalg.fill ins(%cst : f16) outs(%init335 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm337 = linalg.matmul ins(%add334, %w_s2_l6_q : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill336 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init338 = tensor.empty() : tensor<196x512xf16>
  %fill339 = linalg.fill ins(%cst : f16) outs(%init338 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm340 = linalg.matmul ins(%add334, %w_s2_l6_k : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill339 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init341 = tensor.empty() : tensor<196x512xf16>
  %fill342 = linalg.fill ins(%cst : f16) outs(%init341 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm343 = linalg.matmul ins(%add334, %w_s2_l6_v : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill342 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init344 = tensor.empty() : tensor<196x196xf16>
  %fill345 = linalg.fill ins(%cst : f16) outs(%init344 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm346 = linalg.matmul ins(%mm337, %w_s2_l6_kt : tensor<196x512xf16>, tensor<512x196xf16>)
                          outs(%fill345 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty347 = tensor.empty() : tensor<196x196xf16>
  %relu348 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm346 : tensor<196x196xf16>)
    outs(%empty347 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init349 = tensor.empty() : tensor<196x512xf16>
  %fill350 = linalg.fill ins(%cst : f16) outs(%init349 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm351 = linalg.matmul ins(%relu348, %mm343 : tensor<196x196xf16>, tensor<196x512xf16>)
                          outs(%fill350 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init352 = tensor.empty() : tensor<196x512xf16>
  %fill353 = linalg.fill ins(%cst : f16) outs(%init352 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm354 = linalg.matmul ins(%mm351, %w_s2_l6_o : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill353 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty355 = tensor.empty() : tensor<196x512xf16>
  %add356 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm354, %add334 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty355 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init357 = tensor.empty() : tensor<196x2048xf16>
  %fill358 = linalg.fill ins(%cst : f16) outs(%init357 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %mm359 = linalg.matmul ins(%add356, %w_s2_l6_ff_up : tensor<196x512xf16>, tensor<512x2048xf16>)
                          outs(%fill358 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %empty360 = tensor.empty() : tensor<196x2048xf16>
  %relu361 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm359 : tensor<196x2048xf16>)
    outs(%empty360 : tensor<196x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x2048xf16>
  %init362 = tensor.empty() : tensor<196x512xf16>
  %fill363 = linalg.fill ins(%cst : f16) outs(%init362 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm364 = linalg.matmul ins(%relu361, %w_s2_l6_ff_dn : tensor<196x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill363 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty365 = tensor.empty() : tensor<196x512xf16>
  %add366 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm364, %add356 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty365 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init367 = tensor.empty() : tensor<196x512xf16>
  %fill368 = linalg.fill ins(%cst : f16) outs(%init367 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm369 = linalg.matmul ins(%add366, %w_s2_l7_q : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill368 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init370 = tensor.empty() : tensor<196x512xf16>
  %fill371 = linalg.fill ins(%cst : f16) outs(%init370 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm372 = linalg.matmul ins(%add366, %w_s2_l7_k : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill371 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init373 = tensor.empty() : tensor<196x512xf16>
  %fill374 = linalg.fill ins(%cst : f16) outs(%init373 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm375 = linalg.matmul ins(%add366, %w_s2_l7_v : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill374 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init376 = tensor.empty() : tensor<196x196xf16>
  %fill377 = linalg.fill ins(%cst : f16) outs(%init376 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm378 = linalg.matmul ins(%mm369, %w_s2_l7_kt : tensor<196x512xf16>, tensor<512x196xf16>)
                          outs(%fill377 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty379 = tensor.empty() : tensor<196x196xf16>
  %relu380 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm378 : tensor<196x196xf16>)
    outs(%empty379 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init381 = tensor.empty() : tensor<196x512xf16>
  %fill382 = linalg.fill ins(%cst : f16) outs(%init381 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm383 = linalg.matmul ins(%relu380, %mm375 : tensor<196x196xf16>, tensor<196x512xf16>)
                          outs(%fill382 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init384 = tensor.empty() : tensor<196x512xf16>
  %fill385 = linalg.fill ins(%cst : f16) outs(%init384 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm386 = linalg.matmul ins(%mm383, %w_s2_l7_o : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill385 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty387 = tensor.empty() : tensor<196x512xf16>
  %add388 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm386, %add366 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty387 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init389 = tensor.empty() : tensor<196x2048xf16>
  %fill390 = linalg.fill ins(%cst : f16) outs(%init389 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %mm391 = linalg.matmul ins(%add388, %w_s2_l7_ff_up : tensor<196x512xf16>, tensor<512x2048xf16>)
                          outs(%fill390 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %empty392 = tensor.empty() : tensor<196x2048xf16>
  %relu393 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm391 : tensor<196x2048xf16>)
    outs(%empty392 : tensor<196x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x2048xf16>
  %init394 = tensor.empty() : tensor<196x512xf16>
  %fill395 = linalg.fill ins(%cst : f16) outs(%init394 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm396 = linalg.matmul ins(%relu393, %w_s2_l7_ff_dn : tensor<196x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill395 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty397 = tensor.empty() : tensor<196x512xf16>
  %add398 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm396, %add388 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty397 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init399 = tensor.empty() : tensor<196x512xf16>
  %fill400 = linalg.fill ins(%cst : f16) outs(%init399 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm401 = linalg.matmul ins(%add398, %w_s2_l8_q : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill400 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init402 = tensor.empty() : tensor<196x512xf16>
  %fill403 = linalg.fill ins(%cst : f16) outs(%init402 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm404 = linalg.matmul ins(%add398, %w_s2_l8_k : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill403 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init405 = tensor.empty() : tensor<196x512xf16>
  %fill406 = linalg.fill ins(%cst : f16) outs(%init405 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm407 = linalg.matmul ins(%add398, %w_s2_l8_v : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill406 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init408 = tensor.empty() : tensor<196x196xf16>
  %fill409 = linalg.fill ins(%cst : f16) outs(%init408 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm410 = linalg.matmul ins(%mm401, %w_s2_l8_kt : tensor<196x512xf16>, tensor<512x196xf16>)
                          outs(%fill409 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty411 = tensor.empty() : tensor<196x196xf16>
  %relu412 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm410 : tensor<196x196xf16>)
    outs(%empty411 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init413 = tensor.empty() : tensor<196x512xf16>
  %fill414 = linalg.fill ins(%cst : f16) outs(%init413 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm415 = linalg.matmul ins(%relu412, %mm407 : tensor<196x196xf16>, tensor<196x512xf16>)
                          outs(%fill414 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init416 = tensor.empty() : tensor<196x512xf16>
  %fill417 = linalg.fill ins(%cst : f16) outs(%init416 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm418 = linalg.matmul ins(%mm415, %w_s2_l8_o : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill417 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty419 = tensor.empty() : tensor<196x512xf16>
  %add420 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm418, %add398 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty419 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init421 = tensor.empty() : tensor<196x2048xf16>
  %fill422 = linalg.fill ins(%cst : f16) outs(%init421 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %mm423 = linalg.matmul ins(%add420, %w_s2_l8_ff_up : tensor<196x512xf16>, tensor<512x2048xf16>)
                          outs(%fill422 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %empty424 = tensor.empty() : tensor<196x2048xf16>
  %relu425 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm423 : tensor<196x2048xf16>)
    outs(%empty424 : tensor<196x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x2048xf16>
  %init426 = tensor.empty() : tensor<196x512xf16>
  %fill427 = linalg.fill ins(%cst : f16) outs(%init426 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm428 = linalg.matmul ins(%relu425, %w_s2_l8_ff_dn : tensor<196x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill427 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty429 = tensor.empty() : tensor<196x512xf16>
  %add430 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm428, %add420 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty429 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init431 = tensor.empty() : tensor<196x512xf16>
  %fill432 = linalg.fill ins(%cst : f16) outs(%init431 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm433 = linalg.matmul ins(%add430, %w_s2_l9_q : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill432 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init434 = tensor.empty() : tensor<196x512xf16>
  %fill435 = linalg.fill ins(%cst : f16) outs(%init434 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm436 = linalg.matmul ins(%add430, %w_s2_l9_k : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill435 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init437 = tensor.empty() : tensor<196x512xf16>
  %fill438 = linalg.fill ins(%cst : f16) outs(%init437 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm439 = linalg.matmul ins(%add430, %w_s2_l9_v : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill438 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init440 = tensor.empty() : tensor<196x196xf16>
  %fill441 = linalg.fill ins(%cst : f16) outs(%init440 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm442 = linalg.matmul ins(%mm433, %w_s2_l9_kt : tensor<196x512xf16>, tensor<512x196xf16>)
                          outs(%fill441 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty443 = tensor.empty() : tensor<196x196xf16>
  %relu444 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm442 : tensor<196x196xf16>)
    outs(%empty443 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init445 = tensor.empty() : tensor<196x512xf16>
  %fill446 = linalg.fill ins(%cst : f16) outs(%init445 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm447 = linalg.matmul ins(%relu444, %mm439 : tensor<196x196xf16>, tensor<196x512xf16>)
                          outs(%fill446 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init448 = tensor.empty() : tensor<196x512xf16>
  %fill449 = linalg.fill ins(%cst : f16) outs(%init448 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm450 = linalg.matmul ins(%mm447, %w_s2_l9_o : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill449 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty451 = tensor.empty() : tensor<196x512xf16>
  %add452 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm450, %add430 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty451 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init453 = tensor.empty() : tensor<196x2048xf16>
  %fill454 = linalg.fill ins(%cst : f16) outs(%init453 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %mm455 = linalg.matmul ins(%add452, %w_s2_l9_ff_up : tensor<196x512xf16>, tensor<512x2048xf16>)
                          outs(%fill454 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %empty456 = tensor.empty() : tensor<196x2048xf16>
  %relu457 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm455 : tensor<196x2048xf16>)
    outs(%empty456 : tensor<196x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x2048xf16>
  %init458 = tensor.empty() : tensor<196x512xf16>
  %fill459 = linalg.fill ins(%cst : f16) outs(%init458 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm460 = linalg.matmul ins(%relu457, %w_s2_l9_ff_dn : tensor<196x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill459 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty461 = tensor.empty() : tensor<196x512xf16>
  %add462 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm460, %add452 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty461 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init463 = tensor.empty() : tensor<196x512xf16>
  %fill464 = linalg.fill ins(%cst : f16) outs(%init463 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm465 = linalg.matmul ins(%add462, %w_s2_l10_q : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill464 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init466 = tensor.empty() : tensor<196x512xf16>
  %fill467 = linalg.fill ins(%cst : f16) outs(%init466 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm468 = linalg.matmul ins(%add462, %w_s2_l10_k : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill467 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init469 = tensor.empty() : tensor<196x512xf16>
  %fill470 = linalg.fill ins(%cst : f16) outs(%init469 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm471 = linalg.matmul ins(%add462, %w_s2_l10_v : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill470 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init472 = tensor.empty() : tensor<196x196xf16>
  %fill473 = linalg.fill ins(%cst : f16) outs(%init472 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm474 = linalg.matmul ins(%mm465, %w_s2_l10_kt : tensor<196x512xf16>, tensor<512x196xf16>)
                          outs(%fill473 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty475 = tensor.empty() : tensor<196x196xf16>
  %relu476 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm474 : tensor<196x196xf16>)
    outs(%empty475 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init477 = tensor.empty() : tensor<196x512xf16>
  %fill478 = linalg.fill ins(%cst : f16) outs(%init477 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm479 = linalg.matmul ins(%relu476, %mm471 : tensor<196x196xf16>, tensor<196x512xf16>)
                          outs(%fill478 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init480 = tensor.empty() : tensor<196x512xf16>
  %fill481 = linalg.fill ins(%cst : f16) outs(%init480 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm482 = linalg.matmul ins(%mm479, %w_s2_l10_o : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill481 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty483 = tensor.empty() : tensor<196x512xf16>
  %add484 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm482, %add462 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty483 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init485 = tensor.empty() : tensor<196x2048xf16>
  %fill486 = linalg.fill ins(%cst : f16) outs(%init485 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %mm487 = linalg.matmul ins(%add484, %w_s2_l10_ff_up : tensor<196x512xf16>, tensor<512x2048xf16>)
                          outs(%fill486 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %empty488 = tensor.empty() : tensor<196x2048xf16>
  %relu489 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm487 : tensor<196x2048xf16>)
    outs(%empty488 : tensor<196x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x2048xf16>
  %init490 = tensor.empty() : tensor<196x512xf16>
  %fill491 = linalg.fill ins(%cst : f16) outs(%init490 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm492 = linalg.matmul ins(%relu489, %w_s2_l10_ff_dn : tensor<196x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill491 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty493 = tensor.empty() : tensor<196x512xf16>
  %add494 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm492, %add484 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty493 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init495 = tensor.empty() : tensor<196x512xf16>
  %fill496 = linalg.fill ins(%cst : f16) outs(%init495 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm497 = linalg.matmul ins(%add494, %w_s2_l11_q : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill496 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init498 = tensor.empty() : tensor<196x512xf16>
  %fill499 = linalg.fill ins(%cst : f16) outs(%init498 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm500 = linalg.matmul ins(%add494, %w_s2_l11_k : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill499 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init501 = tensor.empty() : tensor<196x512xf16>
  %fill502 = linalg.fill ins(%cst : f16) outs(%init501 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm503 = linalg.matmul ins(%add494, %w_s2_l11_v : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill502 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init504 = tensor.empty() : tensor<196x196xf16>
  %fill505 = linalg.fill ins(%cst : f16) outs(%init504 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm506 = linalg.matmul ins(%mm497, %w_s2_l11_kt : tensor<196x512xf16>, tensor<512x196xf16>)
                          outs(%fill505 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty507 = tensor.empty() : tensor<196x196xf16>
  %relu508 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm506 : tensor<196x196xf16>)
    outs(%empty507 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init509 = tensor.empty() : tensor<196x512xf16>
  %fill510 = linalg.fill ins(%cst : f16) outs(%init509 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm511 = linalg.matmul ins(%relu508, %mm503 : tensor<196x196xf16>, tensor<196x512xf16>)
                          outs(%fill510 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init512 = tensor.empty() : tensor<196x512xf16>
  %fill513 = linalg.fill ins(%cst : f16) outs(%init512 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm514 = linalg.matmul ins(%mm511, %w_s2_l11_o : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill513 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty515 = tensor.empty() : tensor<196x512xf16>
  %add516 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm514, %add494 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty515 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init517 = tensor.empty() : tensor<196x2048xf16>
  %fill518 = linalg.fill ins(%cst : f16) outs(%init517 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %mm519 = linalg.matmul ins(%add516, %w_s2_l11_ff_up : tensor<196x512xf16>, tensor<512x2048xf16>)
                          outs(%fill518 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %empty520 = tensor.empty() : tensor<196x2048xf16>
  %relu521 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm519 : tensor<196x2048xf16>)
    outs(%empty520 : tensor<196x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x2048xf16>
  %init522 = tensor.empty() : tensor<196x512xf16>
  %fill523 = linalg.fill ins(%cst : f16) outs(%init522 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm524 = linalg.matmul ins(%relu521, %w_s2_l11_ff_dn : tensor<196x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill523 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty525 = tensor.empty() : tensor<196x512xf16>
  %add526 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm524, %add516 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty525 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init527 = tensor.empty() : tensor<196x512xf16>
  %fill528 = linalg.fill ins(%cst : f16) outs(%init527 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm529 = linalg.matmul ins(%add526, %w_s2_l12_q : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill528 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init530 = tensor.empty() : tensor<196x512xf16>
  %fill531 = linalg.fill ins(%cst : f16) outs(%init530 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm532 = linalg.matmul ins(%add526, %w_s2_l12_k : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill531 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init533 = tensor.empty() : tensor<196x512xf16>
  %fill534 = linalg.fill ins(%cst : f16) outs(%init533 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm535 = linalg.matmul ins(%add526, %w_s2_l12_v : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill534 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init536 = tensor.empty() : tensor<196x196xf16>
  %fill537 = linalg.fill ins(%cst : f16) outs(%init536 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm538 = linalg.matmul ins(%mm529, %w_s2_l12_kt : tensor<196x512xf16>, tensor<512x196xf16>)
                          outs(%fill537 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty539 = tensor.empty() : tensor<196x196xf16>
  %relu540 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm538 : tensor<196x196xf16>)
    outs(%empty539 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init541 = tensor.empty() : tensor<196x512xf16>
  %fill542 = linalg.fill ins(%cst : f16) outs(%init541 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm543 = linalg.matmul ins(%relu540, %mm535 : tensor<196x196xf16>, tensor<196x512xf16>)
                          outs(%fill542 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init544 = tensor.empty() : tensor<196x512xf16>
  %fill545 = linalg.fill ins(%cst : f16) outs(%init544 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm546 = linalg.matmul ins(%mm543, %w_s2_l12_o : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill545 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty547 = tensor.empty() : tensor<196x512xf16>
  %add548 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm546, %add526 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty547 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init549 = tensor.empty() : tensor<196x2048xf16>
  %fill550 = linalg.fill ins(%cst : f16) outs(%init549 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %mm551 = linalg.matmul ins(%add548, %w_s2_l12_ff_up : tensor<196x512xf16>, tensor<512x2048xf16>)
                          outs(%fill550 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %empty552 = tensor.empty() : tensor<196x2048xf16>
  %relu553 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm551 : tensor<196x2048xf16>)
    outs(%empty552 : tensor<196x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x2048xf16>
  %init554 = tensor.empty() : tensor<196x512xf16>
  %fill555 = linalg.fill ins(%cst : f16) outs(%init554 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm556 = linalg.matmul ins(%relu553, %w_s2_l12_ff_dn : tensor<196x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill555 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty557 = tensor.empty() : tensor<196x512xf16>
  %add558 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm556, %add548 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty557 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init559 = tensor.empty() : tensor<196x512xf16>
  %fill560 = linalg.fill ins(%cst : f16) outs(%init559 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm561 = linalg.matmul ins(%add558, %w_s2_l13_q : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill560 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init562 = tensor.empty() : tensor<196x512xf16>
  %fill563 = linalg.fill ins(%cst : f16) outs(%init562 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm564 = linalg.matmul ins(%add558, %w_s2_l13_k : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill563 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init565 = tensor.empty() : tensor<196x512xf16>
  %fill566 = linalg.fill ins(%cst : f16) outs(%init565 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm567 = linalg.matmul ins(%add558, %w_s2_l13_v : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill566 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init568 = tensor.empty() : tensor<196x196xf16>
  %fill569 = linalg.fill ins(%cst : f16) outs(%init568 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm570 = linalg.matmul ins(%mm561, %w_s2_l13_kt : tensor<196x512xf16>, tensor<512x196xf16>)
                          outs(%fill569 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty571 = tensor.empty() : tensor<196x196xf16>
  %relu572 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm570 : tensor<196x196xf16>)
    outs(%empty571 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init573 = tensor.empty() : tensor<196x512xf16>
  %fill574 = linalg.fill ins(%cst : f16) outs(%init573 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm575 = linalg.matmul ins(%relu572, %mm567 : tensor<196x196xf16>, tensor<196x512xf16>)
                          outs(%fill574 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init576 = tensor.empty() : tensor<196x512xf16>
  %fill577 = linalg.fill ins(%cst : f16) outs(%init576 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm578 = linalg.matmul ins(%mm575, %w_s2_l13_o : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill577 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty579 = tensor.empty() : tensor<196x512xf16>
  %add580 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm578, %add558 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty579 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init581 = tensor.empty() : tensor<196x2048xf16>
  %fill582 = linalg.fill ins(%cst : f16) outs(%init581 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %mm583 = linalg.matmul ins(%add580, %w_s2_l13_ff_up : tensor<196x512xf16>, tensor<512x2048xf16>)
                          outs(%fill582 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %empty584 = tensor.empty() : tensor<196x2048xf16>
  %relu585 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm583 : tensor<196x2048xf16>)
    outs(%empty584 : tensor<196x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x2048xf16>
  %init586 = tensor.empty() : tensor<196x512xf16>
  %fill587 = linalg.fill ins(%cst : f16) outs(%init586 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm588 = linalg.matmul ins(%relu585, %w_s2_l13_ff_dn : tensor<196x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill587 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty589 = tensor.empty() : tensor<196x512xf16>
  %add590 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm588, %add580 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty589 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init591 = tensor.empty() : tensor<196x512xf16>
  %fill592 = linalg.fill ins(%cst : f16) outs(%init591 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm593 = linalg.matmul ins(%add590, %w_s2_l14_q : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill592 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init594 = tensor.empty() : tensor<196x512xf16>
  %fill595 = linalg.fill ins(%cst : f16) outs(%init594 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm596 = linalg.matmul ins(%add590, %w_s2_l14_k : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill595 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init597 = tensor.empty() : tensor<196x512xf16>
  %fill598 = linalg.fill ins(%cst : f16) outs(%init597 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm599 = linalg.matmul ins(%add590, %w_s2_l14_v : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill598 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init600 = tensor.empty() : tensor<196x196xf16>
  %fill601 = linalg.fill ins(%cst : f16) outs(%init600 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm602 = linalg.matmul ins(%mm593, %w_s2_l14_kt : tensor<196x512xf16>, tensor<512x196xf16>)
                          outs(%fill601 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty603 = tensor.empty() : tensor<196x196xf16>
  %relu604 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm602 : tensor<196x196xf16>)
    outs(%empty603 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init605 = tensor.empty() : tensor<196x512xf16>
  %fill606 = linalg.fill ins(%cst : f16) outs(%init605 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm607 = linalg.matmul ins(%relu604, %mm599 : tensor<196x196xf16>, tensor<196x512xf16>)
                          outs(%fill606 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init608 = tensor.empty() : tensor<196x512xf16>
  %fill609 = linalg.fill ins(%cst : f16) outs(%init608 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm610 = linalg.matmul ins(%mm607, %w_s2_l14_o : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill609 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty611 = tensor.empty() : tensor<196x512xf16>
  %add612 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm610, %add590 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty611 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init613 = tensor.empty() : tensor<196x2048xf16>
  %fill614 = linalg.fill ins(%cst : f16) outs(%init613 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %mm615 = linalg.matmul ins(%add612, %w_s2_l14_ff_up : tensor<196x512xf16>, tensor<512x2048xf16>)
                          outs(%fill614 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %empty616 = tensor.empty() : tensor<196x2048xf16>
  %relu617 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm615 : tensor<196x2048xf16>)
    outs(%empty616 : tensor<196x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x2048xf16>
  %init618 = tensor.empty() : tensor<196x512xf16>
  %fill619 = linalg.fill ins(%cst : f16) outs(%init618 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm620 = linalg.matmul ins(%relu617, %w_s2_l14_ff_dn : tensor<196x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill619 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty621 = tensor.empty() : tensor<196x512xf16>
  %add622 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm620, %add612 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty621 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init623 = tensor.empty() : tensor<196x512xf16>
  %fill624 = linalg.fill ins(%cst : f16) outs(%init623 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm625 = linalg.matmul ins(%add622, %w_s2_l15_q : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill624 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init626 = tensor.empty() : tensor<196x512xf16>
  %fill627 = linalg.fill ins(%cst : f16) outs(%init626 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm628 = linalg.matmul ins(%add622, %w_s2_l15_k : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill627 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init629 = tensor.empty() : tensor<196x512xf16>
  %fill630 = linalg.fill ins(%cst : f16) outs(%init629 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm631 = linalg.matmul ins(%add622, %w_s2_l15_v : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill630 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init632 = tensor.empty() : tensor<196x196xf16>
  %fill633 = linalg.fill ins(%cst : f16) outs(%init632 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm634 = linalg.matmul ins(%mm625, %w_s2_l15_kt : tensor<196x512xf16>, tensor<512x196xf16>)
                          outs(%fill633 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty635 = tensor.empty() : tensor<196x196xf16>
  %relu636 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm634 : tensor<196x196xf16>)
    outs(%empty635 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init637 = tensor.empty() : tensor<196x512xf16>
  %fill638 = linalg.fill ins(%cst : f16) outs(%init637 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm639 = linalg.matmul ins(%relu636, %mm631 : tensor<196x196xf16>, tensor<196x512xf16>)
                          outs(%fill638 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init640 = tensor.empty() : tensor<196x512xf16>
  %fill641 = linalg.fill ins(%cst : f16) outs(%init640 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm642 = linalg.matmul ins(%mm639, %w_s2_l15_o : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill641 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty643 = tensor.empty() : tensor<196x512xf16>
  %add644 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm642, %add622 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty643 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init645 = tensor.empty() : tensor<196x2048xf16>
  %fill646 = linalg.fill ins(%cst : f16) outs(%init645 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %mm647 = linalg.matmul ins(%add644, %w_s2_l15_ff_up : tensor<196x512xf16>, tensor<512x2048xf16>)
                          outs(%fill646 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %empty648 = tensor.empty() : tensor<196x2048xf16>
  %relu649 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm647 : tensor<196x2048xf16>)
    outs(%empty648 : tensor<196x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x2048xf16>
  %init650 = tensor.empty() : tensor<196x512xf16>
  %fill651 = linalg.fill ins(%cst : f16) outs(%init650 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm652 = linalg.matmul ins(%relu649, %w_s2_l15_ff_dn : tensor<196x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill651 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty653 = tensor.empty() : tensor<196x512xf16>
  %add654 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm652, %add644 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty653 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init655 = tensor.empty() : tensor<196x512xf16>
  %fill656 = linalg.fill ins(%cst : f16) outs(%init655 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm657 = linalg.matmul ins(%add654, %w_s2_l16_q : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill656 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init658 = tensor.empty() : tensor<196x512xf16>
  %fill659 = linalg.fill ins(%cst : f16) outs(%init658 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm660 = linalg.matmul ins(%add654, %w_s2_l16_k : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill659 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init661 = tensor.empty() : tensor<196x512xf16>
  %fill662 = linalg.fill ins(%cst : f16) outs(%init661 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm663 = linalg.matmul ins(%add654, %w_s2_l16_v : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill662 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init664 = tensor.empty() : tensor<196x196xf16>
  %fill665 = linalg.fill ins(%cst : f16) outs(%init664 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm666 = linalg.matmul ins(%mm657, %w_s2_l16_kt : tensor<196x512xf16>, tensor<512x196xf16>)
                          outs(%fill665 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty667 = tensor.empty() : tensor<196x196xf16>
  %relu668 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm666 : tensor<196x196xf16>)
    outs(%empty667 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init669 = tensor.empty() : tensor<196x512xf16>
  %fill670 = linalg.fill ins(%cst : f16) outs(%init669 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm671 = linalg.matmul ins(%relu668, %mm663 : tensor<196x196xf16>, tensor<196x512xf16>)
                          outs(%fill670 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init672 = tensor.empty() : tensor<196x512xf16>
  %fill673 = linalg.fill ins(%cst : f16) outs(%init672 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm674 = linalg.matmul ins(%mm671, %w_s2_l16_o : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill673 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty675 = tensor.empty() : tensor<196x512xf16>
  %add676 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm674, %add654 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty675 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init677 = tensor.empty() : tensor<196x2048xf16>
  %fill678 = linalg.fill ins(%cst : f16) outs(%init677 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %mm679 = linalg.matmul ins(%add676, %w_s2_l16_ff_up : tensor<196x512xf16>, tensor<512x2048xf16>)
                          outs(%fill678 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %empty680 = tensor.empty() : tensor<196x2048xf16>
  %relu681 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm679 : tensor<196x2048xf16>)
    outs(%empty680 : tensor<196x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x2048xf16>
  %init682 = tensor.empty() : tensor<196x512xf16>
  %fill683 = linalg.fill ins(%cst : f16) outs(%init682 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm684 = linalg.matmul ins(%relu681, %w_s2_l16_ff_dn : tensor<196x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill683 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty685 = tensor.empty() : tensor<196x512xf16>
  %add686 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm684, %add676 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty685 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init687 = tensor.empty() : tensor<196x512xf16>
  %fill688 = linalg.fill ins(%cst : f16) outs(%init687 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm689 = linalg.matmul ins(%add686, %w_s2_l17_q : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill688 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init690 = tensor.empty() : tensor<196x512xf16>
  %fill691 = linalg.fill ins(%cst : f16) outs(%init690 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm692 = linalg.matmul ins(%add686, %w_s2_l17_k : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill691 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init693 = tensor.empty() : tensor<196x512xf16>
  %fill694 = linalg.fill ins(%cst : f16) outs(%init693 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm695 = linalg.matmul ins(%add686, %w_s2_l17_v : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill694 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init696 = tensor.empty() : tensor<196x196xf16>
  %fill697 = linalg.fill ins(%cst : f16) outs(%init696 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm698 = linalg.matmul ins(%mm689, %w_s2_l17_kt : tensor<196x512xf16>, tensor<512x196xf16>)
                          outs(%fill697 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty699 = tensor.empty() : tensor<196x196xf16>
  %relu700 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm698 : tensor<196x196xf16>)
    outs(%empty699 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init701 = tensor.empty() : tensor<196x512xf16>
  %fill702 = linalg.fill ins(%cst : f16) outs(%init701 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm703 = linalg.matmul ins(%relu700, %mm695 : tensor<196x196xf16>, tensor<196x512xf16>)
                          outs(%fill702 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %init704 = tensor.empty() : tensor<196x512xf16>
  %fill705 = linalg.fill ins(%cst : f16) outs(%init704 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm706 = linalg.matmul ins(%mm703, %w_s2_l17_o : tensor<196x512xf16>, tensor<512x512xf16>)
                          outs(%fill705 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty707 = tensor.empty() : tensor<196x512xf16>
  %add708 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm706, %add686 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty707 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>
  %init709 = tensor.empty() : tensor<196x2048xf16>
  %fill710 = linalg.fill ins(%cst : f16) outs(%init709 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %mm711 = linalg.matmul ins(%add708, %w_s2_l17_ff_up : tensor<196x512xf16>, tensor<512x2048xf16>)
                          outs(%fill710 : tensor<196x2048xf16>) -> tensor<196x2048xf16>
  %empty712 = tensor.empty() : tensor<196x2048xf16>
  %relu713 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm711 : tensor<196x2048xf16>)
    outs(%empty712 : tensor<196x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x2048xf16>
  %init714 = tensor.empty() : tensor<196x512xf16>
  %fill715 = linalg.fill ins(%cst : f16) outs(%init714 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %mm716 = linalg.matmul ins(%relu713, %w_s2_l17_ff_dn : tensor<196x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill715 : tensor<196x512xf16>) -> tensor<196x512xf16>
  %empty717 = tensor.empty() : tensor<196x512xf16>
  %add718 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm716, %add708 : tensor<196x512xf16>, tensor<196x512xf16>)
    outs(%empty717 : tensor<196x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x512xf16>

  // Patch merging: [196,512] -> [49,1024]
  %init719 = tensor.empty() : tensor<196x1024xf16>
  %fill720 = linalg.fill ins(%cst : f16) outs(%init719 : tensor<196x1024xf16>) -> tensor<196x1024xf16>
  %mm721 = linalg.matmul ins(%add718, %w_merge2 : tensor<196x512xf16>, tensor<512x1024xf16>)
                          outs(%fill720 : tensor<196x1024xf16>) -> tensor<196x1024xf16>
  %merge_reshape722 = tensor.empty() : tensor<49x1024xf16>
  %fill723 = linalg.fill ins(%cst : f16) outs(%merge_reshape722 : tensor<49x1024xf16>) -> tensor<49x1024xf16>

  // === Swin Stage 3: seq=49, dim=1024 ===
  %init724 = tensor.empty() : tensor<49x1024xf16>
  %fill725 = linalg.fill ins(%cst : f16) outs(%init724 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %mm726 = linalg.matmul ins(%fill723, %w_s3_l0_q : tensor<49x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill725 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %init727 = tensor.empty() : tensor<49x1024xf16>
  %fill728 = linalg.fill ins(%cst : f16) outs(%init727 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %mm729 = linalg.matmul ins(%fill723, %w_s3_l0_k : tensor<49x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill728 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %init730 = tensor.empty() : tensor<49x1024xf16>
  %fill731 = linalg.fill ins(%cst : f16) outs(%init730 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %mm732 = linalg.matmul ins(%fill723, %w_s3_l0_v : tensor<49x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill731 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %init733 = tensor.empty() : tensor<49x49xf16>
  %fill734 = linalg.fill ins(%cst : f16) outs(%init733 : tensor<49x49xf16>) -> tensor<49x49xf16>
  %mm735 = linalg.matmul ins(%mm726, %w_s3_l0_kt : tensor<49x1024xf16>, tensor<1024x49xf16>)
                          outs(%fill734 : tensor<49x49xf16>) -> tensor<49x49xf16>
  %empty736 = tensor.empty() : tensor<49x49xf16>
  %relu737 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm735 : tensor<49x49xf16>)
    outs(%empty736 : tensor<49x49xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<49x49xf16>
  %init738 = tensor.empty() : tensor<49x1024xf16>
  %fill739 = linalg.fill ins(%cst : f16) outs(%init738 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %mm740 = linalg.matmul ins(%relu737, %mm732 : tensor<49x49xf16>, tensor<49x1024xf16>)
                          outs(%fill739 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %init741 = tensor.empty() : tensor<49x1024xf16>
  %fill742 = linalg.fill ins(%cst : f16) outs(%init741 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %mm743 = linalg.matmul ins(%mm740, %w_s3_l0_o : tensor<49x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill742 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %empty744 = tensor.empty() : tensor<49x1024xf16>
  %add745 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm743, %fill723 : tensor<49x1024xf16>, tensor<49x1024xf16>)
    outs(%empty744 : tensor<49x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<49x1024xf16>
  %init746 = tensor.empty() : tensor<49x4096xf16>
  %fill747 = linalg.fill ins(%cst : f16) outs(%init746 : tensor<49x4096xf16>) -> tensor<49x4096xf16>
  %mm748 = linalg.matmul ins(%add745, %w_s3_l0_ff_up : tensor<49x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill747 : tensor<49x4096xf16>) -> tensor<49x4096xf16>
  %empty749 = tensor.empty() : tensor<49x4096xf16>
  %relu750 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm748 : tensor<49x4096xf16>)
    outs(%empty749 : tensor<49x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<49x4096xf16>
  %init751 = tensor.empty() : tensor<49x1024xf16>
  %fill752 = linalg.fill ins(%cst : f16) outs(%init751 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %mm753 = linalg.matmul ins(%relu750, %w_s3_l0_ff_dn : tensor<49x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill752 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %empty754 = tensor.empty() : tensor<49x1024xf16>
  %add755 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm753, %add745 : tensor<49x1024xf16>, tensor<49x1024xf16>)
    outs(%empty754 : tensor<49x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<49x1024xf16>
  %init756 = tensor.empty() : tensor<49x1024xf16>
  %fill757 = linalg.fill ins(%cst : f16) outs(%init756 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %mm758 = linalg.matmul ins(%add755, %w_s3_l1_q : tensor<49x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill757 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %init759 = tensor.empty() : tensor<49x1024xf16>
  %fill760 = linalg.fill ins(%cst : f16) outs(%init759 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %mm761 = linalg.matmul ins(%add755, %w_s3_l1_k : tensor<49x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill760 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %init762 = tensor.empty() : tensor<49x1024xf16>
  %fill763 = linalg.fill ins(%cst : f16) outs(%init762 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %mm764 = linalg.matmul ins(%add755, %w_s3_l1_v : tensor<49x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill763 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %init765 = tensor.empty() : tensor<49x49xf16>
  %fill766 = linalg.fill ins(%cst : f16) outs(%init765 : tensor<49x49xf16>) -> tensor<49x49xf16>
  %mm767 = linalg.matmul ins(%mm758, %w_s3_l1_kt : tensor<49x1024xf16>, tensor<1024x49xf16>)
                          outs(%fill766 : tensor<49x49xf16>) -> tensor<49x49xf16>
  %empty768 = tensor.empty() : tensor<49x49xf16>
  %relu769 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm767 : tensor<49x49xf16>)
    outs(%empty768 : tensor<49x49xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<49x49xf16>
  %init770 = tensor.empty() : tensor<49x1024xf16>
  %fill771 = linalg.fill ins(%cst : f16) outs(%init770 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %mm772 = linalg.matmul ins(%relu769, %mm764 : tensor<49x49xf16>, tensor<49x1024xf16>)
                          outs(%fill771 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %init773 = tensor.empty() : tensor<49x1024xf16>
  %fill774 = linalg.fill ins(%cst : f16) outs(%init773 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %mm775 = linalg.matmul ins(%mm772, %w_s3_l1_o : tensor<49x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill774 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %empty776 = tensor.empty() : tensor<49x1024xf16>
  %add777 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm775, %add755 : tensor<49x1024xf16>, tensor<49x1024xf16>)
    outs(%empty776 : tensor<49x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<49x1024xf16>
  %init778 = tensor.empty() : tensor<49x4096xf16>
  %fill779 = linalg.fill ins(%cst : f16) outs(%init778 : tensor<49x4096xf16>) -> tensor<49x4096xf16>
  %mm780 = linalg.matmul ins(%add777, %w_s3_l1_ff_up : tensor<49x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill779 : tensor<49x4096xf16>) -> tensor<49x4096xf16>
  %empty781 = tensor.empty() : tensor<49x4096xf16>
  %relu782 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm780 : tensor<49x4096xf16>)
    outs(%empty781 : tensor<49x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<49x4096xf16>
  %init783 = tensor.empty() : tensor<49x1024xf16>
  %fill784 = linalg.fill ins(%cst : f16) outs(%init783 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %mm785 = linalg.matmul ins(%relu782, %w_s3_l1_ff_dn : tensor<49x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill784 : tensor<49x1024xf16>) -> tensor<49x1024xf16>
  %empty786 = tensor.empty() : tensor<49x1024xf16>
  %add787 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm785, %add777 : tensor<49x1024xf16>, tensor<49x1024xf16>)
    outs(%empty786 : tensor<49x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<49x1024xf16>

  // Head: [49,1024] x [1024,1000]
  %init788 = tensor.empty() : tensor<49x1000xf16>
  %fill789 = linalg.fill ins(%cst : f16) outs(%init788 : tensor<49x1000xf16>) -> tensor<49x1000xf16>
  %mm790 = linalg.matmul ins(%add787, %w_head : tensor<49x1024xf16>, tensor<1024x1000xf16>)
                          outs(%fill789 : tensor<49x1000xf16>) -> tensor<49x1000xf16>
  return %mm790 : tensor<49x1000xf16>
}
