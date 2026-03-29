func.func @vit_h_14(
    %input: tensor<1x3x224x224xf16>,
    %w_patch: tensor<1280x3x14x14xf16>,
    %w_q0: tensor<1280x1280xf16>,
    %w_k0: tensor<1280x1280xf16>,
    %w_v0: tensor<1280x1280xf16>,
    %w_kt0: tensor<1280x256xf16>,
    %w_o0: tensor<1280x1280xf16>,
    %w_ff0_up: tensor<1280x5120xf16>,
    %w_ff0_down: tensor<5120x1280xf16>,
    %w_q1: tensor<1280x1280xf16>,
    %w_k1: tensor<1280x1280xf16>,
    %w_v1: tensor<1280x1280xf16>,
    %w_kt1: tensor<1280x256xf16>,
    %w_o1: tensor<1280x1280xf16>,
    %w_ff1_up: tensor<1280x5120xf16>,
    %w_ff1_down: tensor<5120x1280xf16>,
    %w_q2: tensor<1280x1280xf16>,
    %w_k2: tensor<1280x1280xf16>,
    %w_v2: tensor<1280x1280xf16>,
    %w_kt2: tensor<1280x256xf16>,
    %w_o2: tensor<1280x1280xf16>,
    %w_ff2_up: tensor<1280x5120xf16>,
    %w_ff2_down: tensor<5120x1280xf16>,
    %w_q3: tensor<1280x1280xf16>,
    %w_k3: tensor<1280x1280xf16>,
    %w_v3: tensor<1280x1280xf16>,
    %w_kt3: tensor<1280x256xf16>,
    %w_o3: tensor<1280x1280xf16>,
    %w_ff3_up: tensor<1280x5120xf16>,
    %w_ff3_down: tensor<5120x1280xf16>,
    %w_q4: tensor<1280x1280xf16>,
    %w_k4: tensor<1280x1280xf16>,
    %w_v4: tensor<1280x1280xf16>,
    %w_kt4: tensor<1280x256xf16>,
    %w_o4: tensor<1280x1280xf16>,
    %w_ff4_up: tensor<1280x5120xf16>,
    %w_ff4_down: tensor<5120x1280xf16>,
    %w_q5: tensor<1280x1280xf16>,
    %w_k5: tensor<1280x1280xf16>,
    %w_v5: tensor<1280x1280xf16>,
    %w_kt5: tensor<1280x256xf16>,
    %w_o5: tensor<1280x1280xf16>,
    %w_ff5_up: tensor<1280x5120xf16>,
    %w_ff5_down: tensor<5120x1280xf16>,
    %w_q6: tensor<1280x1280xf16>,
    %w_k6: tensor<1280x1280xf16>,
    %w_v6: tensor<1280x1280xf16>,
    %w_kt6: tensor<1280x256xf16>,
    %w_o6: tensor<1280x1280xf16>,
    %w_ff6_up: tensor<1280x5120xf16>,
    %w_ff6_down: tensor<5120x1280xf16>,
    %w_q7: tensor<1280x1280xf16>,
    %w_k7: tensor<1280x1280xf16>,
    %w_v7: tensor<1280x1280xf16>,
    %w_kt7: tensor<1280x256xf16>,
    %w_o7: tensor<1280x1280xf16>,
    %w_ff7_up: tensor<1280x5120xf16>,
    %w_ff7_down: tensor<5120x1280xf16>,
    %w_q8: tensor<1280x1280xf16>,
    %w_k8: tensor<1280x1280xf16>,
    %w_v8: tensor<1280x1280xf16>,
    %w_kt8: tensor<1280x256xf16>,
    %w_o8: tensor<1280x1280xf16>,
    %w_ff8_up: tensor<1280x5120xf16>,
    %w_ff8_down: tensor<5120x1280xf16>,
    %w_q9: tensor<1280x1280xf16>,
    %w_k9: tensor<1280x1280xf16>,
    %w_v9: tensor<1280x1280xf16>,
    %w_kt9: tensor<1280x256xf16>,
    %w_o9: tensor<1280x1280xf16>,
    %w_ff9_up: tensor<1280x5120xf16>,
    %w_ff9_down: tensor<5120x1280xf16>,
    %w_q10: tensor<1280x1280xf16>,
    %w_k10: tensor<1280x1280xf16>,
    %w_v10: tensor<1280x1280xf16>,
    %w_kt10: tensor<1280x256xf16>,
    %w_o10: tensor<1280x1280xf16>,
    %w_ff10_up: tensor<1280x5120xf16>,
    %w_ff10_down: tensor<5120x1280xf16>,
    %w_q11: tensor<1280x1280xf16>,
    %w_k11: tensor<1280x1280xf16>,
    %w_v11: tensor<1280x1280xf16>,
    %w_kt11: tensor<1280x256xf16>,
    %w_o11: tensor<1280x1280xf16>,
    %w_ff11_up: tensor<1280x5120xf16>,
    %w_ff11_down: tensor<5120x1280xf16>,
    %w_q12: tensor<1280x1280xf16>,
    %w_k12: tensor<1280x1280xf16>,
    %w_v12: tensor<1280x1280xf16>,
    %w_kt12: tensor<1280x256xf16>,
    %w_o12: tensor<1280x1280xf16>,
    %w_ff12_up: tensor<1280x5120xf16>,
    %w_ff12_down: tensor<5120x1280xf16>,
    %w_q13: tensor<1280x1280xf16>,
    %w_k13: tensor<1280x1280xf16>,
    %w_v13: tensor<1280x1280xf16>,
    %w_kt13: tensor<1280x256xf16>,
    %w_o13: tensor<1280x1280xf16>,
    %w_ff13_up: tensor<1280x5120xf16>,
    %w_ff13_down: tensor<5120x1280xf16>,
    %w_q14: tensor<1280x1280xf16>,
    %w_k14: tensor<1280x1280xf16>,
    %w_v14: tensor<1280x1280xf16>,
    %w_kt14: tensor<1280x256xf16>,
    %w_o14: tensor<1280x1280xf16>,
    %w_ff14_up: tensor<1280x5120xf16>,
    %w_ff14_down: tensor<5120x1280xf16>,
    %w_q15: tensor<1280x1280xf16>,
    %w_k15: tensor<1280x1280xf16>,
    %w_v15: tensor<1280x1280xf16>,
    %w_kt15: tensor<1280x256xf16>,
    %w_o15: tensor<1280x1280xf16>,
    %w_ff15_up: tensor<1280x5120xf16>,
    %w_ff15_down: tensor<5120x1280xf16>,
    %w_q16: tensor<1280x1280xf16>,
    %w_k16: tensor<1280x1280xf16>,
    %w_v16: tensor<1280x1280xf16>,
    %w_kt16: tensor<1280x256xf16>,
    %w_o16: tensor<1280x1280xf16>,
    %w_ff16_up: tensor<1280x5120xf16>,
    %w_ff16_down: tensor<5120x1280xf16>,
    %w_q17: tensor<1280x1280xf16>,
    %w_k17: tensor<1280x1280xf16>,
    %w_v17: tensor<1280x1280xf16>,
    %w_kt17: tensor<1280x256xf16>,
    %w_o17: tensor<1280x1280xf16>,
    %w_ff17_up: tensor<1280x5120xf16>,
    %w_ff17_down: tensor<5120x1280xf16>,
    %w_q18: tensor<1280x1280xf16>,
    %w_k18: tensor<1280x1280xf16>,
    %w_v18: tensor<1280x1280xf16>,
    %w_kt18: tensor<1280x256xf16>,
    %w_o18: tensor<1280x1280xf16>,
    %w_ff18_up: tensor<1280x5120xf16>,
    %w_ff18_down: tensor<5120x1280xf16>,
    %w_q19: tensor<1280x1280xf16>,
    %w_k19: tensor<1280x1280xf16>,
    %w_v19: tensor<1280x1280xf16>,
    %w_kt19: tensor<1280x256xf16>,
    %w_o19: tensor<1280x1280xf16>,
    %w_ff19_up: tensor<1280x5120xf16>,
    %w_ff19_down: tensor<5120x1280xf16>,
    %w_q20: tensor<1280x1280xf16>,
    %w_k20: tensor<1280x1280xf16>,
    %w_v20: tensor<1280x1280xf16>,
    %w_kt20: tensor<1280x256xf16>,
    %w_o20: tensor<1280x1280xf16>,
    %w_ff20_up: tensor<1280x5120xf16>,
    %w_ff20_down: tensor<5120x1280xf16>,
    %w_q21: tensor<1280x1280xf16>,
    %w_k21: tensor<1280x1280xf16>,
    %w_v21: tensor<1280x1280xf16>,
    %w_kt21: tensor<1280x256xf16>,
    %w_o21: tensor<1280x1280xf16>,
    %w_ff21_up: tensor<1280x5120xf16>,
    %w_ff21_down: tensor<5120x1280xf16>,
    %w_q22: tensor<1280x1280xf16>,
    %w_k22: tensor<1280x1280xf16>,
    %w_v22: tensor<1280x1280xf16>,
    %w_kt22: tensor<1280x256xf16>,
    %w_o22: tensor<1280x1280xf16>,
    %w_ff22_up: tensor<1280x5120xf16>,
    %w_ff22_down: tensor<5120x1280xf16>,
    %w_q23: tensor<1280x1280xf16>,
    %w_k23: tensor<1280x1280xf16>,
    %w_v23: tensor<1280x1280xf16>,
    %w_kt23: tensor<1280x256xf16>,
    %w_o23: tensor<1280x1280xf16>,
    %w_ff23_up: tensor<1280x5120xf16>,
    %w_ff23_down: tensor<5120x1280xf16>,
    %w_q24: tensor<1280x1280xf16>,
    %w_k24: tensor<1280x1280xf16>,
    %w_v24: tensor<1280x1280xf16>,
    %w_kt24: tensor<1280x256xf16>,
    %w_o24: tensor<1280x1280xf16>,
    %w_ff24_up: tensor<1280x5120xf16>,
    %w_ff24_down: tensor<5120x1280xf16>,
    %w_q25: tensor<1280x1280xf16>,
    %w_k25: tensor<1280x1280xf16>,
    %w_v25: tensor<1280x1280xf16>,
    %w_kt25: tensor<1280x256xf16>,
    %w_o25: tensor<1280x1280xf16>,
    %w_ff25_up: tensor<1280x5120xf16>,
    %w_ff25_down: tensor<5120x1280xf16>,
    %w_q26: tensor<1280x1280xf16>,
    %w_k26: tensor<1280x1280xf16>,
    %w_v26: tensor<1280x1280xf16>,
    %w_kt26: tensor<1280x256xf16>,
    %w_o26: tensor<1280x1280xf16>,
    %w_ff26_up: tensor<1280x5120xf16>,
    %w_ff26_down: tensor<5120x1280xf16>,
    %w_q27: tensor<1280x1280xf16>,
    %w_k27: tensor<1280x1280xf16>,
    %w_v27: tensor<1280x1280xf16>,
    %w_kt27: tensor<1280x256xf16>,
    %w_o27: tensor<1280x1280xf16>,
    %w_ff27_up: tensor<1280x5120xf16>,
    %w_ff27_down: tensor<5120x1280xf16>,
    %w_q28: tensor<1280x1280xf16>,
    %w_k28: tensor<1280x1280xf16>,
    %w_v28: tensor<1280x1280xf16>,
    %w_kt28: tensor<1280x256xf16>,
    %w_o28: tensor<1280x1280xf16>,
    %w_ff28_up: tensor<1280x5120xf16>,
    %w_ff28_down: tensor<5120x1280xf16>,
    %w_q29: tensor<1280x1280xf16>,
    %w_k29: tensor<1280x1280xf16>,
    %w_v29: tensor<1280x1280xf16>,
    %w_kt29: tensor<1280x256xf16>,
    %w_o29: tensor<1280x1280xf16>,
    %w_ff29_up: tensor<1280x5120xf16>,
    %w_ff29_down: tensor<5120x1280xf16>,
    %w_q30: tensor<1280x1280xf16>,
    %w_k30: tensor<1280x1280xf16>,
    %w_v30: tensor<1280x1280xf16>,
    %w_kt30: tensor<1280x256xf16>,
    %w_o30: tensor<1280x1280xf16>,
    %w_ff30_up: tensor<1280x5120xf16>,
    %w_ff30_down: tensor<5120x1280xf16>,
    %w_q31: tensor<1280x1280xf16>,
    %w_k31: tensor<1280x1280xf16>,
    %w_v31: tensor<1280x1280xf16>,
    %w_kt31: tensor<1280x256xf16>,
    %w_o31: tensor<1280x1280xf16>,
    %w_ff31_up: tensor<1280x5120xf16>,
    %w_ff31_down: tensor<5120x1280xf16>,
    %w_head: tensor<1280x1000xf16>) -> tensor<256x1000xf16> {
  %cst = arith.constant 0.0 : f16

  // Patch embedding: 14x14 conv, 3->1280
  %init0 = tensor.empty() : tensor<1x1280x16x16xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init0 : tensor<1x1280x16x16xf16>) -> tensor<1x1280x16x16xf16>
  %conv2 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<14> : tensor<2xi64>
  } ins(%input, %w_patch : tensor<1x3x224x224xf16>, tensor<1280x3x14x14xf16>)
    outs(%fill1 : tensor<1x1280x16x16xf16>) -> tensor<1x1280x16x16xf16>

  // Conceptual reshape to [256, 1280]
  %seq_empty3 = tensor.empty() : tensor<256x1280xf16>
  %seq_fill4 = linalg.fill ins(%cst : f16) outs(%seq_empty3 : tensor<256x1280xf16>) -> tensor<256x1280xf16>

  // === Transformer Layer 0 ===
  %init5 = tensor.empty() : tensor<256x1280xf16>
  %fill6 = linalg.fill ins(%cst : f16) outs(%init5 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm7 = linalg.matmul ins(%seq_fill4, %w_q0 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill6 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init8 = tensor.empty() : tensor<256x1280xf16>
  %fill9 = linalg.fill ins(%cst : f16) outs(%init8 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm10 = linalg.matmul ins(%seq_fill4, %w_k0 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill9 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init11 = tensor.empty() : tensor<256x1280xf16>
  %fill12 = linalg.fill ins(%cst : f16) outs(%init11 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm13 = linalg.matmul ins(%seq_fill4, %w_v0 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill12 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init14 = tensor.empty() : tensor<256x256xf16>
  %fill15 = linalg.fill ins(%cst : f16) outs(%init14 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm16 = linalg.matmul ins(%mm7, %w_kt0 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill15 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty17 = tensor.empty() : tensor<256x256xf16>
  %relu18 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm16 : tensor<256x256xf16>)
    outs(%empty17 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init19 = tensor.empty() : tensor<256x1280xf16>
  %fill20 = linalg.fill ins(%cst : f16) outs(%init19 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm21 = linalg.matmul ins(%relu18, %mm13 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill20 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init22 = tensor.empty() : tensor<256x1280xf16>
  %fill23 = linalg.fill ins(%cst : f16) outs(%init22 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm24 = linalg.matmul ins(%mm21, %w_o0 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill23 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty25 = tensor.empty() : tensor<256x1280xf16>
  %add26 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm24, %seq_fill4 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty25 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init27 = tensor.empty() : tensor<256x5120xf16>
  %fill28 = linalg.fill ins(%cst : f16) outs(%init27 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm29 = linalg.matmul ins(%add26, %w_ff0_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill28 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty30 = tensor.empty() : tensor<256x5120xf16>
  %relu31 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm29 : tensor<256x5120xf16>)
    outs(%empty30 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init32 = tensor.empty() : tensor<256x1280xf16>
  %fill33 = linalg.fill ins(%cst : f16) outs(%init32 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm34 = linalg.matmul ins(%relu31, %w_ff0_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill33 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty35 = tensor.empty() : tensor<256x1280xf16>
  %add36 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm34, %add26 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty35 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 1 ===
  %init37 = tensor.empty() : tensor<256x1280xf16>
  %fill38 = linalg.fill ins(%cst : f16) outs(%init37 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm39 = linalg.matmul ins(%add36, %w_q1 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill38 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init40 = tensor.empty() : tensor<256x1280xf16>
  %fill41 = linalg.fill ins(%cst : f16) outs(%init40 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm42 = linalg.matmul ins(%add36, %w_k1 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill41 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init43 = tensor.empty() : tensor<256x1280xf16>
  %fill44 = linalg.fill ins(%cst : f16) outs(%init43 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm45 = linalg.matmul ins(%add36, %w_v1 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill44 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init46 = tensor.empty() : tensor<256x256xf16>
  %fill47 = linalg.fill ins(%cst : f16) outs(%init46 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm48 = linalg.matmul ins(%mm39, %w_kt1 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill47 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty49 = tensor.empty() : tensor<256x256xf16>
  %relu50 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm48 : tensor<256x256xf16>)
    outs(%empty49 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init51 = tensor.empty() : tensor<256x1280xf16>
  %fill52 = linalg.fill ins(%cst : f16) outs(%init51 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm53 = linalg.matmul ins(%relu50, %mm45 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill52 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init54 = tensor.empty() : tensor<256x1280xf16>
  %fill55 = linalg.fill ins(%cst : f16) outs(%init54 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm56 = linalg.matmul ins(%mm53, %w_o1 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill55 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty57 = tensor.empty() : tensor<256x1280xf16>
  %add58 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm56, %add36 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty57 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init59 = tensor.empty() : tensor<256x5120xf16>
  %fill60 = linalg.fill ins(%cst : f16) outs(%init59 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm61 = linalg.matmul ins(%add58, %w_ff1_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill60 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty62 = tensor.empty() : tensor<256x5120xf16>
  %relu63 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm61 : tensor<256x5120xf16>)
    outs(%empty62 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init64 = tensor.empty() : tensor<256x1280xf16>
  %fill65 = linalg.fill ins(%cst : f16) outs(%init64 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm66 = linalg.matmul ins(%relu63, %w_ff1_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill65 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty67 = tensor.empty() : tensor<256x1280xf16>
  %add68 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm66, %add58 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty67 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 2 ===
  %init69 = tensor.empty() : tensor<256x1280xf16>
  %fill70 = linalg.fill ins(%cst : f16) outs(%init69 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm71 = linalg.matmul ins(%add68, %w_q2 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill70 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init72 = tensor.empty() : tensor<256x1280xf16>
  %fill73 = linalg.fill ins(%cst : f16) outs(%init72 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm74 = linalg.matmul ins(%add68, %w_k2 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill73 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init75 = tensor.empty() : tensor<256x1280xf16>
  %fill76 = linalg.fill ins(%cst : f16) outs(%init75 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm77 = linalg.matmul ins(%add68, %w_v2 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill76 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init78 = tensor.empty() : tensor<256x256xf16>
  %fill79 = linalg.fill ins(%cst : f16) outs(%init78 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm80 = linalg.matmul ins(%mm71, %w_kt2 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill79 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty81 = tensor.empty() : tensor<256x256xf16>
  %relu82 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm80 : tensor<256x256xf16>)
    outs(%empty81 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init83 = tensor.empty() : tensor<256x1280xf16>
  %fill84 = linalg.fill ins(%cst : f16) outs(%init83 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm85 = linalg.matmul ins(%relu82, %mm77 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill84 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init86 = tensor.empty() : tensor<256x1280xf16>
  %fill87 = linalg.fill ins(%cst : f16) outs(%init86 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm88 = linalg.matmul ins(%mm85, %w_o2 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill87 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty89 = tensor.empty() : tensor<256x1280xf16>
  %add90 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm88, %add68 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty89 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init91 = tensor.empty() : tensor<256x5120xf16>
  %fill92 = linalg.fill ins(%cst : f16) outs(%init91 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm93 = linalg.matmul ins(%add90, %w_ff2_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill92 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty94 = tensor.empty() : tensor<256x5120xf16>
  %relu95 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm93 : tensor<256x5120xf16>)
    outs(%empty94 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init96 = tensor.empty() : tensor<256x1280xf16>
  %fill97 = linalg.fill ins(%cst : f16) outs(%init96 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm98 = linalg.matmul ins(%relu95, %w_ff2_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill97 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty99 = tensor.empty() : tensor<256x1280xf16>
  %add100 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm98, %add90 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty99 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 3 ===
  %init101 = tensor.empty() : tensor<256x1280xf16>
  %fill102 = linalg.fill ins(%cst : f16) outs(%init101 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm103 = linalg.matmul ins(%add100, %w_q3 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill102 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init104 = tensor.empty() : tensor<256x1280xf16>
  %fill105 = linalg.fill ins(%cst : f16) outs(%init104 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm106 = linalg.matmul ins(%add100, %w_k3 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill105 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init107 = tensor.empty() : tensor<256x1280xf16>
  %fill108 = linalg.fill ins(%cst : f16) outs(%init107 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm109 = linalg.matmul ins(%add100, %w_v3 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill108 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init110 = tensor.empty() : tensor<256x256xf16>
  %fill111 = linalg.fill ins(%cst : f16) outs(%init110 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm112 = linalg.matmul ins(%mm103, %w_kt3 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill111 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty113 = tensor.empty() : tensor<256x256xf16>
  %relu114 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm112 : tensor<256x256xf16>)
    outs(%empty113 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init115 = tensor.empty() : tensor<256x1280xf16>
  %fill116 = linalg.fill ins(%cst : f16) outs(%init115 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm117 = linalg.matmul ins(%relu114, %mm109 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill116 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init118 = tensor.empty() : tensor<256x1280xf16>
  %fill119 = linalg.fill ins(%cst : f16) outs(%init118 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm120 = linalg.matmul ins(%mm117, %w_o3 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill119 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty121 = tensor.empty() : tensor<256x1280xf16>
  %add122 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm120, %add100 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty121 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init123 = tensor.empty() : tensor<256x5120xf16>
  %fill124 = linalg.fill ins(%cst : f16) outs(%init123 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm125 = linalg.matmul ins(%add122, %w_ff3_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill124 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty126 = tensor.empty() : tensor<256x5120xf16>
  %relu127 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm125 : tensor<256x5120xf16>)
    outs(%empty126 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init128 = tensor.empty() : tensor<256x1280xf16>
  %fill129 = linalg.fill ins(%cst : f16) outs(%init128 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm130 = linalg.matmul ins(%relu127, %w_ff3_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill129 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty131 = tensor.empty() : tensor<256x1280xf16>
  %add132 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm130, %add122 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty131 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 4 ===
  %init133 = tensor.empty() : tensor<256x1280xf16>
  %fill134 = linalg.fill ins(%cst : f16) outs(%init133 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm135 = linalg.matmul ins(%add132, %w_q4 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill134 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init136 = tensor.empty() : tensor<256x1280xf16>
  %fill137 = linalg.fill ins(%cst : f16) outs(%init136 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm138 = linalg.matmul ins(%add132, %w_k4 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill137 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init139 = tensor.empty() : tensor<256x1280xf16>
  %fill140 = linalg.fill ins(%cst : f16) outs(%init139 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm141 = linalg.matmul ins(%add132, %w_v4 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill140 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init142 = tensor.empty() : tensor<256x256xf16>
  %fill143 = linalg.fill ins(%cst : f16) outs(%init142 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm144 = linalg.matmul ins(%mm135, %w_kt4 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill143 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty145 = tensor.empty() : tensor<256x256xf16>
  %relu146 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm144 : tensor<256x256xf16>)
    outs(%empty145 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init147 = tensor.empty() : tensor<256x1280xf16>
  %fill148 = linalg.fill ins(%cst : f16) outs(%init147 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm149 = linalg.matmul ins(%relu146, %mm141 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill148 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init150 = tensor.empty() : tensor<256x1280xf16>
  %fill151 = linalg.fill ins(%cst : f16) outs(%init150 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm152 = linalg.matmul ins(%mm149, %w_o4 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill151 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty153 = tensor.empty() : tensor<256x1280xf16>
  %add154 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm152, %add132 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty153 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init155 = tensor.empty() : tensor<256x5120xf16>
  %fill156 = linalg.fill ins(%cst : f16) outs(%init155 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm157 = linalg.matmul ins(%add154, %w_ff4_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill156 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty158 = tensor.empty() : tensor<256x5120xf16>
  %relu159 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm157 : tensor<256x5120xf16>)
    outs(%empty158 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init160 = tensor.empty() : tensor<256x1280xf16>
  %fill161 = linalg.fill ins(%cst : f16) outs(%init160 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm162 = linalg.matmul ins(%relu159, %w_ff4_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill161 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty163 = tensor.empty() : tensor<256x1280xf16>
  %add164 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm162, %add154 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty163 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 5 ===
  %init165 = tensor.empty() : tensor<256x1280xf16>
  %fill166 = linalg.fill ins(%cst : f16) outs(%init165 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm167 = linalg.matmul ins(%add164, %w_q5 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill166 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init168 = tensor.empty() : tensor<256x1280xf16>
  %fill169 = linalg.fill ins(%cst : f16) outs(%init168 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm170 = linalg.matmul ins(%add164, %w_k5 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill169 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init171 = tensor.empty() : tensor<256x1280xf16>
  %fill172 = linalg.fill ins(%cst : f16) outs(%init171 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm173 = linalg.matmul ins(%add164, %w_v5 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill172 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init174 = tensor.empty() : tensor<256x256xf16>
  %fill175 = linalg.fill ins(%cst : f16) outs(%init174 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm176 = linalg.matmul ins(%mm167, %w_kt5 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill175 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty177 = tensor.empty() : tensor<256x256xf16>
  %relu178 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm176 : tensor<256x256xf16>)
    outs(%empty177 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init179 = tensor.empty() : tensor<256x1280xf16>
  %fill180 = linalg.fill ins(%cst : f16) outs(%init179 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm181 = linalg.matmul ins(%relu178, %mm173 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill180 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init182 = tensor.empty() : tensor<256x1280xf16>
  %fill183 = linalg.fill ins(%cst : f16) outs(%init182 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm184 = linalg.matmul ins(%mm181, %w_o5 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill183 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty185 = tensor.empty() : tensor<256x1280xf16>
  %add186 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm184, %add164 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty185 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init187 = tensor.empty() : tensor<256x5120xf16>
  %fill188 = linalg.fill ins(%cst : f16) outs(%init187 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm189 = linalg.matmul ins(%add186, %w_ff5_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill188 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty190 = tensor.empty() : tensor<256x5120xf16>
  %relu191 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm189 : tensor<256x5120xf16>)
    outs(%empty190 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init192 = tensor.empty() : tensor<256x1280xf16>
  %fill193 = linalg.fill ins(%cst : f16) outs(%init192 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm194 = linalg.matmul ins(%relu191, %w_ff5_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill193 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty195 = tensor.empty() : tensor<256x1280xf16>
  %add196 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm194, %add186 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty195 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 6 ===
  %init197 = tensor.empty() : tensor<256x1280xf16>
  %fill198 = linalg.fill ins(%cst : f16) outs(%init197 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm199 = linalg.matmul ins(%add196, %w_q6 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill198 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init200 = tensor.empty() : tensor<256x1280xf16>
  %fill201 = linalg.fill ins(%cst : f16) outs(%init200 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm202 = linalg.matmul ins(%add196, %w_k6 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill201 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init203 = tensor.empty() : tensor<256x1280xf16>
  %fill204 = linalg.fill ins(%cst : f16) outs(%init203 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm205 = linalg.matmul ins(%add196, %w_v6 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill204 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init206 = tensor.empty() : tensor<256x256xf16>
  %fill207 = linalg.fill ins(%cst : f16) outs(%init206 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm208 = linalg.matmul ins(%mm199, %w_kt6 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill207 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty209 = tensor.empty() : tensor<256x256xf16>
  %relu210 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm208 : tensor<256x256xf16>)
    outs(%empty209 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init211 = tensor.empty() : tensor<256x1280xf16>
  %fill212 = linalg.fill ins(%cst : f16) outs(%init211 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm213 = linalg.matmul ins(%relu210, %mm205 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill212 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init214 = tensor.empty() : tensor<256x1280xf16>
  %fill215 = linalg.fill ins(%cst : f16) outs(%init214 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm216 = linalg.matmul ins(%mm213, %w_o6 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill215 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty217 = tensor.empty() : tensor<256x1280xf16>
  %add218 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm216, %add196 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty217 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init219 = tensor.empty() : tensor<256x5120xf16>
  %fill220 = linalg.fill ins(%cst : f16) outs(%init219 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm221 = linalg.matmul ins(%add218, %w_ff6_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill220 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty222 = tensor.empty() : tensor<256x5120xf16>
  %relu223 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm221 : tensor<256x5120xf16>)
    outs(%empty222 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init224 = tensor.empty() : tensor<256x1280xf16>
  %fill225 = linalg.fill ins(%cst : f16) outs(%init224 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm226 = linalg.matmul ins(%relu223, %w_ff6_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill225 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty227 = tensor.empty() : tensor<256x1280xf16>
  %add228 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm226, %add218 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty227 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 7 ===
  %init229 = tensor.empty() : tensor<256x1280xf16>
  %fill230 = linalg.fill ins(%cst : f16) outs(%init229 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm231 = linalg.matmul ins(%add228, %w_q7 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill230 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init232 = tensor.empty() : tensor<256x1280xf16>
  %fill233 = linalg.fill ins(%cst : f16) outs(%init232 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm234 = linalg.matmul ins(%add228, %w_k7 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill233 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init235 = tensor.empty() : tensor<256x1280xf16>
  %fill236 = linalg.fill ins(%cst : f16) outs(%init235 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm237 = linalg.matmul ins(%add228, %w_v7 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill236 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init238 = tensor.empty() : tensor<256x256xf16>
  %fill239 = linalg.fill ins(%cst : f16) outs(%init238 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm240 = linalg.matmul ins(%mm231, %w_kt7 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill239 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty241 = tensor.empty() : tensor<256x256xf16>
  %relu242 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm240 : tensor<256x256xf16>)
    outs(%empty241 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init243 = tensor.empty() : tensor<256x1280xf16>
  %fill244 = linalg.fill ins(%cst : f16) outs(%init243 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm245 = linalg.matmul ins(%relu242, %mm237 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill244 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init246 = tensor.empty() : tensor<256x1280xf16>
  %fill247 = linalg.fill ins(%cst : f16) outs(%init246 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm248 = linalg.matmul ins(%mm245, %w_o7 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill247 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty249 = tensor.empty() : tensor<256x1280xf16>
  %add250 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm248, %add228 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty249 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init251 = tensor.empty() : tensor<256x5120xf16>
  %fill252 = linalg.fill ins(%cst : f16) outs(%init251 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm253 = linalg.matmul ins(%add250, %w_ff7_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill252 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty254 = tensor.empty() : tensor<256x5120xf16>
  %relu255 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm253 : tensor<256x5120xf16>)
    outs(%empty254 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init256 = tensor.empty() : tensor<256x1280xf16>
  %fill257 = linalg.fill ins(%cst : f16) outs(%init256 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm258 = linalg.matmul ins(%relu255, %w_ff7_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill257 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty259 = tensor.empty() : tensor<256x1280xf16>
  %add260 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm258, %add250 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty259 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 8 ===
  %init261 = tensor.empty() : tensor<256x1280xf16>
  %fill262 = linalg.fill ins(%cst : f16) outs(%init261 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm263 = linalg.matmul ins(%add260, %w_q8 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill262 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init264 = tensor.empty() : tensor<256x1280xf16>
  %fill265 = linalg.fill ins(%cst : f16) outs(%init264 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm266 = linalg.matmul ins(%add260, %w_k8 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill265 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init267 = tensor.empty() : tensor<256x1280xf16>
  %fill268 = linalg.fill ins(%cst : f16) outs(%init267 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm269 = linalg.matmul ins(%add260, %w_v8 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill268 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init270 = tensor.empty() : tensor<256x256xf16>
  %fill271 = linalg.fill ins(%cst : f16) outs(%init270 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm272 = linalg.matmul ins(%mm263, %w_kt8 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill271 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty273 = tensor.empty() : tensor<256x256xf16>
  %relu274 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm272 : tensor<256x256xf16>)
    outs(%empty273 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init275 = tensor.empty() : tensor<256x1280xf16>
  %fill276 = linalg.fill ins(%cst : f16) outs(%init275 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm277 = linalg.matmul ins(%relu274, %mm269 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill276 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init278 = tensor.empty() : tensor<256x1280xf16>
  %fill279 = linalg.fill ins(%cst : f16) outs(%init278 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm280 = linalg.matmul ins(%mm277, %w_o8 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill279 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty281 = tensor.empty() : tensor<256x1280xf16>
  %add282 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm280, %add260 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty281 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init283 = tensor.empty() : tensor<256x5120xf16>
  %fill284 = linalg.fill ins(%cst : f16) outs(%init283 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm285 = linalg.matmul ins(%add282, %w_ff8_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill284 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty286 = tensor.empty() : tensor<256x5120xf16>
  %relu287 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm285 : tensor<256x5120xf16>)
    outs(%empty286 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init288 = tensor.empty() : tensor<256x1280xf16>
  %fill289 = linalg.fill ins(%cst : f16) outs(%init288 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm290 = linalg.matmul ins(%relu287, %w_ff8_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill289 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty291 = tensor.empty() : tensor<256x1280xf16>
  %add292 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm290, %add282 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty291 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 9 ===
  %init293 = tensor.empty() : tensor<256x1280xf16>
  %fill294 = linalg.fill ins(%cst : f16) outs(%init293 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm295 = linalg.matmul ins(%add292, %w_q9 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill294 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init296 = tensor.empty() : tensor<256x1280xf16>
  %fill297 = linalg.fill ins(%cst : f16) outs(%init296 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm298 = linalg.matmul ins(%add292, %w_k9 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill297 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init299 = tensor.empty() : tensor<256x1280xf16>
  %fill300 = linalg.fill ins(%cst : f16) outs(%init299 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm301 = linalg.matmul ins(%add292, %w_v9 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill300 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init302 = tensor.empty() : tensor<256x256xf16>
  %fill303 = linalg.fill ins(%cst : f16) outs(%init302 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm304 = linalg.matmul ins(%mm295, %w_kt9 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill303 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty305 = tensor.empty() : tensor<256x256xf16>
  %relu306 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm304 : tensor<256x256xf16>)
    outs(%empty305 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init307 = tensor.empty() : tensor<256x1280xf16>
  %fill308 = linalg.fill ins(%cst : f16) outs(%init307 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm309 = linalg.matmul ins(%relu306, %mm301 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill308 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init310 = tensor.empty() : tensor<256x1280xf16>
  %fill311 = linalg.fill ins(%cst : f16) outs(%init310 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm312 = linalg.matmul ins(%mm309, %w_o9 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill311 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty313 = tensor.empty() : tensor<256x1280xf16>
  %add314 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm312, %add292 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty313 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init315 = tensor.empty() : tensor<256x5120xf16>
  %fill316 = linalg.fill ins(%cst : f16) outs(%init315 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm317 = linalg.matmul ins(%add314, %w_ff9_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill316 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty318 = tensor.empty() : tensor<256x5120xf16>
  %relu319 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm317 : tensor<256x5120xf16>)
    outs(%empty318 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init320 = tensor.empty() : tensor<256x1280xf16>
  %fill321 = linalg.fill ins(%cst : f16) outs(%init320 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm322 = linalg.matmul ins(%relu319, %w_ff9_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill321 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty323 = tensor.empty() : tensor<256x1280xf16>
  %add324 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm322, %add314 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty323 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 10 ===
  %init325 = tensor.empty() : tensor<256x1280xf16>
  %fill326 = linalg.fill ins(%cst : f16) outs(%init325 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm327 = linalg.matmul ins(%add324, %w_q10 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill326 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init328 = tensor.empty() : tensor<256x1280xf16>
  %fill329 = linalg.fill ins(%cst : f16) outs(%init328 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm330 = linalg.matmul ins(%add324, %w_k10 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill329 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init331 = tensor.empty() : tensor<256x1280xf16>
  %fill332 = linalg.fill ins(%cst : f16) outs(%init331 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm333 = linalg.matmul ins(%add324, %w_v10 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill332 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init334 = tensor.empty() : tensor<256x256xf16>
  %fill335 = linalg.fill ins(%cst : f16) outs(%init334 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm336 = linalg.matmul ins(%mm327, %w_kt10 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill335 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty337 = tensor.empty() : tensor<256x256xf16>
  %relu338 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm336 : tensor<256x256xf16>)
    outs(%empty337 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init339 = tensor.empty() : tensor<256x1280xf16>
  %fill340 = linalg.fill ins(%cst : f16) outs(%init339 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm341 = linalg.matmul ins(%relu338, %mm333 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill340 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init342 = tensor.empty() : tensor<256x1280xf16>
  %fill343 = linalg.fill ins(%cst : f16) outs(%init342 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm344 = linalg.matmul ins(%mm341, %w_o10 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill343 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty345 = tensor.empty() : tensor<256x1280xf16>
  %add346 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm344, %add324 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty345 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init347 = tensor.empty() : tensor<256x5120xf16>
  %fill348 = linalg.fill ins(%cst : f16) outs(%init347 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm349 = linalg.matmul ins(%add346, %w_ff10_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill348 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty350 = tensor.empty() : tensor<256x5120xf16>
  %relu351 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm349 : tensor<256x5120xf16>)
    outs(%empty350 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init352 = tensor.empty() : tensor<256x1280xf16>
  %fill353 = linalg.fill ins(%cst : f16) outs(%init352 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm354 = linalg.matmul ins(%relu351, %w_ff10_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill353 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty355 = tensor.empty() : tensor<256x1280xf16>
  %add356 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm354, %add346 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty355 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 11 ===
  %init357 = tensor.empty() : tensor<256x1280xf16>
  %fill358 = linalg.fill ins(%cst : f16) outs(%init357 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm359 = linalg.matmul ins(%add356, %w_q11 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill358 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init360 = tensor.empty() : tensor<256x1280xf16>
  %fill361 = linalg.fill ins(%cst : f16) outs(%init360 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm362 = linalg.matmul ins(%add356, %w_k11 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill361 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init363 = tensor.empty() : tensor<256x1280xf16>
  %fill364 = linalg.fill ins(%cst : f16) outs(%init363 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm365 = linalg.matmul ins(%add356, %w_v11 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill364 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init366 = tensor.empty() : tensor<256x256xf16>
  %fill367 = linalg.fill ins(%cst : f16) outs(%init366 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm368 = linalg.matmul ins(%mm359, %w_kt11 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill367 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty369 = tensor.empty() : tensor<256x256xf16>
  %relu370 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm368 : tensor<256x256xf16>)
    outs(%empty369 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init371 = tensor.empty() : tensor<256x1280xf16>
  %fill372 = linalg.fill ins(%cst : f16) outs(%init371 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm373 = linalg.matmul ins(%relu370, %mm365 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill372 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init374 = tensor.empty() : tensor<256x1280xf16>
  %fill375 = linalg.fill ins(%cst : f16) outs(%init374 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm376 = linalg.matmul ins(%mm373, %w_o11 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill375 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty377 = tensor.empty() : tensor<256x1280xf16>
  %add378 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm376, %add356 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty377 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init379 = tensor.empty() : tensor<256x5120xf16>
  %fill380 = linalg.fill ins(%cst : f16) outs(%init379 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm381 = linalg.matmul ins(%add378, %w_ff11_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill380 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty382 = tensor.empty() : tensor<256x5120xf16>
  %relu383 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm381 : tensor<256x5120xf16>)
    outs(%empty382 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init384 = tensor.empty() : tensor<256x1280xf16>
  %fill385 = linalg.fill ins(%cst : f16) outs(%init384 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm386 = linalg.matmul ins(%relu383, %w_ff11_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill385 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty387 = tensor.empty() : tensor<256x1280xf16>
  %add388 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm386, %add378 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty387 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 12 ===
  %init389 = tensor.empty() : tensor<256x1280xf16>
  %fill390 = linalg.fill ins(%cst : f16) outs(%init389 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm391 = linalg.matmul ins(%add388, %w_q12 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill390 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init392 = tensor.empty() : tensor<256x1280xf16>
  %fill393 = linalg.fill ins(%cst : f16) outs(%init392 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm394 = linalg.matmul ins(%add388, %w_k12 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill393 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init395 = tensor.empty() : tensor<256x1280xf16>
  %fill396 = linalg.fill ins(%cst : f16) outs(%init395 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm397 = linalg.matmul ins(%add388, %w_v12 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill396 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init398 = tensor.empty() : tensor<256x256xf16>
  %fill399 = linalg.fill ins(%cst : f16) outs(%init398 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm400 = linalg.matmul ins(%mm391, %w_kt12 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill399 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty401 = tensor.empty() : tensor<256x256xf16>
  %relu402 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm400 : tensor<256x256xf16>)
    outs(%empty401 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init403 = tensor.empty() : tensor<256x1280xf16>
  %fill404 = linalg.fill ins(%cst : f16) outs(%init403 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm405 = linalg.matmul ins(%relu402, %mm397 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill404 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init406 = tensor.empty() : tensor<256x1280xf16>
  %fill407 = linalg.fill ins(%cst : f16) outs(%init406 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm408 = linalg.matmul ins(%mm405, %w_o12 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill407 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty409 = tensor.empty() : tensor<256x1280xf16>
  %add410 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm408, %add388 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty409 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init411 = tensor.empty() : tensor<256x5120xf16>
  %fill412 = linalg.fill ins(%cst : f16) outs(%init411 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm413 = linalg.matmul ins(%add410, %w_ff12_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill412 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty414 = tensor.empty() : tensor<256x5120xf16>
  %relu415 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm413 : tensor<256x5120xf16>)
    outs(%empty414 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init416 = tensor.empty() : tensor<256x1280xf16>
  %fill417 = linalg.fill ins(%cst : f16) outs(%init416 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm418 = linalg.matmul ins(%relu415, %w_ff12_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill417 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty419 = tensor.empty() : tensor<256x1280xf16>
  %add420 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm418, %add410 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty419 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 13 ===
  %init421 = tensor.empty() : tensor<256x1280xf16>
  %fill422 = linalg.fill ins(%cst : f16) outs(%init421 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm423 = linalg.matmul ins(%add420, %w_q13 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill422 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init424 = tensor.empty() : tensor<256x1280xf16>
  %fill425 = linalg.fill ins(%cst : f16) outs(%init424 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm426 = linalg.matmul ins(%add420, %w_k13 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill425 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init427 = tensor.empty() : tensor<256x1280xf16>
  %fill428 = linalg.fill ins(%cst : f16) outs(%init427 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm429 = linalg.matmul ins(%add420, %w_v13 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill428 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init430 = tensor.empty() : tensor<256x256xf16>
  %fill431 = linalg.fill ins(%cst : f16) outs(%init430 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm432 = linalg.matmul ins(%mm423, %w_kt13 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill431 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty433 = tensor.empty() : tensor<256x256xf16>
  %relu434 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm432 : tensor<256x256xf16>)
    outs(%empty433 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init435 = tensor.empty() : tensor<256x1280xf16>
  %fill436 = linalg.fill ins(%cst : f16) outs(%init435 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm437 = linalg.matmul ins(%relu434, %mm429 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill436 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init438 = tensor.empty() : tensor<256x1280xf16>
  %fill439 = linalg.fill ins(%cst : f16) outs(%init438 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm440 = linalg.matmul ins(%mm437, %w_o13 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill439 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty441 = tensor.empty() : tensor<256x1280xf16>
  %add442 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm440, %add420 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty441 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init443 = tensor.empty() : tensor<256x5120xf16>
  %fill444 = linalg.fill ins(%cst : f16) outs(%init443 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm445 = linalg.matmul ins(%add442, %w_ff13_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill444 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty446 = tensor.empty() : tensor<256x5120xf16>
  %relu447 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm445 : tensor<256x5120xf16>)
    outs(%empty446 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init448 = tensor.empty() : tensor<256x1280xf16>
  %fill449 = linalg.fill ins(%cst : f16) outs(%init448 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm450 = linalg.matmul ins(%relu447, %w_ff13_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill449 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty451 = tensor.empty() : tensor<256x1280xf16>
  %add452 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm450, %add442 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty451 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 14 ===
  %init453 = tensor.empty() : tensor<256x1280xf16>
  %fill454 = linalg.fill ins(%cst : f16) outs(%init453 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm455 = linalg.matmul ins(%add452, %w_q14 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill454 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init456 = tensor.empty() : tensor<256x1280xf16>
  %fill457 = linalg.fill ins(%cst : f16) outs(%init456 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm458 = linalg.matmul ins(%add452, %w_k14 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill457 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init459 = tensor.empty() : tensor<256x1280xf16>
  %fill460 = linalg.fill ins(%cst : f16) outs(%init459 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm461 = linalg.matmul ins(%add452, %w_v14 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill460 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init462 = tensor.empty() : tensor<256x256xf16>
  %fill463 = linalg.fill ins(%cst : f16) outs(%init462 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm464 = linalg.matmul ins(%mm455, %w_kt14 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill463 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty465 = tensor.empty() : tensor<256x256xf16>
  %relu466 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm464 : tensor<256x256xf16>)
    outs(%empty465 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init467 = tensor.empty() : tensor<256x1280xf16>
  %fill468 = linalg.fill ins(%cst : f16) outs(%init467 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm469 = linalg.matmul ins(%relu466, %mm461 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill468 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init470 = tensor.empty() : tensor<256x1280xf16>
  %fill471 = linalg.fill ins(%cst : f16) outs(%init470 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm472 = linalg.matmul ins(%mm469, %w_o14 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill471 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty473 = tensor.empty() : tensor<256x1280xf16>
  %add474 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm472, %add452 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty473 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init475 = tensor.empty() : tensor<256x5120xf16>
  %fill476 = linalg.fill ins(%cst : f16) outs(%init475 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm477 = linalg.matmul ins(%add474, %w_ff14_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill476 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty478 = tensor.empty() : tensor<256x5120xf16>
  %relu479 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm477 : tensor<256x5120xf16>)
    outs(%empty478 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init480 = tensor.empty() : tensor<256x1280xf16>
  %fill481 = linalg.fill ins(%cst : f16) outs(%init480 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm482 = linalg.matmul ins(%relu479, %w_ff14_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill481 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty483 = tensor.empty() : tensor<256x1280xf16>
  %add484 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm482, %add474 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty483 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 15 ===
  %init485 = tensor.empty() : tensor<256x1280xf16>
  %fill486 = linalg.fill ins(%cst : f16) outs(%init485 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm487 = linalg.matmul ins(%add484, %w_q15 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill486 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init488 = tensor.empty() : tensor<256x1280xf16>
  %fill489 = linalg.fill ins(%cst : f16) outs(%init488 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm490 = linalg.matmul ins(%add484, %w_k15 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill489 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init491 = tensor.empty() : tensor<256x1280xf16>
  %fill492 = linalg.fill ins(%cst : f16) outs(%init491 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm493 = linalg.matmul ins(%add484, %w_v15 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill492 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init494 = tensor.empty() : tensor<256x256xf16>
  %fill495 = linalg.fill ins(%cst : f16) outs(%init494 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm496 = linalg.matmul ins(%mm487, %w_kt15 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill495 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty497 = tensor.empty() : tensor<256x256xf16>
  %relu498 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm496 : tensor<256x256xf16>)
    outs(%empty497 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init499 = tensor.empty() : tensor<256x1280xf16>
  %fill500 = linalg.fill ins(%cst : f16) outs(%init499 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm501 = linalg.matmul ins(%relu498, %mm493 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill500 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init502 = tensor.empty() : tensor<256x1280xf16>
  %fill503 = linalg.fill ins(%cst : f16) outs(%init502 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm504 = linalg.matmul ins(%mm501, %w_o15 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill503 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty505 = tensor.empty() : tensor<256x1280xf16>
  %add506 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm504, %add484 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty505 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init507 = tensor.empty() : tensor<256x5120xf16>
  %fill508 = linalg.fill ins(%cst : f16) outs(%init507 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm509 = linalg.matmul ins(%add506, %w_ff15_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill508 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty510 = tensor.empty() : tensor<256x5120xf16>
  %relu511 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm509 : tensor<256x5120xf16>)
    outs(%empty510 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init512 = tensor.empty() : tensor<256x1280xf16>
  %fill513 = linalg.fill ins(%cst : f16) outs(%init512 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm514 = linalg.matmul ins(%relu511, %w_ff15_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill513 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty515 = tensor.empty() : tensor<256x1280xf16>
  %add516 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm514, %add506 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty515 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 16 ===
  %init517 = tensor.empty() : tensor<256x1280xf16>
  %fill518 = linalg.fill ins(%cst : f16) outs(%init517 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm519 = linalg.matmul ins(%add516, %w_q16 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill518 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init520 = tensor.empty() : tensor<256x1280xf16>
  %fill521 = linalg.fill ins(%cst : f16) outs(%init520 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm522 = linalg.matmul ins(%add516, %w_k16 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill521 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init523 = tensor.empty() : tensor<256x1280xf16>
  %fill524 = linalg.fill ins(%cst : f16) outs(%init523 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm525 = linalg.matmul ins(%add516, %w_v16 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill524 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init526 = tensor.empty() : tensor<256x256xf16>
  %fill527 = linalg.fill ins(%cst : f16) outs(%init526 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm528 = linalg.matmul ins(%mm519, %w_kt16 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill527 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty529 = tensor.empty() : tensor<256x256xf16>
  %relu530 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm528 : tensor<256x256xf16>)
    outs(%empty529 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init531 = tensor.empty() : tensor<256x1280xf16>
  %fill532 = linalg.fill ins(%cst : f16) outs(%init531 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm533 = linalg.matmul ins(%relu530, %mm525 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill532 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init534 = tensor.empty() : tensor<256x1280xf16>
  %fill535 = linalg.fill ins(%cst : f16) outs(%init534 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm536 = linalg.matmul ins(%mm533, %w_o16 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill535 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty537 = tensor.empty() : tensor<256x1280xf16>
  %add538 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm536, %add516 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty537 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init539 = tensor.empty() : tensor<256x5120xf16>
  %fill540 = linalg.fill ins(%cst : f16) outs(%init539 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm541 = linalg.matmul ins(%add538, %w_ff16_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill540 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty542 = tensor.empty() : tensor<256x5120xf16>
  %relu543 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm541 : tensor<256x5120xf16>)
    outs(%empty542 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init544 = tensor.empty() : tensor<256x1280xf16>
  %fill545 = linalg.fill ins(%cst : f16) outs(%init544 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm546 = linalg.matmul ins(%relu543, %w_ff16_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill545 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty547 = tensor.empty() : tensor<256x1280xf16>
  %add548 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm546, %add538 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty547 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 17 ===
  %init549 = tensor.empty() : tensor<256x1280xf16>
  %fill550 = linalg.fill ins(%cst : f16) outs(%init549 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm551 = linalg.matmul ins(%add548, %w_q17 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill550 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init552 = tensor.empty() : tensor<256x1280xf16>
  %fill553 = linalg.fill ins(%cst : f16) outs(%init552 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm554 = linalg.matmul ins(%add548, %w_k17 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill553 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init555 = tensor.empty() : tensor<256x1280xf16>
  %fill556 = linalg.fill ins(%cst : f16) outs(%init555 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm557 = linalg.matmul ins(%add548, %w_v17 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill556 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init558 = tensor.empty() : tensor<256x256xf16>
  %fill559 = linalg.fill ins(%cst : f16) outs(%init558 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm560 = linalg.matmul ins(%mm551, %w_kt17 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill559 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty561 = tensor.empty() : tensor<256x256xf16>
  %relu562 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm560 : tensor<256x256xf16>)
    outs(%empty561 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init563 = tensor.empty() : tensor<256x1280xf16>
  %fill564 = linalg.fill ins(%cst : f16) outs(%init563 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm565 = linalg.matmul ins(%relu562, %mm557 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill564 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init566 = tensor.empty() : tensor<256x1280xf16>
  %fill567 = linalg.fill ins(%cst : f16) outs(%init566 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm568 = linalg.matmul ins(%mm565, %w_o17 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill567 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty569 = tensor.empty() : tensor<256x1280xf16>
  %add570 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm568, %add548 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty569 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init571 = tensor.empty() : tensor<256x5120xf16>
  %fill572 = linalg.fill ins(%cst : f16) outs(%init571 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm573 = linalg.matmul ins(%add570, %w_ff17_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill572 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty574 = tensor.empty() : tensor<256x5120xf16>
  %relu575 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm573 : tensor<256x5120xf16>)
    outs(%empty574 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init576 = tensor.empty() : tensor<256x1280xf16>
  %fill577 = linalg.fill ins(%cst : f16) outs(%init576 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm578 = linalg.matmul ins(%relu575, %w_ff17_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill577 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty579 = tensor.empty() : tensor<256x1280xf16>
  %add580 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm578, %add570 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty579 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 18 ===
  %init581 = tensor.empty() : tensor<256x1280xf16>
  %fill582 = linalg.fill ins(%cst : f16) outs(%init581 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm583 = linalg.matmul ins(%add580, %w_q18 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill582 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init584 = tensor.empty() : tensor<256x1280xf16>
  %fill585 = linalg.fill ins(%cst : f16) outs(%init584 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm586 = linalg.matmul ins(%add580, %w_k18 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill585 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init587 = tensor.empty() : tensor<256x1280xf16>
  %fill588 = linalg.fill ins(%cst : f16) outs(%init587 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm589 = linalg.matmul ins(%add580, %w_v18 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill588 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init590 = tensor.empty() : tensor<256x256xf16>
  %fill591 = linalg.fill ins(%cst : f16) outs(%init590 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm592 = linalg.matmul ins(%mm583, %w_kt18 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill591 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty593 = tensor.empty() : tensor<256x256xf16>
  %relu594 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm592 : tensor<256x256xf16>)
    outs(%empty593 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init595 = tensor.empty() : tensor<256x1280xf16>
  %fill596 = linalg.fill ins(%cst : f16) outs(%init595 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm597 = linalg.matmul ins(%relu594, %mm589 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill596 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init598 = tensor.empty() : tensor<256x1280xf16>
  %fill599 = linalg.fill ins(%cst : f16) outs(%init598 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm600 = linalg.matmul ins(%mm597, %w_o18 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill599 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty601 = tensor.empty() : tensor<256x1280xf16>
  %add602 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm600, %add580 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty601 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init603 = tensor.empty() : tensor<256x5120xf16>
  %fill604 = linalg.fill ins(%cst : f16) outs(%init603 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm605 = linalg.matmul ins(%add602, %w_ff18_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill604 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty606 = tensor.empty() : tensor<256x5120xf16>
  %relu607 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm605 : tensor<256x5120xf16>)
    outs(%empty606 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init608 = tensor.empty() : tensor<256x1280xf16>
  %fill609 = linalg.fill ins(%cst : f16) outs(%init608 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm610 = linalg.matmul ins(%relu607, %w_ff18_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill609 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty611 = tensor.empty() : tensor<256x1280xf16>
  %add612 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm610, %add602 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty611 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 19 ===
  %init613 = tensor.empty() : tensor<256x1280xf16>
  %fill614 = linalg.fill ins(%cst : f16) outs(%init613 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm615 = linalg.matmul ins(%add612, %w_q19 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill614 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init616 = tensor.empty() : tensor<256x1280xf16>
  %fill617 = linalg.fill ins(%cst : f16) outs(%init616 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm618 = linalg.matmul ins(%add612, %w_k19 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill617 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init619 = tensor.empty() : tensor<256x1280xf16>
  %fill620 = linalg.fill ins(%cst : f16) outs(%init619 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm621 = linalg.matmul ins(%add612, %w_v19 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill620 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init622 = tensor.empty() : tensor<256x256xf16>
  %fill623 = linalg.fill ins(%cst : f16) outs(%init622 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm624 = linalg.matmul ins(%mm615, %w_kt19 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill623 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty625 = tensor.empty() : tensor<256x256xf16>
  %relu626 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm624 : tensor<256x256xf16>)
    outs(%empty625 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init627 = tensor.empty() : tensor<256x1280xf16>
  %fill628 = linalg.fill ins(%cst : f16) outs(%init627 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm629 = linalg.matmul ins(%relu626, %mm621 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill628 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init630 = tensor.empty() : tensor<256x1280xf16>
  %fill631 = linalg.fill ins(%cst : f16) outs(%init630 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm632 = linalg.matmul ins(%mm629, %w_o19 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill631 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty633 = tensor.empty() : tensor<256x1280xf16>
  %add634 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm632, %add612 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty633 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init635 = tensor.empty() : tensor<256x5120xf16>
  %fill636 = linalg.fill ins(%cst : f16) outs(%init635 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm637 = linalg.matmul ins(%add634, %w_ff19_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill636 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty638 = tensor.empty() : tensor<256x5120xf16>
  %relu639 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm637 : tensor<256x5120xf16>)
    outs(%empty638 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init640 = tensor.empty() : tensor<256x1280xf16>
  %fill641 = linalg.fill ins(%cst : f16) outs(%init640 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm642 = linalg.matmul ins(%relu639, %w_ff19_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill641 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty643 = tensor.empty() : tensor<256x1280xf16>
  %add644 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm642, %add634 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty643 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 20 ===
  %init645 = tensor.empty() : tensor<256x1280xf16>
  %fill646 = linalg.fill ins(%cst : f16) outs(%init645 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm647 = linalg.matmul ins(%add644, %w_q20 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill646 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init648 = tensor.empty() : tensor<256x1280xf16>
  %fill649 = linalg.fill ins(%cst : f16) outs(%init648 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm650 = linalg.matmul ins(%add644, %w_k20 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill649 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init651 = tensor.empty() : tensor<256x1280xf16>
  %fill652 = linalg.fill ins(%cst : f16) outs(%init651 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm653 = linalg.matmul ins(%add644, %w_v20 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill652 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init654 = tensor.empty() : tensor<256x256xf16>
  %fill655 = linalg.fill ins(%cst : f16) outs(%init654 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm656 = linalg.matmul ins(%mm647, %w_kt20 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill655 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty657 = tensor.empty() : tensor<256x256xf16>
  %relu658 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm656 : tensor<256x256xf16>)
    outs(%empty657 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init659 = tensor.empty() : tensor<256x1280xf16>
  %fill660 = linalg.fill ins(%cst : f16) outs(%init659 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm661 = linalg.matmul ins(%relu658, %mm653 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill660 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init662 = tensor.empty() : tensor<256x1280xf16>
  %fill663 = linalg.fill ins(%cst : f16) outs(%init662 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm664 = linalg.matmul ins(%mm661, %w_o20 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill663 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty665 = tensor.empty() : tensor<256x1280xf16>
  %add666 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm664, %add644 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty665 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init667 = tensor.empty() : tensor<256x5120xf16>
  %fill668 = linalg.fill ins(%cst : f16) outs(%init667 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm669 = linalg.matmul ins(%add666, %w_ff20_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill668 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty670 = tensor.empty() : tensor<256x5120xf16>
  %relu671 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm669 : tensor<256x5120xf16>)
    outs(%empty670 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init672 = tensor.empty() : tensor<256x1280xf16>
  %fill673 = linalg.fill ins(%cst : f16) outs(%init672 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm674 = linalg.matmul ins(%relu671, %w_ff20_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill673 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty675 = tensor.empty() : tensor<256x1280xf16>
  %add676 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm674, %add666 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty675 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 21 ===
  %init677 = tensor.empty() : tensor<256x1280xf16>
  %fill678 = linalg.fill ins(%cst : f16) outs(%init677 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm679 = linalg.matmul ins(%add676, %w_q21 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill678 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init680 = tensor.empty() : tensor<256x1280xf16>
  %fill681 = linalg.fill ins(%cst : f16) outs(%init680 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm682 = linalg.matmul ins(%add676, %w_k21 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill681 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init683 = tensor.empty() : tensor<256x1280xf16>
  %fill684 = linalg.fill ins(%cst : f16) outs(%init683 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm685 = linalg.matmul ins(%add676, %w_v21 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill684 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init686 = tensor.empty() : tensor<256x256xf16>
  %fill687 = linalg.fill ins(%cst : f16) outs(%init686 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm688 = linalg.matmul ins(%mm679, %w_kt21 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill687 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty689 = tensor.empty() : tensor<256x256xf16>
  %relu690 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm688 : tensor<256x256xf16>)
    outs(%empty689 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init691 = tensor.empty() : tensor<256x1280xf16>
  %fill692 = linalg.fill ins(%cst : f16) outs(%init691 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm693 = linalg.matmul ins(%relu690, %mm685 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill692 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init694 = tensor.empty() : tensor<256x1280xf16>
  %fill695 = linalg.fill ins(%cst : f16) outs(%init694 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm696 = linalg.matmul ins(%mm693, %w_o21 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill695 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty697 = tensor.empty() : tensor<256x1280xf16>
  %add698 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm696, %add676 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty697 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init699 = tensor.empty() : tensor<256x5120xf16>
  %fill700 = linalg.fill ins(%cst : f16) outs(%init699 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm701 = linalg.matmul ins(%add698, %w_ff21_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill700 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty702 = tensor.empty() : tensor<256x5120xf16>
  %relu703 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm701 : tensor<256x5120xf16>)
    outs(%empty702 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init704 = tensor.empty() : tensor<256x1280xf16>
  %fill705 = linalg.fill ins(%cst : f16) outs(%init704 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm706 = linalg.matmul ins(%relu703, %w_ff21_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill705 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty707 = tensor.empty() : tensor<256x1280xf16>
  %add708 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm706, %add698 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty707 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 22 ===
  %init709 = tensor.empty() : tensor<256x1280xf16>
  %fill710 = linalg.fill ins(%cst : f16) outs(%init709 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm711 = linalg.matmul ins(%add708, %w_q22 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill710 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init712 = tensor.empty() : tensor<256x1280xf16>
  %fill713 = linalg.fill ins(%cst : f16) outs(%init712 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm714 = linalg.matmul ins(%add708, %w_k22 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill713 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init715 = tensor.empty() : tensor<256x1280xf16>
  %fill716 = linalg.fill ins(%cst : f16) outs(%init715 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm717 = linalg.matmul ins(%add708, %w_v22 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill716 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init718 = tensor.empty() : tensor<256x256xf16>
  %fill719 = linalg.fill ins(%cst : f16) outs(%init718 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm720 = linalg.matmul ins(%mm711, %w_kt22 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill719 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty721 = tensor.empty() : tensor<256x256xf16>
  %relu722 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm720 : tensor<256x256xf16>)
    outs(%empty721 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init723 = tensor.empty() : tensor<256x1280xf16>
  %fill724 = linalg.fill ins(%cst : f16) outs(%init723 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm725 = linalg.matmul ins(%relu722, %mm717 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill724 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init726 = tensor.empty() : tensor<256x1280xf16>
  %fill727 = linalg.fill ins(%cst : f16) outs(%init726 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm728 = linalg.matmul ins(%mm725, %w_o22 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill727 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty729 = tensor.empty() : tensor<256x1280xf16>
  %add730 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm728, %add708 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty729 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init731 = tensor.empty() : tensor<256x5120xf16>
  %fill732 = linalg.fill ins(%cst : f16) outs(%init731 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm733 = linalg.matmul ins(%add730, %w_ff22_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill732 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty734 = tensor.empty() : tensor<256x5120xf16>
  %relu735 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm733 : tensor<256x5120xf16>)
    outs(%empty734 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init736 = tensor.empty() : tensor<256x1280xf16>
  %fill737 = linalg.fill ins(%cst : f16) outs(%init736 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm738 = linalg.matmul ins(%relu735, %w_ff22_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill737 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty739 = tensor.empty() : tensor<256x1280xf16>
  %add740 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm738, %add730 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty739 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 23 ===
  %init741 = tensor.empty() : tensor<256x1280xf16>
  %fill742 = linalg.fill ins(%cst : f16) outs(%init741 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm743 = linalg.matmul ins(%add740, %w_q23 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill742 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init744 = tensor.empty() : tensor<256x1280xf16>
  %fill745 = linalg.fill ins(%cst : f16) outs(%init744 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm746 = linalg.matmul ins(%add740, %w_k23 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill745 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init747 = tensor.empty() : tensor<256x1280xf16>
  %fill748 = linalg.fill ins(%cst : f16) outs(%init747 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm749 = linalg.matmul ins(%add740, %w_v23 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill748 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init750 = tensor.empty() : tensor<256x256xf16>
  %fill751 = linalg.fill ins(%cst : f16) outs(%init750 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm752 = linalg.matmul ins(%mm743, %w_kt23 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill751 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty753 = tensor.empty() : tensor<256x256xf16>
  %relu754 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm752 : tensor<256x256xf16>)
    outs(%empty753 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init755 = tensor.empty() : tensor<256x1280xf16>
  %fill756 = linalg.fill ins(%cst : f16) outs(%init755 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm757 = linalg.matmul ins(%relu754, %mm749 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill756 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init758 = tensor.empty() : tensor<256x1280xf16>
  %fill759 = linalg.fill ins(%cst : f16) outs(%init758 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm760 = linalg.matmul ins(%mm757, %w_o23 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill759 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty761 = tensor.empty() : tensor<256x1280xf16>
  %add762 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm760, %add740 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty761 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init763 = tensor.empty() : tensor<256x5120xf16>
  %fill764 = linalg.fill ins(%cst : f16) outs(%init763 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm765 = linalg.matmul ins(%add762, %w_ff23_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill764 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty766 = tensor.empty() : tensor<256x5120xf16>
  %relu767 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm765 : tensor<256x5120xf16>)
    outs(%empty766 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init768 = tensor.empty() : tensor<256x1280xf16>
  %fill769 = linalg.fill ins(%cst : f16) outs(%init768 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm770 = linalg.matmul ins(%relu767, %w_ff23_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill769 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty771 = tensor.empty() : tensor<256x1280xf16>
  %add772 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm770, %add762 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty771 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 24 ===
  %init773 = tensor.empty() : tensor<256x1280xf16>
  %fill774 = linalg.fill ins(%cst : f16) outs(%init773 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm775 = linalg.matmul ins(%add772, %w_q24 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill774 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init776 = tensor.empty() : tensor<256x1280xf16>
  %fill777 = linalg.fill ins(%cst : f16) outs(%init776 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm778 = linalg.matmul ins(%add772, %w_k24 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill777 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init779 = tensor.empty() : tensor<256x1280xf16>
  %fill780 = linalg.fill ins(%cst : f16) outs(%init779 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm781 = linalg.matmul ins(%add772, %w_v24 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill780 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init782 = tensor.empty() : tensor<256x256xf16>
  %fill783 = linalg.fill ins(%cst : f16) outs(%init782 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm784 = linalg.matmul ins(%mm775, %w_kt24 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill783 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty785 = tensor.empty() : tensor<256x256xf16>
  %relu786 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm784 : tensor<256x256xf16>)
    outs(%empty785 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init787 = tensor.empty() : tensor<256x1280xf16>
  %fill788 = linalg.fill ins(%cst : f16) outs(%init787 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm789 = linalg.matmul ins(%relu786, %mm781 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill788 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init790 = tensor.empty() : tensor<256x1280xf16>
  %fill791 = linalg.fill ins(%cst : f16) outs(%init790 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm792 = linalg.matmul ins(%mm789, %w_o24 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill791 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty793 = tensor.empty() : tensor<256x1280xf16>
  %add794 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm792, %add772 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty793 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init795 = tensor.empty() : tensor<256x5120xf16>
  %fill796 = linalg.fill ins(%cst : f16) outs(%init795 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm797 = linalg.matmul ins(%add794, %w_ff24_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill796 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty798 = tensor.empty() : tensor<256x5120xf16>
  %relu799 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm797 : tensor<256x5120xf16>)
    outs(%empty798 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init800 = tensor.empty() : tensor<256x1280xf16>
  %fill801 = linalg.fill ins(%cst : f16) outs(%init800 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm802 = linalg.matmul ins(%relu799, %w_ff24_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill801 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty803 = tensor.empty() : tensor<256x1280xf16>
  %add804 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm802, %add794 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty803 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 25 ===
  %init805 = tensor.empty() : tensor<256x1280xf16>
  %fill806 = linalg.fill ins(%cst : f16) outs(%init805 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm807 = linalg.matmul ins(%add804, %w_q25 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill806 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init808 = tensor.empty() : tensor<256x1280xf16>
  %fill809 = linalg.fill ins(%cst : f16) outs(%init808 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm810 = linalg.matmul ins(%add804, %w_k25 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill809 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init811 = tensor.empty() : tensor<256x1280xf16>
  %fill812 = linalg.fill ins(%cst : f16) outs(%init811 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm813 = linalg.matmul ins(%add804, %w_v25 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill812 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init814 = tensor.empty() : tensor<256x256xf16>
  %fill815 = linalg.fill ins(%cst : f16) outs(%init814 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm816 = linalg.matmul ins(%mm807, %w_kt25 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill815 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty817 = tensor.empty() : tensor<256x256xf16>
  %relu818 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm816 : tensor<256x256xf16>)
    outs(%empty817 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init819 = tensor.empty() : tensor<256x1280xf16>
  %fill820 = linalg.fill ins(%cst : f16) outs(%init819 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm821 = linalg.matmul ins(%relu818, %mm813 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill820 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init822 = tensor.empty() : tensor<256x1280xf16>
  %fill823 = linalg.fill ins(%cst : f16) outs(%init822 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm824 = linalg.matmul ins(%mm821, %w_o25 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill823 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty825 = tensor.empty() : tensor<256x1280xf16>
  %add826 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm824, %add804 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty825 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init827 = tensor.empty() : tensor<256x5120xf16>
  %fill828 = linalg.fill ins(%cst : f16) outs(%init827 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm829 = linalg.matmul ins(%add826, %w_ff25_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill828 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty830 = tensor.empty() : tensor<256x5120xf16>
  %relu831 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm829 : tensor<256x5120xf16>)
    outs(%empty830 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init832 = tensor.empty() : tensor<256x1280xf16>
  %fill833 = linalg.fill ins(%cst : f16) outs(%init832 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm834 = linalg.matmul ins(%relu831, %w_ff25_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill833 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty835 = tensor.empty() : tensor<256x1280xf16>
  %add836 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm834, %add826 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty835 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 26 ===
  %init837 = tensor.empty() : tensor<256x1280xf16>
  %fill838 = linalg.fill ins(%cst : f16) outs(%init837 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm839 = linalg.matmul ins(%add836, %w_q26 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill838 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init840 = tensor.empty() : tensor<256x1280xf16>
  %fill841 = linalg.fill ins(%cst : f16) outs(%init840 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm842 = linalg.matmul ins(%add836, %w_k26 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill841 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init843 = tensor.empty() : tensor<256x1280xf16>
  %fill844 = linalg.fill ins(%cst : f16) outs(%init843 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm845 = linalg.matmul ins(%add836, %w_v26 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill844 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init846 = tensor.empty() : tensor<256x256xf16>
  %fill847 = linalg.fill ins(%cst : f16) outs(%init846 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm848 = linalg.matmul ins(%mm839, %w_kt26 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill847 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty849 = tensor.empty() : tensor<256x256xf16>
  %relu850 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm848 : tensor<256x256xf16>)
    outs(%empty849 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init851 = tensor.empty() : tensor<256x1280xf16>
  %fill852 = linalg.fill ins(%cst : f16) outs(%init851 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm853 = linalg.matmul ins(%relu850, %mm845 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill852 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init854 = tensor.empty() : tensor<256x1280xf16>
  %fill855 = linalg.fill ins(%cst : f16) outs(%init854 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm856 = linalg.matmul ins(%mm853, %w_o26 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill855 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty857 = tensor.empty() : tensor<256x1280xf16>
  %add858 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm856, %add836 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty857 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init859 = tensor.empty() : tensor<256x5120xf16>
  %fill860 = linalg.fill ins(%cst : f16) outs(%init859 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm861 = linalg.matmul ins(%add858, %w_ff26_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill860 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty862 = tensor.empty() : tensor<256x5120xf16>
  %relu863 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm861 : tensor<256x5120xf16>)
    outs(%empty862 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init864 = tensor.empty() : tensor<256x1280xf16>
  %fill865 = linalg.fill ins(%cst : f16) outs(%init864 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm866 = linalg.matmul ins(%relu863, %w_ff26_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill865 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty867 = tensor.empty() : tensor<256x1280xf16>
  %add868 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm866, %add858 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty867 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 27 ===
  %init869 = tensor.empty() : tensor<256x1280xf16>
  %fill870 = linalg.fill ins(%cst : f16) outs(%init869 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm871 = linalg.matmul ins(%add868, %w_q27 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill870 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init872 = tensor.empty() : tensor<256x1280xf16>
  %fill873 = linalg.fill ins(%cst : f16) outs(%init872 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm874 = linalg.matmul ins(%add868, %w_k27 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill873 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init875 = tensor.empty() : tensor<256x1280xf16>
  %fill876 = linalg.fill ins(%cst : f16) outs(%init875 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm877 = linalg.matmul ins(%add868, %w_v27 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill876 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init878 = tensor.empty() : tensor<256x256xf16>
  %fill879 = linalg.fill ins(%cst : f16) outs(%init878 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm880 = linalg.matmul ins(%mm871, %w_kt27 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill879 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty881 = tensor.empty() : tensor<256x256xf16>
  %relu882 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm880 : tensor<256x256xf16>)
    outs(%empty881 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init883 = tensor.empty() : tensor<256x1280xf16>
  %fill884 = linalg.fill ins(%cst : f16) outs(%init883 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm885 = linalg.matmul ins(%relu882, %mm877 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill884 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init886 = tensor.empty() : tensor<256x1280xf16>
  %fill887 = linalg.fill ins(%cst : f16) outs(%init886 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm888 = linalg.matmul ins(%mm885, %w_o27 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill887 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty889 = tensor.empty() : tensor<256x1280xf16>
  %add890 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm888, %add868 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty889 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init891 = tensor.empty() : tensor<256x5120xf16>
  %fill892 = linalg.fill ins(%cst : f16) outs(%init891 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm893 = linalg.matmul ins(%add890, %w_ff27_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill892 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty894 = tensor.empty() : tensor<256x5120xf16>
  %relu895 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm893 : tensor<256x5120xf16>)
    outs(%empty894 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init896 = tensor.empty() : tensor<256x1280xf16>
  %fill897 = linalg.fill ins(%cst : f16) outs(%init896 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm898 = linalg.matmul ins(%relu895, %w_ff27_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill897 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty899 = tensor.empty() : tensor<256x1280xf16>
  %add900 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm898, %add890 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty899 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 28 ===
  %init901 = tensor.empty() : tensor<256x1280xf16>
  %fill902 = linalg.fill ins(%cst : f16) outs(%init901 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm903 = linalg.matmul ins(%add900, %w_q28 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill902 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init904 = tensor.empty() : tensor<256x1280xf16>
  %fill905 = linalg.fill ins(%cst : f16) outs(%init904 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm906 = linalg.matmul ins(%add900, %w_k28 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill905 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init907 = tensor.empty() : tensor<256x1280xf16>
  %fill908 = linalg.fill ins(%cst : f16) outs(%init907 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm909 = linalg.matmul ins(%add900, %w_v28 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill908 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init910 = tensor.empty() : tensor<256x256xf16>
  %fill911 = linalg.fill ins(%cst : f16) outs(%init910 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm912 = linalg.matmul ins(%mm903, %w_kt28 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill911 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty913 = tensor.empty() : tensor<256x256xf16>
  %relu914 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm912 : tensor<256x256xf16>)
    outs(%empty913 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init915 = tensor.empty() : tensor<256x1280xf16>
  %fill916 = linalg.fill ins(%cst : f16) outs(%init915 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm917 = linalg.matmul ins(%relu914, %mm909 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill916 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init918 = tensor.empty() : tensor<256x1280xf16>
  %fill919 = linalg.fill ins(%cst : f16) outs(%init918 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm920 = linalg.matmul ins(%mm917, %w_o28 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill919 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty921 = tensor.empty() : tensor<256x1280xf16>
  %add922 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm920, %add900 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty921 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init923 = tensor.empty() : tensor<256x5120xf16>
  %fill924 = linalg.fill ins(%cst : f16) outs(%init923 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm925 = linalg.matmul ins(%add922, %w_ff28_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill924 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty926 = tensor.empty() : tensor<256x5120xf16>
  %relu927 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm925 : tensor<256x5120xf16>)
    outs(%empty926 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init928 = tensor.empty() : tensor<256x1280xf16>
  %fill929 = linalg.fill ins(%cst : f16) outs(%init928 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm930 = linalg.matmul ins(%relu927, %w_ff28_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill929 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty931 = tensor.empty() : tensor<256x1280xf16>
  %add932 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm930, %add922 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty931 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 29 ===
  %init933 = tensor.empty() : tensor<256x1280xf16>
  %fill934 = linalg.fill ins(%cst : f16) outs(%init933 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm935 = linalg.matmul ins(%add932, %w_q29 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill934 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init936 = tensor.empty() : tensor<256x1280xf16>
  %fill937 = linalg.fill ins(%cst : f16) outs(%init936 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm938 = linalg.matmul ins(%add932, %w_k29 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill937 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init939 = tensor.empty() : tensor<256x1280xf16>
  %fill940 = linalg.fill ins(%cst : f16) outs(%init939 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm941 = linalg.matmul ins(%add932, %w_v29 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill940 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init942 = tensor.empty() : tensor<256x256xf16>
  %fill943 = linalg.fill ins(%cst : f16) outs(%init942 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm944 = linalg.matmul ins(%mm935, %w_kt29 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill943 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty945 = tensor.empty() : tensor<256x256xf16>
  %relu946 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm944 : tensor<256x256xf16>)
    outs(%empty945 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init947 = tensor.empty() : tensor<256x1280xf16>
  %fill948 = linalg.fill ins(%cst : f16) outs(%init947 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm949 = linalg.matmul ins(%relu946, %mm941 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill948 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init950 = tensor.empty() : tensor<256x1280xf16>
  %fill951 = linalg.fill ins(%cst : f16) outs(%init950 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm952 = linalg.matmul ins(%mm949, %w_o29 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill951 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty953 = tensor.empty() : tensor<256x1280xf16>
  %add954 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm952, %add932 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty953 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init955 = tensor.empty() : tensor<256x5120xf16>
  %fill956 = linalg.fill ins(%cst : f16) outs(%init955 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm957 = linalg.matmul ins(%add954, %w_ff29_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill956 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty958 = tensor.empty() : tensor<256x5120xf16>
  %relu959 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm957 : tensor<256x5120xf16>)
    outs(%empty958 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init960 = tensor.empty() : tensor<256x1280xf16>
  %fill961 = linalg.fill ins(%cst : f16) outs(%init960 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm962 = linalg.matmul ins(%relu959, %w_ff29_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill961 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty963 = tensor.empty() : tensor<256x1280xf16>
  %add964 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm962, %add954 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty963 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 30 ===
  %init965 = tensor.empty() : tensor<256x1280xf16>
  %fill966 = linalg.fill ins(%cst : f16) outs(%init965 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm967 = linalg.matmul ins(%add964, %w_q30 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill966 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init968 = tensor.empty() : tensor<256x1280xf16>
  %fill969 = linalg.fill ins(%cst : f16) outs(%init968 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm970 = linalg.matmul ins(%add964, %w_k30 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill969 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init971 = tensor.empty() : tensor<256x1280xf16>
  %fill972 = linalg.fill ins(%cst : f16) outs(%init971 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm973 = linalg.matmul ins(%add964, %w_v30 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill972 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init974 = tensor.empty() : tensor<256x256xf16>
  %fill975 = linalg.fill ins(%cst : f16) outs(%init974 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm976 = linalg.matmul ins(%mm967, %w_kt30 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill975 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty977 = tensor.empty() : tensor<256x256xf16>
  %relu978 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm976 : tensor<256x256xf16>)
    outs(%empty977 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init979 = tensor.empty() : tensor<256x1280xf16>
  %fill980 = linalg.fill ins(%cst : f16) outs(%init979 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm981 = linalg.matmul ins(%relu978, %mm973 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill980 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init982 = tensor.empty() : tensor<256x1280xf16>
  %fill983 = linalg.fill ins(%cst : f16) outs(%init982 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm984 = linalg.matmul ins(%mm981, %w_o30 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill983 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty985 = tensor.empty() : tensor<256x1280xf16>
  %add986 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm984, %add964 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty985 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init987 = tensor.empty() : tensor<256x5120xf16>
  %fill988 = linalg.fill ins(%cst : f16) outs(%init987 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm989 = linalg.matmul ins(%add986, %w_ff30_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill988 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty990 = tensor.empty() : tensor<256x5120xf16>
  %relu991 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm989 : tensor<256x5120xf16>)
    outs(%empty990 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init992 = tensor.empty() : tensor<256x1280xf16>
  %fill993 = linalg.fill ins(%cst : f16) outs(%init992 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm994 = linalg.matmul ins(%relu991, %w_ff30_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill993 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty995 = tensor.empty() : tensor<256x1280xf16>
  %add996 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm994, %add986 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty995 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // === Transformer Layer 31 ===
  %init997 = tensor.empty() : tensor<256x1280xf16>
  %fill998 = linalg.fill ins(%cst : f16) outs(%init997 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm999 = linalg.matmul ins(%add996, %w_q31 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill998 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init1000 = tensor.empty() : tensor<256x1280xf16>
  %fill1001 = linalg.fill ins(%cst : f16) outs(%init1000 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm1002 = linalg.matmul ins(%add996, %w_k31 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1001 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init1003 = tensor.empty() : tensor<256x1280xf16>
  %fill1004 = linalg.fill ins(%cst : f16) outs(%init1003 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm1005 = linalg.matmul ins(%add996, %w_v31 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1004 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init1006 = tensor.empty() : tensor<256x256xf16>
  %fill1007 = linalg.fill ins(%cst : f16) outs(%init1006 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %mm1008 = linalg.matmul ins(%mm999, %w_kt31 : tensor<256x1280xf16>, tensor<1280x256xf16>)
                          outs(%fill1007 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %empty1009 = tensor.empty() : tensor<256x256xf16>
  %relu1010 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1008 : tensor<256x256xf16>)
    outs(%empty1009 : tensor<256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x256xf16>
  %init1011 = tensor.empty() : tensor<256x1280xf16>
  %fill1012 = linalg.fill ins(%cst : f16) outs(%init1011 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm1013 = linalg.matmul ins(%relu1010, %mm1005 : tensor<256x256xf16>, tensor<256x1280xf16>)
                          outs(%fill1012 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %init1014 = tensor.empty() : tensor<256x1280xf16>
  %fill1015 = linalg.fill ins(%cst : f16) outs(%init1014 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm1016 = linalg.matmul ins(%mm1013, %w_o31 : tensor<256x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1015 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty1017 = tensor.empty() : tensor<256x1280xf16>
  %add1018 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1016, %add996 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty1017 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>
  %init1019 = tensor.empty() : tensor<256x5120xf16>
  %fill1020 = linalg.fill ins(%cst : f16) outs(%init1019 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %mm1021 = linalg.matmul ins(%add1018, %w_ff31_up : tensor<256x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill1020 : tensor<256x5120xf16>) -> tensor<256x5120xf16>
  %empty1022 = tensor.empty() : tensor<256x5120xf16>
  %relu1023 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1021 : tensor<256x5120xf16>)
    outs(%empty1022 : tensor<256x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<256x5120xf16>
  %init1024 = tensor.empty() : tensor<256x1280xf16>
  %fill1025 = linalg.fill ins(%cst : f16) outs(%init1024 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %mm1026 = linalg.matmul ins(%relu1023, %w_ff31_down : tensor<256x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill1025 : tensor<256x1280xf16>) -> tensor<256x1280xf16>
  %empty1027 = tensor.empty() : tensor<256x1280xf16>
  %add1028 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1026, %add1018 : tensor<256x1280xf16>, tensor<256x1280xf16>)
    outs(%empty1027 : tensor<256x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<256x1280xf16>

  // Classification head: matmul 1280->1000
  %init1029 = tensor.empty() : tensor<256x1000xf16>
  %fill1030 = linalg.fill ins(%cst : f16) outs(%init1029 : tensor<256x1000xf16>) -> tensor<256x1000xf16>
  %mm1031 = linalg.matmul ins(%add1028, %w_head : tensor<256x1280xf16>, tensor<1280x1000xf16>)
                          outs(%fill1030 : tensor<256x1000xf16>) -> tensor<256x1000xf16>
  return %mm1031 : tensor<256x1000xf16>
}
