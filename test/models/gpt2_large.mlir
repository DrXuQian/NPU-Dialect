func.func @gpt2_large(
    %input: tensor<1024x1280xf16>,
    %w_q0: tensor<1280x1280xf16>,
    %w_k0: tensor<1280x1280xf16>,
    %w_v0: tensor<1280x1280xf16>,
    %w_kt0: tensor<1280x1024xf16>,
    %w_o0: tensor<1280x1280xf16>,
    %w_ff0_up: tensor<1280x5120xf16>,
    %w_ff0_down: tensor<5120x1280xf16>,
    %w_q1: tensor<1280x1280xf16>,
    %w_k1: tensor<1280x1280xf16>,
    %w_v1: tensor<1280x1280xf16>,
    %w_kt1: tensor<1280x1024xf16>,
    %w_o1: tensor<1280x1280xf16>,
    %w_ff1_up: tensor<1280x5120xf16>,
    %w_ff1_down: tensor<5120x1280xf16>,
    %w_q2: tensor<1280x1280xf16>,
    %w_k2: tensor<1280x1280xf16>,
    %w_v2: tensor<1280x1280xf16>,
    %w_kt2: tensor<1280x1024xf16>,
    %w_o2: tensor<1280x1280xf16>,
    %w_ff2_up: tensor<1280x5120xf16>,
    %w_ff2_down: tensor<5120x1280xf16>,
    %w_q3: tensor<1280x1280xf16>,
    %w_k3: tensor<1280x1280xf16>,
    %w_v3: tensor<1280x1280xf16>,
    %w_kt3: tensor<1280x1024xf16>,
    %w_o3: tensor<1280x1280xf16>,
    %w_ff3_up: tensor<1280x5120xf16>,
    %w_ff3_down: tensor<5120x1280xf16>,
    %w_q4: tensor<1280x1280xf16>,
    %w_k4: tensor<1280x1280xf16>,
    %w_v4: tensor<1280x1280xf16>,
    %w_kt4: tensor<1280x1024xf16>,
    %w_o4: tensor<1280x1280xf16>,
    %w_ff4_up: tensor<1280x5120xf16>,
    %w_ff4_down: tensor<5120x1280xf16>,
    %w_q5: tensor<1280x1280xf16>,
    %w_k5: tensor<1280x1280xf16>,
    %w_v5: tensor<1280x1280xf16>,
    %w_kt5: tensor<1280x1024xf16>,
    %w_o5: tensor<1280x1280xf16>,
    %w_ff5_up: tensor<1280x5120xf16>,
    %w_ff5_down: tensor<5120x1280xf16>,
    %w_q6: tensor<1280x1280xf16>,
    %w_k6: tensor<1280x1280xf16>,
    %w_v6: tensor<1280x1280xf16>,
    %w_kt6: tensor<1280x1024xf16>,
    %w_o6: tensor<1280x1280xf16>,
    %w_ff6_up: tensor<1280x5120xf16>,
    %w_ff6_down: tensor<5120x1280xf16>,
    %w_q7: tensor<1280x1280xf16>,
    %w_k7: tensor<1280x1280xf16>,
    %w_v7: tensor<1280x1280xf16>,
    %w_kt7: tensor<1280x1024xf16>,
    %w_o7: tensor<1280x1280xf16>,
    %w_ff7_up: tensor<1280x5120xf16>,
    %w_ff7_down: tensor<5120x1280xf16>,
    %w_q8: tensor<1280x1280xf16>,
    %w_k8: tensor<1280x1280xf16>,
    %w_v8: tensor<1280x1280xf16>,
    %w_kt8: tensor<1280x1024xf16>,
    %w_o8: tensor<1280x1280xf16>,
    %w_ff8_up: tensor<1280x5120xf16>,
    %w_ff8_down: tensor<5120x1280xf16>,
    %w_q9: tensor<1280x1280xf16>,
    %w_k9: tensor<1280x1280xf16>,
    %w_v9: tensor<1280x1280xf16>,
    %w_kt9: tensor<1280x1024xf16>,
    %w_o9: tensor<1280x1280xf16>,
    %w_ff9_up: tensor<1280x5120xf16>,
    %w_ff9_down: tensor<5120x1280xf16>,
    %w_q10: tensor<1280x1280xf16>,
    %w_k10: tensor<1280x1280xf16>,
    %w_v10: tensor<1280x1280xf16>,
    %w_kt10: tensor<1280x1024xf16>,
    %w_o10: tensor<1280x1280xf16>,
    %w_ff10_up: tensor<1280x5120xf16>,
    %w_ff10_down: tensor<5120x1280xf16>,
    %w_q11: tensor<1280x1280xf16>,
    %w_k11: tensor<1280x1280xf16>,
    %w_v11: tensor<1280x1280xf16>,
    %w_kt11: tensor<1280x1024xf16>,
    %w_o11: tensor<1280x1280xf16>,
    %w_ff11_up: tensor<1280x5120xf16>,
    %w_ff11_down: tensor<5120x1280xf16>,
    %w_q12: tensor<1280x1280xf16>,
    %w_k12: tensor<1280x1280xf16>,
    %w_v12: tensor<1280x1280xf16>,
    %w_kt12: tensor<1280x1024xf16>,
    %w_o12: tensor<1280x1280xf16>,
    %w_ff12_up: tensor<1280x5120xf16>,
    %w_ff12_down: tensor<5120x1280xf16>,
    %w_q13: tensor<1280x1280xf16>,
    %w_k13: tensor<1280x1280xf16>,
    %w_v13: tensor<1280x1280xf16>,
    %w_kt13: tensor<1280x1024xf16>,
    %w_o13: tensor<1280x1280xf16>,
    %w_ff13_up: tensor<1280x5120xf16>,
    %w_ff13_down: tensor<5120x1280xf16>,
    %w_q14: tensor<1280x1280xf16>,
    %w_k14: tensor<1280x1280xf16>,
    %w_v14: tensor<1280x1280xf16>,
    %w_kt14: tensor<1280x1024xf16>,
    %w_o14: tensor<1280x1280xf16>,
    %w_ff14_up: tensor<1280x5120xf16>,
    %w_ff14_down: tensor<5120x1280xf16>,
    %w_q15: tensor<1280x1280xf16>,
    %w_k15: tensor<1280x1280xf16>,
    %w_v15: tensor<1280x1280xf16>,
    %w_kt15: tensor<1280x1024xf16>,
    %w_o15: tensor<1280x1280xf16>,
    %w_ff15_up: tensor<1280x5120xf16>,
    %w_ff15_down: tensor<5120x1280xf16>,
    %w_q16: tensor<1280x1280xf16>,
    %w_k16: tensor<1280x1280xf16>,
    %w_v16: tensor<1280x1280xf16>,
    %w_kt16: tensor<1280x1024xf16>,
    %w_o16: tensor<1280x1280xf16>,
    %w_ff16_up: tensor<1280x5120xf16>,
    %w_ff16_down: tensor<5120x1280xf16>,
    %w_q17: tensor<1280x1280xf16>,
    %w_k17: tensor<1280x1280xf16>,
    %w_v17: tensor<1280x1280xf16>,
    %w_kt17: tensor<1280x1024xf16>,
    %w_o17: tensor<1280x1280xf16>,
    %w_ff17_up: tensor<1280x5120xf16>,
    %w_ff17_down: tensor<5120x1280xf16>,
    %w_q18: tensor<1280x1280xf16>,
    %w_k18: tensor<1280x1280xf16>,
    %w_v18: tensor<1280x1280xf16>,
    %w_kt18: tensor<1280x1024xf16>,
    %w_o18: tensor<1280x1280xf16>,
    %w_ff18_up: tensor<1280x5120xf16>,
    %w_ff18_down: tensor<5120x1280xf16>,
    %w_q19: tensor<1280x1280xf16>,
    %w_k19: tensor<1280x1280xf16>,
    %w_v19: tensor<1280x1280xf16>,
    %w_kt19: tensor<1280x1024xf16>,
    %w_o19: tensor<1280x1280xf16>,
    %w_ff19_up: tensor<1280x5120xf16>,
    %w_ff19_down: tensor<5120x1280xf16>,
    %w_q20: tensor<1280x1280xf16>,
    %w_k20: tensor<1280x1280xf16>,
    %w_v20: tensor<1280x1280xf16>,
    %w_kt20: tensor<1280x1024xf16>,
    %w_o20: tensor<1280x1280xf16>,
    %w_ff20_up: tensor<1280x5120xf16>,
    %w_ff20_down: tensor<5120x1280xf16>,
    %w_q21: tensor<1280x1280xf16>,
    %w_k21: tensor<1280x1280xf16>,
    %w_v21: tensor<1280x1280xf16>,
    %w_kt21: tensor<1280x1024xf16>,
    %w_o21: tensor<1280x1280xf16>,
    %w_ff21_up: tensor<1280x5120xf16>,
    %w_ff21_down: tensor<5120x1280xf16>,
    %w_q22: tensor<1280x1280xf16>,
    %w_k22: tensor<1280x1280xf16>,
    %w_v22: tensor<1280x1280xf16>,
    %w_kt22: tensor<1280x1024xf16>,
    %w_o22: tensor<1280x1280xf16>,
    %w_ff22_up: tensor<1280x5120xf16>,
    %w_ff22_down: tensor<5120x1280xf16>,
    %w_q23: tensor<1280x1280xf16>,
    %w_k23: tensor<1280x1280xf16>,
    %w_v23: tensor<1280x1280xf16>,
    %w_kt23: tensor<1280x1024xf16>,
    %w_o23: tensor<1280x1280xf16>,
    %w_ff23_up: tensor<1280x5120xf16>,
    %w_ff23_down: tensor<5120x1280xf16>,
    %w_q24: tensor<1280x1280xf16>,
    %w_k24: tensor<1280x1280xf16>,
    %w_v24: tensor<1280x1280xf16>,
    %w_kt24: tensor<1280x1024xf16>,
    %w_o24: tensor<1280x1280xf16>,
    %w_ff24_up: tensor<1280x5120xf16>,
    %w_ff24_down: tensor<5120x1280xf16>,
    %w_q25: tensor<1280x1280xf16>,
    %w_k25: tensor<1280x1280xf16>,
    %w_v25: tensor<1280x1280xf16>,
    %w_kt25: tensor<1280x1024xf16>,
    %w_o25: tensor<1280x1280xf16>,
    %w_ff25_up: tensor<1280x5120xf16>,
    %w_ff25_down: tensor<5120x1280xf16>,
    %w_q26: tensor<1280x1280xf16>,
    %w_k26: tensor<1280x1280xf16>,
    %w_v26: tensor<1280x1280xf16>,
    %w_kt26: tensor<1280x1024xf16>,
    %w_o26: tensor<1280x1280xf16>,
    %w_ff26_up: tensor<1280x5120xf16>,
    %w_ff26_down: tensor<5120x1280xf16>,
    %w_q27: tensor<1280x1280xf16>,
    %w_k27: tensor<1280x1280xf16>,
    %w_v27: tensor<1280x1280xf16>,
    %w_kt27: tensor<1280x1024xf16>,
    %w_o27: tensor<1280x1280xf16>,
    %w_ff27_up: tensor<1280x5120xf16>,
    %w_ff27_down: tensor<5120x1280xf16>,
    %w_q28: tensor<1280x1280xf16>,
    %w_k28: tensor<1280x1280xf16>,
    %w_v28: tensor<1280x1280xf16>,
    %w_kt28: tensor<1280x1024xf16>,
    %w_o28: tensor<1280x1280xf16>,
    %w_ff28_up: tensor<1280x5120xf16>,
    %w_ff28_down: tensor<5120x1280xf16>,
    %w_q29: tensor<1280x1280xf16>,
    %w_k29: tensor<1280x1280xf16>,
    %w_v29: tensor<1280x1280xf16>,
    %w_kt29: tensor<1280x1024xf16>,
    %w_o29: tensor<1280x1280xf16>,
    %w_ff29_up: tensor<1280x5120xf16>,
    %w_ff29_down: tensor<5120x1280xf16>,
    %w_q30: tensor<1280x1280xf16>,
    %w_k30: tensor<1280x1280xf16>,
    %w_v30: tensor<1280x1280xf16>,
    %w_kt30: tensor<1280x1024xf16>,
    %w_o30: tensor<1280x1280xf16>,
    %w_ff30_up: tensor<1280x5120xf16>,
    %w_ff30_down: tensor<5120x1280xf16>,
    %w_q31: tensor<1280x1280xf16>,
    %w_k31: tensor<1280x1280xf16>,
    %w_v31: tensor<1280x1280xf16>,
    %w_kt31: tensor<1280x1024xf16>,
    %w_o31: tensor<1280x1280xf16>,
    %w_ff31_up: tensor<1280x5120xf16>,
    %w_ff31_down: tensor<5120x1280xf16>,
    %w_q32: tensor<1280x1280xf16>,
    %w_k32: tensor<1280x1280xf16>,
    %w_v32: tensor<1280x1280xf16>,
    %w_kt32: tensor<1280x1024xf16>,
    %w_o32: tensor<1280x1280xf16>,
    %w_ff32_up: tensor<1280x5120xf16>,
    %w_ff32_down: tensor<5120x1280xf16>,
    %w_q33: tensor<1280x1280xf16>,
    %w_k33: tensor<1280x1280xf16>,
    %w_v33: tensor<1280x1280xf16>,
    %w_kt33: tensor<1280x1024xf16>,
    %w_o33: tensor<1280x1280xf16>,
    %w_ff33_up: tensor<1280x5120xf16>,
    %w_ff33_down: tensor<5120x1280xf16>,
    %w_q34: tensor<1280x1280xf16>,
    %w_k34: tensor<1280x1280xf16>,
    %w_v34: tensor<1280x1280xf16>,
    %w_kt34: tensor<1280x1024xf16>,
    %w_o34: tensor<1280x1280xf16>,
    %w_ff34_up: tensor<1280x5120xf16>,
    %w_ff34_down: tensor<5120x1280xf16>,
    %w_q35: tensor<1280x1280xf16>,
    %w_k35: tensor<1280x1280xf16>,
    %w_v35: tensor<1280x1280xf16>,
    %w_kt35: tensor<1280x1024xf16>,
    %w_o35: tensor<1280x1280xf16>,
    %w_ff35_up: tensor<1280x5120xf16>,
    %w_ff35_down: tensor<5120x1280xf16>) -> tensor<1024x1280xf16> {
  %cst = arith.constant 0.0 : f16

  // === Transformer Layer 0 ===

  // Q projection
  %init0 = tensor.empty() : tensor<1024x1280xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init0 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm2 = linalg.matmul ins(%input, %w_q0 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init3 = tensor.empty() : tensor<1024x1280xf16>
  %fill4 = linalg.fill ins(%cst : f16) outs(%init3 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm5 = linalg.matmul ins(%input, %w_k0 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill4 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init6 = tensor.empty() : tensor<1024x1280xf16>
  %fill7 = linalg.fill ins(%cst : f16) outs(%init6 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm8 = linalg.matmul ins(%input, %w_v0 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill7 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init9 = tensor.empty() : tensor<1024x1024xf16>
  %fill10 = linalg.fill ins(%cst : f16) outs(%init9 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm11 = linalg.matmul ins(%mm2, %w_kt0 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill10 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty12 = tensor.empty() : tensor<1024x1024xf16>
  %relu13 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm11 : tensor<1024x1024xf16>)
    outs(%empty12 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init14 = tensor.empty() : tensor<1024x1280xf16>
  %fill15 = linalg.fill ins(%cst : f16) outs(%init14 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm16 = linalg.matmul ins(%relu13, %mm8 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill15 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init17 = tensor.empty() : tensor<1024x1280xf16>
  %fill18 = linalg.fill ins(%cst : f16) outs(%init17 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm19 = linalg.matmul ins(%mm16, %w_o0 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill18 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty20 = tensor.empty() : tensor<1024x1280xf16>
  %add21 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm19, %input : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty20 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init22 = tensor.empty() : tensor<1024x5120xf16>
  %fill23 = linalg.fill ins(%cst : f16) outs(%init22 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm24 = linalg.matmul ins(%add21, %w_ff0_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill23 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty25 = tensor.empty() : tensor<1024x5120xf16>
  %relu26 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm24 : tensor<1024x5120xf16>)
    outs(%empty25 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init27 = tensor.empty() : tensor<1024x1280xf16>
  %fill28 = linalg.fill ins(%cst : f16) outs(%init27 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm29 = linalg.matmul ins(%relu26, %w_ff0_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill28 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty30 = tensor.empty() : tensor<1024x1280xf16>
  %add31 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm29, %add21 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty30 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 1 ===

  // Q projection
  %init32 = tensor.empty() : tensor<1024x1280xf16>
  %fill33 = linalg.fill ins(%cst : f16) outs(%init32 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm34 = linalg.matmul ins(%add31, %w_q1 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill33 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init35 = tensor.empty() : tensor<1024x1280xf16>
  %fill36 = linalg.fill ins(%cst : f16) outs(%init35 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm37 = linalg.matmul ins(%add31, %w_k1 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill36 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init38 = tensor.empty() : tensor<1024x1280xf16>
  %fill39 = linalg.fill ins(%cst : f16) outs(%init38 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm40 = linalg.matmul ins(%add31, %w_v1 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill39 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init41 = tensor.empty() : tensor<1024x1024xf16>
  %fill42 = linalg.fill ins(%cst : f16) outs(%init41 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm43 = linalg.matmul ins(%mm34, %w_kt1 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill42 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty44 = tensor.empty() : tensor<1024x1024xf16>
  %relu45 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm43 : tensor<1024x1024xf16>)
    outs(%empty44 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init46 = tensor.empty() : tensor<1024x1280xf16>
  %fill47 = linalg.fill ins(%cst : f16) outs(%init46 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm48 = linalg.matmul ins(%relu45, %mm40 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill47 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init49 = tensor.empty() : tensor<1024x1280xf16>
  %fill50 = linalg.fill ins(%cst : f16) outs(%init49 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm51 = linalg.matmul ins(%mm48, %w_o1 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill50 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty52 = tensor.empty() : tensor<1024x1280xf16>
  %add53 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm51, %add31 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty52 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init54 = tensor.empty() : tensor<1024x5120xf16>
  %fill55 = linalg.fill ins(%cst : f16) outs(%init54 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm56 = linalg.matmul ins(%add53, %w_ff1_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill55 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty57 = tensor.empty() : tensor<1024x5120xf16>
  %relu58 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm56 : tensor<1024x5120xf16>)
    outs(%empty57 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init59 = tensor.empty() : tensor<1024x1280xf16>
  %fill60 = linalg.fill ins(%cst : f16) outs(%init59 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm61 = linalg.matmul ins(%relu58, %w_ff1_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill60 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty62 = tensor.empty() : tensor<1024x1280xf16>
  %add63 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm61, %add53 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty62 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 2 ===

  // Q projection
  %init64 = tensor.empty() : tensor<1024x1280xf16>
  %fill65 = linalg.fill ins(%cst : f16) outs(%init64 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm66 = linalg.matmul ins(%add63, %w_q2 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill65 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init67 = tensor.empty() : tensor<1024x1280xf16>
  %fill68 = linalg.fill ins(%cst : f16) outs(%init67 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm69 = linalg.matmul ins(%add63, %w_k2 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill68 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init70 = tensor.empty() : tensor<1024x1280xf16>
  %fill71 = linalg.fill ins(%cst : f16) outs(%init70 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm72 = linalg.matmul ins(%add63, %w_v2 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill71 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init73 = tensor.empty() : tensor<1024x1024xf16>
  %fill74 = linalg.fill ins(%cst : f16) outs(%init73 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm75 = linalg.matmul ins(%mm66, %w_kt2 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill74 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty76 = tensor.empty() : tensor<1024x1024xf16>
  %relu77 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm75 : tensor<1024x1024xf16>)
    outs(%empty76 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init78 = tensor.empty() : tensor<1024x1280xf16>
  %fill79 = linalg.fill ins(%cst : f16) outs(%init78 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm80 = linalg.matmul ins(%relu77, %mm72 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill79 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init81 = tensor.empty() : tensor<1024x1280xf16>
  %fill82 = linalg.fill ins(%cst : f16) outs(%init81 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm83 = linalg.matmul ins(%mm80, %w_o2 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill82 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty84 = tensor.empty() : tensor<1024x1280xf16>
  %add85 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm83, %add63 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty84 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init86 = tensor.empty() : tensor<1024x5120xf16>
  %fill87 = linalg.fill ins(%cst : f16) outs(%init86 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm88 = linalg.matmul ins(%add85, %w_ff2_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill87 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty89 = tensor.empty() : tensor<1024x5120xf16>
  %relu90 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm88 : tensor<1024x5120xf16>)
    outs(%empty89 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init91 = tensor.empty() : tensor<1024x1280xf16>
  %fill92 = linalg.fill ins(%cst : f16) outs(%init91 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm93 = linalg.matmul ins(%relu90, %w_ff2_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill92 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty94 = tensor.empty() : tensor<1024x1280xf16>
  %add95 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm93, %add85 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty94 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 3 ===

  // Q projection
  %init96 = tensor.empty() : tensor<1024x1280xf16>
  %fill97 = linalg.fill ins(%cst : f16) outs(%init96 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm98 = linalg.matmul ins(%add95, %w_q3 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill97 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init99 = tensor.empty() : tensor<1024x1280xf16>
  %fill100 = linalg.fill ins(%cst : f16) outs(%init99 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm101 = linalg.matmul ins(%add95, %w_k3 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill100 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init102 = tensor.empty() : tensor<1024x1280xf16>
  %fill103 = linalg.fill ins(%cst : f16) outs(%init102 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm104 = linalg.matmul ins(%add95, %w_v3 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill103 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init105 = tensor.empty() : tensor<1024x1024xf16>
  %fill106 = linalg.fill ins(%cst : f16) outs(%init105 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm107 = linalg.matmul ins(%mm98, %w_kt3 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill106 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty108 = tensor.empty() : tensor<1024x1024xf16>
  %relu109 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm107 : tensor<1024x1024xf16>)
    outs(%empty108 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init110 = tensor.empty() : tensor<1024x1280xf16>
  %fill111 = linalg.fill ins(%cst : f16) outs(%init110 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm112 = linalg.matmul ins(%relu109, %mm104 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill111 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init113 = tensor.empty() : tensor<1024x1280xf16>
  %fill114 = linalg.fill ins(%cst : f16) outs(%init113 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm115 = linalg.matmul ins(%mm112, %w_o3 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill114 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty116 = tensor.empty() : tensor<1024x1280xf16>
  %add117 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm115, %add95 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty116 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init118 = tensor.empty() : tensor<1024x5120xf16>
  %fill119 = linalg.fill ins(%cst : f16) outs(%init118 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm120 = linalg.matmul ins(%add117, %w_ff3_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill119 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty121 = tensor.empty() : tensor<1024x5120xf16>
  %relu122 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm120 : tensor<1024x5120xf16>)
    outs(%empty121 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init123 = tensor.empty() : tensor<1024x1280xf16>
  %fill124 = linalg.fill ins(%cst : f16) outs(%init123 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm125 = linalg.matmul ins(%relu122, %w_ff3_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill124 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty126 = tensor.empty() : tensor<1024x1280xf16>
  %add127 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm125, %add117 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty126 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 4 ===

  // Q projection
  %init128 = tensor.empty() : tensor<1024x1280xf16>
  %fill129 = linalg.fill ins(%cst : f16) outs(%init128 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm130 = linalg.matmul ins(%add127, %w_q4 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill129 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init131 = tensor.empty() : tensor<1024x1280xf16>
  %fill132 = linalg.fill ins(%cst : f16) outs(%init131 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm133 = linalg.matmul ins(%add127, %w_k4 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill132 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init134 = tensor.empty() : tensor<1024x1280xf16>
  %fill135 = linalg.fill ins(%cst : f16) outs(%init134 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm136 = linalg.matmul ins(%add127, %w_v4 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill135 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init137 = tensor.empty() : tensor<1024x1024xf16>
  %fill138 = linalg.fill ins(%cst : f16) outs(%init137 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm139 = linalg.matmul ins(%mm130, %w_kt4 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill138 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty140 = tensor.empty() : tensor<1024x1024xf16>
  %relu141 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm139 : tensor<1024x1024xf16>)
    outs(%empty140 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init142 = tensor.empty() : tensor<1024x1280xf16>
  %fill143 = linalg.fill ins(%cst : f16) outs(%init142 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm144 = linalg.matmul ins(%relu141, %mm136 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill143 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init145 = tensor.empty() : tensor<1024x1280xf16>
  %fill146 = linalg.fill ins(%cst : f16) outs(%init145 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm147 = linalg.matmul ins(%mm144, %w_o4 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill146 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty148 = tensor.empty() : tensor<1024x1280xf16>
  %add149 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm147, %add127 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty148 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init150 = tensor.empty() : tensor<1024x5120xf16>
  %fill151 = linalg.fill ins(%cst : f16) outs(%init150 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm152 = linalg.matmul ins(%add149, %w_ff4_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill151 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty153 = tensor.empty() : tensor<1024x5120xf16>
  %relu154 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm152 : tensor<1024x5120xf16>)
    outs(%empty153 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init155 = tensor.empty() : tensor<1024x1280xf16>
  %fill156 = linalg.fill ins(%cst : f16) outs(%init155 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm157 = linalg.matmul ins(%relu154, %w_ff4_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill156 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty158 = tensor.empty() : tensor<1024x1280xf16>
  %add159 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm157, %add149 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty158 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 5 ===

  // Q projection
  %init160 = tensor.empty() : tensor<1024x1280xf16>
  %fill161 = linalg.fill ins(%cst : f16) outs(%init160 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm162 = linalg.matmul ins(%add159, %w_q5 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill161 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init163 = tensor.empty() : tensor<1024x1280xf16>
  %fill164 = linalg.fill ins(%cst : f16) outs(%init163 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm165 = linalg.matmul ins(%add159, %w_k5 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill164 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init166 = tensor.empty() : tensor<1024x1280xf16>
  %fill167 = linalg.fill ins(%cst : f16) outs(%init166 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm168 = linalg.matmul ins(%add159, %w_v5 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill167 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init169 = tensor.empty() : tensor<1024x1024xf16>
  %fill170 = linalg.fill ins(%cst : f16) outs(%init169 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm171 = linalg.matmul ins(%mm162, %w_kt5 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill170 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty172 = tensor.empty() : tensor<1024x1024xf16>
  %relu173 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm171 : tensor<1024x1024xf16>)
    outs(%empty172 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init174 = tensor.empty() : tensor<1024x1280xf16>
  %fill175 = linalg.fill ins(%cst : f16) outs(%init174 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm176 = linalg.matmul ins(%relu173, %mm168 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill175 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init177 = tensor.empty() : tensor<1024x1280xf16>
  %fill178 = linalg.fill ins(%cst : f16) outs(%init177 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm179 = linalg.matmul ins(%mm176, %w_o5 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill178 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty180 = tensor.empty() : tensor<1024x1280xf16>
  %add181 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm179, %add159 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty180 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init182 = tensor.empty() : tensor<1024x5120xf16>
  %fill183 = linalg.fill ins(%cst : f16) outs(%init182 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm184 = linalg.matmul ins(%add181, %w_ff5_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill183 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty185 = tensor.empty() : tensor<1024x5120xf16>
  %relu186 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm184 : tensor<1024x5120xf16>)
    outs(%empty185 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init187 = tensor.empty() : tensor<1024x1280xf16>
  %fill188 = linalg.fill ins(%cst : f16) outs(%init187 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm189 = linalg.matmul ins(%relu186, %w_ff5_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill188 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty190 = tensor.empty() : tensor<1024x1280xf16>
  %add191 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm189, %add181 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty190 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 6 ===

  // Q projection
  %init192 = tensor.empty() : tensor<1024x1280xf16>
  %fill193 = linalg.fill ins(%cst : f16) outs(%init192 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm194 = linalg.matmul ins(%add191, %w_q6 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill193 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init195 = tensor.empty() : tensor<1024x1280xf16>
  %fill196 = linalg.fill ins(%cst : f16) outs(%init195 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm197 = linalg.matmul ins(%add191, %w_k6 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill196 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init198 = tensor.empty() : tensor<1024x1280xf16>
  %fill199 = linalg.fill ins(%cst : f16) outs(%init198 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm200 = linalg.matmul ins(%add191, %w_v6 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill199 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init201 = tensor.empty() : tensor<1024x1024xf16>
  %fill202 = linalg.fill ins(%cst : f16) outs(%init201 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm203 = linalg.matmul ins(%mm194, %w_kt6 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill202 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty204 = tensor.empty() : tensor<1024x1024xf16>
  %relu205 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm203 : tensor<1024x1024xf16>)
    outs(%empty204 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init206 = tensor.empty() : tensor<1024x1280xf16>
  %fill207 = linalg.fill ins(%cst : f16) outs(%init206 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm208 = linalg.matmul ins(%relu205, %mm200 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill207 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init209 = tensor.empty() : tensor<1024x1280xf16>
  %fill210 = linalg.fill ins(%cst : f16) outs(%init209 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm211 = linalg.matmul ins(%mm208, %w_o6 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill210 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty212 = tensor.empty() : tensor<1024x1280xf16>
  %add213 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm211, %add191 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty212 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init214 = tensor.empty() : tensor<1024x5120xf16>
  %fill215 = linalg.fill ins(%cst : f16) outs(%init214 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm216 = linalg.matmul ins(%add213, %w_ff6_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill215 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty217 = tensor.empty() : tensor<1024x5120xf16>
  %relu218 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm216 : tensor<1024x5120xf16>)
    outs(%empty217 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init219 = tensor.empty() : tensor<1024x1280xf16>
  %fill220 = linalg.fill ins(%cst : f16) outs(%init219 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm221 = linalg.matmul ins(%relu218, %w_ff6_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill220 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty222 = tensor.empty() : tensor<1024x1280xf16>
  %add223 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm221, %add213 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty222 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 7 ===

  // Q projection
  %init224 = tensor.empty() : tensor<1024x1280xf16>
  %fill225 = linalg.fill ins(%cst : f16) outs(%init224 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm226 = linalg.matmul ins(%add223, %w_q7 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill225 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init227 = tensor.empty() : tensor<1024x1280xf16>
  %fill228 = linalg.fill ins(%cst : f16) outs(%init227 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm229 = linalg.matmul ins(%add223, %w_k7 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill228 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init230 = tensor.empty() : tensor<1024x1280xf16>
  %fill231 = linalg.fill ins(%cst : f16) outs(%init230 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm232 = linalg.matmul ins(%add223, %w_v7 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill231 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init233 = tensor.empty() : tensor<1024x1024xf16>
  %fill234 = linalg.fill ins(%cst : f16) outs(%init233 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm235 = linalg.matmul ins(%mm226, %w_kt7 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill234 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty236 = tensor.empty() : tensor<1024x1024xf16>
  %relu237 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm235 : tensor<1024x1024xf16>)
    outs(%empty236 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init238 = tensor.empty() : tensor<1024x1280xf16>
  %fill239 = linalg.fill ins(%cst : f16) outs(%init238 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm240 = linalg.matmul ins(%relu237, %mm232 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill239 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init241 = tensor.empty() : tensor<1024x1280xf16>
  %fill242 = linalg.fill ins(%cst : f16) outs(%init241 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm243 = linalg.matmul ins(%mm240, %w_o7 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill242 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty244 = tensor.empty() : tensor<1024x1280xf16>
  %add245 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm243, %add223 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty244 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init246 = tensor.empty() : tensor<1024x5120xf16>
  %fill247 = linalg.fill ins(%cst : f16) outs(%init246 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm248 = linalg.matmul ins(%add245, %w_ff7_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill247 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty249 = tensor.empty() : tensor<1024x5120xf16>
  %relu250 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm248 : tensor<1024x5120xf16>)
    outs(%empty249 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init251 = tensor.empty() : tensor<1024x1280xf16>
  %fill252 = linalg.fill ins(%cst : f16) outs(%init251 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm253 = linalg.matmul ins(%relu250, %w_ff7_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill252 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty254 = tensor.empty() : tensor<1024x1280xf16>
  %add255 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm253, %add245 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty254 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 8 ===

  // Q projection
  %init256 = tensor.empty() : tensor<1024x1280xf16>
  %fill257 = linalg.fill ins(%cst : f16) outs(%init256 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm258 = linalg.matmul ins(%add255, %w_q8 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill257 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init259 = tensor.empty() : tensor<1024x1280xf16>
  %fill260 = linalg.fill ins(%cst : f16) outs(%init259 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm261 = linalg.matmul ins(%add255, %w_k8 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill260 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init262 = tensor.empty() : tensor<1024x1280xf16>
  %fill263 = linalg.fill ins(%cst : f16) outs(%init262 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm264 = linalg.matmul ins(%add255, %w_v8 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill263 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init265 = tensor.empty() : tensor<1024x1024xf16>
  %fill266 = linalg.fill ins(%cst : f16) outs(%init265 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm267 = linalg.matmul ins(%mm258, %w_kt8 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill266 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty268 = tensor.empty() : tensor<1024x1024xf16>
  %relu269 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm267 : tensor<1024x1024xf16>)
    outs(%empty268 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init270 = tensor.empty() : tensor<1024x1280xf16>
  %fill271 = linalg.fill ins(%cst : f16) outs(%init270 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm272 = linalg.matmul ins(%relu269, %mm264 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill271 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init273 = tensor.empty() : tensor<1024x1280xf16>
  %fill274 = linalg.fill ins(%cst : f16) outs(%init273 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm275 = linalg.matmul ins(%mm272, %w_o8 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill274 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty276 = tensor.empty() : tensor<1024x1280xf16>
  %add277 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm275, %add255 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty276 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init278 = tensor.empty() : tensor<1024x5120xf16>
  %fill279 = linalg.fill ins(%cst : f16) outs(%init278 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm280 = linalg.matmul ins(%add277, %w_ff8_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill279 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty281 = tensor.empty() : tensor<1024x5120xf16>
  %relu282 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm280 : tensor<1024x5120xf16>)
    outs(%empty281 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init283 = tensor.empty() : tensor<1024x1280xf16>
  %fill284 = linalg.fill ins(%cst : f16) outs(%init283 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm285 = linalg.matmul ins(%relu282, %w_ff8_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill284 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty286 = tensor.empty() : tensor<1024x1280xf16>
  %add287 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm285, %add277 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty286 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 9 ===

  // Q projection
  %init288 = tensor.empty() : tensor<1024x1280xf16>
  %fill289 = linalg.fill ins(%cst : f16) outs(%init288 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm290 = linalg.matmul ins(%add287, %w_q9 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill289 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init291 = tensor.empty() : tensor<1024x1280xf16>
  %fill292 = linalg.fill ins(%cst : f16) outs(%init291 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm293 = linalg.matmul ins(%add287, %w_k9 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill292 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init294 = tensor.empty() : tensor<1024x1280xf16>
  %fill295 = linalg.fill ins(%cst : f16) outs(%init294 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm296 = linalg.matmul ins(%add287, %w_v9 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill295 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init297 = tensor.empty() : tensor<1024x1024xf16>
  %fill298 = linalg.fill ins(%cst : f16) outs(%init297 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm299 = linalg.matmul ins(%mm290, %w_kt9 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill298 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty300 = tensor.empty() : tensor<1024x1024xf16>
  %relu301 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm299 : tensor<1024x1024xf16>)
    outs(%empty300 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init302 = tensor.empty() : tensor<1024x1280xf16>
  %fill303 = linalg.fill ins(%cst : f16) outs(%init302 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm304 = linalg.matmul ins(%relu301, %mm296 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill303 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init305 = tensor.empty() : tensor<1024x1280xf16>
  %fill306 = linalg.fill ins(%cst : f16) outs(%init305 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm307 = linalg.matmul ins(%mm304, %w_o9 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill306 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty308 = tensor.empty() : tensor<1024x1280xf16>
  %add309 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm307, %add287 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty308 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init310 = tensor.empty() : tensor<1024x5120xf16>
  %fill311 = linalg.fill ins(%cst : f16) outs(%init310 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm312 = linalg.matmul ins(%add309, %w_ff9_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill311 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty313 = tensor.empty() : tensor<1024x5120xf16>
  %relu314 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm312 : tensor<1024x5120xf16>)
    outs(%empty313 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init315 = tensor.empty() : tensor<1024x1280xf16>
  %fill316 = linalg.fill ins(%cst : f16) outs(%init315 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm317 = linalg.matmul ins(%relu314, %w_ff9_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill316 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty318 = tensor.empty() : tensor<1024x1280xf16>
  %add319 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm317, %add309 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty318 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 10 ===

  // Q projection
  %init320 = tensor.empty() : tensor<1024x1280xf16>
  %fill321 = linalg.fill ins(%cst : f16) outs(%init320 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm322 = linalg.matmul ins(%add319, %w_q10 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill321 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init323 = tensor.empty() : tensor<1024x1280xf16>
  %fill324 = linalg.fill ins(%cst : f16) outs(%init323 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm325 = linalg.matmul ins(%add319, %w_k10 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill324 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init326 = tensor.empty() : tensor<1024x1280xf16>
  %fill327 = linalg.fill ins(%cst : f16) outs(%init326 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm328 = linalg.matmul ins(%add319, %w_v10 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill327 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init329 = tensor.empty() : tensor<1024x1024xf16>
  %fill330 = linalg.fill ins(%cst : f16) outs(%init329 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm331 = linalg.matmul ins(%mm322, %w_kt10 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill330 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty332 = tensor.empty() : tensor<1024x1024xf16>
  %relu333 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm331 : tensor<1024x1024xf16>)
    outs(%empty332 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init334 = tensor.empty() : tensor<1024x1280xf16>
  %fill335 = linalg.fill ins(%cst : f16) outs(%init334 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm336 = linalg.matmul ins(%relu333, %mm328 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill335 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init337 = tensor.empty() : tensor<1024x1280xf16>
  %fill338 = linalg.fill ins(%cst : f16) outs(%init337 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm339 = linalg.matmul ins(%mm336, %w_o10 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill338 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty340 = tensor.empty() : tensor<1024x1280xf16>
  %add341 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm339, %add319 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty340 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init342 = tensor.empty() : tensor<1024x5120xf16>
  %fill343 = linalg.fill ins(%cst : f16) outs(%init342 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm344 = linalg.matmul ins(%add341, %w_ff10_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill343 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty345 = tensor.empty() : tensor<1024x5120xf16>
  %relu346 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm344 : tensor<1024x5120xf16>)
    outs(%empty345 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init347 = tensor.empty() : tensor<1024x1280xf16>
  %fill348 = linalg.fill ins(%cst : f16) outs(%init347 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm349 = linalg.matmul ins(%relu346, %w_ff10_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill348 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty350 = tensor.empty() : tensor<1024x1280xf16>
  %add351 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm349, %add341 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty350 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 11 ===

  // Q projection
  %init352 = tensor.empty() : tensor<1024x1280xf16>
  %fill353 = linalg.fill ins(%cst : f16) outs(%init352 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm354 = linalg.matmul ins(%add351, %w_q11 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill353 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init355 = tensor.empty() : tensor<1024x1280xf16>
  %fill356 = linalg.fill ins(%cst : f16) outs(%init355 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm357 = linalg.matmul ins(%add351, %w_k11 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill356 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init358 = tensor.empty() : tensor<1024x1280xf16>
  %fill359 = linalg.fill ins(%cst : f16) outs(%init358 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm360 = linalg.matmul ins(%add351, %w_v11 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill359 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init361 = tensor.empty() : tensor<1024x1024xf16>
  %fill362 = linalg.fill ins(%cst : f16) outs(%init361 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm363 = linalg.matmul ins(%mm354, %w_kt11 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill362 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty364 = tensor.empty() : tensor<1024x1024xf16>
  %relu365 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm363 : tensor<1024x1024xf16>)
    outs(%empty364 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init366 = tensor.empty() : tensor<1024x1280xf16>
  %fill367 = linalg.fill ins(%cst : f16) outs(%init366 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm368 = linalg.matmul ins(%relu365, %mm360 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill367 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init369 = tensor.empty() : tensor<1024x1280xf16>
  %fill370 = linalg.fill ins(%cst : f16) outs(%init369 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm371 = linalg.matmul ins(%mm368, %w_o11 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill370 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty372 = tensor.empty() : tensor<1024x1280xf16>
  %add373 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm371, %add351 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty372 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init374 = tensor.empty() : tensor<1024x5120xf16>
  %fill375 = linalg.fill ins(%cst : f16) outs(%init374 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm376 = linalg.matmul ins(%add373, %w_ff11_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill375 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty377 = tensor.empty() : tensor<1024x5120xf16>
  %relu378 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm376 : tensor<1024x5120xf16>)
    outs(%empty377 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init379 = tensor.empty() : tensor<1024x1280xf16>
  %fill380 = linalg.fill ins(%cst : f16) outs(%init379 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm381 = linalg.matmul ins(%relu378, %w_ff11_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill380 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty382 = tensor.empty() : tensor<1024x1280xf16>
  %add383 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm381, %add373 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty382 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 12 ===

  // Q projection
  %init384 = tensor.empty() : tensor<1024x1280xf16>
  %fill385 = linalg.fill ins(%cst : f16) outs(%init384 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm386 = linalg.matmul ins(%add383, %w_q12 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill385 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init387 = tensor.empty() : tensor<1024x1280xf16>
  %fill388 = linalg.fill ins(%cst : f16) outs(%init387 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm389 = linalg.matmul ins(%add383, %w_k12 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill388 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init390 = tensor.empty() : tensor<1024x1280xf16>
  %fill391 = linalg.fill ins(%cst : f16) outs(%init390 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm392 = linalg.matmul ins(%add383, %w_v12 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill391 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init393 = tensor.empty() : tensor<1024x1024xf16>
  %fill394 = linalg.fill ins(%cst : f16) outs(%init393 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm395 = linalg.matmul ins(%mm386, %w_kt12 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill394 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty396 = tensor.empty() : tensor<1024x1024xf16>
  %relu397 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm395 : tensor<1024x1024xf16>)
    outs(%empty396 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init398 = tensor.empty() : tensor<1024x1280xf16>
  %fill399 = linalg.fill ins(%cst : f16) outs(%init398 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm400 = linalg.matmul ins(%relu397, %mm392 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill399 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init401 = tensor.empty() : tensor<1024x1280xf16>
  %fill402 = linalg.fill ins(%cst : f16) outs(%init401 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm403 = linalg.matmul ins(%mm400, %w_o12 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill402 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty404 = tensor.empty() : tensor<1024x1280xf16>
  %add405 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm403, %add383 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty404 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init406 = tensor.empty() : tensor<1024x5120xf16>
  %fill407 = linalg.fill ins(%cst : f16) outs(%init406 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm408 = linalg.matmul ins(%add405, %w_ff12_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill407 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty409 = tensor.empty() : tensor<1024x5120xf16>
  %relu410 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm408 : tensor<1024x5120xf16>)
    outs(%empty409 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init411 = tensor.empty() : tensor<1024x1280xf16>
  %fill412 = linalg.fill ins(%cst : f16) outs(%init411 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm413 = linalg.matmul ins(%relu410, %w_ff12_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill412 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty414 = tensor.empty() : tensor<1024x1280xf16>
  %add415 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm413, %add405 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty414 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 13 ===

  // Q projection
  %init416 = tensor.empty() : tensor<1024x1280xf16>
  %fill417 = linalg.fill ins(%cst : f16) outs(%init416 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm418 = linalg.matmul ins(%add415, %w_q13 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill417 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init419 = tensor.empty() : tensor<1024x1280xf16>
  %fill420 = linalg.fill ins(%cst : f16) outs(%init419 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm421 = linalg.matmul ins(%add415, %w_k13 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill420 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init422 = tensor.empty() : tensor<1024x1280xf16>
  %fill423 = linalg.fill ins(%cst : f16) outs(%init422 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm424 = linalg.matmul ins(%add415, %w_v13 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill423 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init425 = tensor.empty() : tensor<1024x1024xf16>
  %fill426 = linalg.fill ins(%cst : f16) outs(%init425 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm427 = linalg.matmul ins(%mm418, %w_kt13 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill426 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty428 = tensor.empty() : tensor<1024x1024xf16>
  %relu429 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm427 : tensor<1024x1024xf16>)
    outs(%empty428 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init430 = tensor.empty() : tensor<1024x1280xf16>
  %fill431 = linalg.fill ins(%cst : f16) outs(%init430 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm432 = linalg.matmul ins(%relu429, %mm424 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill431 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init433 = tensor.empty() : tensor<1024x1280xf16>
  %fill434 = linalg.fill ins(%cst : f16) outs(%init433 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm435 = linalg.matmul ins(%mm432, %w_o13 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill434 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty436 = tensor.empty() : tensor<1024x1280xf16>
  %add437 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm435, %add415 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty436 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init438 = tensor.empty() : tensor<1024x5120xf16>
  %fill439 = linalg.fill ins(%cst : f16) outs(%init438 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm440 = linalg.matmul ins(%add437, %w_ff13_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill439 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty441 = tensor.empty() : tensor<1024x5120xf16>
  %relu442 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm440 : tensor<1024x5120xf16>)
    outs(%empty441 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init443 = tensor.empty() : tensor<1024x1280xf16>
  %fill444 = linalg.fill ins(%cst : f16) outs(%init443 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm445 = linalg.matmul ins(%relu442, %w_ff13_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill444 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty446 = tensor.empty() : tensor<1024x1280xf16>
  %add447 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm445, %add437 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty446 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 14 ===

  // Q projection
  %init448 = tensor.empty() : tensor<1024x1280xf16>
  %fill449 = linalg.fill ins(%cst : f16) outs(%init448 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm450 = linalg.matmul ins(%add447, %w_q14 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill449 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init451 = tensor.empty() : tensor<1024x1280xf16>
  %fill452 = linalg.fill ins(%cst : f16) outs(%init451 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm453 = linalg.matmul ins(%add447, %w_k14 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill452 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init454 = tensor.empty() : tensor<1024x1280xf16>
  %fill455 = linalg.fill ins(%cst : f16) outs(%init454 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm456 = linalg.matmul ins(%add447, %w_v14 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill455 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init457 = tensor.empty() : tensor<1024x1024xf16>
  %fill458 = linalg.fill ins(%cst : f16) outs(%init457 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm459 = linalg.matmul ins(%mm450, %w_kt14 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill458 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty460 = tensor.empty() : tensor<1024x1024xf16>
  %relu461 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm459 : tensor<1024x1024xf16>)
    outs(%empty460 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init462 = tensor.empty() : tensor<1024x1280xf16>
  %fill463 = linalg.fill ins(%cst : f16) outs(%init462 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm464 = linalg.matmul ins(%relu461, %mm456 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill463 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init465 = tensor.empty() : tensor<1024x1280xf16>
  %fill466 = linalg.fill ins(%cst : f16) outs(%init465 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm467 = linalg.matmul ins(%mm464, %w_o14 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill466 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty468 = tensor.empty() : tensor<1024x1280xf16>
  %add469 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm467, %add447 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty468 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init470 = tensor.empty() : tensor<1024x5120xf16>
  %fill471 = linalg.fill ins(%cst : f16) outs(%init470 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm472 = linalg.matmul ins(%add469, %w_ff14_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill471 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty473 = tensor.empty() : tensor<1024x5120xf16>
  %relu474 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm472 : tensor<1024x5120xf16>)
    outs(%empty473 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init475 = tensor.empty() : tensor<1024x1280xf16>
  %fill476 = linalg.fill ins(%cst : f16) outs(%init475 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm477 = linalg.matmul ins(%relu474, %w_ff14_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill476 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty478 = tensor.empty() : tensor<1024x1280xf16>
  %add479 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm477, %add469 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty478 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 15 ===

  // Q projection
  %init480 = tensor.empty() : tensor<1024x1280xf16>
  %fill481 = linalg.fill ins(%cst : f16) outs(%init480 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm482 = linalg.matmul ins(%add479, %w_q15 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill481 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init483 = tensor.empty() : tensor<1024x1280xf16>
  %fill484 = linalg.fill ins(%cst : f16) outs(%init483 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm485 = linalg.matmul ins(%add479, %w_k15 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill484 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init486 = tensor.empty() : tensor<1024x1280xf16>
  %fill487 = linalg.fill ins(%cst : f16) outs(%init486 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm488 = linalg.matmul ins(%add479, %w_v15 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill487 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init489 = tensor.empty() : tensor<1024x1024xf16>
  %fill490 = linalg.fill ins(%cst : f16) outs(%init489 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm491 = linalg.matmul ins(%mm482, %w_kt15 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill490 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty492 = tensor.empty() : tensor<1024x1024xf16>
  %relu493 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm491 : tensor<1024x1024xf16>)
    outs(%empty492 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init494 = tensor.empty() : tensor<1024x1280xf16>
  %fill495 = linalg.fill ins(%cst : f16) outs(%init494 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm496 = linalg.matmul ins(%relu493, %mm488 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill495 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init497 = tensor.empty() : tensor<1024x1280xf16>
  %fill498 = linalg.fill ins(%cst : f16) outs(%init497 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm499 = linalg.matmul ins(%mm496, %w_o15 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill498 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty500 = tensor.empty() : tensor<1024x1280xf16>
  %add501 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm499, %add479 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty500 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init502 = tensor.empty() : tensor<1024x5120xf16>
  %fill503 = linalg.fill ins(%cst : f16) outs(%init502 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm504 = linalg.matmul ins(%add501, %w_ff15_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill503 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty505 = tensor.empty() : tensor<1024x5120xf16>
  %relu506 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm504 : tensor<1024x5120xf16>)
    outs(%empty505 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init507 = tensor.empty() : tensor<1024x1280xf16>
  %fill508 = linalg.fill ins(%cst : f16) outs(%init507 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm509 = linalg.matmul ins(%relu506, %w_ff15_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill508 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty510 = tensor.empty() : tensor<1024x1280xf16>
  %add511 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm509, %add501 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty510 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 16 ===

  // Q projection
  %init512 = tensor.empty() : tensor<1024x1280xf16>
  %fill513 = linalg.fill ins(%cst : f16) outs(%init512 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm514 = linalg.matmul ins(%add511, %w_q16 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill513 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init515 = tensor.empty() : tensor<1024x1280xf16>
  %fill516 = linalg.fill ins(%cst : f16) outs(%init515 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm517 = linalg.matmul ins(%add511, %w_k16 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill516 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init518 = tensor.empty() : tensor<1024x1280xf16>
  %fill519 = linalg.fill ins(%cst : f16) outs(%init518 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm520 = linalg.matmul ins(%add511, %w_v16 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill519 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init521 = tensor.empty() : tensor<1024x1024xf16>
  %fill522 = linalg.fill ins(%cst : f16) outs(%init521 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm523 = linalg.matmul ins(%mm514, %w_kt16 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill522 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty524 = tensor.empty() : tensor<1024x1024xf16>
  %relu525 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm523 : tensor<1024x1024xf16>)
    outs(%empty524 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init526 = tensor.empty() : tensor<1024x1280xf16>
  %fill527 = linalg.fill ins(%cst : f16) outs(%init526 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm528 = linalg.matmul ins(%relu525, %mm520 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill527 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init529 = tensor.empty() : tensor<1024x1280xf16>
  %fill530 = linalg.fill ins(%cst : f16) outs(%init529 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm531 = linalg.matmul ins(%mm528, %w_o16 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill530 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty532 = tensor.empty() : tensor<1024x1280xf16>
  %add533 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm531, %add511 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty532 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init534 = tensor.empty() : tensor<1024x5120xf16>
  %fill535 = linalg.fill ins(%cst : f16) outs(%init534 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm536 = linalg.matmul ins(%add533, %w_ff16_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill535 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty537 = tensor.empty() : tensor<1024x5120xf16>
  %relu538 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm536 : tensor<1024x5120xf16>)
    outs(%empty537 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init539 = tensor.empty() : tensor<1024x1280xf16>
  %fill540 = linalg.fill ins(%cst : f16) outs(%init539 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm541 = linalg.matmul ins(%relu538, %w_ff16_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill540 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty542 = tensor.empty() : tensor<1024x1280xf16>
  %add543 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm541, %add533 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty542 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 17 ===

  // Q projection
  %init544 = tensor.empty() : tensor<1024x1280xf16>
  %fill545 = linalg.fill ins(%cst : f16) outs(%init544 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm546 = linalg.matmul ins(%add543, %w_q17 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill545 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init547 = tensor.empty() : tensor<1024x1280xf16>
  %fill548 = linalg.fill ins(%cst : f16) outs(%init547 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm549 = linalg.matmul ins(%add543, %w_k17 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill548 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init550 = tensor.empty() : tensor<1024x1280xf16>
  %fill551 = linalg.fill ins(%cst : f16) outs(%init550 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm552 = linalg.matmul ins(%add543, %w_v17 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill551 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init553 = tensor.empty() : tensor<1024x1024xf16>
  %fill554 = linalg.fill ins(%cst : f16) outs(%init553 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm555 = linalg.matmul ins(%mm546, %w_kt17 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill554 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty556 = tensor.empty() : tensor<1024x1024xf16>
  %relu557 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm555 : tensor<1024x1024xf16>)
    outs(%empty556 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init558 = tensor.empty() : tensor<1024x1280xf16>
  %fill559 = linalg.fill ins(%cst : f16) outs(%init558 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm560 = linalg.matmul ins(%relu557, %mm552 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill559 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init561 = tensor.empty() : tensor<1024x1280xf16>
  %fill562 = linalg.fill ins(%cst : f16) outs(%init561 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm563 = linalg.matmul ins(%mm560, %w_o17 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill562 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty564 = tensor.empty() : tensor<1024x1280xf16>
  %add565 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm563, %add543 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty564 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init566 = tensor.empty() : tensor<1024x5120xf16>
  %fill567 = linalg.fill ins(%cst : f16) outs(%init566 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm568 = linalg.matmul ins(%add565, %w_ff17_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill567 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty569 = tensor.empty() : tensor<1024x5120xf16>
  %relu570 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm568 : tensor<1024x5120xf16>)
    outs(%empty569 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init571 = tensor.empty() : tensor<1024x1280xf16>
  %fill572 = linalg.fill ins(%cst : f16) outs(%init571 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm573 = linalg.matmul ins(%relu570, %w_ff17_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill572 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty574 = tensor.empty() : tensor<1024x1280xf16>
  %add575 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm573, %add565 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty574 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 18 ===

  // Q projection
  %init576 = tensor.empty() : tensor<1024x1280xf16>
  %fill577 = linalg.fill ins(%cst : f16) outs(%init576 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm578 = linalg.matmul ins(%add575, %w_q18 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill577 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init579 = tensor.empty() : tensor<1024x1280xf16>
  %fill580 = linalg.fill ins(%cst : f16) outs(%init579 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm581 = linalg.matmul ins(%add575, %w_k18 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill580 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init582 = tensor.empty() : tensor<1024x1280xf16>
  %fill583 = linalg.fill ins(%cst : f16) outs(%init582 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm584 = linalg.matmul ins(%add575, %w_v18 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill583 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init585 = tensor.empty() : tensor<1024x1024xf16>
  %fill586 = linalg.fill ins(%cst : f16) outs(%init585 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm587 = linalg.matmul ins(%mm578, %w_kt18 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill586 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty588 = tensor.empty() : tensor<1024x1024xf16>
  %relu589 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm587 : tensor<1024x1024xf16>)
    outs(%empty588 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init590 = tensor.empty() : tensor<1024x1280xf16>
  %fill591 = linalg.fill ins(%cst : f16) outs(%init590 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm592 = linalg.matmul ins(%relu589, %mm584 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill591 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init593 = tensor.empty() : tensor<1024x1280xf16>
  %fill594 = linalg.fill ins(%cst : f16) outs(%init593 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm595 = linalg.matmul ins(%mm592, %w_o18 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill594 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty596 = tensor.empty() : tensor<1024x1280xf16>
  %add597 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm595, %add575 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty596 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init598 = tensor.empty() : tensor<1024x5120xf16>
  %fill599 = linalg.fill ins(%cst : f16) outs(%init598 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm600 = linalg.matmul ins(%add597, %w_ff18_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill599 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty601 = tensor.empty() : tensor<1024x5120xf16>
  %relu602 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm600 : tensor<1024x5120xf16>)
    outs(%empty601 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init603 = tensor.empty() : tensor<1024x1280xf16>
  %fill604 = linalg.fill ins(%cst : f16) outs(%init603 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm605 = linalg.matmul ins(%relu602, %w_ff18_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill604 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty606 = tensor.empty() : tensor<1024x1280xf16>
  %add607 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm605, %add597 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty606 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 19 ===

  // Q projection
  %init608 = tensor.empty() : tensor<1024x1280xf16>
  %fill609 = linalg.fill ins(%cst : f16) outs(%init608 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm610 = linalg.matmul ins(%add607, %w_q19 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill609 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init611 = tensor.empty() : tensor<1024x1280xf16>
  %fill612 = linalg.fill ins(%cst : f16) outs(%init611 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm613 = linalg.matmul ins(%add607, %w_k19 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill612 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init614 = tensor.empty() : tensor<1024x1280xf16>
  %fill615 = linalg.fill ins(%cst : f16) outs(%init614 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm616 = linalg.matmul ins(%add607, %w_v19 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill615 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init617 = tensor.empty() : tensor<1024x1024xf16>
  %fill618 = linalg.fill ins(%cst : f16) outs(%init617 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm619 = linalg.matmul ins(%mm610, %w_kt19 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill618 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty620 = tensor.empty() : tensor<1024x1024xf16>
  %relu621 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm619 : tensor<1024x1024xf16>)
    outs(%empty620 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init622 = tensor.empty() : tensor<1024x1280xf16>
  %fill623 = linalg.fill ins(%cst : f16) outs(%init622 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm624 = linalg.matmul ins(%relu621, %mm616 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill623 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init625 = tensor.empty() : tensor<1024x1280xf16>
  %fill626 = linalg.fill ins(%cst : f16) outs(%init625 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm627 = linalg.matmul ins(%mm624, %w_o19 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill626 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty628 = tensor.empty() : tensor<1024x1280xf16>
  %add629 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm627, %add607 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty628 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init630 = tensor.empty() : tensor<1024x5120xf16>
  %fill631 = linalg.fill ins(%cst : f16) outs(%init630 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm632 = linalg.matmul ins(%add629, %w_ff19_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill631 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty633 = tensor.empty() : tensor<1024x5120xf16>
  %relu634 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm632 : tensor<1024x5120xf16>)
    outs(%empty633 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init635 = tensor.empty() : tensor<1024x1280xf16>
  %fill636 = linalg.fill ins(%cst : f16) outs(%init635 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm637 = linalg.matmul ins(%relu634, %w_ff19_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill636 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty638 = tensor.empty() : tensor<1024x1280xf16>
  %add639 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm637, %add629 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty638 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 20 ===

  // Q projection
  %init640 = tensor.empty() : tensor<1024x1280xf16>
  %fill641 = linalg.fill ins(%cst : f16) outs(%init640 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm642 = linalg.matmul ins(%add639, %w_q20 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill641 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init643 = tensor.empty() : tensor<1024x1280xf16>
  %fill644 = linalg.fill ins(%cst : f16) outs(%init643 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm645 = linalg.matmul ins(%add639, %w_k20 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill644 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init646 = tensor.empty() : tensor<1024x1280xf16>
  %fill647 = linalg.fill ins(%cst : f16) outs(%init646 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm648 = linalg.matmul ins(%add639, %w_v20 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill647 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init649 = tensor.empty() : tensor<1024x1024xf16>
  %fill650 = linalg.fill ins(%cst : f16) outs(%init649 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm651 = linalg.matmul ins(%mm642, %w_kt20 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill650 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty652 = tensor.empty() : tensor<1024x1024xf16>
  %relu653 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm651 : tensor<1024x1024xf16>)
    outs(%empty652 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init654 = tensor.empty() : tensor<1024x1280xf16>
  %fill655 = linalg.fill ins(%cst : f16) outs(%init654 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm656 = linalg.matmul ins(%relu653, %mm648 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill655 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init657 = tensor.empty() : tensor<1024x1280xf16>
  %fill658 = linalg.fill ins(%cst : f16) outs(%init657 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm659 = linalg.matmul ins(%mm656, %w_o20 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill658 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty660 = tensor.empty() : tensor<1024x1280xf16>
  %add661 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm659, %add639 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty660 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init662 = tensor.empty() : tensor<1024x5120xf16>
  %fill663 = linalg.fill ins(%cst : f16) outs(%init662 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm664 = linalg.matmul ins(%add661, %w_ff20_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill663 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty665 = tensor.empty() : tensor<1024x5120xf16>
  %relu666 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm664 : tensor<1024x5120xf16>)
    outs(%empty665 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init667 = tensor.empty() : tensor<1024x1280xf16>
  %fill668 = linalg.fill ins(%cst : f16) outs(%init667 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm669 = linalg.matmul ins(%relu666, %w_ff20_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill668 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty670 = tensor.empty() : tensor<1024x1280xf16>
  %add671 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm669, %add661 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty670 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 21 ===

  // Q projection
  %init672 = tensor.empty() : tensor<1024x1280xf16>
  %fill673 = linalg.fill ins(%cst : f16) outs(%init672 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm674 = linalg.matmul ins(%add671, %w_q21 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill673 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init675 = tensor.empty() : tensor<1024x1280xf16>
  %fill676 = linalg.fill ins(%cst : f16) outs(%init675 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm677 = linalg.matmul ins(%add671, %w_k21 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill676 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init678 = tensor.empty() : tensor<1024x1280xf16>
  %fill679 = linalg.fill ins(%cst : f16) outs(%init678 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm680 = linalg.matmul ins(%add671, %w_v21 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill679 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init681 = tensor.empty() : tensor<1024x1024xf16>
  %fill682 = linalg.fill ins(%cst : f16) outs(%init681 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm683 = linalg.matmul ins(%mm674, %w_kt21 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill682 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty684 = tensor.empty() : tensor<1024x1024xf16>
  %relu685 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm683 : tensor<1024x1024xf16>)
    outs(%empty684 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init686 = tensor.empty() : tensor<1024x1280xf16>
  %fill687 = linalg.fill ins(%cst : f16) outs(%init686 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm688 = linalg.matmul ins(%relu685, %mm680 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill687 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init689 = tensor.empty() : tensor<1024x1280xf16>
  %fill690 = linalg.fill ins(%cst : f16) outs(%init689 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm691 = linalg.matmul ins(%mm688, %w_o21 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill690 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty692 = tensor.empty() : tensor<1024x1280xf16>
  %add693 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm691, %add671 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty692 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init694 = tensor.empty() : tensor<1024x5120xf16>
  %fill695 = linalg.fill ins(%cst : f16) outs(%init694 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm696 = linalg.matmul ins(%add693, %w_ff21_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill695 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty697 = tensor.empty() : tensor<1024x5120xf16>
  %relu698 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm696 : tensor<1024x5120xf16>)
    outs(%empty697 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init699 = tensor.empty() : tensor<1024x1280xf16>
  %fill700 = linalg.fill ins(%cst : f16) outs(%init699 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm701 = linalg.matmul ins(%relu698, %w_ff21_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill700 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty702 = tensor.empty() : tensor<1024x1280xf16>
  %add703 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm701, %add693 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty702 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 22 ===

  // Q projection
  %init704 = tensor.empty() : tensor<1024x1280xf16>
  %fill705 = linalg.fill ins(%cst : f16) outs(%init704 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm706 = linalg.matmul ins(%add703, %w_q22 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill705 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init707 = tensor.empty() : tensor<1024x1280xf16>
  %fill708 = linalg.fill ins(%cst : f16) outs(%init707 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm709 = linalg.matmul ins(%add703, %w_k22 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill708 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init710 = tensor.empty() : tensor<1024x1280xf16>
  %fill711 = linalg.fill ins(%cst : f16) outs(%init710 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm712 = linalg.matmul ins(%add703, %w_v22 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill711 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init713 = tensor.empty() : tensor<1024x1024xf16>
  %fill714 = linalg.fill ins(%cst : f16) outs(%init713 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm715 = linalg.matmul ins(%mm706, %w_kt22 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill714 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty716 = tensor.empty() : tensor<1024x1024xf16>
  %relu717 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm715 : tensor<1024x1024xf16>)
    outs(%empty716 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init718 = tensor.empty() : tensor<1024x1280xf16>
  %fill719 = linalg.fill ins(%cst : f16) outs(%init718 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm720 = linalg.matmul ins(%relu717, %mm712 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill719 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init721 = tensor.empty() : tensor<1024x1280xf16>
  %fill722 = linalg.fill ins(%cst : f16) outs(%init721 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm723 = linalg.matmul ins(%mm720, %w_o22 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill722 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty724 = tensor.empty() : tensor<1024x1280xf16>
  %add725 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm723, %add703 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty724 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init726 = tensor.empty() : tensor<1024x5120xf16>
  %fill727 = linalg.fill ins(%cst : f16) outs(%init726 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm728 = linalg.matmul ins(%add725, %w_ff22_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill727 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty729 = tensor.empty() : tensor<1024x5120xf16>
  %relu730 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm728 : tensor<1024x5120xf16>)
    outs(%empty729 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init731 = tensor.empty() : tensor<1024x1280xf16>
  %fill732 = linalg.fill ins(%cst : f16) outs(%init731 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm733 = linalg.matmul ins(%relu730, %w_ff22_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill732 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty734 = tensor.empty() : tensor<1024x1280xf16>
  %add735 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm733, %add725 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty734 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 23 ===

  // Q projection
  %init736 = tensor.empty() : tensor<1024x1280xf16>
  %fill737 = linalg.fill ins(%cst : f16) outs(%init736 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm738 = linalg.matmul ins(%add735, %w_q23 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill737 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init739 = tensor.empty() : tensor<1024x1280xf16>
  %fill740 = linalg.fill ins(%cst : f16) outs(%init739 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm741 = linalg.matmul ins(%add735, %w_k23 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill740 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init742 = tensor.empty() : tensor<1024x1280xf16>
  %fill743 = linalg.fill ins(%cst : f16) outs(%init742 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm744 = linalg.matmul ins(%add735, %w_v23 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill743 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init745 = tensor.empty() : tensor<1024x1024xf16>
  %fill746 = linalg.fill ins(%cst : f16) outs(%init745 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm747 = linalg.matmul ins(%mm738, %w_kt23 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill746 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty748 = tensor.empty() : tensor<1024x1024xf16>
  %relu749 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm747 : tensor<1024x1024xf16>)
    outs(%empty748 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init750 = tensor.empty() : tensor<1024x1280xf16>
  %fill751 = linalg.fill ins(%cst : f16) outs(%init750 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm752 = linalg.matmul ins(%relu749, %mm744 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill751 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init753 = tensor.empty() : tensor<1024x1280xf16>
  %fill754 = linalg.fill ins(%cst : f16) outs(%init753 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm755 = linalg.matmul ins(%mm752, %w_o23 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill754 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty756 = tensor.empty() : tensor<1024x1280xf16>
  %add757 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm755, %add735 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty756 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init758 = tensor.empty() : tensor<1024x5120xf16>
  %fill759 = linalg.fill ins(%cst : f16) outs(%init758 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm760 = linalg.matmul ins(%add757, %w_ff23_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill759 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty761 = tensor.empty() : tensor<1024x5120xf16>
  %relu762 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm760 : tensor<1024x5120xf16>)
    outs(%empty761 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init763 = tensor.empty() : tensor<1024x1280xf16>
  %fill764 = linalg.fill ins(%cst : f16) outs(%init763 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm765 = linalg.matmul ins(%relu762, %w_ff23_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill764 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty766 = tensor.empty() : tensor<1024x1280xf16>
  %add767 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm765, %add757 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty766 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 24 ===

  // Q projection
  %init768 = tensor.empty() : tensor<1024x1280xf16>
  %fill769 = linalg.fill ins(%cst : f16) outs(%init768 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm770 = linalg.matmul ins(%add767, %w_q24 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill769 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init771 = tensor.empty() : tensor<1024x1280xf16>
  %fill772 = linalg.fill ins(%cst : f16) outs(%init771 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm773 = linalg.matmul ins(%add767, %w_k24 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill772 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init774 = tensor.empty() : tensor<1024x1280xf16>
  %fill775 = linalg.fill ins(%cst : f16) outs(%init774 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm776 = linalg.matmul ins(%add767, %w_v24 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill775 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init777 = tensor.empty() : tensor<1024x1024xf16>
  %fill778 = linalg.fill ins(%cst : f16) outs(%init777 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm779 = linalg.matmul ins(%mm770, %w_kt24 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill778 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty780 = tensor.empty() : tensor<1024x1024xf16>
  %relu781 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm779 : tensor<1024x1024xf16>)
    outs(%empty780 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init782 = tensor.empty() : tensor<1024x1280xf16>
  %fill783 = linalg.fill ins(%cst : f16) outs(%init782 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm784 = linalg.matmul ins(%relu781, %mm776 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill783 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init785 = tensor.empty() : tensor<1024x1280xf16>
  %fill786 = linalg.fill ins(%cst : f16) outs(%init785 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm787 = linalg.matmul ins(%mm784, %w_o24 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill786 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty788 = tensor.empty() : tensor<1024x1280xf16>
  %add789 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm787, %add767 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty788 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init790 = tensor.empty() : tensor<1024x5120xf16>
  %fill791 = linalg.fill ins(%cst : f16) outs(%init790 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm792 = linalg.matmul ins(%add789, %w_ff24_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill791 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty793 = tensor.empty() : tensor<1024x5120xf16>
  %relu794 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm792 : tensor<1024x5120xf16>)
    outs(%empty793 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init795 = tensor.empty() : tensor<1024x1280xf16>
  %fill796 = linalg.fill ins(%cst : f16) outs(%init795 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm797 = linalg.matmul ins(%relu794, %w_ff24_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill796 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty798 = tensor.empty() : tensor<1024x1280xf16>
  %add799 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm797, %add789 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty798 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 25 ===

  // Q projection
  %init800 = tensor.empty() : tensor<1024x1280xf16>
  %fill801 = linalg.fill ins(%cst : f16) outs(%init800 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm802 = linalg.matmul ins(%add799, %w_q25 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill801 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init803 = tensor.empty() : tensor<1024x1280xf16>
  %fill804 = linalg.fill ins(%cst : f16) outs(%init803 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm805 = linalg.matmul ins(%add799, %w_k25 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill804 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init806 = tensor.empty() : tensor<1024x1280xf16>
  %fill807 = linalg.fill ins(%cst : f16) outs(%init806 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm808 = linalg.matmul ins(%add799, %w_v25 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill807 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init809 = tensor.empty() : tensor<1024x1024xf16>
  %fill810 = linalg.fill ins(%cst : f16) outs(%init809 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm811 = linalg.matmul ins(%mm802, %w_kt25 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill810 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty812 = tensor.empty() : tensor<1024x1024xf16>
  %relu813 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm811 : tensor<1024x1024xf16>)
    outs(%empty812 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init814 = tensor.empty() : tensor<1024x1280xf16>
  %fill815 = linalg.fill ins(%cst : f16) outs(%init814 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm816 = linalg.matmul ins(%relu813, %mm808 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill815 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init817 = tensor.empty() : tensor<1024x1280xf16>
  %fill818 = linalg.fill ins(%cst : f16) outs(%init817 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm819 = linalg.matmul ins(%mm816, %w_o25 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill818 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty820 = tensor.empty() : tensor<1024x1280xf16>
  %add821 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm819, %add799 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty820 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init822 = tensor.empty() : tensor<1024x5120xf16>
  %fill823 = linalg.fill ins(%cst : f16) outs(%init822 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm824 = linalg.matmul ins(%add821, %w_ff25_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill823 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty825 = tensor.empty() : tensor<1024x5120xf16>
  %relu826 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm824 : tensor<1024x5120xf16>)
    outs(%empty825 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init827 = tensor.empty() : tensor<1024x1280xf16>
  %fill828 = linalg.fill ins(%cst : f16) outs(%init827 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm829 = linalg.matmul ins(%relu826, %w_ff25_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill828 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty830 = tensor.empty() : tensor<1024x1280xf16>
  %add831 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm829, %add821 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty830 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 26 ===

  // Q projection
  %init832 = tensor.empty() : tensor<1024x1280xf16>
  %fill833 = linalg.fill ins(%cst : f16) outs(%init832 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm834 = linalg.matmul ins(%add831, %w_q26 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill833 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init835 = tensor.empty() : tensor<1024x1280xf16>
  %fill836 = linalg.fill ins(%cst : f16) outs(%init835 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm837 = linalg.matmul ins(%add831, %w_k26 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill836 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init838 = tensor.empty() : tensor<1024x1280xf16>
  %fill839 = linalg.fill ins(%cst : f16) outs(%init838 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm840 = linalg.matmul ins(%add831, %w_v26 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill839 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init841 = tensor.empty() : tensor<1024x1024xf16>
  %fill842 = linalg.fill ins(%cst : f16) outs(%init841 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm843 = linalg.matmul ins(%mm834, %w_kt26 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill842 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty844 = tensor.empty() : tensor<1024x1024xf16>
  %relu845 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm843 : tensor<1024x1024xf16>)
    outs(%empty844 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init846 = tensor.empty() : tensor<1024x1280xf16>
  %fill847 = linalg.fill ins(%cst : f16) outs(%init846 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm848 = linalg.matmul ins(%relu845, %mm840 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill847 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init849 = tensor.empty() : tensor<1024x1280xf16>
  %fill850 = linalg.fill ins(%cst : f16) outs(%init849 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm851 = linalg.matmul ins(%mm848, %w_o26 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill850 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty852 = tensor.empty() : tensor<1024x1280xf16>
  %add853 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm851, %add831 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty852 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init854 = tensor.empty() : tensor<1024x5120xf16>
  %fill855 = linalg.fill ins(%cst : f16) outs(%init854 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm856 = linalg.matmul ins(%add853, %w_ff26_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill855 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty857 = tensor.empty() : tensor<1024x5120xf16>
  %relu858 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm856 : tensor<1024x5120xf16>)
    outs(%empty857 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init859 = tensor.empty() : tensor<1024x1280xf16>
  %fill860 = linalg.fill ins(%cst : f16) outs(%init859 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm861 = linalg.matmul ins(%relu858, %w_ff26_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill860 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty862 = tensor.empty() : tensor<1024x1280xf16>
  %add863 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm861, %add853 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty862 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 27 ===

  // Q projection
  %init864 = tensor.empty() : tensor<1024x1280xf16>
  %fill865 = linalg.fill ins(%cst : f16) outs(%init864 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm866 = linalg.matmul ins(%add863, %w_q27 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill865 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init867 = tensor.empty() : tensor<1024x1280xf16>
  %fill868 = linalg.fill ins(%cst : f16) outs(%init867 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm869 = linalg.matmul ins(%add863, %w_k27 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill868 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init870 = tensor.empty() : tensor<1024x1280xf16>
  %fill871 = linalg.fill ins(%cst : f16) outs(%init870 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm872 = linalg.matmul ins(%add863, %w_v27 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill871 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init873 = tensor.empty() : tensor<1024x1024xf16>
  %fill874 = linalg.fill ins(%cst : f16) outs(%init873 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm875 = linalg.matmul ins(%mm866, %w_kt27 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill874 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty876 = tensor.empty() : tensor<1024x1024xf16>
  %relu877 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm875 : tensor<1024x1024xf16>)
    outs(%empty876 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init878 = tensor.empty() : tensor<1024x1280xf16>
  %fill879 = linalg.fill ins(%cst : f16) outs(%init878 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm880 = linalg.matmul ins(%relu877, %mm872 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill879 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init881 = tensor.empty() : tensor<1024x1280xf16>
  %fill882 = linalg.fill ins(%cst : f16) outs(%init881 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm883 = linalg.matmul ins(%mm880, %w_o27 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill882 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty884 = tensor.empty() : tensor<1024x1280xf16>
  %add885 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm883, %add863 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty884 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init886 = tensor.empty() : tensor<1024x5120xf16>
  %fill887 = linalg.fill ins(%cst : f16) outs(%init886 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm888 = linalg.matmul ins(%add885, %w_ff27_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill887 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty889 = tensor.empty() : tensor<1024x5120xf16>
  %relu890 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm888 : tensor<1024x5120xf16>)
    outs(%empty889 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init891 = tensor.empty() : tensor<1024x1280xf16>
  %fill892 = linalg.fill ins(%cst : f16) outs(%init891 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm893 = linalg.matmul ins(%relu890, %w_ff27_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill892 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty894 = tensor.empty() : tensor<1024x1280xf16>
  %add895 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm893, %add885 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty894 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 28 ===

  // Q projection
  %init896 = tensor.empty() : tensor<1024x1280xf16>
  %fill897 = linalg.fill ins(%cst : f16) outs(%init896 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm898 = linalg.matmul ins(%add895, %w_q28 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill897 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init899 = tensor.empty() : tensor<1024x1280xf16>
  %fill900 = linalg.fill ins(%cst : f16) outs(%init899 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm901 = linalg.matmul ins(%add895, %w_k28 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill900 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init902 = tensor.empty() : tensor<1024x1280xf16>
  %fill903 = linalg.fill ins(%cst : f16) outs(%init902 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm904 = linalg.matmul ins(%add895, %w_v28 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill903 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init905 = tensor.empty() : tensor<1024x1024xf16>
  %fill906 = linalg.fill ins(%cst : f16) outs(%init905 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm907 = linalg.matmul ins(%mm898, %w_kt28 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill906 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty908 = tensor.empty() : tensor<1024x1024xf16>
  %relu909 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm907 : tensor<1024x1024xf16>)
    outs(%empty908 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init910 = tensor.empty() : tensor<1024x1280xf16>
  %fill911 = linalg.fill ins(%cst : f16) outs(%init910 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm912 = linalg.matmul ins(%relu909, %mm904 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill911 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init913 = tensor.empty() : tensor<1024x1280xf16>
  %fill914 = linalg.fill ins(%cst : f16) outs(%init913 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm915 = linalg.matmul ins(%mm912, %w_o28 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill914 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty916 = tensor.empty() : tensor<1024x1280xf16>
  %add917 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm915, %add895 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty916 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init918 = tensor.empty() : tensor<1024x5120xf16>
  %fill919 = linalg.fill ins(%cst : f16) outs(%init918 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm920 = linalg.matmul ins(%add917, %w_ff28_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill919 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty921 = tensor.empty() : tensor<1024x5120xf16>
  %relu922 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm920 : tensor<1024x5120xf16>)
    outs(%empty921 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init923 = tensor.empty() : tensor<1024x1280xf16>
  %fill924 = linalg.fill ins(%cst : f16) outs(%init923 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm925 = linalg.matmul ins(%relu922, %w_ff28_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill924 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty926 = tensor.empty() : tensor<1024x1280xf16>
  %add927 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm925, %add917 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty926 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 29 ===

  // Q projection
  %init928 = tensor.empty() : tensor<1024x1280xf16>
  %fill929 = linalg.fill ins(%cst : f16) outs(%init928 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm930 = linalg.matmul ins(%add927, %w_q29 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill929 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init931 = tensor.empty() : tensor<1024x1280xf16>
  %fill932 = linalg.fill ins(%cst : f16) outs(%init931 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm933 = linalg.matmul ins(%add927, %w_k29 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill932 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init934 = tensor.empty() : tensor<1024x1280xf16>
  %fill935 = linalg.fill ins(%cst : f16) outs(%init934 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm936 = linalg.matmul ins(%add927, %w_v29 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill935 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init937 = tensor.empty() : tensor<1024x1024xf16>
  %fill938 = linalg.fill ins(%cst : f16) outs(%init937 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm939 = linalg.matmul ins(%mm930, %w_kt29 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill938 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty940 = tensor.empty() : tensor<1024x1024xf16>
  %relu941 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm939 : tensor<1024x1024xf16>)
    outs(%empty940 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init942 = tensor.empty() : tensor<1024x1280xf16>
  %fill943 = linalg.fill ins(%cst : f16) outs(%init942 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm944 = linalg.matmul ins(%relu941, %mm936 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill943 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init945 = tensor.empty() : tensor<1024x1280xf16>
  %fill946 = linalg.fill ins(%cst : f16) outs(%init945 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm947 = linalg.matmul ins(%mm944, %w_o29 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill946 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty948 = tensor.empty() : tensor<1024x1280xf16>
  %add949 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm947, %add927 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty948 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init950 = tensor.empty() : tensor<1024x5120xf16>
  %fill951 = linalg.fill ins(%cst : f16) outs(%init950 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm952 = linalg.matmul ins(%add949, %w_ff29_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill951 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty953 = tensor.empty() : tensor<1024x5120xf16>
  %relu954 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm952 : tensor<1024x5120xf16>)
    outs(%empty953 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init955 = tensor.empty() : tensor<1024x1280xf16>
  %fill956 = linalg.fill ins(%cst : f16) outs(%init955 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm957 = linalg.matmul ins(%relu954, %w_ff29_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill956 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty958 = tensor.empty() : tensor<1024x1280xf16>
  %add959 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm957, %add949 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty958 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 30 ===

  // Q projection
  %init960 = tensor.empty() : tensor<1024x1280xf16>
  %fill961 = linalg.fill ins(%cst : f16) outs(%init960 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm962 = linalg.matmul ins(%add959, %w_q30 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill961 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init963 = tensor.empty() : tensor<1024x1280xf16>
  %fill964 = linalg.fill ins(%cst : f16) outs(%init963 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm965 = linalg.matmul ins(%add959, %w_k30 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill964 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init966 = tensor.empty() : tensor<1024x1280xf16>
  %fill967 = linalg.fill ins(%cst : f16) outs(%init966 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm968 = linalg.matmul ins(%add959, %w_v30 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill967 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init969 = tensor.empty() : tensor<1024x1024xf16>
  %fill970 = linalg.fill ins(%cst : f16) outs(%init969 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm971 = linalg.matmul ins(%mm962, %w_kt30 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill970 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty972 = tensor.empty() : tensor<1024x1024xf16>
  %relu973 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm971 : tensor<1024x1024xf16>)
    outs(%empty972 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init974 = tensor.empty() : tensor<1024x1280xf16>
  %fill975 = linalg.fill ins(%cst : f16) outs(%init974 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm976 = linalg.matmul ins(%relu973, %mm968 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill975 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init977 = tensor.empty() : tensor<1024x1280xf16>
  %fill978 = linalg.fill ins(%cst : f16) outs(%init977 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm979 = linalg.matmul ins(%mm976, %w_o30 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill978 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty980 = tensor.empty() : tensor<1024x1280xf16>
  %add981 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm979, %add959 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty980 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init982 = tensor.empty() : tensor<1024x5120xf16>
  %fill983 = linalg.fill ins(%cst : f16) outs(%init982 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm984 = linalg.matmul ins(%add981, %w_ff30_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill983 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty985 = tensor.empty() : tensor<1024x5120xf16>
  %relu986 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm984 : tensor<1024x5120xf16>)
    outs(%empty985 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init987 = tensor.empty() : tensor<1024x1280xf16>
  %fill988 = linalg.fill ins(%cst : f16) outs(%init987 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm989 = linalg.matmul ins(%relu986, %w_ff30_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill988 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty990 = tensor.empty() : tensor<1024x1280xf16>
  %add991 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm989, %add981 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty990 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 31 ===

  // Q projection
  %init992 = tensor.empty() : tensor<1024x1280xf16>
  %fill993 = linalg.fill ins(%cst : f16) outs(%init992 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm994 = linalg.matmul ins(%add991, %w_q31 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill993 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init995 = tensor.empty() : tensor<1024x1280xf16>
  %fill996 = linalg.fill ins(%cst : f16) outs(%init995 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm997 = linalg.matmul ins(%add991, %w_k31 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill996 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init998 = tensor.empty() : tensor<1024x1280xf16>
  %fill999 = linalg.fill ins(%cst : f16) outs(%init998 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1000 = linalg.matmul ins(%add991, %w_v31 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill999 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init1001 = tensor.empty() : tensor<1024x1024xf16>
  %fill1002 = linalg.fill ins(%cst : f16) outs(%init1001 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm1003 = linalg.matmul ins(%mm994, %w_kt31 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill1002 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty1004 = tensor.empty() : tensor<1024x1024xf16>
  %relu1005 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1003 : tensor<1024x1024xf16>)
    outs(%empty1004 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init1006 = tensor.empty() : tensor<1024x1280xf16>
  %fill1007 = linalg.fill ins(%cst : f16) outs(%init1006 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1008 = linalg.matmul ins(%relu1005, %mm1000 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill1007 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init1009 = tensor.empty() : tensor<1024x1280xf16>
  %fill1010 = linalg.fill ins(%cst : f16) outs(%init1009 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1011 = linalg.matmul ins(%mm1008, %w_o31 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1010 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty1012 = tensor.empty() : tensor<1024x1280xf16>
  %add1013 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1011, %add991 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty1012 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init1014 = tensor.empty() : tensor<1024x5120xf16>
  %fill1015 = linalg.fill ins(%cst : f16) outs(%init1014 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm1016 = linalg.matmul ins(%add1013, %w_ff31_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill1015 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty1017 = tensor.empty() : tensor<1024x5120xf16>
  %relu1018 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1016 : tensor<1024x5120xf16>)
    outs(%empty1017 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init1019 = tensor.empty() : tensor<1024x1280xf16>
  %fill1020 = linalg.fill ins(%cst : f16) outs(%init1019 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1021 = linalg.matmul ins(%relu1018, %w_ff31_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill1020 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty1022 = tensor.empty() : tensor<1024x1280xf16>
  %add1023 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1021, %add1013 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty1022 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 32 ===

  // Q projection
  %init1024 = tensor.empty() : tensor<1024x1280xf16>
  %fill1025 = linalg.fill ins(%cst : f16) outs(%init1024 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1026 = linalg.matmul ins(%add1023, %w_q32 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1025 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init1027 = tensor.empty() : tensor<1024x1280xf16>
  %fill1028 = linalg.fill ins(%cst : f16) outs(%init1027 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1029 = linalg.matmul ins(%add1023, %w_k32 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1028 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init1030 = tensor.empty() : tensor<1024x1280xf16>
  %fill1031 = linalg.fill ins(%cst : f16) outs(%init1030 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1032 = linalg.matmul ins(%add1023, %w_v32 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1031 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init1033 = tensor.empty() : tensor<1024x1024xf16>
  %fill1034 = linalg.fill ins(%cst : f16) outs(%init1033 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm1035 = linalg.matmul ins(%mm1026, %w_kt32 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill1034 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty1036 = tensor.empty() : tensor<1024x1024xf16>
  %relu1037 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1035 : tensor<1024x1024xf16>)
    outs(%empty1036 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init1038 = tensor.empty() : tensor<1024x1280xf16>
  %fill1039 = linalg.fill ins(%cst : f16) outs(%init1038 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1040 = linalg.matmul ins(%relu1037, %mm1032 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill1039 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init1041 = tensor.empty() : tensor<1024x1280xf16>
  %fill1042 = linalg.fill ins(%cst : f16) outs(%init1041 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1043 = linalg.matmul ins(%mm1040, %w_o32 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1042 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty1044 = tensor.empty() : tensor<1024x1280xf16>
  %add1045 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1043, %add1023 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty1044 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init1046 = tensor.empty() : tensor<1024x5120xf16>
  %fill1047 = linalg.fill ins(%cst : f16) outs(%init1046 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm1048 = linalg.matmul ins(%add1045, %w_ff32_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill1047 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty1049 = tensor.empty() : tensor<1024x5120xf16>
  %relu1050 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1048 : tensor<1024x5120xf16>)
    outs(%empty1049 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init1051 = tensor.empty() : tensor<1024x1280xf16>
  %fill1052 = linalg.fill ins(%cst : f16) outs(%init1051 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1053 = linalg.matmul ins(%relu1050, %w_ff32_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill1052 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty1054 = tensor.empty() : tensor<1024x1280xf16>
  %add1055 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1053, %add1045 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty1054 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 33 ===

  // Q projection
  %init1056 = tensor.empty() : tensor<1024x1280xf16>
  %fill1057 = linalg.fill ins(%cst : f16) outs(%init1056 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1058 = linalg.matmul ins(%add1055, %w_q33 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1057 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init1059 = tensor.empty() : tensor<1024x1280xf16>
  %fill1060 = linalg.fill ins(%cst : f16) outs(%init1059 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1061 = linalg.matmul ins(%add1055, %w_k33 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1060 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init1062 = tensor.empty() : tensor<1024x1280xf16>
  %fill1063 = linalg.fill ins(%cst : f16) outs(%init1062 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1064 = linalg.matmul ins(%add1055, %w_v33 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1063 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init1065 = tensor.empty() : tensor<1024x1024xf16>
  %fill1066 = linalg.fill ins(%cst : f16) outs(%init1065 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm1067 = linalg.matmul ins(%mm1058, %w_kt33 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill1066 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty1068 = tensor.empty() : tensor<1024x1024xf16>
  %relu1069 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1067 : tensor<1024x1024xf16>)
    outs(%empty1068 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init1070 = tensor.empty() : tensor<1024x1280xf16>
  %fill1071 = linalg.fill ins(%cst : f16) outs(%init1070 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1072 = linalg.matmul ins(%relu1069, %mm1064 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill1071 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init1073 = tensor.empty() : tensor<1024x1280xf16>
  %fill1074 = linalg.fill ins(%cst : f16) outs(%init1073 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1075 = linalg.matmul ins(%mm1072, %w_o33 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1074 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty1076 = tensor.empty() : tensor<1024x1280xf16>
  %add1077 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1075, %add1055 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty1076 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init1078 = tensor.empty() : tensor<1024x5120xf16>
  %fill1079 = linalg.fill ins(%cst : f16) outs(%init1078 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm1080 = linalg.matmul ins(%add1077, %w_ff33_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill1079 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty1081 = tensor.empty() : tensor<1024x5120xf16>
  %relu1082 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1080 : tensor<1024x5120xf16>)
    outs(%empty1081 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init1083 = tensor.empty() : tensor<1024x1280xf16>
  %fill1084 = linalg.fill ins(%cst : f16) outs(%init1083 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1085 = linalg.matmul ins(%relu1082, %w_ff33_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill1084 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty1086 = tensor.empty() : tensor<1024x1280xf16>
  %add1087 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1085, %add1077 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty1086 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 34 ===

  // Q projection
  %init1088 = tensor.empty() : tensor<1024x1280xf16>
  %fill1089 = linalg.fill ins(%cst : f16) outs(%init1088 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1090 = linalg.matmul ins(%add1087, %w_q34 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1089 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init1091 = tensor.empty() : tensor<1024x1280xf16>
  %fill1092 = linalg.fill ins(%cst : f16) outs(%init1091 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1093 = linalg.matmul ins(%add1087, %w_k34 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1092 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init1094 = tensor.empty() : tensor<1024x1280xf16>
  %fill1095 = linalg.fill ins(%cst : f16) outs(%init1094 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1096 = linalg.matmul ins(%add1087, %w_v34 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1095 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init1097 = tensor.empty() : tensor<1024x1024xf16>
  %fill1098 = linalg.fill ins(%cst : f16) outs(%init1097 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm1099 = linalg.matmul ins(%mm1090, %w_kt34 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill1098 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty1100 = tensor.empty() : tensor<1024x1024xf16>
  %relu1101 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1099 : tensor<1024x1024xf16>)
    outs(%empty1100 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init1102 = tensor.empty() : tensor<1024x1280xf16>
  %fill1103 = linalg.fill ins(%cst : f16) outs(%init1102 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1104 = linalg.matmul ins(%relu1101, %mm1096 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill1103 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init1105 = tensor.empty() : tensor<1024x1280xf16>
  %fill1106 = linalg.fill ins(%cst : f16) outs(%init1105 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1107 = linalg.matmul ins(%mm1104, %w_o34 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1106 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty1108 = tensor.empty() : tensor<1024x1280xf16>
  %add1109 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1107, %add1087 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty1108 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init1110 = tensor.empty() : tensor<1024x5120xf16>
  %fill1111 = linalg.fill ins(%cst : f16) outs(%init1110 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm1112 = linalg.matmul ins(%add1109, %w_ff34_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill1111 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty1113 = tensor.empty() : tensor<1024x5120xf16>
  %relu1114 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1112 : tensor<1024x5120xf16>)
    outs(%empty1113 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init1115 = tensor.empty() : tensor<1024x1280xf16>
  %fill1116 = linalg.fill ins(%cst : f16) outs(%init1115 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1117 = linalg.matmul ins(%relu1114, %w_ff34_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill1116 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty1118 = tensor.empty() : tensor<1024x1280xf16>
  %add1119 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1117, %add1109 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty1118 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // === Transformer Layer 35 ===

  // Q projection
  %init1120 = tensor.empty() : tensor<1024x1280xf16>
  %fill1121 = linalg.fill ins(%cst : f16) outs(%init1120 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1122 = linalg.matmul ins(%add1119, %w_q35 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1121 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // K projection
  %init1123 = tensor.empty() : tensor<1024x1280xf16>
  %fill1124 = linalg.fill ins(%cst : f16) outs(%init1123 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1125 = linalg.matmul ins(%add1119, %w_k35 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1124 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // V projection
  %init1126 = tensor.empty() : tensor<1024x1280xf16>
  %fill1127 = linalg.fill ins(%cst : f16) outs(%init1126 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1128 = linalg.matmul ins(%add1119, %w_v35 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1127 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>

  // Attention scores: Q x K_transposed
  %init1129 = tensor.empty() : tensor<1024x1024xf16>
  %fill1130 = linalg.fill ins(%cst : f16) outs(%init1129 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm1131 = linalg.matmul ins(%mm1122, %w_kt35 : tensor<1024x1280xf16>, tensor<1280x1024xf16>)
                          outs(%fill1130 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Softmax approximation (relu)
  %empty1132 = tensor.empty() : tensor<1024x1024xf16>
  %relu1133 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1131 : tensor<1024x1024xf16>)
    outs(%empty1132 : tensor<1024x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x1024xf16>

  // Attention output: scores x V
  %init1134 = tensor.empty() : tensor<1024x1280xf16>
  %fill1135 = linalg.fill ins(%cst : f16) outs(%init1134 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1136 = linalg.matmul ins(%relu1133, %mm1128 : tensor<1024x1024xf16>, tensor<1024x1280xf16>)
                          outs(%fill1135 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Output projection
  %init1137 = tensor.empty() : tensor<1024x1280xf16>
  %fill1138 = linalg.fill ins(%cst : f16) outs(%init1137 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1139 = linalg.matmul ins(%mm1136, %w_o35 : tensor<1024x1280xf16>, tensor<1280x1280xf16>)
                          outs(%fill1138 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // Attention residual add
  %empty1140 = tensor.empty() : tensor<1024x1280xf16>
  %add1141 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1139, %add1119 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty1140 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  // FFN: up projection
  %init1142 = tensor.empty() : tensor<1024x5120xf16>
  %fill1143 = linalg.fill ins(%cst : f16) outs(%init1142 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  %mm1144 = linalg.matmul ins(%add1141, %w_ff35_up : tensor<1024x1280xf16>, tensor<1280x5120xf16>)
                          outs(%fill1143 : tensor<1024x5120xf16>) -> tensor<1024x5120xf16>
  // FFN ReLU
  %empty1145 = tensor.empty() : tensor<1024x5120xf16>
  %relu1146 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1144 : tensor<1024x5120xf16>)
    outs(%empty1145 : tensor<1024x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x5120xf16>
  // FFN: down projection
  %init1147 = tensor.empty() : tensor<1024x1280xf16>
  %fill1148 = linalg.fill ins(%cst : f16) outs(%init1147 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  %mm1149 = linalg.matmul ins(%relu1146, %w_ff35_down : tensor<1024x5120xf16>, tensor<5120x1280xf16>)
                          outs(%fill1148 : tensor<1024x1280xf16>) -> tensor<1024x1280xf16>
  // FFN residual add
  %empty1150 = tensor.empty() : tensor<1024x1280xf16>
  %add1151 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1149, %add1141 : tensor<1024x1280xf16>, tensor<1024x1280xf16>)
    outs(%empty1150 : tensor<1024x1280xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1280xf16>

  return %add1151 : tensor<1024x1280xf16>
}
