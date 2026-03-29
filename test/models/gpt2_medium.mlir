func.func @gpt2_medium(
    %input: tensor<1024x1024xf16>,
    %w_q0: tensor<1024x1024xf16>,
    %w_k0: tensor<1024x1024xf16>,
    %w_v0: tensor<1024x1024xf16>,
    %w_kt0: tensor<1024x1024xf16>,
    %w_o0: tensor<1024x1024xf16>,
    %w_ff0_up: tensor<1024x4096xf16>,
    %w_ff0_down: tensor<4096x1024xf16>,
    %w_q1: tensor<1024x1024xf16>,
    %w_k1: tensor<1024x1024xf16>,
    %w_v1: tensor<1024x1024xf16>,
    %w_kt1: tensor<1024x1024xf16>,
    %w_o1: tensor<1024x1024xf16>,
    %w_ff1_up: tensor<1024x4096xf16>,
    %w_ff1_down: tensor<4096x1024xf16>,
    %w_q2: tensor<1024x1024xf16>,
    %w_k2: tensor<1024x1024xf16>,
    %w_v2: tensor<1024x1024xf16>,
    %w_kt2: tensor<1024x1024xf16>,
    %w_o2: tensor<1024x1024xf16>,
    %w_ff2_up: tensor<1024x4096xf16>,
    %w_ff2_down: tensor<4096x1024xf16>,
    %w_q3: tensor<1024x1024xf16>,
    %w_k3: tensor<1024x1024xf16>,
    %w_v3: tensor<1024x1024xf16>,
    %w_kt3: tensor<1024x1024xf16>,
    %w_o3: tensor<1024x1024xf16>,
    %w_ff3_up: tensor<1024x4096xf16>,
    %w_ff3_down: tensor<4096x1024xf16>,
    %w_q4: tensor<1024x1024xf16>,
    %w_k4: tensor<1024x1024xf16>,
    %w_v4: tensor<1024x1024xf16>,
    %w_kt4: tensor<1024x1024xf16>,
    %w_o4: tensor<1024x1024xf16>,
    %w_ff4_up: tensor<1024x4096xf16>,
    %w_ff4_down: tensor<4096x1024xf16>,
    %w_q5: tensor<1024x1024xf16>,
    %w_k5: tensor<1024x1024xf16>,
    %w_v5: tensor<1024x1024xf16>,
    %w_kt5: tensor<1024x1024xf16>,
    %w_o5: tensor<1024x1024xf16>,
    %w_ff5_up: tensor<1024x4096xf16>,
    %w_ff5_down: tensor<4096x1024xf16>,
    %w_q6: tensor<1024x1024xf16>,
    %w_k6: tensor<1024x1024xf16>,
    %w_v6: tensor<1024x1024xf16>,
    %w_kt6: tensor<1024x1024xf16>,
    %w_o6: tensor<1024x1024xf16>,
    %w_ff6_up: tensor<1024x4096xf16>,
    %w_ff6_down: tensor<4096x1024xf16>,
    %w_q7: tensor<1024x1024xf16>,
    %w_k7: tensor<1024x1024xf16>,
    %w_v7: tensor<1024x1024xf16>,
    %w_kt7: tensor<1024x1024xf16>,
    %w_o7: tensor<1024x1024xf16>,
    %w_ff7_up: tensor<1024x4096xf16>,
    %w_ff7_down: tensor<4096x1024xf16>,
    %w_q8: tensor<1024x1024xf16>,
    %w_k8: tensor<1024x1024xf16>,
    %w_v8: tensor<1024x1024xf16>,
    %w_kt8: tensor<1024x1024xf16>,
    %w_o8: tensor<1024x1024xf16>,
    %w_ff8_up: tensor<1024x4096xf16>,
    %w_ff8_down: tensor<4096x1024xf16>,
    %w_q9: tensor<1024x1024xf16>,
    %w_k9: tensor<1024x1024xf16>,
    %w_v9: tensor<1024x1024xf16>,
    %w_kt9: tensor<1024x1024xf16>,
    %w_o9: tensor<1024x1024xf16>,
    %w_ff9_up: tensor<1024x4096xf16>,
    %w_ff9_down: tensor<4096x1024xf16>,
    %w_q10: tensor<1024x1024xf16>,
    %w_k10: tensor<1024x1024xf16>,
    %w_v10: tensor<1024x1024xf16>,
    %w_kt10: tensor<1024x1024xf16>,
    %w_o10: tensor<1024x1024xf16>,
    %w_ff10_up: tensor<1024x4096xf16>,
    %w_ff10_down: tensor<4096x1024xf16>,
    %w_q11: tensor<1024x1024xf16>,
    %w_k11: tensor<1024x1024xf16>,
    %w_v11: tensor<1024x1024xf16>,
    %w_kt11: tensor<1024x1024xf16>,
    %w_o11: tensor<1024x1024xf16>,
    %w_ff11_up: tensor<1024x4096xf16>,
    %w_ff11_down: tensor<4096x1024xf16>,
    %w_q12: tensor<1024x1024xf16>,
    %w_k12: tensor<1024x1024xf16>,
    %w_v12: tensor<1024x1024xf16>,
    %w_kt12: tensor<1024x1024xf16>,
    %w_o12: tensor<1024x1024xf16>,
    %w_ff12_up: tensor<1024x4096xf16>,
    %w_ff12_down: tensor<4096x1024xf16>,
    %w_q13: tensor<1024x1024xf16>,
    %w_k13: tensor<1024x1024xf16>,
    %w_v13: tensor<1024x1024xf16>,
    %w_kt13: tensor<1024x1024xf16>,
    %w_o13: tensor<1024x1024xf16>,
    %w_ff13_up: tensor<1024x4096xf16>,
    %w_ff13_down: tensor<4096x1024xf16>,
    %w_q14: tensor<1024x1024xf16>,
    %w_k14: tensor<1024x1024xf16>,
    %w_v14: tensor<1024x1024xf16>,
    %w_kt14: tensor<1024x1024xf16>,
    %w_o14: tensor<1024x1024xf16>,
    %w_ff14_up: tensor<1024x4096xf16>,
    %w_ff14_down: tensor<4096x1024xf16>,
    %w_q15: tensor<1024x1024xf16>,
    %w_k15: tensor<1024x1024xf16>,
    %w_v15: tensor<1024x1024xf16>,
    %w_kt15: tensor<1024x1024xf16>,
    %w_o15: tensor<1024x1024xf16>,
    %w_ff15_up: tensor<1024x4096xf16>,
    %w_ff15_down: tensor<4096x1024xf16>,
    %w_q16: tensor<1024x1024xf16>,
    %w_k16: tensor<1024x1024xf16>,
    %w_v16: tensor<1024x1024xf16>,
    %w_kt16: tensor<1024x1024xf16>,
    %w_o16: tensor<1024x1024xf16>,
    %w_ff16_up: tensor<1024x4096xf16>,
    %w_ff16_down: tensor<4096x1024xf16>,
    %w_q17: tensor<1024x1024xf16>,
    %w_k17: tensor<1024x1024xf16>,
    %w_v17: tensor<1024x1024xf16>,
    %w_kt17: tensor<1024x1024xf16>,
    %w_o17: tensor<1024x1024xf16>,
    %w_ff17_up: tensor<1024x4096xf16>,
    %w_ff17_down: tensor<4096x1024xf16>,
    %w_q18: tensor<1024x1024xf16>,
    %w_k18: tensor<1024x1024xf16>,
    %w_v18: tensor<1024x1024xf16>,
    %w_kt18: tensor<1024x1024xf16>,
    %w_o18: tensor<1024x1024xf16>,
    %w_ff18_up: tensor<1024x4096xf16>,
    %w_ff18_down: tensor<4096x1024xf16>,
    %w_q19: tensor<1024x1024xf16>,
    %w_k19: tensor<1024x1024xf16>,
    %w_v19: tensor<1024x1024xf16>,
    %w_kt19: tensor<1024x1024xf16>,
    %w_o19: tensor<1024x1024xf16>,
    %w_ff19_up: tensor<1024x4096xf16>,
    %w_ff19_down: tensor<4096x1024xf16>,
    %w_q20: tensor<1024x1024xf16>,
    %w_k20: tensor<1024x1024xf16>,
    %w_v20: tensor<1024x1024xf16>,
    %w_kt20: tensor<1024x1024xf16>,
    %w_o20: tensor<1024x1024xf16>,
    %w_ff20_up: tensor<1024x4096xf16>,
    %w_ff20_down: tensor<4096x1024xf16>,
    %w_q21: tensor<1024x1024xf16>,
    %w_k21: tensor<1024x1024xf16>,
    %w_v21: tensor<1024x1024xf16>,
    %w_kt21: tensor<1024x1024xf16>,
    %w_o21: tensor<1024x1024xf16>,
    %w_ff21_up: tensor<1024x4096xf16>,
    %w_ff21_down: tensor<4096x1024xf16>,
    %w_q22: tensor<1024x1024xf16>,
    %w_k22: tensor<1024x1024xf16>,
    %w_v22: tensor<1024x1024xf16>,
    %w_kt22: tensor<1024x1024xf16>,
    %w_o22: tensor<1024x1024xf16>,
    %w_ff22_up: tensor<1024x4096xf16>,
    %w_ff22_down: tensor<4096x1024xf16>,
    %w_q23: tensor<1024x1024xf16>,
    %w_k23: tensor<1024x1024xf16>,
    %w_v23: tensor<1024x1024xf16>,
    %w_kt23: tensor<1024x1024xf16>,
    %w_o23: tensor<1024x1024xf16>,
    %w_ff23_up: tensor<1024x4096xf16>,
    %w_ff23_down: tensor<4096x1024xf16>) -> tensor<1024x1024xf16> {
  %cst = arith.constant 0.0 : f16

  // === Transformer Layer 0 ===

  // Q projection
  %init0 = tensor.empty() : tensor<1024x1024xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init0 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm2 = linalg.matmul ins(%input, %w_q0 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill1 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init3 = tensor.empty() : tensor<1024x1024xf16>
  %fill4 = linalg.fill ins(%cst : f16) outs(%init3 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm5 = linalg.matmul ins(%input, %w_k0 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill4 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init6 = tensor.empty() : tensor<1024x1024xf16>
  %fill7 = linalg.fill ins(%cst : f16) outs(%init6 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm8 = linalg.matmul ins(%input, %w_v0 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill7 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init9 = tensor.empty() : tensor<1024x1024xf16>
  %fill10 = linalg.fill ins(%cst : f16) outs(%init9 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm11 = linalg.matmul ins(%mm2, %w_kt0 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init14 = tensor.empty() : tensor<1024x1024xf16>
  %fill15 = linalg.fill ins(%cst : f16) outs(%init14 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm16 = linalg.matmul ins(%relu13, %mm8 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill15 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init17 = tensor.empty() : tensor<1024x1024xf16>
  %fill18 = linalg.fill ins(%cst : f16) outs(%init17 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm19 = linalg.matmul ins(%mm16, %w_o0 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill18 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty20 = tensor.empty() : tensor<1024x1024xf16>
  %add21 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm19, %input : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty20 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init22 = tensor.empty() : tensor<1024x4096xf16>
  %fill23 = linalg.fill ins(%cst : f16) outs(%init22 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm24 = linalg.matmul ins(%add21, %w_ff0_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill23 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty25 = tensor.empty() : tensor<1024x4096xf16>
  %relu26 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm24 : tensor<1024x4096xf16>)
    outs(%empty25 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init27 = tensor.empty() : tensor<1024x1024xf16>
  %fill28 = linalg.fill ins(%cst : f16) outs(%init27 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm29 = linalg.matmul ins(%relu26, %w_ff0_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill28 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty30 = tensor.empty() : tensor<1024x1024xf16>
  %add31 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm29, %add21 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty30 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 1 ===

  // Q projection
  %init32 = tensor.empty() : tensor<1024x1024xf16>
  %fill33 = linalg.fill ins(%cst : f16) outs(%init32 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm34 = linalg.matmul ins(%add31, %w_q1 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill33 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init35 = tensor.empty() : tensor<1024x1024xf16>
  %fill36 = linalg.fill ins(%cst : f16) outs(%init35 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm37 = linalg.matmul ins(%add31, %w_k1 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill36 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init38 = tensor.empty() : tensor<1024x1024xf16>
  %fill39 = linalg.fill ins(%cst : f16) outs(%init38 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm40 = linalg.matmul ins(%add31, %w_v1 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill39 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init41 = tensor.empty() : tensor<1024x1024xf16>
  %fill42 = linalg.fill ins(%cst : f16) outs(%init41 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm43 = linalg.matmul ins(%mm34, %w_kt1 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init46 = tensor.empty() : tensor<1024x1024xf16>
  %fill47 = linalg.fill ins(%cst : f16) outs(%init46 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm48 = linalg.matmul ins(%relu45, %mm40 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill47 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init49 = tensor.empty() : tensor<1024x1024xf16>
  %fill50 = linalg.fill ins(%cst : f16) outs(%init49 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm51 = linalg.matmul ins(%mm48, %w_o1 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill50 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty52 = tensor.empty() : tensor<1024x1024xf16>
  %add53 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm51, %add31 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty52 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init54 = tensor.empty() : tensor<1024x4096xf16>
  %fill55 = linalg.fill ins(%cst : f16) outs(%init54 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm56 = linalg.matmul ins(%add53, %w_ff1_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill55 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty57 = tensor.empty() : tensor<1024x4096xf16>
  %relu58 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm56 : tensor<1024x4096xf16>)
    outs(%empty57 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init59 = tensor.empty() : tensor<1024x1024xf16>
  %fill60 = linalg.fill ins(%cst : f16) outs(%init59 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm61 = linalg.matmul ins(%relu58, %w_ff1_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill60 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty62 = tensor.empty() : tensor<1024x1024xf16>
  %add63 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm61, %add53 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty62 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 2 ===

  // Q projection
  %init64 = tensor.empty() : tensor<1024x1024xf16>
  %fill65 = linalg.fill ins(%cst : f16) outs(%init64 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm66 = linalg.matmul ins(%add63, %w_q2 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill65 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init67 = tensor.empty() : tensor<1024x1024xf16>
  %fill68 = linalg.fill ins(%cst : f16) outs(%init67 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm69 = linalg.matmul ins(%add63, %w_k2 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill68 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init70 = tensor.empty() : tensor<1024x1024xf16>
  %fill71 = linalg.fill ins(%cst : f16) outs(%init70 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm72 = linalg.matmul ins(%add63, %w_v2 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill71 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init73 = tensor.empty() : tensor<1024x1024xf16>
  %fill74 = linalg.fill ins(%cst : f16) outs(%init73 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm75 = linalg.matmul ins(%mm66, %w_kt2 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init78 = tensor.empty() : tensor<1024x1024xf16>
  %fill79 = linalg.fill ins(%cst : f16) outs(%init78 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm80 = linalg.matmul ins(%relu77, %mm72 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill79 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init81 = tensor.empty() : tensor<1024x1024xf16>
  %fill82 = linalg.fill ins(%cst : f16) outs(%init81 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm83 = linalg.matmul ins(%mm80, %w_o2 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill82 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty84 = tensor.empty() : tensor<1024x1024xf16>
  %add85 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm83, %add63 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty84 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init86 = tensor.empty() : tensor<1024x4096xf16>
  %fill87 = linalg.fill ins(%cst : f16) outs(%init86 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm88 = linalg.matmul ins(%add85, %w_ff2_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill87 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty89 = tensor.empty() : tensor<1024x4096xf16>
  %relu90 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm88 : tensor<1024x4096xf16>)
    outs(%empty89 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init91 = tensor.empty() : tensor<1024x1024xf16>
  %fill92 = linalg.fill ins(%cst : f16) outs(%init91 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm93 = linalg.matmul ins(%relu90, %w_ff2_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill92 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty94 = tensor.empty() : tensor<1024x1024xf16>
  %add95 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm93, %add85 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty94 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 3 ===

  // Q projection
  %init96 = tensor.empty() : tensor<1024x1024xf16>
  %fill97 = linalg.fill ins(%cst : f16) outs(%init96 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm98 = linalg.matmul ins(%add95, %w_q3 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill97 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init99 = tensor.empty() : tensor<1024x1024xf16>
  %fill100 = linalg.fill ins(%cst : f16) outs(%init99 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm101 = linalg.matmul ins(%add95, %w_k3 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill100 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init102 = tensor.empty() : tensor<1024x1024xf16>
  %fill103 = linalg.fill ins(%cst : f16) outs(%init102 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm104 = linalg.matmul ins(%add95, %w_v3 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill103 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init105 = tensor.empty() : tensor<1024x1024xf16>
  %fill106 = linalg.fill ins(%cst : f16) outs(%init105 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm107 = linalg.matmul ins(%mm98, %w_kt3 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init110 = tensor.empty() : tensor<1024x1024xf16>
  %fill111 = linalg.fill ins(%cst : f16) outs(%init110 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm112 = linalg.matmul ins(%relu109, %mm104 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill111 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init113 = tensor.empty() : tensor<1024x1024xf16>
  %fill114 = linalg.fill ins(%cst : f16) outs(%init113 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm115 = linalg.matmul ins(%mm112, %w_o3 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill114 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty116 = tensor.empty() : tensor<1024x1024xf16>
  %add117 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm115, %add95 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty116 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init118 = tensor.empty() : tensor<1024x4096xf16>
  %fill119 = linalg.fill ins(%cst : f16) outs(%init118 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm120 = linalg.matmul ins(%add117, %w_ff3_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill119 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty121 = tensor.empty() : tensor<1024x4096xf16>
  %relu122 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm120 : tensor<1024x4096xf16>)
    outs(%empty121 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init123 = tensor.empty() : tensor<1024x1024xf16>
  %fill124 = linalg.fill ins(%cst : f16) outs(%init123 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm125 = linalg.matmul ins(%relu122, %w_ff3_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill124 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty126 = tensor.empty() : tensor<1024x1024xf16>
  %add127 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm125, %add117 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty126 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 4 ===

  // Q projection
  %init128 = tensor.empty() : tensor<1024x1024xf16>
  %fill129 = linalg.fill ins(%cst : f16) outs(%init128 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm130 = linalg.matmul ins(%add127, %w_q4 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill129 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init131 = tensor.empty() : tensor<1024x1024xf16>
  %fill132 = linalg.fill ins(%cst : f16) outs(%init131 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm133 = linalg.matmul ins(%add127, %w_k4 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill132 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init134 = tensor.empty() : tensor<1024x1024xf16>
  %fill135 = linalg.fill ins(%cst : f16) outs(%init134 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm136 = linalg.matmul ins(%add127, %w_v4 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill135 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init137 = tensor.empty() : tensor<1024x1024xf16>
  %fill138 = linalg.fill ins(%cst : f16) outs(%init137 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm139 = linalg.matmul ins(%mm130, %w_kt4 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init142 = tensor.empty() : tensor<1024x1024xf16>
  %fill143 = linalg.fill ins(%cst : f16) outs(%init142 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm144 = linalg.matmul ins(%relu141, %mm136 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill143 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init145 = tensor.empty() : tensor<1024x1024xf16>
  %fill146 = linalg.fill ins(%cst : f16) outs(%init145 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm147 = linalg.matmul ins(%mm144, %w_o4 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill146 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty148 = tensor.empty() : tensor<1024x1024xf16>
  %add149 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm147, %add127 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty148 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init150 = tensor.empty() : tensor<1024x4096xf16>
  %fill151 = linalg.fill ins(%cst : f16) outs(%init150 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm152 = linalg.matmul ins(%add149, %w_ff4_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill151 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty153 = tensor.empty() : tensor<1024x4096xf16>
  %relu154 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm152 : tensor<1024x4096xf16>)
    outs(%empty153 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init155 = tensor.empty() : tensor<1024x1024xf16>
  %fill156 = linalg.fill ins(%cst : f16) outs(%init155 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm157 = linalg.matmul ins(%relu154, %w_ff4_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill156 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty158 = tensor.empty() : tensor<1024x1024xf16>
  %add159 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm157, %add149 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty158 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 5 ===

  // Q projection
  %init160 = tensor.empty() : tensor<1024x1024xf16>
  %fill161 = linalg.fill ins(%cst : f16) outs(%init160 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm162 = linalg.matmul ins(%add159, %w_q5 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill161 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init163 = tensor.empty() : tensor<1024x1024xf16>
  %fill164 = linalg.fill ins(%cst : f16) outs(%init163 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm165 = linalg.matmul ins(%add159, %w_k5 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill164 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init166 = tensor.empty() : tensor<1024x1024xf16>
  %fill167 = linalg.fill ins(%cst : f16) outs(%init166 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm168 = linalg.matmul ins(%add159, %w_v5 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill167 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init169 = tensor.empty() : tensor<1024x1024xf16>
  %fill170 = linalg.fill ins(%cst : f16) outs(%init169 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm171 = linalg.matmul ins(%mm162, %w_kt5 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init174 = tensor.empty() : tensor<1024x1024xf16>
  %fill175 = linalg.fill ins(%cst : f16) outs(%init174 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm176 = linalg.matmul ins(%relu173, %mm168 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill175 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init177 = tensor.empty() : tensor<1024x1024xf16>
  %fill178 = linalg.fill ins(%cst : f16) outs(%init177 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm179 = linalg.matmul ins(%mm176, %w_o5 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill178 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty180 = tensor.empty() : tensor<1024x1024xf16>
  %add181 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm179, %add159 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty180 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init182 = tensor.empty() : tensor<1024x4096xf16>
  %fill183 = linalg.fill ins(%cst : f16) outs(%init182 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm184 = linalg.matmul ins(%add181, %w_ff5_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill183 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty185 = tensor.empty() : tensor<1024x4096xf16>
  %relu186 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm184 : tensor<1024x4096xf16>)
    outs(%empty185 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init187 = tensor.empty() : tensor<1024x1024xf16>
  %fill188 = linalg.fill ins(%cst : f16) outs(%init187 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm189 = linalg.matmul ins(%relu186, %w_ff5_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill188 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty190 = tensor.empty() : tensor<1024x1024xf16>
  %add191 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm189, %add181 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty190 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 6 ===

  // Q projection
  %init192 = tensor.empty() : tensor<1024x1024xf16>
  %fill193 = linalg.fill ins(%cst : f16) outs(%init192 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm194 = linalg.matmul ins(%add191, %w_q6 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill193 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init195 = tensor.empty() : tensor<1024x1024xf16>
  %fill196 = linalg.fill ins(%cst : f16) outs(%init195 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm197 = linalg.matmul ins(%add191, %w_k6 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill196 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init198 = tensor.empty() : tensor<1024x1024xf16>
  %fill199 = linalg.fill ins(%cst : f16) outs(%init198 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm200 = linalg.matmul ins(%add191, %w_v6 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill199 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init201 = tensor.empty() : tensor<1024x1024xf16>
  %fill202 = linalg.fill ins(%cst : f16) outs(%init201 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm203 = linalg.matmul ins(%mm194, %w_kt6 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init206 = tensor.empty() : tensor<1024x1024xf16>
  %fill207 = linalg.fill ins(%cst : f16) outs(%init206 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm208 = linalg.matmul ins(%relu205, %mm200 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill207 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init209 = tensor.empty() : tensor<1024x1024xf16>
  %fill210 = linalg.fill ins(%cst : f16) outs(%init209 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm211 = linalg.matmul ins(%mm208, %w_o6 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill210 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty212 = tensor.empty() : tensor<1024x1024xf16>
  %add213 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm211, %add191 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty212 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init214 = tensor.empty() : tensor<1024x4096xf16>
  %fill215 = linalg.fill ins(%cst : f16) outs(%init214 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm216 = linalg.matmul ins(%add213, %w_ff6_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill215 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty217 = tensor.empty() : tensor<1024x4096xf16>
  %relu218 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm216 : tensor<1024x4096xf16>)
    outs(%empty217 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init219 = tensor.empty() : tensor<1024x1024xf16>
  %fill220 = linalg.fill ins(%cst : f16) outs(%init219 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm221 = linalg.matmul ins(%relu218, %w_ff6_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill220 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty222 = tensor.empty() : tensor<1024x1024xf16>
  %add223 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm221, %add213 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty222 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 7 ===

  // Q projection
  %init224 = tensor.empty() : tensor<1024x1024xf16>
  %fill225 = linalg.fill ins(%cst : f16) outs(%init224 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm226 = linalg.matmul ins(%add223, %w_q7 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill225 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init227 = tensor.empty() : tensor<1024x1024xf16>
  %fill228 = linalg.fill ins(%cst : f16) outs(%init227 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm229 = linalg.matmul ins(%add223, %w_k7 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill228 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init230 = tensor.empty() : tensor<1024x1024xf16>
  %fill231 = linalg.fill ins(%cst : f16) outs(%init230 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm232 = linalg.matmul ins(%add223, %w_v7 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill231 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init233 = tensor.empty() : tensor<1024x1024xf16>
  %fill234 = linalg.fill ins(%cst : f16) outs(%init233 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm235 = linalg.matmul ins(%mm226, %w_kt7 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init238 = tensor.empty() : tensor<1024x1024xf16>
  %fill239 = linalg.fill ins(%cst : f16) outs(%init238 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm240 = linalg.matmul ins(%relu237, %mm232 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill239 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init241 = tensor.empty() : tensor<1024x1024xf16>
  %fill242 = linalg.fill ins(%cst : f16) outs(%init241 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm243 = linalg.matmul ins(%mm240, %w_o7 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill242 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty244 = tensor.empty() : tensor<1024x1024xf16>
  %add245 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm243, %add223 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty244 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init246 = tensor.empty() : tensor<1024x4096xf16>
  %fill247 = linalg.fill ins(%cst : f16) outs(%init246 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm248 = linalg.matmul ins(%add245, %w_ff7_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill247 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty249 = tensor.empty() : tensor<1024x4096xf16>
  %relu250 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm248 : tensor<1024x4096xf16>)
    outs(%empty249 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init251 = tensor.empty() : tensor<1024x1024xf16>
  %fill252 = linalg.fill ins(%cst : f16) outs(%init251 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm253 = linalg.matmul ins(%relu250, %w_ff7_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill252 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty254 = tensor.empty() : tensor<1024x1024xf16>
  %add255 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm253, %add245 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty254 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 8 ===

  // Q projection
  %init256 = tensor.empty() : tensor<1024x1024xf16>
  %fill257 = linalg.fill ins(%cst : f16) outs(%init256 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm258 = linalg.matmul ins(%add255, %w_q8 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill257 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init259 = tensor.empty() : tensor<1024x1024xf16>
  %fill260 = linalg.fill ins(%cst : f16) outs(%init259 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm261 = linalg.matmul ins(%add255, %w_k8 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill260 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init262 = tensor.empty() : tensor<1024x1024xf16>
  %fill263 = linalg.fill ins(%cst : f16) outs(%init262 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm264 = linalg.matmul ins(%add255, %w_v8 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill263 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init265 = tensor.empty() : tensor<1024x1024xf16>
  %fill266 = linalg.fill ins(%cst : f16) outs(%init265 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm267 = linalg.matmul ins(%mm258, %w_kt8 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init270 = tensor.empty() : tensor<1024x1024xf16>
  %fill271 = linalg.fill ins(%cst : f16) outs(%init270 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm272 = linalg.matmul ins(%relu269, %mm264 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill271 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init273 = tensor.empty() : tensor<1024x1024xf16>
  %fill274 = linalg.fill ins(%cst : f16) outs(%init273 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm275 = linalg.matmul ins(%mm272, %w_o8 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill274 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty276 = tensor.empty() : tensor<1024x1024xf16>
  %add277 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm275, %add255 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty276 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init278 = tensor.empty() : tensor<1024x4096xf16>
  %fill279 = linalg.fill ins(%cst : f16) outs(%init278 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm280 = linalg.matmul ins(%add277, %w_ff8_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill279 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty281 = tensor.empty() : tensor<1024x4096xf16>
  %relu282 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm280 : tensor<1024x4096xf16>)
    outs(%empty281 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init283 = tensor.empty() : tensor<1024x1024xf16>
  %fill284 = linalg.fill ins(%cst : f16) outs(%init283 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm285 = linalg.matmul ins(%relu282, %w_ff8_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill284 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty286 = tensor.empty() : tensor<1024x1024xf16>
  %add287 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm285, %add277 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty286 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 9 ===

  // Q projection
  %init288 = tensor.empty() : tensor<1024x1024xf16>
  %fill289 = linalg.fill ins(%cst : f16) outs(%init288 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm290 = linalg.matmul ins(%add287, %w_q9 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill289 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init291 = tensor.empty() : tensor<1024x1024xf16>
  %fill292 = linalg.fill ins(%cst : f16) outs(%init291 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm293 = linalg.matmul ins(%add287, %w_k9 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill292 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init294 = tensor.empty() : tensor<1024x1024xf16>
  %fill295 = linalg.fill ins(%cst : f16) outs(%init294 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm296 = linalg.matmul ins(%add287, %w_v9 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill295 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init297 = tensor.empty() : tensor<1024x1024xf16>
  %fill298 = linalg.fill ins(%cst : f16) outs(%init297 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm299 = linalg.matmul ins(%mm290, %w_kt9 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init302 = tensor.empty() : tensor<1024x1024xf16>
  %fill303 = linalg.fill ins(%cst : f16) outs(%init302 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm304 = linalg.matmul ins(%relu301, %mm296 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill303 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init305 = tensor.empty() : tensor<1024x1024xf16>
  %fill306 = linalg.fill ins(%cst : f16) outs(%init305 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm307 = linalg.matmul ins(%mm304, %w_o9 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill306 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty308 = tensor.empty() : tensor<1024x1024xf16>
  %add309 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm307, %add287 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty308 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init310 = tensor.empty() : tensor<1024x4096xf16>
  %fill311 = linalg.fill ins(%cst : f16) outs(%init310 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm312 = linalg.matmul ins(%add309, %w_ff9_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill311 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty313 = tensor.empty() : tensor<1024x4096xf16>
  %relu314 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm312 : tensor<1024x4096xf16>)
    outs(%empty313 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init315 = tensor.empty() : tensor<1024x1024xf16>
  %fill316 = linalg.fill ins(%cst : f16) outs(%init315 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm317 = linalg.matmul ins(%relu314, %w_ff9_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill316 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty318 = tensor.empty() : tensor<1024x1024xf16>
  %add319 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm317, %add309 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty318 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 10 ===

  // Q projection
  %init320 = tensor.empty() : tensor<1024x1024xf16>
  %fill321 = linalg.fill ins(%cst : f16) outs(%init320 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm322 = linalg.matmul ins(%add319, %w_q10 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill321 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init323 = tensor.empty() : tensor<1024x1024xf16>
  %fill324 = linalg.fill ins(%cst : f16) outs(%init323 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm325 = linalg.matmul ins(%add319, %w_k10 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill324 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init326 = tensor.empty() : tensor<1024x1024xf16>
  %fill327 = linalg.fill ins(%cst : f16) outs(%init326 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm328 = linalg.matmul ins(%add319, %w_v10 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill327 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init329 = tensor.empty() : tensor<1024x1024xf16>
  %fill330 = linalg.fill ins(%cst : f16) outs(%init329 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm331 = linalg.matmul ins(%mm322, %w_kt10 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init334 = tensor.empty() : tensor<1024x1024xf16>
  %fill335 = linalg.fill ins(%cst : f16) outs(%init334 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm336 = linalg.matmul ins(%relu333, %mm328 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill335 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init337 = tensor.empty() : tensor<1024x1024xf16>
  %fill338 = linalg.fill ins(%cst : f16) outs(%init337 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm339 = linalg.matmul ins(%mm336, %w_o10 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill338 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty340 = tensor.empty() : tensor<1024x1024xf16>
  %add341 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm339, %add319 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty340 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init342 = tensor.empty() : tensor<1024x4096xf16>
  %fill343 = linalg.fill ins(%cst : f16) outs(%init342 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm344 = linalg.matmul ins(%add341, %w_ff10_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill343 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty345 = tensor.empty() : tensor<1024x4096xf16>
  %relu346 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm344 : tensor<1024x4096xf16>)
    outs(%empty345 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init347 = tensor.empty() : tensor<1024x1024xf16>
  %fill348 = linalg.fill ins(%cst : f16) outs(%init347 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm349 = linalg.matmul ins(%relu346, %w_ff10_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill348 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty350 = tensor.empty() : tensor<1024x1024xf16>
  %add351 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm349, %add341 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty350 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 11 ===

  // Q projection
  %init352 = tensor.empty() : tensor<1024x1024xf16>
  %fill353 = linalg.fill ins(%cst : f16) outs(%init352 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm354 = linalg.matmul ins(%add351, %w_q11 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill353 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init355 = tensor.empty() : tensor<1024x1024xf16>
  %fill356 = linalg.fill ins(%cst : f16) outs(%init355 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm357 = linalg.matmul ins(%add351, %w_k11 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill356 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init358 = tensor.empty() : tensor<1024x1024xf16>
  %fill359 = linalg.fill ins(%cst : f16) outs(%init358 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm360 = linalg.matmul ins(%add351, %w_v11 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill359 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init361 = tensor.empty() : tensor<1024x1024xf16>
  %fill362 = linalg.fill ins(%cst : f16) outs(%init361 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm363 = linalg.matmul ins(%mm354, %w_kt11 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init366 = tensor.empty() : tensor<1024x1024xf16>
  %fill367 = linalg.fill ins(%cst : f16) outs(%init366 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm368 = linalg.matmul ins(%relu365, %mm360 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill367 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init369 = tensor.empty() : tensor<1024x1024xf16>
  %fill370 = linalg.fill ins(%cst : f16) outs(%init369 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm371 = linalg.matmul ins(%mm368, %w_o11 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill370 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty372 = tensor.empty() : tensor<1024x1024xf16>
  %add373 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm371, %add351 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty372 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init374 = tensor.empty() : tensor<1024x4096xf16>
  %fill375 = linalg.fill ins(%cst : f16) outs(%init374 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm376 = linalg.matmul ins(%add373, %w_ff11_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill375 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty377 = tensor.empty() : tensor<1024x4096xf16>
  %relu378 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm376 : tensor<1024x4096xf16>)
    outs(%empty377 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init379 = tensor.empty() : tensor<1024x1024xf16>
  %fill380 = linalg.fill ins(%cst : f16) outs(%init379 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm381 = linalg.matmul ins(%relu378, %w_ff11_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill380 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty382 = tensor.empty() : tensor<1024x1024xf16>
  %add383 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm381, %add373 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty382 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 12 ===

  // Q projection
  %init384 = tensor.empty() : tensor<1024x1024xf16>
  %fill385 = linalg.fill ins(%cst : f16) outs(%init384 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm386 = linalg.matmul ins(%add383, %w_q12 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill385 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init387 = tensor.empty() : tensor<1024x1024xf16>
  %fill388 = linalg.fill ins(%cst : f16) outs(%init387 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm389 = linalg.matmul ins(%add383, %w_k12 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill388 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init390 = tensor.empty() : tensor<1024x1024xf16>
  %fill391 = linalg.fill ins(%cst : f16) outs(%init390 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm392 = linalg.matmul ins(%add383, %w_v12 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill391 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init393 = tensor.empty() : tensor<1024x1024xf16>
  %fill394 = linalg.fill ins(%cst : f16) outs(%init393 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm395 = linalg.matmul ins(%mm386, %w_kt12 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init398 = tensor.empty() : tensor<1024x1024xf16>
  %fill399 = linalg.fill ins(%cst : f16) outs(%init398 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm400 = linalg.matmul ins(%relu397, %mm392 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill399 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init401 = tensor.empty() : tensor<1024x1024xf16>
  %fill402 = linalg.fill ins(%cst : f16) outs(%init401 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm403 = linalg.matmul ins(%mm400, %w_o12 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill402 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty404 = tensor.empty() : tensor<1024x1024xf16>
  %add405 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm403, %add383 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty404 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init406 = tensor.empty() : tensor<1024x4096xf16>
  %fill407 = linalg.fill ins(%cst : f16) outs(%init406 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm408 = linalg.matmul ins(%add405, %w_ff12_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill407 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty409 = tensor.empty() : tensor<1024x4096xf16>
  %relu410 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm408 : tensor<1024x4096xf16>)
    outs(%empty409 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init411 = tensor.empty() : tensor<1024x1024xf16>
  %fill412 = linalg.fill ins(%cst : f16) outs(%init411 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm413 = linalg.matmul ins(%relu410, %w_ff12_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill412 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty414 = tensor.empty() : tensor<1024x1024xf16>
  %add415 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm413, %add405 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty414 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 13 ===

  // Q projection
  %init416 = tensor.empty() : tensor<1024x1024xf16>
  %fill417 = linalg.fill ins(%cst : f16) outs(%init416 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm418 = linalg.matmul ins(%add415, %w_q13 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill417 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init419 = tensor.empty() : tensor<1024x1024xf16>
  %fill420 = linalg.fill ins(%cst : f16) outs(%init419 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm421 = linalg.matmul ins(%add415, %w_k13 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill420 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init422 = tensor.empty() : tensor<1024x1024xf16>
  %fill423 = linalg.fill ins(%cst : f16) outs(%init422 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm424 = linalg.matmul ins(%add415, %w_v13 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill423 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init425 = tensor.empty() : tensor<1024x1024xf16>
  %fill426 = linalg.fill ins(%cst : f16) outs(%init425 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm427 = linalg.matmul ins(%mm418, %w_kt13 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init430 = tensor.empty() : tensor<1024x1024xf16>
  %fill431 = linalg.fill ins(%cst : f16) outs(%init430 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm432 = linalg.matmul ins(%relu429, %mm424 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill431 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init433 = tensor.empty() : tensor<1024x1024xf16>
  %fill434 = linalg.fill ins(%cst : f16) outs(%init433 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm435 = linalg.matmul ins(%mm432, %w_o13 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill434 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty436 = tensor.empty() : tensor<1024x1024xf16>
  %add437 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm435, %add415 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty436 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init438 = tensor.empty() : tensor<1024x4096xf16>
  %fill439 = linalg.fill ins(%cst : f16) outs(%init438 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm440 = linalg.matmul ins(%add437, %w_ff13_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill439 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty441 = tensor.empty() : tensor<1024x4096xf16>
  %relu442 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm440 : tensor<1024x4096xf16>)
    outs(%empty441 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init443 = tensor.empty() : tensor<1024x1024xf16>
  %fill444 = linalg.fill ins(%cst : f16) outs(%init443 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm445 = linalg.matmul ins(%relu442, %w_ff13_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill444 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty446 = tensor.empty() : tensor<1024x1024xf16>
  %add447 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm445, %add437 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty446 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 14 ===

  // Q projection
  %init448 = tensor.empty() : tensor<1024x1024xf16>
  %fill449 = linalg.fill ins(%cst : f16) outs(%init448 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm450 = linalg.matmul ins(%add447, %w_q14 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill449 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init451 = tensor.empty() : tensor<1024x1024xf16>
  %fill452 = linalg.fill ins(%cst : f16) outs(%init451 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm453 = linalg.matmul ins(%add447, %w_k14 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill452 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init454 = tensor.empty() : tensor<1024x1024xf16>
  %fill455 = linalg.fill ins(%cst : f16) outs(%init454 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm456 = linalg.matmul ins(%add447, %w_v14 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill455 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init457 = tensor.empty() : tensor<1024x1024xf16>
  %fill458 = linalg.fill ins(%cst : f16) outs(%init457 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm459 = linalg.matmul ins(%mm450, %w_kt14 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init462 = tensor.empty() : tensor<1024x1024xf16>
  %fill463 = linalg.fill ins(%cst : f16) outs(%init462 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm464 = linalg.matmul ins(%relu461, %mm456 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill463 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init465 = tensor.empty() : tensor<1024x1024xf16>
  %fill466 = linalg.fill ins(%cst : f16) outs(%init465 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm467 = linalg.matmul ins(%mm464, %w_o14 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill466 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty468 = tensor.empty() : tensor<1024x1024xf16>
  %add469 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm467, %add447 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty468 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init470 = tensor.empty() : tensor<1024x4096xf16>
  %fill471 = linalg.fill ins(%cst : f16) outs(%init470 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm472 = linalg.matmul ins(%add469, %w_ff14_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill471 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty473 = tensor.empty() : tensor<1024x4096xf16>
  %relu474 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm472 : tensor<1024x4096xf16>)
    outs(%empty473 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init475 = tensor.empty() : tensor<1024x1024xf16>
  %fill476 = linalg.fill ins(%cst : f16) outs(%init475 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm477 = linalg.matmul ins(%relu474, %w_ff14_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill476 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty478 = tensor.empty() : tensor<1024x1024xf16>
  %add479 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm477, %add469 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty478 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 15 ===

  // Q projection
  %init480 = tensor.empty() : tensor<1024x1024xf16>
  %fill481 = linalg.fill ins(%cst : f16) outs(%init480 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm482 = linalg.matmul ins(%add479, %w_q15 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill481 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init483 = tensor.empty() : tensor<1024x1024xf16>
  %fill484 = linalg.fill ins(%cst : f16) outs(%init483 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm485 = linalg.matmul ins(%add479, %w_k15 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill484 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init486 = tensor.empty() : tensor<1024x1024xf16>
  %fill487 = linalg.fill ins(%cst : f16) outs(%init486 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm488 = linalg.matmul ins(%add479, %w_v15 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill487 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init489 = tensor.empty() : tensor<1024x1024xf16>
  %fill490 = linalg.fill ins(%cst : f16) outs(%init489 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm491 = linalg.matmul ins(%mm482, %w_kt15 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init494 = tensor.empty() : tensor<1024x1024xf16>
  %fill495 = linalg.fill ins(%cst : f16) outs(%init494 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm496 = linalg.matmul ins(%relu493, %mm488 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill495 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init497 = tensor.empty() : tensor<1024x1024xf16>
  %fill498 = linalg.fill ins(%cst : f16) outs(%init497 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm499 = linalg.matmul ins(%mm496, %w_o15 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill498 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty500 = tensor.empty() : tensor<1024x1024xf16>
  %add501 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm499, %add479 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty500 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init502 = tensor.empty() : tensor<1024x4096xf16>
  %fill503 = linalg.fill ins(%cst : f16) outs(%init502 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm504 = linalg.matmul ins(%add501, %w_ff15_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill503 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty505 = tensor.empty() : tensor<1024x4096xf16>
  %relu506 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm504 : tensor<1024x4096xf16>)
    outs(%empty505 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init507 = tensor.empty() : tensor<1024x1024xf16>
  %fill508 = linalg.fill ins(%cst : f16) outs(%init507 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm509 = linalg.matmul ins(%relu506, %w_ff15_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill508 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty510 = tensor.empty() : tensor<1024x1024xf16>
  %add511 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm509, %add501 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty510 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 16 ===

  // Q projection
  %init512 = tensor.empty() : tensor<1024x1024xf16>
  %fill513 = linalg.fill ins(%cst : f16) outs(%init512 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm514 = linalg.matmul ins(%add511, %w_q16 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill513 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init515 = tensor.empty() : tensor<1024x1024xf16>
  %fill516 = linalg.fill ins(%cst : f16) outs(%init515 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm517 = linalg.matmul ins(%add511, %w_k16 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill516 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init518 = tensor.empty() : tensor<1024x1024xf16>
  %fill519 = linalg.fill ins(%cst : f16) outs(%init518 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm520 = linalg.matmul ins(%add511, %w_v16 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill519 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init521 = tensor.empty() : tensor<1024x1024xf16>
  %fill522 = linalg.fill ins(%cst : f16) outs(%init521 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm523 = linalg.matmul ins(%mm514, %w_kt16 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init526 = tensor.empty() : tensor<1024x1024xf16>
  %fill527 = linalg.fill ins(%cst : f16) outs(%init526 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm528 = linalg.matmul ins(%relu525, %mm520 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill527 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init529 = tensor.empty() : tensor<1024x1024xf16>
  %fill530 = linalg.fill ins(%cst : f16) outs(%init529 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm531 = linalg.matmul ins(%mm528, %w_o16 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill530 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty532 = tensor.empty() : tensor<1024x1024xf16>
  %add533 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm531, %add511 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty532 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init534 = tensor.empty() : tensor<1024x4096xf16>
  %fill535 = linalg.fill ins(%cst : f16) outs(%init534 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm536 = linalg.matmul ins(%add533, %w_ff16_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill535 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty537 = tensor.empty() : tensor<1024x4096xf16>
  %relu538 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm536 : tensor<1024x4096xf16>)
    outs(%empty537 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init539 = tensor.empty() : tensor<1024x1024xf16>
  %fill540 = linalg.fill ins(%cst : f16) outs(%init539 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm541 = linalg.matmul ins(%relu538, %w_ff16_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill540 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty542 = tensor.empty() : tensor<1024x1024xf16>
  %add543 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm541, %add533 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty542 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 17 ===

  // Q projection
  %init544 = tensor.empty() : tensor<1024x1024xf16>
  %fill545 = linalg.fill ins(%cst : f16) outs(%init544 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm546 = linalg.matmul ins(%add543, %w_q17 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill545 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init547 = tensor.empty() : tensor<1024x1024xf16>
  %fill548 = linalg.fill ins(%cst : f16) outs(%init547 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm549 = linalg.matmul ins(%add543, %w_k17 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill548 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init550 = tensor.empty() : tensor<1024x1024xf16>
  %fill551 = linalg.fill ins(%cst : f16) outs(%init550 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm552 = linalg.matmul ins(%add543, %w_v17 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill551 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init553 = tensor.empty() : tensor<1024x1024xf16>
  %fill554 = linalg.fill ins(%cst : f16) outs(%init553 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm555 = linalg.matmul ins(%mm546, %w_kt17 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init558 = tensor.empty() : tensor<1024x1024xf16>
  %fill559 = linalg.fill ins(%cst : f16) outs(%init558 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm560 = linalg.matmul ins(%relu557, %mm552 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill559 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init561 = tensor.empty() : tensor<1024x1024xf16>
  %fill562 = linalg.fill ins(%cst : f16) outs(%init561 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm563 = linalg.matmul ins(%mm560, %w_o17 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill562 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty564 = tensor.empty() : tensor<1024x1024xf16>
  %add565 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm563, %add543 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty564 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init566 = tensor.empty() : tensor<1024x4096xf16>
  %fill567 = linalg.fill ins(%cst : f16) outs(%init566 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm568 = linalg.matmul ins(%add565, %w_ff17_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill567 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty569 = tensor.empty() : tensor<1024x4096xf16>
  %relu570 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm568 : tensor<1024x4096xf16>)
    outs(%empty569 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init571 = tensor.empty() : tensor<1024x1024xf16>
  %fill572 = linalg.fill ins(%cst : f16) outs(%init571 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm573 = linalg.matmul ins(%relu570, %w_ff17_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill572 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty574 = tensor.empty() : tensor<1024x1024xf16>
  %add575 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm573, %add565 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty574 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 18 ===

  // Q projection
  %init576 = tensor.empty() : tensor<1024x1024xf16>
  %fill577 = linalg.fill ins(%cst : f16) outs(%init576 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm578 = linalg.matmul ins(%add575, %w_q18 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill577 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init579 = tensor.empty() : tensor<1024x1024xf16>
  %fill580 = linalg.fill ins(%cst : f16) outs(%init579 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm581 = linalg.matmul ins(%add575, %w_k18 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill580 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init582 = tensor.empty() : tensor<1024x1024xf16>
  %fill583 = linalg.fill ins(%cst : f16) outs(%init582 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm584 = linalg.matmul ins(%add575, %w_v18 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill583 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init585 = tensor.empty() : tensor<1024x1024xf16>
  %fill586 = linalg.fill ins(%cst : f16) outs(%init585 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm587 = linalg.matmul ins(%mm578, %w_kt18 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init590 = tensor.empty() : tensor<1024x1024xf16>
  %fill591 = linalg.fill ins(%cst : f16) outs(%init590 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm592 = linalg.matmul ins(%relu589, %mm584 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill591 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init593 = tensor.empty() : tensor<1024x1024xf16>
  %fill594 = linalg.fill ins(%cst : f16) outs(%init593 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm595 = linalg.matmul ins(%mm592, %w_o18 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill594 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty596 = tensor.empty() : tensor<1024x1024xf16>
  %add597 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm595, %add575 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty596 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init598 = tensor.empty() : tensor<1024x4096xf16>
  %fill599 = linalg.fill ins(%cst : f16) outs(%init598 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm600 = linalg.matmul ins(%add597, %w_ff18_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill599 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty601 = tensor.empty() : tensor<1024x4096xf16>
  %relu602 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm600 : tensor<1024x4096xf16>)
    outs(%empty601 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init603 = tensor.empty() : tensor<1024x1024xf16>
  %fill604 = linalg.fill ins(%cst : f16) outs(%init603 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm605 = linalg.matmul ins(%relu602, %w_ff18_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill604 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty606 = tensor.empty() : tensor<1024x1024xf16>
  %add607 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm605, %add597 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty606 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 19 ===

  // Q projection
  %init608 = tensor.empty() : tensor<1024x1024xf16>
  %fill609 = linalg.fill ins(%cst : f16) outs(%init608 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm610 = linalg.matmul ins(%add607, %w_q19 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill609 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init611 = tensor.empty() : tensor<1024x1024xf16>
  %fill612 = linalg.fill ins(%cst : f16) outs(%init611 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm613 = linalg.matmul ins(%add607, %w_k19 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill612 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init614 = tensor.empty() : tensor<1024x1024xf16>
  %fill615 = linalg.fill ins(%cst : f16) outs(%init614 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm616 = linalg.matmul ins(%add607, %w_v19 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill615 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init617 = tensor.empty() : tensor<1024x1024xf16>
  %fill618 = linalg.fill ins(%cst : f16) outs(%init617 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm619 = linalg.matmul ins(%mm610, %w_kt19 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init622 = tensor.empty() : tensor<1024x1024xf16>
  %fill623 = linalg.fill ins(%cst : f16) outs(%init622 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm624 = linalg.matmul ins(%relu621, %mm616 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill623 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init625 = tensor.empty() : tensor<1024x1024xf16>
  %fill626 = linalg.fill ins(%cst : f16) outs(%init625 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm627 = linalg.matmul ins(%mm624, %w_o19 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill626 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty628 = tensor.empty() : tensor<1024x1024xf16>
  %add629 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm627, %add607 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty628 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init630 = tensor.empty() : tensor<1024x4096xf16>
  %fill631 = linalg.fill ins(%cst : f16) outs(%init630 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm632 = linalg.matmul ins(%add629, %w_ff19_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill631 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty633 = tensor.empty() : tensor<1024x4096xf16>
  %relu634 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm632 : tensor<1024x4096xf16>)
    outs(%empty633 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init635 = tensor.empty() : tensor<1024x1024xf16>
  %fill636 = linalg.fill ins(%cst : f16) outs(%init635 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm637 = linalg.matmul ins(%relu634, %w_ff19_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill636 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty638 = tensor.empty() : tensor<1024x1024xf16>
  %add639 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm637, %add629 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty638 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 20 ===

  // Q projection
  %init640 = tensor.empty() : tensor<1024x1024xf16>
  %fill641 = linalg.fill ins(%cst : f16) outs(%init640 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm642 = linalg.matmul ins(%add639, %w_q20 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill641 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init643 = tensor.empty() : tensor<1024x1024xf16>
  %fill644 = linalg.fill ins(%cst : f16) outs(%init643 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm645 = linalg.matmul ins(%add639, %w_k20 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill644 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init646 = tensor.empty() : tensor<1024x1024xf16>
  %fill647 = linalg.fill ins(%cst : f16) outs(%init646 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm648 = linalg.matmul ins(%add639, %w_v20 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill647 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init649 = tensor.empty() : tensor<1024x1024xf16>
  %fill650 = linalg.fill ins(%cst : f16) outs(%init649 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm651 = linalg.matmul ins(%mm642, %w_kt20 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init654 = tensor.empty() : tensor<1024x1024xf16>
  %fill655 = linalg.fill ins(%cst : f16) outs(%init654 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm656 = linalg.matmul ins(%relu653, %mm648 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill655 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init657 = tensor.empty() : tensor<1024x1024xf16>
  %fill658 = linalg.fill ins(%cst : f16) outs(%init657 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm659 = linalg.matmul ins(%mm656, %w_o20 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill658 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty660 = tensor.empty() : tensor<1024x1024xf16>
  %add661 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm659, %add639 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty660 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init662 = tensor.empty() : tensor<1024x4096xf16>
  %fill663 = linalg.fill ins(%cst : f16) outs(%init662 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm664 = linalg.matmul ins(%add661, %w_ff20_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill663 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty665 = tensor.empty() : tensor<1024x4096xf16>
  %relu666 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm664 : tensor<1024x4096xf16>)
    outs(%empty665 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init667 = tensor.empty() : tensor<1024x1024xf16>
  %fill668 = linalg.fill ins(%cst : f16) outs(%init667 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm669 = linalg.matmul ins(%relu666, %w_ff20_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill668 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty670 = tensor.empty() : tensor<1024x1024xf16>
  %add671 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm669, %add661 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty670 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 21 ===

  // Q projection
  %init672 = tensor.empty() : tensor<1024x1024xf16>
  %fill673 = linalg.fill ins(%cst : f16) outs(%init672 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm674 = linalg.matmul ins(%add671, %w_q21 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill673 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init675 = tensor.empty() : tensor<1024x1024xf16>
  %fill676 = linalg.fill ins(%cst : f16) outs(%init675 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm677 = linalg.matmul ins(%add671, %w_k21 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill676 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init678 = tensor.empty() : tensor<1024x1024xf16>
  %fill679 = linalg.fill ins(%cst : f16) outs(%init678 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm680 = linalg.matmul ins(%add671, %w_v21 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill679 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init681 = tensor.empty() : tensor<1024x1024xf16>
  %fill682 = linalg.fill ins(%cst : f16) outs(%init681 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm683 = linalg.matmul ins(%mm674, %w_kt21 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init686 = tensor.empty() : tensor<1024x1024xf16>
  %fill687 = linalg.fill ins(%cst : f16) outs(%init686 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm688 = linalg.matmul ins(%relu685, %mm680 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill687 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init689 = tensor.empty() : tensor<1024x1024xf16>
  %fill690 = linalg.fill ins(%cst : f16) outs(%init689 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm691 = linalg.matmul ins(%mm688, %w_o21 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill690 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty692 = tensor.empty() : tensor<1024x1024xf16>
  %add693 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm691, %add671 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty692 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init694 = tensor.empty() : tensor<1024x4096xf16>
  %fill695 = linalg.fill ins(%cst : f16) outs(%init694 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm696 = linalg.matmul ins(%add693, %w_ff21_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill695 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty697 = tensor.empty() : tensor<1024x4096xf16>
  %relu698 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm696 : tensor<1024x4096xf16>)
    outs(%empty697 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init699 = tensor.empty() : tensor<1024x1024xf16>
  %fill700 = linalg.fill ins(%cst : f16) outs(%init699 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm701 = linalg.matmul ins(%relu698, %w_ff21_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill700 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty702 = tensor.empty() : tensor<1024x1024xf16>
  %add703 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm701, %add693 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty702 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 22 ===

  // Q projection
  %init704 = tensor.empty() : tensor<1024x1024xf16>
  %fill705 = linalg.fill ins(%cst : f16) outs(%init704 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm706 = linalg.matmul ins(%add703, %w_q22 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill705 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init707 = tensor.empty() : tensor<1024x1024xf16>
  %fill708 = linalg.fill ins(%cst : f16) outs(%init707 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm709 = linalg.matmul ins(%add703, %w_k22 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill708 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init710 = tensor.empty() : tensor<1024x1024xf16>
  %fill711 = linalg.fill ins(%cst : f16) outs(%init710 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm712 = linalg.matmul ins(%add703, %w_v22 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill711 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init713 = tensor.empty() : tensor<1024x1024xf16>
  %fill714 = linalg.fill ins(%cst : f16) outs(%init713 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm715 = linalg.matmul ins(%mm706, %w_kt22 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init718 = tensor.empty() : tensor<1024x1024xf16>
  %fill719 = linalg.fill ins(%cst : f16) outs(%init718 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm720 = linalg.matmul ins(%relu717, %mm712 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill719 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init721 = tensor.empty() : tensor<1024x1024xf16>
  %fill722 = linalg.fill ins(%cst : f16) outs(%init721 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm723 = linalg.matmul ins(%mm720, %w_o22 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill722 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty724 = tensor.empty() : tensor<1024x1024xf16>
  %add725 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm723, %add703 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty724 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init726 = tensor.empty() : tensor<1024x4096xf16>
  %fill727 = linalg.fill ins(%cst : f16) outs(%init726 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm728 = linalg.matmul ins(%add725, %w_ff22_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill727 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty729 = tensor.empty() : tensor<1024x4096xf16>
  %relu730 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm728 : tensor<1024x4096xf16>)
    outs(%empty729 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init731 = tensor.empty() : tensor<1024x1024xf16>
  %fill732 = linalg.fill ins(%cst : f16) outs(%init731 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm733 = linalg.matmul ins(%relu730, %w_ff22_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill732 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty734 = tensor.empty() : tensor<1024x1024xf16>
  %add735 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm733, %add725 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty734 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // === Transformer Layer 23 ===

  // Q projection
  %init736 = tensor.empty() : tensor<1024x1024xf16>
  %fill737 = linalg.fill ins(%cst : f16) outs(%init736 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm738 = linalg.matmul ins(%add735, %w_q23 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill737 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // K projection
  %init739 = tensor.empty() : tensor<1024x1024xf16>
  %fill740 = linalg.fill ins(%cst : f16) outs(%init739 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm741 = linalg.matmul ins(%add735, %w_k23 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill740 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // V projection
  %init742 = tensor.empty() : tensor<1024x1024xf16>
  %fill743 = linalg.fill ins(%cst : f16) outs(%init742 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm744 = linalg.matmul ins(%add735, %w_v23 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill743 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>

  // Attention scores: Q x K_transposed
  %init745 = tensor.empty() : tensor<1024x1024xf16>
  %fill746 = linalg.fill ins(%cst : f16) outs(%init745 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm747 = linalg.matmul ins(%mm738, %w_kt23 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
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
  %init750 = tensor.empty() : tensor<1024x1024xf16>
  %fill751 = linalg.fill ins(%cst : f16) outs(%init750 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm752 = linalg.matmul ins(%relu749, %mm744 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill751 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Output projection
  %init753 = tensor.empty() : tensor<1024x1024xf16>
  %fill754 = linalg.fill ins(%cst : f16) outs(%init753 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm755 = linalg.matmul ins(%mm752, %w_o23 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill754 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // Attention residual add
  %empty756 = tensor.empty() : tensor<1024x1024xf16>
  %add757 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm755, %add735 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty756 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  // FFN: up projection
  %init758 = tensor.empty() : tensor<1024x4096xf16>
  %fill759 = linalg.fill ins(%cst : f16) outs(%init758 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  %mm760 = linalg.matmul ins(%add757, %w_ff23_up : tensor<1024x1024xf16>, tensor<1024x4096xf16>)
                          outs(%fill759 : tensor<1024x4096xf16>) -> tensor<1024x4096xf16>
  // FFN ReLU
  %empty761 = tensor.empty() : tensor<1024x4096xf16>
  %relu762 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm760 : tensor<1024x4096xf16>)
    outs(%empty761 : tensor<1024x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1024x4096xf16>
  // FFN: down projection
  %init763 = tensor.empty() : tensor<1024x1024xf16>
  %fill764 = linalg.fill ins(%cst : f16) outs(%init763 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %mm765 = linalg.matmul ins(%relu762, %w_ff23_down : tensor<1024x4096xf16>, tensor<4096x1024xf16>)
                          outs(%fill764 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  // FFN residual add
  %empty766 = tensor.empty() : tensor<1024x1024xf16>
  %add767 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm765, %add757 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%empty766 : tensor<1024x1024xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>

  return %add767 : tensor<1024x1024xf16>
}
