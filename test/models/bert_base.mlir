func.func @bert_base(
    %input: tensor<512x768xf16>,
    %w_q0: tensor<768x768xf16>,
    %w_k0: tensor<768x768xf16>,
    %w_v0: tensor<768x768xf16>,
    %w_kt0: tensor<768x512xf16>,
    %w_o0: tensor<768x768xf16>,
    %w_ff0_up: tensor<768x3072xf16>,
    %w_ff0_down: tensor<3072x768xf16>,
    %w_q1: tensor<768x768xf16>,
    %w_k1: tensor<768x768xf16>,
    %w_v1: tensor<768x768xf16>,
    %w_kt1: tensor<768x512xf16>,
    %w_o1: tensor<768x768xf16>,
    %w_ff1_up: tensor<768x3072xf16>,
    %w_ff1_down: tensor<3072x768xf16>,
    %w_q2: tensor<768x768xf16>,
    %w_k2: tensor<768x768xf16>,
    %w_v2: tensor<768x768xf16>,
    %w_kt2: tensor<768x512xf16>,
    %w_o2: tensor<768x768xf16>,
    %w_ff2_up: tensor<768x3072xf16>,
    %w_ff2_down: tensor<3072x768xf16>,
    %w_q3: tensor<768x768xf16>,
    %w_k3: tensor<768x768xf16>,
    %w_v3: tensor<768x768xf16>,
    %w_kt3: tensor<768x512xf16>,
    %w_o3: tensor<768x768xf16>,
    %w_ff3_up: tensor<768x3072xf16>,
    %w_ff3_down: tensor<3072x768xf16>,
    %w_q4: tensor<768x768xf16>,
    %w_k4: tensor<768x768xf16>,
    %w_v4: tensor<768x768xf16>,
    %w_kt4: tensor<768x512xf16>,
    %w_o4: tensor<768x768xf16>,
    %w_ff4_up: tensor<768x3072xf16>,
    %w_ff4_down: tensor<3072x768xf16>,
    %w_q5: tensor<768x768xf16>,
    %w_k5: tensor<768x768xf16>,
    %w_v5: tensor<768x768xf16>,
    %w_kt5: tensor<768x512xf16>,
    %w_o5: tensor<768x768xf16>,
    %w_ff5_up: tensor<768x3072xf16>,
    %w_ff5_down: tensor<3072x768xf16>,
    %w_q6: tensor<768x768xf16>,
    %w_k6: tensor<768x768xf16>,
    %w_v6: tensor<768x768xf16>,
    %w_kt6: tensor<768x512xf16>,
    %w_o6: tensor<768x768xf16>,
    %w_ff6_up: tensor<768x3072xf16>,
    %w_ff6_down: tensor<3072x768xf16>,
    %w_q7: tensor<768x768xf16>,
    %w_k7: tensor<768x768xf16>,
    %w_v7: tensor<768x768xf16>,
    %w_kt7: tensor<768x512xf16>,
    %w_o7: tensor<768x768xf16>,
    %w_ff7_up: tensor<768x3072xf16>,
    %w_ff7_down: tensor<3072x768xf16>,
    %w_q8: tensor<768x768xf16>,
    %w_k8: tensor<768x768xf16>,
    %w_v8: tensor<768x768xf16>,
    %w_kt8: tensor<768x512xf16>,
    %w_o8: tensor<768x768xf16>,
    %w_ff8_up: tensor<768x3072xf16>,
    %w_ff8_down: tensor<3072x768xf16>,
    %w_q9: tensor<768x768xf16>,
    %w_k9: tensor<768x768xf16>,
    %w_v9: tensor<768x768xf16>,
    %w_kt9: tensor<768x512xf16>,
    %w_o9: tensor<768x768xf16>,
    %w_ff9_up: tensor<768x3072xf16>,
    %w_ff9_down: tensor<3072x768xf16>,
    %w_q10: tensor<768x768xf16>,
    %w_k10: tensor<768x768xf16>,
    %w_v10: tensor<768x768xf16>,
    %w_kt10: tensor<768x512xf16>,
    %w_o10: tensor<768x768xf16>,
    %w_ff10_up: tensor<768x3072xf16>,
    %w_ff10_down: tensor<3072x768xf16>,
    %w_q11: tensor<768x768xf16>,
    %w_k11: tensor<768x768xf16>,
    %w_v11: tensor<768x768xf16>,
    %w_kt11: tensor<768x512xf16>,
    %w_o11: tensor<768x768xf16>,
    %w_ff11_up: tensor<768x3072xf16>,
    %w_ff11_down: tensor<3072x768xf16>) -> tensor<512x768xf16> {
  %cst = arith.constant 0.0 : f16

  // === Transformer Layer 0 ===

  // Q projection
  %init0 = tensor.empty() : tensor<512x768xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init0 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm2 = linalg.matmul ins(%input, %w_q0 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill1 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // K projection
  %init3 = tensor.empty() : tensor<512x768xf16>
  %fill4 = linalg.fill ins(%cst : f16) outs(%init3 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm5 = linalg.matmul ins(%input, %w_k0 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill4 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // V projection
  %init6 = tensor.empty() : tensor<512x768xf16>
  %fill7 = linalg.fill ins(%cst : f16) outs(%init6 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm8 = linalg.matmul ins(%input, %w_v0 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill7 : tensor<512x768xf16>) -> tensor<512x768xf16>

  // Attention scores: Q x K_transposed
  %init9 = tensor.empty() : tensor<512x512xf16>
  %fill10 = linalg.fill ins(%cst : f16) outs(%init9 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm11 = linalg.matmul ins(%mm2, %w_kt0 : tensor<512x768xf16>, tensor<768x512xf16>)
                          outs(%fill10 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Softmax approximation (relu)
  %empty12 = tensor.empty() : tensor<512x512xf16>
  %relu13 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm11 : tensor<512x512xf16>)
    outs(%empty12 : tensor<512x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x512xf16>

  // Attention output: scores x V
  %init14 = tensor.empty() : tensor<512x768xf16>
  %fill15 = linalg.fill ins(%cst : f16) outs(%init14 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm16 = linalg.matmul ins(%relu13, %mm8 : tensor<512x512xf16>, tensor<512x768xf16>)
                          outs(%fill15 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Output projection
  %init17 = tensor.empty() : tensor<512x768xf16>
  %fill18 = linalg.fill ins(%cst : f16) outs(%init17 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm19 = linalg.matmul ins(%mm16, %w_o0 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill18 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Attention residual add
  %empty20 = tensor.empty() : tensor<512x768xf16>
  %add21 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm19, %input : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty20 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // FFN: up projection
  %init22 = tensor.empty() : tensor<512x3072xf16>
  %fill23 = linalg.fill ins(%cst : f16) outs(%init22 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  %mm24 = linalg.matmul ins(%add21, %w_ff0_up : tensor<512x768xf16>, tensor<768x3072xf16>)
                          outs(%fill23 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  // FFN ReLU
  %empty25 = tensor.empty() : tensor<512x3072xf16>
  %relu26 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm24 : tensor<512x3072xf16>)
    outs(%empty25 : tensor<512x3072xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x3072xf16>
  // FFN: down projection
  %init27 = tensor.empty() : tensor<512x768xf16>
  %fill28 = linalg.fill ins(%cst : f16) outs(%init27 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm29 = linalg.matmul ins(%relu26, %w_ff0_down : tensor<512x3072xf16>, tensor<3072x768xf16>)
                          outs(%fill28 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // FFN residual add
  %empty30 = tensor.empty() : tensor<512x768xf16>
  %add31 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm29, %add21 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty30 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // === Transformer Layer 1 ===

  // Q projection
  %init32 = tensor.empty() : tensor<512x768xf16>
  %fill33 = linalg.fill ins(%cst : f16) outs(%init32 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm34 = linalg.matmul ins(%add31, %w_q1 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill33 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // K projection
  %init35 = tensor.empty() : tensor<512x768xf16>
  %fill36 = linalg.fill ins(%cst : f16) outs(%init35 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm37 = linalg.matmul ins(%add31, %w_k1 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill36 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // V projection
  %init38 = tensor.empty() : tensor<512x768xf16>
  %fill39 = linalg.fill ins(%cst : f16) outs(%init38 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm40 = linalg.matmul ins(%add31, %w_v1 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill39 : tensor<512x768xf16>) -> tensor<512x768xf16>

  // Attention scores: Q x K_transposed
  %init41 = tensor.empty() : tensor<512x512xf16>
  %fill42 = linalg.fill ins(%cst : f16) outs(%init41 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm43 = linalg.matmul ins(%mm34, %w_kt1 : tensor<512x768xf16>, tensor<768x512xf16>)
                          outs(%fill42 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Softmax approximation (relu)
  %empty44 = tensor.empty() : tensor<512x512xf16>
  %relu45 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm43 : tensor<512x512xf16>)
    outs(%empty44 : tensor<512x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x512xf16>

  // Attention output: scores x V
  %init46 = tensor.empty() : tensor<512x768xf16>
  %fill47 = linalg.fill ins(%cst : f16) outs(%init46 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm48 = linalg.matmul ins(%relu45, %mm40 : tensor<512x512xf16>, tensor<512x768xf16>)
                          outs(%fill47 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Output projection
  %init49 = tensor.empty() : tensor<512x768xf16>
  %fill50 = linalg.fill ins(%cst : f16) outs(%init49 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm51 = linalg.matmul ins(%mm48, %w_o1 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill50 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Attention residual add
  %empty52 = tensor.empty() : tensor<512x768xf16>
  %add53 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm51, %add31 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty52 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // FFN: up projection
  %init54 = tensor.empty() : tensor<512x3072xf16>
  %fill55 = linalg.fill ins(%cst : f16) outs(%init54 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  %mm56 = linalg.matmul ins(%add53, %w_ff1_up : tensor<512x768xf16>, tensor<768x3072xf16>)
                          outs(%fill55 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  // FFN ReLU
  %empty57 = tensor.empty() : tensor<512x3072xf16>
  %relu58 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm56 : tensor<512x3072xf16>)
    outs(%empty57 : tensor<512x3072xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x3072xf16>
  // FFN: down projection
  %init59 = tensor.empty() : tensor<512x768xf16>
  %fill60 = linalg.fill ins(%cst : f16) outs(%init59 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm61 = linalg.matmul ins(%relu58, %w_ff1_down : tensor<512x3072xf16>, tensor<3072x768xf16>)
                          outs(%fill60 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // FFN residual add
  %empty62 = tensor.empty() : tensor<512x768xf16>
  %add63 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm61, %add53 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty62 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // === Transformer Layer 2 ===

  // Q projection
  %init64 = tensor.empty() : tensor<512x768xf16>
  %fill65 = linalg.fill ins(%cst : f16) outs(%init64 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm66 = linalg.matmul ins(%add63, %w_q2 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill65 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // K projection
  %init67 = tensor.empty() : tensor<512x768xf16>
  %fill68 = linalg.fill ins(%cst : f16) outs(%init67 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm69 = linalg.matmul ins(%add63, %w_k2 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill68 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // V projection
  %init70 = tensor.empty() : tensor<512x768xf16>
  %fill71 = linalg.fill ins(%cst : f16) outs(%init70 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm72 = linalg.matmul ins(%add63, %w_v2 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill71 : tensor<512x768xf16>) -> tensor<512x768xf16>

  // Attention scores: Q x K_transposed
  %init73 = tensor.empty() : tensor<512x512xf16>
  %fill74 = linalg.fill ins(%cst : f16) outs(%init73 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm75 = linalg.matmul ins(%mm66, %w_kt2 : tensor<512x768xf16>, tensor<768x512xf16>)
                          outs(%fill74 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Softmax approximation (relu)
  %empty76 = tensor.empty() : tensor<512x512xf16>
  %relu77 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm75 : tensor<512x512xf16>)
    outs(%empty76 : tensor<512x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x512xf16>

  // Attention output: scores x V
  %init78 = tensor.empty() : tensor<512x768xf16>
  %fill79 = linalg.fill ins(%cst : f16) outs(%init78 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm80 = linalg.matmul ins(%relu77, %mm72 : tensor<512x512xf16>, tensor<512x768xf16>)
                          outs(%fill79 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Output projection
  %init81 = tensor.empty() : tensor<512x768xf16>
  %fill82 = linalg.fill ins(%cst : f16) outs(%init81 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm83 = linalg.matmul ins(%mm80, %w_o2 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill82 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Attention residual add
  %empty84 = tensor.empty() : tensor<512x768xf16>
  %add85 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm83, %add63 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty84 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // FFN: up projection
  %init86 = tensor.empty() : tensor<512x3072xf16>
  %fill87 = linalg.fill ins(%cst : f16) outs(%init86 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  %mm88 = linalg.matmul ins(%add85, %w_ff2_up : tensor<512x768xf16>, tensor<768x3072xf16>)
                          outs(%fill87 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  // FFN ReLU
  %empty89 = tensor.empty() : tensor<512x3072xf16>
  %relu90 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm88 : tensor<512x3072xf16>)
    outs(%empty89 : tensor<512x3072xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x3072xf16>
  // FFN: down projection
  %init91 = tensor.empty() : tensor<512x768xf16>
  %fill92 = linalg.fill ins(%cst : f16) outs(%init91 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm93 = linalg.matmul ins(%relu90, %w_ff2_down : tensor<512x3072xf16>, tensor<3072x768xf16>)
                          outs(%fill92 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // FFN residual add
  %empty94 = tensor.empty() : tensor<512x768xf16>
  %add95 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm93, %add85 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty94 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // === Transformer Layer 3 ===

  // Q projection
  %init96 = tensor.empty() : tensor<512x768xf16>
  %fill97 = linalg.fill ins(%cst : f16) outs(%init96 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm98 = linalg.matmul ins(%add95, %w_q3 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill97 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // K projection
  %init99 = tensor.empty() : tensor<512x768xf16>
  %fill100 = linalg.fill ins(%cst : f16) outs(%init99 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm101 = linalg.matmul ins(%add95, %w_k3 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill100 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // V projection
  %init102 = tensor.empty() : tensor<512x768xf16>
  %fill103 = linalg.fill ins(%cst : f16) outs(%init102 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm104 = linalg.matmul ins(%add95, %w_v3 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill103 : tensor<512x768xf16>) -> tensor<512x768xf16>

  // Attention scores: Q x K_transposed
  %init105 = tensor.empty() : tensor<512x512xf16>
  %fill106 = linalg.fill ins(%cst : f16) outs(%init105 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm107 = linalg.matmul ins(%mm98, %w_kt3 : tensor<512x768xf16>, tensor<768x512xf16>)
                          outs(%fill106 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Softmax approximation (relu)
  %empty108 = tensor.empty() : tensor<512x512xf16>
  %relu109 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm107 : tensor<512x512xf16>)
    outs(%empty108 : tensor<512x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x512xf16>

  // Attention output: scores x V
  %init110 = tensor.empty() : tensor<512x768xf16>
  %fill111 = linalg.fill ins(%cst : f16) outs(%init110 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm112 = linalg.matmul ins(%relu109, %mm104 : tensor<512x512xf16>, tensor<512x768xf16>)
                          outs(%fill111 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Output projection
  %init113 = tensor.empty() : tensor<512x768xf16>
  %fill114 = linalg.fill ins(%cst : f16) outs(%init113 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm115 = linalg.matmul ins(%mm112, %w_o3 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill114 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Attention residual add
  %empty116 = tensor.empty() : tensor<512x768xf16>
  %add117 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm115, %add95 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty116 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // FFN: up projection
  %init118 = tensor.empty() : tensor<512x3072xf16>
  %fill119 = linalg.fill ins(%cst : f16) outs(%init118 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  %mm120 = linalg.matmul ins(%add117, %w_ff3_up : tensor<512x768xf16>, tensor<768x3072xf16>)
                          outs(%fill119 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  // FFN ReLU
  %empty121 = tensor.empty() : tensor<512x3072xf16>
  %relu122 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm120 : tensor<512x3072xf16>)
    outs(%empty121 : tensor<512x3072xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x3072xf16>
  // FFN: down projection
  %init123 = tensor.empty() : tensor<512x768xf16>
  %fill124 = linalg.fill ins(%cst : f16) outs(%init123 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm125 = linalg.matmul ins(%relu122, %w_ff3_down : tensor<512x3072xf16>, tensor<3072x768xf16>)
                          outs(%fill124 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // FFN residual add
  %empty126 = tensor.empty() : tensor<512x768xf16>
  %add127 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm125, %add117 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty126 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // === Transformer Layer 4 ===

  // Q projection
  %init128 = tensor.empty() : tensor<512x768xf16>
  %fill129 = linalg.fill ins(%cst : f16) outs(%init128 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm130 = linalg.matmul ins(%add127, %w_q4 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill129 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // K projection
  %init131 = tensor.empty() : tensor<512x768xf16>
  %fill132 = linalg.fill ins(%cst : f16) outs(%init131 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm133 = linalg.matmul ins(%add127, %w_k4 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill132 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // V projection
  %init134 = tensor.empty() : tensor<512x768xf16>
  %fill135 = linalg.fill ins(%cst : f16) outs(%init134 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm136 = linalg.matmul ins(%add127, %w_v4 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill135 : tensor<512x768xf16>) -> tensor<512x768xf16>

  // Attention scores: Q x K_transposed
  %init137 = tensor.empty() : tensor<512x512xf16>
  %fill138 = linalg.fill ins(%cst : f16) outs(%init137 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm139 = linalg.matmul ins(%mm130, %w_kt4 : tensor<512x768xf16>, tensor<768x512xf16>)
                          outs(%fill138 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Softmax approximation (relu)
  %empty140 = tensor.empty() : tensor<512x512xf16>
  %relu141 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm139 : tensor<512x512xf16>)
    outs(%empty140 : tensor<512x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x512xf16>

  // Attention output: scores x V
  %init142 = tensor.empty() : tensor<512x768xf16>
  %fill143 = linalg.fill ins(%cst : f16) outs(%init142 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm144 = linalg.matmul ins(%relu141, %mm136 : tensor<512x512xf16>, tensor<512x768xf16>)
                          outs(%fill143 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Output projection
  %init145 = tensor.empty() : tensor<512x768xf16>
  %fill146 = linalg.fill ins(%cst : f16) outs(%init145 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm147 = linalg.matmul ins(%mm144, %w_o4 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill146 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Attention residual add
  %empty148 = tensor.empty() : tensor<512x768xf16>
  %add149 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm147, %add127 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty148 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // FFN: up projection
  %init150 = tensor.empty() : tensor<512x3072xf16>
  %fill151 = linalg.fill ins(%cst : f16) outs(%init150 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  %mm152 = linalg.matmul ins(%add149, %w_ff4_up : tensor<512x768xf16>, tensor<768x3072xf16>)
                          outs(%fill151 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  // FFN ReLU
  %empty153 = tensor.empty() : tensor<512x3072xf16>
  %relu154 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm152 : tensor<512x3072xf16>)
    outs(%empty153 : tensor<512x3072xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x3072xf16>
  // FFN: down projection
  %init155 = tensor.empty() : tensor<512x768xf16>
  %fill156 = linalg.fill ins(%cst : f16) outs(%init155 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm157 = linalg.matmul ins(%relu154, %w_ff4_down : tensor<512x3072xf16>, tensor<3072x768xf16>)
                          outs(%fill156 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // FFN residual add
  %empty158 = tensor.empty() : tensor<512x768xf16>
  %add159 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm157, %add149 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty158 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // === Transformer Layer 5 ===

  // Q projection
  %init160 = tensor.empty() : tensor<512x768xf16>
  %fill161 = linalg.fill ins(%cst : f16) outs(%init160 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm162 = linalg.matmul ins(%add159, %w_q5 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill161 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // K projection
  %init163 = tensor.empty() : tensor<512x768xf16>
  %fill164 = linalg.fill ins(%cst : f16) outs(%init163 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm165 = linalg.matmul ins(%add159, %w_k5 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill164 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // V projection
  %init166 = tensor.empty() : tensor<512x768xf16>
  %fill167 = linalg.fill ins(%cst : f16) outs(%init166 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm168 = linalg.matmul ins(%add159, %w_v5 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill167 : tensor<512x768xf16>) -> tensor<512x768xf16>

  // Attention scores: Q x K_transposed
  %init169 = tensor.empty() : tensor<512x512xf16>
  %fill170 = linalg.fill ins(%cst : f16) outs(%init169 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm171 = linalg.matmul ins(%mm162, %w_kt5 : tensor<512x768xf16>, tensor<768x512xf16>)
                          outs(%fill170 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Softmax approximation (relu)
  %empty172 = tensor.empty() : tensor<512x512xf16>
  %relu173 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm171 : tensor<512x512xf16>)
    outs(%empty172 : tensor<512x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x512xf16>

  // Attention output: scores x V
  %init174 = tensor.empty() : tensor<512x768xf16>
  %fill175 = linalg.fill ins(%cst : f16) outs(%init174 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm176 = linalg.matmul ins(%relu173, %mm168 : tensor<512x512xf16>, tensor<512x768xf16>)
                          outs(%fill175 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Output projection
  %init177 = tensor.empty() : tensor<512x768xf16>
  %fill178 = linalg.fill ins(%cst : f16) outs(%init177 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm179 = linalg.matmul ins(%mm176, %w_o5 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill178 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Attention residual add
  %empty180 = tensor.empty() : tensor<512x768xf16>
  %add181 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm179, %add159 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty180 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // FFN: up projection
  %init182 = tensor.empty() : tensor<512x3072xf16>
  %fill183 = linalg.fill ins(%cst : f16) outs(%init182 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  %mm184 = linalg.matmul ins(%add181, %w_ff5_up : tensor<512x768xf16>, tensor<768x3072xf16>)
                          outs(%fill183 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  // FFN ReLU
  %empty185 = tensor.empty() : tensor<512x3072xf16>
  %relu186 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm184 : tensor<512x3072xf16>)
    outs(%empty185 : tensor<512x3072xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x3072xf16>
  // FFN: down projection
  %init187 = tensor.empty() : tensor<512x768xf16>
  %fill188 = linalg.fill ins(%cst : f16) outs(%init187 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm189 = linalg.matmul ins(%relu186, %w_ff5_down : tensor<512x3072xf16>, tensor<3072x768xf16>)
                          outs(%fill188 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // FFN residual add
  %empty190 = tensor.empty() : tensor<512x768xf16>
  %add191 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm189, %add181 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty190 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // === Transformer Layer 6 ===

  // Q projection
  %init192 = tensor.empty() : tensor<512x768xf16>
  %fill193 = linalg.fill ins(%cst : f16) outs(%init192 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm194 = linalg.matmul ins(%add191, %w_q6 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill193 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // K projection
  %init195 = tensor.empty() : tensor<512x768xf16>
  %fill196 = linalg.fill ins(%cst : f16) outs(%init195 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm197 = linalg.matmul ins(%add191, %w_k6 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill196 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // V projection
  %init198 = tensor.empty() : tensor<512x768xf16>
  %fill199 = linalg.fill ins(%cst : f16) outs(%init198 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm200 = linalg.matmul ins(%add191, %w_v6 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill199 : tensor<512x768xf16>) -> tensor<512x768xf16>

  // Attention scores: Q x K_transposed
  %init201 = tensor.empty() : tensor<512x512xf16>
  %fill202 = linalg.fill ins(%cst : f16) outs(%init201 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm203 = linalg.matmul ins(%mm194, %w_kt6 : tensor<512x768xf16>, tensor<768x512xf16>)
                          outs(%fill202 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Softmax approximation (relu)
  %empty204 = tensor.empty() : tensor<512x512xf16>
  %relu205 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm203 : tensor<512x512xf16>)
    outs(%empty204 : tensor<512x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x512xf16>

  // Attention output: scores x V
  %init206 = tensor.empty() : tensor<512x768xf16>
  %fill207 = linalg.fill ins(%cst : f16) outs(%init206 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm208 = linalg.matmul ins(%relu205, %mm200 : tensor<512x512xf16>, tensor<512x768xf16>)
                          outs(%fill207 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Output projection
  %init209 = tensor.empty() : tensor<512x768xf16>
  %fill210 = linalg.fill ins(%cst : f16) outs(%init209 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm211 = linalg.matmul ins(%mm208, %w_o6 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill210 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Attention residual add
  %empty212 = tensor.empty() : tensor<512x768xf16>
  %add213 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm211, %add191 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty212 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // FFN: up projection
  %init214 = tensor.empty() : tensor<512x3072xf16>
  %fill215 = linalg.fill ins(%cst : f16) outs(%init214 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  %mm216 = linalg.matmul ins(%add213, %w_ff6_up : tensor<512x768xf16>, tensor<768x3072xf16>)
                          outs(%fill215 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  // FFN ReLU
  %empty217 = tensor.empty() : tensor<512x3072xf16>
  %relu218 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm216 : tensor<512x3072xf16>)
    outs(%empty217 : tensor<512x3072xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x3072xf16>
  // FFN: down projection
  %init219 = tensor.empty() : tensor<512x768xf16>
  %fill220 = linalg.fill ins(%cst : f16) outs(%init219 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm221 = linalg.matmul ins(%relu218, %w_ff6_down : tensor<512x3072xf16>, tensor<3072x768xf16>)
                          outs(%fill220 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // FFN residual add
  %empty222 = tensor.empty() : tensor<512x768xf16>
  %add223 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm221, %add213 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty222 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // === Transformer Layer 7 ===

  // Q projection
  %init224 = tensor.empty() : tensor<512x768xf16>
  %fill225 = linalg.fill ins(%cst : f16) outs(%init224 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm226 = linalg.matmul ins(%add223, %w_q7 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill225 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // K projection
  %init227 = tensor.empty() : tensor<512x768xf16>
  %fill228 = linalg.fill ins(%cst : f16) outs(%init227 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm229 = linalg.matmul ins(%add223, %w_k7 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill228 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // V projection
  %init230 = tensor.empty() : tensor<512x768xf16>
  %fill231 = linalg.fill ins(%cst : f16) outs(%init230 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm232 = linalg.matmul ins(%add223, %w_v7 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill231 : tensor<512x768xf16>) -> tensor<512x768xf16>

  // Attention scores: Q x K_transposed
  %init233 = tensor.empty() : tensor<512x512xf16>
  %fill234 = linalg.fill ins(%cst : f16) outs(%init233 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm235 = linalg.matmul ins(%mm226, %w_kt7 : tensor<512x768xf16>, tensor<768x512xf16>)
                          outs(%fill234 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Softmax approximation (relu)
  %empty236 = tensor.empty() : tensor<512x512xf16>
  %relu237 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm235 : tensor<512x512xf16>)
    outs(%empty236 : tensor<512x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x512xf16>

  // Attention output: scores x V
  %init238 = tensor.empty() : tensor<512x768xf16>
  %fill239 = linalg.fill ins(%cst : f16) outs(%init238 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm240 = linalg.matmul ins(%relu237, %mm232 : tensor<512x512xf16>, tensor<512x768xf16>)
                          outs(%fill239 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Output projection
  %init241 = tensor.empty() : tensor<512x768xf16>
  %fill242 = linalg.fill ins(%cst : f16) outs(%init241 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm243 = linalg.matmul ins(%mm240, %w_o7 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill242 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Attention residual add
  %empty244 = tensor.empty() : tensor<512x768xf16>
  %add245 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm243, %add223 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty244 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // FFN: up projection
  %init246 = tensor.empty() : tensor<512x3072xf16>
  %fill247 = linalg.fill ins(%cst : f16) outs(%init246 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  %mm248 = linalg.matmul ins(%add245, %w_ff7_up : tensor<512x768xf16>, tensor<768x3072xf16>)
                          outs(%fill247 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  // FFN ReLU
  %empty249 = tensor.empty() : tensor<512x3072xf16>
  %relu250 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm248 : tensor<512x3072xf16>)
    outs(%empty249 : tensor<512x3072xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x3072xf16>
  // FFN: down projection
  %init251 = tensor.empty() : tensor<512x768xf16>
  %fill252 = linalg.fill ins(%cst : f16) outs(%init251 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm253 = linalg.matmul ins(%relu250, %w_ff7_down : tensor<512x3072xf16>, tensor<3072x768xf16>)
                          outs(%fill252 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // FFN residual add
  %empty254 = tensor.empty() : tensor<512x768xf16>
  %add255 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm253, %add245 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty254 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // === Transformer Layer 8 ===

  // Q projection
  %init256 = tensor.empty() : tensor<512x768xf16>
  %fill257 = linalg.fill ins(%cst : f16) outs(%init256 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm258 = linalg.matmul ins(%add255, %w_q8 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill257 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // K projection
  %init259 = tensor.empty() : tensor<512x768xf16>
  %fill260 = linalg.fill ins(%cst : f16) outs(%init259 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm261 = linalg.matmul ins(%add255, %w_k8 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill260 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // V projection
  %init262 = tensor.empty() : tensor<512x768xf16>
  %fill263 = linalg.fill ins(%cst : f16) outs(%init262 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm264 = linalg.matmul ins(%add255, %w_v8 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill263 : tensor<512x768xf16>) -> tensor<512x768xf16>

  // Attention scores: Q x K_transposed
  %init265 = tensor.empty() : tensor<512x512xf16>
  %fill266 = linalg.fill ins(%cst : f16) outs(%init265 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm267 = linalg.matmul ins(%mm258, %w_kt8 : tensor<512x768xf16>, tensor<768x512xf16>)
                          outs(%fill266 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Softmax approximation (relu)
  %empty268 = tensor.empty() : tensor<512x512xf16>
  %relu269 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm267 : tensor<512x512xf16>)
    outs(%empty268 : tensor<512x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x512xf16>

  // Attention output: scores x V
  %init270 = tensor.empty() : tensor<512x768xf16>
  %fill271 = linalg.fill ins(%cst : f16) outs(%init270 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm272 = linalg.matmul ins(%relu269, %mm264 : tensor<512x512xf16>, tensor<512x768xf16>)
                          outs(%fill271 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Output projection
  %init273 = tensor.empty() : tensor<512x768xf16>
  %fill274 = linalg.fill ins(%cst : f16) outs(%init273 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm275 = linalg.matmul ins(%mm272, %w_o8 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill274 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Attention residual add
  %empty276 = tensor.empty() : tensor<512x768xf16>
  %add277 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm275, %add255 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty276 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // FFN: up projection
  %init278 = tensor.empty() : tensor<512x3072xf16>
  %fill279 = linalg.fill ins(%cst : f16) outs(%init278 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  %mm280 = linalg.matmul ins(%add277, %w_ff8_up : tensor<512x768xf16>, tensor<768x3072xf16>)
                          outs(%fill279 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  // FFN ReLU
  %empty281 = tensor.empty() : tensor<512x3072xf16>
  %relu282 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm280 : tensor<512x3072xf16>)
    outs(%empty281 : tensor<512x3072xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x3072xf16>
  // FFN: down projection
  %init283 = tensor.empty() : tensor<512x768xf16>
  %fill284 = linalg.fill ins(%cst : f16) outs(%init283 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm285 = linalg.matmul ins(%relu282, %w_ff8_down : tensor<512x3072xf16>, tensor<3072x768xf16>)
                          outs(%fill284 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // FFN residual add
  %empty286 = tensor.empty() : tensor<512x768xf16>
  %add287 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm285, %add277 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty286 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // === Transformer Layer 9 ===

  // Q projection
  %init288 = tensor.empty() : tensor<512x768xf16>
  %fill289 = linalg.fill ins(%cst : f16) outs(%init288 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm290 = linalg.matmul ins(%add287, %w_q9 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill289 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // K projection
  %init291 = tensor.empty() : tensor<512x768xf16>
  %fill292 = linalg.fill ins(%cst : f16) outs(%init291 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm293 = linalg.matmul ins(%add287, %w_k9 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill292 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // V projection
  %init294 = tensor.empty() : tensor<512x768xf16>
  %fill295 = linalg.fill ins(%cst : f16) outs(%init294 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm296 = linalg.matmul ins(%add287, %w_v9 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill295 : tensor<512x768xf16>) -> tensor<512x768xf16>

  // Attention scores: Q x K_transposed
  %init297 = tensor.empty() : tensor<512x512xf16>
  %fill298 = linalg.fill ins(%cst : f16) outs(%init297 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm299 = linalg.matmul ins(%mm290, %w_kt9 : tensor<512x768xf16>, tensor<768x512xf16>)
                          outs(%fill298 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Softmax approximation (relu)
  %empty300 = tensor.empty() : tensor<512x512xf16>
  %relu301 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm299 : tensor<512x512xf16>)
    outs(%empty300 : tensor<512x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x512xf16>

  // Attention output: scores x V
  %init302 = tensor.empty() : tensor<512x768xf16>
  %fill303 = linalg.fill ins(%cst : f16) outs(%init302 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm304 = linalg.matmul ins(%relu301, %mm296 : tensor<512x512xf16>, tensor<512x768xf16>)
                          outs(%fill303 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Output projection
  %init305 = tensor.empty() : tensor<512x768xf16>
  %fill306 = linalg.fill ins(%cst : f16) outs(%init305 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm307 = linalg.matmul ins(%mm304, %w_o9 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill306 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Attention residual add
  %empty308 = tensor.empty() : tensor<512x768xf16>
  %add309 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm307, %add287 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty308 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // FFN: up projection
  %init310 = tensor.empty() : tensor<512x3072xf16>
  %fill311 = linalg.fill ins(%cst : f16) outs(%init310 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  %mm312 = linalg.matmul ins(%add309, %w_ff9_up : tensor<512x768xf16>, tensor<768x3072xf16>)
                          outs(%fill311 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  // FFN ReLU
  %empty313 = tensor.empty() : tensor<512x3072xf16>
  %relu314 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm312 : tensor<512x3072xf16>)
    outs(%empty313 : tensor<512x3072xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x3072xf16>
  // FFN: down projection
  %init315 = tensor.empty() : tensor<512x768xf16>
  %fill316 = linalg.fill ins(%cst : f16) outs(%init315 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm317 = linalg.matmul ins(%relu314, %w_ff9_down : tensor<512x3072xf16>, tensor<3072x768xf16>)
                          outs(%fill316 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // FFN residual add
  %empty318 = tensor.empty() : tensor<512x768xf16>
  %add319 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm317, %add309 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty318 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // === Transformer Layer 10 ===

  // Q projection
  %init320 = tensor.empty() : tensor<512x768xf16>
  %fill321 = linalg.fill ins(%cst : f16) outs(%init320 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm322 = linalg.matmul ins(%add319, %w_q10 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill321 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // K projection
  %init323 = tensor.empty() : tensor<512x768xf16>
  %fill324 = linalg.fill ins(%cst : f16) outs(%init323 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm325 = linalg.matmul ins(%add319, %w_k10 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill324 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // V projection
  %init326 = tensor.empty() : tensor<512x768xf16>
  %fill327 = linalg.fill ins(%cst : f16) outs(%init326 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm328 = linalg.matmul ins(%add319, %w_v10 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill327 : tensor<512x768xf16>) -> tensor<512x768xf16>

  // Attention scores: Q x K_transposed
  %init329 = tensor.empty() : tensor<512x512xf16>
  %fill330 = linalg.fill ins(%cst : f16) outs(%init329 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm331 = linalg.matmul ins(%mm322, %w_kt10 : tensor<512x768xf16>, tensor<768x512xf16>)
                          outs(%fill330 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Softmax approximation (relu)
  %empty332 = tensor.empty() : tensor<512x512xf16>
  %relu333 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm331 : tensor<512x512xf16>)
    outs(%empty332 : tensor<512x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x512xf16>

  // Attention output: scores x V
  %init334 = tensor.empty() : tensor<512x768xf16>
  %fill335 = linalg.fill ins(%cst : f16) outs(%init334 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm336 = linalg.matmul ins(%relu333, %mm328 : tensor<512x512xf16>, tensor<512x768xf16>)
                          outs(%fill335 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Output projection
  %init337 = tensor.empty() : tensor<512x768xf16>
  %fill338 = linalg.fill ins(%cst : f16) outs(%init337 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm339 = linalg.matmul ins(%mm336, %w_o10 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill338 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Attention residual add
  %empty340 = tensor.empty() : tensor<512x768xf16>
  %add341 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm339, %add319 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty340 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // FFN: up projection
  %init342 = tensor.empty() : tensor<512x3072xf16>
  %fill343 = linalg.fill ins(%cst : f16) outs(%init342 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  %mm344 = linalg.matmul ins(%add341, %w_ff10_up : tensor<512x768xf16>, tensor<768x3072xf16>)
                          outs(%fill343 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  // FFN ReLU
  %empty345 = tensor.empty() : tensor<512x3072xf16>
  %relu346 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm344 : tensor<512x3072xf16>)
    outs(%empty345 : tensor<512x3072xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x3072xf16>
  // FFN: down projection
  %init347 = tensor.empty() : tensor<512x768xf16>
  %fill348 = linalg.fill ins(%cst : f16) outs(%init347 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm349 = linalg.matmul ins(%relu346, %w_ff10_down : tensor<512x3072xf16>, tensor<3072x768xf16>)
                          outs(%fill348 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // FFN residual add
  %empty350 = tensor.empty() : tensor<512x768xf16>
  %add351 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm349, %add341 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty350 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // === Transformer Layer 11 ===

  // Q projection
  %init352 = tensor.empty() : tensor<512x768xf16>
  %fill353 = linalg.fill ins(%cst : f16) outs(%init352 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm354 = linalg.matmul ins(%add351, %w_q11 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill353 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // K projection
  %init355 = tensor.empty() : tensor<512x768xf16>
  %fill356 = linalg.fill ins(%cst : f16) outs(%init355 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm357 = linalg.matmul ins(%add351, %w_k11 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill356 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // V projection
  %init358 = tensor.empty() : tensor<512x768xf16>
  %fill359 = linalg.fill ins(%cst : f16) outs(%init358 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm360 = linalg.matmul ins(%add351, %w_v11 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill359 : tensor<512x768xf16>) -> tensor<512x768xf16>

  // Attention scores: Q x K_transposed
  %init361 = tensor.empty() : tensor<512x512xf16>
  %fill362 = linalg.fill ins(%cst : f16) outs(%init361 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm363 = linalg.matmul ins(%mm354, %w_kt11 : tensor<512x768xf16>, tensor<768x512xf16>)
                          outs(%fill362 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Softmax approximation (relu)
  %empty364 = tensor.empty() : tensor<512x512xf16>
  %relu365 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm363 : tensor<512x512xf16>)
    outs(%empty364 : tensor<512x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x512xf16>

  // Attention output: scores x V
  %init366 = tensor.empty() : tensor<512x768xf16>
  %fill367 = linalg.fill ins(%cst : f16) outs(%init366 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm368 = linalg.matmul ins(%relu365, %mm360 : tensor<512x512xf16>, tensor<512x768xf16>)
                          outs(%fill367 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Output projection
  %init369 = tensor.empty() : tensor<512x768xf16>
  %fill370 = linalg.fill ins(%cst : f16) outs(%init369 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm371 = linalg.matmul ins(%mm368, %w_o11 : tensor<512x768xf16>, tensor<768x768xf16>)
                          outs(%fill370 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // Attention residual add
  %empty372 = tensor.empty() : tensor<512x768xf16>
  %add373 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm371, %add351 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty372 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  // FFN: up projection
  %init374 = tensor.empty() : tensor<512x3072xf16>
  %fill375 = linalg.fill ins(%cst : f16) outs(%init374 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  %mm376 = linalg.matmul ins(%add373, %w_ff11_up : tensor<512x768xf16>, tensor<768x3072xf16>)
                          outs(%fill375 : tensor<512x3072xf16>) -> tensor<512x3072xf16>
  // FFN ReLU
  %empty377 = tensor.empty() : tensor<512x3072xf16>
  %relu378 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm376 : tensor<512x3072xf16>)
    outs(%empty377 : tensor<512x3072xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x3072xf16>
  // FFN: down projection
  %init379 = tensor.empty() : tensor<512x768xf16>
  %fill380 = linalg.fill ins(%cst : f16) outs(%init379 : tensor<512x768xf16>) -> tensor<512x768xf16>
  %mm381 = linalg.matmul ins(%relu378, %w_ff11_down : tensor<512x3072xf16>, tensor<3072x768xf16>)
                          outs(%fill380 : tensor<512x768xf16>) -> tensor<512x768xf16>
  // FFN residual add
  %empty382 = tensor.empty() : tensor<512x768xf16>
  %add383 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm381, %add373 : tensor<512x768xf16>, tensor<512x768xf16>)
    outs(%empty382 : tensor<512x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x768xf16>

  return %add383 : tensor<512x768xf16>
}
