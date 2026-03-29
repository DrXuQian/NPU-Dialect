func.func @bert_medium(
    %input: tensor<512x512xf16>,
    %w_q0: tensor<512x512xf16>,
    %w_k0: tensor<512x512xf16>,
    %w_v0: tensor<512x512xf16>,
    %w_kt0: tensor<512x512xf16>,
    %w_o0: tensor<512x512xf16>,
    %w_ff0_up: tensor<512x2048xf16>,
    %w_ff0_down: tensor<2048x512xf16>,
    %w_q1: tensor<512x512xf16>,
    %w_k1: tensor<512x512xf16>,
    %w_v1: tensor<512x512xf16>,
    %w_kt1: tensor<512x512xf16>,
    %w_o1: tensor<512x512xf16>,
    %w_ff1_up: tensor<512x2048xf16>,
    %w_ff1_down: tensor<2048x512xf16>,
    %w_q2: tensor<512x512xf16>,
    %w_k2: tensor<512x512xf16>,
    %w_v2: tensor<512x512xf16>,
    %w_kt2: tensor<512x512xf16>,
    %w_o2: tensor<512x512xf16>,
    %w_ff2_up: tensor<512x2048xf16>,
    %w_ff2_down: tensor<2048x512xf16>,
    %w_q3: tensor<512x512xf16>,
    %w_k3: tensor<512x512xf16>,
    %w_v3: tensor<512x512xf16>,
    %w_kt3: tensor<512x512xf16>,
    %w_o3: tensor<512x512xf16>,
    %w_ff3_up: tensor<512x2048xf16>,
    %w_ff3_down: tensor<2048x512xf16>,
    %w_q4: tensor<512x512xf16>,
    %w_k4: tensor<512x512xf16>,
    %w_v4: tensor<512x512xf16>,
    %w_kt4: tensor<512x512xf16>,
    %w_o4: tensor<512x512xf16>,
    %w_ff4_up: tensor<512x2048xf16>,
    %w_ff4_down: tensor<2048x512xf16>,
    %w_q5: tensor<512x512xf16>,
    %w_k5: tensor<512x512xf16>,
    %w_v5: tensor<512x512xf16>,
    %w_kt5: tensor<512x512xf16>,
    %w_o5: tensor<512x512xf16>,
    %w_ff5_up: tensor<512x2048xf16>,
    %w_ff5_down: tensor<2048x512xf16>,
    %w_q6: tensor<512x512xf16>,
    %w_k6: tensor<512x512xf16>,
    %w_v6: tensor<512x512xf16>,
    %w_kt6: tensor<512x512xf16>,
    %w_o6: tensor<512x512xf16>,
    %w_ff6_up: tensor<512x2048xf16>,
    %w_ff6_down: tensor<2048x512xf16>,
    %w_q7: tensor<512x512xf16>,
    %w_k7: tensor<512x512xf16>,
    %w_v7: tensor<512x512xf16>,
    %w_kt7: tensor<512x512xf16>,
    %w_o7: tensor<512x512xf16>,
    %w_ff7_up: tensor<512x2048xf16>,
    %w_ff7_down: tensor<2048x512xf16>) -> tensor<512x512xf16> {
  %cst = arith.constant 0.0 : f16

  // === Transformer Layer 0 ===

  // Q projection
  %init0 = tensor.empty() : tensor<512x512xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init0 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm2 = linalg.matmul ins(%input, %w_q0 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill1 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // K projection
  %init3 = tensor.empty() : tensor<512x512xf16>
  %fill4 = linalg.fill ins(%cst : f16) outs(%init3 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm5 = linalg.matmul ins(%input, %w_k0 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill4 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // V projection
  %init6 = tensor.empty() : tensor<512x512xf16>
  %fill7 = linalg.fill ins(%cst : f16) outs(%init6 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm8 = linalg.matmul ins(%input, %w_v0 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill7 : tensor<512x512xf16>) -> tensor<512x512xf16>

  // Attention scores: Q x K_transposed
  %init9 = tensor.empty() : tensor<512x512xf16>
  %fill10 = linalg.fill ins(%cst : f16) outs(%init9 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm11 = linalg.matmul ins(%mm2, %w_kt0 : tensor<512x512xf16>, tensor<512x512xf16>)
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
  %init14 = tensor.empty() : tensor<512x512xf16>
  %fill15 = linalg.fill ins(%cst : f16) outs(%init14 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm16 = linalg.matmul ins(%relu13, %mm8 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill15 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Output projection
  %init17 = tensor.empty() : tensor<512x512xf16>
  %fill18 = linalg.fill ins(%cst : f16) outs(%init17 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm19 = linalg.matmul ins(%mm16, %w_o0 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill18 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Attention residual add
  %empty20 = tensor.empty() : tensor<512x512xf16>
  %add21 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm19, %input : tensor<512x512xf16>, tensor<512x512xf16>)
    outs(%empty20 : tensor<512x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x512xf16>

  // FFN: up projection
  %init22 = tensor.empty() : tensor<512x2048xf16>
  %fill23 = linalg.fill ins(%cst : f16) outs(%init22 : tensor<512x2048xf16>) -> tensor<512x2048xf16>
  %mm24 = linalg.matmul ins(%add21, %w_ff0_up : tensor<512x512xf16>, tensor<512x2048xf16>)
                          outs(%fill23 : tensor<512x2048xf16>) -> tensor<512x2048xf16>
  // FFN ReLU
  %empty25 = tensor.empty() : tensor<512x2048xf16>
  %relu26 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm24 : tensor<512x2048xf16>)
    outs(%empty25 : tensor<512x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x2048xf16>
  // FFN: down projection
  %init27 = tensor.empty() : tensor<512x512xf16>
  %fill28 = linalg.fill ins(%cst : f16) outs(%init27 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm29 = linalg.matmul ins(%relu26, %w_ff0_down : tensor<512x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill28 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // FFN residual add
  %empty30 = tensor.empty() : tensor<512x512xf16>
  %add31 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm29, %add21 : tensor<512x512xf16>, tensor<512x512xf16>)
    outs(%empty30 : tensor<512x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x512xf16>

  // === Transformer Layer 1 ===

  // Q projection
  %init32 = tensor.empty() : tensor<512x512xf16>
  %fill33 = linalg.fill ins(%cst : f16) outs(%init32 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm34 = linalg.matmul ins(%add31, %w_q1 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill33 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // K projection
  %init35 = tensor.empty() : tensor<512x512xf16>
  %fill36 = linalg.fill ins(%cst : f16) outs(%init35 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm37 = linalg.matmul ins(%add31, %w_k1 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill36 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // V projection
  %init38 = tensor.empty() : tensor<512x512xf16>
  %fill39 = linalg.fill ins(%cst : f16) outs(%init38 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm40 = linalg.matmul ins(%add31, %w_v1 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill39 : tensor<512x512xf16>) -> tensor<512x512xf16>

  // Attention scores: Q x K_transposed
  %init41 = tensor.empty() : tensor<512x512xf16>
  %fill42 = linalg.fill ins(%cst : f16) outs(%init41 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm43 = linalg.matmul ins(%mm34, %w_kt1 : tensor<512x512xf16>, tensor<512x512xf16>)
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
  %init46 = tensor.empty() : tensor<512x512xf16>
  %fill47 = linalg.fill ins(%cst : f16) outs(%init46 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm48 = linalg.matmul ins(%relu45, %mm40 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill47 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Output projection
  %init49 = tensor.empty() : tensor<512x512xf16>
  %fill50 = linalg.fill ins(%cst : f16) outs(%init49 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm51 = linalg.matmul ins(%mm48, %w_o1 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill50 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Attention residual add
  %empty52 = tensor.empty() : tensor<512x512xf16>
  %add53 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm51, %add31 : tensor<512x512xf16>, tensor<512x512xf16>)
    outs(%empty52 : tensor<512x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x512xf16>

  // FFN: up projection
  %init54 = tensor.empty() : tensor<512x2048xf16>
  %fill55 = linalg.fill ins(%cst : f16) outs(%init54 : tensor<512x2048xf16>) -> tensor<512x2048xf16>
  %mm56 = linalg.matmul ins(%add53, %w_ff1_up : tensor<512x512xf16>, tensor<512x2048xf16>)
                          outs(%fill55 : tensor<512x2048xf16>) -> tensor<512x2048xf16>
  // FFN ReLU
  %empty57 = tensor.empty() : tensor<512x2048xf16>
  %relu58 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm56 : tensor<512x2048xf16>)
    outs(%empty57 : tensor<512x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x2048xf16>
  // FFN: down projection
  %init59 = tensor.empty() : tensor<512x512xf16>
  %fill60 = linalg.fill ins(%cst : f16) outs(%init59 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm61 = linalg.matmul ins(%relu58, %w_ff1_down : tensor<512x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill60 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // FFN residual add
  %empty62 = tensor.empty() : tensor<512x512xf16>
  %add63 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm61, %add53 : tensor<512x512xf16>, tensor<512x512xf16>)
    outs(%empty62 : tensor<512x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x512xf16>

  // === Transformer Layer 2 ===

  // Q projection
  %init64 = tensor.empty() : tensor<512x512xf16>
  %fill65 = linalg.fill ins(%cst : f16) outs(%init64 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm66 = linalg.matmul ins(%add63, %w_q2 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill65 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // K projection
  %init67 = tensor.empty() : tensor<512x512xf16>
  %fill68 = linalg.fill ins(%cst : f16) outs(%init67 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm69 = linalg.matmul ins(%add63, %w_k2 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill68 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // V projection
  %init70 = tensor.empty() : tensor<512x512xf16>
  %fill71 = linalg.fill ins(%cst : f16) outs(%init70 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm72 = linalg.matmul ins(%add63, %w_v2 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill71 : tensor<512x512xf16>) -> tensor<512x512xf16>

  // Attention scores: Q x K_transposed
  %init73 = tensor.empty() : tensor<512x512xf16>
  %fill74 = linalg.fill ins(%cst : f16) outs(%init73 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm75 = linalg.matmul ins(%mm66, %w_kt2 : tensor<512x512xf16>, tensor<512x512xf16>)
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
  %init78 = tensor.empty() : tensor<512x512xf16>
  %fill79 = linalg.fill ins(%cst : f16) outs(%init78 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm80 = linalg.matmul ins(%relu77, %mm72 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill79 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Output projection
  %init81 = tensor.empty() : tensor<512x512xf16>
  %fill82 = linalg.fill ins(%cst : f16) outs(%init81 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm83 = linalg.matmul ins(%mm80, %w_o2 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill82 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Attention residual add
  %empty84 = tensor.empty() : tensor<512x512xf16>
  %add85 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm83, %add63 : tensor<512x512xf16>, tensor<512x512xf16>)
    outs(%empty84 : tensor<512x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x512xf16>

  // FFN: up projection
  %init86 = tensor.empty() : tensor<512x2048xf16>
  %fill87 = linalg.fill ins(%cst : f16) outs(%init86 : tensor<512x2048xf16>) -> tensor<512x2048xf16>
  %mm88 = linalg.matmul ins(%add85, %w_ff2_up : tensor<512x512xf16>, tensor<512x2048xf16>)
                          outs(%fill87 : tensor<512x2048xf16>) -> tensor<512x2048xf16>
  // FFN ReLU
  %empty89 = tensor.empty() : tensor<512x2048xf16>
  %relu90 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm88 : tensor<512x2048xf16>)
    outs(%empty89 : tensor<512x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x2048xf16>
  // FFN: down projection
  %init91 = tensor.empty() : tensor<512x512xf16>
  %fill92 = linalg.fill ins(%cst : f16) outs(%init91 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm93 = linalg.matmul ins(%relu90, %w_ff2_down : tensor<512x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill92 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // FFN residual add
  %empty94 = tensor.empty() : tensor<512x512xf16>
  %add95 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm93, %add85 : tensor<512x512xf16>, tensor<512x512xf16>)
    outs(%empty94 : tensor<512x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x512xf16>

  // === Transformer Layer 3 ===

  // Q projection
  %init96 = tensor.empty() : tensor<512x512xf16>
  %fill97 = linalg.fill ins(%cst : f16) outs(%init96 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm98 = linalg.matmul ins(%add95, %w_q3 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill97 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // K projection
  %init99 = tensor.empty() : tensor<512x512xf16>
  %fill100 = linalg.fill ins(%cst : f16) outs(%init99 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm101 = linalg.matmul ins(%add95, %w_k3 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill100 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // V projection
  %init102 = tensor.empty() : tensor<512x512xf16>
  %fill103 = linalg.fill ins(%cst : f16) outs(%init102 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm104 = linalg.matmul ins(%add95, %w_v3 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill103 : tensor<512x512xf16>) -> tensor<512x512xf16>

  // Attention scores: Q x K_transposed
  %init105 = tensor.empty() : tensor<512x512xf16>
  %fill106 = linalg.fill ins(%cst : f16) outs(%init105 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm107 = linalg.matmul ins(%mm98, %w_kt3 : tensor<512x512xf16>, tensor<512x512xf16>)
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
  %init110 = tensor.empty() : tensor<512x512xf16>
  %fill111 = linalg.fill ins(%cst : f16) outs(%init110 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm112 = linalg.matmul ins(%relu109, %mm104 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill111 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Output projection
  %init113 = tensor.empty() : tensor<512x512xf16>
  %fill114 = linalg.fill ins(%cst : f16) outs(%init113 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm115 = linalg.matmul ins(%mm112, %w_o3 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill114 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Attention residual add
  %empty116 = tensor.empty() : tensor<512x512xf16>
  %add117 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm115, %add95 : tensor<512x512xf16>, tensor<512x512xf16>)
    outs(%empty116 : tensor<512x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x512xf16>

  // FFN: up projection
  %init118 = tensor.empty() : tensor<512x2048xf16>
  %fill119 = linalg.fill ins(%cst : f16) outs(%init118 : tensor<512x2048xf16>) -> tensor<512x2048xf16>
  %mm120 = linalg.matmul ins(%add117, %w_ff3_up : tensor<512x512xf16>, tensor<512x2048xf16>)
                          outs(%fill119 : tensor<512x2048xf16>) -> tensor<512x2048xf16>
  // FFN ReLU
  %empty121 = tensor.empty() : tensor<512x2048xf16>
  %relu122 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm120 : tensor<512x2048xf16>)
    outs(%empty121 : tensor<512x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x2048xf16>
  // FFN: down projection
  %init123 = tensor.empty() : tensor<512x512xf16>
  %fill124 = linalg.fill ins(%cst : f16) outs(%init123 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm125 = linalg.matmul ins(%relu122, %w_ff3_down : tensor<512x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill124 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // FFN residual add
  %empty126 = tensor.empty() : tensor<512x512xf16>
  %add127 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm125, %add117 : tensor<512x512xf16>, tensor<512x512xf16>)
    outs(%empty126 : tensor<512x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x512xf16>

  // === Transformer Layer 4 ===

  // Q projection
  %init128 = tensor.empty() : tensor<512x512xf16>
  %fill129 = linalg.fill ins(%cst : f16) outs(%init128 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm130 = linalg.matmul ins(%add127, %w_q4 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill129 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // K projection
  %init131 = tensor.empty() : tensor<512x512xf16>
  %fill132 = linalg.fill ins(%cst : f16) outs(%init131 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm133 = linalg.matmul ins(%add127, %w_k4 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill132 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // V projection
  %init134 = tensor.empty() : tensor<512x512xf16>
  %fill135 = linalg.fill ins(%cst : f16) outs(%init134 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm136 = linalg.matmul ins(%add127, %w_v4 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill135 : tensor<512x512xf16>) -> tensor<512x512xf16>

  // Attention scores: Q x K_transposed
  %init137 = tensor.empty() : tensor<512x512xf16>
  %fill138 = linalg.fill ins(%cst : f16) outs(%init137 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm139 = linalg.matmul ins(%mm130, %w_kt4 : tensor<512x512xf16>, tensor<512x512xf16>)
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
  %init142 = tensor.empty() : tensor<512x512xf16>
  %fill143 = linalg.fill ins(%cst : f16) outs(%init142 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm144 = linalg.matmul ins(%relu141, %mm136 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill143 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Output projection
  %init145 = tensor.empty() : tensor<512x512xf16>
  %fill146 = linalg.fill ins(%cst : f16) outs(%init145 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm147 = linalg.matmul ins(%mm144, %w_o4 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill146 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Attention residual add
  %empty148 = tensor.empty() : tensor<512x512xf16>
  %add149 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm147, %add127 : tensor<512x512xf16>, tensor<512x512xf16>)
    outs(%empty148 : tensor<512x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x512xf16>

  // FFN: up projection
  %init150 = tensor.empty() : tensor<512x2048xf16>
  %fill151 = linalg.fill ins(%cst : f16) outs(%init150 : tensor<512x2048xf16>) -> tensor<512x2048xf16>
  %mm152 = linalg.matmul ins(%add149, %w_ff4_up : tensor<512x512xf16>, tensor<512x2048xf16>)
                          outs(%fill151 : tensor<512x2048xf16>) -> tensor<512x2048xf16>
  // FFN ReLU
  %empty153 = tensor.empty() : tensor<512x2048xf16>
  %relu154 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm152 : tensor<512x2048xf16>)
    outs(%empty153 : tensor<512x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x2048xf16>
  // FFN: down projection
  %init155 = tensor.empty() : tensor<512x512xf16>
  %fill156 = linalg.fill ins(%cst : f16) outs(%init155 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm157 = linalg.matmul ins(%relu154, %w_ff4_down : tensor<512x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill156 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // FFN residual add
  %empty158 = tensor.empty() : tensor<512x512xf16>
  %add159 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm157, %add149 : tensor<512x512xf16>, tensor<512x512xf16>)
    outs(%empty158 : tensor<512x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x512xf16>

  // === Transformer Layer 5 ===

  // Q projection
  %init160 = tensor.empty() : tensor<512x512xf16>
  %fill161 = linalg.fill ins(%cst : f16) outs(%init160 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm162 = linalg.matmul ins(%add159, %w_q5 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill161 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // K projection
  %init163 = tensor.empty() : tensor<512x512xf16>
  %fill164 = linalg.fill ins(%cst : f16) outs(%init163 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm165 = linalg.matmul ins(%add159, %w_k5 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill164 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // V projection
  %init166 = tensor.empty() : tensor<512x512xf16>
  %fill167 = linalg.fill ins(%cst : f16) outs(%init166 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm168 = linalg.matmul ins(%add159, %w_v5 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill167 : tensor<512x512xf16>) -> tensor<512x512xf16>

  // Attention scores: Q x K_transposed
  %init169 = tensor.empty() : tensor<512x512xf16>
  %fill170 = linalg.fill ins(%cst : f16) outs(%init169 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm171 = linalg.matmul ins(%mm162, %w_kt5 : tensor<512x512xf16>, tensor<512x512xf16>)
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
  %init174 = tensor.empty() : tensor<512x512xf16>
  %fill175 = linalg.fill ins(%cst : f16) outs(%init174 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm176 = linalg.matmul ins(%relu173, %mm168 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill175 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Output projection
  %init177 = tensor.empty() : tensor<512x512xf16>
  %fill178 = linalg.fill ins(%cst : f16) outs(%init177 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm179 = linalg.matmul ins(%mm176, %w_o5 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill178 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Attention residual add
  %empty180 = tensor.empty() : tensor<512x512xf16>
  %add181 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm179, %add159 : tensor<512x512xf16>, tensor<512x512xf16>)
    outs(%empty180 : tensor<512x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x512xf16>

  // FFN: up projection
  %init182 = tensor.empty() : tensor<512x2048xf16>
  %fill183 = linalg.fill ins(%cst : f16) outs(%init182 : tensor<512x2048xf16>) -> tensor<512x2048xf16>
  %mm184 = linalg.matmul ins(%add181, %w_ff5_up : tensor<512x512xf16>, tensor<512x2048xf16>)
                          outs(%fill183 : tensor<512x2048xf16>) -> tensor<512x2048xf16>
  // FFN ReLU
  %empty185 = tensor.empty() : tensor<512x2048xf16>
  %relu186 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm184 : tensor<512x2048xf16>)
    outs(%empty185 : tensor<512x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x2048xf16>
  // FFN: down projection
  %init187 = tensor.empty() : tensor<512x512xf16>
  %fill188 = linalg.fill ins(%cst : f16) outs(%init187 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm189 = linalg.matmul ins(%relu186, %w_ff5_down : tensor<512x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill188 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // FFN residual add
  %empty190 = tensor.empty() : tensor<512x512xf16>
  %add191 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm189, %add181 : tensor<512x512xf16>, tensor<512x512xf16>)
    outs(%empty190 : tensor<512x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x512xf16>

  // === Transformer Layer 6 ===

  // Q projection
  %init192 = tensor.empty() : tensor<512x512xf16>
  %fill193 = linalg.fill ins(%cst : f16) outs(%init192 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm194 = linalg.matmul ins(%add191, %w_q6 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill193 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // K projection
  %init195 = tensor.empty() : tensor<512x512xf16>
  %fill196 = linalg.fill ins(%cst : f16) outs(%init195 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm197 = linalg.matmul ins(%add191, %w_k6 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill196 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // V projection
  %init198 = tensor.empty() : tensor<512x512xf16>
  %fill199 = linalg.fill ins(%cst : f16) outs(%init198 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm200 = linalg.matmul ins(%add191, %w_v6 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill199 : tensor<512x512xf16>) -> tensor<512x512xf16>

  // Attention scores: Q x K_transposed
  %init201 = tensor.empty() : tensor<512x512xf16>
  %fill202 = linalg.fill ins(%cst : f16) outs(%init201 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm203 = linalg.matmul ins(%mm194, %w_kt6 : tensor<512x512xf16>, tensor<512x512xf16>)
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
  %init206 = tensor.empty() : tensor<512x512xf16>
  %fill207 = linalg.fill ins(%cst : f16) outs(%init206 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm208 = linalg.matmul ins(%relu205, %mm200 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill207 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Output projection
  %init209 = tensor.empty() : tensor<512x512xf16>
  %fill210 = linalg.fill ins(%cst : f16) outs(%init209 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm211 = linalg.matmul ins(%mm208, %w_o6 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill210 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Attention residual add
  %empty212 = tensor.empty() : tensor<512x512xf16>
  %add213 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm211, %add191 : tensor<512x512xf16>, tensor<512x512xf16>)
    outs(%empty212 : tensor<512x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x512xf16>

  // FFN: up projection
  %init214 = tensor.empty() : tensor<512x2048xf16>
  %fill215 = linalg.fill ins(%cst : f16) outs(%init214 : tensor<512x2048xf16>) -> tensor<512x2048xf16>
  %mm216 = linalg.matmul ins(%add213, %w_ff6_up : tensor<512x512xf16>, tensor<512x2048xf16>)
                          outs(%fill215 : tensor<512x2048xf16>) -> tensor<512x2048xf16>
  // FFN ReLU
  %empty217 = tensor.empty() : tensor<512x2048xf16>
  %relu218 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm216 : tensor<512x2048xf16>)
    outs(%empty217 : tensor<512x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x2048xf16>
  // FFN: down projection
  %init219 = tensor.empty() : tensor<512x512xf16>
  %fill220 = linalg.fill ins(%cst : f16) outs(%init219 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm221 = linalg.matmul ins(%relu218, %w_ff6_down : tensor<512x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill220 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // FFN residual add
  %empty222 = tensor.empty() : tensor<512x512xf16>
  %add223 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm221, %add213 : tensor<512x512xf16>, tensor<512x512xf16>)
    outs(%empty222 : tensor<512x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x512xf16>

  // === Transformer Layer 7 ===

  // Q projection
  %init224 = tensor.empty() : tensor<512x512xf16>
  %fill225 = linalg.fill ins(%cst : f16) outs(%init224 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm226 = linalg.matmul ins(%add223, %w_q7 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill225 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // K projection
  %init227 = tensor.empty() : tensor<512x512xf16>
  %fill228 = linalg.fill ins(%cst : f16) outs(%init227 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm229 = linalg.matmul ins(%add223, %w_k7 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill228 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // V projection
  %init230 = tensor.empty() : tensor<512x512xf16>
  %fill231 = linalg.fill ins(%cst : f16) outs(%init230 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm232 = linalg.matmul ins(%add223, %w_v7 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill231 : tensor<512x512xf16>) -> tensor<512x512xf16>

  // Attention scores: Q x K_transposed
  %init233 = tensor.empty() : tensor<512x512xf16>
  %fill234 = linalg.fill ins(%cst : f16) outs(%init233 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm235 = linalg.matmul ins(%mm226, %w_kt7 : tensor<512x512xf16>, tensor<512x512xf16>)
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
  %init238 = tensor.empty() : tensor<512x512xf16>
  %fill239 = linalg.fill ins(%cst : f16) outs(%init238 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm240 = linalg.matmul ins(%relu237, %mm232 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill239 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Output projection
  %init241 = tensor.empty() : tensor<512x512xf16>
  %fill242 = linalg.fill ins(%cst : f16) outs(%init241 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm243 = linalg.matmul ins(%mm240, %w_o7 : tensor<512x512xf16>, tensor<512x512xf16>)
                          outs(%fill242 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // Attention residual add
  %empty244 = tensor.empty() : tensor<512x512xf16>
  %add245 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm243, %add223 : tensor<512x512xf16>, tensor<512x512xf16>)
    outs(%empty244 : tensor<512x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x512xf16>

  // FFN: up projection
  %init246 = tensor.empty() : tensor<512x2048xf16>
  %fill247 = linalg.fill ins(%cst : f16) outs(%init246 : tensor<512x2048xf16>) -> tensor<512x2048xf16>
  %mm248 = linalg.matmul ins(%add245, %w_ff7_up : tensor<512x512xf16>, tensor<512x2048xf16>)
                          outs(%fill247 : tensor<512x2048xf16>) -> tensor<512x2048xf16>
  // FFN ReLU
  %empty249 = tensor.empty() : tensor<512x2048xf16>
  %relu250 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm248 : tensor<512x2048xf16>)
    outs(%empty249 : tensor<512x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<512x2048xf16>
  // FFN: down projection
  %init251 = tensor.empty() : tensor<512x512xf16>
  %fill252 = linalg.fill ins(%cst : f16) outs(%init251 : tensor<512x512xf16>) -> tensor<512x512xf16>
  %mm253 = linalg.matmul ins(%relu250, %w_ff7_down : tensor<512x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill252 : tensor<512x512xf16>) -> tensor<512x512xf16>
  // FFN residual add
  %empty254 = tensor.empty() : tensor<512x512xf16>
  %add255 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm253, %add245 : tensor<512x512xf16>, tensor<512x512xf16>)
    outs(%empty254 : tensor<512x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<512x512xf16>

  return %add255 : tensor<512x512xf16>
}
