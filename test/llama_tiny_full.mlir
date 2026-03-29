func.func @llama_tiny(
    %input: tensor<128x256xf16>,
    %w_q0: tensor<256x256xf16>,
    %w_k0: tensor<256x256xf16>,
    %w_v0: tensor<256x256xf16>,
    %w_kt0: tensor<256x128xf16>,
    %w_o0: tensor<256x256xf16>,
    %w_ff0_up: tensor<256x512xf16>,
    %w_ff0_down: tensor<512x256xf16>,
    %w_q1: tensor<256x256xf16>,
    %w_k1: tensor<256x256xf16>,
    %w_v1: tensor<256x256xf16>,
    %w_kt1: tensor<256x128xf16>,
    %w_o1: tensor<256x256xf16>,
    %w_ff1_up: tensor<256x512xf16>,
    %w_ff1_down: tensor<512x256xf16>) -> tensor<128x256xf16> {
  %cst = arith.constant 0.0 : f16

  // === Transformer Layer 0 ===

  // Q projection
  %init0 = tensor.empty() : tensor<128x256xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init0 : tensor<128x256xf16>) -> tensor<128x256xf16>
  %mm2 = linalg.matmul ins(%input, %w_q0 : tensor<128x256xf16>, tensor<256x256xf16>)
                          outs(%fill1 : tensor<128x256xf16>) -> tensor<128x256xf16>
  // K projection
  %init3 = tensor.empty() : tensor<128x256xf16>
  %fill4 = linalg.fill ins(%cst : f16) outs(%init3 : tensor<128x256xf16>) -> tensor<128x256xf16>
  %mm5 = linalg.matmul ins(%input, %w_k0 : tensor<128x256xf16>, tensor<256x256xf16>)
                          outs(%fill4 : tensor<128x256xf16>) -> tensor<128x256xf16>
  // V projection
  %init6 = tensor.empty() : tensor<128x256xf16>
  %fill7 = linalg.fill ins(%cst : f16) outs(%init6 : tensor<128x256xf16>) -> tensor<128x256xf16>
  %mm8 = linalg.matmul ins(%input, %w_v0 : tensor<128x256xf16>, tensor<256x256xf16>)
                          outs(%fill7 : tensor<128x256xf16>) -> tensor<128x256xf16>

  // Attention scores: Q x K_transposed -> [128,128]
  %init9 = tensor.empty() : tensor<128x128xf16>
  %fill10 = linalg.fill ins(%cst : f16) outs(%init9 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm11 = linalg.matmul ins(%mm2, %w_kt0 : tensor<128x256xf16>, tensor<256x128xf16>)
                          outs(%fill10 : tensor<128x128xf16>) -> tensor<128x128xf16>
  // Softmax approximation (relu)
  %empty12 = tensor.empty() : tensor<128x128xf16>
  %relu13 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm11 : tensor<128x128xf16>)
    outs(%empty12 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>

  // Attention output: softmax_scores x V
  %init14 = tensor.empty() : tensor<128x256xf16>
  %fill15 = linalg.fill ins(%cst : f16) outs(%init14 : tensor<128x256xf16>) -> tensor<128x256xf16>
  %mm16 = linalg.matmul ins(%relu13, %mm8 : tensor<128x128xf16>, tensor<128x256xf16>)
                          outs(%fill15 : tensor<128x256xf16>) -> tensor<128x256xf16>
  // Output projection
  %init17 = tensor.empty() : tensor<128x256xf16>
  %fill18 = linalg.fill ins(%cst : f16) outs(%init17 : tensor<128x256xf16>) -> tensor<128x256xf16>
  %mm19 = linalg.matmul ins(%mm16, %w_o0 : tensor<128x256xf16>, tensor<256x256xf16>)
                          outs(%fill18 : tensor<128x256xf16>) -> tensor<128x256xf16>
  // Attention residual add
  %empty20 = tensor.empty() : tensor<128x256xf16>
  %add21 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm19, %input : tensor<128x256xf16>, tensor<128x256xf16>)
    outs(%empty20 : tensor<128x256xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x256xf16>

  // FFN: up projection [128,256] x [256,512] -> [128,512]
  %init22 = tensor.empty() : tensor<128x512xf16>
  %fill23 = linalg.fill ins(%cst : f16) outs(%init22 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm24 = linalg.matmul ins(%add21, %w_ff0_up : tensor<128x256xf16>, tensor<256x512xf16>)
                          outs(%fill23 : tensor<128x512xf16>) -> tensor<128x512xf16>
  // FFN ReLU
  %empty25 = tensor.empty() : tensor<128x512xf16>
  %relu26 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm24 : tensor<128x512xf16>)
    outs(%empty25 : tensor<128x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x512xf16>
  // FFN: down projection [128,512] x [512,256] -> [128,256]
  %init27 = tensor.empty() : tensor<128x256xf16>
  %fill28 = linalg.fill ins(%cst : f16) outs(%init27 : tensor<128x256xf16>) -> tensor<128x256xf16>
  %mm29 = linalg.matmul ins(%relu26, %w_ff0_down : tensor<128x512xf16>, tensor<512x256xf16>)
                          outs(%fill28 : tensor<128x256xf16>) -> tensor<128x256xf16>
  // FFN residual add
  %empty30 = tensor.empty() : tensor<128x256xf16>
  %add31 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm29, %add21 : tensor<128x256xf16>, tensor<128x256xf16>)
    outs(%empty30 : tensor<128x256xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x256xf16>

  // === Transformer Layer 1 ===

  // Q projection
  %init32 = tensor.empty() : tensor<128x256xf16>
  %fill33 = linalg.fill ins(%cst : f16) outs(%init32 : tensor<128x256xf16>) -> tensor<128x256xf16>
  %mm34 = linalg.matmul ins(%add31, %w_q1 : tensor<128x256xf16>, tensor<256x256xf16>)
                          outs(%fill33 : tensor<128x256xf16>) -> tensor<128x256xf16>
  // K projection
  %init35 = tensor.empty() : tensor<128x256xf16>
  %fill36 = linalg.fill ins(%cst : f16) outs(%init35 : tensor<128x256xf16>) -> tensor<128x256xf16>
  %mm37 = linalg.matmul ins(%add31, %w_k1 : tensor<128x256xf16>, tensor<256x256xf16>)
                          outs(%fill36 : tensor<128x256xf16>) -> tensor<128x256xf16>
  // V projection
  %init38 = tensor.empty() : tensor<128x256xf16>
  %fill39 = linalg.fill ins(%cst : f16) outs(%init38 : tensor<128x256xf16>) -> tensor<128x256xf16>
  %mm40 = linalg.matmul ins(%add31, %w_v1 : tensor<128x256xf16>, tensor<256x256xf16>)
                          outs(%fill39 : tensor<128x256xf16>) -> tensor<128x256xf16>

  // Attention scores: Q x K_transposed -> [128,128]
  %init41 = tensor.empty() : tensor<128x128xf16>
  %fill42 = linalg.fill ins(%cst : f16) outs(%init41 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm43 = linalg.matmul ins(%mm34, %w_kt1 : tensor<128x256xf16>, tensor<256x128xf16>)
                          outs(%fill42 : tensor<128x128xf16>) -> tensor<128x128xf16>
  // Softmax approximation (relu)
  %empty44 = tensor.empty() : tensor<128x128xf16>
  %relu45 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm43 : tensor<128x128xf16>)
    outs(%empty44 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>

  // Attention output: softmax_scores x V
  %init46 = tensor.empty() : tensor<128x256xf16>
  %fill47 = linalg.fill ins(%cst : f16) outs(%init46 : tensor<128x256xf16>) -> tensor<128x256xf16>
  %mm48 = linalg.matmul ins(%relu45, %mm40 : tensor<128x128xf16>, tensor<128x256xf16>)
                          outs(%fill47 : tensor<128x256xf16>) -> tensor<128x256xf16>
  // Output projection
  %init49 = tensor.empty() : tensor<128x256xf16>
  %fill50 = linalg.fill ins(%cst : f16) outs(%init49 : tensor<128x256xf16>) -> tensor<128x256xf16>
  %mm51 = linalg.matmul ins(%mm48, %w_o1 : tensor<128x256xf16>, tensor<256x256xf16>)
                          outs(%fill50 : tensor<128x256xf16>) -> tensor<128x256xf16>
  // Attention residual add
  %empty52 = tensor.empty() : tensor<128x256xf16>
  %add53 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm51, %add31 : tensor<128x256xf16>, tensor<128x256xf16>)
    outs(%empty52 : tensor<128x256xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x256xf16>

  // FFN: up projection [128,256] x [256,512] -> [128,512]
  %init54 = tensor.empty() : tensor<128x512xf16>
  %fill55 = linalg.fill ins(%cst : f16) outs(%init54 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm56 = linalg.matmul ins(%add53, %w_ff1_up : tensor<128x256xf16>, tensor<256x512xf16>)
                          outs(%fill55 : tensor<128x512xf16>) -> tensor<128x512xf16>
  // FFN ReLU
  %empty57 = tensor.empty() : tensor<128x512xf16>
  %relu58 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm56 : tensor<128x512xf16>)
    outs(%empty57 : tensor<128x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x512xf16>
  // FFN: down projection [128,512] x [512,256] -> [128,256]
  %init59 = tensor.empty() : tensor<128x256xf16>
  %fill60 = linalg.fill ins(%cst : f16) outs(%init59 : tensor<128x256xf16>) -> tensor<128x256xf16>
  %mm61 = linalg.matmul ins(%relu58, %w_ff1_down : tensor<128x512xf16>, tensor<512x256xf16>)
                          outs(%fill60 : tensor<128x256xf16>) -> tensor<128x256xf16>
  // FFN residual add
  %empty62 = tensor.empty() : tensor<128x256xf16>
  %add63 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm61, %add53 : tensor<128x256xf16>, tensor<128x256xf16>)
    outs(%empty62 : tensor<128x256xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x256xf16>

  return %add63 : tensor<128x256xf16>
}
