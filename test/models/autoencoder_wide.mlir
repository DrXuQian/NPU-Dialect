func.func @autoencoder_wide(
    %input: tensor<1x2048xf16>,
    %w_enc0: tensor<2048x2048xf16>,
    %w_enc1: tensor<2048x1024xf16>,
    %w_enc2: tensor<1024x512xf16>,
    %w_dec0: tensor<512x1024xf16>,
    %w_dec1: tensor<1024x2048xf16>,
    %w_dec2: tensor<2048x2048xf16>) -> tensor<1x2048xf16> {
  %cst = arith.constant 0.0 : f16

  // === Encoder ===
  // enc0: 2048->2048
  %init0 = tensor.empty() : tensor<1x2048xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init0 : tensor<1x2048xf16>) -> tensor<1x2048xf16>
  %mm2 = linalg.matmul ins(%input, %w_enc0 : tensor<1x2048xf16>, tensor<2048x2048xf16>)
                          outs(%fill1 : tensor<1x2048xf16>) -> tensor<1x2048xf16>
  %empty3 = tensor.empty() : tensor<1x2048xf16>
  %relu4 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm2 : tensor<1x2048xf16>)
    outs(%empty3 : tensor<1x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048xf16>
  // enc1: 2048->1024
  %init5 = tensor.empty() : tensor<1x1024xf16>
  %fill6 = linalg.fill ins(%cst : f16) outs(%init5 : tensor<1x1024xf16>) -> tensor<1x1024xf16>
  %mm7 = linalg.matmul ins(%relu4, %w_enc1 : tensor<1x2048xf16>, tensor<2048x1024xf16>)
                          outs(%fill6 : tensor<1x1024xf16>) -> tensor<1x1024xf16>
  %empty8 = tensor.empty() : tensor<1x1024xf16>
  %relu9 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm7 : tensor<1x1024xf16>)
    outs(%empty8 : tensor<1x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024xf16>
  // enc2: 1024->512
  %init10 = tensor.empty() : tensor<1x512xf16>
  %fill11 = linalg.fill ins(%cst : f16) outs(%init10 : tensor<1x512xf16>) -> tensor<1x512xf16>
  %mm12 = linalg.matmul ins(%relu9, %w_enc2 : tensor<1x1024xf16>, tensor<1024x512xf16>)
                          outs(%fill11 : tensor<1x512xf16>) -> tensor<1x512xf16>
  %empty13 = tensor.empty() : tensor<1x512xf16>
  %relu14 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm12 : tensor<1x512xf16>)
    outs(%empty13 : tensor<1x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512xf16>

  // === Decoder ===
  // dec0: 512->1024
  %init15 = tensor.empty() : tensor<1x1024xf16>
  %fill16 = linalg.fill ins(%cst : f16) outs(%init15 : tensor<1x1024xf16>) -> tensor<1x1024xf16>
  %mm17 = linalg.matmul ins(%relu14, %w_dec0 : tensor<1x512xf16>, tensor<512x1024xf16>)
                          outs(%fill16 : tensor<1x1024xf16>) -> tensor<1x1024xf16>
  %empty18 = tensor.empty() : tensor<1x1024xf16>
  %relu19 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm17 : tensor<1x1024xf16>)
    outs(%empty18 : tensor<1x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024xf16>
  // dec1: 1024->2048
  %init20 = tensor.empty() : tensor<1x2048xf16>
  %fill21 = linalg.fill ins(%cst : f16) outs(%init20 : tensor<1x2048xf16>) -> tensor<1x2048xf16>
  %mm22 = linalg.matmul ins(%relu19, %w_dec1 : tensor<1x1024xf16>, tensor<1024x2048xf16>)
                          outs(%fill21 : tensor<1x2048xf16>) -> tensor<1x2048xf16>
  %empty23 = tensor.empty() : tensor<1x2048xf16>
  %relu24 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm22 : tensor<1x2048xf16>)
    outs(%empty23 : tensor<1x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048xf16>
  // dec2: 2048->2048
  %init25 = tensor.empty() : tensor<1x2048xf16>
  %fill26 = linalg.fill ins(%cst : f16) outs(%init25 : tensor<1x2048xf16>) -> tensor<1x2048xf16>
  %mm27 = linalg.matmul ins(%relu24, %w_dec2 : tensor<1x2048xf16>, tensor<2048x2048xf16>)
                          outs(%fill26 : tensor<1x2048xf16>) -> tensor<1x2048xf16>
  return %mm27 : tensor<1x2048xf16>
}
