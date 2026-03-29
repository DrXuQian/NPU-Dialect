func.func @autoencoder_deep(
    %input: tensor<1x1024xf16>,
    %w_enc0: tensor<1024x1024xf16>,
    %w_enc1: tensor<1024x512xf16>,
    %w_enc2: tensor<512x256xf16>,
    %w_enc3: tensor<256x128xf16>,
    %w_enc4: tensor<128x64xf16>,
    %w_enc5: tensor<64x32xf16>,
    %w_dec0: tensor<32x64xf16>,
    %w_dec1: tensor<64x128xf16>,
    %w_dec2: tensor<128x256xf16>,
    %w_dec3: tensor<256x512xf16>,
    %w_dec4: tensor<512x1024xf16>,
    %w_dec5: tensor<1024x1024xf16>) -> tensor<1x1024xf16> {
  %cst = arith.constant 0.0 : f16

  // === Encoder ===
  // enc0: 1024->1024
  %init0 = tensor.empty() : tensor<1x1024xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init0 : tensor<1x1024xf16>) -> tensor<1x1024xf16>
  %mm2 = linalg.matmul ins(%input, %w_enc0 : tensor<1x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill1 : tensor<1x1024xf16>) -> tensor<1x1024xf16>
  %empty3 = tensor.empty() : tensor<1x1024xf16>
  %relu4 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm2 : tensor<1x1024xf16>)
    outs(%empty3 : tensor<1x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024xf16>
  // enc1: 1024->512
  %init5 = tensor.empty() : tensor<1x512xf16>
  %fill6 = linalg.fill ins(%cst : f16) outs(%init5 : tensor<1x512xf16>) -> tensor<1x512xf16>
  %mm7 = linalg.matmul ins(%relu4, %w_enc1 : tensor<1x1024xf16>, tensor<1024x512xf16>)
                          outs(%fill6 : tensor<1x512xf16>) -> tensor<1x512xf16>
  %empty8 = tensor.empty() : tensor<1x512xf16>
  %relu9 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm7 : tensor<1x512xf16>)
    outs(%empty8 : tensor<1x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512xf16>
  // enc2: 512->256
  %init10 = tensor.empty() : tensor<1x256xf16>
  %fill11 = linalg.fill ins(%cst : f16) outs(%init10 : tensor<1x256xf16>) -> tensor<1x256xf16>
  %mm12 = linalg.matmul ins(%relu9, %w_enc2 : tensor<1x512xf16>, tensor<512x256xf16>)
                          outs(%fill11 : tensor<1x256xf16>) -> tensor<1x256xf16>
  %empty13 = tensor.empty() : tensor<1x256xf16>
  %relu14 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm12 : tensor<1x256xf16>)
    outs(%empty13 : tensor<1x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256xf16>
  // enc3: 256->128
  %init15 = tensor.empty() : tensor<1x128xf16>
  %fill16 = linalg.fill ins(%cst : f16) outs(%init15 : tensor<1x128xf16>) -> tensor<1x128xf16>
  %mm17 = linalg.matmul ins(%relu14, %w_enc3 : tensor<1x256xf16>, tensor<256x128xf16>)
                          outs(%fill16 : tensor<1x128xf16>) -> tensor<1x128xf16>
  %empty18 = tensor.empty() : tensor<1x128xf16>
  %relu19 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm17 : tensor<1x128xf16>)
    outs(%empty18 : tensor<1x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128xf16>
  // enc4: 128->64
  %init20 = tensor.empty() : tensor<1x64xf16>
  %fill21 = linalg.fill ins(%cst : f16) outs(%init20 : tensor<1x64xf16>) -> tensor<1x64xf16>
  %mm22 = linalg.matmul ins(%relu19, %w_enc4 : tensor<1x128xf16>, tensor<128x64xf16>)
                          outs(%fill21 : tensor<1x64xf16>) -> tensor<1x64xf16>
  %empty23 = tensor.empty() : tensor<1x64xf16>
  %relu24 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm22 : tensor<1x64xf16>)
    outs(%empty23 : tensor<1x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64xf16>
  // enc5: 64->32
  %init25 = tensor.empty() : tensor<1x32xf16>
  %fill26 = linalg.fill ins(%cst : f16) outs(%init25 : tensor<1x32xf16>) -> tensor<1x32xf16>
  %mm27 = linalg.matmul ins(%relu24, %w_enc5 : tensor<1x64xf16>, tensor<64x32xf16>)
                          outs(%fill26 : tensor<1x32xf16>) -> tensor<1x32xf16>
  %empty28 = tensor.empty() : tensor<1x32xf16>
  %relu29 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm27 : tensor<1x32xf16>)
    outs(%empty28 : tensor<1x32xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32xf16>

  // === Decoder ===
  // dec0: 32->64
  %init30 = tensor.empty() : tensor<1x64xf16>
  %fill31 = linalg.fill ins(%cst : f16) outs(%init30 : tensor<1x64xf16>) -> tensor<1x64xf16>
  %mm32 = linalg.matmul ins(%relu29, %w_dec0 : tensor<1x32xf16>, tensor<32x64xf16>)
                          outs(%fill31 : tensor<1x64xf16>) -> tensor<1x64xf16>
  %empty33 = tensor.empty() : tensor<1x64xf16>
  %relu34 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm32 : tensor<1x64xf16>)
    outs(%empty33 : tensor<1x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64xf16>
  // dec1: 64->128
  %init35 = tensor.empty() : tensor<1x128xf16>
  %fill36 = linalg.fill ins(%cst : f16) outs(%init35 : tensor<1x128xf16>) -> tensor<1x128xf16>
  %mm37 = linalg.matmul ins(%relu34, %w_dec1 : tensor<1x64xf16>, tensor<64x128xf16>)
                          outs(%fill36 : tensor<1x128xf16>) -> tensor<1x128xf16>
  %empty38 = tensor.empty() : tensor<1x128xf16>
  %relu39 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm37 : tensor<1x128xf16>)
    outs(%empty38 : tensor<1x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128xf16>
  // dec2: 128->256
  %init40 = tensor.empty() : tensor<1x256xf16>
  %fill41 = linalg.fill ins(%cst : f16) outs(%init40 : tensor<1x256xf16>) -> tensor<1x256xf16>
  %mm42 = linalg.matmul ins(%relu39, %w_dec2 : tensor<1x128xf16>, tensor<128x256xf16>)
                          outs(%fill41 : tensor<1x256xf16>) -> tensor<1x256xf16>
  %empty43 = tensor.empty() : tensor<1x256xf16>
  %relu44 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm42 : tensor<1x256xf16>)
    outs(%empty43 : tensor<1x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256xf16>
  // dec3: 256->512
  %init45 = tensor.empty() : tensor<1x512xf16>
  %fill46 = linalg.fill ins(%cst : f16) outs(%init45 : tensor<1x512xf16>) -> tensor<1x512xf16>
  %mm47 = linalg.matmul ins(%relu44, %w_dec3 : tensor<1x256xf16>, tensor<256x512xf16>)
                          outs(%fill46 : tensor<1x512xf16>) -> tensor<1x512xf16>
  %empty48 = tensor.empty() : tensor<1x512xf16>
  %relu49 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm47 : tensor<1x512xf16>)
    outs(%empty48 : tensor<1x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512xf16>
  // dec4: 512->1024
  %init50 = tensor.empty() : tensor<1x1024xf16>
  %fill51 = linalg.fill ins(%cst : f16) outs(%init50 : tensor<1x1024xf16>) -> tensor<1x1024xf16>
  %mm52 = linalg.matmul ins(%relu49, %w_dec4 : tensor<1x512xf16>, tensor<512x1024xf16>)
                          outs(%fill51 : tensor<1x1024xf16>) -> tensor<1x1024xf16>
  %empty53 = tensor.empty() : tensor<1x1024xf16>
  %relu54 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm52 : tensor<1x1024xf16>)
    outs(%empty53 : tensor<1x1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024xf16>
  // dec5: 1024->1024
  %init55 = tensor.empty() : tensor<1x1024xf16>
  %fill56 = linalg.fill ins(%cst : f16) outs(%init55 : tensor<1x1024xf16>) -> tensor<1x1024xf16>
  %mm57 = linalg.matmul ins(%relu54, %w_dec5 : tensor<1x1024xf16>, tensor<1024x1024xf16>)
                          outs(%fill56 : tensor<1x1024xf16>) -> tensor<1x1024xf16>
  return %mm57 : tensor<1x1024xf16>
}
