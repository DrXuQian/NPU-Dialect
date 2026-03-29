func.func @pspnet(
    %input: tensor<1x3x473x473xf16>,
    %w_enc0: tensor<64x3x7x7xf16>,
    %w_enc1: tensor<64x64x3x3xf16>,
    %w_enc2: tensor<256x64x3x3xf16>,
    %w_enc3: tensor<256x256x3x3xf16>,
    %w_enc4: tensor<512x256x3x3xf16>,
    %w_enc5: tensor<512x512x3x3xf16>,
    %w_enc6: tensor<1024x512x3x3xf16>,
    %w_enc7: tensor<1024x1024x3x3xf16>,
    %w_enc8: tensor<2048x1024x3x3xf16>,
    %w_dec0: tensor<512x2048x1x1xf16>,
    %w_dec1: tensor<512x512x3x3xf16>,
    %w_dec2: tensor<256x512x3x3xf16>,
    %w_out: tensor<21x256x1x1xf16>) -> tensor<1x21x30x30xf16> {
  %cst = arith.constant 0.0 : f16

  // === Encoder ===
  // enc0: 7x7 s2 3->64 473x473
  %pad0 = tensor.pad %input low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x473x473xf16> to tensor<1x3x479x479xf16>
  %init1 = tensor.empty() : tensor<1x64x237x237xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x64x237x237xf16>) -> tensor<1x64x237x237xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad0, %w_enc0 : tensor<1x3x479x479xf16>, tensor<64x3x7x7xf16>)
    outs(%fill2 : tensor<1x64x237x237xf16>) -> tensor<1x64x237x237xf16>
  %empty4 = tensor.empty() : tensor<1x64x237x237xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x64x237x237xf16>)
    outs(%empty4 : tensor<1x64x237x237xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x237x237xf16>
  // enc1: 3x3 s2 64->64 237x237
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x237x237xf16> to tensor<1x64x239x239xf16>
  %init7 = tensor.empty() : tensor<1x64x119x119xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x64x119x119xf16>) -> tensor<1x64x119x119xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad6, %w_enc1 : tensor<1x64x239x239xf16>, tensor<64x64x3x3xf16>)
    outs(%fill8 : tensor<1x64x119x119xf16>) -> tensor<1x64x119x119xf16>
  %empty10 = tensor.empty() : tensor<1x64x119x119xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x64x119x119xf16>)
    outs(%empty10 : tensor<1x64x119x119xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x119x119xf16>
  // enc2: 3x3 s1 64->256 119x119
  %pad12 = tensor.pad %relu11 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x119x119xf16> to tensor<1x64x121x121xf16>
  %init13 = tensor.empty() : tensor<1x256x119x119xf16>
  %fill14 = linalg.fill ins(%cst : f16) outs(%init13 : tensor<1x256x119x119xf16>) -> tensor<1x256x119x119xf16>
  %conv15 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad12, %w_enc2 : tensor<1x64x121x121xf16>, tensor<256x64x3x3xf16>)
    outs(%fill14 : tensor<1x256x119x119xf16>) -> tensor<1x256x119x119xf16>
  %empty16 = tensor.empty() : tensor<1x256x119x119xf16>
  %relu17 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv15 : tensor<1x256x119x119xf16>)
    outs(%empty16 : tensor<1x256x119x119xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x119x119xf16>
  // enc3: 3x3 s1 256->256 119x119
  %pad18 = tensor.pad %relu17 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x119x119xf16> to tensor<1x256x121x121xf16>
  %init19 = tensor.empty() : tensor<1x256x119x119xf16>
  %fill20 = linalg.fill ins(%cst : f16) outs(%init19 : tensor<1x256x119x119xf16>) -> tensor<1x256x119x119xf16>
  %conv21 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad18, %w_enc3 : tensor<1x256x121x121xf16>, tensor<256x256x3x3xf16>)
    outs(%fill20 : tensor<1x256x119x119xf16>) -> tensor<1x256x119x119xf16>
  %empty22 = tensor.empty() : tensor<1x256x119x119xf16>
  %relu23 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv21 : tensor<1x256x119x119xf16>)
    outs(%empty22 : tensor<1x256x119x119xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x119x119xf16>
  // enc4: 3x3 s2 256->512 119x119
  %pad24 = tensor.pad %relu23 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x119x119xf16> to tensor<1x256x121x121xf16>
  %init25 = tensor.empty() : tensor<1x512x60x60xf16>
  %fill26 = linalg.fill ins(%cst : f16) outs(%init25 : tensor<1x512x60x60xf16>) -> tensor<1x512x60x60xf16>
  %conv27 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad24, %w_enc4 : tensor<1x256x121x121xf16>, tensor<512x256x3x3xf16>)
    outs(%fill26 : tensor<1x512x60x60xf16>) -> tensor<1x512x60x60xf16>
  %empty28 = tensor.empty() : tensor<1x512x60x60xf16>
  %relu29 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv27 : tensor<1x512x60x60xf16>)
    outs(%empty28 : tensor<1x512x60x60xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x60x60xf16>
  // enc5: 3x3 s1 512->512 60x60
  %pad30 = tensor.pad %relu29 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x60x60xf16> to tensor<1x512x62x62xf16>
  %init31 = tensor.empty() : tensor<1x512x60x60xf16>
  %fill32 = linalg.fill ins(%cst : f16) outs(%init31 : tensor<1x512x60x60xf16>) -> tensor<1x512x60x60xf16>
  %conv33 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad30, %w_enc5 : tensor<1x512x62x62xf16>, tensor<512x512x3x3xf16>)
    outs(%fill32 : tensor<1x512x60x60xf16>) -> tensor<1x512x60x60xf16>
  %empty34 = tensor.empty() : tensor<1x512x60x60xf16>
  %relu35 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv33 : tensor<1x512x60x60xf16>)
    outs(%empty34 : tensor<1x512x60x60xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x60x60xf16>
  // enc6: 3x3 s2 512->1024 60x60
  %pad36 = tensor.pad %relu35 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x60x60xf16> to tensor<1x512x62x62xf16>
  %init37 = tensor.empty() : tensor<1x1024x30x30xf16>
  %fill38 = linalg.fill ins(%cst : f16) outs(%init37 : tensor<1x1024x30x30xf16>) -> tensor<1x1024x30x30xf16>
  %conv39 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad36, %w_enc6 : tensor<1x512x62x62xf16>, tensor<1024x512x3x3xf16>)
    outs(%fill38 : tensor<1x1024x30x30xf16>) -> tensor<1x1024x30x30xf16>
  %empty40 = tensor.empty() : tensor<1x1024x30x30xf16>
  %relu41 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv39 : tensor<1x1024x30x30xf16>)
    outs(%empty40 : tensor<1x1024x30x30xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x30x30xf16>
  // enc7: 3x3 s1 1024->1024 30x30
  %pad42 = tensor.pad %relu41 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1024x30x30xf16> to tensor<1x1024x32x32xf16>
  %init43 = tensor.empty() : tensor<1x1024x30x30xf16>
  %fill44 = linalg.fill ins(%cst : f16) outs(%init43 : tensor<1x1024x30x30xf16>) -> tensor<1x1024x30x30xf16>
  %conv45 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad42, %w_enc7 : tensor<1x1024x32x32xf16>, tensor<1024x1024x3x3xf16>)
    outs(%fill44 : tensor<1x1024x30x30xf16>) -> tensor<1x1024x30x30xf16>
  %empty46 = tensor.empty() : tensor<1x1024x30x30xf16>
  %relu47 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv45 : tensor<1x1024x30x30xf16>)
    outs(%empty46 : tensor<1x1024x30x30xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x30x30xf16>
  // enc8: 3x3 s1 1024->2048 30x30
  %pad48 = tensor.pad %relu47 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1024x30x30xf16> to tensor<1x1024x32x32xf16>
  %init49 = tensor.empty() : tensor<1x2048x30x30xf16>
  %fill50 = linalg.fill ins(%cst : f16) outs(%init49 : tensor<1x2048x30x30xf16>) -> tensor<1x2048x30x30xf16>
  %conv51 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad48, %w_enc8 : tensor<1x1024x32x32xf16>, tensor<2048x1024x3x3xf16>)
    outs(%fill50 : tensor<1x2048x30x30xf16>) -> tensor<1x2048x30x30xf16>
  %empty52 = tensor.empty() : tensor<1x2048x30x30xf16>
  %relu53 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv51 : tensor<1x2048x30x30xf16>)
    outs(%empty52 : tensor<1x2048x30x30xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x30x30xf16>

  // === Decoder ===
  // dec0: 1x1 s1 2048->512 30x30
  %init54 = tensor.empty() : tensor<1x512x30x30xf16>
  %fill55 = linalg.fill ins(%cst : f16) outs(%init54 : tensor<1x512x30x30xf16>) -> tensor<1x512x30x30xf16>
  %conv56 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu53, %w_dec0 : tensor<1x2048x30x30xf16>, tensor<512x2048x1x1xf16>)
    outs(%fill55 : tensor<1x512x30x30xf16>) -> tensor<1x512x30x30xf16>
  %empty57 = tensor.empty() : tensor<1x512x30x30xf16>
  %relu58 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv56 : tensor<1x512x30x30xf16>)
    outs(%empty57 : tensor<1x512x30x30xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x30x30xf16>
  // dec1: 3x3 s1 512->512 30x30
  %pad59 = tensor.pad %relu58 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x30x30xf16> to tensor<1x512x32x32xf16>
  %init60 = tensor.empty() : tensor<1x512x30x30xf16>
  %fill61 = linalg.fill ins(%cst : f16) outs(%init60 : tensor<1x512x30x30xf16>) -> tensor<1x512x30x30xf16>
  %conv62 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad59, %w_dec1 : tensor<1x512x32x32xf16>, tensor<512x512x3x3xf16>)
    outs(%fill61 : tensor<1x512x30x30xf16>) -> tensor<1x512x30x30xf16>
  %empty63 = tensor.empty() : tensor<1x512x30x30xf16>
  %relu64 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv62 : tensor<1x512x30x30xf16>)
    outs(%empty63 : tensor<1x512x30x30xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x30x30xf16>
  // dec2: 3x3 s1 512->256 30x30
  %pad65 = tensor.pad %relu64 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x30x30xf16> to tensor<1x512x32x32xf16>
  %init66 = tensor.empty() : tensor<1x256x30x30xf16>
  %fill67 = linalg.fill ins(%cst : f16) outs(%init66 : tensor<1x256x30x30xf16>) -> tensor<1x256x30x30xf16>
  %conv68 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad65, %w_dec2 : tensor<1x512x32x32xf16>, tensor<256x512x3x3xf16>)
    outs(%fill67 : tensor<1x256x30x30xf16>) -> tensor<1x256x30x30xf16>
  %empty69 = tensor.empty() : tensor<1x256x30x30xf16>
  %relu70 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv68 : tensor<1x256x30x30xf16>)
    outs(%empty69 : tensor<1x256x30x30xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x30x30xf16>

  // Output: 1x1 256->21
  %init71 = tensor.empty() : tensor<1x21x30x30xf16>
  %fill72 = linalg.fill ins(%cst : f16) outs(%init71 : tensor<1x21x30x30xf16>) -> tensor<1x21x30x30xf16>
  %conv73 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu70, %w_out : tensor<1x256x30x30xf16>, tensor<21x256x1x1xf16>)
    outs(%fill72 : tensor<1x21x30x30xf16>) -> tensor<1x21x30x30xf16>
  return %conv73 : tensor<1x21x30x30xf16>
}
