func.func @segnet(
    %input: tensor<1x3x256x256xf16>,
    %w_enc0: tensor<64x3x3x3xf16>,
    %w_enc1: tensor<64x64x3x3xf16>,
    %w_enc2: tensor<128x64x3x3xf16>,
    %w_enc3: tensor<128x128x3x3xf16>,
    %w_enc4: tensor<256x128x3x3xf16>,
    %w_enc5: tensor<256x256x3x3xf16>,
    %w_enc6: tensor<256x256x3x3xf16>,
    %w_enc7: tensor<512x256x3x3xf16>,
    %w_enc8: tensor<512x512x3x3xf16>,
    %w_enc9: tensor<512x512x3x3xf16>,
    %w_enc10: tensor<512x512x3x3xf16>,
    %w_enc11: tensor<512x512x3x3xf16>,
    %w_dec0: tensor<512x512x3x3xf16>,
    %w_dec1: tensor<512x512x3x3xf16>,
    %w_dec2: tensor<256x512x3x3xf16>,
    %w_dec3: tensor<256x256x3x3xf16>,
    %w_dec4: tensor<128x256x3x3xf16>,
    %w_dec5: tensor<64x128x3x3xf16>,
    %w_out: tensor<12x64x1x1xf16>) -> tensor<1x12x16x16xf16> {
  %cst = arith.constant 0.0 : f16

  // === Encoder ===
  // enc0: 3x3 s1 3->64 256x256
  %pad0 = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x256x256xf16> to tensor<1x3x258x258xf16>
  %init1 = tensor.empty() : tensor<1x64x256x256xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x64x256x256xf16>) -> tensor<1x64x256x256xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad0, %w_enc0 : tensor<1x3x258x258xf16>, tensor<64x3x3x3xf16>)
    outs(%fill2 : tensor<1x64x256x256xf16>) -> tensor<1x64x256x256xf16>
  %empty4 = tensor.empty() : tensor<1x64x256x256xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x64x256x256xf16>)
    outs(%empty4 : tensor<1x64x256x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x256x256xf16>
  // enc1: 3x3 s2 64->64 256x256
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x256x256xf16> to tensor<1x64x258x258xf16>
  %init7 = tensor.empty() : tensor<1x64x128x128xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x64x128x128xf16>) -> tensor<1x64x128x128xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad6, %w_enc1 : tensor<1x64x258x258xf16>, tensor<64x64x3x3xf16>)
    outs(%fill8 : tensor<1x64x128x128xf16>) -> tensor<1x64x128x128xf16>
  %empty10 = tensor.empty() : tensor<1x64x128x128xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x64x128x128xf16>)
    outs(%empty10 : tensor<1x64x128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x128x128xf16>
  // enc2: 3x3 s1 64->128 128x128
  %pad12 = tensor.pad %relu11 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x128x128xf16> to tensor<1x64x130x130xf16>
  %init13 = tensor.empty() : tensor<1x128x128x128xf16>
  %fill14 = linalg.fill ins(%cst : f16) outs(%init13 : tensor<1x128x128x128xf16>) -> tensor<1x128x128x128xf16>
  %conv15 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad12, %w_enc2 : tensor<1x64x130x130xf16>, tensor<128x64x3x3xf16>)
    outs(%fill14 : tensor<1x128x128x128xf16>) -> tensor<1x128x128x128xf16>
  %empty16 = tensor.empty() : tensor<1x128x128x128xf16>
  %relu17 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv15 : tensor<1x128x128x128xf16>)
    outs(%empty16 : tensor<1x128x128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x128x128xf16>
  // enc3: 3x3 s2 128->128 128x128
  %pad18 = tensor.pad %relu17 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x128x128xf16> to tensor<1x128x130x130xf16>
  %init19 = tensor.empty() : tensor<1x128x64x64xf16>
  %fill20 = linalg.fill ins(%cst : f16) outs(%init19 : tensor<1x128x64x64xf16>) -> tensor<1x128x64x64xf16>
  %conv21 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad18, %w_enc3 : tensor<1x128x130x130xf16>, tensor<128x128x3x3xf16>)
    outs(%fill20 : tensor<1x128x64x64xf16>) -> tensor<1x128x64x64xf16>
  %empty22 = tensor.empty() : tensor<1x128x64x64xf16>
  %relu23 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv21 : tensor<1x128x64x64xf16>)
    outs(%empty22 : tensor<1x128x64x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x64x64xf16>
  // enc4: 3x3 s1 128->256 64x64
  %pad24 = tensor.pad %relu23 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x64x64xf16> to tensor<1x128x66x66xf16>
  %init25 = tensor.empty() : tensor<1x256x64x64xf16>
  %fill26 = linalg.fill ins(%cst : f16) outs(%init25 : tensor<1x256x64x64xf16>) -> tensor<1x256x64x64xf16>
  %conv27 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad24, %w_enc4 : tensor<1x128x66x66xf16>, tensor<256x128x3x3xf16>)
    outs(%fill26 : tensor<1x256x64x64xf16>) -> tensor<1x256x64x64xf16>
  %empty28 = tensor.empty() : tensor<1x256x64x64xf16>
  %relu29 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv27 : tensor<1x256x64x64xf16>)
    outs(%empty28 : tensor<1x256x64x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x64x64xf16>
  // enc5: 3x3 s1 256->256 64x64
  %pad30 = tensor.pad %relu29 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x64x64xf16> to tensor<1x256x66x66xf16>
  %init31 = tensor.empty() : tensor<1x256x64x64xf16>
  %fill32 = linalg.fill ins(%cst : f16) outs(%init31 : tensor<1x256x64x64xf16>) -> tensor<1x256x64x64xf16>
  %conv33 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad30, %w_enc5 : tensor<1x256x66x66xf16>, tensor<256x256x3x3xf16>)
    outs(%fill32 : tensor<1x256x64x64xf16>) -> tensor<1x256x64x64xf16>
  %empty34 = tensor.empty() : tensor<1x256x64x64xf16>
  %relu35 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv33 : tensor<1x256x64x64xf16>)
    outs(%empty34 : tensor<1x256x64x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x64x64xf16>
  // enc6: 3x3 s2 256->256 64x64
  %pad36 = tensor.pad %relu35 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x64x64xf16> to tensor<1x256x66x66xf16>
  %init37 = tensor.empty() : tensor<1x256x32x32xf16>
  %fill38 = linalg.fill ins(%cst : f16) outs(%init37 : tensor<1x256x32x32xf16>) -> tensor<1x256x32x32xf16>
  %conv39 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad36, %w_enc6 : tensor<1x256x66x66xf16>, tensor<256x256x3x3xf16>)
    outs(%fill38 : tensor<1x256x32x32xf16>) -> tensor<1x256x32x32xf16>
  %empty40 = tensor.empty() : tensor<1x256x32x32xf16>
  %relu41 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv39 : tensor<1x256x32x32xf16>)
    outs(%empty40 : tensor<1x256x32x32xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x32x32xf16>
  // enc7: 3x3 s1 256->512 32x32
  %pad42 = tensor.pad %relu41 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x32x32xf16> to tensor<1x256x34x34xf16>
  %init43 = tensor.empty() : tensor<1x512x32x32xf16>
  %fill44 = linalg.fill ins(%cst : f16) outs(%init43 : tensor<1x512x32x32xf16>) -> tensor<1x512x32x32xf16>
  %conv45 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad42, %w_enc7 : tensor<1x256x34x34xf16>, tensor<512x256x3x3xf16>)
    outs(%fill44 : tensor<1x512x32x32xf16>) -> tensor<1x512x32x32xf16>
  %empty46 = tensor.empty() : tensor<1x512x32x32xf16>
  %relu47 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv45 : tensor<1x512x32x32xf16>)
    outs(%empty46 : tensor<1x512x32x32xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x32x32xf16>
  // enc8: 3x3 s1 512->512 32x32
  %pad48 = tensor.pad %relu47 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x32x32xf16> to tensor<1x512x34x34xf16>
  %init49 = tensor.empty() : tensor<1x512x32x32xf16>
  %fill50 = linalg.fill ins(%cst : f16) outs(%init49 : tensor<1x512x32x32xf16>) -> tensor<1x512x32x32xf16>
  %conv51 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad48, %w_enc8 : tensor<1x512x34x34xf16>, tensor<512x512x3x3xf16>)
    outs(%fill50 : tensor<1x512x32x32xf16>) -> tensor<1x512x32x32xf16>
  %empty52 = tensor.empty() : tensor<1x512x32x32xf16>
  %relu53 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv51 : tensor<1x512x32x32xf16>)
    outs(%empty52 : tensor<1x512x32x32xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x32x32xf16>
  // enc9: 3x3 s2 512->512 32x32
  %pad54 = tensor.pad %relu53 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x32x32xf16> to tensor<1x512x34x34xf16>
  %init55 = tensor.empty() : tensor<1x512x16x16xf16>
  %fill56 = linalg.fill ins(%cst : f16) outs(%init55 : tensor<1x512x16x16xf16>) -> tensor<1x512x16x16xf16>
  %conv57 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad54, %w_enc9 : tensor<1x512x34x34xf16>, tensor<512x512x3x3xf16>)
    outs(%fill56 : tensor<1x512x16x16xf16>) -> tensor<1x512x16x16xf16>
  %empty58 = tensor.empty() : tensor<1x512x16x16xf16>
  %relu59 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv57 : tensor<1x512x16x16xf16>)
    outs(%empty58 : tensor<1x512x16x16xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x16x16xf16>
  // enc10: 3x3 s1 512->512 16x16
  %pad60 = tensor.pad %relu59 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x16x16xf16> to tensor<1x512x18x18xf16>
  %init61 = tensor.empty() : tensor<1x512x16x16xf16>
  %fill62 = linalg.fill ins(%cst : f16) outs(%init61 : tensor<1x512x16x16xf16>) -> tensor<1x512x16x16xf16>
  %conv63 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad60, %w_enc10 : tensor<1x512x18x18xf16>, tensor<512x512x3x3xf16>)
    outs(%fill62 : tensor<1x512x16x16xf16>) -> tensor<1x512x16x16xf16>
  %empty64 = tensor.empty() : tensor<1x512x16x16xf16>
  %relu65 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv63 : tensor<1x512x16x16xf16>)
    outs(%empty64 : tensor<1x512x16x16xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x16x16xf16>
  // enc11: 3x3 s1 512->512 16x16
  %pad66 = tensor.pad %relu65 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x16x16xf16> to tensor<1x512x18x18xf16>
  %init67 = tensor.empty() : tensor<1x512x16x16xf16>
  %fill68 = linalg.fill ins(%cst : f16) outs(%init67 : tensor<1x512x16x16xf16>) -> tensor<1x512x16x16xf16>
  %conv69 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad66, %w_enc11 : tensor<1x512x18x18xf16>, tensor<512x512x3x3xf16>)
    outs(%fill68 : tensor<1x512x16x16xf16>) -> tensor<1x512x16x16xf16>
  %empty70 = tensor.empty() : tensor<1x512x16x16xf16>
  %relu71 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv69 : tensor<1x512x16x16xf16>)
    outs(%empty70 : tensor<1x512x16x16xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x16x16xf16>

  // === Decoder ===
  // dec0: 3x3 s1 512->512 16x16
  %pad72 = tensor.pad %relu71 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x16x16xf16> to tensor<1x512x18x18xf16>
  %init73 = tensor.empty() : tensor<1x512x16x16xf16>
  %fill74 = linalg.fill ins(%cst : f16) outs(%init73 : tensor<1x512x16x16xf16>) -> tensor<1x512x16x16xf16>
  %conv75 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad72, %w_dec0 : tensor<1x512x18x18xf16>, tensor<512x512x3x3xf16>)
    outs(%fill74 : tensor<1x512x16x16xf16>) -> tensor<1x512x16x16xf16>
  %empty76 = tensor.empty() : tensor<1x512x16x16xf16>
  %relu77 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv75 : tensor<1x512x16x16xf16>)
    outs(%empty76 : tensor<1x512x16x16xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x16x16xf16>
  // dec1: 3x3 s1 512->512 16x16
  %pad78 = tensor.pad %relu77 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x16x16xf16> to tensor<1x512x18x18xf16>
  %init79 = tensor.empty() : tensor<1x512x16x16xf16>
  %fill80 = linalg.fill ins(%cst : f16) outs(%init79 : tensor<1x512x16x16xf16>) -> tensor<1x512x16x16xf16>
  %conv81 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad78, %w_dec1 : tensor<1x512x18x18xf16>, tensor<512x512x3x3xf16>)
    outs(%fill80 : tensor<1x512x16x16xf16>) -> tensor<1x512x16x16xf16>
  %empty82 = tensor.empty() : tensor<1x512x16x16xf16>
  %relu83 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv81 : tensor<1x512x16x16xf16>)
    outs(%empty82 : tensor<1x512x16x16xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x16x16xf16>
  // dec2: 3x3 s1 512->256 16x16
  %pad84 = tensor.pad %relu83 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x16x16xf16> to tensor<1x512x18x18xf16>
  %init85 = tensor.empty() : tensor<1x256x16x16xf16>
  %fill86 = linalg.fill ins(%cst : f16) outs(%init85 : tensor<1x256x16x16xf16>) -> tensor<1x256x16x16xf16>
  %conv87 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad84, %w_dec2 : tensor<1x512x18x18xf16>, tensor<256x512x3x3xf16>)
    outs(%fill86 : tensor<1x256x16x16xf16>) -> tensor<1x256x16x16xf16>
  %empty88 = tensor.empty() : tensor<1x256x16x16xf16>
  %relu89 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv87 : tensor<1x256x16x16xf16>)
    outs(%empty88 : tensor<1x256x16x16xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x16x16xf16>
  // dec3: 3x3 s1 256->256 16x16
  %pad90 = tensor.pad %relu89 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x16x16xf16> to tensor<1x256x18x18xf16>
  %init91 = tensor.empty() : tensor<1x256x16x16xf16>
  %fill92 = linalg.fill ins(%cst : f16) outs(%init91 : tensor<1x256x16x16xf16>) -> tensor<1x256x16x16xf16>
  %conv93 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad90, %w_dec3 : tensor<1x256x18x18xf16>, tensor<256x256x3x3xf16>)
    outs(%fill92 : tensor<1x256x16x16xf16>) -> tensor<1x256x16x16xf16>
  %empty94 = tensor.empty() : tensor<1x256x16x16xf16>
  %relu95 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv93 : tensor<1x256x16x16xf16>)
    outs(%empty94 : tensor<1x256x16x16xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x16x16xf16>
  // dec4: 3x3 s1 256->128 16x16
  %pad96 = tensor.pad %relu95 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x16x16xf16> to tensor<1x256x18x18xf16>
  %init97 = tensor.empty() : tensor<1x128x16x16xf16>
  %fill98 = linalg.fill ins(%cst : f16) outs(%init97 : tensor<1x128x16x16xf16>) -> tensor<1x128x16x16xf16>
  %conv99 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad96, %w_dec4 : tensor<1x256x18x18xf16>, tensor<128x256x3x3xf16>)
    outs(%fill98 : tensor<1x128x16x16xf16>) -> tensor<1x128x16x16xf16>
  %empty100 = tensor.empty() : tensor<1x128x16x16xf16>
  %relu101 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv99 : tensor<1x128x16x16xf16>)
    outs(%empty100 : tensor<1x128x16x16xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x16x16xf16>
  // dec5: 3x3 s1 128->64 16x16
  %pad102 = tensor.pad %relu101 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x16x16xf16> to tensor<1x128x18x18xf16>
  %init103 = tensor.empty() : tensor<1x64x16x16xf16>
  %fill104 = linalg.fill ins(%cst : f16) outs(%init103 : tensor<1x64x16x16xf16>) -> tensor<1x64x16x16xf16>
  %conv105 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad102, %w_dec5 : tensor<1x128x18x18xf16>, tensor<64x128x3x3xf16>)
    outs(%fill104 : tensor<1x64x16x16xf16>) -> tensor<1x64x16x16xf16>
  %empty106 = tensor.empty() : tensor<1x64x16x16xf16>
  %relu107 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv105 : tensor<1x64x16x16xf16>)
    outs(%empty106 : tensor<1x64x16x16xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x16x16xf16>

  // Output: 1x1 64->12
  %init108 = tensor.empty() : tensor<1x12x16x16xf16>
  %fill109 = linalg.fill ins(%cst : f16) outs(%init108 : tensor<1x12x16x16xf16>) -> tensor<1x12x16x16xf16>
  %conv110 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu107, %w_out : tensor<1x64x16x16xf16>, tensor<12x64x1x1xf16>)
    outs(%fill109 : tensor<1x12x16x16xf16>) -> tensor<1x12x16x16xf16>
  return %conv110 : tensor<1x12x16x16xf16>
}
