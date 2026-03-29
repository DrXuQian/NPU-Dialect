func.func @deeplabv3_resnet50(
    %input: tensor<1x3x512x512xf16>,
    %w_enc0: tensor<64x3x7x7xf16>,
    %w_enc1: tensor<64x64x3x3xf16>,
    %w_enc2: tensor<256x64x3x3xf16>,
    %w_enc3: tensor<256x256x3x3xf16>,
    %w_enc4: tensor<512x256x3x3xf16>,
    %w_enc5: tensor<512x512x3x3xf16>,
    %w_enc6: tensor<1024x512x3x3xf16>,
    %w_enc7: tensor<1024x1024x3x3xf16>,
    %w_enc8: tensor<2048x1024x3x3xf16>,
    %w_enc9: tensor<2048x2048x3x3xf16>,
    %w_enc10: tensor<256x2048x1x1xf16>,
    %w_enc11: tensor<256x256x3x3xf16>,
    %w_enc12: tensor<256x256x3x3xf16>,
    %w_dec0: tensor<256x256x3x3xf16>,
    %w_dec1: tensor<256x256x3x3xf16>,
    %w_out: tensor<21x256x1x1xf16>) -> tensor<1x21x32x32xf16> {
  %cst = arith.constant 0.0 : f16

  // === Encoder ===
  // enc0: 7x7 s2 3->64 512x512
  %pad0 = tensor.pad %input low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x512x512xf16> to tensor<1x3x518x518xf16>
  %init1 = tensor.empty() : tensor<1x64x256x256xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x64x256x256xf16>) -> tensor<1x64x256x256xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad0, %w_enc0 : tensor<1x3x518x518xf16>, tensor<64x3x7x7xf16>)
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
  // enc2: 3x3 s1 64->256 128x128
  %pad12 = tensor.pad %relu11 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x128x128xf16> to tensor<1x64x130x130xf16>
  %init13 = tensor.empty() : tensor<1x256x128x128xf16>
  %fill14 = linalg.fill ins(%cst : f16) outs(%init13 : tensor<1x256x128x128xf16>) -> tensor<1x256x128x128xf16>
  %conv15 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad12, %w_enc2 : tensor<1x64x130x130xf16>, tensor<256x64x3x3xf16>)
    outs(%fill14 : tensor<1x256x128x128xf16>) -> tensor<1x256x128x128xf16>
  %empty16 = tensor.empty() : tensor<1x256x128x128xf16>
  %relu17 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv15 : tensor<1x256x128x128xf16>)
    outs(%empty16 : tensor<1x256x128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x128x128xf16>
  // enc3: 3x3 s1 256->256 128x128
  %pad18 = tensor.pad %relu17 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x128x128xf16> to tensor<1x256x130x130xf16>
  %init19 = tensor.empty() : tensor<1x256x128x128xf16>
  %fill20 = linalg.fill ins(%cst : f16) outs(%init19 : tensor<1x256x128x128xf16>) -> tensor<1x256x128x128xf16>
  %conv21 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad18, %w_enc3 : tensor<1x256x130x130xf16>, tensor<256x256x3x3xf16>)
    outs(%fill20 : tensor<1x256x128x128xf16>) -> tensor<1x256x128x128xf16>
  %empty22 = tensor.empty() : tensor<1x256x128x128xf16>
  %relu23 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv21 : tensor<1x256x128x128xf16>)
    outs(%empty22 : tensor<1x256x128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x128x128xf16>
  // enc4: 3x3 s2 256->512 128x128
  %pad24 = tensor.pad %relu23 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x128x128xf16> to tensor<1x256x130x130xf16>
  %init25 = tensor.empty() : tensor<1x512x64x64xf16>
  %fill26 = linalg.fill ins(%cst : f16) outs(%init25 : tensor<1x512x64x64xf16>) -> tensor<1x512x64x64xf16>
  %conv27 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad24, %w_enc4 : tensor<1x256x130x130xf16>, tensor<512x256x3x3xf16>)
    outs(%fill26 : tensor<1x512x64x64xf16>) -> tensor<1x512x64x64xf16>
  %empty28 = tensor.empty() : tensor<1x512x64x64xf16>
  %relu29 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv27 : tensor<1x512x64x64xf16>)
    outs(%empty28 : tensor<1x512x64x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x64x64xf16>
  // enc5: 3x3 s1 512->512 64x64
  %pad30 = tensor.pad %relu29 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x64x64xf16> to tensor<1x512x66x66xf16>
  %init31 = tensor.empty() : tensor<1x512x64x64xf16>
  %fill32 = linalg.fill ins(%cst : f16) outs(%init31 : tensor<1x512x64x64xf16>) -> tensor<1x512x64x64xf16>
  %conv33 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad30, %w_enc5 : tensor<1x512x66x66xf16>, tensor<512x512x3x3xf16>)
    outs(%fill32 : tensor<1x512x64x64xf16>) -> tensor<1x512x64x64xf16>
  %empty34 = tensor.empty() : tensor<1x512x64x64xf16>
  %relu35 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv33 : tensor<1x512x64x64xf16>)
    outs(%empty34 : tensor<1x512x64x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x64x64xf16>
  // enc6: 3x3 s2 512->1024 64x64
  %pad36 = tensor.pad %relu35 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x64x64xf16> to tensor<1x512x66x66xf16>
  %init37 = tensor.empty() : tensor<1x1024x32x32xf16>
  %fill38 = linalg.fill ins(%cst : f16) outs(%init37 : tensor<1x1024x32x32xf16>) -> tensor<1x1024x32x32xf16>
  %conv39 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad36, %w_enc6 : tensor<1x512x66x66xf16>, tensor<1024x512x3x3xf16>)
    outs(%fill38 : tensor<1x1024x32x32xf16>) -> tensor<1x1024x32x32xf16>
  %empty40 = tensor.empty() : tensor<1x1024x32x32xf16>
  %relu41 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv39 : tensor<1x1024x32x32xf16>)
    outs(%empty40 : tensor<1x1024x32x32xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x32x32xf16>
  // enc7: 3x3 s1 1024->1024 32x32
  %pad42 = tensor.pad %relu41 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1024x32x32xf16> to tensor<1x1024x34x34xf16>
  %init43 = tensor.empty() : tensor<1x1024x32x32xf16>
  %fill44 = linalg.fill ins(%cst : f16) outs(%init43 : tensor<1x1024x32x32xf16>) -> tensor<1x1024x32x32xf16>
  %conv45 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad42, %w_enc7 : tensor<1x1024x34x34xf16>, tensor<1024x1024x3x3xf16>)
    outs(%fill44 : tensor<1x1024x32x32xf16>) -> tensor<1x1024x32x32xf16>
  %empty46 = tensor.empty() : tensor<1x1024x32x32xf16>
  %relu47 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv45 : tensor<1x1024x32x32xf16>)
    outs(%empty46 : tensor<1x1024x32x32xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x32x32xf16>
  // enc8: 3x3 s1 1024->2048 32x32
  %pad48 = tensor.pad %relu47 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1024x32x32xf16> to tensor<1x1024x34x34xf16>
  %init49 = tensor.empty() : tensor<1x2048x32x32xf16>
  %fill50 = linalg.fill ins(%cst : f16) outs(%init49 : tensor<1x2048x32x32xf16>) -> tensor<1x2048x32x32xf16>
  %conv51 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad48, %w_enc8 : tensor<1x1024x34x34xf16>, tensor<2048x1024x3x3xf16>)
    outs(%fill50 : tensor<1x2048x32x32xf16>) -> tensor<1x2048x32x32xf16>
  %empty52 = tensor.empty() : tensor<1x2048x32x32xf16>
  %relu53 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv51 : tensor<1x2048x32x32xf16>)
    outs(%empty52 : tensor<1x2048x32x32xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x32x32xf16>
  // enc9: 3x3 s1 2048->2048 32x32
  %pad54 = tensor.pad %relu53 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x2048x32x32xf16> to tensor<1x2048x34x34xf16>
  %init55 = tensor.empty() : tensor<1x2048x32x32xf16>
  %fill56 = linalg.fill ins(%cst : f16) outs(%init55 : tensor<1x2048x32x32xf16>) -> tensor<1x2048x32x32xf16>
  %conv57 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad54, %w_enc9 : tensor<1x2048x34x34xf16>, tensor<2048x2048x3x3xf16>)
    outs(%fill56 : tensor<1x2048x32x32xf16>) -> tensor<1x2048x32x32xf16>
  %empty58 = tensor.empty() : tensor<1x2048x32x32xf16>
  %relu59 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv57 : tensor<1x2048x32x32xf16>)
    outs(%empty58 : tensor<1x2048x32x32xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x32x32xf16>
  // enc10: 1x1 s1 2048->256 32x32
  %init60 = tensor.empty() : tensor<1x256x32x32xf16>
  %fill61 = linalg.fill ins(%cst : f16) outs(%init60 : tensor<1x256x32x32xf16>) -> tensor<1x256x32x32xf16>
  %conv62 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu59, %w_enc10 : tensor<1x2048x32x32xf16>, tensor<256x2048x1x1xf16>)
    outs(%fill61 : tensor<1x256x32x32xf16>) -> tensor<1x256x32x32xf16>
  %empty63 = tensor.empty() : tensor<1x256x32x32xf16>
  %relu64 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv62 : tensor<1x256x32x32xf16>)
    outs(%empty63 : tensor<1x256x32x32xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x32x32xf16>
  // enc11: 3x3 s1 256->256 32x32
  %pad65 = tensor.pad %relu64 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x32x32xf16> to tensor<1x256x34x34xf16>
  %init66 = tensor.empty() : tensor<1x256x32x32xf16>
  %fill67 = linalg.fill ins(%cst : f16) outs(%init66 : tensor<1x256x32x32xf16>) -> tensor<1x256x32x32xf16>
  %conv68 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad65, %w_enc11 : tensor<1x256x34x34xf16>, tensor<256x256x3x3xf16>)
    outs(%fill67 : tensor<1x256x32x32xf16>) -> tensor<1x256x32x32xf16>
  %empty69 = tensor.empty() : tensor<1x256x32x32xf16>
  %relu70 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv68 : tensor<1x256x32x32xf16>)
    outs(%empty69 : tensor<1x256x32x32xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x32x32xf16>
  // enc12: 3x3 s1 256->256 32x32
  %pad71 = tensor.pad %relu70 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x32x32xf16> to tensor<1x256x34x34xf16>
  %init72 = tensor.empty() : tensor<1x256x32x32xf16>
  %fill73 = linalg.fill ins(%cst : f16) outs(%init72 : tensor<1x256x32x32xf16>) -> tensor<1x256x32x32xf16>
  %conv74 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad71, %w_enc12 : tensor<1x256x34x34xf16>, tensor<256x256x3x3xf16>)
    outs(%fill73 : tensor<1x256x32x32xf16>) -> tensor<1x256x32x32xf16>
  %empty75 = tensor.empty() : tensor<1x256x32x32xf16>
  %relu76 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv74 : tensor<1x256x32x32xf16>)
    outs(%empty75 : tensor<1x256x32x32xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x32x32xf16>

  // === Decoder ===
  // dec0: 3x3 s1 256->256 32x32
  %pad77 = tensor.pad %relu76 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x32x32xf16> to tensor<1x256x34x34xf16>
  %init78 = tensor.empty() : tensor<1x256x32x32xf16>
  %fill79 = linalg.fill ins(%cst : f16) outs(%init78 : tensor<1x256x32x32xf16>) -> tensor<1x256x32x32xf16>
  %conv80 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad77, %w_dec0 : tensor<1x256x34x34xf16>, tensor<256x256x3x3xf16>)
    outs(%fill79 : tensor<1x256x32x32xf16>) -> tensor<1x256x32x32xf16>
  %empty81 = tensor.empty() : tensor<1x256x32x32xf16>
  %relu82 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv80 : tensor<1x256x32x32xf16>)
    outs(%empty81 : tensor<1x256x32x32xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x32x32xf16>
  // dec1: 3x3 s1 256->256 32x32
  %pad83 = tensor.pad %relu82 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x32x32xf16> to tensor<1x256x34x34xf16>
  %init84 = tensor.empty() : tensor<1x256x32x32xf16>
  %fill85 = linalg.fill ins(%cst : f16) outs(%init84 : tensor<1x256x32x32xf16>) -> tensor<1x256x32x32xf16>
  %conv86 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad83, %w_dec1 : tensor<1x256x34x34xf16>, tensor<256x256x3x3xf16>)
    outs(%fill85 : tensor<1x256x32x32xf16>) -> tensor<1x256x32x32xf16>
  %empty87 = tensor.empty() : tensor<1x256x32x32xf16>
  %relu88 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv86 : tensor<1x256x32x32xf16>)
    outs(%empty87 : tensor<1x256x32x32xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x32x32xf16>

  // Output: 1x1 256->21
  %init89 = tensor.empty() : tensor<1x21x32x32xf16>
  %fill90 = linalg.fill ins(%cst : f16) outs(%init89 : tensor<1x21x32x32xf16>) -> tensor<1x21x32x32xf16>
  %conv91 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu88, %w_out : tensor<1x256x32x32xf16>, tensor<21x256x1x1xf16>)
    outs(%fill90 : tensor<1x21x32x32xf16>) -> tensor<1x21x32x32xf16>
  return %conv91 : tensor<1x21x32x32xf16>
}
