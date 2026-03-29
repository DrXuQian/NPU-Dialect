func.func @inception_v3(
    %input: tensor<1x3x299x299xf16>,
    %w0: tensor<32x3x3x3xf16>,
    %w1: tensor<32x32x3x3xf16>,
    %w2: tensor<64x32x3x3xf16>,
    %w3: tensor<64x64x3x3xf16>,
    %w4: tensor<80x64x1x1xf16>,
    %w5: tensor<192x80x3x3xf16>,
    %w6: tensor<192x192x3x3xf16>,
    %w7: tensor<256x192x1x1xf16>,
    %w8: tensor<288x256x3x3xf16>,
    %w9: tensor<288x288x1x1xf16>,
    %w10: tensor<288x288x3x3xf16>,
    %w11: tensor<768x288x3x3xf16>,
    %w12: tensor<768x768x1x1xf16>,
    %w13: tensor<768x768x3x3xf16>,
    %w14: tensor<768x768x1x1xf16>,
    %w15: tensor<768x768x3x3xf16>,
    %w16: tensor<1280x768x3x3xf16>,
    %w17: tensor<2048x1280x3x3xf16>,
    %w18: tensor<2048x2048x1x1xf16>,
    %w_fc: tensor<1000x2048x1x1xf16>) -> tensor<1x1000x10x10xf16> {
  %cst = arith.constant 0.0 : f16

  // conv0: 3x3 s2 p1 3->32 299x299
  %pad0 = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x299x299xf16> to tensor<1x3x301x301xf16>
  %init1 = tensor.empty() : tensor<1x32x150x150xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x32x150x150xf16>) -> tensor<1x32x150x150xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad0, %w0 : tensor<1x3x301x301xf16>, tensor<32x3x3x3xf16>)
    outs(%fill2 : tensor<1x32x150x150xf16>) -> tensor<1x32x150x150xf16>
  %empty4 = tensor.empty() : tensor<1x32x150x150xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x32x150x150xf16>)
    outs(%empty4 : tensor<1x32x150x150xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x150x150xf16>

  // conv1: 3x3 s1 p1 32->32 150x150
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x32x150x150xf16> to tensor<1x32x152x152xf16>
  %init7 = tensor.empty() : tensor<1x32x150x150xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x32x150x150xf16>) -> tensor<1x32x150x150xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad6, %w1 : tensor<1x32x152x152xf16>, tensor<32x32x3x3xf16>)
    outs(%fill8 : tensor<1x32x150x150xf16>) -> tensor<1x32x150x150xf16>
  %empty10 = tensor.empty() : tensor<1x32x150x150xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x32x150x150xf16>)
    outs(%empty10 : tensor<1x32x150x150xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x150x150xf16>

  // conv2: 3x3 s1 p1 32->64 150x150
  %pad12 = tensor.pad %relu11 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x32x150x150xf16> to tensor<1x32x152x152xf16>
  %init13 = tensor.empty() : tensor<1x64x150x150xf16>
  %fill14 = linalg.fill ins(%cst : f16) outs(%init13 : tensor<1x64x150x150xf16>) -> tensor<1x64x150x150xf16>
  %conv15 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad12, %w2 : tensor<1x32x152x152xf16>, tensor<64x32x3x3xf16>)
    outs(%fill14 : tensor<1x64x150x150xf16>) -> tensor<1x64x150x150xf16>
  %empty16 = tensor.empty() : tensor<1x64x150x150xf16>
  %relu17 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv15 : tensor<1x64x150x150xf16>)
    outs(%empty16 : tensor<1x64x150x150xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x150x150xf16>

  // conv3: 3x3 s2 p1 64->64 150x150
  %pad18 = tensor.pad %relu17 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x150x150xf16> to tensor<1x64x152x152xf16>
  %init19 = tensor.empty() : tensor<1x64x75x75xf16>
  %fill20 = linalg.fill ins(%cst : f16) outs(%init19 : tensor<1x64x75x75xf16>) -> tensor<1x64x75x75xf16>
  %conv21 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad18, %w3 : tensor<1x64x152x152xf16>, tensor<64x64x3x3xf16>)
    outs(%fill20 : tensor<1x64x75x75xf16>) -> tensor<1x64x75x75xf16>
  %empty22 = tensor.empty() : tensor<1x64x75x75xf16>
  %relu23 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv21 : tensor<1x64x75x75xf16>)
    outs(%empty22 : tensor<1x64x75x75xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x75x75xf16>

  // conv4: 1x1 s1 p0 64->80 75x75
  %init24 = tensor.empty() : tensor<1x80x75x75xf16>
  %fill25 = linalg.fill ins(%cst : f16) outs(%init24 : tensor<1x80x75x75xf16>) -> tensor<1x80x75x75xf16>
  %conv26 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu23, %w4 : tensor<1x64x75x75xf16>, tensor<80x64x1x1xf16>)
    outs(%fill25 : tensor<1x80x75x75xf16>) -> tensor<1x80x75x75xf16>
  %empty27 = tensor.empty() : tensor<1x80x75x75xf16>
  %relu28 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv26 : tensor<1x80x75x75xf16>)
    outs(%empty27 : tensor<1x80x75x75xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x80x75x75xf16>

  // conv5: 3x3 s1 p1 80->192 75x75
  %pad29 = tensor.pad %relu28 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x80x75x75xf16> to tensor<1x80x77x77xf16>
  %init30 = tensor.empty() : tensor<1x192x75x75xf16>
  %fill31 = linalg.fill ins(%cst : f16) outs(%init30 : tensor<1x192x75x75xf16>) -> tensor<1x192x75x75xf16>
  %conv32 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad29, %w5 : tensor<1x80x77x77xf16>, tensor<192x80x3x3xf16>)
    outs(%fill31 : tensor<1x192x75x75xf16>) -> tensor<1x192x75x75xf16>
  %empty33 = tensor.empty() : tensor<1x192x75x75xf16>
  %relu34 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv32 : tensor<1x192x75x75xf16>)
    outs(%empty33 : tensor<1x192x75x75xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x75x75xf16>

  // conv6: 3x3 s2 p1 192->192 75x75
  %pad35 = tensor.pad %relu34 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x192x75x75xf16> to tensor<1x192x77x77xf16>
  %init36 = tensor.empty() : tensor<1x192x38x38xf16>
  %fill37 = linalg.fill ins(%cst : f16) outs(%init36 : tensor<1x192x38x38xf16>) -> tensor<1x192x38x38xf16>
  %conv38 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad35, %w6 : tensor<1x192x77x77xf16>, tensor<192x192x3x3xf16>)
    outs(%fill37 : tensor<1x192x38x38xf16>) -> tensor<1x192x38x38xf16>
  %empty39 = tensor.empty() : tensor<1x192x38x38xf16>
  %relu40 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv38 : tensor<1x192x38x38xf16>)
    outs(%empty39 : tensor<1x192x38x38xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x38x38xf16>

  // conv7: 1x1 s1 p0 192->256 38x38
  %init41 = tensor.empty() : tensor<1x256x38x38xf16>
  %fill42 = linalg.fill ins(%cst : f16) outs(%init41 : tensor<1x256x38x38xf16>) -> tensor<1x256x38x38xf16>
  %conv43 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu40, %w7 : tensor<1x192x38x38xf16>, tensor<256x192x1x1xf16>)
    outs(%fill42 : tensor<1x256x38x38xf16>) -> tensor<1x256x38x38xf16>
  %empty44 = tensor.empty() : tensor<1x256x38x38xf16>
  %relu45 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv43 : tensor<1x256x38x38xf16>)
    outs(%empty44 : tensor<1x256x38x38xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x38x38xf16>

  // conv8: 3x3 s1 p1 256->288 38x38
  %pad46 = tensor.pad %relu45 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x38x38xf16> to tensor<1x256x40x40xf16>
  %init47 = tensor.empty() : tensor<1x288x38x38xf16>
  %fill48 = linalg.fill ins(%cst : f16) outs(%init47 : tensor<1x288x38x38xf16>) -> tensor<1x288x38x38xf16>
  %conv49 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad46, %w8 : tensor<1x256x40x40xf16>, tensor<288x256x3x3xf16>)
    outs(%fill48 : tensor<1x288x38x38xf16>) -> tensor<1x288x38x38xf16>
  %empty50 = tensor.empty() : tensor<1x288x38x38xf16>
  %relu51 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv49 : tensor<1x288x38x38xf16>)
    outs(%empty50 : tensor<1x288x38x38xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x288x38x38xf16>

  // conv9: 1x1 s1 p0 288->288 38x38
  %init52 = tensor.empty() : tensor<1x288x38x38xf16>
  %fill53 = linalg.fill ins(%cst : f16) outs(%init52 : tensor<1x288x38x38xf16>) -> tensor<1x288x38x38xf16>
  %conv54 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu51, %w9 : tensor<1x288x38x38xf16>, tensor<288x288x1x1xf16>)
    outs(%fill53 : tensor<1x288x38x38xf16>) -> tensor<1x288x38x38xf16>
  %empty55 = tensor.empty() : tensor<1x288x38x38xf16>
  %relu56 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv54 : tensor<1x288x38x38xf16>)
    outs(%empty55 : tensor<1x288x38x38xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x288x38x38xf16>

  // conv10: 3x3 s1 p1 288->288 38x38
  %pad57 = tensor.pad %relu56 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x288x38x38xf16> to tensor<1x288x40x40xf16>
  %init58 = tensor.empty() : tensor<1x288x38x38xf16>
  %fill59 = linalg.fill ins(%cst : f16) outs(%init58 : tensor<1x288x38x38xf16>) -> tensor<1x288x38x38xf16>
  %conv60 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad57, %w10 : tensor<1x288x40x40xf16>, tensor<288x288x3x3xf16>)
    outs(%fill59 : tensor<1x288x38x38xf16>) -> tensor<1x288x38x38xf16>
  %empty61 = tensor.empty() : tensor<1x288x38x38xf16>
  %relu62 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv60 : tensor<1x288x38x38xf16>)
    outs(%empty61 : tensor<1x288x38x38xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x288x38x38xf16>

  // conv11: 3x3 s2 p1 288->768 38x38
  %pad63 = tensor.pad %relu62 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x288x38x38xf16> to tensor<1x288x40x40xf16>
  %init64 = tensor.empty() : tensor<1x768x19x19xf16>
  %fill65 = linalg.fill ins(%cst : f16) outs(%init64 : tensor<1x768x19x19xf16>) -> tensor<1x768x19x19xf16>
  %conv66 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad63, %w11 : tensor<1x288x40x40xf16>, tensor<768x288x3x3xf16>)
    outs(%fill65 : tensor<1x768x19x19xf16>) -> tensor<1x768x19x19xf16>
  %empty67 = tensor.empty() : tensor<1x768x19x19xf16>
  %relu68 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv66 : tensor<1x768x19x19xf16>)
    outs(%empty67 : tensor<1x768x19x19xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x19x19xf16>

  // conv12: 1x1 s1 p0 768->768 19x19
  %init69 = tensor.empty() : tensor<1x768x19x19xf16>
  %fill70 = linalg.fill ins(%cst : f16) outs(%init69 : tensor<1x768x19x19xf16>) -> tensor<1x768x19x19xf16>
  %conv71 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu68, %w12 : tensor<1x768x19x19xf16>, tensor<768x768x1x1xf16>)
    outs(%fill70 : tensor<1x768x19x19xf16>) -> tensor<1x768x19x19xf16>
  %empty72 = tensor.empty() : tensor<1x768x19x19xf16>
  %relu73 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv71 : tensor<1x768x19x19xf16>)
    outs(%empty72 : tensor<1x768x19x19xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x19x19xf16>

  // conv13: 3x3 s1 p1 768->768 19x19
  %pad74 = tensor.pad %relu73 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x768x19x19xf16> to tensor<1x768x21x21xf16>
  %init75 = tensor.empty() : tensor<1x768x19x19xf16>
  %fill76 = linalg.fill ins(%cst : f16) outs(%init75 : tensor<1x768x19x19xf16>) -> tensor<1x768x19x19xf16>
  %conv77 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad74, %w13 : tensor<1x768x21x21xf16>, tensor<768x768x3x3xf16>)
    outs(%fill76 : tensor<1x768x19x19xf16>) -> tensor<1x768x19x19xf16>
  %empty78 = tensor.empty() : tensor<1x768x19x19xf16>
  %relu79 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv77 : tensor<1x768x19x19xf16>)
    outs(%empty78 : tensor<1x768x19x19xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x19x19xf16>

  // conv14: 1x1 s1 p0 768->768 19x19
  %init80 = tensor.empty() : tensor<1x768x19x19xf16>
  %fill81 = linalg.fill ins(%cst : f16) outs(%init80 : tensor<1x768x19x19xf16>) -> tensor<1x768x19x19xf16>
  %conv82 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu79, %w14 : tensor<1x768x19x19xf16>, tensor<768x768x1x1xf16>)
    outs(%fill81 : tensor<1x768x19x19xf16>) -> tensor<1x768x19x19xf16>
  %empty83 = tensor.empty() : tensor<1x768x19x19xf16>
  %relu84 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv82 : tensor<1x768x19x19xf16>)
    outs(%empty83 : tensor<1x768x19x19xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x19x19xf16>

  // conv15: 3x3 s1 p1 768->768 19x19
  %pad85 = tensor.pad %relu84 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x768x19x19xf16> to tensor<1x768x21x21xf16>
  %init86 = tensor.empty() : tensor<1x768x19x19xf16>
  %fill87 = linalg.fill ins(%cst : f16) outs(%init86 : tensor<1x768x19x19xf16>) -> tensor<1x768x19x19xf16>
  %conv88 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad85, %w15 : tensor<1x768x21x21xf16>, tensor<768x768x3x3xf16>)
    outs(%fill87 : tensor<1x768x19x19xf16>) -> tensor<1x768x19x19xf16>
  %empty89 = tensor.empty() : tensor<1x768x19x19xf16>
  %relu90 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv88 : tensor<1x768x19x19xf16>)
    outs(%empty89 : tensor<1x768x19x19xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x768x19x19xf16>

  // conv16: 3x3 s2 p1 768->1280 19x19
  %pad91 = tensor.pad %relu90 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x768x19x19xf16> to tensor<1x768x21x21xf16>
  %init92 = tensor.empty() : tensor<1x1280x10x10xf16>
  %fill93 = linalg.fill ins(%cst : f16) outs(%init92 : tensor<1x1280x10x10xf16>) -> tensor<1x1280x10x10xf16>
  %conv94 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad91, %w16 : tensor<1x768x21x21xf16>, tensor<1280x768x3x3xf16>)
    outs(%fill93 : tensor<1x1280x10x10xf16>) -> tensor<1x1280x10x10xf16>
  %empty95 = tensor.empty() : tensor<1x1280x10x10xf16>
  %relu96 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv94 : tensor<1x1280x10x10xf16>)
    outs(%empty95 : tensor<1x1280x10x10xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1280x10x10xf16>

  // conv17: 3x3 s1 p1 1280->2048 10x10
  %pad97 = tensor.pad %relu96 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1280x10x10xf16> to tensor<1x1280x12x12xf16>
  %init98 = tensor.empty() : tensor<1x2048x10x10xf16>
  %fill99 = linalg.fill ins(%cst : f16) outs(%init98 : tensor<1x2048x10x10xf16>) -> tensor<1x2048x10x10xf16>
  %conv100 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad97, %w17 : tensor<1x1280x12x12xf16>, tensor<2048x1280x3x3xf16>)
    outs(%fill99 : tensor<1x2048x10x10xf16>) -> tensor<1x2048x10x10xf16>
  %empty101 = tensor.empty() : tensor<1x2048x10x10xf16>
  %relu102 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv100 : tensor<1x2048x10x10xf16>)
    outs(%empty101 : tensor<1x2048x10x10xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x10x10xf16>

  // conv18: 1x1 s1 p0 2048->2048 10x10
  %init103 = tensor.empty() : tensor<1x2048x10x10xf16>
  %fill104 = linalg.fill ins(%cst : f16) outs(%init103 : tensor<1x2048x10x10xf16>) -> tensor<1x2048x10x10xf16>
  %conv105 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu102, %w18 : tensor<1x2048x10x10xf16>, tensor<2048x2048x1x1xf16>)
    outs(%fill104 : tensor<1x2048x10x10xf16>) -> tensor<1x2048x10x10xf16>
  %empty106 = tensor.empty() : tensor<1x2048x10x10xf16>
  %relu107 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv105 : tensor<1x2048x10x10xf16>)
    outs(%empty106 : tensor<1x2048x10x10xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x10x10xf16>

  // FC as 1x1 conv: 2048->1000
  %init108 = tensor.empty() : tensor<1x1000x10x10xf16>
  %fill109 = linalg.fill ins(%cst : f16) outs(%init108 : tensor<1x1000x10x10xf16>) -> tensor<1x1000x10x10xf16>
  %conv110 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu107, %w_fc : tensor<1x2048x10x10xf16>, tensor<1000x2048x1x1xf16>)
    outs(%fill109 : tensor<1x1000x10x10xf16>) -> tensor<1x1000x10x10xf16>
  return %conv110 : tensor<1x1000x10x10xf16>
}
