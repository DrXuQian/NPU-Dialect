func.func @googlenet(
    %input: tensor<1x3x224x224xf16>,
    %w0: tensor<64x3x7x7xf16>,
    %w1: tensor<64x64x3x3xf16>,
    %w2: tensor<64x64x1x1xf16>,
    %w3: tensor<192x64x3x3xf16>,
    %w4: tensor<192x192x3x3xf16>,
    %w5: tensor<256x192x1x1xf16>,
    %w6: tensor<256x256x3x3xf16>,
    %w7: tensor<480x256x1x1xf16>,
    %w8: tensor<480x480x3x3xf16>,
    %w9: tensor<480x480x3x3xf16>,
    %w10: tensor<512x480x1x1xf16>,
    %w11: tensor<512x512x3x3xf16>,
    %w12: tensor<512x512x1x1xf16>,
    %w13: tensor<512x512x3x3xf16>,
    %w14: tensor<528x512x1x1xf16>,
    %w15: tensor<832x528x3x3xf16>,
    %w16: tensor<832x832x1x1xf16>,
    %w17: tensor<832x832x3x3xf16>,
    %w18: tensor<1024x832x1x1xf16>,
    %w_fc: tensor<1000x1024x1x1xf16>) -> tensor<1x1000x7x7xf16> {
  %cst = arith.constant 0.0 : f16

  // conv0: 7x7 s2 p3 3->64 224x224
  %pad0 = tensor.pad %input low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x224x224xf16> to tensor<1x3x230x230xf16>
  %init1 = tensor.empty() : tensor<1x64x112x112xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad0, %w0 : tensor<1x3x230x230xf16>, tensor<64x3x7x7xf16>)
    outs(%fill2 : tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
  %empty4 = tensor.empty() : tensor<1x64x112x112xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x64x112x112xf16>)
    outs(%empty4 : tensor<1x64x112x112xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x112x112xf16>

  // conv1: 3x3 s2 p1 64->64 112x112
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x112x112xf16> to tensor<1x64x114x114xf16>
  %init7 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad6, %w1 : tensor<1x64x114x114xf16>, tensor<64x64x3x3xf16>)
    outs(%fill8 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty10 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x64x56x56xf16>)
    outs(%empty10 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // conv2: 1x1 s1 p0 64->64 56x56
  %init12 = tensor.empty() : tensor<1x64x56x56xf16>
  %fill13 = linalg.fill ins(%cst : f16) outs(%init12 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %conv14 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu11, %w2 : tensor<1x64x56x56xf16>, tensor<64x64x1x1xf16>)
    outs(%fill13 : tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
  %empty15 = tensor.empty() : tensor<1x64x56x56xf16>
  %relu16 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv14 : tensor<1x64x56x56xf16>)
    outs(%empty15 : tensor<1x64x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x56x56xf16>

  // conv3: 3x3 s1 p1 64->192 56x56
  %pad17 = tensor.pad %relu16 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x56x56xf16> to tensor<1x64x58x58xf16>
  %init18 = tensor.empty() : tensor<1x192x56x56xf16>
  %fill19 = linalg.fill ins(%cst : f16) outs(%init18 : tensor<1x192x56x56xf16>) -> tensor<1x192x56x56xf16>
  %conv20 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad17, %w3 : tensor<1x64x58x58xf16>, tensor<192x64x3x3xf16>)
    outs(%fill19 : tensor<1x192x56x56xf16>) -> tensor<1x192x56x56xf16>
  %empty21 = tensor.empty() : tensor<1x192x56x56xf16>
  %relu22 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv20 : tensor<1x192x56x56xf16>)
    outs(%empty21 : tensor<1x192x56x56xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x56x56xf16>

  // conv4: 3x3 s2 p1 192->192 56x56
  %pad23 = tensor.pad %relu22 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x192x56x56xf16> to tensor<1x192x58x58xf16>
  %init24 = tensor.empty() : tensor<1x192x28x28xf16>
  %fill25 = linalg.fill ins(%cst : f16) outs(%init24 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %conv26 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad23, %w4 : tensor<1x192x58x58xf16>, tensor<192x192x3x3xf16>)
    outs(%fill25 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %empty27 = tensor.empty() : tensor<1x192x28x28xf16>
  %relu28 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv26 : tensor<1x192x28x28xf16>)
    outs(%empty27 : tensor<1x192x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x28x28xf16>

  // conv5: 1x1 s1 p0 192->256 28x28
  %init29 = tensor.empty() : tensor<1x256x28x28xf16>
  %fill30 = linalg.fill ins(%cst : f16) outs(%init29 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %conv31 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu28, %w5 : tensor<1x192x28x28xf16>, tensor<256x192x1x1xf16>)
    outs(%fill30 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %empty32 = tensor.empty() : tensor<1x256x28x28xf16>
  %relu33 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv31 : tensor<1x256x28x28xf16>)
    outs(%empty32 : tensor<1x256x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x28x28xf16>

  // conv6: 3x3 s1 p1 256->256 28x28
  %pad34 = tensor.pad %relu33 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x28x28xf16> to tensor<1x256x30x30xf16>
  %init35 = tensor.empty() : tensor<1x256x28x28xf16>
  %fill36 = linalg.fill ins(%cst : f16) outs(%init35 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %conv37 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad34, %w6 : tensor<1x256x30x30xf16>, tensor<256x256x3x3xf16>)
    outs(%fill36 : tensor<1x256x28x28xf16>) -> tensor<1x256x28x28xf16>
  %empty38 = tensor.empty() : tensor<1x256x28x28xf16>
  %relu39 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv37 : tensor<1x256x28x28xf16>)
    outs(%empty38 : tensor<1x256x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x28x28xf16>

  // conv7: 1x1 s1 p0 256->480 28x28
  %init40 = tensor.empty() : tensor<1x480x28x28xf16>
  %fill41 = linalg.fill ins(%cst : f16) outs(%init40 : tensor<1x480x28x28xf16>) -> tensor<1x480x28x28xf16>
  %conv42 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu39, %w7 : tensor<1x256x28x28xf16>, tensor<480x256x1x1xf16>)
    outs(%fill41 : tensor<1x480x28x28xf16>) -> tensor<1x480x28x28xf16>
  %empty43 = tensor.empty() : tensor<1x480x28x28xf16>
  %relu44 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv42 : tensor<1x480x28x28xf16>)
    outs(%empty43 : tensor<1x480x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x480x28x28xf16>

  // conv8: 3x3 s1 p1 480->480 28x28
  %pad45 = tensor.pad %relu44 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x480x28x28xf16> to tensor<1x480x30x30xf16>
  %init46 = tensor.empty() : tensor<1x480x28x28xf16>
  %fill47 = linalg.fill ins(%cst : f16) outs(%init46 : tensor<1x480x28x28xf16>) -> tensor<1x480x28x28xf16>
  %conv48 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad45, %w8 : tensor<1x480x30x30xf16>, tensor<480x480x3x3xf16>)
    outs(%fill47 : tensor<1x480x28x28xf16>) -> tensor<1x480x28x28xf16>
  %empty49 = tensor.empty() : tensor<1x480x28x28xf16>
  %relu50 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv48 : tensor<1x480x28x28xf16>)
    outs(%empty49 : tensor<1x480x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x480x28x28xf16>

  // conv9: 3x3 s2 p1 480->480 28x28
  %pad51 = tensor.pad %relu50 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x480x28x28xf16> to tensor<1x480x30x30xf16>
  %init52 = tensor.empty() : tensor<1x480x14x14xf16>
  %fill53 = linalg.fill ins(%cst : f16) outs(%init52 : tensor<1x480x14x14xf16>) -> tensor<1x480x14x14xf16>
  %conv54 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad51, %w9 : tensor<1x480x30x30xf16>, tensor<480x480x3x3xf16>)
    outs(%fill53 : tensor<1x480x14x14xf16>) -> tensor<1x480x14x14xf16>
  %empty55 = tensor.empty() : tensor<1x480x14x14xf16>
  %relu56 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv54 : tensor<1x480x14x14xf16>)
    outs(%empty55 : tensor<1x480x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x480x14x14xf16>

  // conv10: 1x1 s1 p0 480->512 14x14
  %init57 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill58 = linalg.fill ins(%cst : f16) outs(%init57 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv59 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu56, %w10 : tensor<1x480x14x14xf16>, tensor<512x480x1x1xf16>)
    outs(%fill58 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty60 = tensor.empty() : tensor<1x512x14x14xf16>
  %relu61 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv59 : tensor<1x512x14x14xf16>)
    outs(%empty60 : tensor<1x512x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x14x14xf16>

  // conv11: 3x3 s1 p1 512->512 14x14
  %pad62 = tensor.pad %relu61 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x16x16xf16>
  %init63 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill64 = linalg.fill ins(%cst : f16) outs(%init63 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv65 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad62, %w11 : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>)
    outs(%fill64 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty66 = tensor.empty() : tensor<1x512x14x14xf16>
  %relu67 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv65 : tensor<1x512x14x14xf16>)
    outs(%empty66 : tensor<1x512x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x14x14xf16>

  // conv12: 1x1 s1 p0 512->512 14x14
  %init68 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill69 = linalg.fill ins(%cst : f16) outs(%init68 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv70 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu67, %w12 : tensor<1x512x14x14xf16>, tensor<512x512x1x1xf16>)
    outs(%fill69 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty71 = tensor.empty() : tensor<1x512x14x14xf16>
  %relu72 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv70 : tensor<1x512x14x14xf16>)
    outs(%empty71 : tensor<1x512x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x14x14xf16>

  // conv13: 3x3 s1 p1 512->512 14x14
  %pad73 = tensor.pad %relu72 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x14x14xf16> to tensor<1x512x16x16xf16>
  %init74 = tensor.empty() : tensor<1x512x14x14xf16>
  %fill75 = linalg.fill ins(%cst : f16) outs(%init74 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %conv76 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad73, %w13 : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>)
    outs(%fill75 : tensor<1x512x14x14xf16>) -> tensor<1x512x14x14xf16>
  %empty77 = tensor.empty() : tensor<1x512x14x14xf16>
  %relu78 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv76 : tensor<1x512x14x14xf16>)
    outs(%empty77 : tensor<1x512x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x14x14xf16>

  // conv14: 1x1 s1 p0 512->528 14x14
  %init79 = tensor.empty() : tensor<1x528x14x14xf16>
  %fill80 = linalg.fill ins(%cst : f16) outs(%init79 : tensor<1x528x14x14xf16>) -> tensor<1x528x14x14xf16>
  %conv81 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu78, %w14 : tensor<1x512x14x14xf16>, tensor<528x512x1x1xf16>)
    outs(%fill80 : tensor<1x528x14x14xf16>) -> tensor<1x528x14x14xf16>
  %empty82 = tensor.empty() : tensor<1x528x14x14xf16>
  %relu83 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv81 : tensor<1x528x14x14xf16>)
    outs(%empty82 : tensor<1x528x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x528x14x14xf16>

  // conv15: 3x3 s2 p1 528->832 14x14
  %pad84 = tensor.pad %relu83 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x528x14x14xf16> to tensor<1x528x16x16xf16>
  %init85 = tensor.empty() : tensor<1x832x7x7xf16>
  %fill86 = linalg.fill ins(%cst : f16) outs(%init85 : tensor<1x832x7x7xf16>) -> tensor<1x832x7x7xf16>
  %conv87 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad84, %w15 : tensor<1x528x16x16xf16>, tensor<832x528x3x3xf16>)
    outs(%fill86 : tensor<1x832x7x7xf16>) -> tensor<1x832x7x7xf16>
  %empty88 = tensor.empty() : tensor<1x832x7x7xf16>
  %relu89 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv87 : tensor<1x832x7x7xf16>)
    outs(%empty88 : tensor<1x832x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x832x7x7xf16>

  // conv16: 1x1 s1 p0 832->832 7x7
  %init90 = tensor.empty() : tensor<1x832x7x7xf16>
  %fill91 = linalg.fill ins(%cst : f16) outs(%init90 : tensor<1x832x7x7xf16>) -> tensor<1x832x7x7xf16>
  %conv92 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu89, %w16 : tensor<1x832x7x7xf16>, tensor<832x832x1x1xf16>)
    outs(%fill91 : tensor<1x832x7x7xf16>) -> tensor<1x832x7x7xf16>
  %empty93 = tensor.empty() : tensor<1x832x7x7xf16>
  %relu94 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv92 : tensor<1x832x7x7xf16>)
    outs(%empty93 : tensor<1x832x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x832x7x7xf16>

  // conv17: 3x3 s1 p1 832->832 7x7
  %pad95 = tensor.pad %relu94 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x832x7x7xf16> to tensor<1x832x9x9xf16>
  %init96 = tensor.empty() : tensor<1x832x7x7xf16>
  %fill97 = linalg.fill ins(%cst : f16) outs(%init96 : tensor<1x832x7x7xf16>) -> tensor<1x832x7x7xf16>
  %conv98 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad95, %w17 : tensor<1x832x9x9xf16>, tensor<832x832x3x3xf16>)
    outs(%fill97 : tensor<1x832x7x7xf16>) -> tensor<1x832x7x7xf16>
  %empty99 = tensor.empty() : tensor<1x832x7x7xf16>
  %relu100 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv98 : tensor<1x832x7x7xf16>)
    outs(%empty99 : tensor<1x832x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x832x7x7xf16>

  // conv18: 1x1 s1 p0 832->1024 7x7
  %init101 = tensor.empty() : tensor<1x1024x7x7xf16>
  %fill102 = linalg.fill ins(%cst : f16) outs(%init101 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %conv103 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu100, %w18 : tensor<1x832x7x7xf16>, tensor<1024x832x1x1xf16>)
    outs(%fill102 : tensor<1x1024x7x7xf16>) -> tensor<1x1024x7x7xf16>
  %empty104 = tensor.empty() : tensor<1x1024x7x7xf16>
  %relu105 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv103 : tensor<1x1024x7x7xf16>)
    outs(%empty104 : tensor<1x1024x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x7x7xf16>

  // FC as 1x1 conv: 1024->1000
  %init106 = tensor.empty() : tensor<1x1000x7x7xf16>
  %fill107 = linalg.fill ins(%cst : f16) outs(%init106 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  %conv108 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu105, %w_fc : tensor<1x1024x7x7xf16>, tensor<1000x1024x1x1xf16>)
    outs(%fill107 : tensor<1x1000x7x7xf16>) -> tensor<1x1000x7x7xf16>
  return %conv108 : tensor<1x1000x7x7xf16>
}
