func.func @alexnet(
    %input: tensor<1x3x224x224xf16>,
    %w0: tensor<64x3x11x11xf16>,
    %w1: tensor<64x64x3x3xf16>,
    %w2: tensor<192x64x5x5xf16>,
    %w3: tensor<192x192x3x3xf16>,
    %w4: tensor<384x192x3x3xf16>,
    %w5: tensor<256x384x3x3xf16>,
    %w6: tensor<256x256x3x3xf16>,
    %w7: tensor<256x256x3x3xf16>,
    %w_fc: tensor<1000x256x7x7xf16>) -> tensor<1x1000x1x1xf16> {
  %cst = arith.constant 0.0 : f16

  // conv0: 11x11 s4 p2 3->64 224x224
  %pad0 = tensor.pad %input low[0, 0, 2, 2] high[0, 0, 2, 2] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x224x224xf16> to tensor<1x3x228x228xf16>
  %init1 = tensor.empty() : tensor<1x64x55x55xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x64x55x55xf16>) -> tensor<1x64x55x55xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<4> : tensor<2xi64>
  } ins(%pad0, %w0 : tensor<1x3x228x228xf16>, tensor<64x3x11x11xf16>)
    outs(%fill2 : tensor<1x64x55x55xf16>) -> tensor<1x64x55x55xf16>
  %empty4 = tensor.empty() : tensor<1x64x55x55xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x64x55x55xf16>)
    outs(%empty4 : tensor<1x64x55x55xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x55x55xf16>

  // conv1: 3x3 s2 p1 64->64 55x55
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x55x55xf16> to tensor<1x64x57x57xf16>
  %init7 = tensor.empty() : tensor<1x64x28x28xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x64x28x28xf16>) -> tensor<1x64x28x28xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad6, %w1 : tensor<1x64x57x57xf16>, tensor<64x64x3x3xf16>)
    outs(%fill8 : tensor<1x64x28x28xf16>) -> tensor<1x64x28x28xf16>
  %empty10 = tensor.empty() : tensor<1x64x28x28xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x64x28x28xf16>)
    outs(%empty10 : tensor<1x64x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x28x28xf16>

  // conv2: 5x5 s1 p2 64->192 28x28
  %pad12 = tensor.pad %relu11 low[0, 0, 2, 2] high[0, 0, 2, 2] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x28x28xf16> to tensor<1x64x32x32xf16>
  %init13 = tensor.empty() : tensor<1x192x28x28xf16>
  %fill14 = linalg.fill ins(%cst : f16) outs(%init13 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %conv15 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad12, %w2 : tensor<1x64x32x32xf16>, tensor<192x64x5x5xf16>)
    outs(%fill14 : tensor<1x192x28x28xf16>) -> tensor<1x192x28x28xf16>
  %empty16 = tensor.empty() : tensor<1x192x28x28xf16>
  %relu17 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv15 : tensor<1x192x28x28xf16>)
    outs(%empty16 : tensor<1x192x28x28xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x28x28xf16>

  // conv3: 3x3 s2 p1 192->192 28x28
  %pad18 = tensor.pad %relu17 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x192x28x28xf16> to tensor<1x192x30x30xf16>
  %init19 = tensor.empty() : tensor<1x192x14x14xf16>
  %fill20 = linalg.fill ins(%cst : f16) outs(%init19 : tensor<1x192x14x14xf16>) -> tensor<1x192x14x14xf16>
  %conv21 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad18, %w3 : tensor<1x192x30x30xf16>, tensor<192x192x3x3xf16>)
    outs(%fill20 : tensor<1x192x14x14xf16>) -> tensor<1x192x14x14xf16>
  %empty22 = tensor.empty() : tensor<1x192x14x14xf16>
  %relu23 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv21 : tensor<1x192x14x14xf16>)
    outs(%empty22 : tensor<1x192x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x192x14x14xf16>

  // conv4: 3x3 s1 p1 192->384 14x14
  %pad24 = tensor.pad %relu23 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x192x14x14xf16> to tensor<1x192x16x16xf16>
  %init25 = tensor.empty() : tensor<1x384x14x14xf16>
  %fill26 = linalg.fill ins(%cst : f16) outs(%init25 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %conv27 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad24, %w4 : tensor<1x192x16x16xf16>, tensor<384x192x3x3xf16>)
    outs(%fill26 : tensor<1x384x14x14xf16>) -> tensor<1x384x14x14xf16>
  %empty28 = tensor.empty() : tensor<1x384x14x14xf16>
  %relu29 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv27 : tensor<1x384x14x14xf16>)
    outs(%empty28 : tensor<1x384x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x384x14x14xf16>

  // conv5: 3x3 s1 p1 384->256 14x14
  %pad30 = tensor.pad %relu29 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x384x14x14xf16> to tensor<1x384x16x16xf16>
  %init31 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill32 = linalg.fill ins(%cst : f16) outs(%init31 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv33 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad30, %w5 : tensor<1x384x16x16xf16>, tensor<256x384x3x3xf16>)
    outs(%fill32 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty34 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu35 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv33 : tensor<1x256x14x14xf16>)
    outs(%empty34 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // conv6: 3x3 s1 p1 256->256 14x14
  %pad36 = tensor.pad %relu35 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init37 = tensor.empty() : tensor<1x256x14x14xf16>
  %fill38 = linalg.fill ins(%cst : f16) outs(%init37 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %conv39 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad36, %w6 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill38 : tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
  %empty40 = tensor.empty() : tensor<1x256x14x14xf16>
  %relu41 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv39 : tensor<1x256x14x14xf16>)
    outs(%empty40 : tensor<1x256x14x14xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x14x14xf16>

  // conv7: 3x3 s2 p1 256->256 14x14
  %pad42 = tensor.pad %relu41 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x14x14xf16> to tensor<1x256x16x16xf16>
  %init43 = tensor.empty() : tensor<1x256x7x7xf16>
  %fill44 = linalg.fill ins(%cst : f16) outs(%init43 : tensor<1x256x7x7xf16>) -> tensor<1x256x7x7xf16>
  %conv45 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad42, %w7 : tensor<1x256x16x16xf16>, tensor<256x256x3x3xf16>)
    outs(%fill44 : tensor<1x256x7x7xf16>) -> tensor<1x256x7x7xf16>
  %empty46 = tensor.empty() : tensor<1x256x7x7xf16>
  %relu47 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv45 : tensor<1x256x7x7xf16>)
    outs(%empty46 : tensor<1x256x7x7xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x7x7xf16>

  // FC as 7x7 conv: 256->1000
  %init48 = tensor.empty() : tensor<1x1000x1x1xf16>
  %fill49 = linalg.fill ins(%cst : f16) outs(%init48 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  %fc50 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu47, %w_fc : tensor<1x256x7x7xf16>, tensor<1000x256x7x7xf16>)
    outs(%fill49 : tensor<1x1000x1x1xf16>) -> tensor<1x1000x1x1xf16>
  return %fc50 : tensor<1x1000x1x1xf16>
}
