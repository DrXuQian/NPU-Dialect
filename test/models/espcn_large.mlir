func.func @espcn_large(
    %input: tensor<1x3x224x224xf16>,
    %w0: tensor<128x3x5x5xf16>,
    %w1: tensor<64x128x3x3xf16>,
    %w2: tensor<32x64x3x3xf16>,
    %w3: tensor<27x32x3x3xf16>,
    %w_fc: tensor<27x27x1x1xf16>) -> tensor<1x27x224x224xf16> {
  %cst = arith.constant 0.0 : f16

  // conv0: 5x5 s1 p2 3->128 224x224
  %pad0 = tensor.pad %input low[0, 0, 2, 2] high[0, 0, 2, 2] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x224x224xf16> to tensor<1x3x228x228xf16>
  %init1 = tensor.empty() : tensor<1x128x224x224xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x128x224x224xf16>) -> tensor<1x128x224x224xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad0, %w0 : tensor<1x3x228x228xf16>, tensor<128x3x5x5xf16>)
    outs(%fill2 : tensor<1x128x224x224xf16>) -> tensor<1x128x224x224xf16>
  %empty4 = tensor.empty() : tensor<1x128x224x224xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x128x224x224xf16>)
    outs(%empty4 : tensor<1x128x224x224xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x224x224xf16>

  // conv1: 3x3 s1 p1 128->64 224x224
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x224x224xf16> to tensor<1x128x226x226xf16>
  %init7 = tensor.empty() : tensor<1x64x224x224xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x64x224x224xf16>) -> tensor<1x64x224x224xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad6, %w1 : tensor<1x128x226x226xf16>, tensor<64x128x3x3xf16>)
    outs(%fill8 : tensor<1x64x224x224xf16>) -> tensor<1x64x224x224xf16>
  %empty10 = tensor.empty() : tensor<1x64x224x224xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x64x224x224xf16>)
    outs(%empty10 : tensor<1x64x224x224xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x224x224xf16>

  // conv2: 3x3 s1 p1 64->32 224x224
  %pad12 = tensor.pad %relu11 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x224x224xf16> to tensor<1x64x226x226xf16>
  %init13 = tensor.empty() : tensor<1x32x224x224xf16>
  %fill14 = linalg.fill ins(%cst : f16) outs(%init13 : tensor<1x32x224x224xf16>) -> tensor<1x32x224x224xf16>
  %conv15 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad12, %w2 : tensor<1x64x226x226xf16>, tensor<32x64x3x3xf16>)
    outs(%fill14 : tensor<1x32x224x224xf16>) -> tensor<1x32x224x224xf16>
  %empty16 = tensor.empty() : tensor<1x32x224x224xf16>
  %relu17 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv15 : tensor<1x32x224x224xf16>)
    outs(%empty16 : tensor<1x32x224x224xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x224x224xf16>

  // conv3: 3x3 s1 p1 32->27 224x224
  %pad18 = tensor.pad %relu17 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x32x224x224xf16> to tensor<1x32x226x226xf16>
  %init19 = tensor.empty() : tensor<1x27x224x224xf16>
  %fill20 = linalg.fill ins(%cst : f16) outs(%init19 : tensor<1x27x224x224xf16>) -> tensor<1x27x224x224xf16>
  %conv21 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad18, %w3 : tensor<1x32x226x226xf16>, tensor<27x32x3x3xf16>)
    outs(%fill20 : tensor<1x27x224x224xf16>) -> tensor<1x27x224x224xf16>

  // FC as 1x1 conv: 27->27
  %init22 = tensor.empty() : tensor<1x27x224x224xf16>
  %fill23 = linalg.fill ins(%cst : f16) outs(%init22 : tensor<1x27x224x224xf16>) -> tensor<1x27x224x224xf16>
  %conv24 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv21, %w_fc : tensor<1x27x224x224xf16>, tensor<27x27x1x1xf16>)
    outs(%fill23 : tensor<1x27x224x224xf16>) -> tensor<1x27x224x224xf16>
  return %conv24 : tensor<1x27x224x224xf16>
}
