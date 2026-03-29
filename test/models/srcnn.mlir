func.func @srcnn(
    %input: tensor<1x3x224x224xf16>,
    %w0: tensor<64x3x9x9xf16>,
    %w1: tensor<32x64x1x1xf16>,
    %w2: tensor<3x32x5x5xf16>,
    %w_fc: tensor<3x3x1x1xf16>) -> tensor<1x3x224x224xf16> {
  %cst = arith.constant 0.0 : f16

  // conv0: 9x9 s1 p4 3->64 224x224
  %pad0 = tensor.pad %input low[0, 0, 4, 4] high[0, 0, 4, 4] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x224x224xf16> to tensor<1x3x232x232xf16>
  %init1 = tensor.empty() : tensor<1x64x224x224xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x64x224x224xf16>) -> tensor<1x64x224x224xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad0, %w0 : tensor<1x3x232x232xf16>, tensor<64x3x9x9xf16>)
    outs(%fill2 : tensor<1x64x224x224xf16>) -> tensor<1x64x224x224xf16>
  %empty4 = tensor.empty() : tensor<1x64x224x224xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x64x224x224xf16>)
    outs(%empty4 : tensor<1x64x224x224xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x224x224xf16>

  // conv1: 1x1 s1 p0 64->32 224x224
  %init6 = tensor.empty() : tensor<1x32x224x224xf16>
  %fill7 = linalg.fill ins(%cst : f16) outs(%init6 : tensor<1x32x224x224xf16>) -> tensor<1x32x224x224xf16>
  %conv8 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu5, %w1 : tensor<1x64x224x224xf16>, tensor<32x64x1x1xf16>)
    outs(%fill7 : tensor<1x32x224x224xf16>) -> tensor<1x32x224x224xf16>
  %empty9 = tensor.empty() : tensor<1x32x224x224xf16>
  %relu10 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv8 : tensor<1x32x224x224xf16>)
    outs(%empty9 : tensor<1x32x224x224xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x32x224x224xf16>

  // conv2: 5x5 s1 p2 32->3 224x224
  %pad11 = tensor.pad %relu10 low[0, 0, 2, 2] high[0, 0, 2, 2] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x32x224x224xf16> to tensor<1x32x228x228xf16>
  %init12 = tensor.empty() : tensor<1x3x224x224xf16>
  %fill13 = linalg.fill ins(%cst : f16) outs(%init12 : tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16>
  %conv14 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad11, %w2 : tensor<1x32x228x228xf16>, tensor<3x32x5x5xf16>)
    outs(%fill13 : tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16>

  // FC as 1x1 conv: 3->3
  %init15 = tensor.empty() : tensor<1x3x224x224xf16>
  %fill16 = linalg.fill ins(%cst : f16) outs(%init15 : tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16>
  %conv17 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%conv14, %w_fc : tensor<1x3x224x224xf16>, tensor<3x3x1x1xf16>)
    outs(%fill16 : tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16>
  return %conv17 : tensor<1x3x224x224xf16>
}
