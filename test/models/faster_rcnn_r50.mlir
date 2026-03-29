func.func @faster_rcnn_r50(
    %input: tensor<1x3x800x800xf16>,
    %w0: tensor<64x3x7x7xf16>,
    %w1: tensor<64x64x3x3xf16>,
    %w2: tensor<64x64x1x1xf16>,
    %w3: tensor<64x64x3x3xf16>,
    %w4: tensor<256x64x1x1xf16>,
    %w5: tensor<64x256x1x1xf16>,
    %w6: tensor<64x64x3x3xf16>,
    %w7: tensor<256x64x1x1xf16>,
    %w8: tensor<64x256x1x1xf16>,
    %w9: tensor<64x64x3x3xf16>,
    %w10: tensor<256x64x1x1xf16>,
    %w11: tensor<128x256x1x1xf16>,
    %w12: tensor<128x128x3x3xf16>,
    %w13: tensor<512x128x1x1xf16>,
    %w14: tensor<128x512x1x1xf16>,
    %w15: tensor<128x128x3x3xf16>,
    %w16: tensor<512x128x1x1xf16>,
    %w17: tensor<256x512x1x1xf16>,
    %w18: tensor<256x256x3x3xf16>,
    %w19: tensor<1024x256x1x1xf16>,
    %w20: tensor<256x1024x1x1xf16>,
    %w21: tensor<256x256x3x3xf16>,
    %w22: tensor<1024x256x1x1xf16>,
    %w23: tensor<512x1024x1x1xf16>,
    %w24: tensor<512x512x3x3xf16>,
    %w25: tensor<2048x512x1x1xf16>,
    %w26: tensor<256x2048x3x3xf16>,
    %w27: tensor<18x256x1x1xf16>,
    %w_det: tensor<36x18x1x1xf16>) -> tensor<1x36x25x25xf16> {
  %cst = arith.constant 0.0 : f16

  // conv0: 7x7 s2 3->64 800x800
  %pad0 = tensor.pad %input low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x3x800x800xf16> to tensor<1x3x806x806xf16>
  %init1 = tensor.empty() : tensor<1x64x400x400xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<1x64x400x400xf16>) -> tensor<1x64x400x400xf16>
  %conv3 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad0, %w0 : tensor<1x3x806x806xf16>, tensor<64x3x7x7xf16>)
    outs(%fill2 : tensor<1x64x400x400xf16>) -> tensor<1x64x400x400xf16>
  %empty4 = tensor.empty() : tensor<1x64x400x400xf16>
  %relu5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv3 : tensor<1x64x400x400xf16>)
    outs(%empty4 : tensor<1x64x400x400xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x400x400xf16>

  // conv1: 3x3 s2 64->64 400x400
  %pad6 = tensor.pad %relu5 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x400x400xf16> to tensor<1x64x402x402xf16>
  %init7 = tensor.empty() : tensor<1x64x200x200xf16>
  %fill8 = linalg.fill ins(%cst : f16) outs(%init7 : tensor<1x64x200x200xf16>) -> tensor<1x64x200x200xf16>
  %conv9 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad6, %w1 : tensor<1x64x402x402xf16>, tensor<64x64x3x3xf16>)
    outs(%fill8 : tensor<1x64x200x200xf16>) -> tensor<1x64x200x200xf16>
  %empty10 = tensor.empty() : tensor<1x64x200x200xf16>
  %relu11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv9 : tensor<1x64x200x200xf16>)
    outs(%empty10 : tensor<1x64x200x200xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x200x200xf16>

  // conv2: 1x1 s1 64->64 200x200
  %init12 = tensor.empty() : tensor<1x64x200x200xf16>
  %fill13 = linalg.fill ins(%cst : f16) outs(%init12 : tensor<1x64x200x200xf16>) -> tensor<1x64x200x200xf16>
  %conv14 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu11, %w2 : tensor<1x64x200x200xf16>, tensor<64x64x1x1xf16>)
    outs(%fill13 : tensor<1x64x200x200xf16>) -> tensor<1x64x200x200xf16>
  %empty15 = tensor.empty() : tensor<1x64x200x200xf16>
  %relu16 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv14 : tensor<1x64x200x200xf16>)
    outs(%empty15 : tensor<1x64x200x200xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x200x200xf16>

  // conv3: 3x3 s1 64->64 200x200
  %pad17 = tensor.pad %relu16 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x200x200xf16> to tensor<1x64x202x202xf16>
  %init18 = tensor.empty() : tensor<1x64x200x200xf16>
  %fill19 = linalg.fill ins(%cst : f16) outs(%init18 : tensor<1x64x200x200xf16>) -> tensor<1x64x200x200xf16>
  %conv20 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad17, %w3 : tensor<1x64x202x202xf16>, tensor<64x64x3x3xf16>)
    outs(%fill19 : tensor<1x64x200x200xf16>) -> tensor<1x64x200x200xf16>
  %empty21 = tensor.empty() : tensor<1x64x200x200xf16>
  %relu22 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv20 : tensor<1x64x200x200xf16>)
    outs(%empty21 : tensor<1x64x200x200xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x200x200xf16>

  // conv4: 1x1 s1 64->256 200x200
  %init23 = tensor.empty() : tensor<1x256x200x200xf16>
  %fill24 = linalg.fill ins(%cst : f16) outs(%init23 : tensor<1x256x200x200xf16>) -> tensor<1x256x200x200xf16>
  %conv25 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu22, %w4 : tensor<1x64x200x200xf16>, tensor<256x64x1x1xf16>)
    outs(%fill24 : tensor<1x256x200x200xf16>) -> tensor<1x256x200x200xf16>
  %empty26 = tensor.empty() : tensor<1x256x200x200xf16>
  %relu27 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv25 : tensor<1x256x200x200xf16>)
    outs(%empty26 : tensor<1x256x200x200xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x200x200xf16>

  // conv5: 1x1 s1 256->64 200x200
  %init28 = tensor.empty() : tensor<1x64x200x200xf16>
  %fill29 = linalg.fill ins(%cst : f16) outs(%init28 : tensor<1x64x200x200xf16>) -> tensor<1x64x200x200xf16>
  %conv30 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu27, %w5 : tensor<1x256x200x200xf16>, tensor<64x256x1x1xf16>)
    outs(%fill29 : tensor<1x64x200x200xf16>) -> tensor<1x64x200x200xf16>
  %empty31 = tensor.empty() : tensor<1x64x200x200xf16>
  %relu32 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv30 : tensor<1x64x200x200xf16>)
    outs(%empty31 : tensor<1x64x200x200xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x200x200xf16>

  // conv6: 3x3 s1 64->64 200x200
  %pad33 = tensor.pad %relu32 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x200x200xf16> to tensor<1x64x202x202xf16>
  %init34 = tensor.empty() : tensor<1x64x200x200xf16>
  %fill35 = linalg.fill ins(%cst : f16) outs(%init34 : tensor<1x64x200x200xf16>) -> tensor<1x64x200x200xf16>
  %conv36 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad33, %w6 : tensor<1x64x202x202xf16>, tensor<64x64x3x3xf16>)
    outs(%fill35 : tensor<1x64x200x200xf16>) -> tensor<1x64x200x200xf16>
  %empty37 = tensor.empty() : tensor<1x64x200x200xf16>
  %relu38 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv36 : tensor<1x64x200x200xf16>)
    outs(%empty37 : tensor<1x64x200x200xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x200x200xf16>

  // conv7: 1x1 s1 64->256 200x200
  %init39 = tensor.empty() : tensor<1x256x200x200xf16>
  %fill40 = linalg.fill ins(%cst : f16) outs(%init39 : tensor<1x256x200x200xf16>) -> tensor<1x256x200x200xf16>
  %conv41 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu38, %w7 : tensor<1x64x200x200xf16>, tensor<256x64x1x1xf16>)
    outs(%fill40 : tensor<1x256x200x200xf16>) -> tensor<1x256x200x200xf16>
  %empty42 = tensor.empty() : tensor<1x256x200x200xf16>
  %relu43 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv41 : tensor<1x256x200x200xf16>)
    outs(%empty42 : tensor<1x256x200x200xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x200x200xf16>

  // conv8: 1x1 s1 256->64 200x200
  %init44 = tensor.empty() : tensor<1x64x200x200xf16>
  %fill45 = linalg.fill ins(%cst : f16) outs(%init44 : tensor<1x64x200x200xf16>) -> tensor<1x64x200x200xf16>
  %conv46 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu43, %w8 : tensor<1x256x200x200xf16>, tensor<64x256x1x1xf16>)
    outs(%fill45 : tensor<1x64x200x200xf16>) -> tensor<1x64x200x200xf16>
  %empty47 = tensor.empty() : tensor<1x64x200x200xf16>
  %relu48 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv46 : tensor<1x64x200x200xf16>)
    outs(%empty47 : tensor<1x64x200x200xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x200x200xf16>

  // conv9: 3x3 s1 64->64 200x200
  %pad49 = tensor.pad %relu48 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x200x200xf16> to tensor<1x64x202x202xf16>
  %init50 = tensor.empty() : tensor<1x64x200x200xf16>
  %fill51 = linalg.fill ins(%cst : f16) outs(%init50 : tensor<1x64x200x200xf16>) -> tensor<1x64x200x200xf16>
  %conv52 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad49, %w9 : tensor<1x64x202x202xf16>, tensor<64x64x3x3xf16>)
    outs(%fill51 : tensor<1x64x200x200xf16>) -> tensor<1x64x200x200xf16>
  %empty53 = tensor.empty() : tensor<1x64x200x200xf16>
  %relu54 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv52 : tensor<1x64x200x200xf16>)
    outs(%empty53 : tensor<1x64x200x200xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x64x200x200xf16>

  // conv10: 1x1 s1 64->256 200x200
  %init55 = tensor.empty() : tensor<1x256x200x200xf16>
  %fill56 = linalg.fill ins(%cst : f16) outs(%init55 : tensor<1x256x200x200xf16>) -> tensor<1x256x200x200xf16>
  %conv57 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu54, %w10 : tensor<1x64x200x200xf16>, tensor<256x64x1x1xf16>)
    outs(%fill56 : tensor<1x256x200x200xf16>) -> tensor<1x256x200x200xf16>
  %empty58 = tensor.empty() : tensor<1x256x200x200xf16>
  %relu59 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv57 : tensor<1x256x200x200xf16>)
    outs(%empty58 : tensor<1x256x200x200xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x200x200xf16>

  // conv11: 1x1 s1 256->128 200x200
  %init60 = tensor.empty() : tensor<1x128x200x200xf16>
  %fill61 = linalg.fill ins(%cst : f16) outs(%init60 : tensor<1x128x200x200xf16>) -> tensor<1x128x200x200xf16>
  %conv62 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu59, %w11 : tensor<1x256x200x200xf16>, tensor<128x256x1x1xf16>)
    outs(%fill61 : tensor<1x128x200x200xf16>) -> tensor<1x128x200x200xf16>
  %empty63 = tensor.empty() : tensor<1x128x200x200xf16>
  %relu64 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv62 : tensor<1x128x200x200xf16>)
    outs(%empty63 : tensor<1x128x200x200xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x200x200xf16>

  // conv12: 3x3 s2 128->128 200x200
  %pad65 = tensor.pad %relu64 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x200x200xf16> to tensor<1x128x202x202xf16>
  %init66 = tensor.empty() : tensor<1x128x100x100xf16>
  %fill67 = linalg.fill ins(%cst : f16) outs(%init66 : tensor<1x128x100x100xf16>) -> tensor<1x128x100x100xf16>
  %conv68 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad65, %w12 : tensor<1x128x202x202xf16>, tensor<128x128x3x3xf16>)
    outs(%fill67 : tensor<1x128x100x100xf16>) -> tensor<1x128x100x100xf16>
  %empty69 = tensor.empty() : tensor<1x128x100x100xf16>
  %relu70 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv68 : tensor<1x128x100x100xf16>)
    outs(%empty69 : tensor<1x128x100x100xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x100x100xf16>

  // conv13: 1x1 s1 128->512 100x100
  %init71 = tensor.empty() : tensor<1x512x100x100xf16>
  %fill72 = linalg.fill ins(%cst : f16) outs(%init71 : tensor<1x512x100x100xf16>) -> tensor<1x512x100x100xf16>
  %conv73 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu70, %w13 : tensor<1x128x100x100xf16>, tensor<512x128x1x1xf16>)
    outs(%fill72 : tensor<1x512x100x100xf16>) -> tensor<1x512x100x100xf16>
  %empty74 = tensor.empty() : tensor<1x512x100x100xf16>
  %relu75 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv73 : tensor<1x512x100x100xf16>)
    outs(%empty74 : tensor<1x512x100x100xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x100x100xf16>

  // conv14: 1x1 s1 512->128 100x100
  %init76 = tensor.empty() : tensor<1x128x100x100xf16>
  %fill77 = linalg.fill ins(%cst : f16) outs(%init76 : tensor<1x128x100x100xf16>) -> tensor<1x128x100x100xf16>
  %conv78 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu75, %w14 : tensor<1x512x100x100xf16>, tensor<128x512x1x1xf16>)
    outs(%fill77 : tensor<1x128x100x100xf16>) -> tensor<1x128x100x100xf16>
  %empty79 = tensor.empty() : tensor<1x128x100x100xf16>
  %relu80 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv78 : tensor<1x128x100x100xf16>)
    outs(%empty79 : tensor<1x128x100x100xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x100x100xf16>

  // conv15: 3x3 s1 128->128 100x100
  %pad81 = tensor.pad %relu80 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x128x100x100xf16> to tensor<1x128x102x102xf16>
  %init82 = tensor.empty() : tensor<1x128x100x100xf16>
  %fill83 = linalg.fill ins(%cst : f16) outs(%init82 : tensor<1x128x100x100xf16>) -> tensor<1x128x100x100xf16>
  %conv84 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad81, %w15 : tensor<1x128x102x102xf16>, tensor<128x128x3x3xf16>)
    outs(%fill83 : tensor<1x128x100x100xf16>) -> tensor<1x128x100x100xf16>
  %empty85 = tensor.empty() : tensor<1x128x100x100xf16>
  %relu86 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv84 : tensor<1x128x100x100xf16>)
    outs(%empty85 : tensor<1x128x100x100xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x128x100x100xf16>

  // conv16: 1x1 s1 128->512 100x100
  %init87 = tensor.empty() : tensor<1x512x100x100xf16>
  %fill88 = linalg.fill ins(%cst : f16) outs(%init87 : tensor<1x512x100x100xf16>) -> tensor<1x512x100x100xf16>
  %conv89 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu86, %w16 : tensor<1x128x100x100xf16>, tensor<512x128x1x1xf16>)
    outs(%fill88 : tensor<1x512x100x100xf16>) -> tensor<1x512x100x100xf16>
  %empty90 = tensor.empty() : tensor<1x512x100x100xf16>
  %relu91 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv89 : tensor<1x512x100x100xf16>)
    outs(%empty90 : tensor<1x512x100x100xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x100x100xf16>

  // conv17: 1x1 s1 512->256 100x100
  %init92 = tensor.empty() : tensor<1x256x100x100xf16>
  %fill93 = linalg.fill ins(%cst : f16) outs(%init92 : tensor<1x256x100x100xf16>) -> tensor<1x256x100x100xf16>
  %conv94 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu91, %w17 : tensor<1x512x100x100xf16>, tensor<256x512x1x1xf16>)
    outs(%fill93 : tensor<1x256x100x100xf16>) -> tensor<1x256x100x100xf16>
  %empty95 = tensor.empty() : tensor<1x256x100x100xf16>
  %relu96 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv94 : tensor<1x256x100x100xf16>)
    outs(%empty95 : tensor<1x256x100x100xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x100x100xf16>

  // conv18: 3x3 s2 256->256 100x100
  %pad97 = tensor.pad %relu96 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x100x100xf16> to tensor<1x256x102x102xf16>
  %init98 = tensor.empty() : tensor<1x256x50x50xf16>
  %fill99 = linalg.fill ins(%cst : f16) outs(%init98 : tensor<1x256x50x50xf16>) -> tensor<1x256x50x50xf16>
  %conv100 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad97, %w18 : tensor<1x256x102x102xf16>, tensor<256x256x3x3xf16>)
    outs(%fill99 : tensor<1x256x50x50xf16>) -> tensor<1x256x50x50xf16>
  %empty101 = tensor.empty() : tensor<1x256x50x50xf16>
  %relu102 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv100 : tensor<1x256x50x50xf16>)
    outs(%empty101 : tensor<1x256x50x50xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x50x50xf16>

  // conv19: 1x1 s1 256->1024 50x50
  %init103 = tensor.empty() : tensor<1x1024x50x50xf16>
  %fill104 = linalg.fill ins(%cst : f16) outs(%init103 : tensor<1x1024x50x50xf16>) -> tensor<1x1024x50x50xf16>
  %conv105 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu102, %w19 : tensor<1x256x50x50xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill104 : tensor<1x1024x50x50xf16>) -> tensor<1x1024x50x50xf16>
  %empty106 = tensor.empty() : tensor<1x1024x50x50xf16>
  %relu107 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv105 : tensor<1x1024x50x50xf16>)
    outs(%empty106 : tensor<1x1024x50x50xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x50x50xf16>

  // conv20: 1x1 s1 1024->256 50x50
  %init108 = tensor.empty() : tensor<1x256x50x50xf16>
  %fill109 = linalg.fill ins(%cst : f16) outs(%init108 : tensor<1x256x50x50xf16>) -> tensor<1x256x50x50xf16>
  %conv110 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu107, %w20 : tensor<1x1024x50x50xf16>, tensor<256x1024x1x1xf16>)
    outs(%fill109 : tensor<1x256x50x50xf16>) -> tensor<1x256x50x50xf16>
  %empty111 = tensor.empty() : tensor<1x256x50x50xf16>
  %relu112 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv110 : tensor<1x256x50x50xf16>)
    outs(%empty111 : tensor<1x256x50x50xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x50x50xf16>

  // conv21: 3x3 s1 256->256 50x50
  %pad113 = tensor.pad %relu112 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x50x50xf16> to tensor<1x256x52x52xf16>
  %init114 = tensor.empty() : tensor<1x256x50x50xf16>
  %fill115 = linalg.fill ins(%cst : f16) outs(%init114 : tensor<1x256x50x50xf16>) -> tensor<1x256x50x50xf16>
  %conv116 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad113, %w21 : tensor<1x256x52x52xf16>, tensor<256x256x3x3xf16>)
    outs(%fill115 : tensor<1x256x50x50xf16>) -> tensor<1x256x50x50xf16>
  %empty117 = tensor.empty() : tensor<1x256x50x50xf16>
  %relu118 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv116 : tensor<1x256x50x50xf16>)
    outs(%empty117 : tensor<1x256x50x50xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x50x50xf16>

  // conv22: 1x1 s1 256->1024 50x50
  %init119 = tensor.empty() : tensor<1x1024x50x50xf16>
  %fill120 = linalg.fill ins(%cst : f16) outs(%init119 : tensor<1x1024x50x50xf16>) -> tensor<1x1024x50x50xf16>
  %conv121 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu118, %w22 : tensor<1x256x50x50xf16>, tensor<1024x256x1x1xf16>)
    outs(%fill120 : tensor<1x1024x50x50xf16>) -> tensor<1x1024x50x50xf16>
  %empty122 = tensor.empty() : tensor<1x1024x50x50xf16>
  %relu123 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv121 : tensor<1x1024x50x50xf16>)
    outs(%empty122 : tensor<1x1024x50x50xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x50x50xf16>

  // conv23: 1x1 s1 1024->512 50x50
  %init124 = tensor.empty() : tensor<1x512x50x50xf16>
  %fill125 = linalg.fill ins(%cst : f16) outs(%init124 : tensor<1x512x50x50xf16>) -> tensor<1x512x50x50xf16>
  %conv126 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu123, %w23 : tensor<1x1024x50x50xf16>, tensor<512x1024x1x1xf16>)
    outs(%fill125 : tensor<1x512x50x50xf16>) -> tensor<1x512x50x50xf16>
  %empty127 = tensor.empty() : tensor<1x512x50x50xf16>
  %relu128 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv126 : tensor<1x512x50x50xf16>)
    outs(%empty127 : tensor<1x512x50x50xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x50x50xf16>

  // conv24: 3x3 s2 512->512 50x50
  %pad129 = tensor.pad %relu128 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x50x50xf16> to tensor<1x512x52x52xf16>
  %init130 = tensor.empty() : tensor<1x512x25x25xf16>
  %fill131 = linalg.fill ins(%cst : f16) outs(%init130 : tensor<1x512x25x25xf16>) -> tensor<1x512x25x25xf16>
  %conv132 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad129, %w24 : tensor<1x512x52x52xf16>, tensor<512x512x3x3xf16>)
    outs(%fill131 : tensor<1x512x25x25xf16>) -> tensor<1x512x25x25xf16>
  %empty133 = tensor.empty() : tensor<1x512x25x25xf16>
  %relu134 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv132 : tensor<1x512x25x25xf16>)
    outs(%empty133 : tensor<1x512x25x25xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x25x25xf16>

  // conv25: 1x1 s1 512->2048 25x25
  %init135 = tensor.empty() : tensor<1x2048x25x25xf16>
  %fill136 = linalg.fill ins(%cst : f16) outs(%init135 : tensor<1x2048x25x25xf16>) -> tensor<1x2048x25x25xf16>
  %conv137 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu134, %w25 : tensor<1x512x25x25xf16>, tensor<2048x512x1x1xf16>)
    outs(%fill136 : tensor<1x2048x25x25xf16>) -> tensor<1x2048x25x25xf16>
  %empty138 = tensor.empty() : tensor<1x2048x25x25xf16>
  %relu139 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv137 : tensor<1x2048x25x25xf16>)
    outs(%empty138 : tensor<1x2048x25x25xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x25x25xf16>

  // conv26: 3x3 s1 2048->256 25x25
  %pad140 = tensor.pad %relu139 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x2048x25x25xf16> to tensor<1x2048x27x27xf16>
  %init141 = tensor.empty() : tensor<1x256x25x25xf16>
  %fill142 = linalg.fill ins(%cst : f16) outs(%init141 : tensor<1x256x25x25xf16>) -> tensor<1x256x25x25xf16>
  %conv143 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad140, %w26 : tensor<1x2048x27x27xf16>, tensor<256x2048x3x3xf16>)
    outs(%fill142 : tensor<1x256x25x25xf16>) -> tensor<1x256x25x25xf16>
  %empty144 = tensor.empty() : tensor<1x256x25x25xf16>
  %relu145 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv143 : tensor<1x256x25x25xf16>)
    outs(%empty144 : tensor<1x256x25x25xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x25x25xf16>

  // conv27: 1x1 s1 256->18 25x25
  %init146 = tensor.empty() : tensor<1x18x25x25xf16>
  %fill147 = linalg.fill ins(%cst : f16) outs(%init146 : tensor<1x18x25x25xf16>) -> tensor<1x18x25x25xf16>
  %conv148 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu145, %w27 : tensor<1x256x25x25xf16>, tensor<18x256x1x1xf16>)
    outs(%fill147 : tensor<1x18x25x25xf16>) -> tensor<1x18x25x25xf16>
  %empty149 = tensor.empty() : tensor<1x18x25x25xf16>
  %relu150 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv148 : tensor<1x18x25x25xf16>)
    outs(%empty149 : tensor<1x18x25x25xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x18x25x25xf16>

  // Detection head: 1x1 18->36
  %init151 = tensor.empty() : tensor<1x36x25x25xf16>
  %fill152 = linalg.fill ins(%cst : f16) outs(%init151 : tensor<1x36x25x25xf16>) -> tensor<1x36x25x25xf16>
  %conv153 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu150, %w_det : tensor<1x18x25x25xf16>, tensor<36x18x1x1xf16>)
    outs(%fill152 : tensor<1x36x25x25xf16>) -> tensor<1x36x25x25xf16>
  return %conv153 : tensor<1x36x25x25xf16>
}
