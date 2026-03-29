func.func @fcos(
    %input: tensor<1x3x800x800xf16>,
    %w0: tensor<64x3x7x7xf16>,
    %w1: tensor<64x64x3x3xf16>,
    %w2: tensor<256x64x3x3xf16>,
    %w3: tensor<256x256x3x3xf16>,
    %w4: tensor<512x256x3x3xf16>,
    %w5: tensor<512x512x3x3xf16>,
    %w6: tensor<1024x512x3x3xf16>,
    %w7: tensor<1024x1024x3x3xf16>,
    %w8: tensor<2048x1024x3x3xf16>,
    %w9: tensor<2048x2048x3x3xf16>,
    %w10: tensor<256x2048x1x1xf16>,
    %w11: tensor<256x256x3x3xf16>,
    %w12: tensor<256x256x3x3xf16>,
    %w_det: tensor<80x256x1x1xf16>) -> tensor<1x80x25x25xf16> {
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

  // conv2: 3x3 s1 64->256 200x200
  %pad12 = tensor.pad %relu11 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x64x200x200xf16> to tensor<1x64x202x202xf16>
  %init13 = tensor.empty() : tensor<1x256x200x200xf16>
  %fill14 = linalg.fill ins(%cst : f16) outs(%init13 : tensor<1x256x200x200xf16>) -> tensor<1x256x200x200xf16>
  %conv15 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad12, %w2 : tensor<1x64x202x202xf16>, tensor<256x64x3x3xf16>)
    outs(%fill14 : tensor<1x256x200x200xf16>) -> tensor<1x256x200x200xf16>
  %empty16 = tensor.empty() : tensor<1x256x200x200xf16>
  %relu17 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv15 : tensor<1x256x200x200xf16>)
    outs(%empty16 : tensor<1x256x200x200xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x200x200xf16>

  // conv3: 3x3 s1 256->256 200x200
  %pad18 = tensor.pad %relu17 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x200x200xf16> to tensor<1x256x202x202xf16>
  %init19 = tensor.empty() : tensor<1x256x200x200xf16>
  %fill20 = linalg.fill ins(%cst : f16) outs(%init19 : tensor<1x256x200x200xf16>) -> tensor<1x256x200x200xf16>
  %conv21 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad18, %w3 : tensor<1x256x202x202xf16>, tensor<256x256x3x3xf16>)
    outs(%fill20 : tensor<1x256x200x200xf16>) -> tensor<1x256x200x200xf16>
  %empty22 = tensor.empty() : tensor<1x256x200x200xf16>
  %relu23 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv21 : tensor<1x256x200x200xf16>)
    outs(%empty22 : tensor<1x256x200x200xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x200x200xf16>

  // conv4: 3x3 s2 256->512 200x200
  %pad24 = tensor.pad %relu23 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x200x200xf16> to tensor<1x256x202x202xf16>
  %init25 = tensor.empty() : tensor<1x512x100x100xf16>
  %fill26 = linalg.fill ins(%cst : f16) outs(%init25 : tensor<1x512x100x100xf16>) -> tensor<1x512x100x100xf16>
  %conv27 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad24, %w4 : tensor<1x256x202x202xf16>, tensor<512x256x3x3xf16>)
    outs(%fill26 : tensor<1x512x100x100xf16>) -> tensor<1x512x100x100xf16>
  %empty28 = tensor.empty() : tensor<1x512x100x100xf16>
  %relu29 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv27 : tensor<1x512x100x100xf16>)
    outs(%empty28 : tensor<1x512x100x100xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x100x100xf16>

  // conv5: 3x3 s1 512->512 100x100
  %pad30 = tensor.pad %relu29 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x100x100xf16> to tensor<1x512x102x102xf16>
  %init31 = tensor.empty() : tensor<1x512x100x100xf16>
  %fill32 = linalg.fill ins(%cst : f16) outs(%init31 : tensor<1x512x100x100xf16>) -> tensor<1x512x100x100xf16>
  %conv33 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad30, %w5 : tensor<1x512x102x102xf16>, tensor<512x512x3x3xf16>)
    outs(%fill32 : tensor<1x512x100x100xf16>) -> tensor<1x512x100x100xf16>
  %empty34 = tensor.empty() : tensor<1x512x100x100xf16>
  %relu35 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv33 : tensor<1x512x100x100xf16>)
    outs(%empty34 : tensor<1x512x100x100xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x512x100x100xf16>

  // conv6: 3x3 s2 512->1024 100x100
  %pad36 = tensor.pad %relu35 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x512x100x100xf16> to tensor<1x512x102x102xf16>
  %init37 = tensor.empty() : tensor<1x1024x50x50xf16>
  %fill38 = linalg.fill ins(%cst : f16) outs(%init37 : tensor<1x1024x50x50xf16>) -> tensor<1x1024x50x50xf16>
  %conv39 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad36, %w6 : tensor<1x512x102x102xf16>, tensor<1024x512x3x3xf16>)
    outs(%fill38 : tensor<1x1024x50x50xf16>) -> tensor<1x1024x50x50xf16>
  %empty40 = tensor.empty() : tensor<1x1024x50x50xf16>
  %relu41 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv39 : tensor<1x1024x50x50xf16>)
    outs(%empty40 : tensor<1x1024x50x50xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x50x50xf16>

  // conv7: 3x3 s1 1024->1024 50x50
  %pad42 = tensor.pad %relu41 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1024x50x50xf16> to tensor<1x1024x52x52xf16>
  %init43 = tensor.empty() : tensor<1x1024x50x50xf16>
  %fill44 = linalg.fill ins(%cst : f16) outs(%init43 : tensor<1x1024x50x50xf16>) -> tensor<1x1024x50x50xf16>
  %conv45 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad42, %w7 : tensor<1x1024x52x52xf16>, tensor<1024x1024x3x3xf16>)
    outs(%fill44 : tensor<1x1024x50x50xf16>) -> tensor<1x1024x50x50xf16>
  %empty46 = tensor.empty() : tensor<1x1024x50x50xf16>
  %relu47 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv45 : tensor<1x1024x50x50xf16>)
    outs(%empty46 : tensor<1x1024x50x50xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x1024x50x50xf16>

  // conv8: 3x3 s2 1024->2048 50x50
  %pad48 = tensor.pad %relu47 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x1024x50x50xf16> to tensor<1x1024x52x52xf16>
  %init49 = tensor.empty() : tensor<1x2048x25x25xf16>
  %fill50 = linalg.fill ins(%cst : f16) outs(%init49 : tensor<1x2048x25x25xf16>) -> tensor<1x2048x25x25xf16>
  %conv51 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%pad48, %w8 : tensor<1x1024x52x52xf16>, tensor<2048x1024x3x3xf16>)
    outs(%fill50 : tensor<1x2048x25x25xf16>) -> tensor<1x2048x25x25xf16>
  %empty52 = tensor.empty() : tensor<1x2048x25x25xf16>
  %relu53 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv51 : tensor<1x2048x25x25xf16>)
    outs(%empty52 : tensor<1x2048x25x25xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x25x25xf16>

  // conv9: 3x3 s1 2048->2048 25x25
  %pad54 = tensor.pad %relu53 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x2048x25x25xf16> to tensor<1x2048x27x27xf16>
  %init55 = tensor.empty() : tensor<1x2048x25x25xf16>
  %fill56 = linalg.fill ins(%cst : f16) outs(%init55 : tensor<1x2048x25x25xf16>) -> tensor<1x2048x25x25xf16>
  %conv57 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad54, %w9 : tensor<1x2048x27x27xf16>, tensor<2048x2048x3x3xf16>)
    outs(%fill56 : tensor<1x2048x25x25xf16>) -> tensor<1x2048x25x25xf16>
  %empty58 = tensor.empty() : tensor<1x2048x25x25xf16>
  %relu59 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv57 : tensor<1x2048x25x25xf16>)
    outs(%empty58 : tensor<1x2048x25x25xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x2048x25x25xf16>

  // conv10: 1x1 s1 2048->256 25x25
  %init60 = tensor.empty() : tensor<1x256x25x25xf16>
  %fill61 = linalg.fill ins(%cst : f16) outs(%init60 : tensor<1x256x25x25xf16>) -> tensor<1x256x25x25xf16>
  %conv62 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu59, %w10 : tensor<1x2048x25x25xf16>, tensor<256x2048x1x1xf16>)
    outs(%fill61 : tensor<1x256x25x25xf16>) -> tensor<1x256x25x25xf16>
  %empty63 = tensor.empty() : tensor<1x256x25x25xf16>
  %relu64 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv62 : tensor<1x256x25x25xf16>)
    outs(%empty63 : tensor<1x256x25x25xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x25x25xf16>

  // conv11: 3x3 s1 256->256 25x25
  %pad65 = tensor.pad %relu64 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x25x25xf16> to tensor<1x256x27x27xf16>
  %init66 = tensor.empty() : tensor<1x256x25x25xf16>
  %fill67 = linalg.fill ins(%cst : f16) outs(%init66 : tensor<1x256x25x25xf16>) -> tensor<1x256x25x25xf16>
  %conv68 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad65, %w11 : tensor<1x256x27x27xf16>, tensor<256x256x3x3xf16>)
    outs(%fill67 : tensor<1x256x25x25xf16>) -> tensor<1x256x25x25xf16>
  %empty69 = tensor.empty() : tensor<1x256x25x25xf16>
  %relu70 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv68 : tensor<1x256x25x25xf16>)
    outs(%empty69 : tensor<1x256x25x25xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x25x25xf16>

  // conv12: 3x3 s1 256->256 25x25
  %pad71 = tensor.pad %relu70 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):
    tensor.yield %cst : f16
  } : tensor<1x256x25x25xf16> to tensor<1x256x27x27xf16>
  %init72 = tensor.empty() : tensor<1x256x25x25xf16>
  %fill73 = linalg.fill ins(%cst : f16) outs(%init72 : tensor<1x256x25x25xf16>) -> tensor<1x256x25x25xf16>
  %conv74 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%pad71, %w12 : tensor<1x256x27x27xf16>, tensor<256x256x3x3xf16>)
    outs(%fill73 : tensor<1x256x25x25xf16>) -> tensor<1x256x25x25xf16>
  %empty75 = tensor.empty() : tensor<1x256x25x25xf16>
  %relu76 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv74 : tensor<1x256x25x25xf16>)
    outs(%empty75 : tensor<1x256x25x25xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<1x256x25x25xf16>

  // Detection head: 1x1 256->80
  %init77 = tensor.empty() : tensor<1x80x25x25xf16>
  %fill78 = linalg.fill ins(%cst : f16) outs(%init77 : tensor<1x80x25x25xf16>) -> tensor<1x80x25x25xf16>
  %conv79 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%relu76, %w_det : tensor<1x256x25x25xf16>, tensor<80x256x1x1xf16>)
    outs(%fill78 : tensor<1x80x25x25xf16>) -> tensor<1x80x25x25xf16>
  return %conv79 : tensor<1x80x25x25xf16>
}
