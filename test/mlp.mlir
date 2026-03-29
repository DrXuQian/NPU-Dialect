// Test: MLP (matmul -> relu -> matmul -> relu)

func.func @mlp(%input: tensor<128x256xf16>,
               %w1: tensor<256x512xf16>,
               %w2: tensor<512x256xf16>) -> tensor<128x256xf16> {
  %cst = arith.constant 0.0 : f16

  // Layer 1: matmul + relu
  %init1 = tensor.empty() : tensor<128x512xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init1 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm1 = linalg.matmul ins(%input, %w1 : tensor<128x256xf16>, tensor<256x512xf16>)
                        outs(%fill1 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty1 = tensor.empty() : tensor<128x512xf16>
  %relu1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm1 : tensor<128x512xf16>) outs(%empty1 : tensor<128x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x512xf16>

  // Layer 2: matmul + relu
  %init2 = tensor.empty() : tensor<128x256xf16>
  %fill2 = linalg.fill ins(%cst : f16) outs(%init2 : tensor<128x256xf16>) -> tensor<128x256xf16>
  %mm2 = linalg.matmul ins(%relu1, %w2 : tensor<128x512xf16>, tensor<512x256xf16>)
                        outs(%fill2 : tensor<128x256xf16>) -> tensor<128x256xf16>
  %empty2 = tensor.empty() : tensor<128x256xf16>
  %relu2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm2 : tensor<128x256xf16>) outs(%empty2 : tensor<128x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x256xf16>

  return %relu2 : tensor<128x256xf16>
}
