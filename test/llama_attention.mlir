// Test: Simplified LLaMA self-attention block (single head, no KV cache)
// seq_len=128, hidden=512, head_dim=64
// Q/K/V projections via matmul, attention scores, softmax (approximated), output projection

func.func @llama_attention(
    %input: tensor<128x512xf16>,
    %wq: tensor<512x64xf16>,
    %wk: tensor<512x64xf16>,
    %wv: tensor<512x64xf16>,
    %kt: tensor<64x128xf16>,
    %wo: tensor<64x512xf16>) -> tensor<128x512xf16> {
  %cst = arith.constant 0.0 : f16

  // Q projection: [128, 512] x [512, 64] -> [128, 64]
  %init_q = tensor.empty() : tensor<128x64xf16>
  %fill_q = linalg.fill ins(%cst : f16) outs(%init_q : tensor<128x64xf16>) -> tensor<128x64xf16>
  %q = linalg.matmul ins(%input, %wq : tensor<128x512xf16>, tensor<512x64xf16>)
                      outs(%fill_q : tensor<128x64xf16>) -> tensor<128x64xf16>

  // K projection: [128, 512] x [512, 64] -> [128, 64]
  %init_k = tensor.empty() : tensor<128x64xf16>
  %fill_k = linalg.fill ins(%cst : f16) outs(%init_k : tensor<128x64xf16>) -> tensor<128x64xf16>
  %k = linalg.matmul ins(%input, %wk : tensor<128x512xf16>, tensor<512x64xf16>)
                      outs(%fill_k : tensor<128x64xf16>) -> tensor<128x64xf16>

  // V projection: [128, 512] x [512, 64] -> [128, 64]
  %init_v = tensor.empty() : tensor<128x64xf16>
  %fill_v = linalg.fill ins(%cst : f16) outs(%init_v : tensor<128x64xf16>) -> tensor<128x64xf16>
  %v = linalg.matmul ins(%input, %wv : tensor<128x512xf16>, tensor<512x64xf16>)
                      outs(%fill_v : tensor<128x64xf16>) -> tensor<128x64xf16>

  // Attention scores: Q x K^T = [128, 64] x [64, 128] -> [128, 128]
  // K transposed is passed as a pre-transposed argument %kt
  %init_scores = tensor.empty() : tensor<128x128xf16>
  %fill_scores = linalg.fill ins(%cst : f16) outs(%init_scores : tensor<128x128xf16>) -> tensor<128x128xf16>
  %scores = linalg.matmul ins(%q, %kt : tensor<128x64xf16>, tensor<64x128xf16>)
                           outs(%fill_scores : tensor<128x128xf16>) -> tensor<128x128xf16>

  // Softmax approximation: elementwise pass-through with maximumf(x, 0) as simplified activation
  // (A true softmax would require reduction ops; this is a placeholder for the NPU compiler test)
  %empty_soft = tensor.empty() : tensor<128x128xf16>
  %softmax = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%scores : tensor<128x128xf16>)
    outs(%empty_soft : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>

  // Attention output: scores x V = [128, 128] x [128, 64] -> [128, 64]
  %init_attn = tensor.empty() : tensor<128x64xf16>
  %fill_attn = linalg.fill ins(%cst : f16) outs(%init_attn : tensor<128x64xf16>) -> tensor<128x64xf16>
  %attn = linalg.matmul ins(%softmax, %v : tensor<128x128xf16>, tensor<128x64xf16>)
                         outs(%fill_attn : tensor<128x64xf16>) -> tensor<128x64xf16>

  // Output projection: [128, 64] x [64, 512] -> [128, 512]
  %init_out = tensor.empty() : tensor<128x512xf16>
  %fill_out = linalg.fill ins(%cst : f16) outs(%init_out : tensor<128x512xf16>) -> tensor<128x512xf16>
  %output = linalg.matmul ins(%attn, %wo : tensor<128x64xf16>, tensor<64x512xf16>)
                           outs(%fill_out : tensor<128x512xf16>) -> tensor<128x512xf16>

  return %output : tensor<128x512xf16>
}
