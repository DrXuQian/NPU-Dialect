func.func @swin_tiny(
    %input: tensor<1x3x224x224xf16>,
    %w_patch: tensor<96x3x4x4xf16>,
    %w_s0_l0_q: tensor<96x96xf16>,
    %w_s0_l0_k: tensor<96x96xf16>,
    %w_s0_l0_v: tensor<96x96xf16>,
    %w_s0_l0_kt: tensor<96x3136xf16>,
    %w_s0_l0_o: tensor<96x96xf16>,
    %w_s0_l0_ff_up: tensor<96x384xf16>,
    %w_s0_l0_ff_dn: tensor<384x96xf16>,
    %w_s0_l1_q: tensor<96x96xf16>,
    %w_s0_l1_k: tensor<96x96xf16>,
    %w_s0_l1_v: tensor<96x96xf16>,
    %w_s0_l1_kt: tensor<96x3136xf16>,
    %w_s0_l1_o: tensor<96x96xf16>,
    %w_s0_l1_ff_up: tensor<96x384xf16>,
    %w_s0_l1_ff_dn: tensor<384x96xf16>,
    %w_merge0: tensor<96x192xf16>,
    %w_s1_l0_q: tensor<192x192xf16>,
    %w_s1_l0_k: tensor<192x192xf16>,
    %w_s1_l0_v: tensor<192x192xf16>,
    %w_s1_l0_kt: tensor<192x784xf16>,
    %w_s1_l0_o: tensor<192x192xf16>,
    %w_s1_l0_ff_up: tensor<192x768xf16>,
    %w_s1_l0_ff_dn: tensor<768x192xf16>,
    %w_s1_l1_q: tensor<192x192xf16>,
    %w_s1_l1_k: tensor<192x192xf16>,
    %w_s1_l1_v: tensor<192x192xf16>,
    %w_s1_l1_kt: tensor<192x784xf16>,
    %w_s1_l1_o: tensor<192x192xf16>,
    %w_s1_l1_ff_up: tensor<192x768xf16>,
    %w_s1_l1_ff_dn: tensor<768x192xf16>,
    %w_merge1: tensor<192x384xf16>,
    %w_s2_l0_q: tensor<384x384xf16>,
    %w_s2_l0_k: tensor<384x384xf16>,
    %w_s2_l0_v: tensor<384x384xf16>,
    %w_s2_l0_kt: tensor<384x196xf16>,
    %w_s2_l0_o: tensor<384x384xf16>,
    %w_s2_l0_ff_up: tensor<384x1536xf16>,
    %w_s2_l0_ff_dn: tensor<1536x384xf16>,
    %w_s2_l1_q: tensor<384x384xf16>,
    %w_s2_l1_k: tensor<384x384xf16>,
    %w_s2_l1_v: tensor<384x384xf16>,
    %w_s2_l1_kt: tensor<384x196xf16>,
    %w_s2_l1_o: tensor<384x384xf16>,
    %w_s2_l1_ff_up: tensor<384x1536xf16>,
    %w_s2_l1_ff_dn: tensor<1536x384xf16>,
    %w_s2_l2_q: tensor<384x384xf16>,
    %w_s2_l2_k: tensor<384x384xf16>,
    %w_s2_l2_v: tensor<384x384xf16>,
    %w_s2_l2_kt: tensor<384x196xf16>,
    %w_s2_l2_o: tensor<384x384xf16>,
    %w_s2_l2_ff_up: tensor<384x1536xf16>,
    %w_s2_l2_ff_dn: tensor<1536x384xf16>,
    %w_s2_l3_q: tensor<384x384xf16>,
    %w_s2_l3_k: tensor<384x384xf16>,
    %w_s2_l3_v: tensor<384x384xf16>,
    %w_s2_l3_kt: tensor<384x196xf16>,
    %w_s2_l3_o: tensor<384x384xf16>,
    %w_s2_l3_ff_up: tensor<384x1536xf16>,
    %w_s2_l3_ff_dn: tensor<1536x384xf16>,
    %w_s2_l4_q: tensor<384x384xf16>,
    %w_s2_l4_k: tensor<384x384xf16>,
    %w_s2_l4_v: tensor<384x384xf16>,
    %w_s2_l4_kt: tensor<384x196xf16>,
    %w_s2_l4_o: tensor<384x384xf16>,
    %w_s2_l4_ff_up: tensor<384x1536xf16>,
    %w_s2_l4_ff_dn: tensor<1536x384xf16>,
    %w_s2_l5_q: tensor<384x384xf16>,
    %w_s2_l5_k: tensor<384x384xf16>,
    %w_s2_l5_v: tensor<384x384xf16>,
    %w_s2_l5_kt: tensor<384x196xf16>,
    %w_s2_l5_o: tensor<384x384xf16>,
    %w_s2_l5_ff_up: tensor<384x1536xf16>,
    %w_s2_l5_ff_dn: tensor<1536x384xf16>,
    %w_merge2: tensor<384x768xf16>,
    %w_s3_l0_q: tensor<768x768xf16>,
    %w_s3_l0_k: tensor<768x768xf16>,
    %w_s3_l0_v: tensor<768x768xf16>,
    %w_s3_l0_kt: tensor<768x49xf16>,
    %w_s3_l0_o: tensor<768x768xf16>,
    %w_s3_l0_ff_up: tensor<768x3072xf16>,
    %w_s3_l0_ff_dn: tensor<3072x768xf16>,
    %w_s3_l1_q: tensor<768x768xf16>,
    %w_s3_l1_k: tensor<768x768xf16>,
    %w_s3_l1_v: tensor<768x768xf16>,
    %w_s3_l1_kt: tensor<768x49xf16>,
    %w_s3_l1_o: tensor<768x768xf16>,
    %w_s3_l1_ff_up: tensor<768x3072xf16>,
    %w_s3_l1_ff_dn: tensor<3072x768xf16>,
    %w_head: tensor<768x1000xf16>) -> tensor<49x1000xf16> {
  %cst = arith.constant 0.0 : f16

  // Patch embedding: 4x4 conv
  %init0 = tensor.empty() : tensor<1x96x56x56xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init0 : tensor<1x96x56x56xf16>) -> tensor<1x96x56x56xf16>
  %conv2 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<4> : tensor<2xi64>
  } ins(%input, %w_patch : tensor<1x3x224x224xf16>, tensor<96x3x4x4xf16>)
    outs(%fill1 : tensor<1x96x56x56xf16>) -> tensor<1x96x56x56xf16>

  // Reshape to [3136, 96]
  %seq3 = tensor.empty() : tensor<3136x96xf16>
  %fill4 = linalg.fill ins(%cst : f16) outs(%seq3 : tensor<3136x96xf16>) -> tensor<3136x96xf16>

  // === Swin Stage 0: seq=3136, dim=96 ===
  %init5 = tensor.empty() : tensor<3136x96xf16>
  %fill6 = linalg.fill ins(%cst : f16) outs(%init5 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %mm7 = linalg.matmul ins(%fill4, %w_s0_l0_q : tensor<3136x96xf16>, tensor<96x96xf16>)
                          outs(%fill6 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %init8 = tensor.empty() : tensor<3136x96xf16>
  %fill9 = linalg.fill ins(%cst : f16) outs(%init8 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %mm10 = linalg.matmul ins(%fill4, %w_s0_l0_k : tensor<3136x96xf16>, tensor<96x96xf16>)
                          outs(%fill9 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %init11 = tensor.empty() : tensor<3136x96xf16>
  %fill12 = linalg.fill ins(%cst : f16) outs(%init11 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %mm13 = linalg.matmul ins(%fill4, %w_s0_l0_v : tensor<3136x96xf16>, tensor<96x96xf16>)
                          outs(%fill12 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %init14 = tensor.empty() : tensor<3136x3136xf16>
  %fill15 = linalg.fill ins(%cst : f16) outs(%init14 : tensor<3136x3136xf16>) -> tensor<3136x3136xf16>
  %mm16 = linalg.matmul ins(%mm7, %w_s0_l0_kt : tensor<3136x96xf16>, tensor<96x3136xf16>)
                          outs(%fill15 : tensor<3136x3136xf16>) -> tensor<3136x3136xf16>
  %empty17 = tensor.empty() : tensor<3136x3136xf16>
  %relu18 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm16 : tensor<3136x3136xf16>)
    outs(%empty17 : tensor<3136x3136xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<3136x3136xf16>
  %init19 = tensor.empty() : tensor<3136x96xf16>
  %fill20 = linalg.fill ins(%cst : f16) outs(%init19 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %mm21 = linalg.matmul ins(%relu18, %mm13 : tensor<3136x3136xf16>, tensor<3136x96xf16>)
                          outs(%fill20 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %init22 = tensor.empty() : tensor<3136x96xf16>
  %fill23 = linalg.fill ins(%cst : f16) outs(%init22 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %mm24 = linalg.matmul ins(%mm21, %w_s0_l0_o : tensor<3136x96xf16>, tensor<96x96xf16>)
                          outs(%fill23 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %empty25 = tensor.empty() : tensor<3136x96xf16>
  %add26 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm24, %fill4 : tensor<3136x96xf16>, tensor<3136x96xf16>)
    outs(%empty25 : tensor<3136x96xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<3136x96xf16>
  %init27 = tensor.empty() : tensor<3136x384xf16>
  %fill28 = linalg.fill ins(%cst : f16) outs(%init27 : tensor<3136x384xf16>) -> tensor<3136x384xf16>
  %mm29 = linalg.matmul ins(%add26, %w_s0_l0_ff_up : tensor<3136x96xf16>, tensor<96x384xf16>)
                          outs(%fill28 : tensor<3136x384xf16>) -> tensor<3136x384xf16>
  %empty30 = tensor.empty() : tensor<3136x384xf16>
  %relu31 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm29 : tensor<3136x384xf16>)
    outs(%empty30 : tensor<3136x384xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<3136x384xf16>
  %init32 = tensor.empty() : tensor<3136x96xf16>
  %fill33 = linalg.fill ins(%cst : f16) outs(%init32 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %mm34 = linalg.matmul ins(%relu31, %w_s0_l0_ff_dn : tensor<3136x384xf16>, tensor<384x96xf16>)
                          outs(%fill33 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %empty35 = tensor.empty() : tensor<3136x96xf16>
  %add36 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm34, %add26 : tensor<3136x96xf16>, tensor<3136x96xf16>)
    outs(%empty35 : tensor<3136x96xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<3136x96xf16>
  %init37 = tensor.empty() : tensor<3136x96xf16>
  %fill38 = linalg.fill ins(%cst : f16) outs(%init37 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %mm39 = linalg.matmul ins(%add36, %w_s0_l1_q : tensor<3136x96xf16>, tensor<96x96xf16>)
                          outs(%fill38 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %init40 = tensor.empty() : tensor<3136x96xf16>
  %fill41 = linalg.fill ins(%cst : f16) outs(%init40 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %mm42 = linalg.matmul ins(%add36, %w_s0_l1_k : tensor<3136x96xf16>, tensor<96x96xf16>)
                          outs(%fill41 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %init43 = tensor.empty() : tensor<3136x96xf16>
  %fill44 = linalg.fill ins(%cst : f16) outs(%init43 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %mm45 = linalg.matmul ins(%add36, %w_s0_l1_v : tensor<3136x96xf16>, tensor<96x96xf16>)
                          outs(%fill44 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %init46 = tensor.empty() : tensor<3136x3136xf16>
  %fill47 = linalg.fill ins(%cst : f16) outs(%init46 : tensor<3136x3136xf16>) -> tensor<3136x3136xf16>
  %mm48 = linalg.matmul ins(%mm39, %w_s0_l1_kt : tensor<3136x96xf16>, tensor<96x3136xf16>)
                          outs(%fill47 : tensor<3136x3136xf16>) -> tensor<3136x3136xf16>
  %empty49 = tensor.empty() : tensor<3136x3136xf16>
  %relu50 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm48 : tensor<3136x3136xf16>)
    outs(%empty49 : tensor<3136x3136xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<3136x3136xf16>
  %init51 = tensor.empty() : tensor<3136x96xf16>
  %fill52 = linalg.fill ins(%cst : f16) outs(%init51 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %mm53 = linalg.matmul ins(%relu50, %mm45 : tensor<3136x3136xf16>, tensor<3136x96xf16>)
                          outs(%fill52 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %init54 = tensor.empty() : tensor<3136x96xf16>
  %fill55 = linalg.fill ins(%cst : f16) outs(%init54 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %mm56 = linalg.matmul ins(%mm53, %w_s0_l1_o : tensor<3136x96xf16>, tensor<96x96xf16>)
                          outs(%fill55 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %empty57 = tensor.empty() : tensor<3136x96xf16>
  %add58 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm56, %add36 : tensor<3136x96xf16>, tensor<3136x96xf16>)
    outs(%empty57 : tensor<3136x96xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<3136x96xf16>
  %init59 = tensor.empty() : tensor<3136x384xf16>
  %fill60 = linalg.fill ins(%cst : f16) outs(%init59 : tensor<3136x384xf16>) -> tensor<3136x384xf16>
  %mm61 = linalg.matmul ins(%add58, %w_s0_l1_ff_up : tensor<3136x96xf16>, tensor<96x384xf16>)
                          outs(%fill60 : tensor<3136x384xf16>) -> tensor<3136x384xf16>
  %empty62 = tensor.empty() : tensor<3136x384xf16>
  %relu63 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm61 : tensor<3136x384xf16>)
    outs(%empty62 : tensor<3136x384xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<3136x384xf16>
  %init64 = tensor.empty() : tensor<3136x96xf16>
  %fill65 = linalg.fill ins(%cst : f16) outs(%init64 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %mm66 = linalg.matmul ins(%relu63, %w_s0_l1_ff_dn : tensor<3136x384xf16>, tensor<384x96xf16>)
                          outs(%fill65 : tensor<3136x96xf16>) -> tensor<3136x96xf16>
  %empty67 = tensor.empty() : tensor<3136x96xf16>
  %add68 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm66, %add58 : tensor<3136x96xf16>, tensor<3136x96xf16>)
    outs(%empty67 : tensor<3136x96xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<3136x96xf16>

  // Patch merging: [3136,96] -> [784,192]
  %init69 = tensor.empty() : tensor<3136x192xf16>
  %fill70 = linalg.fill ins(%cst : f16) outs(%init69 : tensor<3136x192xf16>) -> tensor<3136x192xf16>
  %mm71 = linalg.matmul ins(%add68, %w_merge0 : tensor<3136x96xf16>, tensor<96x192xf16>)
                          outs(%fill70 : tensor<3136x192xf16>) -> tensor<3136x192xf16>
  %merge_reshape72 = tensor.empty() : tensor<784x192xf16>
  %fill73 = linalg.fill ins(%cst : f16) outs(%merge_reshape72 : tensor<784x192xf16>) -> tensor<784x192xf16>

  // === Swin Stage 1: seq=784, dim=192 ===
  %init74 = tensor.empty() : tensor<784x192xf16>
  %fill75 = linalg.fill ins(%cst : f16) outs(%init74 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %mm76 = linalg.matmul ins(%fill73, %w_s1_l0_q : tensor<784x192xf16>, tensor<192x192xf16>)
                          outs(%fill75 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %init77 = tensor.empty() : tensor<784x192xf16>
  %fill78 = linalg.fill ins(%cst : f16) outs(%init77 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %mm79 = linalg.matmul ins(%fill73, %w_s1_l0_k : tensor<784x192xf16>, tensor<192x192xf16>)
                          outs(%fill78 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %init80 = tensor.empty() : tensor<784x192xf16>
  %fill81 = linalg.fill ins(%cst : f16) outs(%init80 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %mm82 = linalg.matmul ins(%fill73, %w_s1_l0_v : tensor<784x192xf16>, tensor<192x192xf16>)
                          outs(%fill81 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %init83 = tensor.empty() : tensor<784x784xf16>
  %fill84 = linalg.fill ins(%cst : f16) outs(%init83 : tensor<784x784xf16>) -> tensor<784x784xf16>
  %mm85 = linalg.matmul ins(%mm76, %w_s1_l0_kt : tensor<784x192xf16>, tensor<192x784xf16>)
                          outs(%fill84 : tensor<784x784xf16>) -> tensor<784x784xf16>
  %empty86 = tensor.empty() : tensor<784x784xf16>
  %relu87 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm85 : tensor<784x784xf16>)
    outs(%empty86 : tensor<784x784xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<784x784xf16>
  %init88 = tensor.empty() : tensor<784x192xf16>
  %fill89 = linalg.fill ins(%cst : f16) outs(%init88 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %mm90 = linalg.matmul ins(%relu87, %mm82 : tensor<784x784xf16>, tensor<784x192xf16>)
                          outs(%fill89 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %init91 = tensor.empty() : tensor<784x192xf16>
  %fill92 = linalg.fill ins(%cst : f16) outs(%init91 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %mm93 = linalg.matmul ins(%mm90, %w_s1_l0_o : tensor<784x192xf16>, tensor<192x192xf16>)
                          outs(%fill92 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %empty94 = tensor.empty() : tensor<784x192xf16>
  %add95 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm93, %fill73 : tensor<784x192xf16>, tensor<784x192xf16>)
    outs(%empty94 : tensor<784x192xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<784x192xf16>
  %init96 = tensor.empty() : tensor<784x768xf16>
  %fill97 = linalg.fill ins(%cst : f16) outs(%init96 : tensor<784x768xf16>) -> tensor<784x768xf16>
  %mm98 = linalg.matmul ins(%add95, %w_s1_l0_ff_up : tensor<784x192xf16>, tensor<192x768xf16>)
                          outs(%fill97 : tensor<784x768xf16>) -> tensor<784x768xf16>
  %empty99 = tensor.empty() : tensor<784x768xf16>
  %relu100 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm98 : tensor<784x768xf16>)
    outs(%empty99 : tensor<784x768xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<784x768xf16>
  %init101 = tensor.empty() : tensor<784x192xf16>
  %fill102 = linalg.fill ins(%cst : f16) outs(%init101 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %mm103 = linalg.matmul ins(%relu100, %w_s1_l0_ff_dn : tensor<784x768xf16>, tensor<768x192xf16>)
                          outs(%fill102 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %empty104 = tensor.empty() : tensor<784x192xf16>
  %add105 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm103, %add95 : tensor<784x192xf16>, tensor<784x192xf16>)
    outs(%empty104 : tensor<784x192xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<784x192xf16>
  %init106 = tensor.empty() : tensor<784x192xf16>
  %fill107 = linalg.fill ins(%cst : f16) outs(%init106 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %mm108 = linalg.matmul ins(%add105, %w_s1_l1_q : tensor<784x192xf16>, tensor<192x192xf16>)
                          outs(%fill107 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %init109 = tensor.empty() : tensor<784x192xf16>
  %fill110 = linalg.fill ins(%cst : f16) outs(%init109 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %mm111 = linalg.matmul ins(%add105, %w_s1_l1_k : tensor<784x192xf16>, tensor<192x192xf16>)
                          outs(%fill110 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %init112 = tensor.empty() : tensor<784x192xf16>
  %fill113 = linalg.fill ins(%cst : f16) outs(%init112 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %mm114 = linalg.matmul ins(%add105, %w_s1_l1_v : tensor<784x192xf16>, tensor<192x192xf16>)
                          outs(%fill113 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %init115 = tensor.empty() : tensor<784x784xf16>
  %fill116 = linalg.fill ins(%cst : f16) outs(%init115 : tensor<784x784xf16>) -> tensor<784x784xf16>
  %mm117 = linalg.matmul ins(%mm108, %w_s1_l1_kt : tensor<784x192xf16>, tensor<192x784xf16>)
                          outs(%fill116 : tensor<784x784xf16>) -> tensor<784x784xf16>
  %empty118 = tensor.empty() : tensor<784x784xf16>
  %relu119 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm117 : tensor<784x784xf16>)
    outs(%empty118 : tensor<784x784xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<784x784xf16>
  %init120 = tensor.empty() : tensor<784x192xf16>
  %fill121 = linalg.fill ins(%cst : f16) outs(%init120 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %mm122 = linalg.matmul ins(%relu119, %mm114 : tensor<784x784xf16>, tensor<784x192xf16>)
                          outs(%fill121 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %init123 = tensor.empty() : tensor<784x192xf16>
  %fill124 = linalg.fill ins(%cst : f16) outs(%init123 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %mm125 = linalg.matmul ins(%mm122, %w_s1_l1_o : tensor<784x192xf16>, tensor<192x192xf16>)
                          outs(%fill124 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %empty126 = tensor.empty() : tensor<784x192xf16>
  %add127 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm125, %add105 : tensor<784x192xf16>, tensor<784x192xf16>)
    outs(%empty126 : tensor<784x192xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<784x192xf16>
  %init128 = tensor.empty() : tensor<784x768xf16>
  %fill129 = linalg.fill ins(%cst : f16) outs(%init128 : tensor<784x768xf16>) -> tensor<784x768xf16>
  %mm130 = linalg.matmul ins(%add127, %w_s1_l1_ff_up : tensor<784x192xf16>, tensor<192x768xf16>)
                          outs(%fill129 : tensor<784x768xf16>) -> tensor<784x768xf16>
  %empty131 = tensor.empty() : tensor<784x768xf16>
  %relu132 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm130 : tensor<784x768xf16>)
    outs(%empty131 : tensor<784x768xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<784x768xf16>
  %init133 = tensor.empty() : tensor<784x192xf16>
  %fill134 = linalg.fill ins(%cst : f16) outs(%init133 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %mm135 = linalg.matmul ins(%relu132, %w_s1_l1_ff_dn : tensor<784x768xf16>, tensor<768x192xf16>)
                          outs(%fill134 : tensor<784x192xf16>) -> tensor<784x192xf16>
  %empty136 = tensor.empty() : tensor<784x192xf16>
  %add137 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm135, %add127 : tensor<784x192xf16>, tensor<784x192xf16>)
    outs(%empty136 : tensor<784x192xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<784x192xf16>

  // Patch merging: [784,192] -> [196,384]
  %init138 = tensor.empty() : tensor<784x384xf16>
  %fill139 = linalg.fill ins(%cst : f16) outs(%init138 : tensor<784x384xf16>) -> tensor<784x384xf16>
  %mm140 = linalg.matmul ins(%add137, %w_merge1 : tensor<784x192xf16>, tensor<192x384xf16>)
                          outs(%fill139 : tensor<784x384xf16>) -> tensor<784x384xf16>
  %merge_reshape141 = tensor.empty() : tensor<196x384xf16>
  %fill142 = linalg.fill ins(%cst : f16) outs(%merge_reshape141 : tensor<196x384xf16>) -> tensor<196x384xf16>

  // === Swin Stage 2: seq=196, dim=384 ===
  %init143 = tensor.empty() : tensor<196x384xf16>
  %fill144 = linalg.fill ins(%cst : f16) outs(%init143 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm145 = linalg.matmul ins(%fill142, %w_s2_l0_q : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill144 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init146 = tensor.empty() : tensor<196x384xf16>
  %fill147 = linalg.fill ins(%cst : f16) outs(%init146 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm148 = linalg.matmul ins(%fill142, %w_s2_l0_k : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill147 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init149 = tensor.empty() : tensor<196x384xf16>
  %fill150 = linalg.fill ins(%cst : f16) outs(%init149 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm151 = linalg.matmul ins(%fill142, %w_s2_l0_v : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill150 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init152 = tensor.empty() : tensor<196x196xf16>
  %fill153 = linalg.fill ins(%cst : f16) outs(%init152 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm154 = linalg.matmul ins(%mm145, %w_s2_l0_kt : tensor<196x384xf16>, tensor<384x196xf16>)
                          outs(%fill153 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty155 = tensor.empty() : tensor<196x196xf16>
  %relu156 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm154 : tensor<196x196xf16>)
    outs(%empty155 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init157 = tensor.empty() : tensor<196x384xf16>
  %fill158 = linalg.fill ins(%cst : f16) outs(%init157 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm159 = linalg.matmul ins(%relu156, %mm151 : tensor<196x196xf16>, tensor<196x384xf16>)
                          outs(%fill158 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init160 = tensor.empty() : tensor<196x384xf16>
  %fill161 = linalg.fill ins(%cst : f16) outs(%init160 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm162 = linalg.matmul ins(%mm159, %w_s2_l0_o : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill161 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %empty163 = tensor.empty() : tensor<196x384xf16>
  %add164 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm162, %fill142 : tensor<196x384xf16>, tensor<196x384xf16>)
    outs(%empty163 : tensor<196x384xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x384xf16>
  %init165 = tensor.empty() : tensor<196x1536xf16>
  %fill166 = linalg.fill ins(%cst : f16) outs(%init165 : tensor<196x1536xf16>) -> tensor<196x1536xf16>
  %mm167 = linalg.matmul ins(%add164, %w_s2_l0_ff_up : tensor<196x384xf16>, tensor<384x1536xf16>)
                          outs(%fill166 : tensor<196x1536xf16>) -> tensor<196x1536xf16>
  %empty168 = tensor.empty() : tensor<196x1536xf16>
  %relu169 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm167 : tensor<196x1536xf16>)
    outs(%empty168 : tensor<196x1536xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x1536xf16>
  %init170 = tensor.empty() : tensor<196x384xf16>
  %fill171 = linalg.fill ins(%cst : f16) outs(%init170 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm172 = linalg.matmul ins(%relu169, %w_s2_l0_ff_dn : tensor<196x1536xf16>, tensor<1536x384xf16>)
                          outs(%fill171 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %empty173 = tensor.empty() : tensor<196x384xf16>
  %add174 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm172, %add164 : tensor<196x384xf16>, tensor<196x384xf16>)
    outs(%empty173 : tensor<196x384xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x384xf16>
  %init175 = tensor.empty() : tensor<196x384xf16>
  %fill176 = linalg.fill ins(%cst : f16) outs(%init175 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm177 = linalg.matmul ins(%add174, %w_s2_l1_q : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill176 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init178 = tensor.empty() : tensor<196x384xf16>
  %fill179 = linalg.fill ins(%cst : f16) outs(%init178 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm180 = linalg.matmul ins(%add174, %w_s2_l1_k : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill179 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init181 = tensor.empty() : tensor<196x384xf16>
  %fill182 = linalg.fill ins(%cst : f16) outs(%init181 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm183 = linalg.matmul ins(%add174, %w_s2_l1_v : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill182 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init184 = tensor.empty() : tensor<196x196xf16>
  %fill185 = linalg.fill ins(%cst : f16) outs(%init184 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm186 = linalg.matmul ins(%mm177, %w_s2_l1_kt : tensor<196x384xf16>, tensor<384x196xf16>)
                          outs(%fill185 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty187 = tensor.empty() : tensor<196x196xf16>
  %relu188 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm186 : tensor<196x196xf16>)
    outs(%empty187 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init189 = tensor.empty() : tensor<196x384xf16>
  %fill190 = linalg.fill ins(%cst : f16) outs(%init189 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm191 = linalg.matmul ins(%relu188, %mm183 : tensor<196x196xf16>, tensor<196x384xf16>)
                          outs(%fill190 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init192 = tensor.empty() : tensor<196x384xf16>
  %fill193 = linalg.fill ins(%cst : f16) outs(%init192 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm194 = linalg.matmul ins(%mm191, %w_s2_l1_o : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill193 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %empty195 = tensor.empty() : tensor<196x384xf16>
  %add196 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm194, %add174 : tensor<196x384xf16>, tensor<196x384xf16>)
    outs(%empty195 : tensor<196x384xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x384xf16>
  %init197 = tensor.empty() : tensor<196x1536xf16>
  %fill198 = linalg.fill ins(%cst : f16) outs(%init197 : tensor<196x1536xf16>) -> tensor<196x1536xf16>
  %mm199 = linalg.matmul ins(%add196, %w_s2_l1_ff_up : tensor<196x384xf16>, tensor<384x1536xf16>)
                          outs(%fill198 : tensor<196x1536xf16>) -> tensor<196x1536xf16>
  %empty200 = tensor.empty() : tensor<196x1536xf16>
  %relu201 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm199 : tensor<196x1536xf16>)
    outs(%empty200 : tensor<196x1536xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x1536xf16>
  %init202 = tensor.empty() : tensor<196x384xf16>
  %fill203 = linalg.fill ins(%cst : f16) outs(%init202 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm204 = linalg.matmul ins(%relu201, %w_s2_l1_ff_dn : tensor<196x1536xf16>, tensor<1536x384xf16>)
                          outs(%fill203 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %empty205 = tensor.empty() : tensor<196x384xf16>
  %add206 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm204, %add196 : tensor<196x384xf16>, tensor<196x384xf16>)
    outs(%empty205 : tensor<196x384xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x384xf16>
  %init207 = tensor.empty() : tensor<196x384xf16>
  %fill208 = linalg.fill ins(%cst : f16) outs(%init207 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm209 = linalg.matmul ins(%add206, %w_s2_l2_q : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill208 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init210 = tensor.empty() : tensor<196x384xf16>
  %fill211 = linalg.fill ins(%cst : f16) outs(%init210 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm212 = linalg.matmul ins(%add206, %w_s2_l2_k : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill211 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init213 = tensor.empty() : tensor<196x384xf16>
  %fill214 = linalg.fill ins(%cst : f16) outs(%init213 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm215 = linalg.matmul ins(%add206, %w_s2_l2_v : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill214 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init216 = tensor.empty() : tensor<196x196xf16>
  %fill217 = linalg.fill ins(%cst : f16) outs(%init216 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm218 = linalg.matmul ins(%mm209, %w_s2_l2_kt : tensor<196x384xf16>, tensor<384x196xf16>)
                          outs(%fill217 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty219 = tensor.empty() : tensor<196x196xf16>
  %relu220 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm218 : tensor<196x196xf16>)
    outs(%empty219 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init221 = tensor.empty() : tensor<196x384xf16>
  %fill222 = linalg.fill ins(%cst : f16) outs(%init221 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm223 = linalg.matmul ins(%relu220, %mm215 : tensor<196x196xf16>, tensor<196x384xf16>)
                          outs(%fill222 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init224 = tensor.empty() : tensor<196x384xf16>
  %fill225 = linalg.fill ins(%cst : f16) outs(%init224 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm226 = linalg.matmul ins(%mm223, %w_s2_l2_o : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill225 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %empty227 = tensor.empty() : tensor<196x384xf16>
  %add228 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm226, %add206 : tensor<196x384xf16>, tensor<196x384xf16>)
    outs(%empty227 : tensor<196x384xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x384xf16>
  %init229 = tensor.empty() : tensor<196x1536xf16>
  %fill230 = linalg.fill ins(%cst : f16) outs(%init229 : tensor<196x1536xf16>) -> tensor<196x1536xf16>
  %mm231 = linalg.matmul ins(%add228, %w_s2_l2_ff_up : tensor<196x384xf16>, tensor<384x1536xf16>)
                          outs(%fill230 : tensor<196x1536xf16>) -> tensor<196x1536xf16>
  %empty232 = tensor.empty() : tensor<196x1536xf16>
  %relu233 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm231 : tensor<196x1536xf16>)
    outs(%empty232 : tensor<196x1536xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x1536xf16>
  %init234 = tensor.empty() : tensor<196x384xf16>
  %fill235 = linalg.fill ins(%cst : f16) outs(%init234 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm236 = linalg.matmul ins(%relu233, %w_s2_l2_ff_dn : tensor<196x1536xf16>, tensor<1536x384xf16>)
                          outs(%fill235 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %empty237 = tensor.empty() : tensor<196x384xf16>
  %add238 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm236, %add228 : tensor<196x384xf16>, tensor<196x384xf16>)
    outs(%empty237 : tensor<196x384xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x384xf16>
  %init239 = tensor.empty() : tensor<196x384xf16>
  %fill240 = linalg.fill ins(%cst : f16) outs(%init239 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm241 = linalg.matmul ins(%add238, %w_s2_l3_q : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill240 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init242 = tensor.empty() : tensor<196x384xf16>
  %fill243 = linalg.fill ins(%cst : f16) outs(%init242 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm244 = linalg.matmul ins(%add238, %w_s2_l3_k : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill243 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init245 = tensor.empty() : tensor<196x384xf16>
  %fill246 = linalg.fill ins(%cst : f16) outs(%init245 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm247 = linalg.matmul ins(%add238, %w_s2_l3_v : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill246 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init248 = tensor.empty() : tensor<196x196xf16>
  %fill249 = linalg.fill ins(%cst : f16) outs(%init248 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm250 = linalg.matmul ins(%mm241, %w_s2_l3_kt : tensor<196x384xf16>, tensor<384x196xf16>)
                          outs(%fill249 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty251 = tensor.empty() : tensor<196x196xf16>
  %relu252 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm250 : tensor<196x196xf16>)
    outs(%empty251 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init253 = tensor.empty() : tensor<196x384xf16>
  %fill254 = linalg.fill ins(%cst : f16) outs(%init253 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm255 = linalg.matmul ins(%relu252, %mm247 : tensor<196x196xf16>, tensor<196x384xf16>)
                          outs(%fill254 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init256 = tensor.empty() : tensor<196x384xf16>
  %fill257 = linalg.fill ins(%cst : f16) outs(%init256 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm258 = linalg.matmul ins(%mm255, %w_s2_l3_o : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill257 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %empty259 = tensor.empty() : tensor<196x384xf16>
  %add260 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm258, %add238 : tensor<196x384xf16>, tensor<196x384xf16>)
    outs(%empty259 : tensor<196x384xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x384xf16>
  %init261 = tensor.empty() : tensor<196x1536xf16>
  %fill262 = linalg.fill ins(%cst : f16) outs(%init261 : tensor<196x1536xf16>) -> tensor<196x1536xf16>
  %mm263 = linalg.matmul ins(%add260, %w_s2_l3_ff_up : tensor<196x384xf16>, tensor<384x1536xf16>)
                          outs(%fill262 : tensor<196x1536xf16>) -> tensor<196x1536xf16>
  %empty264 = tensor.empty() : tensor<196x1536xf16>
  %relu265 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm263 : tensor<196x1536xf16>)
    outs(%empty264 : tensor<196x1536xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x1536xf16>
  %init266 = tensor.empty() : tensor<196x384xf16>
  %fill267 = linalg.fill ins(%cst : f16) outs(%init266 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm268 = linalg.matmul ins(%relu265, %w_s2_l3_ff_dn : tensor<196x1536xf16>, tensor<1536x384xf16>)
                          outs(%fill267 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %empty269 = tensor.empty() : tensor<196x384xf16>
  %add270 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm268, %add260 : tensor<196x384xf16>, tensor<196x384xf16>)
    outs(%empty269 : tensor<196x384xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x384xf16>
  %init271 = tensor.empty() : tensor<196x384xf16>
  %fill272 = linalg.fill ins(%cst : f16) outs(%init271 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm273 = linalg.matmul ins(%add270, %w_s2_l4_q : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill272 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init274 = tensor.empty() : tensor<196x384xf16>
  %fill275 = linalg.fill ins(%cst : f16) outs(%init274 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm276 = linalg.matmul ins(%add270, %w_s2_l4_k : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill275 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init277 = tensor.empty() : tensor<196x384xf16>
  %fill278 = linalg.fill ins(%cst : f16) outs(%init277 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm279 = linalg.matmul ins(%add270, %w_s2_l4_v : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill278 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init280 = tensor.empty() : tensor<196x196xf16>
  %fill281 = linalg.fill ins(%cst : f16) outs(%init280 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm282 = linalg.matmul ins(%mm273, %w_s2_l4_kt : tensor<196x384xf16>, tensor<384x196xf16>)
                          outs(%fill281 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty283 = tensor.empty() : tensor<196x196xf16>
  %relu284 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm282 : tensor<196x196xf16>)
    outs(%empty283 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init285 = tensor.empty() : tensor<196x384xf16>
  %fill286 = linalg.fill ins(%cst : f16) outs(%init285 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm287 = linalg.matmul ins(%relu284, %mm279 : tensor<196x196xf16>, tensor<196x384xf16>)
                          outs(%fill286 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init288 = tensor.empty() : tensor<196x384xf16>
  %fill289 = linalg.fill ins(%cst : f16) outs(%init288 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm290 = linalg.matmul ins(%mm287, %w_s2_l4_o : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill289 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %empty291 = tensor.empty() : tensor<196x384xf16>
  %add292 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm290, %add270 : tensor<196x384xf16>, tensor<196x384xf16>)
    outs(%empty291 : tensor<196x384xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x384xf16>
  %init293 = tensor.empty() : tensor<196x1536xf16>
  %fill294 = linalg.fill ins(%cst : f16) outs(%init293 : tensor<196x1536xf16>) -> tensor<196x1536xf16>
  %mm295 = linalg.matmul ins(%add292, %w_s2_l4_ff_up : tensor<196x384xf16>, tensor<384x1536xf16>)
                          outs(%fill294 : tensor<196x1536xf16>) -> tensor<196x1536xf16>
  %empty296 = tensor.empty() : tensor<196x1536xf16>
  %relu297 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm295 : tensor<196x1536xf16>)
    outs(%empty296 : tensor<196x1536xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x1536xf16>
  %init298 = tensor.empty() : tensor<196x384xf16>
  %fill299 = linalg.fill ins(%cst : f16) outs(%init298 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm300 = linalg.matmul ins(%relu297, %w_s2_l4_ff_dn : tensor<196x1536xf16>, tensor<1536x384xf16>)
                          outs(%fill299 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %empty301 = tensor.empty() : tensor<196x384xf16>
  %add302 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm300, %add292 : tensor<196x384xf16>, tensor<196x384xf16>)
    outs(%empty301 : tensor<196x384xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x384xf16>
  %init303 = tensor.empty() : tensor<196x384xf16>
  %fill304 = linalg.fill ins(%cst : f16) outs(%init303 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm305 = linalg.matmul ins(%add302, %w_s2_l5_q : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill304 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init306 = tensor.empty() : tensor<196x384xf16>
  %fill307 = linalg.fill ins(%cst : f16) outs(%init306 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm308 = linalg.matmul ins(%add302, %w_s2_l5_k : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill307 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init309 = tensor.empty() : tensor<196x384xf16>
  %fill310 = linalg.fill ins(%cst : f16) outs(%init309 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm311 = linalg.matmul ins(%add302, %w_s2_l5_v : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill310 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init312 = tensor.empty() : tensor<196x196xf16>
  %fill313 = linalg.fill ins(%cst : f16) outs(%init312 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %mm314 = linalg.matmul ins(%mm305, %w_s2_l5_kt : tensor<196x384xf16>, tensor<384x196xf16>)
                          outs(%fill313 : tensor<196x196xf16>) -> tensor<196x196xf16>
  %empty315 = tensor.empty() : tensor<196x196xf16>
  %relu316 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm314 : tensor<196x196xf16>)
    outs(%empty315 : tensor<196x196xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x196xf16>
  %init317 = tensor.empty() : tensor<196x384xf16>
  %fill318 = linalg.fill ins(%cst : f16) outs(%init317 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm319 = linalg.matmul ins(%relu316, %mm311 : tensor<196x196xf16>, tensor<196x384xf16>)
                          outs(%fill318 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %init320 = tensor.empty() : tensor<196x384xf16>
  %fill321 = linalg.fill ins(%cst : f16) outs(%init320 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm322 = linalg.matmul ins(%mm319, %w_s2_l5_o : tensor<196x384xf16>, tensor<384x384xf16>)
                          outs(%fill321 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %empty323 = tensor.empty() : tensor<196x384xf16>
  %add324 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm322, %add302 : tensor<196x384xf16>, tensor<196x384xf16>)
    outs(%empty323 : tensor<196x384xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x384xf16>
  %init325 = tensor.empty() : tensor<196x1536xf16>
  %fill326 = linalg.fill ins(%cst : f16) outs(%init325 : tensor<196x1536xf16>) -> tensor<196x1536xf16>
  %mm327 = linalg.matmul ins(%add324, %w_s2_l5_ff_up : tensor<196x384xf16>, tensor<384x1536xf16>)
                          outs(%fill326 : tensor<196x1536xf16>) -> tensor<196x1536xf16>
  %empty328 = tensor.empty() : tensor<196x1536xf16>
  %relu329 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm327 : tensor<196x1536xf16>)
    outs(%empty328 : tensor<196x1536xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<196x1536xf16>
  %init330 = tensor.empty() : tensor<196x384xf16>
  %fill331 = linalg.fill ins(%cst : f16) outs(%init330 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %mm332 = linalg.matmul ins(%relu329, %w_s2_l5_ff_dn : tensor<196x1536xf16>, tensor<1536x384xf16>)
                          outs(%fill331 : tensor<196x384xf16>) -> tensor<196x384xf16>
  %empty333 = tensor.empty() : tensor<196x384xf16>
  %add334 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm332, %add324 : tensor<196x384xf16>, tensor<196x384xf16>)
    outs(%empty333 : tensor<196x384xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<196x384xf16>

  // Patch merging: [196,384] -> [49,768]
  %init335 = tensor.empty() : tensor<196x768xf16>
  %fill336 = linalg.fill ins(%cst : f16) outs(%init335 : tensor<196x768xf16>) -> tensor<196x768xf16>
  %mm337 = linalg.matmul ins(%add334, %w_merge2 : tensor<196x384xf16>, tensor<384x768xf16>)
                          outs(%fill336 : tensor<196x768xf16>) -> tensor<196x768xf16>
  %merge_reshape338 = tensor.empty() : tensor<49x768xf16>
  %fill339 = linalg.fill ins(%cst : f16) outs(%merge_reshape338 : tensor<49x768xf16>) -> tensor<49x768xf16>

  // === Swin Stage 3: seq=49, dim=768 ===
  %init340 = tensor.empty() : tensor<49x768xf16>
  %fill341 = linalg.fill ins(%cst : f16) outs(%init340 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %mm342 = linalg.matmul ins(%fill339, %w_s3_l0_q : tensor<49x768xf16>, tensor<768x768xf16>)
                          outs(%fill341 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %init343 = tensor.empty() : tensor<49x768xf16>
  %fill344 = linalg.fill ins(%cst : f16) outs(%init343 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %mm345 = linalg.matmul ins(%fill339, %w_s3_l0_k : tensor<49x768xf16>, tensor<768x768xf16>)
                          outs(%fill344 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %init346 = tensor.empty() : tensor<49x768xf16>
  %fill347 = linalg.fill ins(%cst : f16) outs(%init346 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %mm348 = linalg.matmul ins(%fill339, %w_s3_l0_v : tensor<49x768xf16>, tensor<768x768xf16>)
                          outs(%fill347 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %init349 = tensor.empty() : tensor<49x49xf16>
  %fill350 = linalg.fill ins(%cst : f16) outs(%init349 : tensor<49x49xf16>) -> tensor<49x49xf16>
  %mm351 = linalg.matmul ins(%mm342, %w_s3_l0_kt : tensor<49x768xf16>, tensor<768x49xf16>)
                          outs(%fill350 : tensor<49x49xf16>) -> tensor<49x49xf16>
  %empty352 = tensor.empty() : tensor<49x49xf16>
  %relu353 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm351 : tensor<49x49xf16>)
    outs(%empty352 : tensor<49x49xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<49x49xf16>
  %init354 = tensor.empty() : tensor<49x768xf16>
  %fill355 = linalg.fill ins(%cst : f16) outs(%init354 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %mm356 = linalg.matmul ins(%relu353, %mm348 : tensor<49x49xf16>, tensor<49x768xf16>)
                          outs(%fill355 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %init357 = tensor.empty() : tensor<49x768xf16>
  %fill358 = linalg.fill ins(%cst : f16) outs(%init357 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %mm359 = linalg.matmul ins(%mm356, %w_s3_l0_o : tensor<49x768xf16>, tensor<768x768xf16>)
                          outs(%fill358 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %empty360 = tensor.empty() : tensor<49x768xf16>
  %add361 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm359, %fill339 : tensor<49x768xf16>, tensor<49x768xf16>)
    outs(%empty360 : tensor<49x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<49x768xf16>
  %init362 = tensor.empty() : tensor<49x3072xf16>
  %fill363 = linalg.fill ins(%cst : f16) outs(%init362 : tensor<49x3072xf16>) -> tensor<49x3072xf16>
  %mm364 = linalg.matmul ins(%add361, %w_s3_l0_ff_up : tensor<49x768xf16>, tensor<768x3072xf16>)
                          outs(%fill363 : tensor<49x3072xf16>) -> tensor<49x3072xf16>
  %empty365 = tensor.empty() : tensor<49x3072xf16>
  %relu366 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm364 : tensor<49x3072xf16>)
    outs(%empty365 : tensor<49x3072xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<49x3072xf16>
  %init367 = tensor.empty() : tensor<49x768xf16>
  %fill368 = linalg.fill ins(%cst : f16) outs(%init367 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %mm369 = linalg.matmul ins(%relu366, %w_s3_l0_ff_dn : tensor<49x3072xf16>, tensor<3072x768xf16>)
                          outs(%fill368 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %empty370 = tensor.empty() : tensor<49x768xf16>
  %add371 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm369, %add361 : tensor<49x768xf16>, tensor<49x768xf16>)
    outs(%empty370 : tensor<49x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<49x768xf16>
  %init372 = tensor.empty() : tensor<49x768xf16>
  %fill373 = linalg.fill ins(%cst : f16) outs(%init372 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %mm374 = linalg.matmul ins(%add371, %w_s3_l1_q : tensor<49x768xf16>, tensor<768x768xf16>)
                          outs(%fill373 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %init375 = tensor.empty() : tensor<49x768xf16>
  %fill376 = linalg.fill ins(%cst : f16) outs(%init375 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %mm377 = linalg.matmul ins(%add371, %w_s3_l1_k : tensor<49x768xf16>, tensor<768x768xf16>)
                          outs(%fill376 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %init378 = tensor.empty() : tensor<49x768xf16>
  %fill379 = linalg.fill ins(%cst : f16) outs(%init378 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %mm380 = linalg.matmul ins(%add371, %w_s3_l1_v : tensor<49x768xf16>, tensor<768x768xf16>)
                          outs(%fill379 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %init381 = tensor.empty() : tensor<49x49xf16>
  %fill382 = linalg.fill ins(%cst : f16) outs(%init381 : tensor<49x49xf16>) -> tensor<49x49xf16>
  %mm383 = linalg.matmul ins(%mm374, %w_s3_l1_kt : tensor<49x768xf16>, tensor<768x49xf16>)
                          outs(%fill382 : tensor<49x49xf16>) -> tensor<49x49xf16>
  %empty384 = tensor.empty() : tensor<49x49xf16>
  %relu385 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm383 : tensor<49x49xf16>)
    outs(%empty384 : tensor<49x49xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<49x49xf16>
  %init386 = tensor.empty() : tensor<49x768xf16>
  %fill387 = linalg.fill ins(%cst : f16) outs(%init386 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %mm388 = linalg.matmul ins(%relu385, %mm380 : tensor<49x49xf16>, tensor<49x768xf16>)
                          outs(%fill387 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %init389 = tensor.empty() : tensor<49x768xf16>
  %fill390 = linalg.fill ins(%cst : f16) outs(%init389 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %mm391 = linalg.matmul ins(%mm388, %w_s3_l1_o : tensor<49x768xf16>, tensor<768x768xf16>)
                          outs(%fill390 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %empty392 = tensor.empty() : tensor<49x768xf16>
  %add393 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm391, %add371 : tensor<49x768xf16>, tensor<49x768xf16>)
    outs(%empty392 : tensor<49x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<49x768xf16>
  %init394 = tensor.empty() : tensor<49x3072xf16>
  %fill395 = linalg.fill ins(%cst : f16) outs(%init394 : tensor<49x3072xf16>) -> tensor<49x3072xf16>
  %mm396 = linalg.matmul ins(%add393, %w_s3_l1_ff_up : tensor<49x768xf16>, tensor<768x3072xf16>)
                          outs(%fill395 : tensor<49x3072xf16>) -> tensor<49x3072xf16>
  %empty397 = tensor.empty() : tensor<49x3072xf16>
  %relu398 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm396 : tensor<49x3072xf16>)
    outs(%empty397 : tensor<49x3072xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<49x3072xf16>
  %init399 = tensor.empty() : tensor<49x768xf16>
  %fill400 = linalg.fill ins(%cst : f16) outs(%init399 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %mm401 = linalg.matmul ins(%relu398, %w_s3_l1_ff_dn : tensor<49x3072xf16>, tensor<3072x768xf16>)
                          outs(%fill400 : tensor<49x768xf16>) -> tensor<49x768xf16>
  %empty402 = tensor.empty() : tensor<49x768xf16>
  %add403 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm401, %add393 : tensor<49x768xf16>, tensor<49x768xf16>)
    outs(%empty402 : tensor<49x768xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<49x768xf16>

  // Head: [49,768] x [768,1000]
  %init404 = tensor.empty() : tensor<49x1000xf16>
  %fill405 = linalg.fill ins(%cst : f16) outs(%init404 : tensor<49x1000xf16>) -> tensor<49x1000xf16>
  %mm406 = linalg.matmul ins(%add403, %w_head : tensor<49x768xf16>, tensor<768x1000xf16>)
                          outs(%fill405 : tensor<49x1000xf16>) -> tensor<49x1000xf16>
  return %mm406 : tensor<49x1000xf16>
}
