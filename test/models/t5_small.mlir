func.func @t5_small(
    %input: tensor<128x512xf16>,
    %w_qe0: tensor<512x512xf16>,
    %w_ke0: tensor<512x512xf16>,
    %w_ve0: tensor<512x512xf16>,
    %w_kte0: tensor<512x128xf16>,
    %w_oe0: tensor<512x512xf16>,
    %w_ffe0_up: tensor<512x2048xf16>,
    %w_ffe0_down: tensor<2048x512xf16>,
    %w_qe1: tensor<512x512xf16>,
    %w_ke1: tensor<512x512xf16>,
    %w_ve1: tensor<512x512xf16>,
    %w_kte1: tensor<512x128xf16>,
    %w_oe1: tensor<512x512xf16>,
    %w_ffe1_up: tensor<512x2048xf16>,
    %w_ffe1_down: tensor<2048x512xf16>,
    %w_qe2: tensor<512x512xf16>,
    %w_ke2: tensor<512x512xf16>,
    %w_ve2: tensor<512x512xf16>,
    %w_kte2: tensor<512x128xf16>,
    %w_oe2: tensor<512x512xf16>,
    %w_ffe2_up: tensor<512x2048xf16>,
    %w_ffe2_down: tensor<2048x512xf16>,
    %w_qe3: tensor<512x512xf16>,
    %w_ke3: tensor<512x512xf16>,
    %w_ve3: tensor<512x512xf16>,
    %w_kte3: tensor<512x128xf16>,
    %w_oe3: tensor<512x512xf16>,
    %w_ffe3_up: tensor<512x2048xf16>,
    %w_ffe3_down: tensor<2048x512xf16>,
    %w_qe4: tensor<512x512xf16>,
    %w_ke4: tensor<512x512xf16>,
    %w_ve4: tensor<512x512xf16>,
    %w_kte4: tensor<512x128xf16>,
    %w_oe4: tensor<512x512xf16>,
    %w_ffe4_up: tensor<512x2048xf16>,
    %w_ffe4_down: tensor<2048x512xf16>,
    %w_qe5: tensor<512x512xf16>,
    %w_ke5: tensor<512x512xf16>,
    %w_ve5: tensor<512x512xf16>,
    %w_kte5: tensor<512x128xf16>,
    %w_oe5: tensor<512x512xf16>,
    %w_ffe5_up: tensor<512x2048xf16>,
    %w_ffe5_down: tensor<2048x512xf16>,
    %w_qd0: tensor<512x512xf16>,
    %w_kd0: tensor<512x512xf16>,
    %w_vd0: tensor<512x512xf16>,
    %w_ktd0: tensor<512x128xf16>,
    %w_od0: tensor<512x512xf16>,
    %w_xqd0: tensor<512x512xf16>,
    %w_xkd0: tensor<512x512xf16>,
    %w_xvd0: tensor<512x512xf16>,
    %w_xktd0: tensor<512x128xf16>,
    %w_xod0: tensor<512x512xf16>,
    %w_ffd0_up: tensor<512x2048xf16>,
    %w_ffd0_down: tensor<2048x512xf16>,
    %w_qd1: tensor<512x512xf16>,
    %w_kd1: tensor<512x512xf16>,
    %w_vd1: tensor<512x512xf16>,
    %w_ktd1: tensor<512x128xf16>,
    %w_od1: tensor<512x512xf16>,
    %w_xqd1: tensor<512x512xf16>,
    %w_xkd1: tensor<512x512xf16>,
    %w_xvd1: tensor<512x512xf16>,
    %w_xktd1: tensor<512x128xf16>,
    %w_xod1: tensor<512x512xf16>,
    %w_ffd1_up: tensor<512x2048xf16>,
    %w_ffd1_down: tensor<2048x512xf16>,
    %w_qd2: tensor<512x512xf16>,
    %w_kd2: tensor<512x512xf16>,
    %w_vd2: tensor<512x512xf16>,
    %w_ktd2: tensor<512x128xf16>,
    %w_od2: tensor<512x512xf16>,
    %w_xqd2: tensor<512x512xf16>,
    %w_xkd2: tensor<512x512xf16>,
    %w_xvd2: tensor<512x512xf16>,
    %w_xktd2: tensor<512x128xf16>,
    %w_xod2: tensor<512x512xf16>,
    %w_ffd2_up: tensor<512x2048xf16>,
    %w_ffd2_down: tensor<2048x512xf16>,
    %w_qd3: tensor<512x512xf16>,
    %w_kd3: tensor<512x512xf16>,
    %w_vd3: tensor<512x512xf16>,
    %w_ktd3: tensor<512x128xf16>,
    %w_od3: tensor<512x512xf16>,
    %w_xqd3: tensor<512x512xf16>,
    %w_xkd3: tensor<512x512xf16>,
    %w_xvd3: tensor<512x512xf16>,
    %w_xktd3: tensor<512x128xf16>,
    %w_xod3: tensor<512x512xf16>,
    %w_ffd3_up: tensor<512x2048xf16>,
    %w_ffd3_down: tensor<2048x512xf16>,
    %w_qd4: tensor<512x512xf16>,
    %w_kd4: tensor<512x512xf16>,
    %w_vd4: tensor<512x512xf16>,
    %w_ktd4: tensor<512x128xf16>,
    %w_od4: tensor<512x512xf16>,
    %w_xqd4: tensor<512x512xf16>,
    %w_xkd4: tensor<512x512xf16>,
    %w_xvd4: tensor<512x512xf16>,
    %w_xktd4: tensor<512x128xf16>,
    %w_xod4: tensor<512x512xf16>,
    %w_ffd4_up: tensor<512x2048xf16>,
    %w_ffd4_down: tensor<2048x512xf16>,
    %w_qd5: tensor<512x512xf16>,
    %w_kd5: tensor<512x512xf16>,
    %w_vd5: tensor<512x512xf16>,
    %w_ktd5: tensor<512x128xf16>,
    %w_od5: tensor<512x512xf16>,
    %w_xqd5: tensor<512x512xf16>,
    %w_xkd5: tensor<512x512xf16>,
    %w_xvd5: tensor<512x512xf16>,
    %w_xktd5: tensor<512x128xf16>,
    %w_xod5: tensor<512x512xf16>,
    %w_ffd5_up: tensor<512x2048xf16>,
    %w_ffd5_down: tensor<2048x512xf16>) -> tensor<128x512xf16> {
  %cst = arith.constant 0.0 : f16

  // === Encoder ===
  // Encoder layer 0
  %init0 = tensor.empty() : tensor<128x512xf16>
  %fill1 = linalg.fill ins(%cst : f16) outs(%init0 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm2 = linalg.matmul ins(%input, %w_qe0 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill1 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init3 = tensor.empty() : tensor<128x512xf16>
  %fill4 = linalg.fill ins(%cst : f16) outs(%init3 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm5 = linalg.matmul ins(%input, %w_ke0 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill4 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init6 = tensor.empty() : tensor<128x512xf16>
  %fill7 = linalg.fill ins(%cst : f16) outs(%init6 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm8 = linalg.matmul ins(%input, %w_ve0 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill7 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init9 = tensor.empty() : tensor<128x128xf16>
  %fill10 = linalg.fill ins(%cst : f16) outs(%init9 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm11 = linalg.matmul ins(%mm2, %w_kte0 : tensor<128x512xf16>, tensor<512x128xf16>)
                          outs(%fill10 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %empty12 = tensor.empty() : tensor<128x128xf16>
  %relu13 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm11 : tensor<128x128xf16>)
    outs(%empty12 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>
  %init14 = tensor.empty() : tensor<128x512xf16>
  %fill15 = linalg.fill ins(%cst : f16) outs(%init14 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm16 = linalg.matmul ins(%relu13, %mm8 : tensor<128x128xf16>, tensor<128x512xf16>)
                          outs(%fill15 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init17 = tensor.empty() : tensor<128x512xf16>
  %fill18 = linalg.fill ins(%cst : f16) outs(%init17 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm19 = linalg.matmul ins(%mm16, %w_oe0 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill18 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty20 = tensor.empty() : tensor<128x512xf16>
  %add21 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm19, %input : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty20 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>
  %init22 = tensor.empty() : tensor<128x2048xf16>
  %fill23 = linalg.fill ins(%cst : f16) outs(%init22 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %mm24 = linalg.matmul ins(%add21, %w_ffe0_up : tensor<128x512xf16>, tensor<512x2048xf16>)
                          outs(%fill23 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %empty25 = tensor.empty() : tensor<128x2048xf16>
  %relu26 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm24 : tensor<128x2048xf16>)
    outs(%empty25 : tensor<128x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x2048xf16>
  %init27 = tensor.empty() : tensor<128x512xf16>
  %fill28 = linalg.fill ins(%cst : f16) outs(%init27 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm29 = linalg.matmul ins(%relu26, %w_ffe0_down : tensor<128x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill28 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty30 = tensor.empty() : tensor<128x512xf16>
  %add31 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm29, %add21 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty30 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>

  // Encoder layer 1
  %init32 = tensor.empty() : tensor<128x512xf16>
  %fill33 = linalg.fill ins(%cst : f16) outs(%init32 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm34 = linalg.matmul ins(%add31, %w_qe1 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill33 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init35 = tensor.empty() : tensor<128x512xf16>
  %fill36 = linalg.fill ins(%cst : f16) outs(%init35 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm37 = linalg.matmul ins(%add31, %w_ke1 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill36 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init38 = tensor.empty() : tensor<128x512xf16>
  %fill39 = linalg.fill ins(%cst : f16) outs(%init38 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm40 = linalg.matmul ins(%add31, %w_ve1 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill39 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init41 = tensor.empty() : tensor<128x128xf16>
  %fill42 = linalg.fill ins(%cst : f16) outs(%init41 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm43 = linalg.matmul ins(%mm34, %w_kte1 : tensor<128x512xf16>, tensor<512x128xf16>)
                          outs(%fill42 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %empty44 = tensor.empty() : tensor<128x128xf16>
  %relu45 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm43 : tensor<128x128xf16>)
    outs(%empty44 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>
  %init46 = tensor.empty() : tensor<128x512xf16>
  %fill47 = linalg.fill ins(%cst : f16) outs(%init46 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm48 = linalg.matmul ins(%relu45, %mm40 : tensor<128x128xf16>, tensor<128x512xf16>)
                          outs(%fill47 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init49 = tensor.empty() : tensor<128x512xf16>
  %fill50 = linalg.fill ins(%cst : f16) outs(%init49 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm51 = linalg.matmul ins(%mm48, %w_oe1 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill50 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty52 = tensor.empty() : tensor<128x512xf16>
  %add53 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm51, %add31 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty52 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>
  %init54 = tensor.empty() : tensor<128x2048xf16>
  %fill55 = linalg.fill ins(%cst : f16) outs(%init54 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %mm56 = linalg.matmul ins(%add53, %w_ffe1_up : tensor<128x512xf16>, tensor<512x2048xf16>)
                          outs(%fill55 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %empty57 = tensor.empty() : tensor<128x2048xf16>
  %relu58 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm56 : tensor<128x2048xf16>)
    outs(%empty57 : tensor<128x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x2048xf16>
  %init59 = tensor.empty() : tensor<128x512xf16>
  %fill60 = linalg.fill ins(%cst : f16) outs(%init59 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm61 = linalg.matmul ins(%relu58, %w_ffe1_down : tensor<128x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill60 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty62 = tensor.empty() : tensor<128x512xf16>
  %add63 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm61, %add53 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty62 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>

  // Encoder layer 2
  %init64 = tensor.empty() : tensor<128x512xf16>
  %fill65 = linalg.fill ins(%cst : f16) outs(%init64 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm66 = linalg.matmul ins(%add63, %w_qe2 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill65 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init67 = tensor.empty() : tensor<128x512xf16>
  %fill68 = linalg.fill ins(%cst : f16) outs(%init67 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm69 = linalg.matmul ins(%add63, %w_ke2 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill68 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init70 = tensor.empty() : tensor<128x512xf16>
  %fill71 = linalg.fill ins(%cst : f16) outs(%init70 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm72 = linalg.matmul ins(%add63, %w_ve2 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill71 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init73 = tensor.empty() : tensor<128x128xf16>
  %fill74 = linalg.fill ins(%cst : f16) outs(%init73 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm75 = linalg.matmul ins(%mm66, %w_kte2 : tensor<128x512xf16>, tensor<512x128xf16>)
                          outs(%fill74 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %empty76 = tensor.empty() : tensor<128x128xf16>
  %relu77 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm75 : tensor<128x128xf16>)
    outs(%empty76 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>
  %init78 = tensor.empty() : tensor<128x512xf16>
  %fill79 = linalg.fill ins(%cst : f16) outs(%init78 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm80 = linalg.matmul ins(%relu77, %mm72 : tensor<128x128xf16>, tensor<128x512xf16>)
                          outs(%fill79 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init81 = tensor.empty() : tensor<128x512xf16>
  %fill82 = linalg.fill ins(%cst : f16) outs(%init81 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm83 = linalg.matmul ins(%mm80, %w_oe2 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill82 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty84 = tensor.empty() : tensor<128x512xf16>
  %add85 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm83, %add63 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty84 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>
  %init86 = tensor.empty() : tensor<128x2048xf16>
  %fill87 = linalg.fill ins(%cst : f16) outs(%init86 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %mm88 = linalg.matmul ins(%add85, %w_ffe2_up : tensor<128x512xf16>, tensor<512x2048xf16>)
                          outs(%fill87 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %empty89 = tensor.empty() : tensor<128x2048xf16>
  %relu90 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm88 : tensor<128x2048xf16>)
    outs(%empty89 : tensor<128x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x2048xf16>
  %init91 = tensor.empty() : tensor<128x512xf16>
  %fill92 = linalg.fill ins(%cst : f16) outs(%init91 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm93 = linalg.matmul ins(%relu90, %w_ffe2_down : tensor<128x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill92 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty94 = tensor.empty() : tensor<128x512xf16>
  %add95 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm93, %add85 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty94 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>

  // Encoder layer 3
  %init96 = tensor.empty() : tensor<128x512xf16>
  %fill97 = linalg.fill ins(%cst : f16) outs(%init96 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm98 = linalg.matmul ins(%add95, %w_qe3 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill97 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init99 = tensor.empty() : tensor<128x512xf16>
  %fill100 = linalg.fill ins(%cst : f16) outs(%init99 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm101 = linalg.matmul ins(%add95, %w_ke3 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill100 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init102 = tensor.empty() : tensor<128x512xf16>
  %fill103 = linalg.fill ins(%cst : f16) outs(%init102 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm104 = linalg.matmul ins(%add95, %w_ve3 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill103 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init105 = tensor.empty() : tensor<128x128xf16>
  %fill106 = linalg.fill ins(%cst : f16) outs(%init105 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm107 = linalg.matmul ins(%mm98, %w_kte3 : tensor<128x512xf16>, tensor<512x128xf16>)
                          outs(%fill106 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %empty108 = tensor.empty() : tensor<128x128xf16>
  %relu109 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm107 : tensor<128x128xf16>)
    outs(%empty108 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>
  %init110 = tensor.empty() : tensor<128x512xf16>
  %fill111 = linalg.fill ins(%cst : f16) outs(%init110 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm112 = linalg.matmul ins(%relu109, %mm104 : tensor<128x128xf16>, tensor<128x512xf16>)
                          outs(%fill111 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init113 = tensor.empty() : tensor<128x512xf16>
  %fill114 = linalg.fill ins(%cst : f16) outs(%init113 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm115 = linalg.matmul ins(%mm112, %w_oe3 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill114 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty116 = tensor.empty() : tensor<128x512xf16>
  %add117 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm115, %add95 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty116 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>
  %init118 = tensor.empty() : tensor<128x2048xf16>
  %fill119 = linalg.fill ins(%cst : f16) outs(%init118 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %mm120 = linalg.matmul ins(%add117, %w_ffe3_up : tensor<128x512xf16>, tensor<512x2048xf16>)
                          outs(%fill119 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %empty121 = tensor.empty() : tensor<128x2048xf16>
  %relu122 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm120 : tensor<128x2048xf16>)
    outs(%empty121 : tensor<128x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x2048xf16>
  %init123 = tensor.empty() : tensor<128x512xf16>
  %fill124 = linalg.fill ins(%cst : f16) outs(%init123 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm125 = linalg.matmul ins(%relu122, %w_ffe3_down : tensor<128x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill124 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty126 = tensor.empty() : tensor<128x512xf16>
  %add127 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm125, %add117 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty126 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>

  // Encoder layer 4
  %init128 = tensor.empty() : tensor<128x512xf16>
  %fill129 = linalg.fill ins(%cst : f16) outs(%init128 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm130 = linalg.matmul ins(%add127, %w_qe4 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill129 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init131 = tensor.empty() : tensor<128x512xf16>
  %fill132 = linalg.fill ins(%cst : f16) outs(%init131 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm133 = linalg.matmul ins(%add127, %w_ke4 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill132 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init134 = tensor.empty() : tensor<128x512xf16>
  %fill135 = linalg.fill ins(%cst : f16) outs(%init134 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm136 = linalg.matmul ins(%add127, %w_ve4 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill135 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init137 = tensor.empty() : tensor<128x128xf16>
  %fill138 = linalg.fill ins(%cst : f16) outs(%init137 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm139 = linalg.matmul ins(%mm130, %w_kte4 : tensor<128x512xf16>, tensor<512x128xf16>)
                          outs(%fill138 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %empty140 = tensor.empty() : tensor<128x128xf16>
  %relu141 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm139 : tensor<128x128xf16>)
    outs(%empty140 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>
  %init142 = tensor.empty() : tensor<128x512xf16>
  %fill143 = linalg.fill ins(%cst : f16) outs(%init142 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm144 = linalg.matmul ins(%relu141, %mm136 : tensor<128x128xf16>, tensor<128x512xf16>)
                          outs(%fill143 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init145 = tensor.empty() : tensor<128x512xf16>
  %fill146 = linalg.fill ins(%cst : f16) outs(%init145 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm147 = linalg.matmul ins(%mm144, %w_oe4 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill146 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty148 = tensor.empty() : tensor<128x512xf16>
  %add149 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm147, %add127 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty148 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>
  %init150 = tensor.empty() : tensor<128x2048xf16>
  %fill151 = linalg.fill ins(%cst : f16) outs(%init150 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %mm152 = linalg.matmul ins(%add149, %w_ffe4_up : tensor<128x512xf16>, tensor<512x2048xf16>)
                          outs(%fill151 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %empty153 = tensor.empty() : tensor<128x2048xf16>
  %relu154 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm152 : tensor<128x2048xf16>)
    outs(%empty153 : tensor<128x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x2048xf16>
  %init155 = tensor.empty() : tensor<128x512xf16>
  %fill156 = linalg.fill ins(%cst : f16) outs(%init155 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm157 = linalg.matmul ins(%relu154, %w_ffe4_down : tensor<128x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill156 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty158 = tensor.empty() : tensor<128x512xf16>
  %add159 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm157, %add149 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty158 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>

  // Encoder layer 5
  %init160 = tensor.empty() : tensor<128x512xf16>
  %fill161 = linalg.fill ins(%cst : f16) outs(%init160 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm162 = linalg.matmul ins(%add159, %w_qe5 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill161 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init163 = tensor.empty() : tensor<128x512xf16>
  %fill164 = linalg.fill ins(%cst : f16) outs(%init163 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm165 = linalg.matmul ins(%add159, %w_ke5 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill164 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init166 = tensor.empty() : tensor<128x512xf16>
  %fill167 = linalg.fill ins(%cst : f16) outs(%init166 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm168 = linalg.matmul ins(%add159, %w_ve5 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill167 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init169 = tensor.empty() : tensor<128x128xf16>
  %fill170 = linalg.fill ins(%cst : f16) outs(%init169 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm171 = linalg.matmul ins(%mm162, %w_kte5 : tensor<128x512xf16>, tensor<512x128xf16>)
                          outs(%fill170 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %empty172 = tensor.empty() : tensor<128x128xf16>
  %relu173 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm171 : tensor<128x128xf16>)
    outs(%empty172 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>
  %init174 = tensor.empty() : tensor<128x512xf16>
  %fill175 = linalg.fill ins(%cst : f16) outs(%init174 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm176 = linalg.matmul ins(%relu173, %mm168 : tensor<128x128xf16>, tensor<128x512xf16>)
                          outs(%fill175 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init177 = tensor.empty() : tensor<128x512xf16>
  %fill178 = linalg.fill ins(%cst : f16) outs(%init177 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm179 = linalg.matmul ins(%mm176, %w_oe5 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill178 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty180 = tensor.empty() : tensor<128x512xf16>
  %add181 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm179, %add159 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty180 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>
  %init182 = tensor.empty() : tensor<128x2048xf16>
  %fill183 = linalg.fill ins(%cst : f16) outs(%init182 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %mm184 = linalg.matmul ins(%add181, %w_ffe5_up : tensor<128x512xf16>, tensor<512x2048xf16>)
                          outs(%fill183 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %empty185 = tensor.empty() : tensor<128x2048xf16>
  %relu186 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm184 : tensor<128x2048xf16>)
    outs(%empty185 : tensor<128x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x2048xf16>
  %init187 = tensor.empty() : tensor<128x512xf16>
  %fill188 = linalg.fill ins(%cst : f16) outs(%init187 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm189 = linalg.matmul ins(%relu186, %w_ffe5_down : tensor<128x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill188 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty190 = tensor.empty() : tensor<128x512xf16>
  %add191 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm189, %add181 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty190 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>

  // === Decoder ===
  // Decoder layer 0
  %init192 = tensor.empty() : tensor<128x512xf16>
  %fill193 = linalg.fill ins(%cst : f16) outs(%init192 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm194 = linalg.matmul ins(%add191, %w_qd0 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill193 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init195 = tensor.empty() : tensor<128x512xf16>
  %fill196 = linalg.fill ins(%cst : f16) outs(%init195 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm197 = linalg.matmul ins(%add191, %w_kd0 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill196 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init198 = tensor.empty() : tensor<128x512xf16>
  %fill199 = linalg.fill ins(%cst : f16) outs(%init198 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm200 = linalg.matmul ins(%add191, %w_vd0 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill199 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init201 = tensor.empty() : tensor<128x128xf16>
  %fill202 = linalg.fill ins(%cst : f16) outs(%init201 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm203 = linalg.matmul ins(%mm194, %w_ktd0 : tensor<128x512xf16>, tensor<512x128xf16>)
                          outs(%fill202 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %empty204 = tensor.empty() : tensor<128x128xf16>
  %relu205 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm203 : tensor<128x128xf16>)
    outs(%empty204 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>
  %init206 = tensor.empty() : tensor<128x512xf16>
  %fill207 = linalg.fill ins(%cst : f16) outs(%init206 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm208 = linalg.matmul ins(%relu205, %mm200 : tensor<128x128xf16>, tensor<128x512xf16>)
                          outs(%fill207 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init209 = tensor.empty() : tensor<128x512xf16>
  %fill210 = linalg.fill ins(%cst : f16) outs(%init209 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm211 = linalg.matmul ins(%mm208, %w_od0 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill210 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty212 = tensor.empty() : tensor<128x512xf16>
  %add213 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm211, %add191 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty212 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>
  %init214 = tensor.empty() : tensor<128x512xf16>
  %fill215 = linalg.fill ins(%cst : f16) outs(%init214 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm216 = linalg.matmul ins(%add213, %w_xqd0 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill215 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init217 = tensor.empty() : tensor<128x512xf16>
  %fill218 = linalg.fill ins(%cst : f16) outs(%init217 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm219 = linalg.matmul ins(%add191, %w_xkd0 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill218 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init220 = tensor.empty() : tensor<128x512xf16>
  %fill221 = linalg.fill ins(%cst : f16) outs(%init220 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm222 = linalg.matmul ins(%add191, %w_xvd0 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill221 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init223 = tensor.empty() : tensor<128x128xf16>
  %fill224 = linalg.fill ins(%cst : f16) outs(%init223 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm225 = linalg.matmul ins(%mm216, %w_xktd0 : tensor<128x512xf16>, tensor<512x128xf16>)
                          outs(%fill224 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %empty226 = tensor.empty() : tensor<128x128xf16>
  %relu227 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm225 : tensor<128x128xf16>)
    outs(%empty226 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>
  %init228 = tensor.empty() : tensor<128x512xf16>
  %fill229 = linalg.fill ins(%cst : f16) outs(%init228 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm230 = linalg.matmul ins(%relu227, %mm222 : tensor<128x128xf16>, tensor<128x512xf16>)
                          outs(%fill229 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init231 = tensor.empty() : tensor<128x512xf16>
  %fill232 = linalg.fill ins(%cst : f16) outs(%init231 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm233 = linalg.matmul ins(%mm230, %w_xod0 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill232 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty234 = tensor.empty() : tensor<128x512xf16>
  %add235 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm233, %add213 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty234 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>
  %init236 = tensor.empty() : tensor<128x2048xf16>
  %fill237 = linalg.fill ins(%cst : f16) outs(%init236 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %mm238 = linalg.matmul ins(%add235, %w_ffd0_up : tensor<128x512xf16>, tensor<512x2048xf16>)
                          outs(%fill237 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %empty239 = tensor.empty() : tensor<128x2048xf16>
  %relu240 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm238 : tensor<128x2048xf16>)
    outs(%empty239 : tensor<128x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x2048xf16>
  %init241 = tensor.empty() : tensor<128x512xf16>
  %fill242 = linalg.fill ins(%cst : f16) outs(%init241 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm243 = linalg.matmul ins(%relu240, %w_ffd0_down : tensor<128x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill242 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty244 = tensor.empty() : tensor<128x512xf16>
  %add245 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm243, %add235 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty244 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>

  // Decoder layer 1
  %init246 = tensor.empty() : tensor<128x512xf16>
  %fill247 = linalg.fill ins(%cst : f16) outs(%init246 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm248 = linalg.matmul ins(%add245, %w_qd1 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill247 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init249 = tensor.empty() : tensor<128x512xf16>
  %fill250 = linalg.fill ins(%cst : f16) outs(%init249 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm251 = linalg.matmul ins(%add245, %w_kd1 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill250 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init252 = tensor.empty() : tensor<128x512xf16>
  %fill253 = linalg.fill ins(%cst : f16) outs(%init252 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm254 = linalg.matmul ins(%add245, %w_vd1 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill253 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init255 = tensor.empty() : tensor<128x128xf16>
  %fill256 = linalg.fill ins(%cst : f16) outs(%init255 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm257 = linalg.matmul ins(%mm248, %w_ktd1 : tensor<128x512xf16>, tensor<512x128xf16>)
                          outs(%fill256 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %empty258 = tensor.empty() : tensor<128x128xf16>
  %relu259 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm257 : tensor<128x128xf16>)
    outs(%empty258 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>
  %init260 = tensor.empty() : tensor<128x512xf16>
  %fill261 = linalg.fill ins(%cst : f16) outs(%init260 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm262 = linalg.matmul ins(%relu259, %mm254 : tensor<128x128xf16>, tensor<128x512xf16>)
                          outs(%fill261 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init263 = tensor.empty() : tensor<128x512xf16>
  %fill264 = linalg.fill ins(%cst : f16) outs(%init263 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm265 = linalg.matmul ins(%mm262, %w_od1 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill264 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty266 = tensor.empty() : tensor<128x512xf16>
  %add267 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm265, %add245 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty266 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>
  %init268 = tensor.empty() : tensor<128x512xf16>
  %fill269 = linalg.fill ins(%cst : f16) outs(%init268 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm270 = linalg.matmul ins(%add267, %w_xqd1 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill269 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init271 = tensor.empty() : tensor<128x512xf16>
  %fill272 = linalg.fill ins(%cst : f16) outs(%init271 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm273 = linalg.matmul ins(%add191, %w_xkd1 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill272 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init274 = tensor.empty() : tensor<128x512xf16>
  %fill275 = linalg.fill ins(%cst : f16) outs(%init274 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm276 = linalg.matmul ins(%add191, %w_xvd1 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill275 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init277 = tensor.empty() : tensor<128x128xf16>
  %fill278 = linalg.fill ins(%cst : f16) outs(%init277 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm279 = linalg.matmul ins(%mm270, %w_xktd1 : tensor<128x512xf16>, tensor<512x128xf16>)
                          outs(%fill278 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %empty280 = tensor.empty() : tensor<128x128xf16>
  %relu281 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm279 : tensor<128x128xf16>)
    outs(%empty280 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>
  %init282 = tensor.empty() : tensor<128x512xf16>
  %fill283 = linalg.fill ins(%cst : f16) outs(%init282 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm284 = linalg.matmul ins(%relu281, %mm276 : tensor<128x128xf16>, tensor<128x512xf16>)
                          outs(%fill283 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init285 = tensor.empty() : tensor<128x512xf16>
  %fill286 = linalg.fill ins(%cst : f16) outs(%init285 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm287 = linalg.matmul ins(%mm284, %w_xod1 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill286 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty288 = tensor.empty() : tensor<128x512xf16>
  %add289 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm287, %add267 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty288 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>
  %init290 = tensor.empty() : tensor<128x2048xf16>
  %fill291 = linalg.fill ins(%cst : f16) outs(%init290 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %mm292 = linalg.matmul ins(%add289, %w_ffd1_up : tensor<128x512xf16>, tensor<512x2048xf16>)
                          outs(%fill291 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %empty293 = tensor.empty() : tensor<128x2048xf16>
  %relu294 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm292 : tensor<128x2048xf16>)
    outs(%empty293 : tensor<128x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x2048xf16>
  %init295 = tensor.empty() : tensor<128x512xf16>
  %fill296 = linalg.fill ins(%cst : f16) outs(%init295 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm297 = linalg.matmul ins(%relu294, %w_ffd1_down : tensor<128x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill296 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty298 = tensor.empty() : tensor<128x512xf16>
  %add299 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm297, %add289 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty298 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>

  // Decoder layer 2
  %init300 = tensor.empty() : tensor<128x512xf16>
  %fill301 = linalg.fill ins(%cst : f16) outs(%init300 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm302 = linalg.matmul ins(%add299, %w_qd2 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill301 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init303 = tensor.empty() : tensor<128x512xf16>
  %fill304 = linalg.fill ins(%cst : f16) outs(%init303 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm305 = linalg.matmul ins(%add299, %w_kd2 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill304 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init306 = tensor.empty() : tensor<128x512xf16>
  %fill307 = linalg.fill ins(%cst : f16) outs(%init306 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm308 = linalg.matmul ins(%add299, %w_vd2 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill307 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init309 = tensor.empty() : tensor<128x128xf16>
  %fill310 = linalg.fill ins(%cst : f16) outs(%init309 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm311 = linalg.matmul ins(%mm302, %w_ktd2 : tensor<128x512xf16>, tensor<512x128xf16>)
                          outs(%fill310 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %empty312 = tensor.empty() : tensor<128x128xf16>
  %relu313 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm311 : tensor<128x128xf16>)
    outs(%empty312 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>
  %init314 = tensor.empty() : tensor<128x512xf16>
  %fill315 = linalg.fill ins(%cst : f16) outs(%init314 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm316 = linalg.matmul ins(%relu313, %mm308 : tensor<128x128xf16>, tensor<128x512xf16>)
                          outs(%fill315 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init317 = tensor.empty() : tensor<128x512xf16>
  %fill318 = linalg.fill ins(%cst : f16) outs(%init317 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm319 = linalg.matmul ins(%mm316, %w_od2 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill318 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty320 = tensor.empty() : tensor<128x512xf16>
  %add321 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm319, %add299 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty320 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>
  %init322 = tensor.empty() : tensor<128x512xf16>
  %fill323 = linalg.fill ins(%cst : f16) outs(%init322 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm324 = linalg.matmul ins(%add321, %w_xqd2 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill323 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init325 = tensor.empty() : tensor<128x512xf16>
  %fill326 = linalg.fill ins(%cst : f16) outs(%init325 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm327 = linalg.matmul ins(%add191, %w_xkd2 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill326 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init328 = tensor.empty() : tensor<128x512xf16>
  %fill329 = linalg.fill ins(%cst : f16) outs(%init328 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm330 = linalg.matmul ins(%add191, %w_xvd2 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill329 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init331 = tensor.empty() : tensor<128x128xf16>
  %fill332 = linalg.fill ins(%cst : f16) outs(%init331 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm333 = linalg.matmul ins(%mm324, %w_xktd2 : tensor<128x512xf16>, tensor<512x128xf16>)
                          outs(%fill332 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %empty334 = tensor.empty() : tensor<128x128xf16>
  %relu335 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm333 : tensor<128x128xf16>)
    outs(%empty334 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>
  %init336 = tensor.empty() : tensor<128x512xf16>
  %fill337 = linalg.fill ins(%cst : f16) outs(%init336 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm338 = linalg.matmul ins(%relu335, %mm330 : tensor<128x128xf16>, tensor<128x512xf16>)
                          outs(%fill337 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init339 = tensor.empty() : tensor<128x512xf16>
  %fill340 = linalg.fill ins(%cst : f16) outs(%init339 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm341 = linalg.matmul ins(%mm338, %w_xod2 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill340 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty342 = tensor.empty() : tensor<128x512xf16>
  %add343 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm341, %add321 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty342 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>
  %init344 = tensor.empty() : tensor<128x2048xf16>
  %fill345 = linalg.fill ins(%cst : f16) outs(%init344 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %mm346 = linalg.matmul ins(%add343, %w_ffd2_up : tensor<128x512xf16>, tensor<512x2048xf16>)
                          outs(%fill345 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %empty347 = tensor.empty() : tensor<128x2048xf16>
  %relu348 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm346 : tensor<128x2048xf16>)
    outs(%empty347 : tensor<128x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x2048xf16>
  %init349 = tensor.empty() : tensor<128x512xf16>
  %fill350 = linalg.fill ins(%cst : f16) outs(%init349 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm351 = linalg.matmul ins(%relu348, %w_ffd2_down : tensor<128x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill350 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty352 = tensor.empty() : tensor<128x512xf16>
  %add353 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm351, %add343 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty352 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>

  // Decoder layer 3
  %init354 = tensor.empty() : tensor<128x512xf16>
  %fill355 = linalg.fill ins(%cst : f16) outs(%init354 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm356 = linalg.matmul ins(%add353, %w_qd3 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill355 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init357 = tensor.empty() : tensor<128x512xf16>
  %fill358 = linalg.fill ins(%cst : f16) outs(%init357 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm359 = linalg.matmul ins(%add353, %w_kd3 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill358 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init360 = tensor.empty() : tensor<128x512xf16>
  %fill361 = linalg.fill ins(%cst : f16) outs(%init360 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm362 = linalg.matmul ins(%add353, %w_vd3 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill361 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init363 = tensor.empty() : tensor<128x128xf16>
  %fill364 = linalg.fill ins(%cst : f16) outs(%init363 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm365 = linalg.matmul ins(%mm356, %w_ktd3 : tensor<128x512xf16>, tensor<512x128xf16>)
                          outs(%fill364 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %empty366 = tensor.empty() : tensor<128x128xf16>
  %relu367 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm365 : tensor<128x128xf16>)
    outs(%empty366 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>
  %init368 = tensor.empty() : tensor<128x512xf16>
  %fill369 = linalg.fill ins(%cst : f16) outs(%init368 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm370 = linalg.matmul ins(%relu367, %mm362 : tensor<128x128xf16>, tensor<128x512xf16>)
                          outs(%fill369 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init371 = tensor.empty() : tensor<128x512xf16>
  %fill372 = linalg.fill ins(%cst : f16) outs(%init371 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm373 = linalg.matmul ins(%mm370, %w_od3 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill372 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty374 = tensor.empty() : tensor<128x512xf16>
  %add375 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm373, %add353 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty374 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>
  %init376 = tensor.empty() : tensor<128x512xf16>
  %fill377 = linalg.fill ins(%cst : f16) outs(%init376 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm378 = linalg.matmul ins(%add375, %w_xqd3 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill377 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init379 = tensor.empty() : tensor<128x512xf16>
  %fill380 = linalg.fill ins(%cst : f16) outs(%init379 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm381 = linalg.matmul ins(%add191, %w_xkd3 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill380 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init382 = tensor.empty() : tensor<128x512xf16>
  %fill383 = linalg.fill ins(%cst : f16) outs(%init382 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm384 = linalg.matmul ins(%add191, %w_xvd3 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill383 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init385 = tensor.empty() : tensor<128x128xf16>
  %fill386 = linalg.fill ins(%cst : f16) outs(%init385 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm387 = linalg.matmul ins(%mm378, %w_xktd3 : tensor<128x512xf16>, tensor<512x128xf16>)
                          outs(%fill386 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %empty388 = tensor.empty() : tensor<128x128xf16>
  %relu389 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm387 : tensor<128x128xf16>)
    outs(%empty388 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>
  %init390 = tensor.empty() : tensor<128x512xf16>
  %fill391 = linalg.fill ins(%cst : f16) outs(%init390 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm392 = linalg.matmul ins(%relu389, %mm384 : tensor<128x128xf16>, tensor<128x512xf16>)
                          outs(%fill391 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init393 = tensor.empty() : tensor<128x512xf16>
  %fill394 = linalg.fill ins(%cst : f16) outs(%init393 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm395 = linalg.matmul ins(%mm392, %w_xod3 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill394 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty396 = tensor.empty() : tensor<128x512xf16>
  %add397 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm395, %add375 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty396 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>
  %init398 = tensor.empty() : tensor<128x2048xf16>
  %fill399 = linalg.fill ins(%cst : f16) outs(%init398 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %mm400 = linalg.matmul ins(%add397, %w_ffd3_up : tensor<128x512xf16>, tensor<512x2048xf16>)
                          outs(%fill399 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %empty401 = tensor.empty() : tensor<128x2048xf16>
  %relu402 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm400 : tensor<128x2048xf16>)
    outs(%empty401 : tensor<128x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x2048xf16>
  %init403 = tensor.empty() : tensor<128x512xf16>
  %fill404 = linalg.fill ins(%cst : f16) outs(%init403 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm405 = linalg.matmul ins(%relu402, %w_ffd3_down : tensor<128x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill404 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty406 = tensor.empty() : tensor<128x512xf16>
  %add407 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm405, %add397 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty406 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>

  // Decoder layer 4
  %init408 = tensor.empty() : tensor<128x512xf16>
  %fill409 = linalg.fill ins(%cst : f16) outs(%init408 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm410 = linalg.matmul ins(%add407, %w_qd4 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill409 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init411 = tensor.empty() : tensor<128x512xf16>
  %fill412 = linalg.fill ins(%cst : f16) outs(%init411 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm413 = linalg.matmul ins(%add407, %w_kd4 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill412 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init414 = tensor.empty() : tensor<128x512xf16>
  %fill415 = linalg.fill ins(%cst : f16) outs(%init414 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm416 = linalg.matmul ins(%add407, %w_vd4 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill415 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init417 = tensor.empty() : tensor<128x128xf16>
  %fill418 = linalg.fill ins(%cst : f16) outs(%init417 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm419 = linalg.matmul ins(%mm410, %w_ktd4 : tensor<128x512xf16>, tensor<512x128xf16>)
                          outs(%fill418 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %empty420 = tensor.empty() : tensor<128x128xf16>
  %relu421 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm419 : tensor<128x128xf16>)
    outs(%empty420 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>
  %init422 = tensor.empty() : tensor<128x512xf16>
  %fill423 = linalg.fill ins(%cst : f16) outs(%init422 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm424 = linalg.matmul ins(%relu421, %mm416 : tensor<128x128xf16>, tensor<128x512xf16>)
                          outs(%fill423 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init425 = tensor.empty() : tensor<128x512xf16>
  %fill426 = linalg.fill ins(%cst : f16) outs(%init425 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm427 = linalg.matmul ins(%mm424, %w_od4 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill426 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty428 = tensor.empty() : tensor<128x512xf16>
  %add429 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm427, %add407 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty428 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>
  %init430 = tensor.empty() : tensor<128x512xf16>
  %fill431 = linalg.fill ins(%cst : f16) outs(%init430 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm432 = linalg.matmul ins(%add429, %w_xqd4 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill431 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init433 = tensor.empty() : tensor<128x512xf16>
  %fill434 = linalg.fill ins(%cst : f16) outs(%init433 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm435 = linalg.matmul ins(%add191, %w_xkd4 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill434 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init436 = tensor.empty() : tensor<128x512xf16>
  %fill437 = linalg.fill ins(%cst : f16) outs(%init436 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm438 = linalg.matmul ins(%add191, %w_xvd4 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill437 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init439 = tensor.empty() : tensor<128x128xf16>
  %fill440 = linalg.fill ins(%cst : f16) outs(%init439 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm441 = linalg.matmul ins(%mm432, %w_xktd4 : tensor<128x512xf16>, tensor<512x128xf16>)
                          outs(%fill440 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %empty442 = tensor.empty() : tensor<128x128xf16>
  %relu443 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm441 : tensor<128x128xf16>)
    outs(%empty442 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>
  %init444 = tensor.empty() : tensor<128x512xf16>
  %fill445 = linalg.fill ins(%cst : f16) outs(%init444 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm446 = linalg.matmul ins(%relu443, %mm438 : tensor<128x128xf16>, tensor<128x512xf16>)
                          outs(%fill445 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init447 = tensor.empty() : tensor<128x512xf16>
  %fill448 = linalg.fill ins(%cst : f16) outs(%init447 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm449 = linalg.matmul ins(%mm446, %w_xod4 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill448 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty450 = tensor.empty() : tensor<128x512xf16>
  %add451 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm449, %add429 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty450 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>
  %init452 = tensor.empty() : tensor<128x2048xf16>
  %fill453 = linalg.fill ins(%cst : f16) outs(%init452 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %mm454 = linalg.matmul ins(%add451, %w_ffd4_up : tensor<128x512xf16>, tensor<512x2048xf16>)
                          outs(%fill453 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %empty455 = tensor.empty() : tensor<128x2048xf16>
  %relu456 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm454 : tensor<128x2048xf16>)
    outs(%empty455 : tensor<128x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x2048xf16>
  %init457 = tensor.empty() : tensor<128x512xf16>
  %fill458 = linalg.fill ins(%cst : f16) outs(%init457 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm459 = linalg.matmul ins(%relu456, %w_ffd4_down : tensor<128x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill458 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty460 = tensor.empty() : tensor<128x512xf16>
  %add461 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm459, %add451 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty460 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>

  // Decoder layer 5
  %init462 = tensor.empty() : tensor<128x512xf16>
  %fill463 = linalg.fill ins(%cst : f16) outs(%init462 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm464 = linalg.matmul ins(%add461, %w_qd5 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill463 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init465 = tensor.empty() : tensor<128x512xf16>
  %fill466 = linalg.fill ins(%cst : f16) outs(%init465 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm467 = linalg.matmul ins(%add461, %w_kd5 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill466 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init468 = tensor.empty() : tensor<128x512xf16>
  %fill469 = linalg.fill ins(%cst : f16) outs(%init468 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm470 = linalg.matmul ins(%add461, %w_vd5 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill469 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init471 = tensor.empty() : tensor<128x128xf16>
  %fill472 = linalg.fill ins(%cst : f16) outs(%init471 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm473 = linalg.matmul ins(%mm464, %w_ktd5 : tensor<128x512xf16>, tensor<512x128xf16>)
                          outs(%fill472 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %empty474 = tensor.empty() : tensor<128x128xf16>
  %relu475 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm473 : tensor<128x128xf16>)
    outs(%empty474 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>
  %init476 = tensor.empty() : tensor<128x512xf16>
  %fill477 = linalg.fill ins(%cst : f16) outs(%init476 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm478 = linalg.matmul ins(%relu475, %mm470 : tensor<128x128xf16>, tensor<128x512xf16>)
                          outs(%fill477 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init479 = tensor.empty() : tensor<128x512xf16>
  %fill480 = linalg.fill ins(%cst : f16) outs(%init479 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm481 = linalg.matmul ins(%mm478, %w_od5 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill480 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty482 = tensor.empty() : tensor<128x512xf16>
  %add483 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm481, %add461 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty482 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>
  %init484 = tensor.empty() : tensor<128x512xf16>
  %fill485 = linalg.fill ins(%cst : f16) outs(%init484 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm486 = linalg.matmul ins(%add483, %w_xqd5 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill485 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init487 = tensor.empty() : tensor<128x512xf16>
  %fill488 = linalg.fill ins(%cst : f16) outs(%init487 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm489 = linalg.matmul ins(%add191, %w_xkd5 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill488 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init490 = tensor.empty() : tensor<128x512xf16>
  %fill491 = linalg.fill ins(%cst : f16) outs(%init490 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm492 = linalg.matmul ins(%add191, %w_xvd5 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill491 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init493 = tensor.empty() : tensor<128x128xf16>
  %fill494 = linalg.fill ins(%cst : f16) outs(%init493 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm495 = linalg.matmul ins(%mm486, %w_xktd5 : tensor<128x512xf16>, tensor<512x128xf16>)
                          outs(%fill494 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %empty496 = tensor.empty() : tensor<128x128xf16>
  %relu497 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm495 : tensor<128x128xf16>)
    outs(%empty496 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x128xf16>
  %init498 = tensor.empty() : tensor<128x512xf16>
  %fill499 = linalg.fill ins(%cst : f16) outs(%init498 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm500 = linalg.matmul ins(%relu497, %mm492 : tensor<128x128xf16>, tensor<128x512xf16>)
                          outs(%fill499 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %init501 = tensor.empty() : tensor<128x512xf16>
  %fill502 = linalg.fill ins(%cst : f16) outs(%init501 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm503 = linalg.matmul ins(%mm500, %w_xod5 : tensor<128x512xf16>, tensor<512x512xf16>)
                          outs(%fill502 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty504 = tensor.empty() : tensor<128x512xf16>
  %add505 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm503, %add483 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty504 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>
  %init506 = tensor.empty() : tensor<128x2048xf16>
  %fill507 = linalg.fill ins(%cst : f16) outs(%init506 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %mm508 = linalg.matmul ins(%add505, %w_ffd5_up : tensor<128x512xf16>, tensor<512x2048xf16>)
                          outs(%fill507 : tensor<128x2048xf16>) -> tensor<128x2048xf16>
  %empty509 = tensor.empty() : tensor<128x2048xf16>
  %relu510 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm508 : tensor<128x2048xf16>)
    outs(%empty509 : tensor<128x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %z = arith.constant 0.0 : f16
    %m = arith.maximumf %in, %z : f16
    linalg.yield %m : f16
  } -> tensor<128x2048xf16>
  %init511 = tensor.empty() : tensor<128x512xf16>
  %fill512 = linalg.fill ins(%cst : f16) outs(%init511 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %mm513 = linalg.matmul ins(%relu510, %w_ffd5_down : tensor<128x2048xf16>, tensor<2048x512xf16>)
                          outs(%fill512 : tensor<128x512xf16>) -> tensor<128x512xf16>
  %empty514 = tensor.empty() : tensor<128x512xf16>
  %add515 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm513, %add505 : tensor<128x512xf16>, tensor<128x512xf16>)
    outs(%empty514 : tensor<128x512xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  } -> tensor<128x512xf16>

  return %add515 : tensor<128x512xf16>
}
