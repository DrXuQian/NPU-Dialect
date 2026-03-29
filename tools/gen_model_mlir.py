#!/usr/bin/env python3
"""Generate complete model MLIR files for NPU compiler testing.

Outputs valid MLIR textual IR to stdout, parseable by npu-opt.
Only uses: linalg.matmul, linalg.conv_2d_nchw_fchw, linalg.generic,
           linalg.fill, tensor.empty, tensor.pad, arith.constant

Usage:
    python tools/gen_model_mlir.py --model resnet18
    python tools/gen_model_mlir.py --model yolo_tiny
    python tools/gen_model_mlir.py --model llama_tiny
    python tools/gen_model_mlir.py --model mobilenetv2
"""

import argparse
import sys


class MLIRBuilder:
    """Generates MLIR SSA values with sequential naming."""

    def __init__(self):
        self._counter = 0
        self._lines = []
        self._params = []  # (name, type_str) for function arguments
        self._indent = "  "

    def _next_var(self, prefix="v"):
        name = f"%{prefix}{self._counter}"
        self._counter += 1
        return name

    def _tensor_4d(self, n, c, h, w):
        return f"tensor<{n}x{c}x{h}x{w}xf16>"

    def _tensor_2d(self, m, n):
        return f"tensor<{m}x{n}xf16>"

    def add_param(self, name, type_str):
        """Register a function parameter."""
        self._params.append((name, type_str))

    def emit(self, line):
        self._lines.append(f"{self._indent}{line}")

    def emit_blank(self):
        self._lines.append("")

    def emit_comment(self, text):
        self._lines.append(f"{self._indent}// {text}")

    # ---- Core operations ----

    def constant_zero(self):
        """Emit %cst = arith.constant 0.0 : f16 (only call once)."""
        self.emit("%cst = arith.constant 0.0 : f16")
        return "%cst"

    def conv2d(self, input_var, input_type, filter_var, filter_type,
               out_n, out_c, out_h, out_w, stride):
        """Emit tensor.empty + linalg.fill + linalg.conv_2d_nchw_fchw."""
        out_type = self._tensor_4d(out_n, out_c, out_h, out_w)
        init_var = self._next_var("init")
        fill_var = self._next_var("fill")
        conv_var = self._next_var("conv")

        self.emit(f"{init_var} = tensor.empty() : {out_type}")
        self.emit(f"{fill_var} = linalg.fill ins(%cst : f16) "
                  f"outs({init_var} : {out_type}) -> {out_type}")
        self.emit(f"{conv_var} = linalg.conv_2d_nchw_fchw {{")
        self.emit(f"  dilations = dense<1> : tensor<2xi64>,")
        self.emit(f"  strides = dense<{stride}> : tensor<2xi64>")
        self.emit(f"}} ins({input_var}, {filter_var} : {input_type}, {filter_type})")
        self.emit(f"  outs({fill_var} : {out_type}) -> {out_type}")
        return conv_var, out_type

    def pad_spatial(self, input_var, input_type, n, c, h, w, pad):
        """Emit tensor.pad on H and W dimensions."""
        new_h = h + 2 * pad
        new_w = w + 2 * pad
        out_type = self._tensor_4d(n, c, new_h, new_w)
        pad_var = self._next_var("pad")

        self.emit(f"{pad_var} = tensor.pad {input_var} "
                  f"low[0, 0, {pad}, {pad}] high[0, 0, {pad}, {pad}] {{")
        self.emit(f"^bb0(%a0: index, %a1: index, %a2: index, %a3: index):")
        self.emit(f"  tensor.yield %cst : f16")
        self.emit(f"}} : {input_type} to {out_type}")
        return pad_var, out_type, new_h, new_w

    def relu_4d(self, input_var, input_type, n, c, h, w):
        """Emit linalg.generic relu for 4D tensor."""
        out_type = self._tensor_4d(n, c, h, w)
        empty_var = self._next_var("empty")
        relu_var = self._next_var("relu")

        self.emit(f"{empty_var} = tensor.empty() : {out_type}")
        self.emit(f"{relu_var} = linalg.generic {{")
        self.emit(f"  indexing_maps = ["
                  f"affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,")
        self.emit(f"                   "
                  f"affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],")
        self.emit(f"  iterator_types = "
                  f"[\"parallel\", \"parallel\", \"parallel\", \"parallel\"]")
        self.emit(f"}} ins({input_var} : {out_type})")
        self.emit(f"  outs({empty_var} : {out_type}) {{")
        self.emit(f"^bb0(%in: f16, %out: f16):")
        self.emit(f"  %z = arith.constant 0.0 : f16")
        self.emit(f"  %m = arith.maximumf %in, %z : f16")
        self.emit(f"  linalg.yield %m : f16")
        self.emit(f"}} -> {out_type}")
        return relu_var, out_type

    def relu_2d(self, input_var, input_type, m, n):
        """Emit linalg.generic relu for 2D tensor."""
        out_type = self._tensor_2d(m, n)
        empty_var = self._next_var("empty")
        relu_var = self._next_var("relu")

        self.emit(f"{empty_var} = tensor.empty() : {out_type}")
        self.emit(f"{relu_var} = linalg.generic {{")
        self.emit(f"  indexing_maps = ["
                  f"affine_map<(d0, d1) -> (d0, d1)>,")
        self.emit(f"                   "
                  f"affine_map<(d0, d1) -> (d0, d1)>],")
        self.emit(f"  iterator_types = "
                  f"[\"parallel\", \"parallel\"]")
        self.emit(f"}} ins({input_var} : {out_type})")
        self.emit(f"  outs({empty_var} : {out_type}) {{")
        self.emit(f"^bb0(%in: f16, %out: f16):")
        self.emit(f"  %z = arith.constant 0.0 : f16")
        self.emit(f"  %m = arith.maximumf %in, %z : f16")
        self.emit(f"  linalg.yield %m : f16")
        self.emit(f"}} -> {out_type}")
        return relu_var, out_type

    def add_4d(self, a_var, a_type, b_var, b_type, n, c, h, w):
        """Emit linalg.generic elementwise add for 4D tensors."""
        out_type = self._tensor_4d(n, c, h, w)
        empty_var = self._next_var("empty")
        add_var = self._next_var("add")

        self.emit(f"{empty_var} = tensor.empty() : {out_type}")
        self.emit(f"{add_var} = linalg.generic {{")
        self.emit(f"  indexing_maps = ["
                  f"affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,")
        self.emit(f"                   "
                  f"affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,")
        self.emit(f"                   "
                  f"affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],")
        self.emit(f"  iterator_types = "
                  f"[\"parallel\", \"parallel\", \"parallel\", \"parallel\"]")
        self.emit(f"}} ins({a_var}, {b_var} : {a_type}, {b_type})")
        self.emit(f"  outs({empty_var} : {out_type}) {{")
        self.emit(f"^bb0(%a: f16, %b: f16, %out: f16):")
        self.emit(f"  %sum = arith.addf %a, %b : f16")
        self.emit(f"  linalg.yield %sum : f16")
        self.emit(f"}} -> {out_type}")
        return add_var, out_type

    def add_2d(self, a_var, a_type, b_var, b_type, m, n):
        """Emit linalg.generic elementwise add for 2D tensors."""
        out_type = self._tensor_2d(m, n)
        empty_var = self._next_var("empty")
        add_var = self._next_var("add")

        self.emit(f"{empty_var} = tensor.empty() : {out_type}")
        self.emit(f"{add_var} = linalg.generic {{")
        self.emit(f"  indexing_maps = ["
                  f"affine_map<(d0, d1) -> (d0, d1)>,")
        self.emit(f"                   "
                  f"affine_map<(d0, d1) -> (d0, d1)>,")
        self.emit(f"                   "
                  f"affine_map<(d0, d1) -> (d0, d1)>],")
        self.emit(f"  iterator_types = "
                  f"[\"parallel\", \"parallel\"]")
        self.emit(f"}} ins({a_var}, {b_var} : {a_type}, {b_type})")
        self.emit(f"  outs({empty_var} : {out_type}) {{")
        self.emit(f"^bb0(%a: f16, %b: f16, %out: f16):")
        self.emit(f"  %sum = arith.addf %a, %b : f16")
        self.emit(f"  linalg.yield %sum : f16")
        self.emit(f"}} -> {out_type}")
        return add_var, out_type

    def matmul(self, a_var, a_type, b_var, b_type, m, k, n):
        """Emit tensor.empty + linalg.fill + linalg.matmul."""
        out_type = self._tensor_2d(m, n)
        init_var = self._next_var("init")
        fill_var = self._next_var("fill")
        mm_var = self._next_var("mm")

        self.emit(f"{init_var} = tensor.empty() : {out_type}")
        self.emit(f"{fill_var} = linalg.fill ins(%cst : f16) "
                  f"outs({init_var} : {out_type}) -> {out_type}")
        self.emit(f"{mm_var} = linalg.matmul "
                  f"ins({a_var}, {b_var} : {a_type}, {b_type})")
        self.emit(f"                        "
                  f"outs({fill_var} : {out_type}) -> {out_type}")
        return mm_var, out_type

    # ---- Higher-level helpers ----

    def conv_block(self, input_var, n, in_c, in_h, in_w,
                   out_c, kernel, stride, pad, filter_var,
                   with_relu=True):
        """Emit pad (if needed) + conv + optional relu.

        Returns (output_var, output_type, out_h, out_w).
        """
        cur_var = input_var
        cur_type = self._tensor_4d(n, in_c, in_h, in_w)
        cur_h, cur_w = in_h, in_w

        # Pad if needed
        if pad > 0:
            cur_var, cur_type, cur_h, cur_w = self.pad_spatial(
                cur_var, cur_type, n, in_c, in_h, in_w, pad)

        # Compute output spatial dims
        out_h = (cur_h - kernel) // stride + 1
        out_w = (cur_w - kernel) // stride + 1

        filter_type = self._tensor_4d(out_c, in_c, kernel, kernel)

        conv_var, conv_type = self.conv2d(
            cur_var, cur_type, filter_var, filter_type,
            n, out_c, out_h, out_w, stride)

        if with_relu:
            relu_var, relu_type = self.relu_4d(conv_var, conv_type,
                                                n, out_c, out_h, out_w)
            return relu_var, relu_type, out_h, out_w
        else:
            return conv_var, conv_type, out_h, out_w

    def build_function(self, func_name, return_var, return_type):
        """Assemble the full function string."""
        # Build parameter list
        param_strs = [f"{name}: {typ}" for name, typ in self._params]
        params = ",\n    ".join(param_strs)

        lines = []
        lines.append(f"func.func @{func_name}(")
        lines.append(f"    {params}) -> {return_type} {{")
        lines.extend(self._lines)
        lines.append(f"  return {return_var} : {return_type}")
        lines.append(f"}}")
        return "\n".join(lines)


def _conv_out_size(input_size, kernel, stride, pad):
    """Compute conv output spatial dimension."""
    padded = input_size + 2 * pad
    return (padded - kernel) // stride + 1


class ParamTracker:
    """Track weight parameters for function signature."""

    def __init__(self):
        self._params = []
        self._counter = 0

    def add_conv_weight(self, out_c, in_c, kernel, name=None):
        if name is None:
            name = f"%w{self._counter}"
            self._counter += 1
        type_str = f"tensor<{out_c}x{in_c}x{kernel}x{kernel}xf16>"
        self._params.append((name, type_str))
        return name, type_str

    def add_matmul_weight(self, k, n, name=None):
        if name is None:
            name = f"%w{self._counter}"
            self._counter += 1
        type_str = f"tensor<{k}x{n}xf16>"
        self._params.append((name, type_str))
        return name, type_str

    @property
    def params(self):
        return self._params


# ===========================================================================
# ResNet-18
# ===========================================================================

def gen_resnet18():
    b = MLIRBuilder()
    pt = ParamTracker()

    # Input
    input_name = "%input"
    input_type = b._tensor_4d(1, 3, 224, 224)
    b.add_param(input_name, input_type)

    # Pre-register all weight parameters
    # conv1: 7x7 3->64
    w_conv1, _ = pt.add_conv_weight(64, 3, 7)

    # We use a stride-2 3x3 conv to go from 112x112 -> 56x56 (simulating maxpool)
    w_pool, _ = pt.add_conv_weight(64, 64, 3, "%w_pool")

    # Layer1: 2 basic blocks, 64->64, 56x56, no downsample
    # Block 0: conv3x3 + conv3x3
    w_l1_b0_c1, _ = pt.add_conv_weight(64, 64, 3, "%w_l1_b0_c1")
    w_l1_b0_c2, _ = pt.add_conv_weight(64, 64, 3, "%w_l1_b0_c2")
    # Block 1
    w_l1_b1_c1, _ = pt.add_conv_weight(64, 64, 3, "%w_l1_b1_c1")
    w_l1_b1_c2, _ = pt.add_conv_weight(64, 64, 3, "%w_l1_b1_c2")

    # Layer2: 2 basic blocks, 64->128, first block stride 2 (56->28)
    # Block 0: stride-2 conv + conv + 1x1 shortcut
    w_l2_b0_c1, _ = pt.add_conv_weight(128, 64, 3, "%w_l2_b0_c1")
    w_l2_b0_c2, _ = pt.add_conv_weight(128, 128, 3, "%w_l2_b0_c2")
    w_l2_b0_sc, _ = pt.add_conv_weight(128, 64, 1, "%w_l2_b0_sc")
    # Block 1
    w_l2_b1_c1, _ = pt.add_conv_weight(128, 128, 3, "%w_l2_b1_c1")
    w_l2_b1_c2, _ = pt.add_conv_weight(128, 128, 3, "%w_l2_b1_c2")

    # Layer3: 2 basic blocks, 128->256, first block stride 2 (28->14)
    w_l3_b0_c1, _ = pt.add_conv_weight(256, 128, 3, "%w_l3_b0_c1")
    w_l3_b0_c2, _ = pt.add_conv_weight(256, 256, 3, "%w_l3_b0_c2")
    w_l3_b0_sc, _ = pt.add_conv_weight(256, 128, 1, "%w_l3_b0_sc")
    w_l3_b1_c1, _ = pt.add_conv_weight(256, 256, 3, "%w_l3_b1_c1")
    w_l3_b1_c2, _ = pt.add_conv_weight(256, 256, 3, "%w_l3_b1_c2")

    # Layer4: 2 basic blocks, 256->512, first block stride 2 (14->7)
    w_l4_b0_c1, _ = pt.add_conv_weight(512, 256, 3, "%w_l4_b0_c1")
    w_l4_b0_c2, _ = pt.add_conv_weight(512, 512, 3, "%w_l4_b0_c2")
    w_l4_b0_sc, _ = pt.add_conv_weight(512, 256, 1, "%w_l4_b0_sc")
    w_l4_b1_c1, _ = pt.add_conv_weight(512, 512, 3, "%w_l4_b1_c1")
    w_l4_b1_c2, _ = pt.add_conv_weight(512, 512, 3, "%w_l4_b1_c2")

    # FC weight: 25088 x 1000
    w_fc, _ = pt.add_matmul_weight(25088, 1000, "%w_fc")

    # Register all weight params
    for name, typ in pt.params:
        b.add_param(name, typ)

    # --- Build the body ---
    b.constant_zero()
    b.emit_blank()

    # conv1: 7x7 stride 2, pad 3. Input 224x224 -> pad to 230x230 -> output 112x112
    b.emit_comment("conv1: 7x7 stride 2, 3->64, 224->112")
    cur_var, cur_type, cur_h, cur_w = b.conv_block(
        input_name, 1, 3, 224, 224, 64, 7, 2, 3, w_conv1)
    cur_c = 64
    # cur_h, cur_w = 112, 112
    b.emit_blank()

    # Stride-2 3x3 conv to simulate maxpool: 112->56
    b.emit_comment("stride-2 3x3 conv (simulating maxpool): 112->56")
    cur_var, cur_type, cur_h, cur_w = b.conv_block(
        cur_var, 1, cur_c, cur_h, cur_w, 64, 3, 2, 1, w_pool)
    # cur_h, cur_w = 56, 56
    b.emit_blank()

    def basic_block(b, input_var, n, in_c, out_c, h, w, stride,
                    w_c1, w_c2, w_sc=None):
        """Generate a ResNet basic block.

        Returns (output_var, output_type, out_h, out_w).
        """
        # First conv: 3x3, possibly stride 2
        b.emit_comment(f"BasicBlock: {in_c}->{out_c}, {h}x{w}, stride {stride}")
        c1_var, c1_type, out_h, out_w = b.conv_block(
            input_var, n, in_c, h, w, out_c, 3, stride, 1, w_c1)

        # Second conv: 3x3 stride 1
        c2_var, c2_type, out_h2, out_w2 = b.conv_block(
            c1_var, n, out_c, out_h, out_w, out_c, 3, 1, 1, w_c2,
            with_relu=False)

        # Shortcut
        if w_sc is not None:
            # 1x1 stride-2 shortcut
            b.emit_comment("1x1 stride-2 shortcut")
            sc_out_h = (h - 1) // stride + 1
            sc_out_w = (w - 1) // stride + 1
            sc_filter_type = b._tensor_4d(out_c, in_c, 1, 1)
            sc_input_type = b._tensor_4d(n, in_c, h, w)
            sc_var, sc_type = b.conv2d(
                input_var, sc_input_type, w_sc, sc_filter_type,
                n, out_c, sc_out_h, sc_out_w, stride)
            residual_var = sc_var
            residual_type = sc_type
        else:
            residual_var = input_var
            residual_type = b._tensor_4d(n, in_c, h, w)

        # Add residual
        add_var, add_type = b.add_4d(
            c2_var, c2_type, residual_var, residual_type,
            n, out_c, out_h2, out_w2)

        # Final relu
        relu_var, relu_type = b.relu_4d(add_var, add_type,
                                         n, out_c, out_h2, out_w2)
        return relu_var, relu_type, out_h2, out_w2

    # Layer 1: 2 blocks, 64->64, 56x56, no downsample
    b.emit_comment("=== Layer 1 ===")
    cur_var, cur_type, cur_h, cur_w = basic_block(
        b, cur_var, 1, 64, 64, cur_h, cur_w, 1,
        w_l1_b0_c1, w_l1_b0_c2)
    cur_c = 64
    b.emit_blank()
    cur_var, cur_type, cur_h, cur_w = basic_block(
        b, cur_var, 1, 64, 64, cur_h, cur_w, 1,
        w_l1_b1_c1, w_l1_b1_c2)
    b.emit_blank()

    # Layer 2: 2 blocks, 64->128, first stride 2 (56->28)
    b.emit_comment("=== Layer 2 ===")
    cur_var, cur_type, cur_h, cur_w = basic_block(
        b, cur_var, 1, 64, 128, cur_h, cur_w, 2,
        w_l2_b0_c1, w_l2_b0_c2, w_l2_b0_sc)
    cur_c = 128
    b.emit_blank()
    cur_var, cur_type, cur_h, cur_w = basic_block(
        b, cur_var, 1, 128, 128, cur_h, cur_w, 1,
        w_l2_b1_c1, w_l2_b1_c2)
    b.emit_blank()

    # Layer 3: 2 blocks, 128->256, first stride 2 (28->14)
    b.emit_comment("=== Layer 3 ===")
    cur_var, cur_type, cur_h, cur_w = basic_block(
        b, cur_var, 1, 128, 256, cur_h, cur_w, 2,
        w_l3_b0_c1, w_l3_b0_c2, w_l3_b0_sc)
    cur_c = 256
    b.emit_blank()
    cur_var, cur_type, cur_h, cur_w = basic_block(
        b, cur_var, 1, 256, 256, cur_h, cur_w, 1,
        w_l3_b1_c1, w_l3_b1_c2)
    b.emit_blank()

    # Layer 4: 2 blocks, 256->512, first stride 2 (14->7)
    b.emit_comment("=== Layer 4 ===")
    cur_var, cur_type, cur_h, cur_w = basic_block(
        b, cur_var, 1, 256, 512, cur_h, cur_w, 2,
        w_l4_b0_c1, w_l4_b0_c2, w_l4_b0_sc)
    cur_c = 512
    b.emit_blank()
    cur_var, cur_type, cur_h, cur_w = basic_block(
        b, cur_var, 1, 512, 512, cur_h, cur_w, 1,
        w_l4_b1_c1, w_l4_b1_c2)
    b.emit_blank()

    # FC: flatten 512*7*7=25088 -> matmul -> 1000
    # We model the "flatten" conceptually: the 4D tensor [1, 512, 7, 7] is
    # reshaped to [1, 25088] via a linalg.generic copy. For simplicity and to
    # stay within allowed ops, we treat the FC as a matmul on a 2D tensor.
    # We use a linalg.generic to "reshape" (copy) the 4D to 2D.
    b.emit_comment("Flatten [1,512,7,7] -> [1,25088] via generic reshape")
    flat_dim = cur_c * cur_h * cur_w  # 512*7*7 = 25088
    flat_type = b._tensor_2d(1, flat_dim)
    flat_empty = b._next_var("flat_empty")
    flat_var = b._next_var("flat")

    # We emit a generic that reads from 4D and writes to 2D.
    # indexing_maps: input (d0, d1) -> unflattened indices, output (d0, d1) -> (d0, d1)
    # Actually, the safest approach is to NOT do a reshape (which needs
    # tensor.collapse_shape) but instead model the FC as a 2D matmul
    # where the input is [1, 25088]. Since we can't use tensor.collapse_shape,
    # and a linalg.generic reshape is complex, let's just do the matmul
    # with the understanding that the "flatten" is implicit.
    # We'll use the 4D output directly with a 2D matmul by first doing a
    # linalg.generic that copies data (identity map won't work for rank change).
    #
    # Simplest valid approach: emit the matmul as [1, 25088] x [25088, 1000]
    # and use the fact that npu-opt doesn't enforce data flow beyond type checking.
    # But we need a valid SSA source for the [1, 25088] tensor.
    # Let's create it as a tensor.empty and fill via linalg.generic from 4D input.
    #
    # Actually, we should just use a single linalg.generic that maps 4D to 2D.
    # But for linalg.generic, all inputs and outputs must have the same rank
    # (same number of dimensions in the indexing maps).
    #
    # The cleanest approach within the allowed ops is to use tensor.empty for
    # the flat tensor and then do a matmul treating it as if it was filled.
    # This is a modeling simplification - the "flatten" is conceptual.
    #
    # Let's just use the 4D result and do a "reshape" by emitting a
    # linalg.generic with 4D input and 4D "flat" output [1, 25088, 1, 1]:
    reshape_type = b._tensor_4d(1, flat_dim, 1, 1)
    reshape_empty = b._next_var("reshape_empty")
    reshape_var = b._next_var("reshape")
    b.emit(f"{reshape_empty} = tensor.empty() : {reshape_type}")
    b.emit(f"{reshape_var} = linalg.generic {{")
    b.emit(f"  indexing_maps = ["
           f"affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,")
    b.emit(f"                   "
           f"affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],")
    b.emit(f"  iterator_types = "
           f"[\"parallel\", \"parallel\", \"parallel\", \"parallel\"]")
    # Wait - this won't work because the shapes are different.
    # Let's take a different approach: skip the reshape entirely.
    # Model the FC layer as a 1x1 conv: [1, 512, 7, 7] conv [1000, 512, 7, 7] -> [1, 1000, 1, 1]
    # This is a valid way to model a fully-connected layer as a convolution!
    b._lines = b._lines[:-6]  # Remove the last 6 lines we just added
    b._counter -= 2  # Undo the counter

    b.emit_comment("FC as conv: [1,512,7,7] conv [1000,512,7,7] -> [1,1000,1,1]")
    fc_filter_type = b._tensor_4d(1000, 512, 7, 7)
    fc_out_type = b._tensor_4d(1, 1000, 1, 1)
    fc_init = b._next_var("fc_init")
    fc_fill = b._next_var("fc_fill")
    fc_conv = b._next_var("fc_conv")
    b.emit(f"{fc_init} = tensor.empty() : {fc_out_type}")
    b.emit(f"{fc_fill} = linalg.fill ins(%cst : f16) "
           f"outs({fc_init} : {fc_out_type}) -> {fc_out_type}")
    b.emit(f"{fc_conv} = linalg.conv_2d_nchw_fchw {{")
    b.emit(f"  dilations = dense<1> : tensor<2xi64>,")
    b.emit(f"  strides = dense<1> : tensor<2xi64>")
    b.emit(f"}} ins({cur_var}, {w_fc} : {cur_type}, {fc_filter_type})")
    b.emit(f"  outs({fc_fill} : {fc_out_type}) -> {fc_out_type}")

    # Hmm wait - w_fc was registered as a 2D matmul weight (25088x1000).
    # We need to change the FC weight to be a 4D conv weight instead.
    # Let me fix the param tracker.
    # Actually, let me just redo the weight registration.
    # The cleanest approach: use the matmul for FC, but we need a 2D input.
    # Let me rethink...

    # OK, simplest approach that works within the op constraints:
    # Model FC as matmul [1, 25088] x [25088, 1000] -> [1, 1000]
    # Create the [1, 25088] tensor via tensor.empty (conceptual flatten).
    # This is acceptable because npu-opt only checks MLIR syntax/types, not
    # semantic correctness of the data flow.

    # Let me restart the FC section cleanly.
    # Remove the conv-based FC lines
    b._lines = b._lines[:-8]  # Remove the FC conv lines
    b._counter -= 3

    # Actually, for the FC, let's just model it cleanly as matmul.
    # We need the input to be 2D. We'll use a linalg.generic identity to
    # "reshape" - but that requires same rank.
    #
    # The absolute simplest: just use tensor.empty to create a [1,25088] tensor
    # and do the matmul. Yes, the data won't flow from the conv output, but
    # npu-opt will still parse it. However, this creates a disconnected graph.
    #
    # Better: model FC as a 7x7 conv with 1000 output channels.
    # Weight shape: [1000, 512, 7, 7]. This IS a valid conv2d.
    # We just need to fix the weight parameter type.

    # I need to go back and fix the weight registration. Let me rebuild
    # the whole function with the corrected FC approach.
    pass

    # The issue is we already registered w_fc as 2D. Let me just rebuild.
    # Rather than hack the builder, let me restructure this properly.
    return _gen_resnet18_v2()


def _gen_resnet18_v2():
    """Clean implementation of ResNet-18."""
    b = MLIRBuilder()

    # Input
    b.add_param("%input", "tensor<1x3x224x224xf16>")

    # All weight parameters
    weights = [
        ("%w_conv1", "tensor<64x3x7x7xf16>"),
        ("%w_pool", "tensor<64x64x3x3xf16>"),
        # Layer1 block0
        ("%w_l1_b0_c1", "tensor<64x64x3x3xf16>"),
        ("%w_l1_b0_c2", "tensor<64x64x3x3xf16>"),
        # Layer1 block1
        ("%w_l1_b1_c1", "tensor<64x64x3x3xf16>"),
        ("%w_l1_b1_c2", "tensor<64x64x3x3xf16>"),
        # Layer2 block0 (64->128, stride 2)
        ("%w_l2_b0_c1", "tensor<128x64x3x3xf16>"),
        ("%w_l2_b0_c2", "tensor<128x128x3x3xf16>"),
        ("%w_l2_b0_sc", "tensor<128x64x1x1xf16>"),
        # Layer2 block1
        ("%w_l2_b1_c1", "tensor<128x128x3x3xf16>"),
        ("%w_l2_b1_c2", "tensor<128x128x3x3xf16>"),
        # Layer3 block0 (128->256, stride 2)
        ("%w_l3_b0_c1", "tensor<256x128x3x3xf16>"),
        ("%w_l3_b0_c2", "tensor<256x256x3x3xf16>"),
        ("%w_l3_b0_sc", "tensor<256x128x1x1xf16>"),
        # Layer3 block1
        ("%w_l3_b1_c1", "tensor<256x256x3x3xf16>"),
        ("%w_l3_b1_c2", "tensor<256x256x3x3xf16>"),
        # Layer4 block0 (256->512, stride 2)
        ("%w_l4_b0_c1", "tensor<512x256x3x3xf16>"),
        ("%w_l4_b0_c2", "tensor<512x512x3x3xf16>"),
        ("%w_l4_b0_sc", "tensor<512x256x1x1xf16>"),
        # Layer4 block1
        ("%w_l4_b1_c1", "tensor<512x512x3x3xf16>"),
        ("%w_l4_b1_c2", "tensor<512x512x3x3xf16>"),
        # FC modeled as 7x7 conv: [1000, 512, 7, 7]
        ("%w_fc", "tensor<1000x512x7x7xf16>"),
    ]
    for name, typ in weights:
        b.add_param(name, typ)

    b.constant_zero()
    b.emit_blank()

    # State tracking
    cur_var = "%input"
    cur_c, cur_h, cur_w = 3, 224, 224

    def emit_conv(in_var, n, in_c, in_h, in_w, out_c, kernel, stride, pad,
                  w_var, relu=True):
        """Emit pad + conv + relu, return (var, type, out_h, out_w)."""
        return b.conv_block(in_var, n, in_c, in_h, in_w,
                            out_c, kernel, stride, pad, w_var,
                            with_relu=relu)

    def basic_block(in_var, in_c, out_c, h, w, stride,
                    w_c1, w_c2, w_sc=None):
        """ResNet basic block."""
        b.emit_comment(f"BasicBlock {in_c}->{out_c} {h}x{w} stride={stride}")

        # conv1: 3x3 with stride
        c1_var, _, c1_h, c1_w = emit_conv(
            in_var, 1, in_c, h, w, out_c, 3, stride, 1, w_c1, relu=True)

        # conv2: 3x3 stride 1, NO relu
        c2_var, c2_type, c2_h, c2_w = emit_conv(
            c1_var, 1, out_c, c1_h, c1_w, out_c, 3, 1, 1, w_c2, relu=False)

        # Shortcut
        if w_sc is not None:
            b.emit_comment("1x1 stride-2 shortcut conv")
            in_type = b._tensor_4d(1, in_c, h, w)
            sc_out_h = (h - 1) // stride + 1
            sc_out_w = (w - 1) // stride + 1
            sc_filter_type = b._tensor_4d(out_c, in_c, 1, 1)
            sc_var, sc_type = b.conv2d(
                in_var, in_type, w_sc, sc_filter_type,
                1, out_c, sc_out_h, sc_out_w, stride)
            res_var, res_type = sc_var, sc_type
        else:
            res_var = in_var
            res_type = b._tensor_4d(1, in_c, h, w)

        # Add + relu
        add_var, add_type = b.add_4d(
            c2_var, c2_type, res_var, res_type,
            1, out_c, c2_h, c2_w)
        relu_var, relu_type = b.relu_4d(
            add_var, add_type, 1, out_c, c2_h, c2_w)

        return relu_var, relu_type, c2_h, c2_w

    # conv1: 7x7 stride 2 pad 3, 3->64, 224->112
    b.emit_comment("conv1: 7x7 stride 2, 3->64, 224->112")
    cur_var, cur_type, cur_h, cur_w = emit_conv(
        cur_var, 1, cur_c, cur_h, cur_w, 64, 7, 2, 3, "%w_conv1")
    cur_c = 64
    b.emit_blank()

    # Simulated maxpool: stride-2 3x3 conv 64->64, 112->56
    b.emit_comment("stride-2 3x3 conv simulating maxpool: 112->56")
    cur_var, cur_type, cur_h, cur_w = emit_conv(
        cur_var, 1, cur_c, cur_h, cur_w, 64, 3, 2, 1, "%w_pool")
    b.emit_blank()

    # Layer 1
    b.emit_comment("=== Layer 1 ===")
    cur_var, cur_type, cur_h, cur_w = basic_block(
        cur_var, 64, 64, cur_h, cur_w, 1,
        "%w_l1_b0_c1", "%w_l1_b0_c2")
    b.emit_blank()
    cur_var, cur_type, cur_h, cur_w = basic_block(
        cur_var, 64, 64, cur_h, cur_w, 1,
        "%w_l1_b1_c1", "%w_l1_b1_c2")
    cur_c = 64
    b.emit_blank()

    # Layer 2
    b.emit_comment("=== Layer 2 ===")
    cur_var, cur_type, cur_h, cur_w = basic_block(
        cur_var, 64, 128, cur_h, cur_w, 2,
        "%w_l2_b0_c1", "%w_l2_b0_c2", "%w_l2_b0_sc")
    b.emit_blank()
    cur_var, cur_type, cur_h, cur_w = basic_block(
        cur_var, 128, 128, cur_h, cur_w, 1,
        "%w_l2_b1_c1", "%w_l2_b1_c2")
    cur_c = 128
    b.emit_blank()

    # Layer 3
    b.emit_comment("=== Layer 3 ===")
    cur_var, cur_type, cur_h, cur_w = basic_block(
        cur_var, 128, 256, cur_h, cur_w, 2,
        "%w_l3_b0_c1", "%w_l3_b0_c2", "%w_l3_b0_sc")
    b.emit_blank()
    cur_var, cur_type, cur_h, cur_w = basic_block(
        cur_var, 256, 256, cur_h, cur_w, 1,
        "%w_l3_b1_c1", "%w_l3_b1_c2")
    cur_c = 256
    b.emit_blank()

    # Layer 4
    b.emit_comment("=== Layer 4 ===")
    cur_var, cur_type, cur_h, cur_w = basic_block(
        cur_var, 256, 512, cur_h, cur_w, 2,
        "%w_l4_b0_c1", "%w_l4_b0_c2", "%w_l4_b0_sc")
    b.emit_blank()
    cur_var, cur_type, cur_h, cur_w = basic_block(
        cur_var, 512, 512, cur_h, cur_w, 1,
        "%w_l4_b1_c1", "%w_l4_b1_c2")
    cur_c = 512
    b.emit_blank()

    # FC as 7x7 conv: [1,512,7,7] -> [1,1000,1,1]
    b.emit_comment("FC modeled as 7x7 conv: [1,512,7,7] -> [1,1000,1,1]")
    fc_out_type = b._tensor_4d(1, 1000, 1, 1)
    fc_in_type = cur_type
    fc_w_type = b._tensor_4d(1000, 512, 7, 7)
    fc_init = b._next_var("fc_init")
    fc_fill = b._next_var("fc_fill")
    fc_var = b._next_var("fc")
    b.emit(f"{fc_init} = tensor.empty() : {fc_out_type}")
    b.emit(f"{fc_fill} = linalg.fill ins(%cst : f16) "
           f"outs({fc_init} : {fc_out_type}) -> {fc_out_type}")
    b.emit(f"{fc_var} = linalg.conv_2d_nchw_fchw {{")
    b.emit(f"  dilations = dense<1> : tensor<2xi64>,")
    b.emit(f"  strides = dense<1> : tensor<2xi64>")
    b.emit(f"}} ins({cur_var}, %w_fc : {fc_in_type}, {fc_w_type})")
    b.emit(f"  outs({fc_fill} : {fc_out_type}) -> {fc_out_type}")

    return b.build_function("resnet18", fc_var, fc_out_type)


# ===========================================================================
# YOLOv3-tiny backbone
# ===========================================================================

def gen_yolo_tiny():
    b = MLIRBuilder()

    b.add_param("%input", "tensor<1x3x416x416xf16>")

    # Weights for 6 conv layers + detection head
    conv_specs = [
        # (in_c, out_c, kernel, stride, pad)
        (3, 16, 3, 1, 1),      # conv0: 416->416
        (16, 32, 3, 2, 1),     # conv1: 416->208
        (32, 64, 3, 2, 1),     # conv2: 208->104
        (64, 128, 3, 2, 1),    # conv3: 104->52
        (128, 256, 3, 2, 1),   # conv4: 52->26
        (256, 512, 3, 2, 1),   # conv5: 26->13
    ]

    weight_names = []
    for i, (in_c, out_c, k, s, p) in enumerate(conv_specs):
        name = f"%w{i}"
        b.add_param(name, f"tensor<{out_c}x{in_c}x{k}x{k}xf16>")
        weight_names.append(name)

    # Detection head: 1x1 conv 512->255
    b.add_param("%w_det", "tensor<255x512x1x1xf16>")

    b.constant_zero()
    b.emit_blank()

    cur_var = "%input"
    cur_c, cur_h, cur_w = 3, 416, 416

    for i, (in_c, out_c, k, s, p) in enumerate(conv_specs):
        b.emit_comment(f"conv{i}: {k}x{k} stride {s}, {in_c}->{out_c}, "
                       f"{cur_h}x{cur_w}")
        cur_var, cur_type, cur_h, cur_w = b.conv_block(
            cur_var, 1, in_c, cur_h, cur_w, out_c, k, s, p,
            weight_names[i])
        cur_c = out_c
        b.emit_blank()

    # Detection head: 1x1 conv stride 1, no pad
    b.emit_comment(f"detection head: 1x1 conv 512->255, {cur_h}x{cur_w}")
    cur_var, cur_type, cur_h, cur_w = b.conv_block(
        cur_var, 1, cur_c, cur_h, cur_w, 255, 1, 1, 0,
        "%w_det", with_relu=False)

    return b.build_function("yolo_tiny", cur_var, cur_type)


# ===========================================================================
# LLaMA-tiny (2-layer transformer)
# ===========================================================================

def gen_llama_tiny():
    b = MLIRBuilder()

    seq, hidden, ffn_dim = 128, 256, 512

    b.add_param("%input", f"tensor<{seq}x{hidden}xf16>")

    # Layer 0 attention weights
    b.add_param("%w_q0", f"tensor<{hidden}x{hidden}xf16>")
    b.add_param("%w_k0", f"tensor<{hidden}x{hidden}xf16>")
    b.add_param("%w_v0", f"tensor<{hidden}x{hidden}xf16>")
    b.add_param("%w_kt0", f"tensor<{hidden}x{seq}xf16>")  # pre-transposed K for attn scores
    b.add_param("%w_o0", f"tensor<{hidden}x{hidden}xf16>")
    # Layer 0 FFN weights
    b.add_param("%w_ff0_up", f"tensor<{hidden}x{ffn_dim}xf16>")
    b.add_param("%w_ff0_down", f"tensor<{ffn_dim}x{hidden}xf16>")
    # Layer 1 attention weights
    b.add_param("%w_q1", f"tensor<{hidden}x{hidden}xf16>")
    b.add_param("%w_k1", f"tensor<{hidden}x{hidden}xf16>")
    b.add_param("%w_v1", f"tensor<{hidden}x{hidden}xf16>")
    b.add_param("%w_kt1", f"tensor<{hidden}x{seq}xf16>")
    b.add_param("%w_o1", f"tensor<{hidden}x{hidden}xf16>")
    # Layer 1 FFN weights
    b.add_param("%w_ff1_up", f"tensor<{hidden}x{ffn_dim}xf16>")
    b.add_param("%w_ff1_down", f"tensor<{ffn_dim}x{hidden}xf16>")

    b.constant_zero()
    b.emit_blank()

    cur_var = "%input"
    cur_type = f"tensor<{seq}x{hidden}xf16>"

    for layer in range(2):
        sfx = str(layer)
        b.emit_comment(f"=== Transformer Layer {layer} ===")
        b.emit_blank()

        residual_var = cur_var
        residual_type = cur_type

        # Q projection: [128, 256] x [256, 256] -> [128, 256]
        b.emit_comment(f"Q projection")
        q_var, q_type = b.matmul(
            cur_var, cur_type,
            f"%w_q{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq, hidden, hidden)

        # K projection
        b.emit_comment(f"K projection")
        k_var, k_type = b.matmul(
            cur_var, cur_type,
            f"%w_k{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq, hidden, hidden)

        # V projection
        b.emit_comment(f"V projection")
        v_var, v_type = b.matmul(
            cur_var, cur_type,
            f"%w_v{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq, hidden, hidden)
        b.emit_blank()

        # Attention scores: Q x K^T = [128,256] x [256,128] -> [128,128]
        # We use a pre-transposed weight for simplicity
        b.emit_comment("Attention scores: Q x K_transposed -> [128,128]")
        scores_var, scores_type = b.matmul(
            q_var, q_type,
            f"%w_kt{sfx}", f"tensor<{hidden}x{seq}xf16>",
            seq, hidden, seq)

        # Softmax approximation (relu)
        b.emit_comment("Softmax approximation (relu)")
        soft_var, soft_type = b.relu_2d(
            scores_var, scores_type, seq, seq)
        b.emit_blank()

        # Attention output: scores x V = [128,128] x [128,256] -> [128,256]
        b.emit_comment("Attention output: softmax_scores x V")
        attn_var, attn_type = b.matmul(
            soft_var, soft_type,
            v_var, v_type,
            seq, seq, hidden)

        # Output projection: [128,256] x [256,256] -> [128,256]
        b.emit_comment("Output projection")
        out_proj_var, out_proj_type = b.matmul(
            attn_var, attn_type,
            f"%w_o{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq, hidden, hidden)

        # Residual add
        b.emit_comment("Attention residual add")
        attn_res_var, attn_res_type = b.add_2d(
            out_proj_var, out_proj_type,
            residual_var, residual_type,
            seq, hidden)
        b.emit_blank()

        # FFN
        b.emit_comment("FFN: up projection [128,256] x [256,512] -> [128,512]")
        ffn_residual_var = attn_res_var
        ffn_residual_type = attn_res_type

        ff_up_var, ff_up_type = b.matmul(
            attn_res_var, attn_res_type,
            f"%w_ff{sfx}_up", f"tensor<{hidden}x{ffn_dim}xf16>",
            seq, hidden, ffn_dim)

        # ReLU
        b.emit_comment("FFN ReLU")
        ff_relu_var, ff_relu_type = b.relu_2d(
            ff_up_var, ff_up_type, seq, ffn_dim)

        # Down projection: [128,512] x [512,256] -> [128,256]
        b.emit_comment("FFN: down projection [128,512] x [512,256] -> [128,256]")
        ff_down_var, ff_down_type = b.matmul(
            ff_relu_var, ff_relu_type,
            f"%w_ff{sfx}_down", f"tensor<{ffn_dim}x{hidden}xf16>",
            seq, ffn_dim, hidden)

        # FFN residual add
        b.emit_comment("FFN residual add")
        ffn_res_var, ffn_res_type = b.add_2d(
            ff_down_var, ff_down_type,
            ffn_residual_var, ffn_residual_type,
            seq, hidden)

        cur_var = ffn_res_var
        cur_type = ffn_res_type
        b.emit_blank()

    return b.build_function("llama_tiny", cur_var, cur_type)


# ===========================================================================
# MobileNetV2 backbone
# ===========================================================================

def gen_mobilenetv2():
    b = MLIRBuilder()

    b.add_param("%input", "tensor<1x3x224x224xf16>")

    # Build weight list
    # Initial conv: 3x3 stride 2, 3->32
    weight_params = [
        ("%w_conv0", "tensor<32x3x3x3xf16>"),
    ]

    # Inverted residual blocks (simplified):
    # Each block: 1x1 expand (if expand_ratio > 1), 3x3 conv, 1x1 project
    # Block specs: (in_c, out_c, stride, expand_ratio, num_blocks)
    block_specs = [
        (32, 16, 1, 1, 1),    # no expand
        (16, 24, 2, 6, 2),    # first stride 2, second stride 1
        (24, 32, 2, 6, 3),    # first stride 2, rest stride 1
    ]

    block_idx = 0
    for in_c, out_c, first_stride, expand_ratio, count in block_specs:
        for i in range(count):
            stride = first_stride if i == 0 else 1
            cur_in_c = in_c if i == 0 else out_c
            mid_c = cur_in_c * expand_ratio

            if expand_ratio > 1:
                weight_params.append(
                    (f"%w_blk{block_idx}_exp",
                     f"tensor<{mid_c}x{cur_in_c}x1x1xf16>"))

            weight_params.append(
                (f"%w_blk{block_idx}_dw",
                 f"tensor<{mid_c}x{mid_c}x3x3xf16>"))
            weight_params.append(
                (f"%w_blk{block_idx}_proj",
                 f"tensor<{out_c}x{mid_c}x1x1xf16>"))
            block_idx += 1

    # Final 1x1 conv 32->1280
    weight_params.append(("%w_final", "tensor<1280x32x1x1xf16>"))
    # FC modeled as 1x1 conv on [1, 1280, Hf, Wf] -> ... actually
    # For FC we need to know final spatial dims. Let's compute:
    # 224 -> conv0 stride 2 -> 112
    # block (32->16, stride 1) -> 112
    # block (16->24, stride 2) -> 56
    # block (24->24, stride 1) -> 56
    # block (24->32, stride 2) -> 28
    # block (32->32, stride 1) -> 28
    # block (32->32, stride 1) -> 28
    # Then final 1x1 -> 28x28
    # FC: global average pool (conceptual) then [1,1280] -> [1,1000]
    # Model FC as conv [1,1280,28,28] with filter [1000,1280,28,28] -> ridiculous.
    # Better: model FC as a matmul on conceptually flattened tensor.
    # Actually, let's model GAP as a 28x28 conv: [1,1280,28,28] conv [1000,1280,28,28]
    # That's a huge weight. Instead, let's just do a 1x1 conv after the 1x1 final:
    # [1,1280,28,28] -> 1x1 conv -> [1,1000,28,28]
    # Then we can say the output is [1,1000,28,28] (simplified, no GAP).
    # OR we can use matmul: we need a 2D path.
    #
    # Simplest: model FC as 1x1 conv 1280->1000 (keeping spatial dims).
    # The output would be [1, 1000, 28, 28] which is fine for testing.
    weight_params.append(("%w_fc", "tensor<1000x1280x1x1xf16>"))

    for name, typ in weight_params:
        b.add_param(name, typ)

    b.constant_zero()
    b.emit_blank()

    # Initial conv: 3x3 stride 2 pad 1, 3->32, 224->112
    b.emit_comment("Initial conv: 3x3 stride 2, 3->32, 224->112")
    cur_var, cur_type, cur_h, cur_w = b.conv_block(
        "%input", 1, 3, 224, 224, 32, 3, 2, 1, "%w_conv0")
    cur_c = 32
    b.emit_blank()

    block_idx = 0
    for in_c, out_c, first_stride, expand_ratio, count in block_specs:
        for i in range(count):
            stride = first_stride if i == 0 else 1
            cur_in_c = in_c if i == 0 else out_c
            mid_c = cur_in_c * expand_ratio
            has_residual = (stride == 1 and cur_in_c == out_c)

            b.emit_comment(f"Inverted residual block {block_idx}: "
                           f"{cur_in_c}->{out_c}, mid={mid_c}, "
                           f"stride={stride}, {cur_h}x{cur_w}")

            block_input_var = cur_var
            block_input_type = cur_type

            # 1x1 expand (if expand_ratio > 1)
            if expand_ratio > 1:
                b.emit_comment("1x1 expand conv")
                cur_var, cur_type, cur_h, cur_w = b.conv_block(
                    cur_var, 1, cur_in_c, cur_h, cur_w, mid_c, 1, 1, 0,
                    f"%w_blk{block_idx}_exp")

            # 3x3 conv (with stride if first block of group)
            b.emit_comment("3x3 conv")
            cur_var, cur_type, cur_h, cur_w = b.conv_block(
                cur_var, 1, mid_c, cur_h, cur_w, mid_c, 3, stride, 1,
                f"%w_blk{block_idx}_dw")

            # 1x1 project (no relu)
            b.emit_comment("1x1 project conv (no relu)")
            cur_var, cur_type, cur_h, cur_w = b.conv_block(
                cur_var, 1, mid_c, cur_h, cur_w, out_c, 1, 1, 0,
                f"%w_blk{block_idx}_proj", with_relu=False)

            # Residual add (only if same shape)
            if has_residual:
                b.emit_comment("Residual add")
                cur_var, cur_type = b.add_4d(
                    cur_var, cur_type,
                    block_input_var, block_input_type,
                    1, out_c, cur_h, cur_w)

            cur_c = out_c
            block_idx += 1
            b.emit_blank()

    # Final 1x1 conv: 32->1280
    b.emit_comment(f"Final 1x1 conv: {cur_c}->1280, {cur_h}x{cur_w}")
    cur_var, cur_type, cur_h, cur_w = b.conv_block(
        cur_var, 1, cur_c, cur_h, cur_w, 1280, 1, 1, 0,
        "%w_final")
    cur_c = 1280
    b.emit_blank()

    # FC as 1x1 conv: 1280->1000 (simplified, no global avg pool)
    b.emit_comment(f"FC as 1x1 conv: 1280->1000, {cur_h}x{cur_w}")
    cur_var, cur_type, cur_h, cur_w = b.conv_block(
        cur_var, 1, cur_c, cur_h, cur_w, 1000, 1, 1, 0,
        "%w_fc", with_relu=False)

    return b.build_function("mobilenetv2", cur_var, cur_type)


# ===========================================================================
# Main
# ===========================================================================

MODELS = {
    "resnet18": gen_resnet18,
    "yolo_tiny": gen_yolo_tiny,
    "llama_tiny": gen_llama_tiny,
    "mobilenetv2": gen_mobilenetv2,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate model MLIR files for NPU compiler testing")
    parser.add_argument("--model", required=True,
                        choices=sorted(MODELS.keys()),
                        help="Model to generate")
    args = parser.parse_args()

    mlir_text = MODELS[args.model]()
    print(mlir_text)


if __name__ == "__main__":
    main()
