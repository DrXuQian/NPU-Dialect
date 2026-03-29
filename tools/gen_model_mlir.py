#!/usr/bin/env python3
"""Generate complete model MLIR files for NPU compiler testing.

Outputs valid MLIR textual IR to stdout, parseable by npu-opt.
Only uses: linalg.matmul, linalg.conv_2d_nchw_fchw, linalg.generic,
           linalg.fill, tensor.empty, tensor.pad, arith.constant

Usage:
    python tools/gen_model_mlir.py --model resnet18
    python tools/gen_model_mlir.py --list
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


# ===========================================================================
# Reusable building blocks
# ===========================================================================

def _make_resnet_basic_block(b, in_var, in_c, out_c, h, w, stride,
                             w_c1, w_c2, w_sc=None):
    """ResNet basic block (two 3x3 convs + residual)."""
    b.emit_comment(f"BasicBlock {in_c}->{out_c} {h}x{w} stride={stride}")

    c1_var, _, c1_h, c1_w = b.conv_block(
        in_var, 1, in_c, h, w, out_c, 3, stride, 1, w_c1, with_relu=True)

    c2_var, c2_type, c2_h, c2_w = b.conv_block(
        c1_var, 1, out_c, c1_h, c1_w, out_c, 3, 1, 1, w_c2, with_relu=False)

    if w_sc is not None:
        b.emit_comment("1x1 shortcut conv")
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

    add_var, add_type = b.add_4d(
        c2_var, c2_type, res_var, res_type, 1, out_c, c2_h, c2_w)
    relu_var, relu_type = b.relu_4d(add_var, add_type, 1, out_c, c2_h, c2_w)
    return relu_var, relu_type, c2_h, c2_w


def _make_resnet_bottleneck_block(b, in_var, in_c, out_c, h, w, stride,
                                  w_c1, w_c2, w_c3, w_sc=None,
                                  override_mid_c=None):
    """ResNet bottleneck block (1x1 -> 3x3 -> 1x1 + residual).
    out_c is the final (expanded) channel count. mid_c = out_c // 4 by default.
    """
    mid_c = override_mid_c if override_mid_c is not None else out_c // 4
    b.emit_comment(f"Bottleneck {in_c}->{out_c} mid={mid_c} {h}x{w} stride={stride}")

    # 1x1 reduce
    c1_var, _, c1_h, c1_w = b.conv_block(
        in_var, 1, in_c, h, w, mid_c, 1, 1, 0, w_c1, with_relu=True)

    # 3x3 with stride
    c2_var, _, c2_h, c2_w = b.conv_block(
        c1_var, 1, mid_c, c1_h, c1_w, mid_c, 3, stride, 1, w_c2, with_relu=True)

    # 1x1 expand
    c3_var, c3_type, c3_h, c3_w = b.conv_block(
        c2_var, 1, mid_c, c2_h, c2_w, out_c, 1, 1, 0, w_c3, with_relu=False)

    if w_sc is not None:
        b.emit_comment("1x1 shortcut conv")
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

    add_var, add_type = b.add_4d(
        c3_var, c3_type, res_var, res_type, 1, out_c, c3_h, c3_w)
    relu_var, relu_type = b.relu_4d(add_var, add_type, 1, out_c, c3_h, c3_w)
    return relu_var, relu_type, c3_h, c3_w


# ===========================================================================
# ResNet family
# ===========================================================================

def build_resnet_variant(layers, block_type='basic', num_classes=1000):
    """Build ResNet variant.
    layers: list of block counts per stage, e.g. [2,2,2,2] for ResNet-18.
    block_type: 'basic' or 'bottleneck'.
    """
    b = MLIRBuilder()
    b.add_param("%input", "tensor<1x3x224x224xf16>")

    stage_channels = [64, 128, 256, 512]
    if block_type == 'bottleneck':
        # Bottleneck expands by 4x
        stage_out_channels = [ch * 4 for ch in stage_channels]
    else:
        stage_out_channels = list(stage_channels)

    # Collect all weights
    weights = [
        ("%w_conv1", "tensor<64x3x7x7xf16>"),
        ("%w_pool", "tensor<64x64x3x3xf16>"),
    ]

    for stage_idx in range(4):
        out_c = stage_out_channels[stage_idx]
        in_c_first = stage_out_channels[stage_idx - 1] if stage_idx > 0 else 64

        for blk_idx in range(layers[stage_idx]):
            cur_in_c = in_c_first if blk_idx == 0 else out_c
            stride = 2 if (stage_idx > 0 and blk_idx == 0) else 1
            prefix = f"%w_s{stage_idx}_b{blk_idx}"

            if block_type == 'basic':
                weights.append((f"{prefix}_c1", f"tensor<{out_c}x{cur_in_c}x3x3xf16>"))
                weights.append((f"{prefix}_c2", f"tensor<{out_c}x{out_c}x3x3xf16>"))
            else:
                mid_c = out_c // 4
                weights.append((f"{prefix}_c1", f"tensor<{mid_c}x{cur_in_c}x1x1xf16>"))
                weights.append((f"{prefix}_c2", f"tensor<{mid_c}x{mid_c}x3x3xf16>"))
                weights.append((f"{prefix}_c3", f"tensor<{out_c}x{mid_c}x1x1xf16>"))

            if cur_in_c != out_c or stride > 1:
                weights.append((f"{prefix}_sc", f"tensor<{out_c}x{cur_in_c}x1x1xf16>"))

    final_c = stage_out_channels[3]
    # FC as 7x7 conv
    weights.append(("%w_fc", f"tensor<{num_classes}x{final_c}x7x7xf16>"))

    for name, typ in weights:
        b.add_param(name, typ)

    b.constant_zero()
    b.emit_blank()

    # conv1: 7x7 stride 2 pad 3
    b.emit_comment("conv1: 7x7 stride 2, 3->64, 224->112")
    cur_var, cur_type, cur_h, cur_w = b.conv_block(
        "%input", 1, 3, 224, 224, 64, 7, 2, 3, "%w_conv1")
    cur_c = 64
    b.emit_blank()

    # Simulated maxpool: stride-2 3x3 conv
    b.emit_comment("stride-2 3x3 conv simulating maxpool: 112->56")
    cur_var, cur_type, cur_h, cur_w = b.conv_block(
        cur_var, 1, 64, cur_h, cur_w, 64, 3, 2, 1, "%w_pool")
    b.emit_blank()

    for stage_idx in range(4):
        out_c = stage_out_channels[stage_idx]
        in_c_first = stage_out_channels[stage_idx - 1] if stage_idx > 0 else 64

        b.emit_comment(f"=== Stage {stage_idx} ===")
        for blk_idx in range(layers[stage_idx]):
            cur_in_c = in_c_first if blk_idx == 0 else out_c
            stride = 2 if (stage_idx > 0 and blk_idx == 0) else 1
            prefix = f"%w_s{stage_idx}_b{blk_idx}"
            needs_sc = (cur_in_c != out_c or stride > 1)
            w_sc = f"{prefix}_sc" if needs_sc else None

            if block_type == 'basic':
                cur_var, cur_type, cur_h, cur_w = _make_resnet_basic_block(
                    b, cur_var, cur_in_c, out_c, cur_h, cur_w, stride,
                    f"{prefix}_c1", f"{prefix}_c2", w_sc)
            else:
                cur_var, cur_type, cur_h, cur_w = _make_resnet_bottleneck_block(
                    b, cur_var, cur_in_c, out_c, cur_h, cur_w, stride,
                    f"{prefix}_c1", f"{prefix}_c2", f"{prefix}_c3", w_sc)

            cur_c = out_c
            b.emit_blank()

    # FC as 7x7 conv
    b.emit_comment(f"FC as {cur_h}x{cur_w} conv: {cur_c}->{num_classes}")
    fc_out_type = b._tensor_4d(1, num_classes, 1, 1)
    fc_w_type = b._tensor_4d(num_classes, cur_c, cur_h, cur_w)
    fc_init = b._next_var("fc_init")
    fc_fill = b._next_var("fc_fill")
    fc_var = b._next_var("fc")
    b.emit(f"{fc_init} = tensor.empty() : {fc_out_type}")
    b.emit(f"{fc_fill} = linalg.fill ins(%cst : f16) "
           f"outs({fc_init} : {fc_out_type}) -> {fc_out_type}")
    b.emit(f"{fc_var} = linalg.conv_2d_nchw_fchw {{")
    b.emit(f"  dilations = dense<1> : tensor<2xi64>,")
    b.emit(f"  strides = dense<1> : tensor<2xi64>")
    b.emit(f"}} ins({cur_var}, %w_fc : {cur_type}, {fc_w_type})")
    b.emit(f"  outs({fc_fill} : {fc_out_type}) -> {fc_out_type}")

    name = f"resnet_{block_type}_{'_'.join(str(l) for l in layers)}"
    return b.build_function(name, fc_var, fc_out_type)


# ===========================================================================
# VGG family
# ===========================================================================

def build_vgg_variant(cfg, name, num_classes=1000):
    """Build VGG variant.
    cfg: list of (num_convs, channels) per block. Pools between blocks.
    """
    b = MLIRBuilder()
    b.add_param("%input", "tensor<1x3x224x224xf16>")

    weights = []
    cur_c = 3
    for block_idx, (num_convs, channels) in enumerate(cfg):
        for conv_idx in range(num_convs):
            ic = cur_c if conv_idx == 0 else channels
            wname = f"%w_b{block_idx}_c{conv_idx}"
            weights.append((wname, f"tensor<{channels}x{ic}x3x3xf16>"))
            cur_c = channels
        # Pool simulated as stride-2 3x3 conv (same channels)
        wname = f"%w_b{block_idx}_pool"
        weights.append((wname, f"tensor<{channels}x{channels}x3x3xf16>"))

    # FC layers: use 1x1 convs on final spatial
    # After 5 pools from 224: 224->112->56->28->14->7
    final_c = cfg[-1][1]
    weights.append(("%w_fc1", f"tensor<4096x{final_c}x7x7xf16>"))
    weights.append(("%w_fc2", f"tensor<4096x4096x1x1xf16>"))
    weights.append(("%w_fc3", f"tensor<{num_classes}x4096x1x1xf16>"))

    for wn, wt in weights:
        b.add_param(wn, wt)

    b.constant_zero()
    b.emit_blank()

    cur_var = "%input"
    cur_c, cur_h, cur_w = 3, 224, 224

    for block_idx, (num_convs, channels) in enumerate(cfg):
        b.emit_comment(f"VGG block {block_idx}: {num_convs} convs, {channels} channels")
        for conv_idx in range(num_convs):
            ic = cur_c if conv_idx == 0 else channels
            cur_var, cur_type, cur_h, cur_w = b.conv_block(
                cur_var, 1, ic, cur_h, cur_w, channels, 3, 1, 1,
                f"%w_b{block_idx}_c{conv_idx}")
            cur_c = channels

        # Pool: stride-2 3x3 conv
        b.emit_comment(f"Pool: {cur_h}->{cur_h // 2}")
        cur_var, cur_type, cur_h, cur_w = b.conv_block(
            cur_var, 1, cur_c, cur_h, cur_w, cur_c, 3, 2, 1,
            f"%w_b{block_idx}_pool")
        b.emit_blank()

    # FC1 as 7x7 conv
    b.emit_comment(f"FC1 as {cur_h}x{cur_w} conv -> 4096")
    fc1_out_type = b._tensor_4d(1, 4096, 1, 1)
    fc1_w_type = b._tensor_4d(4096, cur_c, cur_h, cur_w)
    fc1_init = b._next_var("init")
    fc1_fill = b._next_var("fill")
    fc1_var = b._next_var("fc1")
    b.emit(f"{fc1_init} = tensor.empty() : {fc1_out_type}")
    b.emit(f"{fc1_fill} = linalg.fill ins(%cst : f16) "
           f"outs({fc1_init} : {fc1_out_type}) -> {fc1_out_type}")
    b.emit(f"{fc1_var} = linalg.conv_2d_nchw_fchw {{")
    b.emit(f"  dilations = dense<1> : tensor<2xi64>,")
    b.emit(f"  strides = dense<1> : tensor<2xi64>")
    b.emit(f"}} ins({cur_var}, %w_fc1 : {cur_type}, {fc1_w_type})")
    b.emit(f"  outs({fc1_fill} : {fc1_out_type}) -> {fc1_out_type}")
    fc1_relu, _ = b.relu_4d(fc1_var, fc1_out_type, 1, 4096, 1, 1)
    b.emit_blank()

    # FC2 as 1x1 conv
    b.emit_comment("FC2 as 1x1 conv -> 4096")
    fc2_var, fc2_type, _, _ = b.conv_block(
        fc1_relu, 1, 4096, 1, 1, 4096, 1, 1, 0, "%w_fc2")
    b.emit_blank()

    # FC3 as 1x1 conv
    b.emit_comment(f"FC3 as 1x1 conv -> {num_classes}")
    fc3_var, fc3_type, _, _ = b.conv_block(
        fc2_var, 1, 4096, 1, 1, num_classes, 1, 1, 0, "%w_fc3", with_relu=False)

    return b.build_function(name, fc3_var, fc3_type)


# ===========================================================================
# DenseNet family (simplified: conv blocks without dense connections for
# shape correctness, uses channel concatenation modeled as wider convs)
# ===========================================================================

def build_densenet_variant(block_layers, growth_rate, name, num_classes=1000):
    """Simplified DenseNet: sequential conv blocks with growing channels.
    block_layers: e.g. [6, 12, 24, 16] for DenseNet-121.
    growth_rate: channel growth per layer (32 typically).

    We model each dense layer as bottleneck(1x1) + 3x3 conv where the
    data flows sequentially (no concat). Channels: growth_rate throughout.
    """
    b = MLIRBuilder()
    b.add_param("%input", "tensor<1x3x224x224xf16>")

    bn_out = 4 * growth_rate  # bottleneck intermediate channels
    oc = growth_rate

    weights = [
        ("%w_stem", "tensor<64x3x7x7xf16>"),
        ("%w_pool", f"tensor<{oc}x64x3x3xf16>"),  # pool also transitions 64->growth_rate
    ]

    cur_c = oc
    for bidx, num_layers in enumerate(block_layers):
        for lidx in range(num_layers):
            weights.append((f"%w_d{bidx}_l{lidx}_bn",
                            f"tensor<{bn_out}x{cur_c}x1x1xf16>"))
            weights.append((f"%w_d{bidx}_l{lidx}_conv",
                            f"tensor<{oc}x{bn_out}x3x3xf16>"))
            cur_c = oc  # output is always growth_rate

        if bidx < len(block_layers) - 1:
            trans_out = cur_c  # keep same channels
            weights.append((f"%w_t{bidx}_conv", f"tensor<{trans_out}x{cur_c}x1x1xf16>"))
            weights.append((f"%w_t{bidx}_pool", f"tensor<{trans_out}x{trans_out}x3x3xf16>"))
            cur_c = trans_out

    weights.append(("%w_fc", f"tensor<{num_classes}x{cur_c}x1x1xf16>"))

    for wn, wt in weights:
        b.add_param(wn, wt)

    b.constant_zero()
    b.emit_blank()

    b.emit_comment("Stem: 7x7 conv stride 2, 3->64")
    cur_var, cur_type, cur_h, cur_w = b.conv_block(
        "%input", 1, 3, 224, 224, 64, 7, 2, 3, "%w_stem")
    b.emit_blank()

    b.emit_comment(f"Pool: stride-2 3x3 conv, 64->{oc}")
    cur_var, cur_type, cur_h, cur_w = b.conv_block(
        cur_var, 1, 64, cur_h, cur_w, oc, 3, 2, 1, "%w_pool")
    cur_c = oc
    b.emit_blank()

    for bidx, num_layers in enumerate(block_layers):
        b.emit_comment(f"=== Dense Block {bidx}: {num_layers} layers ===")
        for lidx in range(num_layers):
            cur_var, cur_type, cur_h, cur_w = b.conv_block(
                cur_var, 1, cur_c, cur_h, cur_w, bn_out, 1, 1, 0,
                f"%w_d{bidx}_l{lidx}_bn")
            cur_var, cur_type, cur_h, cur_w = b.conv_block(
                cur_var, 1, bn_out, cur_h, cur_w, oc, 3, 1, 1,
                f"%w_d{bidx}_l{lidx}_conv")
            cur_c = oc
        b.emit_blank()

        if bidx < len(block_layers) - 1:
            b.emit_comment(f"Transition {bidx}: 1x1 conv + stride-2 pool")
            cur_var, cur_type, cur_h, cur_w = b.conv_block(
                cur_var, 1, cur_c, cur_h, cur_w, cur_c, 1, 1, 0,
                f"%w_t{bidx}_conv")
            cur_var, cur_type, cur_h, cur_w = b.conv_block(
                cur_var, 1, cur_c, cur_h, cur_w, cur_c, 3, 2, 1,
                f"%w_t{bidx}_pool")
            b.emit_blank()

    b.emit_comment(f"FC: 1x1 conv {cur_c}->{num_classes}")
    cur_var, cur_type, _, _ = b.conv_block(
        cur_var, 1, cur_c, cur_h, cur_w, num_classes, 1, 1, 0,
        "%w_fc", with_relu=False)

    return b.build_function(name, cur_var, cur_type)


# ===========================================================================
# Generic sequential CNN builder (for VGG-like, EfficientNet-like, MobileNet-like)
# ===========================================================================

def build_sequential_cnn(conv_specs, name, input_c=3, input_h=224, input_w=224,
                         fc_classes=1000, fc_as_1x1=True):
    """Build a generic sequential CNN from a list of conv specs.
    conv_specs: list of (out_c, kernel, stride, pad, relu) tuples.
    """
    b = MLIRBuilder()
    b.add_param("%input", f"tensor<1x{input_c}x{input_h}x{input_w}xf16>")

    # Register weights
    weights = []
    cur_c = input_c
    for i, (oc, k, s, p, _) in enumerate(conv_specs):
        weights.append((f"%w{i}", f"tensor<{oc}x{cur_c}x{k}x{k}xf16>"))
        cur_c = oc

    # Compute final spatial dims
    h, w = input_h, input_w
    for oc, k, s, p, _ in conv_specs:
        h = _conv_out_size(h, k, s, p)
        w = _conv_out_size(w, k, s, p)

    if fc_as_1x1:
        weights.append(("%w_fc", f"tensor<{fc_classes}x{cur_c}x1x1xf16>"))
    else:
        weights.append(("%w_fc", f"tensor<{fc_classes}x{cur_c}x{h}x{w}xf16>"))

    for wn, wt in weights:
        b.add_param(wn, wt)

    b.constant_zero()
    b.emit_blank()

    cur_var = "%input"
    cur_c = input_c
    cur_h, cur_w = input_h, input_w

    for i, (oc, k, s, p, relu) in enumerate(conv_specs):
        b.emit_comment(f"conv{i}: {k}x{k} s{s} p{p} {cur_c}->{oc} {cur_h}x{cur_w}")
        cur_var, cur_type, cur_h, cur_w = b.conv_block(
            cur_var, 1, cur_c, cur_h, cur_w, oc, k, s, p,
            f"%w{i}", with_relu=relu)
        cur_c = oc
        b.emit_blank()

    if fc_as_1x1:
        b.emit_comment(f"FC as 1x1 conv: {cur_c}->{fc_classes}")
        cur_var, cur_type, _, _ = b.conv_block(
            cur_var, 1, cur_c, cur_h, cur_w, fc_classes, 1, 1, 0,
            "%w_fc", with_relu=False)
    else:
        b.emit_comment(f"FC as {cur_h}x{cur_w} conv: {cur_c}->{fc_classes}")
        fc_out_type = b._tensor_4d(1, fc_classes, 1, 1)
        fc_w_type = b._tensor_4d(fc_classes, cur_c, cur_h, cur_w)
        fc_init = b._next_var("init")
        fc_fill = b._next_var("fill")
        fc_var = b._next_var("fc")
        b.emit(f"{fc_init} = tensor.empty() : {fc_out_type}")
        b.emit(f"{fc_fill} = linalg.fill ins(%cst : f16) "
               f"outs({fc_init} : {fc_out_type}) -> {fc_out_type}")
        b.emit(f"{fc_var} = linalg.conv_2d_nchw_fchw {{")
        b.emit(f"  dilations = dense<1> : tensor<2xi64>,")
        b.emit(f"  strides = dense<1> : tensor<2xi64>")
        b.emit(f"}} ins({cur_var}, %w_fc : {cur_type}, {fc_w_type})")
        b.emit(f"  outs({fc_fill} : {fc_out_type}) -> {fc_out_type}")
        cur_var = fc_var
        cur_type = fc_out_type

    return b.build_function(name, cur_var, cur_type)


# ===========================================================================
# MobileNet family (inverted residuals)
# ===========================================================================

def build_mobilenet_variant(block_specs, name, stem_c=32, input_h=224,
                            final_c=1280, num_classes=1000):
    """Build MobileNet-style network with inverted residual blocks.
    block_specs: list of (in_c, out_c, stride, expand_ratio, count).
    """
    b = MLIRBuilder()
    b.add_param("%input", f"tensor<1x3x{input_h}x{input_h}xf16>")

    weights = [("%w_stem", f"tensor<{stem_c}x3x3x3xf16>")]

    block_idx = 0
    for in_c, out_c, first_stride, expand_ratio, count in block_specs:
        for i in range(count):
            stride = first_stride if i == 0 else 1
            cur_in_c = in_c if i == 0 else out_c
            mid_c = cur_in_c * expand_ratio

            if expand_ratio > 1:
                weights.append((f"%w_blk{block_idx}_exp",
                                f"tensor<{mid_c}x{cur_in_c}x1x1xf16>"))
            weights.append((f"%w_blk{block_idx}_dw",
                            f"tensor<{mid_c}x{mid_c}x3x3xf16>"))
            weights.append((f"%w_blk{block_idx}_proj",
                            f"tensor<{out_c}x{mid_c}x1x1xf16>"))
            block_idx += 1

    weights.append(("%w_final", f"tensor<{final_c}x{block_specs[-1][1]}x1x1xf16>"))
    weights.append(("%w_fc", f"tensor<{num_classes}x{final_c}x1x1xf16>"))

    for wn, wt in weights:
        b.add_param(wn, wt)

    b.constant_zero()
    b.emit_blank()

    # Stem
    b.emit_comment(f"Stem: 3x3 stride 2, 3->{stem_c}")
    cur_var, cur_type, cur_h, cur_w = b.conv_block(
        "%input", 1, 3, input_h, input_h, stem_c, 3, 2, 1, "%w_stem")
    cur_c = stem_c
    b.emit_blank()

    block_idx = 0
    for in_c, out_c, first_stride, expand_ratio, count in block_specs:
        for i in range(count):
            stride = first_stride if i == 0 else 1
            cur_in_c = in_c if i == 0 else out_c
            mid_c = cur_in_c * expand_ratio
            has_residual = (stride == 1 and cur_in_c == out_c)

            b.emit_comment(f"IRB {block_idx}: {cur_in_c}->{out_c} mid={mid_c} "
                           f"s={stride} {cur_h}x{cur_w}")

            block_input_var = cur_var
            block_input_type = cur_type

            if expand_ratio > 1:
                cur_var, cur_type, cur_h, cur_w = b.conv_block(
                    cur_var, 1, cur_in_c, cur_h, cur_w, mid_c, 1, 1, 0,
                    f"%w_blk{block_idx}_exp")

            cur_var, cur_type, cur_h, cur_w = b.conv_block(
                cur_var, 1, mid_c, cur_h, cur_w, mid_c, 3, stride, 1,
                f"%w_blk{block_idx}_dw")

            cur_var, cur_type, cur_h, cur_w = b.conv_block(
                cur_var, 1, mid_c, cur_h, cur_w, out_c, 1, 1, 0,
                f"%w_blk{block_idx}_proj", with_relu=False)

            if has_residual:
                cur_var, cur_type = b.add_4d(
                    cur_var, cur_type, block_input_var, block_input_type,
                    1, out_c, cur_h, cur_w)

            cur_c = out_c
            block_idx += 1
            b.emit_blank()

    # Final 1x1 conv
    b.emit_comment(f"Final 1x1: {cur_c}->{final_c}")
    cur_var, cur_type, cur_h, cur_w = b.conv_block(
        cur_var, 1, cur_c, cur_h, cur_w, final_c, 1, 1, 0, "%w_final")
    b.emit_blank()

    # FC as 1x1 conv
    b.emit_comment(f"FC: 1x1 {final_c}->{num_classes}")
    cur_var, cur_type, _, _ = b.conv_block(
        cur_var, 1, final_c, cur_h, cur_w, num_classes, 1, 1, 0,
        "%w_fc", with_relu=False)

    return b.build_function(name, cur_var, cur_type)


# ===========================================================================
# Transformer family (NLP/Vision)
# ===========================================================================

def build_transformer_variant(num_layers, hidden, ffn_dim, seq_len, name,
                              num_heads=8):
    """Build transformer stack (encoder-only, simplified attention)."""
    b = MLIRBuilder()
    b.add_param("%input", f"tensor<{seq_len}x{hidden}xf16>")

    for layer in range(num_layers):
        sfx = str(layer)
        b.add_param(f"%w_q{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_k{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_v{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_kt{sfx}", f"tensor<{hidden}x{seq_len}xf16>")
        b.add_param(f"%w_o{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_ff{sfx}_up", f"tensor<{hidden}x{ffn_dim}xf16>")
        b.add_param(f"%w_ff{sfx}_down", f"tensor<{ffn_dim}x{hidden}xf16>")

    b.constant_zero()
    b.emit_blank()

    cur_var = "%input"
    cur_type = f"tensor<{seq_len}x{hidden}xf16>"

    for layer in range(num_layers):
        sfx = str(layer)
        b.emit_comment(f"=== Transformer Layer {layer} ===")
        b.emit_blank()

        residual_var = cur_var
        residual_type = cur_type

        # Q, K, V projections
        b.emit_comment("Q projection")
        q_var, q_type = b.matmul(
            cur_var, cur_type,
            f"%w_q{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)

        b.emit_comment("K projection")
        k_var, k_type = b.matmul(
            cur_var, cur_type,
            f"%w_k{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)

        b.emit_comment("V projection")
        v_var, v_type = b.matmul(
            cur_var, cur_type,
            f"%w_v{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)
        b.emit_blank()

        # Attention scores: Q x K^T
        b.emit_comment("Attention scores: Q x K_transposed")
        scores_var, scores_type = b.matmul(
            q_var, q_type,
            f"%w_kt{sfx}", f"tensor<{hidden}x{seq_len}xf16>",
            seq_len, hidden, seq_len)

        # Softmax approximation (relu)
        b.emit_comment("Softmax approximation (relu)")
        soft_var, soft_type = b.relu_2d(scores_var, scores_type, seq_len, seq_len)
        b.emit_blank()

        # Attention output: scores x V
        b.emit_comment("Attention output: scores x V")
        attn_var, attn_type = b.matmul(
            soft_var, soft_type, v_var, v_type,
            seq_len, seq_len, hidden)

        # Output projection
        b.emit_comment("Output projection")
        out_proj_var, out_proj_type = b.matmul(
            attn_var, attn_type,
            f"%w_o{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)

        # Residual add
        b.emit_comment("Attention residual add")
        attn_res_var, attn_res_type = b.add_2d(
            out_proj_var, out_proj_type,
            residual_var, residual_type,
            seq_len, hidden)
        b.emit_blank()

        # FFN
        ffn_residual_var = attn_res_var
        ffn_residual_type = attn_res_type

        b.emit_comment("FFN: up projection")
        ff_up_var, ff_up_type = b.matmul(
            attn_res_var, attn_res_type,
            f"%w_ff{sfx}_up", f"tensor<{hidden}x{ffn_dim}xf16>",
            seq_len, hidden, ffn_dim)

        b.emit_comment("FFN ReLU")
        ff_relu_var, ff_relu_type = b.relu_2d(ff_up_var, ff_up_type, seq_len, ffn_dim)

        b.emit_comment("FFN: down projection")
        ff_down_var, ff_down_type = b.matmul(
            ff_relu_var, ff_relu_type,
            f"%w_ff{sfx}_down", f"tensor<{ffn_dim}x{hidden}xf16>",
            seq_len, ffn_dim, hidden)

        # FFN residual add
        b.emit_comment("FFN residual add")
        ffn_res_var, ffn_res_type = b.add_2d(
            ff_down_var, ff_down_type,
            ffn_residual_var, ffn_residual_type,
            seq_len, hidden)

        cur_var = ffn_res_var
        cur_type = ffn_res_type
        b.emit_blank()

    return b.build_function(name, cur_var, cur_type)


# ===========================================================================
# Vision Transformer family (patch embed + transformer)
# ===========================================================================

def build_vit_variant(patch_size, hidden, num_layers, ffn_dim, img_size, name,
                      num_classes=1000):
    """Build ViT: patch embedding (conv) + transformer + FC head."""
    b = MLIRBuilder()
    b.add_param("%input", f"tensor<1x3x{img_size}x{img_size}xf16>")

    # Patch embedding: conv with kernel=patch_size, stride=patch_size
    num_patches_side = img_size // patch_size
    num_patches = num_patches_side * num_patches_side
    # Model patch embed as conv 3->hidden with kernel=patch_size stride=patch_size
    b.add_param("%w_patch", f"tensor<{hidden}x3x{patch_size}x{patch_size}xf16>")

    # Transformer weights
    for layer in range(num_layers):
        sfx = str(layer)
        b.add_param(f"%w_q{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_k{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_v{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_kt{sfx}", f"tensor<{hidden}x{num_patches}xf16>")
        b.add_param(f"%w_o{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_ff{sfx}_up", f"tensor<{hidden}x{ffn_dim}xf16>")
        b.add_param(f"%w_ff{sfx}_down", f"tensor<{ffn_dim}x{hidden}xf16>")

    # Classification head
    b.add_param("%w_head", f"tensor<{hidden}x{num_classes}xf16>")

    b.constant_zero()
    b.emit_blank()

    # Patch embedding: [1,3,img,img] -> [1,hidden,num_patches_side,num_patches_side]
    b.emit_comment(f"Patch embedding: {patch_size}x{patch_size} conv, 3->{hidden}")
    cur_var, cur_type, cur_h, cur_w = b.conv_block(
        "%input", 1, 3, img_size, img_size, hidden, patch_size, patch_size, 0,
        "%w_patch", with_relu=False)
    b.emit_blank()

    # Flatten to 2D: [num_patches, hidden]
    # Model as a series of matmuls on [num_patches, hidden] conceptual tensor.
    # We create a [num_patches, hidden] tensor from tensor.empty (conceptual reshape).
    seq_len = num_patches
    seq_type = b._tensor_2d(seq_len, hidden)
    seq_empty = b._next_var("seq_empty")
    b.emit_comment(f"Conceptual reshape to [{seq_len}, {hidden}]")
    b.emit(f"{seq_empty} = tensor.empty() : {seq_type}")

    # Fill from the 4D output (conceptual)
    seq_fill = b._next_var("seq_fill")
    b.emit(f"{seq_fill} = linalg.fill ins(%cst : f16) "
           f"outs({seq_empty} : {seq_type}) -> {seq_type}")
    b.emit_blank()

    cur_var = seq_fill
    cur_type = seq_type

    # Transformer layers
    for layer in range(num_layers):
        sfx = str(layer)
        b.emit_comment(f"=== Transformer Layer {layer} ===")
        residual_var = cur_var
        residual_type = cur_type

        q_var, q_type = b.matmul(
            cur_var, cur_type,
            f"%w_q{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)

        k_var, k_type = b.matmul(
            cur_var, cur_type,
            f"%w_k{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)

        v_var, v_type = b.matmul(
            cur_var, cur_type,
            f"%w_v{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)

        scores_var, scores_type = b.matmul(
            q_var, q_type,
            f"%w_kt{sfx}", f"tensor<{hidden}x{seq_len}xf16>",
            seq_len, hidden, seq_len)

        soft_var, soft_type = b.relu_2d(scores_var, scores_type, seq_len, seq_len)

        attn_var, attn_type = b.matmul(
            soft_var, soft_type, v_var, v_type,
            seq_len, seq_len, hidden)

        out_proj_var, out_proj_type = b.matmul(
            attn_var, attn_type,
            f"%w_o{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)

        attn_res_var, attn_res_type = b.add_2d(
            out_proj_var, out_proj_type,
            residual_var, residual_type,
            seq_len, hidden)

        ffn_residual_var = attn_res_var
        ffn_residual_type = attn_res_type

        ff_up_var, ff_up_type = b.matmul(
            attn_res_var, attn_res_type,
            f"%w_ff{sfx}_up", f"tensor<{hidden}x{ffn_dim}xf16>",
            seq_len, hidden, ffn_dim)

        ff_relu_var, ff_relu_type = b.relu_2d(ff_up_var, ff_up_type, seq_len, ffn_dim)

        ff_down_var, ff_down_type = b.matmul(
            ff_relu_var, ff_relu_type,
            f"%w_ff{sfx}_down", f"tensor<{ffn_dim}x{hidden}xf16>",
            seq_len, ffn_dim, hidden)

        cur_var, cur_type = b.add_2d(
            ff_down_var, ff_down_type,
            ffn_residual_var, ffn_residual_type,
            seq_len, hidden)
        b.emit_blank()

    # Classification head: [seq_len, hidden] x [hidden, num_classes] -> [seq_len, num_classes]
    b.emit_comment(f"Classification head: matmul {hidden}->{num_classes}")
    cur_var, cur_type = b.matmul(
        cur_var, cur_type,
        "%w_head", f"tensor<{hidden}x{num_classes}xf16>",
        seq_len, hidden, num_classes)

    return b.build_function(name, cur_var, cur_type)


# ===========================================================================
# Detection models
# ===========================================================================

def build_yolo_variant(conv_specs, det_channels, name,
                       input_h=416, input_w=416):
    """Build YOLO-like detection network."""
    b = MLIRBuilder()
    b.add_param("%input", f"tensor<1x3x{input_h}x{input_w}xf16>")

    weights = []
    cur_c = 3
    for i, (oc, k, s, p) in enumerate(conv_specs):
        weights.append((f"%w{i}", f"tensor<{oc}x{cur_c}x{k}x{k}xf16>"))
        cur_c = oc
    weights.append(("%w_det", f"tensor<{det_channels}x{cur_c}x1x1xf16>"))

    for wn, wt in weights:
        b.add_param(wn, wt)

    b.constant_zero()
    b.emit_blank()

    cur_var = "%input"
    cur_c, cur_h, cur_w = 3, input_h, input_w

    for i, (oc, k, s, p) in enumerate(conv_specs):
        b.emit_comment(f"conv{i}: {k}x{k} s{s} {cur_c}->{oc} {cur_h}x{cur_w}")
        cur_var, cur_type, cur_h, cur_w = b.conv_block(
            cur_var, 1, cur_c, cur_h, cur_w, oc, k, s, p, f"%w{i}")
        cur_c = oc
        b.emit_blank()

    # Detection head
    b.emit_comment(f"Detection head: 1x1 {cur_c}->{det_channels}")
    cur_var, cur_type, _, _ = b.conv_block(
        cur_var, 1, cur_c, cur_h, cur_w, det_channels, 1, 1, 0,
        "%w_det", with_relu=False)

    return b.build_function(name, cur_var, cur_type)


# ===========================================================================
# Segmentation models (encoder-decoder)
# ===========================================================================

def build_encoder_decoder_variant(enc_specs, dec_specs, name,
                                  input_h=256, input_w=256, in_c=3,
                                  out_classes=21):
    """Build encoder-decoder segmentation network.
    enc_specs: list of (out_c, kernel, stride, pad) for encoder.
    dec_specs: list of (out_c, kernel, stride, pad) for decoder.
    The decoder uses stride-1 convs (upsampling is conceptual).
    """
    b = MLIRBuilder()
    b.add_param("%input", f"tensor<1x{in_c}x{input_h}x{input_w}xf16>")

    weights = []
    cur_c = in_c
    for i, (oc, k, s, p) in enumerate(enc_specs):
        weights.append((f"%w_enc{i}", f"tensor<{oc}x{cur_c}x{k}x{k}xf16>"))
        cur_c = oc

    for i, (oc, k, s, p) in enumerate(dec_specs):
        weights.append((f"%w_dec{i}", f"tensor<{oc}x{cur_c}x{k}x{k}xf16>"))
        cur_c = oc

    weights.append(("%w_out", f"tensor<{out_classes}x{cur_c}x1x1xf16>"))

    for wn, wt in weights:
        b.add_param(wn, wt)

    b.constant_zero()
    b.emit_blank()

    cur_var = "%input"
    cur_c = in_c
    cur_h, cur_w = input_h, input_w

    b.emit_comment("=== Encoder ===")
    for i, (oc, k, s, p) in enumerate(enc_specs):
        b.emit_comment(f"enc{i}: {k}x{k} s{s} {cur_c}->{oc} {cur_h}x{cur_w}")
        cur_var, cur_type, cur_h, cur_w = b.conv_block(
            cur_var, 1, cur_c, cur_h, cur_w, oc, k, s, p, f"%w_enc{i}")
        cur_c = oc
    b.emit_blank()

    b.emit_comment("=== Decoder ===")
    for i, (oc, k, s, p) in enumerate(dec_specs):
        b.emit_comment(f"dec{i}: {k}x{k} s{s} {cur_c}->{oc} {cur_h}x{cur_w}")
        cur_var, cur_type, cur_h, cur_w = b.conv_block(
            cur_var, 1, cur_c, cur_h, cur_w, oc, k, s, p, f"%w_dec{i}")
        cur_c = oc
    b.emit_blank()

    # Output conv
    b.emit_comment(f"Output: 1x1 {cur_c}->{out_classes}")
    cur_var, cur_type, _, _ = b.conv_block(
        cur_var, 1, cur_c, cur_h, cur_w, out_classes, 1, 1, 0,
        "%w_out", with_relu=False)

    return b.build_function(name, cur_var, cur_type)


# ===========================================================================
# GAN / Generative models (sequential CNN, generator only)
# ===========================================================================

def build_gan_generator(conv_specs, name, input_c=100, input_h=4, input_w=4):
    """Build a generator network (sequential convs, no stride >1 up).
    Uses 3x3 stride-1 convs with padding to maintain spatial dims.
    """
    b = MLIRBuilder()
    b.add_param("%input", f"tensor<1x{input_c}x{input_h}x{input_w}xf16>")

    weights = []
    cur_c = input_c
    for i, (oc, k, s, p, relu) in enumerate(conv_specs):
        weights.append((f"%w{i}", f"tensor<{oc}x{cur_c}x{k}x{k}xf16>"))
        cur_c = oc

    for wn, wt in weights:
        b.add_param(wn, wt)

    b.constant_zero()
    b.emit_blank()

    cur_var = "%input"
    cur_c = input_c
    cur_h, cur_w = input_h, input_w

    for i, (oc, k, s, p, relu) in enumerate(conv_specs):
        b.emit_comment(f"conv{i}: {k}x{k} s{s} p{p} {cur_c}->{oc} {cur_h}x{cur_w}")
        cur_var, cur_type, cur_h, cur_w = b.conv_block(
            cur_var, 1, cur_c, cur_h, cur_w, oc, k, s, p,
            f"%w{i}", with_relu=relu)
        cur_c = oc
        b.emit_blank()

    return b.build_function(name, cur_var, cur_type)


# ===========================================================================
# Autoencoder / VAE (encoder -> bottleneck -> decoder, 2D matmul based)
# ===========================================================================

def build_autoencoder_variant(layer_sizes, name, input_dim=784):
    """Build autoencoder with fully-connected layers (matmul).
    layer_sizes: list of hidden dims for encoder. Decoder mirrors.
    """
    b = MLIRBuilder()
    b.add_param("%input", f"tensor<1x{input_dim}xf16>")

    # Encoder weights
    weights = []
    prev = input_dim
    for i, sz in enumerate(layer_sizes):
        weights.append((f"%w_enc{i}", f"tensor<{prev}x{sz}xf16>"))
        prev = sz

    # Decoder weights (mirror)
    dec_sizes = list(reversed(layer_sizes[:-1])) + [input_dim]
    for i, sz in enumerate(dec_sizes):
        weights.append((f"%w_dec{i}", f"tensor<{prev}x{sz}xf16>"))
        prev = sz

    for wn, wt in weights:
        b.add_param(wn, wt)

    b.constant_zero()
    b.emit_blank()

    cur_var = "%input"
    cur_type = f"tensor<1x{input_dim}xf16>"
    cur_dim = input_dim

    # Encoder
    b.emit_comment("=== Encoder ===")
    for i, sz in enumerate(layer_sizes):
        b.emit_comment(f"enc{i}: {cur_dim}->{sz}")
        cur_var, cur_type = b.matmul(
            cur_var, cur_type,
            f"%w_enc{i}", f"tensor<{cur_dim}x{sz}xf16>",
            1, cur_dim, sz)
        cur_var, cur_type = b.relu_2d(cur_var, cur_type, 1, sz)
        cur_dim = sz
    b.emit_blank()

    # Decoder
    b.emit_comment("=== Decoder ===")
    for i, sz in enumerate(dec_sizes):
        is_last = (i == len(dec_sizes) - 1)
        b.emit_comment(f"dec{i}: {cur_dim}->{sz}")
        cur_var, cur_type = b.matmul(
            cur_var, cur_type,
            f"%w_dec{i}", f"tensor<{cur_dim}x{sz}xf16>",
            1, cur_dim, sz)
        if not is_last:
            cur_var, cur_type = b.relu_2d(cur_var, cur_type, 1, sz)
        cur_dim = sz

    return b.build_function(name, cur_var, cur_type)


# ===========================================================================
# Swin Transformer (simplified as sequential transformer blocks with
# different sequence lengths per stage)
# ===========================================================================

def build_swin_variant(depths, embed_dim, name, img_size=224, patch_size=4,
                       num_classes=1000):
    """Build Swin Transformer (simplified).
    depths: list of transformer block counts per stage, e.g. [2, 2, 6, 2].
    embed_dim: base embedding dimension (96 for Tiny).
    Each stage has dim = embed_dim * 2^stage_idx.
    Sequence length halves each stage (patch merging).
    """
    b = MLIRBuilder()
    b.add_param("%input", f"tensor<1x3x{img_size}x{img_size}xf16>")

    num_patches_side = img_size // patch_size
    num_patches = num_patches_side * num_patches_side

    # Patch embedding as conv
    b.add_param("%w_patch", f"tensor<{embed_dim}x3x{patch_size}x{patch_size}xf16>")

    # Weights per stage
    for stage_idx, depth in enumerate(depths):
        dim = embed_dim * (2 ** stage_idx)
        seq_len = num_patches // (4 ** stage_idx)
        ffn_dim = dim * 4

        for layer in range(depth):
            pfx = f"%w_s{stage_idx}_l{layer}"
            b.add_param(f"{pfx}_q", f"tensor<{dim}x{dim}xf16>")
            b.add_param(f"{pfx}_k", f"tensor<{dim}x{dim}xf16>")
            b.add_param(f"{pfx}_v", f"tensor<{dim}x{dim}xf16>")
            b.add_param(f"{pfx}_kt", f"tensor<{dim}x{seq_len}xf16>")
            b.add_param(f"{pfx}_o", f"tensor<{dim}x{dim}xf16>")
            b.add_param(f"{pfx}_ff_up", f"tensor<{dim}x{ffn_dim}xf16>")
            b.add_param(f"{pfx}_ff_dn", f"tensor<{ffn_dim}x{dim}xf16>")

        # Patch merging (except last stage): project dim -> 2*dim, seq_len -> seq_len/4
        if stage_idx < len(depths) - 1:
            next_dim = embed_dim * (2 ** (stage_idx + 1))
            b.add_param(f"%w_merge{stage_idx}", f"tensor<{dim}x{next_dim}xf16>")

    # Head
    final_dim = embed_dim * (2 ** (len(depths) - 1))
    final_seq = num_patches // (4 ** (len(depths) - 1))
    b.add_param("%w_head", f"tensor<{final_dim}x{num_classes}xf16>")

    b.constant_zero()
    b.emit_blank()

    # Patch embedding
    b.emit_comment(f"Patch embedding: {patch_size}x{patch_size} conv")
    cur_var, cur_type, cur_h, cur_w = b.conv_block(
        "%input", 1, 3, img_size, img_size, embed_dim,
        patch_size, patch_size, 0, "%w_patch", with_relu=False)
    b.emit_blank()

    # Flatten to 2D
    seq_len = num_patches
    dim = embed_dim
    seq_type = b._tensor_2d(seq_len, dim)
    seq_var = b._next_var("seq")
    b.emit_comment(f"Reshape to [{seq_len}, {dim}]")
    b.emit(f"{seq_var} = tensor.empty() : {seq_type}")
    seq_fill = b._next_var("fill")
    b.emit(f"{seq_fill} = linalg.fill ins(%cst : f16) "
           f"outs({seq_var} : {seq_type}) -> {seq_type}")
    cur_var = seq_fill
    cur_type = seq_type
    b.emit_blank()

    for stage_idx, depth in enumerate(depths):
        dim = embed_dim * (2 ** stage_idx)
        seq_len = num_patches // (4 ** stage_idx)
        ffn_dim = dim * 4

        b.emit_comment(f"=== Swin Stage {stage_idx}: seq={seq_len}, dim={dim} ===")

        for layer in range(depth):
            pfx = f"%w_s{stage_idx}_l{layer}"
            residual_var = cur_var
            residual_type = cur_type

            q_var, q_type = b.matmul(cur_var, cur_type,
                f"{pfx}_q", f"tensor<{dim}x{dim}xf16>", seq_len, dim, dim)
            k_var, k_type = b.matmul(cur_var, cur_type,
                f"{pfx}_k", f"tensor<{dim}x{dim}xf16>", seq_len, dim, dim)
            v_var, v_type = b.matmul(cur_var, cur_type,
                f"{pfx}_v", f"tensor<{dim}x{dim}xf16>", seq_len, dim, dim)

            scores_var, scores_type = b.matmul(q_var, q_type,
                f"{pfx}_kt", f"tensor<{dim}x{seq_len}xf16>", seq_len, dim, seq_len)
            soft_var, soft_type = b.relu_2d(scores_var, scores_type, seq_len, seq_len)

            attn_var, attn_type = b.matmul(soft_var, soft_type, v_var, v_type,
                seq_len, seq_len, dim)
            out_var, out_type = b.matmul(attn_var, attn_type,
                f"{pfx}_o", f"tensor<{dim}x{dim}xf16>", seq_len, dim, dim)

            res_var, res_type = b.add_2d(out_var, out_type,
                residual_var, residual_type, seq_len, dim)

            ffn_res = res_var
            ffn_res_type = res_type

            up_var, up_type = b.matmul(res_var, res_type,
                f"{pfx}_ff_up", f"tensor<{dim}x{ffn_dim}xf16>", seq_len, dim, ffn_dim)
            relu_var, relu_type = b.relu_2d(up_var, up_type, seq_len, ffn_dim)
            dn_var, dn_type = b.matmul(relu_var, relu_type,
                f"{pfx}_ff_dn", f"tensor<{ffn_dim}x{dim}xf16>", seq_len, ffn_dim, dim)

            cur_var, cur_type = b.add_2d(dn_var, dn_type,
                ffn_res, ffn_res_type, seq_len, dim)
        b.emit_blank()

        # Patch merging: reduce seq_len by 4x, increase dim by 2x
        if stage_idx < len(depths) - 1:
            next_dim = embed_dim * (2 ** (stage_idx + 1))
            next_seq = num_patches // (4 ** (stage_idx + 1))
            b.emit_comment(f"Patch merging: [{seq_len},{dim}] -> [{next_seq},{next_dim}]")
            # Model as matmul [seq_len, dim] x [dim, next_dim] -> [seq_len, next_dim]
            # Then conceptual reshape to [next_seq, next_dim]
            merge_var, merge_type = b.matmul(cur_var, cur_type,
                f"%w_merge{stage_idx}", f"tensor<{dim}x{next_dim}xf16>",
                seq_len, dim, next_dim)
            # Conceptual reshape to [next_seq, next_dim]
            new_type = b._tensor_2d(next_seq, next_dim)
            new_var = b._next_var("merge_reshape")
            b.emit(f"{new_var} = tensor.empty() : {new_type}")
            fill_var = b._next_var("fill")
            b.emit(f"{fill_var} = linalg.fill ins(%cst : f16) "
                   f"outs({new_var} : {new_type}) -> {new_type}")
            cur_var = fill_var
            cur_type = new_type
            seq_len = next_seq
            dim = next_dim
            b.emit_blank()

    # Classification head
    final_dim = embed_dim * (2 ** (len(depths) - 1))
    final_seq = num_patches // (4 ** (len(depths) - 1))
    b.emit_comment(f"Head: [{final_seq},{final_dim}] x [{final_dim},{num_classes}]")
    cur_var, cur_type = b.matmul(cur_var, cur_type,
        "%w_head", f"tensor<{final_dim}x{num_classes}xf16>",
        final_seq, final_dim, num_classes)

    return b.build_function(name, cur_var, cur_type)


# ===========================================================================
# EfficientNet family (uses sequential CNN with varying widths/depths)
# ===========================================================================

def _make_efficientnet_specs(width_mult, depth_mult, input_size):
    """Generate EfficientNet conv specs given scaling factors."""
    def _round(x, mult):
        return max(1, int(x * mult))

    # Base EfficientNet-B0 blocks: (out_c, kernel, stride, pad, num_repeats, expand_ratio)
    base_blocks = [
        (16, 3, 1, 1, 1, 1),
        (24, 3, 2, 1, 2, 6),
        (40, 3, 2, 1, 2, 6),
        (80, 3, 2, 1, 3, 6),
        (112, 3, 1, 1, 3, 6),
        (192, 3, 2, 1, 4, 6),
        (320, 3, 1, 1, 1, 6),
    ]

    # Stem
    stem_c = _round(32, width_mult)
    specs = [(stem_c, 3, 2, 1)]  # stem conv

    cur_c = stem_c
    for oc, k, s, p, repeats, expand in base_blocks:
        scaled_oc = _round(oc, width_mult)
        scaled_repeats = max(1, int(repeats * depth_mult + 0.5))
        for r in range(scaled_repeats):
            stride = s if r == 0 else 1
            mid_c = cur_c * expand
            if expand > 1:
                specs.append((mid_c, 1, 1, 0))  # expand
            specs.append((mid_c, k, stride, p))  # depthwise
            specs.append((scaled_oc, 1, 1, 0))   # project
            cur_c = scaled_oc

    # Head
    final_c = _round(1280, width_mult)
    specs.append((final_c, 1, 1, 0))  # final conv

    return specs, input_size


def build_efficientnet_variant(width_mult, depth_mult, input_size, name):
    """Build EfficientNet variant."""
    specs, img_size = _make_efficientnet_specs(width_mult, depth_mult, input_size)
    conv_specs = [(oc, k, s, p, True) for oc, k, s, p in specs]
    return build_sequential_cnn(conv_specs, name, input_c=3,
                                input_h=img_size, input_w=img_size,
                                fc_classes=1000, fc_as_1x1=True)


# ===========================================================================
# ShuffleNet family
# ===========================================================================

def build_shufflenet_variant(stage_channels, stage_repeats, name):
    """Build ShuffleNet variant as sequential convs.
    stage_channels: list of output channels per stage.
    stage_repeats: list of repeat counts per stage.
    """
    conv_specs = []
    # Stem: 3x3 stride 2 + 3x3 stride 2 (simulated maxpool)
    conv_specs.append((24, 3, 2, 1, True))
    conv_specs.append((24, 3, 2, 1, True))

    cur_c = 24
    for sc, sr in zip(stage_channels, stage_repeats):
        # First block: stride 2
        conv_specs.append((sc, 3, 2, 1, True))
        cur_c = sc
        # Remaining blocks: stride 1
        for _ in range(sr - 1):
            conv_specs.append((sc, 3, 1, 1, True))

    return build_sequential_cnn(conv_specs, name, input_c=3,
                                input_h=224, input_w=224,
                                fc_classes=1000, fc_as_1x1=True)


# ===========================================================================
# SqueezeNet family
# ===========================================================================

def build_squeezenet_variant(fire_configs, name):
    """Build SqueezeNet variant.
    fire_configs: list of (squeeze_c, expand_c) for fire modules.
    Data flow: squeeze(1x1) -> expand_1x1 -> expand_3x3 (serialized).
    """
    b = MLIRBuilder()
    b.add_param("%input", "tensor<1x3x224x224xf16>")

    weights = []
    weights.append(("%w_stem", "tensor<96x3x7x7xf16>"))

    cur_c = 96
    for i, (sq, ex) in enumerate(fire_configs):
        weights.append((f"%w_fire{i}_sq", f"tensor<{sq}x{cur_c}x1x1xf16>"))
        weights.append((f"%w_fire{i}_e1", f"tensor<{ex}x{sq}x1x1xf16>"))
        # e3 takes input from e1 output (ex channels), not from squeeze (sq channels)
        weights.append((f"%w_fire{i}_e3", f"tensor<{ex}x{ex}x3x3xf16>"))
        cur_c = ex

    weights.append(("%w_final", f"tensor<1000x{cur_c}x1x1xf16>"))

    for wn, wt in weights:
        b.add_param(wn, wt)

    b.constant_zero()
    b.emit_blank()

    b.emit_comment("Stem: 7x7 stride 2, 3->96")
    cur_var, cur_type, cur_h, cur_w = b.conv_block(
        "%input", 1, 3, 224, 224, 96, 7, 2, 3, "%w_stem")
    cur_c = 96
    b.emit_blank()

    for i, (sq, ex) in enumerate(fire_configs):
        b.emit_comment(f"Fire{i}: squeeze={sq}, expand={ex}")
        cur_var, cur_type, cur_h, cur_w = b.conv_block(
            cur_var, 1, cur_c, cur_h, cur_w, sq, 1, 1, 0,
            f"%w_fire{i}_sq")
        cur_var, cur_type, cur_h, cur_w = b.conv_block(
            cur_var, 1, sq, cur_h, cur_w, ex, 1, 1, 0,
            f"%w_fire{i}_e1")
        cur_var, cur_type, cur_h, cur_w = b.conv_block(
            cur_var, 1, ex, cur_h, cur_w, ex, 3, 1, 1,
            f"%w_fire{i}_e3")
        cur_c = ex
        b.emit_blank()

    b.emit_comment(f"Final 1x1 conv: {cur_c}->1000")
    cur_var, cur_type, _, _ = b.conv_block(
        cur_var, 1, cur_c, cur_h, cur_w, 1000, 1, 1, 0,
        "%w_final", with_relu=False)

    return b.build_function(name, cur_var, cur_type)


# ===========================================================================
# Super-resolution models
# ===========================================================================

def build_srcnn(name="srcnn"):
    """SRCNN: 3 conv layers for super-resolution."""
    conv_specs = [
        (64, 9, 1, 4, True),    # patch extraction
        (32, 1, 1, 0, True),    # non-linear mapping
        (3, 5, 1, 2, False),    # reconstruction
    ]
    return build_sequential_cnn(conv_specs, name, input_c=3,
                                input_h=224, input_w=224,
                                fc_classes=3, fc_as_1x1=True)


def build_espcn(name="espcn"):
    """ESPCN: efficient sub-pixel CNN for super-resolution."""
    # Simple version: conv layers maintaining spatial dims
    conv_specs = [
        (64, 5, 1, 2, True),
        (32, 3, 1, 1, True),
        (12, 3, 1, 1, False),  # 12 = 3 * 2^2 for 2x upscale
    ]
    return build_sequential_cnn(conv_specs, name, input_c=3,
                                input_h=224, input_w=224,
                                fc_classes=12, fc_as_1x1=True)


# ===========================================================================
# Wide ResNet
# ===========================================================================

def build_wide_resnet_variant(layers, widen_factor, name):
    """Wide ResNet: ResNet with wider channels.
    E.g., WRN-50-2 has resnet50 layers with 2x channel width.
    """
    # Use bottleneck blocks with wider mid channels
    b = MLIRBuilder()
    b.add_param("%input", "tensor<1x3x224x224xf16>")

    stage_base = [64, 128, 256, 512]
    stage_channels = [ch * widen_factor for ch in stage_base]
    # Bottleneck expansion = 4, but mid channels are widened
    stage_out_channels = [ch * 4 for ch in stage_base]  # standard expansion

    # For Wide ResNet, the mid_c is wider. We model it as bottleneck with
    # mid_c = base_ch * widen_factor and out_c = base_ch * 4
    # Actually for WRN-50-2, typically out_c = base * 4, mid = base * widen
    weights = [
        ("%w_conv1", "tensor<64x3x7x7xf16>"),
        ("%w_pool", "tensor<64x64x3x3xf16>"),
    ]

    for stage_idx in range(4):
        base_ch = stage_base[stage_idx]
        mid_c = base_ch * widen_factor
        out_c = base_ch * 4
        in_c_first = stage_base[stage_idx - 1] * 4 if stage_idx > 0 else 64

        for blk_idx in range(layers[stage_idx]):
            cur_in_c = in_c_first if blk_idx == 0 else out_c
            stride = 2 if (stage_idx > 0 and blk_idx == 0) else 1
            prefix = f"%w_s{stage_idx}_b{blk_idx}"

            weights.append((f"{prefix}_c1", f"tensor<{mid_c}x{cur_in_c}x1x1xf16>"))
            weights.append((f"{prefix}_c2", f"tensor<{mid_c}x{mid_c}x3x3xf16>"))
            weights.append((f"{prefix}_c3", f"tensor<{out_c}x{mid_c}x1x1xf16>"))

            if cur_in_c != out_c or stride > 1:
                weights.append((f"{prefix}_sc", f"tensor<{out_c}x{cur_in_c}x1x1xf16>"))

    final_c = stage_base[3] * 4
    weights.append(("%w_fc", f"tensor<1000x{final_c}x7x7xf16>"))

    for wn, wt in weights:
        b.add_param(wn, wt)

    b.constant_zero()
    b.emit_blank()

    cur_var, cur_type, cur_h, cur_w = b.conv_block(
        "%input", 1, 3, 224, 224, 64, 7, 2, 3, "%w_conv1")
    cur_c = 64
    b.emit_blank()

    cur_var, cur_type, cur_h, cur_w = b.conv_block(
        cur_var, 1, 64, cur_h, cur_w, 64, 3, 2, 1, "%w_pool")
    b.emit_blank()

    for stage_idx in range(4):
        base_ch = stage_base[stage_idx]
        mid_c = base_ch * widen_factor
        out_c = base_ch * 4
        in_c_first = stage_base[stage_idx - 1] * 4 if stage_idx > 0 else 64

        b.emit_comment(f"=== Stage {stage_idx} (wide) ===")
        for blk_idx in range(layers[stage_idx]):
            cur_in_c = in_c_first if blk_idx == 0 else out_c
            stride = 2 if (stage_idx > 0 and blk_idx == 0) else 1
            prefix = f"%w_s{stage_idx}_b{blk_idx}"
            needs_sc = (cur_in_c != out_c or stride > 1)
            w_sc = f"{prefix}_sc" if needs_sc else None

            cur_var, cur_type, cur_h, cur_w = _make_resnet_bottleneck_block(
                b, cur_var, cur_in_c, out_c, cur_h, cur_w, stride,
                f"{prefix}_c1", f"{prefix}_c2", f"{prefix}_c3", w_sc,
                override_mid_c=mid_c)
            cur_c = out_c
            b.emit_blank()

    # FC
    fc_out_type = b._tensor_4d(1, 1000, 1, 1)
    fc_w_type = b._tensor_4d(1000, cur_c, cur_h, cur_w)
    fc_init = b._next_var("fc_init")
    fc_fill = b._next_var("fc_fill")
    fc_var = b._next_var("fc")
    b.emit(f"{fc_init} = tensor.empty() : {fc_out_type}")
    b.emit(f"{fc_fill} = linalg.fill ins(%cst : f16) "
           f"outs({fc_init} : {fc_out_type}) -> {fc_out_type}")
    b.emit(f"{fc_var} = linalg.conv_2d_nchw_fchw {{")
    b.emit(f"  dilations = dense<1> : tensor<2xi64>,")
    b.emit(f"  strides = dense<1> : tensor<2xi64>")
    b.emit(f"}} ins({cur_var}, %w_fc : {cur_type}, {fc_w_type})")
    b.emit(f"  outs({fc_fill} : {fc_out_type}) -> {fc_out_type}")

    return b.build_function(name, fc_var, fc_out_type)


# ===========================================================================
# ConvNeXt family
# ===========================================================================

def build_convnext_variant(depths, dims, name, num_classes=1000):
    """Build ConvNeXt (simplified).
    depths: e.g. [3, 3, 9, 3] for Tiny.
    dims: e.g. [96, 192, 384, 768] for Tiny.
    """
    b = MLIRBuilder()
    b.add_param("%input", "tensor<1x3x224x224xf16>")

    weights = []
    # Stem: 4x4 conv stride 4
    weights.append(("%w_stem", f"tensor<{dims[0]}x3x4x4xf16>"))

    for stage_idx, (depth, dim) in enumerate(zip(depths, dims)):
        # Downsample between stages (except first)
        if stage_idx > 0:
            prev_dim = dims[stage_idx - 1]
            weights.append((f"%w_ds{stage_idx}",
                            f"tensor<{dim}x{prev_dim}x3x3xf16>"))

        for blk_idx in range(depth):
            pfx = f"%w_s{stage_idx}_b{blk_idx}"
            # ConvNeXt block: depthwise 7x7 + 1x1 + 1x1
            weights.append((f"{pfx}_dw", f"tensor<{dim}x{dim}x7x7xf16>"))
            weights.append((f"{pfx}_pw1", f"tensor<{dim * 4}x{dim}x1x1xf16>"))
            weights.append((f"{pfx}_pw2", f"tensor<{dim}x{dim * 4}x1x1xf16>"))

    # Head
    weights.append(("%w_fc", f"tensor<{num_classes}x{dims[-1]}x1x1xf16>"))

    for wn, wt in weights:
        b.add_param(wn, wt)

    b.constant_zero()
    b.emit_blank()

    # Stem: 4x4 stride 4
    b.emit_comment(f"Stem: 4x4 stride 4, 3->{dims[0]}")
    cur_var, cur_type, cur_h, cur_w = b.conv_block(
        "%input", 1, 3, 224, 224, dims[0], 4, 4, 0, "%w_stem", with_relu=False)
    cur_c = dims[0]
    b.emit_blank()

    for stage_idx, (depth, dim) in enumerate(zip(depths, dims)):
        if stage_idx > 0:
            prev_dim = dims[stage_idx - 1]
            b.emit_comment(f"Downsample: {prev_dim}->{dim}, stride 2")
            cur_var, cur_type, cur_h, cur_w = b.conv_block(
                cur_var, 1, prev_dim, cur_h, cur_w, dim, 3, 2, 1,
                f"%w_ds{stage_idx}", with_relu=False)
            cur_c = dim
            b.emit_blank()

        b.emit_comment(f"=== ConvNeXt Stage {stage_idx}: dim={dim} ===")
        for blk_idx in range(depth):
            pfx = f"%w_s{stage_idx}_b{blk_idx}"
            block_input = cur_var
            block_input_type = cur_type

            # Depthwise 7x7
            cur_var, cur_type, cur_h, cur_w = b.conv_block(
                cur_var, 1, dim, cur_h, cur_w, dim, 7, 1, 3, f"{pfx}_dw",
                with_relu=False)

            # Pointwise expand
            cur_var, cur_type, cur_h, cur_w = b.conv_block(
                cur_var, 1, dim, cur_h, cur_w, dim * 4, 1, 1, 0, f"{pfx}_pw1")

            # Pointwise project
            cur_var, cur_type, cur_h, cur_w = b.conv_block(
                cur_var, 1, dim * 4, cur_h, cur_w, dim, 1, 1, 0, f"{pfx}_pw2",
                with_relu=False)

            # Residual
            cur_var, cur_type = b.add_4d(
                cur_var, cur_type, block_input, block_input_type,
                1, dim, cur_h, cur_w)
            cur_c = dim
        b.emit_blank()

    # FC
    b.emit_comment(f"FC: 1x1 {cur_c}->{num_classes}")
    cur_var, cur_type, _, _ = b.conv_block(
        cur_var, 1, cur_c, cur_h, cur_w, num_classes, 1, 1, 0,
        "%w_fc", with_relu=False)

    return b.build_function(name, cur_var, cur_type)


# ===========================================================================
# RegNet family (simplified as sequential stages of grouped convs)
# ===========================================================================

def build_regnet_variant(stage_widths, stage_depths, name, num_classes=1000):
    """Build RegNet variant as sequential conv blocks.
    stage_widths: list of channel widths per stage.
    stage_depths: list of block counts per stage.
    """
    conv_specs = []
    # Stem
    conv_specs.append((32, 3, 2, 1, True))

    cur_c = 32
    for width, depth in zip(stage_widths, stage_depths):
        # First block with stride 2
        conv_specs.append((width, 3, 2, 1, True))
        cur_c = width
        for _ in range(depth - 1):
            conv_specs.append((width, 3, 1, 1, True))

    return build_sequential_cnn(conv_specs, name, input_c=3,
                                input_h=224, input_w=224,
                                fc_classes=num_classes, fc_as_1x1=True)


# ===========================================================================
# MNASNet family
# ===========================================================================

def build_mnasnet_variant(width_mult, name):
    """Build MNASNet variant (similar to MobileNetV2)."""
    def _scale(c):
        return max(8, int(c * width_mult + 0.5))

    block_specs = [
        (_scale(32), _scale(16), 1, 1, 1),
        (_scale(16), _scale(24), 2, 3, 3),
        (_scale(24), _scale(40), 2, 3, 3),
        (_scale(40), _scale(80), 2, 6, 3),
        (_scale(80), _scale(96), 1, 6, 2),
        (_scale(96), _scale(192), 2, 6, 4),
        (_scale(192), _scale(320), 1, 6, 1),
    ]
    return build_mobilenet_variant(block_specs, name, stem_c=_scale(32),
                                   final_c=_scale(1280))


# ===========================================================================
# Inception V3 (simplified)
# ===========================================================================

def build_inception_v3(name="inception_v3"):
    """Simplified Inception V3 with sequential conv blocks."""
    conv_specs = [
        # Stem
        (32, 3, 2, 1, True),
        (32, 3, 1, 1, True),
        (64, 3, 1, 1, True),
        (64, 3, 2, 1, True),   # pool
        (80, 1, 1, 0, True),
        (192, 3, 1, 1, True),
        (192, 3, 2, 1, True),  # pool
        # Inception blocks (simplified as conv sequences)
        (256, 1, 1, 0, True),
        (288, 3, 1, 1, True),
        (288, 1, 1, 0, True),
        (288, 3, 1, 1, True),
        # Reduction
        (768, 3, 2, 1, True),
        # More inception blocks
        (768, 1, 1, 0, True),
        (768, 3, 1, 1, True),
        (768, 1, 1, 0, True),
        (768, 3, 1, 1, True),
        # Reduction
        (1280, 3, 2, 1, True),
        # Final inception blocks
        (2048, 3, 1, 1, True),
        (2048, 1, 1, 0, True),
    ]
    return build_sequential_cnn(conv_specs, name, input_c=3,
                                input_h=299, input_w=299)


# ===========================================================================
# GoogLeNet (simplified)
# ===========================================================================

def build_googlenet(name="googlenet"):
    """Simplified GoogLeNet."""
    conv_specs = [
        (64, 7, 2, 3, True),    # conv1
        (64, 3, 2, 1, True),    # pool
        (64, 1, 1, 0, True),
        (192, 3, 1, 1, True),   # conv2
        (192, 3, 2, 1, True),   # pool
        # Inception modules (simplified)
        (256, 1, 1, 0, True),
        (256, 3, 1, 1, True),
        (480, 1, 1, 0, True),
        (480, 3, 1, 1, True),
        (480, 3, 2, 1, True),   # pool
        (512, 1, 1, 0, True),
        (512, 3, 1, 1, True),
        (512, 1, 1, 0, True),
        (512, 3, 1, 1, True),
        (528, 1, 1, 0, True),
        (832, 3, 2, 1, True),   # pool
        (832, 1, 1, 0, True),
        (832, 3, 1, 1, True),
        (1024, 1, 1, 0, True),
    ]
    return build_sequential_cnn(conv_specs, name, input_c=3,
                                input_h=224, input_w=224)


# ===========================================================================
# AlexNet
# ===========================================================================

def build_alexnet(name="alexnet"):
    """Simplified AlexNet."""
    conv_specs = [
        (64, 11, 4, 2, True),   # conv1: 224->55
        (64, 3, 2, 1, True),    # pool: 55->28
        (192, 5, 1, 2, True),   # conv2
        (192, 3, 2, 1, True),   # pool: 28->14
        (384, 3, 1, 1, True),   # conv3
        (256, 3, 1, 1, True),   # conv4
        (256, 3, 1, 1, True),   # conv5
        (256, 3, 2, 1, True),   # pool: 14->7
    ]
    return build_sequential_cnn(conv_specs, name, input_c=3,
                                input_h=224, input_w=224,
                                fc_classes=1000, fc_as_1x1=False)


# ===========================================================================
# SSD detection (MobileNet or VGG backbone)
# ===========================================================================

def build_ssd_mobilenet(name="ssd_mobilenet"):
    """SSD with MobileNet-like backbone."""
    conv_specs = [
        (32, 3, 2, 1),
        (64, 3, 1, 1),
        (128, 3, 2, 1),
        (128, 3, 1, 1),
        (256, 3, 2, 1),
        (256, 3, 1, 1),
        (512, 3, 2, 1),
        (512, 3, 1, 1),
        (512, 3, 1, 1),
        (1024, 3, 2, 1),
        (1024, 3, 1, 1),
    ]
    # 6 default boxes * (num_classes + 4) per cell, ~255 for 80 classes
    return build_yolo_variant(conv_specs, 255, name, input_h=300, input_w=300)


def build_ssd_vgg16(name="ssd_vgg16"):
    """SSD with VGG16-like backbone."""
    conv_specs = [
        (64, 3, 1, 1), (64, 3, 1, 1),
        (64, 3, 2, 1),  # pool
        (128, 3, 1, 1), (128, 3, 1, 1),
        (128, 3, 2, 1),  # pool
        (256, 3, 1, 1), (256, 3, 1, 1), (256, 3, 1, 1),
        (256, 3, 2, 1),  # pool
        (512, 3, 1, 1), (512, 3, 1, 1), (512, 3, 1, 1),
        (512, 3, 2, 1),  # pool
        (512, 3, 1, 1), (512, 3, 1, 1), (512, 3, 1, 1),
        # Extra layers for SSD
        (1024, 3, 1, 1),
        (1024, 1, 1, 0),
        (256, 1, 1, 0),
        (512, 3, 2, 1),
    ]
    return build_yolo_variant(conv_specs, 255, name, input_h=300, input_w=300)


# ===========================================================================
# Faster R-CNN (simplified backbone + RPN head)
# ===========================================================================

def build_faster_rcnn(name="faster_rcnn_r50"):
    """Simplified Faster R-CNN with ResNet-50-like backbone + RPN head."""
    # Backbone as sequential convs
    conv_specs = [
        (64, 7, 2, 3), (64, 3, 2, 1),  # stem + pool
        (64, 1, 1, 0), (64, 3, 1, 1), (256, 1, 1, 0),  # stage1 block
        (64, 1, 1, 0), (64, 3, 1, 1), (256, 1, 1, 0),
        (64, 1, 1, 0), (64, 3, 1, 1), (256, 1, 1, 0),
        (128, 1, 1, 0), (128, 3, 2, 1), (512, 1, 1, 0),  # stage2
        (128, 1, 1, 0), (128, 3, 1, 1), (512, 1, 1, 0),
        (256, 1, 1, 0), (256, 3, 2, 1), (1024, 1, 1, 0),  # stage3
        (256, 1, 1, 0), (256, 3, 1, 1), (1024, 1, 1, 0),
        (512, 1, 1, 0), (512, 3, 2, 1), (2048, 1, 1, 0),  # stage4
    ]
    # RPN: 3x3 conv + 1x1 for objectness (9 anchors * 2)
    rpn_specs = [(256, 3, 1, 1), (18, 1, 1, 0)]
    all_specs = conv_specs + rpn_specs
    return build_yolo_variant(all_specs, 36, name, input_h=800, input_w=800)


# ===========================================================================
# RetinaNet (simplified)
# ===========================================================================

def build_retinanet(name="retinanet_r50"):
    """Simplified RetinaNet with ResNet-50-like backbone + FPN head."""
    conv_specs = [
        (64, 7, 2, 3), (64, 3, 2, 1),  # stem + pool
        (256, 3, 1, 1), (256, 3, 1, 1), (256, 3, 1, 1),  # stage1
        (512, 3, 2, 1), (512, 3, 1, 1), (512, 3, 1, 1),  # stage2
        (1024, 3, 2, 1), (1024, 3, 1, 1), (1024, 3, 1, 1),  # stage3
        (2048, 3, 2, 1), (2048, 3, 1, 1),  # stage4
        # FPN
        (256, 1, 1, 0), (256, 3, 1, 1),
        # Classification subnet
        (256, 3, 1, 1), (256, 3, 1, 1), (256, 3, 1, 1), (256, 3, 1, 1),
    ]
    return build_yolo_variant(conv_specs, 720, name, input_h=800, input_w=800)


# ===========================================================================
# FCOS
# ===========================================================================

def build_fcos(name="fcos"):
    """Simplified FCOS detection."""
    conv_specs = [
        (64, 7, 2, 3), (64, 3, 2, 1),
        (256, 3, 1, 1), (256, 3, 1, 1),
        (512, 3, 2, 1), (512, 3, 1, 1),
        (1024, 3, 2, 1), (1024, 3, 1, 1),
        (2048, 3, 2, 1), (2048, 3, 1, 1),
        # Head
        (256, 1, 1, 0), (256, 3, 1, 1), (256, 3, 1, 1),
    ]
    return build_yolo_variant(conv_specs, 80, name, input_h=800, input_w=800)


# ===========================================================================
# CenterNet
# ===========================================================================

def build_centernet(name="centernet"):
    """Simplified CenterNet."""
    conv_specs = [
        (64, 7, 2, 3), (64, 3, 2, 1),
        (128, 3, 1, 1), (128, 3, 1, 1),
        (256, 3, 2, 1), (256, 3, 1, 1),
        (512, 3, 2, 1), (512, 3, 1, 1),
        # Upsampling path (stride 1 convs, conceptual upsample)
        (256, 3, 1, 1), (128, 3, 1, 1),
    ]
    return build_yolo_variant(conv_specs, 80, name, input_h=512, input_w=512)


# ===========================================================================
# T5 (encoder-decoder transformer)
# ===========================================================================

def build_t5_variant(num_enc_layers, num_dec_layers, hidden, ffn_dim, seq_len, name):
    """Build T5-like encoder-decoder transformer."""
    b = MLIRBuilder()
    b.add_param("%input", f"tensor<{seq_len}x{hidden}xf16>")

    # Encoder weights
    for layer in range(num_enc_layers):
        sfx = f"e{layer}"
        b.add_param(f"%w_q{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_k{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_v{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_kt{sfx}", f"tensor<{hidden}x{seq_len}xf16>")
        b.add_param(f"%w_o{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_ff{sfx}_up", f"tensor<{hidden}x{ffn_dim}xf16>")
        b.add_param(f"%w_ff{sfx}_down", f"tensor<{ffn_dim}x{hidden}xf16>")

    # Decoder weights
    for layer in range(num_dec_layers):
        sfx = f"d{layer}"
        b.add_param(f"%w_q{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_k{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_v{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_kt{sfx}", f"tensor<{hidden}x{seq_len}xf16>")
        b.add_param(f"%w_o{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        # Cross-attention
        b.add_param(f"%w_xq{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_xk{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_xv{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_xkt{sfx}", f"tensor<{hidden}x{seq_len}xf16>")
        b.add_param(f"%w_xo{sfx}", f"tensor<{hidden}x{hidden}xf16>")
        b.add_param(f"%w_ff{sfx}_up", f"tensor<{hidden}x{ffn_dim}xf16>")
        b.add_param(f"%w_ff{sfx}_down", f"tensor<{ffn_dim}x{hidden}xf16>")

    b.constant_zero()
    b.emit_blank()

    cur_var = "%input"
    cur_type = f"tensor<{seq_len}x{hidden}xf16>"

    def _transformer_layer(b, cur_var, cur_type, seq_len, hidden, ffn_dim, sfx):
        residual_var = cur_var
        residual_type = cur_type

        q_var, q_type = b.matmul(cur_var, cur_type,
            f"%w_q{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)
        k_var, k_type = b.matmul(cur_var, cur_type,
            f"%w_k{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)
        v_var, v_type = b.matmul(cur_var, cur_type,
            f"%w_v{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)

        scores_var, scores_type = b.matmul(q_var, q_type,
            f"%w_kt{sfx}", f"tensor<{hidden}x{seq_len}xf16>",
            seq_len, hidden, seq_len)
        soft_var, soft_type = b.relu_2d(scores_var, scores_type, seq_len, seq_len)

        attn_var, attn_type = b.matmul(soft_var, soft_type, v_var, v_type,
            seq_len, seq_len, hidden)
        out_var, out_type = b.matmul(attn_var, attn_type,
            f"%w_o{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)

        res_var, res_type = b.add_2d(out_var, out_type,
            residual_var, residual_type, seq_len, hidden)

        ffn_res = res_var
        ffn_res_type = res_type

        up_var, up_type = b.matmul(res_var, res_type,
            f"%w_ff{sfx}_up", f"tensor<{hidden}x{ffn_dim}xf16>",
            seq_len, hidden, ffn_dim)
        relu_var, relu_type = b.relu_2d(up_var, up_type, seq_len, ffn_dim)
        dn_var, dn_type = b.matmul(relu_var, relu_type,
            f"%w_ff{sfx}_down", f"tensor<{ffn_dim}x{hidden}xf16>",
            seq_len, ffn_dim, hidden)

        final_var, final_type = b.add_2d(dn_var, dn_type,
            ffn_res, ffn_res_type, seq_len, hidden)
        return final_var, final_type

    # Encoder
    b.emit_comment("=== Encoder ===")
    for layer in range(num_enc_layers):
        b.emit_comment(f"Encoder layer {layer}")
        cur_var, cur_type = _transformer_layer(
            b, cur_var, cur_type, seq_len, hidden, ffn_dim, f"e{layer}")
        b.emit_blank()

    enc_output_var = cur_var
    enc_output_type = cur_type

    # Decoder (uses encoder output for cross-attention)
    b.emit_comment("=== Decoder ===")
    # Start decoder with a fresh input (same shape)
    dec_var = cur_var  # In practice, decoder input would be different
    dec_type = cur_type

    for layer in range(num_dec_layers):
        sfx = f"d{layer}"
        b.emit_comment(f"Decoder layer {layer}")

        # Self-attention
        residual_var = dec_var
        residual_type = dec_type

        q_var, q_type = b.matmul(dec_var, dec_type,
            f"%w_q{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)
        k_var, k_type = b.matmul(dec_var, dec_type,
            f"%w_k{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)
        v_var, v_type = b.matmul(dec_var, dec_type,
            f"%w_v{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)
        scores_var, scores_type = b.matmul(q_var, q_type,
            f"%w_kt{sfx}", f"tensor<{hidden}x{seq_len}xf16>",
            seq_len, hidden, seq_len)
        soft_var, soft_type = b.relu_2d(scores_var, scores_type, seq_len, seq_len)
        attn_var, attn_type = b.matmul(soft_var, soft_type, v_var, v_type,
            seq_len, seq_len, hidden)
        out_var, out_type = b.matmul(attn_var, attn_type,
            f"%w_o{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)
        dec_var, dec_type = b.add_2d(out_var, out_type,
            residual_var, residual_type, seq_len, hidden)

        # Cross-attention (query from decoder, key/value from encoder)
        xres_var = dec_var
        xres_type = dec_type

        xq_var, xq_type = b.matmul(dec_var, dec_type,
            f"%w_xq{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)
        xk_var, xk_type = b.matmul(enc_output_var, enc_output_type,
            f"%w_xk{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)
        xv_var, xv_type = b.matmul(enc_output_var, enc_output_type,
            f"%w_xv{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)
        xscores_var, xscores_type = b.matmul(xq_var, xq_type,
            f"%w_xkt{sfx}", f"tensor<{hidden}x{seq_len}xf16>",
            seq_len, hidden, seq_len)
        xsoft_var, xsoft_type = b.relu_2d(xscores_var, xscores_type, seq_len, seq_len)
        xattn_var, xattn_type = b.matmul(xsoft_var, xsoft_type, xv_var, xv_type,
            seq_len, seq_len, hidden)
        xout_var, xout_type = b.matmul(xattn_var, xattn_type,
            f"%w_xo{sfx}", f"tensor<{hidden}x{hidden}xf16>",
            seq_len, hidden, hidden)
        dec_var, dec_type = b.add_2d(xout_var, xout_type,
            xres_var, xres_type, seq_len, hidden)

        # FFN
        ffn_res = dec_var
        ffn_res_type = dec_type
        up_var, up_type = b.matmul(dec_var, dec_type,
            f"%w_ff{sfx}_up", f"tensor<{hidden}x{ffn_dim}xf16>",
            seq_len, hidden, ffn_dim)
        relu_var, relu_type = b.relu_2d(up_var, up_type, seq_len, ffn_dim)
        dn_var, dn_type = b.matmul(relu_var, relu_type,
            f"%w_ff{sfx}_down", f"tensor<{ffn_dim}x{hidden}xf16>",
            seq_len, ffn_dim, hidden)
        dec_var, dec_type = b.add_2d(dn_var, dn_type,
            ffn_res, ffn_res_type, seq_len, hidden)
        b.emit_blank()

    return b.build_function(name, dec_var, dec_type)


# ===========================================================================
# Model registry
# ===========================================================================

MODELS = {
    # ===== Classification: ResNet family =====
    "resnet18": lambda: build_resnet_variant([2, 2, 2, 2], 'basic'),
    "resnet34": lambda: build_resnet_variant([3, 4, 6, 3], 'basic'),
    "resnet50": lambda: build_resnet_variant([3, 4, 6, 3], 'bottleneck'),
    "resnet101": lambda: build_resnet_variant([3, 4, 23, 3], 'bottleneck'),
    "resnet152": lambda: build_resnet_variant([3, 8, 36, 3], 'bottleneck'),

    # ===== Classification: VGG family =====
    "vgg11": lambda: build_vgg_variant(
        [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)], "vgg11"),
    "vgg13": lambda: build_vgg_variant(
        [(2, 64), (2, 128), (2, 256), (2, 512), (2, 512)], "vgg13"),
    "vgg16": lambda: build_vgg_variant(
        [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)], "vgg16"),
    "vgg19": lambda: build_vgg_variant(
        [(2, 64), (2, 128), (4, 256), (4, 512), (4, 512)], "vgg19"),

    # ===== Classification: DenseNet family =====
    "densenet121": lambda: build_densenet_variant([6, 12, 24, 16], 32, "densenet121"),
    "densenet169": lambda: build_densenet_variant([6, 12, 32, 32], 32, "densenet169"),
    "densenet201": lambda: build_densenet_variant([6, 12, 48, 32], 32, "densenet201"),
    "densenet261": lambda: build_densenet_variant([6, 12, 64, 48], 32, "densenet261"),

    # ===== Classification: EfficientNet family =====
    "efficientnet_b0": lambda: build_efficientnet_variant(1.0, 1.0, 224, "efficientnet_b0"),
    "efficientnet_b1": lambda: build_efficientnet_variant(1.0, 1.1, 240, "efficientnet_b1"),
    "efficientnet_b2": lambda: build_efficientnet_variant(1.1, 1.2, 260, "efficientnet_b2"),
    "efficientnet_b3": lambda: build_efficientnet_variant(1.2, 1.4, 300, "efficientnet_b3"),
    "efficientnet_b4": lambda: build_efficientnet_variant(1.4, 1.8, 380, "efficientnet_b4"),
    "efficientnet_b5": lambda: build_efficientnet_variant(1.6, 2.2, 456, "efficientnet_b5"),
    "efficientnet_b6": lambda: build_efficientnet_variant(1.8, 2.6, 528, "efficientnet_b6"),
    "efficientnet_b7": lambda: build_efficientnet_variant(2.0, 3.1, 600, "efficientnet_b7"),

    # ===== Classification: MobileNet family =====
    "mobilenetv1": lambda: build_mobilenet_variant(
        [(32, 64, 1, 1, 1), (64, 128, 2, 1, 1), (128, 128, 1, 1, 1),
         (128, 256, 2, 1, 1), (256, 256, 1, 1, 1), (256, 512, 2, 1, 1),
         (512, 512, 1, 1, 5), (512, 1024, 2, 1, 1), (1024, 1024, 1, 1, 1)],
        "mobilenetv1"),
    "mobilenetv2": lambda: build_mobilenet_variant(
        [(32, 16, 1, 1, 1), (16, 24, 2, 6, 2), (24, 32, 2, 6, 3),
         (32, 64, 2, 6, 4), (64, 96, 1, 6, 3), (96, 160, 2, 6, 3),
         (160, 320, 1, 6, 1)],
        "mobilenetv2"),
    "mobilenetv3_small": lambda: build_mobilenet_variant(
        [(16, 16, 2, 1, 1), (16, 24, 2, 4, 1), (24, 24, 1, 3, 1),
         (24, 40, 2, 3, 2), (40, 48, 1, 3, 2), (48, 96, 2, 6, 3)],
        "mobilenetv3_small", stem_c=16, final_c=576),
    "mobilenetv3_large": lambda: build_mobilenet_variant(
        [(16, 16, 1, 1, 1), (16, 24, 2, 4, 1), (24, 24, 1, 3, 1),
         (24, 40, 2, 3, 2), (40, 40, 1, 3, 1), (40, 80, 2, 6, 1),
         (80, 80, 1, 2, 1), (80, 80, 1, 2, 1), (80, 112, 1, 6, 2),
         (112, 160, 2, 6, 2), (160, 160, 1, 6, 1)],
        "mobilenetv3_large", stem_c=16, final_c=960),

    # ===== Classification: ShuffleNet family =====
    "shufflenetv1": lambda: build_shufflenet_variant(
        [144, 288, 576], [4, 8, 4], "shufflenetv1"),
    "shufflenetv2": lambda: build_shufflenet_variant(
        [116, 232, 464], [4, 8, 4], "shufflenetv2"),

    # ===== Classification: SqueezeNet family =====
    "squeezenet1_0": lambda: build_squeezenet_variant(
        [(16, 64), (16, 64), (32, 128), (32, 128),
         (48, 192), (48, 192), (64, 256), (64, 256)],
        "squeezenet1_0"),
    "squeezenet1_1": lambda: build_squeezenet_variant(
        [(16, 64), (16, 64), (32, 128), (32, 128),
         (48, 192), (48, 192), (64, 256), (64, 256)],
        "squeezenet1_1"),

    # ===== Classification: Inception family =====
    "inception_v3": lambda: build_inception_v3("inception_v3"),
    "googlenet": lambda: build_googlenet("googlenet"),

    # ===== Classification: AlexNet =====
    "alexnet": lambda: build_alexnet("alexnet"),

    # ===== Classification: MNASNet family =====
    "mnasnet0_5": lambda: build_mnasnet_variant(0.5, "mnasnet0_5"),
    "mnasnet1_0": lambda: build_mnasnet_variant(1.0, "mnasnet1_0"),

    # ===== Classification: RegNet family =====
    "regnet_y_400mf": lambda: build_regnet_variant(
        [48, 104, 208, 440], [1, 3, 6, 6], "regnet_y_400mf"),
    "regnet_y_800mf": lambda: build_regnet_variant(
        [64, 128, 320, 768], [1, 3, 8, 2], "regnet_y_800mf"),
    "regnet_y_1_6gf": lambda: build_regnet_variant(
        [48, 120, 336, 888], [2, 6, 17, 2], "regnet_y_1_6gf"),
    "regnet_y_4gf": lambda: build_regnet_variant(
        [48, 104, 208, 440], [2, 6, 12, 2], "regnet_y_4gf"),

    # ===== Classification: ConvNeXt family =====
    "convnext_tiny": lambda: build_convnext_variant(
        [3, 3, 9, 3], [96, 192, 384, 768], "convnext_tiny"),
    "convnext_small": lambda: build_convnext_variant(
        [3, 3, 27, 3], [96, 192, 384, 768], "convnext_small"),
    "convnext_base": lambda: build_convnext_variant(
        [3, 3, 27, 3], [128, 256, 512, 1024], "convnext_base"),

    # ===== Classification: Wide ResNet family =====
    "wide_resnet50_2": lambda: build_wide_resnet_variant(
        [3, 4, 6, 3], 2, "wide_resnet50_2"),
    "wide_resnet101_2": lambda: build_wide_resnet_variant(
        [3, 4, 23, 3], 2, "wide_resnet101_2"),

    # ===== Detection =====
    "yolov3": lambda: build_yolo_variant(
        [(32, 3, 1, 1), (64, 3, 2, 1), (128, 3, 2, 1), (256, 3, 2, 1),
         (512, 3, 2, 1), (1024, 3, 2, 1),
         (512, 1, 1, 0), (1024, 3, 1, 1), (512, 1, 1, 0)],
        255, "yolov3"),
    "yolov3_tiny": lambda: build_yolo_variant(
        [(16, 3, 1, 1), (32, 3, 2, 1), (64, 3, 2, 1), (128, 3, 2, 1),
         (256, 3, 2, 1), (512, 3, 2, 1)],
        255, "yolov3_tiny"),
    "yolov5s": lambda: build_yolo_variant(
        [(32, 3, 1, 1), (64, 3, 2, 1), (64, 3, 1, 1),
         (128, 3, 2, 1), (128, 3, 1, 1), (128, 3, 1, 1),
         (256, 3, 2, 1), (256, 3, 1, 1), (256, 3, 1, 1),
         (512, 3, 2, 1), (512, 3, 1, 1)],
        255, "yolov5s"),
    "yolov8n": lambda: build_yolo_variant(
        [(16, 3, 1, 1), (32, 3, 2, 1), (32, 3, 1, 1),
         (64, 3, 2, 1), (64, 3, 1, 1),
         (128, 3, 2, 1), (128, 3, 1, 1),
         (256, 3, 2, 1), (256, 3, 1, 1)],
        255, "yolov8n"),
    "ssd_mobilenet": lambda: build_ssd_mobilenet(),
    "ssd_vgg16": lambda: build_ssd_vgg16(),
    "faster_rcnn_r50": lambda: build_faster_rcnn(),
    "retinanet_r50": lambda: build_retinanet(),
    "fcos": lambda: build_fcos(),
    "centernet": lambda: build_centernet(),

    # ===== Segmentation =====
    "fcn_resnet50": lambda: build_encoder_decoder_variant(
        [(64, 7, 2, 3), (64, 3, 2, 1),  # stem
         (256, 3, 1, 1), (256, 3, 1, 1),  # stage1
         (512, 3, 2, 1), (512, 3, 1, 1),  # stage2
         (1024, 3, 2, 1), (1024, 3, 1, 1),  # stage3
         (2048, 3, 2, 1), (2048, 3, 1, 1)],  # stage4
        [(512, 3, 1, 1), (256, 3, 1, 1)],
        "fcn_resnet50", input_h=512, input_w=512, out_classes=21),
    "deeplabv3_resnet50": lambda: build_encoder_decoder_variant(
        [(64, 7, 2, 3), (64, 3, 2, 1),
         (256, 3, 1, 1), (256, 3, 1, 1),
         (512, 3, 2, 1), (512, 3, 1, 1),
         (1024, 3, 2, 1), (1024, 3, 1, 1),
         (2048, 3, 1, 1), (2048, 3, 1, 1),  # dilated
         # ASPP module (simplified)
         (256, 1, 1, 0), (256, 3, 1, 1), (256, 3, 1, 1)],
        [(256, 3, 1, 1), (256, 3, 1, 1)],
        "deeplabv3_resnet50", input_h=512, input_w=512, out_classes=21),
    "unet": lambda: build_encoder_decoder_variant(
        [(64, 3, 1, 1), (64, 3, 1, 1),
         (64, 3, 2, 1),  # pool
         (128, 3, 1, 1), (128, 3, 1, 1),
         (128, 3, 2, 1),  # pool
         (256, 3, 1, 1), (256, 3, 1, 1),
         (256, 3, 2, 1),  # pool
         (512, 3, 1, 1), (512, 3, 1, 1),
         (512, 3, 2, 1),  # pool
         (1024, 3, 1, 1), (1024, 3, 1, 1)],
        [(512, 3, 1, 1), (512, 3, 1, 1),
         (256, 3, 1, 1), (256, 3, 1, 1),
         (128, 3, 1, 1), (128, 3, 1, 1),
         (64, 3, 1, 1), (64, 3, 1, 1)],
        "unet", input_h=256, input_w=256, out_classes=2),
    "segnet": lambda: build_encoder_decoder_variant(
        [(64, 3, 1, 1), (64, 3, 2, 1),
         (128, 3, 1, 1), (128, 3, 2, 1),
         (256, 3, 1, 1), (256, 3, 1, 1), (256, 3, 2, 1),
         (512, 3, 1, 1), (512, 3, 1, 1), (512, 3, 2, 1),
         (512, 3, 1, 1), (512, 3, 1, 1)],
        [(512, 3, 1, 1), (512, 3, 1, 1),
         (256, 3, 1, 1), (256, 3, 1, 1),
         (128, 3, 1, 1), (64, 3, 1, 1)],
        "segnet", input_h=256, input_w=256, out_classes=12),
    "pspnet": lambda: build_encoder_decoder_variant(
        [(64, 7, 2, 3), (64, 3, 2, 1),
         (256, 3, 1, 1), (256, 3, 1, 1),
         (512, 3, 2, 1), (512, 3, 1, 1),
         (1024, 3, 2, 1), (1024, 3, 1, 1),
         (2048, 3, 1, 1)],
        # PSP module (simplified as 1x1 convs)
        [(512, 1, 1, 0), (512, 3, 1, 1), (256, 3, 1, 1)],
        "pspnet", input_h=473, input_w=473, out_classes=21),

    # ===== NLP/Transformer =====
    "bert_tiny": lambda: build_transformer_variant(2, 128, 512, 128, "bert_tiny"),
    "bert_base": lambda: build_transformer_variant(12, 768, 3072, 512, "bert_base"),
    "bert_large": lambda: build_transformer_variant(24, 1024, 4096, 512, "bert_large"),
    "gpt2_small": lambda: build_transformer_variant(12, 768, 3072, 1024, "gpt2_small"),
    "gpt2_medium": lambda: build_transformer_variant(24, 1024, 4096, 1024, "gpt2_medium"),
    "llama_tiny": lambda: build_transformer_variant(2, 256, 512, 128, "llama_tiny"),
    "llama_small": lambda: build_transformer_variant(6, 512, 2048, 256, "llama_small"),
    "t5_small": lambda: build_t5_variant(6, 6, 512, 2048, 128, "t5_small"),

    # ===== Vision Transformers =====
    "vit_b_16": lambda: build_vit_variant(16, 768, 12, 3072, 224, "vit_b_16"),
    "vit_l_16": lambda: build_vit_variant(16, 1024, 24, 4096, 224, "vit_l_16"),
    "vit_h_14": lambda: build_vit_variant(14, 1280, 32, 5120, 224, "vit_h_14"),
    "deit_tiny": lambda: build_vit_variant(16, 192, 12, 768, 224, "deit_tiny"),
    "deit_small": lambda: build_vit_variant(16, 384, 12, 1536, 224, "deit_small"),
    "deit_base": lambda: build_vit_variant(16, 768, 12, 3072, 224, "deit_base"),

    # ===== Swin Transformers =====
    "swin_tiny": lambda: build_swin_variant([2, 2, 6, 2], 96, "swin_tiny"),
    "swin_small": lambda: build_swin_variant([2, 2, 18, 2], 96, "swin_small"),
    "swin_base": lambda: build_swin_variant([2, 2, 18, 2], 128, "swin_base"),

    # ===== GAN/Generative =====
    "dcgan_generator": lambda: build_gan_generator(
        [(512, 3, 1, 1, True), (256, 3, 1, 1, True),
         (128, 3, 1, 1, True), (64, 3, 1, 1, True),
         (3, 3, 1, 1, False)],
        "dcgan_generator", input_c=100, input_h=4, input_w=4),
    "cyclegan_generator": lambda: build_gan_generator(
        [(64, 7, 1, 3, True),
         (128, 3, 1, 1, True), (256, 3, 1, 1, True),
         # Residual blocks (simplified as convs)
         (256, 3, 1, 1, True), (256, 3, 1, 1, True),
         (256, 3, 1, 1, True), (256, 3, 1, 1, True),
         (256, 3, 1, 1, True), (256, 3, 1, 1, True),
         (128, 3, 1, 1, True), (64, 3, 1, 1, True),
         (3, 7, 1, 3, False)],
        "cyclegan_generator", input_c=3, input_h=256, input_w=256),

    # ===== Super-resolution =====
    "srcnn": lambda: build_srcnn(),
    "espcn": lambda: build_espcn(),

    # ===== Autoencoders =====
    "autoencoder": lambda: build_autoencoder_variant(
        [512, 256, 128, 64], "autoencoder", input_dim=784),
    "vae": lambda: build_autoencoder_variant(
        [512, 256, 128, 32], "vae", input_dim=784),

    # ===== Additional classification models to reach 100+ =====
    "resnet_deep_20": lambda: build_resnet_variant([3, 3, 3, 3], 'basic'),
    "resnet_deep_44": lambda: build_resnet_variant([7, 7, 7, 7], 'basic'),
    "resnet_deep_56": lambda: build_resnet_variant([9, 9, 9, 9], 'basic'),
    "resnet_deep_110": lambda: build_resnet_variant([18, 18, 18, 18], 'basic'),

    # More EfficientNet-like variants with different scales
    "efficientnet_lite0": lambda: build_efficientnet_variant(1.0, 1.0, 224, "efficientnet_lite0"),
    "efficientnet_lite1": lambda: build_efficientnet_variant(1.0, 1.1, 240, "efficientnet_lite1"),
    "efficientnet_lite2": lambda: build_efficientnet_variant(1.1, 1.2, 260, "efficientnet_lite2"),
    "efficientnet_lite3": lambda: build_efficientnet_variant(1.2, 1.4, 280, "efficientnet_lite3"),
    "efficientnet_lite4": lambda: build_efficientnet_variant(1.4, 1.8, 300, "efficientnet_lite4"),

    # Additional MobileNet variants
    "mobilenetv2_0_5": lambda: build_mobilenet_variant(
        [(16, 8, 1, 1, 1), (8, 12, 2, 6, 2), (12, 16, 2, 6, 3),
         (16, 32, 2, 6, 4), (32, 48, 1, 6, 3), (48, 80, 2, 6, 3),
         (80, 160, 1, 6, 1)],
        "mobilenetv2_0_5", stem_c=16, final_c=640),
    "mobilenetv2_1_4": lambda: build_mobilenet_variant(
        [(48, 24, 1, 1, 1), (24, 32, 2, 6, 2), (32, 48, 2, 6, 3),
         (48, 88, 2, 6, 4), (88, 136, 1, 6, 3), (136, 224, 2, 6, 3),
         (224, 448, 1, 6, 1)],
        "mobilenetv2_1_4", stem_c=48, final_c=1792),

    # Additional RegNet variants
    "regnet_y_8gf": lambda: build_regnet_variant(
        [80, 240, 560, 1360], [2, 7, 17, 1], "regnet_y_8gf"),
    "regnet_y_16gf": lambda: build_regnet_variant(
        [96, 336, 672, 1344], [2, 4, 14, 1], "regnet_y_16gf"),

    # Additional ConvNeXt variant
    "convnext_large": lambda: build_convnext_variant(
        [3, 3, 27, 3], [192, 384, 768, 1536], "convnext_large"),

    # Additional Swin variant
    "swin_large": lambda: build_swin_variant([2, 2, 18, 2], 192, "swin_large"),

    # Additional transformer variants
    "gpt2_large": lambda: build_transformer_variant(36, 1280, 5120, 1024, "gpt2_large"),
    "bert_medium": lambda: build_transformer_variant(8, 512, 2048, 512, "bert_medium"),
    "bert_small": lambda: build_transformer_variant(4, 256, 1024, 512, "bert_small"),

    # Additional ViT variants
    "vit_s_16": lambda: build_vit_variant(16, 384, 12, 1536, 224, "vit_s_16"),
    "vit_ti_16": lambda: build_vit_variant(16, 192, 12, 768, 224, "vit_ti_16"),

    # Additional detection variants
    "yolov5m": lambda: build_yolo_variant(
        [(48, 3, 1, 1), (96, 3, 2, 1), (96, 3, 1, 1), (96, 3, 1, 1),
         (192, 3, 2, 1), (192, 3, 1, 1), (192, 3, 1, 1), (192, 3, 1, 1),
         (384, 3, 2, 1), (384, 3, 1, 1), (384, 3, 1, 1),
         (768, 3, 2, 1), (768, 3, 1, 1)],
        255, "yolov5m"),
    "yolov5l": lambda: build_yolo_variant(
        [(64, 3, 1, 1), (128, 3, 2, 1), (128, 3, 1, 1), (128, 3, 1, 1), (128, 3, 1, 1),
         (256, 3, 2, 1), (256, 3, 1, 1), (256, 3, 1, 1), (256, 3, 1, 1), (256, 3, 1, 1),
         (512, 3, 2, 1), (512, 3, 1, 1), (512, 3, 1, 1), (512, 3, 1, 1),
         (1024, 3, 2, 1), (1024, 3, 1, 1)],
        255, "yolov5l"),

    # Additional segmentation
    "deeplabv3plus_resnet50": lambda: build_encoder_decoder_variant(
        [(64, 7, 2, 3), (64, 3, 2, 1),
         (256, 3, 1, 1), (256, 3, 1, 1),
         (512, 3, 2, 1), (512, 3, 1, 1),
         (1024, 3, 2, 1), (1024, 3, 1, 1),
         (2048, 3, 1, 1),
         (256, 1, 1, 0), (256, 3, 1, 1)],
        [(256, 3, 1, 1), (256, 3, 1, 1), (128, 3, 1, 1)],
        "deeplabv3plus_resnet50", input_h=512, input_w=512, out_classes=21),

    # Additional autoencoder variants
    "autoencoder_deep": lambda: build_autoencoder_variant(
        [1024, 512, 256, 128, 64, 32], "autoencoder_deep", input_dim=1024),
    "autoencoder_wide": lambda: build_autoencoder_variant(
        [2048, 1024, 512], "autoencoder_wide", input_dim=2048),
    "vae_deep": lambda: build_autoencoder_variant(
        [1024, 512, 256, 128, 64], "vae_deep", input_dim=1024),

    # Additional GAN variants
    "dcgan_generator_large": lambda: build_gan_generator(
        [(1024, 3, 1, 1, True), (512, 3, 1, 1, True),
         (256, 3, 1, 1, True), (128, 3, 1, 1, True),
         (64, 3, 1, 1, True), (3, 3, 1, 1, False)],
        "dcgan_generator_large", input_c=200, input_h=4, input_w=4),

    # Super-resolution variants
    "srcnn_large": lambda: build_sequential_cnn(
        [(128, 9, 1, 4, True), (64, 1, 1, 0, True),
         (32, 1, 1, 0, True), (3, 5, 1, 2, False)],
        "srcnn_large", input_c=3, input_h=224, input_w=224,
        fc_classes=3, fc_as_1x1=True),
    "espcn_large": lambda: build_sequential_cnn(
        [(128, 5, 1, 2, True), (64, 3, 1, 1, True),
         (32, 3, 1, 1, True), (27, 3, 1, 1, False)],
        "espcn_large", input_c=3, input_h=224, input_w=224,
        fc_classes=27, fc_as_1x1=True),
}


# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate model MLIR files for NPU compiler testing")
    parser.add_argument("--model",
                        help="Model to generate")
    parser.add_argument("--list", action="store_true",
                        help="List all available models")
    args = parser.parse_args()

    if args.list:
        for name in sorted(MODELS.keys()):
            print(name)
        return

    if not args.model:
        parser.error("--model is required (or use --list)")

    if args.model not in MODELS:
        print(f"Error: unknown model '{args.model}'", file=sys.stderr)
        print(f"Available models: {', '.join(sorted(MODELS.keys()))}", file=sys.stderr)
        sys.exit(1)

    mlir_text = MODELS[args.model]()
    print(mlir_text)


if __name__ == "__main__":
    main()
