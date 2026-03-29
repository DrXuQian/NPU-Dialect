#!/usr/bin/env python3
"""Convert PyTorch models to NPU compiler MLIR input.

Route: PyTorch → ONNX → parse ONNX graph → generate linalg-on-tensor MLIR

Usage:
    python torch2npu.py --model resnet18 --batch 1 -o resnet18_torch.mlir
    python torch2npu.py --model mobilenet_v2 --batch 1
    python torch2npu.py --onnx model.onnx -o model.mlir
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnx
    from onnx import numpy_helper
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# ──────────────────────────────────────────────────────────────
# ONNX → MLIR converter
# ──────────────────────────────────────────────────────────────

DTYPE_MAP = {
    1: ("f32", 4),   # FLOAT
    10: ("f16", 2),  # FLOAT16
    7: ("i64", 8),   # INT64
    6: ("i32", 4),   # INT32
    9: ("i8", 1),    # INT8 (actually bool in ONNX, but treat as i8)
}


def onnx_shape_to_str(shape, dtype_str="f16"):
    dims = "x".join(str(d) for d in shape)
    return f"tensor<{dims}x{dtype_str}>"


def onnx_shape_to_memref(shape, dtype_str="f16"):
    dims = "x".join(str(d) for d in shape)
    return f"memref<{dims}x{dtype_str}>"


class ONNXToMLIR:
    """Convert ONNX model graph to linalg-on-tensor MLIR."""

    def __init__(self, model, dtype="f16"):
        self.model = model
        self.graph = model.graph
        self.dtype = dtype
        self.var_counter = 0
        self.value_map = {}  # onnx value name → MLIR SSA name
        self.lines = []
        self.func_args = []
        self.weights = {}  # initializer name → shape

    def next_var(self):
        name = f"%v{self.var_counter}"
        self.var_counter += 1
        return name

    def get_shape(self, name):
        """Get shape of an ONNX value."""
        if hasattr(self, '_shape_cache') and name in self._shape_cache:
            return self._shape_cache[name]
        return None

    def convert(self):
        """Convert the ONNX graph to MLIR string."""
        # Build shape cache from all value_info + inputs + outputs
        self._shape_cache = {}
        for vi in self.graph.value_info:
            shape = []
            for dim in vi.type.tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value > 0 else 1)
            if shape:
                self._shape_cache[vi.name] = shape
        for inp in self.graph.input:
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value > 0 else 1)
            if shape:
                self._shape_cache[inp.name] = shape
        for out in self.graph.output:
            if out.type.tensor_type.HasField("shape"):
                shape = []
                for dim in out.type.tensor_type.shape.dim:
                    shape.append(dim.dim_value if dim.dim_value > 0 else 1)
                if shape:
                    self._shape_cache[out.name] = shape
        for init in self.graph.initializer:
            self._shape_cache[init.name] = list(init.dims)

        # Collect initializers (weights)
        init_names = set()
        for init in self.graph.initializer:
            init_names.add(init.name)
            self.weights[init.name] = list(init.dims)

        # Determine function inputs (non-initializer graph inputs)
        graph_inputs = []
        weight_inputs = []
        for inp in self.graph.input:
            shape = self.get_shape(inp.name)
            if inp.name in init_names:
                weight_inputs.append((inp.name, shape))
            else:
                graph_inputs.append((inp.name, shape))
        # Also add initializers that are NOT in graph.input (ONNX opset 18 style)
        input_names = {inp.name for inp in self.graph.input}
        for init in self.graph.initializer:
            if init.name not in input_names:
                weight_inputs.append((init.name, list(init.dims)))

        # Determine function outputs
        outputs = []
        for out in self.graph.output:
            shape = self.get_shape(out.name)
            outputs.append((out.name, shape))

        # Build function signature
        all_inputs = graph_inputs + weight_inputs
        arg_strs = []
        for i, (name, shape) in enumerate(all_inputs):
            arg_name = f"%arg{i}"
            self.value_map[name] = arg_name
            ttype = onnx_shape_to_str(shape, self.dtype)
            arg_strs.append(f"{arg_name}: {ttype}")

        ret_shape = outputs[0][1] if outputs and outputs[0][1] else [1, 1000]
        ret_type = onnx_shape_to_str(ret_shape, self.dtype)

        func_name = self.graph.name or "model"
        func_name = func_name.replace("/", "_").replace(".", "_").replace("-", "_")

        self.lines.append(f'func.func @{func_name}({", ".join(arg_strs)}) -> {ret_type} {{')

        # Process nodes
        for node in self.graph.node:
            self._convert_node(node)

        # Return
        if outputs:
            ret_val = self.value_map.get(outputs[0][0], "%v0")
            self.lines.append(f"  return {ret_val} : {ret_type}")

        self.lines.append("}")

        return "\n".join(self.lines)

    def _convert_node(self, node):
        op = node.op_type

        if op == "Conv":
            self._conv(node)
        elif op == "MatMul" or op == "Gemm":
            self._matmul(node)
        elif op == "Relu":
            self._relu(node)
        elif op == "Add":
            self._add(node)
        elif op == "Sigmoid":
            self._sigmoid(node)
        elif op in ("GlobalAveragePool", "AveragePool", "MaxPool"):
            # Skip pooling — just pass through
            if node.input[0] in self.value_map:
                self.value_map[node.output[0]] = self.value_map[node.input[0]]
        elif op in ("Reshape", "Flatten", "Squeeze", "Unsqueeze", "Transpose"):
            # Pass through for now
            if node.input[0] in self.value_map:
                self.value_map[node.output[0]] = self.value_map[node.input[0]]
        elif op in ("BatchNormalization", "InstanceNorm", "LayerNorm"):
            # Skip normalization — pass through
            if node.input[0] in self.value_map:
                self.value_map[node.output[0]] = self.value_map[node.input[0]]
        elif op == "Pad":
            self._pad(node)
        elif op == "Concat":
            # Use first input as passthrough
            if node.input[0] in self.value_map:
                self.value_map[node.output[0]] = self.value_map[node.input[0]]
        else:
            # Unknown op — pass through with warning
            if node.input and node.input[0] in self.value_map:
                self.value_map[node.output[0]] = self.value_map[node.input[0]]

    def _conv(self, node):
        inp = self.value_map.get(node.input[0])
        weight = self.value_map.get(node.input[1])
        if not inp or not weight:
            return

        inp_shape = self.get_shape(node.input[0])
        w_shape = self.get_shape(node.input[1])
        out_shape = self.get_shape(node.output[0])

        if not inp_shape or not w_shape or not out_shape:
            return

        # Get attributes
        attrs = {a.name: a for a in node.attribute}
        pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]
        strides = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]

        # Emit padding if needed
        padded_inp = inp
        if any(p > 0 for p in pads):
            pad_h, pad_w = pads[0], pads[1]
            padded_shape = list(inp_shape)
            padded_shape[2] += 2 * pad_h
            padded_shape[3] += 2 * pad_w
            padded_type = onnx_shape_to_str(padded_shape, self.dtype)
            padded_var = self.next_var()
            cst_pad = self.next_var()
            self.lines.append(f'  {cst_pad} = arith.constant 0.0 : {self.dtype}')
            self.lines.append(f'  {padded_var} = tensor.pad {inp} low[0, 0, {pad_h}, {pad_w}] high[0, 0, {pad_h}, {pad_w}] {{')
            self.lines.append(f'  ^bb0(%a0: index, %a1: index, %a2: index, %a3: index):')
            self.lines.append(f'    tensor.yield {cst_pad} : {self.dtype}')
            self.lines.append(f'  }} : {onnx_shape_to_str(inp_shape, self.dtype)} to {padded_type}')
            padded_inp = padded_var

        # Emit fill + conv
        out_type = onnx_shape_to_str(out_shape, self.dtype)
        empty_var = self.next_var()
        fill_var = self.next_var()
        conv_var = self.next_var()
        cst_z = self.next_var()

        self.lines.append(f'  {cst_z} = arith.constant 0.0 : {self.dtype}')
        self.lines.append(f'  {empty_var} = tensor.empty() : {out_type}')
        self.lines.append(f'  {fill_var} = linalg.fill ins({cst_z} : {self.dtype}) outs({empty_var} : {out_type}) -> {out_type}')

        padded_type = onnx_shape_to_str(
            [inp_shape[0], inp_shape[1], inp_shape[2] + 2*pads[0], inp_shape[3] + 2*pads[1]],
            self.dtype) if any(p > 0 for p in pads) else onnx_shape_to_str(inp_shape, self.dtype)
        w_type = onnx_shape_to_str(w_shape, self.dtype)

        self.lines.append(
            f'  {conv_var} = linalg.conv_2d_nchw_fchw {{'
            f'dilations = dense<1> : tensor<2xi64>, '
            f'strides = dense<[{strides[0]}, {strides[1]}]> : tensor<2xi64>'
            f'}} ins({padded_inp}, {weight} : {padded_type}, {w_type}) '
            f'outs({fill_var} : {out_type}) -> {out_type}')

        self.value_map[node.output[0]] = conv_var

    def _matmul(self, node):
        a = self.value_map.get(node.input[0])
        b = self.value_map.get(node.input[1])
        if not a or not b:
            return

        a_shape = self.get_shape(node.input[0])
        b_shape = self.get_shape(node.input[1])
        out_shape = self.get_shape(node.output[0])

        if not a_shape or not b_shape or not out_shape:
            return

        out_type = onnx_shape_to_str(out_shape, self.dtype)
        empty_var = self.next_var()
        fill_var = self.next_var()
        mm_var = self.next_var()

        cst_z = self.next_var()
        self.lines.append(f'  {cst_z} = arith.constant 0.0 : {self.dtype}')
        self.lines.append(f'  {empty_var} = tensor.empty() : {out_type}')
        self.lines.append(f'  {fill_var} = linalg.fill ins({cst_z} : {self.dtype}) outs({empty_var} : {out_type}) -> {out_type}')

        a_type = onnx_shape_to_str(a_shape, self.dtype)
        b_type = onnx_shape_to_str(b_shape, self.dtype)

        if len(a_shape) == 2 and len(b_shape) == 2:
            self.lines.append(
                f'  {mm_var} = linalg.matmul '
                f'ins({a}, {b} : {a_type}, {b_type}) '
                f'outs({fill_var} : {out_type}) -> {out_type}')
        else:
            # batch matmul or higher-dim — use matmul on last 2 dims
            self.lines.append(
                f'  {mm_var} = linalg.matmul '
                f'ins({a}, {b} : {a_type}, {b_type}) '
                f'outs({fill_var} : {out_type}) -> {out_type}')

        self.value_map[node.output[0]] = mm_var

    def _relu(self, node):
        inp = self.value_map.get(node.input[0])
        if not inp:
            return
        shape = self.get_shape(node.input[0]) or self.get_shape(node.output[0])
        if not shape:
            return

        rank = len(shape)
        ttype = onnx_shape_to_str(shape, self.dtype)
        maps = ", ".join([f"affine_map<({', '.join(f'd{i}' for i in range(rank))}) -> ({', '.join(f'd{i}' for i in range(rank))})>"] * 2)
        iters = ", ".join(['"parallel"'] * rank)

        empty = self.next_var()
        relu = self.next_var()
        self.lines.append(f'  {empty} = tensor.empty() : {ttype}')
        self.lines.append(f'  {relu} = linalg.generic {{')
        self.lines.append(f'    indexing_maps = [{maps}],')
        self.lines.append(f'    iterator_types = [{iters}]')
        self.lines.append(f'  }} ins({inp} : {ttype}) outs({empty} : {ttype}) {{')
        self.lines.append(f'  ^bb0(%in: {self.dtype}, %out: {self.dtype}):')
        self.lines.append(f'    %zero = arith.constant 0.0 : {self.dtype}')
        self.lines.append(f'    %max = arith.maximumf %in, %zero : {self.dtype}')
        self.lines.append(f'    linalg.yield %max : {self.dtype}')
        self.lines.append(f'  }} -> {ttype}')
        self.value_map[node.output[0]] = relu

    def _add(self, node):
        a = self.value_map.get(node.input[0])
        b = self.value_map.get(node.input[1])
        if not a or not b:
            if a:
                self.value_map[node.output[0]] = a
            return

        shape = self.get_shape(node.output[0]) or self.get_shape(node.input[0])
        if not shape:
            return

        rank = len(shape)
        ttype = onnx_shape_to_str(shape, self.dtype)
        maps = ", ".join([f"affine_map<({', '.join(f'd{i}' for i in range(rank))}) -> ({', '.join(f'd{i}' for i in range(rank))})>"] * 3)
        iters = ", ".join(['"parallel"'] * rank)

        empty = self.next_var()
        add = self.next_var()
        self.lines.append(f'  {empty} = tensor.empty() : {ttype}')
        self.lines.append(f'  {add} = linalg.generic {{')
        self.lines.append(f'    indexing_maps = [{maps}],')
        self.lines.append(f'    iterator_types = [{iters}]')
        self.lines.append(f'  }} ins({a}, {b} : {ttype}, {ttype}) outs({empty} : {ttype}) {{')
        self.lines.append(f'  ^bb0(%x: {self.dtype}, %y: {self.dtype}, %out: {self.dtype}):')
        self.lines.append(f'    %sum = arith.addf %x, %y : {self.dtype}')
        self.lines.append(f'    linalg.yield %sum : {self.dtype}')
        self.lines.append(f'  }} -> {ttype}')
        self.value_map[node.output[0]] = add

    def _sigmoid(self, node):
        inp = self.value_map.get(node.input[0])
        if not inp:
            return
        shape = self.get_shape(node.input[0]) or self.get_shape(node.output[0])
        if not shape:
            return

        # Approximate sigmoid as relu (simplified for compiler testing)
        self._relu_like(node, inp, shape)

    def _relu_like(self, node, inp, shape):
        """Generic relu-like activation."""
        rank = len(shape)
        ttype = onnx_shape_to_str(shape, self.dtype)
        maps = ", ".join([f"affine_map<({', '.join(f'd{i}' for i in range(rank))}) -> ({', '.join(f'd{i}' for i in range(rank))})>"] * 2)
        iters = ", ".join(['"parallel"'] * rank)

        empty = self.next_var()
        out = self.next_var()
        self.lines.append(f'  {empty} = tensor.empty() : {ttype}')
        self.lines.append(f'  {out} = linalg.generic {{')
        self.lines.append(f'    indexing_maps = [{maps}],')
        self.lines.append(f'    iterator_types = [{iters}]')
        self.lines.append(f'  }} ins({inp} : {ttype}) outs({empty} : {ttype}) {{')
        self.lines.append(f'  ^bb0(%in: {self.dtype}, %out_: {self.dtype}):')
        self.lines.append(f'    %zero = arith.constant 0.0 : {self.dtype}')
        self.lines.append(f'    %max = arith.maximumf %in, %zero : {self.dtype}')
        self.lines.append(f'    linalg.yield %max : {self.dtype}')
        self.lines.append(f'  }} -> {ttype}')
        self.value_map[node.output[0]] = out

    def _pad(self, node):
        # Just pass through
        if node.input[0] in self.value_map:
            self.value_map[node.output[0]] = self.value_map[node.input[0]]


# ──────────────────────────────────────────────────────────────
# PyTorch → ONNX → MLIR
# ──────────────────────────────────────────────────────────────

TORCHVISION_MODELS = {
    "resnet18": "torchvision.models.resnet18",
    "resnet50": "torchvision.models.resnet50",
    "mobilenet_v2": "torchvision.models.mobilenet_v2",
    "vgg16": "torchvision.models.vgg16",
    "efficientnet_b0": "torchvision.models.efficientnet_b0",
    "densenet121": "torchvision.models.densenet121",
}


def torch_to_onnx(model_name, batch_size=1, opset=13):
    """Export a torchvision model to ONNX."""
    import torchvision.models as models

    model_fn = getattr(models, model_name, None)
    if model_fn is None:
        print(f"Unknown model: {model_name}", file=sys.stderr)
        sys.exit(1)

    model = model_fn(weights=None)
    model.eval()

    dummy = torch.randn(batch_size, 3, 224, 224)
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    torch.onnx.export(model, dummy, tmp.name, opset_version=opset,
                       input_names=["input"], output_names=["output"],
                       dynamic_axes=None)
    return tmp.name


def onnx_to_mlir(onnx_path, dtype="f16"):
    """Convert ONNX model to MLIR string."""
    model = onnx.load(onnx_path)
    # Run shape inference
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    converter = ONNXToMLIR(model, dtype=dtype)
    return converter.convert()


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch/ONNX models to NPU MLIR")
    parser.add_argument("--model", type=str, help="torchvision model name")
    parser.add_argument("--onnx", type=str, help="Path to ONNX model file")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--dtype", type=str, default="f16", choices=["f16", "f32"])
    parser.add_argument("-o", "--output", type=str, help="Output .mlir file")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        print("Available torchvision models:")
        for name in sorted(TORCHVISION_MODELS):
            print(f"  {name}")
        return

    onnx_path = args.onnx
    if args.model:
        if not HAS_TORCH:
            print("Error: torch not installed", file=sys.stderr)
            sys.exit(1)
        print(f"Exporting {args.model} to ONNX...", file=sys.stderr)
        onnx_path = torch_to_onnx(args.model, args.batch)
        print(f"ONNX saved to {onnx_path}", file=sys.stderr)

    if not onnx_path:
        parser.print_help()
        sys.exit(1)

    if not HAS_ONNX:
        print("Error: onnx not installed", file=sys.stderr)
        sys.exit(1)

    mlir = onnx_to_mlir(onnx_path, args.dtype)

    if args.output:
        with open(args.output, "w") as f:
            f.write(mlir)
        print(f"MLIR written to {args.output}", file=sys.stderr)
    else:
        print(mlir)

    # Cleanup temp ONNX file if we created it
    if args.model and onnx_path:
        import os
        os.unlink(onnx_path)


if __name__ == "__main__":
    main()
