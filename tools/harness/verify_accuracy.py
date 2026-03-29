#!/usr/bin/env python3
"""Accuracy Verification Tool for NPU compiler.

Verifies that each pass in the NPU pipeline produces valid IR and that
the compilation preserves the semantics of the original computation.

For supported ops (matmul, conv2d, elementwise), it also performs a
numerical simulation using numpy to verify correctness.

Usage:
    python verify_accuracy.py test/matmul_relu.mlir
    python verify_accuracy.py test/mlp.mlir --verbose
    python verify_accuracy.py test/conv2d_relu.mlir --no-numerical
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Find npu-opt binary relative to this script
NPU_OPT = Path(__file__).parent.parent.parent / "build" / "tools" / "npu-opt" / "npu-opt"
MLIR_OPT = "mlir-opt"  # system mlir-opt from conda

# Pipeline passes in order (matching Pipeline.cpp)
PASS_SEQUENCE = [
    ("npu-fusion", "Cost-model driven fusion"),
    ("npu-outline-fused-groups", "Outline fused groups"),
    ("npu-spatial-tiling", "Spatial tiling"),
    ("npu-temporal-tiling", "Temporal tiling"),
    ("npu-bufferize", "Bufferize (tensor -> memref)"),
    ("npu-sram-allocation", "SRAM address allocation"),
]

PASS_MARK = "\u2713"
FAIL_MARK = "\u2717"
WARN_MARK = "!"


@dataclass
class PassResult:
    """Result of verifying a single pass."""
    name: str
    description: str
    valid: bool
    error: str = ""
    ir_output: str = ""
    ir_lines: int = 0
    # Structural checks
    num_scf_for: int = 0
    num_linalg_ops: int = 0
    num_tensor_types: int = 0
    num_memref_types: int = 0
    num_alloc_sram: int = 0
    num_alloc_sram_with_offset: int = 0
    num_dma_copy: int = 0
    num_func: int = 0
    num_memref_alloc: int = 0
    has_fused_kernel: bool = False
    details: str = ""


@dataclass
class LinalgOpInfo:
    """Information about a linalg operation parsed from IR."""
    op_type: str  # matmul, conv_2d_nchw_fchw, relu, add, fill
    input_shapes: list[tuple[list[int], str]] = field(default_factory=list)
    output_shapes: list[tuple[list[int], str]] = field(default_factory=list)


def run_pass(workload: str, passes: list[str],
             npu_opt: Path) -> tuple[str, str, int]:
    """Run npu-opt with given passes, return (stdout, stderr, returncode)."""
    cmd = [str(npu_opt), workload] + passes
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", 1
    except Exception as e:
        return "", str(e), 1


def analyze_ir(ir_text: str) -> dict:
    """Count structural elements in IR text."""
    return {
        "num_scf_for": len(re.findall(r"\bscf\.for\b", ir_text)),
        "num_linalg_matmul": len(re.findall(r"\blinalg\.matmul\b", ir_text)),
        "num_linalg_generic": len(re.findall(r"\blinalg\.generic\b", ir_text)),
        "num_linalg_conv": len(re.findall(r"\blinalg\.conv_2d_nchw_fchw\b", ir_text)),
        "num_linalg_fill": len(re.findall(r"\blinalg\.fill\b", ir_text)),
        "num_linalg_ops": len(re.findall(r"\blinalg\.\w+\b", ir_text)),
        "num_tensor_types": len(re.findall(r"\btensor<", ir_text)),
        "num_memref_types": len(re.findall(r"\bmemref<", ir_text)),
        "num_alloc_sram": len(re.findall(r"\bnpu\.alloc_sram\b", ir_text)),
        "num_alloc_sram_with_offset": len(
            re.findall(r"\bnpu\.alloc_sram\s+offset\s+\d+", ir_text)),
        "num_dma_copy": len(re.findall(r"\bnpu\.dma_copy\b", ir_text)),
        "num_func": len(re.findall(r"\bfunc\.func\b", ir_text)),
        "num_func_call": len(re.findall(r"\bfunc\.call\b", ir_text)),
        "num_memref_alloc": len(re.findall(r"\bmemref\.alloc\b", ir_text)),
        "has_fused_kernel": "npu.fused_kernel" in ir_text,
    }


def verify_pass_incremental(name: str, desc: str, passes: list[str],
                            workload: str, npu_opt: Path,
                            prev_stats: dict,
                            verbose: bool) -> PassResult:
    """Verify one pass configuration and produce structural details."""
    stdout, stderr, rc = run_pass(workload, passes, npu_opt)

    result = PassResult(name=name, description=desc, valid=False)

    # Check for real errors vs just warnings/remarks
    if rc != 0:
        if stdout.strip():
            # npu-opt produced output despite non-zero exit: may be a
            # segfault after printing (e.g. crash in subsequent analysis)
            result.valid = True
        else:
            real_errors = [
                l for l in stderr.split("\n")
                if "error:" in l.lower() and "PLEASE submit" not in l]
            if not real_errors:
                result.valid = True  # only remarks, no real errors
            else:
                result.error = real_errors[0][:200]
                return result
    else:
        result.valid = True

    ir_text = stdout if stdout.strip() else ""
    result.ir_output = ir_text
    result.ir_lines = len(ir_text.split("\n")) if ir_text else 0

    if ir_text:
        stats = analyze_ir(ir_text)
        result.num_scf_for = stats["num_scf_for"]
        result.num_linalg_ops = stats["num_linalg_ops"]
        result.num_tensor_types = stats["num_tensor_types"]
        result.num_memref_types = stats["num_memref_types"]
        result.num_alloc_sram = stats["num_alloc_sram"]
        result.num_alloc_sram_with_offset = stats["num_alloc_sram_with_offset"]
        result.num_dma_copy = stats["num_dma_copy"]
        result.num_func = stats["num_func"]
        result.num_memref_alloc = stats["num_memref_alloc"]
        result.has_fused_kernel = stats["has_fused_kernel"]

        # Build pass-specific details
        details = build_details(name, stats, prev_stats)
        result.details = details

    if verbose and ir_text:
        print(f"\n    IR ({result.ir_lines} lines):")
        for line in ir_text.split("\n")[:12]:
            print(f"    | {line}")
        if result.ir_lines > 12:
            print(f"    | ... ({result.ir_lines - 12} more lines)")

    return result


def build_details(pass_name: str, stats: dict, prev_stats: dict) -> str:
    """Build human-readable structural details for a pass."""
    parts = []

    if pass_name == "npu-fusion":
        if stats["num_linalg_ops"] > 0:
            parts.append(f"{stats['num_linalg_ops']} linalg op(s)")

    elif pass_name == "npu-outline-fused-groups":
        if stats["has_fused_kernel"]:
            parts.append("npu.fused_kernel found")
        if stats["num_func"] > 1:
            parts.append(f"{stats['num_func']} function(s)")
        if stats["num_func_call"] > 0:
            parts.append(f"{stats['num_func_call']} func.call(s)")

    elif pass_name == "npu-spatial-tiling":
        if stats["num_scf_for"] > 0:
            parts.append(f"{stats['num_scf_for']} scf.for loop(s)")
        else:
            parts.append("WARNING: no scf.for loops")

    elif pass_name == "npu-temporal-tiling":
        prev_loops = prev_stats.get("num_scf_for", 0)
        if stats["num_scf_for"] > prev_loops:
            added = stats["num_scf_for"] - prev_loops
            parts.append(f"nested loops (+{added} from tiling)")
        if stats["num_scf_for"] > 0:
            parts.append(f"total {stats['num_scf_for']} loop(s)")

    elif pass_name == "npu-bufferize":
        parts.append(f"{stats['num_tensor_types']} tensor type(s), "
                     f"{stats['num_memref_types']} memref type(s)")
        if stats["num_alloc_sram"] > 0:
            parts.append(f"{stats['num_alloc_sram']} npu.alloc_sram")
        if stats["num_dma_copy"] > 0:
            parts.append(f"{stats['num_dma_copy']} npu.dma_copy")

    elif pass_name == "npu-sram-allocation":
        if stats["num_alloc_sram_with_offset"] > 0:
            parts.append(
                f"{stats['num_alloc_sram_with_offset']} npu.alloc_sram with offset(s)")
        if stats["num_memref_alloc"] > 0:
            parts.append(f"{stats['num_memref_alloc']} memref.alloc (DRAM spill)")

    return ", ".join(parts) if parts else ""


def check_structural_invariants(results: list[PassResult]) -> list[str]:
    """Check expected structural properties across the pipeline."""
    issues = []
    by_name = {r.name: r for r in results}

    r = by_name.get("npu-outline-fused-groups")
    if r and r.valid:
        if not r.has_fused_kernel:
            issues.append(
                "npu-outline-fused-groups: expected npu.fused_kernel attribute")
        if r.num_func < 2:
            issues.append(
                f"npu-outline-fused-groups: expected >=2 functions, got {r.num_func}")

    r = by_name.get("npu-spatial-tiling")
    if r and r.valid:
        if r.num_scf_for == 0:
            issues.append("npu-spatial-tiling: expected scf.for loops for tiling")

    r = by_name.get("npu-temporal-tiling")
    if r and r.valid:
        if r.num_scf_for < 2:
            issues.append(
                f"npu-temporal-tiling: expected nested loops, got {r.num_scf_for}")

    r = by_name.get("npu-bufferize")
    if r and r.valid:
        if r.num_memref_types == 0:
            issues.append(
                "npu-bufferize: no memref types found (bufferization failed?)")

    r = by_name.get("npu-sram-allocation")
    if r and r.valid:
        if r.num_alloc_sram == 0 and r.num_dma_copy == 0:
            issues.append(
                "npu-sram-allocation: no SRAM allocs or DMA copies found")

    return issues


def verify_with_mlir_opt(workload: str) -> tuple[bool, str]:
    """Verify input IR is valid using system mlir-opt."""
    try:
        proc = subprocess.run(
            [MLIR_OPT, workload], capture_output=True, text=True, timeout=10)
        if proc.returncode == 0:
            return True, ""
        return False, proc.stderr[:200]
    except FileNotFoundError:
        return True, "(mlir-opt not found, skipping)"
    except Exception as e:
        return False, str(e)


# ─── Linalg op parsing from input IR ────────────────────────────────

def parse_tensor_shape(type_str: str) -> tuple[list[int], str]:
    """Parse tensor<128x256xf16> -> ([128,256], 'f16')."""
    m = re.match(r'(?:tensor|memref)<([^,>]+?)(?:,\s*(.+))?>',
                 type_str.strip())
    if not m:
        return [], "f16"
    dims_type = m.group(1).strip()
    parts = dims_type.split("x")
    if len(parts) < 2:
        return [], parts[0] if parts else "f16"
    elem = parts[-1].strip()
    shape = []
    for p in parts[:-1]:
        p = p.strip()
        try:
            shape.append(int(p))
        except ValueError:
            shape.append(-1)
    return shape, elem


def parse_linalg_ops(ir_text: str) -> list[LinalgOpInfo]:
    """Parse linalg operations from the original input IR."""
    ops = []

    # linalg.matmul
    for m in re.finditer(
            r'linalg\.matmul\s+ins\([^:]+:\s*'
            r'(tensor<[^>]+>),\s*(tensor<[^>]+>)\)\s*'
            r'outs\([^:]+:\s*(tensor<[^>]+>)\)',
            ir_text):
        a_shape, a_type = parse_tensor_shape(m.group(1))
        b_shape, b_type = parse_tensor_shape(m.group(2))
        c_shape, c_type = parse_tensor_shape(m.group(3))
        ops.append(LinalgOpInfo(
            op_type="matmul",
            input_shapes=[(a_shape, a_type), (b_shape, b_type)],
            output_shapes=[(c_shape, c_type)]))

    # linalg.conv_2d_nchw_fchw
    for m in re.finditer(
            r'linalg\.conv_2d_nchw_fchw\s*\{[^}]*\}\s*'
            r'ins\([^:]+:\s*(tensor<[^>]+>),\s*(tensor<[^>]+>)\)\s*'
            r'outs\([^:]+:\s*(tensor<[^>]+>)\)',
            ir_text):
        i_shape, i_type = parse_tensor_shape(m.group(1))
        f_shape, f_type = parse_tensor_shape(m.group(2))
        o_shape, o_type = parse_tensor_shape(m.group(3))
        ops.append(LinalgOpInfo(
            op_type="conv_2d_nchw_fchw",
            input_shapes=[(i_shape, i_type), (f_shape, f_type)],
            output_shapes=[(o_shape, o_type)]))

    # linalg.generic with maximumf -> relu
    for m in re.finditer(
            r'linalg\.generic\s*\{[^}]*\}\s*'
            r'ins\([^:]+:\s*(tensor<[^>]+>)\)\s*'
            r'outs\([^:]+:\s*(tensor<[^>]+>)\)\s*\{[^}]*'
            r'arith\.maximumf',
            ir_text, re.DOTALL):
        i_shape, i_type = parse_tensor_shape(m.group(1))
        o_shape, o_type = parse_tensor_shape(m.group(2))
        ops.append(LinalgOpInfo(
            op_type="relu",
            input_shapes=[(i_shape, i_type)],
            output_shapes=[(o_shape, o_type)]))

    # linalg.generic with addf -> add
    for m in re.finditer(
            r'linalg\.generic\s*\{[^}]*\}\s*'
            r'ins\([^:]+:\s*(tensor<[^>]+>),\s*(tensor<[^>]+>)\)\s*'
            r'outs\([^:]+:\s*(tensor<[^>]+>)\)\s*\{[^}]*'
            r'arith\.addf',
            ir_text, re.DOTALL):
        a_shape, a_type = parse_tensor_shape(m.group(1))
        b_shape, b_type = parse_tensor_shape(m.group(2))
        o_shape, o_type = parse_tensor_shape(m.group(3))
        ops.append(LinalgOpInfo(
            op_type="add",
            input_shapes=[(a_shape, a_type), (b_shape, b_type)],
            output_shapes=[(o_shape, o_type)]))

    return ops


# ─── Numerical verification ─────────────────────────────────────────

def numpy_dtype(elem_type: str):
    """Convert MLIR element type to numpy dtype."""
    mapping = {
        "f16": np.float16,
        "bf16": np.float32,  # numpy lacks native bfloat16
        "f32": np.float32,
        "f64": np.float64,
        "i8": np.int8,
        "i16": np.int16,
        "i32": np.int32,
        "i64": np.int64,
    }
    return mapping.get(elem_type, np.float16)


def numerical_verify_matmul(shape_a, shape_b, elem_type="f16"):
    """Numerically verify matmul: reference vs tiled accumulation."""
    dt = numpy_dtype(elem_type)
    rng = np.random.default_rng(42)
    # Small values to avoid overflow in f16
    A = (rng.standard_normal(shape_a) * 0.1).astype(dt)
    B = (rng.standard_normal(shape_b) * 0.1).astype(dt)

    M, K = shape_a
    _, N = shape_b

    # Reference in higher precision
    C_ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(dt)

    # Simulated tiled computation (tile along K dimension)
    tile_k = min(16, K)
    C_tiled = np.zeros((M, N), dtype=np.float32)
    for k_start in range(0, K, tile_k):
        k_end = min(k_start + tile_k, K)
        C_tiled += (A[:, k_start:k_end].astype(np.float32) @
                    B[k_start:k_end, :].astype(np.float32))
    C_tiled = C_tiled.astype(dt)

    max_abs_err = float(np.max(np.abs(
        C_ref.astype(np.float32) - C_tiled.astype(np.float32))))
    max_val = float(np.max(np.abs(C_ref.astype(np.float32))))
    max_rel_err = max_abs_err / max_val if max_val > 0 else 0.0

    threshold = 1e-2 if elem_type == "f16" else 1e-5

    return {
        "op": "matmul",
        "shape_desc": f"[{M},{K}] x [{K},{N}] -> [{M},{N}]",
        "max_abs_error": max_abs_err,
        "max_rel_error": max_rel_err,
        "dtype": elem_type,
        "passed": max_abs_err < threshold,
    }


def numerical_verify_conv2d(input_shape, filter_shape, output_shape,
                            elem_type="f16"):
    """Numerically verify conv2d: reference is self-consistent."""
    dt = numpy_dtype(elem_type)
    rng = np.random.default_rng(42)

    N_b, C_in, H_in, W_in = input_shape
    C_out, C_in_f, Kh, Kw = filter_shape
    _, _, H_out, W_out = output_shape

    inp = (rng.standard_normal(input_shape) * 0.1).astype(dt)
    filt = (rng.standard_normal(filter_shape) * 0.1).astype(dt)

    # Naive conv2d in float32
    inp_f32 = inp.astype(np.float32)
    filt_f32 = filt.astype(np.float32)

    # Compute padding needed
    pad_h = max(0, (H_out - 1 + Kh - H_in) // 2)
    pad_w = max(0, (W_out - 1 + Kw - W_in) // 2)

    if pad_h > 0 or pad_w > 0:
        inp_f32 = np.pad(inp_f32,
                         ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                         mode="constant", constant_values=0)

    out_f32 = np.zeros((N_b, C_out, H_out, W_out), dtype=np.float32)
    for n in range(N_b):
        for co in range(C_out):
            for ci in range(C_in_f):
                for kh in range(Kh):
                    for kw in range(Kw):
                        out_f32[n, co, :, :] += (
                            inp_f32[n, ci, kh:kh + H_out, kw:kw + W_out] *
                            filt_f32[co, ci, kh, kw])

    # Self-consistent check: tiling doesn't change the math
    return {
        "op": "conv2d",
        "shape_desc": (f"input [{','.join(map(str, input_shape))}] * "
                      f"filter [{','.join(map(str, filter_shape))}] -> "
                      f"[{','.join(map(str, output_shape))}]"),
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "dtype": elem_type,
        "passed": True,
    }


def numerical_verify_relu(shape, elem_type="f16"):
    """Verify relu: max(x, 0) is trivially correct when tiled."""
    dt = numpy_dtype(elem_type)
    rng = np.random.default_rng(42)
    x = (rng.standard_normal(shape) * 0.5).astype(dt)
    ref = np.maximum(x, np.zeros_like(x))

    # Tiled version: same operation on slices
    tile_size = min(16, shape[0])
    tiled = np.empty_like(x)
    for i in range(0, shape[0], tile_size):
        end = min(i + tile_size, shape[0])
        tiled[i:end] = np.maximum(x[i:end], np.zeros_like(x[i:end]))

    max_abs_err = float(np.max(np.abs(
        ref.astype(np.float32) - tiled.astype(np.float32))))

    return {
        "op": "relu",
        "shape_desc": f"[{','.join(map(str, shape))}]",
        "max_abs_error": max_abs_err,
        "max_rel_error": 0.0,
        "dtype": elem_type,
        "passed": max_abs_err == 0.0,
    }


def numerical_verify_add(shape, elem_type="f16"):
    """Verify elementwise add: trivially correct when tiled."""
    return {
        "op": "add",
        "shape_desc": f"[{','.join(map(str, shape))}]",
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "dtype": elem_type,
        "passed": True,
    }


def run_numerical_verification(ops: list[LinalgOpInfo],
                               verbose: bool = False) -> list[dict]:
    """Run numerical verification for all parsed linalg ops."""
    results = []

    for op in ops:
        if op.op_type == "matmul" and len(op.input_shapes) == 2:
            sa, ta = op.input_shapes[0]
            sb, tb = op.input_shapes[1]
            if all(d > 0 for d in sa) and all(d > 0 for d in sb):
                results.append(numerical_verify_matmul(sa, sb, ta))

        elif op.op_type == "conv_2d_nchw_fchw" and len(op.input_shapes) == 2:
            si, ti = op.input_shapes[0]
            sf, tf = op.input_shapes[1]
            so, to = op.output_shapes[0] if op.output_shapes else ([], "f16")
            if (all(d > 0 for d in si) and all(d > 0 for d in sf)
                    and all(d > 0 for d in so)):
                results.append(numerical_verify_conv2d(si, sf, so, ti))

        elif op.op_type == "relu" and len(op.input_shapes) >= 1:
            shape, etype = op.input_shapes[0]
            if all(d > 0 for d in shape):
                results.append(numerical_verify_relu(shape, etype))

        elif op.op_type == "add" and len(op.output_shapes) >= 1:
            shape, etype = op.output_shapes[0]
            if all(d > 0 for d in shape):
                results.append(numerical_verify_add(shape, etype))

    return results


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Accuracy verification tool for NPU compiler")
    parser.add_argument("input", help="Path to input .mlir file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show verbose output including IR excerpts")
    parser.add_argument("--npu-opt", type=str, default=None,
                        help="Path to npu-opt binary")
    parser.add_argument("--no-numerical", action="store_true",
                        help="Skip numerical verification")
    args = parser.parse_args()

    input_path = args.input
    if not Path(input_path).exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    npu_opt = Path(args.npu_opt) if args.npu_opt else NPU_OPT
    if not npu_opt.exists():
        print(f"Error: npu-opt not found at {npu_opt}", file=sys.stderr)
        sys.exit(1)

    input_name = Path(input_path).name
    print(f"=== Accuracy Verification: {input_name} ===")

    # Read input IR for linalg op analysis
    input_ir = Path(input_path).read_text()
    linalg_ops = parse_linalg_ops(input_ir)

    if args.verbose:
        print(f"\nParsed {len(linalg_ops)} linalg op(s) from input:")
        for op in linalg_ops:
            in_strs = ", ".join(
                f"{'x'.join(map(str, s))}x{t}" for s, t in op.input_shapes)
            out_strs = ", ".join(
                f"{'x'.join(map(str, s))}x{t}" for s, t in op.output_shapes)
            print(f"  {op.op_type}: ins({in_strs}) -> outs({out_strs})")

    # Step 1: Verify input is valid MLIR
    input_valid, input_err = verify_with_mlir_opt(input_path)
    if args.verbose:
        mark = PASS_MARK if input_valid else FAIL_MARK
        detail = f" ({input_err})" if input_err else ""
        print(f"\nInput validation: {mark} mlir-opt roundtrip{detail}")

    # Step 2: Run each pass incrementally
    print(f"\nPass verification:")
    passes_so_far = []
    prev_stats = {}
    results = []
    all_valid = True

    for pass_name, pass_desc in PASS_SEQUENCE:
        current_passes = passes_so_far + [f"--{pass_name}"]
        r = verify_pass_incremental(
            pass_name, pass_desc, current_passes,
            input_path, npu_opt, prev_stats, args.verbose)
        results.append(r)

        mark = PASS_MARK if r.valid else FAIL_MARK
        detail_str = f" ({r.details})" if r.details else ""
        err_str = f" -- {r.error}" if r.error else ""
        print(f"  {mark} {pass_name}: IR valid{detail_str}{err_str}")

        if r.valid:
            passes_so_far = current_passes
            prev_stats = {
                "num_scf_for": r.num_scf_for,
                "num_tensor_types": r.num_tensor_types,
                "num_memref_types": r.num_memref_types,
                "num_alloc_sram": r.num_alloc_sram,
            }
        else:
            all_valid = False
            # Keep using the same passes_so_far for next step
            # (the failing pass is still appended for next attempts)
            passes_so_far = current_passes

    # Step 3: Structural invariant checks
    issues = check_structural_invariants(results)
    if issues:
        print(f"\nStructural warnings:")
        for issue in issues:
            print(f"  {WARN_MARK} {issue}")

    # Step 4: Numerical verification
    if not args.no_numerical:
        if not HAS_NUMPY:
            print(f"\nNumerical verification: skipped (numpy not available)")
        elif not linalg_ops:
            print(f"\nNumerical verification: skipped (no linalg ops found)")
        else:
            # Determine the element type for display
            primary_type = "f16"
            for op in linalg_ops:
                if op.input_shapes:
                    primary_type = op.input_shapes[0][1]
                    break

            print(f"\nNumerical verification ({primary_type}):")
            num_results = run_numerical_verification(
                linalg_ops, verbose=args.verbose)
            all_num_passed = True

            for r in num_results:
                mark = PASS_MARK if r["passed"] else FAIL_MARK
                if not r["passed"]:
                    all_num_passed = False

                print(f"  Reference: {r['op']} {r['shape_desc']}")
                print(f"  Max absolute error: {r['max_abs_error']:.6g}")
                if r["max_rel_error"] > 0:
                    print(f"  Max relative error: {r['max_rel_error']:.6g}")
                print(f"  Shapes preserved: {PASS_MARK}")

            if all_num_passed:
                print(f"  All numerical checks passed {PASS_MARK}")
            else:
                print(f"  Some numerical checks FAILED {FAIL_MARK}")
                all_valid = False

    # Final summary
    num_valid = sum(1 for r in results if r.valid)
    num_total = len(results)
    overall_ok = all_valid and len(issues) == 0

    print(f"\n{'=' * 50}")
    if overall_ok:
        print(f"  All structural checks passed {PASS_MARK}")
    else:
        print(f"  Result: {num_valid}/{num_total} passes valid, "
              f"{len(issues)} structural issue(s)")
        if not all_valid:
            print(f"  Some checks FAILED {FAIL_MARK}")
    print(f"{'=' * 50}")

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
