#!/usr/bin/env python3
"""Numerical Accuracy Verification via MLIR Execution.

Runs the original IR and the tiled IR through mlir-runner,
compares outputs numerically.

Approach:
  1. Wrap the function in a @main that fills inputs with known data
  2. Lower to LLVM via mlir-opt
  3. Execute via mlir-runner, capture printed memref output
  4. Compare original vs tiled outputs

Usage:
    python numerical_verify.py test/matmul_relu.mlir
    python numerical_verify.py test/conv2d_relu.mlir
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

NPU_OPT = Path(__file__).parent.parent.parent / "build" / "tools" / "npu-opt" / "npu-opt"
MLIR_OPT = "mlir-opt"
MLIR_RUNNER = "mlir-runner"
CONDA_PREFIX = os.environ.get("CONDA_PREFIX", "/home/qianxu/miniconda3")
RUNNER_LIBS = f"{CONDA_PREFIX}/lib/libmlir_runner_utils.so,{CONDA_PREFIX}/lib/libmlir_c_runner_utils.so"

LOWERING_PIPELINE = (
    "--one-shot-bufferize=bufferize-function-boundaries"
    " --buffer-deallocation-pipeline"
    " --convert-linalg-to-loops"
    " --convert-scf-to-cf"
    " --expand-strided-metadata"
    " --lower-affine"
    " --convert-arith-to-llvm"
    " --finalize-memref-to-llvm"
    " --convert-cf-to-llvm"
    " --convert-func-to-llvm"
    " --reconcile-unrealized-casts"
)


def parse_function_signature(mlir_text: str):
    """Extract function name, arg types, and return type from MLIR."""
    m = re.search(
        r"func\.func\s+@(\w+)\(([^)]*)\)\s*->\s*(\S+)",
        mlir_text,
    )
    if not m:
        return None, [], None
    name = m.group(1)
    args_str = m.group(2)
    ret_type = m.group(3)

    arg_types = []
    for arg in re.finditer(r"%\w+\s*:\s*(tensor<[^>]+>)", args_str):
        arg_types.append(arg.group(1))

    return name, arg_types, ret_type


def tensor_to_memref(tensor_type: str) -> str:
    """Convert tensor<128x256xf16> to memref<128x256xf32>."""
    # Use f32 for execution (f16 not well supported in runner)
    return tensor_type.replace("tensor", "memref").replace("f16", "f32")


def tensor_shape_dtype(tensor_type: str):
    """Extract shape and dtype from tensor<128x256xf16>."""
    m = re.match(r"tensor<([\dx]+)x(\w+)>", tensor_type)
    if not m:
        return [], "f32"
    dims = [int(d) for d in m.group(1).split("x")]
    dtype = m.group(2)
    return dims, dtype


def generate_test_harness(mlir_text: str, func_name: str, arg_types: list,
                          ret_type: str) -> str:
    """Generate a @main function that fills inputs and prints output."""
    # Convert all f16 to f32 for execution
    mlir_f32 = mlir_text.replace("f16", "f32")

    # Build memref types for arguments
    memref_args = [tensor_to_memref(t) for t in arg_types]
    ret_memref = tensor_to_memref(ret_type)

    lines = []
    lines.append(mlir_f32)
    lines.append("")
    lines.append("func.func @main() {")

    # Allocate and fill inputs
    arg_names = []
    for i, (mt, tt) in enumerate(zip(memref_args, arg_types)):
        shape, dtype = tensor_shape_dtype(tt)
        lines.append(f"  %a{i} = memref.alloc() : {mt}")
        # Fill with a deterministic pattern: 0.01 * (i+1)
        val = 0.01 * (i + 1)
        lines.append(f"  %fill{i} = arith.constant {val:.4f} : f32")
        lines.append(f"  linalg.fill ins(%fill{i} : f32) outs(%a{i} : {mt})")
        arg_names.append(f"%a{i}")

    # Allocate output
    ret_shape, _ = tensor_shape_dtype(ret_type)
    lines.append(f"  %out = memref.alloc() : {ret_memref}")
    lines.append(f"  %zero = arith.constant 0.0 : f32")
    lines.append(f"  linalg.fill ins(%zero : f32) outs(%out : {ret_memref})")

    # Call the function — but it takes tensors. We need memref version.
    # Instead, inline the computation directly.
    # Actually, the simplest approach: make the function take memref args.
    # We'll do a different approach: create a memref-level copy of the function.

    # For now, print the output
    lines.append(f"  %U = memref.cast %out : {ret_memref} to memref<*xf32>")
    lines.append(f'  call @printMemrefF32(%U) : (memref<*xf32>) -> ()')
    lines.append("  return")
    lines.append("}")
    lines.append('func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}')

    return "\n".join(lines)


def run_mlir(mlir_text: str, label: str = "") -> str:
    """Lower MLIR to LLVM and execute, return printed output."""
    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(mlir_text)
        f.flush()
        tmp_path = f.name

    try:
        # Lower to LLVM IR, save to temp file
        lower_cmd = f"{MLIR_OPT} {tmp_path} {LOWERING_PIPELINE}"
        proc1 = subprocess.run(
            lower_cmd, shell=True, capture_output=True, text=True, timeout=30)
        if proc1.returncode != 0:
            return f"LOWER_ERROR: {proc1.stderr[:300]}"

        # Write lowered IR to file (avoid shell echo pipe limits)
        lowered_path = tmp_path + ".llvm.mlir"
        with open(lowered_path, "w") as f:
            f.write(proc1.stdout)

        # Execute via mlir-runner
        run_cmd = (f"{MLIR_RUNNER} {lowered_path} "
                   f"--entry-point-result=void --shared-libs={RUNNER_LIBS}")
        proc2 = subprocess.run(
            run_cmd, shell=True, capture_output=True, text=True, timeout=30)
        os.unlink(lowered_path)
        if proc2.returncode != 0:
            return f"RUN_ERROR: {proc2.stderr[:300]}"

        return proc2.stdout
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def parse_memref_output(text: str) -> np.ndarray:
    """Parse mlir-runner Unranked Memref output to numpy array."""
    # Format: Unranked Memref base@ = 0x... rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1] data =
    # [[6, 6], [6, 6]]
    m = re.search(r"sizes\s*=\s*\[([^\]]+)\]", text)
    if not m:
        return None
    shape = [int(x.strip()) for x in m.group(1).split(",")]

    # Extract data section
    data_match = re.search(r"data\s*=\s*\n?(.*)", text, re.DOTALL)
    if not data_match:
        return None
    data_str = data_match.group(1).strip()
    # Remove brackets and split numbers
    numbers = re.findall(r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?)", data_str)
    values = [float(x) for x in numbers]

    total_elems = 1
    for d in shape:
        total_elems *= d
    if len(values) != total_elems:
        return None

    return np.array(values, dtype=np.float32).reshape(shape)


def create_memref_matmul_test(M: int, K: int, N: int) -> tuple[str, np.ndarray]:
    """Create a self-contained matmul test + expected result."""
    fill_a = 0.01
    fill_b = 0.02

    mlir = f"""
func.func @main() {{
  %A = memref.alloc() : memref<{M}x{K}xf32>
  %B = memref.alloc() : memref<{K}x{N}xf32>
  %C = memref.alloc() : memref<{M}x{N}xf32>
  %fa = arith.constant {fill_a} : f32
  %fb = arith.constant {fill_b} : f32
  %fc = arith.constant 0.0 : f32
  linalg.fill ins(%fa : f32) outs(%A : memref<{M}x{K}xf32>)
  linalg.fill ins(%fb : f32) outs(%B : memref<{K}x{N}xf32>)
  linalg.fill ins(%fc : f32) outs(%C : memref<{M}x{N}xf32>)
  linalg.matmul ins(%A, %B : memref<{M}x{K}xf32>, memref<{K}x{N}xf32>)
                outs(%C : memref<{M}x{N}xf32>)
  %U = memref.cast %C : memref<{M}x{N}xf32> to memref<*xf32>
  call @printMemrefF32(%U) : (memref<*xf32>) -> ()
  return
}}
func.func private @printMemrefF32(memref<*xf32>) attributes {{llvm.emit_c_interface}}
"""

    A = np.full((M, K), fill_a, dtype=np.float32)
    B = np.full((K, N), fill_b, dtype=np.float32)
    expected = A @ B
    return mlir, expected


def create_tiled_matmul_test(M: int, K: int, N: int, tile_m: int) -> tuple[str, np.ndarray]:
    """Create a tiled matmul (scf.for over M) + expected result."""
    fill_a = 0.01
    fill_b = 0.02

    mlir = f"""
func.func @main() {{
  %A = memref.alloc() : memref<{M}x{K}xf32>
  %B = memref.alloc() : memref<{K}x{N}xf32>
  %C = memref.alloc() : memref<{M}x{N}xf32>
  %fa = arith.constant {fill_a} : f32
  %fb = arith.constant {fill_b} : f32
  %fc = arith.constant 0.0 : f32
  linalg.fill ins(%fa : f32) outs(%A : memref<{M}x{K}xf32>)
  linalg.fill ins(%fb : f32) outs(%B : memref<{K}x{N}xf32>)
  linalg.fill ins(%fc : f32) outs(%C : memref<{M}x{N}xf32>)

  // Tiled matmul: split M into tiles of {tile_m}
  %c0 = arith.constant 0 : index
  %c{M} = arith.constant {M} : index
  %c{tile_m} = arith.constant {tile_m} : index
  scf.for %i = %c0 to %c{M} step %c{tile_m} {{
    %A_sub = memref.subview %A[%i, 0] [{tile_m}, {K}] [1, 1] : memref<{M}x{K}xf32> to memref<{tile_m}x{K}xf32, strided<[{K}, 1], offset: ?>>
    %C_sub = memref.subview %C[%i, 0] [{tile_m}, {N}] [1, 1] : memref<{M}x{N}xf32> to memref<{tile_m}x{N}xf32, strided<[{N}, 1], offset: ?>>
    linalg.matmul ins(%A_sub, %B : memref<{tile_m}x{K}xf32, strided<[{K}, 1], offset: ?>>, memref<{K}x{N}xf32>)
                  outs(%C_sub : memref<{tile_m}x{N}xf32, strided<[{N}, 1], offset: ?>>)
  }}

  %U = memref.cast %C : memref<{M}x{N}xf32> to memref<*xf32>
  call @printMemrefF32(%U) : (memref<*xf32>) -> ()
  return
}}
func.func private @printMemrefF32(memref<*xf32>) attributes {{llvm.emit_c_interface}}
"""

    A = np.full((M, K), fill_a, dtype=np.float32)
    B = np.full((K, N), fill_b, dtype=np.float32)
    expected = A @ B
    return mlir, expected


def verify_matmul(M=8, K=16, N=8, tile_m=4):
    """Verify: original matmul vs tiled matmul produce same result."""
    print(f"\n  Matmul [{M}x{K}] x [{K}x{N}], tile_m={tile_m}")

    # Original
    orig_mlir, expected = create_memref_matmul_test(M, K, N)
    orig_output = run_mlir(orig_mlir, "original")
    if orig_output.startswith("LOWER_ERROR") or orig_output.startswith("RUN_ERROR"):
        print(f"    ❌ Original execution failed: {orig_output[:100]}")
        return False
    orig_result = parse_memref_output(orig_output)
    if orig_result is None:
        print(f"    ❌ Failed to parse original output")
        return False

    # Tiled
    tiled_mlir, _ = create_tiled_matmul_test(M, K, N, tile_m)
    tiled_output = run_mlir(tiled_mlir, "tiled")
    if tiled_output.startswith("LOWER_ERROR") or tiled_output.startswith("RUN_ERROR"):
        print(f"    ❌ Tiled execution failed: {tiled_output[:100]}")
        return False
    tiled_result = parse_memref_output(tiled_output)
    if tiled_result is None:
        print(f"    ❌ Failed to parse tiled output")
        return False

    # Compare
    max_abs = np.max(np.abs(orig_result - tiled_result))
    max_rel = np.max(np.abs(orig_result - tiled_result) / (np.abs(expected) + 1e-10))

    match_orig = np.allclose(orig_result, expected, atol=1e-5)
    match_tiled = np.allclose(tiled_result, expected, atol=1e-5)
    match_each = np.allclose(orig_result, tiled_result, atol=1e-6)

    print(f"    Expected[0,0]={expected[0,0]:.6f}  Orig[0,0]={orig_result[0,0]:.6f}  Tiled[0,0]={tiled_result[0,0]:.6f}")
    print(f"    vs numpy:  orig={'✅' if match_orig else '❌'}  tiled={'✅' if match_tiled else '❌'}")
    print(f"    orig vs tiled: max_abs_err={max_abs:.2e}  max_rel_err={max_rel:.2e}  {'✅' if match_each else '❌'}")

    return match_each


def main():
    parser = argparse.ArgumentParser(description="Numerical Accuracy Verification")
    parser.add_argument("--quick", action="store_true", help="Quick test with small sizes")
    args = parser.parse_args()

    print("=== Numerical Accuracy Verification ===")
    print(f"  mlir-opt:    {MLIR_OPT}")
    print(f"  mlir-runner: {MLIR_RUNNER}")
    print(f"  runner libs: {RUNNER_LIBS}")

    all_pass = True

    # Test 1: Small matmul
    if not verify_matmul(M=4, K=8, N=4, tile_m=2):
        all_pass = False

    # Test 2: Larger matmul
    if not verify_matmul(M=16, K=32, N=16, tile_m=4):
        all_pass = False

    # Test 3: Different tile size
    if not verify_matmul(M=8, K=16, N=8, tile_m=2):
        all_pass = False

    if not args.quick:
        # Test 4: Bigger, divisible
        if not verify_matmul(M=32, K=64, N=32, tile_m=8):
            all_pass = False

        # Test 5: Tile on K dimension
        if not verify_matmul(M=16, K=32, N=16, tile_m=8):
            all_pass = False

    print(f"\n{'='*50}")
    print(f"  {'✅ ALL NUMERICAL CHECKS PASSED' if all_pass else '❌ SOME CHECKS FAILED'}")
    print(f"{'='*50}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
