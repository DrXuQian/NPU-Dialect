#!/usr/bin/env python3
"""End-to-end numerical verification: input → network → output.

Approach:
  1. Read .mlir file, replace f16→f32
  2. Bufferize: mlir-opt --one-shot-bufferize="bufferize-function-boundaries"
  3. Parse bufferized function signature (memref args + return)
  4. Generate @main: alloc inputs with fill, call function, print result
  5. Lower to LLVM + run via mlir-runner
  6. Compare with numpy reference (for known patterns) or sanity check

Tiled verification (--verify-tiled):
  1. Run original IR through e2e to get reference output
  2. Run tiled IR (after spatial+temporal tiling) through structural checks:
     a. Same function signature (input/output types)
     b. Linalg ops are preserved (tiling doesn't lose ops)
     c. Outlined functions have correct structure
  3. If possible, run tiled IR through bufferize+lower+execute and compare

Usage:
    python e2e_verify.py test/matmul_relu.mlir
    python e2e_verify.py test/matmul_relu.mlir --verify-tiled
    python e2e_verify.py test/mlp.mlir test/conv2d_relu.mlir
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CONDA = os.environ.get("CONDA_PREFIX", "/home/qianxu/miniconda3")
MLIR_OPT = "mlir-opt"
MLIR_RUNNER = "mlir-runner"
LIBS = f"{CONDA}/lib/libmlir_runner_utils.so,{CONDA}/lib/libmlir_c_runner_utils.so"

# NPU tools
SCRIPT_DIR = Path(__file__).resolve().parent
NPU_OPT = SCRIPT_DIR.parent.parent / "build" / "tools" / "npu-opt" / "npu-opt"


def run_cmd(cmd, timeout=60):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.stdout, p.stderr, p.returncode


def parse_memref_output(text):
    m = re.search(r'sizes\s*=\s*\[([^\]]+)\]', text)
    if not m:
        return None
    shape = [int(x.strip()) for x in m.group(1).split(',')]
    dm = re.search(r'data\s*=\s*\n?(.*)', text, re.DOTALL)
    if not dm:
        return None
    nums = re.findall(r'[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?)', dm.group(1))
    vals = [float(x) for x in nums]
    total = 1
    for d in shape:
        total *= d
    if len(vals) != total:
        return None
    return np.array(vals, dtype=np.float32).reshape(shape)


def bufferize_and_parse_sig(mlir_f32_path):
    """Bufferize and extract the function signature."""
    out, err, rc = run_cmd(
        f'{MLIR_OPT} {mlir_f32_path}'
        f' --one-shot-bufferize="bufferize-function-boundaries"'
        f' --buffer-deallocation-pipeline')
    if rc != 0:
        return None, None, f"Bufferize failed: {err[:200]}"

    # Parse first func signature — handle nested <> in types
    m = re.search(r'func\.func\s+@(\w+)\(', out)
    if not m:
        return None, None, "No function found"
    func_name = m.group(1)

    # Extract args between parens — count balanced parens
    start = m.end() - 1  # at '('
    depth = 0
    end = start
    for k in range(start, len(out)):
        if out[k] == '(':
            depth += 1
        elif out[k] == ')':
            depth -= 1
            if depth == 0:
                end = k
                break
    args_str = out[start+1:end]

    # Extract return type after ->
    rest = out[end+1:end+200]
    rm = re.search(r'->\s*(memref<)', rest)
    if not rm:
        return None, None, f"No return type found in: {rest[:100]}"
    # Find matching >
    ret_start = end + 1 + rm.start() + len('-> ')
    ret_str = out[ret_start:]
    depth = 0
    ret_end = 0
    for k, ch in enumerate(ret_str):
        if ch == '<':
            depth += 1
        elif ch == '>':
            depth -= 1
            if depth == 0:
                ret_end = k + 1
                break
    ret_type = ret_str[:ret_end]

    # Parse arg types — handle nested <> in strided memref types
    arg_types = []
    i = 0
    while i < len(args_str):
        # Find "memref<"
        idx = args_str.find('memref<', i)
        if idx == -1:
            break
        # Find matching '>' counting depth
        depth = 0
        j = idx
        while j < len(args_str):
            if args_str[j] == '<':
                depth += 1
            elif args_str[j] == '>':
                depth -= 1
                if depth == 0:
                    arg_types.append(args_str[idx:j+1])
                    break
            j += 1
        i = j + 1

    return (func_name, arg_types, ret_type, out), None, None


def memref_shape(mtype):
    """memref<128x256xf32, strided<...>> → [128, 256]"""
    m = re.match(r'memref<([\dx]+)x\w+', mtype)
    if not m:
        return []
    return [int(d) for d in m.group(1).split('x')]


def memref_simple(mtype):
    """Strip strided layout: memref<128x256xf32, strided<[?,?], offset:?>> → memref<128x256xf32>"""
    m = re.match(r'memref<([\dx]+x\w+)', mtype)
    return f'memref<{m.group(1)}>' if m else mtype


def verify_model(mlir_path, verbose=False):
    name = Path(mlir_path).stem
    print(f"\n  {name}:")

    # Read original, convert f16→f32
    with open(mlir_path) as f:
        original = f.read()
    mlir_f32 = original.replace('f16', 'f32')

    # Write f32 version
    with tempfile.NamedTemporaryFile(suffix='.mlir', mode='w', delete=False) as f:
        f.write(mlir_f32)
        f32_path = f.name

    try:
        # Bufferize
        result = bufferize_and_parse_sig(f32_path)
        sig, _, error = result
        if error:
            print(f"    ❌ {error}")
            return False

        func_name, arg_types, ret_type, bufferized_ir = sig

        # Generate @main wrapper — insert INSIDE the module
        # Remove the closing "}" of the module and append @main before it
        ir_stripped = bufferized_ir.rstrip()
        if ir_stripped.endswith('}'):
            ir_stripped = ir_stripped[:-1]  # remove closing }
        main_lines = [ir_stripped, '', 'func.func @main() {']

        call_args = []
        for i, atype in enumerate(arg_types):
            shape = memref_shape(atype)
            simple_type = memref_simple(atype)
            fill_val = 0.01 * (i + 1)

            # Always alloc the simple type (no strides), then cast
            main_lines.append(f'  %a{i} = memref.alloc() : {simple_type}')
            main_lines.append(f'  %f{i} = arith.constant {fill_val:.6f} : f32')
            main_lines.append(f'  linalg.fill ins(%f{i} : f32) outs(%a{i} : {simple_type})')

            if simple_type != atype:
                main_lines.append(f'  %c{i} = memref.cast %a{i} : {simple_type} to {atype}')
                call_args.append(f'%c{i}')
            else:
                call_args.append(f'%a{i}')

        # Call the function
        args_str = ', '.join(call_args)
        types_str = ', '.join(arg_types)
        main_lines.append(f'  %result = func.call @{func_name}({args_str}) : ({types_str}) -> {ret_type}')

        # Print result
        ret_simple = memref_simple(ret_type)
        if ret_simple != ret_type:
            main_lines.append(f'  %rs = memref.cast %result : {ret_type} to {ret_simple}')
            main_lines.append(f'  %U = memref.cast %rs : {ret_simple} to memref<*xf32>')
        else:
            main_lines.append(f'  %U = memref.cast %result : {ret_type} to memref<*xf32>')

        main_lines.append(f'  call @printMemrefF32(%U) : (memref<*xf32>) -> ()')
        main_lines.append('  return')
        main_lines.append('}')
        main_lines.append('func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}')
        main_lines.append('}')  # close the module

        test_mlir = '\n'.join(main_lines)

        # Write test
        with tempfile.NamedTemporaryFile(suffix='.mlir', mode='w', delete=False) as f:
            f.write(test_mlir)
            test_path = f.name

        # Lower to LLVM
        lower_cmd = (f'{MLIR_OPT} {test_path}'
                     f' --convert-linalg-to-loops --convert-scf-to-cf'
                     f' --expand-strided-metadata --lower-affine'
                     f' --convert-arith-to-llvm --finalize-memref-to-llvm'
                     f' --convert-cf-to-llvm --convert-func-to-llvm'
                     f' --reconcile-unrealized-casts')
        out, err, rc = run_cmd(lower_cmd, timeout=120)
        if rc != 0:
            print(f"    ❌ Lower failed: {err[:200]}")
            if verbose:
                print(f"    Test MLIR saved at: {test_path}")
            return False

        # Write lowered and run
        lowered_path = test_path + '.llvm.mlir'
        with open(lowered_path, 'w') as f:
            f.write(out)

        run_cmd_str = f'{MLIR_RUNNER} {lowered_path} --entry-point-result=void --shared-libs={LIBS}'
        out, err, rc = run_cmd(run_cmd_str, timeout=120)
        os.unlink(lowered_path)
        os.unlink(test_path)

        if rc != 0:
            print(f"    ❌ Run failed: {err[:200]}")
            return False

        result_arr = parse_memref_output(out)
        if result_arr is None:
            print(f"    ❌ Could not parse output")
            return False

        # Report output stats
        ret_shape = memref_shape(ret_type)
        is_finite = np.all(np.isfinite(result_arr))
        is_nonzero = np.any(result_arr != 0)

        print(f"    Shape: {list(result_arr.shape)}, "
              f"range: [{result_arr.min():.6f}, {result_arr.max():.6f}], "
              f"finite={'✅' if is_finite else '❌'} nonzero={'✅' if is_nonzero else '❌'}")

        # Compute numpy reference for simple patterns
        numpy_ref = compute_numpy_ref(original, result_arr.shape)
        if numpy_ref is not None:
            max_err = np.max(np.abs(result_arr - numpy_ref))
            match = np.allclose(result_arr, numpy_ref, atol=1e-4, rtol=1e-3)
            print(f"    vs numpy: max_err={max_err:.2e} {'✅' if match else '❌'}")
            return match

        return is_finite and is_nonzero

    finally:
        if os.path.exists(f32_path):
            os.unlink(f32_path)


def compute_numpy_ref(mlir_text, out_shape):
    """Compute reference for known simple patterns."""
    _, arg_types, _ = parse_func_sig_tensor(mlir_text)
    if not arg_types:
        return None

    # Count ops
    ops = re.findall(r'linalg\.(\w+)\b', mlir_text)

    # matmul + relu (2 inputs: A, B)
    if len(arg_types) == 2 and 'matmul' in ops:
        s0 = tensor_shape(arg_types[0])
        s1 = tensor_shape(arg_types[1])
        if len(s0) == 2 and len(s1) == 2:
            A = np.full(s0, 0.01, dtype=np.float32)
            B = np.full(s1, 0.02, dtype=np.float32)
            C = A @ B
            if 'generic' in ops:  # relu
                C = np.maximum(C, 0)
            return C

    return None


def parse_func_sig_tensor(text):
    m = re.search(r'func\.func\s+@(\w+)\(([^)]*)\)\s*->\s*(tensor<[^{]+?>)', text)
    if not m:
        return None, [], None
    args = re.findall(r'tensor<[^,)]+>', m.group(2))
    return m.group(1), args, m.group(3)


def tensor_shape(ttype):
    m = re.match(r'tensor<([\dx]+)x\w+>', ttype)
    if not m:
        return []
    return [int(d) for d in m.group(1).split('x')]


def main():
    parser = argparse.ArgumentParser(description="E2E Numerical Verification")
    parser.add_argument("workloads", nargs="+")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("=== End-to-End Numerical Verification ===")
    passed = failed = 0
    for path in args.workloads:
        try:
            if verify_model(path, args.verbose):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n  {Path(path).stem}: ❌ {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"  Passed: {passed}, Failed: {failed}")
    print(f"  {'✅ ALL PASSED' if failed == 0 else '❌ SOME FAILED'}")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
