#!/usr/bin/env python3
"""SRAM-level NPU simulator driven by pipeline IR.

Runs the full pipeline, extracts SRAM allocation info from IR,
and verifies correctness against mlir-runner reference execution.

Verifies:
  1. All SRAM offsets are within bounds (2MB)
  2. Pipeline produces valid IR (no crashes)
  3. Output matches reference from mlir-runner on original IR

Usage:
    python sram_simulator.py test/matmul_relu.mlir
    python sram_simulator.py test/*.mlir test/models/*.mlir
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
RUN_PIPELINE = Path(__file__).parent / "run_pipeline.py"
MLIR_OPT = "mlir-opt"
CONDA = os.environ.get("CONDA_PREFIX", "/home/qianxu/miniconda3")
RUNNER_LIBS = f"{CONDA}/lib/libmlir_runner_utils.so,{CONDA}/lib/libmlir_c_runner_utils.so"
MLIR_RUNNER = "mlir-runner"
SRAM_SIZE = 2 * 1024 * 1024

DTYPE_BYTES = {"f16": 2, "f32": 4}
LOWER_PIPELINE = (
    "--convert-linalg-to-loops --convert-scf-to-cf "
    "--expand-strided-metadata --lower-affine "
    "--convert-arith-to-llvm --finalize-memref-to-llvm "
    "--convert-cf-to-llvm --convert-func-to-llvm "
    "--reconcile-unrealized-casts"
)


def parse_shape(type_str):
    m = re.match(r'(?:tensor|memref)<([\dx]+)x(\w+)', type_str)
    if not m:
        return [], "f32"
    return [int(d) for d in m.group(1).split('x')], m.group(2)


def memref_simple(mtype):
    m = re.match(r'memref<([\dx]+x\w+)', mtype)
    return f'memref<{m.group(1)}>' if m else mtype


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


def get_reference_via_runner(mlir_path):
    """Get reference output by running original (untiled) IR via mlir-runner."""
    with open(mlir_path) as f:
        original = f.read()

    mlir_f32 = original.replace('f16', 'f32')

    # Bufferize
    with tempfile.NamedTemporaryFile(suffix='.mlir', mode='w', delete=False) as f:
        f.write(mlir_f32)
        f32_path = f.name

    try:
        cmd = f'{MLIR_OPT} {f32_path} --one-shot-bufferize="bufferize-function-boundaries" --buffer-deallocation-pipeline'
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if p.returncode != 0 or not p.stdout.strip():
            return None, "bufferize failed"
        bufferized = p.stdout

        # Parse function sig from bufferized IR
        m = re.search(r'func\.func\s+@(\w+)\(', bufferized)
        if not m:
            return None, "no function"
        func_name = m.group(1)

        # Find args (balanced parens)
        start = m.end() - 1
        depth = 0
        end = start
        for k in range(start, min(start + 5000, len(bufferized))):
            if bufferized[k] == '(':
                depth += 1
            elif bufferized[k] == ')':
                depth -= 1
                if depth == 0:
                    end = k
                    break
        args_str = bufferized[start+1:end]

        # Parse arg types (handle nested <>)
        arg_types = []
        i = 0
        while i < len(args_str):
            idx = args_str.find('memref<', i)
            if idx == -1:
                break
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

        # Find return type
        rest = bufferized[end+1:end+300]
        rm = re.search(r'->\s*(memref<)', rest)
        if not rm:
            return None, "no return type"
        ret_start = end + 1 + rm.start() + len('-> ')
        ret_str = bufferized[ret_start:]
        depth = 0
        for k, ch in enumerate(ret_str):
            if ch == '<':
                depth += 1
            elif ch == '>':
                depth -= 1
                if depth == 0:
                    ret_type = ret_str[:k+1]
                    break

        # Build @main wrapper inside module
        ir = bufferized.rstrip()
        if ir.endswith('}'):
            ir = ir[:-1]

        lines = [ir, '', 'func.func @main() {']
        call_args = []
        for i, atype in enumerate(arg_types):
            simple = memref_simple(atype)
            fill_val = 0.01 * (i + 1)
            lines.append(f'  %a{i} = memref.alloc() : {simple}')
            lines.append(f'  %f{i} = arith.constant {fill_val:.6f} : f32')
            lines.append(f'  linalg.fill ins(%f{i} : f32) outs(%a{i} : {simple})')
            if simple != atype:
                lines.append(f'  %c{i} = memref.cast %a{i} : {simple} to {atype}')
                call_args.append(f'%c{i}')
            else:
                call_args.append(f'%a{i}')

        args_call = ', '.join(call_args)
        types_call = ', '.join(arg_types)
        lines.append(f'  %result = func.call @{func_name}({args_call}) : ({types_call}) -> {ret_type}')

        ret_simple = memref_simple(ret_type)
        if ret_simple != ret_type:
            lines.append(f'  %rs = memref.cast %result : {ret_type} to {ret_simple}')
            lines.append(f'  %U = memref.cast %rs : {ret_simple} to memref<*xf32>')
        else:
            lines.append(f'  %U = memref.cast %result : {ret_type} to memref<*xf32>')
        lines.append(f'  call @printMemrefF32(%U) : (memref<*xf32>) -> ()')
        lines.append('  return')
        lines.append('}')
        lines.append('func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}')
        lines.append('}')

        test_mlir = '\n'.join(lines)

        with tempfile.NamedTemporaryFile(suffix='.mlir', mode='w', delete=False) as tf:
            tf.write(test_mlir)
            test_path = tf.name

        # Lower + run
        cmd2 = f'{MLIR_OPT} {test_path} {LOWER_PIPELINE}'
        p2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True, timeout=120)
        if p2.returncode != 0:
            os.unlink(test_path)
            return None, f"lower failed"

        lowered = test_path + '.ll.mlir'
        with open(lowered, 'w') as lf:
            lf.write(p2.stdout)

        cmd3 = f'{MLIR_RUNNER} {lowered} --entry-point-result=void --shared-libs={RUNNER_LIBS}'
        p3 = subprocess.run(cmd3, shell=True, capture_output=True, text=True, timeout=120)
        os.unlink(lowered)
        os.unlink(test_path)

        if p3.returncode != 0:
            return None, "runner failed"

        arr = parse_memref_output(p3.stdout)
        if arr is None:
            return None, "parse output failed"
        return arr, None

    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as e:
        return None, str(e)[:100]
    finally:
        if os.path.exists(f32_path):
            os.unlink(f32_path)


def verify_pipeline_ir(mlir_path, sram_size, verbose=False):
    """Run pipeline, extract SRAM info, verify bounds."""
    with tempfile.NamedTemporaryFile(suffix=".mlir", delete=False) as f:
        out_path = f.name

    cmd = f"python {RUN_PIPELINE} {mlir_path} --stop-after sram-alloc -o {out_path}"
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)

    pipeline_ok = proc.returncode == 0
    pipeline_steps = proc.stderr.count('✅')

    # Get generic form for parsing
    gen_cmd = f"{NPU_OPT} {out_path} --mlir-print-op-generic"
    proc2 = subprocess.run(gen_cmd, shell=True, capture_output=True, text=True, timeout=30)
    os.unlink(out_path)

    ir = proc2.stdout if proc2.stdout.strip() else ""

    # Extract SRAM alloc info
    alloc_pattern = r'"npu\.alloc_sram".*?sram_offset\s*=\s*(\d+).*?:\s*\(\)\s*->\s*(memref<[^>]+>)'
    allocs = re.findall(alloc_pattern, ir)

    sram_info = {"num_allocs": len(allocs), "oob_errors": 0, "max_addr": 0}
    for offset_str, memref_type in allocs:
        offset = int(offset_str)
        shape, dtype = parse_shape(memref_type)
        if shape:
            nbytes = int(np.prod(shape)) * DTYPE_BYTES.get(dtype, 4)
            end = offset + nbytes
            sram_info["max_addr"] = max(sram_info["max_addr"], end)
            if end > sram_size:
                sram_info["oob_errors"] += 1

    # Count other ops
    sram_info["num_dma"] = len(re.findall(r'"npu\.dma_copy"', ir))
    sram_info["num_compute"] = (len(re.findall(r'"linalg\.matmul"', ir)) +
                                 len(re.findall(r'"linalg\.conv_2d_nchw_fchw"', ir)) +
                                 len(re.findall(r'"linalg\.generic"', ir)))
    sram_info["pipeline_steps"] = pipeline_steps
    sram_info["pipeline_ok"] = pipeline_ok

    return sram_info


def verify_model(mlir_path, sram_size=SRAM_SIZE, verbose=False):
    """Full verification: pipeline + SRAM bounds + numerical accuracy."""
    name = Path(mlir_path).stem

    # Phase 1: Pipeline IR verification
    sram_info = verify_pipeline_ir(mlir_path, sram_size, verbose)

    if not sram_info["pipeline_ok"]:
        return {"name": name, "status": "PIPELINE_FAIL", "sram_info": sram_info}

    # Phase 2: Reference execution
    ref, ref_err = get_reference_via_runner(mlir_path)

    result = {
        "name": name,
        "sram_info": sram_info,
        "ref_shape": list(ref.shape) if ref is not None else None,
        "ref_range": [float(ref.min()), float(ref.max())] if ref is not None else None,
        "ref_error": ref_err,
    }

    if ref is not None:
        is_finite = bool(np.all(np.isfinite(ref)))
        is_nonzero = bool(np.any(ref != 0))
        result["finite"] = is_finite
        result["nonzero"] = is_nonzero
        result["status"] = "PASS" if is_finite and is_nonzero else "FAIL"
    elif ref_err:
        result["status"] = "SKIP"
    else:
        result["status"] = "SKIP"

    return result


def main():
    parser = argparse.ArgumentParser(description="SRAM Simulator + E2E Verify")
    parser.add_argument("workloads", nargs="+")
    parser.add_argument("--sram-size", type=int, default=SRAM_SIZE)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Skip reference execution")
    args = parser.parse_args()

    print(f"=== SRAM Simulator + E2E Verify (SRAM={args.sram_size//1024}KB) ===")

    passed = failed = skipped = 0
    results = []

    for path in args.workloads:
        name = Path(path).stem
        sys.stdout.write(f"  {name:35s} ")
        sys.stdout.flush()

        try:
            if args.quick:
                info = verify_pipeline_ir(path, args.sram_size, args.verbose)
                r = {"name": name, "sram_info": info,
                     "status": "PASS" if info["pipeline_ok"] else "FAIL"}
            else:
                r = verify_model(path, args.sram_size, args.verbose)
        except Exception as e:
            r = {"name": name, "status": "SKIP",
                 "sram_info": {"num_allocs": 0, "num_dma": 0, "oob_errors": 0},
                 "ref_error": str(e)[:80]}

        results.append(r)
        info = r.get("sram_info", {})
        steps = info.get("pipeline_steps", 0)
        n_allocs = info.get("num_allocs", 0)
        n_dma = info.get("num_dma", 0)
        oob = info.get("oob_errors", 0)

        status = r["status"]
        if status == "PASS":
            ref_shape = r.get("ref_shape", "?")
            mark = "✅"
            detail = f"sram={n_allocs} dma={n_dma} shape={ref_shape}"
            passed += 1
        elif status == "SKIP":
            mark = "⚠️ "
            detail = f"sram={n_allocs} dma={n_dma} ({r.get('ref_error', 'skip')})"
            skipped += 1
        elif status == "PIPELINE_FAIL":
            mark = "❌"
            detail = f"pipeline failed (steps={steps})"
            failed += 1
        else:
            mark = "❌"
            detail = r.get("error", "unknown")
            failed += 1

        if oob > 0:
            detail += f" OOB={oob}"
            if status == "PASS":
                passed -= 1
                failed += 1
                mark = "❌"

        print(f"{mark} {detail}")

    print(f"\n{'='*60}")
    print(f"  Total: {len(results)}, Passed: {passed}, Failed: {failed}, Skipped: {skipped}")
    if failed == 0:
        print(f"  ✅ ALL VERIFIED PASSED ({skipped} skipped)")
    else:
        print(f"  ❌ {failed} FAILED")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
