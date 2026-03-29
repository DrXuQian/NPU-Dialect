#!/usr/bin/env python3
"""Run the NPU compilation pipeline as sequential process invocations.

This avoids MLIR PassManager scheduling issues that cause crashes
when running all passes in a single npu-opt invocation.

Each pass runs in a separate npu-opt/mlir-opt process, piping IR between them.

Usage:
    python run_pipeline.py test/matmul_relu.mlir
    python run_pipeline.py test/matmul_relu.mlir -o output.mlir
    python run_pipeline.py test/resnet18_full.mlir --stop-after bufferize
"""

import argparse
import subprocess
import sys
import tempfile
import os
from pathlib import Path

NPU_OPT = Path(__file__).parent.parent.parent / "build" / "tools" / "npu-opt" / "npu-opt"
MLIR_OPT = "mlir-opt"

PIPELINE_STEPS = [
    ("fusion",       [str(NPU_OPT), "--npu-fusion"]),
    ("outline",      [str(NPU_OPT), "--npu-outline-fused-groups"]),
    ("spatial",      [str(NPU_OPT), "--npu-spatial-tiling"]),
    ("core-map",     [str(NPU_OPT), "--npu-core-mapping"]),
    ("temporal",     [str(NPU_OPT), "--npu-temporal-tiling"]),
    ("canonicalize", [MLIR_OPT, "--canonicalize", "--cse"]),
    ("bufferize",    [str(NPU_OPT), "--npu-bufferize"]),
    ("sram-alloc",   [str(NPU_OPT), "--npu-sram-allocation"]),
    ("cost-eval",    [str(NPU_OPT), "--npu-cost-evaluate"]),
]


def run_step(name, cmd, input_ir, verbose=False):
    """Run one pipeline step, return (output_ir, stderr, success)."""
    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(input_ir)
        tmp = f.name

    try:
        full_cmd = cmd + [tmp]
        proc = subprocess.run(full_cmd, capture_output=True, text=True, timeout=120)

        if verbose:
            if proc.stderr.strip():
                for line in proc.stderr.strip().split("\n")[:5]:
                    if "remark" in line or "COST_JSON" in line:
                        print(f"    {line.strip()}")

        if proc.returncode != 0 and not proc.stdout.strip():
            return None, proc.stderr, False

        return proc.stdout, proc.stderr, True
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT", False
    finally:
        os.unlink(tmp)


def main():
    parser = argparse.ArgumentParser(description="NPU Pipeline Runner")
    parser.add_argument("input", help="Input .mlir file")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("--stop-after", help="Stop after this step",
                        choices=[s[0] for s in PIPELINE_STEPS])
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.input) as f:
        ir = f.read()

    print(f"Pipeline: {Path(args.input).name}", file=sys.stderr)

    for name, cmd in PIPELINE_STEPS:
        ir_out, stderr, ok = run_step(name, cmd, ir, args.verbose)

        if not ok:
            # cost-eval is analysis-only, failure is non-fatal
            if name == "cost-eval":
                print(f"  ⚠️  {name:15s} → skipped (non-fatal)", file=sys.stderr)
                break
            print(f"  ❌ {name}: FAILED", file=sys.stderr)
            if stderr:
                for line in stderr.split("\n")[:3]:
                    if "error" in line.lower():
                        print(f"     {line.strip()}", file=sys.stderr)
            sys.exit(1)

        lines = len(ir_out.strip().split("\n")) if ir_out.strip() else 0
        # Count key IR elements
        allocs = ir_out.count("npu.alloc_sram")
        dma = ir_out.count("npu.dma_copy")
        memref = ir_out.count("memref<")
        extra = ""
        if allocs > 0: extra += f" sram={allocs}"
        if dma > 0: extra += f" dma={dma}"
        if memref > 0: extra += f" memref={memref}"

        print(f"  ✅ {name:15s} → {lines:5d} lines{extra}", file=sys.stderr)

        ir = ir_out

        if args.stop_after == name:
            break

    # Output final IR
    if args.output:
        with open(args.output, "w") as f:
            f.write(ir)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(ir)


if __name__ == "__main__":
    main()
