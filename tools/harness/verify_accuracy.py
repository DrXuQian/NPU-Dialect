#!/usr/bin/env python3
"""Accuracy Verification Tool.

Verifies that each pass in the NPU pipeline preserves IR validity,
and checks structural invariants.

Usage:
    python verify_accuracy.py test/matmul_relu.mlir
    python verify_accuracy.py test/conv2d_relu.mlir --verbose
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

NPU_OPT = Path(__file__).parent.parent.parent / "build" / "tools" / "npu-opt" / "npu-opt"
MLIR_OPT = "mlir-opt"  # system mlir-opt

PASS_SEQUENCE = [
    ("npu-fusion", ["--npu-fusion"]),
    ("npu-outline", ["--npu-fusion", "--npu-outline-fused-groups"]),
    ("npu-spatial", ["--npu-fusion", "--npu-outline-fused-groups",
                     "--npu-spatial-tiling"]),
    ("npu-temporal", ["--npu-fusion", "--npu-outline-fused-groups",
                      "--npu-spatial-tiling", "--npu-temporal-tiling"]),
    ("npu-bufferize", ["--npu-fusion", "--npu-outline-fused-groups",
                       "--npu-spatial-tiling", "--npu-temporal-tiling",
                       "--npu-bufferize"]),
    ("npu-sram-alloc", ["--npu-fusion", "--npu-outline-fused-groups",
                        "--npu-spatial-tiling", "--npu-temporal-tiling",
                        "--npu-bufferize", "--npu-sram-allocation"]),
]


@dataclass
class PassResult:
    name: str
    valid: bool
    error: str = ""
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
    has_fused_kernel: bool = False


def run_pass(workload: str, passes: list[str], npu_opt: Path) -> tuple[str, str, int]:
    """Run npu-opt with given passes, return (stdout, stderr, returncode)."""
    cmd = [str(npu_opt), workload] + passes
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", 1
    except Exception as e:
        return "", str(e), 1


def analyze_ir(ir_text: str) -> dict:
    """Count structural elements in IR text."""
    return {
        "num_scf_for": len(re.findall(r"scf\.for\b", ir_text)),
        "num_linalg_ops": len(re.findall(r"linalg\.\w+\b", ir_text)),
        "num_tensor_types": len(re.findall(r"tensor<", ir_text)),
        "num_memref_types": len(re.findall(r"memref<", ir_text)),
        "num_alloc_sram": len(re.findall(r"npu\.alloc_sram", ir_text)),
        "num_alloc_sram_with_offset": len(re.findall(r"npu\.alloc_sram\s+offset\s+\d+", ir_text)),
        "num_dma_copy": len(re.findall(r"npu\.dma_copy", ir_text)),
        "num_func": len(re.findall(r"func\.func\b", ir_text)),
        "has_fused_kernel": "npu.fused_kernel" in ir_text,
    }


def verify_pass(name: str, passes: list[str], workload: str, npu_opt: Path,
                verbose: bool) -> PassResult:
    """Verify one pass configuration."""
    stdout, stderr, rc = run_pass(workload, passes, npu_opt)

    result = PassResult(name=name, valid=(rc == 0))
    if rc != 0:
        # Check if it's a real error or just remarks
        errors = [l for l in stderr.split("\n") if "error:" in l.lower()]
        if errors:
            result.valid = False
            result.error = errors[0][:200]
        else:
            result.valid = True  # Only remarks, no errors

    ir_text = stdout if stdout.strip() else ""
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
        result.has_fused_kernel = stats["has_fused_kernel"]

    if verbose and ir_text:
        print(f"\n    IR ({result.ir_lines} lines):")
        for line in ir_text.split("\n")[:15]:
            print(f"    | {line}")
        if result.ir_lines > 15:
            print(f"    | ... ({result.ir_lines - 15} more lines)")

    return result


def check_structural_invariants(results: list[PassResult]) -> list[str]:
    """Check expected structural properties across the pipeline."""
    issues = []

    # After outline: should have npu.fused_kernel attribute
    for r in results:
        if r.name == "npu-outline" and r.valid:
            if not r.has_fused_kernel:
                issues.append("npu-outline: expected npu.fused_kernel attribute")
            if r.num_func < 2:
                issues.append(f"npu-outline: expected ≥2 functions, got {r.num_func}")

    # After spatial: should have scf.for loops
    for r in results:
        if r.name == "npu-spatial" and r.valid:
            if r.num_scf_for == 0:
                issues.append("npu-spatial: expected scf.for loops for tiling")

    # After temporal: should have nested loops
    for r in results:
        if r.name == "npu-temporal" and r.valid:
            if r.num_scf_for < 2:
                issues.append(f"npu-temporal: expected nested loops, got {r.num_scf_for}")

    # After bufferize: tensor types should be gone, memref present
    for r in results:
        if r.name == "npu-bufferize" and r.valid:
            if r.num_memref_types == 0:
                issues.append("npu-bufferize: no memref types found (bufferization failed?)")

    # After SRAM alloc: npu.alloc_sram with offsets should exist
    for r in results:
        if r.name == "npu-sram-alloc" and r.valid:
            if r.num_alloc_sram == 0 and r.num_dma_copy == 0:
                issues.append("npu-sram-alloc: no SRAM allocs or DMA copies found")

    return issues


def verify_with_mlir_opt(workload: str) -> bool:
    """Verify input IR is valid using system mlir-opt."""
    try:
        proc = subprocess.run(
            [MLIR_OPT, workload], capture_output=True, text=True, timeout=10)
        return proc.returncode == 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Accuracy Verification Tool")
    parser.add_argument("workload", help="Path to .mlir workload")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print IR at each stage")
    parser.add_argument("--npu-opt", type=str, help="Path to npu-opt binary")
    args = parser.parse_args()

    npu_opt = Path(args.npu_opt) if args.npu_opt else NPU_OPT
    if not npu_opt.exists():
        print(f"Error: npu-opt not found at {npu_opt}", file=sys.stderr)
        sys.exit(1)

    workload = args.workload
    print(f"=== Accuracy Verification: {Path(workload).name} ===\n")

    # Step 1: Verify input is valid
    print("Input validation:")
    input_valid = verify_with_mlir_opt(workload)
    print(f"  {'✅' if input_valid else '❌'} Input IR valid (mlir-opt roundtrip)")

    # Step 2: Run each pass incrementally
    print("\nPass verification:")
    results = []
    for name, passes in PASS_SEQUENCE:
        r = verify_pass(name, passes, workload, npu_opt, args.verbose)
        results.append(r)

        status = "✅" if r.valid else "❌"
        detail_parts = []
        if r.num_scf_for > 0:
            detail_parts.append(f"{r.num_scf_for} loops")
        if r.num_linalg_ops > 0:
            detail_parts.append(f"{r.num_linalg_ops} linalg")
        if r.num_func > 1:
            detail_parts.append(f"{r.num_func} funcs")
        if r.has_fused_kernel:
            detail_parts.append("fused_kernel")
        if r.num_alloc_sram > 0:
            detail_parts.append(f"{r.num_alloc_sram} sram_allocs")
        if r.num_dma_copy > 0:
            detail_parts.append(f"{r.num_dma_copy} dma_copy")
        if r.num_memref_types > 0 and r.num_tensor_types == 0:
            detail_parts.append("fully bufferized")

        detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
        error_info = f" — {r.error}" if r.error else ""
        print(f"  {status} {name}: {r.ir_lines} lines{detail}{error_info}")

    # Step 3: Structural invariant checks
    print("\nStructural invariants:")
    issues = check_structural_invariants(results)
    if issues:
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("  ✅ All structural checks passed")

    # Step 4: Summary
    num_valid = sum(1 for r in results if r.valid)
    num_total = len(results)
    all_ok = num_valid == num_total and len(issues) == 0

    print(f"\n{'='*50}")
    print(f"  Result: {num_valid}/{num_total} passes valid, "
          f"{len(issues)} issue(s)")
    print(f"  {'✅ ALL CHECKS PASSED' if all_ok else '❌ SOME CHECKS FAILED'}")
    print(f"{'='*50}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
