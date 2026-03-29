#!/usr/bin/env python3
"""NPU Compiler Optimization Harness.

Runs the npu-opt pipeline with different configurations, evaluates
cost using the roofline model, and reports the best configuration.

Usage:
    python optimize.py --workload test/matmul_relu.mlir
    python optimize.py --workload test/conv2d_relu.mlir --sweep
    python optimize.py --workload test/mlp.mlir --compare-strategies
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from itertools import product

# Find npu-opt binary
NPU_OPT = Path(__file__).parent.parent.parent / "build" / "tools" / "npu-opt" / "npu-opt"


@dataclass
class CostResult:
    """Parsed cost from npu-cost-evaluate JSON output."""
    name: str = ""
    flops: float = 0
    mem_bytes: float = 0
    arithmetic_intensity: float = 0
    time_sec: float = 0
    is_memory_bound: bool = True
    efficiency: float = 0
    matrix_cycles: int = 0
    dsp_cycles: int = 0
    peak_sram_bytes: int = 0
    num_tile_iterations: int = 1
    bottleneck: str = "memory"

    @property
    def total_cycles(self) -> int:
        return self.matrix_cycles + self.dsp_cycles


@dataclass
class RunResult:
    """Result of one pipeline run."""
    config: dict = field(default_factory=dict)
    kernels: list[CostResult] = field(default_factory=list)
    total_time_sec: float = 0
    success: bool = False
    error: str = ""

    @property
    def overall_cost(self) -> float:
        """Objective function: total wall time in seconds."""
        return sum(k.time_sec for k in self.kernels) if self.kernels else float("inf")


def run_pipeline(workload: str, passes: list[str], extra_args: list[str] = None) -> RunResult:
    """Run npu-opt with given passes and parse cost output."""
    cmd = [str(NPU_OPT), workload] + passes
    if extra_args:
        cmd += extra_args
    # Always add cost evaluate
    cmd.append("--npu-cost-evaluate")

    result = RunResult(config={"passes": passes, "extra_args": extra_args or []})
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        # Parse COST_JSON lines from stderr
        for line in proc.stderr.split("\n"):
            if "COST_JSON:" in line:
                json_str = line.split("COST_JSON:", 1)[1].strip()
                try:
                    data = json.loads(json_str)
                    kc = CostResult(
                        name=data.get("name", ""),
                        flops=data.get("flops", 0),
                        mem_bytes=data.get("mem_bytes", 0),
                        arithmetic_intensity=data.get("arithmetic_intensity", 0),
                        time_sec=data.get("time_sec", 0),
                        is_memory_bound=data.get("is_memory_bound", True),
                        efficiency=data.get("efficiency", 0),
                        matrix_cycles=data.get("matrix_cycles", 0),
                        dsp_cycles=data.get("dsp_cycles", 0),
                        peak_sram_bytes=data.get("peak_sram_bytes", 0),
                        num_tile_iterations=data.get("num_tile_iterations", 1),
                        bottleneck=data.get("bottleneck", "memory"),
                    )
                    result.kernels.append(kc)
                except json.JSONDecodeError:
                    pass

        result.success = proc.returncode == 0 and len(result.kernels) > 0
        if proc.returncode != 0:
            result.error = proc.stderr[:500]
    except subprocess.TimeoutExpired:
        result.error = "timeout"
    except Exception as e:
        result.error = str(e)

    return result


def sweep_configurations(workload: str) -> list[RunResult]:
    """Sweep over different pipeline configurations."""
    search_space = {
        "num_cores": [1, 2, 4, 8],
        "sram_size": [128 * 1024, 256 * 1024, 512 * 1024],
    }

    results = []
    for cores, sram in product(search_space["num_cores"], search_space["sram_size"]):
        passes = [
            "--npu-fusion",
            "--npu-outline-fused-groups",
            f"--npu-spatial-tiling=num-cores={cores}",
            f"--npu-temporal-tiling=sram-size={sram}",
        ]
        r = run_pipeline(workload, passes)
        r.config["num_cores"] = cores
        r.config["sram_size"] = sram
        results.append(r)
        status = f"cost={r.overall_cost:.2e}" if r.success else f"FAIL: {r.error[:50]}"
        print(f"  cores={cores} sram={sram//1024}KB → {status}")

    return results


def compare_strategies(workload: str) -> dict[str, RunResult]:
    """Compare different optimization strategies."""
    strategies = {
        # Basic combinations
        "no_optimization": [],
        "fusion_only": ["--npu-fusion", "--npu-outline-fused-groups"],
        "spatial_only_2c": ["--npu-spatial-tiling=num-cores=2"],
        "spatial_only_4c": ["--npu-spatial-tiling=num-cores=4"],
        "temporal_only_256k": ["--npu-temporal-tiling=sram-size=262144"],
        "temporal_only_512k": ["--npu-temporal-tiling=sram-size=524288"],
        "tiling_2c_256k": [
            "--npu-spatial-tiling=num-cores=2",
            "--npu-temporal-tiling=sram-size=262144",
        ],
        "tiling_4c_256k": [
            "--npu-spatial-tiling=num-cores=4",
            "--npu-temporal-tiling=sram-size=262144",
        ],
        # Full pipeline variants
        "fuse+tile_2c_256k": [
            "--npu-fusion", "--npu-outline-fused-groups",
            "--npu-spatial-tiling=num-cores=2",
            "--npu-temporal-tiling=sram-size=262144",
        ],
        "fuse+tile_4c_256k": [
            "--npu-fusion", "--npu-outline-fused-groups",
            "--npu-spatial-tiling=num-cores=4",
            "--npu-temporal-tiling=sram-size=262144",
        ],
        "fuse+tile_4c_512k": [
            "--npu-fusion", "--npu-outline-fused-groups",
            "--npu-spatial-tiling=num-cores=4",
            "--npu-temporal-tiling=sram-size=524288",
        ],
        "fuse+tile_8c_512k": [
            "--npu-fusion", "--npu-outline-fused-groups",
            "--npu-spatial-tiling=num-cores=8",
            "--npu-temporal-tiling=sram-size=524288",
        ],
    }

    results = {}
    for name, passes in strategies.items():
        r = run_pipeline(workload, passes)
        results[name] = r
        cost = f"{r.overall_cost:.2e}s" if r.success else "FAIL"
        nkernels = len(r.kernels)
        print(f"  {name:30s} → {cost} ({nkernels} kernel(s))")

    # Summary table
    valid = {n: r for n, r in results.items() if r.success and r.overall_cost > 0}
    if valid:
        baseline_cost = results.get("no_optimization")
        base_time = baseline_cost.overall_cost if baseline_cost and baseline_cost.success else None

        print(f"\n  {'Strategy':30s} {'Time(s)':>12s} {'Speedup':>8s} {'Bound':>10s} {'AI':>8s}")
        print(f"  {'-'*68}")
        for name in sorted(valid, key=lambda n: valid[n].overall_cost):
            r = valid[name]
            t = r.overall_cost
            speedup = f"{base_time/t:.2f}x" if base_time and t > 0 else "—"
            # Use first kernel's metrics
            k = r.kernels[0] if r.kernels else CostResult()
            print(f"  {name:30s} {t:12.4e} {speedup:>8s} {k.bottleneck:>10s} {k.arithmetic_intensity:8.1f}")

    return results


def batch_compare(workloads: list[str]) -> None:
    """Run strategy comparison across multiple workloads."""
    all_results = {}
    for wl in workloads:
        print(f"\n{'='*60}")
        print(f"  Workload: {wl}")
        print(f"{'='*60}")
        all_results[wl] = compare_strategies(wl)

    # Cross-workload summary
    print(f"\n{'='*60}")
    print(f"  Cross-Workload Summary")
    print(f"{'='*60}")
    strategies = set()
    for wl_results in all_results.values():
        strategies.update(wl_results.keys())

    header = f"{'Strategy':30s}"
    for wl in workloads:
        header += f" {Path(wl).stem:>14s}"
    print(f"  {header}")
    print(f"  {'-'*(30 + 15*len(workloads))}")

    for strat in sorted(strategies):
        row = f"  {strat:30s}"
        for wl in workloads:
            r = all_results.get(wl, {}).get(strat)
            if r and r.success and r.overall_cost > 0:
                row += f" {r.overall_cost:14.2e}"
            else:
                row += f" {'FAIL':>14s}"
        print(row)


def print_report(results: list[RunResult], title: str = "Sweep Results"):
    """Print a formatted report."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    valid = [r for r in results if r.success]
    if not valid:
        print("  No valid results.")
        return

    # Sort by cost
    # Filter out zero-cost results (no linalg ops found)
    valid = [r for r in valid if r.overall_cost > 0]
    if not valid:
        print("  No valid results with nonzero cost.")
        return

    valid.sort(key=lambda r: r.overall_cost)
    best = valid[0]
    worst = valid[-1]

    print(f"\n  Best:  cost={best.overall_cost:.4e}s  config={best.config}")
    print(f"  Worst: cost={worst.overall_cost:.4e}s  config={worst.config}")
    if worst.overall_cost > 0 and best.overall_cost > 0:
        speedup = worst.overall_cost / best.overall_cost
        print(f"  Speedup: {speedup:.2f}x")

    print(f"\n  {'Config':40s} {'Cost (s)':>12s} {'Bound':>10s} {'AI':>8s} {'Eff':>6s}")
    print(f"  {'-'*76}")
    for r in valid[:10]:
        for k in r.kernels:
            cfg = f"cores={r.config.get('num_cores','?')} sram={r.config.get('sram_size',0)//1024}KB"
            print(f"  {cfg:40s} {k.time_sec:12.4e} {k.bottleneck:>10s} {k.arithmetic_intensity:8.1f} {k.efficiency:6.1%}")


def main():
    parser = argparse.ArgumentParser(description="NPU Compiler Optimization Harness")
    parser.add_argument("--workload", required=True, nargs="+", help="Path(s) to .mlir workload(s)")
    parser.add_argument("--sweep", action="store_true", help="Sweep over configurations")
    parser.add_argument("--compare-strategies", action="store_true", help="Compare optimization strategies")
    parser.add_argument("--batch", action="store_true", help="Compare strategies across all workloads")
    parser.add_argument("--npu-opt", type=str, help="Path to npu-opt binary")
    args = parser.parse_args()

    global NPU_OPT
    if args.npu_opt:
        NPU_OPT = Path(args.npu_opt)

    if not NPU_OPT.exists():
        print(f"Error: npu-opt not found at {NPU_OPT}", file=sys.stderr)
        sys.exit(1)

    workloads = args.workload
    print(f"Workload(s): {workloads}")
    print(f"npu-opt:     {NPU_OPT}")

    if args.batch:
        batch_compare(workloads)
        return

    # Single workload mode
    workload = workloads[0]

    # Always show baseline cost
    print("\n--- Baseline (no optimization) ---")
    baseline = run_pipeline(workload, [])
    if baseline.success:
        for k in baseline.kernels:
            print(f"  {k.name}: {k.flops:.2e} FLOP, {k.mem_bytes:.2e} B, "
                  f"AI={k.arithmetic_intensity:.1f}, {k.bottleneck}-bound, "
                  f"eff={k.efficiency:.1%}, time={k.time_sec:.4e}s")
    else:
        print(f"  Failed: {baseline.error[:100]}")

    if args.compare_strategies:
        print("\n--- Strategy Comparison ---")
        compare_strategies(workload)

    if args.sweep:
        print("\n--- Configuration Sweep ---")
        results = sweep_configurations(workload)
        print_report(results, f"Sweep: {Path(workload).stem}")

    if not args.sweep and not args.compare_strategies:
        print("\n--- Standard Pipeline ---")
        standard = run_pipeline(workload, [
            "--npu-fusion", "--npu-outline-fused-groups",
            "--npu-spatial-tiling", "--npu-temporal-tiling",
        ])
        if standard.success:
            for k in standard.kernels:
                print(f"  {k.name}: {k.flops:.2e} FLOP, AI={k.arithmetic_intensity:.1f}, "
                      f"{k.bottleneck}-bound, eff={k.efficiency:.1%}, time={k.time_sec:.4e}s")
            if baseline.success:
                speedup = baseline.overall_cost / standard.overall_cost if standard.overall_cost > 0 else 0
                print(f"\n  Speedup vs baseline: {speedup:.2f}x")


if __name__ == "__main__":
    main()
