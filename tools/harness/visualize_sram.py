#!/usr/bin/env python3
"""SRAM Storage Management Visualizer.

Runs npu-opt with the NPU pipeline and visualizes the SRAM layout
from the resulting IR. Shows allocated SRAM buffers, their offsets and
sizes, and any buffers spilled to DRAM.

Usage:
    python visualize_sram.py test/matmul_relu.mlir
    python visualize_sram.py test/conv2d_relu.mlir --sram-size 524288
    python visualize_sram.py test/mlp.mlir --verbose
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


# Find npu-opt binary relative to this script
NPU_OPT = Path(__file__).parent.parent.parent / "build" / "tools" / "npu-opt" / "npu-opt"

# Width of the ASCII bar area (in characters)
BAR_WIDTH = 40


@dataclass
class SRAMBuffer:
    """A buffer allocated in SRAM."""
    name: str
    offset: int
    shape: list[int]
    elem_type: str
    size_bytes: int
    layout: str = ""  # strided layout info if present
    func_name: str = ""  # enclosing function name
    loop_depth: int = 0  # nesting depth within scf.for loops


@dataclass
class DRAMBuffer:
    """A buffer spilled to DRAM (memref.alloc)."""
    name: str
    shape: list[int]
    elem_type: str
    size_bytes: int
    func_name: str = ""


@dataclass
class FuncInfo:
    """Information about a function in the IR."""
    name: str
    sram_buffers: list[SRAMBuffer] = field(default_factory=list)
    dram_buffers: list[DRAMBuffer] = field(default_factory=list)
    num_scf_for: int = 0


def element_size(elem_type: str) -> int:
    """Return the size of an element type in bytes."""
    sizes = {
        "f16": 2, "bf16": 2, "f32": 4, "f64": 8,
        "i8": 1, "i16": 2, "i32": 4, "i64": 8,
        "index": 8,
    }
    return sizes.get(elem_type, 2)  # default f16


def compute_memref_bytes(shape: list[int], elem_type: str) -> int:
    """Compute the size of a memref in bytes from shape and element type."""
    num_elems = 1
    for d in shape:
        if d > 0:
            num_elems *= d
    return num_elems * element_size(elem_type)


def parse_memref_type(type_str: str) -> tuple[list[int], str]:
    """Parse a memref type string like 'memref<16x256xf16>' or
    'memref<16x256xf16, strided<...>>'.
    Returns (shape, elem_type)."""
    # Match memref<dims x elem_type[, layout]>
    m = re.match(r"memref<([^,>]+?)(?:,\s*(.+))?>", type_str)
    if not m:
        return [], "f16"

    dims_and_type = m.group(1).strip()
    # Split on 'x' to get dimensions and final element type
    parts = dims_and_type.split("x")
    if len(parts) < 2:
        return [], parts[0] if parts else "f16"

    elem_type = parts[-1].strip()
    shape = []
    for p in parts[:-1]:
        p = p.strip()
        if p == "?":
            shape.append(-1)  # dynamic dim
        else:
            try:
                shape.append(int(p))
            except ValueError:
                shape.append(-1)

    return shape, elem_type


def run_npu_opt(input_file: str, npu_opt_path: str = None,
                extra_passes: list[str] = None) -> tuple[str, str, int]:
    """Run npu-opt and return (stdout, stderr, returncode).

    Runs the pipeline up through --npu-sram-allocation (skipping
    --npu-cost-evaluate which may crash on some inputs).
    """
    binary = Path(npu_opt_path) if npu_opt_path else NPU_OPT
    if not binary.exists():
        print(f"Error: npu-opt not found at {binary}", file=sys.stderr)
        sys.exit(1)

    passes = extra_passes or [
        "--npu-fusion",
        "--npu-outline-fused-groups",
        "--npu-spatial-tiling",
        "--npu-temporal-tiling",
        "--npu-bufferize",
        "--npu-sram-allocation",
    ]

    cmd = [str(binary), input_file] + passes
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout after 60 seconds", -1
    except Exception as e:
        return "", str(e), -1


def parse_ir(ir_text: str) -> list[FuncInfo]:
    """Parse the textual IR to extract SRAM and DRAM allocations per function."""
    funcs = []
    current_func = None
    loop_depth = 0

    for line in ir_text.split("\n"):
        stripped = line.strip()

        # Detect function boundaries
        func_match = re.match(
            r'func\.func\s+(?:private\s+)?@(\w+)\s*\(', stripped)
        if func_match:
            if current_func:
                funcs.append(current_func)
            current_func = FuncInfo(name=func_match.group(1))
            loop_depth = 0
            continue

        if current_func is None:
            continue

        # Track loop depth
        if re.search(r'\bscf\.for\b', stripped):
            current_func.num_scf_for += 1
            loop_depth += 1

        # Track closing braces (approximate loop end)
        # This is approximate since we're parsing textual IR, not an AST
        if stripped == "}":
            if loop_depth > 0:
                loop_depth -= 1

        # Parse npu.alloc_sram with offset
        # Pattern: %name = npu.alloc_sram offset <N> : memref<...>
        sram_match = re.match(
            r'(%\w+)\s*=\s*npu\.alloc_sram\s+offset\s+(\d+)\s*:\s*(memref<[^>]+>)',
            stripped)
        if sram_match:
            name = sram_match.group(1)
            offset = int(sram_match.group(2))
            type_str = sram_match.group(3)
            shape, elem_type = parse_memref_type(type_str)
            size_bytes = compute_memref_bytes(shape, elem_type)
            buf = SRAMBuffer(
                name=name, offset=offset, shape=shape, elem_type=elem_type,
                size_bytes=size_bytes, func_name=current_func.name,
                loop_depth=loop_depth)
            current_func.sram_buffers.append(buf)
            continue

        # Parse npu.alloc_sram without offset (before sram-allocation pass)
        sram_no_offset = re.match(
            r'(%\w+)\s*=\s*npu\.alloc_sram\s*:\s*(memref<[^>]+>)',
            stripped)
        if sram_no_offset:
            name = sram_no_offset.group(1)
            type_str = sram_no_offset.group(2)
            shape, elem_type = parse_memref_type(type_str)
            size_bytes = compute_memref_bytes(shape, elem_type)
            buf = SRAMBuffer(
                name=name, offset=-1, shape=shape, elem_type=elem_type,
                size_bytes=size_bytes, func_name=current_func.name,
                loop_depth=loop_depth)
            current_func.sram_buffers.append(buf)
            continue

        # Parse memref.alloc (DRAM spill buffers)
        # Pattern: %name = memref.alloc() ... : memref<...>
        dram_match = re.match(
            r'(%\w+)\s*=\s*memref\.alloc\(\)\s*(?:\{[^}]*\})?\s*:\s*(memref<[^>]+>)',
            stripped)
        if dram_match:
            name = dram_match.group(1)
            type_str = dram_match.group(2)
            shape, elem_type = parse_memref_type(type_str)
            size_bytes = compute_memref_bytes(shape, elem_type)
            buf = DRAMBuffer(
                name=name, shape=shape, elem_type=elem_type,
                size_bytes=size_bytes, func_name=current_func.name)
            current_func.dram_buffers.append(buf)
            continue

    # Don't forget the last function
    if current_func:
        funcs.append(current_func)

    return funcs


def format_size(bytes_val: int) -> str:
    """Format byte size in human-readable form."""
    if bytes_val >= 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.1f}MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.0f}KB"
    else:
        return f"{bytes_val}B"


def format_shape(shape: list[int], elem_type: str) -> str:
    """Format shape and element type like [32x256xf16]."""
    dims = "x".join(str(d) if d >= 0 else "?" for d in shape)
    return f"[{dims}x{elem_type}]"


def draw_bar(size_bytes: int, max_bytes: int) -> str:
    """Draw an ASCII bar proportional to size."""
    if max_bytes <= 0:
        return ""
    # At least 1 char for non-zero buffers
    width = max(1, int(BAR_WIDTH * size_bytes / max_bytes))
    return "\u2588" * width


def visualize_func(func: FuncInfo, sram_size: int, verbose: bool = False):
    """Print ASCII visualization for one function."""
    sram_bufs = func.sram_buffers
    dram_bufs = func.dram_buffers

    if not sram_bufs and not dram_bufs:
        return

    print(f"\n{'=' * 70}")
    print(f"  Function: @{func.name}")
    print(f"{'=' * 70}")

    # Deduplicate SRAM buffers by offset (same offset = reuse at different iterations)
    # Group by (offset, size) to identify unique allocation slots
    seen_offsets = {}
    unique_sram = []
    for buf in sram_bufs:
        key = (buf.offset, buf.size_bytes)
        if key not in seen_offsets:
            seen_offsets[key] = buf
            unique_sram.append(buf)
        elif verbose:
            # In verbose mode, show all allocations including duplicates
            unique_sram.append(buf)

    # Sort SRAM buffers by offset
    allocated = [b for b in unique_sram if b.offset >= 0]
    unallocated = [b for b in unique_sram if b.offset < 0]
    allocated.sort(key=lambda b: b.offset)

    # Compute total SRAM used
    if allocated:
        max_sram_used = max(b.offset + b.size_bytes for b in allocated)
    else:
        max_sram_used = 0

    total_sram_alloc = sum(b.size_bytes for b in allocated)
    total_dram_alloc = sum(b.size_bytes for b in dram_bufs)

    # Print SRAM layout
    print(f"\n  SRAM Layout ({format_size(sram_size)} budget):")
    print(f"  \u250c{'─' * 66}\u2510")

    if allocated:
        # Find max buffer size for bar scaling
        max_buf_size = max(b.size_bytes for b in allocated)

        for i, buf in enumerate(allocated):
            shape_str = format_shape(buf.shape, buf.elem_type)
            size_str = format_size(buf.size_bytes)
            bar = draw_bar(buf.size_bytes, max_buf_size)

            # Offset in hex
            offset_hex = f"0x{buf.offset:05X}"

            # Build the description
            desc = f"{offset_hex}: {buf.name} {shape_str} {size_str}"

            # Check if this is near the high end
            high_marker = ""
            buf_end = buf.offset + buf.size_bytes
            if buf_end > sram_size * 0.8:
                high_marker = " <- high end"

            line = f"  \u2502 {desc:<42s} {bar}{high_marker}"
            # Pad/truncate to fit box
            line = line[:68]
            line = f"{line:<68s} \u2502"
            print(line)

            # Show gap between buffers
            if i < len(allocated) - 1:
                next_buf = allocated[i + 1]
                gap = next_buf.offset - (buf.offset + buf.size_bytes)
                if gap > 0:
                    gap_str = f"  ... (gap: {format_size(gap)} free) ..."
                    gap_line = f"  \u2502 {gap_str:<64s} \u2502"
                    print(gap_line)
    else:
        print(f"  \u2502 {'(no SRAM allocations with offsets)':<64s} \u2502")

    print(f"  \u2514{'─' * 66}\u2518")

    # Summary line
    usage_pct = (max_sram_used / sram_size * 100) if sram_size > 0 else 0
    print(f"  Used: {format_size(max_sram_used)} / {format_size(sram_size)} ({usage_pct:.1f}%)")

    if dram_bufs:
        print(f"  Spilled to DRAM: {len(dram_bufs)} buffer(s) "
              f"(total {format_size(total_dram_alloc)})")
    else:
        print(f"  Spilled to DRAM: none")

    # Show unallocated SRAM buffers if any
    if unallocated and verbose:
        print(f"\n  Unallocated SRAM buffers (no offset assigned):")
        for buf in unallocated:
            shape_str = format_shape(buf.shape, buf.elem_type)
            size_str = format_size(buf.size_bytes)
            print(f"    {buf.name} {shape_str} {size_str}")

    # Show DRAM buffers detail
    if dram_bufs and verbose:
        print(f"\n  DRAM (spilled) buffers:")
        for buf in dram_bufs:
            shape_str = format_shape(buf.shape, buf.elem_type)
            size_str = format_size(buf.size_bytes)
            print(f"    {buf.name} {shape_str} {size_str}")

    # Timeline view: show buffers grouped by loop depth / iteration
    if func.num_scf_for > 0:
        print(f"\n  Tile loops: {func.num_scf_for} scf.for loop(s)")
        draw_timeline(allocated, sram_size)


def draw_timeline(buffers: list[SRAMBuffer], sram_size: int):
    """Draw a simple timeline view of SRAM allocations."""
    if not buffers:
        return

    # Group buffers by loop depth
    depths = {}
    for buf in buffers:
        d = buf.loop_depth
        if d not in depths:
            depths[d] = []
        depths[d].append(buf)

    print(f"\n  Address map (low -> high):")

    # Sort all by offset and show a compact timeline
    all_sorted = sorted(buffers, key=lambda b: b.offset)

    # Build a simple visual strip of the SRAM address space
    # Divide SRAM into columns
    NUM_COLS = 60
    strip = ["."] * NUM_COLS

    for buf in all_sorted:
        start_col = int(buf.offset / sram_size * NUM_COLS)
        end_col = int((buf.offset + buf.size_bytes) / sram_size * NUM_COLS)
        start_col = max(0, min(start_col, NUM_COLS - 1))
        end_col = max(start_col + 1, min(end_col, NUM_COLS))
        for c in range(start_col, end_col):
            strip[c] = "="

    strip_str = "".join(strip)
    print(f"  [{strip_str}]")
    print(f"   {'^ 0x00000':<30s}{'^ 0x{:05X}'.format(sram_size):>30s}")


def main():
    parser = argparse.ArgumentParser(
        description="SRAM Storage Management Visualizer for NPU compiler")
    parser.add_argument("input", help="Path to input .mlir file")
    parser.add_argument("--sram-size", type=int, default=262144,
                        help="SRAM budget in bytes (default: 262144 = 256KB)")
    parser.add_argument("--npu-opt", type=str, default=None,
                        help="Path to npu-opt binary")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed information")
    parser.add_argument("--passes", type=str, default=None,
                        help="Override pipeline passes (comma-separated)")
    args = parser.parse_args()

    input_path = args.input
    if not Path(input_path).exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Parse custom passes if provided
    extra_passes = None
    if args.passes:
        extra_passes = [f"--{p.strip().lstrip('-')}" for p in args.passes.split(",")]

    # Run npu-opt
    print(f"Running npu-opt on {input_path} ...")
    stdout, stderr, rc = run_npu_opt(
        input_path, npu_opt_path=args.npu_opt, extra_passes=extra_passes)

    if rc != 0 and not stdout:
        print(f"Error: npu-opt failed (exit code {rc})", file=sys.stderr)
        if stderr:
            # Filter out LLVM crash boilerplate
            err_lines = []
            for line in stderr.split("\n"):
                if "PLEASE submit a bug report" in line:
                    break
                err_lines.append(line)
            relevant = "\n".join(err_lines[-10:])
            print(f"stderr:\n{relevant}", file=sys.stderr)
        sys.exit(1)

    if not stdout.strip():
        print("Error: npu-opt produced no output", file=sys.stderr)
        if stderr:
            print(f"stderr (last 5 lines):", file=sys.stderr)
            for line in stderr.strip().split("\n")[-5:]:
                print(f"  {line}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"npu-opt returned code {rc}")

    # Parse the IR
    funcs = parse_ir(stdout)

    if not funcs:
        print("Warning: no functions found in IR output")
        sys.exit(0)

    # Print header
    input_name = Path(input_path).name
    print(f"\n{'#' * 70}")
    print(f"  SRAM Visualization: {input_name}")
    print(f"  SRAM budget: {format_size(args.sram_size)}")
    print(f"{'#' * 70}")

    # Visualize each function
    total_sram_bufs = 0
    total_dram_bufs = 0
    for func in funcs:
        if func.sram_buffers or func.dram_buffers:
            visualize_func(func, args.sram_size, verbose=args.verbose)
            total_sram_bufs += len(func.sram_buffers)
            total_dram_bufs += len(func.dram_buffers)

    # Overall summary
    print(f"\n{'─' * 70}")
    print(f"  Summary: {len(funcs)} function(s), "
          f"{total_sram_bufs} SRAM alloc(s), "
          f"{total_dram_bufs} DRAM spill(s)")
    print(f"{'─' * 70}")


if __name__ == "__main__":
    main()
