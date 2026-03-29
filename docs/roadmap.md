# NPU Compiler Roadmap: Next Steps

## 1. PyTorch Frontend (torch-mlir integration)

### Approach
```
PyTorch model → torch-mlir → linalg-on-tensor IR → npu-opt pipeline → lowered IR
```

Install: `pip install torch-mlir` (nightly wheels available)

Usage:
```python
import torch, torch_mlir

model = torchvision.models.resnet18()
example_input = torch.randn(1, 3, 224, 224)
module = torch_mlir.compile(model, example_input,
                            output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
# Save to file
with open("resnet18.mlir", "w") as f:
    f.write(module.operation.get_asm())
# Then: npu-opt resnet18.mlir --npu-pipeline
```

### Tool to build
`tools/torch2npu.py`: wrapper that takes a torchvision model name and produces .mlir

Reference: https://github.com/llvm/torch-mlir

## 2. Backend Code Generation

### Reference implementations
- **MLIR-AIE** (AMD/Xilinx): NPU instruction generation from AIEX dialect
  - Converts high-level ops → binary instruction streams
  - https://deepwiki.com/Xilinx/mlir-aie/3.5-npu-instruction-generation
- **ARIES** (Cornell): MLIR-based flow for FPGA accelerator codegen
  - https://www.csl.cornell.edu/~zhiruz/pdfs/aries-fpga2025.pdf
- **Snitch/Occamy** (ETH): multi-level backend for accelerated micro-kernels
  - Progressive lowering from linalg → custom ISA

### Our approach
Lower from our NPU dialect to a target-specific format:
```
npu.compute { linalg.matmul } → matrix_engine.matmul(sram_addr_A, sram_addr_B, sram_addr_C, M, N, K)
npu.dma_copy %src to %dst    → dma_engine.transfer(dram_addr, sram_addr, size, direction)
npu.core_execute core 0 { }  → core_dispatch(core_id=0, instruction_buffer)
```

## 3. Runtime Design

```
┌─────────────────────────────────────┐
│          Host Application           │
├─────────────────────────────────────┤
│       NPU Runtime Library           │
│  ┌───────────┐ ┌──────────────────┐ │
│  │ Scheduler  │ │ Memory Manager   │ │
│  │ (dispatch  │ │ (SRAM pool,      │ │
│  │  kernels   │ │  DRAM alloc,     │ │
│  │  to cores) │ │  DMA queue)      │ │
│  └───────────┘ └──────────────────┘ │
├─────────────────────────────────────┤
│          Hardware Driver            │
│  DMA engine | Matrix unit | DSP     │
└─────────────────────────────────────┘
```

## 4. Simulator Options

| Simulator | Features | Language | NPU-relevant |
|-----------|----------|----------|--------------|
| **ONNXim** | Cycle-level multi-core, DMA, NoC (Booksim2), DRAM (Ramulator) | C++ | ✅ Best match |
| **mNPUsim** | Cycle-accurate, shared memory, multi-core | Python | ✅ Good |
| **NPUsim** | Full-model, value-aware | C++ | ✅ |
| **tiny-NPU** | DMA engine, SRAM, DDR shim | Verilog+C++ | ✅ HW-level |

Recommendation: **ONNXim** — models DMA/NoC contention, multi-core, cycle-level.
GitHub: https://github.com/PSAL-POSTECH/ONNXim

## 5. Double Buffering in Tiling

Current: synchronous DMA (load tile → compute → store → load next tile)

Target: pipelined (DMA_in[i+1] overlaps with compute[i])

```
Tile 0:  [DMA_in A,B] [Compute C=A*B] [DMA_out C]
Tile 1:                [DMA_in A,B]    [Compute C=A*B] [DMA_out C]
         ↑ ping buffer                 ↑ pong buffer
```

Implementation in TemporalTilingPass:
- Allocate 2x SRAM for each DMA'd tensor (ping + pong)
- Generate `npu.dma_start` + `npu.dma_wait` instead of `npu.dma_copy`
- Alternate between ping/pong addresses per tile iteration
- SRAM budget check: working_set * 2 ≤ sram_per_core

## 6. Weight Prefetch in Memory Allocation

Strategy: look ahead in the tile schedule, find the next tile's weights,
and if current SRAM has space, start loading them early.

```
SRAM snapshot at tile i:
  [input_A_i] [weight_B_i] [output_C_i] [... free space ...]

If free space >= weight_B_{i+1}:
  Start DMA for weight_B_{i+1} into free space
  When tile i finishes, weight_B_{i+1} is already in SRAM
```

Algorithm:
1. After SRAM allocation for tile i, compute free_bytes = sram_size - peak_usage
2. Look at tile i+1's weight tensors (identified by OperatorTilingSpec as Shared)
3. For each weight: if free_bytes >= weight_size, allocate at the furthest
   free address from current allocations (to minimize fragmentation)
4. Insert `npu.dma_start` for prefetch BEFORE the compute of tile i
5. Insert `npu.dma_wait` at the start of tile i+1 (before using the weight)

This is essentially a **software-managed cache prefetch** using SRAM.

## References

- [torch-mlir](https://github.com/llvm/torch-mlir) — PyTorch to MLIR
- [MLIR-AIE NPU instruction gen](https://deepwiki.com/Xilinx/mlir-aie/3.5-npu-instruction-generation)
- [ONNXim simulator](https://github.com/PSAL-POSTECH/ONNXim)
- [mNPUsim](https://github.com/casys-kaist/mNPUsim)
- [ARIES FPGA flow](https://www.csl.cornell.edu/~zhiruz/pdfs/aries-fpga2025.pdf)
- [FOSDEM 2026: NPU gen from Linalg](https://fosdem.org/2026/schedule/event/GTQRZE-programmable-npu-generation-from-linalg-mlir-circt/)
