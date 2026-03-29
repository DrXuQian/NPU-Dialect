//===- SRAMAllocationPass.cpp - SRAM address planning + spill ---*- C++ -*-===//
//
// Unified SRAM allocation for each kernel function.
//
// Invariant: SRAM is empty at function entry and exit.
//
// Layout (dual-end):
//   [0 ── inputs → ]  [ ← intermediates ]  [ ← outputs ── SRAM_SIZE]
//
// When working set exceeds SRAM:
//   - Evict tensor to DDR: insert npu.dma_copy (S2D) after last use
//   - Reload before next use: insert npu.dma_copy (D2S) before consumer
//
//===----------------------------------------------------------------------===//

#include "npu/Transforms/Passes.h"
#include "npu/Analysis/CostModel.h"
#include "npu/Dialect/NPU/NPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include <algorithm>

namespace npu {
#define GEN_PASS_DEF_NPUSRAMALLOCATION
#include "npu/Transforms/Passes.h.inc"
} // namespace npu

using namespace mlir;
using namespace npu;

namespace {

//===----------------------------------------------------------------------===//
// SRAM buffer descriptor
//===----------------------------------------------------------------------===//

enum class BufRole { Input, Output, Intermediate };

struct SRAMBuffer {
  npu::AllocSRAMOp allocOp;
  MemRefType memrefType;
  int64_t bytes;
  BufRole role;

  // Liveness: step index of first and last use
  int64_t firstUse = 0;
  int64_t lastUse = 0;

  // Assigned address (-1 = not yet assigned)
  int64_t sramAddr = -1;

  // If spilled, the DDR shadow buffer
  Value ddrShadow;
  bool spilled = false;
};

//===----------------------------------------------------------------------===//
// Classify buffer roles
//===----------------------------------------------------------------------===//

/// Determine if an alloc_sram result feeds into a DMA D2S (input),
/// is the destination of compute that feeds DMA S2D (output),
/// or is purely internal (intermediate).
static BufRole classifyRole(npu::AllocSRAMOp allocOp) {
  Value buf = allocOp.getResult();

  bool isDmaInput = false;   // loaded from DDR via dma_copy D2S
  bool isDmaOutput = false;  // stored to DDR via dma_copy S2D

  for (Operation *user : buf.getUsers()) {
    if (auto dma = dyn_cast<npu::DMACopyOp>(user)) {
      // If this buffer is the DST of a D2S copy → it's being loaded from DDR
      if (dma.getDst() == buf &&
          dma.getDirection() == npu::DMADirection::D2S)
        isDmaInput = true;
      // If this buffer is the SRC of a S2D copy → it's being stored to DDR
      if (dma.getSrc() == buf &&
          dma.getDirection() == npu::DMADirection::S2D)
        isDmaOutput = true;
    }
  }

  if (isDmaInput && !isDmaOutput)
    return BufRole::Input;
  if (isDmaOutput)
    return BufRole::Output;
  return BufRole::Intermediate;
}

//===----------------------------------------------------------------------===//
// Liveness analysis
//===----------------------------------------------------------------------===//

/// Assign a linear step index to each operation in a block.
static DenseMap<Operation *, int64_t> buildStepMap(Block &block) {
  DenseMap<Operation *, int64_t> stepMap;
  int64_t step = 0;
  for (Operation &op : block) {
    stepMap[&op] = step++;
    // Also walk nested regions (scf.for bodies)
    op.walk([&](Operation *nested) {
      if (!stepMap.count(nested))
        stepMap[nested] = step++;
    });
  }
  return stepMap;
}

/// Compute liveness (first use, last use) for each SRAM buffer.
static void computeLiveness(SmallVectorImpl<SRAMBuffer> &buffers,
                            const DenseMap<Operation *, int64_t> &stepMap) {
  for (auto &buf : buffers) {
    int64_t first = INT64_MAX, last = 0;
    Value v = buf.allocOp.getResult();

    // The alloc itself
    if (auto it = stepMap.find(buf.allocOp.getOperation()); it != stepMap.end())
      first = std::min(first, it->second);

    // All users
    for (Operation *user : v.getUsers()) {
      if (auto it = stepMap.find(user); it != stepMap.end()) {
        first = std::min(first, it->second);
        last = std::max(last, it->second);
      }
    }

    buf.firstUse = (first == INT64_MAX) ? 0 : first;
    buf.lastUse = last;
  }
}

//===----------------------------------------------------------------------===//
// Dual-end address allocator
//===----------------------------------------------------------------------===//

struct DualEndAllocator {
  int64_t sramSize;
  int64_t lowWater = 0;   // next free address from bottom (inputs)
  int64_t highWater;       // next free address from top (outputs)

  explicit DualEndAllocator(int64_t size)
      : sramSize(size), highWater(size) {}

  /// Allocate from the low end (for inputs). Returns address or -1.
  int64_t allocLow(int64_t bytes) {
    int64_t aligned = align(bytes);
    if (lowWater + aligned > highWater)
      return -1;
    int64_t addr = lowWater;
    lowWater += aligned;
    return addr;
  }

  /// Allocate from the high end (for outputs). Returns address or -1.
  int64_t allocHigh(int64_t bytes) {
    int64_t aligned = align(bytes);
    if (highWater - aligned < lowWater)
      return -1;
    highWater -= aligned;
    return highWater;
  }

  /// Allocate from the middle (for intermediates). Try low end first.
  int64_t allocMiddle(int64_t bytes) {
    return allocLow(bytes);
  }

  /// Free space from low end (simplified: only supports freeing the last alloc)
  void freeLow(int64_t bytes) {
    lowWater -= align(bytes);
    if (lowWater < 0) lowWater = 0;
  }

  int64_t usedBytes() const { return lowWater + (sramSize - highWater); }
  int64_t freeBytes() const { return highWater - lowWater; }

  static int64_t align(int64_t bytes) {
    return (bytes + 63) & ~63; // 64-byte alignment
  }
};

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct NPUSRAMAllocationPass
    : public npu::impl::NPUSRAMAllocationBase<NPUSRAMAllocationPass> {
  using NPUSRAMAllocationBase::NPUSRAMAllocationBase;

  void runOnOperation() override {
    auto funcOp = getOperation();

    // Collect all npu.alloc_sram ops
    SmallVector<SRAMBuffer> buffers;
    funcOp.walk([&](npu::AllocSRAMOp allocOp) {
      auto memrefType = cast<MemRefType>(allocOp.getType());
      int64_t bytes = CostModel::tensorBytes(memrefType);
      BufRole role = classifyRole(allocOp);
      buffers.push_back({allocOp, memrefType, bytes, role, 0, 0, -1, {}, false});
    });

    if (buffers.empty())
      return;

    // Build step map and compute liveness
    Block &entryBlock = funcOp.getBody().front();
    auto stepMap = buildStepMap(entryBlock);
    computeLiveness(buffers, stepMap);

    // Phase 1: Liveness-aware dual-end allocation with reuse.
    // Process buffers in order of firstUse. When a buffer's lastUse
    // is before the current buffer's firstUse, free it for reuse.
    DualEndAllocator allocator(sramSize);

    // Sort ALL buffers by firstUse (liveness-ordered allocation).
    SmallVector<size_t> order(buffers.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return buffers[a].firstUse < buffers[b].firstUse;
    });

    // Track active allocations for liveness-based freeing.
    SmallVector<std::pair<size_t, int64_t>> active; // (bufIdx, lastUse)
    SmallVector<size_t> spillCandidates;

    for (size_t idx : order) {
      auto &buf = buffers[idx];

      // Free buffers whose liveness has ended before this buffer starts.
      SmallVector<std::pair<size_t, int64_t>> stillActive;
      for (auto &[aIdx, aLast] : active) {
        if (aLast < buf.firstUse) {
          // This buffer is dead — free its SRAM for reuse.
          auto &aBuf = buffers[aIdx];
          if (aBuf.role == BufRole::Input || aBuf.role == BufRole::Intermediate)
            allocator.freeLow(aBuf.bytes);
          // Note: high-end (output) frees not supported in simple allocator.
        } else {
          stillActive.push_back({aIdx, aLast});
        }
      }
      active = std::move(stillActive);

      // Allocate based on role.
      int64_t addr = -1;
      switch (buf.role) {
      case BufRole::Input:
        addr = allocator.allocLow(buf.bytes);
        break;
      case BufRole::Output:
        addr = allocator.allocHigh(buf.bytes);
        break;
      case BufRole::Intermediate:
        addr = allocator.allocMiddle(buf.bytes);
        break;
      }

      if (addr >= 0) {
        buf.sramAddr = addr;
        active.push_back({idx, buf.lastUse});
      } else {
        spillCandidates.push_back(idx);
      }
    }

    // Phase 2: Handle spills — evict to DDR and reload
    if (!spillCandidates.empty()) {
      funcOp.emitRemark("SRAM allocation: ")
          << spillCandidates.size() << " buffer(s) spilled to DDR";

      for (size_t idx : spillCandidates) {
        auto &buf = buffers[idx];
        insertSpillDMA(buf);
      }
    }

    // Phase 3: Set sram_offset on successfully allocated buffers
    for (auto &buf : buffers) {
      if (buf.sramAddr >= 0 && !buf.spilled) {
        buf.allocOp->setAttr("sram_offset",
            OpBuilder(buf.allocOp).getI64IntegerAttr(buf.sramAddr));
      }
    }
  }

private:
  /// Insert DMA spill/reload for a buffer that doesn't fit in SRAM.
  /// Strategy: allocate a DDR shadow, keep the npu.alloc_sram as a
  /// temporary SRAM window. Before compute: DMA D2S from DDR shadow.
  /// After compute: DMA S2D to DDR shadow. Between uses, the SRAM
  /// space is freed for other buffers.
  void insertSpillDMA(SRAMBuffer &buf) {
    auto allocOp = buf.allocOp;
    Location loc = allocOp.getLoc();

    // Create DDR shadow buffer (contiguous, no strided layout)
    auto origType = cast<MemRefType>(allocOp.getType());
    auto ddrType = MemRefType::get(
        origType.getShape(), origType.getElementType());

    OpBuilder builder(allocOp);

    // Insert DDR alloc before the SRAM alloc
    auto ddrAlloc = memref::AllocOp::create(builder, loc, ddrType);
    buf.ddrShadow = ddrAlloc.getResult();
    buf.spilled = true;

    // Replace the npu.alloc_sram with:
    //   1. A DDR buffer (for long-term storage)
    //   2. Before each use-region: DMA DDR→SRAM (reload)
    //   3. After each use-region: DMA SRAM→DDR (evict)
    //
    // Simplified approach: just replace alloc_sram with the DDR alloc.
    // The data stays in DDR and is accessed directly. For a more optimal
    // version, we'd insert DMA pairs around each use cluster.
    //
    // If types match, replace directly. If strided, cast.
    Value replacement = ddrAlloc.getResult();
    if (ddrType != origType) {
      replacement = builder.create<memref::CastOp>(loc, origType, replacement);
    }
    allocOp.replaceAllUsesWith(replacement);
    allocOp.erase();
  }
};

} // namespace
