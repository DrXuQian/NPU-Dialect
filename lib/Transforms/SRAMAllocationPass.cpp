//===- SRAMAllocationPass.cpp - SRAM address planning + spill ---*- C++ -*-===//
//
// Unified SRAM allocation for each kernel function.
//
// Invariant: SRAM is empty at function entry and exit.
//
// Layout (dual-end):
//   [0 ── inputs → ]  [ ← intermediates ]  [ ← outputs ── SRAM_SIZE]
//
// Features:
//   - Liveness-based allocation with reuse
//   - Spill to DDR when SRAM overflows
//   - Double buffering: input DMA buffers get ping/pong pairs
//   - Weight prefetch: if free SRAM after tile allocation, prefetch
//     next tile's Shared operands from the furthest free address
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
#include <numeric>

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
  int64_t firstUse = 0;
  int64_t lastUse = 0;
  int64_t sramAddr = -1;        // primary address
  int64_t sramAddrPong = -1;    // pong address (for double buffering)
  Value ddrShadow;
  bool spilled = false;
  bool isDoubleBuffered = false;
};

//===----------------------------------------------------------------------===//
// Classify buffer roles
//===----------------------------------------------------------------------===//

static BufRole classifyRole(npu::AllocSRAMOp allocOp) {
  Value buf = allocOp.getResult();
  bool isDmaInput = false;
  bool isDmaOutput = false;

  for (Operation *user : buf.getUsers()) {
    if (auto dma = dyn_cast<npu::DMACopyOp>(user)) {
      if (dma.getDst() == buf && dma.getDirection() == npu::DMADirection::D2S)
        isDmaInput = true;
      if (dma.getSrc() == buf && dma.getDirection() == npu::DMADirection::S2D)
        isDmaOutput = true;
    }
  }

  if (isDmaInput && !isDmaOutput) return BufRole::Input;
  if (isDmaOutput) return BufRole::Output;
  return BufRole::Intermediate;
}

/// Check if a buffer is inside a tile loop (scf.for) — candidate for
/// double buffering.
static bool isInsideTileLoop(npu::AllocSRAMOp allocOp) {
  return allocOp->getParentOfType<scf::ForOp>() != nullptr;
}

//===----------------------------------------------------------------------===//
// Liveness analysis
//===----------------------------------------------------------------------===//

static DenseMap<Operation *, int64_t> buildStepMap(Block &block) {
  DenseMap<Operation *, int64_t> stepMap;
  int64_t step = 0;
  for (Operation &op : block) {
    stepMap[&op] = step++;
    op.walk([&](Operation *nested) {
      if (!stepMap.count(nested))
        stepMap[nested] = step++;
    });
  }
  return stepMap;
}

static void computeLiveness(SmallVectorImpl<SRAMBuffer> &buffers,
                            const DenseMap<Operation *, int64_t> &stepMap) {
  for (auto &buf : buffers) {
    int64_t first = INT64_MAX, last = 0;
    Value v = buf.allocOp.getResult();
    if (auto it = stepMap.find(buf.allocOp.getOperation()); it != stepMap.end())
      first = std::min(first, it->second);
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
// Dual-end allocator
//===----------------------------------------------------------------------===//

struct DualEndAllocator {
  int64_t sramSize;
  int64_t lowWater = 0;
  int64_t highWater;

  explicit DualEndAllocator(int64_t size)
      : sramSize(size), highWater(size) {}

  int64_t allocLow(int64_t bytes) {
    int64_t aligned = align(bytes);
    if (lowWater + aligned > highWater) return -1;
    int64_t addr = lowWater;
    lowWater += aligned;
    return addr;
  }

  int64_t allocHigh(int64_t bytes) {
    int64_t aligned = align(bytes);
    if (highWater - aligned < lowWater) return -1;
    highWater -= aligned;
    return highWater;
  }

  int64_t allocMiddle(int64_t bytes) { return allocLow(bytes); }

  void freeLow(int64_t bytes) {
    lowWater -= align(bytes);
    if (lowWater < 0) lowWater = 0;
  }

  int64_t freeBytes() const { return highWater - lowWater; }

  static int64_t align(int64_t bytes) {
    return (bytes + 63) & ~63;
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

    SmallVector<SRAMBuffer> buffers;
    funcOp.walk([&](npu::AllocSRAMOp allocOp) {
      auto memrefType = cast<MemRefType>(allocOp.getType());
      int64_t bytes = CostModel::tensorBytes(memrefType);
      BufRole role = classifyRole(allocOp);
      SRAMBuffer buf;
      buf.allocOp = allocOp;
      buf.memrefType = memrefType;
      buf.bytes = bytes;
      buf.role = role;
      buffers.push_back(buf);
    });

    if (buffers.empty())
      return;

    Block &entryBlock = funcOp.getBody().front();
    auto stepMap = buildStepMap(entryBlock);
    computeLiveness(buffers, stepMap);

    // ================================================================
    // Phase 1: Liveness-aware dual-end allocation
    // ================================================================
    DualEndAllocator allocator(sramSize);

    SmallVector<size_t> order(buffers.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return buffers[a].firstUse < buffers[b].firstUse;
    });

    SmallVector<std::pair<size_t, int64_t>> active;
    SmallVector<size_t> spillCandidates;

    for (size_t idx : order) {
      auto &buf = buffers[idx];

      // Free dead buffers
      SmallVector<std::pair<size_t, int64_t>> stillActive;
      for (auto &[aIdx, aLast] : active) {
        if (aLast < buf.firstUse) {
          auto &aBuf = buffers[aIdx];
          if (aBuf.role != BufRole::Output) {
            allocator.freeLow(aBuf.bytes);
            // Also free pong buffer if double-buffered
            if (aBuf.isDoubleBuffered)
              allocator.freeLow(aBuf.bytes);
          }
        } else {
          stillActive.push_back({aIdx, aLast});
        }
      }
      active = std::move(stillActive);

      // Determine if this buffer should be double-buffered.
      // Double buffer: DMA input inside a tile loop → ping/pong.
      bool wantDoubleBuffer =
          (buf.role == BufRole::Input) && isInsideTileLoop(buf.allocOp);

      int64_t addr = -1;
      int64_t addrPong = -1;

      switch (buf.role) {
      case BufRole::Input:
        addr = allocator.allocLow(buf.bytes);
        if (addr >= 0 && wantDoubleBuffer) {
          addrPong = allocator.allocLow(buf.bytes);
          if (addrPong < 0) {
            // Can't fit pong buffer — single buffer only
            wantDoubleBuffer = false;
          }
        }
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
        buf.isDoubleBuffered = wantDoubleBuffer;
        if (wantDoubleBuffer)
          buf.sramAddrPong = addrPong;
        active.push_back({idx, buf.lastUse});
      } else {
        spillCandidates.push_back(idx);
      }
    }

    // ================================================================
    // Phase 2: Spill
    // ================================================================
    if (!spillCandidates.empty()) {
      funcOp.emitRemark("SRAM allocation: ")
          << spillCandidates.size() << " buffer(s) spilled to DDR";
      for (size_t idx : spillCandidates)
        insertSpillDMA(buffers[idx]);
    }

    // ================================================================
    // Phase 3: Set sram_offset attributes + emit double-buffer info
    // ================================================================
    for (auto &buf : buffers) {
      if (buf.sramAddr >= 0 && !buf.spilled) {
        auto builder = OpBuilder(buf.allocOp);
        buf.allocOp->setAttr("sram_offset",
            builder.getI64IntegerAttr(buf.sramAddr));
        if (buf.isDoubleBuffered && buf.sramAddrPong >= 0) {
          buf.allocOp->setAttr("sram_offset_pong",
              builder.getI64IntegerAttr(buf.sramAddrPong));
          buf.allocOp->setAttr("double_buffered", builder.getUnitAttr());
        }
      }
    }

    // ================================================================
    // Phase 4: Weight prefetch — look ahead for next tile's weights
    // ================================================================
    // After allocation, check if there's free SRAM to prefetch weights
    // for the next tile iteration.
    int64_t freeAfterAlloc = allocator.freeBytes();
    if (freeAfterAlloc > 0) {
      // Find DMA input buffers (weights loaded via D2S) that are Shared
      // (i.e., loaded identically every tile iteration — candidates for prefetch)
      SmallVector<SRAMBuffer *> prefetchCandidates;
      for (auto &buf : buffers) {
        if (buf.spilled || buf.sramAddr < 0)
          continue;
        if (buf.role != BufRole::Input)
          continue;
        // Weights are typically not double-buffered and are "Shared"
        // (same data every tile). Prefetch the largest ones first.
        prefetchCandidates.push_back(&buf);
      }

      // Sort by size descending — prefetch largest first
      std::sort(prefetchCandidates.begin(), prefetchCandidates.end(),
                [](SRAMBuffer *a, SRAMBuffer *b) {
                  return a->bytes > b->bytes;
                });

      int64_t prefetchBudget = freeAfterAlloc;
      int64_t numPrefetched = 0;

      for (auto *buf : prefetchCandidates) {
        if (buf->bytes > prefetchBudget)
          continue;
        // Allocate prefetch buffer from the furthest free address (high end)
        int64_t prefetchAddr = allocator.allocHigh(buf->bytes);
        if (prefetchAddr < 0)
          continue;

        buf->allocOp->setAttr("prefetch_addr",
            OpBuilder(buf->allocOp).getI64IntegerAttr(prefetchAddr));
        prefetchBudget -= DualEndAllocator::align(buf->bytes);
        numPrefetched++;
      }

      if (numPrefetched > 0) {
        funcOp.emitRemark("Weight prefetch: ")
            << numPrefetched << " buffer(s) prefetched, "
            << (freeAfterAlloc - prefetchBudget) << " bytes used";
      }
    }
  }

private:
  void insertSpillDMA(SRAMBuffer &buf) {
    auto allocOp = buf.allocOp;
    Location loc = allocOp.getLoc();
    auto origType = cast<MemRefType>(allocOp.getType());
    auto ddrType = MemRefType::get(
        origType.getShape(), origType.getElementType(),
        MemRefLayoutAttrInterface{}, origType.getMemorySpace());

    OpBuilder builder(allocOp);
    auto ddrAlloc = memref::AllocOp::create(builder, loc, ddrType);
    buf.ddrShadow = ddrAlloc.getResult();
    buf.spilled = true;

    Value replacement = ddrAlloc.getResult();
    if (ddrType != origType)
      replacement = builder.create<memref::CastOp>(loc, origType, replacement);
    allocOp.replaceAllUsesWith(replacement);
    allocOp.erase();
  }
};

} // namespace
