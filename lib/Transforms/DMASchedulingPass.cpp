//===- DMASchedulingPass.cpp - Barrier resource allocation -----*- C++ -*-===//
//
// Assigns hardware barrier resources to DMA/compute subgraphs.
//
// Algorithm:
//   1. Walk ops, group into subgraphs (DMA_in → compute → DMA_out)
//   2. Assign barrier IDs sequentially from a fixed pool
//   3. When pool exhausted → CUT (new generation)
//   4. Double buffer: generation n+2 reuses generation n's barriers
//      (guaranteed safe because n is complete by the time n+2 starts)
//
// No signal/wait ops — barrier_group attribute on each op tells
// hardware which barrier resource governs this op's synchronization.
//
//===----------------------------------------------------------------------===//

#include "npu/Transforms/Passes.h"
#include "npu/Analysis/CostModel.h"
#include "npu/Dialect/NPU/NPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

namespace npu {
#define GEN_PASS_DEF_NPUBARRIERALLOC
#include "npu/Transforms/Passes.h.inc"
} // namespace npu

using namespace mlir;
using namespace npu;

namespace {

//===----------------------------------------------------------------------===//
// Subgraph: a group of ops that share one barrier resource
//===----------------------------------------------------------------------===//

struct Subgraph {
  SmallVector<Operation *> dmaInOps;
  SmallVector<Operation *> computeOps;
  SmallVector<Operation *> dmaOutOps;

  /// Estimated cost (cycles) from cost model — used for timeline ordering.
  int64_t estimatedCycles = 0;

  /// Assigned barrier ID and generation.
  int64_t barrierId = -1;
  int64_t generation = 0;

  SmallVector<Operation *> allOps() const {
    SmallVector<Operation *> all;
    all.append(dmaInOps.begin(), dmaInOps.end());
    all.append(computeOps.begin(), computeOps.end());
    all.append(dmaOutOps.begin(), dmaOutOps.end());
    return all;
  }
};

/// Classify an op into a scheduling category.
enum class OpKind { DMAIn, DMAOut, Compute, Ignore };

static OpKind classifyOp(Operation *op) {
  if (auto dma = dyn_cast<npu::DMACopyOp>(op)) {
    if (dma.getDirection() == npu::DMADirection::D2S)
      return OpKind::DMAIn;
    if (dma.getDirection() == npu::DMADirection::S2D)
      return OpKind::DMAOut;
    return OpKind::DMAIn;
  }
  if (isa<linalg::LinalgOp>(op))
    return OpKind::Compute;
  if (isa<memref::CopyOp>(op))
    return OpKind::Compute;
  return OpKind::Ignore;
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct NPUBarrierAllocPass
    : public npu::impl::NPUBarrierAllocBase<NPUBarrierAllocPass> {
  using NPUBarrierAllocBase::NPUBarrierAllocBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    int64_t poolSize = barrierPoolSize;

    // Collect ALL schedulable ops across the entire function
    // (including nested loop bodies) in program order.
    SmallVector<Operation *> allOps;
    funcOp.walk([&](Operation *op) {
      if (classifyOp(op) != OpKind::Ignore)
        allOps.push_back(op);
    });

    if (allOps.empty())
      return;

    // Build subgraphs from the flat op list.
    SmallVector<Subgraph> subgraphs;
    Subgraph current;
    OpKind prevKind = OpKind::Ignore;

    for (Operation *op : allOps) {
      OpKind kind = classifyOp(op);

      // New subgraph when DMA_in follows DMA_out or Compute
      bool cut = (kind == OpKind::DMAIn) &&
                 (prevKind == OpKind::DMAOut || prevKind == OpKind::Compute);

      if (cut && !current.allOps().empty()) {
        subgraphs.push_back(std::move(current));
        current = Subgraph();
      }

      switch (kind) {
      case OpKind::DMAIn:  current.dmaInOps.push_back(op); break;
      case OpKind::Compute: current.computeOps.push_back(op); break;
      case OpKind::DMAOut: current.dmaOutOps.push_back(op); break;
      default: break;
      }
      prevKind = kind;
    }
    if (!current.allOps().empty())
      subgraphs.push_back(std::move(current));

    if (subgraphs.empty())
      return;

    // Assign barrier IDs with double-buffer generations.
    for (size_t i = 0; i < subgraphs.size(); ++i) {
      subgraphs[i].barrierId = static_cast<int64_t>(i % poolSize);
      subgraphs[i].generation = static_cast<int64_t>(i / poolSize);
    }

    // Annotate ops.
    for (auto &sg : subgraphs) {
      for (Operation *op : sg.allOps()) {
        OpBuilder builder(op);
        op->setAttr("barrier_group",
                     builder.getI64IntegerAttr(sg.barrierId));
        op->setAttr("barrier_gen",
                     builder.getI64IntegerAttr(sg.generation));
      }
    }

    int64_t numGens = subgraphs.back().generation + 1;
    funcOp->emitRemark("barrier alloc: ")
        << subgraphs.size() << " subgraphs, "
        << poolSize << " barriers, "
        << numGens << " generation(s)"
        << (numGens > 1 ? " (double-buffered)" : "");
  }

private:
  void processBlock(Block *block, int64_t poolSize) {
    // Step 1: Build subgraphs by grouping consecutive ops.
    // A subgraph boundary occurs when we see a DMA_in after a DMA_out
    // (= next tile's load after previous tile's store).
    SmallVector<Subgraph> subgraphs;
    Subgraph current;
    OpKind prevKind = OpKind::Ignore;

    for (auto &op : *block) {
      OpKind kind = classifyOp(&op);
      if (kind == OpKind::Ignore)
        continue;

      // Detect subgraph boundary: DMA_in after DMA_out or Compute
      // means we're starting a new DMA→compute→DMA group.
      bool newSubgraph = false;
      if (kind == OpKind::DMAIn &&
          (prevKind == OpKind::DMAOut || prevKind == OpKind::Compute) &&
          !current.dmaInOps.empty()) {
        newSubgraph = true;
      }

      if (newSubgraph && !current.allOps().empty()) {
        subgraphs.push_back(std::move(current));
        current = Subgraph();
      }

      switch (kind) {
      case OpKind::DMAIn:
        current.dmaInOps.push_back(&op);
        break;
      case OpKind::Compute:
        current.computeOps.push_back(&op);
        break;
      case OpKind::DMAOut:
        current.dmaOutOps.push_back(&op);
        break;
      default:
        break;
      }
      prevKind = kind;
    }
    // Flush last subgraph
    if (!current.allOps().empty())
      subgraphs.push_back(std::move(current));

    if (subgraphs.empty())
      return;

    // Step 2: Assign barrier IDs with double-buffer generations.
    //
    // barrier_id cycles through [0, poolSize)
    // generation increments every poolSize subgraphs
    //
    // Subgraph n gets: barrier = n % poolSize, generation = n / poolSize
    // Subgraph n+2*poolSize reuses the same barrier+generation pair.
    // Hardware ensures: subgraph n must finish before n+2*poolSize starts
    // (because there are 2*poolSize subgraphs in between = "double buffer").

    for (size_t i = 0; i < subgraphs.size(); ++i) {
      auto &sg = subgraphs[i];
      sg.barrierId = static_cast<int64_t>(i % poolSize);
      sg.generation = static_cast<int64_t>(i / poolSize);
    }

    // Step 3: Annotate ops with barrier_group and generation.
    for (auto &sg : subgraphs) {
      for (Operation *op : sg.allOps()) {
        OpBuilder builder(op);
        op->setAttr("barrier_group",
                     builder.getI64IntegerAttr(sg.barrierId));
        op->setAttr("barrier_gen",
                     builder.getI64IntegerAttr(sg.generation));
      }
    }

    // Emit remark
    if (auto *parentOp = block->getParentOp()) {
      int64_t numGenerations =
          subgraphs.empty() ? 0 : subgraphs.back().generation + 1;
      parentOp->emitRemark("barrier alloc: ")
          << subgraphs.size() << " subgraphs, "
          << poolSize << " barriers, "
          << numGenerations << " generation(s)"
          << (numGenerations > 1 ? " (double-buffered)" : "");
    }
  }
};

} // namespace
