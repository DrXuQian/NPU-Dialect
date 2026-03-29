//===- BufferizePass.cpp - SRAM allocation + DMA insertion ----*- C++ -*-===//
//
// Three phases:
//   1. One-Shot Module Bufferize (tensor -> memref)
//   2. SRAM allocation: replace tile-local memref.alloc with npu.alloc_sram
//   3. DMA insertion: insert npu.dma_copy for DRAM<->SRAM transfers
//
//===----------------------------------------------------------------------===//

#include "npu/Transforms/Passes.h"
#include "npu/Analysis/CostModel.h"
#include "npu/Dialect/NPU/NPUDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace npu {
#define GEN_PASS_DEF_NPUBUFFERIZE
#include "npu/Transforms/Passes.h.inc"
} // namespace npu

using namespace mlir;
using namespace npu;

namespace {

struct NPUBufferizePass
    : public npu::impl::NPUBufferizeBase<NPUBufferizePass> {

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // ================================================================
    // Phase 1: One-Shot Module Bufferize (tensor -> memref)
    // ================================================================
    {
      bool hasTensors = false;
      moduleOp.walk([&](Operation *op) {
        for (auto result : op->getResults()) {
          if (isa<RankedTensorType>(result.getType())) {
            hasTensors = true;
            return;
          }
        }
      });

      if (hasTensors) {
        bufferization::OneShotBufferizationOptions options;
        options.allowReturnAllocsFromLoops = true;
        options.copyBeforeWrite = true;
        bufferization::BufferizationState state;
        LogicalResult result = bufferization::runOneShotModuleBufferize(
            moduleOp, options, state, /*statistics=*/nullptr);
        if (failed(result)) {
          moduleOp.emitError("NPUBufferize: one-shot module bufferize failed");
          return signalPassFailure();
        }
      }
    }

    // ================================================================
    // Phase 2: SRAM allocation
    // ================================================================
    // Replace memref.alloc inside scf.for loops with npu.alloc_sram
    {
      SmallVector<memref::AllocOp> allocsToConvert;
      moduleOp.walk([&](memref::AllocOp allocOp) {
        if (allocOp->getParentOfType<scf::ForOp>() ||
            allocOp->getParentOfType<scf::ForallOp>()) {
          allocsToConvert.push_back(allocOp);
        }
      });

      for (auto allocOp : allocsToConvert) {
        OpBuilder builder(allocOp);
        auto sramAlloc = npu::AllocSRAMOp::create(
            builder, allocOp.getLoc(), allocOp.getType(),
            /*sram_offset=*/IntegerAttr());
        allocOp.replaceAllUsesWith(sramAlloc.getResult());
        allocOp.erase();
      }
    }

    // ================================================================
    // Phase 3: DMA insertion
    // ================================================================
    // For memref values defined outside the OUTERMOST tile loop that are
    // used inside, insert npu.dma_copy (DRAM→SRAM) at the loop entry.
    // Only process top-level scf.for ops (not nested ones) to avoid
    // dominance issues with loop-carried values.
    {
      SmallVector<scf::ForOp> topLevelLoops;
      moduleOp.walk([&](scf::ForOp forOp) {
        // Only process outermost loops (not nested inside another for)
        if (!forOp->getParentOfType<scf::ForOp>())
          topLevelLoops.push_back(forOp);
      });

      for (auto forOp : topLevelLoops) {
        SmallVector<std::pair<OpOperand *, Value>> dramReads;
        forOp.getBody()->walk([&](Operation *innerOp) {
          for (OpOperand &operand : innerOp->getOpOperands()) {
            Value val = operand.get();
            if (!isa<MemRefType>(val.getType()))
              continue;
            // Must be defined outside the outermost loop
            auto *defOp = val.getDefiningOp();
            if (!defOp)
              continue; // block argument — skip (includes iter_args)
            if (forOp->isAncestor(defOp))
              continue; // defined inside — skip
            if (isa<npu::AllocSRAMOp>(defOp))
              continue;
            if (isa<npu::DMACopyOp>(defOp))
              continue;
            dramReads.push_back({&operand, val});
          }
        });

        DenseMap<Value, Value> dramToSram;
        OpBuilder builder(forOp.getBody(), forOp.getBody()->begin());

        for (auto &[operand, dramVal] : dramReads) {
          if (dramToSram.count(dramVal)) {
            operand->set(dramToSram[dramVal]);
            continue;
          }

          auto memrefType = cast<MemRefType>(dramVal.getType());
          auto sramBuf = npu::AllocSRAMOp::create(
              builder, forOp.getLoc(), memrefType,
              /*sram_offset=*/IntegerAttr());
          npu::DMACopyOp::create(
              builder, forOp.getLoc(), dramVal, sramBuf.getResult(),
              npu::DMADirection::D2S);

          dramToSram[dramVal] = sramBuf.getResult();
          operand->set(sramBuf.getResult());
        }
      }
    }
  }
};

} // namespace
