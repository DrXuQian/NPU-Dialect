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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SetVector.h"

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
        options.allowUnknownOps = true;

        bufferization::BufferizationState state;
        LogicalResult result = bufferization::runOneShotModuleBufferize(
            moduleOp, options, state, /*statistics=*/nullptr);
        if (failed(result)) {
          moduleOp.emitRemark("NPUBufferize: module bufferize failed, continuing");
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
    // For each scf.for loop (including nested ones), identify memref
    // values defined OUTSIDE that loop and used INSIDE it.  Insert
    // npu.alloc_sram + npu.dma_copy (DRAM→SRAM) at the loop body
    // start, replace uses inside with the SRAM buffer, and insert a
    // writeback npu.dma_copy (SRAM→DRAM) before scf.yield for output
    // values (those written inside the loop).
    //
    // We process loops innermost-first (reverse walk order) so that
    // inner loops see SRAM buffers already placed by outer loops.
    {
      // Collect all scf.for ops.  Walk collects innermost first when
      // we reverse the resulting list, but MLIR's walk is pre-order,
      // so we want to process in reverse (innermost first) to avoid
      // re-processing values that get rewritten.  However, processing
      // outermost-first is actually correct here: the outer loop's
      // DMA replacement propagates to inner loops automatically
      // because we replace all uses inside the loop body.
      // Process ALL loops (not just outermost).
      SmallVector<scf::ForOp> allLoops;
      moduleOp.walk([&](scf::ForOp forOp) {
        allLoops.push_back(forOp);
      });

      // Process outermost loops first, then inner loops.  This way,
      // when an outer loop's DRAM value is replaced with an SRAM
      // buffer, inner loops will already see the SRAM value and skip
      // it (since it's defined by npu.alloc_sram).
      for (auto forOp : allLoops) {
        // --- Identify memref values from DRAM used inside this loop ---
        // Use a SetVector to maintain insertion order while deduplicating.
        llvm::SetVector<Value> dramInputs;
        // Track which DRAM values are written to inside the loop (outputs).
        llvm::DenseSet<Value> dramOutputs;

        forOp.getBody()->walk([&](Operation *innerOp) {
          // Skip the yield terminator for the input scan.
          if (isa<scf::YieldOp>(innerOp))
            return;

          for (OpOperand &operand : innerOp->getOpOperands()) {
            Value val = operand.get();
            if (!isa<MemRefType>(val.getType()))
              continue;

            // Skip block arguments (induction var, iter_args).
            auto *defOp = val.getDefiningOp();
            if (!defOp)
              continue;

            // Skip values defined inside THIS loop.
            if (forOp->isAncestor(defOp))
              continue;

            // Skip values already in SRAM.
            if (isa<npu::AllocSRAMOp>(defOp))
              continue;

            // Skip values that are results of a DMA copy (already managed).
            if (isa<npu::DMACopyOp>(defOp))
              continue;

            // Skip constants.
            if (isa<arith::ConstantOp>(defOp))
              continue;

            dramInputs.insert(val);

            // Detect writes: linalg ops write to their output operands
            // (DPS inits), and memref.store also writes.
            if (auto linalgOp = dyn_cast<linalg::LinalgOp>(innerOp)) {
              // Check if this operand is an init (output) of the linalg op.
              for (OpOperand &initOperand : linalgOp.getDpsInitsMutable()) {
                if (&initOperand == &operand) {
                  dramOutputs.insert(val);
                  break;
                }
              }
            } else if (isa<memref::StoreOp>(innerOp)) {
              // memref.store's first operand is the value, second is the memref
              if (&operand == &innerOp->getOpOperand(1))
                dramOutputs.insert(val);
            }
          }
        });

        if (dramInputs.empty())
          continue;

        // --- Insert DRAM→SRAM DMA at loop body start ---
        DenseMap<Value, Value> dramToSram;
        OpBuilder startBuilder(forOp.getBody(), forOp.getBody()->begin());

        for (Value dramVal : dramInputs) {
          auto memrefType = cast<MemRefType>(dramVal.getType());
          auto sramBuf = npu::AllocSRAMOp::create(
              startBuilder, forOp.getLoc(), memrefType,
              /*sram_offset=*/IntegerAttr());
          npu::DMACopyOp::create(
              startBuilder, forOp.getLoc(), dramVal, sramBuf.getResult(),
              npu::DMADirection::D2S);
          dramToSram[dramVal] = sramBuf.getResult();
        }

        // --- Replace uses inside the loop body ---
        forOp.getBody()->walk([&](Operation *innerOp) {
          for (OpOperand &operand : innerOp->getOpOperands()) {
            Value val = operand.get();
            auto it = dramToSram.find(val);
            if (it != dramToSram.end())
              operand.set(it->second);
          }
        });

        // --- Insert SRAM→DRAM writeback for output values ---
        if (!dramOutputs.empty()) {
          // Find the scf.yield terminator to insert before it.
          auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
          OpBuilder endBuilder(yieldOp);

          for (Value dramVal : dramInputs) {
            if (!dramOutputs.contains(dramVal))
              continue;
            Value sramVal = dramToSram[dramVal];
            npu::DMACopyOp::create(
                endBuilder, forOp.getLoc(), sramVal, dramVal,
                npu::DMADirection::S2D);
          }
        }
      }
    }
  }
};

} // namespace
