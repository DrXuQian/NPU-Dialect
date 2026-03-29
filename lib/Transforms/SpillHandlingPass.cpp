//===- SpillHandlingPass.cpp - SRAM overflow handling ----------*- C++ -*-===//
//
// If a tile's working set exceeds SRAM:
// 1. Use CostModel.evaluateSpillStrategy() to decide retile vs spill
// 2. For spill: demote npu.alloc_sram back to memref.alloc (DRAM)
//    and insert npu.dma_copy pairs for DMA prefetch/writeback
//
//===----------------------------------------------------------------------===//

#include "npu/Transforms/Passes.h"
#include "npu/Analysis/CostModel.h"
#include "npu/Dialect/NPU/NPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace npu {
#define GEN_PASS_DEF_NPUSPILLHANDLING
#include "npu/Transforms/Passes.h.inc"
} // namespace npu

using namespace mlir;
using namespace npu;

namespace {

struct NPUSpillHandlingPass
    : public npu::impl::NPUSpillHandlingBase<NPUSpillHandlingPass> {

  void runOnOperation() override {
    auto funcOp = getOperation();

    Operation *module = funcOp->getParentOfType<ModuleOp>();
    if (!module)
      module = funcOp;
    CostModelAnalysis analysis(module);
    const CostModel &costModel = analysis.getCostModel();
    int64_t sramBudget = costModel.target().sramPerCore;

    // Walk each scf.for body to check SRAM budget.
    funcOp.walk([&](scf::ForOp forOp) {
      SmallVector<npu::AllocSRAMOp> sramAllocs;
      int64_t totalSramBytes = 0;

      forOp.getBody()->walk([&](npu::AllocSRAMOp allocOp) {
        sramAllocs.push_back(allocOp);
        auto memrefType = cast<MemRefType>(allocOp.getType());
        totalSramBytes += CostModel::tensorBytes(memrefType);
      });

      if (sramAllocs.empty() || totalSramBytes <= sramBudget)
        return;

      // SRAM overflow detected
      forOp.emitRemark("SRAM overflow: ")
          << totalSramBytes << " / " << sramBudget << " bytes";

      // Sort by size descending — spill largest first
      SmallVector<std::pair<npu::AllocSRAMOp, int64_t>> sorted;
      for (auto op : sramAllocs) {
        auto type = cast<MemRefType>(op.getType());
        sorted.push_back({op, CostModel::tensorBytes(type)});
      }
      llvm::sort(sorted, [](auto &a, auto &b) { return a.second > b.second; });

      int64_t current = totalSramBytes;
      for (auto &[allocOp, bytes] : sorted) {
        if (current <= sramBudget)
          break;

        CostModel::SpillStrategy strategy =
            costModel.evaluateSpillStrategy(current, bytes);

        if (strategy == CostModel::SpillStrategy::Retile) {
          allocOp->setAttr("npu.needs_retile",
                           OpBuilder(allocOp).getUnitAttr());
          forOp.emitRemark("spill: recommend retile for ") << bytes << " bytes";
        } else {
          // Spill: demote npu.alloc_sram → memref.alloc (stays in DRAM)
          // This is the simplest correct spill: the buffer lives in DRAM
          // and is accessed directly. A more advanced version would add
          // explicit DMA prefetch/writeback around uses.
          OpBuilder builder(allocOp);
          auto origType = cast<MemRefType>(allocOp.getType());
          // Create a contiguous memref type (strip dynamic strides/offsets)
          auto dramType = MemRefType::get(
              origType.getShape(), origType.getElementType(),
              MemRefLayoutAttrInterface{}, origType.getMemorySpace());
          auto dramAlloc = memref::AllocOp::create(
              builder, allocOp.getLoc(), dramType);
          // Cast back to original type if needed
          Value result = dramAlloc.getResult();
          if (dramType != origType) {
            result = builder.create<memref::CastOp>(
                allocOp.getLoc(), origType, result);
          }
          allocOp.replaceAllUsesWith(result);
          allocOp.erase();

          current -= bytes;
          forOp.emitRemark("spill: demoted ") << bytes << " bytes to DRAM";
        }
      }
    });
  }
};

} // namespace
