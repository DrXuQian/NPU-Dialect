//===- CostEvaluatePass.cpp - Roofline cost evaluation --------*- C++ -*-===//
//
// Walks the function body and computes cost metrics using the roofline model.
// Emits cost breakdown as remarks and JSON to stderr.
// Does NOT modify the IR.
//
//===----------------------------------------------------------------------===//

#include "npu/Transforms/Passes.h"
#include "npu/Analysis/CostModel.h"
#include "npu/Dialect/NPU/NPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace npu {
#define GEN_PASS_DEF_NPUCOSTEVALUATE
#include "npu/Transforms/Passes.h.inc"
} // namespace npu

using namespace mlir;
using namespace npu;

namespace {

struct KernelCost {
  std::string name;
  double flops = 0;
  double memBytes = 0;
  double arithmeticIntensity = 0;
  double timeSec = 0;
  bool isMemoryBound = true;
  double efficiency = 0;
  int64_t dmaInBytes = 0;
  int64_t dmaOutBytes = 0;
  int64_t matrixCycles = 0;
  int64_t dspCycles = 0;
  int64_t peakSRAMBytes = 0;
  int64_t numTileIterations = 1;
  int64_t numLinalgOps = 0;
};

/// Extract constant trip count from an scf.for loop. Returns 1 if non-constant.
static int64_t getTripCount(scf::ForOp forOp) {
  auto lb = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ub = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto step = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!lb || !ub || !step || step.value() == 0)
    return 1;
  return (ub.value() - lb.value() + step.value() - 1) / step.value();
}

struct NPUCostEvaluatePass
    : public npu::impl::NPUCostEvaluateBase<NPUCostEvaluatePass> {

  void runOnOperation() override {
    auto funcOp = getOperation();
    Operation *module = funcOp->getParentOfType<ModuleOp>();
    if (!module) module = funcOp;
    CostModelAnalysis analysis(module);
    const CostModel &costModel = analysis.getCostModel();

    KernelCost kc;
    kc.name = funcOp.getName().str();

    // --- Collect linalg ops for roofline ---
    SmallVector<Operation *> linalgOps;
    funcOp.walk([&](linalg::LinalgOp op) {
      linalgOps.push_back(op.getOperation());
    });
    kc.numLinalgOps = linalgOps.size();

    // Roofline over all linalg ops (fused view if in same function)
    if (!linalgOps.empty()) {
      RooflineCost rc = costModel.evaluateRoofline(linalgOps);
      kc.flops = rc.flops;
      kc.memBytes = rc.memBytes;
      kc.arithmeticIntensity = rc.arithmeticIntensity;
      kc.timeSec = rc.timeSec;
      kc.isMemoryBound = rc.isMemoryBound;
      kc.efficiency = rc.efficiency;
    }

    // --- Accumulate DMA bytes ---
    funcOp.walk([&](npu::DMACopyOp dmaOp) {
      auto srcType = dyn_cast<ShapedType>(dmaOp.getSrc().getType());
      if (!srcType) return;
      int64_t bytes = CostModel::tensorBytes(srcType);
      if (dmaOp.getDirection() == npu::DMADirection::D2S)
        kc.dmaInBytes += bytes;
      else if (dmaOp.getDirection() == npu::DMADirection::S2D)
        kc.dmaOutBytes += bytes;
    });

    // --- Accumulate compute by engine ---
    funcOp.walk([&](linalg::LinalgOp op) {
      bool isMatrix = isa<linalg::MatmulOp>(op.getOperation()) ||
                      isa<linalg::BatchMatmulOp>(op.getOperation()) ||
                      isa<linalg::Conv2DNchwFchwOp>(op.getOperation());
      int64_t cycles = costModel.computeCycles(op);
      if (isMatrix)
        kc.matrixCycles += cycles;
      else
        kc.dspCycles += cycles;
    });

    // --- Track peak SRAM ---
    funcOp.walk([&](npu::AllocSRAMOp allocOp) {
      auto type = dyn_cast<MemRefType>(allocOp.getType());
      if (!type) return;
      int64_t bytes = CostModel::tensorBytes(type);
      int64_t offset = 0;
      if (auto attr = allocOp.getSramOffsetAttr())
        offset = attr.getInt();
      kc.peakSRAMBytes = std::max(kc.peakSRAMBytes, offset + bytes);
    });

    // --- Count tile iterations from outermost scf.for ---
    funcOp.walk([&](scf::ForOp forOp) {
      if (!forOp->getParentOfType<scf::ForOp>()) {
        // Outermost loop — multiply trip counts of nested loops
        int64_t trips = getTripCount(forOp);
        forOp.walk([&](scf::ForOp innerFor) {
          if (innerFor != forOp)
            trips *= getTripCount(innerFor);
        });
        kc.numTileIterations = std::max(kc.numTileIterations, trips);
      }
    });

    // --- Emit JSON to stderr ---
    llvm::json::Object json;
    json["name"] = kc.name;
    json["flops"] = kc.flops;
    json["mem_bytes"] = kc.memBytes;
    json["arithmetic_intensity"] = kc.arithmeticIntensity;
    json["time_sec"] = kc.timeSec;
    json["is_memory_bound"] = kc.isMemoryBound;
    json["efficiency"] = kc.efficiency;
    json["dma_in_bytes"] = kc.dmaInBytes;
    json["dma_out_bytes"] = kc.dmaOutBytes;
    json["matrix_cycles"] = kc.matrixCycles;
    json["dsp_cycles"] = kc.dspCycles;
    json["peak_sram_bytes"] = kc.peakSRAMBytes;
    json["num_tile_iterations"] = kc.numTileIterations;
    json["num_linalg_ops"] = kc.numLinalgOps;
    json["sram_budget"] = costModel.target().sramPerCore;
    json["bottleneck"] = kc.isMemoryBound ? "memory" : "compute";

    // Build the full line as a string first, then write atomically
    // to avoid interleaving when multiple FuncOps are processed.
    std::string jsonLine;
    llvm::raw_string_ostream os(jsonLine);
    os << "COST_JSON: " << llvm::json::Value(std::move(json)) << "\n";
    llvm::errs() << jsonLine;

    // --- Also emit as remark on the function ---
    funcOp->emitRemark("roofline: ")
        << llvm::formatv("{0:.2e} FLOP, {1:.2e} bytes, AI={2:.1f} FLOP/B, "
                         "{3}, eff={4:.1%}",
                         kc.flops, kc.memBytes, kc.arithmeticIntensity,
                         kc.isMemoryBound ? "MEM-bound" : "COMPUTE-bound",
                         kc.efficiency);
  }
};

} // namespace
