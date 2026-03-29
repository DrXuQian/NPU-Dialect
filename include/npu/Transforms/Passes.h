//===- Passes.h - NPU transformation passes -------------------*- C++ -*-===//
#ifndef NPU_TRANSFORMS_PASSES_H
#define NPU_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "npu/Analysis/CostModel.h"

namespace npu {

// ── Generated pass declarations (creates createNPU*() and Options structs) ──
#define GEN_PASS_DECL
#include "npu/Transforms/Passes.h.inc"

// ── Pipeline registration ────────────────────────────────────
void registerNPUPipeline();

// ── Generated pass registration ──────────────────────────────
#define GEN_PASS_REGISTRATION
#include "npu/Transforms/Passes.h.inc"

} // namespace npu

#endif // NPU_TRANSFORMS_PASSES_H
