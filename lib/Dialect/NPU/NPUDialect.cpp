//===- NPUDialect.cpp - NPU dialect implementation -----------*- C++ -*-===//

#include "npu/Dialect/NPU/NPUDialect.h"

using namespace mlir;

// ── Include generated definitions ────────────────────────────
#include "npu/Dialect/NPU/NPUDialect.cpp.inc"
#include "npu/Dialect/NPU/NPUEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "npu/Dialect/NPU/NPUTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "npu/Dialect/NPU/NPUAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "npu/Dialect/NPU/NPUOps.cpp.inc"

// ── Dialect initialization ───────────────────────────────────
void npu::NPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "npu/Dialect/NPU/NPUOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "npu/Dialect/NPU/NPUTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "npu/Dialect/NPU/NPUAttrs.cpp.inc"
      >();
}

Operation *npu::NPUDialect::materializeConstant(OpBuilder &builder,
                                                 Attribute value, Type type,
                                                 Location loc) {
  return nullptr;
}
