//===- NPUDialect.h - NPU dialect declaration -----------------*- C++ -*-===//
#ifndef NPU_DIALECT_NPU_NPUDIALECT_H
#define NPU_DIALECT_NPU_NPUDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/TypeSwitch.h"

// Include generated enum declarations
#include "npu/Dialect/NPU/NPUEnums.h.inc"

// Include generated type declarations
#define GET_TYPEDEF_CLASSES
#include "npu/Dialect/NPU/NPUTypes.h.inc"

// Include generated attribute declarations
#define GET_ATTRDEF_CLASSES
#include "npu/Dialect/NPU/NPUAttrs.h.inc"

// Include generated dialect declaration
#include "npu/Dialect/NPU/NPUDialect.h.inc"

// Include generated op declarations
#define GET_OP_CLASSES
#include "npu/Dialect/NPU/NPUOps.h.inc"

#endif // NPU_DIALECT_NPU_NPUDIALECT_H
