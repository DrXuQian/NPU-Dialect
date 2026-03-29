add_mlir_dialect_library(NPUDialect
  NPUDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npu/Dialect/NPU

  DEPENDS
  NPUDialectIncGen

  LINK_LIBS PUBLIC
  MLIR
)
