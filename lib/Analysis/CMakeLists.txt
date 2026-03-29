add_mlir_library(NPUAnalysis
  CostModel.cpp
  OperatorTilingSpec.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npu/Analysis

  DEPENDS
  NPUDialectIncGen

  LINK_LIBS PUBLIC
  MLIR
  NPUDialect
)
