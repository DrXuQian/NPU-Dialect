add_mlir_library(NPUTransforms
  FusionPass.cpp
  OutlineFusedGroupsPass.cpp
  SpatialTilingPass.cpp
  CoreMappingPass.cpp
  TemporalTilingPass.cpp
  BufferizePass.cpp
  SRAMAllocationPass.cpp
  DMASchedulingPass.cpp
  CostEvaluatePass.cpp
  SpillHandlingPass.cpp
  Pipeline.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npu/Transforms

  DEPENDS
  NPUPassesIncGen

  LINK_LIBS PUBLIC
  MLIR
  NPUDialect
  NPUAnalysis
)
