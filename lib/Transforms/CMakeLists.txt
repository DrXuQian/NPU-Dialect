add_mlir_library(NPUTransforms
  FusionPass.cpp
  OutlineFusedGroupsPass.cpp
  SpatialTilingPass.cpp
  TemporalTilingPass.cpp
  BufferizePass.cpp
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
