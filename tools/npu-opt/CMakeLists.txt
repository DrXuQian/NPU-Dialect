add_llvm_executable(npu-opt
  npu-opt.cpp
)

target_link_libraries(npu-opt PRIVATE
  MLIR
  NPUDialect
  NPUAnalysis
  NPUTransforms
  NPUAnalysis
  NPUDialect
  NPUTransforms
)

llvm_update_compile_flags(npu-opt)
