add_executable(gemm_ws gemm_ws.cu)
target_link_libraries(gemm_ws PUBLIC -lcublas)
set_target_properties(
  gemm_ws
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)