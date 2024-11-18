add_executable(gemm_streamk gemm_streamk.cu)
target_link_libraries(gemm_streamk PUBLIC -lcublas)
set_target_properties(
  gemm_streamk
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)