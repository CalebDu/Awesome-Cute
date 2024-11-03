
add_executable(gemm_multistage gemm_multistage.cu)
target_link_libraries(gemm_multistage PUBLIC -lcublas)
set_target_properties(
  gemm_multistage
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)