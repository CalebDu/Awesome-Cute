add_executable(gemm_ws_sm8x gemm_naive_ws_sm8x.cu)
target_link_libraries(gemm_ws_sm8x PUBLIC -lcublas)
set_target_properties(
  gemm_ws_sm8x
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)

add_executable(gemm_ws_sm90a gemm_ws_sm90a.cu)
target_link_libraries(gemm_ws_sm90a PUBLIC -lcublas)
