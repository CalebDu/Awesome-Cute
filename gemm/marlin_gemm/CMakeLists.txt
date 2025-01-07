add_library(marlin_lib SHARED marlin_cute_kernel.cu marlin_gemm.cpp marlin_official_kernel.cu)
target_include_directories(marlin_lib PRIVATE ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
target_link_libraries(marlin_lib PUBLIC -lcudart ${PYTHON_LIBRARIES} ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

set_target_properties(marlin_lib PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/gemm/marlin_gemm/"
  PREFIX ""
  SUFFIX ".so"
)