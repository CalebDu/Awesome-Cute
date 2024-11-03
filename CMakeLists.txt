# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.18)

# Project name and language
project(AWESOME_CUTE VERSION 1.0 LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/${PLATFORM_ABI})
set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr -O2")
# set(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA REQUIRED)

include_directories(cutlass/include)
include_directories(cutlass/tools/util/include)
include_directories(cutlass/examples/common)
include_directories(include)

add_subdirectory(gemm)
