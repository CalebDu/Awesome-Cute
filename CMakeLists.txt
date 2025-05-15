# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.18)

# Project name and language
project(AWESOME_CUTE VERSION 1.0 LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90a)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/${PLATFORM_ABI})

# set(CMAKE_CUDA_FLAGS "-lineinfo")
set(CMAKE_CUDA_FLAGS "-DNDEBUG --expt-relaxed-constexpr -O3 --expt-extended-lambda -Xptxas -v --use_fast_math")
set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__")

# set(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA REQUIRED)
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)

# find system torch cmake path
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)

list(APPEND CMAKE_PREFIX_PATH "${PYTORCH_CMAKE_PREFIX_PATH}/Torch")
message("PYTORCH_CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PREFIX_PATH}")

# message("CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(cutlass/include)
include_directories(cutlass/tools/util/include)
include_directories(cutlass/examples/common)
include_directories(include)

# include_directories(${TORCH_INCLUDE_DIRS})
add_subdirectory(gemm)
