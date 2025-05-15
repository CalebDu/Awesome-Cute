#!/bin/bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0a"
# cmake configuration
cmake -S . -B build -G Ninja 
# build all execution
cd build
ninja all