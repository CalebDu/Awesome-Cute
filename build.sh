#!/bin/bash
# cmake configuration
cmake -S . -B build -G Ninja 
# build all execution
cd build
ninja all