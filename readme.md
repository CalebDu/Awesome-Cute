# Awesome-Cute

Cute is new programming model in Nvidia GPU introduced from Cutlass3.0. Cute provides efficient abstraction for GPU programming to help developpers implement high performance CUDA code and more flexibly extend operator, such as epilogue or OP fusion.

This repo aims to implement some basic operators in deep learning by Cute from scratch.

## Support Matrix
- gemm_multstage
- gemm_streamk
- gemm_naive_warp_specialization
- gemm_marlin_w4a16
- more implementation to update

## Build project
``` shell
git submodule update --init --recursive
bash ./build.sh
```

