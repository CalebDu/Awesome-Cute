#pragma once
#include "cublas_v2.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "helper.h"
#include "utils.h"
#include <cmath>
/*
cutlass golden compute
*/
template <typename Atype, typename ALayout, typename Btype, typename BLayout,
          typename Ctype, typename CLayout>
inline void
cutlass_gemmTN_ref(cutlass::HostTensor<Atype, ALayout> const &A, // row-major
                   cutlass::HostTensor<Btype, BLayout> const &B, // col-major
                   cutlass::HostTensor<Ctype, CLayout> &C,
                   Ctype alpha = static_cast<Ctype>(1),
                   Ctype beta = static_cast<Ctype>(0), int repeat = 1000) {
  int m = A.extent().row();
  int n = B.extent().column();
  int k = A.extent().column();
  float gflop = 2.0 * m * n * k / 1e9;
  using Gemm = cutlass::gemm::device::Gemm<Atype, ALayout, Btype, BLayout,
                                           Ctype, CLayout, Ctype,
                                           cutlass::arch::OpClassTensorOp>;
  // cute::print_type(Gemm{});
  Gemm op;
  typename Gemm::Arguments args({m, n, k}, {A.device_data(), k},
                                {B.device_data(), k}, {C.device_data(), n},
                                {C.device_data(), n}, {alpha, beta});
  GpuTimer timer;

  CUTLASS_CHECK(op(args));
  timer.start();
  for (int iter = 0; iter < repeat; iter++) {
    op(args);
  }
  timer.stop();
  float duration_ms = timer.elapsed_millis() / repeat;
  float tflops = gflop / duration_ms;
  printf("cutlass ref: %f tflops\n", tflops);
}
/*
cublas golden compute
*/
template <typename Atype, typename ALayout, typename Btype, typename BLayout,
          typename Ctype, typename CLayout,
          typename ScalarType = typename UnderlyingType<Ctype>::type>
inline void
cublas_gemmTN_ref(cutlass::HostTensor<Atype, ALayout> const &A, // row-major
                  cutlass::HostTensor<Btype, BLayout> const &B, // col-major
                  cutlass::HostTensor<Ctype, CLayout> &C,
                  ScalarType alpha = static_cast<ScalarType>(1),
                  ScalarType beta = static_cast<ScalarType>(0),
                  int repeat = 1000) {
  int m = A.extent().row();
  int n = B.extent().column();
  int k = A.extent().column();
  using Atype_ = typename UnderlyingType<Atype>::type;
  using Btype_ = typename UnderlyingType<Btype>::type;
  using Ctype_ = typename UnderlyingType<Ctype>::type;
  float gflop = 2.0 * m * n * k / 1e9;
  cublasHandle_t handle;
  cublasCreate(&handle);
  GpuTimer timer;
  cublasStatus_t ret =
      cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                  (Btype_ *)B.device_data(), k, (Atype_ *)A.device_data(), k,
                  &beta, (Ctype_ *)C.device_data(), n);
  timer.start();
  for (int iter = 0; iter < repeat; iter++) {
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                (Btype_ *)B.device_data(), k, (Atype_ *)A.device_data(), k,
                &beta, (Ctype_ *)C.device_data(), n);
  }
  timer.stop();
  float duration_ms = timer.elapsed_millis() / repeat;
  float tflops = gflop / duration_ms;
  printf("cublas ref: %f tflops\n", tflops);

  if (ret != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Got cublas error at : " << __LINE__ << std::endl;
  }
  cublasDestroy(handle);
}

template <typename T>
__global__ static void gpu_compare_kernel(const T *x, const T *y, int n,
                                          float threshold, int *count,
                                          float *max_error) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= n) {
    return;
  }

  float v0 = x[idx];
  float v1 = y[idx];

  float diff = fabs(v0 - v1);
  if (diff > threshold) {
    atomicAdd(count, 1);

    // for positive floating point, there int representation is in the same
    // order.
    int int_diff = *((int *)(&diff));
    atomicMax((int *)max_error, int_diff);
  }
}

template <typename T>
void gpu_compare(const T *x, const T *y, size_t n, float threshold = 1e-1) {
  int *num_count;
  float *max_error;
  cudaMalloc(&num_count, sizeof(int));
  cudaMalloc(&max_error, sizeof(float));
  cudaMemset(num_count, 0, sizeof(int));
  cudaMemset(max_error, 0, sizeof(float));

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  gpu_compare_kernel<<<grid, block>>>(x, y, n, threshold, num_count, max_error);
  int num = 0;
  float error = 0;
  cudaMemcpy(&num, num_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&error, max_error, sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  if (num == 0) {
    printf_pass("check ok, max_error = %f\n", error);
  } else {
    float p = (100.f * num) / n;
    printf_fail("===============================\n");
    printf_fail("check fail: diff %.1f%% = %d/%d max_error = %f\n", p, num, n,
                error);
    printf_fail("===============================\n");
  }
  cudaFree(num_count);
  cudaFree(max_error);
}

template <typename T>
void cpu_cosine_similarity(T *x, T *y, size_t n, float threshold = 0.999) {

  float xy = 0.0f;
  float x_2 = 0.0f;
  float y_2 = 0.0f;
  for (size_t i = 0; i < n; i++) {
    auto xi = static_cast<float>(x[i]);
    auto yi = static_cast<float>(y[i]);
    xy += xi * yi;
    x_2 += xi * xi;
    y_2 += yi * yi;
  }
  float cos_similarity = xy / (std::sqrt(x_2) * std::sqrt(y_2));
  if (cos_similarity < threshold) {
    printf_fail("check fail, cos_similarity = %f\n", cos_similarity);
  } else {
    printf_pass("check ok, cos_similarity = %f\n", cos_similarity);
  }
}