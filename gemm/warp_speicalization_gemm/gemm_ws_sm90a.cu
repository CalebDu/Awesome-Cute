#include "gemm_ws_cutlass_sm90a.h"
#include "gemm_ws_kernel_sm90a.h"

#include "common.h"
#include "reference.h"
using type = cutlass::half_t;

template <typename Kernel>
__global__ void launch_kernel(__grid_constant__
                              typename Kernel::Param const param) {
  extern __shared__ char smem[];
  Kernel kernel;
  kernel(param, smem);
}

int main(int argc, const char *argv[]) {
  int m = 4096, n = 4096, k = 1024, swizzle = 4;
  int warmup = 5, repeat = 100;
  type alpha{1.0f};
  type beta{0.0f};

  if (argc > 1) {
    m = atoi(argv[1]);
  }
  if (argc > 2) {
    n = atoi(argv[2]);
  }
  if (argc > 3) {
    k = atoi(argv[3]);
  }
  if (argc > 4) {
    swizzle = atoi(argv[4]);
  }

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  printf("fp16 gemm m:%d n:%d k:%d cta_swizzle:%d\n", m, n, k, swizzle);
  // init
  auto A_tensor = make_cutlass_rowmajor_tensor<type>(m, k);
  auto B_tensor = make_cutlass_colmajor_tensor<type>(k, n);
  auto C_tensor = make_cutlass_rowmajor_tensor<type>(m, n);
  auto C_cublas_tensor = make_cutlass_rowmajor_tensor<type>(m, n);
  auto C_cutlass_tensor = make_cutlass_rowmajor_tensor<type>(m, n);

  cutlass::reference::host::TensorFillRandomUniform(A_tensor.host_view(), 0, -2,
                                                    2);
  cutlass::reference::host::TensorFillRandomUniform(B_tensor.host_view(), 0, -2,
                                                    2);
  // H2D copy
  A_tensor.sync_device();
  B_tensor.sync_device();

  auto run = [&](auto gemm_kernel, std::string kernel_name = "kernel") {
    using Kernel = decltype(gemm_kernel);
    using CutlassKernel = CutlassRunner<Kernel>;
    CutlassKernel cutlass_kernel;
    cutlass_kernel.run(A_tensor.device_data(), B_tensor.device_data(),
                       C_cutlass_tensor.device_data(), m, n, k, swizzle, warmup,
                       repeat, stream);
    auto launch_config = Kernel::get_launch_config(stream);
    typename Kernel::Args args = {make_shape(m, n, k), A_tensor.device_data(),
                                  B_tensor.device_data(),
                                  C_tensor.device_data(), swizzle};
    auto param = Kernel::initialize_param(args);

    config_smem(launch_kernel<Kernel>, launch_config.dynamicSmemBytes);
    auto kernel = [&] {
      CUDA_CHECK(
          cudaLaunchKernelEx(&launch_config, launch_kernel<Kernel>, param));
    };
    for (int i = 0; i < warmup; i++) {
      kernel();
    }

    auto duration_ms = launch_with_timer(kernel, repeat, stream);
    float flop = 2.0 * m * n * k;
    auto tflops = compute_tflops(flop, duration_ms);
    printf("%s %f tflops, %f ms latency\n", kernel_name.c_str(), tflops,
           duration_ms);
    CUDA_CHECK(cudaDeviceSynchronize());
    C_tensor.sync_host();
    C_cutlass_tensor.sync_host();

    cpu_cosine_similarity(C_tensor.host_data(), C_cutlass_tensor.host_data(),
                          C_cutlass_tensor.capacity(), 0.99);
  };
  run(GemmKernelSM90A<KernelTag::WASP, Shape<_128, _128, _64>,
                      Shape<_2, _1, _1>, 6>{},
      "my_ws_kernel");
  run(GemmKernelSM90A<KernelTag::WASP_COOP, Shape<_128, _128, _64>,
                      Shape<_2, _1, _1>, 6>{},
      "my_ws_coop_kernel");

  run(GemmKernelSM90A<KernelTag::WASP_PIPO, Shape<_128, _128, _64>,
                      Shape<_2, _1, _1>, 6>{},
      "my_ws_pipo_kernel");

  cublas_gemmTN_ref(A_tensor, B_tensor, C_cublas_tensor, alpha.to_half(),
                    beta.to_half(), repeat, stream);
  C_cublas_tensor.sync_host();
  // std::cout << "result:" << std::endl << C_tensor.host_view() << std::endl;
  // std::cout << "ref:" << std::endl << C_cublas_tensor.host_view() <<
  // std::endl;
  return 0;
}