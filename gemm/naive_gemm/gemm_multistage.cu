#include "common.h"
#include "cute/tensor.hpp"
#include "gemm_multistage.h"
#include "reference.h"

using type = cutlass::half_t;

int main(int argc, const char *argv[]) {
  int m = 128, n = 128, k = 32;
  if (argc > 1) {
    m = atoi(argv[1]);
  }
  if (argc > 2) {
    n = atoi(argv[2]);
  }
  if (argc > 3) {
    k = atoi(argv[3]);
  }
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  printf("fp16 gemm m:%d n:%d k:%d\n", m, n, k);
  // init
  auto A_tensor = make_cutlass_rowmajor_tensor<type>(m, k);
  auto B_tensor = make_cutlass_colmajor_tensor<type>(k, n);
  auto C_tensor = make_cutlass_rowmajor_tensor<type>(m, n);
  auto C_ref_tensor = make_cutlass_rowmajor_tensor<type>(m, n);

  cutlass::reference::host::TensorFillRandomUniform(A_tensor.host_view(), 0, -2,
                                                    2);
  cutlass::reference::host::TensorFillRandomUniform(B_tensor.host_view(), 0, -2,
                                                    2);
  // cutlass::reference::host::TensorFill(A_tensor.host_view(), type(1));
  // cutlass::reference::host::TensorFill(B_tensor.host_view(), type(1));
  A_tensor.sync_device();
  B_tensor.sync_device();

  cublas_gemmTN_ref(A_tensor, B_tensor, C_ref_tensor);
  auto run = [&](auto gemm_traits, std::string kernel_name = "kernel") {
    int warmup = 5;
    using Traits = decltype(gemm_traits);
    dim3 cta(size(typename Traits::MMA{}));
    dim3 grid(ceil_div(m, Traits::kCTAM), ceil_div(n, Traits::kCTAN));
    int smem_size = Traits::kAllSmemSize;
    config_smem(gemmTN_multistage<Traits>, smem_size);

    auto kernel = [&] {
      gemmTN_multistage<Traits><<<grid, cta, smem_size, stream>>>(
          A_tensor.device_data(), B_tensor.device_data(),
          C_tensor.device_data(), m, n, k);
    };
    for (int i = 0; i < warmup; i++) {
      kernel();
    }
    auto duration_ms = launch_with_timer(kernel,1000, stream);
    float flop = 2.0 * m * n * k;
    auto tflops = compute_tflops(flop, duration_ms);
    printf("%s: %f tflops, %f ms latency\n", kernel_name.c_str(), tflops,
           duration_ms);
  };
  run(GemmTraits<decltype(make_shape(_128{}, _128{}, _32{})), 3>{});

  cudaDeviceSynchronize();
  C_tensor.sync_host();
  C_ref_tensor.sync_host();
  cpu_cosine_similarity(C_tensor.host_data(), C_ref_tensor.host_data(),
                        C_ref_tensor.capacity());
  cudaStreamDestroy(stream);

  // auto A_cute = make_tensor(make_gmem_ptr<type>(A_tensor.host_data()),
  //                           make_shape(m, k), make_stride(k, _1{}));
  // auto B_cute = make_tensor(make_gmem_ptr<type>(B_tensor.host_data()),
  //                           make_shape(n, k), make_stride(k, _1{}));
  // print("\nA\n");
  // print_tensor(A_cute);
  // print("\nB\n");
  // print_tensor(B_cute);
  // std::cout << "result:" << std::endl << C_tensor.host_view() << std::endl;
  // std::cout << "ref:" << std::endl << C_ref_tensor.host_view() << std::endl;
  return 0;
}
