#include "common.h"
#include "cute/tensor.hpp"
#include "gemm_streamk.h"
#include "reference.h"

using type = cutlass::half_t;

int main(int argc, const char *argv[]) {
  int m = 4096, n = 4096, k = 1024;
  int warmup = 5, repeat = 10000;
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
  // fill all 1 for debug
  // cutlass::reference::host::TensorFill(A_tensor.host_view(), type(1));
  // cutlass::reference::host::TensorFill(B_tensor.host_view(), type(1));
  A_tensor.sync_device();
  B_tensor.sync_device();

  cublas_gemmTN_ref(A_tensor, B_tensor, C_ref_tensor, alpha.to_half(),
                    beta.to_half(), repeat, stream);
  auto run = [&](auto gemm_traits,
                 SK_DP_Block_Strategy strategy =
                     SK_DP_Block_Strategy::sk1tile_dp,
                 std::string kernel_name = "kernel") {
    using Traits = decltype(gemm_traits);
    using Arguments = typename Traits::Arguments;
    constexpr int smem_size = Traits::kAllSmemSize;
    constexpr int block_size = Traits::kThread;
    config_smem(gemmTN_streamk_dp<Traits>, smem_size);
    int device_sm = get_device_sm();
    int sm_occupancy =
        get_sm_occupancy(gemmTN_streamk_dp<Traits>, block_size, smem_size);
    printf("launch kernel: %s:\n", kernel_name.c_str());
    printf("sm occupancy:%d, shared mem size: %.1fKiB\n", sm_occupancy,
           smem_size / 1e3);
    Arguments args(make_shape(m, n, k), A_tensor.device_data(),
                   B_tensor.device_data(), C_tensor.device_data(), sm_occupancy,
                   device_sm, device_sm, strategy);
    args.init_workspace(stream); // init workspace for partial sum

    dim3 cta(block_size);
    dim3 grid(args.get_grid_dims());

    auto kernel = [&] {
      gemmTN_streamk_dp<Traits><<<grid, cta, smem_size, stream>>>(args);
    };
    for (int i = 0; i < warmup; i++) {
      kernel();
    }
    auto duration_ms = launch_with_timer(kernel, repeat, stream);
    float flop = 2.0 * m * n * k;
    auto tflops = compute_tflops(flop, duration_ms);
    printf("%f tflops, %f ms latency\n", tflops, duration_ms);
    args.free_workspace();
    C_tensor.sync_host();
    C_ref_tensor.sync_host();
    cpu_cosine_similarity(C_tensor.host_data(), C_ref_tensor.host_data(),
                          C_ref_tensor.capacity());
  };
  run(GemmTraits<decltype(make_shape(_128{}, _128{}, _32{})), 3>{},
      SK_DP_Block_Strategy::sk1tile_dp,
      "gemm_streamk_1sk_dp_cta_128*128*32_stage3");
  run(GemmTraits<decltype(make_shape(_128{}, _128{}, _32{})), 3>{},
      SK_DP_Block_Strategy::sk2tile_dp,
      "gemm_streamk_2sk_dp_cta_128*128*32_stage3");

  run(GemmTraits<decltype(make_shape(_128{}, _256{}, _32{})), 3>{},
      SK_DP_Block_Strategy::sk1tile_dp,
      "gemm_streamk_1sk_dp_cta_128*256*32_stage3");
  run(GemmTraits<decltype(make_shape(_128{}, _256{}, _32{})), 3>{},
      SK_DP_Block_Strategy::sk2tile_dp,
      "gemm_streamk_2sk_dp_cta_128*256*32_stage3");

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
