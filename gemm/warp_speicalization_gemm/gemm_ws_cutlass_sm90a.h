#include "gemm_ws_kernel_sm90a.h"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/tensor_ref.h"

// #include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "utils.h"

// only for hopper arch
#if (__CUDACC_VER_MAJOR__ > 12 ||                                              \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 0))

// partial specialization
template <KernelTag tag> struct KernelScheduleTrait;

template <> struct KernelScheduleTrait<KernelTag::WASP> {
  using MainloopType = cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogType = cutlass::epilogue::TmaWarpSpecialized;
  constexpr static char kernel_name[] = "cutlass_ws_kernel";
};
template <> struct KernelScheduleTrait<KernelTag::WASP_COOP> {
  using MainloopType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogType = cutlass::epilogue::TmaWarpSpecializedCooperative;
  constexpr static char kernel_name[] = "cutlass_coop_kernel";
};
template <> struct KernelScheduleTrait<KernelTag::WASP_PIPO> {
  using MainloopType = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogType = cutlass::epilogue::TmaWarpSpecialized;
  constexpr static char kernel_name[] = "cutlass_ws_pipo_kernel";
};

template <typename KernelTraits> struct CutlassRunner {
  using ABtype = typename KernelTraits::ABtype;
  using Acctype = typename KernelTraits::Acctype;
  using Ctype = typename KernelTraits::Ctype;

  using LayoutA = typename KernelTraits::LayoutA;
  using LayoutB = typename KernelTraits::LayoutB;
  using LayoutC = typename KernelTraits::LayoutC;

  constexpr static int AlignmentA = 128 / cutlass::sizeof_bits<ABtype>::value;
  constexpr static int AlignmentB = 128 / cutlass::sizeof_bits<ABtype>::value;
  constexpr static int AlignmentC = 128 / cutlass::sizeof_bits<Ctype>::value;

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using CtaTile = typename KernelTraits::CtaTile;
  using ClusterShape = typename KernelTraits::ClusterShape;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelScheduleTrait = KernelScheduleTrait<KernelTraits::kernel_tag>;
  using KernelSchedule = typename KernelScheduleTrait::MainloopType;
  using EpilogSchedule = typename KernelScheduleTrait::EpilogType;
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, CtaTile, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, Acctype, Acctype,
          Ctype, LayoutC, AlignmentC, Ctype, LayoutC, AlignmentC,
          EpilogSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ABtype, LayoutA, AlignmentA, ABtype, LayoutB,
          AlignmentB, Acctype, CtaTile, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  void run(ABtype *A_ptr, ABtype *B_ptr, ABtype *C_ptr, int m, int n, int k,
           int swizzle = 1, int warmup = 5, int repeat = 100,
           cudaStream_t stream = 0) {

    Gemm gemm;
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});

    int device_id;
    cudaGetDevice(&device_id);
    cutlass::KernelHardwareInfo kernel_hw_info =
        cutlass::KernelHardwareInfo::make_kernel_hardware_info<
            typename Gemm::GemmKernel>(device_id);

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k},
        {A_ptr, stride_A, B_ptr, stride_B},
        {{1.0f, 0.0f}, C_ptr, stride_C, C_ptr, stride_C},
        kernel_hw_info,
        // {swizzle,
        // cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::
        //      RasterOrderOptions::AlongN}
    };

    // Using the arguments, query for extra workspace required for matrix
    // multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Check if the problem size is supported or not
    CUTLASS_CHECK(gemm.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
    auto kernel = [&] { CUTLASS_CHECK(gemm.run(stream)); };
    // Correctness / Warmup iteration
    for (int i = 0; i < warmup; i++) {
      kernel();
    }
    auto duration_ms = launch_with_timer(kernel, repeat, stream);
    float flop = 2.0 * m * n * k;
    auto tflops = compute_tflops(flop, duration_ms);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("%s:%f tflops, %f ms latency\n", KernelScheduleTrait::kernel_name,
           tflops, duration_ms);
  }
};

#else
template <typename KernelTraits> struct CutlassRunner {
  void run(ABtype *A_ptr, ABtype *B_ptr, ABtype *C_ptr, int m, int n, int k,
           int swizzle = 1, int warmup = 5, int repeat = 100,
           cudaStream_t stream = 0) {
    printf("error: CutlassRunner only support for hopper gpu!\n");
  }
}
#endif