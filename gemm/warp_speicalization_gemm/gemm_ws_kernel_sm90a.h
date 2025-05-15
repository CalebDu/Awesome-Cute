#pragma once

#include "common.h"
#include "cuda_fp16.h"
#include "gemm_ws_scheduler_sm90a.h"
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
using namespace cute;
using namespace cutlass::gemm::collective::detail;

enum class KernelTag {
  WASP = 0,      // warp specialization
  WASP_COOP = 1, // warp specialization cooperative
  WASP_PIPO = 2, // warp specialization pingpong
};

// kernel
template <KernelTag kernel_tag_, class CtaTile_, class ClusterShape_,
          int Stage_>
struct GemmKernelSM90A {

  static constexpr int WarpSize = 32;
  static constexpr int WarpGroupSize = 128;
  // default fp16, AB k-major
  using ProblemSize = Shape<int, int, int>;
  using ABtype = cutlass::half_t;
  using Acctype = float;
  using Ctype = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using StrideA = cute::Stride<int, cute::Int<1>>;
  using StrideB = cute::Stride<int, cute::Int<1>>;
  using StrideC = cute::Stride<int, cute::Int<1>>;

  static constexpr KernelTag kernel_tag = kernel_tag_;
  using CtaTile = CtaTile_;
  using ClusterShape = ClusterShape_;
  static constexpr int Stage = Stage_;
  // Epilog store tile
  using EpilogTile = std::conditional_t<kernel_tag == KernelTag::WASP_COOP,
                                        Shape<_128, _32>, Shape<_64, _32>>;
  static constexpr int EpilogStage = 4;
  // pingpong stage
  static constexpr int PipoOrderedStage = 2;
  static_assert(size(ClusterShape{}) < 4, "limit cluster size < 4");
  static_assert(size<0>(CtaTile{}) % size<0>(EpilogTile{}) == 0 &&
                    size<1>(CtaTile{}) % size<1>(EpilogTile{}) == 0,
                "cta tile must align up to [64, 32] epilog tile");
  // tiled wgmma
  static constexpr auto GmmaMajorA = gmma_ss_tag_to_major_A<ABtype, LayoutA>();
  static constexpr auto GmmaMajorB = gmma_ss_tag_to_major_B<ABtype, LayoutB>();
  static constexpr auto GmmaMajorC = cute::GMMA::Major::K;

  // ws_cooperative use 2 warp group
  using AtomLayoutMNK =
      std::conditional_t<kernel_tag == KernelTag::WASP_COOP,
                         Layout<Shape<_2, _1, _1>>, Layout<Shape<_1, _1, _1>>>;
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<ABtype, ABtype, Acctype, CtaTile, GmmaMajorA,
                                 GmmaMajorB>(),
      AtomLayoutMNK{}));

  // ping-pong use 2 tiled_mma
  static constexpr int MmaThreads_coeff =
      kernel_tag == KernelTag::WASP_PIPO ? 2 : 1;
  // thread [0:MmaThreads*MmaThreads_coeff-1] is mma thread
  static constexpr int MmaThreads = size(TiledMma{});

  // thread [MmaThreads:MmaThreads+127] is tma thread
  static constexpr int TmaThreads = WarpGroupSize;
  static constexpr int Threads = TmaThreads + MmaThreads * MmaThreads_coeff;
  static constexpr int WarpGroupCnt = Threads / WarpGroupSize;
  // tma g2s load
  using TmaG2STiledCopyA =
      decltype(sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape{})));
  using TmaG2STiledCopyB =
      decltype(sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));
  // r2s store
  using R2SCopyAtomC = Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>;
  ;
  using R2STiledCopyAtomC =
      decltype(make_tiled_copy_C_atom(R2SCopyAtomC{}, TiledMma{}));
  using R2STiledCopyC =
      decltype(make_tiled_copy_S(R2SCopyAtomC{}, R2STiledCopyAtomC{}));
  // tma s2g store
  using TmaS2GTiledCopyC = SM90_TMA_STORE;

  using SmemLayoutAtomA =
      decltype(ss_smem_selector<GmmaMajorA, ABtype,
                                decltype(cute::get<0>(CtaTile{})),
                                decltype(cute::get<2>(CtaTile{}))>());
  using SmemLayoutAtomB =
      decltype(ss_smem_selector<GmmaMajorB, ABtype,
                                decltype(cute::get<1>(CtaTile{})),
                                decltype(cute::get<2>(CtaTile{}))>());
  using SmemLayoutAtomC =
      decltype(ss_smem_selector<GmmaMajorC, Ctype,
                                decltype(cute::get<0>(EpilogTile{})),
                                decltype(cute::get<1>(EpilogTile{}))>());
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(CtaTile{}), shape<2>(CtaTile{}), Int<Stage>{}),
      Step<_1, _2, _3>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(CtaTile{}), shape<2>(CtaTile{}), Int<Stage>{}),
      Step<_1, _2, _3>{}));

  using SmemLayoutC = decltype(tile_to_shape(SmemLayoutAtomC{},
                                             make_shape(shape<0>(EpilogTile{}),
                                                        shape<1>(EpilogTile{}),
                                                        Int<EpilogStage>{}),
                                             Step<_1, _2, _3>{}));

  // pipeline barrier expect tx bytes
  static constexpr int TmaLoadABytes = cutlass::bits_to_bytes(
      size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) *
      static_cast<uint32_t>(cutlass::sizeof_bits<ABtype>::value));

  static constexpr int TmaLoadBBytes = cutlass::bits_to_bytes(
      size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) *
      static_cast<uint32_t>(cutlass::sizeof_bits<ABtype>::value));
  static constexpr int TmaLoadTotalBytes = TmaLoadABytes + TmaLoadBBytes;
  // pipeline barrier arrive count
  static constexpr int ConsumerArvCnt =
      size(TiledMma{}) * size(ClusterShape{}) / WarpSize;
  static constexpr int ProducerArvCnt = 1;
  // ping-pong barrier arrive count
  static constexpr int PipoArvCnt = WarpGroupSize;
  // pipeline barrier type
  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  using PipoOrderedBarrier = cutlass::arch::ClusterBarrier;
  static constexpr int orderedSequence = 2; // 2 consumer sequentially execution
  // scheduler
  using Scheduler = PersistentTileScheduler;
  using SchedulerParam = Scheduler::Param;
  using TileInfo = Scheduler::TileInfo;
  struct Args {
    ProblemSize problem_size; // mnk
    ABtype *A_ptr, *B_ptr;
    Ctype *C_ptr;
    int swizzle{1};
  };

  struct Param {
    using TMA_A = decltype(make_tma_copy_A_sm90(
        TmaG2STiledCopyA{},
        make_tensor(static_cast<ABtype const *>(nullptr),
                    repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_, _, _0{}), CtaTile{}, ClusterShape{}));
    using TMA_B = decltype(make_tma_copy_B_sm90(
        TmaG2STiledCopyB{},
        make_tensor(static_cast<ABtype const *>(nullptr),
                    repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_, _, _0{}), CtaTile{}, ClusterShape{}));
    using TMA_C = decltype(make_tma_copy_C_sm90(
        TmaS2GTiledCopyC{},
        make_tensor(make_gmem_ptr<Ctype>(nullptr),
                    repeat_like(StrideC{}, int32_t(0)), StrideC{}),
        SmemLayoutC{}(_, _, _0{}), EpilogTile{}));
    ProblemSize problem_size; // mnk
    SchedulerParam scheduler;
    TMA_A tma_a;
    TMA_B tma_b;
    TMA_C tma_c;
  };
  template <int kStage = Stage> struct PipelineState {
    uint32_t phase;
    uint32_t stage_idx;
    uint32_t count{0};

    DEVICE
    void operator++(int) {
      count += 1;
      if ((++stage_idx) == kStage) {
        phase ^= 1;
        stage_idx = 0;
      }
    }
    DEVICE void advance(uint32_t num_iterations) {
      if constexpr (kStage > 0) {
        // Number of iterations cross over the stage boundary => flipped phase
        if ((num_iterations < kStage) &&
            (stage_idx + num_iterations) >= kStage) {
          phase ^= 1;
        }
        // How many times number of iterations cross over the stage boundary and
        // end up on a odd number => flipped phase
        if ((num_iterations >= kStage) &&
            (((stage_idx + num_iterations) / kStage) % 2) == 1) {
          phase ^= 1;
        }
        stage_idx = (stage_idx + num_iterations) % kStage;
        count += num_iterations;
      }
    }
  };
  struct alignas(128) SharedStorage {
    struct alignas(128) TensorStorage {
      cute::array_aligned<ABtype, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array_aligned<ABtype, cute::cosize_v<SmemLayoutB>> smem_B;
      cute::array_aligned<Ctype, cute::cosize_v<SmemLayoutC>> smem_C;
    } tensors;
    struct alignas(16) PipelineStorage {
      // mainloop pipeline barrier
      FullBarrier mainloop_full_bar[Stage];
      EmptyBarrier mainloop_empty_bar[Stage];
      // 2stage consumer pingpong barrier
      PipoOrderedBarrier pipo_mainloop_bar[PipoOrderedStage];
      PipoOrderedBarrier pipo_epilog_bar[PipoOrderedStage];
    } pipelines;
  };

  static Param initialize_param(Args const &args) {

    auto [m, n, k] = args.problem_size;
    Tensor A_tensor = make_tensor(make_gmem_ptr<ABtype const>(args.A_ptr),
                                  make_layout(make_shape(m, k), LayoutRight{}));
    Tensor B_tensor = make_tensor(make_gmem_ptr<ABtype const>(args.B_ptr),
                                  make_layout(make_shape(n, k), LayoutRight{}));
    Tensor C_tensor = make_tensor(make_gmem_ptr<Ctype>(args.C_ptr),
                                  make_layout(make_shape(m, n), LayoutRight{}));

    typename Param::TMA_A tma_a = make_tma_copy_A_sm90(
        TmaG2STiledCopyA{}, A_tensor, SmemLayoutA{}(_, _, _0{}), CtaTile{},
        ClusterShape{});
    typename Param::TMA_B tma_b = make_tma_copy_B_sm90(
        TmaG2STiledCopyB{}, B_tensor, SmemLayoutB{}(_, _, _0{}), CtaTile{},
        ClusterShape{});
    typename Param::TMA_C tma_c = make_tma_copy_C_sm90(
        TmaS2GTiledCopyC{}, C_tensor, SmemLayoutC{}(_, _, _0{}), EpilogTile{});

    auto scheduler_param = SchedulerParam(args.problem_size, CtaTile{},
                                          ClusterShape{}, args.swizzle);
    Param param{args.problem_size, scheduler_param, tma_a, tma_b, tma_c};
    return param;
  }

  DEVICE
  void operator()(Param const &param, char *smem) {
#if __CUDA_ARCH_FEAT_SM90_ALL
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem);

    auto thread_idx = threadIdx.x;
    auto block_idx = blockIdx.x;
    auto lane_idx = thread_idx & 31;
    auto warp_idx = __shfl_sync(0xffffffff, thread_idx / WarpSize, 0);
    auto warp_idx_in_group = __shfl_sync(0xffffffff, warp_idx % 4, 0);

    auto block_rank_in_cluster = cute::block_rank_in_cluster();
    // warp group0: consumer
    // warp group1(2): producer
    auto warp_group_idx =
        __shfl_sync(0xffffffff, thread_idx / WarpGroupSize, 0);
    auto consumer_thread_idx = thread_idx % MmaThreads;
    int lane_predicate = cute::elect_one_sync();

    if (warp_idx == 0 && lane_predicate) {
      // prefetch tma tensormap
      cute::prefetch_tma_descriptor(param.tma_a.get_tma_descriptor());
      cute::prefetch_tma_descriptor(param.tma_b.get_tma_descriptor());
      cute::prefetch_tma_descriptor(param.tma_c.get_tma_descriptor());

      // initialize barrier
#pragma unroll
      for (int i = 0; i < Stage; i++) {
        shared_storage.pipelines.mainloop_full_bar[i].init(ProducerArvCnt);
        shared_storage.pipelines.mainloop_empty_bar[i].init(ConsumerArvCnt);
      }
#pragma unroll
      for (int i = 0; i < PipoOrderedStage; i++) {
        shared_storage.pipelines.pipo_mainloop_bar[i].init(PipoArvCnt);
        shared_storage.pipelines.pipo_epilog_bar[i].init(PipoArvCnt);
      }
    }
    // visiablity for barrier
    cutlass::arch::fence_barrier_init();

    // sync for barrier initialization
    if constexpr (size(ClusterShape{}) > 1) {
      cute::cluster_arrive_relaxed();
      cute::cluster_wait();
    } else {
      __syncthreads();
    }

    // WASP: consumer wg0, producer wg1
    // WASP_COOP: consumer wg0 wg1, producer wg2
    // WASP_PIPO: consumer wg0 wg1, producer wg2
    if (warp_group_idx == WarpGroupCnt - 1) {
      // producer
      // alloc 40 register for tma load
      cutlass::arch::warpgroup_reg_dealloc<40>();
      // elect 1 thread issue tma load
      if (warp_idx_in_group == 0 && elect_one_sync()) {
        producer(param, shared_storage, block_rank_in_cluster);
      }
    } else {
      // consumer
      // alloc 232 register for mma compute
      cutlass::arch::warpgroup_reg_alloc<232>();

      if constexpr (kernel_tag == KernelTag::WASP ||
                    kernel_tag == KernelTag::WASP_COOP) {
        ws_consumer(param, shared_storage);
      } else if constexpr (kernel_tag == KernelTag::WASP_PIPO) {
        ws_pipo_consumer(param, shared_storage);
      }
    }
#else
    if (thread0()) {
      PRINT("error: kernel only for hopper gpu!\n");
    }
#endif
  }

  DEVICE void producer(Param const &param, SharedStorage &shared_storage,
                       uint32_t block_rank_in_cluster) {

    auto [m, n, k] = param.problem_size;
    Tensor sA = make_tensor(make_smem_ptr(shared_storage.tensors.smem_A.data()),
                            SmemLayoutA{}); // (BLK_M,BLK_K,Stage)
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.tensors.smem_B.data()),
                            SmemLayoutB{}); // (BLK_N,BLK_K,Stage)

    uint2 cluster_idx = {
        block_rank_in_cluster % param.scheduler.cluster_m_shape,
        block_rank_in_cluster / param.scheduler.cluster_m_shape};

    Tensor A = param.tma_a.get_tma_tensor(make_shape(m, k));
    Tensor B = param.tma_b.get_tma_tensor(make_shape(n, k));
    Tensor gA_mk = local_tile(A, CtaTile{}, make_coord(_, _, _),
                              Step<_1, X, _1>{}); // (BLK_M,BLK_K,m,k)
    Tensor gB_nk = local_tile(B, CtaTile{}, make_coord(_, _, _),
                              Step<X, _1, _1>{}); // (BLK_N,BLK_K,n,k)
    Scheduler scheduler(param.scheduler);
    auto tile_info = scheduler.get_tile_id();

    auto g2s_tma_a = param.tma_a.get_slice(cluster_idx.y);
    auto g2s_tma_b = param.tma_b.get_slice(cluster_idx.x);
    auto cluster_layout = make_layout(ClusterShape{});

    // init tma multicast mask
    uint16_t mcast_mask_a = 0, mcast_mask_b = 0;
    if constexpr (is_same_v<TmaG2STiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
#pragma unroll
      for (int n = 0; n < size<1>(cluster_layout); n++) {
        mcast_mask_a |= (static_cast<uint16_t>(1)
                         << cluster_layout(cluster_idx.x, n, _0{}));
      }
    }

    if constexpr (is_same_v<TmaG2STiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
#pragma unroll
      for (int m = 0; m < size<0>(cluster_layout); m++) {
        mcast_mask_b |= (static_cast<uint16_t>(1)
                         << cluster_layout(m, cluster_idx.y, _0{}));
      }
    }
    // init pipeline states
    PipelineState<Stage> pipeline_states{1, 0};

    while (tile_info.is_valid) {

      // print("bidx %d tile id [%d %d]\n", blockIdx.x, tile_info.m_idx,
      //       tile_info.n_idx);

      Tensor gA = gA_mk(_, _, tile_info.m_idx, _); //(BLK_M,BLK_K,k)
      Tensor gB = gB_nk(_, _, tile_info.n_idx, _); //(BLK_n,BLK_K,k)

      Tensor tAgA = g2s_tma_a.partition_S(gA);
      Tensor tAsA = g2s_tma_a.partition_D(sA);

      Tensor tBgB = g2s_tma_b.partition_S(gB);
      Tensor tBsB = g2s_tma_b.partition_D(sB);

      int k_loop_cnt = size<2>(gA);

#pragma unroll 1
      for (int k_idx = 0; k_idx < k_loop_cnt; k_idx++) {
        uint64_t *full_barrier_ptr = reinterpret_cast<uint64_t *>(
            &shared_storage.pipelines
                 .mainloop_full_bar[pipeline_states.stage_idx]);
        // wait consumer
        shared_storage.pipelines.mainloop_empty_bar[pipeline_states.stage_idx]
            .wait(pipeline_states.phase);
        // notify consumer
        shared_storage.pipelines.mainloop_full_bar[pipeline_states.stage_idx]
            .arrive_and_expect_tx(TmaLoadTotalBytes);
        // tma load
        copy(param.tma_a.with(*full_barrier_ptr, mcast_mask_a),
             tAgA(_, _, _, k_idx), tAsA(_, _, _, pipeline_states.stage_idx));
        copy(param.tma_b.with(*full_barrier_ptr, mcast_mask_b),
             tBgB(_, _, _, k_idx), tBsB(_, _, _, pipeline_states.stage_idx));

        // update pipeline states
        pipeline_states++;
      }
      // next tile
      scheduler.advance_next_tile();
      tile_info = scheduler.get_tile_id();
    }
    // load tail: make sure all consumers have use data
#pragma unroll
    for (int i = 0; i < Stage; i++) {
      shared_storage.pipelines.mainloop_empty_bar[pipeline_states.stage_idx]
          .wait(pipeline_states.phase);
      pipeline_states++;
    }
#if debug_pipeline
    print("bidx %d producer quit\n", blockIdx.x);
#endif
  }

  // consumer func for wasp_pipo
  DEVICE void ws_pipo_consumer(Param const &param,
                               SharedStorage &shared_storage) {
    Scheduler scheduler(param.scheduler);
    auto warp_group_idx =
        __shfl_sync(0xffffffff, threadIdx.x / WarpGroupSize, 0);
    auto other_warp_group_idx =
        __shfl_sync(0xffffffff, ((warp_group_idx + 1) & 1), 0);
    // wg0 phase = 1, wg1 phase = 0 (wait wg0)
    PipelineState<PipoOrderedStage> pipo_states{warp_group_idx == 0, 0};
    PipelineState<Stage> pipeline_states{0, 0};
    // wg1 advance next tile
    if (warp_group_idx != 0) {
      scheduler.advance_next_tile();
      pipeline_states.advance(param.scheduler.k_loop_cnt);
    }
    auto tile_info = scheduler.get_tile_id();
    TiledMma tiled_mma;
    Tensor acc = partition_fragment_C(tiled_mma, take<0, 2>(CtaTile{}));
    uint32_t thread_idx_in_mma = threadIdx.x % MmaThreads;
    while (tile_info.is_valid) {
      // 2 wg orderedly execute mainloop
      shared_storage.pipelines.pipo_mainloop_bar[warp_group_idx].wait(
          pipo_states.phase);

      issue_mma(param, shared_storage, acc, pipeline_states, tiled_mma);
      // notify other wg execute mainloop
      shared_storage.pipelines.pipo_mainloop_bar[other_warp_group_idx].arrive();
      pipo_states++; // stage_idx ++
      mma_tail(shared_storage, pipeline_states);
      // pipeline state extra advance k_loop_cnt
      pipeline_states.advance(param.scheduler.k_loop_cnt);
      // 2 wg orderedly execute epilog
      shared_storage.pipelines.pipo_epilog_bar[warp_group_idx].wait(
          pipo_states.phase);
      issue_epilog(param, shared_storage, tile_info, acc, thread_idx_in_mma);
      // notify other wg execute epilog
      shared_storage.pipelines.pipo_epilog_bar[other_warp_group_idx].arrive();
      pipo_states++; // stage_idx ++, phase reverse
      // advance 2 tiles
      scheduler.advance_next_tile();
      scheduler.advance_next_tile();
      tile_info = scheduler.get_tile_id();
    }
  }

  // consumer func for wasp/wasp_coop
  DEVICE void ws_consumer(Param const &param, SharedStorage &shared_storage) {
    Scheduler scheduler(param.scheduler);
    auto tile_info = scheduler.get_tile_id();

    TiledMma tiled_mma;
    Tensor acc = partition_fragment_C(tiled_mma, take<0, 2>(CtaTile{}));
    PipelineState<Stage> pipeline_states{0, 0};
    uint32_t thread_idx_in_mma = threadIdx.x % MmaThreads;
    while (tile_info.is_valid) {
      issue_mma(param, shared_storage, acc, pipeline_states, tiled_mma);
      mma_tail(shared_storage, pipeline_states);
      issue_epilog(param, shared_storage, tile_info, acc, thread_idx_in_mma);
      scheduler.advance_next_tile();
      tile_info = scheduler.get_tile_id();
    }
  }

  template <typename AccTensor>
  DEVICE void issue_epilog(Param const &param, SharedStorage &shared_storage,
                           TileInfo &tile_info, AccTensor &acc,
                           uint32_t thread_idx_in_mma) {
    auto [m, n, k] = param.problem_size;
    Tensor sC_ =
        make_tensor(make_smem_ptr<Ctype>(shared_storage.tensors.smem_C.data()),
                    SmemLayoutC{});
    Tensor sC = as_position_independent_swizzle_tensor(sC_);
    Tensor C = param.tma_c.get_tma_tensor(make_shape(m, n));

    Tensor gC = local_tile(
        C, take<0, 2>(CtaTile{}),
        make_coord(tile_info.m_idx, tile_info.n_idx)); // (CTA_M,CTA_N)
    Tensor gC_epilog =
        flat_divide(gC, EpilogTile{}); // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

    auto r2s_copy_c = R2STiledCopyC{};
    auto thr_r2s_copy_c = r2s_copy_c.get_slice(thread_idx_in_mma);
    auto r2s_rC = thr_r2s_copy_c.retile_S(acc);   //((R2S,R2S_V),MMA_M,MMA_N)
    auto r2s_sC = thr_r2s_copy_c.partition_D(sC); // (R2S,R2S_M,R2S_N,PIPE)
    auto r2s_rC_frag =
        make_tensor<Ctype>(make_layout(r2s_sC(_, _, _, _0{}).shape()));

    auto thr_s2g_tma_copy_c = param.tma_c.get_slice(_0{});
    auto s2g_sC = thr_s2g_tma_copy_c.partition_S(sC); //(S2G, )
    auto s2g_gC = thr_s2g_tma_copy_c.partition_D(gC_epilog);
    constexpr auto mma_tile_m = size<0>(CtaTile{}) / size<1>(r2s_rC);
    constexpr auto mma_tile_n = size<1>(CtaTile{}) / size<2>(r2s_rC);
    constexpr auto epilog_tile_m = size<0>(EpilogTile{});
    constexpr auto epilog_tile_n = size<1>(EpilogTile{});

    auto is_issue_tma = thread_idx_in_mma == 0;

    uint32_t epilog_m_loop = size<2>(gC_epilog);
    uint32_t epilog_n_loop = size<3>(gC_epilog);
    PipelineState<EpilogStage> pipeline_states{0, 0, 0};

#if 0
    if (thread0()) {
      PRINT(epilog_m_loop) PRINT(epilog_n_loop);
      PRINT(mma_tile_m) PRINT(mma_tile_n) PRINT(epilog_tile_m);
      PRINT(epilog_tile_n) PRINT(acc) PRINT(sC) PRINT(gC);
      PRINT(gC_epilog) PRINT(r2s_copy_c) PRINT(r2s_rC);
      PRINT(r2s_rC_frag) PRINT(r2s_sC) PRINT(s2g_sC);
      PRINT(s2g_gC);
    }
#endif

#pragma unroll
    for (int epilog_m = 0; epilog_m < epilog_m_loop; epilog_m++) {
#pragma unroll
      for (int epilog_n = 0; epilog_n < epilog_n_loop; epilog_n++) {
        int mma_m = epilog_m;
        int mma_n = epilog_n * epilog_tile_n / mma_tile_m;
        int reg_offset =
            (epilog_n % (mma_tile_n / epilog_tile_n)) * size(r2s_rC_frag);

        Tensor cur_r2s_rC = r2s_rC(_, mma_m, mma_n);
        // convert acc fp32 to fp16 r2r copy
#pragma unroll
        for (int i = 0; i < size(r2s_rC_frag); i++) {
          r2s_rC_frag(i) = __float2half(cur_r2s_rC(reg_offset + i));
        }

        if (is_issue_tma) {
          // wait smem available
          if (pipeline_states.count > EpilogStage - 1) {
            // keep EpilogStage-1 tma in-flight
            tma_store_wait<EpilogStage - 1>();
          }
        }
        cutlass::arch::NamedBarrier(MmaThreads).sync();
        // r2s copy
        copy(r2s_copy_c, r2s_rC_frag,
             r2s_sC(_, _, _, pipeline_states.stage_idx));
        // fence for visiblity
        cutlass::arch::fence_view_async_shared();
        // sync all consumer threads in tiled mma 
        cutlass::arch::NamedBarrier(MmaThreads).sync();
        if (is_issue_tma) {
          // issue tma s2g store
          copy(param.tma_c, s2g_sC(_, _, _, pipeline_states.stage_idx),
               s2g_gC(_, _, _, epilog_m, epilog_n));
          // commit s2g tma
          tma_store_arrive();
        }
        pipeline_states++;
      }
    }
    if (is_issue_tma) {
      // store tail wait all tma store completion
      tma_store_wait<0>();
    }
  }

  template <typename TensorAcc, typename TiledMma>
  DEVICE void issue_mma(Param const &param, SharedStorage &shared_storage,
                        TensorAcc &acc, PipelineState<Stage> &pipeline_states,
                        TiledMma &tiled_mma) {
    constexpr int MmaWarpGroups = size(TiledMma{}) / WarpGroupSize;
    Layout warp_group_thread_layout =
        make_layout(Int<MmaWarpGroups>{}, Int<WarpGroupSize>{});
    int warp_group_idx =
        __shfl_sync(0xffffffff, threadIdx.x / WarpGroupSize, 0);
    auto thread_mma =
        tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    Tensor sA = make_tensor(make_smem_ptr(shared_storage.tensors.smem_A.data()),
                            SmemLayoutA{}); // (BLK_M,BLK_K,Stage)
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.tensors.smem_B.data()),
                            SmemLayoutB{});   // (BLK_N,BLK_K,Stage)
    Tensor tCsA = thread_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB); // (MMA,MMA_N,MMA_K,PIPE)

    uint32_t lane_idx = threadIdx.x & 31;
    uint32_t target_cta = lane_idx;
    // lane_id thread notify cluster_id cta barrier
    uint32_t pred_arrive = lane_idx < size(ClusterShape{});
    // fisrt mma with no accumulation to avoid init zeros
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    auto pipeline_states_read = pipeline_states;
    // Prologue mma
    warpgroup_fence_operand(acc);
    {
      // wait producer
      bool wait_complete =
          shared_storage.pipelines
              .mainloop_full_bar[pipeline_states_read.stage_idx]
              .try_wait(pipeline_states_read.phase);
      if (!wait_complete) {
        shared_storage.pipelines
            .mainloop_full_bar[pipeline_states_read.stage_idx]
            .wait(pipeline_states_read.phase);
      }

      warpgroup_arrive();
#pragma unroll
      for (int k_inner = 0; k_inner < size<2>(tCrA); k_inner++) {
        gemm(tiled_mma, tCrA(_, _, k_inner, pipeline_states_read.stage_idx),
             tCrB(_, _, k_inner, pipeline_states_read.stage_idx), acc);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      pipeline_states_read++;
    }
    warpgroup_fence_operand(acc);

#pragma unroll 1
    for (int k_idx = 1; k_idx < param.scheduler.k_loop_cnt; k_idx++) {
      // wait producer
      shared_storage.pipelines.mainloop_full_bar[pipeline_states_read.stage_idx]
          .wait(pipeline_states_read.phase);
      warpgroup_fence_operand(acc);
      warpgroup_arrive();
      gemm(tiled_mma, tCrA(_, _, _, pipeline_states_read.stage_idx),
           tCrB(_, _, _, pipeline_states_read.stage_idx), acc);
      warpgroup_commit_batch();
      // keep 1 wgmma commit in-flight
      warpgroup_wait<1>();
      warpgroup_fence_operand(acc);
      // notify producer
      shared_storage.pipelines.mainloop_empty_bar[pipeline_states.stage_idx]
          .arrive(target_cta, pred_arrive);
      // update pipeline states
      pipeline_states++;
      pipeline_states_read++;
    }
    warpgroup_fence_operand(acc);
  }

  DEVICE void mma_tail(SharedStorage &shared_storage,
                       PipelineState<Stage> &pipeline_states) {
    uint32_t lane_idx = threadIdx.x & 31;
    uint32_t target_cta = lane_idx;
    uint32_t pred_arrive = lane_idx < size(ClusterShape{});
    // wait last wgmma commit completion, release barrier
    warpgroup_wait<0>();
    shared_storage.pipelines.mainloop_empty_bar[pipeline_states.stage_idx]
        .arrive(target_cta, pred_arrive);
    pipeline_states++;
  }

  inline static cudaLaunchConfig_t get_launch_config(cudaStream_t stream = 0) {
    static cudaLaunchConfig_t launch_config;
    static cudaLaunchAttribute launch_attr;
    launch_config.gridDim = Scheduler::get_grid_dim(ClusterShape{});
    launch_config.blockDim = {Threads};
    launch_config.dynamicSmemBytes = sizeof(SharedStorage);
    launch_config.stream = stream;

    // cluster shape
    launch_attr.id = cudaLaunchAttributeClusterDimension;
    launch_attr.val.clusterDim = {size<0>(ClusterShape{}),
                                  size<1>(ClusterShape{}),
                                  size<2>(ClusterShape{})};

    launch_config.numAttrs = 1;
    launch_config.attrs = &launch_attr;

    return launch_config;
  }
};