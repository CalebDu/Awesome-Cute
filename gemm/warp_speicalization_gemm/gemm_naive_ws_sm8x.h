// reference: https://github.com/reed-lau/cute-gemm
#pragma once
#include "common.h"
#include "cooperative_groups.h"
#include "cuda/pipeline"
#include "cute/tensor.hpp"
#include <cooperative_groups/memcpy_async.h>

using namespace cute;

template <int Producer, class CTA_tile, int Stage, bool Bound_Check = false>
struct GemmTraits {
  // fp16 example
  using ABtype = cutlass::half_t;
  using Ctype = cutlass::half_t;
  using Gemm_Shape = Shape<int, int, int>;
  static constexpr int kCTAM = size<0>(CTA_tile{});
  static constexpr int kCTAN = size<1>(CTA_tile{});
  static constexpr int kCTAK = size<2>(CTA_tile{});
  static constexpr int kStage = Stage;
  static constexpr bool kBound_Check = Bound_Check;
  // mma
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  // tiled mma shape[16*2, 8*2*2, 16], warp layout [2, 2, 1]
  static constexpr int kMmaThrLayoutM = 2;
  static constexpr int kMmaThrLayoutN = 2;
  static constexpr int kMmaThrLayoutK = 1;
  using mma_atom_shape = mma_traits::Shape_MNK;

  using MmaThrLayout = decltype(make_layout(make_shape(
      Int<kMmaThrLayoutM>{}, Int<kMmaThrLayoutN>{}, Int<kMmaThrLayoutK>{})));
  static constexpr int kMmaPermuteM =
      kMmaThrLayoutM * get<0>(mma_atom_shape{}); // 32

  static constexpr int kMmaPermuteN =
      2 * kMmaThrLayoutN * get<1>(mma_atom_shape{}); // 32

  static constexpr int kMmaPermuteK =
      kMmaThrLayoutK * get<2>(mma_atom_shape{}); // 16
  // using MmaPermutations = decltype(make_tile(
  //     Int<kMmaPermuteM>{}, kMmaPermuteN{}, Int<kMmaPermuteK>{}));
  using MmaPermutations = decltype(make_tile(
      Int<kMmaPermuteM>{}, Int<kMmaPermuteN>{}, Int<kMmaPermuteK>{}));

  static_assert(kCTAM % (kMmaThrLayoutM * get<0>(mma_atom_shape{})) == 0,
                "kCTAM must be divided by 32");
  static_assert(kCTAN % (kMmaThrLayoutN * get<1>(mma_atom_shape{})) == 0,
                "kCTAN must be divided by 16");

  using MMA =
      decltype(make_tiled_mma(mma_atom{}, MmaThrLayout{}, MmaPermutations{}));
  static constexpr int kconsumerThread = size(MMA{});
  static_assert(Producer % 32 == 0,
                "amount of producer thread must be sets of warp");
  static constexpr int kProducerThread = Producer;
  static constexpr int kThread = kconsumerThread + kProducerThread;

  // smem
  static constexpr int kShmLoadSwizzleB = 3; // 8
  static constexpr int kShmLoadSwizzleM = 3; // 8
  static constexpr int kShmLoadSwizzleS = 3; // 8

  using SmemLayoutAtom = decltype(composition(
      Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
      make_layout(make_shape(Int<8>{}, Int<kCTAK>{}),
                  make_stride(Int<kCTAK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<kCTAM>{}, Int<kCTAK>{}, Int<kStage>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<kCTAN>{}, Int<kCTAK>{}, Int<kStage>{})));
  static constexpr int kASmemSize = cosize(SmemLayoutA{});
  static constexpr int kBSmemSize = cosize(SmemLayoutB{});

  static constexpr int kABSmemSize = (kASmemSize + kBSmemSize) * sizeof(ABtype);
  static constexpr int kSmemLayoutCStage = 2;

  using SmemLayoutAtomC = decltype(composition(
      Swizzle<2, 3, 3>{},
      make_layout(make_shape(Int<kMmaPermuteM>{}, Int<kMmaPermuteN>{}),
                  make_stride(Int<kMmaPermuteN>{}, Int<1>{}))));
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{}, make_shape(Int<kMmaPermuteM>{}, Int<kMmaPermuteN>{},
                                    Int<kSmemLayoutCStage>{})));
  static constexpr int kCSmemSize = cosize(SmemLayoutC{});
  static constexpr int kAllSmemSize = cute::max(kABSmemSize, kCSmemSize);

  // producer g2s copy
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, ABtype>;
  static constexpr int g2s_vec_len = sizeof(cute::uint128_t) / sizeof(ABtype);
  static constexpr int g2s_thread_k = kCTAK / g2s_vec_len;
  static constexpr int g2s_thread_m = kProducerThread / g2s_thread_k;
  using G2SCopyA = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<g2s_thread_m>{}, Int<g2s_thread_k>{}),
                  make_stride(Int<g2s_thread_k>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<g2s_vec_len>{}))));
  using G2SCopyB = G2SCopyA;

  // consumer s2r copy
  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, ABtype>;

  using S2RCopyAtomA = s2r_copy_atom;
  using S2RCopyAtomB = s2r_copy_atom;
  // consumer r2s copy
  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, Ctype>;
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, Ctype>;
  static constexpr int s2g_vec_len = sizeof(cute::uint128_t) / sizeof(Ctype);
  static constexpr int s2g_thread_n = kMmaPermuteN / s2g_vec_len;
  static constexpr int s2g_thread_m = kconsumerThread / s2g_thread_n;
  using S2GCopyC = decltype(make_tiled_copy(
      S2GCopyAtomC{},
      make_layout(make_shape(Int<s2g_thread_m>{}, Int<s2g_thread_n>{}),
                  make_stride(Int<s2g_thread_n>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<s2g_vec_len>{}))));

  struct Arguments {
    void *a_ptr;
    void *b_ptr;
    void *c_ptr;

    Gemm_Shape problem_shape;
    Gemm_Shape tile_shape;
    cudaStream_t stream = nullptr;
    Arguments() = delete;
    Arguments(Gemm_Shape problem_shape_, void *a_ptr_, void *b_ptr_,
              void *c_ptr_)
        : problem_shape(problem_shape_), a_ptr(a_ptr_), b_ptr(b_ptr_),
          c_ptr(c_ptr_) {
      int m = size<0>(problem_shape);
      int n = size<1>(problem_shape);
      int k = size<2>(problem_shape);

      int m_tiles = ceil_div(m, kCTAM);
      int n_tiles = ceil_div(n, kCTAN);
      int k_tiles = ceil_div(k, kCTAK);
      tile_shape = make_shape(m_tiles, n_tiles, k_tiles);
    }

    dim3 get_grid_dims() {
      return dim3(ceil_div(size<0>(problem_shape), kCTAM),
                  ceil_div(size<1>(problem_shape), kCTAN));
    }
  };

  template <typename Pipeline, typename CEngine, typename CLayout>
  DEVICE static auto main_loop(Arguments const &args, Pipeline &pipeline,
                               void *smem_ptr,
                               Tensor<CEngine, CLayout> const &gC) {
    using T = ABtype;
    auto tidx = get_consumer_tidx();
    // int bidm = blockIdx.x;
    // int bidn = blockIdx.y;
    T *ASmemPtr = reinterpret_cast<ABtype *>(smem_ptr);
    T *BSmemPtr = ASmemPtr + kASmemSize;
    auto sA = make_tensor(make_smem_ptr<T>(ASmemPtr),
                          SmemLayoutA{}); //[CTAM, CTAK, stage]
    auto sB = make_tensor(make_smem_ptr<T>(BSmemPtr),
                          SmemLayoutB{}); //[CTAN, CTAK, stage]

    // tiled mma
    MMA mma;
    auto thr_mma = mma.get_slice(tidx);

    auto tArA = thr_mma.partition_fragment_A(sA(_, _, 0));
    auto tBrB = thr_mma.partition_fragment_B(sB(_, _, 0));
    auto tCrC = thr_mma.partition_fragment_C(gC);
    clear(tCrC);

    // s2r copy
    auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, mma);
    auto thr_s2r_copy_a = s2r_copy_a.get_slice(tidx);
    auto s2r_tAsA_copy = thr_s2r_copy_a.partition_S(sA);
    auto s2r_tArA_copy = thr_s2r_copy_a.retile_D(tArA);

    auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, mma);
    auto thr_s2r_copy_b = s2r_copy_b.get_slice(tidx);
    auto s2r_tBsB_copy = thr_s2r_copy_b.partition_S(sB);
    auto s2r_tBrB_copy = thr_s2r_copy_b.retile_D(tBrB);

    const int k_main_loop_cnt = size<2>(args.tile_shape);
    const int k_inner_loop_cnt = size<2>(tArA);
    // int m_tile_bound = (bidm + 1) * kCTAM;
    // int n_tile_bound = (bidn + 1) * kCTAN;
    // int g2s_s_write_cnt = 0;
    // int g2s_g_read_cnt = 0;
    int s2r_s_read_cnt = 0;
    int next_s2r_s_read_cnt = 0;

    if (k_inner_loop_cnt > 1) {
      // wait first producer commit
      pipeline.consumer_wait();
      // if (thread(32)) {
      //   print_tensor(sA(_, _, s2r_s_read_cnt));
      //   print_tensor(sB(_, _, s2r_s_read_cnt));
      // }
      // __syncwarp();
      // load first s2r
      copy(s2r_copy_a, s2r_tAsA_copy(_, _, 0, s2r_s_read_cnt),
           s2r_tArA_copy(_, _, 0));
      copy(s2r_copy_b, s2r_tBsB_copy(_, _, 0, s2r_s_read_cnt),
           s2r_tBrB_copy(_, _, 0));
    }
    for (int k_main_loop_idx = 0; k_main_loop_idx < k_main_loop_cnt;
         k_main_loop_idx++) {
#pragma unroll
      for (int k_inner_loop_idx = 0; k_inner_loop_idx < k_inner_loop_cnt;
           k_inner_loop_idx++) {
        int next_k_inner_loop_idx = (k_inner_loop_idx + 1) % k_inner_loop_cnt;
        // wait next stage commit
        if (k_inner_loop_idx == k_inner_loop_cnt - 1 &&
            k_main_loop_idx < k_main_loop_cnt - 1) {
          pipeline.consumer_wait(); // wait producer
          s2r_s_read_cnt = next_s2r_s_read_cnt;
          // s2r_s_read_cnt = (s2r_s_read_cnt + 1) % kStage;
        }
        // s2r pipeline
        copy(s2r_copy_a,
             s2r_tAsA_copy(_, _, next_k_inner_loop_idx, s2r_s_read_cnt),
             s2r_tArA_copy(_, _, next_k_inner_loop_idx));
        copy(s2r_copy_b,
             s2r_tBsB_copy(_, _, next_k_inner_loop_idx, s2r_s_read_cnt),
             s2r_tBrB_copy(_, _, next_k_inner_loop_idx));
        if (k_inner_loop_idx == 0) {
          pipeline.consumer_release(); // trigger producer
          next_s2r_s_read_cnt = (s2r_s_read_cnt + 1) % kStage;
        }
        // gemm
        gemm(mma, tArA(_, _, k_inner_loop_idx), tBrB(_, _, k_inner_loop_idx),
             tCrC);
      }
    }
    // if (thread(tid, 1)) {
    //   print_tensor(tCrC);
    // }
    return tCrC; // cfrag
  }

  // epilog
  template <typename AccEngine, typename AccLayout, typename CEngine,
            typename CLayout, typename CPredicate>
  DEVICE static void epilog(Arguments const &args, void *smem_ptr,
                            Tensor<AccEngine, AccLayout> const &acc,
                            Tensor<CEngine, CLayout> &gC, CPredicate &gC_pred) {
    // check all consumer thread finish mainloop, producer thread early exit
    __syncthreads();
    int tidx = get_consumer_tidx();
    // printf("tidx:%d consumer:%d\n", threadIdx.x, tidx);
    auto sC = make_tensor(make_smem_ptr<Ctype>(smem_ptr), SmemLayoutC{});

    auto r2s_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, MMA{});

    auto thr_r2s_copy_c = r2s_copy_c.get_slice(tidx);
    auto r2s_tCrC_copy = thr_r2s_copy_c.retile_S(acc);
    auto r2s_tCsC_copy = thr_r2s_copy_c.partition_D(sC);

    S2GCopyC s2g_copy_c;
    auto thr_s2g_copy_c = s2g_copy_c.get_slice(tidx);
    auto s2g_tCsC = thr_s2g_copy_c.partition_S(sC);
    auto s2g_tCgC = thr_s2g_copy_c.partition_D(gC);
    auto s2g_tCgC_pred = thr_s2g_copy_c.partition_D(gC_pred);
    // epilog r2s/s2g pipeline
    auto s2g_tCgC_view = group_modes<1, 3>(s2g_tCgC);
    auto r2s_tCrC_view = group_modes<1, 3>(r2s_tCrC_copy);
    auto s2g_tCgC_pred_view = group_modes<1, 3>(s2g_tCgC_pred);

    const int epilog_cnt = size<1>(r2s_tCrC_view);
    const int epilog_stage = size<3>(r2s_tCsC_copy);

#pragma unroll
    for (int epilog_idx = 0; epilog_idx < epilog_cnt;
         epilog_idx += epilog_stage) {

#pragma unroll
      for (int epilog_stage_idx = 0; epilog_stage_idx < epilog_stage;
           epilog_stage_idx++) {
        // r2s
        copy(r2s_copy_c, r2s_tCrC_view(_, epilog_idx + epilog_stage_idx),
             r2s_tCsC_copy(_, 0, 0, epilog_stage_idx));
      }
      __syncthreads();
      // if (thread(tid)) {
      //   print("\nsC\n");
      //   print_tensor(sC);
      // }
      // __syncthreads();
#pragma unroll
      for (int epilog_stage_idx = 0; epilog_stage_idx < epilog_stage;
           epilog_stage_idx++) {
        // s2g
        if constexpr (kBound_Check) {
          copy_if(
              s2g_copy_c,
              [&](auto... coords) {
                auto pred =
                    s2g_tCgC_pred_view(_, epilog_idx + epilog_stage_idx);
                return elem_less(pred(_0{}, coords...),
                                 select<0, 1>(args.problem_shape));
              },
              s2g_tCsC(_, 0, 0, epilog_stage_idx),
              s2g_tCgC_view(_, epilog_idx + epilog_stage_idx));
        } else {
          copy(s2g_copy_c, s2g_tCsC(_, 0, 0, epilog_stage_idx),
               s2g_tCgC_view(_, epilog_idx + epilog_stage_idx));
        }
      }
      __syncthreads();
    }
  }

  template <typename Pipeline, typename CEngine, typename CLayout,
            typename CPredicate>
  DEVICE static void consumer(Arguments const &args, Pipeline &pipeline,
                              void *smem_ptr, Tensor<CEngine, CLayout> &gC,
                              CPredicate &gC_pred) {
    auto tCrC = main_loop(args, pipeline, smem_ptr, gC);
    epilog(args, smem_ptr, tCrC, gC, gC_pred);
  }

  template <typename Pipeline, typename AEngine, typename ALayout,
            typename BEngine, typename BLayout, typename APredicate,
            typename BPredicate>
  DEVICE static auto
  producer(Arguments const &args, Pipeline &pipeline, void *smem_ptr,
           Tensor<AEngine, ALayout> const &gA,
           Tensor<BEngine, BLayout> const &gB, APredicate const &gA_pred,
           BPredicate const &gB_pred) {
    using T = typename GemmTraits::ABtype;
    int bidm = blockIdx.x;
    int bidn = blockIdx.y;
    int tidx = get_producer_tidx();
    T *ASmemPtr = reinterpret_cast<T *>(smem_ptr);
    T *BSmemPtr = ASmemPtr + kASmemSize;
    auto sA = make_tensor(make_smem_ptr<T>(ASmemPtr),
                          SmemLayoutA{}); //[CTAM, CTAK, stage]
    auto sB = make_tensor(make_smem_ptr<T>(BSmemPtr),
                          SmemLayoutB{}); //[CTAN, CTAK, stage]
    // g2s copy async
    G2SCopyA g2s_copy_a;
    G2SCopyB g2s_copy_b;
    auto thr_g2s_copy_a = g2s_copy_a.get_slice(tidx);
    auto g2s_tAgA_copy = thr_g2s_copy_a.partition_S(gA);
    auto g2s_tAsA_copy = thr_g2s_copy_a.partition_D(sA);
    auto g2s_tAgA_copy_pred = thr_g2s_copy_a.partition_S(gA_pred);

    auto thr_g2s_copy_b = g2s_copy_b.get_slice(tidx);
    auto g2s_tBgB_copy = thr_g2s_copy_b.partition_S(gB);
    auto g2s_tBsB_copy = thr_g2s_copy_b.partition_D(sB);
    auto g2s_tBgB_copy_pred = thr_g2s_copy_b.partition_S(gB_pred);

    const int k_main_loop_cnt = size<2>(gA);
    int m_tile_bound = (bidm + 1) * kCTAM;
    int n_tile_bound = (bidn + 1) * kCTAN;
    int g2s_s_write_cnt = 0;
    int g2s_g_read_cnt = 0;
    // int s2r_s_read_cnt = 0;
    // int next_s2r_s_read_cnt = 0;

    for (int k_main_loop_idx = 0, multi_stage_idx = 0;
         k_main_loop_idx < k_main_loop_cnt; k_main_loop_idx++) {

      for (; multi_stage_idx < k_main_loop_cnt &&
             multi_stage_idx < (k_main_loop_idx + kStage);
           multi_stage_idx++) {
        auto a_tile_bound =
            make_tuple(m_tile_bound, (g2s_g_read_cnt + 1) * kCTAK);
        auto b_tile_bound =
            make_tuple(n_tile_bound, (g2s_g_read_cnt + 1) * kCTAK);

        pipeline.producer_acquire();
        if constexpr (kBound_Check) {
          copy_strip_zfill(g2s_copy_a,
                           g2s_tAgA_copy_pred(_, _, _, g2s_g_read_cnt),
                           g2s_tAgA_copy(_, _, _, g2s_g_read_cnt),
                           g2s_tAsA_copy(_, _, _, g2s_s_write_cnt),
                           a_tile_bound, select<0, 2>(args.problem_shape));
          copy_strip_zfill(g2s_copy_b,
                           g2s_tBgB_copy_pred(_, _, _, g2s_g_read_cnt),
                           g2s_tBgB_copy(_, _, _, g2s_g_read_cnt),
                           g2s_tBsB_copy(_, _, _, g2s_s_write_cnt),
                           b_tile_bound, select<1, 2>(args.problem_shape));
        } else {
          copy(g2s_copy_a, g2s_tAgA_copy(_, _, _, g2s_g_read_cnt),
               g2s_tAsA_copy(_, _, _, g2s_s_write_cnt));
          copy(g2s_copy_b, g2s_tBgB_copy(_, _, _, g2s_g_read_cnt),
               g2s_tBsB_copy(_, _, _, g2s_s_write_cnt));
        }
        // cp_async_fence();
        pipeline.producer_commit();
        g2s_g_read_cnt++;
        g2s_s_write_cnt = (g2s_s_write_cnt + 1) % kStage;
        // if (thread0()) {
        //   print("produer %d\n", multi_stage_idx);
        // }
      }
    }
  }

  DEVICE static int get_producer_tidx() { return threadIdx.x; }

  DEVICE static int get_consumer_tidx() {
    return threadIdx.x - kProducerThread;
  }
};
// https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/pipeline/shared_state.html
#pragma nv_diag_suppress static_var_with_dynamic_init
template <typename GemmTraits>
__global__ void gemmTN_naive_ws(typename GemmTraits::Arguments args) {
  using ABtype = typename GemmTraits::ABtype;
  using Ctype = typename GemmTraits::Ctype;
  constexpr int kCTAM = GemmTraits::kCTAM;
  constexpr int kCTAN = GemmTraits::kCTAN;
  constexpr int kCTAK = GemmTraits::kCTAK;
  constexpr int kStage = GemmTraits::kStage;

  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  auto tidx = block.thread_rank();
  auto bidm = blockIdx.x;
  auto bidn = blockIdx.y;
  // initialize pipeline
  const cuda::pipeline_role thread_role =
      block.thread_rank() < GemmTraits::kProducerThread
          ? cuda::pipeline_role::producer
          : cuda::pipeline_role::consumer;
  extern __shared__ ABtype smem_ptr[];
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block,
                                         kStage>
      shared_state;
  auto pipeline = cuda::make_pipeline(block, &shared_state, thread_role);

  auto A =
      make_tensor(make_gmem_ptr<ABtype>(args.a_ptr),
                  select<0, 2>(args.problem_shape), LayoutRight{}); // [m,k]
  auto B =
      make_tensor(make_gmem_ptr<ABtype>(args.b_ptr),
                  select<1, 2>(args.problem_shape), LayoutRight{}); // [n,k]
  auto C =
      make_tensor(make_gmem_ptr<Ctype>(args.c_ptr),
                  select<0, 1>(args.problem_shape), LayoutRight{}); // [m,n]

  Tensor A_pred = make_identity_tensor(shape(A));
  Tensor B_pred = make_identity_tensor(shape(B));
  Tensor C_pred = make_identity_tensor(shape(C));

  Tensor gA =
      local_tile(A, make_tile(Int<kCTAM>{}, Int<kCTAK>{}), make_coord(bidm, _));
  Tensor gB =
      local_tile(B, make_tile(Int<kCTAN>{}, Int<kCTAK>{}), make_coord(bidn, _));
  Tensor gC = local_tile(C, make_tile(Int<kCTAM>{}, Int<kCTAN>{}),
                         make_coord(bidm, bidn));

  Tensor gA_pred = local_tile(A_pred, make_tile(Int<kCTAM>{}, Int<kCTAK>{}),
                              make_coord(bidm, _));
  Tensor gB_pred = local_tile(B_pred, make_tile(Int<kCTAN>{}, Int<kCTAK>{}),
                              make_coord(bidn, _));
  Tensor gC_pred = local_tile(C_pred, make_tile(Int<kCTAM>{}, Int<kCTAN>{}),
                              make_coord(bidm, bidn));

  if (thread_role == cuda::pipeline_role::producer) {
    GemmTraits::producer(args, pipeline, smem_ptr, gA, gB, gA_pred, gB_pred);
  } else {
    GemmTraits::consumer(args, pipeline, smem_ptr, gC, gC_pred);
  }
}
