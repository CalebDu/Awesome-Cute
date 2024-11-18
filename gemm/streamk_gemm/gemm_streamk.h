#pragma once
#include "barrier.h"
#include "common.h"
#include "cute/tensor.hpp"
#include "dp_sk_block.h"
#include "utils.h"
#define tid 64
template <class CTA_tile, int Stage, bool Bound_Check = false>
struct GemmTraits {
  // fp16 example
  using ABtype = cutlass::half_t;
  using Ctype = cutlass::half_t;
  using Ctype_raw = UnderlyingType<Ctype>::type;
  using Ctype_pack = half2;
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
  static constexpr int kThread = size(MMA{});

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

  // g2s copy
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, ABtype>;
  static constexpr int g2s_vec_len = sizeof(cute::uint128_t) / sizeof(ABtype);
  static constexpr int g2s_thread_k = kCTAK / g2s_vec_len;
  static constexpr int g2s_thread_m = kThread / g2s_thread_k;
  using G2SCopyA = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<g2s_thread_m>{}, Int<g2s_thread_k>{}),
                  make_stride(Int<g2s_thread_k>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<g2s_vec_len>{}))));
  using G2SCopyB = G2SCopyA;

  // s2r copy
  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, ABtype>;

  using S2RCopyAtomA = s2r_copy_atom;
  using S2RCopyAtomB = s2r_copy_atom;
  // r2s copy
  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, Ctype>;
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, Ctype>;
  static constexpr int s2g_vec_len = sizeof(cute::uint128_t) / sizeof(Ctype);
  static constexpr int s2g_thread_n = kMmaPermuteN / s2g_vec_len;
  static constexpr int s2g_thread_m = kThread / s2g_thread_n;
  using S2GCopyC = decltype(make_tiled_copy(
      S2GCopyAtomC{},
      make_layout(make_shape(Int<s2g_thread_m>{}, Int<s2g_thread_n>{}),
                  make_stride(Int<s2g_thread_n>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<s2g_vec_len>{}))));

  // reference  cutlass/include/cutlass/gemm/kernel/gemm_universal_streamk.h
  struct Arguments {
    using Block_Wrapper = SK_DP_Block_Wrapper<CTA_tile>;
    using Gemm_Shape = typename Block_Wrapper::Gemm_Shape;
    static constexpr int kWorkspaceBytesPerBlock =
        kCTAM * kCTAN *
        sizeof(Ctype); // partial sum workspace size for cta tile
    void *a_ptr;
    void *b_ptr;
    void *c_ptr;
    Block_Wrapper block_wrapper;
    Gemm_Shape problem_shape;
    void *workspace_ptr = nullptr;
    void *barrier_workspace = nullptr;     // intra-cta sync
    void *partial_sum_workspace = nullptr; // dram store partial sum
    cudaStream_t stream = nullptr;
    Arguments() = delete;
    Arguments(Gemm_Shape problem_shape_, void *a_ptr_, void *b_ptr_,
              void *c_ptr_, int const sm_occupancy_, int const device_sm_,
              int const avail_sms_,
              SK_DP_Block_Strategy strategy_ = SK_DP_Block_Strategy::sk2tile_dp)
        : problem_shape(problem_shape_), a_ptr(a_ptr_), b_ptr(b_ptr_),
          c_ptr(c_ptr_) {

      block_wrapper = Block_Wrapper(problem_shape, sm_occupancy_, device_sm_,
                                    avail_sms_, strategy_);
    }

    // Pad the given allocation size up to the nearest cache line
    static size_t cacheline_align_up(size_t size) {
      static const int CACHELINE_SIZE = 128;
      return (size + CACHELINE_SIZE - 1) / CACHELINE_SIZE * CACHELINE_SIZE;
    }

    // Get the workspace size needed for barrier
    size_t get_barrier_workspace_size() const {
      // sync each sk_block in each sk_tile
      int sk_tiles = block_wrapper.sk_tiles;

      return cacheline_align_up(sizeof(typename Barrier::T) * sk_tiles);
    }

    // Get the workspace size needed for intermediate partial sums
    size_t get_partials_workspace_size() const {
      int sk_tiles = block_wrapper.sk_tiles;
      return cacheline_align_up(kWorkspaceBytesPerBlock * sk_tiles);
    }

    // Get all workspace size
    size_t get_workspace_size() const {
      return get_partials_workspace_size() + get_barrier_workspace_size();
    }

    void alloc_reset_workspace(cudaStream_t stream = 0) {
      size_t workspace_size = get_workspace_size();
      size_t barrier_size = get_barrier_workspace_size();
      size_t partial_sum_size = get_partials_workspace_size();
      if (workspace_size > 0) {
        CUDA_CHECK(cudaMallocAsync(&workspace_ptr, workspace_size, stream));
        if (partial_sum_size > 0) {
          partial_sum_workspace = static_cast<uint8_t *>(workspace_ptr);
        }
        if (barrier_size > 0) {
          barrier_workspace =
              static_cast<uint8_t *>(workspace_ptr) + partial_sum_size;
          // clear
          CUDA_CHECK(
              cudaMemsetAsync(barrier_workspace, 0, barrier_size, stream));
        }
      }
    }

    void init_workspace(cudaStream_t stream_ = 0) {
      stream = stream_;
      alloc_reset_workspace(stream);
    }

    void free_workspace() {
      if (workspace_ptr) {
        CUDA_CHECK(cudaFreeAsync(workspace_ptr, stream));
        workspace_ptr = nullptr;
        barrier_workspace = nullptr;
        partial_sum_workspace = nullptr;
      }
    }

    dim3 get_grid_dims() { return block_wrapper.get_grid_dims(); }
  };
  using Gemm_Shape = typename Arguments::Gemm_Shape;

  /// Tile work descriptor
  struct TileWorkDesc {
    /// The linear tile index
    int tile_idx;

    /// The location of this tile (in threadblock-tile coordinates) in the
    /// output matrix
    Gemm_Shape tiled_coord;

    // The first global-scoped MAC-iteration this threadblock will perform for
    // this tile
    int iter_begin;

    // The starting index in the k-domain for MAC-iterations this threadblock
    // will perform for this tile
    int k_iter_begin;

    // The ending index (one-past) in the k-domain for MAC-iterations this
    // threadblock will perform for this tile
    int k_iter_end;

    /// The number of remaining MAC-iterations this threadblock will perform for
    /// this tile
    int k_iters_remaining;

    // Whether this block will perform the first iteration of this tile
    CUTLASS_DEVICE
    bool tile_started() { return (k_iter_begin == 0); }

    // Whether this block will perform the last iteration of this tile
    CUTLASS_DEVICE
    bool tile_finished(Arguments const &args) {
      return (k_iter_end == size<2>(args.block_wrapper.tile_shape));
    }
  };

  // main loop
  template <typename AEngine, typename ALayout, typename BEngine,
            typename BLayout, typename CEngine, typename CLayout,
            typename APredicate, typename BPredicate>
  DEVICE static auto
  main_loop(Arguments const &args, TileWorkDesc &tile_work, void *smem_ptr,
            Tensor<AEngine, ALayout> const &gA,
            Tensor<BEngine, BLayout> const &gB,
            Tensor<CEngine, CLayout> const &gC, APredicate const &gA_pred,
            BPredicate const &gB_pred) {
    using T = ABtype;
    auto tidx = threadIdx.x;

    T *ASmemPtr = reinterpret_cast<ABtype *>(smem_ptr);
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

    // tiled mma
    MMA mma;
    auto thr_mma = mma.get_slice(tidx);
    auto tAgA = thr_mma.partition_A(gA);
    auto tBgB = thr_mma.partition_B(gB);
    auto tCgC = thr_mma.partition_C(gC);

    auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));
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

    const int k_main_loop_cnt = tile_work.k_iters_remaining;
    const int k_inner_loop_cnt = size<2>(tArA);
    int m_tile_bound = (size<0>(tile_work.tiled_coord) + 1) * kCTAM;
    int n_tile_bound = (size<1>(tile_work.tiled_coord) + 1) * kCTAN;
    int g2s_s_write_cnt = 0;
    int g2s_g_read_cnt = tile_work.k_iter_begin;
    int s2r_s_read_cnt = 0;
    int next_s2r_s_read_cnt = 0;

#if 0
    if (thread(tid)) {
      printf("\ngA\n");
      print(gA);
      printf("\nsA\n");
      print(sA);
      printf("\ntaga\n");
      print(tAgA);
      printf("\ntara\n");
      print(tArA);
      printf("\ngB\n");
      print(gB);
      printf("\nsB\n");
      print(sB);
      printf("\ntbgb\n");
      print(tBgB);
      printf("\ntbrb\n");
      print(tBrB);
    }
#endif
    // if (thread(0, 1)) {
    //   print(shape(gA));
    //   print("share bidx: %d, tile_idx %d, k_begin %d, k_end "
    //         "%d, k_remaining %d\n",
    //         blockIdx.x, tile_work.tile_idx, tile_work.k_iter_begin,
    //         tile_work.k_iter_end, tile_work.k_iters_remaining);
    // }
    // __syncthreads();
    // if (thread(0, 1)) {
    //   print_tensor(gA(_, _, g2s_g_read_cnt));
    //   print_tensor(gB(_, _, g2s_g_read_cnt));
    //   // print_tensor(sA);
    //   // print_tensor(sB);
    // }
    // __syncthreads();
#pragma unroll
    for (int i_stage = 0; i_stage < kStage - 1; i_stage++) {
      auto a_tile_bound = make_tuple(m_tile_bound, (i_stage + 1) * kCTAK);
      auto b_tile_bound = make_tuple(n_tile_bound, (i_stage + 1) * kCTAK);
      if (g2s_g_read_cnt < tile_work.k_iter_end) {
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
      }
      g2s_g_read_cnt++;
      g2s_s_write_cnt++;
      cp_async_fence();
    }

    if (k_inner_loop_cnt > 1) {
      // wait first cp_async commit
      cp_async_wait<kStage - 2>();
      __syncthreads();
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
        if (k_inner_loop_idx == k_inner_loop_cnt - 1) {
          cp_async_wait<kStage - 2>();
          __syncthreads();
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
        // load last stage
        if (k_inner_loop_idx == 0) {
          auto a_tile_bound =
              make_tuple(m_tile_bound, (g2s_g_read_cnt + 1) * kCTAK);
          auto b_tile_bound =
              make_tuple(n_tile_bound, (g2s_g_read_cnt + 1) * kCTAK);
          // OOB do not g2s copy
          if (g2s_g_read_cnt < tile_work.k_iter_end) {
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
          }
          g2s_g_read_cnt++;
          g2s_s_write_cnt = s2r_s_read_cnt;
          next_s2r_s_read_cnt = (s2r_s_read_cnt + 1) % kStage;
          cp_async_fence();
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
    int tidx = threadIdx.x;
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

  DEVICE static void init_sk_tile(Arguments const &args,
                                  TileWorkDesc &tile_work, int tile_idx,
                                  int block_iter_begin, int block_iter_end) {
    tile_work.tile_idx = tile_idx;

    int global_k_iter_begin = tile_idx * args.block_wrapper.iter_per_tile;
    // (tile_idx + 1) * args.block_wrapper.iter_per_tile
    int global_k_iter_end =
        global_k_iter_begin + args.block_wrapper.iter_per_tile;

    tile_work.k_iter_begin =
        max(block_iter_begin, global_k_iter_begin) - global_k_iter_begin;
    tile_work.k_iter_end =
        min(block_iter_end, global_k_iter_end) - global_k_iter_begin;

    tile_work.k_iters_remaining = tile_work.k_iter_end - tile_work.k_iter_begin;

    tile_work.tiled_coord = args.block_wrapper.get_tile_offset(tile_idx);
  }

  DEVICE static void init_dp_tile(Arguments const &args,
                                  TileWorkDesc &tile_work, int tile_idx) {
    tile_work.tile_idx = tile_idx;

    // tile_work.iter_begin = tile_idx * args.block_wrapper.iter_per_tile;
    tile_work.k_iters_remaining = args.block_wrapper.iter_per_tile;

    tile_work.k_iter_begin = 0;

    tile_work.k_iter_end = args.block_wrapper.iter_per_tile;

    tile_work.tiled_coord = args.block_wrapper.get_tile_offset(tile_idx);
  }

  template <typename AccEngine, typename AccLayout, typename PartialEngine,
            typename PartialLayout>
  DEVICE static void
  sk_tile_reduce(Arguments const &args, TileWorkDesc &tile_work, int block_idx,
                 Tensor<AccEngine, AccLayout> &acc,
                 Tensor<PartialEngine, PartialLayout> &partial_sum) {
    int first_iter = tile_work.tile_idx * args.block_wrapper.iter_per_tile;
    int first_block = args.block_wrapper.get_sk_block_idx(first_iter);
    if (!tile_work.tile_finished(args)) {
      share_accumulators(args, tile_work, block_idx, first_block, acc,
                         partial_sum);
    } else if (!tile_work.tile_started()) {
      acquire_accumulators(args, tile_work, block_idx, first_block, acc,
                           partial_sum);
    }
  }

  template <typename AccEngine, typename AccLayout, typename PartialEngine,
            typename PartialLayout>
  DEVICE static void
  share_accumulators(Arguments const &args, TileWorkDesc &tile_work,
                     int block_idx, int first_block,
                     Tensor<AccEngine, AccLayout> const &acc,
                     Tensor<PartialEngine, PartialLayout> &partial_sum) {
    int tidx = threadIdx.x;
    auto copy_acc = make_tiled_copy_C(R2SCopyAtomC{}, MMA{});
    auto thr_copy_acc = copy_acc.get_slice(tidx);
    auto copy_acc_s = thr_copy_acc.retile_S(acc);
    auto copy_acc_d = thr_copy_acc.partition_D(partial_sum);

    if (block_idx == first_block) {
      // store acc to dram partial_sum
      copy(copy_acc, copy_acc_s, copy_acc_d);
    } else {
      int wait_block_count = block_idx - first_block;
      auto pack_copy_acc_s = recast<Ctype_pack>(copy_acc_s);
      auto pack_copy_acc_d = recast<Ctype_pack>(copy_acc_d);
      // Turnstile reduction order deterministicly: wait all previous block
      // complete
      Barrier::wait_eq(args.barrier_workspace, tidx, tile_work.tile_idx,
                       wait_block_count);
      // atomic reduce
#pragma unroll
      for (int idx = 0; idx < size(pack_copy_acc_s); idx++) {
        atomicAdd(&pack_copy_acc_d(idx), pack_copy_acc_s(idx));
      }
    }
    // arrive counter ++
    Barrier::arrive_inc(args.barrier_workspace, tidx, tile_work.tile_idx);
  }

  template <typename AccEngine, typename AccLayout, typename PartialEngine,
            typename PartialLayout>
  DEVICE static void
  acquire_accumulators(Arguments const &args, TileWorkDesc &tile_work,
                       int block_idx, int first_block,
                       Tensor<AccEngine, AccLayout> &acc,
                       Tensor<PartialEngine, PartialLayout> &partial_sum) {

    int tidx = threadIdx.x;
    auto copy_acc = make_tiled_copy_C(R2SCopyAtomC{}, MMA{});
    auto thr_copy_acc = copy_acc.get_slice(tidx);
    auto copy_acc_s = thr_copy_acc.retile_S(acc);
    auto copy_acc_d = thr_copy_acc.partition_D(partial_sum);
    int wait_block_cnt = block_idx - first_block;
    Barrier::wait_eq_reset(args.barrier_workspace, tidx, tile_work.tile_idx,
                           wait_block_cnt);
    auto pack_copy_acc_s = recast<Ctype_pack>(copy_acc_s);
    auto pack_copy_acc_d = recast<Ctype_pack>(copy_acc_d);
    // reduce
#pragma unroll
    for (int idx = 0; idx < size(pack_copy_acc_s); idx++) {
      pack_copy_acc_s(idx) = pack_copy_acc_s(idx) + pack_copy_acc_d(idx);
    }
  }
};

template <typename GemmTraits>
__global__ void gemmTN_streamk_dp(typename GemmTraits::Arguments args) {

  using ABtype = typename GemmTraits::ABtype;
  using Ctype = typename GemmTraits::Ctype;
  using TileWorkDesc = typename GemmTraits::TileWorkDesc;
  constexpr int kCTAM = GemmTraits::kCTAM;
  constexpr int kCTAN = GemmTraits::kCTAN;
  constexpr int kCTAK = GemmTraits::kCTAK;

  extern __shared__ ABtype smem_ptr[];
  int bidx = blockIdx.x;

  // Initialize block's iteration range
  int tile_idx = 0;
  int block_iter_begin = 0;
  int block_iters_remaining = 0;
  int sk_tiles = args.block_wrapper.sk_tiles;
  int sk_blocks = args.block_wrapper.sk_blocks;
  bool is_sk_block = bidx < sk_blocks;

  TileWorkDesc tile_work;

  auto A =
      make_tensor(make_gmem_ptr<ABtype>(args.a_ptr),
                  select<0, 2>(args.problem_shape), LayoutRight{}); // [m,k]
  auto B =
      make_tensor(make_gmem_ptr<ABtype>(args.b_ptr),
                  select<1, 2>(args.problem_shape), LayoutRight{}); // [n,k]
  auto C =
      make_tensor(make_gmem_ptr<Ctype>(args.c_ptr),
                  select<0, 1>(args.problem_shape), LayoutRight{}); // [m,n]
  auto partial_sum =
      make_tensor(make_gmem_ptr<Ctype>(args.partial_sum_workspace),
                  make_shape(sk_tiles, Int<kCTAM>{}, Int<kCTAN>{}),
                  LayoutRight{}); // [sk_tile, CTAM, CTAN]

  Tensor gA = local_tile(A, make_tile(Int<kCTAM>{}, Int<kCTAK>{}),
                         make_coord(_, _)); // [ctam, ctak, m_loop, k_loop]
  Tensor gB = local_tile(B, make_tile(Int<kCTAN>{}, Int<kCTAK>{}),
                         make_coord(_, _)); // [ctan, ctak, n_loop, k_loop]
  Tensor gC = local_tile(C, make_tile(Int<kCTAM>{}, Int<kCTAN>{}),
                         make_coord(_, _)); // [ctam, ctan, m_loop, n_loop]

  Tensor A_pred = make_identity_tensor(shape(A));
  Tensor B_pred = make_identity_tensor(shape(B));
  Tensor C_pred = make_identity_tensor(shape(C));

  Tensor gA_pred = local_tile(A_pred, make_tile(Int<kCTAM>{}, Int<kCTAK>{}),
                              make_coord(_, _)); // [ctam, ctak, m_loop, k_loop]
  Tensor gB_pred = local_tile(B_pred, make_tile(Int<kCTAN>{}, Int<kCTAK>{}),
                              make_coord(_, _)); // [ctan, ctak, n_loop, k_loop]
  Tensor gC_pred = local_tile(C_pred, make_tile(Int<kCTAM>{}, Int<kCTAN>{}),
                              make_coord(_, _)); // [ctam, ctan, n_loop, k_loop]

  if (is_sk_block) {
    int block_iter_end;

    args.block_wrapper.get_iter_extents(bidx, block_iter_begin, block_iter_end);
    block_iters_remaining = block_iter_end - block_iter_begin;
    tile_idx = args.block_wrapper.get_sk_tile_idx(block_iter_end - 1);
    GemmTraits::init_sk_tile(args, tile_work, tile_idx, block_iter_begin,
                             block_iter_end);
  } else {                                    // dp block
    tile_idx = (bidx - sk_blocks) + sk_tiles; // dp tile idx
    block_iters_remaining = args.block_wrapper.iter_per_tile;
    GemmTraits::init_dp_tile(args, tile_work, tile_idx);
  }
#pragma unroll 1 // no unroll
  while (true) {
    auto tiled_coord = tile_work.tiled_coord;        // [bidm, bidn, 1]
    auto cur_gA = gA(_, _, size<0>(tiled_coord), _); // [ctam, ctak, k_loop]
    auto cur_gB = gB(_, _, size<1>(tiled_coord), _); // [ctan, ctak, k_loop]
    auto cur_gC =
        gC(_, _, size<0>(tiled_coord), size<1>(tiled_coord)); // [ctam, ctan]

    auto cur_gA_pred =
        gA_pred(_, _, size<0>(tiled_coord), _); // [ctam, ctak, k_loop]
    auto cur_gB_pred =
        gB_pred(_, _, size<1>(tiled_coord), _); // [ctan, ctak, k_loop]
    auto cur_gC_pred = gC_pred(_, _, size<0>(tiled_coord),
                               size<1>(tiled_coord)); // [ctam, ctan]
    auto tCrC = GemmTraits::main_loop(args, tile_work, smem_ptr, cur_gA, cur_gB,
                                      cur_gC, cur_gA_pred, cur_gB_pred);
    if (is_sk_block) {
      // sk_tile reduce
      auto sk_tile_idx = tile_work.tile_idx;
      auto cur_partial_sum = partial_sum(sk_tile_idx, _, _);
      GemmTraits::sk_tile_reduce(args, tile_work, bidx, tCrC, cur_partial_sum);
    }

    // epilog
    if ((!is_sk_block) || (is_sk_block && tile_work.tile_finished(args))) {
      GemmTraits::epilog(args, smem_ptr, tCrC, cur_gC, cur_gC_pred);
    }
    block_iters_remaining -= tile_work.k_iters_remaining;
    if (block_iters_remaining == 0) {
      break;
    }
    __syncthreads();
    if (is_sk_block) {
      tile_idx--;
      GemmTraits::init_sk_tile(args, tile_work, tile_idx, block_iter_begin,
                               block_iter_begin + block_iters_remaining);
    }
  }
}
