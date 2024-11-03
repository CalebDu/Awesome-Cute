// reference: https://github.com/reed-lau/cute-gemm
#include "common.h"
#include "cute/tensor.hpp"

using namespace cute;

template <class CTA_tile, int Stage> struct GemmTraits {
  // fp16 example
  using ABtype = cutlass::half_t;
  using Ctype = cutlass::half_t;

  static constexpr int kCTAM = size<0>(CTA_tile{});
  static constexpr int kCTAN = size<1>(CTA_tile{});
  static constexpr int kCTAK = size<2>(CTA_tile{});
  static constexpr int kStage = Stage;
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
  static constexpr int kAllSmemSize = kABSmemSize;

  // mma
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  // tiled mma shape[16*2, 8*2, 16], warp layout [2, 2, 1]
  static constexpr int kMmaThrLayoutM = 2;
  static constexpr int kMmaThrLayoutN = 2;
  static constexpr int kMmaThrLayoutK = 1;
  using mma_atom_shape = mma_traits::Shape_MNK;

  using MmaThrLayout = decltype(make_layout(make_shape(
      Int<kMmaThrLayoutM>{}, Int<kMmaThrLayoutN>{}, Int<kMmaThrLayoutK>{})));
  static constexpr int kMmaPermuteM = kMmaThrLayoutM * get<0>(mma_atom_shape{});

  // make c frag continuous for each thread
  // using kMmaPermuteN = decltype(make_layout(make_shape(_2{}, _4{}, _2{}),
  //                                           make_stride(_1{}, _4{}, _2{})));
  static constexpr int kMmaPermuteN =
      2 * kMmaThrLayoutN * get<1>(mma_atom_shape{});

  static constexpr int kMmaPermuteK = kMmaThrLayoutK * get<2>(mma_atom_shape{});
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
  // r2g copy
  using R2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, Ctype>;

  // static constexpr int r2g_vec_len = sizeof(cute::uint128_t) / sizeof(Ctype);
  // static constexpr int r2g_thread_k = kCTAN / r2g_vec_len;
  // static constexpr int r2g_thread_m = kThread / r2g_thread_k;

  // using r2GCopyC = decltype(make_tiled_copy(
  //     r2GCopyAtomC{},
  //     make_layout(make_shape(Int<r2g_thread_m>{}, Int<r2g_thread_k>{}),
  //                 make_stride(Int<r2g_thread_k>{}, Int<1>{})),
  //     make_layout(make_shape(Int<1>{}, Int<r2g_vec_len>{}))));
};

template <typename GemmTraits>
__global__ void
gemmTN_multistage(void *__restrict__ Aptr, void *__restrict__ Bptr,
                  void *__restrict__ Cptr, int m, int n, int k) {
  using T = typename GemmTraits::ABtype;
  using ACC_T = typename GemmTraits::Ctype;
  using MMA = typename GemmTraits::MMA;
  using SmemLayoutA = typename GemmTraits::SmemLayoutA;
  using SmemLayoutB = typename GemmTraits::SmemLayoutB;
  using G2SCopyA = typename GemmTraits::G2SCopyA;
  using G2SCopyB = typename GemmTraits::G2SCopyB;
  using S2RCopyAtomA = typename GemmTraits::S2RCopyAtomA;
  using S2RCopyAtomB = typename GemmTraits::S2RCopyAtomB;
  using R2GCopyAtomC = typename GemmTraits::R2GCopyAtomC;
  int bidm = blockIdx.x;
  int bidn = blockIdx.y;
  int tidx = threadIdx.x;

  extern __shared__ T smem[];
  constexpr int kCTAM = GemmTraits::kCTAM;
  constexpr int kCTAN = GemmTraits::kCTAN;
  constexpr int kCTAK = GemmTraits::kCTAK;
  constexpr int kStage = GemmTraits::kStage;
  constexpr int kASmemSize = GemmTraits::kASmemSize;

  T *ASmemPtr = smem;
  T *BSmemPtr = smem + kASmemSize;

  Tensor A = make_tensor(make_gmem_ptr<T>(Aptr), make_shape(m, k),
                         make_stride(k, _1{}));
  Tensor B = make_tensor(make_gmem_ptr<T>(Bptr), make_shape(n, k),
                         make_stride(k, _1{}));
  Tensor C = make_tensor(make_gmem_ptr<ACC_T>(Cptr), make_shape(m, n),
                         make_stride(n, _1{}));

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
  // r2g store copy
  // auto r2s_copy_c = make_tiled_copy_C(R2GCopyAtomC{}, mma);
  // auto thr_r2g_copy_c = r2s_copy_c.get_slice(tidx);
  // auto thr_r2g_tCrC_copy = thr_r2g_copy_c.retile_S(tCrC);
  // auto thr_r2g_tCgC_copy = thr_r2g_copy_c.partition_D(gC);
  // auto thr_r2g_tCgC_copy_pred = thr_r2g_copy_c.partition_D(gC_pred);

  const int k_main_loop_cnt = size<2>(gA);
  const int k_inner_loop_cnt = size<2>(tArA);
  int m_tile_bound = (bidm + 1) * kCTAM;
  int n_tile_bound = (bidn + 1) * kCTAN;
  int g2s_s_write_cnt = 0;
  int g2s_g_read_cnt = 0;
  int s2r_s_read_cnt = 0;
  int next_s2r_s_read_cnt = 0;
#if 0 // print for debug
  if (thread0()) {
    print("\n smem_a size \n");
    print(cosize(SmemLayoutA{}));
    print("\n smem_b size \n");
    print(cosize(SmemLayoutB{}));
    print("\nmma\n");
    print(mma);
    print("\ng2s_A\n");
    print(g2s_copy_a);
    print("\ng2s_B\n");
    print(g2s_copy_b);
    print("\nthr_g2s_copy_a\n");
    print(thr_g2s_copy_a);
    print("\nthr_g2s_copy_b\n");
    print(thr_g2s_copy_b);

    printf("\nA\n");
    print(A);
    printf("\ngA\n");
    print(gA);
    printf("\nsA\n");
    print(sA);
    printf("\ntaga\n");
    print(tAgA);
    printf("\ntara\n");
    print(tArA);
    printf("\ng2s_tAgA_copy\n");
    print(g2s_tAgA_copy);
    printf("\ng2s_tAsA_copy\n");
    print(g2s_tAsA_copy);
    printf("\ns2r_tAsA_copy\n");
    print(s2r_tAsA_copy);
    printf("\ns2r_tArA\n");
    print(s2r_tArA_copy);

    printf("\nB\n");
    print(B);
    printf("\ngB\n");
    print(gB);
    printf("\nsB\n");
    print(sB);
    printf("\ntbgb\n");
    print(tBgB);
    printf("\ntbrb\n");
    print(tBrB);
    printf("\ng2s_tBgB_copy\n");
    print(g2s_tBgB_copy);
    printf("\ng2s_tBsB_copy\n");
    print(g2s_tBsB_copy);
    printf("\ns2r_tBsB_copy\n");
    print(s2r_tBsB_copy);
    printf("\ns2r_tBrB\n");
    print(s2r_tBrB_copy);

    // printf("\ngC\n");
    // print(gC);
    // printf("\ntcgc\n");
    // print(tCgC);
    // printf("\ntcrc\n");
    // print(tCrC);

    // printf("\nA\n");
    // print_tensor(A);
    // printf("\nB\n");
    // print_tensor(B);
    // printf("\ngA\n");
    // print_tensor(gA);
    // printf("\ngB\n");
    // print_tensor(gB);
  }
#endif
#pragma unroll
  for (int i_stage = 0; i_stage < kStage - 1; i_stage++) {
    auto a_tile_bound = make_tuple(m_tile_bound, (i_stage + 1) * kCTAK);
    auto b_tile_bound = make_tuple(n_tile_bound, (i_stage + 1) * kCTAK);
    if (g2s_g_read_cnt < k_main_loop_cnt) {
      copy_strip_zfill(g2s_copy_a, g2s_tAgA_copy_pred(_, _, _, i_stage),
                       g2s_tAgA_copy(_, _, _, i_stage),
                       g2s_tAsA_copy(_, _, _, i_stage), a_tile_bound, shape(A));
      copy_strip_zfill(g2s_copy_b, g2s_tBgB_copy_pred(_, _, _, i_stage),
                       g2s_tBgB_copy(_, _, _, i_stage),
                       g2s_tBsB_copy(_, _, _, i_stage), b_tile_bound, shape(B));
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
        if (g2s_g_read_cnt < k_main_loop_cnt) {
          copy_strip_zfill(
              g2s_copy_a, g2s_tAgA_copy_pred(_, _, _, g2s_g_read_cnt),
              g2s_tAgA_copy(_, _, _, g2s_g_read_cnt),
              g2s_tAsA_copy(_, _, _, g2s_s_write_cnt), a_tile_bound, shape(A));
          copy_strip_zfill(
              g2s_copy_b, g2s_tBgB_copy_pred(_, _, _, g2s_g_read_cnt),
              g2s_tBgB_copy(_, _, _, g2s_g_read_cnt),
              g2s_tBsB_copy(_, _, _, g2s_s_write_cnt), b_tile_bound, shape(B));
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
  copy(tCrC, tCgC);
  // copy_if(
  //     r2s_copy_c,
  //     [&](auto... coords) {
  //       return elem_less(thr_r2g_tCgC_copy_pred(_0{}, coords...), shape(C));
  //     },
  //     thr_r2g_tCrC_copy, thr_r2g_tCgC_copy);
}