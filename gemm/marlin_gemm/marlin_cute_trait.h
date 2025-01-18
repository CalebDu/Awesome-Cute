#pragma once
#include "barrier.h"
#include "common.h"
#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"
using namespace cute;

template <int lut> DEVICE static int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16
// values. We mostly follow the strategy in the link below, with some small
// changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
DEVICE static auto dequant(int q) {
  auto half4_frag = make_tensor<half2>(make_shape(_2{}));
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  half4_frag[0] = __hsub2(*reinterpret_cast<half2 *>(&lo),
                          *reinterpret_cast<const half2 *>(&SUB));
  half4_frag[1] = __hfma2(*reinterpret_cast<half2 *>(&hi),
                          *reinterpret_cast<const half2 *>(&MUL),
                          *reinterpret_cast<const half2 *>(&ADD));
  return half4_frag;
}

template <class CTA_Tile, int Stage, int GroupSize> struct MarlinGemmTraits {
  // A(fp16)*B(int4 dequant->fp16)-> acc fp32 -> store fp16
  using Atype = half_t;
  using Btype = uint4_t;
  using ACCtype = float;
  using Ctype = half_t;
  using LockType = int32_t;
  using Coord = Shape<int, int, int>;

  static constexpr int kThread = 256; // 8 warps
  static constexpr int kWarp_size = 32;
  static constexpr int kWarp = kThread / kWarp_size;

  // cta shape (m, 256, 64) / (m, 128, 128) [m <= 64]
  static constexpr int kCTAM = size<0>(CTA_Tile{});
  static constexpr int kCTAN = size<1>(CTA_Tile{});
  static constexpr int kCTAK = size<2>(CTA_Tile{});
  static constexpr int kStage = Stage;
  static constexpr int kGroupSize = GroupSize;
  static_assert(GroupSize == -1 || GroupSize == 128,
                "only support GroupSize=-1or128");
  static constexpr int IterPerGroup =
      (GroupSize == -1) ? GroupSize : GroupSize / kCTAK;
  // warp tile
  static constexpr int kWarpM = kCTAM;
  static constexpr int kWarpN = 64;
  static constexpr int kWarpK = 16;
  using ATile = decltype(make_tile(Int<kCTAM>(), Int<kCTAK>()));
  using BTile = decltype(make_tile(Int<kCTAK / kWarpK>{}, Int<kCTAN / kWarpN>{},
                                   _32{}, _32{}));
  using CTile = decltype(make_tile(Int<kCTAM>{}, Int<kCTAN>{}));
  using ScaleTile = decltype(make_tile(Int<1>{}, Int<kCTAN>{}));

  using mma_op = SM80_16x8x16_F32F16F16F32_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  using mma_atom_shape = mma_traits::Shape_MNK;

  // mma
  static constexpr int kMmaThrLayoutM = 1;
  static constexpr int kMmaThrLayoutN = kCTAN / kWarpN;
  static constexpr int kMmaThrLayoutK = kWarp / kMmaThrLayoutN;
  using MmaThrLayout = decltype(make_layout(make_shape(
      Int<kMmaThrLayoutM>{}, Int<kMmaThrLayoutN>{}, Int<kMmaThrLayoutK>{})));

  static constexpr int kMmaPermuteM = kMmaThrLayoutM * get<0>(mma_atom_shape{});

  static constexpr int kMmaPermuteN = kCTAN;

  static constexpr int kMmaPermuteK = kMmaThrLayoutK * get<2>(mma_atom_shape{});

  static_assert(kMmaThrLayoutM * kMmaThrLayoutN * kMmaThrLayoutK == kWarp,
                "warp num mismatch");

  using kMmaPermuteNLayout =
      Layout<Shape<_2, _4, Int<kMmaThrLayoutN>, _8>, Stride<_1, _2, _64, _8>>;
  using MmaPermutations = decltype(make_tile(
      Int<kMmaPermuteM>{}, kMmaPermuteNLayout{}, Int<kMmaPermuteK>{}));
  using MMA =
      decltype(make_tiled_mma(mma_atom{}, MmaThrLayout{}, MmaPermutations{}));

  // smem A layout
  static constexpr int kShmLoadSwizzleB = 3; // 8 row
  static constexpr int kShmLoadSwizzleM = 3; // 8 fp16 = int128
  static constexpr int kShmLoadSwizzleS = cutlass::log2_down<
      kCTAK / (1 << kShmLoadSwizzleM)>::value; // 8[ctak=64]/ 16[ctak=256]

  using SmemALayoutAtom = decltype(composition(
      Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
      make_layout(make_shape(Int<8>{}, Int<kCTAK>{}),
                  make_stride(Int<kCTAK>{}, Int<1>{}))));
  using SmemALayout = decltype(tile_to_shape(
      SmemALayoutAtom{},
      make_shape(Int<kCTAM>{}, Int<kCTAK>{}, Int<kStage>{})));
  static constexpr int kASmemSize =
      cosize(SmemALayout{}) * sizeof_bits<Atype>::value / 8;
  // smem B layout

  // [ctak/16, ctan/64, 16(k), 64(n)]->
  // [ctak/16, ctan/64, 32(warp), 32(32int4=int128)]
  using SmemBLayoutAtom = decltype(make_layout(
      make_shape(Int<kCTAK / kWarpK>{}, Int<kCTAN / kWarpN>{}, _32{}, _32{}),
      LayoutRight{}));
  // [ctak/16, ctan/64, 32(warp), 32(32int4=int128), stage]
  using SmemBLayout = decltype(tile_to_shape(
      SmemBLayoutAtom{},
      make_shape(Int<kCTAK / kWarpK>{}, Int<kCTAN / kWarpN>{}, _32{}, _32{},
                 Int<kStage>{}),
      make_step(_3{}, _2{}, _1{}, _0{}, _4{})));

  static constexpr int kBSmemSize =
      cosize(SmemBLayout{}) * sizeof_bits<Btype>::value / 8;
  // smem scale layout
  // [1, ctan, stage]
  using SmemScaleLayout =
      decltype(make_layout(make_shape(_1{}, Int<kCTAN>{}, Int<Stage>{}),
                           make_stride(_0{}, _1{}, Int<kCTAN>{})));

  static constexpr int kScaleSmemSize =
      cosize(SmemScaleLayout{}) * sizeof_bits<Atype>::value / 8;

  // [16*64, num_warpn, num_warpk]
  using SmemEpilogCTAReduceLayout = decltype(make_layout(
      make_shape(Int<kWarpN * get<0>(mma_atom_shape{})>{},
                 Int<kMmaThrLayoutN>{}, Int<kMmaThrLayoutK>{}),
      LayoutLeft{}));
  // padding 8fp16(16B) to avoid bank conflict
  using SmemEpilogR2SLayout =
      decltype(make_layout(make_shape(Int<kCTAM>{}, Int<kCTAN>{}),
                           make_stride(Int<kCTAN + 8>{}, _1{})));
  // ((16, ctam/16), (32, ctan/32))  make (16, 32) sub tile continuous
  // to avoid bank conflict
  using SmemEpilogGlobalReduceCLayout =
      decltype(make_layout(make_shape(make_shape(_16{}, Int<kCTAM / 16>{}),
                                      make_shape(_32{}, Int<kCTAN / 32>{})),
                           make_stride(make_stride(_32{}, Int<16 * kCTAN>{}),
                                       make_stride(_1{}, _512{}))));
  // G2S copy
  using G2SAcopyOp = SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>;
  using G2SAcopyTraits = Copy_Traits<G2SAcopyOp>;
  using G2SAcopyAtom = Copy_Atom<G2SAcopyTraits, Atype>;
  static constexpr int G2SAvecLen =
      sizeof(cute::uint128_t) / sizeof(Atype); // 8 fp16
  static constexpr int G2SThreadK = kCTAK / G2SAvecLen;
  static constexpr int G2SThreadM = kThread / G2SThreadK;

  using G2SACopy = decltype(make_tiled_copy(
      G2SAcopyAtom{},
      make_layout(make_shape(Int<G2SThreadM>{}, Int<G2SThreadK>{}),
                  make_stride(Int<G2SThreadK>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<G2SAvecLen>{}))));

  // do not cache weight in L2
  using G2SBCopyOp = SM80_CP_ASYNC_CACHEGLOBAL_EVICT<cute::uint128_t>;
  using G2SBCopyTraits = Copy_Traits<G2SBCopyOp>;
  using G2SBCopyAtom = Copy_Atom<G2SBCopyTraits, Btype>;
  static constexpr int G2SBvecLen =
      sizeof_bits<uint128_t>::value / sizeof_bits<Btype>::value; // 32 int4

  // [8/warpn, warpn, 32(warp), 1]
  using G2SBThrLayout = decltype(make_layout(
      make_shape(Int<kMmaThrLayoutK>{}, Int<kMmaThrLayoutN>{}, _32{}, _1{}),
      LayoutRight{}));
  //[1, 1, 1, 32(32int4)]
  using G2SBThrValLayout =
      decltype(make_layout(make_shape(_1{}, _1{}, _1{}, Int<G2SBvecLen>{})));

  using G2SBCopy = decltype(make_tiled_copy(G2SBCopyAtom{}, G2SBThrLayout{},
                                            G2SBThrValLayout{}));
  // [1, ctan] tiled copy
  using G2SScaleCopyAtom = Copy_Atom<G2SBCopyTraits, Atype>;
  using G2SScaleCopy = decltype(make_tiled_copy(
      G2SScaleCopyAtom{},
      make_layout(make_shape(_1{}, Int<kCTAN / G2SAvecLen>{})),
      make_layout(make_shape(_1{}, Int<G2SAvecLen>{}))));

  // S2R copy
  using S2RACopyOp = SM75_U32x4_LDSM_N;
  using S2RACopyTraits = Copy_Traits<S2RACopyOp>;
  using S2RACopyAtom = Copy_Atom<S2RACopyTraits, Atype>;

  using S2RBCopyOp = UniversalCopy<cute::uint128_t>;
  using S2RBCopyTraits = Copy_Traits<S2RBCopyOp>;
  using S2RBCopyAtom = Copy_Atom<S2RBCopyTraits, Btype>;
  using S2RBThrLayout = G2SBThrLayout;
  using S2RBThrValLayout = G2SBThrValLayout;
  using S2RBCopy = decltype(make_tiled_copy(S2RBCopyAtom{}, S2RBThrLayout{},
                                            S2RBThrValLayout{}));

  // epilog store R2S C copy
  using R2SCCopyAtom = Copy_Atom<UniversalCopy<int>, Ctype>; // int32 = 2fp16
  using S2GCCopyAtom =
      Copy_Atom<UniversalCopy<cute::uint128_t>, Ctype>; // int128 = 8fp16
  static constexpr int R2SCvecLen =
      sizeof_bits<uint128_t>::value / sizeof_bits<Ctype>::value;
  static constexpr int R2SThreadN = kCTAN / R2SCvecLen;
  static constexpr int R2SThreadM = kThread / R2SThreadN;
  using S2GCCopy = decltype(make_tiled_copy(
      S2GCCopyAtom{},
      make_layout(make_shape(Int<R2SThreadM>{}, Int<R2SThreadN>{}),
                  make_stride(Int<R2SThreadN>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<R2SCvecLen>{}))));
  // epilog store s2g C copy
  using G2SGlobalReduceCCopyAtom =
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Ctype>;
  // only kWarp_size * kMmaThrLayoutN threads execute global reduce
  static constexpr int G2SGlobalReduceThreadN = kCTAN / R2SCvecLen;
  static constexpr int G2SGlobalReduceThreadM =
      kWarp_size * kMmaThrLayoutN / R2SThreadN;
  // epilog global reduce g2s C copy
  using G2SGlobalReduceCCopy = decltype(make_tiled_copy(
      G2SGlobalReduceCCopyAtom{},
      make_layout(make_shape(_8{}, make_shape(_4{}, Int<kMmaThrLayoutN>{})),
                  make_stride(_4{}, make_stride(_1{}, _32{}))),
      make_layout(make_shape(Int<1>{}, Int<R2SCvecLen>{}))));
  // epilog global reduce s2r C copy
  using R2GGlobalReduceCCopyAtom =
      Copy_Atom<UniversalCopy<cute::uint128_t>, Ctype>; // int128 = 8fp16
  // using S2RScaleCopyAtom = S2RBCopyAtom;
  // // per channel quant(group size = -1)
  // using S2RScaleCopyPerChannel = decltype(make_tiled_copy(
  //     S2RScaleCopyAtom{}, S2RBThrLayout{}, S2RBThrValLayout{}));
  // // per group quant(group size = 128)  [ctan] tiled copy
  // using S2RScaleCopyPerGroup = decltype(make_tiled_copy(
  //     S2RScaleCopyAtom{}, make_layout(make_shape(Int<kCTAN /
  //     G2SAvecLen>{})), make_layout(make_shape(Int<G2SAvecLen>{}))));

  // kernel Arguments
  struct Arguments {
    const void *A;   // fp16 activation [m*k]
    const void *B;   // int4 weight [k/16, n/64, 32, 32]
    void *C;         // fp16 output [m, n]
    const void *s;   // fp16 scale [k/group, n]
    void *workspace; // int32 communication lock [n/128*16]
    Coord problem_shape;
    Arguments() = delete;
    Arguments(const void *A_, const void *B_, void *C_, const void *s_,
              void *workspace_, int m, int n, int k)
        : A(A_), B(B_), C(C_), s(s_), workspace(workspace_),
          problem_shape(make_shape(m, n, k)) {}
  };

  // struct for tile descriptor
  struct TileWorkDesc {

    int tile_idx;     // 1d idx for tile
    Coord tile_coord; // [midx, nidx, kidx]

    int k_iter_begin;
    int k_iter_end;
    int k_iters_remaining;
    bool is_tile_end;
    // init tile descriptor
    DEVICE
    void init(Coord tile_shape, int tile_idx, int block_iter_begin,
              int block_iter_end) {

      auto get_tile_coord = [&](int tile_idx) {
        int n_idx = tile_idx % size<1>(tile_shape);
        int m_idx = tile_idx / size<1>(tile_shape);
        return make_shape(m_idx, n_idx, 1);
      };

      this->tile_idx = tile_idx;
      int global_k_iter_begin = tile_idx * size<2>(tile_shape);
      int global_k_iter_end = (tile_idx + 1) * size<2>(tile_shape);
      this->k_iter_begin =
          max(block_iter_begin, global_k_iter_begin) - global_k_iter_begin;
      this->k_iter_end =
          min(block_iter_end, global_k_iter_end) - global_k_iter_begin;
      this->k_iters_remaining = this->k_iter_end - this->k_iter_begin;
      this->tile_coord = get_tile_coord(tile_idx);
      this->is_tile_end = (k_iter_end == size<2>(tile_shape));
    }
    DEVICE
    bool tile_started() { return (k_iter_begin == 0); }

    DEVICE
    bool tile_finished() { return is_tile_end; }

    // check tile is splited
    DEVICE
    bool tile_splited() { return !tile_started() || !tile_finished(); }
  };

  DEVICE void operator()(Arguments &args) {
    extern __shared__ int8_t smem_ptr[];

    auto bidx = blockIdx.x, tidx = threadIdx.x;
    // auto widx = tidx >> 5, lidx = tidx & 31;
    int m = size<0>(args.problem_shape);
    int n = size<1>(args.problem_shape);
    int k = size<2>(args.problem_shape);

    // gmem tensor
    auto A_tensor = make_tensor(make_gmem_ptr<Atype>(args.A), make_shape(m, k),
                                LayoutRight{});

    auto B_tensor = make_tensor(make_gmem_ptr<Btype>(args.B),
                                make_shape(k / _16{}, n / _64{}, _32{}, _32{}),
                                LayoutRight{});
    auto C_tensor = make_tensor(make_gmem_ptr<Ctype>(args.C), make_shape(m, n),
                                LayoutRight{});
    auto scale_tensor = make_tensor(
        make_gmem_ptr<Atype>(args.s),
        make_shape(GroupSize == -1 ? 1 : k / GroupSize, n), LayoutRight{});
    int32_t *lock = reinterpret_cast<int32_t *>(args.workspace);

    auto A_pred = make_identity_tensor(shape(A_tensor));
    auto C_pred = make_identity_tensor(shape(C_tensor));
    auto scale_pred = make_identity_tensor(shape(scale_tensor));

    auto gA = local_tile(A_tensor, ATile{},
                         make_coord(_, _)); //[ctam, ctak, tile_m, tile_k]
    auto gB = local_tile(
        B_tensor, BTile{},
        make_coord(_, _, 0, 0)); //[ctak/16, ctan/64, 32, 32, tile_k, tile_n]
    auto gC = local_tile(C_tensor, CTile{},
                         make_coord(_, _)); //[ctam, ctan, tile_m, tile_n]
    auto gScale = local_tile(scale_tensor, ScaleTile{},
                             make_coord(_, _)); //[1, ctan, tile_k, tile_n]
    auto gA_pred = local_tile(A_pred, ATile{},
                              make_coord(_, _)); //[ctam, ctak, tile_m, tile_k]
    auto gC_pred = local_tile(C_pred, CTile{},
                              make_coord(_, _)); //[ctam, ctan, tile_m, tile_n]
    auto gScale_pred = local_tile(scale_pred, ScaleTile{},
                                  make_coord(_, _)); //[1, ctan, tile_k, tile_n]

    // smem tensor
    Atype *sA_ptr = reinterpret_cast<Atype *>(smem_ptr);
    Btype *sB_ptr = reinterpret_cast<Btype *>(smem_ptr + kASmemSize);
    Atype *sScale_ptr =
        reinterpret_cast<Ctype *>(smem_ptr + kASmemSize + kBSmemSize);
    ACCtype *sCTA_reduce_ptr = reinterpret_cast<ACCtype *>(smem_ptr);
    Ctype *sR2S_ptr = reinterpret_cast<Ctype *>(smem_ptr);

    auto sA = make_tensor(make_smem_ptr<Atype>(sA_ptr),
                          SmemALayout{}); // [ctam, ctak, stage]
    auto sB = make_tensor(make_smem_ptr<Btype>(sB_ptr),
                          SmemBLayout{}); // [ctak/16, ctan/64, 32, 32, stage]
    auto sScale = make_tensor(make_smem_ptr<Atype>(sScale_ptr),
                              SmemScaleLayout{}); // [1, ctan, stage]
    auto sScale_tile =
        local_tile(sScale, make_tile(_1{}, Int<G2SAvecLen>{}),
                   make_coord(_0{}, _))(_0{}, _, _, _); //[8, ctan/8, stage]
    auto sCTA_reduce = make_tensor(
        make_smem_ptr<ACCtype>(sCTA_reduce_ptr),
        SmemEpilogCTAReduceLayout{}); // [16*64, num_warpn, num_warpk]
    // [4(frag_c), 256, num_warpn, num_warpk]
    auto sCTA_reduce_tile =
        local_tile(sCTA_reduce, make_tile(_4{}), make_coord(_));
    // [ctam, ctan]
    auto sC =
        make_tensor(make_smem_ptr<Ctype>(sR2S_ptr), SmemEpilogR2SLayout{});
    // [ctam, ctan]
    auto sEpilogGlobalReduceC = make_tensor(make_smem_ptr<Ctype>(sR2S_ptr),
                                            SmemEpilogGlobalReduceCLayout{});

    int s2r_sScale_tile_idx;
    if constexpr (GroupSize != -1) {
      // (warp_idx % (ctan/64)) *  8(8fp16) + (lane_idx) / 4
      s2r_sScale_tile_idx =
          _8{} * ((tidx >> 5) % (kMmaThrLayoutN)) + ((tidx & 31) >> 2);
    } else {
      // (warp_idx % (ctan/64)) * 8(8fp16) + (lane_idx) % 4
      s2r_sScale_tile_idx =
          _8{} * ((tidx >> 5) % (kMmaThrLayoutN)) + ((tidx & 31) & 3);
    }
    // print("tidx:%d scale_idx%d\n", tidx, s2r_sScale_tile_idx);

    // mma
    MMA mma{};
    auto thr_mma = mma.get_slice(tidx);
    // alloc register for mma
    auto tArA_mma =
        thr_mma.partition_fragment_A(gA(_, _, 0, 0)); //[(2,2,2), m, k] fp16
    auto tCrC_mma =
        thr_mma.partition_fragment_C(gC(_, _, 0, 0));       //[(2,2), m, n] fp32
    auto tCrC_mma_fp16 = make_tensor_like<Ctype>(tCrC_mma); //[(2,2), m, n] fp16
    // auto tCrC_mma_view_half = recast<half>(tCrC_mma_fp16);  //[(2,2), m, n]
    // half
    auto tCrC_mma_view_half2 =
        recast<half2>(tCrC_mma_fp16); //[(1,2), m, n] half2
    auto tSrS = make_tensor<Atype>(
        make_layout(make_shape(Int<G2SAvecLen>{}, _2{}))); //[8, 2] fp16
    auto tSrS_native = recast<half>(tSrS);
    // main loop g2s copy
    G2SACopy g2s_a_copy{};
    auto thr_g2s_a_copy = g2s_a_copy.get_slice(tidx);
    auto g2s_tAgA_copy = thr_g2s_a_copy.partition_S(
        gA); // [8, cpy_m, cpy_k, tile_m, tile_k] fp16
    auto g2s_tAsA_copy =
        thr_g2s_a_copy.partition_D(sA); // [8, cpy_m, cpy_k, stage] fp16
    auto g2s_tAgA_copy_pred = thr_g2s_a_copy.partition_S(gA_pred);

    G2SBCopy g2s_b_copy{};
    auto thr_g2s_b_copy = g2s_b_copy.get_slice(tidx);
    auto g2s_tBgB_copy = thr_g2s_b_copy.partition_S(gB)(
        _, _, _, 0, 0, _, _); // [32, cpy_k, c py_n, tile_n, tile_k] int4
    auto g2s_tBsB_copy = thr_g2s_b_copy.partition_D(sB)(
        _, _, _, 0, 0, _); // [32, cpy_k, cpy_n, stage] int4

    G2SScaleCopy g2s_s_copy{};
    auto thr_g2s_s_copy = g2s_s_copy.get_slice(tidx);
    auto g2s_tSgS_copy = thr_g2s_s_copy.partition_S(
        gScale); //[8, cpy_k, cpy_n, tile_k, tile_n] fp16
    auto g2s_tSgS_copy_pred = thr_g2s_s_copy.partition_S(
        gScale_pred); //[8, cpy_k, cpy_n, tile_k, tile_n] fp16

    auto g2s_tSsS_copy =
        thr_g2s_s_copy.partition_D(sScale); //[8, cpy_k, cpy_n, stage] fp16
    //// auto g2s_tBgB_copy_pred = thr_g2s_b_copy.partition_S(gB_pred);

    // main loop s2r copy
    auto s2r_a_copy = make_tiled_copy_A(S2RACopyAtom{}, mma);
    auto thr_s2r_a_copy = s2r_a_copy.get_slice(tidx);
    auto s2r_tAsA_copy =
        thr_s2r_a_copy.partition_S(sA); // [8, cpy_m, cpy_k, stage] fp16
    auto s2r_tArA_copy =
        thr_s2r_a_copy.retile_D(tArA_mma); //[8, cpy_m, cpy_k] fp16

    S2RBCopy s2r_b_copy{};
    auto thr_s2r_b_copy = s2r_b_copy.get_slice(tidx);
    auto s2r_tBsB_copy = thr_s2r_b_copy.partition_S(sB)(
        _, _, _, 0, 0, _); // [32, cpy_k, cpy_n, stage] int4
    auto s2r_tBrB_copy =
        make_tensor_like(s2r_tBsB_copy(_, _, _, 0)); // [32, cpy_k, cpy_n] int4
    auto s2r_tBrB_copy_view = composition(
        s2r_tBrB_copy,
        select<0, 2, 1>(s2r_tBrB_copy.layout())); // [32, cpy_n, cpy_k] int4
    auto s2r_tBrB_copy_view_i32 =
        recast<int32_t>(s2r_tBrB_copy_view); // [4, cpy_n, cpy_k] int32

    // epilog store r2s C copy
    auto r2s_epilog_copy = make_tiled_copy_C(R2SCCopyAtom{}, mma);
    auto thr_r2s_epilog_copy = r2s_epilog_copy.get_slice(tidx);
    auto r2s_tCrC_fp16 =
        thr_r2s_epilog_copy.retile_S(tCrC_mma_fp16); //[8, copy_m, copy_n] fp16
    auto r2s_tCsC =
        thr_r2s_epilog_copy.partition_D(sC); //[8, copy_m, copy_n] fp16

    // epilog store s2g C copy
    S2GCCopy s2g_epilog_copy{};
    auto thr_s2g_epilog_copy = s2g_epilog_copy.get_slice(tidx);
    auto s2g_tCsC =
        thr_s2g_epilog_copy.partition_S(sC); //[8, copy_m, copy_n] fp16
    auto s2g_tCgC = thr_s2g_epilog_copy.partition_D(
        gC); //[8, copy_m, copy_n,tile_m, tile_n] fp16
    auto s2g_tCgC_pred = thr_s2g_epilog_copy.partition_D(gC_pred);

    // epilog global reduce g2s C opy
    G2SGlobalReduceCCopy g2s_epilog_global_reduce_copy{};
    auto thr_g2s_epilog_global_reduce_copy =
        g2s_epilog_global_reduce_copy.get_slice(tidx);
    auto gr_g2s_tCgC = thr_g2s_epilog_global_reduce_copy.partition_S(
        gC); //[8, copy_m, copy_n, tile_m, tile_n] fp16
    auto gr_g2s_tCgC_pred =
        thr_g2s_epilog_global_reduce_copy.partition_S(gC_pred);
    auto gr_g2s_tCsC = thr_g2s_epilog_global_reduce_copy.partition_D(
        sEpilogGlobalReduceC); //[8, copy_m, copy_n] fp16

    // tile work init
    int m_tiles_parallel = size<2>(gA);
    int k_iters = size<3>(gA);
    int n_tiles = size<3>(gC);
    const int k_inner_cnt = size<2>(tArA_mma);

    Coord tile_shape = make_shape(m_tiles_parallel, n_tiles, k_iters);
    TileWorkDesc tile_work;
    int iters_per_block =
        ceil_div(m_tiles_parallel * k_iters * n_tiles, gridDim.x);
    // make sure each block handle multiple Group Size
    if constexpr (GroupSize != -1) {
      iters_per_block = round_up(iters_per_block, GroupSize / kCTAK);
    }
    int total_iters = m_tiles_parallel * k_iters * n_tiles;

    int block_iter_begin = bidx * iters_per_block;
    int block_iter_end = min((bidx + 1) * iters_per_block, total_iters);
    // int remaining_iters = iters_per_block;
    int remaining_iters = block_iter_end - block_iter_begin; // actual k_iters
    int begin_tile_idx = block_iter_begin / k_iters;
    int end_tile_idx = (block_iter_end - 1) / k_iters;
    int tile_idx =
        min(end_tile_idx, m_tiles_parallel * n_tiles -
                              1); // reverse traversal for skew balance
    tile_work.init(tile_shape, tile_idx, block_iter_begin, block_iter_end);

    auto &tile_coord = tile_work.tile_coord;
    int g2s_k_main_loop_offset = tile_work.k_iter_begin;
    int gemm_cnt = 1;
#if 0
    // print for debug
    if (thread(0)) {
      // auto B_tensor_logical =
      //     make_tensor(make_gmem_ptr<Btype>(args.B), make_shape(k / 16, n *
      //     16),LayoutRight{});
      PRINT(args.problem_shape) PRINT(gA) PRINT(gB);
      PRINT(gC) PRINT(gScale) PRINT(sA) PRINT(sB);
      PRINT(sScale) PRINT(sScale_tile) PRINT(mma) PRINT(g2s_a_copy);
      PRINT(tArA_mma) PRINT(tCrC_mma) PRINT(tSrS);
      PRINT(g2s_tAgA_copy) PRINT(g2s_tAsA_copy) PRINT(g2s_tBgB_copy);
      PRINT(g2s_tBsB_copy) PRINT(g2s_tSgS_copy) PRINT(g2s_tSsS_copy);
      PRINT(s2r_tAsA_copy) PRINT(s2r_tArA_copy) PRINT(s2r_tBsB_copy);
      PRINT(s2r_tBrB_copy) PRINT(s2r_tBrB_copy_view);
      PRINT(s2r_tBrB_copy_view_i32) PRINT(g2s_tAgA_copy_pred);
      PRINT(g2s_tSgS_copy_pred) PRINT(tile_coord) PRINT(tile_shape);
      PRINT(tile_work.k_iter_begin) PRINT(tile_work.k_iter_end);
      PRINT(tile_work.tile_splited()) PRINT(iters_per_block);
      PRINT(tile_work.k_iters_remaining) PRINT(tCrC_mma_fp16);
      PRINT(r2s_tCrC_fp16) PRINT(r2s_tCsC);
      PRINT(coalesce(recast<half2>(tSrS)))
      PRINT(s2g_tCsC) PRINT(s2g_tCgC) PRINT(gr_g2s_tCgC) PRINT(gr_g2s_tCsC);
      PRINT(sEpilogGlobalReduceC);
      // PRINT_TENSOR(A_tensor)
      // PRINT_TENSOR(B_tensor_logical)
      // PRINT_TENSOR(gScale);
    }
    __syncthreads();
#endif

    if (tile_idx < begin_tile_idx) {
      // check cta OOB, early stop
      // if (thread(0, bidx)) {
      //   print("bidx %d early stop\n", bidx);
      // }
      return;
    }
    auto launch_g2s = [&](int pipe_idx, int k_idx, bool pred) {
      if (pred) {
        int g2s_k_idx = g2s_k_main_loop_offset + k_idx;
        // pred copy A g2s
#pragma unroll
        for (int ki = 0; ki < size<2>(g2s_tAgA_copy); ki++) {
          copy_if(
              g2s_a_copy,
              [&](auto... coords) {
                auto pred_v = g2s_tAgA_copy_pred(_, _, ki, size<0>(tile_coord),
                                                 g2s_k_idx);
                return elem_less(pred_v(_0{}, coords...), make_shape(m, k));
              },
              g2s_tAgA_copy(_, _, ki, size<0>(tile_coord), g2s_k_idx),
              g2s_tAsA_copy(_, _, ki, pipe_idx));
        }
        // copy B g2s
        copy(g2s_b_copy, g2s_tBgB_copy(_, _, _, g2s_k_idx, size<1>(tile_coord)),
             g2s_tBsB_copy(_, _, _, pipe_idx));

        // only new group k idx load scale
        if (GroupSize != -1 && pipe_idx % IterPerGroup == 0) {
          auto g2s_scale_k_idx = g2s_k_idx / IterPerGroup;
#pragma unroll
          for (int ki = 0; ki < size<1>(g2s_tSgS_copy); ki++) {
            copy_if(
                g2s_s_copy,
                [&](auto... coords) {
                  auto pred_v = g2s_tSgS_copy_pred(_, ki, _, g2s_scale_k_idx,
                                                   size<1>(tile_coord));
                  return elem_less(
                      pred_v(_0{}, coords...),
                      make_shape(g2s_scale_k_idx + 1,
                                 (size<1>(tile_coord) + 1) * kCTAN));
                },
                g2s_tSgS_copy(_, ki, _, g2s_scale_k_idx, size<1>(tile_coord)),
                g2s_tSsS_copy(_, ki, _, pipe_idx));
          }
        }
      }
      cp_async_fence();
    };

    auto wait_stage = [&] {
      cp_async_wait<kStage - 2>();
      __syncthreads();
    };

    auto launch_s2r = [&](int pipe_idx, int k_inner_idx) {
      // copy scale s2r
      if constexpr (GroupSize != -1) {
        int scale_pipe_idx = round_down(pipe_idx, IterPerGroup);
        copy_aligned(sScale_tile(_, s2r_sScale_tile_idx, scale_pipe_idx),
                     tSrS(_, k_inner_idx & 1));
      }
      // copy A s2r
      copy(s2r_a_copy, s2r_tAsA_copy(_, _, k_inner_idx & 1, pipe_idx),
           s2r_tArA_copy(_, _, k_inner_idx & 1));
      // copy B s2r
      copy(s2r_b_copy, s2r_tBsB_copy(_, k_inner_idx & 1, _, pipe_idx),
           s2r_tBrB_copy(_, k_inner_idx & 1, _));
      // __syncthreads();
      // if (thread0()) {
      //   printf("gemm %d kidx %d pipe %d\n", gemm_cnt, k_inner_idx&1,
      //   pipe_idx); PRINT_TENSOR(sScale); PRINT_TENSOR(tSrS(_, k_inner_idx &
      //   1));
      //   // PRINT_TENSOR(s2r_tArA_copy(_, _, k_inner_idx & 1));
      //   // PRINT_TENSOR(s2r_tBrB_copy_view_i32(_, _, k_inner_idx & 1));
      // }
      // __syncthreads();
    };

    auto launch_pipeline = [&] {
#pragma unroll
      for (int pipe_idx = 0; pipe_idx < kStage - 1; pipe_idx++) {
        launch_g2s(pipe_idx, pipe_idx, pipe_idx < tile_work.k_iters_remaining);
      }
      wait_stage();

      // if (thread(64)) {
      //   print("pipeline\n");
      //   // PRINT_TENSOR(gA(_, _, _0{}, _0{}));
      //   // PRINT_TENSOR(gB(_0{}, _0{}, _, _, _0{}, _0{}));
      //   // PRINT_TENSOR(sA(_, _, _0{}));
      //   // PRINT_TENSOR(sB(0, 0, _, _, _0{}));
      //   PRINT_TENSOR(sScale(_, _, _));
      //   // PRINT_TENSOR(tSrS);
      // }
      // __syncthreads();
      launch_s2r(0, 0);
      // acc set zeros
      clear(tCrC_mma);
      g2s_k_main_loop_offset += kStage - 1;
    };

    auto launch_gemm = [&](int k_inner_idx) {
#pragma unroll
      for (int n_idx = 0; n_idx < size<2>(tCrC_mma); n_idx += 2) {
        // warp tile [16, 64] = 8 * [16, 8] mma, for each thread holds
        // 8frag_b(4fp16) so int32 = 8 int4 = 2 quant frag_b(4int4), load 1
        // int32 compute 2 times mma op
        int quant_w = s2r_tBrB_copy_view_i32(n_idx >> 1, _0{}, k_inner_idx & 1);
        int quant_w_shift = quant_w >> 8;
        auto dequant_w = dequant(quant_w);
        auto dequant_w_shift = dequant(quant_w_shift);
        auto tBrB_mma_col0 = recast<Atype>(dequant_w);
        auto tBrB_mma_col1 = recast<Atype>(dequant_w_shift);
        if constexpr (GroupSize != -1) {
          half2 scale0_pack = __half2half2(tSrS_native(n_idx, k_inner_idx & 1));
          half2 scale1_pack =
              __half2half2(tSrS_native(n_idx + 1, k_inner_idx & 1));
#pragma unroll
          for (int i = 0; i < size<0>(dequant_w); i++) {
            dequant_w(i) = __hmul2(dequant_w(i), scale0_pack);
            dequant_w_shift(i) = __hmul2(dequant_w_shift(i), scale1_pack);
          }
        }
        // __syncthreads();
        // if (thread(0)) {
        //   printf("gemm %d nidx %d\n", gemm_cnt, n_idx);

        //   PRINT(tSrS(n_idx, k_inner_idx & 1));
        //   PRINT(tSrS(n_idx + 1, k_inner_idx & 1));
        //   PRINT_TENSOR(tArA_mma(_, _0{}, k_inner_idx & 1));
        //   PRINT(s2r_tBrB_copy_view_i32(n_idx >> 1, _0{}, k_inner_idx & 1));
        //   // PRINT_TENSOR(tBrB_mma_col0);
        //   // PRINT_TENSOR(tBrB_mma_col1);
        // }
        // __syncthreads();
#pragma unroll
        for (int m_idx = 0; m_idx < size<1>(tCrC_mma); m_idx++) {
          gemm(mma, tArA_mma(_, m_idx, k_inner_idx & 1), tBrB_mma_col0,
               tCrC_mma(_, m_idx, n_idx));
          gemm(mma, tArA_mma(_, m_idx, k_inner_idx & 1), tBrB_mma_col1,
               tCrC_mma(_, m_idx, n_idx + 1));
        }
      }
    };
    // multiple warp compute partial sum of one cta, so need to reduce intra-cta
    // result
    auto launch_epilog_cta_reduce = [&]() {
      // when tile mma thread k > 1, need to reduce partial sum
      if constexpr (kMmaThrLayoutK > 1) {
        int epilog_r2s_cta_reduce_idx = (tidx & 31);
        constexpr int epilog_r2s_cta_reduce_offset =
            kWarp_size; // r2s store offset
        int warpn_idx = (tidx >> 5) % kMmaThrLayoutN;
        int warpk_idx = (tidx >> 5) / kMmaThrLayoutN;

#pragma unroll
        for (int m_idx = 0; m_idx < size<1>(tCrC_mma); m_idx++) {
#pragma unroll
          for (int warp_offset = kMmaThrLayoutK / 2; warp_offset > 0;
               warp_offset >>= 1) {
            // Parallel logarithmic shared memory reduction.
            if (warp_offset <= warpk_idx && warpk_idx < 2 * warp_offset) {
              int reducek_idx = warpk_idx - warp_offset;
#pragma unroll
              for (int n_idx = 0; n_idx < size<2>(tCrC_mma); n_idx++) {
                int r2s_idx = epilog_r2s_cta_reduce_idx +
                              n_idx * epilog_r2s_cta_reduce_offset;
                if (warp_offset < kMmaThrLayoutK / 2) {
                  auto partial_sum0 =
                      make_tensor_like(tCrC_mma(_, m_idx, n_idx));
                  auto partial_sum1 = make_tensor_like(partial_sum0);
                  copy_aligned(
                      sCTA_reduce_tile(_, r2s_idx, warpn_idx, 2 * reducek_idx),
                      partial_sum0);
                  copy_aligned(sCTA_reduce_tile(_, r2s_idx, warpn_idx,
                                                2 * reducek_idx + 1),
                               partial_sum1);
#pragma unroll
                  for (int fragc_idx = 0; fragc_idx < size(partial_sum0);
                       fragc_idx++) {
                    tCrC_mma(fragc_idx, m_idx, n_idx) +=
                        partial_sum0(fragc_idx) + partial_sum1(fragc_idx);
                  }
                }
                copy_aligned(
                    tCrC_mma(_, m_idx, n_idx),
                    sCTA_reduce_tile(_, r2s_idx, warpn_idx, reducek_idx));
              }
            }
            __syncthreads();
          }
          if (warpk_idx == 0) {
#pragma unroll
            for (int n_idx = 0; n_idx < size<2>(tCrC_mma); n_idx++) {
              int r2s_idx = epilog_r2s_cta_reduce_idx +
                            n_idx * epilog_r2s_cta_reduce_offset;
              auto partial_sum = make_tensor_like(tCrC_mma(_, m_idx, n_idx));
              copy_aligned(sCTA_reduce_tile(_, r2s_idx, warpn_idx, _0{}),
                           partial_sum);
#pragma unroll
              for (int fragc_idx = 0; fragc_idx < size(partial_sum);
                   fragc_idx++) {
                tCrC_mma(fragc_idx, m_idx, n_idx) += partial_sum(fragc_idx);
              }
            }
          }
          __syncthreads();
        }
      }
    };
    // store frag c back to dram
    auto launch_epilog_r2s2g = [&]() {
      if (tidx < kWarp_size * kMmaThrLayoutN) {
        auto tSrS_view_half2 = coalesce(recast<half2>(tSrS)); //[8] half2
// cvt fp32 to fp16
#pragma unroll
        for (int m_idx = 0; m_idx < size<1>(tCrC_mma); m_idx++) {
#pragma unroll
          for (int n_idx = 0; n_idx < size<2>(tCrC_mma); n_idx++) {
#pragma unroll
            for (int i = 0; i < size<0>(tCrC_mma); i += 2) {
              half2 frag_c =
                  __halves2half2(__float2half(tCrC_mma(i, m_idx, n_idx)),
                                 __float2half(tCrC_mma(i + 1, m_idx, n_idx)));
              if constexpr (GroupSize == -1) {
                frag_c = __hmul2(frag_c, tSrS_view_half2(n_idx));
              }
              tCrC_mma_view_half2(i >> 1, m_idx, n_idx) = frag_c;
            }
          }
        }
        // __syncthreads();
        // if (thread(28)) {
        //   print("tidx %dcheck frag c to fp16\n", threadIdx.x);
        //   PRINT_TENSOR(coalesce(tCrC_mma_fp16));
        // }
        // __syncthreads();
        copy(r2s_epilog_copy, r2s_tCrC_fp16, r2s_tCsC);
      }
      __syncthreads();
      // if (thread(0)) {
      //   PRINT_TENSOR(sC);
      // }
      // __syncthreads();
#pragma unroll
      for (int n_idx = 0; n_idx < size<2>(s2g_tCsC); n_idx++) {
        copy_if(
            s2g_epilog_copy,
            [&](auto... coords) {
              auto pred_v = s2g_tCgC_pred(_, _, n_idx, size<0>(tile_coord),
                                          size<1>(tile_coord));
              return elem_less(pred_v(_0{}, coords...), shape(C_tensor));
            },
            s2g_tCsC(_, _, n_idx),
            s2g_tCgC(_, _, n_idx, size<0>(tile_coord), size<1>(tile_coord)));
      }
    };
    // multiple cta compute partial sum of one tile, so need to reduce inter-cta
    // result
    auto launch_epilog_global_reduce = [&]() {
      if (tidx < kWarp_size * kMmaThrLayoutN) {

        if (!tile_work.tile_started()) {
// g2s load C partial sum
#pragma unroll
          for (int n_idx = 0; n_idx < size<2>(gr_g2s_tCgC); n_idx++) {
            copy_if(
                g2s_epilog_global_reduce_copy,
                [&](auto... coords) {
                  auto pred_v = gr_g2s_tCgC_pred(
                      _, _, n_idx, size<0>(tile_coord), size<1>(tile_coord));
                  return elem_less(pred_v(_0{}, coords...), shape(C_tensor));
                },
                gr_g2s_tCgC(_, _, n_idx, size<0>(tile_coord),
                            size<1>(tile_coord)),
                gr_g2s_tCsC(_, _, n_idx));
          }
          cp_async_fence();
          cp_async_wait<0>();
        }

        auto new_tCrC_half =
            make_tensor_like<half>(gr_g2s_tCsC); //((_8,_1),mma_m*2, 2) half

        // frag c((2[n],2[m]), mma_m, 8[n])
        // store to new frag c(8[n], 2[m]*mma_m, 2[n])
#pragma unroll
        for (int mx2_idx = 0; mx2_idx < size<1>(new_tCrC_half); mx2_idx++) {
          int mma_m_idx = mx2_idx >> 1, row_m_idx = (mx2_idx & 1) * 2;

          if (!tile_work.tile_started()) {
            copy_aligned(gr_g2s_tCsC(_, mx2_idx, _),
                         new_tCrC_half(_, mx2_idx, _));
#pragma unroll
            for (int n8_idx = 0; n8_idx < size<0>(new_tCrC_half); n8_idx++) {
#pragma unroll
              for (int n2_idx = 0; n2_idx < size<2>(new_tCrC_half); n2_idx++) {
                tCrC_mma(n2_idx + row_m_idx, mma_m_idx, n8_idx) +=
                    __half2float(new_tCrC_half(n8_idx, mx2_idx, n2_idx));
              }
            }
          }
          // store C partial sum to dram
          if (!tile_work.tile_finished()) {
#pragma unroll
            for (int n8_idx = 0; n8_idx < size<0>(new_tCrC_half); n8_idx++) {
#pragma unroll
              for (int n2_idx = 0; n2_idx < size<2>(new_tCrC_half); n2_idx++) {
                new_tCrC_half(n8_idx, mx2_idx, n2_idx) = __float2half(
                    tCrC_mma(n2_idx + row_m_idx, mma_m_idx, n8_idx));
              }
            }
            copy_if(
                R2GGlobalReduceCCopyAtom{},
                [&](auto... coords) {
                  auto pred_v = gr_g2s_tCgC_pred(
                      _, mx2_idx, _, size<0>(tile_coord), size<1>(tile_coord));
                  return elem_less(pred_v(_0{}, coords...), shape(C_tensor));
                },
                new_tCrC_half(_, mx2_idx, _),
                gr_g2s_tCgC(_, mx2_idx, _, size<0>(tile_coord),
                            size<1>(tile_coord)));
          }
        }
      }
    };

    // start to compute

    launch_pipeline();
#pragma unroll 1
    while (true) {
      // launch main loop
      remaining_iters -= tile_work.k_iters_remaining;
#pragma unroll 1
      while (tile_work.k_iters_remaining) {
#pragma unroll
        for (int pipe_idx = 0; pipe_idx < kStage;) {
#pragma unroll
          for (int k_inner_idx = 0; k_inner_idx < k_inner_cnt; k_inner_idx++) {
            launch_s2r(pipe_idx % kStage, k_inner_idx + 1);
            if (k_inner_idx == k_inner_cnt - 2) {
              // load next stage
              launch_g2s((pipe_idx + kStage - 1) % kStage, pipe_idx,
                         tile_work.k_iters_remaining >= kStage);
              pipe_idx++;
              wait_stage();
            }
            launch_gemm(k_inner_idx);
            gemm_cnt++;
          }
          tile_work.k_iters_remaining--;
          if (tile_work.k_iters_remaining == 0) {
            break;
          }
        }
        g2s_k_main_loop_offset += kStage;
      }
      // __syncthreads();
      // if (thread(0)) {
      //   print("tidx %d check mainloop result ", threadIdx.x);
      //   PRINT_TENSOR(coalesce(tCrC_mma));
      // }
      // launch epilog
      cp_async_wait<0>(); // clean all cp_async queue
      if (GroupSize == -1 && tile_work.tile_finished()) {
        // g2s load per_channel quant scale
        copy_if(
            g2s_s_copy,
            [&](auto... coords) {
              auto pred_v =
                  g2s_tSgS_copy_pred(_, _0{}, _, _0{}, size<1>(tile_coord));
              return elem_less(pred_v(_0{}, coords...), shape(scale_tensor));
            },
            g2s_tSgS_copy(_, _0{}, _, _0{}, size<1>(tile_coord)),
            g2s_tSsS_copy(_, _0{}, _, _0{}));
        cp_async_fence();
      }
      // epilog intra-cta reduce
      launch_epilog_cta_reduce();
      if (GroupSize == -1 && tile_work.tile_finished()) {
        cp_async_wait<0>();
        __syncthreads();
        if (tidx < kMmaThrLayoutN * kWarp_size) {
          copy_aligned(sScale_tile(_, s2r_sScale_tile_idx, _0{}),
                       tSrS(_, _0{}));
          copy_aligned(sScale_tile(_, s2r_sScale_tile_idx + _4{}, _0{}),
                       tSrS(_, _1{}));
        }
      }

      // epilog inter-cta global reduce
      if (tile_work.tile_splited()) {
        int cur_tile_first_block = tile_idx * k_iters / iters_per_block;
        int wait_block = bidx - cur_tile_first_block;
        // wait prev block
        Barrier::wait_eq(lock, tidx, tile_idx, wait_block);
        launch_epilog_global_reduce();
        if (tile_work.tile_finished()) {
          // reset lock 0
          if (tidx == 0) {
            lock[tile_idx] = 0;
          }
        } else {
          // lock count ++
          Barrier::arrive_inc(lock, tidx, tile_idx);
        }
      }

      // epilog r2s2g store

      if (tile_work.tile_finished()) {
        launch_epilog_r2s2g();
      }
      // if (threadIdx.x == 0) {
      //   printf("bidx %d tile idx %d finished \n", bidx, tile_idx);
      // }
      if (remaining_iters == 0) {
        break;
      }
      // init next tile
      tile_idx--;
      tile_work.init(tile_shape, tile_idx, block_iter_begin,
                     block_iter_begin + remaining_iters);
      g2s_k_main_loop_offset = tile_work.k_iter_begin;
      launch_pipeline();
    }
  }
};
