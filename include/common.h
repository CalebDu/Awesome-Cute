#pragma once

#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"
// #include "cutlass/cutlass.h"
// #include "cutlass/util/reference/host/gemm.h"
// #include "cutlass/util/reference/host/tensor_compare.h"
// #include "cutlass/util/reference/host/tensor_copy.h"

#define ceil_div(x, y) ((x) + (y) - 1) / (y)
#define round_up(x, y) ceil_div(x, y) * (y)

using namespace cute;

template <typename T> inline auto make_A_cutlass_tensor(int m, int k) {
  cutlass::HostTensor<T, cutlass::layout::RowMajor> A(
      cutlass::MatrixCoord({m, k}));
  return A;
}

template <typename T> inline auto make_B_cutlass_tensor(int n, int k) {
  cutlass::HostTensor<T, cutlass::layout::ColumnMajor> B(
      cutlass::MatrixCoord({k, n}));
  return B;
}

template <typename T> inline auto make_C_cutlass_tensor(int m, int n) {
  cutlass::HostTensor<T, cutlass::layout::RowMajor> C(
      cutlass::MatrixCoord({m, n}));
  return C;
}

template <class... CopyArgs, class PredTensor, class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout, class StripTuple, class ZfillTuple>
__device__ __forceinline__ static void
copy_strip_zfill(Copy_Atom<CopyArgs...> const &copy, PredTensor const &pred,
                 Tensor<SrcEngine, SrcLayout> const &src,
                 Tensor<DstEngine, DstLayout> dst,
                 StripTuple const &strip_bound, ZfillTuple const &zfill_bound) {
  static_assert(SrcLayout::rank == DstLayout::rank,
                "dst and src mismatch rank ");
  constexpr int Rank = SrcLayout::rank;
  // print_type(Rank);
  auto src_v = group_modes<1, Rank>(src);   // [copy, copy_m * copy_n]
  auto dst_v = group_modes<1, Rank>(dst);   // [copy, copy_m * copy_n]
  auto pred_v = group_modes<1, Rank>(pred); // [copy, copy_m * copy_n]
#pragma unroll
  for (int idx = 0; idx < size<1>(pred_v); idx++) {
    auto pred_coord = pred_v(_0{}, idx);
    // strip data OOB block tile
    if (elem_less(pred_coord, strip_bound)) {
      // fill zeros OOB global shape into block tile
      copy_if(
          copy,
          [&](auto... coords) {
            return elem_less(pred_v(_0{}, coords...), zfill_bound);
          },
          src_v(_, _), dst_v(_, _));
    }
  }
}
