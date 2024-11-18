// reference:
// cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h

#pragma once
#include "common.h"
#include "cute/tensor.hpp"
using namespace cute;

// 2 strategy based on streamk paper: (2tile sk + dp), (1tile sk + dp)
enum class SK_DP_Block_Strategy { sk1tile_dp, sk2tile_dp };

//  streamk + data parallel hybridization
template <class CTA_tile> struct SK_DP_Block_Wrapper {
  using Gemm_Shape = Shape<int, int, int>;
  static constexpr int kCTAM = size<0>(CTA_tile{});
  static constexpr int kCTAN = size<1>(CTA_tile{});
  static constexpr int kCTAK = size<2>(CTA_tile{});
  Gemm_Shape problem_shape;
  Gemm_Shape tile_shape;
  int sm_occupancy;
  int device_sm;
  int avail_sms;
  int dp_blocks;
  int sk_blocks;
  int sk_tiles;
  int iter_per_sk_normal_block;
  int iter_per_sk_big_block;
  int iter_per_tile;
  int sk_iter;
  int sk_big_blocks;
  SK_DP_Block_Strategy strategy;

  SK_DP_Block_Wrapper() = default;
  SK_DP_Block_Wrapper(
      Gemm_Shape prob_shape_, int const sm_occupancy_, int const device_sm_,
      int const avail_sms_,
      SK_DP_Block_Strategy strategy_ = SK_DP_Block_Strategy::sk2tile_dp)
      : problem_shape(prob_shape_), sm_occupancy(sm_occupancy_),
        device_sm(device_sm_), avail_sms(avail_sms_), dp_blocks(0),
        sk_blocks(0), sk_tiles(0), iter_per_sk_normal_block(0),
        iter_per_sk_big_block(0), iter_per_tile(0), sk_iter(0),
        sk_big_blocks(0), strategy(strategy_) {
    //
    int m = size<0>(problem_shape);
    int n = size<1>(problem_shape);
    int k = size<2>(problem_shape);

    int m_tile = ceil_div(m, kCTAM);
    int n_tile = ceil_div(n, kCTAN);
    iter_per_tile = ceil_div(k, kCTAK);
    tile_shape = make_shape(m_tile, n_tile, iter_per_tile);
    int output_tiles = m_tile * n_tile;
    // int n_wave = ceil_div(output_tiles, avail_sms);
    int dp_tiles;

    get_blocks(dp_tiles, sk_blocks, output_tiles, iter_per_tile, avail_sms,
               sm_occupancy, strategy);
    sk_tiles = output_tiles - dp_tiles;
    dp_blocks = dp_tiles;
    if (dp_tiles < 0) { // illegal
      dp_tiles = output_tiles;
      sk_tiles = 0;
      sk_blocks = 0;
      dp_blocks = dp_tiles;
    }
    printf("output tiles: %d, dp tiles: %d, sk tiles: %d\n", output_tiles,
           dp_tiles, sk_tiles);
    printf("dp blocks: %d, sk blocks: %d\n", dp_blocks, sk_blocks);
    // enable streamk
    if (sk_blocks) {
      sk_iter = sk_tiles * iter_per_tile;
      sk_blocks = min(sk_blocks, sk_iter);
      iter_per_sk_normal_block = sk_iter / sk_blocks;
      sk_big_blocks = sk_iter - iter_per_sk_normal_block * sk_blocks;
      iter_per_sk_big_block = iter_per_sk_normal_block + 1;
      // printf("iter_per_sk_normal_block %d\n", iter_per_sk_normal_block);
      // printf("sk_big_blocks %d\n", sk_big_blocks);
      // printf("iter_per_sk_big_block %d\n", iter_per_sk_big_block);
    }
  }

  static void get_blocks(int &dp_tiles, int &sk_blocks, int output_tiles,
                         int iter_per_tile, int avail_sms, int sm_occupancy,
                         SK_DP_Block_Strategy strategy) {
    dp_tiles = output_tiles;
    int full_waves = output_tiles / avail_sms;
    int full_wave_tiles = full_waves * avail_sms;
    int partial_wave_tiles = output_tiles - full_wave_tiles;
    int score = -1;
    if (partial_wave_tiles == 0) {
      // Perfect quantization
      return;
    }
    if (strategy == SK_DP_Block_Strategy::sk1tile_dp) {
      int max_sk_occupancy = sm_occupancy - ((full_waves) % sm_occupancy);
      dp_tiles = full_wave_tiles;

      get_sk_blocks(sk_blocks, score, partial_wave_tiles, iter_per_tile,
                    avail_sms, max_sk_occupancy, true);
      if (score < 0) {
        printf("disable streamk\n");
        sk_blocks = 0;
        dp_tiles = output_tiles;
      }
    } else if (strategy == SK_DP_Block_Strategy::sk2tile_dp) {
      int max_sk_occupancy = sm_occupancy - ((full_waves - 1) % sm_occupancy);
      dp_tiles = full_wave_tiles - avail_sms;
      get_sk_blocks(sk_blocks, score, partial_wave_tiles + avail_sms,
                    iter_per_tile, avail_sms, max_sk_occupancy, true);
      if (score < 0) {
        printf("disable streamk\n");
        sk_blocks = 0;
        dp_tiles = output_tiles;
      }
    }
  }

  // Compute sk_blocks to dispatch for a given number of sk_tiles
  static void get_sk_blocks(int &sk_blocks,     /// [out]
                            int &savings_iters, /// [out]
                            int sk_tiles, int iters_per_tile, int avail_sms,
                            int max_sk_occupancy, bool allow_partial_wave) {
    savings_iters = INT_MIN;
    sk_blocks = 0;

    if (sk_tiles == 0) {
      return;
    }

    int sk_iters = sk_tiles * iters_per_tile;

    int dp_equiv_waves = (sk_tiles + avail_sms - 1) / avail_sms;
    int dp_equiv_iters = iters_per_tile * dp_equiv_waves;

    int min_sk_blocks =
        (allow_partial_wave) ? min(avail_sms, sk_tiles + 1) : avail_sms;
    int max_sk_blocks = min(avail_sms * max_sk_occupancy,
                            sk_iters / 2); // kMinItersPerSkBlock = 2

    for (int trial_sk_blocks = min_sk_blocks; trial_sk_blocks <= max_sk_blocks;
         ++trial_sk_blocks) {
      int sk_waves = (trial_sk_blocks + avail_sms - 1) / avail_sms;
      int max_sk_iters_per_block =
          (sk_iters + trial_sk_blocks - 1) / trial_sk_blocks;
      int sk_iter_equiv = max_sk_iters_per_block * sk_waves;

      int num_peers = ((trial_sk_blocks + sk_tiles - 1) / sk_tiles) +
                      1; // add one for alignment skew

      float iter_cost = 0.02f * float(num_peers) * float(sk_iter_equiv);

      if (trial_sk_blocks % sk_tiles == 0) {
        // aligned
        num_peers = (trial_sk_blocks / sk_tiles);

        iter_cost = 0.0f;
      }

      float peer_cost = 2.0f * float(num_peers);

      float base_cost = 2.0f * float(sk_waves);

      int fixup_iter_equiv = int(base_cost + iter_cost + peer_cost);

      int trial_savings_iters =
          dp_equiv_iters - sk_iter_equiv - fixup_iter_equiv;

      if (trial_savings_iters >= savings_iters) {
        savings_iters = trial_savings_iters;
        sk_blocks = trial_sk_blocks;
      }
    }
  }

  int get_blocks_nums() { return dp_blocks + sk_blocks; }

  dim3 get_grid_dims() { return dim3(get_blocks_nums(), 1, 1); }

  DEVICE
  void get_iter_extents(int sk_block_idx, int &block_iter_begin,
                        int &block_iter_end) {
    if (sk_block_idx < sk_big_blocks) {
      block_iter_begin = sk_block_idx * iter_per_sk_big_block;
      block_iter_end = block_iter_begin + iter_per_sk_big_block;
    } else {
      block_iter_begin =
          sk_block_idx * iter_per_sk_normal_block + sk_big_blocks;
      block_iter_end = block_iter_begin + iter_per_sk_normal_block;
    }
  }

  DEVICE
  int get_sk_tile_idx(int iter_idx) const { return iter_idx / iter_per_tile; }

  DEVICE
  Gemm_Shape get_tile_offset(int tile_idx) const {
    int m_idx, n_idx;
    // column-major
    n_idx = tile_idx / size<0>(tile_shape);
    m_idx = tile_idx % size<0>(tile_shape);

    return make_shape(m_idx, n_idx, 1);
  }

  DEVICE
  int get_sk_block_idx(int iter) const {
    int iter_big_block = sk_big_blocks * iter_per_sk_big_block;
    int iter_normal_block = iter - iter_big_block;

    int big_block_idx = iter / iter_per_sk_big_block;
    int norm_block_idx =
        sk_big_blocks + iter_normal_block / iter_per_sk_normal_block;
    int sk_block_idx =
        (big_block_idx < sk_big_blocks) ? big_block_idx : norm_block_idx;
    return sk_block_idx;
  }
};