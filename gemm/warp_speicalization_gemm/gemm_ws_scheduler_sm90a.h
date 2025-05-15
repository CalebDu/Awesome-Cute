#pragma once

#include <common.h>
#include <cute/tensor.hpp>
using namespace cute;

class PersistentTileScheduler {
  // default alongN cta swizzling
public:
  struct TileInfo {
    uint32_t m_idx;
    uint32_t n_idx;
    bool is_valid;
  };
  struct Param {
    int swizzle_ = 0; // swizzle size
    int problem_tiles;
    int cluster_size;
    int cluster_n_shape;  // n cluster
    int cluster_m_shape;  // m cluster
    int cluster_n_blocks; // n blocks
    int k_loop_cnt;
    template <typename Problem_size, typename CTA_tile, typename Cluster_shape>
    Param(Problem_size problem_size, CTA_tile tile_size,
          Cluster_shape cluster_shape, int swizzle)
        : swizzle_(swizzle), cluster_n_shape(size<1>(cluster_shape)),
          cluster_m_shape(size<0>(cluster_shape)) {
      assert((swizzle_ & (swizzle_ - 1)) == 0 &&
             "tile scheduler swizzle is limited to power of 2");
      cluster_size = size<0>(cluster_shape) * size<1>(cluster_shape);
      auto tiles_m = ceil_div(size<0>(problem_size), size<0>(tile_size));
      auto tiles_n = ceil_div(size<1>(problem_size), size<1>(tile_size));
      k_loop_cnt = ceil_div(size<2>(problem_size), size<2>(tile_size));
      // align up cluster tiles
      auto align_tiles_m = round_up(tiles_m, size<0>(cluster_shape) * swizzle_);
      auto align_tiles_n = round_up(tiles_n, size<1>(cluster_shape));

      cluster_n_blocks = align_tiles_n / cluster_n_shape;
      problem_tiles = align_tiles_m * align_tiles_n;
      // PRINT(align_tiles_m);
      // PRINT(align_tiles_n);
      // PRINT(problem_tiles);
    }
  };

  DEVICE
  PersistentTileScheduler(Param const &param) : param_(param) {
    linear_block_idx = blockIdx.x;
    linear_grid_size = gridDim.x * gridDim.y * gridDim.z;
    iter_idx = 0;
    // if(thread(blockDim.x-1)) {
    //   PRINT(linear_block_idx);
    //   PRINT(linear_grid_size);
    //   PRINT(param_.problem_tiles);
    // }
  }

  DEVICE
  TileInfo get_tile_id() {
    if (linear_block_idx >= param_.problem_tiles) {
      return {-1u, -1u, false};
    }
    auto linear_cluster_idx = linear_block_idx / param_.cluster_size;
    auto block_offset = linear_block_idx % param_.cluster_size;

    auto block_m_offset = block_offset % param_.cluster_m_shape;
    auto block_n_offset = block_offset / param_.cluster_m_shape;

    auto cluster_swizzle_offset = linear_cluster_idx % param_.swizzle_;
    auto cluster_swizzle_extra = linear_cluster_idx / param_.swizzle_;

    auto cluster_m_swizzle_offset =
        cluster_swizzle_extra / param_.cluster_n_blocks;
    auto cluster_m_idx =
        cluster_m_swizzle_offset * param_.swizzle_ + cluster_swizzle_offset;
    auto cluster_n_idx = cluster_swizzle_extra % param_.cluster_n_blocks;

    auto tile_m = cluster_m_idx * param_.cluster_m_shape + block_m_offset;
    auto tile_n = cluster_n_idx * param_.cluster_n_shape + block_n_offset;

    return {tile_m, tile_n, true};
  }

  DEVICE
  void advance_next_tile() {
    // stride grid size to next tile
    linear_block_idx += linear_grid_size;
    iter_idx += 1;
  }

  template <typename Cluster_shape>
  static dim3 get_grid_dim(Cluster_shape cluster_shape) {
    cudaDeviceProp prop;
    int device_id;
    cudaGetDevice(&device_id);
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    auto cluster_size = size<0>(cluster_shape) * size<1>(cluster_shape);
    auto sm_count = prop.multiProcessorCount;
    auto grid_size = sm_count / cluster_size * cluster_size;
    return dim3(grid_size, 1, 1);
  }

private:
  uint32_t linear_block_idx;
  uint32_t linear_grid_size;
  uint32_t iter_idx;

public:
  Param param_;
};