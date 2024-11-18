#pragma once
// copy from cutlass/include/cutlass/barrier.h
#include "common.h"

struct SyncthreadsSync {
  DEVICE
  static void sync() {
    __syncthreads();
  }
};

template <class Sync>
struct GenericBarrier {

public:

  /// Flag type
  using T = int;

  /// Initial flag value
  static const T INIT = 0;


protected:

  /// Load flag, as a strong acquire operation (int specialization)
  CUTLASS_DEVICE
  static int ld_acquire(int *ptr)
  {
    int state = 0;

#if (__CUDA_ARCH__ >= 700)
    /// SM70 and newer use memory consistency qualifiers

    // Acquire pattern using acquire modifier
    asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));

#else
    asm volatile ("ld.cg.global.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
#endif // (__CUDA_ARCH__ >= 700)

    return state;
  }


  /// Reduce into flag, with release pattern (int specialization)
  CUTLASS_DEVICE
  static void red_release(int *ptr, int val)
  {
#if (__CUDA_ARCH__ >= 700)
    /// SM70 and newer use memory consistency qualifiers

    // Release pattern using acq_rel fence + relaxed modifier.  (The fence also releases data
    // that was weakly-written by other threads prior to the last syncthreads)
    asm volatile ("fence.acq_rel.gpu;\n");
    asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(ptr), "r"(val));

#else
    __threadfence();
    atomicAdd(ptr, val);
#endif // (__CUDA_ARCH__ >= 700)
  }


public:

  /// Uses thread[0] to wait for at least the specified count of signals on the given flag counter
  CUTLASS_DEVICE
  static void wait_lt(void *lock_ptr, int thread_idx, int flag_idx, int count)
  {
    T *flag_ptr = reinterpret_cast<T*>(lock_ptr) + flag_idx;

    if (thread_idx == 0)
    {
        // Spin-loop
        #pragma unroll 1
        while(ld_acquire(flag_ptr) < count) {}
    }

    Sync::sync();
  }

  /// Uses thread[0] to wait for at least the specified count of signals on the given flag counter
  CUTLASS_DEVICE
  static void wait_eq(void *lock_ptr, int thread_idx, int flag_idx, T val = 1)
  {
    T *flag_ptr = reinterpret_cast<T*>(lock_ptr) + flag_idx;

    if (thread_idx == 0)
    {
        // Spin-loop
        #pragma unroll 1
        while(ld_acquire(flag_ptr) != val) {}
    }
    Sync::sync();
  }

  /// Uses thread[0] to wait for the specified count of signals on the given flag counter
  CUTLASS_DEVICE
  static void wait_eq_reset(void *lock_ptr, int thread_idx, int flag_idx, T val = 1) {
    T *flag_ptr = reinterpret_cast<T*>(lock_ptr) + flag_idx;

    if (thread_idx == 0)
    {
        // Spin-loop
        #pragma unroll 1
        while(atomicCAS(flag_ptr, val, 0) != val) {}
    }

    Sync::sync();
  }

  /// Increment the arrival count for a flag
  CUTLASS_DEVICE
  static void arrive_inc(void *lock_ptr, int thread_idx, int flag_idx, int val = 1)
  {
    T* flag_ptr = reinterpret_cast<T*>(lock_ptr) + flag_idx;

    Sync::sync();

    if (thread_idx == 0)
    {
      red_release(flag_ptr, val);
    }
  }


  /// Increment the arrival counts for a range of flags
  CUTLASS_DEVICE
  static void arrive_range_inc(void *lock_ptr, int thread_idx, int first_flag_idx, int count = 1, int val = 1)
  {
    int flag_idx = first_flag_idx + thread_idx;
    T* flag_ptr = reinterpret_cast<T*>(lock_ptr) + flag_idx;

    // Barrier to make sure all other threads in group have written their data
    Sync::sync();

    // Select threads increment their flags
    if (thread_idx < count) {
      red_release(flag_ptr, val);
    }
  }
};

using Barrier = GenericBarrier<SyncthreadsSync>;