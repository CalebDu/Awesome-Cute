#include "common.h"
#include "marlin_cute_trait.h"

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

constexpr int stage = 4;
constexpr int cta_size = 256;

#define CALL_IF(THREAD_M, THREAD_N, THREAD_K, GROUP_SIZE)                      \
  else if (thread_m == THREAD_M && thread_n == THREAD_N &&                     \
           thread_k == THREAD_K && groupsize == GROUP_SIZE) {                  \
    using CTA_Tile = decltype(make_shape(Int<THREAD_M>{}, Int<THREAD_N>{},     \
                                         Int<THREAD_K>{}));                    \
    using Trait = MarlinGemmTraits<CTA_Tile, stage, GROUP_SIZE>;               \
    using Arguments = Trait::Arguments;                                        \
    Arguments args(A_ptr, B_ptr, C_ptr, s_ptr, locks, prob_m, prob_n, prob_k); \
    config_smem(launch<Trait>, max_smem_size);                                 \
    launch<Trait><<<blocks, cta_size, max_smem_size, stream>>>(args);          \
  }

template <typename KernelTrait>
__global__ void launch(typename KernelTrait::Arguments args) {
  KernelTrait kernel;
  kernel(args);
}

int marlin_cute(const void *A, const void *B, void *C, void *s, int prob_m,
                int prob_n, int prob_k, void *workspace, int groupsize = -1,
                int dev = 0, cudaStream_t stream = 0, int thread_k = -1,
                int thread_n = -1, int sms = -1, int max_par = 16) {
  int tot_m = prob_m;
  int tot_m_blocks = ceil_div(tot_m, 16);
  int pad = 16 * tot_m_blocks - tot_m;
  int max_smem_size = get_max_smem_size();
  int thread_m;
  if (sms == -1)
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (thread_k == -1 || thread_n == -1) {
    if (prob_m <= 16) {
      // For small batchizes, better partioning is slightly more important than
      // better compute utilization
      thread_k = 128;
      thread_n = 128;
    } else {
      thread_k = 64;
      thread_n = 256;
    }
  }

  // int thread_k_blocks = thread_k / 16;
  // int thread_n_blocks = thread_n / 16;
  int group_blocks = (groupsize == -1) ? -1 : groupsize / 16;
  int blocks = sms;

  if (prob_n % thread_n != 0 || prob_k % thread_k != 0 ||
      (group_blocks != -1 && prob_k % group_blocks != 0))
    return ERR_PROB_SHAPE;
  if (prob_m == 0 || prob_n == 0 || prob_k == 0)
    return 0;

  const int4 *A_ptr = (const int4 *)A;
  const int4 *B_ptr = (const int4 *)B;
  int4 *C_ptr = (int4 *)C;
  const int4 *s_ptr = (const int4 *)s;

  // int cols = prob_n / thread_n;
  int *locks = (int *)workspace;

  int ret = 0;
  for (int i = 0; i < tot_m_blocks; i += 4) {
    int thread_m_blocks = tot_m_blocks - i;
    prob_m = tot_m - 16 * i;
    int par = 1;
    if (thread_m_blocks > 4) {
      // Note that parallel > 1 currently only works for inputs without any
      // padding
      par = (16 * thread_m_blocks - pad) / 64;
      if (par > max_par)
        par = max_par;
      prob_m = 64 * par;
      i += 4 * (par - 1);
      thread_m_blocks = 4;
    }
    thread_m = thread_m_blocks * 16;

    // For compilation speed, we only define the kernel configurations that
    // have seemed useful (in terms of performance) in our testing, however many
    // more are, in principle, possible.
    // printf("problem %d %d %d\n", prob_m, prob_n, prob_k);
    // printf("thread_m %d thread_n %d thread_k %d group%d parallel %d \n",
    //        thread_m, thread_n, thread_k, groupsize, par);
    if (false) {
    }
    CALL_IF(16, 128, 128, 128)
    CALL_IF(16, 128, 128, -1)
    CALL_IF(16, 256, 64, 128)
    CALL_IF(16, 256, 64, -1)
    CALL_IF(32, 256, 64, 128)
    CALL_IF(32, 256, 64, -1)
    CALL_IF(48, 256, 64, 128)
    CALL_IF(48, 256, 64, -1)
    CALL_IF(64, 256, 64, 128)
    CALL_IF(64, 256, 64, -1)
    else {
      ret = ERR_KERN_SHAPE;
    }
    // cudaDeviceSynchronize();
    // printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;
    C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
  }

  return ret;
}
