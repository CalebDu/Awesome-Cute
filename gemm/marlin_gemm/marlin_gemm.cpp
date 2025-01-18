#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/python.h>

int marlin_cute(const void *A, const void *B, void *C, void *s, int prob_m,
                int prob_n, int prob_k, void *workspace, int groupsize = -1,
                int dev = 0, cudaStream_t stream = 0, int thread_k = -1,
                int thread_n = -1, int sms = -1, int max_par = 16);

int marlin_official(const void *A, const void *B, void *C, void *s, int prob_m,
                    int prob_n, int prob_k, void *workspace, int groupsize = -1,
                    int dev = 0, cudaStream_t stream = 0, int thread_k = -1,
                    int thread_n = -1, int sms = -1, int max_par = 16);

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

void mul(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C,
         const torch::Tensor &s, torch::Tensor &workspace, int thread_k = -1,
         int thread_n = -1, int sms = -1, int max_par = 8,
         bool cute_version = true) {
  int prob_m = A.size(0);
  int prob_n = C.size(1);
  int prob_k = A.size(1);
  int groupsize = (s.size(0) == 1) ? -1 : prob_k / s.size(0);
  if (groupsize != -1 && groupsize * s.size(0) != prob_k)
    AT_ERROR("k=", prob_k, " not compatible with ", s.size(0), " groups.");
  if (workspace.numel() < prob_n / 128 * max_par)
    AT_ERROR("workspace must be of size at least ", prob_n / 128 * max_par,
             ".");
  int dev = A.get_device();
  int err;
  if (cute_version) {
    // printf("cute version\n");
    err = marlin_cute(A.data_ptr(), B.data_ptr(), C.data_ptr(), s.data_ptr(),
                      prob_m, prob_n, prob_k, workspace.data_ptr(), groupsize,
                      dev, at::cuda::getCurrentCUDAStream(dev), thread_k,
                      thread_n, sms, max_par);
  } else {
    // printf("official version\n");
    err = marlin_official(
        A.data_ptr(), B.data_ptr(), C.data_ptr(), s.data_ptr(), prob_m, prob_n,
        prob_k, workspace.data_ptr(), groupsize, dev,
        at::cuda::getCurrentCUDAStream(dev), thread_k, thread_n, sms, max_par);
  }

  if (err == ERR_PROB_SHAPE) {
    AT_ERROR("Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
             " not compatible with thread_k=", thread_k,
             ", thread_n=", thread_n, ".");
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR("No kernel implementation for thread_k=", thread_k,
             ", thread_n=", thread_n, ", groupsize=", groupsize, ".");
  }
}

PYBIND11_MODULE(marlin_lib, m) {
  m.def("mul", &mul, "Marlin FP16xINT4 matmul.");
}