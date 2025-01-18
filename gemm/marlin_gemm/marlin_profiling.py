import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import marlin
import numpy as np
import pandas as pd

seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)
DEV = torch.device('cuda')

perf_data = []
def gen_quant4(m, n, groupsize=-1):
    tile = 16
    maxq = 2 ** 4 - 1
    w = torch.randn((m, n), dtype=torch.half, device=DEV)
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()
    # Workaround to test some special cases that are forbidden by the API
    layer = marlin.Layer(256, 256, groupsize=groupsize, cute_version=False)
    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t())
    q = layer.B
    s = layer.s
    return ref, q, s


def run_profiling(m, n, k, cta_k=-1, cta_n=-1, group_size=-1, iter=1000, model=""):
      print(f"start profiling m:{m} n:{n} k:{k} group_size:{group_size}")
      A = torch.randn((m, k), dtype=torch.half, device=DEV)
      B_ref, B, s = gen_quant4(k, n, groupsize=group_size)
      C = torch.zeros((m, n), dtype=torch.half, device=DEV)
      workspace = torch.zeros(n // 128 * 16, device=DEV) 
      marlin.mul(A, B, C, s, workspace, cta_k, cta_n, -1, cute_version=True)
      with profile(
          activities=[ProfilerActivity.CUDA], 
          record_shapes=True,  # Record input shapes
          profile_memory=True,  # Profile memory usage
          with_stack=True  # Include stack traces
      ) as prof:
          with record_function("model_inference"):
              for _ in range(iter):  # Run a few iterations
                  marlin.mul(A, B, C, s, workspace, cta_k, cta_n, -1, cute_version=True)
                  marlin.mul(A, B, C, s, workspace, cta_k, cta_n, -1, cute_version=False)  
      
      # print(prof.key_averages().table(sort_by="cuda_time_total"))
      result = prof.key_averages()
      print(result.table(sort_by="cuda_time_total"))

      perf = {
        "shape(m_n_k_group)":f"{model}_{m}_{n}_{k}_{group_size}",
      }
      for event in result:
          if "MarlinGemmTraits" in event.key:
              perf["marlin_cute"] = f"{event.device_time:.3f} us"
          elif "Marlin" in event.key:
              perf["marlin_official"] = f"{event.device_time:.3f} us"

      perf_data.append(perf)

if __name__ == "__main__":
    MODELS = {
      ' 7B': [
          (4096, 3 * 4096),
          (4096, 4096),
          (4096, 2 * 10752),
          (10752, 4096)
      ],
      '13B': [
          (5120, 3 * 5120),
          (5120, 5120),
          (5120, 2 * 13568),
          (13568, 5120)
      ],
      '33B': [
          (6656, 3 * 6656),
          (6656, 6656),
          (6656, 2 * 17664), # official fail with 0.001
          (17664, 6656)
      ],
      '70B': [
          (8192, 3 * 8192),
          (8192, 8192),
          (8192, 2 * 21760), # official fail with 0.001
          (21760, 8192)
      ]
    }
    for model, layers in MODELS.items():
        print(model)
        for layer in layers:
            for group_size in [-1, 128]:
                for batch in [1, 16]:
                    run_profiling(batch, layer[1], layer[0], group_size=group_size, model=model)
    
    pd.DataFrame(perf_data).to_csv("marlin_perf.csv",index=False)