"""Raw compute benchmark — FP16 matmul TFLOPS on a single GPU.

This is the cleanest cross-GPU number: it isolates raw tensor throughput, independent of
batch size, model, or CPU/IO. Use it to sanity-check the model benchmarks (a big datacenter
GPU should dominate here even when small-batch latency tests are misleading).
"""
import time
import torch
from utils import update_results

dev = "cuda:0"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

N = 8192
a = torch.randn(N, N, device=dev, dtype=torch.float16)
b = torch.randn(N, N, device=dev, dtype=torch.float16)

for _ in range(5):          # warm-up
    c = a @ b
torch.cuda.synchronize()

iters = 100
start = time.time()
for _ in range(iters):
    c = a @ b
torch.cuda.synchronize()
dur = time.time() - start

tflops = (2 * (N ** 3) * iters) / dur / 1e12
update_results("Compute", {
    "Matmul_FP16_TFLOPS": round(tflops, 1),
    "Matrix_N": N,
    "Iters": iters,
})
