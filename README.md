# Faster and Faster Kernels
## Benchmark Info
- GPU: RTX 4060 Laptop
- CUDA Toolkit Version: 12.6
- Driver Version: 595.71
- PyTorch Version: 2.11.0+cu126
- M = N = K = 4096
- Metric: median latency from `triton.testing.do_bench`

## FP32 Matmul

Kernel name                                             | Duration (ms) | % of torch.matmul | Throughput (TFLOP/s)
--------------------------------------------------------|---------------|-------------------|---------------------
`torch.matmul` (PyTorch baseline)                       |         15.61 |           100.00% |                 8.80
v1 (naive one-thread-per-output)                        |        154.88 |            10.08% |                 0.89
v2 (shared-memory CTA tiling)                           |        132.77 |            11.76% |                 1.04
v3 (thread coarsening)                                  |        107.74 |            14.49% |                 1.28
v4 (thread tiling with register blocking)               |         24.90 |            62.70% |                 5.52
v5 (warp tiling, transposed/skewed SMEM, vectorized B)  |         21.20 |            73.65% |                 6.48

## BF16 Matmul

Kernel name                                             | Duration (ms) | % of torch.matmul | Throughput (TFLOP/s)
--------------------------------------------------------|---------------|-------------------|---------------------
`torch.matmul` (PyTorch baseline)                       |          6.10 |           100.00% |                22.53
v1 (Tensor Core MMA, ldmatrix, swizzled shared memory)  |          5.57 |           109.48% |                24.67
