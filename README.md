# Faster and Faster Kernels
## Benchmark Info
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- NVIDIA Driver Version: 595.71
- CUDA Toolkit: 13.0
- PyTorch Version: 2.11.0+cu130
- M = N = K = 4096
- TFLOPS metric: `2 * M * N * K / duration`
- `%SOL` metric: `TFLOPS / theoretical_peak_TFLOPS * 100`
- Peak basis: 24 SMs, CUDA runtime device clock 2010 MHz; FP32 peak 12.35 TFLOPS, BF16 Tensor Core dense peak 49.40 TFLOPS

## FP32 Matmul

Kernel name                                             | Duration (ms) | % of torch.matmul | TFLOPS (%SOL)
--------------------------------------------------------|---------------|-------------------|---------------
`torch.matmul` (PyTorch baseline)                       |         16.27 |           100.00% |   8.45 (68.38%)
v1 (naive one-thread-per-output)                        |        162.15 |            10.04% |    0.85 (6.86%)
v2 (shared-memory CTA tiling)                           |        137.55 |            11.83% |    1.00 (8.09%)
v3 (thread coarsening)                                  |        108.00 |            15.07% |   1.27 (10.30%)
v4 (thread tiling with register blocking)               |         26.24 |            62.03% |   5.24 (42.42%)
v5 (warp tiling, transposed/skewed SMEM, vectorized B)  |         21.83 |            74.54% |   6.29 (50.97%)

## BF16 Matmul

Kernel name                                             | Duration (ms) | % of torch.matmul | TFLOPS (%SOL)
--------------------------------------------------------|---------------|-------------------|---------------
`torch.matmul` (PyTorch baseline)                       |          6.12 |           100.00% |  22.47 (45.50%)
v1 (tensor core MMA, hierarchical tiling)               |         13.14 |            46.54% |  10.46 (21.17%)
v2 (vectorized memory copy)                             |          7.30 |            83.81% |  18.83 (38.13%)
v3 (swizzled shared memory)                             |          6.30 |            97.03% |  21.81 (44.14%)
v4 (flat shared-memory addressing)                      |          5.30 |           115.31% |  25.92 (52.46%)
