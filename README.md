# Faster and Faster Kernels
## Benchmark Info
- GPU: NVIDIA GeForce RTX 5090
- NVIDIA Driver Version: 595.58.03
- CUDA Toolkit: 13.0
- PyTorch Version: 2.12.1+cu130
- M = N = K = 4096
- TFLOPS metric: `2 * M * N * K / duration`
- `%SOL` metric: `TFLOPS / theoretical_peak_TFLOPS * 100`
- Peak basis: 170 SMs, CUDA runtime device clock 2407 MHz; FP32 peak 104.75 TFLOPS, BF16 Tensor Core dense peak 419.01 TFLOPS

## FP32 Matmul

Kernel name                                             | Duration (ms) | % of torch.matmul | TFLOPS (%SOL)
--------------------------------------------------------|---------------|-------------------|---------------
`torch.matmul` (PyTorch baseline)                       |          2.05 |           100.00% |  67.08 (64.03%)
v1 (naive one-thread-per-output)                        |         18.28 |            11.21% |    7.52 (7.18%)
v2 (shared-memory CTA tiling)                           |         14.66 |            13.98% |    9.38 (8.95%)
v3 (thread coarsening)                                  |          6.52 |            31.45% |  21.09 (20.14%)
v4 (thread tiling with register blocking)               |          3.55 |            57.71% |  38.71 (36.95%)
v5 (warp tiling, transposed/skewed SMEM, vectorized B)  |          3.52 |            58.24% |  39.06 (37.29%)

## BF16 Matmul

Kernel name                                             | Duration (ms) | % of torch.matmul | TFLOPS (%SOL)
--------------------------------------------------------|---------------|-------------------|---------------
`torch.matmul` (PyTorch baseline)                       |          0.63 |           100.00% | 219.67 (52.43%)
v1 (tensor core MMA, hierarchical tiling)               |          1.89 |            33.12% |  72.75 (17.36%)
v2 (vectorized memory copy)                             |          0.96 |            65.49% | 143.86 (34.33%)
v3 (swizzled shared memory)                             |          0.78 |            80.45% | 176.72 (42.18%)
v4 (flat shared-memory addressing)                      |          0.78 |            80.24% | 176.26 (42.07%)
v4 tuned (CTA 64x128x64, warp 64x32)                    |          0.72 |            86.54% | 190.10 (45.37%)
v5 (double-buffered async copy pipeline)                |          0.78 |            79.81% | 175.32 (41.84%)
v5 tuned (CTA 64x128x64, warp 32x64)                    |          0.68 |            92.51% | 203.22 (48.50%)
