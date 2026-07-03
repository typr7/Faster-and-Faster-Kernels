# Faster and Faster Kernels

Fixed M = N = K = 4096. Compile with CUDA Toolkit 13.0 and report `duration; % of torch.matmul; TFLOPS (%SOL)`.

- TFLOPS metric: `2 * M * N * K / duration`
- `%SOL` metric: `TFLOPS / theoretical_peak_TFLOPS * 100`
- BF16 inputs use A row-major x B column-major. FP32 inputs use A row-major x B row-major.

GPU                           | Driver    | PyTorch        | Peak basis
------------------------------|-----------|----------------|-----------
NVIDIA GeForce RTX 5090       | 595.58.03 | 2.12.1+cu130   | 170 SMs, CUDA runtime device clock 2407 MHz; FP32 peak 104.75 TFLOPS, BF16 Tensor Core dense peak 419.01 TFLOPS
NVIDIA A100-PCIE-40GB         | 595.71.05 | 2.12.1+cu130   | 108 SMs, CUDA runtime device clock 1410 MHz; FP32 peak 19.49 TFLOPS, BF16 Tensor Core dense peak 311.87 TFLOPS

## FP32 Matmul

Kernel name                                             | RTX 5090                         | A100-PCIE-40GB
--------------------------------------------------------|----------------------------------|-------------------------------
`torch.matmul` (PyTorch baseline)                       | 2.05 ms; 100.00%; 67.08 (64.03%) | 7.72 ms; 100.00%; 17.79 (91.29%)
v1 (naive one-thread-per-output)                        | 18.28 ms; 11.21%; 7.52 (7.18%)   | 45.79 ms; 16.87%; 3.00 (15.40%)
v2 (shared-memory CTA tiling)                           | 14.66 ms; 13.98%; 9.38 (8.95%)   | 26.73 ms; 28.89%; 5.14 (26.37%)
v3 (thread coarsening)                                  | 6.52 ms; 31.45%; 21.09 (20.14%)  | 27.35 ms; 28.24%; 5.02 (25.78%)
v4 (thread tiling with register blocking)               | 3.55 ms; 57.71%; 38.71 (36.95%)  | 10.96 ms; 70.48%; 12.54 (64.34%)
v5 (warp tiling, transposed/skewed SMEM, vectorized B)  | 3.52 ms; 58.24%; 39.06 (37.29%)  | 10.73 ms; 71.97%; 12.81 (65.70%)

## BF16 Matmul

Kernel name                                             | RTX 5090                          | A100-PCIE-40GB
--------------------------------------------------------|-----------------------------------|--------------------------------
`torch.matmul` (PyTorch baseline)                       | 0.63 ms; 100.00%; 219.67 (52.43%) | 0.64 ms; 100.00%; 215.78 (69.19%)
v1 (tensor core MMA, hierarchical tiling)               | 1.89 ms; 33.12%; 72.75 (17.36%)   | 4.71 ms; 13.52%; 29.17 (9.35%)
v2 (vectorized memory copy)                             | 0.96 ms; 65.49%; 143.86 (34.33%)  | 2.84 ms; 22.46%; 48.47 (15.54%)
v3 (swizzled shared memory)                             | 0.78 ms; 80.45%; 176.72 (42.18%)  | 1.77 ms; 35.91%; 77.49 (24.85%)
v4 (flat shared-memory addressing)                      | 0.78 ms; 80.24%; 176.26 (42.07%)  | 1.13 ms; 56.49%; 121.91 (39.09%)
v4 tuned                                                | 0.72 ms; 86.54%; 190.10 (45.37%)<br>CTA 64x128x64, warp 64x32 | 1.03 ms; 61.95%; 133.68 (42.87%)<br>CTA 128x128x64, warp 64x32
v5 (double-buffered async copy pipeline)                | 0.78 ms; 79.81%; 175.32 (41.84%)  | 0.83 ms; 76.41%; 164.89 (52.87%)
v5 tuned                                                | 0.68 ms; 92.51%; 203.22 (48.50%)<br>CTA 64x128x64, warp 32x64 | 0.83 ms; 76.60%; 165.29 (53.00%)<br>CTA 128x128x64, warp 128x32
