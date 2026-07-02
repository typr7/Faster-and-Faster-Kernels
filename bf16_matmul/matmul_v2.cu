#include <cstdint>

#include <cuda_bf16.h>

#include "common.hpp"


namespace
{

template <uint32_t TB_SIZE, uint32_t TILE_M, uint32_t TILE_N>
__device__ __forceinline__
void load_tile_to_smem(
    const nv_bfloat16* __restrict__ src,
    nv_bfloat16* __restrict__ smem,
    uint32_t src_stride
) {
    constexpr uint32_t TILE_N_U4 = TILE_N / BF16_NUM_PER_U4;
    const uint32_t src_stride_u4 = src_stride / BF16_NUM_PER_U4;

    const auto* __restrict__ src_u4 = reinterpret_cast<const uint4*>(&src[0]);
    auto* __restrict__ smem_u4 = reinterpret_cast<uint4*>(&smem[0]);

    for (uint32_t i = threadIdx.x; i < TILE_M * TILE_N_U4; i += TB_SIZE) {
        const uint32_t y = i / TILE_N_U4;
        const uint32_t x = i % TILE_N_U4;
        smem_u4[y * TILE_N_U4 + x] = src_u4[y * src_stride_u4 + x];
    }
}

template <
    uint32_t TB_SIZE,
    uint32_t CTA_TILE_M, uint32_t CTA_TILE_N, uint32_t CTA_TILE_K,
    uint32_t WARP_TILE_M, uint32_t WARP_TILE_N
> __launch_bounds__(TB_SIZE) __global__
void matmul_kernel(
    const nv_bfloat16* __restrict__ A,
    const nv_bfloat16* __restrict__ B,
    nv_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane_id = tid % WARP_SIZE;
    const uint32_t warp_id = tid / WARP_SIZE;

    const uint32_t cta_tile_offset_m = blockIdx.y * CTA_TILE_M;
    const uint32_t cta_tile_offset_n = blockIdx.x * CTA_TILE_N;

    constexpr uint32_t WARP_TILES_N = CTA_TILE_N / WARP_TILE_N;

    const uint32_t warp_tile_y = warp_id / WARP_TILES_N;
    const uint32_t warp_tile_x = warp_id % WARP_TILES_N;

    const uint32_t warp_tile_offset_m = warp_tile_y * WARP_TILE_M;
    const uint32_t warp_tile_offset_n = warp_tile_x * WARP_TILE_N;

    const uint32_t offset_m = cta_tile_offset_m + warp_tile_offset_m;
    const uint32_t offset_n = cta_tile_offset_n + warp_tile_offset_n;

    constexpr uint32_t MMA_TILES_M = WARP_TILE_M / MMA_M;
    constexpr uint32_t MMA_TILES_N = WARP_TILE_N / MMA_N;

    constexpr uint32_t ACC_REGS_PER_THREAD = MMA_M * MMA_N / WARP_SIZE;
    constexpr uint32_t A_REGS_PER_THREAD
        = MMA_M * MMA_K * sizeof(nv_bfloat16) / WARP_SIZE / sizeof(uint32_t);
    constexpr uint32_t B_REGS_PER_THREAD
        = MMA_K * MMA_N * sizeof(nv_bfloat16) / WARP_SIZE / sizeof(uint32_t);

    A += cta_tile_offset_m * K;
    B += cta_tile_offset_n * K;
    C += offset_m * N + offset_n;

    __shared__ nv_bfloat16 A_smem[CTA_TILE_M][CTA_TILE_K];
    __shared__ nv_bfloat16 B_smem[CTA_TILE_N][CTA_TILE_K];

    float acc_reg[MMA_TILES_M][MMA_TILES_N][ACC_REGS_PER_THREAD] = {0.f};
    for (uint32_t cta_tile_offset_k = 0; cta_tile_offset_k < K; cta_tile_offset_k += CTA_TILE_K) {
        load_tile_to_smem<TB_SIZE, CTA_TILE_M, CTA_TILE_K>(A, &A_smem[0][0], K);
        load_tile_to_smem<TB_SIZE, CTA_TILE_N, CTA_TILE_K>(B, &B_smem[0][0], K);
        __syncthreads();

        for (uint32_t k = 0; k < CTA_TILE_K; k += MMA_K) {
            uint32_t B_reg[MMA_TILES_N][B_REGS_PER_THREAD];

            // (16x8)
            #pragma unroll
            for (uint32_t n = 0; n < MMA_TILES_N; n++) {
                const uint32_t ldmatrix_lane = lane_id % 16;
                const uint32_t smem_n = warp_tile_offset_n + n * MMA_N + ldmatrix_lane % 8;
                const uint32_t smem_k = k + (ldmatrix_lane / 8) * 8;
                ldmatrix_x2(B_reg[n], cvta_shared(&B_smem[smem_n][smem_k]));
            }

            // (16x16)
            #pragma unroll
            for (uint32_t m = 0; m < MMA_TILES_M; m++) {
                uint32_t A_reg[A_REGS_PER_THREAD];

                const uint32_t row = lane_id % 8;
                const uint32_t matrix_id = lane_id / 8;
                const uint32_t smem_m = warp_tile_offset_m + m * MMA_M + (matrix_id & 0b1) * 8 + row;
                const uint32_t smem_k = k + (lane_id / 16) * 8;
                ldmatrix_x4(A_reg, cvta_shared(&A_smem[smem_m][smem_k]));

                #pragma unroll
                for (uint32_t n = 0; n < MMA_TILES_N; n++) {
                    mma_m16n8k16(A_reg, B_reg[n], acc_reg[m][n]);
                }
            }
        }
        __syncthreads();

        A += CTA_TILE_K;
        B += CTA_TILE_K;
    }

    #pragma unroll
    for (uint32_t m = 0; m < MMA_TILES_M; m++) {
        #pragma unroll
        for (uint32_t n = 0; n < MMA_TILES_N; n++) {
            const uint32_t y = m * MMA_M + lane_id / 4;
            const uint32_t x = n * MMA_N + (lane_id % 4) * 2;

            const float* reg = acc_reg[m][n];
            reinterpret_cast<nv_bfloat162*>(&C[y * N + x])[0] =
                __float22bfloat162_rn(make_float2(reg[0], reg[1]));
            reinterpret_cast<nv_bfloat162*>(&C[(y + 8) * N + x])[0] =
                __float22bfloat162_rn(make_float2(reg[2], reg[3]));
        }
    }
}

}

// no bound check
void matmul_v2(
    const nv_bfloat16* A,
    const nv_bfloat16* B,
    nv_bfloat16* C,
    int M, int N, int K
) {
    constexpr uint32_t CTA_TILE_M = 128;
    constexpr uint32_t CTA_TILE_N = 128;
    constexpr uint32_t CTA_TILE_K = 64;

    constexpr uint32_t WARP_TILE_M = 64;
    constexpr uint32_t WARP_TILE_N = 64;

    constexpr uint32_t WARP_TILES_M = CTA_TILE_M / WARP_TILE_M;
    constexpr uint32_t WARP_TILES_N = CTA_TILE_N / WARP_TILE_N;

    constexpr uint32_t TB_SIZE = WARP_TILES_M * WARP_TILES_N * WARP_SIZE;

    const dim3 grid_size(N / CTA_TILE_N, M / CTA_TILE_M);

    matmul_kernel<
        TB_SIZE,
        CTA_TILE_M, CTA_TILE_N, CTA_TILE_K,
        WARP_TILE_M, WARP_TILE_N
    ><<<grid_size, TB_SIZE>>>(A, B, C, M, N, K);
}
