#include <cublas_v2.h>


namespace
{

constexpr int WARP_SIZE = 32;

// (a + b - 1) // b
__device__ __host__
constexpr
int cdiv(int a, int b)
{
    return (a + b - 1) / b;
}

__global__
void matmul_v1_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int offset_n = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset_m = blockIdx.y * blockDim.y + threadIdx.y;

    if (offset_m > M || offset_n > N) {
        return;
    }

    float acc = 0.f;
    for (int i = 0; i < K; i++) {
        acc += A[offset_m * K + i] * B[i * N + offset_n];
    }
    C[offset_m * N + offset_n] = acc;
}

template <int CTA_TILE_SIZE>
__global__
void matmul_v2_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int thread_tile_offset_n = threadIdx.x;
    const int thread_tile_offset_m = threadIdx.y;

    const int cta_tile_offset_n = blockIdx.x * CTA_TILE_SIZE;
    const int cta_tile_offset_m = blockIdx.y * CTA_TILE_SIZE;

    __shared__ float A_smem[CTA_TILE_SIZE][CTA_TILE_SIZE];
    __shared__ float B_smem[CTA_TILE_SIZE][CTA_TILE_SIZE];

    A += cta_tile_offset_m * K;
    B += cta_tile_offset_n;
    C += cta_tile_offset_m * N + cta_tile_offset_n;

    float acc = 0.f;
    for (int cta_tile_offset_k = 0; cta_tile_offset_k < K; cta_tile_offset_k += CTA_TILE_SIZE) {
        A_smem[thread_tile_offset_m][thread_tile_offset_n] =
            (cta_tile_offset_k + thread_tile_offset_n < K) &&
            (cta_tile_offset_m + thread_tile_offset_m < M)
                ? A[thread_tile_offset_m * K + thread_tile_offset_n]
                : 0.f;
        B_smem[thread_tile_offset_m][thread_tile_offset_n] =
            (cta_tile_offset_n + thread_tile_offset_n < N) &&
            (cta_tile_offset_k + thread_tile_offset_m < K)
                ? B[thread_tile_offset_m * N + thread_tile_offset_n]
                : 0.f;
        __syncthreads();

        for (int k = 0; k < CTA_TILE_SIZE; k++) {
            acc += A_smem[thread_tile_offset_m][k] * B_smem[k][thread_tile_offset_n];
        }
        __syncthreads();

        A += CTA_TILE_SIZE;
        B += CTA_TILE_SIZE * N;
    }

    if ((cta_tile_offset_n + thread_tile_offset_n < N) && (cta_tile_offset_m + thread_tile_offset_m < M)) {
        C[thread_tile_offset_m * N + thread_tile_offset_n] = acc;
    }
}

template <
    int TB_SIZE,
    int SMEM_TILE_M,
    int SMEM_TILE_N,
    bool TRANSPOSE = false,
    int SMEM_STRIDE_N = TRANSPOSE ? SMEM_TILE_M : SMEM_TILE_N
>
__device__ __forceinline__
void load_block_to_smem(
    const float* src,
    float* smem,
    int src_stride_n,
    int src_max_m, // cta_tile_offset_m + y < M
    int src_max_n
) {
    const int tid = threadIdx.x;
    for (int i = tid; i < SMEM_TILE_M * SMEM_TILE_N; i += TB_SIZE) {
        const int y = i / SMEM_TILE_N;
        const int x = i % SMEM_TILE_N;
        const float data = (y < src_max_m) && (x < src_max_n) ? src[y * src_stride_n + x]: 0.f;
        if constexpr (TRANSPOSE) {
            smem[x * SMEM_STRIDE_N + y] = data;
        } else {
            smem[y * SMEM_STRIDE_N + x] = data;
        }
    }
}

template <int TB_SIZE, int CTA_TILE_M, int CTA_TILE_N, int CTA_TILE_K>
__global__
void matmul_v3_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int cta_tile_offset_m = blockIdx.y * CTA_TILE_M;
    const int cta_tile_offset_n = blockIdx.x * CTA_TILE_N;
    constexpr int cta_tile_size = CTA_TILE_M * CTA_TILE_N;

    A += cta_tile_offset_m * K;
    B += cta_tile_offset_n;
    C += cta_tile_offset_m * N + cta_tile_offset_n;

    __shared__ float A_smem[CTA_TILE_M][CTA_TILE_K];
    __shared__ float B_smem[CTA_TILE_K][CTA_TILE_N];

    static_assert(cta_tile_size % TB_SIZE == 0);
    float acc[cta_tile_size / TB_SIZE] = {0.f};
    for (int cta_tile_offset_k = 0; cta_tile_offset_k < K; cta_tile_offset_k += CTA_TILE_K) {
        load_block_to_smem<TB_SIZE, CTA_TILE_M, CTA_TILE_K>(A, reinterpret_cast<float*>(A_smem), K, M - cta_tile_offset_m, K - cta_tile_offset_k);
        load_block_to_smem<TB_SIZE, CTA_TILE_K, CTA_TILE_N>(B, reinterpret_cast<float*>(B_smem), N, K - cta_tile_offset_k, N - cta_tile_offset_n);
        __syncthreads();

        for (int i = tid, acc_idx = 0; i < cta_tile_size; i += TB_SIZE, acc_idx++) {
            const int cta_elem_m = i / CTA_TILE_N;
            const int cta_elem_n = i % CTA_TILE_N;
            for (int k = 0; k < CTA_TILE_K; k++) {
                acc[acc_idx] += A_smem[cta_elem_m][k] * B_smem[k][cta_elem_n];
            }
        }
        __syncthreads();

        A += CTA_TILE_K;
        B += CTA_TILE_K * N;
    }

    for (int i = tid, acc_idx = 0; i < cta_tile_size; i += TB_SIZE, acc_idx++) {
        const int cta_elem_m = i / CTA_TILE_N;
        const int cta_elem_n = i % CTA_TILE_N;
        if ((cta_elem_m + cta_tile_offset_m < M) && (cta_elem_n + cta_tile_offset_n < N)) {
            C[cta_elem_m * N + cta_elem_n] = acc[acc_idx];
        }
    }
}

template <int TB_SIZE, int CTA_TILE_M, int CTA_TILE_N, int CTA_TILE_K, int THREAD_TILE_M, int THREAD_TILE_N>
__global__
void matmul_v4_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int tid = threadIdx.x;

    const int cta_tile_offset_m = blockIdx.y * CTA_TILE_M;
    const int cta_tile_offset_n = blockIdx.x * CTA_TILE_N;

    const int thread_tile_offset_m = (tid / (CTA_TILE_N / THREAD_TILE_N)) * THREAD_TILE_M;
    const int thread_tile_offset_n = (tid % (CTA_TILE_N / THREAD_TILE_N)) * THREAD_TILE_N;
    const int offset_m = cta_tile_offset_m + thread_tile_offset_m;
    const int offset_n = cta_tile_offset_n + thread_tile_offset_n;

    A += cta_tile_offset_m * K;
    B += cta_tile_offset_n;
    C += offset_m * N + offset_n;

    __shared__ float A_smem[CTA_TILE_M][CTA_TILE_K];
    __shared__ float B_smem[CTA_TILE_K][CTA_TILE_N];

    float acc[THREAD_TILE_M][THREAD_TILE_N] = {0.f};
    for (int cta_tile_offset_k = 0; cta_tile_offset_k < K; cta_tile_offset_k += CTA_TILE_K) {
        load_block_to_smem<TB_SIZE, CTA_TILE_M, CTA_TILE_K>(A, reinterpret_cast<float*>(A_smem), K, M - cta_tile_offset_m, K - cta_tile_offset_k);
        load_block_to_smem<TB_SIZE, CTA_TILE_K, CTA_TILE_N>(B, reinterpret_cast<float*>(B_smem), N, K - cta_tile_offset_k, N - cta_tile_offset_n);
        __syncthreads();

        for (int k = 0; k < CTA_TILE_K; k++) {
            float A_reg[THREAD_TILE_M];
            float B_reg[THREAD_TILE_N];

            for (int i = 0; i < THREAD_TILE_M; i++) {
                A_reg[i] = A_smem[thread_tile_offset_m + i][k];
            }

            for (int i = 0; i < THREAD_TILE_N; i++) {
                B_reg[i] = B_smem[k][thread_tile_offset_n + i];
            }

            for (int y = 0; y < THREAD_TILE_M; y++) {
                for (int x = 0; x < THREAD_TILE_N; x++) {
                    acc[y][x] += A_reg[y] * B_reg[x];
                }
            }
        } 
        __syncthreads();

        A += CTA_TILE_K;
        B += CTA_TILE_K * N;
    }

    for (int y = 0; y < THREAD_TILE_M; y++) {
        for (int x = 0; x < THREAD_TILE_N; x++) {
            if ((offset_m + y < M) && (offset_n + x < N)) {
                C[y * N + x] = acc[y][x];
            }
        }
    }

}

template <int TB_SIZE, int CTA_TILE_M, int CTA_TILE_N, int CTA_TILE_K, int THREAD_TILE_M, int THREAD_TILE_N, int WARP_M, int WARP_N>
__global__
void matmul_v5_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int tid = threadIdx.x;

    // const int tile_offset_m = blockIdx.y * CTA_TILE_M;
    // const int tile_offset_n = blockIdx.x * CTA_TILE_N;

    const int cta_tile_offset_m = blockIdx.y * CTA_TILE_M;
    const int cta_tile_offset_n = blockIdx.x * CTA_TILE_N; 

    // const int warp_idx = tid / warp_size;
    // const int warp_x = warp_idx % (CTA_TILE_N / THREAD_TILE_N / WARP_N);
    // const int warp_y = warp_idx / (CTA_TILE_N / THREAD_TILE_N / WARP_N);
    // const int warp_offset_m = warp_y * WARP_M * THREAD_TILE_M;
    // const int warp_offset_n = warp_x * WARP_N * THREAD_TILE_N;

    constexpr int WARP_TILE_M = WARP_M * THREAD_TILE_M;
    constexpr int WARP_TILE_N = WARP_N * THREAD_TILE_N;

    const int warp_id = tid / WARP_SIZE;
    const int warp_tiles_n = CTA_TILE_N / WARP_TILE_N;
    const int warp_y = warp_id / warp_tiles_n;
    const int warp_x = warp_id % warp_tiles_n;

    const int warp_tile_offset_m = warp_y * WARP_TILE_M;
    const int warp_tile_offset_n = warp_x * WARP_TILE_N;

    // const int tid_in_warp = tid % warp_size;
    // const int thrd_offset_m_in_warp = (tid_in_warp / WARP_N) * THREAD_TILE_M;
    // const int thrd_offset_n_in_warp = (tid_in_warp % WARP_N) * THREAD_TILE_N;
    // const int offset_m = tile_offset_m + warp_offset_m + thrd_offset_m_in_warp;
    // const int offset_n = tile_offset_n + warp_offset_n + thrd_offset_n_in_warp;

    const int lane_id = tid % WARP_SIZE;
    const int thread_tile_offset_m = (lane_id / WARP_N) * THREAD_TILE_M;
    const int thread_tile_offset_n = (lane_id % WARP_N) * THREAD_TILE_N;

    const int offset_m = cta_tile_offset_m + warp_tile_offset_m + thread_tile_offset_m;
    const int offset_n = cta_tile_offset_n + warp_tile_offset_n + thread_tile_offset_n;

    A += cta_tile_offset_m * K;
    B += cta_tile_offset_n;
    C += offset_m * N + offset_n;

    constexpr int A_SMEM_STRIDE_M = CTA_TILE_M + 1;

    __shared__ float A_smem[CTA_TILE_K][A_SMEM_STRIDE_M]; // transposed, skewed to avoid bank conflict
    __shared__ float B_smem[CTA_TILE_K][CTA_TILE_N];

    float acc[THREAD_TILE_M][THREAD_TILE_N] = {0.f};
    for (int cta_tile_offset_k = 0; cta_tile_offset_k < K; cta_tile_offset_k += CTA_TILE_K) {
        load_block_to_smem<TB_SIZE, CTA_TILE_M, CTA_TILE_K, true, A_SMEM_STRIDE_M>(A, reinterpret_cast<float*>(A_smem), K, M - cta_tile_offset_m, K - cta_tile_offset_k);
        load_block_to_smem<TB_SIZE, CTA_TILE_K, CTA_TILE_N>(B, reinterpret_cast<float*>(B_smem), N, K - cta_tile_offset_k, N - cta_tile_offset_n);
        __syncthreads();

        for (int k = 0; k < CTA_TILE_K; k++) {
            static_assert(THREAD_TILE_N % 4 == 0);
            float A_reg[THREAD_TILE_M];
            float4 B_reg[THREAD_TILE_N / 4];

            // 2-way bank conflict
            for (int i = 0; i < THREAD_TILE_M; i++) {
                A_reg[i] = A_smem[k][warp_tile_offset_m + thread_tile_offset_m + i];
            }

            // conflict free
            for (int i = 0; i < THREAD_TILE_N / 4; i++) {
                B_reg[i] = *reinterpret_cast<float4*>(&B_smem[k][warp_tile_offset_n + thread_tile_offset_n + 4 * i]);
            }

            const float* B_reg_p = reinterpret_cast<float*>(B_reg);
            for (int y = 0; y < THREAD_TILE_M; y++) {
                for (int x = 0; x < THREAD_TILE_N; x++) {
                    acc[y][x] += A_reg[y] * B_reg_p[x];
                }
            }
        }
        __syncthreads();

        A += CTA_TILE_K;
        B += CTA_TILE_K * N;
    }

    for (int y = 0; y < THREAD_TILE_M; y++) {
        for (int x = 0; x < THREAD_TILE_N; x++) {
            if ((offset_m + y < M) && (offset_n + x < N)) {
                C[y * N + x] = acc[y][x];
            }
        }
    }
}

}

void matmul_cublas_gemm(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    constexpr float alpha = 1.f;
    constexpr float beta  = 0.f;
    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

void matmul_v1(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    int cta_threads;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &cta_threads, matmul_v1_kernel, 0, 0);

    dim3 cta_shape(WARP_SIZE, cta_threads / WARP_SIZE);
    dim3 grid_size(cdiv(N, WARP_SIZE), cdiv(M, cta_shape.y));
    matmul_v1_kernel<<<grid_size, cta_shape>>>(A, B, C, M, N, K);
}

void matmul_v2(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    constexpr int CTA_TILE_SIZE = 32;
    dim3 cta_shape(CTA_TILE_SIZE, CTA_TILE_SIZE);
    dim3 grid_size(cdiv(N, CTA_TILE_SIZE), cdiv(M, CTA_TILE_SIZE));
    matmul_v2_kernel<CTA_TILE_SIZE><<<grid_size, cta_shape>>>(A, B, C, M, N, K);
}

void matmul_v3(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    constexpr int CTA_TILE_M = 128;
    constexpr int CTA_TILE_N = 128;
    constexpr int CTA_TILE_K = 32;
    constexpr int TB_SIZE = 256;
    dim3 grid_size(cdiv(N, CTA_TILE_N), cdiv(M, CTA_TILE_M));
    matmul_v3_kernel<TB_SIZE, CTA_TILE_M, CTA_TILE_N, CTA_TILE_K>
        <<<grid_size, TB_SIZE>>>(A, B, C, M, N, K);
}

void matmul_v4(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    constexpr int CTA_TILE_M = 128;
    constexpr int CTA_TILE_N = 128;
    constexpr int CTA_TILE_K = 32;
    constexpr int THREAD_TILE_M = 8;
    constexpr int THREAD_TILE_N = 8;
    constexpr int TB_SIZE  = (CTA_TILE_M * CTA_TILE_N) / (THREAD_TILE_M * THREAD_TILE_N);

    static_assert((CTA_TILE_M * CTA_TILE_N) % (THREAD_TILE_M * THREAD_TILE_N) == 0);
    static_assert((CTA_TILE_M % THREAD_TILE_M == 0) && (CTA_TILE_N % THREAD_TILE_N == 0));

    dim3 grid_size(cdiv(N, CTA_TILE_N), cdiv(M, CTA_TILE_M));
    matmul_v4_kernel<TB_SIZE, CTA_TILE_M, CTA_TILE_N, CTA_TILE_K, THREAD_TILE_M, THREAD_TILE_N>
        <<<grid_size, TB_SIZE>>>(A, B, C, M, N, K);
}

void matmul_v5(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    constexpr int CTA_TILE_M = 128;
    constexpr int CTA_TILE_N = 128;
    constexpr int CTA_TILE_K = 32;

    constexpr int THREAD_TILE_M = 8;
    constexpr int THREAD_TILE_N = 8;

    constexpr int WARP_M = 8;
    constexpr int WARP_N = 32 / WARP_M;

    constexpr int TB_SIZE  = (CTA_TILE_M * CTA_TILE_N) / (THREAD_TILE_M * THREAD_TILE_N);

    static_assert((CTA_TILE_M * CTA_TILE_N) % (THREAD_TILE_M * THREAD_TILE_N) == 0);
    static_assert((CTA_TILE_M % THREAD_TILE_M == 0) && (CTA_TILE_N % THREAD_TILE_N == 0));
    static_assert(((CTA_TILE_M / THREAD_TILE_M) % WARP_M == 0) && ((CTA_TILE_N / THREAD_TILE_N) % WARP_N == 0));
    static_assert(WARP_M * WARP_N == 32);

    dim3 grid_size(cdiv(N, CTA_TILE_N), cdiv(M, CTA_TILE_M));
    matmul_v5_kernel<TB_SIZE, CTA_TILE_M, CTA_TILE_N, CTA_TILE_K, THREAD_TILE_M, THREAD_TILE_N, WARP_M, WARP_N>
        <<<grid_size, TB_SIZE>>>(A, B, C, M, N, K);
}
