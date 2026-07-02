#include <cstdint>
#include <random>

#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>


using MatmulFn = void(
    const nv_bfloat16* A,
    const nv_bfloat16* B,
    nv_bfloat16* C,
    int M, int N, int K
);

#ifndef PROFILE_MATMUL_SYMBOL
#error "PROFILE_MATMUL_SYMBOL must name the matmul function to profile"
#endif

MatmulFn PROFILE_MATMUL_SYMBOL;

struct ProfileConfig {
    int M = 4096;
    int N = 4096;
    int K = 4096;
    int warmup_iters = 1;
    int profile_iters = 1;
};

void randn(nv_bfloat16* x, int64_t count)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.f, 1.f);

    for (int64_t i = 0; i < count; ++i) {
        x[i] = __float2bfloat16(dist(gen));
    }
}

void launch(MatmulFn kernel, nv_bfloat16* A, nv_bfloat16* B, nv_bfloat16* C, const ProfileConfig& config)
{
    kernel(A, B, C, config.M, config.N, config.K);
    cudaDeviceSynchronize();
}

void profile_kernel(MatmulFn matmul, const ProfileConfig& config)
{
    const int64_t elems_a = static_cast<int64_t>(config.M) * config.K;
    const int64_t elems_b = static_cast<int64_t>(config.N) * config.K;
    const int64_t elems_c = static_cast<int64_t>(config.M) * config.N;
    const size_t bytes_a = elems_a * sizeof(nv_bfloat16);
    const size_t bytes_b = elems_b * sizeof(nv_bfloat16);
    const size_t bytes_c = elems_c * sizeof(nv_bfloat16);

    nv_bfloat16* A_h = nullptr;
    nv_bfloat16* B_h = nullptr;
    nv_bfloat16* A = nullptr;
    nv_bfloat16* B = nullptr;
    nv_bfloat16* C = nullptr;

    cudaMallocHost(&A_h, bytes_a);
    cudaMallocHost(&B_h, bytes_b);
    cudaMalloc(&A, bytes_a);
    cudaMalloc(&B, bytes_b);
    cudaMalloc(&C, bytes_c);

    randn(A_h, elems_a);
    randn(B_h, elems_b);
    cudaMemcpy(A, A_h, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, bytes_b, cudaMemcpyHostToDevice);

    for (int i = 0; i < config.warmup_iters; i++) {
        launch(matmul, A, B, C, config);
    }

    cudaProfilerStart();
    for (int i = 0; i < config.profile_iters; i++) {
        launch(matmul, A, B, C, config);
    }
    cudaProfilerStop();

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFreeHost(A_h);
    cudaFreeHost(B_h);
}

int main()
{
    ProfileConfig config{ .warmup_iters = 10 };
    profile_kernel(PROFILE_MATMUL_SYMBOL, config);

    return 0;
}
