#include <random>

#include <cuda_runtime.h>
#include <cublas_v2.h>


void matmul_v5(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
);

void randn(float* A, int M, int N)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::normal_distribution<float> dist(0.f, 1.f);

    for (int i = 0; i < M * N; i++) {
        A[i] = dist(gen);
    }
}

int main()
{
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;

    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    cudaMallocHost(reinterpret_cast<float**>(&A), M * K * sizeof(float));
    cudaMallocHost(reinterpret_cast<float**>(&B), K * N * sizeof(float));
    cudaMallocHost(reinterpret_cast<float**>(&C), M * N * sizeof(float));

    randn(A, M, K);
    randn(B, K, N);

    float* A_d = nullptr;
    float* B_d = nullptr;
    float* C_d = nullptr;

    cudaMalloc(reinterpret_cast<float**>(&A_d), M * K * sizeof(float));
    cudaMalloc(reinterpret_cast<float**>(&B_d), K * N * sizeof(float));
    cudaMalloc(reinterpret_cast<float**>(&C_d), M * N * sizeof(float));

    cudaMemcpyAsync(A_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaStreamSynchronize(0);

    constexpr float alpha = 1.f;
    constexpr float beta  = 0.f;


    // matmul_v5(A_d, B_d, C_d, M, N, K);
    void matmul_cublas_gemm(
        const float* A,
        const float* B,
        float* C,
        int M, int N, int K
    );   
    matmul_cublas_gemm(A_d, B_d, C_d, M, N, K);

    cudaMemcpy(C, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}