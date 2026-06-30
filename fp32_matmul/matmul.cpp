#include <torch/extension.h>

using MatmulFn = void(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
);

MatmulFn matmul_cublas_gemm;
MatmulFn matmul_v1;
MatmulFn matmul_v2;
MatmulFn matmul_v3;
MatmulFn matmul_v4;
MatmulFn matmul_v5;

template <MatmulFn matmul_fn>
torch::Tensor matmul(torch::Tensor A, torch::Tensor B)
{
    const int M = A.size(0);
    const int N = B.size(1);
    const int K = A.size(1);
    auto C = torch::empty({M, N}, A.options());
    matmul_fn(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cublas_gemm", &matmul<matmul_cublas_gemm>, "cublasGemmEx");
    m.def("matmul_v1", &matmul<matmul_v1>, "Matrix multiplication v1");
    m.def("matmul_v2", &matmul<matmul_v2>, "Matrix multiplication v2");
    m.def("matmul_v3", &matmul<matmul_v3>, "Matrix multiplication v3");
    m.def("matmul_v4", &matmul<matmul_v4>, "Matrix multiplication v4");
    m.def("matmul_v5", &matmul<matmul_v5>, "Matrix multiplication v5");
}