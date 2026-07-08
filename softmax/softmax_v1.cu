#include <math_constants.h>
#include <cstdint>


namespace
{

__global__
void softmax_kernel(const float* in, float* out, int N, int H)
{
    const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        const float* in_begin = in + n * H;
        float* out_begin = out + n * H;

        float row_max = -CUDART_INF_F;
        for (uint32_t i = 0; i < H; i++) {
            row_max = fmaxf(row_max, in_begin[i]);
        }
        float row_sum = 0.f;
        for (uint32_t i = 0; i < H; i++) {
            out_begin[i] = expf(in_begin[i] - row_max);
            row_sum += out_begin[i];
        }
        for (uint32_t i = 0; i < H; i++) {
            out_begin[i] /= row_sum;
        }
    }
}

}

void softmax_v1(const float* in, float* out, int N, int H)
{
    constexpr uint32_t TB_SIZE = 256;
    const uint32_t GRID_SIZE = (N + TB_SIZE - 1) / TB_SIZE;
    softmax_kernel<<<GRID_SIZE, TB_SIZE>>>(in, out, N, H);
}
