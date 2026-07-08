#include <math_constants.h>
#include <cstdint>


namespace
{

template <uint32_t TB_SIZE>
__global__
void softmax_kernel(const float* in, float* out, int N, int H)
{
    extern __shared__ float vec[];
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;

    const float* in_begin = in + bid * H;

    // pass 1: max reduce
    float local_max = -CUDART_INF_F;
    for (uint32_t i = tid; i < H; i += TB_SIZE) {
        local_max = fmaxf(local_max, in_begin[i]);
    }
    vec[tid] = local_max;
    __syncthreads();

    for (uint32_t offset = (TB_SIZE >> 1); offset > 32; offset >>= 1) {
        if (tid < offset) {
            vec[tid] = fmaxf(vec[tid], vec[tid + offset]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile float* v = vec;
        v[tid] = fmaxf(v[tid], v[tid + 32]);
        v[tid] = fmaxf(v[tid], v[tid + 16]);
        v[tid] = fmaxf(v[tid], v[tid + 8]);
        v[tid] = fmaxf(v[tid], v[tid + 4]);
        v[tid] = fmaxf(v[tid], v[tid + 2]);
        v[tid] = fmaxf(v[tid], v[tid + 1]);
    }
    __syncthreads();
    const float row_max = vec[0];

    // pass 2: sum reduce
    float* out_begin = out + bid * H;
    float local_sum = 0.f;
    for (uint32_t i = tid; i < H; i += TB_SIZE) {
        local_sum += expf(in_begin[i] - row_max);
    }
    vec[tid] = local_sum;
    __syncthreads();

    for (uint32_t offset = (TB_SIZE >> 1); offset > 32; offset >>= 1) {
        if (tid < offset) {
            vec[tid] += vec[tid + offset];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile float* v = vec;
        v[tid] += v[tid + 32];
        v[tid] += v[tid + 16];
        v[tid] += v[tid + 8];
        v[tid] += v[tid + 4];
        v[tid] += v[tid + 2];
        v[tid] += v[tid + 1];
    }
    __syncthreads();

    // pass 3: write
    const float norm = 1.f / vec[0];
    for (uint32_t i = tid; i < H; i += TB_SIZE) {
        out_begin[i] = expf(in_begin[i] - row_max) * norm;
    }
}

}

void softmax_v3a(const float* in, float* out, int N, int H)
{
    constexpr uint32_t TB_SIZE = 256;
    constexpr uint32_t SMEM_BYTE_SIZE = TB_SIZE * sizeof(float);
    softmax_kernel<TB_SIZE>
        <<<N, TB_SIZE, SMEM_BYTE_SIZE>>>(in, out, N, H);
}
