#include "solve.h"

#include <cuda_runtime.h>

namespace {

__global__ void reduce_sum_kernel(const float* input, float* output, int n) {
    extern __shared__ float partials[];
    const int lane = threadIdx.x;
    float sum = 0.0F;
    for (int index = lane; index < n; index += blockDim.x) {
        sum += input[index];
    }
    partials[lane] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            partials[lane] += partials[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        output[0] = partials[0];
    }
}

}  // namespace

void solve(const float* input,
           float* output,
           int n,
           cudaStream_t stream) {
    constexpr int threads = 256;
    reduce_sum_kernel<<<1, threads, threads * sizeof(float), stream>>>(
        input, output, n);
}

