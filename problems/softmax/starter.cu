#include "solve.h"

#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

__global__ void softmax_rows_kernel(const float* input,
                                    float* output,
                                    int cols) {
    __shared__ float scratch[kThreads];
    const int row = blockIdx.x;
    const int row_offset = row * cols;

    float local_max = -CUDART_INF_F;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, input[row_offset + col]);
    }
    scratch[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] =
                fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = scratch[0];

    float local_sum = 0.0F;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        local_sum += expf(input[row_offset + col] - row_max);
    }
    scratch[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denominator = scratch[0];

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        output[row_offset + col] =
            expf(input[row_offset + col] - row_max) / denominator;
    }
}

}  // namespace

void solve(const float* input,
           float* output,
           int rows,
           int cols,
           cudaStream_t stream) {
    softmax_rows_kernel<<<rows, kThreads, 0, stream>>>(input, output, cols);
}
