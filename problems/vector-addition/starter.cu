#include "solve.h"

#include <cuda_runtime.h>

namespace {

__global__ void vector_add_kernel(const float* a,
                                  const float* b,
                                  float* output,
                                  int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        output[index] = a[index] + b[index];
    }
}

}  // namespace

void solve(const float* a,
           const float* b,
           float* output,
           int n,
           cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads, 0, stream>>>(a, b, output, n);
}

