#include "solve.h"

#include <cuda_runtime.h>

namespace {

constexpr int kTile = 16;

__global__ void matrix_multiplication_kernel(const float* a,
                                             const float* b,
                                             float* c,
                                             int m,
                                             int k,
                                             int n) {
    __shared__ float a_tile[kTile][kTile];
    __shared__ float b_tile[kTile][kTile];

    const int row = static_cast<int>(blockIdx.y) * kTile + threadIdx.y;
    const int col = static_cast<int>(blockIdx.x) * kTile + threadIdx.x;
    float accumulator = 0.0F;

    for (int tile_start = 0; tile_start < k; tile_start += kTile) {
        const int a_col = tile_start + threadIdx.x;
        const int b_row = tile_start + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] =
            row < m && a_col < k ? a[row * k + a_col] : 0.0F;
        b_tile[threadIdx.y][threadIdx.x] =
            b_row < k && col < n ? b[b_row * n + col] : 0.0F;
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < kTile; ++inner) {
            accumulator += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) c[row * n + col] = accumulator;
}

}  // namespace

void solve(const float* a,
           const float* b,
           float* c,
           int m,
           int k,
           int n,
           cudaStream_t stream) {
    const dim3 threads(kTile, kTile);
    const dim3 blocks((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
    matrix_multiplication_kernel<<<blocks, threads, 0, stream>>>(a, b, c, m, k, n);
}
