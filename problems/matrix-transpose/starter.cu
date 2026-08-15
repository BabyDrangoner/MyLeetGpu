#include "solve.h"

#include <cuda_runtime.h>

namespace {

constexpr int kTile = 32;
constexpr int kBlockRows = 8;

__global__ void transpose_kernel(const float* input,
                                 float* output,
                                 int rows,
                                 int cols) {
    __shared__ float tile[kTile][kTile + 1];

    const int input_col = blockIdx.x * kTile + threadIdx.x;
    const int input_row = blockIdx.y * kTile + threadIdx.y;
    for (int offset = 0; offset < kTile; offset += kBlockRows) {
        if (input_col < cols && input_row + offset < rows) {
            tile[threadIdx.y + offset][threadIdx.x] =
                input[(input_row + offset) * cols + input_col];
        }
    }
    __syncthreads();

    const int output_row = blockIdx.x * kTile + threadIdx.y;
    const int output_col = blockIdx.y * kTile + threadIdx.x;
    for (int offset = 0; offset < kTile; offset += kBlockRows) {
        if (output_row + offset < cols && output_col < rows) {
            output[(output_row + offset) * rows + output_col] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

}  // namespace

void solve(const float* input,
           float* output,
           int rows,
           int cols,
           cudaStream_t stream) {
    const dim3 threads(kTile, kBlockRows);
    const dim3 blocks((cols + kTile - 1) / kTile,
                      (rows + kTile - 1) / kTile);
    transpose_kernel<<<blocks, threads, 0, stream>>>(input, output, rows, cols);
}

