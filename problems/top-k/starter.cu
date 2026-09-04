#include "solve.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace {

__global__ void top_k_rows_kernel(const float* input,
                                  float* output_values,
                                  int* output_indices,
                                  int cols,
                                  int k,
                                  int padded_cols) {
    extern __shared__ float values[];
    int* indices = reinterpret_cast<int*>(values + padded_cols);
    const int row = blockIdx.x;
    const int input_offset = row * cols;

    for (int col = threadIdx.x; col < padded_cols; col += blockDim.x) {
        if (col < cols) {
            values[col] = input[input_offset + col];
            indices[col] = col;
        } else {
            values[col] = -CUDART_INF_F;
            indices[col] = col;
        }
    }
    __syncthreads();

    for (int width = 2; width <= padded_cols; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            for (int left = threadIdx.x; left < padded_cols; left += blockDim.x) {
                const int right = left ^ stride;
                if (right > left) {
                    const bool descending = (left & width) == 0;
                    const float left_value = values[left];
                    const float right_value = values[right];
                    const bool swap = descending ? left_value < right_value
                                                 : left_value > right_value;
                    if (swap) {
                        values[left] = right_value;
                        values[right] = left_value;
                        const int left_index = indices[left];
                        indices[left] = indices[right];
                        indices[right] = left_index;
                    }
                }
            }
            __syncthreads();
        }
    }

    const int output_offset = row * k;
    for (int rank = threadIdx.x; rank < k; rank += blockDim.x) {
        output_values[output_offset + rank] = values[rank];
        output_indices[output_offset + rank] = indices[rank];
    }
}

}  // namespace

void solve(const float* input,
           float* values,
           int* indices,
           int rows,
           int cols,
           int k,
           cudaStream_t stream) {
    int padded_cols = 1;
    while (padded_cols < cols) padded_cols <<= 1;
    const int threads = padded_cols < 256 ? padded_cols : 256;
    const std::size_t shared_bytes =
        static_cast<std::size_t>(padded_cols) * (sizeof(float) + sizeof(int));
    top_k_rows_kernel<<<rows, threads, shared_bytes, stream>>>(
        input, values, indices, cols, k, padded_cols);
}
