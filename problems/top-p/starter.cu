#include "solve.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace {

constexpr int kThreads = 256;

__global__ void top_p_rows_kernel(const float* probabilities,
                                  float* output,
                                  int* counts,
                                  int cols,
                                  float p,
                                  int sort_size) {
    extern __shared__ float ordered[];
    __shared__ int retained_count;

    const int row = blockIdx.x;
    const int row_offset = row * cols;

    for (int rank = threadIdx.x; rank < sort_size; rank += blockDim.x) {
        ordered[rank] = rank < cols ? probabilities[row_offset + rank] : -1.0F;
    }
    __syncthreads();

    for (int size = 2; size <= sort_size; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int index = threadIdx.x; index < sort_size; index += blockDim.x) {
                const int partner = index ^ stride;
                if (partner > index) {
                    const float left = ordered[index];
                    const float right = ordered[partner];
                    const bool descending_half = (index & size) == 0;
                    const bool should_swap =
                        descending_half ? left < right : left > right;
                    if (should_swap) {
                        ordered[index] = right;
                        ordered[partner] = left;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        int keep = cols;
        if (p < 1.0F) {
            float cumulative = 0.0F;
            for (int rank = 0; rank < cols; ++rank) {
                cumulative += ordered[rank];
                if (cumulative >= p) {
                    keep = rank + 1;
                    break;
                }
            }
        }
        retained_count = keep;
        counts[row] = keep;
    }
    __syncthreads();

    for (int rank = threadIdx.x; rank < cols; rank += blockDim.x) {
        output[row_offset + rank] =
            rank < retained_count ? ordered[rank] : 0.0F;
    }
}

int next_power_of_two(int value) {
    int result = 1;
    while (result < value) result <<= 1;
    return result;
}

}  // namespace

void solve(const float* probabilities,
           float* output,
           int* counts,
           int rows,
           int cols,
           float p,
           cudaStream_t stream) {
    const int sort_size = next_power_of_two(cols);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(sort_size) * sizeof(float);
    top_p_rows_kernel<<<rows, kThreads, shared_bytes, stream>>>(
        probabilities, output, counts, cols, p, sort_size);
}
