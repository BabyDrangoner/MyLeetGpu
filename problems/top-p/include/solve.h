#pragma once

#include <cuda_runtime_api.h>

void solve(const float* probabilities,
           float* output,
           int* counts,
           int rows,
           int cols,
           float p,
           cudaStream_t stream);
