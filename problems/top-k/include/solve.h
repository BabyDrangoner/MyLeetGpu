#pragma once

#include <cuda_runtime_api.h>

void solve(const float* input,
           float* values,
           int* indices,
           int rows,
           int cols,
           int k,
           cudaStream_t stream);
