#pragma once

#include <cuda_runtime_api.h>

void solve(const float* input,
           float* output,
           int rows,
           int cols,
           cudaStream_t stream);

