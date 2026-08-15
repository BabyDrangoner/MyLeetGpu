#pragma once

#include <cuda_runtime_api.h>

void solve(const float* a,
           const float* b,
           float* output,
           int n,
           cudaStream_t stream);

