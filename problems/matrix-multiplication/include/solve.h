#pragma once

#include <cuda_runtime_api.h>

void solve(const float* a,
           const float* b,
           float* c,
           int m,
           int k,
           int n,
           cudaStream_t stream);
