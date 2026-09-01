import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m: tl.constexpr,
    k: tl.constexpr,
    n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    inner_offsets = tl.arange(0, BLOCK_K)
    a_offsets = row_offsets[:, None] * k + inner_offsets[None, :]
    b_offsets = inner_offsets[:, None] * n + col_offsets[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for inner_start in range(0, k, BLOCK_K):
        a_values = tl.load(
            a_ptr + a_offsets,
            mask=(row_offsets[:, None] < m) & (inner_start + inner_offsets[None, :] < k),
            other=0.0,
        )
        b_values = tl.load(
            b_ptr + b_offsets,
            mask=(inner_start + inner_offsets[:, None] < k) & (col_offsets[None, :] < n),
            other=0.0,
        )
        accumulator += tl.dot(a_values, b_values)
        a_offsets += BLOCK_K
        b_offsets += BLOCK_K * n

    output_offsets = row_offsets[:, None] * n + col_offsets[None, :]
    output_mask = (row_offsets[:, None] < m) & (col_offsets[None, :] < n)
    tl.store(c_ptr + output_offsets, accumulator, mask=output_mask)


def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    m: int,
    k: int,
    n: int,
) -> None:
    block_m = 16
    block_k = 32
    block_n = 16
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    matrix_multiplication_kernel[grid](
        a,
        b,
        c,
        m,
        k,
        n,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_N=block_n,
        num_warps=4,
    )
