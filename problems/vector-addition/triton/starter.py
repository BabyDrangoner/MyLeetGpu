import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a_ptr, b_ptr, output_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, a + b, mask=mask)


def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    n: int,
) -> None:
    block_size = 256
    grid = (triton.cdiv(n, block_size),)
    vector_add_kernel[grid](a, b, output, n, BLOCK_SIZE=block_size)
