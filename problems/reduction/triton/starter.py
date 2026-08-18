import torch
import triton
import triton.language as tl


@triton.jit
def reduction_kernel(
    input_ptr,
    output_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets, mask=offsets < n, other=0.0)
    partial = tl.sum(values, axis=0)
    tl.atomic_add(output_ptr, partial)


def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    n: int,
) -> None:
    # Each call must independently produce a complete result. zero_ and the Triton
    # launch both use the stream selected by the platform harness.
    output.zero_()
    block_size = 1024
    grid = (triton.cdiv(n, block_size),)
    reduction_kernel[grid](input, output, n, BLOCK_SIZE=block_size, num_warps=4)
