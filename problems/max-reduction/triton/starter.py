import torch
import triton
import triton.language as tl


@triton.jit
def initialize_max_kernel(output_ptr):
    tl.store(output_ptr, -3.4028234663852886e38)


@triton.jit
def reduction_kernel(
    input_ptr,
    output_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        input_ptr + offsets,
        mask=offsets < n,
        other=-3.4028234663852886e38,
    )
    partial = tl.max(values, axis=0)
    # Triton's float atomic_max distinguishes the -0.0 bit pattern internally.
    # Canonicalize either signed zero so an all--0.0 input can replace -FLT_MAX.
    partial = tl.where(partial == 0.0, 0.0, partial)
    tl.atomic_max(output_ptr, partial)


def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    n: int,
) -> None:
    # Both launches use the stream selected by the platform harness.
    initialize_max_kernel[(1,)](output)
    block_size = 1024
    grid = (triton.cdiv(n, block_size),)
    reduction_kernel[grid](input, output, n, BLOCK_SIZE=block_size, num_warps=4)
