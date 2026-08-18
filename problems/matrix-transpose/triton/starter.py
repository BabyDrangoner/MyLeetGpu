import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    input_ptr,
    output_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    count = rows * cols
    mask = offsets < count
    row = offsets // cols
    col = offsets - row * cols
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + col * rows + row, values, mask=mask)


def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    rows: int,
    cols: int,
) -> None:
    block_size = 256
    grid = (triton.cdiv(rows * cols, block_size),)
    transpose_kernel[grid](
        input,
        output,
        rows,
        cols,
        BLOCK_SIZE=block_size,
    )
