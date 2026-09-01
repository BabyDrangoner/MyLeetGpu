import torch
import triton
import triton.language as tl


@triton.jit
def softmax_rows_kernel(
    input_ptr,
    output_ptr,
    cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < cols
    values = tl.load(input_ptr + row * cols + offsets, mask=mask, other=-1.0e20)
    row_max = tl.max(values, axis=0)
    numerators = tl.exp(values - row_max)
    denominator = tl.sum(numerators, axis=0)
    tl.store(output_ptr + row * cols + offsets, numerators / denominator, mask=mask)


def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    rows: int,
    cols: int,
) -> None:
    block_size = triton.next_power_of_2(cols)
    softmax_rows_kernel[(rows,)](
        input,
        output,
        cols,
        BLOCK_SIZE=block_size,
    )
