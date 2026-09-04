import torch
import triton
import triton.language as tl


@triton.jit
def top_k_rows_kernel(
    input_ptr,
    values_ptr,
    indices_ptr,
    cols,
    k,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < cols
    candidates = tl.load(input_ptr + row * cols + offsets, mask=mask, other=-1.0e20)

    for rank in range(0, k):
        best_value = tl.max(candidates, axis=0)
        best_index = tl.argmax(candidates, axis=0, tie_break_left=True)
        tl.store(values_ptr + row * k + rank, best_value)
        tl.store(indices_ptr + row * k + rank, best_index)
        candidates = tl.where(offsets == best_index, -1.0e20, candidates)


def solve(
    input: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    rows: int,
    cols: int,
    k: int,
) -> None:
    block_size = triton.next_power_of_2(cols)
    top_k_rows_kernel[(rows,)](
        input,
        values,
        indices,
        cols,
        k,
        BLOCK_SIZE=block_size,
    )
