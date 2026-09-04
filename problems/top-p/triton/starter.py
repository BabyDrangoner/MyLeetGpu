import torch
import triton
import triton.language as tl


@triton.jit
def top_p_rows_kernel(
    probabilities_ptr,
    output_ptr,
    counts_ptr,
    cols,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    valid = offsets < cols
    values = tl.load(
        probabilities_ptr + row * cols + offsets,
        mask=valid,
        other=-1.0,
    )
    ordered = tl.sort(values, descending=True)
    cumulative = tl.cumsum(ordered, axis=0)
    below_threshold = valid & (cumulative < p)
    candidate_count = tl.sum(below_threshold.to(tl.int32), axis=0) + 1
    retained_count = tl.where(
        p >= 1.0,
        cols,
        tl.minimum(candidate_count, cols),
    )
    filtered = tl.where(offsets < retained_count, ordered, 0.0)
    tl.store(output_ptr + row * cols + offsets, filtered, mask=valid)
    tl.store(counts_ptr + row, retained_count)


def solve(
    probabilities: torch.Tensor,
    output: torch.Tensor,
    counts: torch.Tensor,
    rows: int,
    cols: int,
    p: float,
) -> None:
    block_size = triton.next_power_of_2(cols)
    top_p_rows_kernel[(rows,)](
        probabilities,
        output,
        counts,
        cols,
        p,
        BLOCK_SIZE=block_size,
    )
