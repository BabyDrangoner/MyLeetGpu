import torch


def solve(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    repeats_per_group = query.shape[1] // key.shape[1]
    grouped_key = torch.repeat_interleave(key, repeats_per_group, dim=1)
    grouped_value = torch.repeat_interleave(value, repeats_per_group, dim=1)
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query, grouped_key.transpose(-2, -1)) * scale
    masked_scores = scores.masked_fill(~attention_mask, float("-inf"))
    weights = torch.softmax(masked_scores, dim=-1)
    return torch.matmul(weights, grouped_value)
