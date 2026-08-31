import torch


def solve(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    masked_scores = scores.masked_fill(~attention_mask, float("-inf"))
    weights = torch.softmax(masked_scores, dim=-1)
    return torch.matmul(weights, value)
