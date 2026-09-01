import torch


class MultiHeadAttention:
    def __init__(
        self,
        numHeads: int,
        qWeight: torch.Tensor,
        kWeight: torch.Tensor,
        vWeight: torch.Tensor,
        outputWeight: torch.Tensor,
    ):
        self.numHeads = numHeads
        self.qWeight = qWeight
        self.kWeight = kWeight
        self.vWeight = vWeight
        self.outputWeight = outputWeight

    def forward(self, X: torch.Tensor, isCasual: bool) -> torch.Tensor:
        batchSize = X.shape[0]
        sequenceLength = X.shape[1]
        embedDim = X.shape[2]
        headDim = embedDim // self.numHeads

        query = torch.matmul(X, self.qWeight)
        key = torch.matmul(X, self.kWeight)
        value = torch.matmul(X, self.vWeight)

        query = query.reshape(batchSize, sequenceLength, self.numHeads, headDim).transpose(1, 2)
        key = key.reshape(batchSize, sequenceLength, self.numHeads, headDim).transpose(1, 2)
        value = value.reshape(batchSize, sequenceLength, self.numHeads, headDim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) * (headDim**-0.5)
        if isCasual:
            positions = torch.arange(sequenceLength, device=X.device)
            causalMask = positions.reshape(sequenceLength, 1) >= positions.reshape(
                1, sequenceLength
            )
            scores = scores.masked_fill(~causalMask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, value)
        concatenated = (
            context.transpose(1, 2).contiguous().reshape(batchSize, sequenceLength, embedDim)
        )
        return torch.matmul(concatenated, self.outputWeight)
