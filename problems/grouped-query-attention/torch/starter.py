import torch


class GroupedQueryAttention:
    def __init__(
        self,
        numQueryHeads: int,
        numKeyValueHeads: int,
        qWeight: torch.Tensor,
        kWeight: torch.Tensor,
        vWeight: torch.Tensor,
        outputWeight: torch.Tensor,
    ):
        self.numQueryHeads = numQueryHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.qWeight = qWeight
        self.kWeight = kWeight
        self.vWeight = vWeight
        self.outputWeight = outputWeight

    def forward(self, X: torch.Tensor, isCasual: bool) -> torch.Tensor:
        batchSize, sequenceLength, embeddingDim = X.shape
        headDim = embeddingDim // self.numQueryHeads
        keyValueDim = self.numKeyValueHeads * headDim

        query = torch.matmul(X, self.qWeight)
        key = torch.matmul(X, self.kWeight)
        value = torch.matmul(X, self.vWeight)

        query = query.reshape(batchSize, sequenceLength, self.numQueryHeads, headDim).transpose(
            1, 2
        )
        key = key.reshape(batchSize, sequenceLength, self.numKeyValueHeads, headDim).transpose(1, 2)
        value = value.reshape(batchSize, sequenceLength, self.numKeyValueHeads, headDim).transpose(
            1, 2
        )

        repeatsPerGroup = self.numQueryHeads // self.numKeyValueHeads
        groupedKey = torch.repeat_interleave(key, repeatsPerGroup, dim=1)
        groupedValue = torch.repeat_interleave(value, repeatsPerGroup, dim=1)

        scores = torch.matmul(query, groupedKey.transpose(-2, -1)) * (headDim**-0.5)
        if isCasual:
            positions = torch.arange(sequenceLength, device=X.device)
            causalMask = positions.unsqueeze(0) <= positions.unsqueeze(1)
            scores = scores.masked_fill(~causalMask, float("-inf"))

        probabilities = torch.softmax(scores, dim=-1)
        context = torch.matmul(probabilities, groupedValue)
        merged = (
            context.transpose(1, 2)
            .contiguous()
            .reshape(batchSize, sequenceLength, keyValueDim * repeatsPerGroup)
        )
        return torch.matmul(merged, self.outputWeight)
