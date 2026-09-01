# 多头自注意力

请实现一个普通 Python 类 `MultiHeadAttention`，使用同一个输入 `X` 生成 query、key 和 value，并完成多头缩放点积自注意力。类不继承 `torch.nn.Module`；平台把固定的投影权重传给构造器，随后只用 `X` 和 `isCasual` 调用 `forward`。

令 `X` 的形状为 `[batch, sequence_length, embed_dim]`，所有权重均为 `[embed_dim, embed_dim]`。投影使用右乘约定：

```text
Q = X @ qWeight
K = X @ kWeight
V = X @ vWeight
```

`embed_dim` 一定能被 `numHeads` 整除。把 Q、K、V 拆成 `numHeads` 个 head 后，对每个 batch 和 head 计算：

```text
scores = Q @ K.transpose(-2, -1) / sqrt(head_dim)
weights = softmax(mask(scores), dim=-1)
context = weights @ V
output = concatenate(context heads) @ outputWeight
```

这是自注意力，因此 query 和 key 的序列长度相同。题目不包含 bias、dropout、残差连接、归一化或反向传播。

## Causal 语义

`isCasual` 是平台传入的 Python `bool`，名称按接口约定保留。当它为 `False` 时，每个 token 可以关注整个序列；当它为 `True` 时，第 `i` 个 query 只能关注位置 `0..i`，即使用包含对角线的下三角 mask。必须在 softmax 之前屏蔽未来位置。

## 输入与状态约束

- `X` 和四个权重都是 GPU 0 上连续、有限的 `torch.float32` Tensor。
- `X` 的形状为 `[batch, sequence_length, embed_dim]`。
- `qWeight`、`kWeight`、`vWeight` 和 `outputWeight` 的形状均为 `[embed_dim, embed_dim]`。
- 构造器只保存平台传入的 head 数和权重；不得修改或替换这些状态。
- `forward` 必须返回新的 `[batch, sequence_length, embed_dim]` CUDA Tensor，不得修改或复用 `X` 或任一权重的存储。
- 平台已经进入受控 CUDA stream 和 inference mode；不要移动数据到 CPU、显式同步设备或依赖跨调用的可变状态。

## 正确性

参考实现使用 float64 完成投影、缩放、mask、softmax、head 合并和输出投影，最后转换为 float32。结果需满足：

```text
abs(actual - expected) <= 4e-4 + 4e-4 * abs(expected)
```

NaN 和无穷值不会通过。公开用例覆盖 causal 与 non-causal、自注意力投影和非二次幂 head dimension；完整验证还覆盖单 token、多 batch、多 head、head 隔离、投影方向和极端 logits。

## 性能说明

平台在计时前生成并上传 `X` 与权重、构造类实例并完成正确性检查和预热。计时区间只包含同一实例的 `forward(X, isCasual)`；每个样本由当前 stream 上的一对 CUDA Events 测量。短用例会在一个计时区间内重复调用并折算为单次耗时。
