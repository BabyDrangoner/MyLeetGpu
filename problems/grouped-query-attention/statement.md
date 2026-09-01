# 分组查询自注意力

实现一个带线性投影的 Grouped Query Self-Attention（GQA）类。类的 `forward` 只接收同一个输入 `X` 和 Python 布尔值 `isCasual`；query、key、value 都必须从 `X` 计算，因此本题是自注意力而不是交叉注意力。

平台通过构造器提供 query head 数、key/value head 数和四个只读权重：

```text
X            : [batch, sequence_length, embedding_dim]
qWeight      : [embedding_dim, embedding_dim]
kWeight      : [embedding_dim, key_value_dim]
vWeight      : [embedding_dim, key_value_dim]
outputWeight : [embedding_dim, embedding_dim]

head_dim     = embedding_dim / numQueryHeads
key_value_dim = numKeyValueHeads * head_dim
```

所有权重都在右侧参与矩阵乘法，不需要转置：

```text
Q = X @ qWeight
K = X @ kWeight
V = X @ vWeight
```

把 `Q` 拆为 `numQueryHeads` 个 heads，把 `K`、`V` 拆为 `numKeyValueHeads` 个 heads。平台保证：

```text
embedding_dim % numQueryHeads == 0
numQueryHeads % numKeyValueHeads == 0
group_size = numQueryHeads / numKeyValueHeads
kv_head(query_head) = query_head // group_size
```

也就是说，一组连续的 query heads 共享同一个 K/V head。扩展 K/V 时必须使用与 `repeat_interleave(..., dim=1)` 等价的连续分组语义，不能循环重复完整 head 序列。

对每个 query head 计算缩放点积注意力：

```text
scores = Q @ grouped_K.transpose(-2, -1) / sqrt(head_dim)
weights = softmax(mask(scores), dim=-1)
context = weights @ grouped_V
output = merge_heads(context) @ outputWeight
```

当 `isCasual` 为 `True` 时，位置 `i` 只能关注位置 `0..i`，即使用包含主对角线的下三角 causal mask；为 `False` 时不屏蔽任何位置。参数名按题目接口拼写为 `isCasual`，其语义是标准的 `is_causal` 开关。

## 实现约束

提交必须定义题目指定的普通 Python 类，不继承 `torch.nn.Module`。平台只在每个验证用例开始时构造一次实例，并复用该实例调用 `forward`。构造器参数是只读状态，不能在 `forward` 中修改或替换；不得依赖跨调用缓存。

`X` 与四个权重均为 GPU 0 上连续、有限的 `torch.float32` Tensor。请返回形状为 `[batch, sequence_length, embedding_dim]` 的新 CUDA `torch.float32` Tensor；输出不得复用 `X` 或任一权重的底层存储，也不得修改任何输入或构造器状态。

本题没有 bias、dropout、padding mask、输出残差或反向传播。平台已经进入受控 CUDA stream 和 inference mode；不要移动数据到 CPU、显式同步设备或修改全局 PyTorch 状态。

`numKeyValueHeads == 1` 是 Multi-Query Attention 边界，`numKeyValueHeads == numQueryHeads` 则退化为普通 MHA，两种情况都必须正确处理。

## 正确性

参考实现使用 CPU `float64` 独立完成投影、分组、mask、softmax 和输出投影，判定条件为：

```text
abs(actual - expected) <= 3e-4 + 3e-4 * abs(expected)
```

NaN 和无穷值不会通过。公开与隐藏用例覆盖可辨认的连续 head 分组、MQA、退化 MHA、奇数组大小、单元素边界、causal/non-causal 成对输入、极端 logits 和较大随机输入。

## 性能说明

Benchmark 在计时前生成并上传 `X` 与权重，构造一次 `GroupedQueryAttention` 实例并完成正确性检查。预热后，平台使用当前 stream 上的 CUDA Events 只采样重复调用 `forward(X, isCasual)` 产生的 GPU 工作；类构造不计入耗时，输出分配、投影、mask、softmax 和输出投影均计入耗时。
