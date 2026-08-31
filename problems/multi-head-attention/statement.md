# 多头缩放点积注意力

给定已经完成线性投影的多头 `query`、`key` 和 `value`，请实现多头缩放点积注意力的前向计算。对每个 batch 和 head，定义：

```text
scores = query @ key.transpose(-2, -1) / sqrt(head_dim)
weights = softmax(mask(scores), dim=-1)
output = weights @ value
```

`attention_mask` 中的 `True` 表示该 key 可以参与注意力，`False` 表示对应分数必须在 softmax 前被屏蔽。平台保证每个 query 至少有一个可参与的 key。

## 实现约束

本题只考察预投影后的 attention 核心，不包含 Q/K/V 线性投影、输出投影、head 拼接、dropout 和反向传播。四个输入均为 GPU 0 上的连续 Tensor：

- `query` 的形状为 `[batch, heads, query_length, head_dim]`；
- `key`、`value` 的形状为 `[batch, heads, key_length, head_dim]`；
- `attention_mask` 的形状为 `[batch, 1, query_length, key_length]`，dtype 为 `torch.bool`，在 head 维广播。

浮点输入均为有限 `torch.float32`。请返回形状为 `[batch, heads, query_length, head_dim]` 的新 `torch.float32` CUDA Tensor；不得修改或复用任何输入的存储。平台已进入受控 CUDA stream 和 inference mode，不要移动数据到 CPU、执行设备级同步或修改全局 PyTorch 状态。

## 正确性

参考实现使用更高精度计算缩放、mask、softmax 和加权求和。结果需满足：

```text
abs(actual - expected) <= 2e-4 + 2e-4 * abs(expected)
```

NaN 和无穷值不会通过。公开用例覆盖手工小例子、padding mask 和 causal mask；完整验证还覆盖单 token、交叉注意力、非二次幂 head_dim、每行仅一个有效 key、head 隔离和极端 logits，以检查缩放维度、softmax 轴、mask 语义及数值稳定性。

## 性能说明

平台在计时前完成输入生成和上传，并执行固定次数预热。每个样本由当前 stream 上的一对 CUDA Events 包围若干次 `solve`，报告值会除以内部重复次数。计时包含提交实现产生的全部 GPU 工作与输出分配；不能调用平台禁止的内置 attention、编译器或计时 API 绕过题目。
