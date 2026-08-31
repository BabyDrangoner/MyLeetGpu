# 分组查询注意力

Grouped Query Attention（GQA）让一组连续的 query heads 共享同一个 key/value head。给定 `query_heads` 个 query heads 和 `key_value_heads` 个 key/value heads，平台保证：

```text
query_heads % key_value_heads == 0
group_size = query_heads / key_value_heads
kv_head(query_head) = query_head // group_size
```

对每个 query head，使用映射到的 key/value head 完成缩放点积注意力：

```text
scores = query @ grouped_key.transpose(-2, -1) / sqrt(head_dim)
weights = softmax(mask(scores), dim=-1)
output = weights @ grouped_value
```

`attention_mask` 中的 `True` 表示对应 key 可以参与注意力，`False` 表示必须在 softmax 前屏蔽。平台保证每个 query 至少有一个可参与的 key。

## 实现约束

本题输入已经完成线性投影，不包含输出投影、dropout 和反向传播。四个输入均为 GPU 0 上的连续 Tensor：

- `query` 的形状为 `[batch, query_heads, query_length, head_dim]`；
- `key`、`value` 的形状为 `[batch, key_value_heads, key_length, head_dim]`；
- `attention_mask` 的形状为 `[batch, 1, query_length, key_length]`，dtype 为 `torch.bool`。

浮点输入均为有限 `torch.float32`。请返回形状为 `[batch, query_heads, query_length, head_dim]` 的新 `torch.float32` CUDA Tensor；不得修改或复用输入存储。平台已进入受控 CUDA stream 和 inference mode，不要移动数据到 CPU、执行设备级同步或修改全局 PyTorch 状态。

`key_value_heads == 1` 是 Multi-Query Attention 边界，`key_value_heads == query_heads` 则退化为普通 MHA；实现必须同时正确处理。特别注意连续分组要求 `repeat_interleave` 语义，而不是把所有 key/value heads 循环重复。

## 正确性

参考实现使用更高精度独立计算，判定条件为：

```text
abs(actual - expected) <= 2e-4 + 2e-4 * abs(expected)
```

NaN 和无穷值不会通过。公开用例覆盖可辨认的 head 分组、padding mask 和 causal mask；完整验证还覆盖 MQA、退化 MHA、奇数组大小、单 query、极端 logits 和固定随机大输入，以检查 head 映射、缩放、mask、softmax 轴与数值稳定性。

## 性能说明

平台预先生成并上传输入，完成预热后使用当前 stream 上的 CUDA Events 采样。短用例会在一个计时区间内重复调用 `solve` 并折算为单次耗时。计时包含实现产生的全部 GPU 工作与输出分配；不能调用平台禁止的内置 attention、编译器或计时 API 绕过题目。
