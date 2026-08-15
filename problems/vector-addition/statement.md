# 向量逐元素相加

给定两个长度相同的单精度浮点向量 `a` 和 `b`，请在 GPU 上计算：

```text
output[i] = a[i] + b[i]
```

## 接口

你只需要实现 `solve`，不能提供 `main`：

```cpp
void solve(
    const float* a,
    const float* b,
    float* output,
    int n,
    cudaStream_t stream);
```

`a`、`b` 和 `output` 都指向设备内存，且彼此不重叠。`n` 的范围为
`1 <= n <= 16,777,216`。输入只包含 `[-100, 100]` 内的有限浮点数。

所有异步工作必须提交到传入的 `stream`。函数返回前不需要同步；测试平台负责在正确的边界同步并检查 CUDA 错误。不要分配或释放输入、输出指针。

## 正确性

每个输出元素与 CPU 参考值进行浮点比较，判定条件为：

```text
abs(actual - expected) <= 1e-6 + 1e-6 * abs(expected)
```

NaN 不会被视为相等，无穷值必须符号一致。公开测试覆盖极小输入和非整块长度，完整验证还会使用固定随机种子的边界及大规模输入。

## 性能说明

性能测试只计量 `solve` 提交到指定 stream 的 GPU 工作。平台预先完成输入生成、设备内存分配和 H2D 拷贝，并先预热再采样。短任务会在单个 CUDA Event 区间中重复执行多次，最终报告每次调用的耗时。

