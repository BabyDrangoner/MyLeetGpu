# 行主序矩阵转置

输入是一个按行主序连续存放的 `rows x cols` 单精度浮点矩阵。请在 GPU 上生成它的转置矩阵；输出也按行主序存放，形状为 `cols x rows`：

```text
output[col * rows + row] = input[row * cols + col]
```

## 接口

你只需要实现以下函数，不能提供 `main`：

```cpp
void solve(
    const float* input,
    float* output,
    int rows,
    int cols,
    cudaStream_t stream);
```

`input` 和 `output` 是互不重叠的设备内存。`rows`、`cols` 均至少为 1、至多为 8192，且元素总数不超过 16,777,216。输入都是 `[-50, 50]` 内的有限值。

所有异步操作必须使用传入的 `stream`。函数返回前无需同步，也不要释放或替换平台传入的缓冲区。

## 正确性

转置只改变元素位置，因此结果应与对应输入位模式一致。公开样例包含方阵以外的矩阵；完整验证还会检查尺寸不是 CUDA block 整数倍的高矩阵、宽矩阵和较大方阵。内部用例由固定种子生成，执行结果可复现，但失败输出不会暴露其输入。

## 性能说明

平台会在计时前创建 CUDA 上下文、生成并上传输入、分配内存和执行预热。每个样本使用同一条 stream 上的一对 CUDA Events，仅包围若干次 `solve`；报告值会除以重复次数。可考虑合并访存以及 shared-memory bank conflict 对转置吞吐的影响。

