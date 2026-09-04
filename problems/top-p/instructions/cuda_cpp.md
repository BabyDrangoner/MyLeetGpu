## CUDA C++ 接口

提交源码只实现下列函数，不要提供 `main`：

```cpp
void solve(
    const float* probabilities,
    float* output,
    int* counts,
    int rows,
    int cols,
    float p,
    cudaStream_t stream);
```

三个指针都指向设备内存。请将每行概率按降序排名写入 `output`，只保留累积和首次达到 `p` 的最短前缀，将其余排名位置写成 `+0.0`，并把保留数写入 `counts`。当 `p == 1` 时必须保留全行。

所有 kernel 和异步操作必须提交到传入的 `stream`；不要修改 `probabilities`，不要释放或替换平台传入的指针，也不要调用设备级同步。
