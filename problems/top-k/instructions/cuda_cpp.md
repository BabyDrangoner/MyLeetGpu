## CUDA C++ 接口

提交源码只实现下列函数，不要提供 `main`：

```cpp
void solve(
    const float* input,
    float* values,
    int* indices,
    int rows,
    int cols,
    int k,
    cudaStream_t stream);
```

三个指针都指向设备内存。`input` 按 `rows x cols` 行主序存放，`values` 和 `indices` 按 `rows x k` 行主序存放。所有 kernel 和异步操作必须提交到传入的 `stream`；不要释放、替换平台传入的指针，也不要调用设备级同步。
