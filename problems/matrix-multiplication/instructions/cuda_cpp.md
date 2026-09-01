## CUDA C++ 接口

提交源码只实现下列函数，不要提供 `main`：

```cpp
void solve(
    const float* a,
    const float* b,
    float* c,
    int m,
    int k,
    int n,
    cudaStream_t stream);
```

三个指针都指向设备内存。`a`、`b` 是只读输入，`c` 是平台预先分配的输出，它们互不重叠。所有 kernel 和异步操作必须提交到传入的 `stream`；不要释放、替换平台传入的指针，也不要调用设备级同步。
