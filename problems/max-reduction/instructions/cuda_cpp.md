## CUDA C++ 接口

提交源码只实现下列函数，不要提供 `main`：

```cpp
void solve(
    const float* input,
    float* output,
    int n,
    cudaStream_t stream);
```

两个指针都指向设备内存。每次调用必须独立写出完整结果；若使用原子归约，必须先在同一 `stream` 上把 `output[0]` 初始化为不大于任意合法输入的值。不要释放、替换平台传入的指针，也不要调用设备级同步。
