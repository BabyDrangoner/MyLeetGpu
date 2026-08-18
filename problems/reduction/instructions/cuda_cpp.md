## CUDA C++ 接口

提交源码只实现下列函数，不要提供 `main`：

```cpp
void solve(
    const float* input,
    float* output,
    int n,
    cudaStream_t stream);
```

两个指针都指向设备内存。每次调用必须独立写出完整结果；若使用原子累加，必须先在同一 `stream` 上清零 `output[0]`。不要释放、替换平台传入的指针，也不要调用设备级同步。
