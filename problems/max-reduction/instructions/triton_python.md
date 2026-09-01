## Triton Python 接口

提交源码应定义下列函数，并可在同一文件中定义一个或多个 `@triton.jit` kernel：

```python
def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    n: int,
) -> None:
    ...
```

`input` 是位于 GPU 0、连续存放的一维 `torch.float32` Tensor，`output` 是 GPU 上含有一个元素的 `torch.float32` Tensor。每次调用必须独立写出完整结果。平台在调用前已经进入受控的 `torch.cuda.stream(stream)` 上下文。请写入已有的 `output`，返回 `None`；不要创建替代输出、移动 Tensor 到 CPU 或调用设备级同步。

提交使用受限 Triton 子集：模块级只允许精确的 `import torch`、`import triton`、`import triton.language as tl`、字面量常量、`@triton.jit` 函数和一个无装饰器的 `solve`。`solve` 是直线式 launcher，只能计算 launch 参数、调用 `triton.cdiv`、调用允许的输出初始化方法并启动本文件的 JIT kernel；kernel 内只能调用平台白名单中的 `tl` 运算。文件/网络/进程/线程、反射、dunder、动态执行、打印、断点、内联汇编、设备打印及非白名单 Torch/Python 调用都会在预检阶段被拒绝。
