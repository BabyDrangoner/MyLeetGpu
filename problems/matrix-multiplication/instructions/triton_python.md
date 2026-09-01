## Triton Python 接口

提交源码应定义下列函数，并可在同一文件中定义一个或多个 `@triton.jit` kernel：

```python
def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    m: int,
    k: int,
    n: int,
) -> None:
    ...
```

`a`、`b`、`c` 是位于 GPU 0、连续存放的一维 `torch.float32` Tensor；请按题目给出的行主序索引解释其形状。平台在调用前已经进入受控的 `torch.cuda.stream(stream)` 上下文。请写入已有的 `c`，返回 `None`；不要创建替代输出、移动 Tensor 到 CPU 或调用设备级同步。

提交使用受限 Triton 子集：模块级只允许精确的 `import torch`、`import triton`、`import triton.language as tl`、字面量常量、`@triton.jit` 函数和一个无装饰器的 `solve`。`solve` 是直线式 launcher，只能计算 launch 参数、调用 `triton.cdiv` 并启动本文件的 JIT kernel；kernel 内只能调用平台白名单中的 `tl` 运算（包括 `tl.dot`、block pointer 构造与布局提示）。文件/网络/进程/线程、反射、dunder、动态执行、打印、断点、内联汇编、设备打印及非白名单 Torch/Python 调用都会在预检阶段被拒绝。
