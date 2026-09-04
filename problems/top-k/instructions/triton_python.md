## Triton Python 接口

提交源码应定义下列函数，并可在同一文件中定义一个或多个 `@triton.jit` kernel：

```python
def solve(
    input: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    rows: int,
    cols: int,
    k: int,
) -> None:
    ...
```

`input`、`values`、`indices` 是位于 GPU 0 的连续 Tensor，dtype 分别为 `torch.float32`、`torch.float32` 和 `torch.int32`。平台在调用前已经进入受控的 `torch.cuda.stream(stream)` 上下文。请写入已有的两个输出并返回 `None`；不要创建替代输出、移动 Tensor 到 CPU 或调用设备级同步。

当前运行时为 Triton 3.1，不提供 `tl.argsort`。可以结合 `tl.max` 和带 `tie_break_left=True` 的 `tl.argmax` 逐次选择最大元素，并在候选向量中屏蔽已选位置。题目保证输入不小于 `-10000`，因此可使用更小的有限值作为屏蔽哨兵。

提交使用受限 Triton 子集：模块级只允许精确的 `import torch`、`import triton`、`import triton.language as tl`、字面量常量、`@triton.jit` 函数和一个无装饰器的 `solve`。`solve` 是直线式 launcher，只能计算 launch 参数、调用平台允许的 Triton host helper 并启动本文件的 JIT kernel；kernel 内只能调用平台白名单中的 `tl` 运算。文件/网络/进程/线程、反射、dunder、动态执行、打印、断点、内联汇编、设备打印及非白名单 Torch/Python 调用都会在预检阶段被拒绝。
