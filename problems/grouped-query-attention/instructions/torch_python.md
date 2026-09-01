## PyTorch Python 接口

提交源码只需精确导入 `torch` 并定义下列普通类：

```python
class GroupedQueryAttention:
    def __init__(
        self,
        numQueryHeads: int,
        numKeyValueHeads: int,
        qWeight: torch.Tensor,
        kWeight: torch.Tensor,
        vWeight: torch.Tensor,
        outputWeight: torch.Tensor,
    ):
        ...

    def forward(self, X: torch.Tensor, isCasual: bool) -> torch.Tensor:
        ...
```

不要继承 `torch.nn.Module`，也不要定义其他函数、方法或类。平台会为每个用例构造一次实例，并在受控的 `torch.cuda.stream(stream)` 与 `torch.inference_mode()` 上下文中直接调用 `forward`。

`forward` 的业务输入只有 `X` 和 `isCasual`。四个权重均按题面给出的形状直接右乘：`X @ qWeight`、`X @ kWeight`、`X @ vWeight`，合并 heads 后再右乘 `outputWeight`。构造器中的 Tensor 和 `X` 均为只读状态；返回值必须是新的 CUDA `torch.float32` Tensor，形状与 `X` 相同。

当 `isCasual` 为 `True` 时，需要动态构造包含主对角线的下三角 mask。可以使用 `torch.arange`、广播比较、`masked_fill` 与 `float("-inf")` 完成；为 `False` 时不要应用 mask。

提交使用版本化的受限 PyTorch 子集。模块级只允许精确的 `import torch` 和一个满足接口的 `GroupedQueryAttention` 类；类只允许指定的构造器和 `forward`。题目需要的 reshape、transpose、contiguous、matmul、repeat-interleave、arange、比较、mask 和 softmax 已加入白名单。文件、网络、进程、线程、反射、任意 dunder 访问、动态执行、打印、断点、全局状态修改、forward 状态写入、CUDA 计时/同步、`torch.compile`、`torch.jit`、`torch.ops` 以及现成 scaled-dot-product attention 都会在预检阶段被拒绝。
