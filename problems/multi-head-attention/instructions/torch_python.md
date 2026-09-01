## PyTorch Python 类接口

提交源码只需精确导入 `torch` 并定义下列普通类；不要继承 `torch.nn.Module`：

```python
class MultiHeadAttention:
    def __init__(
        self,
        numHeads: int,
        qWeight: torch.Tensor,
        kWeight: torch.Tensor,
        vWeight: torch.Tensor,
        outputWeight: torch.Tensor,
    ):
        ...

    def forward(self, X: torch.Tensor, isCasual: bool) -> torch.Tensor:
        ...
```

平台为每个用例构造一次实例。构造器参数 `numHeads` 是正整数，四个权重都是 GPU 0 上连续的 `[embed_dim, embed_dim]`、`torch.float32` Tensor。请原样保存它们，不得克隆、转换、修改或替换。`forward` 接收连续的 `X` 和 Python `bool`；返回值必须是新的 GPU 0 `torch.float32` Tensor，形状与 `X` 相同。

投影约定是 Tensor 右乘权重，即 `torch.matmul(X, self.qWeight)`。`isCasual=True` 时，可用当前设备上的位置索引比较构造包含对角线的下三角 mask；不要把 mask 或 Tensor 移到 CPU。

提交使用版本化的受限 PyTorch 子集。模块级只允许精确的 `import torch`、不可变字面量常量和这个类；类中只允许精确的 `__init__` 与 `forward`。可以使用题目所需的白名单 Tensor reshape、transpose、matmul、mask 和 softmax 操作。文件、网络、进程、线程、反射、动态执行、打印、额外方法、继承、decorator、状态修改、CUDA 计时/同步、`torch.compile`、`torch.jit`、`torch.ops` 以及现成的 scaled-dot-product attention 都会被拒绝。
