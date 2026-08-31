## PyTorch Python 接口

提交源码只需导入 `torch` 并定义下列函数：

```python
def solve(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    ...
```

四个参数都是 GPU 0 上的连续 Tensor，平台调用时已经进入受控的 `torch.cuda.stream(stream)` 与 `torch.inference_mode()` 上下文。返回值必须是新的 CUDA `torch.float32` Tensor，形状为 `[batch, heads, query_length, head_dim]`。不要修改输入、移动 Tensor 到 CPU、显式同步设备或依赖上一次调用留下的状态。

提交使用版本化的受限 PyTorch 子集。模块级只允许精确的 `import torch`、不可变字面量常量和一个 `solve`；函数内可使用题目所需的白名单 Tensor 运算。文件、网络、进程、线程、反射、dunder、动态执行、打印、断点、全局状态修改、CUDA 计时/同步、`torch.compile`、`torch.jit`、`torch.ops` 以及 `torch.nn.functional.scaled_dot_product_attention` 都会在预检阶段被拒绝。
