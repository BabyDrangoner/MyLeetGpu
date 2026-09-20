## Python 标准库接口

```python
def online_softmax_update(
    running_max: float, running_sum: float, chunk: list[float]
) -> tuple[float, float]:
    ...
```

返回二元 `tuple` `(new_max, new_sum)`；`float` 为双精度，指数可用标准库 `math.exp`。不要修改输入列表，不使用 PyTorch、NumPy 或任何 GPU 依赖。不需要编写输入输出或入口函数。
