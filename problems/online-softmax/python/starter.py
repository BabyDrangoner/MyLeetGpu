import math


def online_softmax_update(
    running_max: float, running_sum: float, chunk: list[float]
) -> tuple[float, float]:
    # 可运行的在线基线：新最大值出现时，重新缩放之前的指数和。
    for value in chunk:
        new_max = max(running_max, value)
        running_sum = running_sum * math.exp(running_max - new_max) + math.exp(value - new_max)
        running_max = new_max
    return running_max, running_sum
