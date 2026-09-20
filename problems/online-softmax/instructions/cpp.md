## C++17 接口

```cpp
#include <utility>
#include <vector>

std::pair<double, double> online_softmax_update(
    double running_max,
    double running_sum,
    const std::vector<double>& chunk);
```

返回 `{new_max, new_sum}`，使用 `double` 和标准库 `std::exp`。不要编写 `main`，不要使用 CUDA；输入向量只读。评测使用普通 C++17 编译器 `-std=c++17 -O3`。
