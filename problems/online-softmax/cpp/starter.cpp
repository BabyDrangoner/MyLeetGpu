#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

std::pair<double, double> online_softmax_update(
    double running_max, double running_sum, const std::vector<double>& chunk) {
    // 可运行的在线基线：新最大值出现时，重新缩放之前的指数和。
    for (const double value : chunk) {
        const double new_max = std::max(running_max, value);
        running_sum = running_sum * std::exp(running_max - new_max)
                    + std::exp(value - new_max);
        running_max = new_max;
    }
    return {running_max, running_sum};
}
