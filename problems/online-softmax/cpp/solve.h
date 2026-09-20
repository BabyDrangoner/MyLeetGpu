#pragma once

#include <utility>
#include <vector>

std::pair<double, double> online_softmax_update(
    double running_max, double running_sum, const std::vector<double>& chunk);
