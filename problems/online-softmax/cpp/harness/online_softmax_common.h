#pragma once

#include "solve.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace online_softmax_harness {

using State = std::pair<double, double>;
using Chunks = std::vector<std::vector<double>>;
constexpr double kTolerance = 1.0e-9;
constexpr int kWarmup = 2;
constexpr int kIterations = 7;

inline State empty_state() { return {-std::numeric_limits<double>::infinity(), 0.0}; }

inline std::vector<double> random_values(std::size_t count, std::uint32_t seed = 424242) {
    std::vector<double> values;
    values.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        seed = 1664525U * seed + 1013904223U;
        values.push_back((static_cast<int>(seed % 2000001U) - 1000000) / 100.0);
    }
    return values;
}

inline Chunks partition(const std::vector<double>& values, std::size_t chunk_size = 0) {
    Chunks chunks(1);
    std::uint32_t seed = 271828;
    for (std::size_t position = 0; position < values.size();) {
        seed = 1664525U * seed + 1013904223U;
        const std::size_t length = chunk_size ? chunk_size : 1 + seed % 509U;
        const std::size_t end = std::min(position + length, values.size());
        chunks.emplace_back(values.begin() + position, values.begin() + end);
        position = end;
        if (chunks.size() % 5 == 0) chunks.emplace_back();
    }
    chunks.emplace_back();
    return chunks;
}

struct TestCase {
    std::string name;
    std::vector<double> history;
    Chunks chunks;
};

inline std::vector<TestCase> cases(bool full) {
    std::vector<TestCase> tests = {
        {"sample_empty", {}, {{}, {}}},
        {"sample_singleton", {}, {{}, {3.5}, {}}},
        {"sample_chunked", {}, {{1.0, 2.0}, {}, {3.0}, {0.0, -1.0}}},
    };
    if (!full) return tests;
    tests.push_back({"internal_equal", {}, partition(std::vector<double>(257, 17.0))});
    std::vector<double> increasing, decreasing, extremes, close_maxima;
    for (int i = 0; i < 257; ++i) {
        increasing.push_back(i - 128.0);
        decreasing.push_back(10000.0 - 30.0 * i);
    }
    for (int i = 0; i < 511; ++i) {
        extremes.push_back(i % 3 == 0 ? -10000.0 : (i % 3 == 1 ? 10000.0 : 0.0));
    }
    for (int i = 0; i < 1025; ++i) close_maxima.push_back(9999.5 + (i % 7) / 14.0);
    tests.push_back({"internal_increasing", {}, partition(increasing, 1)});
    tests.push_back({"internal_extremes", {}, partition(extremes)});
    tests.push_back({"internal_decreasing", {}, partition(decreasing)});
    tests.push_back({"internal_close_maxima", {}, partition(close_maxima)});
    std::vector<double> carried_values;
    for (int i = 0; i < 97; ++i) carried_values.push_back(5.0 + (i % 11) / 3.0);
    tests.push_back({"internal_carried_state", {7.0, 9.0, -4.0, 9.0}, partition(carried_values)});
    tests.push_back({"internal_random_chunks", {}, partition(random_values(4099))});
    tests.push_back({"internal_long", {}, partition(random_values(131072))});
    return tests;
}

// Independent two-pass reference; long double accumulation avoids copying the starter's recurrence.
inline State reference(const std::vector<double>& values) {
    if (values.empty()) return empty_state();
    const double maximum = *std::max_element(values.begin(), values.end());
    long double denominator = 0.0L;
    for (const double value : values) {
        denominator += std::exp(static_cast<long double>(value) - maximum);
    }
    return {maximum, static_cast<double>(denominator)};
}

inline State merge_reference(State state, const std::vector<double>& chunk) {
    if (chunk.empty()) return state;
    const auto incoming = reference(chunk);
    const double maximum = std::max(state.first, incoming.first);
    const long double denominator =
        state.second * std::exp(static_cast<long double>(state.first) - maximum)
        + incoming.second * std::exp(static_cast<long double>(incoming.first) - maximum);
    return {maximum, static_cast<double>(denominator)};
}

inline bool close(double actual, double expected) {
    return std::abs(actual - expected) <= kTolerance + kTolerance * std::abs(expected);
}

inline void check_state(State actual, State expected) {
    if (expected == empty_state()) {
        if (actual != expected) throw std::logic_error("empty state must remain (-inf, 0)");
        return;
    }
    if (!std::isfinite(actual.first) || !std::isfinite(actual.second) || actual.second <= 0.0
        || !close(actual.first, expected.first) || !close(actual.second, expected.second)) {
        throw std::logic_error("incorrect running maximum or rescaled exponential sum");
    }
}

inline void check_probabilities(const std::vector<double>& values, State state) {
    const auto expected = reference(values);
    check_state(state, expected);
    if (values.empty()) return;
    long double total = 0.0L;
    for (const double value : values) {
        const double probability = std::exp(value - state.first) / state.second;
        const double expected_probability = std::exp(value - expected.first) / expected.second;
        if (!close(probability, expected_probability)) {
            throw std::logic_error("final probability differs from stable softmax");
        }
        total += probability;
    }
    if (!close(static_cast<double>(total), 1.0)) {
        throw std::logic_error("final probabilities are not normalized");
    }
}

inline void check_stream(const TestCase& test) {
    State state = reference(test.history);
    State expected = state;
    std::vector<double> values = test.history;
    for (const auto& input : test.chunks) {
        auto chunk = input;
        expected = merge_reference(expected, input);
        state = online_softmax_update(state.first, state.second, chunk);
        if (chunk != input) throw std::logic_error("chunk must not be modified");
        check_state(state, expected);
        values.insert(values.end(), input.begin(), input.end());
    }
    check_probabilities(values, state);
}

inline State run_stream(const Chunks& chunks) {
    State state = empty_state();
    for (const auto& chunk : chunks) {
        state = online_softmax_update(state.first, state.second, chunk);
    }
    return state;
}

}  // namespace online_softmax_harness
