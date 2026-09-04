#include "solve.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kBoundaryMargin = 1.0e-5;

void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

class Stream {
public:
    Stream() { cuda_check(cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking)); }
    ~Stream() { if (value_ != nullptr) cudaStreamDestroy(value_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return value_; }
private:
    cudaStream_t value_ = nullptr;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data_ != nullptr) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

struct TestCase {
    std::string name;
    int rows;
    int cols;
    float p;
    std::string pattern;
    bool internal;
};

struct Reference {
    std::vector<float> output;
    std::vector<int> counts;
};

struct CaseResult {
    std::string name;
    bool passed;
    std::string message;
};

std::string json_string(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20U) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch) << std::dec;
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    out << '"';
    return out.str();
}

double weight_for_rank(int rank, int cols, const std::string& pattern) {
    const double value = static_cast<double>(rank);
    if (pattern == "dominant" && rank == cols) {
        return 4.0 * static_cast<double>(cols) * static_cast<double>(cols);
    }
    if (pattern == "quadratic") return value * value;
    if (pattern == "near_uniform") return static_cast<double>(cols) + value;
    return value;
}

std::vector<float> make_probabilities(const TestCase& test) {
    const std::size_t count =
        static_cast<std::size_t>(test.rows) * static_cast<std::size_t>(test.cols);
    if (test.pattern == "singleton") return std::vector<float>(count, 1.0F);

    std::vector<float> probabilities(count);
    for (int row = 0; row < test.rows; ++row) {
        const std::size_t row_offset = static_cast<std::size_t>(row) * test.cols;
        const int shift = (row * 17) % test.cols;
        double denominator = 0.0;
        for (int rank = 1; rank <= test.cols; ++rank) {
            denominator += weight_for_rank(rank, test.cols, test.pattern);
        }

        int maximum_col = 0;
        for (int col = 0; col < test.cols; ++col) {
            const int rank = ((col + shift) % test.cols) + 1;
            probabilities[row_offset + col] = static_cast<float>(
                weight_for_rank(rank, test.cols, test.pattern) / denominator);
            if (rank == test.cols) maximum_col = col;
        }

        double other_sum = 0.0;
        for (int col = 0; col < test.cols; ++col) {
            if (col != maximum_col) {
                other_sum += static_cast<double>(probabilities[row_offset + col]);
            }
        }
        probabilities[row_offset + maximum_col] =
            static_cast<float>(1.0 - other_sum);
    }
    return probabilities;
}

Reference make_reference(const std::vector<float>& probabilities,
                         const TestCase& test) {
    Reference reference{
        std::vector<float>(probabilities.size(), 0.0F),
        std::vector<int>(static_cast<std::size_t>(test.rows), test.cols),
    };
    std::vector<float> ordered(static_cast<std::size_t>(test.cols));
    for (int row = 0; row < test.rows; ++row) {
        const std::size_t row_offset = static_cast<std::size_t>(row) * test.cols;
        std::copy_n(probabilities.begin() + static_cast<std::ptrdiff_t>(row_offset),
                    test.cols, ordered.begin());
        std::sort(ordered.begin(), ordered.end(), std::greater<float>());

        int retained = test.cols;
        if (test.p < 1.0F) {
            double cumulative = 0.0;
            double closest_boundary = std::numeric_limits<double>::infinity();
            bool found = false;
            for (int rank = 0; rank < test.cols; ++rank) {
                cumulative += static_cast<double>(ordered[rank]);
                closest_boundary = std::min(
                    closest_boundary,
                    std::fabs(cumulative - static_cast<double>(test.p)));
                if (!found && cumulative >= static_cast<double>(test.p)) {
                    retained = rank + 1;
                    found = true;
                }
            }
            if (!found || closest_boundary <= kBoundaryMargin) {
                throw std::runtime_error("invalid test boundary margin");
            }
        }
        reference.counts[static_cast<std::size_t>(row)] = retained;
        std::copy_n(ordered.begin(), retained,
                    reference.output.begin() + static_cast<std::ptrdiff_t>(row_offset));
    }
    return reference;
}

std::uint32_t float_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::string validate_observation(const std::vector<float>& actual,
                                 const std::vector<int>& actual_counts,
                                 const Reference& reference,
                                 const TestCase& test) {
    for (int row = 0; row < test.rows; ++row) {
        const int retained = actual_counts[static_cast<std::size_t>(row)];
        if (retained < 1 || retained > test.cols ||
            retained != reference.counts[static_cast<std::size_t>(row)]) {
            return test.internal ? "count mismatch"
                                 : "retained count mismatch at row " +
                                       std::to_string(row);
        }
        if (test.p >= 1.0F && retained != test.cols) {
            return test.internal ? "count mismatch"
                                 : "p == 1 must retain the complete row";
        }

        const std::size_t row_offset = static_cast<std::size_t>(row) * test.cols;
        double cumulative = 0.0;
        double previous = 0.0;
        for (int rank = 0; rank < test.cols; ++rank) {
            const std::size_t index = row_offset + static_cast<std::size_t>(rank);
            const float value = actual[index];
            if (float_bits(value) != float_bits(reference.output[index])) {
                return test.internal ? "output mismatch"
                                     : "output mismatch at row " +
                                           std::to_string(row) + ", rank " +
                                           std::to_string(rank);
            }
            if (!std::isfinite(value) || value < 0.0F) {
                return test.internal ? "output mismatch"
                                     : "output must contain finite nonnegative values";
            }
            if (rank < retained) {
                if (rank > 0 && actual[index - 1] < value) {
                    return test.internal ? "output mismatch"
                                         : "retained prefix is not sorted descending";
                }
                previous = cumulative;
                cumulative += static_cast<double>(value);
            } else if (float_bits(value) != 0U) {
                return test.internal ? "output mismatch"
                                     : "filtered tail must be exact positive zero";
            }
        }
        if (test.p < 1.0F &&
            (cumulative < static_cast<double>(test.p) ||
             previous >= static_cast<double>(test.p))) {
            return test.internal ? "prefix semantics mismatch"
                                 : "output is not the shortest prefix reaching p";
        }
    }
    return "";
}

CaseResult run_case(const TestCase& test, cudaStream_t stream) {
    const std::size_t count =
        static_cast<std::size_t>(test.rows) * static_cast<std::size_t>(test.cols);
    const std::size_t bytes = count * sizeof(float);
    const std::size_t count_bytes = static_cast<std::size_t>(test.rows) * sizeof(int);
    const std::vector<float> probabilities = make_probabilities(test);
    const Reference reference = make_reference(probabilities, test);
    std::vector<float> first_output(count);
    std::vector<float> second_output(count);
    std::vector<float> observed_input(count);
    std::vector<int> first_counts(static_cast<std::size_t>(test.rows));
    std::vector<int> second_counts(static_cast<std::size_t>(test.rows));

    DeviceBuffer<float> device_input(count);
    DeviceBuffer<float> device_output(count);
    DeviceBuffer<int> device_counts(static_cast<std::size_t>(test.rows));
    cuda_check(cudaMemcpyAsync(device_input.get(), probabilities.data(), bytes,
                               cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemsetAsync(device_output.get(), 0xFF, bytes, stream));
    cuda_check(cudaMemsetAsync(device_counts.get(), 0xA5, count_bytes, stream));
    cudaGetLastError();
    solve(device_input.get(), device_output.get(), device_counts.get(),
          test.rows, test.cols, test.p, stream);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemcpyAsync(first_output.data(), device_output.get(), bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(first_counts.data(), device_counts.get(), count_bytes,
                               cudaMemcpyDeviceToHost, stream));

    cuda_check(cudaMemsetAsync(device_output.get(), 0x7F, bytes, stream));
    cuda_check(cudaMemsetAsync(device_counts.get(), 0x5A, count_bytes, stream));
    cudaGetLastError();
    solve(device_input.get(), device_output.get(), device_counts.get(),
          test.rows, test.cols, test.p, stream);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemcpyAsync(second_output.data(), device_output.get(), bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(second_counts.data(), device_counts.get(), count_bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(observed_input.data(), device_input.get(), bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));

    if (std::memcmp(observed_input.data(), probabilities.data(), bytes) != 0) {
        return {test.name, false,
                test.internal ? "input modified" : "input must remain unchanged"};
    }
    const std::string first_error =
        validate_observation(first_output, first_counts, reference, test);
    if (!first_error.empty()) return {test.name, false, first_error};
    const std::string second_error =
        validate_observation(second_output, second_counts, reference, test);
    if (!second_error.empty()) return {test.name, false, second_error};
    if (std::memcmp(first_output.data(), second_output.data(), bytes) != 0 ||
        first_counts != second_counts) {
        return {test.name, false,
                test.internal ? "call independence mismatch"
                              : "repeated calls produced different results"};
    }
    return {test.name, true, ""};
}

void print_result(const std::string& status,
                  const std::vector<CaseResult>& results) {
    std::size_t passed = 0;
    std::ostringstream out;
    out << "MYLEETGPU_RESULT={\"status\":" << json_string(status)
        << ",\"cases\":[";
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (i != 0) out << ',';
        const CaseResult& result = results[i];
        passed += result.passed ? 1U : 0U;
        out << "{\"name\":" << json_string(result.name)
            << ",\"passed\":" << (result.passed ? "true" : "false");
        if (!result.message.empty()) out << ",\"message\":" << json_string(result.message);
        out << '}';
    }
    out << "],\"summary\":{\"total\":" << results.size()
        << ",\"passed\":" << passed
        << ",\"failed\":" << (results.size() - passed) << "}}";
    std::cout << out.str() << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    bool public_only = false;
    if (argc == 3 && std::string(argv[1]) == "--mode") {
        const std::string mode(argv[2]);
        if (mode == "public") public_only = true;
        else if (mode != "full") {
            print_result("runtime_error", {{"configuration", false, "invalid mode"}});
            return 2;
        }
    } else if (argc != 1) {
        print_result("runtime_error", {{"configuration", false, "invalid arguments"}});
        return 2;
    }

    std::vector<TestCase> tests = {
        {"sample_singleton", 1, 1, 0.5F, "singleton", false},
        {"sample_non_power_of_two", 3, 5, 0.7F, "ramp", false},
    };
    if (!public_only) {
        tests.push_back({"internal_case_1", 7, 31, 0.55F, "ramp", true});
        tests.push_back({"internal_case_2", 257, 3, 0.8F, "dominant", true});
        tests.push_back({"internal_case_3", 19, 257, 0.02F, "quadratic", true});
        tests.push_back({"internal_case_4", 37, 513, 1.0F, "near_uniform", true});
        tests.push_back({"internal_case_5", 64, 1024, 0.9F, "quadratic", true});
        tests.push_back({"internal_case_6", 4096, 1024, 0.95F, "ramp", true});
        tests.push_back({"internal_case_7", 65536, 1, 0.5F, "singleton", true});
    }

    std::vector<CaseResult> results;
    bool runtime_error = false;
    try {
        Stream stream;
        for (const TestCase& test : tests) {
            try {
                results.push_back(run_case(test, stream.get()));
            } catch (const std::exception& error) {
                results.push_back({test.name, false,
                                   std::string("CUDA execution failed: ") + error.what()});
                runtime_error = true;
                break;
            }
        }
    } catch (const std::exception& error) {
        results.push_back({"setup", false,
                           std::string("CUDA setup failed: ") + error.what()});
        runtime_error = true;
    }
    const bool all_passed =
        !runtime_error && results.size() == tests.size() &&
        std::all_of(results.begin(), results.end(),
                    [](const CaseResult& result) { return result.passed; });
    print_result(runtime_error ? "runtime_error"
                               : (all_passed ? "passed" : "wrong_answer"),
                 results);
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
