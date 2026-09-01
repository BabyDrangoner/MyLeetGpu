#include "solve.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kAbsoluteTolerance = 3.0e-5F;
constexpr float kRelativeTolerance = 3.0e-4F;
constexpr double kRowSumTolerance = 1.0e-3;

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
    std::uint32_t seed;
    std::string pattern;
    bool internal;
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
                } else out << static_cast<char>(ch);
        }
    }
    out << '"';
    return out.str();
}

std::vector<float> make_input(const TestCase& test) {
    const std::size_t count =
        static_cast<std::size_t>(test.rows) * static_cast<std::size_t>(test.cols);
    std::vector<float> input(count);
    std::mt19937 generator(test.seed);
    std::uniform_real_distribution<float> distribution(-100.0F, 100.0F);
    for (int row = 0; row < test.rows; ++row) {
        for (int col = 0; col < test.cols; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * test.cols + col;
            if (test.pattern == "singleton") {
                input[index] = 37.0F;
            } else if (test.pattern == "sequence") {
                input[index] = 0.5F * static_cast<float>(
                    static_cast<int>((index * 17U + static_cast<std::size_t>(row) * 3U) % 29U)
                    - 14);
            } else if (test.pattern == "random") {
                input[index] = distribution(generator);
            } else if (test.pattern == "repeated") {
                input[index] = static_cast<float>((row % 17) - 8) * 12.5F;
            } else {
                switch (col % 6) {
                    case 0: input[index] = 100.0F - 0.125F * (row % 5); break;
                    case 1: input[index] = -100.0F; break;
                    case 2: input[index] = 80.0F; break;
                    case 3: input[index] = -80.0F; break;
                    case 4: input[index] = 0.0F; break;
                    default: input[index] = 99.0F - 0.25F * (row % 3); break;
                }
            }
        }
    }
    return input;
}

std::vector<float> reference_softmax(const std::vector<float>& input,
                                     int rows,
                                     int cols) {
    std::vector<float> expected(input.size());
    for (int row = 0; row < rows; ++row) {
        const std::size_t row_offset = static_cast<std::size_t>(row) * cols;
        double row_max = -std::numeric_limits<double>::infinity();
        for (int col = 0; col < cols; ++col) {
            row_max = std::max(row_max, static_cast<double>(input[row_offset + col]));
        }
        double denominator = 0.0;
        for (int col = 0; col < cols; ++col) {
            denominator += std::exp(static_cast<double>(input[row_offset + col]) - row_max);
        }
        for (int col = 0; col < cols; ++col) {
            expected[row_offset + col] = static_cast<float>(
                std::exp(static_cast<double>(input[row_offset + col]) - row_max) /
                denominator);
        }
    }
    return expected;
}

bool close_float(float actual, float expected) {
    if (!std::isfinite(actual)) return false;
    const float threshold =
        kAbsoluteTolerance + kRelativeTolerance * std::fabs(expected);
    return std::fabs(actual - expected) <= threshold;
}

CaseResult run_case(const TestCase& test, cudaStream_t stream) {
    const std::size_t count =
        static_cast<std::size_t>(test.rows) * static_cast<std::size_t>(test.cols);
    const std::size_t bytes = count * sizeof(float);
    const std::vector<float> input = make_input(test);
    std::vector<float> observed_input(count);
    const std::vector<float> expected = reference_softmax(input, test.rows, test.cols);
    std::vector<float> actual(count, 0.0F);

    DeviceBuffer<float> device_input(count);
    DeviceBuffer<float> device_output(count);
    cuda_check(cudaMemcpyAsync(device_input.get(), input.data(), bytes,
                               cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemsetAsync(device_output.get(), 0xFF, bytes, stream));
    cudaGetLastError();
    solve(device_input.get(), device_output.get(), test.rows, test.cols, stream);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemcpyAsync(actual.data(), device_output.get(), bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(observed_input.data(), device_input.get(), bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));

    if (std::memcmp(observed_input.data(), input.data(), bytes) != 0) {
        return {test.name, false,
                test.internal ? "input modified" : "input must remain unchanged"};
    }

    for (std::size_t index = 0; index < count; ++index) {
        if (!close_float(actual[index], expected[index])) {
            if (test.internal) return {test.name, false, "output mismatch"};
            const int row = static_cast<int>(index / static_cast<std::size_t>(test.cols));
            const int col = static_cast<int>(index % static_cast<std::size_t>(test.cols));
            return {test.name, false,
                    "output mismatch at row " + std::to_string(row) +
                        ", col " + std::to_string(col)};
        }
    }
    for (int row = 0; row < test.rows; ++row) {
        double row_sum = 0.0;
        for (int col = 0; col < test.cols; ++col) {
            const float value = actual[static_cast<std::size_t>(row) * test.cols + col];
            if (value < 0.0F) {
                return {test.name, false,
                        test.internal ? "output mismatch"
                                      : "output contains a negative probability"};
            }
            row_sum += static_cast<double>(value);
        }
        if (std::fabs(row_sum - 1.0) > kRowSumTolerance) {
            return {test.name, false,
                    test.internal ? "output mismatch"
                                  : "row probabilities do not sum to one"};
        }
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
        {"sample_1", 1, 1, 4242U, "singleton", false},
        {"sample_2", 3, 5, 4242U, "sequence", false},
    };
    if (!public_only) {
        tests.push_back({"internal_case_1", 7, 31, 424242U, "random", true});
        tests.push_back({"internal_case_2", 257, 3, 424242U, "extreme", true});
        tests.push_back({"internal_case_3", 19, 257, 8675309U, "repeated", true});
        tests.push_back({"internal_case_4", 64, 4096, 8675309U, "extreme", true});
        tests.push_back({"internal_case_5", 1024, 513, 424242U, "random", true});
        tests.push_back({"internal_case_6", 65536, 1, 8675309U, "singleton", true});
        tests.push_back({"internal_case_7", 4096, 4095, 8675309U, "random", true});
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
