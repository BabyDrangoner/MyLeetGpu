#include "solve.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kAtol = 0.02F;
constexpr float kRtol = 0.00005F;

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
    int n;
    std::uint32_t seed;
    int pattern;
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

bool close_enough(float actual, float expected) {
    if (std::isnan(actual) || std::isnan(expected)) return false;
    if (std::isinf(actual) || std::isinf(expected)) return actual == expected;
    return std::fabs(actual - expected) <=
           kAtol + kRtol * std::fabs(expected);
}

CaseResult run_case(const TestCase& test, cudaStream_t stream) {
    const std::size_t count = static_cast<std::size_t>(test.n);
    const std::size_t bytes = count * sizeof(float);
    std::vector<float> input(count);
    std::mt19937 generator(test.seed);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    for (int i = 0; i < test.n; ++i) {
        float value = 0.0F;
        if (test.pattern == 0) {
            value = -0.75F;
        } else if (test.pattern == 1) {
            const float magnitude = static_cast<float>((i % 19) + 1) / 19.0F;
            value = (i % 2 == 0) ? magnitude : -magnitude;
        } else if (test.pattern == 2) {
            value = 1.0F;
        } else {
            value = distribution(generator);
        }
        input[static_cast<std::size_t>(i)] = value;
    }
    double reference = 0.0;
    for (const float value : input) reference += static_cast<double>(value);
    const float expected = static_cast<float>(reference);
    float actual = 0.0F;

    DeviceBuffer<float> device_input(count);
    DeviceBuffer<float> device_output(1);
    cuda_check(cudaMemcpyAsync(device_input.get(), input.data(), bytes,
                               cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemsetAsync(device_output.get(), 0xA5, sizeof(float), stream));
    cudaGetLastError();
    solve(device_input.get(), device_output.get(), test.n, stream);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemcpyAsync(&actual, device_output.get(), sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    if (!close_enough(actual, expected)) {
        return {test.name, false,
                test.internal ? "output mismatch" : "sum does not match reference"};
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
        {"sample_1", 1, 57721U, 0, false},
        {"sample_2", 1000, 57721U, 1, false},
    };
    if (!public_only) {
        tests.push_back({"internal_case_1", 31, 112358U, 1, true});
        tests.push_back({"internal_case_2", 4097, 112358U, 3, true});
        tests.push_back({"internal_case_3", 65537, 271828U, 2, true});
        tests.push_back({"internal_case_4", 1048576, 271828U, 3, true});
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

