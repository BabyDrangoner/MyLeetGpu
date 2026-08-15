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
#include <utility>
#include <vector>

namespace {

constexpr float kAtol = 1.0e-6F;
constexpr float kRtol = 1.0e-6F;

void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

class Stream {
public:
    Stream() { cuda_check(cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking)); }
    ~Stream() {
        if (value_ != nullptr) {
            cudaStreamDestroy(value_);
        }
    }
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
    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }
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
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    out << '"';
    return out.str();
}

bool close_enough(float actual, float expected) {
    if (std::isnan(actual) || std::isnan(expected)) {
        return false;
    }
    if (std::isinf(actual) || std::isinf(expected)) {
        return actual == expected;
    }
    return std::fabs(actual - expected) <=
           kAtol + kRtol * std::fabs(expected);
}

CaseResult run_case(const TestCase& test, cudaStream_t stream) {
    std::vector<float> a(static_cast<std::size_t>(test.n));
    std::vector<float> b(static_cast<std::size_t>(test.n));
    std::vector<float> expected(static_cast<std::size_t>(test.n));
    std::vector<float> actual(static_cast<std::size_t>(test.n), 0.0F);

    std::mt19937 generator(test.seed);
    std::uniform_real_distribution<float> distribution(-100.0F, 100.0F);
    for (int i = 0; i < test.n; ++i) {
        if (test.pattern == 0) {
            a[static_cast<std::size_t>(i)] = -100.0F;
            b[static_cast<std::size_t>(i)] = 100.0F;
        } else if (test.pattern == 1) {
            const float magnitude = static_cast<float>((i % 23) + 1);
            a[static_cast<std::size_t>(i)] = (i % 2 == 0) ? magnitude : -magnitude;
            b[static_cast<std::size_t>(i)] = (i % 3 == 0) ? -magnitude : magnitude * 0.5F;
        } else {
            a[static_cast<std::size_t>(i)] = distribution(generator);
            b[static_cast<std::size_t>(i)] = distribution(generator);
        }
        expected[static_cast<std::size_t>(i)] =
            a[static_cast<std::size_t>(i)] + b[static_cast<std::size_t>(i)];
    }

    DeviceBuffer<float> device_a(a.size());
    DeviceBuffer<float> device_b(b.size());
    DeviceBuffer<float> device_output(actual.size());
    const std::size_t bytes = a.size() * sizeof(float);
    cuda_check(cudaMemcpyAsync(device_a.get(), a.data(), bytes,
                               cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyAsync(device_b.get(), b.data(), bytes,
                               cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemsetAsync(device_output.get(), 0xA5, bytes, stream));
    cudaGetLastError();
    solve(device_a.get(), device_b.get(), device_output.get(), test.n, stream);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemcpyAsync(actual.data(), device_output.get(), bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));

    for (int i = 0; i < test.n; ++i) {
        if (!close_enough(actual[static_cast<std::size_t>(i)],
                          expected[static_cast<std::size_t>(i)])) {
            const std::string message = test.internal
                ? "output mismatch"
                : "output mismatch at index " + std::to_string(i);
            return {test.name, false, message};
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
        if (i != 0) {
            out << ',';
        }
        const CaseResult& result = results[i];
        passed += result.passed ? 1U : 0U;
        out << "{\"name\":" << json_string(result.name)
            << ",\"passed\":" << (result.passed ? "true" : "false");
        if (!result.message.empty()) {
            out << ",\"message\":" << json_string(result.message);
        }
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
        if (mode == "public") {
            public_only = true;
        } else if (mode != "full") {
            print_result("runtime_error", {{"configuration", false, "invalid mode"}});
            return 2;
        }
    } else if (argc != 1) {
        print_result("runtime_error", {{"configuration", false, "invalid arguments"}});
        return 2;
    }

    std::vector<TestCase> tests = {
        {"sample_1", 1, 1701U, 0, false},
        {"sample_2", 257, 1701U, 2, false},
    };
    if (!public_only) {
        tests.push_back({"internal_case_1", 31, 481516U, 1, true});
        tests.push_back({"internal_case_2", 4097, 481516U, 2, true});
        tests.push_back({"internal_case_3", 65537, 8675309U, 2, true});
        tests.push_back({"internal_case_4", 1048576, 8675309U, 2, true});
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
    const std::string status = runtime_error
        ? "runtime_error"
        : (all_passed ? "passed" : "wrong_answer");
    print_result(status, results);
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

