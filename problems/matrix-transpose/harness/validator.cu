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
    bool sequence;
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

bool same_float(float actual, float expected) {
    if (std::isnan(actual) || std::isnan(expected)) return false;
    if (std::isinf(actual) || std::isinf(expected)) return actual == expected;
    return actual == expected;
}

CaseResult run_case(const TestCase& test, cudaStream_t stream) {
    const std::size_t count =
        static_cast<std::size_t>(test.rows) * static_cast<std::size_t>(test.cols);
    const std::size_t bytes = count * sizeof(float);
    std::vector<float> input(count);
    std::vector<float> expected(count);
    std::vector<float> actual(count, 0.0F);
    std::mt19937 generator(test.seed);
    std::uniform_real_distribution<float> distribution(-50.0F, 50.0F);
    for (std::size_t i = 0; i < count; ++i) {
        input[i] = test.sequence
            ? static_cast<float>(static_cast<int>(i % 101U) - 50)
            : distribution(generator);
    }
    for (int row = 0; row < test.rows; ++row) {
        for (int col = 0; col < test.cols; ++col) {
            expected[static_cast<std::size_t>(col) * test.rows + row] =
                input[static_cast<std::size_t>(row) * test.cols + col];
        }
    }

    DeviceBuffer<float> device_input(count);
    DeviceBuffer<float> device_output(count);
    cuda_check(cudaMemcpyAsync(device_input.get(), input.data(), bytes,
                               cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemsetAsync(device_output.get(), 0xA5, bytes, stream));
    cudaGetLastError();
    solve(device_input.get(), device_output.get(), test.rows, test.cols, stream);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemcpyAsync(actual.data(), device_output.get(), bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));

    for (std::size_t i = 0; i < count; ++i) {
        if (!same_float(actual[i], expected[i])) {
            return {test.name, false,
                    test.internal ? "output mismatch"
                                  : "output mismatch at flattened index " +
                                        std::to_string(i)};
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
        {"sample_1", 1, 1, 2718U, true, false},
        {"sample_2", 3, 5, 2718U, true, false},
    };
    if (!public_only) {
        tests.push_back({"internal_case_1", 17, 33, 314159U, false, true});
        tests.push_back({"internal_case_2", 513, 1024, 314159U, false, true});
        tests.push_back({"internal_case_3", 1024, 513, 1618033U, false, true});
        tests.push_back({"internal_case_4", 2048, 2048, 1618033U, false, true});
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

