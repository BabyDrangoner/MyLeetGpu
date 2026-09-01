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

std::vector<float> make_input(const TestCase& test) {
    std::vector<float> input(static_cast<std::size_t>(test.n));
    std::mt19937 generator(test.seed);
    if (test.pattern == 0) {
        std::fill(input.begin(), input.end(), std::numeric_limits<float>::lowest());
    } else if (test.pattern == 1) {
        std::uniform_real_distribution<float> distribution(-1000.0F, -0.001F);
        for (float& value : input) value = distribution(generator);
    } else if (test.pattern == 2) {
        std::uniform_real_distribution<float> distribution(-1000.0F, 1000.0F);
        for (float& value : input) value = distribution(generator);
        input.front() = std::numeric_limits<float>::lowest();
        input[static_cast<std::size_t>(test.n / 2)] =
            std::numeric_limits<float>::max();
    } else if (test.pattern == 3) {
        for (int i = 0; i < test.n; ++i) {
            const float magnitude = static_cast<float>((i % 257) + 1);
            input[static_cast<std::size_t>(i)] =
                (i % 2 == 0) ? -magnitude : magnitude;
        }
    } else if (test.pattern == 4) {
        std::fill(input.begin(), input.end(), -0.0F);
    } else {
        std::uniform_real_distribution<float> distribution(-1000000.0F, 1000000.0F);
        for (float& value : input) value = distribution(generator);
    }
    return input;
}

CaseResult run_case(const TestCase& test, cudaStream_t stream) {
    std::vector<float> input = make_input(test);
    std::vector<float> observed_input(input.size());
    const float expected = *std::max_element(input.begin(), input.end());
    const std::size_t bytes = input.size() * sizeof(float);
    float actual = 0.0F;

    DeviceBuffer<float> device_input(input.size());
    DeviceBuffer<float> device_output(1);
    const float output_poison = std::numeric_limits<float>::max();
    cuda_check(cudaMemcpyAsync(device_input.get(), input.data(), bytes,
                               cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyAsync(device_output.get(), &output_poison, sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    cudaGetLastError();
    solve(device_input.get(), device_output.get(), test.n, stream);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemcpyAsync(&actual, device_output.get(), sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(observed_input.data(), device_input.get(), bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    if (std::memcmp(observed_input.data(), input.data(), bytes) != 0) {
        return {test.name, false,
                test.internal ? "input modified" : "input must remain unchanged"};
    }
    if (std::isnan(actual) || actual != expected) {
        return {test.name, false,
                test.internal ? "output mismatch" : "maximum does not match reference"};
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
        {"sample_1", 1, 91453U, 0, false},
        {"sample_2", 1000, 91453U, 1, false},
    };
    if (!public_only) {
        tests.push_back({"internal_case_1", 31, 161803U, 2, true});
        tests.push_back({"internal_case_2", 4097, 161803U, 1, true});
        tests.push_back({"internal_case_3", 65537, 141421U, 3, true});
        tests.push_back({"internal_case_4", 513, 141421U, 4, true});
        tests.push_back({"internal_case_5", 1048576, 141421U, 5, true});
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
