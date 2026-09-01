#include "solve.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kAbsoluteTolerance = 0.03;
constexpr double kRelativeTolerance = 0.005;

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
    int m;
    int k;
    int n;
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

bool close_enough(float actual, float expected) {
    if (std::isnan(actual) || std::isnan(expected)) return false;
    if (std::isinf(actual) || std::isinf(expected)) return actual == expected;
    const double difference = std::abs(static_cast<double>(actual) - expected);
    return difference <=
           kAbsoluteTolerance + kRelativeTolerance * std::abs(static_cast<double>(expected));
}

void make_inputs(const TestCase& test,
                 std::vector<float>& a,
                 std::vector<float>& b) {
    if (test.sequence) {
        for (std::size_t index = 0; index < a.size(); ++index) {
            a[index] = static_cast<float>(static_cast<int>(index % 7U) - 3) * 0.25F;
        }
        for (std::size_t index = 0; index < b.size(); ++index) {
            b[index] = static_cast<float>(static_cast<int>((index * 3U) % 11U) - 5) * 0.2F;
        }
        return;
    }
    std::mt19937 generator(test.seed);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    for (float& value : a) value = distribution(generator);
    for (float& value : b) value = distribution(generator);
}

std::vector<float> reference_multiply(const std::vector<float>& a,
                                      const std::vector<float>& b,
                                      int m,
                                      int k,
                                      int n) {
    std::vector<float> expected(static_cast<std::size_t>(m) * n);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double accumulator = 0.0;
            for (int inner = 0; inner < k; ++inner) {
                accumulator += static_cast<double>(a[static_cast<std::size_t>(row) * k + inner]) *
                               static_cast<double>(b[static_cast<std::size_t>(inner) * n + col]);
            }
            expected[static_cast<std::size_t>(row) * n + col] =
                static_cast<float>(accumulator);
        }
    }
    return expected;
}

CaseResult run_case(const TestCase& test, cudaStream_t stream) {
    const std::size_t a_count = static_cast<std::size_t>(test.m) * test.k;
    const std::size_t b_count = static_cast<std::size_t>(test.k) * test.n;
    const std::size_t c_count = static_cast<std::size_t>(test.m) * test.n;
    std::vector<float> a(a_count);
    std::vector<float> b(b_count);
    std::vector<float> observed_a(a_count);
    std::vector<float> observed_b(b_count);
    std::vector<float> actual(c_count, 0.0F);
    make_inputs(test, a, b);
    const std::vector<float> expected = reference_multiply(a, b, test.m, test.k, test.n);

    DeviceBuffer<float> device_a(a_count);
    DeviceBuffer<float> device_b(b_count);
    DeviceBuffer<float> device_c(c_count);
    cuda_check(cudaMemcpyAsync(device_a.get(), a.data(), a_count * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyAsync(device_b.get(), b.data(), b_count * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemsetAsync(device_c.get(), 0xFF, c_count * sizeof(float), stream));
    cudaGetLastError();
    solve(device_a.get(), device_b.get(), device_c.get(),
          test.m, test.k, test.n, stream);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemcpyAsync(actual.data(), device_c.get(), c_count * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(observed_a.data(), device_a.get(), a_count * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(observed_b.data(), device_b.get(), b_count * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));

    if (std::memcmp(observed_a.data(), a.data(), a_count * sizeof(float)) != 0 ||
        std::memcmp(observed_b.data(), b.data(), b_count * sizeof(float)) != 0) {
        return {test.name, false,
                test.internal ? "input modified" : "input buffers must remain unchanged"};
    }

    for (std::size_t index = 0; index < c_count; ++index) {
        if (!close_enough(actual[index], expected[index])) {
            return {test.name, false,
                    test.internal ? "output mismatch"
                                  : "output mismatch at flattened index " +
                                        std::to_string(index)};
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
    for (std::size_t index = 0; index < results.size(); ++index) {
        if (index != 0) out << ',';
        const CaseResult& result = results[index];
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
        {"sample_1", 2, 3, 2, 4242U, true, false},
        {"sample_2", 3, 5, 4, 4242U, true, false},
    };
    if (!public_only) {
        tests.push_back({"internal_case_1", 17, 19, 23, 271828U, false, true});
        tests.push_back({"internal_case_2", 1, 257, 37, 314159U, false, true});
        tests.push_back({"internal_case_3", 73, 31, 1, 271828U, false, true});
        tests.push_back({"internal_case_4", 64, 513, 96, 314159U, false, true});
        tests.push_back({"internal_case_5", 193, 127, 257, 271828U, false, true});
        tests.push_back({"internal_case_6", 3, 4096, 5, 314159U, false, true});
        tests.push_back({"internal_case_7", 4096, 7, 3, 271828U, false, true});
        tests.push_back({"internal_case_8", 5, 9, 4096, 314159U, false, true});
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
