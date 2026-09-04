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
    int k;
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
    std::uniform_real_distribution<float> distribution(-10000.0F, 10000.0F);
    for (int row = 0; row < test.rows; ++row) {
        for (int col = 0; col < test.cols; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * test.cols + col;
            if (test.pattern == "sequence") {
                input[index] = 0.25F * static_cast<float>((col * 37 + row * 11) % 257 - 128);
            } else if (test.pattern == "ties") {
                input[index] = 2.0F * static_cast<float>((col * 5 + row * 3) % 7 - 3);
            } else if (test.pattern == "descending") {
                input[index] = static_cast<float>(test.cols - col) +
                               0.001F * static_cast<float>(row % 7);
            } else if (test.pattern == "all_equal") {
                input[index] = -17.25F;
            } else if (test.pattern == "extreme") {
                switch (col % 8) {
                    case 0: input[index] = 10000.0F; break;
                    case 1: input[index] = -10000.0F; break;
                    case 2: input[index] = 9999.0F; break;
                    case 3: input[index] = -9999.0F; break;
                    case 4: input[index] = 0.0F; break;
                    case 5: input[index] = 42.0F; break;
                    case 6: input[index] = -42.0F; break;
                    default: input[index] = static_cast<float>(row % 5); break;
                }
            } else {
                input[index] = distribution(generator);
            }
        }
    }
    return input;
}

std::vector<float> reference_values(const std::vector<float>& input,
                                    int rows,
                                    int cols,
                                    int k) {
    std::vector<float> expected(static_cast<std::size_t>(rows) * k);
    std::vector<float> row_values(static_cast<std::size_t>(cols));
    for (int row = 0; row < rows; ++row) {
        const std::size_t input_offset = static_cast<std::size_t>(row) * cols;
        std::copy_n(input.begin() + static_cast<std::ptrdiff_t>(input_offset),
                    cols, row_values.begin());
        std::sort(row_values.begin(), row_values.end(), std::greater<float>());
        std::copy_n(row_values.begin(), k,
                    expected.begin() + static_cast<std::ptrdiff_t>(row) * k);
    }
    return expected;
}

std::string validate_output(const std::vector<float>& input,
                            const std::vector<float>& expected,
                            const std::vector<float>& values,
                            const std::vector<int>& indices,
                            int rows,
                            int cols,
                            int k) {
    std::vector<unsigned char> seen(static_cast<std::size_t>(cols));
    for (int row = 0; row < rows; ++row) {
        std::fill(seen.begin(), seen.end(), 0U);
        const std::size_t input_offset = static_cast<std::size_t>(row) * cols;
        const std::size_t output_offset = static_cast<std::size_t>(row) * k;
        for (int rank = 0; rank < k; ++rank) {
            const std::size_t position = output_offset + rank;
            const int index = indices[position];
            if (index < 0 || index >= cols) {
                return "index out of range at row " + std::to_string(row) +
                       ", rank " + std::to_string(rank);
            }
            if (seen[static_cast<std::size_t>(index)] != 0U) {
                return "duplicate index at row " + std::to_string(row);
            }
            seen[static_cast<std::size_t>(index)] = 1U;
            const float value = values[position];
            if (!std::isfinite(value)) {
                return "non-finite value at row " + std::to_string(row) +
                       ", rank " + std::to_string(rank);
            }
            if (value != input[input_offset + static_cast<std::size_t>(index)]) {
                return "value does not match its index at row " + std::to_string(row) +
                       ", rank " + std::to_string(rank);
            }
            if (rank > 0 && values[position - 1U] < value) {
                return "values are not in descending order at row " +
                       std::to_string(row);
            }
            if (value != expected[position]) {
                return "selected values do not match Top-K at row " +
                       std::to_string(row);
            }
        }
    }
    return "";
}

CaseResult run_case(const TestCase& test, cudaStream_t stream) {
    const std::size_t input_count =
        static_cast<std::size_t>(test.rows) * static_cast<std::size_t>(test.cols);
    const std::size_t output_count =
        static_cast<std::size_t>(test.rows) * static_cast<std::size_t>(test.k);
    const std::size_t input_bytes = input_count * sizeof(float);
    const std::size_t value_bytes = output_count * sizeof(float);
    const std::size_t index_bytes = output_count * sizeof(int);
    const std::vector<float> input = make_input(test);
    const std::vector<float> expected =
        reference_values(input, test.rows, test.cols, test.k);
    std::vector<float> observed_input(input_count);
    std::vector<float> first_values(output_count);
    std::vector<float> second_values(output_count);
    std::vector<int> first_indices(output_count);
    std::vector<int> second_indices(output_count);

    DeviceBuffer<float> device_input(input_count);
    DeviceBuffer<float> device_values(output_count);
    DeviceBuffer<int> device_indices(output_count);
    cuda_check(cudaMemcpyAsync(device_input.get(), input.data(), input_bytes,
                               cudaMemcpyHostToDevice, stream));

    cuda_check(cudaMemsetAsync(device_values.get(), 0xFF, value_bytes, stream));
    cuda_check(cudaMemsetAsync(device_indices.get(), 0xFF, index_bytes, stream));
    cudaGetLastError();
    solve(device_input.get(), device_values.get(), device_indices.get(),
          test.rows, test.cols, test.k, stream);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemcpyAsync(first_values.data(), device_values.get(), value_bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(first_indices.data(), device_indices.get(), index_bytes,
                               cudaMemcpyDeviceToHost, stream));

    cuda_check(cudaMemsetAsync(device_values.get(), 0x7F, value_bytes, stream));
    cuda_check(cudaMemsetAsync(device_indices.get(), 0x7F, index_bytes, stream));
    cudaGetLastError();
    solve(device_input.get(), device_values.get(), device_indices.get(),
          test.rows, test.cols, test.k, stream);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemcpyAsync(second_values.data(), device_values.get(), value_bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(second_indices.data(), device_indices.get(), index_bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(observed_input.data(), device_input.get(), input_bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));

    if (std::memcmp(observed_input.data(), input.data(), input_bytes) != 0) {
        return {test.name, false,
                test.internal ? "input modified" : "input must remain unchanged"};
    }
    const std::string first_error = validate_output(
        input, expected, first_values, first_indices, test.rows, test.cols, test.k);
    if (!first_error.empty()) {
        return {test.name, false, test.internal ? "output mismatch" : first_error};
    }
    const std::string second_error = validate_output(
        input, expected, second_values, second_indices, test.rows, test.cols, test.k);
    if (!second_error.empty()) {
        return {test.name, false,
                test.internal ? "output mismatch"
                              : "repeated call depends on prior output: " + second_error};
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
        {"sample_1", 2, 5, 1, 73013U, "sequence", false},
        {"sample_2", 3, 9, 3, 73013U, "ties", false},
    };
    if (!public_only) {
        tests.push_back({"internal_case_1", 7, 31, 7, 271828U, "random", true});
        tests.push_back({"internal_case_2", 5, 64, 64, 271828U, "descending", true});
        tests.push_back({"internal_case_3", 4096, 17, 4, 314159U, "ties", true});
        tests.push_back({"internal_case_4", 65536, 1, 1, 314159U, "all_equal", true});
        tests.push_back({"internal_case_5", 4096, 1024, 64, 271828U, "extreme", true});
        tests.push_back({"internal_case_6", 257, 513, 1, 314159U, "random", true});
        tests.push_back({"internal_case_7", 127, 1023, 63, 271828U, "ties", true});
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
