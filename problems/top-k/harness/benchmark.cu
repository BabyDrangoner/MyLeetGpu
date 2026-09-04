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
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kWarmup = 8;
constexpr int kIterations = 20;
constexpr const char* kProtocolVersion = "1";

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

class Event {
public:
    Event() { cuda_check(cudaEventCreate(&value_)); }
    ~Event() { if (value_ != nullptr) cudaEventDestroy(value_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t get() const { return value_; }
private:
    cudaEvent_t value_ = nullptr;
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

struct BenchmarkCase {
    const char* label;
    int rows;
    int cols;
    int k;
    int inner_repetitions;
};

struct Measurement {
    std::string label;
    std::int64_t size;
    int inner_repetitions;
    std::vector<double> samples_ms;
    double median_ms;
    double p95_ms;
    double min_ms;
    double cv;
};

std::string json_string(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char ch : value) {
        if (ch == '"') out << "\\\"";
        else if (ch == '\\') out << "\\\\";
        else if (ch == '\n') out << "\\n";
        else if (ch == '\r') out << "\\r";
        else if (ch == '\t') out << "\\t";
        else if (ch < 0x20U) {
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                << static_cast<int>(ch) << std::dec;
        } else out << static_cast<char>(ch);
    }
    out << '"';
    return out.str();
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

void check_output(const std::vector<float>& input,
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
            if (index < 0 || index >= cols || seen[static_cast<std::size_t>(index)] != 0U) {
                throw std::runtime_error("correctness check failed before timing");
            }
            seen[static_cast<std::size_t>(index)] = 1U;
            if (!std::isfinite(values[position]) ||
                values[position] != input[input_offset + static_cast<std::size_t>(index)] ||
                values[position] != expected[position] ||
                (rank > 0 && values[position - 1U] < values[position])) {
                throw std::runtime_error("correctness check failed before timing");
            }
        }
    }
}

Measurement summarize(const BenchmarkCase& benchmark_case,
                      std::vector<double> samples) {
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t count = sorted.size();
    const double median = count % 2 == 0
        ? (sorted[count / 2 - 1] + sorted[count / 2]) * 0.5
        : sorted[count / 2];
    const std::size_t p95_index =
        static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(count))) - 1U;
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                        static_cast<double>(count);
    double squared_error = 0.0;
    for (const double sample : samples) {
        const double delta = sample - mean;
        squared_error += delta * delta;
    }
    const double deviation = std::sqrt(squared_error / static_cast<double>(count));
    return {benchmark_case.label,
            static_cast<std::int64_t>(benchmark_case.rows) * benchmark_case.cols,
            benchmark_case.inner_repetitions,
            std::move(samples), median, sorted[p95_index], sorted.front(),
            mean > 0.0 ? deviation / mean : 0.0};
}

Measurement run_benchmark(const BenchmarkCase& benchmark_case,
                          cudaStream_t stream,
                          std::mt19937& generator) {
    const std::size_t input_count = static_cast<std::size_t>(benchmark_case.rows) *
                                    static_cast<std::size_t>(benchmark_case.cols);
    const std::size_t output_count = static_cast<std::size_t>(benchmark_case.rows) *
                                     static_cast<std::size_t>(benchmark_case.k);
    const std::size_t input_bytes = input_count * sizeof(float);
    const std::size_t value_bytes = output_count * sizeof(float);
    const std::size_t index_bytes = output_count * sizeof(int);
    std::uniform_real_distribution<float> distribution(-10000.0F, 10000.0F);
    std::vector<float> input(input_count);
    std::vector<float> observed_input(input_count);
    std::vector<float> output_values(output_count);
    std::vector<int> output_indices(output_count);
    for (float& value : input) value = distribution(generator);
    const std::vector<float> expected = reference_values(
        input, benchmark_case.rows, benchmark_case.cols, benchmark_case.k);

    DeviceBuffer<float> device_input(input_count);
    DeviceBuffer<float> device_values(output_count);
    DeviceBuffer<int> device_indices(output_count);
    cuda_check(cudaMemcpyAsync(device_input.get(), input.data(), input_bytes,
                               cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemsetAsync(device_values.get(), 0xFF, value_bytes, stream));
    cuda_check(cudaMemsetAsync(device_indices.get(), 0xFF, index_bytes, stream));
    cuda_check(cudaStreamSynchronize(stream));
    for (int i = 0; i < kWarmup * benchmark_case.inner_repetitions; ++i) {
        solve(device_input.get(), device_values.get(), device_indices.get(),
              benchmark_case.rows, benchmark_case.cols, benchmark_case.k, stream);
    }
    cuda_check(cudaGetLastError());
    cuda_check(cudaStreamSynchronize(stream));

    cuda_check(cudaMemcpyAsync(output_values.data(), device_values.get(), value_bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(output_indices.data(), device_indices.get(), index_bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(observed_input.data(), device_input.get(), input_bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    if (std::memcmp(observed_input.data(), input.data(), input_bytes) != 0) {
        throw std::runtime_error("input was modified");
    }
    check_output(input, expected, output_values, output_indices,
                 benchmark_case.rows, benchmark_case.cols, benchmark_case.k);

    Event start;
    Event stop;
    std::vector<double> samples;
    samples.reserve(kIterations);
    for (int sample = 0; sample < kIterations; ++sample) {
        cudaGetLastError();
        cuda_check(cudaEventRecord(start.get(), stream));
        for (int inner = 0; inner < benchmark_case.inner_repetitions; ++inner) {
            solve(device_input.get(), device_values.get(), device_indices.get(),
                  benchmark_case.rows, benchmark_case.cols, benchmark_case.k, stream);
        }
        cuda_check(cudaGetLastError());
        cuda_check(cudaEventRecord(stop.get(), stream));
        cuda_check(cudaEventSynchronize(stop.get()));
        float elapsed_ms = 0.0F;
        cuda_check(cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get()));
        samples.push_back(static_cast<double>(elapsed_ms) /
                          static_cast<double>(benchmark_case.inner_repetitions));
    }
    return summarize(benchmark_case, std::move(samples));
}

void print_success(const std::vector<Measurement>& measurements) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "MYLEETGPU_RESULT={\"status\":\"passed\",\"measurements\":[";
    for (std::size_t i = 0; i < measurements.size(); ++i) {
        if (i != 0) out << ',';
        const Measurement& value = measurements[i];
        out << "{\"size\":" << value.size
            << ",\"label\":" << json_string(value.label)
            << ",\"samples_ms\":[";
        for (std::size_t j = 0; j < value.samples_ms.size(); ++j) {
            if (j != 0) out << ',';
            out << value.samples_ms[j];
        }
        out << "],\"median_ms\":" << value.median_ms
            << ",\"p95_ms\":" << value.p95_ms
            << ",\"min_ms\":" << value.min_ms
            << ",\"cv\":" << value.cv
            << ",\"inner_repetitions\":" << value.inner_repetitions << '}';
    }
    out << "],\"protocol_version\":\"" << kProtocolVersion << "\"}";
    std::cout << out.str() << std::endl;
}

void print_error(const std::string& message) {
    std::cout << "MYLEETGPU_RESULT={\"status\":\"runtime_error\","
              << "\"measurements\":[],\"protocol_version\":\""
              << kProtocolVersion << "\",\"message\":"
              << json_string(message) << '}' << std::endl;
}

}  // namespace

int main() {
    const std::vector<BenchmarkCase> cases = {
        {"32768x32-k4", 32768, 32, 4, 8},
        {"8192x256-k16", 8192, 256, 16, 4},
        {"4096x1024-k64", 4096, 1024, 64, 1},
    };
    try {
        cuda_check(cudaFree(nullptr));
        Stream stream;
        std::mt19937 generator(20240902U);
        std::vector<Measurement> measurements;
        measurements.reserve(cases.size());
        for (const BenchmarkCase& benchmark_case : cases) {
            measurements.push_back(run_benchmark(benchmark_case, stream.get(), generator));
        }
        print_success(measurements);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        print_error(std::string("benchmark failed: ") + error.what());
        return EXIT_FAILURE;
    }
}
