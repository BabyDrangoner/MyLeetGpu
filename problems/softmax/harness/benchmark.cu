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
constexpr float kAbsoluteTolerance = 3.0e-5F;
constexpr float kRelativeTolerance = 3.0e-4F;
constexpr double kRowSumTolerance = 1.0e-3;
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
    return std::fabs(actual - expected) <=
           kAbsoluteTolerance + kRelativeTolerance * std::fabs(expected);
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
    const std::size_t count = static_cast<std::size_t>(benchmark_case.rows) *
                              static_cast<std::size_t>(benchmark_case.cols);
    const std::size_t bytes = count * sizeof(float);
    std::uniform_real_distribution<float> distribution(-20.0F, 20.0F);
    std::vector<float> input(count);
    std::vector<float> observed_input(count);
    std::vector<float> output(count);
    for (float& value : input) value = distribution(generator);
    const std::vector<float> expected =
        reference_softmax(input, benchmark_case.rows, benchmark_case.cols);

    DeviceBuffer<float> device_input(count);
    DeviceBuffer<float> device_output(count);
    cuda_check(cudaMemcpyAsync(device_input.get(), input.data(), bytes,
                               cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemsetAsync(device_output.get(), 0xFF, bytes, stream));
    cuda_check(cudaStreamSynchronize(stream));
    for (int i = 0; i < kWarmup * benchmark_case.inner_repetitions; ++i) {
        solve(device_input.get(), device_output.get(), benchmark_case.rows,
              benchmark_case.cols, stream);
    }
    cuda_check(cudaGetLastError());
    cuda_check(cudaStreamSynchronize(stream));

    cuda_check(cudaMemcpyAsync(output.data(), device_output.get(), bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(observed_input.data(), device_input.get(), bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    if (std::memcmp(observed_input.data(), input.data(), bytes) != 0) {
        throw std::runtime_error("input was modified");
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (!close_float(output[index], expected[index])) {
            throw std::runtime_error("correctness check failed before timing");
        }
    }
    for (int row = 0; row < benchmark_case.rows; ++row) {
        double row_sum = 0.0;
        for (int col = 0; col < benchmark_case.cols; ++col) {
            const float value =
                output[static_cast<std::size_t>(row) * benchmark_case.cols + col];
            if (value < 0.0F) {
                throw std::runtime_error("correctness check failed before timing");
            }
            row_sum += static_cast<double>(value);
        }
        if (std::fabs(row_sum - 1.0) > kRowSumTolerance) {
            throw std::runtime_error("correctness check failed before timing");
        }
    }

    Event start;
    Event stop;
    std::vector<double> samples;
    samples.reserve(kIterations);
    for (int sample = 0; sample < kIterations; ++sample) {
        cudaGetLastError();
        cuda_check(cudaEventRecord(start.get(), stream));
        for (int inner = 0; inner < benchmark_case.inner_repetitions; ++inner) {
            solve(device_input.get(), device_output.get(), benchmark_case.rows,
                  benchmark_case.cols, stream);
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
        {"8192x128", 8192, 128, 16},
        {"4096x1024", 4096, 1024, 8},
        {"2048x4096", 2048, 4096, 4},
    };
    try {
        cuda_check(cudaFree(nullptr));
        Stream stream;
        std::mt19937 generator(20240901U);
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
