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
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kWarmup = 6;
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
    float p;
    int inner_repetitions;
};

struct Reference {
    std::vector<float> output;
    std::vector<int> counts;
};

struct Measurement {
    std::string label;
    int size;
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
        } else {
            out << static_cast<char>(ch);
        }
    }
    out << '"';
    return out.str();
}

std::vector<float> make_probabilities(int rows, int cols) {
    const std::size_t count =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    std::vector<float> probabilities(count);
    const double denominator =
        static_cast<double>(cols) * static_cast<double>(cols + 1) * 0.5;
    for (int row = 0; row < rows; ++row) {
        const std::size_t row_offset = static_cast<std::size_t>(row) * cols;
        const int shift = (row * 17) % cols;
        int maximum_col = 0;
        for (int col = 0; col < cols; ++col) {
            const int rank = ((col + shift) % cols) + 1;
            probabilities[row_offset + col] =
                static_cast<float>(static_cast<double>(rank) / denominator);
            if (rank == cols) maximum_col = col;
        }
        double other_sum = 0.0;
        for (int col = 0; col < cols; ++col) {
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
                         const BenchmarkCase& benchmark_case) {
    Reference reference{
        std::vector<float>(probabilities.size(), 0.0F),
        std::vector<int>(static_cast<std::size_t>(benchmark_case.rows),
                         benchmark_case.cols),
    };
    std::vector<float> ordered(static_cast<std::size_t>(benchmark_case.cols));
    for (int row = 0; row < benchmark_case.rows; ++row) {
        const std::size_t row_offset =
            static_cast<std::size_t>(row) * benchmark_case.cols;
        std::copy_n(probabilities.begin() + static_cast<std::ptrdiff_t>(row_offset),
                    benchmark_case.cols, ordered.begin());
        std::sort(ordered.begin(), ordered.end(), std::greater<float>());
        double cumulative = 0.0;
        int retained = benchmark_case.cols;
        for (int rank = 0; rank < benchmark_case.cols; ++rank) {
            cumulative += static_cast<double>(ordered[rank]);
            if (cumulative >= static_cast<double>(benchmark_case.p)) {
                retained = rank + 1;
                break;
            }
        }
        reference.counts[static_cast<std::size_t>(row)] = retained;
        std::copy_n(ordered.begin(), retained,
                    reference.output.begin() + static_cast<std::ptrdiff_t>(row_offset));
    }
    return reference;
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
    return {
        benchmark_case.label,
        benchmark_case.rows * benchmark_case.cols,
        benchmark_case.inner_repetitions,
        std::move(samples),
        median,
        sorted[p95_index],
        sorted.front(),
        mean > 0.0 ? deviation / mean : 0.0,
    };
}

Measurement run_benchmark(const BenchmarkCase& benchmark_case,
                          cudaStream_t stream) {
    const std::size_t count =
        static_cast<std::size_t>(benchmark_case.rows) * benchmark_case.cols;
    const std::size_t bytes = count * sizeof(float);
    const std::size_t count_bytes =
        static_cast<std::size_t>(benchmark_case.rows) * sizeof(int);
    const std::vector<float> probabilities =
        make_probabilities(benchmark_case.rows, benchmark_case.cols);
    const Reference reference = make_reference(probabilities, benchmark_case);
    std::vector<float> output(count);
    std::vector<float> observed_input(count);
    std::vector<int> counts(static_cast<std::size_t>(benchmark_case.rows));

    DeviceBuffer<float> device_input(count);
    DeviceBuffer<float> device_output(count);
    DeviceBuffer<int> device_counts(static_cast<std::size_t>(benchmark_case.rows));
    cuda_check(cudaMemcpyAsync(device_input.get(), probabilities.data(), bytes,
                               cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemsetAsync(device_output.get(), 0xFF, bytes, stream));
    cuda_check(cudaMemsetAsync(device_counts.get(), 0xA5, count_bytes, stream));
    cuda_check(cudaStreamSynchronize(stream));

    for (int repeat = 0;
         repeat < kWarmup * benchmark_case.inner_repetitions;
         ++repeat) {
        solve(device_input.get(), device_output.get(), device_counts.get(),
              benchmark_case.rows, benchmark_case.cols, benchmark_case.p, stream);
    }
    cuda_check(cudaGetLastError());
    cuda_check(cudaStreamSynchronize(stream));
    cuda_check(cudaMemcpyAsync(output.data(), device_output.get(), bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(counts.data(), device_counts.get(), count_bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(observed_input.data(), device_input.get(), bytes,
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));

    if (std::memcmp(observed_input.data(), probabilities.data(), bytes) != 0) {
        throw std::runtime_error("input was modified");
    }
    if (counts != reference.counts ||
        std::memcmp(output.data(), reference.output.data(), bytes) != 0) {
        throw std::runtime_error("correctness check failed before timing");
    }

    Event start;
    Event stop;
    std::vector<double> samples;
    samples.reserve(kIterations);
    for (int sample = 0; sample < kIterations; ++sample) {
        cudaGetLastError();
        cuda_check(cudaEventRecord(start.get(), stream));
        for (int inner = 0; inner < benchmark_case.inner_repetitions; ++inner) {
            solve(device_input.get(), device_output.get(), device_counts.get(),
                  benchmark_case.rows, benchmark_case.cols,
                  benchmark_case.p, stream);
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
        {"8192x128", 8192, 128, 0.5F, 8},
        {"4096x512", 4096, 512, 0.9F, 4},
        {"4096x1024", 4096, 1024, 0.95F, 2},
    };
    try {
        cuda_check(cudaFree(nullptr));
        Stream stream;
        std::vector<Measurement> measurements;
        measurements.reserve(cases.size());
        for (const BenchmarkCase& benchmark_case : cases) {
            measurements.push_back(run_benchmark(benchmark_case, stream.get()));
        }
        print_success(measurements);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        print_error(std::string("benchmark failed: ") + error.what());
        return EXIT_FAILURE;
    }
}
