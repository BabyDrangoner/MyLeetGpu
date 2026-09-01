#include "solve.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

constexpr int kWarmup = 4;
constexpr int kIterations = 15;
constexpr const char* kProtocolVersion = "1";
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
    int m;
    int k;
    int n;
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

bool close_enough(float actual, float expected) {
    if (std::isnan(actual) || std::isnan(expected)) return false;
    if (std::isinf(actual) || std::isinf(expected)) return actual == expected;
    const double difference = std::abs(static_cast<double>(actual) - expected);
    return difference <=
           kAbsoluteTolerance + kRelativeTolerance * std::abs(static_cast<double>(expected));
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
    const std::int64_t size =
        static_cast<std::int64_t>(benchmark_case.m) * benchmark_case.k +
        static_cast<std::int64_t>(benchmark_case.k) * benchmark_case.n +
        static_cast<std::int64_t>(benchmark_case.m) * benchmark_case.n;
    return {benchmark_case.label, size, benchmark_case.inner_repetitions,
            std::move(samples), median, sorted[p95_index], sorted.front(),
            mean > 0.0 ? deviation / mean : 0.0};
}

Measurement run_benchmark(const BenchmarkCase& benchmark_case,
                          cudaStream_t stream,
                          std::mt19937& generator) {
    const std::size_t a_count =
        static_cast<std::size_t>(benchmark_case.m) * benchmark_case.k;
    const std::size_t b_count =
        static_cast<std::size_t>(benchmark_case.k) * benchmark_case.n;
    const std::size_t c_count =
        static_cast<std::size_t>(benchmark_case.m) * benchmark_case.n;
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> a(a_count);
    std::vector<float> b(b_count);
    std::vector<float> observed_a(a_count);
    std::vector<float> observed_b(b_count);
    std::vector<float> output(c_count);
    for (float& value : a) value = distribution(generator);
    for (float& value : b) value = distribution(generator);
    const std::vector<float> expected = reference_multiply(
        a, b, benchmark_case.m, benchmark_case.k, benchmark_case.n);

    DeviceBuffer<float> device_a(a_count);
    DeviceBuffer<float> device_b(b_count);
    DeviceBuffer<float> device_c(c_count);
    cuda_check(cudaMemcpyAsync(device_a.get(), a.data(), a_count * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyAsync(device_b.get(), b.data(), b_count * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemsetAsync(device_c.get(), 0xFF, c_count * sizeof(float), stream));
    cuda_check(cudaStreamSynchronize(stream));
    for (int repeat = 0; repeat < kWarmup * benchmark_case.inner_repetitions; ++repeat) {
        solve(device_a.get(), device_b.get(), device_c.get(),
              benchmark_case.m, benchmark_case.k, benchmark_case.n, stream);
    }
    cuda_check(cudaGetLastError());
    cuda_check(cudaStreamSynchronize(stream));

    cuda_check(cudaMemcpyAsync(output.data(), device_c.get(), c_count * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(observed_a.data(), device_a.get(), a_count * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(observed_b.data(), device_b.get(), b_count * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    if (std::memcmp(observed_a.data(), a.data(), a_count * sizeof(float)) != 0 ||
        std::memcmp(observed_b.data(), b.data(), b_count * sizeof(float)) != 0) {
        throw std::runtime_error("input buffers were modified");
    }
    for (std::size_t index = 0; index < c_count; ++index) {
        if (!close_enough(output[index], expected[index])) {
            throw std::runtime_error("correctness check failed before timing");
        }
    }

    Event start;
    Event stop;
    std::vector<double> samples;
    samples.reserve(kIterations);
    for (int sample = 0; sample < kIterations; ++sample) {
        cudaGetLastError();
        cuda_check(cudaMemsetAsync(device_c.get(), 0xFF, c_count * sizeof(float), stream));
        cuda_check(cudaEventRecord(start.get(), stream));
        for (int inner = 0; inner < benchmark_case.inner_repetitions; ++inner) {
            solve(device_a.get(), device_b.get(), device_c.get(),
                  benchmark_case.m, benchmark_case.k, benchmark_case.n, stream);
        }
        cuda_check(cudaGetLastError());
        cuda_check(cudaEventRecord(stop.get(), stream));
        cuda_check(cudaEventSynchronize(stop.get()));
        float elapsed_ms = 0.0F;
        cuda_check(cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get()));
        samples.push_back(static_cast<double>(elapsed_ms) /
                          static_cast<double>(benchmark_case.inner_repetitions));

        cuda_check(cudaMemcpyAsync(output.data(), device_c.get(), c_count * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaMemcpyAsync(observed_a.data(), device_a.get(), a_count * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaMemcpyAsync(observed_b.data(), device_b.get(), b_count * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaStreamSynchronize(stream));
        if (std::memcmp(observed_a.data(), a.data(), a_count * sizeof(float)) != 0 ||
            std::memcmp(observed_b.data(), b.data(), b_count * sizeof(float)) != 0) {
            throw std::runtime_error("input buffers were modified during timing");
        }
        for (std::size_t index = 0; index < c_count; ++index) {
            if (!close_enough(output[index], expected[index])) {
                throw std::runtime_error("correctness check failed during timing");
            }
        }
    }
    return summarize(benchmark_case, std::move(samples));
}

void print_success(const std::vector<Measurement>& measurements) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "MYLEETGPU_RESULT={\"status\":\"passed\",\"measurements\":[";
    for (std::size_t index = 0; index < measurements.size(); ++index) {
        if (index != 0) out << ',';
        const Measurement& value = measurements[index];
        out << "{\"size\":" << value.size
            << ",\"label\":" << json_string(value.label)
            << ",\"samples_ms\":[";
        for (std::size_t sample = 0; sample < value.samples_ms.size(); ++sample) {
            if (sample != 0) out << ',';
            out << value.samples_ms[sample];
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
        {"128x128x128", 128, 128, 128, 1},
        {"256x256x256", 256, 256, 256, 1},
        {"512x512x512", 512, 512, 512, 1},
    };
    try {
        cuda_check(cudaFree(nullptr));
        Stream stream;
        std::mt19937 generator(987654U);
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
