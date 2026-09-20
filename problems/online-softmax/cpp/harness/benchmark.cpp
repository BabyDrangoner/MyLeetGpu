#include "online_softmax_common.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
    using namespace online_softmax_harness;
    try {
        if (argc != 3 || std::string(argv[1]) != "--mode" || std::string(argv[2]) != "benchmark") {
            throw std::runtime_error("expected --mode benchmark");
        }
        struct Size { const char* label; std::size_t count, chunk_size; int repetitions; };
        const Size sizes[] = {
            {"N1024-C64", 1024, 64, 4},
            {"N16384-C256", 16384, 256, 1},
            {"N131072-C1024", 131072, 1024, 1},
        };
        std::ostringstream measurements;
        measurements << std::setprecision(17);
        for (std::size_t i = 0; i < 3; ++i) {
            const auto size = sizes[i];
            const auto values = random_values(size.count, 20260920);
            auto chunks = partition(values, size.chunk_size);
            const auto original_chunks = chunks;
            check_stream({"benchmark", {}, chunks});
            for (int warmup = 0; warmup < kWarmup; ++warmup) run_stream(chunks);
            if (i) measurements << ',';
            measurements << "{\"label\":\"" << size.label << "\",\"inner_repetitions\":"
                         << size.repetitions << ",\"samples_ms\":[";
            for (int sample = 0; sample < kIterations; ++sample) {
                State state;
                const auto start = std::chrono::steady_clock::now();
                for (int repetition = 0; repetition < size.repetitions; ++repetition) {
                    state = run_stream(chunks);
                }
                const auto end = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
                if (chunks != original_chunks) throw std::logic_error("chunk was modified");
                check_probabilities(values, state);
                if (sample) measurements << ',';
                measurements << std::max(elapsed, 0.000001) / size.repetitions;
            }
            measurements << "]}";
        }
        std::cout << "MYLEETGPU_RESULT={\"status\":\"passed\",\"protocol_version\":\"1\","
                  << "\"measurements\":[" << measurements.str() << "]}\n";
        return 0;
    } catch (...) {
        std::cout << "MYLEETGPU_RESULT={\"status\":\"runtime_error\",\"protocol_version\":\"1\","
                  << "\"measurements\":[]}\n";
        return 1;
    }
}
