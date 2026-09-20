#include "online_softmax_common.h"

#include <iostream>

int main(int argc, char** argv) {
    using namespace online_softmax_harness;
    if (argc != 3 || std::string(argv[1]) != "--mode"
        || (std::string(argv[2]) != "public" && std::string(argv[2]) != "full")) {
        std::cout << "MYLEETGPU_RESULT={\"status\":\"runtime_error\",\"cases\":[]}\n";
        return 2;
    }
    const auto tests = cases(std::string(argv[2]) == "full");
    std::vector<bool> passed;
    bool runtime_error = false;
    std::size_t passed_count = 0;
    for (const auto& test : tests) {
        bool success = false;
        try {
            check_stream(test);
            success = true;
            ++passed_count;
        } catch (const std::logic_error&) {
            // The validator's contract checks use logic_error for wrong answers.
        } catch (...) {
            runtime_error = true;
        }
        passed.push_back(success);
    }
    const char* status = runtime_error ? "runtime_error"
        : (passed_count == tests.size() ? "passed" : "wrong_answer");
    std::cout << "MYLEETGPU_RESULT={\"status\":\"" << status << "\",\"cases\":[";
    for (std::size_t i = 0; i < tests.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << "{\"name\":\"" << tests[i].name << "\",\"passed\":"
                  << (passed[i] ? "true" : "false");
        if (!passed[i]) std::cout << ",\"message\":\"online statistics or input contract failed\"";
        std::cout << '}';
    }
    std::cout << "],\"summary\":{\"total\":" << tests.size()
              << ",\"passed\":" << passed_count << ",\"failed\":" << tests.size() - passed_count
              << "}}\n";
    return passed_count == tests.size() ? 0 : 1;
}
