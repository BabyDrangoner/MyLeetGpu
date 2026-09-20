from __future__ import annotations

import json
import sys
from time import perf_counter_ns

from online_softmax_common import (
    BENCHMARK_CASES,
    ITERATIONS,
    WARMUP,
    check_probabilities,
    check_stream,
    load_submission,
    partition,
    random_values,
    run_stream,
)


def main(argv: list[str]) -> int:
    measurements = []
    status = "passed"
    try:
        if argv[1:] != ["--mode", "benchmark"]:
            raise ValueError("expected --mode benchmark")
        update = load_submission()
        for label, count, chunk_size, repetitions in BENCHMARK_CASES:
            values = random_values(count, 20260920)
            chunks = partition(values, chunk_size)
            originals = [chunk.copy() for chunk in chunks]
            check_stream(update, [], chunks)
            for _ in range(WARMUP):
                run_stream(update, chunks)
            samples = []
            for _ in range(ITERATIONS):
                start = perf_counter_ns()
                for _ in range(repetitions):
                    state = run_stream(update, chunks)
                duration_ns = perf_counter_ns() - start
                samples.append(max(duration_ns, 1) / 1_000_000 / repetitions)
                if chunks != originals:
                    raise AssertionError("chunk was modified during timing")
                check_probabilities(values, state)
            measurements.append(
                {"label": label, "samples_ms": samples, "inner_repetitions": repetitions}
            )
    except Exception as error:
        status = "compile_error" if isinstance(error, SyntaxError) else "runtime_error"
        measurements = []
    print(
        "MYLEETGPU_RESULT="
        + json.dumps(
            {
                "status": status,
                "protocol_version": "1",
                "measurements": measurements,
            },
            separators=(",", ":"),
        )
    )
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
