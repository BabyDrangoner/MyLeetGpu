from __future__ import annotations

import importlib.util
import math
import os
from collections.abc import Callable
from pathlib import Path

Update = Callable[[float, float, list[float]], tuple[float, float]]
EMPTY = (float("-inf"), 0.0)
ATOL = RTOL = 1.0e-9
BENCHMARK_CASES = (
    ("N1024-C64", 1024, 64, 4),
    ("N16384-C256", 16384, 256, 1),
    ("N131072-C1024", 131072, 1024, 1),
)
WARMUP = 2
ITERATIONS = 7


def load_submission() -> Update:
    path = Path(os.environ.get("MYLEETGPU_SOURCE_PATH", Path(__file__).with_name("source.py")))
    spec = importlib.util.spec_from_file_location("_online_softmax_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load submission")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "online_softmax_update", None)
    if not callable(function):
        raise TypeError("online_softmax_update must be a function")
    return function


def random_values(count: int, seed: int = 424242) -> list[float]:
    values = []
    for _ in range(count):
        seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
        values.append((seed % 2000001 - 1000000) / 100.0)
    return values


def partition(values: list[float], chunk_size: int = 0) -> list[list[float]]:
    chunks: list[list[float]] = [[]]
    position, seed = 0, 271828
    while position < len(values):
        seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
        length = chunk_size or 1 + seed % 509
        chunks.append(values[position : position + length])
        position += length
        if len(chunks) % 5 == 0:
            chunks.append([])
    chunks.append([])
    return chunks


def cases(full: bool) -> list[tuple[str, list[float], list[list[float]]]]:
    tests = [
        ("sample_empty", [], [[], []]),
        ("sample_singleton", [], [[], [3.5], []]),
        ("sample_chunked", [], [[1.0, 2.0], [], [3.0], [0.0, -1.0]]),
    ]
    if full:
        tests.extend(
            [
                ("internal_equal", [], partition([17.0] * 257)),
                ("internal_increasing", [], partition([float(i - 128) for i in range(257)], 1)),
                ("internal_extremes", [], partition([-10000.0, 10000.0, 0.0] * 170 + [-10000.0])),
                ("internal_decreasing", [], partition([10000.0 - 30.0 * i for i in range(257)])),
                (
                    "internal_close_maxima",
                    [],
                    partition([9999.5 + (i % 7) / 14.0 for i in range(1025)]),
                ),
                (
                    "internal_carried_state",
                    [7.0, 9.0, -4.0, 9.0],
                    partition([5.0 + (i % 11) / 3.0 for i in range(97)]),
                ),
                ("internal_random_chunks", [], partition(random_values(4099))),
                ("internal_long", [], partition(random_values(131072))),
            ]
        )
    return tests


def reference(values: list[float]) -> tuple[float, float]:
    if not values:
        return EMPTY
    maximum = max(values)
    return maximum, math.fsum(math.exp(value - maximum) for value in values)


def merge_reference(state: tuple[float, float], chunk: list[float]) -> tuple[float, float]:
    if not chunk:
        return state
    chunk_max, chunk_sum = reference(chunk)
    maximum = max(state[0], chunk_max)
    return maximum, math.fsum(
        (
            state[1] * math.exp(state[0] - maximum),
            chunk_sum * math.exp(chunk_max - maximum),
        )
    )


def check_state(actual: object, expected: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise AssertionError("return a tuple (new_max, new_sum)")
    if any(isinstance(value, bool) or not isinstance(value, float | int) for value in actual):
        raise AssertionError("statistics must be real scalar numbers")
    maximum, denominator = actual
    if expected == EMPTY:
        if maximum != -math.inf or denominator != 0.0:
            raise AssertionError("empty state must remain (-inf, 0)")
    elif not (
        math.isfinite(maximum)
        and math.isfinite(denominator)
        and denominator > 0.0
        and math.isclose(maximum, expected[0], abs_tol=ATOL, rel_tol=RTOL)
        and math.isclose(denominator, expected[1], abs_tol=ATOL, rel_tol=RTOL)
    ):
        raise AssertionError("incorrect running maximum or rescaled exponential sum")
    return maximum, denominator


def check_probabilities(values: list[float], state: tuple[float, float]) -> None:
    expected = reference(values)
    check_state(state, expected)
    if not values:
        return
    probabilities = [math.exp(value - state[0]) / state[1] for value in values]
    if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=ATOL, rel_tol=RTOL):
        raise AssertionError("final probabilities are not normalized")
    for value, probability in zip(values, probabilities, strict=True):
        expected_probability = math.exp(value - expected[0]) / expected[1]
        if not math.isclose(probability, expected_probability, abs_tol=ATOL, rel_tol=RTOL):
            raise AssertionError("final probability differs from stable softmax")


def check_stream(update: Update, history: list[float], chunks: list[list[float]]) -> None:
    state = expected = reference(history)
    values = list(history)
    for chunk in chunks:
        original = chunk.copy()
        expected = merge_reference(expected, original)
        actual = update(*state, chunk)
        if chunk != original:
            raise AssertionError("chunk must not be modified")
        state = check_state(actual, expected)
        values.extend(original)
    check_probabilities(values, state)


def run_stream(update: Update, chunks: list[list[float]]) -> tuple[float, float]:
    state = EMPTY
    for chunk in chunks:
        state = update(*state, chunk)
    return state
