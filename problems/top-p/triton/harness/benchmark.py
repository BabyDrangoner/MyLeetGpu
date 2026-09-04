from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

RESULT_PREFIX = "MYLEETGPU_RESULT="
PROTOCOL_VERSION = "1"
WARMUP = 6
ITERATIONS = 20
CASES = (
    ("8192x128", 8192, 128, 0.5, 8),
    ("4096x512", 4096, 512, 0.9, 4),
    ("4096x1024", 4096, 1024, 0.95, 2),
)


class SubmissionCompileError(RuntimeError):
    pass


def is_compilation_error(error: BaseException) -> bool:
    if isinstance(error, SyntaxError | SubmissionCompileError):
        return True
    names = {"CompilationError", "CompileTimeAssertionFailure", "OutOfResources", "PTXASError"}
    return any(
        cls.__name__ in names and cls.__module__.startswith("triton") for cls in type(error).__mro__
    )


def load_submission() -> ModuleType:
    source_path = Path(os.environ.get("MYLEETGPU_SOURCE_PATH", "/work/source.py"))
    policy_path = Path("/work/submission_policy.py")
    spec = importlib.util.spec_from_file_location("_myleetgpu_submission_policy", policy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the platform submission policy")
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    return policy.load_submission(
        source_path,
        expected_parameters=("probabilities", "output", "counts", "rows", "cols", "p"),
    )


def make_probabilities(rows: int, cols: int) -> torch.Tensor:
    row_indices = torch.arange(rows, dtype=torch.int64).unsqueeze(1)
    col_indices = torch.arange(cols, dtype=torch.int64).unsqueeze(0)
    ranks = (col_indices + row_indices * 17).remainder(cols) + 1
    weights = ranks.to(torch.float64)
    probabilities = (weights / weights.sum(dim=1, keepdim=True)).to(torch.float32)
    row_indexes = torch.arange(rows, dtype=torch.int64)
    maximum_cols = (cols - 1 - (row_indexes * 17).remainder(cols)).remainder(cols)
    other_sum = probabilities.to(torch.float64).sum(dim=1) - probabilities[
        row_indexes, maximum_cols
    ].to(torch.float64)
    probabilities[row_indexes, maximum_cols] = (1.0 - other_sum).to(torch.float32)
    return probabilities.contiguous().reshape(-1)


def make_reference(
    probabilities: torch.Tensor, rows: int, cols: int, p: float
) -> tuple[torch.Tensor, torch.Tensor]:
    ordered = torch.sort(probabilities.reshape(rows, cols), dim=1, descending=True).values
    cumulative = ordered.to(torch.float64).cumsum(dim=1)
    counts = ((cumulative < p).sum(dim=1) + 1).clamp(max=cols).to(torch.int32)
    ranks = torch.arange(cols, dtype=torch.int32).unsqueeze(0)
    expected = torch.where(ranks < counts.unsqueeze(1), ordered, torch.zeros_like(ordered))
    return expected.reshape(-1), counts


def run_benchmark(
    module: ModuleType,
    label: str,
    rows: int,
    cols: int,
    p: float,
    inner_repetitions: int,
    stream: torch.cuda.Stream,
) -> dict[str, Any]:
    probabilities = make_probabilities(rows, cols)
    expected, expected_counts = make_reference(probabilities, rows, cols, p)
    with torch.cuda.stream(stream):
        device_input = probabilities.cuda(non_blocking=False)
        output = torch.full_like(device_input, float("nan"))
        counts = torch.full((rows,), -1515870811, dtype=torch.int32, device="cuda")
    stream.synchronize()

    with torch.cuda.stream(stream):
        for _ in range(WARMUP * inner_repetitions):
            returned = module.solve(device_input, output, counts, rows, cols, p)
            if returned is not None:
                raise RuntimeError("solve must return None")
    stream.synchronize()
    if not torch.equal(device_input.cpu().view(torch.int32), probabilities.view(torch.int32)):
        raise RuntimeError("input was modified")
    actual = output.cpu()
    actual_counts = counts.cpu()
    if not torch.equal(actual.view(torch.int32), expected.view(torch.int32)) or not torch.equal(
        actual_counts, expected_counts
    ):
        raise RuntimeError("correctness check failed before timing")

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(ITERATIONS):
        with torch.cuda.stream(stream):
            start.record(stream)
            for _ in range(inner_repetitions):
                returned = module.solve(device_input, output, counts, rows, cols, p)
                if returned is not None:
                    raise RuntimeError("solve must return None")
            stop.record(stream)
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)) / inner_repetitions)
    return {
        "size": rows * cols,
        "label": label,
        "samples_ms": samples,
        "inner_repetitions": inner_repetitions,
    }


def main(argv: list[str]) -> int:
    trusted_load_submission = load_submission
    trusted_run_benchmark = run_benchmark
    trusted_is_compilation_error = is_compilation_error
    trusted_dumps = json.dumps
    trusted_print = print

    def emit(payload: dict[str, Any]) -> None:
        trusted_print(
            RESULT_PREFIX + trusted_dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )

    if argv[1:] != ["--mode", "benchmark"]:
        payload: dict[str, Any] = {
            "status": "runtime_error",
            "measurements": [],
            "protocol_version": PROTOCOL_VERSION,
            "message": "invalid arguments",
        }
        emit(payload)
        return 2
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(0)
        module = trusted_load_submission()
        stream = torch.cuda.Stream(device=0)
        measurements = [
            trusted_run_benchmark(module, label, rows, cols, p, repetitions, stream)
            for label, rows, cols, p, repetitions in CASES
        ]
        payload = {
            "status": "passed",
            "measurements": measurements,
            "protocol_version": PROTOCOL_VERSION,
        }
        returncode = 0
    except Exception as error:
        failed_to_compile = trusted_is_compilation_error(error)
        payload = {
            "status": "compile_error" if failed_to_compile else "runtime_error",
            "measurements": [],
            "protocol_version": PROTOCOL_VERSION,
            "message": "Triton compilation failed" if failed_to_compile else "benchmark failed",
        }
        returncode = 1
    emit(payload)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
