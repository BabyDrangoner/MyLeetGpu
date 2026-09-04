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
WARMUP = 8
ITERATIONS = 20
CASES = (
    ("32768x32-k4", 32768, 32, 4, 8),
    ("8192x256-k16", 8192, 256, 16, 4),
    ("4096x1024-k64", 4096, 1024, 64, 1),
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
        expected_parameters=("input", "values", "indices", "rows", "cols", "k"),
    )


def check_output(
    input_values: torch.Tensor,
    expected_values: torch.Tensor,
    output_values: torch.Tensor,
    output_indices: torch.Tensor,
    rows: int,
    cols: int,
    k: int,
) -> None:
    value_rows = output_values.reshape(rows, k)
    index_rows = output_indices.reshape(rows, k)
    if not bool(((index_rows >= 0) & (index_rows < cols)).all()):
        raise RuntimeError("correctness check failed before timing")
    if k > 1:
        ordered_indices = torch.sort(index_rows.to(torch.int64), dim=1).values
        if bool((ordered_indices[:, 1:] == ordered_indices[:, :-1]).any()):
            raise RuntimeError("correctness check failed before timing")
    gathered = torch.gather(input_values.reshape(rows, cols), 1, index_rows.to(torch.int64))
    matches = torch.isfinite(value_rows) & (value_rows == gathered)
    descending = k == 1 or not bool((value_rows[:, 1:] > value_rows[:, :-1]).any())
    correct_values = bool((value_rows == expected_values.reshape(rows, k)).all())
    if not (bool(matches.all()) and descending and correct_values):
        raise RuntimeError("correctness check failed before timing")


def run_benchmark(
    module: ModuleType,
    label: str,
    rows: int,
    cols: int,
    k: int,
    inner_repetitions: int,
    generator: torch.Generator,
    stream: torch.cuda.Stream,
) -> dict[str, Any]:
    input_count = rows * cols
    output_count = rows * k
    input_values = torch.empty(input_count, dtype=torch.float32).uniform_(
        -10000.0, 10000.0, generator=generator
    )
    expected_values = torch.topk(
        input_values.reshape(rows, cols), k, dim=1, largest=True, sorted=True
    ).values.reshape(-1)
    with torch.cuda.stream(stream):
        device_input = input_values.cuda(non_blocking=False)
        output_values = torch.full(
            (output_count,), float("nan"), device="cuda", dtype=torch.float32
        )
        output_indices = torch.full((output_count,), -1, device="cuda", dtype=torch.int32)
    stream.synchronize()

    with torch.cuda.stream(stream):
        for _ in range(WARMUP * inner_repetitions):
            returned = module.solve(device_input, output_values, output_indices, rows, cols, k)
            if returned is not None:
                raise RuntimeError("solve must return None")
    stream.synchronize()
    if not torch.equal(device_input.cpu().view(torch.int32), input_values.view(torch.int32)):
        raise RuntimeError("input was modified")
    check_output(
        input_values,
        expected_values,
        output_values.cpu(),
        output_indices.cpu(),
        rows,
        cols,
        k,
    )

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(ITERATIONS):
        with torch.cuda.stream(stream):
            start.record(stream)
            for _ in range(inner_repetitions):
                returned = module.solve(device_input, output_values, output_indices, rows, cols, k)
                if returned is not None:
                    raise RuntimeError("solve must return None")
            stop.record(stream)
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)) / inner_repetitions)
    return {
        "size": input_count,
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
        generator = torch.Generator(device="cpu").manual_seed(20240902)
        measurements = [
            trusted_run_benchmark(module, label, rows, cols, k, repetitions, generator, stream)
            for label, rows, cols, k, repetitions in CASES
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
