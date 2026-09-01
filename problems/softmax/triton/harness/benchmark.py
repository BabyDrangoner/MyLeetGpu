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
ABSOLUTE_TOLERANCE = 3.0e-5
RELATIVE_TOLERANCE = 3.0e-4
ROW_SUM_TOLERANCE = 1.0e-3
CASES = (
    ("8192x128", 8192, 128, 16),
    ("4096x1024", 4096, 1024, 8),
    ("2048x4096", 2048, 4096, 4),
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
        expected_parameters=("input", "output", "rows", "cols"),
    )


def run_benchmark(
    module: ModuleType,
    label: str,
    rows: int,
    cols: int,
    inner_repetitions: int,
    generator: torch.Generator,
    stream: torch.cuda.Stream,
) -> dict[str, Any]:
    count = rows * cols
    input_values = torch.empty(count, dtype=torch.float32).uniform_(
        -20.0, 20.0, generator=generator
    )
    expected = (
        torch.softmax(input_values.reshape(rows, cols).to(torch.float64), dim=1)
        .to(torch.float32)
        .reshape(-1)
    )
    with torch.cuda.stream(stream):
        device_input = input_values.cuda(non_blocking=False)
        output = torch.full_like(device_input, float("nan"))
    stream.synchronize()

    with torch.cuda.stream(stream):
        for _ in range(WARMUP * inner_repetitions):
            returned = module.solve(device_input, output, rows, cols)
            if returned is not None:
                raise RuntimeError("solve must return None")
    stream.synchronize()
    if not torch.equal(device_input.cpu().view(torch.int32), input_values.view(torch.int32)):
        raise RuntimeError("input was modified")
    actual = output.cpu()
    actual_rows = actual.reshape(rows, cols)
    elementwise = bool(
        (
            torch.isfinite(actual)
            & torch.isclose(
                actual,
                expected,
                rtol=RELATIVE_TOLERANCE,
                atol=ABSOLUTE_TOLERANCE,
            )
        ).all()
    )
    nonnegative = bool((actual_rows >= 0.0).all())
    normalized = bool(
        torch.isclose(
            actual_rows.to(torch.float64).sum(dim=1),
            torch.ones(rows, dtype=torch.float64),
            rtol=0.0,
            atol=ROW_SUM_TOLERANCE,
        ).all()
    )
    if not (elementwise and nonnegative and normalized):
        raise RuntimeError("correctness check failed before timing")

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(ITERATIONS):
        with torch.cuda.stream(stream):
            start.record(stream)
            for _ in range(inner_repetitions):
                returned = module.solve(device_input, output, rows, cols)
                if returned is not None:
                    raise RuntimeError("solve must return None")
            stop.record(stream)
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)) / inner_repetitions)
    return {
        "size": count,
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
        generator = torch.Generator(device="cpu").manual_seed(20240901)
        measurements = [
            trusted_run_benchmark(module, label, rows, cols, repetitions, generator, stream)
            for label, rows, cols, repetitions in CASES
        ]
        payload: dict[str, Any] = {
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
