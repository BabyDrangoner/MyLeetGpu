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
WARMUP = 4
ITERATIONS = 15
ATOL = 0.03
RTOL = 0.005
CASES = (
    ("128x128x128", 128, 128, 128, 1),
    ("256x256x256", 256, 256, 256, 1),
    ("512x512x512", 512, 512, 512, 1),
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
        expected_parameters=("a", "b", "c", "m", "k", "n"),
    )


def run_benchmark(
    module: ModuleType,
    label: str,
    m: int,
    k: int,
    n: int,
    inner_repetitions: int,
    generator: torch.Generator,
    stream: torch.cuda.Stream,
) -> dict[str, Any]:
    a = torch.empty(m * k, dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)
    b = torch.empty(k * n, dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)
    expected = (
        (a.reshape(m, k).to(torch.float64) @ b.reshape(k, n).to(torch.float64))
        .to(torch.float32)
        .reshape(-1)
    )
    with torch.cuda.stream(stream):
        device_a = a.cuda(non_blocking=False)
        device_b = b.cuda(non_blocking=False)
        output = torch.full((m * n,), float("nan"), dtype=torch.float32, device="cuda")
    stream.synchronize()

    with torch.cuda.stream(stream):
        for _ in range(WARMUP * inner_repetitions):
            returned = module.solve(device_a, device_b, output, m, k, n)
            if returned is not None:
                raise RuntimeError("solve must return None")
    stream.synchronize()
    a_unchanged = torch.equal(device_a.cpu().view(torch.int32), a.view(torch.int32))
    b_unchanged = torch.equal(device_b.cpu().view(torch.int32), b.view(torch.int32))
    if not a_unchanged or not b_unchanged:
        raise RuntimeError("input tensors were modified")
    if not torch.allclose(output.cpu(), expected, atol=ATOL, rtol=RTOL, equal_nan=False):
        raise RuntimeError("correctness check failed before timing")

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(ITERATIONS):
        with torch.cuda.stream(stream):
            output.fill_(float("nan"))
            start.record(stream)
            for _ in range(inner_repetitions):
                returned = module.solve(device_a, device_b, output, m, k, n)
                if returned is not None:
                    raise RuntimeError("solve must return None")
            stop.record(stream)
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)) / inner_repetitions)
        a_unchanged = torch.equal(device_a.cpu().view(torch.int32), a.view(torch.int32))
        b_unchanged = torch.equal(device_b.cpu().view(torch.int32), b.view(torch.int32))
        if not a_unchanged or not b_unchanged:
            raise RuntimeError("input tensors were modified during timing")
        if not torch.allclose(output.cpu(), expected, atol=ATOL, rtol=RTOL, equal_nan=False):
            raise RuntimeError("correctness check failed during timing")
    return {
        "size": m * k + k * n + m * n,
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
        generator = torch.Generator(device="cpu").manual_seed(987654)
        measurements = [
            trusted_run_benchmark(module, label, m, k, n, repetitions, generator, stream)
            for label, m, k, n, repetitions in CASES
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
