from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

RESULT_PREFIX = "MYLEETGPU_RESULT="


class SubmissionCompileError(RuntimeError):
    pass


def is_compilation_error(error: BaseException) -> bool:
    if isinstance(error, SyntaxError | SubmissionCompileError):
        return True
    names = {"CompilationError", "CompileTimeAssertionFailure", "OutOfResources", "PTXASError"}
    return any(
        cls.__name__ in names and cls.__module__.startswith("triton") for cls in type(error).__mro__
    )


def public_error(prefix: str, error: BaseException) -> str:
    first_line = next((line.strip() for line in str(error).splitlines() if line.strip()), "error")
    first_line = first_line.replace("/work/source.py", "source.py").replace(
        "/work/platform.py", "platform.py"
    )
    return f"{prefix}: {first_line[:300]}"


@dataclass(frozen=True)
class TestCase:
    name: str
    n: int
    seed: int
    pattern: str
    internal: bool


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
        expected_parameters=("input", "output", "n"),
    )


def make_input(test: TestCase) -> torch.Tensor:
    if test.pattern == "extreme_low":
        return torch.full((test.n,), torch.finfo(torch.float32).min, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(test.seed)
    if test.pattern == "all_negative":
        return torch.empty(test.n, dtype=torch.float32).uniform_(
            -1000.0, -0.001, generator=generator
        )
    if test.pattern == "extremes":
        values = torch.empty(test.n, dtype=torch.float32).uniform_(
            -1000.0, 1000.0, generator=generator
        )
        values[0] = torch.finfo(torch.float32).min
        values[test.n // 2] = torch.finfo(torch.float32).max
        return values
    if test.pattern == "alternating":
        indexes = torch.arange(test.n, dtype=torch.int64)
        magnitude = (indexes.remainder(257) + 1).to(torch.float32)
        return torch.where(indexes.remainder(2) == 0, -magnitude, magnitude)
    if test.pattern == "signed_zero":
        return torch.full((test.n,), -0.0, dtype=torch.float32)
    return torch.empty(test.n, dtype=torch.float32).uniform_(
        -1000000.0, 1000000.0, generator=generator
    )


def run_case(module: ModuleType, test: TestCase, stream: torch.cuda.Stream) -> dict[str, Any]:
    input_values = make_input(test)
    expected = float(input_values.max().item())
    with torch.cuda.stream(stream):
        device_input = input_values.cuda(non_blocking=False)
        output = torch.full(
            (1,), torch.finfo(torch.float32).max, device="cuda", dtype=torch.float32
        )
        returned = module.solve(device_input, output, test.n)
        if returned is not None:
            raise RuntimeError("solve must return None")
    stream.synchronize()
    if not torch.equal(device_input.cpu().view(torch.int32), input_values.view(torch.int32)):
        message = "input modified" if test.internal else "input must remain unchanged"
        return {"name": test.name, "passed": False, "message": message}
    actual = float(output.cpu()[0].item())
    if not math.isnan(actual) and actual == expected:
        return {"name": test.name, "passed": True}
    message = "output mismatch" if test.internal else "maximum does not match reference"
    return {"name": test.name, "passed": False, "message": message}


def result_payload(status: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(case.get("passed") is True for case in cases)
    return {
        "status": status,
        "cases": cases,
        "summary": {"total": len(cases), "passed": passed, "failed": len(cases) - passed},
    }


def main(argv: list[str]) -> int:
    trusted_load_submission = load_submission
    trusted_run_case = run_case
    trusted_is_compilation_error = is_compilation_error
    trusted_public_error = public_error
    trusted_result_payload = result_payload
    trusted_dumps = json.dumps
    trusted_print = print

    def emit(payload: dict[str, Any]) -> None:
        trusted_print(
            RESULT_PREFIX + trusted_dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )

    if len(argv) != 3 or argv[1] != "--mode" or argv[2] not in {"public", "full"}:
        payload = trusted_result_payload(
            "runtime_error",
            [{"name": "configuration", "passed": False, "message": "invalid arguments"}],
        )
        emit(payload)
        return 2

    tests = [
        TestCase("sample_1", 1, 91453, "extreme_low", False),
        TestCase("sample_2", 1000, 91453, "all_negative", False),
    ]
    if argv[2] == "full":
        tests.extend(
            [
                TestCase("internal_case_1", 31, 161803, "extremes", True),
                TestCase("internal_case_2", 4097, 161803, "all_negative", True),
                TestCase("internal_case_3", 65537, 141421, "alternating", True),
                TestCase("internal_case_4", 513, 141421, "signed_zero", True),
                TestCase("internal_case_5", 1048576, 141421, "signed_random", True),
            ]
        )

    results: list[dict[str, Any]] = []
    runtime_error = False
    compile_error = False
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(0)
        module = trusted_load_submission()
        stream = torch.cuda.Stream(device=0)
        for test in tests:
            try:
                results.append(trusted_run_case(module, test, stream))
            except Exception as error:
                failed_to_compile = trusted_is_compilation_error(error)
                compile_error = compile_error or failed_to_compile
                if argv[2] == "full":
                    message = (
                        "Triton compilation failed"
                        if failed_to_compile
                        else "CUDA execution failed"
                    )
                else:
                    prefix = (
                        "Triton compilation failed"
                        if failed_to_compile
                        else "CUDA execution failed"
                    )
                    message = trusted_public_error(prefix, error)
                results.append({"name": test.name, "passed": False, "message": message})
                runtime_error = runtime_error or not failed_to_compile
                break
    except Exception as error:
        failed_to_compile = trusted_is_compilation_error(error)
        compile_error = failed_to_compile
        prefix = "Triton compilation failed" if failed_to_compile else "CUDA setup failed"
        message = prefix if argv[2] == "full" else trusted_public_error(prefix, error)
        results.append({"name": "setup", "passed": False, "message": message})
        runtime_error = not failed_to_compile

    all_passed = len(results) == len(tests) and all(case.get("passed") is True for case in results)
    if compile_error:
        status = "compile_error"
    elif runtime_error:
        status = "runtime_error"
    else:
        status = "passed" if all_passed else "wrong_answer"
    payload = trusted_result_payload(status, results)
    emit(payload)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
