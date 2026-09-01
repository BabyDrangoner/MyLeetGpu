from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

RESULT_PREFIX = "MYLEETGPU_RESULT="
ATOL = 0.03
RTOL = 0.005


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
    m: int
    k: int
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
        expected_parameters=("a", "b", "c", "m", "k", "n"),
    )


def make_inputs(test: TestCase) -> tuple[torch.Tensor, torch.Tensor]:
    a_count = test.m * test.k
    b_count = test.k * test.n
    if test.pattern == "sequence":
        a = ((torch.arange(a_count, dtype=torch.int64).remainder(7) - 3) * 0.25).to(torch.float32)
        indices = torch.arange(b_count, dtype=torch.int64)
        b = (((indices * 3).remainder(11) - 5) * 0.2).to(torch.float32)
        return a, b
    generator = torch.Generator(device="cpu").manual_seed(test.seed)
    a = torch.empty(a_count, dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)
    b = torch.empty(b_count, dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)
    return a, b


def reference_multiply(
    a: torch.Tensor,
    b: torch.Tensor,
    m: int,
    k: int,
    n: int,
) -> torch.Tensor:
    return (
        (a.reshape(m, k).to(torch.float64) @ b.reshape(k, n).to(torch.float64))
        .to(torch.float32)
        .reshape(-1)
    )


def run_case(module: ModuleType, test: TestCase, stream: torch.cuda.Stream) -> dict[str, Any]:
    a, b = make_inputs(test)
    expected = reference_multiply(a, b, test.m, test.k, test.n)
    with torch.cuda.stream(stream):
        device_a = a.cuda(non_blocking=False)
        device_b = b.cuda(non_blocking=False)
        output = torch.full((test.m * test.n,), float("nan"), dtype=torch.float32, device="cuda")
        returned = module.solve(device_a, device_b, output, test.m, test.k, test.n)
        if returned is not None:
            raise RuntimeError("solve must return None")
    stream.synchronize()
    a_unchanged = torch.equal(device_a.cpu().view(torch.int32), a.view(torch.int32))
    b_unchanged = torch.equal(device_b.cpu().view(torch.int32), b.view(torch.int32))
    if not a_unchanged or not b_unchanged:
        message = "input modified" if test.internal else "input tensors must remain unchanged"
        return {"name": test.name, "passed": False, "message": message}
    actual = output.cpu()
    if torch.allclose(actual, expected, atol=ATOL, rtol=RTOL, equal_nan=False):
        return {"name": test.name, "passed": True}
    message = "output mismatch"
    if not test.internal:
        close = torch.isclose(actual, expected, atol=ATOL, rtol=RTOL, equal_nan=False)
        mismatch = int(torch.nonzero(~close, as_tuple=False)[0].item())
        message = f"output mismatch at flattened index {mismatch}"
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
        TestCase("sample_1", 2, 3, 2, 4242, "sequence", False),
        TestCase("sample_2", 3, 5, 4, 4242, "sequence", False),
    ]
    if argv[2] == "full":
        tests.extend(
            [
                TestCase("internal_case_1", 17, 19, 23, 271828, "random", True),
                TestCase("internal_case_2", 1, 257, 37, 314159, "random", True),
                TestCase("internal_case_3", 73, 31, 1, 271828, "random", True),
                TestCase("internal_case_4", 64, 513, 96, 314159, "random", True),
                TestCase("internal_case_5", 193, 127, 257, 271828, "random", True),
                TestCase("internal_case_6", 3, 4096, 5, 314159, "random", True),
                TestCase("internal_case_7", 4096, 7, 3, 271828, "random", True),
                TestCase("internal_case_8", 5, 9, 4096, 314159, "random", True),
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
