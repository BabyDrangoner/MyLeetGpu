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
ABSOLUTE_TOLERANCE = 3.0e-5
RELATIVE_TOLERANCE = 3.0e-4
ROW_SUM_TOLERANCE = 1.0e-3


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
    rows: int
    cols: int
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
        expected_parameters=("input", "output", "rows", "cols"),
    )


def make_input(test: TestCase) -> torch.Tensor:
    count = test.rows * test.cols
    if test.pattern == "singleton":
        return torch.full((count,), 37.0, dtype=torch.float32)
    if test.pattern == "sequence":
        indices = torch.arange(count, dtype=torch.int64)
        rows = torch.div(indices, test.cols, rounding_mode="floor")
        return ((indices * 17 + rows * 3).remainder(29) - 14).to(torch.float32) * 0.5
    if test.pattern == "random":
        generator = torch.Generator(device="cpu").manual_seed(test.seed)
        return torch.empty(count, dtype=torch.float32).uniform_(-100.0, 100.0, generator=generator)
    row_indices = torch.arange(test.rows, dtype=torch.int64).repeat_interleave(test.cols)
    if test.pattern == "repeated":
        return ((row_indices.remainder(17) - 8).to(torch.float32) * 12.5).contiguous()
    col_indices = torch.arange(test.cols, dtype=torch.int64).repeat(test.rows)
    values = torch.zeros(count, dtype=torch.float32)
    selector = col_indices.remainder(6)
    values[selector == 0] = 100.0 - 0.125 * row_indices[selector == 0].remainder(5)
    values[selector == 1] = -100.0
    values[selector == 2] = 80.0
    values[selector == 3] = -80.0
    values[selector == 4] = 0.0
    values[selector == 5] = 99.0 - 0.25 * row_indices[selector == 5].remainder(3)
    return values


def run_case(module: ModuleType, test: TestCase, stream: torch.cuda.Stream) -> dict[str, Any]:
    input_values = make_input(test)
    expected = (
        torch.softmax(input_values.reshape(test.rows, test.cols).to(torch.float64), dim=1)
        .to(torch.float32)
        .reshape(-1)
    )
    with torch.cuda.stream(stream):
        device_input = input_values.cuda(non_blocking=False)
        output = torch.full_like(device_input, float("nan"))
        returned = module.solve(device_input, output, test.rows, test.cols)
        if returned is not None:
            raise RuntimeError("solve must return None")
    stream.synchronize()
    if not torch.equal(device_input.cpu().view(torch.int32), input_values.view(torch.int32)):
        message = "input modified" if test.internal else "input must remain unchanged"
        return {"name": test.name, "passed": False, "message": message}
    actual = output.cpu()
    matches = torch.isfinite(actual) & torch.isclose(
        actual,
        expected,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )
    actual_rows = actual.reshape(test.rows, test.cols)
    nonnegative = bool((actual_rows >= 0.0).all())
    normalized = bool(
        torch.isclose(
            actual_rows.to(torch.float64).sum(dim=1),
            torch.ones(test.rows, dtype=torch.float64),
            rtol=0.0,
            atol=ROW_SUM_TOLERANCE,
        ).all()
    )
    if bool(matches.all()) and nonnegative and normalized:
        return {"name": test.name, "passed": True}
    message = "output mismatch"
    if not test.internal:
        if not bool(matches.all()):
            mismatch = int(torch.nonzero(~matches, as_tuple=False)[0].item())
            row, col = divmod(mismatch, test.cols)
            message = f"output mismatch at row {row}, col {col}"
        else:
            message = "row probabilities must be non-negative and sum to one"
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
        TestCase("sample_1", 1, 1, 4242, "singleton", False),
        TestCase("sample_2", 3, 5, 4242, "sequence", False),
    ]
    if argv[2] == "full":
        tests.extend(
            [
                TestCase("internal_case_1", 7, 31, 424242, "random", True),
                TestCase("internal_case_2", 257, 3, 424242, "extreme", True),
                TestCase("internal_case_3", 19, 257, 8675309, "repeated", True),
                TestCase("internal_case_4", 64, 4096, 8675309, "extreme", True),
                TestCase("internal_case_5", 1024, 513, 424242, "random", True),
                TestCase("internal_case_6", 65536, 1, 8675309, "singleton", True),
                TestCase("internal_case_7", 4096, 4095, 8675309, "random", True),
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
