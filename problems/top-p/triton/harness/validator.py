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
BOUNDARY_MARGIN = 1.0e-5


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
    p: float
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
        expected_parameters=("probabilities", "output", "counts", "rows", "cols", "p"),
    )


def make_probabilities(test: TestCase) -> torch.Tensor:
    if test.pattern == "singleton":
        return torch.ones(test.rows * test.cols, dtype=torch.float32)

    row_indices = torch.arange(test.rows, dtype=torch.int64).unsqueeze(1)
    col_indices = torch.arange(test.cols, dtype=torch.int64).unsqueeze(0)
    ranks = (col_indices + row_indices * 17).remainder(test.cols) + 1
    weights = ranks.to(torch.float64)
    if test.pattern == "dominant":
        dominant = 4.0 * float(test.cols * test.cols)
        weights = torch.where(ranks == test.cols, dominant, weights)
    elif test.pattern == "quadratic":
        weights = weights * weights
    elif test.pattern == "near_uniform":
        weights = weights + float(test.cols)

    probabilities = (weights / weights.sum(dim=1, keepdim=True)).to(torch.float32)
    rows = torch.arange(test.rows, dtype=torch.int64)
    maximum_cols = (test.cols - 1 - (rows * 17).remainder(test.cols)).remainder(test.cols)
    other_sum = probabilities.to(torch.float64).sum(dim=1) - probabilities[rows, maximum_cols].to(
        torch.float64
    )
    probabilities[rows, maximum_cols] = (1.0 - other_sum).to(torch.float32)
    return probabilities.contiguous().reshape(-1)


def make_reference(
    probabilities: torch.Tensor, test: TestCase
) -> tuple[torch.Tensor, torch.Tensor]:
    ordered = torch.sort(probabilities.reshape(test.rows, test.cols), dim=1, descending=True).values
    if test.p >= 1.0:
        counts = torch.full((test.rows,), test.cols, dtype=torch.int32)
    else:
        cumulative = ordered.to(torch.float64).cumsum(dim=1)
        closest = float((cumulative - test.p).abs().min().item())
        if closest <= BOUNDARY_MARGIN:
            raise RuntimeError("invalid test boundary margin")
        counts = ((cumulative < test.p).sum(dim=1) + 1).clamp(max=test.cols).to(torch.int32)
    ranks = torch.arange(test.cols, dtype=torch.int32).unsqueeze(0)
    expected = torch.where(ranks < counts.unsqueeze(1), ordered, torch.zeros_like(ordered))
    return expected.reshape(-1), counts


def validate_observation(
    actual: torch.Tensor,
    actual_counts: torch.Tensor,
    expected: torch.Tensor,
    expected_counts: torch.Tensor,
    test: TestCase,
) -> str | None:
    if bool(((actual_counts < 1) | (actual_counts > test.cols)).any()) or not torch.equal(
        actual_counts, expected_counts
    ):
        return "count mismatch" if test.internal else "retained counts do not match reference"
    if test.p >= 1.0 and not bool((actual_counts == test.cols).all()):
        return "count mismatch" if test.internal else "p == 1 must retain every entry"

    if not torch.equal(actual.view(torch.int32), expected.view(torch.int32)):
        if test.internal:
            return "output mismatch"
        mismatch = int(
            torch.nonzero(actual.view(torch.int32) != expected.view(torch.int32), as_tuple=False)[
                0
            ].item()
        )
        row, rank = divmod(mismatch, test.cols)
        return f"output mismatch at row {row}, rank {rank}"

    actual_rows = actual.reshape(test.rows, test.cols)
    if not bool(torch.isfinite(actual_rows).all()) or bool((actual_rows < 0.0).any()):
        return "output mismatch" if test.internal else "output must be finite and nonnegative"
    rank_indices = torch.arange(test.cols, dtype=torch.int32).unsqueeze(0)
    tail_mask = rank_indices >= actual_counts.unsqueeze(1)
    if bool((actual_rows.view(torch.int32)[tail_mask] != 0).any()):
        return "output mismatch" if test.internal else "filtered tail must be exact positive zero"

    for row in range(test.rows):
        retained = int(actual_counts[row].item())
        prefix = actual_rows[row, :retained]
        if retained > 1 and bool((prefix[:-1] < prefix[1:]).any()):
            return "output mismatch" if test.internal else "retained prefix is not descending"
        if test.p < 1.0:
            cumulative = prefix.to(torch.float64).cumsum(dim=0)
            if float(cumulative[-1].item()) < test.p or (
                retained > 1 and float(cumulative[-2].item()) >= test.p
            ):
                return (
                    "prefix semantics mismatch"
                    if test.internal
                    else "output is not the shortest prefix reaching p"
                )
    return None


def run_case(module: ModuleType, test: TestCase, stream: torch.cuda.Stream) -> dict[str, Any]:
    probabilities = make_probabilities(test)
    expected, expected_counts = make_reference(probabilities, test)
    with torch.cuda.stream(stream):
        device_input = probabilities.cuda(non_blocking=False)
        output = torch.full_like(device_input, float("nan"))
        counts = torch.full((test.rows,), -1515870811, dtype=torch.int32, device="cuda")
        returned = module.solve(device_input, output, counts, test.rows, test.cols, test.p)
        if returned is not None:
            raise RuntimeError("solve must return None")
    stream.synchronize()
    first_output = output.cpu()
    first_counts = counts.cpu()

    with torch.cuda.stream(stream):
        output.fill_(-1.0)
        counts.fill_(1515870810)
        returned = module.solve(device_input, output, counts, test.rows, test.cols, test.p)
        if returned is not None:
            raise RuntimeError("solve must return None")
    stream.synchronize()
    second_output = output.cpu()
    second_counts = counts.cpu()
    observed_input = device_input.cpu()

    if not torch.equal(observed_input.view(torch.int32), probabilities.view(torch.int32)):
        message = "input modified" if test.internal else "input must remain unchanged"
        return {"name": test.name, "passed": False, "message": message}
    first_error = validate_observation(first_output, first_counts, expected, expected_counts, test)
    if first_error is not None:
        return {"name": test.name, "passed": False, "message": first_error}
    second_error = validate_observation(
        second_output, second_counts, expected, expected_counts, test
    )
    if second_error is not None:
        return {"name": test.name, "passed": False, "message": second_error}
    if not torch.equal(
        first_output.view(torch.int32), second_output.view(torch.int32)
    ) or not torch.equal(first_counts, second_counts):
        message = "call independence mismatch" if test.internal else "repeated calls differ"
        return {"name": test.name, "passed": False, "message": message}
    return {"name": test.name, "passed": True}


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
        TestCase("sample_singleton", 1, 1, 0.5, "singleton", False),
        TestCase("sample_non_power_of_two", 3, 5, 0.7, "ramp", False),
    ]
    if argv[2] == "full":
        tests.extend(
            [
                TestCase("internal_case_1", 7, 31, 0.55, "ramp", True),
                TestCase("internal_case_2", 257, 3, 0.8, "dominant", True),
                TestCase("internal_case_3", 19, 257, 0.02, "quadratic", True),
                TestCase("internal_case_4", 37, 513, 1.0, "near_uniform", True),
                TestCase("internal_case_5", 64, 1024, 0.9, "quadratic", True),
                TestCase("internal_case_6", 4096, 1024, 0.95, "ramp", True),
                TestCase("internal_case_7", 65536, 1, 0.5, "singleton", True),
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
