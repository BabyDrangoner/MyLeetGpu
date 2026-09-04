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
    k: int
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
        expected_parameters=("input", "values", "indices", "rows", "cols", "k"),
    )


def make_input(test: TestCase) -> torch.Tensor:
    count = test.rows * test.cols
    flat = torch.arange(count, dtype=torch.int64)
    row_indices = torch.div(flat, test.cols, rounding_mode="floor")
    col_indices = flat.remainder(test.cols)
    if test.pattern == "sequence":
        return ((col_indices * 37 + row_indices * 11).remainder(257) - 128).to(torch.float32) * 0.25
    if test.pattern == "ties":
        return ((col_indices * 5 + row_indices * 3).remainder(7) - 3).to(torch.float32) * 2.0
    if test.pattern == "descending":
        return (test.cols - col_indices).to(torch.float32) + row_indices.remainder(7).to(
            torch.float32
        ) * 0.001
    if test.pattern == "all_equal":
        return torch.full((count,), -17.25, dtype=torch.float32)
    if test.pattern == "extreme":
        selector = col_indices.remainder(8)
        output = torch.empty(count, dtype=torch.float32)
        output[selector == 0] = 10000.0
        output[selector == 1] = -10000.0
        output[selector == 2] = 9999.0
        output[selector == 3] = -9999.0
        output[selector == 4] = 0.0
        output[selector == 5] = 42.0
        output[selector == 6] = -42.0
        output[selector == 7] = row_indices[selector == 7].remainder(5).to(torch.float32)
        return output
    generator = torch.Generator(device="cpu").manual_seed(test.seed)
    return torch.empty(count, dtype=torch.float32).uniform_(-10000.0, 10000.0, generator=generator)


def validate_output(
    input_values: torch.Tensor,
    expected_values: torch.Tensor,
    actual_values: torch.Tensor,
    actual_indices: torch.Tensor,
    rows: int,
    cols: int,
    k: int,
) -> str | None:
    value_rows = actual_values.reshape(rows, k)
    index_rows = actual_indices.reshape(rows, k)
    valid_indices = (index_rows >= 0) & (index_rows < cols)
    if not bool(valid_indices.all()):
        position = torch.nonzero(~valid_indices, as_tuple=False)[0]
        return f"index out of range at row {int(position[0])}, rank {int(position[1])}"
    if k > 1:
        ordered_indices = torch.sort(index_rows.to(torch.int64), dim=1).values
        if bool((ordered_indices[:, 1:] == ordered_indices[:, :-1]).any()):
            row = int(
                torch.nonzero(
                    (ordered_indices[:, 1:] == ordered_indices[:, :-1]).any(dim=1),
                    as_tuple=False,
                )[0].item()
            )
            return f"duplicate index at row {row}"
    gathered = torch.gather(input_values.reshape(rows, cols), 1, index_rows.to(torch.int64))
    matching = torch.isfinite(value_rows) & (value_rows == gathered)
    if not bool(matching.all()):
        position = torch.nonzero(~matching, as_tuple=False)[0]
        return f"value does not match its index at row {int(position[0])}, rank {int(position[1])}"
    if k > 1 and bool((value_rows[:, 1:] > value_rows[:, :-1]).any()):
        row = int(
            torch.nonzero((value_rows[:, 1:] > value_rows[:, :-1]).any(dim=1), as_tuple=False)[
                0
            ].item()
        )
        return f"values are not in descending order at row {row}"
    expected_rows = expected_values.reshape(rows, k)
    if not bool((value_rows == expected_rows).all()):
        row = int(torch.nonzero((value_rows != expected_rows).any(dim=1), as_tuple=False)[0].item())
        return f"selected values do not match Top-K at row {row}"
    return None


def run_case(module: ModuleType, test: TestCase, stream: torch.cuda.Stream) -> dict[str, Any]:
    input_values = make_input(test)
    expected_values = torch.topk(
        input_values.reshape(test.rows, test.cols),
        test.k,
        dim=1,
        largest=True,
        sorted=True,
    ).values.reshape(-1)
    output_count = test.rows * test.k
    with torch.cuda.stream(stream):
        device_input = input_values.cuda(non_blocking=False)
        output_values = torch.empty(output_count, device="cuda", dtype=torch.float32)
        output_indices = torch.empty(output_count, device="cuda", dtype=torch.int32)
    stream.synchronize()

    observations: list[tuple[torch.Tensor, torch.Tensor]] = []
    for value_poison, index_poison in ((float("nan"), -1), (3.0e38, 2_000_000_000)):
        with torch.cuda.stream(stream):
            output_values.fill_(value_poison)
            output_indices.fill_(index_poison)
            returned = module.solve(
                device_input,
                output_values,
                output_indices,
                test.rows,
                test.cols,
                test.k,
            )
            if returned is not None:
                raise RuntimeError("solve must return None")
        stream.synchronize()
        observations.append((output_values.cpu(), output_indices.cpu()))

    if not torch.equal(device_input.cpu().view(torch.int32), input_values.view(torch.int32)):
        message = "input modified" if test.internal else "input must remain unchanged"
        return {"name": test.name, "passed": False, "message": message}

    for round_index, (actual_values, actual_indices) in enumerate(observations):
        error = validate_output(
            input_values,
            expected_values,
            actual_values,
            actual_indices,
            test.rows,
            test.cols,
            test.k,
        )
        if error is not None:
            if test.internal:
                message = "output mismatch"
            elif round_index == 0:
                message = error
            else:
                message = f"repeated call depends on prior output: {error}"
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
        TestCase("sample_1", 2, 5, 1, 73013, "sequence", False),
        TestCase("sample_2", 3, 9, 3, 73013, "ties", False),
    ]
    if argv[2] == "full":
        tests.extend(
            [
                TestCase("internal_case_1", 7, 31, 7, 271828, "random", True),
                TestCase("internal_case_2", 5, 64, 64, 271828, "descending", True),
                TestCase("internal_case_3", 4096, 17, 4, 314159, "ties", True),
                TestCase("internal_case_4", 65536, 1, 1, 314159, "all_equal", True),
                TestCase("internal_case_5", 4096, 1024, 64, 271828, "extreme", True),
                TestCase("internal_case_6", 257, 513, 1, 314159, "random", True),
                TestCase("internal_case_7", 127, 1023, 63, 271828, "ties", True),
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
    emit(trusted_result_payload(status, results))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
