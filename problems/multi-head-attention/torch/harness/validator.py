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
ATOL = 2.0e-4
RTOL = 2.0e-4


class SubmissionCompileError(RuntimeError):
    pass


def is_compilation_error(error: BaseException) -> bool:
    return isinstance(error, SyntaxError | SubmissionCompileError) or type(error).__name__ in {
        "SubmissionPolicyError",
    }


def public_error(prefix: str, error: BaseException) -> str:
    first_line = next((line.strip() for line in str(error).splitlines() if line.strip()), "error")
    first_line = first_line.replace("/work/source.py", "source.py").replace(
        "/work/platform.py", "platform.py"
    )
    return f"{prefix}: {first_line[:300]}"


@dataclass(frozen=True)
class TestCase:
    name: str
    batch: int
    heads: int
    query_length: int
    key_length: int
    head_dim: int
    seed: int
    pattern: str
    mask_pattern: str
    internal: bool


PUBLIC_CASES = (
    ("sample_handcrafted", 1, 2, 2, 3, 4, 230901, "sequence", "all"),
    ("sample_padding_mask", 2, 4, 5, 7, 8, 230901, "random", "prefix"),
    ("sample_causal_mask", 1, 3, 9, 9, 16, 230901, "random", "causal"),
)
INTERNAL_CASES = (
    ("internal_case_1", 1, 1, 1, 1, 1, 314159, "constant", "all"),
    ("internal_case_2", 2, 1, 1, 17, 3, 314159, "random", "prefix"),
    ("internal_case_3", 2, 4, 7, 11, 16, 271828, "random", "single_key"),
    ("internal_case_4", 1, 7, 33, 33, 24, 271828, "random", "causal"),
    ("internal_case_5", 1, 4, 5, 6, 8, 1618033, "head_markers", "staggered"),
    ("internal_case_6", 1, 4, 17, 19, 64, 1618033, "extreme_logits", "staggered"),
    ("internal_case_7", 2, 8, 65, 97, 32, 314159, "random", "prefix"),
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
        expected_parameters=("query", "key", "value", "attention_mask"),
    )


def make_mask(test: TestCase) -> torch.Tensor:
    shape = (test.batch, 1, test.query_length, test.key_length)
    if test.mask_pattern == "all":
        return torch.ones(shape, dtype=torch.bool)
    if test.mask_pattern == "causal":
        if test.query_length != test.key_length:
            raise RuntimeError("causal test configuration must be square")
        causal = torch.ones((test.query_length, test.key_length), dtype=torch.bool).tril()
        return (
            causal.reshape(1, 1, test.query_length, test.key_length)
            .expand(test.batch, 1, test.query_length, test.key_length)
            .contiguous()
        )
    if test.mask_pattern == "prefix":
        mask = torch.zeros(shape, dtype=torch.bool)
        for batch_index in range(test.batch):
            valid = max(1, test.key_length - ((batch_index * 3 + 1) % test.key_length))
            mask[batch_index, 0, :, :valid] = True
        return mask
    if test.mask_pattern == "single_key":
        mask = torch.zeros(shape, dtype=torch.bool)
        for batch_index in range(test.batch):
            for query_index in range(test.query_length):
                key_index = (batch_index + query_index * 3) % test.key_length
                mask[batch_index, 0, query_index, key_index] = True
        return mask
    if test.mask_pattern == "staggered":
        batches = torch.arange(test.batch).reshape(test.batch, 1, 1, 1)
        queries = torch.arange(test.query_length).reshape(1, 1, test.query_length, 1)
        keys = torch.arange(test.key_length).reshape(1, 1, 1, test.key_length)
        mask = ((batches + queries * 5 + keys * 3) % 7) < 4
        mask[..., 0] = True
        return mask.contiguous()
    raise RuntimeError("unknown mask pattern")


def make_inputs(
    test: TestCase,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query_shape = (test.batch, test.heads, test.query_length, test.head_dim)
    key_shape = (test.batch, test.heads, test.key_length, test.head_dim)
    generator = torch.Generator(device="cpu").manual_seed(test.seed)
    if test.pattern == "constant":
        query = torch.full(query_shape, 0.25, dtype=torch.float32)
        key = torch.full(key_shape, -0.5, dtype=torch.float32)
        value = torch.full(key_shape, 0.375, dtype=torch.float32)
    elif test.pattern == "sequence":
        query_indexes = torch.arange(math.prod(query_shape), dtype=torch.int64)
        key_indexes = torch.arange(math.prod(key_shape), dtype=torch.int64)
        query = ((query_indexes % 17).to(torch.float32) - 8.0).reshape(query_shape) / 8.0
        key = (((key_indexes * 3) % 19).to(torch.float32) - 9.0).reshape(key_shape) / 9.0
        value = (((key_indexes * 5) % 23).to(torch.float32) - 11.0).reshape(key_shape) / 11.0
    else:
        query = torch.randn(query_shape, generator=generator, dtype=torch.float32) * 0.7
        key = torch.randn(key_shape, generator=generator, dtype=torch.float32) * 0.7
        value = torch.randn(key_shape, generator=generator, dtype=torch.float32) * 0.5
        if test.pattern == "head_markers":
            offsets = torch.arange(test.heads, dtype=torch.float32).reshape(1, -1, 1, 1)
            query = query + offsets * 0.07
            key = key - offsets * 0.11
            value = value + offsets * 0.25
        elif test.pattern == "extreme_logits":
            query = query * 32.0
            key = key * 32.0
        elif test.pattern != "random":
            raise RuntimeError("unknown input pattern")
    attention_mask = make_mask(test)
    expected = reference_attention(query, key, value, attention_mask)
    return query, key, value, attention_mask, expected


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    query64 = query.to(torch.float64)
    key64 = key.to(torch.float64)
    value64 = value.to(torch.float64)
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query64, key64.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~attention_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value64).to(torch.float32)


def validate_output_contract(
    output: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise RuntimeError("solve must return a torch.Tensor")
    if output.device != query.device or output.device.type != "cuda" or output.device.index != 0:
        raise RuntimeError("solve must return a CUDA tensor on device 0")
    if output.dtype != torch.float32:
        raise RuntimeError("solve must return torch.float32")
    if tuple(output.shape) != expected_shape:
        raise RuntimeError("solve returned an incorrect shape")
    input_storage_pointers = {
        tensor.untyped_storage().data_ptr() for tensor in (query, key, value, attention_mask)
    }
    if output.untyped_storage().data_ptr() in input_storage_pointers:
        raise RuntimeError("solve output must not alias an input")
    return output


def run_case(module: ModuleType, test: TestCase, stream: torch.cuda.Stream) -> dict[str, Any]:
    query, key, value, attention_mask, expected = make_inputs(test)
    with torch.inference_mode(), torch.cuda.stream(stream):
        device_query = query.cuda(non_blocking=False)
        device_key = key.cuda(non_blocking=False)
        device_value = value.cuda(non_blocking=False)
        device_mask = attention_mask.cuda(non_blocking=False)
        snapshots = tuple(
            tensor.clone() for tensor in (device_query, device_key, device_value, device_mask)
        )
        returned = module.solve(device_query, device_key, device_value, device_mask)
        output = validate_output_contract(
            returned,
            device_query,
            device_key,
            device_value,
            device_mask,
            tuple(expected.shape),
        )
    stream.synchronize()
    for current, snapshot in zip(
        (device_query, device_key, device_value, device_mask), snapshots, strict=True
    ):
        if not torch.equal(current, snapshot):
            raise RuntimeError("solve must not modify its inputs")
    actual = output.detach().cpu()
    if not bool(torch.all(torch.isfinite(actual))):
        return {"name": test.name, "passed": False, "message": "output contains non-finite values"}
    close = torch.isclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False)
    if bool(torch.all(close)):
        return {"name": test.name, "passed": True}
    message = "output mismatch"
    if not test.internal:
        mismatch = int(torch.nonzero(~close, as_tuple=False)[0].flatten()[0].item())
        max_error = float(torch.max(torch.abs(actual - expected)).item())
        message = f"output mismatch near dimension index {mismatch}; max abs error {max_error:.6g}"
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
        emit(
            trusted_result_payload(
                "runtime_error",
                [{"name": "configuration", "passed": False, "message": "invalid arguments"}],
            )
        )
        return 2

    tests = [TestCase(*case, False) for case in PUBLIC_CASES]
    if argv[2] == "full":
        tests.extend(TestCase(*case, True) for case in INTERNAL_CASES)

    results: list[dict[str, Any]] = []
    runtime_error = False
    compile_error = False
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
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
                        "PyTorch submission policy failed"
                        if failed_to_compile
                        else "CUDA execution failed"
                    )
                else:
                    prefix = (
                        "PyTorch submission policy failed"
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
        prefix = "PyTorch submission policy failed" if failed_to_compile else "CUDA setup failed"
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
