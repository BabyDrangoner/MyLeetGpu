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
ATOL = 4.0e-4
RTOL = 4.0e-4
CLASS_NAME = "MultiHeadAttention"
INIT_PARAMETERS = ("numHeads", "qWeight", "kWeight", "vWeight", "outputWeight")
FORWARD_PARAMETERS = ("X", "isCasual")


class SubmissionCompileError(RuntimeError):
    pass


def is_compilation_error(error: BaseException) -> bool:
    return isinstance(error, SyntaxError | SubmissionCompileError) or type(error).__name__ in {
        "SubmissionPolicyError",
        "TorchSubmissionPolicyError",
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
    sequence_length: int
    embed_dim: int
    num_heads: int
    is_casual: bool
    seed: int
    pattern: str
    internal: bool


PUBLIC_CASES = (
    ("sample_noncausal", 1, 3, 8, 2, False, 230901, "sequence"),
    ("sample_causal", 2, 7, 32, 4, True, 230901, "random"),
    ("sample_odd_head_dim", 1, 5, 21, 3, False, 230901, "head_markers"),
)
INTERNAL_CASES = (
    ("internal_case_1", 1, 1, 1, 1, True, 314159, "constant"),
    ("internal_case_2", 2, 1, 12, 3, False, 314159, "random"),
    ("internal_case_3", 1, 9, 20, 5, True, 271828, "random"),
    ("internal_case_4", 1, 33, 168, 7, True, 271828, "random"),
    ("internal_case_5", 2, 6, 48, 6, False, 1618033, "projection_markers"),
    ("internal_case_6", 1, 17, 256, 4, True, 1618033, "extreme_logits"),
    ("internal_case_7", 2, 65, 256, 8, False, 314159, "random"),
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
        expected_class_name=CLASS_NAME,
        expected_init_parameters=INIT_PARAMETERS,
        expected_forward_parameters=FORWARD_PARAMETERS,
    )


def submission_class(module: ModuleType) -> type:
    candidate = module.__dict__.get(CLASS_NAME)
    if not isinstance(candidate, type):
        raise SubmissionCompileError(f"{CLASS_NAME} must be a class")
    return candidate


def patterned_weight(embed_dim: int, multiplier: int, offset: int) -> torch.Tensor:
    indexes = torch.arange(embed_dim * embed_dim, dtype=torch.int64).reshape(embed_dim, embed_dim)
    denominator = 14.0 * math.sqrt(embed_dim)
    return (((indexes * multiplier + offset) % 29).to(torch.float32) - 14.0) / denominator


def make_inputs(
    test: TestCase,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    shape = (test.batch, test.sequence_length, test.embed_dim)
    generator = torch.Generator(device="cpu").manual_seed(test.seed)
    weight_scale = 0.8 / math.sqrt(test.embed_dim)

    if test.pattern == "constant":
        X = torch.full(shape, 0.25, dtype=torch.float32)
        q_weight = torch.full((test.embed_dim, test.embed_dim), 0.5, dtype=torch.float32)
        k_weight = torch.full((test.embed_dim, test.embed_dim), -0.75, dtype=torch.float32)
        v_weight = torch.full((test.embed_dim, test.embed_dim), 0.375, dtype=torch.float32)
        output_weight = torch.full((test.embed_dim, test.embed_dim), 1.25, dtype=torch.float32)
    else:
        X = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.6
        q_weight = (
            torch.randn((test.embed_dim, test.embed_dim), generator=generator, dtype=torch.float32)
            * weight_scale
        )
        k_weight = (
            torch.randn((test.embed_dim, test.embed_dim), generator=generator, dtype=torch.float32)
            * weight_scale
        )
        v_weight = (
            torch.randn((test.embed_dim, test.embed_dim), generator=generator, dtype=torch.float32)
            * weight_scale
        )
        output_weight = (
            torch.randn((test.embed_dim, test.embed_dim), generator=generator, dtype=torch.float32)
            * weight_scale
        )

        if test.pattern == "sequence":
            indexes = torch.arange(math.prod(shape), dtype=torch.int64)
            X = ((indexes % 23).to(torch.float32) - 11.0).reshape(shape) / 11.0
        elif test.pattern == "head_markers":
            channels = torch.arange(test.embed_dim, dtype=torch.float32).reshape(1, 1, -1)
            positions = torch.arange(test.sequence_length, dtype=torch.float32).reshape(1, -1, 1)
            X = X + channels * 0.015 + positions * 0.025
        elif test.pattern == "projection_markers":
            q_weight = patterned_weight(test.embed_dim, 3, 1)
            k_weight = patterned_weight(test.embed_dim, 5, 7)
            v_weight = patterned_weight(test.embed_dim, 11, 2)
            output_weight = patterned_weight(test.embed_dim, 13, 9)
        elif test.pattern == "extreme_logits":
            X = X * 8.0
            q_weight = q_weight * 2.0
            k_weight = k_weight * 2.0
        elif test.pattern != "random":
            raise RuntimeError("unknown input pattern")

    expected = reference_attention(
        X,
        test.is_casual,
        test.num_heads,
        q_weight,
        k_weight,
        v_weight,
        output_weight,
    )
    return X, q_weight, k_weight, v_weight, output_weight, expected


def reference_attention(
    X: torch.Tensor,
    is_casual: bool,
    num_heads: int,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    output_weight: torch.Tensor,
) -> torch.Tensor:
    batch, sequence_length, embed_dim = X.shape
    head_dim = embed_dim // num_heads
    X64 = X.to(torch.float64)
    query = torch.matmul(X64, q_weight.to(torch.float64))
    key = torch.matmul(X64, k_weight.to(torch.float64))
    value = torch.matmul(X64, v_weight.to(torch.float64))
    query = query.reshape(batch, sequence_length, num_heads, head_dim).transpose(1, 2)
    key = key.reshape(batch, sequence_length, num_heads, head_dim).transpose(1, 2)
    value = value.reshape(batch, sequence_length, num_heads, head_dim).transpose(1, 2)
    scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim**-0.5)
    if is_casual:
        causal_mask = torch.ones((sequence_length, sequence_length), dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    context = torch.matmul(probabilities, value)
    concatenated = context.transpose(1, 2).contiguous().reshape(batch, sequence_length, embed_dim)
    return torch.matmul(concatenated, output_weight.to(torch.float64)).to(torch.float32)


def validate_output_contract(
    output: object,
    tensors: tuple[torch.Tensor, ...],
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    X = tensors[0]
    if not isinstance(output, torch.Tensor):
        raise RuntimeError("forward must return a torch.Tensor")
    if output.device != X.device or output.device.type != "cuda" or output.device.index != 0:
        raise RuntimeError("forward must return a CUDA tensor on device 0")
    if output.dtype != torch.float32:
        raise RuntimeError("forward must return torch.float32")
    if tuple(output.shape) != expected_shape:
        raise RuntimeError("forward returned an incorrect shape")
    input_storage_pointers = {tensor.untyped_storage().data_ptr() for tensor in tensors}
    if output.untyped_storage().data_ptr() in input_storage_pointers:
        raise RuntimeError("forward output must not alias X or a weight tensor")
    return output


def run_case(module: ModuleType, test: TestCase, stream: torch.cuda.Stream) -> dict[str, Any]:
    X, q_weight, k_weight, v_weight, output_weight, expected = make_inputs(test)
    attention_type = submission_class(module)
    with torch.inference_mode(), torch.cuda.stream(stream):
        device_tensors = tuple(
            tensor.cuda(non_blocking=False)
            for tensor in (X, q_weight, k_weight, v_weight, output_weight)
        )
        snapshots = tuple(tensor.clone() for tensor in device_tensors)
        device_X, device_q, device_k, device_v, device_output = device_tensors
        instance = attention_type(
            test.num_heads,
            device_q,
            device_k,
            device_v,
            device_output,
        )
        returned = instance.forward(device_X, test.is_casual)
        output = validate_output_contract(returned, device_tensors, tuple(expected.shape))
    stream.synchronize()

    for current, snapshot in zip(device_tensors, snapshots, strict=True):
        if not torch.equal(current, snapshot):
            raise RuntimeError("forward and constructor must not modify X or weight tensors")

    actual = output.detach().cpu()
    if not bool(torch.all(torch.isfinite(actual))):
        return {"name": test.name, "passed": False, "message": "output contains non-finite values"}
    close = torch.isclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False)
    if bool(torch.all(close)):
        return {"name": test.name, "passed": True}
    message = "output mismatch"
    if not test.internal:
        first_index = tuple(
            int(value) for value in torch.nonzero(~close, as_tuple=False)[0].tolist()
        )
        max_error = float(torch.max(torch.abs(actual - expected)).item())
        message = f"output mismatch near index {first_index}; max abs error {max_error:.6g}"
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
