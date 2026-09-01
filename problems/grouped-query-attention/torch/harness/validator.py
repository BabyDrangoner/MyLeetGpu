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
ATOL = 3.0e-4
RTOL = 3.0e-4


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
    embedding_dim: int
    query_heads: int
    key_value_heads: int
    seed: int
    pattern: str
    is_causal: bool
    internal: bool


PUBLIC_CASES = (
    ("sample_group_mapping", 1, 4, 8, 4, 2, 470311, "group_markers", False),
    ("sample_causal", 2, 7, 16, 8, 2, 470311, "random", True),
    ("sample_mqa", 1, 9, 12, 6, 1, 470311, "random", True),
)
INTERNAL_CASES = (
    ("internal_scalar", 1, 1, 1, 1, 1, 141421, "constant", False),
    ("internal_degenerate_mha", 1, 5, 12, 3, 3, 141421, "random", False),
    ("internal_odd_group", 1, 11, 18, 6, 2, 173205, "group_markers", True),
    ("internal_causal_pair_full", 1, 13, 24, 6, 3, 173205, "random", False),
    ("internal_causal_pair_masked", 1, 13, 24, 6, 3, 173205, "random", True),
    ("internal_extreme_logits", 1, 17, 48, 12, 3, 223607, "extreme_logits", True),
    ("internal_larger_random", 2, 65, 64, 16, 4, 141421, "random", False),
)


def load_submission() -> ModuleType:
    source_path = Path(os.environ.get("MYLEETGPU_SOURCE_PATH", "/work/source.py"))
    policy_path = Path("/work/submission_policy.py")
    spec = importlib.util.spec_from_file_location("_myleetgpu_submission_policy", policy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the platform submission policy")
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    module = policy.load_submission(
        source_path,
        expected_class_name="GroupedQueryAttention",
        expected_init_parameters=(
            "numQueryHeads",
            "numKeyValueHeads",
            "qWeight",
            "kWeight",
            "vWeight",
            "outputWeight",
        ),
        expected_forward_parameters=("X", "isCasual"),
    )
    attention_type = module.__dict__.get("GroupedQueryAttention")
    if not isinstance(attention_type, type):
        raise SubmissionCompileError("GroupedQueryAttention class is required")
    return module


def reference_attention(
    X: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    output_weight: torch.Tensor,
    query_heads: int,
    key_value_heads: int,
    is_causal: bool,
) -> torch.Tensor:
    batch, sequence_length, embedding_dim = X.shape
    head_dim = embedding_dim // query_heads
    key_value_dim = key_value_heads * head_dim
    X64 = X.to(torch.float64)
    query = torch.matmul(X64, q_weight.to(torch.float64))
    key = torch.matmul(X64, k_weight.to(torch.float64))
    value = torch.matmul(X64, v_weight.to(torch.float64))
    query = query.reshape(batch, sequence_length, query_heads, head_dim).transpose(1, 2)
    key = key.reshape(batch, sequence_length, key_value_heads, head_dim).transpose(1, 2)
    value = value.reshape(batch, sequence_length, key_value_heads, head_dim).transpose(1, 2)
    repeats_per_group = query_heads // key_value_heads
    grouped_key = torch.repeat_interleave(key, repeats_per_group, dim=1)
    grouped_value = torch.repeat_interleave(value, repeats_per_group, dim=1)
    scores = torch.matmul(query, grouped_key.transpose(-2, -1)) * (head_dim**-0.5)
    if is_causal:
        positions = torch.arange(sequence_length)
        causal_mask = positions.unsqueeze(0) <= positions.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    context = torch.matmul(probabilities, grouped_value)
    merged = (
        context.transpose(1, 2)
        .contiguous()
        .reshape(batch, sequence_length, key_value_dim * repeats_per_group)
    )
    return torch.matmul(merged, output_weight.to(torch.float64)).to(torch.float32)


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
    embedding_dim = test.embedding_dim
    head_dim = embedding_dim // test.query_heads
    key_value_dim = test.key_value_heads * head_dim
    generator = torch.Generator(device="cpu").manual_seed(test.seed)
    input_shape = (test.batch, test.sequence_length, embedding_dim)
    weight_scale = embedding_dim**-0.5

    X = torch.randn(input_shape, generator=generator, dtype=torch.float32) * 0.5
    q_weight = (
        torch.randn((embedding_dim, embedding_dim), generator=generator, dtype=torch.float32)
        * weight_scale
    )
    k_weight = (
        torch.randn((embedding_dim, key_value_dim), generator=generator, dtype=torch.float32)
        * weight_scale
    )
    v_weight = (
        torch.randn((embedding_dim, key_value_dim), generator=generator, dtype=torch.float32)
        * weight_scale
    )
    output_weight = (
        torch.randn((embedding_dim, embedding_dim), generator=generator, dtype=torch.float32)
        * weight_scale
    )

    if test.pattern == "constant":
        X = torch.full(input_shape, 0.25, dtype=torch.float32)
        q_weight = torch.eye(embedding_dim, dtype=torch.float32) * 0.5
        k_weight = torch.full((embedding_dim, key_value_dim), -0.125, dtype=torch.float32)
        v_weight = torch.full((embedding_dim, key_value_dim), 0.375, dtype=torch.float32)
        output_weight = torch.eye(embedding_dim, dtype=torch.float32)
    elif test.pattern == "group_markers":
        values = torch.arange(
            test.batch * test.sequence_length * embedding_dim, dtype=torch.float32
        )
        X = values.reshape(input_shape) / max(1, embedding_dim * test.sequence_length)
        X = X - 0.5
        q_weight = torch.eye(embedding_dim, dtype=torch.float32)
        k_weight = torch.zeros((embedding_dim, key_value_dim), dtype=torch.float32)
        v_weight = torch.zeros((embedding_dim, key_value_dim), dtype=torch.float32)
        for kv_head in range(test.key_value_heads):
            for dimension in range(head_dim):
                output_index = kv_head * head_dim + dimension
                source_index = ((kv_head + 1) * head_dim + dimension) % embedding_dim
                k_weight[source_index, output_index] = 0.5 + kv_head * 0.375
                v_weight[source_index, output_index] = 1.0 + kv_head * 0.75
        output_weight = torch.eye(embedding_dim, dtype=torch.float32)
    elif test.pattern == "extreme_logits":
        X = X * 2.0
        q_weight = q_weight * 64.0
        k_weight = k_weight * 64.0
    elif test.pattern != "random":
        raise RuntimeError("unknown input pattern")

    expected = reference_attention(
        X,
        q_weight,
        k_weight,
        v_weight,
        output_weight,
        test.query_heads,
        test.key_value_heads,
        test.is_causal,
    )
    return X, q_weight, k_weight, v_weight, output_weight, expected


def validate_output_contract(
    output: object,
    inputs: tuple[torch.Tensor, ...],
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    X = inputs[0]
    if not isinstance(output, torch.Tensor):
        raise RuntimeError("forward must return a torch.Tensor")
    if output.device != X.device or output.device.type != "cuda" or output.device.index != 0:
        raise RuntimeError("forward must return a CUDA tensor on device 0")
    if output.dtype != torch.float32:
        raise RuntimeError("forward must return torch.float32")
    if tuple(output.shape) != expected_shape:
        raise RuntimeError("forward returned an incorrect shape")
    input_storage_pointers = {tensor.untyped_storage().data_ptr() for tensor in inputs}
    if output.untyped_storage().data_ptr() in input_storage_pointers:
        raise RuntimeError("forward output must not alias X or a constructor weight")
    return output


def run_case(module: ModuleType, test: TestCase, stream: torch.cuda.Stream) -> dict[str, Any]:
    X, q_weight, k_weight, v_weight, output_weight, expected = make_inputs(test)
    with torch.inference_mode(), torch.cuda.stream(stream):
        device_X = X.cuda(non_blocking=False)
        device_q_weight = q_weight.cuda(non_blocking=False)
        device_k_weight = k_weight.cuda(non_blocking=False)
        device_v_weight = v_weight.cuda(non_blocking=False)
        device_output_weight = output_weight.cuda(non_blocking=False)
        inputs = (
            device_X,
            device_q_weight,
            device_k_weight,
            device_v_weight,
            device_output_weight,
        )
        snapshots = tuple(tensor.clone() for tensor in inputs)
        attention_type = module.__dict__["GroupedQueryAttention"]
        attention = attention_type(
            test.query_heads,
            test.key_value_heads,
            device_q_weight,
            device_k_weight,
            device_v_weight,
            device_output_weight,
        )
        returned = attention.forward(device_X, test.is_causal)
        output = validate_output_contract(returned, inputs, tuple(expected.shape))
    stream.synchronize()

    for current, snapshot in zip(inputs, snapshots, strict=True):
        if not torch.equal(current, snapshot):
            raise RuntimeError("forward must not modify X or constructor weights")

    actual = output.detach().cpu()
    if not bool(torch.all(torch.isfinite(actual))):
        return {"name": test.name, "passed": False, "message": "output contains non-finite values"}
    close = torch.isclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False)
    if bool(torch.all(close)):
        return {"name": test.name, "passed": True}
    message = "output mismatch"
    if not test.internal:
        mismatch = torch.nonzero(~close, as_tuple=False)[0].tolist()
        max_error = float(torch.max(torch.abs(actual - expected)).item())
        message = f"output mismatch near index {mismatch}; max abs error {max_error:.6g}"
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
