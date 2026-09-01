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
PROTOCOL_VERSION = "2"
WARMUP = 5
ITERATIONS = 20
ATOL = 3.0e-4
RTOL = 3.0e-4
CASES = (
    ("B1-S128-E256-QH8-KVH2-C0", 1, 128, 256, 8, 2, False, 8),
    ("B2-S384-E512-QH16-KVH4-C1", 2, 384, 512, 16, 4, True, 1),
    ("B1-S1024-E512-QH32-KVH8-C1", 1, 1024, 512, 32, 8, True, 1),
)


class SubmissionCompileError(RuntimeError):
    pass


def is_compilation_error(error: BaseException) -> bool:
    return isinstance(error, SyntaxError | SubmissionCompileError) or type(error).__name__ in {
        "SubmissionPolicyError",
        "TorchSubmissionPolicyError",
    }


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
    query = torch.matmul(X, q_weight)
    key = torch.matmul(X, k_weight)
    value = torch.matmul(X, v_weight)
    query = query.reshape(batch, sequence_length, query_heads, head_dim).transpose(1, 2)
    key = key.reshape(batch, sequence_length, key_value_heads, head_dim).transpose(1, 2)
    value = value.reshape(batch, sequence_length, key_value_heads, head_dim).transpose(1, 2)
    repeats_per_group = query_heads // key_value_heads
    grouped_key = torch.repeat_interleave(key, repeats_per_group, dim=1)
    grouped_value = torch.repeat_interleave(value, repeats_per_group, dim=1)
    scores = torch.matmul(query, grouped_key.transpose(-2, -1)) * (head_dim**-0.5)
    if is_causal:
        positions = torch.arange(sequence_length, device=X.device)
        causal_mask = positions.unsqueeze(0) <= positions.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    context = torch.matmul(probabilities, grouped_value)
    merged = (
        context.transpose(1, 2)
        .contiguous()
        .reshape(batch, sequence_length, key_value_dim * repeats_per_group)
    )
    return torch.matmul(merged, output_weight)


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


def ensure_inputs_unchanged(
    tensors: tuple[torch.Tensor, ...], snapshots: tuple[torch.Tensor, ...]
) -> None:
    for current, snapshot in zip(tensors, snapshots, strict=True):
        if not torch.equal(current, snapshot):
            raise RuntimeError("forward must not modify X or constructor weights")


def ensure_correct(output: torch.Tensor, expected: torch.Tensor) -> None:
    actual_cpu = output.detach().cpu()
    expected_cpu = expected.detach().cpu()
    if not bool(torch.all(torch.isfinite(actual_cpu))):
        raise RuntimeError("correctness check failed before timing")
    if not bool(
        torch.all(torch.isclose(actual_cpu, expected_cpu, rtol=RTOL, atol=ATOL, equal_nan=False))
    ):
        raise RuntimeError("correctness check failed before timing")


def run_benchmark(
    module: ModuleType,
    label: str,
    batch: int,
    sequence_length: int,
    embedding_dim: int,
    query_heads: int,
    key_value_heads: int,
    is_causal: bool,
    inner_repetitions: int,
    generator: torch.Generator,
    stream: torch.cuda.Stream,
) -> dict[str, Any]:
    head_dim = embedding_dim // query_heads
    key_value_dim = key_value_heads * head_dim
    weight_scale = embedding_dim**-0.5
    X = (
        torch.randn(
            (batch, sequence_length, embedding_dim),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.5
    )
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
            query_heads,
            key_value_heads,
            device_q_weight,
            device_k_weight,
            device_v_weight,
            device_output_weight,
        )
        expected = reference_attention(
            device_X,
            device_q_weight,
            device_k_weight,
            device_v_weight,
            device_output_weight,
            query_heads,
            key_value_heads,
            is_causal,
        )
        before_warmup = validate_output_contract(
            attention.forward(device_X, is_causal),
            inputs,
            (batch, sequence_length, embedding_dim),
        )
    stream.synchronize()
    ensure_correct(before_warmup, expected)
    ensure_inputs_unchanged(inputs, snapshots)

    with torch.inference_mode(), torch.cuda.stream(stream):
        warmed = before_warmup
        for _ in range(WARMUP * inner_repetitions):
            warmed = validate_output_contract(
                attention.forward(device_X, is_causal),
                inputs,
                (batch, sequence_length, embedding_dim),
            )
    stream.synchronize()
    ensure_correct(warmed, expected)
    ensure_inputs_unchanged(inputs, snapshots)

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    last_output = warmed
    for _ in range(ITERATIONS):
        with torch.inference_mode(), torch.cuda.stream(stream):
            start.record(stream)
            for _ in range(inner_repetitions):
                last_output = validate_output_contract(
                    attention.forward(device_X, is_causal),
                    inputs,
                    (batch, sequence_length, embedding_dim),
                )
            stop.record(stream)
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)) / inner_repetitions)

    ensure_correct(last_output, expected)
    ensure_inputs_unchanged(inputs, snapshots)
    return {
        "size": batch * query_heads * sequence_length * sequence_length * head_dim,
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
        emit(
            {
                "status": "runtime_error",
                "measurements": [],
                "protocol_version": PROTOCOL_VERSION,
                "message": "invalid arguments",
            }
        )
        return 2
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
        generator = torch.Generator(device="cpu").manual_seed(20260902)
        measurements = [
            trusted_run_benchmark(
                module,
                label,
                batch,
                sequence_length,
                embedding_dim,
                query_heads,
                key_value_heads,
                is_causal,
                repetitions,
                generator,
                stream,
            )
            for (
                label,
                batch,
                sequence_length,
                embedding_dim,
                query_heads,
                key_value_heads,
                is_causal,
                repetitions,
            ) in CASES
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
            "message": (
                "PyTorch submission policy failed" if failed_to_compile else "benchmark failed"
            ),
        }
        returncode = 1
    emit(payload)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
