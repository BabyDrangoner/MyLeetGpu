from __future__ import annotations

import importlib.util
import json
import math
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
ATOL = 4.0e-4
RTOL = 4.0e-4
CLASS_NAME = "MultiHeadAttention"
INIT_PARAMETERS = ("numHeads", "qWeight", "kWeight", "vWeight", "outputWeight")
FORWARD_PARAMETERS = ("X", "isCasual")
CASES = (
    ("B1-H8-S128-E512", 1, 128, 512, 8, False, 4),
    ("B2-H12-S384-E768", 2, 384, 768, 12, True, 1),
    ("B1-H16-S1024-E1024", 1, 1024, 1024, 16, True, 1),
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
    query = torch.matmul(X, q_weight)
    key = torch.matmul(X, k_weight)
    value = torch.matmul(X, v_weight)
    query = query.reshape(batch, sequence_length, num_heads, head_dim).transpose(1, 2)
    key = key.reshape(batch, sequence_length, num_heads, head_dim).transpose(1, 2)
    value = value.reshape(batch, sequence_length, num_heads, head_dim).transpose(1, 2)
    scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim**-0.5)
    if is_casual:
        positions = torch.arange(sequence_length, device=X.device)
        causal_mask = positions.reshape(sequence_length, 1) >= positions.reshape(1, sequence_length)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    context = torch.matmul(probabilities, value)
    concatenated = context.transpose(1, 2).contiguous().reshape(batch, sequence_length, embed_dim)
    return torch.matmul(concatenated, output_weight)


def validate_output_contract(
    output: object,
    tensors: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    X = tensors[0]
    if not isinstance(output, torch.Tensor):
        raise RuntimeError("forward must return a torch.Tensor")
    if output.device != X.device or output.device.type != "cuda" or output.device.index != 0:
        raise RuntimeError("forward must return a CUDA tensor on device 0")
    if output.dtype != torch.float32:
        raise RuntimeError("forward must return torch.float32")
    if tuple(output.shape) != tuple(X.shape):
        raise RuntimeError("forward returned an incorrect shape")
    input_storage_pointers = {tensor.untyped_storage().data_ptr() for tensor in tensors}
    if output.untyped_storage().data_ptr() in input_storage_pointers:
        raise RuntimeError("forward output must not alias X or a weight tensor")
    return output


def ensure_inputs_unchanged(
    tensors: tuple[torch.Tensor, ...], snapshots: tuple[torch.Tensor, ...]
) -> None:
    for current, snapshot in zip(tensors, snapshots, strict=True):
        if not torch.equal(current, snapshot):
            raise RuntimeError("forward and constructor must not modify X or weight tensors")


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
    attention_type: type,
    label: str,
    batch: int,
    sequence_length: int,
    embed_dim: int,
    num_heads: int,
    is_casual: bool,
    inner_repetitions: int,
    generator: torch.Generator,
    stream: torch.cuda.Stream,
) -> dict[str, Any]:
    X = (
        torch.randn((batch, sequence_length, embed_dim), generator=generator, dtype=torch.float32)
        * 0.5
    )
    weight_scale = 0.8 / math.sqrt(embed_dim)
    weights = tuple(
        torch.randn((embed_dim, embed_dim), generator=generator, dtype=torch.float32) * weight_scale
        for _ in range(4)
    )

    with torch.inference_mode(), torch.cuda.stream(stream):
        device_tensors = tuple(tensor.cuda(non_blocking=False) for tensor in (X, *weights))
        snapshots = tuple(tensor.clone() for tensor in device_tensors)
        device_X, device_q, device_k, device_v, device_output = device_tensors
        instance = attention_type(
            num_heads,
            device_q,
            device_k,
            device_v,
            device_output,
        )
        expected = reference_attention(
            device_X,
            is_casual,
            num_heads,
            device_q,
            device_k,
            device_v,
            device_output,
        )
        before_warmup = validate_output_contract(
            instance.forward(device_X, is_casual), device_tensors
        )
    stream.synchronize()
    ensure_correct(before_warmup, expected)
    ensure_inputs_unchanged(device_tensors, snapshots)

    with torch.inference_mode(), torch.cuda.stream(stream):
        warmed = before_warmup
        for _ in range(WARMUP * inner_repetitions):
            warmed = validate_output_contract(instance.forward(device_X, is_casual), device_tensors)
    stream.synchronize()
    ensure_correct(warmed, expected)
    ensure_inputs_unchanged(device_tensors, snapshots)

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    last_output = warmed
    for _ in range(ITERATIONS):
        with torch.inference_mode(), torch.cuda.stream(stream):
            start.record(stream)
            for _ in range(inner_repetitions):
                last_output = validate_output_contract(
                    instance.forward(device_X, is_casual), device_tensors
                )
            stop.record(stream)
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)) / inner_repetitions)

    ensure_correct(last_output, expected)
    ensure_inputs_unchanged(device_tensors, snapshots)
    return {
        "size": batch * sequence_length * embed_dim,
        "label": label,
        "samples_ms": samples,
        "inner_repetitions": inner_repetitions,
    }


def main(argv: list[str]) -> int:
    trusted_load_submission = load_submission
    trusted_submission_class = submission_class
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
        attention_type = trusted_submission_class(module)
        stream = torch.cuda.Stream(device=0)
        generator = torch.Generator(device="cpu").manual_seed(20260902)
        measurements = [
            trusted_run_benchmark(
                attention_type,
                label,
                batch,
                sequence_length,
                embed_dim,
                num_heads,
                is_casual,
                inner_repetitions,
                generator,
                stream,
            )
            for (
                label,
                batch,
                sequence_length,
                embed_dim,
                num_heads,
                is_casual,
                inner_repetitions,
            ) in CASES
        ]
    except Exception as error:
        failed_to_compile = trusted_is_compilation_error(error)
        message = "PyTorch submission policy failed" if failed_to_compile else "benchmark failed"
        emit(
            {
                "status": "compile_error" if failed_to_compile else "runtime_error",
                "measurements": [],
                "protocol_version": PROTOCOL_VERSION,
                "message": message,
            }
        )
        return 1

    emit(
        {
            "status": "passed",
            "measurements": measurements,
            "protocol_version": PROTOCOL_VERSION,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
