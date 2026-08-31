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
PROTOCOL_VERSION = "1"
WARMUP = 5
ITERATIONS = 20
ATOL = 2.0e-4
RTOL = 2.0e-4
CASES = (
    ("B1-H8-Q128-K128-D64", 1, 8, 128, 128, 64, "all", 8),
    ("B2-H12-Q384-K512-D64", 2, 12, 384, 512, 64, "prefix", 1),
    ("B1-H16-Q1024-K1024-D64", 1, 16, 1024, 1024, 64, "causal", 1),
)


class SubmissionCompileError(RuntimeError):
    pass


def is_compilation_error(error: BaseException) -> bool:
    return isinstance(error, SyntaxError | SubmissionCompileError) or type(error).__name__ in {
        "SubmissionPolicyError",
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
        expected_parameters=("query", "key", "value", "attention_mask"),
    )


def make_mask(
    batch: int,
    query_length: int,
    key_length: int,
    mask_pattern: str,
) -> torch.Tensor:
    shape = (batch, 1, query_length, key_length)
    if mask_pattern == "all":
        return torch.ones(shape, dtype=torch.bool)
    if mask_pattern == "causal":
        if query_length != key_length:
            raise RuntimeError("causal benchmark configuration must be square")
        causal = torch.ones((query_length, key_length), dtype=torch.bool).tril()
        return (
            causal.reshape(1, 1, query_length, key_length)
            .expand(batch, 1, query_length, key_length)
            .contiguous()
        )
    if mask_pattern == "prefix":
        mask = torch.zeros(shape, dtype=torch.bool)
        for batch_index in range(batch):
            valid = max(1, key_length - ((batch_index * 37 + key_length // 8) % key_length))
            mask[batch_index, 0, :, :valid] = True
        return mask
    raise RuntimeError("unknown benchmark mask pattern")


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~attention_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value)


def validate_output_contract(
    output: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise RuntimeError("solve must return a torch.Tensor")
    if output.device != query.device or output.device.type != "cuda" or output.device.index != 0:
        raise RuntimeError("solve must return a CUDA tensor on device 0")
    if output.dtype != torch.float32:
        raise RuntimeError("solve must return torch.float32")
    expected_shape = (query.shape[0], query.shape[1], query.shape[2], query.shape[3])
    if tuple(output.shape) != expected_shape:
        raise RuntimeError("solve returned an incorrect shape")
    input_storage_pointers = {
        tensor.untyped_storage().data_ptr() for tensor in (query, key, value, attention_mask)
    }
    if output.untyped_storage().data_ptr() in input_storage_pointers:
        raise RuntimeError("solve output must not alias an input")
    return output


def ensure_inputs_unchanged(
    tensors: tuple[torch.Tensor, ...], snapshots: tuple[torch.Tensor, ...]
) -> None:
    for current, snapshot in zip(tensors, snapshots, strict=True):
        if not torch.equal(current, snapshot):
            raise RuntimeError("solve must not modify its inputs")


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
    heads: int,
    query_length: int,
    key_length: int,
    head_dim: int,
    mask_pattern: str,
    inner_repetitions: int,
    generator: torch.Generator,
    stream: torch.cuda.Stream,
) -> dict[str, Any]:
    query = (
        torch.randn(
            (batch, heads, query_length, head_dim), generator=generator, dtype=torch.float32
        )
        * 0.5
    )
    key = (
        torch.randn((batch, heads, key_length, head_dim), generator=generator, dtype=torch.float32)
        * 0.5
    )
    value = (
        torch.randn((batch, heads, key_length, head_dim), generator=generator, dtype=torch.float32)
        * 0.5
    )
    attention_mask = make_mask(batch, query_length, key_length, mask_pattern)

    with torch.inference_mode(), torch.cuda.stream(stream):
        device_query = query.cuda(non_blocking=False)
        device_key = key.cuda(non_blocking=False)
        device_value = value.cuda(non_blocking=False)
        device_mask = attention_mask.cuda(non_blocking=False)
        inputs = (device_query, device_key, device_value, device_mask)
        snapshots = tuple(tensor.clone() for tensor in inputs)
        expected = reference_attention(device_query, device_key, device_value, device_mask)

        before_warmup = validate_output_contract(
            module.solve(device_query, device_key, device_value, device_mask),
            device_query,
            device_key,
            device_value,
            device_mask,
        )
    stream.synchronize()
    ensure_correct(before_warmup, expected)
    ensure_inputs_unchanged(inputs, snapshots)

    with torch.inference_mode(), torch.cuda.stream(stream):
        warmed = before_warmup
        for _ in range(WARMUP * inner_repetitions):
            warmed = validate_output_contract(
                module.solve(device_query, device_key, device_value, device_mask),
                device_query,
                device_key,
                device_value,
                device_mask,
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
                    module.solve(device_query, device_key, device_value, device_mask),
                    device_query,
                    device_key,
                    device_value,
                    device_mask,
                )
            stop.record(stream)
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)) / inner_repetitions)

    ensure_correct(last_output, expected)
    ensure_inputs_unchanged(inputs, snapshots)
    return {
        "size": batch * heads * query_length * key_length * head_dim,
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
        generator = torch.Generator(device="cpu").manual_seed(20260831)
        measurements = [
            trusted_run_benchmark(
                module,
                label,
                batch,
                heads,
                query_length,
                key_length,
                head_dim,
                mask_pattern,
                repetitions,
                generator,
                stream,
            )
            for (
                label,
                batch,
                heads,
                query_length,
                key_length,
                head_dim,
                mask_pattern,
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
