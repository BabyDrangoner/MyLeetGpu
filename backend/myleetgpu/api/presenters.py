"""Pure response mappings; infrastructure-only fields never leave this adapter."""

from __future__ import annotations

from typing import Any

from myleetgpu.api.schemas import JobResponse
from myleetgpu.infrastructure.models import (
    BenchmarkRunRecord,
    DraftRecord,
    EnvironmentSnapshotRecord,
    JobRecord,
    VersionRecord,
)


def draft_response(record: DraftRecord) -> dict[str, Any]:
    return {
        "problem_id": record.problem_id,
        "language": record.language,
        "source": record.source_code,
        "updated_at": record.updated_at,
    }


def job_response(record: JobRecord) -> JobResponse:
    return JobResponse(
        id=record.id,
        problem_id=record.problem_id,
        problem_revision=record.problem_revision,
        language=record.language,
        execution_target=record.payload_json.get("execution_target", "local"),
        action=record.action,
        status=record.status,
        phase=record.phase,
        progress=record.progress,
        source_hash=record.source_hash,
        result=record.result_json,
        error=record.error_json,
        diagnostics=record.diagnostics,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
    )


def version_response(record: VersionRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "problem_id": record.problem_id,
        "problem_revision": record.problem_revision,
        "language": record.language,
        "name": record.name,
        "notes": record.notes,
        "source_code": record.source_code,
        "source_hash": record.source_hash,
        "compile_flags": record.compile_flags_json,
        "correctness_status": record.correctness_status,
        "suite_hash": record.suite_hash,
        "protocol_version": record.protocol_version,
        "created_at": record.created_at,
        "environment": environment_response(record.environment),
        "benchmark_runs": [benchmark_response(item) for item in record.benchmark_runs],
    }


def benchmark_response(record: BenchmarkRunRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "suite_hash": record.suite_hash,
        "protocol_version": record.protocol_version,
        "compile_flags": record.compile_flags_json,
        "input_sizes": record.input_sizes_json,
        "seed": record.seed,
        "warmup": record.warmup,
        "iterations": record.iterations,
        "measurements": record.measurements_json,
        "raw_samples": record.raw_samples_json,
        "environment": environment_response(record.environment),
        "created_at": record.created_at,
    }


def environment_response(record: EnvironmentSnapshotRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "status": "healthy" if record.healthy else "unavailable",
        "healthy": record.healthy,
        "fingerprint": record.fingerprint,
        "backend": record.backend,
        "execution_target": record.toolchain_json.get("execution_target", "local"),
        "gpu_name": record.gpu_name,
        "compute_capability": record.compute_capability,
        "driver_version": record.driver_version,
        "cuda_runtime_version": record.cuda_runtime_version,
        "nvcc_version": record.nvcc_version,
        "cuda_image": record.cuda_image,
        "image_digest": record.image_digest,
        "cuda_arch": record.cuda_arch,
        "toolchain": record.toolchain_json,
        "telemetry": {**unavailable_telemetry(), **record.telemetry_json},
        "error": record.error,
        "observed_at": record.observed_at,
    }


def unavailable_telemetry() -> dict[str, None]:
    return {
        "temperature_c": None,
        "power_w": None,
        "sm_clock_mhz": None,
        "gpu_busy_percent": None,
    }
