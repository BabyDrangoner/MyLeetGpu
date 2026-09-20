from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from myleetgpu.domain.benchmark import stable_hash
from myleetgpu.infrastructure.repository import Repository
from myleetgpu.runner.models import EnvironmentProbe


def probe_response(probe: EnvironmentProbe) -> dict[str, Any]:
    payload = asdict(probe)
    payload["status"] = "healthy" if probe.healthy else "unavailable"
    payload["execution_target"] = probe.toolchain.get("execution_target", "local")
    payload["observed_at"] = datetime.now(UTC).isoformat()
    return payload


def failed_probe(target: str, language: str, message: str, local_image: str) -> EnvironmentProbe:
    return EnvironmentProbe(
        healthy=False,
        gpu_name=None,
        compute_capability=None,
        driver_version=None,
        cuda_runtime_version=None,
        nvcc_version=None,
        cuda_image={"colab": "colab-native", "cpu": "cpu-native"}.get(target, local_image),
        image_digest=None,
        cuda_arch=None,
        backend=language,
        error=message,
        toolchain={"execution_target": target},
        fingerprint=stable_hash({"target": target, "language": language, "error": message}),
    )


async def wait_for_probe(
    repository: Repository, target: str, language: str, timeout: float
) -> dict[str, Any]:
    """Only the worker executes probes, serialized with compilation and benchmarks."""
    probe_id = repository.enqueue_execution_probe(target, language, timeout)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            record = repository.execution_probe(probe_id)
            if record and record.status == "completed" and record.result_json is not None:
                return record.result_json
            await asyncio.sleep(0.15)
        raise TimeoutError("等待运行环境探测超时；Worker 可能正在执行任务，请稍后重试")
    finally:
        repository.cancel_execution_probe(probe_id)
