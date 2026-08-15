from __future__ import annotations

from collections.abc import Sequence

from myleetgpu.infrastructure.models import VersionRecord
from myleetgpu.infrastructure.repository import Repository
from myleetgpu.runner.models import EnvironmentProbe


def make_probe(
    fingerprint: str = "env-a",
    *,
    healthy: bool = True,
) -> EnvironmentProbe:
    return EnvironmentProbe(
        healthy=healthy,
        gpu_name="NVIDIA GeForce RTX 4060" if healthy else None,
        compute_capability="8.9" if healthy else None,
        driver_version="999.1" if healthy else None,
        cuda_runtime_version="12.4" if healthy else None,
        nvcc_version="Cuda compilation tools, release 12.4" if healthy else None,
        cuda_image="nvidia/cuda:12.4.1-devel-ubuntu22.04",
        image_digest="nvidia/cuda@sha256:" + "d" * 64 if healthy else None,
        cuda_arch="89" if healthy else None,
        telemetry={
            "temperature_c": None,
            "power_w": None,
            "sm_clock_mhz": None,
            "gpu_busy_percent": None,
        },
        error=None if healthy else "probe failed",
        fingerprint=fingerprint,
    )


def create_saved_version(
    repository: Repository,
    *,
    problem_id: str = "vector-addition",
    revision: str = "1",
    name: str = "baseline",
    source: str = "void solve() {}",
    source_digest: str = "a" * 64,
    environment_fingerprint: str = "env-a",
    suite_digest: str = "s" * 64,
    protocol_version: str = "1",
    compile_flags: Sequence[str] = ("--std=c++17", "-O3"),
    sizes: Sequence[str] = ("64K", "1M"),
    medians: Sequence[float] = (4.0, 8.0),
    iterations: int = 3,
) -> VersionRecord:
    environment = repository.save_environment(make_probe(environment_fingerprint))
    measurements = [
        {
            "size": size,
            "samples_ms": [median] * iterations,
            "median_ms": median,
            "p95_ms": median,
            "min_ms": median,
            "cv": 0.0,
            "mad_ms": 0.0,
            "inner_repetitions": 1,
        }
        for size, median in zip(sizes, medians, strict=True)
    ]
    raw_samples = [
        {"size": size, "samples_ms": [median] * iterations}
        for size, median in zip(sizes, medians, strict=True)
    ]
    created = repository.create_version_with_benchmark(
        problem_id=problem_id,
        problem_revision=revision,
        name=name,
        notes=None,
        source_code=source,
        source_hash=source_digest,
        compile_flags=list(compile_flags),
        environment_id=environment.id,
        suite_hash=suite_digest,
        protocol_version=protocol_version,
        input_sizes=list(sizes),
        seed=42,
        warmup=2,
        iterations=iterations,
        measurements=measurements,
        raw_samples=raw_samples,
    )
    loaded = repository.get_version(created.id)
    assert loaded is not None
    return loaded
