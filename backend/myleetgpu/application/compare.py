from __future__ import annotations

from typing import Any

from myleetgpu.domain.benchmark import ComparabilityKey
from myleetgpu.infrastructure.models import BenchmarkRunRecord, VersionRecord
from myleetgpu.infrastructure.repository import Repository


class ComparisonError(ValueError):
    pass


def _latest_run(version: VersionRecord) -> BenchmarkRunRecord:
    if not version.benchmark_runs:
        raise ComparisonError(f"version {version.id} has no benchmark run")
    return max(version.benchmark_runs, key=lambda item: (item.created_at, item.id))


def _key(version: VersionRecord, run: BenchmarkRunRecord) -> ComparabilityKey:
    return ComparabilityKey(
        language=version.language,
        problem_revision=version.problem_revision,
        suite_hash=run.suite_hash,
        input_sizes=run.input_sizes_json,
        compile_flags=run.compile_flags_json,
        environment_fingerprint=run.environment.fingerprint,
    )


def compare_versions(
    repository: Repository,
    *,
    problem_id: str,
    version_ids: list[str],
    baseline_id: str,
    language: str | None = None,
) -> dict[str, Any]:
    if len(version_ids) < 2 or len(version_ids) > 8 or len(set(version_ids)) != len(version_ids):
        raise ComparisonError("select 2 to 8 unique versions")
    if baseline_id not in version_ids:
        raise ComparisonError("baseline_id must be one of version_ids")
    versions = repository.get_versions(version_ids)
    if len(versions) != len(version_ids):
        raise ComparisonError("one or more versions do not exist")
    if any(version.problem_id != problem_id for version in versions):
        raise ComparisonError("all versions must belong to the same problem")
    if language is not None and any(version.language != language for version in versions):
        raise ComparisonError("all versions must use the requested implementation language")

    by_id = {version.id: version for version in versions}
    baseline = by_id[baseline_id]
    runs = {version.id: _latest_run(version) for version in versions}
    baseline_key = _key(baseline, runs[baseline_id])
    differences = {
        version.id: baseline_key.differences(_key(version, runs[version.id]))
        for version in versions
    }
    comparable = all(not reasons for reasons in differences.values())
    environment_consistent = all(
        runs[version.id].environment.fingerprint == runs[baseline_id].environment.fingerprint
        for version in versions
    )
    all_reasons = sorted({reason for reasons in differences.values() for reason in reasons})

    measurement_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for version in versions:
        measurement_maps[version.id] = {
            str(item["size"]): item for item in runs[version.id].measurements_json
        }
    rows: list[dict[str, Any]] = []
    sizes = runs[baseline_id].input_sizes_json
    for size in sizes:
        metrics: dict[str, Any] = {}
        baseline_measurement = measurement_maps[baseline_id].get(size)
        baseline_median = float(baseline_measurement["median_ms"]) if baseline_measurement else None
        for version in versions:
            measurement = measurement_maps[version.id].get(size)
            if measurement is None:
                metrics[version.id] = None
                continue
            median = float(measurement["median_ms"])
            metrics[version.id] = {
                **measurement,
                "sample_count": len(measurement.get("samples_ms", [])),
                "speedup": (
                    baseline_median / median
                    if comparable and baseline_median is not None and median > 0
                    else None
                ),
            }
        rows.append({"size": size, "metrics": metrics})

    return {
        "problem_id": problem_id,
        "baseline_id": baseline_id,
        "comparable": comparable,
        "environment_consistent": environment_consistent,
        "reasons": all_reasons,
        "version_compatibility": differences,
        "versions": [
            {
                "id": version.id,
                "name": version.name,
                "problem_revision": version.problem_revision,
                "language": version.language,
                "source_hash": version.source_hash,
                "environment": _environment_dict(runs[version.id]),
                "suite_hash": runs[version.id].suite_hash,
                "protocol_version": runs[version.id].protocol_version,
                "compile_flags": runs[version.id].compile_flags_json,
            }
            for version in versions
        ],
        "rows": rows,
    }


def _environment_dict(run: BenchmarkRunRecord) -> dict[str, Any]:
    environment = run.environment
    return {
        "fingerprint": environment.fingerprint,
        "backend": environment.backend,
        "gpu_name": environment.gpu_name,
        "compute_capability": environment.compute_capability,
        "driver_version": environment.driver_version,
        "cuda_runtime_version": environment.cuda_runtime_version,
        "nvcc_version": environment.nvcc_version,
        "cuda_image": environment.cuda_image,
        "image_digest": environment.image_digest,
        "toolchain": environment.toolchain_json,
    }
