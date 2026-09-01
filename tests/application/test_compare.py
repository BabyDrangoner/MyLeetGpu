from __future__ import annotations

from pathlib import Path

import pytest
from myleetgpu.application.compare import ComparisonError, compare_versions
from myleetgpu.config import Settings
from myleetgpu.infrastructure.database import Base, build_engine, build_session_factory
from myleetgpu.infrastructure.repository import Repository

from tests.factories import create_saved_version, make_probe


@pytest.fixture
def repository(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "data",
        database_url_override=f"sqlite:///{(tmp_path / 'compare.db').as_posix()}",
        _env_file=None,
    )
    engine = build_engine(settings)
    Base.metadata.create_all(engine)
    value = Repository(build_session_factory(engine))
    try:
        yield value
    finally:
        engine.dispose()


def test_comparable_versions_report_per_size_speedup_against_baseline(
    repository: Repository,
) -> None:
    baseline = create_saved_version(
        repository,
        name="baseline",
        source_digest="1" * 64,
        medians=(4.0, 8.0),
    )
    candidate = create_saved_version(
        repository,
        name="candidate",
        source_digest="2" * 64,
        medians=(2.0, 4.0),
    )

    result = compare_versions(
        repository,
        problem_id="vector-addition",
        version_ids=[candidate.id, baseline.id],
        baseline_id=baseline.id,
    )

    assert result["comparable"] is True
    assert result["environment_consistent"] is True
    assert result["reasons"] == []
    assert result["version_compatibility"] == {candidate.id: [], baseline.id: []}
    assert [item["id"] for item in result["versions"]] == [candidate.id, baseline.id]
    assert [row["size"] for row in result["rows"]] == ["64K", "1M"]
    for row in result["rows"]:
        assert row["metrics"][baseline.id]["speedup"] == 1.0
        assert row["metrics"][candidate.id]["speedup"] == 2.0
        assert row["metrics"][candidate.id]["sample_count"] == 3


def test_torch_versions_compare_with_runtime_profile_as_execution_identity(
    repository: Repository,
) -> None:
    runtime_profile = (
        "backend=torch_python",
        "policy=restricted_torch_v2",
        "python=3.11.10",
        "torch=2.5.1",
        "torch_cuda=12.4",
        "arch=sm_89",
    )
    baseline = create_saved_version(
        repository,
        problem_id="multi-head-attention",
        language="torch_python",
        compile_flags=runtime_profile,
        source_digest="3" * 64,
        medians=(2.0, 4.0),
    )
    candidate = create_saved_version(
        repository,
        problem_id="multi-head-attention",
        language="torch_python",
        compile_flags=runtime_profile,
        source_digest="4" * 64,
        medians=(1.0, 2.0),
    )

    result = compare_versions(
        repository,
        problem_id="multi-head-attention",
        version_ids=[baseline.id, candidate.id],
        baseline_id=baseline.id,
        language="torch_python",
    )

    assert result["comparable"] is True
    assert result["versions"][0]["language"] == "torch_python"
    assert result["versions"][0]["compile_flags"] == list(runtime_profile)
    assert all(row["metrics"][candidate.id]["speedup"] == 2.0 for row in result["rows"])


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"language": "triton_python"}, "实现语言不同"),
        ({"revision": "2"}, "题目版本不同"),
        ({"suite_digest": "t" * 64}, "benchmark suite 不同"),
        ({"sizes": ("16M",), "medians": (1.0,)}, "输入规模不同"),
        ({"compile_flags": ("--std=c++17", "-O2")}, "编译配置不同"),
        ({"environment_fingerprint": "env-b"}, "GPU/CUDA 环境指纹不同"),
    ],
)
def test_any_comparability_key_mismatch_suppresses_speedups_and_ranking(
    repository: Repository,
    overrides: dict[str, object],
    expected_reason: str,
) -> None:
    baseline = create_saved_version(repository, source_digest="1" * 64)
    candidate = create_saved_version(
        repository,
        name="candidate",
        source_digest="2" * 64,
        **overrides,
    )

    result = compare_versions(
        repository,
        problem_id="vector-addition",
        version_ids=[baseline.id, candidate.id],
        baseline_id=baseline.id,
    )

    assert result["comparable"] is False
    assert expected_reason in result["reasons"]
    assert expected_reason in result["version_compatibility"][candidate.id]
    for row in result["rows"]:
        for metrics in row["metrics"].values():
            if metrics is not None:
                assert metrics["speedup"] is None


def test_environment_consistency_is_reported_separately(repository: Repository) -> None:
    baseline = create_saved_version(repository, source_digest="1" * 64)
    candidate = create_saved_version(
        repository,
        source_digest="2" * 64,
        environment_fingerprint="different-environment",
    )

    result = compare_versions(
        repository,
        problem_id="vector-addition",
        version_ids=[baseline.id, candidate.id],
        baseline_id=baseline.id,
    )

    assert result["environment_consistent"] is False
    candidate_details = next(item for item in result["versions"] if item["id"] == candidate.id)
    assert candidate_details["environment"]["fingerprint"] == "different-environment"
    assert candidate_details["environment"]["gpu_name"] == "NVIDIA GeForce RTX 4060"


def test_explicit_compare_language_rejects_mismatched_versions(repository: Repository) -> None:
    cuda = create_saved_version(repository, source_digest="1" * 64)
    triton = create_saved_version(
        repository,
        source_digest="2" * 64,
        language="triton_python",
        compile_flags=("backend=triton_python",),
    )

    with pytest.raises(ComparisonError, match="requested implementation language"):
        compare_versions(
            repository,
            problem_id="vector-addition",
            version_ids=[cuda.id, triton.id],
            baseline_id=cuda.id,
            language="cuda_cpp",
        )


@pytest.mark.parametrize(
    ("version_ids", "baseline", "message"),
    [
        ([], "none", "2 to 8"),
        (["same", "same"], "same", "unique"),
        (["one", "two"], "not-selected", "baseline_id"),
    ],
)
def test_compare_validates_selection_before_database_access(
    repository: Repository,
    version_ids: list[str],
    baseline: str,
    message: str,
) -> None:
    with pytest.raises(ComparisonError, match=message):
        compare_versions(
            repository,
            problem_id="vector-addition",
            version_ids=version_ids,
            baseline_id=baseline,
        )


def test_compare_rejects_missing_or_cross_problem_versions(repository: Repository) -> None:
    vector = create_saved_version(repository, problem_id="vector-addition")
    reduction = create_saved_version(
        repository,
        problem_id="reduction",
        source_digest="2" * 64,
    )

    with pytest.raises(ComparisonError, match="do not exist"):
        compare_versions(
            repository,
            problem_id="vector-addition",
            version_ids=[vector.id, "missing"],
            baseline_id=vector.id,
        )

    with pytest.raises(ComparisonError, match="same problem"):
        compare_versions(
            repository,
            problem_id="vector-addition",
            version_ids=[vector.id, reduction.id],
            baseline_id=vector.id,
        )


def test_compare_uses_latest_benchmark_run_without_rewriting_history(
    repository: Repository,
) -> None:
    baseline = create_saved_version(repository, source_digest="1" * 64, medians=(4.0, 8.0))
    candidate = create_saved_version(repository, source_digest="2" * 64, medians=(2.0, 4.0))
    environment = repository.save_environment(make_probe("env-a"))
    repository.add_benchmark_runs(
        [
            {
                "version_id": candidate.id,
                "environment_snapshot_id": environment.id,
                "suite_hash": "s" * 64,
                "protocol_version": "1",
                "compile_flags_json": ["--std=c++17", "-O3"],
                "input_sizes_json": ["64K", "1M"],
                "seed": 42,
                "warmup": 2,
                "iterations": 3,
                "measurements_json": [
                    {
                        "size": "64K",
                        "samples_ms": [1.0, 1.0, 1.0],
                        "median_ms": 1.0,
                        "p95_ms": 1.0,
                        "min_ms": 1.0,
                        "cv": 0.0,
                        "mad_ms": 0.0,
                    },
                    {
                        "size": "1M",
                        "samples_ms": [2.0, 2.0, 2.0],
                        "median_ms": 2.0,
                        "p95_ms": 2.0,
                        "min_ms": 2.0,
                        "cv": 0.0,
                        "mad_ms": 0.0,
                    },
                ],
                "raw_samples_json": [],
            }
        ]
    )

    result = compare_versions(
        repository,
        problem_id="vector-addition",
        version_ids=[baseline.id, candidate.id],
        baseline_id=baseline.id,
    )

    assert result["rows"][0]["metrics"][candidate.id]["speedup"] == 4.0
    assert len(repository.get_version(candidate.id).benchmark_runs) == 2  # type: ignore[union-attr]


def test_zero_median_never_produces_infinite_or_misleading_speedup(
    repository: Repository,
) -> None:
    baseline = create_saved_version(repository, source_digest="1" * 64)
    candidate = create_saved_version(
        repository,
        source_digest="2" * 64,
        medians=(0.0, 0.0),
    )

    result = compare_versions(
        repository,
        problem_id="vector-addition",
        version_ids=[baseline.id, candidate.id],
        baseline_id=baseline.id,
    )

    assert all(row["metrics"][candidate.id]["speedup"] is None for row in result["rows"])
