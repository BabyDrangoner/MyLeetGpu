from __future__ import annotations

from pathlib import Path

import pytest
from myleetgpu.config import Settings
from myleetgpu.infrastructure.database import (
    Base,
    build_engine,
    build_session_factory,
    session_scope,
)
from myleetgpu.infrastructure.models import (
    BenchmarkRunRecord,
    EnvironmentSnapshotRecord,
    JobRecord,
    VersionRecord,
)
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError


@pytest.fixture
def database(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "data",
        database_url_override=f"sqlite:///{(tmp_path / 'database.db').as_posix()}",
        _env_file=None,
    )
    engine = build_engine(settings)
    Base.metadata.create_all(engine)
    try:
        yield engine, build_session_factory(engine)
    finally:
        engine.dispose()


def test_sqlite_connections_enable_wal_foreign_keys_and_busy_timeout(database) -> None:
    engine, _ = database

    with engine.connect() as connection:
        journal_mode = connection.execute(text("PRAGMA journal_mode")).scalar_one()
        foreign_keys = connection.execute(text("PRAGMA foreign_keys")).scalar_one()
        busy_timeout = connection.execute(text("PRAGMA busy_timeout")).scalar_one()
        synchronous = connection.execute(text("PRAGMA synchronous")).scalar_one()

    assert journal_mode.lower() == "wal"
    assert foreign_keys == 1
    assert busy_timeout == 30_000
    assert synchronous == 1  # NORMAL


def test_job_schema_never_contains_a_source_code_column(database) -> None:
    engine, _ = database

    columns = {column["name"] for column in inspect(engine).get_columns("jobs")}

    assert "source_code" not in columns
    assert {"source_hash", "spool_path"} <= columns


def test_ordinary_job_persists_only_hash_and_temporary_spool_reference(database) -> None:
    engine, factory = database
    source = 'extern "C" __global__ void secret() {}'
    job = JobRecord(
        problem_id="vector-addition",
        problem_revision="1",
        action="validate",
        status="queued",
        phase="queued",
        progress=0.0,
        source_hash="a" * 64,
        spool_path="job-id/source.cu",
        payload_json={"mode": "full"},
    )

    with session_scope(factory) as session:
        session.add(job)

    database_bytes = Path(engine.url.database).read_bytes()
    assert source.encode() not in database_bytes
    with session_scope(factory) as session:
        stored = session.get(JobRecord, job.id)
        assert stored is not None
        assert stored.source_hash == "a" * 64
        assert stored.spool_path == "job-id/source.cu"
        assert "source" not in stored.payload_json


def test_foreign_keys_reject_orphan_benchmark_run(database) -> None:
    _, factory = database
    orphan = BenchmarkRunRecord(
        version_id="missing-version",
        environment_snapshot_id="missing-environment",
        suite_hash="s" * 64,
        protocol_version="1",
        compile_flags_json=["-O3"],
        input_sizes_json=["64K"],
        seed=1,
        warmup=1,
        iterations=2,
        measurements_json=[],
        raw_samples_json=[],
    )

    with pytest.raises(IntegrityError), session_scope(factory) as session:
        session.add(orphan)


def test_version_and_benchmark_round_trip_with_environment_snapshot(database) -> None:
    _, factory = database
    environment = EnvironmentSnapshotRecord(
        fingerprint="f" * 64,
        healthy=True,
        gpu_name="NVIDIA GeForce RTX 4060",
        compute_capability="8.9",
        driver_version="999.1",
        cuda_runtime_version="12.4",
        nvcc_version="12.4",
        cuda_image="nvidia/cuda:12.4.1-devel-ubuntu22.04",
        image_digest="sha256:" + "d" * 64,
        cuda_arch="89",
        telemetry_json={"temperature": "unavailable"},
    )
    version = VersionRecord(
        problem_id="vector-addition",
        problem_revision="1",
        name="coalesced",
        notes="manual snapshot",
        source_code="void solve() {}",
        source_hash="c" * 64,
        compile_flags_json=["-O3", "-arch=sm_89"],
        correctness_status="passed",
        environment=environment,
        suite_hash="s" * 64,
        protocol_version="1",
    )
    run = BenchmarkRunRecord(
        version=version,
        environment=environment,
        suite_hash="s" * 64,
        protocol_version="1",
        compile_flags_json=["-O3", "-arch=sm_89"],
        input_sizes_json=["64K"],
        seed=42,
        warmup=8,
        iterations=20,
        measurements_json=[{"size": "64K", "median_ms": 0.02, "p95_ms": 0.03}],
        raw_samples_json=[{"size": "64K", "samples_ms": [0.02, 0.03]}],
    )

    with session_scope(factory) as session:
        session.add(version)
        session.add(run)

    with session_scope(factory) as session:
        stored = session.scalar(select(VersionRecord).where(VersionRecord.id == version.id))
        assert stored is not None
        assert stored.environment.fingerprint == "f" * 64
        assert len(stored.benchmark_runs) == 1
        assert stored.benchmark_runs[0].iterations == 20


def test_session_scope_rolls_back_failed_unit_of_work(database) -> None:
    _, factory = database
    job = JobRecord(
        problem_id="reduction",
        problem_revision="1",
        action="compile",
        status="queued",
        phase="queued",
        progress=0.0,
        payload_json={},
    )

    with pytest.raises(RuntimeError, match="abort"), session_scope(factory) as session:
        session.add(job)
        session.flush()
        raise RuntimeError("abort")

    with session_scope(factory) as session:
        assert session.get(JobRecord, job.id) is None
