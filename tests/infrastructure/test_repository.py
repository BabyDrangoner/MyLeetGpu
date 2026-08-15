from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Barrier

import pytest
from myleetgpu.config import Settings
from myleetgpu.domain.jobs import JobStatus
from myleetgpu.infrastructure.database import Base, build_engine, build_session_factory
from myleetgpu.infrastructure.models import JobRecord, utc_now
from myleetgpu.infrastructure.repository import Repository
from sqlalchemy.exc import IntegrityError

from tests.factories import create_saved_version, make_probe


@pytest.fixture
def repository(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "data",
        database_url_override=f"sqlite:///{(tmp_path / 'repository.db').as_posix()}",
        _env_file=None,
    )
    engine = build_engine(settings)
    Base.metadata.create_all(engine)
    value = Repository(build_session_factory(engine))
    try:
        yield value
    finally:
        engine.dispose()


def make_job(identifier: str, *, created_offset: int = 0) -> JobRecord:
    return JobRecord(
        id=identifier,
        problem_id="vector-addition",
        problem_revision="1",
        action="compile",
        status="queued",
        phase="queued",
        progress=0.0,
        spool_path=f"/temporary/{identifier}",
        payload_json={},
        created_at=utc_now() + timedelta(seconds=created_offset),
    )


def test_draft_upsert_keeps_one_mutable_record_per_problem(repository: Repository) -> None:
    first = repository.upsert_draft("vector-addition", "first")
    second = repository.upsert_draft("vector-addition", "second")

    assert first.id == second.id
    assert repository.get_draft("vector-addition").source_code == "second"  # type: ignore[union-attr]
    assert repository.counts()["drafts"] == 1


def test_claim_next_job_is_fifo_and_atomic(repository: Repository) -> None:
    repository.add_job(make_job("later", created_offset=10))
    repository.add_job(make_job("first", created_offset=0))

    claimed = repository.claim_next_job("worker-a")

    assert claimed is not None
    assert claimed.id == "first"
    assert claimed.status == JobStatus.COMPILING.value
    assert claimed.worker_id == "worker-a"
    assert claimed.started_at is not None
    assert repository.claim_next_job("worker-b").id == "later"  # type: ignore[union-attr]
    assert repository.claim_next_job("worker-c") is None


def test_transition_updates_terminal_metadata_and_clears_spool_reference(
    repository: Repository,
) -> None:
    repository.add_job(make_job("complete-me"))
    repository.claim_next_job("worker")

    completed = repository.transition_job(
        "complete-me",
        JobStatus.SUCCEEDED,
        phase="completed",
        progress=0.8,
        result={"compiled": True},
        diagnostics="ok",
    )

    assert completed.status == "succeeded"
    assert completed.phase == "completed"
    assert completed.progress == 1.0
    assert completed.completed_at is not None
    assert completed.spool_path is None
    assert completed.result_json == {"compiled": True}
    assert completed.diagnostics == "ok"


def test_transition_rejects_unknown_job_and_invalid_state(repository: Repository) -> None:
    with pytest.raises(KeyError, match="unknown job"):
        repository.transition_job("missing", JobStatus.SUCCEEDED)

    repository.add_job(make_job("queued"))
    with pytest.raises(ValueError, match="invalid job transition"):
        repository.transition_job("queued", JobStatus.SUCCEEDED)


def test_single_gpu_lease_has_one_owner_and_can_be_released(repository: Repository) -> None:
    assert not repository.has_active_lease("gpu:0")
    assert repository.acquire_lease("gpu:0", "worker-a")
    assert repository.has_active_lease("gpu:0")
    assert repository.acquire_lease("gpu:0", "worker-a")
    assert not repository.acquire_lease("gpu:0", "worker-b")

    repository.release_lease("gpu:0", "not-owner")
    assert not repository.acquire_lease("gpu:0", "worker-b")
    repository.release_lease("gpu:0", "worker-a")
    assert not repository.has_active_lease("gpu:0")
    assert repository.acquire_lease("gpu:0", "worker-b")


def test_expired_gpu_lease_can_be_taken_over(repository: Repository) -> None:
    assert repository.acquire_lease("gpu:0", "dead-worker", ttl_seconds=-1)
    assert not repository.has_active_lease("gpu:0")
    assert repository.acquire_lease("gpu:0", "replacement")


def test_concurrent_gpu_lease_attempts_have_exactly_one_winner(repository: Repository) -> None:
    contenders = 8
    barrier = Barrier(contenders)

    def acquire(index: int) -> tuple[str, bool]:
        owner = f"worker-{index}"
        barrier.wait()
        return owner, repository.acquire_lease("gpu:0", owner)

    with ThreadPoolExecutor(max_workers=contenders) as executor:
        results = list(executor.map(acquire, range(contenders)))

    winners = [owner for owner, acquired in results if acquired]
    assert len(winners) == 1
    assert repository.has_active_lease("gpu:0")
    repository.release_lease("gpu:0", winners[0])


def test_worker_recovery_fails_only_active_jobs_owned_by_other_workers(
    repository: Repository,
) -> None:
    own = make_job("own")
    other = make_job("other")
    queued = make_job("queued")
    repository.add_job(own)
    repository.add_job(other)
    repository.add_job(queued)
    repository.claim_next_job("current")
    repository.claim_next_job("old-worker")

    failed = repository.fail_orphaned_jobs("current")

    assert failed == ["other"]
    assert repository.get_job("own").status == "compiling"  # type: ignore[union-attr]
    recovered = repository.get_job("other")
    assert recovered is not None
    assert recovered.status == "system_error"
    assert recovered.error_json["retryable"] is True  # type: ignore[index]
    assert repository.get_job("queued").status == "queued"  # type: ignore[union-attr]


def test_environment_snapshot_is_deduplicated_by_fingerprint(repository: Repository) -> None:
    first = repository.save_environment(make_probe("same-fingerprint"))
    second = repository.save_environment(make_probe("same-fingerprint"))

    assert first.id == second.id
    assert repository.latest_environment().id == first.id  # type: ignore[union-attr]


def test_benchmark_can_force_an_immutable_environment_snapshot_with_same_fingerprint(
    repository: Repository,
) -> None:
    status_snapshot = repository.save_environment(make_probe("stable-fingerprint"))
    reused_status = repository.save_environment(make_probe("stable-fingerprint"))
    first_benchmark = repository.save_environment(make_probe("stable-fingerprint"), force_new=True)
    second_benchmark = repository.save_environment(make_probe("stable-fingerprint"), force_new=True)

    assert reused_status.id == status_snapshot.id
    assert first_benchmark.id not in {status_snapshot.id, second_benchmark.id}
    assert second_benchmark.id != status_snapshot.id
    assert {
        status_snapshot.fingerprint,
        first_benchmark.fingerprint,
        second_benchmark.fingerprint,
    } == {"stable-fingerprint"}
    assert repository.latest_environment().id == second_benchmark.id  # type: ignore[union-attr]


def test_draft_version_and_benchmark_survive_engine_recreation(tmp_path: Path) -> None:
    database_path = tmp_path / "persistent.db"
    settings = Settings(
        data_dir=tmp_path / "data",
        database_url_override=f"sqlite:///{database_path.as_posix()}",
        _env_file=None,
    )
    first_engine = build_engine(settings)
    Base.metadata.create_all(first_engine)
    first_repository = Repository(build_session_factory(first_engine))
    draft = first_repository.upsert_draft("vector-addition", "persistent draft")
    version = create_saved_version(
        first_repository,
        source="persistent version source",
        source_digest="p" * 64,
    )
    first_engine.dispose()

    second_engine = build_engine(settings)
    second_repository = Repository(build_session_factory(second_engine))
    try:
        restored_draft = second_repository.get_draft("vector-addition")
        restored_version = second_repository.get_version(version.id)

        assert restored_draft is not None
        assert restored_draft.id == draft.id
        assert restored_draft.source_code == "persistent draft"
        assert restored_version is not None
        assert restored_version.source_code == "persistent version source"
        assert len(restored_version.benchmark_runs) == 1
        assert restored_version.benchmark_runs[0].measurements_json[0]["median_ms"] == 4.0
    finally:
        second_engine.dispose()


def test_version_and_benchmark_are_created_in_one_transaction(repository: Repository) -> None:
    version = create_saved_version(repository, name="manual-save")

    assert repository.counts() == {
        "drafts": 0,
        "jobs": 0,
        "versions": 1,
        "benchmark_runs": 1,
    }
    stored = repository.get_version(version.id)
    assert stored is not None
    assert stored.source_code == "void solve() {}"
    assert stored.correctness_status == "passed"
    assert len(stored.benchmark_runs) == 1


def test_failed_benchmark_insert_rolls_back_version_atomically(repository: Repository) -> None:
    with pytest.raises(IntegrityError):
        repository.create_version_with_benchmark(
            problem_id="vector-addition",
            problem_revision="1",
            name="must-not-exist",
            notes=None,
            source_code="secret source",
            source_hash="a" * 64,
            compile_flags=["-O3"],
            environment_id="missing-environment",
            suite_hash="s" * 64,
            protocol_version="1",
            input_sizes=["64K"],
            seed=42,
            warmup=2,
            iterations=3,
            measurements=[],
            raw_samples=[],
        )

    assert repository.counts()["versions"] == 0
    assert repository.counts()["benchmark_runs"] == 0


def test_duplicate_lookup_is_problem_scoped(repository: Repository) -> None:
    first = create_saved_version(repository, name="one", source_digest="d" * 64)
    create_saved_version(
        repository,
        problem_id="reduction",
        name="other-problem",
        source_digest="d" * 64,
    )

    duplicates = repository.find_duplicate_versions("vector-addition", "d" * 64)

    assert [item.id for item in duplicates] == [first.id]


def test_version_metadata_can_change_without_mutating_snapshot(repository: Repository) -> None:
    version = create_saved_version(repository, name="before", source="immutable")

    renamed = repository.update_version(
        version.id,
        name="after",
        notes="new notes",
        update_notes=True,
    )

    assert renamed is not None
    assert renamed.name == "after"
    assert renamed.notes == "new notes"
    assert renamed.source_code == "immutable"
    assert renamed.source_hash == "a" * 64


def test_delete_version_cascades_benchmark_but_not_environment(repository: Repository) -> None:
    version = create_saved_version(repository)

    assert repository.delete_version(version.id)
    assert not repository.delete_version(version.id)

    assert repository.counts()["versions"] == 0
    assert repository.counts()["benchmark_runs"] == 0
    assert repository.latest_environment() is not None
