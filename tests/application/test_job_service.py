from __future__ import annotations

import stat
from pathlib import Path

import pytest
from myleetgpu.application.jobs import (
    MAX_SOURCE_BYTES,
    DuplicateSourceError,
    JobService,
    JobSubmissionError,
)
from myleetgpu.config import Settings
from myleetgpu.domain.benchmark import source_hash
from myleetgpu.domain.jobs import JobAction, JobStatus
from myleetgpu.domain.problems import ProblemCatalog
from myleetgpu.infrastructure.database import Base, build_engine, build_session_factory
from myleetgpu.infrastructure.repository import Repository

from tests.factories import create_saved_version

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = "// unique-source-marker\nvoid solve() {}\n"


@pytest.fixture
def service_bundle(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "data",
        problems_dir=PROJECT_ROOT / "problems",
        database_url_override=f"sqlite:///{(tmp_path / 'jobs.db').as_posix()}",
        _env_file=None,
    )
    settings.ensure_directories()
    engine = build_engine(settings)
    Base.metadata.create_all(engine)
    repository = Repository(build_session_factory(engine))
    catalog = ProblemCatalog(settings.problems_dir).load()
    service = JobService(settings, catalog, repository)
    try:
        yield settings, repository, service
    finally:
        engine.dispose()


def test_compile_run_and_validate_never_create_versions(service_bundle) -> None:
    _, repository, service = service_bundle

    jobs = [
        service.submit(
            problem_id="vector-addition",
            action=action,
            source=f"{SOURCE}// {action.value}-{index}",
        )
        for action in (JobAction.COMPILE, JobAction.RUN, JobAction.VALIDATE)
        for index in range(3)
    ]

    assert len(jobs) == 9
    assert repository.counts()["jobs"] == 9
    assert repository.counts()["versions"] == 0
    assert repository.counts()["benchmark_runs"] == 0


def test_submission_writes_normalized_private_snapshot_but_not_source_to_job_record(
    service_bundle,
) -> None:
    settings, repository, service = service_bundle
    source = "line one\r\nline two\r"

    job = service.submit(
        problem_id="vector-addition",
        action=JobAction.VALIDATE,
        source=source,
    )

    spool = Path(job.spool_path)  # type: ignore[arg-type]
    snapshot = spool / "source.cu"
    assert snapshot.read_bytes() == b"line one\nline two\n"
    assert stat.S_IMODE(spool.stat().st_mode) == 0o700
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o600
    assert job.source_hash == source_hash(source)
    assert job.payload_json == {}
    assert not hasattr(job, "source_code")

    persisted = repository.get_job(job.id)
    assert persisted is not None
    assert persisted.source_hash == source_hash(source)
    assert "source" not in persisted.payload_json
    database_files = [
        path for path in settings.data_dir.parent.iterdir() if path.name.startswith("jobs.db")
    ]
    assert all(source.encode() not in path.read_bytes() for path in database_files)


def test_triton_submission_uses_an_isolated_python_snapshot(service_bundle) -> None:
    _, repository, service = service_bundle
    source = "def solve(a, b, output, n):\r\n    return None\r\n"

    job = service.submit(
        problem_id="vector-addition",
        language="triton_python",
        action=JobAction.VALIDATE,
        source=source,
    )

    spool = Path(job.spool_path)  # type: ignore[arg-type]
    assert job.language == "triton_python"
    assert (spool / "source.py").read_text(encoding="utf-8") == (
        "def solve(a, b, output, n):\n    return None\n"
    )
    assert not (spool / "source.cu").exists()
    assert repository.get_job(job.id).language == "triton_python"  # type: ignore[union-attr]


def test_torch_only_problem_uses_manifest_default_and_python_snapshot(service_bundle) -> None:
    _, repository, service = service_bundle
    source = (
        "import torch\n\n"
        "def solve(query, key, value, attention_mask):\n"
        "    return torch.matmul(query, value.transpose(-2, -1))\n"
    )

    job = service.submit(
        problem_id="multi-head-attention",
        action=JobAction.VALIDATE,
        source=source,
    )

    spool = Path(job.spool_path)  # type: ignore[arg-type]
    assert job.language == "torch_python"
    assert (spool / "source.py").read_text(encoding="utf-8") == source
    assert not (spool / "source.cu").exists()
    assert repository.get_job(job.id).language == "torch_python"  # type: ignore[union-attr]

    with pytest.raises(JobSubmissionError, match="does not support|不支持"):
        service.submit(
            problem_id="multi-head-attention",
            language="cuda_cpp",
            action=JobAction.COMPILE,
            source=SOURCE,
        )

    with pytest.raises(JobSubmissionError, match="does not support|不支持"):
        service.submit(
            problem_id="multi-head-attention",
            language="",
            action=JobAction.COMPILE,
            source=SOURCE,
        )


def test_language_is_part_of_duplicate_and_rebenchmark_identity(service_bundle) -> None:
    _, repository, service = service_bundle
    existing = create_saved_version(
        repository,
        source=SOURCE,
        source_digest=source_hash(SOURCE),
        language="cuda_cpp",
    )

    triton = service.submit(
        problem_id="vector-addition",
        language="triton_python",
        action=JobAction.SAVE_VERSION,
        source=SOURCE,
        version_name="same bytes, different language",
    )
    assert triton.status == "queued"

    with pytest.raises(JobSubmissionError, match="requested language"):
        service.submit(
            problem_id="vector-addition",
            language="triton_python",
            action=JobAction.REBENCHMARK,
            version_ids=[existing.id],
        )


def test_save_version_submission_only_queues_work_and_freezes_click_time_source(
    service_bundle,
) -> None:
    _, repository, service = service_bundle
    clicked_source = "snapshot before later edits\r\n"

    job = service.submit(
        problem_id="reduction",
        action=JobAction.SAVE_VERSION,
        source=clicked_source,
        version_name="  tuned reduction  ",
        notes="kept only if validation and benchmark pass",
    )

    assert job.action == "save_version"
    assert job.payload_json == {
        "version_name": "tuned reduction",
        "notes": "kept only if validation and benchmark pass",
        "allow_duplicate": False,
    }
    assert Path(job.spool_path, "source.cu").read_text(encoding="utf-8") == (
        "snapshot before later edits\n"
    )
    assert repository.counts()["versions"] == 0
    assert repository.counts()["benchmark_runs"] == 0


@pytest.mark.parametrize(
    ("action", "source", "name", "message"),
    [
        (JobAction.COMPILE, None, None, "source is required"),
        (JobAction.RUN, "  \n", None, "source is required"),
        (JobAction.VALIDATE, "bad\x00source", None, "NUL"),
        (JobAction.SAVE_VERSION, SOURCE, None, "version_name is required"),
        (JobAction.SAVE_VERSION, SOURCE, "x" * 121, "longer than 120"),
    ],
)
def test_invalid_submissions_are_rejected_without_spool_or_database_side_effects(
    service_bundle,
    action: JobAction,
    source: str | None,
    name: str | None,
    message: str,
) -> None:
    settings, repository, service = service_bundle

    with pytest.raises(JobSubmissionError, match=message):
        service.submit(
            problem_id="vector-addition",
            action=action,
            source=source,
            version_name=name,
        )

    assert list(settings.jobs_dir.iterdir()) == []
    assert repository.counts()["jobs"] == 0


def test_source_size_limit_is_measured_in_utf8_bytes(service_bundle) -> None:
    settings, repository, service = service_bundle
    oversized = "测" * (MAX_SOURCE_BYTES // 3 + 1)

    with pytest.raises(JobSubmissionError, match="byte limit"):
        service.submit(
            problem_id="vector-addition",
            action=JobAction.COMPILE,
            source=oversized,
        )

    assert list(settings.jobs_dir.iterdir()) == []
    assert repository.counts()["jobs"] == 0


def test_unknown_problem_is_rejected_without_creating_spool(service_bundle) -> None:
    settings, repository, service = service_bundle

    with pytest.raises(JobSubmissionError, match="unknown problem"):
        service.submit(problem_id="missing", action=JobAction.COMPILE, source=SOURCE)

    assert list(settings.jobs_dir.iterdir()) == []
    assert repository.counts()["jobs"] == 0


def test_repository_failure_removes_snapshot_and_spool(
    service_bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, repository, service = service_bundle

    def fail_add_job(_record: object) -> None:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(repository, "add_job", fail_add_job)

    with pytest.raises(RuntimeError, match="database unavailable"):
        service.submit(
            problem_id="vector-addition",
            action=JobAction.COMPILE,
            source=SOURCE,
        )

    assert list(settings.jobs_dir.iterdir()) == []


def test_rebenchmark_accepts_only_existing_unique_versions_of_the_problem(
    service_bundle,
) -> None:
    _, _, service = service_bundle
    repository = service.repository
    first = create_saved_version(repository, name="first", source_digest="1" * 64)
    second = create_saved_version(repository, name="second", source_digest="2" * 64)

    job = service.submit(
        problem_id="vector-addition",
        action=JobAction.REBENCHMARK,
        version_ids=[first.id, second.id],
    )

    assert job.source_hash is None
    assert job.payload_json == {"version_ids": [first.id, second.id]}
    assert list(Path(job.spool_path).iterdir()) == []  # type: ignore[arg-type]

    for invalid in ([], [first.id, first.id], [first.id, "missing"]):
        with pytest.raises(JobSubmissionError, match="unique version ids|must belong"):
            service.submit(
                problem_id="vector-addition",
                action=JobAction.REBENCHMARK,
                version_ids=invalid,
            )


def test_rebenchmark_rejects_version_from_another_problem(service_bundle) -> None:
    _, repository, service = service_bundle
    other = create_saved_version(repository, problem_id="reduction")

    with pytest.raises(JobSubmissionError, match="must belong"):
        service.submit(
            problem_id="vector-addition",
            action=JobAction.REBENCHMARK,
            version_ids=[other.id],
        )


def test_duplicate_source_requires_explicit_confirmation_only_for_save_version(
    service_bundle,
) -> None:
    settings, repository, service = service_bundle
    existing = create_saved_version(
        repository,
        source=SOURCE,
        source_digest=source_hash(SOURCE),
    )

    with pytest.raises(DuplicateSourceError) as captured:
        service.submit(
            problem_id="vector-addition",
            action=JobAction.SAVE_VERSION,
            source=SOURCE,
            version_name="duplicate",
        )

    assert captured.value.duplicates == [{"id": existing.id, "name": existing.name}]
    assert repository.counts()["jobs"] == 0
    assert list(settings.jobs_dir.iterdir()) == []

    ordinary = service.submit(
        problem_id="vector-addition",
        action=JobAction.VALIDATE,
        source=SOURCE,
    )
    confirmed = service.submit(
        problem_id="vector-addition",
        action=JobAction.SAVE_VERSION,
        source=SOURCE,
        version_name="confirmed duplicate",
        allow_duplicate=True,
    )

    assert ordinary.status == "queued"
    assert confirmed.payload_json["allow_duplicate"] is True
    assert repository.counts()["versions"] == 1


def test_stale_spool_cleanup_preserves_active_job_and_removes_only_orphans(
    service_bundle,
) -> None:
    settings, repository, service = service_bundle
    active = service.submit(
        problem_id="vector-addition",
        action=JobAction.COMPILE,
        source=SOURCE,
    )
    active_path = Path(active.spool_path)  # type: ignore[arg-type]
    orphan = settings.jobs_dir / "orphan"
    orphan.mkdir()
    (orphan / "artifact").write_text("temporary", encoding="utf-8")

    assert service.cleanup_stale_spool() == ["orphan"]
    assert active_path.is_dir()
    assert not orphan.exists()

    repository.transition_job(active.id, JobStatus.CANCELLED)
    assert service.cleanup_stale_spool() == [active.id]
    assert not active_path.exists()


def test_spool_path_guard_rejects_root_nested_and_outside_paths(service_bundle) -> None:
    settings, _, service = service_bundle

    for unsafe in (
        settings.jobs_dir,
        settings.jobs_dir / "job" / "nested",
        settings.data_dir / "outside",
    ):
        with pytest.raises(JobSubmissionError, match="escaped"):
            service._ensure_safe_spool_path(unsafe)


def test_stale_cleanup_does_not_follow_symlink_outside_spool(service_bundle) -> None:
    settings, _, service = service_bundle
    outside = settings.data_dir.parent / "must-survive"
    outside.mkdir()
    marker = outside / "user-data"
    marker.write_text("keep", encoding="utf-8")
    link = settings.jobs_dir / "malicious-link"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(JobSubmissionError, match="escaped"):
        service.cleanup_stale_spool()

    assert marker.read_text(encoding="utf-8") == "keep"
