from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from myleetgpu.application.jobs import JobService
from myleetgpu.config import Settings
from myleetgpu.domain.benchmark import source_hash
from myleetgpu.domain.jobs import JobAction
from myleetgpu.domain.problems import Problem, ProblemCatalog
from myleetgpu.infrastructure.database import Base, build_engine, build_session_factory
from myleetgpu.infrastructure.repository import Repository
from myleetgpu.runner.models import CompileResult, EnvironmentProbe, ExecutionResult
from myleetgpu.worker import Worker

from tests.factories import create_saved_version, make_probe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "// worker test source\nvoid solve() {}\n"


class FakeRunner:
    """Deterministic CPU-only test double; never presented as a GPU acceptance test."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.compile_failure: CompileResult | None = None
        self.execution_overrides: dict[str, ExecutionResult] = {}
        self.full_execution_factory: Callable[[int], ExecutionResult] | None = None
        self.full_calls = 0
        self.last_problem: Problem | None = None
        self.owner: str | None = None
        self.cleaned_owned_containers = False

    def assign_owner(self, owner: str) -> None:
        self.owner = owner

    def cleanup_owned_containers(self) -> list[str]:
        self.cleaned_owned_containers = True
        return ["owned-container"]

    def compile(
        self,
        task_root: Path,
        problem: Problem,
        source_path: Path,
        *,
        harness_kind: str = "validator",
    ) -> CompileResult:
        self.calls.append(("compile", harness_kind, task_root, source_path))
        self.last_problem = problem
        if self.compile_failure is not None:
            return self.compile_failure
        executable = task_root / f"fake-{harness_kind}-program"
        executable.write_bytes(b"not executed by this CPU-only test double")
        return CompileResult(True, "fake nvcc ok", executable, 0.01)

    def execute(
        self,
        task_root: Path,
        executable: Path,
        *,
        mode: str,
        timeout_seconds: float,
    ) -> ExecutionResult:
        self.calls.append(("execute", mode, task_root, executable, timeout_seconds))
        if mode == "full":
            self.full_calls += 1
            if self.full_execution_factory is not None:
                return self.full_execution_factory(self.full_calls)
        if mode in self.execution_overrides:
            return self.execution_overrides[mode]
        if mode == "benchmark":
            assert self.last_problem is not None
            benchmark = self.last_problem.manifest.benchmark
            parsed = {
                "status": "passed",
                "protocol_version": benchmark.protocol_version,
                "measurements": [
                    {
                        "label": size.label,
                        "samples_ms": [float(index + 1)] * benchmark.iterations,
                        "inner_repetitions": size.inner_repetitions,
                    }
                    for index, size in enumerate(benchmark.sizes)
                ],
            }
        else:
            parsed = {"status": "passed", "passed": 6, "total": 6}
        return ExecutionResult(True, "fake platform output", parsed, 0.02, 0)

    def probe_environment(self, *, force: bool = False):
        self.calls.append(("probe_environment", force))
        return make_probe("fake-runner-env")

    @staticmethod
    def effective_compile_flags(problem: Problem, probe: EnvironmentProbe) -> list[str]:
        return [*problem.compile_flags, f"-arch=sm_{probe.cuda_arch}"]

    def cleanup_task(self, task_root: Path) -> None:
        self.calls.append(("cleanup_task", task_root))
        shutil.rmtree(task_root, ignore_errors=True)


@pytest.fixture
def worker_bundle(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "data",
        problems_dir=PROJECT_ROOT / "problems",
        database_url_override=f"sqlite:///{(tmp_path / 'worker.db').as_posix()}",
        _env_file=None,
    )
    settings.ensure_directories()
    engine = build_engine(settings)
    Base.metadata.create_all(engine)
    repository = Repository(build_session_factory(engine))
    catalog = ProblemCatalog(settings.problems_dir).load()
    service = JobService(settings, catalog, repository)
    runner = FakeRunner()
    worker = Worker(
        settings,
        catalog,
        repository,
        runner,  # type: ignore[arg-type]
        worker_id="cpu-unit-test-worker",
    )
    try:
        yield settings, repository, service, runner, worker
    finally:
        engine.dispose()


@pytest.mark.parametrize("action", [JobAction.COMPILE, JobAction.RUN, JobAction.VALIDATE])
def test_ordinary_worker_actions_succeed_without_creating_versions(
    worker_bundle,
    action: JobAction,
) -> None:
    _, repository, service, runner, worker = worker_bundle
    submitted = service.submit(
        problem_id="vector-addition",
        action=action,
        source=SOURCE,
    )
    spool = Path(submitted.spool_path)  # type: ignore[arg-type]

    assert worker.process_next()

    completed = repository.get_job(submitted.id)
    assert completed is not None
    assert completed.status == "succeeded"
    assert completed.progress == 1.0
    assert completed.spool_path is None
    assert not spool.exists()
    assert repository.counts()["versions"] == 0
    assert repository.counts()["benchmark_runs"] == 0
    assert any(call[0] == "cleanup_task" for call in runner.calls)


def test_worker_reports_no_work_without_calling_runner(worker_bundle) -> None:
    _, _, _, runner, worker = worker_bundle

    assert not worker.process_next()
    assert runner.calls == []


def test_lease_loss_stops_worker_and_owned_containers(worker_bundle) -> None:
    _, _, _, runner, worker = worker_bundle

    worker._handle_lease_loss("test lease loss")

    assert worker.stopping.is_set()
    assert runner.cleaned_owned_containers is True


def test_save_version_validates_then_benchmarks_then_persists_atomically(
    worker_bundle,
) -> None:
    _, repository, service, runner, worker = worker_bundle
    status_snapshot = repository.save_environment(make_probe("fake-runner-env"))
    source = "click-time immutable snapshot\r\n"
    submitted = service.submit(
        problem_id="vector-addition",
        action=JobAction.SAVE_VERSION,
        source=source,
        version_name="manual tuned",
        notes="first real measurement",
    )

    assert worker.process_next()

    completed = repository.get_job(submitted.id)
    assert completed is not None
    assert completed.status == "succeeded"
    assert completed.result_json is not None
    version = repository.get_version(completed.result_json["version_id"])
    assert version is not None
    assert version.name == "manual tuned"
    assert version.notes == "first real measurement"
    assert version.source_code == "click-time immutable snapshot\n"
    assert version.correctness_status == "passed"
    assert version.compile_flags_json == ["--std=c++17", "-O3", "-arch=sm_89"]
    assert version.environment_snapshot_id != status_snapshot.id
    assert version.environment.fingerprint == status_snapshot.fingerprint
    assert len(version.benchmark_runs) == 1
    measurements = version.benchmark_runs[0].measurements_json
    assert [item["size"] for item in measurements] == ["64K", "1M", "16M"]
    assert all(item["median_ms"] is not None for item in measurements)
    assert repository.counts()["versions"] == 1
    assert repository.counts()["benchmark_runs"] == 1

    stage_calls = [(call[0], call[1]) for call in runner.calls if call[0] in {"compile", "execute"}]
    assert stage_calls == [
        ("compile", "validator"),
        ("execute", "full"),
        ("compile", "benchmark"),
        ("execute", "benchmark"),
    ]


def test_compile_failure_does_not_execute_or_create_version(worker_bundle) -> None:
    _, repository, service, runner, worker = worker_bundle
    runner.compile_failure = CompileResult(
        False,
        "source.cu:4: error: expected ';'",
        None,
        0.01,
    )
    submitted = service.submit(
        problem_id="vector-addition",
        action=JobAction.SAVE_VERSION,
        source=SOURCE,
        version_name="must not persist",
    )

    assert worker.process_next()

    failed = repository.get_job(submitted.id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_json["code"] == "compile_error"  # type: ignore[index]
    assert failed.diagnostics == "source.cu:4: error: expected ';'"
    assert not any(call[0] == "execute" for call in runner.calls)
    assert repository.counts()["versions"] == 0


def test_wrong_answer_stops_before_benchmark_and_creates_no_version(worker_bundle) -> None:
    _, repository, service, runner, worker = worker_bundle
    runner.execution_overrides["full"] = ExecutionResult(
        False,
        "safe wrong-answer summary",
        {"status": "wrong_answer", "passed": 5, "total": 6},
        0.02,
        0,
    )
    submitted = service.submit(
        problem_id="vector-addition",
        action=JobAction.SAVE_VERSION,
        source=SOURCE,
        version_name="must not persist",
    )

    worker.process_next()

    failed = repository.get_job(submitted.id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_json["code"] == "wrong_answer"  # type: ignore[index]
    assert failed.diagnostics is None
    assert "safe wrong-answer summary" not in str(failed.error_json)
    assert not any(call[:2] == ("compile", "benchmark") for call in runner.calls)
    assert repository.counts()["versions"] == 0


def test_full_validation_does_not_return_untrusted_stdout(worker_bundle) -> None:
    _, repository, service, runner, worker = worker_bundle
    runner.execution_overrides["full"] = ExecutionResult(
        True,
        "sensitive internal input printed by submitted code",
        {"status": "passed", "passed": 6, "total": 6},
        0.02,
        0,
    )
    submitted = service.submit(
        problem_id="vector-addition",
        action=JobAction.VALIDATE,
        source=SOURCE,
    )

    worker.process_next()

    completed = repository.get_job(submitted.id)
    assert completed is not None
    assert completed.status == "succeeded"
    assert completed.result_json is not None
    assert completed.result_json["output"] is None
    assert "sensitive internal input" not in str(completed.result_json)


@pytest.mark.parametrize(
    "benchmark_result",
    [
        ExecutionResult(False, "kernel failed", None, 0.1, 1),
        ExecutionResult(
            True,
            "malformed benchmark",
            {"status": "passed", "protocol_version": "wrong", "measurements": []},
            0.1,
            0,
        ),
        ExecutionResult(
            True,
            "too few samples",
            {
                "status": "passed",
                "protocol_version": "1",
                "measurements": [
                    {"label": "64K", "samples_ms": [1.0]},
                    {"label": "1M", "samples_ms": [1.0]},
                    {"label": "16M", "samples_ms": [1.0]},
                ],
            },
            0.1,
            0,
        ),
    ],
    ids=["runtime-failure", "protocol-mismatch", "sample-count-mismatch"],
)
def test_failed_or_malformed_benchmark_never_creates_a_partial_version(
    worker_bundle,
    benchmark_result: ExecutionResult,
) -> None:
    _, repository, service, runner, worker = worker_bundle
    runner.execution_overrides["benchmark"] = benchmark_result
    submitted = service.submit(
        problem_id="vector-addition",
        action=JobAction.SAVE_VERSION,
        source=SOURCE,
        version_name="must roll back",
    )

    worker.process_next()

    failed = repository.get_job(submitted.id)
    assert failed is not None and failed.status == "failed"
    assert repository.counts()["versions"] == 0
    assert repository.counts()["benchmark_runs"] == 0


def test_database_failure_during_final_save_is_system_error_without_partial_version(
    worker_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, repository, service, _, worker = worker_bundle

    def fail_transaction(**_values: object) -> None:
        raise RuntimeError("commit failed")

    monkeypatch.setattr(repository, "create_version_with_benchmark", fail_transaction)
    submitted = service.submit(
        problem_id="vector-addition",
        action=JobAction.SAVE_VERSION,
        source=SOURCE,
        version_name="must roll back",
    )

    worker.process_next()

    failed = repository.get_job(submitted.id)
    assert failed is not None and failed.status == "system_error"
    assert failed.error_json["code"] == "internal_error"  # type: ignore[index]
    assert repository.counts()["versions"] == 0
    assert repository.counts()["benchmark_runs"] == 0


def test_tampered_spool_snapshot_is_rejected_before_compile(worker_bundle) -> None:
    _, repository, service, runner, worker = worker_bundle
    submitted = service.submit(
        problem_id="vector-addition",
        action=JobAction.VALIDATE,
        source=SOURCE,
    )
    Path(submitted.spool_path, "source.cu").write_text("tampered", encoding="utf-8")

    worker.process_next()

    failed = repository.get_job(submitted.id)
    assert failed is not None and failed.status == "failed"
    assert failed.error_json["code"] == "internal_error"  # type: ignore[index]
    assert failed.phase == "spool"
    assert not any(call[0] == "compile" for call in runner.calls)


def test_unconfirmed_duplicate_created_while_queued_is_rejected_before_compile(
    worker_bundle,
) -> None:
    _, repository, service, runner, worker = worker_bundle
    submitted = service.submit(
        problem_id="vector-addition",
        action=JobAction.SAVE_VERSION,
        source=SOURCE,
        version_name="queued candidate",
    )
    create_saved_version(
        repository,
        source=SOURCE,
        source_digest=submitted.source_hash,
    )

    worker.process_next()

    failed = repository.get_job(submitted.id)
    assert failed is not None and failed.status == "failed"
    assert failed.phase == "queued"
    assert repository.counts()["versions"] == 1
    assert not any(call[0] == "compile" for call in runner.calls)


def test_unconfirmed_duplicate_created_during_benchmark_is_rechecked_before_save(
    worker_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, repository, service, runner, worker = worker_bundle
    submitted = service.submit(
        problem_id="vector-addition",
        action=JobAction.SAVE_VERSION,
        source=SOURCE,
        version_name="racing candidate",
    )
    original_execute = runner.execute

    def execute_with_race(
        task_root: Path,
        executable: Path,
        *,
        mode: str,
        timeout_seconds: float,
    ) -> ExecutionResult:
        result = original_execute(
            task_root,
            executable,
            mode=mode,
            timeout_seconds=timeout_seconds,
        )
        if mode == "benchmark":
            create_saved_version(
                repository,
                source=SOURCE,
                source_digest=submitted.source_hash,
            )
        return result

    monkeypatch.setattr(runner, "execute", execute_with_race)

    worker.process_next()

    failed = repository.get_job(submitted.id)
    assert failed is not None and failed.status == "failed"
    assert failed.phase == "benchmarking"
    assert repository.counts()["versions"] == 1


def test_confirmed_duplicate_is_allowed_through_worker_persistence(worker_bundle) -> None:
    _, repository, service, _, worker = worker_bundle
    create_saved_version(
        repository,
        source=SOURCE,
        source_digest=source_hash(SOURCE),
    )
    submitted = service.submit(
        problem_id="vector-addition",
        action=JobAction.SAVE_VERSION,
        source=SOURCE,
        version_name="confirmed duplicate",
        allow_duplicate=True,
    )

    worker.process_next()

    completed = repository.get_job(submitted.id)
    assert completed is not None and completed.status == "succeeded"
    assert repository.counts()["versions"] == 2
    assert repository.counts()["benchmark_runs"] == 2


def test_rebenchmark_adds_history_without_changing_version_count(worker_bundle) -> None:
    _, repository, service, _, worker = worker_bundle
    first = create_saved_version(repository, source="first", source_digest="1" * 64)
    second = create_saved_version(repository, source="second", source_digest="2" * 64)
    submitted = service.submit(
        problem_id="vector-addition",
        action=JobAction.REBENCHMARK,
        version_ids=[first.id, second.id],
    )

    worker.process_next()

    completed = repository.get_job(submitted.id)
    assert completed is not None and completed.status == "succeeded"
    assert repository.counts()["versions"] == 2
    assert repository.counts()["benchmark_runs"] == 4
    assert len(repository.get_version(first.id).benchmark_runs) == 2  # type: ignore[union-attr]
    assert len(repository.get_version(second.id).benchmark_runs) == 2  # type: ignore[union-attr]


def test_rebenchmark_failure_on_later_version_adds_no_partial_runs(worker_bundle) -> None:
    _, repository, service, runner, worker = worker_bundle
    first = create_saved_version(repository, source="first", source_digest="1" * 64)
    second = create_saved_version(repository, source="second", source_digest="2" * 64)

    def full_result(call_number: int) -> ExecutionResult:
        if call_number == 2:
            return ExecutionResult(
                False,
                "second version is now wrong",
                {"status": "wrong_answer"},
                0.02,
                0,
            )
        return ExecutionResult(
            True,
            "passed",
            {"status": "passed", "passed": 6, "total": 6},
            0.02,
            0,
        )

    runner.full_execution_factory = full_result
    submitted = service.submit(
        problem_id="vector-addition",
        action=JobAction.REBENCHMARK,
        version_ids=[first.id, second.id],
    )

    worker.process_next()

    failed = repository.get_job(submitted.id)
    assert failed is not None and failed.status == "failed"
    assert repository.counts()["versions"] == 2
    assert repository.counts()["benchmark_runs"] == 2
