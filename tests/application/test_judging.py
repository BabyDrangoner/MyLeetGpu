from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from myleetgpu.application.judging import JobExecutor
from myleetgpu.application.results import JobFailed
from myleetgpu.config import Settings
from myleetgpu.domain.benchmark import source_hash
from myleetgpu.domain.problems import ProblemCatalog
from myleetgpu.infrastructure.models import JobRecord
from myleetgpu.infrastructure.repository import Repository
from myleetgpu.runner.models import CompileResult, ExecutionResult, RunnerFailure
from myleetgpu.runner.protocols import Runner

from tests.factories import make_probe


@pytest.fixture
def judging_bundle(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        problems_dir=Path(__file__).resolve().parents[2] / "problems",
        _env_file=None,
    )
    settings.ensure_directories()
    catalog = ProblemCatalog(settings.problems_dir).load()
    repository = Mock(spec=Repository)
    repository.find_duplicate_versions.return_value = []
    runner = Mock(spec=Runner)
    lease = Mock()
    executor = JobExecutor(settings, catalog, repository, runner, check_lease=lease)
    spool = settings.jobs_dir / "test-job"
    spool.mkdir()
    source = "void solve() {}\n"
    (spool / "source.cu").write_text(source)
    runner.compile.return_value = CompileResult(True, "ok", spool / "validator", 0.1)
    runner.execute.return_value = ExecutionResult(
        True, "public output", {"status": "passed", "total": 2, "passed": 2}, 0.1, 0
    )
    runner.probe_environment.return_value = make_probe()
    job = JobRecord(
        id="test-job",
        problem_id="vector-addition",
        problem_revision="1",
        language="cuda_cpp",
        action="compile",
        status="compiling",
        phase="compiling",
        source_hash=source_hash(source),
        spool_path=str(spool),
        payload_json={},
    )
    return executor, repository, runner, lease, job, spool


def test_compile_use_case_does_not_need_worker_loop_database_or_gpu_lease(judging_bundle):
    executor, repository, runner, lease, job, spool = judging_bundle
    result = executor.execute(job, spool)
    assert result["compiled"] is True
    lease.assert_not_called()
    runner.execute.assert_not_called()
    repository.claim_next_job.assert_not_called()
    repository.acquire_lease.assert_not_called()
    repository.create_version_with_benchmark.assert_not_called()
    runner.cleanup_task.assert_not_called()  # Resource ownership remains with the host.


def test_execution_checks_injected_lease_before_any_gpu_call(judging_bundle):
    executor, repository, runner, lease, job, spool = judging_bundle
    job.action = "run"
    lease.side_effect = RunnerFailure("lease lost")
    with pytest.raises(RunnerFailure, match="lease lost"):
        executor.execute(job, spool)
    lease.assert_called_once()
    runner.execute.assert_not_called()
    runner.probe_environment.assert_not_called()
    repository.create_version_with_benchmark.assert_not_called()


def test_ordinary_run_returns_public_result_without_version_persistence(judging_bundle):
    executor, repository, runner, lease, job, spool = judging_bundle
    job.action = "run"
    result = executor.execute(job, spool)
    assert result["correctness"]["status"] == "passed"
    assert result["output"] == "public output"
    lease.assert_called_once()
    repository.save_environment.assert_called_once()
    repository.create_version_with_benchmark.assert_not_called()
    assert runner.execute.call_args.kwargs["mode"] == "public"


def test_source_integrity_precedes_compilation(judging_bundle):
    executor, repository, runner, lease, job, spool = judging_bundle
    (spool / "source.cu").write_text("changed after submission")
    with pytest.raises(JobFailed) as caught:
        executor.execute(job, spool)
    assert caught.value.error.stage == "spool"
    runner.compile.assert_not_called()
    lease.assert_not_called()
    repository.create_version_with_benchmark.assert_not_called()
