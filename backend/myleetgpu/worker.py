from __future__ import annotations

import logging
import os
import shutil
import signal
import socket
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from myleetgpu.application.execution import failed_probe, probe_response
from myleetgpu.application.judging import JobExecutor
from myleetgpu.application.results import JobFailed
from myleetgpu.config import Settings, get_settings
from myleetgpu.domain.jobs import GPU_RESOURCE, ErrorCode, JobError, JobStatus
from myleetgpu.domain.problems import KernelLanguage, ProblemCatalog
from myleetgpu.infrastructure.database import Base, build_engine, build_session_factory
from myleetgpu.infrastructure.logging import configure_logging
from myleetgpu.infrastructure.repository import Repository
from myleetgpu.runner.models import EnvironmentProbe, RunnerFailure
from myleetgpu.runner.protocols import CpuEnvironmentRunner, PreparedRunner, Runner, TargetRunner
from myleetgpu.runner.router import ExecutionRouter

LOGGER = logging.getLogger("myleetgpu.worker")


class Worker:
    def __init__(
        self,
        settings: Settings,
        catalog: ProblemCatalog,
        repository: Repository,
        runner: Runner,
        *,
        worker_id: str | None = None,
    ):
        self.settings = settings
        self.catalog = catalog
        self.repository = repository
        self.runner = runner
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.stopping = threading.Event()
        self._lease_thread: threading.Thread | None = None
        self._last_environment_probe = 0.0
        self._lease_required = False
        self.runner.assign_owner(self.worker_id)
        self.executor = JobExecutor(
            settings, catalog, repository, runner, check_lease=self._assert_gpu_lease
        )

    def run_forever(self) -> None:
        if not self.repository.acquire_lease(GPU_RESOURCE, self.worker_id):
            raise RuntimeError("another worker holds the single-GPU lease")
        self._lease_required = True
        self._lease_thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._lease_thread.start()
        try:
            self._probe_cpu_environments()
            orphaned = self.repository.fail_orphaned_jobs(self.worker_id)
            # Local source snapshots can be reclaimed immediately. Remote/device
            # orphans are cleaned lazily before that provider's first real use.
            self._cleanup_job_ids(orphaned, local_only=True)
            while not self.stopping.is_set():
                if not self.process_next():
                    if time.monotonic() - self._last_environment_probe >= 60:
                        self._probe_cpu_environments()
                    self.stopping.wait(self.settings.job_poll_seconds)
        finally:
            self.stopping.set()
            if self._lease_thread is not None:
                self._lease_thread.join(timeout=6)
            self.repository.release_lease(GPU_RESOURCE, self.worker_id)
            self._lease_required = False

    def process_next(self) -> bool:
        if isinstance(self.runner, TargetRunner) and self._process_execution_probe():
            return True
        job = self.repository.claim_next_job(self.worker_id)
        if job is None:
            return False
        spool = Path(job.spool_path) if job.spool_path else self.settings.jobs_dir / job.id
        log_fields = {
            "job_id": job.id,
            "worker_id": self.worker_id,
            "action": job.action,
            "problem_id": job.problem_id,
        }
        LOGGER.info("job started", extra=log_fields)
        try:
            target = (
                "cpu"
                if KernelLanguage(job.language).is_cpu
                else job.payload_json.get("execution_target", "local")
            )
            if isinstance(self.runner, TargetRunner):
                if target == "colab" and not job.payload_json.get("colab_acknowledged"):
                    raise RunnerFailure("Colab 任务缺少可信代码执行确认")
                self.runner.select_target(target)
            if isinstance(self.runner, PreparedRunner):
                self.runner.prepare_target()
            result = self.executor.execute(job, spool)
            result["execution_target"] = target
            self.repository.transition_job(
                job.id,
                JobStatus.SUCCEEDED,
                phase="completed",
                result=result,
                progress=1.0,
            )
            LOGGER.info("job succeeded", extra={**log_fields, "status": "succeeded"})
        except JobFailed as failure:
            status = (
                JobStatus.TIMED_OUT if failure.error.code is ErrorCode.TIMEOUT else JobStatus.FAILED
            )
            self.repository.transition_job(
                job.id,
                status,
                phase=failure.error.stage,
                error=failure.error.model_dump(mode="json"),
                diagnostics=failure.diagnostics,
            )
            LOGGER.warning("job rejected", extra={**log_fields, "status": status.value})
        except RunnerFailure as failure:
            if isinstance(self.runner, PreparedRunner):
                self.runner.invalidate_preparation()
            error = JobError(
                code=ErrorCode.RUNNER_UNHEALTHY,
                message=str(failure),
                stage="runner",
                retryable=True,
            )
            self.repository.transition_job(
                job.id,
                JobStatus.SYSTEM_ERROR,
                phase="runner",
                error=error.model_dump(mode="json"),
            )
            LOGGER.error("runner unavailable", extra={**log_fields, "status": "system_error"})
        except BaseException as failure:
            LOGGER.exception("job %s failed with a system error", job.id)
            error = JobError(
                code=ErrorCode.INTERNAL_ERROR,
                message="平台处理任务时发生内部错误",
                stage="worker",
                retryable=True,
                details={"type": type(failure).__name__},
            )
            self.repository.transition_job(
                job.id,
                JobStatus.SYSTEM_ERROR,
                phase="worker",
                error=error.model_dump(mode="json"),
            )
        finally:
            try:
                self.runner.cleanup_task(spool)
            except (OSError, ValueError, RunnerFailure):
                LOGGER.exception("failed to clean spool for job %s", job.id)
        return True

    def stop(self) -> None:
        self.stopping.set()

    def _heartbeat(self) -> None:
        while not self.stopping.wait(5):
            try:
                if not self.repository.acquire_lease(GPU_RESOURCE, self.worker_id):
                    self._handle_lease_loss("single-GPU lease was lost")
                    return
            except Exception:
                LOGGER.exception("failed to renew the single-GPU lease")
                self._handle_lease_loss("single-GPU lease renewal failed")
                return

    def _assert_gpu_lease(self) -> None:
        if not self._lease_required:
            return
        try:
            owned = self.repository.owns_active_lease(GPU_RESOURCE, self.worker_id)
        except Exception as error:
            self._handle_lease_loss("single-GPU lease check failed")
            raise RunnerFailure("single-GPU lease check failed") from error
        if not owned:
            self._handle_lease_loss("single-GPU lease was lost")
            raise RunnerFailure("single-GPU lease was lost")

    def _handle_lease_loss(self, reason: str) -> None:
        LOGGER.critical(reason)
        self.stopping.set()
        try:
            removed = self.runner.cleanup_owned_containers()
            if removed:
                LOGGER.warning("stopped %d containers after lease loss", len(removed))
        except Exception:
            LOGGER.exception("failed to stop containers after lease loss")

    def _probe_and_record_environment(self) -> EnvironmentProbe:
        if isinstance(self.runner, TargetRunner):
            self.runner.select_target(self.repository.execution_settings()["target"])
        probe = self.runner.probe_environment(
            force=True,
            ignore_circuit_breaker=True,
        )
        self.repository.save_environment(probe)
        self._last_environment_probe = time.monotonic()
        return probe

    def _process_execution_probe(self) -> bool:
        request = self.repository.claim_execution_probe()
        if request is None:
            return False
        try:
            self._assert_gpu_lease()
            assert isinstance(self.runner, TargetRunner)
            self.runner.select_target(request.target)
            if isinstance(self.runner, PreparedRunner):
                self.runner.prepare_target()
            probe = self.runner.probe_connection(request.language)
            if not probe.healthy and isinstance(self.runner, PreparedRunner):
                self.runner.invalidate_preparation()
            self.repository.save_environment(probe)
            result = probe_response(probe)
        except RunnerFailure as error:
            if isinstance(self.runner, PreparedRunner):
                self.runner.invalidate_preparation()
            probe = failed_probe(
                request.target, request.language, str(error), self.settings.cuda_image
            )
            self.repository.save_environment(probe)
            result = probe_response(probe)
        except Exception:
            if isinstance(self.runner, PreparedRunner):
                self.runner.invalidate_preparation()
            LOGGER.exception("execution environment probe failed")
            probe = failed_probe(
                request.target,
                request.language,
                "运行环境探测失败，请检查 Worker 日志和已配置的执行连接",
                self.settings.cuda_image,
            )
            self.repository.save_environment(probe)
            result = probe_response(probe)
        self.repository.complete_execution_probe(request.id, result)
        return True

    def _probe_cpu_environments(self) -> None:
        """Record CPU readiness even when the configured GPU provider is unavailable."""
        self._last_environment_probe = time.monotonic()
        if not isinstance(self.runner, CpuEnvironmentRunner):
            return
        if isinstance(self.runner, TargetRunner):
            try:
                self.runner.select_target("cpu")
            except RunnerFailure:
                LOGGER.warning("configured router has no CPU execution adapter")
                return
        for language in (KernelLanguage.CPP, KernelLanguage.PYTHON):
            try:
                probe = self.runner.probe_cpu_environment(language.value, force=True)
                self.repository.save_environment(probe)
            except (OSError, RunnerFailure):
                LOGGER.warning("CPU toolchain probe failed for %s", language, exc_info=True)

    def _cleanup_job_ids(self, job_ids: list[str], *, local_only: bool = False) -> None:
        for job_id in job_ids:
            path = self.settings.jobs_dir / job_id
            try:
                if local_only:
                    resolved = path.resolve()
                    if resolved.parent != self.settings.jobs_dir.resolve() or not resolved.name:
                        raise ValueError("orphan source path escaped the configured job spool")
                    if resolved.exists():
                        shutil.rmtree(resolved)
                    continue
                if isinstance(self.runner, TargetRunner):
                    job = self.repository.get_job(job_id)
                    target = job.payload_json.get("execution_target", "local") if job else "local"
                    self.runner.select_target(target)
                self.runner.cleanup_task(path)
            except (OSError, ValueError, RunnerFailure):
                LOGGER.exception("failed to clean orphaned job %s", job_id)


@contextmanager
def create_worker(settings: Settings | None = None) -> Iterator[Worker]:
    """Own process resources, including failed startup and exceptional shutdown."""
    settings = settings or get_settings()
    settings.ensure_directories()
    engine = build_engine(settings)
    try:
        Base.metadata.create_all(engine)
        factory: sessionmaker[Session] = build_session_factory(engine)
        catalog = ProblemCatalog(settings.problems_dir).load()
        yield Worker(settings, catalog, Repository(factory), ExecutionRouter(settings))
    finally:
        engine.dispose()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    with create_worker(settings) as worker:

        def stop_worker(_signum: int, _frame: Any) -> None:
            worker.stop()

        signal.signal(signal.SIGTERM, stop_worker)
        signal.signal(signal.SIGINT, stop_worker)
        LOGGER.info("worker %s started", worker.worker_id)
        worker.run_forever()


if __name__ == "__main__":
    main()
