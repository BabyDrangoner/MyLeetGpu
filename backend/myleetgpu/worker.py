from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from myleetgpu.config import Settings, get_settings
from myleetgpu.domain.benchmark import Measurement, source_hash
from myleetgpu.domain.jobs import GPU_RESOURCE, ErrorCode, JobAction, JobError, JobStatus
from myleetgpu.domain.problems import Problem, ProblemCatalog
from myleetgpu.infrastructure.database import Base, build_engine, build_session_factory
from myleetgpu.infrastructure.logging import configure_logging
from myleetgpu.infrastructure.models import JobRecord, VersionRecord
from myleetgpu.infrastructure.repository import Repository
from myleetgpu.runner.docker import DockerRunner
from myleetgpu.runner.models import CompileResult, EnvironmentProbe, ExecutionResult, RunnerFailure

LOGGER = logging.getLogger("myleetgpu.worker")


class JobFailed(RuntimeError):
    def __init__(self, error: JobError, diagnostics: str | None = None):
        super().__init__(error.message)
        self.error = error
        self.diagnostics = diagnostics


class Worker:
    def __init__(
        self,
        settings: Settings,
        catalog: ProblemCatalog,
        repository: Repository,
        runner: DockerRunner,
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

    def run_forever(self) -> None:
        if not self.repository.acquire_lease(GPU_RESOURCE, self.worker_id):
            raise RuntimeError("another worker holds the single-GPU lease")
        self._lease_required = True
        self._lease_thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._lease_thread.start()
        try:
            removed = self.runner.cleanup_orphan_containers()
            if removed:
                LOGGER.warning("removed %d orphaned runner containers", len(removed))
            probe = self._probe_and_record_environment()
            if not probe.healthy:
                LOGGER.error("GPU environment is unavailable: %s", probe.error)
            orphaned = self.repository.fail_orphaned_jobs(self.worker_id)
            self._cleanup_job_ids(orphaned)
            while not self.stopping.is_set():
                if not self.process_next():
                    if time.monotonic() - self._last_environment_probe >= 60:
                        self._probe_and_record_environment()
                    self.stopping.wait(self.settings.job_poll_seconds)
        finally:
            self.stopping.set()
            if self._lease_thread is not None:
                self._lease_thread.join(timeout=6)
            self.repository.release_lease(GPU_RESOURCE, self.worker_id)
            self._lease_required = False

    def process_next(self) -> bool:
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
            result = self._process(job, spool)
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
            except (OSError, ValueError):
                LOGGER.exception("failed to clean spool for job %s", job.id)
        return True

    def stop(self) -> None:
        self.stopping.set()

    def _process(self, job: JobRecord, spool: Path) -> dict[str, Any]:
        problem = self.catalog.get(job.problem_id)
        if problem.manifest.revision != job.problem_revision:
            raise JobFailed(
                JobError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="排队期间题目版本发生变化，请重新提交",
                    stage="queued",
                )
            )
        action = JobAction(job.action)
        if action is JobAction.REBENCHMARK:
            return self._rebenchmark(job, spool, problem)

        source_path = self._verified_source(job, spool)
        if (
            action is JobAction.SAVE_VERSION
            and not job.payload_json.get("allow_duplicate")
            and self.repository.find_duplicate_versions(
                problem.manifest.slug, job.source_hash or ""
            )
        ):
            raise JobFailed(
                JobError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="相同源码已存在，请确认重复后重试",
                    stage="queued",
                )
            )
        validator = self.runner.compile(spool, problem, source_path, harness_kind="validator")
        self._require_compile(validator)
        if action is JobAction.COMPILE:
            return {
                "compiled": True,
                "diagnostics": validator.diagnostics,
                "duration_seconds": validator.duration_seconds,
            }

        mode = "public" if action is JobAction.RUN else "full"
        stage_status = JobStatus.RUNNING if action is JobAction.RUN else JobStatus.VALIDATING
        self.repository.transition_job(
            job.id,
            stage_status,
            phase=mode,
            progress=0.45 if action is JobAction.RUN else 0.35,
            diagnostics=validator.diagnostics,
        )
        timeout = self._execution_timeout(problem, mode)
        self._assert_gpu_lease()
        validation = self.runner.execute(
            spool,
            validator.executable,
            mode=mode,
            timeout_seconds=timeout,  # type: ignore[arg-type]
        )
        safe_correctness = self._safe_correctness(problem, mode, validation.parsed)
        self._require_execution(validation, mode, safe_correctness)
        if action in {JobAction.RUN, JobAction.VALIDATE}:
            return {
                "correctness": safe_correctness,
                "output": validation.output if mode == "public" else None,
                "compile_diagnostics": validator.diagnostics,
            }

        self.repository.transition_job(
            job.id,
            JobStatus.BENCHMARKING,
            phase="benchmarking",
            progress=0.62,
        )
        benchmark = self._run_benchmark(spool, problem, source_path)
        probe = self.runner.probe_environment(force=True)
        if not probe.healthy:
            raise RunnerFailure(probe.error or "GPU environment became unhealthy")
        environment = self.repository.save_environment(probe, force_new=True)
        source = source_path.read_text(encoding="utf-8")
        measurements, raw_samples = self._normalize_measurements(problem, benchmark)
        payload = job.payload_json
        if not payload.get("allow_duplicate") and self.repository.find_duplicate_versions(
            problem.manifest.slug, job.source_hash or source_hash(source)
        ):
            raise JobFailed(
                JobError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="相同源码已在排队期间保存，请确认重复后重试",
                    stage="benchmarking",
                )
            )
        version = self.repository.create_version_with_benchmark(
            problem_id=problem.manifest.slug,
            problem_revision=problem.manifest.revision,
            name=str(payload["version_name"]),
            notes=payload.get("notes"),
            source_code=source,
            source_hash=job.source_hash or source_hash(source),
            compile_flags=self.runner.effective_compile_flags(problem, probe),
            environment_id=environment.id,
            suite_hash=problem.suite_hash,
            protocol_version=problem.manifest.benchmark.protocol_version,
            input_sizes=[item.label for item in problem.manifest.benchmark.sizes],
            seed=problem.manifest.benchmark.suite_seed,
            warmup=problem.manifest.benchmark.warmup,
            iterations=problem.manifest.benchmark.iterations,
            measurements=measurements,
            raw_samples=raw_samples,
        )
        return {
            "version_id": version.id,
            "correctness": safe_correctness,
            "benchmark": {"measurements": measurements},
        }

    def _rebenchmark(self, job: JobRecord, spool: Path, problem: Problem) -> dict[str, Any]:
        version_ids = list(job.payload_json.get("version_ids", []))
        versions = self.repository.get_versions(version_ids)
        if len(versions) != len(version_ids):
            raise JobFailed(
                JobError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="一个或多个版本已不存在",
                    stage="validating",
                )
            )
        if any(version.problem_revision != problem.manifest.revision for version in versions):
            raise JobFailed(
                JobError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="旧题目 revision 无法用当前 harness 重新测试",
                    stage="validating",
                )
            )
        probe = self.runner.probe_environment(force=True)
        if not probe.healthy:
            raise RunnerFailure(probe.error or "GPU environment is unavailable")
        environment = self.repository.save_environment(probe, force_new=True)
        pending_rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        prepared: list[tuple[VersionRecord, Path, Path]] = []
        self.repository.transition_job(
            job.id,
            JobStatus.VALIDATING,
            phase="validating",
            progress=0.2,
        )
        for index, version in enumerate(versions):
            version_root = spool / f"version-{index + 1}"
            version_root.mkdir(mode=0o700)
            source_path = version_root / "source.cu"
            source_path.write_text(version.source_code, encoding="utf-8", newline="\n")
            source_path.chmod(0o600)
            validator = self.runner.compile(
                version_root, problem, source_path, harness_kind="validator"
            )
            self._require_compile(validator)
            self._assert_gpu_lease()
            validation = self.runner.execute(
                version_root,
                validator.executable,  # type: ignore[arg-type]
                mode="full",
                timeout_seconds=self._execution_timeout(problem, "full"),
            )
            self._require_execution(
                validation,
                "full",
                self._safe_correctness(problem, "full", validation.parsed),
            )
            prepared.append((version, version_root, source_path))
            self.repository.transition_job(
                job.id,
                JobStatus.VALIDATING,
                phase="validating",
                progress=0.2 + 0.25 * ((index + 1) / len(versions)),
            )

        self.repository.transition_job(
            job.id,
            JobStatus.BENCHMARKING,
            phase="benchmarking",
            progress=0.5,
        )
        for index, (version, version_root, source_path) in enumerate(prepared):
            benchmark = self._run_benchmark(version_root, problem, source_path)
            measurements, raw_samples = self._normalize_measurements(problem, benchmark)
            pending_rows.append(
                {
                    "version_id": version.id,
                    "environment_snapshot_id": environment.id,
                    "suite_hash": problem.suite_hash,
                    "protocol_version": problem.manifest.benchmark.protocol_version,
                    "compile_flags_json": self.runner.effective_compile_flags(problem, probe),
                    "input_sizes_json": [item.label for item in problem.manifest.benchmark.sizes],
                    "seed": problem.manifest.benchmark.suite_seed,
                    "warmup": problem.manifest.benchmark.warmup,
                    "iterations": problem.manifest.benchmark.iterations,
                    "measurements_json": measurements,
                    "raw_samples_json": raw_samples,
                }
            )
            summaries.append({"version_id": version.id, "measurements": measurements})
            self.repository.transition_job(
                job.id,
                JobStatus.BENCHMARKING,
                phase="benchmarking",
                progress=0.5 + 0.45 * ((index + 1) / len(versions)),
            )
        self.repository.add_benchmark_runs(pending_rows)
        return {"rebenchmarked": summaries, "environment_fingerprint": probe.fingerprint}

    def _run_benchmark(self, spool: Path, problem: Problem, source_path: Path) -> ExecutionResult:
        compiled = self.runner.compile(spool, problem, source_path, harness_kind="benchmark")
        self._require_compile(compiled)
        self._assert_gpu_lease()
        result = self.runner.execute(
            spool,
            compiled.executable,  # type: ignore[arg-type]
            mode="benchmark",
            timeout_seconds=min(
                self.settings.benchmark_timeout_seconds,
                problem.manifest.timeouts.benchmark_ms / 1000,
            ),
        )
        self._require_execution(result, "benchmark")
        return result

    @staticmethod
    def _verified_source(job: JobRecord, spool: Path) -> Path:
        path = spool / "source.cu"
        if not path.is_file():
            raise JobFailed(
                JobError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message="任务源码快照缺失",
                    stage="spool",
                    retryable=True,
                )
            )
        actual = source_hash(path.read_text(encoding="utf-8"))
        if not job.source_hash or actual != job.source_hash:
            raise JobFailed(
                JobError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message="任务源码快照完整性校验失败",
                    stage="spool",
                    retryable=False,
                )
            )
        return path

    @staticmethod
    def _require_compile(result: CompileResult) -> None:
        if result.succeeded:
            return
        if result.timed_out:
            code, message = ErrorCode.TIMEOUT, "NVCC 编译超时"
        elif result.output_limited:
            code, message = ErrorCode.OUTPUT_LIMIT, "NVCC 诊断输出超过限制"
        else:
            code, message = ErrorCode.COMPILE_ERROR, "NVCC 编译失败"
        raise JobFailed(JobError(code=code, message=message, stage="compiling"), result.diagnostics)

    @staticmethod
    def _require_execution(
        result: ExecutionResult,
        stage: str,
        safe_correctness: dict[str, Any] | None = None,
    ) -> None:
        if result.succeeded:
            return
        if result.timed_out:
            error = JobError(code=ErrorCode.TIMEOUT, message=f"{stage} 超时", stage=stage)
        elif result.output_limited:
            error = JobError(
                code=ErrorCode.OUTPUT_LIMIT,
                message=f"{stage} 输出超过限制",
                stage=stage,
            )
        elif result.parsed and result.parsed.get("status") == "wrong_answer":
            error = JobError(
                code=ErrorCode.WRONG_ANSWER,
                message="结果与参考实现不一致",
                stage=stage,
                details={"correctness": safe_correctness or {"status": "wrong_answer"}},
            )
        else:
            error = JobError(
                code=ErrorCode.RUNTIME_ERROR,
                message=f"{stage} 运行失败",
                stage=stage,
                details={
                    "returncode": result.returncode,
                    "result": safe_correctness or {"status": result.parsed.get("status")}
                    if result.parsed
                    else None,
                },
            )
        diagnostics = result.output if stage == "public" else None
        raise JobFailed(error, diagnostics)

    @staticmethod
    def _safe_correctness(
        problem: Problem,
        mode: str,
        parsed: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if parsed is None:
            return None
        if mode == "public":
            return parsed

        allowed_statuses = {"passed", "wrong_answer", "runtime_error"}
        status = str(parsed.get("status", "runtime_error"))
        if status not in allowed_statuses:
            status = "runtime_error"
        raw_cases = parsed.get("cases")
        safe_cases: list[dict[str, Any]] = []
        public_cases = problem.manifest.public.cases
        if isinstance(raw_cases, list):
            for index, raw_case in enumerate(raw_cases):
                if not isinstance(raw_case, dict):
                    continue
                is_public = index < len(public_cases)
                name = (
                    str(public_cases[index].get("name", f"sample_{index + 1}"))
                    if is_public
                    else f"internal_case_{index - len(public_cases) + 1}"
                )
                passed = raw_case.get("passed") is True
                safe_case: dict[str, Any] = {
                    "name": name,
                    "scope": "public" if is_public else "internal",
                    "passed": passed,
                }
                if not passed:
                    safe_case["message"] = "用例未通过"
                safe_cases.append(safe_case)

        if safe_cases:
            passed_count = sum(item["passed"] is True for item in safe_cases)
            summary = {
                "total": len(safe_cases),
                "passed": passed_count,
                "failed": len(safe_cases) - passed_count,
            }
        else:
            raw_summary = parsed.get("summary")
            source = raw_summary if isinstance(raw_summary, dict) else parsed
            total = source.get("total", 0)
            passed_count = source.get("passed", 0)
            total = total if isinstance(total, int) and total >= 0 else 0
            passed_count = (
                passed_count if isinstance(passed_count, int) and 0 <= passed_count <= total else 0
            )
            summary = {
                "total": total,
                "passed": passed_count,
                "failed": total - passed_count,
            }
        return {"status": status, "cases": safe_cases, "summary": summary}

    @staticmethod
    def _normalize_measurements(
        problem: Problem, result: ExecutionResult
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        parsed = result.parsed or {}
        protocol = str(parsed.get("protocol_version", ""))
        expected_protocol = problem.manifest.benchmark.protocol_version
        if protocol != expected_protocol:
            raise JobFailed(
                JobError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message="benchmark protocol version mismatch",
                    stage="benchmarking",
                )
            )
        raw = parsed.get("measurements")
        if not isinstance(raw, list) or len(raw) != len(problem.manifest.benchmark.sizes):
            raise JobFailed(
                JobError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message="benchmark returned an invalid measurement set",
                    stage="benchmarking",
                )
            )
        normalized: list[dict[str, Any]] = []
        samples: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                raise JobFailed(
                    JobError(
                        code=ErrorCode.INTERNAL_ERROR,
                        message="benchmark returned a malformed measurement",
                        stage="benchmarking",
                    )
                )
            label = str(item.get("label", ""))
            if label in seen:
                raise JobFailed(
                    JobError(
                        code=ErrorCode.INTERNAL_ERROR,
                        message="benchmark returned duplicate input sizes",
                        stage="benchmarking",
                    )
                )
            seen.add(label)
            measurement = Measurement(
                size=label,
                samples_ms=item.get("samples_ms", []),
                inner_repetitions=item.get("inner_repetitions", 1),
            ).with_statistics()
            if len(measurement.samples_ms) != problem.manifest.benchmark.iterations:
                raise JobFailed(
                    JobError(
                        code=ErrorCode.INTERNAL_ERROR,
                        message="benchmark sample count does not match the manifest",
                        stage="benchmarking",
                    )
                )
            normalized.append(measurement.model_dump(mode="json"))
            samples.append({"size": label, "samples_ms": measurement.samples_ms[:200]})
        expected_labels = [item.label for item in problem.manifest.benchmark.sizes]
        if [item["size"] for item in normalized] != expected_labels:
            raise JobFailed(
                JobError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message="benchmark input sizes do not match the manifest",
                    stage="benchmarking",
                )
            )
        return normalized, samples

    def _execution_timeout(self, problem: Problem, mode: str) -> float:
        manifest_timeout = (
            problem.manifest.timeouts.public_ms
            if mode == "public"
            else problem.manifest.timeouts.validation_ms
        )
        configured = (
            self.settings.run_timeout_seconds
            if mode == "public"
            else self.settings.validate_timeout_seconds
        )
        return min(configured, manifest_timeout / 1000)

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
        probe = self.runner.probe_environment(
            force=True,
            ignore_circuit_breaker=True,
        )
        self.repository.save_environment(probe)
        self._last_environment_probe = time.monotonic()
        return probe

    def _cleanup_job_ids(self, job_ids: list[str]) -> None:
        for job_id in job_ids:
            path = self.settings.jobs_dir / job_id
            try:
                self.runner.cleanup_task(path)
            except (OSError, ValueError):
                LOGGER.exception("failed to clean orphaned job %s", job_id)


def create_worker() -> Worker:
    settings = get_settings()
    settings.ensure_directories()
    engine = build_engine(settings)
    Base.metadata.create_all(engine)
    factory: sessionmaker[Session] = build_session_factory(engine)
    catalog = ProblemCatalog(settings.problems_dir).load()
    return Worker(settings, catalog, Repository(factory), DockerRunner(settings))


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    worker = create_worker()

    def stop_worker(_signum: int, _frame: Any) -> None:
        worker.stop()

    signal.signal(signal.SIGTERM, stop_worker)
    signal.signal(signal.SIGINT, stop_worker)
    LOGGER.info("worker %s started", worker.worker_id)
    worker.run_forever()


if __name__ == "__main__":
    main()
