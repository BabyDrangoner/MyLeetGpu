"""Execute one claimed job; the host owns scheduling, leases, and final cleanup.

A job is never executed inside a database transaction. Versions and rebenchmark
batches are persisted only after all required validation and measurements pass.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from myleetgpu.application.results import (
    JobFailed,
    normalize_measurements,
    require_compile,
    require_execution,
    safe_correctness,
)
from myleetgpu.config import Settings
from myleetgpu.domain.benchmark import source_hash
from myleetgpu.domain.jobs import ErrorCode, JobAction, JobError, JobStatus
from myleetgpu.domain.problems import KernelLanguage, Problem, ProblemCatalog, ProblemImplementation
from myleetgpu.filesystem import ensure_mode
from myleetgpu.infrastructure.models import JobRecord, VersionRecord
from myleetgpu.infrastructure.repository import Repository
from myleetgpu.runner.models import EnvironmentProbe, ExecutionResult, RunnerFailure
from myleetgpu.runner.protocols import CpuEnvironmentRunner, Runner


class JobExecutor:
    def __init__(
        self,
        settings: Settings,
        catalog: ProblemCatalog,
        repository: Repository,
        runner: Runner,
        *,
        check_lease: Callable[[], None],
    ) -> None:
        self.settings = settings
        self.catalog = catalog
        self.repository = repository
        self.runner = runner
        self.check_lease = check_lease

    def execute(self, job: JobRecord, spool: Path) -> dict[str, Any]:
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
        try:
            implementation = problem.get_implementation(job.language)
        except KeyError as error:
            raise JobFailed(
                JobError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="题目不再支持任务指定的实现语言",
                    stage="queued",
                )
            ) from error
        if action is JobAction.REBENCHMARK:
            return self._rebenchmark(job, spool, problem, implementation)

        source_path = self._verified_source(job, spool, implementation)
        if (
            action is JobAction.SAVE_VERSION
            and not job.payload_json.get("allow_duplicate")
            and self.repository.find_duplicate_versions(
                problem.manifest.slug, job.source_hash or "", job.language
            )
        ):
            raise JobFailed(
                JobError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="相同源码已存在，请确认重复后重试",
                    stage="queued",
                )
            )
        validator = self.runner.compile(
            spool,
            problem,
            source_path,
            harness_kind="validator",
            language=job.language,
            implementation=implementation,
        )
        require_compile(validator, job.language)
        if action is JobAction.COMPILE:
            return {
                "compiled": True,
                "diagnostics": validator.diagnostics,
                "duration_seconds": validator.duration_seconds,
                "language": job.language,
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
        self.check_lease()
        self._require_runtime_environment(job.language)
        validation = self.runner.execute(
            spool,
            validator.executable,
            mode=mode,
            timeout_seconds=timeout,  # type: ignore[arg-type]
            language=job.language,
        )
        correctness = safe_correctness(problem, mode, validation.parsed)
        require_execution(validation, mode, correctness)
        if action in {JobAction.RUN, JobAction.VALIDATE}:
            return {
                "correctness": correctness,
                "output": validation.output if mode == "public" else None,
                "compile_diagnostics": validator.diagnostics,
                "language": job.language,
            }

        self.repository.transition_job(
            job.id,
            JobStatus.BENCHMARKING,
            phase="benchmarking",
            progress=0.62,
        )
        benchmark = self._run_benchmark(spool, problem, implementation, source_path, job.language)
        probe = self._probe_for_language(job.language, force=True)
        if not probe.healthy:
            raise RunnerFailure(probe.error or "Execution environment became unhealthy")
        environment = self.repository.save_environment(probe, force_new=True)
        source = source_path.read_text(encoding="utf-8")
        measurements, raw_samples = normalize_measurements(problem, benchmark)
        payload = job.payload_json
        if not payload.get("allow_duplicate") and self.repository.find_duplicate_versions(
            problem.manifest.slug,
            job.source_hash or source_hash(source),
            job.language,
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
            language=job.language,
            name=str(payload["version_name"]),
            notes=payload.get("notes"),
            source_code=source,
            source_hash=job.source_hash or source_hash(source),
            compile_flags=self.runner.effective_compile_flags(
                problem,
                probe,
                language=job.language,
                implementation=implementation,
            ),
            environment_id=environment.id,
            suite_hash=implementation.suite_hash,
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
            "correctness": correctness,
            "benchmark": {"measurements": measurements},
            "language": job.language,
        }

    def _rebenchmark(
        self,
        job: JobRecord,
        spool: Path,
        problem: Problem,
        implementation: ProblemImplementation,
    ) -> dict[str, Any]:
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
        if any(version.language != job.language for version in versions):
            raise JobFailed(
                JobError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="统一重测只能包含同一种实现语言",
                    stage="validating",
                )
            )
        probe = self._probe_for_language(job.language, force=True)
        if not probe.healthy:
            raise RunnerFailure(probe.error or "Execution environment is unavailable")
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
            source_path = version_root / f"source{implementation.source_suffix}"
            source_path.write_text(version.source_code, encoding="utf-8", newline="\n")
            ensure_mode(source_path, 0o600)
            validator = self.runner.compile(
                version_root,
                problem,
                source_path,
                harness_kind="validator",
                language=job.language,
                implementation=implementation,
            )
            require_compile(validator, job.language)
            self.check_lease()
            validation = self.runner.execute(
                version_root,
                validator.executable,  # type: ignore[arg-type]
                mode="full",
                timeout_seconds=self._execution_timeout(problem, "full"),
                language=job.language,
            )
            require_execution(
                validation,
                "full",
                safe_correctness(problem, "full", validation.parsed),
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
            benchmark = self._run_benchmark(
                version_root,
                problem,
                implementation,
                source_path,
                job.language,
            )
            measurements, raw_samples = normalize_measurements(problem, benchmark)
            pending_rows.append(
                {
                    "version_id": version.id,
                    "environment_snapshot_id": environment.id,
                    "suite_hash": implementation.suite_hash,
                    "protocol_version": problem.manifest.benchmark.protocol_version,
                    "compile_flags_json": self.runner.effective_compile_flags(
                        problem,
                        probe,
                        language=job.language,
                        implementation=implementation,
                    ),
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
        return {
            "rebenchmarked": summaries,
            "environment_fingerprint": probe.fingerprint,
            "language": job.language,
        }

    def _run_benchmark(
        self,
        spool: Path,
        problem: Problem,
        implementation: ProblemImplementation,
        source_path: Path,
        language: str,
    ) -> ExecutionResult:
        compiled = self.runner.compile(
            spool,
            problem,
            source_path,
            harness_kind="benchmark",
            language=language,
            implementation=implementation,
        )
        require_compile(compiled, language)
        self.check_lease()
        result = self.runner.execute(
            spool,
            compiled.executable,  # type: ignore[arg-type]
            mode="benchmark",
            timeout_seconds=min(
                self.settings.benchmark_timeout_seconds,
                problem.manifest.timeouts.benchmark_ms / 1000,
            ),
            language=language,
        )
        require_execution(result, "benchmark")
        return result

    @staticmethod
    def _verified_source(
        job: JobRecord, spool: Path, implementation: ProblemImplementation
    ) -> Path:
        path = spool / f"source{implementation.source_suffix}"
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

    def _probe_for_language(self, language: str, *, force: bool) -> EnvironmentProbe:
        if KernelLanguage(language).is_cpu:
            if not isinstance(self.runner, CpuEnvironmentRunner):
                raise RunnerFailure("The configured runner does not support CPU execution")
            return self.runner.probe_cpu_environment(language, force=force)
        if language == KernelLanguage.TRITON_PYTHON.value:
            return self.runner.probe_triton_environment(force=force)
        if language == KernelLanguage.TORCH_PYTHON.value:
            return self.runner.probe_torch_environment(force=force)
        return self.runner.probe_environment(force=force)

    def _require_runtime_environment(self, language: str) -> EnvironmentProbe:
        probe = self._probe_for_language(language, force=False)
        # Make a language-specific status observation visible even for ordinary
        # run/validate jobs. Repository deduplication keeps this mutable status
        # row separate from immutable benchmark snapshots.
        self.repository.save_environment(probe)
        if not probe.healthy:
            raise RunnerFailure(probe.error or f"{language} runtime is unavailable")
        return probe
