from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from myleetgpu.config import Settings
from myleetgpu.domain.benchmark import stable_hash
from myleetgpu.domain.problems import Problem
from myleetgpu.runner.docker import DockerRunner
from myleetgpu.runner.models import (
    CompileResult,
    EnvironmentProbe,
    ExecutionResult,
    RunnerLanguage,
    RunnerUnavailable,
)
from myleetgpu.runner.protocols import CpuEnvironmentRunner, LanguageRecovery, Runner

LOGGER = logging.getLogger("myleetgpu.runner.router")


class ExecutionRouter:
    """One worker, one active operation; a job's target is fixed before compilation."""

    def __init__(self, settings: Settings, *, runners: dict[str, Runner] | None = None):
        if runners is None:
            from myleetgpu.runner.colab import ColabRunner
            from myleetgpu.runner.cpu import CpuRunner

            runners = {
                "local": DockerRunner(settings),
                "colab": ColabRunner(settings),
                "cpu": CpuRunner(settings),
            }
        self.runners = runners
        self.target = "local"
        self._prepared_targets: set[str] = set()

    def select_target(self, target: str) -> None:
        if target not in {"local", "colab", "cpu"} or target not in self.runners:
            raise RunnerUnavailable("未知执行位置；任务不会自动回退至其他设备")
        self.target = target

    def assign_owner(self, owner: str) -> None:
        for runner in self.runners.values():
            runner.assign_owner(owner)

    def cleanup_orphan_containers(self) -> list[str]:
        # Cleanup is a provider operation, not a discovery sweep: merely using CPU
        # must never contact Docker or an expired Colab session.
        return self._active.cleanup_orphan_containers()

    def prepare_target(self) -> None:
        if self.target in self._prepared_targets:
            return
        removed = self.cleanup_orphan_containers()
        if removed:
            LOGGER.warning("removed %d orphaned tasks from %s", len(removed), self.target)
        self._prepared_targets.add(self.target)

    def invalidate_preparation(self) -> None:
        # A provider that was offline during cleanup must be cleaned again after
        # reconnecting. This also covers adapters with best-effort cleanup.
        self._prepared_targets.discard(self.target)

    @property
    def _active(self) -> Runner:
        return self.runners[self.target]

    def assert_healthy(self) -> None:
        self._active.assert_healthy()

    def mark_unhealthy(self, reason: str) -> None:
        self._active.mark_unhealthy(reason)

    def compile(
        self,
        task_root: Path,
        problem: Problem,
        source_path: Path,
        *,
        harness_kind: str = "validator",
        language: RunnerLanguage | str | None = None,
        implementation: Any | None = None,
    ) -> CompileResult:
        return self._active.compile(
            task_root,
            problem,
            source_path,
            harness_kind=harness_kind,
            language=language,
            implementation=implementation,
        )

    def effective_compile_flags(
        self,
        problem: Problem,
        probe: EnvironmentProbe,
        *,
        language: RunnerLanguage | str | None = None,
        implementation: Any | None = None,
    ) -> list[str]:
        return self._active.effective_compile_flags(
            problem, probe, language=language, implementation=implementation
        )

    def execute(
        self,
        task_root: Path,
        executable: Path,
        *,
        mode: str,
        timeout_seconds: float,
        language: RunnerLanguage | str | None = None,
    ) -> ExecutionResult:
        return self._active.execute(
            task_root,
            executable,
            mode=mode,
            timeout_seconds=timeout_seconds,
            language=language,
        )

    def cleanup_task(self, task_root: Path) -> None:
        self._active.cleanup_task(task_root)

    def cleanup_owned_containers(self) -> list[str]:
        return self._active.cleanup_owned_containers()

    def _tag(self, probe: EnvironmentProbe) -> EnvironmentProbe:
        # Legacy local benchmark fingerprints remain comparable. A remote observation
        # can never match a local one even if hardware/toolchain happen to be identical.
        return replace(
            probe,
            toolchain={**probe.toolchain, "execution_target": self.target},
            fingerprint=(
                stable_hash({"target": "colab", "environment": probe.fingerprint})
                if self.target == "colab"
                else probe.fingerprint
            ),
        )

    def probe_environment(
        self, *, force: bool = False, ignore_circuit_breaker: bool = False
    ) -> EnvironmentProbe:
        if ignore_circuit_breaker:
            probe = self._active.probe_environment(force=force, ignore_circuit_breaker=True)
        else:
            probe = self._active.probe_environment(force=force)
        return self._tag(probe)

    def probe_triton_environment(
        self, *, force: bool = False, ignore_circuit_breaker: bool = False
    ) -> EnvironmentProbe:
        if ignore_circuit_breaker:
            probe = self._active.probe_triton_environment(force=force, ignore_circuit_breaker=True)
        else:
            probe = self._active.probe_triton_environment(force=force)
        return self._tag(probe)

    def probe_torch_environment(
        self, *, force: bool = False, ignore_circuit_breaker: bool = False
    ) -> EnvironmentProbe:
        if ignore_circuit_breaker:
            probe = self._active.probe_torch_environment(force=force, ignore_circuit_breaker=True)
        else:
            probe = self._active.probe_torch_environment(force=force)
        return self._tag(probe)

    def probe_cpu_environment(
        self,
        language: RunnerLanguage | str = "cpp",
        *,
        force: bool = False,
        ignore_circuit_breaker: bool = False,
    ) -> EnvironmentProbe:
        if self.target != "cpu" or not isinstance(self._active, CpuEnvironmentRunner):
            raise RunnerUnavailable("CPU runtime must use the CPU execution target")
        return self._tag(
            self._active.probe_cpu_environment(
                language, force=force, ignore_circuit_breaker=ignore_circuit_breaker
            )
        )

    def probe_connection(self, language: str) -> EnvironmentProbe:
        runner = self._active
        if self.target in {"colab", "cpu"} and hasattr(runner, "recover"):
            # A manual, worker-serialized smoke test is the remote recovery path.
            return self._tag(cast(LanguageRecovery, runner).recover(language=language))
        probe = {
            "cuda_cpp": self.probe_environment,
            "triton_python": self.probe_triton_environment,
            "torch_python": self.probe_torch_environment,
        }[language]
        return probe(force=True)

    def recover(self) -> EnvironmentProbe:
        return self._tag(self._active.recover())
