"""Application-facing execution ports, independent of Docker and SSH adapters.

Protocols are structural: deterministic test runners and future adapters need
not inherit a framework base class. Target selection is a separate capability
so single-target runners remain useful to the worker and the judging service.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from myleetgpu.domain.problems import Problem
from myleetgpu.runner.models import CompileResult, EnvironmentProbe, ExecutionResult, RunnerLanguage


@runtime_checkable
class Runner(Protocol):
    """Operations required to judge jobs and manage their execution lifecycle."""

    def assign_owner(self, owner: str) -> None: ...

    def assert_healthy(self) -> None: ...

    def mark_unhealthy(self, reason: str) -> None: ...

    def probe_environment(
        self, *, force: bool = False, ignore_circuit_breaker: bool = False
    ) -> EnvironmentProbe: ...

    def probe_triton_environment(
        self, *, force: bool = False, ignore_circuit_breaker: bool = False
    ) -> EnvironmentProbe: ...

    def probe_torch_environment(
        self, *, force: bool = False, ignore_circuit_breaker: bool = False
    ) -> EnvironmentProbe: ...

    def recover(self) -> EnvironmentProbe: ...

    def compile(
        self,
        task_root: Path,
        problem: Problem,
        source_path: Path,
        *,
        harness_kind: str = "validator",
        language: RunnerLanguage | str | None = None,
        implementation: Any | None = None,
    ) -> CompileResult: ...

    def effective_compile_flags(
        self,
        problem: Problem,
        probe: EnvironmentProbe,
        *,
        language: RunnerLanguage | str | None = None,
        implementation: Any | None = None,
    ) -> list[str]: ...

    def execute(
        self,
        task_root: Path,
        executable: Path,
        *,
        mode: str,
        timeout_seconds: float,
        language: RunnerLanguage | str | None = None,
    ) -> ExecutionResult: ...

    def cleanup_task(self, task_root: Path) -> None: ...

    def cleanup_orphan_containers(self) -> list[str]: ...

    def cleanup_owned_containers(self) -> list[str]: ...


@runtime_checkable
class TargetRunner(Protocol):
    """Optional multi-target selection and serialized connection-test capability."""

    def select_target(self, target: str) -> None: ...

    def probe_connection(self, language: str) -> EnvironmentProbe: ...


class LanguageRecovery(Protocol):
    """Optional recovery for a specific language toolchain, used by Colab."""

    def recover(self, *, language: RunnerLanguage | str) -> EnvironmentProbe: ...


@runtime_checkable
class PreparedRunner(Protocol):
    """Lazy cleanup of only the provider explicitly requested by a queued operation."""

    def prepare_target(self) -> None: ...

    def invalidate_preparation(self) -> None: ...


@runtime_checkable
class CpuEnvironmentRunner(Protocol):
    """Optional CPU toolchain probes, independent of GPU runtime availability."""

    def probe_cpu_environment(
        self,
        language: RunnerLanguage | str = "cpp",
        *,
        force: bool = False,
        ignore_circuit_breaker: bool = False,
    ) -> EnvironmentProbe: ...
