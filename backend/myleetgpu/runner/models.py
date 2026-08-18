from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

type RunnerLanguage = Literal["cuda_cpp", "triton_python"]


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    output: str
    duration_seconds: float
    timed_out: bool = False
    output_limited: bool = False


@dataclass(frozen=True)
class CompileResult:
    succeeded: bool
    diagnostics: str
    executable: Path | None
    duration_seconds: float
    timed_out: bool = False
    output_limited: bool = False


@dataclass(frozen=True)
class ExecutionResult:
    succeeded: bool
    output: str
    parsed: dict[str, Any] | None
    duration_seconds: float
    returncode: int
    timed_out: bool = False
    output_limited: bool = False


@dataclass(frozen=True)
class EnvironmentProbe:
    healthy: bool
    gpu_name: str | None
    compute_capability: str | None
    driver_version: str | None
    cuda_runtime_version: str | None
    nvcc_version: str | None
    cuda_image: str
    image_digest: str | None
    cuda_arch: str | None
    telemetry: dict[str, str | None] = field(default_factory=dict)
    error: str | None = None
    fingerprint: str = ""
    backend: str = "cuda_cpp"
    toolchain: dict[str, Any] = field(default_factory=dict)


class RunnerFailure(RuntimeError):
    pass


class RunnerUnavailable(RunnerFailure):
    pass


class RunnerUnhealthy(RunnerFailure):
    pass
