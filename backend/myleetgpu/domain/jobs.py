from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

GPU_RESOURCE = "gpu:0"


class JobAction(StrEnum):
    COMPILE = "compile"
    RUN = "run"
    VALIDATE = "validate"
    SAVE_VERSION = "save_version"
    REBENCHMARK = "rebenchmark"

    @property
    def needs_gpu(self) -> bool:
        return self is not JobAction.COMPILE


class JobStatus(StrEnum):
    QUEUED = "queued"
    COMPILING = "compiling"
    RUNNING = "running"
    VALIDATING = "validating"
    BENCHMARKING = "benchmarking"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    SYSTEM_ERROR = "system_error"

    @property
    def terminal(self) -> bool:
        return self in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.TIMED_OUT,
            JobStatus.CANCELLED,
            JobStatus.SYSTEM_ERROR,
        }


ALLOWED_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.QUEUED: {JobStatus.COMPILING, JobStatus.CANCELLED, JobStatus.SYSTEM_ERROR},
    JobStatus.COMPILING: {
        JobStatus.RUNNING,
        JobStatus.VALIDATING,
        JobStatus.FAILED,
        JobStatus.TIMED_OUT,
        JobStatus.CANCELLED,
        JobStatus.SYSTEM_ERROR,
        JobStatus.SUCCEEDED,
    },
    JobStatus.RUNNING: {
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.TIMED_OUT,
        JobStatus.CANCELLED,
        JobStatus.SYSTEM_ERROR,
    },
    JobStatus.VALIDATING: {
        JobStatus.BENCHMARKING,
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.TIMED_OUT,
        JobStatus.CANCELLED,
        JobStatus.SYSTEM_ERROR,
    },
    JobStatus.BENCHMARKING: {
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.TIMED_OUT,
        JobStatus.CANCELLED,
        JobStatus.SYSTEM_ERROR,
    },
}


class ErrorCode(StrEnum):
    COMPILE_ERROR = "compile_error"
    WRONG_ANSWER = "wrong_answer"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    OUTPUT_LIMIT = "output_limit"
    RUNNER_UNHEALTHY = "runner_unhealthy"
    INVALID_REQUEST = "invalid_request"
    INTERNAL_ERROR = "internal_error"


class JobError(BaseModel):
    code: ErrorCode
    message: str
    stage: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


def assert_transition(current: JobStatus, target: JobStatus) -> None:
    if current == target:
        return
    allowed = ALLOWED_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise ValueError(f"invalid job transition: {current} -> {target}")
