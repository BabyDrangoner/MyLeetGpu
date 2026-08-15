from __future__ import annotations

import pytest
from myleetgpu.domain.jobs import (
    ALLOWED_TRANSITIONS,
    JobAction,
    JobStatus,
    assert_transition,
)

TERMINAL_STATUSES = {
    JobStatus.SUCCEEDED,
    JobStatus.FAILED,
    JobStatus.TIMED_OUT,
    JobStatus.CANCELLED,
    JobStatus.SYSTEM_ERROR,
}


def test_terminal_property_matches_the_state_machine_contract() -> None:
    assert {status for status in JobStatus if status.terminal} == TERMINAL_STATUSES


def test_terminal_states_have_no_outgoing_transitions() -> None:
    assert all(not ALLOWED_TRANSITIONS.get(status, set()) for status in TERMINAL_STATUSES)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (JobStatus.QUEUED, JobStatus.COMPILING),
        (JobStatus.COMPILING, JobStatus.SUCCEEDED),
        (JobStatus.COMPILING, JobStatus.RUNNING),
        (JobStatus.COMPILING, JobStatus.VALIDATING),
        (JobStatus.RUNNING, JobStatus.SUCCEEDED),
        (JobStatus.VALIDATING, JobStatus.SUCCEEDED),
        (JobStatus.VALIDATING, JobStatus.BENCHMARKING),
        (JobStatus.BENCHMARKING, JobStatus.SUCCEEDED),
    ],
)
def test_expected_happy_path_transitions_are_allowed(current: JobStatus, target: JobStatus) -> None:
    assert_transition(current, target)


@pytest.mark.parametrize("target", list(TERMINAL_STATUSES - {JobStatus.SUCCEEDED}))
@pytest.mark.parametrize(
    "active",
    [JobStatus.COMPILING, JobStatus.RUNNING, JobStatus.VALIDATING, JobStatus.BENCHMARKING],
)
def test_active_stage_can_end_with_any_failure_status(active: JobStatus, target: JobStatus) -> None:
    assert_transition(active, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (JobStatus.QUEUED, JobStatus.RUNNING),
        (JobStatus.QUEUED, JobStatus.SUCCEEDED),
        (JobStatus.RUNNING, JobStatus.COMPILING),
        (JobStatus.VALIDATING, JobStatus.RUNNING),
        (JobStatus.BENCHMARKING, JobStatus.VALIDATING),
        (JobStatus.SUCCEEDED, JobStatus.QUEUED),
        (JobStatus.FAILED, JobStatus.COMPILING),
        (JobStatus.TIMED_OUT, JobStatus.CANCELLED),
    ],
)
def test_invalid_transition_is_rejected(current: JobStatus, target: JobStatus) -> None:
    with pytest.raises(ValueError, match="invalid job transition"):
        assert_transition(current, target)


@pytest.mark.parametrize("status", list(JobStatus))
def test_idempotent_status_updates_are_allowed(status: JobStatus) -> None:
    assert_transition(status, status)


def test_only_compile_action_does_not_need_a_gpu() -> None:
    assert not JobAction.COMPILE.needs_gpu
    assert all(action.needs_gpu for action in JobAction if action is not JobAction.COMPILE)
