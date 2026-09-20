from __future__ import annotations

import copy
from pathlib import Path

import pytest
from myleetgpu.application.results import (
    JobFailed,
    normalize_measurements,
    require_compile,
    require_execution,
    safe_correctness,
)
from myleetgpu.domain.jobs import ErrorCode
from myleetgpu.domain.problems import ProblemCatalog
from myleetgpu.runner.models import CompileResult, ExecutionResult


@pytest.fixture
def problem():
    return (
        ProblemCatalog(Path(__file__).resolve().parents[2] / "problems")
        .load()
        .get("vector-addition")
    )


@pytest.mark.parametrize("language", ["cuda_cpp", "triton_python", "torch_python"])
@pytest.mark.parametrize(
    ("timed_out", "output_limited", "code"),
    [
        (False, False, ErrorCode.COMPILE_ERROR),
        (True, False, ErrorCode.TIMEOUT),
        (False, True, ErrorCode.OUTPUT_LIMIT),
    ],
)
def test_compile_failure_policy_is_independent_of_worker(language, timed_out, output_limited, code):
    result = CompileResult(False, "compiler diagnostics", None, 1.0, timed_out, output_limited)
    with pytest.raises(JobFailed) as caught:
        require_compile(result, language)
    assert caught.value.error.code is code
    assert caught.value.error.stage == "compiling"
    assert caught.value.diagnostics == "compiler diagnostics"


def test_full_correctness_never_exposes_internal_data_or_mutates_input(problem):
    parsed = {
        "status": "wrong_answer",
        "cases": [
            {"name": "private name", "passed": False, "message": "secret input"}
            for _ in range(len(problem.manifest.public.cases) + 1)
        ],
        "private": "secret reference",
    }
    original = copy.deepcopy(parsed)
    result = safe_correctness(problem, "full", parsed)
    assert result is not None
    assert result["cases"][-1]["name"] == "internal_case_1"
    assert "secret" not in str(result)
    assert "private name" not in str(result)
    assert parsed == original
    assert safe_correctness(problem, "public", parsed) is parsed


@pytest.mark.parametrize("stage", ["full", "benchmark"])
def test_private_execution_failure_drops_raw_diagnostics(stage):
    result = ExecutionResult(False, "hidden input and reference", None, 1.0, 1)
    with pytest.raises(JobFailed) as caught:
        require_execution(result, stage)
    assert caught.value.diagnostics is None
    assert "hidden" not in str(caught.value.error.model_dump())


def benchmark_payload(problem):
    return {
        "status": "passed",
        "protocol_version": problem.manifest.benchmark.protocol_version,
        "measurements": [
            {
                "label": size.label,
                "samples_ms": [2.0] * problem.manifest.benchmark.iterations,
                "inner_repetitions": size.inner_repetitions,
                "median_ms": -999,
            }
            for size in problem.manifest.benchmark.sizes
        ],
    }


def test_benchmark_statistics_are_computed_from_samples(problem):
    payload = benchmark_payload(problem)
    measurements, raw = normalize_measurements(problem, ExecutionResult(True, "", payload, 0.1, 0))
    assert len(measurements) == len(problem.manifest.benchmark.sizes)
    assert all(item["median_ms"] == 2.0 for item in measurements)
    assert all(item["samples_ms"] == [2.0] * problem.manifest.benchmark.iterations for item in raw)
    assert payload["measurements"][0]["median_ms"] == -999


@pytest.mark.parametrize("mutation", ["protocol", "size_count", "duplicate", "samples", "order"])
def test_invalid_benchmark_contract_is_rejected_before_persistence(problem, mutation):
    payload = benchmark_payload(problem)
    if mutation == "protocol":
        payload["protocol_version"] = "unexpected"
    elif mutation == "size_count":
        payload["measurements"].pop()
    elif mutation == "duplicate":
        payload["measurements"][1]["label"] = payload["measurements"][0]["label"]
    elif mutation == "samples":
        payload["measurements"][0]["samples_ms"].pop()
    else:
        payload["measurements"].reverse()
    with pytest.raises(JobFailed) as caught:
        normalize_measurements(problem, ExecutionResult(True, "", payload, 0.1, 0))
    assert caught.value.error.stage == "benchmarking"
