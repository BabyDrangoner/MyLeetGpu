from __future__ import annotations

import hashlib
import math

import pytest
from myleetgpu.domain.benchmark import (
    ComparabilityKey,
    Measurement,
    source_hash,
    stable_hash,
    suite_hash,
)
from pydantic import ValidationError


def comparability_key(**overrides: object) -> ComparabilityKey:
    values: dict[str, object] = {
        "problem_revision": "7",
        "suite_hash": "suite-a",
        "input_sizes": ["64K", "1M"],
        "compile_flags": ["--std=c++17", "-O3", "-arch=sm_89"],
        "environment_fingerprint": "env-a",
    }
    values.update(overrides)
    return ComparabilityKey(**values)


def test_measurement_computes_deterministic_distribution_statistics() -> None:
    measurement = Measurement(size="1M", samples_ms=[10.0, 2.0, 4.0, 8.0, 6.0])

    result = measurement.with_statistics()

    assert result.samples_ms == [10.0, 2.0, 4.0, 8.0, 6.0]
    assert result.median_ms == 6.0
    assert result.p95_ms == 10.0
    assert result.min_ms == 2.0
    assert result.cv == pytest.approx(math.sqrt(8.0) / 6.0)
    assert result.mad_ms == 2.0


def test_measurement_uses_nearest_rank_p95() -> None:
    result = Measurement(size="tiny", samples_ms=list(range(1, 21))).with_statistics()

    assert result.p95_ms == 19


def test_single_zero_sample_has_zero_variation() -> None:
    result = Measurement(size="empty-work", samples_ms=[0.0]).with_statistics()

    assert result.median_ms == 0.0
    assert result.p95_ms == 0.0
    assert result.min_ms == 0.0
    assert result.cv == 0.0
    assert result.mad_ms == 0.0


@pytest.mark.parametrize("sample", [-0.001, float("nan"), float("inf"), -float("inf")])
def test_measurement_rejects_invalid_samples(sample: float) -> None:
    with pytest.raises(ValidationError, match="finite and non-negative"):
        Measurement(size="bad", samples_ms=[sample])


def test_measurement_requires_samples_and_bounded_raw_sample_count() -> None:
    with pytest.raises(ValidationError):
        Measurement(size="none", samples_ms=[])
    with pytest.raises(ValidationError):
        Measurement(size="too-many", samples_ms=[1.0] * 501)


def test_source_hash_normalizes_all_text_line_endings() -> None:
    expected = hashlib.sha256(b"first\nsecond\n").hexdigest()

    assert source_hash("first\nsecond\n") == expected
    assert source_hash("first\r\nsecond\r\n") == expected
    assert source_hash("first\rsecond\r") == expected


def test_source_hash_preserves_meaningful_source_bytes() -> None:
    assert source_hash("x\n") != source_hash("x")
    assert source_hash("x") != source_hash("x ")


def test_stable_hash_is_key_order_independent_and_unicode_safe() -> None:
    left = {"题目": "归约", "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "题目": "归约"}

    assert stable_hash(left) == stable_hash(right)
    assert len(stable_hash(left)) == 64


def test_suite_hash_is_file_order_independent() -> None:
    files = [("validator.cu", b"validator"), ("benchmark.cu", b"benchmark")]
    manifest = {"sizes": [{"n": 1024}], "seed": 42}

    assert suite_hash(files, manifest) == suite_hash(reversed(files), manifest)


@pytest.mark.parametrize(
    ("files", "manifest"),
    [
        ([("validator.cu", b"changed"), ("benchmark.cu", b"benchmark")], {"seed": 42}),
        ([("renamed.cu", b"validator"), ("benchmark.cu", b"benchmark")], {"seed": 42}),
        ([("validator.cu", b"validator"), ("benchmark.cu", b"benchmark")], {"seed": 43}),
    ],
)
def test_suite_hash_changes_for_protocol_inputs(
    files: list[tuple[str, bytes]], manifest: object
) -> None:
    baseline = suite_hash(
        [("validator.cu", b"validator"), ("benchmark.cu", b"benchmark")],
        {"seed": 42},
    )

    assert suite_hash(files, manifest) != baseline


def test_identical_comparability_keys_have_no_differences() -> None:
    assert comparability_key().differences(comparability_key()) == []


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"problem_revision": "8"}, "题目版本不同"),
        ({"suite_hash": "suite-b"}, "benchmark suite 不同"),
        ({"input_sizes": ["1M", "64K"]}, "输入规模不同"),
        ({"compile_flags": ["-O2"]}, "编译配置不同"),
        ({"environment_fingerprint": "env-b"}, "GPU/CUDA 环境指纹不同"),
    ],
)
def test_comparability_reports_each_mismatch(override: dict[str, object], message: str) -> None:
    assert comparability_key().differences(comparability_key(**override)) == [message]


def test_comparability_reports_all_mismatches_without_claiming_a_ranking() -> None:
    other = comparability_key(
        problem_revision="8",
        suite_hash="suite-b",
        input_sizes=["16M"],
        compile_flags=["-O2"],
        environment_fingerprint="env-b",
    )

    assert len(comparability_key().differences(other)) == 5
