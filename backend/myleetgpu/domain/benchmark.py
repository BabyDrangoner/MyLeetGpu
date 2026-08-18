from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field, field_validator

PROTOCOL_VERSION = "1"


class Measurement(BaseModel):
    size: str
    samples_ms: list[float] = Field(min_length=1, max_length=500)
    median_ms: float | None = None
    p95_ms: float | None = None
    min_ms: float | None = None
    cv: float | None = None
    mad_ms: float | None = None
    inner_repetitions: int = Field(default=1, ge=1)

    @field_validator("samples_ms")
    @classmethod
    def finite_nonnegative_samples(cls, values: list[float]) -> list[float]:
        if any(not math.isfinite(item) or item < 0 for item in values):
            raise ValueError("benchmark samples must be finite and non-negative")
        return values

    def with_statistics(self) -> Measurement:
        ordered = sorted(self.samples_ms)
        median = statistics.median(ordered)
        p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
        mean = statistics.fmean(ordered)
        stddev = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
        return self.model_copy(
            update={
                "median_ms": median,
                "p95_ms": ordered[p95_index],
                "min_ms": ordered[0],
                "cv": stddev / mean if mean else 0.0,
                "mad_ms": statistics.median(abs(sample - median) for sample in ordered),
            }
        )


class ComparabilityKey(BaseModel):
    language: str = "cuda_cpp"
    problem_revision: str
    suite_hash: str
    input_sizes: list[str]
    compile_flags: list[str]
    environment_fingerprint: str

    def differences(self, other: ComparabilityKey) -> list[str]:
        labels = {
            "language": "实现语言不同",
            "problem_revision": "题目版本不同",
            "suite_hash": "benchmark suite 不同",
            "input_sizes": "输入规模不同",
            "compile_flags": "编译配置不同",
            "environment_fingerprint": "GPU/CUDA 环境指纹不同",
        }
        left = self.model_dump()
        right = other.model_dump()
        return [message for field, message in labels.items() if left[field] != right[field]]


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def source_hash(source: str) -> str:
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def suite_hash(files: Iterable[tuple[str, bytes]], manifest_benchmark: Any) -> str:
    digest = hashlib.sha256()
    for name, content in sorted(files):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    digest.update(stable_hash(manifest_benchmark).encode("ascii"))
    return digest.hexdigest()
