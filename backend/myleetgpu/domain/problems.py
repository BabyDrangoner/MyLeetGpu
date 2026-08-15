from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from myleetgpu.domain.benchmark import suite_hash


class SignatureManifest(BaseModel):
    header: str
    symbol: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    declaration: str


class ToleranceManifest(BaseModel):
    mode: Literal["integer", "float"]
    atol: float = Field(default=0.0, ge=0)
    rtol: float = Field(default=0.0, ge=0)
    nan_equal: bool = False
    infinity: str = "same_sign"


class HarnessManifest(BaseModel):
    validator: str
    benchmark: str


class PublicTestsManifest(BaseModel):
    seed: int
    timeout_ms: int = Field(gt=0, le=300_000)
    cases: list[dict[str, Any]] = Field(min_length=1)


class InternalTestsManifest(BaseModel):
    seeds: list[int] = Field(min_length=1)
    timeout_ms: int = Field(gt=0, le=300_000)
    cases: list[dict[str, Any]] = Field(min_length=1)


class BenchmarkSize(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str = Field(min_length=1, max_length=64)
    inner_repetitions: int = Field(default=1, ge=1, le=100_000)


class BenchmarkManifest(BaseModel):
    protocol_version: str
    suite_seed: int
    sizes: list[BenchmarkSize] = Field(min_length=1, max_length=32)
    warmup: int = Field(ge=1, le=10_000)
    iterations: int = Field(ge=3, le=500)
    timeout_ms: int = Field(gt=0, le=1_800_000)


class TimeoutsManifest(BaseModel):
    compile_ms: int = Field(gt=0, le=600_000)
    public_ms: int = Field(gt=0, le=300_000)
    validation_ms: int = Field(gt=0, le=900_000)
    benchmark_ms: int = Field(gt=0, le=1_800_000)


class CompilerManifest(BaseModel):
    executable: Literal["nvcc"]
    standard: Literal["c++17", "c++20"]
    optimization: Literal["O2", "O3"]
    architecture: Literal["detected"]
    allowed_flags: list[str] = Field(min_length=1, max_length=16)
    include_paths: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("allowed_flags")
    @classmethod
    def enforce_flag_allowlist(cls, values: list[str]) -> list[str]:
        allowlist = {"--std=c++17", "--std=c++20", "-O2", "-O3", "--use_fast_math"}
        unknown = set(values) - allowlist
        if unknown:
            raise ValueError(f"compiler flags are not allowed: {sorted(unknown)}")
        return values


class ProblemManifest(BaseModel):
    schema_version: Literal[1]
    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=128)
    title: str = Field(min_length=1, max_length=160)
    difficulty: Literal["easy", "medium", "hard"]
    revision: str
    language: Literal["cuda_cpp"]
    signature: SignatureManifest
    types: dict[str, list[dict[str, Any]]]
    constraints: dict[str, Any]
    tolerance: ToleranceManifest
    starter: str
    harness: HarnessManifest
    public: PublicTestsManifest
    internal: InternalTestsManifest
    benchmark: BenchmarkManifest
    timeouts: TimeoutsManifest
    compiler: CompilerManifest

    @field_validator("revision", mode="before")
    @classmethod
    def stringify_revision(cls, value: Any) -> str:
        return str(value)

    @model_validator(mode="after")
    def consistent_timeouts(self) -> ProblemManifest:
        if self.public.timeout_ms > self.timeouts.public_ms:
            raise ValueError("public timeout exceeds the top-level timeout")
        if self.internal.timeout_ms > self.timeouts.validation_ms:
            raise ValueError("internal timeout exceeds the top-level timeout")
        if self.benchmark.timeout_ms > self.timeouts.benchmark_ms:
            raise ValueError("benchmark timeout exceeds the top-level timeout")
        return self


class Problem:
    def __init__(self, root: Path, manifest: ProblemManifest):
        self.root = root.resolve()
        self.manifest = manifest
        self.statement_markdown = self._read_text("statement.md")
        self.starter_code = self._read_text(manifest.starter)
        self.header_path = self._resolve_file(manifest.signature.header)
        self.validator_path = self._resolve_file(manifest.harness.validator)
        self.benchmark_path = self._resolve_file(manifest.harness.benchmark)
        self.compile_flags = [*manifest.compiler.allowed_flags]
        files = [
            ("benchmark.cu", self.benchmark_path.read_bytes()),
            ("solve.h", self.header_path.read_bytes()),
        ]
        self.suite_hash = suite_hash(files, manifest.benchmark.model_dump(mode="json"))

    def _resolve_file(self, relative: str) -> Path:
        pure = PurePosixPath(relative.replace("\\", "/"))
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"problem path must stay relative: {relative}")
        candidate = (self.root / Path(*pure.parts)).resolve()
        if self.root not in candidate.parents or not candidate.is_file():
            raise ValueError(f"missing or escaped problem file: {relative}")
        return candidate

    def _read_text(self, relative: str) -> str:
        return self._resolve_file(relative).read_text(encoding="utf-8")

    def public_summary(self) -> dict[str, Any]:
        manifest = self.manifest
        return {
            "slug": manifest.slug,
            "title": manifest.title,
            "difficulty": manifest.difficulty,
            "revision": manifest.revision,
            "summary": self.statement_markdown.split("\n", 1)[0].lstrip("# ").strip(),
        }

    def public_detail(self) -> dict[str, Any]:
        manifest = self.manifest
        return {
            **self.public_summary(),
            "language": manifest.language,
            "statement_markdown": self.statement_markdown,
            "starter_code": self.starter_code,
            "signature": manifest.signature.model_dump(),
            "types": manifest.types,
            "constraints": manifest.constraints,
            "tolerance": manifest.tolerance.model_dump(),
            "benchmark": {
                "protocol_version": manifest.benchmark.protocol_version,
                "sizes": [item.model_dump() for item in manifest.benchmark.sizes],
                "warmup": manifest.benchmark.warmup,
                "iterations": manifest.benchmark.iterations,
            },
        }


class ProblemCatalog:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self._problems: dict[str, Problem] = {}

    def load(self) -> ProblemCatalog:
        if not self.root.is_dir():
            raise ValueError(f"problems directory does not exist: {self.root}")
        loaded: dict[str, Problem] = {}
        manifests = sorted(self.root.glob("*/problem.yaml"))
        if not manifests:
            raise ValueError("no problem manifests found")
        for manifest_path in manifests:
            raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            manifest = ProblemManifest.model_validate(raw)
            if manifest_path.parent.name != manifest.slug:
                raise ValueError(f"problem directory must equal slug: {manifest.slug}")
            if manifest.slug in loaded:
                raise ValueError(f"duplicate problem slug: {manifest.slug}")
            loaded[manifest.slug] = Problem(manifest_path.parent, manifest)
        self._problems = loaded
        return self

    def list(self) -> list[Problem]:
        return sorted(self._problems.values(), key=lambda item: item.manifest.slug)

    def get(self, slug: str) -> Problem:
        try:
            return self._problems[slug]
        except KeyError as error:
            raise KeyError(f"unknown problem: {slug}") from error

    def __len__(self) -> int:
        return len(self._problems)
