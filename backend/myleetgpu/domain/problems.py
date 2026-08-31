from __future__ import annotations

from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from myleetgpu.domain.benchmark import suite_hash


class KernelLanguage(StrEnum):
    CUDA_CPP = "cuda_cpp"
    TRITON_PYTHON = "triton_python"
    TORCH_PYTHON = "torch_python"


class StrictManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SignatureManifest(StrictManifestModel):
    header: str
    symbol: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    declaration: str


class PythonSignatureManifest(StrictManifestModel):
    symbol: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    declaration: str


# Kept as an alias for callers that imported the original schema name.
TritonSignatureManifest = PythonSignatureManifest


class ToleranceManifest(StrictManifestModel):
    mode: Literal["integer", "float"]
    atol: float = Field(default=0.0, ge=0)
    rtol: float = Field(default=0.0, ge=0)
    nan_equal: bool = False
    infinity: str = "same_sign"


class HarnessManifest(StrictManifestModel):
    validator: str
    benchmark: str
    support_files: list[str] = Field(default_factory=list, max_length=16)


class PublicTestsManifest(StrictManifestModel):
    seed: int
    timeout_ms: int = Field(gt=0, le=300_000)
    cases: list[dict[str, Any]] = Field(min_length=1)


class InternalTestsManifest(StrictManifestModel):
    seeds: list[int] = Field(min_length=1)
    timeout_ms: int = Field(gt=0, le=300_000)
    cases: list[dict[str, Any]] = Field(min_length=1)


class BenchmarkSize(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str = Field(min_length=1, max_length=64)
    inner_repetitions: int = Field(default=1, ge=1, le=100_000)


class BenchmarkManifest(StrictManifestModel):
    protocol_version: str
    suite_seed: int
    sizes: list[BenchmarkSize] = Field(min_length=1, max_length=32)
    warmup: int = Field(ge=1, le=10_000)
    iterations: int = Field(ge=3, le=500)
    timeout_ms: int = Field(gt=0, le=1_800_000)


class TimeoutsManifest(StrictManifestModel):
    compile_ms: int = Field(gt=0, le=600_000)
    public_ms: int = Field(gt=0, le=300_000)
    validation_ms: int = Field(gt=0, le=900_000)
    benchmark_ms: int = Field(gt=0, le=1_800_000)


class CompilerManifest(StrictManifestModel):
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


class TritonRuntimeManifest(StrictManifestModel):
    profile: Literal["triton_torch_cuda_v1"] = "triton_torch_cuda_v1"
    executable: Literal["python3"] = "python3"
    syntax_check: Literal["py_compile"] = "py_compile"


class TorchRuntimeManifest(StrictManifestModel):
    profile: Literal["torch_cuda_v1"] = "torch_cuda_v1"
    executable: Literal["python3"] = "python3"
    syntax_check: Literal["py_compile"] = "py_compile"


class CudaImplementationManifest(StrictManifestModel):
    language: Literal[KernelLanguage.CUDA_CPP]
    starter: str
    statement_appendix: str | None = None
    source_suffix: Literal[".cu"] = ".cu"
    signature: SignatureManifest
    harness: HarnessManifest
    compiler: CompilerManifest


class TritonImplementationManifest(StrictManifestModel):
    language: Literal[KernelLanguage.TRITON_PYTHON]
    starter: str
    statement_appendix: str | None = None
    source_suffix: Literal[".py"] = ".py"
    signature: TritonSignatureManifest
    harness: HarnessManifest
    runtime: TritonRuntimeManifest = Field(default_factory=TritonRuntimeManifest)


class TorchImplementationManifest(StrictManifestModel):
    language: Literal[KernelLanguage.TORCH_PYTHON]
    starter: str
    statement_appendix: str | None = None
    source_suffix: Literal[".py"] = ".py"
    signature: PythonSignatureManifest
    harness: HarnessManifest
    runtime: TorchRuntimeManifest = Field(default_factory=TorchRuntimeManifest)


ImplementationManifest = Annotated[
    CudaImplementationManifest | TritonImplementationManifest | TorchImplementationManifest,
    Field(discriminator="language"),
]


class ProblemManifest(StrictManifestModel):
    """A normalized problem manifest with schema-v1 compatibility."""

    schema_version: Literal[1, 2]
    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=128)
    title: str = Field(min_length=1, max_length=160)
    difficulty: Literal["easy", "medium", "hard"]
    revision: str
    default_language: KernelLanguage = KernelLanguage.CUDA_CPP
    implementations: dict[KernelLanguage, ImplementationManifest] = Field(min_length=1)
    types: dict[str, list[dict[str, Any]]]
    constraints: dict[str, Any]
    tolerance: ToleranceManifest
    public: PublicTestsManifest
    internal: InternalTestsManifest
    benchmark: BenchmarkManifest
    timeouts: TimeoutsManifest

    # Legacy schema-v1 fields. Schema v2 rejects these fields when they are populated.
    language: Literal[KernelLanguage.CUDA_CPP] | None = None
    signature: SignatureManifest | None = None
    starter: str | None = None
    harness: HarnessManifest | None = None
    compiler: CompilerManifest | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_schema_v1(cls, value: Any) -> Any:
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            return value
        normalized = dict(value)
        if normalized.get("implementations"):
            raise ValueError("schema v1 cannot declare implementations")
        legacy_fields = ("language", "signature", "starter", "harness", "compiler")
        missing = [field for field in legacy_fields if normalized.get(field) is None]
        if missing:
            raise ValueError(f"schema v1 is missing fields: {missing}")
        normalized["default_language"] = KernelLanguage.CUDA_CPP.value
        normalized["implementations"] = {
            KernelLanguage.CUDA_CPP.value: {
                "language": KernelLanguage.CUDA_CPP.value,
                "starter": normalized["starter"],
                "signature": normalized["signature"],
                "harness": normalized["harness"],
                "compiler": normalized["compiler"],
            }
        }
        return normalized

    @field_validator("revision", mode="before")
    @classmethod
    def stringify_revision(cls, value: Any) -> str:
        return str(value)

    @model_validator(mode="after")
    def validate_protocol(self) -> ProblemManifest:
        if self.schema_version == 2 and any(
            value is not None
            for value in (self.language, self.signature, self.starter, self.harness, self.compiler)
        ):
            raise ValueError("schema v2 must declare language assets only in implementations")
        if self.default_language not in self.implementations:
            raise ValueError("default_language is not present in implementations")
        for key, implementation in self.implementations.items():
            if key != implementation.language:
                raise ValueError("implementation key must match its language")
        if self.public.timeout_ms > self.timeouts.public_ms:
            raise ValueError("public timeout exceeds the top-level timeout")
        if self.internal.timeout_ms > self.timeouts.validation_ms:
            raise ValueError("internal timeout exceeds the top-level timeout")
        if self.benchmark.timeout_ms > self.timeouts.benchmark_ms:
            raise ValueError("benchmark timeout exceeds the top-level timeout")
        return self


class ProblemImplementation:
    def __init__(
        self,
        problem: Problem,
        manifest: (
            CudaImplementationManifest | TritonImplementationManifest | TorchImplementationManifest
        ),
    ):
        self.problem = problem
        self.manifest = manifest
        self.language = KernelLanguage(manifest.language)
        self.source_suffix = manifest.source_suffix
        self.editor_language = "cpp" if self.language is KernelLanguage.CUDA_CPP else "python"
        self.display_name = {
            KernelLanguage.CUDA_CPP: "CUDA C++",
            KernelLanguage.TRITON_PYTHON: "Triton Python",
            KernelLanguage.TORCH_PYTHON: "PyTorch Python",
        }[self.language]
        self.signature = manifest.signature
        self.starter_path = problem._resolve_file(manifest.starter)
        self.starter_code = self.starter_path.read_text(encoding="utf-8")
        self.validator_path = problem._resolve_file(manifest.harness.validator)
        self.benchmark_path = problem._resolve_file(manifest.harness.benchmark)
        self.support_paths = tuple(
            problem._resolve_file(relative) for relative in manifest.harness.support_files
        )
        self.statement_appendix = (
            problem._read_text(manifest.statement_appendix)
            if manifest.statement_appendix is not None
            else None
        )

        if isinstance(manifest, CudaImplementationManifest):
            self.header_path: Path | None = problem._resolve_file(manifest.signature.header)
            self.compile_flags = [*manifest.compiler.allowed_flags]
            self.toolchain_profile = "cuda_nvcc_v1"
        else:
            self.header_path = None
            self.compile_flags = []
            self.toolchain_profile = manifest.runtime.profile

        hash_files = [(self.benchmark_path.name, self.benchmark_path.read_bytes())]
        if self.header_path is not None:
            hash_files.append((self.header_path.name, self.header_path.read_bytes()))
        hash_files.extend(
            (path.relative_to(problem.root).as_posix(), path.read_bytes())
            for path in self.support_paths
        )
        self.suite_hash = suite_hash(
            hash_files,
            problem.manifest.benchmark.model_dump(mode="json"),
        )

    def public_payload(self) -> dict[str, Any]:
        return {
            "language": self.language.value,
            "display_name": self.display_name,
            "source_suffix": self.source_suffix,
            "editor_language": self.editor_language,
            "starter_code": self.starter_code,
            "statement_appendix": self.statement_appendix,
            "signature": self.signature.model_dump(mode="json"),
        }


class Problem:
    def __init__(self, root: Path, manifest: ProblemManifest):
        self.root = root.resolve()
        self.manifest = manifest
        self.statement_markdown = self._read_text("statement.md")
        self.default_language = KernelLanguage(manifest.default_language)
        self.implementations = {
            language: ProblemImplementation(self, implementation)
            for language, implementation in manifest.implementations.items()
        }
        self.supported_languages = tuple(
            language for language in KernelLanguage if language in self.implementations
        )

        # Backwards-compatible attributes map to the default (CUDA for built-ins) implementation.
        default = self.get_implementation()
        self.starter_code = default.starter_code
        self.header_path = default.header_path
        self.validator_path = default.validator_path
        self.benchmark_path = default.benchmark_path
        self.compile_flags = default.compile_flags
        self.suite_hash = default.suite_hash

    def get_implementation(
        self, language: KernelLanguage | str | None = None
    ) -> ProblemImplementation:
        try:
            selected = self.default_language if language is None else KernelLanguage(language)
            return self.implementations[selected]
        except (KeyError, ValueError) as error:
            requested = self.default_language.value if language is None else str(language)
            raise KeyError(
                f"problem {self.manifest.slug} does not support language: {requested}"
            ) from error

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
            "languages": [language.value for language in self.supported_languages],
        }

    def public_detail(self) -> dict[str, Any]:
        manifest = self.manifest
        default = self.get_implementation()
        return {
            **self.public_summary(),
            # Preserve the schema-v1 API for the default implementation.
            "language": default.language.value,
            "starter_code": default.starter_code,
            "signature": default.signature.model_dump(mode="json"),
            "default_language": self.default_language.value,
            "supported_languages": [language.value for language in self.supported_languages],
            "implementations": {
                language.value: implementation.public_payload()
                for language, implementation in self.implementations.items()
            },
            "statement_markdown": self.statement_markdown,
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
