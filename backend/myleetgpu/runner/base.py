"""Transport-neutral source preparation and result handling for execution adapters.

This class deliberately has no execution/probing interface: those ports are
structural protocols, implemented independently by the Docker and Colab adapters.
Only local file preparation, language policy and bounded result handling are
shared here. Provider-specific health checks, transport and cleanup stay local.
"""

from __future__ import annotations

import json
import shutil
import stat
import time
from pathlib import Path
from typing import Any

from myleetgpu.config import Settings
from myleetgpu.domain.problems import Problem
from myleetgpu.filesystem import ensure_mode
from myleetgpu.runner.models import EnvironmentProbe, RunnerLanguage, RunnerUnavailable
from myleetgpu.runner.submission_policy import POLICY_VERSION as TRITON_POLICY_VERSION
from myleetgpu.runner.torch_submission_policy import POLICY_VERSION as TORCH_POLICY_VERSION

RESULT_PREFIX = "MYLEETGPU_RESULT="
CUDA_CPP: RunnerLanguage = "cuda_cpp"
TRITON_PYTHON: RunnerLanguage = "triton_python"
TORCH_PYTHON: RunnerLanguage = "torch_python"
PYTHON_LANGUAGES = frozenset({TRITON_PYTHON, TORCH_PYTHON})
TRITON_POLICY_FILENAME = "submission_policy.py"
TORCH_POLICY_SOURCE_FILENAME = "torch_submission_policy.py"


class BaseRunner:
    """Shared local mechanics; never constructs or probes a transport."""

    def __init__(self, settings: Settings, *, health_filename: str):
        self.settings = settings
        self._cached_probes: dict[RunnerLanguage, tuple[float, EnvironmentProbe]] = {}
        self._health_file = settings.data_dir / health_filename

    def mark_unhealthy(self, reason: str) -> None:
        self.settings.ensure_directories()
        payload = {"reason": reason[:2000], "marked_at": time.time()}
        temporary = self._health_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self._health_file)

    @staticmethod
    def _resolve_language(
        language: RunnerLanguage | str | None,
        *,
        implementation: Any | None = None,
        artifact: Path | None = None,
    ) -> RunnerLanguage:
        selected = language
        if selected is None and implementation is not None:
            selected = getattr(implementation, "language", None)
        if selected is None and artifact is not None and artifact.suffix == ".py":
            raise ValueError(
                "Python artifact language is ambiguous; pass triton_python or torch_python"
            )
        if selected is None:
            selected = CUDA_CPP
        normalized = str(getattr(selected, "value", selected))
        if normalized not in {CUDA_CPP, TRITON_PYTHON, TORCH_PYTHON}:
            raise ValueError(f"unknown runner language: {normalized}")
        return normalized  # type: ignore[return-value]

    @staticmethod
    def _harness_path(
        problem: Problem,
        implementation: Any | None,
        harness_kind: str,
    ) -> Path:
        owner = implementation if implementation is not None else problem
        attribute = "validator_path" if harness_kind == "validator" else "benchmark_path"
        path = getattr(owner, attribute, None)
        if not isinstance(path, Path) or not path.is_file():
            raise ValueError(f"{attribute} is required for the selected implementation")
        return path

    def prepare_compile(
        self,
        task_root: Path,
        problem: Problem,
        source_path: Path,
        harness_kind: str,
        *,
        language: RunnerLanguage | str | None = None,
        implementation: Any | None = None,
    ) -> Path:
        if harness_kind not in {"validator", "benchmark"}:
            raise ValueError("unknown harness kind")
        selected_language = self._resolve_language(language, implementation=implementation)
        harness_path = self._harness_path(problem, implementation, harness_kind)
        compile_dir = task_root / f"compile-{harness_kind}"
        compile_dir.mkdir(parents=True, exist_ok=False)
        if selected_language in PYTHON_LANGUAGES:
            source_target = compile_dir / "source.py"
            harness_target = compile_dir / "platform.py"
            policy_target = compile_dir / TRITON_POLICY_FILENAME
            shutil.copyfile(source_path, source_target)
            shutil.copyfile(harness_path, harness_target)
            policy_source = (
                TORCH_POLICY_SOURCE_FILENAME
                if selected_language == TORCH_PYTHON
                else TRITON_POLICY_FILENAME
            )
            shutil.copyfile(Path(__file__).with_name(policy_source), policy_target)
            for path in (source_target, harness_target, policy_target):
                ensure_mode(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            ensure_mode(compile_dir, 0o755)
            return compile_dir

        source_target = compile_dir / "source.cu"
        header_target = compile_dir / "solve.h"
        harness_target = compile_dir / "platform.cu"
        shutil.copyfile(source_path, source_target)
        owner = implementation if implementation is not None else problem
        header_path = getattr(owner, "header_path", None)
        if not isinstance(header_path, Path) or not header_path.is_file():
            raise ValueError("header_path is required for the CUDA C++ implementation")
        shutil.copyfile(header_path, header_target)
        shutil.copyfile(harness_path, harness_target)
        for path in (source_target, header_target, harness_target):
            ensure_mode(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        ensure_mode(compile_dir, 0o777)
        return compile_dir

    @staticmethod
    def effective_compile_flags(
        problem: Problem,
        probe: EnvironmentProbe,
        *,
        language: RunnerLanguage | str | None = None,
        implementation: Any | None = None,
    ) -> list[str]:
        selected_language = BaseRunner._resolve_language(language, implementation=implementation)
        if not probe.cuda_arch:
            raise RunnerUnavailable("CUDA architecture was not detected")
        if selected_language == TRITON_PYTHON:
            toolchain = probe.toolchain
            return [
                f"backend={TRITON_PYTHON}",
                f"policy={TRITON_POLICY_VERSION}",
                f"python={toolchain.get('python_version', 'unknown')}",
                f"torch={toolchain.get('torch_version', 'unknown')}",
                f"triton={toolchain.get('triton_version', 'unknown')}",
                f"torch_cuda={toolchain.get('torch_cuda_version', 'unknown')}",
                f"arch=sm_{probe.cuda_arch}",
            ]
        if selected_language == TORCH_PYTHON:
            toolchain = probe.toolchain
            return [
                f"backend={TORCH_PYTHON}",
                f"policy={TORCH_POLICY_VERSION}",
                f"python={toolchain.get('python_version', 'unknown')}",
                f"torch={toolchain.get('torch_version', 'unknown')}",
                f"torch_cuda={toolchain.get('torch_cuda_version', 'unknown')}",
                f"arch=sm_{probe.cuda_arch}",
                "float32_matmul_precision=highest",
                "tf32=false",
                "deterministic_algorithms=true",
            ]
        owner = implementation if implementation is not None else problem
        compile_flags = getattr(owner, "compile_flags", None)
        if not isinstance(compile_flags, list):
            raise ValueError("compile_flags are required for the CUDA C++ implementation")
        return [*compile_flags, f"-arch=sm_{probe.cuda_arch}"]

    def cleanup_task(self, task_root: Path) -> None:
        root = self.settings.jobs_dir.resolve()
        target = task_root.resolve()
        if target.parent != root or not target.name:
            raise ValueError(f"refusing to clean path outside job spool: {target}")
        if target.exists():
            shutil.rmtree(target)

    def _clean_diagnostics(self, output: str, task_dir: Path) -> str:
        cleaned = self._clean_output(output)
        replacements = {
            task_dir.as_posix(): "<job>",
            str(task_dir): "<job>",
            "/work/source.cu": "source.cu",
            "/work/platform.cu": "<platform>",
            "platform.cu": "<platform>",
            "/work/source.py": "source.py",
            "/work/platform.py": "<platform>",
            "platform.py": "<platform>",
            f"/work/{TRITON_POLICY_FILENAME}": "<platform-policy>",
            TRITON_POLICY_FILENAME: "<platform-policy>",
        }
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        return cleaned

    def _clean_output(self, output: str) -> str:
        sanitized = output.replace("\x00", "")
        if len(sanitized.encode("utf-8")) > self.settings.output_limit_bytes:
            encoded = sanitized.encode("utf-8")[: self.settings.output_limit_bytes]
            sanitized = encoded.decode("utf-8", errors="ignore") + "\n[platform] output truncated"
        return sanitized.strip()

    @staticmethod
    def _parse_result(output: str) -> dict[str, Any] | None:
        records = [line for line in output.splitlines() if line.startswith(RESULT_PREFIX)]
        if len(records) != 1:
            return None
        try:
            parsed = json.loads(records[0].removeprefix(RESULT_PREFIX))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _looks_like_gpu_health_failure(output: str) -> bool:
        lowered = output.lower()
        markers = (
            "cudaerrorunknown",
            "cuda driver version is insufficient",
            "no cuda-capable device",
            "failed to initialize nvml",
            "xid",
        )
        return any(marker in lowered for marker in markers)
