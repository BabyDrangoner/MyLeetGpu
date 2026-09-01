from __future__ import annotations

import json
import os
import queue
import re
import shutil
import stat
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from myleetgpu.config import Settings
from myleetgpu.domain.benchmark import stable_hash
from myleetgpu.domain.problems import Problem
from myleetgpu.filesystem import ensure_mode
from myleetgpu.runner.models import (
    CommandResult,
    CompileResult,
    EnvironmentProbe,
    ExecutionResult,
    RunnerLanguage,
    RunnerUnavailable,
    RunnerUnhealthy,
)
from myleetgpu.runner.submission_policy import POLICY_VERSION as TRITON_POLICY_VERSION
from myleetgpu.runner.torch_submission_policy import (
    POLICY_VERSION as TORCH_POLICY_VERSION,
)
from myleetgpu.runner.torch_submission_policy import (
    submission_contract_from_declaration,
)

RESULT_PREFIX = "MYLEETGPU_RESULT="
CONTAINER_USER = "65534:65534"
CONTAINER_WORKDIR = "/work"
CONTAINER_FILE_BYTES = 64 * 1024 * 1024
RUNNER_LABEL = "com.myleetgpu.runner=true"
INSTALLATION_LABEL_KEY = "com.myleetgpu.installation"
OWNER_LABEL_KEY = "com.myleetgpu.owner"
CUDA_CPP: RunnerLanguage = "cuda_cpp"
TRITON_PYTHON: RunnerLanguage = "triton_python"
TORCH_PYTHON: RunnerLanguage = "torch_python"
PYTHON_LANGUAGES = frozenset({TRITON_PYTHON, TORCH_PYTHON})
PYTHON_TMPFS = "/tmp:rw,nosuid,nodev,exec,size=512m"
TRITON_TMPFS = PYTHON_TMPFS
TORCH_TMPFS = "/tmp:rw,nosuid,nodev,noexec,size=512m"
CUDA_TMPFS = "/tmp:rw,nosuid,nodev,noexec,size=64m"
TRITON_POLICY_FILENAME = "submission_policy.py"
TORCH_POLICY_SOURCE_FILENAME = "torch_submission_policy.py"


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", value)[:80]


class DockerRunner:
    """The only adapter allowed to translate platform operations into Docker arguments."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._cached_probes: dict[RunnerLanguage, tuple[float, EnvironmentProbe]] = {}
        self._health_file = settings.data_dir / "runner-unhealthy.json"
        self._docker = os.environ.get("MYLEETGPU_DOCKER_BIN", "docker")
        installation = stable_hash(settings.host_data_mount)[:16]
        self._installation_label = f"{INSTALLATION_LABEL_KEY}={installation}"
        self._owner_label: str | None = None

    def assign_owner(self, owner: str) -> None:
        self._owner_label = f"{OWNER_LABEL_KEY}={stable_hash(owner)[:16]}"

    def assert_healthy(self) -> None:
        if self._health_file.exists():
            try:
                reason = json.loads(self._health_file.read_text(encoding="utf-8")).get("reason")
            except (OSError, ValueError, TypeError):
                reason = "GPU runner has been marked unhealthy"
            raise RunnerUnhealthy(
                f"{reason}. Run `make doctor`, resolve the GPU problem, "
                "then run the recovery command."
            )

    def mark_unhealthy(self, reason: str) -> None:
        self.settings.ensure_directories()
        payload = {"reason": reason[:2000], "marked_at": time.time()}
        temporary = self._health_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self._health_file)

    def recover(self) -> EnvironmentProbe:
        probe = self.probe_environment(force=True, ignore_circuit_breaker=True)
        if not probe.healthy:
            raise RunnerUnavailable(probe.error or "GPU environment probe failed")
        self._health_file.unlink(missing_ok=True)
        return probe

    def probe_environment(
        self, *, force: bool = False, ignore_circuit_breaker: bool = False
    ) -> EnvironmentProbe:
        if not ignore_circuit_breaker:
            self.assert_healthy()
        cached = self._cached_probes.get(CUDA_CPP)
        if cached and not force and time.monotonic() - cached[0] < 60:
            return cached[1]

        try:
            server = self._run_limited(
                [self._docker, "version", "--format", "{{.Server.Version}}"],
                timeout=10,
                limit=8192,
            )
            if server.returncode != 0 or not server.output.strip():
                message = self._clean_output(server.output) or "Docker daemon unavailable"
                raise RunnerUnavailable(message)
            inspect = self._run_limited(
                [
                    self._docker,
                    "image",
                    "inspect",
                    "--format",
                    "{{json .RepoDigests}}",
                    self.settings.cuda_image,
                ],
                timeout=15,
                limit=16_384,
            )
            if inspect.returncode != 0:
                raise RunnerUnavailable(
                    f"fixed CUDA image is unavailable: {self._clean_output(inspect.output)}"
                )
            digests = json.loads(inspect.output.strip() or "[]")
            image_digest = digests[0] if digests else None
            gpu = self._docker_probe(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                gpu=True,
                timeout=20,
            )
            if gpu.returncode != 0:
                raise RunnerUnavailable(f"GPU probe failed: {self._clean_output(gpu.output)}")
            gpu_fields = [part.strip() for part in gpu.output.strip().splitlines()[0].split(",")]
            if len(gpu_fields) != 3:
                raise RunnerUnavailable("nvidia-smi returned an unexpected GPU description")
            gpu_name, driver_version, compute_capability = gpu_fields
            nvcc = self._docker_probe(["nvcc", "--version"], gpu=False, timeout=20)
            if nvcc.returncode != 0:
                raise RunnerUnavailable(f"NVCC probe failed: {self._clean_output(nvcc.output)}")
            nvcc_line = next(
                (line.strip() for line in reversed(nvcc.output.splitlines()) if "release" in line),
                nvcc.output.strip().splitlines()[-1],
            )
            runtime_version = self._image_runtime_version()
            telemetry = self._probe_telemetry()
            configured = self.settings.cuda_arch
            cuda_arch = compute_capability.replace(".", "") if configured == "auto" else configured
            fingerprint_payload = {
                "gpu": gpu_name,
                "compute_capability": compute_capability,
                "driver": driver_version,
                "runtime": runtime_version,
                "nvcc": nvcc_line,
                "image": image_digest or self.settings.cuda_image,
                "arch": cuda_arch,
            }
            probe = EnvironmentProbe(
                healthy=True,
                gpu_name=gpu_name,
                compute_capability=compute_capability,
                driver_version=driver_version,
                cuda_runtime_version=runtime_version,
                nvcc_version=nvcc_line,
                cuda_image=self.settings.cuda_image,
                image_digest=image_digest,
                cuda_arch=cuda_arch,
                telemetry=telemetry,
                fingerprint=stable_hash(fingerprint_payload),
                backend=CUDA_CPP,
                toolchain={
                    "cuda_runtime_version": runtime_version,
                    "nvcc_version": nvcc_line,
                },
            )
        except (
            OSError,
            subprocess.SubprocessError,
            ValueError,
            IndexError,
            RunnerUnavailable,
        ) as error:
            probe = EnvironmentProbe(
                healthy=False,
                gpu_name=None,
                compute_capability=None,
                driver_version=None,
                cuda_runtime_version=None,
                nvcc_version=None,
                cuda_image=self.settings.cuda_image,
                image_digest=None,
                cuda_arch=None,
                telemetry={
                    "temperature_c": None,
                    "power_w": None,
                    "sm_clock_mhz": None,
                    "gpu_busy_percent": None,
                },
                error=str(error),
                fingerprint=stable_hash(
                    {"healthy": False, "image": self.settings.cuda_image, "error": str(error)}
                ),
                backend=CUDA_CPP,
            )
        self._cached_probes[CUDA_CPP] = (time.monotonic(), probe)
        return probe

    def probe_triton_environment(
        self, *, force: bool = False, ignore_circuit_breaker: bool = False
    ) -> EnvironmentProbe:
        """Probe the optional Triton toolchain without mutating CUDA runner health."""

        if not ignore_circuit_breaker:
            self.assert_healthy()
        cached = self._cached_probes.get(TRITON_PYTHON)
        if cached and not force and time.monotonic() - cached[0] < 60:
            return cached[1]

        image = self.settings.triton_image
        try:
            server = self._run_limited(
                [self._docker, "version", "--format", "{{.Server.Version}}"],
                timeout=10,
                limit=8192,
            )
            if server.returncode != 0 or not server.output.strip():
                message = self._clean_output(server.output) or "Docker daemon unavailable"
                raise RunnerUnavailable(message)
            image_digest = self._inspect_image_digest(image)
            gpu = self._docker_probe(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                gpu=True,
                timeout=20,
                image=image,
                platform_owned=False,
            )
            if gpu.returncode != 0:
                raise RunnerUnavailable(
                    f"Triton GPU probe failed: {self._clean_output(gpu.output)}"
                )
            gpu_fields = [part.strip() for part in gpu.output.strip().splitlines()[0].split(",")]
            if len(gpu_fields) != 3:
                raise RunnerUnavailable("nvidia-smi returned an unexpected GPU description")
            gpu_name, driver_version, compute_capability = gpu_fields

            script = (
                "import json,platform,torch,triton;"
                "print(json.dumps({"
                "'python_version':platform.python_version(),"
                "'torch_version':torch.__version__,"
                "'triton_version':triton.__version__,"
                "'torch_cuda_version':torch.version.cuda,"
                "'cuda_available':torch.cuda.is_available(),"
                "'cuda_device_count':torch.cuda.device_count()}))"
            )
            versions = self._docker_probe(
                ["python", "-I", "-B", "-c", script],
                gpu=True,
                timeout=30,
                image=image,
                platform_owned=False,
            )
            if versions.returncode != 0:
                raise RunnerUnavailable(
                    f"Triton toolchain probe failed: {self._clean_output(versions.output)}"
                )
            toolchain = json.loads(versions.output.strip())
            required = {
                "python_version",
                "torch_version",
                "triton_version",
                "torch_cuda_version",
            }
            if not isinstance(toolchain, dict) or not required.issubset(toolchain):
                raise RunnerUnavailable("Triton toolchain probe returned an unexpected payload")
            if any(
                not isinstance(toolchain[key], str) or not toolchain[key].strip()
                for key in required
            ):
                raise RunnerUnavailable("Triton toolchain probe returned incomplete versions")
            if toolchain.get("cuda_available") is not True:
                raise RunnerUnavailable("PyTorch cannot access CUDA in the Triton runner image")
            if toolchain.get("cuda_device_count") != 1:
                raise RunnerUnavailable("Triton runner must expose exactly GPU 0")

            configured = self.settings.cuda_arch
            cuda_arch = compute_capability.replace(".", "") if configured == "auto" else configured
            public_toolchain = {key: toolchain[key] for key in sorted(required)}
            fingerprint_payload = {
                "backend": TRITON_PYTHON,
                "gpu": gpu_name,
                "compute_capability": compute_capability,
                "driver": driver_version,
                "image": image_digest or image,
                "arch": cuda_arch,
                "toolchain": public_toolchain,
            }
            probe = EnvironmentProbe(
                healthy=True,
                gpu_name=gpu_name,
                compute_capability=compute_capability,
                driver_version=driver_version,
                cuda_runtime_version=str(toolchain["torch_cuda_version"]),
                nvcc_version=None,
                cuda_image=image,
                image_digest=image_digest,
                cuda_arch=cuda_arch,
                telemetry=self._probe_telemetry(image=image, platform_owned=False),
                fingerprint=stable_hash(fingerprint_payload),
                backend=TRITON_PYTHON,
                toolchain=public_toolchain,
            )
        except (
            OSError,
            subprocess.SubprocessError,
            ValueError,
            TypeError,
            IndexError,
            KeyError,
            RunnerUnavailable,
        ) as error:
            probe = EnvironmentProbe(
                healthy=False,
                gpu_name=None,
                compute_capability=None,
                driver_version=None,
                cuda_runtime_version=None,
                nvcc_version=None,
                cuda_image=image,
                image_digest=None,
                cuda_arch=None,
                telemetry={
                    "temperature_c": None,
                    "power_w": None,
                    "sm_clock_mhz": None,
                    "gpu_busy_percent": None,
                },
                error=str(error),
                fingerprint=stable_hash(
                    {
                        "healthy": False,
                        "backend": TRITON_PYTHON,
                        "image": image,
                        "error": str(error),
                    }
                ),
                backend=TRITON_PYTHON,
                toolchain={},
            )
        self._cached_probes[TRITON_PYTHON] = (time.monotonic(), probe)
        return probe

    def probe_torch_environment(
        self, *, force: bool = False, ignore_circuit_breaker: bool = False
    ) -> EnvironmentProbe:
        """Probe the pure PyTorch backend independently from Triton availability."""

        if not ignore_circuit_breaker:
            self.assert_healthy()
        cached = self._cached_probes.get(TORCH_PYTHON)
        if cached and not force and time.monotonic() - cached[0] < 60:
            return cached[1]

        image = self.settings.triton_image
        try:
            server = self._run_limited(
                [self._docker, "version", "--format", "{{.Server.Version}}"],
                timeout=10,
                limit=8192,
            )
            if server.returncode != 0 or not server.output.strip():
                message = self._clean_output(server.output) or "Docker daemon unavailable"
                raise RunnerUnavailable(message)
            image_digest = self._inspect_image_digest(image)
            gpu = self._docker_probe(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                gpu=True,
                timeout=20,
                image=image,
                platform_owned=False,
            )
            if gpu.returncode != 0:
                raise RunnerUnavailable(
                    f"PyTorch GPU probe failed: {self._clean_output(gpu.output)}"
                )
            gpu_fields = [part.strip() for part in gpu.output.strip().splitlines()[0].split(",")]
            if len(gpu_fields) != 3:
                raise RunnerUnavailable("nvidia-smi returned an unexpected GPU description")
            gpu_name, driver_version, compute_capability = gpu_fields

            script = (
                "import json,platform,torch;"
                "print(json.dumps({"
                "'python_version':platform.python_version(),"
                "'torch_version':torch.__version__,"
                "'torch_cuda_version':torch.version.cuda,"
                "'cuda_available':torch.cuda.is_available(),"
                "'cuda_device_count':torch.cuda.device_count()}))"
            )
            versions = self._docker_probe(
                ["python", "-I", "-B", "-c", script],
                gpu=True,
                timeout=30,
                image=image,
                platform_owned=False,
            )
            if versions.returncode != 0:
                raise RunnerUnavailable(
                    f"PyTorch toolchain probe failed: {self._clean_output(versions.output)}"
                )
            toolchain = json.loads(versions.output.strip())
            required = {"python_version", "torch_version", "torch_cuda_version"}
            if not isinstance(toolchain, dict) or not required.issubset(toolchain):
                raise RunnerUnavailable("PyTorch toolchain probe returned an unexpected payload")
            if any(
                not isinstance(toolchain[key], str) or not toolchain[key].strip()
                for key in required
            ):
                raise RunnerUnavailable("PyTorch toolchain probe returned incomplete versions")
            if toolchain.get("cuda_available") is not True:
                raise RunnerUnavailable("PyTorch cannot access CUDA in the runner image")
            if toolchain.get("cuda_device_count") != 1:
                raise RunnerUnavailable("PyTorch runner must expose exactly GPU 0")

            configured = self.settings.cuda_arch
            cuda_arch = compute_capability.replace(".", "") if configured == "auto" else configured
            public_toolchain = {key: toolchain[key] for key in sorted(required)}
            fingerprint_payload = {
                "backend": TORCH_PYTHON,
                "gpu": gpu_name,
                "compute_capability": compute_capability,
                "driver": driver_version,
                "image": image_digest or image,
                "arch": cuda_arch,
                "toolchain": public_toolchain,
            }
            probe = EnvironmentProbe(
                healthy=True,
                gpu_name=gpu_name,
                compute_capability=compute_capability,
                driver_version=driver_version,
                cuda_runtime_version=str(toolchain["torch_cuda_version"]),
                nvcc_version=None,
                cuda_image=image,
                image_digest=image_digest,
                cuda_arch=cuda_arch,
                telemetry=self._probe_telemetry(image=image, platform_owned=False),
                fingerprint=stable_hash(fingerprint_payload),
                backend=TORCH_PYTHON,
                toolchain=public_toolchain,
            )
        except (
            OSError,
            subprocess.SubprocessError,
            ValueError,
            TypeError,
            IndexError,
            KeyError,
            RunnerUnavailable,
        ) as error:
            probe = EnvironmentProbe(
                healthy=False,
                gpu_name=None,
                compute_capability=None,
                driver_version=None,
                cuda_runtime_version=None,
                nvcc_version=None,
                cuda_image=image,
                image_digest=None,
                cuda_arch=None,
                telemetry={
                    "temperature_c": None,
                    "power_w": None,
                    "sm_clock_mhz": None,
                    "gpu_busy_percent": None,
                },
                error=str(error),
                fingerprint=stable_hash(
                    {
                        "healthy": False,
                        "backend": TORCH_PYTHON,
                        "image": image,
                        "error": str(error),
                    }
                ),
                backend=TORCH_PYTHON,
                toolchain={},
            )
        self._cached_probes[TORCH_PYTHON] = (time.monotonic(), probe)
        return probe

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

    def compile(
        self,
        task_root: Path,
        problem: Problem,
        source_path: Path,
        *,
        harness_kind: str = "validator",
        language: RunnerLanguage | str | None = None,
        implementation: Any | None = None,
    ) -> CompileResult:
        selected_language = self._resolve_language(language, implementation=implementation)
        if selected_language in PYTHON_LANGUAGES:
            return self._compile_python(
                task_root,
                problem,
                source_path,
                harness_kind=harness_kind,
                language=selected_language,
                implementation=implementation,
            )

        probe = self.probe_environment()
        if not probe.healthy or not probe.cuda_arch:
            raise RunnerUnavailable(probe.error or "CUDA environment is unavailable")
        compile_dir = self.prepare_compile(
            task_root,
            problem,
            source_path,
            harness_kind,
            language=CUDA_CPP,
            implementation=implementation,
        )
        container_name = _safe_name(f"myleetgpu-{task_root.name}-compile-{harness_kind}")
        command = [
            *self._base_container_args(container_name, compile_dir, gpu=False),
            "--entrypoint",
            "nvcc",
            self.settings.cuda_image,
            *self.effective_compile_flags(
                problem,
                probe,
                language=CUDA_CPP,
                implementation=implementation,
            ),
            "-I/work",
            "/work/source.cu",
            "/work/platform.cu",
            "-o",
            "/work/program",
        ]
        timeout = min(
            self.settings.compile_timeout_seconds,
            problem.manifest.timeouts.compile_ms / 1000,
        )
        result = self._run_container(
            command,
            container_name,
            timeout=timeout,
            platform_owned=True,
        )
        diagnostics = self._clean_diagnostics(result.output, compile_dir)
        executable = compile_dir / "program"
        succeeded = result.returncode == 0 and executable.is_file()
        return CompileResult(
            succeeded=succeeded,
            diagnostics=diagnostics,
            executable=executable if succeeded else None,
            duration_seconds=result.duration_seconds,
            timed_out=result.timed_out,
            output_limited=result.output_limited,
        )

    def _compile_python(
        self,
        task_root: Path,
        problem: Problem,
        source_path: Path,
        *,
        harness_kind: str,
        language: RunnerLanguage,
        implementation: Any | None,
    ) -> CompileResult:
        policy_arguments = [f"/work/{TRITON_POLICY_FILENAME}", "/work/source.py"]
        if language == TORCH_PYTHON:
            owner = (
                implementation
                if implementation is not None
                else problem.get_implementation(TORCH_PYTHON)
            )
            contract = submission_contract_from_declaration(
                owner.signature.symbol,
                owner.signature.declaration,
            )
            policy_arguments.append(
                json.dumps(contract, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            )
        # Refuse Docker's implicit pull path. The optional runtime must be
        # provisioned deliberately, and its absence must not trip CUDA health.
        self._inspect_image_digest(self.settings.triton_image)
        compile_dir = self.prepare_compile(
            task_root,
            problem,
            source_path,
            harness_kind,
            language=language,
            implementation=implementation,
        )
        container_name = _safe_name(f"myleetgpu-{task_root.name}-compile-{harness_kind}-{language}")
        command = [
            *self._base_container_args(
                container_name,
                compile_dir,
                gpu=False,
                language=language,
                read_only_workdir=True,
            ),
            "--entrypoint",
            "python",
            self.settings.triton_image,
            "-I",
            "-B",
            *policy_arguments,
        ]
        timeout = min(
            self.settings.compile_timeout_seconds,
            problem.manifest.timeouts.compile_ms / 1000,
        )
        result = self._run_container(
            command,
            container_name,
            timeout=timeout,
            platform_owned=False,
        )
        diagnostics = self._clean_diagnostics(result.output, compile_dir)
        artifact = compile_dir / "source.py"
        succeeded = result.returncode == 0 and artifact.is_file()
        return CompileResult(
            succeeded=succeeded,
            diagnostics=diagnostics,
            executable=artifact if succeeded else None,
            duration_seconds=result.duration_seconds,
            timed_out=result.timed_out,
            output_limited=result.output_limited,
        )

    @staticmethod
    def effective_compile_flags(
        problem: Problem,
        probe: EnvironmentProbe,
        *,
        language: RunnerLanguage | str | None = None,
        implementation: Any | None = None,
    ) -> list[str]:
        selected_language = DockerRunner._resolve_language(language, implementation=implementation)
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

    def execute(
        self,
        task_root: Path,
        executable: Path,
        *,
        mode: str,
        timeout_seconds: float,
        language: RunnerLanguage | str | None = None,
    ) -> ExecutionResult:
        if mode not in {"public", "full", "benchmark"}:
            raise ValueError("unknown execution mode")
        selected_language = self._resolve_language(language, artifact=executable)
        self.assert_healthy()
        run_dir = task_root / f"run-{mode}"
        run_dir.mkdir(parents=True, exist_ok=False)
        if selected_language in PYTHON_LANGUAGES:
            run_target = run_dir / "source.py"
            platform_source = executable.parent / "platform.py"
            if not platform_source.is_file():
                raise ValueError("compiled Python artifact is missing platform.py")
            platform_target = run_dir / "platform.py"
            policy_source = executable.parent / TRITON_POLICY_FILENAME
            if not policy_source.is_file():
                raise ValueError("compiled Python artifact is missing submission policy")
            policy_target = run_dir / TRITON_POLICY_FILENAME
            shutil.copyfile(executable, run_target)
            shutil.copyfile(platform_source, platform_target)
            shutil.copyfile(policy_source, policy_target)
            ensure_mode(run_target, 0o444)
            ensure_mode(platform_target, 0o444)
            ensure_mode(policy_target, 0o444)
        else:
            run_target = run_dir / "program"
            shutil.copyfile(executable, run_target)
            ensure_mode(run_target, 0o555)
        # The bind mount is already readonly inside the submitted-code
        # container. Keep the host owner write bit so a non-root Worker can
        # unlink the executable when the job spool is cleaned.
        ensure_mode(run_dir, 0o755)
        container_name = _safe_name(f"myleetgpu-{task_root.name}-run-{mode}")
        if selected_language in PYTHON_LANGUAGES:
            command = [
                *self._base_container_args(
                    container_name,
                    run_dir,
                    gpu=True,
                    language=selected_language,
                ),
                "--entrypoint",
                "python",
                self.settings.triton_image,
                "-I",
                "-B",
                "/work/platform.py",
                "--mode",
                mode,
            ]
        else:
            program_args: list[str] = []
            if mode in {"public", "full"}:
                program_args.extend(["--mode", mode])
            command = [
                *self._base_container_args(container_name, run_dir, gpu=True),
                "--entrypoint",
                "/work/program",
                self.settings.cuda_image,
                *program_args,
            ]
        result = self._run_container(command, container_name, timeout=timeout_seconds)
        output = self._clean_diagnostics(result.output, run_dir)
        parsed = self._parse_result(output)
        succeeded = (
            result.returncode == 0 and parsed is not None and parsed.get("status") == "passed"
        )
        if result.returncode != 0 and self._looks_like_gpu_health_failure(output):
            trusted_probe = self._docker_probe(["nvidia-smi", "-L"], gpu=True, timeout=15)
            if trusted_probe.returncode != 0:
                reason = self._clean_output(trusted_probe.output) or "trusted GPU probe failed"
                self.mark_unhealthy(f"trusted GPU probe failed after execution: {reason}")
        return ExecutionResult(
            succeeded=succeeded,
            output=output,
            parsed=parsed,
            duration_seconds=result.duration_seconds,
            returncode=result.returncode,
            timed_out=result.timed_out,
            output_limited=result.output_limited,
        )

    def cleanup_task(self, task_root: Path) -> None:
        root = self.settings.jobs_dir.resolve()
        target = task_root.resolve()
        if target.parent != root or not target.name:
            raise ValueError(f"refusing to clean path outside job spool: {target}")
        if target.exists():
            shutil.rmtree(target)

    def cleanup_orphan_containers(self) -> list[str]:
        """Remove platform containers left behind by a terminated worker."""

        return self._cleanup_containers([RUNNER_LABEL, self._installation_label])

    def cleanup_owned_containers(self) -> list[str]:
        """Stop only containers belonging to this Runner owner after lease loss."""

        if self._owner_label is None:
            return []
        return self._cleanup_containers([RUNNER_LABEL, self._installation_label, self._owner_label])

    def _cleanup_containers(self, labels: Sequence[str]) -> list[str]:
        try:
            command = [self._docker, "ps", "-aq"]
            for label in labels:
                command.extend(["--filter", f"label={label}"])
            listed = self._run_limited(command, timeout=10, limit=16_384)
        except OSError:
            return []
        if listed.returncode != 0:
            return []
        identifiers = [line.strip() for line in listed.output.splitlines() if line.strip()]
        if identifiers:
            self._run_limited([self._docker, "rm", "-f", *identifiers], timeout=20, limit=16_384)
        return identifiers

    def _base_container_args(
        self,
        name: str,
        task_dir: Path,
        *,
        gpu: bool,
        language: RunnerLanguage | str = CUDA_CPP,
        read_only_workdir: bool | None = None,
    ) -> list[str]:
        selected_language = self._resolve_language(language)
        host_path = self._host_task_path(task_dir)
        mount_read_only = gpu if read_only_workdir is None else read_only_workdir
        args = [
            self._docker,
            "run",
            "--rm",
            "--name",
            name,
            "--label",
            RUNNER_LABEL,
            "--label",
            self._installation_label,
            "--network",
            "none",
            "--log-driver",
            "none",
            "--read-only",
            "--user",
            CONTAINER_USER,
            "--cap-drop=ALL",
            "--security-opt",
            "no-new-privileges=true",
            "--ipc",
            "private",
            "--pids-limit",
            "128",
            "--memory",
            self.settings.container_memory,
            "--memory-swap",
            self.settings.container_memory,
            "--cpus",
            str(self.settings.container_cpus),
            "--ulimit",
            f"fsize={CONTAINER_FILE_BYTES}:{CONTAINER_FILE_BYTES}",
            "--tmpfs",
            (
                TORCH_TMPFS
                if selected_language == TORCH_PYTHON
                else PYTHON_TMPFS
                if selected_language == TRITON_PYTHON
                else CUDA_TMPFS
            ),
        ]
        environment = (
            [
                "HOME=/tmp",
                "CUDA_VISIBLE_DEVICES=0",
                "PYTHONHASHSEED=0",
                "OMP_NUM_THREADS=1",
                "MKL_NUM_THREADS=1",
                "OPENBLAS_NUM_THREADS=1",
                "TRITON_CACHE_DIR=/tmp/triton-cache",
                "XDG_CACHE_HOME=/tmp/cache",
                "PYTHONPYCACHEPREFIX=/tmp/pycache",
                "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
            ]
            if selected_language in PYTHON_LANGUAGES
            else ["HOME=/tmp", "CUDA_CACHE_DISABLE=1"]
        )
        for value in environment:
            args.extend(["--env", value])
        if selected_language == TORCH_PYTHON:
            args.extend(["--env", "CUBLAS_WORKSPACE_CONFIG=:4096:8"])
            args.extend(["--env", "NVIDIA_TF32_OVERRIDE=0"])
        args.extend(
            [
                "--mount",
                (
                    f"type=bind,src={host_path},dst={CONTAINER_WORKDIR},readonly"
                    if mount_read_only
                    else f"type=bind,src={host_path},dst={CONTAINER_WORKDIR}"
                ),
                "--workdir",
                CONTAINER_WORKDIR,
                "--stop-timeout",
                "1",
            ]
        )
        if self._owner_label is not None:
            args[args.index("--network") : args.index("--network")] = [
                "--label",
                self._owner_label,
            ]
        if gpu:
            args.extend(["--gpus", "device=0"])
        return args

    def _host_task_path(self, task_dir: Path) -> str:
        jobs_root = self.settings.jobs_dir.resolve()
        resolved = task_dir.resolve()
        if jobs_root not in resolved.parents:
            raise ValueError("container mount must be an exact child of the job spool")
        relative = resolved.relative_to(self.settings.data_dir.resolve())
        relative_text = relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("resolved host mount escaped the configured data directory")
        return f"{self.settings.host_data_mount}/{relative_text}"

    def _docker_probe(
        self,
        command: Sequence[str],
        *,
        gpu: bool,
        timeout: float,
        image: str | None = None,
        platform_owned: bool = True,
    ) -> CommandResult:
        selected_image = image or self.settings.cuda_image
        name = _safe_name(f"myleetgpu-probe-{time.time_ns()}")
        args = [
            self._docker,
            "run",
            "--rm",
            "--name",
            name,
            "--label",
            RUNNER_LABEL,
            "--label",
            self._installation_label,
            "--network",
            "none",
            "--log-driver",
            "none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt",
            "no-new-privileges=true",
            "--pids-limit",
            "64",
            "--memory",
            "1g" if selected_image == self.settings.triton_image else "512m",
        ]
        if self._owner_label is not None:
            args[args.index("--network") : args.index("--network")] = [
                "--label",
                self._owner_label,
            ]
        if gpu:
            args.extend(["--gpus", "device=0"])
        args.extend(["--entrypoint", command[0], selected_image, *command[1:]])
        return self._run_container(
            args,
            name,
            timeout=timeout,
            limit=32_768,
            platform_owned=platform_owned,
        )

    def _inspect_image_digest(self, image: str) -> str | None:
        inspect = self._run_limited(
            [
                self._docker,
                "image",
                "inspect",
                "--format",
                "{{json .RepoDigests}}",
                image,
            ],
            timeout=15,
            limit=16_384,
        )
        if inspect.returncode != 0:
            raise RunnerUnavailable(
                f"fixed PyTorch/Triton image is unavailable: {self._clean_output(inspect.output)}"
            )
        digests = json.loads(inspect.output.strip() or "[]")
        if isinstance(digests, list) and digests:
            return str(digests[0])
        if "@sha256:" in image:
            return image
        return None

    def _image_runtime_version(self) -> str | None:
        result = self._docker_probe(["cat", "/usr/local/cuda/version.json"], gpu=False, timeout=10)
        if result.returncode != 0:
            match = re.search(r"cuda:(\d+\.\d+(?:\.\d+)?)", self.settings.cuda_image)
            return match.group(1) if match else None
        try:
            payload = json.loads(result.output)
            cuda = payload.get("cuda", payload)
            return str(cuda.get("version")) if cuda.get("version") else None
        except (ValueError, AttributeError):
            return None

    def _probe_telemetry(
        self,
        *,
        image: str | None = None,
        platform_owned: bool = True,
    ) -> dict[str, str | None]:
        names = ["temperature_c", "power_w", "sm_clock_mhz", "gpu_busy_percent"]
        result = self._docker_probe(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            gpu=True,
            timeout=15,
            image=image,
            platform_owned=platform_owned,
        )
        if result.returncode != 0 or not result.output.strip():
            return dict.fromkeys(names)
        values = [part.strip() for part in result.output.splitlines()[0].split(",")]
        return {
            name: (
                None
                if index >= len(values) or values[index] in {"N/A", "[N/A]", ""}
                else values[index]
            )
            for index, name in enumerate(names)
        }

    def _run_container(
        self,
        args: Sequence[str],
        name: str,
        *,
        timeout: float,
        limit: int | None = None,
        platform_owned: bool = False,
    ) -> CommandResult:
        result = self._run_limited(
            args, timeout=timeout, limit=limit or self.settings.output_limit_bytes
        )
        if result.timed_out or result.output_limited:
            subprocess.run(
                [self._docker, "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        if (
            platform_owned
            and result.returncode == 125
            and not result.timed_out
            and not result.output_limited
        ):
            reason = self._clean_output(result.output) or "Docker could not start the container"
            self.mark_unhealthy(reason)
            raise RunnerUnavailable(f"container runtime failed: {reason}")
        return result

    @staticmethod
    def _run_limited(args: Sequence[str], *, timeout: float, limit: int) -> CommandResult:
        start = time.monotonic()
        process = subprocess.Popen(
            list(args),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
        )
        chunks: queue.Queue[bytes | None] = queue.Queue(maxsize=16)

        def read_output() -> None:
            assert process.stdout is not None
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                chunks.put(chunk)
            chunks.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        collected = bytearray()
        timed_out = False
        output_limited = False
        reader_done = False
        while process.poll() is None or not reader_done:
            try:
                chunk = chunks.get(timeout=0.02)
                if chunk is None:
                    reader_done = True
                elif len(collected) + len(chunk) > limit:
                    remaining = max(0, limit - len(collected))
                    collected.extend(chunk[:remaining])
                    output_limited = True
                    break
                else:
                    collected.extend(chunk)
            except queue.Empty:
                pass
            if time.monotonic() - start > timeout:
                timed_out = True
                break
        if timed_out or output_limited:
            process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        drain_deadline = time.monotonic() + 2
        while not reader_done and time.monotonic() < drain_deadline:
            try:
                chunk = chunks.get(timeout=0.02)
            except queue.Empty:
                continue
            if chunk is None:
                reader_done = True
            elif len(collected) < limit:
                collected.extend(chunk[: limit - len(collected)])
        reader.join(timeout=0.2)
        suffix = b""
        if timed_out:
            suffix = b"\n[platform] process timed out"
        elif output_limited:
            suffix = b"\n[platform] output limit exceeded"
        output = (bytes(collected) + suffix).decode("utf-8", errors="replace")
        return CommandResult(
            args=tuple(args),
            returncode=process.returncode if process.returncode is not None else -9,
            output=output,
            duration_seconds=time.monotonic() - start,
            timed_out=timed_out,
            output_limited=output_limited,
        )

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
