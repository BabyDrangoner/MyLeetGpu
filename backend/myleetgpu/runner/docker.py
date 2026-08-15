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
    RunnerUnavailable,
    RunnerUnhealthy,
)

RESULT_PREFIX = "MYLEETGPU_RESULT="
CONTAINER_USER = "65534:65534"
CONTAINER_WORKDIR = "/work"
CONTAINER_FILE_BYTES = 64 * 1024 * 1024
RUNNER_LABEL = "com.myleetgpu.runner=true"
INSTALLATION_LABEL_KEY = "com.myleetgpu.installation"
OWNER_LABEL_KEY = "com.myleetgpu.owner"


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", value)[:80]


class DockerRunner:
    """The only adapter allowed to translate platform operations into Docker arguments."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._cached_probe: tuple[float, EnvironmentProbe] | None = None
        self._health_file = settings.data_dir / "runner-unhealthy.json"
        self._docker = os.environ.get("MYLEETGPU_DOCKER_BIN", "docker")
        installation = stable_hash(str(settings.resolved_host_data_dir.resolve()))[:16]
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
        if self._cached_probe and not force and time.monotonic() - self._cached_probe[0] < 60:
            return self._cached_probe[1]

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
            )
        self._cached_probe = (time.monotonic(), probe)
        return probe

    def prepare_compile(
        self, task_root: Path, problem: Problem, source_path: Path, harness_kind: str
    ) -> Path:
        if harness_kind not in {"validator", "benchmark"}:
            raise ValueError("unknown harness kind")
        compile_dir = task_root / f"compile-{harness_kind}"
        compile_dir.mkdir(parents=True, exist_ok=False)
        source_target = compile_dir / "source.cu"
        header_target = compile_dir / "solve.h"
        harness_target = compile_dir / "platform.cu"
        shutil.copyfile(source_path, source_target)
        shutil.copyfile(problem.header_path, header_target)
        shutil.copyfile(
            problem.validator_path if harness_kind == "validator" else problem.benchmark_path,
            harness_target,
        )
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
    ) -> CompileResult:
        probe = self.probe_environment()
        if not probe.healthy or not probe.cuda_arch:
            raise RunnerUnavailable(probe.error or "CUDA environment is unavailable")
        compile_dir = self.prepare_compile(task_root, problem, source_path, harness_kind)
        container_name = _safe_name(f"myleetgpu-{task_root.name}-compile-{harness_kind}")
        command = [
            *self._base_container_args(container_name, compile_dir, gpu=False),
            "--entrypoint",
            "nvcc",
            self.settings.cuda_image,
            *self.effective_compile_flags(problem, probe),
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

    @staticmethod
    def effective_compile_flags(problem: Problem, probe: EnvironmentProbe) -> list[str]:
        if not probe.cuda_arch:
            raise RunnerUnavailable("CUDA architecture was not detected")
        return [*problem.compile_flags, f"-arch=sm_{probe.cuda_arch}"]

    def execute(
        self,
        task_root: Path,
        executable: Path,
        *,
        mode: str,
        timeout_seconds: float,
    ) -> ExecutionResult:
        if mode not in {"public", "full", "benchmark"}:
            raise ValueError("unknown execution mode")
        self.assert_healthy()
        run_dir = task_root / f"run-{mode}"
        run_dir.mkdir(parents=True, exist_ok=False)
        run_target = run_dir / "program"
        shutil.copyfile(executable, run_target)
        ensure_mode(run_target, 0o555)
        # The bind mount is already readonly inside the submitted-code
        # container. Keep the host owner write bit so a non-root Worker can
        # unlink the executable when the job spool is cleaned.
        ensure_mode(run_dir, 0o755)
        container_name = _safe_name(f"myleetgpu-{task_root.name}-run-{mode}")
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
        output = self._clean_output(result.output)
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

    def _base_container_args(self, name: str, task_dir: Path, *, gpu: bool) -> list[str]:
        host_path = self._host_task_path(task_dir)
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
            "/tmp:rw,nosuid,nodev,noexec,size=64m",
            "--env",
            "HOME=/tmp",
            "--env",
            "CUDA_CACHE_DISABLE=1",
            "--mount",
            (
                f"type=bind,src={host_path},dst={CONTAINER_WORKDIR},readonly"
                if gpu
                else f"type=bind,src={host_path},dst={CONTAINER_WORKDIR}"
            ),
            "--workdir",
            CONTAINER_WORKDIR,
            "--stop-timeout",
            "1",
        ]
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
        host = (self.settings.resolved_host_data_dir / relative).resolve()
        host_data = self.settings.resolved_host_data_dir
        if host_data not in host.parents:
            raise ValueError("resolved host mount escaped the configured data directory")
        return host.as_posix()

    def _docker_probe(self, command: Sequence[str], *, gpu: bool, timeout: float) -> CommandResult:
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
            "512m",
        ]
        if self._owner_label is not None:
            args[args.index("--network") : args.index("--network")] = [
                "--label",
                self._owner_label,
            ]
        if gpu:
            args.extend(["--gpus", "device=0"])
        args.extend(["--entrypoint", command[0], self.settings.cuda_image, *command[1:]])
        return self._run_container(
            args,
            name,
            timeout=timeout,
            limit=32_768,
            platform_owned=True,
        )

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

    def _probe_telemetry(self) -> dict[str, str | None]:
        names = ["temperature_c", "power_w", "sm_clock_mhz", "gpu_busy_percent"]
        result = self._docker_probe(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            gpu=True,
            timeout=15,
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
