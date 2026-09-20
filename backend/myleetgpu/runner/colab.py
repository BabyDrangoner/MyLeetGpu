from __future__ import annotations

import contextlib
import json
import math
import os
import re
import selectors
import shlex
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from myleetgpu.config import Settings
from myleetgpu.domain.benchmark import stable_hash
from myleetgpu.domain.problems import Problem
from myleetgpu.runner import colab_bridge
from myleetgpu.runner.base import CUDA_CPP, TORCH_PYTHON, TRITON_PYTHON, BaseRunner
from myleetgpu.runner.models import (
    CommandResult,
    CompileResult,
    EnvironmentProbe,
    ExecutionResult,
    RunnerLanguage,
    RunnerUnavailable,
    RunnerUnhealthy,
)
from myleetgpu.runner.torch_submission_policy import submission_contract_from_declaration


class ColabRunner(BaseRunner):
    """Run trusted submissions in an existing SSH-enabled Colab VM.

    No Docker, Colab provisioning, OAuth flow, installation, or arbitrary client
    SSH configuration is invoked. The native process supervisor is deliberately
    not advertised as a security sandbox.
    """

    def __init__(self, settings: Settings):
        super().__init__(settings, health_filename="colab-runner-unhealthy.json")
        self._installation = stable_hash(str(settings.data_dir.resolve()))[:16]
        self._owner = stable_hash(f"colab-{uuid.uuid4()}")[:16]
        self._assigned_owner = False
        self._bridge_code = Path(colab_bridge.__file__).read_text(encoding="utf-8")

    def assign_owner(self, owner: str) -> None:
        self._owner = stable_hash(owner)[:16]
        self._assigned_owner = True

    def assert_healthy(self) -> None:
        if self._health_file.exists():
            raise RunnerUnhealthy(
                "Colab runner is marked unhealthy; check the existing runtime connection "
                "and use the environment connection test before resubmitting."
            )

    def recover(self, *, language: RunnerLanguage | str = CUDA_CPP) -> EnvironmentProbe:
        selected = self._resolve_language(language)
        # Manual recovery is serialized with jobs by Worker. Do not resume GPU
        # measurements while this installation may have a disconnected orphan.
        self._removed_tasks(self._rpc("cleanup_all"))
        probe = self._probe(selected, force=True, ignore_circuit_breaker=True)
        if not probe.healthy:
            raise RunnerUnavailable(probe.error or "Colab environment recovery probe failed")
        self._health_file.unlink(missing_ok=True)
        return probe

    @staticmethod
    def _transport(
        args: list[str], *, timeout: float, limit: int, input_data: bytes = b""
    ) -> CommandResult:
        """Bound even SSH/ProxyCommand output; a lost connection cannot hang Worker."""
        started = time.monotonic()
        output = bytearray()
        timed_out = limited = False
        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            start_new_session=True,
        )

        def write_input() -> None:
            with contextlib.suppress(OSError, ValueError):
                assert process.stdin is not None
                process.stdin.write(input_data)
                process.stdin.close()

        writer = threading.Thread(target=write_input, daemon=True)
        writer.start()
        try:
            assert process.stdout is not None
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    if time.monotonic() - started >= timeout:
                        timed_out = True
                        break
                    for key, _ in selector.select(0.05):
                        chunk = os.read(key.fileobj.fileno(), 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        remaining = limit - len(output)
                        output.extend(chunk[:remaining])
                        if len(chunk) > remaining:
                            limited = True
                            break
                    if limited:
                        break
                    if process.poll() is not None and not selector.select(0):
                        break
            if not timed_out and not limited:
                try:
                    process.wait(timeout=max(0.01, timeout - time.monotonic() + started))
                except subprocess.TimeoutExpired:
                    timed_out = True
        finally:
            # Never terminate a shared SSH ControlMaster: only this invocation's
            # newly created process group is ours.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
            writer.join(timeout=1)
            if process.stdout is not None:
                process.stdout.close()
        return CommandResult(
            tuple(args),
            process.returncode,
            output.decode("utf-8", errors="replace"),
            time.monotonic() - started,
            timed_out,
            limited,
        )

    def _assert_existing_session(self) -> None:
        # colab ssh may implicitly allocate a missing session. The read-only
        # status operation must positively identify a live GPU session first.
        result = self._transport(
            [
                self.settings.colab_cli_bin,
                "--auth",
                "oauth2",
                "status",
                "-s",
                self.settings.colab_session,
            ],
            timeout=self.settings.colab_connect_timeout_seconds,
            limit=16_384,
        )
        session = re.escape(self.settings.colab_session)
        allowed = re.compile(
            rf"^\[{session}\] [^\r\n]+ \| Hardware: [^\r\n]+ \| Shape: [^\r\n]+"
            rf" \| Variant: GPU \| Status: (?:IDLE|BUSY(?: \([^\r\n]*\))?)$",
            re.MULTILINE,
        )
        if (
            result.returncode != 0
            or result.timed_out
            or result.output_limited
            or allowed.search(result.output) is None
        ):
            # Status can contain notebook metadata: never echo it to API logs.
            raise RunnerUnavailable(
                "Cannot confirm an existing Colab GPU session. Check `colab --auth oauth2 "
                f"status -s {self.settings.colab_session}` and sign in locally if needed. "
                "No runtime was created and SSH was not started."
            )

    def _assert_control_master(self) -> None:
        result = self._transport(
            [
                self.settings.colab_ssh_bin,
                "-O",
                "check",
                "-o",
                "BatchMode=yes",
                "-o",
                "ProxyCommand=false",
                "-o",
                "ControlMaster=no",
                self.settings.colab_ssh_host,
            ],
            timeout=self.settings.colab_connect_timeout_seconds,
            limit=4096,
        )
        if result.returncode != 0 or result.timed_out or result.output_limited:
            raise RunnerUnavailable(
                "No reusable Colab SSH ControlMaster is available on the Worker host. "
                "Confirm the existing session, then connect with "
                f"`ssh {self.settings.colab_ssh_host} true` locally. The application only "
                "reuses an established connection and never opens a new Colab bridge."
            )

    def _rpc(self, action: str, *, rpc_timeout: float = 30, **payload: Any) -> dict[str, Any]:
        try:
            self._assert_existing_session()
            self._assert_control_master()
            request = {
                "version": 1,
                "action": action,
                "root": self.settings.colab_remote_root,
                "installation": self._installation,
                "owner": self._owner,
                "limit": self.settings.output_limit_bytes,
                **payload,
            }
            encoded = json.dumps(request, ensure_ascii=True, allow_nan=False).encode("utf-8")
            if len(encoded) > colab_bridge.MAX_REQUEST_BYTES:
                raise RunnerUnavailable("Colab task files exceed the bounded transfer limit")
            remote = shlex.join(
                [self.settings.colab_python_bin, "-I", "-B", "-c", self._bridge_code]
            )
            command = [
                self.settings.colab_ssh_bin,
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "ControlMaster=no",
                "-o",
                "ProxyCommand=false",
                "-o",
                f"ConnectTimeout={self.settings.colab_connect_timeout_seconds}",
                "-o",
                "ConnectionAttempts=1",
                "-o",
                "ServerAliveInterval=10",
                "-o",
                "ServerAliveCountMax=2",
                self.settings.colab_ssh_host,
                remote,
            ]
            result = self._transport(
                command,
                timeout=rpc_timeout + self.settings.colab_connect_timeout_seconds + 10,
                limit=8 * self.settings.output_limit_bytes + 65_536,
                input_data=encoded,
            )
            if result.returncode != 0 or result.timed_out or result.output_limited:
                if "404" in result.output or "SSH not exposed" in result.output:
                    raise RunnerUnavailable(
                        "The existing Colab runtime does not expose SSH (HTTP 404). "
                        "Preserve /content work and explicitly recreate an SSH-enabled runtime "
                        "outside this application; no runtime was replaced."
                    )
                if "429" in result.output:
                    raise RunnerUnavailable(
                        "Colab SSH endpoint is busy (HTTP 429). Check the configured "
                        "ControlMaster or reconnect its existing VS Code client."
                    )
                if "401" in result.output or "Permission denied" in result.output:
                    raise RunnerUnavailable(
                        "Colab SSH authentication failed. Sign in using the configured Colab "
                        "CLI locally and check the dedicated SSH key."
                    )
                reason = "timed out" if result.timed_out else "failed"
                # An SSH ProxyCommand may print OAuth URLs or one-time codes.
                # Do not echo arbitrary CLI diagnostics into the web API.
                raise RunnerUnavailable(
                    f"Colab SSH {reason}. Check the existing runtime and configured SSH alias "
                    "locally; the application does not start or replace Colab compute."
                )
            lines = [
                line[len(colab_bridge.RPC_PREFIX) :]
                for line in result.output.splitlines()
                if line.startswith(colab_bridge.RPC_PREFIX)
            ]
            if len(lines) != 1:
                raise RunnerUnavailable("Colab returned an invalid RPC response")
            response = json.loads(lines[0])
            if not isinstance(response, dict) or response.get("version") != 1:
                raise RunnerUnavailable("Colab returned an unsupported RPC response")
            if response.get("ok") is not True:
                error = response.get("error")
                detail = error[:2000] if isinstance(error, str) else "unknown remote error"
                raise RunnerUnavailable(f"Colab runner: {detail}")
            value = response.get("result")
            if not isinstance(value, dict):
                raise RunnerUnavailable("Colab returned an invalid result payload")
            return value
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise RunnerUnavailable(f"Colab connection unavailable: {error}") from error

    def _probe(
        self, language: RunnerLanguage, *, force: bool, ignore_circuit_breaker: bool
    ) -> EnvironmentProbe:
        if not ignore_circuit_breaker:
            self.assert_healthy()
        cached = self._cached_probes.get(language)
        if cached and not force and time.monotonic() - cached[0] < 60:
            return cached[1]
        toolchain: dict[str, Any] = {"execution_target": "colab", "isolation": "trusted-native"}
        try:
            data = self._rpc("probe", rpc_timeout=120, language=language)
            for key in ("gpu_name", "compute_capability", "driver_version"):
                if not isinstance(data.get(key), str) or not data[key]:
                    raise RunnerUnavailable(f"Colab probe is missing {key}")
            if not re.fullmatch(r"\d+\.\d+", data["compute_capability"]):
                raise RunnerUnavailable("Colab returned an invalid compute capability")
            versions = data.get("toolchain")
            required = (
                {"nvcc_version", "cuda_runtime_version"}
                if language == CUDA_CPP
                else {"python_version", "torch_version", "torch_cuda_version"}
            )
            if language == TRITON_PYTHON:
                required.add("triton_version")
            if not isinstance(versions, dict) or any(
                not isinstance(versions.get(key), str) or not versions[key] for key in required
            ):
                raise RunnerUnavailable("Colab probe returned incomplete toolchain versions")
            toolchain.update({key: versions[key] for key in sorted(required)})
            cuda_arch = (
                data["compute_capability"].replace(".", "")
                if self.settings.cuda_arch == "auto"
                else self.settings.cuda_arch
            )
            telemetry = data.get("telemetry")
            if not isinstance(telemetry, dict):
                telemetry = {}
            probe = EnvironmentProbe(
                healthy=True,
                gpu_name=data["gpu_name"],
                compute_capability=data["compute_capability"],
                driver_version=data["driver_version"],
                cuda_runtime_version=versions.get("cuda_runtime_version")
                or versions.get("torch_cuda_version"),
                nvcc_version=versions.get("nvcc_version"),
                cuda_image="colab-native",
                image_digest=None,
                cuda_arch=cuda_arch,
                telemetry=telemetry,
                fingerprint=stable_hash(
                    {
                        "execution_target": "colab",
                        "backend": language,
                        "gpu": data["gpu_name"],
                        "arch": cuda_arch,
                        "driver": data["driver_version"],
                        "toolchain": toolchain,
                    }
                ),
                backend=language,
                toolchain=toolchain,
            )
        except RunnerUnavailable as error:
            probe = EnvironmentProbe(
                healthy=False,
                gpu_name=None,
                compute_capability=None,
                driver_version=None,
                cuda_runtime_version=None,
                nvcc_version=None,
                cuda_image="colab-native",
                image_digest=None,
                cuda_arch=None,
                telemetry={},
                error=str(error),
                fingerprint=stable_hash(
                    {
                        "execution_target": "colab",
                        "backend": language,
                        "healthy": False,
                        "error": str(error),
                    }
                ),
                backend=language,
                toolchain=toolchain,
            )
        self._cached_probes[language] = time.monotonic(), probe
        return probe

    def probe_environment(
        self, *, force: bool = False, ignore_circuit_breaker: bool = False
    ) -> EnvironmentProbe:
        return self._probe(CUDA_CPP, force=force, ignore_circuit_breaker=ignore_circuit_breaker)

    def probe_triton_environment(
        self, *, force: bool = False, ignore_circuit_breaker: bool = False
    ) -> EnvironmentProbe:
        return self._probe(
            TRITON_PYTHON, force=force, ignore_circuit_breaker=ignore_circuit_breaker
        )

    def probe_torch_environment(
        self, *, force: bool = False, ignore_circuit_breaker: bool = False
    ) -> EnvironmentProbe:
        return self._probe(TORCH_PYTHON, force=force, ignore_circuit_breaker=ignore_circuit_breaker)

    def _task_id(self, task_root: Path) -> str:
        root = task_root.resolve()
        spool = self.settings.jobs_dir.resolve()
        if root.parent == spool:
            return colab_bridge.check_uuid(root.name)
        if root.parent.parent == spool and re.fullmatch(r"version-[1-9]\d{0,5}", root.name):
            return colab_bridge.check_uuid(root.parent.name)
        raise ValueError("Colab task must be inside the local job spool")

    def _subtask(self, task_root: Path) -> str | None:
        self._task_id(task_root)
        if task_root.resolve().parent != self.settings.jobs_dir.resolve():
            return task_root.name
        return None

    @staticmethod
    def _command_result(value: dict[str, Any], *, limit: int) -> CommandResult:
        returncode, duration = value.get("returncode"), value.get("duration_seconds")
        if (
            isinstance(returncode, bool)
            or not isinstance(returncode, int)
            or isinstance(duration, bool)
            or not isinstance(duration, int | float)
            or not math.isfinite(duration)
            or duration < 0
            or not isinstance(value.get("output"), str)
            or not isinstance(value.get("timed_out"), bool)
            or not isinstance(value.get("output_limited"), bool)
        ):
            raise RunnerUnavailable("Colab returned malformed command metadata")
        if len(value["output"].encode("utf-8")) > 3 * limit:
            raise RunnerUnavailable("Colab response exceeded the output limit")
        return CommandResult(
            (), returncode, value["output"], duration, value["timed_out"], value["output_limited"]
        )

    def _remote_diagnostics(self, output: str, task_id: str) -> str:
        remote = f"{self.settings.colab_remote_root}/{self._installation}/{task_id}"
        return self._clean_output(output.replace(remote, "/work"))

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
        self.assert_healthy()
        task_id = self._task_id(task_root)
        selected = self._resolve_language(language, implementation=implementation)
        probe = self._probe(selected, force=False, ignore_circuit_breaker=False)
        if not probe.healthy:
            raise RunnerUnavailable(probe.error or "Colab environment unavailable")
        compile_dir = self.prepare_compile(
            task_root,
            problem,
            source_path,
            harness_kind,
            language=selected,
            implementation=implementation,
        )
        timeout = min(
            self.settings.compile_timeout_seconds, problem.manifest.timeouts.compile_ms / 1000
        )
        extra: dict[str, Any] = {}
        if selected == CUDA_CPP:
            extra["flags"] = self.effective_compile_flags(
                problem, probe, language=selected, implementation=implementation
            )
        elif selected == TORCH_PYTHON:
            owner = (
                implementation
                if implementation is not None
                else problem.get_implementation(TORCH_PYTHON)
            )
            extra["contract"] = submission_contract_from_declaration(
                owner.signature.symbol, owner.signature.declaration
            )
        value = self._rpc(
            "compile",
            rpc_timeout=timeout + 5,
            task=task_id,
            subtask=self._subtask(task_root),
            stage=compile_dir.name,
            language=selected,
            files={path.name: path.read_text(encoding="utf-8") for path in compile_dir.iterdir()},
            timeout=timeout,
            **extra,
        )
        result = self._command_result(value, limit=self.settings.output_limit_bytes)
        if not isinstance(value.get("artifact_exists"), bool):
            raise RunnerUnavailable("Colab returned malformed compilation artifact metadata")
        succeeded = (
            result.returncode == 0
            and value["artifact_exists"]
            and not result.timed_out
            and not result.output_limited
        )
        artifact = compile_dir / ("program" if selected == CUDA_CPP else "source.py")
        if succeeded and selected == CUDA_CPP:
            # An opaque marker, never a locally executed or downloaded GPU binary.
            artifact.write_text(
                json.dumps(
                    {"execution_target": "colab", "task": task_id, "stage": compile_dir.name}
                ),
                encoding="utf-8",
            )
        return CompileResult(
            succeeded=succeeded,
            diagnostics=self._remote_diagnostics(result.output, task_id),
            executable=artifact if succeeded else None,
            duration_seconds=result.duration_seconds,
            timed_out=result.timed_out,
            output_limited=result.output_limited,
        )

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
        self.assert_healthy()
        task_id = self._task_id(task_root)
        selected = self._resolve_language(language, artifact=executable)
        if (
            executable.parent.parent.resolve() != task_root.resolve()
            or executable.parent.name not in colab_bridge.STAGES
            or not executable.is_file()
        ):
            raise ValueError("Colab compilation artifact is missing or outside this task")
        value = self._rpc(
            "execute",
            rpc_timeout=timeout_seconds + 5,
            task=task_id,
            subtask=self._subtask(task_root),
            stage=executable.parent.name,
            language=selected,
            mode=mode,
            timeout=timeout_seconds,
        )
        result = self._command_result(value, limit=self.settings.output_limit_bytes)
        output = self._remote_diagnostics(result.output, task_id)
        parsed = self._parse_result(output)
        succeeded = (
            result.returncode == 0
            and parsed is not None
            and parsed.get("status") == "passed"
            and not result.timed_out
            and not result.output_limited
        )
        if result.returncode != 0 and self._looks_like_gpu_health_failure(output):
            probe = self._probe(selected, force=True, ignore_circuit_breaker=True)
            if not probe.healthy:
                self.mark_unhealthy(probe.error or "Colab GPU probe failed after execution")
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
        if task_root.resolve().parent != self.settings.jobs_dir.resolve():
            raise ValueError("Colab cleanup requires a top-level UUID job directory")
        task_id = self._task_id(task_root)
        try:
            self._rpc("cleanup", task=task_id)
        finally:
            super().cleanup_task(task_root)

    def cleanup_orphan_containers(self) -> list[str]:
        """Compatibility name: clean this installation's native task directories."""
        try:
            value = self._rpc("cleanup_all")
        except RunnerUnavailable:
            return []
        return self._removed_tasks(value)

    def cleanup_owned_containers(self) -> list[str]:
        if not self._assigned_owner:
            return []
        try:
            value = self._rpc("cleanup_owner")
        except RunnerUnavailable:
            return []
        return self._removed_tasks(value)

    @staticmethod
    def _removed_tasks(value: dict[str, Any]) -> list[str]:
        removed = value.get("removed")
        if not isinstance(removed, list) or any(not isinstance(item, str) for item in removed):
            raise RunnerUnavailable("Colab returned malformed cleanup metadata")
        return removed
