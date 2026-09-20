"""CPU C++/stdlib-Python adapter for trusted, single-user local development.

No Docker, SSH, GPU probing, or package installation is performed. A clean
environment, process limits, bounded output and deadlines reduce accidental
damage, but native submissions retain the local user's filesystem privileges.
"""

from __future__ import annotations

import contextlib
import os
import platform
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from myleetgpu.config import Settings
from myleetgpu.domain.benchmark import stable_hash
from myleetgpu.domain.problems import Problem
from myleetgpu.runner.base import BaseRunner
from myleetgpu.runner.models import (
    CommandResult,
    CompileResult,
    EnvironmentProbe,
    ExecutionResult,
    RunnerLanguage,
    RunnerUnavailable,
    RunnerUnhealthy,
)

CPU_LANGUAGES = frozenset({"cpp", "python"})


@lru_cache(maxsize=1)
def _cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            result = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        elif sys.platform.startswith("linux"):
            with Path("/proc/cpuinfo").open(encoding="utf-8") as cpuinfo:
                for line in cpuinfo:
                    if line.startswith("model name"):
                        return line.partition(":")[2].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


class CpuRunner(BaseRunner):
    def __init__(self, settings: Settings):
        super().__init__(settings, health_filename="cpu-cpp-runner-unhealthy.json")
        self._language = "cpp"
        self._owner = "cpu"
        self._processes: dict[int, subprocess.Popen[bytes]] = {}
        self._process_lock = threading.Lock()

    @staticmethod
    def _cpu_language(language: str | None, implementation: Any | None = None) -> str:
        selected = str(language or getattr(implementation, "language", "cpp"))
        if selected not in CPU_LANGUAGES:
            raise RunnerUnavailable("CPU runner only supports ordinary C++ and Python")
        return selected

    def _select(self, language: str) -> None:
        self._language = self._cpu_language(language)
        self._health_file = self.settings.data_dir / f"cpu-{self._language}-runner-unhealthy.json"

    def assign_owner(self, owner: str) -> None:
        self._owner = stable_hash(owner)[:16]

    def assert_healthy(self) -> None:
        if self._health_file.exists():
            raise RunnerUnhealthy(
                f"{self._language} CPU runner is unavailable; test its CPU connection to recover"
            )

    def mark_unhealthy(self, reason: str) -> None:
        super().mark_unhealthy(reason)
        self._cached_probes.pop(self._language, None)

    def _compiler(self) -> str:
        candidate = os.environ.get("MYLEETGPU_CXX_BIN") or self.settings.cxx_bin
        resolved = shutil.which(candidate) if candidate else None
        if not resolved:
            raise RunnerUnavailable(
                "C++ compiler not found; install C++17 or set MYLEETGPU_CXX_BIN"
            )
        return resolved

    @staticmethod
    def _environment(directory: Path) -> dict[str, str]:
        temporary = directory / "tmp"
        temporary.mkdir(exist_ok=True)
        environment = {
            "PATH": os.defpath,
            "HOME": str(directory),
            "TMPDIR": str(temporary),
            "TMP": str(temporary),
            "TEMP": str(temporary),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONUNBUFFERED": "1",
            "MYLEETGPU_SOURCE_PATH": str(directory / "source.py"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        for name in ("SDKROOT", "DEVELOPER_DIR", "SYSTEMROOT"):
            if name in os.environ:
                environment[name] = os.environ[name]
        return environment

    def _run_limited(
        self, args: list[str], *, cwd: Path, timeout: float, limit: int
    ) -> CommandResult:
        started = time.monotonic()
        output = bytearray()
        timed_out = output_limited = False
        wrapper = Path(__file__).with_name("cpu_process.py")
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", str(wrapper), str(timeout), str(os.getpid()), *args],
            cwd=cwd,
            env=self._environment(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with self._process_lock:
            self._processes[process.pid] = process
        try:
            assert process.stdout is not None
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        timed_out = True
                        break
                    for key, _ in selector.select(min(0.05, remaining)):
                        chunk = os.read(key.fileobj.fileno(), 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        space = max(0, limit - len(output))
                        output.extend(chunk[:space])
                        if len(chunk) > space:
                            output_limited = True
                            break
                    if output_limited:
                        break
            if not timed_out and not output_limited:
                try:
                    process.wait(timeout=max(0.01, timeout - time.monotonic() + started))
                except subprocess.TimeoutExpired:
                    timed_out = True
        finally:
            # Kill only the process group created by this invocation, including children.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
            if process.stdout is not None:
                process.stdout.close()
            with self._process_lock:
                self._processes.pop(process.pid, None)
        return CommandResult(
            tuple(args),
            process.returncode,
            output.decode("utf-8", errors="replace"),
            time.monotonic() - started,
            timed_out,
            output_limited,
        )

    def probe_cpu_environment(
        self, language: str = "cpp", *, force: bool = False, ignore_circuit_breaker: bool = False
    ) -> EnvironmentProbe:
        selected = self._cpu_language(language)
        self._select(selected)
        cached = self._cached_probes.get(selected)
        if cached and not force and time.monotonic() - cached[0] < 60:
            if not ignore_circuit_breaker:
                self.assert_healthy()
            return cached[1]
        toolchain: dict[str, Any] = {
            "execution_target": "cpu",
            "isolation": "trusted-native",
            "cpu_name": _cpu_name(),
            "platform": f"{platform.system()} {platform.release()}",
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
        }
        error = None
        try:
            if not ignore_circuit_breaker:
                self.assert_healthy()
            self.settings.ensure_directories()
            with tempfile.TemporaryDirectory(
                prefix="cpu-probe-", dir=self.settings.data_dir
            ) as temp:
                directory = Path(temp).resolve()
                if selected == "cpp":
                    compiler = self._compiler()
                    version = self._run_limited(
                        [compiler, "--version"], cwd=directory, timeout=10, limit=8192
                    )
                    if version.returncode != 0 or version.timed_out or version.output_limited:
                        raise RunnerUnavailable("Unable to identify the C++ compiler")
                    toolchain["compiler_version"] = version.output.splitlines()[0]
                    toolchain["compiler"] = compiler
                    (directory / "probe.cpp").write_text(
                        "#include <cmath>\nint main(){return std::abs(std::exp(0.0)-1.0)>1e-12;}\n",
                        encoding="utf-8",
                    )
                    compiled = self._run_limited(
                        [compiler, "-std=c++17", "-O2", "probe.cpp", "-o", "probe"],
                        cwd=directory,
                        timeout=30,
                        limit=8192,
                    )
                    if compiled.returncode != 0 or compiled.timed_out or compiled.output_limited:
                        raise RunnerUnavailable(
                            "C++17 smoke compilation failed: " + compiled.output
                        )
                    command = [str(directory / "probe")]
                else:
                    toolchain["runtime_profile"] = "python_cpu_v1"
                    command = [
                        sys.executable,
                        "-I",
                        "-S",
                        "-c",
                        "import math; assert math.exp(0.0) == 1.0",
                    ]
                result = self._run_limited(command, cwd=directory, timeout=5, limit=8192)
                if result.returncode != 0 or result.timed_out or result.output_limited:
                    raise RunnerUnavailable("CPU runtime smoke check failed")
        except (OSError, ValueError, RunnerUnavailable, RunnerUnhealthy) as failure:
            error = self._clean_output(str(failure))
        probe = EnvironmentProbe(
            healthy=error is None,
            gpu_name=None,
            compute_capability=None,
            driver_version=None,
            cuda_runtime_version=None,
            nvcc_version=None,
            cuda_image="native-cpu",
            image_digest=None,
            cuda_arch=None,
            backend=selected,
            toolchain=toolchain,
            error=error,
            fingerprint=stable_hash({"backend": selected, "toolchain": toolchain}),
        )
        self._cached_probes[selected] = (time.monotonic(), probe)
        return probe

    def probe_environment(
        self, *, force: bool = False, ignore_circuit_breaker: bool = False
    ) -> EnvironmentProbe:
        return self.probe_cpu_environment(
            "cpp", force=force, ignore_circuit_breaker=ignore_circuit_breaker
        )

    def probe_triton_environment(self, **kwargs: Any) -> EnvironmentProbe:
        raise RunnerUnavailable("Triton requires a GPU runner")

    def probe_torch_environment(self, **kwargs: Any) -> EnvironmentProbe:
        raise RunnerUnavailable("PyTorch GPU problems require a GPU runner")

    def recover(self, *, language: str = "cpp") -> EnvironmentProbe:
        probe = self.probe_cpu_environment(language, force=True, ignore_circuit_breaker=True)
        if not probe.healthy:
            raise RunnerUnavailable(probe.error or "CPU environment is unavailable")
        self._health_file.unlink(missing_ok=True)
        return probe

    def effective_compile_flags(
        self,
        problem: Problem,
        probe: EnvironmentProbe,
        *,
        language: RunnerLanguage | str | None = None,
        implementation: Any | None = None,
    ) -> list[str]:
        selected = self._cpu_language(language, implementation)
        implementation = implementation or problem.get_implementation(selected)
        if selected == "cpp":
            return [
                *implementation.compile_flags,
                f"compiler={probe.toolchain.get('compiler_version', 'unknown')}",
            ]
        return [
            "backend=python",
            "profile=python_cpu_v1",
            "isolated-stdlib=true",
            f"python={probe.toolchain.get('python_version', 'unknown')}",
        ]

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
        selected = self._cpu_language(language, implementation)
        self._select(selected)
        self.assert_healthy()
        if harness_kind not in {"validator", "benchmark"}:
            raise ValueError("unknown harness kind")
        implementation = implementation or problem.get_implementation(selected)
        directory = task_root / f"compile-{harness_kind}"
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        suffix = ".cpp" if selected == "cpp" else ".py"
        shutil.copyfile(source_path, directory / f"source{suffix}")
        shutil.copyfile(
            self._harness_path(problem, implementation, harness_kind),
            directory / f"platform{suffix}",
        )
        reserved = {"source.cpp", "source.py", "platform.cpp", "platform.py", "program", "tmp"}
        if selected == "cpp":
            header = implementation.header_path
            if header.name in reserved:
                raise ValueError("CPU signature header collides with a reserved file name")
            shutil.copyfile(header, directory / header.name)
            reserved.add(header.name)
        for path in implementation.support_paths:
            if path.name in reserved:
                raise ValueError("CPU harness support file collides with reserved file name")
            shutil.copyfile(path, directory / path.name)
            reserved.add(path.name)
        if selected == "cpp":
            flags = implementation.compile_flags
            allowed = {"-std=c++17", "-std=c++20", "-O2", "-O3"}
            if set(flags) - allowed:
                raise ValueError("unsupported CPU compiler flags")
            executable = directory / "program"
            args = [
                self._compiler(),
                *flags,
                "-I",
                str(directory),
                "source.cpp",
                "platform.cpp",
                "-o",
                str(executable.resolve()),
            ]
        else:
            executable = directory / "platform.py"
            args = [sys.executable, "-I", "-S", "-m", "py_compile", "source.py", "platform.py"]
        result = self._run_limited(
            args,
            cwd=directory.resolve(),
            timeout=min(
                self.settings.compile_timeout_seconds, problem.manifest.timeouts.compile_ms / 1000
            ),
            limit=self.settings.output_limit_bytes,
        )
        success = (
            result.returncode == 0
            and not result.timed_out
            and not result.output_limited
            and executable.is_file()
        )
        return CompileResult(
            success,
            self._clean_diagnostics(result.output, directory),
            executable if success else None,
            result.duration_seconds,
            result.timed_out,
            result.output_limited,
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
        selected = self._cpu_language(language)
        self._select(selected)
        self.assert_healthy()
        if mode not in {"public", "full", "benchmark"}:
            raise ValueError("unknown execution mode")
        executable = executable.resolve()
        if not executable.is_relative_to(task_root.resolve()) or not executable.is_file():
            raise ValueError("CPU executable is not inside the job directory")
        if selected == "python":
            code = (
                "import runpy,sys; sys.path.insert(0, '.'); "
                "runpy.run_path('platform.py', run_name='__main__')"
            )
            args = [sys.executable, "-I", "-S", "-c", code, "--mode", mode]
        else:
            args = [str(executable), "--mode", mode]
        result = self._run_limited(
            args,
            cwd=executable.parent,
            timeout=timeout_seconds,
            limit=self.settings.output_limit_bytes,
        )
        parsed = self._parse_result(result.output)
        success = (
            result.returncode == 0
            and not result.timed_out
            and not result.output_limited
            and parsed is not None
            and parsed.get("status") == "passed"
        )
        return ExecutionResult(
            success,
            self._clean_output(result.output),
            parsed,
            result.duration_seconds,
            result.returncode,
            result.timed_out,
            result.output_limited,
        )

    def cleanup_orphan_containers(self) -> list[str]:
        # Never discover/kill unknown user processes on the host.
        return []

    def cleanup_owned_containers(self) -> list[str]:
        with self._process_lock:
            processes = list(self._processes.values())
        removed = []
        for process in processes:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                    removed.append(f"cpu-{self._owner}-{process.pid}")
        return removed
