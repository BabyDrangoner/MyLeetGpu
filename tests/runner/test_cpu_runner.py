from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
from myleetgpu.config import Settings
from myleetgpu.domain.problems import ProblemCatalog
from myleetgpu.runner.cpu import CpuRunner
from myleetgpu.runner.models import RunnerUnavailable, RunnerUnhealthy

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runner(tmp_path: Path) -> CpuRunner:
    settings = Settings(data_dir=tmp_path / "data", problems_dir=ROOT / "problems", _env_file=None)
    settings.ensure_directories()
    return CpuRunner(settings)


def test_python_probe_is_native_without_gpu(runner: CpuRunner) -> None:
    probe = runner.probe_cpu_environment("python")
    assert probe.healthy, probe.error
    assert probe.gpu_name is None and probe.cuda_arch is None
    assert probe.toolchain["execution_target"] == "cpu"
    assert probe.toolchain["isolation"] == "trusted-native"
    assert probe.toolchain["cpu_name"]


def test_bad_cpp_compiler_does_not_block_python(runner: CpuRunner, monkeypatch) -> None:
    monkeypatch.setenv("MYLEETGPU_CXX_BIN", "/does-not-exist/compiler")
    assert not runner.probe_cpu_environment("cpp").healthy
    assert runner.probe_cpu_environment("python").healthy


def test_cpp_compiler_can_be_configured_in_settings(runner: CpuRunner, monkeypatch) -> None:
    monkeypatch.delenv("MYLEETGPU_CXX_BIN", raising=False)
    runner.settings.cxx_bin = "/does-not-exist/configured-compiler"
    with pytest.raises(RunnerUnavailable, match="compiler not found"):
        runner._compiler()


def test_cpu_health_markers_are_language_scoped(runner: CpuRunner) -> None:
    runner._select("cpp")
    runner.mark_unhealthy("test cpp-only failure")
    with pytest.raises(RunnerUnhealthy):
        runner.assert_healthy()
    assert runner.probe_cpu_environment("python").healthy
    assert runner.settings.data_dir.joinpath("cpu-cpp-runner-unhealthy.json").exists()
    assert not runner.settings.data_dir.joinpath("runner-unhealthy.json").exists()


def test_python_recovery_clears_only_python_marker(runner: CpuRunner) -> None:
    runner._select("python")
    runner.mark_unhealthy("test")
    assert runner.recover(language="python").healthy
    assert not runner.settings.data_dir.joinpath("cpu-python-runner-unhealthy.json").exists()


def test_cpu_rejects_gpu_languages(runner: CpuRunner) -> None:
    with pytest.raises(RunnerUnavailable):
        runner.probe_cpu_environment("cuda_cpp")
    with pytest.raises(RunnerUnavailable):
        runner.probe_triton_environment()


def test_process_does_not_inherit_credentials(
    runner: CpuRunner, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MYLEETGPU_TEST_SECRET", "must-not-reach-a-submission")
    command = [
        sys.executable,
        "-I",
        "-S",
        "-c",
        "import os; print(os.environ.get('MYLEETGPU_TEST_SECRET', 'absent'))",
    ]
    result = runner._run_limited(command, cwd=tmp_path, timeout=5, limit=1024)
    assert result.returncode == 0
    assert result.output.strip() == "absent"
    assert runner._processes == {}


def test_process_timeout_is_bounded(runner: CpuRunner, tmp_path: Path) -> None:
    result = runner._run_limited(
        [sys.executable, "-I", "-S", "-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        timeout=0.2,
        limit=1024,
    )
    assert result.timed_out and result.duration_seconds < 5
    assert result.returncode != 0
    assert not runner._processes


def test_process_output_limit_is_bounded(runner: CpuRunner, tmp_path: Path) -> None:
    result = runner._run_limited(
        [sys.executable, "-I", "-S", "-c", "print('x' * 100000)"],
        cwd=tmp_path,
        timeout=5,
        limit=1024,
    )
    assert result.output_limited
    assert len(result.output.encode()) <= 1024
    assert not runner._processes


@pytest.mark.parametrize("language", ["cpp", "python"])
def test_real_online_softmax_compile_validate_benchmark(runner: CpuRunner, language: str) -> None:
    if language == "cpp" and not shutil.which("c++"):
        pytest.skip("requires a C++17 compiler")
    problem = ProblemCatalog(runner.settings.problems_dir).load().get("online-softmax")
    implementation = problem.get_implementation(language)
    task = runner.settings.jobs_dir / f"acceptance-{language}"
    task.mkdir()
    source = task / f"source{implementation.source_suffix}"
    source.write_text(implementation.starter_code, encoding="utf-8")
    for harness_kind, modes in (("validator", ("public", "full")), ("benchmark", ("benchmark",))):
        compiled = runner.compile(
            task,
            problem,
            source,
            language=language,
            implementation=implementation,
            harness_kind=harness_kind,
        )
        assert compiled.succeeded, compiled.diagnostics
        for mode in modes:
            result = runner.execute(
                task, compiled.executable, mode=mode, timeout_seconds=30, language=language
            )
            assert result.succeeded, result.output
            assert result.parsed["status"] == "passed"
            if mode == "benchmark":
                assert result.parsed["protocol_version"] == "1"
                assert len(result.parsed["measurements"]) == len(problem.manifest.benchmark.sizes)
    runner.cleanup_task(task)
    assert not task.exists()


def test_python_syntax_error_is_compile_failure(runner: CpuRunner) -> None:
    problem = ProblemCatalog(runner.settings.problems_dir).load().get("online-softmax")
    task = runner.settings.jobs_dir / "syntax-error"
    task.mkdir()
    source = task / "source.py"
    source.write_text("def broken(:\n", encoding="utf-8")
    result = runner.compile(task, problem, source, language="python")
    assert not result.succeeded and result.executable is None
    assert "SyntaxError" in result.diagnostics


def test_execute_rejects_artifact_outside_job(runner: CpuRunner, tmp_path: Path) -> None:
    script = tmp_path / "outside.py"
    script.write_text("raise RuntimeError('must not run')", encoding="utf-8")
    with pytest.raises(ValueError, match="inside the job"):
        runner.execute(
            runner.settings.jobs_dir, script, mode="public", timeout_seconds=1, language="python"
        )


def test_duplicate_protocol_records_are_rejected(runner: CpuRunner, tmp_path: Path) -> None:
    task = runner.settings.jobs_dir / "duplicate-output"
    task.mkdir()
    script = task / "platform.py"
    payload = "MYLEETGPU_RESULT=" + json.dumps({"status": "passed"})
    script.write_text(f"print({payload!r}); print({payload!r})", encoding="utf-8")
    result = runner.execute(task, script, mode="public", timeout_seconds=5, language="python")
    assert not result.succeeded and result.parsed is None
