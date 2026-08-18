from __future__ import annotations

import io
import json
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from myleetgpu.config import Settings
from myleetgpu.domain.problems import Problem, ProblemCatalog
from myleetgpu.runner.docker import (
    CONTAINER_FILE_BYTES,
    CONTAINER_USER,
    CUDA_TMPFS,
    RESULT_PREFIX,
    RUNNER_LABEL,
    TRITON_POLICY_FILENAME,
    TRITON_PYTHON,
    TRITON_TMPFS,
    DockerRunner,
    _safe_name,
)
from myleetgpu.runner.models import (
    CommandResult,
    EnvironmentProbe,
    RunnerUnavailable,
    RunnerUnhealthy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    value = Settings(
        data_dir=tmp_path / "container-data",
        host_data_dir=tmp_path / "docker-visible-data",
        problems_dir=PROJECT_ROOT / "problems",
        output_limit_bytes=4096,
        _env_file=None,
    )
    value.ensure_directories()
    return value


@pytest.fixture
def runner(settings: Settings) -> DockerRunner:
    return DockerRunner(settings)


@pytest.fixture(scope="module")
def vector_problem() -> Problem:
    return ProblemCatalog(PROJECT_ROOT / "problems").load().get("vector-addition")


@pytest.fixture(scope="module")
def triton_implementation(vector_problem: Problem):
    return vector_problem.get_implementation(TRITON_PYTHON)


def healthy_probe(settings: Settings) -> EnvironmentProbe:
    return EnvironmentProbe(
        healthy=True,
        gpu_name="NVIDIA GeForce RTX 4060",
        compute_capability="8.9",
        driver_version="999.1",
        cuda_runtime_version="12.4",
        nvcc_version="12.4",
        cuda_image=settings.cuda_image,
        image_digest="sha256:" + "d" * 64,
        cuda_arch="89",
        fingerprint="f" * 64,
    )


def option_value(args: Sequence[str], option: str) -> str:
    index = args.index(option)
    return args[index + 1]


def test_safe_container_name_removes_metacharacters_and_is_bounded() -> None:
    value = _safe_name("job;$(touch /tmp/pwned) `whoami` /" + "x" * 100)

    assert len(value) == 80
    assert value.startswith("job---touch--tmp-pwned---whoami---")
    assert set(value) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")


@pytest.mark.parametrize(("gpu", "expects_gpu"), [(False, False), (True, True)])
def test_container_arguments_enforce_the_sandbox_baseline(
    runner: DockerRunner,
    settings: Settings,
    gpu: bool,
    expects_gpu: bool,
) -> None:
    task_dir = settings.jobs_dir / "job-123" / "stage"
    task_dir.mkdir(parents=True)

    args = runner._base_container_args("safe-name", task_dir, gpu=gpu)

    assert args[:2] == ["docker", "run"]
    assert "--rm" in args
    assert option_value(args, "--name") == "safe-name"
    labels = [args[index + 1] for index, value in enumerate(args) if value == "--label"]
    assert RUNNER_LABEL in labels
    assert any(label.startswith("com.myleetgpu.installation=") for label in labels)
    assert option_value(args, "--network") == "none"
    assert option_value(args, "--log-driver") == "none"
    assert "--read-only" in args
    assert option_value(args, "--user") == CONTAINER_USER
    assert "--cap-drop=ALL" in args
    assert "no-new-privileges=true" in args
    security_options = [
        args[index + 1] for index, value in enumerate(args) if value == "--security-opt"
    ]
    assert security_options == ["no-new-privileges=true"]
    # Omitting an explicit seccomp option keeps Docker's built-in default
    # profile enabled while remaining compatible with older Docker CLIs.
    assert "seccomp=unconfined" not in security_options
    assert "--pid" not in args  # Docker's default PID namespace is private.
    assert option_value(args, "--ipc") == "private"
    assert option_value(args, "--pids-limit") == "128"
    assert option_value(args, "--memory") == settings.container_memory
    assert option_value(args, "--memory-swap") == settings.container_memory
    assert option_value(args, "--cpus") == str(settings.container_cpus)
    assert option_value(args, "--ulimit") == (
        f"fsize={CONTAINER_FILE_BYTES}:{CONTAINER_FILE_BYTES}"
    )
    assert option_value(args, "--tmpfs") == "/tmp:rw,nosuid,nodev,noexec,size=64m"
    assert option_value(args, "--workdir") == "/work"
    assert option_value(args, "--stop-timeout") == "1"
    assert "--privileged" not in args
    assert "host" not in args
    assert "/var/run/docker.sock" not in " ".join(args)
    assert ("--gpus" in args) is expects_gpu
    if expects_gpu:
        assert option_value(args, "--gpus") == "device=0"

    expected_host = f"{settings.host_data_mount}/jobs/job-123/stage"
    expected_mount = f"type=bind,src={expected_host},dst=/work"
    if gpu:
        expected_mount += ",readonly"
    assert option_value(args, "--mount") == expected_mount
    assert str(PROJECT_ROOT) not in option_value(args, "--mount")
    environment_values = [args[index + 1] for index, value in enumerate(args) if value == "--env"]
    assert environment_values == ["HOME=/tmp", "CUDA_CACHE_DISABLE=1"]


def test_windows_host_mount_path_is_not_resolved_as_a_posix_relative_path(
    tmp_path: Path,
) -> None:
    selected = Settings(
        data_dir=tmp_path / "container-data",
        host_data_dir=Path("D:/MyLeetGpu/data"),
        problems_dir=PROJECT_ROOT / "problems",
        _env_file=None,
    )
    selected.ensure_directories()
    task_dir = selected.jobs_dir / "job-windows-host" / "stage"
    task_dir.mkdir(parents=True)

    args = DockerRunner(selected)._base_container_args("windows-host", task_dir, gpu=True)

    assert option_value(args, "--mount") == (
        "type=bind,src=D:/MyLeetGpu/data/jobs/job-windows-host/stage,dst=/work,readonly"
    )


@pytest.mark.parametrize("gpu", [False, True])
def test_triton_container_arguments_keep_sandbox_and_use_private_executable_cache(
    runner: DockerRunner,
    settings: Settings,
    gpu: bool,
) -> None:
    task_dir = settings.jobs_dir / "job-triton-sandbox" / ("run" if gpu else "compile")
    task_dir.mkdir(parents=True)

    args = runner._base_container_args(
        "triton-safe",
        task_dir,
        gpu=gpu,
        language=TRITON_PYTHON,
        read_only_workdir=True,
    )

    assert option_value(args, "--network") == "none"
    assert option_value(args, "--user") == CONTAINER_USER
    assert "--read-only" in args
    assert "--cap-drop=ALL" in args
    assert "no-new-privileges=true" in args
    assert option_value(args, "--ipc") == "private"
    assert option_value(args, "--pids-limit") == "128"
    assert option_value(args, "--memory") == settings.container_memory
    assert option_value(args, "--memory-swap") == settings.container_memory
    assert option_value(args, "--cpus") == str(settings.container_cpus)
    assert option_value(args, "--tmpfs") == TRITON_TMPFS
    assert "noexec" not in option_value(args, "--tmpfs")
    assert "exec" in option_value(args, "--tmpfs")
    assert option_value(args, "--mount").endswith("dst=/work,readonly")
    assert ("--gpus" in args) is gpu
    if gpu:
        assert option_value(args, "--gpus") == "device=0"
    environments = [args[index + 1] for index, value in enumerate(args) if value == "--env"]
    assert environments == [
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
    assert "CUDA_CACHE_DISABLE=1" not in environments
    assert "/var/run/docker.sock" not in " ".join(args)


def test_cuda_tmpfs_stays_small_and_non_executable(
    runner: DockerRunner,
    settings: Settings,
) -> None:
    task_dir = settings.jobs_dir / "job-cuda-tmpfs" / "stage"
    task_dir.mkdir(parents=True)

    args = runner._base_container_args("cuda-safe", task_dir, gpu=True)

    assert option_value(args, "--tmpfs") == CUDA_TMPFS


def test_compile_container_never_receives_gpu_access(
    runner: DockerRunner,
    settings: Settings,
    vector_problem: Problem,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_root = settings.jobs_dir / "job-compile"
    task_root.mkdir()
    source_path = task_root / "snapshot.cu"
    source_path.write_text(vector_problem.starter_code, encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(runner, "probe_environment", lambda: healthy_probe(settings))

    def fake_run(
        args: Sequence[str],
        name: str,
        *,
        timeout: float,
        limit: int | None = None,
        platform_owned: bool = False,
    ) -> CommandResult:
        captured.update(
            args=list(args),
            name=name,
            timeout=timeout,
            limit=limit,
            platform_owned=platform_owned,
        )
        (task_root / "compile-validator" / "program").write_bytes(b"executable")
        return CommandResult(tuple(args), 0, "", 0.1)

    monkeypatch.setattr(runner, "_run_container", fake_run)

    result = runner.compile(task_root, vector_problem, source_path)
    args = captured["args"]

    assert result.succeeded
    assert isinstance(args, list)
    assert "--gpus" not in args
    assert args[-11:] == [
        "--entrypoint",
        "nvcc",
        settings.cuda_image,
        *vector_problem.compile_flags,
        "-arch=sm_89",
        "-I/work",
        "/work/source.cu",
        "/work/platform.cu",
        "-o",
        "/work/program",
    ]
    assert captured["timeout"] == 30.0


@pytest.mark.parametrize(
    ("kind", "expected_harness"),
    [("validator", "validator_path"), ("benchmark", "benchmark_path")],
)
def test_prepare_compile_copies_only_the_exact_required_inputs(
    runner: DockerRunner,
    settings: Settings,
    vector_problem: Problem,
    kind: str,
    expected_harness: str,
) -> None:
    task_root = settings.jobs_dir / f"job-{kind}"
    task_root.mkdir()
    source_path = task_root / "source-snapshot"
    source_path.write_text("// immutable click-time snapshot", encoding="utf-8")

    compile_dir = runner.prepare_compile(task_root, vector_problem, source_path, kind)

    assert {path.name for path in compile_dir.iterdir()} == {
        "source.cu",
        "solve.h",
        "platform.cu",
    }
    assert (compile_dir / "source.cu").read_text(encoding="utf-8") == (
        "// immutable click-time snapshot"
    )
    assert (compile_dir / "solve.h").read_bytes() == vector_problem.header_path.read_bytes()
    assert (compile_dir / "platform.cu").read_bytes() == getattr(
        vector_problem, expected_harness
    ).read_bytes()
    for filename in ("source.cu", "solve.h", "platform.cu"):
        mode = stat.S_IMODE((compile_dir / filename).stat().st_mode)
        assert mode & stat.S_IWOTH == 0


def test_prepare_compile_rejects_unknown_harness_without_creating_a_directory(
    runner: DockerRunner,
    settings: Settings,
    vector_problem: Problem,
) -> None:
    task_root = settings.jobs_dir / "job-unknown-harness"
    task_root.mkdir()
    source_path = task_root / "source.cu"
    source_path.write_text(vector_problem.starter_code, encoding="utf-8")

    with pytest.raises(ValueError, match="unknown harness"):
        runner.prepare_compile(task_root, vector_problem, source_path, "attacker-choice")

    assert list(task_root.iterdir()) == [source_path]


def test_prepare_compile_rejects_unknown_language_before_creating_stage(
    runner: DockerRunner,
    settings: Settings,
    vector_problem: Problem,
) -> None:
    task_root = settings.jobs_dir / "job-unknown-language"
    task_root.mkdir()
    source_path = task_root / "source.txt"
    source_path.write_text("untrusted", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown runner language"):
        runner.prepare_compile(
            task_root,
            vector_problem,
            source_path,
            "validator",
            language="shell",
        )

    assert list(task_root.iterdir()) == [source_path]


def test_prepare_triton_compile_copies_source_platform_and_policy(
    runner: DockerRunner,
    settings: Settings,
    vector_problem: Problem,
    triton_implementation,
) -> None:
    task_root = settings.jobs_dir / "job-triton-prepare"
    task_root.mkdir()
    source_path = task_root / "snapshot.py"
    source_path.write_text("def solve(*args):\n    return None\n", encoding="utf-8")

    compile_dir = runner.prepare_compile(
        task_root,
        vector_problem,
        source_path,
        "validator",
        language=TRITON_PYTHON,
        implementation=triton_implementation,
    )

    assert {path.name for path in compile_dir.iterdir()} == {
        "source.py",
        "platform.py",
        TRITON_POLICY_FILENAME,
    }
    assert (compile_dir / "source.py").read_text(encoding="utf-8") == (
        "def solve(*args):\n    return None\n"
    )
    assert (compile_dir / "platform.py").read_bytes() == (
        triton_implementation.validator_path.read_bytes()
    )
    assert stat.S_IMODE(compile_dir.stat().st_mode) == 0o755
    assert (compile_dir / TRITON_POLICY_FILENAME).read_bytes() == (
        PROJECT_ROOT / "backend/myleetgpu/runner/submission_policy.py"
    ).read_bytes()
    for filename in ("source.py", "platform.py", TRITON_POLICY_FILENAME):
        assert stat.S_IMODE((compile_dir / filename).stat().st_mode) & stat.S_IWOTH == 0


def test_triton_compile_uses_no_gpu_readonly_source_and_pinned_image(
    runner: DockerRunner,
    settings: Settings,
    vector_problem: Problem,
    triton_implementation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_root = settings.jobs_dir / "job-triton-compile"
    task_root.mkdir()
    source_path = task_root / "snapshot.py"
    source_path.write_text("def solve(*args):\n    return None\n", encoding="utf-8")
    captured: dict[str, object] = {}
    monkeypatch.setattr(runner, "_inspect_image_digest", lambda image: image)

    def fake_run(
        args: Sequence[str],
        name: str,
        *,
        timeout: float,
        limit: int | None = None,
        platform_owned: bool = False,
    ) -> CommandResult:
        captured.update(
            args=list(args),
            name=name,
            timeout=timeout,
            limit=limit,
            platform_owned=platform_owned,
        )
        return CommandResult(tuple(args), 0, "", 0.1)

    monkeypatch.setattr(runner, "_run_container", fake_run)

    result = runner.compile(
        task_root,
        vector_problem,
        source_path,
        harness_kind="validator",
        language=TRITON_PYTHON,
        implementation=triton_implementation,
    )
    args = captured["args"]

    assert result.succeeded
    assert result.executable == task_root / "compile-validator" / "source.py"
    assert isinstance(args, list)
    assert "--gpus" not in args
    assert option_value(args, "--mount").endswith("dst=/work,readonly")
    assert args[-7:] == [
        "--entrypoint",
        "python",
        settings.triton_image,
        "-I",
        "-B",
        f"/work/{TRITON_POLICY_FILENAME}",
        "/work/source.py",
    ]
    assert captured["timeout"] == 30.0
    assert captured["platform_owned"] is False


def test_triton_compile_missing_optional_image_does_not_create_stage_or_trip_cuda(
    runner: DockerRunner,
    settings: Settings,
    vector_problem: Problem,
    triton_implementation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_root = settings.jobs_dir / "job-triton-image-missing"
    task_root.mkdir()
    source_path = task_root / "snapshot.py"
    source_path.write_text("pass\n", encoding="utf-8")

    def missing(_image: str) -> None:
        raise RunnerUnavailable("fixed Triton image is unavailable")

    monkeypatch.setattr(runner, "_inspect_image_digest", missing)

    with pytest.raises(RunnerUnavailable, match="fixed Triton image is unavailable"):
        runner.compile(
            task_root,
            vector_problem,
            source_path,
            language=TRITON_PYTHON,
            implementation=triton_implementation,
        )

    assert {path.name for path in task_root.iterdir()} == {"snapshot.py"}
    assert not runner._health_file.exists()
    runner.assert_healthy()


def test_execute_uses_gpu_and_parses_only_platform_result(
    runner: DockerRunner,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_root = settings.jobs_dir / "job-run"
    task_root.mkdir()
    executable = task_root / "compiled-program"
    executable.write_bytes(b"program")
    captured: dict[str, object] = {}
    payload = {"status": "passed", "passed": 2, "total": 2}

    def fake_run(
        args: Sequence[str], name: str, *, timeout: float, limit: int | None = None
    ) -> CommandResult:
        captured.update(args=list(args), name=name, timeout=timeout, limit=limit)
        output = "untrusted user text\n" + RESULT_PREFIX + json.dumps(payload)
        return CommandResult(tuple(args), 0, output, 0.2)

    monkeypatch.setattr(runner, "_run_container", fake_run)

    result = runner.execute(task_root, executable, mode="full", timeout_seconds=12.5)
    args = captured["args"]

    assert result.succeeded
    assert result.parsed == payload
    assert isinstance(args, list)
    assert option_value(args, "--gpus") == "device=0"
    assert args[-5:] == [
        "--entrypoint",
        "/work/program",
        settings.cuda_image,
        "--mode",
        "full",
    ]
    assert captured["timeout"] == 12.5
    run_dir = task_root / "run-full"
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o755
    assert stat.S_IMODE((run_dir / "program").stat().st_mode) == 0o555
    runner.cleanup_task(task_root)
    assert not task_root.exists()


@pytest.mark.parametrize("mode", ["public", "full", "benchmark"])
def test_execute_triton_uses_gpu_pinned_image_and_readonly_exact_inputs(
    runner: DockerRunner,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    task_root = settings.jobs_dir / f"job-triton-run-{mode}"
    compile_dir = task_root / "compile-validator"
    compile_dir.mkdir(parents=True)
    source = compile_dir / "source.py"
    platform = compile_dir / "platform.py"
    policy = compile_dir / TRITON_POLICY_FILENAME
    source.write_text("def solve(*args):\n    return None\n", encoding="utf-8")
    platform.write_text("# trusted platform harness\n", encoding="utf-8")
    policy.write_text("# trusted submission policy\n", encoding="utf-8")
    captured: dict[str, object] = {}
    payload = {"status": "passed", "mode": mode}

    def fake_run(
        args: Sequence[str],
        name: str,
        *,
        timeout: float,
        limit: int | None = None,
        platform_owned: bool = False,
    ) -> CommandResult:
        captured.update(args=list(args), name=name, timeout=timeout)
        return CommandResult(tuple(args), 0, RESULT_PREFIX + json.dumps(payload), 0.2)

    monkeypatch.setattr(runner, "_run_container", fake_run)

    result = runner.execute(
        task_root,
        source,
        mode=mode,
        timeout_seconds=13.5,
        language=TRITON_PYTHON,
    )
    args = captured["args"]

    assert result.succeeded
    assert result.parsed == payload
    assert isinstance(args, list)
    assert option_value(args, "--gpus") == "device=0"
    assert option_value(args, "--mount").endswith("dst=/work,readonly")
    assert option_value(args, "--tmpfs") == TRITON_TMPFS
    assert args[-8:] == [
        "--entrypoint",
        "python",
        settings.triton_image,
        "-I",
        "-B",
        "/work/platform.py",
        "--mode",
        mode,
    ]
    assert captured["timeout"] == 13.5
    run_dir = task_root / f"run-{mode}"
    assert {path.name for path in run_dir.iterdir()} == {
        "source.py",
        "platform.py",
        TRITON_POLICY_FILENAME,
    }
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o755
    assert stat.S_IMODE((run_dir / "source.py").stat().st_mode) == 0o444
    assert stat.S_IMODE((run_dir / "platform.py").stat().st_mode) == 0o444
    assert stat.S_IMODE((run_dir / TRITON_POLICY_FILENAME).stat().st_mode) == 0o444


def test_execute_infers_triton_language_from_python_artifact(
    runner: DockerRunner,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_root = settings.jobs_dir / "job-triton-inferred"
    compile_dir = task_root / "compile-validator"
    compile_dir.mkdir(parents=True)
    source = compile_dir / "source.py"
    source.write_text("pass\n", encoding="utf-8")
    (compile_dir / "platform.py").write_text("pass\n", encoding="utf-8")
    (compile_dir / TRITON_POLICY_FILENAME).write_text("pass\n", encoding="utf-8")
    captured: dict[str, list[str]] = {}

    def fake_run(args, name, *, timeout, limit=None, platform_owned=False):
        captured["args"] = list(args)
        return CommandResult(tuple(args), 0, RESULT_PREFIX + '{"status":"passed"}', 0.1)

    monkeypatch.setattr(runner, "_run_container", fake_run)

    result = runner.execute(task_root, source, mode="public", timeout_seconds=1)

    assert result.succeeded
    assert settings.triton_image in captured["args"]


def test_execute_triton_rejects_artifact_without_submission_policy_before_container_start(
    runner: DockerRunner,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_root = settings.jobs_dir / "job-triton-missing-policy"
    compile_dir = task_root / "compile-validator"
    compile_dir.mkdir(parents=True)
    source = compile_dir / "source.py"
    source.write_text("pass\n", encoding="utf-8")
    (compile_dir / "platform.py").write_text("pass\n", encoding="utf-8")
    invoked = False

    def fake_run(*_args, **_kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("container must not start without the trusted policy")

    monkeypatch.setattr(runner, "_run_container", fake_run)

    with pytest.raises(ValueError, match="missing submission policy"):
        runner.execute(
            task_root,
            source,
            mode="public",
            timeout_seconds=1,
            language=TRITON_PYTHON,
        )

    assert invoked is False


def test_execute_rejects_unknown_mode_before_touching_task_directory(
    runner: DockerRunner, settings: Settings
) -> None:
    task_root = settings.jobs_dir / "job-invalid-mode"
    task_root.mkdir()
    executable = task_root / "program"
    executable.write_bytes(b"program")

    with pytest.raises(ValueError, match="unknown execution mode"):
        runner.execute(task_root, executable, mode="shell", timeout_seconds=1)

    assert {path.name for path in task_root.iterdir()} == {"program"}


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (RESULT_PREFIX + '{"status":"passed"}', {"status": "passed"}),
        ("noise\n" + RESULT_PREFIX + '{"status":"failed"}\nmore', {"status": "failed"}),
        (RESULT_PREFIX + "[]", None),
        (RESULT_PREFIX + "not-json", None),
        ("user says status=passed", None),
    ],
)
def test_result_parser_requires_a_prefixed_json_object(
    output: str, expected: dict[str, object] | None
) -> None:
    assert DockerRunner._parse_result(output) == expected


def test_result_parser_rejects_multiple_platform_records() -> None:
    output = "\n".join(
        [
            RESULT_PREFIX + '{"status":"failed"}',
            "user output",
            RESULT_PREFIX + '{"status":"passed","total":3}',
        ]
    )

    assert DockerRunner._parse_result(output) is None


def test_container_start_failure_is_classified_as_runner_unavailable(
    runner: DockerRunner, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "_run_limited",
        lambda args, *, timeout, limit: CommandResult(
            tuple(args), 125, "docker: Error response from daemon: runtime unavailable", 0.1
        ),
    )

    with pytest.raises(RunnerUnavailable, match="container runtime failed"):
        runner._run_container(
            ["docker", "run"],
            "failed-container",
            timeout=1,
            platform_owned=True,
        )

    assert (settings.data_dir / "runner-unhealthy.json").is_file()


def test_diagnostics_remove_job_and_harness_paths(runner: DockerRunner, settings: Settings) -> None:
    task_dir = settings.jobs_dir / "private-job" / "compile-validator"
    output = (
        f"{task_dir.as_posix()}/source.cu:7: error\n"
        "/work/platform.cu:99: internal reference\n"
        "platform.cu leaked"
    )

    cleaned = runner._clean_diagnostics(output, task_dir)

    assert str(settings.data_dir) not in cleaned
    assert "/work" not in cleaned
    assert "platform.cu" not in cleaned
    assert "source.cu:7: error" in cleaned


def test_python_traceback_keeps_submission_location_but_hides_platform_path(
    runner: DockerRunner,
    settings: Settings,
) -> None:
    task_dir = settings.jobs_dir / "private-triton-job" / "run-full"
    output = (
        'Traceback (most recent call last):\n  File "/work/platform.py", line 88, in main\n'
        '  File "/work/source.py", line 7, in solve\nRuntimeError: invalid kernel\n'
        '  File "/work/submission_policy.py", line 99, in load_submission\n'
        f"host detail: {task_dir.as_posix()}/platform.py"
    )

    cleaned = runner._clean_diagnostics(output, task_dir)

    assert "/work" not in cleaned
    assert "platform.py" not in cleaned
    assert "submission_policy.py" not in cleaned
    assert str(settings.data_dir) not in cleaned
    assert 'File "source.py", line 7' in cleaned
    assert "<platform>" in cleaned


def test_clean_output_removes_nul_and_truncates_by_bytes(
    runner: DockerRunner, settings: Settings
) -> None:
    output = "ok\x00" + "测" * settings.output_limit_bytes

    cleaned = runner._clean_output(output)

    assert "\x00" not in cleaned
    assert cleaned.endswith("[platform] output truncated")
    assert len(cleaned.encode("utf-8")) <= (
        settings.output_limit_bytes + len(b"\n[platform] output truncated")
    )


def test_run_limited_stops_process_at_output_limit() -> None:
    result = DockerRunner._run_limited(
        [sys.executable, "-c", "import sys; sys.stdout.write('x' * 100000); sys.stdout.flush()"],
        timeout=5,
        limit=1024,
    )

    assert result.output_limited
    assert not result.timed_out
    assert result.output.endswith("[platform] output limit exceeded")
    assert len(result.output.encode()) <= 1024 + len(b"\n[platform] output limit exceeded")


def test_run_limited_kills_process_at_wall_clock_timeout() -> None:
    result = DockerRunner._run_limited(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        timeout=0.1,
        limit=1024,
    )

    assert result.timed_out
    assert not result.output_limited
    assert result.output.endswith("[platform] process timed out")
    assert result.duration_seconds < 3


@pytest.mark.parametrize(("timed_out", "output_limited"), [(True, False), (False, True)])
def test_container_timeout_or_output_limit_forces_named_container_removal(
    runner: DockerRunner,
    monkeypatch: pytest.MonkeyPatch,
    timed_out: bool,
    output_limited: bool,
) -> None:
    monkeypatch.setattr(
        runner,
        "_run_limited",
        lambda args, *, timeout, limit: CommandResult(
            tuple(args),
            -9,
            "stopped",
            0.1,
            timed_out=timed_out,
            output_limited=output_limited,
        ),
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_subprocess_run(args: list[str], **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr("myleetgpu.runner.docker.subprocess.run", fake_subprocess_run)

    runner._run_container(["docker", "run"], "myleetgpu-safe-id", timeout=1)

    assert calls == [
        (
            ["docker", "rm", "-f", "myleetgpu-safe-id"],
            {
                "stdout": -3,
                "stderr": -3,
                "timeout": 10,
                "check": False,
            },
        )
    ]


def test_exact_job_cleanup_removes_only_requested_direct_child(
    runner: DockerRunner, settings: Settings
) -> None:
    target = settings.jobs_dir / "job-to-clean"
    sibling = settings.jobs_dir / "job-to-keep"
    target.mkdir()
    sibling.mkdir()
    (target / "artifact").write_text("temporary", encoding="utf-8")
    (sibling / "artifact").write_text("keep", encoding="utf-8")

    runner.cleanup_task(target)

    assert not target.exists()
    assert (sibling / "artifact").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("target_kind", ["jobs-root", "outside", "nested-stage"])
def test_cleanup_refuses_any_path_other_than_one_exact_job_directory(
    runner: DockerRunner, settings: Settings, tmp_path: Path, target_kind: str
) -> None:
    targets = {
        "jobs-root": settings.jobs_dir,
        "outside": tmp_path / "outside",
        "nested-stage": settings.jobs_dir / "job" / "stage",
    }
    target = targets[target_kind]
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "must-remain"
    marker.write_text("safe", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to clean path outside job spool"):
        runner.cleanup_task(target)

    assert marker.is_file()


def test_mount_resolution_rejects_outside_and_jobs_root_paths(
    runner: DockerRunner, settings: Settings, tmp_path: Path
) -> None:
    outside = tmp_path / "outside-mount"
    outside.mkdir()

    with pytest.raises(ValueError, match="exact child"):
        runner._host_task_path(outside)
    with pytest.raises(ValueError, match="exact child"):
        runner._host_task_path(settings.jobs_dir)


def test_mount_and_cleanup_refuse_symlink_that_escapes_job_spool(
    runner: DockerRunner, settings: Settings, tmp_path: Path
) -> None:
    outside = tmp_path / "outside-user-data"
    outside.mkdir()
    marker = outside / "must-survive"
    marker.write_text("keep", encoding="utf-8")
    link = settings.jobs_dir / "fake-job"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="exact child"):
        runner._host_task_path(link)
    with pytest.raises(ValueError, match="refusing to clean"):
        runner.cleanup_task(link)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_orphan_cleanup_removes_only_ids_returned_by_runner_label_filter(
    runner: DockerRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(args: Sequence[str], *, timeout: float, limit: int) -> CommandResult:
        calls.append(tuple(args))
        if "ps" in args:
            return CommandResult(tuple(args), 0, "abc123\n\ndef456\n", 0.01)
        return CommandResult(tuple(args), 0, "", 0.01)

    monkeypatch.setattr(runner, "_run_limited", fake_run)

    removed = runner.cleanup_orphan_containers()

    assert removed == ["abc123", "def456"]
    assert calls == [
        (
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label={RUNNER_LABEL}",
            "--filter",
            f"label={runner._installation_label}",
        ),
        ("docker", "rm", "-f", "abc123", "def456"),
    ]


def test_gpu_health_error_trips_persistent_circuit_breaker(
    runner: DockerRunner,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_root = settings.jobs_dir / "job-gpu-failure"
    task_root.mkdir()
    executable = task_root / "program"
    executable.write_bytes(b"program")
    monkeypatch.setattr(
        runner,
        "_run_container",
        lambda args, name, *, timeout, limit=None, platform_owned=False: CommandResult(
            tuple(args), 1, "CUDA driver version is insufficient", 0.1
        ),
    )

    result = runner.execute(task_root, executable, mode="public", timeout_seconds=1)

    assert not result.succeeded
    assert runner._health_file.is_file()
    with pytest.raises(RunnerUnhealthy, match="trusted GPU probe failed"):
        runner.assert_healthy()


def test_submitted_stdout_cannot_trip_the_runner_circuit_breaker(
    runner: DockerRunner,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_root = settings.jobs_dir / "job-spoofed-health-error"
    task_root.mkdir()
    executable = task_root / "program"
    executable.write_bytes(b"program")
    calls = 0

    def fake_container(args, name, *, timeout, limit=None, platform_owned=False):
        nonlocal calls
        calls += 1
        if calls == 1:
            return CommandResult(tuple(args), 1, "user says: Xid no cuda-capable device", 0.1)
        return CommandResult(tuple(args), 0, "GPU 0: NVIDIA GeForce RTX 4060", 0.1)

    monkeypatch.setattr(runner, "_run_container", fake_container)

    result = runner.execute(task_root, executable, mode="public", timeout_seconds=1)

    assert not result.succeeded
    assert calls == 2
    assert not runner._health_file.exists()


def test_triton_probe_records_exact_toolchain_and_uses_optional_image_without_global_trip(
    runner: DockerRunner,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], str | None, bool]] = []

    def fake_limited(args: Sequence[str], *, timeout: float, limit: int) -> CommandResult:
        if "version" in args:
            return CommandResult(tuple(args), 0, "29.0.0\n", 0.01)
        assert "inspect" in args
        return CommandResult(
            tuple(args),
            0,
            json.dumps(["pytorch/pytorch@sha256:" + "a" * 64]),
            0.01,
        )

    def fake_probe(
        command: Sequence[str],
        *,
        gpu: bool,
        timeout: float,
        image: str | None = None,
        platform_owned: bool = True,
    ) -> CommandResult:
        calls.append((tuple(command), image, platform_owned))
        if command[0] == "nvidia-smi":
            return CommandResult(
                tuple(command),
                0,
                "NVIDIA GeForce RTX 4060, 566.26, 8.9\n",
                0.01,
            )
        payload = {
            "python_version": "3.11.11",
            "torch_version": "2.5.1+cu124",
            "triton_version": "3.1.0",
            "torch_cuda_version": "12.4",
            "cuda_available": True,
            "cuda_device_count": 1,
        }
        return CommandResult(tuple(command), 0, json.dumps(payload), 0.01)

    monkeypatch.setattr(runner, "_run_limited", fake_limited)
    monkeypatch.setattr(runner, "_docker_probe", fake_probe)
    monkeypatch.setattr(
        runner,
        "_probe_telemetry",
        lambda *, image=None, platform_owned=True: {
            "temperature_c": "50",
            "power_w": "20",
            "sm_clock_mhz": "2100",
            "gpu_busy_percent": "0",
        },
    )

    probe = runner.probe_triton_environment(force=True)

    assert probe.healthy
    assert probe.backend == TRITON_PYTHON
    assert probe.cuda_image == settings.triton_image
    assert probe.image_digest == "pytorch/pytorch@sha256:" + "a" * 64
    assert probe.gpu_name == "NVIDIA GeForce RTX 4060"
    assert probe.driver_version == "566.26"
    assert probe.compute_capability == "8.9"
    assert probe.cuda_arch == "89"
    assert probe.cuda_runtime_version == "12.4"
    assert probe.nvcc_version is None
    assert probe.toolchain == {
        "python_version": "3.11.11",
        "torch_cuda_version": "12.4",
        "torch_version": "2.5.1+cu124",
        "triton_version": "3.1.0",
    }
    assert len(probe.fingerprint) == 64
    assert all(image == settings.triton_image for _, image, _ in calls)
    assert all(platform_owned is False for _, _, platform_owned in calls)
    assert not runner._health_file.exists()


def test_missing_triton_image_is_backend_local_and_does_not_trip_cuda_circuit(
    runner: DockerRunner,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_limited(args: Sequence[str], *, timeout: float, limit: int) -> CommandResult:
        if "version" in args:
            return CommandResult(tuple(args), 0, "29.0.0\n", 0.01)
        return CommandResult(tuple(args), 1, "Error: No such image", 0.01)

    monkeypatch.setattr(runner, "_run_limited", fake_limited)

    probe = runner.probe_triton_environment(force=True)

    assert not probe.healthy
    assert probe.backend == TRITON_PYTHON
    assert probe.cuda_image == settings.triton_image
    assert "fixed Triton image is unavailable" in (probe.error or "")
    assert not runner._health_file.exists()
    runner.assert_healthy()


def test_triton_build_config_records_language_toolchain_and_architecture(
    vector_problem: Problem,
    settings: Settings,
    triton_implementation,
) -> None:
    probe = EnvironmentProbe(
        healthy=True,
        gpu_name="NVIDIA GeForce RTX 4060",
        compute_capability="8.9",
        driver_version="566.26",
        cuda_runtime_version="12.4",
        nvcc_version=None,
        cuda_image=settings.triton_image,
        image_digest="sha256:" + "a" * 64,
        cuda_arch="89",
        backend=TRITON_PYTHON,
        toolchain={
            "python_version": "3.11.11",
            "torch_version": "2.5.1+cu124",
            "triton_version": "3.1.0",
            "torch_cuda_version": "12.4",
        },
    )

    flags = DockerRunner.effective_compile_flags(
        vector_problem,
        probe,
        language=TRITON_PYTHON,
        implementation=triton_implementation,
    )

    assert flags == [
        "backend=triton_python",
        "policy=restricted_triton_v1",
        "python=3.11.11",
        "torch=2.5.1+cu124",
        "triton=3.1.0",
        "torch_cuda=12.4",
        "arch=sm_89",
    ]


def test_mark_unhealthy_bounds_persisted_reason(
    runner: DockerRunner,
) -> None:
    runner.mark_unhealthy("x" * 5000)

    payload = json.loads(runner._health_file.read_text(encoding="utf-8"))
    assert payload["reason"] == "x" * 2000
    assert isinstance(payload["marked_at"], float)


def test_runner_invokes_processes_without_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FinishedProcess:
        returncode = 0

        def __init__(self) -> None:
            self.stdout = io.BytesIO()

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float) -> int:
            return 0

    def fake_popen(args: list[str], **kwargs: object) -> FinishedProcess:
        captured.update(args=args, kwargs=kwargs)
        return FinishedProcess()

    monkeypatch.setattr("myleetgpu.runner.docker.subprocess.Popen", fake_popen)

    DockerRunner._run_limited(["docker", "version; touch /tmp/pwned"], timeout=1, limit=10)

    assert captured["args"] == ["docker", "version; touch /tmp/pwned"]
    assert captured["kwargs"]["shell"] is False  # type: ignore[index]
