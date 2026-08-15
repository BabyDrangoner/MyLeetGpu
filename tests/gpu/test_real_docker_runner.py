from __future__ import annotations

from pathlib import Path

import pytest
from myleetgpu.config import Settings
from myleetgpu.domain.problems import Problem, ProblemCatalog
from myleetgpu.runner.docker import DockerRunner, _safe_name

pytestmark = pytest.mark.gpu

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def problem_catalog() -> ProblemCatalog:
    return ProblemCatalog(PROJECT_ROOT / "problems").load()


@pytest.fixture
def real_runner(tmp_path: Path) -> DockerRunner:
    """Construct a real runner: GPU tests deliberately install no mocks or fakes."""

    data_dir = tmp_path / "real-docker-data"
    settings = Settings(
        data_dir=data_dir,
        host_data_dir=data_dir,
        problems_dir=PROJECT_ROOT / "problems",
        compile_timeout_seconds=90,
        run_timeout_seconds=30,
        validate_timeout_seconds=90,
        benchmark_timeout_seconds=180,
        _env_file=None,
    )
    settings.ensure_directories()
    return DockerRunner(settings)


def assert_real_environment(runner: DockerRunner) -> None:
    probe = runner.probe_environment(force=True)
    assert probe.healthy, probe.error
    assert probe.gpu_name is not None and "RTX 4060" in probe.gpu_name
    assert probe.compute_capability == "8.9"
    assert probe.cuda_arch == "89"
    assert probe.driver_version
    assert probe.cuda_runtime_version
    assert probe.nvcc_version
    assert probe.cuda_image == "nvidia/cuda:12.4.1-devel-ubuntu22.04"
    assert probe.image_digest, "the real image digest must be recorded"
    assert len(probe.fingerprint) == 64


def compile_snapshot(
    runner: DockerRunner,
    problem: Problem,
    task_name: str,
    source: str,
    *,
    harness_kind: str = "validator",
) -> tuple[Path, Path]:
    task_root = runner.settings.jobs_dir / task_name
    task_root.mkdir()
    source_path = task_root / "source-snapshot.cu"
    source_path.write_text(source, encoding="utf-8")
    compiled = runner.compile(
        task_root,
        problem,
        source_path,
        harness_kind=harness_kind,
    )
    assert compiled.succeeded, compiled.diagnostics
    assert not compiled.timed_out
    assert not compiled.output_limited
    assert compiled.executable is not None
    return task_root, compiled.executable


def test_real_probe_reports_the_target_rtx_4060_environment(real_runner: DockerRunner) -> None:
    assert_real_environment(real_runner)


def test_real_compile_sandbox_has_no_gpu_device(real_runner: DockerRunner) -> None:
    assert_real_environment(real_runner)
    task_dir = real_runner.settings.jobs_dir / "compile-no-gpu" / "stage"
    task_dir.mkdir(parents=True)
    name = "myleetgpu-test-compile-no-gpu"
    command = [
        *real_runner._base_container_args(name, task_dir, gpu=False),
        "--entrypoint",
        "bash",
        real_runner.settings.cuda_image,
        "-ceu",
        "test ! -e /dev/nvidiactl; test ! -e /dev/nvidia0; printf COMPILE_NO_GPU_OK",
    ]

    try:
        result = real_runner._run_container(command, name, timeout=20)
        assert result.returncode == 0, result.output
        assert not result.timed_out
        assert result.output == "COMPILE_NO_GPU_OK"
    finally:
        real_runner.cleanup_task(task_dir.parent)


@pytest.mark.parametrize("slug", ["vector-addition", "matrix-transpose", "reduction"])
def test_real_runner_compiles_and_fully_validates_each_builtin_problem(
    real_runner: DockerRunner,
    problem_catalog: ProblemCatalog,
    slug: str,
) -> None:
    assert_real_environment(real_runner)
    problem = problem_catalog.get(slug)
    task_root, executable = compile_snapshot(
        real_runner,
        problem,
        f"validate-{slug}",
        problem.starter_code,
    )
    try:
        result = real_runner.execute(
            task_root,
            executable,
            mode="full",
            timeout_seconds=problem.manifest.timeouts.validation_ms / 1000,
        )

        assert result.succeeded, result.output
        assert not result.timed_out
        assert not result.output_limited
        assert result.parsed is not None
        assert result.parsed["status"] == "passed"
        assert result.parsed["summary"]["passed"] == result.parsed["summary"]["total"]
    finally:
        real_runner.cleanup_task(task_root)


def test_real_runner_executes_platform_timed_benchmark(
    real_runner: DockerRunner,
    problem_catalog: ProblemCatalog,
) -> None:
    assert_real_environment(real_runner)
    problem = problem_catalog.get("vector-addition")
    task_root, executable = compile_snapshot(
        real_runner,
        problem,
        "benchmark-vector-addition",
        problem.starter_code,
        harness_kind="benchmark",
    )
    try:
        result = real_runner.execute(
            task_root,
            executable,
            mode="benchmark",
            timeout_seconds=problem.manifest.timeouts.benchmark_ms / 1000,
        )

        assert result.succeeded, result.output
        assert result.parsed is not None
        assert result.parsed["status"] == "passed"
        measurements = result.parsed["measurements"]
        assert len(measurements) == len(problem.manifest.benchmark.sizes)
        for measurement in measurements:
            samples = measurement["samples_ms"]
            assert len(samples) == problem.manifest.benchmark.iterations
            assert all(isinstance(sample, int | float) and sample >= 0 for sample in samples)
    finally:
        real_runner.cleanup_task(task_root)


def test_real_gpu_sandbox_enforces_runtime_security_controls(real_runner: DockerRunner) -> None:
    assert_real_environment(real_runner)
    task_root = real_runner.settings.jobs_dir / "runtime-security"
    run_dir = task_root / "stage"
    run_dir.mkdir(parents=True)
    run_dir.chmod(0o555)
    name = "myleetgpu-test-runtime-security"
    checks = r"""
test "$(id -u)" = 65534
grep -Eq '^NoNewPrivs:[[:space:]]+1$' /proc/self/status
grep -Eq '^CapEff:[[:space:]]+0+$' /proc/self/status
! touch /root-filesystem-must-be-read-only
! touch /work/task-mount-must-not-be-writable
touch /tmp/tmpfs-is-writable
rm /tmp/tmpfs-is-writable
test ! -S /var/run/docker.sock
test ! -e /data/myleetgpu.db
test ! -e /app/problems
test ! -e /workspace
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'RTX 4060'
! timeout 3 bash -c 'exec 3<>/dev/tcp/1.1.1.1/53'
printf SECURITY_OK
""".strip()
    command = [
        *real_runner._base_container_args(name, run_dir, gpu=True),
        "--entrypoint",
        "bash",
        real_runner.settings.cuda_image,
        "-ceu",
        checks,
    ]

    try:
        result = real_runner._run_container(command, name, timeout=20)
        assert result.returncode == 0, result.output
        assert not result.timed_out
        assert result.output.endswith("SECURITY_OK")
    finally:
        real_runner.cleanup_task(task_root)


def test_real_submitted_code_cannot_see_network_source_repo_database_or_docker_socket(
    real_runner: DockerRunner,
    problem_catalog: ProblemCatalog,
) -> None:
    assert_real_environment(real_runner)
    problem = problem_catalog.get("vector-addition")
    source = r"""
#include "solve.h"

#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

void sandbox_probe() __attribute__((constructor));

void sandbox_probe() {
    const char* forbidden[] = {
        "/var/run/docker.sock",
        "/data/myleetgpu.db",
        "/app/problems",
        "/workspace",
        "/work/source.cu",
        "/work/platform.cu",
        "/work/solve.h",
    };
    for (const char* path : forbidden) {
        if (access(path, F_OK) == 0) {
            _exit(91);
        }
    }

    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd >= 0) {
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(53);
        inet_pton(AF_INET, "1.1.1.1", &address.sin_addr);
        alarm(2);
        const int connected = connect(
            fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address));
        alarm(0);
        close(fd);
        if (connected == 0) {
            _exit(92);
        }
    }
}

__global__ void add(const float* a, const float* b, float* output, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        output[index] = a[index] + b[index];
    }
}

}  // namespace

void solve(const float* a,
           const float* b,
           float* output,
           int n,
           cudaStream_t stream) {
    add<<<(n + 255) / 256, 256, 0, stream>>>(a, b, output, n);
}
""".strip()
    task_root, executable = compile_snapshot(
        real_runner,
        problem,
        "malicious-visibility-probe",
        source,
    )

    try:
        result = real_runner.execute(
            task_root,
            executable,
            mode="public",
            timeout_seconds=15,
        )
        assert result.succeeded, result.output
        assert result.parsed is not None and result.parsed["status"] == "passed"
    finally:
        real_runner.cleanup_task(task_root)


def test_real_timed_out_submission_is_removed_and_next_gpu_job_still_runs(
    real_runner: DockerRunner,
    problem_catalog: ProblemCatalog,
) -> None:
    assert_real_environment(real_runner)
    problem = problem_catalog.get("vector-addition")
    hanging_source = r"""
#include "solve.h"

#include <time.h>

void solve(const float*, const float*, float*, int, cudaStream_t) {
    const timespec interval{1, 0};
    while (true) {
        nanosleep(&interval, nullptr);
    }
}
""".strip()
    hanging_root, hanging_executable = compile_snapshot(
        real_runner,
        problem,
        "safe-host-timeout",
        hanging_source,
    )
    container_name = _safe_name(f"myleetgpu-{hanging_root.name}-run-public")
    try:
        timed_out = real_runner.execute(
            hanging_root,
            hanging_executable,
            mode="public",
            timeout_seconds=1,
        )
        assert timed_out.timed_out
        assert not timed_out.succeeded

        inspect = real_runner._run_limited(
            [
                real_runner._docker,
                "ps",
                "-a",
                "--filter",
                f"name=^{container_name}$",
                "--format",
                "{{.ID}}",
            ],
            timeout=10,
            limit=8192,
        )
        assert inspect.returncode == 0, inspect.output
        assert not inspect.output.strip(), "timed-out container was not removed"
    finally:
        real_runner.cleanup_task(hanging_root)

    recovery_root, recovery_executable = compile_snapshot(
        real_runner,
        problem,
        "post-timeout-recovery",
        problem.starter_code,
    )
    try:
        recovery = real_runner.execute(
            recovery_root,
            recovery_executable,
            mode="public",
            timeout_seconds=15,
        )
        assert recovery.succeeded, recovery.output
    finally:
        real_runner.cleanup_task(recovery_root)
