"""CPU language integration without Docker, SSH or GPU dependencies."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from myleetgpu.api.main import create_app
from myleetgpu.config import Settings
from myleetgpu.domain.jobs import GPU_RESOURCE, JobAction
from myleetgpu.runner.router import ExecutionRouter
from myleetgpu.worker import Worker

from tests.factories import make_probe
from tests.test_worker import FakeRunner


class CpuFakeRunner(FakeRunner):
    def probe_cpu_environment(self, language="cpp", *, force=False, ignore_circuit_breaker=False):
        self.calls.append(("probe_cpu_environment", language, force))
        return replace(
            make_probe(f"cpu-{language}-test", backend=language),
            gpu_name=None,
            cuda_arch=None,
            compute_capability=None,
            nvcc_version=None,
            cuda_runtime_version=None,
            cuda_image="cpu-native",
            toolchain={"execution_target": "cpu", "isolation": "trusted-native"},
        )

    def recover(self, *, language="cpp"):
        return self.probe_cpu_environment(language, force=True)


@pytest.fixture
def cpu_bundle(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "data",
        problems_dir=Path(__file__).resolve().parents[1] / "problems",
        database_url_override=f"sqlite:///{tmp_path / 'cpu.db'}",
        _env_file=None,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        local, colab, cpu = FakeRunner(), FakeRunner(), CpuFakeRunner()
        router = ExecutionRouter(settings, runners={"local": local, "colab": colab, "cpu": cpu})
        worker = Worker(
            settings, app.state.catalog, app.state.repository, router, worker_id="cpu-test"
        )
        yield client, app, local, colab, cpu, worker


@pytest.mark.parametrize("language", ["cpp", "python"])
@pytest.mark.parametrize("action", ["compile", "run", "validate", "save_version"])
def test_cpu_jobs_ignore_gpu_provider_and_use_cpu_adapter(cpu_bundle, language, action):
    client, app, local, colab, cpu, worker = cpu_bundle
    app.state.repository.update_execution_settings("colab", True)
    problem = app.state.catalog.get("online-softmax")
    source = problem.get_implementation(language).starter_code
    submitted = client.post(
        "/api/jobs",
        json={
            "problem_id": "online-softmax",
            "language": language,
            "action": action,
            "source": source,
            "version_name": "cpu test" if action == "save_version" else None,
        },
    )
    assert submitted.status_code == 202, submitted.text
    assert submitted.json()["execution_target"] == "cpu"
    job = app.state.repository.get_job(submitted.json()["id"])
    assert job.payload_json["colab_acknowledged"] is False
    suffix = ".cpp" if language == "cpp" else ".py"
    assert Path(job.spool_path, f"source{suffix}").read_text() == source
    assert worker.process_next()
    completed = client.get(f"/api/jobs/{job.id}").json()
    assert completed["status"] == "succeeded", completed
    assert completed["result"]["execution_target"] == "cpu"
    assert any(call[0] == "compile" and call[4] == language for call in cpu.calls)
    if action != "compile":
        assert any(call[:2] == ("probe_cpu_environment", language) for call in cpu.calls)
    assert not local.calls and not colab.calls
    assert app.state.repository.execution_settings()["target"] == "colab"


@pytest.mark.parametrize("language", ["cpp", "python"])
def test_cpu_submission_does_not_require_colab_acknowledgement(cpu_bundle, monkeypatch, language):
    client, app, _, _, _, _ = cpu_bundle
    monkeypatch.setattr(
        app.state.repository,
        "execution_settings",
        lambda: {"target": "colab", "colab_acknowledged": False},
    )
    response = client.post(
        "/api/jobs",
        json={
            "problem_id": "online-softmax",
            "language": language,
            "action": "compile",
            "source": "a trusted local CPU submission",
        },
    )
    assert response.status_code == 202
    assert response.json()["execution_target"] == "cpu"


def test_cpu_readiness_and_circuit_breakers_are_language_isolated(cpu_bundle):
    client, app, local, colab, _, worker = cpu_bundle
    repository = app.state.repository
    repository.update_execution_settings("colab", True)
    repository.acquire_lease(GPU_RESOURCE, worker.worker_id)
    worker._probe_cpu_environments()
    for name in ("runner-unhealthy.json", "colab-runner-unhealthy.json"):
        (app.state.settings.data_dir / name).write_text(json.dumps({"reason": "GPU unavailable"}))
    for language in ("cpp", "python"):
        environment = client.get(f"/api/environment?language={language}").json()
        assert environment["healthy"] is True
        assert environment["execution_target"] == "cpu"
        assert environment["gpu_name"] is None
        assert client.get(f"/api/ready?language={language}").status_code == 200
    assert client.get("/api/ready").status_code == 503
    (app.state.settings.data_dir / "cpu-cpp-runner-unhealthy.json").write_text(
        json.dumps({"reason": "C++ runtime failed"})
    )
    assert client.get("/api/environment?language=cpp").json()["healthy"] is False
    assert client.get("/api/environment?language=python").json()["healthy"] is True
    assert client.get("/api/ready?language=cpp").status_code == 503
    assert client.get("/api/ready?language=python").status_code == 200
    assert not local.calls and not colab.calls


@pytest.mark.parametrize(
    "body",
    [
        {"target": "cpu", "language": "cuda_cpp"},
        {"target": "cpu", "language": "triton_python"},
        {"target": "local", "language": "cpp"},
        {"target": "colab", "language": "python"},
    ],
)
def test_probes_reject_mismatched_language_and_target(cpu_bundle, body):
    client, _, local, colab, cpu, _ = cpu_bundle
    assert client.post("/api/execution-settings/probe", json=body).status_code == 422
    assert not local.calls and not colab.calls and not cpu.calls


@pytest.mark.parametrize("language", ["cpp", "python"])
def test_cpu_connection_probes_are_worker_serialized(cpu_bundle, language):
    _, app, local, colab, cpu, worker = cpu_bundle
    probe_id = app.state.repository.enqueue_execution_probe("cpu", language, 10)
    assert not cpu.calls
    assert worker.process_next()
    result = app.state.repository.execution_probe(probe_id)
    assert result.status == "completed"
    assert result.result_json["healthy"] is True
    assert result.result_json["backend"] == language
    assert result.result_json["execution_target"] == "cpu"
    assert not local.calls and not colab.calls


def test_cpu_is_not_a_gpu_provider_setting(cpu_bundle):
    client, _, _, _, _, _ = cpu_bundle
    assert client.put("/api/execution-settings", json={"target": "cpu"}).status_code == 422


def test_worker_loop_starts_and_judges_cpu_without_touching_unavailable_gpu(
    cpu_bundle, monkeypatch
):
    client, app, local, colab, _, worker = cpu_bundle
    repository = app.state.repository
    repository.update_execution_settings("colab", True)
    orphan = app.state.jobs.submit(
        problem_id="vector-addition", action=JobAction.COMPILE, source="void solve() {}"
    )
    repository.claim_next_job("previous-worker")
    orphan_spool = Path(orphan.spool_path)
    source = app.state.catalog.get("online-softmax").get_implementation("python").starter_code
    submitted = client.post(
        "/api/jobs",
        json={
            "problem_id": "online-softmax",
            "language": "python",
            "action": "run",
            "source": source,
        },
    ).json()

    def no_gpu(*args, **kwargs):
        pytest.fail("CPU Worker startup or judging contacted a GPU provider")

    for runner in (local, colab):
        for method in ("probe_environment", "cleanup_orphan_containers", "cleanup_task", "compile"):
            monkeypatch.setattr(runner, method, no_gpu)
    process_next = worker.process_next

    def process_once():
        processed = process_next()
        worker.stop()
        assert processed
        return processed

    monkeypatch.setattr(worker, "process_next", process_once)
    worker.run_forever()
    assert client.get(f"/api/jobs/{submitted['id']}").json()["status"] == "succeeded"
    assert repository.get_job(orphan.id).status == "system_error"
    assert not orphan_spool.exists()
    assert not local.calls and not colab.calls
    assert not repository.has_active_lease(GPU_RESOURCE)


def test_idle_worker_refreshes_cpu_without_periodic_gpu_probe(cpu_bundle, monkeypatch):
    _, _, local, colab, cpu, worker = cpu_bundle

    def no_gpu(*args, **kwargs):
        pytest.fail("Idle CPU Worker contacted a GPU provider")

    for runner in (local, colab):
        monkeypatch.setattr(runner, "probe_environment", no_gpu)
        monkeypatch.setattr(runner, "cleanup_orphan_containers", no_gpu)
    original_probe = worker._probe_cpu_environments
    probe_count = 0

    def refresh():
        nonlocal probe_count
        original_probe()
        probe_count += 1
        if probe_count == 1:
            worker._last_environment_probe = 0
        else:
            worker.stop()

    monkeypatch.setattr(worker, "_probe_cpu_environments", refresh)
    worker.run_forever()
    assert probe_count == 2
    assert len([call for call in cpu.calls if call[0] == "probe_cpu_environment"]) == 4
    assert not local.calls and not colab.calls
