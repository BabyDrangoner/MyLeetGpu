"""CPU-only execution-routing tests; no SSH, Docker, or real GPU is invoked."""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from myleetgpu import cli
from myleetgpu.api.main import create_app
from myleetgpu.application.execution import wait_for_probe
from myleetgpu.config import Settings
from myleetgpu.domain.jobs import GPU_RESOURCE, JobAction
from myleetgpu.infrastructure.models import ExecutionProbeRecord, JobRecord, utc_now
from myleetgpu.infrastructure.repository import Repository
from myleetgpu.runner.models import ExecutionResult, RunnerUnavailable
from myleetgpu.runner.router import ExecutionRouter
from myleetgpu.worker import Worker
from pydantic import ValidationError
from sqlalchemy import select

from tests.factories import make_probe
from tests.test_worker import FakeRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "// CPU-only routing fixture\nvoid solve() {}\n"


@pytest.fixture
def execution_bundle(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "data",
        problems_dir=PROJECT_ROOT / "problems",
        database_url_override=f"sqlite:///{(tmp_path / 'execution.db').as_posix()}",
        execution_probe_timeout_seconds=1,
        _env_file=None,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        local, colab = FakeRunner(), FakeRunner()
        router = ExecutionRouter(settings, runners={"local": local, "colab": colab})
        worker = Worker(
            settings,
            app.state.catalog,
            app.state.repository,
            router,
            worker_id="cpu-execution-routing-test",
        )
        yield client, app, local, colab, router, worker


def test_selection_defaults_local_and_persists_across_api_restart(execution_bundle) -> None:
    client, app, local, colab, _, _ = execution_bundle
    initial = client.get("/api/execution-settings")
    assert initial.status_code == 200
    assert initial.json()["target"] == "local"
    assert initial.json()["colab_acknowledged"] is False
    assert initial.json()["colab"] == {
        "ssh_host": "colab-vscode",
        "remote_root": "/content/project/myleetgpu-runner",
        "isolation": "trusted-native",
    }

    saved = client.put(
        "/api/execution-settings", json={"target": "colab", "colab_acknowledged": True}
    )
    assert saved.status_code == 200
    assert saved.json()["updated_at"] is not None
    restarted_app = create_app(app.state.settings)
    with TestClient(restarted_app) as restarted_client:
        assert restarted_client.get("/api/execution-settings").json() == saved.json()
        restored = restarted_client.put("/api/execution-settings", json={"target": "local"})
        assert restored.json()["target"] == "local"
        assert restored.json()["colab_acknowledged"] is False
    assert client.get("/api/execution-settings").json()["target"] == "local"
    assert not local.calls and not colab.calls


@pytest.mark.parametrize(
    "body",
    [
        {"target": "colab"},
        {"target": "colab", "colab_acknowledged": False},
        {"target": "unconfigured-host", "colab_acknowledged": True},
        {"target": "local; touch /tmp/injected"},
        {"target": "local", "ssh_host": "attacker.invalid"},
        {"target": "local", "remote_root": "/content"},
        {"target": "local", "command": "echo injected"},
    ],
)
def test_settings_reject_unacknowledged_or_injected_configuration(execution_bundle, body) -> None:
    client, _, local, colab, _, _ = execution_bundle
    rejected = client.put("/api/execution-settings", json=body)
    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "invalid_request"
    assert client.get("/api/execution-settings").json()["target"] == "local"
    assert not local.calls and not colab.calls


@pytest.mark.parametrize(
    "body",
    [
        {"target": "attacker.invalid"},
        {"target": "colab", "ssh_host": "attacker.invalid"},
        {"target": "colab", "language": "bash"},
        {"target": "colab", "command": "id"},
    ],
)
def test_probe_rejects_arbitrary_connection_and_command_inputs(execution_bundle, body) -> None:
    client, _, local, colab, _, _ = execution_bundle
    assert client.post("/api/execution-settings/probe", json=body).status_code == 422
    assert not local.calls and not colab.calls


@pytest.mark.parametrize(
    "overrides",
    [
        {"colab_ssh_host": "-oProxyCommand=bad"},
        {"colab_ssh_host": "colab-vscode; id"},
        {"colab_session": "vscode-colab\nnew-command"},
        {"colab_remote_root": "/"},
        {"colab_remote_root": "/content"},
        {"colab_remote_root": "/content/project"},
        {"colab_remote_root": "/content/project/../other"},
        {"colab_remote_root": "/content/project/task;id"},
    ],
)
def test_server_connection_config_requires_safe_aliases_and_dedicated_root(overrides) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


def test_repository_enforces_acknowledgement_independent_of_api(execution_bundle) -> None:
    _, app, _, _, _, _ = execution_bundle
    repository = app.state.repository
    for target, acknowledged in [("colab", False), ("unknown", True)]:
        with pytest.raises(ValueError):
            repository.update_execution_settings(target, acknowledged)
    assert repository.execution_settings()["target"] == "local"


def test_submission_snapshots_target_and_acknowledgement_before_queueing(execution_bundle) -> None:
    client, app, local, colab, _, worker = execution_bundle
    local_job = client.post(
        "/api/jobs", json={"problem_id": "vector-addition", "action": "compile", "source": SOURCE}
    )
    assert local_job.status_code == 202
    assert local_job.json()["execution_target"] == "local"
    client.put("/api/execution-settings", json={"target": "colab", "colab_acknowledged": True})
    colab_job = client.post(
        "/api/jobs", json={"problem_id": "vector-addition", "action": "compile", "source": SOURCE}
    )
    assert colab_job.status_code == 202
    assert colab_job.json()["execution_target"] == "colab"
    client.put("/api/execution-settings", json={"target": "local"})

    assert worker.process_next()
    assert any(call[0] == "compile" for call in local.calls)
    assert not colab.calls
    assert worker.process_next()
    assert any(call[0] == "compile" for call in colab.calls)
    for submitted, target in [(local_job, "local"), (colab_job, "colab")]:
        result = client.get(f"/api/jobs/{submitted.json()['id']}").json()
        assert result["status"] == "succeeded"
        assert result["execution_target"] == target
        assert result["result"]["execution_target"] == target
        record = app.state.repository.get_job(result["id"])
        assert record.payload_json["execution_target"] == target
        assert record.payload_json["colab_acknowledged"] == (target == "colab")
    assert app.state.repository.execution_settings()["target"] == "local"


@pytest.mark.parametrize("target", ["local", "colab"])
@pytest.mark.parametrize("action", [JobAction.RUN, JobAction.VALIDATE])
def test_language_jobs_route_to_their_selected_adapter(execution_bundle, target, action) -> None:
    client, app, local, colab, _, worker = execution_bundle
    app.state.repository.update_execution_settings(target, target == "colab")
    job = app.state.jobs.submit(
        problem_id="vector-addition", language="triton_python", action=action, source=SOURCE
    )
    assert worker.process_next()
    assert client.get(f"/api/jobs/{job.id}").json()["status"] == "succeeded"
    selected, other = (local, colab) if target == "local" else (colab, local)
    assert any(call[0] == "probe_triton_environment" for call in selected.calls)
    assert any(call[0] == "execute" and call[-1] == "triton_python" for call in selected.calls)
    assert not other.calls


def test_legacy_job_without_target_still_runs_locally(execution_bundle) -> None:
    client, app, local, colab, _, worker = execution_bundle
    job = app.state.jobs.submit(
        problem_id="vector-addition", action=JobAction.COMPILE, source=SOURCE
    )
    with app.state.repository.session_factory.begin() as session:
        record = session.get(JobRecord, job.id)
        record.payload_json = {}
    app.state.repository.update_execution_settings("colab", True)
    assert worker.process_next()
    result = client.get(f"/api/jobs/{job.id}").json()
    assert result["status"] == "succeeded"
    assert result["execution_target"] == "local"
    assert any(call[0] == "compile" for call in local.calls)
    assert not colab.calls


def test_worker_rejects_remote_job_missing_acknowledgement(execution_bundle) -> None:
    client, app, local, colab, _, worker = execution_bundle
    job = app.state.jobs.submit(
        problem_id="vector-addition", action=JobAction.COMPILE, source=SOURCE
    )
    with app.state.repository.session_factory.begin() as session:
        session.get(JobRecord, job.id).payload_json = {"execution_target": "colab"}
    assert worker.process_next()
    result = client.get(f"/api/jobs/{job.id}").json()
    assert result["status"] == "system_error"
    assert result["error"]["code"] == "runner_unhealthy"
    assert not any(call[0] == "compile" for call in local.calls + colab.calls)


@pytest.mark.parametrize("target", ["local", "colab"])
def test_unavailable_target_fails_without_fallback(execution_bundle, monkeypatch, target) -> None:
    client, app, local, colab, _, worker = execution_bundle
    selected, other = (local, colab) if target == "local" else (colab, local)

    def unavailable(*args, **kwargs):
        raise RunnerUnavailable("configured target offline")

    monkeypatch.setattr(selected, "compile", unavailable)
    app.state.repository.update_execution_settings(target, target == "colab")
    job = app.state.jobs.submit(
        problem_id="vector-addition", action=JobAction.COMPILE, source=SOURCE
    )
    assert worker.process_next()
    result = client.get(f"/api/jobs/{job.id}").json()
    assert result["status"] == "system_error"
    assert result["execution_target"] == target
    assert result["error"]["code"] == "runner_unhealthy"
    assert not other.calls


def test_remote_execution_timeout_stays_timeout_and_never_retries_locally(execution_bundle) -> None:
    client, app, local, colab, _, worker = execution_bundle
    app.state.repository.update_execution_settings("colab", True)
    colab.execution_overrides["public"] = ExecutionResult(
        False, "timed out", None, 30.0, 124, timed_out=True
    )
    job = app.state.jobs.submit(problem_id="vector-addition", action=JobAction.RUN, source=SOURCE)
    assert worker.process_next()
    result = client.get(f"/api/jobs/{job.id}").json()
    assert result["status"] == "timed_out"
    assert result["error"]["code"] == "timeout"
    assert result["execution_target"] == "colab"
    assert not local.calls


@pytest.mark.parametrize(
    "method", ["probe_environment", "probe_triton_environment", "probe_torch_environment"]
)
def test_router_namespaces_remote_fingerprints_but_preserves_legacy_local_ones(
    execution_bundle, method
) -> None:
    _, _, local, colab, router, _ = execution_bundle
    raw = getattr(local, method)()
    local_probe = getattr(router, method)(force=True)
    router.select_target("colab")
    colab_probe = getattr(router, method)(force=True)
    assert local_probe.fingerprint == raw.fingerprint
    assert colab_probe.fingerprint != local_probe.fingerprint
    assert colab_probe.fingerprint == getattr(router, method)(force=True).fingerprint
    assert local_probe.toolchain["execution_target"] == "local"
    assert colab_probe.toolchain["execution_target"] == "colab"
    assert "execution_target" not in getattr(colab, method)().toolchain
    assert local.owner == colab.owner == "cpu-execution-routing-test"


def test_router_rejects_unknown_target_without_changing_active_adapter(execution_bundle) -> None:
    _, _, _, _, router, _ = execution_bundle
    router.select_target("colab")
    with pytest.raises(RunnerUnavailable):
        router.select_target("unknown")
    assert router.target == "colab"


def test_environment_queries_filter_target_and_language_including_legacy_records(
    execution_bundle,
) -> None:
    client, app, _, _, router, _ = execution_bundle
    repository = app.state.repository
    legacy = repository.save_environment(make_probe("legacy-local"))
    router.select_target("colab")
    remote = repository.save_environment(router.probe_environment())
    remote_triton = repository.save_environment(router.probe_triton_environment())
    assert repository.latest_environment("cuda_cpp", execution_target="local").id == legacy.id
    assert repository.latest_environment("cuda_cpp", execution_target="colab").id == remote.id
    assert (
        repository.latest_environment("triton_python", execution_target="colab").id
        == remote_triton.id
    )
    assert repository.latest_environment("torch_python", execution_target="colab") is None
    repository.acquire_lease(GPU_RESOURCE, "cpu-execution-routing-test")
    assert client.get("/api/environment").json()["id"] == legacy.id
    repository.update_execution_settings("colab", True)
    assert client.get("/api/environment").json()["id"] == remote.id
    assert client.get("/api/environment").json()["execution_target"] == "colab"
    unknown = client.get("/api/environment", params={"language": "torch_python"}).json()
    assert unknown["status"] == "unknown"
    assert unknown["execution_target"] == "colab"


def test_local_health_cannot_make_unprobed_or_unhealthy_colab_ready(execution_bundle) -> None:
    client, app, _, _, router, _ = execution_bundle
    repository = app.state.repository
    repository.acquire_lease(GPU_RESOURCE, "cpu-execution-routing-test")
    repository.save_environment(router.probe_environment())
    assert client.get("/api/ready").status_code == 200
    repository.update_execution_settings("colab", True)
    assert client.get("/api/ready").status_code == 503
    router.select_target("colab")
    repository.save_environment(replace(router.probe_environment(), healthy=False, error="offline"))
    assert client.get("/api/ready").status_code == 503
    assert client.get("/api/environment").json()["error"] == "offline"
    repository.update_execution_settings("local", False)
    assert client.get("/api/ready").status_code == 200


def test_probe_requires_live_worker_and_does_not_enqueue_when_offline(execution_bundle) -> None:
    client, app, local, colab, _, _ = execution_bundle
    response = client.post("/api/execution-settings/probe", json={"target": "colab"})
    assert response.status_code == 503
    with app.state.repository.session_factory() as session:
        assert session.scalars(select(ExecutionProbeRecord)).all() == []
    assert not local.calls and not colab.calls


def test_probe_api_waits_for_worker_and_returns_requested_target_without_selecting_it(
    execution_bundle, monkeypatch
) -> None:
    client, app, local, colab, _, worker = execution_bundle
    repository = app.state.repository
    repository.acquire_lease(GPU_RESOURCE, "cpu-execution-routing-test")
    queued = threading.Event()
    original_enqueue = repository.enqueue_execution_probe

    def signal_enqueue(target, language, timeout):
        probe_id = original_enqueue(target, language, timeout)
        queued.set()
        return probe_id

    monkeypatch.setattr(repository, "enqueue_execution_probe", signal_enqueue)
    with ThreadPoolExecutor(max_workers=1) as pool:
        response_future = pool.submit(
            client.post,
            "/api/execution-settings/probe",
            json={"target": "colab", "language": "triton_python"},
        )
        assert queued.wait(timeout=3)
        assert not local.calls and not colab.calls
        assert worker.process_next()
        response = response_future.result(timeout=3)
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["execution_target"] == "colab"
    assert response.json()["backend"] == "triton_python"
    assert client.get("/api/execution-settings").json()["target"] == "local"
    assert not local.calls
    assert colab.calls == [("probe_triton_environment", True)]


@pytest.mark.parametrize("language", ["cuda_cpp", "triton_python", "torch_python"])
def test_probe_is_queued_and_processed_serially_without_changing_selection(
    execution_bundle, language
) -> None:
    _, app, local, colab, _, worker = execution_bundle
    repository = app.state.repository
    job = app.state.jobs.submit(
        problem_id="vector-addition", action=JobAction.COMPILE, source=SOURCE
    )

    async def request_and_process():
        pending = asyncio.create_task(wait_for_probe(repository, "colab", language, 2))
        await asyncio.sleep(0)
        assert not local.calls and not colab.calls
        assert worker.process_next()
        assert repository.get_job(job.id).status == "queued"
        assert not any(call[0] == "compile" for call in local.calls + colab.calls)
        result = await pending
        assert result["healthy"] is True
        assert result["status"] == "healthy"
        assert result["execution_target"] == "colab"
        assert result["backend"] == language
        assert result["observed_at"]

    asyncio.run(request_and_process())
    assert repository.execution_settings()["target"] == "local"
    assert worker.process_next()
    assert repository.get_job(job.id).status == "succeeded"
    assert any(call[0] == "compile" for call in local.calls)
    assert not any(call[0] == "compile" for call in colab.calls)


def test_probe_failure_is_reported_without_changing_selected_target_or_running_jobs(
    execution_bundle, monkeypatch
) -> None:
    _, app, local, colab, _, worker = execution_bundle

    def unavailable(*args, **kwargs):
        raise RunnerUnavailable("CPU fixture connection failure")

    monkeypatch.setattr(colab, "probe_environment", unavailable)
    probe_id = app.state.repository.enqueue_execution_probe("colab", "cuda_cpp", 2)
    assert worker.process_next()
    probe = app.state.repository.execution_probe(probe_id)
    assert probe.status == "completed"
    assert probe.result_json["healthy"] is False
    assert probe.result_json["execution_target"] == "colab"
    assert app.state.repository.execution_settings()["target"] == "local"
    assert not local.calls


def test_expired_or_cancelled_probes_are_never_executed(execution_bundle) -> None:
    _, app, local, colab, _, worker = execution_bundle
    repository = app.state.repository
    expired = repository.enqueue_execution_probe("colab", "cuda_cpp", 10)
    cancelled = repository.enqueue_execution_probe("colab", "triton_python", 10)
    with repository.session_factory.begin() as session:
        session.get(ExecutionProbeRecord, expired).expires_at = utc_now() - timedelta(seconds=1)
    repository.cancel_execution_probe(cancelled)
    assert repository.execution_probe(cancelled).status == "cancelled"
    assert not worker.process_next()
    assert not local.calls and not colab.calls


def test_probe_claim_and_completion_cannot_overwrite_cancelled_request(execution_bundle) -> None:
    _, app, _, _, _, _ = execution_bundle
    repository = app.state.repository
    cancelled = repository.enqueue_execution_probe("colab", "cuda_cpp", 10)
    repository.cancel_execution_probe(cancelled)
    repository.complete_execution_probe(cancelled, {"healthy": True})
    assert repository.execution_probe(cancelled).result_json is None
    active = repository.enqueue_execution_probe("colab", "cuda_cpp", 10)
    assert repository.claim_execution_probe().id == active
    assert repository.claim_execution_probe() is None
    repository.complete_execution_probe(active, {"healthy": True})
    assert repository.execution_probe(active).status == "completed"


def test_probe_api_timeout_cancels_queued_request_and_never_executes_it_later(
    execution_bundle,
) -> None:
    client, app, local, colab, _, worker = execution_bundle
    repository = app.state.repository
    repository.acquire_lease(GPU_RESOURCE, "cpu-execution-routing-test")
    response = client.post("/api/execution-settings/probe", json={"target": "colab"})
    assert response.status_code == 504
    with repository.session_factory() as session:
        requests = session.scalars(select(ExecutionProbeRecord)).all()
    assert len(requests) == 1
    assert requests[0].status == "cancelled"
    assert not worker.process_next()
    assert not local.calls and not colab.calls


def test_cancelled_probe_waiter_cancels_queued_request(execution_bundle) -> None:
    _, app, local, colab, _, worker = execution_bundle
    repository = app.state.repository

    async def cancel_waiter():
        pending = asyncio.create_task(wait_for_probe(repository, "colab", "cuda_cpp", 10))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(cancel_waiter())
    with repository.session_factory() as session:
        requests = session.scalars(select(ExecutionProbeRecord)).all()
    assert len(requests) == 1
    assert requests[0].status == "cancelled"
    assert not worker.process_next()
    assert not local.calls and not colab.calls


def test_settings_are_shared_between_repository_instances(execution_bundle) -> None:
    _, app, _, _, _, _ = execution_bundle
    repository = app.state.repository
    independent_repository = Repository(repository.session_factory)
    repository.update_execution_settings("colab", True)
    assert independent_repository.execution_settings()["target"] == "colab"
    independent_repository.update_execution_settings("local", False)
    assert repository.execution_settings()["target"] == "local"


@pytest.mark.parametrize("target", ["local", "colab"])
@pytest.mark.parametrize("language", ["cuda_cpp", "triton_python", "torch_python"])
def test_failed_manual_probe_replaces_only_matching_target_language_health(
    execution_bundle, monkeypatch, target, language
) -> None:
    _, app, local, colab, router, worker = execution_bundle
    repository = app.state.repository
    methods = {
        "cuda_cpp": "probe_environment",
        "triton_python": "probe_triton_environment",
        "torch_python": "probe_torch_environment",
    }
    for previous_target in ("local", "colab"):
        router.select_target(previous_target)
        repository.save_environment(getattr(router, methods[language])())
    selected = local if target == "local" else colab

    def unavailable(**kwargs):
        raise RunnerUnavailable("CPU-only test: connection interrupted")

    monkeypatch.setattr(selected, methods[language], unavailable)
    probe_id = repository.enqueue_execution_probe(target, language, 10)
    assert worker.process_next()
    failed = repository.latest_environment(language, execution_target=target)
    assert failed is not None
    assert failed.healthy is False
    assert failed.backend == language
    assert failed.toolchain_json["execution_target"] == target
    assert failed.error == "CPU-only test: connection interrupted"
    if target == "colab":
        assert failed.cuda_image == "colab-native"
    result = repository.execution_probe(probe_id).result_json
    assert result["status"] == "unavailable"
    assert result["fingerprint"] == failed.fingerprint
    assert result["execution_target"] == target
    other = "colab" if target == "local" else "local"
    assert repository.latest_environment(language, execution_target=other).healthy is True
    assert repository.execution_settings()["target"] == "local"


@pytest.mark.parametrize("active_target", ["local", "colab"])
@pytest.mark.parametrize("unavailable_target", [None, "local", "colab"])
def test_cleanup_only_visits_the_explicitly_selected_target(
    execution_bundle, monkeypatch, active_target, unavailable_target
) -> None:
    _, _, local, colab, router, _ = execution_bundle
    calls = []

    def cleanup_local():
        calls.append("local")
        if unavailable_target == "local":
            raise RunnerUnavailable("CPU-only local offline")
        return ["old-local-container"]

    def cleanup_colab():
        calls.append("colab")
        if unavailable_target == "colab":
            raise RunnerUnavailable("CPU-only Colab offline")
        return ["old-remote-task"]

    monkeypatch.setattr(local, "cleanup_orphan_containers", cleanup_local, raising=False)
    monkeypatch.setattr(colab, "cleanup_orphan_containers", cleanup_colab, raising=False)
    router.select_target(active_target)
    if active_target == unavailable_target:
        with pytest.raises(RunnerUnavailable):
            router.cleanup_orphan_containers()
    else:
        removed = router.cleanup_orphan_containers()
        assert removed == (
            ["old-local-container"] if active_target == "local" else ["old-remote-task"]
        )
    assert calls == [active_target]
    assert router.target == active_target


@pytest.mark.parametrize("target", ["local", "colab"])
def test_gpu_orphan_cleanup_is_lazy_once_per_provider_and_retries_after_failure(
    execution_bundle, monkeypatch, target
):
    _, app, local, colab, _, worker = execution_bundle
    selected, other = (local, colab) if target == "local" else (colab, local)
    app.state.repository.update_execution_settings(target, target == "colab")
    cleanups = []
    monkeypatch.setattr(
        selected, "cleanup_orphan_containers", lambda: cleanups.append(target) or []
    )
    original_compile = selected.compile

    def submit():
        job = app.state.jobs.submit(
            problem_id="vector-addition", action=JobAction.RUN, source=SOURCE
        )
        assert worker.process_next()
        return app.state.repository.get_job(job.id)

    assert not cleanups
    assert submit().status == "succeeded"
    assert submit().status == "succeeded"
    assert cleanups == [target]
    assert any(call[0] == "probe_environment" for call in selected.calls)

    def unavailable(*args, **kwargs):
        raise RunnerUnavailable("provider disconnected")

    monkeypatch.setattr(selected, "compile", unavailable)
    assert submit().status == "system_error"
    monkeypatch.setattr(selected, "compile", original_compile)
    assert submit().status == "succeeded"
    assert cleanups == [target, target]
    assert not other.calls


@pytest.mark.parametrize("current_target", ["local", "colab"])
def test_orphan_job_cleanup_uses_job_snapshot_instead_of_current_setting(
    execution_bundle, current_target
) -> None:
    _, app, local, colab, router, worker = execution_bundle
    repository = app.state.repository
    local_job = app.state.jobs.submit(
        problem_id="vector-addition", action=JobAction.COMPILE, source=SOURCE
    )
    # Legacy jobs without a snapshot also belong to local execution.
    with repository.session_factory.begin() as session:
        session.get(JobRecord, local_job.id).payload_json = {}
    repository.update_execution_settings("colab", True)
    colab_job = app.state.jobs.submit(
        problem_id="vector-addition", action=JobAction.COMPILE, source=SOURCE
    )
    repository.update_execution_settings(current_target, current_target == "colab")
    router.select_target(current_target)

    worker._cleanup_job_ids([colab_job.id, local_job.id])

    assert local.calls == [("cleanup_task", Path(local_job.spool_path))]
    assert colab.calls == [("cleanup_task", Path(colab_job.spool_path))]
    assert not Path(local_job.spool_path).exists()
    assert not Path(colab_job.spool_path).exists()
    assert repository.execution_settings()["target"] == current_target


@pytest.mark.parametrize("language", ["cuda_cpp", "triton_python", "torch_python"])
def test_colab_connection_probe_calls_language_recovery_and_tags_result(
    execution_bundle, monkeypatch, language
) -> None:
    _, _, local, colab, router, _ = execution_bundle
    raw_probe = make_probe("same-hardware-fingerprint", backend=language)

    def recover(*, language):
        colab.calls.append(("recover", language))
        return raw_probe

    monkeypatch.setattr(colab, "recover", recover, raising=False)
    router.select_target("colab")
    result = router.probe_connection(language)
    assert colab.calls == [("recover", language)]
    assert not local.calls
    assert result.healthy is True
    assert result.backend == language
    assert result.toolchain["execution_target"] == "colab"
    assert result.fingerprint != raw_probe.fingerprint
    assert "execution_target" not in raw_probe.toolchain


def test_cli_components_read_persisted_target_without_contacting_devices(
    execution_bundle, monkeypatch
) -> None:
    _, app, local, colab, router, _ = execution_bundle
    app.state.repository.update_execution_settings("colab", True)
    monkeypatch.setattr(cli, "get_settings", lambda: app.state.settings)
    monkeypatch.setattr(cli, "ExecutionRouter", lambda settings: router)
    repository, _, selected_router = cli.build_components()
    try:
        assert selected_router.target == "colab"
        assert not local.calls and not colab.calls
    finally:
        repository.session_factory.kw["bind"].dispose()


@pytest.mark.parametrize("command", ["environment", "recover-runner"])
@pytest.mark.parametrize("target", ["local", "colab"])
def test_cli_target_override_routes_operation_without_changing_saved_selection(
    execution_bundle, monkeypatch, capsys, command, target
) -> None:
    _, app, local, colab, router, _ = execution_bundle
    saved_target = "colab" if target == "local" else "local"
    app.state.repository.update_execution_settings(saved_target, saved_target == "colab")
    router.select_target(saved_target)
    selected = local if target == "local" else colab
    calls = []

    def probe(**kwargs):
        calls.append(kwargs)
        return make_probe("CPU-only-cli-environment")

    monkeypatch.setattr(selected, "recover", probe, raising=False)
    monkeypatch.setattr(selected, "probe_environment", probe)
    monkeypatch.setattr(
        cli, "build_components", lambda: (app.state.repository, app.state.jobs, router)
    )
    monkeypatch.setattr("sys.argv", ["myleetgpu", command, "--target", target])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["toolchain"]["execution_target"] == target
    assert calls == (
        [{}] if command == "recover-runner" else [{"force": True, "ignore_circuit_breaker": True}]
    )
    assert app.state.repository.latest_environment("cuda_cpp", execution_target=target).healthy
    assert app.state.repository.execution_settings()["target"] == saved_target


@pytest.mark.parametrize("command", ["environment", "recover-runner"])
def test_cli_refuses_direct_probe_or_recovery_while_worker_holds_lease(
    execution_bundle, monkeypatch, capsys, command
) -> None:
    _, app, local, colab, router, _ = execution_bundle
    app.state.repository.acquire_lease(GPU_RESOURCE, "cpu-execution-routing-test")
    monkeypatch.setattr(
        cli, "build_components", lambda: (app.state.repository, app.state.jobs, router)
    )
    monkeypatch.setattr("sys.argv", ["myleetgpu", command, "--target", "colab"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "Worker 正在运行" in capsys.readouterr().err
    assert not local.calls and not colab.calls
