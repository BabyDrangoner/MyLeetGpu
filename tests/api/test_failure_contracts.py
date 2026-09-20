"""Readiness fails closed and every HTTP exception retains the public envelope."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from fastapi.testclient import TestClient
from myleetgpu.api.dependencies import get_engine, get_repository
from myleetgpu.api.main import create_app
from myleetgpu.api.runtime_status import runner_circuit_error
from myleetgpu.config import Settings
from myleetgpu.domain.jobs import GPU_RESOURCE
from myleetgpu.infrastructure.repository import Repository

from tests.factories import make_probe

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def api(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "data",
        problems_dir=PROJECT_ROOT / "problems",
        database_url_override=f"sqlite:///{(tmp_path / 'api.db').as_posix()}",
        _env_file=None,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, app


@pytest.mark.parametrize(
    "failure", ["connect", "scalar", "execution_settings", "latest_environment", "has_active_lease"]
)
def test_readiness_database_failure_returns_503_without_further_database_access(
    api, failure: str
) -> None:
    client, app = api
    repository = Mock(spec=Repository, wraps=app.state.repository)
    app.dependency_overrides[get_repository] = lambda: repository
    private_error = RuntimeError("private database path and credentials")
    if failure in {"connect", "scalar"}:
        engine = MagicMock()
        if failure == "connect":
            engine.connect.side_effect = private_error
        else:
            connection = engine.connect.return_value.__enter__.return_value
            connection.execute.return_value.scalar.return_value = 0
        app.dependency_overrides[get_engine] = lambda: engine
    else:
        getattr(repository, failure).side_effect = private_error

    response = client.get("/api/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "database": False,
        "problems": 11,
        "runner": "unavailable",
        "worker_active": False,
        "runner_error": "数据库不可用，无法读取运行环境状态",
    }
    reads = ["execution_settings", "latest_environment", "has_active_lease"]
    expected = [] if failure in {"connect", "scalar"} else reads[: reads.index(failure) + 1]
    assert [call[0] for call in repository.mock_calls] == expected
    assert "private" not in response.text
    # Liveness is independent of the unavailable database and runner.
    assert client.get("/api/health").status_code == 200


def configure_healthy_target(app, target: str) -> Path:
    repository = app.state.repository
    repository.update_execution_settings(target, target == "colab")
    repository.save_environment(
        replace(make_probe(f"healthy-{target}"), toolchain={"execution_target": target})
    )
    repository.acquire_lease(GPU_RESOURCE, "failure-contract-worker")
    filename = "colab-runner-unhealthy.json" if target == "colab" else "runner-unhealthy.json"
    return app.state.settings.data_dir / filename


@pytest.mark.parametrize("target", ["local", "colab"])
@pytest.mark.parametrize(
    "payload", ["[]", "null", "0", '"text"', "false", "{}", '{"reason": []}', "not JSON"]
)
def test_malformed_circuit_marker_keeps_readiness_and_environment_unavailable(
    api, target: str, payload: str
) -> None:
    client, app = api
    path = configure_healthy_target(app, target)
    path.write_text(payload, encoding="utf-8")

    readiness = client.get("/api/ready")
    environment = client.get("/api/environment")

    assert readiness.status_code == 503
    assert readiness.json()["database"] is True
    assert readiness.json()["runner"] == "unavailable"
    assert "GPU Runner 已熔断" in readiness.json()["runner_error"]
    assert environment.status_code == 200
    assert environment.json()["healthy"] is False
    assert environment.json()["status"] == "unavailable"
    assert environment.json()["execution_target"] == target
    assert "GPU Runner 已熔断" in environment.json()["error"]


@pytest.mark.parametrize("target", ["local", "colab"])
def test_unreadable_circuit_marker_is_unavailable_not_healthy(
    api, target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, app = api
    marker = configure_healthy_target(app, target)
    original_read = Path.read_text

    def fail_marker_read(path: Path, *args, **kwargs):
        if path == marker:
            raise PermissionError("private filesystem diagnostics")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_marker_read)

    readiness = client.get("/api/ready")
    environment = client.get("/api/environment")

    assert readiness.status_code == 503
    assert environment.json()["healthy"] is False
    assert "private" not in readiness.text + environment.text


def test_absent_marker_is_healthy_and_valid_reason_is_preserved(api) -> None:
    client, app = api
    marker = configure_healthy_target(app, "local")

    assert runner_circuit_error(app.state.settings) is None
    assert client.get("/api/ready").status_code == 200

    marker.write_text('{"reason": "trusted GPU probe failed"}', encoding="utf-8")
    response = client.get("/api/ready")
    assert response.status_code == 503
    assert response.json()["runner_error"].startswith("trusted GPU probe failed")


@pytest.mark.parametrize(
    ("method", "path", "status_code", "code", "message"),
    [
        ("get", "/api/not-a-route", 404, "not_found", "Not Found"),
        ("post", "/api/problems", 405, "request_rejected", "Method Not Allowed"),
        ("get", "/api/jobs/missing", 404, "not_found", "任务不存在"),
    ],
)
def test_framework_and_explicit_http_exceptions_share_the_public_error_envelope(
    api, method: str, path: str, status_code: int, code: str, message: str
) -> None:
    client, _ = api
    response = getattr(client, method)(path)

    assert response.status_code == status_code
    assert response.json() == {"error": {"code": code, "message": message}}
    if status_code == 405:
        assert response.headers["Allow"] == "GET"


def test_request_validation_error_keeps_its_detailed_422_envelope(api) -> None:
    client, _ = api
    response = client.post("/api/jobs", json={})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["message"] == "请求参数无效"
    assert {tuple(detail["loc"]) for detail in error["details"]} == {
        ("body", "problem_id"),
        ("body", "action"),
    }
