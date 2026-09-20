"""HTTP composition contracts, independent dependencies, and resource ownership."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from myleetgpu.api import dependencies
from myleetgpu.api.main import create_app
from myleetgpu.config import Settings
from myleetgpu.infrastructure.repository import Repository

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        problems_dir=PROJECT_ROOT / "problems",
        database_url_override=f"sqlite:///{(tmp_path / 'api.db').as_posix()}",
        _env_file=None,
    )


def test_app_factory_does_not_open_resources_before_lifespan(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_engine = Mock(side_effect=AssertionError("factory must not create an engine"))
    monkeypatch.setattr(dependencies, "build_engine", build_engine)

    app = create_app(settings)

    assert app.openapi()["info"]["title"] == "MyLeetGpu API"
    assert not settings.data_dir.exists()
    build_engine.assert_not_called()


@pytest.mark.parametrize("phase", ["schema", "session_factory", "catalog", "repository", "jobs"])
def test_lifespan_disposes_engine_when_initialization_fails(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    engine = dependencies.build_engine(settings)
    dispose = Mock(wraps=engine.dispose)
    monkeypatch.setattr(engine, "dispose", dispose)
    monkeypatch.setattr(dependencies, "build_engine", lambda _: engine)
    failure = Mock(side_effect=RuntimeError(f"failed to initialize {phase}"))
    if phase == "schema":
        monkeypatch.setattr(dependencies.Base.metadata, "create_all", failure)
    elif phase == "catalog":
        monkeypatch.setattr(dependencies.ProblemCatalog, "load", failure)
    else:
        name = {
            "session_factory": "build_session_factory",
            "repository": "Repository",
            "jobs": "JobService",
        }[phase]
        monkeypatch.setattr(dependencies, name, failure)

    app = create_app(settings)
    with pytest.raises(RuntimeError, match=f"failed to initialize {phase}"), TestClient(app):
        pytest.fail("startup must fail before serving requests")

    dispose.assert_called_once_with()
    assert not hasattr(app.state, "repository")


@pytest.mark.parametrize("abnormal_exit", [False, True])
def test_lifespan_disposes_engine_on_normal_and_exceptional_exit(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, abnormal_exit: bool
) -> None:
    engine = dependencies.build_engine(settings)
    dispose = Mock(wraps=engine.dispose)
    monkeypatch.setattr(engine, "dispose", dispose)
    monkeypatch.setattr(dependencies, "build_engine", lambda _: engine)
    app = create_app(settings)

    def use_application() -> None:
        with TestClient(app) as client:
            assert client.get("/api/health").status_code == 200
            assert app.state.engine is engine
            assert app.state.settings is settings
            assert isinstance(app.state.repository, Repository)
            assert len(app.state.catalog) == 11
            assert app.state.jobs is not None
            dispose.assert_not_called()
            if abnormal_exit:
                raise RuntimeError("interrupted owner")

    if abnormal_exit:
        with pytest.raises(RuntimeError, match="interrupted owner"):
            use_application()
    else:
        use_application()

    dispose.assert_called_once_with()


def test_dependency_overrides_are_app_scoped_and_do_not_mutate_persisted_settings(
    settings: Settings,
) -> None:
    first_app = create_app(settings)
    second_app = create_app(settings)
    repository = Mock(spec=Repository)
    repository.execution_settings.return_value = {
        "target": "colab",
        "colab_acknowledged": True,
        "updated_at": None,
    }
    first_app.dependency_overrides[dependencies.get_repository] = lambda: repository
    first_app.dependency_overrides[dependencies.get_settings] = lambda: settings.model_copy(
        update={"colab_ssh_host": "isolated-test-colab"}
    )

    with TestClient(first_app) as first_client, TestClient(second_app) as second_client:
        overridden = first_client.get("/api/execution-settings").json()
        original = second_client.get("/api/execution-settings").json()

        assert overridden["target"] == "colab"
        assert overridden["colab"]["ssh_host"] == "isolated-test-colab"
        assert original["target"] == "local"
        assert original["colab"]["ssh_host"] == settings.colab_ssh_host
        assert first_app.state.repository.execution_settings()["target"] == "local"
        assert first_app.state.repository is not second_app.state.repository
        assert first_app.state.engine is not second_app.state.engine
    repository.execution_settings.assert_called_once_with()


def test_catalog_dependency_can_be_replaced_without_changing_routes(settings: Settings) -> None:
    app = create_app(settings)
    catalog = Mock(spec=dependencies.ProblemCatalog)
    catalog.list.return_value = []
    app.dependency_overrides[dependencies.get_catalog] = lambda: catalog

    with TestClient(app) as client:
        assert client.get("/api/problems").json() == {"items": [], "total": 0}
        assert len(app.state.catalog) == 11
    catalog.list.assert_called_once_with()


def test_job_service_dependency_is_replaceable_and_retains_http_error_contract(
    settings: Settings,
) -> None:
    app = create_app(settings)
    jobs = Mock(spec=dependencies.JobService)
    jobs.submit.side_effect = HTTPException(
        status_code=429, detail="busy", headers={"Retry-After": "3"}
    )
    app.dependency_overrides[dependencies.get_jobs] = lambda: jobs

    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            json={"problem_id": "vector-addition", "action": "run", "source": "test source"},
        )
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "3"
        assert response.json() == {"error": {"code": "request_rejected", "message": "busy"}}
        assert app.state.repository.counts()["jobs"] == 0
    jobs.submit.assert_called_once()


def test_unexpected_dependency_failure_uses_safe_public_error(settings: Settings) -> None:
    app = create_app(settings)

    def fail_repository() -> Repository:
        raise RuntimeError("private database credentials must not appear in a response")

    app.dependency_overrides[dependencies.get_repository] = fail_repository
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/jobs/missing")

    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "API 处理请求时发生内部错误"}
    }


def test_router_composition_preserves_paths_operation_ids_and_public_inputs(
    settings: Settings,
) -> None:
    schema = create_app(settings).openapi()
    operations = {
        (path, method): operation
        for path, methods in schema["paths"].items()
        for method, operation in methods.items()
    }
    expected = {
        ("/api/health", "get"): "health",
        ("/api/ready", "get"): "ready",
        ("/api/problems", "get"): "list_problems",
        ("/api/problems/{slug}", "get"): "get_problem",
        ("/api/drafts/{problem_id}", "get"): "get_draft",
        ("/api/drafts/{problem_id}", "put"): "put_draft",
        ("/api/jobs", "post"): "create_job",
        ("/api/jobs/{job_id}", "get"): "get_job",
        ("/api/environment", "get"): "environment",
        ("/api/execution-settings", "get"): "get_execution_settings",
        ("/api/execution-settings", "put"): "put_execution_settings",
        ("/api/execution-settings/probe", "post"): "test_execution_connection",
        ("/api/problems/{problem_id}/versions", "get"): "list_versions",
        ("/api/versions/duplicates", "get"): "duplicate_versions",
        ("/api/versions/{version_id}", "patch"): "update_version",
        ("/api/versions/{version_id}", "delete"): "delete_version",
        ("/api/versions/compare", "post"): "compare",
    }
    assert operations.keys() == expected.keys()
    for (path, method), operation in operations.items():
        path_identifier = (
            path.replace("/", "_").replace("{", "_").replace("}", "_").replace("-", "_")
        )
        assert operation["operationId"] == f"{expected[path, method]}{path_identifier}_{method}"
        parameters = {parameter["name"] for parameter in operation.get("parameters", [])}
        assert not parameters & {"repository", "settings", "catalog", "jobs", "engine"}

    assert operations["/api/jobs", "post"]["responses"]["202"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/JobResponse"}
    duplicate_inputs = operations["/api/versions/duplicates", "get"]["parameters"]
    assert {item["name"] for item in duplicate_inputs} == {
        "problem_id",
        "language",
        "source_hash",
        "source",
    }
