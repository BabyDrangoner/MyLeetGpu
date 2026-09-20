from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Query, Response, status
from sqlalchemy import text

from myleetgpu.api.dependencies import (
    CatalogDependency,
    EngineDependency,
    RepositoryDependency,
    SettingsDependency,
)
from myleetgpu.api.runtime_status import runner_circuit_error
from myleetgpu.api.schemas import KernelLanguage
from myleetgpu.domain.jobs import GPU_RESOURCE

router = APIRouter()


@router.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "myleetgpu-api",
        "time": datetime.now(UTC).isoformat(),
    }


@router.get("/ready")
def ready(
    response: Response,
    repository: RepositoryDependency,
    engine: EngineDependency,
    catalog: CatalogDependency,
    settings: SettingsDependency,
    language: Annotated[KernelLanguage, Query()] = "cuda_cpp",
) -> dict[str, Any]:
    database_ok = False
    target = "local"
    environment = None
    worker_active = False
    try:
        with engine.connect() as connection:
            database_ok = connection.execute(text("SELECT 1")).scalar() == 1
        if database_ok:
            target = (
                "cpu"
                if language in {"cpp", "python"}
                else repository.execution_settings()["target"]
            )
            environment = repository.latest_environment(language, execution_target=target)
            worker_active = repository.has_active_lease(GPU_RESOURCE)
    except Exception:
        database_ok = False
    if not database_ok:
        # A failed database check must not be followed by more database reads.
        # Keep readiness useful to operators without exposing storage errors.
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "database": False,
            "problems": len(catalog),
            "runner": "unavailable",
            "worker_active": False,
            "runner_error": "数据库不可用，无法读取运行环境状态",
        }
    circuit_error = runner_circuit_error(settings, target, language)
    runner_ok = bool(
        environment and environment.healthy and worker_active and circuit_error is None
    )
    ready_now = database_ok and len(catalog) > 0 and runner_ok
    if not ready_now:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if ready_now else "not_ready",
        "database": database_ok,
        "problems": len(catalog),
        "runner": "healthy" if runner_ok else "unavailable",
        "worker_active": worker_active,
        "runner_error": circuit_error
        or (environment.error if environment else "worker has not probed this runtime yet"),
    }
