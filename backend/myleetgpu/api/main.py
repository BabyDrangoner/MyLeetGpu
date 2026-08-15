from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from myleetgpu.api.schemas import (
    CompareRequest,
    DraftUpdate,
    JobCreate,
    JobResponse,
    VersionUpdate,
)
from myleetgpu.application.compare import ComparisonError, compare_versions
from myleetgpu.application.jobs import DuplicateSourceError, JobService, JobSubmissionError
from myleetgpu.config import Settings, get_settings
from myleetgpu.domain.benchmark import source_hash
from myleetgpu.domain.jobs import GPU_RESOURCE
from myleetgpu.domain.problems import ProblemCatalog
from myleetgpu.infrastructure.database import Base, build_engine, build_session_factory
from myleetgpu.infrastructure.logging import configure_logging
from myleetgpu.infrastructure.models import (
    BenchmarkRunRecord,
    EnvironmentSnapshotRecord,
    JobRecord,
    VersionRecord,
)
from myleetgpu.infrastructure.repository import Repository

LOGGER = logging.getLogger("myleetgpu.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    selected_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        selected_settings.ensure_directories()
        engine = build_engine(selected_settings)
        Base.metadata.create_all(engine)
        factory = build_session_factory(engine)
        catalog = ProblemCatalog(selected_settings.problems_dir).load()
        repository = Repository(factory)
        app.state.settings = selected_settings
        app.state.engine = engine
        app.state.catalog = catalog
        app.state.repository = repository
        app.state.jobs = JobService(selected_settings, catalog, repository)
        yield
        engine.dispose()

    app = FastAPI(
        title="MyLeetGpu API",
        version="0.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver", "api"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "请求参数无效",
                    "details": jsonable_encoder(error.errors()),
                }
            },
        )

    @app.exception_handler(JobSubmissionError)
    async def submission_error(_request: Request, error: JobSubmissionError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": {"code": "invalid_request", "message": str(error)}},
        )

    @app.exception_handler(DuplicateSourceError)
    async def duplicate_source(_request: Request, error: DuplicateSourceError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": {
                    "code": "duplicate_source",
                    "message": str(error),
                    "duplicates": error.duplicates,
                }
            },
        )

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, error: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            headers=error.headers,
            content={
                "error": {
                    "code": "not_found" if error.status_code == 404 else "request_rejected",
                    "message": str(error.detail),
                }
            },
        )

    @app.exception_handler(Exception)
    async def internal_error(request: Request, error: Exception) -> JSONResponse:
        LOGGER.exception("unhandled API error", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "API 处理请求时发生内部错误",
                }
            },
        )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "myleetgpu-api",
            "time": datetime.now(UTC).isoformat(),
        }

    @app.get("/api/ready")
    def ready(request: Request, response: Response) -> dict[str, Any]:
        repository = _repository(request)
        database_ok = False
        try:
            with request.app.state.engine.connect() as connection:
                database_ok = connection.execute(text("SELECT 1")).scalar() == 1
        except Exception:
            database_ok = False
        environment = repository.latest_environment()
        worker_active = repository.has_active_lease(GPU_RESOURCE)
        circuit_error = _runner_circuit_error(request.app.state.settings)
        runner_ok = bool(
            environment and environment.healthy and worker_active and circuit_error is None
        )
        ready_now = database_ok and len(request.app.state.catalog) > 0 and runner_ok
        if not ready_now:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "ready" if ready_now else "not_ready",
            "database": database_ok,
            "problems": len(request.app.state.catalog),
            "runner": "healthy" if runner_ok else "unavailable",
            "worker_active": worker_active,
            "runner_error": circuit_error
            or (environment.error if environment else "worker has not probed the GPU yet"),
        }

    @app.get("/api/problems")
    def list_problems(request: Request) -> dict[str, Any]:
        items = [problem.public_summary() for problem in request.app.state.catalog.list()]
        return {"items": items, "total": len(items)}

    @app.get("/api/problems/{slug}")
    def get_problem(slug: str, request: Request) -> dict[str, Any]:
        try:
            return request.app.state.catalog.get(slug).public_detail()
        except KeyError as error:
            raise HTTPException(status_code=404, detail="题目不存在") from error

    @app.get("/api/drafts/{problem_id}")
    def get_draft(problem_id: str, request: Request) -> dict[str, Any]:
        _problem_or_404(request, problem_id)
        record = _repository(request).get_draft(problem_id)
        if record is None:
            raise HTTPException(status_code=404, detail="尚未保存草稿")
        return {
            "problem_id": record.problem_id,
            "source": record.source_code,
            "updated_at": record.updated_at,
        }

    @app.put("/api/drafts/{problem_id}")
    def put_draft(problem_id: str, body: DraftUpdate, request: Request) -> dict[str, Any]:
        _problem_or_404(request, problem_id)
        record = _repository(request).upsert_draft(problem_id, body.source)
        return {
            "problem_id": record.problem_id,
            "source": record.source_code,
            "updated_at": record.updated_at,
        }

    @app.post("/api/jobs", response_model=JobResponse, status_code=202)
    def create_job(body: JobCreate, request: Request) -> JobResponse:
        record = request.app.state.jobs.submit(**body.model_dump())
        return _job_response(record)

    @app.get("/api/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str, request: Request) -> JobResponse:
        record = _repository(request).get_job(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return _job_response(record)

    @app.get("/api/environment")
    def environment(request: Request) -> dict[str, Any]:
        record = _repository(request).latest_environment()
        if record is None:
            return {
                "status": "unknown",
                "healthy": False,
                "error": "Worker 尚未完成环境探测",
                "telemetry": _unavailable_telemetry(),
            }
        payload = _environment_response(record)
        worker_active = _repository(request).has_active_lease(GPU_RESOURCE)
        circuit_error = _runner_circuit_error(request.app.state.settings)
        payload["worker_active"] = worker_active
        if not worker_active or circuit_error:
            payload["healthy"] = False
            payload["status"] = "unavailable"
            payload["error"] = circuit_error or "GPU Worker 未运行或租约已过期"
        return payload

    @app.get("/api/problems/{problem_id}/versions")
    def list_versions(problem_id: str, request: Request) -> dict[str, Any]:
        _problem_or_404(request, problem_id)
        items = [_version_response(item) for item in _repository(request).list_versions(problem_id)]
        return {"items": items, "total": len(items)}

    @app.get("/api/versions/duplicates")
    def duplicate_versions(
        request: Request,
        problem_id: str = Query(min_length=1, max_length=128),
        source_hash_value: str | None = Query(default=None, alias="source_hash", min_length=64),
        source: str | None = Query(default=None, max_length=262_144),
    ) -> dict[str, Any]:
        _problem_or_404(request, problem_id)
        digest = source_hash_value or (source_hash(source) if source is not None else None)
        if digest is None:
            raise HTTPException(status_code=422, detail="source_hash 或 source 必填")
        items = _repository(request).find_duplicate_versions(problem_id, digest)
        return {
            "duplicate": bool(items),
            "items": [
                {"id": item.id, "name": item.name, "created_at": item.created_at} for item in items
            ],
        }

    @app.patch("/api/versions/{version_id}")
    def update_version(version_id: str, body: VersionUpdate, request: Request) -> dict[str, Any]:
        record = _repository(request).update_version(
            version_id,
            name=body.name.strip() if body.name is not None else None,
            notes=body.notes,
            update_notes="notes" in body.model_fields_set,
        )
        if record is None:
            raise HTTPException(status_code=404, detail="版本不存在")
        return _version_response(record)

    @app.delete("/api/versions/{version_id}", status_code=204)
    def delete_version(
        version_id: str,
        request: Request,
        confirmed: bool = Query(default=False),
    ) -> Response:
        if not confirmed:
            raise HTTPException(status_code=409, detail="删除版本需要二次确认")
        if not _repository(request).delete_version(version_id):
            raise HTTPException(status_code=404, detail="版本不存在")
        return Response(status_code=204)

    @app.post("/api/versions/compare")
    def compare(body: CompareRequest, request: Request) -> dict[str, Any]:
        _problem_or_404(request, body.problem_id)
        try:
            return compare_versions(_repository(request), **body.model_dump())
        except ComparisonError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


def _repository(request: Request) -> Repository:
    return request.app.state.repository


def _problem_or_404(request: Request, slug: str) -> None:
    try:
        request.app.state.catalog.get(slug)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="题目不存在") from error


def _job_response(record: JobRecord) -> JobResponse:
    return JobResponse(
        id=record.id,
        problem_id=record.problem_id,
        problem_revision=record.problem_revision,
        action=record.action,
        status=record.status,
        phase=record.phase,
        progress=record.progress,
        source_hash=record.source_hash,
        result=record.result_json,
        error=record.error_json,
        diagnostics=record.diagnostics,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
    )


def _version_response(record: VersionRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "problem_id": record.problem_id,
        "problem_revision": record.problem_revision,
        "name": record.name,
        "notes": record.notes,
        "source_code": record.source_code,
        "source_hash": record.source_hash,
        "compile_flags": record.compile_flags_json,
        "correctness_status": record.correctness_status,
        "suite_hash": record.suite_hash,
        "protocol_version": record.protocol_version,
        "created_at": record.created_at,
        "environment": _environment_response(record.environment),
        "benchmark_runs": [_benchmark_response(item) for item in record.benchmark_runs],
    }


def _benchmark_response(record: BenchmarkRunRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "suite_hash": record.suite_hash,
        "protocol_version": record.protocol_version,
        "compile_flags": record.compile_flags_json,
        "input_sizes": record.input_sizes_json,
        "seed": record.seed,
        "warmup": record.warmup,
        "iterations": record.iterations,
        "measurements": record.measurements_json,
        "raw_samples": record.raw_samples_json,
        "environment": _environment_response(record.environment),
        "created_at": record.created_at,
    }


def _environment_response(record: EnvironmentSnapshotRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "status": "healthy" if record.healthy else "unavailable",
        "healthy": record.healthy,
        "fingerprint": record.fingerprint,
        "gpu_name": record.gpu_name,
        "compute_capability": record.compute_capability,
        "driver_version": record.driver_version,
        "cuda_runtime_version": record.cuda_runtime_version,
        "nvcc_version": record.nvcc_version,
        "cuda_image": record.cuda_image,
        "image_digest": record.image_digest,
        "cuda_arch": record.cuda_arch,
        "telemetry": {**_unavailable_telemetry(), **record.telemetry_json},
        "error": record.error,
        "observed_at": record.observed_at,
    }


def _unavailable_telemetry() -> dict[str, None]:
    return {
        "temperature_c": None,
        "power_w": None,
        "sm_clock_mhz": None,
        "gpu_busy_percent": None,
    }


def _runner_circuit_error(settings: Settings) -> str | None:
    path = settings.data_dir / "runner-unhealthy.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        reason = str(payload.get("reason", "GPU Runner 已熔断"))
    except (OSError, ValueError, TypeError):
        reason = "GPU Runner 已熔断"
    return f"{reason}；运行 make doctor 并执行 make recover-runner"


app = create_app()


if __name__ == "__main__":
    runtime_settings = get_settings()
    configure_logging(runtime_settings.log_level)
    uvicorn.run(
        "myleetgpu.api.main:app",
        host=runtime_settings.api_host,
        port=runtime_settings.api_port,
        log_level=runtime_settings.log_level.lower(),
    )
