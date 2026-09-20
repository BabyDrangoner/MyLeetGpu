from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from myleetgpu.api.dependencies import create_lifespan
from myleetgpu.api.errors import register_exception_handlers

# Compatibility for existing embedded callers; new code imports the presenter directly.
from myleetgpu.api.presenters import job_response as _job_response  # noqa: F401
from myleetgpu.api.routers import router
from myleetgpu.config import Settings, get_settings
from myleetgpu.infrastructure.logging import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    """Compose the HTTP adapter without opening resources until ASGI startup."""
    app = FastAPI(
        title="MyLeetGpu API",
        version="0.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=create_lifespan(settings or get_settings()),
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver", "api"],
    )
    register_exception_handlers(app)
    app.include_router(router)
    return app


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
