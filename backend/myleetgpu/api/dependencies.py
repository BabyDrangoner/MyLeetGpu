"""Application-scoped resources and overridable HTTP dependencies.

Only this module knows how resources are stored on ASGI state. Routers receive
their dependencies explicitly and never construct a database or GPU runner.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from sqlalchemy import Engine

from myleetgpu.application.jobs import JobService
from myleetgpu.config import Settings
from myleetgpu.domain.problems import Problem, ProblemCatalog, ProblemImplementation
from myleetgpu.infrastructure.database import Base, build_engine, build_session_factory
from myleetgpu.infrastructure.repository import Repository


def create_lifespan(
    settings: Settings,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.ensure_directories()
        engine = build_engine(settings)
        try:
            Base.metadata.create_all(engine)
            factory = build_session_factory(engine)
            catalog = ProblemCatalog(settings.problems_dir).load()
            repository = Repository(factory)
            jobs = JobService(settings, catalog, repository)
            # Preserve the established app.state contract for embedded clients.
            app.state.settings = settings
            app.state.engine = engine
            app.state.catalog = catalog
            app.state.repository = repository
            app.state.jobs = jobs
            yield
        finally:
            # Also runs when schema/catalog/service initialization or shutdown fails.
            engine.dispose()

    return lifespan


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_engine(request: Request) -> Engine:
    return request.app.state.engine


def get_repository(request: Request) -> Repository:
    return request.app.state.repository


def get_catalog(request: Request) -> ProblemCatalog:
    return request.app.state.catalog


def get_jobs(request: Request) -> JobService:
    return request.app.state.jobs


SettingsDependency = Annotated[Settings, Depends(get_settings)]
EngineDependency = Annotated[Engine, Depends(get_engine)]
RepositoryDependency = Annotated[Repository, Depends(get_repository)]
CatalogDependency = Annotated[ProblemCatalog, Depends(get_catalog)]
JobsDependency = Annotated[JobService, Depends(get_jobs)]


def problem_or_404(catalog: ProblemCatalog, slug: str) -> Problem:
    try:
        return catalog.get(slug)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="题目不存在") from error


def implementation_or_404(
    catalog: ProblemCatalog, slug: str, language: str | None
) -> ProblemImplementation:
    try:
        return catalog.get(slug).get_implementation(language)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="题目或实现语言不存在") from error
