"""Compose feature routers under the stable /api prefix."""

from fastapi import APIRouter

from myleetgpu.api.routers import drafts, environment, health, jobs, problems, versions

router = APIRouter(prefix="/api")
for feature in (health, problems, drafts, jobs, environment, versions):
    router.include_router(feature.router)
