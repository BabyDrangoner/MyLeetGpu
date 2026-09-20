from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Response

from myleetgpu.api.dependencies import (
    CatalogDependency,
    RepositoryDependency,
    implementation_or_404,
    problem_or_404,
)
from myleetgpu.api.presenters import version_response
from myleetgpu.api.schemas import CompareRequest, KernelLanguage, VersionUpdate
from myleetgpu.application.compare import ComparisonError, compare_versions
from myleetgpu.domain.benchmark import source_hash

router = APIRouter()


@router.get("/problems/{problem_id}/versions")
def list_versions(
    problem_id: str,
    catalog: CatalogDependency,
    repository: RepositoryDependency,
    language: Annotated[KernelLanguage | None, Query()] = None,
) -> dict[str, Any]:
    problem_or_404(catalog, problem_id)
    if language is not None:
        implementation_or_404(catalog, problem_id, language)
    items = [version_response(item) for item in repository.list_versions(problem_id, language)]
    return {"items": items, "total": len(items)}


@router.get("/versions/duplicates")
def duplicate_versions(
    catalog: CatalogDependency,
    repository: RepositoryDependency,
    problem_id: str = Query(min_length=1, max_length=128),
    language: Annotated[KernelLanguage | None, Query()] = None,
    source_hash_value: str | None = Query(default=None, alias="source_hash", min_length=64),
    source: str | None = Query(default=None, max_length=262_144),
) -> dict[str, Any]:
    implementation = implementation_or_404(catalog, problem_id, language)
    digest = source_hash_value or (source_hash(source) if source is not None else None)
    if digest is None:
        raise HTTPException(status_code=422, detail="source_hash 或 source 必填")
    items = repository.find_duplicate_versions(problem_id, digest, implementation.language.value)
    return {
        "duplicate": bool(items),
        "items": [
            {"id": item.id, "name": item.name, "created_at": item.created_at} for item in items
        ],
    }


@router.patch("/versions/{version_id}")
def update_version(
    version_id: str, body: VersionUpdate, repository: RepositoryDependency
) -> dict[str, Any]:
    record = repository.update_version(
        version_id,
        name=body.name.strip() if body.name is not None else None,
        notes=body.notes,
        update_notes="notes" in body.model_fields_set,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="版本不存在")
    return version_response(record)


@router.delete("/versions/{version_id}", status_code=204)
def delete_version(
    version_id: str,
    repository: RepositoryDependency,
    confirmed: bool = Query(default=False),
) -> Response:
    if not confirmed:
        raise HTTPException(status_code=409, detail="删除版本需要二次确认")
    if not repository.delete_version(version_id):
        raise HTTPException(status_code=404, detail="版本不存在")
    return Response(status_code=204)


@router.post("/versions/compare")
def compare(
    body: CompareRequest,
    catalog: CatalogDependency,
    repository: RepositoryDependency,
) -> dict[str, Any]:
    problem_or_404(catalog, body.problem_id)
    try:
        return compare_versions(repository, **body.model_dump())
    except ComparisonError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
