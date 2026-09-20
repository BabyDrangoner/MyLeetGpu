from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query

from myleetgpu.api.dependencies import (
    CatalogDependency,
    RepositoryDependency,
    implementation_or_404,
)
from myleetgpu.api.presenters import draft_response
from myleetgpu.api.schemas import DraftUpdate, KernelLanguage

router = APIRouter()


@router.get("/drafts/{problem_id}")
def get_draft(
    problem_id: str,
    catalog: CatalogDependency,
    repository: RepositoryDependency,
    language: Annotated[KernelLanguage | None, Query()] = None,
) -> dict[str, Any]:
    implementation = implementation_or_404(catalog, problem_id, language)
    record = repository.get_draft(problem_id, implementation.language.value)
    if record is None:
        raise HTTPException(status_code=404, detail="尚未保存草稿")
    return draft_response(record)


@router.put("/drafts/{problem_id}")
def put_draft(
    problem_id: str,
    body: DraftUpdate,
    catalog: CatalogDependency,
    repository: RepositoryDependency,
) -> dict[str, Any]:
    implementation = implementation_or_404(catalog, problem_id, body.language)
    record = repository.upsert_draft(problem_id, body.source, implementation.language.value)
    return draft_response(record)
