from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from myleetgpu.api.dependencies import CatalogDependency, problem_or_404

router = APIRouter()


@router.get("/problems")
def list_problems(catalog: CatalogDependency) -> dict[str, Any]:
    items = [problem.public_summary() for problem in catalog.list()]
    return {"items": items, "total": len(items)}


@router.get("/problems/{slug}")
def get_problem(slug: str, catalog: CatalogDependency) -> dict[str, Any]:
    return problem_or_404(catalog, slug).public_detail()
