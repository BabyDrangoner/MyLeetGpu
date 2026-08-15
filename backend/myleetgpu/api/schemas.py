from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from myleetgpu.domain.jobs import JobAction


class JobCreate(BaseModel):
    problem_id: str = Field(min_length=1, max_length=128)
    action: JobAction
    source: str | None = None
    version_name: str | None = Field(default=None, max_length=120)
    notes: str | None = Field(default=None, max_length=4000)
    version_ids: list[str] | None = Field(default=None, max_length=8)
    allow_duplicate: bool = False


class JobResponse(BaseModel):
    id: str
    problem_id: str
    problem_revision: str
    action: str
    status: str
    phase: str
    progress: float
    source_hash: str | None
    result: dict[str, Any] | None
    error: dict[str, Any] | None
    diagnostics: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class DraftUpdate(BaseModel):
    source: str = Field(min_length=1, max_length=262_144)


class VersionUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    notes: str | None = Field(default=None, max_length=4000)

    @field_validator("name")
    @classmethod
    def nonblank_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("name cannot be blank")
        return value

    @model_validator(mode="after")
    def require_field(self) -> VersionUpdate:
        if not self.model_fields_set:
            raise ValueError("at least one field must be supplied")
        return self


class CompareRequest(BaseModel):
    problem_id: str = Field(min_length=1, max_length=128)
    version_ids: list[str] = Field(min_length=2, max_length=8)
    baseline_id: str
