from __future__ import annotations

from fastapi import APIRouter, HTTPException

from myleetgpu.api.dependencies import JobsDependency, RepositoryDependency
from myleetgpu.api.presenters import job_response
from myleetgpu.api.schemas import JobCreate, JobResponse

router = APIRouter()


@router.post("/jobs", response_model=JobResponse, status_code=202)
def create_job(body: JobCreate, jobs: JobsDependency) -> JobResponse:
    return job_response(jobs.submit(**body.model_dump()))


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, repository: RepositoryDependency) -> JobResponse:
    record = repository.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job_response(record)
