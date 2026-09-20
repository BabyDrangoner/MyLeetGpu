"""Translate application failures to the stable public HTTP error envelope."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from myleetgpu.application.jobs import DuplicateSourceError, JobSubmissionError

LOGGER = logging.getLogger("myleetgpu.api")


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


async def submission_error(_request: Request, error: JobSubmissionError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": {"code": "invalid_request", "message": str(error)}},
    )


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


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, validation_error)
    app.add_exception_handler(JobSubmissionError, submission_error)
    app.add_exception_handler(DuplicateSourceError, duplicate_source)
    app.add_exception_handler(HTTPException, http_error)
    app.add_exception_handler(Exception, internal_error)
