from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from myleetgpu.config import Settings
from myleetgpu.domain.benchmark import source_hash
from myleetgpu.domain.jobs import JobAction, JobStatus
from myleetgpu.domain.problems import ProblemCatalog
from myleetgpu.filesystem import ensure_mode
from myleetgpu.infrastructure.models import JobRecord
from myleetgpu.infrastructure.repository import Repository

MAX_SOURCE_BYTES = 256 * 1024


class JobSubmissionError(ValueError):
    pass


class DuplicateSourceError(JobSubmissionError):
    def __init__(self, duplicates: list[dict[str, str]]):
        super().__init__("相同源码已存在，请确认后再保存")
        self.duplicates = duplicates


class JobService:
    def __init__(self, settings: Settings, catalog: ProblemCatalog, repository: Repository):
        self.settings = settings
        self.catalog = catalog
        self.repository = repository

    def submit(
        self,
        *,
        problem_id: str,
        language: str | None = None,
        action: JobAction,
        source: str | None = None,
        version_name: str | None = None,
        notes: str | None = None,
        version_ids: list[str] | None = None,
        allow_duplicate: bool = False,
    ) -> JobRecord:
        try:
            problem = self.catalog.get(problem_id)
        except KeyError as error:
            raise JobSubmissionError(str(error)) from error
        selected_language = problem.default_language.value if language is None else language
        try:
            implementation = problem.get_implementation(selected_language)
        except KeyError as error:
            raise JobSubmissionError(f"题目不支持实现语言: {selected_language}") from error

        if action is JobAction.REBENCHMARK:
            selected = version_ids or []
            if len(selected) < 1 or len(selected) > 8 or len(set(selected)) != len(selected):
                raise JobSubmissionError("rebenchmark requires 1 to 8 unique version ids")
            versions = self.repository.get_versions(selected)
            if len(versions) != len(selected) or any(
                version.problem_id != problem_id for version in versions
            ):
                raise JobSubmissionError("every selected version must belong to the problem")
            if any(version.language != selected_language for version in versions):
                raise JobSubmissionError("rebenchmark versions must use one requested language")
        else:
            self._validate_source(source)
        submitted_hash = source_hash(source) if source is not None else None

        if action is JobAction.SAVE_VERSION:
            normalized_name = (version_name or "").strip()
            if not normalized_name:
                raise JobSubmissionError(
                    "version_name is required when saving a performance version"
                )
            if len(normalized_name) > 120:
                raise JobSubmissionError("version_name is longer than 120 characters")
            if notes is not None and len(notes) > 4000:
                raise JobSubmissionError("notes is longer than 4000 characters")
            duplicates = self.repository.find_duplicate_versions(
                problem_id, submitted_hash or "", selected_language
            )
            if duplicates and not allow_duplicate:
                raise DuplicateSourceError(
                    [{"id": item.id, "name": item.name} for item in duplicates]
                )
        else:
            normalized_name = None

        job_id = str(uuid.uuid4())
        spool_dir = self.settings.jobs_dir / job_id
        self._ensure_safe_spool_path(spool_dir)
        spool_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        stored_hash: str | None = None
        try:
            if source is not None:
                stored_hash = submitted_hash
                self._write_snapshot(spool_dir / f"source{implementation.source_suffix}", source)
            payload: dict[str, Any] = {}
            if normalized_name is not None:
                payload["version_name"] = normalized_name
                payload["notes"] = notes
                payload["allow_duplicate"] = allow_duplicate
            if version_ids:
                payload["version_ids"] = version_ids
            record = JobRecord(
                id=job_id,
                problem_id=problem_id,
                problem_revision=problem.manifest.revision,
                language=selected_language,
                action=action.value,
                status=JobStatus.QUEUED.value,
                phase="queued",
                progress=0.0,
                source_hash=stored_hash,
                spool_path=str(spool_dir),
                payload_json=payload,
            )
            return self.repository.add_job(record)
        except BaseException:
            shutil.rmtree(spool_dir, ignore_errors=True)
            raise

    def cleanup_stale_spool(self) -> list[str]:
        active = set()
        # Jobs are retained as metadata; only records with an active spool path protect a directory.
        with self.repository.session_factory() as session:
            rows = (
                session.query(JobRecord.spool_path).filter(JobRecord.spool_path.is_not(None)).all()
            )
            active = {Path(path).resolve() for (path,) in rows if path}
        removed: list[str] = []
        root = self.settings.jobs_dir.resolve()
        if not root.exists():
            return removed
        for candidate in root.iterdir():
            if candidate.is_dir() and candidate.resolve() not in active:
                self._ensure_safe_spool_path(candidate)
                shutil.rmtree(candidate)
                removed.append(candidate.name)
        return removed

    def _ensure_safe_spool_path(self, path: Path) -> None:
        root = self.settings.jobs_dir.resolve()
        resolved = path.resolve()
        if resolved.parent != root or not resolved.name:
            raise JobSubmissionError("job spool path escaped the configured jobs directory")

    @staticmethod
    def _validate_source(source: str | None) -> None:
        if source is None or not source.strip():
            raise JobSubmissionError("source is required")
        if "\x00" in source:
            raise JobSubmissionError("source cannot contain NUL bytes")
        if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise JobSubmissionError(f"source exceeds the {MAX_SOURCE_BYTES} byte limit")

    @staticmethod
    def _write_snapshot(path: Path, source: str) -> None:
        temporary = path.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(source.replace("\r\n", "\n").replace("\r", "\n"))
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
            ensure_mode(path, 0o600)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
