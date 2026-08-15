from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import Select, delete, exists, func, or_, select, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session, sessionmaker

from myleetgpu.domain.jobs import JobStatus, assert_transition
from myleetgpu.infrastructure.models import (
    BenchmarkRunRecord,
    DraftRecord,
    EnvironmentSnapshotRecord,
    JobRecord,
    ResourceLeaseRecord,
    VersionRecord,
    utc_now,
)
from myleetgpu.runner.models import EnvironmentProbe

_UNSET = object()


class Repository:
    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def get_draft(self, problem_id: str) -> DraftRecord | None:
        with self.session_factory() as session:
            return session.scalar(select(DraftRecord).where(DraftRecord.problem_id == problem_id))

    def upsert_draft(self, problem_id: str, source_code: str) -> DraftRecord:
        now = utc_now()
        with self.session_factory.begin() as session:
            statement = insert(DraftRecord).values(
                problem_id=problem_id, source_code=source_code, updated_at=now
            )
            statement = statement.on_conflict_do_update(
                index_elements=[DraftRecord.problem_id],
                set_={"source_code": source_code, "updated_at": now},
            )
            session.execute(statement)
            return session.scalar(select(DraftRecord).where(DraftRecord.problem_id == problem_id))

    def add_job(self, record: JobRecord) -> JobRecord:
        with self.session_factory.begin() as session:
            session.add(record)
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        with self.session_factory() as session:
            return session.get(JobRecord, job_id)

    def claim_next_job(self, worker_id: str) -> JobRecord | None:
        candidate = (
            select(JobRecord.id)
            .where(JobRecord.status == JobStatus.QUEUED.value)
            .order_by(JobRecord.created_at, JobRecord.id)
            .limit(1)
            .scalar_subquery()
        )
        now = utc_now()
        statement = (
            update(JobRecord)
            .where(JobRecord.id == candidate, JobRecord.status == JobStatus.QUEUED.value)
            .values(
                status=JobStatus.COMPILING.value,
                phase="compiling",
                progress=0.1,
                worker_id=worker_id,
                started_at=now,
            )
            .returning(JobRecord.id)
        )
        with self.session_factory.begin() as session:
            claimed_id = session.scalar(statement)
            if claimed_id is None:
                return None
            return session.get(JobRecord, claimed_id)

    def transition_job(
        self,
        job_id: str,
        status: JobStatus,
        *,
        phase: str | None = None,
        progress: float | None = None,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        diagnostics: str | None | object = _UNSET,
    ) -> JobRecord:
        with self.session_factory.begin() as session:
            record = session.get(JobRecord, job_id)
            if record is None:
                raise KeyError(f"unknown job: {job_id}")
            current = JobStatus(record.status)
            assert_transition(current, status)
            record.status = status.value
            record.phase = phase or status.value
            if progress is not None:
                record.progress = min(1.0, max(0.0, progress))
            if result is not None:
                record.result_json = result
            if error is not None:
                record.error_json = error
            if diagnostics is not _UNSET:
                record.diagnostics = diagnostics if isinstance(diagnostics, str) else None
            if status.terminal:
                record.completed_at = utc_now()
                record.progress = 1.0
                record.spool_path = None
            return record

    def fail_orphaned_jobs(self, worker_id: str) -> list[str]:
        active = {
            JobStatus.COMPILING.value,
            JobStatus.RUNNING.value,
            JobStatus.VALIDATING.value,
            JobStatus.BENCHMARKING.value,
        }
        with self.session_factory.begin() as session:
            rows = list(
                session.scalars(
                    select(JobRecord).where(
                        JobRecord.status.in_(active),
                        or_(JobRecord.worker_id.is_(None), JobRecord.worker_id != worker_id),
                    )
                )
            )
            now = utc_now()
            for record in rows:
                record.status = JobStatus.SYSTEM_ERROR.value
                record.phase = "worker_recovery"
                record.error_json = {
                    "code": "internal_error",
                    "message": "Worker restarted before the task completed",
                    "stage": "worker",
                    "retryable": True,
                    "details": {},
                }
                record.completed_at = now
                record.progress = 1.0
                record.spool_path = None
            return [record.id for record in rows]

    def acquire_lease(self, resource: str, owner: str, ttl_seconds: int = 30) -> bool:
        now = utc_now()
        expires = now + timedelta(seconds=ttl_seconds)
        with self.session_factory.begin() as session:
            session.execute(
                insert(ResourceLeaseRecord)
                .values(
                    resource=resource,
                    owner=owner,
                    expires_at=expires,
                    heartbeat_at=now,
                )
                .on_conflict_do_nothing(index_elements=[ResourceLeaseRecord.resource])
            )
            result = session.execute(
                update(ResourceLeaseRecord)
                .where(
                    ResourceLeaseRecord.resource == resource,
                    or_(
                        ResourceLeaseRecord.owner == owner,
                        ResourceLeaseRecord.expires_at < now,
                    ),
                )
                .values(owner=owner, expires_at=expires, heartbeat_at=now)
            )
            return result.rowcount == 1

    def release_lease(self, resource: str, owner: str) -> None:
        with self.session_factory.begin() as session:
            session.execute(
                delete(ResourceLeaseRecord).where(
                    ResourceLeaseRecord.resource == resource,
                    ResourceLeaseRecord.owner == owner,
                )
            )

    def has_active_lease(self, resource: str) -> bool:
        now = utc_now()
        with self.session_factory() as session:
            return (
                session.scalar(
                    select(ResourceLeaseRecord.resource).where(
                        ResourceLeaseRecord.resource == resource,
                        ResourceLeaseRecord.expires_at > now,
                    )
                )
                is not None
            )

    def owns_active_lease(self, resource: str, owner: str) -> bool:
        now = utc_now()
        with self.session_factory() as session:
            return (
                session.scalar(
                    select(ResourceLeaseRecord.resource).where(
                        ResourceLeaseRecord.resource == resource,
                        ResourceLeaseRecord.owner == owner,
                        ResourceLeaseRecord.expires_at > now,
                    )
                )
                is not None
            )

    def save_environment(
        self, probe: EnvironmentProbe, *, force_new: bool = False
    ) -> EnvironmentSnapshotRecord:
        with self.session_factory.begin() as session:
            observed_at = utc_now()
            if not force_new:
                existing = session.scalar(
                    select(EnvironmentSnapshotRecord)
                    .where(
                        EnvironmentSnapshotRecord.fingerprint == probe.fingerprint,
                        ~exists().where(
                            VersionRecord.environment_snapshot_id == EnvironmentSnapshotRecord.id
                        ),
                        ~exists().where(
                            BenchmarkRunRecord.environment_snapshot_id
                            == EnvironmentSnapshotRecord.id
                        ),
                    )
                    .order_by(EnvironmentSnapshotRecord.observed_at.desc())
                )
                if existing:
                    existing.healthy = probe.healthy
                    existing.gpu_name = probe.gpu_name
                    existing.compute_capability = probe.compute_capability
                    existing.driver_version = probe.driver_version
                    existing.cuda_runtime_version = probe.cuda_runtime_version
                    existing.nvcc_version = probe.nvcc_version
                    existing.cuda_image = probe.cuda_image
                    existing.image_digest = probe.image_digest
                    existing.cuda_arch = probe.cuda_arch
                    existing.telemetry_json = probe.telemetry
                    existing.error = probe.error
                    existing.observed_at = observed_at
                    session.flush()
                    return existing
            record = EnvironmentSnapshotRecord(
                fingerprint=probe.fingerprint,
                healthy=probe.healthy,
                gpu_name=probe.gpu_name,
                compute_capability=probe.compute_capability,
                driver_version=probe.driver_version,
                cuda_runtime_version=probe.cuda_runtime_version,
                nvcc_version=probe.nvcc_version,
                cuda_image=probe.cuda_image,
                image_digest=probe.image_digest,
                cuda_arch=probe.cuda_arch,
                telemetry_json=probe.telemetry,
                error=probe.error,
                observed_at=observed_at,
            )
            session.add(record)
            session.flush()
            return record

    def latest_environment(self) -> EnvironmentSnapshotRecord | None:
        with self.session_factory() as session:
            return session.scalar(
                select(EnvironmentSnapshotRecord).order_by(
                    EnvironmentSnapshotRecord.observed_at.desc(),
                    EnvironmentSnapshotRecord.created_at.desc(),
                )
            )

    def create_version_with_benchmark(
        self,
        *,
        problem_id: str,
        problem_revision: str,
        name: str,
        notes: str | None,
        source_code: str,
        source_hash: str,
        compile_flags: list[str],
        environment_id: str,
        suite_hash: str,
        protocol_version: str,
        input_sizes: list[str],
        seed: int,
        warmup: int,
        iterations: int,
        measurements: list[dict[str, Any]],
        raw_samples: list[dict[str, Any]],
    ) -> VersionRecord:
        with self.session_factory.begin() as session:
            version = VersionRecord(
                problem_id=problem_id,
                problem_revision=problem_revision,
                name=name,
                notes=notes,
                source_code=source_code,
                source_hash=source_hash,
                compile_flags_json=compile_flags,
                correctness_status="passed",
                environment_snapshot_id=environment_id,
                suite_hash=suite_hash,
                protocol_version=protocol_version,
            )
            session.add(version)
            session.flush()
            run = BenchmarkRunRecord(
                version_id=version.id,
                environment_snapshot_id=environment_id,
                suite_hash=suite_hash,
                protocol_version=protocol_version,
                compile_flags_json=compile_flags,
                input_sizes_json=input_sizes,
                seed=seed,
                warmup=warmup,
                iterations=iterations,
                measurements_json=measurements,
                raw_samples_json=raw_samples,
            )
            session.add(run)
            session.flush()
            version.benchmark_runs = [run]
            return version

    def add_benchmark_runs(self, rows: list[dict[str, Any]]) -> list[BenchmarkRunRecord]:
        records: list[BenchmarkRunRecord] = []
        with self.session_factory.begin() as session:
            for row in rows:
                record = BenchmarkRunRecord(**row)
                session.add(record)
                records.append(record)
            session.flush()
        return records

    def list_versions(self, problem_id: str) -> list[VersionRecord]:
        with self.session_factory() as session:
            statement: Select[tuple[VersionRecord]] = (
                select(VersionRecord)
                .where(VersionRecord.problem_id == problem_id)
                .order_by(VersionRecord.created_at.desc(), VersionRecord.id)
            )
            return list(session.scalars(statement).unique())

    def get_version(self, version_id: str) -> VersionRecord | None:
        with self.session_factory() as session:
            return session.scalar(select(VersionRecord).where(VersionRecord.id == version_id))

    def get_versions(self, version_ids: list[str]) -> list[VersionRecord]:
        with self.session_factory() as session:
            records = list(
                session.scalars(
                    select(VersionRecord).where(VersionRecord.id.in_(version_ids))
                ).unique()
            )
        by_id = {record.id: record for record in records}
        return [by_id[item] for item in version_ids if item in by_id]

    def find_duplicate_versions(self, problem_id: str, source_hash: str) -> list[VersionRecord]:
        with self.session_factory() as session:
            return list(
                session.scalars(
                    select(VersionRecord)
                    .where(
                        VersionRecord.problem_id == problem_id,
                        VersionRecord.source_hash == source_hash,
                    )
                    .order_by(VersionRecord.created_at.desc())
                ).unique()
            )

    def update_version(
        self,
        version_id: str,
        *,
        name: str | None = None,
        notes: str | None = None,
        update_notes: bool = False,
    ) -> VersionRecord | None:
        with self.session_factory.begin() as session:
            record = session.get(VersionRecord, version_id)
            if record is None:
                return None
            if name is not None:
                record.name = name
            if update_notes:
                record.notes = notes
            return record

    def delete_version(self, version_id: str) -> bool:
        with self.session_factory.begin() as session:
            result = session.execute(delete(VersionRecord).where(VersionRecord.id == version_id))
            return result.rowcount == 1

    def counts(self) -> dict[str, int]:
        with self.session_factory() as session:
            return {
                "drafts": session.scalar(select(func.count()).select_from(DraftRecord)) or 0,
                "jobs": session.scalar(select(func.count()).select_from(JobRecord)) or 0,
                "versions": session.scalar(select(func.count()).select_from(VersionRecord)) or 0,
                "benchmark_runs": session.scalar(
                    select(func.count()).select_from(BenchmarkRunRecord)
                )
                or 0,
            }
