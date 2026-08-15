from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from myleetgpu.infrastructure.database import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


def uuid_string() -> str:
    return str(uuid.uuid4())


class DraftRecord(Base):
    __tablename__ = "drafts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    problem_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    source_code: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class EnvironmentSnapshotRecord(Base):
    __tablename__ = "environment_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    fingerprint: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    healthy: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    gpu_name: Mapped[str | None] = mapped_column(String(256))
    compute_capability: Mapped[str | None] = mapped_column(String(16))
    driver_version: Mapped[str | None] = mapped_column(String(64))
    cuda_runtime_version: Mapped[str | None] = mapped_column(String(64))
    nvcc_version: Mapped[str | None] = mapped_column(String(128))
    cuda_image: Mapped[str] = mapped_column(String(512), nullable=False)
    image_digest: Mapped[str | None] = mapped_column(String(512))
    cuda_arch: Mapped[str | None] = mapped_column(String(16))
    telemetry_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class JobRecord(Base):
    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_queue", "status", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    problem_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    problem_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    phase: Mapped[str] = mapped_column(String(64), nullable=False, default="queued")
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    source_hash: Mapped[str | None] = mapped_column(String(64))
    spool_path: Mapped[str | None] = mapped_column(Text)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    diagnostics: Mapped[str | None] = mapped_column(Text)
    worker_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ResourceLeaseRecord(Base):
    __tablename__ = "resource_leases"

    resource: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str] = mapped_column(String(128), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class VersionRecord(Base):
    __tablename__ = "versions"
    __table_args__ = (Index("ix_versions_problem_created", "problem_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    problem_id: Mapped[str] = mapped_column(String(128), nullable=False)
    problem_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    source_code: Mapped[str] = mapped_column(Text, nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    compile_flags_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    correctness_status: Mapped[str] = mapped_column(String(32), nullable=False)
    environment_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("environment_snapshots.id", ondelete="RESTRICT"), nullable=False
    )
    suite_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    protocol_version: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    environment: Mapped[EnvironmentSnapshotRecord] = relationship(lazy="joined")
    benchmark_runs: Mapped[list[BenchmarkRunRecord]] = relationship(
        back_populates="version", cascade="all, delete-orphan", lazy="selectin"
    )


class BenchmarkRunRecord(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    version_id: Mapped[str] = mapped_column(
        ForeignKey("versions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    environment_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("environment_snapshots.id", ondelete="RESTRICT"), nullable=False
    )
    suite_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    protocol_version: Mapped[str] = mapped_column(String(32), nullable=False)
    compile_flags_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    input_sizes_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    warmup: Mapped[int] = mapped_column(Integer, nullable=False)
    iterations: Mapped[int] = mapped_column(Integer, nullable=False)
    measurements_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    raw_samples_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    version: Mapped[VersionRecord] = relationship(back_populates="benchmark_runs")
    environment: Mapped[EnvironmentSnapshotRecord] = relationship(lazy="joined")
