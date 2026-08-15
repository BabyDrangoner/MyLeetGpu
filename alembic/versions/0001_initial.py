"""Create drafts, jobs, environment snapshots, versions and benchmark runs."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "drafts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("problem_id", sa.String(128), nullable=False, unique=True),
        sa.Column("source_code", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "environment_snapshots",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("healthy", sa.Boolean(), nullable=False),
        sa.Column("gpu_name", sa.String(256)),
        sa.Column("compute_capability", sa.String(16)),
        sa.Column("driver_version", sa.String(64)),
        sa.Column("cuda_runtime_version", sa.String(64)),
        sa.Column("nvcc_version", sa.String(128)),
        sa.Column("cuda_image", sa.String(512), nullable=False),
        sa.Column("image_digest", sa.String(512)),
        sa.Column("cuda_arch", sa.String(16)),
        sa.Column("telemetry_json", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_environment_snapshots_fingerprint", "environment_snapshots", ["fingerprint"]
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("problem_id", sa.String(128), nullable=False),
        sa.Column("problem_revision", sa.String(64), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("phase", sa.String(64), nullable=False),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("source_hash", sa.String(64)),
        sa.Column("spool_path", sa.Text()),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("result_json", sa.JSON()),
        sa.Column("error_json", sa.JSON()),
        sa.Column("diagnostics", sa.Text()),
        sa.Column("worker_id", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_jobs_problem_id", "jobs", ["problem_id"])
    op.create_index("ix_jobs_queue", "jobs", ["status", "created_at"])
    op.create_table(
        "resource_leases",
        sa.Column("resource", sa.String(64), primary_key=True),
        sa.Column("owner", sa.String(128), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("problem_id", sa.String(128), nullable=False),
        sa.Column("problem_revision", sa.String(64), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("notes", sa.Text()),
        sa.Column("source_code", sa.Text(), nullable=False),
        sa.Column("source_hash", sa.String(64), nullable=False),
        sa.Column("compile_flags_json", sa.JSON(), nullable=False),
        sa.Column("correctness_status", sa.String(32), nullable=False),
        sa.Column(
            "environment_snapshot_id",
            sa.String(36),
            sa.ForeignKey("environment_snapshots.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("suite_hash", sa.String(64), nullable=False),
        sa.Column("protocol_version", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_versions_source_hash", "versions", ["source_hash"])
    op.create_index("ix_versions_problem_created", "versions", ["problem_id", "created_at"])
    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "version_id",
            sa.String(36),
            sa.ForeignKey("versions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "environment_snapshot_id",
            sa.String(36),
            sa.ForeignKey("environment_snapshots.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("suite_hash", sa.String(64), nullable=False),
        sa.Column("protocol_version", sa.String(32), nullable=False),
        sa.Column("compile_flags_json", sa.JSON(), nullable=False),
        sa.Column("input_sizes_json", sa.JSON(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("warmup", sa.Integer(), nullable=False),
        sa.Column("iterations", sa.Integer(), nullable=False),
        sa.Column("measurements_json", sa.JSON(), nullable=False),
        sa.Column("raw_samples_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_benchmark_runs_version_id", "benchmark_runs", ["version_id"])


def downgrade() -> None:
    op.drop_table("benchmark_runs")
    op.drop_table("versions")
    op.drop_table("resource_leases")
    op.drop_table("jobs")
    op.drop_table("environment_snapshots")
    op.drop_table("drafts")
