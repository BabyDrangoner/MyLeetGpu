"""Make kernel language and runtime toolchain first-class data."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_kernel_languages"
down_revision: str | None = "0004_restore_environment_fingerprint_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NAMING_CONVENTION = {"uq": "uq_%(table_name)s_%(column_0_name)s"}


def upgrade() -> None:
    # SQLite cannot replace the old one-column UNIQUE constraint in place.
    # Batch mode preserves every existing CUDA draft while rebuilding the table.
    with op.batch_alter_table(
        "drafts", recreate="always", naming_convention=_NAMING_CONVENTION
    ) as batch_op:
        batch_op.add_column(
            sa.Column(
                "language",
                sa.String(32),
                nullable=False,
                server_default="cuda_cpp",
            )
        )
        batch_op.drop_constraint("uq_drafts_problem_id", type_="unique")
        batch_op.create_unique_constraint(
            "uq_drafts_problem_language", ["problem_id", "language"]
        )

    with op.batch_alter_table("jobs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "language",
                sa.String(32),
                nullable=False,
                server_default="cuda_cpp",
            )
        )

    with op.batch_alter_table("versions") as batch_op:
        batch_op.add_column(
            sa.Column(
                "language",
                sa.String(32),
                nullable=False,
                server_default="cuda_cpp",
            )
        )

    with op.batch_alter_table("environment_snapshots") as batch_op:
        batch_op.add_column(
            sa.Column(
                "backend",
                sa.String(32),
                nullable=False,
                server_default="cuda_cpp",
            )
        )
        batch_op.add_column(
            sa.Column(
                "toolchain_json",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'{}'"),
            )
        )


def downgrade() -> None:
    # The v1 schema can represent only one draft per problem. Keep the CUDA
    # draft, if present, and discard only Triton rows during an explicit
    # downgrade rather than allowing a non-deterministic UNIQUE violation.
    op.execute("DELETE FROM drafts WHERE language != 'cuda_cpp'")
    with op.batch_alter_table(
        "drafts", recreate="always", naming_convention=_NAMING_CONVENTION
    ) as batch_op:
        batch_op.drop_constraint("uq_drafts_problem_language", type_="unique")
        batch_op.create_unique_constraint("uq_drafts_problem_id", ["problem_id"])
        batch_op.drop_column("language")

    with op.batch_alter_table("environment_snapshots") as batch_op:
        batch_op.drop_column("toolchain_json")
        batch_op.drop_column("backend")
    with op.batch_alter_table("versions") as batch_op:
        batch_op.drop_column("language")
    with op.batch_alter_table("jobs") as batch_op:
        batch_op.drop_column("language")
