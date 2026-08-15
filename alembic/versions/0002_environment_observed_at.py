"""Track the latest observation separately from immutable snapshot creation."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_environment_observed_at"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "environment_snapshots",
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute("UPDATE environment_snapshots SET observed_at = created_at")
    with op.batch_alter_table("environment_snapshots") as batch_op:
        batch_op.alter_column("observed_at", existing_type=sa.DateTime(timezone=True), nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("environment_snapshots") as batch_op:
        batch_op.drop_column("observed_at")
