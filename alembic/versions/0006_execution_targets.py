"""Persist execution selection and serialize requested probes through the worker."""

import sqlalchemy as sa
from alembic import op

revision = "0006_execution_targets"
down_revision = "0005_kernel_languages"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "execution_settings",
        sa.Column("id", sa.String(16), primary_key=True),
        sa.Column("target", sa.String(16), nullable=False),
        sa.Column("colab_acknowledged", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "execution_probes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("target", sa.String(16), nullable=False),
        sa.Column("language", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("result_json", sa.JSON()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("execution_probes")
    op.drop_table("execution_settings")
