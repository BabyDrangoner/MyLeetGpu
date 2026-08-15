"""Allow immutable benchmark snapshots to share an environment fingerprint."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_allow_repeated_environment_fingerprints"
down_revision: str | None = "0002_environment_observed_at"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NAMING_CONVENTION = {"uq": "uq_%(table_name)s_%(column_0_name)s"}
_CONSTRAINT_NAME = "uq_environment_snapshots_fingerprint"


def _fingerprint_is_unique() -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(
        constraint.get("column_names") == ["fingerprint"]
        for constraint in inspector.get_unique_constraints("environment_snapshots")
    )


def upgrade() -> None:
    # Early development databases used an unnamed UNIQUE constraint here.
    # Status observations are still deduplicated in Repository, but every
    # benchmark needs its own immutable row even when the fingerprint matches.
    if _fingerprint_is_unique():
        with op.batch_alter_table(
            "environment_snapshots", naming_convention=_NAMING_CONVENTION
        ) as batch_op:
            batch_op.drop_constraint(_CONSTRAINT_NAME, type_="unique")


def downgrade() -> None:
    if not _fingerprint_is_unique():
        with op.batch_alter_table("environment_snapshots") as batch_op:
            batch_op.create_unique_constraint(_CONSTRAINT_NAME, ["fingerprint"])
