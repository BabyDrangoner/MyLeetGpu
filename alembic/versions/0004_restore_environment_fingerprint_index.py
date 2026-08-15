"""Restore the non-unique environment fingerprint lookup index."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_restore_environment_fingerprint_index"
down_revision: str | None = "0003_allow_repeated_environment_fingerprints"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX_NAME = "ix_environment_snapshots_fingerprint"


def upgrade() -> None:
    indexes = sa.inspect(op.get_bind()).get_indexes("environment_snapshots")
    if not any(index.get("name") == _INDEX_NAME for index in indexes):
        op.create_index(
            _INDEX_NAME,
            "environment_snapshots",
            ["fingerprint"],
            unique=False,
        )


def downgrade() -> None:
    # Revision 0001 already defines this index for clean databases. This
    # restorative migration therefore intentionally leaves it in place.
    pass
