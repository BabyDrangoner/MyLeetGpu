from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from myleetgpu.config import reset_settings_cache

from alembic import command
from alembic.config import Config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_kernel_language_migration_preserves_and_backfills_v1_data(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "legacy.db"
    database_url = f"sqlite:///{database.as_posix()}"
    monkeypatch.setenv("MYLEETGPU_DATABASE_URL_OVERRIDE", database_url)
    reset_settings_cache()
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))

    command.upgrade(config, "0004_restore_environment_fingerprint_index")
    now = datetime.now(UTC).isoformat()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO drafts (id, problem_id, source_code, updated_at) VALUES (?, ?, ?, ?)",
            ("draft-id", "vector-addition", "legacy draft", now),
        )
        connection.execute(
            """
            INSERT INTO environment_snapshots (
                id, fingerprint, healthy, cuda_image, telemetry_json, created_at, observed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("environment-id", "f" * 64, 1, "cuda:test", json.dumps({}), now, now),
        )
        connection.execute(
            """
            INSERT INTO jobs (
                id, problem_id, problem_revision, action, status, phase, progress,
                payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-id",
                "vector-addition",
                "1",
                "compile",
                "queued",
                "queued",
                0.0,
                json.dumps({}),
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO versions (
                id, problem_id, problem_revision, name, source_code, source_hash,
                compile_flags_json, correctness_status, environment_snapshot_id,
                suite_hash, protocol_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "version-id",
                "vector-addition",
                "1",
                "legacy",
                "void solve() {}",
                "s" * 64,
                json.dumps(["-O3"]),
                "passed",
                "environment-id",
                "h" * 64,
                "1",
                now,
            ),
        )

    command.upgrade(config, "head")

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT source_code, language FROM drafts WHERE id = 'draft-id'"
        ).fetchone() == ("legacy draft", "cuda_cpp")
        assert connection.execute("SELECT language FROM jobs WHERE id = 'job-id'").fetchone() == (
            "cuda_cpp",
        )
        assert connection.execute(
            "SELECT language FROM versions WHERE id = 'version-id'"
        ).fetchone() == ("cuda_cpp",)
        assert connection.execute(
            "SELECT backend, toolchain_json FROM environment_snapshots WHERE id = 'environment-id'"
        ).fetchone() == ("cuda_cpp", "{}")

        # The former one-column draft UNIQUE constraint has become composite.
        connection.execute(
            """
            INSERT INTO drafts (id, problem_id, language, source_code, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("triton-draft", "vector-addition", "triton_python", "python", now),
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM drafts WHERE problem_id = 'vector-addition'"
        ).fetchone() == (2,)

    reset_settings_cache()
