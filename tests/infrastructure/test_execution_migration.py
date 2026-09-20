"""CPU-only migration checks for the online execution-target selector."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from myleetgpu.api.main import _job_response
from myleetgpu.config import Settings, reset_settings_cache
from myleetgpu.infrastructure.database import build_engine, build_session_factory
from myleetgpu.infrastructure.repository import Repository

from alembic import command
from alembic.config import Config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_execution_migration_preserves_legacy_jobs_and_supports_downgrade(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "execution-migration.db"
    database_url = f"sqlite:///{database.as_posix()}"
    monkeypatch.setenv("MYLEETGPU_DATABASE_URL_OVERRIDE", database_url)
    reset_settings_cache()
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    legacy_payload = {"version_name": "before-execution-targets"}
    try:
        command.upgrade(config, "0005_kernel_languages")
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(database) as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, problem_id, problem_revision, language, action, status,
                    phase, progress, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "legacy-job",
                    "vector-addition",
                    "1",
                    "triton_python",
                    "compile",
                    "queued",
                    "queued",
                    0.0,
                    json.dumps(legacy_payload),
                    now,
                ),
            )

        command.upgrade(config, "0006_execution_targets")
        engine = build_engine(Settings(database_url_override=database_url, _env_file=None))
        try:
            repository = Repository(build_session_factory(engine))
            assert repository.execution_settings() == {
                "target": "local",
                "colab_acknowledged": False,
                "updated_at": None,
            }
            job = repository.get_job("legacy-job")
            assert job is not None
            assert job.language == "triton_python"
            assert job.payload_json == legacy_payload
            assert _job_response(job).execution_target == "local"
            repository.update_execution_settings("colab", True)
            assert repository.execution_settings()["target"] == "colab"
            probe_id = repository.enqueue_execution_probe("colab", "cuda_cpp", 10)
            assert repository.execution_probe(probe_id).status == "queued"
        finally:
            engine.dispose()

        command.downgrade(config, "0005_kernel_languages")
        with sqlite3.connect(database) as connection:
            tables = {
                name
                for (name,) in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            assert "execution_settings" not in tables
            assert "execution_probes" not in tables
            legacy = connection.execute(
                "SELECT language, payload_json, status FROM jobs WHERE id = 'legacy-job'"
            ).fetchone()
            assert legacy == ("triton_python", json.dumps(legacy_payload), "queued")

        command.upgrade(config, "0006_execution_targets")
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT COUNT(*) FROM execution_settings").fetchone() == (0,)
            assert connection.execute("SELECT COUNT(*) FROM execution_probes").fetchone() == (0,)
            assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone() == (1,)
    finally:
        reset_settings_cache()
