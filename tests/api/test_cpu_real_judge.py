"""Real native CPU judging through API/queue/persistence, in an isolated database."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from myleetgpu.api.main import create_app
from myleetgpu.config import Settings
from myleetgpu.runner.cpu import CpuRunner
from myleetgpu.runner.router import ExecutionRouter
from myleetgpu.worker import Worker


@pytest.mark.parametrize("language", ["cpp", "python"])
def test_real_cpu_judge_round_trip_without_gpu(tmp_path: Path, language: str) -> None:
    if language == "cpp" and not shutil.which("c++"):
        pytest.skip("requires a C++17 compiler")
    settings = Settings(
        data_dir=tmp_path / "data",
        problems_dir=Path(__file__).resolve().parents[2] / "problems",
        _env_file=None,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        repository = app.state.repository
        # Colab need not exist; no GPU adapter is even installed in this test router.
        repository.update_execution_settings("colab", True)
        router = ExecutionRouter(settings, runners={"cpu": CpuRunner(settings)})
        worker = Worker(settings, app.state.catalog, repository, router, worker_id="real-cpu-test")
        source = app.state.catalog.get("online-softmax").get_implementation(language).starter_code

        def submit(action: str, **fields):
            response = client.post(
                "/api/jobs",
                json={
                    "problem_id": "online-softmax",
                    "language": language,
                    "action": action,
                    "source": source,
                    **fields,
                },
            )
            assert response.status_code == 202, response.text
            job_id = response.json()["id"]
            assert response.json()["execution_target"] == "cpu"
            assert worker.process_next()
            completed = client.get(f"/api/jobs/{job_id}").json()
            assert completed["status"] == "succeeded", completed
            assert completed["result"]["execution_target"] == "cpu"
            return completed["result"]

        for action in ("compile", "run", "validate"):
            submit(action)
        saved = submit("save_version", version_name=f"CPU acceptance {language}")
        version = repository.get_versions([saved["version_id"]])[0]
        assert version.language == language
        assert version.environment.toolchain_json["execution_target"] == "cpu"
        assert len(version.benchmark_runs) == 1
        submit("rebenchmark", version_ids=[version.id])
        refreshed = repository.get_versions([version.id])[0]
        assert len(refreshed.benchmark_runs) == 2
        assert repository.execution_settings()["target"] == "colab"
