from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from myleetgpu.api.main import create_app
from myleetgpu.config import Settings
from myleetgpu.domain.benchmark import source_hash
from myleetgpu.domain.jobs import GPU_RESOURCE

from tests.factories import create_saved_version, make_probe

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = "// api source must stay out of responses\nvoid solve() {}"


@pytest.fixture
def api(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "data",
        problems_dir=PROJECT_ROOT / "problems",
        database_url_override=f"sqlite:///{(tmp_path / 'api.db').as_posix()}",
        _env_file=None,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, app


def test_health_is_live_without_triggering_gpu_probe(api) -> None:
    client, _ = api

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "myleetgpu-api"


def test_readiness_distinguishes_database_from_unprobed_runner(api) -> None:
    client, app = api

    unavailable = client.get("/api/ready")

    assert unavailable.status_code == 503
    assert unavailable.json() == {
        "status": "not_ready",
        "database": True,
        "problems": 5,
        "runner": "unavailable",
        "worker_active": False,
        "runner_error": "worker has not probed the GPU yet",
    }

    app.state.repository.save_environment(make_probe("ready-environment"))
    app.state.repository.acquire_lease(GPU_RESOURCE, "api-test-worker")
    ready = client.get("/api/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["runner"] == "healthy"


def test_problem_list_and_detail_expose_all_public_manifests_only(api) -> None:
    client, _ = api

    listing = client.get("/api/problems")
    detail = client.get("/api/problems/reduction")

    assert listing.status_code == 200
    assert listing.json()["total"] == 5
    listed_by_slug = {item["slug"]: item for item in listing.json()["items"]}
    assert set(listed_by_slug) == {
        "grouped-query-attention",
        "multi-head-attention",
        "vector-addition",
        "matrix-transpose",
        "reduction",
    }
    assert listed_by_slug["multi-head-attention"]["languages"] == ["torch_python"]
    assert listed_by_slug["vector-addition"]["languages"] == [
        "cuda_cpp",
        "triton_python",
    ]
    assert detail.status_code == 200
    payload = detail.json()
    serialized = detail.text
    assert payload["slug"] == "reduction"
    assert "starter_code" in payload
    assert payload["default_language"] == "cuda_cpp"
    assert set(payload["implementations"]) == {"cuda_cpp", "triton_python"}
    assert payload["implementations"]["triton_python"]["editor_language"] == "python"
    assert "internal" not in payload
    assert "public" not in payload
    assert "harness" not in payload
    assert "suite_seed" not in serialized
    assert "signed_random" not in serialized
    assert "harness/validator.cu" not in serialized

    torch_detail = client.get("/api/problems/multi-head-attention").json()
    assert torch_detail["default_language"] == "torch_python"
    assert torch_detail["supported_languages"] == ["torch_python"]
    assert set(torch_detail["implementations"]) == {"torch_python"}
    assert torch_detail["implementations"]["torch_python"]["editor_language"] == "python"


def test_unknown_problem_is_404(api) -> None:
    client, _ = api

    response = client.get("/api/problems/not-installed")

    assert response.status_code == 404
    assert response.json()["error"] == {"code": "not_found", "message": "题目不存在"}


def test_draft_round_trip_is_problem_scoped_and_does_not_create_version(api) -> None:
    client, app = api

    assert client.get("/api/drafts/vector-addition").status_code == 404
    saved = client.put(
        "/api/drafts/vector-addition",
        json={"source": "first draft"},
    )
    updated = client.put(
        "/api/drafts/vector-addition",
        json={"source": "second draft"},
    )
    fetched = client.get("/api/drafts/vector-addition")

    assert saved.status_code == 200
    assert updated.status_code == 200
    assert fetched.json()["source"] == "second draft"
    assert app.state.repository.counts()["drafts"] == 1
    assert app.state.repository.counts()["versions"] == 0


def test_cuda_and_triton_drafts_are_independent_api_resources(api) -> None:
    client, app = api

    cuda = client.put(
        "/api/drafts/vector-addition",
        json={"language": "cuda_cpp", "source": "cuda draft"},
    )
    triton = client.put(
        "/api/drafts/vector-addition",
        json={"language": "triton_python", "source": "triton draft"},
    )

    assert cuda.json()["language"] == "cuda_cpp"
    assert triton.json()["language"] == "triton_python"
    assert (
        client.get("/api/drafts/vector-addition", params={"language": "cuda_cpp"}).json()["source"]
        == "cuda draft"
    )
    assert (
        client.get("/api/drafts/vector-addition", params={"language": "triton_python"}).json()[
            "source"
        ]
        == "triton draft"
    )
    assert app.state.repository.counts()["drafts"] == 2


def test_torch_only_problem_defaults_drafts_jobs_and_duplicates_to_torch(api) -> None:
    client, app = api
    source = "import torch\ndef solve(query, key, value, attention_mask):\n    return query\n"

    saved = client.put(
        "/api/drafts/multi-head-attention",
        json={"source": source},
    )
    fetched = client.get("/api/drafts/multi-head-attention")
    submitted = client.post(
        "/api/jobs",
        json={
            "problem_id": "multi-head-attention",
            "action": "validate",
            "source": source,
        },
    )
    version = create_saved_version(
        app.state.repository,
        problem_id="multi-head-attention",
        language="torch_python",
        source=source,
        source_digest=source_hash(source),
        compile_flags=("backend=torch_python", "policy=restricted_torch_v1"),
    )
    duplicate = client.get(
        "/api/versions/duplicates",
        params={"problem_id": "multi-head-attention", "source_hash": source_hash(source)},
    )

    assert saved.status_code == 200
    assert saved.json()["language"] == "torch_python"
    assert fetched.status_code == 200
    assert fetched.json()["language"] == "torch_python"
    assert submitted.status_code == 202
    assert submitted.json()["language"] == "torch_python"
    job = app.state.repository.get_job(submitted.json()["id"])
    assert job is not None and job.language == "torch_python"
    assert Path(job.spool_path, "source.py").is_file()
    assert duplicate.status_code == 200
    assert duplicate.json()["items"][0]["id"] == version.id

    unsupported = client.put(
        "/api/drafts/multi-head-attention",
        json={"language": "cuda_cpp", "source": SOURCE},
    )
    assert unsupported.status_code == 404


def test_triton_job_language_is_persisted_and_returned(api) -> None:
    client, app = api

    response = client.post(
        "/api/jobs",
        json={
            "problem_id": "vector-addition",
            "language": "triton_python",
            "action": "run",
            "source": "def solve(a, b, output, n):\n    return None\n",
        },
    )

    assert response.status_code == 202
    assert response.json()["language"] == "triton_python"
    job = app.state.repository.get_job(response.json()["id"])
    assert job is not None and job.language == "triton_python"
    assert Path(job.spool_path, "source.py").is_file()


def test_repeated_ordinary_job_posts_never_create_versions_or_echo_source(api) -> None:
    client, app = api
    created: list[str] = []

    for action in ("compile", "run", "validate"):
        for index in range(3):
            response = client.post(
                "/api/jobs",
                json={
                    "problem_id": "vector-addition",
                    "action": action,
                    "source": f"{SOURCE}\n// {action}-{index}",
                },
            )
            assert response.status_code == 202
            assert SOURCE not in response.text
            payload = response.json()
            assert payload["status"] == "queued"
            assert "source" not in payload
            assert "spool_path" not in payload
            assert "payload_json" not in payload
            created.append(payload["id"])

    assert app.state.repository.counts()["jobs"] == 9
    assert app.state.repository.counts()["versions"] == 0
    assert app.state.repository.counts()["benchmark_runs"] == 0
    fetched = client.get(f"/api/jobs/{created[0]}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == created[0]
    assert SOURCE not in fetched.text


def test_save_version_api_only_queues_and_does_not_persist_early(api) -> None:
    client, app = api

    response = client.post(
        "/api/jobs",
        json={
            "problem_id": "vector-addition",
            "action": "save_version",
            "source": SOURCE,
            "version_name": "manual",
            "notes": "wait for benchmark",
        },
    )

    assert response.status_code == 202
    assert response.json()["action"] == "save_version"
    assert app.state.repository.counts()["versions"] == 0
    assert app.state.repository.counts()["benchmark_runs"] == 0


def test_duplicate_save_returns_409_until_user_explicitly_confirms(api) -> None:
    client, app = api
    existing = create_saved_version(
        app.state.repository,
        source=SOURCE,
        source_digest=source_hash(SOURCE),
    )
    payload = {
        "problem_id": "vector-addition",
        "action": "save_version",
        "source": SOURCE,
        "version_name": "duplicate",
    }

    rejected = client.post("/api/jobs", json=payload)
    confirmed = client.post(
        "/api/jobs",
        json={**payload, "allow_duplicate": True},
    )

    assert rejected.status_code == 409
    assert rejected.json()["error"] == {
        "code": "duplicate_source",
        "message": "相同源码已存在，请确认后再保存",
        "duplicates": [{"id": existing.id, "name": existing.name}],
    }
    assert confirmed.status_code == 202
    assert confirmed.json()["status"] == "queued"
    assert app.state.repository.counts()["versions"] == 1


@pytest.mark.parametrize(
    ("payload", "status_code"),
    [
        ({"problem_id": "vector-addition", "action": "compile"}, 400),
        (
            {
                "problem_id": "missing",
                "action": "compile",
                "source": SOURCE,
            },
            400,
        ),
        (
            {
                "problem_id": "vector-addition",
                "action": "debug",
                "source": SOURCE,
            },
            422,
        ),
    ],
)
def test_job_errors_use_structured_invalid_request_envelope(
    api, payload: dict[str, object], status_code: int
) -> None:
    client, _ = api

    response = client.post("/api/jobs", json=payload)

    assert response.status_code == status_code
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
    assert body["error"]["message"]


def test_job_response_returns_structured_404_for_unknown_identifier(api) -> None:
    client, _ = api

    response = client.get("/api/jobs/not-a-job")

    assert response.status_code == 404
    assert response.json()["error"] == {"code": "not_found", "message": "任务不存在"}


def test_environment_endpoint_is_honest_about_unavailable_telemetry(api) -> None:
    client, app = api

    unknown = client.get("/api/environment")
    assert unknown.status_code == 200
    assert unknown.json()["status"] == "unknown"
    assert set(unknown.json()["telemetry"].values()) == {None}

    app.state.repository.save_environment(make_probe("environment-api"))
    app.state.repository.acquire_lease(GPU_RESOURCE, "api-test-worker")
    healthy = client.get("/api/environment")
    payload = healthy.json()
    assert payload["status"] == "healthy"
    assert payload["gpu_name"] == "NVIDIA GeForce RTX 4060"
    assert payload["compute_capability"] == "8.9"
    assert payload["cuda_arch"] == "89"
    assert set(payload["telemetry"].values()) == {None}

    app.state.repository.save_environment(
        make_probe("torch-environment-api", backend="torch_python")
    )
    torch_environment = client.get("/api/environment", params={"language": "torch_python"}).json()
    assert torch_environment["status"] == "healthy"
    assert torch_environment["backend"] == "torch_python"
    assert torch_environment["toolchain"] == {
        "python_version": "3.11.10",
        "torch_version": "2.5.1",
        "torch_cuda_version": "12.4",
    }


def test_version_list_duplicate_lookup_and_metadata_update(api) -> None:
    client, app = api
    repository = app.state.repository
    version = create_saved_version(
        repository,
        name="before",
        source="immutable snapshot",
        source_digest=source_hash("immutable snapshot"),
    )

    listing = client.get("/api/problems/vector-addition/versions")
    duplicate = client.get(
        "/api/versions/duplicates",
        params={
            "problem_id": "vector-addition",
            "source_hash": source_hash("immutable snapshot"),
        },
    )
    updated = client.patch(
        f"/api/versions/{version.id}",
        json={"name": "after", "notes": "new note"},
    )

    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    assert listing.json()["items"][0]["source_code"] == "immutable snapshot"
    assert len(listing.json()["items"][0]["benchmark_runs"]) == 1
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True
    assert duplicate.json()["items"][0]["id"] == version.id
    assert updated.status_code == 200
    assert updated.json()["name"] == "after"
    assert updated.json()["notes"] == "new note"
    assert updated.json()["source_code"] == "immutable snapshot"


def test_version_update_rejects_empty_or_whitespace_only_name(api) -> None:
    client, app = api
    version = create_saved_version(app.state.repository)

    empty_body = client.patch(f"/api/versions/{version.id}", json={})
    whitespace_name = client.patch(f"/api/versions/{version.id}", json={"name": "   "})

    assert empty_body.status_code == 422
    assert empty_body.json()["error"]["code"] == "invalid_request"
    assert whitespace_name.status_code == 422
    assert whitespace_name.json()["error"]["code"] == "invalid_request"
    assert app.state.repository.get_version(version.id).name == "baseline"


def test_delete_version_requires_confirmation_and_cascades_benchmark(api) -> None:
    client, app = api
    version = create_saved_version(app.state.repository)

    unconfirmed = client.delete(f"/api/versions/{version.id}")
    assert unconfirmed.status_code == 409
    assert app.state.repository.counts()["versions"] == 1

    confirmed = client.delete(f"/api/versions/{version.id}", params={"confirmed": "true"})
    assert confirmed.status_code == 204
    assert app.state.repository.counts()["versions"] == 0
    assert app.state.repository.counts()["benchmark_runs"] == 0


def test_compare_endpoint_returns_speedups_only_for_comparable_versions(api) -> None:
    client, app = api
    baseline = create_saved_version(
        app.state.repository,
        source_digest="1" * 64,
        medians=(4.0, 8.0),
    )
    candidate = create_saved_version(
        app.state.repository,
        source_digest="2" * 64,
        medians=(2.0, 4.0),
    )

    response = client.post(
        "/api/versions/compare",
        json={
            "problem_id": "vector-addition",
            "version_ids": [baseline.id, candidate.id],
            "baseline_id": baseline.id,
        },
    )

    assert response.status_code == 200
    assert response.json()["comparable"] is True
    assert response.json()["rows"][0]["metrics"][candidate.id]["speedup"] == 2.0


def test_untrusted_host_header_is_rejected(api) -> None:
    client, _ = api

    response = client.get("/api/health", headers={"host": "evil.example"})

    assert response.status_code == 400


def test_draft_version_and_benchmark_survive_full_app_restart(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "persistent-data",
        problems_dir=PROJECT_ROOT / "problems",
        database_url_override=f"sqlite:///{(tmp_path / 'persistent-api.db').as_posix()}",
        _env_file=None,
    )
    first_app = create_app(settings)
    with TestClient(first_app) as first_client:
        saved_draft = first_client.put(
            "/api/drafts/vector-addition",
            json={"source": "draft survives restart"},
        )
        assert saved_draft.status_code == 200
        version = create_saved_version(
            first_app.state.repository,
            source="version survives restart",
            source_digest="r" * 64,
        )

    second_app = create_app(settings)
    with TestClient(second_app) as second_client:
        restored_draft = second_client.get("/api/drafts/vector-addition")
        restored_versions = second_client.get("/api/problems/vector-addition/versions")

        assert restored_draft.status_code == 200
        assert restored_draft.json()["source"] == "draft survives restart"
        assert restored_versions.status_code == 200
        assert restored_versions.json()["total"] == 1
        restored = restored_versions.json()["items"][0]
        assert restored["id"] == version.id
        assert restored["source_code"] == "version survives restart"
        assert len(restored["benchmark_runs"]) == 1
        assert restored["benchmark_runs"][0]["measurements"][0]["median_ms"] == 4.0
