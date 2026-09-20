from __future__ import annotations

import pytest

from tests.conftest import pytest_collection_modifyitems


def collection_item(request: pytest.FixtureRequest, *, directory: str = "tests") -> pytest.Item:
    item = pytest.Function.from_parent(request.session, name="example", callobj=lambda: None)
    # Pytest includes ancestor directory names in keywords even without a marker.
    item.keywords[directory] = True
    return item


@pytest.mark.parametrize("opt_in", [None, "0", "true"])
def test_explicit_docker_gpu_marker_requires_exact_opt_in(monkeypatch, request, opt_in):
    if opt_in is None:
        monkeypatch.delenv("MYLEETGPU_RUN_GPU_TESTS", raising=False)
    else:
        monkeypatch.setenv("MYLEETGPU_RUN_GPU_TESTS", opt_in)
    item = collection_item(request)
    item.add_marker(pytest.mark.gpu)

    pytest_collection_modifyitems(request.config, [item])

    skipped = item.get_closest_marker("skip")
    assert skipped is not None
    assert "MYLEETGPU_RUN_GPU_TESTS=1" in skipped.kwargs["reason"]


def test_explicit_docker_gpu_marker_runs_when_opted_in(monkeypatch, request):
    monkeypatch.setenv("MYLEETGPU_RUN_GPU_TESTS", "1")
    item = collection_item(request)
    item.add_marker(pytest.mark.gpu)

    pytest_collection_modifyitems(request.config, [item])

    assert item.get_closest_marker("skip") is None


@pytest.mark.parametrize("colab_marker", [False, True])
def test_gpu_directory_keyword_does_not_require_docker_opt_in(monkeypatch, request, colab_marker):
    monkeypatch.delenv("MYLEETGPU_RUN_GPU_TESTS", raising=False)
    monkeypatch.setenv("MYLEETGPU_RUN_COLAB_TESTS", "1")
    item = collection_item(request, directory="gpu")
    if colab_marker:
        item.add_marker(pytest.mark.colab_gpu)
    assert "gpu" in item.keywords
    assert item.get_closest_marker("gpu") is None

    pytest_collection_modifyitems(request.config, [item])

    assert item.get_closest_marker("skip") is None


def test_cpu_test_is_unchanged_without_gpu_opt_in(monkeypatch, request):
    monkeypatch.delenv("MYLEETGPU_RUN_GPU_TESTS", raising=False)
    monkeypatch.delenv("MYLEETGPU_RUN_COLAB_TESTS", raising=False)
    item = collection_item(request)
    original_markers = list(item.iter_markers())

    pytest_collection_modifyitems(request.config, [item])

    assert list(item.iter_markers()) == original_markers
