"""CPU-only architecture contracts: ports are explicit and adapters are peers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import create_autospec

import pytest
from myleetgpu.config import Settings
from myleetgpu.runner.base import BaseRunner
from myleetgpu.runner.colab import ColabRunner
from myleetgpu.runner.docker import DockerRunner
from myleetgpu.runner.models import RunnerUnavailable
from myleetgpu.runner.protocols import Runner, TargetRunner
from myleetgpu.runner.router import ExecutionRouter

from tests.factories import make_probe


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", _env_file=None)


@pytest.fixture
def routed(settings: Settings):
    local = create_autospec(Runner, instance=True)
    colab = create_autospec(Runner, instance=True)
    return ExecutionRouter(settings, runners={"local": local, "colab": colab}), local, colab


def test_colab_is_not_a_docker_subtype_and_never_initializes_docker(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args, **kwargs):
        pytest.fail("Constructing Colab must not construct or probe Docker")

    monkeypatch.setattr(DockerRunner, "__init__", fail)
    monkeypatch.setattr(DockerRunner, "_run_limited", fail)

    colab = ColabRunner(settings)

    assert isinstance(colab, BaseRunner)
    assert not isinstance(colab, DockerRunner)
    assert "_docker" not in vars(colab)
    assert "_installation_label" not in vars(colab)
    assert "_owner_label" not in vars(colab)
    assert colab._health_file.name == "colab-runner-unhealthy.json"


def test_adapters_and_router_satisfy_the_explicit_runner_port(settings: Settings) -> None:
    local, colab = DockerRunner(settings), ColabRunner(settings)
    router = ExecutionRouter(settings, runners={"local": local, "colab": colab})

    assert all(isinstance(adapter, Runner) for adapter in (local, colab, router))
    assert isinstance(router, TargetRunner)
    assert not isinstance(local, TargetRunner)
    assert not isinstance(colab, TargetRunner)


def test_shared_mechanics_are_not_copied_between_adapters() -> None:
    for method in (
        "prepare_compile",
        "effective_compile_flags",
        "_resolve_language",
        "_harness_path",
        "_parse_result",
        "_clean_output",
        "mark_unhealthy",
    ):
        assert getattr(DockerRunner, method) is getattr(BaseRunner, method)
        assert getattr(ColabRunner, method) is getattr(BaseRunner, method)


def test_shared_mechanics_do_not_share_mutable_provider_state(settings: Settings) -> None:
    local, colab = DockerRunner(settings), ColabRunner(settings)
    local._cached_probes["cuda_cpp"] = (0.0, make_probe("local-only"))
    colab.mark_unhealthy("remote health failure")

    assert not colab._cached_probes
    assert not local._health_file.exists()
    assert colab._health_file.exists()


def test_router_does_not_expose_adapter_specific_attributes(settings: Settings) -> None:
    router = ExecutionRouter(settings)
    assert "__getattr__" not in ExecutionRouter.__dict__
    for target in ("local", "colab"):
        router.select_target(target)
        with pytest.raises(AttributeError):
            _ = router._health_file


@pytest.mark.parametrize("target", ["local", "colab"])
@pytest.mark.parametrize(
    ("method", "args", "kwargs"),
    [
        ("assert_healthy", (), {}),
        ("mark_unhealthy", ("trusted probe failed",), {}),
        ("cleanup_task", (Path("job-id"),), {}),
        ("cleanup_owned_containers", (), {}),
        (
            "compile",
            (Path("job"), object(), Path("source")),
            {"harness_kind": "benchmark", "language": "torch_python", "implementation": object()},
        ),
        (
            "execute",
            (Path("job"), Path("artifact.py")),
            {"mode": "full", "timeout_seconds": 4.5, "language": "triton_python"},
        ),
        (
            "effective_compile_flags",
            (object(), make_probe("flags")),
            {"language": "cuda_cpp", "implementation": object()},
        ),
    ],
)
def test_router_explicitly_delegates_only_to_the_selected_target(
    routed, target: str, method: str, args: tuple, kwargs: dict
) -> None:
    router, local, colab = routed
    selected, unselected = (local, colab) if target == "local" else (colab, local)
    router.select_target(target)

    getattr(router, method)(*args, **kwargs)

    getattr(selected, method).assert_called_once_with(*args, **kwargs)
    assert not unselected.mock_calls


@pytest.mark.parametrize(
    "method", ["probe_environment", "probe_triton_environment", "probe_torch_environment"]
)
@pytest.mark.parametrize("target", ["local", "colab"])
def test_explicit_probe_forwarding_preserves_target_fingerprint_boundaries(
    routed, target: str, method: str
) -> None:
    router, local, colab = routed
    selected, unselected = (local, colab) if target == "local" else (colab, local)
    original = make_probe("shared-hardware-fingerprint")
    getattr(selected, method).return_value = original
    router.select_target(target)

    probe = getattr(router, method)(force=True, ignore_circuit_breaker=True)

    getattr(selected, method).assert_called_once_with(force=True, ignore_circuit_breaker=True)
    assert probe.toolchain["execution_target"] == target
    assert (probe.fingerprint == original.fingerprint) is (target == "local")
    assert "execution_target" not in original.toolchain
    assert not unselected.mock_calls


def test_owner_assignment_covers_all_but_cleanup_only_selected_provider(routed) -> None:
    router, local, colab = routed
    local.cleanup_orphan_containers.return_value = ["local-orphan"]
    colab.cleanup_orphan_containers.return_value = ["remote-orphan"]

    router.assign_owner("lease-owner")
    removed = router.cleanup_orphan_containers()

    local.assign_owner.assert_called_once_with("lease-owner")
    colab.assign_owner.assert_called_once_with("lease-owner")
    assert removed == ["local-orphan"]
    colab.cleanup_orphan_containers.assert_not_called()


def test_unavailable_cleanup_never_falls_back_to_another_provider(routed) -> None:
    router, local, colab = routed
    local.cleanup_orphan_containers.side_effect = RunnerUnavailable("local offline")
    colab.cleanup_orphan_containers.return_value = ["remote-orphan"]

    with pytest.raises(RunnerUnavailable, match="local offline"):
        router.cleanup_orphan_containers()
    colab.cleanup_orphan_containers.assert_not_called()
    router.select_target("colab")
    assert router.cleanup_orphan_containers() == ["remote-orphan"]
    colab.cleanup_orphan_containers.assert_called_once_with()


def test_rejected_target_never_changes_the_active_provider(routed) -> None:
    router, _, _ = routed
    router.select_target("colab")

    with pytest.raises(RunnerUnavailable):
        router.select_target("unknown-provider")

    assert router.target == "colab"
