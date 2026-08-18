from __future__ import annotations

from pathlib import Path

import pytest
from myleetgpu.config import Settings
from pydantic import ValidationError


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "data_dir": tmp_path / "data",
        "problems_dir": tmp_path / "problems",
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


def test_defaults_pin_cuda_image_and_require_runtime_arch_detection(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    assert settings.cuda_image == "nvidia/cuda:12.4.1-devel-ubuntu22.04"
    assert settings.cuda_arch == "auto"
    assert settings.api_host == "127.0.0.1"
    assert settings.cuda_image != "latest"
    assert not settings.cuda_image.endswith(":latest")


def test_defaults_pin_triton_image_by_digest(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    assert settings.triton_image == (
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
        "@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014"
    )
    assert "@sha256:" in settings.triton_image


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [("89", "89"), ("sm_89", "89"), ("compute_89", "89"), ("AUTO", "auto")],
)
def test_cuda_arch_is_normalized(tmp_path: Path, raw: str, normalized: str) -> None:
    assert make_settings(tmp_path, cuda_arch=raw).cuda_arch == normalized


@pytest.mark.parametrize("invalid", ["", "sm_", "ada", "8", "1234", "8.9"])
def test_cuda_arch_rejects_unusable_values(tmp_path: Path, invalid: str) -> None:
    with pytest.raises(ValidationError, match="CUDA arch"):
        make_settings(tmp_path, cuda_arch=invalid)


@pytest.mark.parametrize(
    "invalid",
    ["nvidia/cuda", "nvidia/cuda:latest", "nvidia/cuda:latest ", "", "   "],
)
def test_cuda_image_rejects_unpinned_or_latest_references(tmp_path: Path, invalid: str) -> None:
    with pytest.raises(ValidationError, match="explicit tag or digest"):
        make_settings(tmp_path, cuda_image=invalid)


def test_digest_pinned_cuda_image_is_allowed(tmp_path: Path) -> None:
    image = "nvidia/cuda@sha256:" + "a" * 64

    settings = make_settings(tmp_path, cuda_image=image)

    assert settings.cuda_image == image


@pytest.mark.parametrize(
    "invalid",
    ["pytorch/pytorch", "pytorch/pytorch:latest", "pytorch/pytorch:latest ", "", "   "],
)
def test_triton_image_rejects_unpinned_or_latest_references(tmp_path: Path, invalid: str) -> None:
    with pytest.raises(ValidationError, match="explicit tag or digest"):
        make_settings(tmp_path, triton_image=invalid)


def test_digest_pinned_triton_image_is_allowed(tmp_path: Path) -> None:
    image = "pytorch/pytorch:2.5.1@sha256:" + "b" * 64

    settings = make_settings(tmp_path, triton_image=image)

    assert settings.triton_image == image


def test_paths_and_database_are_isolated_under_configured_data_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "private-data"
    settings = make_settings(tmp_path, data_dir=data_dir)

    settings.ensure_directories()

    assert settings.jobs_dir == data_dir / "jobs"
    assert settings.jobs_dir.is_dir()
    assert settings.database_url == f"sqlite:///{(data_dir / 'myleetgpu.db').as_posix()}"
    assert settings.resolved_host_data_dir == data_dir.resolve()
    assert settings.host_data_mount == data_dir.resolve().as_posix()


def test_windows_host_data_mount_is_preserved_inside_linux_worker(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, host_data_dir=Path("D:/MyLeetGpu/data"))

    assert settings.host_data_mount == "D:/MyLeetGpu/data"


def test_database_override_wins_without_touching_default_location(tmp_path: Path) -> None:
    override = f"sqlite:///{(tmp_path / 'override.db').as_posix()}"
    settings = make_settings(tmp_path, database_url_override=override)

    assert settings.database_url == override
