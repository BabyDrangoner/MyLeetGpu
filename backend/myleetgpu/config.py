from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings; every mutable or host-specific value comes from the environment."""

    model_config = SettingsConfigDict(
        env_prefix="MYLEETGPU_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    problems_dir: Path = Path("problems")
    host_data_dir: Path | None = None
    cuda_image: str = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
    triton_image: str = (
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
        "@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014"
    )
    cuda_arch: str = "auto"
    log_level: str = "INFO"
    job_poll_seconds: float = Field(default=0.25, ge=0.05, le=10)
    output_limit_bytes: int = Field(default=65_536, ge=1024, le=1_048_576)
    container_memory: str = "2g"
    container_cpus: float = Field(default=2.0, ge=0.25, le=16)
    compile_timeout_seconds: int = Field(default=90, ge=5, le=600)
    run_timeout_seconds: int = Field(default=30, ge=1, le=300)
    validate_timeout_seconds: int = Field(default=90, ge=1, le=900)
    benchmark_timeout_seconds: int = Field(default=180, ge=1, le=1800)
    allow_unsafe_host_runner: bool = False
    database_url_override: str | None = None
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)

    @field_validator("cuda_image", "triton_image")
    @classmethod
    def reject_latest_image(cls, value: str) -> str:
        value = value.strip()
        if not value or value.endswith(":latest") or ":" not in value:
            raise ValueError("runner image must use an explicit tag or digest, never latest")
        return value

    @field_validator("cuda_arch")
    @classmethod
    def validate_arch(cls, value: str) -> str:
        normalized = value.lower().removeprefix("sm_").removeprefix("compute_")
        if normalized != "auto" and (not normalized.isdigit() or len(normalized) not in {2, 3}):
            raise ValueError("CUDA arch must be auto or a numeric compute capability such as 89")
        return normalized

    @field_validator("api_host")
    @classmethod
    def restrict_api_bind(cls, value: str) -> str:
        if value not in {"127.0.0.1", "::1", "0.0.0.0"}:
            raise ValueError("API bind host is not allowed")
        return value

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return self.database_url_override
        return f"sqlite:///{(self.data_dir / 'myleetgpu.db').resolve().as_posix()}"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def resolved_host_data_dir(self) -> Path:
        raw = self.host_data_dir or self.data_dir
        return raw.expanduser().resolve()

    @property
    def host_data_mount(self) -> str:
        """Return a Docker-daemon path without corrupting Windows drive paths.

        The Worker itself runs in Linux, but a stack launched from PowerShell
        legitimately receives a value such as ``D:/MyLeetGpu/data``. Resolving
        that value with POSIX pathlib would incorrectly prefix the container's
        current directory.
        """

        raw = str(self.host_data_dir or self.data_dir).replace("\\", "/").rstrip("/")
        if re.match(r"^[A-Za-z]:/", raw):
            return raw
        return self.resolved_host_data_dir.as_posix()

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Test helper for environment-isolated settings."""

    get_settings.cache_clear()
    os.environ.pop("MYLEETGPU_TEST_SETTINGS_SENTINEL", None)
