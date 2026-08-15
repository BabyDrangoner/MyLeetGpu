from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Real-GPU tests are opt-in and must never silently fall back to mocks."""

    if os.environ.get("MYLEETGPU_RUN_GPU_TESTS") == "1":
        return

    skip_gpu = pytest.mark.skip(
        reason="set MYLEETGPU_RUN_GPU_TESTS=1 to run real Docker/NVIDIA acceptance tests"
    )
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
