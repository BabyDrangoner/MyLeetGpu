"""Opt-in real Colab acceptance; never replaces GPU execution with a test double."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from myleetgpu.config import Settings
from myleetgpu.domain.problems import ProblemCatalog
from myleetgpu.runner.colab import ColabRunner

pytestmark = [
    pytest.mark.colab_gpu,
    pytest.mark.skipif(
        os.environ.get("MYLEETGPU_RUN_COLAB_TESTS") != "1",
        reason="set MYLEETGPU_RUN_COLAB_TESTS=1 with an existing Colab SSH master",
    ),
]


@pytest.mark.parametrize(
    ("slug", "language"),
    [
        ("vector-addition", "cuda_cpp"),
        ("vector-addition", "triton_python"),
        ("multi-head-attention", "torch_python"),
        ("grouped-query-attention", "torch_python"),
    ],
)
def test_real_colab_compile_validate_and_benchmark(
    tmp_path: Path, slug: str, language: str
) -> None:
    settings = Settings(
        data_dir=tmp_path / "colab-acceptance",
        problems_dir=Path(__file__).resolve().parents[2] / "problems",
        _env_file=None,
    )
    settings.ensure_directories()
    runner = ColabRunner(settings)
    runner.assign_owner(f"colab-acceptance-{uuid.uuid4()}")
    probe = runner.recover(language=language)
    assert probe.healthy, probe.error
    assert probe.toolchain["execution_target"] == "colab"
    assert probe.gpu_name and probe.fingerprint
    problem = ProblemCatalog(settings.problems_dir).load().get(slug)
    implementation = problem.get_implementation(language)
    task = settings.jobs_dir / str(uuid.uuid4())
    task.mkdir()
    source = task / f"source{implementation.source_suffix}"
    source.write_text(implementation.starter_code, encoding="utf-8")
    try:
        for stage, modes in [("validator", ("public", "full")), ("benchmark", ("benchmark",))]:
            compiled = runner.compile(
                task,
                problem,
                source,
                harness_kind=stage,
                language=language,
                implementation=implementation,
            )
            assert compiled.succeeded, compiled.diagnostics
            assert compiled.executable is not None
            for mode in modes:
                result = runner.execute(
                    task,
                    compiled.executable,
                    mode=mode,
                    timeout_seconds=180,
                    language=language,
                )
                assert result.succeeded, result.output
                assert result.parsed and result.parsed["status"] == "passed"
    finally:
        runner.cleanup_task(task)
