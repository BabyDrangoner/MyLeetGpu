from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from myleetgpu import worker as worker_module
from myleetgpu.config import Settings
from myleetgpu.infrastructure.database import build_engine


@pytest.mark.parametrize("failure", [None, "catalog", "runner", "run"])
def test_worker_factory_disposes_engine_on_every_exit(tmp_path, monkeypatch, failure):
    settings = Settings(
        data_dir=tmp_path / "data",
        problems_dir=Path(__file__).resolve().parents[1] / "problems",
        _env_file=None,
    )
    settings.ensure_directories()
    engine = build_engine(settings)
    dispose = Mock(wraps=engine.dispose)
    monkeypatch.setattr(engine, "dispose", dispose)
    monkeypatch.setattr(worker_module, "build_engine", lambda _: engine)
    runner = Mock()
    runner_factory = Mock(return_value=runner)
    monkeypatch.setattr(worker_module, "ExecutionRouter", runner_factory)
    if failure == "catalog":
        monkeypatch.setattr(
            worker_module.ProblemCatalog, "load", Mock(side_effect=ValueError("catalog failed"))
        )
    elif failure == "runner":
        runner_factory.side_effect = ValueError("runner failed")

    def start():
        with worker_module.create_worker(settings) as worker:
            assert worker.runner is runner
            dispose.assert_not_called()
            if failure == "run":
                raise ValueError("run failed")

    if failure:
        with pytest.raises(ValueError, match=failure):
            start()
    else:
        start()
    dispose.assert_called_once()
