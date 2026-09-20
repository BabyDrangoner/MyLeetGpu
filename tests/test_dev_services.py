from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SPEC = importlib.util.spec_from_file_location(
    "dev_services", Path(__file__).resolve().parents[1] / "scripts/dev_services.py"
)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture
def manager(tmp_path):
    result = launcher.DevServices(tmp_path)
    result.directory.mkdir(parents=True)
    return result


def mock_start(manager, monkeypatch, active=()):
    monkeypatch.setattr(manager, "running", lambda name: name in active)
    for method in ("check_conflicts", "migrate", "launch", "wait_ready", "status", "stop"):
        monkeypatch.setattr(manager, method, Mock())
    monkeypatch.setattr(
        manager, "commands", Mock(return_value={name: [name] for name in ("api", "worker", "web")})
    )


def test_start_is_idempotent_and_never_migrates_active_services(manager, monkeypatch):
    mock_start(manager, monkeypatch, active=("api", "worker", "web"))
    manager.start()
    manager.launch.assert_not_called()
    manager.migrate.assert_not_called()
    manager.commands.assert_not_called()
    manager.wait_ready.assert_called_once()


def test_partial_start_preserves_running_worker_and_skips_migration(manager, monkeypatch):
    mock_start(manager, monkeypatch, active=("worker",))
    manager.start()
    manager.migrate.assert_not_called()
    assert [call.args[0] for call in manager.launch.call_args_list] == ["api", "web"]


def test_fresh_start_migrates_before_launch(manager, monkeypatch):
    mock_start(manager, monkeypatch)
    events = []
    manager.migrate.side_effect = lambda: events.append("migrate")
    manager.launch.side_effect = lambda name, command: events.append(name)
    manager.start()
    assert events == ["migrate", "api", "worker", "web"]


def test_failed_migration_does_not_launch_anything(manager, monkeypatch):
    mock_start(manager, monkeypatch)
    manager.migrate.side_effect = subprocess.CalledProcessError(1, "alembic")
    with pytest.raises(subprocess.CalledProcessError):
        manager.start()
    manager.launch.assert_not_called()


def test_start_failure_only_rolls_back_new_services(manager, monkeypatch):
    mock_start(manager, monkeypatch, active=("worker",))
    manager.wait_ready.side_effect = RuntimeError("not ready")
    with pytest.raises(RuntimeError, match="not ready"):
        manager.start()
    manager.stop.assert_called_once_with(["api", "web"])


def test_absolute_paths_do_not_depend_on_current_directory(manager, monkeypatch, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    for variable in ("MYLEETGPU_DATA_DIR", "MYLEETGPU_HOST_DATA_DIR"):
        assert manager.environment[variable] == str(tmp_path / "data")
    assert manager.environment["PYTHONPATH"] == str(tmp_path / "backend")
    assert manager.environment["MYLEETGPU_API_HOST"] == "127.0.0.1"


def test_missing_dependencies_fail_without_installing(manager):
    with pytest.raises(RuntimeError, match="Missing executable.*make install"):
        manager.commands()


@pytest.mark.parametrize("override", [True, False])
def test_node_override_or_bundled_fallback(manager, monkeypatch, tmp_path, override):
    node = tmp_path / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
    for executable in (manager.python, tmp_path / ".venv/bin/alembic", node):
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.touch()
        executable.chmod(0o700)
    vite = tmp_path / "apps/web/node_modules/vite/bin/vite.js"
    vite.parent.mkdir(parents=True)
    vite.touch()
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    monkeypatch.setattr(launcher.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("MYLEETGPU_NODE_BIN", raising=False)
    if override:
        monkeypatch.setenv("MYLEETGPU_NODE_BIN", str(node))
    assert manager.commands()["web"] == [
        str(node),
        str(vite),
        "--host",
        "127.0.0.1",
        "--port",
        "3000",
        "--strictPort",
    ]


def test_port_conflict_is_not_killed_or_adopted(manager, monkeypatch):
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        monkeypatch.setattr(launcher, "PORTS", {"web": occupied.getsockname()[1]})
        with pytest.raises(RuntimeError, match="unmanaged process"):
            manager.check_conflicts(set())
    assert manager.records == {}


def test_recently_closed_connection_does_not_block_restart(manager, monkeypatch):
    with socket.socket() as listener, socket.socket() as client:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen(1)
        client.connect(("127.0.0.1", port))
        connection, _ = listener.accept()
        # Server closes first, leaving its port in TIME_WAIT after the client closes.
        connection.shutdown(socket.SHUT_WR)
        connection.close()
        assert client.recv(1) == b""
    monkeypatch.setattr(launcher, "PORTS", {"api": port})
    monkeypatch.setattr(
        launcher.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="")
    )
    manager.check_conflicts(set())


def test_reusable_live_listener_is_still_rejected(manager, monkeypatch):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        monkeypatch.setattr(launcher, "PORTS", {"api": listener.getsockname()[1]})
        with pytest.raises(RuntimeError, match="unmanaged process"):
            manager.check_conflicts(set())


def test_unmanaged_worker_prevents_migration(manager, monkeypatch):
    monkeypatch.setattr(launcher, "PORTS", {})
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="1234 python -m myleetgpu.worker\n"),
    )
    with pytest.raises(RuntimeError, match="Unmanaged MyLeetGpu worker"):
        manager.check_conflicts(set())


def test_stale_pid_is_never_signalled(manager, monkeypatch):
    manager.records = {"worker": {"pid": 999, "fingerprint": "old creation time"}}
    monkeypatch.setattr(launcher, "fingerprint", lambda pid: "new creation time")
    signal = Mock()
    monkeypatch.setattr(launcher.os, "killpg", signal)
    manager.stop()
    signal.assert_not_called()
    assert manager.records == {}


def test_launch_and_stop_only_disposable_test_process(manager):
    # No project service, GPU, SSH, fixed port, or real project data is involved.
    manager.launch("worker", [sys.executable, "-c", "import time; time.sleep(60)"])
    pid = manager.records["worker"]["pid"]
    try:
        assert os.getpgid(pid) == pid
        assert manager.running("worker")
        reloaded = launcher.DevServices(manager.root)
        reloaded.load()
        assert reloaded.running("worker")
        reloaded.stop()
        assert not reloaded.running("worker")
        assert (manager.directory / "worker.log").exists()
    finally:
        manager.stop()
        # subprocess may already have reaped the exited child during another ps call.
        with suppress(ChildProcessError):
            os.waitpid(pid, 0)


def test_health_does_not_require_gpu_ready(manager, monkeypatch):
    monkeypatch.setattr(manager, "running", lambda name: True)
    response = Mock()
    response.__enter__ = Mock(return_value=SimpleNamespace(status=200))
    response.__exit__ = Mock(return_value=None)
    request = Mock(return_value=response)
    monkeypatch.setattr(launcher.urllib.request, "urlopen", request)
    manager.wait_ready()
    assert [call.args[0] for call in request.call_args_list] == [
        "http://127.0.0.1:8000/api/health",
        "http://127.0.0.1:3000/",
    ]
