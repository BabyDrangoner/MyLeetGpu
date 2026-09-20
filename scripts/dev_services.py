"""Detached local development services: python scripts/dev_services.py start|status|stop."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORTS = {"api": 8000, "web": 3000}


def fingerprint(pid: int) -> str | None:
    """Include creation time and full command so a recycled PID is never stopped."""
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat=", "-o", "lstart=", "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    fields = result.stdout.strip().split(None, 1)
    return fields[1] if len(fields) == 2 and not fields[0].startswith("Z") else None


class DevServices:
    def __init__(self, root: Path = ROOT):
        self.root = root.resolve()
        self.directory = self.root / "data" / "dev-services"
        self.state_path = self.directory / "processes.json"
        self.python = self.root / ".venv" / "bin" / "python"
        self.records: dict[str, dict] = {}
        self.environment = {
            **os.environ,
            "PYTHONPATH": str(self.root / "backend"),
            "PYTHONUNBUFFERED": "1",
            "MYLEETGPU_DATA_DIR": str(self.root / "data"),
            "MYLEETGPU_HOST_DATA_DIR": str(self.root / "data"),
            "MYLEETGPU_PROBLEMS_DIR": str(self.root / "problems"),
            "MYLEETGPU_API_HOST": "127.0.0.1",
            "MYLEETGPU_API_PORT": "8000",
            "MYLEETGPU_API_ORIGIN": "http://127.0.0.1:8000",
        }

    def load(self) -> None:
        if self.state_path.exists():
            self.records = json.loads(self.state_path.read_text())
            if not isinstance(self.records, dict) or not all(
                isinstance(record, dict) for record in self.records.values()
            ):
                raise RuntimeError(f"Invalid process registry: {self.state_path}")

    def save(self) -> None:
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.records, indent=2) + "\n")
        temporary.replace(self.state_path)

    def running(self, name: str) -> bool:
        record = self.records.get(name, {})
        pid = record.get("pid")
        return (
            isinstance(pid, int)
            and pid > 1
            and bool(record.get("fingerprint"))
            and fingerprint(pid) == record["fingerprint"]
        )

    def commands(self) -> dict[str, list[str]]:
        bundled_node = (
            Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
        )
        node = os.environ.get("MYLEETGPU_NODE_BIN") or shutil.which("node")
        node = node or (str(bundled_node) if bundled_node.is_file() else None)
        vite = self.root / "apps/web/node_modules/vite/bin/vite.js"
        for executable in (self.python, self.root / ".venv/bin/alembic", node):
            if not executable or not os.access(executable, os.X_OK):
                raise RuntimeError(
                    f"Missing executable: {executable or 'node'}. "
                    "Install project dependencies first (make install); "
                    "set MYLEETGPU_NODE_BIN if Node is not on PATH."
                )
        if not vite.is_file():
            raise RuntimeError("Missing frontend dependencies; run make install first.")
        return {
            "api": [str(self.python), "-m", "myleetgpu.api.main"],
            "worker": [str(self.python), "-m", "myleetgpu.worker"],
            "web": [str(node), str(vite), "--host", "127.0.0.1", "--port", "3000", "--strictPort"],
        }

    def check_conflicts(self, active: set[str]) -> None:
        for name, port in PORTS.items():
            if name not in active:
                with socket.socket() as probe:
                    # Match the servers' normal restart semantics: TIME_WAIT is not a
                    # live listener. Do not enable SO_REUSEPORT, which could hide one.
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        probe.bind(("127.0.0.1", port))
                    except OSError as error:
                        raise RuntimeError(
                            f"Port {port} is occupied by an unmanaged process; "
                            "stop it explicitly before starting services."
                        ) from error
        processes = subprocess.run(
            ["ps", "-ax", "-o", "pid=", "-o", "command="],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        known = {str(self.records[name]["pid"]) for name in active}
        for line in processes.splitlines():
            fields = line.strip().split(None, 1)
            if len(fields) == 2 and "-m myleetgpu.worker" in fields[1] and fields[0] not in known:
                raise RuntimeError(
                    f"Unmanaged MyLeetGpu worker (PID {fields[0]}) is running; "
                    "stop it explicitly to avoid duplicate workers or a live migration."
                )

    def migrate(self) -> None:
        with (self.directory / "migration.log").open("ab") as output:
            subprocess.run(
                [str(self.root / ".venv/bin/alembic"), "upgrade", "head"],
                cwd=self.root,
                env=self.environment,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=60,
                check=True,
            )

    def launch(self, name: str, command: list[str]) -> None:
        with (self.directory / f"{name}.log").open("ab") as output:
            process = subprocess.Popen(
                command,
                cwd=self.root / "apps/web" if name == "web" else self.root,
                env=self.environment,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        identity = fingerprint(process.pid)
        if identity is None:
            raise RuntimeError(
                f"{name} exited immediately; inspect {self.directory / (name + '.log')}"
            )
        self.records[name] = {"pid": process.pid, "fingerprint": identity}
        self.save()

    def wait_ready(self) -> None:
        deadline = time.monotonic() + 30
        urls = {"api": "http://127.0.0.1:8000/api/health", "web": "http://127.0.0.1:3000/"}
        while time.monotonic() < deadline:
            if not all(self.running(name) for name in ("api", "worker", "web")):
                raise RuntimeError(f"A service exited; inspect logs in {self.directory}")
            ready = True
            for url in urls.values():
                try:
                    with urllib.request.urlopen(url, timeout=1) as response:
                        ready = ready and response.status == 200
                except (OSError, urllib.error.URLError):
                    ready = False
            if ready:
                return
            time.sleep(0.25)
        raise RuntimeError(
            f"Services did not become ready within 30 seconds; logs: {self.directory}"
        )

    def start(self) -> None:
        active = {name for name in ("api", "worker", "web") if self.running(name)}
        if active == {"api", "worker", "web"}:
            self.wait_ready()
            print("Services already running; no processes restarted or migrations applied.")
            self.status()
            return
        commands = self.commands()
        self.check_conflicts(active)
        if not active:
            self.migrate()
        else:
            print(
                "Existing services detected; migration skipped. "
                "Use stop/start after schema changes."
            )
        started: list[str] = []
        try:
            for name, command in commands.items():
                if name not in active:
                    self.launch(name, command)
                    started.append(name)
            self.wait_ready()
        except BaseException:
            self.stop(started)
            raise
        self.status()
        print(
            "Open http://127.0.0.1:3000 (GPU/Colab availability is separate from service health)."
        )

    def stop(self, names: list[str] | None = None) -> None:
        failed = []
        for name in names if names is not None else ["web", "worker", "api"]:
            if self.running(name):
                pid = self.records[name]["pid"]
                try:
                    # Each managed process owns a session; never signal an unrelated group.
                    if os.getpgid(pid) != pid:
                        raise RuntimeError(
                            f"Refusing to stop {name}: process group identity changed"
                        )
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 10
                while self.running(name) and time.monotonic() < deadline:
                    time.sleep(0.1)
                if self.running(name):
                    failed.append(name)
                    continue
            self.records.pop(name, None)
        self.save()
        if failed:
            raise RuntimeError(
                f"Still shutting down: {', '.join(failed)}. Retry stop later; no SIGKILL sent."
            )

    def status(self) -> None:
        for name in ("api", "worker", "web"):
            state = (
                f"running (PID {self.records[name]['pid']})" if self.running(name) else "stopped"
            )
            print(f"{name}: {state}")
        print(f"Logs: {self.directory}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop"))
    args = parser.parse_args()
    manager = DevServices()
    manager.directory.mkdir(parents=True, exist_ok=True)
    try:
        with (manager.directory / "launcher.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manager.load()
            getattr(manager, args.action)()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"dev-services: {error}\nLogs: {manager.directory}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
