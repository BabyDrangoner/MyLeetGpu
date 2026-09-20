from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from myleetgpu.runner import cpu_process

SUPERVISOR = Path(cpu_process.__file__).resolve()


def running(pid: int) -> bool:
    """A zombie has ended even when a container's PID 1 has not reaped it yet."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )
    return bool(status.stdout.strip()) and not status.stdout.lstrip().startswith("Z")


def wait_for_exit(pids: list[int], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(running(pid) for pid in pids):
            return
        time.sleep(0.03)
    assert not any(running(pid) for pid in pids), f"test processes remained alive: {pids}"


def read_processes(path: Path, timeout: float = 5.0) -> dict[str, int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            with contextlib.suppress(json.JSONDecodeError):
                return json.loads(path.read_text())
        time.sleep(0.03)
    pytest.fail("test child did not report its process IDs")


@pytest.mark.parametrize("exit_code", [0, 7])
def test_supervisor_preserves_command_exit_code(exit_code: int) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(SUPERVISOR),
            "5",
            str(os.getpid()),
            sys.executable,
            "-I",
            "-S",
            "-c",
            f"print('ready'); raise SystemExit({exit_code})",
        ],
        start_new_session=True,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert completed.returncode == exit_code
    assert completed.stdout.strip() == "ready"


@pytest.mark.parametrize("parent_exit", ["normal", "sigkill"])
def test_worker_exit_stops_supervisor_and_its_sleeping_descendants(
    tmp_path: Path, parent_exit: str
) -> None:
    # Every PID in this test is created by this fixture. No real Worker is touched.
    report = tmp_path / "child.json"
    release = tmp_path / "release"
    child_code = (
        "import json,os,pathlib,subprocess,sys,time; "
        "grandchild=subprocess.Popen([sys.executable,'-I','-S','-c',"
        "'import time; time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({"
        "'supervisor':os.getppid(),'child':os.getpid(),'grandchild':grandchild.pid})); "
        "time.sleep(60)"
    )
    parent_code = (
        "import os,pathlib,subprocess,sys,time\n"
        "subprocess.Popen([sys.executable,'-I','-S',sys.argv[1],'30',str(os.getpid()),"
        "sys.executable,'-I','-S','-c',sys.argv[4],sys.argv[2]],start_new_session=True,"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
        "while not pathlib.Path(sys.argv[3]).exists(): time.sleep(0.01)\n"
    )
    parent = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            parent_code,
            str(SUPERVISOR),
            str(report),
            str(release),
            child_code,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    processes: dict[str, int] = {}
    try:
        processes = read_processes(report)
        assert all(running(pid) for pid in processes.values())
        assert os.getpgid(processes["supervisor"]) == processes["supervisor"]
        assert os.getpgid(processes["child"]) == processes["supervisor"]
        assert os.getpgid(processes["grandchild"]) == processes["supervisor"]
        if parent_exit == "sigkill":
            parent.kill()
        else:
            release.touch()
        parent.wait(timeout=3)
        wait_for_exit(list(processes.values()))
        # Our test process remains alive and was never a member of the killed group.
        assert os.getpgrp() != processes["supervisor"]
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=3)
        if processes and running(processes["supervisor"]):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(processes["supervisor"], signal.SIGKILL)
        if parent.stderr is not None:
            parent.stderr.close()


def test_supervisor_wall_deadline_stops_sleep_without_worker_cleanup(tmp_path: Path) -> None:
    report = tmp_path / "sleep.json"
    command = (
        "import json,os,pathlib,sys,time; "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'child':os.getpid()})); "
        "time.sleep(60)"
    )
    started = time.monotonic()
    supervisor = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            str(SUPERVISOR),
            "0.2",
            str(os.getpid()),
            sys.executable,
            "-I",
            "-S",
            "-c",
            command,
            str(report),
        ],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        processes = read_processes(report)
        assert supervisor.wait(timeout=4) == -signal.SIGKILL
        wait_for_exit(list(processes.values()))
        assert time.monotonic() - started < 4
    finally:
        if supervisor.poll() is None:
            os.killpg(supervisor.pid, signal.SIGKILL)
        supervisor.wait(timeout=3)
        if supervisor.stderr is not None:
            supervisor.stderr.close()


def test_supervisor_refuses_shared_process_group_without_signalling_it() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(SUPERVISOR),
            "1",
            str(os.getpid()),
            sys.executable,
            "-c",
            "raise SystemExit(99)",
        ],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    assert completed.returncode == 1
    assert "own process group and session" in completed.stderr


def test_supervisor_refuses_stale_parent_before_starting_command(tmp_path: Path) -> None:
    output = tmp_path / "must-not-start"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(SUPERVISOR),
            "1",
            "1",
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
            str(output),
        ],
        start_new_session=True,
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    assert completed.returncode == 1
    assert "owner has already exited" in completed.stderr
    assert not output.exists()
