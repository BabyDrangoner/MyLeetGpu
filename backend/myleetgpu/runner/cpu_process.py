"""Supervise one CPU command independently of the threaded Worker's lifetime.

These are accident/resource guards for trusted local code, not a security sandbox.
The caller must start this process in a new session and pass its own PID. Commands
inherit that dedicated process group, so a dead Worker cannot orphan sleeping or
I/O-bound submissions which RLIMIT_CPU alone would never stop.
"""

from __future__ import annotations

import math
import os
import resource
import signal
import subprocess
import sys
import time

POLL_SECONDS = 0.05
# CpuRunner owns the normal deadline/result classification. This fallback fires
# slightly later so a live Worker can continue reporting CommandResult.timed_out.
WATCHDOG_GRACE_SECONDS = 0.5


def apply_limits(timeout: float) -> None:
    cpu_seconds = max(1, math.ceil(timeout) + 1)
    limits = [
        (resource.RLIMIT_CORE, 0),
        (resource.RLIMIT_CPU, cpu_seconds),
        (resource.RLIMIT_FSIZE, 64 * 1024 * 1024),
        (resource.RLIMIT_NOFILE, 256),
    ]
    # Darwin's large shared-cache address space makes a small RLIMIT_AS invalid.
    if sys.platform.startswith("linux"):
        limits.append((resource.RLIMIT_AS, 2 * 1024**3))
    for kind, maximum in limits:
        _, hard = resource.getrlimit(kind)
        value = maximum if hard == resource.RLIM_INFINITY else min(maximum, hard)
        resource.setrlimit(kind, (value, value))


def main() -> int:
    try:
        timeout = float(sys.argv[1])
        parent_pid = int(sys.argv[2])
    except (IndexError, ValueError) as error:
        raise SystemExit("invalid CPU execution request") from error
    command = sys.argv[3:]
    if not command or not math.isfinite(timeout) or timeout <= 0 or parent_pid <= 0:
        raise SystemExit("invalid CPU execution request")

    group_id = os.getpid()
    # Never signal an inherited/shared group, even if invoked incorrectly. The
    # explicit parent PID also handles Worker death before this script starts.
    if os.getpgrp() != group_id or os.getsid(0) != group_id:
        raise SystemExit("CPU supervisor requires its own process group and session")
    if os.getppid() != parent_pid:
        raise SystemExit("CPU execution owner has already exited")

    def stop_group(_signum: int = 0, _frame: object = None) -> None:
        os.killpg(group_id, signal.SIGKILL)

    signal.signal(signal.SIGTERM, stop_group)
    signal.signal(signal.SIGINT, stop_group)
    deadline = time.monotonic() + timeout + WATCHDOG_GRACE_SECONDS
    apply_limits(timeout)
    child = subprocess.Popen(command, close_fds=True)
    while True:
        if os.getppid() != parent_pid or time.monotonic() >= deadline:
            stop_group()
        returncode = child.poll()
        if returncode is not None:
            return returncode if returncode >= 0 else 128 - returncode
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
