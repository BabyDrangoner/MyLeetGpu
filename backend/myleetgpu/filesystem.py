from __future__ import annotations

import errno
import stat
from pathlib import Path


def ensure_mode(path: Path, mode: int) -> None:
    """Apply a mode, tolerating metadata-less DrvFS when required bits already exist.

    WSL DrvFS mounts without metadata can report bind-mounted files as owned by
    root and reject chmod with EPERM even though their effective mode is 0777.
    Docker mount flags remain the security boundary; this fallback only proves
    that every bit required for host/container operation is already present.
    """

    try:
        path.chmod(mode)
    except OSError as error:
        tolerated = {errno.EPERM, errno.EOPNOTSUPP}
        if hasattr(errno, "ENOTSUP"):
            tolerated.add(errno.ENOTSUP)
        actual = stat.S_IMODE(path.stat().st_mode)
        if error.errno not in tolerated or actual & mode != mode:
            raise
