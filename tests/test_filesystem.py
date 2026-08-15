from __future__ import annotations

import errno
from pathlib import Path

import pytest
from myleetgpu.filesystem import ensure_mode


def test_ensure_mode_accepts_drvfs_eperm_when_required_bits_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "program"
    target.write_bytes(b"binary")
    target.chmod(0o777)

    monkeypatch.setattr(Path, "chmod", lambda self, mode: _raise_eperm())

    ensure_mode(target, 0o555)


def test_ensure_mode_rejects_eperm_when_required_bits_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "program"
    target.write_bytes(b"binary")
    target.chmod(0o600)

    monkeypatch.setattr(Path, "chmod", lambda self, mode: _raise_eperm())

    with pytest.raises(PermissionError):
        ensure_mode(target, 0o555)


def _raise_eperm() -> None:
    raise PermissionError(errno.EPERM, "operation not permitted")
