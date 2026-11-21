"""Clean up the temp folder OmniGibson makes for every process that imports it.

``omnigibson.tempdir`` is a ``tempfile.mkdtemp()`` folder holding the scene file and its prebuilt USD (about
860 MB for an ARAT task). OmniGibson removes it in ``og.shutdown()``, which a simulator that is killed, crashes in
PhysX or segfaults on exit never reaches: 26 such folders (16.3 GB) had piled up in /tmp between 2026-11-15 and
2026-11-20, when the system disk ran full.

A launcher marks its own folder with its process identity and, at start, removes the marked folders whose
process is gone. Folders without a marker are only removed when they hold a scene file and nothing in them
has changed for a week.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

MARKER = "dex_teleop_owner.json"
UNMARKED_MAX_AGE_S = 7 * 24 * 3600.0


def _start_ticks(pid: int) -> int | None:
    """The kernel's start time of ``pid`` (clock ticks since boot), or None when there is no such process."""

    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return int(stat.rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def mark_tempdir(path: str | os.PathLike) -> None:
    """Say which process owns ``path`` (pid and start time, so a reused pid is not mistaken for it)."""

    try:
        Path(path, MARKER).write_text(json.dumps({"pid": os.getpid(), "start_ticks": _start_ticks(os.getpid())}))
    except OSError:
        pass  # an unmarked folder is merely not cleaned up automatically


def _owner_gone(folder: Path) -> bool | None:
    """True / False for a marked folder whose owner is gone / alive; None for a folder without a readable marker."""

    try:
        owner = json.loads((folder / MARKER).read_text())
        pid, start = int(owner["pid"]), owner.get("start_ticks")
    except (OSError, ValueError, KeyError, TypeError):
        return None
    now = _start_ticks(pid)
    return now is None or (start is not None and now != start)


def _newest_change(folder: Path) -> float:
    newest = folder.stat().st_mtime
    for entry in folder.iterdir():
        newest = max(newest, entry.stat().st_mtime)
    return newest


def stale_tempdirs(root: str | os.PathLike | None = None, *, now: float | None = None) -> list[Path]:
    """The OmniGibson temp folders under ``root`` (default: the system temp folder) that no process uses."""

    root = Path(tempfile.gettempdir() if root is None else root)
    now = time.time() if now is None else now
    stale = []
    try:
        candidates = sorted(root.glob("tmp*"))
    except OSError:
        return stale
    for folder in candidates:
        try:
            if folder.is_symlink() or not folder.is_dir() or folder.stat().st_uid != os.getuid():
                continue
            gone = _owner_gone(folder)
            if gone is None:
                has_scene = any(folder.glob("scene_file_*.json"))
                gone = has_scene and now - _newest_change(folder) > UNMARKED_MAX_AGE_S
            if gone:
                stale.append(folder)
        except OSError:
            continue
    return stale


def remove_stale_tempdirs(root: str | os.PathLike | None = None) -> tuple[int, int]:
    """Remove them; returns (folders removed, bytes freed). Never raises: cleaning up must not stop a launch."""

    removed = freed = 0
    for folder in stale_tempdirs(root):
        try:
            size = sum(entry.stat().st_size for entry in folder.rglob("*") if entry.is_file() and not entry.is_symlink())
            shutil.rmtree(folder)
        except OSError:
            continue
        removed += 1
        freed += size
    return removed, freed
