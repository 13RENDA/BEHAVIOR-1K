"""OmniGibson temp folders left by simulators that never reached og.shutdown() are removed at the next launch."""

import json
import os
import subprocess
import sys
import time

from dex_teleop.omnigibson import tempdirs
from dex_teleop.omnigibson.tempdirs import MARKER, mark_tempdir, remove_stale_tempdirs, stale_tempdirs


def _folder(root, name, *, scene=True, marker=None, age_s=0.0):
    folder = root / name
    folder.mkdir()
    if scene:
        (folder / "scene_file_abc.json").write_text("{}")
        (folder / "scene_file_abc.jsonx1.usd").write_bytes(b"u" * 1000)
    if marker is not None:
        (folder / MARKER).write_text(json.dumps(marker))
    if age_s:
        old = time.time() - age_s
        for path in [*folder.iterdir(), folder]:
            os.utime(path, (old, old))
    return folder


def test_marked_folders_go_when_their_process_is_gone_and_stay_while_it_lives(tmp_path):
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait()
    dead = _folder(tmp_path, "tmpdead0001", marker={"pid": finished.pid, "start_ticks": 1})
    mine = tmp_path / "tmpmine0001"
    mine.mkdir()
    (mine / "scene_file_abc.json").write_text("{}")
    mark_tempdir(mine)
    assert json.loads((mine / MARKER).read_text())["pid"] == os.getpid()
    # this process's pid with another start time: the pid was reused, the owner is gone
    reused = _folder(tmp_path, "tmpreuse001", marker={"pid": os.getpid(), "start_ticks": 1})
    assert stale_tempdirs(tmp_path) == [dead, reused]
    removed, freed = remove_stale_tempdirs(tmp_path)
    assert removed == 2 and freed >= 2000
    assert mine.is_dir() and not dead.exists() and not reused.exists()


def test_unmarked_folders_need_a_scene_file_and_a_week_of_silence(tmp_path):
    week = tempdirs.UNMARKED_MAX_AGE_S
    old_scene = _folder(tmp_path, "tmpold00001", age_s=week + 3600)
    recent_scene = _folder(tmp_path, "tmpnew00001", age_s=week - 3600)
    other_tool = _folder(tmp_path, "tmpother001", scene=False, age_s=10 * week)
    not_a_temp_name = _folder(tmp_path, "scratch", age_s=10 * week)
    unreadable_marker = _folder(tmp_path, "tmpbadmark1", age_s=10 * week)
    (unreadable_marker / MARKER).write_text("not json")
    old = time.time() - 10 * week
    os.utime(unreadable_marker / MARKER, (old, old))
    os.utime(unreadable_marker, (old, old))
    assert stale_tempdirs(tmp_path) == [unreadable_marker, old_scene]
    assert remove_stale_tempdirs(tmp_path)[0] == 2
    assert recent_scene.is_dir() and other_tool.is_dir() and not_a_temp_name.is_dir()


def test_cleaning_up_never_raises(tmp_path):
    assert remove_stale_tempdirs(tmp_path / "missing") == (0, 0)
    mark_tempdir(tmp_path / "missing")  # no folder: silently skipped
