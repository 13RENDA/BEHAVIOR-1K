"""F = reset + mark the episode failed: labels written into the recording, read back for analysis."""

import json

import h5py
import numpy as np
import pytest

from dex_teleop.omnigibson.episode_labels import (
    EPISODE_LABEL_ATTRIBUTE, FAILED_EPISODES_ATTRIBUTE, kept_episodes, read_failed_episodes, write_episode_labels)


def _recording(path, lengths):
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for k, n in enumerate(lengths):
            g = data.create_group(f"demo_{k}")
            g.attrs["num_samples"] = n
            g.create_dataset("action", data=np.zeros((n, 3)))


def test_failed_episodes_are_labelled_and_skipped(tmp_path):
    path = tmp_path / "rec.hdf5"
    _recording(path, [120, 80, 95, 0])  # the last one: an empty episode after the final reset
    write_episode_labels(path, {1})
    with h5py.File(path, "r") as f:
        assert [f[f"data/demo_{k}"].attrs[EPISODE_LABEL_ATTRIBUTE] for k in range(4)] == ["kept", "failed", "kept", "kept"]
        assert json.loads(f["data"].attrs[FAILED_EPISODES_ATTRIBUTE]) == [1]
    assert read_failed_episodes(path) == [1]
    assert kept_episodes(path) == [0, 2]


def test_no_failed_episode_and_old_recordings(tmp_path):
    path = tmp_path / "rec.hdf5"
    _recording(path, [10, 20])
    assert read_failed_episodes(path) == [] and kept_episodes(path) == [0, 1]  # before F existed: no attribute
    write_episode_labels(path, set())
    assert read_failed_episodes(path) == []
    with pytest.raises(RuntimeError, match="not in the recording"):
        write_episode_labels(path, {5})


def test_launcher_binds_f_to_reset_and_mark_failed():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src/dex_teleop/omnigibson/launcher.py").read_text()
    assert "KeyboardInput.F, request_reset_failed" in source
    assert 'pending_command == "reset_failed"' in source and "failed_episodes=failed_episodes" in source
