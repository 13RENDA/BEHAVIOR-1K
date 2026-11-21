"""Operator labels of the episodes of a teleoperation recording (2026-11-16).

R resets and starts the next episode; F does the same but marks the episode that just ended as failed. Every
episode stays in the file (demo_N, its EMG rows, its hand tracking); the label only says which ones to skip:
``data/demo_N.attrs["dex_teleop_episode_label"]`` is "failed" or "kept", and ``data.attrs["dex_teleop_failed_episodes"]``
lists the failed episode numbers as JSON.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

EPISODE_LABEL_ATTRIBUTE = "dex_teleop_episode_label"
FAILED_EPISODES_ATTRIBUTE = "dex_teleop_failed_episodes"
FAILED, KEPT = "failed", "kept"


def write_episode_labels(recording_path: str | Path, failed_episodes: Iterable[int]) -> None:
    import h5py

    failed = sorted({int(episode) for episode in failed_episodes})
    with h5py.File(Path(recording_path), "r+") as recording:
        data = recording["data"]
        demos = {int(name.removeprefix("demo_")) for name in data if name.startswith("demo_")}
        missing = [episode for episode in failed if episode not in demos]
        if missing:
            raise RuntimeError(f"Episodes marked failed are not in the recording: {missing}")
        for episode in demos:
            data[f"demo_{episode}"].attrs[EPISODE_LABEL_ATTRIBUTE] = FAILED if episode in failed else KEPT
        data.attrs[FAILED_EPISODES_ATTRIBUTE] = json.dumps(failed)


def read_failed_episodes(recording_path: str | Path) -> list[int]:
    """Episode numbers the operator marked failed with F ([] for recordings made before F existed)."""

    import h5py

    with h5py.File(Path(recording_path), "r") as recording:
        value = recording["data"].attrs.get(FAILED_EPISODES_ATTRIBUTE)
    return [] if value is None else [int(episode) for episode in json.loads(value)]


def kept_episodes(recording_path: str | Path) -> list[int]:
    """Episode numbers not marked failed and with at least one step, in order."""

    import h5py

    failed = set(read_failed_episodes(recording_path))
    with h5py.File(Path(recording_path), "r") as recording:
        data = recording["data"]
        episodes = sorted(int(name.removeprefix("demo_")) for name in data if name.startswith("demo_"))
        return [
            episode for episode in episodes
            if episode not in failed and int(data[f"demo_{episode}"].attrs.get("num_samples", 0)) > 0
        ]
