"""Append action timing and native-rate EMG to a completed OG recording."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


EMG_GROUP = "emg"
ACTION_TIMING_GROUP = "action_timing"
SYNC_GROUP = "synchronization"
EMG_SCHEMA_VERSION = 1
ACTION_TIMING_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ActionTimingSample:
    """Wall-time boundaries and simulator time for one recorded action."""

    action_apply_monotonic_ns: int
    step_return_monotonic_ns: int
    sim_time_before_s: float
    sim_time_after_s: float

    def __post_init__(self) -> None:
        if self.action_apply_monotonic_ns < 0 or self.step_return_monotonic_ns < 0:
            raise ValueError("Action monotonic timestamps must be non-negative")
        if self.step_return_monotonic_ns < self.action_apply_monotonic_ns:
            raise ValueError("Action step return precedes action application")
        if not np.isfinite(self.sim_time_before_s) or not np.isfinite(self.sim_time_after_s):
            raise ValueError("Action simulator timestamps must be finite")
        if self.sim_time_after_s < self.sim_time_before_s:
            raise ValueError("Simulator time moved backwards during an action")


def _episode_groups(group: h5py.Group) -> list[tuple[int, h5py.Group]]:
    episodes = []
    for name, episode in group.items():
        if not name.startswith("demo_") or not isinstance(episode, h5py.Group):
            continue
        try:
            episode_id = int(name.removeprefix("demo_"))
        except ValueError:
            continue
        episodes.append((episode_id, episode))
    return sorted(episodes)


def write_action_timing_episodes(
    input_path: str | Path,
    episodes: list[list[ActionTimingSample]],
) -> None:
    """Store one timing row for every action recorded by the OG wrapper."""

    input_path = Path(input_path)
    with h5py.File(input_path, "r+") as recording:
        trajectory_episodes = _episode_groups(recording["data"])
        if len(episodes) != len(trajectory_episodes):
            raise RuntimeError(
                f"Action-timing episode count {len(episodes)} does not match trajectory count "
                f"{len(trajectory_episodes)}"
            )

        if ACTION_TIMING_GROUP in recording:
            del recording[ACTION_TIMING_GROUP]
        timing = recording.create_group(ACTION_TIMING_GROUP)
        timing.attrs["schema_version"] = ACTION_TIMING_SCHEMA_VERSION
        timing.attrs["clock"] = "time.monotonic_ns"
        timing.attrs["action_dataset"] = "/data/demo_N/action"
        timing.attrs["row_alignment"] = "one timing row for each action row with the same index"
        timing.attrs["action_semantics"] = "timestamp immediately before env.step(action)"
        timing.attrs["step_return_semantics"] = "timestamp immediately after env.step(action) returns"

        for samples, (episode_id, trajectory) in zip(episodes, trajectory_episodes, strict=True):
            expected_steps = int(trajectory.attrs["num_samples"])
            if len(samples) != expected_steps:
                raise RuntimeError(
                    f"Action timing demo_{episode_id} has {len(samples)} steps; trajectory has {expected_steps}"
                )
            episode = timing.create_group(f"demo_{episode_id}")
            episode.create_dataset(
                "action_apply_monotonic_ns",
                data=np.asarray([sample.action_apply_monotonic_ns for sample in samples], dtype=np.int64),
            )
            episode.create_dataset(
                "step_return_monotonic_ns",
                data=np.asarray([sample.step_return_monotonic_ns for sample in samples], dtype=np.int64),
            )
            episode.create_dataset(
                "sim_time_before_s",
                data=np.asarray([sample.sim_time_before_s for sample in samples], dtype=np.float64),
            )
            episode.create_dataset(
                "sim_time_after_s",
                data=np.asarray([sample.sim_time_after_s for sample in samples], dtype=np.float64),
            )
        recording.flush()


def _recorded_row_range(estimated_time: np.ndarray, action_timing: h5py.Group) -> tuple[int, int]:
    """Half-open EMG row range from the first recorded action's application to the last step's return."""

    first, last = None, None
    for _episode_id, timing in _episode_groups(action_timing):
        apply_times = np.asarray(timing["action_apply_monotonic_ns"], dtype=np.int64)
        return_times = np.asarray(timing["step_return_monotonic_ns"], dtype=np.int64)
        if len(apply_times) == 0:
            continue
        first = int(apply_times[0]) if first is None else min(first, int(apply_times[0]))
        last = int(return_times[-1]) if last is None else max(last, int(return_times[-1]))
    if first is None or len(estimated_time) == 0:
        return 0, 0
    return (
        int(np.searchsorted(estimated_time, first, side="left")),
        int(np.searchsorted(estimated_time, last, side="right")),
    )


def _copy_trimmed_emg(source: h5py.Group, recording: h5py.File, row_start: int, row_end: int) -> None:
    """Copy the EMG group keeping sample rows [row_start, row_end) only (and the callbacks that hold them)."""

    target = recording.create_group(EMG_GROUP)
    for key, value in source.attrs.items():
        target.attrs[key] = value
    for name in source:
        if name not in ("samples", "batches"):
            source.copy(name, target)
    total = int(source["samples/sample_index"].shape[0])
    samples = target.create_group("samples")
    first_rows = np.asarray(source["batches/first_sample_row"], dtype=np.int64)
    counts = np.asarray(source["batches/sample_count"], dtype=np.int64)
    kept = np.flatnonzero((first_rows + counts > row_start) & (first_rows < row_end)) if row_end > row_start else np.zeros(0, dtype=np.int64)
    first_batch = int(kept[0]) if len(kept) else 0
    for name, dataset in source["samples"].items():
        data = dataset[row_start:row_end]
        if name == "batch_id":
            data = data - first_batch
        samples.create_dataset(name, data=data, dtype=dataset.dtype, chunks=True, maxshape=(None, *dataset.shape[1:]))
    batches = target.create_group("batches")
    batch_slice = slice(first_batch, first_batch + len(kept))
    for name, dataset in source["batches"].items():
        data = dataset[batch_slice]
        if name == "first_sample_row":
            data = np.maximum(first_rows[batch_slice], row_start) - row_start
        elif name == "sample_count":
            ends = np.minimum(first_rows[batch_slice] + counts[batch_slice], row_end)
            data = (ends - np.maximum(first_rows[batch_slice], row_start)).astype(dataset.dtype)
        batches.create_dataset(name, data=data, dtype=dataset.dtype, chunks=True, maxshape=(None, *dataset.shape[1:]))
    target.attrs["trimmed_to_recording"] = True
    target.attrs["trim_semantics"] = (
        "samples before the first recorded action (the first SPACE) and after the last recorded step were "
        "dropped; their host times were estimated on the whole stream before the cut"
    )
    target.attrs["untrimmed_sample_count"] = total
    target.attrs["trimmed_rows_before"] = int(row_start)
    target.attrs["trimmed_rows_after"] = int(total - row_end)
    target.attrs["sample_count"] = int(row_end - row_start)
    target.attrs["batch_count"] = int(len(kept))
    if row_end > row_start:
        target.attrs["first_sample_index"] = int(samples["sample_index"][0])
        target.attrs["estimated_first_sample_monotonic_ns"] = int(samples["estimated_monotonic_ns"][0])
    if "fitted_segment_first_row" in source.attrs:
        starts = np.asarray(source.attrs["fitted_segment_first_row"], dtype=np.int64)
        ends = np.append(starts[1:], total)
        overlap = (ends > row_start) & (starts < row_end)
        target.attrs["fitted_segment_first_row"] = np.maximum(starts[overlap], row_start) - row_start
        for key in ("fitted_sample_period_ns", "fitted_segment_period_measured"):
            if key in source.attrs:
                target.attrs[key] = np.asarray(source.attrs[key])[overlap]


GLOVE_STREAM = "hand_tracking/articulation_streams/articulation.manus"


def read_glove_ergonomics(recording_path: str | Path, stream: str = GLOVE_STREAM) -> dict[str, np.ndarray]:
    """The MANUS ergonomics angles a glove teleoperation recording carries with its glove frames (bridges from
    2026-11-16 on): the EMG decoder's training labels, for evaluating a decoder on teleoperation takes.

    Returns arrays over the frames that carry them: ``capture_monotonic_ns`` (the glove frame), ``publish_time`` and
    ``arrival_monotonic_ns`` (the ergonomics callback; consecutive frames can repeat the same ergonomics: tell them
    apart by ``arrival_monotonic_ns``, the MANUS Integrated SDK 3.2.1 leaves the ergonomics ``publish_time`` at 0) and
    ``angles_deg`` (n, 20) in MANUS SDK 3.2 enum order (manus_python_bridge.ERGONOMICS_ORDER), the order HandSync
    records; dexemg's ``reindex_to_emg2pose`` turns it into the decoder's joint order.
    """

    import json

    with h5py.File(recording_path, "r") as recording:
        if stream not in recording:
            raise KeyError(f"{recording_path}: no glove stream {stream}")
        group = recording[stream]
        capture = np.asarray(group["capture_monotonic_ns"], dtype=np.int64)
        provenance = [json.loads(value.decode() if isinstance(value, bytes) else value) for value in group["provenance_json"][:]]
    rows = [k for k, entry in enumerate(provenance) if "ergonomics_deg" in entry]
    if not rows:
        raise ValueError(f"{recording_path}: the glove frames carry no ergonomics (recorded before 2026-11-16?)")
    return {
        "capture_monotonic_ns": capture[rows],
        "publish_time": np.asarray([provenance[k]["ergonomics_publish_time"] for k in rows], dtype=np.int64),
        "arrival_monotonic_ns": np.asarray([provenance[k]["ergonomics_arrival_monotonic_ns"] for k in rows], dtype=np.int64),
        "angles_deg": np.asarray([json.loads(provenance[k]["ergonomics_deg"]) for k in rows], dtype=np.float32),
    }


def merge_emg_recording(recording_path: str | Path, emg_path: str | Path, *, trim_to_recording: bool = False) -> None:
    """Copy a closed sidecar recording and derive per-episode EMG row ranges.

    ``trim_to_recording``: keep only the EMG between the first recorded action (the operator's first SPACE)
    and the end of the last recorded step, like every other stream of the recording. The sidecar streams
    from launch to exit; the samples outside that span are dropped here.
    """

    recording_path = Path(recording_path)
    emg_path = Path(emg_path)
    with h5py.File(emg_path, "r") as emg_recording, h5py.File(recording_path, "r+") as recording:
        if EMG_GROUP not in emg_recording:
            raise RuntimeError(f"EMG sidecar recording has no '{EMG_GROUP}' group: {emg_path}")
        if ACTION_TIMING_GROUP not in recording:
            raise RuntimeError("Action timing must be written before EMG synchronization")
        if EMG_GROUP in recording:
            del recording[EMG_GROUP]
        if trim_to_recording:
            source = emg_recording[EMG_GROUP]
            row_start, row_end = _recorded_row_range(
                np.asarray(source["samples/estimated_monotonic_ns"], dtype=np.int64), recording[ACTION_TIMING_GROUP]
            )
            _copy_trimmed_emg(source, recording, row_start, row_end)
        else:
            emg_recording.copy(EMG_GROUP, recording)
        emg = recording[EMG_GROUP]
        emg.attrs["schema_version"] = EMG_SCHEMA_VERSION

        estimated_time = np.asarray(emg["samples/estimated_monotonic_ns"], dtype=np.int64)
        if len(estimated_time) > 1 and np.any(np.diff(estimated_time) <= 0):
            raise RuntimeError("Estimated EMG timestamps are not strictly increasing")
        if SYNC_GROUP in recording:
            del recording[SYNC_GROUP]
        synchronization = recording.create_group(SYNC_GROUP)
        synchronization.attrs["schema_version"] = 1
        synchronization.attrs["mapping"] = (
            "EMG half-open row ranges aligned to action application times in the desktop monotonic clock"
        )

        timing_episodes = _episode_groups(recording[ACTION_TIMING_GROUP])
        trajectory_episodes = _episode_groups(recording["data"])
        if [item[0] for item in timing_episodes] != [item[0] for item in trajectory_episodes]:
            raise RuntimeError("Action timing episode IDs do not match trajectory episode IDs")

        for episode_id, timing in timing_episodes:
            apply_times = np.asarray(timing["action_apply_monotonic_ns"], dtype=np.int64)
            return_times = np.asarray(timing["step_return_monotonic_ns"], dtype=np.int64)
            if len(apply_times) > 1 and np.any(np.diff(apply_times) <= 0):
                raise RuntimeError(f"Action application timestamps move backwards in demo_{episode_id}")
            if len(apply_times) == 0 or len(estimated_time) == 0:
                episode_row_start = episode_row_end = 0
                action_row_start = np.zeros(len(apply_times), dtype=np.int64)
                action_row_end = np.zeros(len(apply_times), dtype=np.int64)
            else:
                episode_row_start = int(np.searchsorted(estimated_time, apply_times[0], side="left"))
                episode_row_end = int(np.searchsorted(estimated_time, return_times[-1], side="right"))
                action_end_times = np.concatenate((apply_times[1:], return_times[-1:]))
                action_row_start = np.searchsorted(estimated_time, apply_times, side="left").astype(np.int64)
                action_row_end = np.searchsorted(estimated_time, action_end_times, side="left").astype(np.int64)
                action_row_end[-1] = episode_row_end
            episode = synchronization.create_group(f"demo_{episode_id}")
            episode.create_dataset("episode_emg_row_start", data=np.int64(episode_row_start))
            episode.create_dataset("episode_emg_row_end", data=np.int64(episode_row_end))
            episode.create_dataset("action_emg_row_start", data=action_row_start)
            episode.create_dataset("action_emg_row_end", data=action_row_end)
            episode.attrs["range_semantics"] = "half-open [row_start, row_end) into /emg/samples"
            episode.attrs["action_interval_semantics"] = (
                "action apply through next action apply; the final action ends when env.step returns"
            )
        recording.flush()
