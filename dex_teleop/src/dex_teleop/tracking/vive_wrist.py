"""Glove-free wrist pose from the VIVE Tracker Ultimate (``--wrist-source vive_ultimate``).

Reads VIVE Hub for Linux's ``tracker_server`` socket directly (no MANUS SDK, no glove) and produces the
same anatomical wrist pose the MANUS path produces with its default calibration:

    wrist = reference_from_core_world * vut_pose_to_core_world(tracker) * tracker_to_wrist

i.e. exactly ``manus_python_bridge.compose_tracker_wrist`` followed by ``ManusCoreCalibration.default()``
(checked in ``tests/test_vive_wrist.py``), so ``tracking_yaw_deg``, ``position_sensitivity``, the
anchoring at engagement and the tracker-to-wrist offset behave as with ``--wrist-source manus``.
Only ``trackable`` poses of the expected serial are accepted; ``check_health`` fails once the tracker has
not produced one for ``stale_timeout`` seconds, like ``manus_tracker_stale_timeout``.
"""

from __future__ import annotations

from collections import deque
import threading
import time
from typing import Any

import numpy as np

from dex_teleop.tracking.base import SourceUnavailableError
from dex_teleop.tracking.manus_calibration import MANUS_DEFAULT_REFERENCE_FRAME, ManusCoreCalibration
from dex_teleop.tracking.transforms import RigidTransform, compose_transforms
from dex_teleop.tracking.vive_ultimate import (
    VUT_DEFAULT_FRAME,
    VUT_DEFAULT_SOCKET,
    VutPoseFeed,
    VutTrackerPose,
    vut_pose_to_core_world,
)
from dex_teleop.types import OPENXR_ANATOMICAL_WRIST_FRAME, Handedness, WristPoseSample

VIVE_SOURCE_NAME = "vive_ultimate"
# Same numbers as manus_python_bridge.DEFAULT_TRACKER_TO_CASING_TRANSLATION (MANUS Universal Mount example
# offset, x mirrored for the right hand); duplicated to keep this module free of the MANUS sidecar import.
DEFAULT_TRACKER_TO_WRIST_TRANSLATION = {
    Handedness.LEFT: (0.003, -0.058, -0.043),
    Handedness.RIGHT: (-0.003, -0.058, -0.043),
}


def default_tracker_to_wrist(handedness: Handedness | str) -> RigidTransform:
    return RigidTransform(DEFAULT_TRACKER_TO_WRIST_TRANSLATION[Handedness(handedness)], (0.0, 0.0, 0.0, 1.0))


def vive_wrist_pose(
    position,
    quaternion_xyzw,
    *,
    handedness: Handedness | str,
    tracker_to_wrist: RigidTransform,
    frame: str = VUT_DEFAULT_FRAME,
) -> RigidTransform:
    """Tracker pose from VIVE Hub -> anatomical wrist in the launcher's tracking frame."""

    core = vut_pose_to_core_world(position, quaternion_xyzw, frame=frame)
    wrist_core = compose_transforms(core, tracker_to_wrist)
    return ManusCoreCalibration.default().apply(Handedness(handedness), wrist_core)


class ViveUltimateWristSource:
    """A draining wrist source fed by one VIVE Tracker Ultimate."""

    def __init__(
        self,
        socket_path: str = VUT_DEFAULT_SOCKET,
        *,
        tracker_id: str,
        frame: str = VUT_DEFAULT_FRAME,
        handedness: Handedness | str = Handedness.RIGHT,
        tracker_to_wrist: RigidTransform | None = None,
        stale_timeout: float = 1.0,
        connect_timeout: float = 5.0,
        queue_capacity: int = 8192,
    ) -> None:
        if not tracker_id:
            raise ValueError("A VIVE wrist source needs the tracker serial (--vive-tracker-id)")
        if not np.isfinite(stale_timeout) or stale_timeout <= 0:
            raise ValueError("stale_timeout must be positive")
        self.socket_path = str(socket_path)
        self.tracker_id = str(tracker_id)
        self.frame = frame
        self.handedness = Handedness(handedness)
        self.tracker_to_wrist = tracker_to_wrist or default_tracker_to_wrist(self.handedness)
        self.stale_timeout = float(stale_timeout)
        self.connect_timeout = float(connect_timeout)
        self._capacity = int(queue_capacity)
        self._lock = threading.Lock()
        self._queue: deque[WristPoseSample] = deque()
        self._latest: WristPoseSample | None = None
        self._last_trackable: float | None = None
        self._last_seen: VutTrackerPose | None = None
        self._other_ids: set[str] = set()
        self._overflow = False
        self._feed: VutPoseFeed | None = None
        self._sequence = 0

    def start(self) -> None:
        if self._feed is not None:
            return
        feed = VutPoseFeed(self.socket_path, listener=self._on_pose)
        feed.start()
        deadline = time.monotonic() + self.connect_timeout
        while not feed.connected:
            if time.monotonic() >= deadline:
                feed.close()
                raise SourceUnavailableError(
                    f"VIVE Hub's tracker_server is not answering on {self.socket_path}; start VIVE Hub "
                    "(./run-vivehub.sh) and wake the tracker"
                )
            time.sleep(0.05)
        self._feed = feed

    def close(self) -> None:
        feed, self._feed = self._feed, None
        if feed is not None:
            feed.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_exc):
        self.close()

    def _on_pose(self, pose: VutTrackerPose) -> None:
        if pose.tracker_id != self.tracker_id:
            with self._lock:
                self._other_ids.add(pose.tracker_id)
            return
        quality = pose.quality
        with self._lock:
            self._last_seen = pose
        if quality != "trackable" or not pose.pose_valid:
            return
        wrist = vive_wrist_pose(pose.position, pose.quaternion_xyzw, handedness=self.handedness,
                                tracker_to_wrist=self.tracker_to_wrist, frame=self.frame)
        timestamp = pose.receipt_monotonic_ns / 1e9
        with self._lock:
            self._sequence += 1
            sample = WristPoseSample(
                timestamp=timestamp,
                receipt_timestamp=timestamp,
                source_timestamp_ns=int(pose.receipt_monotonic_ns),
                source_frame_id=self._sequence,
                handedness=self.handedness,
                position=wrist.translation,
                quaternion_xyzw=wrist.quaternion_xyzw,
                source=VIVE_SOURCE_NAME,
                reference_frame=MANUS_DEFAULT_REFERENCE_FRAME,
                anatomical_frame=OPENXR_ANATOMICAL_WRIST_FRAME,
                provenance={
                    "validated_tracker_id": self.tracker_id,
                    "validated_tracker_quality": quality,
                    "tracker_health": "healthy",
                    "tracker_health_reason": None,
                    "tracking_system": "vive_hub_linux",
                    "vut_state": pose.state_name,
                    "tracker_timestamp_us": int(pose.timestamp_us),
                },
            )
            if len(self._queue) >= self._capacity:
                self._overflow = True
            else:
                self._queue.append(sample)
            self._latest = sample
            self._last_trackable = time.monotonic()

    def read_wrist(self, handedness: Handedness) -> WristPoseSample | None:
        if Handedness(handedness) is not self.handedness:
            return None
        with self._lock:
            return self._latest

    def drain_wrists(self, handedness: Handedness) -> tuple[WristPoseSample, ...]:
        if Handedness(handedness) is not self.handedness:
            return ()
        with self._lock:
            samples = tuple(self._queue)
            self._queue.clear()
        return samples

    def check_health(self) -> None:
        with self._lock:
            overflow, last, seen, others = self._overflow, self._last_trackable, self._last_seen, set(self._other_ids)
        if overflow:
            raise SourceUnavailableError("VIVE wrist queue overflowed; samples were lost")
        if self._feed is None or last is None:
            return
        age = time.monotonic() - last
        if age > self.stale_timeout:
            if seen is None:
                detail = f"tracker {self.tracker_id} sends nothing (other trackers: {sorted(others) or 'none'})"
            else:
                detail = f"tracker {self.tracker_id} is {seen.state_name}/{seen.quality}"
            raise SourceUnavailableError(
                f"VIVE wrist stale for {age:.2f} s: {detail}; uncover the tracker's cameras or rebuild the map"
            )

    def recording_metadata(self) -> dict[str, Any]:
        return {
            "provider": "VIVE Tracker Ultimate via VIVE Hub for Linux (no MANUS)",
            "socket": self.socket_path,
            "vive_ultimate_frame": self.frame,
            "tracker_id": self.tracker_id,
            "tracker_to_wrist": self.tracker_to_wrist.as_mapping(),
            "reference_frame": MANUS_DEFAULT_REFERENCE_FRAME,
            "composition": "reference_from_core_world * vut_pose_to_core_world(tracker) * tracker_to_wrist "
                           "(= MANUS sidecar bridge composition + default Core calibration)",
            "stale_timeout_s": self.stale_timeout,
        }


__all__ = [
    "DEFAULT_TRACKER_TO_WRIST_TRANSLATION",
    "VIVE_SOURCE_NAME",
    "ViveUltimateWristSource",
    "default_tracker_to_wrist",
    "vive_wrist_pose",
]
