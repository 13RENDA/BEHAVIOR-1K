"""Glove-free VIVE wrist: same pose as the MANUS path, live source over a fake VIVE Hub socket."""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dex_teleop.tracking.base import SourceUnavailableError
from dex_teleop.tracking.manus_calibration import ManusCoreCalibration
from dex_teleop.tracking.manus_python_bridge import DEFAULT_TRACKER_TO_CASING_TRANSLATION, compose_tracker_wrist
from dex_teleop.tracking.transforms import RigidTransform
from dex_teleop.tracking.vive_ultimate import STATE_READY, STATE_SYNCING, vut_pose_to_core_world
from dex_teleop.tracking.vive_wrist import (
    DEFAULT_TRACKER_TO_WRIST_TRANSLATION,
    ViveUltimateWristSource,
    default_tracker_to_wrist,
    vive_wrist_pose,
)
from dex_teleop.types import Handedness

from fake_vut_server import FakeVutServer


def test_default_offset_is_the_manus_sidecars():
    for side in ("left", "right"):
        assert DEFAULT_TRACKER_TO_WRIST_TRANSLATION[Handedness(side)] == DEFAULT_TRACKER_TO_CASING_TRANSLATION[side]


@pytest.mark.parametrize("frame", ["x_right_y_up_z_back", "x_forward_z_up"])
def test_wrist_equals_manus_bridge_composition_plus_default_calibration(frame):
    rng = np.random.default_rng(0)
    offset = default_tracker_to_wrist("right")
    calibration = ManusCoreCalibration.default()
    for _ in range(20):
        position = rng.uniform(-1.0, 1.0, 3)
        quaternion = Rotation.random(random_state=rng).as_quat()
        # MANUS path: the sidecar re-roots the raw skeleton at tracker * offset, ManusSource applies the
        # default Core calibration to the wrist node.
        skeleton = [(7, rng.uniform(-1, 1, 3), Rotation.random(random_state=rng).as_quat())]
        tracker_core = vut_pose_to_core_world(position, quaternion, frame=frame)
        (_, node_position, node_quaternion), = compose_tracker_wrist(skeleton, 7, tracker_core, offset)
        manus = calibration.apply(Handedness.RIGHT, RigidTransform(node_position, node_quaternion))
        vive = vive_wrist_pose(position, quaternion, handedness="right", tracker_to_wrist=offset, frame=frame)
        np.testing.assert_allclose(vive.translation, manus.translation, atol=1e-9)
        assert abs(abs(float(np.dot(vive.quaternion_xyzw, manus.quaternion_xyzw))) - 1.0) < 1e-9


def test_source_streams_trackable_poses_and_fails_when_tracking_is_lost(tmp_path):
    socket_path = str(tmp_path / "vut.sock")
    server = FakeVutServer(socket_path, serial="FA5AG3B00121", rate_hz=200.0, state=STATE_READY,
                           extra_trackers={3: "OTHER"}).start()
    source = ViveUltimateWristSource(socket_path, tracker_id="FA5AG3B00121", stale_timeout=0.2)
    try:
        source.start()
        deadline = time.monotonic() + 3.0
        while source.read_wrist(Handedness.RIGHT) is None and time.monotonic() < deadline:
            time.sleep(0.02)
        sample = source.read_wrist(Handedness.RIGHT)
        assert sample is not None and sample.source == "vive_ultimate"
        assert sample.provenance["validated_tracker_id"] == "FA5AG3B00121"
        assert sample.provenance["validated_tracker_quality"] == "trackable"
        expected = vive_wrist_pose(server.position, server.quaternion_xyzw, handedness="right",
                                   tracker_to_wrist=default_tracker_to_wrist("right"))
        np.testing.assert_allclose(sample.position, expected.translation, atol=1e-5)
        time.sleep(0.1)  # 200 Hz server: ~20 more poses
        assert len(source.drain_wrists(Handedness.RIGHT)) >= 2
        source.check_health()
        server.state = STATE_SYNCING
        time.sleep(0.5)
        with pytest.raises(SourceUnavailableError, match="FA5AG3B00121"):
            source.check_health()
    finally:
        source.close()
        server.close()


def test_source_names_a_missing_vive_hub(tmp_path):
    source = ViveUltimateWristSource(str(tmp_path / "absent.sock"), tracker_id="X", connect_timeout=0.3)
    with pytest.raises(SourceUnavailableError, match="run-vivehub"):
        source.start()
