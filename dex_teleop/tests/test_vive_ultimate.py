"""VIVE Hub for Linux tracker_server protocol client."""

import struct
import time

import numpy as np
import pytest

from dex_teleop.tracking.manus_calibration import (
    MANUS_CORE_WORLD_TO_TRACKING_QUATERNION_XYZW,
)
from dex_teleop.tracking.transforms import RigidTransform, compose_transforms
from dex_teleop.tracking.vive_ultimate import (
    FLAG_ROTATION_ONLY,
    FLAG_SETUP_REQUIRED,
    HEADER_SIZE,
    MSG_UPDATE,
    STATE_READY,
    STATE_SYNCING,
    UPDATE_SIZE,
    VutIpcClient,
    VutPoseFeed,
    decode_update,
    encode_header,
    encode_update_payload,
    select_tracker,
    vut_pose_to_core_world,
)

from fake_vut_server import FakeVutServer


def _wait(predicate, timeout=3.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_wire_format_matches_the_sdk_reference_layout():
    header = encode_header(3, MSG_UPDATE, UPDATE_SIZE, bytes((STATE_READY, 1, 2, 90)))
    assert len(header) == HEADER_SIZE == 16
    tracker_id, message, payload_size, params = struct.unpack("<IBB10s", header)
    assert (tracker_id, message, payload_size) == (3, MSG_UPDATE, 64)
    assert params[:4] == bytes((STATE_READY, 1, 2, 90))

    payload = encode_update_payload(
        (0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0), timestamp_us=123456, flags=FLAG_ROTATION_ONLY
    )
    pose = decode_update(3, params, payload, tracker_id="SN1", receipt_monotonic_ns=7)
    assert pose.position.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert pose.quaternion_xyzw.tolist() == [0.0, 0.0, 0.0, 1.0]
    assert pose.timestamp_us == 123456
    assert pose.state == STATE_READY and pose.state_name == "ready"
    assert pose.battery == 90 and pose.button == 1 and pose.clicks == 2
    assert pose.rotation_only and pose.quality == "bad"
    with pytest.raises(ValueError, match="64 bytes"):
        decode_update(3, params, payload[:10], tracker_id="SN1", receipt_monotonic_ns=7)


@pytest.mark.parametrize(
    ("state", "flags", "quality"),
    (
        (STATE_READY, 0, "trackable"),
        (STATE_READY, FLAG_ROTATION_ONLY, "bad"),
        (STATE_READY, FLAG_SETUP_REQUIRED, "untrackable"),
        (STATE_SYNCING, 0, "untrackable"),
    ),
)
def test_quality_follows_state_and_flags(state, flags, quality):
    payload = encode_update_payload((0, 0, 0), (0, 0, 0, 1), timestamp_us=1, flags=flags)
    pose = decode_update(0, bytes((state, 0, 0, 50)), payload, tracker_id="x", receipt_monotonic_ns=1)
    assert pose.quality == quality


def test_openxr_frame_is_core_world_and_isaac_frame_is_rotated_into_it():
    identity = vut_pose_to_core_world((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    assert identity.translation.tolist() == [1.0, 2.0, 3.0]

    # Isaac/standalone-SDK axes: x forward, y left, z up.
    forward = vut_pose_to_core_world((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), frame="x_forward_z_up")
    left = vut_pose_to_core_world((0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0), frame="x_forward_z_up")
    up = vut_pose_to_core_world((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), frame="x_forward_z_up")
    # Core world: x right, y up, z toward the viewer (forward = -z).
    assert forward.translation.tolist() == pytest.approx([0.0, 0.0, -1.0])
    assert left.translation.tolist() == pytest.approx([-1.0, 0.0, 0.0])
    assert up.translation.tolist() == pytest.approx([0.0, 1.0, 0.0])
    # Round trip through the launcher's default Core-world -> tracking rotation.
    back = compose_transforms(
        RigidTransform((0, 0, 0), MANUS_CORE_WORLD_TO_TRACKING_QUATERNION_XYZW), forward
    )
    assert back.translation.tolist() == pytest.approx([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="frame must be one of"):
        vut_pose_to_core_world((0, 0, 0), (0, 0, 0, 1), frame="y_up_left_handed")


def test_client_decodes_add_update_and_remove_from_a_live_socket(tmp_path):
    socket_path = str(tmp_path / "vut.sock")
    server = FakeVutServer(socket_path, serial="SN-TEST", slot=2, rate_hz=200.0).start()
    events = []
    client = VutIpcClient(socket_path)
    client.on_add = lambda tracker_id, serial: events.append(("add", tracker_id, serial))
    client.on_update = lambda pose: events.append(("update", pose))
    client.on_remove = lambda tracker_id: events.append(("remove", tracker_id))
    try:
        assert _wait(lambda: client.poll() and any(kind == "update" for kind, *_ in events))
        assert ("add", 2, "SN-TEST") in events
        pose = next(event[1] for event in events if event[0] == "update")
        assert pose.slot == 2 and pose.tracker_id == "SN-TEST"
        assert pose.position.tolist() == pytest.approx([0.25, 1.1, -0.4])
        assert pose.quality == "trackable" and pose.battery == 87
        server.remove_tracker()
        assert _wait(lambda: client.poll() and ("remove", 2) in events)
    finally:
        client.disconnect()
        server.close()


def test_pose_feed_reconnects_and_tracks_the_latest_pose(tmp_path):
    socket_path = str(tmp_path / "vut.sock")
    seen = []
    feed = VutPoseFeed(socket_path, poll_hz=500.0, listener=seen.append, reconnect_interval=0.05)
    feed.start()
    try:
        time.sleep(0.2)
        assert not feed.connected and not seen  # daemon not running yet
        server = FakeVutServer(socket_path, rate_hz=200.0).start()
        try:
            assert _wait(lambda: feed.update_count >= 5)
            assert feed.connected
            latest = feed.latest("59-A33R03250")
            assert latest is not None and latest.state_name == "ready"
            assert select_tracker(feed.snapshot(), None) is latest
            assert select_tracker(feed.snapshot(), "other") is None
            server.position = (0.5, 1.0, 0.0)
            assert _wait(
                lambda: feed.latest("59-A33R03250") is not None
                and np.allclose(feed.latest("59-A33R03250").position, (0.5, 1.0, 0.0))
            )
            timestamps = [pose.timestamp_us for pose in seen[-5:]]
            assert timestamps == sorted(timestamps) and len(set(timestamps)) == len(timestamps)
        finally:
            server.close()
        assert _wait(lambda: not feed.connected)
    finally:
        feed.close()
