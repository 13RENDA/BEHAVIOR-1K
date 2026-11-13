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
    BATTERY_UNKNOWN,
    DEV_LOADING,
    DEV_LOST_TRACKING,
    DEV_NOT_CONNECTED,
    DEV_OFF,
    DEV_PREPARING,
    DEV_READY,
    DEV_ROTATION_ONLY,
    DEV_SETUP_REQUIRED,
    DEV_SYNCING,
    FLAG_CHARGING,
    FLAG_POSE_STABLE,
    FLAG_ROTATION_ONLY,
    FLAG_SETUP_REQUIRED,
    HEADER_SIZE,
    MSG_UPDATE,
    SDK_HEADER_SIZE,
    SDK_MSG_EVT_POSE,
    SDK_POSE_SIZE,
    SDK_STATUS_SIZE,
    STATE_BUILD_MAP,
    STATE_LOADING,
    STATE_LOST_TRACKING,
    STATE_NOT_CONNECTED,
    STATE_OFF,
    STATE_PREPARING,
    STATE_READY,
    STATE_SYNCING,
    UPDATE_SIZE,
    VUT_PROTOCOL_SEQPACKET,
    VUT_PROTOCOL_STREAM,
    VutIpcClient,
    VutPoseFeed,
    decode_device_list,
    decode_device_state,
    decode_pose_event,
    decode_status_event,
    decode_update,
    encode_device_list,
    encode_header,
    encode_pose_event,
    encode_sdk_header,
    encode_status_event,
    encode_update_payload,
    select_tracker,
    vut_pose_to_core_world,
)

from fake_vut_server import FakeVutSdkServer, FakeVutServer


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
        assert client.protocol == VUT_PROTOCOL_STREAM  # auto-detected: 1.0.1 slot protocol
        assert ("add", 2, "SN-TEST") in events
        pose = next(event[1] for event in events if event[0] == "update")
        assert pose.slot == 2 and pose.tracker_id == "SN-TEST"
        assert pose.position.tolist() == pytest.approx([0.25, 1.1, -0.4])
        assert pose.quality == "trackable" and pose.battery == 87 and pose.battery_known
        assert pose.protocol == VUT_PROTOCOL_STREAM
        server.remove_tracker()
        assert _wait(lambda: client.poll() and ("remove", 2) in events)
    finally:
        client.disconnect()
        server.close()


# ---------------------------------------------------------------------------
# VIVE Hub 1.0.3+: the VUT SDK protocol on a SEQPACKET socket


def test_sdk_wire_format_matches_vut_types_h():
    header = encode_sdk_header(SDK_MSG_EVT_POSE, SDK_POSE_SIZE, request_id=7)
    assert len(header) == SDK_HEADER_SIZE == 12
    assert struct.unpack("<BBHII", header) == (SDK_MSG_EVT_POSE, 1, 0, 7, SDK_POSE_SIZE)

    pose_payload = encode_pose_event(
        3, (0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0), timestamp_ns=123_456_789, serial="FA5AG3B00121"
    )
    assert len(pose_payload) == SDK_POSE_SIZE == 100
    pose = decode_pose_event(pose_payload, raw_state=DEV_READY, receipt_monotonic_ns=9)
    assert pose.slot == 2 and pose.tracker_id == "FA5AG3B00121"  # device_id = slot + 1
    assert pose.position.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert pose.timestamp_us == 123_456 and pose.receipt_monotonic_ns == 9
    assert pose.state == STATE_READY and pose.quality == "trackable"
    assert pose.battery == BATTERY_UNKNOWN and not pose.battery_known
    assert pose.protocol == VUT_PROTOCOL_SEQPACKET
    assert pose.as_mapping()["protocol"] == VUT_PROTOCOL_SEQPACKET
    anonymous = decode_pose_event(
        encode_pose_event(1, (0, 0, 0), (0, 0, 0, 1), timestamp_ns=1), raw_state=DEV_READY, receipt_monotonic_ns=1
    )
    assert anonymous.tracker_id == "vive_tracker_0"
    with pytest.raises(ValueError, match="100 bytes"):
        decode_pose_event(pose_payload[:40], raw_state=DEV_READY, receipt_monotonic_ns=1)
    with pytest.raises(ValueError, match="broadcast"):
        decode_pose_event(
            encode_pose_event(0, (0, 0, 0), (0, 0, 0, 1), timestamp_ns=1), raw_state=DEV_READY, receipt_monotonic_ns=1
        )

    status_payload = encode_status_event(3, DEV_SYNCING, timestamp_ns=42, error_code=5)
    assert len(status_payload) == SDK_STATUS_SIZE == 24
    assert decode_status_event(status_payload) == (3, DEV_SYNCING, 5, 42)

    devices = decode_device_list(encode_device_list([(3, 0x12, "FA5AG3B00121"), (1, DEV_OFF, "VIVE Ultimate Tracker 1")]))
    assert devices == [(3, 0x12, "FA5AG3B00121"), (1, DEV_OFF, "VIVE Ultimate Tracker 1")]
    with pytest.raises(ValueError, match="truncated"):
        decode_device_list(struct.pack("<I", 2))


@pytest.mark.parametrize(
    ("raw_state", "state", "flags", "quality", "name"),
    (
        (DEV_READY, STATE_READY, 0, "trackable", "ready"),
        # tracker_server 1.0.3 reports the raw status byte: Running_good + pose-stable bit.
        (0x12, STATE_READY, FLAG_POSE_STABLE, "trackable", "ready"),
        (DEV_ROTATION_ONLY, STATE_READY, FLAG_ROTATION_ONLY, "bad", "ready"),
        (DEV_ROTATION_ONLY | FLAG_CHARGING, STATE_READY, FLAG_ROTATION_ONLY | FLAG_CHARGING, "bad", "ready"),
        (DEV_READY | FLAG_ROTATION_ONLY, STATE_READY, FLAG_ROTATION_ONLY, "bad", "ready"),
        (DEV_SETUP_REQUIRED, STATE_BUILD_MAP, FLAG_SETUP_REQUIRED, "untrackable", "build_map"),
        (DEV_LOST_TRACKING, STATE_LOST_TRACKING, 0, "untrackable", "lost_tracking"),
        (DEV_SYNCING, STATE_SYNCING, 0, "untrackable", "syncing"),
        (DEV_LOADING, STATE_LOADING, 0, "untrackable", "loading"),
        (DEV_PREPARING, STATE_PREPARING, 0, "untrackable", "preparing"),
        (DEV_NOT_CONNECTED, STATE_NOT_CONNECTED, 0, "untrackable", "not_connected"),
        (DEV_OFF, STATE_OFF, 0, "untrackable", "off"),
    ),
)
def test_sdk_device_state_maps_onto_the_shared_tracker_states(raw_state, state, flags, quality, name):
    assert decode_device_state(raw_state) == (state, flags)
    pose = decode_pose_event(
        encode_pose_event(1, (0, 0, 0), (0, 0, 0, 1), timestamp_ns=1, serial="X"),
        raw_state=raw_state,
        receipt_monotonic_ns=1,
    )
    assert (pose.state, pose.flags, pose.quality, pose.state_name) == (state, flags, quality, name)


def test_client_auto_detects_the_seqpacket_daemon_and_follows_status_events(tmp_path):
    socket_path = str(tmp_path / "vut.sock")
    server = FakeVutSdkServer(socket_path, serial="FA5AG3B00121", slot=2, rate_hz=200.0).start()
    events = []
    client = VutIpcClient(socket_path)
    client.on_server_start = lambda: events.append(("start",))
    client.on_add = lambda tracker_id, serial: events.append(("add", tracker_id, serial))
    client.on_update = lambda pose: events.append(("update", pose))
    client.on_status = lambda tracker_id, state, error: events.append(("status", tracker_id, state, error))
    client.on_remove = lambda tracker_id: events.append(("remove", tracker_id))
    try:
        assert _wait(lambda: client.poll() and any(kind == "update" for kind, *_ in events))
        assert client.protocol == VUT_PROTOCOL_SEQPACKET
        assert client.session_id == 1 and client.server_version == 1 and server.hello_count == 1
        assert events[0] == ("start",)
        assert ("add", 2, "FA5AG3B00121") in events
        pose = next(event[1] for event in events if event[0] == "update")
        assert pose.slot == 2 and pose.tracker_id == "FA5AG3B00121"
        assert pose.position.tolist() == pytest.approx([0.25, 1.1, -0.4])
        # Default fake state 0x12: Running_good plus the pose-stable bit, as on hardware.
        assert pose.state == STATE_READY and pose.pose_stable and pose.quality == "trackable"
        assert pose.battery == BATTERY_UNKNOWN
        assert 0 <= pose.receipt_monotonic_ns - pose.timestamp_us * 1000 < 2_000_000_000

        server.state = DEV_ROTATION_ONLY
        assert _wait(
            lambda: client.poll()
            and any(event[0] == "update" and event[1].quality == "bad" for event in events)
        )
        assert ("status", 2, STATE_READY, 0) in events  # rotation-only is READY plus a flag
        # Setup required stops the pose stream; the status alone must reach listeners.
        server.state = DEV_SETUP_REQUIRED
        assert _wait(
            lambda: client.poll()
            and any(event[0] == "update" and event[1].state_name == "build_map" for event in events)
        )
        degraded = next(event[1] for event in reversed(events) if event[0] == "update")
        assert degraded.quality == "untrackable" and degraded.setup_required
        assert degraded.position.tolist() == pytest.approx([0.25, 1.1, -0.4])
        assert ("status", 2, STATE_BUILD_MAP, 0) in events
        server.remove_tracker()
        assert _wait(lambda: client.poll() and ("remove", 2) in events)
        assert client.send_command("Echo") is False  # no command channel on this daemon
        assert client.request_device_list() is True
    finally:
        client.disconnect()
        server.close()


def test_sdk_client_learns_a_late_serial_from_the_pose_stream(tmp_path):
    socket_path = str(tmp_path / "vut.sock")
    server = FakeVutSdkServer(socket_path, serial="FA5AG3B00121", slot=0, rate_hz=200.0, placeholder_name=True).start()
    adds = []
    client = VutIpcClient(socket_path)
    client.on_add = lambda tracker_id, serial: adds.append((tracker_id, serial))
    try:
        assert _wait(lambda: client.poll() and 0 in client.serials)
        assert adds == [(0, None)]
        assert client.serials[0] == "vive_tracker_0"
        server.placeholder_name = False
        assert _wait(lambda: client.poll() and client.serials.get(0) == "FA5AG3B00121")
        assert adds == [(0, None), (0, "FA5AG3B00121")]
    finally:
        client.disconnect()
        server.close()


def test_explicit_protocol_must_match_the_daemon(tmp_path):
    socket_path = str(tmp_path / "vut.sock")
    server = FakeVutSdkServer(socket_path, rate_hz=200.0).start()
    try:
        stream_only = VutIpcClient(socket_path, protocol=VUT_PROTOCOL_STREAM)
        assert stream_only.connect() is False and stream_only.protocol is None
        seqpacket = VutIpcClient(socket_path, protocol=VUT_PROTOCOL_SEQPACKET)
        assert seqpacket.connect() is True and seqpacket.protocol == VUT_PROTOCOL_SEQPACKET
        seqpacket.disconnect()
    finally:
        server.close()
    with pytest.raises(ValueError, match="protocol must be one of"):
        VutIpcClient(socket_path, protocol="datagram")


def test_pose_feed_tracks_the_seqpacket_daemon(tmp_path):
    socket_path = str(tmp_path / "vut.sock")
    seen = []
    feed = VutPoseFeed(socket_path, poll_hz=500.0, listener=seen.append, reconnect_interval=0.05)
    feed.start()
    server = FakeVutSdkServer(socket_path, rate_hz=200.0, extra_trackers={4: "OTHER-TRACKER"}).start()
    try:
        assert _wait(lambda: feed.update_count >= 10)
        assert feed.connected and feed.protocol == VUT_PROTOCOL_SEQPACKET
        assert feed.serials() == {2: "FA5AG3B00121", 4: "OTHER-TRACKER"}
        latest = feed.latest("FA5AG3B00121")
        assert latest is not None and latest.state_name == "ready" and latest.quality == "trackable"
        assert select_tracker(feed.snapshot(), "FA5AG3B00121") is latest
        assert select_tracker(feed.snapshot(), None) is None  # two trackers: ambiguous
        server.position = (0.5, 1.0, 0.0)
        assert _wait(
            lambda: feed.latest("FA5AG3B00121") is not None
            and np.allclose(feed.latest("FA5AG3B00121").position, (0.5, 1.0, 0.0))
        )
        ours = [pose for pose in seen if pose.tracker_id == "FA5AG3B00121"][-5:]
        timestamps = [pose.timestamp_us for pose in ours]
        assert timestamps == sorted(timestamps) and len(set(timestamps)) == len(timestamps)
        server.state = DEV_SYNCING
        assert _wait(lambda: feed.latest("FA5AG3B00121").quality == "untrackable")
        assert feed.latest("FA5AG3B00121").state_name == "syncing"
    finally:
        server.close()
    assert _wait(lambda: not feed.connected)
    feed.close()


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
