"""The pure-Python MANUS sidecar on the (fake) MANUS Core 3.2 Python SDK."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from dex_teleop.tracking import SourceUnavailableError
from dex_teleop.tracking.manus import (
    MANUS_PROTOCOL_VERSION,
    ManusIntegratedSource,
    discover_manus_python_sdk,
)
from dex_teleop.tracking.manus_calibration import ManusCoreCalibration
from dex_teleop.tracking.manus_python_bridge import (
    DEFAULT_TRACKER_TO_CASING_TRANSLATION,
    PROTOCOL_VERSION,
    compose_tracker_wrist,
    parse_options,
)
from dex_teleop.tracking.transforms import RigidTransform, compose_transforms, inverse_transform
from dex_teleop.tracking.vive_ultimate import DEV_SYNCING, STATE_SYNCING, VUT_TRACKING_SYSTEM
from dex_teleop.types import Handedness

from fake_vut_server import FakeVutSdkServer, FakeVutServer

TESTS = Path(__file__).resolve().parent
FAKE_SDK = TESTS / "fake_manus_sdk"
SERIAL = "59-A33R03250"
GLOVE_ID = 1417281806
GLOVE_USER_ID = 7


def _start_vut_server(daemon: str, socket_path: str, **kwargs):
    """A fake VIVE Hub daemon: ``stream`` (1.0.1 slot protocol) or ``seqpacket`` (1.0.3 VUT SDK)."""

    if daemon == "stream":
        return FakeVutServer(socket_path, rate_hz=120.0, **kwargs).start()
    return FakeVutSdkServer(socket_path, rate_hz=120.0, **kwargs).start()


def _lose_tracking(daemon: str, server) -> None:
    server.state = STATE_SYNCING if daemon == "stream" else DEV_SYNCING


VUT_DAEMONS = ("stream", "seqpacket")


def _wait(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _read_log(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def fake_sdk(monkeypatch, tmp_path):
    log = tmp_path / "fake_manus_sdk.jsonl"
    monkeypatch.setenv("DEX_TELEOP_MANUS_PYTHON_SDK_PATH", str(FAKE_SDK))
    monkeypatch.setenv("DEX_TELEOP_FAKE_MANUS_LOG", str(log))
    monkeypatch.setenv("DEX_TELEOP_FAKE_MANUS_FRAME_HZ", "200")
    monkeypatch.delenv("DEX_TELEOP_FAKE_MANUS_SCENARIO", raising=False)
    return log


def _source(**kwargs):
    kwargs.setdefault("backend", "python")
    kwargs.setdefault("mode", "integrated")
    kwargs.setdefault("startup_timeout", 15.0)
    kwargs.setdefault("required_handedness", Handedness.RIGHT)
    return ManusIntegratedSource(**kwargs)


# ---------------------------------------------------------------------------
# Unit level


def test_protocol_version_probe_runs_without_the_sdk(monkeypatch):
    monkeypatch.setenv("DEX_TELEOP_MANUS_PYTHON_SDK_PATH", "/nonexistent/manus/python")
    env = {**os.environ, "PYTHONPATH": str(TESTS.parent / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "dex_teleop.tracking.manus_python_bridge", "--protocol-version"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"dex_teleop.manus {PROTOCOL_VERSION}"
    assert PROTOCOL_VERSION == MANUS_PROTOCOL_VERSION == 3


def test_options_default_the_mount_offset_and_tracker_diagnostics():
    options = parse_options(["--mode", "integrated", "--tracker-feed", "vive_ultimate", "--required-hand", "right"])
    assert options.tracker_diagnostics is True
    assert options.wrist_composition == "bridge"
    assert options.tracker_to_casing.translation.tolist() == pytest.approx(
        list(DEFAULT_TRACKER_TO_CASING_TRANSLATION["right"])
    )
    assert options.tracker_to_wrist.translation.tolist() == pytest.approx(
        list(DEFAULT_TRACKER_TO_CASING_TRANSLATION["right"])
    )
    explicit = parse_options(
        [
            "--tracker-feed", "vive_ultimate",
            "--tracker-to-wrist-translation", "0.01", "0.02", "0.03",
            "--tracker-to-wrist-rotation-xyzw", "0", "0", "0.70710678", "0.70710678",
        ]
    )
    assert explicit.tracker_to_wrist.translation.tolist() == pytest.approx([0.01, 0.02, 0.03])
    assert explicit.tracker_to_wrist.quaternion_xyzw.tolist() == pytest.approx([0, 0, math.sqrt(0.5), math.sqrt(0.5)])
    with pytest.raises(SystemExit):
        parse_options(["--mode", "remote", "--tracker-feed", "vive_ultimate"])
    with pytest.raises(SystemExit):
        parse_options(["--mode", "integrated", "--core-host", "1.2.3.4"])


def test_compose_tracker_wrist_moves_the_hand_rigidly():
    root = (0, np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]))
    finger = (5, np.array([1.0, 2.1, 3.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    tracker = RigidTransform((0.5, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    offset = RigidTransform((0.0, -0.05, 0.0), (0.0, 0.0, 0.0, 1.0))

    composed = compose_tracker_wrist([root, finger], 0, tracker, offset)

    new_root = RigidTransform(composed[0][1], composed[0][2])
    assert new_root.translation.tolist() == pytest.approx([0.5, -0.05, 0.0])
    assert new_root.quaternion_xyzw.tolist() == pytest.approx([0, 0, 0, 1])
    # The finger keeps its wrist-local pose exactly.
    before = compose_transforms(inverse_transform(RigidTransform(root[1], root[2])), RigidTransform(finger[1], finger[2]))
    after = compose_transforms(inverse_transform(new_root), RigidTransform(composed[1][1], composed[1][2]))
    assert after.translation.tolist() == pytest.approx(before.translation.tolist(), abs=1e-9)
    assert abs(float(np.dot(after.quaternion_xyzw, before.quaternion_xyzw))) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="wrist node"):
        compose_tracker_wrist([finger], 0, tracker, offset)


def test_python_sdk_discovery_finds_the_newest_release(tmp_path, monkeypatch):
    import dex_teleop.tracking.manus as manus_module

    # Only the explicit root counts here; the machine's ~/manus_setup must not leak in.
    monkeypatch.setattr(
        manus_module,
        "_candidate_manus_roots",
        lambda explicit_root: (Path(explicit_root).expanduser().resolve(),),
    )
    vendor = tmp_path / "vendor"
    for release in ("ManusSDK_v3.2.0", "ManusSDK_v3.2.1"):
        package = vendor / release / "Python" / "manus_sdk"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
    assert discover_manus_python_sdk(tmp_path) == (vendor / "ManusSDK_v3.2.1" / "Python").resolve()
    assert discover_manus_python_sdk(vendor / "ManusSDK_v3.2.0") == (vendor / "ManusSDK_v3.2.0" / "Python").resolve()
    with pytest.raises(SourceUnavailableError, match="MANUS Core 3.2 Python SDK"):
        discover_manus_python_sdk(tmp_path / "elsewhere")


def test_source_rejects_feeds_and_haptics_on_the_native_backend():
    with pytest.raises(ValueError, match="python MANUS backend"):
        ManusIntegratedSource(backend="native", tracker_feed="vive_ultimate")
    with pytest.raises(ValueError, match="python MANUS backend"):
        ManusIntegratedSource(backend="native", haptics=True)
    with pytest.raises(ValueError, match="Integrated mode"):
        ManusIntegratedSource(backend="python", mode="remote", tracker_feed="vive_ultimate")
    with pytest.raises(ValueError, match="bridge_executable applies"):
        ManusIntegratedSource(backend="python", bridge_executable="/tmp/x")


# ---------------------------------------------------------------------------
# End to end through the sidecar process


def test_integrated_python_bridge_streams_the_glove(fake_sdk):
    source = _source()
    with source:
        # start() returns only after a validated right-hand frame arrived.
        articulation = source.read_articulation(Handedness.RIGHT)
        wrist = source.read_wrist(Handedness.RIGHT)
        assert _wait(lambda: bool(source.drain_hand_tracking(Handedness.RIGHT).articulations))
        metadata = source.recording_metadata()

    assert articulation is not None and wrist is not None
    assert articulation.source == "manus_integrated"
    assert len(articulation.joint_names) == 25
    assert articulation.provenance["bridge_protocol_version"] == 3
    assert articulation.provenance["supported_protocol_version"] == MANUS_PROTOCOL_VERSION
    assert "haptic_commands" in articulation.provenance["bridge_capabilities"]
    # Same raw node layout as the C++ fake bridge: thumb tip sits at the known local offset.
    assert articulation.joint_positions["index_tip"].tolist() == pytest.approx([0.009, 0.018, -0.009], abs=1e-6)
    assert wrist.position.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert metadata["backend"] == "python"
    assert metadata["sdk_version"] == "3.2.1" and metadata["core_version"] == "3.2.1"
    assert metadata["linked_mode"] == "python"
    assert metadata["tracker_feed"] == "none"
    assert metadata["required_glove_user_id"] == GLOVE_USER_ID
    calls = [entry["call"] for entry in _read_log(fake_sdk)]
    assert calls[:2] == ["CoreSdk_InitializeIntegrated", "CoreSdk_InitializeCoordinateSystemWithVUH"]
    assert "CoreSdk_ShutDown" in calls
    coordinates = next(entry for entry in _read_log(fake_sdk) if entry["call"] == "CoreSdk_InitializeCoordinateSystemWithVUH")
    assert (coordinates["view"], coordinates["up"], coordinates["handedness"]) == (6, 5, 2)  # ZToViewer, +Y, right-handed


def test_integrated_python_bridge_reports_a_missing_license(fake_sdk, monkeypatch):
    monkeypatch.setenv("DEX_TELEOP_FAKE_MANUS_SCENARIO", "no_license")
    source = _source(startup_timeout=10.0, glove_timeout=2)
    with pytest.raises(SourceUnavailableError, match="Integrated SDK"):
        source.start()
    source.close()


@pytest.mark.parametrize("daemon", VUT_DAEMONS)
def test_vive_ultimate_feed_drives_a_tracker_gated_wrist(fake_sdk, tmp_path, daemon):
    socket_path = str(tmp_path / "vut.sock")
    server = _start_vut_server(daemon, socket_path, serial=SERIAL)
    calibration = ManusCoreCalibration.default()
    source = _source(
        tracker_feed="vive_ultimate",
        vive_ultimate_socket=socket_path,
        require_tracker_for_wrist=True,
        tracker_stale_timeout=1.0,
        expected_tracker_id=SERIAL,
        expected_tracker_user_id=None,
        core_calibration=calibration,
    )
    try:
        with source:
            assert source.hand_motion == "tracker"
            batch = None

            def _pairs():
                nonlocal batch
                batch = source.drain_hand_tracking(Handedness.RIGHT)
                return bool(batch.wrists)

            assert _wait(_pairs)
            wrist = batch.wrists[-1]
            diagnostics = source.tracker_diagnostics()
            metadata = source.recording_metadata()
    finally:
        server.close()

    # Bridge-composed wrist: tracker pose (identity rotation) + the default
    # right-hand mount offset, then the launcher's Core-world -> tracking rotation.
    offset = np.asarray(DEFAULT_TRACKER_TO_CASING_TRANSLATION["right"])
    expected = calibration.apply(
        Handedness.RIGHT,
        RigidTransform(np.asarray(server.position) + offset, (0.0, 0.0, 0.0, 1.0)),
    )
    assert wrist.position.tolist() == pytest.approx(expected.translation.tolist(), abs=1e-6)
    assert abs(float(np.dot(wrist.quaternion_xyzw, expected.quaternion_xyzw))) == pytest.approx(1.0, abs=1e-9)
    assert wrist.confidence == 1.0
    assert wrist.provenance["tracker_health"] == "healthy"
    assert wrist.provenance["validated_tracker_id"] == SERIAL
    assert wrist.provenance["validated_tracker_system"] == VUT_TRACKING_SYSTEM
    assert wrist.provenance["hand_motion"] == "tracker"
    assert diagnostics is not None
    selected = next(tracker for tracker in diagnostics.trackers if tracker["id"] == SERIAL)
    assert selected["type"] == "right_hand" and selected["user_id"] == GLOVE_USER_ID
    assert selected["tracking_system"] == VUT_TRACKING_SYSTEM and selected["quality"] == "trackable"
    assert metadata["tracker_feed"] == "vive_ultimate"
    assert metadata["wrist_composition"] == "bridge"
    assert metadata["expected_tracker_system"] == VUT_TRACKING_SYSTEM
    assert metadata["tracker_healthy"] is True

    log = _read_log(fake_sdk)
    sends = [entry for entry in log if entry["call"] == "CoreSdk_SendDataForTrackers"]
    assert sends and all(entry["result"] == "Success" for entry in sends)
    injected = sends[-1]
    assert injected["id"] == SERIAL
    assert injected["tracker_type"] == 4  # TrackerType.RightHand
    assert injected["tracked_point"] == 1  # TrackedPoint.Casing
    assert injected["quality"] == 2  # TrackingQuality.Trackable
    assert injected["position"] == pytest.approx(list(server.position))
    assert any(entry["call"] == "CoreSdk_AssignTrackerToUser" and entry["user_id"] == GLOVE_USER_ID for entry in log)
    assert any(entry["call"] == "CoreSdk_AssignRoleToTracker" and entry["role"] == 4 for entry in log)
    offsets = [entry for entry in log if entry["call"] == "CoreSdk_SetTrackerOffset"]
    assert offsets and offsets[0]["user_id"] == GLOVE_USER_ID and offsets[0]["entry_type"] == 15
    assert offsets[0]["translation"] == pytest.approx(list(DEFAULT_TRACKER_TO_CASING_TRANSLATION["right"]))
    motions = [entry["motion"] for entry in log if entry["call"] == "CoreSdk_SetRawSkeletonHandMotion"]
    assert motions == [2]  # HandMotion.Tracker


@pytest.mark.parametrize("daemon", VUT_DAEMONS)
def test_vive_ultimate_tracking_loss_fails_wrist_control(fake_sdk, tmp_path, daemon):
    socket_path = str(tmp_path / "vut.sock")
    server = _start_vut_server(daemon, socket_path, serial=SERIAL)
    source = _source(
        tracker_feed="vive_ultimate",
        vive_ultimate_socket=socket_path,
        require_tracker_for_wrist=True,
        tracker_stale_timeout=1.0,
        expected_tracker_id=SERIAL,
        core_calibration=ManusCoreCalibration.default(),
    )
    try:
        with source:
            assert _wait(lambda: bool(source.drain_hand_tracking(Handedness.RIGHT).wrists))
            _lose_tracking(daemon, server)

            def _unhealthy():
                try:
                    source.check_health()
                except SourceUnavailableError as error:
                    return "untrackable" in str(error)
                return False

            assert _wait(_unhealthy)
            assert source.recording_metadata()["tracker_healthy"] is False
    finally:
        server.close()


def test_wrong_tracker_serial_never_qualifies(fake_sdk, tmp_path):
    socket_path = str(tmp_path / "vut.sock")
    server = FakeVutServer(socket_path, serial="SOME-OTHER-TRACKER", rate_hz=120.0).start()
    source = _source(
        startup_timeout=2.5,
        tracker_feed="vive_ultimate",
        vive_ultimate_socket=socket_path,
        require_tracker_for_wrist=True,
        expected_tracker_id=SERIAL,
        core_calibration=ManusCoreCalibration.default(),
    )
    try:
        with pytest.raises(SourceUnavailableError, match="absent"):
            source.start()
    finally:
        source.close()
        server.close()


def test_haptic_commands_reach_the_glove_and_stop_on_close(fake_sdk):
    source = _source(haptics=True)
    with source:
        assert source.supports_haptics
        assert _wait(lambda: source.read_articulation(Handedness.RIGHT) is not None)
        assert source.vibrate_fingers(Handedness.RIGHT, [1.0, 0.0, 0.5, 0.0, 0.0])
        assert _wait(
            lambda: any(
                entry["call"] == "CoreSdk_VibrateFingersForGlove" and entry["powers"] == [1.0, 0.0, 0.5, 0.0, 0.0]
                for entry in _read_log(fake_sdk)
            )
        )
        assert source.recording_metadata()["haptic_command_count"] == 1
        with pytest.raises(ValueError, match="five finite"):
            source.vibrate_fingers(Handedness.RIGHT, [1.0, 0.0])
    vibrations = [entry for entry in _read_log(fake_sdk) if entry["call"] == "CoreSdk_VibrateFingersForGlove"]
    assert vibrations[0]["glove_id"] == GLOVE_ID
    assert vibrations[-1]["powers"] == [0.0] * 5  # silenced before shutdown


def test_haptics_are_off_by_default(fake_sdk):
    source = _source()
    with source:
        assert not source.supports_haptics
        assert source.vibrate_fingers(Handedness.RIGHT, [1.0] * 5) is False
