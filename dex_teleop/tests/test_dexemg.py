"""The dexemg finger source: bridge protocol client, preflight, retargeting, launcher wiring."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import threading
import time

import numpy as np
import pytest

from dex_teleop.arat import AratTaskCatalog
from dex_teleop.emg.dexemg_protocol import COORDINATE_FRAME, LANDMARK_NAMES, PROTOCOL, decode, encode
from dex_teleop.omnigibson.launch_config import parse_launch_arguments
from dex_teleop.omnigibson.launcher import (
    _create_tracking_sources,
    _parser,
    _resolve_tracking_selection,
    _validate_manus_selection,
)
from dex_teleop.retargeting.retargeter import LandmarkRetargeter
from dex_teleop.tracking.base import SourceUnavailableError
from dex_teleop.tracking.dexemg import DexemgArticulationSource, probe_dexemg_bridge
from dex_teleop.tracking.fusion import HandObservationFuser
from dex_teleop.tracking.openxr import articulation_to_mediapipe21, observation_to_hand_frame
from dex_teleop.tracking.vive_wrist import ViveUltimateWristSource
from dex_teleop.types import Handedness, WristPoseSample

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "dexemg_landmarks.json").read_text())
TASK = "arat_grasp_block_10cm"


class FakeBridge:
    """Speaks the bridge side of the protocol on a free local port."""

    def __init__(self, *, hello=None, state="streaming", frames=True, rate_hz=50.0) -> None:
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(4)
        self.server.settimeout(0.05)
        self.port = self.server.getsockname()[1]
        self.hello = hello or {"type": "hello", "protocol": PROTOCOL, "landmark_names": list(LANDMARK_NAMES),
                               "coordinate_frame": COORDINATE_FRAME, "handedness": "right", "checkpoint": "fake.pt"}
        self.state = state
        self.frames = frames
        self.rate_hz = rate_hz
        self.landmarks = FIXTURE["closed"]["landmarks_m"]
        self.commands: list[dict] = []
        self._clients: list[socket.socket] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _status(self):
        return {"type": "status", "state": self.state, "message": "test", "rate_hz": 500.0,
                "scale_ratio_to_training": [1.0] * 8, "scale_warning": None, "channel_scale": [1.0] * 8}

    def _run(self) -> None:
        seq = 0
        next_frame = time.monotonic()
        while not self._stop.is_set():
            try:
                conn, _ = self.server.accept()
                conn.sendall(encode(self.hello) + encode(self._status()))
                conn.settimeout(0.01)
                self._clients.append(conn)
            except socket.timeout:
                pass
            for conn in list(self._clients):
                try:
                    data = conn.recv(4096)
                    for line in data.split(b"\n"):
                        if line.strip():
                            self.commands.append(decode(line))
                except (socket.timeout, BlockingIOError):
                    pass
                except OSError:
                    self._clients.remove(conn)
            if self.frames and time.monotonic() >= next_frame:
                next_frame += 1.0 / self.rate_hz
                seq += 1
                now = time.monotonic_ns()
                message = encode({"type": "frame", "seq": seq, "timestamp_ns": now - 2_000_000, "emit_ns": now,
                                  "sample_index": seq * 10, "angles": [0.0] * 20, "landmarks": self.landmarks})
                for conn in list(self._clients):
                    try:
                        conn.sendall(message)
                    except OSError:
                        self._clients.remove(conn)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        for conn in self._clients:
            conn.close()
        self.server.close()


@pytest.fixture
def bridge():
    fake = FakeBridge()
    yield fake
    fake.close()


def test_source_streams_wrist_local_mediapipe_samples(bridge):
    source = DexemgArticulationSource("127.0.0.1", bridge.port, startup_timeout=5.0)
    source.start()
    try:
        time.sleep(0.2)
        sample = source.read_articulation(Handedness.RIGHT)
        assert sample is not None and sample.schema == "mediapipe21"
        assert sample.coordinate_frame == COORDINATE_FRAME and sample.source == "dexemg"
        assert source.read_articulation(Handedness.LEFT) is None
        drained = source.drain_articulations(Handedness.RIGHT)
        assert len(drained) >= 3
        assert [s.source_frame_id for s in drained] == sorted(s.source_frame_id for s in drained)
        assert source.drain_articulations(Handedness.RIGHT) == () or len(source.drain_articulations(Handedness.RIGHT)) < 3
        converted = articulation_to_mediapipe21(sample)
        np.testing.assert_allclose(converted.positions(LANDMARK_NAMES), FIXTURE["closed"]["landmarks_m"])
        source.check_health()
        source.request_neutral_calibration(12.0)
        time.sleep(0.1)
        assert {"type": "cmd", "action": "calibrate_neutral", "seconds": 12.0} in bridge.commands
        assert source.recording_metadata()["checkpoint"] == "fake.pt"
    finally:
        source.close()


def test_source_fails_health_when_the_bridge_goes_quiet(bridge):
    source = DexemgArticulationSource("127.0.0.1", bridge.port, startup_timeout=5.0, stale_timeout=0.2)
    source.start()
    try:
        bridge.frames = False
        time.sleep(0.4)
        with pytest.raises(SourceUnavailableError, match="No dexemg hand"):
            source.check_health()
    finally:
        source.close()


def test_source_explains_a_missing_neutral():
    fake = FakeBridge(state="waiting_for_neutral", frames=False)
    try:
        source = DexemgArticulationSource("127.0.0.1", fake.port, startup_timeout=0.5)
        with pytest.raises(SourceUnavailableError, match="neutral"):
            source.start()
        source.close()
        with pytest.raises(SystemExit, match="neutral"):
            probe_dexemg_bridge("127.0.0.1", fake.port, timeout=1.0)
    finally:
        fake.close()


def test_source_rejects_an_incompatible_bridge():
    fake = FakeBridge(hello={"type": "hello", "protocol": "something/0", "landmark_names": [],
                             "coordinate_frame": "x", "handedness": "left"})
    try:
        source = DexemgArticulationSource("127.0.0.1", fake.port, startup_timeout=2.0)
        with pytest.raises(SourceUnavailableError, match="Incompatible dexemg bridge"):
            source.start()
        source.close()
    finally:
        fake.close()


def test_missing_bridge_names_the_fix():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    source = DexemgArticulationSource("127.0.0.1", port, startup_timeout=0.3)
    with pytest.raises(SourceUnavailableError, match="dexemg_bridge.py"):
        source.start()
    with pytest.raises(SystemExit, match="dexemg_bridge.py"):
        probe_dexemg_bridge("127.0.0.1", port, timeout=0.3)


def test_probe_reports_a_streaming_bridge(bridge):
    assert probe_dexemg_bridge("127.0.0.1", bridge.port).startswith("dexemg bridge OK")


def _retarget(landmarks_wrist_local):
    from dex_teleop.types import HandArticulationSample

    articulation = HandArticulationSample(
        timestamp=10.0, handedness=Handedness.RIGHT,
        joint_positions=dict(zip(LANDMARK_NAMES, np.asarray(landmarks_wrist_local))),
        source="dexemg", schema="mediapipe21", coordinate_frame=COORDINATE_FRAME,
    )
    wrist = WristPoseSample(timestamp=10.0, handedness=Handedness.RIGHT, position=np.array([0.4, 0.0, 1.0]),
                            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), source="vive_ultimate",
                            reference_frame="tracking", anatomical_frame=COORDINATE_FRAME)
    fuser = HandObservationFuser(maximum_skew_seconds=0.05)
    fuser.add_wrist(wrist)
    observation = fuser.fuse(articulation, wrist_source="vive_ultimate")
    assert observation is not None
    retargeter = LandmarkRetargeter.from_hand_model("sharpa", hand_side="right")
    return retargeter, retargeter.retarget(observation_to_hand_frame(observation), apply_filter=False)


def test_decoded_landmarks_retarget_to_a_closing_sharpa_hand():
    retargeter, open_command = _retarget(FIXTURE["open"]["landmarks_m"])
    _, closed_command = _retarget(FIXTURE["closed"]["landmarks_m"])
    names = list(retargeter.joint_names)
    flexion = [i for i, name in enumerate(names) if any(f in name.lower() for f in ("index", "middle", "ring", "pinky"))
               and not any(s in name.lower() for s in ("abd", "spread", "_aa", "mcp_ab"))]
    assert flexion, names
    delta = np.asarray(closed_command.joint_positions)[flexion] - np.asarray(open_command.joint_positions)[flexion]
    assert delta.mean() > 0.3, dict(zip([names[i] for i in flexion], delta.round(2)))


def _parse(*argv):
    return parse_launch_arguments(_parser(AratTaskCatalog()), list(argv), default_config=None)


def test_dexemg_preset_selects_emg_fingers_and_a_glove_free_wrist():
    args, config = _parse("--config", "dexemg", "--task", TASK)
    selection = _resolve_tracking_selection(args)
    assert (selection.hand_source, selection.wrist_source) == ("dexemg", "vive_ultimate")
    assert args.vive_tracker_id == "FA5AG3B00121" and args.tracking_yaw_deg == 180.0
    _validate_manus_selection(args, selection)
    sources = _create_tracking_sources(args, selection)
    assert set(sources) == {"dexemg", "vive_ultimate"}
    assert isinstance(sources["dexemg"], DexemgArticulationSource)
    assert isinstance(sources["vive_ultimate"], ViveUltimateWristSource)


def test_glove_as_ground_truth_combination_is_valid():
    args, _ = _parse("--config", "manus_core", "--task", TASK, "--hand-source", "dexemg", "--record-hand-source", "manus")
    selection = _resolve_tracking_selection(args)
    assert selection.articulation_sources == ("dexemg", "manus") and selection.wrist_source == "manus"
    _validate_manus_selection(args, selection)


@pytest.mark.parametrize(
    "argv, message",
    [
        (("--hand-source", "quest", "--wrist-source", "quest", "--vive-tracker-id", "X"), "--vive-tracker-"),
        (("--hand-source", "quest", "--wrist-source", "quest", "--dexemg-port", "9000"), "--dexemg-"),
        (("--hand-source", "dexemg", "--wrist-source", "vive_ultimate"), "--vive-tracker-id"),
        (("--hand-source", "dexemg", "--wrist-source", "quest", "--emg"), "owns it"),
        (("--hand-source", "dexemg", "--wrist-source", "quest", "--hand", "left"), "right hand only"),
    ],
)
def test_invalid_dexemg_and_vive_combinations_fail_before_kit(argv, message):
    args, _ = _parse("--task", TASK, *argv)
    with pytest.raises(SystemExit, match=message):
        _validate_manus_selection(args, _resolve_tracking_selection(args))


@pytest.mark.parametrize("argv", [("--wrist-source", "dexemg"), ("--hand-source", "vive_ultimate")])
def test_role_mismatched_sources_are_parser_errors(argv):
    with pytest.raises(SystemExit):
        _parse("--task", TASK, *argv)
