from functools import partial
import multiprocessing
import os
import pickle
import queue
import threading
import time
from types import MappingProxyType

import numpy as np
import pytest

from dex_teleop.omnigibson.hand_tracking_recording import HandTrackingRecordingSession
from dex_teleop.runtime import MultiSourceTrackingWorker, ProcessTrackingWorker
from dex_teleop.tracking import HandObservationFuser, SourceUnavailableError, observation_to_hand_frame
from dex_teleop.tracking.multimodal import HandTrackingSampleBatch
from dex_teleop.types import Handedness
from test_multi_source_worker import _FakeRetargeter, _articulation, _wrist


class _QueueSource:
    """Same-callback articulation/wrist source fed by the test through a queue."""

    def __init__(self, messages, start_error=None):
        self._messages = messages
        self._start_error = start_error
        self._pending = []
        self._health_error = None
        # check_health runs on the status loop while the worker thread drains.
        self._lock = threading.Lock()

    def start(self):
        if self._start_error is not None:
            raise RuntimeError(self._start_error)

    def _pump(self):
        while True:
            try:
                kind, value, timestamp = self._messages.get_nowait()
            except queue.Empty:
                return
            if kind == "pair":
                self._pending.append(
                    (
                        _articulation(timestamp, source="hts", frame_id=value),
                        _wrist(timestamp, source="hts", frame_id=value, x=float(value)),
                    )
                )
            else:
                self._health_error = value

    def drain_hand_tracking(self, handedness):
        with self._lock:
            self._pump()
            pairs, self._pending = self._pending, []
        return HandTrackingSampleBatch(
            articulations=tuple(pair[0] for pair in pairs),
            wrists=tuple(pair[1] for pair in pairs),
        )

    def read_articulation(self, handedness):
        return None

    def read_wrist(self, handedness):
        return None

    def check_health(self):
        with self._lock:
            self._pump()
            health_error = self._health_error
        if health_error is not None:
            raise RuntimeError(health_error)

    def close(self):
        pass


def _queue_worker(messages, once_per_snapshot=False, start_error=None):
    return MultiSourceTrackingWorker(
        {"quest": _QueueSource(messages, start_error)},
        articulation_source="quest",
        wrist_source="quest",
        retargeter=_FakeRetargeter(),
        handedness=Handedness.RIGHT,
        retarget_once_per_snapshot=once_per_snapshot,
        unread_command_lifetime=10.0,
    )


def _tracking_process_id(worker):
    return os.getpid(), type(worker.sources["quest"]).__name__


def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Condition was not satisfied before timeout")


@pytest.fixture
def messages():
    return multiprocessing.get_context("spawn").Queue()


def _push(messages, frame_id):
    messages.put(("pair", frame_id, time.monotonic()))


def test_frozen_samples_round_trip_through_pickle_unchanged():
    timestamp = time.monotonic()
    articulation = _articulation(timestamp, source="hts", frame_id=3)
    wrist = _wrist(timestamp, source="hts", frame_id=3, x=0.5)
    frame = observation_to_hand_frame(
        HandObservationFuser(interpolate_wrist=False).fuse_paired(articulation, wrist)
    )

    def round_trip(value):
        return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))

    restored = round_trip(articulation)
    assert isinstance(restored.joint_positions, MappingProxyType)
    assert not restored.joint_positions["wrist"].flags.writeable
    np.testing.assert_array_equal(restored.positions(), articulation.positions())
    np.testing.assert_array_equal(restored.validity(), articulation.validity())
    assert (restored.timestamp, restored.source_frame_id, dict(restored.provenance)) == (
        articulation.timestamp,
        articulation.source_frame_id,
        dict(articulation.provenance),
    )
    with pytest.raises(AttributeError):
        restored.source = "changed"

    restored_wrist = round_trip(wrist)
    assert isinstance(restored_wrist.provenance, MappingProxyType)
    assert not restored_wrist.position.flags.writeable
    np.testing.assert_array_equal(restored_wrist.quaternion_xyzw, wrist.quaternion_xyzw)

    restored_frame = round_trip(frame)
    assert isinstance(restored_frame.joints, MappingProxyType)
    np.testing.assert_array_equal(restored_frame.mediapipe_landmarks(), frame.mediapipe_landmarks())
    assert restored_frame.receipt_timestamp == frame.receipt_timestamp


def test_tracking_process_forwards_recorded_rows_before_the_snapshots_that_use_them(messages):
    worker = ProcessTrackingWorker(partial(_queue_worker, messages))
    session = HandTrackingRecordingSession()
    worker.start()
    try:
        assert worker.retargeting_stream == "fakeretargeter.quest+quest" or worker.retargeting_stream.endswith(
            ".quest+quest"
        )
        worker.set_recording_session(session)
        session.begin_episode()
        for frame_id in range(1, 6):
            _push(messages, frame_id)
        _wait_until(lambda: (snapshot := worker.snapshot()) is not None and snapshot.frame.source_frame_id == 5)

        snapshot = worker.snapshot()
        assert session.append_action_selection(snapshot.action_selection()) == 0
        recorded = session.writer_kwargs()
        assert [s.source_frame_id for s in recorded["articulation_streams"]["articulation.quest"]] == [1, 2, 3, 4, 5]
        assert [s.source_frame_id for s in recorded["wrist_streams"]["wrist.quest"]] == [1, 2, 3, 4, 5]
        retargeted = recorded["retargeting_streams"][worker.retargeting_stream]
        assert retargeted[-1].articulation_row == 4
        assert snapshot.command.joint_positions.tolist() == [5.0]
        worker.check_stream_freshness(5.0)

        worker.set_recording_session(None)
        _push(messages, 6)
        _wait_until(lambda: (latest := worker.snapshot()) is not None and latest.frame.source_frame_id == 6)
        assert len(session.writer_kwargs()["articulation_streams"]["articulation.quest"]) == 5
        assert worker.snapshot().action_selection() is None
    finally:
        worker.close()
    assert not worker._process.is_alive()


def test_tracking_process_reset_discards_snapshots_from_before_the_boundary(messages):
    worker = ProcessTrackingWorker(partial(_queue_worker, messages))
    worker.start()
    try:
        _push(messages, 1)
        assert worker.wait_for_first(10.0).frame.source_frame_id == 1

        worker.reset()
        assert worker.snapshot() is None
        with pytest.raises(SourceUnavailableError, match=r"within 0\.2s.*missing initial samples"):
            worker.wait_for_first(0.2)
        _push(messages, 2)
        assert worker.wait_for_first(10.0).frame.source_frame_id == 2
    finally:
        worker.close()


def test_tracking_process_retargets_once_per_snapshot_read(messages):
    worker = ProcessTrackingWorker(partial(_queue_worker, messages, once_per_snapshot=True))
    session = HandTrackingRecordingSession()
    worker.start()
    try:
        worker.set_recording_session(session)

        def recorded(kind):
            return session.writer_kwargs()[kind]

        _push(messages, 1)
        _wait_until(lambda: worker.retargeting_stream in recorded("retargeting_streams"))
        for frame_id in range(2, 6):
            _push(messages, frame_id)
        _wait_until(lambda: len(recorded("articulation_streams")["articulation.quest"]) == 5)
        time.sleep(0.1)
        assert [s.articulation_row for s in recorded("retargeting_streams")[worker.retargeting_stream]] == [0]

        assert worker.snapshot().frame.source_frame_id == 1
        _wait_until(lambda: worker.snapshot().frame.source_frame_id == 5)
        assert [s.articulation_row for s in recorded("retargeting_streams")[worker.retargeting_stream]] == [0, 4]
    finally:
        worker.close()


def test_tracking_process_runs_queries_next_to_the_sources(messages):
    worker = ProcessTrackingWorker(partial(_queue_worker, messages))
    worker.start()
    try:
        process_id, source_type = worker.query(_tracking_process_id)
        assert process_id == worker._process.pid != os.getpid()
        assert source_type == "_QueueSource"
    finally:
        worker.close()


def test_tracking_process_reports_source_health_and_start_failures(messages):
    failing = ProcessTrackingWorker(partial(_queue_worker, messages, start_error="glove missing"))
    with pytest.raises(SourceUnavailableError, match="Failed to start selected articulation source 'quest'.*glove missing"):
        failing.start()
    assert not failing._process.is_alive()

    worker = ProcessTrackingWorker(partial(_queue_worker, messages))
    worker.start()
    try:
        worker.check_health()
        messages.put(("health", "tracker untrackable", 0.0))
        _wait_until(lambda: worker._status.get("health") is not None)
        with pytest.raises(SourceUnavailableError, match="Unhealthy selected articulation source 'quest'.*tracker untrackable"):
            worker.check_health()
    finally:
        worker.close()


def test_tracking_process_death_is_reported_as_a_worker_failure(messages):
    worker = ProcessTrackingWorker(partial(_queue_worker, messages))
    worker.start()
    try:
        worker._process.kill()
        _wait_until(lambda: worker._exited)
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            worker.check_health()
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            worker.reset()
    finally:
        worker.close()
