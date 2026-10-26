"""Contact-force finger haptics (Isaac Teleop HapticSink contract) and worker relay."""

from functools import partial
import multiprocessing
import time

import numpy as np
import pytest

from dex_teleop.arat.eval.hand_model import HandSemantics
from dex_teleop.omnigibson.haptics import (
    HAPTIC_FINGER_ORDER,
    ContactForceToFingerAmplitudes,
    ContactHapticFeedback,
    FingerContactForces,
    HapticAmplitudes,
    HapticConfig,
    HapticSink,
    ManusGloveHapticDevice,
)
from dex_teleop.runtime import MultiSourceTrackingWorker, ProcessTrackingWorker
from dex_teleop.types import Handedness
from test_multi_source_worker import _FakeRetargeter
from test_tracking_process import _QueueSource, _push, _wait_until

SEMANTICS = HandSemantics.sharpa("right")


def _contact(link: str, force):
    return (f"/World/robot/{link}", "/World/block/base_link", (0, 0, 0), (0, 0, 1), np.asarray(force), 0.0)


def test_finger_order_matches_the_metaglove_actuators():
    assert HAPTIC_FINGER_ORDER == ("thumb", "index", "middle", "ring", "pinky")


def test_contact_forces_sum_per_digit_with_pad_and_dorsal_weights():
    contacts = [
        _contact("right_thumb_elastomer", (3.0, 0.0, 0.0)),  # thumb pad: 3 N
        _contact("right_thumb_elastomer", (0.0, 4.0, 0.0)),  # thumb pad: +4 N
        _contact("right_index_PP", (0.0, 0.0, 2.0)),  # index dorsal: 2 N * 0.5
        _contact("right_hand_C_MC", (9.0, 0.0, 0.0)),  # palm: no actuator
        _contact("right_arm_link_4", (9.0, 0.0, 0.0)),  # not a hand link
    ]
    forces = FingerContactForces.from_contacts(contacts, SEMANTICS.classify_link)
    assert forces.as_mapping() == pytest.approx({"thumb": 7.0, "index": 1.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0})
    with pytest.raises(ValueError, match="non-negative"):
        FingerContactForces(np.array([-1.0, 0, 0, 0, 0]))


def test_retargeter_maps_force_to_amplitude_with_floor_and_decay():
    config = HapticConfig(minimum_force_n=1.0, maximum_force_n=5.0, minimum_amplitude=0.2, smoothing=0.5)
    retargeter = ContactForceToFingerAmplitudes(config)

    silent = retargeter(FingerContactForces.zero())
    assert not silent.active

    light = retargeter(FingerContactForces(np.array([1.5, 0, 0, 0, 0])))
    assert light.values[0] == pytest.approx(0.2 + 0.8 * 0.125)  # floor + proportional share
    assert light.values[1:].tolist() == [0.0] * 4

    strong = retargeter(FingerContactForces(np.array([50.0, 5.0, 0, 0, 0])))
    assert strong.values[0] == 1.0 and strong.values[1] == 1.0  # saturated, rising edge immediate

    released = retargeter(FingerContactForces.zero())
    assert released.values[0] == pytest.approx(0.5)  # exponential decay
    for _ in range(10):
        released = retargeter(FingerContactForces.zero())
    assert not released.active  # settles to exactly zero
    retargeter.reset()
    assert not retargeter(FingerContactForces(np.array([0.5, 0, 0, 0, 0]))).active  # below threshold


def test_haptic_config_rejects_nonsense():
    with pytest.raises(ValueError):
        HapticConfig(minimum_force_n=5.0, maximum_force_n=1.0)
    with pytest.raises(ValueError):
        HapticConfig(smoothing=0.0)
    with pytest.raises(ValueError):
        HapticConfig(update_hz=0.0)
    with pytest.raises(ValueError):
        HapticAmplitudes(np.array([float("nan")] * 5))


class _RecordingWorker:
    def __init__(self):
        self.calls = []
        self.haptic_sources = ("manus",)

    def apply_haptic_feedback(self, handedness, amplitudes):
        self.calls.append((handedness, list(amplitudes)))
        return True


def test_sink_rate_limits_resends_only_changes_and_keeps_a_heartbeat():
    worker = _RecordingWorker()
    device = ManusGloveHapticDevice(worker, endpoint="right")
    now = [100.0]
    sink = HapticSink(device, config=HapticConfig(update_hz=10.0, heartbeat_s=0.5, change_epsilon=0.02), clock=lambda: now[0])

    sink.write("right", HapticAmplitudes(np.array([0.5, 0, 0, 0, 0])))
    assert sink.flush() == ["right"]
    # Changed but inside the 100 ms rate window: deferred (stays pending).
    now[0] += 0.05
    sink.write("right", HapticAmplitudes(np.array([0.9, 0, 0, 0, 0])))
    assert sink.flush() == []
    # Unchanged within epsilon after the window: nothing new is sent.
    now[0] += 0.15
    sink.write("right", HapticAmplitudes(np.array([0.51, 0, 0, 0, 0])))
    assert sink.flush() == []
    assert len(worker.calls) == 1
    # Heartbeat keeps an active, unchanged command alive.
    now[0] += 0.5
    sink.write("right", HapticAmplitudes(np.array([0.5, 0, 0, 0, 0])))
    assert sink.flush() == ["right"]
    # Falling silent is always sent once the window allows it.
    now[0] += 0.2
    sink.write("right", HapticAmplitudes.silent())
    assert sink.flush() == ["right"]
    assert worker.calls[-1] == ("right", [0.0] * 5)
    with pytest.raises(KeyError):
        sink.write("left", HapticAmplitudes.silent())
    sink.silence()
    assert worker.calls[-1] == ("right", [0.0] * 5)


def test_controller_reads_robot_contacts_and_silences_on_reset():
    class _Robot:
        def __init__(self):
            self.contacts = [_contact("right_index_elastomer", (0.0, 0.0, 6.0))]

        def get_finger_contact_data(self, arm):
            assert arm == "right_arm"
            return self.contacts

    worker = _RecordingWorker()
    robot = _Robot()
    now = [0.0]
    controller = ContactHapticFeedback(
        robot,
        SEMANTICS.classify_link,
        ManusGloveHapticDevice(worker, endpoint="right"),
        arm="right_arm",
        endpoint="right",
        config=HapticConfig(update_hz=1000.0),
        clock=lambda: now[0],
    )
    amplitudes = controller.step()
    assert amplitudes.values[1] == 1.0 and worker.calls == [("right", amplitudes.as_list())]
    now[0] += 1.0
    robot.contacts = []
    controller.step()
    assert worker.calls[-1][1][1] < 1.0
    controller.reset()
    assert worker.calls[-1] == ("right", [0.0] * 5)
    assert controller.metadata()["finger_order"] == list(HAPTIC_FINGER_ORDER)
    with pytest.raises(TypeError, match="apply_haptic_feedback"):
        ManusGloveHapticDevice(object())


class _HapticSource(_QueueSource):
    supports_haptics = True

    def __init__(self, messages, sink):
        super().__init__(messages)
        self._sink = sink

    def vibrate_fingers(self, handedness, powers):
        self._sink.put((getattr(handedness, "value", handedness), list(powers)))
        return True


def _haptic_worker(messages, sink):
    return MultiSourceTrackingWorker(
        {"manus": _HapticSource(messages, sink)},
        articulation_source="manus",
        wrist_source="manus",
        retargeter=_FakeRetargeter(),
        handedness=Handedness.RIGHT,
        unread_command_lifetime=10.0,
    )


def test_in_process_worker_relays_amplitudes_to_haptic_sources():
    context = multiprocessing.get_context("spawn")
    messages, sink = context.Queue(), context.Queue()
    worker = _haptic_worker(messages, sink)
    assert worker.haptic_sources == ("manus",)
    assert worker.apply_haptic_feedback(Handedness.RIGHT, [1, 0, 0, 0, 0]) is True
    assert sink.get(timeout=2.0) == ("right", [1, 0, 0, 0, 0])


def test_process_worker_forwards_amplitudes_to_the_tracking_process():
    context = multiprocessing.get_context("spawn")
    messages, sink = context.Queue(), context.Queue()
    worker = ProcessTrackingWorker(partial(_haptic_worker, messages, sink))
    with worker:
        assert worker.haptic_sources == ("manus",)
        _push(messages, 1)
        _wait_until(lambda: worker.snapshot() is not None)
        assert worker.apply_haptic_feedback(Handedness.RIGHT, [0.25, 0.5, 0.75, 1.0, 0.0]) is True
        assert sink.get(timeout=5.0) == ("right", [0.25, 0.5, 0.75, 1.0, 0.0])
        time.sleep(0.05)
        worker.check_health()
