import numpy as np
import pytest

from scipy.spatial.transform import Rotation

from dex_teleop.hands import HAND_PROFILES, get_hand_profile
from dex_teleop.retargeting import (
    LandmarkRetargeter,
    RetargeterUnavailableError,
    create_hand_retargeter,
)
from dex_teleop.retargeting.dexpilot import DexPilotRetargeter
from dex_teleop.retargeting.optimizer import LPFilter
from dex_teleop.tracking.openxr import observation_to_hand_frame
from dex_teleop.types import (
    FusedHandObservation,
    HandArticulationSample,
    Handedness,
    MEDIAPIPE_JOINT_NAMES,
    OPENXR_ANATOMICAL_WRIST_FRAME,
    WristPoseSample,
)


LANDMARKS = np.array(
    [
        [0, 0, 0],
        [-0.03, 0.01, 0],
        [-0.05, 0.03, 0],
        [-0.065, 0.055, 0],
        [-0.075, 0.08, 0],
        [-0.02, 0.04, 0],
        [-0.02, 0.075, 0],
        [-0.02, 0.105, 0],
        [-0.02, 0.13, 0],
        [0, 0.045, 0],
        [0, 0.085, 0],
        [0, 0.12, 0],
        [0, 0.15, 0],
        [0.02, 0.04, 0],
        [0.025, 0.078, 0],
        [0.03, 0.11, 0],
        [0.035, 0.137, 0],
        [0.04, 0.03, 0],
        [0.05, 0.06, 0],
        [0.058, 0.085, 0],
        [0.065, 0.108, 0],
    ],
    dtype=np.float64,
)


@pytest.mark.parametrize("hand_model", tuple(HAND_PROFILES))
def test_each_hand_model_retargets_landmarks(hand_model):
    profile = HAND_PROFILES[hand_model]
    retargeter = LandmarkRetargeter.from_hand_model(hand_model, "right")

    positions = retargeter.retarget_landmarks(LANDMARKS, apply_filter=False)

    assert positions.shape == (profile.degrees_of_freedom,)
    assert np.isfinite(positions).all()


def _filtered_step(rate_hz: float, duration_s: float, reference_hz: float = 60.0) -> float:
    lowpass = LPFilter(0.2, reference_period=1.0 / reference_hz)
    lowpass.next(np.zeros(1), timestamp=0.0)
    output = None
    for index in range(1, round(duration_s * rate_hz) + 1):
        output = lowpass.next(np.ones(1), timestamp=index / rate_hz)
    return float(output[0])


def test_timestamped_lowpass_response_does_not_depend_on_the_retarget_rate():
    at_30_hz = _filtered_step(30.0, 0.2)
    at_120_hz = _filtered_step(120.0, 0.2)

    assert at_30_hz == pytest.approx(at_120_hz, abs=1e-12)
    assert at_120_hz == pytest.approx(1.0 - 0.8**12, abs=1e-12)


def test_timestamped_lowpass_at_its_reference_rate_matches_the_per_call_filter():
    timed = LPFilter(0.2, reference_period=1.0 / 60.0)
    per_call = LPFilter(0.2)
    for index, value in enumerate(np.linspace(0.0, 1.0, 20)):
        sample = np.array([value, -value])
        np.testing.assert_allclose(timed.next(sample, timestamp=index / 60.0), per_call.next(sample))


def test_timestamped_lowpass_holds_its_output_for_a_repeated_capture_time():
    lowpass = LPFilter(0.2, reference_period=1.0 / 60.0)
    lowpass.next(np.zeros(1), timestamp=1.0)
    first = lowpass.next(np.ones(1), timestamp=1.1)

    np.testing.assert_array_equal(lowpass.next(np.full(1, 5.0), timestamp=1.1), first)
    lowpass.reset()
    np.testing.assert_array_equal(lowpass.next(np.full(1, 5.0), timestamp=0.5), [5.0])


def test_lowpass_without_timestamps_applies_alpha_per_call():
    lowpass = LPFilter(0.2, reference_period=1.0 / 60.0)
    lowpass.next(np.zeros(1))

    np.testing.assert_allclose(lowpass.next(np.ones(1)), [0.2])
    np.testing.assert_allclose(lowpass.next(np.ones(1)), [0.36])


def test_retargeter_smooths_by_capture_time():
    filtered = LandmarkRetargeter.from_hand_model("sharpa", "right")
    raw = LandmarkRetargeter.from_hand_model("sharpa", "right")
    curled = LANDMARKS * np.array([1.0, 0.6, 1.0])
    opened = filtered.retarget_landmarks(LANDMARKS, timestamp=0.0)
    np.testing.assert_array_equal(opened, raw.retarget_landmarks(LANDMARKS, apply_filter=False))
    target = raw.retarget_landmarks(curled, apply_filter=False)

    half_period = filtered.retarget_landmarks(curled, timestamp=0.5 / 60.0)
    np.testing.assert_allclose(half_period, opened + (1.0 - 0.8**0.5) * (target - opened), atol=1e-9)


@pytest.mark.parametrize("side", ("left", "right"))
@pytest.mark.parametrize("hand_model", tuple(HAND_PROFILES))
def test_packaged_urdfs_cover_both_hands(hand_model, side):
    retargeter = LandmarkRetargeter.from_hand_model(hand_model, side)

    assert len(retargeter.joint_names) == HAND_PROFILES[hand_model].degrees_of_freedom


def test_retargeter_factory_preserves_adaptive_backend():
    retargeter = create_hand_retargeter("adaptive", "sharpa", "right")

    assert isinstance(retargeter, LandmarkRetargeter)


def test_dexpilot_reports_optional_dependency_when_not_installed(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def reject_dex_retargeting(name, *args, **kwargs):
        if name.startswith("dex_retargeting"):
            raise ImportError("not installed for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_dex_retargeting)

    with pytest.raises(RetargeterUnavailableError, match="dex-retargeting==0.4.6"):
        DexPilotRetargeter.from_hand_model("sharpa", "right")


def test_dexpilot_rich_observation_matches_legacy_world_frame_conversion():
    class _Optimizer:
        target_link_human_indices = np.array([[0, 0, 0, 0, 0], [4, 8, 12, 16, 20]])

    class _Sequence:
        optimizer = _Optimizer()

        def __init__(self):
            self.references = []

        def retarget(self, reference):
            self.references.append(reference.copy())
            return np.zeros(22)

    retargeter = object.__new__(DexPilotRetargeter)
    retargeter.hand_model = "sharpa"
    retargeter.hand_side = "right"
    retargeter.hand_profile = get_hand_profile("sharpa")
    retargeter.handtracking_to_baselink = np.array(
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    )
    retargeter.joint_names = tuple(f"joint_{index}" for index in range(22))
    retargeter._retargeting = _Sequence()
    articulation = HandArticulationSample(
        timestamp=1.0,
        handedness=Handedness.RIGHT,
        joint_positions=dict(zip(MEDIAPIPE_JOINT_NAMES, LANDMARKS, strict=True)),
        source="manus",
        schema="mediapipe21",
        coordinate_frame=OPENXR_ANATOMICAL_WRIST_FRAME,
    )
    wrist = WristPoseSample(
        timestamp=1.0,
        handedness=Handedness.RIGHT,
        position=np.array([0.4, -0.2, 0.8]),
        quaternion_xyzw=Rotation.from_euler(
            "xyz", [20.0, -35.0, 70.0], degrees=True
        ).as_quat(),
        source="quest",
        anatomical_frame=OPENXR_ANATOMICAL_WRIST_FRAME,
    )
    observation = FusedHandObservation(articulation, wrist)

    retargeter.retarget(observation_to_hand_frame(observation))
    legacy_reference = retargeter._retargeting.references[-1]
    retargeter.retarget_observation(observation)
    rich_reference = retargeter._retargeting.references[-1]

    np.testing.assert_allclose(rich_reference, legacy_reference, atol=1e-12)


def test_dexpilot_rich_observation_rejects_an_unconfigured_local_basis():
    retargeter = object.__new__(DexPilotRetargeter)
    articulation = HandArticulationSample(
        timestamp=1.0,
        handedness=Handedness.RIGHT,
        joint_positions=dict(zip(MEDIAPIPE_JOINT_NAMES, LANDMARKS, strict=True)),
        source="vendor",
        schema="mediapipe21",
        coordinate_frame="vendor_wrist",
    )
    wrist = WristPoseSample(
        timestamp=1.0,
        handedness=Handedness.RIGHT,
        position=np.zeros(3),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        source="vendor",
        anatomical_frame="vendor_wrist",
    )

    with pytest.raises(ValueError, match="ArticulationFrameTransform"):
        retargeter.retarget_observation(FusedHandObservation(articulation, wrist))
