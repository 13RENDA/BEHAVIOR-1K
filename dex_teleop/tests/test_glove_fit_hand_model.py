"""The glove-fit hand model: 20 decoded joint angles -> the glove's own 21 landmarks."""

import json

import numpy as np
import pytest

from dex_teleop.emg.hand_model import (
    ANGLE_NAMES, FINGER_CHAIN, FINGERS, MANUS_TO_DECODER_INDEX, GloveFitHandModel, manus_ergonomics_to_decoder_rad)
from dex_teleop.types import MEDIAPIPE_JOINT_NAMES


def synthetic_glove(n=4000, seed=0):
    """A rigid toy hand: each bone turns in its finger's plane by the running sum of the flexion angles, the plane
    itself turns with the abduction angle. Angle layout per finger as the decoder's (thumb: FE, AA, FE, FE)."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-0.3, 1.2, (n, 20))
    lengths = np.array([[0.04, 0.05, 0.035, 0.03], [0.10, 0.045, 0.025, 0.022], [0.10, 0.05, 0.03, 0.024],
                        [0.095, 0.048, 0.03, 0.024], [0.093, 0.037, 0.02, 0.021]])
    base = np.array([[-0.5, -0.5, -0.7], [-0.2, 0.0, -1.0], [0.0, 0.0, -1.0], [0.2, 0.0, -1.0], [0.4, 0.0, -1.0]])
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    landmarks = np.zeros((n, 21, 3))
    for f in range(5):
        a = angles[:, 4 * f:4 * f + 4]
        abduction = a[:, 1] if f == 0 else a[:, 0]
        flexion = np.stack([a[:, 0], a[:, 2], a[:, 3]], axis=1) if f == 0 else a[:, 1:]
        position = np.repeat(lengths[f, 0] * base[f][None], n, axis=0)
        landmarks[:, 1 + 4 * f] = position
        bend = np.zeros(n)
        for b in range(3):
            bend = bend + flexion[:, b]
            direction = np.stack([base[f, 0] + 0.5 * np.sin(abduction), -np.sin(bend), -np.cos(bend)], axis=1)
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            position = position + lengths[f, b + 1] * direction
            landmarks[:, 2 + 4 * f + b] = position
    return angles, landmarks, lengths


def test_fit_reproduces_a_rigid_hand_and_keeps_its_bone_lengths():
    angles, landmarks, lengths = synthetic_glove()
    model = GloveFitHandModel.fit(angles[:3000], landmarks[:3000])
    predicted = model(angles[3000:])
    error_mm = np.linalg.norm(predicted - landmarks[3000:], axis=2) * 1000
    assert predicted.shape == (1000, 21, 3) and np.abs(predicted[:, 0]).max() == 0.0
    assert error_mm.mean() < 1.0 and error_mm.max() < 6.0
    for f, finger in enumerate(FINGERS):
        bones = np.linalg.norm(np.diff(predicted[:, list(FINGER_CHAIN[finger])], axis=1), axis=2)
        np.testing.assert_allclose(bones, np.repeat(lengths[f][None], 1000, axis=0), rtol=1e-6)


def test_poses_outside_the_fitted_range_are_clipped_not_extrapolated():
    angles, landmarks, _ = synthetic_glove()
    model = GloveFitHandModel.fit(angles, landmarks)
    wild = np.full((1, 20), 25.0)  # far outside
    edge = model.angle_high_rad[None]
    np.testing.assert_allclose(model(wild), model(edge))
    assert np.isfinite(model(-wild)).all()
    assert model(angles[0]).shape == (1, 21, 3)  # a single pose is accepted
    with pytest.raises(ValueError, match="20 joint angles"):
        model(np.zeros((3, 19)))


def test_model_round_trips_through_its_file(tmp_path):
    angles, landmarks, _ = synthetic_glove(n=1500)
    model = GloveFitHandModel.fit(angles, landmarks, metadata={"operator": "test", "frames": 1500})
    path = tmp_path / "fit.json"
    model.save(path)
    loaded = GloveFitHandModel.load(path)
    np.testing.assert_allclose(loaded(angles[:50]), model(angles[:50]))
    assert loaded.metadata == {"operator": "test", "frames": 1500}
    data = json.loads(path.read_text())
    assert data["angle_names"] == list(ANGLE_NAMES) and data["landmark_names"] == list(MEDIAPIPE_JOINT_NAMES)
    data["schema"] = "something/else"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not a glove-fit hand model"):
        GloveFitHandModel.load(path)


def test_glove_ergonomics_reorder_to_the_decoder_order():
    sdk = np.arange(20, dtype=float)  # SDK order: thumb spread (AA) first, then thumb CMC stretch (FE)
    decoder = np.degrees(manus_ergonomics_to_decoder_rad(sdk))
    assert decoder[:4].round(6).tolist() == [1.0, 0.0, 2.0, 3.0] and decoder[4:].round(6).tolist() == list(range(4, 20))
    assert ANGLE_NAMES[0] == "THUMB_CMC_FE" and ANGLE_NAMES[1] == "THUMB_CMC_AA" and len(MANUS_TO_DECODER_INDEX) == 20


def test_bridge_accepts_a_hand_model_file():
    from dex_teleop.emg.dexemg_bridge import _parser

    args = _parser().parse_args(["--checkpoint", "x.pt", "--hand-model", "fit.json"])
    assert args.hand_model.name == "fit.json"
    assert _parser().parse_args(["--checkpoint", "x.pt"]).hand_model is None
