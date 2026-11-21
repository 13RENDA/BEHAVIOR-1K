"""A hand model fitted to one operator's MANUS glove: 20 joint angles -> the glove's own 21 landmarks.

The EMG decoder outputs the glove's 20 ergonomics angles (the labels it was trained on). The bridge used to turn
them into landmarks with emg2pose's generic hand, whose thumb does not match the glove's skeleton: for the SAME
angles the thumb landmarks sit 19-91 mm away and the thumb metacarpal points 48 deg elsewhere (2026-11-18,
9,766 frames), so the Sharpa retargeter, which glove teleoperation feeds with the glove skeleton, got a different
hand and drove the thumb, the DIP joints and the pinky to their limits.

This model reproduces the glove skeleton instead. The skeleton is rigid (constant bone lengths, fixed palm), so
each of the 20 bones keeps its measured length and only its direction is a function of its finger's four angles
(a linear regression on polynomial and kinematic-chain terms, renormalized to unit length). Inputs are clipped to the angle range seen
in the fit, so a pose outside it degrades to the nearest fitted pose instead of extrapolating a polynomial.

Fit it with ``scripts/fit_glove_hand_model.py`` from teleoperation recordings (which hold the glove skeleton and its
ergonomics on the same frames); pass the resulting JSON to the bridge with ``--hand-model``. numpy only.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from dex_teleop.types import MEDIAPIPE_JOINT_NAMES

SCHEMA = "dex_teleop.glove_fit_hand_model/1"
# The decoder's angle order (emg2pose joint order; dexemg finetune_data.ergonomics.EMG2POSE_JOINT_NAMES).
ANGLE_NAMES = (
    "THUMB_CMC_FE", "THUMB_CMC_AA", "THUMB_MCP_FE", "THUMB_IP_FE",
    "INDEX_MCP_AA", "INDEX_MCP_FE", "INDEX_PIP_FE", "INDEX_DIP_FE",
    "MIDDLE_MCP_AA", "MIDDLE_MCP_FE", "MIDDLE_PIP_FE", "MIDDLE_DIP_FE",
    "RING_MCP_AA", "RING_MCP_FE", "RING_PIP_FE", "RING_DIP_FE",
    "PINKY_MCP_AA", "PINKY_MCP_FE", "PINKY_PIP_FE", "PINKY_DIP_FE",
)
# decoder_angles = manus_ergonomics[MANUS_TO_DECODER_INDEX]: the glove's 20 ergonomics values of one hand in SDK
# order (manus_python_bridge.ERGONOMICS_ORDER) differ from the decoder order only in the first two thumb entries
# (dexemg finetune_data.ergonomics.MANUS_TO_EMG2POSE_INDEX, checked 2026-11-18).
MANUS_TO_DECODER_INDEX = (1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# Per finger: its four angle indices and its landmark chain from the wrist (MediaPipe-21 indices).
FINGER_ANGLES = {finger: tuple(range(4 * k, 4 * k + 4)) for k, finger in enumerate(FINGERS)}
FINGER_CHAIN = {finger: (0, *range(1 + 4 * k, 5 + 4 * k)) for k, finger in enumerate(FINGERS)}
CLIP_MARGIN_RAD = float(np.radians(5.0))


def manus_ergonomics_to_decoder_rad(ergonomics_deg: np.ndarray) -> np.ndarray:
    """(..., 20) glove ergonomics in SDK order, degrees -> the decoder's order, radians."""

    return np.radians(np.asarray(ergonomics_deg, dtype=np.float64)[..., list(MANUS_TO_DECODER_INDEX)])


FEATURE_COUNT = 41


def _features(angles: np.ndarray, thumb: bool) -> np.ndarray:
    """(n, 4) one finger's angles (rad) -> (n, 41).

    1, a, a_i a_j, sin a, cos a (23), plus the terms a planar finger chain is made of: sin and cos of the running
    sum of the three flexion angles (6), alone and times sin / cos of the abduction angle (12). On the 13 glove
    recordings of 2026-11-16/17 the second group lowered the held-out landmark error (middle 1.47 -> 0.73 mm,
    ring 2.09 -> 1.17 mm). Angle layout: thumb FE, AA, FE, FE; other fingers AA, FE, FE, FE.
    """

    columns = [np.ones(len(angles))] + [angles[:, i] for i in range(4)]
    columns += [angles[:, i] * angles[:, j] for i, j in itertools.combinations_with_replacement(range(4), 2)]
    columns += [np.sin(angles[:, i]) for i in range(4)] + [np.cos(angles[:, i]) for i in range(4)]
    abduction = angles[:, 1] if thumb else angles[:, 0]
    flexion = np.stack([angles[:, 0], angles[:, 2], angles[:, 3]], axis=1) if thumb else angles[:, 1:]
    running = np.cumsum(flexion, axis=1)
    chain = np.concatenate([np.sin(running), np.cos(running)], axis=1)
    return np.concatenate(
        [np.stack(columns, axis=1), chain, chain * np.sin(abduction)[:, None], chain * np.cos(abduction)[:, None]], axis=1
    )


class GloveFitHandModel:
    """``model(angles_rad (n, 20)) -> landmarks (n, 21, 3)`` in meters, MediaPipe-21 order, wrist at the origin, in
    the frame of the landmarks it was fitted on (the glove articulation's OpenXR anatomical wrist frame)."""

    def __init__(self, bone_lengths_m, weights, angle_low_rad, angle_high_rad, metadata=None) -> None:
        self.bone_lengths_m = np.asarray(bone_lengths_m, dtype=np.float64)        # (5, 4)
        self.weights = np.asarray(weights, dtype=np.float64)                      # (5, 4, FEATURE_COUNT, 3)
        self.angle_low_rad = np.asarray(angle_low_rad, dtype=np.float64)          # (20,)
        self.angle_high_rad = np.asarray(angle_high_rad, dtype=np.float64)
        self.metadata = dict(metadata or {})
        if self.bone_lengths_m.shape != (5, 4) or self.weights.shape != (5, 4, FEATURE_COUNT, 3):
            raise ValueError("glove-fit hand model has the wrong shape")
        if self.angle_low_rad.shape != (20,) or self.angle_high_rad.shape != (20,):
            raise ValueError("glove-fit hand model needs 20 angle bounds")

    @classmethod
    def fit(cls, angles_rad: np.ndarray, landmarks_m: np.ndarray, *, ridge: float = 1e-6, metadata=None) -> "GloveFitHandModel":
        angles = np.asarray(angles_rad, dtype=np.float64)
        landmarks = np.asarray(landmarks_m, dtype=np.float64)
        if angles.ndim != 2 or angles.shape[1] != 20 or landmarks.shape != (len(angles), 21, 3):
            raise ValueError("fit needs angles (n, 20) and landmarks (n, 21, 3)")
        landmarks = landmarks - landmarks[:, :1]
        lengths = np.zeros((5, 4))
        weights = np.zeros((5, 4, FEATURE_COUNT, 3))
        for f, finger in enumerate(FINGERS):
            bones = np.diff(landmarks[:, list(FINGER_CHAIN[finger])], axis=1)      # (n, 4, 3)
            norms = np.linalg.norm(bones, axis=2)
            lengths[f] = norms.mean(axis=0)
            x = _features(angles[:, list(FINGER_ANGLES[finger])], finger == "thumb")
            gram = x.T @ x + ridge * len(x) * np.eye(x.shape[1])
            for b in range(4):
                weights[f, b] = np.linalg.solve(gram, x.T @ (bones[:, b] / norms[:, b:b + 1]))
        return cls(lengths, weights, angles.min(axis=0) - CLIP_MARGIN_RAD, angles.max(axis=0) + CLIP_MARGIN_RAD, metadata)

    def __call__(self, angles_rad: np.ndarray) -> np.ndarray:
        angles = np.atleast_2d(np.asarray(angles_rad, dtype=np.float64))
        if angles.shape[1] != 20:
            raise ValueError(f"expected 20 joint angles, got {angles.shape[1]}")
        angles = np.clip(angles, self.angle_low_rad, self.angle_high_rad)
        landmarks = np.zeros((len(angles), 21, 3))
        for f, finger in enumerate(FINGERS):
            x = _features(angles[:, list(FINGER_ANGLES[finger])], finger == "thumb")
            chain = FINGER_CHAIN[finger]
            position = np.zeros((len(angles), 3))
            for b in range(4):
                direction = x @ self.weights[f, b]
                direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
                position = position + self.bone_lengths_m[f, b] * direction
                landmarks[:, chain[b + 1]] = position
        return landmarks

    def to_json(self) -> dict:
        return {
            "schema": SCHEMA,
            "angle_names": list(ANGLE_NAMES),
            "angle_units": "rad",
            "landmark_names": list(MEDIAPIPE_JOINT_NAMES),
            "landmark_units": "m",
            "fingers": list(FINGERS),
            "bone_lengths_m": self.bone_lengths_m.tolist(),
            "direction_weights": self.weights.tolist(),
            "angle_low_rad": self.angle_low_rad.tolist(),
            "angle_high_rad": self.angle_high_rad.tolist(),
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json()))

    @classmethod
    def load(cls, path: str | Path) -> "GloveFitHandModel":
        data = json.loads(Path(path).read_text())
        if data.get("schema") != SCHEMA:
            raise ValueError(f"{path}: not a glove-fit hand model ({data.get('schema')!r}, expected {SCHEMA!r})")
        if tuple(data["angle_names"]) != ANGLE_NAMES or tuple(data["landmark_names"]) != tuple(MEDIAPIPE_JOINT_NAMES):
            raise ValueError(f"{path}: angle or landmark order differs from this version")
        return cls(data["bone_lengths_m"], data["direction_weights"], data["angle_low_rad"], data["angle_high_rad"],
                   data.get("metadata"))


__all__ = ["ANGLE_NAMES", "GloveFitHandModel", "MANUS_TO_DECODER_INDEX", "SCHEMA", "manus_ergonomics_to_decoder_rad"]
