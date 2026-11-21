#!/usr/bin/env python3
"""Fit the glove hand model (20 joint angles -> the glove's 21 landmarks) from teleoperation recordings.

Every glove frame of a recording made since 2026-11-16 holds the glove skeleton and the glove's ergonomics angles
of the same instant. The model (dex_teleop.emg.hand_model.GloveFitHandModel) is fitted on all given recordings;
before that, each recording is held out once to report how well the others predict its landmarks.

    python dex_teleop/scripts/fit_glove_hand_model.py ~/emg_teleop_data/*.hdf5 --output ~/claude_work/emg_teleop/hand_model/glove_fit.json

Then: dexemg_gui.py --checkpoint ... --hand-model <that file>.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dex_teleop.emg.hand_model import FINGER_CHAIN, FINGERS, GloveFitHandModel, manus_ergonomics_to_decoder_rad  # noqa: E402
from dex_teleop.tracking.openxr import MEDIAPIPE_TO_OPENXR_JOINTS  # noqa: E402
from dex_teleop.types import MEDIAPIPE_JOINT_NAMES  # noqa: E402

GLOVE_STREAM = "hand_tracking/articulation_streams/articulation.manus"


def load_recording(path: Path, stride: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(angles_rad (n, 20), landmarks_m (n, 21, 3)) of the glove frames that carry a new ergonomics reading."""

    with h5py.File(path, "r") as recording:
        if GLOVE_STREAM not in recording:
            return None
        group = recording[GLOVE_STREAM]
        names = json.loads(group.attrs["joint_names"])
        columns = [names.index(MEDIAPIPE_TO_OPENXR_JOINTS[name]) for name in MEDIAPIPE_JOINT_NAMES]
        valid = np.asarray(group["joint_validity"]).all(axis=1)
        provenance = group["provenance_json"][:]
        positions = np.asarray(group["joint_positions"])
    angles, rows, last_arrival = [], [], None
    for row in range(0, len(provenance), stride):
        if not valid[row]:
            continue
        entry = json.loads(provenance[row].decode() if isinstance(provenance[row], bytes) else provenance[row])
        if "ergonomics_deg" not in entry or entry.get("ergonomics_arrival_monotonic_ns") == last_arrival:
            continue
        # the skeleton and the ergonomics of one glove frame arrive within a few ms of each other
        if abs(entry["callback_capture_monotonic_ns"] - entry["ergonomics_arrival_monotonic_ns"]) > 20_000_000:
            continue
        last_arrival = entry["ergonomics_arrival_monotonic_ns"]
        angles.append(json.loads(entry["ergonomics_deg"]))
        rows.append(row)
    if not rows:
        return None
    return manus_ergonomics_to_decoder_rad(np.asarray(angles)), positions[rows][:, columns]


def errors_mm(model: GloveFitHandModel, angles: np.ndarray, landmarks: np.ndarray) -> dict[str, float]:
    distance = np.linalg.norm(model(angles) - (landmarks - landmarks[:, :1]), axis=2) * 1000.0
    return {finger: float(distance[:, list(FINGER_CHAIN[finger][1:])].mean()) for finger in FINGERS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3, help="use every Nth glove frame (default 3)")
    parser.add_argument("--operator", default="", help="whose glove calibration this is (stored in the file)")
    args = parser.parse_args()

    data = {}
    for path in args.recordings:
        loaded = load_recording(path.expanduser(), args.stride)
        if loaded is None:
            print(f"skipped {path.name}: no glove frames with ergonomics")
            continue
        data[path.name] = loaded
    if len(data) < 2:
        raise SystemExit("need at least two recordings with glove ergonomics")
    held_out = {}
    for name in data:
        rest = [value for key, value in data.items() if key != name]
        model = GloveFitHandModel.fit(np.concatenate([a for a, _ in rest]), np.concatenate([l for _, l in rest]))
        held_out[name] = errors_mm(model, *data[name])
    worst = max(held_out, key=lambda key: max(held_out[key].values()))
    mean = {finger: float(np.mean([e[finger] for e in held_out.values()])) for finger in FINGERS}
    angles = np.concatenate([a for a, _ in data.values()])
    landmarks = np.concatenate([l for _, l in data.values()])
    metadata = {
        "fitted": datetime.datetime.now().isoformat(timespec="seconds"),
        "operator": args.operator,
        "recordings": sorted(data),
        "frames": int(len(angles)),
        "stride": args.stride,
        "held_out_landmark_error_mm_mean": mean,
        "held_out_worst_recording": {"name": worst, "error_mm": held_out[worst]},
        "angle_range_deg": {"low": np.degrees(angles.min(axis=0)).round(1).tolist(),
                            "high": np.degrees(angles.max(axis=0)).round(1).tolist()},
    }
    model = GloveFitHandModel.fit(angles, landmarks, metadata=metadata)
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output.expanduser())
    print(f"{len(data)} recordings, {len(angles)} frames")
    print("held-out landmark error, mm (each recording predicted from the others), mean: "
          + ", ".join(f"{finger} {value:.2f}" for finger, value in mean.items()))
    print(f"worst recording {worst}: " + ", ".join(f"{finger} {value:.2f}" for finger, value in held_out[worst].items()))
    print(f"in-sample error of the saved model, mm: " + ", ".join(f"{f} {v:.2f}" for f, v in errors_mm(model, angles, landmarks).items()))
    print(f"saved {args.output.expanduser()}")


if __name__ == "__main__":
    main()
