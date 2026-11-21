"""EMG-decoded hand against the MANUS glove, as Sharpa joint commands (simulator-free statistics).

Both hands are retargeted with the same retargeter, so the comparison is in the space the robot is
commanded in: "what would the Sharpa fingers have been told to do by the glove, and what were they told by
the EMG decoder, at the same moment". It therefore contains the decoder's error and the difference between
the decoder's generic hand model and the operator's glove skeleton; it is not the human joint-angle error
of the offline studies.

Every number is reported next to a constant baseline scored on the same samples: the glove command's own
mean over the session (a constant pose that already knows this session's average).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# Below this constant-baseline error the glove hand has not really moved yet and a ratio means nothing.
MINIMUM_CONSTANT_DEG = 1.0


def finger_of(joint_name: str) -> str:
    """``right_index_MCP_FE`` -> ``index``."""

    return joint_name.split("_")[1]


def is_flexion(joint_name: str) -> bool:
    return not joint_name.endswith("_AA")


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator > 1e-12 else float("nan")


def estimate_lag_s(t_a: np.ndarray, a: np.ndarray, t_b: np.ndarray, b: np.ndarray, *, rate_hz: float = 100.0,
                   max_lag_s: float = 0.5) -> tuple[float, float] | None:
    """Lag of series ``a`` behind ``b`` in seconds (positive = a is late) and the peak correlation.

    Both series are resampled to ``rate_hz`` on their common time span and cross-correlated within
    ``max_lag_s``.
    """

    start, stop = max(t_a[0], t_b[0]), min(t_a[-1], t_b[-1])
    if stop - start < 4.0 * max_lag_s:
        return None
    grid = np.arange(start, stop, 1.0 / rate_hz)
    x = np.interp(grid, t_a, a)
    y = np.interp(grid, t_b, b)
    x = x - x.mean()
    y = y - y.mean()
    if x.std() < 1e-9 or y.std() < 1e-9:
        return None
    max_shift = int(round(max_lag_s * rate_hz))
    best = (0, -2.0)
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            u, v = x[shift:], y[: len(y) - shift]
        else:
            u, v = x[:shift], y[-shift:]
        r = pearson(u, v)
        if r > best[1]:
            best = (shift, r)
    return best[0] / rate_hz, best[1]


@dataclass
class CommandComparison:
    """Accumulates paired (EMG command, glove command) samples and summarizes them."""

    joint_names: Sequence[str]
    times: list[float] = field(default_factory=list)          # when the pair was formed (monotonic s)
    emg_times: list[float] = field(default_factory=list)      # control time of the EMG hand
    emg_sample_times: list[float] = field(default_factory=list)  # time of its EMG sample
    glove_times: list[float] = field(default_factory=list)    # capture time of the glove hand
    emg: list[np.ndarray] = field(default_factory=list)       # (22,) rad
    glove: list[np.ndarray] = field(default_factory=list)     # (22,) rad

    def add(self, time_s: float, emg_time: float, emg_sample_time: float, glove_time: float,
            emg_command: np.ndarray, glove_command: np.ndarray) -> None:
        emg_command = np.asarray(emg_command, dtype=np.float64).reshape(-1)
        glove_command = np.asarray(glove_command, dtype=np.float64).reshape(-1)
        if emg_command.shape != (len(self.joint_names),) or glove_command.shape != emg_command.shape:
            raise ValueError("commands must have one value per joint")
        self.times.append(float(time_s))
        self.emg_times.append(float(emg_time))
        self.emg_sample_times.append(float(emg_sample_time))
        self.glove_times.append(float(glove_time))
        self.emg.append(emg_command)
        self.glove.append(glove_command)

    def reset(self) -> None:
        for values in (self.times, self.emg_times, self.emg_sample_times, self.glove_times, self.emg, self.glove):
            values.clear()

    def __len__(self) -> int:
        return len(self.times)

    def _arrays(self, window_s: float | None):
        times = np.asarray(self.times)
        emg = np.asarray(self.emg)
        glove = np.asarray(self.glove)
        if window_s is not None and len(times):
            keep = times >= times[-1] - window_s
            return times[keep], emg[keep], glove[keep]
        return times, emg, glove

    def snapshot(self, window_s: float | None = None, minimum_samples: int = 10) -> dict | None:
        """MAE, constant baseline, ratio and correlation, overall and per finger, in degrees.

        ``window_s`` restricts the errors to the most recent seconds; the constant baseline is always the
        glove's mean over the whole session so far.
        """

        times, emg, glove = self._arrays(window_s)
        if len(times) < minimum_samples:
            return None
        session_mean = np.asarray(self.glove).mean(axis=0)
        error = np.degrees(np.abs(emg - glove))
        constant = np.degrees(np.abs(glove - session_mean[None, :]))
        joint_mae, joint_const = error.mean(axis=0), constant.mean(axis=0)
        joint_r = [pearson(emg[:, j], glove[:, j]) for j in range(emg.shape[1])]
        fingers = {}
        for finger in FINGERS:
            index = [j for j, name in enumerate(self.joint_names) if finger_of(name) == finger]
            if index:
                r = [joint_r[j] for j in index if math.isfinite(joint_r[j])]
                fingers[finger] = {
                    "mae_deg": float(joint_mae[index].mean()),
                    "const_deg": float(joint_const[index].mean()),
                    "r": float(np.mean(r)) if r else float("nan"),
                }
        mae, const = float(joint_mae.mean()), float(joint_const.mean())
        finite_r = [r for r in joint_r if math.isfinite(r)]
        return {
            "samples": int(len(times)),
            "seconds": float(times[-1] - times[0]),
            "mae_deg": mae,
            "const_deg": const,
            "ratio": mae / const if const >= MINIMUM_CONSTANT_DEG else float("nan"),
            "glove_moving": bool(const >= MINIMUM_CONSTANT_DEG),
            "mean_r": float(np.mean(finite_r)) if finite_r else float("nan"),
            "fingers": fingers,
            "joints": [
                {"joint": name, "mae_deg": float(joint_mae[j]), "const_deg": float(joint_const[j]), "r": joint_r[j]}
                for j, name in enumerate(self.joint_names)
            ],
        }

    def summary(self) -> dict | None:
        """Whole-session report: :meth:`snapshot` plus lag and the open / closed extremes."""

        report = self.snapshot(None)
        if report is None:
            return None
        emg = np.asarray(self.emg)
        glove = np.asarray(self.glove)
        flexion = [j for j, name in enumerate(self.joint_names) if is_flexion(name) and finger_of(name) != "thumb"]
        emg_flexion, glove_flexion = emg[:, flexion].mean(axis=1), glove[:, flexion].mean(axis=1)
        session_mean = glove.mean(axis=0)
        low, high = np.percentile(glove_flexion, [10.0, 90.0])
        extremes = {}
        for name, mask in (("open", glove_flexion <= low), ("closed", glove_flexion >= high)):
            if mask.sum() >= 5:
                extremes[name] = {
                    "samples": int(mask.sum()),
                    "mae_deg": float(np.degrees(np.abs(emg[mask] - glove[mask])).mean()),
                    "const_deg": float(np.degrees(np.abs(glove[mask] - session_mean[None, :])).mean()),
                    # EMG minus glove on the finger flexion joints: negative = EMG commands less flexion
                    "flexion_bias_deg": float(np.degrees(emg_flexion[mask] - glove_flexion[mask]).mean()),
                }
        report["extremes"] = extremes
        report["flexion_r"] = pearson(emg_flexion, glove_flexion)
        lags = {}
        for label, emg_time in (("control_time", self.emg_times), ("emg_sample_time", self.emg_sample_times)):
            # the series are only as dense as the pairs; unique, increasing times are required
            t_emg, first = np.unique(np.asarray(emg_time), return_index=True)
            t_glove, first_glove = np.unique(np.asarray(self.glove_times), return_index=True)
            result = None
            if len(t_emg) > 20 and len(t_glove) > 20:
                result = estimate_lag_s(t_emg, emg_flexion[first], t_glove, glove_flexion[first_glove])
            lags[label] = None if result is None else {"emg_behind_glove_ms": 1e3 * result[0], "peak_r": result[1]}
        report["lag"] = lags
        return report


def format_snapshot(label: str, snapshot: dict | None) -> str:
    if snapshot is None:
        return f"{label}: collecting..."
    fingers = "  ".join(f"{name} {row['mae_deg']:.1f}" for name, row in snapshot["fingers"].items())
    if not snapshot["glove_moving"]:
        return (
            f"{label}: the glove hand is not moving yet (its command varies {snapshot['const_deg']:.2f} deg); "
            f"EMG vs glove {snapshot['mae_deg']:.2f} deg | per finger: {fingers}"
        )
    return (
        f"{label}: EMG vs glove {snapshot['mae_deg']:.2f} deg | constant {snapshot['const_deg']:.2f} deg | "
        f"ratio {snapshot['ratio']:.2f} | mean r {snapshot['mean_r']:.2f} | per finger: {fingers}"
    )


__all__ = ["FINGERS", "MINIMUM_CONSTANT_DEG", "CommandComparison", "estimate_lag_s", "finger_of", "format_snapshot", "is_flexion", "pearson"]
