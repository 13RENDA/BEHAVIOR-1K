"""Periodic wrist-teleoperation telemetry shared by the ARAT launcher and the wrist bench.

One line per interval while engaged: the tracker's identity, quality and
health, how far and how much the tracked wrist moved since the anchor, the
commanded-target to measured-EEF error, the frame age, and the control-loop
rate with the ``env.step`` wall time behind it. The numbers make the usual
"it feels laggy/unstable" reports attributable: a loop well below 30 Hz with
long steps is a rendering/recording load problem, a healthy loop with a growing
target-EEF error is the arm, a stale frame age is tracking, and a degraded
tracker quality is the Ultimate. Simulator-free so it is unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation


def _quaternion_angle_deg(first, second) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return math.degrees(2.0 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


def _resolve(value):
    return value() if callable(value) else value


def target_position_world(target_position_robot, base_position_world, base_quaternion_world_xyzw) -> np.ndarray:
    """Express a robot-frame target position in the world frame."""

    rotation = Rotation.from_quat(np.asarray(base_quaternion_world_xyzw, dtype=np.float64))
    return rotation.apply(np.asarray(target_position_robot, dtype=np.float64)) + np.asarray(
        base_position_world, dtype=np.float64
    )


@dataclass(frozen=True)
class WristTelemetrySample:
    """One engaged-period observation of the tracked wrist and the arm that follows it."""

    elapsed_s: float
    frame_age_s: float
    tracker_id: str | None
    tracker_quality: str | None
    tracker_health: str | None
    tracker_health_reason: str | None
    displacement_m: float
    rotation_deg: float
    tracking_error_m: float
    gate_decision: str | None
    loop_hz: float | None = None
    step_seconds_mean: float | None = None
    step_seconds_max: float | None = None
    rate_limit_scale: float | None = None

    def format(self, label: str = "wrist bench") -> str:
        tracker = self.tracker_id or "no tracker"
        quality = self.tracker_quality or "n/a"
        health = self.tracker_health or "n/a"
        if self.tracker_health == "unhealthy" and self.tracker_health_reason:
            health = f"unhealthy ({self.tracker_health_reason})"
        gate = f" gate={self.gate_decision}" if self.gate_decision not in (None, "accepted") else ""
        loop = ""
        if self.loop_hz is not None:
            loop = f" | loop {self.loop_hz:4.1f} Hz"
            if self.step_seconds_mean is not None and self.step_seconds_max is not None:
                loop += f", env.step {self.step_seconds_mean * 1000:3.0f}/{self.step_seconds_max * 1000:3.0f} ms"
            if self.rate_limit_scale is not None and self.rate_limit_scale > 1.0:
                loop += f", rate limits x{self.rate_limit_scale:.1f}"
        return (
            f"[{label} t={self.elapsed_s:6.1f}s] {tracker} quality={quality} health={health} | "
            f"wrist moved {self.displacement_m * 100:5.1f} cm, rotated {self.rotation_deg:5.1f} deg since anchor | "
            f"target-EEF error {self.tracking_error_m * 1000:5.1f} mm | frame age {self.frame_age_s * 1000:4.0f} ms"
            f"{loop}{gate}"
        )


class WristTelemetry:
    """Accumulate the anchor pose and loop statistics; emit a sample every ``interval_s``.

    ``observe`` is called once per applied action. ``target_position_world``
    and ``eef_position_world`` may be arrays or zero-argument callables; the
    callables are evaluated only when a sample is due, so the simulator is not
    queried on every step just for telemetry.
    """

    def __init__(self, interval_s: float = 1.0) -> None:
        if not math.isfinite(interval_s) or interval_s <= 0.0:
            raise ValueError("interval_s must be positive and finite")
        self.interval_s = float(interval_s)
        self.reset()

    def reset(self) -> None:
        self._anchor_position: np.ndarray | None = None
        self._anchor_quaternion: np.ndarray | None = None
        self._engaged_at: float | None = None
        self._last_emit: float | None = None
        self._steps_since_emit = 0
        self._step_seconds: list[float] = []
        self._max_rate_limit_scale: float | None = None

    def observe(
        self,
        *,
        now: float,
        wrist_position,
        wrist_quaternion_xyzw,
        frame_receipt_timestamp: float,
        wrist_provenance: Mapping[str, Any] | None,
        target_position_world,
        eef_position_world,
        gate_decision: str | None,
        step_seconds: float | None = None,
        rate_limit_scale: float | None = None,
    ) -> WristTelemetrySample | None:
        """Record one applied action; return a sample when the interval elapsed."""

        position = np.asarray(wrist_position, dtype=np.float64)
        quaternion = np.asarray(wrist_quaternion_xyzw, dtype=np.float64)
        if self._anchor_position is None:
            self._anchor_position = position.copy()
            self._anchor_quaternion = quaternion.copy()
            self._engaged_at = now
            window_start = now
        else:
            window_start = self._last_emit if self._last_emit is not None else self._engaged_at
        self._steps_since_emit += 1
        if step_seconds is not None and math.isfinite(step_seconds):
            self._step_seconds.append(float(step_seconds))
        if rate_limit_scale is not None and math.isfinite(rate_limit_scale):
            current = self._max_rate_limit_scale
            self._max_rate_limit_scale = float(rate_limit_scale) if current is None else max(current, float(rate_limit_scale))
        if self._last_emit is not None and now - self._last_emit < self.interval_s:
            return None

        window = now - (window_start if window_start is not None else now)
        loop_hz = None if self._last_emit is None else (self._steps_since_emit / window if window > 0.0 else None)
        step_seconds_mean = float(np.mean(self._step_seconds)) if self._step_seconds else None
        step_seconds_max = float(np.max(self._step_seconds)) if self._step_seconds else None
        sample = WristTelemetrySample(
            elapsed_s=now - (self._engaged_at if self._engaged_at is not None else now),
            frame_age_s=max(0.0, now - frame_receipt_timestamp),
            tracker_id=(wrist_provenance or {}).get("validated_tracker_id"),
            tracker_quality=(wrist_provenance or {}).get("validated_tracker_quality"),
            tracker_health=(wrist_provenance or {}).get("tracker_health"),
            tracker_health_reason=(wrist_provenance or {}).get("tracker_health_reason"),
            displacement_m=float(np.linalg.norm(position - self._anchor_position)),
            rotation_deg=_quaternion_angle_deg(quaternion, self._anchor_quaternion),
            tracking_error_m=float(
                np.linalg.norm(
                    np.asarray(_resolve(target_position_world), dtype=np.float64)
                    - np.asarray(_resolve(eef_position_world), dtype=np.float64)
                )
            ),
            gate_decision=gate_decision,
            loop_hz=loop_hz,
            step_seconds_mean=step_seconds_mean,
            step_seconds_max=step_seconds_max,
            rate_limit_scale=self._max_rate_limit_scale,
        )
        self._last_emit = now
        self._steps_since_emit = 0
        self._step_seconds = []
        self._max_rate_limit_scale = None
        return sample


__all__ = ["WristTelemetry", "WristTelemetrySample", "target_position_world"]
