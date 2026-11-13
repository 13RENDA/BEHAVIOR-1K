"""Wrist teleoperation telemetry shared by launch_og.py and the wrist bench (no simulator required)."""

import pytest
from scipy.spatial.transform import Rotation

from dex_teleop.omnigibson.teleop_telemetry import WristTelemetry, target_position_world


PROVENANCE = {
    "tracker_health": "healthy",
    "validated_tracker_id": "FA5AG3B00121",
    "validated_tracker_quality": "trackable",
}


def _observe(telemetry, now, **overrides):
    arguments = dict(
        now=now,
        wrist_position=(0.1, 0.2, 0.3),
        wrist_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_receipt_timestamp=now - 0.05,
        wrist_provenance=PROVENANCE,
        target_position_world=(0.0, 0.0, 1.0),
        eef_position_world=(0.0, 0.0, 1.0),
        gate_decision="accepted",
    )
    arguments.update(overrides)
    return telemetry.observe(**arguments)


def test_telemetry_measures_motion_since_the_anchor_and_reports_the_tracker():
    telemetry = WristTelemetry(interval_s=1.0)
    first = _observe(telemetry, 100.0)
    assert first is not None and first.elapsed_s == 0.0 and first.displacement_m == 0.0
    assert first.tracker_id == "FA5AG3B00121" and first.tracker_quality == "trackable"
    assert first.frame_age_s == pytest.approx(0.05)
    assert first.loop_hz is None  # no window before the first line
    assert _observe(telemetry, 100.5) is None  # within the interval
    yaw_30 = Rotation.from_euler("z", 30, degrees=True).as_quat()
    second = _observe(
        telemetry,
        101.2,
        wrist_position=(0.1, 0.25, 0.3),
        wrist_quaternion_xyzw=yaw_30,
        frame_receipt_timestamp=101.19,
        wrist_provenance={
            **PROVENANCE,
            "tracker_health": "unhealthy",
            "tracker_health_reason": "tracker quality is 'bad'",
            "validated_tracker_quality": "bad",
        },
        eef_position_world=(0.0, 0.012, 1.0),
        gate_decision="held",
    )
    assert second is not None
    assert second.elapsed_s == pytest.approx(1.2) and second.displacement_m == pytest.approx(0.05)
    assert second.rotation_deg == pytest.approx(30.0)
    assert second.tracking_error_m == pytest.approx(0.012)
    line = second.format()
    assert line.startswith("[wrist bench t=   1.2s] FA5AG3B00121 quality=bad health=unhealthy (tracker quality is 'bad')")
    assert "wrist moved   5.0 cm" in line and "rotated  30.0 deg" in line and "12.0 mm" in line
    assert line.endswith("gate=held")
    # Re-anchoring restarts the clock and the reference pose.
    telemetry.reset()
    again = _observe(telemetry, 200.0, wrist_position=(1.0, 1.0, 1.0), wrist_quaternion_xyzw=yaw_30)
    assert again is not None and again.displacement_m == 0.0 and again.rotation_deg == 0.0 and again.elapsed_s == 0.0
    no_tracker = _observe(WristTelemetry(0.1), 1.0, wrist_provenance=None, gate_decision=None)
    assert no_tracker is not None and "no tracker quality=n/a health=n/a" in no_tracker.format("arat_grasp_block_10cm")
    assert no_tracker.format("arat_grasp_block_10cm").startswith("[arat_grasp_block_10cm t=")
    with pytest.raises(ValueError, match="interval_s"):
        WristTelemetry(0.0)


def test_telemetry_reports_loop_rate_step_time_and_rate_limit_scale():
    telemetry = WristTelemetry(interval_s=1.0)
    assert _observe(telemetry, 10.0, step_seconds=0.03) is not None
    # Twelve applied actions over the next second, each env.step 60-90 ms, two of them rate-limit scaled.
    now = 10.0
    for index in range(12):
        now += 1.0 / 12.0
        sample = _observe(
            telemetry,
            now,
            step_seconds=0.06 + 0.03 * (index % 2),
            rate_limit_scale=2.5 if index in (3, 7) else 1.0,
        )
        if index < 11:
            assert sample is None
    assert sample is not None
    assert sample.loop_hz == pytest.approx(12.0, rel=1e-6)
    assert sample.step_seconds_mean == pytest.approx(0.075) and sample.step_seconds_max == pytest.approx(0.09)
    assert sample.rate_limit_scale == pytest.approx(2.5)
    assert "loop 12.0 Hz, env.step  75/ 90 ms, rate limits x2.5" in sample.format()
    # A loop that keeps up (scale 1.0) does not mention the rate limits.
    keeping_up = WristTelemetry(0.5)
    _observe(keeping_up, 0.0, step_seconds=0.02, rate_limit_scale=1.0)
    quick = _observe(keeping_up, 0.5, step_seconds=0.02, rate_limit_scale=1.0)
    assert quick is not None and "rate limits" not in quick.format() and "loop  2.0 Hz" in quick.format()


def test_telemetry_evaluates_pose_callables_only_when_a_line_is_due():
    calls = []
    telemetry = WristTelemetry(interval_s=1.0)

    def eef():
        calls.append("eef")
        return (0.0, 0.0, 1.0)

    assert _observe(telemetry, 0.0, eef_position_world=eef) is not None
    assert _observe(telemetry, 0.2, eef_position_world=eef) is None
    assert _observe(telemetry, 0.4, eef_position_world=eef) is None
    assert calls == ["eef"]
    assert _observe(telemetry, 1.1, eef_position_world=eef) is not None
    assert calls == ["eef", "eef"]


def test_target_position_world_applies_the_robot_base_pose():
    yaw_90 = Rotation.from_euler("z", 90, degrees=True).as_quat()
    world = target_position_world((1.0, 0.0, 0.5), (-0.5, 0.0, 0.0), yaw_90)
    assert world.tolist() == pytest.approx([-0.5, 1.0, 0.5])
