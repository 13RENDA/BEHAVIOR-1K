"""EMG-vs-glove command comparison statistics (simulator-free)."""

import math

import numpy as np
import pytest

from dex_teleop.emg.compare import CommandComparison, estimate_lag_s, finger_of, format_snapshot, is_flexion

JOINTS = ("right_index_MCP_FE", "right_index_MCP_AA", "right_index_PIP", "right_thumb_CMC_FE", "right_thumb_IP")


def _session(error_deg=0.0, lag_s=0.0, seconds=30.0, rate=20.0, still=False):
    comparison = CommandComparison(JOINTS)
    times = np.arange(0.0, seconds, 1.0 / rate)

    def glove_at(t):
        flexion = 0.0 if still else 0.5 + 0.4 * np.sin(2 * np.pi * 0.4 * t) + 0.15 * np.sin(2 * np.pi * 1.1 * t + 1.0)
        return np.array([flexion, 0.05, 0.8 * flexion, 0.3 * flexion, 0.5 * flexion])

    for t in times:
        emg = glove_at(t - lag_s) + np.radians(error_deg)
        comparison.add(t, t, t - 0.06, t, emg, glove_at(t))
    return comparison


def test_joint_name_helpers():
    assert finger_of("right_index_MCP_FE") == "index" and finger_of("right_thumb_IP") == "thumb"
    assert is_flexion("right_index_PIP") and not is_flexion("right_index_MCP_AA")


def test_identical_hands_score_zero_error_and_full_correlation():
    snapshot = _session().snapshot()
    assert snapshot["mae_deg"] == pytest.approx(0.0, abs=1e-9)
    assert snapshot["const_deg"] > 3.0 and snapshot["ratio"] == pytest.approx(0.0, abs=1e-9)
    assert snapshot["fingers"]["index"]["r"] == pytest.approx(1.0)
    assert snapshot["glove_moving"] and set(snapshot["fingers"]) == {"thumb", "index"}


def test_constant_offset_is_reported_in_degrees_next_to_the_constant_baseline():
    comparison = _session(error_deg=5.0)
    snapshot = comparison.snapshot()
    assert snapshot["mae_deg"] == pytest.approx(5.0, abs=1e-6)
    glove = np.asarray(comparison.glove)
    expected_const = np.degrees(np.abs(glove - glove.mean(axis=0)).mean(axis=0)).mean()
    assert snapshot["const_deg"] == pytest.approx(expected_const)
    assert snapshot["ratio"] == pytest.approx(5.0 / expected_const)
    recent = comparison.snapshot(window_s=5.0)
    assert recent["seconds"] <= 5.0 + 1e-9 and recent["samples"] < snapshot["samples"]
    assert "ratio" in format_snapshot("last 5 s", recent)


def test_a_still_glove_gives_no_ratio():
    snapshot = _session(error_deg=4.0, still=True).snapshot()
    assert not snapshot["glove_moving"] and math.isnan(snapshot["ratio"])
    assert "not moving" in format_snapshot("now", snapshot)
    assert format_snapshot("now", None).endswith("collecting...")


def test_summary_reports_lag_and_extremes():
    summary = _session(lag_s=0.1).summary()
    lag = summary["lag"]["control_time"]
    assert lag["emg_behind_glove_ms"] == pytest.approx(100.0, abs=15.0) and lag["peak_r"] > 0.95
    # the EMG sample time is 60 ms earlier than the control time
    assert summary["lag"]["emg_sample_time"]["emg_behind_glove_ms"] == pytest.approx(40.0, abs=15.0)
    assert set(summary["extremes"]) == {"open", "closed"}
    under = _session(error_deg=-6.0).summary()
    assert under["extremes"]["closed"]["flexion_bias_deg"] == pytest.approx(-6.0, abs=1e-6)


def test_lag_needs_enough_overlap():
    t = np.arange(0.0, 1.0, 0.05)
    assert estimate_lag_s(t, np.sin(t), t, np.sin(t)) is None


def test_reset_and_validation():
    comparison = _session()
    comparison.reset()
    assert len(comparison) == 0 and comparison.snapshot() is None and comparison.summary() is None
    with pytest.raises(ValueError):
        comparison.add(0.0, 0.0, 0.0, 0.0, np.zeros(3), np.zeros(3))
