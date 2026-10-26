from types import SimpleNamespace
import json

import pytest

from dex_teleop.omnigibson.launcher import (
    _effective_manus_hand_motion,
    _manus_core_calibration,
    _parser,
    _resolve_tracking_selection,
    _set_hand_tracking_metadata,
    _validate_manus_selection,
)
from dex_teleop.omnigibson.hand_tracking_recording import (
    HandTrackingRecordingSession,
)
from dex_teleop.tracking.manus_calibration import (
    MANUS_DEFAULT_REFERENCE_FRAME,
    ManusCoreCalibration,
)


def _arguments(*arguments):
    catalog = SimpleNamespace(tasks={"activity": object()}, subscales={})
    return _parser(catalog).parse_args(arguments)


def _write_calibration(path):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "core_world_frame": "manus_core_world_y_up_rh_z_to_viewer_m",
                "reference_frame": "robot_world",
                "reference_from_core_world": {
                    "translation": [0.0, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "wrists": [
                    {
                        "handedness": "right",
                        "core_tracker_offset_applied": True,
                        "tracker_id": "ultimate-right",
                        "skeleton_wrist_to_anatomical_wrist": {
                            "translation": [0.0, 0.0, 0.0],
                            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_existing_manus_hand_invocation_remains_integrated_gloves_only():
    args = _arguments("--hand-source", "manus")
    selection = _resolve_tracking_selection(args)

    _validate_manus_selection(args, selection)

    assert selection.hand_source == "manus"
    assert selection.wrist_source == "quest"
    assert args.manus_mode == "integrated"
    assert _effective_manus_hand_motion(selection) == "auto"


def test_hand_motion_is_derived_from_the_wrist_role():
    articulation_only = _resolve_tracking_selection(
        _arguments("--hand-source", "manus", "--manus-mode", "remote")
    )
    record_only_wrist = _resolve_tracking_selection(
        _arguments("--record-wrist-source", "manus", "--manus-mode", "remote")
    )

    assert _effective_manus_hand_motion(articulation_only) == "auto"
    assert _effective_manus_hand_motion(record_only_wrist) == "tracker"
    with pytest.raises(SystemExit):
        _arguments("--hand-source", "manus", "--manus-hand-motion", "tracker")


def test_remote_manus_wrist_falls_back_to_the_default_core_calibration():
    args = _arguments(
        "--wrist-source",
        "manus",
        "--manus-mode",
        "remote",
        "--manus-expected-tracker-id",
        "ultimate-right",
        "--manus-expected-tracker-user-id",
        "1",
    )
    selection = _resolve_tracking_selection(args)

    _validate_manus_selection(args, selection)

    calibration = _manus_core_calibration(args, selection)
    assert isinstance(calibration, ManusCoreCalibration) and calibration.is_default
    assert calibration.reference_frame == MANUS_DEFAULT_REFERENCE_FRAME
    # Articulation-only MANUS keeps the raw Core world: its root is never used.
    assert (
        _manus_core_calibration(
            args, _resolve_tracking_selection(_arguments("--hand-source", "manus"))
        )
        is None
    )
    # An explicit file still wins.
    assert _manus_core_calibration(
        _arguments(
            "--wrist-source",
            "manus",
            "--manus-mode",
            "remote",
            "--manus-core-calibration",
            "measured.json",
        ),
        selection,
    ) == "measured.json"


def test_launcher_preserves_negotiated_manus_protocol_metadata():
    class _LegacySource:
        def recording_metadata(self):
            return {
                "bridge_protocol_version": 1,
                "supported_protocol_version": 2,
                "bridge_capabilities": ["integrated_wire_v1"],
            }

    source = _LegacySource()
    worker = SimpleNamespace(
        fuser=SimpleNamespace(articulation_to_wrist=None),
        articulation_streams={"manus": "articulation.manus"},
        wrist_streams={"manus": "wrist.manus"},
        retargeting_stream="retargeted",
        sources={"manus": source},
        articulation_source="manus",
        wrist_source="manus",
        control_wrist_stream="wrist.control",
    )
    worker.query = lambda function, *args: function(worker, *args)
    session = HandTrackingRecordingSession()
    args = _arguments("--hand-source", "manus")

    _set_hand_tracking_metadata(session, worker, args)
    metadata = session.writer_kwargs()["stream_metadata"]

    assert metadata["articulation.manus"]["bridge_protocol_version"] == 1
    assert metadata["wrist.manus"]["bridge_protocol_version"] == 1


def test_remote_manus_can_serve_both_selected_modalities(tmp_path):
    calibration = _write_calibration(tmp_path / "calibration.json")
    args = _arguments(
        "--hand-source",
        "manus",
        "--wrist-source",
        "manus",
        "--manus-mode",
        "remote",
        "--manus-core-host",
        "192.0.2.10",
        "--manus-core-calibration",
        str(calibration),
        "--manus-expected-tracker-id",
        "ultimate-right",
        "--manus-expected-tracker-user-id",
        "1",
    )
    selection = _resolve_tracking_selection(args)

    _validate_manus_selection(args, selection)

    assert selection.articulation_sources == ("manus",)
    assert selection.wrist_sources == ("manus",)
    assert _effective_manus_hand_motion(selection) == "tracker"


def test_remote_manus_can_be_a_record_only_wrist_source(tmp_path):
    calibration = _write_calibration(tmp_path / "calibration.json")
    args = _arguments(
        "--record-wrist-source",
        "manus",
        "--manus-mode",
        "remote",
        "--manus-core-calibration",
        str(calibration),
        "--manus-expected-tracker-id",
        "ultimate-right",
        "--manus-expected-tracker-user-id",
        "1",
    )
    selection = _resolve_tracking_selection(args)

    _validate_manus_selection(args, selection)

    assert selection.wrist_source == "quest"
    assert selection.record_wrist_sources == ("manus",)


def test_remote_manus_wrist_policy_is_separate_from_optional_calibration_id(tmp_path):
    calibration = _write_calibration(tmp_path / "calibration.json")
    document = json.loads(calibration.read_text(encoding="utf-8"))
    document["wrists"][0]["tracker_id"] = None
    calibration.write_text(json.dumps(document), encoding="utf-8")
    args = _arguments(
        "--wrist-source",
        "manus",
        "--manus-mode",
        "remote",
        "--manus-core-calibration",
        str(calibration),
        "--manus-expected-tracker-id",
        "ultimate-right",
        "--manus-expected-tracker-user-id",
        "1",
    )
    selection = _resolve_tracking_selection(args)

    _validate_manus_selection(args, selection)


@pytest.mark.parametrize(
    ("policy_arguments", "message"),
    (
        (
            ("--manus-expected-tracker-user-id", "1"),
            "requires --manus-expected-tracker-id",
        ),
        (
            ("--manus-expected-tracker-id", "ultimate-right"),
            "requires --manus-expected-tracker-user-id",
        ),
    ),
)
def test_remote_manus_wrist_requires_complete_runtime_policy(
    tmp_path, policy_arguments, message
):
    calibration = _write_calibration(tmp_path / "calibration.json")
    args = _arguments(
        "--wrist-source",
        "manus",
        "--manus-mode",
        "remote",
        "--manus-core-calibration",
        str(calibration),
        *policy_arguments,
    )
    selection = _resolve_tracking_selection(args)

    with pytest.raises(SystemExit, match=message):
        _validate_manus_selection(args, selection)


def test_remote_manus_wrist_rejects_calibration_tracker_id_mismatch(tmp_path):
    calibration = _write_calibration(tmp_path / "calibration.json")
    args = _arguments(
        "--wrist-source",
        "manus",
        "--manus-mode",
        "remote",
        "--manus-core-calibration",
        str(calibration),
        "--manus-expected-tracker-id",
        "different-ultimate",
        "--manus-expected-tracker-user-id",
        "1",
    )
    selection = _resolve_tracking_selection(args)

    with pytest.raises(SystemExit, match="audit metadata must match"):
        _validate_manus_selection(args, selection)


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (
            ("--wrist-source", "manus"),
            "needs a tracker: --manus-tracker-feed vive_ultimate",
        ),
        (
            (
                "--hand-source",
                "manus",
                "--manus-core-calibration",
                "calibration.json",
            ),
            "--manus-core-calibration.*requires a selected or record-only MANUS wrist source",
        ),
        (
            (
                "--wrist-source",
                "manus",
                "--manus-mode",
                "remote",
            ),
            "requires --manus-expected-tracker-id",
        ),
        (
            (
                "--manus-mode",
                "remote",
                "--manus-core-host",
                "windows-core",
            ),
            "require a selected or record-only MANUS source",
        ),
        (
            (
                "--wrist-source",
                "manus",
                "--manus-mode",
                "remote",
                "--manus-core-calibration",
                "calibration.json",
                "--no-manus-tracker-diagnostics",
            ),
            "requires tracker monitoring",
        ),
        (
            ("--manus-tracker-feed", "vive_ultimate"),
            "require a selected or record-only MANUS source",
        ),
        (
            ("--hand-source", "manus", "--manus-tracker-feed", "vive_ultimate"),
            "--manus-tracker-feed supplies the MANUS wrist",
        ),
        (
            (
                "--wrist-source",
                "manus",
                "--manus-mode",
                "remote",
                "--manus-tracker-feed",
                "vive_ultimate",
            ),
            "requires --manus-mode integrated",
        ),
        (
            (
                "--wrist-source",
                "manus",
                "--manus-backend",
                "native",
                "--manus-tracker-feed",
                "vive_ultimate",
            ),
            "use --manus-backend python",
        ),
        (
            ("--hand-source", "manus", "--manus-backend", "native", "--manus-haptics"),
            "use --manus-backend python",
        ),
        (
            ("--hand-source", "manus", "--manus-bridge", "/opt/manus_bridge"),
            "requires --manus-backend native",
        ),
        (
            ("--wrist-source", "manus", "--manus-tracker-feed", "vive_ultimate"),
            "requires --manus-expected-tracker-id.*serial",
        ),
    ),
)
def test_manus_launcher_rejects_unsafe_or_unused_combinations(arguments, message):
    args = _arguments(*arguments)
    selection = _resolve_tracking_selection(args)

    with pytest.raises(SystemExit, match=message):
        _validate_manus_selection(args, selection)


def test_integrated_vive_ultimate_feed_serves_the_wrist_without_a_user_id():
    args = _arguments(
        "--hand-source",
        "manus",
        "--wrist-source",
        "manus",
        "--manus-tracker-feed",
        "vive_ultimate",
        "--manus-expected-tracker-id",
        "59-A33R03250",
        "--manus-tracker-to-wrist-translation=-0.003,-0.058,-0.043",
    )
    selection = _resolve_tracking_selection(args)

    _validate_manus_selection(args, selection)

    assert args.manus_backend == "python" and args.manus_mode == "integrated"
    assert args.manus_tracker_to_wrist_translation == (-0.003, -0.058, -0.043)
    assert _effective_manus_hand_motion(selection) == "tracker"
    calibration = _manus_core_calibration(args, selection)
    assert isinstance(calibration, ManusCoreCalibration) and calibration.is_default
    # An explicit user id is still accepted and validated.
    pinned = _arguments(
        "--wrist-source", "manus", "--manus-tracker-feed", "vive_ultimate",
        "--manus-expected-tracker-id", "59-A33R03250", "--manus-expected-tracker-user-id", "-1",
    )
    with pytest.raises(SystemExit, match="non-negative"):
        _validate_manus_selection(pinned, _resolve_tracking_selection(pinned))
    with pytest.raises(SystemExit):  # argparse: "expected 3 finite comma-separated numbers"
        _arguments("--manus-tracker-to-wrist-translation", "1,2")


def test_launcher_builds_the_python_manus_source_with_the_feed(monkeypatch):
    from dex_teleop.omnigibson import launcher

    captured = {}

    class _Recorder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(launcher, "ManusIntegratedSource", _Recorder)
    args = _arguments(
        "--hand-source", "manus", "--wrist-source", "manus",
        "--manus-tracker-feed", "vive_ultimate",
        "--manus-expected-tracker-id", "59-A33R03250",
        "--vive-ultimate-socket", "/run/vut.sock",
        "--manus-haptics",
    )
    selection = _resolve_tracking_selection(args)
    _validate_manus_selection(args, selection)

    sources = launcher._create_tracking_sources(args, selection)

    assert set(sources) == {"manus"}
    assert captured["backend"] == "python"
    assert captured["mode"] == "integrated"
    assert captured["tracker_feed"] == "vive_ultimate"
    assert captured["vive_ultimate_socket"] == "/run/vut.sock"
    assert captured["wrist_composition"] == "bridge"
    assert captured["hand_motion"] == "tracker"
    assert captured["require_tracker_for_wrist"] is True
    assert captured["expected_tracker_user_id"] is None
    assert captured["haptics"] is True
    assert captured["tracker_to_wrist"] is None and captured["tracker_to_casing"] is None


def test_manus_sdk_metadata_names_the_python_bridge():
    from dex_teleop.omnigibson.launcher import _manus_sdk_metadata

    metadata = _manus_sdk_metadata(_arguments("--hand-source", "manus"))

    assert metadata["backend"] == "python"
    assert metadata["bridge"]["file"] == "manus_python_bridge.py"
    assert len(metadata["bridge"]["sha256"]) == 64
