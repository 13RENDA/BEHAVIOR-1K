"""YAML launch presets for launch_og.py: shipped presets, precedence, validation."""

from pathlib import Path

import pytest

from dex_teleop.arat import AratTaskCatalog
from dex_teleop.omnigibson import launcher
from dex_teleop.omnigibson.launch_config import (
    DEFAULT_LAUNCH_CONFIG,
    LAUNCH_CONFIG_ROOT,
    available_launch_configs,
    load_launch_config,
    parse_launch_arguments,
    resolve_launch_config_path,
)
from dex_teleop.omnigibson.launcher import (
    _effective_manus_hand_motion,
    _parser,
    _resolve_tracking_selection,
    _validate_manus_selection,
    main,
)

TASK = "arat_grasp_block_10cm"


def _parse(*argv: str, default_config: str | None = DEFAULT_LAUNCH_CONFIG):
    return parse_launch_arguments(
        _parser(AratTaskCatalog()), list(argv), default_config=default_config
    )


def _write(tmp_path: Path, document: str) -> Path:
    path = tmp_path / "preset.yaml"
    path.write_text(document, encoding="utf-8")
    return path


def test_default_preset_is_the_linux_manus_core_path():
    assert DEFAULT_LAUNCH_CONFIG == "manus_core"
    assert {"manus_core", "hts"} <= set(available_launch_configs())

    args, config = _parse("--task", TASK)

    assert config is not None
    assert config.path == (LAUNCH_CONFIG_ROOT / "manus_core.yaml").resolve()
    assert len(config.sha256) == 64
    selection = _resolve_tracking_selection(args)
    assert (selection.hand_source, selection.wrist_source) == ("manus", "manus")
    # MANUS Core 3.2 Python SDK, Integrated on Linux, Ultimate from VIVE Hub for
    # Linux: no Windows Core and no C++ sidecar anywhere in the default path.
    assert args.manus_backend == "python"
    assert args.manus_mode == "integrated"
    assert args.manus_tracker_feed == "vive_ultimate"
    assert args.vive_ultimate_socket == "/tmp/vut.sock"
    assert args.vive_ultimate_frame == "x_right_y_up_z_back"
    assert args.manus_wrist_composition == "bridge"
    assert args.manus_core_host is None and args.manus_loopback_only is False
    assert _effective_manus_hand_motion(selection) == "tracker"
    assert args.manus_tracker_diagnostics is True
    assert args.manus_tracker_stale_timeout == 1.0
    assert args.record_hand_poses is True
    assert args.manus_haptics is False
    # The Core-world calibration falls back to the built-in default; the rig's
    # tracker serial is filled in and the glove owner is resolved at runtime.
    assert args.manus_core_calibration is None
    assert args.manus_expected_tracker_id == "FA5AG3B00121"
    assert args.manus_expected_tracker_user_id is None
    _validate_manus_selection(args, selection)


def test_default_preset_without_a_tracker_serial_names_the_missing_key(tmp_path):
    document = (LAUNCH_CONFIG_ROOT / "manus_core.yaml").read_text(encoding="utf-8")
    document = document.replace('manus_expected_tracker_id: "FA5AG3B00121"', "manus_expected_tracker_id: null")
    assert "manus_expected_tracker_id: null" in document
    preset = _write(tmp_path, document)

    args, _ = _parse("--config", str(preset), "--task", TASK)

    with pytest.raises(SystemExit, match="requires --manus-expected-tracker-id.*serial"):
        _validate_manus_selection(args, _resolve_tracking_selection(args))


def test_hts_preset_selects_quest_for_both_components():
    args, config = _parse("--config", "hts", "--task", TASK)

    assert config is not None and config.name == "hts"
    selection = _resolve_tracking_selection(args)
    assert (selection.hand_source, selection.wrist_source) == ("quest", "quest")
    assert (args.host, args.port, args.protocol) == ("0.0.0.0", 9000, "udp")
    assert args.manus_mode == "integrated"
    _validate_manus_selection(args, selection)


def test_no_config_restores_the_built_in_defaults():
    args, config = _parse("--no-config", "--task", TASK)

    assert config is None
    assert args.hand_source is None and args.wrist_source is None
    selection = _resolve_tracking_selection(args)
    assert (selection.hand_source, selection.wrist_source) == ("quest", "quest")
    assert args.record_hand_poses is False
    assert args.manus_mode == "integrated"


def test_command_line_flags_override_preset_values(tmp_path):
    preset = _write(
        tmp_path,
        "hand_source: manus\n"
        "wrist_source: manus\n"
        "manus_mode: remote\n"
        "port: 9100\n"
        "record_hand_source: [quest]\n",
    )

    args, config = _parse(
        "--config",
        str(preset),
        "--wrist-source",
        "quest",
        "--manus-mode",
        "integrated",
        "--record-hand-source",
        "manus",
        "--task",
        TASK,
    )

    assert config is not None and config.path == preset.resolve()
    assert args.hand_source == "manus"
    assert args.wrist_source == "quest"
    assert args.manus_mode == "integrated"
    assert args.port == 9100
    # An explicit command-line list replaces the preset list instead of appending.
    assert args.record_hand_source == ["manus"]


def test_preset_lists_are_kept_when_the_command_line_does_not_repeat_them(tmp_path):
    preset = _write(tmp_path, "record_hand_source: [quest]\nrecord_wrist_source: manus\n")

    args, _ = _parse("--config", str(preset), "--task", TASK)

    assert args.record_hand_source == ["quest"]
    assert args.record_wrist_source == ["manus"]


def test_null_keeps_the_built_in_default(tmp_path):
    preset = _write(tmp_path, "hand_source: manus\nmanus_mode: null\nmanus_core_host: ~\n")

    args, config = _parse("--config", str(preset), "--task", TASK)

    assert config is not None and "manus_mode" not in config.values
    assert args.hand_source == "manus"
    assert args.manus_mode == "integrated"
    assert args.manus_core_host is None


def test_dashed_keys_booleans_and_typed_values_follow_the_command_line_contract(tmp_path):
    preset = _write(
        tmp_path,
        "hand-source: manus\n"
        "assisted-grasp: true\n"
        "manus-tracker-diagnostics: false\n"
        "manus-expected-tracker-user-id: 3\n"
        "assisted-grasp-break-force: none\n"
        "assisted-grasp-break-torque: 45\n"
        "emg-notch: 60\n"
        "manus-expected-tracker-id: 12345\n",
    )

    args, _ = _parse("--config", str(preset), "--task", TASK)

    assert args.hand_source == "manus"
    assert args.assisted_grasp is True
    assert args.manus_tracker_diagnostics is False
    assert args.manus_expected_tracker_user_id == 3
    assert args.assisted_grasp_break_force is None
    assert args.assisted_grasp_break_torque == 45.0
    assert args.emg_notch == "60"
    assert args.manus_expected_tracker_id == "12345"


@pytest.mark.parametrize(
    ("document", "message"),
    (
        ("hand_source: vive\n", "must be one of quest, manus"),
        ("hand_source: hts\n", "must be one of quest, manus"),
        ("source: hts\n", "unknown launch_og.py option 'source'"),
        ("hand_sorce: manus\n", "unknown launch_og.py option 'hand_sorce'; did you mean 'hand_source'"),
        ("assisted_grasp: yes please\n", "expected true or false"),
        ("auto_anchor: true\n", "unknown launch_og.py option 'auto_anchor'"),
        ("manus_hand_motion: tracker\n", "unknown launch_og.py option 'manus_hand_motion'"),
        ("port: eighty\n", "invalid value 'eighty'"),
        ("port: true\n", "not true/false"),
        ("hand_source: [manus]\n", "single scalar value"),
        ("config: hts\n", "cannot select another launch config"),
        ("no_config: true\n", "cannot select another launch config"),
        ("hand_source: manus\nhand-source: quest\n", "set more than once"),
        (f"task: {TASK}\nsubscale: grasp\n", "--task and --subscale cannot both be set"),
        ("- hand_source\n", "must be a mapping"),
        ("hand_source: [\n", "not valid YAML"),
    ),
)
def test_invalid_presets_name_the_offending_option(tmp_path, document, message):
    preset = _write(tmp_path, document)

    with pytest.raises(SystemExit, match=message):
        load_launch_config(preset, _parser(AratTaskCatalog()))


def test_preset_names_resolve_inside_configs_and_paths_resolve_anywhere(tmp_path):
    assert resolve_launch_config_path("hts") == (LAUNCH_CONFIG_ROOT / "hts.yaml").resolve()
    explicit = _write(tmp_path, "hand_source: quest\n")
    assert resolve_launch_config_path(str(explicit)) == explicit.resolve()

    with pytest.raises(SystemExit, match="Unknown launch config preset 'nope'.*manus_core"):
        resolve_launch_config_path("nope")
    with pytest.raises(SystemExit, match="not found"):
        resolve_launch_config_path(str(tmp_path / "missing.yaml"))
    with pytest.raises(SystemExit, match="requires a preset name"):
        resolve_launch_config_path("   ")


def test_no_config_cannot_be_combined_with_config():
    with pytest.raises(SystemExit):
        _parse("--no-config", "--config", "hts", "--task", TASK)


def test_main_reports_the_default_preset_and_passes_validation(monkeypatch, capsys):
    class _Stop(Exception):
        pass

    def _stop(_tasks):
        raise _Stop()

    monkeypatch.delenv("OMNIGIBSON_DATA_PATH", raising=False)
    # Stop at asset validation: the shipped preset must clear option validation
    # (tracker serial filled in, Linux feed selected) without touching hardware.
    monkeypatch.setattr(launcher, "validate_runtime_assets", _stop)

    with pytest.raises(_Stop):
        main(["--task", TASK])

    output = capsys.readouterr().out
    assert "Launch config:" in output
    assert "manus_core.yaml" in output


def test_legacy_source_flag_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        main(["--task", TASK, "--source", "hts"])


def test_view_only_and_list_tasks_do_not_apply_the_preset():
    for flag in launcher.PRESET_SKIP_FLAGS:
        args, config = parse_launch_arguments(
            _parser(AratTaskCatalog()),
            ["--task", TASK, flag],
            skip_for_flags=launcher.PRESET_SKIP_FLAGS,
        )
        assert config is None
        assert args.hand_source is None
        assert args.record_hand_poses is False

    # Explicit command-line options still parse in those modes.
    args, _ = parse_launch_arguments(
        _parser(AratTaskCatalog()),
        ["--task", TASK, "--view-only", "--camera-rig", "arat_default"],
        skip_for_flags=launcher.PRESET_SKIP_FLAGS,
    )
    assert args.camera_rig == "arat_default"


def test_main_view_only_runs_without_the_preset_core_fields(monkeypatch, capsys):
    class _Stop(Exception):
        pass

    def _stop(_tasks):
        raise _Stop()

    monkeypatch.delenv("OMNIGIBSON_DATA_PATH", raising=False)
    monkeypatch.setattr(launcher, "validate_runtime_assets", _stop)

    # Reaching asset validation means neither the preset's recording flags nor
    # the MANUS wrist policy blocked a run that constructs no tracking source.
    with pytest.raises(_Stop):
        main(["--task", TASK, "--view-only"])

    assert "not applied with --view-only" in capsys.readouterr().out
