"""Wrist bench: IK-controlled arm + Sharpa over the hand-bench table, arm selection, presets, telemetry (no simulator)."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from dex_teleop.arat.camera_rig import camera_rig_names, layout_camera_ids, load_camera_rig
from dex_teleop.arat.reset_poses import RESET_POSES as ARAT_RESET_POSES
from dex_teleop.arat.reset_poses import reset_joint_positions as arat_reset_joint_positions
from dex_teleop.hand_bench import HAND_BENCH_PATH, TABLE_NAME, load_hand_bench_scene, load_wrist_bench_arms
from dex_teleop.hand_bench import wrist_bench
from dex_teleop.hands import SHARPA_ACTION_JOINTS
from dex_teleop.runtime import MultiSourceTrackingWorker
from dex_teleop.tracking import HTSSource
from dex_teleop.tracking.vive_ultimate import DEV_ROTATION_ONLY, DEV_SETUP_REQUIRED, STATE_SYNCING
from dex_teleop.types import Handedness

from fake_vut_server import FakeVutSdkServer, FakeVutServer


def _parse(*argv):
    return wrist_bench.parse_launch_arguments(
        wrist_bench._parser(), list(argv), skip_for_flags=wrist_bench.PRESET_SKIP_FLAGS
    )


def _write_arms(tmp_path, mutate):
    document = yaml.safe_load(HAND_BENCH_PATH.read_text(encoding="utf-8"))
    mutate(document["wrist_bench"])
    path = tmp_path / "hand_bench.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Arm selection (hand_bench.yaml wrist_bench)


def test_yaml_defines_the_franka_and_tianji_arms_with_franka_as_default():
    arms = load_wrist_bench_arms()

    assert arms.default_arm == "franka" and arms.names == ("franka", "tianji")
    assert wrist_bench.ARMS.names == arms.names and wrist_bench.DEFAULT_ARM == "franka"
    for key in arms.names:
        arm = arms.arm(key)
        assert arm.key == key and arm.palm_link == "right_hand_C_MC"
        assert arm.camera_rig in camera_rig_names()
        assert load_camera_rig(arm.camera_rig).layout(wrist_bench.CAMERA_LAYOUT)
        assert arm.default_reset_pose in arm.reset_poses
        assert all(len(pose) == 7 for pose in arm.reset_poses.values())
        assert len(arm.reset_joint_positions()) == 7 + len(SHARPA_ACTION_JOINTS)
        # Both arms stand on the floor behind the table, facing it along +X.
        assert arm.robot.position[2] == 0.0 and arm.robot.position[0] < -0.245
        assert list(arm.robot.orientation_xyzw) == [0.0, 0.0, 0.0, 1.0]
        assert arm.robot.prim_path == f"/controllable__{arm.robot.composed_model}__{arm.robot.name}"
        assert arm.workspace.minimum_z > 0.75 and arm.workspace.maximum_z > arm.workspace.minimum_z
        assert arm.workspace.shoulder_position[2] > 1.0
    assert arms.arm() is arms.arm("franka")
    with pytest.raises(ValueError, match="Unknown wrist bench arm 'ur5'"):
        arms.arm("ur5")
    with pytest.raises(ValueError, match="no reset pose 'compact'"):
        arms.arm("tianji").reset_joint_positions("compact")


def test_franka_arm_is_the_arat_robot_with_the_arat_reset_poses():
    arm = load_wrist_bench_arms().arm("franka")

    assert arm.robot.model == "franka" and arm.robot.end_effector == "sharpa_right"
    assert arm.robot.composed_model == "franka_sharpa_right"
    assert arm.robot.name == "wrist_bench_sharpa_right" and arm.robot.arm_name == "0"
    assert (arm.robot.arm_controller, arm.robot.gripper_controller) == ("arm_0", "gripper_0")
    assert arm.robot.position == (-0.5, 0.0, 0.0)
    assert arm.default_reset_pose == "extended"
    # The YAML copies must stay the ARAT presets launch_og.py uses.
    assert set(arm.reset_poses) == set(ARAT_RESET_POSES)
    for name, pose in arm.reset_poses.items():
        assert arm.reset_joint_positions(name) == pytest.approx(arat_reset_joint_positions(name))
    # SharpaAdapterConfig defaults, so the Franka bench behaves exactly as before.
    assert arm.workspace.shoulder_position == (0.0, 0.0, 1.194)
    assert (arm.workspace.shoulder_reach, arm.workspace.minimum_z, arm.workspace.maximum_z) == (0.8, 0.82, 1.75)
    assert arm.camera_rig == "wrist_bench"


def test_tianji_arm_carries_the_hand_on_the_marvin_m6_right_flange():
    arm = load_wrist_bench_arms().arm("tianji")

    assert arm.robot.model == "tianji_m6_ccs_sharpa_right" and arm.robot.end_effector is None
    assert arm.robot.composed_model == arm.robot.model  # no end-effector variants: the prim is named after the model
    assert arm.robot.arm_name == "right"
    assert (arm.robot.arm_controller, arm.robot.gripper_controller) == ("arm_right", "gripper_right")
    assert "end_effector" not in arm.robot.environment_config()
    # The 0.754 m base plate must clear the table's rear legs (x >= -0.225).
    assert arm.robot.position[0] + 0.377 < -0.225
    assert set(arm.reset_poses) == {"extended", "home", "rest"} and arm.default_reset_pose == "extended"
    assert arm.reset_poses["home"] == pytest.approx([-1.7, -1.1, 1.1, -2.0, 0.37, 0.13, -0.55])
    # Right shoulder of the Marvin Pro: 0.2 m to the right of the torso, 1.2 m above the floor;
    # the reach sphere stays inside the 809 mm J2-to-flange reach (M6-S CCS-809) plus the hand offset.
    assert arm.workspace.shoulder_position == pytest.approx((0.0, -0.2005, 1.203))
    assert 0.5 < arm.workspace.shoulder_reach < 0.809 + 0.03
    assert arm.camera_rig == "wrist_bench_tianji"


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda section: section.__setitem__("arm", "ur5"), "wrist_bench.arm 'ur5' is not one of"),
        (lambda section: section["arms"]["tianji"].__setitem__("default_reset_pose", "compact"), "default_reset_pose 'compact'"),
        (lambda section: section["arms"]["tianji"]["reset_poses"].__setitem__("home", [0.0] * 6), "must contain 7 finite numbers"),
        (lambda section: section["arms"]["tianji"]["robot"].__setitem__("name", "wrist_bench_sharpa_right"), "used by another arm"),
        (lambda section: section["arms"]["franka"]["workspace"].__setitem__("maximum_z", 0.5), "maximum_z must exceed"),
        (lambda section: section["arms"]["franka"].pop("palm_link"), "palm_link must be a non-empty string"),
        (lambda section: section["arms"]["franka"]["robot"].pop("composed_model"), "composed_model must be a non-empty string"),
    ),
)
def test_arm_yaml_validation_rejects_inconsistent_definitions(tmp_path, mutate, message):
    path = _write_arms(tmp_path, mutate)
    with pytest.raises(ValueError, match=message):
        load_wrist_bench_arms(path)


# ---------------------------------------------------------------------------
# Environment configuration


def test_environment_config_drives_the_arm_with_the_arat_ik_controller():
    scene = load_hand_bench_scene()
    arm = wrist_bench.ARMS.arm("franka")
    config = wrist_bench.build_wrist_bench_config(scene, viewport_name=None)
    assert config == wrist_bench.build_wrist_bench_config(scene, arm, viewport_name=None)  # the default arm

    robot = config["robots"][0]
    assert robot["name"] == arm.robot.name == "wrist_bench_sharpa_right"
    assert (robot["model"], robot["end_effector"], robot["dataset_name"]) == ("franka", "sharpa_right", "omnigibson-robot-assets")
    assert robot["controller_config"]["arm_0"]["name"] == "InverseKinematicsController"
    assert robot["controller_config"]["arm_0"]["mode"] == "absolute_pose"
    assert robot["controller_config"]["gripper_0"]["mode"] == "independent"
    assert robot["reset_joint_pos"] == arat_reset_joint_positions("extended")
    assert len(robot["reset_joint_pos"]) == 7 + len(SHARPA_ACTION_JOINTS)
    assert robot["fixed_base"] is True and robot["grasping_mode"] == "physical"
    # Behind the table centre, facing it along +X, standing on the floor.
    assert list(robot["position"]) == list(arm.robot.position) == [-0.5, 0.0, 0.0]
    assert robot["position"][0] < -scene.table.tabletop_height / 2 and robot["position"][2] == 0.0
    assert list(robot["orientation"]) == [0.0, 0.0, 0.0, 1.0]
    assert config["task"] == {"type": "DummyTask"}
    assert config["scene"]["use_skybox"] is False
    objects = {obj["name"]: obj for obj in config["objects"]}
    assert objects[TABLE_NAME]["model"] == scene.table.model and objects[TABLE_NAME]["fixed_base"] is True
    for light in scene.lights:
        assert objects[light["name"]]["type"] == "LightObject"
    sensors = {sensor["name"]: sensor for sensor in config["env"]["external_sensors"]}
    assert set(sensors) == {
        "wrist_bench_overview_camera",
        "wrist_bench_side_camera",
        "wrist_bench_front_camera",
    }
    assert all(sensor["sensor_kwargs"]["viewport_name"] is None for sensor in sensors.values())
    compact = wrist_bench.build_wrist_bench_config(scene, reset_pose="compact")
    assert compact["robots"][0]["reset_joint_pos"] == arat_reset_joint_positions("compact")
    assert arm.robot.prim_path.endswith(f"__{arm.robot.name}")


def test_environment_config_for_the_tianji_arm_uses_its_controllers_placement_and_rig():
    scene = load_hand_bench_scene()
    arm = wrist_bench.ARMS.arm("tianji")
    config = wrist_bench.build_wrist_bench_config(scene, arm, viewport_name=None)

    robot = config["robots"][0]
    assert robot["model"] == "tianji_m6_ccs_sharpa_right" and "end_effector" not in robot
    assert robot["name"] == "wrist_bench_tianji_sharpa_right"
    assert set(robot["controller_config"]) == {"arm_right", "gripper_right"}
    assert robot["controller_config"]["arm_right"]["name"] == "InverseKinematicsController"
    assert robot["controller_config"]["gripper_right"]["name"] == "MultiFingerGripperController"
    assert robot["reset_joint_pos"] == arm.reset_joint_positions("extended")
    assert robot["reset_joint_pos"][7:] == [0.0] * len(SHARPA_ACTION_JOINTS)
    assert list(robot["position"]) == [-0.63, 0.0, 0.0]
    # Everything that is not the arm is shared with the Franka bench.
    franka = wrist_bench.build_wrist_bench_config(scene, viewport_name=None)
    assert config["objects"] == franka["objects"] and config["scene"] == franka["scene"]
    sensors = {sensor["name"]: sensor for sensor in config["env"]["external_sensors"]}
    assert set(sensors) == {sensor["name"] for sensor in franka["env"]["external_sensors"]}
    # Same sensor names, the Tianji rig's own placements.
    franka_sensors = {sensor["name"]: sensor for sensor in franka["env"]["external_sensors"]}
    assert sensors["wrist_bench_side_camera"]["position"] != franka_sensors["wrist_bench_side_camera"]["position"]
    home = wrist_bench.build_wrist_bench_config(scene, arm, reset_pose="home", viewport_name=None)
    assert home["robots"][0]["reset_joint_pos"][:7] == pytest.approx([-1.7, -1.1, 1.1, -2.0, 0.37, 0.13, -0.55])
    with pytest.raises(ValueError, match="no reset pose 'compact'"):
        wrist_bench.build_wrist_bench_config(scene, arm, reset_pose="compact", viewport_name=None)


@pytest.mark.parametrize("arm_key", ("franka", "tianji"))
def test_wrist_bench_camera_rigs_frame_the_hand_from_three_sides(arm_key):
    arm = wrist_bench.ARMS.arm(arm_key)
    assert arm.camera_rig in camera_rig_names()
    rig = load_camera_rig(arm.camera_rig)
    layout = rig.layout(wrist_bench.CAMERA_LAYOUT)

    assert layout["viewports"]["main"]["camera"] == "overview"
    assert layout["toggles"] == [{"key": "B", "viewport": "main", "cameras": ["overview", "side", "front"]}]
    assert layout_camera_ids(rig, wrist_bench.CAMERA_LAYOUT) == ("overview", "side", "front")
    assert layout["workspace"]["viewport_only"] is True
    # Both rigs aim near the hand's reset position above the table.
    aim = np.array([0.0, 0.0, 0.95]) if arm_key == "franka" else np.array([-0.10, -0.10, 1.00])
    for camera_id in ("overview", "side", "front"):
        camera = rig.camera(camera_id)
        assert camera["parent"] == {"frame": "scene"}
        position = np.asarray(camera["pose"]["position"])
        assert position[2] > 1.4 and np.linalg.norm(position[:2]) > 1.5
        # USD cameras look along their local -Z; every camera must look at the hand.
        forward = Rotation.from_quat(camera["pose"]["orientation_xyzw"]).apply([0.0, 0.0, -1.0])
        to_aim = aim - position
        assert float(np.dot(forward, to_aim / np.linalg.norm(to_aim))) > 0.98
    assert rig.camera("overview")["pose"]["position"][1] < 0
    assert rig.camera("front")["pose"]["position"][0] > 1.0 > rig.camera("overview")["pose"]["position"][0]
    if arm_key == "franka":
        # The side view comes from the robot's left (+Y).
        assert rig.camera("side")["pose"]["position"][1] > 1.5
    else:
        # The Tianji's welded left arm hangs on +Y, so its side view comes from the right (-Y).
        assert rig.camera("side")["pose"]["position"][1] < -1.5


def test_default_preset_selects_the_glove_and_the_ultimate_tracker():
    args, config = _parse()

    assert config is not None and config.name == "manus_core"
    selection = wrist_bench._resolve_tracking_selection(args)
    assert (selection.hand_source, selection.wrist_source) == ("manus", "manus")
    assert args.manus_tracker_feed == "vive_ultimate" and args.manus_mode == "integrated"
    assert args.manus_expected_tracker_id == "FA5AG3B00121"
    assert args.record_hand_poses is True  # accepted from the preset, reported as ignored
    assert [key for key in wrist_bench.IGNORED_PRESET_OPTIONS if key in config.values] == ["record_hand_poses"]
    assert args.arm == "franka" and args.camera_rig is None and args.reset_pose is None and args.arm_markers is True
    wrist_bench._validate_args(args, selection)
    arm = wrist_bench._selected_arm(args)
    assert arm.key == "franka"
    assert wrist_bench._camera_rig_name(args, arm) == "wrist_bench"
    assert wrist_bench._reset_pose_name(args, arm) == "extended"

    explicit, _ = _parse("--position-sensitivity", "1", "--no-arm-markers", "--telemetry-interval", "0.5")
    assert explicit.position_sensitivity == 1.0 and explicit.arm_markers is False
    assert explicit.telemetry_interval == 0.5


def test_arm_option_selects_the_tianji_robot_and_its_defaults():
    args, config = _parse("--arm", "tianji")
    assert config is not None and config.name == "manus_core"  # the same tracking preset applies to both arms
    wrist_bench._validate_args(args, wrist_bench._resolve_tracking_selection(args))
    arm = wrist_bench._selected_arm(args)
    assert arm.key == "tianji" and arm.robot.model == "tianji_m6_ccs_sharpa_right"
    assert wrist_bench._camera_rig_name(args, arm) == "wrist_bench_tianji"
    assert wrist_bench._reset_pose_name(args, arm) == "extended"

    home, _ = _parse("--arm", "tianji", "--reset-pose", "home", "--camera-rig", "wrist_bench")
    wrist_bench._validate_args(home, wrist_bench._resolve_tracking_selection(home))
    assert wrist_bench._reset_pose_name(home, wrist_bench._selected_arm(home)) == "home"
    assert wrist_bench._camera_rig_name(home, wrist_bench._selected_arm(home)) == "wrist_bench"

    view_only, _ = _parse("--arm", "tianji", "--view-only")
    wrist_bench._validate_args(view_only, wrist_bench._resolve_tracking_selection(view_only))
    with pytest.raises(SystemExit) as rejected:  # argparse: invalid choice 'ur5'
        _parse("--arm", "ur5")
    assert rejected.value.code == 2


def test_hts_preset_and_view_only_parse():
    args, config = _parse("--config", "hts")
    assert config is not None and config.name == "hts"
    selection = wrist_bench._resolve_tracking_selection(args)
    assert (selection.hand_source, selection.wrist_source) == ("quest", "quest")
    wrist_bench._validate_args(args, selection)

    view_only, config = _parse("--view-only")
    assert config is None and view_only.view_only is True
    wrist_bench._validate_args(view_only, wrist_bench._resolve_tracking_selection(view_only))


@pytest.mark.parametrize(
    ("argv", "message"),
    (
        (["--hand", "left"], "right-hand Sharpa"),
        (["--record-hand-source", "quest"], "records nothing"),
        (["--telemetry-interval", "-1"], "must be finite and positive"),
        (["--position-sensitivity", "-1"], "must be finite and positive"),
        (["--no-config", "--hand-source", "manus", "--wrist-source", "manus"], "needs a tracker"),
        (["--no-config", "--hand-source", "manus", "--wrist-source", "manus", "--manus-tracker-feed", "vive_ultimate"], "requires --manus-expected-tracker-id"),
        (["--engage-after", "-1"], "engage-after"),
        (["--view-only", "--engage-after", "2"], "drop --view-only"),
        (["--reset-pose", "home"], "not a reset pose of arm 'franka'; choose one of: compact, extended"),
        (["--arm", "tianji", "--reset-pose", "compact"], "not a reset pose of arm 'tianji'; choose one of: extended, home"),
    ),
)
def test_cli_rejects_unsupported_or_invalid_options(argv, message):
    args, _ = _parse(*argv)
    with pytest.raises(SystemExit, match=message):
        wrist_bench._validate_args(args, wrist_bench._resolve_tracking_selection(args))


def test_engage_after_is_optional_and_headless_friendly():
    args, _ = _parse("--engage-after", "2.5", "--steps", "300")
    wrist_bench._validate_args(args, wrist_bench._resolve_tracking_selection(args))
    assert args.engage_after == 2.5 and args.steps == 300
    default, _ = _parse()
    assert default.engage_after is None


def test_create_worker_is_the_arat_multi_source_worker():
    args, _ = _parse("--config", "hts", "--port", "9100")
    selection = wrist_bench._resolve_tracking_selection(args)

    worker = wrist_bench.create_worker(args, selection, None)

    assert isinstance(worker, MultiSourceTrackingWorker)
    assert worker.articulation_source == "quest" and worker.wrist_source == "quest"
    assert isinstance(worker.sources["quest"], HTSSource)
    assert worker.handedness == Handedness.RIGHT
    assert set(worker.retargeter.joint_names) == set(SHARPA_ACTION_JOINTS)


def test_bench_reexports_the_shared_telemetry():
    from dex_teleop.omnigibson import teleop_telemetry

    assert wrist_bench.WristTelemetry is teleop_telemetry.WristTelemetry
    assert wrist_bench.target_position_world is teleop_telemetry.target_position_world
    args, _ = _parse("--telemetry-interval", "0")
    wrist_bench._validate_args(args, wrist_bench._resolve_tracking_selection(args))  # 0 disables the lines


def test_preflight_passes_on_a_tracking_ultimate_from_either_daemon(tmp_path):
    sdk_socket = str(tmp_path / "sdk.sock")
    sdk = FakeVutSdkServer(sdk_socket, serial="FA5AG3B00121", rate_hz=200.0, extra_trackers={4: "OTHER"}).start()
    try:
        summary = wrist_bench.vive_ultimate_preflight(sdk_socket, "FA5AG3B00121", seconds=2.0)
    finally:
        sdk.close()
    assert summary.startswith("VIVE Ultimate feed OK (seqpacket protocol")
    assert "FA5AG3B00121 ready/trackable" in summary and "also streaming: OTHER" in summary

    stream_socket = str(tmp_path / "stream.sock")
    stream = FakeVutServer(stream_socket, serial="59-A33R03250", rate_hz=200.0).start()
    try:
        summary = wrist_bench.vive_ultimate_preflight(stream_socket, "59-A33R03250", seconds=2.0)
        anonymous = wrist_bench.vive_ultimate_preflight(stream_socket, None, seconds=2.0)
    finally:
        stream.close()
    assert "stream protocol" in summary and "59-A33R03250 ready/trackable" in summary
    assert "59-A33R03250" in anonymous


def test_preflight_names_the_missing_daemon(tmp_path):
    with pytest.raises(SystemExit, match="not listening on .*run-vivehub.sh"):
        wrist_bench.vive_ultimate_preflight(str(tmp_path / "absent.sock"), "FA5AG3B00121", seconds=0.3)


def test_preflight_explains_a_listed_but_silent_tracker(tmp_path):
    # VIVE Hub 1.0.3 keeps listing an Ultimate that went to standby as "ready" while no poses arrive.
    socket_path = str(tmp_path / "vut.sock")
    server = FakeVutSdkServer(socket_path, serial="FA5AG3B00121", rate_hz=200.0, state=DEV_SETUP_REQUIRED).start()
    try:
        with pytest.raises(SystemExit, match="needs map setup .*Start setup"):
            wrist_bench.vive_ultimate_preflight(socket_path, "FA5AG3B00121", seconds=0.5)
        with pytest.raises(SystemExit, match="not streaming from VIVE Hub; seen: FA5AG3B00121 \\(build_map\\)"):
            wrist_bench.vive_ultimate_preflight(socket_path, "SOME-OTHER", seconds=0.5)
    finally:
        server.close()

    # Exactly what a hardware Ultimate in standby looks like: still "ready" in the device list, no poses.
    silent = FakeVutSdkServer(str(tmp_path / "silent.sock"), serial="FA5AG3B00121", rate_hz=200.0, streaming=False).start()
    try:
        with pytest.raises(SystemExit, match="listed by VIVE Hub \\(state ready\\) but streams no poses.*standby.*run-vivehub.sh"):
            wrist_bench.vive_ultimate_preflight(str(tmp_path / "silent.sock"), "FA5AG3B00121", seconds=0.5)
    finally:
        silent.close()


def test_preflight_rejects_degraded_tracking_and_wrong_serials(tmp_path):
    socket_path = str(tmp_path / "vut.sock")
    server = FakeVutSdkServer(socket_path, serial="FA5AG3B00121", rate_hz=200.0, state=DEV_ROTATION_ONLY).start()
    try:
        with pytest.raises(SystemExit, match="streaming but bad: rotation-only"):
            wrist_bench.vive_ultimate_preflight(socket_path, "FA5AG3B00121", seconds=0.5)
        with pytest.raises(SystemExit, match="not streaming from VIVE Hub; seen: FA5AG3B00121\\. Set manus_expected_tracker_id"):
            wrist_bench.vive_ultimate_preflight(socket_path, "59-A33R03250", seconds=0.5)
    finally:
        server.close()

    stream_socket = str(tmp_path / "stream.sock")
    syncing = FakeVutServer(stream_socket, serial="59-A33R03250", rate_hz=200.0, state=STATE_SYNCING).start()
    try:
        with pytest.raises(SystemExit, match="streaming but untrackable: state syncing"):
            wrist_bench.vive_ultimate_preflight(stream_socket, "59-A33R03250", seconds=0.5)
    finally:
        syncing.close()


@pytest.mark.parametrize(
    ("forward_motion", "expected_yaw"),
    (
        ((0.15, 0.0, 0.02), 0.0),      # map forward already along the robot's +X
        ((-0.15, 0.0, 0.0), 180.0),    # both horizontal axes reversed
        ((0.0, 0.15, 0.0), -90.0),     # forward comes out as the robot's left
        ((0.0, -0.15, 0.0), 90.0),     # forward comes out as the robot's right
        ((0.12, 0.12, 0.0), -45.0),
    ),
)
def test_yaw_calibration_turns_a_forward_motion_into_plus_x(forward_motion, expected_yaw):
    yaw = wrist_bench.yaw_correction_from_forward_motion(forward_motion)
    assert yaw == pytest.approx(expected_yaw)
    # The same quaternion SharpaActionAdapter receives must rotate that motion onto +X.
    rotated = Rotation.from_quat(wrist_bench.tracking_to_world_quaternion(yaw)).apply(forward_motion)
    assert rotated[0] > 0.1 and rotated[1] == pytest.approx(0.0, abs=1e-9)
    assert rotated[2] == pytest.approx(forward_motion[2])


def test_yaw_calibration_rejects_too_little_horizontal_motion():
    with pytest.raises(ValueError, match="at least 10 cm horizontally .*measured 3.0 cm"):
        wrist_bench.yaw_correction_from_forward_motion((0.03, 0.0, 0.5))
    assert wrist_bench.yaw_correction_from_forward_motion((0.03, 0.0, 0.0), minimum_m=0.02) == 0.0

