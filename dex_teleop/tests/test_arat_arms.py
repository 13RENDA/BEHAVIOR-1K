"""ARAT arms (--arm): the unchanged Franka, the Tianji placement and its reach, provenance, replay."""

from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from dex_teleop.arat import AratTaskCatalog
from dex_teleop.arat import placement as pl
from dex_teleop.arat.arms import (
    ARM_PROVENANCE_ATTRIBUTE,
    ARMS_PATH,
    arm_provenance,
    layout_group,
    load_arat_arms,
    recorded_stance,
    recording_arm,
    task_path_reach,
    task_reach,
    write_arm_provenance,
)
from dex_teleop.arat.eval.regions import MANNEQUIN_REGIONS
from dex_teleop.arat.layout import Layout, apply_layout, moved_object_names
from dex_teleop.arat.reset_poses import reset_joint_positions
from dex_teleop.arat.scene import (
    ROBOT_DATASET_NAME,
    ROBOT_END_EFFECTOR,
    ROBOT_MODEL,
    ROBOT_NAME,
    get_task_scene_data,
    get_task_tro_data,
)
from dex_teleop.omnigibson.launch_config import parse_launch_arguments
from dex_teleop.omnigibson.launcher import ROBOT_PRIM_PATH, _parser, _validate_arm_selection, build_environment_config
from dex_teleop.omnigibson.replay import build_replay_camera_configs
from dex_teleop.omnigibson.sharpa_adapter import SharpaAdapterConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROBOT_ASSETS = REPOSITORY_ROOT / "datasets" / "omnigibson-robot-assets"
TASK_INSTANCES = REPOSITORY_ROOT / "datasets" / "arat-task-instances"
TIANJI_DEFINITION = ROBOT_ASSETS / "models" / "tianji_m6_ccs_sharpa_right" / "tianji_m6_ccs_sharpa_right.yaml"
needs_instances = pytest.mark.skipif(not TASK_INSTANCES.is_dir(), reason="ARAT task instances not installed")
needs_tianji = pytest.mark.skipif(not TIANJI_DEFINITION.is_file(), reason="tianji_m6_ccs_sharpa_right asset not built")
needs_pinocchio = pytest.mark.skipif(importlib.util.find_spec("pinocchio") is None, reason="pinocchio not installed")

CATALOG = AratTaskCatalog()
ARMS = load_arat_arms()
TABLE_TASK = CATALOG.tasks["arat_grasp_block_10cm"]
POUR_TASK = CATALOG.tasks["arat_grip_pour_water"]
MANNEQUIN_TASK = CATALOG.tasks["arat_gross_movement_hand_mouth"]
# The Tianji with its 15 box tasks at table_front (the apparatus turned a quarter turn): the arrangement in
# which every task was checked. arms.yaml defines it but has not assigned it since 2026-11-15, when the user
# asked for the apparatus to stay as saved; the tests of that arrangement use this arm.
TURNED_TIANJI = replace(ARMS.arm("tianji"), layouts={**ARMS.arm("tianji").layouts, "table": "table_front"})


def _parse(*argv, default_config=None):
    return parse_launch_arguments(_parser(CATALOG), list(argv), default_config=default_config)


# --------------------------------------------------------------------------- the Franka is unchanged


def test_franka_is_the_default_and_matches_the_arat_constants():
    assert ARMS.default_arm == "franka" and set(ARMS.names) == {"franka", "tianji"}
    franka = ARMS.arm()
    assert franka.stances is None and franka.stance(TABLE_TASK) is None and franka.excluded_reason(TABLE_TASK) is None
    assert franka.operator_yaw_deg(TABLE_TASK) == 0.0
    assert franka.robot_identity() == {
        "model": ROBOT_MODEL,
        "dataset_name": ROBOT_DATASET_NAME,
        "name": ROBOT_NAME,
        "end_effector": ROBOT_END_EFFECTOR,
    }
    assert franka.prim_path == ROBOT_PRIM_PATH
    assert (franka.arm_controller, franka.gripper_controller) == ("arm_0", "gripper_0")
    for task in (TABLE_TASK, MANNEQUIN_TASK):
        assert franka.reset_pose_names(task) == ("compact", "extended")
        for name in ("compact", "extended"):
            assert franka.reset_joint_positions(task, name) == reset_joint_positions(name)
    # the adapter clamp the Franka keeps is the one its wrist-bench definition states
    defaults = SharpaAdapterConfig()
    assert franka.workspace == pl.Workspace(
        defaults.shoulder_position, defaults.shoulder_reach, defaults.minimum_z, defaults.maximum_z
    )
    assert task_reach(franka, TABLE_TASK) is None


@needs_instances
@pytest.mark.parametrize("reset_pose", ["extended", "compact"])
def test_default_environment_config_is_the_franka_at_the_saved_pose(reset_pose):
    config = build_environment_config(TABLE_TASK, assisted_grasp=True, reset_pose=reset_pose)
    assert config == build_environment_config(TABLE_TASK, assisted_grasp=True, reset_pose=reset_pose, arm=ARMS.arm("franka"))
    (robot,) = config["robots"]
    saved = get_task_tro_data(TABLE_TASK)["robot_poses"]["robot"][0]
    assert (robot["model"], robot["end_effector"], robot["name"], robot["dataset_name"]) == (
        "franka", "sharpa_right", "franka_sharpa_right", "omnigibson-robot-assets",
    )
    assert robot["position"] == list(saved["position"]) and robot["orientation"] == list(saved["orientation"])
    assert robot["reset_joint_pos"] == reset_joint_positions(reset_pose)
    assert set(robot["controller_config"]) == {"arm_0", "gripper_0"}
    assert robot["grasping_mode"] == "assisted" and robot["disable_grasp_handling"] is True
    metadata = config["scene"]["scene_file"]["metadata"]["task"]
    assert metadata["inst_to_name"]["agent.n.01_1"] == "franka_sharpa_right"
    assert metadata["robot_poses"] == get_task_tro_data(TABLE_TASK)["robot_poses"]
    wrist = [s for s in config["env"]["external_sensors"] if "wrist" in s["name"]]
    assert wrist and all(ROBOT_PRIM_PATH in s["relative_prim_path"] or ROBOT_PRIM_PATH in json.dumps(s) for s in wrist)


def test_arm_option_and_preset_key(tmp_path):
    args, _ = _parse("--task", TABLE_TASK.activity)
    assert args.arm == "franka" and args.allow_unreachable is False
    assert _parse("--task", TABLE_TASK.activity, "--arm", "tianji")[0].arm == "tianji"
    preset = tmp_path / "preset.yaml"
    preset.write_text("arm: tianji\n", encoding="utf-8")
    assert _parse("--config", str(preset), "--task", TABLE_TASK.activity)[0].arm == "tianji"
    with pytest.raises(SystemExit):
        _parse("--task", TABLE_TASK.activity, "--arm", "ur5")


# --------------------------------------------------------------------------- the Tianji


def test_tianji_reuses_the_wrist_bench_arm_and_has_a_stance_for_every_task():
    tianji = ARMS.arm("tianji")
    bench = tianji.bench
    assert bench.key == "tianji" and bench.robot.model == "tianji_m6_ccs_sharpa_right"
    assert tianji.robot_name != ARMS.arm("franka").robot_name
    assert tianji.prim_path == f"/controllable__{bench.robot.composed_model}__{tianji.robot_name}"
    assert (tianji.arm_controller, tianji.gripper_controller) == ("arm_right", "gripper_right")
    assert tianji.workspace.shoulder_reach == bench.workspace.shoulder_reach == pytest.approx(0.773)  # the 809 mm arm
    assert set(tianji.stances) == {
        "table_front_saved", "table_front", "table_front_cups", "table_side", "mannequin_side", "mannequin_front"
    }
    assert layout_group(TABLE_TASK) == "table" and layout_group(MANNEQUIN_TASK) == "mannequin"
    stances = {task.activity: tianji.stance(task).name for task in CATALOG.tasks.values()}
    assert stances.pop("arat_grip_pour_water") == "table_front_cups"
    assert stances.pop("arat_gross_movement_hand_mouth") == "mannequin_front"
    assert stances.pop("arat_gross_movement_hand_top_head") == stances.pop("arat_gross_movement_hand_behind_head") == "mannequin_side"
    # every other table task from the front with the apparatus as saved; the turned arrangement and the
    # side stance are defined but unassigned
    assert set(stances.values()) == {"table_front_saved"} and len(stances) == 15
    front = tianji.stances["table_front"]
    saved_front = tianji.stances["table_front_saved"].base
    assert saved_front == pl.BasePose((-0.52, 0.1, 0.0), 15.0)  # 9 cm from the cups' place, turned toward the box
    assert front.base == pl.BasePose((-0.57, 0.173, 0.0), 10.0)
    # 4 cm behind it since 2026-11-18: the water starts in the red cup on the right
    assert tianji.stances["table_front_cups"].base == pl.BasePose((-0.61, 0.173, 0.0), 10.0)
    assert tianji.layout(TABLE_TASK) is None and tianji.stances["table_front_saved"].layout is None
    assert front.layout == Layout(yaw_deg=-90.0, about=(0.2415, -0.1085), translation=(0.0, 0.0))
    assert TURNED_TIANJI.stance(TABLE_TASK) is front and TURNED_TIANJI.layout(TABLE_TASK) == front.layout
    assert tianji.layout(POUR_TASK) is None and tianji.layout(MANNEQUIN_TASK) is None
    assert ARMS.arm("franka").layout(TABLE_TASK) is None
    assert tianji.excluded_tasks == {}
    for task in CATALOG.tasks.values():
        # the table stances in use also have the hand-on-the-table start pose
        expected = ("extended",) if layout_group(task) == "mannequin" else ("extended", "rest")
        assert tianji.reset_pose_names(task) == expected
        joints = tianji.reset_joint_positions(task)
        assert len(joints) == 7 + 22 and joints[7:] == [0.0] * 22
        assert tianji.operator_yaw_deg(task) == tianji.base_pose(task).yaw_deg
    with pytest.raises(ValueError, match="no reset pose 'compact'"):
        tianji.reset_joint_positions(TABLE_TASK, "compact")


@needs_instances
def test_tianji_environment_config_stands_at_its_own_pose():
    tianji = ARMS.arm("tianji")
    for task in (TABLE_TASK, POUR_TASK, MANNEQUIN_TASK, CATALOG.tasks["arat_gross_movement_hand_top_head"]):
        config = build_environment_config(task, arm=tianji)
        (robot,) = config["robots"]
        pose = tianji.base_pose(task)
        assert robot["model"] == "tianji_m6_ccs_sharpa_right" and "end_effector" not in robot
        assert robot["name"] == tianji.robot_name
        assert robot["position"] == list(pose.position)
        assert robot["orientation"] == pytest.approx(list(pose.orientation_xyzw))
        assert robot["reset_joint_pos"] == tianji.reset_joint_positions(task)
        assert set(robot["controller_config"]) == {"arm_right", "gripper_right"}
        metadata = config["scene"]["scene_file"]["metadata"]["task"]
        assert metadata["inst_to_name"]["agent.n.01_1"] == tianji.robot_name
        # BehaviorTask re-applies the presampled pose at reset: it must be the Tianji's, not the Franka's
        assert metadata["robot_poses"] == {
            "robot": [{"position": list(pose.position), "orientation": list(pose.orientation_xyzw)}]
        }
        assert tianji.prim_path in json.dumps(config["env"]["external_sensors"])
        assert ROBOT_PRIM_PATH not in json.dumps(config["env"]["external_sensors"])


@needs_instances
def test_scene_cameras_turn_with_the_stance_and_wrist_cameras_do_not():
    tianji, franka = ARMS.arm("tianji"), ARMS.arm("franka")
    reference = {s["name"]: s for s in build_environment_config(TABLE_TASK, arm=franka)["env"]["external_sensors"]}
    for task in (TABLE_TASK, POUR_TASK, MANNEQUIN_TASK):
        stance = tianji.stance(task)
        turned = {s["name"]: s for s in build_environment_config(task, arm=tianji)["env"]["external_sensors"]}
        assert set(turned) == set(reference)
        turn = Rotation.from_euler("z", stance.base.yaw_deg, degrees=True)
        focus = np.asarray(stance.camera_focus)
        # the cameras turn about the focus as it is in the saved layout and are carried to where the layout puts it
        saved_focus = focus if stance.layout is None else np.asarray(stance.layout.inverse().point(focus))
        pivot = np.array([saved_focus[0], saved_focus[1], 0.0])
        carried = np.array([focus[0], focus[1], 0.0])
        scene_cameras = 0
        for name, before in reference.items():
            after = turned[name]
            if before["pose_frame"] != "scene":
                # on the palm link: same mount, the Tianji's prim
                assert after["position"] == before["position"] and after["orientation"] == before["orientation"]
                continue
            scene_cameras += 1
            if name == tianji.head_camera[0]:
                # on the robot's head, in front of the head box, looking at the work area without roll
                position = np.asarray(after["position"])
                assert stance.base.to_base(position) == pytest.approx(tianji.head_camera[1])
                axes = Rotation.from_quat(after["orientation"]).as_matrix()
                to_focus = (focus - position) / np.linalg.norm(focus - position)
                assert -axes[:, 2] == pytest.approx(to_focus, abs=1e-9)  # the optical axis (-Z) meets the focus
                assert axes[2, 0] == pytest.approx(0.0, abs=1e-9) and axes[2, 1] > 0.5  # +X level, +Y up
                assert 0.3 < np.linalg.norm(focus - position) < 0.9
                continue
            assert after["position"] == pytest.approx(turn.apply(np.asarray(before["position"]) - pivot) + carried)
            # the camera's axes turned with it, so it frames the same spot about the pivot
            expected = (turn * Rotation.from_quat(before["orientation"])).as_matrix()
            assert Rotation.from_quat(after["orientation"]).as_matrix() == pytest.approx(expected, abs=1e-9)
            assert after["position"][2] == pytest.approx(before["position"][2])
            if stance.base.yaw_deg == 0.0 and stance.layout is None:
                assert after["position"] == before["position"] and after["orientation"] == before["orientation"]
        assert scene_cameras == sum(1 for s in reference.values() if s["pose_frame"] == "scene") >= 3
    # nothing moves for the Franka
    sensors = [dict(s) for s in reference.values()]
    franka.place_scene_cameras(TABLE_TASK, sensors)
    assert sensors == list(reference.values())


@needs_instances
def test_key_points_come_from_the_task_instances_and_the_scoring_targets():
    points = {p.name: p for p in pl.task_key_points(TABLE_TASK)}
    assert set(points) == {"start", "shelf"}
    # the 10 cm block starts on the open lid; the shelf is the top of the box shell, 37 cm higher
    assert points["start"].position == pytest.approx((-0.034, -0.282, 1.075), abs=2e-3)
    assert points["shelf"].position[2] - points["start"].position[2] == pytest.approx(0.378, abs=0.01)
    assert points["shelf"].position[0] > points["start"].position[0] + 0.25
    tin = {p.name: p for p in pl.task_key_points(CATALOG.tasks["arat_pinch_marble_index"])}["tin"]
    assert tin.position == pytest.approx((0.292, -0.282, 1.428), abs=2e-3)
    regions = {t: pl.task_key_points(CATALOG.tasks[f"arat_gross_movement_hand_{t}"])[0] for t in ("behind_head", "top_head", "mouth")}
    assert (regions["behind_head"].approach, regions["top_head"].approach, regions["mouth"].approach) == ("back", "top", "front")
    assert regions["behind_head"].position[0] > regions["top_head"].position[0] > regions["mouth"].position[0]
    assert regions["top_head"].position[2] == pytest.approx(1.42, abs=0.02)
    assert all(len(pl.task_key_points(task)) in (1, 2) for task in CATALOG.tasks.values())


@needs_instances
def test_the_assigned_stances_reach_every_task_with_the_809_mm_arm():
    """What is assigned since 2026-11-15: the front of the table with the apparatus as saved."""
    tianji = ARMS.arm("tianji")
    for task in CATALOG.tasks.values():
        reach = task_reach(tianji, task)
        assert reach.conflicts == () and reach.excluded_reason is None
        assert reach.reachable, reach.describe()
        assert all(point.in_height_band for point in reach.points)


@needs_instances
def test_tianji_reaches_every_task_from_its_stance_and_stands_clear():
    tianji = TURNED_TIANJI
    for task in CATALOG.tasks.values():
        reach = task_reach(tianji, task)
        assert reach.stance is tianji.stance(task)
        assert reach.conflicts == (), (task.activity, reach.conflicts)
        assert reach.excluded_reason is None
        assert reach.reachable, reach.describe()
        assert all(point.in_height_band for point in reach.points)
        if layout_group(task) == "mannequin":
            # the palm position of the recorded touch is inside the adapter's reach sphere
            (point,) = reach.points
            assert point.key_point.name == "palm at the touch" and point.palm_margin > 0.0


@needs_instances
def test_the_franka_spot_and_a_square_on_stance_do_not_work_for_the_tianji():
    """Why arms.yaml exists: neither the saved Franka pose nor facing the table squarely reaches the shelf."""

    tianji = ARMS.arm("tianji")
    shelf = next(p for p in pl.task_key_points(TABLE_TASK) if p.name == "shelf")
    saved = get_task_tro_data(TABLE_TASK)["robot_poses"]["robot"][0]["position"]
    at_franka_spot = pl.sphere_reach(shelf, pl.BasePose(tuple(saved[:2]) + (0.0,), 0.0), tianji.workspace)
    assert at_franka_spot.palm_margin < -0.05  # the palm itself stays 8 cm short of the shelf from there
    return  # the rest held for the 696 mm arm; the 809 mm arm reaches the shelf from the front (table_front_saved)
    # square on, as close as the base plate can get to the table legs (0.377 m half plate + 1 cm)
    best_square = max(
        pl.sphere_reach(shelf, pl.BasePose((-0.134 - 0.377 - 0.01, y, 0.0), 0.0), tianji.workspace).grasp_margin
        for y in np.arange(-0.5, 0.5, 0.01)
    )
    assert best_square < 0.0
    # ... and the plate cannot go between the legs square on
    conflicts = pl.footprint_conflicts(TABLE_TASK, pl.BasePose((-0.40, -0.108, 0.0), 0.0), tianji.floor_boxes, tianji.fixed_points)
    assert any("table leg" in conflict for conflict in conflicts)


def test_rectangles_overlap_is_a_separating_axis_test():
    square = pl._aabb_rectangle((0.0, 1.0), (0.0, 1.0))
    assert pl.rectangles_overlap(square, pl._aabb_rectangle((0.9, 2.0), (0.9, 2.0)))
    assert not pl.rectangles_overlap(square, pl._aabb_rectangle((1.1, 2.0), (0.0, 1.0)))
    diamond = pl._rectangle((0.0, 0.0), (0.5, 0.5), 45.0, (1.6, 0.5))  # tip at x = 0.893
    assert pl.rectangles_overlap(square, diamond)
    assert not pl.rectangles_overlap(square, pl._rectangle((0.0, 0.0), (0.5, 0.5), 45.0, (1.75, 0.5)))


def test_base_pose_transforms():
    base = pl.BasePose((1.0, 2.0, 0.0), 90.0)
    assert base.orientation_xyzw == pytest.approx((0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))
    assert base.to_base((1.0, 3.0, 0.5)) == pytest.approx((1.0, 0.0, 0.5))


def _tianji_ik():
    definition = yaml.safe_load(TIANJI_DEFINITION.read_text(encoding="utf-8"))
    tianji = ARMS.arm("tianji")
    return pl.ArmIk(ROBOT_ASSETS / definition["urdf_path"], definition["manipulation"]["arm_joint_names"]["right"], tianji.bench.palm_link)


@needs_instances
@needs_tianji
@needs_pinocchio
def test_tianji_palm_can_be_led_along_every_task_path_from_its_start_configuration():
    tianji = TURNED_TIANJI
    ik = _tianji_ik()
    for task in CATALOG.tasks.values():
        result = task_path_reach(tianji, task, ik)
        assert result.follows, (task.activity, result)
        assert result.max_position_error_m < 1e-3
        assert result.ok and result.joint_limit_margin_deg > 10.0, (task.activity, result)
    assert task_path_reach(ARMS.arm("franka"), TABLE_TASK, ik) is None
    # the assigned stances (apparatus as saved) work as well with the 809 mm arm
    for task in CATALOG.tasks.values():
        result = task_path_reach(ARMS.arm("tianji"), task, ik)
        assert result.ok and result.joint_limit_margin_deg > 10.0, (task.activity, result)


@needs_instances
@needs_tianji
@needs_pinocchio
def test_an_ik_solution_is_not_enough_the_abandoned_close_stance_cannot_be_led():
    """Why the stances are checked by leading the palm: the first stance chosen by static IK fails it.

    Base (-0.42, -0.04) yawed 45 degrees has an IK solution with joint limits for the table key points
    (pick points 0.25 m from the shoulder), yet from its start configuration the palm cannot be led to them.
    """

    tianji = ARMS.arm("tianji")
    ik = _tianji_ik()
    close = pl.BasePose((-0.42, -0.04, 0.0), 45.0)
    assert not pl.footprint_conflicts(TABLE_TASK, close, tianji.floor_boxes, tianji.fixed_points)
    assert all(point.reachable for point in pl.task_sphere_reach(TABLE_TASK, close, tianji.workspace))
    start = pl.task_key_points(TABLE_TASK)[0]
    assert pl.ik_reach(start, close, tianji.workspace, ik, seeds=[tianji.reset_arm_joint_positions(TABLE_TASK)]).solved
    # its start configuration: palm down over (-0.10, -0.45, 1.28), fingers along +X
    start_configuration = [1.684598, -1.561114, 2.582228, -2.282263, 0.305508, -0.746781, -1.265716]
    result = pl.path_reach(TABLE_TASK, close, ik, start_configuration)
    assert not result.follows and result.max_position_error_m > 0.05
    assert result.joint_limit_margin_deg == pytest.approx(0.0, abs=1e-6)  # stopped at a joint limit


@needs_instances
@needs_tianji
@needs_pinocchio
def test_tianji_start_configurations_are_palm_down_inside_the_clamp_and_clear_of_the_task_points():
    tianji = ARMS.arm("tianji")
    ik = _tianji_ik()
    workspace = tianji.workspace
    for task in CATALOG.tasks.values():
        base = tianji.base_pose(task)
        q = np.asarray(tianji.reset_arm_joint_positions(task))
        assert np.all(q > ik.lower + math.radians(10.0)) and np.all(q < ik.upper - math.radians(10.0))
        placement = ik.forward(q)
        palm_base = placement.translation
        assert np.linalg.norm(palm_base - np.asarray(workspace.shoulder_position)) < workspace.shoulder_reach - 0.05
        assert workspace.minimum_z < palm_base[2] < workspace.maximum_z
        assert placement.rotation[2, 0] < -0.99  # palm normal straight down
        assert placement.rotation[0, 2] > 0.99  # fingers along the robot's +X
        world = Rotation.from_euler("z", base.yaw_deg, degrees=True).apply(palm_base) + np.asarray(base.position)
        targets = [np.asarray(point.position) for point in pl.task_key_points(task, tianji.layout(task))]
        touch = tianji.stance(task).touches.get(task.activity)
        if touch is not None:
            targets.append(np.asarray(touch.waypoints[-1][0]))
        for target in targets:
            assert np.linalg.norm(world - target) > 0.15, (task.activity, target, world)
        if layout_group(task) == "table":
            assert world[2] > 1.026 + 0.15  # at least 15 cm above the tabletop


@needs_instances
@needs_tianji
@needs_pinocchio
def test_rest_start_pose_lays_the_open_hand_just_above_the_lid_beside_the_object():
    """--reset-pose rest: the operator's hand-on-the-table posture (the EMG neutral), for the table stances in use."""
    tianji = ARMS.arm("tianji")
    ik = _tianji_ik()
    for task in CATALOG.tasks.values():
        if layout_group(task) == "mannequin":
            with pytest.raises(ValueError, match="no reset pose 'rest'"):
                tianji.reset_arm_joint_positions(task, "rest")
            continue
        base = tianji.base_pose(task)
        q = np.asarray(tianji.reset_arm_joint_positions(task, "rest"))
        assert np.all(q > ik.lower + math.radians(8.0)) and np.all(q < ik.upper - math.radians(8.0))
        placement = ik.forward(q)
        assert placement.rotation[2, 0] < -0.99 and placement.rotation[0, 2] > 0.99  # palm down, fingers ahead
        world = Rotation.from_euler("z", base.yaw_deg, degrees=True).apply(placement.translation) + np.asarray(base.position)
        # the open hand reaches 34.8 mm below the palm link (measured in the simulator); the lid's top is at 1.044 m
        assert world[2] - 0.0348 - 1.044 == pytest.approx(0.020, abs=0.002)
        # beside the object, not over it
        pick = np.asarray(pl.task_key_points(task)[0].position)
        assert abs(world[1] - pick[1]) > 0.10
        # the palm can be led along the task's path from there (the joint margin is small at transport height)
        assert pl.path_reach(task, base, ik, q).follows, task.activity


@needs_instances
def test_recorded_touches_put_the_hand_inside_the_scoring_regions():
    tianji = ARMS.arm("tianji")
    for name in ("behind_head", "top_head", "mouth"):
        task = CATALOG.tasks[f"arat_gross_movement_hand_{name}"]
        touch = tianji.stance(task).touches[task.activity]
        position, orientation = touch.waypoints[-1]
        contact = np.asarray(position) + Rotation.from_quat(orientation).apply(touch.contact_in_palm)
        states, init_info = pl._object_states(task)
        lower, upper = pl.mannequin_aabb(states, init_info)
        region_name = pl._rubric(task)["target"]["region"]
        assert MANNEQUIN_REGIONS[region_name].contains(tuple(contact), tuple(lower), tuple(upper)), (name, contact)
        # on the mannequin, not in the air beside it
        assert np.all(contact >= lower - 0.01) and np.all(contact <= upper + 0.01)


@needs_tianji
def test_tianji_hand_links_and_assisted_grasp_points_match_the_franka_sharpa_hand():
    tianji = yaml.safe_load(TIANJI_DEFINITION.read_text(encoding="utf-8"))["manipulation"]
    franka = yaml.safe_load((ROBOT_ASSETS / "models" / "franka" / "franka.yaml").read_text(encoding="utf-8"))
    sharpa = franka["manipulation"]["end_effectors"]["sharpa_right"]
    assert tianji["eef_link_names"]["right"] == sharpa["eef_link_names"]["0"] == "right_hand_C_MC"
    assert tianji["finger_link_names"]["right"] == sharpa["finger_link_names"]["0"]
    assert sorted(tianji["finger_joint_names"]["right"]) == sorted(sharpa["finger_joint_names"]["0"])
    assert tianji["assisted_grasp_start_points"]["right"] == sharpa["ag_start_points"]
    assert tianji["assisted_grasp_end_points"]["right"] == sharpa["ag_end_points"]


# --------------------------------------------------------------------------- launcher validation


class _Args:
    def __init__(self, reset_pose="extended", allow_unreachable=False):
        self.reset_pose, self.allow_unreachable = reset_pose, allow_unreachable


@needs_instances
def test_launcher_rejects_franka_only_start_poses_for_the_tianji(capsys):
    tianji = TURNED_TIANJI
    with pytest.raises(SystemExit, match="--reset-pose compact is not a start pose of arm 'tianji'"):
        _validate_arm_selection(tianji, (TABLE_TASK,), _Args("compact"))
    _validate_arm_selection(tianji, (TABLE_TASK, MANNEQUIN_TASK), _Args())
    printed = capsys.readouterr().out
    assert "Arm tianji reach, arat_grasp_block_10cm" in printed and "palm margin" in printed
    _validate_arm_selection(ARMS.arm("franka"), (TABLE_TASK,), _Args("compact"))
    assert capsys.readouterr().out == ""  # nothing new is printed for the Franka
    # as assigned (apparatus as saved) every task loads too
    _validate_arm_selection(ARMS.arm("tianji"), (TABLE_TASK, POUR_TASK, MANNEQUIN_TASK), _Args())


@needs_instances
def test_launcher_refuses_an_unreachable_or_excluded_task_unless_told_otherwise(capsys):
    tianji = ARMS.arm("tianji")
    far = {name: replace(stance, base=pl.BasePose((-2.0, 0.0, 0.0), 0.0)) for name, stance in tianji.stances.items()}
    moved = replace(tianji, stances=far)
    with pytest.raises(SystemExit, match="cannot do: arat_grasp_block_10cm"):
        _validate_arm_selection(moved, (TABLE_TASK,), _Args())
    _validate_arm_selection(moved, (TABLE_TASK,), _Args(allow_unreachable=True))
    excluded = replace(tianji, excluded_tasks={TABLE_TASK.activity: "no stance passes the path check"})
    assert not task_reach(excluded, TABLE_TASK).reachable
    capsys.readouterr()
    with pytest.raises(SystemExit, match="cannot do: arat_grasp_block_10cm"):
        _validate_arm_selection(excluded, (TABLE_TASK, POUR_TASK), _Args())
    assert "EXCLUDED: no stance passes the path check" in capsys.readouterr().out


# --------------------------------------------------------------------------- the apparatus re-placed on the table


def test_layout_is_a_rigid_motion_in_the_table_plane():
    layout = Layout(yaw_deg=-90.0, about=(0.2415, -0.1085), translation=(0.03, -0.02))
    assert layout.point([0.2415, -0.1085, 1.0]) == pytest.approx([0.2715, -0.1285, 1.0])  # the pivot only translates
    # -90 deg about +Z: (dx, dy) -> (dy, -dx); height kept
    assert layout.point([0.3415, -0.0085, 1.3]) == pytest.approx([0.2415 + 0.1 + 0.03, -0.1085 - 0.1 - 0.02, 1.3])
    assert layout.direction([1.0, 0.0, 0.5]) == pytest.approx([0.0, -1.0, 0.5])
    turned = Rotation.from_quat(layout.orientation([0.0, 0.0, 0.0, 1.0]))
    assert turned.as_euler("zyx", degrees=True) == pytest.approx([-90.0, 0.0, 0.0])
    tilted = Rotation.from_euler("xyz", [20.0, -35.0, 50.0], degrees=True)
    expected = Rotation.from_euler("z", -90.0, degrees=True) * tilted
    assert Rotation.from_quat(layout.orientation(tilted.as_quat())).as_matrix() == pytest.approx(expected.as_matrix(), abs=1e-12)
    back = layout.inverse()
    assert back.point(layout.point([0.4, 0.7, 1.1])) == pytest.approx([0.4, 0.7, 1.1])
    assert Layout.from_json(layout.to_json()) == layout and Layout.from_json(None) is None
    assert Layout().is_identity and Layout(yaw_deg=360.0).is_identity and not layout.is_identity
    # the scorer and OnShelf read axis-aligned boxes of the shell and the tins: quarter turns only
    with pytest.raises(ValueError, match="multiple of 90 degrees"):
        Layout(yaw_deg=10.0)
    hash(layout)  # placement caches the scene per layout


@needs_instances
def test_a_layout_moves_the_apparatus_as_one_piece_and_nothing_else():
    layout = TURNED_TIANJI.layout(TABLE_TASK)
    for task in (TABLE_TASK, CATALOG.tasks["arat_grip_alloy_tube_2_5cm"], CATALOG.tasks["arat_pinch_marble_index"], POUR_TASK):
        saved = get_task_scene_data(task)
        moved = get_task_scene_data(task, layout=layout)
        assert get_task_scene_data(task, layout=None) == saved == get_task_scene_data(task, layout=Layout())
        names = moved_object_names(saved)
        assert "table" not in names and "arat_box" in names and not any("light" in name for name in names)
        before, after = (scene["state"]["registry"]["object_registry"] for scene in (saved, moved))
        for name in before:
            if name not in names:
                assert after[name] == before[name]  # the table and the lights stay
                continue
            position, rotation = pl._root_pose(after, name)
            position_before, rotation_before = pl._root_pose(before, name)
            assert position == pytest.approx(layout.point(position_before)) and position[2] == position_before[2]
            assert (rotation * rotation_before.inv()).as_euler("zyx", degrees=True) == pytest.approx([-90.0, 0.0, 0.0], abs=1e-6)
            assert {k: v for k, v in after[name].items() if k != "root_link"} == {k: v for k, v in before[name].items() if k != "root_link"}
        # one rigid motion: every distance between apparatus parts is what it was
        for first in names:
            for second in names:
                d_before = np.linalg.norm(pl._root_pose(before, first)[0] - pl._root_pose(before, second)[0])
                d_after = np.linalg.norm(pl._root_pose(after, first)[0] - pl._root_pose(after, second)[0])
                assert d_after == pytest.approx(d_before, abs=1e-12)
        assert moved["objects_info"] == saved["objects_info"] and moved["init_info"] == saved["init_info"]
    # the water goes with its cup
    water = lambda scene: np.asarray(scene["state"]["registry"]["system_registry"]["water"]["particle_states"]["waterInstancer0"]["particle_positions"])
    cup = lambda scene: np.asarray(scene["state"]["registry"]["object_registry"]["plastic_tumbler_2"]["root_link"]["pos"])
    assert water(moved) - cup(moved) == pytest.approx(np.array([layout.direction(v) for v in water(saved) - cup(saved)]), abs=1e-12)
    # in place and returned; a scene without anything to move is untouched
    scene = get_task_scene_data(MANNEQUIN_TASK)
    assert apply_layout(scene, layout) is scene and scene == get_task_scene_data(MANNEQUIN_TASK)


@needs_instances
def test_the_franka_and_the_cups_get_the_saved_scene_and_the_tianji_box_tasks_the_turned_one():
    tianji, franka = TURNED_TIANJI, ARMS.arm("franka")
    saved = get_task_scene_data(TABLE_TASK)["state"]
    assert build_environment_config(TABLE_TASK, arm=franka)["scene"]["scene_file"]["state"] == saved
    # as assigned since 2026-11-15 the Tianji gets the saved scene too
    assert build_environment_config(TABLE_TASK, arm=ARMS.arm("tianji"))["scene"]["scene_file"]["state"] == saved
    assert build_environment_config(POUR_TASK, arm=tianji)["scene"]["scene_file"]["state"] == get_task_scene_data(POUR_TASK)["state"]
    turned = build_environment_config(TABLE_TASK, arm=tianji)["scene"]["scene_file"]["state"]["registry"]["object_registry"]
    box = saved["registry"]["object_registry"]["arat_box"]["root_link"]
    assert turned["arat_box"]["root_link"]["pos"] == pytest.approx(tianji.layout(TABLE_TASK).point(box["pos"]))
    assert turned["table"] == saved["registry"]["object_registry"]["table"]
    # the box now stands end-on to the robot: its long side (the shelf, 0.55 m) runs along world X
    (x_range, y_range, top) = pl.shelf_world(turned)
    assert x_range[1] - x_range[0] == pytest.approx(0.546, abs=1e-3) and y_range[1] - y_range[0] == pytest.approx(0.15, abs=1e-3)
    assert top == pytest.approx(pl.shelf_world(saved["registry"]["object_registry"])[2])


@needs_instances
def test_table_front_is_the_image_of_the_side_stance_under_the_quarter_turn():
    """Robot and apparatus are arranged exactly as in table_side, which is what was checked there."""

    tianji = ARMS.arm("tianji")
    front, side = tianji.stances["table_front"], tianji.stances["table_side"]
    assert front.base.position[:2] == pytest.approx(front.layout.point(side.base.position[:2]))
    assert front.base.yaw_deg == pytest.approx(side.base.yaw_deg + front.layout.yaw_deg)
    assert front.camera_focus == pytest.approx(front.layout.point(side.camera_focus))
    # (with the 696 mm arm the two start configurations were identical; for the 809 mm arm each was solved again)
    for task in CATALOG.tasks.values():
        if tianji.stance(task) is not front:
            continue
        turned = [front.base.to_base(point.position) for point in pl.task_key_points(task, front.layout)]
        as_saved = [side.base.to_base(point.position) for point in pl.task_key_points(task)]
        assert np.asarray(turned) == pytest.approx(np.asarray(as_saved), abs=1e-9)
    # it is on the Franka's side of the table, facing it, and nearly square to it
    (legs, top) = pl.table_obstacles(*pl._object_states(TABLE_TASK))
    assert front.base.position[0] < top[:, 0].min() - 0.3 and top[:, 1].min() < front.base.position[1] < top[:, 1].max()
    assert abs(front.base.yaw_deg) <= 10.0


@needs_instances
def test_the_turned_apparatus_stands_on_the_tabletop():
    tianji = TURNED_TIANJI
    layout = tianji.layout(TABLE_TASK)
    for task in CATALOG.tasks.values():
        if tianji.layout(task) is None:
            continue
        margin, name = pl.tabletop_margin(task, layout)
        assert margin > 0.05 and name == "arat_box", (task.activity, margin, name)  # the lid's end, 5.5 cm inside
        assert pl.layout_conflicts(task, layout) == []
        # as saved the lid's end is 5.1 cm from the Franka's edge of the table: the apparatus cannot come closer as a whole
        assert pl.tabletop_margin(task)[0] == pytest.approx(0.051, abs=2e-3)
    footprints = pl.apparatus_footprints(TABLE_TASK, layout)
    lower, upper = footprints["arat_box"]
    assert upper[0] - lower[0] == pytest.approx(0.546, abs=1e-3) and upper[1] - lower[1] == pytest.approx(0.462, abs=1e-3)
    off_the_edge = Layout(yaw_deg=-90.0, about=layout.about, translation=(0.0, 0.05))
    assert any("from the tabletop edge" in conflict for conflict in pl.layout_conflicts(TABLE_TASK, off_the_edge))
    assert pl.layout_conflicts(MANNEQUIN_TASK, layout) == ["a layout was given for a scene without a table"]
    assert pl.layout_conflicts(TABLE_TASK, None) == [] and pl.layout_conflicts(TABLE_TASK, Layout()) == []
    # a stance whose layout does not fit is not reachable, and says why
    stances = {**tianji.stances, "table_front": replace(tianji.stances["table_front"], layout=off_the_edge)}
    reach = task_reach(replace(tianji, stances=stances), TABLE_TASK)
    assert not reach.reachable and "cannot be set up" in reach.describe()


@needs_instances
@needs_tianji
@needs_pinocchio
def test_the_saved_layout_is_worked_from_the_front_only_close_and_turned_toward_the_box():
    """With the 809 mm arm shelving works from the front of the table, but only from close by."""

    tianji = ARMS.arm("tianji")
    ik = _tianji_ik()
    start = tianji.reset_arm_joint_positions(TABLE_TASK)
    assert task_path_reach(tianji, TABLE_TASK, ik).ok
    # 15 cm further back the shelf is out of reach again
    back = pl.BasePose((-0.67, 0.1, 0.0), 15.0)
    assert not pl.footprint_conflicts(TABLE_TASK, back, tianji.floor_boxes, tianji.fixed_points)
    assert not pl.path_reach(TABLE_TASK, back, ik, start).ok
    front = tianji.stances["table_front"]
    assert pl.path_reach(TABLE_TASK, front.base, ik, front.reset_poses["extended"], None, front.layout).ok
    # the cups are in reach of the same place as they stand
    assert task_path_reach(tianji, POUR_TASK, ik).ok and tianji.layout(POUR_TASK) is None


# --------------------------------------------------------------------------- provenance and replay


def _recording(path, robot, provenance=None):
    with h5py.File(path, "w") as recording:
        data = recording.create_group("data")
        data.attrs["config"] = json.dumps({"robots": [robot], "task": {"activity_name": TABLE_TASK.activity}})
        data.create_group("demo_0").attrs["num_samples"] = 1
    if provenance is not None:
        write_arm_provenance(path, provenance)


def test_arm_provenance_round_trip_and_mismatch(tmp_path):
    tianji, franka = TURNED_TIANJI, ARMS.arm("franka")
    saved_record = arm_provenance(ARMS.arm("tianji"), TABLE_TASK)  # as assigned: the apparatus as saved
    assert saved_record["stance"] == "table_front_saved" and saved_record["layout"] is None
    record = arm_provenance(tianji, TABLE_TASK)
    assert saved_record["base_position"] == [-0.52, 0.1, 0.0] and saved_record["operator_yaw_deg"] == 15.0
    assert record["arm"] == "tianji" and record["base_pose_source"] == "arat/arms.yaml stance"
    assert record["stance"] == "table_front" and record["operator_yaw_deg"] == 10.0
    assert record["layout"] == {"yaw_deg": -90.0, "about": [0.2415, -0.1085], "translation": [0.0, 0.0]}
    assert record["base_position"] == list(tianji.base_pose(TABLE_TASK).position)
    assert record["scene_camera_focus"] == list(tianji.stance(TABLE_TASK).camera_focus)
    assert record["head_camera"] == ["arat_mobile_chest_camera", [0.13, -0.03, 1.5]]
    assert record["reset_arm_joint_positions"] == list(tianji.reset_arm_joint_positions(TABLE_TASK))
    pour_record = arm_provenance(tianji, POUR_TASK)
    assert pour_record["stance"] == "table_front_cups" and pour_record["layout"] is None
    assert pour_record["base_position"] == [-0.61, 0.173, 0.0] and record["base_position"] == [-0.57, 0.173, 0.0]
    franka_record = arm_provenance(franka, TABLE_TASK, "compact")
    assert franka_record["base_position"] is None and franka_record["base_pose_source"].startswith("task instance")
    assert franka_record["stance"] is None and franka_record["operator_yaw_deg"] == 0.0
    assert franka_record["layout"] is None
    assert franka_record["head_camera"] is None and franka.head_camera is None
    json.dumps(record), json.dumps(franka_record)

    path = tmp_path / "tianji.hdf5"
    _recording(path, {**tianji.robot_identity(), "position": [0, 0, 0]}, record)
    arm, provenance = recording_arm(path, ARMS)
    assert arm.key == "tianji" and provenance == record
    with h5py.File(path, "r") as recording:
        assert ARM_PROVENANCE_ATTRIBUTE in recording["data"].attrs

    legacy = tmp_path / "legacy.hdf5"  # recorded before --arm existed: no attribute, Franka robot config
    _recording(legacy, {"model": "franka", "end_effector": "sharpa_right", "name": "franka_sharpa_right"})
    arm, provenance = recording_arm(legacy, ARMS)
    assert arm.key == "franka" and provenance is None

    inconsistent = tmp_path / "inconsistent.hdf5"
    _recording(inconsistent, franka.robot_identity(), record)
    with pytest.raises(ValueError, match="inconsistent"):
        recording_arm(inconsistent, ARMS)
    unknown = tmp_path / "unknown.hdf5"
    _recording(unknown, {"model": "ur5e", "name": "ur5e"})
    with pytest.raises(ValueError, match="No ARAT arm uses robot model 'ur5e'"):
        recording_arm(unknown, ARMS)


def test_replay_refuses_another_arm_and_hangs_the_wrist_cameras_on_the_recorded_one(tmp_path):
    from dex_teleop.omnigibson import replay

    tianji = ARMS.arm("tianji")
    path = tmp_path / "tianji.hdf5"
    _recording(path, tianji.robot_identity(), arm_provenance(tianji, TABLE_TASK))
    with pytest.raises(SystemExit, match="made with arm 'tianji'.*cannot be replayed as 'franka'"):
        replay.main([str(path), "--episode-id", "0", "--arm", "franka"])
    cameras = json.dumps(build_replay_camera_configs(TABLE_TASK.camera_rig, tianji.prim_path))
    assert tianji.prim_path in cameras and ROBOT_PRIM_PATH not in cameras
    assert ROBOT_PRIM_PATH in json.dumps(build_replay_camera_configs(TABLE_TASK.camera_rig))


def _recording_with_scene(path, arm, task, provenance, scene):
    with h5py.File(path, "w") as recording:
        data = recording.create_group("data")
        data.attrs["config"] = json.dumps({"robots": [arm.robot_identity()], "task": {"activity_name": task.activity}})
        data.attrs["scene_file"] = json.dumps(scene)
        data.create_group("demo_0").attrs["num_samples"] = 1
    write_arm_provenance(path, provenance)


@needs_instances
def test_replay_uses_the_recorded_stance_and_refuses_a_recording_whose_layout_does_not_match(tmp_path):
    tianji, franka = TURNED_TIANJI, ARMS.arm("franka")
    record = arm_provenance(tianji, TABLE_TASK)
    scene = build_environment_config(TABLE_TASK, arm=tianji)["scene"]["scene_file"]
    good = tmp_path / "front.hdf5"
    _recording_with_scene(good, tianji, TABLE_TASK, record, scene)
    assert recorded_stance(good, tianji, TABLE_TASK, record) is tianji.stances["table_front"]
    # ... also now that arms.yaml assigns the box tasks to another stance
    assert recorded_stance(good, ARMS.arm("tianji"), TABLE_TASK, record) is tianji.stances["table_front"]
    assert recorded_stance(good, franka, TABLE_TASK, None) is None

    # a recording made at the table's side before the layout existed: no layout key, the scene as saved
    side_record = {key: value for key, value in record.items() if key != "layout"}
    side = tianji.stances["table_side"]
    side_record.update(stance="table_side", base_position=list(side.base.position), base_yaw_deg=side.base.yaw_deg)
    older = tmp_path / "side.hdf5"
    _recording_with_scene(older, tianji, TABLE_TASK, side_record, get_task_scene_data(TABLE_TASK))
    assert recorded_stance(older, tianji, TABLE_TASK, side_record) is side
    # its cameras are the side stance's, not those of the stance the task has today
    cameras = lambda stance: build_replay_camera_configs(TABLE_TASK.camera_rig, tianji.prim_path)
    today, recorded = cameras(None), cameras(side)
    tianji.place_scene_cameras(TABLE_TASK, today)
    tianji.place_scene_cameras(TABLE_TASK, recorded, stance=side)
    assert today != recorded

    # the stance was redefined since: another layout, or another base pose
    moved = replace(tianji.stances["table_front"], layout=Layout(yaw_deg=-90.0, about=(0.2415, -0.1085), translation=(0.02, 0.0)))
    with pytest.raises(ValueError, match="arat/arms.yaml now defines it"):
        recorded_stance(good, replace(tianji, stances={**tianji.stances, "table_front": moved}), TABLE_TASK, record)
    shifted_base = replace(tianji.stances["table_front"], base=pl.BasePose((-0.60, 0.173, 0.0), 10.0))
    with pytest.raises(ValueError, match="arat/arms.yaml now defines it"):
        recorded_stance(good, replace(tianji, stances={**tianji.stances, "table_front": shifted_base}), TABLE_TASK, record)
    without = {name: stance for name, stance in tianji.stances.items() if name != "table_front"}
    with pytest.raises(ValueError, match="no longer\\s+defines|no longer defines"):
        recorded_stance(good, replace(tianji, stances=without), TABLE_TASK, record)
    with pytest.raises(ValueError, match="without a stance"):
        recorded_stance(good, tianji, TABLE_TASK, None)

    # the provenance says "turned" but the recorded scene has the box where it was saved
    inconsistent = tmp_path / "inconsistent.hdf5"
    _recording_with_scene(inconsistent, tianji, TABLE_TASK, record, get_task_scene_data(TABLE_TASK))
    with pytest.raises(ValueError, match="its scene has arat_box at .* the recording is inconsistent"):
        recorded_stance(inconsistent, tianji, TABLE_TASK, record)

    # the replay entry point refuses before the simulator starts
    from dex_teleop.omnigibson import replay

    stale = tmp_path / "stale.hdf5"
    stale_record = {**record, "layout": {"yaw_deg": 90.0, "about": [0.2415, -0.1085], "translation": [0.0, 0.0]}}
    _recording_with_scene(stale, tianji, TABLE_TASK, stale_record, scene)
    with pytest.raises(SystemExit, match="Cannot replay: .*arat/arms.yaml now defines it"):
        replay.main([str(stale), "--episode-id", "0"])


def test_arms_yaml_is_validated(tmp_path):
    document = yaml.safe_load(ARMS_PATH.read_text(encoding="utf-8"))
    path = tmp_path / "arms.yaml"

    def load(tianji_changes=None, **top_level):
        changed = {**document, **top_level}
        if tianji_changes is not None:
            changed["arms"] = {**document["arms"], "tianji": {**document["arms"]["tianji"], **tianji_changes}}
        path.write_text(yaml.safe_dump(changed), encoding="utf-8")
        return load_arat_arms(path)

    assert load().names == ARMS.names
    with pytest.raises(ValueError, match="layouts must name a stance for exactly"):
        load({"layouts": {"table": "table_side"}})
    stances = document["arms"]["tianji"]["stances"]
    with pytest.raises(ValueError, match="multiple of 90 degrees"):
        load({"stances": {**stances, "table_front": {**stances["table_front"], "layout": {"yaw_deg": -80.0}}}})
    with pytest.raises(ValueError, match="Unknown layout keys: rotation"):
        load({"stances": {**stances, "table_front": {**stances["table_front"], "layout": {"rotation": -90.0}}}})
    with pytest.raises(ValueError, match="mannequin touches cannot move the apparatus"):
        load({"stances": {**stances, "mannequin_side": {**stances["mannequin_side"], "layout": {"yaw_deg": 90.0}}}})
    with pytest.raises(ValueError, match="mannequin stance cannot move the apparatus"):
        load({"layouts": {"table": "table_front", "mannequin": "table_front"}})
    # the side stance is still defined and can be assigned again
    assert load({"layouts": {"table": "table_side", "mannequin": "mannequin_side"}}).arm("tianji").stance(TABLE_TASK).name == "table_side"
    with pytest.raises(ValueError, match="unknown stance 'nowhere'"):
        load({"task_stances": {"arat_grip_pour_water": "nowhere"}})
    with pytest.raises(ValueError, match="reset_poses must include the default pose 'home'"):
        load({"default_reset_pose": "home"})
    with pytest.raises(ValueError, match="default_arm"):
        load(default_arm="ur5")
    with pytest.raises(ValueError, match="distinct robot names"):
        load({"robot_name": document["arms"]["franka"]["robot_name"]})
    excluded = load({"excluded_tasks": {"arat_grip_pour_water": "the cups are out of reach"}})
    assert excluded.arm("tianji").excluded_reason(POUR_TASK) == "the cups are out of reach"
