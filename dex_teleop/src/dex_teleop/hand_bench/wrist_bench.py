"""Wrist bench: glove fingers plus a tracker-driven wrist, no ARAT task.

The hand bench pins the wrist; this bench is its complement. The arm that
carries the right Sharpa hand stays visible over the same plain table and runs
the ARAT ``InverseKinematicsController``, so the selected articulation source
drives the 22 finger joints and the selected wrist source (the VIVE Ultimate
Tracker on the glove's MANUS mount, through the MANUS sidecar) drives the
wrist exactly as in ARAT teleoperation, with the same ``SharpaActionAdapter``,
anchoring, workspace clamps, and tracking presets. There is no task, scoring,
or recording: the bench exists to prove a glove + tracker rig before an ARAT
session and to tune the tracker-to-wrist mount offset.

The arm is selectable: ``wrist_bench.arms`` in ``hand_bench.yaml`` describes
each one (the Franka Emika Panda of the ARAT task, and the Tianji Marvin M6 CCS
of the Marvin Pro with the hand on its official end flange), ``wrist_bench.arm``
names the default and ``--arm`` overrides it. Each entry carries the robot
model and placement, the palm link, the named reset configurations
(``--reset-pose``), the adapter's workspace clamps and the camera rig that
frames it.

SPACE engages and anchors (the current tracker pose maps onto the current hand
pose; SPACE again re-anchors), R returns the arm to its reset pose, T/E toggle
the commanded-target (magenta) and measured-EEF (cyan) frames, B cycles the
cameras. While engaged the bench prints one telemetry line per
``--telemetry-interval``: tracker identity, quality and health, how far and how
much the tracked wrist moved since the anchor, and the commanded-target to
measured-EEF error, so the three hardware checks of the runbook can be read
off directly: translating the tracker moves the wrist, rotating about the
wrist barely translates it (mount offset), and covering the tracker's cameras
degrades the quality and stops the wrist.
"""

from __future__ import annotations

import argparse
from functools import partial
import logging
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from dex_teleop.arat.camera_rig import build_camera_sensor_configs, camera_rig_names, load_camera_rig
from dex_teleop.hand_bench.scene import (
    TABLE_NAME,
    HandBenchScene,
    WristBenchArm,
    WristBenchArms,
    load_hand_bench_scene,
    load_wrist_bench_arms,
    validate_wrist_bench_assets,
)
from dex_teleop.omnigibson.launch_config import add_launch_config_arguments, parse_launch_arguments
from dex_teleop.omnigibson.teleop_telemetry import (  # noqa: F401 - re-exported for the bench's callers
    WristTelemetry,
    WristTelemetrySample,
    target_position_world,
)
from dex_teleop.omnigibson.launcher import (
    GracefulShutdown,
    TrackingSelection,
    _configure_camera_layout,
    _create_worker,
    _freeze_loaded_heap,
    _reset_and_wait_for_tracking,
    _resolve_tracking_selection,
    _show_robot_end_effectors,
    _shutdown_omnigibson,
    _validate_manus_selection,
    add_tracking_arguments,
)
from dex_teleop.runtime import MultiSourceTrackingWorker, ProcessTrackingWorker
from dex_teleop.tracking import ArticulationFrameTransform
from dex_teleop.tracking.vive_ultimate import (
    STATE_BUILD_MAP,
    STATE_READY,
    TRACKER_STATE_NAMES,
    VutPoseFeed,
)


LOGGER = logging.getLogger(__name__)
CAMERA_LAYOUT = "teleop"
# The selectable arms (hand_bench.yaml: wrist_bench.arms); robot placement, reset
# poses, palm link, workspace clamps and camera rig all come from the chosen entry.
ARMS: WristBenchArms = load_wrist_bench_arms()
DEFAULT_ARM = ARMS.default_arm
MINIMUM_PALM_CLEARANCE_M = 0.15
PRESET_SKIP_FLAGS = ("--view-only",)
WAIT_MESSAGE_INTERVAL_SECONDS = 5.0
# How long the pre-flight listens on VIVE Hub's socket for the expected tracker
# before Isaac Sim starts; the Ultimate streams at ~120 Hz, so a healthy rig
# passes within a frame or two.
VIVE_PREFLIGHT_SECONDS = 3.0
VIVE_HUB_HINT = "start VIVE Hub (`./run-vivehub.sh` in the VIVEHub-Linux folder) and keep it running"
# Keys of the launch presets that only the ARAT launcher acts on; the bench
# accepts them so configs/manus_core.yaml applies unchanged, and says so.
IGNORED_PRESET_OPTIONS = ("record_hand_poses",)


# ---------------------------------------------------------------------------
# Scene


def reset_joint_positions(arm: WristBenchArm, reset_pose: str | None = None) -> list[float]:
    """The arm's named reset configuration followed by an open Sharpa hand."""

    return arm.reset_joint_positions(reset_pose)


def build_wrist_bench_config(
    scene: HandBenchScene,
    arm: WristBenchArm | None = None,
    *,
    camera_rig_name: str | None = None,
    layout_name: str = CAMERA_LAYOUT,
    reset_pose: str | None = None,
    viewport_name: str | None = "Viewport",
) -> dict:
    """Build the OmniGibson environment: the hand-bench table and the selected IK-controlled arm + Sharpa.

    ``arm`` defaults to the YAML default arm; ``camera_rig_name`` and ``reset_pose`` default to
    that arm's rig and reset pose.
    """

    if arm is None:
        arm = ARMS.arm()
    robot_config = arm.robot
    camera_rig = load_camera_rig(arm.camera_rig if camera_rig_name is None else camera_rig_name)
    sensors = build_camera_sensor_configs(
        camera_rig,
        layout_name,
        robot_prim_path=robot_config.prim_path,
        viewport_name=viewport_name,
    )
    table = scene.table
    objects = [
        {
            "type": "DatasetObject",
            "name": TABLE_NAME,
            "category": table.category,
            "model": table.model,
            "dataset_name": table.dataset_name,
            "fixed_base": True,
            "position": list(table.position),
            "orientation": list(table.orientation_xyzw),
        }
    ]
    for light in scene.lights:
        objects.append(
            {
                "type": "LightObject",
                "name": light["name"],
                "light_type": light["light_type"],
                "radius": light["radius"],
                "intensity": light["intensity"],
                "position": list(light["position"]),
            }
        )
    return {
        "env": {
            "action_frequency": 30.0,
            "automatic_reset": False,
            "external_sensors": sensors,
        },
        "scene": {
            "type": "Scene",
            "use_floor_plane": True,
            "floor_plane_visible": True,
            "floor_plane_color": list(scene.floor_plane_color),
            "use_skybox": False,
        },
        "objects": objects,
        "robots": [
            {
                **robot_config.environment_config(),
                "grasping_direction": "upper",
                "grasping_mode": "physical",
                "obs_modalities": [],
                "action_normalize": False,
                "fixed_base": True,
                "self_collisions": False,
                "reset_joint_pos": reset_joint_positions(arm, reset_pose),
                "controller_config": {
                    # The ARAT arm controller: 6-D absolute pose targets from SharpaActionAdapter.
                    robot_config.arm_controller: {
                        "name": "InverseKinematicsController",
                        "mode": "absolute_pose",
                        "command_input_limits": None,
                        "command_output_limits": None,
                    },
                    robot_config.gripper_controller: {
                        "name": "MultiFingerGripperController",
                        "mode": "independent",
                        "motor_type": "position",
                        "inverted": False,
                        "command_input_limits": None,
                        "command_output_limits": None,
                    },
                },
            }
        ],
        "task": {"type": "DummyTask"},
    }


# ---------------------------------------------------------------------------
# Yaw calibration (simulator-free, unit-tested); telemetry lives in teleop_telemetry


YAW_CALIBRATION_MINIMUM_M = 0.10


def yaw_correction_from_forward_motion(displacement_tracking, *, minimum_m: float = YAW_CALIBRATION_MINIMUM_M) -> float:
    """The ``tracking_yaw_deg`` that turns a tracked "forward" motion into the robot's +X.

    ``displacement_tracking`` is the raw wrist displacement in the tracking
    frame (before any yaw) recorded while the operator moves the tracker
    straight toward the table. VIVE Hub's map frame points wherever the
    tracker pointed when *Start setup* was pressed, so the residual yaw is
    arbitrary; this measures it instead of guessing (180 when both horizontal
    axes come out reversed, 90/-90 when forward comes out sideways).
    """

    displacement = np.asarray(displacement_tracking, dtype=np.float64).reshape(3)
    horizontal = math.hypot(displacement[0], displacement[1])
    if not math.isfinite(horizontal) or horizontal < minimum_m:
        raise ValueError(
            f"move the tracker at least {minimum_m * 100:.0f} cm horizontally toward the table "
            f"(measured {horizontal * 100:.1f} cm)"
        )
    yaw = -math.degrees(math.atan2(displacement[1], displacement[0])) % 360.0
    return float(yaw - 360.0 if yaw > 180.0 else yaw)  # (-180, 180]; a reversal reads as 180


def tracking_to_world_quaternion(yaw_deg: float) -> tuple[float, float, float, float]:
    half = math.radians(yaw_deg) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


# ---------------------------------------------------------------------------
# CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_launch_config_arguments(parser)
    add_tracking_arguments(parser)
    arm_summaries = "; ".join(f"{key}: {ARMS.arm(key).description}" for key in ARMS.names)
    parser.add_argument(
        "--arm",
        choices=ARMS.names,
        default=DEFAULT_ARM,
        help=f"Arm carrying the Sharpa hand, from hand_bench.yaml wrist_bench.arms ({arm_summaries}); "
        f"default: {DEFAULT_ARM}",
    )
    parser.add_argument(
        "--camera-rig",
        choices=camera_rig_names(),
        default=None,
        help="Camera rig with a 'teleop' layout (default: the selected arm's rig, "
        + ", ".join(f"{key}: {ARMS.arm(key).camera_rig}" for key in ARMS.names)
        + ")",
    )
    parser.add_argument(
        "--reset-pose",
        default=None,
        metavar="NAME",
        help="Start configuration of the selected arm, one of its hand_bench.yaml reset_poses ("
        + "; ".join(f"{key}: {', '.join(ARMS.arm(key).reset_poses)}" for key in ARMS.names)
        + "); default: the arm's default_reset_pose ("
        + ", ".join(f"{key}: {ARMS.arm(key).default_reset_pose}" for key in ARMS.names)
        + ")",
    )
    parser.add_argument(
        "--arm-markers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start with the commanded-target and measured-EEF frames visible (T/E toggle them)",
    )
    parser.add_argument(
        "--telemetry-interval",
        type=float,
        default=1.0,
        help="Seconds between wrist telemetry lines while engaged; 0 disables them (default: 1.0)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Stop after this many simulator steps; 0 runs until Ctrl+C (default: 0)",
    )
    parser.add_argument(
        "--engage-after",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Engage automatically this long after the bench is ready (the only way to engage headless)",
    )
    parser.add_argument("--maximum-frame-age", type=float, default=0.25)
    parser.add_argument(
        "--stale-frame-policy",
        choices=("error", "hold"),
        default="error",
        help="Error on a stale selected component, or hold the robot until fresh tracking resumes",
    )
    parser.add_argument("--initial-frame-timeout", type=float, default=10.0)
    parser.add_argument(
        "--tracking-yaw-deg",
        type=float,
        default=0.0,
        help="Rotate tracking translation and orientation about world Z before anchoring",
    )
    parser.add_argument(
        "--position-sensitivity",
        type=float,
        default=1.0,
        help="Scale tracked wrist translation about the engagement anchor (default: 1.0, i.e. 1:1)",
    )
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="Load the bench without any tracking source; the arm holds its reset pose",
    )
    parser.add_argument(
        "--record-hand-poses",
        action="store_true",
        help="Accepted for launch-preset compatibility and ignored: the wrist bench never records",
    )
    return parser


def _validate_args(args, selection: TrackingSelection) -> None:
    if (
        args.steps < 0
        or not math.isfinite(args.maximum_frame_age)
        or args.maximum_frame_age <= 0
        or not math.isfinite(args.initial_frame_timeout)
        or args.initial_frame_timeout <= 0
        or not math.isfinite(args.maximum_source_skew)
        or args.maximum_source_skew < 0
        or not math.isfinite(args.telemetry_interval)
        or args.telemetry_interval < 0
        or not math.isfinite(args.tracking_yaw_deg)
        or not math.isfinite(args.position_sensitivity)
        or args.position_sensitivity <= 0
        or (args.engage_after is not None and (not math.isfinite(args.engage_after) or args.engage_after < 0))
    ):
        raise SystemExit(
            "--steps must be >= 0 and the frame-age, timeout, skew, telemetry-interval, yaw, "
            "position-sensitivity and engage-after options must be finite and positive"
        )
    if args.view_only and args.engage_after is not None:
        raise SystemExit("--engage-after needs a tracking source; drop --view-only")
    if (
        not math.isfinite(args.manus_startup_timeout)
        or args.manus_startup_timeout <= 0
        or args.manus_connect_timeout < 0
        or args.manus_glove_timeout < 0
        or args.manus_reconnect_timeout < 0
        or args.manus_discovery_wait <= 0
        or not math.isfinite(args.manus_tracker_stale_timeout)
        or args.manus_tracker_stale_timeout <= 0
    ):
        raise SystemExit("MANUS timeouts are invalid")
    if args.hand_model != "sharpa" or args.hand != "right":
        raise SystemExit("The wrist bench drives the right-hand Sharpa robot only")
    if args.record_hand_source or args.record_wrist_source:
        raise SystemExit("The wrist bench records nothing; drop --record-hand-source/--record-wrist-source")
    arm = _selected_arm(args)
    if args.reset_pose is not None and args.reset_pose not in arm.reset_poses:
        raise SystemExit(
            f"--reset-pose {args.reset_pose!r} is not a reset pose of arm {arm.key!r}; "
            f"choose one of: {', '.join(arm.reset_poses)}"
        )
    if args.camera_rig is not None:
        try:
            load_camera_rig(args.camera_rig).layout(CAMERA_LAYOUT)
        except ValueError as error:
            raise SystemExit(f"--camera-rig {args.camera_rig!r} has no {CAMERA_LAYOUT!r} layout: {error}") from error
    if args.view_only:
        return
    _validate_manus_selection(args, selection)


def _selected_arm(args) -> WristBenchArm:
    """The arm chosen with ``--arm`` (or the YAML default)."""

    return ARMS.arm(getattr(args, "arm", None))


def _camera_rig_name(args, arm: WristBenchArm) -> str:
    return arm.camera_rig if args.camera_rig is None else args.camera_rig


def _reset_pose_name(args, arm: WristBenchArm) -> str:
    return arm.default_reset_pose if args.reset_pose is None else args.reset_pose


def _force_data_path() -> Path:
    data_root = Path(__file__).resolve().parents[4] / "datasets"
    configured = os.environ.get("OMNIGIBSON_DATA_PATH")
    if configured is not None and Path(configured).expanduser().resolve() != data_root.resolve():
        raise SystemExit(
            "OMNIGIBSON_DATA_PATH points outside Behavior-1K-arat; unset it or set it to " f"{data_root}"
        )
    os.environ["OMNIGIBSON_DATA_PATH"] = str(data_root)
    return data_root


def create_worker(
    args, selection: TrackingSelection, articulation_frame_transform: ArticulationFrameTransform | None
) -> MultiSourceTrackingWorker:
    """The ARAT tracking worker: same sources, fuser, and retargeter as launch_og.py."""

    return _create_worker(args, selection, articulation_frame_transform)


def vive_ultimate_preflight(
    socket_path: str,
    expected_tracker_id: str | None,
    *,
    seconds: float = VIVE_PREFLIGHT_SECONDS,
) -> str:
    """Listen on VIVE Hub's socket and confirm the expected Ultimate is streaming ``trackable`` poses.

    Returns a one-line summary, or raises ``SystemExit`` naming what is wrong
    and how to fix it. Without this, a sleeping or missing tracker surfaces only
    as the MANUS sidecar's 35 s "tracker packet not received" timeout, after
    Isaac Sim has already loaded.
    """

    feed = VutPoseFeed(socket_path, poll_hz=240.0, reconnect_interval=0.1)
    feed.start()
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            pose = None if expected_tracker_id is None else feed.latest(expected_tracker_id)
            if pose is not None and pose.quality == "trackable":
                break
            if expected_tracker_id is None and feed.snapshot():
                break
            time.sleep(0.05)
        connected = feed.connected_once
        protocol = feed.protocol
        snapshot = feed.snapshot()
        serials = feed.serials()
        states = feed.states()
    finally:
        feed.close()

    if not connected:
        raise SystemExit(
            f"VIVE Hub's tracker_server is not listening on {socket_path}; {VIVE_HUB_HINT}, "
            "pair the Ultimate and finish map setup, then relaunch"
        )
    streaming = {pose.tracker_id: pose for pose in snapshot.values()}
    if expected_tracker_id is None:
        if not streaming:
            raise SystemExit(f"VIVE Hub ({protocol}) streams no tracker poses on {socket_path}; wake the Ultimate")
        pose = next(iter(streaming.values()))
    else:
        pose = streaming.get(expected_tracker_id)
    if pose is None:
        # Listed by the daemon (serial known) but silent: standby, off, or awaiting map setup.
        listed = {serial: states.get(slot) for slot, serial in serials.items()}
        if expected_tracker_id in listed:
            state_code = listed[expected_tracker_id]
            state = TRACKER_STATE_NAMES.get(state_code, "unknown")
            if state_code == STATE_BUILD_MAP:
                raise SystemExit(
                    f"Tracker {expected_tracker_id} is paired but needs map setup (state {state}); "
                    "press Start setup in VIVE Hub and sweep the room, then relaunch"
                )
            raise SystemExit(
                f"Tracker {expected_tracker_id} is listed by VIVE Hub (state {state}) but streams no poses: "
                "it is in standby or switched off. Press its button or move it until its LED is on and VIVE Hub "
                f"shows it tracking; if the VIVE Hub window was closed, {VIVE_HUB_HINT}. If the window already "
                "says tracking, its tracker_server is stale (it keeps the last status after the radio link "
                "dropped): close the window, `pkill -f tracker_server`, then run ./run-vivehub.sh again"
            )
        seen = sorted(streaming) or [
            f"{serial} ({TRACKER_STATE_NAMES.get(code, 'unknown')})" for serial, code in listed.items()
        ]
        raise SystemExit(
            f"Expected tracker {expected_tracker_id!r} is not streaming from VIVE Hub; seen: "
            f"{', '.join(seen) or 'none'}. Set manus_expected_tracker_id to the serial VIVE Hub reports "
            "(configs/manus_core.yaml)"
        )
    if pose.quality != "trackable":
        reason = (
            "rotation-only tracking (cameras covered or no visual features)"
            if pose.state == STATE_READY
            else f"state {pose.state_name}"
        )
        raise SystemExit(
            f"Tracker {pose.tracker_id} is streaming but {pose.quality}: {reason}. Uncover its cameras, "
            "improve the lighting, or rebuild the map in VIVE Hub, then relaunch"
        )
    others = sorted(set(streaming) - {pose.tracker_id})
    return (
        f"VIVE Ultimate feed OK ({protocol} protocol on {socket_path}): {pose.tracker_id} {pose.state_name}/{pose.quality}"
        + (f"; also streaming: {', '.join(others)}" if others else "")
    )


# ---------------------------------------------------------------------------
# Simulator


def _validate_bench(env, robot, table, scene: HandBenchScene, arm: WristBenchArm) -> None:
    """Assert the table height, a clear palm above the table, the expected arm, and no robot-table contact."""

    from omnigibson.utils.usd_utils import RigidContactAPI

    _, aabb_max = table.aabb
    tabletop = float(aabb_max[2])
    if abs(tabletop - scene.table.tabletop_height) > scene.table.height_tolerance_m:
        raise RuntimeError(
            f"Tabletop is at {tabletop:.4f} m, expected {scene.table.tabletop_height:.4f} m; "
            "update table.position or table.tabletop_height in hand_bench.yaml"
        )
    if list(robot.arm_names) != [arm.robot.arm_name]:
        raise RuntimeError(
            f"Robot {arm.robot.model!r} exposes arms {list(robot.arm_names)}; the bench expects exactly "
            f"{arm.robot.arm_name!r} (hand_bench.yaml wrist_bench.arms.{arm.key}.robot.arm_name)"
        )
    if robot.eef_link_names[arm.robot.arm_name] != arm.palm_link:
        raise RuntimeError(
            f"Robot {arm.robot.model!r} drives {robot.eef_link_names[arm.robot.arm_name]!r} as its end effector, "
            f"but the bench expects the palm link {arm.palm_link!r}"
        )
    # robot.prim_path is absolute (/World/scene_N/...); the bench's cameras address the scene-relative path.
    if not robot.prim_path.endswith(arm.robot.prim_path):
        raise RuntimeError(
            f"Robot prim is {robot.prim_path}, expected it to end with {arm.robot.prim_path}; fix composed_model/name "
            f"for arm {arm.key!r} in hand_bench.yaml"
        )
    link = robot.links.get(arm.palm_link)
    if link is None:
        raise RuntimeError(f"Robot has no link {arm.palm_link!r}; available: {sorted(robot.links)}")
    position, _ = link.get_position_orientation()
    position = position.cpu().numpy()
    clearance = float(position[2]) - tabletop
    if clearance < MINIMUM_PALM_CLEARANCE_M:
        raise RuntimeError(
            f"{arm.palm_link} settled {clearance * 100:.1f} cm above the tabletop at reset; expected at least "
            f"{MINIMUM_PALM_CLEARANCE_M * 100:.0f} cm (check --reset-pose and the robot placement)"
        )
    if RigidContactAPI.is_in_contact(
        scene_idx=env.scene.idx,
        query_set=[robot],
        with_set=[table],
        ignore_set=None,
        current_only=True,
    ):
        raise RuntimeError(f"The {arm.key} arm or the Sharpa hand is touching the table at reset")
    print(
        f"Wrist bench ready ({arm.key} arm): {arm.palm_link} at {np.round(position, 3).tolist()}, "
        f"{clearance * 100:.1f} cm above the tabletop ({tabletop:.3f} m)"
    )


def _eef_world_pose(robot, arm_name: str) -> tuple[np.ndarray, np.ndarray]:
    position, quaternion = robot.eef_links[arm_name].get_position_orientation()
    return position.cpu().numpy(), quaternion.cpu().numpy()


def _base_world_pose(robot) -> tuple[np.ndarray, np.ndarray]:
    position, quaternion = robot.get_position_orientation()
    return position.cpu().numpy(), quaternion.cpu().numpy()


def run_bench(
    scene: HandBenchScene,
    args,
    worker: MultiSourceTrackingWorker | ProcessTrackingWorker | None,
    shutdown: GracefulShutdown,
) -> None:
    import omnigibson as og
    import omnigibson.lazy as lazy
    import torch as th
    from omnigibson.macros import gm
    from omnigibson.utils import transform_utils as T
    from omnigibson.utils.ui_utils import KeyboardEventHandler

    from dex_teleop.omnigibson.sharpa_adapter import SharpaActionAdapter, SharpaAdapterConfig

    headless = os.environ.get("OMNIGIBSON_HEADLESS") == "1"
    arm = _selected_arm(args)
    camera_rig_name = _camera_rig_name(args, arm)
    reset_pose = _reset_pose_name(args, arm)
    config = build_wrist_bench_config(
        scene,
        arm,
        camera_rig_name=camera_rig_name,
        reset_pose=reset_pose,
        viewport_name=None if headless else "Viewport",
    )
    print(
        f"\nLoading the Sharpa wrist bench (arm: {arm.key}, {arm.description}; "
        f"camera rig: {camera_rig_name}, reset pose: {reset_pose})"
    )
    env = og.Environment(configs=config)
    shutdown.install()
    env.reset()
    robot = env.robots[0]
    table = env.scene.object_registry("name", TABLE_NAME)
    if table is None:
        raise RuntimeError("Wrist bench scene did not load its table")
    reset_positions = th.tensor(reset_joint_positions(arm, reset_pose), dtype=th.float32)

    def hold_reset() -> None:
        robot.set_joint_positions(reset_positions)
        robot.keep_still()

    hold_reset()
    _show_robot_end_effectors(robot)
    _freeze_loaded_heap()
    for _ in range(10):
        hold_reset()
        og.sim.step()
    _validate_bench(env, robot, table, scene, arm)
    _configure_camera_layout(env, camera_rig_name, CAMERA_LAYOUT, robot)

    steps = 0
    if worker is None:
        print("View-only wrist bench; press B to switch cameras, Ctrl+C to exit.")
        while (args.steps <= 0 or steps < args.steps) and not shutdown.requested:
            hold_reset()
            og.sim.step()
            steps += 1
        return

    workspace = arm.workspace

    def make_adapter(yaw_deg: float) -> SharpaActionAdapter:
        return SharpaActionAdapter(
            robot,
            config=SharpaAdapterConfig(
                control_hz=30.0,
                position_sensitivity=args.position_sensitivity,
                tracking_to_world_quaternion_xyzw=tracking_to_world_quaternion(yaw_deg),
                # The arm's reach sphere and height band (hand_bench.yaml wrist_bench.arms.<arm>.workspace).
                shoulder_position=workspace.shoulder_position,
                shoulder_reach=workspace.shoulder_reach,
                minimum_z=workspace.minimum_z,
                maximum_z=workspace.maximum_z,
            ),
        )

    # The adapter is rebuilt when Y measures a new tracking-frame yaw, so it lives in a holder.
    rig = {"adapter": make_adapter(args.tracking_yaw_deg), "yaw_deg": float(args.tracking_yaw_deg), "yaw_start": None}
    print(
        f"Wrist position sensitivity: {args.position_sensitivity:g}x"
        + (f"; tracking frame yaw: {args.tracking_yaw_deg:g} degrees" if args.tracking_yaw_deg else "")
        + f"; {arm.key} workspace: {workspace.shoulder_reach:g} m about the shoulder at "
        f"{list(workspace.shoulder_position)}, z in [{workspace.minimum_z:g}, {workspace.maximum_z:g}] m"
    )
    arm_marker_visualizer = None
    if not gm.HEADLESS:
        from dex_teleop.omnigibson.workspace_visualization import ArmPoseMarkerVisualizer

        marker_draw = lazy.isaacsim.util.debug_draw._debug_draw.acquire_debug_draw_interface()
        arm_marker_visualizer = ArmPoseMarkerVisualizer(
            marker_draw, target_visible=args.arm_markers, eef_visible=args.arm_markers
        )
    telemetry = WristTelemetry(args.telemetry_interval) if args.telemetry_interval > 0 else None
    control = {
        "engaged": False,
        "anchor_after": None,
        "pending_command": None,
        "pending_marker_toggles": set(),
        "last_marker_toggle": {"target": float("-inf"), "eef": float("-inf")},
    }

    def request(command: str):
        def callback():
            control["pending_command"] = command

        return callback

    def request_marker_toggle(marker: str):
        now = time.monotonic()
        if now - control["last_marker_toggle"][marker] >= 0.5:
            control["last_marker_toggle"][marker] = now
            control["pending_marker_toggles"].add(marker)

    if not gm.HEADLESS:
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.SPACE, request("engage"))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.R, request("reset"))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.Y, request("yaw"))
        if arm_marker_visualizer is not None:
            KeyboardEventHandler.add_keyboard_callback(
                lazy.carb.input.KeyboardInput.T, lambda: request_marker_toggle("target")
            )
            KeyboardEventHandler.add_keyboard_callback(
                lazy.carb.input.KeyboardInput.E, lambda: request_marker_toggle("eef")
            )

    def rearm_anchor() -> None:
        rig["adapter"].request_anchor()
        if telemetry is not None:
            telemetry.reset()
        if arm_marker_visualizer is not None:
            arm_marker_visualizer.reset()
        control["anchor_after"] = time.monotonic()

    def engage() -> None:
        _reset_and_wait_for_tracking(
            worker, args.initial_frame_timeout, context="Tracking was not ready for engagement"
        )
        rearm_anchor()
        control["engaged"] = True
        print("Engaged; the next fresh hand frame anchors the tracked wrist onto the current hand pose")

    def reset() -> None:
        control["engaged"] = False
        control["anchor_after"] = None
        rig["yaw_start"] = None
        _reset_and_wait_for_tracking(worker, args.initial_frame_timeout, context="Tracking was not ready for reset")
        env.reset()
        hold_reset()
        _show_robot_end_effectors(robot)
        rig["adapter"].request_anchor()
        if telemetry is not None:
            telemetry.reset()
        if arm_marker_visualizer is not None:
            arm_marker_visualizer.reset()
        print("Arm reset; press SPACE to engage again")

    def calibrate_yaw(snapshot) -> None:
        """Y twice: measure the tracking-frame yaw from a forward motion and apply it."""

        if not control["engaged"] or snapshot is None:
            print("Yaw calibration needs an engaged, tracking hand: press SPACE first")
            return
        position = np.asarray(snapshot.frame.wrist_position, dtype=np.float64).copy()
        if rig["yaw_start"] is None:
            rig["yaw_start"] = position
            print(
                f"Yaw calibration: hold the hand still, move the tracker straight toward the table (the robot's +X) "
                f"by at least {YAW_CALIBRATION_MINIMUM_M * 100:.0f} cm, then press Y again"
            )
            return
        displacement = position - rig["yaw_start"]
        rig["yaw_start"] = None
        try:
            yaw = yaw_correction_from_forward_motion(displacement)
        except ValueError as error:
            print(f"Yaw calibration discarded: {error}; press Y to start over")
            return
        previous = rig["yaw_deg"]
        rig["yaw_deg"] = yaw
        rig["adapter"] = make_adapter(yaw)
        rearm_anchor()
        print(
            f"Tracking frame yaw measured: {yaw:.1f} degrees (was {previous:g}); applied and re-anchoring on the next "
            f"frame. Make it permanent with --tracking-yaw-deg {yaw:.1f} or tracking_yaw_deg: {yaw:.1f} in "
            "configs/manus_core.yaml"
        )

    print(
        "Wrist bench checks: (1) translate the tracker: the hand translates the same way "
        f"(x{args.position_sensitivity:g}); (2) rotate the hand about the wrist: the printed displacement "
        "stays within the mount-offset error; (3) cover the tracker's cameras: quality drops to bad/untrackable "
        "and the wrist stops. If forward/left come out turned or reversed, press Y, move the tracker toward the "
        "table, press Y again: the bench measures and applies the tracking-frame yaw.\n"
        "Press SPACE to engage (SPACE again re-anchors), R to reset, Y to calibrate yaw, T/E to toggle target/EEF "
        "frames, B to switch cameras; Ctrl+C exits."
    )
    holding_stale = False
    engage_at = None if args.engage_after is None else time.monotonic() + args.engage_after
    if engage_at is not None:
        print(f"Engaging automatically in {args.engage_after:g} s (--engage-after)")
    primary_error: BaseException | None = None
    try:
        while (args.steps <= 0 or steps < args.steps) and not shutdown.requested:
            if engage_at is not None and time.monotonic() >= engage_at:
                engage_at = None
                if not control["engaged"]:
                    control["pending_command"] = "engage"
            pending = control["pending_command"]
            control["pending_command"] = None
            if pending == "reset":
                reset()
            elif pending == "engage":
                engage()
            elif pending == "yaw":
                calibrate_yaw(worker.snapshot())
            adapter = rig["adapter"]
            if arm_marker_visualizer is not None:
                toggles = control["pending_marker_toggles"]
                if "target" in toggles:
                    print(f"Commanded target frame {'shown' if arm_marker_visualizer.toggle_target() else 'hidden'}")
                if "eef" in toggles:
                    print(f"Measured EEF frame {'shown' if arm_marker_visualizer.toggle_eef() else 'hidden'}")
                toggles.clear()
            worker.check_health()
            if not control["engaged"]:
                hold_reset()
                og.sim.step()
                if arm_marker_visualizer is not None and arm_marker_visualizer.update_due():
                    base_position, base_quaternion = robot.get_position_orientation()
                    eef_position, eef_quaternion = robot.eef_links[adapter.arm_name].get_position_orientation()
                    relative_position, relative_quaternion = T.relative_pose_transform(
                        eef_position, eef_quaternion, base_position, base_quaternion
                    )
                    arm_marker_visualizer.update(
                        target_position_robot=relative_position.cpu().numpy(),
                        target_quaternion_robot_xyzw=relative_quaternion.cpu().numpy(),
                        actual_position_world=eef_position.cpu().numpy(),
                        actual_quaternion_world_xyzw=eef_quaternion.cpu().numpy(),
                        base_position_world=base_position.cpu().numpy(),
                        base_quaternion_world_xyzw=base_quaternion.cpu().numpy(),
                    )
                steps += 1
                continue
            snapshot = worker.snapshot()
            anchor_after = control["anchor_after"]
            now = time.monotonic()
            if snapshot is not None:
                receipts = [snapshot.frame.receipt_timestamp]
                if snapshot.observation is not None:
                    receipts = [
                        snapshot.observation.articulation.receipt_timestamp,
                        snapshot.observation.wrist.receipt_timestamp,
                    ]
                frame_age = max(now - receipt for receipt in receipts)
                if frame_age > args.maximum_frame_age:
                    if args.stale_frame_policy == "error":
                        worker.snapshot(maximum_age=args.maximum_frame_age)
                    if not holding_stale:
                        LOGGER.warning(
                            "Selected hand observation is stale (%.3fs > %.3fs); holding until tracking resumes",
                            frame_age,
                            args.maximum_frame_age,
                        )
                        holding_stale = True
                    if anchor_after is not None and now - anchor_after > args.initial_frame_timeout:
                        raise RuntimeError(
                            f"No fresh {args.hand}-hand frame arrived within {args.initial_frame_timeout:.1f}s after anchoring"
                        )
                    robot.keep_still()
                    og.sim.step()
                    steps += 1
                    continue
                if holding_stale:
                    print("Fresh tracking resumed")
                    holding_stale = False
            if snapshot is None or (anchor_after is not None and snapshot.frame.receipt_timestamp <= anchor_after):
                if anchor_after is not None and now - anchor_after > args.initial_frame_timeout:
                    raise RuntimeError(
                        f"No fresh {args.hand}-hand frame arrived within {args.initial_frame_timeout:.1f}s after anchoring"
                    )
                robot.keep_still()
                og.sim.step()
                steps += 1
                continue
            if control["anchor_after"] is not None:
                control["anchor_after"] = None
                print("Anchored: the arm now follows the tracked wrist and the fingers follow the glove")
            action = adapter.action(snapshot)
            diagnostics = adapter.last_wrist_diagnostics
            if diagnostics is None:
                raise RuntimeError("Sharpa adapter did not expose wrist diagnostics after building an action")
            step_started = time.monotonic()
            env.step(action)
            step_seconds = time.monotonic() - step_started
            base_position, base_quaternion = _base_world_pose(robot)
            eef_position, eef_quaternion = _eef_world_pose(robot, adapter.arm_name)
            target_world = target_position_world(diagnostics.filtered_position, base_position, base_quaternion)
            if arm_marker_visualizer is not None and arm_marker_visualizer.update_due():
                filtered_quaternion = T.axisangle2quat(th.as_tensor(diagnostics.filtered_axis_angle, dtype=th.float32))
                arm_marker_visualizer.update(
                    target_position_robot=diagnostics.filtered_position,
                    target_quaternion_robot_xyzw=filtered_quaternion.cpu().numpy(),
                    actual_position_world=eef_position,
                    actual_quaternion_world_xyzw=eef_quaternion,
                    base_position_world=base_position,
                    base_quaternion_world_xyzw=base_quaternion,
                )
            if telemetry is not None:
                wrist_provenance = None
                if snapshot.observation is not None:
                    wrist_provenance = snapshot.observation.wrist.provenance
                sample = telemetry.observe(
                    now=time.monotonic(),
                    wrist_position=snapshot.frame.wrist_position,
                    wrist_quaternion_xyzw=snapshot.frame.wrist_quaternion_xyzw,
                    frame_receipt_timestamp=snapshot.frame.receipt_timestamp,
                    wrist_provenance=wrist_provenance,
                    target_position_world=target_world,
                    eef_position_world=eef_position,
                    gate_decision=diagnostics.gate_decision,
                    step_seconds=step_seconds,
                    rate_limit_scale=diagnostics.tick_scale,
                )
                if sample is not None:
                    print(sample.format())
            steps += 1
        print(f"Wrist bench stopped after {steps} steps")
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if arm_marker_visualizer is not None:
            try:
                arm_marker_visualizer.close()
            except BaseException as error:  # noqa: BLE001 - report, do not mask the primary error
                message = f"Could not close the arm-marker visualizer: {error}"
                if primary_error is not None:
                    primary_error.add_note(message)
                LOGGER.error(message)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args, launch_config = parse_launch_arguments(_parser(), argv, skip_for_flags=PRESET_SKIP_FLAGS)
    selection = _resolve_tracking_selection(args)
    _validate_args(args, selection)
    if launch_config is not None:
        print(f"Launch config: {launch_config.path} (command-line flags override it; --no-config uses the built-in defaults)")
        ignored = [key for key in IGNORED_PRESET_OPTIONS if key in launch_config.values]
        if ignored:
            print(f"Launch config keys not used by the wrist bench (it records nothing): {', '.join(ignored)}")
    elif args.view_only and not args.no_config:
        print("Launch config: not applied with --view-only (no tracking source is constructed)")
    articulation_frame_transform = None
    if not args.view_only and args.articulation_wrist_calibration is not None:
        try:
            articulation_frame_transform = ArticulationFrameTransform.load(args.articulation_wrist_calibration)
        except (OSError, ValueError) as error:
            raise SystemExit(f"Invalid --articulation-wrist-calibration: {error}") from error
    scene = load_hand_bench_scene()
    arm = _selected_arm(args)
    _force_data_path()
    validate_wrist_bench_assets(scene, arm)
    layout = load_camera_rig(_camera_rig_name(args, arm)).layout(CAMERA_LAYOUT)

    import omnigibson as og
    from omnigibson.macros import gm

    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = False
    gm.USE_PBR_MATERIALS = True
    gm.GUI_VIEWPORT_ONLY = bool(layout.get("workspace", {}).get("viewport_only", False))
    gm.RENDER_VIEWER_CAMERA = False
    logging.basicConfig(level=logging.INFO)

    if not args.view_only:
        print(
            f"Tracking: fingers from {selection.hand_source}, wrist from {selection.wrist_source}"
            + (
                f" (MANUS {args.manus_mode}, tracker feed {args.manus_tracker_feed}, "
                f"expected tracker {args.manus_expected_tracker_id})"
                if "manus" in (selection.hand_source, selection.wrist_source)
                else ""
            )
        )
        if "manus" in selection.wrist_sources and args.manus_tracker_feed == "vive_ultimate":
            # Fail in seconds with the fix spelled out, not after the sidecar's 35 s timeout.
            print(vive_ultimate_preflight(args.vive_ultimate_socket, args.manus_expected_tracker_id))
    worker = None
    shutdown = GracefulShutdown()
    launcher_error: BaseException | None = None
    try:
        if not args.view_only:
            factory = partial(create_worker, args, selection, articulation_frame_transform)
            worker = ProcessTrackingWorker(factory) if args.tracking_process else factory()
            worker.start()
        run_bench(scene, args, worker, shutdown)
    except KeyboardInterrupt:
        print("\nStopping the Sharpa wrist bench")
    except BaseException as error:
        launcher_error = error
        if isinstance(error, Exception):
            print("\nWrist bench failed before shutdown:", file=sys.stderr, flush=True)
            traceback.print_exc()
            sys.stderr.flush()
        raise
    finally:
        cleanup_errors: list[tuple[str, BaseException]] = []
        for label, close in (
            ("tracking worker", None if worker is None else worker.close),
            ("OmniGibson", lambda: _shutdown_omnigibson(og)),
        ):
            if close is None:
                continue
            try:
                close()
            except BaseException as error:  # noqa: BLE001 - collected and re-raised below
                cleanup_errors.append((label, error))
        messages = [f"Could not close {label}: {error}" for label, error in cleanup_errors]
        if launcher_error is not None:
            for message in messages:
                launcher_error.add_note(message)
                LOGGER.error(message)
        elif cleanup_errors:
            failure = RuntimeError("Wrist bench resources failed to close cleanly")
            for message in messages:
                failure.add_note(message)
            raise failure from cleanup_errors[0][1]


if __name__ == "__main__":
    main()
