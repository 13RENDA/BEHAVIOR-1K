"""The arms that can carry the right Sharpa hand through the ARAT tasks (``--arm``).

An ARAT arm is a wrist-bench arm (``hand_bench.yaml`` ``wrist_bench.arms``: robot model, palm link,
controller names, adapter workspace) plus what only the ARAT scenes need: the robot's name in the
scene, where it stands for each task (its *stance*), and the start configuration for that stance.

* ``franka`` is the arm the task instances were laid out for. It stands at the pose saved with every
  task instance and starts from ``arat.reset_poses``; nothing about it is read from ``arms.yaml``
  beyond its identity, so its behaviour is exactly what it was before ``--arm`` existed.
* ``tianji`` (Marvin Pro torso, right M6 CCS arm, 0.66 m reach) cannot work from the Franka's spot:
  ``arms.yaml`` gives it named stances (base pose, start configuration), says which stance each
  layout group or task uses, and lists the tasks no stance was found for. The stances were chosen
  and checked with ``arat.placement`` (see ``docs/tianji-arat.md``).

A stance that is yawed against the world turns the operator's station with it: the tracked wrist
motion and the scene-frame cameras are rotated by the stance yaw (the cameras about the stance's
``camera_focus``), so "forward" for the operator is forward for the robot and the side views keep
framing the work area. The rig's main camera stands where the Franka leaves room behind it; an arm
with a torso there (``head_camera``) carries that camera on its head instead, looking at the focus.

A stance may also re-place the apparatus on the table (``layout``, see ``arat.layout``): one rigid
motion of the box, its lid and everything on them, for an arm that cannot work the saved layout from
the side of the table it has to stand at. The scene cameras then move with the apparatus as well, a
recording stores the layout with the stance, and a replay refuses a recording whose stance no
longer matches ``arms.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from dex_teleop.arat.catalog import AratTask
from dex_teleop.arat.layout import Layout
from dex_teleop.arat.placement import BasePose, Workspace
from dex_teleop.arat.reset_poses import DEFAULT_RESET_POSE, RESET_POSES
from dex_teleop.hand_bench.scene import WristBenchArm, load_wrist_bench_arms
from dex_teleop.hands import SHARPA_ACTION_JOINTS

ARMS_PATH = Path(__file__).with_name("arms.yaml")
SUPPORTED_SCHEMA_VERSION = 1
LAYOUT_GROUPS = ("table", "mannequin")
ARM_DOFS = 7


def layout_group(task: AratTask) -> str:
    """``mannequin`` for the gross-movement items, ``table`` for everything on the ARAT table."""

    return "mannequin" if task.subscale == "gross_movement" else "table"


@dataclass(frozen=True)
class Touch:
    """A palm path (world frame) that ends with the hand on a mannequin scoring region."""

    # the point of the hand that lands on the mannequin, in the palm link frame
    contact_in_palm: tuple[float, float, float]
    # (palm position, palm orientation xyzw) in the world, in order; the last one is the touch
    waypoints: tuple[tuple[tuple[float, float, float], tuple[float, float, float, float]], ...]


@dataclass(frozen=True)
class Stance:
    """Where an arm stands for a group of tasks and how it starts there."""

    name: str
    description: str
    base: BasePose
    # centre of the work area (world): the scene-frame cameras turn with the stance about its
    # (x, y), and the arm's head camera looks at it
    camera_focus: tuple[float, float, float]
    # reset pose name -> 7 arm joint positions
    reset_poses: Mapping[str, tuple[float, ...]]
    # task activity -> the touch that reaches its mannequin region
    touches: Mapping[str, Touch] = field(default_factory=dict)
    # where the apparatus stands on the table for this stance; None: as saved with the task instance
    layout: Layout | None = None


@dataclass(frozen=True)
class AratArm:
    key: str
    description: str
    robot_name: str
    bench: WristBenchArm
    # None: the arm the task instances were laid out for (saved pose, arat.reset_poses)
    stances: Mapping[str, Stance] | None
    # layout group -> stance name, and per-task exceptions
    layouts: Mapping[str, str]
    task_stances: Mapping[str, str]
    # task activity -> why no stance of this arm works for it
    excluded_tasks: Mapping[str, str]
    default_reset_pose: str
    # (name, centre_xy, half_xy, taller_than_the_table) rectangles of the robot on the floor, base frame
    floor_boxes: tuple[tuple[str, tuple[float, float], tuple[float, float], bool], ...] = ()
    # base-frame (x, y) of fixed parts at tabletop height (e.g. a welded second arm)
    fixed_points: tuple[tuple[float, float], ...] = ()
    # the palm camera pods are modelled on the robot asset (the camera sensors exist either way)
    wrist_camera_pods: bool = True
    # (sensor name, base-frame position): the rig camera this arm carries on its head, or None
    head_camera: tuple[str, tuple[float, float, float]] | None = None

    @property
    def prim_path(self) -> str:
        return f"/controllable__{self.bench.robot.composed_model}__{self.robot_name}"

    @property
    def arm_controller(self) -> str:
        return self.bench.robot.arm_controller

    @property
    def gripper_controller(self) -> str:
        return self.bench.robot.gripper_controller

    @property
    def workspace(self) -> Workspace:
        workspace = self.bench.workspace
        return Workspace(workspace.shoulder_position, workspace.shoulder_reach, workspace.minimum_z, workspace.maximum_z)

    def robot_identity(self) -> dict:
        """The ``robots`` entry fields that select this robot (pose excluded)."""

        robot = self.bench.robot
        config = {"model": robot.model, "dataset_name": robot.dataset_name, "name": self.robot_name}
        if robot.end_effector is not None:
            config["end_effector"] = robot.end_effector
        return config

    def stance(self, task: AratTask) -> Stance | None:
        """This arm's stance for ``task``; None for the arm the task instances were laid out for."""

        if self.stances is None:
            return None
        return self.stances[self.task_stances.get(task.activity, self.layouts[layout_group(task)])]

    def base_pose(self, task: AratTask) -> BasePose | None:
        """Where this arm stands for ``task``, or None to use the pose saved with the task instance."""

        stance = self.stance(task)
        return None if stance is None else stance.base

    def layout(self, task: AratTask) -> Layout | None:
        """Where the apparatus stands on the table for ``task`` with this arm; None: as saved."""

        stance = self.stance(task)
        return None if stance is None else stance.layout

    def excluded_reason(self, task: AratTask) -> str | None:
        """Why this arm cannot do ``task`` from any of its stances, or None when it can."""

        return self.excluded_tasks.get(task.activity)

    def operator_yaw_deg(self, task: AratTask) -> float:
        """Yaw of the operator's station for ``task``: the stance yaw (0 for the saved Franka pose)."""

        stance = self.stance(task)
        return 0.0 if stance is None else stance.base.yaw_deg

    def place_scene_cameras(self, task: AratTask, sensor_configs: Sequence[dict], stance: Stance | None = None) -> None:
        """Place the scene-frame cameras of ``sensor_configs`` for this arm's stance, in place.

        They turn with the stance yaw about the stance's focus; the arm's head camera, if it has
        one, goes to its place on the robot and looks at the focus. Under a layout the focus is where
        the layout puts the work area, and the cameras are carried there with it: each keeps the
        view of the apparatus it would have had of the saved layout from a stance turned the same
        way against it. Cameras on robot links are untouched, and without a stance (the Franka)
        nothing changes. ``stance`` overrides the task's stance (a replay passes the recorded one).
        """

        import numpy as np
        from scipy.spatial.transform import Rotation

        stance = self.stance(task) if stance is None else stance
        if stance is None:
            return
        turn = Rotation.from_euler("z", stance.base.yaw_deg, degrees=True)
        focus = np.asarray(stance.camera_focus, dtype=np.float64)
        moved = stance.layout is not None and not stance.layout.is_identity
        # the focus in the saved layout, which the rig's scene cameras were arranged around
        saved_focus = np.asarray(stance.layout.inverse().point(focus), dtype=np.float64) if moved else focus
        pivot = np.array([saved_focus[0], saved_focus[1], 0.0])
        carried = np.array([focus[0], focus[1], 0.0])
        for sensor in sensor_configs:
            if sensor.get("pose_frame") != "scene":
                continue
            if self.head_camera is not None and sensor.get("name") == self.head_camera[0]:
                position = turn.apply(self.head_camera[1]) + np.asarray(stance.base.position, dtype=np.float64)
                # USD camera axes: looks along -Z, +Y up, +X right; no roll
                forward = (focus - position) / np.linalg.norm(focus - position)
                right = np.cross(forward, [0.0, 0.0, 1.0])
                right /= np.linalg.norm(right)
                orientation = Rotation.from_matrix(np.column_stack([right, np.cross(right, forward), -forward]))
            elif stance.base.yaw_deg == 0.0 and not moved:
                continue
            else:
                position = turn.apply(np.asarray(sensor["position"], dtype=np.float64) - pivot) + carried
                orientation = turn * Rotation.from_quat(sensor["orientation"])
            sensor["position"] = [float(value) for value in position]
            sensor["orientation"] = [float(value) for value in orientation.as_quat()]

    def _reset_poses(self, task: AratTask) -> Mapping[str, tuple[float, ...]]:
        stance = self.stance(task)
        if stance is None:
            return {name: tuple(pose[:ARM_DOFS]) for name, pose in RESET_POSES.items()}
        return stance.reset_poses

    def reset_pose_names(self, task: AratTask) -> tuple[str, ...]:
        return tuple(self._reset_poses(task))

    def reset_arm_joint_positions(self, task: AratTask, name: str | None = None) -> tuple[float, ...]:
        poses = self._reset_poses(task)
        name = self.default_reset_pose if name is None else name
        try:
            return poses[name]
        except KeyError as error:
            raise ValueError(
                f"Arm {self.key!r} has no reset pose {name!r} for {task.activity}; expected one of: {', '.join(poses)}"
            ) from error

    def reset_joint_positions(self, task: AratTask, name: str | None = None) -> list[float]:
        """The named start configuration for ``task``, followed by an open hand."""

        return list(self.reset_arm_joint_positions(task, name)) + [0.0] * len(SHARPA_ACTION_JOINTS)


@dataclass(frozen=True)
class AratArms:
    default_arm: str
    arms: Mapping[str, AratArm]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.arms)

    def arm(self, key: str | None = None) -> AratArm:
        name = self.default_arm if key is None else key
        try:
            return self.arms[name]
        except KeyError as error:
            raise ValueError(f"Unknown ARAT arm {name!r}; expected one of: {', '.join(self.arms)}") from error

    def by_robot(self, model: str, name: str | None = None) -> AratArm:
        """The arm a recording's robot config belongs to."""

        for arm in self.arms.values():
            if arm.bench.robot.model == model and (name is None or arm.robot_name == name):
                return arm
        known = ", ".join(f"{arm.key} ({arm.bench.robot.model}/{arm.robot_name})" for arm in self.arms.values())
        raise ValueError(f"No ARAT arm uses robot model {model!r}" + (f" named {name!r}" if name else "") + f"; known: {known}")


def _vector(value, length: int, description: str) -> tuple[float, ...]:
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be {length} numbers") from error
    if len(vector) != length or not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{description} must be {length} finite numbers")
    return vector


def _stance(name: str, raw: Mapping, where: str, default_reset_pose: str) -> Stance:
    poses_raw = raw.get("reset_poses")
    if not isinstance(poses_raw, dict) or default_reset_pose not in poses_raw:
        raise ValueError(f"{where}.reset_poses must include the default pose {default_reset_pose!r}")
    touches = {}
    for activity, touch in (raw.get("touches") or {}).items():
        waypoints = tuple(
            (_vector(waypoint[:3], 3, f"{where}.touches.{activity}"), _vector(waypoint[3:], 4, f"{where}.touches.{activity}"))
            for waypoint in touch["waypoints"]
        )
        if not waypoints:
            raise ValueError(f"{where}.touches.{activity} needs at least one waypoint")
        touches[str(activity)] = Touch(_vector(touch["contact_in_palm"], 3, f"{where}.touches.{activity}.contact_in_palm"), waypoints)
    try:
        layout = Layout.from_json(raw.get("layout"))
    except ValueError as error:
        raise ValueError(f"{where}.layout: {error}") from error
    if layout is not None and touches:
        raise ValueError(f"{where}: a stance with mannequin touches cannot move the apparatus (layout)")
    return Stance(
        name=name,
        description=str(raw.get("description", "")),
        base=BasePose(_vector(raw["position"], 3, f"{where}.position"), float(raw["yaw_deg"])),
        camera_focus=_vector(raw["camera_focus"], 3, f"{where}.camera_focus"),
        reset_poses={str(pose): _vector(value, ARM_DOFS, f"{where}.reset_poses.{pose}") for pose, value in poses_raw.items()},
        touches=touches,
        layout=layout,
    )


def load_arat_arms(path: str | Path = ARMS_PATH) -> AratArms:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or raw.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"{path} must use schema_version {SUPPORTED_SCHEMA_VERSION}")
    arms_raw = raw.get("arms")
    if not isinstance(arms_raw, dict) or not arms_raw:
        raise ValueError(f"{path} must define at least one arm")
    bench_arms = load_wrist_bench_arms()
    arms: dict[str, AratArm] = {}
    for key, arm_raw in arms_raw.items():
        where = f"arms.{key}"
        bench = bench_arms.arm(str(arm_raw.get("bench_arm", key)))
        if arm_raw.get("placement") == "task_instance":
            # The arm the instances were laid out for: saved pose, the ARAT reset poses.
            stances, layouts, task_stances, excluded = None, {}, {}, {}
            default_reset_pose = DEFAULT_RESET_POSE
        else:
            default_reset_pose = str(arm_raw["default_reset_pose"])
            stances_raw = arm_raw.get("stances")
            if not isinstance(stances_raw, dict) or not stances_raw:
                raise ValueError(f"{where}.stances must define at least one stance (or use placement: task_instance)")
            stances = {str(name): _stance(str(name), value, f"{where}.stances.{name}", default_reset_pose) for name, value in stances_raw.items()}
            layouts = {str(group): str(name) for group, name in (arm_raw.get("layouts") or {}).items()}
            if set(layouts) != set(LAYOUT_GROUPS):
                raise ValueError(f"{where}.layouts must name a stance for exactly: {', '.join(LAYOUT_GROUPS)}")
            task_stances = {str(activity): str(name) for activity, name in (arm_raw.get("task_stances") or {}).items()}
            for name in (*layouts.values(), *task_stances.values()):
                if name not in stances:
                    raise ValueError(f"{where}: unknown stance {name!r}; expected one of: {', '.join(stances)}")
            if stances[layouts["mannequin"]].layout is not None:
                raise ValueError(f"{where}: the mannequin stance cannot move the apparatus (layout); there is none")
            excluded = {str(activity): str(reason) for activity, reason in (arm_raw.get("excluded_tasks") or {}).items()}
        floor_boxes = tuple(
            (str(box["name"]), _vector(box["centre"], 2, f"{where}.floor_boxes"), _vector(box["half_size"], 2, f"{where}.floor_boxes"),
             bool(box.get("taller_than_table", False)))
            for box in arm_raw.get("floor_boxes", ())
        )
        fixed_points = tuple(_vector(point, 2, f"{where}.fixed_points") for point in arm_raw.get("fixed_points", ()))
        head_camera = arm_raw.get("head_camera")
        if head_camera is not None:
            head_camera = (str(head_camera["sensor_name"]), _vector(head_camera["position"], 3, f"{where}.head_camera.position"))
        arms[key] = AratArm(
            key=key,
            description=str(arm_raw.get("description", bench.description)),
            robot_name=str(arm_raw["robot_name"]),
            bench=bench,
            stances=stances,
            layouts=layouts,
            task_stances=task_stances,
            excluded_tasks=excluded,
            default_reset_pose=default_reset_pose,
            floor_boxes=floor_boxes,
            fixed_points=fixed_points,
            wrist_camera_pods=bool(arm_raw.get("wrist_camera_pods", True)),
            head_camera=head_camera,
        )
    default_arm = str(raw.get("default_arm"))
    if default_arm not in arms:
        raise ValueError(f"default_arm {default_arm!r} is not one of: {', '.join(arms)}")
    names = [arm.robot_name for arm in arms.values()]
    if len(set(names)) != len(names):
        raise ValueError("ARAT arms must use distinct robot names")
    return AratArms(default_arm=default_arm, arms=arms)


# --------------------------------------------------------------------------- reach


@dataclass(frozen=True)
class TaskReach:
    """Whether an arm with its own stances reaches one task's key points from where it stands."""

    task: AratTask
    stance: Stance
    points: tuple
    conflicts: tuple[str, ...]
    excluded_reason: str | None = None

    @property
    def base(self) -> BasePose:
        return self.stance.base

    @property
    def layout(self) -> Layout | None:
        return self.stance.layout

    @property
    def reachable(self) -> bool:
        return self.excluded_reason is None and not self.conflicts and all(point.reachable for point in self.points)

    @property
    def worst_grasp_margin(self) -> float:
        return min(point.grasp_margin for point in self.points)

    @property
    def worst_palm_margin(self) -> float:
        return min(point.palm_margin for point in self.points)

    def describe(self) -> str:
        parts = [
            f"{point.key_point.name} {point.distance:.2f} m from the shoulder (palm margin {point.palm_margin:+.3f} m, "
            f"with the hand's reach {point.grasp_margin:+.3f} m)"
            + ("" if point.in_height_band else " OUTSIDE the height band")
            for point in self.points
        ]
        text = f"{self.task.activity} [stance {self.stance.name}]: " + "; ".join(parts)
        if self.conflicts:
            text += " | cannot be set up: " + ", ".join(self.conflicts)
        if self.excluded_reason:
            text += f" | EXCLUDED: {self.excluded_reason}"
        return text


def task_reach(arm: AratArm, task: AratTask) -> TaskReach | None:
    """The reach of ``arm`` for ``task`` from its stance; None for the arm the instances were laid out for.

    Static part only (reach sphere, height band, floor footprint, the exclusion list). Whether the
    palm can be led along the task's path is ``task_path_reach`` (needs Pinocchio).
    """

    from dex_teleop.arat.placement import footprint_conflicts, layout_conflicts, palm_sphere_reach, task_sphere_reach

    stance = arm.stance(task)
    if stance is None:
        return None
    touch = stance.touches.get(task.activity)
    if touch is not None:
        # The palm position of the touch is exact: test it, not the region's centre.
        points = (palm_sphere_reach("palm at the touch", touch.waypoints[-1][0], stance.base, arm.workspace),)
    else:
        points = task_sphere_reach(task, stance.base, arm.workspace, stance.layout)
    return TaskReach(
        task=task,
        stance=stance,
        points=points,
        conflicts=(
            *footprint_conflicts(task, stance.base, arm.floor_boxes, arm.fixed_points),
            *layout_conflicts(task, stance.layout),
        ),
        excluded_reason=arm.excluded_reason(task),
    )


def task_path_reach(arm: AratArm, task: AratTask, ik, reset_pose: str | None = None):
    """``placement.PathReach`` of leading the palm along ``task``'s path from the stance's start configuration.

    Table tasks: above the pick point, onto it, up, across at transport height, onto the target, with
    the start orientation. Mannequin tasks: the stance's recorded touch. None without a stance; raises
    ``ValueError`` for a mannequin task whose stance has no touch.
    """

    from scipy.spatial.transform import Rotation

    from dex_teleop.arat.placement import path_reach

    stance = arm.stance(task)
    if stance is None:
        return None
    touch = None
    if layout_group(task) == "mannequin":
        recorded = stance.touches.get(task.activity)
        if recorded is None:
            raise ValueError(f"Stance {stance.name!r} of arm {arm.key!r} has no touch for {task.activity}")
        touch = [(position, Rotation.from_quat(orientation)) for position, orientation in recorded.waypoints]
    return path_reach(task, stance.base, ik, arm.reset_arm_joint_positions(task, reset_pose), touch, stance.layout)


# --------------------------------------------------------------------------- provenance

ARM_PROVENANCE_ATTRIBUTE = "dex_teleop_arm"


def arm_provenance(arm: AratArm, task: AratTask, reset_pose: str | None = None) -> dict:
    """What a recording must say about the arm it was made with (JSON-safe)."""

    stance = arm.stance(task)
    name = arm.default_reset_pose if reset_pose is None else reset_pose
    robot = arm.bench.robot
    return {
        "arm": arm.key,
        "robot_model": robot.model,
        "robot_end_effector": robot.end_effector,
        "robot_name": arm.robot_name,
        "layout_group": layout_group(task),
        "stance": None if stance is None else stance.name,
        "base_pose_source": "task instance (saved robot pose)" if stance is None else "arat/arms.yaml stance",
        "base_position": None if stance is None else list(stance.base.position),
        "base_yaw_deg": None if stance is None else stance.base.yaw_deg,
        "operator_yaw_deg": arm.operator_yaw_deg(task),
        # Where the apparatus stands on the table: None is the layout saved with the task instance.
        "layout": None if stance is None or stance.layout is None else stance.layout.to_json(),
        "scene_camera_focus": None if stance is None else list(stance.camera_focus),
        "head_camera": None if stance is None or arm.head_camera is None else [arm.head_camera[0], list(arm.head_camera[1])],
        "excluded_reason": arm.excluded_reason(task),
        "reset_pose": name,
        "reset_arm_joint_positions": list(arm.reset_arm_joint_positions(task, name)),
        "workspace": {
            "shoulder_position": list(arm.workspace.shoulder_position),
            "shoulder_reach": arm.workspace.shoulder_reach,
            "minimum_z": arm.workspace.minimum_z,
            "maximum_z": arm.workspace.maximum_z,
        },
    }


def write_arm_provenance(recording_path: str | Path, provenance: Mapping) -> None:
    import json

    import h5py

    with h5py.File(Path(recording_path), "r+") as recording:
        recording["data"].attrs[ARM_PROVENANCE_ATTRIBUTE] = json.dumps(dict(provenance), sort_keys=True)


def recording_arm(recording_path: str | Path, arms: AratArms | None = None) -> tuple[AratArm, dict | None]:
    """The arm a recording was made with, from its saved robot config and its arm provenance.

    Recordings made before ``--arm`` existed carry no provenance attribute; their robot config
    identifies the Franka. Raises ``ValueError`` when the two disagree or the robot is unknown.
    """

    import json

    import h5py

    arms = load_arat_arms() if arms is None else arms
    with h5py.File(Path(recording_path), "r") as recording:
        attributes = recording["data"].attrs
        config = attributes["config"]
        config = json.loads(config.decode("utf-8") if isinstance(config, bytes) else config)
        encoded = attributes.get(ARM_PROVENANCE_ATTRIBUTE)
    robots = config.get("robots") or []
    if len(robots) != 1:
        raise ValueError(f"{recording_path}: expected exactly one recorded robot, found {len(robots)}")
    arm = arms.by_robot(str(robots[0].get("model")), robots[0].get("name"))
    provenance = None
    if encoded is not None:
        provenance = json.loads(encoded.decode("utf-8") if isinstance(encoded, bytes) else encoded)
        if provenance.get("arm") != arm.key:
            raise ValueError(
                f"{recording_path}: its arm provenance says {provenance.get('arm')!r} but its saved robot "
                f"is {robots[0].get('model')}/{robots[0].get('name')} ({arm.key!r}); the recording is inconsistent"
            )
    return arm, provenance


def recorded_stance(recording_path: str | Path, arm: AratArm, task: AratTask, provenance: Mapping | None) -> Stance | None:
    """The stance a recording was made in, checked against ``arms.yaml`` and the recorded scene.

    None for the arm the instances were laid out for. Raises ``ValueError`` when the recorded stance
    is not defined any more, when its base pose or its layout differs from the definition (the
    cameras and the reach it documents would not be the recording's), or when the recorded scene
    does not have the apparatus where the recorded layout puts it.
    """

    import json

    import h5py

    if arm.stances is None:
        return None
    if provenance is None or provenance.get("stance") is None:
        raise ValueError(f"{recording_path}: recorded with arm {arm.key!r} but without a stance in its arm provenance")
    name = str(provenance["stance"])
    if name not in arm.stances:
        raise ValueError(
            f"{recording_path}: recorded in stance {name!r} of arm {arm.key!r}, which arat/arms.yaml no longer "
            f"defines (it has: {', '.join(arm.stances)})"
        )
    stance = arm.stances[name]
    recorded_layout = Layout.from_json(provenance.get("layout"))  # absent before layouts existed: as saved
    defined_layout = stance.layout
    same_layout = (recorded_layout.to_json() if recorded_layout is not None and not recorded_layout.is_identity else None) == (
        defined_layout.to_json() if defined_layout is not None and not defined_layout.is_identity else None
    )
    position, yaw = provenance.get("base_position"), provenance.get("base_yaw_deg")
    same_base = (
        position is not None
        and yaw is not None
        and all(abs(float(a) - float(b)) < 1.0e-6 for a, b in zip(position, stance.base.position))
        and abs(float(yaw) - stance.base.yaw_deg) < 1.0e-6
    )
    if not same_layout or not same_base:
        raise ValueError(
            f"{recording_path}: recorded in stance {name!r} with base {position} yaw {yaw} and apparatus layout "
            f"{provenance.get('layout')}, but arat/arms.yaml now defines it with base {list(stance.base.position)} yaw "
            f"{stance.base.yaw_deg:g} and layout {None if defined_layout is None else defined_layout.to_json()}. "
            "Restore that stance definition to replay this recording."
        )
    # The scene the recording will be replayed in must be the one that layout produces.
    from dex_teleop.arat.layout import moved_object_names
    from dex_teleop.arat.scene import get_task_scene_data

    with h5py.File(Path(recording_path), "r") as recording:
        encoded = recording["data"].attrs.get("scene_file")
    if encoded is not None:
        recorded_scene = json.loads(encoded.decode("utf-8") if isinstance(encoded, bytes) else encoded)
        expected = get_task_scene_data(task, layout=recorded_layout)
        recorded_states = recorded_scene["state"]["registry"]["object_registry"]
        expected_states = expected["state"]["registry"]["object_registry"]
        for object_name in moved_object_names(expected):
            if object_name not in recorded_states:
                continue
            init = recorded_scene["objects_info"]["init_info"].get(object_name, {}).get("args", {})
            if not init.get("fixed_base"):
                continue  # free objects are where the operator left them
            recorded_position = recorded_states[object_name]["root_link"]["pos"]
            expected_position = expected_states[object_name]["root_link"]["pos"]
            if max(abs(float(a) - float(b)) for a, b in zip(recorded_position, expected_position)) > 1.0e-3:
                raise ValueError(
                    f"{recording_path}: its scene has {object_name} at {[round(float(v), 4) for v in recorded_position]}, "
                    f"but the recorded layout {provenance.get('layout')} puts it at "
                    f"{[round(float(v), 4) for v in expected_position]}; the recording is inconsistent"
                )
    return stance


__all__ = [
    "ARMS_PATH",
    "ARM_PROVENANCE_ATTRIBUTE",
    "AratArm",
    "AratArms",
    "LAYOUT_GROUPS",
    "Layout",
    "Stance",
    "TaskReach",
    "Touch",
    "arm_provenance",
    "layout_group",
    "load_arat_arms",
    "recorded_stance",
    "recording_arm",
    "task_path_reach",
    "task_reach",
    "write_arm_provenance",
]
