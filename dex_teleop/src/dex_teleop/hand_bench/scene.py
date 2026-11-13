"""Hand-bench scene: a fixed right Sharpa hand above a table, defined in YAML."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml

from dex_teleop.arat.camera_rig import build_camera_sensor_configs, load_camera_rig
from dex_teleop.arat.scene import ROBOT_DATASET_NAME, ROBOT_END_EFFECTOR, ROBOT_MODEL
from dex_teleop.hands import SHARPA_ACTION_JOINTS


HAND_BENCH_PATH = Path(__file__).with_name("hand_bench.yaml")
SUPPORTED_SCHEMA_VERSION = 1
DEFAULT_CAMERA_RIG = "hand_bench"
CAMERA_LAYOUT = "teleop"
ROBOT_NAME = "hand_bench_sharpa_right"
ROBOT_COMPOSED_MODEL = f"{ROBOT_MODEL}_{ROBOT_END_EFFECTOR}"
ROBOT_PRIM_PATH = f"/controllable__{ROBOT_COMPOSED_MODEL}__{ROBOT_NAME}"
TABLE_NAME = "table"
FRANKA_ARM_DOFS = 7

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DATASETS_ROOT = _REPOSITORY_ROOT / "datasets"


def _vector(value, length: int, description: str) -> tuple[float, ...]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != length
        or not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
    ):
        raise ValueError(f"{description} must contain {length} finite numbers")
    return tuple(float(item) for item in value)


def _positive(value, description: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{description} must be a positive finite number")
    return float(value)


@dataclass(frozen=True)
class ExpectedHandPose:
    """Where the Sharpa palm base must land once the hidden arm holds its configuration."""

    link: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    position_tolerance_m: float
    orientation_tolerance_deg: float


@dataclass(frozen=True)
class HandBenchTable:
    dataset_name: str
    category: str
    model: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    tabletop_height: float
    height_tolerance_m: float


@dataclass(frozen=True)
class HandBenchRobot:
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    arm_joint_positions: tuple[float, ...]
    hidden_prim_prefixes: tuple[str, ...]
    expected_hand_pose: ExpectedHandPose


@dataclass(frozen=True)
class HandBenchScene:
    floor_plane_color: tuple[float, float, float]
    table: HandBenchTable
    lights: tuple[dict, ...]
    robot: HandBenchRobot


# ---------------------------------------------------------------------------
# Wrist bench arms (hand_bench.yaml ``wrist_bench``)


@dataclass(frozen=True)
class WristBenchRobot:
    """The OmniGibson robot that carries the hand on the wrist bench, and where it stands."""

    model: str
    dataset_name: str
    end_effector: str | None
    composed_model: str
    name: str
    arm_name: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]

    @property
    def prim_path(self) -> str:
        return f"/controllable__{self.composed_model}__{self.name}"

    @property
    def arm_controller(self) -> str:
        return f"arm_{self.arm_name}"

    @property
    def gripper_controller(self) -> str:
        return f"gripper_{self.arm_name}"

    def environment_config(self) -> dict:
        """The ``robots`` entry fields that identify and place this robot."""

        config = {
            "model": self.model,
            "dataset_name": self.dataset_name,
            "name": self.name,
            "position": list(self.position),
            "orientation": list(self.orientation_xyzw),
        }
        if self.end_effector is not None:
            config["end_effector"] = self.end_effector
        return config


@dataclass(frozen=True)
class WristBenchWorkspace:
    """``SharpaActionAdapter`` clamps for one arm, in the robot frame."""

    shoulder_position: tuple[float, float, float]
    shoulder_reach: float
    minimum_z: float
    maximum_z: float


@dataclass(frozen=True)
class WristBenchArm:
    """One selectable arm of the wrist bench."""

    key: str
    description: str
    robot: WristBenchRobot
    palm_link: str
    reset_poses: Mapping[str, tuple[float, ...]]
    default_reset_pose: str
    workspace: WristBenchWorkspace
    camera_rig: str

    def reset_joint_positions(self, reset_pose: str | None = None) -> list[float]:
        """The named arm configuration followed by an open Sharpa hand."""

        name = self.default_reset_pose if reset_pose is None else reset_pose
        try:
            arm = self.reset_poses[name]
        except KeyError as error:
            known = ", ".join(sorted(self.reset_poses))
            raise ValueError(f"Arm {self.key!r} has no reset pose {name!r}; expected one of: {known}") from error
        return list(arm) + [0.0] * len(SHARPA_ACTION_JOINTS)


@dataclass(frozen=True)
class WristBenchArms:
    default_arm: str
    arms: Mapping[str, WristBenchArm]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.arms)

    def arm(self, key: str | None = None) -> WristBenchArm:
        name = self.default_arm if key is None else key
        try:
            return self.arms[name]
        except KeyError as error:
            raise ValueError(f"Unknown wrist bench arm {name!r}; expected one of: {', '.join(self.arms)}") from error


def load_hand_bench_scene(path: str | Path = HAND_BENCH_PATH) -> HandBenchScene:
    """Load and validate the packaged hand-bench scene definition."""

    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or raw.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"{path} must use schema_version {SUPPORTED_SCHEMA_VERSION}")
    scene = raw.get("scene")
    if not isinstance(scene, dict):
        raise ValueError(f"{path} must define a scene mapping")

    table_raw = scene.get("table")
    if not isinstance(table_raw, dict):
        raise ValueError("Hand bench scene must define a table")
    for field in ("dataset_name", "category", "model"):
        if not isinstance(table_raw.get(field), str) or not table_raw[field]:
            raise ValueError(f"Hand bench table {field} must be a non-empty string")
    table = HandBenchTable(
        dataset_name=table_raw["dataset_name"],
        category=table_raw["category"],
        model=table_raw["model"],
        position=_vector(table_raw.get("position"), 3, "Table position"),
        orientation_xyzw=_vector(table_raw.get("orientation_xyzw"), 4, "Table orientation_xyzw"),
        tabletop_height=_positive(table_raw.get("tabletop_height"), "Table tabletop_height"),
        height_tolerance_m=_positive(table_raw.get("height_tolerance_m", 0.01), "Table height_tolerance_m"),
    )

    lights_raw = scene.get("lights", [])
    if not isinstance(lights_raw, list):
        raise ValueError("Hand bench lights must be a list")
    lights = []
    light_names = set()
    for light in lights_raw:
        if not isinstance(light, dict) or not isinstance(light.get("name"), str) or not light["name"]:
            raise ValueError("Every hand bench light must be a mapping with a name")
        if light["name"] in light_names:
            raise ValueError(f"Hand bench light name {light['name']!r} is repeated")
        light_names.add(light["name"])
        if not isinstance(light.get("light_type"), str):
            raise ValueError(f"Light {light['name']!r} must define light_type")
        lights.append(
            {
                "name": light["name"],
                "light_type": light["light_type"],
                "radius": _positive(light.get("radius", 1.0), f"Light {light['name']!r} radius"),
                "intensity": _positive(light.get("intensity", 50000.0), f"Light {light['name']!r} intensity"),
                "position": list(_vector(light.get("position"), 3, f"Light {light['name']!r} position")),
            }
        )

    robot_raw = scene.get("robot")
    if not isinstance(robot_raw, dict):
        raise ValueError("Hand bench scene must define a robot")
    prefixes = robot_raw.get("hidden_prim_prefixes", [])
    if not isinstance(prefixes, list) or not all(isinstance(prefix, str) and prefix for prefix in prefixes):
        raise ValueError("Robot hidden_prim_prefixes must be a list of non-empty strings")
    if any("right_hand".startswith(prefix) or prefix.startswith("right_hand") for prefix in prefixes):
        raise ValueError("hidden_prim_prefixes may not hide the Sharpa hand itself")
    expected_raw = robot_raw.get("expected_hand_pose")
    if not isinstance(expected_raw, dict) or not isinstance(expected_raw.get("link"), str):
        raise ValueError("Robot expected_hand_pose must name the palm link")
    expected = ExpectedHandPose(
        link=expected_raw["link"],
        position=_vector(expected_raw.get("position"), 3, "Expected hand position"),
        orientation_xyzw=_vector(expected_raw.get("orientation_xyzw"), 4, "Expected hand orientation_xyzw"),
        position_tolerance_m=_positive(expected_raw.get("position_tolerance_m", 0.01), "Hand position tolerance"),
        orientation_tolerance_deg=_positive(
            expected_raw.get("orientation_tolerance_deg", 2.0), "Hand orientation tolerance"
        ),
    )
    robot = HandBenchRobot(
        position=_vector(robot_raw.get("position"), 3, "Robot position"),
        orientation_xyzw=_vector(robot_raw.get("orientation_xyzw"), 4, "Robot orientation_xyzw"),
        arm_joint_positions=_vector(
            robot_raw.get("arm_joint_positions"), FRANKA_ARM_DOFS, "Robot arm_joint_positions"
        ),
        hidden_prim_prefixes=tuple(prefixes),
        expected_hand_pose=expected,
    )
    return HandBenchScene(
        floor_plane_color=_vector(scene.get("floor_plane_color", [0.5, 0.5, 0.5]), 3, "floor_plane_color"),
        table=table,
        lights=tuple(lights),
        robot=robot,
    )


def _identifier(value, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string")
    return value


def load_wrist_bench_arms(path: str | Path = HAND_BENCH_PATH) -> WristBenchArms:
    """Load and validate the wrist bench's selectable arms (``wrist_bench`` in hand_bench.yaml)."""

    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or raw.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"{path} must use schema_version {SUPPORTED_SCHEMA_VERSION}")
    section = raw.get("wrist_bench")
    if not isinstance(section, dict):
        raise ValueError(f"{path} must define a wrist_bench mapping")
    arms_raw = section.get("arms")
    if not isinstance(arms_raw, dict) or not arms_raw:
        raise ValueError("wrist_bench.arms must map at least one arm name to its definition")
    default_arm = _identifier(section.get("arm"), "wrist_bench.arm")
    if default_arm not in arms_raw:
        raise ValueError(f"wrist_bench.arm {default_arm!r} is not one of the defined arms: {', '.join(arms_raw)}")

    arms: dict[str, WristBenchArm] = {}
    robot_names = set()
    for key, arm_raw in arms_raw.items():
        where = f"wrist_bench.arms.{key}"
        if not isinstance(key, str) or not key or not isinstance(arm_raw, dict):
            raise ValueError("Every wrist bench arm must be a named mapping")
        robot_raw = arm_raw.get("robot")
        if not isinstance(robot_raw, dict):
            raise ValueError(f"{where}.robot must be a mapping")
        end_effector = robot_raw.get("end_effector")
        if end_effector is not None:
            end_effector = _identifier(end_effector, f"{where}.robot.end_effector")
        arm_name = robot_raw.get("arm_name")
        if isinstance(arm_name, int) and not isinstance(arm_name, bool):
            arm_name = str(arm_name)
        robot = WristBenchRobot(
            model=_identifier(robot_raw.get("model"), f"{where}.robot.model"),
            dataset_name=_identifier(robot_raw.get("dataset_name"), f"{where}.robot.dataset_name"),
            end_effector=end_effector,
            composed_model=_identifier(robot_raw.get("composed_model"), f"{where}.robot.composed_model"),
            name=_identifier(robot_raw.get("name"), f"{where}.robot.name"),
            arm_name=_identifier(arm_name, f"{where}.robot.arm_name"),
            position=_vector(robot_raw.get("position"), 3, f"{where}.robot.position"),
            orientation_xyzw=_vector(robot_raw.get("orientation_xyzw"), 4, f"{where}.robot.orientation_xyzw"),
        )
        if robot.name in robot_names:
            raise ValueError(f"{where}.robot.name {robot.name!r} is used by another arm")
        robot_names.add(robot.name)

        poses_raw = arm_raw.get("reset_poses")
        if not isinstance(poses_raw, dict) or not poses_raw:
            raise ValueError(f"{where}.reset_poses must map at least one pose name to 7 joint positions")
        reset_poses = {
            _identifier(name, f"{where}.reset_poses key"): _vector(values, FRANKA_ARM_DOFS, f"{where}.reset_poses.{name}")
            for name, values in poses_raw.items()
        }
        default_reset_pose = _identifier(arm_raw.get("default_reset_pose"), f"{where}.default_reset_pose")
        if default_reset_pose not in reset_poses:
            raise ValueError(
                f"{where}.default_reset_pose {default_reset_pose!r} is not one of: {', '.join(reset_poses)}"
            )

        workspace_raw = arm_raw.get("workspace")
        if not isinstance(workspace_raw, dict):
            raise ValueError(f"{where}.workspace must be a mapping")
        workspace = WristBenchWorkspace(
            shoulder_position=_vector(workspace_raw.get("shoulder_position"), 3, f"{where}.workspace.shoulder_position"),
            shoulder_reach=_positive(workspace_raw.get("shoulder_reach"), f"{where}.workspace.shoulder_reach"),
            minimum_z=_positive(workspace_raw.get("minimum_z"), f"{where}.workspace.minimum_z"),
            maximum_z=_positive(workspace_raw.get("maximum_z"), f"{where}.workspace.maximum_z"),
        )
        if workspace.maximum_z <= workspace.minimum_z:
            raise ValueError(f"{where}.workspace.maximum_z must exceed minimum_z")

        arms[key] = WristBenchArm(
            key=key,
            description=_identifier(arm_raw.get("description", key), f"{where}.description"),
            robot=robot,
            palm_link=_identifier(arm_raw.get("palm_link"), f"{where}.palm_link"),
            reset_poses=reset_poses,
            default_reset_pose=default_reset_pose,
            workspace=workspace,
            camera_rig=_identifier(arm_raw.get("camera_rig"), f"{where}.camera_rig"),
        )
    return WristBenchArms(default_arm=default_arm, arms=arms)


def reset_joint_positions(scene: HandBenchScene) -> list[float]:
    """The held Franka configuration followed by an open Sharpa hand."""

    return list(scene.robot.arm_joint_positions) + [0.0] * len(SHARPA_ACTION_JOINTS)


def hidden_prim_names(names: Iterable[str], prefixes: Sequence[str]) -> tuple[str, ...]:
    """Return the link or child-prim names whose visuals the bench hides, in the given order."""

    return tuple(name for name in names if any(name.startswith(prefix) for prefix in prefixes))


def hand_pose_error(
    position: Sequence[float],
    quaternion_xyzw: Sequence[float],
    expected: ExpectedHandPose,
) -> tuple[float, float]:
    """Return (metres, degrees) between a measured palm pose and the expected one."""

    measured_position = np.asarray(position, dtype=np.float64).reshape(3)
    measured_quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    measured_quaternion = measured_quaternion / np.linalg.norm(measured_quaternion)
    expected_quaternion = np.asarray(expected.orientation_xyzw, dtype=np.float64)
    expected_quaternion = expected_quaternion / np.linalg.norm(expected_quaternion)
    translation = float(np.linalg.norm(measured_position - np.asarray(expected.position)))
    dot = min(1.0, abs(float(np.dot(measured_quaternion, expected_quaternion))))
    return translation, math.degrees(2.0 * math.acos(dot))


def _load_robot_definition(dataset_name: str, model: str) -> tuple[Path, dict]:
    definition_path = _DATASETS_ROOT / dataset_name / "models" / model / f"{model}.yaml"
    try:
        with definition_path.open("r", encoding="utf-8") as stream:
            definition = yaml.safe_load(stream)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing robot definition: {definition_path}") from error
    if not isinstance(definition, dict):
        raise ValueError(f"{definition_path} is not a robot definition mapping")
    return definition_path, definition


def robot_usd_path(
    model: str = ROBOT_MODEL,
    dataset_name: str = ROBOT_DATASET_NAME,
    end_effector: str | None = ROBOT_END_EFFECTOR,
    *,
    composed_model: str | None = None,
) -> Path:
    """Resolve the USD an OmniGibson robot definition declares (by default the hand bench's Franka + Sharpa).

    With an ``end_effector`` the USD comes from that end effector's entry, as OmniGibson resolves
    it; otherwise from the model's top-level ``usd_path`` (or OmniGibson's standard location).
    ``composed_model`` is checked against the model OmniGibson embeds in the robot's prim path.
    """

    definition_path, definition = _load_robot_definition(dataset_name, model)
    if end_effector is not None:
        try:
            entry = definition["manipulation"]["end_effectors"][end_effector]
        except (KeyError, TypeError) as error:
            raise ValueError(f"{definition_path} does not declare the {end_effector} end effector") from error
        usd_path = entry.get("usd_path")
        prim_model = entry.get("model", model)
        if usd_path is None:
            raise ValueError(f"{definition_path}: end effector {end_effector} declares no usd_path")
    else:
        if (definition.get("manipulation") or {}).get("supported_end_effector"):
            raise ValueError(f"{definition_path} has end-effector variants; select one with end_effector")
        usd_path = definition.get("usd_path") or f"models/{model}/usd/{model}.usda"
        prim_model = model
    if composed_model is not None and composed_model != prim_model:
        raise ValueError(
            f"{definition_path}: OmniGibson names this robot's prim after {prim_model!r}, "
            f"but the bench configuration says composed_model: {composed_model!r}"
        )
    return _DATASETS_ROOT / dataset_name / usd_path


def _table_usd_path(table: HandBenchTable) -> Path:
    return _DATASETS_ROOT / table.dataset_name / "objects" / table.category / table.model / "usd" / f"{table.model}.usd"


def validate_runtime_assets(scene: HandBenchScene) -> None:
    """Fail early when the table or the Franka + Sharpa asset is unavailable."""

    required = [_table_usd_path(scene.table), robot_usd_path()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required hand bench assets:\n" + "\n".join(missing))


def validate_wrist_bench_assets(scene: HandBenchScene, arm: WristBenchArm) -> None:
    """Fail early when the table or the selected arm's robot asset is unavailable."""

    robot = arm.robot
    required = [
        _table_usd_path(scene.table),
        robot_usd_path(robot.model, robot.dataset_name, robot.end_effector, composed_model=robot.composed_model),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        hint = ""
        if arm.key == "tianji":
            hint = (
                "\nBuild the Tianji + Sharpa robot with dex_teleop/scripts/build_tianji_sharpa_right.py "
                "(see dex_teleop/docs/tianji-m6-sharpa-mount.md)"
            )
        raise FileNotFoundError(f"Missing required wrist bench assets for arm {arm.key!r}:\n" + "\n".join(missing) + hint)


def build_hand_bench_config(
    scene: HandBenchScene,
    *,
    camera_rig_name: str = DEFAULT_CAMERA_RIG,
    layout_name: str = CAMERA_LAYOUT,
    viewport_name: str | None = "Viewport",
    sensor_modalities: Sequence[str] | None = None,
) -> dict:
    """Build the OmniGibson environment configuration for the hand bench.

    The Franka arm is held by a ``NullJointController`` at the scene's solved
    configuration so ``env.action_space`` is exactly the 22 Sharpa finger
    joints; the wrist therefore cannot be commanded at all.
    """

    import torch as th

    camera_rig = load_camera_rig(camera_rig_name)
    sensors = build_camera_sensor_configs(
        camera_rig,
        layout_name,
        robot_prim_path=ROBOT_PRIM_PATH,
        viewport_name=viewport_name,
    )
    if sensor_modalities is not None:
        for sensor in sensors:
            sensor["modalities"] = list(sensor_modalities)
    table = scene.table
    robot = scene.robot
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
            # The dome light would over-light the floor that serves as backdrop.
            "use_skybox": False,
        },
        "objects": objects,
        "robots": [
            {
                "model": ROBOT_MODEL,
                "dataset_name": ROBOT_DATASET_NAME,
                "end_effector": ROBOT_END_EFFECTOR,
                "name": ROBOT_NAME,
                "position": list(robot.position),
                "orientation": list(robot.orientation_xyzw),
                "grasping_direction": "upper",
                "grasping_mode": "physical",
                "obs_modalities": [],
                "action_normalize": False,
                "fixed_base": True,
                "self_collisions": False,
                "reset_joint_pos": reset_joint_positions(scene),
                "controller_config": {
                    "arm_0": {
                        "name": "NullJointController",
                        "motor_type": "position",
                        "default_goal": th.tensor(robot.arm_joint_positions, dtype=th.float32),
                        "command_input_limits": None,
                        "command_output_limits": None,
                    },
                    "gripper_0": {
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
