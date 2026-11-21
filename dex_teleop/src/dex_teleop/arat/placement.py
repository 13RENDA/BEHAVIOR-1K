"""Whether an arm standing at a given base pose reaches every ARAT task (simulator-free).

The saved ARAT task instances were laid out for the Franka (0.80 m reach). An arm with a different
reach needs its own base pose, and the claim "task X works with arm Y" needs evidence. This module
derives, from the task instances and the scorer's own target definitions, the world points each task
needs (where the tracked object starts, where it has to end up, or which mannequin region must be
touched) and tests them against the arm:

* ``sphere``: the clamp the teleoperation adapter applies to the palm link (reach sphere about the
  shoulder, height band), reported as a margin in metres. ``palm_margin`` is the conservative number
  (the palm link itself at the point); ``grasp_margin`` allows the palm to stay ``GRASP_REACH_M`` short
  of the point, since the palm link sits behind what the hand holds.
* ``ik``: an inverse-kinematics solution with joint limits for a palm pose that puts the hand's grasp
  point on the target, over a small family of approach orientations (needs Pinocchio and the arm's
  URDF). A necessary condition only: a stance chosen by it alone put the pick points 0.25 m from the
  Tianji's shoulder, where the arm has solutions but cannot be led to them.
* ``path``: the palm is led from the stance's start configuration along the task's path (above the
  pick point, onto it, up, across, onto the target; or the recorded touch of a mannequin region)
  the way OmniGibson's IK controller would lead it: one damped pseudo-inverse step per small target
  step, joint limits clipped, no null-space term. Reported: whether the palm follows and the
  smallest joint-limit margin on the way. This is the check a stance has to pass.

Collisions are not checked by any of these.

Every function that reads the task's objects takes an optional ``layout`` (``arat.layout.Layout``):
the rigid re-placement of the apparatus on the table that an arm's stance asks for. The key points,
the palm path and ``layout_conflicts`` (does the moved apparatus still stand on the tabletop) are
computed on the scene as that arm will load it; None is the saved layout.

Geometry sources: object poses from the task instance JSON; the ARAT box shelf and the hand's grasp
point are constants measured on the assets (see the comments); the mannequin's bounding box comes
from its asset metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from dex_teleop.arat.catalog import AratTask
from dex_teleop.arat.eval.regions import MANNEQUIN_REGIONS
from dex_teleop.arat.layout import Layout, moved_object_names
from dex_teleop.arat.scene import DATASET_NAME, get_task_scene_data

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_ASSET_ROOT = _REPOSITORY_ROOT / "datasets" / DATASET_NAME
_RUBRICS_PATH = Path(__file__).with_name("eval") / "rubrics.yaml"

# Top face of the ARAT box's base shell (the 37 cm shelf OnShelf tests), in the box's own frame.
# Measured on arat-assets-v1/objects/arat_box/aratbx/usd/aratbx.usd (base_link visual extents).
SHELF_X = (0.0, 0.546)
SHELF_Y = (-0.018, 0.132)
SHELF_TOP_Z = 0.349
SHELF_EDGE_INSET_M = 0.03  # OnShelf wants the object's centre over the shelf; stay 3 cm inside
# The scorer's mannequin faces world -X (arat.eval.live.MANNEQUIN_FRONT_DIR_XY).
MANNEQUIN_FRONT_DIR_XY = (-1.0, 0.0)
# The hand's grasp point in the palm-link frame (right_hand_C_MC: +X palm normal, +Y pinky->thumb,
# +Z wrist->fingers). With the hand open the thumb and index pads are at (0.015, 0.104, 0.091) and
# (0.001, 0.030, 0.174) (Pinocchio on the Sharpa model), 14.9 cm from the palm origin at their
# midpoint; a closed grasp sits nearer the palm.
GRASP_POINT_PALM = (0.05, 0.03, 0.10)
GRASP_REACH_M = 0.10
# Clearance of the palm path above what it reaches for (metres above the object's reference point).
LIFT_ABOVE_START_M = 0.03
LIFT_ABOVE_SHELF_M = 0.05
LIFT_ABOVE_TIN_M = 0.03
LIFT_ABOVE_PEG_M = 0.10
LIFT_ABOVE_POUR_M = 0.15
REACHABLE_MARGIN_M = 0.03


@dataclass(frozen=True)
class KeyPoint:
    """A world point a task needs the hand at, and from which side."""

    name: str
    position: tuple[float, float, float]
    approach: str  # "top": palm down; "front": palm toward world +X; "back": palm toward world -X


@dataclass(frozen=True)
class BasePose:
    """Where the arm's base stands in the world (yaw about +Z only)."""

    position: tuple[float, float, float]
    yaw_deg: float

    @property
    def orientation_xyzw(self) -> tuple[float, float, float, float]:
        half = math.radians(self.yaw_deg) / 2.0
        return (0.0, 0.0, math.sin(half), math.cos(half))

    def to_base(self, world_point: Sequence[float]) -> np.ndarray:
        offset = np.asarray(world_point, dtype=np.float64) - np.asarray(self.position, dtype=np.float64)
        return Rotation.from_euler("z", -self.yaw_deg, degrees=True).apply(offset)

    def rotation_to_base(self, world_rotation: Rotation) -> Rotation:
        return Rotation.from_euler("z", -self.yaw_deg, degrees=True) * world_rotation


@dataclass(frozen=True)
class Workspace:
    """The adapter's palm clamp in the arm's base frame."""

    shoulder_position: tuple[float, float, float]
    shoulder_reach: float
    minimum_z: float
    maximum_z: float


@dataclass(frozen=True)
class SphereReach:
    key_point: KeyPoint
    distance: float      # shoulder to the point
    palm_margin: float   # reach - distance: positive when the palm link itself can be there
    grasp_margin: float  # palm_margin + GRASP_REACH_M
    in_height_band: bool
    required_margin: float = REACHABLE_MARGIN_M  # 0 when the point is an exact palm position

    @property
    def reachable(self) -> bool:
        return self.in_height_band and self.grasp_margin >= self.required_margin


@lru_cache(maxsize=None)
def _scene_states(activity: str, task: AratTask, layout: Layout | None) -> tuple[dict, dict]:
    scene = get_task_scene_data(task, layout=layout)
    return scene["state"]["registry"]["object_registry"], scene["objects_info"]["init_info"]


def _object_states(task: AratTask, layout: Layout | None = None) -> tuple[dict, dict]:
    """(object states, object init info) of the task's scene under ``layout``; read-only, cached."""

    return _scene_states(task.activity, _HashableTask(task), None if layout is None or layout.is_identity else layout)


class _HashableTask:
    """AratTask holds a dict and is not hashable; cache on the activity name only."""

    def __init__(self, task: AratTask) -> None:
        self.task = task

    def __hash__(self) -> int:
        return hash(self.task.activity)

    def __eq__(self, other) -> bool:
        return isinstance(other, _HashableTask) and other.task.activity == self.task.activity

    def __getattr__(self, name):
        return getattr(self.task, name)


def _root_pose(states: Mapping, name: str) -> tuple[np.ndarray, Rotation]:
    root = states[name]["root_link"]
    return np.asarray(root["pos"], dtype=np.float64), Rotation.from_quat(root["ori"])


@lru_cache(maxsize=1)
def _rubrics() -> dict:
    with _RUBRICS_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _rubric(task: AratTask) -> dict:
    rubrics = _rubrics()
    items = rubrics.get("items", rubrics)
    try:
        return items[task.activity]
    except KeyError as error:
        raise ValueError(f"{_RUBRICS_PATH} has no entry for {task.activity}") from error


def shelf_world(states: Mapping, box_name: str = "arat_box") -> tuple[tuple[float, float], tuple[float, float], float]:
    """((x_lo, x_hi), (y_lo, y_hi), top_z) of the shelf surface in the world."""

    position, rotation = _root_pose(states, box_name)
    corners = np.array([[x, y, SHELF_TOP_Z] for x in SHELF_X for y in SHELF_Y])
    world = rotation.apply(corners) + position
    return ((float(world[:, 0].min()), float(world[:, 0].max())),
            (float(world[:, 1].min()), float(world[:, 1].max())), float(world[:, 2].max()))


def mannequin_aabb(states: Mapping, init_info: Mapping, name: str = "mannequin") -> tuple[np.ndarray, np.ndarray]:
    """World AABB of the mannequin from its asset metadata (collision box of the base link), scale and pose."""

    args = init_info[name]["args"]
    metadata_path = _ASSET_ROOT / "objects" / args["category"] / args["model"] / "misc" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    box = metadata["link_bounding_boxes"]["base_link"]["collision"]["axis_aligned"]
    extent = np.asarray(box["extent"], dtype=np.float64)
    transform = np.asarray(box["transform"], dtype=np.float64)
    corners = np.array([[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)]) * extent
    local = corners @ transform[:3, :3].T + transform[:3, 3]
    position, rotation = _root_pose(states, name)
    world = rotation.apply(local * np.asarray(args.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)) + position
    return world.min(axis=0), world.max(axis=0)


def task_key_points(task: AratTask, layout: Layout | None = None) -> tuple[KeyPoint, ...]:
    """The world points ``task`` needs the hand at, from its instance (under ``layout``) and its scoring target."""

    states, init_info = _object_states(task, layout)
    rubric = _rubric(task)
    target = rubric["target"]
    kind = target["kind"]
    points: list[KeyPoint] = []

    def scene_name(instance: str) -> str:
        return task.instances[instance]

    start = None
    if rubric.get("tracked_instance"):
        position, _ = _root_pose(states, scene_name(rubric["tracked_instance"]))
        start = position
        points.append(KeyPoint("start", (float(position[0]), float(position[1]), float(position[2]) + LIFT_ABOVE_START_M), "top"))

    def shelf_point() -> KeyPoint:
        (x_lo, x_hi), (y_lo, y_hi), top = shelf_world(states, scene_name(target["shelf_instance"]))
        reference = start if start is not None else np.array([x_lo, (y_lo + y_hi) / 2.0, top])
        x = float(np.clip(reference[0], x_lo + SHELF_EDGE_INSET_M, x_hi - SHELF_EDGE_INSET_M))
        y = float(np.clip(reference[1], y_lo + SHELF_EDGE_INSET_M, y_hi - SHELF_EDGE_INSET_M))
        return KeyPoint("shelf", (x, y, top + LIFT_ABOVE_SHELF_M), "top")

    def tin_point() -> KeyPoint:
        position, _ = _root_pose(states, scene_name(target["tin_instance"]))
        return KeyPoint("tin", (float(position[0]), float(position[1]), float(position[2]) + LIFT_ABOVE_TIN_M), "top")

    if kind == "shelf":
        points.append(shelf_point())
    elif kind == "tin":
        points.append(tin_point())
    elif kind == "shelf_or_tin":
        # Either satisfies the goal; the shelf point nearest the start is never farther than the tin.
        points.append(shelf_point())
    elif kind == "pegged":
        position, _ = _root_pose(states, scene_name(target["peg_instance"]))
        points.append(KeyPoint("peg", (float(position[0]), float(position[1]), float(position[2]) + LIFT_ABOVE_PEG_M), "top"))
    elif kind == "pour":
        position, _ = _root_pose(states, scene_name(target["dest_instance"]))
        points.append(KeyPoint("pour", (float(position[0]), float(position[1]), float(position[2]) + LIFT_ABOVE_POUR_M), "top"))
    elif kind == "region":
        lower, upper = mannequin_aabb(states, init_info, scene_name(target["region_instance"]))
        region = MANNEQUIN_REGIONS[target["region"]]
        centre = region.center(tuple(lower), tuple(upper), MANNEQUIN_FRONT_DIR_XY)
        approach = {"front": "front", "back": "back"}.get(region.half, "top")
        if approach == "top":
            centre = (centre[0], centre[1], float(upper[2]))  # the palm lands on the crown
        points.append(KeyPoint(target["region"], tuple(float(c) for c in centre), approach))
    else:
        raise ValueError(f"{task.activity}: unsupported scoring target kind {kind!r}")
    return tuple(points)


def sphere_reach(key_point: KeyPoint, base: BasePose, workspace: Workspace) -> SphereReach:
    point = base.to_base(key_point.position)
    distance = float(np.linalg.norm(point - np.asarray(workspace.shoulder_position, dtype=np.float64)))
    palm_margin = workspace.shoulder_reach - distance
    # The palm may sit up to GRASP_REACH_M from the point, so the band is tested with that slack.
    in_band = workspace.minimum_z - GRASP_REACH_M <= point[2] <= workspace.maximum_z + GRASP_REACH_M
    return SphereReach(key_point, distance, palm_margin, palm_margin + GRASP_REACH_M, bool(in_band))


def task_sphere_reach(task: AratTask, base: BasePose, workspace: Workspace,
                      layout: Layout | None = None) -> tuple[SphereReach, ...]:
    return tuple(sphere_reach(point, base, workspace) for point in task_key_points(task, layout))


# --------------------------------------------------------------------------- floor footprint

# The ARAT table (breakfast_table/nvoqyl) in its own unscaled frame, measured on the collision meshes of
# arat-assets-v1/objects/breakfast_table/nvoqyl/usd/nvoqyl.usd: the four legs (the only geometry below
# the apron) and the top's outline.
TABLE_LEGS_LOCAL = (
    ((-0.381, -0.334), (0.550, 0.601)),
    ((-0.370, -0.334), (-0.590, -0.551)),
    ((0.334, 0.381), (0.551, 0.602)),
    ((0.334, 0.370), (-0.590, -0.551)),
)
TABLE_TOP_LOCAL = ((-0.382, 0.382), (-0.602, 0.602))
FOOTPRINT_CLEARANCE_M = 0.01
# Below 0.7 m (under its spread arms) the mannequin mesh stays within this distance of its
# bounding box's Y centre: legs and trunk span y -0.263..0.046 about the centre -0.108.
MANNEQUIN_STAND_HALF_WIDTH_M = 0.16


def _rectangle(centre_xy: Sequence[float], half_xy: Sequence[float], yaw_deg: float, origin_xy: Sequence[float]) -> np.ndarray:
    """Corners (4, 2) of a rectangle given in a frame at ``origin_xy`` yawed by ``yaw_deg``."""

    c, s_ = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    rotation = np.array([[c, -s_], [s_, c]])
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]) * np.asarray(half_xy) + np.asarray(centre_xy)
    return corners @ rotation.T + np.asarray(origin_xy, dtype=np.float64)


def _aabb_rectangle(x_range: Sequence[float], y_range: Sequence[float], inflate: float = 0.0) -> np.ndarray:
    x_lo, x_hi = x_range[0] - inflate, x_range[1] + inflate
    y_lo, y_hi = y_range[0] - inflate, y_range[1] + inflate
    return np.array([[x_lo, y_lo], [x_hi, y_lo], [x_hi, y_hi], [x_lo, y_hi]])


def rectangles_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    """Separating-axis test for two convex quadrilaterals given as (4, 2) corner loops."""

    for polygon in (first, second):
        for index in range(4):
            edge = polygon[(index + 1) % 4] - polygon[index]
            axis = np.array([-edge[1], edge[0]])
            a, b = first @ axis, second @ axis
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True


def table_obstacles(states: Mapping, init_info: Mapping, name: str = "table") -> tuple[list[np.ndarray], np.ndarray] | None:
    """(leg rectangles, tabletop rectangle) of the task's table in the world, or None without a table."""

    if name not in states:
        return None
    position, rotation = _root_pose(states, name)
    scale = np.asarray(init_info[name]["args"].get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    yaw = rotation.as_euler("zyx", degrees=True)[0]

    def world(x_range, y_range):
        centre = np.array([(x_range[0] + x_range[1]) / 2.0, (y_range[0] + y_range[1]) / 2.0]) * scale[:2]
        half = np.array([(x_range[1] - x_range[0]) / 2.0, (y_range[1] - y_range[0]) / 2.0]) * scale[:2]
        return _rectangle(centre, half, yaw, position[:2])

    return [world(*leg) for leg in TABLE_LEGS_LOCAL], world(*TABLE_TOP_LOCAL)


def footprint_conflicts(task: AratTask, base: BasePose, floor_boxes: Sequence[tuple], above_table_points: Sequence[Sequence[float]] = (),
                        clearance: float = FOOTPRINT_CLEARANCE_M) -> list[str]:
    """Why the arm cannot stand at ``base`` for ``task`` (empty when it can).

    ``floor_boxes``: (name, centre_xy, half_xy, stays_outside_tabletop) rectangles of the robot in its
    base frame. Each must clear the table legs; those flagged must also stay outside the tabletop's
    outline (parts taller than the table). ``above_table_points``: base-frame (x, y) points of fixed
    robot parts at tabletop height (e.g. a welded second arm) that must stay outside the outline.
    For the mannequin layouts the flagged boxes and the fixed points must clear the mannequin's
    bounding box (its arms are spread at 0.7-1.2 m); the other boxes only where it stands.
    """

    states, init_info = _object_states(task)
    conflicts: list[str] = []
    obstacles = table_obstacles(states, init_info)
    rectangles = [(name, _rectangle(centre, half, base.yaw_deg, base.position[:2]), tall) for name, centre, half, tall in floor_boxes]
    if obstacles is not None:
        legs, top = obstacles
        for name, rectangle, tall in rectangles:
            for index, leg in enumerate(legs):
                # push every leg corner outward by the clearance
                leg_inflated = leg + clearance * np.sign(leg - leg.mean(axis=0))
                if rectangles_overlap(rectangle, leg_inflated):
                    conflicts.append(f"{name} overlaps table leg {index}")
            if tall and rectangles_overlap(rectangle, top):
                conflicts.append(f"{name} is under/inside the tabletop outline")
        top_lo, top_hi = top.min(axis=0), top.max(axis=0)
        for point in above_table_points:
            world = Rotation.from_euler("z", base.yaw_deg, degrees=True).apply([point[0], point[1], 0.0])[:2] + np.asarray(base.position[:2])
            if np.all(world >= top_lo - 0.03) and np.all(world <= top_hi + 0.03):
                conflicts.append(f"fixed part at base-frame {tuple(round(float(v), 3) for v in point)} is over the tabletop")
    if "mannequin" in states:
        lower, upper = mannequin_aabb(states, init_info)
        body = _aabb_rectangle((lower[0], upper[0]), (lower[1], upper[1]), inflate=clearance)
        # Below its spread arms the mannequin is only as wide as its legs and trunk.
        centre_y = (lower[1] + upper[1]) / 2.0
        stand = _aabb_rectangle(
            (lower[0], upper[0]), (centre_y - MANNEQUIN_STAND_HALF_WIDTH_M, centre_y + MANNEQUIN_STAND_HALF_WIDTH_M), inflate=clearance
        )
        for name, rectangle, tall in rectangles:
            if rectangles_overlap(rectangle, body if tall else stand):
                conflicts.append(f"{name} overlaps the mannequin" + ("'s bounding box" if tall else "'s legs"))
        for point in above_table_points:
            world = Rotation.from_euler("z", base.yaw_deg, degrees=True).apply([point[0], point[1], 0.0])[:2] + np.asarray(base.position[:2])
            if np.all(world >= lower[:2] - 0.03) and np.all(world <= upper[:2] + 0.03):
                conflicts.append(f"fixed part at base-frame {tuple(round(float(v), 3) for v in point)} is inside the mannequin's bounding box")
    return conflicts


# --------------------------------------------------------------------------- the apparatus on the table

LAYOUT_EDGE_MARGIN_M = 0.02


@lru_cache(maxsize=None)
def _asset_box(category: str, model: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(bounding-box size, its centre in the base-link frame) of an ARAT asset, from its metadata."""

    metadata = json.loads((_ASSET_ROOT / "objects" / category / model / "misc" / "metadata.json").read_text(encoding="utf-8"))
    return tuple(float(v) for v in metadata["bbox_size"]), tuple(float(v) for v in metadata["base_link_offset"])


def apparatus_footprints(task: AratTask, layout: Layout | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """World (x, y) extents ``(lower, upper)`` of every object the layout moves, from the asset bounding boxes.

    The ARAT box's bounding box includes its open lid, as saved.
    """

    states, init_info = _object_states(task, layout)
    footprints: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in moved_object_names({"objects_info": {"init_info": init_info}}):
        args = init_info[name]["args"]
        if "category" not in args or "model" not in args:
            continue
        size, centre = _asset_box(args["category"], args["model"])
        scale = np.asarray(args.get("scale") or [1.0, 1.0, 1.0], dtype=np.float64)
        corners = (np.array([[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)]) * size + centre) * scale
        position, rotation = _root_pose(states, name)
        world = rotation.apply(corners) + position
        footprints[name] = (world[:, :2].min(axis=0), world[:, :2].max(axis=0))
    return footprints


def tabletop_margin(task: AratTask, layout: Layout | None = None) -> tuple[float, str] | None:
    """(smallest distance of any moved object's footprint to the tabletop's edge, that object); None without a table.

    Negative when an object overhangs the edge.
    """

    states, init_info = _object_states(task, layout)
    obstacles = table_obstacles(states, init_info)
    if obstacles is None:
        return None
    top = obstacles[1]
    top_lo, top_hi = top.min(axis=0), top.max(axis=0)
    worst: tuple[float, str] | None = None
    for name, (lower, upper) in apparatus_footprints(task, layout).items():
        margin = float(min((lower - top_lo).min(), (top_hi - upper).min()))
        if worst is None or margin < worst[0]:
            worst = (margin, name)
    return worst


def layout_conflicts(task: AratTask, layout: Layout | None = None, margin: float = LAYOUT_EDGE_MARGIN_M) -> list[str]:
    """Why the apparatus cannot stand where ``layout`` puts it (empty when it can).

    Every moved object's footprint must stay ``margin`` inside the tabletop's outline. A layout is a
    rigid motion, so the apparatus parts cannot collide with each other that did not before; a
    layout on a scene without a table (the mannequin items) is refused.
    """

    if layout is None or layout.is_identity:
        return []
    worst = tabletop_margin(task, layout)
    if worst is None:
        return ["a layout was given for a scene without a table"]
    if worst[0] < margin:
        return [f"{worst[1]} is {worst[0] * 100:+.1f} cm from the tabletop edge under the layout (needs {margin * 100:.0f} cm)"]
    return []


# --------------------------------------------------------------------------- inverse kinematics


def palm_rotations(approach: str, toward_xy: Sequence[float]) -> list[tuple[str, Rotation]]:
    """World orientations of the palm link to try for one approach.

    Palm frame: +X palm normal, +Z along the fingers. ``toward_xy`` is the horizontal direction
    from the shoulder to the target, the natural direction for the fingers to point.
    """

    heading = math.degrees(math.atan2(toward_xy[1], toward_xy[0]))
    candidates: list[tuple[str, Rotation]] = []
    if approach == "top":
        # palm normal straight down, fingers horizontal; then pitched so the fingers dip toward the object
        for yaw_offset in (0.0, -30.0, 30.0, -60.0, 60.0):
            for pitch in (0.0, 30.0, 60.0):
                # palm x = -Z world, palm z = +X world at zero yaw/pitch
                base = Rotation.from_matrix(np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]))
                rotation = Rotation.from_euler("z", heading + yaw_offset, degrees=True) * Rotation.from_euler("y", pitch, degrees=True) * base
                candidates.append((f"palm down, fingers {yaw_offset:+.0f} deg from the reach direction, pitched {pitch:.0f} deg", rotation))
    elif approach in ("front", "back"):
        normal = np.array([1.0, 0.0, 0.0]) if approach == "front" else np.array([-1.0, 0.0, 0.0])
        for label, fingers in (("up", (0.0, 0.0, 1.0)), ("down", (0.0, 0.0, -1.0)), ("toward +Y", (0.0, 1.0, 0.0)), ("toward -Y", (0.0, -1.0, 0.0))):
            z_axis = np.asarray(fingers)
            y_axis = np.cross(z_axis, normal)
            matrix = np.column_stack([normal, y_axis, z_axis])
            candidates.append((f"palm facing {'+X' if approach == 'front' else '-X'}, fingers {label}", Rotation.from_matrix(matrix)))
    else:
        raise ValueError(f"Unknown approach {approach!r}")
    return candidates


@dataclass(frozen=True)
class IkReach:
    key_point: KeyPoint
    solved: bool
    candidate: str | None
    joint_limit_margin_deg: float | None  # smallest distance of any arm joint to its limit
    joint_positions: tuple[float, ...] | None


class ArmIk:
    """Damped least-squares IK of the palm link with joint limits (Pinocchio, lazily imported)."""

    def __init__(self, urdf_path: str | Path, arm_joint_names: Sequence[str], palm_link: str, *, seed: int = 0) -> None:
        import pinocchio as pin

        self._pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        joint_ids = [self.model.getJointId(name) for name in arm_joint_names]
        self._q_index = np.array([self.model.joints[joint_id].idx_q for joint_id in joint_ids])
        self._v_index = np.array([self.model.joints[joint_id].idx_v for joint_id in joint_ids])
        self._frame = self.model.getFrameId(palm_link)
        self.lower = self.model.lowerPositionLimit[self._q_index]
        self.upper = self.model.upperPositionLimit[self._q_index]
        self._rng = np.random.default_rng(seed)

    def forward(self, q_arm: Sequence[float]):
        pin = self._pin
        q = pin.neutral(self.model)
        q[self._q_index] = np.asarray(q_arm, dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self._frame]

    def solve(self, position: Sequence[float], rotation: Rotation, *, seeds: Sequence[Sequence[float]] = (),
              restarts: int = 12, iterations: int = 200, position_tolerance: float = 2.0e-3,
              orientation_tolerance: float = math.radians(1.0)) -> tuple[np.ndarray, float] | None:
        """Arm joint positions reaching the palm pose (base frame) and their limit margin, or None."""

        pin = self._pin
        target = pin.SE3(rotation.as_matrix(), np.asarray(position, dtype=np.float64))
        starts = [np.clip(np.asarray(seed, dtype=np.float64), self.lower, self.upper) for seed in seeds]
        starts += [self._rng.uniform(self.lower, self.upper) for _ in range(restarts)]
        best: tuple[np.ndarray, float] | None = None
        q_full = pin.neutral(self.model)
        for q_arm in starts:
            q_arm = q_arm.copy()
            for _ in range(iterations):
                q_full[self._q_index] = q_arm
                pin.forwardKinematics(self.model, self.data, q_full)
                pin.updateFramePlacements(self.model, self.data)
                error = pin.log6(self.data.oMf[self._frame].actInv(target)).vector
                if np.linalg.norm(error) < 1.0e-6:
                    break
                jacobian = pin.computeFrameJacobian(self.model, self.data, q_full, self._frame, pin.LOCAL)[:, self._v_index]
                step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1.0e-5 * np.eye(6), error)
                q_arm = np.clip(q_arm + np.clip(step, -0.2, 0.2), self.lower, self.upper)
            placement = self.forward(q_arm)
            position_error = float(np.linalg.norm(placement.translation - target.translation))
            orientation_error = float(np.linalg.norm(pin.log3(placement.rotation.T @ target.rotation)))
            if position_error > position_tolerance or orientation_error > orientation_tolerance:
                continue
            margin = float(np.min(np.minimum(q_arm - self.lower, self.upper - q_arm)))
            if best is None or margin > best[1]:
                best = (q_arm, margin)
        return best


def palm_position_for(key_point: KeyPoint, rotation: Rotation) -> np.ndarray:
    """World position of the palm link that puts the hand's grasp point on the key point."""

    return np.asarray(key_point.position, dtype=np.float64) - rotation.apply(np.asarray(GRASP_POINT_PALM))


def ik_reach(key_point: KeyPoint, base: BasePose, workspace: Workspace, ik: ArmIk, *,
             seeds: Sequence[Sequence[float]] = ()) -> IkReach:
    shoulder_world = np.asarray(base.position) + Rotation.from_euler("z", base.yaw_deg, degrees=True).apply(
        np.asarray(workspace.shoulder_position)
    )
    toward = np.asarray(key_point.position)[:2] - shoulder_world[:2]
    if np.linalg.norm(toward) < 1.0e-6:
        toward = np.array([1.0, 0.0])
    best: tuple[str, np.ndarray, float] | None = None
    for label, rotation in palm_rotations(key_point.approach, toward):
        palm_world = palm_position_for(key_point, rotation)
        palm_base = base.to_base(palm_world)
        # The adapter would clamp a palm target outside its sphere or band; such a pose is not usable.
        distance = float(np.linalg.norm(palm_base - np.asarray(workspace.shoulder_position)))
        if distance > workspace.shoulder_reach or not (workspace.minimum_z <= palm_base[2] <= workspace.maximum_z):
            continue
        solution = ik.solve(palm_base, base.rotation_to_base(rotation), seeds=seeds)
        if solution is not None and (best is None or solution[1] > best[2]):
            best = (label, solution[0], solution[1])
    if best is None:
        return IkReach(key_point, False, None, None, None)
    return IkReach(key_point, True, best[0], math.degrees(best[2]), tuple(float(q) for q in best[1]))


# --------------------------------------------------------------------------- following a path

FOLLOW_POSITION_TOLERANCE_M = 0.010
FOLLOW_ORIENTATION_TOLERANCE_DEG = 3.0
PATH_JOINT_MARGIN_DEG = 5.0
HOVER_ABOVE_PICK_M = 0.10
TRANSPORT_CLEARANCE_M = 0.08


@dataclass(frozen=True)
class PathReach:
    """How well the palm followed a path from a start configuration."""

    max_position_error_m: float
    max_orientation_error_deg: float
    joint_limit_margin_deg: float  # smallest distance of any arm joint to its limit along the path

    @property
    def follows(self) -> bool:
        return (
            self.max_position_error_m < FOLLOW_POSITION_TOLERANCE_M
            and self.max_orientation_error_deg < FOLLOW_ORIENTATION_TOLERANCE_DEG
        )

    @property
    def ok(self) -> bool:
        return self.follows and self.joint_limit_margin_deg >= PATH_JOINT_MARGIN_DEG


def track_palm(ik: ArmIk, joint_positions: Sequence[float], waypoints: Sequence[tuple[Sequence[float], Rotation]], *,
               step_m: float = 0.005, step_rad: float = math.radians(1.5), iterations: int = 2,
               damping: float = 1.0e-4) -> tuple[np.ndarray, PathReach]:
    """Lead the palm through base-frame ``(position, Rotation)`` waypoints from ``joint_positions``.

    The target moves along straight segments (orientation slerped) in steps of at most ``step_m`` and
    ``step_rad``; each step takes ``iterations`` damped pseudo-inverse updates clipped to the joint
    limits. No null-space term and no re-seeding: like the simulator's IK controller, the arm stays in
    the IK family it started in, and stops following when a joint reaches its limit.
    """

    from scipy.spatial.transform import Slerp

    pin = ik._pin
    q_arm = np.clip(np.asarray(joint_positions, dtype=np.float64), ik.lower, ik.upper)
    q_full = pin.neutral(ik.model)
    start = ik.forward(q_arm)
    position, rotation = start.translation.copy(), Rotation.from_matrix(start.rotation)
    worst_position = worst_angle = 0.0
    margin = float(np.min(np.minimum(q_arm - ik.lower, ik.upper - q_arm)))
    for target_position, target_rotation in waypoints:
        target_position = np.asarray(target_position, dtype=np.float64)
        length = float(np.linalg.norm(target_position - position))
        angle = float((rotation.inv() * target_rotation).magnitude())
        steps = max(1, int(math.ceil(max(length / step_m, angle / step_rad))))
        slerp = Slerp([0.0, 1.0], Rotation.concatenate([rotation, target_rotation]))
        for index in range(1, steps + 1):
            fraction = index / steps
            goal_position = position + (target_position - position) * fraction
            goal_rotation = slerp(fraction)
            goal = pin.SE3(goal_rotation.as_matrix(), goal_position)
            for _ in range(iterations):
                q_full[ik._q_index] = q_arm
                pin.forwardKinematics(ik.model, ik.data, q_full)
                pin.updateFramePlacements(ik.model, ik.data)
                error = pin.log6(ik.data.oMf[ik._frame].actInv(goal)).vector
                jacobian = pin.computeFrameJacobian(ik.model, ik.data, q_full, ik._frame, pin.LOCAL)[:, ik._v_index]
                step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(6), error)
                q_arm = np.clip(q_arm + step, ik.lower, ik.upper)
            placement = ik.forward(q_arm)
            worst_position = max(worst_position, float(np.linalg.norm(placement.translation - goal_position)))
            worst_angle = max(worst_angle, float((Rotation.from_matrix(placement.rotation).inv() * goal_rotation).magnitude()))
            margin = min(margin, float(np.min(np.minimum(q_arm - ik.lower, ik.upper - q_arm))))
        position, rotation = target_position, target_rotation
    return q_arm, PathReach(worst_position, math.degrees(worst_angle), math.degrees(margin))


def task_palm_path(task: AratTask, start_rotation: Rotation,
                   touch: Sequence[tuple[Sequence[float], Rotation]] | None = None,
                   layout: Layout | None = None) -> list[tuple[np.ndarray, Rotation]]:
    """World palm waypoints of ``task``.

    Pick-and-place tasks keep ``start_rotation`` (the palm orientation of the start configuration):
    ``HOVER_ABOVE_PICK_M`` above the pick point, onto it, back up, across ``TRANSPORT_CLEARANCE_M``
    above the higher of the two points, onto the target. Mannequin tasks follow ``touch``.
    """

    points = task_key_points(task, layout)
    if len(points) == 1:
        if not touch:
            raise ValueError(f"{task.activity}: a mannequin region needs a recorded touch")
        return [(np.asarray(position, dtype=np.float64), rotation) for position, rotation in touch]
    pick, place = (palm_position_for(point, start_rotation) for point in points)
    top = max(pick[2], place[2]) + TRANSPORT_CLEARANCE_M
    hover = pick + np.array([0.0, 0.0, HOVER_ABOVE_PICK_M])
    return [(position, start_rotation) for position in (
        hover, pick, hover, np.array([pick[0], pick[1], top]), np.array([place[0], place[1], top]), place,
    )]


def path_reach(task: AratTask, base: BasePose, ik: ArmIk, start_joint_positions: Sequence[float],
               touch: Sequence[tuple[Sequence[float], Rotation]] | None = None,
               layout: Layout | None = None) -> PathReach:
    """Lead the palm along ``task``'s path (under ``layout``) from ``start_joint_positions`` with the base at ``base``."""

    start = ik.forward(start_joint_positions)
    start_rotation = Rotation.from_euler("z", base.yaw_deg, degrees=True) * Rotation.from_matrix(start.rotation)
    waypoints = [(base.to_base(position), base.rotation_to_base(rotation)) for position, rotation in task_palm_path(task, start_rotation, touch, layout)]
    return track_palm(ik, start_joint_positions, waypoints)[1]


def palm_sphere_reach(name: str, palm_position: Sequence[float], base: BasePose, workspace: Workspace) -> SphereReach:
    """The adapter clamp tested against an exact palm position (world), e.g. the end of a touch."""

    point = base.to_base(palm_position)
    distance = float(np.linalg.norm(point - np.asarray(workspace.shoulder_position, dtype=np.float64)))
    margin = workspace.shoulder_reach - distance
    return SphereReach(
        key_point=KeyPoint(name, tuple(float(c) for c in palm_position), "palm"),
        distance=distance,
        palm_margin=margin,
        grasp_margin=margin,
        in_height_band=bool(workspace.minimum_z <= point[2] <= workspace.maximum_z),
        required_margin=0.0,
    )


__all__ = [
    "ArmIk",
    "BasePose",
    "GRASP_POINT_PALM",
    "GRASP_REACH_M",
    "IkReach",
    "KeyPoint",
    "LAYOUT_EDGE_MARGIN_M",
    "Layout",
    "PATH_JOINT_MARGIN_DEG",
    "PathReach",
    "REACHABLE_MARGIN_M",
    "SphereReach",
    "Workspace",
    "apparatus_footprints",
    "footprint_conflicts",
    "ik_reach",
    "layout_conflicts",
    "mannequin_aabb",
    "palm_position_for",
    "palm_rotations",
    "palm_sphere_reach",
    "path_reach",
    "shelf_world",
    "sphere_reach",
    "task_key_points",
    "task_palm_path",
    "task_sphere_reach",
    "tabletop_margin",
    "track_palm",
]
