#!/usr/bin/env python3
"""Build the Tianji Marvin M6 CCS (right arm) + right SharpaWave flange URDF asset.

The wrist bench's second arm. Starting from the Marvin Pro dual-arm model
already in ``omnigibson-robot-assets`` (``source/tianji_m6_ccs``, from Tianji's
Gento teleoperation description), this removes the right parallel gripper and
bolts Sharpa's official ``right_sharpa_wave_with_flange`` model onto the right
arm's end flange, exactly where Tianji's own kinematic model puts that flange:

* **95 mm from the J7 axis**, along the Link7 ``-Y`` axis. This is the tool
  row of Tianji's official M6 4.0 kinematic configuration
  (``CommonConfig/ccs_m6_40.MvKDCfg`` in the TJ_FX_ROBOT_CONTRL_SDK: modified-DH
  ``alpha=90, a=0, d=95, theta=90``), the extent of the official ``Link7_R.STL``
  mesh (its flange face lies at ``y = -0.095``), and the number that closes the
  published "J2 axis to end flange" working reach: 287 + 314 + 95 = 696 mm.
  The Gento description places its ``flange_R`` frame 88 mm out, 7 mm inside
  the physical face; that frame is not used here.
* **ISO 9409-1-50-4-M6 interface**, measured on the official Link7 mesh: four
  M6 holes on a 50 mm pitch circle at +-45/+-135 degrees, a 6 mm locating-pin
  hole on the same circle along Link7 ``+X``, a 31.5 mm centring recess and a
  62 mm face. Sharpa's flange adapter (``flange_B.STL``) carries the matching
  Ø50 x 4 x M6 pattern, a Ø31.5 pilot boss and its pin hole along its own
  ``+X``, as does Franka's DIN ISO 9409-1-A50 flange the adapter was already
  mounted on. The roll about the flange normal is therefore fixed by the pin:
  Sharpa ``+X`` (the palm normal of ``right_hand_C_MC``) lands on Link7 ``+X``.

Composed, the attachment joint ``Arm_R7_Link -> right_hand_flange`` is
``xyz="0 -0.095 0" rpy="pi/2 0 0"``: the hand axis (Sharpa ``+Z``) leaves the
flange along its outward normal, as on the real robot (Tianji lists Sharpa
among the dexterous hands its Marvin arms carry).

Other changes to the upstream dual-arm model:

* ``base_link`` is re-origined at the floor contact plane of the torso mesh
  (the upstream origin sits 82 mm above the base plate's underside), so the
  OmniGibson importer's bottom-surface normalisation becomes a no-op and the
  shoulder joints stay where the torso mesh puts them.
* The base plate, column, shoulder yoke and head of the torso get box
  colliders (expressed as a scaled unit-cube mesh, as the Franka pedestal
  does) in place of the upstream placeholder cylinder.
* The left arm and its gripper are welded hanging beside the torso
  (``LEFT_ARM_TUCKED_POSE``; the upstream Marvin Pro home pose until 2026-11-15),
  so the robot is a single 7-DoF right arm + 22-DoF hand (the same 28-D
  IK + finger action as the Franka + Sharpa bench robot) with the whole
  Marvin Pro visible.
* Mass-less frames keep a 1 µg placeholder mass (Sharpa's own fingertip
  frames use the same value) so every articulation link has positive mass.

The script also writes the OmniGibson robot definition
(``tianji_m6_ccs_sharpa_right.yaml``) and importer configuration
(``tianji_m6_ccs_sharpa_right.import.yaml``). Convert the URDF to USD with::

    OMNI_KIT_ACCEPT_EULA=YES python -m omnigibson.examples.robots.import_custom_robot \\
        --config datasets/omnigibson-robot-assets/models/tianji_m6_ccs_sharpa_right/tianji_m6_ccs_sharpa_right.import.yaml

then move ``objects/robot/tianji_m6_ccs_sharpa_right/{usd,misc}`` and the
``*_with_meta_links.urdf`` next to the URDF (``--finish-import`` does this).
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import shutil
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROBOT_ASSETS = REPOSITORY_ROOT / "datasets" / "omnigibson-robot-assets"
# The Marvin Pro with the 809 mm Marvin M6-S CCS-809 arms (since 2026-11-15; made by
# make_tianji_m6_809_source.py). The 696 mm description this script was written for is
# source/tianji_m6_ccs: pass it with --tianji-urdf (and --right-arm-pose) to build that arm.
TIANJI_SOURCE_URDF = ROBOT_ASSETS / "source" / "tianji_m6_ccs_809" / "tianji_m6_ccs_source.urdf"
TIANJI_696_SOURCE_URDF = ROBOT_ASSETS / "source" / "tianji_m6_ccs" / "tianji_m6_ccs_source.urdf"
TIANJI_DEFINITION = ROBOT_ASSETS / "models" / "tianji_m6_ccs" / "tianji_m6_ccs.yaml"
FRANKA_SHARPA_ROOT = ROBOT_ASSETS / "models" / "franka" / "franka_dexhand" / "franka_sharpa_right"
SHARPA_URDF = FRANKA_SHARPA_ROOT / "source" / "right_sharpa_wave_with_flange.urdf"
SHARPA_MESHES = FRANKA_SHARPA_ROOT / "urdf" / "meshes" / "sharpa"
SHARPA_NOTICES = FRANKA_SHARPA_ROOT / "source"
MOUNT_DOC = REPOSITORY_ROOT / "dex_teleop" / "docs" / "tianji-m6-sharpa-mount.md"
MODEL_NAME = "tianji_m6_ccs_sharpa_right"
OUTPUT_DIR = ROBOT_ASSETS / "models" / MODEL_NAME

# --- Official Tianji flange (see the module docstring and docs/tianji-m6-sharpa-mount.md)
FLANGE_OFFSET_M = 0.095
# Arm_R7_Link -> right_hand_flange: hand +Z along the flange normal (Link7 -Y), hand +X on the pin (Link7 +X).
ATTACHMENT_XYZ = (0.0, -FLANGE_OFFSET_M, 0.0)
ATTACHMENT_RPY = (math.pi / 2.0, 0.0, 0.0)
ATTACHMENT_JOINT = "tianji_right_hand_flange_joint"

RIGHT_ARM_LINKS = ("Base_R",) + tuple(f"Arm_R{i}_Link" for i in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f"Arm_R{i}_Joint" for i in range(1, 8))
RIGHT_TOOL_LINKS = {
    "flange_R",
    "right_gripper_mount",
    "right_gripper_base",
    "right_gripper_left_finger",
    "right_gripper_right_finger",
    "right_tool",
}
PLACEHOLDER_MASS = 1.0e-6
PLACEHOLDER_INERTIA = 2.0e-12
TORSO_LINK = "base_link"
TORSO_MESH = "meshes/marvin_pro/meshes/base/ZJ_Robot_link.STL"
# Box colliders for the Marvin Pro torso, in the re-origined base_link frame (z = 0 at the
# floor). Measured on horizontal cross-sections of ZJ_Robot_link.STL: a 0.75 m square base
# plate 82 mm thick, a 108 x 102 mm column to 1.08 m, the shoulder yoke (y from -0.125 to
# 0.114 at the 1.2 m shoulder height) and the forward head/camera block.
TORSO_BOXES = (
    # name, centre xyz, size xyz
    ("plate", (0.0, -0.006, 0.041), (0.754, 0.754, 0.082)),
    ("column", (0.0, -0.006, 0.582), (0.108, 0.102, 1.000)),
    ("yoke", (0.0, -0.0055, 1.192), (0.150, 0.239, 0.060)),
    ("head", (0.0355, -0.006, 1.333), (0.099, 0.128, 0.103)),
)
UNIT_BOX_OBJ = """\
v -0.5 -0.5 -0.5
v 0.5 -0.5 -0.5
v 0.5 0.5 -0.5
v -0.5 0.5 -0.5
v -0.5 -0.5 0.5
v 0.5 -0.5 0.5
v 0.5 0.5 0.5
v -0.5 0.5 0.5
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""

# Sharpa finger joints in the order every dex_teleop Sharpa action uses (dex_teleop.hands).
SHARPA_FINGER_JOINTS = (
    "right_thumb_CMC_FE", "right_thumb_CMC_AA", "right_thumb_MCP_FE", "right_thumb_MCP_AA", "right_thumb_IP",
    "right_index_MCP_FE", "right_index_MCP_AA", "right_index_PIP", "right_index_DIP",
    "right_middle_MCP_FE", "right_middle_MCP_AA", "right_middle_PIP", "right_middle_DIP",
    "right_ring_MCP_FE", "right_ring_MCP_AA", "right_ring_PIP", "right_ring_DIP",
    "right_pinky_CMC", "right_pinky_MCP_FE", "right_pinky_MCP_AA", "right_pinky_PIP", "right_pinky_DIP",
)
SHARPA_FINGER_LINKS = (
    "right_thumb_PP", "right_thumb_DP", "right_thumb_elastomer",
    "right_index_PP", "right_index_MP", "right_index_DP", "right_index_elastomer",
    "right_middle_PP", "right_middle_MP", "right_middle_DP", "right_middle_elastomer",
    "right_ring_PP", "right_ring_MP", "right_ring_DP", "right_ring_elastomer",
    "right_pinky_PP", "right_pinky_MP", "right_pinky_DP", "right_pinky_elastomer",
)
PALM_LINK = "right_hand_C_MC"
# Default right-arm configuration written to the robot definition: the wrist bench's
# "extended" pose (hand_bench.yaml wrist_bench.arms.tianji.reset_poses.extended). Solved with
# `solve_reset_pose` below (Pinocchio, damped least squares, 120 random restarts, the solution
# with the largest joint-limit margin and the elbow below and outside the shoulder) for
# right_hand_C_MC at (0.50, -0.20, 1.10) m in the base_link frame, palm normal straight down,
# fingers along +X, thumb toward +Y: in front of the right shoulder, 0.35 m above the bench
# table, 24 degrees from the nearest joint limit (J4; 39 degrees with the 696 mm arm).
# `--solve-reset-pose` re-derives it.
# 809 mm arm (--solve-reset-pose, 2026-11-15). The 696 mm arm's pose for the same target was
# (-1.233317, -1.338584, 1.411068, -1.857169, -1.387896, 0.215559, 0.585308).
DEFAULT_RIGHT_ARM_POSE = (-1.131583, -1.55489, 1.555222, -2.114454, -1.564793, 0.022198, 0.982678)
RESET_POSE_TARGET_POSITION = (0.50, -0.20, 1.10)
# Arm_L1..L7 of the welded left arm: hanging beside the torso (elbow 0.81 m, gripper 0.30 m above the floor,
# inside the base plate's outline), so that it does not keep the robot away from a table. Until 2026-11-15
# it was welded at the upstream Marvin Pro home pose (forearm 0.5 m forward at 0.95 m, tabletop height):
# --left-arm-pose home restores that.
LEFT_ARM_TUCKED_POSE = (1.5708, -1.345, 0.0, -0.275, 1.7, 0.0, 0.0)
# Rotation matrix (rows) of right_hand_C_MC in the base frame; its columns are the palm axes:
# +X (palm normal) = -Z, +Y (pinky->thumb) = +Y, +Z (wrist->fingers) = +X.
RESET_POSE_TARGET_ROTATION = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Small helpers


def _floats(text: str) -> np.ndarray:
    return np.array([float(value) for value in text.split()], dtype=np.float64)


def _format(values) -> str:
    return " ".join(f"{float(value):.7g}" for value in values)


def _origin(element: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    origin = element.find("origin")
    if origin is None:
        return np.zeros(3), np.zeros(3)
    return _floats(origin.get("xyz", "0 0 0")), _floats(origin.get("rpy", "0 0 0"))


def _set_origin(element: ET.Element, xyz, rpy) -> None:
    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin")
        # Keep <origin> first, where URDF readers expect it.
        element.remove(origin)
        element.insert(0, origin)
    origin.set("xyz", _format(xyz))
    origin.set("rpy", _format(rpy))


def stl_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds of a binary or ASCII STL, without trimesh."""

    data = path.read_bytes()
    if len(data) >= 84 and not data[:5].lower().startswith(b"solid"):
        (count,) = struct.unpack_from("<I", data, 80)
        if 84 + 50 * count == len(data):
            triangles = np.frombuffer(data, dtype=np.uint8, count=50 * count, offset=84).reshape(count, 50)
            vertices = triangles[:, 12:48].copy().view("<f4").reshape(-1, 3).astype(np.float64)
            return vertices.min(axis=0), vertices.max(axis=0)
    vertices = np.array(
        [
            [float(value) for value in line.split()[1:4]]
            for line in data.decode("ascii", errors="ignore").splitlines()
            if line.strip().startswith("vertex")
        ]
    )
    if vertices.size == 0:
        raise ValueError(f"{path} contains no vertices")
    return vertices.min(axis=0), vertices.max(axis=0)


def _link(root: ET.Element, name: str) -> ET.Element:
    matches = [link for link in root.findall("link") if link.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one link {name!r}, found {len(matches)}")
    return matches[0]


def _joint(root: ET.Element, name: str) -> ET.Element:
    matches = [joint for joint in root.findall("joint") if joint.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one joint {name!r}, found {len(matches)}")
    return matches[0]


def _subtree_links(root: ET.Element, start: str) -> set[str]:
    children: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint.find("child").get("link"))
    seen, stack = set(), [start]
    while stack:
        link = stack.pop()
        if link in seen:
            continue
        seen.add(link)
        stack.extend(children.get(link, []))
    return seen


# ---------------------------------------------------------------------------
# Transformations of the upstream model


def reorigin_base_link_at_floor(root: ET.Element, urdf_dir: Path) -> float:
    """Move base_link's origin down to the torso mesh's lowest point; returns the shift (m)."""

    base = _link(root, TORSO_LINK)
    mesh_file = None
    for visual in base.findall("visual"):
        mesh = visual.find("geometry/mesh")
        if mesh is not None:
            mesh_file = mesh.get("filename")
    if mesh_file is None:
        raise ValueError(f"{TORSO_LINK} has no visual mesh")
    lower, _ = stl_bounds(urdf_dir / mesh_file)
    shift = -float(lower[2])
    if shift <= 0.0:
        raise ValueError(f"{TORSO_LINK} mesh bottom is at z={lower[2]:.4f}; expected it below the link origin")
    for element in base.findall("visual") + base.findall("collision") + base.findall("inertial"):
        xyz, rpy = _origin(element)
        _set_origin(element, xyz + np.array([0.0, 0.0, shift]), rpy)
    for joint in root.findall("joint"):
        if joint.find("parent").get("link") == TORSO_LINK:
            xyz, rpy = _origin(joint)
            _set_origin(joint, xyz + np.array([0.0, 0.0, shift]), rpy)
    return shift


def replace_torso_collision(root: ET.Element, mesh_dir: Path) -> None:
    """Swap the upstream placeholder cylinder for the measured torso boxes."""

    base = _link(root, TORSO_LINK)
    for collision in base.findall("collision"):
        base.remove(collision)
    unit_box = mesh_dir / "base" / "unit_box.obj"
    unit_box.parent.mkdir(parents=True, exist_ok=True)
    unit_box.write_text(UNIT_BOX_OBJ)
    for name, centre, size in TORSO_BOXES:
        collision = ET.SubElement(base, "collision", {"name": f"{TORSO_LINK}_{name}_collision"})
        ET.SubElement(collision, "origin", {"xyz": _format(centre), "rpy": "0 0 0"})
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "mesh", {"filename": "meshes/base/unit_box.obj", "scale": _format(size)})


def weld_joints(root: ET.Element, joint_positions: dict[str, float]) -> None:
    """Turn movable joints into fixed ones at the given positions (folded into the origin)."""

    for name, position in joint_positions.items():
        joint = _joint(root, name)
        joint_type = joint.get("type")
        axis_element = joint.find("axis")
        axis = _floats(axis_element.get("xyz")) if axis_element is not None else np.array([1.0, 0.0, 0.0])
        xyz, rpy = _origin(joint)
        rotation = Rotation.from_euler("xyz", rpy)
        if joint_type in ("revolute", "continuous"):
            rotation = rotation * Rotation.from_rotvec(axis / np.linalg.norm(axis) * position)
        elif joint_type == "prismatic":
            xyz = xyz + rotation.apply(axis / np.linalg.norm(axis) * position)
        elif joint_type != "fixed":
            raise ValueError(f"cannot weld joint {name!r} of type {joint_type!r}")
        joint.set("type", "fixed")
        _set_origin(joint, xyz, rotation.as_euler("xyz"))
        for tag in ("axis", "limit", "dynamics", "mimic", "calibration", "safety_controller"):
            for element in joint.findall(tag):
                joint.remove(element)
    for transmission in list(root.findall("transmission")):
        joint = transmission.find("joint")
        if joint is not None and joint.get("name") in joint_positions:
            root.remove(transmission)


def remove_right_tool(root: ET.Element) -> None:
    """Drop the Gento flange_R frame and the right parallel gripper subtree."""

    for joint in list(root.findall("joint")):
        if joint.find("child").get("link") in RIGHT_TOOL_LINKS or joint.find("parent").get("link") in RIGHT_TOOL_LINKS:
            root.remove(joint)
    for link in list(root.findall("link")):
        if link.get("name") in RIGHT_TOOL_LINKS:
            root.remove(link)


def give_placeholder_mass(root: ET.Element) -> list[str]:
    """Replace zero masses (upstream frame links) with Sharpa's fingertip-frame placeholder."""

    patched = []
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            ET.SubElement(inertial, "mass", {"value": "0"})
            ET.SubElement(
                inertial, "inertia", {"ixx": "0", "ixy": "0", "ixz": "0", "iyy": "0", "iyz": "0", "izz": "0"}
            )
        mass = inertial.find("mass")
        if float(mass.get("value", "0")) <= 0.0:
            mass.set("value", repr(PLACEHOLDER_MASS))
            inertia = inertial.find("inertia")
            for key in ("ixx", "iyy", "izz"):
                inertia.set(key, repr(PLACEHOLDER_INERTIA))
            for key in ("ixy", "ixz", "iyz"):
                inertia.set(key, "0")
            patched.append(link.get("name"))
    return patched


def append_sharpa_hand(root: ET.Element, sharpa_urdf: Path) -> ET.Element:
    """Append Sharpa's with-flange model and bolt it onto Arm_R7_Link at the official flange."""

    sharpa_root = ET.parse(sharpa_urdf).getroot()
    root_links = {link.get("name") for link in sharpa_root.findall("link")} - {
        joint.find("child").get("link") for joint in sharpa_root.findall("joint")
    }
    if root_links != {"right_hand_flange"}:
        raise ValueError(f"Sharpa with-flange model should be rooted at right_hand_flange, got {sorted(root_links)}")
    for mesh in sharpa_root.iter("mesh"):
        filename = mesh.attrib.get("filename", "")
        if filename.startswith("package://"):
            mesh.attrib["filename"] = f"meshes/sharpa/{Path(filename).name}"
    for tag in ("link", "joint", "material", "transmission"):
        for element in sharpa_root.findall(tag):
            root.append(deepcopy(element))
    root.append(
        ET.Comment(
            " Sharpa flange adapter on the Marvin M6 CCS end flange: Tianji's official tool frame "
            f"({FLANGE_OFFSET_M * 1000:.0f} mm from the J7 axis along Link7 -Y, ccs_m6_40.MvKDCfg / Link7_R.STL), "
            "ISO 9409-1-50-4-M6 pattern, locating pin on Link7 +X = Sharpa +X. "
            "See dex_teleop/docs/tianji-m6-sharpa-mount.md "
        )
    )
    attachment = ET.SubElement(root, "joint", {"name": ATTACHMENT_JOINT, "type": "fixed"})
    ET.SubElement(attachment, "origin", {"xyz": _format(ATTACHMENT_XYZ), "rpy": _format(ATTACHMENT_RPY)})
    ET.SubElement(attachment, "parent", {"link": "Arm_R7_Link"})
    ET.SubElement(attachment, "child", {"link": "right_hand_flange"})
    return attachment


def referenced_meshes(root: ET.Element) -> set[str]:
    return {mesh.get("filename") for mesh in root.iter("mesh") if mesh.get("filename")}


def load_left_home_pose(definition_path: Path) -> dict[str, float]:
    """The upstream Marvin Pro home pose of the left arm and gripper, by joint name."""

    definition = yaml.safe_load(definition_path.read_text(encoding="utf-8"))
    manipulation = definition["manipulation"]
    left_joints = list(manipulation["arm_joint_names"]["left"]) + list(manipulation["finger_joint_names"]["left"])
    default = definition["default_joint_pos"]
    if definition["raw_controller_order"][:2] != ["arm_left", "gripper_left"]:
        raise ValueError(f"{definition_path}: expected the left arm and gripper first in raw_controller_order")
    return {name: float(value) for name, value in zip(left_joints, default[: len(left_joints)], strict=True)}


# ---------------------------------------------------------------------------
# Build


def build(
    tianji_urdf: Path,
    tianji_definition: Path,
    sharpa_urdf: Path,
    sharpa_meshes: Path,
    output_dir: Path,
    *,
    right_arm_pose=DEFAULT_RIGHT_ARM_POSE,
    left_arm_pose=LEFT_ARM_TUCKED_POSE,
) -> Path:
    if not tianji_urdf.is_file():
        raise FileNotFoundError(tianji_urdf)
    if not sharpa_urdf.is_file():
        raise FileNotFoundError(sharpa_urdf)
    if not sharpa_meshes.is_dir():
        raise FileNotFoundError(sharpa_meshes)

    urdf_dir = output_dir / "urdf"
    mesh_dir = urdf_dir / "meshes"
    source_dir = output_dir / "source"
    for directory in (urdf_dir, mesh_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)

    root = ET.parse(tianji_urdf).getroot()
    tianji_dir = tianji_urdf.parent

    shift = reorigin_base_link_at_floor(root, tianji_dir)
    replace_torso_collision(root, mesh_dir)
    left_pose = load_left_home_pose(tianji_definition)
    if left_arm_pose is not None:
        left_pose.update({f"Arm_L{index}_Joint": float(value) for index, value in enumerate(left_arm_pose, start=1)})
    weld_joints(root, left_pose)
    remove_right_tool(root)
    append_sharpa_hand(root, sharpa_urdf)
    patched = give_placeholder_mass(root)

    # Copy exactly the meshes the merged model references (Tianji visual + collision, Sharpa).
    for filename in sorted(referenced_meshes(root)):
        if filename.startswith("meshes/sharpa/"):
            source = sharpa_meshes / Path(filename).name
        elif filename == "meshes/base/unit_box.obj":
            continue
        else:
            source = tianji_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = urdf_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    root.attrib["name"] = MODEL_NAME
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    output_urdf = urdf_dir / f"{MODEL_NAME}.urdf"
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)

    shutil.copy2(output_urdf, source_dir / f"{MODEL_NAME}_merged.urdf")
    shutil.copy2(sharpa_urdf, source_dir / sharpa_urdf.name)
    for filename in ("LICENSE.txt", "NOTICE.txt", "UPSTREAM_COMMIT"):
        candidate = SHARPA_NOTICES / filename
        if candidate.is_file():
            shutil.copy2(candidate, source_dir / f"sharpa_{filename}")
    upstream_notes = tianji_dir / "UPSTREAM.md"
    if upstream_notes.is_file():
        shutil.copy2(upstream_notes, source_dir / "tianji_UPSTREAM.md")
    if MOUNT_DOC.is_file():
        shutil.copy2(MOUNT_DOC, source_dir / "FLANGE.md")
    (source_dir / "BUILD.md").write_text(
        "Generated by dex_teleop/scripts/build_tianji_sharpa_right.py\n\n"
        f"- Tianji dual-arm source: {tianji_urdf}\n"
        f"- Sharpa with-flange model: {sharpa_urdf}\n"
        f"- base_link re-origined {shift * 1000:.1f} mm down to the torso's floor contact plane\n"
        f"- Sharpa right_hand_flange on Arm_R7_Link at xyz={_format(ATTACHMENT_XYZ)} rpy={_format(ATTACHMENT_RPY)}\n"
        f"- placeholder mass on: {', '.join(patched) or 'none'}\n",
        encoding="utf-8",
    )

    write_import_config(output_dir)
    write_robot_definition(output_dir, right_arm_pose)
    return output_urdf


def write_import_config(output_dir: Path) -> Path:
    """OmniGibson importer configuration: every link ships a dedicated collision proxy, so
    nothing is re-decomposed (the Tianji links keep their upstream hulls and primitives, the
    Sharpa links their own collision STLs, the torso its measured boxes)."""

    path = output_dir / f"{MODEL_NAME}.import.yaml"
    config = {
        "urdf_path": f"datasets/omnigibson-robot-assets/models/{MODEL_NAME}/urdf/{MODEL_NAME}.urdf",
        "name": MODEL_NAME,
        "dataset_name": "omnigibson-robot-assets",
        "headless": True,
        "overwrite": True,
        "merge_fixed_joints": False,
        "base_motion": {
            "wheel_links": [],
            "wheel_joints": [],
            "use_sphere_wheels": False,
            "use_holonomic_joints": False,
        },
        "collision": {
            "decompose_method": "convex",
            "hull_count": 8,
            "coacd_links": [],
            "convex_links": [],
            "no_decompose_links": [],
            "no_collision_links": [],
        },
        "eef_vis_links": [],
        "camera_links": [],
        "lidar_links": [],
        "curobo": {},
    }
    path.write_text(
        "# Import configuration for the Marvin M6 CCS right arm + right Sharpa hand (wrist bench).\n"
        + yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return path


def write_robot_definition(output_dir: Path, right_arm_pose) -> Path:
    """The OmniGibson robot definition: one arm named ``right`` carrying the 22-DoF Sharpa hand."""

    if len(right_arm_pose) != 7:
        raise ValueError("right_arm_pose must have 7 joint positions")
    path = output_dir / f"{MODEL_NAME}.yaml"
    arm_pose = ", ".join(f"{float(value):.6f}" for value in right_arm_pose)
    fingers = ", ".join(["0.0"] * len(SHARPA_FINGER_JOINTS))
    finger_links = ",\n          ".join(f'"{name}"' for name in SHARPA_FINGER_LINKS)
    finger_joints = ",\n          ".join(f'"{name}"' for name in SHARPA_FINGER_JOINTS)
    arm_links = ", ".join(f'"{name}"' for name in RIGHT_ARM_LINKS)
    arm_joints = ", ".join(f'"{name}"' for name in RIGHT_ARM_JOINTS)
    text = f"""# Tianji Marvin M6 CCS (Marvin Pro torso, right arm) carrying the right Sharpa Wave hand.
# Built by dex_teleop/scripts/build_tianji_sharpa_right.py; the left arm is welded at the
# upstream home pose, so the only DoFs are the 7 right-arm joints and the 22 Sharpa finger
# joints (the Sharpa upper limits are the closed position: use grasping_direction: upper).
raw_controller_order: ["arm_right", "gripper_right"]
default_controllers:
  arm_right: InverseKinematicsController
  gripper_right: MultiFingerGripperController

usd_path: "models/{MODEL_NAME}/usd/{MODEL_NAME}.usda"
urdf_path: "models/{MODEL_NAME}/urdf/{MODEL_NAME}.urdf"
curobo_path: null

# Right arm (Arm_R1..R7) in the wrist bench's "extended" pose: palm down over the table,
# then the open hand. Solved with Pinocchio IK on this URDF (build script --solve-reset-pose).
default_joint_pos: [{arm_pose},
                    {fingers}]

self_collisions: false
base_footprint_link_name: "base_link"

manipulation:
  n_arms: 1
  arm_names: ["right"]
  arm_link_names:
    right: [{arm_links}]
  arm_joint_names:
    right: [{arm_joints}]
  eef_link_names:
    right: "{PALM_LINK}"
  finger_link_names:
    right: [{finger_links}]
  finger_joint_names:
    right: [{finger_joints}]
  assisted_grasp_start_points:
    right: [["right_thumb_elastomer", [0.0, 0.0, 0.0]]]
  assisted_grasp_end_points:
    right: [["right_index_elastomer", [0.0, 0.0, 0.0]], ["right_middle_elastomer", [0.0, 0.0, 0.0]]]
"""
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Reset pose (Pinocchio IK on the merged URDF)


def solve_reset_pose(
    urdf_path: Path,
    target_position_base: np.ndarray,
    target_rotation_base: Rotation,
    *,
    seed: np.ndarray | None = None,
    iterations: int = 400,
) -> tuple[np.ndarray, float, float]:
    """Joint positions of the 7 right-arm joints placing ``right_hand_C_MC`` at the target.

    Damped least-squares IK with joint-limit clamping and a weak pull toward ``seed``;
    returns (q, position error [m], orientation error [rad]). Pinocchio imports lazily so the
    build itself never needs it.
    """

    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    joint_ids = [model.getJointId(name) for name in RIGHT_ARM_JOINTS]
    q_indices = np.array([model.joints[joint_id].idx_q for joint_id in joint_ids])
    v_indices = np.array([model.joints[joint_id].idx_v for joint_id in joint_ids])
    frame_id = model.getFrameId(PALM_LINK)
    lower, upper = model.lowerPositionLimit[q_indices], model.upperPositionLimit[q_indices]
    q_full = pin.neutral(model)
    q_arm = np.clip(np.zeros(7) if seed is None else np.asarray(seed, dtype=np.float64), lower, upper)
    target = pin.SE3(target_rotation_base.as_matrix(), np.asarray(target_position_base, dtype=np.float64))
    for _ in range(iterations):
        q_full[q_indices] = q_arm
        pin.forwardKinematics(model, data, q_full)
        pin.updateFramePlacements(model, data)
        error = pin.log6(data.oMf[frame_id].actInv(target)).vector  # in the palm frame
        if np.linalg.norm(error) < 1.0e-7:
            break
        jacobian = pin.computeFrameJacobian(model, data, q_full, frame_id, pin.LOCAL)[:, v_indices]
        damping = 1.0e-6 * np.eye(6)
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping, error)
        if seed is not None:
            nullspace = np.eye(7) - np.linalg.pinv(jacobian) @ jacobian
            step = step + 0.05 * nullspace @ (np.asarray(seed) - q_arm)
        q_arm = np.clip(q_arm + np.clip(step, -0.2, 0.2), lower, upper)
    q_full[q_indices] = q_arm
    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)
    placement = data.oMf[frame_id]
    position_error = float(np.linalg.norm(placement.translation - target.translation))
    orientation_error = float(np.linalg.norm(pin.log3(placement.rotation.T @ target.rotation)))
    return q_arm, position_error, orientation_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tianji-urdf", type=Path, default=TIANJI_SOURCE_URDF)
    parser.add_argument("--tianji-definition", type=Path, default=TIANJI_DEFINITION)
    parser.add_argument(
        "--sharpa-repo",
        type=Path,
        default=None,
        help="Checkout of https://github.com/sharpa-robotics/sharpa-urdf-usd-xml (default: the copy kept in "
        "models/franka/franka_dexhand/franka_sharpa_right/source)",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--right-arm-pose",
        type=float,
        nargs=7,
        default=list(DEFAULT_RIGHT_ARM_POSE),
        metavar="Q",
        help="Arm_R1..R7 default_joint_pos for the robot definition (radians)",
    )
    parser.add_argument(
        "--left-arm-pose",
        nargs="+",
        default=[str(value) for value in LEFT_ARM_TUCKED_POSE],
        metavar="Q",
        help="Arm_L1..L7 at which the left arm is welded (radians), or 'home' for the upstream Marvin Pro home pose "
        "(default: hanging beside the torso)",
    )
    parser.add_argument(
        "--finish-import",
        action="store_true",
        help="After import_custom_robot ran, move objects/robot/<model>/{usd,misc} next to the URDF",
    )
    parser.add_argument(
        "--solve-reset-pose",
        action="store_true",
        help="Re-derive the default right-arm pose with Pinocchio IK on the built URDF and check it against "
        "DEFAULT_RIGHT_ARM_POSE (requires pinocchio)",
    )
    parser.add_argument("--restarts", type=int, default=120, help="Random IK restarts for --solve-reset-pose")
    args = parser.parse_args()

    if args.sharpa_repo is not None:
        sharpa_urdf = args.sharpa_repo / "wave_01" / "right_sharpa_wave" / "right_sharpa_wave_with_flange.urdf"
        sharpa_meshes = args.sharpa_repo / "wave_01" / "right_sharpa_wave" / "meshes"
    else:
        sharpa_urdf, sharpa_meshes = SHARPA_URDF, SHARPA_MESHES

    if args.finish_import:
        print(finish_import(args.output_dir))
        return
    if args.solve_reset_pose:
        print(describe_reset_pose(args.output_dir.resolve() / "urdf" / f"{MODEL_NAME}.urdf", restarts=args.restarts))
        return
    output = build(
        args.tianji_urdf.resolve(),
        args.tianji_definition.resolve(),
        sharpa_urdf.resolve(),
        sharpa_meshes.resolve(),
        args.output_dir.resolve(),
        right_arm_pose=tuple(args.right_arm_pose),
        left_arm_pose=None if args.left_arm_pose == ["home"] else tuple(float(value) for value in args.left_arm_pose),
    )
    print(output)


def describe_reset_pose(urdf_path: Path, *, restarts: int = 120, seed: int = 2) -> str:
    """Multi-start IK for the bench's extended pose; reports the best solution and the stored one."""

    import pinocchio as pin

    if not urdf_path.is_file():
        raise FileNotFoundError(f"{urdf_path} is missing; run the build first")
    rotation = Rotation.from_matrix(np.array(RESET_POSE_TARGET_ROTATION, dtype=np.float64))
    target = np.array(RESET_POSE_TARGET_POSITION, dtype=np.float64)
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    q_indices = np.array([model.joints[model.getJointId(name)].idx_q for name in RIGHT_ARM_JOINTS])
    lower, upper = model.lowerPositionLimit[q_indices], model.upperPositionLimit[q_indices]
    elbow_frame = model.getFrameId("Arm_R4_Link")
    shoulder_frame = model.getFrameId("Arm_R1_Link")

    def elbow_position(q_arm):
        q_full = pin.neutral(model)
        q_full[q_indices] = q_arm
        pin.forwardKinematics(model, data, q_full)
        pin.updateFramePlacements(model, data)
        return data.oMf[elbow_frame].translation.copy(), data.oMf[shoulder_frame].translation.copy()

    def cost(q_arm):
        margin = np.degrees(np.minimum(q_arm - lower, upper - q_arm).min())
        elbow, shoulder = elbow_position(q_arm)
        # Prefer a humanlike posture: elbow below the shoulder and not swung in across the body.
        penalty = 100.0 * max(0.0, elbow[2] - (shoulder[2] - 0.10)) + 100.0 * max(0.0, elbow[1] - (shoulder[1] + 0.02))
        return -margin + penalty

    rng = np.random.default_rng(seed)
    solutions = []
    for _ in range(restarts):
        q_arm, position_error, orientation_error = solve_reset_pose(
            urdf_path, target, rotation, seed=rng.uniform(lower, upper), iterations=300
        )
        if position_error < 5.0e-4 and orientation_error < 5.0e-3:
            q_arm, position_error, orientation_error = solve_reset_pose(
                urdf_path, target, rotation, seed=q_arm, iterations=2000
            )
            solutions.append((cost(q_arm), q_arm, position_error, orientation_error))
    if not solutions:
        return "IK did not converge for the extended target from any restart"
    solutions.sort(key=lambda item: item[0])
    best_cost, best, position_error, orientation_error = solutions[0]
    stored = np.array(DEFAULT_RIGHT_ARM_POSE)
    _, stored_position_error, stored_orientation_error = solve_reset_pose(
        urdf_path, target, rotation, seed=stored, iterations=0
    )
    lines = [
        f"target right_hand_C_MC at {target.tolist()} (base_link frame), palm down, fingers +X",
        f"{len(solutions)}/{restarts} restarts converged; best (cost {best_cost:.1f}): {[round(float(v), 6) for v in best]}",
        f"  position error {position_error * 1000:.4f} mm, orientation error {math.degrees(orientation_error):.4f} deg",
        f"  joint-limit margins (deg): {[round(float(v), 1) for v in np.degrees(np.minimum(best - lower, upper - best))]}",
        f"stored DEFAULT_RIGHT_ARM_POSE: {list(DEFAULT_RIGHT_ARM_POSE)}",
        f"  reaches the target within {stored_position_error * 1000:.4f} mm / {math.degrees(stored_orientation_error):.4f} deg",
        f"  differs from the best solution by {np.degrees(np.abs(best - stored)).max():.2f} deg at most",
    ]
    return "\n".join(lines)


def finish_import(output_dir: Path) -> str:
    """Move the importer's output (written under objects/robot/<model>) into the model folder."""

    imported = output_dir.parents[1] / "objects" / "robot" / MODEL_NAME
    moved = []
    for name in ("usd", "misc"):
        source = imported / name
        if not source.is_dir():
            raise FileNotFoundError(f"{source} is missing; run import_custom_robot first")
        destination = output_dir / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(source), str(destination))
        moved.append(str(destination))
    for candidate in (imported / "urdf").glob("*_with_meta_links.urdf") if (imported / "urdf").is_dir() else ():
        shutil.move(str(candidate), str(output_dir / "urdf" / candidate.name))
        moved.append(str(output_dir / "urdf" / candidate.name))
    return "moved:\n" + "\n".join(moved)


if __name__ == "__main__":
    main()
