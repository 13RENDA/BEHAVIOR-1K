#!/usr/bin/env python3
"""Make the ``tianji_m6_ccs_809`` source description: the Marvin Pro with the long M6-S CCS-809 arms.

``source/tianji_m6_ccs`` (Gento's Marvin Pro description) carries the 696 mm Marvin M6 CCS arms. Tianji's own
export of the Marvin Pro with the 809 mm arms (``Pro-m6-809-gripper/urdf/Pro-m6-809.urdf``, received from the
user on 2026-11-15) describes the same torso and the same arm with a longer upper arm: of the seven link
meshes per side, Link2/4/5/6/7 are byte-identical to the ones already here, Link1 is 3 mm longer and Link3
(the upper arm) 112 mm longer; the J2-to-J3 offset is 400 mm instead of 287 mm, so the J2-to-flange reach is
400 + 314 + 95 = 809 mm.

This script therefore keeps the existing description (torso, frames, link and joint names, grippers, the
teleoperation joint limits) and takes from Tianji's export only what differs:

* the ``Arm_[LR]3_Joint`` origin (the upper-arm length);
* the ``Arm_[LR]1_Link`` and ``Arm_[LR]3_Link`` meshes;
* the inertial parameters of all seven links of both arms;
* the collision shape of ``Arm_[LR]5_Link``: the link's mesh instead of a placeholder cylinder.

Before writing anything it checks that the two descriptions agree on everything else it relies on: every other
arm joint has the same transform, axis and limits-or-wider, and the shoulders sit at the same place on the
torso (Tianji's export has its base frame turned a quarter turn: +Y of this description is its -X).

    python dex_teleop/scripts/make_tianji_m6_809_source.py --vendor-dir <.../Pro-m6-809-gripper>
    python dex_teleop/scripts/build_tianji_sharpa_right.py        # builds from the 809 source by default
"""

from __future__ import annotations

import argparse
import copy
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCES = REPOSITORY_ROOT / "datasets" / "omnigibson-robot-assets" / "source"
BASE_SOURCE = SOURCES / "tianji_m6_ccs"
OUTPUT = SOURCES / "tianji_m6_ccs_809"
URDF_NAME = "tianji_m6_ccs_source.urdf"
ARM_MESHES = Path("meshes") / "marvin_pro" / "meshes" / "m6"
CHANGED_MESHES = (1, 3)
UPPER_ARM_M = {"696": 0.287, "809": 0.400}


def _floats(text: str | None) -> np.ndarray:
    return np.array([float(v) for v in (text or "0 0 0").split()], dtype=np.float64)


def _rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _transform(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    matrix = np.eye(4)
    matrix[:3, :3] = _rpy(_floats(None if origin is None else origin.get("rpy")))
    matrix[:3, 3] = _floats(None if origin is None else origin.get("xyz"))
    return matrix


def _named(root: ET.Element, tag: str, name: str) -> ET.Element:
    for element in root.findall(tag):
        if element.get("name") == name:
            return element
    raise KeyError(f"{tag} {name!r}")


def check_same_arm(base: ET.Element, vendor: ET.Element) -> list[str]:
    """Raise unless the vendor arm is this arm with a longer upper arm; returns the notes for UPSTREAM.md."""

    notes = []
    quarter_turn = np.eye(4)
    quarter_turn[:3, :3] = _rpy(np.array([0.0, 0.0, math.pi / 2.0]))  # this description <- vendor base frame
    for side in "RL":
        # shoulder: base_link -> Base_S -> Arm_S1_Joint here, base_link -> Joint1_S in the vendor's frame
        here = _transform(_named(base, "joint", f"base_link_to_Base_{side}")) @ _transform(_named(base, "joint", f"Arm_{side}1_Joint"))
        there = quarter_turn @ _transform(_named(vendor, "joint", f"Joint1_{side}"))
        if not np.allclose(here, there, atol=2.0e-3):
            raise ValueError(f"shoulder {side} differs:\n{np.round(here, 4)}\n{np.round(there, 4)}")
        for index in range(2, 8):
            ours, theirs = _named(base, "joint", f"Arm_{side}{index}_Joint"), _named(vendor, "joint", f"Joint{index}_{side}")
            a, b = _transform(ours), _transform(theirs)
            if not np.allclose(a[:3, :3], b[:3, :3], atol=1.0e-4):
                raise ValueError(f"joint {index}{side}: frames differ")
            difference = b[:3, 3] - a[:3, 3]
            expected = np.array([0.0, UPPER_ARM_M["809"] - UPPER_ARM_M["696"], 0.0]) if index == 3 else np.zeros(3)
            if not np.allclose(difference, expected, atol=1.0e-6):
                raise ValueError(f"joint {index}{side}: offset differs by {difference}")
            if not np.allclose(_floats(ours.find("axis").get("xyz")), _floats(theirs.find("axis").get("xyz"))):
                raise ValueError(f"joint {index}{side}: axis differs")
        for index in range(1, 8):
            ours = _named(base, "joint", f"Arm_{side}{index}_Joint").find("limit")
            theirs = _named(vendor, "joint", f"Joint{index}_{side}").find("limit")
            lower, upper = float(ours.get("lower")), float(ours.get("upper"))
            vendor_lower, vendor_upper = float(theirs.get("lower")), float(theirs.get("upper"))
            if lower < vendor_lower - 1.0e-4 or upper > vendor_upper + 1.0e-4:
                raise ValueError(f"joint {index}{side}: kept limits ({lower}, {upper}) exceed the vendor's ({vendor_lower}, {vendor_upper})")
            if abs(lower - vendor_lower) > 1.0e-4 or abs(upper - vendor_upper) > 1.0e-4:
                notes.append(f"Arm_{side}{index}_Joint keeps ({lower}, {upper}) rad; Tianji's export allows ({vendor_lower}, {vendor_upper})")
            if float(ours.get("effort")) != float(theirs.get("effort")):
                raise ValueError(f"joint {index}{side}: effort differs")
    return notes


def make(vendor_dir: Path, output: Path = OUTPUT, base_source: Path = BASE_SOURCE) -> Path:
    vendor_urdf = vendor_dir / "urdf" / "Pro-m6-809.urdf"
    base_tree = ET.parse(base_source / URDF_NAME)
    base, vendor = base_tree.getroot(), ET.parse(vendor_urdf).getroot()
    notes = check_same_arm(base, vendor)

    for side in "RL":
        joint = _named(base, "joint", f"Arm_{side}3_Joint")
        xyz = _floats(joint.find("origin").get("xyz"))
        xyz[1] = _floats(_named(vendor, "joint", f"Joint3_{side}").find("origin").get("xyz"))[1]
        joint.find("origin").set("xyz", " ".join(f"{v:.10g}" for v in xyz))
        for index in range(1, 8):
            link = _named(base, "link", f"Arm_{side}{index}_Link")
            inertial = _named(vendor, "link", f"Link{index}_{side}").find("inertial")
            link.remove(link.find("inertial"))
            link.insert(0, copy.deepcopy(inertial))

    # Arm_[LR]5_Link (the forearm, wrist end) carries a placeholder collision cylinder in the 696 description:
    # radius 0.06 m, length 0.32 m, centred on the wrist, i.e. 0.16 m long on the hand's side of the wrist
    # where the link itself ends after 0.023 m. With the forearm vertical above a table (how the long arm
    # picks up close to the body) that phantom reached into the tabletop. Tianji's export collides with the
    # link's own mesh, as every other arm link here does; so does this description.
    for side in "RL":
        link = _named(base, "link", f"Arm_{side}5_Link")
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            if geometry.find("cylinder") is None:
                raise ValueError(f"Arm_{side}5_Link: expected the placeholder cylinder")
            geometry.remove(geometry.find("cylinder"))
            ET.SubElement(geometry, "mesh", {"filename": str(ARM_MESHES / f"Arm_{side}5_Link.STL")})

    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(base_source / "meshes", output / "meshes")
    for side in "RL":
        for index in CHANGED_MESHES:
            shutil.copy2(vendor_dir / "meshes" / f"Link{index}_{side}.STL", output / ARM_MESHES / f"Arm_{side}{index}_Link.STL")
    (output / "vendor").mkdir()
    shutil.copy2(vendor_urdf, output / "vendor" / vendor_urdf.name)
    ET.indent(base_tree, space="  ")
    base_tree.write(output / URDF_NAME, encoding="utf-8", xml_declaration=True)
    upstream = (base_source / "UPSTREAM.md").read_text()
    (output / "UPSTREAM.md").write_text(
        "# tianji_m6_ccs_809 source (made 2026-11-15)\n\n"
        "The Marvin Pro of `source/tianji_m6_ccs` with the 809 mm Marvin M6-S CCS-809 arms, made by\n"
        "`dex_teleop/scripts/make_tianji_m6_809_source.py` from that description and Tianji's own export\n"
        "`Pro-m6-809-gripper/urdf/Pro-m6-809.urdf` (received from the user as `Pro-m6-809-gripper(1).zip`,\n"
        "files dated 2026-09-29 to 2026-10-06; a copy of the URDF is in `vendor/`).\n\n"
        "Taken from Tianji's export: the `Arm_[LR]3_Joint` offset (0.400 m instead of 0.287 m), the\n"
        "`Arm_[LR]1_Link` and `Arm_[LR]3_Link` meshes (its `Link1_*`, `Link3_*`), and the inertial parameters\n"
        "of the seven links of both arms. Everything else is the 696 description unchanged: torso, frames,\n"
        "names, grippers. The script checked that the other joint frames, axes and efforts are the same in\n"
        "both and that the shoulders are at the same place (the export's base frame is turned a quarter turn).\n\n"
        "The placeholder collision cylinder of `Arm_[LR]5_Link` (0.32 m long, centred on the wrist) is replaced\n"
        "by the link's mesh, as in Tianji's export: the cylinder reached 0.14 m past the end of the link.\n\n"
        "Joint limits are the ones of the 696 description (the teleoperation limits). Where Tianji's export\n"
        "allows more:\n\n" + "".join(f"- {note}\n" for note in notes) + "\n"
        "Not taken: the export's torso mesh (same outline as the one here), its camera frames and its gripper.\n\n"
        "---\n\nThe description this was made from:\n\n" + upstream
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vendor-dir", type=Path, required=True, help="The extracted Pro-m6-809-gripper folder (with urdf/ and meshes/)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(make(args.vendor_dir.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
