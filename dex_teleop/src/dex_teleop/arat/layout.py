"""Rigid re-placement of the ARAT apparatus on its table for an arm that cannot work the saved layout.

The saved task instances put the ARAT box across the table from the Franka, its open lid toward the
robot and the objects on the lid: picking up and shelving is a 0.30 m reach *away* from the robot.
An arm with a shorter or narrower working band needs that line somewhere else. A :class:`Layout`
is one rigid motion in the table plane (a rotation about +Z through a pivot, then a translation)
applied to everything the task puts on the table: the box with its lid, tins, planks, bolts, the
task objects, the cups and their water. The table, the lights and the mannequin stay.

Because it is one rigid motion, every relative pose between apparatus parts is what it was, heights
are unchanged, and anything derived from the live objects (the scorer's shelf, tin, peg and pour
targets, BDDL predicates) follows. The rotation must be a multiple of 90 degrees: the scorer and
``OnShelf`` read axis-aligned bounding boxes of the box shell and the tins.

Pure Python (no NumPy): ``arat.scene`` applies it to the scene description before OmniGibson loads.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

# Scene objects a layout never moves: the table carries the apparatus, the mannequin layouts have
# no apparatus (and the scorer assumes the mannequin faces world -X).
FIXED_OBJECT_NAMES = frozenset({"table", "mannequin"})
FIXED_OBJECT_CLASSES = frozenset({"LightObject"})


@dataclass(frozen=True)
class Layout:
    """Rotate by ``yaw_deg`` about +Z through ``about`` (world x, y), then translate by ``translation``."""

    yaw_deg: float = 0.0
    about: tuple[float, float] = (0.0, 0.0)
    translation: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        values = (self.yaw_deg, *self.about, *self.translation)
        if len(self.about) != 2 or len(self.translation) != 2 or not all(math.isfinite(float(v)) for v in values):
            raise ValueError("A layout needs a finite yaw, a pivot (x, y) and a translation (dx, dy)")
        if abs(self.yaw_deg / 90.0 - round(self.yaw_deg / 90.0)) > 1.0e-9:
            raise ValueError(
                f"Layout yaw must be a multiple of 90 degrees (got {self.yaw_deg:g}): the scorer reads "
                "axis-aligned bounding boxes of the box shell and the tins"
            )

    @property
    def is_identity(self) -> bool:
        return self.yaw_deg % 360.0 == 0.0 and tuple(self.translation) == (0.0, 0.0)

    def _cos_sin(self) -> tuple[float, float]:
        # exact for the multiples of 90 degrees a layout is restricted to
        quarter = int(round(self.yaw_deg / 90.0)) % 4
        return ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[quarter]

    def point(self, position: Sequence[float]) -> list[float]:
        """Where a world point goes; z (and any further components) is kept."""

        c, s = self._cos_sin()
        dx, dy = float(position[0]) - self.about[0], float(position[1]) - self.about[1]
        moved = [
            self.about[0] + c * dx - s * dy + self.translation[0],
            self.about[1] + s * dx + c * dy + self.translation[1],
        ]
        return moved + [float(value) for value in position[2:]]

    def direction(self, vector: Sequence[float]) -> list[float]:
        """A world direction (or velocity) after the layout's rotation."""

        c, s = self._cos_sin()
        x, y = float(vector[0]), float(vector[1])
        return [c * x - s * y, s * x + c * y] + [float(value) for value in vector[2:]]

    def orientation(self, quaternion_xyzw: Sequence[float]) -> list[float]:
        """A world orientation after the layout's rotation (yaw quaternion times the given one)."""

        half = math.radians(self.yaw_deg) / 2.0
        qz, qw = math.sin(half), math.cos(half)
        x, y, z, w = (float(value) for value in quaternion_xyzw)
        return [qw * x - qz * y, qw * y + qz * x, qw * z + qz * w, qw * w - qz * z]

    def inverse(self) -> "Layout":
        """The layout that undoes this one."""

        back = Layout(yaw_deg=-self.yaw_deg, about=self.about)
        shift = back.direction([-self.translation[0], -self.translation[1]])
        return Layout(yaw_deg=-self.yaw_deg, about=self.about, translation=(shift[0], shift[1]))

    def to_json(self) -> dict:
        return {"yaw_deg": float(self.yaw_deg), "about": list(self.about), "translation": list(self.translation)}

    @classmethod
    def from_json(cls, raw: Mapping | None) -> "Layout | None":
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("A layout must be a mapping with yaw_deg, about and translation")
        unknown = set(raw).difference({"yaw_deg", "about", "translation"})
        if unknown:
            raise ValueError(f"Unknown layout keys: {', '.join(sorted(unknown))}")
        return cls(
            yaw_deg=float(raw.get("yaw_deg", 0.0)),
            about=tuple(float(value) for value in raw.get("about", (0.0, 0.0))),
            translation=tuple(float(value) for value in raw.get("translation", (0.0, 0.0))),
        )


def moved_object_names(scene: Mapping) -> tuple[str, ...]:
    """The scene objects a layout moves: everything but the table, the mannequin and the lights."""

    init_info = scene["objects_info"]["init_info"]
    return tuple(
        name
        for name, info in init_info.items()
        if name not in FIXED_OBJECT_NAMES and info.get("class_name") not in FIXED_OBJECT_CLASSES
    )


def apply_layout(scene: dict, layout: Layout | None) -> dict:
    """Move the apparatus of a loaded scene description in place and return it.

    Root poses and velocities of the moved objects and the positions (orientations, velocities) of
    every particle are transformed; joint states, scales and the fixed objects are untouched. With
    ``None`` or an identity layout the scene is returned exactly as given.
    """

    if layout is None or layout.is_identity:
        return scene
    registry = scene["state"]["registry"]
    states = registry["object_registry"]
    for name in moved_object_names(scene):
        root = states[name]["root_link"]
        root["pos"] = layout.point(root["pos"])
        root["ori"] = layout.orientation(root["ori"])
        for key in ("lin_vel", "ang_vel"):
            if key in root:
                root[key] = layout.direction(root[key])
        # Poses spelled out in the constructor arguments (none in the version-1 instances) move too.
        args = scene["objects_info"]["init_info"][name].get("args", {})
        if args.get("position") is not None:
            args["position"] = layout.point(args["position"])
        if args.get("orientation") is not None:
            args["orientation"] = layout.orientation(args["orientation"])
    for system in registry.get("system_registry", {}).values():
        for particles in (system.get("particle_states") or {}).values():
            if "particle_positions" in particles:
                particles["particle_positions"] = [layout.point(p) for p in particles["particle_positions"]]
            if "particle_velocities" in particles:
                particles["particle_velocities"] = [layout.direction(v) for v in particles["particle_velocities"]]
            if "particle_orientations" in particles:
                particles["particle_orientations"] = [layout.orientation(q) for q in particles["particle_orientations"]]
    return scene


__all__ = ["FIXED_OBJECT_CLASSES", "FIXED_OBJECT_NAMES", "Layout", "apply_layout", "moved_object_names"]
