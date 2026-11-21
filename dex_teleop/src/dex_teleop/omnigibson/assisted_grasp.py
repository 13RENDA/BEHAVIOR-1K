"""Movement-driven assisted grasping for the teleoperated dexterous hand.

JoyLo enables OmniGibson's built-in assisted grasping by constructing its robot with
``grasping_mode="assisted"``: once the gripper closes around an object, a constraint welds
the object to the hand so it can be carried reliably. The built-in decision logic is
written for simple grippers and pre-determines the participating fingers:

- Its ray filter runs only between the robot definition's fixed AG points (for the Sharpa
  hand: thumb tip to index and middle tips), so a grasp using any other digit combination
  -- e.g. the ARAT thumb-ring pinch -- can never trigger.
- Its closing test treats any gripper DOF commanded away from its open limit as "applying
  a grasp". A retargeted 22-DOF hand always has joints away from their limits, so a grasp
  would trigger on any sustained touch and, once established, never release.

This module keeps OmniGibson's assisted-grasp machinery (constraint creation, release,
state serialization for recordings and replay) but derives the grasp decision from what
the teleoperator's hand is actually doing: which digits touch the candidate object, and
whether their contact normals oppose. The thumb must be among the touching digits (every
ARAT grasp is thumb-opposed), but beyond that the digits taking part in a grasp are
chosen by the operator's movement, never by the robot definition.

While an object is welded, the grasping digits' joints are frozen (the launcher feeds
:attr:`AssistedGraspSupervisor.frozen_fingers` back into the action adapter) so the hand
visibly holds its grasp pose instead of wiggling around a rigidly attached object, and so
the operator's deepening squeeze cannot fight the weld through the arm. Uninvolved
fingers keep tracking. Because frozen fingers hold their contacts forever, release is
decided from the operator's *commanded* hand instead of from contacts: the weld drops
once the digits still flexed near their grasp posture no longer form a holdable set.

The supervisor drives the robot's grasp-handling methods directly, which is the usage
OmniGibson documents for ``disable_grasp_handling=True``. The decision layer operates on
plain data so it can be unit-tested without a simulator.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
import logging
import math
from typing import Callable, Iterable, Mapping, Sequence


LOGGER = logging.getLogger(__name__)

PALM = "palm"
THUMB = "thumb"
PAD = "pad"
CRITERIA = ("contact_opposition", "friction_closure")
WELD_MODES = ("rigid", "compliant")
# How the operator's opening hand releases a weld. "digits": once the digits still near their weld-time flexion can
# no longer hold (fewer than two fingers, or palm + finger), i.e. nearly every grasping digit must open. "mean": once
# the grasping digits' mean retreat from weld time reaches the margin. EMG-decoded fingers move little and the thumb
# and pinky least, so with "digits" a five-digit grasp could not be released by EMG (2026-11-16).
RELEASE_RULES = ("digits", "mean")
GRAVITY_M_S2 = 9.81
# Convention of robot.get_finger_contact_data, measured in the simulator on 2026-11-13 (a 0.49 kg block
# resting on the Sharpa hand, 360 contact samples, /home/svl/claude_work/emg_teleop/ag_contact_convention.py):
# the reported normal points OUT of the touched object, toward the hand link (normal . (object COM -
# contact point) < 0 in every sample), and the force is a signed scalar normal force in newtons (its
# magnitudes summed to 0.90 m g at rest; an impulse would be 120 times smaller). The friction-closure
# criterion works with inward normals (into the object) and force magnitudes.
REPORTED_NORMAL_SIGN_INTO_OBJECT = -1.0
FRICTION_CONE_EDGES = 8


@dataclass(frozen=True)
class DigitContact:
    """One contact point between a hand part and a candidate object link."""

    digit: str  # "thumb" | "index" | "middle" | "ring" | "pinky" | "palm"
    part: str  # "pad" | "dorsal" | "palm"
    normal: tuple[float, float, float] | None  # world frame, as reported (out of the object); None when unavailable
    position: tuple[float, float, float] | None = None  # world frame contact point
    force_n: float = 0.0  # magnitude of the normal force, newtons


@dataclass(frozen=True)
class AssistedGraspConfig:
    # Mirrors OmniGibson's m.GRASP_WINDOW: opposition must persist this long to weld.
    grasp_window_s: float = 0.3
    # How long the operator's hand must stay opened before the weld drops; absorbs
    # tracking noise and brief dropouts while carrying.
    release_window_s: float = 0.15
    # Two contacts from different digits oppose when their world normals are at least
    # 90 degrees apart (dot <= 0.0). This accepts corner and lateral grasps while
    # rejecting several digits resting on the same face of the object.
    opposition_max_normal_dot: float = 0.0
    # A grasping digit counts as opened once its flexion commands retreat this far (mean
    # over the digit's non-abduction joints) from their values at weld time.
    release_open_margin_rad: float = 0.25
    release_rule: str = "digits"
    # Frozen flexion targets sit at the measured joint positions plus this closing bias,
    # capped at the operator's live command: enough sustained pressure to keep firm
    # contact, without the deep-penetration squeeze that kicks the arm through the weld.
    frozen_squeeze_bias_rad: float = 0.05
    # A weld only establishes when the thumb touches the object: every ARAT grasp is
    # thumb-opposed, and thumb-less patterns (the scoring guide's inappropriate or
    # score-0 movements) should not be magically assisted. Release is unaffected.
    require_thumb: bool = True
    # Snap the weld before the operator can wedge the held object deep into static
    # geometry (a real grip would slip); authored as physics:breakForce / breakTorque on
    # the weld joint. None disables either limit.
    weld_break_force: float | None = 100.0
    weld_break_torque: float | None = 30.0
    # If the held object separates this far from its weld-time offset to the palm, the
    # weld has broken (or is being violated beyond credibility); treat it as released.
    weld_drift_release_m: float = 0.05
    # How the held object is attached to the palm.
    # "rigid": OmniGibson's fixed joint with the break limits above (the original behaviour). Measured on
    # 2026-11-15 by re-simulating recorded welds (ag_resim.py): with the 100 N break force the joint broke
    # within 0-21 control steps of every marble and tube weld, because the stiff arm and the frozen,
    # still-squeezing fingers load a rigid joint far beyond 100 N as soon as the object touches what it
    # rests on; after that only finger friction carries the object, which small objects do not survive.
    # "compliant": a 6-DOF spring-damper joint. Its force saturates at weld_break_force /
    # weld_break_torque instead of snapping (an overloaded grip stretches, and the drift release above
    # drops it), hand-object collisions are filtered while the object is held so the frozen fingers do
    # not fight the spring, and the grasping digits freeze at their measured positions (no squeeze bias).
    weld_mode: str = "rigid"
    # Compliant weld: linear and angular stiffness of the spring. 3000 N/m sags 3 mm per kilogram held
    # and reaches the 100 N limit at 3.3 cm; 20 Nm/rad lets a tube on a peg align itself (1.8 degrees of
    # play on the ARAT bolt costs 0.6 Nm). Linear damping is critical for the held mass (clipped).
    compliant_stiffness_n_per_m: float = 3000.0
    compliant_angular_stiffness_nm_per_rad: float = 20.0
    compliant_angular_damping_nm_s_per_rad: float = 0.2
    # After a compliant weld is released, hand-object collisions come back once no hand link's bounding
    # box overlaps the object's, or after this long at the latest.
    compliant_collision_restore_timeout_s: float = 1.0
    # Which rule decides that a weld may be established. "contact_opposition": thumb touching, two
    # digits (or palm + finger), one pair of contact normals at least 90 degrees apart. "friction_closure":
    # thumb pad contact, closure under Coulomb friction, enough normal force for the object's weight,
    # and the grasping digits commanded past their contact (see evaluate_friction_closure and
    # FrictionClosureWindow). Release, freezing and weld limits are the same for both.
    criterion: str = "contact_opposition"
    # Coulomb friction coefficient. 0.5 is what PhysX applies here: neither the Sharpa hand links nor the
    # ARAT objects bind a physics material (checked 2026-11-13), so both use the PhysX default material
    # (static = dynamic friction = 0.5).
    friction_mu: float = 0.5
    # (c) mu * (F_thumb + F_opp) >= weight_factor * m * g on the window mean, and both sides >= the floor.
    friction_weight_factor: float = 2.0
    friction_force_floor_n: float = 0.5
    # (d) each grasping digit's mean (live target - measured) over its flexion joints, window mean, rad.
    # PROVISIONAL: to be set from glove debug traces (low end of the squeeze depth of glove grasps that
    # satisfy (a)-(c)); none existed on 2026-11-13.
    friction_squeeze_delta_rad: float = 0.03
    # Positive-span branch of (b): the origin must lie this far inside the convex hull of the unit
    # friction-cone edge vectors.
    friction_closure_margin: float = 0.02

    def __post_init__(self) -> None:
        if self.criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}, got {self.criterion!r}")
        if self.weld_mode not in WELD_MODES:
            raise ValueError(f"weld_mode must be one of {WELD_MODES}, got {self.weld_mode!r}")
        if self.release_rule not in RELEASE_RULES:
            raise ValueError(f"release_rule must be one of {RELEASE_RULES}, got {self.release_rule!r}")
        if not math.isfinite(self.release_open_margin_rad) or self.release_open_margin_rad <= 0:
            raise ValueError("release_open_margin_rad must be positive and finite")
        for name in ("compliant_stiffness_n_per_m", "compliant_angular_stiffness_nm_per_rad"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("compliant_angular_damping_nm_s_per_rad", "compliant_collision_restore_timeout_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")
        for name in ("friction_mu", "friction_weight_factor"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("friction_force_floor_n", "friction_squeeze_delta_rad", "friction_closure_margin"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")


def _revalidate_physics_views() -> bool:
    """Rebuild the physics tensor views if a collision-filter edit invalidated them; True if it had to.

    Changing an object's filtered pairs makes PhysX re-create its collision shapes. For an object with several
    collision meshes that a tensor view holds (2026-11-16: plastic_tumbler_2 of pour water, 'collisions/wall_15
    was deleted while being used by a shape in a tensor view class'), that invalidates the shared simulation
    view and the very next pose read fails ('Failed to get rigid body transforms from backend'). OmniGibson's
    own answer after such edits (remove_prim, adding objects) is sim.update_handles().
    """

    import omnigibson as og
    import omnigibson.lazy as lazy

    view = lazy.isaacsim.core.simulation_manager.SimulationManager._physics_sim_view
    if view is None or view.is_valid:
        return False
    og.sim.update_handles()
    return True


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def can_hold_digits(digits: Iterable[str]) -> bool:
    """Digits able to keep holding: two distinct fingers, or the palm plus a finger."""
    digits = frozenset(digits)
    fingers = digits - {PALM}
    return len(fingers) >= 2 or (PALM in digits and len(fingers) >= 1)


def can_hold(contacts: Sequence[DigitContact]) -> bool:
    """Contact-pattern form of :func:`can_hold_digits`."""
    return can_hold_digits(contact.digit for contact in contacts)


def has_opposition(contacts: Sequence[DigitContact], max_normal_dot: float) -> bool:
    """Whether the contact pattern pinches the object rather than resting on it.

    All normals come from the robot's contact view with one consistent sensor/filter
    convention, so the pairwise comparison is invariant to which way that convention
    points: a pinch yields anti-parallel normals, digits resting on the same face
    yield parallel ones.
    """
    if not can_hold(contacts):
        return False
    for first, second in combinations(contacts, 2):
        if first.digit == second.digit or first.normal is None or second.normal is None:
            continue
        if _dot(first.normal, second.normal) <= max_normal_dot:
            return True
    return False


# --------------------------------------------------------------------------- friction closure


def _unit(vector: Sequence[float]) -> tuple[float, float, float] | None:
    norm = math.sqrt(sum(float(c) ** 2 for c in vector))
    if not math.isfinite(norm) or norm < 1e-9:
        return None
    return tuple(float(c) / norm for c in vector)


def inward_normal(contact: DigitContact) -> tuple[float, float, float] | None:
    """The contact normal pointing into the object (see REPORTED_NORMAL_SIGN_INTO_OBJECT)."""

    if contact.normal is None:
        return None
    return _unit(tuple(REPORTED_NORMAL_SIGN_INTO_OBJECT * float(c) for c in contact.normal))


def _digit_aggregates(contacts: Sequence[DigitContact]) -> dict[str, tuple[tuple, tuple]]:
    """Per digit: (force-weighted mean contact point, unit force-weighted mean inward normal)."""

    grouped: dict[str, list[tuple[float, tuple, tuple]]] = {}
    for contact in contacts:
        normal = inward_normal(contact)
        if normal is None or contact.position is None:
            continue
        grouped.setdefault(contact.digit, []).append((max(float(contact.force_n), 0.0), contact.position, normal))
    aggregates = {}
    for digit, items in grouped.items():
        total = sum(weight for weight, _, _ in items)
        weights = [weight / total for weight, _, _ in items] if total > 0 else [1.0 / len(items)] * len(items)
        point = tuple(sum(w * float(position[axis]) for w, (_, position, _) in zip(weights, items)) for axis in range(3))
        normal = _unit(tuple(sum(w * normal[axis] for w, (_, _, normal) in zip(weights, items)) for axis in range(3)))
        if normal is not None:
            aggregates[digit] = (point, normal)
    return aggregates


def antipodal_pair(contacts: Sequence[DigitContact], mu: float) -> tuple[str, str] | None:
    """A (thumb, other digit or palm) pair whose connecting line lies inside both friction cones."""

    aggregates = _digit_aggregates(contacts)
    if THUMB not in aggregates:
        return None
    half_angle = math.atan(mu)
    thumb_point, thumb_normal = aggregates[THUMB]
    for digit, (point, normal) in sorted(aggregates.items()):
        if digit == THUMB:
            continue
        line = _unit(tuple(b - a for a, b in zip(thumb_point, point)))
        if line is None:
            continue
        thumb_angle = math.acos(max(-1.0, min(1.0, _dot(line, thumb_normal))))
        other_angle = math.acos(max(-1.0, min(1.0, -_dot(line, normal))))
        if thumb_angle <= half_angle + 1e-9 and other_angle <= half_angle + 1e-9:
            return (THUMB, digit)
    return None


def friction_cone_edges(normal: Sequence[float], mu: float, edges: int = FRICTION_CONE_EDGES) -> list[tuple]:
    """Unit edge vectors of the linearized friction cone about an inward unit normal."""

    reference = (1.0, 0.0, 0.0) if abs(normal[0]) < 0.9 else (0.0, 1.0, 0.0)
    tangent_1 = _unit(_cross(normal, reference))
    tangent_2 = _cross(normal, tangent_1)
    result = []
    for k in range(edges):
        angle = 2.0 * math.pi * k / edges
        edge = tuple(
            normal[axis] + mu * (math.cos(angle) * tangent_1[axis] + math.sin(angle) * tangent_2[axis])
            for axis in range(3)
        )
        result.append(_unit(edge))
    return result


def cones_positively_span(contacts: Sequence[DigitContact], mu: float, margin: float) -> bool:
    """Whether three or more contacts including the thumb can push in every direction.

    True when the origin lies strictly inside the convex hull of the linearized friction-cone edge
    vectors of all contacts, at least ``margin`` from every facet (edge vectors are unit length).
    """

    usable = [(contact, inward_normal(contact)) for contact in contacts]
    usable = [(contact, normal) for contact, normal in usable if normal is not None]
    if len(usable) < 3 or not any(contact.digit == THUMB for contact, _ in usable):
        return False
    import numpy as np
    from scipy.spatial import ConvexHull, QhullError

    vectors = np.asarray([edge for _, normal in usable for edge in friction_cone_edges(normal, mu)], dtype=np.float64)
    try:
        hull = ConvexHull(vectors)
    except (QhullError, ValueError):
        return False  # flat or degenerate: the cones do not span R^3
    # Facet equations are n . x + offset <= 0 inside the hull, with unit n: -offset is the origin's depth.
    return bool(np.all(-hull.equations[:, 3] >= margin))


@dataclass(frozen=True)
class FrictionClosureStep:
    """Instantaneous part of the friction-closure criterion for one object link."""

    thumb_pad: bool
    closure: str | None  # "antipodal:<digit>" | "span" | None
    thumb_force_n: float
    opposing_force_n: float
    digits: frozenset

    @property
    def passes(self) -> bool:
        return self.thumb_pad and self.closure is not None


def evaluate_friction_closure(contacts: Sequence[DigitContact], *, mu: float, margin: float) -> FrictionClosureStep:
    """(a) thumb pad and (b) closure for one step, plus the forces (c) averages over the window."""

    thumb_pad = any(contact.digit == THUMB and contact.part == PAD for contact in contacts)
    closure = None
    pair = antipodal_pair(contacts, mu)
    if pair is not None:
        closure = f"antipodal:{pair[1]}"
    elif cones_positively_span(contacts, mu, margin):
        closure = "span"
    return FrictionClosureStep(
        thumb_pad=thumb_pad,
        closure=closure,
        thumb_force_n=sum(float(c.force_n) for c in contacts if c.digit == THUMB),
        opposing_force_n=sum(float(c.force_n) for c in contacts if c.digit != THUMB),
        digits=frozenset(contact.digit for contact in contacts),
    )


def squeeze_depth_by_digit(
    live_fingers: Mapping[str, float], measured_fingers: Mapping[str, float], side: str
) -> dict[str, float]:
    """Per digit: mean (live target - measured position) over its flexion (non-abduction) joints, rad."""

    sums: dict[str, list[float]] = {}
    prefix = f"{side}_"
    for name, live in live_fingers.items():
        if not name.startswith(prefix) or name.endswith("_AA") or name not in measured_fingers:
            continue
        digit = name.removeprefix(prefix).split("_", 1)[0]
        sums.setdefault(digit, []).append(float(live) - float(measured_fingers[name]))
    return {digit: sum(values) / len(values) for digit, values in sums.items()}


def friction_window_verdict(
    steps: Sequence[FrictionClosureStep],
    squeezes: Sequence[Mapping[str, float]],
    *,
    mass_kg: float,
    config: AssistedGraspConfig,
) -> dict:
    """(c) force and (d) intent on the window mean; ``steps`` all passed (a) and (b)."""

    count = len(steps)
    thumb = sum(step.thumb_force_n for step in steps) / count
    opposing = sum(step.opposing_force_n for step in steps) / count
    required = config.friction_weight_factor * float(mass_kg) * GRAVITY_M_S2 / config.friction_mu
    force_ok = (thumb + opposing) >= required and min(thumb, opposing) >= config.friction_force_floor_n
    digits = sorted(steps[-1].digits - {PALM})
    squeeze = {
        digit: sum(float(sample.get(digit, 0.0)) for sample in squeezes) / len(squeezes) for digit in digits
    }
    lacking = sorted(digit for digit, depth in squeeze.items() if depth < config.friction_squeeze_delta_rad)
    return {
        "thumb_force_n": thumb,
        "opposing_force_n": opposing,
        "required_total_force_n": required,
        "force_ok": bool(force_ok),
        "squeeze_depth_rad": squeeze,
        "digits_below_delta": lacking,
        "intent_ok": not lacking,
        "weld": bool(force_ok and not lacking),
    }


def _as_normal(vector) -> tuple[float, float, float] | None:
    normal = (float(vector[0]), float(vector[1]), float(vector[2]))
    if normal == (0.0, 0.0, 0.0):
        return None
    return normal


def _as_point(vector) -> tuple[float, float, float] | None:
    values = tuple(float(c) for c in vector)
    if len(values) != 3 or not all(math.isfinite(c) for c in values):
        return None
    return values


def _force_magnitude(force) -> float:
    """Normal-force magnitude in newtons from the reported signed scalar (or a force vector)."""

    if hasattr(force, "tolist"):
        force = force.tolist()
    values = [float(force)] if isinstance(force, (int, float)) else [float(c) for c in force]
    magnitude = abs(values[0]) if len(values) == 1 else math.sqrt(sum(c * c for c in values))
    return magnitude if math.isfinite(magnitude) else 0.0


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _rotate_inverse(quaternion_xyzw, vector) -> tuple[float, float, float]:
    """Rotate a world-frame vector into the frame of the given (x, y, z, w) quaternion."""
    x, y, z, w = (float(component) for component in quaternion_xyzw)
    axis = (-x, -y, -z)
    vector = tuple(float(component) for component in vector)
    doubled = tuple(2.0 * component for component in _cross(axis, vector))
    correction = _cross(axis, doubled)
    return tuple(vector[i] + w * doubled[i] + correction[i] for i in range(3))


# The joint type of a compliant weld: a generic (6-DOF, all free) UsdPhysics.Joint held by drives.
# OmniGibson records it as a "FixedJoint" (see _establish_weld), so a recorded state replays with a
# rigid joint; replay reloads the state every step, so nothing there depends on the spring.
COMPLIANT_JOINT_TYPE = "Joint"
COMPLIANT_LINEAR_DAMPING_RANGE = (10.0, 200.0)
# Drive force limit authored when the break limit is disabled (a finite stand-in for "no limit").
COMPLIANT_UNLIMITED_FORCE = 1.0e6


def describe_weld(config: AssistedGraspConfig) -> str:
    """One line for the launchers to print at start: which weld holds a grasped object, and its limits."""

    force = "no force limit" if config.weld_break_force is None else f"{config.weld_break_force:g} N"
    torque = "no torque limit" if config.weld_break_torque is None else f"{config.weld_break_torque:g} Nm"
    if config.weld_mode == "compliant":
        return (
            f"Assisted-grasp weld: COMPLIANT (spring {config.compliant_stiffness_n_per_m:g} N/m, "
            f"{config.compliant_angular_stiffness_nm_per_rad:g} Nm/rad, saturating at {force} / {torque}; "
            f"hand-object collisions off while held); weld after {config.grasp_window_s:g} s of the rule holding"
        )
    return (
        f"Assisted-grasp weld: RIGID (fixed joint, breaks at {force} / {torque}). Small objects (marble, washer, "
        "tubes) are not carried by it: add --assisted-grasp-weld compliant"
    )


def compliant_drive_parameters(config: AssistedGraspConfig, mass_kg: float) -> dict[str, tuple[float, float, float | None]]:
    """USD drive (stiffness, damping, max force) per axis of a compliant weld.

    Linear axes are in N/m, N s/m and N. Angular axes are per degree, as UsdPhysics.DriveAPI expects
    (Nm/deg, Nm s/deg), with the limit in Nm. Linear damping is critical for the held mass, clipped.
    """

    mass = mass_kg if math.isfinite(mass_kg) and mass_kg > 0 else 0.1
    linear_k = float(config.compliant_stiffness_n_per_m)
    low, high = COMPLIANT_LINEAR_DAMPING_RANGE
    linear_d = min(high, max(low, 2.0 * math.sqrt(linear_k * mass)))
    per_degree = math.pi / 180.0
    angular_k = float(config.compliant_angular_stiffness_nm_per_rad) * per_degree
    angular_d = float(config.compliant_angular_damping_nm_s_per_rad) * per_degree
    force = None if config.weld_break_force is None else float(config.weld_break_force)
    torque = None if config.weld_break_torque is None else float(config.weld_break_torque)
    drives = {axis: (linear_k, linear_d, force) for axis in ("transX", "transY", "transZ")}
    drives.update({axis: (angular_k, angular_d, torque) for axis in ("rotX", "rotY", "rotZ")})
    return drives


def boxes_overlap(lo_a, hi_a, lo_b, hi_b, margin: float = 0.002) -> bool:
    """Whether two axis-aligned boxes come within ``margin`` metres of each other on every axis."""

    return all(la - margin <= hb and lb - margin <= ha for la, ha, lb, hb in zip(lo_a, hi_a, lo_b, hi_b))


def _as_floats(values) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


class AssistedGraspSupervisor:
    """Drives one arm's OmniGibson assisted-grasp machinery from hand contacts.

    Call :meth:`step` once per control step while teleoperation is engaged, and
    :meth:`reset` whenever the environment resets.
    """

    def __init__(
        self,
        robot,
        semantics,
        *,
        dt: float,
        config: AssistedGraspConfig = AssistedGraspConfig(),
        announce: Callable[[str], None] | None = None,
    ) -> None:
        if robot.grasping_mode != "assisted":
            raise ValueError("Assisted-grasp supervision requires grasping_mode='assisted'")
        if not robot._disable_grasp_handling:
            raise ValueError(
                "Assisted-grasp supervision requires disable_grasp_handling=True so the built-in "
                "closing-command heuristic does not fight the contact-driven decisions"
            )
        if getattr(robot, "_grasping_direction", "upper") != "upper":
            raise ValueError(
                "Assisted-grasp supervision assumes grasping_direction='upper' (larger joint value = closed)"
            )
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.robot = robot
        self.config = config
        self._semantics = semantics
        self._dt = dt
        self._announce = announce
        self._arm = robot.arm_names[0]
        self._pending_key: tuple[str, str] | None = None
        self._pending_elapsed = 0.0
        self._release_elapsed = 0.0
        # While holding: the digits that formed the grasp, the frozen targets fed back to
        # the action adapter, the operator's live posture at weld time (release baseline),
        # each grasping digit's flexion joints, and the object's weld-time offset in the
        # palm frame (for detecting a broken or violated weld).
        self._grasp_digits: frozenset[str] = frozenset()
        self._frozen_fingers: dict[str, float] | None = None
        self._open_baseline: dict[str, float] = {}
        self._flexion_by_digit: dict[str, tuple[str, ...]] = {}
        self._weld_offset: tuple[float, float, float] | None = None
        # Compliant weld: (hand link, object link) pairs whose collision is filtered, the object they
        # belong to, and the time since its release while the filter is still waiting to come off.
        self._compliant_weld = False
        self._unbreakable_weld = False  # a rigid weld made in compliant mode (objects with meta links)
        self._filtered_pairs: list[tuple[object, object]] = []
        self._filtered_object = None
        self._restore_elapsed = 0.0
        self._last_event: dict | None = None
        # friction_closure: the last grasp_window_s of steps that passed (a) and (b) for the pending
        # object link, as (FrictionClosureStep, squeeze depth by digit), and the latest (c)/(d) verdict.
        self._window_steps = max(1, math.ceil(config.grasp_window_s / dt - 1e-9))
        self._friction_window: deque = deque(maxlen=self._window_steps)
        self._last_gate: dict | None = None

    @property
    def arm(self) -> str:
        return self._arm

    @property
    def control_dt(self) -> float:
        return self._dt

    @property
    def last_event(self) -> dict | None:
        """Structured event emitted by the most recent :meth:`step`, if any."""

        return None if self._last_event is None else dict(self._last_event)

    @property
    def frozen_fingers(self) -> dict[str, float] | None:
        """Joint targets to hold while grasping (by joint name), or None when not holding.

        The launcher feeds this into ``SharpaActionAdapter.action(..., frozen_fingers=...)``
        so the grasping digits keep their weld-time posture while uninvolved fingers track.
        """
        return None if self._frozen_fingers is None else dict(self._frozen_fingers)

    @property
    def holding_digits(self) -> tuple[str, ...]:
        """The digits holding the welded object (empty when nothing is held); for the haptic grasp cue."""

        if self.robot._ag_obj_in_hand[self._arm] is None:
            return ()
        return tuple(sorted(self._grasp_digits))

    def reset(self) -> None:
        """Clear decision windows and drop any active weld (no-op when none exists)."""
        self._last_event = None
        self._pending_key = None
        self._pending_elapsed = 0.0
        self._release_elapsed = 0.0
        self._friction_window.clear()
        self._last_gate = None
        self._clear_freeze()
        self.robot.release_grasp_immediately(arm=self._arm)
        self._restore_hand_collisions()

    def step(self, live_fingers: Mapping[str, float], measured_fingers: Mapping[str, float]) -> None:
        """Advance the grasp/release decision by one control step.

        Args:
            live_fingers: the operator's current retargeted finger targets by joint name
                (what would be commanded without any freeze)
            measured_fingers: the robot's current measured finger joint positions by joint name
        """
        self._last_event = None
        held = self.robot._ag_obj_in_hand[self._arm]
        if held is not None:
            if self._frozen_fingers is None:
                # A weld this supervisor did not create (e.g. restored from a saved state).
                self._adopt_existing_grasp(held, live_fingers, measured_fingers)
            self._step_release(held, live_fingers)
        else:
            self._release_elapsed = 0.0
            self._clear_freeze()
            self._step_collision_restore()
            self._step_establish(self._contacts_by_object_link(), live_fingers, measured_fingers)

    def _step_release(self, held, live_fingers) -> None:
        self._pending_key = None
        self._pending_elapsed = 0.0
        self._friction_window.clear()
        if self._weld_offset is not None:
            offset = self._held_offset(held)
            drift = math.sqrt(sum((a - b) ** 2 for a, b in zip(offset, self._weld_offset)))
            if drift > self.config.weld_drift_release_m:
                # PhysX broke the weld (break force) or is violating it beyond credibility
                # (object wedged into static geometry); stop pretending it is held.
                held_name = held.name
                self.robot.release_grasp_immediately(arm=self._arm)
                self._release_elapsed = 0.0
                self._clear_freeze()
                self._last_event = {"kind": "broke", "object": held_name, "drift_m": drift}
                self._say(f"assisted grasp broke on {held_name} (weld separated {drift * 100:.0f} cm)")
                return
        if self.config.release_rule == "mean":
            retreat = self.flexion_retreat(live_fingers)
            still_holding = not retreat or sum(retreat.values()) / len(retreat) < self.config.release_open_margin_rad
        else:
            engaged = {digit for digit in self._grasp_digits if self._digit_engaged(digit, live_fingers)}
            still_holding = can_hold_digits(engaged)
        if still_holding:
            self._release_elapsed = 0.0
            return
        self._release_elapsed += self._dt
        if self._release_elapsed < self.config.release_window_s:
            return
        held_name = held.name
        self.robot.release_grasp_immediately(arm=self._arm)
        self._release_elapsed = 0.0
        self._clear_freeze()
        self._last_event = {"kind": "released", "object": held_name}
        self._say(f"assisted grasp released {held_name}")

    def release_now(self) -> bool:
        """Drop the held object at once (the operator's release key); False if nothing is held."""
        held = self.robot._ag_obj_in_hand[self._arm]
        if held is None:
            return False
        self.robot.release_grasp_immediately(arm=self._arm)
        self._release_elapsed = 0.0
        self._pending_key = None
        self._pending_elapsed = 0.0
        self._clear_freeze()
        self._last_event = {"kind": "released", "object": held.name, "manual": True}
        self._say(f"assisted grasp released {held.name} (release key)")
        return True

    def _digit_engaged(self, digit: str, live_fingers: Mapping[str, float]) -> bool:
        """Whether the operator still holds this digit near its weld-time flexion."""
        flexion = self._flexion_by_digit.get(digit, ())
        retreats = [self._open_baseline[name] - float(live_fingers[name]) for name in flexion if name in live_fingers]
        if not retreats:
            return True  # the palm (or a digit without joint data) cannot signal opening
        return sum(retreats) / len(retreats) <= self.config.release_open_margin_rad

    def _capture_freeze(self, digits, live_fingers, measured_fingers) -> None:
        """Freeze the grasping digits' joints and record the operator's grasp posture."""
        side = self._semantics.side
        # With hand-object collisions filtered nothing would stop a squeeze bias: hold the measured pose.
        squeeze_bias = 0.0 if self._weld_is_compliant() else self.config.frozen_squeeze_bias_rad
        frozen: dict[str, float] = {}
        baseline: dict[str, float] = {}
        flexion_by_digit: dict[str, tuple[str, ...]] = {}
        for digit in digits:
            if digit == PALM:
                continue
            joints = tuple(name for name in live_fingers if name.startswith(f"{side}_{digit}_"))
            flexion = tuple(name for name in joints if not name.endswith("_AA"))
            flexion_by_digit[digit] = flexion
            for name in joints:
                live = float(live_fingers[name])
                measured = float(measured_fingers.get(name, live))
                if name in flexion:
                    # Hold gentle pressure at the surface instead of the operator's deep
                    # squeeze target, which would fight the weld through the arm.
                    frozen[name] = min(live, measured + squeeze_bias)
                    baseline[name] = live
                else:
                    frozen[name] = measured
        self._grasp_digits = frozenset(digits)
        self._frozen_fingers = frozen
        self._open_baseline = baseline
        self._flexion_by_digit = flexion_by_digit

    def _adopt_existing_grasp(self, held, live_fingers, measured_fingers) -> None:
        side = self._semantics.side
        digits = frozenset(
            name.removeprefix(f"{side}_").split("_", 1)[0] for name in live_fingers if name.startswith(f"{side}_")
        )
        self._capture_freeze(digits, live_fingers, measured_fingers)
        self._weld_offset = self._held_offset(held)
        self._author_weld_break_limits()
        self._last_event = {"kind": "adopted", "object": held.name, "digits": sorted(digits)}
        self._say("assisted grasp adopted an existing weld; all fingers hold until the hand opens")

    def diagnostic_state(self) -> dict:
        """Return the decision and weld state in a JSON-safe representation."""

        held = self.robot._ag_obj_in_hand[self._arm]
        drift = None
        target_link_name = None
        if held is not None:
            params = getattr(self.robot, "_ag_obj_constraint_params", {}).get(self._arm)
            target_link_name = None if params is None else params.get("target_link_name")
            if self._weld_offset is not None:
                offset = self._held_offset(held)
                drift = math.sqrt(sum((a - b) ** 2 for a, b in zip(offset, self._weld_offset)))
        return {
            "status": "held" if held is not None else ("pending" if self._pending_key is not None else "idle"),
            "held_object": None if held is None else held.name,
            "target_link_name": target_link_name,
            "grasp_digits": sorted(self._grasp_digits),
            "pending_object": None if self._pending_key is None else self._pending_key[0],
            "pending_link_name": None if self._pending_key is None else self._pending_key[1],
            "pending_elapsed_s": self._pending_elapsed,
            "release_elapsed_s": self._release_elapsed,
            "weld_drift_m": drift,
            "criterion": self.config.criterion,
            "weld_mode": None if held is None else ("compliant" if self._weld_is_compliant() else "rigid"),
            "hand_collisions_filtered": bool(self._filtered_pairs),
            "friction_gate": None if self._last_gate is None else dict(self._last_gate),
        }

    def flexion_retreat(self, live_fingers: Mapping[str, float]) -> dict[str, float]:
        """While welded: how far each grasp digit's commanded flexion has retreated from weld time, rad.

        The release rule compares exactly this value with ``release_open_margin_rad``.
        """

        retreat = {}
        for digit in sorted(self._grasp_digits):
            flexion = self._flexion_by_digit.get(digit, ())
            values = [self._open_baseline[name] - float(live_fingers[name]) for name in flexion if name in live_fingers]
            if values:
                retreat[digit] = sum(values) / len(values)
        return retreat

    def squeeze_depth(self, live_fingers: Mapping[str, float], measured_fingers: Mapping[str, float]) -> dict:
        """Per digit: mean (live target - measured) over its flexion joints, rad (criterion (d))."""

        return squeeze_depth_by_digit(live_fingers, measured_fingers, self._semantics.side)

    def contact_snapshot(self) -> list[dict]:
        """Every hand contact on a scene object, plain data, for the diagnostic trace."""

        snapshot = []
        for obj, link_name, contacts in self._contacts_by_object_link().values():
            snapshot.append(
                {
                    "object": obj.name,
                    "link_name": link_name,
                    "mass_kg": self._weld_mass(obj, link_name),
                    "graspable": bool(self._graspable(obj, link_name)),
                    "contacts": [
                        {
                            "digit": contact.digit,
                            "part": contact.part,
                            "position_m": None if contact.position is None else list(contact.position),
                            "normal": None if contact.normal is None else list(contact.normal),
                            "force_n": contact.force_n,
                        }
                        for contact in contacts
                    ],
                }
            )
        return snapshot

    def classify_hand_link(self, link_name: str):
        """Expose the shared hand-link classification to diagnostic collectors."""

        return self._semantics.classify_link(link_name)

    def _held_offset(self, held) -> tuple[float, float, float]:
        """The held object's position in the palm frame; constant while the weld is intact."""
        object_position = held.get_position_orientation()[0]
        eef_position, eef_quaternion = self.robot.eef_links[self._arm].get_position_orientation()
        world = tuple(float(o) - float(e) for o, e in zip(object_position, eef_position))
        return _rotate_inverse(eef_quaternion, world)

    def _establish_weld(self, obj, link_name) -> None:
        """Create the assisted-grasp joint: OmniGibson's own, or a free 6-DOF joint for a compliant weld."""
        robot = self.robot
        if self.config.weld_mode != "compliant":
            robot._maybe_establish_grasp(target_obj=obj, target_link_name=link_name, arm=self._arm)
            return
        if self._has_meta_links(obj):
            # The pour-water cups (their fillable volume is a meta link). Their hand collisions cannot be
            # filtered (see _filter_hand_collisions), so the spring weld would work against the frozen
            # fingers, and next to the water it ends the GPU physics (CUDA error 700): both live sessions
            # of 2026-11-18 that gripped the filled cup, and on replay every time the weld formed at one
            # particular step (11 of 11), and after a spring weld had been made and released (1 of 2).
            # OmniGibson's fixed joint never did in the same replays (11 runs). It gets no break limits:
            # with 100 N it let go of the cup once or twice in each replayed carry. See the change log of
            # docs/assisted-grasp-reference.md, 2026-11-19.
            robot._maybe_establish_grasp(target_obj=obj, target_link_name=link_name, arm=self._arm)
            if robot._ag_obj_in_hand[self._arm] is obj:
                self._unbreakable_weld = True
                self._say(f"assisted grasp welds {obj.name} rigidly, without break limits (object with meta links)")
            return
        # Hand-object collisions of a previous compliant weld must be back before a new one starts.
        self._restore_hand_collisions()
        # The same final checks as robot._maybe_establish_grasp, with a generic joint in place of the
        # fixed one. A moving link of an articulated object keeps OmniGibson's spherical joint.
        joint_type = robot._get_assisted_grasp_joint_type(obj, link_name)
        if joint_type != "FixedJoint":
            robot._maybe_establish_grasp(target_obj=obj, target_link_name=link_name, arm=self._arm)
            return
        contact_position = robot._find_finger_contact_position(self._arm, obj.links[link_name].prim_path)
        if contact_position is None:
            return
        robot._establish_grasp(obj, link_name, self._arm, contact_position, COMPLIANT_JOINT_TYPE)
        if robot._ag_obj_in_hand[self._arm] is obj:
            # OmniGibson's bookkeeping only knows fixed and spherical welds and serializes this one as
            # fixed. Say so in the live record too: loading back a state saved during this hold then
            # finds the weld unchanged and leaves the spring joint alone, instead of swapping the joint
            # prim for a fixed one in place (which left PhysX with stale handles when tried, 2026-11-15).
            robot._ag_obj_constraint_params[self._arm]["joint_type"] = "FixedJoint"
            self._compliant_weld = True

    @staticmethod
    def _has_meta_links(obj) -> bool:
        return any(str(name).startswith("meta__") for name in obj.links)

    def _weld_joint_prim(self):
        constraints = getattr(self.robot, "_ag_obj_constraints", None)
        return None if constraints is None else constraints.get(self._arm)

    def _weld_is_compliant(self) -> bool:
        """Whether the current weld is the spring joint this supervisor creates in compliant mode."""
        return self._compliant_weld and self.robot._ag_obj_in_hand[self._arm] is not None

    def _author_weld_break_limits(self) -> None:
        """Author the weld's load limits: break force/torque on a rigid weld so a wedged grip slips like
        a real one, or the spring drives (saturating at the same limits) on a compliant weld."""
        joint_prim = self._weld_joint_prim()
        if joint_prim is None:
            return
        if self._weld_is_compliant():
            self._author_compliant_drives(joint_prim)
            self._filter_hand_collisions(self.robot._ag_obj_in_hand[self._arm])
            return
        if self._unbreakable_weld:
            return
        if self.config.weld_break_force is None and self.config.weld_break_torque is None:
            return
        import omnigibson as og

        with og.sim.editing_usd():
            if self.config.weld_break_force is not None:
                joint_prim.GetAttribute("physics:breakForce").Set(float(self.config.weld_break_force))
            if self.config.weld_break_torque is not None:
                joint_prim.GetAttribute("physics:breakTorque").Set(float(self.config.weld_break_torque))

    def _author_compliant_drives(self, joint_prim) -> None:
        """Turn the free 6-DOF weld joint into a spring: position drives to zero on every axis."""
        params = self.robot._ag_obj_constraint_params[self._arm]
        mass = self._weld_mass(params["target_obj"], params["target_link_name"])
        self._apply_joint_drives(joint_prim, compliant_drive_parameters(self.config, mass))

    def _apply_joint_drives(self, joint_prim, drives) -> None:
        import omnigibson as og
        import omnigibson.lazy as lazy

        with og.sim.editing_usd():
            # Author every attribute before the schema is applied, so PhysX first meets each drive complete.
            # (Applying the schema first makes PhysX reject a half-specified drive once per axis; authoring
            # inside an Sdf.ChangeBlock left the drives dead and crashed the GPU solver, 2026-11-15.)
            for axis, (stiffness, damping, max_force) in drives.items():
                drive = lazy.pxr.UsdPhysics.DriveAPI(joint_prim, axis)
                drive.CreateMaxForceAttr(COMPLIANT_UNLIMITED_FORCE if max_force is None else max_force)
                drive.CreateTypeAttr("force")
                drive.CreateTargetPositionAttr(0.0)
                drive.CreateTargetVelocityAttr(0.0)
                drive.CreateDampingAttr(damping)
                drive.CreateStiffnessAttr(stiffness)
            for axis in drives:
                lazy.pxr.UsdPhysics.DriveAPI.Apply(joint_prim, axis)

    def _hand_links(self) -> list:
        """Every link of the hand (fingers and palm), not the arm."""
        robot = self.robot
        prefix = f"{self._semantics.side}_"
        links = [link for name, link in robot.links.items() if name.startswith(prefix)]
        if not links:
            links = list(robot.finger_links[self._arm]) + [robot.eef_links[self._arm]]
        return links

    def _filter_hand_collisions(self, obj) -> None:
        """Stop the hand colliding with the object it holds on a compliant weld."""
        if obj is None or self._filtered_pairs:
            return
        if self._has_meta_links(obj):
            # Objects with meta links (a cup's fillable volume, 'meta__base_link_fillable_...'): editing their
            # filtered pairs makes PhysX re-create their collision shapes, which invalidates the physics views,
            # and rebuilding the views then fails on the meta link (pour water, plastic_tumbler_2, 2026-11-16:
            # the launcher died on the next pose read). The weld itself is fine; only the collisions stay on.
            self._say(f"assisted grasp keeps hand collisions with {obj.name} on (object with meta links)")
            return
        pairs = [(hand_link, object_link) for hand_link in self._hand_links() for object_link in obj.links.values()]
        self._set_collision_filter(pairs, filtered=True)
        self._filtered_pairs = pairs
        self._filtered_object = obj
        self._restore_elapsed = 0.0

    def _restore_hand_collisions(self) -> None:
        if not self._filtered_pairs:
            return
        self._set_collision_filter(self._filtered_pairs, filtered=False)
        self._filtered_pairs = []
        self._filtered_object = None
        self._restore_elapsed = 0.0

    def _set_collision_filter(self, pairs, *, filtered: bool) -> None:
        """Add or remove the (hand link, object link) pairs from each other's filtered-collision list."""
        import omnigibson as og

        # Resolve the relationships first: reading a link's filter API may itself edit USD (it applies
        # the API on first use), and editing_usd() contexts cannot nest.
        edits = [
            (first._collision_filter_api.GetFilteredPairsRel(), second.prim_path)
            for hand_link, object_link in pairs
            for first, second in ((hand_link, object_link), (object_link, hand_link))
        ]
        with og.sim.editing_usd():
            for relationship, target in edits:
                if filtered:
                    relationship.AddTarget(target)
                else:
                    relationship.RemoveTarget(target)
        _revalidate_physics_views()

    def _hand_overlaps_filtered_object(self) -> bool:
        """Whether any hand link's bounding box still overlaps the released object's."""
        try:
            object_boxes = [link.aabb for link in self._filtered_object.links.values()]
            hand_boxes = [link.aabb for link in {id(hand): hand for hand, _ in self._filtered_pairs}.values()]
        except Exception:  # noqa: BLE001 - a missing box must not keep the filter on forever
            return False
        return any(
            boxes_overlap(_as_floats(hand_lo), _as_floats(hand_hi), _as_floats(obj_lo), _as_floats(obj_hi))
            for hand_lo, hand_hi in hand_boxes
            for obj_lo, obj_hi in object_boxes
        )

    def _step_collision_restore(self) -> None:
        """After a compliant weld is dropped: bring hand-object collisions back once the hand is clear."""
        if not self._filtered_pairs:
            return
        self._restore_elapsed += self._dt
        timed_out = self._restore_elapsed >= self.config.compliant_collision_restore_timeout_s
        if timed_out or not self._hand_overlaps_filtered_object():
            name = getattr(self._filtered_object, "name", "object")
            self._restore_hand_collisions()
            if timed_out:
                self._say(f"assisted grasp restored hand collisions with {name} while the hand still overlapped it")

    def _clear_freeze(self) -> None:
        self._compliant_weld = False
        self._unbreakable_weld = False  # a rigid weld made in compliant mode (objects with meta links)
        self._grasp_digits = frozenset()
        self._frozen_fingers = None
        self._open_baseline = {}
        self._flexion_by_digit = {}
        self._weld_offset = None

    def _step_establish(self, contacts_by_link, live_fingers, measured_fingers) -> None:
        candidate = self._select_candidate(contacts_by_link)
        key = None if candidate is None else (candidate[0].name, candidate[1])
        if key != self._pending_key:
            self._pending_key = key
            self._pending_elapsed = 0.0
            self._friction_window.clear()
            self._last_gate = None
        if key is None:
            return
        self._pending_elapsed += self._dt
        obj, link_name, contacts = candidate
        if self.config.criterion == "friction_closure":
            # (a) and (b) held on this step (they select the candidate); (c) and (d) are judged on the
            # mean of the last grasp_window_s, and keep being re-judged while the hold persists.
            self._friction_window.append(
                (
                    evaluate_friction_closure(
                        contacts, mu=self.config.friction_mu, margin=self.config.friction_closure_margin
                    ),
                    squeeze_depth_by_digit(live_fingers, measured_fingers, self._semantics.side),
                )
            )
            if len(self._friction_window) < self._window_steps:
                return
            self._last_gate = friction_window_verdict(
                [step for step, _ in self._friction_window],
                [squeeze for _, squeeze in self._friction_window],
                mass_kg=self._weld_mass(obj, link_name),
                config=self.config,
            )
            if not self._last_gate["weld"]:
                return
        elif self._pending_elapsed < self.config.grasp_window_s:
            return
        self._pending_key = None
        self._pending_elapsed = 0.0
        self._friction_window.clear()
        self._establish_weld(obj, link_name)
        if self.robot._ag_obj_in_hand[self._arm] is obj:
            digits = frozenset(contact.digit for contact in contacts)
            self._capture_freeze(digits, live_fingers, measured_fingers)
            self._weld_offset = self._held_offset(obj)
            self._author_weld_break_limits()
            self._last_event = {
                "kind": "welded",
                "object": obj.name,
                "target_link_name": link_name,
                "digits": sorted(digits),
            }
            if self._weld_is_compliant():
                self._last_event["weld_mode"] = "compliant"
            self._say(f"assisted grasp welded {obj.name}:{link_name} via {'+'.join(sorted(digits))}")

    def _contacts_by_object_link(self):
        """Group the arm's current contact points as {(object name, link name): (obj, link name, contacts)}."""
        grouped: dict[tuple[str, str], tuple[object, str, list[DigitContact]]] = {}
        for hand_path, other_path, _position, normal, _force, _separation in self.robot.get_finger_contact_data(
            self._arm
        ):
            classified = self._semantics.classify_link(hand_path.rsplit("/", 1)[-1])
            if classified is None:
                continue
            digit, part = classified
            obj_prim_path, link_name = other_path.rsplit("/", 1)
            obj = self.robot.scene.object_registry("prim_path", obj_prim_path, None)
            if obj is None or link_name not in obj.links:
                continue
            entry = grouped.setdefault((obj.name, link_name), (obj, link_name, []))
            entry[2].append(
                DigitContact(
                    digit=digit,
                    part=part,
                    normal=_as_normal(normal),
                    position=_as_point(_position),
                    force_n=_force_magnitude(_force),
                )
            )
        return grouped

    def _weld_mass(self, obj, link_name) -> float:
        """Mass the grasp must carry: the whole object, or the touched link of a fixed-base object."""

        try:
            if not getattr(obj, "fixed_base", False) and getattr(obj, "mass", None) is not None:
                return float(obj.mass)
            return float(obj.links[link_name].mass)
        except (TypeError, ValueError, KeyError, AttributeError):
            return float("nan")

    def _select_candidate(self, contacts_by_link):
        candidates = []
        for obj, link_name, contacts in contacts_by_link.values():
            if self.config.criterion == "friction_closure":
                if not evaluate_friction_closure(
                    contacts, mu=self.config.friction_mu, margin=self.config.friction_closure_margin
                ).passes:
                    continue
            else:
                if self.config.require_thumb and not any(contact.digit == THUMB for contact in contacts):
                    continue
                if not has_opposition(contacts, self.config.opposition_max_normal_dot):
                    continue
            if not self._graspable(obj, link_name):
                continue
            candidates.append((self._distance_to_eef(obj.links[link_name]), (obj, link_name, contacts)))
        if not candidates:
            return None
        return min(candidates, key=lambda pair: pair[0])[1]

    def _graspable(self, obj, link_name) -> bool:
        if getattr(obj, "kinematic_only", False):
            return False
        # The root link of a fixed-base object can never be lifted; welding to it would
        # pin the hand (table, tins, mannequin). Articulated links of fixed objects
        # (e.g. the box lid) keep OmniGibson's stock door-grasp behavior.
        if obj.fixed_base and link_name == obj.root_link_name:
            return False
        return self.robot._get_assisted_grasp_joint_type(obj, link_name) is not None

    def _distance_to_eef(self, link) -> float:
        eef_position = self.robot.eef_links[self._arm].get_position_orientation()[0]
        link_position = link.get_position_orientation()[0]
        return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(link_position, eef_position)))

    def _say(self, message: str) -> None:
        if self._announce is not None:
            self._announce(message)
        else:
            LOGGER.info(message)
