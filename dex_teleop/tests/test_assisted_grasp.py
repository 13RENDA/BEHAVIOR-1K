"""Unit tests for movement-driven assisted grasping (no simulator required)."""

from dataclasses import replace

import pytest

from dex_teleop.arat.eval.hand_model import HandSemantics
from dex_teleop.hands import SHARPA_ACTION_JOINTS
from dex_teleop.omnigibson.assisted_grasp import (
    AssistedGraspConfig,
    AssistedGraspSupervisor,
    DigitContact,
    boxes_overlap,
    can_hold,
    can_hold_digits,
    compliant_drive_parameters,
    describe_weld,
    has_opposition,
)

UP = (0.0, 0.0, 1.0)
DOWN = (0.0, 0.0, -1.0)
LEFT = (-1.0, 0.0, 0.0)
RIGHT = (1.0, 0.0, 0.0)

# Clean multiples of the 0.1 s test step so no window check lands on a float boundary:
# welds on the 3rd consecutive step, releases on the 4th.
DT = 0.1
TEST_CONFIG = AssistedGraspConfig(grasp_window_s=0.25, release_window_s=0.35)
WELD_STEPS = 3
RELEASE_STEPS = 4

MAX_DOT = TEST_CONFIG.opposition_max_normal_dot

LIVE_CLOSED = 0.9
MEASURED_CLOSED = 0.8  # fingers physically stopped at the object surface


def digit_of(joint_name):
    return joint_name.split("_")[1]


def hand(flexion, aa=0.0):
    return {name: (aa if name.endswith("_AA") else flexion) for name in SHARPA_ACTION_JOINTS}


def opened(fingers, digits, amount=0.6):
    out = dict(fingers)
    for name in out:
        if digit_of(name) in digits and not name.endswith("_AA"):
            out[name] -= amount
    return out


LIVE = hand(LIVE_CLOSED)
MEASURED = hand(MEASURED_CLOSED)


def contact(digit, normal, part="pad"):
    return DigitContact(digit=digit, part=part, normal=normal)


# --------------------------------------------------------------------------- decision layer


def test_can_hold_requires_two_fingers_or_palm_plus_finger():
    assert not can_hold([])
    assert not can_hold([contact("index", UP)])
    assert not can_hold([contact("index", UP), contact("index", DOWN)])  # one finger, two links
    assert not can_hold([contact("palm", UP)])
    assert can_hold([contact("thumb", UP), contact("ring", DOWN)])
    assert can_hold([contact("palm", UP), contact("index", DOWN)])
    assert not can_hold_digits({"thumb"})
    assert not can_hold_digits({"palm"})
    assert can_hold_digits({"palm", "middle"})
    assert can_hold_digits({"index", "middle"})


def test_pinch_between_thumb_and_ring_opposes():
    assert has_opposition([contact("thumb", LEFT), contact("ring", RIGHT)], MAX_DOT)


def test_two_links_of_one_finger_do_not_oppose():
    assert not has_opposition([contact("index", LEFT), contact("index", RIGHT)], MAX_DOT)


def test_flat_hand_resting_on_object_does_not_oppose():
    resting = [contact("thumb", UP), contact("index", UP), contact("middle", UP)]
    assert not has_opposition(resting, MAX_DOT)


def test_corner_grasp_at_ninety_degrees_opposes():
    assert has_opposition([contact("thumb", UP), contact("index", RIGHT)], MAX_DOT)


def test_palm_and_finger_oppose():
    assert has_opposition([contact("palm", UP, part="palm"), contact("middle", DOWN)], MAX_DOT)


def test_contacts_without_normals_never_oppose():
    assert not has_opposition([contact("thumb", None), contact("ring", None)], MAX_DOT)


# --------------------------------------------------------------------------- supervisor


class FakeLink:
    def __init__(self, prim_path, position=(0.0, 0.0, 0.0), mass=0.1):
        self.prim_path = prim_path
        self.mass = mass
        self._position = position

    def get_position_orientation(self):
        return self._position, (0.0, 0.0, 0.0, 1.0)


class FakeObject:
    def __init__(self, name, prim_path, link_names, *, fixed_base=False, kinematic_only=False, mass=0.1):
        self.name = name
        self.prim_path = prim_path
        self.fixed_base = fixed_base
        self.kinematic_only = kinematic_only
        self.root_link_name = link_names[0]
        self.links = {link_name: FakeLink(f"{prim_path}/{link_name}", mass=mass) for link_name in link_names}

    def get_position_orientation(self):
        return self.links[self.root_link_name].get_position_orientation()


class FakeScene:
    def __init__(self, objects):
        self._by_prim_path = {obj.prim_path: obj for obj in objects}

    def object_registry(self, key, value, default=None):
        assert key == "prim_path"
        return self._by_prim_path.get(value, default)


class FakeRobot:
    _grasping_direction = "upper"

    def __init__(self, objects, grasping_mode="assisted", disable_grasp_handling=True, weld_succeeds=True):
        self.grasping_mode = grasping_mode
        self._disable_grasp_handling = disable_grasp_handling
        self.arm_names = ["0"]
        self._ag_obj_in_hand = {"0": None}
        self.scene = FakeScene(objects)
        self.eef_links = {"0": FakeLink("/World/robot/right_hand_C_MC")}
        self.contact_rows = []
        self.establish_calls = []
        self.release_calls = 0
        self._weld_succeeds = weld_succeeds

    def get_finger_contact_data(self, arm):
        assert arm == "0"
        return list(self.contact_rows)

    def _get_assisted_grasp_joint_type(self, target_obj, target_link_name):
        return None if target_obj.links[target_link_name].mass > 10.0 else "FixedJoint"

    def _maybe_establish_grasp(self, target_obj, target_link_name, arm):
        self.establish_calls.append((target_obj.name, target_link_name))
        if self._weld_succeeds:
            self._ag_obj_in_hand[arm] = target_obj

    def release_grasp_immediately(self, arm="default"):
        if self._ag_obj_in_hand[arm] is not None:
            self.release_calls += 1
        self._ag_obj_in_hand[arm] = None


def hand_path(link_name):
    return f"/World/robot/{link_name}"


def contact_row(hand_link, obj, normal, link_name=None):
    link_name = obj.root_link_name if link_name is None else link_name
    return (hand_path(hand_link), f"{obj.prim_path}/{link_name}", (0.0, 0.0, 0.0), normal, (0.0, 0.0, 0.0), 0.0)


def pinch_rows(obj, link_name=None):
    return [
        contact_row("right_thumb_elastomer", obj, LEFT, link_name),
        contact_row("right_ring_elastomer", obj, RIGHT, link_name),
    ]


def make_supervisor(robot, announce=None, config=TEST_CONFIG):
    return AssistedGraspSupervisor(robot, HandSemantics.sharpa("right"), dt=DT, config=config, announce=announce)


def block(name="block", position=(0.1, 0.0, 0.0), **kwargs):
    obj = FakeObject(name, f"/World/scene_0/{name}", ["base_link"], **kwargs)
    obj.links["base_link"]._position = position
    return obj


def weld_pinch(supervisor, robot, obj):
    robot.contact_rows = pinch_rows(obj)
    for _ in range(WELD_STEPS):
        supervisor.step(LIVE, MEASURED)
    assert robot._ag_obj_in_hand["0"] is obj


def test_supervisor_welds_after_sustained_opposition():
    obj = block()
    robot = FakeRobot([obj])
    messages = []
    supervisor = make_supervisor(robot, announce=messages.append)
    robot.contact_rows = pinch_rows(obj)
    for _ in range(WELD_STEPS - 1):
        supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == []
    supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == [("block", "base_link")]
    assert robot._ag_obj_in_hand["0"] is obj
    assert messages == ["assisted grasp welded block:base_link via ring+thumb"]
    assert supervisor.last_event == {
        "kind": "welded",
        "object": "block",
        "target_link_name": "base_link",
        "digits": ["ring", "thumb"],
    }


def test_weld_freezes_grasping_digits_at_measured_posture():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    weld_pinch(supervisor, robot, obj)
    frozen = supervisor.frozen_fingers
    assert frozen is not None
    assert set(frozen) == {n for n in SHARPA_ACTION_JOINTS if digit_of(n) in ("thumb", "ring")}
    for name, value in frozen.items():
        if name.endswith("_AA"):
            assert value == pytest.approx(0.0)  # abduction frozen where the finger physically is
        else:
            # min(live 0.9, measured 0.8 + bias 0.05): surface posture plus gentle pressure
            assert value == pytest.approx(MEASURED_CLOSED + TEST_CONFIG.frozen_squeeze_bias_rad)


def test_freeze_never_exceeds_the_operators_live_command():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    barely_closed = hand(0.3)
    robot.contact_rows = pinch_rows(obj)
    for _ in range(WELD_STEPS):
        supervisor.step(barely_closed, MEASURED)
    for name, value in supervisor.frozen_fingers.items():
        if not name.endswith("_AA"):
            assert value == pytest.approx(0.3)  # capped at live, not measured + bias


def test_supervisor_window_resets_when_contact_is_lost():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    robot.contact_rows = pinch_rows(obj)
    for _ in range(WELD_STEPS - 1):
        supervisor.step(LIVE, MEASURED)
    robot.contact_rows = []
    supervisor.step(LIVE, MEASURED)
    robot.contact_rows = pinch_rows(obj)
    for _ in range(WELD_STEPS - 1):
        supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == []
    supervisor.step(LIVE, MEASURED)
    assert robot._ag_obj_in_hand["0"] is obj


def test_supervisor_never_welds_a_resting_hand():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    robot.contact_rows = [
        contact_row("right_thumb_elastomer", obj, UP),
        contact_row("right_index_elastomer", obj, UP),
        contact_row("right_middle_elastomer", obj, UP),
    ]
    for _ in range(WELD_STEPS * 4):
        supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == []


def test_supervisor_skips_fixed_root_and_kinematic_and_heavy_objects():
    fixed = block("mannequin", fixed_base=True)
    kinematic = block("prop", kinematic_only=True)
    heavy = block("anvil", mass=25.0)
    robot = FakeRobot([fixed, kinematic, heavy])
    supervisor = make_supervisor(robot)
    for obj in (fixed, kinematic, heavy):
        robot.contact_rows = pinch_rows(obj)
        for _ in range(WELD_STEPS * 2):
            supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == []


def test_supervisor_allows_articulated_link_of_fixed_object():
    box = FakeObject("arat_box", "/World/scene_0/arat_box", ["base_link", "front_cover_link"], fixed_base=True)
    robot = FakeRobot([box])
    supervisor = make_supervisor(robot)
    robot.contact_rows = pinch_rows(box, link_name="front_cover_link")
    for _ in range(WELD_STEPS):
        supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == [("arat_box", "front_cover_link")]


def test_supervisor_prefers_the_candidate_closest_to_the_palm():
    near = block("near", position=(0.1, 0.0, 0.0))
    far = block("far", position=(0.5, 0.0, 0.0))
    robot = FakeRobot([near, far])
    supervisor = make_supervisor(robot)
    robot.contact_rows = pinch_rows(far) + pinch_rows(near)
    for _ in range(WELD_STEPS):
        supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == [("near", "base_link")]


def test_supervisor_releases_when_the_grasping_digits_open():
    obj = block()
    robot = FakeRobot([obj])
    messages = []
    supervisor = make_supervisor(robot, announce=messages.append)
    weld_pinch(supervisor, robot, obj)
    open_live = opened(LIVE, ("thumb", "ring"))
    for _ in range(RELEASE_STEPS - 1):
        supervisor.step(open_live, MEASURED)
    assert robot.release_calls == 0
    supervisor.step(open_live, MEASURED)
    assert robot.release_calls == 1
    assert robot._ag_obj_in_hand["0"] is None
    assert supervisor.frozen_fingers is None
    assert messages[-1] == "assisted grasp released block"
    assert supervisor.last_event == {"kind": "released", "object": "block"}


def test_supervisor_releases_when_opposition_breaks_on_one_side():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    weld_pinch(supervisor, robot, obj)
    open_live = opened(LIVE, ("ring",))  # only the ring opens; thumb alone cannot hold
    for _ in range(RELEASE_STEPS):
        supervisor.step(open_live, MEASURED)
    assert robot.release_calls == 1


def test_supervisor_spherical_hold_survives_a_single_digit_glitch():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    robot.contact_rows = [
        contact_row("right_thumb_elastomer", obj, LEFT),
        contact_row("right_index_elastomer", obj, RIGHT),
        contact_row("right_middle_elastomer", obj, RIGHT),
        contact_row("right_ring_elastomer", obj, RIGHT),
    ]
    for _ in range(WELD_STEPS):
        supervisor.step(LIVE, MEASURED)
    assert robot._ag_obj_in_hand["0"] is obj
    glitched = opened(LIVE, ("pinky", "ring"))  # pinky is not even a grasp digit
    for _ in range(RELEASE_STEPS * 5):
        supervisor.step(glitched, MEASURED)
    assert robot.release_calls == 0  # thumb+index+middle still engaged


def test_supervisor_ignores_sub_margin_and_abduction_movement():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    weld_pinch(supervisor, robot, obj)
    slightly_open = opened(LIVE, ("thumb", "ring"), amount=0.1)  # below the 0.25 rad margin
    for name in slightly_open:
        if name.endswith("_AA"):
            slightly_open[name] = 0.4  # abduction wiggles must not count as opening
    for _ in range(RELEASE_STEPS * 5):
        supervisor.step(slightly_open, MEASURED)
    assert robot.release_calls == 0


def test_supervisor_release_survives_opening_flicker():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    weld_pinch(supervisor, robot, obj)
    open_live = opened(LIVE, ("thumb", "ring"))
    for _ in range(2):
        for _ in range(RELEASE_STEPS - 1):
            supervisor.step(open_live, MEASURED)
        supervisor.step(LIVE, MEASURED)  # re-closed before the window elapsed
    assert robot.release_calls == 0
    assert robot._ag_obj_in_hand["0"] is obj


def test_supervisor_palm_grasp_releases_only_when_its_fingers_open():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    robot.contact_rows = [
        contact_row("right_hand_C_MC", obj, UP),
        contact_row("right_thumb_elastomer", obj, DOWN),
        contact_row("right_index_elastomer", obj, DOWN),
    ]
    for _ in range(WELD_STEPS):
        supervisor.step(LIVE, MEASURED)
    assert robot._ag_obj_in_hand["0"] is obj
    one_open = opened(LIVE, ("index",))
    for _ in range(RELEASE_STEPS * 3):
        supervisor.step(one_open, MEASURED)  # palm + thumb still hold
    assert robot.release_calls == 0
    both_open = opened(LIVE, ("thumb", "index"))
    for _ in range(RELEASE_STEPS):
        supervisor.step(both_open, MEASURED)  # the palm alone cannot hold
    assert robot.release_calls == 1


def test_supervisor_only_welds_when_the_thumb_is_involved():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    thumbless = [
        contact_row("right_index_elastomer", obj, LEFT),
        contact_row("right_middle_elastomer", obj, RIGHT),
        contact_row("right_hand_C_MC", obj, UP),
    ]
    robot.contact_rows = thumbless
    for _ in range(WELD_STEPS * 3):
        supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == []  # opposed and holdable, but no thumb
    permissive = make_supervisor(FakeRobot([obj]), config=replace(TEST_CONFIG, require_thumb=False))
    permissive.robot.contact_rows = thumbless
    for _ in range(WELD_STEPS):
        permissive.step(LIVE, MEASURED)
    assert permissive.robot.establish_calls == [("block", "base_link")]


def test_supervisor_treats_a_broken_weld_as_released():
    obj = block()
    robot = FakeRobot([obj])
    messages = []
    supervisor = make_supervisor(robot, announce=messages.append)
    weld_pinch(supervisor, robot, obj)
    obj.links["base_link"]._position = (0.13, 0.0, 0.0)  # 3 cm of drift: weld strained, still credible
    supervisor.step(LIVE, MEASURED)
    assert robot.release_calls == 0
    obj.links["base_link"]._position = (0.22, 0.0, 0.0)  # 12 cm: the weld is gone
    supervisor.step(LIVE, MEASURED)
    assert robot.release_calls == 1
    assert supervisor.frozen_fingers is None
    assert "broke" in messages[-1]
    assert supervisor.last_event["kind"] == "broke"
    assert supervisor.last_event["object"] == "block"
    assert supervisor.last_event["drift_m"] == pytest.approx(0.12)


def test_supervisor_adopts_an_unknown_weld_until_the_hand_opens():
    obj = block()
    robot = FakeRobot([obj])
    messages = []
    supervisor = make_supervisor(robot, announce=messages.append)
    robot._ag_obj_in_hand["0"] = obj  # e.g. restored from a saved state
    supervisor.step(LIVE, MEASURED)
    assert "adopted" in messages[0]
    assert set(supervisor.frozen_fingers) == set(SHARPA_ACTION_JOINTS)
    all_open = opened(LIVE, ("thumb", "index", "middle", "ring", "pinky"))
    for _ in range(RELEASE_STEPS):
        supervisor.step(all_open, MEASURED)
    assert robot.release_calls == 1


def test_supervisor_regrasps_after_release():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    weld_pinch(supervisor, robot, obj)
    robot.contact_rows = []
    open_live = opened(LIVE, ("thumb", "ring"))
    for _ in range(RELEASE_STEPS):
        supervisor.step(open_live, MEASURED)
    assert robot._ag_obj_in_hand["0"] is None
    weld_pinch(supervisor, robot, obj)
    assert len(robot.establish_calls) == 2


def test_supervisor_retries_a_full_window_after_a_failed_weld():
    obj = block()
    robot = FakeRobot([obj], weld_succeeds=False)
    supervisor = make_supervisor(robot)
    robot.contact_rows = pinch_rows(obj)
    for _ in range(WELD_STEPS * 2):
        supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == [("block", "base_link")] * 2


def test_supervisor_reset_clears_pending_window_freeze_and_weld():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    robot.contact_rows = pinch_rows(obj)
    for _ in range(WELD_STEPS - 1):
        supervisor.step(LIVE, MEASURED)
    supervisor.reset()
    for _ in range(WELD_STEPS - 1):
        supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == []
    supervisor.step(LIVE, MEASURED)
    assert robot._ag_obj_in_hand["0"] is obj
    assert supervisor.frozen_fingers is not None
    supervisor.reset()
    assert robot.release_calls == 1
    assert robot._ag_obj_in_hand["0"] is None
    assert supervisor.frozen_fingers is None


def test_supervisor_requires_assisted_mode_with_handling_disabled():
    with pytest.raises(ValueError, match="grasping_mode"):
        make_supervisor(FakeRobot([], grasping_mode="physical"))
    with pytest.raises(ValueError, match="disable_grasp_handling"):
        make_supervisor(FakeRobot([], disable_grasp_handling=False))
    lower = FakeRobot([])
    lower._grasping_direction = "lower"
    with pytest.raises(ValueError, match="grasping_direction"):
        make_supervisor(lower)


# --------------------------------------------------------------------------- friction closure

import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

from dex_teleop.omnigibson.assisted_grasp import (  # noqa: E402
    REPORTED_NORMAL_SIGN_INTO_OBJECT,
    antipodal_pair,
    cones_positively_span,
    evaluate_friction_closure,
    friction_window_verdict,
    inward_normal,
    squeeze_depth_by_digit,
)

MU = 0.5
MARGIN = 0.02
FRICTION_CONFIG = replace(TEST_CONFIG, criterion="friction_closure")
BLOCK_MASS = 0.1
ENOUGH_N = 3.0  # per side: mu * 6 N = 3 N >= 2 * 0.1 kg * 9.81 = 1.96 N


def reported(inward):
    """A contact normal as the simulator reports it, given the direction into the object."""
    return tuple(REPORTED_NORMAL_SIGN_INTO_OBJECT * c for c in inward)


def fc(digit, position, inward, force=ENOUGH_N, part="pad"):
    return DigitContact(digit=digit, part=part, normal=reported(inward), position=position, force_n=force)


OPPOSITE_FACES = [fc("thumb", (-0.05, 0.0, 0.0), (1.0, 0.0, 0.0)), fc("index", (0.05, 0.0, 0.0), (-1.0, 0.0, 0.0))]
ADJACENT_FACES = [fc("thumb", (-0.05, 0.0, 0.0), (1.0, 0.0, 0.0)), fc("index", (0.0, -0.05, 0.0), (0.0, 1.0, 0.0))]


def around_sphere(radius=0.035):
    contacts = []
    for digit, degrees in (("thumb", 0.0), ("index", 120.0), ("middle", 240.0)):
        angle = math.radians(degrees)
        point = (radius * math.cos(angle), radius * math.sin(angle), 0.0)
        contacts.append(fc(digit, point, (-math.cos(angle), -math.sin(angle), 0.0)))
    return contacts


def test_reported_contact_normals_point_out_of_the_object_in_the_simulator():
    """Guards REPORTED_NORMAL_SIGN_INTO_OBJECT with contacts captured from the simulator.

    A 0.49 kg block resting on the Sharpa hand (robot.get_finger_contact_data, 2026-11-13). The block's
    centre of mass lies on the object side of every contact, and the scalar forces carry its weight.
    """
    data = json.loads((Path(__file__).parent / "fixtures" / "sharpa_contact_convention.json").read_text())
    assert REPORTED_NORMAL_SIGN_INTO_OBJECT == -1.0
    dots = []
    for sample in data["samples"]:
        total = 0.0
        for row in sample["contacts"]:
            assert len(row["force"]) == 1  # signed scalar normal force
            contact = DigitContact("index", "dorsal", tuple(row["normal"]), tuple(row["position_m"]), abs(row["force"][0]))
            inward = inward_normal(contact)
            dots.append(sum(n * (c - p) for n, c, p in zip(inward, sample["object_com_m"], row["position_m"])))
            total += abs(row["force"][0]) * abs(inward[2])
        # newtons, not impulses (those would be 120 times smaller); the hand surface is not level,
        # so friction carries part of the weight
        assert 0.6 < total / (sample["object_mass_kg"] * 9.81) < 1.1
    # 40 of the 48 captured contacts are clearly on the object side, 8 are at ~0 (COM in the contact plane), none negative
    assert len(dots) >= 20 and min(dots) > -1e-6 and sum(d > 1e-4 for d in dots) / len(dots) > 0.8


def test_opposite_faces_are_antipodal():
    step = evaluate_friction_closure(OPPOSITE_FACES, mu=MU, margin=MARGIN)
    assert step.passes and step.closure == "antipodal:index"
    assert step.thumb_force_n == ENOUGH_N and step.opposing_force_n == ENOUGH_N


def test_adjacent_faces_fail_at_mu_half():
    assert antipodal_pair(ADJACENT_FACES, MU) is None
    assert not evaluate_friction_closure(ADJACENT_FACES, mu=MU, margin=MARGIN).passes
    # ... though the old criterion accepts this corner grasp
    assert has_opposition(ADJACENT_FACES, MAX_DOT)


def test_thumb_and_two_fingers_around_a_sphere_pass_without_an_antipodal_pair():
    contacts = around_sphere()
    assert antipodal_pair(contacts, MU) is None  # 30 degrees off each normal, cone half-angle 26.6
    assert cones_positively_span(contacts, MU, MARGIN)
    step = evaluate_friction_closure(contacts, mu=MU, margin=MARGIN)
    assert step.passes and step.closure == "span"


def test_three_contacts_on_one_side_do_not_span():
    same_side = [fc(d, (x, 0.0, 0.0), (0.0, 0.0, -1.0)) for d, x in (("thumb", -0.02), ("index", 0.0), ("middle", 0.02))]
    assert not cones_positively_span(same_side, MU, MARGIN)
    assert not evaluate_friction_closure(same_side, mu=MU, margin=MARGIN).passes


def test_span_needs_the_thumb():
    fingers_only = [replace(c, digit=d) for c, d in zip(around_sphere(), ("ring", "index", "middle"))]
    assert not cones_positively_span(fingers_only, MU, MARGIN)


def _verdict(contacts, squeeze, mass=BLOCK_MASS, config=FRICTION_CONFIG):
    step = evaluate_friction_closure(contacts, mu=config.friction_mu, margin=config.friction_closure_margin)
    return friction_window_verdict([step] * 3, [squeeze] * 3, mass_kg=mass, config=config)


def test_antipodal_pair_with_too_little_force_fails():
    weak = [replace(c, force_n=0.6) for c in OPPOSITE_FACES]  # above the floor, mu * 1.2 N < 1.96 N
    verdict = _verdict(weak, {"thumb": 0.1, "index": 0.1})
    assert not verdict["force_ok"] and verdict["intent_ok"] and not verdict["weld"]
    one_sided = [OPPOSITE_FACES[0], replace(OPPOSITE_FACES[1], force_n=0.2)]  # enough in total, one side under the floor
    assert not _verdict(one_sided, {"thumb": 0.1, "index": 0.1}, mass=0.01)["force_ok"]


def test_enough_force_but_shallow_squeeze_fails():
    verdict = _verdict(OPPOSITE_FACES, {"thumb": 0.1, "index": 0.01})
    assert verdict["force_ok"] and not verdict["intent_ok"] and verdict["digits_below_delta"] == ["index"]
    assert not verdict["weld"]
    assert _verdict(OPPOSITE_FACES, {"thumb": 0.1, "index": 0.1})["weld"]


def test_force_requirement_scales_with_object_weight():
    assert _verdict(OPPOSITE_FACES, {"thumb": 0.1, "index": 0.1}, mass=0.1)["weld"]
    heavy = _verdict(OPPOSITE_FACES, {"thumb": 0.1, "index": 0.1}, mass=1.0)  # needs 39.2 N in total
    assert heavy["required_total_force_n"] == pytest.approx(2.0 * 1.0 * 9.81 / 0.5) and not heavy["weld"]


def test_thumb_pad_against_the_side_of_the_index_passes():
    lateral = [OPPOSITE_FACES[0], replace(OPPOSITE_FACES[1], part="dorsal")]
    assert evaluate_friction_closure(lateral, mu=MU, margin=MARGIN).passes
    assert _verdict(lateral, {"thumb": 0.1, "index": 0.1})["weld"]


def test_no_thumb_pad_contact_fails():
    thumb_side = [replace(OPPOSITE_FACES[0], part="dorsal"), OPPOSITE_FACES[1]]
    step = evaluate_friction_closure(thumb_side, mu=MU, margin=MARGIN)
    assert step.closure == "antipodal:index" and not step.thumb_pad and not step.passes
    no_thumb = [replace(OPPOSITE_FACES[0], digit="middle"), OPPOSITE_FACES[1]]
    assert not evaluate_friction_closure(no_thumb, mu=MU, margin=MARGIN).passes


def test_squeeze_depth_uses_flexion_joints_only():
    live, measured = hand(0.9, aa=0.5), hand(0.8, aa=0.0)
    depth = squeeze_depth_by_digit(live, measured, "right")
    assert set(depth) == {"thumb", "index", "middle", "ring", "pinky"}
    assert all(value == pytest.approx(0.1) for value in depth.values())


def friction_rows(obj, *, force=ENOUGH_N, thumb_link="right_thumb_elastomer", index_inward=(-1.0, 0.0, 0.0),
                  index_position=(0.05, 0.0, 0.0)):
    path = f"{obj.prim_path}/{obj.root_link_name}"
    return [
        (hand_path(thumb_link), path, (-0.05, 0.0, 0.0), reported((1.0, 0.0, 0.0)), (-force,), 0.0),
        (hand_path("right_index_elastomer"), path, index_position, reported(index_inward), (-force,), 0.0),
    ]


def test_friction_supervisor_welds_only_after_a_full_window_of_force_and_intent():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot, config=FRICTION_CONFIG)
    robot.contact_rows = friction_rows(obj)
    for _ in range(WELD_STEPS - 1):
        supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == []
    supervisor.step(LIVE, MEASURED)
    assert robot._ag_obj_in_hand["0"] is obj
    assert supervisor.last_event["kind"] == "welded" and supervisor.last_event["digits"] == ["index", "thumb"]
    assert supervisor.frozen_fingers is not None


def test_friction_supervisor_rejects_what_the_contact_criterion_accepts():
    for rows_kwargs, live in (
        ({"index_inward": (0.0, 1.0, 0.0), "index_position": (0.0, -0.05, 0.0)}, LIVE),  # corner grasp
        ({"force": 0.6}, LIVE),  # light touch
        ({"thumb_link": "right_thumb_DP"}, LIVE),  # no thumb pad
        ({}, hand(MEASURED_CLOSED + 0.01)),  # touching without squeezing
    ):
        obj = block()
        new, old = FakeRobot([obj]), FakeRobot([obj])
        new.contact_rows = old.contact_rows = friction_rows(obj, **rows_kwargs)
        new_supervisor, old_supervisor = make_supervisor(new, config=FRICTION_CONFIG), make_supervisor(old)
        for _ in range(3 * WELD_STEPS):
            new_supervisor.step(live, MEASURED)
            old_supervisor.step(live, MEASURED)
        assert new._ag_obj_in_hand["0"] is None, rows_kwargs
        assert old._ag_obj_in_hand["0"] is obj, rows_kwargs


def test_friction_supervisor_welds_once_the_squeeze_deepens_and_reports_its_gate():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot, config=FRICTION_CONFIG)
    robot.contact_rows = friction_rows(obj)
    shallow = hand(MEASURED_CLOSED + 0.01)
    for _ in range(2 * WELD_STEPS):
        supervisor.step(shallow, MEASURED)
    gate = supervisor.diagnostic_state()["friction_gate"]
    assert gate["force_ok"] and not gate["intent_ok"] and robot._ag_obj_in_hand["0"] is None
    for _ in range(WELD_STEPS):  # the window mean must clear delta, so it takes up to one more window
        supervisor.step(LIVE, MEASURED)
    assert robot._ag_obj_in_hand["0"] is obj


def test_friction_release_rule_is_unchanged():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot, config=FRICTION_CONFIG)
    robot.contact_rows = friction_rows(obj)
    for _ in range(WELD_STEPS):
        supervisor.step(LIVE, MEASURED)
    assert supervisor.flexion_retreat(LIVE) == {"index": pytest.approx(0.0), "thumb": pytest.approx(0.0)}
    for _ in range(RELEASE_STEPS - 1):
        supervisor.step(opened(LIVE, {"thumb", "index"}), MEASURED)
    assert robot._ag_obj_in_hand["0"] is obj
    assert supervisor.flexion_retreat(opened(LIVE, {"thumb", "index"}))["thumb"] == pytest.approx(0.6)
    supervisor.step(opened(LIVE, {"thumb", "index"}), MEASURED)
    assert robot._ag_obj_in_hand["0"] is None and supervisor.last_event["kind"] == "released"


def test_criterion_is_validated():
    with pytest.raises(ValueError, match="criterion"):
        AssistedGraspConfig(criterion="magic")
    with pytest.raises(ValueError, match="friction_mu"):
        AssistedGraspConfig(friction_mu=0.0)


# --------------------------------------------------------------------------- compliant weld


class BoxedLink(FakeLink):
    """A link with a bounding box (centre +- 1 cm) for the collision-restore check."""

    def __init__(self, prim_path, position=(0.0, 0.0, 0.0), mass=0.1):
        super().__init__(prim_path, position, mass)

    @property
    def aabb(self):
        return tuple(p - 0.01 for p in self._position), tuple(p + 0.01 for p in self._position)


class CompliantRobot(FakeRobot):
    """FakeRobot with the pieces of OmniGibson's robot that a compliant weld uses."""

    def __init__(self, objects, contact_position=(0.1, 0.0, 0.0)):
        super().__init__(objects)
        self.links = {
            name: BoxedLink(hand_path(name))
            for name in ("right_hand_C_MC", "right_thumb_elastomer", "right_ring_elastomer", "panda_link7")
        }
        self.eef_links = {"0": self.links["right_hand_C_MC"]}
        self.finger_links = {"0": [self.links["right_thumb_elastomer"], self.links["right_ring_elastomer"]]}
        self._ag_obj_constraints = {"0": None}
        self._ag_obj_constraint_params = {"0": None}
        self._contact_position = contact_position
        self.grasp_calls = []

    def _find_finger_contact_position(self, arm, link_prim_path):
        return self._contact_position

    def _establish_grasp(self, target_obj, target_link_name, arm, contact_pos_world, joint_type):
        self.grasp_calls.append((target_obj.name, target_link_name, contact_pos_world, joint_type))
        self._ag_obj_in_hand[arm] = target_obj
        self._ag_obj_constraints[arm] = object()
        self._ag_obj_constraint_params[arm] = {
            "target_obj": target_obj,
            "target_link_name": target_link_name,
            "joint_type": joint_type,
        }

    def release_grasp_immediately(self, arm="default"):
        super().release_grasp_immediately(arm)
        self._ag_obj_constraints[arm] = None
        self._ag_obj_constraint_params[arm] = None


COMPLIANT_CONFIG = replace(TEST_CONFIG, weld_mode="compliant")


def make_compliant(robot, messages=None, config=COMPLIANT_CONFIG):
    supervisor = make_supervisor(robot, announce=None if messages is None else messages.append, config=config)
    supervisor.drive_calls = []
    supervisor.filter_calls = []
    supervisor._apply_joint_drives = lambda joint_prim, drives: supervisor.drive_calls.append(drives)
    supervisor._set_collision_filter = lambda pairs, *, filtered: supervisor.filter_calls.append(
        (filtered, sorted((hand.prim_path, link.prim_path) for hand, link in pairs))
    )
    return supervisor


def boxed_block(position=(0.1, 0.0, 0.0), mass=0.0054):
    obj = block(mass=mass)
    obj.links["base_link"] = BoxedLink(obj.links["base_link"].prim_path, position, mass)
    obj.mass = mass
    return obj


def test_weld_mode_is_validated():
    with pytest.raises(ValueError, match="weld_mode"):
        AssistedGraspConfig(weld_mode="glue")
    with pytest.raises(ValueError, match="compliant_stiffness_n_per_m"):
        AssistedGraspConfig(compliant_stiffness_n_per_m=0.0)
    assert AssistedGraspConfig().weld_mode == "rigid"


def test_compliant_drive_parameters_saturate_at_the_break_limits():
    drives = compliant_drive_parameters(AssistedGraspConfig(weld_mode="compliant"), mass_kg=0.5)
    stiffness, damping, limit = drives["transZ"]
    assert (stiffness, limit) == (3000.0, 100.0)
    assert damping == pytest.approx(2.0 * (3000.0 * 0.5) ** 0.5)  # critical for the held mass
    angular_stiffness, _, torque_limit = drives["rotX"]
    assert angular_stiffness == pytest.approx(20.0 * 3.141592653589793 / 180.0)  # USD drives are per degree
    assert torque_limit == 30.0
    assert set(drives) == {"transX", "transY", "transZ", "rotX", "rotY", "rotZ"}
    # a marble is over-damped at the lower clip; no limits means no drive force cap
    assert compliant_drive_parameters(AssistedGraspConfig(), 0.0054)["transX"][1] == 10.0
    unlimited = compliant_drive_parameters(AssistedGraspConfig(weld_break_force=None, weld_break_torque=None), 0.1)
    assert unlimited["transX"][2] is None and unlimited["rotZ"][2] is None


def test_boxes_overlap_uses_a_small_margin():
    assert boxes_overlap((0, 0, 0), (1, 1, 1), (1.001, 0, 0), (2, 1, 1))
    assert not boxes_overlap((0, 0, 0), (1, 1, 1), (1.01, 0, 0), (2, 1, 1))
    assert not boxes_overlap((0, 0, 0), (1, 1, 1), (0, 0, 1.5), (1, 1, 2))


def test_compliant_weld_is_a_spring_joint_with_hand_collisions_filtered():
    obj = boxed_block()
    robot = CompliantRobot([obj])
    messages = []
    supervisor = make_compliant(robot, messages)
    weld_pinch(supervisor, robot, obj)

    # the supervisor made the joint itself, as a generic (free) joint at the finger contact
    assert robot.establish_calls == []
    assert robot.grasp_calls == [("block", "base_link", (0.1, 0.0, 0.0), "Joint")]
    # ... and recorded it the way OmniGibson serializes it, so reloading a state leaves the joint alone
    assert robot._ag_obj_constraint_params["0"]["joint_type"] == "FixedJoint"
    assert supervisor.last_event["weld_mode"] == "compliant"
    # spring drives on all six axes, capped at the break force / torque
    (drives,) = supervisor.drive_calls
    assert drives["transX"] == (3000.0, 10.0, 100.0) and drives["rotY"][2] == 30.0
    # every hand link (not the arm) stops colliding with the held object
    (filtered, pairs) = supervisor.filter_calls[0]
    assert filtered is True
    assert [hand for hand, _ in pairs] == sorted(
        hand_path(name) for name in ("right_hand_C_MC", "right_thumb_elastomer", "right_ring_elastomer")
    )
    assert {link for _, link in pairs} == {obj.links["base_link"].prim_path}
    # nothing resists a squeeze bias once collisions are off: the digits hold their measured pose
    frozen = supervisor.frozen_fingers
    flexion = [name for name in frozen if digit_of(name) in ("thumb", "ring") and not name.endswith("_AA")]
    assert flexion and all(frozen[name] == pytest.approx(MEASURED_CLOSED) for name in flexion)
    state = supervisor.diagnostic_state()
    assert state["weld_mode"] == "compliant" and state["hand_collisions_filtered"] is True


def test_compliant_mode_welds_objects_with_meta_links_rigidly_and_without_break_limits():
    """A cup with a fillable meta link. Its hand collisions cannot be filtered (editing its filtered pairs broke the
    physics views, pour water, 2026-11-16), and the spring weld next to the water ended the GPU physics (CUDA error
    700, 2026-11-18): it gets OmniGibson's fixed joint, with no break limits, and the rigid weld's squeeze bias."""
    obj = boxed_block()
    obj.links["meta__base_link_fillable_0_0_link"] = BoxedLink(obj.links["base_link"].prim_path + "_meta", (0.1, 0.0, 0.0), 0.0)
    robot = CompliantRobot([obj])
    messages = []
    supervisor = make_compliant(robot, messages)
    authored = []
    supervisor._weld_joint_prim = lambda: authored.append("joint prim read") or object()
    weld_pinch(supervisor, robot, obj)
    assert robot.establish_calls == [("block", "base_link")] and robot.grasp_calls == []
    assert supervisor.drive_calls == [] and supervisor.filter_calls == []
    # the joint was looked at, and neither spring drives nor break limits were authored on it
    assert authored == ["joint prim read"] and supervisor._unbreakable_weld is True
    assert any("welds block rigidly, without break limits" in message for message in messages)
    state = supervisor.diagnostic_state()
    assert state["hand_collisions_filtered"] is False and state["weld_mode"] != "compliant"
    frozen = supervisor.frozen_fingers
    name = next(n for n in frozen if digit_of(n) == "thumb" and not n.endswith("_AA"))
    assert frozen[name] == pytest.approx(MEASURED_CLOSED + COMPLIANT_CONFIG.frozen_squeeze_bias_rad)
    # an object without meta links still gets the spring weld from the same supervisor afterwards
    supervisor.reset()
    assert supervisor._unbreakable_weld is False


def test_rigid_weld_keeps_the_squeeze_bias_and_filters_nothing():
    obj = boxed_block()
    robot = CompliantRobot([obj])
    supervisor = make_compliant(robot, config=TEST_CONFIG)
    weld_pinch(supervisor, robot, obj)
    assert robot.establish_calls == [("block", "base_link")] and robot.grasp_calls == []
    assert supervisor.drive_calls == [] and supervisor.filter_calls == []
    frozen = supervisor.frozen_fingers
    name = next(n for n in frozen if digit_of(n) == "thumb" and not n.endswith("_AA"))
    assert frozen[name] == pytest.approx(MEASURED_CLOSED + TEST_CONFIG.frozen_squeeze_bias_rad)
    assert supervisor.diagnostic_state()["hand_collisions_filtered"] is False


def test_compliant_release_restores_collisions_once_the_hand_is_clear():
    obj = boxed_block(position=(0.0, 0.0, 0.0))  # inside the hand links' boxes
    robot = CompliantRobot([obj])
    supervisor = make_compliant(robot)
    weld_pinch(supervisor, robot, obj)
    robot.contact_rows = []
    for _ in range(RELEASE_STEPS):
        supervisor.step(hand(0.0), MEASURED)
    assert robot._ag_obj_in_hand["0"] is None and supervisor.last_event["kind"] == "released"
    # the joint is gone but the object still overlaps the hand: collisions stay filtered
    assert len(supervisor.filter_calls) == 1
    supervisor.step(hand(0.0), MEASURED)
    assert len(supervisor.filter_calls) == 1 and supervisor.diagnostic_state()["hand_collisions_filtered"]
    # the object leaves the hand: collisions come back on the next step
    obj.links["base_link"]._position = (0.5, 0.0, 0.0)
    supervisor.step(hand(0.0), MEASURED)
    assert supervisor.filter_calls[-1][0] is False and len(supervisor.filter_calls) == 2
    assert supervisor.filter_calls[-1][1] == supervisor.filter_calls[0][1]
    assert not supervisor.diagnostic_state()["hand_collisions_filtered"]


def test_compliant_release_restores_collisions_after_the_timeout():
    obj = boxed_block(position=(0.0, 0.0, 0.0))
    robot = CompliantRobot([obj])
    messages = []
    supervisor = make_compliant(
        robot, messages, config=replace(COMPLIANT_CONFIG, compliant_collision_restore_timeout_s=0.25)
    )
    weld_pinch(supervisor, robot, obj)
    robot.contact_rows = []
    for _ in range(RELEASE_STEPS):
        supervisor.step(hand(0.0), MEASURED)
    for _ in range(2):
        supervisor.step(hand(0.0), MEASURED)
    assert len(supervisor.filter_calls) == 1  # 0.2 s after the release: still waiting
    supervisor.step(hand(0.0), MEASURED)
    assert supervisor.filter_calls[-1][0] is False
    assert messages[-1] == "assisted grasp restored hand collisions with block while the hand still overlapped it"


def test_compliant_reset_restores_collisions_immediately():
    obj = boxed_block(position=(0.0, 0.0, 0.0))
    robot = CompliantRobot([obj])
    supervisor = make_compliant(robot)
    weld_pinch(supervisor, robot, obj)
    supervisor.reset()
    assert robot._ag_obj_in_hand["0"] is None
    assert [filtered for filtered, _ in supervisor.filter_calls] == [True, False]


def test_compliant_weld_needs_a_finger_contact_position():
    obj = boxed_block()
    robot = CompliantRobot([obj], contact_position=None)
    supervisor = make_compliant(robot)
    robot.contact_rows = pinch_rows(obj)
    for _ in range(WELD_STEPS + 1):
        supervisor.step(LIVE, MEASURED)
    assert robot._ag_obj_in_hand["0"] is None and robot.grasp_calls == [] and supervisor.filter_calls == []


def test_the_start_message_names_the_weld():
    assert "RIGID" in describe_weld(AssistedGraspConfig()) and "--assisted-grasp-weld compliant" in describe_weld(AssistedGraspConfig())
    compliant = describe_weld(AssistedGraspConfig(weld_mode="compliant", weld_break_force=None))
    assert "COMPLIANT" in compliant and "3000 N/m" in compliant and "no force limit / 30 Nm" in compliant


def test_mean_release_rule_compares_the_mean_opening_with_the_margin():
    """'mean' releases on the grasping digits' mean opening (EMG fingers open little, thumb and pinky least);
    'digits' releases once fewer than two digits stay near their weld-time flexion."""

    def released(rule, ring_opening):
        obj = boxed_block()
        robot = CompliantRobot([obj])
        supervisor = make_compliant(robot, config=replace(COMPLIANT_CONFIG, release_rule=rule))
        weld_pinch(supervisor, robot, obj)  # ring + thumb
        robot.contact_rows = []  # no re-weld after a release
        live = {name: value - (ring_opening if "_ring_" in name else 0.0) for name, value in supervisor._open_baseline.items()}
        for _ in range(RELEASE_STEPS + 2):
            supervisor.step(live, live)
        return robot._ag_obj_in_hand["0"] is None

    # ring opens 0.4 rad, thumb stays: mean 0.2 < 0.25 keeps holding; the digit rule releases (only the thumb engaged)
    assert released("mean", 0.4) is False and released("digits", 0.4) is True
    # ring opens 0.6 rad: mean 0.3 >= 0.25 releases
    assert released("mean", 0.6) is True


def test_release_now_drops_the_object_and_says_so():
    obj = boxed_block()
    robot = CompliantRobot([obj])
    messages = []
    supervisor = make_compliant(robot, messages)
    assert supervisor.release_now() is False
    weld_pinch(supervisor, robot, obj)
    assert supervisor.release_now() is True
    assert robot._ag_obj_in_hand["0"] is None and supervisor.frozen_fingers is None
    assert supervisor.last_event == {"kind": "released", "object": "block", "manual": True}
    assert any("release key" in m for m in messages)


def test_release_rule_is_validated():
    with pytest.raises(ValueError, match="release_rule"):
        AssistedGraspConfig(release_rule="vote")
    with pytest.raises(ValueError, match="release_open_margin_rad"):
        AssistedGraspConfig(release_open_margin_rad=0.0)
