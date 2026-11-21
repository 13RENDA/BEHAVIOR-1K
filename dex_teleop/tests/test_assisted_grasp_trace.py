"""Tests for simulator-independent assisted-grasp HDF5 diagnostics."""

import h5py
import pytest

from dex_teleop.omnigibson.assisted_grasp_trace import (
    AssistedGraspTraceCollector,
    _contact_aggregate,
    _contact_force,
    read_assisted_grasp_config,
    read_assisted_grasp_episode,
    summarize_assisted_grasp_episode,
    write_assisted_grasp_episodes,
)


def _recording(path, samples=3):
    with h5py.File(path, "w") as recording:
        data = recording.create_group("data")
        data.create_group("demo_0").attrs["num_samples"] = samples


def test_assisted_grasp_trace_round_trips_and_classifies_physx_break(tmp_path):
    path = tmp_path / "recording.hdf5"
    _recording(path)
    steps = [
        {
            "episode_time_s": 0.0,
            "supervisor_event": {"kind": "welded", "object": "ball"},
            "joint_break_events": [],
            "held_contacts": {"aggregate": {"max_force_norm_n": None, "min_separation_m": None}},
        },
        {
            "episode_time_s": 1.0 / 30.0,
            "supervisor_event": None,
            "joint_break_events": [{"joint_path": "/robot/hand/ag_constraint", "simulation_time_s": 1.2}],
            "held_contacts": {"aggregate": {"max_force_norm_n": 42.0, "min_separation_m": -0.003}},
            "external_contacts": [{"held_link": "/ball/base", "other": "/tin/base"}],
        },
        {
            "episode_time_s": 2.0 / 30.0,
            "supervisor_event": {"kind": "broke", "object": "ball", "drift_m": 0.08},
            "joint_break_events": [],
            "held_contacts": {"aggregate": {"max_force_norm_n": 15.0, "min_separation_m": -0.001}},
        },
    ]

    write_assisted_grasp_episodes(path, [steps], {"weld_break_force": 100.0})

    assert read_assisted_grasp_config(path) == {"weld_break_force": 100.0}
    assert read_assisted_grasp_episode(path, 0, expected_steps=3) == steps
    failure = summarize_assisted_grasp_episode(steps)["breaks"][0]
    assert failure["classification"] == "physx_break_limit"
    assert failure["max_finger_contact_force_n"] == 42.0
    assert failure["min_contact_separation_m"] == -0.003
    assert failure["external_contacts"] == ["/tin/base"]


def test_assisted_grasp_trace_distinguishes_drift_with_external_contact():
    steps = [
        {
            "supervisor_event": {"kind": "welded", "object": "ball"},
            "joint_break_events": [],
            "external_contacts": [],
        },
        {
            "supervisor_event": {"kind": "broke", "object": "ball", "drift_m": 0.06},
            "joint_break_events": [],
            "external_contacts": [{"held_link": "/ball/base", "other": "/box/base"}],
        },
    ]

    failure = summarize_assisted_grasp_episode(steps)["breaks"][0]

    assert failure["classification"] == "drift_with_external_contact"
    assert failure["external_contacts"] == ["/box/base"]


def test_assisted_grasp_trace_rejects_step_misalignment(tmp_path):
    path = tmp_path / "recording.hdf5"
    _recording(path, samples=1)

    with pytest.raises(RuntimeError, match="has 0 steps; trajectory has 1"):
        write_assisted_grasp_episodes(path, [[]], {})


def test_scalar_isaac_contact_force_is_converted_along_normal():
    reported_force, force_vector = _contact_force((0.0, 0.0, -1.0), [12.5])

    assert reported_force == [12.5]
    assert force_vector == [0.0, 0.0, -12.5]
    aggregate = _contact_aggregate(
        [{"force_n": force_vector, "force_norm_n": 12.5, "separation_m": -0.002}]
    )
    assert aggregate["net_force_n"] == [0.0, 0.0, -12.5]
    assert aggregate["max_force_norm_n"] == 12.5


def test_collection_error_becomes_aligned_diagnostic_row():
    class BrokenSupervisor:
        arm = "0"
        control_dt = 1.0 / 30.0
        last_event = {"kind": "welded", "object": "ball"}

        @staticmethod
        def diagnostic_state():
            raise RuntimeError("unsupported simulator value")

    collector = object.__new__(AssistedGraspTraceCollector)
    collector._supervisor = BrokenSupervisor()
    collector._robot = object()
    collector._joint_break_events = [{"joint_path": "/robot/hand/ag_constraint"}]

    step = collector.capture_before_supervisor(
        episode_step=4,
        sim_time_before_s=1.0,
        sim_time_after_s=1.1,
        live_fingers={"finger": 0.5},
        measured_fingers={"finger": 0.4},
        frozen_fingers=None,
    )
    step = collector.finish_after_supervisor(step)

    assert step["episode_step"] == 4
    assert step["joint_break_events"] == [{"joint_path": "/robot/hand/ag_constraint"}]
    assert step["supervisor_event"] == {"kind": "welded", "object": "ball"}
    assert [error["phase"] for error in step["diagnostic_errors"]] == [
        "capture_before_supervisor",
        "finish_after_supervisor",
    ]


# --------------------------------------------------------------------------- offline criterion comparison

from dataclasses import asdict, replace  # noqa: E402

from dex_teleop.omnigibson.assisted_grasp import AssistedGraspConfig  # noqa: E402
from dex_teleop.omnigibson.assisted_grasp_trace import (  # noqa: E402
    ASSISTED_GRASP_SCHEMA_VERSION,
    compare_criteria_episode,
)

import test_assisted_grasp as ag  # noqa: E402  (shared fakes and contact builders)

CONFIG = asdict(replace(ag.TEST_CONFIG, criterion="contact_opposition"))


def _trace_step(index, contacts, squeeze=0.1, retreat=None, event=None, digits=("index", "thumb")):
    return {
        "episode_time_s": index * ag.DT,
        "hand_contacts": [
            {
                "object": "block",
                "link_name": "base_link",
                "mass_kg": 0.1,
                "graspable": True,
                "contacts": [
                    {"digit": c.digit, "part": c.part, "position_m": list(c.position), "normal": list(c.normal), "force_n": c.force_n}
                    for c in contacts
                ],
            }
        ]
        if contacts
        else [],
        "squeeze_depth_rad": {digit: squeeze for digit in ("thumb", "index", "middle", "ring", "pinky")},
        "flexion_retreat_rad": retreat,
        "supervisor_before": {"grasp_digits": list(digits)},
        "supervisor_event": event,
    }


def test_comparison_splits_contact_only_steps_by_cause_and_times_first_weld():
    light = [replace(c, force_n=0.6) for c in ag.OPPOSITE_FACES]
    thumb_side = [replace(ag.OPPOSITE_FACES[0], part="dorsal"), ag.OPPOSITE_FACES[1]]
    phases = (
        [[]] * 2
        + [ag.ADJACENT_FACES] * 4  # corner: closure fails
        + [thumb_side] * 4  # no thumb pad
        + [light] * 4  # too little force
        + [ag.OPPOSITE_FACES] * 4  # everything holds
    )
    steps = [_trace_step(i, contacts) for i, contacts in enumerate(phases)]
    for i in range(10, 18):  # from the light-touch phase on, the targets sit behind the measured joints
        steps[i]["squeeze_depth_rad"] = {digit: -0.1 for digit in steps[i]["squeeze_depth_rad"]}
    result = compare_criteria_episode(steps, CONFIG)
    assert result["evaluable"] and result["window_steps"] == ag.WELD_STEPS
    assert result["first_contact_step"] == 2
    # contact_opposition: third consecutive touching step
    assert result["contact_to_first_weld_s"]["contact_opposition"] == pytest.approx(2 * ag.DT)
    assert result["contact_to_first_weld_s"]["friction_closure"] is None
    causes = result["contact_opposition_only_steps_by_cause"]
    assert causes["closure"] >= 2 and causes["thumb_part"] >= 2 and causes["force"] >= 2 and causes["intent"] >= 2
    assert sum(causes.values()) == result["steps_holding"]["contact_opposition"]
    assert result["friction_closure_only_steps"] == 0

    steps += [_trace_step(18 + i, ag.OPPOSITE_FACES) for i in range(4)]
    result = compare_criteria_episode(steps, CONFIG)
    # the squeeze returns at step 18: the 3-step window mean (-0.1, 0.1, 0.1) clears delta one step later
    assert result["contact_to_first_weld_s"]["friction_closure"] == pytest.approx((19 - 2) * ag.DT)
    assert result["steps_holding"]["both"] == 3
    # parameters can be re-tried offline: with almost no force requirement nothing is lost to "force"
    looser = compare_criteria_episode(steps, {**CONFIG, "friction_weight_factor": 0.01, "friction_force_floor_n": 0.0})
    assert looser["contact_opposition_only_steps_by_cause"]["force"] == 0


def test_comparison_counts_virtual_releases_per_margin():
    held = [_trace_step(i, ag.OPPOSITE_FACES, retreat={"thumb": 0.0, "index": 0.0}) for i in range(3)]
    held[0]["supervisor_event"] = {"kind": "welded", "object": "block"}
    # the operator relaxes by 0.2 rad for 5 steps, re-squeezes, then opens fully and the weld is released
    held += [_trace_step(3 + i, ag.OPPOSITE_FACES, retreat={"thumb": 0.2, "index": 0.2}) for i in range(5)]
    held += [_trace_step(8 + i, ag.OPPOSITE_FACES, retreat={"thumb": 0.0, "index": 0.0}) for i in range(2)]
    held += [_trace_step(10 + i, ag.OPPOSITE_FACES, retreat={"thumb": 0.6, "index": 0.6}) for i in range(4)]
    held[-1]["supervisor_event"] = {"kind": "released", "object": "block"}
    held += [_trace_step(14, [])]
    result = compare_criteria_episode(held, CONFIG)
    assert result["events"] == {"welded": 1, "released": 1, "broke": 0}
    virtual = result["virtual_releases_by_margin_rad"]
    assert virtual["0.25"]["drops"] == 1  # the configured margin reproduces the recorded release
    assert virtual["0.15"]["drops"] == 2 and virtual["0.10"]["drops"] == 2
    assert virtual["0.10"]["weld_segments_affected"] == 1


def test_schema_one_traces_are_reported_as_not_evaluable():
    result = compare_criteria_episode([{"episode_time_s": 0.0, "supervisor_event": None}], CONFIG)
    assert not result["evaluable"] and "schema 1" in result["reason"]


@pytest.mark.parametrize("criterion", ["contact_opposition", "friction_closure"])
def test_replaying_a_collected_trace_reproduces_the_recorded_weld(criterion):
    obj = ag.block()
    robot = ag.FakeRobot([obj])
    config = replace(ag.TEST_CONFIG, criterion=criterion)
    supervisor = ag.make_supervisor(robot, config=config)
    collector = object.__new__(AssistedGraspTraceCollector)
    collector._supervisor, collector._robot, collector._joint_break_events = supervisor, robot, []
    steps = []
    for index in range(10):
        robot.contact_rows = [] if index < 2 else ag.friction_rows(obj)
        live = ag.LIVE if index < 8 else ag.opened(ag.LIVE, {"thumb", "index"})
        step = collector.capture_before_supervisor(
            episode_step=index, sim_time_before_s=0.0, sim_time_after_s=0.0,
            live_fingers=live, measured_fingers=ag.MEASURED, frozen_fingers=supervisor.frozen_fingers,
        )
        step["episode_time_s"] = index * ag.DT
        supervisor.step(live, ag.MEASURED)
        steps.append(collector.finish_after_supervisor(step))
    assert steps[0]["schema_version"] == ASSISTED_GRASP_SCHEMA_VERSION == 2
    welded_at = next(i for i, s in enumerate(steps) if s["supervisor_event"] and s["supervisor_event"]["kind"] == "welded")
    assert welded_at == 2 + ag.WELD_STEPS - 1
    contact = steps[2]["hand_contacts"][0]
    assert contact["mass_kg"] == pytest.approx(0.1) and {c["part"] for c in contact["contacts"]} == {"pad"}
    assert steps[2]["squeeze_depth_rad"]["thumb"] == pytest.approx(0.1)
    assert steps[welded_at]["flexion_retreat_rad"] is None and steps[welded_at + 1]["flexion_retreat_rad"] is not None
    assert steps[9]["flexion_retreat_rad"]["thumb"] == pytest.approx(0.6)

    result = compare_criteria_episode(steps, asdict(config))
    assert result["recorded_first_weld_step"] == welded_at
    assert result["contact_to_first_weld_s"][criterion] == pytest.approx(result["contact_to_first_weld_s"]["recorded"])
    assert result["recorded_criterion"] == criterion
