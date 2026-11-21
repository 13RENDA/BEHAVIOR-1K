"""Action-aligned diagnostics for teleoperated assisted grasps."""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
from typing import Mapping

import h5py


LOGGER = logging.getLogger(__name__)

ASSISTED_GRASP_SCHEMA_VERSION = 2
ASSISTED_GRASP_GROUP = "assisted_grasp"
ASSISTED_GRASP_STEPS_DATASET = "steps"


def assisted_grasp_config_dict(config) -> dict:
    """Return the dataclass configuration in JSON-safe form."""

    return asdict(config)


def _episode_groups(group) -> list[tuple[int, h5py.Group]]:
    episodes = []
    for name, episode in group.items():
        if not name.startswith("demo_") or not isinstance(episode, h5py.Group):
            continue
        try:
            episode_id = int(name.removeprefix("demo_"))
        except ValueError:
            continue
        episodes.append((episode_id, episode))
    return sorted(episodes)


def write_assisted_grasp_episodes(
    input_path: str | Path,
    episodes: list[list[dict]],
    config: Mapping[str, object],
) -> None:
    """Append action-aligned assisted-grasp JSON to a completed recording."""

    input_path = Path(input_path)
    with h5py.File(input_path, "r+") as recording:
        trajectory_episodes = _episode_groups(recording["data"])
        if len(episodes) != len(trajectory_episodes):
            raise RuntimeError(
                f"Assisted-grasp episode count {len(episodes)} does not match trajectory count "
                f"{len(trajectory_episodes)}"
            )

        if ASSISTED_GRASP_GROUP in recording:
            del recording[ASSISTED_GRASP_GROUP]
        diagnostics = recording.create_group(ASSISTED_GRASP_GROUP)
        diagnostics.attrs["schema_version"] = ASSISTED_GRASP_SCHEMA_VERSION
        diagnostics.attrs["config_json"] = json.dumps(dict(config), separators=(",", ":"), sort_keys=True)
        string_dtype = h5py.string_dtype(encoding="utf-8")

        for trace_steps, (episode_id, trajectory) in zip(episodes, trajectory_episodes):
            expected_steps = int(trajectory.attrs["num_samples"])
            if len(trace_steps) != expected_steps:
                raise RuntimeError(
                    f"Assisted-grasp demo_{episode_id} has {len(trace_steps)} steps; "
                    f"trajectory has {expected_steps}"
                )
            episode = diagnostics.create_group(f"demo_{episode_id}")
            encoded = [json.dumps(step, separators=(",", ":"), ensure_ascii=False) for step in trace_steps]
            episode.create_dataset(ASSISTED_GRASP_STEPS_DATASET, data=encoded, dtype=string_dtype)
        recording.flush()


def read_assisted_grasp_config(input_path: str | Path) -> dict:
    """Load the assisted-grasp configuration without launching OmniGibson."""

    with h5py.File(Path(input_path), "r") as recording:
        try:
            encoded = recording[ASSISTED_GRASP_GROUP].attrs["config_json"]
        except KeyError as error:
            raise ValueError(
                "This recording has no assisted-grasp diagnostic trace. Record it with "
                "--assisted-grasp --assisted-grasp-debug."
            ) from error
    if isinstance(encoded, bytes):
        encoded = encoded.decode("utf-8")
    return json.loads(encoded)


def read_assisted_grasp_episode(
    input_path: str | Path,
    episode_id: int,
    expected_steps: int | None = None,
) -> list[dict]:
    """Load one assisted-grasp trace without launching OmniGibson."""

    with h5py.File(Path(input_path), "r") as recording:
        try:
            dataset = recording[ASSISTED_GRASP_GROUP][f"demo_{episode_id}"][ASSISTED_GRASP_STEPS_DATASET]
        except KeyError as error:
            raise ValueError(
                "This recording has no assisted-grasp diagnostic trace for "
                f"demo_{episode_id}. Record it with --assisted-grasp --assisted-grasp-debug."
            ) from error
        steps = []
        for encoded in dataset:
            if isinstance(encoded, bytes):
                encoded = encoded.decode("utf-8")
            steps.append(json.loads(encoded))

    if expected_steps is not None and len(steps) != expected_steps:
        raise ValueError(
            f"Assisted-grasp trace has {len(steps)} steps, but demo_{episode_id} has "
            f"{expected_steps} trajectory steps"
        )
    return steps


def summarize_assisted_grasp_episode(steps: list[dict]) -> dict:
    """Classify weld losses and summarize the evidence accumulated while held."""

    events = []
    breaks = []
    weld_step = None
    joint_break_events = []
    for index, step in enumerate(steps):
        supervisor_event = step.get("supervisor_event")
        if supervisor_event is not None:
            event = {"step": index, **supervisor_event}
            events.append(event)
            if supervisor_event.get("kind") in {"welded", "adopted"}:
                weld_step = index
                joint_break_events = []

        joint_break_events.extend(step.get("joint_break_events", []))
        if supervisor_event is None or supervisor_event.get("kind") != "broke":
            continue

        start = index if weld_step is None else weld_step
        held_steps = steps[start : index + 1]
        forces = [
            held_step.get("held_contacts", {}).get("aggregate", {}).get("max_force_norm_n")
            for held_step in held_steps
        ]
        forces = [float(value) for value in forces if value is not None]
        separations = [
            held_step.get("held_contacts", {}).get("aggregate", {}).get("min_separation_m")
            for held_step in held_steps
        ]
        separations = [float(value) for value in separations if value is not None]
        external_contacts = sorted(
            {
                pair["other"]
                for held_step in held_steps
                for pair in held_step.get("external_contacts", [])
                if "other" in pair
            }
        )
        matching_breaks = [
            event for event in joint_break_events if str(event.get("joint_path", "")).endswith("/ag_constraint")
        ]
        if matching_breaks:
            classification = "physx_break_limit"
        elif external_contacts:
            classification = "drift_with_external_contact"
        else:
            classification = "drift_without_joint_break_event"
        breaks.append(
            {
                "step": index,
                "episode_time_s": step.get("episode_time_s"),
                "object": supervisor_event.get("object"),
                "drift_m": supervisor_event.get("drift_m"),
                "classification": classification,
                "joint_break_events": matching_breaks,
                "max_finger_contact_force_n": max(forces) if forces else None,
                "min_contact_separation_m": min(separations) if separations else None,
                "external_contacts": external_contacts,
            }
        )
        joint_break_events = []

    return {"steps": len(steps), "events": events, "breaks": breaks}


def _vector(value) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(component) for component in value]


def _norm(vector) -> float:
    return math.sqrt(sum(float(component) ** 2 for component in vector))


def _contact_force(normal, reported_force) -> tuple[list[float], list[float]]:
    """Convert Isaac's scalar normal force or vector force into a world-frame vector."""

    normal = _vector(normal)
    reported_force = _vector(reported_force)
    if len(reported_force) == 3:
        return reported_force, reported_force
    if len(reported_force) == 1 and len(normal) == 3:
        normal_norm = _norm(normal)
        if normal_norm == 0.0:
            return reported_force, [0.0, 0.0, 0.0]
        magnitude = reported_force[0]
        return reported_force, [magnitude * component / normal_norm for component in normal]
    raise ValueError(
        f"Expected contact force with 1 or 3 components, got {len(reported_force)} "
        f"(normal has {len(normal)})"
    )


def _body_state(body) -> dict:
    position, orientation = body.get_position_orientation()
    return {
        "position_m": _vector(position),
        "orientation_xyzw": _vector(orientation),
        "linear_velocity_m_s": _vector(body.get_linear_velocity()),
        "angular_velocity_rad_s": _vector(body.get_angular_velocity()),
    }


class AssistedGraspTraceCollector:
    """Collect simulator-only evidence immediately after every recorded action."""

    def __init__(self, robot, supervisor) -> None:
        import omnigibson as og
        import omnigibson.lazy as lazy
        from omnigibson.utils.usd_utils import RigidContactAPI

        self._robot = robot
        self._supervisor = supervisor
        self._sim = og.sim
        self._lazy = lazy
        self._rigid_contact_api = RigidContactAPI
        self._joint_break_events: list[dict] = []
        stream = self._sim._physics_context._physx_interface.get_simulation_event_stream_v2()
        self._event_subscription = stream.create_subscription_to_pop(self._on_simulation_event)

    def close(self) -> None:
        self._event_subscription = None

    def _on_simulation_event(self, event) -> None:
        try:
            break_type = int(self._lazy.omni.physx.bindings._physx.SimulationEvent.JOINT_BREAK)
            if event.type != break_type:
                return
            encoded_path = event.payload["jointPath"]
            joint_path = str(
                self._lazy.pxr.PhysicsSchemaTools.decodeSdfPath(encoded_path[0], encoded_path[1])
            )
            if joint_path.endswith("/ag_constraint"):
                self._joint_break_events.append(
                    {"joint_path": joint_path, "simulation_time_s": float(self._sim.current_time)}
                )
        except Exception:
            LOGGER.exception("Could not decode a PhysX joint-break event for assisted-grasp diagnostics")

    def capture_before_supervisor(
        self,
        *,
        episode_step: int,
        sim_time_before_s: float,
        sim_time_after_s: float,
        live_fingers: Mapping[str, float],
        measured_fingers: Mapping[str, float],
        frozen_fingers: Mapping[str, float] | None,
    ) -> dict:
        """Capture the still-welded state before the supervisor can release it."""

        joint_break_events = self._joint_break_events
        self._joint_break_events = []
        step = {
            "schema_version": ASSISTED_GRASP_SCHEMA_VERSION,
            "episode_step": episode_step,
            "episode_time_s": episode_step * self._supervisor.control_dt,
            "simulation_time_before_s": sim_time_before_s,
            "simulation_time_after_s": sim_time_after_s,
            "supervisor_before": None,
            "joint": None,
            "joint_break_events": joint_break_events,
            "eef": None,
            "held_object": None,
            "held_contacts": {"points": [], "aggregate": _contact_aggregate([])},
            "external_contacts": [],
            # Schema 2: everything both weld criteria read, on every step (welded or not), so either
            # can be re-evaluated offline with other parameters: every hand contact per object link
            # (digit, part, world position, reported normal, normal-force magnitude in N), the mass the
            # grasp would carry, each digit's squeeze depth (live target - measured, flexion joints), and
            # while welded each grasp digit's commanded flexion retreat (what the release rule compares
            # with release_open_margin_rad).
            "hand_contacts": [],
            "squeeze_depth_rad": {},
            "flexion_retreat_rad": None,
            "finger_commands_rad": {
                "live": _float_mapping(live_fingers),
                "measured": _float_mapping(measured_fingers),
                "frozen": None if frozen_fingers is None else _float_mapping(frozen_fingers),
            },
            "diagnostic_errors": [],
        }
        try:
            step["supervisor_before"] = self._supervisor.diagnostic_state()
            step["hand_contacts"] = self._supervisor.contact_snapshot()
            step["squeeze_depth_rad"] = self._supervisor.squeeze_depth(live_fingers, measured_fingers)
            if self._robot._ag_obj_in_hand[self._supervisor.arm] is not None:
                step["flexion_retreat_rad"] = self._supervisor.flexion_retreat(live_fingers)
            step["joint"] = self._joint_state()
            step["eef"] = _body_state(self._robot.eef_links[self._supervisor.arm])
            held = self._robot._ag_obj_in_hand[self._supervisor.arm]
            if held is not None:
                step["held_object"] = self._held_object_state(held)
                contacts = self._held_contacts(held)
                step["held_contacts"] = {"points": contacts, "aggregate": _contact_aggregate(contacts)}
                step["external_contacts"] = self._external_contacts(held)
        except Exception as error:
            _record_diagnostic_error(step, "capture_before_supervisor", error)
            LOGGER.exception("Could not collect part of an assisted-grasp diagnostic step; recording continues")
        return step

    def finish_after_supervisor(self, step: dict) -> dict:
        try:
            step["supervisor_event"] = self._supervisor.last_event
            step["supervisor_after"] = self._supervisor.diagnostic_state()
        except Exception as error:
            step.setdefault("supervisor_event", None)
            step.setdefault("supervisor_after", None)
            _record_diagnostic_error(step, "finish_after_supervisor", error)
            LOGGER.exception("Could not finish an assisted-grasp diagnostic step; recording continues")
        return step

    def _joint_state(self) -> dict:
        joint = self._robot._ag_obj_constraints[self._supervisor.arm]
        if joint is None:
            return {"present": False}
        path = joint.GetPath().pathString
        state = {"present": bool(joint.IsValid()), "path": path}
        for field, attribute_name in (
            ("enabled", "physics:jointEnabled"),
            ("break_force_n", "physics:breakForce"),
            ("break_torque_nm", "physics:breakTorque"),
        ):
            value = joint.GetAttribute(attribute_name).Get()
            if isinstance(value, bool) or value is None:
                state[field] = value
            else:
                parsed = float(value)
                state[field] = parsed if math.isfinite(parsed) else str(parsed)
        return state

    def _held_object_state(self, held) -> dict:
        params = self._robot._ag_obj_constraint_params[self._supervisor.arm]
        target_link_name = held.root_link_name if params is None else params["target_link_name"]
        target_link = held.links[target_link_name]
        return {
            "name": held.name,
            "prim_path": held.prim_path,
            "target_link_name": target_link_name,
            "target_link": _body_state(target_link),
        }

    def _held_contacts(self, held) -> list[dict]:
        contacts = []
        held_prefix = f"{held.prim_path}/"
        for hand_path, other_path, position, normal, force, separation in self._robot.get_finger_contact_data(
            self._supervisor.arm
        ):
            if not (other_path == held.prim_path or other_path.startswith(held_prefix)):
                continue
            classified = self._supervisor.classify_hand_link(hand_path.rsplit("/", 1)[-1])
            normal_vector = _vector(normal)
            reported_force, force_vector = _contact_force(normal_vector, force)
            contact = {
                "hand_link": hand_path,
                "object_link": other_path,
                "position_m": _vector(position),
                "normal": normal_vector,
                "reported_force_n": reported_force,
                "force_n": force_vector,
                "force_norm_n": _norm(force_vector),
                "separation_m": float(separation),
            }
            if classified is not None:
                contact["digit"], contact["part"] = classified
            contacts.append(contact)
        return contacts

    def _external_contacts(self, held) -> list[dict]:
        robot_paths = set(self._robot.link_prim_paths)
        held_paths = {link.prim_path for link in held.links.values()}
        pairs = self._rigid_contact_api.get_contact_pairs(
            scene_idx=self._robot.scene.idx,
            query_set={held},
            with_set=None,
            current_only=True,
        )
        return [
            {"held_link": held_link, "other": other}
            for held_link, other in sorted(pairs)
            if other not in robot_paths and other not in held_paths
        ]


def _float_mapping(values: Mapping[str, float]) -> dict[str, float]:
    return {str(name): float(value) for name, value in values.items()}


def _record_diagnostic_error(step: dict, phase: str, error: Exception) -> None:
    step.setdefault("diagnostic_errors", []).append(
        {"phase": phase, "type": type(error).__name__, "message": str(error)}
    )


def _contact_aggregate(contacts: list[dict]) -> dict:
    if not contacts:
        return {
            "count": 0,
            "net_force_n": [0.0, 0.0, 0.0],
            "net_force_norm_n": 0.0,
            "sum_force_norm_n": 0.0,
            "max_force_norm_n": None,
            "min_separation_m": None,
        }
    net_force = [
        sum(contact.get("force_n", [])[axis] for contact in contacts if len(contact.get("force_n", [])) > axis)
        for axis in range(3)
    ]
    return {
        "count": len(contacts),
        "net_force_n": net_force,
        "net_force_norm_n": _norm(net_force),
        "sum_force_norm_n": sum(contact["force_norm_n"] for contact in contacts),
        "max_force_norm_n": max(contact["force_norm_n"] for contact in contacts),
        "min_separation_m": min(contact["separation_m"] for contact in contacts),
    }


# --------------------------------------------------------------------------- offline criterion comparison

VIRTUAL_RELEASE_MARGINS_RAD = (0.10, 0.15, 0.25)


def _trace_contacts(entry: Mapping) -> list:
    from dex_teleop.omnigibson.assisted_grasp import DigitContact

    contacts = []
    for row in entry.get("contacts", []):
        contacts.append(
            DigitContact(
                digit=row["digit"],
                part=row["part"],
                normal=None if row.get("normal") is None else tuple(row["normal"]),
                position=None if row.get("position_m") is None else tuple(row["position_m"]),
                force_n=float(row.get("force_n", 0.0)),
            )
        )
    return contacts


def compare_criteria_episode(
    steps: list[dict],
    config: Mapping[str, object],
    *,
    release_margins_rad: tuple[float, ...] = VIRTUAL_RELEASE_MARGINS_RAD,
) -> dict:
    """Re-evaluate both weld criteria on one recorded episode, without the simulator.

    Each criterion is replayed on the recorded hand contacts and finger commands with the recording's own
    parameters (``config`` may be edited to try others). A criterion "holds" at a step when its full
    decision is satisfied there: for ``contact_opposition`` the thumb touches and two digits oppose on
    one graspable object link for a whole ``grasp_window_s``; for ``friction_closure`` thumb pad and
    closure hold on every step of the window and force and intent hold on the window mean.

    Caveat stated in the result: after the recorded weld the grasping fingers were frozen and the object
    was attached, so contacts from then on are not what the other criterion would have produced. The
    comparison is exact up to the first recorded weld and indicative after it.
    """

    from dataclasses import fields

    from dex_teleop.omnigibson.assisted_grasp import (
        THUMB,
        AssistedGraspConfig,
        can_hold_digits,
        evaluate_friction_closure,
        friction_window_verdict,
        has_opposition,
    )

    known = {field.name for field in fields(AssistedGraspConfig)}
    grasp_config = AssistedGraspConfig(**{key: value for key, value in config.items() if key in known})
    if not steps:
        return {"steps": 0, "evaluable": False, "reason": "empty episode"}
    if "hand_contacts" not in steps[0]:
        return {
            "steps": len(steps),
            "evaluable": False,
            "reason": "trace schema 1 has no pre-weld contacts; record again with this version",
        }
    times = [float(step["episode_time_s"]) for step in steps]
    dt = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else 1.0 / 30.0
    window_steps = max(1, math.ceil(grasp_config.grasp_window_s / dt - 1e-9))

    old_run: dict[tuple, int] = {}
    new_run: dict[tuple, list] = {}
    last_break: dict[tuple, str] = {}  # why (a)/(b) last failed per object link
    first_contact = first_old = first_new = None
    old_only = {"thumb_part": 0, "closure": 0, "force": 0, "intent": 0}
    new_only = 0
    both = 0
    old_steps = new_steps = 0
    for index, step in enumerate(steps):
        squeeze = step.get("squeeze_depth_rad", {})
        seen = set()
        old_holds = new_holds = False
        causes = set()
        for entry in step["hand_contacts"]:
            if not entry.get("graspable", True):
                continue
            key = (entry["object"], entry["link_name"])
            seen.add(key)
            contacts = _trace_contacts(entry)
            if first_contact is None and contacts:
                first_contact = index
            old_ok = (not grasp_config.require_thumb or any(c.digit == THUMB for c in contacts)) and has_opposition(
                contacts, grasp_config.opposition_max_normal_dot
            )
            old_run[key] = old_run.get(key, 0) + 1 if old_ok else 0
            link_old = old_run[key] >= window_steps
            evaluation = evaluate_friction_closure(
                contacts, mu=grasp_config.friction_mu, margin=grasp_config.friction_closure_margin
            )
            link_new = False
            cause = None
            if evaluation.passes:
                run = new_run.setdefault(key, [])
                run.append((evaluation, squeeze))
                del run[:-window_steps]
                if len(run) >= window_steps:
                    verdict = friction_window_verdict(
                        [item for item, _ in run],
                        [item for _, item in run],
                        mass_kg=float(entry.get("mass_kg") or float("nan")),
                        config=grasp_config,
                    )
                    link_new = verdict["weld"]
                    if not link_new:
                        cause = "force" if not verdict["force_ok"] else "intent"
                else:
                    # (a) and (b) have not yet held for a whole window: the step that broke the run decides
                    cause = last_break.get(key, "closure")
            else:
                new_run[key] = []
                cause = "thumb_part" if not evaluation.thumb_pad else "closure"
                last_break[key] = cause
            old_holds |= link_old
            new_holds |= link_new
            if link_old and not link_new and cause is not None:
                causes.add(cause)
        for key in [k for k in old_run if k not in seen]:
            old_run[key] = 0
        for key in [k for k in new_run if k not in seen]:
            new_run[key] = []
        old_steps += old_holds
        new_steps += new_holds
        if old_holds and first_old is None:
            first_old = index
        if new_holds and first_new is None:
            first_new = index
        if old_holds and new_holds:
            both += 1
        elif old_holds:
            for cause in ("thumb_part", "closure", "force", "intent"):
                if cause in causes:
                    old_only[cause] += 1
                    break
        elif new_holds:
            new_only += 1

    events = [step.get("supervisor_event") for step in steps]
    kinds = [event.get("kind") for event in events if event]
    recorded_weld = next((i for i, event in enumerate(events) if event and event.get("kind") in {"welded", "adopted"}), None)

    def seconds(index):
        return None if index is None or first_contact is None else (index - first_contact) * dt

    virtual = {}
    for margin in release_margins_rad:
        drops = 0
        segments = set()
        elapsed = 0.0
        dropped = False
        segment = 0
        was_held = False
        for step in steps:
            retreat = step.get("flexion_retreat_rad")
            if retreat is None:
                was_held, elapsed, dropped = False, 0.0, False
                continue
            if not was_held:
                segment += 1
                was_held = True
            digits = set((step.get("supervisor_before") or {}).get("grasp_digits", retreat.keys()))
            engaged = {digit for digit in digits if digit not in retreat or float(retreat[digit]) <= margin}
            if can_hold_digits(engaged):
                elapsed, dropped = 0.0, False
                continue
            elapsed += dt
            if elapsed >= grasp_config.release_window_s - 1e-9 and not dropped:
                drops += 1
                dropped = True
                segments.add(segment)
        virtual[f"{margin:.2f}"] = {"drops": drops, "weld_segments_affected": len(segments)}

    return {
        "steps": len(steps),
        "evaluable": True,
        "control_dt_s": dt,
        "window_steps": window_steps,
        "recorded_criterion": config.get("criterion", "contact_opposition"),
        "first_contact_step": first_contact,
        "recorded_first_weld_step": recorded_weld,
        "contact_to_first_weld_s": {
            "contact_opposition": seconds(first_old),
            "friction_closure": seconds(first_new),
            "recorded": seconds(recorded_weld),
        },
        "steps_holding": {"contact_opposition": old_steps, "friction_closure": new_steps, "both": both},
        "contact_opposition_only_steps_by_cause": old_only,
        "friction_closure_only_steps": new_only,
        "events": {
            "welded": kinds.count("welded") + kinds.count("adopted"),
            "released": kinds.count("released"),
            "broke": kinds.count("broke"),
        },
        "virtual_releases_by_margin_rad": virtual,
        "caveat": "exact up to the first recorded weld; afterwards fingers were frozen and the object welded",
    }
