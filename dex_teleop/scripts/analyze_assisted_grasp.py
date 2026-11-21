#!/usr/bin/env python3
"""Summarize assisted-grasp weld failures recorded by launch_og.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "dex_teleop" / "src"))

from dex_teleop.omnigibson.assisted_grasp_trace import (  # noqa: E402
    ASSISTED_GRASP_GROUP,
    read_assisted_grasp_config,
    compare_criteria_episode,
    read_assisted_grasp_episode,
    summarize_assisted_grasp_episode,
)


def _episode_ids(recording_path: Path) -> list[int]:
    with h5py.File(recording_path, "r") as recording:
        try:
            names = recording[ASSISTED_GRASP_GROUP].keys()
        except KeyError as error:
            raise ValueError(
                "This recording has no assisted-grasp trace. Record it with "
                "--assisted-grasp --assisted-grasp-debug."
            ) from error
        episode_ids = []
        for name in names:
            if name.startswith("demo_"):
                try:
                    episode_ids.append(int(name.removeprefix("demo_")))
                except ValueError:
                    pass
    return sorted(episode_ids)


def _format_optional(value, unit: str, precision: int = 3) -> str:
    return "unavailable" if value is None else f"{float(value):.{precision}f} {unit}"


def _print_comparison(result: dict) -> None:
    if not result.get("evaluable"):
        print(f"  Criteria comparison unavailable: {result.get('reason')}")
        return

    def seconds(value) -> str:
        return "never" if value is None else f"{value:.2f} s"

    delay = result["contact_to_first_weld_s"]
    print(
        f"  Criteria (recorded with {result['recorded_criterion']}): first contact to first weld "
        f"contact_opposition {seconds(delay['contact_opposition'])}, friction_closure "
        f"{seconds(delay['friction_closure'])}, recorded {seconds(delay['recorded'])}"
    )
    holding = result["steps_holding"]
    causes = result["contact_opposition_only_steps_by_cause"]
    print(
        f"    steps satisfied: contact_opposition {holding['contact_opposition']}, friction_closure "
        f"{holding['friction_closure']}, both {holding['both']}"
    )
    print(
        "    contact_opposition only, by what friction_closure lacks: "
        f"thumb part {causes['thumb_part']}, closure {causes['closure']}, force {causes['force']}, "
        f"intent {causes['intent']}; friction_closure only: {result['friction_closure_only_steps']}"
    )
    events = result["events"]
    print(f"    events: {events['welded']} weld, {events['released']} release, {events['broke']} broke")
    virtual = result["virtual_releases_by_margin_rad"]
    print(
        "    virtual releases (weld would have dropped) by release_open_margin_rad: "
        + ", ".join(f"{margin} rad: {row['drops']}" for margin, row in virtual.items())
    )
    print(f"    note: {result['caveat']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--episode", type=int, help="Analyze only this demo_N episode")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable summary")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a criterion parameter for the offline comparison, e.g. --set friction_mu=0.4 "
        "--set friction_squeeze_delta_rad=0.05 (may be repeated)",
    )
    args = parser.parse_args(argv)

    recording_path = args.recording.expanduser().resolve()
    config = read_assisted_grasp_config(recording_path)
    episode_ids = [args.episode] if args.episode is not None else _episode_ids(recording_path)
    comparison_config = dict(config)
    for item in args.set:
        key, _, value = item.partition("=")
        if not _ or key not in comparison_config and not key.startswith("friction_"):
            raise SystemExit(f"--set expects a known KEY=VALUE, got {item!r}")
        comparison_config[key] = float(value)
    traces = {episode_id: read_assisted_grasp_episode(recording_path, episode_id) for episode_id in episode_ids}
    summaries = {episode_id: summarize_assisted_grasp_episode(steps) for episode_id, steps in traces.items()}
    comparisons = {episode_id: compare_criteria_episode(steps, comparison_config) for episode_id, steps in traces.items()}
    if args.json:
        print(
            json.dumps(
                {
                    "recording": str(recording_path),
                    "config": config,
                    "comparison_config": comparison_config,
                    "episodes": summaries,
                    "criteria": comparisons,
                },
                indent=2,
            )
        )
        return

    print(f"Recording: {recording_path}")
    print(
        "Weld config: "
        f"break force={config.get('weld_break_force')} N, "
        f"break torque={config.get('weld_break_torque')} Nm, "
        f"squeeze bias={config.get('frozen_squeeze_bias_rad')} rad, "
        f"drift release={config.get('weld_drift_release_m')} m"
    )
    for episode_id, summary in summaries.items():
        weld_count = sum(event["kind"] in {"welded", "adopted"} for event in summary["events"])
        release_count = sum(event["kind"] == "released" for event in summary["events"])
        print(
            f"\ndemo_{episode_id}: {summary['steps']} action steps, {weld_count} weld(s), "
            f"{release_count} intentional release(s), {len(summary['breaks'])} broken weld(s)"
        )
        for failure in summary["breaks"]:
            time_s = failure["episode_time_s"]
            time_text = "unknown time" if time_s is None else f"t={float(time_s):.3f}s"
            print(
                f"  step {failure['step']} ({time_text}): {failure['object']} -> "
                f"{failure['classification']}, drift={_format_optional(failure['drift_m'], 'm')}, "
                f"max finger force={_format_optional(failure['max_finger_contact_force_n'], 'N')}, "
                f"deepest separation={_format_optional(failure['min_contact_separation_m'], 'm', 5)}"
            )
            if failure["joint_break_events"]:
                paths = ", ".join(event["joint_path"] for event in failure["joint_break_events"])
                print(f"    PhysX JOINT_BREAK: {paths}")
            if failure["external_contacts"]:
                print(f"    Held-object external contacts: {', '.join(failure['external_contacts'])}")
        if not summary["breaks"]:
            print("  No supervisor-detected broken welds in this episode.")
        _print_comparison(comparisons[episode_id])


if __name__ == "__main__":
    main()
