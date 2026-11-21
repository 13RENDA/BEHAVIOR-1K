#!/usr/bin/env python3
"""Print whether an ARAT arm reaches every task from its stance (no simulator).

    python dex_teleop/scripts/arat_arm_reach.py --arm tianji            # reach sphere, footprint
    python dex_teleop/scripts/arat_arm_reach.py --arm tianji --path    # + lead the palm along each task's path (Pinocchio)

For each task: its stance, the points it needs (where the tracked object starts and where it must end
up, or the palm position of the recorded touch of a mannequin region), their distance from the
shoulder, the margin inside the adapter's reach sphere for the palm link itself and with the hand's
own reach, and any conflict of the robot's floor footprint with the table legs or the mannequin.
A stance that re-places the apparatus on the table (``layout`` in arat/arms.yaml) is tested on the
scene as it will load: the points are the moved ones, and an apparatus that would overhang the
tabletop is a conflict.
``--path`` leads the palm from the stance's start configuration along the task's path the way the
simulator's IK controller would (``dex_teleop.arat.placement.track_palm``) and reports whether it
follows and the smallest joint-limit margin on the way. Collisions are not checked.
``--markdown`` prints the table of docs/tianji-arat.md. Exit status 1 when a task is not reachable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "dex_teleop" / "src"))

from dex_teleop.arat import AratTaskCatalog  # noqa: E402
from dex_teleop.arat.arms import load_arat_arms, task_path_reach, task_reach  # noqa: E402
from dex_teleop.arat.placement import ArmIk, tabletop_margin  # noqa: E402


def arm_ik(arm) -> ArmIk:
    """Pinocchio model of the arm from its robot definition (URDF path, arm joint names)."""

    import yaml

    robot = arm.bench.robot
    root = REPOSITORY_ROOT / "datasets" / robot.dataset_name
    definition = yaml.safe_load((root / "models" / robot.model / f"{robot.model}.yaml").read_text(encoding="utf-8"))
    joints = list(definition["manipulation"]["arm_joint_names"][robot.arm_name])
    return ArmIk(root / definition["urdf_path"], joints, arm.bench.palm_link)


def main(argv: list[str] | None = None) -> int:
    arms = load_arat_arms()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=arms.names, default="tianji")
    parser.add_argument("--path", action="store_true", help="also lead the palm along every task's path (Pinocchio)")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)
    arm = arms.arm(args.arm)
    if arm.stances is None:
        print(f"Arm {arm.key!r} stands at the pose saved with each task instance: the instances were laid out for it.")
        return 0
    ik = arm_ik(arm) if args.path else None
    rows = []
    all_reachable = True
    for task in AratTaskCatalog().tasks.values():
        reach = task_reach(arm, task)
        path = task_path_reach(arm, task, ik) if ik else None
        verdict = reach.reachable and (path is None or path.ok)
        all_reachable &= verdict
        rows.append((task, reach, path, verdict))
    if args.markdown:
        header = "| Task | Stance | Point | Shoulder distance (m) | Palm margin (m) | With hand reach (m) |"
        divider = "| --- | --- | --- | --- | --- | --- |"
        if ik:
            header += " Path followed | Joint-limit margin on the path (deg) |"
            divider += " --- | --- |"
        print(header + " Reachable |")
        print(divider + " --- |")
    for task, reach, path, verdict in rows:
        for index, point in enumerate(reach.points):
            first = index == 0
            if args.markdown:
                line = (f"| {task.activity if first else ''} | {reach.stance.name if first else ''} | {point.key_point.name} | "
                        f"{point.distance:.3f} | {point.palm_margin:+.3f} | {point.grasp_margin:+.3f} |")
                if path is not None:
                    line += (f" {'yes' if path.follows else 'NO'} ({path.max_position_error_m * 1000:.1f} mm) | {path.joint_limit_margin_deg:.1f} |"
                             if first else " | |")
                print(line + (f" {'yes' if verdict else 'NO'} |" if first else " |"))
            else:
                text = (f"{task.activity:38s} {reach.stance.name:15s} {point.key_point.name:17s} shoulder {point.distance:.3f} m  "
                        f"palm margin {point.palm_margin:+.3f}  with hand {point.grasp_margin:+.3f}")
                if path is not None and first:
                    text += (f"  path {'followed' if path.follows else 'NOT followed'} (max error {path.max_position_error_m * 1000:.1f} mm, "
                             f"{path.max_orientation_error_deg:.1f} deg), joint margin {path.joint_limit_margin_deg:.1f} deg")
                print(text + ("" if not first or verdict else "  NOT REACHABLE"))
        if reach.conflicts:
            print(f"{'' if args.markdown else task.activity + ': '}footprint conflicts: {', '.join(reach.conflicts)}")
        if reach.excluded_reason:
            print(f"{'' if args.markdown else task.activity + ': '}excluded: {reach.excluded_reason}")
    print()
    in_use = {reach.stance.name for _, reach, _, _ in rows}
    for stance in arm.stances.values():
        text = f"{arm.key} stance {stance.name}: position {list(stance.base.position)}, yaw {stance.base.yaw_deg:g} deg ({stance.description})"
        if stance.layout is not None:
            margins = [tabletop_margin(task, stance.layout) for task, reach, _, _ in rows if reach.stance.name == stance.name]
            worst = min((margin for margin in margins if margin is not None), default=None)
            text += (f"; apparatus turned {stance.layout.yaw_deg:g} deg about {[round(v, 4) for v in stance.layout.about]} and moved by "
                     f"{list(stance.layout.translation)} m" + ("" if worst is None else f", at least {worst[0] * 100:.1f} cm inside the tabletop edge ({worst[1]})"))
        print(text + ("" if stance.name in in_use else " [not assigned to any task]"))
    print("every task reachable" if all_reachable else "NOT every task is reachable")
    return 0 if all_reachable else 1


if __name__ == "__main__":
    sys.exit(main())
