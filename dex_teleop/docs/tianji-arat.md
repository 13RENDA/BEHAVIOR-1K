# The ARAT tasks with the Tianji Marvin M6 CCS arm (`--arm tianji`)

`launch_og.py --arm tianji` runs the ARAT tasks with the Tianji Marvin Pro torso, its right Marvin
M6 CCS arm and the right Sharpa hand instead of the Franka. The robot is the wrist bench's
(`hand_bench.yaml` `wrist_bench.arms.tianji`, asset `tianji_m6_ccs_sharpa_right`, a reconstruction:
see `datasets/omnigibson-robot-assets/source/tianji_m6_ccs/UPSTREAM.md` and
`tianji-m6-sharpa-mount.md`). What only the ARAT scenes need is in
`src/dex_teleop/arat/arms.yaml`: where the robot stands for each task, how it starts there, and for
the box tasks where the apparatus stands on the table. `--arm franka` (the default) is unchanged:
saved base pose, saved scene, `arat.reset_poses`, same cameras.

## Current assignment (2026-11-15): the 809 mm arm, front of the table, apparatus as saved

The robot now carries the Marvin M6-S CCS-809 arms (809 mm from the J2 axis to the flange instead of 696 mm;
see "The 809 mm arm" in `tianji-m6-sharpa-mount.md`). The user does not want the robot at the table's side and
does not want the apparatus turned; with the longer arm neither is needed:

| Stance | Base (x, y), yaw | Tasks | Apparatus |
| --- | --- | --- | --- |
| `table_front_saved` | (-0.52, 0.10), 15 deg | the 15 box tasks | as saved |
| `table_front_cups` | (-0.61, 0.173), 10 deg | pour water | as saved |
| `mannequin_side` | (-0.46, -0.66), 45 deg | top and back of the head | - |
| `mannequin_front` | (-0.66, -0.472), 15 deg | mouth | - |

`table_front` (apparatus turned a quarter turn) and `table_side` stay defined with start configurations for
the new arm; `layouts.table` in `arms.yaml` switches. Recordings made with the 696 mm arm no longer match
the robot.

What was checked for this assignment:

- Offline palm-path check (`arat_arm_reach.py --arm tianji --path`): all 19 tasks follow, joint margin along
  the paths 11.6 to 17.3 deg.
- In the simulator (scripted palm path through the IK controller, open hand, nothing grasped; scripts and logs in
  `~/claude_work/tianji/m6_809/`):

  | Task | Worst palm error at a waypoint | Arm or body link touching anything |
  | --- | --- | --- |
  | blocks 2.5, 5, 7.5, 10 cm | 1.0 mm | none |
  | cricket ball | 3.0 mm | none (the open palm rests on the ball) |
  | sharpening stone | 0.8 mm | none |
  | marble (index), ball bearing (ring) | 3.9 mm | none |
  | washer over bolt | 0.4 mm | none (fingers touch the box at the peg) |
  | tube 2.5 cm | 1.6 mm | `Arm_R6_Link` at the peg |
  | tube 1 cm | 26.6 mm at the peg | `Arm_R6_Link`, `Arm_R7_Link` at the peg |
  | pour water | 0.0 mm | none |

  The 7.5 and 10 cm blocks, the tubes and pour water were run without lowering the open hand onto the object
  (it would press on it). The other four pinch tasks share the marble's geometry and were not run separately.
- `launch_og.py --config dexemg --arm tianji` headless for a box task and a mannequin task: loads, records,
  replays.

Not checked: the mannequin touches in the simulator (offline only), any grasp, a person teleoperating. The peg
placements are tight against the box: the wrist touches it with the hand turned the way the script turns it.

Three things had to change with the arm:

- The start configurations are in another elbow family than the 696 mm arm's (J1 about +100 deg instead of
  -50 deg): from the old family the long arm cannot raise a level palm above about 1.3 m.
- The left arm is welded hanging beside the torso instead of at the upstream home pose, where it reached
  0.5 m forward at tabletop height and kept the robot away from the table.
- `Arm_[LR]5_Link` collides with its own mesh instead of the placeholder cylinder of the 696 description
  (0.32 m long, centred on the wrist), which reached 0.14 m past the end of the forearm and into the
  tabletop whenever the forearm was vertical above a pick point.

Joint limits are still the teleoperation limits of the 696 description (J2 and J4 stop at -5.7 deg); Tianji's
export of the 809 arm allows J2 +-120 deg and J4 -145..+60 deg.

### Starting with the hand on the table (`--reset-pose rest`)

The operator records the EMG neutral with the hand lying on a table. `--reset-pose rest` starts the robot the
same way: the open hand palm-down 2 cm above the box lid, beside the task object (the default stays
`extended`, the hand 0.35 m above the table).

| Stance | Palm link at (world) | Where | Joint margin at rest |
| --- | --- | --- | --- |
| `table_front_saved` | (0.020, -0.100, 1.099) | 18 cm to the left of the object | 15.0 deg |
| `table_front_cups` | (-0.050, 0.120, 1.099) | left of the cups | 8.5 deg |

The mannequin stances have no table and no `rest` pose. The wrist bench has one too (palm link 7.5 cm above
its table, the floor of the adapter's workspace; with it the bench puts its object beyond the fingertips).

Checked in the simulator from `rest` (same scripted paths as above): nothing touches while holding the pose;
sharpening stone 0.8 mm, block 5 cm 1.1 mm, marble 4.0 mm, block 10 cm 1.1 mm, tube 2.5 cm 3.6 mm worst
palm error. Two things differ from `extended`:

- the joint margin on the way to the shelf is small: J6 comes within 0.2 to 3.7 deg of its limit at transport
  height (13 deg or more from `extended`), although the palm still follows;
- seen from the head camera the forearm lies across the object at the start; the other views show it.

The pose is in the arm family of `extended` (led down from it). Start configurations found independently at
that spot could not be led along the paths, and neither could the hand turned inward by 20 to 60 deg.

### Before the 809 mm arm (kept for the record)

With the 696 mm arm the box tasks could not be finished from the front with the apparatus as saved:

| Point | Distance from the shoulder (m) | Palm margin (m) | With the hand's reach (m) |
| --- | --- | --- | --- |
| pick points (15 tasks) | 0.546 to 0.578 | +0.082 to +0.114 | +0.182 to +0.214 |
| shelf top (6 grasp tasks) | 0.870 to 0.871 | -0.210 to -0.211 | -0.110 to -0.111 |
| tin on the shelf (6 pinch tasks) | 0.895 | -0.235 | -0.135 |
| peg (2 tubes, washer) | 0.764 to 0.765 | -0.104 to -0.105 | -0.004 to -0.005 |

Other front placements with the apparatus as saved (offline, `~/claude_work/tianji/front/`):

- Best reach-sphere margin facing the table within 20 deg (2404 stances, 2 cm grid): base (-0.50, 0.08),
  yaw 20 deg: shelf -0.094 m, tin -0.117 m, peg +0.019 m.
- Turned further toward the box (yaw 50 to 60 deg, base near (-0.38, -0.10)) every sphere margin is
  positive (+0.095 to +0.107 m at worst), but the palm path does not follow: the pick points are then
  0.21 to 0.23 m from the shoulder and the arm ends at a joint limit (path errors 0.13 to 0.55 m).
- Path check on a coarse grid (896 stances, 4 cm, yaw 0 to 80 deg in steps of 20; block to shelf, tube to
  peg, pinch to tin): none follows all three, none follows two.
- A shift of the apparatus toward the robot is limited to 5.1 cm (the lid's near end to the table's front
  edge) and is equivalent to moving the base by the same amount.
- Leaning the body forward (asked 2026-11-15; `~/claude_work/tianji/lean/`). The Marvin Pro has no waist
  joint (the public description has only the two 7-axis arms; the vendor lists 14 degrees of freedom), so
  this is hypothetical: the whole robot tilted about the floor, or a waist joint 0.8 m up. Same palm-path
  check, palm level, front placements on a 4 cm grid, yaw 0 and 10 deg:
  - a fixed lean of 10 to 40 deg, canonical start configuration, 1911 placements per pivot: none follows
    all 15 paths, the best follow 1 of the 3 reference paths;
  - a fixed lean of 0 to 30 deg about the floor with any arm configuration (6 cm grid, 101 placements with
    every reference point within 2 cm of the reach sphere): none follows more than 1 of 3; about the waist
    (20 to 40 deg) no placement brings every reference point that close;
  - a lean that changes during the task (upright for the pick, 10 to 40 deg for the placement, ramping while
    the object is carried), canonical start configuration, 702 placements per pivot: none follows more than
    1 of 3.
  A lean moves the workspace forward without making it deeper. Only one lean schedule and the level palm
  were tried.

## The turned arrangement (`table_front`, not assigned)

For every table task the robot stands at one place on the Franka's side of the table. It cannot work
the saved layout from there, so for the 15 tasks that use the ARAT box the apparatus is turned a
quarter turn on the table (see "Where the robot stands").

Everything here was checked with offline kinematics and with scripted headless runs. Nobody has
teleoperated the Tianji through an ARAT task yet; see "Not verified" at the end.

## Running it

```bash
# glove + VIVE (the default preset), original assisted-grasp rule
python dex_teleop/scripts/launch_og.py --arm tianji --task arat_grasp_block_5cm --assisted-grasp --assisted-grasp-debug
# glove + VIVE, friction-closure rule
python dex_teleop/scripts/launch_og.py --arm tianji --task arat_grasp_block_5cm --assisted-grasp --assisted-grasp-debug \
    --assisted-grasp-criterion friction_closure
# EMG band + VIVE (bridge running, see emg-teleop-runbook.md), original rule / friction-closure rule
python dex_teleop/scripts/launch_og.py --config dexemg --arm tianji --task arat_grasp_block_5cm --assisted-grasp --assisted-grasp-debug
python dex_teleop/scripts/launch_og.py --config dexemg --arm tianji --task arat_grasp_block_5cm --assisted-grasp --assisted-grasp-debug \
    --assisted-grasp-criterion friction_closure
```

Add `--assisted-grasp-weld compliant` for small objects and the tubes (see `assisted-grasp-weld.md`).

What the operator sees in the 15 box tasks: the robot in front of the table, the ARAT box end-on
ahead and to the right, its lid running to the left with the object at its end, straight ahead of
the robot. Shelving is a move of the hand 0.30 m to the right and up, not away from the body. In `arat_grip_pour_water` the scene is the saved one. Since 2026-11-18 its water starts in the red cup on the right (`plastic_tumbler_2`) and is poured into the blue one; the robot stands 4 cm farther from the table than in the box tasks and starts with the hand over the red cup.

`arm: tianji` in a preset does the same. `--reset-pose` accepts only `extended` for the Tianji.
A task the arm cannot do from its stance is refused before the simulator starts
(`--allow-unreachable` loads it anyway); at present every task has a stance. The reach table below
is printed by

```bash
python dex_teleop/scripts/arat_arm_reach.py --arm tianji --path [--markdown]
```

## Where the robot stands

The M6 reaches 0.66 m from its shoulder (the Franka's clamp is 0.80 m), it stands on a 0.754 m square
base plate that does not fit between the table's legs (0.696 m apart), and its arm cannot be led
closer than about 0.35 m to its own shoulder. `arms.yaml` defines its stances and says which task
uses which:

| Stance | Base position (m) | Yaw (deg) | Apparatus | Tasks |
| --- | --- | --- | --- | --- |
| `table_front` | (-0.57, 0.173, 0) | 10 | turned -90 deg about the table's centre (0.2415, -0.1085) | the 15 table tasks whose objects start on the lid of the ARAT box |
| `table_front_cups` | (-0.61, 0.173, 0) | 10 | as saved | `arat_grip_pour_water` |
| `mannequin_side` | (-0.375, -0.575, 0) | 45 | (no table) | `arat_gross_movement_hand_behind_head`, `..._hand_top_head` |
| `mannequin_front` | (-0.575, -0.325, 0) | 15 | (no table) | `arat_gross_movement_hand_mouth` |
| `table_side` | (-0.04, -0.92, 0) | 100 | as saved | none (the first placement, kept so its recordings replay) |

For all 16 table tasks the robot is at the same place: at the table's -X side (where the Franka
stands), 0.44 m in front of the table's edge, turned 10 degrees to its left. A corner of its base
plate is under the tabletop between the two front legs, and its welded left arm hangs past the table's
+Y edge.

### Why the apparatus is turned

As saved, the ARAT box stands across the table from the robot with its open lid toward the robot and
the objects at the lid's near end: picking up and shelving is a 0.30 m reach *away* from the robot and
0.38 m up. The M6 cannot do that from the front:

* The band in which its palm can be led from table height to above the shelf without leaving the
  IK family of the start configuration is 0.15 m deep (palm 0.40 to 0.55 m in front of the base, palm
  down, fingers forward; scratch map on a 5 cm grid). The task needs 0.30 m of depth.
* Tried directly: of 1443 base poses on the table's -X side that stand clear of the table (2 cm
  grid, yaw -10, 0 and 10 degrees), none could follow the path of `arat_grasp_block_10cm` with the
  saved layout (`test_the_saved_layout_cannot_be_worked_from_the_front_of_the_table` keeps a few).
* Moving the apparatus toward the robot as a whole does not help and is not possible: the box, its
  lid and the objects on the lid are one rigid arrangement (the lid is a link of the box), the lid's
  near end is already 5.1 cm from the table's edge, and the 0.30 m between pick point and shelf stays
  0.30 m wherever it stands.

Across the robot, the same band is long enough (this is what the stance at the table's side used).
So `table_front` has a `layout`: everything the task puts on the table (box with its lid, tins,
planks, bolts, the task objects) is turned -90 degrees about the table's centre as one rigid piece.
The box then stands end-on in front of the robot, the lid runs to the robot's left, and shelving is
a move from left to right at a constant 0.6 m in front of the right shoulder. The table itself, the
lights and the mannequin scenes are not touched, and the Franka's scenes are exactly the saved ones
(`arat/layout.py`; a layout is applied to the scene description before OmniGibson loads it).

`table_front` is the image of `table_side` under that quarter turn: robot and apparatus are arranged
exactly as they were there (a test compares every key point in the two base frames), so the reach
table below has the numbers the side stance had. No translation is used: moving the apparatus by a
vector is the same as moving the base by its opposite, and a scan of base poses and shifts found no
better combination than this one. The turned apparatus stands at least 5.5 cm inside the tabletop's
edge (the lid's end, toward +Y).

What follows from the layout being one rigid motion by a quarter turn:

* relative poses of all apparatus parts and all heights are unchanged; the BDDL predicates and the
  scorer's targets (shelf, tins, pegs, pour) are read from the live objects and follow. The scorer
  and `OnShelf` use axis-aligned bounding boxes of the box shell and the tins, which is why a layout
  may only turn by multiples of 90 degrees (`Layout` refuses anything else);
* the cups of `arat_grip_pour_water` are in reach of the same place where they stand, so that task
  keeps the saved scene (`table_front_cups`); the box in it is where it was saved.

A stance square to the table also passes for all 15 tasks, with less reach: base (-0.54, 0.24),
yaw 0 has a worst palm margin of -0.023 m (10 degrees: -0.006 m), a joint-limit margin of 15.2
degrees and its left gripper 3.0 cm from the tabletop's edge (4.6 cm here). To use it, change
`position` and `yaw_deg` of the two front stances; the start configuration stays.

Every stance starts from one configuration (`reset_poses.extended`): the wrist bench's `extended`
pose led to a start palm pose, palm down, fingers along the robot's +X, 0.25 m above the tabletop
for the table stances.

### The mannequin stances

The gross-movement scenes have no table; the two stances are unchanged. Both are in front of the
mannequin (it faces -X): `mannequin_front` 0.62 m in front of its front plane, turned 15 degrees,
for the mouth; `mannequin_side` 0.42 m in front of it and 0.47 m to the side (toward -Y), turned 45
degrees, for the top and the back of the head. A stance squarely in front does not do those two: in
the stance scan (0.1 m / 30 degree grid, see below) the yaw-0 stances its log lists reach the top of
the head with at most 7.9 degrees of joint margin or the mouth with 11.5 degrees, none of them the
back of the head, which was reached only from stances turned 30 degrees or more or standing behind
the mannequin; no stance at any yaw reached all three regions. A layout cannot help here: the
scorer's regions assume the mannequin faces world -X, so it cannot be turned.

### The operator's frame and the cameras

A stance yawed against the world turns the operator's station with it:

* the tracked wrist motion is rotated by the stance yaw on top of `--tracking-yaw-deg`, so pushing
  the hand forward moves the palm along the robot's +X (the adapter now also rotates the tracked
  displacement into the base frame; for the Franka's identity base orientation that changes nothing);
* the scene-frame cameras (the two shoulder cameras, the overview) turn by the stance yaw about the
  stance's `camera_focus`, so they frame the work area as they do for the Franka. Under a layout the
  focus is where the layout puts the work area and the cameras are carried there with it: each has
  the view of the apparatus it would have of the saved layout from a stance turned the same way
  against it (for `table_front`: the views of `table_side`);
* the rig's main camera stands 0.2 m behind the Franka. Turned with a Tianji stance it looks at the
  back of the torso (seen in the first replay), so the Tianji carries it on its head
  (`head_camera`, base frame (0.13, -0.03, 1.50)) looking at the focus;
* the wrist cameras mount on `right_hand_C_MC` as before. The reconstructed asset has no camera pod
  geometry, the sensors are there all the same.

## Reach, task by task

Stance, the points the task needs, their distance from the shoulder, the margin inside the adapter's
0.66 m reach sphere for the palm link itself and with the 0.10 m the hand reaches beyond it, whether
the palm could be led along the task's path from the start configuration, and the smallest
joint-limit margin on that path. The points of the `table_front` tasks are those of the turned
apparatus. For the mannequin tasks the point is the palm position of the touch recorded in
`arms.yaml`. No footprint conflict and no apparatus overhanging the tabletop was reported for any task.

| Task | Stance | Point | Shoulder distance (m) | Palm margin (m) | With hand reach (m) | Path followed | Joint-limit margin on the path (deg) | Reachable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arat_grasp_block_10cm | table_front | start | 0.646 | +0.014 | +0.114 | yes (0.0 mm) | 16.0 | yes |
|  |  | shelf | 0.661 | -0.001 | +0.099 | | | |
| arat_grasp_block_2_5cm | table_front | start | 0.658 | +0.002 | +0.102 | yes (0.0 mm) | 16.0 | yes |
|  |  | shelf | 0.661 | -0.001 | +0.099 | | | |
| arat_grasp_block_5cm | table_front | start | 0.654 | +0.006 | +0.106 | yes (0.0 mm) | 16.0 | yes |
|  |  | shelf | 0.661 | -0.001 | +0.099 | | | |
| arat_grasp_block_7_5cm | table_front | start | 0.650 | +0.010 | +0.110 | yes (0.0 mm) | 16.0 | yes |
|  |  | shelf | 0.661 | -0.001 | +0.099 | | | |
| arat_grasp_cricket_ball | table_front | start | 0.647 | +0.013 | +0.113 | yes (0.0 mm) | 16.0 | yes |
|  |  | shelf | 0.661 | -0.001 | +0.099 | | | |
| arat_grasp_sharpening_stone | table_front | start | 0.649 | +0.011 | +0.111 | yes (0.0 mm) | 16.0 | yes |
|  |  | shelf | 0.666 | -0.006 | +0.094 | | | |
| arat_grip_pour_water | table_front_cups | start | 0.555 | +0.105 | +0.205 | yes (0.0 mm) | 16.0 | yes |
|  |  | pour | 0.576 | +0.084 | +0.184 | | | |
| arat_grip_alloy_tube_2_5cm | table_front | start | 0.650 | +0.010 | +0.110 | yes (0.0 mm) | 16.0 | yes |
|  |  | peg | 0.605 | +0.055 | +0.155 | | | |
| arat_grip_alloy_tube_1cm | table_front | start | 0.650 | +0.010 | +0.110 | yes (0.0 mm) | 16.0 | yes |
|  |  | peg | 0.605 | +0.055 | +0.155 | | | |
| arat_grip_washer_over_bolt | table_front | start | 0.647 | +0.013 | +0.113 | yes (0.0 mm) | 16.0 | yes |
|  |  | peg | 0.606 | +0.054 | +0.154 | | | |
| arat_pinch_ball_bearing_ring | table_front | start | 0.647 | +0.013 | +0.113 | yes (0.0 mm) | 16.0 | yes |
|  |  | tin | 0.658 | +0.002 | +0.102 | | | |
| arat_pinch_marble_index | table_front | start | 0.647 | +0.013 | +0.113 | yes (0.0 mm) | 16.0 | yes |
|  |  | tin | 0.658 | +0.002 | +0.102 | | | |
| arat_pinch_ball_bearing_middle | table_front | start | 0.647 | +0.013 | +0.113 | yes (0.0 mm) | 16.0 | yes |
|  |  | tin | 0.658 | +0.002 | +0.102 | | | |
| arat_pinch_ball_bearing_index | table_front | start | 0.647 | +0.013 | +0.113 | yes (0.0 mm) | 16.0 | yes |
|  |  | tin | 0.658 | +0.002 | +0.102 | | | |
| arat_pinch_marble_ring | table_front | start | 0.647 | +0.013 | +0.113 | yes (0.0 mm) | 16.0 | yes |
|  |  | tin | 0.658 | +0.002 | +0.102 | | | |
| arat_pinch_marble_middle | table_front | start | 0.647 | +0.013 | +0.113 | yes (0.0 mm) | 16.0 | yes |
|  |  | tin | 0.658 | +0.002 | +0.102 | | | |
| arat_gross_movement_hand_behind_head | mannequin_side | palm at the touch | 0.567 | +0.093 | +0.093 | yes (0.0 mm) | 15.1 | yes |
| arat_gross_movement_hand_top_head | mannequin_side | palm at the touch | 0.650 | +0.010 | +0.010 | yes (0.0 mm) | 11.3 | yes |
| arat_gross_movement_hand_mouth | mannequin_front | palm at the touch | 0.626 | +0.034 | +0.034 | yes (0.0 mm) | 16.0 | yes |

Reading the palm margins: around 0 at the table means the palm link stays about 0.10 m short of the
object, which is where it is when the hand holds it. The adapter clamps the palm at 0.66 m.

## How the stances were chosen and what was checked

In the repository (`arat/placement.py`, `scripts/arat_arm_reach.py`, `tests/test_arat_arms.py`):

1. Key points from the task instances and the scorer's own targets (pick point, shelf / tin / peg /
   pour point, mannequin region).
2. Reach sphere and height band of the adapter, as margins.
3. Floor footprint: base plate against the table legs, torso column and the welded left arm against
   the tabletop outline; for the mannequin the base plate against its legs and the tall parts against
   its whole bounding box. Under a layout, every moved object's footprint (asset bounding boxes, the
   box with its open lid) at least 2 cm inside the tabletop's outline (`layout_conflicts`).
4. Leading the palm (`track_palm`): from the start configuration the palm target moves in 5 mm steps
   above the pick point, onto it, up, across at transport height and onto the target (or along the
   recorded touch), each step one damped pseudo-inverse update clipped to the joint limits, no
   null-space term. This is the structure of OmniGibson's IK controller, so it tells whether an
   operator could lead the arm there, which a static IK solution does not.

Step 4 is there because the first stance failed without it. A stance chosen by sphere margin and
static IK alone, base (-0.42, -0.04) yawed 45 degrees on the Franka's side, has an IK solution with
at least 9 degrees of joint margin for every table key point, with the pick points 0.25 m from the
shoulder. Led from its start configuration the palm stops at a joint limit more than 10 cm from
every target (`test_an_ik_solution_is_not_enough_the_abandoned_close_stance_cannot_be_led`).

Outside the repository (scratch tools in `/home/svl/claude_work/tianji/`, Pinocchio + hpp-fcl on the
URDF collision meshes, the scene reduced to boxes: tabletop slab, ARAT box shell, its open lid, the
mannequin in six height slices):

* `workmap.py`, `stance_scan.py`: where the palm can be led from `extended` in the base frame, and
  a scan of stances against it with the saved layout. Only stances at the table's -Y side passed for
  14 or 15 of the 16 table tasks; with the palm down and the fingers forward the band that reaches
  from table height to above the shelf is 0.15 m deep.
* `front/eval_front.py`, `front/saved_layout_front.py` (2026-11-15): the repository's path check for
  front stances. Saved layout: 0 of 1443 stances follow `arat_grasp_block_10cm`. Apparatus turned
  -90 degrees about the table's centre, base on a 2 cm grid, yaw 0, 10 and 20 degrees, apparatus
  shifted by 0 or 4 cm: 274 combinations follow all 15 box tasks before the scan was stopped; a shift
  of the apparatus gave the same numbers as the opposite shift of the base. Among those
  with the full 16.0 degrees of joint margin none has more reach margin than the stance in
  `arms.yaml` (worst palm margin -0.006 m; the best on the scan's grid has -0.013 m).
* `front/front_clearance.py`: the path of step 4 with clearances for `table_front` and
  `table_front_cups` (scene boxes: tabletop, box shell and lid where the layout puts them).
  15 of 15 box tasks followed, joint margin 16.0 degrees, forearm at least 0.025 m above the
  tabletop at the pick point, the fixed body 0.046 m from the scene (a finger of the welded left
  gripper beside the tabletop's +Y edge). For the three peg tasks (both alloy tubes, the washer) the
  reported clearance (0.027-0.034 m) is with the hand turned 30 degrees clockwise at the target, as
  at the side stance, where held straight the forearm's collision cylinder intersected the shell
  box. Pour water from the same place: followed, 16.0 degrees, 0.055 m.
* `mannequin_scan.py`, `mannequin_touches.py`: 1612 stances on a 0.1 m / 30 degree grid, then 832
  around the best on a 2.5 cm / 15 degree grid, for touches of the three regions (palm, finger pads
  or fingertips on the head). No stance reached all three; hence two stances. Recorded touches: behind the head 15.1 degrees and 0.110 m arm
  clearance, top of the head 11.3 degrees and 0.085 m, mouth 16.0 degrees and 0.050 m.

The scorer counts any hand contact inside a region, so the touches use the finger pads or fingertips
(`test_recorded_touches_put_the_hand_inside_the_scoring_regions`).

## Assisted grasp, scoring, recording, replay

* The Tianji robot definition has the same Sharpa finger links and assisted-grasp start/end points
  as the Franka's `sharpa_right` (a test pins this), so `--assisted-grasp`, both criteria,
  `--assisted-grasp-debug` and the ARAT scorer work unchanged.
* A recording stores the arm in `data.attrs["dex_teleop_arm"]` (arm, robot model, stance, base pose,
  the apparatus `layout` or null for the saved one, operator yaw, camera focus, start configuration,
  adapter workspace) next to the saved environment config and scene, which already contain the turned
  apparatus. `replay_data.py` replays that scene with the recorded arm and places the cameras for
  the recorded stance, not for whatever stance the task has today. Before the simulator starts it
  refuses `--arm` naming another arm, a recorded stance that `arms.yaml` no longer defines or defines
  with another base pose or layout, and a recording whose scene does not have the fixed apparatus
  parts where its recorded layout puts them (`recorded_stance`). Recordings made at `table_side`
  (no `layout` key) replay as before; recordings from before `--arm` existed replay as the Franka.

## Headless runs made

### At the front stance (2026-11-15)

`launch_og.py --config dexemg --arm tianji --task arat_grasp_block_5cm --assisted-grasp
--assisted-grasp-debug --no-score` with replayed EMG fingers and a fake VIVE socket (264 recorded
steps, exit 0): stance `table_front`, tracking frame yaw 190 degrees (preset 180 + stance 10). The
recorded scene has the box at (0.5146, -0.1137, 1.0537), which is where the layout puts it, and the
block at (0.0685, 0.1915) with z 0.5 mm below the saved value; the palm started at
(-0.1303, 0.0982, 1.280). The recording's arm provenance carries the stance and the layout and passes
`recorded_stance`. The scripted tracker motion of this run led the palm away from the block (wrong
sign in the script's forward axis), so it shows the load, the start pose and the recording, not a
grasp.

`front/sim_front_check.py` (scratch): the launcher's environment config for the Tianji with the
cameras dropped, 60 steps holding the start pose, then the palm path of the reach table commanded
to the simulator's IK controller in 5 mm steps with an open hand (nothing is grasped).

| Task (stance) | Holding the start pose | Palm path without lowering the hand onto the object | With the hand lowered to the pick point |
| --- | --- | --- | --- |
| `arat_grasp_block_10cm` (`table_front`) | objects within 0.5 mm of the layout's poses, no robot contact | every waypoint within 0.01 mm, lag on the way at most 4.9 mm, joint margin at least 16.3 degrees, no contact | the open hand lands on the 10 cm block (palm 35 mm short), afterwards the smallest joint margin seen at a waypoint was 6.2 degrees; shelf reached within 0.1 mm |
| `arat_grip_alloy_tube_2_5cm` (`table_front`, hand turned -30 degrees at the peg) | 0.0 mm, no contact | every waypoint within 0.01 mm, joint margin at least 17.0 degrees; at the peg pose the pinky's proximal link touches the box shell | the open hand lands on the upright tube (palm 42 mm short); afterwards a joint was at its limit on the way to the peg, palm still within 0.4 mm |
| `arat_pinch_marble_index` (`table_front`) | 0.5 mm, no robot contact | not run | every waypoint within 0.02 mm, lag at most 5.8 mm, joint margin at least 16.5 degrees, no contact |
| `arat_grasp_block_5cm` (`table_front`) | 0.5 mm, no robot contact | not run | every waypoint within 0.01 mm, lag at most 4.9 mm, joint margin at least 16.1 degrees, no contact (the palm stops 3 cm above the block) |
| `arat_grip_pour_water` (`table_front_cups`) | 0.5 mm, no contact | every waypoint within 0.01 mm, joint margin at least 16.5 degrees, no contact | the open hand lands on the cup (palm 50 mm short, a joint at its limit); it then stopped 44 mm short of the point above the second cup with a joint at its limit, and reached the pour point |

No arm or body link (anything but the hand) touched the scene in any of these. The right column is
what the repository's palm path does literally: it lowers a flat open hand onto the object, which
for a block of 10 cm, an upright tube or a cup is a collision of 35 to 50 mm, and the simulator's
IK controller did not always return to a configuration away from the joint limits afterwards. An
operator approaches these objects from the side; the left column is the path without that push.
In the first run of the right column the contact query for the table failed (fixed for the other
runs), so "no arm contact" there covers the task objects and the box only.

### At the side stance (2026-11-14)

The same arrangement of robot and apparatus as `table_front` (see above), before the layout existed.
Replayed EMG fingers (the dexemg bridge replaying a recorded take) and a fake VIVE socket whose
tracker makes a scripted reach to the pick pose; `--config dexemg --arm tianji --task
arat_grasp_block_5cm --assisted-grasp --assisted-grasp-debug`:

| Run | Steps | Result |
| --- | --- | --- |
| original rule (`contact_opposition`) | 293 | 1 weld (index, middle, ring, pinky, thumb) 0.50 s after first contact, 0 broken, held for the remaining 173 steps, weld drift 4.8 mm |
| `friction_closure` | 308 | contacts in 217 steps, pending 31 steps, 0 welds: the thumb touched with a dorsal part, and where it did not, the squeeze test failed |

In both the palm started at the stance's start pose (0.035, -0.480, 1.280). It ended 0.6 mm from the
intended pick pose (-0.012, -0.375, 1.125) in the `friction_closure` run and 9.8 mm from it, holding
the block, in the other; the largest target-to-palm error on the way was 7.1 mm and 9.7 mm.
`analyze_assisted_grasp.py` ran on both recordings, and the first was replayed to a video with the
recorded arm. The `friction_closure` recording was made before `head_camera` existed; its provenance
attribute has `scene_camera_pivot` in place of `scene_camera_focus` and `head_camera`.

## Not verified

* No person has teleoperated the Tianji in an ARAT task. The runs above are scripted.
* A `friction_closure` weld on the Tianji: the scripted grasp was refused, as that rule is meant to
  refuse a grasp without thumb-pad contact and squeeze. It needs a human grasp.
* In the simulator: `table_front` with five of its 15 tasks and `table_front_cups` as described
  above, `table_side` on 2026-11-14. No grasp and no weld has been made at the front stance (the one
  weld above is from the side stance, the same arrangement). `mannequin_side` and `mannequin_front`
  were checked offline only.
* After a hard collision of the hand the simulator's IK controller can leave the arm at a joint
  limit (see the runs above). Whether that happens in teleoperation at this stance is not known.
* Collisions are not part of the repository's checks. The scratch clearances use boxes for the scene
  and exclude the hand; the three peg tasks are tight against the box shell (see above).
* Lifting an object to the shelf, the tin lid, a peg or pouring was run as a palm path with an
  open hand, never with an object in the hand.
* The scene cameras of `table_front` are those of `table_side` carried along with the apparatus
  (a test pins the transform); the turned shoulder cameras and the head camera were looked at in one
  replay at `table_side` on 2026-11-14. The views of `table_front_cups` and of the mannequin stances
  have not been rendered.
* The Tianji asset is a reconstruction; link shapes, joint limits and the 0.66 m reach come from it.
