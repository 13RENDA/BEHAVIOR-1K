# Assisted grasp: modes, commands and parameters (living reference)

Keep this page current: every new assisted-grasp option, default or finding goes here first (change log at
the end). Background and measurements: `assisted-grasp-weld.md` (welds), `tianji-arat.md` (arm, stances,
start poses), `TASK_assisted_grasp_criterion_2026-11-13.md` (the criterion task).

Last updated: 2026-11-15.

## The modes

Assisted grasp is off unless `--assisted-grasp` is given. With it on, two independent choices make four modes:

- **Criterion** (when may a weld form): `contact_opposition` (default) or `friction_closure`.
- **Weld** (how the object is held once welded): `rigid` (default) or `compliant`.

| # | Criterion | Weld | Extra flags after `--assisted-grasp` | Status |
| --- | --- | --- | --- | --- |
| 0 | off | - | (omit `--assisted-grasp`) | plain physics grasping |
| 1 | `contact_opposition` | `rigid` | (none: the defaults) | the original behaviour. The weld breaks within steps (100 N limit), so only finger friction carries the object: blocks and the cricket ball work, marble, washer and tubes do not |
| 2 | `contact_opposition` | `compliant` | `--assisted-grasp-weld compliant` | **use this one for now.** With a person on the Tianji (glove, 2026-11-15): the 2.5 cm tube lifted off its peg and carried to the target peg twice, the washer lifted 0.46 m. Ball bearing: add `--assisted-grasp-window 0.1` |
| 3 | `friction_closure` | `rigid` | `--assisted-grasp-criterion friction_closure` | not recommended: stricter weld rule on top of the weld that breaks |
| 4 | `friction_closure` | `compliant` | `--assisted-grasp-criterion friction_closure --assisted-grasp-weld compliant` | experimental: with the default `delta` (0.03) real grasps are refused; add `--assisted-grasp-squeeze-delta-rad 0` to try it |

Both launchers print the active criterion and weld at start ("Assisted-grasp weld: RIGID ..." / "COMPLIANT ...").

## Commands

From `/home/svl/BEHAVIOR-1K/.claude/worktrees/emg-teleop`, `behavior_dex` environment. `<AG>` stands for the
flags of the chosen mode, e.g. mode 2: `--assisted-grasp --assisted-grasp-weld compliant`.

| What | Command |
| --- | --- |
| ARAT task, glove | `python dex_teleop/scripts/launch_og.py --arm tianji --task <task> <AG> --assisted-grasp-debug --no-score --recording-path <file.hdf5>` |
| ARAT task, EMG band | same with `--config dexemg` (bridge / `dexemg_gui.py` running) |
| Wrist bench, glove | `python dex_teleop/scripts/launch_wrist_bench.py --arm tianji <AG> --grasp-object marble` |
| Wrist bench, EMG band | same with `--config dexemg` |
| Record the EMG during any of the `launch_og.py` rows | add `--emg-bridge` (bridge / `dexemg_gui.py` running, also with the glove); see `emg-teleop-runbook.md` |
| Hand on the table at start | add `--reset-pose rest` (Tianji table tasks and wrist bench) |
| Franka instead of Tianji | drop `--arm tianji` (no `rest` pose) |
| Offline comparison of the criteria | `python dex_teleop/scripts/analyze_assisted_grasp.py <file.hdf5> [--set friction_squeeze_delta_rad=0]` |

Tasks: `arat_grasp_block_2_5cm`, `_5cm`, `_7_5cm`, `_10cm`, `arat_grasp_cricket_ball`,
`arat_grasp_sharpening_stone`, `arat_grip_pour_water`, `arat_grip_alloy_tube_2_5cm`, `_1cm`,
`arat_grip_washer_over_bolt`, `arat_pinch_marble_index` / `_middle` / `_ring`,
`arat_pinch_ball_bearing_index` / `_middle` / `_ring`, `arat_gross_movement_hand_behind_head` / `_top_head` /
`_mouth`. Wrist bench objects: see `--grasp-object` (`launch_wrist_bench.py --help`).

## Parameters

Preset key = the flag without the leading dashes, dashes as underscores (e.g. `assisted_grasp_weld: compliant`
in `configs/*.yaml`). "WB" = also accepted by the wrist bench.

| Flag | Default | Used by | Meaning | Notes |
| --- | --- | --- | --- | --- |
| `--assisted-grasp` | off | all (WB) | turn assisted grasp on | |
| `--assisted-grasp-debug` | off | all (WB: accepted, records nothing) | store the per-step trace needed by `analyze_assisted_grasp.py` | always use it when recording |
| `--assisted-grasp-criterion` | `contact_opposition` | all (WB) | weld rule: thumb + another digit with opposing normals for 0.3 s, or `friction_closure` | |
| `--assisted-grasp-window` | `0.3` s | all (WB) | how long the weld rule must hold without interruption before welding | `0.1` for the 6 mm ball bearing (it stays pinched 0.1 to 0.23 s at a time); shorter = welds on briefer touches |
| `--assisted-grasp-release-rule` | `digits` | all (launch_og) | how opening the hand drops the object: `digits` = fewer than two grasping digits still near their weld-time flexion; `mean` = the grasping digits' mean opening reaches the margin | use `mean` with EMG fingers: with `digits` a five-digit EMG grasp was never released (thumb and pinky barely open); on the one EMG hold of 2026-11-16 `mean` would have released 1.0 s before the glove reached 15 deg of opening, no early drop |
| `--assisted-grasp-release-margin` | `0.25` rad | all (launch_og) | how far a digit must open from its weld-time command | 0.20 or less released 6 s before the glove opened on that hold (risk of dropping while carrying) |
| key `G` | - | launch_og with `--assisted-grasp` | drops the held object at once | manual fallback |
| `--assisted-grasp-weld` | `rigid` | all (WB) | `rigid` fixed joint or `compliant` spring joint | compliant: hand-object collisions off while held, fingers frozen at measured pose. Objects with meta links (the pour-water cups) get the fixed joint instead, without break limits, also in `compliant` mode (2026-11-19) |
| `--assisted-grasp-break-force` | `100` N (`none` disables) | rigid: PhysX break force; compliant: where the spring force saturates | | `none` with rigid diverged once on the tube |
| `--assisted-grasp-break-torque` | `30` Nm (`none` disables) | rigid: break torque; compliant: torque saturation | | |
| `--assisted-grasp-weld-stiffness` | `3000` N/m | compliant (WB) | linear spring stiffness | first guess, not tuned |
| `--assisted-grasp-weld-angular-stiffness` | `20` Nm/rad | compliant (WB) | angular spring stiffness | first guess, not tuned |
| `--assisted-grasp-squeeze-bias-rad` | `0.05` rad | rigid | extra closing of the frozen fingers while welded | ignored by compliant (0) |
| `--assisted-grasp-mu` | `0.5` | friction_closure (WB) | Coulomb friction coefficient | 0.5 = PhysX default material, which hand and objects use |
| `--assisted-grasp-weight-factor` | `2.0` | friction_closure (WB) | `k` in `mu (F_thumb + F_opp) >= k m g` | |
| `--assisted-grasp-force-floor` | `0.5` N | friction_closure (WB) | minimum normal force on the thumb and on the opposing side | blocked one real 2.5 cm block grasp (2026-11-15), not examined yet |
| `--assisted-grasp-squeeze-delta-rad` | `0.03` rad | friction_closure (WB) | `delta`: how far the grasping digits must be commanded past their contact | **not usable as it is**: real grasps show 0.000 to 0.020 rad; 0.03, 0.01 and 0.005 refuse them. 2026-11-16 glove teleop on the Tianji (5 grasps, 4 objects, reconstructed from the recorded states): the least-squeezed grasping digit reads 0.000 to 0.005 rad, free-air noise p99 0.002 to 0.004 rad, so only 0 accepts all 5 and no value separates grasp from noise. Use `0` to switch the gate off; a better intent signal is undecided |
| `--object-depenetration-velocity` | `3.0` m/s | all (WB), with `--assisted-grasp` | cap on how fast PhysX pushes an object out of a penetrating finger or table | `0.3` against flings of thin objects (sharpening stone): in re-simulation of three flings the stone moved 0.2 to 2.9 cm instead of 3.8 to 25.6 cm. Not yet tried by a person |
| `--manus-haptics` | off | launch_og, MANUS glove (python backend, Metaglove Pro Haptic) | per-finger vibration from the simulated contact force (0.5 N starts, 6 N saturates: `--manus-haptics-min-force`, `--manus-haptics-max-force`, `--manus-haptics-min-amplitude`, `--manus-haptics-smoothing`, `--manus-haptics-rate`) | existed before; not available with the EMG band (no actuator) or in the wrist bench |
| `--manus-haptics-grasp-hold` | `0.35` | with `--manus-haptics` and `--assisted-grasp` | amplitude on the grasping digits while an object is held (0 = no grasp cue) | new 2026-11-15; needed because a compliant weld removes hand-object contact |
| `--manus-haptics-grasp-pulse` | `0.15` s | same | full-amplitude pulse on the grasping digits when the weld forms | new 2026-11-15; unit-tested only, not felt on a glove yet |
| `--reset-pose` | `extended` | all (WB) | start pose; `rest` = hand lying just above the table | Tianji table stances and wrist bench only |
| `--no-score` | off | launch_og | do not end the episode when the ARAT scorer finishes | use when recording several grasps |
| `--allow-unreachable` | off | launch_og | load a task the arm cannot reach | not needed since the 809 mm arm |

Fixed in code (no flag): release window 0.15 s, release when the grasping digits open
0.25 rad, drift release at 5 cm of weld stretch, thumb required, collisions back at most 1 s after a compliant
release, compliant angular damping 0.2 Nm s/rad.

## What is known to work and not (2026-11-15)

| Object | Mode 1 (rigid) | Mode 2 (compliant) |
| --- | --- | --- |
| blocks, cricket ball | lifted (by finger friction) in the 2026-11-14 glove sessions | not yet tried by a person |
| tube 2.5 cm (on a 13.5 cm peg) | not lifted (2026-11-14) | **person, Tianji, 2026-11-15** (`tianji_block_2_5cm.hdf5`, which is the tube task): lifted off the peg and carried to the target peg in 2 of 2 welds, held 901 and 987 steps, weld stretch up to 37.7 mm |
| washer | stayed in its lid (3 welds, 2026-11-15 first session), later flicked away by a fingertip on its rim | **person, Tianji, 2026-11-15** (`tianji_washer.hdf5`): 1 weld, held 867 steps, lifted from 1.046 to 1.511 m, stretch up to 25.0 mm |
| ball bearing (6 mm, 1.1 g) | - | **person, Tianji, 2026-11-15** (`tianji_bearing_index.hdf5`): no weld in 3 episodes with the 0.3 s window (thumb + index pinched it for at most 7 steps). Re-simulated with `--assisted-grasp-window 0.1`: welds in both episodes replayed and rises with the palm (5.6 and 5.7 cm, stretch 0 mm). Not yet tried by a person with 0.1 |
| marble | stayed in its lid (2026-11-14) | follows in re-simulation (5 of 5 welds); not yet tried by a person |
| sharpening stone (10 x 2.5 x 1 cm on its edge, 60 g) | the weld rarely forms at all (two digits rarely touch it) | **person, Tianji, 2026-11-15** (`tianji_sharpening_stone.hdf5`, two sessions): never welded; one finger (index) pressed on it with up to 326 N and it was flung away in 5 episodes. Thumb + index pinched it for at most 3 steps. Try `--object-depenetration-velocity 0.3 --assisted-grasp-window 0.1 --manus-haptics` (none of the three tried by a person yet). A heavier stone does not help reliably (x5 mass: one of three re-simulated flings got worse) |

## Change log

- 2026-11-19: in `compliant` mode an object with meta links (the two pour-water cups, whose fillable volume is a meta link) is now welded with OmniGibson's fixed joint, without break limits, instead of the spring joint. The launcher says `assisted grasp welds <object> rigidly, without break limits (object with meta links)`. Reason: both pour-water sessions of 2026-11-18 that gripped the filled cup ended on a PhysX GPU failure (CUDA error 700, `arat_grip_pour_water_1118_212438` and `_222803`). Replays of those recordings (`~/claude_work/emg_teleop/pour/repro_weld_water.py`, headless): with the spring weld the failure came every time the weld formed at recorded step 422 of the second session (11 of 11), in 1 of 2 runs where it formed at step 431, and in 1 of 2 replays of the first session from its first weld on; spring welds formed at other steps did not fail. No failure without water in the cup (28 runs), with the fixed joint (11 runs, at the same steps and over the same sessions), or in replays that made no weld (5 runs). Freezing every finger at the weld, or limiting how far past contact a finger is commanded, did not prevent it (1 run each, weld at step 422). With the fixed joint and its 100 N / 30 Nm break limits the cup was let go once or twice in each of 4 replays of one carry, so the limits are left off for these objects. After the change: no failure at steps 422 and 431 (2 runs each) nor over the first session (2 runs). Why the spring joint next to a fluid breaks the GPU solver is not known. Not yet run by a person. `rigid` mode and all other objects are unchanged.
- 2026-11-18 (night): no assisted-grasp behaviour changed. Creating a weld holds the control loop: measured from the recordings, 0.75 to 1.87 s at every weld of the 2.5 cm tube (compliant weld), about 0.7 to 1.2 s at the first weld of a session and 0.20 to 0.36 s at later ones for the blocks and the marble. Two glove sessions on the tube ended on "selected articulation source 'manus' is stale (0.297 s / 0.430 s > 0.250 s)" right after a weld: the launcher judged the glove stream by receipts it had fallen behind on during its own stall. `launch_og.py` now lets that bookkeeping catch up (up to 0.5 s, `wait_for_fresh_streams`) after any loop iteration longer than `--maximum-frame-age` before the freshness check; a stream that really stopped is still reported. Not yet confirmed in a live session. The weld's own stall is unchanged.
- 2026-11-16 (night, later): `--assisted-grasp-release-rule {digits,mean}` and `--assisted-grasp-release-margin` (defaults unchanged: digits, 0.25) and the G key (drop the held object). Reason: EMG teleop could not release grasps (`~/claude_work/emg_teleop/release/`).
- 2026-11-16 (night): compliant weld no longer filters hand collisions for objects with meta links (the pour-water cups' fillable volume): editing their filtered pairs invalidated the physics views and the launcher died on the next step ("Failed to get rigid body transforms from backend"). Those objects are still welded with the spring joint; only their collisions with the hand stay on. Reproduced and checked on the user's recording (`~/claude_work/emg_teleop/pour/`); washer re-check unchanged.
- 2026-11-16: no behaviour change. `delta` re-checked on the day's five glove teleop takes (`~/claude_work/emg_teleop/delta/`): still not settable (see its row). Ctrl+C no longer loses recordings (MANUS bridge and EMG recorder in their own session).
- 2026-11-15 (night): `--emg-bridge` added to `launch_og.py` and `launch_hand_bench.py` (raw EMG recorded through
  the dexemg bridge during glove or EMG teleoperation). No assisted-grasp behaviour changed.
- 2026-11-15 (evening): `--object-depenetration-velocity` added (default 3.0 unchanged) against flings; haptic grasp
  cue added to the existing `--manus-haptics` (`--manus-haptics-grasp-hold`, `--manus-haptics-grasp-pulse`).
  Sharpening-stone sessions analysed.
- 2026-11-15 (later): `--assisted-grasp-window` added (default 0.3 s unchanged) for the ball bearing. First sessions
  with a person on the Tianji in mode 2: tube and washer carried. Corrected: the recording named
  `tianji_block_2_5cm` is the 2.5 cm tube task, not a block.
- 2026-11-15: page created. Compliant weld added (mode 2 and 4). `--reset-pose rest`. Launchers print the
  active weld at start. Washer session analysed (it ran with the rigid weld).
- 2026-11-13: `friction_closure` criterion and its four parameters added; default criterion unchanged.
