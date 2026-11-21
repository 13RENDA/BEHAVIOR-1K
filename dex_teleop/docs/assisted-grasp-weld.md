# Assisted grasp: rigid and compliant welds

`--assisted-grasp` attaches a grasped object to the palm with a joint ("weld") and freezes the grasping
fingers. `--assisted-grasp-weld` chooses the joint. The default (`rigid`) is the behaviour of the original
supervisor and is unchanged; `compliant` was added on 2026-11-15 because small objects could not be lifted.

```bash
# from the worktree root, behavior_dex env; works with every hand source, arm and criterion
python dex_teleop/scripts/launch_og.py --task arat_pinch_marble_index --assisted-grasp --assisted-grasp-weld compliant \
    --assisted-grasp-debug --no-score
python dex_teleop/scripts/launch_wrist_bench.py --assisted-grasp --assisted-grasp-weld compliant --grasp-object marble
```

Preset keys: `assisted_grasp_weld`, `assisted_grasp_weld_stiffness`, `assisted_grasp_weld_angular_stiffness`.

## Why the rigid weld drops small objects

Measured by re-simulating the operator's own recordings (`arat_pinch_marble_index.hdf5`,
`arat_grip_alloy_tube_2_5cm.hdf5`, glove, Franka, 2026-11-14) from the simulator state stored at each step,
with the recorded palm motion and finger commands (`~/claude_work/emg_teleop/ag_resim.py`,
`ag_resim_supervisor.py`):

- With the 100 N break force PhysX broke the weld joint within 0 to 6 control steps of all five marble welds,
  and at step 0 of the one tube weld that was traced step by step (the other three tube welds moved exactly as
  with no weld at all). The arm and the fingers are stiff position-controlled drives; a rigid joint to the
  palm is loaded far beyond 100 N as soon as the frozen fingers keep pressing or the object is pushed a
  millimetre into what it rests on (tin lid, peg, table). With hand-object collisions switched off the joint
  still broke in three of the five marble welds (after 5 to 21 steps). Two marble welds only survived a limit
  of 3000 N; 1000 N was not enough.
- The supervisor still reports "held" until the object has drifted 5 cm, but from the break on only the
  friction of the frozen fingers carries the object. With the fingers kept frozen and the recorded palm path,
  the marble stayed behind in three of the five welds (palm rose 3.2, 1.9 and 3.3 cm, marble 0.0 to 0.2 cm).
  The earlier block and cricket-ball recordings show the same break event one step after nearly every weld,
  and those objects were still lifted.
- Removing the force limit (`--assisted-grasp-break-force none`) carried the marble in all five welds, but with
  the tube, which sits on a peg, the rigid joint fought the peg through the arm and the simulation diverged in
  one of four welds. It is not a safe default.
- The tube itself is not stuck: its collision shape has 1.8 mm of radial play on the peg (11.77 mm inner
  apothem, 10.0 mm peg radius, 115 mm long, so about 1.8 degrees of tilt), and without a hand it slides off
  freely (commanded straight up with gravity off, upright and tilted 1 and 2 degrees: 20.0 of 20.0 cm). The peg is 13.5 cm tall, so the lift must be that high and roughly straight.

## What the compliant weld does

- The joint is a free 6-DOF joint held by spring-damper drives: 3000 N/m and 20 Nm/rad by default, linear
  damping critical for the held mass. The drive force saturates at `--assisted-grasp-break-force` (100 N) and
  `--assisted-grasp-break-torque` (30 Nm) instead of snapping, so an overloaded grip stretches. The existing
  drift release (5 cm) then drops it, with the same "assisted grasp broke" message.
- Collisions between the hand links and the held object are filtered while it is held, so the frozen fingers
  no longer fight the joint. The grasping digits freeze at their measured positions (no squeeze bias; nothing
  would resist it). After a release the collisions come back once no hand link's bounding box overlaps the
  object, or after 1 s at the latest.
- Weld decision (criterion), release rule, drift release and the ARAT scorer are untouched.

## Results on the recorded attempts (closed loop, real supervisor code)

Each recorded weld was restarted 14 steps earlier and replayed through `AssistedGraspSupervisor` in both
modes. Only the part up to the first release is a fair comparison, because the operator's later motion
reacted to the failed grasp.

| Recording | Weld | Operator's palm rise while held | Object rise, rigid | Object rise, compliant | Largest weld stretch, compliant |
| --- | --- | --- | --- | --- | --- |
| marble | 0 | 3.2 cm | 0.1 cm | 4.2 cm | 0 mm |
| marble | 1 | 6.2 cm | 6.7 cm | 6.7 cm | 5 mm |
| marble | 2 | 6.3 cm | 6.2 cm, weld "broke" after 30 steps | 6.8 cm, still held after 120 steps | 1 mm |
| marble | 3 | 0.1 cm | 0.2 cm | 0.8 cm | 6 mm |
| marble | 4 | 0.7 cm | 0.2 cm | 1.0 cm | 2 mm |
| tube 2.5 cm | 0 | 0.6 to 1.5 cm | 4.0 cm | 3.3 cm | 49 mm |
| tube 2.5 cm | 1 | 4.4 cm | 1.9 cm | 2.9 cm | 41 mm |
| tube 2.5 cm | 2 | 0.9 cm | 0.9 cm | 1.1 cm | 26 mm |
| tube 2.5 cm | 3 | 14.0 cm (4.7 cm before the rigid weld broke) | 2.1 cm | 16.9 cm, clear of the peg | 32 mm |

Tube welds 0 to 2 are attempts in which the hand did not rise the 13.5 cm the peg needs; weld 3 is the only
full lift and the only one that can show whether the tube comes off. No run diverged in compliant mode (top
object speed 1.3 m/s, the released tube falling).

## Not verified, known limits

- The numbers above are re-simulations of two Franka recordings. Since then a person has used the compliant
  weld on the Tianji (2026-11-15): tube 2.5 cm off its peg and onto the target peg twice, washer lifted 0.46 m;
  see `assisted-grasp-reference.md`. The 1 cm tube and the marble have not been tried by a person.
- The sharpening stone is a different problem, see below: the weld type does not decide it.
- While the tube is on the peg a sideways hand motion stretches the spring one for one (the peg holds the
  tube); more than 5 cm drops the grasp. Lift straight.
- For up to 1 s after a release the hand passes through the released object if it is closed onto it again,
  and when collisions return with the hand still overlapping, the object is pushed out (0.17 m/s in the one
  case where it was measured; two more such restores in the marble session were not measured).
- A recorded episode stores the weld as a fixed joint (OmniGibson serializes nothing else), so replay holds
  the object rigidly; the stored configuration (`assisted_grasp/config_json`, with `--assisted-grasp-debug`)
  says `weld_mode: compliant`. Replay of a compliant recording was not run.
- Loading a simulator state that keeps the same compliant weld held (save and load in place) crashed the
  PhysX GPU solver on the next grasp when it was done during the first grasp of a process. No launcher does
  this; resetting an episode while holding (R) was tested as the first grasp and later, in both orders of
  `env.reset()` and `supervisor.reset()`, without a crash.
- Stiffness values are first guesses that worked on these recordings, not tuned.

## The sharpening stone: the grasp is rarely established at all

`arat_grasp_sharpening_stone.hdf5` (glove, Franka, 2026-11-14, eight non-empty episodes). The stone is
10 x 2.5 x 1 cm and stands on its 1 cm edge on the box cover.

- In the six episodes analysed (1,945 steps) two digits touched the stone in 46 steps, 45 of them in one
  episode (`demo_6`); in the others the hand touched the stone in 1 to 17 steps. A weld needs the thumb and
  a second digit opposing each other for 0.3 s.
- `demo_6` welded twice (index + thumb); both welds were dropped by the drift release, after 17 and 2 steps.
  Before the first weld the hand touched the box cover in 75 steps and the stone in 55.
- Re-simulating `demo_6` does not reproduce this reliably, because the stone is already spinning between
  the fingers when the weld forms (0.6 m/s, 50 rad/s one step before it in one run). Rigid, one run: both
  welds held for the 52 steps replayed. Compliant, weld 0 in five runs: held and lifted in two (stretch 0 mm
  and 25 mm, one with a 4.5 m/s jerk), no weld in two, welded and dropped one step later in one. Compliant,
  weld 1 in two runs: held in both (stretch 1 to 3 mm).

So for the stone there is no evidence that the compliant weld is better; what limits it is getting the thumb
and a finger onto the two long faces without knocking it over.
