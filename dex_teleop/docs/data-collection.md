# Data collection: EMG + glove teleoperation recordings

How to record teleoperation data on the Tianji arm + Sharpa hand in OmniGibson with the EMG wristband, the MANUS
glove and the VIVE wrist tracker. One HDF5 file per simulator session holds the robot actions and states, the raw
EMG, the glove's skeleton and joint angles, the wrist pose, and the alignment between them.

This page is the operating procedure. Background and measurements are in
[`emg-teleop-runbook.md`](emg-teleop-runbook.md), the hardware setup in
[`manus-linux-vive-ultimate-runbook.md`](manus-linux-vive-ultimate-runbook.md), the arm placement in
[`tianji-arat.md`](tianji-arat.md), assisted grasping in [`assisted-grasp-reference.md`](assisted-grasp-reference.md).

## What is recorded, and the two modes

| Mode | Fingers of the robot follow | Wrist follows | Recorded next to it | File prefix |
| --- | --- | --- | --- | --- |
| Glove-driven | MANUS glove | glove + VIVE tracker | raw EMG | `<task>_...` |
| EMG-driven | EMG decoder (the bridge) | glove + VIVE tracker | raw EMG, and the glove as ground truth | `emgctl_<task>_...` |

Both modes need all three devices worn on the right arm: the EMG wristband, the MANUS glove and the VIVE Ultimate
tracker on the wrist mount. Glove-driven data is the usual training data (clean labels, the task succeeds).
EMG-driven data shows how the operator moves when the decoder is in the loop.

## One-time setup

1. Simulator environment, assets and the MANUS SDK: sections 0 to 5 of the [README](../README.md).
2. VIVE Hub for Linux with the tracker paired and its map built: `manus-linux-vive-ultimate-runbook.md`.
3. The bridge runs in a second conda environment, `emg2pose` (PyTorch plus the decoder code). It needs:
   - the `dexemg` repository with its vendored `emg2pose` at `~/dexemg` (or `DEXEMG_ROOT=<path>`). Only the
     EMG-driven mode loads a model; a record-only bridge does not.
   - the Synchroni SDK build the wristband firmware expects, in a folder of its own:
     `pip install --no-deps --target <dir> sensor-sdk==0.9.11 bleak==1.1.1`, then pass `--sdk-path <dir>` or set
     `DEXEMG_SYNCHRONI_SDK=<dir>`. On the lab machine it is `~/claude_work/emg_teleop/sensor_sdk_0911`.
4. Task scenes. `datasets/` is not in git. Two scenes differ from the distributed zip and must match this code:
   `arat_grip_washer_over_bolt` (the washer lies on the lid, no tin) and `arat_grip_pour_water` (the water starts
   in the red cup on the right). Either copy `datasets/arat-task-instances/` from the lab machine, or regenerate
   it with `python scripts/arat/generate_task_instances.py` followed by
   `python scripts/arat/validate_task_instances.py` (both start the simulator).
5. EMG-driven mode only: a decoder checkpoint and the operator's hand model.
   - Checkpoint: produced by the decoder training code (the `dexemg` repository), normally fine-tuned on the
     day's calibration takes. Not covered here.
   - Hand model: fitted once per operator and glove calibration from glove-driven recordings,
     `python dex_teleop/scripts/fit_glove_hand_model.py ~/emg_teleop_data/*.hdf5 --output glove_fit_<operator>.json --operator <operator>`.

Run every command below from the repository root.

## Each session

Start things in this order. Each numbered item is its own terminal.

**0. Calibration takes (HandSync).** With the band and glove on as they will stay, record the day's calibration
takes in HandSync: `calibrate-neutral`, `calibrate-articulate`, `calibrate-open_close`, `calibrate-free`,
`calibrate-arat` (see `~/HandSync/README.md`). They are the decoder's fine-tuning data for this wearing.
**Close HandSync afterwards**: it owns the wristband and the MANUS dongle while it runs.

**1. VIVE Hub.** `cd ~/vivehub/VIVEHub-Linux && ./run-vivehub.sh`. Wait until the tracker shows *tracking*.

**2. The bridge window** (`conda activate emg2pose`). The bridge owns the wristband and hands its raw EMG to the
simulator.

```bash
# Glove-driven sessions: record only, no model
python dex_teleop/scripts/dexemg_gui.py --record-only

# EMG-driven sessions: decoder + the operator's hand model
python dex_teleop/scripts/dexemg_gui.py --checkpoint <decoder.pt> \
  --hand-model <glove_fit_operator.json> --smoothing 0.9 --deadband-deg 4
```

Wait for `CONNECTED`. For EMG-driven sessions press **Record neutral** and hold the neutral posture of the
calibration takes for 12 s; do this again whenever the band has been taken off. The window shows the sample
rate, lost samples, each channel's level and impedance, and whether a simulator is recording.

**3. The simulator** (`conda activate behavior_dex`). Define the two helpers once per terminal:

```bash
# Glove-driven: the glove moves the robot, the EMG is recorded
gloverec() {
  python dex_teleop/scripts/launch_og.py --arm tianji --task "$1" \
    --assisted-grasp --assisted-grasp-weld compliant --no-score \
    --emg-bridge --recording-path ~/emg_teleop_data/${1}_$(date +%m%d_%H%M%S).hdf5
}

# EMG-driven: the decoder moves the fingers, the glove is recorded as ground truth
emgrec() {
  python dex_teleop/scripts/launch_og.py --arm tianji --task "$1" \
    --hand-source dexemg --record-hand-source manus \
    --assisted-grasp --assisted-grasp-weld compliant --no-score \
    --emg-bridge --recording-path ~/emg_teleop_data/emgctl_${1}_$(date +%m%d_%H%M%S).hdf5 \
    --maximum-frame-age 0.5 --stale-frame-policy hold
}
```

Then one line per task, for example `gloverec arat_pinch_marble_index` or `emgrec arat_grasp_block_10cm`.
Always give `--recording-path` an absolute path, as the helpers do: a bare file name lands in the current folder.

Tasks:

| Subscale | `--task` |
| --- | --- |
| Grasp | `arat_grasp_block_10cm`, `arat_grasp_block_7_5cm`, `arat_grasp_block_5cm`, `arat_grasp_block_2_5cm`, `arat_grasp_cricket_ball`, `arat_grasp_sharpening_stone` |
| Grip | `arat_grip_pour_water`, `arat_grip_alloy_tube_2_5cm`, `arat_grip_alloy_tube_1cm`, `arat_grip_washer_over_bolt` |
| Pinch | `arat_pinch_marble_index`, `arat_pinch_marble_middle`, `arat_pinch_marble_ring`, `arat_pinch_ball_bearing_index`, `arat_pinch_ball_bearing_middle`, `arat_pinch_ball_bearing_ring` |
| Gross movement | `arat_gross_movement_hand_behind_head`, `arat_gross_movement_hand_top_head`, `arat_gross_movement_hand_mouth` |

Keys in the simulator window:

| Key | Effect |
| --- | --- |
| SPACE | engage teleoperation and start recording an episode (again: re-anchor the wrist) |
| R | reset the scene and the robot; the episode is kept |
| F | reset and mark the episode as failed (it stays in the file, labelled) |
| G | drop the object the hand holds |
| B | switch the main camera (chest / overview) |
| T / E | show the commanded wrist target / the measured hand frame |
| Ctrl+C | save and quit |

One launch can hold many episodes: SPACE, do the task, R or F, SPACE again. Stop with Ctrl+C and wait for
`Recording saved: ...`. The launcher refuses to start unless the tracker is trackable, the glove streams and
the bridge answers.

The EMG is saved only for the episodes (from SPACE to the reset), not for the time in between.

### Fixed wrist (no tracker)

For finger-only data, `launch_hand_bench.py` shows the hand alone and needs no VIVE tracker:

```bash
python dex_teleop/scripts/launch_hand_bench.py --hand-source manus --manus-calibration <right.mcal> \
  --emg-bridge --recording-path ~/emg_teleop_data/bench_$(date +%m%d_%H%M%S).hdf5
```

## What is in a recording

| Group | Content |
| --- | --- |
| `data/demo_N` | `action` (28 per step: 6 arm, 22 Sharpa joints), `state` (simulator state, `state_size` valid entries per row), `reward`, `terminated`, `truncated`. Attribute `dex_teleop_episode_label`: `kept` or `failed` |
| `data` attributes | `config`, `scene_file`, `dex_teleop_arm` (robot, stance, reset pose), `dex_teleop_failed_episodes` |
| `action_timing/demo_N` | host monotonic time just before and after each `env.step`, simulator time |
| `evaluation/demo_N/steps` | per step, the task's goal conditions and whether each holds (JSON) |
| `emg/samples` | `signal_uv` (n, 8) raw microvolts at about 500 Hz, `sample_index`, `is_lost`, `estimated_monotonic_ns` |
| `emg/batches` | per Bluetooth packet: arrival time, impedance per channel, saturation |
| `emg/neutral_N` | the neutral recordings of the bridge (`signal_uv`, `channel_scale_uv`; attribute `active`) |
| `emg` attributes | band, firmware filters, bridge settings and, in EMG-driven mode, the checkpoint |
| `synchronization/demo_N` | `action_emg_row_start` / `_end`: the EMG rows that belong to each action |
| `hand_tracking/articulation_streams/articulation.manus` | glove skeleton (25 joints) at its own rate; each frame's `provenance_json` holds `ergonomics_deg`, the glove's 20 joint angles |
| `hand_tracking/articulation_streams/articulation.dexemg` | EMG-driven only: the decoded hand (21 landmarks); `provenance_json` holds `angles_rad_json`, the 20 decoded angles |
| `hand_tracking/retargeting_streams/*` | the 22 Sharpa joint commands the retargeter produced |
| `hand_tracking/wrist_streams/*` | wrist pose from the tracker |
| `hand_tracking/action_alignment/demo_N` | for each action, the row of each stream it was computed from |
| `assisted_grasp/demo_N/steps` | only with `--assisted-grasp-debug`: contacts, forces and the grasp supervisor's state per step |

All host times are `time.monotonic_ns` of the recording machine, so the streams can be aligned directly.

Helpers:

```python
from dex_teleop.emg.recording import read_glove_ergonomics            # glove angles with their times
from dex_teleop.omnigibson.episode_labels import kept_episodes, read_failed_episodes
```

`python dex_teleop/scripts/replay_data.py <file.hdf5> --episode-id <N> --headless --output <clip.mp4>` renders an
episode.

## Checks before leaving

- The file name printed after `Recording saved:` exists in `~/emg_teleop_data/` and no `.in_progress` file is
  left next to it (one is left only when a session was killed).
- Episodes that went wrong were ended with F, not R.
- The bridge window showed no lost samples and "raw EMG being recorded by the simulator: YES" during episodes.
- Close the bridge window when done. A bridge left open with the band off keeps scanning for it.

## Known behaviour

- **Grasping** is assisted: when the thumb and another finger hold an object for 0.3 s of simulated time, it is
  attached to the hand until the hand opens. Creating the attachment pauses the simulator for 0.2 to 1.9 s.
- **Pour water**: the water starts in the red cup on the right and is poured into the blue one. The cups are
  attached rigidly (the spring attachment used for the other objects crashes the GPU physics next to water).
  This was verified on replays only; report a crash with the simulator log.
- **Tracker not trackable at start**: wait for VIVE Hub to show *tracking*; at night it can take one or two
  minutes. If it never does, rebuild the map in VIVE Hub.
- **Bridge cannot bind port 8765**: an earlier bridge left helper processes behind. `ss -ltnp | grep 8765` names
  them; end them and start the bridge again.
- **"manus is stale" right after a grasp** on a build from before 2026-11-18: update; the launcher now waits for
  the tracking data to catch up after the pause.
- **Disk**: a recording takes 39 to 47 MB per recorded minute (10 sessions of 2026-11-18/19). A crashed
  simulator leaves about 860 MB in `/tmp`; the next launch removes it.

## Tests

```bash
# in behavior_dex; no hardware or GPU needed. The deselected test fails for a reason unrelated to this pipeline.
cd dex_teleop && python -m pytest tests -q --deselect tests/test_hand_bench.py::test_hidden_prim_names_and_pose_error
```
