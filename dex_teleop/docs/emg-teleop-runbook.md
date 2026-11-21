# EMG teleoperation runbook (wrist EMG band -> Sharpa fingers, VIVE Ultimate -> wrist)

```text
wrist EMG band ──BLE──▶ dexemg bridge (emg2pose env) ──TCP 127.0.0.1:8765, JSON lines──▶ DexemgArticulationSource ┐
  BC:93:2A:3F:80:E7      neutral scale, streaming decoder,                                  (--hand-source dexemg)  ├▶ worker ▶ retargeter ▶ Sharpa
                         forward kinematics -> MediaPipe-21                                                          │
VIVE Ultimate ──dongle──▶ VIVE Hub for Linux tracker_server ──/tmp/vut.sock──▶ ViveUltimateWristSource ──────────────┘
                                                                               (--wrist-source vive_ultimate, no glove)
```

The fingers come from dexemg's wrist-band decoder, the wrist from the VIVE Tracker Ultimate read
directly from VIVE Hub (no MANUS glove, no MANUS SDK). Preset: `configs/dexemg.yaml`.

## What the bridge does (and why)

`dex_teleop/src/dex_teleop/emg/dexemg_bridge.py`, a separate process in the `emg2pose` conda env (torch,
dexemg, Synchroni SDK). Every choice matches how the decoder was trained:

| Step | Live | Training data |
| --- | --- | --- |
| Band | wrist band `BC:93:2A:3F:80:E7` (HandSync slot `right/wrist`) | same band |
| Firmware filters | HPF, LPF, 50 Hz and 60 Hz notch all ON, verified by read-back | all four ON in every 11-02..11-06 manifest |
| Samples | raw SDK microvolts, 500 Hz, 8 per BLE packet | raw microvolts (exporter does no software filtering) |
| Normalization | neutral of the same wearing, 12 s, per-channel std after trimming 0.5 s per end | `compute_noise_floor_scale` on the nearest neutral take (11-13 s) |
| Decoding | 100 ms chunks, 50 ms overlap, window mean-centred / scale, LSTM state and pose carried, never reset | live_pose_demo `LiveInference` (checked bit-equal) |
| Landmarks | emg2pose FK on the mirrored (right) generic hand, MediaPipe-21, OpenXR wrist frame, m | - |

The live demo's rest + max (MVC) scaling is not offered: offline it scored worse than a constant pose.

Checks (`/home/svl/claude_work/emg_teleop/check_bridge_replay.py`, 2026-11-13): the bridge's neutral scale
equals training's exactly; its streamed angles equal `LiveInference` (max 6e-7 rad) and the offline streaming
study's predictions (float16 storage); closed glove frames bend the landmarks to the palm side of a right hand.

## Model

`/home/svl/claude_work/emg_teleop/deploy_wrist_s0.pt`: wrist-band EMGDecoderModel, stage-1 init, E8 recipe
(Adam 1e-4, 400-sample windows, 10 epochs), every labelled two-band take of 11-02..11-05 (378 min, both
postures, with and without item). On 11-06, never trained on, through the bridge's own streaming decoder:

| Model | 11-06 streaming MAE | constant mean pose | ratio |
| --- | --- | --- | --- |
| deploy_wrist_s0 (11-02..11-05) | 4.41 deg | 7.71 deg | 0.57 |
| e8_k18_wrist_s0 (11-02, 11-03 + 18 takes of 11-04) | 4.62 deg | 7.71 deg | 0.60 |

10 takes (rounds 41-44), 30.1 min, each with its nearest neutral (`eval_1106.py`). Known from the streaming
study: closed grasps come out under-flexed (about 8 deg at the MCPs), the decoder leads the glove by ~60 ms.

## Run it

1. **Wear** the wrist band where it is worn for recordings and the VIVE Ultimate on the wrist mount.
   Close HandSync: it would own the band (and the MANUS dongle).
2. **VIVE Hub**: `cd ~/vivehub/VIVEHub-Linux && ./run-vivehub.sh`; the tracker must show *tracking*.
3. **Bridge with its control window** (terminal 1):

   ```bash
   conda activate emg2pose
   python dex_teleop/scripts/dexemg_gui.py --checkpoint ~/claude_work/emg_teleop/deploy_wrist_s0.pt
   ```

   The window shows, colour-coded, whether the band is connected (every Bluetooth adapter is scanned
   in turn and the search retries until the band answers), the firmware filters read back, the sample
   rate and losses, the live level of each channel, the neutral recording (button, countdown, progress,
   ratio to the training neutrals, a warning beyond 3x), the decoder rate and whether the simulator is
   attached. Press **Record neutral** in the neutral pose of the HandSync neutral takes and hold still
   for 12 s; record it again every time the band is put on. Closing the window stops the bridge.
   Without a display: `python dex_teleop/scripts/dexemg_bridge.py --checkpoint ...` (Enter records the
   neutral, q + Enter quits). `--adapter hci1` pins one adapter.

   The bridge imports the Synchroni SDK build HandSync records with (sensor-sdk 0.9.11 + bleak 1.1.1),
   installed for the emg2pose interpreter in `~/claude_work/emg_teleop/sensor_sdk_0911`
   (`pip install --no-deps --target <dir> sensor-sdk==0.9.11 bleak==1.1.1`; `--sdk-path` overrides).
   Do not use `HandSync/synchroni-sensor-sdk` (1.2.6): it treats the gForceUltra bands as the old EMG
   type, without firmware filters and with another microvolt conversion.
4. **Check axes and offset** (terminal 2, `conda activate behavior_dex`):
   `python dex_teleop/scripts/launch_wrist_bench.py --config dexemg` -- SPACE engages, Y measures
   `tracking_yaw_deg`, rotating in place checks `vive_tracker_to_wrist_*` (as in the MANUS runbook).
5. **Teleoperate**: `python dex_teleop/scripts/launch_og.py --config dexemg --task <activity>`.

Both launchers refuse to start Isaac Sim unless the bridge answers and streams and the tracker is
`trackable` (pre-flight, a few seconds).

Glove on as ground truth (fingers still from EMG, the glove recorded next to them):
`launch_og.py --config manus_core --hand-source dexemg --record-hand-source manus --task <activity>`.
Fixed wrist, fingers only: `launch_hand_bench.py --hand-source dexemg`.

### Starting from the neutral posture (Tianji)

If the neutral was recorded with the hand lying on a table, start the robot the same way with
`--reset-pose rest` (`launch_og.py --arm tianji` table tasks and `launch_wrist_bench.py --arm tianji`): the open
hand lies palm-down just above the table, and engaging anchors your resting hand onto it. See
`tianji-arat.md` for where the hand lies and what was checked.

## Comparing with the glove (live)

Wear the band and the right MANUS glove together, with the bridge streaming. In `behavior_dex`:

```bash
python dex_teleop/scripts/dexemg_compare.py --manus-calibration ~/manus_setup/calibration/right.mcal
```

It retargets the EMG hand and the glove hand with two copies of the teleoperation retargeter and shows
the difference of the 22 Sharpa joint commands: MAE next to a constant baseline (the glove command's own
session mean), ratio, correlation, per finger, for the last 10 s and the whole session. On exit it saves
`summary.json` (per joint, lag, open/closed extremes) and `samples.npz` (paired commands, both raw
landmark streams, the decoder's 20 angles) under `~/dexemg_teleop_logs/compare_<time>/`. `--no-gui`
prints a line every 2 s. The tool owns the glove dongle: run the simulator with the EMG hand
(`launch_hand_bench.py --hand-source dexemg`, or `--config dexemg`), not with the glove, at the same
time; it needs no simulator. The number is the finger difference between glove and EMG teleoperation
(decoder error plus the generic-hand-model vs glove-skeleton difference), not the human joint-angle
error of the offline studies. Checked without hardware by replaying take 11-04 grasp round 30 against
its own glove angles (`claude_work/emg_teleop/run_compare_replay.sh`): 4.55 deg vs constant 7.94 over
60 s with the other-days model, as in the offline end-to-end check.

Fixed wrist, arm resting as in the training recordings (no VIVE needed):
`python dex_teleop/scripts/launch_hand_bench.py --hand-source dexemg --stale-frame-policy hold`.

## Recording the EMG while teleoperating (`--emg-bridge`)

Add `--emg-bridge` to a launcher and the raw EMG of the band is saved inside the recording, aligned with
every action. The bridge keeps the band (one process per device) and passes every sample on to the
launcher, so it works with either hand source:

```bash
# terminal 1 (emg2pose): the bridge window, as above; wear the band, record the neutral
python dex_teleop/scripts/dexemg_gui.py --checkpoint ~/claude_work/emg_teleop/deploy_wrist_s0.pt

# terminal 2 (behavior_dex), GLOVE teleoperation + EMG recorded
python dex_teleop/scripts/launch_og.py --arm tianji --task <task> --assisted-grasp --assisted-grasp-weld compliant \
  --no-score --emg-bridge --recording-path ~/emg_teleop_data/<name>.hdf5

# or EMG teleoperation + EMG recorded
python dex_teleop/scripts/launch_og.py --config dexemg --arm tianji --task <task> --emg-bridge \
  --recording-path ~/emg_teleop_data/<name>.hdf5

# fixed wrist, glove (hand bench)
python dex_teleop/scripts/launch_hand_bench.py --hand-source manus --manus-calibration <right.mcal> --emg-bridge \
  --recording-path ~/emg_teleop_data/<name>.hdf5
```

For glove teleoperation the bridge does not need the decoder: `python dex_teleop/scripts/dexemg_gui.py --record-only`
(or `dexemg_bridge.py --record-only`) starts it without a checkpoint and without loading a model. It owns the band
and passes the raw EMG on; nothing is decoded, so `--hand-source dexemg` refuses such a bridge. A neutral is optional
there: one recorded in the window is saved with the recording but not checked against the training neutrals.

`--emg-bridge` implies `--emg` (and recording); `--no-emg-display` drops the waveform panel. The launcher
checks the bridge before Isaac Sim starts: the band must be streaming; with the glove a neutral is asked for
but not required. The bridge window shows "raw EMG being recorded by the simulator: YES" while a launcher
records. The wrist bench does not record.

What the recording gets (same layout as `--emg`, see the README):

| Path | Content |
| --- | --- |
| `/emg/samples/signal_uv` | every sample, 8 channels, microvolts exactly as the SDK delivers them (what HandSync records), from the first recorded action (the first SPACE) to the end of the last recorded step |
| `/emg/samples/sample_index`, `is_lost` | the band's sample counter (made continuous across a reconnect, `sample_index_resets` counts them) and its lost flag |
| `/emg/samples/estimated_monotonic_ns` | host time (`time.monotonic_ns`) of every sample, see below |
| `/emg/batches/receive_monotonic_ns` | host time of every SDK callback in the bridge |
| `/emg/neutral_<i>` | every neutral of the bridge session: `signal_uv` (the 12 s), `channel_scale_uv`, time, `rejected`, `active`; a neutral recorded during the session is added when it is recorded |
| `/synchronization/demo_N` | EMG row range of the episode and of every action |
| `/hand_tracking/articulation_streams/articulation.manus` | the glove at its own rate with host capture times. From 2026-11-16 every frame's `provenance_json` also carries the glove's 20 MANUS ergonomics angles (`ergonomics_deg`, SDK 3.2 order, with their own publish/arrival times): the decoder's training labels. `dex_teleop.emg.recording.read_glove_ergonomics(path)` reads them; recordings made before then have only the skeleton |

Several takes of one task in one launch: SPACE starts, R resets the robot and objects and ends the episode
(`demo_0`, `demo_1`, ... in the same file, each with its own `/synchronization/demo_N` rows), SPACE again for the
next; Ctrl+C at the end. F does what R does but marks the episode that just ended as failed: it stays in the file,
labelled (`data/demo_N.attrs["dex_teleop_episode_label"]`, the list in `data.attrs["dex_teleop_failed_episodes"]`);
`dex_teleop.omnigibson.episode_labels.kept_episodes(path)` lists the ones to use. Needs `--no-score` (the ARAT
scorer ends the session when an item finishes). F exists in `launch_og.py` only.

Electrode impedance: the bridge reads the band's impedance packets (as HandSync does) and shows them per channel in
its window ("Z ... kOhm": green up to 500 kOhm, amber up to 999 kOhm, red above; grey when older than 5 s); with
`--emg-bridge` they also reach the simulator's EMG panel and the recording (`emg/batches/impedance_ohm`, the latest
reading at each callback, NaN before the first; `impedance_available`). Not yet seen with the real band: if the band
sends no impedance packets while streaming, the fields stay "-" / NaN.

The sidecar streams from launch to exit, but like every other stream the saved EMG starts with the recording:
the samples before the first recorded action and after the last step are dropped when the file is written
(`trimmed_rows_before` / `trimmed_rows_after`; this applies to `--emg` too). Their times are estimated on the
whole stream first, so the cut does not shorten the data the clock fit uses. A session in which SPACE was never
pressed saves no EMG.

Not available through the bridge (zero / -1 / NaN, flagged in the attributes): `raw_adc`, the SDK's millisecond
timestamp, saturation.

Clock: the band does not run at the host's 500 Hz. Bridge logs of 2026-11-14 (566 s, 327 s, 977 s) show
501.37, 501.43 and 501.35 samples per host second, no sample missing. `origin + index x 2 ms`, the estimate
of `--emg`, therefore drifts 2.7 ms per second of recording: on those logs the callbacks at the start come
1,545 / 934 / 2,599 ms after their estimate while those at the end come 76 / 60 / 108 ms after. With
`--emg-bridge` the sample period is measured per recording (per stretch between counter restarts) from the
least-delayed callback of every 5 s (`fitted_sample_period_ns`); on the same logs start and end then agree
(54 vs 49, 60 vs 56, 61 vs 54 ms; `~/claude_work/emg_teleop/emg_rec/drift_check.py`). `--emg` itself (the
SDK sidecar) keeps the nominal period (its filter default is all four filters on since 2026-11-15). The remaining constant, the delay of the fastest
callback behind the muscle signal, is not measured.

Checked without hardware (replay bridge on its own port, headless): `launch_og.py --config dexemg --arm tianji`
and `launch_hand_bench.py --hand-source dexemg`, both with `--emg-bridge` (`run_smoke_emg_bridge.sh`,
`run_smoke_emg_bridge_hb.sh` in `~/claude_work/emg_teleop/`): the recorded samples are bit-identical to the
replayed take, the neutral is stored, every action has its row range. The record-only bridge was run as a real process on a replayed take and
recorded through `EmgSession` (`run_smoke_record_only.sh`): 25.04 s, 12,520 samples, bit-identical. Not yet run with
the real band or with the glove as hand source.

## EMG teleoperation: finger accuracy and options (2026-11-16)

Tonight's EMG-driven takes with the glove worn as ground truth (`--hand-source dexemg --record-hand-source manus`,
checkpoint `deploy_wrist_rest_ft1116pm_s0.pt`, 6 takes, 940.5 s), re-decoded from the recorded EMG and compared
with the glove's own MANUS ergonomics (the training labels), MAE deg model / constant, r of the flexion sum:
thumb 6.67 / 8.92, 0.64; index 12.07 / 14.16, 0.61; middle 10.27 / 13.87, 0.78; ring 10.48 / 13.61, 0.81;
pinky 8.94 / 9.78, 0.69; all 9.69 / 12.07. The model hardly opens the fingers beyond straight: flexion-sum p5 glove
vs model index -38 / -15, middle -46 / -17, ring -40 / -6, pinky -26 / +15 deg. Script and numbers:
`~/claude_work/emg_teleop/fingers/`.

Options for EMG fingers: `--assisted-grasp-release-rule mean` (opening releases on the mean over the grasping
digits; the default rule needs all but one digit to open, which EMG fingers did not), the G key (drop the held
object), `--pinky-follows-ring W` (pinky flexion blended toward the ring's; 0.25 lowered the pinky error 9.00 ->
8.56 deg on those takes, 0.75 and 1.0 raised it). The ergonomics `publish_time` of the Integrated SDK 3.2.1 is 0:
tell ergonomics callbacks apart by `arrival_monotonic_ns`.

## Hand model: the glove-fit one (`--hand-model`, 2026-11-18)

The decoder outputs the glove's 20 joint angles; the bridge turns them into 21 landmarks, which the Sharpa
retargeter consumes exactly as it consumes the glove's skeleton in glove teleoperation. By default that step uses
emg2pose's generic hand, and its thumb is not the glove's: for the SAME (true) joint angles the landmarks differ by
57.2 mm on the thumb (tip 90.6 mm, thumb metacarpal direction 47.9 deg off) and 17 to 27 mm on the other fingers
(9,766 moments of the 2026-11-16/17 EMG-driven takes). The retargeter then drives the Sharpa thumb, the DIP joints
and the pinky to their limits: with the glove's own angles the Sharpa command is 8.50 deg from what glove
teleoperation commands (thumb 16.12), and the EMG command is 11.87 deg, worse than a constant (10.53).

`dex_teleop.emg.hand_model.GloveFitHandModel` reproduces the glove skeleton instead (rigid bones of the glove's
lengths, each bone's direction regressed on its finger's four angles, inputs clipped to the fitted range):

```bash
# fit (behavior_dex): every teleoperation recording since 2026-11-16 holds the glove skeleton and its angles
python dex_teleop/scripts/fit_glove_hand_model.py ~/emg_teleop_data/*.hdf5 \
  --output ~/claude_work/emg_teleop/hand_model/glove_fit_OP02.json --operator OP02
# use (emg2pose): add --hand-model to the bridge or its window
python dex_teleop/scripts/dexemg_gui.py --checkpoint <ckpt> --hand-model ~/claude_work/emg_teleop/hand_model/glove_fit_OP02.json
```

The bridge prints `hand model: GLOVE FIT ...` and reports it in its `hello`. The file is specific to one operator's
hand and glove calibration: refit for another person or after a new glove calibration. `glove_fit_OP02.json`: 29
recordings, 251,284 glove frames; landmark error with each recording held out 0.21 (thumb) to 1.03 mm (pinky),
worst recording 6.25 mm (pinky). The default stays the generic hand.

Sharpa command error against what glove teleoperation commands (the glove skeleton through the same retargeter), on
9,766 moments of the EMG-driven takes, with the hand model fitted on the glove-driven takes only (deg; all joints /
thumb), and the share of joint-frames at a limit:

| input, hand model | all | thumb | at a limit | thumb IP at its limit |
| --- | --- | --- | --- | --- |
| glove's own angles, generic hand | 8.50 | 16.12 | 55.2% | 88.0% |
| EMG angles, generic hand (the default path) | 11.87 | 15.13 | 50.0% | 92.0% |
| glove's own angles, glove-fit hand | 0.50 | 0.41 | 39.2% | 56.5% |
| EMG angles, glove-fit hand | 8.19 | 7.95 | 28.4% | 27.4% |
| constant (mean glove command) | 10.53 | 9.00 | | |
| glove teleoperation itself | | | 39.1% | 58.0% |

The offline path reproduces the Sharpa commands glove teleoperation recorded (600 consecutive frames: p95 difference
0.08 deg). With the range-of-motion recording `rom_glove_1118_151525.hdf5` added, the file in use is fitted on 31
recordings (264,907 frames); the file from before it is `glove_fit_OP02_before_rom.json`. Not yet judged in the
simulator by a person. Scripts: `~/claude_work/emg_teleop/retarget_sat/`.

## Finger jitter: smoothing and deadband (2026-11-18)

Four EMG-driven takes of 2026-11-18 (999 s; bridge `--smoothing 0.98`, glove-fit hand model), EMG hand vs the worn
glove as 15 finger bend angles at 50 Hz. In the 0.5 s windows where the glove is still (431 s; glove peak-to-peak
0.31 deg) the EMG bends wander 3.28 deg peak-to-peak (median; p90 14.36, worst joint p90 30.33): small jitter plus
occasional slow drifts of the decoder. Candidate filters on the recorded stream
(`~/claude_work/emg_teleop/jitter/jitter_filters.py`):

| filter | wander of a still hand, median / p90 (deg) | change between simulator steps, RMS (deg) | error vs glove (deg) | added delay |
| --- | --- | --- | --- | --- |
| as recorded | 3.28 / 14.36 | 3.83 | 8.85 | 0 |
| deadband 4 deg | 0.00 / 9.19 | 2.95 | 8.80 | 0 ms, motion starts 4 deg late |
| deadband 8 deg | 0.00 / 4.92 | 2.26 | 9.10 | 0 ms, motion starts 8 deg late |
| extra smoothing 0.2 s | 1.92 / 8.88 | 2.16 | 8.64 | 80 ms |
| extra smoothing 0.4 s | 1.35 / 6.23 | 1.49 | 8.50 | 100 ms |
| smoothing 0.2 s + deadband 4 deg | 0.00 / 5.10 | 1.66 | 8.63 | 135 ms |
| smoothing 0.4 s + deadband 4 deg | 0.00 / 3.30 | 1.13 | 8.54 | 205 ms |

Bridge options: `--smoothing F` (an exponential average per 500 Hz sample, time constant 1 / (500 (1 - F)) s: 0.98 =
0.1 s, 0.9933 = 0.3 s, 0.996 = 0.5 s) and `--deadband-deg X` (per joint, after the smoothing; the output moves only
as far as needed to stay within X of the input, so it never jumps; default 0 = off). A deadband removes the small
jitter without delay; it cannot remove the decoder's larger slow drifts.

## Logs

The bridge writes `~/dexemg_teleop_logs/<start time>/` on exit: `emg.npz` (raw microvolts, sample index,
monotonic time), `decoded.npz` (every decoded frame), `neutral_<n>.npy` (each neutral recording),
`session.json` (checkpoint, scales, filters, status). `--no-log` disables it.

## Timing

Each decoded hand is stamped with the time it reached the teleop process (its EMG sample time is kept in
`source_timestamp_ns`). The newest hand is then paired with the newest wrist: the decoder already leads the
glove, and the arm does not wait for the EMG pipeline. Budget measured on 11-04 replay: bridge sample-to-send
17-19 ms for the newest frame of each chunk (also with Isaac Sim running), TCP 1 ms, retargeting 11-42 ms, plus
the 100 ms chunking (50 ms on average). The decoder runs on the CPU by default, on two threads that sleep between chunks (`--threads`, default 2, since 2026-11-18). Before that it used PyTorch's default of one thread per core, whose idle threads spin: on replayed EMG the bridge process took 1339% CPU with 33.8 ms per chunk and a 90 ms median send delay, against 19 to 22% CPU, 12.7 to 19.8 ms per chunk (idle machine / simulator running) and 60 to 67 ms with the new default. The EMG-driven sessions of 2026-11-17/18, recorded next to the 13-core bridge, ran at 5.7 to 8.3 simulator steps per second with a median frame age of 340 to 414 ms; the glove-driven ones of 2026-11-17 (record-only bridge) at 9.5 to 12.3 and 124 to 183 ms. Three of them ended on "record-only articulation source 'manus' is stale (> 0.5 s)"; `--maximum-frame-age 1.0` tolerates such a stall. Found on 2026-11-18 (night): creating an assisted-grasp weld holds the control loop for 0.75 to 1.87 s on the tube (0.20 to 0.36 s on later welds of the blocks), and the launcher then judged the glove stream by receipts it had fallen behind on; this, not only the bridge's CPU use, is a likely cause of those three endings (their logs are gone, so not verified). The launcher now lets the stream bookkeeping catch up after such an iteration (see `assisted-grasp-reference.md`). Earlier figure: about 10 ms per chunk, steady;
`--device cuda:0` halves that on an idle GPU but competes with Isaac Sim. `--chunk-ms` is
exposed, but the decoder's accuracy was only verified at 100 ms chunks / 50 ms overlap.

A bridge that cannot find the band scans again every few seconds. Until 2026-11-18 every such retry left four
SDK threads behind (the SDK controller is one object per process and its `terminate()` only works the first
time): a bridge that had been retrying for 58 min had 864 threads and used 132% CPU without decoding anything.
The bridge now stops the SDK's BLE host on every retry. Checked without hardware, against a Bluetooth adapter
name that does not exist: 22 threads after 6 retries before the change, 1 thread after 10 retries with it. Not
yet checked with the band itself. A bridge started before the change should be restarted after it has sat
disconnected for a while.

Disk use (2026-11-20, after the system disk ran full). Two things grew without bound: (1) the SDK wrote a log file
of about 1 MB per connection attempt to `~/Documents/sensorsdklog`, so a bridge left open with the band off wrote
about 485 files and 0.47 GB an hour; the bridge now turns the SDK's log files off (`--sdk-log` keeps them). Checked
only against the SDK's own switch (`setLogPath(enabled=False)`), not yet with the band. (2) every simulator that was
killed or crashed left OmniGibson's temp folder (scene file and prebuilt USD, about 860 MB) in `/tmp`; `launch_og.py`
and `launch_hand_bench.py` now remove the folders of simulators that are gone when they start, and say how much
they freed. The bridge's own session log (`~/dexemg_teleop_logs/<start time>/`) is written when the bridge stops,
so a running bridge has no folder yet and a killed one never gets one. The bridge now also ends the SDK's helper
processes before it writes that log: two of them outlived a bridge on 2026-11-20 and kept port 8765.

## Testing without hardware

```bash
D=~/dexemg/finetune_data/aligned_two_band
python dex_teleop/scripts/dexemg_bridge.py --checkpoint <ckpt> \
  --replay-npz $D/arat_grasp_all_right_030_OP02_x0_y0_z0.npz \
  --neutral-npz $D/neutral_dual_010_OP02_x0_y0_z0.npz --realtime --loop
```

`tests/test_dexemg.py` and `tests/test_vive_wrist.py` cover the protocol client, the pre-flight, the
preset, option validation, retargeting of decoded landmarks, and the VIVE wrist's equality with the MANUS
composition.

## Troubleshooting

- `No dexemg bridge at 127.0.0.1:8765`: start the bridge first (step 3).
- `The dexemg bridge is 'waiting_for_neutral'` / `'calibrating'`: finish the neutral (Enter in the bridge).
- Window stays on `BAND NOT CONNECTED - searching`: the band is off, out of range, flat, or connected to
  HandSync. The bridge keeps retrying on every adapter; nothing needs restarting once the band is on.
- Bridge: `EMG firmware filter verification failed`: the band did not accept a filter; reconnect.
- `VIVE wrist stale ... tracker FA5AG3B00121 is syncing/untrackable`: uncover the cameras, wake the
  tracker, rebuild the map if needed (MANUS runbook, section 3).
- Fingers look systematically too open or closed after re-wearing the band: re-record the neutral.
- Left hand: not supported (the decoder is trained on the right hand).
