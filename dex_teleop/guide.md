# `launch_og.py` input-source guide

This guide describes what the current `dex_teleop/scripts/launch_og.py` implementation does with each
tracking input. It covers control selection, comparison-only sources, synchronization, retargeting,
OmniGibson execution, recording, resets, and failure behavior.

This is a guide to the code that exists today, not a list of planned providers. Two scope limits are
especially important:

- `quest` currently means the existing Quest Hand Tracking Streamer (HTS) socket receiver. It is not a
  native OpenXR or OmniGibson OVXR source.
- `launch_og.py` currently executes only a right-hand Sharpa command on the Franka + Sharpa OmniGibson
  robot. Some lower-level retargeting classes support other sides and hand models, but the launcher rejects
  them before starting simulation.

## 0. Launch presets (`--config`)

Every option below can be set on the command line, but the normal way to run is a YAML preset in
`dex_teleop/configs/`. Keys are the option names with the leading dashes removed (`-` and `_` are
interchangeable, so `manus-core-host` and `manus_core_host` both address `--manus-core-host`).

| Preset | Selects | Purpose |
| --- | --- | --- |
| `manus_core.yaml` (**default**) | `--hand-source manus --wrist-source manus --manus-mode remote` | MANUS gloves + VIVE Tracker Ultimate through MANUS Core 3 on Windows. |
| `hts.yaml` | `--hand-source quest --wrist-source quest` | Quest Hand Tracking Streamer for fingers and wrist. |

Rules:

- With no `--config`, `manus_core.yaml` is loaded. `--config hts` selects the other shipped preset;
  `--config /path/to/file.yaml` loads any other file; `--no-config` starts from the built-in defaults.
- The preset is applied as *defaults*: any flag given on the command line overrides it. For repeatable
  options (`--record-hand-source`) a command-line value replaces the preset list.
- `null` keeps the built-in default. Booleans take `true`/`false`; other values follow the same `type`
  and `choices` as the command line, and an unknown key, wrong choice, or wrong type fails before Kit
  starts, naming the file and key.
- `--view-only` and `--list-tasks` never apply a preset: presets describe teleoperation runs, and those
  modes construct no tracking source or recording. Explicit flags still work there.
- The launcher prints the active preset path at startup.

`manus_core.yaml` ships with the rig-specific fields (`manus_core_host`, `manus_expected_tracker_id`,
`manus_expected_tracker_user_id`) set to `null`; the launcher refuses the MANUS wrist role until the two
tracker-policy fields are filled in. `manus_core_calibration` may stay `null` (built-in default, see
section 6). The loader is `src/dex_teleop/omnigibson/launch_config.py`.

## 1. The four concepts that should not be conflated

The launcher makes four independent choices:

| Choice | Relevant flag | Meaning |
| --- | --- | --- |
| Selected articulation | `--hand-source` | The joint skeleton that drives the robot fingers. |
| Selected wrist | `--wrist-source` | The tracked anatomical wrist pose that drives the Franka end effector. |
| Record-only inputs | `--record-hand-source`, `--record-wrist-source` | Additional native streams saved for comparison. They never control the robot. |
| Retargeter | `--retargeter` | The one backend that maps selected articulation to Sharpa joint angles. |

In this guide, **articulation** means wrist-local human hand joints. It does not include a world or tracking
frame wrist pose. A **wrist pose** is the position and orientation of the anatomical wrist in a named
reference frame. The worker combines exactly one selected articulation source and one selected wrist
source into a fused observation.

The high-level flow is:

```text
selected articulation source ----+
                                  +--> timestamp/frame fusion --> retargeter --> 22 Sharpa finger targets
selected wrist source ------------+                                  |
                                                                     +--> OmniGibson action adapter
                                                                          - wrist -> Franka arm target
                                                                          - articulation -> finger target

record-only articulation/wrist sources --> native-rate HDF5 streams only
```

There is no automatic source fallback. If MANUS is selected and fails, Quest does not take over. If the
MANUS wrist becomes stale, the Quest wrist does not take over. Record-only sources remain passive.

## 2. Source names and capabilities

| Name (CLI/YAML = registry key) | Articulation | Wrist | Physical integration | Important requirement |
| --- | --- | --- | --- | --- |
| `quest` | Yes | Yes | Quest Hand Tracking Streamer over UDP or TCP | The sender must provide a matched wrist + landmark pair. |
| `manus` | Yes | Remote only | Mode-specific MANUS SDK C++ sidecar | Remote uses Windows Core for glove + Ultimate; Integrated remains gloves-only. |

The VIVE Tracker Ultimate is not a separate source: its pose reaches the launcher only through MANUS
Core on Windows (Remote mode). The former lighthouse VIVE provider (`--wrist-source vive`,
`--vive-calibration`, `pysurvive`), the legacy `--source hts|ovxr` preset flag, and the `hts` alias of
`quest` have been removed; argparse rejects them. A MANUS wrist role requires Remote mode, `tracker` hand
motion, Core-world geometric calibration, and explicit expected Ultimate ID and Core user ID runtime
policy.

### Defaults

- With no `--config` and no source flags, the `manus_core` preset selects MANUS for both components.
- With `--no-config` and no source flags, both components fall back to the built-in Quest default;
  `--hand-source quest --wrist-source quest` (or `--config hts`) is the explicit spelling.
- If only `--hand-source manus` is provided (with `--no-config`), the wrist independently defaults to
  Quest.
- Repeated record-only flags are deduplicated: `--record-hand-source quest --record-hand-source quest`
  creates one comparison stream.

## 3. Complete control-source matrix

These are the canonical control combinations that `launch_og.py` can construct:

| Selected articulation | Selected wrist | What drives fingers | What drives the arm | Physical sources |
| --- | --- | --- | --- | --- |
| Quest | Quest | Quest MP21 landmarks | Quest wrist pose | One shared HTS receiver |
| Quest | MANUS | Quest MP21 landmarks | Calibrated Core raw-skeleton wrist | HTS + one Remote MANUS sidecar |
| MANUS | Quest | MANUS glove skeleton | Quest wrist pose | MANUS sidecar + HTS |
| MANUS | MANUS | MANUS glove skeleton | Calibrated tracker-driven Core wrist | One shared Remote MANUS sidecar |

Quest/Quest and Remote MANUS/MANUS each capture their two components as one
matched frame. They use strict frame/time identity and bypass skew search. The
mixed combinations have independent clocks and transports, so they pass
through timestamp-aware fusion.

### Physical-source deduplication

The source registry creates at most one object for each canonical provider. If Quest serves selected
articulation, selected wrist, and one or more comparison roles, the worker:

1. opens one socket,
2. starts one receive thread,
3. atomically drains each matched wrist/landmark pair once,
4. routes the two components to their configured roles, and
5. health-checks and closes the receiver once.

Remote MANUS is deduplicated the same way. One raw-skeleton callback is
atomically drained as articulation plus wrist even if only one component is
selected and the other is record-only.

Start failure rolls back already-started sources. Normal shutdown closes sources in reverse start order.

## 4. What happens at launcher startup

For a normal teleoperation run, the launcher performs these steps:

1. Parse and canonicalize source names.
2. Reject incompatible flags and the current non-right/non-Sharpa execution modes.
3. Launch OmniGibson and construct every selected or record-only physical source.
4. Construct one retargeter: `adaptive` or `dexpilot`.
5. Construct `MultiSourceTrackingWorker` and start each physical source.
6. For each task, create the OmniGibson environment and its staging HDF5 recording.
7. Reset temporal and retargeter state at the task boundary and let the scene load; the devices may
   still be starting.
8. On SPACE: wait for source readiness, attach the multi-source recording session (if hand recording or
   EMG is active), anchor, and engage. Nothing is recorded before that first SPACE.
9. Fuse, retarget, and publish snapshots asynchronously while OmniGibson steps at 30 Hz.

The worker persists across activities in a subscale, so physical devices start once. A new recording
collector is attached for each task, and source row numbering restarts in the new task file.

## 5. Quest / HTS in detail

### What HTS must send

Each usable HTS frame contains both:

- a wrist position and quaternion, and
- 21 MediaPipe landmarks.

Both records must carry the same source frame ID `f` and source monotonic-nanosecond timestamp `t`. This
remains true even when Quest is configured only as an articulation role or only as a wrist role: the
receiver publishes a component only after assembling the complete matched pair.

Transport behavior differs slightly:

- For UDP, every represented frame's wrist + landmarks pair must be complete in one datagram. One datagram
  may contain multiple complete frame pairs.
- TCP permits the newline-delimited records to arrive in separate reads, but a mismatch, stalled partial
  pair, or connection close during a pair fails the source.

The desktop listener defaults to UDP on `0.0.0.0:9000` and is configured with:

```text
--host HOST
--port PORT
--protocol {udp,tcp}
```

`--host` is the local bind address, not the Quest headset address. `0.0.0.0` listens on all desktop
interfaces.

### Conversion and output

HTS arrives in Unity's left-handed coordinates. The source converts it to the project's right-handed
tracking convention and then exposes two rich components:

- articulation: `schema="mediapipe21"`, 21 wrist-local positions, no invented joint orientations;
- wrist: tracking-frame position and quaternion;
- anatomical basis for both: `openxr_anatomical_wrist`.

The unchanged legacy `HandFrame` is also reconstructed for compatibility. Quest/Quest rich fusion and the
legacy HTS frame are tested to produce the same world landmarks.

### Timing

HTS source time is mapped into host `CLOCK_MONOTONIC` with a per-hand affine clock estimator. The estimator
updates for drift, never estimates capture after receipt, and keeps the mapped stream strictly increasing.
An already-completed duplicate or backward source timestamp is ignored instead of being republished.
Capture and receipt timestamps are both retained.

### Failure behavior

The source fails rather than silently accepting:

- inconsistent frame IDs or source timestamps,
- malformed wrist or landmark values,
- incomplete UDP pairs,
- an unhealthy TCP stream,
- more than 256 unread paired frames after lossless draining has been enabled.

An all-coincident landmark skeleton is ignored as degenerate input.

## 6. MANUS Remote and Integrated in detail

### Explicit SDK modes

`--manus-mode integrated` is the built-in default (used with `--no-config`)
and starts the gloves-only fallback with `CoreSdk_InitializeIntegrated`.
Integrated access is exclusive, so MANUS Core or another Integrated client
must not own the glove.

`--manus-mode remote` (selected by the default `manus_core` preset) starts the production client with
`CoreSdk_InitializeCore`. MANUS Core 3 on Windows owns the glove and receives
VIVE Tracker Ultimate poses through VIVE Hub/SteamVR. The bridge discovers
Core through `CoreSdk_LookForHosts`; `--manus-core-host` selects an exact IP,
exact name, or zero-based index after deterministic sorting. It never assumes
a network port. Remote checks SDK/Core compatibility and the `sdk` license
feature, while Integrated checks the `integrated` feature.

In `launch_og.py`, `--manus-sdk-root` defaults to `~/Desktop/emg/manus`.
Discovery then considers the explicit root, `MANUS_SDK_ROOT`,
`~/manus_setup`, and supported package subdirectories. The two default bridge
paths are:

```text
~/.cache/dex_teleop/manus_bridge/manus_bridge
~/.cache/dex_teleop/manus_bridge/manus_bridge_remote
```

The build selects `libManusSDK_Integrated.so` or `libManusSDK.so` explicitly
and gives each variant a private RPATH directory. Remote refuses to build when
the vendor library's gRPC, Protobuf, ZeroMQ, or zlib SONAMEs are unresolved.
The proprietary libraries remain external and are not copied into recordings.

Relevant options are:

```text
--manus-mode {integrated,remote}
--manus-core-host IP_OR_NAME_OR_INDEX
--manus-loopback-only
--[no-]manus-tracker-diagnostics
--manus-tracker-stale-timeout SECONDS
--manus-expected-tracker-id ID
--manus-expected-tracker-user-id INTEGER
--manus-sdk-root PATH
--manus-bridge PATH
--manus-calibration PATH_TO_MCAL
--manus-core-calibration PATH_TO_JSON
--manus-settings-dir PATH
--manus-log-dir PATH
--manus-startup-timeout SECONDS
--manus-connect-timeout SECONDS
--manus-glove-timeout SECONDS
--manus-reconnect-timeout SECONDS
--manus-discovery-wait SECONDS
```

The raw-skeleton hand motion requested from Core is derived, not configured:
`tracker` whenever MANUS supplies a wrist (selected or record-only), so Core
cannot fall back away from the Ultimate, and the SDK default `auto` otherwise,
where the root is discarded anyway because every joint is re-expressed
wrist-locally. `--manus-calibration` is the vendor glove `.mcal`. A MANUS wrist
role requires Remote mode and the explicit tracker ID/Core-user policy flags;
`--manus-core-calibration` is optional (see below).

Before launch, Python probes `--protocol-version`. Protocol v2 receives native
required-hand and generation-safety flags. A genuine protocol-v1 Integrated
bridge omits v2-only flags and retains Python required-hand validation; v1 is
never accepted for Remote. Recording metadata and per-row provenance contain
the negotiated version and capabilities, separately from the maximum protocol
version supported by this package.

### One callback, two components

The sidecar preserves the raw skeleton's SDK-global wrist before localization.
Python validates the topology and emits:

- articulation: `openxr25_no_palm`, 25 named wrist-local positions and
  orientations with no fabricated palm;
- wrist: the same callback's baked global raw-skeleton wrist, optionally
  transformed from Core world into the configured teleoperation reference;
- shared identity: mode-stable source ID, sequence/frame ID, exact callback
  monotonic nanoseconds, handedness, and receipt time;
- provenance: glove ID, publish time, host and SDK/Core versions, hand-motion
  setting, world frame, and calibration hash.

When MANUS supplies both selected components, the worker verifies exact
identity and capture-time equality, then fuses them directly with zero skew.
It does not apply the heuristic used for independent devices. Optional tracker
stream diagnostics preserve Core tracker IDs, role, system, quality, pose, and
publish/callback timing for hardware acceptance.

The 25 joints include non-thumb metacarpals. Current retargeters map the
skeleton to the 21 MediaPipe/MANO-equivalent joints, so extra metacarpals and
finger orientations remain available in native-rate recordings.

### Calibration, readiness, and failures

Core-world calibration is source-neutral and separate from the glove `.mcal`.
It contains `reference_T_core_world`, records Core's applied
tracker-offset preset/ID, and applies an optional residual
`skeleton_wrist_T_anatomical_wrist`.

Without `--manus-core-calibration` the wrist role uses the **built-in
default** (`ManusCoreCalibration.default()`): Core's Y-up world is rotated into
the launcher's Z-up tracking frame (`x = -z_core`, `y = -x_core`, `z = y_core`,
frame name `dex_teleop_tracking_rh_x_forward_z_up_m`) with no wrist residual.
That is the same treatment the Quest wrist gets: the adapter anchors position
and orientation at engagement, so only axis directions matter, and yaw is
aligned by pointing SteamVR room-setup "forward" at the robot's +X or with
`--tracking-yaw-deg`. Provide a measured file when recorded MANUS wrist poses
must sit in a surveyed robot frame; its schema and the hardware workflow are in
[`docs/manus-remote-vive-ultimate-runbook.md`](docs/manus-remote-vive-ultimate-runbook.md).
Whichever is used is written to the recording metadata (`core_calibration`).

MANUS becomes ready only after a validated frame for the requested hand.
Remote recoverable disconnects clear stale topology/samples, reconnect to the
same selected host, revalidate versions/hand motion/topology, and then resume.
The run fails explicitly on a missing or wrong SDK library, unresolved Remote
dependencies, missing license, ambiguous host selector, version mismatch,
wrong/absent hand, unsupported topology, duplicate/backward sequence, queue
overflow, or failed reconnect.

## 7. VIVE Tracker Ultimate

The Ultimate is not a launcher source of its own. It is paired in VIVE Hub on the Windows host, its pose
reaches MANUS Core through SteamVR, Core bakes the selected mount preset into the raw-skeleton global
wrist, and the Remote sidecar receives that wrist together with the articulation in one callback (section
6). The launcher only validates it: exact Ultimate ID and Core user ID, the `<hand>_hand` role, the
`openvr` tracking system, a non-HMD identity, a valid and advancing pose, and freshness within
`--manus-tracker-stale-timeout`.

The earlier lighthouse provider that read base-station trackers directly through `pysurvive`/libsurvive
(`--wrist-source vive`, `--vive-calibration`, `calibrate_vive.py`) has been removed; those flags are now
rejected by argparse.

## 8. Synchronization and coordinate frames

Selected articulation is the timing reference for control. Only the selected wrist source is admitted to
fusion; record-only wrists never participate.

### Wrist matching rules

The default `--maximum-source-skew` is 0.05 seconds. For an articulation capture timestamp, the fuser tries:

| Priority | Condition | Result |
| --- | --- | --- |
| 1 | Wrist with exactly the same capture timestamp | Use it directly. |
| 2 | Earlier and later wrist samples are each within the skew bound | Linearly interpolate position and SLERP orientation. |
| 3 | No valid bracket, but a nearest wrist is within the bound | Use the nearest; prefer the earlier wrist on an exact-distance tie. |
| 4 | No wrist is within the bound | Publish no observation. |

For a nearest-only match, the worker waits up to `min(maximum skew, 1/60 second)` for a future bracket. At
the default skew this is about 16.7 ms. This internal interpolation wait is not currently a launcher flag.

Setting `--maximum-source-skew 0` requires exact capture timestamps and is
therefore normally useful only for the paired Quest/Quest or Remote
MANUS/MANUS paths. The latter bypasses this setting after exact callback
identity validation. If independent clocks never overlap within the configured
bound, startup or engagement eventually reports that no in-skew fused
observation was available.

### Anatomical basis versus external reference frame

These are separate calibration questions:

- `HandArticulationSample.coordinate_frame` and `WristPoseSample.anatomical_frame` describe local wrist
  axes. They must match before joint vectors can be rotated by the wrist quaternion.
- `WristPoseSample.reference_frame` describes the external frame containing wrist position and orientation.
  Remote MANUS uses its Core-world calibration to place the pose in the intended
  Quest/world/teleoperation frame.

All built-in sources declare `openxr_anatomical_wrist`. A future provider with different local axes must use
`--articulation-wrist-calibration`. That JSON transform maps the provider's local articulation basis to the
selected wrist's anatomical basis. It does not replace the MANUS Core-world extrinsic calibration and
does not tune human-hand scale.

Frame name changes, schema changes, backward capture time, or backward receipt time during a run are fatal.
A fuser reset clears its buffered frame state, but the worker still requires each provider's schema and
frame identity to remain stable for the complete launcher process.

### Mixed-source clock accuracy

The implementation puts all capture times into host monotonic seconds, but every mixed pair (MANUS +
Quest in either role) is still software-synchronized rather than hardware-triggered. Network, USB,
SDK callback, and polling latency can bias the estimated alignment. Recordings retain capture and receipt
timestamps so that this bias can be measured rather than hidden.

## 9. Retargeting after fusion

Fusion always constructs a compatibility world-frame `HandFrame` for the action adapter and legacy
recording. The worker then selects the retargeter API:

- if the backend implements `retarget_observation`, it receives the rich fused observation;
- otherwise, it receives the compatibility 21-landmark `HandFrame`.

Only one retargeter runs in a launch. A record-only articulation source is not passed through a second
shadow retargeter.

### Retargeter comparison

| Property | `adaptive` | `dexpilot` |
| --- | --- | --- |
| Default | Yes | No |
| Input boundary | Fused world-frame MP21 `HandFrame` | Rich observation, then named MP21-equivalent local positions |
| Current objective | In-tree adaptive full-hand/tip position-direction vectors | `dex-retargeting` DexPilot relative vectors |
| MANUS finger orientations | Recorded but unused | Recorded but unused |
| Sharpa objective input | 21-position input → 16 unique wrist/PIP-or-MCP/DIP-or-IP/tip points | 21-position input → 6 unique wrist/tip points |
| Filtering | In-tree low-pass filter | Dex-retargeting low-pass filter (`alpha=0.2`) |
| Dependencies | Direct project dependencies | Optional exact `dex-retargeting==0.4.6` plus PyTorch |
| Hand models at library level | Shadow, Sharpa, Wuji | Sharpa only |
| Launcher execution | Right Sharpa only | Right Sharpa only |

The adaptive Sharpa configuration was originally tuned for Quest 3 HTS landmarks. MANUS is accepted after
the named 25-to-21 mapping, but MANUS-specific scale, segment, and pinch behavior should be evaluated with
recorded data rather than assumed identical. `--articulation-wrist-calibration` fixes coordinate basis, not
these retargeting weights.

DexPilot bypasses the legacy world-frame round trip: it uses the selected articulation's local named
positions directly. It requires all 21 MediaPipe-equivalent joints to be valid. The packaged Sharpa config
then constructs its vector objective from the wrist and five fingertips; it does not optimize against all
MANUS joint orientations.

Install the optional backend before selecting it:

```bash
conda run -n behavior_dex python -m pip install -e './dex_teleop[dexpilot]'
```

If construction fails, the launcher reports the failure. It never falls back to `adaptive`. Pink/V2D is
not an implemented backend in this repository.

## 10. How the fused result becomes an OmniGibson action

Both current backends output a named 22-joint right Sharpa command. `SharpaActionAdapter` combines two
independent parts:

1. The selected wrist pose becomes the Franka end-effector position and orientation target.
2. The retargeted selected articulation becomes 22 Sharpa finger targets in explicit joint-name order.

The resulting action has 28 elements: 3 position, 3 axis-angle orientation, and 22 finger joints.

On the first fresh frame after engagement, the adapter anchors the tracked wrist to the robot's current end
effector. Therefore the tracking system does not need to report coordinates already centered on the Franka.
The adapter then applies:

- `--tracking-yaw-deg` about world Z,
- `--position-sensitivity` to wrist translation,
- workspace and maximum-step limits,
- a wrist-orientation jump/slew gate, and
- the final arm/finger rate and smoothing safety filter.

An implausible wrist-orientation flip may be held by the orientation gate even though the source itself is
fresh and healthy. This is downstream safety behavior, not source switching.

## 11. Engagement, reset, readiness, and staleness

### Readiness means all requested streams

`wait_for_first` requires both:

1. one valid in-skew fused observation from the selected articulation and wrist, and
2. at least one sample from every explicitly configured record-only modality.

This prevents a MANUS/Quest comparison episode from starting with an empty comparison stream. It also means
a missing record-only source can block engagement even when the selected control pair is healthy.

### SPACE starts engagement and recording

The scene may load before tracking begins; nothing is controlled or recorded until the operator presses
SPACE. The first SPACE of a task:

1. drains pre-boundary native samples,
2. clears fusion and pending control state,
3. resets the retargeter filter and warm start,
4. waits for new data from every configured role,
5. attaches the native-rate recording session and begins `demo_0` (when hand recording or EMG is active),
6. resets the adapter's wrist anchor and safety state, and
7. engages on the next fresh frame.

Pressing SPACE while already engaged performs the same re-anchor boundary without touching the recording.
There is no automatic engagement: the former `--auto-anchor` option was removed so that a run never starts
recording without a deliberate key press. Hold your hand roughly in the robot's start posture when you
press SPACE; orientation is transferred relative to that anchor for both Quest and MANUS.

### On-screen ARAT status

With scoring on (the default), the main viewport shows, under the BDDL goal list, a yellow status line
`ARAT <activity>: t=12.3s  provisional 2/3 (exceeded_5s)` updated every engaged step from the scorer's
`evaluation_breakdown`, plus the last three scorer events (`t=1.2s voluntary_opening   t=3.4s
grasp_established …`). Before SPACE it reads `press SPACE to start (60 s engaged-time limit)`, an R reset
returns it to that state, and when the item finishes it turns green with the final `FINAL 2/3, completed at
t=7.8s (reasons)` line. The same information is printed to the console and written to the results JSON.

### R reset

Pressing R disengages and establishes a new tracking boundary. Only after all required streams produce
fresh data does the launcher reset OmniGibson and begin a new action-alignment episode (`demo_N+1`); the
next SPACE re-engages without restarting the recording. Native streams remain continuous within the task
HDF5 file once started, while action alignment is split into `demo_N` groups. R before the first SPACE
resets the scene only.

### Staleness

The default `--maximum-frame-age` is 0.25 seconds and uses receipt time, not capture time.

- Without native hand/EMG recording, `--stale-frame-policy error` raises on stale selected input.
- Without native hand/EMG recording, `--stale-frame-policy hold` keeps the robot still until selected input
  becomes fresh again.
- With `--record-hand-poses` or `--emg`, every selected and record-only stream is checked strictly on every
  simulator iteration from the first SPACE onward, including while disengaged after an R reset. A stale
  comparison stream invalidates the run instead of silently creating a gap, so `hold` does not relax
  recording integrity. Before the first SPACE nothing is recorded, so nothing is enforced.

The default `--initial-frame-timeout` is 10 seconds for readiness and post-anchor fresh-frame waits. Source
startup has additional provider-specific deadlines described above.

## 12. Record-only sources and HDF5 output

Every teleoperation task creates a base OmniGibson trajectory HDF5 file, even without hand-pose flags. The
default destination is `dex_teleop/outputs/recordings/<activity>.hdf5`. `--recording-path` overrides it only
for a single explicit `--task`; it is rejected for a subscale or `--view-only` run. Hand/EMG recording
augments this same task file rather than creating a separate tracking file.

`--record-hand-source` and `--record-wrist-source` are repeatable. They require either
`--record-hand-poses` or `--emg`; either mode creates the native tracking recorder. A redundant record-only
source equal to the selected source is removed because the selected native stream is already recorded.

Record-only data has three guarantees:

- it never changes the selected source;
- it never gets retargeted into an additional command; and
- it must remain healthy and fresh for a strict comparison recording.

### Cadence

Each acquisition source feeds a bounded, loss-detecting queue. The worker has a nominal 2 ms polling
interval, although source and optimizer work can extend an iteration. It records every accepted native
sample, so slow retargeting does not silently overwrite raw MANUS or Quest comparison samples.

Control and recording cadences are deliberately different:

- native articulation and wrist rows retain device acquisition cadence;
- retargeting uses selected articulation at processing cadence and may skip intermediate candidates under
  load;
- OmniGibson reads the latest snapshot at 30 Hz and may reuse one retargeting row for several actions; and
- action-alignment rows store those repeated references explicitly.

Queue overflow is fatal because silently dropping comparison frames would make latency analysis misleading.

Native articulation, native wrist, and retargeting rows are task-session-wide. They may include samples
captured during readiness waits, while disengaged, between simulator actions, and across R resets. Their
row numbers remain global across all `demo_N` groups in one task HDF5 file and restart only for a new task
recording session. `/human_hand_pose/demo_N` and `/hand_tracking/action_alignment/demo_N` identify the
action-used subset.

### HDF5 layout

The base OmniGibson trajectory remains under `/data`. With hand recording or EMG enabled, the same file also
contains:

```text
/data/demo_N/action
    actual filtered 28-D action passed to env.step

/human_hand_pose/demo_N
    selected, fused, action-aligned MP21 compatibility pose

/hand_tracking/articulation_streams/articulation.quest
/hand_tracking/articulation_streams/articulation.manus
    native articulation streams that were enabled

/hand_tracking/wrist_streams/wrist.quest
/hand_tracking/wrist_streams/wrist.manus
    native wrist streams that were enabled

/hand_tracking/wrist_streams/wrist.<selected>.control
    present when interpolation produced a derived wrist with no exact native row

/hand_tracking/retargeting_streams/<backend>.<articulation_source>+<wrist_source>
    selected backend output, for example adaptive.manus+quest

/hand_tracking/action_alignment/demo_N
    exact source/output stream IDs, row numbers, and synchronization skew used by each simulator action
```

Articulation rows store joint names, schema, coordinate frame, positions, optional orientations,
orientation-availability masks, validity, confidence, capture/receipt time, source time/frame ID, and JSON
provenance. Wrist rows store their two frame names, position, quaternion, and the same timing/provenance
fields.

Each stream also has reproducibility metadata. Depending on the provider, it includes the Quest endpoint;
resolved MANUS SDK and bridge protocol plus bridge/glove/Core-world calibration fingerprints; the
derived-wrist fusion settings and articulation-frame transform fingerprint; or
the retargeter backend, hand model, side, and configuration fingerprint. MANUS calibration bytes are not
stored, but the calibration basename and SHA fingerprint are. These attributes exist when native hand
recording or EMG has enabled `/hand_tracking`.

If an exact or nearest native wrist is used, the retargeting row references `wrist.<key>` directly. An
interpolated wrist is stored under `wrist.<selected>.control` with its supporting before/after frame IDs and
timestamps in provenance.

Each action-alignment row includes `synchronization_skew_seconds`. It is zero for an exact wrist match, the
absolute capture-time difference for a nearest match, or the larger distance from articulation time to
either supporting wrist sample for an interpolated match.

`/human_hand_pose` is intentionally narrower. It contains only the selected fused observation at action
rate for compatibility with existing analysis. Raw HTS Unity arrays are present there only when the same
Quest receiver supplies both selected articulation and selected wrist. For MANUS + Quest control,
`raw_available` is false in `/human_hand_pose`; the full canonical Quest comparison remains under
`/hand_tracking` when requested.

The retargeting stream stores the pre-adapter 22-finger command. `/data/demo_N/action` stores the actual
post-anchor, post-gating, post-safety 28-D command sent to OmniGibson.

## 13. Recommended command recipes

Replace the example task, calibration files, and recording path as needed. Each recipe starts from a
preset (section 0) and adds only the flags that differ from it.

### MANUS gloves + VIVE Tracker Ultimate through Windows Core (default preset)

Fill in `manus_core_host`, `manus_expected_tracker_id`, and `manus_expected_tracker_user_id` in
`dex_teleop/configs/manus_core.yaml` once, then:

```bash
conda run -n behavior_dex python dex_teleop/scripts/launch_og.py \
  --task arat_grasp_block_10cm
```

One Remote MANUS sidecar serves both selected components; the preset enables `--record-hand-poses`, and
the built-in Core-world default is used unless `manus_core_calibration` names a measured file. The same
values can be given per run instead (`--manus-core-host 192.0.2.10 --manus-expected-tracker-id <id>
--manus-expected-tracker-user-id 1`). Press SPACE to engage and start recording. Hardware acceptance is
described in the runbook (section 6).

### Quest/HTS path

```bash
conda run -n behavior_dex python dex_teleop/scripts/launch_og.py \
  --config hts \
  --task arat_grasp_block_10cm
```

This opens one HTS receiver, uses the paired Quest skeleton and wrist, and runs the default adaptive
retargeter. It is the explicit spelling of `--no-config --hand-source quest --wrist-source quest`.

### MANUS fingers (Integrated, gloves only) + Quest wrist, adaptive retargeting

```bash
conda run -n behavior_dex python dex_teleop/scripts/launch_og.py \
  --config hts \
  --task arat_grasp_block_10cm \
  --hand-source manus \
  --manus-calibration /path/to/right-hand.mcal
```

Starting from the HTS preset keeps `--manus-mode integrated` and the Quest wrist; only the articulation
source is overridden. The Quest landmarks are received because HTS must assemble complete pairs, but they
are not retained as a comparison stream unless `--record-hand-source quest` is added.

### MANUS control + Quest articulation comparison + DexPilot

```bash
conda run -n behavior_dex python dex_teleop/scripts/launch_og.py \
  --config hts \
  --task arat_grasp_block_10cm \
  --hand-source manus \
  --record-hand-source quest \
  --retargeter dexpilot \
  --manus-calibration /path/to/right-hand.mcal
```

MANUS drives the fingers, the Quest wrist drives the arm, and native Quest articulation is saved for
comparison. The Quest comparison is not also retargeted online. This command requires the optional
DexPilot installation.

### Windows Core MANUS control with a complete Quest comparison

```bash
conda run -n behavior_dex python dex_teleop/scripts/launch_og.py \
  --task arat_grasp_block_10cm \
  --record-hand-source quest \
  --record-wrist-source quest
```

The default preset supplies both MANUS roles; one shared HTS receiver records both Quest components.

### Quest control with passive MANUS comparison

```bash
conda run -n behavior_dex python dex_teleop/scripts/launch_og.py \
  --config hts \
  --task arat_grasp_block_10cm \
  --record-hand-source manus \
  --manus-calibration /path/to/right-hand.mcal
```

This preserves Quest control while collecting Integrated MANUS articulation. The run waits for and
continuously freshness-checks both devices.

## 14. Designing comparison experiments

### Compare MANUS and Quest articulation with the same wrist

Use MANUS as selected articulation, Quest as selected wrist, and Quest as record-only articulation:

```text
--hand-source manus --wrist-source quest --record-hand-source quest --record-hand-poses
```

This keeps the arm/wrist source constant while comparing glove and optical skeletons. Only MANUS is
retargeted online. To compare resulting robot commands, retarget the saved Quest stream offline or run a
second trial with Quest selected; the launcher does not currently compute shadow retargeter outputs.

### Compare Quest and Ultimate wrist tracking

Select one wrist for control and record the other. With the default MANUS preset:

```text
--record-wrist-source quest --record-hand-source quest
```

Only the Core wrist controls the arm. The recording preserves both native wrist trajectories and the exact
selected wrist row used for each action; MANUS/MANUS pairs share one callback, so the control row is always a
native row. The mirror image (`--config hts --record-wrist-source manus --manus-mode remote` plus the
Core-world calibration and tracker policy flags) keeps Quest in control and records the Core wrist
comparison-only; for a mixed pair the control row may be a derived interpolated row with the supporting
native timestamps in provenance. Either way, offline alignment by capture or receipt time is possible.

## 15. Invalid or misleading combinations

The results below describe a teleoperation run. `--view-only` does not construct a robot, retargeter, or
tracking source, so source/model selections are not exercised in that mode.

| Combination | Result |
| --- | --- |
| `--source …`, `--hand-source hts`, `--wrist-source vive`, `--vive-calibration`, `--wrist-flip-*` | Rejected by argparse; the legacy preset flag, the `hts` alias, the lighthouse VIVE provider, and the wrist-flip diagnostics were removed. |
| `--wrist-source manus` in Integrated mode | Rejected; Integrated remains gloves-only. |
| MANUS wrist with Auto/IMU/none/rotation-only motion | Rejected; use Remote `tracker` so Core cannot silently fall back away from the Ultimate. |
| MANUS bridge/calibration/settings/log flags when MANUS is absent | Rejected as unused/misleading. |
| Unknown key, wrong choice, or wrong type in a `--config` file | Rejected before Kit starts, naming the file and key. |
| Record-only flags without `--record-hand-poses` or `--emg` | Rejected because no native collector exists. |
| `--retargeter dexpilot` without its optional dependency | Explicit construction error; no fallback. |
| `--hand left` or a non-Sharpa `--hand-model` | Teleoperation is rejected by the current OmniGibson execution adapter. |

`--view-only` is for inspecting the saved scene, not validating a tracking-source configuration. For
example, it can accept a left hand or a non-Sharpa model because neither is constructed.

## 16. Troubleshooting by symptom

### “HTS listener did not bind”

- Check that no other process owns the configured UDP/TCP port.
- Check `--host`, `--port`, and `--protocol` against the sender.
- Do not instantiate two launchers on the same endpoint.

### HTS binds, but no hand observation appears

- Ensure the sender includes both wrist and landmarks records.
- Enable frame ID `f` and source timestamp `t` metadata.
- For UDP, send the complete pair in one datagram.
- Confirm the selected `--hand right` stream is being sent.

### MANUS cannot start

- Confirm the SDK root contains a matching header and the selected mode's
  `libManusSDK_Integrated.so` or `libManusSDK.so`.
- Confirm a C++17 compiler is available for automatic bridge building.
- For Remote, run `ldd` on `libManusSDK.so` and install the exact missing
  vendor gRPC/Protobuf/ZeroMQ/zlib dependencies.
- Confirm the selected `integrated` or `sdk` license feature is active.
- For Integrated, stop MANUS Core and other Integrated clients. For Remote,
  confirm the intended Core appears in discovery and the selector is unique.
- Confirm SDK/Core versions are compatible in Remote mode.
- Confirm the right glove is connected and publishing the validated topology.
- Check that the selected `.mcal` and Core-world calibration files exist.

### “A MANUS wrist source requires --manus-expected-tracker-id” right after “Launch config: …manus_core.yaml”

The default preset selects the Windows Core path but ships with the rig-specific keys unset. Fill in
`manus_expected_tracker_id` and `manus_expected_tracker_user_id` (and usually `manus_core_host`) in
`dex_teleop/configs/manus_core.yaml`, pass them as flags, or switch presets with `--config hts`.

### The arm moves in the wrong direction on the MANUS path

The built-in Core-world default only converts axes (Core Y-up to Z-up); yaw comes from SteamVR room setup.
Re-run room setup facing the robot's +X, or set `tracking_yaw_deg`, or provide a measured
`manus_core_calibration` file. Left/right or up/down inversions mean the SteamVR world is not Y-up /
forward = -Z as assumed; check the tracker pose in SteamVR before blaming the mount preset.

### The Ultimate wrist is rejected although the glove streams

- Run the hardware acceptance diagnostic from the runbook and compare the reported tracker ID, user ID,
  role, and `tracking_system` with `--manus-expected-tracker-id` / `--manus-expected-tracker-user-id`.
- In Core, confirm the Ultimate is assigned to the right hand and tracker motion is enabled for the raw
  skeleton; the launcher always requests `tracker` hand motion for the wrist role.
- Check `--manus-tracker-stale-timeout` against the Ultimate update rate seen in the diagnostic JSONL.

### “No in-skew fused observation”

- Inspect capture and receipt timestamps in a recording or source diagnostics.
- Increase `--maximum-source-skew` only after measuring the actual transport offset.
- Check that HTS and MANUS monotonic time are mapped correctly.
- Check anatomical frame names and any `--articulation-wrist-calibration`.
- Remember that the runtime does not try a record-only wrist as fallback.

### A comparison run stops when an unused device pauses

This is intentional. Explicit record-only streams are loss-detecting and freshness-enforced. Remove the
record-only flag if the device is not required for that trial, or restore its stream. `--stale-frame-policy
hold` does not permit gaps in native comparison recordings.

### “Acquisition overflowed its queue”

The consumer failed to drain fast enough or an unused calibrated hand accumulated samples. The run stops
instead of producing a silently incomplete comparison. Check CPU load, optimizer latency, and duplicate
device roles.

### DexPilot is unavailable

Install the exact optional extra shown above. The launcher deliberately does not substitute the adaptive
backend because that would change the experiment without consent.

## 17. Current non-features

The following are not implied by the multi-source CLI:

- `quest` is not a native OpenXR/OVXR provider;
- there is no separate custom Linux SteamVR/OpenVR Ultimate sidecar; the
  production Ultimate path is Windows SteamVR → MANUS Core → Remote SDK;
- automatic Core-world registration/capture is not implemented; the runtime
  consumes and validates a measured calibration JSON;
- record-only sources do not control the robot and are not shadow-retargeted;
- two retargeters cannot run simultaneously from `launch_og.py`;
- Pink/V2D retargeting is not implemented;
- live source switching is not implemented;
- automatic source fallback is not implemented; and
- the current OmniGibson action sink is not bimanual and does not execute non-Sharpa hands.

These constraints are intentional boundaries: acquisition, fusion, retargeting, and execution can be
extended independently without changing the recorded named-source and named-joint contracts.

## 18. Implementation map

The main implementation points behind this guide are:

| Concern | File |
| --- | --- |
| CLI validation, source factories, task loop, and recording attachment | `src/dex_teleop/omnigibson/launcher.py` |
| YAML launch presets (`--config`) validated against the parser | `src/dex_teleop/omnigibson/launch_config.py`, `configs/*.yaml` |
| Multi-source acquisition, lifecycle, readiness, and retarget dispatch | `src/dex_teleop/runtime/worker.py` |
| Quest/HTS receiver and clock alignment | `src/dex_teleop/tracking/hts.py` |
| MANUS Remote/Integrated sidecar client | `src/dex_teleop/tracking/manus.py` |
| MANUS Core-world calibration | `src/dex_teleop/tracking/manus_calibration.py` |
| MANUS hardware acceptance diagnostic | `src/dex_teleop/tracking/manus_diagnostics.py` |
| MANUS native SDK bridge | `src/dex_teleop/native/manus_bridge/manus_bridge.cpp` |
| Timestamp matching and wrist interpolation | `src/dex_teleop/tracking/fusion.py` |
| Named OpenXR/MediaPipe conversion | `src/dex_teleop/tracking/openxr.py` |
| Existing adaptive retargeter | `src/dex_teleop/retargeting/retargeter.py` |
| Optional DexPilot backend | `src/dex_teleop/retargeting/dexpilot.py` |
| Multi-source HDF5 schema and writer | `src/dex_teleop/omnigibson/hand_tracking_recording.py` |
| Wrist-to-arm and finger command execution | `src/dex_teleop/omnigibson/sharpa_adapter.py` |

Paths in this table are relative to the `dex_teleop/` directory.
