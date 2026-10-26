# MANUS Core 3.2 (Python SDK) + VIVE Tracker Ultimate on Linux runbook

This is the default `launch_og.py` path (`configs/manus_core.yaml`): MANUS
gloves and a VIVE Tracker Ultimate, both read on the Linux teleoperation host.
Nothing runs on Windows and nothing is compiled:

```text
MANUS gloves ──dongle──▶ manus_sdk (MANUS Core 3.2 Python SDK, Integrated) ─┐
                                                                            ├─▶ dex_teleop.tracking.manus_python_bridge ──JSON lines──▶ ManusSource ──▶ launcher
VIVE Ultimate ──dongle──▶ VIVE Hub for Linux tracker_server ──/tmp/vut.sock─┘        (injects the tracker into Core, composes the wrist,
                                                                                      relays haptics back to the glove)
```

What changed with MANUS Core 3.2 (September 2026):

- The SDK ships a **Python package** (`manus_sdk`, CFFI, Python 3.9-3.14,
  Linux x86_64 and ARM64) and a single `libManusSDK` with gRPC/protobuf
  statically linked. The C++ sidecar and its gRPC runtime build are no longer
  needed (they remain available as `--manus-backend native`).
- **MANUS Core itself is still Windows-only.** On Linux the SDK runs
  *Integrated*: it embeds Core, talks to the glove dongle directly, and needs
  no Core application. The Linux SDK package also contains a **MANUS Core
  Dashboard (Linux build)** that attaches to an Integrated SDK for glove
  management and `.mcal` calibration.
- HTC released **VIVE Hub for Linux** and a **VIVE Ultimate Tracker SDK**
  (beta, September 2026, GitHub `ViveRoboticsSoftware/VIVEHub-Linux` and
  `ViveRoboticsSoftware/vive-ultimate-tracker-sdk`). Pairing, tracking-map
  creation and pose streaming for the Ultimate now work on Linux without
  SteamVR, VIVE Hub for Windows, or a headset.

## Does the VIVE Tracker Ultimate calibration still need Windows?

No. Two different "calibrations" used to require the Windows machine:

| Step | Before (Core 3.1 / Windows path) | Now (Linux path) |
| --- | --- | --- |
| Pair the Ultimate with its dongle | VIVE Hub (Windows) | VIVE Hub for Linux (`run-vivehub.sh`) |
| Build / rebuild the inside-out tracking map ("room setup") | VIVE Hub (Windows) | VIVE Hub for Linux map wizard (x86_64 binary; runs under box64 on arm64) |
| Deliver the pose to MANUS | SteamVR → MANUS Core (Windows) OpenVR tracking system | `tracker_server` → `/tmp/vut.sock` → the Python sidecar → `CoreSdk_SendDataForTrackers` |
| Assign the tracker to the glove's hand and pick the mount offset preset | MANUS Core Dashboard (Windows) | The sidecar assigns it (`CoreSdk_AssignTrackerToUser` / `AssignRoleToTracker`) and sends the mount offset (`CoreSdk_SetTrackerOffset`); the offset values come from the YAML |
| Glove `.mcal` calibration | MANUS Core (Windows) | MANUS Core Dashboard (Linux build) in Integrated mode, or an existing `.mcal` |
| Core world → robot frame | `--manus-core-calibration` JSON or the built-in default | unchanged |

Two caveats:

1. VIVE Hub for Linux is a **beta** that needs a graphical desktop session
   (the map wizard is a GUI) and a dongle with USB access (`install-udev.sh`).
2. MANUS Core's named mount preset ("HTC Vive Ultimate Tracker") lives in the
   Windows Dashboard. On Linux you supply the tracker-to-wrist offset
   yourself (`manus_tracker_to_wrist_*` in the YAML). The default is the
   MANUS Universal Mount's documented example tracker-to-casing offset; if
   you still have a Windows Core, export the preset once (profile settings →
   hand offsets → *Save to XML*) and copy the values.

## 1. Linux host prerequisites

Ubuntu 22.04/24.04 (x86_64 or ARM64), the `behavior_dex` conda environment,
Python 3.9-3.14 (the SDK ships binaries for those), and `cffi`:

```bash
conda activate behavior_dex
pip install "cffi>=1.15"                       # or: pip install -e './dex_teleop[manus]'
sudo apt install libusb-1.0-0 libzmq5 libudev1 zlib1g libgomp1 libcurl4
```

Dongle permissions (once; the SDK's `C++/SDKClient/install-dependencies.sh`
does the same):

```bash
sudo tee /etc/udev/rules.d/70-manus-hid.rules >/dev/null <<'EOF'
# HIDAPI/libusb
SUBSYSTEMS=="usb", ATTRS{idVendor}=="3325", MODE:="0666"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="83fd", MODE:="0666"
# HIDAPI/hidraw
KERNEL=="hidraw*", ATTRS{idVendor}=="3325", MODE:="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger   # then re-plug the MANUS dongle
```

## 2. MANUS Core 3.2.1 SDK (Linux package)

Download `MANUS_Core_3.2.1_SDK_Linux.tar.gz` from the MANUS Knowledge Center
(3.2.1 downloads) and extract it where the launcher looks:

```bash
mkdir -p ~/manus_setup/vendor/ManusSDK_v3.2.1
tar -xzf MANUS_Core_3.2.1_SDK_Linux.tar.gz -C ~/manus_setup/vendor/ManusSDK_v3.2.1
ls ~/manus_setup/vendor/ManusSDK_v3.2.1/Python/manus_sdk/linux/amd64/
#   libManusSDK-amd64.so  _manus_sdk.cpython-311-x86_64-linux-gnu.so ... (3.9-3.14)
```

`configs/manus_core.yaml` points `manus_sdk_root` at that folder; the sidecar
imports `manus_sdk` from `<root>/Python` (no `pip install` needed). When
several releases sit under `~/manus_setup/vendor`, the newest is used unless
the root is explicit. Alternatively `pip install
~/manus_setup/vendor/ManusSDK_v3.2.1/Python` into the environment.

License: Integrated mode requires a MANUS license with the **SDK**,
**SDK (integrated)** and **Linux** features on the dongle. Without one the
sidecar reports `integrated_license_unavailable` as soon as a glove appears
(the SDK logs `No license found for session type 'CoreSDK'`). The 3.2 SDK can
fetch the newest license online (`CoreSdk_RetrieveLicense`, also offered by
the Dashboard's *Retrieve License* button).

Glove check without any tracker (gloves on, dongle plugged in; move the
fingers for 10 s):

```bash
dex-teleop-manus-diagnostics --mode integrated --hand right --duration 10 \
  --sdk-root ~/manus_setup/vendor/ManusSDK_v3.2.1 \
  --glove-calibration ~/manus_setup/calibration/right.mcal \
  --output /tmp/manus_integrated.jsonl
```

`"passed": true` means dongle, glove, license and SDK work. `no_glove`
means the SDK ran but no licensed, paired glove streamed within
`--glove-timeout`.

### Glove calibration on Linux

`MANUS Core Dashboard (Linux build)/MANUS_Core` in the SDK package attaches to
an Integrated SDK session (stop `launch_og.py`/the diagnostic first; only one
process may own the dongle) for pairing, firmware, user assignment and the
guided 8-step Metaglove Pro calibration. Export the result as `.mcal` and set
`manus_calibration` in the YAML, or leave it `null` to use the calibration
the Dashboard stored for the user.

## 3. VIVE Hub for Linux

1. Download `VIVEHub-Linux-<version>-x86_64.tar.gz` (or `-arm64`) from the
   latest release of `ViveRoboticsSoftware/VIVEHub-Linux` (1.0.1 or newer; it
   also ships the VUT C++ SDK under `sdk/`, which `dex_teleop` does **not**
   need: the sidecar speaks the socket protocol in Python).
2. `tar xzf VIVEHub-Linux-*.tar.gz && cd VIVEHub-Linux && sudo ./install-udev.sh`
   then re-plug the Ultimate dongle (`sudo ./install-box64.sh` on arm64).
3. `./run-vivehub.sh` (always through the script). In the UI: pair each
   Ultimate, press *Start setup* and sweep the tracker slowly to build the
   map (stand ~1.2 m high, 1.5 m from obstacles; feature-rich, evenly lit,
   non-reflective walls; include low angles), then confirm every tracker
   shows as *tracking*. Rebuild the map whenever the room changes.
4. Keep VIVE Hub running: its `tracker_server` publishes poses on
   `/tmp/vut.sock`. Check from Python without any other software:

```bash
cd dex_teleop && python - <<'EOF'
import time
from dex_teleop.tracking.vive_ultimate import VutPoseFeed
feed = VutPoseFeed("/tmp/vut.sock"); feed.start(); time.sleep(2.0)
for slot, pose in feed.snapshot().items():
    print(slot, pose.tracker_id, pose.state_name, pose.quality, pose.position.round(3), f"{pose.battery}%")
feed.close()
EOF
```

The printed `tracker_id` is the serial that goes into
`manus_expected_tracker_id` (fallback `vive_tracker_<slot>` when VIVE Hub
reports no serial). Face the robot's `+X` when you build the map so the
built-in Core-world calibration needs no `tracking_yaw_deg`.

Frame convention: VIVE Hub reports OpenXR axes (x right, y up, -z forward),
which are also MANUS Core's world axes, so `vive_ultimate_frame:
x_right_y_up_z_back` (default) injects poses unchanged. HTC's standalone SDK
demo server (`/tmp/vive_tracker.sock`) documents Isaac-Sim axes instead; use
`vive_ultimate_frame: x_forward_z_up` for it. If the hand moves along the
wrong axes in the simulator, this switch is the first thing to check.

## 4. How the sidecar combines them

`dex_teleop.tracking.manus_python_bridge` (protocol `dex_teleop.manus` v3):

- initializes `CoreSdk_InitializeIntegrated`, registers the raw-skeleton,
  landscape, connect/disconnect, log and tracker callbacks, sets the Y-up
  right-handed coordinate system (the same `manus_core_world_y_up_rh_z_to_viewer_m`
  frame the Windows path used, so `--manus-core-calibration` files stay valid),
  discovers the embedded Core on loopback and connects;
- reads the Ultimate from `/tmp/vut.sock`, converts it into the Core world
  frame, and on every update calls `CoreSdk_SendDataForTrackers` with
  `TrackerType_<hand>Hand`, `TrackedPoint_Casing`, the glove owner's user id
  and the VUT quality; once per user it calls `CoreSdk_AssignTrackerToUser`,
  `CoreSdk_AssignRoleToTracker` and `CoreSdk_SetTrackerOffset(<Hand>TrackerToCasing)`;
- with `manus_wrist_composition: bridge` (default) it re-roots every raw
  skeleton of the required hand so the wrist sits at
  `tracker ⊗ tracker_to_wrist`, preserving each node's wrist-local pose
  exactly. This makes the published hand tracker-driven even if embedded
  Core ignores injected trackers for the raw skeleton; `core` trusts Core's
  own tracker handling instead (`CoreSdk_SetRawSkeletonHandMotion(Tracker)` is
  requested in both cases);
- emits `trackers` packets for the Ultimate itself (`tracking_system:
  vive_hub_linux`, quality `trackable`/`bad` (rotation-only)/`untrackable`),
  so `ManusSource` applies the same identity/role/quality/freshness gate as
  before: the exact serial, `<hand>_hand`, the glove owner's user id, a valid
  advancing pose, and `manus_tracker_stale_timeout`;
- accepts `{"type": "vibrate", "side": "right", "powers": [t, i, m, r, p]}`
  on stdin and forwards it to `CoreSdk_VibrateFingersForGlove` (haptics).

Recording metadata now carries `backend`, `tracker_feed`,
`vive_ultimate_socket`/`_frame`, `wrist_composition`, `tracker_to_wrist`,
`required_glove_user_id` and the sidecar's own SHA-256.

## 5. Acceptance diagnostic (run first)

Gloves on, Ultimate tracking in VIVE Hub, nothing else attached to the
dongles. Move the wrist by at least 5 cm and rotate it while it runs:

```bash
dex-teleop-manus-diagnostics \
  --mode integrated --hand right --duration 15 \
  --sdk-root ~/manus_setup/vendor/ManusSDK_v3.2.1 \
  --tracker-feed vive_ultimate --vive-ultimate-socket /tmp/vut.sock \
  --expected-tracker-id 59-A33R03250 \
  --require-tracker-motion --require-trackable-tracker \
  --tracker-stale-timeout 1.0 \
  --output outputs/manus-linux-acceptance.jsonl
```

It exits nonzero for a missing stream, a raw-skeleton rate below
`--minimum-rate-hz`, a same-callback identity mismatch, less than 5 cm of
global wrist translation, or a tracker that is not the expected serial,
role, owner, `vive_hub_linux` system, valid pose, advancing timestamp or
`trackable` quality. Hardware-only checks:

- the embedded Core reports `license_integrated: true` in the `landscape`
  event and the right glove's topology arrives;
- the VUT snapshot lists the serial in state `ready`, not `syncing`/`build_map`;
- moving only the Ultimate moves the global wrist while articulation stays
  wrist-local; rotating the hand about the wrist does not translate the
  wrist by more than the mount offset error (tune `manus_tracker_to_wrist_*`);
- covering the tracker's cameras flips quality to `bad`/`untrackable` and the
  wrist stops (the launcher raises after `manus_tracker_stale_timeout`);
- Ctrl+C leaves no `manus_python_bridge` or glove vibration behind.

## 6. Launch teleoperation

```bash
conda run -n behavior_dex python dex_teleop/scripts/launch_og.py --task <activity>
```

Press SPACE to engage and start recording. The spelled-out equivalent of the
preset is:

```bash
python dex_teleop/scripts/launch_og.py --no-config --task <activity> \
  --hand right --hand-source manus --wrist-source manus \
  --manus-backend python --manus-mode integrated \
  --manus-sdk-root ~/manus_setup/vendor/ManusSDK_v3.2.1 \
  --manus-tracker-feed vive_ultimate --vive-ultimate-socket /tmp/vut.sock \
  --manus-expected-tracker-id 59-A33R03250 \
  --manus-tracker-stale-timeout 1.0 --record-hand-poses
```

Gloves-only (no Ultimate): `--config hts --hand-source manus --manus-sdk-root ~/manus_setup/vendor/ManusSDK_v3.2.1`.

## 7. Haptic feedback (Metaglove Pro Haptic)

`manus_haptics: true` (or `--manus-haptics`) turns simulated contact into
per-finger vibration, following the NVIDIA Isaac Teleop haptic contract
(tactile vector → retargeter → `HapticSink` → `IHapticDevice`, amplitudes
ordered thumb..pinky) implemented in `dex_teleop/omnigibson/haptics.py`:

- after every physics step the launcher reads the Sharpa hand's contact
  points (`robot.get_finger_contact_data`, the same view the assisted grasp
  uses), sums the force magnitude per digit (fingertip pads fully, other
  finger links at half weight, palm ignored);
- `manus_haptics_min_force` (0.5 N) starts vibration at
  `manus_haptics_min_amplitude` (0.15) and `manus_haptics_max_force` (6 N)
  saturates it; onsets are immediate, releases decay with
  `manus_haptics_smoothing`;
- commands are rate-limited to `manus_haptics_rate` (30 Hz), resent only when
  they change (or every 0.5 s while active), and zeroed when teleoperation
  disengages, on episode reset, and at shutdown;
- the sidecar calls `CoreSdk_VibrateFingersForGlove(glove_id, powers)`; a
  glove without haptics (`isHaptics` false in the landscape) silently ignores
  them.

This is the same loop Isaac Teleop closes with its MANUS plugin
(`HapticCommand` on the `manus_glove_haptic` collection); the glove-side
contract is identical, so a future Isaac Teleop host can reuse
`ContactForceToFingerAmplitudes` and `ManusGloveHapticDevice` unchanged.

## 8. Troubleshooting

- `Could not find the MANUS Core 3.2 Python SDK`: `manus_sdk_root` must be
  the extracted tarball (containing `Python/manus_sdk`) or a parent with
  `vendor/ManusSDK_v*`; or `pip install <SDK>/Python`.
- `The Manus Python SDK requires the 'cffi' package`: `pip install cffi` in
  `behavior_dex`.
- `CFFI bindings not found for linux/amd64`: the interpreter's Python
  version has no prebuilt extension (3.9-3.14 only).
- `integrated_license_unavailable`: the dongle's license lacks *SDK
  (integrated)*/*Linux*; retrieve the newest license in the Linux Dashboard.
- `no_glove ... did not qualify license, right-hand topology, raw frame`:
  glove off/unpaired, dongle not readable (udev), or another process (the
  Dashboard, a second launcher) owns the dongle.
- `expected tracker '...' is absent`: VIVE Hub is not running, the socket
  path differs (`vive_ultimate_socket`), or the serial is not the one VIVE
  Hub reports; list them with the Python snippet in section 3.
- `tracker quality is 'untrackable'`/`'bad'`: the Ultimate is `syncing`,
  needs *Setup*, or lost tracking (rotation-only); check lighting, features,
  lens cleanliness, and rebuild the map.
- The hand moves along the wrong axes: toggle `vive_ultimate_frame`
  (`x_right_y_up_z_back` for VIVE Hub, `x_forward_z_up` for HTC's SDK demo
  server), then `tracking_yaw_deg` for a residual yaw.
- Haptics never trigger: the glove must be a Metaglove Pro *Haptic*
  (`haptics: true` in the sidecar's `topology` event); lower
  `manus_haptics_min_force`.
- Still have a Windows MANUS Core? `manus_mode: remote` with
  `manus_backend: python` connects to it (discovery, `manus_core_host`,
  reconnect) exactly like the C++ bridge did; see
  `manus-remote-vive-ultimate-runbook.md`.
