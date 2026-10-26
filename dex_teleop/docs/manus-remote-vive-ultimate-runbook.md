# MANUS Core Remote (Windows) + VIVE Tracker Ultimate runbook (legacy path)

> Since MANUS Core 3.2 the default `dex_teleop` path runs entirely on Linux:
> the MANUS Core 3.2 Python SDK in Integrated mode plus VIVE Hub for Linux for
> the Ultimate (`configs/manus_core.yaml`, documented in
> `manus-linux-vive-ultimate-runbook.md`). This document describes the
> previous production path, still selectable with `manus_mode: remote`
> (either `manus_backend: python` or the C++ `native` bridge described below).

In this path VIVE Hub, SteamVR, MANUS Core 3, MANUS gloves, VIVE Tracker
Ultimate, and Quest run on Windows 10/11. Linux connects to that Core
instance through the MANUS Remote SDK. MANUS Integrated on Linux is the
gloves-only fallback here.

Do not configure a guessed MANUS network port. The client uses
`CoreSdk_LookForHosts`; allow the MANUS Core applications through Windows
Firewall on the private network profile and use the hosts reported by the
diagnostic.

## 1. Windows host

1. Install/update VIVE Hub, SteamVR, MANUS Core 3, the Quest desktop software,
   and the GPU drivers.
2. Pair each Ultimate in VIVE Hub and complete its room/map setup. Confirm the
   trackers move correctly in SteamVR before opening MANUS Core.
3. Connect Quest by Link/Air Link and confirm its SteamVR/OpenXR view uses the
   same Windows tracking session. Quest is not a substitute transport for an
   Ultimate; Core receives the tracker pose through SteamVR.
4. Pair the MANUS gloves in Core, assign left/right users, and complete glove
   calibration.
5. Assign each Ultimate to the matching hand in Core. Select the physical
   tracker-to-wrist mounting preset that matches the mount. Enable tracker
   motion for the raw skeleton.
6. Confirm Core's raw skeleton has finger articulation and a global wrist that
   follows the Ultimate. Confirm the installed license enables the **SDK**
   feature (Remote), not only **Integrated**.
7. Put Windows and Linux on a network where host discovery is permitted.
   Prefer a stable Windows address or hostname and pass it explicitly after
   proving discovery.

Steps 2–7 are hardware-only acceptance prerequisites.

## 2. Linux SDK and bridge

Use the matching Linux SDK package for the Core version. The install root must
contain the SDK header and both libraries when both modes are needed:

```text
ManusSDK/include/ManusSDK.h
ManusSDK/lib/libManusSDK.so
ManusSDK/lib/libManusSDK_Integrated.so
```

For MANUS Core 3.2.1 use the Linux SDK 3.2.1 package
(`MANUS_Core_3.2.1_SDK_Linux.tar.gz` on the MANUS 3.2.1 downloads page). It
ships one `lib/amd64/libManusSDK-amd64.so` for both modes with gRPC and
Protobuf linked in, so it needs no separate Remote runtime:

```bash
mkdir -p ~/manus_setup/vendor/ManusSDK_v3.2.1
tar -xzf MANUS_Core_3.2.1_SDK_Linux.tar.gz -C ~/manus_setup/vendor/ManusSDK_v3.2.1
cd /home/minyeongk/Desktop/emg/Behavior-1K-arat/dex_teleop
for mode in integrated remote; do
  bash src/dex_teleop/native/manus_bridge/build_manus_bridge.sh \
    --mode "$mode" --sdk-root ~/manus_setup/vendor/ManusSDK_v3.2.1
done
```

Set `manus_sdk_root` in `configs/manus_core.yaml` to the same root. When
several releases are installed under `~/manus_setup/vendor`, the newest is
used unless a root is given. The gRPC runtime below applies only to SDK v3.1.1.

MANUS SDK v3.1.1's authoritative
`SDKClient_Linux/Dockerfile` checks out gRPC tag `v1.28.1` (commit
`cb81fe0dfaa424eb50de26fb7c904a27a78c3f76`) and recursively checks out its
Protobuf v3.11.2 submodule (commit
`fe1790ca0df67173702f70d5646b82f48f412b99`). It configures Protobuf with
`--enable-shared`, builds and checks Protobuf, then builds gRPC with the
upstream Makefile. Reproduce that build in the isolated user cache:

```bash
cd /home/minyeongk/Desktop/emg/Behavior-1K-arat/dex_teleop

export MANUS_SDK_ROOT=/home/minyeongk/manus_setup
export DEX_TELEOP_MANUS_REMOTE_RUNTIME="$HOME/.cache/dex_teleop/manus_sdk_v3.1.1_remote_runtime"

bash src/dex_teleop/native/manus_bridge/build_manus_remote_runtime.sh \
  --prefix "$DEX_TELEOP_MANUS_REMOTE_RUNTIME" \
  --source-dir "$HOME/.cache/dex_teleop/sources/grpc-v1.28.1"
```

The runtime builder verifies the upstream tag object, creates a disposable
detached worktree at the exact gRPC commit, initializes and checks every
recursive submodule gitlink, and requires the disposable tree to have no
tracked or untracked changes before compilation. A dirty reusable
`--source-dir` therefore cannot alter the built inputs. `BUILD-METADATA.txt`
records the actual commits, pre/post-build dirty state hashes, recursive
submodule-manifest hash, compiler/tool versions, SDK v3.1.1 fingerprints, and
installed binary hashes. `MANIFEST.sha256` and the ownership marker
`.dex-teleop-manus-remote-runtime` are validated on every reuse, together with
the exact files, symlink containment, pkg-config/protoc versions, ELF SONAMEs,
and `ldd -r` closure.

gRPC v1.28.1 intentionally reports C-core pkg-config version `9.0.0` (matching
`libgrpc.so.9`) and C++ pkg-config version `1.28.1`. On Ubuntu hosts that
happen to have OpenSSL development headers, the builder explicitly selects
the bundled BoringSSL provider that the vendor's Ubuntu Focal image selects
because that Dockerfile does not install OpenSSL development headers.

The default prefix is the one approved destructive target. Additional managed
prefixes may be children of
`$XDG_CACHE_HOME/dex_teleop/manus_remote_runtimes/`. Existing paths are
replaced only when their exact canonical-path ownership marker is valid (the
strictly recognized legacy default is migrated by its first forced rebuild).
Repository, home, SDK, system, and Conda roots are always rejected, including
through symlink aliases. A non-protected location outside the approved cache
requires both an existing valid marker and the high-friction
`--allow-owned-prefix-outside-cache` flag.

The builder does not invoke `sudo`, a package manager, or the system
`ldconfig`, and it restores a previous isolated prefix if a forced rebuild
fails. Existing system ZeroMQ/zlib and all Conda libraries remain untouched.
Never create global compatibility symlinks for these SONAMEs.

Build each bridge mode explicitly:

```bash
bash src/dex_teleop/native/manus_bridge/build_manus_bridge.sh \
  --mode integrated --sdk-root "$MANUS_SDK_ROOT"
bash src/dex_teleop/native/manus_bridge/build_manus_bridge.sh \
  --mode remote \
  --sdk-root "$MANUS_SDK_ROOT" \
  --runtime-prefix "$DEX_TELEOP_MANUS_REMOTE_RUNTIME"
```

The bridge builder fingerprint-checks the complete v3.1.1 or v3.2.1 header/library set,
builds the executable and mode-private SDK link in a staging bundle, verifies
RPATH, `ldd -r`, and the side-effect-free protocol probe, and only then
atomically replaces the published runtime link and executable. A compile,
link, or pre-publication verification failure leaves the previous pair
unchanged.

The defaults are
`~/.cache/dex_teleop/manus_bridge/manus_bridge` and
`~/.cache/dex_teleop/manus_bridge/manus_bridge_remote`. Each has a private
mode-specific RPATH. Remote additionally embeds the explicit isolated runtime
`lib` directory in its RPATH, so normal launches do not need
`LD_LIBRARY_PATH`. Integrated ignores `DEX_TELEOP_MANUS_REMOTE_RUNTIME` and
does not search or link that prefix. In SDK v3.1.1, settings/log-location calls
configure embedded Core and return `FunctionNotAvailable` in Remote mode, so
`--settings-dir` and `--log-dir` apply only to Integrated.

Verify versions, SONAMEs, and loader resolution:

```bash
bash src/dex_teleop/native/manus_bridge/build_manus_remote_runtime.sh \
  --verify-only \
  --prefix "$DEX_TELEOP_MANUS_REMOTE_RUNTIME"
(cd "$DEX_TELEOP_MANUS_REMOTE_RUNTIME" &&
  sha256sum --quiet -c MANIFEST.sha256)

for library in \
  "$DEX_TELEOP_MANUS_REMOTE_RUNTIME/lib/libgrpc.so.9" \
  "$DEX_TELEOP_MANUS_REMOTE_RUNTIME/lib/libgrpc++.so.1" \
  "$DEX_TELEOP_MANUS_REMOTE_RUNTIME/lib/libprotobuf.so.22"; do
  readelf -d "$library" | grep SONAME
done

# Command-scoped only: direct inspection of the vendor library has no RPATH.
env LD_LIBRARY_PATH="$DEX_TELEOP_MANUS_REMOTE_RUNTIME/lib" \
  ldd "$MANUS_SDK_ROOT/vendor/ManusSDK_v3.1.1/ROS2/ManusSDK/lib/libManusSDK.so"

readelf -d "$HOME/.cache/dex_teleop/manus_bridge/manus_bridge_remote" |
  grep -E 'RPATH|RUNPATH'
ldd "$HOME/.cache/dex_teleop/manus_bridge/manus_bridge_remote"
"$HOME/.cache/dex_teleop/manus_bridge/manus_bridge_remote" --protocol-version
```

With no reachable Windows Core, a loader/startup smoke test should reach MANUS
host discovery, emit `host_connection_failed`, and exit 14 rather than report a
shared-library error:

```bash
"$HOME/.cache/dex_teleop/manus_bridge/manus_bridge_remote" \
  --mode remote \
  --connect-timeout 2 \
  --discovery-wait 1 \
  --no-tracker-diagnostics
```

For a clean cache-only rebuild, leave SDK, system, and Conda libraries in
place:

```bash
cd /home/minyeongk/Desktop/emg/Behavior-1K-arat/dex_teleop
bash src/dex_teleop/native/manus_bridge/build_manus_remote_runtime.sh \
  --force \
  --prefix "$DEX_TELEOP_MANUS_REMOTE_RUNTIME" \
  --source-dir "$HOME/.cache/dex_teleop/sources/grpc-v1.28.1"
bash src/dex_teleop/native/manus_bridge/build_manus_bridge.sh \
  --mode remote \
  --sdk-root "$MANUS_SDK_ROOT" \
  --runtime-prefix "$DEX_TELEOP_MANUS_REMOTE_RUNTIME"
```

Do not replace the Remote library with the Integrated library to bypass a
missing dependency.

The Python source probes each executable with the side-effect-free
`--protocol-version` command before launch. It passes generation and native
required-hand options only to protocol v2. Protocol-v1 binaries remain an
Integrated-only fallback with Python-side required-hand validation; their
negotiated v1 capability set is recorded rather than mislabeled as v2.

## 3. Core-world calibration

Core world is SteamVR's standing universe: right-handed, Y up, forward = -Z as
set during room setup. The launcher never treats it as the robot world
implicitly; it applies one of two calibrations:

**Built-in default (no `--manus-core-calibration`).** Core world is rotated
into the launcher's Z-up tracking frame (`x = -z_core`, `y = -x_core`,
`z = y_core`; quaternion xyzw `(0.5, -0.5, -0.5, 0.5)`), with no translation
and an identity `skeleton_wrist_to_anatomical_wrist`. This is the same
treatment the Quest wrist gets: position and orientation are anchored to the
robot's end effector at engagement, so only axis directions matter. Run SteamVR
room setup facing the robot's +X (or set `--tracking-yaw-deg`) and the default
is complete. The default is recorded in the session metadata as
`core_calibration` with `core_calibration_file: null`.

**Measured file.** Needed only when the recorded MANUS wrist must be expressed
in a surveyed robot frame, or when Core's wrist-node axes turn out not to match
the OpenXR anatomical basis. Create a JSON file containing
`reference_T_core_world` and one explicit wrist entry per hand:

```json
{
  "schema_version": 1,
  "core_world_frame": "manus_core_world_y_up_rh_z_to_viewer_m",
  "reference_frame": "robot_world",
  "reference_from_core_world": {
    "translation": [0.0, 0.0, 0.0],
    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
  },
  "wrists": [
    {
      "handedness": "right",
      "source_pose": "core_baked_skeleton_wrist",
      "core_tracker_offset_applied": true,
      "tracker_id": "replace-with-Core-tracker-id",
      "tracker_offset_preset": "describe-the-Core-preset",
      "tracker_to_anatomical_wrist": null,
      "skeleton_wrist_to_anatomical_wrist": {
        "translation": [0.0, 0.0, 0.0],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
      }
    }
  ]
}
```

All quaternions use `xyzw`. `reference_from_core_world` is measured by
registering at least three non-collinear points visible in both Core/SteamVR
world and the target reference, then solving and validating a proper rigid
transform. Core's selected tracker mount preset must already be baked into the
raw-skeleton wrist. `skeleton_wrist_to_anatomical_wrist` is a passive frame
relation (`skeleton_wrist_T_anatomical_wrist`), not an active whole-hand
correction. The runtime composes it into the global anatomical wrist and applies
its inverse to every wrist-local articulation transform, preserving every
global joint exactly. `tracker_to_anatomical_wrist` is optional audit metadata
for the physical mount and is not applied a second time.

Tracker acceptance policy is deliberately not configurable through this
geometric calibration. The launch/diagnostic runtime separately requires the
exact tracker ID and Core user ID, always requires the selected-hand role
(`<hand>_hand`) and `openvr` tracking system, and rejects HMDs. Calibration
documents containing `tracker_type`, `tracker_system`, or `tracker_user_id` are
rejected so they cannot weaken that policy. `tracker_id` in this document is
optional audit metadata; when present it must match the runtime policy.

Reject a calibration if held-out points or known wrist poses do not land in the
expected reference frame. Record the exact JSON with each session; the runtime
also records its SHA-256.

## 4. Hardware acceptance diagnostic

Run this first. Move the wrist by at least 5 cm and rotate it while it runs:

```bash
conda run -n behavior_dex env PYTHONPATH=src \
  python -m dex_teleop.tracking.manus_diagnostics \
  --mode remote \
  --core-host 192.0.2.10 \
  --hand right \
  --hand-motion tracker \
  --sdk-root ~/manus_setup \
  --core-calibration config/manus-core-right.json \
  --duration 15 \
  --require-tracker-motion \
  --require-trackable-tracker \
  --expected-tracker-id replace-with-Core-tracker-id \
  --expected-tracker-user-id 1 \
  --output outputs/manus-remote-acceptance.jsonl
```

The command exits nonzero for a missing stream, a raw-skeleton rate below the
threshold, a same-callback identity mismatch, insufficient global wrist
translation, or failure of the exact expected tracker ID, hand/user role,
OpenVR system, non-HMD identity, valid pose, advancing update identity,
freshness, or `trackable` quality. The JSONL contains Core
host/version, callback and receipt times, publish time, frame/sequence,
global-wrist poses, tracker IDs/quality, calibration, and a final summary.

Hardware-only checks:

- discovery lists the intended Windows Core and explicit selection is stable;
- SDK/Core versions are compatible and the Remote SDK license is accepted;
- the expected glove ID/side and Ultimate tracker ID/role are present;
- raw skeleton remains near the expected rate without queue overflow;
- every articulation/wrist pair has identical callback time and sequence;
- moving only the Ultimate moves the global wrist while articulation remains
  wrist-local;
- unplug/restart Core causes a full Linux SDK shutdown/reinitialize, reconnects
  the locked host, and resumes only after license/topology/required-hand
  requalification;
- the calibrated wrist lands in the `dex_teleop` reference without reflection
  or a 90/180-degree basis error.

## 5. Launch teleoperation

This path is the launcher's default preset, `configs/manus_core.yaml`
(`hand_source: manus`, `wrist_source: manus`, `manus_mode: remote`,
`record_hand_poses: true`). Fill in the rig-specific keys once:

```yaml
manus_core_host: 192.0.2.10
manus_expected_tracker_id: replace-with-Core-tracker-id
manus_expected_tracker_user_id: 1
# manus_core_calibration: config/manus-core-right.json   # only for a surveyed robot frame
```

then launch with only the activity and press SPACE to engage and start
recording:

```bash
conda run -n behavior_dex python scripts/launch_og.py --task <activity>
```

Every key can still be given as its `--flag` on the command line, which
overrides the file; the fully spelled-out equivalent is:

```bash
conda run -n behavior_dex python scripts/launch_og.py \
  --no-config \
  --task <activity> \
  --hand right \
  --hand-source manus \
  --wrist-source manus \
  --manus-mode remote \
  --manus-core-host 192.0.2.10 \
  --manus-sdk-root ~/manus_setup \
  --manus-expected-tracker-id replace-with-Core-tracker-id \
  --manus-expected-tracker-user-id 1 \
  --record-hand-poses
```

MANUS can instead be record-only with `--record-hand-source manus` and/or
`--record-wrist-source manus`. A MANUS wrist role requires Remote mode and the
explicit exact Ultimate ID and Core user ID policy flags. The raw-skeleton hand
motion is not configurable: the launcher requests `tracker` from Core whenever
MANUS supplies a wrist, so Core cannot silently fall back away from the
Ultimate, and `auto` otherwise (the root is then unused).

The gloves-only Integrated fallback starts from the Quest preset and overrides
only the articulation source:

```bash
conda run -n behavior_dex python scripts/launch_og.py \
  --config hts \
  --task <activity> \
  --hand-source manus \
  --manus-sdk-root ~/manus_setup
```

The former lighthouse VIVE wrist provider (`--wrist-source vive`,
`--vive-calibration`) has been removed; the Ultimate reaches the launcher only
through Core as described above.
