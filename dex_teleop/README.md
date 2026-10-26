# dex_teleop: Sharpa hand bench quick start

A right Sharpa hand fixed above a table in OmniGibson (Isaac Sim 5.1). Its 22
finger joints follow a MANUS glove (or Quest hand tracking); the wrist is held
by the simulator, so no wrist tracker is involved. Press `B` to switch between
the default view (back of the hand) and the egocentric view (palm). Sessions
can be recorded to HDF5 together with the OYMotion EMG wristband (step 7).

These steps were verified on Ubuntu 22.04 with an RTX 4090 in a fresh conda
environment. Budget roughly 30 minutes of downloads (Isaac Sim wheels plus
3.8 GB of robot assets) and 40 GB of disk.

## 0. Prerequisites

- Ubuntu 22.04, an NVIDIA RTX GPU with about 6 GB of free VRAM, and a recent
  NVIDIA driver (the default install uses CUDA 12.8 PyTorch wheels; pass
  `--cuda-version 12.6` to `setup.sh` if the driver is older).
- `git`, `conda` (Miniconda is fine), `curl`, `unzip`.
- Two asset archives from Minyeong: `hand_bench_arat_datasets.zip` and
  `hand_bench_sharpa_robot_assets.zip` (see step 3).
- For the glove: MANUS Metagloves with their wireless dongle, the Linux SDK
  package `MANUS_Core_3.2.1_SDK_Linux.tar.gz` (MANUS Knowledge Center, 3.2.1
  downloads; it contains the Python SDK), and a `.mcal` glove calibration for
  the person wearing the glove.

## 1. Get the code

```bash
git clone --branch arat-no-wrist git@github.com:kmy17518/BEHAVIOR-1K.git Behavior-1K-arat
# or: git clone --branch arat-no-wrist https://github.com/kmy17518/BEHAVIOR-1K.git Behavior-1K-arat
cd Behavior-1K-arat
```

Already have a clone? `git fetch origin && git checkout arat-no-wrist && git pull`.
Everything below runs from this `Behavior-1K-arat` directory.

## 2. Create the conda environment

```bash
./setup.sh --new-env behavior_dex --omnigibson --bddl --dex-teleop
conda activate behavior_dex
```

Answer `y` to the Conda/NVIDIA terms prompt (or add `--accept-conda-tos
--accept-nvidia-eula`). Do **not** pass `--dataset`: it downloads the 30+ GB
BEHAVIOR-1K scene assets, which the bench does not use. The script installs
Python 3.11, PyTorch, BDDL, OmniGibson, Isaac Sim 5.1, and `dex_teleop`
(editable, with its NLopt/Pinocchio retargeting dependencies).

## 3. Assets

All assets live in `datasets/` inside the checkout (git ignores them).

```bash
# 3a. Official OmniGibson robot assets, 3.8 GB from HuggingFace -> datasets/omnigibson-robot-assets
OMNI_KIT_ACCEPT_EULA=YES python -m omnigibson.utils.asset_utils --download_omnigibson_robot_assets

# 3b. Sharpa hand + ARAT assets from the two zips (they are not downloadable)
unzip -o /path/to/hand_bench_sharpa_robot_assets.zip -d datasets/
unzip -o /path/to/hand_bench_arat_datasets.zip -d datasets/

# 3c. Check
python -c "from dex_teleop.hand_bench import load_hand_bench_scene, validate_runtime_assets; validate_runtime_assets(load_hand_bench_scene()); print('hand bench assets OK')"
```

After unzipping, `datasets/` contains:

```text
datasets/omnigibson-robot-assets/models/franka/franka.yaml                     <- replaced: adds the sharpa_right end effector
datasets/omnigibson-robot-assets/models/franka/franka_dexhand/franka_sharpa_right/  <- Franka + Sharpa USD/URDF/meshes
datasets/omnigibson-robot-assets/models/franka_mounted_sharpa_{right,left}/     <- standalone Franka+Sharpa robot models
datasets/arat-assets-v1/                                                        <- table used by the bench, ARAT objects, arat_base scene
datasets/arat-task-instances/                                                   <- 19 ARAT task scene instances (ARAT launcher only)
```

## 4. Smoke test without hardware

```bash
# Interactive window: the hand stays open; B toggles the view, Ctrl+C exits
python dex_teleop/scripts/launch_hand_bench.py --view-only

# No display (e.g. over SSH): render both views to PNG and exit
OMNIGIBSON_HEADLESS=1 python dex_teleop/scripts/launch_hand_bench.py --screenshot-dir dex_teleop/outputs/hand_bench
```

The first launch compiles shaders and can take several minutes; later launches
take about 30 s. A successful load prints `Hand bench ready: right_hand_C_MC at
[0.0, 0.0, 0.891] ... tabletop at 0.750 m`.

## 5. MANUS Core 3.2 Python SDK (Linux, gloves only)

The bench talks to the gloves through a small pure-Python sidecar
(`dex_teleop.tracking.manus_python_bridge`) built on the official MANUS Core
3.2 Python SDK (`manus_sdk`, pre-built CFFI binaries for Python 3.9-3.14 on
x86_64 and ARM64). Nothing is compiled and the proprietary SDK files stay
outside the repository. The legacy C++ sidecar is still available with
`--manus-backend native` (see `docs/manus-remote-vive-ultimate-runbook.md`).

```bash
# 5a. Runtime libraries the SDK needs, plus cffi in the conda environment
sudo apt install libusb-1.0-0 libzmq5 libudev1 zlib1g libgomp1 libcurl4
pip install "cffi>=1.15"            # or: pip install -e './dex_teleop[manus]'

# 5b. Unpack the SDK where dex_teleop looks for it; MANUS_SDK_ROOT=<dir> overrides the search
mkdir -p ~/manus_setup/vendor/ManusSDK_v3.2.1
tar -xzf MANUS_Core_3.2.1_SDK_Linux.tar.gz -C ~/manus_setup/vendor/ManusSDK_v3.2.1
ls ~/manus_setup/vendor/ManusSDK_v3.2.1/Python/manus_sdk/linux/amd64/
#   must show libManusSDK-amd64.so and _manus_sdk.cpython-3XX-x86_64-linux-gnu.so

# 5c. Let your user open the dongle (once), then re-plug the dongle
sudo tee /etc/udev/rules.d/70-manus-hid.rules >/dev/null <<'EOF'
# HIDAPI/libusb
SUBSYSTEMS=="usb", ATTRS{idVendor}=="3325", MODE:="0666"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="83fd", MODE:="0666"

# HIDAPI/hidraw
KERNEL=="hidraw*", ATTRS{idVendor}=="3325", MODE:="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger

# 5d. Validate with the glove on, dongle plugged in; move your fingers for 10 s
dex-teleop-manus-diagnostics --mode integrated --hand right --duration 10 \
  --sdk-root ~/manus_setup/vendor/ManusSDK_v3.2.1 \
  --glove-calibration /path/to/right-hand.mcal --output /tmp/manus_diagnostics.jsonl
```

The diagnostics print a JSON summary and exit 0 when `"passed": true`.
Pairing, firmware updates, license retrieval, and the guided `.mcal`
calibration can now be done on Linux with the `MANUS Core Dashboard (Linux
build)` shipped in the same SDK package (it attaches to an Integrated SDK
session; stop the bench/diagnostic first, only one process may own the
dongle), or on a Windows MANUS Core as before. The `.mcal` is optional
(`--manus-calibration` may be omitted) but finger angles are only accurate
with the wearer's own calibration.

### Already have the SDK and dongle set up? Start at 5d

Skip 5a-5c if all of this holds; check it first:

```bash
# SDK 3.2.x Python package present for your interpreter (3.9-3.14)
ls <SDK>/Python/manus_sdk/linux/amd64/   # <SDK> is your ManusSDK_v3.2.1 folder
python -c "import cffi; print(cffi.__version__)"

# Dongle enumerated (glove may still be off)
lsusb -d 3325:      # expect: ID 3325:0049 Manus VR (https://www.manus-vr.com) Sensor Dongle

# Your user may open it: the dongle's hidraw node must be crw-rw-rw- (root-only => do 5c)
for h in /sys/class/hidraw/hidraw*; do v=$(udevadm info -q property -p "$h" | sed -n 's/^ID_VENDOR_ID=//p'); [ "$v" = 3325 ] && ls -l "/dev/$(basename "$h")"; done
```

Then run 5d with `--sdk-root <SDK>` (the search also accepts a parent with
`vendor/ManusSDK_v*/`, newest release first, or an installed `manus_sdk`
package) and your existing `.mcal`. Reading 5d:

- `"passed": true`: glove, dongle, license, and calibration are fine; go to 6.
- `Could not find the MANUS Core 3.2 Python SDK`: wrong `--sdk-root` or a
  3.1.x package (no `Python/` folder).
- `no_glove: MANUS connection did not qualify license, right-hand topology,
  raw frame`: SDK and sidecar work, but no glove data: glove off or not paired
  to this dongle, dongle not readable (permissions), or no `SDK (integrated)`
  license on it (`integrated_license_unavailable` once a glove is seen).
- `Calibration rejected for right glove`: the `.mcal` belongs to another glove
  or SDK version; omit `--glove-calibration` to test without it.

The official `Python/examples/minimal_client.py` from the SDK package (mode
`1`, Integrated) is an equivalent independent glove check.

## 6. Run the hand bench with the glove

```bash
python dex_teleop/scripts/launch_hand_bench.py \
  --hand-source manus --manus-calibration /path/to/right-hand.mcal
```

The fingers follow the glove as soon as the first frame arrives. Keys: `B`
toggles default/egocentric view, `SPACE` pauses and resumes, `R` reopens the
hand, Ctrl+C exits. Useful options: `--stale-frame-policy hold` (hold the last
pose instead of failing on a stale stream), `--finger-target-scale 0.8`
(shrink flexion), `--retargeter dexpilot` (optional alternative backend,
`pip install -e './dex_teleop[dexpilot]'`), `--show-arm` (reveal the hidden
Franka that carries the hand), `--no-tracking-process` (retarget inside the
simulator process instead of a child process; slower, see
`docs/README_full.md`). Wrist options (`--wrist-source`, `--tracking-yaw-deg`,
MANUS tracker flags) are rejected on purpose. Quest hand tracking works too:
`--hand-source quest` listens for the HTS app on UDP 9000.

## 7. Record a session, with or without the EMG wristband

```bash
# Glove only -> dex_teleop/outputs/recordings/hand_bench_<timestamp>.hdf5
python dex_teleop/scripts/launch_hand_bench.py \
  --hand-source manus --manus-calibration /path/to/right-hand.mcal --record

# Glove + OYMotion EMG (implies --record); the band must be on and paired to no one else
python dex_teleop/scripts/launch_hand_bench.py \
  --hand-source manus --manus-calibration /path/to/right-hand.mcal \
  --emg --emg-device D8:71:4D:8D:99:92 --emg-adapter hci1 \
  --recording-path dex_teleop/outputs/recordings/session_01.hdf5
```

### Synchroni SDK (required for `--emg`)

`setup.sh` does not install the OYMotion SDK; use our fork:

```bash
git clone --branch my https://github.com/13RENDA/synchroni-sensor-sdk.git ../synchroni-sensor-sdk
pip install -e ../synchroni-sensor-sdk --no-deps
pip install "bleak>=2.1.1,<3" "flatbuffers>=25.0.0"

# Pre-check: band on, adapter from `bluetoothctl list`; expect e.g. [('OYWW1000', 'D8:71:4D:8D:99:92')]
SYNCHRONI_BLE_ADAPTER=hci1 python -c "from sensor import SensorController; \
  print([(d.Name, d.Address) for d in SensorController().scan(5000)])"
```

If the band does not connect or stream, consider switching to the versions: `pip install "bleak>=3.0.2" "flatbuffers>=25.0.0"`.

`--emg` scans for `--emg-scan-ms` (5 s) and needs exactly one match: the
`--emg-device` address or name substring, or, without the flag, the single
`OY*`/`Sync*`/`gForce*` device. Firmware filters default to HPF on, LPF on,
60 Hz notch (`--no-emg-hpf`, `--no-emg-lpf`, `--emg-notch 50|both|off`). The
waveform monitor docks right of the main view (`--no-emg-display` to skip it);
`--display decoder` adds the VEMG2Pose hand from the `emg2pose` checkout
beside this repository.

While recording, `R` ends the episode and starts a new one with an open hand
(`demo_0`, `demo_1`, ...); paused or stale steps are not recorded. Ctrl+C
finishes the episode and publishes the file atomically. The layout is the one
the ARAT launcher and its replay/analysis tools use:

```text
/data/demo_N/{action,state,...}       22 finger targets per step + serialized simulator state
/hand_tracking/articulation_streams   native-rate glove/Quest articulation (articulation.manus | articulation.quest)
/hand_tracking/wrist_streams          wrist.fixed_wrist (constant; the bench holds the wrist)
/hand_tracking/retargeting_streams    adaptive.<source>+fixed_wrist Sharpa joint results
/hand_tracking/action_alignment       exact stream rows behind every action
/human_hand_pose/demo_N               action-aligned 21 landmarks (raw_available is false: no Quest wrist)
/action_timing/demo_N                 monotonic clock before/after each env.step
/emg/samples, /emg/batches            native-rate microvolt samples and callback diagnostics   (--emg)
/synchronization/demo_N               EMG row ranges per episode and per action                (--emg)
```

## 8. Tests

```bash
pytest -q dex_teleop/tests
```

## Troubleshooting

- `Out of GPU memory` during startup: another process holds the VRAM; free
  about 6 GB.
- `OMNIGIBSON_DATA_PATH points outside Behavior-1K-arat`: unset that variable;
  the launchers always use the checkout's `datasets/`.
- `Missing required hand bench assets`: step 3 is incomplete; the message
  lists the missing files.
- `Could not find the MANUS Core 3.2 Python SDK`: check the path in 5b or set
  `MANUS_SDK_ROOT=~/manus_setup/vendor/ManusSDK_v3.2.1`.
- `no_glove ...` / `MANUS bridge did not publish a validated right-hand
  articulation within N seconds` (diagnostics or the bench): the SDK and
  sidecar are fine but no glove data arrived. Power the glove, re-plug the
  dongle, check the udev rule (5c), and confirm in the MANUS Core Dashboard
  (Linux build, or Windows) that the dongle carries the SDK (integrated)
  license.
- `BleakError: adapter 'hci0' not found`: pass the adapter listed by
  `bluetoothctl list` / `ls /sys/class/bluetooth` with `--emg-adapter`.
- `Expected exactly one EMG device matching ..., found 0`: the band is off,
  out of range, or connected to another host; the scan runs before Isaac Sim
  starts, so nothing else was touched.
- `Synchroni SDK is unavailable in this Python environment`: install the SDK
  as in step 7 (or pass `--emg-sdk-path`); the sidecar imports it from the
  same conda environment as the bench.

## More

The ARAT task launcher (`dex_teleop/scripts/launch_og.py`) reads its options
from a YAML preset in `dex_teleop/configs/` (`manus_core.yaml`, the default:
MANUS gloves + VIVE Tracker Ultimate, entirely on this Linux host through the
MANUS Core 3.2 Python SDK and VIVE Hub for Linux, with optional Metaglove Pro
Haptic feedback; `hts.yaml`: Quest hand and wrist tracking). Select one with
`--config <name>`, override any key with its command-line flag, or pass
`--no-config` for the built-in defaults. Setup, calibration, acceptance
checks and haptics for the default path are in
`dex_teleop/docs/manus-linux-vive-ultimate-runbook.md`; the previous Windows
MANUS Core path in `dex_teleop/docs/manus-remote-vive-ultimate-runbook.md`.
The launcher, recording format, replay, Quest wrist tracking, and EMG
recording are documented in `dex_teleop/guide.md` and the previous README,
kept in `dex_teleop/docs/` and in the `arat` branch history
(`git show a8ddd6bfd:dex_teleop/README.md`).
