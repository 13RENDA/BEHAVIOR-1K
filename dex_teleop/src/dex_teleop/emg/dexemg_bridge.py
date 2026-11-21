#!/usr/bin/env python3
"""EMG wrist band -> dexemg streaming decoder -> right-hand landmarks for dex_teleop.

Runs in the ``emg2pose`` conda environment (torch, dexemg, the Synchroni SDK), as its own process,
before the simulator starts::

    python dex_teleop/scripts/dexemg_bridge.py --checkpoint <deploy_wrist_s0.pt> --adapter hci0

The launchers then take the fingers from it with ``--hand-source dexemg`` (see ``configs/dexemg.yaml``).

What it does, and why each step is the way it is (matched to how the decoder was trained and to the
offline streaming study of 2026-11-13, /home/svl/claude_work/x9/stream_eval.py):

1. Acquisition: the wrist band (default BC:93:2A:3F:80:E7, the band HandSync records in slot
   ``right/wrist``) through the Synchroni SDK, with every firmware filter ON (HPF, LPF, 50 Hz and 60 Hz
   notch), as in every two-band training recording of 11-02 to 11-06 (their manifests read back all four
   ON), 8 samples per BLE packet (HandSync's batch size), no battery polling while streaming (the
   recording sidecar's workaround for GATT drops).  Raw SDK microvolts, no software filtering, which is
   what the exporter feeds the model.
2. Normalization: a neutral recording of the same wearing (default 12 s, the length of the training
   neutral takes); channel scale = per-channel std after trimming 0.5 s at each end, exactly
   ``finetune_data.dataset.compute_noise_floor_scale``.  The live demo's rest + max rule is deliberately
   not offered: offline it scored worse than predicting a constant pose.
3. Decoding: ``live_pose_demo.LiveInference``'s streaming scheme (100 ms chunks, 50 ms overlap, each
   window mean-centred and divided by the channel scale, LSTM state and last pose carried across chunks,
   never reset), re-implemented here without the demo's calibration code and checked equal to it in
   ``dex_teleop/tests/test_dexemg_bridge.py``.
4. Landmarks: emg2pose forward kinematics on the MIRRORED generic hand model (the generic model is a left
   hand; mirrored it is a right hand), mapped to MediaPipe-21 names, the thumb CMC taken at the CMC
   flexion pivot (UmeTrack has no landmark there), rotated into the OpenXR anatomical wrist frame
   (+Y dorsal, -Z toward the fingers) and scaled to meters.

Keys in this terminal: Enter re-records the neutral (3 s countdown, then hold still); ``q`` + Enter quits.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import math
import os
from pathlib import Path
import queue
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dex_teleop.emg.dexemg_protocol import (  # noqa: E402
    COORDINATE_FRAME,
    DEFAULT_HOST,
    DEFAULT_PORT,
    LANDMARK_NAMES,
    LANDMARK_SCHEMA,
    PROTOCOL,
    decode,
    encode,
)

SAMPLE_RATE_HZ = 500
CHANNELS = 8
DEFAULT_BAND = "BC:93:2A:3F:80:E7"
DEFAULT_NEUTRAL_SECONDS = 12.0
NEUTRAL_TRIM_SECONDS = 0.5
COUNTDOWN_SECONDS = 3.0
RECONNECT_SECONDS = 3.0
CLOSE_TIMEOUT_SECONDS = 8.0
# The Synchroni SDK build HandSync records with: sensor-sdk 0.9.11 with bleak 1.1.1 (HandSync's .venv,
# Python 3.13). The same versions installed for this interpreter with
#   pip install --no-deps --target ~/claude_work/emg_teleop/sensor_sdk_0911 sensor-sdk==0.9.11 bleak==1.1.1
# Other builds on this machine are wrong for the gForceUltra bands: the source checkout in
# HandSync/synchroni-sensor-sdk (1.2.6) treats every "gForce*" device as the old EMG type (no firmware
# filters, a different microvolt conversion).
DEFAULT_SDK_PATHS = (Path.home() / "claude_work" / "emg_teleop" / "sensor_sdk_0911",)


def default_sdk_path() -> Path | None:
    override = os.environ.get("DEXEMG_SYNCHRONI_SDK")
    if override:
        return Path(override).expanduser()
    return next((path for path in DEFAULT_SDK_PATHS if (path / "sensor" / "__init__.py").is_file()), None)


def bluetooth_adapters() -> list[str]:
    root = Path("/sys/class/bluetooth")
    return sorted(p.name for p in root.glob("hci*")) if root.is_dir() else []
DEFAULT_PACKAGE_COUNT = 8
# The decoder is small. Measured 2026-11-18 on replayed EMG (CPU of the bridge process / decode time per 100 ms chunk /
# median send delay): PyTorch's default of 16 threads 1339% / 33.8 ms / 90 ms (its idle threads spin between chunks);
# two threads that sleep when idle 19% / 12.7 ms / 60 ms; one thread 26% / 24.5 ms / 69 ms; four sleeping threads
# 32% / 29.2 ms / 81 ms.
DEFAULT_DECODER_THREADS = 2
SCALE_RATIO_WARN = 1.5
# A neutral whose per-channel std is this many times the training neutrals' median is not a neutral: the
# 11 training neutrals span 0.93-1.56 uV per channel (essentially the amplifier noise floor of a relaxed
# forearm), at most 1.4x their median. A tense or lifted arm, or a contact artifact, inflates the scale and
# the decoder is then under-driven (11-14 live session: ratios 1.9 / 4.5 / 2.9 and visibly weak motion).
SCALE_RATIO_REJECT = 2.0
FIRMWARE_FILTERS = {"hpf": True, "lpf": True, "notch": "both"}

# UmeTrack landmark index for each MediaPipe-21 name (emg2pose.constants.LANDMARKS).  "thumb_cmc" has no
# UmeTrack landmark and is filled from the CMC flexion pivot instead (see HandLandmarker).
UMETRACK_INDEX = {
    "wrist": 5,
    "thumb_mcp": 6, "thumb_ip": 7, "thumb_tip": 0,
    "index_mcp": 8, "index_pip": 9, "index_dip": 10, "index_tip": 1,
    "middle_mcp": 11, "middle_pip": 12, "middle_dip": 13, "middle_tip": 2,
    "ring_mcp": 14, "ring_pip": 15, "ring_dip": 16, "ring_tip": 3,
    "pinky_mcp": 17, "pinky_pip": 18, "pinky_dip": 19, "pinky_tip": 4,
}


def dexemg_root() -> Path:
    return Path(os.environ.get("DEXEMG_ROOT", Path.home() / "dexemg")).expanduser()


def _import_dexemg():
    """Put dexemg and its vendored emg2pose on sys.path; return (torch, EMGDecoderModel)."""

    root = dexemg_root()
    for path in (root, root / "third_party" / "emg2pose"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch  # noqa: WPS433
    from models.emg_decoder import EMGDecoderModel
    from training.ref_repo_compat import apply_all

    apply_all()
    return torch, EMGDecoderModel


def neutral_channel_scale(samples: np.ndarray, sample_rate_hz: int = SAMPLE_RATE_HZ,
                          trim_seconds: float = NEUTRAL_TRIM_SECONDS) -> np.ndarray:
    """(T, 8) neutral EMG -> (8,) std after trimming each end: compute_noise_floor_scale on one take."""

    samples = np.asarray(samples, dtype=np.float32)
    trim = int(round(trim_seconds * sample_rate_hz))
    kept = samples[trim:-trim] if trim > 0 else samples
    if kept.shape[0] < sample_rate_hz:
        raise ValueError(f"neutral recording too short: {kept.shape[0]} samples after trimming")
    return kept.T.std(axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------
# Model side


def apply_deadband(samples: np.ndarray, last: np.ndarray | None, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint deadband without jumps: the output moves only as far as needed to stay within ``width`` of the
    input, so a joint whose decoded angle wanders by less than ``width`` stays exactly still, and a real motion is
    followed from ``width`` behind. ``samples`` (n, joints) in time order, ``last`` the previous output (None at the
    start). Returns (filtered samples, new last)."""

    out = np.empty_like(samples)
    previous = samples[0] if last is None else last
    for t in range(samples.shape[0]):
        previous = np.clip(previous, samples[t] - width, samples[t] + width)
        out[t] = previous
    return out, previous


class StreamingDecoder:
    """dexemg's EMGDecoderModel driven like live_pose_demo.LiveInference (channel-scale path)."""

    def __init__(self, checkpoint: Path, device: str, chunk_steps: int = 50, overlap_steps: int = 25,
                 smoothing: float = 0.0, deadband_rad: float = 0.0) -> None:
        self.torch, model_class = _import_dexemg()
        torch = self.torch
        self.device = torch.device(device)
        ckpt = torch.load(str(checkpoint), map_location=self.device, weights_only=False)
        config = ckpt["config"]
        if config.get("tds_encoder", {}).get("normalize_input", True):
            raise SystemExit(f"{checkpoint}: expects internal per-window normalization; the bridge serves "
                             "fixed-scale (neutral-normalized) checkpoints only")
        self.model = model_class(config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.checkpoint = Path(checkpoint)
        reference = ckpt.get("reference_wrist_scale")
        if reference is None and "channel_scales" in ckpt:
            wrist = [np.asarray(v, np.float32) for k, v in ckpt["channel_scales"].items() if k.endswith(":wrist")]
            reference = np.median(np.stack(wrist), axis=0) if wrist else None
        self.reference_scale = None if reference is None else np.asarray(reference, np.float32)
        self.metadata = {k: ckpt.get(k) for k in ("arm", "band", "seed", "epochs", "days", "train_windows")
                         if ckpt.get(k) is not None}
        self.chunk_steps = int(chunk_steps)
        self.overlap_steps = int(overlap_steps)
        self.smoothing = float(np.clip(smoothing, 0.0, 0.999))
        self.deadband_rad = max(0.0, float(deadband_rad))  # after the smoothing; 0 = off
        self._deadband_last: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self._h = None
        self._prev_pose = None
        self._tail = np.zeros((self.overlap_steps, CHANNELS), dtype=np.float32)
        self._smoothed_last: np.ndarray | None = None

    def set_scale(self, scale: np.ndarray) -> None:
        self.scale = np.asarray(scale, dtype=np.float32).reshape(CHANNELS)

    def feed(self, new_samples: np.ndarray) -> np.ndarray:
        """(chunk_steps, 8) raw uV -> (chunk_steps, 20) joint angles (rad)."""

        torch = self.torch
        assert new_samples.shape == (self.chunk_steps, CHANNELS), new_samples.shape
        assert self.scale is not None, "no channel scale yet"
        new_samples = np.asarray(new_samples, dtype=np.float32)
        window = np.concatenate([self._tail, new_samples], axis=0)
        if self.overlap_steps > 0:
            self._tail = new_samples[-self.overlap_steps:]
        mean = window.mean(axis=0, keepdims=True)
        window = (window - mean) / (self.scale[None, :] + 1e-6)
        with torch.no_grad():
            x = torch.from_numpy(window.T).float().to(self.device).unsqueeze(0)
            angles, _velocities, (h, c) = self.model(
                x, initial_pose=self._prev_pose, initial_hidden=self._h,
                output_length=self.overlap_steps + self.chunk_steps, return_hidden=True,
            )
            self._h = (h.detach(), c.detach())
            self._prev_pose = angles[:, :, -1].detach()
            kept = angles[0, :, self.overlap_steps:].detach().cpu().numpy().T
        if self.smoothing > 0.0:
            out = np.empty_like(kept)
            prev = self._smoothed_last if self._smoothed_last is not None else kept[0]
            for t in range(kept.shape[0]):
                prev = self.smoothing * prev + (1.0 - self.smoothing) * kept[t]
                out[t] = prev
            self._smoothed_last = prev
            kept = out
        if self.deadband_rad > 0.0:
            kept, self._deadband_last = apply_deadband(kept, self._deadband_last, self.deadband_rad)
        return kept


class RecordOnlyDecoder:
    """Stands in for StreamingDecoder in a --record-only bridge: nothing is decoded."""

    reference_scale = None
    metadata: dict = {}

    def __init__(self, chunk_steps: int) -> None:
        self.chunk_steps = chunk_steps

    def set_scale(self, scale: np.ndarray) -> None:
        pass

    def feed(self, new_samples: np.ndarray) -> np.ndarray:
        raise RuntimeError("a record-only bridge does not decode")


class HandLandmarker:
    """(n, 20) joint angles -> (n, 21, 3) right-hand MediaPipe-21 landmarks, OpenXR wrist frame, meters."""

    def __init__(self, device: str = "cpu") -> None:
        torch, _ = _import_dexemg()
        from emg2pose.kinematics import TorchHandModel, forward_kinematics, load_default_hand_model
        from emg2pose.visualization import mirror_profile

        self.torch = torch
        self._fk = forward_kinematics
        self.device = torch.device(device)
        profile = mirror_profile(load_default_hand_model())  # generic model is a left hand
        self.model = TorchHandModel(profile).to(self.device)
        rest = self._umetrack(np.zeros((1, 20), np.float32))[0]
        wrist = rest[UMETRACK_INDEX["wrist"]]
        index_mcp, middle_mcp, pinky_mcp = (rest[UMETRACK_INDEX[n]] - wrist for n in ("index_mcp", "middle_mcp", "pinky_mcp"))
        distal = middle_mcp / np.linalg.norm(middle_mcp)
        palm = np.cross(index_mcp, pinky_mcp)  # palm side for a right hand
        dorsal = -palm - np.dot(-palm, distal) * distal
        y = dorsal / np.linalg.norm(dorsal)
        z = -distal
        x = np.cross(y, z)
        self.rotation = np.stack([x, y, z])  # rows: OpenXR wrist axes in UmeTrack coordinates
        if not np.isclose(np.linalg.det(self.rotation), 1.0, atol=1e-6):
            raise RuntimeError("UmeTrack -> OpenXR wrist rotation is not proper")
        # CMC flexion pivot (joint 0): fixed in the wrist frame, the closest UmeTrack point to MediaPipe's CMC.
        self.thumb_cmc = np.asarray(profile.joint_rest_positions[0], dtype=np.float64) - wrist
        self._index = np.asarray([UMETRACK_INDEX.get(name, -1) for name in LANDMARK_NAMES])

    def _umetrack(self, angles: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(np.asarray(angles, np.float32).T[None]).to(self.device)  # (1, 20, n)
        with torch.no_grad():
            return self._fk(x, self.model)[0].cpu().numpy().astype(np.float64)  # (n, 21, 3) mm

    def __call__(self, angles: np.ndarray) -> np.ndarray:
        ume = self._umetrack(angles)
        wrist = ume[:, UMETRACK_INDEX["wrist"]][:, None, :]
        picked = ume[:, np.maximum(self._index, 0)] - wrist
        picked[:, LANDMARK_NAMES.index("thumb_cmc")] = self.thumb_cmc
        return picked @ self.rotation.T * 1e-3


# ---------------------------------------------------------------------------------------------------------
# Acquisition


@dataclass
class SampleBlock:
    values: np.ndarray        # (n, 8) float32 uV
    sample_index: np.ndarray  # (n,) int64, device counter (replay: running index)
    t_ns: np.ndarray          # (n,) int64, estimated CLOCK_MONOTONIC per sample
    lost: int = 0
    lost_mask: np.ndarray | None = None  # (n,) bool, the SDK's isLost flag per sample (None: not known)
    received_ns: int | None = None       # CLOCK_MONOTONIC at the SDK callback (None: replay)


IMPEDANCE_DATA_TYPES = (0x12, 0x16)  # NTF_IMPEDANCE, NTF_IMPEDANCE_EXT (as HandSync reads them)


def _is_impedance_packet(kind) -> bool:
    try:
        return int(kind) in IMPEDANCE_DATA_TYPES
    except (TypeError, ValueError):
        return False


def impedance_from_packet(item) -> list[float] | None:
    """Per-channel electrode impedance (ohm) of one impedance packet: the last sample of each channel, as HandSync
    reads it; None unless all 8 channels carry a finite, non-negative value."""

    values = []
    for channel in item.channelSamples:
        channel = list(channel)
        if not channel:
            return None
        values.append(float(channel[-1].impedance))
    if len(values) != CHANNELS or not all(np.isfinite(v) and v >= 0.0 for v in values):
        return None
    return values


class BandAcquisition:
    """The wrist band over BLE, configured like the two-band training recordings."""

    def __init__(self, blocks: "queue.Queue[SampleBlock]", events: "queue.Queue[tuple[str, str]]", *,
                 device: str, adapter: str | None, scan_ms: int, package_count: int,
                 sdk_path: Path | None = None, sdk_log: bool = False) -> None:
        self.blocks, self.events = blocks, events
        self.device, self.adapter, self.scan_ms, self.package_count = device, adapter, scan_ms, package_count
        self.sdk_path = sdk_path
        self.sdk_log = sdk_log
        self.controller = self.profile = None
        self.metadata: dict = {}
        self._stopping = False
        # (ohm per channel, CLOCK_MONOTONIC ns) of the latest impedance packet; written by the SDK callback
        self.latest_impedance: tuple[list[float], int] | None = None

    def start(self) -> None:
        self._stopping = False
        os.environ.setdefault("SYNCHRONI_AUTO_RECONNECT", "0")
        if self.sdk_path is not None and str(self.sdk_path) not in sys.path:
            sys.path.insert(0, str(self.sdk_path))
        import sensor as sensor_package
        from sensor import DataType, DeviceStateEx, SensorController, sensor_utils
        from dex_teleop.emg.sidecar import (
            POWER_REFRESH_PERIOD_MS, SDK_COMMAND_TIMEOUT_S, SDK_CONNECT_TIMEOUT_S,
            _configure_hardware_filters, _select_device,
        )

        sensor_utils._TIMEOUT = max(sensor_utils._TIMEOUT, SDK_COMMAND_TIMEOUT_S)
        # The band answers on whichever adapter the dongle enumerated as today (hci0 on 11-06, hci1
        # before): without --adapter every adapter is scanned in turn.
        adapters = [self.adapter] if self.adapter else (bluetooth_adapters() or [None])
        device = None
        seen: list[str] = []
        for adapter in adapters:
            if adapter:
                os.environ["SYNCHRONI_BLE_ADAPTER"] = adapter
            self.events.put(("status", f"scanning on {adapter or 'the default adapter'} for {self.device}"))
            self.controller = SensorController()
            self._quiet_sdk_log()
            ensure_host = getattr(self.controller, "_ensure_bleak_host", None)
            if callable(ensure_host):
                ensure_host()
                self._quiet_sdk_log()  # again, now that the BLE host process exists to be told
            devices = self.controller.scan(self.scan_ms)
            try:
                device = _select_device(devices, self.device)
                self.used_adapter = adapter
                break
            except RuntimeError:
                seen.append(f"{adapter or 'default'}: {len(devices)} device(s)")
                self._terminate_controller()
        if device is None:
            raise RuntimeError(
                f"band {self.device} not found ({'; '.join(seen)}). Is it switched on, charged, and not "
                "connected to HandSync?"
            )
        profile = self.controller.requireSensor(device)
        if profile is None:
            raise RuntimeError(f"Failed to create a sensor profile for {device.Name}")
        period_ns = int(1e9 / SAMPLE_RATE_HZ)

        def data_type(data):
            getter = getattr(data, "getDataType", None)
            return getter() if callable(getter) else getattr(data, "dataType", None)

        def on_data(_profile, data) -> None:
            for item in data if isinstance(data, list) else [data]:
                if item is not None and item.channelSamples and _is_impedance_packet(data_type(item)):
                    try:
                        impedance = impedance_from_packet(item)
                        if impedance is not None:
                            self.latest_impedance = (impedance, time.monotonic_ns())
                    except Exception as error:  # noqa: BLE001 - reported, never swallowed silently
                        self.events.put(("error", f"Failed to read an impedance packet: {error}"))
                    continue
                if item is None or data_type(item) != DataType.NTF_EMG or not item.channelSamples:
                    continue
                try:
                    received_ns = time.monotonic_ns()
                    # sample.data is the SDK-scaled microvolt value HandSync records (sample fields differ
                    # between SDK builds, so only data / sampleIndex / isLost are read).
                    channels = item.channelSamples
                    n = len(channels[0])
                    if len(channels) != CHANNELS or n == 0 or any(len(channel) != n for channel in channels):
                        raise ValueError(f"unexpected EMG packet shape: {[len(c) for c in channels]}")
                    values = np.asarray([[sample.data for sample in channel] for channel in channels], dtype=np.float32).T
                    index = np.asarray([sample.sampleIndex for sample in channels[0]], dtype=np.int64)
                    lost_mask = np.asarray([bool(getattr(sample, "isLost", False)) for sample in channels[0]], dtype=np.bool_)
                    t_ns = received_ns - (n - 1 - np.arange(n, dtype=np.int64)) * period_ns
                    self.blocks.put_nowait(SampleBlock(values, index, t_ns, int(lost_mask.sum()), lost_mask, received_ns))
                except queue.Full:
                    self.events.put(("error", "EMG queue is full: the decoder fell behind"))
                except Exception as error:  # noqa: BLE001 - reported, never swallowed silently
                    self.events.put(("error", f"Failed to copy EMG callback: {error}"))

        def on_error(_profile, reason) -> None:
            self.events.put(("error", f"Synchroni SDK error: {reason}"))

        def on_state(_profile, state) -> None:
            if state == DeviceStateEx.Disconnected and not self._stopping:
                self.events.put(("error", "EMG band disconnected"))

        profile.onDataCallback = on_data
        profile.onErrorCallback = on_error
        profile.onStateChanged = on_state
        try:
            profile.autoReconnect = False
        except Exception:  # noqa: BLE001 - older SDK builds lack the attribute
            pass
        self.events.put(("status", f"connecting to {device.Name} {device.Address}"))
        if not sensor_utils.sync_call(profile._connect(), _timeout=SDK_CONNECT_TIMEOUT_S):
            raise RuntimeError(f"Failed to connect to {device.Name} ({device.Address})")
        if not profile.hasInited:
            profile.setParam("NTF_IMU", "OFF")
            if not profile.init(self.package_count, POWER_REFRESH_PERIOD_MS):
                raise RuntimeError(f"Failed to initialize {device.Name}")
        info = profile.getDeviceInfo()
        if int(info.EmgChannelCount) != CHANNELS or int(info.EmgSampleRate) != SAMPLE_RATE_HZ:
            raise RuntimeError(f"{device.Name}: expected {CHANNELS} channels at {SAMPLE_RATE_HZ} Hz, device reports "
                               f"{info.EmgChannelCount} at {info.EmgSampleRate} Hz")
        filters = _configure_hardware_filters(profile, **FIRMWARE_FILTERS)
        self.profile = profile
        self.metadata = {"device_name": device.Name, "device_address": device.Address,
                         "ble_adapter": getattr(self, "used_adapter", None) or "SDK default",
                         "sdk_path": None if self.sdk_path is None else str(self.sdk_path),
                         "firmware_version": info.FirmwareVersion,
                         "hardware_version": info.HardwareVersion, "model_name": info.ModelName,
                         "sdk_version": getattr(sensor_package, "__version__", "unknown"),
                         "firmware_filters": str(filters), "package_count": self.package_count}
        if not profile.startDataNotification():
            raise RuntimeError("Failed to start EMG data notifications")

    def _quiet_sdk_log(self) -> None:
        """Stop the SDK writing a log file for every connection attempt (unless --sdk-log).

        The SDK writes ~/Documents/sensorsdklog/sensor_controller_log_<time>.txt per controller start. A bridge
        left open with the band off retries every few seconds: about 485 files and 0.47 GB an hour, which
        helped fill the system disk on 2026-11-20 (4.6 GB in 11 hours).
        """

        if self.sdk_log or self.controller is None:
            return
        set_log_path = getattr(self.controller, "setLogPath", None)
        try:
            if callable(set_log_path):
                set_log_path(enabled=False)  # sensor-sdk 0.9.11: setLogPath(enabled, path); False = no log files
        except Exception as error:  # noqa: BLE001 - a build without the switch keeps its log files
            if not getattr(self, "_sdk_log_warned", False):
                self._sdk_log_warned = True
                self.events.put(("status", f"could not turn the SDK's log files off ({error})"))

    def close(self) -> None:
        self._stopping = True
        try:
            if self.profile is not None and getattr(self.profile, "isDataTransfering", False):
                self.profile.stopDataNotification()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.profile is not None:
                self.profile.disconnect()
        except Exception:  # noqa: BLE001
            pass
        self.profile = None
        self._terminate_controller()

    def _terminate_controller(self) -> None:
        controller, self.controller = self.controller, None
        try:
            if controller is not None:
                # SensorController() is one object per process and its terminate() works once: it sets
                # _terminated and every later call returns at once, so each later BLE host kept its threads
                # (two event loops, a queue feeder, a polling worker) after its process was killed. Measured
                # 2026-11-18 on a bridge retrying "band not found" for 58 min: 864 threads, 132% CPU.
                if getattr(controller, "_terminated", False):
                    controller._terminated = False
                controller.terminate()
        except BaseException:  # noqa: BLE001 - terminate() may raise SystemExit
            pass
        gc.collect()  # the stopped event loops are in reference cycles; their idle worker threads end with them


def kill_child_processes() -> int:
    """SIGKILL every descendant (the SDK's BLE host processes); orphans would keep the adapter and port."""

    parents: dict[int, int] = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
            parents[int(stat.parent.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    doomed, frontier = [], [os.getpid()]
    while frontier:
        parent = frontier.pop()
        for pid, ppid in parents.items():
            if ppid == parent:
                doomed.append(pid)
                frontier.append(pid)
    for pid in doomed:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return len(doomed)


def close_with_timeout(acquisition, seconds: float = CLOSE_TIMEOUT_SECONDS) -> bool:
    """Close the acquisition without letting a stuck BLE teardown hang the bridge."""

    worker = threading.Thread(target=acquisition.close, daemon=True, name="dexemg-close")
    worker.start()
    worker.join(seconds)
    return not worker.is_alive()


class ReplayAcquisition:
    """Replays the wrist channels of an exported take (finetune_data/aligned_two_band/*.npz) for testing."""

    def __init__(self, blocks: "queue.Queue[SampleBlock]", events, *, path: Path, realtime: bool,
                 package_count: int, loop: bool = False) -> None:
        self.blocks, self.events = blocks, events
        self.path, self.realtime, self.package_count, self.loop = Path(path), realtime, package_count, loop
        with np.load(self.path) as data:
            self.values = np.ascontiguousarray(np.asarray(data["emg"], dtype=np.float32)[:, :CHANNELS])
        self.metadata = {"replay": str(self.path), "realtime": realtime, "loop": loop,
                         "samples": int(self.values.shape[0])}
        self._thread = threading.Thread(target=self._run, daemon=True, name="dexemg-replay")
        self._stop = threading.Event()
        self.finished = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        n = self.package_count
        period = n / SAMPLE_RATE_HZ
        next_t = time.monotonic()
        total = self.values.shape[0] - self.values.shape[0] % n
        counter = 0
        while not self._stop.is_set():
            start = counter % total
            if start == 0 and counter > 0 and not self.loop:
                break
            counter += n
            if self.realtime:
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            now = time.monotonic_ns()
            idx = np.arange(counter - n, counter, dtype=np.int64)
            t_ns = now - (n - 1 - np.arange(n, dtype=np.int64)) * int(1e9 / SAMPLE_RATE_HZ)
            while not self._stop.is_set():
                try:
                    self.blocks.put(SampleBlock(self.values[start:start + n].copy(), idx, t_ns), timeout=0.5)
                    break
                except queue.Full:
                    continue
        self.finished.set()
        self.events.put(("replay_done", str(self.path)))

    def close(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------------------------------------
# Server


class FrameServer:
    """Broadcast newline-JSON to every client; hand inbound commands to ``on_command``."""

    def __init__(self, host: str, port: int, on_command, greeting, on_query=None) -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, port))
        self._server.listen(4)
        self._on_command, self._greeting = on_command, greeting
        # on_query(message) -> messages sent back to the asking client only (e.g. the neutral recording)
        self._on_query = on_query
        self._clients: list[socket.socket] = []
        # clients that asked for the raw EMG stream (a recorder), a subset of _clients
        self._raw: set[socket.socket] = set()
        self._lock = threading.Lock()
        self._closed = False
        threading.Thread(target=self._accept, daemon=True, name="dexemg-accept").start()

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    @property
    def raw_client_count(self) -> int:
        with self._lock:
            return len(self._raw)

    def _accept(self) -> None:
        while not self._closed:
            try:
                conn, _addr = self._server.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                for message in self._greeting():
                    conn.sendall(encode(message))
            except OSError:
                conn.close()
                continue
            with self._lock:
                self._clients.append(conn)
            threading.Thread(target=self._read, args=(conn,), daemon=True, name="dexemg-read").start()

    def _read(self, conn: socket.socket) -> None:
        buffer = b""
        while not self._closed:
            try:
                chunk = conn.recv(65536)
            except OSError:
                return
            if not chunk:
                return
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if line.strip():
                    try:
                        message = decode(line)
                    except ValueError as error:
                        print(f"[bridge] ignoring malformed client line: {error}", flush=True)
                        continue
                    if message["type"] != "cmd":
                        continue
                    action = message.get("action")
                    if action == "subscribe_raw":
                        with self._lock:
                            self._raw.add(conn)
                        print("[bridge] a recorder subscribed to the raw EMG stream", flush=True)
                    elif action == "get_neutral" and self._on_query is not None:
                        try:
                            for reply in self._on_query(message):
                                conn.sendall(encode(reply))
                        except OSError:
                            return
                    else:
                        self._on_command(message)

    def broadcast(self, message: dict, *, raw_only: bool = False) -> None:
        line = encode(message)
        with self._lock:
            alive = []
            for conn in self._clients:
                if raw_only and conn not in self._raw:
                    alive.append(conn)
                    continue
                try:
                    conn.sendall(line)
                    alive.append(conn)
                except OSError:
                    conn.close()
                    self._raw.discard(conn)
            self._clients = alive

    def broadcast_raw(self, message: dict) -> None:
        """Send to the clients that subscribed to the raw EMG stream only."""

        self.broadcast(message, raw_only=True)

    def close(self) -> None:
        self._closed = True
        with self._lock:
            for conn in self._clients:
                conn.close()
            self._clients = []
            self._raw = set()
        self._server.close()


# ---------------------------------------------------------------------------------------------------------
# Engine


class BridgeEngine:
    """Neutral calibration, chunking, decoding, landmarks, frame emission, logging."""

    def __init__(self, decoder: StreamingDecoder, landmarker: HandLandmarker, *, emit_hz: float,
                 neutral_seconds: float, log: bool, reject_ratio: float | None = SCALE_RATIO_REJECT,
                 record_only: bool = False) -> None:
        # record_only: own the band and pass the raw samples on to recorders; nothing is decoded and no
        # hand is sent (decoder is a RecordOnlyDecoder, landmarker None). A neutral can still be recorded.
        self.record_only = bool(record_only)
        self.reject_ratio = reject_ratio
        self.neutral_rejection: str | None = None
        self.decoder, self.landmarker = decoder, landmarker
        self.decimation = max(1, int(round(SAMPLE_RATE_HZ / emit_hz)))
        self.neutral_samples = int(round(neutral_seconds * SAMPLE_RATE_HZ))
        self.state = "connecting"
        self.message = ""
        self.scale: np.ndarray | None = None
        self.scale_ratio: list[float] | None = None
        self.scale_warning: str | None = None
        self.calibration_started_ns: int | None = None
        self._calib_start_ns: int | None = None
        self._calib: list[np.ndarray] = []
        self._calib_collected = 0
        self._pending_values: list[np.ndarray] = []
        self._pending_t: list[np.ndarray] = []
        self._pending_idx: list[np.ndarray] = []
        self._pending_n = 0
        self._sample_counter = 0
        self.seq = 0
        self.lost = 0
        self.gaps = 0
        self._last_index: int | None = None
        self.decode_ms: list[float] = []
        self._rate_window: list[tuple[int, int]] = []
        self.log = log
        self.log_blocks: list[SampleBlock] = []
        self.log_frames: list[tuple[int, int, int, np.ndarray]] = []
        self.neutral_history: list[dict] = []
        self.emit = lambda message: None
        self.connected = False
        self.band: dict = {}
        self.clients = lambda: 0
        self.recorders = lambda: 0           # clients subscribed to the raw EMG stream
        self.emit_raw = None                 # callable(SampleBlock): raw packets to recorders
        self.emit_neutral = None             # callable(): the latest neutral to recorders
        self.emit_impedance = None           # callable(dict): an impedance reading to recorders
        self.impedance: list[float] | None = None   # ohm per channel, latest impedance packet
        self.impedance_ns: int | None = None
        self._recent: list[np.ndarray] = []  # last ~0.5 s of raw samples, for the per-channel level display
        self._recent_n = 0

    # -- calibration
    def _idle_state(self) -> str:
        """The state of a connected bridge that is not calibrating."""

        if self.record_only:
            return "recording_only"
        return "waiting_for_neutral" if self.scale is None else "streaming"

    def request_neutral(self, seconds: float | None = None, countdown: float = COUNTDOWN_SECONDS) -> None:
        if seconds is not None:
            self.neutral_samples = int(round(float(seconds) * SAMPLE_RATE_HZ))
        self._calib = []
        self._calib_collected = 0
        self._calib_start_ns = time.monotonic_ns() + int(countdown * 1e9)
        self.state = "calibrating"
        self.message = (f"neutral in {countdown:.0f} s: hold the neutral pose (as in the HandSync neutral takes) "
                        f"for {self.neutral_samples / SAMPLE_RATE_HZ:.0f} s")
        print(f"[bridge] {self.message}", flush=True)

    def use_reference_scale(self) -> None:
        """Fallback: the median of the training neutrals instead of a neutral of this wearing."""

        if self.decoder.reference_scale is None:
            self.message = "this checkpoint carries no training reference scale"
            return
        self._calib_start_ns = None
        self._set_scale(self.decoder.reference_scale, "training reference", None)

    def load_neutral(self, samples: np.ndarray, origin: str) -> None:
        self._set_scale(neutral_channel_scale(samples), origin, samples)

    def _set_scale(self, scale: np.ndarray, origin: str, samples: np.ndarray | None) -> None:
        scale = np.asarray(scale, np.float32)
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            self.message = f"neutral rejected: non-positive channel std {scale.round(3).tolist()}"
            print(f"[bridge] {self.message}", flush=True)
            self.state = self._idle_state()
            return
        reference = self.decoder.reference_scale
        if reference is not None and self.reject_ratio is not None and origin != "training reference":
            ratio = scale / reference
            bad = [i for i, r in enumerate(ratio) if r > self.reject_ratio or r < 1.0 / self.reject_ratio]
            if bad:
                self.neutral_rejection = (
                    f"neutral REJECTED: channels {bad} are {', '.join(f'{ratio[i]:.1f}x' for i in bad)} the "
                    "training neutrals (allowed up to "
                    f"{self.reject_ratio:g}x). The forearm was not relaxed or the band moved: rest the arm as "
                    "in the HandSync neutral takes, wait until every channel bar reads 'relaxed', and record again"
                )
                self.message = self.neutral_rejection + ("; the previous neutral stays active" if self.scale is not None else "")
                print(f"[bridge] {self.message} (ratio {ratio.round(2).tolist()})", flush=True)
                self.neutral_history.append({"time": _dt.datetime.now().isoformat(timespec="seconds"), "origin": origin,
                                             "scale": scale.tolist(), "ratio_to_training": ratio.round(3).tolist(),
                                             "rejected": True, "samples": None if samples is None else int(len(samples))})
                if samples is not None:
                    self.neutral_history[-1]["raw"] = np.asarray(samples, np.float32)
                if self.emit_neutral is not None:
                    self.emit_neutral()
                self.state = self._idle_state()
                return
        self.neutral_rejection = None
        self.scale = scale
        self.decoder.set_scale(scale)
        warning = None
        if self.decoder.reference_scale is not None:
            ratio = scale / self.decoder.reference_scale
            self.scale_ratio = ratio.round(3).tolist()
            bad = [i for i, r in enumerate(ratio) if r > SCALE_RATIO_WARN or r < 1 / SCALE_RATIO_WARN]
            if bad:
                warning = (f"channels {bad} differ more than {SCALE_RATIO_WARN:g}x from the training neutrals "
                           f"(ratio {ratio.round(2).tolist()}): check band contact / placement")
        self.scale_warning = warning
        self.neutral_history.append({"time": _dt.datetime.now().isoformat(timespec="seconds"), "origin": origin,
                                     "scale": scale.tolist(), "ratio_to_training": self.scale_ratio,
                                     "warning": warning,
                                     "samples": None if samples is None else int(len(samples))})
        if samples is not None:
            self.neutral_history[-1]["raw"] = np.asarray(samples, np.float32)
        if self.emit_neutral is not None:
            self.emit_neutral()
        self.state = self._idle_state()
        self.message = (f"neutral recorded ({origin}); record-only, nothing is decoded" if self.record_only
                        else f"neutral set from {origin}; streaming" + (f" -- WARNING: {warning}" if warning else ""))
        print(f"[bridge] channel scale {scale.round(3).tolist()}", flush=True)
        if self.scale_ratio is not None:
            print(f"[bridge] ratio to the training neutrals {np.round(self.scale_ratio, 2).tolist()}", flush=True)
        print(f"[bridge] {self.message}", flush=True)

    # -- samples
    def set_impedance(self, values, received_ns: int) -> None:
        """A new electrode impedance reading (ohm per channel): kept for the status and passed on to recorders."""

        self.impedance = [float(v) for v in values]
        self.impedance_ns = int(received_ns)
        if self.emit_impedance is not None:
            self.emit_impedance({"type": "impedance", "values": self.impedance, "received_ns": self.impedance_ns})

    def raw_message(self, block: SampleBlock) -> dict:
        """One received packet for a recorder: raw microvolts with the device counter and host times."""

        return {"type": "emg", "sample_index": block.sample_index.tolist(), "t_ns": block.t_ns.tolist(),
                "received_ns": int(block.t_ns[-1]) if block.received_ns is None else int(block.received_ns),
                "values": block.values.tolist(),
                "is_lost": None if block.lost_mask is None else block.lost_mask.tolist()}

    def neutral_messages(self, latest_only: bool = False) -> list[dict]:
        """Every neutral of this session (accepted or rejected) with its raw samples, for a recorder."""

        entries = list(enumerate(self.neutral_history))
        if latest_only:
            entries = entries[-1:]
        messages = []
        for index, entry in entries:
            raw = entry.get("raw")
            messages.append({"type": "neutral", "index": index, "time": entry.get("time"), "origin": entry.get("origin"),
                             "rejected": bool(entry.get("rejected", False)), "scale": entry.get("scale"),
                             "ratio_to_training": entry.get("ratio_to_training"),
                             "active": (not entry.get("rejected", False)) and index == self._active_neutral_index(),
                             "trim_seconds": NEUTRAL_TRIM_SECONDS, "sample_rate_hz": SAMPLE_RATE_HZ,
                             "samples": None if raw is None else np.asarray(raw, np.float32).tolist()})
        return messages

    def _active_neutral_index(self) -> int | None:
        accepted = [i for i, entry in enumerate(self.neutral_history) if not entry.get("rejected")]
        return accepted[-1] if accepted else None

    def on_block(self, block: SampleBlock) -> None:
        if self.emit_raw is not None:
            self.emit_raw(block)
        if self.log:
            self.log_blocks.append(block)
        self._recent.append(block.values)
        self._recent_n += len(block.values)
        while self._recent_n - len(self._recent[0]) >= SAMPLE_RATE_HZ // 2:
            self._recent_n -= len(self._recent.pop(0))
        self.lost += block.lost
        if self._last_index is not None and block.sample_index[0] != self._last_index + 1:
            self.gaps += 1
        self._last_index = int(block.sample_index[-1])
        self._rate_window.append((int(block.t_ns[-1]), len(block.values)))
        if len(self._rate_window) > 400:
            del self._rate_window[:200]
        if self.state == "connecting":
            self.state = self._idle_state()
        if self._calib_start_ns is not None:
            keep = block.t_ns >= self._calib_start_ns
            if keep.any():
                if self._calib_collected == 0:
                    print("[bridge] recording neutral now -- hold still", flush=True)
                self._calib.append(block.values[keep])
                self._calib_collected += int(keep.sum())
                if self._calib_collected >= self.neutral_samples:
                    samples = np.concatenate(self._calib)[: self.neutral_samples]
                    self._calib_start_ns = None
                    self._calib = []
                    self.load_neutral(samples, "live neutral recording")
        if self.scale is None or self.record_only:
            return
        self._pending_values.append(block.values)
        self._pending_t.append(block.t_ns)
        self._pending_idx.append(block.sample_index)
        self._pending_n += len(block.values)
        chunk = self.decoder.chunk_steps
        while self._pending_n >= chunk:
            values = np.concatenate(self._pending_values)
            t_ns = np.concatenate(self._pending_t)
            idx = np.concatenate(self._pending_idx)
            self._pending_values, self._pending_t, self._pending_idx = [values[chunk:]], [t_ns[chunk:]], [idx[chunk:]]
            self._pending_n -= chunk
            self._decode(values[:chunk], t_ns[:chunk], idx[:chunk])

    def _decode(self, values: np.ndarray, t_ns: np.ndarray, idx: np.ndarray) -> None:
        t0 = time.perf_counter()
        angles = self.decoder.feed(values)
        positions = np.arange(self._sample_counter, self._sample_counter + len(values))
        self._sample_counter += len(values)
        pick = np.flatnonzero(positions % self.decimation == self.decimation - 1)
        landmarks = self.landmarker(angles[pick]) if len(pick) else np.zeros((0, 21, 3))
        self.decode_ms.append((time.perf_counter() - t0) * 1e3)
        if len(self.decode_ms) > 600:
            del self.decode_ms[:300]
        emit_ns = time.monotonic_ns()
        for k, i in enumerate(pick):
            self.seq += 1
            self.emit({"type": "frame", "seq": self.seq, "timestamp_ns": int(t_ns[i]), "emit_ns": emit_ns,
                       "sample_index": int(idx[i]), "angles": np.round(angles[i], 6).tolist(),
                       "landmarks": np.round(landmarks[k], 6).tolist()})
        if self.log:
            self.log_frames.append((int(t_ns[-1]), self.seq, emit_ns, angles))

    # -- status
    def rate_hz(self) -> float | None:
        if len(self._rate_window) < 2:
            return None
        span = (self._rate_window[-1][0] - self._rate_window[0][0]) / 1e9
        count = sum(n for _, n in self._rate_window[1:])
        return count / span if span > 0 else None

    def status(self) -> dict:
        progress = None
        if self._calib_start_ns is not None:
            progress = {"collected": self._calib_collected, "target": self.neutral_samples,
                        "starts_in_s": max(0.0, (self._calib_start_ns - time.monotonic_ns()) / 1e9)}
        rms = None
        if self.connected and self._recent:
            recent = np.concatenate(self._recent)
            rms = (recent - recent.mean(axis=0, keepdims=True)).std(axis=0).round(3).tolist()
        return {"type": "status", "state": self.state, "message": self.message, "calibration": progress,
                "connected": self.connected, "band": self.band, "clients": self.clients(),
                "recorders": self.recorders(), "record_only": self.record_only,
                "impedance_ohm": self.impedance,
                "impedance_age_s": None if self.impedance_ns is None else (time.monotonic_ns() - self.impedance_ns) / 1e9,
                "channel_rms_uv": rms,
                "neutral_recorded": next((n["time"] + (" (training reference)" if n["origin"] == "training reference" else "")
                                          for n in reversed(self.neutral_history) if not n.get("rejected")), None),
                "neutral_rejection": self.neutral_rejection,
                "reference_scale": None if self.decoder.reference_scale is None else self.decoder.reference_scale.tolist(),
                "channel_scale": None if self.scale is None else self.scale.tolist(),
                "scale_ratio_to_training": self.scale_ratio, "scale_warning": self.scale_warning,
                "rate_hz": self.rate_hz(), "lost_samples": self.lost, "index_gaps": self.gaps,
                "frames": self.seq,
                "decode_ms_median": float(np.median(self.decode_ms)) if self.decode_ms else None}


def _write_log(directory: Path, engine: BridgeEngine, meta: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if engine.log_blocks:
        np.savez_compressed(directory / "emg.npz",
                            values=np.concatenate([b.values for b in engine.log_blocks]),
                            sample_index=np.concatenate([b.sample_index for b in engine.log_blocks]),
                            t_ns=np.concatenate([b.t_ns for b in engine.log_blocks]))
    if engine.log_frames:
        np.savez_compressed(directory / "decoded.npz",
                            chunk_end_t_ns=np.asarray([f[0] for f in engine.log_frames], np.int64),
                            last_seq=np.asarray([f[1] for f in engine.log_frames], np.int64),
                            emit_ns=np.asarray([f[2] for f in engine.log_frames], np.int64),
                            angles=np.stack([f[3] for f in engine.log_frames]).astype(np.float32))
    neutrals = []
    for i, entry in enumerate(engine.neutral_history):
        entry = dict(entry)
        raw = entry.pop("raw", None)
        if raw is not None:
            np.save(directory / f"neutral_{i}.npy", raw)
        neutrals.append(entry)
    (directory / "session.json").write_text(json.dumps({**meta, "neutrals": neutrals, "final_status": engine.status()},
                                                       indent=1, default=str))
    print(f"[bridge] log written to {directory}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, help="wrist-band dexemg checkpoint (fixed-scale); required unless --record-only")
    parser.add_argument("--record-only", action="store_true",
                        help="own the band and pass its raw EMG on to a recording launcher (--emg-bridge) without "
                             "decoding: no checkpoint, no model, no hand for --hand-source dexemg. A neutral can still "
                             "be recorded and is saved with the recording, but is not checked against the training neutrals")
    parser.add_argument("--device", default="cpu",
                        help="torch device (default cpu: ~10 ms per 100 ms chunk and no contention with Isaac Sim's GPU)")
    parser.add_argument("--band", default=DEFAULT_BAND, help=f"band address or name substring (default {DEFAULT_BAND})")
    parser.add_argument("--adapter", default=None,
                        help="Bluetooth adapter, e.g. hci1 (default: every adapter is scanned until the band answers)")
    parser.add_argument("--sdk-path", type=Path, default=default_sdk_path(),
                        help="Directory holding the Synchroni SDK to import (default: the 0.9.11 build HandSync records with, %(default)s)")
    parser.add_argument("--sdk-log", action="store_true",
                        help="keep the SDK's own log files (~/Documents/sensorsdklog, one of about 1 MB per connection "
                             "attempt); off by default since a bridge waiting for the band wrote 0.47 GB of them an hour")
    parser.add_argument("--scan-ms", type=int, default=5000)
    parser.add_argument("--package-count", type=int, default=DEFAULT_PACKAGE_COUNT,
                        help="samples per BLE packet (8 = HandSync's; larger adds latency)")
    parser.add_argument("--neutral-seconds", type=float, default=DEFAULT_NEUTRAL_SECONDS)
    parser.add_argument("--neutral-max-ratio", type=float, default=SCALE_RATIO_REJECT,
                        help="reject a neutral whose channel std exceeds this multiple of the training neutrals' "
                             "(default %(default)s; 0 accepts anything)")
    parser.add_argument("--no-neutral-on-start", action="store_true",
                        help="do not start the neutral recording automatically after connecting")
    parser.add_argument("--neutral-npz", type=Path, default=None,
                        help="take the neutral from an exported recording (its wrist channels) instead of recording it")
    parser.add_argument("--chunk-ms", type=float, default=100.0)
    parser.add_argument("--overlap-ms", type=float, default=50.0)
    parser.add_argument("--emit-hz", type=float, default=50.0, help="landmark frames per second sent to the simulator")
    parser.add_argument("--smoothing", type=float, default=0.0, help="EMA on the sent pose in [0, 1); 0 = off")
    parser.add_argument("--threads", type=int, default=DEFAULT_DECODER_THREADS,
                        help="CPU threads for the decoder (default %(default)s; 0 = PyTorch's own default, one per core). "
                             "With PyTorch's default the idle threads spin between the 100 ms chunks and the bridge "
                             "holds about 13 cores (2026-11-18), which slows the simulator next to it")
    parser.add_argument("--deadband-deg", type=float, default=0.0,
                        help="per-joint deadband on the sent pose, degrees (0 = off): a decoded joint angle that "
                             "wanders by less than this stays still; real motion follows from this far behind. On the "
                             "2026-11-18 EMG teleop takes 4 deg made the typical wander of a still hand 0 (from 3.3 deg "
                             "within 0.5 s) without added delay; larger slow drifts of the decoder remain")
    parser.add_argument("--hand-model", type=Path, default=None,
                        help="glove-fit hand model (JSON from scripts/fit_glove_hand_model.py) that turns the decoded "
                             "angles into the landmarks the operator's glove would give; default: emg2pose's generic "
                             "hand, whose thumb sits 19-91 mm from the glove's for the same angles")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--replay-npz", type=Path, default=None,
                        help="replay an exported take instead of the band (testing; pair with --neutral-npz)")
    parser.add_argument("--realtime", action="store_true", help="pace --replay-npz at 500 Hz")
    parser.add_argument("--exit-after-replay", action="store_true")
    parser.add_argument("--loop", action="store_true", help="repeat --replay-npz forever (demos, smoke tests)")
    parser.add_argument("--log-dir", type=Path, default=Path.home() / "dexemg_teleop_logs",
                        help="raw EMG, decoded angles and neutrals are saved under <log-dir>/<timestamp>/ on exit")
    parser.add_argument("--no-log", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    hand_model = {"type": "generic", "file": None}
    if args.replay_npz is not None and args.neutral_npz is None:
        raise SystemExit("--replay-npz needs --neutral-npz (the take's own neutral)")
    if not 0.0 <= args.smoothing < 1.0:
        raise SystemExit("--smoothing must be in [0, 1)")
    if args.threads < 0:
        raise SystemExit("--threads must be 0 (PyTorch default) or positive")
    if not 0.0 <= args.deadband_deg <= 45.0:
        raise SystemExit("--deadband-deg must be between 0 and 45")
    chunk = int(round(args.chunk_ms / 1000 * SAMPLE_RATE_HZ))
    overlap = int(round(args.overlap_ms / 1000 * SAMPLE_RATE_HZ))
    if chunk <= 0 or overlap < 0:
        raise SystemExit("--chunk-ms must be positive and --overlap-ms non-negative")
    device = args.device
    if args.record_only:
        if args.checkpoint is not None:
            raise SystemExit("--record-only does not decode: drop --checkpoint (or drop --record-only)")
        if args.hand_model is not None:
            raise SystemExit("--record-only does not decode: drop --hand-model (or drop --record-only)")
        print("[bridge] record-only: no model is loaded and no hand is decoded", flush=True)
        decoder, landmarker = RecordOnlyDecoder(chunk), None
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required (or use --record-only to pass the raw EMG on without decoding)")
        if args.threads > 0:  # before torch starts its thread pool
            for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ[variable] = str(args.threads)
            # Idle decoder threads sleep instead of spinning until the next chunk (measured 2026-11-18 with two
            # threads: 19% of one core and 12.7 ms per chunk, against 115% and 14.3 ms when they spin).
            for variable, value in (("OMP_WAIT_POLICY", "PASSIVE"), ("KMP_BLOCKTIME", "0"), ("GOMP_SPINCOUNT", "0")):
                os.environ.setdefault(variable, value)
        torch, _ = _import_dexemg()
        if args.threads > 0:
            torch.set_num_threads(args.threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass  # already started by an earlier import; the intra-op limit above is the one that matters
        print(f"[bridge] decoder threads: {torch.get_num_threads()}"
              + ("" if args.threads > 0 else " (PyTorch default)"), flush=True)
        print(f"[bridge] loading {args.checkpoint} on {device}", flush=True)
        decoder = StreamingDecoder(args.checkpoint, device, chunk, overlap, args.smoothing,
                                   deadband_rad=math.radians(args.deadband_deg))
        if args.deadband_deg > 0:
            print(f"[bridge] deadband: {args.deadband_deg:g} deg per joint on the sent pose", flush=True)
        if args.hand_model is not None:
            from dex_teleop.emg.hand_model import GloveFitHandModel

            landmarker = GloveFitHandModel.load(args.hand_model.expanduser())
            fitted = landmarker.metadata
            hand_model = {"type": "glove_fit", "file": args.hand_model.name,
                          **{k: fitted.get(k) for k in ("fitted", "operator", "frames", "held_out_landmark_error_mm_mean")}}
            print(f"[bridge] hand model: GLOVE FIT {args.hand_model.name} (fitted {fitted.get('fitted')}, "
                  f"{fitted.get('frames')} glove frames, operator {fitted.get('operator') or 'not recorded'})", flush=True)
        else:
            landmarker = HandLandmarker(device)
            print("[bridge] hand model: generic emg2pose hand (pass --hand-model for the glove-fit one)", flush=True)
    engine = BridgeEngine(decoder, landmarker, emit_hz=args.emit_hz, neutral_seconds=args.neutral_seconds,
                          log=not args.no_log, reject_ratio=args.neutral_max_ratio or None,
                          record_only=args.record_only)
    blocks: "queue.Queue[SampleBlock]" = queue.Queue(maxsize=4096)
    events: "queue.Queue[tuple[str, str]]" = queue.Queue()
    commands: "queue.Queue[dict]" = queue.Queue()
    started = _dt.datetime.now()
    meta = {"protocol": PROTOCOL, "started": started.isoformat(timespec="seconds"),
            "checkpoint": None if args.checkpoint is None else str(args.checkpoint.resolve()),
            "record_only": args.record_only, "model": decoder.metadata, "device": device, "hand_model": hand_model,
            "chunk_steps": chunk, "overlap_steps": overlap, "emit_hz": SAMPLE_RATE_HZ / engine.decimation,
            "smoothing": args.smoothing, "deadband_deg": args.deadband_deg, "neutral_seconds": args.neutral_seconds,
            "neutral_trim_seconds": NEUTRAL_TRIM_SECONDS, "firmware_filters": FIRMWARE_FILTERS,
            "reference_wrist_scale": None if decoder.reference_scale is None else decoder.reference_scale.tolist()}

    def greeting():
        return [{"type": "hello", "protocol": PROTOCOL, "landmark_names": list(LANDMARK_NAMES),
                 "schema": LANDMARK_SCHEMA, "coordinate_frame": COORDINATE_FRAME, "handedness": "right",
                 "units": "m", "sample_rate_hz": SAMPLE_RATE_HZ, **{k: v for k, v in meta.items() if k != "protocol"}},
                engine.status()]

    server = FrameServer(args.host, args.port, commands.put, greeting, on_query=lambda _message: engine.neutral_messages())
    engine.emit = server.broadcast
    engine.recorders = lambda: server.raw_client_count
    engine.emit_raw = lambda block: server.broadcast_raw(engine.raw_message(block)) if server.raw_client_count else None
    engine.emit_impedance = lambda message: server.broadcast_raw(message) if server.raw_client_count else None
    engine.emit_neutral = lambda: (
        [server.broadcast_raw(message) for message in engine.neutral_messages(latest_only=True)]
        if server.raw_client_count
        else None
    )
    print(f"[bridge] serving {PROTOCOL} on {args.host}:{args.port}", flush=True)
    if args.neutral_npz is not None:
        with np.load(args.neutral_npz) as data:
            engine.load_neutral(np.asarray(data["emg"], np.float32)[:, :CHANNELS], f"file {args.neutral_npz.name}")
    if args.replay_npz is not None:
        acquisition = ReplayAcquisition(blocks, events, path=args.replay_npz, realtime=args.realtime,
                                        package_count=args.package_count, loop=args.loop)
    else:
        acquisition = BandAcquisition(blocks, events, device=args.band, adapter=args.adapter,
                                      scan_ms=args.scan_ms, package_count=args.package_count,
                                      sdk_path=args.sdk_path, sdk_log=args.sdk_log)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    def keyboard() -> None:
        for line in sys.stdin:
            if line.strip().lower() in {"q", "quit", "exit"}:
                stop.set()
                return
            commands.put({"type": "cmd", "action": "calibrate_neutral"})

    if sys.stdin is not None and sys.stdin.isatty():
        threading.Thread(target=keyboard, daemon=True, name="dexemg-keys").start()
    exit_code = 0
    live = args.replay_npz is None
    engine.clients = lambda: server.client_count
    link = {"thread": None, "up": False, "first": True}

    def connect() -> None:
        """Connect (and keep retrying) off the main loop, so status keeps flowing to the GUI/simulator."""
        while not stop.is_set():
            try:
                acquisition.start()
            except BaseException as error:  # noqa: BLE001 - reported and retried
                engine.message = f"not connected: {error}; retrying in {RECONNECT_SECONDS:g} s"
                print(f"[bridge] {engine.message}", flush=True)
                close_with_timeout(acquisition)
                kill_child_processes()
                stop.wait(RECONNECT_SECONDS)
                continue
            meta["acquisition"] = acquisition.metadata
            engine.band = {k: acquisition.metadata.get(k) for k in
                           ("device_name", "device_address", "ble_adapter", "firmware_version", "firmware_filters")}
            link["up"] = True
            return

    def start_connecting() -> None:
        engine.connected = False
        if engine.scale is None:
            engine.state = "connecting"
        link["up"] = False
        link["thread"] = threading.Thread(target=connect, daemon=True, name="dexemg-connect")
        link["thread"].start()

    try:
        if live:
            start_connecting()
        else:
            acquisition.start()
            meta["acquisition"] = acquisition.metadata
            engine.connected = True
        last_status = 0.0
        while not stop.is_set():
            if live and link["up"] and not engine.connected:
                engine.connected = True
                print(f"[bridge] connected: {engine.band}", flush=True)
                if link["first"]:
                    link["first"] = False
                    if engine.scale is None and not args.no_neutral_on_start and not engine.record_only:
                        engine.request_neutral()
                    else:
                        engine.message = ("connected; record-only" if engine.record_only else
                                          "connected; no neutral yet" if engine.scale is None else "connected; streaming")
                else:
                    engine.state = engine._idle_state()
                    engine.message = ("reconnected (record-only). If the band was taken off or moved, record a new neutral"
                                      if engine.record_only else
                                      "reconnected. If the band was taken off or moved, record the neutral again"
                                      if engine.scale is not None else "reconnected; record the neutral")
                    print(f"[bridge] {engine.message}", flush=True)
            impedance = getattr(acquisition, "latest_impedance", None)
            if impedance is not None and impedance[1] != engine.impedance_ns:
                engine.set_impedance(*impedance)
            try:
                engine.on_block(blocks.get(timeout=0.05))
                while True:
                    engine.on_block(blocks.get_nowait())
            except queue.Empty:
                pass
            while not commands.empty():
                command = commands.get_nowait()
                if command.get("action") == "calibrate_neutral":
                    if engine.connected:
                        engine.request_neutral(command.get("seconds"))
                    else:
                        engine.message = "cannot record the neutral: the band is not connected"
                elif command.get("action") == "use_reference_scale":
                    engine.use_reference_scale()
                elif command.get("action") == "quit":
                    stop.set()
            while not events.empty():
                kind, text = events.get_nowait()
                if kind == "error":
                    if not live:
                        raise RuntimeError(text)
                    if engine.connected:  # lost the band mid-stream: say so, tear down, reconnect
                        print(f"[bridge] connection lost: {text}", flush=True)
                        engine.message = f"connection lost ({text}); reconnecting"
                        engine._calib_start_ns = None
                        close_with_timeout(acquisition)
                        kill_child_processes()
                        start_connecting()
                        engine.state = "connecting"
                elif kind == "replay_done":
                    print(f"[bridge] replay finished: {engine.seq} frames", flush=True)
                    if args.exit_after_replay:
                        while not blocks.empty():
                            engine.on_block(blocks.get_nowait())
                        stop.set()
                else:
                    engine.message = text
                    print(f"[bridge] {text}", flush=True)
            now = time.monotonic()
            if now - last_status >= 0.25:
                last_status = now
                server.broadcast(engine.status())
    except Exception as error:  # noqa: BLE001 - report to the client, then exit nonzero
        exit_code = 1
        engine.state = "error"
        engine.message = f"{type(error).__name__}: {error}"
        print(f"[bridge] ERROR {engine.message}", flush=True)
        server.broadcast({"type": "error", "message": engine.message})
        server.broadcast(engine.status())
    finally:
        stop.set()
        if not close_with_timeout(acquisition):
            print("[bridge] the Bluetooth teardown did not finish in time; exiting anyway", flush=True)
        # Before the log is written: the window kills a bridge that takes more than 12 s to stop, and the SDK's
        # BLE host processes it would leave behind hold a copy of the listening socket (2026-11-20: two of them
        # kept port 8765 after the bridge was gone).
        kill_child_processes()
        server.close()
        if not args.no_log:
            _write_log(args.log_dir / started.strftime("%Y%m%d_%H%M%S"), engine, meta)
    sys.stdout.flush()
    # The SDK's BLE host can leave non-daemon threads/processes behind; never hang on exit and
    # never leave them holding the Bluetooth adapter or the listening port.
    kill_child_processes()
    os._exit(exit_code)


if __name__ == "__main__":
    sys.exit(main())
