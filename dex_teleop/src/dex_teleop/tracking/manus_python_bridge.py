"""MANUS acquisition sidecar on the official MANUS Core 3.2 Python SDK.

This is the pure-Python replacement for ``native/manus_bridge/manus_bridge.cpp``.
It runs as a child process of :class:`dex_teleop.tracking.manus.ManusSource`
and emits the same versioned JSON-lines protocol (``dex_teleop.manus``) on a
dedicated descriptor, so every validation, pairing, provenance and recording
rule in ``manus.py`` applies unchanged. Nothing is compiled: the SDK's
pre-built CFFI extension (``manus_sdk``, shipped in the ``Python`` folder of
``MANUS_Core_3.2.x_SDK_Linux.tar.gz``) is imported from the SDK root.

Modes:

- ``integrated`` (the Linux default): the SDK embeds MANUS Core, talks to the
  glove dongle itself and needs no Windows machine. The global wrist comes
  from a VIVE Ultimate Tracker streamed by VIVE Hub for Linux
  (:mod:`dex_teleop.tracking.vive_ultimate`); the bridge injects that pose into
  Core as a custom tracker (``CoreSdk_SendDataForTrackers``) and, by default,
  also composes the raw-skeleton wrist from it directly so the published hand
  is tracker-driven regardless of how embedded Core treats injected trackers.
- ``remote``: connects to a MANUS Core 3.2 running on Windows, like the
  historical C++ bridge; Core then supplies the tracker-driven wrist.

Protocol version 3 adds to version 2: an inbound JSON-lines command channel
on stdin (``{"type": "vibrate", "side": "right", "powers": [thumb..pinky]}``)
that drives Metaglove Pro Haptic actuators through
``CoreSdk_VibrateFingersForGlove``, tracker packets whose ``tracking_system`` is
``vive_hub_linux`` for poses the bridge injected itself, and
``wrist_composition``/``required_glove_user_id`` connection metadata.

Powered by MANUS. The proprietary SDK is supplied separately and is not
redistributed with this repository.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from dex_teleop.tracking.vive_ultimate import (
    VUT_DEFAULT_FRAME,
    VUT_DEFAULT_SOCKET,
    VUT_FRAMES,
    VUT_TRACKING_SYSTEM,
    VutPoseFeed,
    VutTrackerPose,
    vut_pose_to_core_world,
)
from dex_teleop.tracking.transforms import (
    RigidTransform,
    compose_transforms,
    inverse_transform,
)


PROTOCOL = "dex_teleop.manus"
PROTOCOL_VERSION = 3
CORE_WORLD_FRAME = "manus_core_world_y_up_rh_z_to_viewer_m"
MODES = ("integrated", "remote")
HAND_MOTIONS = ("auto", "tracker", "tracker_rotation_only", "imu", "none")
WRIST_COMPOSITIONS = ("bridge", "core")
TRACKER_FEEDS = ("none", "vive_ultimate")
# The MANUS Universal Mount's documented example tracker-to-casing offset for a
# left-hand tracker (meters, client coordinate system; x mirrored for the right
# hand). Replace with the values MANUS Core shows for the HTC VIVE Ultimate
# Tracker preset of your mount when they are available.
DEFAULT_TRACKER_TO_CASING_TRANSLATION = {
    "left": (0.003, -0.058, -0.043),
    "right": (-0.003, -0.058, -0.043),
}
SDK_PYTHON_PATH_ENVIRONMENT = "DEX_TELEOP_MANUS_PYTHON_SDK_PATH"
LOGGER = logging.getLogger("manus_python_bridge")


# ---------------------------------------------------------------------------
# SDK loading


def load_manus_sdk(explicit_path: str | None = None):
    """Import ``manus_sdk`` (lib, ffi, wrapper, enums), optionally from a folder.

    The SDK's ``Python`` folder (containing ``manus_sdk/``) may be given
    explicitly, through ``DEX_TELEOP_MANUS_PYTHON_SDK_PATH``, or the package
    may simply be installed in this interpreter (``pip install <SDK>/Python``).
    """

    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    environment = os.environ.get(SDK_PYTHON_PATH_ENVIRONMENT)
    if environment:
        candidates.append(Path(environment).expanduser())
    for candidate in candidates:
        package = candidate / "manus_sdk" / "__init__.py"
        if not package.is_file():
            raise ImportError(
                f"MANUS Python SDK folder {candidate} does not contain manus_sdk/__init__.py"
            )
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    import manus_sdk  # noqa: PLC0415 - optional proprietary dependency
    from manus_sdk import ManusSDK, ffi, lib  # noqa: PLC0415
    from manus_sdk.generated import _enums as enums  # noqa: PLC0415

    if lib is None or ffi is None:
        raise ImportError(
            "manus_sdk imported but its CFFI extension did not load; install cffi "
            "and use a Python version the SDK ships binaries for (3.9-3.14)"
        )
    return manus_sdk, lib, ffi, ManusSDK, enums


# ---------------------------------------------------------------------------
# Options


def _parse_timeout(value: str, option: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{option} requires an integer in [0, 3600]") from error
    if not 0 <= parsed <= 3600:
        raise argparse.ArgumentTypeError(f"{option} requires an integer in [0, 3600]")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dex_teleop.tracking.manus_python_bridge",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--protocol-fd", type=int, default=1, help="JSON-lines output descriptor")
    parser.add_argument("--protocol-version", action="store_true", help="Print the protocol version and exit")
    parser.add_argument("--mode", choices=MODES, default="integrated")
    parser.add_argument("--sdk-python-path", help="SDK 'Python' folder containing manus_sdk/")
    parser.add_argument("--connect-timeout", type=lambda value: _parse_timeout(value, "--connect-timeout"), default=15)
    parser.add_argument("--glove-timeout", type=lambda value: _parse_timeout(value, "--glove-timeout"), default=15)
    parser.add_argument("--reconnect-timeout", type=lambda value: _parse_timeout(value, "--reconnect-timeout"), default=15)
    parser.add_argument("--discovery-wait", type=lambda value: _parse_timeout(value, "--discovery-wait"), default=1)
    parser.add_argument("--core-host", help="Remote Core: exact IP, exact name, or sorted zero-based index")
    parser.add_argument("--required-hand", choices=("left", "right"), default="right")
    parser.add_argument("--loopback-only", action="store_true")
    parser.add_argument("--hand-motion", choices=HAND_MOTIONS, default="auto")
    tracker_group = parser.add_mutually_exclusive_group()
    tracker_group.add_argument("--tracker-diagnostics", dest="tracker_diagnostics", action="store_true", default=None)
    tracker_group.add_argument("--no-tracker-diagnostics", dest="tracker_diagnostics", action="store_false")
    parser.add_argument("--settings-dir")
    parser.add_argument("--log-dir")
    parser.add_argument("--left-calibration")
    parser.add_argument("--right-calibration")
    parser.add_argument("--haptics", action="store_true", help="Accept vibrate commands on stdin")
    parser.add_argument("--tracker-feed", choices=TRACKER_FEEDS, default="none")
    parser.add_argument("--vive-ultimate-socket", default=VUT_DEFAULT_SOCKET)
    parser.add_argument("--vive-ultimate-frame", choices=VUT_FRAMES, default=VUT_DEFAULT_FRAME)
    parser.add_argument("--tracker-id", help="Serial of the Ultimate tracker that drives the required hand")
    parser.add_argument("--tracker-user-id", type=int, help="Core user id for the tracker; default: the owner of the required glove")
    parser.add_argument("--tracker-stale-timeout", type=float, default=1.0)
    parser.add_argument("--wrist-composition", choices=WRIST_COMPOSITIONS, default="bridge")
    parser.add_argument("--tracker-to-wrist-translation", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--tracker-to-wrist-rotation-xyzw", type=float, nargs=4, metavar=("X", "Y", "Z", "W"))
    parser.add_argument("--tracker-to-casing-translation", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--tracker-to-casing-rotation-xyzw", type=float, nargs=4, metavar=("X", "Y", "Z", "W"))
    return parser


def parse_options(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    options = parser.parse_args(argv)
    if options.discovery_wait == 0:
        parser.error("--discovery-wait must be at least 1 second")
    if options.mode == "integrated" and options.core_host is not None:
        parser.error("--core-host is only valid in --mode remote")
    if options.tracker_diagnostics is None:
        options.tracker_diagnostics = options.mode == "remote" or options.tracker_feed != "none"
    if options.tracker_feed != "none" and options.mode != "integrated":
        parser.error("--tracker-feed requires --mode integrated; Remote Core supplies its own trackers")
    if not math.isfinite(options.tracker_stale_timeout) or options.tracker_stale_timeout <= 0.0:
        parser.error("--tracker-stale-timeout must be positive and finite")
    side = options.required_hand
    casing = options.tracker_to_casing_translation or DEFAULT_TRACKER_TO_CASING_TRANSLATION[side]
    options.tracker_to_casing = RigidTransform(
        casing, options.tracker_to_casing_rotation_xyzw or (0.0, 0.0, 0.0, 1.0)
    )
    options.tracker_to_wrist = RigidTransform(
        options.tracker_to_wrist_translation or casing,
        options.tracker_to_wrist_rotation_xyzw or options.tracker_to_casing_rotation_xyzw or (0.0, 0.0, 0.0, 1.0),
    )
    home = Path.home()
    options.settings_dir = str(Path(options.settings_dir or home / ".config/manus").expanduser()) + "/"
    options.log_dir = str(Path(options.log_dir or home / ".local/state/dex_teleop/manus").expanduser()) + "/"
    return options


# ---------------------------------------------------------------------------
# Helpers


def _read_calibration(path: str | None) -> bytes:
    if not path:
        return b""
    data = Path(path).expanduser().read_bytes()
    if not data:
        raise ValueError(f"MANUS calibration file is empty: {path}")
    return data


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _char_array_to_str(ffi, array) -> str:
    return ffi.string(array).decode("utf-8", errors="replace")


def _set_char_array(ffi, array, text: str) -> None:
    size = ffi.sizeof(array)
    encoded = text.encode("utf-8")[: max(size - 1, 0)]
    ffi.memmove(array, encoded.ljust(size, b"\0"), size)


def _version_string(version) -> str:
    return f"{int(version.major)}.{int(version.minor)}.{int(version.patch)}"


class _Names:
    """Enum-to-wire-name tables matching the C++ bridge."""

    def __init__(self, enums) -> None:
        self.enums = enums
        self.side = {int(enums.Side.Left): "left", int(enums.Side.Right): "right"}
        self.hand_motion = {
            int(enums.HandMotion.None_): "none",
            int(enums.HandMotion.IMU): "imu",
            int(enums.HandMotion.Tracker): "tracker",
            int(enums.HandMotion.Tracker_RotationOnly): "tracker_rotation_only",
            int(enums.HandMotion.Auto): "auto",
        }
        self.hand_motion_values = {name: value for value, name in self.hand_motion.items()}
        self.tracker_type = {
            int(enums.TrackerType.LeftHand): "left_hand",
            int(enums.TrackerType.RightHand): "right_hand",
            int(enums.TrackerType.Head): "head",
            int(enums.TrackerType.Waist): "waist",
            int(enums.TrackerType.LeftFoot): "left_foot",
            int(enums.TrackerType.RightFoot): "right_foot",
            int(enums.TrackerType.Controller): "controller",
            int(enums.TrackerType.Camera): "camera",
        }
        self.quality = {
            int(enums.TrackingQuality.Untrackable): "untrackable",
            int(enums.TrackingQuality.BadTracking): "bad",
            int(enums.TrackingQuality.Trackable): "trackable",
        }
        self.quality_values = {name: value for value, name in self.quality.items()}
        self.tracker_system = {
            int(enums.TrackerSystemType.OpenVR): "openvr",
            int(enums.TrackerSystemType.OpenXR): "openxr",
            int(enums.TrackerSystemType.ART): "art",
            int(enums.TrackerSystemType.Optitrack): "optitrack",
            int(enums.TrackerSystemType.Vicon): "vicon",
            int(enums.TrackerSystemType.Antilatency): "antilatency",
        }
        self.chain = {
            int(enums.ChainType.Hand): "hand",
            int(enums.ChainType.FingerThumb): "thumb",
            int(enums.ChainType.FingerIndex): "index",
            int(enums.ChainType.FingerMiddle): "middle",
            int(enums.ChainType.FingerRing): "ring",
            int(enums.ChainType.FingerPinky): "little",
        }
        self.joint = {
            int(enums.FingerJointType.Metacarpal): "metacarpal",
            int(enums.FingerJointType.Proximal): "proximal",
            int(enums.FingerJointType.Intermediate): "intermediate",
            int(enums.FingerJointType.Distal): "distal",
            int(enums.FingerJointType.Tip): "tip",
        }


def compose_tracker_wrist(
    nodes: Sequence[tuple[int, np.ndarray, np.ndarray]],
    wrist_node_id: int,
    tracker_core_world: RigidTransform,
    tracker_to_wrist: RigidTransform,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Re-root raw-skeleton nodes so the wrist follows ``tracker * tracker_to_wrist``.

    ``nodes`` are ``(id, position, quaternion_xyzw)`` in SDK global space. The
    rigid relation of every node to the wrist is preserved exactly; only the
    hand's global placement changes.
    """

    root = next((node for node in nodes if node[0] == wrist_node_id), None)
    if root is None:
        raise ValueError(f"wrist node {wrist_node_id} missing from raw skeleton")
    target = compose_transforms(tracker_core_world, tracker_to_wrist)
    delta = compose_transforms(target, inverse_transform(RigidTransform(root[1], root[2])))
    delta_rotation = Rotation.from_quat(delta.quaternion_xyzw)
    positions = np.asarray([node[1] for node in nodes], dtype=np.float64)
    quaternions = np.asarray([node[2] for node in nodes], dtype=np.float64)
    new_positions = delta_rotation.apply(positions) + delta.translation
    new_quaternions = (delta_rotation * Rotation.from_quat(quaternions)).as_quat()
    return [
        (node[0], new_positions[index], new_quaternions[index])
        for index, node in enumerate(nodes)
    ]


# ---------------------------------------------------------------------------
# Bridge


class Bridge:
    def __init__(self, options: argparse.Namespace, sdk) -> None:
        self.options = options
        self.module, self.lib, self.ffi, self.wrapper_class, self.enums = sdk
        self.names = _Names(self.enums)
        self.success = int(self.enums.SDKReturnCode.Success)
        self.required_side = int(
            self.enums.Side.Left if options.required_hand == "left" else self.enums.Side.Right
        )
        self.hand_motion_value = self.names.hand_motion_values[options.hand_motion]
        self._calibrations = {
            "left": _read_calibration(options.left_calibration),
            "right": _read_calibration(options.right_calibration),
        }
        self._output_lock = threading.Lock()
        self._state = threading.Condition()
        self._stop = threading.Event()
        self._wrapper = None
        self._callbacks_registered = False
        self._sdk_initialized = False
        self._connected = False
        self._restarting_sdk = False
        self._recovering = False
        self._stopping = False
        self._disconnect_pending = False
        self._connection_generation = 0
        self._connect_callback_generation = 0
        self._selected_host: dict[str, Any] | None = None
        self._sdk_version = ""
        self._core_version = ""
        self._landscape_core_version = ""
        self._versions_compatible = False
        self._versions_qualified = False
        self._hand_motion_qualified = False
        self._license_qualified = False
        self._license_error_emitted = False
        self._required_topology_qualified = False
        self._required_frame_seen = False
        self._streaming_enabled = False
        self._session_id = 0
        self._sequence = 0
        self._tracker_sequence = 0
        self._terminal_exit_code = 0
        self._topologies: dict[int, dict[str, Any]] = {}
        self._tracker_systems: dict[str, str] = {}
        self._last_landscape_payload = ""
        self._required_glove_user_id: int | None = None
        self._haptic_gloves: dict[str, int] = {}
        self._vut_feed: VutPoseFeed | None = None
        self._vut_latest: VutTrackerPose | None = None
        self._vut_latest_core: RigidTransform | None = None
        self._vut_assigned_user: int | None = None
        self._vut_offset_user: int | None = None
        self._vut_send_failures = 0
        self._vut_other: dict[str, VutTrackerPose] = {}
        self._stdin_thread: threading.Thread | None = None

    # -- output -----------------------------------------------------------

    def emit(self, payload: Mapping[str, Any]) -> None:
        line = (json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        with self._output_lock:
            view = memoryview(line)
            while view:
                try:
                    written = os.write(self.options.protocol_fd, view)
                except InterruptedError:
                    continue
                except OSError:
                    self._stop.set()
                    return
                view = view[written:]

    def envelope(self, event_type: str) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "version": PROTOCOL_VERSION,
            "type": event_type,
            "mode": self.options.mode,
            "linked_mode": "python",
            "connection_generation": self._connection_generation,
        }

    def emit_status(self, state: str, message: str, **extra: Any) -> None:
        self.emit({**self.envelope("status"), "state": state, "message": message, **extra})

    def fail(self, code: str, message: str, sdk_code: int | None = None, exit_code: int = 0) -> int:
        payload = {**self.envelope("error"), "code": code, "message": message}
        if sdk_code is not None:
            payload["sdk_return_code"] = int(sdk_code)
        self.emit(payload)
        suffix = "" if sdk_code is None else f" (SDK code {int(sdk_code)})"
        print(f"[manus_python_bridge] {message}{suffix}", file=sys.stderr, flush=True)
        return exit_code

    def connection_fields_locked(self) -> dict[str, Any]:
        host = self._selected_host or {}
        return {
            "core_host_name": host.get("name", ""),
            "core_host_ip": host.get("ip", ""),
            "sdk_version": self._sdk_version,
            "core_version": self._core_version,
            "versions_compatible": self._versions_compatible,
            "session_id": self._session_id,
            "hand_motion": self.options.hand_motion,
            "core_world_frame": CORE_WORLD_FRAME,
            "wrist_composition": (
                self.options.wrist_composition if self.options.tracker_feed != "none" else "core"
            ),
            "tracker_feed": self.options.tracker_feed,
            "required_glove_user_id": self._required_glove_user_id,
            "tracker_to_wrist": self.options.tracker_to_wrist.as_mapping(),
        }

    def connection_fields(self) -> dict[str, Any]:
        with self._state:
            return self.connection_fields_locked()

    # -- lifecycle helpers ---------------------------------------------------

    def _callback_is_current(self, generation: int) -> bool:
        return (
            not self._restarting_sdk
            and generation == self._connection_generation
            and self._connect_callback_generation == self._connection_generation
        )

    def _qualification_complete(self) -> bool:
        return (
            self._versions_qualified
            and self._hand_motion_qualified
            and self._license_qualified
            and self._required_topology_qualified
            and self._required_frame_seen
        )

    # -- run ---------------------------------------------------------------

    def run(self) -> int:
        self._connection_generation = 1
        self.emit_status(
            "starting",
            f"Powered by MANUS; initializing the MANUS Core 3.2 Python SDK ({self.options.mode})",
        )
        try:
            if not self.initialize_sdk():
                return self._terminal_exit_code or 10
            self.start_tracker_feed()
            if not self.connect():
                return 14
            metadata_deadline = time.monotonic() + self.options.connect_timeout
            if not self.check_versions(metadata_deadline):
                return 14
            if not self.configure_hand_motion():
                return 15
            self.emit_status("sdk_connected", f"MANUS {self.options.mode} host connected", **self.connection_fields())
            self.start_command_channel()
            qualification_deadline = time.monotonic() + self.options.glove_timeout
            if not self.wait_for_stream_qualification(qualification_deadline, reconnecting=False):
                return self._terminal_exit_code or 15
            while not self._stop.is_set():
                with self._state:
                    self._state.wait_for(
                        lambda: self._disconnect_pending or self._stop.is_set(), timeout=0.25
                    )
                    reconnect = self._disconnect_pending
                    self._disconnect_pending = False
                if not reconnect or self._stop.is_set():
                    continue
                if self.options.mode != "remote" or self.options.reconnect_timeout == 0:
                    self._terminal_exit_code = 17
                    self.fail(
                        "host_disconnected",
                        f"MANUS {self.options.mode.title()} host disconnected; automatic reconnect is disabled for this mode",
                        exit_code=17,
                    )
                    self._stop.set()
                    break
                if not self.restart_and_reconnect():
                    self._terminal_exit_code = 17
                    self._stop.set()
                    break
            self.emit_status("stopping", "Stop requested")
            return self._terminal_exit_code
        finally:
            self.shutdown()

    # -- SDK initialization ---------------------------------------------------

    def initialize_sdk(self) -> bool:
        lib = self.lib
        if self.options.mode == "remote":
            result = lib.CoreSdk_InitializeCore()
            call = "CoreSdk_InitializeCore"
        else:
            result = lib.CoreSdk_InitializeIntegrated()
            call = "CoreSdk_InitializeIntegrated"
        if result != self.success:
            self.fail("sdk_initialization_failed", f"{call} failed", result, 10)
            return False
        self._sdk_initialized = True
        if self.options.mode == "integrated":
            Path(self.options.settings_dir).mkdir(parents=True, exist_ok=True)
            Path(self.options.log_dir).mkdir(parents=True, exist_ok=True)
            result = lib.CoreSdk_SetSessionType(int(self.enums.SessionType.CoreSDK))
            if result != self.success:
                self.fail("session_configuration_failed", "CoreSdk_SetSessionType failed", result, 11)
                return False
            result = lib.CoreSdk_SetSettingsLocation(self.options.settings_dir.encode("utf-8"))
            if result != self.success:
                self.fail("settings_configuration_failed", "CoreSdk_SetSettingsLocation failed", result, 11)
                return False
            result = lib.CoreSdk_SetLogLocation(self.options.log_dir.encode("utf-8"))
            if result != self.success:
                self.fail("log_configuration_failed", "CoreSdk_SetLogLocation failed", result, 11)
                return False
        if not self.register_callbacks():
            return False
        coordinates = self.ffi.new("CoordinateSystemVUH*")
        coordinates.handedness = int(self.enums.Side.Right)
        coordinates.up = int(self.enums.AxisPolarity.PositiveY)
        coordinates.view = int(self.enums.AxisView.ZToViewer)
        coordinates.unitScale = 1.0
        result = lib.CoreSdk_InitializeCoordinateSystemWithVUH(coordinates[0], True)
        if result != self.success:
            self.fail("coordinate_configuration_failed", "Could not configure Y-up right-handed meters", result, 13)
            return False
        return True

    def register_callbacks(self) -> bool:
        lib = self.lib
        if self._wrapper is None:
            self._wrapper = self.wrapper_class()
            results = [
                ("raw skeleton", self._wrapper.register_raw_skeleton_callback(self.on_raw_skeleton)),
                ("landscape", self._wrapper.register_landscape_callback(self.on_landscape)),
                ("connect", self._wrapper.register_connected_callback(self.on_connect)),
                ("disconnect", self._wrapper.register_disconnected_callback(self.on_disconnect)),
                ("log", self._wrapper.register_log_callback(self.on_log)),
            ]
            if self.options.tracker_diagnostics:
                results.append(("tracker", self._wrapper.register_tracker_callback(self.on_tracker)))
        else:
            # The extern "Python" trampolines already exist; re-point the SDK at them.
            results = [
                ("raw skeleton", lib.CoreSdk_RegisterCallbackForRawSkeletonStream(lib.python_raw_skeleton_callback)),
                ("landscape", lib.CoreSdk_RegisterCallbackForLandscapeStream(lib.python_landscape_callback)),
                ("connect", lib.CoreSdk_RegisterCallbackForOnConnect(lib.python_connected_callback)),
                ("disconnect", lib.CoreSdk_RegisterCallbackForOnDisconnect(lib.python_disconnected_callback)),
                ("log", lib.CoreSdk_RegisterCallbackForOnLog(lib.python_log_callback)),
            ]
            if self.options.tracker_diagnostics:
                results.append(("tracker", lib.CoreSdk_RegisterCallbackForTrackerStream(lib.python_tracker_callback)))
        for name, result in results:
            if result != self.success:
                self.fail("callback_registration_failed", f"Could not register {name} callback", result, 12)
                return False
        self._callbacks_registered = True
        return True

    def unregister_callbacks(self) -> None:
        if not self._callbacks_registered:
            return
        lib, null = self.lib, self.ffi.NULL
        for register in (
            lib.CoreSdk_RegisterCallbackForRawSkeletonStream,
            lib.CoreSdk_RegisterCallbackForLandscapeStream,
            lib.CoreSdk_RegisterCallbackForOnConnect,
            lib.CoreSdk_RegisterCallbackForOnDisconnect,
            lib.CoreSdk_RegisterCallbackForOnLog,
        ):
            try:
                register(null)
            except Exception:  # noqa: BLE001 - best effort during teardown
                pass
        if self.options.tracker_diagnostics:
            try:
                lib.CoreSdk_RegisterCallbackForTrackerStream(null)
            except Exception:  # noqa: BLE001
                pass
        self._callbacks_registered = False

    # -- connection -----------------------------------------------------------

    def _discover_hosts(self, loopback_only: bool) -> list[dict[str, Any]] | None:
        lib, ffi = self.lib, self.ffi
        result = lib.CoreSdk_LookForHosts(self.options.discovery_wait, loopback_only)
        if result != self.success:
            return None
        count = ffi.new("uint32_t*")
        if lib.CoreSdk_GetNumberOfAvailableHostsFound(count) != self.success or count[0] == 0:
            return []
        hosts = ffi.new("ManusHost[]", count[0])
        if lib.CoreSdk_GetAvailableHostsFound(hosts, count[0]) != self.success:
            return None
        found = []
        for index in range(count[0]):
            host = hosts[index]
            found.append(
                {
                    "name": _char_array_to_str(ffi, host.hostName),
                    "ip": _char_array_to_str(ffi, host.ipAddress),
                    "version": _version_string(host.manusCoreVersion),
                    "_version_tuple": (
                        int(host.manusCoreVersion.major),
                        int(host.manusCoreVersion.minor),
                        int(host.manusCoreVersion.patch),
                    ),
                }
            )
        found.sort(key=lambda host: (host["name"].lower(), host["ip"], host["_version_tuple"]))
        return found

    def _host_struct(self, host: Mapping[str, Any]):
        struct = self.ffi.new("ManusHost*")
        _set_char_array(self.ffi, struct.hostName, host.get("name", ""))
        _set_char_array(self.ffi, struct.ipAddress, host.get("ip", ""))
        version = host.get("_version_tuple")
        if version is not None:
            struct.manusCoreVersion.major, struct.manusCoreVersion.minor, struct.manusCoreVersion.patch = version
        return struct

    def emit_hosts(self, hosts: Sequence[Mapping[str, Any]]) -> None:
        self.emit(
            {
                **self.envelope("hosts"),
                "selector": self.options.core_host or "",
                "loopback_only": self.options.mode == "integrated" or self.options.loopback_only,
                "hosts": [
                    {"index": index, "name": host["name"], "ip": host["ip"], "version": host["version"]}
                    for index, host in enumerate(hosts)
                ],
            }
        )

    def select_host(self, hosts: Sequence[Mapping[str, Any]]) -> int | None:
        if not hosts:
            return None
        selector = self.options.core_host
        if not selector:
            if self.options.mode == "remote" and len(hosts) != 1:
                self._terminal_exit_code = 14
                self.fail(
                    "core_host_ambiguous",
                    f"Remote discovery found {len(hosts)} Core hosts; select one with --core-host by exact IP, exact name, or sorted index",
                    exit_code=14,
                )
                return None
            return 0
        folded = selector.lower()
        matches = [
            index
            for index, host in enumerate(hosts)
            if selector == host["ip"] or folded == host["name"].lower()
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            self._terminal_exit_code = 14
            self.fail(
                "core_host_ambiguous",
                f"Requested Core host name {selector} matched more than one discovered Core; use an exact IP or sorted index",
                exit_code=14,
            )
            return None
        if selector.isdigit():
            index = int(selector)
            if index < len(hosts):
                return index
            self._terminal_exit_code = 14
            self.fail(
                "core_host_not_found",
                f"Requested sorted Core host index {selector} is outside the discovered host list",
                exit_code=14,
            )
            return None
        self._terminal_exit_code = 14
        self.fail(
            "core_host_not_found",
            f"Requested Core host {selector} did not match any discovered Core host; use an exact IP, exact name, or sorted index",
            exit_code=14,
        )
        return None

    def wait_for_connect_callback(self, deadline: float, generation: int) -> bool:
        with self._state:
            while not self._stop.is_set():
                if self._callback_is_current(generation):
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._state.wait(remaining)
        return False

    def connect(self) -> bool:
        deadline = time.monotonic() + self.options.connect_timeout
        generation = self._connection_generation
        with self._state:
            self._connect_callback_generation = 0
        lib = self.lib
        if self.options.mode == "integrated":
            # MANUS Core 3.2 clients discover the embedded Core on loopback; the
            # 3.1.1 empty-host connect remains the fallback.
            last_result = None
            hosts = self._discover_hosts(True)
            if hosts:
                self.emit_hosts(hosts)
                self._selected_host = {key: value for key, value in hosts[0].items() if not key.startswith("_")}
                last_result = lib.CoreSdk_ConnectToHost(self._host_struct(hosts[0])[0])
                if last_result == self.success and self.wait_for_connect_callback(deadline, generation):
                    return True
            empty = self.ffi.new("ManusHost*")
            last_result = lib.CoreSdk_ConnectToHost(empty[0])
            if last_result == self.success and self.wait_for_connect_callback(deadline, generation):
                return True
            self.fail(
                "host_connection_failed",
                f"MANUS Integrated did not deliver its on-connect callback before the {self.options.connect_timeout} second timeout",
                last_result,
                14,
            )
            return False

        last_result = int(self.enums.SDKReturnCode.NotConnected)
        while True:
            hosts = self._discover_hosts(self.options.loopback_only)
            if hosts:
                self.emit_hosts(hosts)
                index = self.select_host(hosts)
                if self._terminal_exit_code != 0:
                    return False
                if index is not None:
                    selected = hosts[index]
                    self._selected_host = {key: value for key, value in selected.items() if not key.startswith("_")}
                    self._selected_host_struct = self._host_struct(selected)
                    last_result = lib.CoreSdk_ConnectToHost(self._selected_host_struct[0])
                    if last_result == self.success and self.wait_for_connect_callback(deadline, generation):
                        return True
            if self._stop.is_set():
                return False
            if time.monotonic() >= deadline:
                break
            self._stop.wait(0.25)
        selector = self.options.core_host or "automatic single-host selection"
        self.fail(
            "host_connection_failed",
            f"Could not discover and connect to a MANUS remote host before the {self.options.connect_timeout} second timeout "
            f"(selector: {selector}). Verify Core mode, host/network discovery, firewall policy, and matching SDK/Core versions",
            last_result,
            14,
        )
        return False

    def check_versions(self, deadline: float) -> bool:
        lib, ffi = self.lib, self.ffi
        last_result = int(self.enums.SDKReturnCode.NotConnected)
        while True:
            sdk_version = ffi.new("ManusVersion*")
            core_version = ffi.new("ManusVersion*")
            compatible = ffi.new("bool*")
            last_result = lib.CoreSdk_GetVersionsAndCheckCompatibility(sdk_version, core_version, compatible)
            session = ffi.new("uint32_t*")
            session_result = lib.CoreSdk_GetSessionId(session)
            sdk_text = _char_array_to_str(ffi, sdk_version.versionInfo)
            core_text = _char_array_to_str(ffi, core_version.versionInfo)
            versions_ready = bool(sdk_text) and bool(core_text)
            if last_result == self.success and versions_ready and not compatible[0]:
                self.fail(
                    "version_incompatible",
                    f"MANUS SDK {sdk_text} is incompatible with Core {core_text}; install matching SDK/Core major-minor releases",
                    exit_code=14,
                )
                return False
            session_ready = self.options.mode == "integrated" or (
                session_result == self.success and session[0] != 0
            )
            if last_result == self.success and versions_ready and compatible[0] and session_ready:
                with self._state:
                    self._sdk_version = sdk_text
                    self._core_version = core_text
                    self._versions_compatible = True
                    self._session_id = int(session[0]) if session_result == self.success else 0
                    self._versions_qualified = True
                return True
            if self._stop.is_set() or time.monotonic() >= deadline:
                break
            time.sleep(0.025)
        if self.options.mode == "integrated" and last_result == self.success:
            with self._state:
                self._sdk_version = sdk_text or "unreported-by-integrated-sdk"
                self._core_version = (
                    core_text or self._landscape_core_version or "unreported-by-integrated-sdk"
                )
                self._versions_compatible = True
                self._versions_qualified = True
            return True
        self.fail(
            "version_metadata_not_ready",
            "MANUS on-connect callback arrived, but non-empty compatible SDK/Core versions and required session metadata were not ready before the connection deadline",
            last_result,
            14,
        )
        return False

    def configure_hand_motion(self) -> bool:
        lib, ffi = self.lib, self.ffi
        result = lib.CoreSdk_SetRawSkeletonHandMotion(self.hand_motion_value)
        if result != self.success:
            self.fail(
                "hand_motion_configuration_failed",
                f"CoreSdk_SetRawSkeletonHandMotion({self.options.hand_motion}) failed",
                result,
                15,
            )
            return False
        actual = ffi.new("HandMotion*")
        read_result = lib.CoreSdk_GetRawSkeletonHandMotion(actual)
        if read_result != self.success or int(actual[0]) != self.hand_motion_value:
            self.fail(
                "hand_motion_verification_failed",
                f"Core did not confirm requested raw-skeleton hand motion {self.options.hand_motion}",
                read_result,
                15,
            )
            return False
        with self._state:
            self._hand_motion_qualified = True
        return True

    def wait_for_stream_qualification(self, deadline: float, *, reconnecting: bool) -> bool:
        with self._state:
            generation = self._connection_generation
            while not self._stop.is_set() and not self._disconnect_pending:
                if self._connection_generation == generation and self._qualification_complete():
                    if reconnecting:
                        payload = {
                            **self.envelope("status"),
                            "state": "reconnected",
                            "message": "MANUS Remote connection and required-hand stream requalified",
                        }
                    else:
                        payload = {
                            **self.envelope("status"),
                            "state": "topology_ready",
                            "message": "MANUS required-hand license, topology, and raw stream qualified",
                        }
                    payload.update(self.connection_fields_locked())
                    self._streaming_enabled = True
                    self._recovering = False
                    self.emit(payload)
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._state.wait(remaining)
            if self._stop.is_set() or self._disconnect_pending:
                return False
            missing = []
            if not self._license_qualified:
                missing.append("license")
            if not self._required_topology_qualified:
                missing.append(f"{self.options.required_hand}-hand topology")
            if not self._required_frame_seen:
                missing.append("raw frame")
        exit_code = 17 if reconnecting else 15
        self._terminal_exit_code = exit_code
        self.fail(
            "reconnect_stream_not_qualified" if reconnecting else "no_glove",
            "MANUS connection did not qualify "
            + ", ".join(missing)
            + " before the bounded deadline (is the glove on, paired to this dongle, and licensed for SDK Integrated?)",
            exit_code=exit_code,
        )
        return False

    def reset_qualification_locked(self) -> None:
        self._topologies.clear()
        self._tracker_systems.clear()
        self._last_landscape_payload = ""
        self._sdk_version = ""
        self._core_version = ""
        self._landscape_core_version = ""
        self._session_id = 0
        self._versions_compatible = False
        self._versions_qualified = False
        self._hand_motion_qualified = False
        self._license_qualified = False
        self._required_topology_qualified = False
        self._required_frame_seen = False
        self._streaming_enabled = False
        self._connect_callback_generation = 0
        self._license_error_emitted = False
        self._required_glove_user_id = None
        self._haptic_gloves.clear()

    def restart_and_reconnect(self) -> bool:
        with self._state:
            if self._selected_host is None:
                self.fail("reconnect_failed", "Cannot reconnect because no deterministic Core host was selected", exit_code=17)
                return False
            selected = dict(self._selected_host)
            self._restarting_sdk = True
            self.reset_qualification_locked()
        self.emit_status(
            "reconnecting",
            f"Restarting the MANUS SDK before reconnecting to locked Core host {selected.get('name', '')} ({selected.get('ip', '')})",
        )
        deadline = time.monotonic() + self.options.reconnect_timeout
        self.unregister_callbacks()
        shutdown_result = self.lib.CoreSdk_ShutDown()
        self._sdk_initialized = False
        if shutdown_result != self.success:
            self.fail("sdk_restart_failed", "CoreSdk_ShutDown failed while preparing Remote reconnect", shutdown_result, 17)
            return False
        if not self.initialize_sdk():
            self.fail("sdk_restart_failed", "MANUS SDK reinitialization failed during Remote reconnect", exit_code=17)
            return False
        with self._state:
            self._restarting_sdk = False
            self._selected_host = selected
        generation = self._connection_generation
        connected = False
        connect_result = int(self.enums.SDKReturnCode.NotConnected)
        host_struct = self._host_struct(selected)
        while not self._stop.is_set() and time.monotonic() < deadline:
            connect_result = self.lib.CoreSdk_ConnectToHost(host_struct[0])
            if connect_result == self.success:
                connected = self.wait_for_connect_callback(deadline, generation)
                break
            self._stop.wait(0.25)
        if not connected:
            self.fail("reconnect_failed", "Restarted MANUS SDK did not reconnect to the locked Core host before the bounded deadline", connect_result, 17)
            return False
        return (
            self.check_versions(deadline)
            and self.configure_hand_motion()
            and self.wait_for_stream_qualification(deadline, reconnecting=True)
        )

    # -- SDK callbacks ---------------------------------------------------------

    def on_connect(self, host) -> None:
        try:
            with self._state:
                if self._stopping:
                    return
                if host != self.ffi.NULL:
                    name = _char_array_to_str(self.ffi, host.hostName)
                    ip = _char_array_to_str(self.ffi, host.ipAddress)
                    if name or ip:
                        self._selected_host = {"name": name, "ip": ip, "version": _version_string(host.manusCoreVersion)}
                self._connected = True
                self._connect_callback_generation = self._connection_generation
                self._state.notify_all()
        except Exception as error:  # noqa: BLE001
            self._callback_failure("on_connect", error)

    def on_disconnect(self, _host) -> None:
        try:
            with self._state:
                if self._stopping or self._restarting_sdk or self._recovering:
                    return
                recoverable = (
                    self.options.mode == "remote"
                    and self.options.reconnect_timeout > 0
                    and not self._stop.is_set()
                )
                previous_generation = self._connection_generation
                previous_session = self._session_id
                self._connection_generation = previous_generation + 1
                self._recovering = True
                self._streaming_enabled = False
                self._required_frame_seen = False
                self._required_topology_qualified = False
                self._license_qualified = False
                self._topologies.clear()
                self._tracker_systems.clear()
                self._last_landscape_payload = ""
                self._connected = False
                self._disconnect_pending = not self._stop.is_set()
                payload = {
                    **self.envelope("status"),
                    "state": "disconnected",
                    "message": f"MANUS {self.options.mode.title()} host disconnected",
                    "recoverable": recoverable,
                    "previous_connection_generation": previous_generation,
                    "previous_session_id": previous_session,
                    **self.connection_fields_locked(),
                }
                self._state.notify_all()
            self.emit(payload)
        except Exception as error:  # noqa: BLE001
            self._callback_failure("on_disconnect", error)

    def on_log(self, severity, message, length) -> None:
        try:
            if int(severity) < int(self.enums.LogSeverity.Warn):
                return
            text = self.ffi.string(message, int(length)).decode("utf-8", errors="replace")
            print(f"[manus_sdk] {text}", file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001 - logging must never break acquisition
            pass

    def _callback_failure(self, where: str, error: BaseException) -> None:
        self._terminal_exit_code = self._terminal_exit_code or 18
        self.fail("bridge_callback_failed", f"{where} failed: {error!r}", exit_code=18)
        self._stop.set()
        with self._state:
            self._state.notify_all()

    def on_landscape(self, landscape) -> None:
        try:
            self._handle_landscape(landscape)
        except Exception as error:  # noqa: BLE001
            self._callback_failure("on_landscape", error)

    def _handle_landscape(self, landscape) -> None:
        if landscape == self.ffi.NULL:
            return
        lib, ffi, names = self.lib, self.ffi, self.names
        generation = self._connection_generation
        with self._state:
            if not self._callback_is_current(generation):
                return
            settings = landscape.settings
            self._landscape_core_version = _version_string(settings.manusCoreVersion)
            trackers = []
            tracker_systems: dict[str, str] = {}
            for index in range(int(landscape.trackers.trackerCount)):
                tracker = landscape.trackers.trackers[index]
                tracker_id = _char_array_to_str(ffi, tracker.id)
                system = names.tracker_system.get(int(tracker.systemType), "unknown")
                tracker_systems[tracker_id] = system
                trackers.append(
                    {
                        "id": tracker_id,
                        "type": names.tracker_type.get(int(tracker.type), "unknown"),
                        "system": system,
                        "user_id": int(tracker.user),
                        "is_hmd": bool(tracker.isHMD),
                        "manufacturer": _char_array_to_str(ffi, tracker.manufacturer),
                        "product_name": _char_array_to_str(ffi, tracker.productName),
                    }
                )
            self._tracker_systems = tracker_systems
            payload = {
                **self.envelope("landscape"),
                "core_version": self._landscape_core_version,
                "license_sdk": bool(settings.license.sdk),
                "license_integrated": bool(settings.license.integrated),
                "license_linux": bool(settings.license.linuxTarget),
                "trackers": trackers,
            }
            encoded = json.dumps(payload, sort_keys=True)
            if encoded != self._last_landscape_payload:
                self._last_landscape_payload = encoded
                self.emit(payload)

            devices = landscape.gloveDevices
            gloves = [devices.gloves[index] for index in range(int(devices.gloveCount))]
            supported = [glove for glove in gloves if int(glove.side) in names.side]
            license_available = (
                bool(settings.license.sdk) if self.options.mode == "remote" else bool(settings.license.integrated)
            )
            if supported and not license_available:
                if not self._license_error_emitted:
                    self._license_error_emitted = True
                    self._terminal_exit_code = 16
                    feature = "SDK" if self.options.mode == "remote" else "Integrated SDK"
                    self.fail(
                        "sdk_license_unavailable" if self.options.mode == "remote" else "integrated_license_unavailable",
                        f"The connected MANUS license does not enable the required {feature} feature",
                        exit_code=16,
                    )
                    self._stop.set()
                    self._state.notify_all()
                return
            if supported and license_available:
                self._license_qualified = True

            users = [landscape.users.users[index] for index in range(int(landscape.users.userCount))]
            present: list[int] = []
            for glove in supported:
                glove_id = int(glove.id)
                present.append(glove_id)
                side = names.side[int(glove.side)]
                owner = None
                for user in users:
                    glove_field = user.leftGloveID if side == "left" else user.rightGloveID
                    if int(glove_field) == glove_id:
                        owner = int(user.id)
                        break
                node_count = ffi.new("uint32_t*")
                if lib.CoreSdk_GetRawSkeletonNodeCount(glove_id, node_count) != self.success or node_count[0] == 0:
                    continue
                count = int(node_count[0])
                info_array = ffi.new("NodeInfo[]", count)
                if lib.CoreSdk_GetRawSkeletonNodeInfoArray(glove_id, info_array, count) != self.success:
                    continue
                topology = [
                    {
                        "id": int(info_array[index].nodeId),
                        "parent_id": int(info_array[index].parentId),
                        "side": names.side.get(int(info_array[index].side), "invalid"),
                        "chain": names.chain.get(int(info_array[index].chainType), "invalid"),
                        "joint": names.joint.get(int(info_array[index].fingerJointType), "invalid"),
                    }
                    for index in range(count)
                ]
                state = self._topologies.setdefault(
                    glove_id, {"side": side, "topology": None, "calibration_attempted": False, "user_id": None}
                )
                state["side"] = side
                state["haptics"] = bool(glove.isHaptics)
                if not self._apply_calibration(glove_id, side, state):
                    self._terminal_exit_code = 16
                    self._stop.set()
                    self._state.notify_all()
                    return
                if bool(glove.isHaptics):
                    self._haptic_gloves[side] = glove_id
                if state["topology"] != topology or state["user_id"] != owner:
                    state["topology"] = topology
                    state["user_id"] = owner
                    self.emit(
                        {
                            **self.envelope("topology"),
                            "glove_id": glove_id,
                            "side": side,
                            "transform_space": "sdk_global",
                            "user_id": owner,
                            "haptics": bool(glove.isHaptics),
                            "nodes": topology,
                        }
                    )
                if int(glove.side) == self.required_side:
                    self._required_topology_qualified = True
                    self._required_glove_user_id = owner
            removed = [
                (glove_id, state["side"])
                for glove_id, state in self._topologies.items()
                if glove_id not in present
            ]
            for glove_id, side in removed:
                self._topologies.pop(glove_id, None)
                self._haptic_gloves.pop(side, None)
                self.emit(
                    {
                        **self.envelope("glove_removed"),
                        "glove_id": glove_id,
                        "side": side,
                        "reason": "landscape_absent",
                    }
                )
                if side == self.options.required_hand:
                    self._required_topology_qualified = False
                    self._required_frame_seen = False
                    self._streaming_enabled = False
                    self._required_glove_user_id = None
            self._state.notify_all()
        self._maybe_assign_tracker()

    def _apply_calibration(self, glove_id: int, side: str, state: dict[str, Any]) -> bool:
        if state["calibration_attempted"]:
            return True
        state["calibration_attempted"] = True
        calibration = self._calibrations[side]
        if not calibration:
            return True
        buffer = self.ffi.new("unsigned char[]", calibration)
        result_code = self.ffi.new("SetGloveCalibrationReturnCode*")
        sdk_result = self.lib.CoreSdk_SetGloveCalibration(glove_id, buffer, len(calibration), result_code)
        if sdk_result != self.success or int(result_code[0]) != int(self.enums.SetGloveCalibrationReturnCode.Success):
            self.fail(
                "calibration_failed",
                f"Calibration rejected for {side} glove {glove_id} (result {int(result_code[0])})",
                sdk_result,
                0,
            )
            return False
        return True

    def on_raw_skeleton(self, stream) -> None:
        try:
            self._handle_raw_skeleton(stream)
        except Exception as error:  # noqa: BLE001
            self._callback_failure("on_raw_skeleton", error)

    def _handle_raw_skeleton(self, stream) -> None:
        if stream == self.ffi.NULL:
            return
        lib, ffi = self.lib, self.ffi
        generation = self._connection_generation
        callback_monotonic_ns = time.monotonic_ns()
        publish_time = int(stream.publishTime.time)
        for index in range(int(stream.skeletonsCount)):
            info = ffi.new("RawSkeletonInfo*")
            if lib.CoreSdk_GetRawSkeletonInfo(index, info) != self.success or info.nodesCount == 0:
                continue
            count = int(info.nodesCount)
            nodes_array = ffi.new("SkeletonNode[]", count)
            if lib.CoreSdk_GetRawSkeletonData(index, nodes_array, count) != self.success:
                continue
            glove_id = int(info.gloveId)
            nodes: list[tuple[int, np.ndarray, np.ndarray]] = []
            for node_index in range(count):
                node = nodes_array[node_index]
                transform = node.transform
                position = np.array(
                    (transform.position.x, transform.position.y, transform.position.z), dtype=np.float64
                )
                quaternion = np.array(
                    (transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w),
                    dtype=np.float64,
                )
                nodes.append((int(node.id), position, quaternion))
            with self._state:
                if not self._callback_is_current(generation):
                    continue
                state = self._topologies.get(glove_id)
                if state is None or state["topology"] is None:
                    continue
                side = state["side"]
                topology = state["topology"]
                wrist_node_id = next((node["id"] for node in topology if node["chain"] == "hand"), None)
                if len(nodes) != len(topology) or wrist_node_id is None:
                    continue
                may_publish = self._streaming_enabled and self._callback_is_current(generation)
                if not may_publish:
                    if (
                        side == self.options.required_hand
                        and self._versions_qualified
                        and self._hand_motion_qualified
                        and self._license_qualified
                        and self._required_topology_qualified
                    ):
                        self._required_frame_seen = True
                        self._state.notify_all()
                    continue
                tracker_pose_age_ns = None
                composition = "core"
                if (
                    self.options.tracker_feed != "none"
                    and self.options.wrist_composition == "bridge"
                    and side == self.options.required_hand
                ):
                    latest, latest_core = self._vut_latest, self._vut_latest_core
                    if latest is None or latest_core is None:
                        # No tracker pose yet: the consumer gates on tracker health
                        # and discards this wrist; the fingers are still valid.
                        pass
                    else:
                        nodes = compose_tracker_wrist(nodes, wrist_node_id, latest_core, self.options.tracker_to_wrist)
                        tracker_pose_age_ns = callback_monotonic_ns - latest.receipt_monotonic_ns
                        composition = "bridge"
                self._sequence += 1
                payload = {
                    **self.envelope("articulation"),
                    "sequence": self._sequence,
                    "glove_id": glove_id,
                    "side": side,
                    "manus_publish_time": publish_time,
                    "capture_monotonic_ns": callback_monotonic_ns,
                    "callback_capture_monotonic_ns": callback_monotonic_ns,
                    "wrist_node_id": wrist_node_id,
                    "wrist_composition": composition,
                    "tracker_pose_age_ns": tracker_pose_age_ns,
                    **self.connection_fields_locked(),
                    "nodes": [
                        {
                            "id": node_id,
                            "position": [float(value) for value in position],
                            "orientation_xyzw": [float(value) for value in quaternion],
                        }
                        for node_id, position, quaternion in nodes
                    ],
                }
            if not all(
                _finite(*node["position"], *node["orientation_xyzw"]) for node in payload["nodes"]
            ):
                continue
            self.emit(payload)

    def on_tracker(self, stream) -> None:
        try:
            if self.options.tracker_feed != "none":
                # The injected Ultimate feed is authoritative for tracker health;
                # Core's echo would only duplicate it.
                return
            self._handle_tracker_stream(stream)
        except Exception as error:  # noqa: BLE001
            self._callback_failure("on_tracker", error)

    def _handle_tracker_stream(self, stream) -> None:
        if stream == self.ffi.NULL:
            return
        lib, ffi = self.lib, self.ffi
        generation = self._connection_generation
        callback_monotonic_ns = time.monotonic_ns()
        trackers = []
        for index in range(int(stream.trackerCount)):
            data = ffi.new("TrackerData*")
            if lib.CoreSdk_GetTrackerData(index, data) != self.success:
                continue
            trackers.append(self._tracker_record(data))
        with self._state:
            if not self._callback_is_current(generation):
                return
            self._tracker_sequence += 1
            payload = {
                **self.envelope("trackers"),
                "sequence": self._tracker_sequence,
                "manus_publish_time": int(stream.publishTime.time),
                "capture_monotonic_ns": callback_monotonic_ns,
                **self.connection_fields_locked(),
                "trackers": trackers,
            }
        self.emit(payload)

    def _tracker_record(self, data) -> dict[str, Any]:
        ffi, names = self.ffi, self.names
        values = (
            data.position.x, data.position.y, data.position.z,
            data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w,
        )
        pose_valid = _finite(*values)
        tracker_id = _char_array_to_str(ffi, data.trackerId.id)
        record = {
            "id": tracker_id,
            "user_id": int(data.userId),
            "is_hmd": bool(data.isHmd),
            "type": names.tracker_type.get(int(data.trackerType), "unknown"),
            "quality": names.quality.get(int(data.quality), "unknown"),
            "tracking_system": self._tracker_systems.get(tracker_id, "unknown"),
            "last_update_time": int(data.lastUpdateTime.time),
            "pose_valid": pose_valid,
        }
        if pose_valid:
            record["position"] = [float(value) for value in values[:3]]
            record["orientation_xyzw"] = [float(value) for value in values[3:]]
        return record

    # -- VIVE Ultimate tracker feed -------------------------------------------

    def start_tracker_feed(self) -> None:
        if self.options.tracker_feed == "none":
            return
        self._vut_feed = VutPoseFeed(self.options.vive_ultimate_socket, listener=self._on_vut_update)
        self._vut_feed.start()
        self.emit_status(
            "tracker_feed_starting",
            f"Reading VIVE Ultimate poses from {self.options.vive_ultimate_socket} "
            f"(frame {self.options.vive_ultimate_frame}, tracker {self.options.tracker_id or 'single'})",
        )

    def _tracker_user_id(self) -> int | None:
        if self.options.tracker_user_id is not None:
            return int(self.options.tracker_user_id)
        with self._state:
            return self._required_glove_user_id

    def _on_vut_update(self, pose: VutTrackerPose) -> None:
        try:
            self._handle_vut_update(pose)
        except Exception as error:  # noqa: BLE001
            self._callback_failure("vive_ultimate_update", error)

    def _handle_vut_update(self, pose: VutTrackerPose) -> None:
        requested = self.options.tracker_id
        is_selected = pose.tracker_id == requested if requested is not None else True
        if requested is None:
            feed = self._vut_feed
            if feed is not None and len(feed.snapshot()) > 1:
                # Ambiguous without an explicit serial; publish for diagnostics only.
                is_selected = False
        core_pose = None
        if pose.pose_valid:
            core_pose = vut_pose_to_core_world(
                pose.position, pose.quaternion_xyzw, frame=self.options.vive_ultimate_frame
            )
        user_id = self._tracker_user_id()
        if is_selected:
            with self._state:
                if core_pose is not None and pose.quality != "untrackable":
                    self._vut_latest = pose
                    self._vut_latest_core = core_pose
            if core_pose is not None:
                self._inject_tracker(pose, core_pose, user_id)
        else:
            self._vut_other[pose.tracker_id] = pose
        self._emit_vut_trackers(pose, user_id)

    def _inject_tracker(self, pose: VutTrackerPose, core_pose: RigidTransform, user_id: int | None) -> None:
        lib, ffi, enums = self.lib, self.ffi, self.enums
        data = ffi.new("TrackerData*")
        _set_char_array(ffi, data.trackerId.id, pose.tracker_id)
        data.userId = 0 if user_id is None else int(user_id)
        data.isHmd = False
        data.trackerType = int(
            enums.TrackerType.LeftHand if self.options.required_hand == "left" else enums.TrackerType.RightHand
        )
        data.position.x, data.position.y, data.position.z = (float(value) for value in core_pose.translation)
        x, y, z, w = (float(value) for value in core_pose.quaternion_xyzw)
        data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w = x, y, z, w
        data.quality = self.names.quality_values[pose.quality]
        data.trackedPoint = int(enums.TrackedPoint.Casing)
        data.lastUpdateTime.time = max(0, int(pose.timestamp_us))
        result = lib.CoreSdk_SendDataForTrackers(data, 1)
        if result != self.success:
            self._vut_send_failures += 1
            if self._vut_send_failures == 1:
                message = f"CoreSdk_SendDataForTrackers rejected the VIVE Ultimate pose (SDK code {int(result)})"
                if self.options.wrist_composition == "core":
                    self._terminal_exit_code = 19
                    self.fail("tracker_injection_failed", message, result, 19)
                    self._stop.set()
                    with self._state:
                        self._state.notify_all()
                else:
                    print(f"[manus_python_bridge] {message}; the bridge keeps composing the wrist itself", file=sys.stderr, flush=True)
            return
        self._maybe_assign_tracker(pose.tracker_id)

    def _maybe_assign_tracker(self, tracker_id: str | None = None) -> None:
        """Assign the injected tracker to the glove's user once the owner is known."""

        if self.options.tracker_feed == "none":
            return
        if tracker_id is None:
            latest = self._vut_latest
            if latest is None:
                return
            tracker_id = latest.tracker_id
        user_id = self._tracker_user_id()
        if user_id is None:
            return
        lib, ffi, enums = self.lib, self.ffi, self.enums
        if self._vut_assigned_user != user_id:
            identity = ffi.new("TrackerId*")
            _set_char_array(ffi, identity.id, tracker_id)
            role = int(enums.TrackerType.LeftHand if self.options.required_hand == "left" else enums.TrackerType.RightHand)
            assign = lib.CoreSdk_AssignTrackerToUser(identity[0], int(user_id))
            role_result = lib.CoreSdk_AssignRoleToTracker(identity[0], role)
            if assign == self.success and role_result == self.success:
                self._vut_assigned_user = user_id
            else:
                print(
                    f"[manus_python_bridge] tracker assignment to user {user_id} returned {int(assign)}/{int(role_result)}",
                    file=sys.stderr,
                    flush=True,
                )
                self._vut_assigned_user = user_id  # do not spam; Core may not support it
        if self._vut_offset_user != user_id:
            offset = ffi.new("TrackerOffset*")
            offset.entryType = int(
                enums.TrackerOffsetType.LeftHandTrackerToCasing
                if self.options.required_hand == "left"
                else enums.TrackerOffsetType.RightHandTrackerToCasing
            )
            casing = self.options.tracker_to_casing
            offset.translation.x, offset.translation.y, offset.translation.z = (float(v) for v in casing.translation)
            x, y, z, w = (float(v) for v in casing.quaternion_xyzw)
            offset.rotation.x, offset.rotation.y, offset.rotation.z, offset.rotation.w = x, y, z, w
            result = lib.CoreSdk_SetTrackerOffset(int(user_id), offset)
            if result != self.success:
                print(
                    f"[manus_python_bridge] CoreSdk_SetTrackerOffset for user {user_id} returned {int(result)}",
                    file=sys.stderr,
                    flush=True,
                )
            self._vut_offset_user = user_id

    def _vut_tracker_record(self, pose: VutTrackerPose, user_id: int | None, *, selected: bool) -> dict[str, Any]:
        record = {
            "id": pose.tracker_id,
            "user_id": 0 if (user_id is None or not selected) else int(user_id),
            "is_hmd": False,
            "type": (f"{self.options.required_hand}_hand" if selected else "unknown"),
            "quality": pose.quality,
            "tracking_system": VUT_TRACKING_SYSTEM,
            "last_update_time": max(0, int(pose.timestamp_us)),
            "pose_valid": pose.pose_valid,
            "state": pose.state_name,
            "battery": int(pose.battery),
            "flags": int(pose.flags),
        }
        if pose.pose_valid:
            core_pose = vut_pose_to_core_world(
                pose.position, pose.quaternion_xyzw, frame=self.options.vive_ultimate_frame
            )
            record["position"] = [float(value) for value in core_pose.translation]
            record["orientation_xyzw"] = [float(value) for value in core_pose.quaternion_xyzw]
        return record

    def _emit_vut_trackers(self, pose: VutTrackerPose, user_id: int | None) -> None:
        requested = self.options.tracker_id
        with self._state:
            if not self._callback_is_current(self._connection_generation):
                return
            latest = self._vut_latest
            selected_id = requested if requested is not None else (
                latest.tracker_id if latest is not None else None
            )
            records = []
            if pose.tracker_id == selected_id:
                # Always report the selected tracker's *current* state, including
                # a loss of tracking, so the consumer's health gate sees it.
                records.append(self._vut_tracker_record(pose, user_id, selected=True))
            elif latest is not None and latest.tracker_id == selected_id:
                records.append(self._vut_tracker_record(latest, user_id, selected=True))
            for other in self._vut_other.values():
                if other.tracker_id != selected_id:
                    records.append(self._vut_tracker_record(other, user_id, selected=False))
            self._tracker_sequence += 1
            payload = {
                **self.envelope("trackers"),
                "sequence": self._tracker_sequence,
                "manus_publish_time": max(0, int(pose.timestamp_us)),
                "capture_monotonic_ns": pose.receipt_monotonic_ns,
                **self.connection_fields_locked(),
                "trackers": records,
            }
        self.emit(payload)

    # -- haptics command channel ---------------------------------------------

    def start_command_channel(self) -> None:
        if not self.options.haptics or self._stdin_thread is not None:
            return
        self._stdin_thread = threading.Thread(target=self._read_commands, name="manus-bridge-commands", daemon=True)
        self._stdin_thread.start()

    def _read_commands(self) -> None:
        stream = sys.stdin.buffer if hasattr(sys.stdin, "buffer") else sys.stdin
        try:
            for raw in stream:
                if self._stop.is_set():
                    break
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                if not line.strip():
                    continue
                try:
                    command = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[manus_python_bridge] ignoring malformed command {line.strip()!r}", file=sys.stderr, flush=True)
                    continue
                self._handle_command(command)
        except (OSError, ValueError):
            pass

    def _handle_command(self, command: Mapping[str, Any]) -> None:
        kind = command.get("type")
        if kind == "vibrate":
            side = str(command.get("side", self.options.required_hand))
            powers = command.get("powers")
            self.vibrate(side, powers)
        elif kind == "stop_haptics":
            self.stop_haptics()
        elif kind == "shutdown":
            self._stop.set()
            with self._state:
                self._state.notify_all()
        else:
            print(f"[manus_python_bridge] unknown command type {kind!r}", file=sys.stderr, flush=True)

    def vibrate(self, side: str, powers) -> bool:
        try:
            values = [min(1.0, max(0.0, float(value))) for value in powers]
        except (TypeError, ValueError):
            return False
        if len(values) != 5 or not _finite(*values):
            return False
        with self._state:
            glove_id = self._haptic_gloves.get(side)
        if glove_id is None:
            return False
        buffer = self.ffi.new("float[5]", values)
        return self.lib.CoreSdk_VibrateFingersForGlove(int(glove_id), buffer) == self.success

    def stop_haptics(self) -> None:
        with self._state:
            sides = list(self._haptic_gloves)
        for side in sides:
            self.vibrate(side, [0.0] * 5)

    # -- teardown -------------------------------------------------------------

    def shutdown(self) -> None:
        with self._state:
            self._stopping = True
            self._streaming_enabled = False
        if self._vut_feed is not None:
            self._vut_feed.close()
            self._vut_feed = None
        if self.options.haptics:
            try:
                self.stop_haptics()
            except Exception:  # noqa: BLE001
                pass
        self.unregister_callbacks()
        if self._connected:
            try:
                self.lib.CoreSdk_Disconnect()
            except Exception:  # noqa: BLE001
                pass
            self._connected = False
        if self._sdk_initialized:
            try:
                self.lib.CoreSdk_ShutDown()
            except Exception:  # noqa: BLE001
                pass
            self._sdk_initialized = False

    def request_stop(self) -> None:
        self._stop.set()
        with self._state:
            self._state.notify_all()


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_options(argv)
    if options.protocol_version:
        print(f"{PROTOCOL} {PROTOCOL_VERSION}")
        return 0
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="[manus_python_bridge] %(message)s")
    try:
        sdk = load_manus_sdk(options.sdk_python_path)
    except ImportError as error:
        print(f"[manus_python_bridge] {error}", file=sys.stderr, flush=True)
        return 3
    try:
        bridge = Bridge(options, sdk)
    except (OSError, ValueError) as error:
        print(f"[manus_python_bridge] {error}", file=sys.stderr, flush=True)
        return 2

    def _request_stop(_signum, _frame) -> None:
        bridge.request_stop()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGPIPE, _request_stop)
    return bridge.run()


if __name__ == "__main__":
    raise SystemExit(main())
