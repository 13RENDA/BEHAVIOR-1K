"""VIVE Ultimate Tracker poses from VIVE Hub for Linux, in pure Python.

VIVE Hub for Linux (HTC, beta since September 2026) pairs the Ultimate
trackers with their dongle, builds the inside-out tracking map and runs a
``tracker_server`` daemon that publishes every tracker pose over a Unix domain
socket (``/tmp/vut.sock``). Two wire protocols have shipped on that socket;
this module speaks both with the standard library only, so no VIVE C++ SDK, no
SteamVR and no Windows machine is involved in reading the trackers. The
protocol is detected from the socket type when connecting (``protocol="auto"``).

VIVE Hub 1.0.3 and newer: the **VUT SDK protocol** (``SOCK_SEQPACKET``), the
wire format of HTC's ``libvut_sdk`` (``sdk/include/vut/vut_types.h`` in the
VIVE Hub package). Every datagram is one message:

- header, 12 bytes: ``uint8 type, uint8 version, uint16 flags,
  uint32 request_id, uint32 payload_len``
- the client opens with ``HELLO`` (``uint16 client_version, uint16 reserved,
  uint32 subscribe_mask, char client_name[32]``) and receives ``HELLO_ACK``;
  ``LIST_DEVICES`` returns ``uint32 count`` followed by ``DeviceInfo``
  entries (``uint32 device_id, uint32 state, char name[32]``);
- ``EVT_POSE`` (100 bytes): ``uint32 device_id, uint32 flags,
  uint64 timestamp_ns, float position[3], float orientation[4] (qx qy qz qw),
  float velocity[3], float angular_velocity[3], char serial[32]``;
- ``EVT_STATUS`` (24 bytes): ``uint32 device_id, uint32 state,
  uint32 error_code, uint32 reserved, uint64 timestamp_ns``.

``device_id`` is the dongle slot plus one, ``timestamp_ns`` is the daemon's
``CLOCK_MONOTONIC`` (the same clock as :func:`time.monotonic_ns` on this
host), and poses are in the OpenXR/OpenVR frame (x right, y up, -z forward).
The daemon reports the tracker's raw status byte as ``state``: the pose status
(0..7, labelled as in the VIVE Hub UI) in bits 0-2 and the flag bits below in
bits 3-7; ``8`` means the tracker went off.

VIVE Hub 1.0.1 and HTC's standalone VIVE Ultimate Tracker SDK demo server
(``/tmp/vive_tracker.sock``): the **slot protocol** (``SOCK_STREAM``),
documented by ``tracker_ipc_client_ref.py`` in
``ViveRoboticsSoftware/vive-ultimate-tracker-sdk``:

- header, 16 bytes: ``uint32 Id, uint8 Msg, uint8 Payload, uint8 Params[10]``
- ``Msg``: Stop=0, Start=1, Add=2, Update=3, Click=4, Remove=5, Status=6
- Update payload, 64 bytes: ``float Location[3]; float Rotation[4] (qx qy qz qw);
  float Velocity[3]; float AngularVelocity[3]; int64 Timestamp_us; uint32 Flags``
  with ``Params = [State, Button, Clicks, Battery, ...]``
- Add payload: the tracker serial (not NUL-terminated); Remove/Start/Stop: none
- Status: ``Params[0:5]`` = per-slot ``TrackerState``, payload = command string
- commands are a 64-byte zero-padded string (``"Echo"``, ``"Setup"``, ...)

Both protocols decode into the same :class:`VutTrackerPose`: the slot
protocol's ``TrackerState`` vocabulary is kept and extended with the states
only the VUT SDK protocol reports, so ``state_name``, ``quality`` and the flag
predicates mean the same thing whichever daemon is running.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import errno
import logging
import socket
import struct
import threading
import time
from typing import Callable, Mapping

import numpy as np

from dex_teleop.tracking.transforms import RigidTransform, compose_transforms


LOGGER = logging.getLogger(__name__)

# VIVE Hub for Linux's tracker_server; the standalone SDK demo server uses
# /tmp/vive_tracker.sock with the slot protocol.
VUT_DEFAULT_SOCKET = "/tmp/vut.sock"
VUT_TRACKING_SYSTEM = "vive_hub_linux"
VUT_SLOT_COUNT = 5

# Wire protocols on the socket: "seqpacket" is the VUT SDK protocol of VIVE Hub
# 1.0.3+, "stream" the slot protocol of VIVE Hub 1.0.1 and the standalone SDK
# demo server; "auto" tries both (newest first) and keeps whichever connects.
VUT_PROTOCOL_SEQPACKET = "seqpacket"
VUT_PROTOCOL_STREAM = "stream"
VUT_PROTOCOLS = ("auto", VUT_PROTOCOL_SEQPACKET, VUT_PROTOCOL_STREAM)
VUT_DEFAULT_PROTOCOL = "auto"
_PROTOCOL_MISMATCH_ERRNOS = frozenset(
    code
    for code in (
        getattr(errno, "EPROTOTYPE", None),
        getattr(errno, "ESOCKTNOSUPPORT", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if code is not None
)

# ---------------------------------------------------------------------------
# Slot protocol (VIVE Hub 1.0.1, standalone SDK demo server)

MSG_STOP, MSG_START, MSG_ADD, MSG_UPDATE, MSG_CLICK, MSG_REMOVE, MSG_STATUS = range(7)
MSG_NAMES = {
    MSG_STOP: "stop",
    MSG_START: "start",
    MSG_ADD: "add",
    MSG_UPDATE: "update",
    MSG_CLICK: "click",
    MSG_REMOVE: "remove",
    MSG_STATUS: "status",
}

STATE_UNPAIRED, STATE_PAIRING, STATE_PAIRED, STATE_CONNECTING = 0, 1, 2, 3
STATE_CONNECTED, STATE_INITIALIZED, STATE_SYNCING, STATE_BUILD_MAP, STATE_READY = 4, 5, 6, 7, 8
# States only the VUT SDK protocol reports; numbered past the slot states so
# one VutTrackerPose.state vocabulary covers both daemons.
STATE_LOST_TRACKING, STATE_LOADING, STATE_PREPARING, STATE_NOT_CONNECTED, STATE_OFF = 9, 10, 11, 12, 13
TRACKER_STATE_NAMES = {
    STATE_UNPAIRED: "unpaired",
    STATE_PAIRING: "pairing",
    STATE_PAIRED: "paired",
    STATE_CONNECTING: "connecting",
    STATE_CONNECTED: "connected",
    STATE_INITIALIZED: "initialized",
    STATE_SYNCING: "syncing",
    STATE_BUILD_MAP: "build_map",
    STATE_READY: "ready",
    STATE_LOST_TRACKING: "lost_tracking",
    STATE_LOADING: "loading",
    STATE_PREPARING: "preparing",
    STATE_NOT_CONNECTED: "not_connected",
    STATE_OFF: "off",
}

FLAG_POSE_STATE_MASK = 0x07
FLAG_CHARGING = 1 << 3
FLAG_POSE_STABLE = 1 << 4
FLAG_ROTATION_ONLY = 1 << 5
FLAG_IS_HOST_TRACKER = 1 << 6
FLAG_SETUP_REQUIRED = 1 << 7

HEADER_FORMAT = "<IBB10s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
UPDATE_FORMAT = "<3f4f3f3fqI"
UPDATE_SIZE = struct.calcsize(UPDATE_FORMAT)
COMMAND_BUFFER_SIZE = 64
assert HEADER_SIZE == 16 and UPDATE_SIZE == 64

# ---------------------------------------------------------------------------
# VUT SDK protocol (VIVE Hub 1.0.3+)

SDK_PROTOCOL_VERSION = 1
SDK_MSG_HELLO, SDK_MSG_HELLO_ACK = 0x01, 0x02
SDK_MSG_LIST_DEVICES, SDK_MSG_LIST_DEVICES_RESP = 0x10, 0x11
SDK_MSG_GET_DEVICE_STATE, SDK_MSG_DEVICE_STATE_RESP = 0x12, 0x13
SDK_MSG_EVT_POSE, SDK_MSG_EVT_STATUS = 0x30, 0x31
SDK_MSG_SUBSCRIBE = 0x40
SDK_MSG_GET_STATS, SDK_MSG_STATS_RESP = 0x50, 0x51
SDK_MSG_NAMES = {
    SDK_MSG_HELLO: "hello",
    SDK_MSG_HELLO_ACK: "hello_ack",
    SDK_MSG_LIST_DEVICES: "list_devices",
    SDK_MSG_LIST_DEVICES_RESP: "list_devices_resp",
    SDK_MSG_GET_DEVICE_STATE: "get_device_state",
    SDK_MSG_DEVICE_STATE_RESP: "device_state_resp",
    SDK_MSG_EVT_POSE: "evt_pose",
    SDK_MSG_EVT_STATUS: "evt_status",
    SDK_MSG_SUBSCRIBE: "subscribe",
    SDK_MSG_GET_STATS: "get_stats",
    SDK_MSG_STATS_RESP: "stats_resp",
}
SDK_SUB_POSE, SDK_SUB_STATUS = 1 << 0, 1 << 1
SDK_SUB_ALL = SDK_SUB_POSE | SDK_SUB_STATUS
SDK_DEVICE_NAME_LEN = 32
SDK_MAX_PAYLOAD = 64 * 1024
SDK_CLIENT_NAME = "dex_teleop"
# Device.name before the daemon has read the serial ("VIVE Ultimate Tracker N").
SDK_PLACEHOLDER_NAME_PREFIX = "VIVE Ultimate Tracker"

SDK_HEADER_FORMAT = "<BBHII"
SDK_HEADER_SIZE = struct.calcsize(SDK_HEADER_FORMAT)
SDK_HELLO_FORMAT = "<HHI32s"
SDK_HELLO_SIZE = struct.calcsize(SDK_HELLO_FORMAT)
SDK_HELLO_ACK_FORMAT = "<HHIQ"
SDK_HELLO_ACK_SIZE = struct.calcsize(SDK_HELLO_ACK_FORMAT)
SDK_DEVICE_INFO_FORMAT = "<II32s"
SDK_DEVICE_INFO_SIZE = struct.calcsize(SDK_DEVICE_INFO_FORMAT)
SDK_POSE_FORMAT = "<IIQ3f4f3f3f32s"
SDK_POSE_SIZE = struct.calcsize(SDK_POSE_FORMAT)
SDK_STATUS_FORMAT = "<IIIIQ"
SDK_STATUS_SIZE = struct.calcsize(SDK_STATUS_FORMAT)
assert SDK_HEADER_SIZE == 12 and SDK_HELLO_SIZE == 40 and SDK_HELLO_ACK_SIZE == 16
assert SDK_DEVICE_INFO_SIZE == 40 and SDK_POSE_SIZE == 100 and SDK_STATUS_SIZE == 24

# vut::DeviceState: the tracker's POSE_STATUS (0..7) plus OFF, labelled as the
# VIVE Hub UI shows them. Poses flow while the state is READY..SYNCING.
DEV_NOT_CONNECTED, DEV_LOADING, DEV_READY, DEV_LOST_TRACKING = 0, 1, 2, 3
DEV_ROTATION_ONLY, DEV_PREPARING, DEV_SYNCING, DEV_SETUP_REQUIRED, DEV_OFF = 4, 5, 6, 7, 8
DEV_STATE_MASK = 0x07
DEV_STATE_NAMES = {
    DEV_NOT_CONNECTED: "Not connected",
    DEV_LOADING: "Loading",
    DEV_READY: "Ready",
    DEV_LOST_TRACKING: "Lost tracking",
    DEV_ROTATION_ONLY: "Rotation only",
    DEV_PREPARING: "Preparing",
    DEV_SYNCING: "Syncing",
    DEV_SETUP_REQUIRED: "Setup required",
    DEV_OFF: "Off",
}
_DEV_TO_TRACKER_STATE = {
    DEV_NOT_CONNECTED: (STATE_NOT_CONNECTED, 0),
    DEV_LOADING: (STATE_LOADING, 0),
    DEV_READY: (STATE_READY, 0),
    DEV_LOST_TRACKING: (STATE_LOST_TRACKING, 0),
    DEV_ROTATION_ONLY: (STATE_READY, FLAG_ROTATION_ONLY),
    DEV_PREPARING: (STATE_PREPARING, 0),
    DEV_SYNCING: (STATE_SYNCING, 0),
    DEV_SETUP_REQUIRED: (STATE_BUILD_MAP, FLAG_SETUP_REQUIRED),
}

# The VUT SDK protocol carries no battery level.
BATTERY_UNKNOWN = -1

# Pose axis conventions the daemon may report. VIVE Hub for Linux reports
# OpenXR axes (x right, y up, -z forward), which is also MANUS Core's world
# frame; HTC's standalone SDK documents Isaac Sim axes (x forward, z up).
VUT_FRAMES = ("x_right_y_up_z_back", "x_forward_z_up")
VUT_DEFAULT_FRAME = "x_right_y_up_z_back"
# x_forward_z_up -> x_right_y_up_z_back (inverse of the MANUS Core-world ->
# tracking-frame rotation used by ManusCoreCalibration.default()).
_X_FORWARD_Z_UP_TO_CORE_WORLD_XYZW = (-0.5, 0.5, 0.5, 0.5)


def default_tracker_id(slot: int) -> str:
    """Fallback identity for a tracker whose serial the daemon has not reported."""

    return f"vive_tracker_{int(slot)}"


def decode_device_state(raw_state: int) -> tuple[int, int]:
    """Map a VUT SDK ``DeviceState`` value to ``(tracker state, flags)``.

    ``tracker_server`` 1.0.3 forwards the tracker's raw status byte: the pose
    status in bits 0-2 and, above it, the same flag bits the slot protocol's
    ``Update.Flags`` carries (charging, stable, rotation-only, host, setup
    required). ``DEV_OFF`` is set by the daemon itself when a tracker leaves.
    """

    raw = int(raw_state) & 0xFF
    if raw == DEV_OFF:
        return STATE_OFF, 0
    state, implied = _DEV_TO_TRACKER_STATE[raw & DEV_STATE_MASK]
    return state, (raw & ~DEV_STATE_MASK & 0xFF) | implied


@dataclass(frozen=True)
class VutTrackerPose:
    """One decoded pose plus the identity and state the daemon reported for it."""

    slot: int
    tracker_id: str
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    timestamp_us: int
    flags: int
    state: int
    button: int
    clicks: int
    battery: int
    receipt_monotonic_ns: int
    protocol: str = VUT_PROTOCOL_STREAM

    @property
    def state_name(self) -> str:
        return TRACKER_STATE_NAMES.get(self.state, f"unknown_{self.state}")

    @property
    def rotation_only(self) -> bool:
        return bool(self.flags & FLAG_ROTATION_ONLY)

    @property
    def setup_required(self) -> bool:
        return bool(self.flags & FLAG_SETUP_REQUIRED)

    @property
    def pose_stable(self) -> bool:
        return bool(self.flags & FLAG_POSE_STABLE)

    @property
    def is_host_tracker(self) -> bool:
        return bool(self.flags & FLAG_IS_HOST_TRACKER)

    @property
    def battery_known(self) -> bool:
        return self.battery != BATTERY_UNKNOWN

    @property
    def quality(self) -> str:
        """MANUS-style tracking quality: trackable, bad, or untrackable."""

        if self.state != STATE_READY or self.setup_required:
            return "untrackable"
        if self.rotation_only:
            return "bad"
        return "trackable"

    @property
    def pose_valid(self) -> bool:
        return bool(
            np.isfinite(self.position).all()
            and np.isfinite(self.quaternion_xyzw).all()
            and float(np.linalg.norm(self.quaternion_xyzw)) > 0.0
        )

    def as_mapping(self) -> dict:
        return {
            "slot": self.slot,
            "tracker_id": self.tracker_id,
            "position": self.position.tolist(),
            "quaternion_xyzw": self.quaternion_xyzw.tolist(),
            "timestamp_us": self.timestamp_us,
            "flags": self.flags,
            "state": self.state_name,
            "quality": self.quality,
            "battery": self.battery,
            "receipt_monotonic_ns": self.receipt_monotonic_ns,
            "protocol": self.protocol,
        }


def decode_update(
    slot: int,
    params: bytes,
    payload: bytes,
    *,
    tracker_id: str,
    receipt_monotonic_ns: int,
) -> VutTrackerPose:
    """Decode a slot-protocol Update payload exactly like the C++ ``Poll()`` does."""

    if len(payload) < UPDATE_SIZE:
        raise ValueError(
            f"VUT update payload must contain {UPDATE_SIZE} bytes, got {len(payload)}"
        )
    values = struct.unpack(UPDATE_FORMAT, payload[:UPDATE_SIZE])
    return VutTrackerPose(
        slot=slot,
        tracker_id=tracker_id,
        position=np.asarray(values[0:3], dtype=np.float64),
        quaternion_xyzw=np.asarray(values[3:7], dtype=np.float64),
        velocity=np.asarray(values[7:10], dtype=np.float64),
        angular_velocity=np.asarray(values[10:13], dtype=np.float64),
        timestamp_us=int(values[13]),
        flags=int(values[14]),
        state=params[0],
        button=params[1],
        clicks=params[2],
        battery=params[3],
        receipt_monotonic_ns=receipt_monotonic_ns,
        protocol=VUT_PROTOCOL_STREAM,
    )


def encode_header(tracker_id: int, message: int, payload_size: int, params: bytes = b"") -> bytes:
    """Build a slot-protocol header; used by tests and diagnostics servers."""

    params = bytes(params)[:10].ljust(10, b"\0")
    return struct.pack(HEADER_FORMAT, tracker_id, message, payload_size, params)


def encode_update_payload(
    position,
    quaternion_xyzw,
    *,
    timestamp_us: int,
    flags: int = 0,
    velocity=(0.0, 0.0, 0.0),
    angular_velocity=(0.0, 0.0, 0.0),
) -> bytes:
    return struct.pack(
        UPDATE_FORMAT,
        *(float(value) for value in position),
        *(float(value) for value in quaternion_xyzw),
        *(float(value) for value in velocity),
        *(float(value) for value in angular_velocity),
        int(timestamp_us),
        int(flags),
    )


def _c_string(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace").strip()


def _c_bytes(text: str, size: int) -> bytes:
    return text.encode("utf-8")[: size - 1].ljust(size, b"\0")


def decode_pose_event(
    payload: bytes,
    *,
    raw_state: int,
    fallback_tracker_id: str | None = None,
    receipt_monotonic_ns: int,
) -> VutTrackerPose:
    """Decode a VUT SDK ``EVT_POSE`` payload.

    The event carries no state, so the caller supplies the device's latest
    ``DeviceState`` (from ``LIST_DEVICES_RESP``/``EVT_STATUS``); its serial is
    used when present, ``fallback_tracker_id`` (or the slot default) otherwise.
    """

    if len(payload) < SDK_POSE_SIZE:
        raise ValueError(
            f"VUT pose event must contain {SDK_POSE_SIZE} bytes, got {len(payload)}"
        )
    values = struct.unpack(SDK_POSE_FORMAT, payload[:SDK_POSE_SIZE])
    device_id = int(values[0])
    if device_id <= 0:
        raise ValueError("VUT pose event names the reserved broadcast device id 0")
    slot = device_id - 1
    serial = _c_string(values[16])
    state, flags = decode_device_state(raw_state)
    return VutTrackerPose(
        slot=slot,
        tracker_id=serial or fallback_tracker_id or default_tracker_id(slot),
        position=np.asarray(values[3:6], dtype=np.float64),
        quaternion_xyzw=np.asarray(values[6:10], dtype=np.float64),
        velocity=np.asarray(values[10:13], dtype=np.float64),
        angular_velocity=np.asarray(values[13:16], dtype=np.float64),
        timestamp_us=int(values[2]) // 1000,
        flags=flags,
        state=state,
        button=0,
        clicks=0,
        battery=BATTERY_UNKNOWN,
        receipt_monotonic_ns=receipt_monotonic_ns,
        protocol=VUT_PROTOCOL_SEQPACKET,
    )


def decode_status_event(payload: bytes) -> tuple[int, int, int, int]:
    """Return ``(device_id, raw_state, error_code, timestamp_ns)`` of an ``EVT_STATUS``."""

    if len(payload) < SDK_STATUS_SIZE:
        raise ValueError(
            f"VUT status event must contain {SDK_STATUS_SIZE} bytes, got {len(payload)}"
        )
    device_id, state, error_code, _reserved, timestamp_ns = struct.unpack(
        SDK_STATUS_FORMAT, payload[:SDK_STATUS_SIZE]
    )
    return int(device_id), int(state), int(error_code), int(timestamp_ns)


def decode_device_list(payload: bytes) -> list[tuple[int, int, str]]:
    """Return ``(device_id, raw_state, name)`` entries of a ``LIST_DEVICES_RESP``."""

    if len(payload) < 4:
        raise ValueError("VUT device list is missing its count")
    (count,) = struct.unpack_from("<I", payload)
    if len(payload) < 4 + count * SDK_DEVICE_INFO_SIZE:
        raise ValueError(f"VUT device list announces {count} devices but is truncated")
    devices = []
    for index in range(count):
        device_id, state, name = struct.unpack_from(
            SDK_DEVICE_INFO_FORMAT, payload, 4 + index * SDK_DEVICE_INFO_SIZE
        )
        devices.append((int(device_id), int(state), _c_string(name)))
    return devices


def encode_sdk_header(message: int, payload_size: int, *, request_id: int = 0, flags: int = 0) -> bytes:
    """Build a VUT SDK message header; used by the client, tests and fake daemons."""

    return struct.pack(
        SDK_HEADER_FORMAT, int(message), SDK_PROTOCOL_VERSION, int(flags), int(request_id), int(payload_size)
    )


def encode_hello(subscribe_mask: int = SDK_SUB_ALL, client_name: str = SDK_CLIENT_NAME) -> bytes:
    return struct.pack(
        SDK_HELLO_FORMAT, SDK_PROTOCOL_VERSION, 0, int(subscribe_mask), _c_bytes(client_name, SDK_DEVICE_NAME_LEN)
    )


def encode_hello_ack(session_id: int, server_time_ns: int) -> bytes:
    return struct.pack(SDK_HELLO_ACK_FORMAT, SDK_PROTOCOL_VERSION, 0, int(session_id), int(server_time_ns))


def encode_device_list(devices) -> bytes:
    """Encode ``(device_id, raw_state, name)`` entries as a ``LIST_DEVICES_RESP`` payload."""

    entries = list(devices)
    payload = struct.pack("<I", len(entries))
    for device_id, state, name in entries:
        payload += struct.pack(SDK_DEVICE_INFO_FORMAT, int(device_id), int(state), _c_bytes(name, SDK_DEVICE_NAME_LEN))
    return payload


def encode_pose_event(
    device_id: int,
    position,
    quaternion_xyzw,
    *,
    timestamp_ns: int,
    serial: str = "",
    flags: int = 0,
    velocity=(0.0, 0.0, 0.0),
    angular_velocity=(0.0, 0.0, 0.0),
) -> bytes:
    return struct.pack(
        SDK_POSE_FORMAT,
        int(device_id),
        int(flags),
        int(timestamp_ns),
        *(float(value) for value in position),
        *(float(value) for value in quaternion_xyzw),
        *(float(value) for value in velocity),
        *(float(value) for value in angular_velocity),
        _c_bytes(serial, SDK_DEVICE_NAME_LEN),
    )


def encode_status_event(device_id: int, raw_state: int, *, timestamp_ns: int, error_code: int = 0) -> bytes:
    return struct.pack(SDK_STATUS_FORMAT, int(device_id), int(raw_state), int(error_code), 0, int(timestamp_ns))


def vut_pose_to_core_world(
    position, quaternion_xyzw, *, frame: str = VUT_DEFAULT_FRAME
) -> RigidTransform:
    """Express a daemon pose in MANUS Core's world axes (x right, y up, -z forward)."""

    if frame not in VUT_FRAMES:
        raise ValueError(f"VUT frame must be one of {VUT_FRAMES}, got {frame!r}")
    pose = RigidTransform(position, quaternion_xyzw)
    if frame == "x_right_y_up_z_back":
        return pose
    rotation = RigidTransform((0.0, 0.0, 0.0), _X_FORWARD_Z_UP_TO_CORE_WORLD_XYZW)
    return compose_transforms(rotation, pose)


class VutIpcClient:
    """Non-blocking client for the tracker_server Unix socket, either protocol.

    Call :meth:`poll` regularly; it decodes every pending packet and invokes
    the ``on_*`` callbacks on the calling thread, mirroring
    ``vive::ITrackerIPCClient::Poll()`` and ``vut::Client``. ``protocol``
    selects the wire protocol; ``"auto"`` tries the VUT SDK protocol first and
    falls back to the slot protocol when the socket type does not match.
    """

    def __init__(
        self,
        socket_path: str = VUT_DEFAULT_SOCKET,
        *,
        protocol: str = VUT_DEFAULT_PROTOCOL,
        client_name: str = SDK_CLIENT_NAME,
    ) -> None:
        if protocol not in VUT_PROTOCOLS:
            raise ValueError(f"VUT protocol must be one of {VUT_PROTOCOLS}, got {protocol!r}")
        self.socket_path = str(socket_path)
        self.requested_protocol = protocol
        self.client_name = str(client_name)
        #: Wire protocol negotiated by the last successful :meth:`connect`.
        self.protocol: str | None = None
        self.session_id: int | None = None
        self.server_version: int | None = None
        self._socket: socket.socket | None = None
        self._buffer = bytearray()
        self._request_id = 0
        self.serials: dict[int, str] = {}
        self.states: dict[int, int] = {}
        self._raw_states: dict[int, int] = {}
        self._latest: dict[int, VutTrackerPose] = {}

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def _candidate_protocols(self) -> tuple[str, ...]:
        if self.requested_protocol == "auto":
            return (VUT_PROTOCOL_SEQPACKET, VUT_PROTOCOL_STREAM)
        return (self.requested_protocol,)

    def connect(self) -> bool:
        if self._socket is not None:
            return True
        candidates = self._candidate_protocols()
        for protocol in candidates:
            kind = socket.SOCK_SEQPACKET if protocol == VUT_PROTOCOL_SEQPACKET else socket.SOCK_STREAM
            candidate = socket.socket(socket.AF_UNIX, kind)
            try:
                candidate.connect(self.socket_path)
            except OSError as error:
                candidate.close()
                if error.errno in _PROTOCOL_MISMATCH_ERRNOS and len(candidates) > 1:
                    continue
                return False
            self._socket = candidate
            self.protocol = protocol
            break
        else:
            return False
        self._buffer.clear()
        self.session_id = None
        self.server_version = None
        if self.protocol == VUT_PROTOCOL_SEQPACKET:
            try:
                self._send_sdk(SDK_MSG_HELLO, encode_hello(SDK_SUB_ALL, self.client_name))
                self._send_sdk(SDK_MSG_LIST_DEVICES)
            except OSError as error:
                LOGGER.warning("VUT SDK handshake failed: %s", error)
                self.disconnect()
                return False
        self._socket.setblocking(False)
        return True

    def disconnect(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        self._socket = None
        self._buffer.clear()

    def poll(self) -> bool:
        """Decode every pending packet. Returns False while not connected."""

        if self._socket is None and not self.connect():
            return False
        assert self._socket is not None
        try:
            if self.protocol == VUT_PROTOCOL_SEQPACKET:
                self._poll_seqpacket()
            else:
                self._poll_stream()
        except (ConnectionError, OSError) as error:
            LOGGER.warning("VUT tracker server disconnected: %s", error)
            self.disconnect()
            self.on_server_stop()
            return False
        return True

    # -- slot protocol -------------------------------------------------------

    def _poll_stream(self) -> None:
        assert self._socket is not None
        while True:
            try:
                chunk = self._socket.recv(65536)
            except BlockingIOError:
                break
            if not chunk:
                raise ConnectionError("tracker server closed the connection")
            self._buffer.extend(chunk)
        while len(self._buffer) >= HEADER_SIZE:
            tracker_id, message, payload_size, params = struct.unpack(
                HEADER_FORMAT, bytes(self._buffer[:HEADER_SIZE])
            )
            total = HEADER_SIZE + payload_size
            if len(self._buffer) < total:
                break
            payload = bytes(self._buffer[HEADER_SIZE:total])
            del self._buffer[:total]
            self._dispatch(tracker_id, message, params, payload)

    def _dispatch(self, tracker_id: int, message: int, params: bytes, payload: bytes) -> None:
        receipt = time.monotonic_ns()
        if message == MSG_ADD:
            serial = _c_string(payload)
            self.serials[tracker_id] = serial or default_tracker_id(tracker_id)
            self.on_add(tracker_id, serial or None)
        elif message == MSG_UPDATE:
            if len(payload) < UPDATE_SIZE:
                LOGGER.warning("Dropping short VUT update (%d bytes)", len(payload))
                return
            identity = self.serials.get(tracker_id) or default_tracker_id(tracker_id)
            pose = decode_update(
                tracker_id,
                params,
                payload,
                tracker_id=identity,
                receipt_monotonic_ns=receipt,
            )
            self.states[tracker_id] = pose.state
            self._latest[tracker_id] = pose
            self.on_update(pose)
        elif message == MSG_CLICK:
            self.on_click(tracker_id, params[0], params[1])
        elif message == MSG_REMOVE:
            self.states.pop(tracker_id, None)
            self._latest.pop(tracker_id, None)
            self.on_remove(tracker_id)
        elif message == MSG_START:
            self.on_server_start()
        elif message == MSG_STOP:
            self.on_server_stop()
        elif message == MSG_STATUS:
            command = _c_string(payload)
            states = list(params[:VUT_SLOT_COUNT])
            result = "failed" if tracker_id == 0 else "ignored" if tracker_id == 1 else "ok"
            self.on_command_response(command, result, states)
        else:
            LOGGER.warning("Unknown VUT message type %d", message)

    # -- VUT SDK protocol ----------------------------------------------------

    def _send_sdk(self, message: int, payload: bytes = b"") -> int:
        assert self._socket is not None
        self._request_id += 1
        self._socket.sendall(encode_sdk_header(message, len(payload), request_id=self._request_id) + payload)
        return self._request_id

    def _poll_seqpacket(self) -> None:
        assert self._socket is not None
        while True:
            try:
                datagram = self._socket.recv(SDK_HEADER_SIZE + SDK_MAX_PAYLOAD)
            except BlockingIOError:
                break
            if not datagram:
                raise ConnectionError("tracker server closed the connection")
            if len(datagram) < SDK_HEADER_SIZE:
                LOGGER.warning("Dropping short VUT SDK datagram (%d bytes)", len(datagram))
                continue
            message, _version, _flags, _request_id, payload_size = struct.unpack(
                SDK_HEADER_FORMAT, datagram[:SDK_HEADER_SIZE]
            )
            self._dispatch_sdk(message, datagram[SDK_HEADER_SIZE : SDK_HEADER_SIZE + payload_size])

    def _dispatch_sdk(self, message: int, payload: bytes) -> None:
        receipt = time.monotonic_ns()
        if message == SDK_MSG_HELLO_ACK:
            if len(payload) >= SDK_HELLO_ACK_SIZE:
                server_version, _reserved, session_id, _server_time_ns = struct.unpack(
                    SDK_HELLO_ACK_FORMAT, payload[:SDK_HELLO_ACK_SIZE]
                )
                self.server_version = int(server_version)
                self.session_id = int(session_id)
            self.on_server_start()
        elif message == SDK_MSG_LIST_DEVICES_RESP:
            try:
                devices = decode_device_list(payload)
            except ValueError as error:
                LOGGER.warning("Dropping malformed VUT device list: %s", error)
                return
            for device_id, raw_state, name in devices:
                if device_id <= 0:
                    continue
                self._register_sdk_device(device_id - 1, raw_state, name)
        elif message == SDK_MSG_EVT_POSE:
            if len(payload) < SDK_POSE_SIZE:
                LOGGER.warning("Dropping short VUT pose event (%d bytes)", len(payload))
                return
            (device_id,) = struct.unpack_from("<I", payload)
            if device_id <= 0:
                return
            slot = device_id - 1
            serial = _c_string(payload[SDK_POSE_SIZE - SDK_DEVICE_NAME_LEN : SDK_POSE_SIZE])
            if slot not in self.states or (serial and self.serials.get(slot) != serial):
                self._register_sdk_device(slot, None, serial)
            pose = decode_pose_event(
                payload,
                raw_state=self._raw_states.get(slot, DEV_NOT_CONNECTED),
                fallback_tracker_id=self.serials.get(slot),
                receipt_monotonic_ns=receipt,
            )
            self._latest[slot] = pose
            self.on_update(pose)
        elif message == SDK_MSG_EVT_STATUS:
            try:
                device_id, raw_state, error_code, _timestamp_ns = decode_status_event(payload)
            except ValueError as error:
                LOGGER.warning("Dropping malformed VUT status event: %s", error)
                return
            if device_id <= 0:
                return
            slot = device_id - 1
            previous = self._raw_states.get(slot)
            state, flags = decode_device_state(raw_state)
            self._raw_states[slot] = raw_state
            self.states[slot] = state
            self.on_status(slot, state, error_code)
            if state == STATE_OFF:
                self._latest.pop(slot, None)
                self.on_remove(slot)
            elif previous != raw_state and slot in self._latest:
                # Poses carry no state, so re-publish the last pose under the
                # new state; a loss of tracking then reaches listeners at once.
                refreshed = replace(self._latest[slot], state=state, flags=flags, receipt_monotonic_ns=receipt)
                self._latest[slot] = refreshed
                self.on_update(refreshed)
        elif message in (SDK_MSG_DEVICE_STATE_RESP, SDK_MSG_STATS_RESP):
            self.on_command_response(SDK_MSG_NAMES[message], "ok", list(self.states.values()))
        else:
            LOGGER.warning("Unknown VUT SDK message type %#x", message)

    def _register_sdk_device(self, slot: int, raw_state: int | None, name: str) -> None:
        serial = name if name and not name.startswith(SDK_PLACEHOLDER_NAME_PREFIX) else None
        new_slot = slot not in self.states
        if raw_state is not None:
            self._raw_states[slot] = raw_state
            self.states[slot] = decode_device_state(raw_state)[0]
        else:
            self._raw_states.setdefault(slot, DEV_NOT_CONNECTED)
            self.states.setdefault(slot, STATE_NOT_CONNECTED)
        previous_serial = self.serials.get(slot)
        self.serials[slot] = serial or previous_serial or default_tracker_id(slot)
        if new_slot or (serial is not None and serial != previous_serial):
            self.on_add(slot, serial)

    def request_device_list(self) -> bool:
        """Ask a VUT SDK daemon to re-send its device list (no-op on the slot protocol)."""

        if self._socket is None or self.protocol != VUT_PROTOCOL_SEQPACKET:
            return False
        try:
            self._send_sdk(SDK_MSG_LIST_DEVICES)
            return True
        except OSError as error:
            LOGGER.warning("VUT device list request failed: %s", error)
            self.disconnect()
            return False

    def send_command(self, command: str) -> bool:
        """Send a slot-protocol command (``Echo``, ``Setup``, ``Pairing 0``, ...).

        The VUT SDK protocol has no command channel: pairing and map setup are
        done in the VIVE Hub UI, so the call returns False on that daemon.
        """

        if self._socket is None:
            return False
        if self.protocol == VUT_PROTOCOL_SEQPACKET:
            LOGGER.warning(
                "VIVE Hub's VUT SDK protocol accepts no tracker_server commands (%r); use the VIVE Hub UI",
                command,
            )
            return False
        encoded = command.encode("utf-8")[:COMMAND_BUFFER_SIZE]
        buffer = encoded.ljust(COMMAND_BUFFER_SIZE, b"\0")
        try:
            self._socket.setblocking(True)
            self._socket.sendall(buffer)
            return True
        except OSError as error:
            LOGGER.warning("VUT command failed: %s", error)
            self.disconnect()
            return False
        finally:
            if self._socket is not None:
                self._socket.setblocking(False)

    # Override in subclasses or assign instance attributes.
    def on_server_start(self) -> None:
        pass

    def on_server_stop(self) -> None:
        pass

    def on_add(self, tracker_id: int, serial: str | None) -> None:
        pass

    def on_update(self, pose: VutTrackerPose) -> None:
        pass

    def on_status(self, tracker_id: int, state: int, error_code: int) -> None:
        pass

    def on_click(self, tracker_id: int, button_id: int, clicks: int) -> None:
        pass

    def on_remove(self, tracker_id: int) -> None:
        pass

    def on_command_response(self, command: str, result: str, states: list[int]) -> None:
        pass


class VutPoseFeed:
    """Background poller that keeps the latest pose per tracker.

    ``listener`` is invoked on the polling thread for every Update packet; it
    must return quickly. Reconnects automatically while the daemon is absent.
    """

    def __init__(
        self,
        socket_path: str = VUT_DEFAULT_SOCKET,
        *,
        poll_hz: float = 240.0,
        listener: Callable[[VutTrackerPose], None] | None = None,
        reconnect_interval: float = 0.5,
        protocol: str = VUT_DEFAULT_PROTOCOL,
    ) -> None:
        if poll_hz <= 0.0:
            raise ValueError("poll_hz must be positive")
        self.socket_path = str(socket_path)
        self._interval = 1.0 / float(poll_hz)
        self._reconnect_interval = float(reconnect_interval)
        self.listener = listener
        self._client = VutIpcClient(self.socket_path, protocol=protocol)
        self._client.on_update = self._handle_update
        self._client.on_add = self._handle_add
        self._client.on_remove = self._handle_remove
        self._client.on_server_stop = self._handle_server_stop
        self._lock = threading.Lock()
        self._latest: dict[int, VutTrackerPose] = {}
        self._serials: dict[int, str] = {}
        self._connected = False
        self._connected_once = False
        self._update_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="vive-ultimate-vut", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._client.disconnect()
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            connected = self._client.poll()
            with self._lock:
                was_connected = self._connected
                self._connected = connected
                if connected:
                    self._connected_once = True
            if connected and not was_connected:
                LOGGER.info(
                    "Connected to VIVE Ultimate tracker server %s (%s protocol)",
                    self.socket_path,
                    self._client.protocol,
                )
            self._stop.wait(self._interval if connected else self._reconnect_interval)

    def _handle_add(self, tracker_id: int, serial: str | None) -> None:
        with self._lock:
            self._serials[tracker_id] = serial or default_tracker_id(tracker_id)

    def _handle_remove(self, tracker_id: int) -> None:
        with self._lock:
            self._latest.pop(tracker_id, None)

    def _handle_server_stop(self) -> None:
        with self._lock:
            self._latest.clear()

    def _handle_update(self, pose: VutTrackerPose) -> None:
        with self._lock:
            self._latest[pose.slot] = pose
            self._update_count += 1
        listener = self.listener
        if listener is not None:
            listener(pose)

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def connected_once(self) -> bool:
        with self._lock:
            return self._connected_once

    @property
    def protocol(self) -> str | None:
        """Wire protocol of the last successful connection (``None`` before one)."""

        return self._client.protocol

    @property
    def update_count(self) -> int:
        with self._lock:
            return self._update_count

    def snapshot(self) -> Mapping[int, VutTrackerPose]:
        with self._lock:
            return dict(self._latest)

    def serials(self) -> Mapping[int, str]:
        with self._lock:
            return dict(self._serials)

    def states(self) -> Mapping[int, int]:
        """Latest tracker state per slot as the daemon reported it (``TRACKER_STATE_NAMES`` codes).

        Unlike :meth:`snapshot`, this also covers trackers that are listed but
        stream no poses (asleep, off, or waiting for map setup).
        """

        return dict(self._client.states)

    def latest(self, tracker_id: str) -> VutTrackerPose | None:
        """The newest pose of the tracker with this serial (or fallback id)."""

        with self._lock:
            for pose in self._latest.values():
                if pose.tracker_id == tracker_id:
                    return pose
        return None

    def send_command(self, command: str) -> bool:
        return self._client.send_command(command)


def select_tracker(
    poses: Mapping[int, VutTrackerPose], requested_id: str | None
) -> VutTrackerPose | None:
    """Pick the requested tracker, or the single tracker when none is requested."""

    if requested_id is not None:
        for pose in poses.values():
            if pose.tracker_id == requested_id:
                return pose
        return None
    if len(poses) == 1:
        return next(iter(poses.values()))
    return None


__all__ = [
    "BATTERY_UNKNOWN",
    "DEV_LOADING",
    "DEV_LOST_TRACKING",
    "DEV_NOT_CONNECTED",
    "DEV_OFF",
    "DEV_PREPARING",
    "DEV_READY",
    "DEV_ROTATION_ONLY",
    "DEV_SETUP_REQUIRED",
    "DEV_STATE_NAMES",
    "DEV_SYNCING",
    "FLAG_IS_HOST_TRACKER",
    "FLAG_POSE_STABLE",
    "FLAG_ROTATION_ONLY",
    "FLAG_SETUP_REQUIRED",
    "HEADER_SIZE",
    "MSG_ADD",
    "MSG_CLICK",
    "MSG_REMOVE",
    "MSG_START",
    "MSG_STATUS",
    "MSG_STOP",
    "MSG_UPDATE",
    "SDK_HEADER_SIZE",
    "SDK_MSG_EVT_POSE",
    "SDK_MSG_EVT_STATUS",
    "SDK_MSG_HELLO",
    "SDK_MSG_HELLO_ACK",
    "SDK_MSG_LIST_DEVICES",
    "SDK_MSG_LIST_DEVICES_RESP",
    "SDK_POSE_SIZE",
    "SDK_STATUS_SIZE",
    "SDK_SUB_ALL",
    "STATE_LOST_TRACKING",
    "STATE_OFF",
    "STATE_READY",
    "STATE_SYNCING",
    "TRACKER_STATE_NAMES",
    "UPDATE_SIZE",
    "VUT_DEFAULT_FRAME",
    "VUT_DEFAULT_PROTOCOL",
    "VUT_DEFAULT_SOCKET",
    "VUT_FRAMES",
    "VUT_PROTOCOLS",
    "VUT_PROTOCOL_SEQPACKET",
    "VUT_PROTOCOL_STREAM",
    "VUT_TRACKING_SYSTEM",
    "VutIpcClient",
    "VutPoseFeed",
    "VutTrackerPose",
    "decode_device_list",
    "decode_device_state",
    "decode_pose_event",
    "decode_status_event",
    "decode_update",
    "default_tracker_id",
    "encode_device_list",
    "encode_header",
    "encode_hello",
    "encode_hello_ack",
    "encode_pose_event",
    "encode_sdk_header",
    "encode_status_event",
    "encode_update_payload",
    "select_tracker",
    "vut_pose_to_core_world",
]
