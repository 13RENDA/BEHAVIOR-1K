"""VIVE Ultimate Tracker poses from VIVE Hub for Linux, in pure Python.

VIVE Hub for Linux (HTC, beta since September 2026) pairs the Ultimate
trackers with their dongle, builds the inside-out tracking map and runs a
``tracker_server`` daemon that publishes every tracker pose over a Unix domain
socket (``/tmp/vut.sock``). HTC's VIVE Ultimate Tracker SDK documents the wire
protocol of that socket and ships a Python reference client for it
(``tracker_ipc_client_ref.py`` in ``ViveRoboticsSoftware/vive-ultimate-tracker-sdk``);
the standalone SDK's demo server speaks the same protocol on
``/tmp/vive_tracker.sock``. This module implements that protocol with the
standard library only, so no VIVE C++ SDK, no SteamVR and no Windows machine is
involved in reading the trackers.

Wire protocol (little-endian, packed):

- header, 16 bytes: ``uint32 Id, uint8 Msg, uint8 Payload, uint8 Params[10]``
- ``Msg``: Stop=0, Start=1, Add=2, Update=3, Click=4, Remove=5, Status=6
- Update payload, 64 bytes: ``float Location[3]; float Rotation[4] (qx qy qz qw);
  float Velocity[3]; float AngularVelocity[3]; int64 Timestamp_us; uint32 Flags``
  with ``Params = [State, Button, Clicks, Battery, ...]``
- Add payload: the tracker serial (not NUL-terminated); Remove/Start/Stop: none
- Status: ``Params[0:5]`` = per-slot ``TrackerState``, payload = command string
- commands are a 64-byte zero-padded string (``"Echo"``, ``"Setup"``, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
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
# /tmp/vive_tracker.sock with the same protocol.
VUT_DEFAULT_SOCKET = "/tmp/vut.sock"
VUT_TRACKING_SYSTEM = "vive_hub_linux"
VUT_SLOT_COUNT = 5

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

# Pose axis conventions the daemon may report. VIVE Hub for Linux reports
# OpenXR axes (x right, y up, -z forward), which is also MANUS Core's world
# frame; HTC's standalone SDK documents Isaac Sim axes (x forward, z up).
VUT_FRAMES = ("x_right_y_up_z_back", "x_forward_z_up")
VUT_DEFAULT_FRAME = "x_right_y_up_z_back"
# x_forward_z_up -> x_right_y_up_z_back (inverse of the MANUS Core-world ->
# tracking-frame rotation used by ManusCoreCalibration.default()).
_X_FORWARD_Z_UP_TO_CORE_WORLD_XYZW = (-0.5, 0.5, 0.5, 0.5)


def default_tracker_id(slot: int) -> str:
    """Fallback identity for a tracker whose Add packet carried no serial."""

    return f"vive_tracker_{int(slot)}"


@dataclass(frozen=True)
class VutTrackerPose:
    """One decoded Update packet plus the identity learned from its Add packet."""

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
        }


def decode_update(
    slot: int,
    params: bytes,
    payload: bytes,
    *,
    tracker_id: str,
    receipt_monotonic_ns: int,
) -> VutTrackerPose:
    """Decode an Update payload exactly like the C++ ``Poll()`` does."""

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
    )


def encode_header(tracker_id: int, message: int, payload_size: int, params: bytes = b"") -> bytes:
    """Build a protocol header; used by tests and diagnostics servers."""

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
    """Non-blocking client for the tracker_server Unix socket.

    Call :meth:`poll` regularly; it decodes every pending packet and invokes
    the ``on_*`` callbacks on the calling thread, mirroring
    ``vive::ITrackerIPCClient::Poll()``.
    """

    def __init__(self, socket_path: str = VUT_DEFAULT_SOCKET) -> None:
        self.socket_path = str(socket_path)
        self._socket: socket.socket | None = None
        self._buffer = bytearray()
        self.serials: dict[int, str] = {}
        self.states: dict[int, int] = {}

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> bool:
        if self._socket is not None:
            return True
        try:
            candidate = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            candidate.connect(self.socket_path)
            candidate.setblocking(False)
        except OSError:
            return False
        self._socket = candidate
        self._buffer.clear()
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
            while True:
                try:
                    chunk = self._socket.recv(65536)
                except BlockingIOError:
                    break
                if not chunk:
                    raise ConnectionError("tracker server closed the connection")
                self._buffer.extend(chunk)
        except (ConnectionError, OSError) as error:
            LOGGER.warning("VUT tracker server disconnected: %s", error)
            self.disconnect()
            self.on_server_stop()
            return False
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
        return True

    def _dispatch(self, tracker_id: int, message: int, params: bytes, payload: bytes) -> None:
        receipt = time.monotonic_ns()
        if message == MSG_ADD:
            serial = payload.split(b"\0", 1)[0].decode("utf-8", "replace").strip()
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
            self.on_update(pose)
        elif message == MSG_CLICK:
            self.on_click(tracker_id, params[0], params[1])
        elif message == MSG_REMOVE:
            self.states.pop(tracker_id, None)
            self.on_remove(tracker_id)
        elif message == MSG_START:
            self.on_server_start()
        elif message == MSG_STOP:
            self.on_server_stop()
        elif message == MSG_STATUS:
            command = payload.split(b"\0", 1)[0].decode("utf-8", "replace")
            states = list(params[:VUT_SLOT_COUNT])
            result = "failed" if tracker_id == 0 else "ignored" if tracker_id == 1 else "ok"
            self.on_command_response(command, result, states)
        else:
            LOGGER.warning("Unknown VUT message type %d", message)

    def send_command(self, command: str) -> bool:
        """Send a tracker_server command (``Echo``, ``Setup``, ``Pairing 0``, ...)."""

        if self._socket is None:
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
    ) -> None:
        if poll_hz <= 0.0:
            raise ValueError("poll_hz must be positive")
        self.socket_path = str(socket_path)
        self._interval = 1.0 / float(poll_hz)
        self._reconnect_interval = float(reconnect_interval)
        self.listener = listener
        self._client = VutIpcClient(self.socket_path)
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
                LOGGER.info("Connected to VIVE Ultimate tracker server %s", self.socket_path)
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
    def update_count(self) -> int:
        with self._lock:
            return self._update_count

    def snapshot(self) -> Mapping[int, VutTrackerPose]:
        with self._lock:
            return dict(self._latest)

    def serials(self) -> Mapping[int, str]:
        with self._lock:
            return dict(self._serials)

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
    "STATE_READY",
    "STATE_SYNCING",
    "TRACKER_STATE_NAMES",
    "UPDATE_SIZE",
    "VUT_DEFAULT_FRAME",
    "VUT_DEFAULT_SOCKET",
    "VUT_FRAMES",
    "VUT_TRACKING_SYSTEM",
    "VutIpcClient",
    "VutPoseFeed",
    "VutTrackerPose",
    "decode_update",
    "default_tracker_id",
    "encode_header",
    "encode_update_payload",
    "select_tracker",
    "vut_pose_to_core_world",
]
