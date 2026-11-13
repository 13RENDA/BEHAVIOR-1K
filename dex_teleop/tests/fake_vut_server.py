"""In-process stand-ins for VIVE Hub for Linux's ``tracker_server`` socket.

``FakeVutServer`` speaks the slot protocol of VIVE Hub 1.0.1 (``SOCK_STREAM``);
``FakeVutSdkServer`` speaks the VUT SDK protocol of VIVE Hub 1.0.3+
(``SOCK_SEQPACKET``). Both stream one or more trackers at a fixed rate and let
tests mutate the state, flags, or pose while running.
"""

from __future__ import annotations

import selectors
import socket
import struct
import threading
import time

from dex_teleop.tracking.vive_ultimate import (
    DEV_OFF,
    DEV_READY,
    DEV_SYNCING,
    MSG_ADD,
    MSG_REMOVE,
    MSG_START,
    MSG_STOP,
    MSG_UPDATE,
    SDK_HEADER_FORMAT,
    SDK_HEADER_SIZE,
    SDK_MSG_EVT_POSE,
    SDK_MSG_EVT_STATUS,
    SDK_MSG_GET_STATS,
    SDK_MSG_HELLO,
    SDK_MSG_HELLO_ACK,
    SDK_MSG_LIST_DEVICES,
    SDK_MSG_LIST_DEVICES_RESP,
    SDK_MSG_STATS_RESP,
    SDK_MSG_SUBSCRIBE,
    STATE_READY,
    encode_device_list,
    encode_header,
    encode_hello_ack,
    encode_pose_event,
    encode_sdk_header,
    encode_status_event,
    encode_update_payload,
)


class FakeVutServer:
    """Streams one or more trackers over a Unix socket at a fixed rate.

    Mutate ``state``, ``flags``, ``position`` or ``quaternion_xyzw`` while
    running to simulate tracking loss or motion; the next packets carry them.
    """

    def __init__(
        self,
        socket_path: str,
        *,
        serial: str = "59-A33R03250",
        slot: int = 0,
        rate_hz: float = 60.0,
        position=(0.25, 1.1, -0.4),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        state: int = STATE_READY,
        flags: int = 0,
        battery: int = 87,
        extra_trackers: dict[int, str] | None = None,
    ) -> None:
        self.socket_path = socket_path
        self.serial = serial
        self.slot = slot
        self.rate_hz = rate_hz
        self.position = tuple(position)
        self.quaternion_xyzw = tuple(quaternion_xyzw)
        self.state = state
        self.flags = flags
        self.battery = battery
        self.extra_trackers = dict(extra_trackers or {})
        self.sent_updates = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._server: socket.socket | None = None
        self._clients: list[socket.socket] = []
        self._thread: threading.Thread | None = None
        self._timestamp_us = 1_000_000

    def start(self) -> "FakeVutServer":
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.socket_path)
        server.listen(4)
        server.settimeout(0.05)
        self._server = server
        self._thread = threading.Thread(target=self._run, name="fake-vut-server", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            clients, self._clients = self._clients, []
        for client in clients:
            try:
                client.close()
            except OSError:
                pass
        if self._server is not None:
            self._server.close()

    def _welcome(self, client: socket.socket) -> None:
        client.sendall(encode_header(0, MSG_START, 0))
        serial = self.serial.encode("utf-8")
        client.sendall(encode_header(self.slot, MSG_ADD, len(serial)) + serial)
        for slot, extra_serial in self.extra_trackers.items():
            encoded = extra_serial.encode("utf-8")
            client.sendall(encode_header(slot, MSG_ADD, len(encoded)) + encoded)

    def _update_packet(self, slot: int, position, quaternion_xyzw, state: int, flags: int) -> bytes:
        payload = encode_update_payload(
            position, quaternion_xyzw, timestamp_us=self._timestamp_us, flags=flags
        )
        params = bytes((state & 0xFF, 0, 0, self.battery & 0xFF))
        return encode_header(slot, MSG_UPDATE, len(payload), params) + payload

    def remove_tracker(self) -> None:
        packet = encode_header(self.slot, MSG_REMOVE, 0)
        with self._lock:
            for client in self._clients:
                try:
                    client.sendall(packet)
                except OSError:
                    pass

    def _run(self) -> None:
        assert self._server is not None
        period = 1.0 / self.rate_hz
        next_send = time.monotonic()
        while not self._stop.is_set():
            try:
                client, _ = self._server.accept()
            except socket.timeout:
                client = None
            except OSError:
                break
            if client is not None:
                try:
                    self._welcome(client)
                except OSError:
                    client.close()
                else:
                    with self._lock:
                        self._clients.append(client)
            now = time.monotonic()
            if now >= next_send:
                next_send = now + period
                self._timestamp_us += int(period * 1e6)
                with self._lock:
                    packets = [
                        self._update_packet(self.slot, self.position, self.quaternion_xyzw, self.state, self.flags)
                    ]
                    for slot in self.extra_trackers:
                        packets.append(
                            self._update_packet(slot, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), STATE_READY, 0)
                        )
                    dead = []
                    for connected in self._clients:
                        try:
                            for packet in packets:
                                connected.sendall(packet)
                        except OSError:
                            dead.append(connected)
                    for connected in dead:
                        self._clients.remove(connected)
                        connected.close()
                    self.sent_updates += 1
            time.sleep(min(period / 4.0, 0.005))
        with self._lock:
            for connected in self._clients:
                try:
                    connected.sendall(encode_header(0, MSG_STOP, 0))
                except OSError:
                    pass


class FakeVutSdkServer:
    """VIVE Hub 1.0.3+ ``tracker_server``: the VUT SDK protocol on a SEQPACKET socket.

    ``state`` is the raw ``DeviceState`` byte the real daemon reports (pose
    status in bits 0-2 plus the tracker's flag bits); the default ``0x12`` is a
    tracking tracker with its pose-stable bit set, exactly as observed on
    hardware. Poses stream only while the status is READY..SYNCING; changing
    ``state`` broadcasts an ``EVT_STATUS`` to every client.
    """

    def __init__(
        self,
        socket_path: str,
        *,
        serial: str = "FA5AG3B00121",
        slot: int = 2,
        rate_hz: float = 120.0,
        position=(0.25, 1.1, -0.4),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        state: int = DEV_READY | 0x10,
        extra_trackers: dict[int, str] | None = None,
        placeholder_name: bool = False,
        streaming: bool = True,
    ) -> None:
        self.socket_path = socket_path
        self.serial = serial
        self.slot = slot
        self.rate_hz = rate_hz
        self.position = tuple(position)
        self.quaternion_xyzw = tuple(quaternion_xyzw)
        self._state = int(state)
        self.extra_trackers = dict(extra_trackers or {})
        #: Report the pre-serial "VIVE Ultimate Tracker N" device name and empty pose serials.
        self.placeholder_name = placeholder_name
        #: False mimics an Ultimate in standby: still listed in its last state, but no pose events.
        self.streaming = streaming
        self.sent_updates = 0
        self.hello_count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._server: socket.socket | None = None
        self._clients: list[socket.socket] = []
        # Like the real daemon, events flow only to sessions that completed HELLO.
        self._sessions: set[socket.socket] = set()
        self._thread: threading.Thread | None = None
        self._session_id = 0

    @property
    def device_id(self) -> int:
        return self.slot + 1

    @property
    def state(self) -> int:
        return self._state

    @state.setter
    def state(self, value: int) -> None:
        with self._lock:
            if int(value) == self._state:
                return
            self._state = int(value)
            self._broadcast_locked(
                SDK_MSG_EVT_STATUS,
                encode_status_event(self.device_id, self._state, timestamp_ns=time.monotonic_ns()),
            )

    def remove_tracker(self) -> None:
        self.state = DEV_OFF

    def start(self) -> "FakeVutSdkServer":
        server = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        server.bind(self.socket_path)
        server.listen(4)
        server.setblocking(False)
        self._server = server
        self._thread = threading.Thread(target=self._run, name="fake-vut-sdk-server", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            clients, self._clients = self._clients, []
        for client in clients:
            try:
                client.close()
            except OSError:
                pass
        if self._server is not None:
            self._server.close()

    def _device_name(self, slot: int, serial: str) -> str:
        return f"VIVE Ultimate Tracker {slot + 1}" if self.placeholder_name else serial

    def _device_list(self) -> bytes:
        devices = [(self.device_id, self._state, self._device_name(self.slot, self.serial))]
        devices.extend(
            (slot + 1, DEV_READY, self._device_name(slot, serial)) for slot, serial in self.extra_trackers.items()
        )
        return encode_device_list(devices)

    def _send(self, client: socket.socket, message: int, payload: bytes = b"", *, request_id: int = 0) -> None:
        client.send(encode_sdk_header(message, len(payload), request_id=request_id) + payload)

    def _broadcast_locked(self, message: int, payload: bytes) -> None:
        dead = []
        for client in self._clients:
            if client not in self._sessions:
                continue
            try:
                self._send(client, message, payload)
            except OSError:
                dead.append(client)
        for client in dead:
            self._clients.remove(client)
            self._sessions.discard(client)
            client.close()

    def _handle_request(self, client: socket.socket, datagram: bytes) -> None:
        if len(datagram) < SDK_HEADER_SIZE:
            return
        message, _version, _flags, request_id, _payload_len = struct.unpack(SDK_HEADER_FORMAT, datagram[:SDK_HEADER_SIZE])
        if message == SDK_MSG_HELLO:
            self.hello_count += 1
            self._session_id += 1
            self._send(client, SDK_MSG_HELLO_ACK, encode_hello_ack(self._session_id, time.monotonic_ns()), request_id=request_id)
            self._sessions.add(client)
        elif message == SDK_MSG_LIST_DEVICES:
            self._send(client, SDK_MSG_LIST_DEVICES_RESP, self._device_list(), request_id=request_id)
        elif message == SDK_MSG_GET_STATS:
            self._send(client, SDK_MSG_STATS_RESP, bytes(48), request_id=request_id)
        elif message == SDK_MSG_SUBSCRIBE:
            pass

    def _pose_payloads(self) -> list[bytes]:
        now = time.monotonic_ns()
        payloads = []
        if self.streaming and (self._state & 0x07) in range(DEV_READY, DEV_SYNCING + 1) and self._state != DEV_OFF:
            payloads.append(
                encode_pose_event(
                    self.device_id,
                    self.position,
                    self.quaternion_xyzw,
                    timestamp_ns=now,
                    serial="" if self.placeholder_name else self.serial,
                )
            )
        for slot, serial in self.extra_trackers.items():
            payloads.append(
                encode_pose_event(
                    slot + 1,
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                    timestamp_ns=now,
                    serial="" if self.placeholder_name else serial,
                )
            )
        return payloads

    def _run(self) -> None:
        assert self._server is not None
        selector = selectors.DefaultSelector()
        selector.register(self._server, selectors.EVENT_READ)
        period = 1.0 / self.rate_hz
        next_send = time.monotonic()
        while not self._stop.is_set():
            for key, _ in selector.select(timeout=min(period / 4.0, 0.005)):
                if key.fileobj is self._server:
                    try:
                        client, _ = self._server.accept()
                    except OSError:
                        continue
                    client.setblocking(False)
                    selector.register(client, selectors.EVENT_READ)
                    with self._lock:
                        self._clients.append(client)
                    continue
                client = key.fileobj
                try:
                    datagram = client.recv(SDK_HEADER_SIZE + 64 * 1024)
                except BlockingIOError:
                    continue
                except OSError:
                    datagram = b""
                if not datagram:
                    selector.unregister(client)
                    with self._lock:
                        if client in self._clients:
                            self._clients.remove(client)
                        self._sessions.discard(client)
                    client.close()
                    continue
                with self._lock:
                    try:
                        self._handle_request(client, datagram)
                    except OSError:
                        pass
            now = time.monotonic()
            if now >= next_send:
                next_send = now + period
                with self._lock:
                    payloads = self._pose_payloads()
                    for payload in payloads:
                        self._broadcast_locked(SDK_MSG_EVT_POSE, payload)
                    if payloads:
                        self.sent_updates += 1
        selector.close()
