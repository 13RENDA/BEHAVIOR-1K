"""In-process stand-in for VIVE Hub for Linux's ``tracker_server`` socket."""

from __future__ import annotations

import socket
import threading
import time

from dex_teleop.tracking.vive_ultimate import (
    MSG_ADD,
    MSG_REMOVE,
    MSG_START,
    MSG_STOP,
    MSG_UPDATE,
    STATE_READY,
    encode_header,
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
