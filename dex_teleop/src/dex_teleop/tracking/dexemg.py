"""Finger articulation decoded from the wrist EMG band by dexemg (``--hand-source dexemg``).

The decoding runs in a separate process, :mod:`dex_teleop.emg.dexemg_bridge`, started by the operator in
the ``emg2pose`` environment before the simulator (it owns the band, records the neutral pose and runs
the model).  This source is its TCP client: every ``frame`` message becomes a wrist-local MediaPipe-21
:class:`HandArticulationSample` in the OpenXR anatomical wrist frame, so any wrist source (``vive``,
``manus``, ``quest``) and both retargeters compose with it unchanged.
"""

from __future__ import annotations

from collections import deque
import json
import socket
import threading
import time
from typing import Any

import numpy as np

from dex_teleop.emg.dexemg_protocol import (
    COORDINATE_FRAME,
    DEFAULT_HOST,
    DEFAULT_PORT,
    LANDMARK_NAMES,
    LANDMARK_SCHEMA,
    PROTOCOL,
    SOURCE_NAME,
    decode,
    encode,
)
from dex_teleop.tracking.base import SourceUnavailableError
from dex_teleop.types import HandArticulationSample, Handedness

DEXEMG_SOURCE_NAME = SOURCE_NAME
BRIDGE_HINT = (
    "start the bridge first, in the emg2pose environment: "
    "python dex_teleop/scripts/dexemg_bridge.py --checkpoint <wrist checkpoint> --adapter <hciN>"
)


class DexemgArticulationSource:
    """Client of the dexemg bridge; a draining articulation source for the right hand."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        startup_timeout: float = 30.0,
        stale_timeout: float = 1.0,
        queue_capacity: int = 4096,
        handedness: Handedness | str = Handedness.RIGHT,
    ) -> None:
        if not np.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")
        if not np.isfinite(stale_timeout) or stale_timeout <= 0:
            raise ValueError("stale_timeout must be positive")
        self.host, self.port = str(host), int(port)
        self.startup_timeout = float(startup_timeout)
        self.stale_timeout = float(stale_timeout)
        self.handedness = Handedness(handedness)
        if self.handedness is not Handedness.RIGHT:
            raise ValueError("The dexemg decoder serves a right hand only")
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._queue: deque[HandArticulationSample] = deque()
        self._capacity = int(queue_capacity)
        self._latest: HandArticulationSample | None = None
        self._last_frame_monotonic: float | None = None
        self._hello: dict[str, Any] | None = None
        self._status: dict[str, Any] | None = None
        self._error: str | None = None
        self._overflow = False
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._closing = False
        self._last_control_ns = 0
        self.frames_received = 0

    # -- lifecycle -------------------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        deadline = time.monotonic() + min(5.0, self.startup_timeout)
        last_error: OSError | None = None
        while True:
            try:
                self._socket = socket.create_connection((self.host, self.port), timeout=1.0)
                break
            except OSError as error:
                last_error = error
                if time.monotonic() >= deadline:
                    raise SourceUnavailableError(
                        f"No dexemg bridge at {self.host}:{self.port} ({last_error}); {BRIDGE_HINT}"
                    ) from error
                time.sleep(0.2)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket.settimeout(0.5)
        self._thread = threading.Thread(target=self._read_loop, name="dexemg-client", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + self.startup_timeout
        with self._condition:
            while self._latest is None and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    state = (self._status or {}).get("state", "no status")
                    message = (self._status or {}).get("message", "")
                    raise SourceUnavailableError(
                        f"The dexemg bridge sent no hand within {self.startup_timeout:g} s (bridge state: {state}"
                        + (f", {message}" if message else "")
                        + "). Record the neutral in the bridge terminal (Enter) and relaunch."
                    )
                self._condition.wait(min(remaining, 0.25))
            if self._error is not None and self._latest is None:
                raise SourceUnavailableError(self._error)

    def close(self) -> None:
        self._closing = True
        sock, self._socket = self._socket, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_exc):
        self.close()

    # -- reading ---------------------------------------------------------------------------------------
    def read_articulation(self, handedness: Handedness) -> HandArticulationSample | None:
        if Handedness(handedness) is not self.handedness:
            return None
        with self._lock:
            return self._latest

    def drain_articulations(self, handedness: Handedness) -> tuple[HandArticulationSample, ...]:
        if Handedness(handedness) is not self.handedness:
            return ()
        with self._lock:
            samples = tuple(self._queue)
            self._queue.clear()
        return samples

    def check_health(self) -> None:
        with self._lock:
            error, overflow, last = self._error, self._overflow, self._last_frame_monotonic
            status = dict(self._status or {})
        if error is not None:
            raise SourceUnavailableError(error)
        if overflow:
            raise SourceUnavailableError("dexemg articulation queue overflowed; samples were lost")
        if last is not None and time.monotonic() - last > self.stale_timeout:
            raise SourceUnavailableError(
                f"No dexemg hand for {time.monotonic() - last:.2f} s (bridge state {status.get('state')!r}"
                f"{', ' + status['message'] if status.get('message') else ''})"
            )

    # -- extras ----------------------------------------------------------------------------------------
    def request_neutral_calibration(self, seconds: float | None = None) -> None:
        """Ask the bridge to re-record the neutral (3 s countdown); streaming continues meanwhile."""

        sock = self._socket
        if sock is None:
            raise SourceUnavailableError("dexemg source is not connected")
        message: dict[str, Any] = {"type": "cmd", "action": "calibrate_neutral"}
        if seconds is not None:
            message["seconds"] = float(seconds)
        sock.sendall(encode(message))

    @property
    def bridge_status(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._status is None else dict(self._status)

    def recording_metadata(self) -> dict[str, Any]:
        with self._lock:
            hello = dict(self._hello or {})
            status = dict(self._status or {})
        return {
            "provider": "dexemg bridge (wrist EMG band -> streaming decoder)",
            "endpoint": f"tcp://{self.host}:{self.port}",
            "protocol": hello.get("protocol"),
            "checkpoint": hello.get("checkpoint"),
            "model": hello.get("model"),
            "chunk_steps": hello.get("chunk_steps"),
            "overlap_steps": hello.get("overlap_steps"),
            "emit_hz": hello.get("emit_hz"),
            "smoothing": hello.get("smoothing"),
            "acquisition": hello.get("acquisition"),
            "channel_scale": status.get("channel_scale"),
            "scale_ratio_to_training": status.get("scale_ratio_to_training"),
        }

    # -- internals -------------------------------------------------------------------------------------
    def _fail(self, message: str) -> None:
        with self._condition:
            if self._error is None:
                self._error = message
            self._condition.notify_all()

    def _read_loop(self) -> None:
        buffer = b""
        sock = self._socket
        while not self._closing and sock is not None:
            try:
                chunk = sock.recv(1 << 16)
            except socket.timeout:
                continue
            except OSError as error:
                if not self._closing:
                    self._fail(f"dexemg bridge connection failed: {error}")
                return
            if not chunk:
                if not self._closing:
                    self._fail("dexemg bridge closed the connection")
                return
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    self._handle(decode(line), time.monotonic_ns())
                except (ValueError, KeyError, TypeError) as error:
                    self._fail(f"Invalid dexemg bridge message: {error}")
                    return

    def _handle(self, message: dict[str, Any], receipt_ns: int) -> None:
        kind = message["type"]
        if kind == "hello":
            problems = []
            if message.get("protocol") != PROTOCOL:
                problems.append(f"protocol {message.get('protocol')!r} != {PROTOCOL!r}")
            if tuple(message.get("landmark_names", ())) != tuple(LANDMARK_NAMES):
                problems.append("landmark order differs")
            if message.get("coordinate_frame") != COORDINATE_FRAME:
                problems.append(f"frame {message.get('coordinate_frame')!r}")
            if message.get("handedness") != self.handedness.value:
                problems.append(f"handedness {message.get('handedness')!r}")
            if problems:
                self._fail("Incompatible dexemg bridge: " + "; ".join(problems))
                return
            with self._lock:
                self._hello = message
        elif kind == "status":
            with self._lock:
                self._status = message
            if message.get("state") == "error":
                self._fail(f"dexemg bridge error: {message.get('message')}")
        elif kind == "error":
            self._fail(f"dexemg bridge error: {message.get('message')}")
        elif kind == "frame":
            if self._hello is None:
                raise ValueError("frame before hello")
            landmarks = np.asarray(message["landmarks"], dtype=np.float64)
            if landmarks.shape != (len(LANDMARK_NAMES), 3) or not np.isfinite(landmarks).all():
                raise ValueError(f"bad landmark array {landmarks.shape}")
            emg_sample_ns = int(message["timestamp_ns"])
            # Control time = when the decoded hand became available, not the EMG sample time: the decoder's
            # output leads the glove by ~60 ms (streaming study, 2026-11-13), so the newest decoded hand is
            # the operator's hand now; pairing it with the newest wrist keeps the arm from lagging by the EMG
            # pipeline, and a hand stamped before a fuser reset could otherwise never find a wrist in skew.
            # The EMG sample time stays in source_timestamp_ns for recordings. Strictly increasing per line.
            control_ns = max(receipt_ns, self._last_control_ns + 1000)
            self._last_control_ns = control_ns
            sample = HandArticulationSample(
                timestamp=control_ns / 1e9,
                receipt_timestamp=control_ns / 1e9,
                source_timestamp_ns=emg_sample_ns,
                source_frame_id=int(message["seq"]),
                handedness=self.handedness,
                joint_positions=dict(zip(LANDMARK_NAMES, landmarks)),
                joint_validity={name: True for name in LANDMARK_NAMES},
                source=DEXEMG_SOURCE_NAME,
                schema=LANDMARK_SCHEMA,
                coordinate_frame=COORDINATE_FRAME,
                provenance={"sample_index": int(message["sample_index"]),
                            "emg_sample_monotonic_ns": emg_sample_ns,
                            "emit_monotonic_ns": int(message["emit_ns"]),
                            # the decoder's 20 joint angles (emg2pose order, rad) behind these landmarks;
                            # provenance values must be scalars, hence one compact JSON string
                            "angles_rad_json": json.dumps([round(float(a), 6) for a in message.get("angles", ())],
                                                          separators=(",", ":"))},
            )
            with self._condition:
                if len(self._queue) >= self._capacity:
                    self._overflow = True
                else:
                    self._queue.append(sample)
                self._latest = sample
                self._last_frame_monotonic = receipt_ns / 1e9
                self.frames_received += 1
                self._condition.notify_all()


def probe_dexemg_bridge(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 3.0, *, require_decoder: bool = True
) -> str:
    """Pre-flight: connect, read hello + status, say what is wrong (raise ``SystemExit``) or what is right.

    ``require_decoder=False`` is the check for recording the raw EMG only (``--emg-bridge`` with another hand
    source): the band must be connected, a neutral is asked for but not required.
    """

    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except OSError as error:
        raise SystemExit(f"No dexemg bridge at {host}:{port} ({error}); {BRIDGE_HINT}") from error
    hello = status = None
    frames = 0
    buffer = b""
    deadline = time.monotonic() + timeout
    try:
        sock.settimeout(0.25)
        while time.monotonic() < deadline and not (hello and status and (frames or status.get("state") != "streaming")):
            try:
                chunk = sock.recv(1 << 16)
            except socket.timeout:
                continue
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                message = decode(line)
                if message["type"] == "hello":
                    hello = message
                elif message["type"] == "status":
                    status = message
                elif message["type"] == "frame":
                    frames += 1
    finally:
        sock.close()
    if hello is None or status is None:
        raise SystemExit(f"The process at {host}:{port} did not answer as a dexemg bridge")
    if hello.get("protocol") != PROTOCOL:
        raise SystemExit(f"dexemg bridge speaks {hello.get('protocol')!r}, expected {PROTOCOL!r}")
    state = status.get("state")
    if not require_decoder:
        if state in ("connecting", "error"):
            raise SystemExit(
                f"The dexemg bridge is {state!r}: {status.get('message') or 'the band is not connected'}. "
                "Wait until its window shows the band streaming, then relaunch."
            )
        recorded = status.get("neutral_recorded")
        neutral = (
            f"neutral {recorded} (saved with the recording)"
            if recorded and "training reference" not in str(recorded)
            else "NO OWN NEUTRAL YET -- record one in the bridge window (it is saved with the recording "
            "as soon as it is recorded)"
        )
        mode = " (record-only bridge)" if status.get("record_only") else ""
        return f"dexemg bridge OK for EMG recording{mode}: {status.get('rate_hz') or 0:.0f} Hz EMG, {neutral}"
    if status.get("record_only"):
        raise SystemExit(
            "The dexemg bridge runs with --record-only: it passes the raw EMG on but decodes no hand. "
            "For --hand-source dexemg restart it with --checkpoint (without --record-only)."
        )
    if state != "streaming" and not frames:
        raise SystemExit(
            f"The dexemg bridge is {state!r}: {status.get('message') or 'no hand yet'}. "
            "Record the neutral first (Enter in the bridge terminal), then relaunch."
        )
    warning = status.get("scale_warning")
    return (
        f"dexemg bridge OK: {hello.get('checkpoint')} streaming "
        f"{status.get('rate_hz') or 0:.0f} Hz EMG, neutral ratio to training {status.get('scale_ratio_to_training')}"
        + (f" -- WARNING {warning}" if warning else "")
    )


__all__ = ["BRIDGE_HINT", "DEXEMG_SOURCE_NAME", "DexemgArticulationSource", "probe_dexemg_bridge"]
