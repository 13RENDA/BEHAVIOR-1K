"""Run the multi-source tracking worker in a child process.

Source parsing, fusion and retargeting are CPU-bound Python. In the simulator's
process they compete for the GIL with the many small PyTorch calls OmniGibson
makes every step, which tripled the hand bench's step time. In a child process
they run in parallel; the simulator process only unpickles the snapshots it
applies and the native samples it records.

The child forwards, in order, every row appended to its recording session and
then each snapshot that references those rows, so the parent's
``HandTrackingRecordingSession`` sees exactly the rows a same-process worker
would have appended, before the control loop can select them.
"""

from __future__ import annotations

from collections import deque
import itertools
import logging
import math
import multiprocessing
import signal
import threading
import time
import traceback
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable

from dex_teleop.runtime.worker import (
    MultiSourceTrackingWorker,
    RetargetingSnapshot,
    _SelectedRoleChecks,
)
from dex_teleop.tracking import SourceUnavailableError

if TYPE_CHECKING:
    from dex_teleop.omnigibson.hand_tracking_recording import (
        HandTrackingRecordingSession,
    )

_STATUS_PERIOD_SECONDS = 0.01
# Added to worker-side timeouts so the child's own timeout error wins the race.
_REPLY_MARGIN_SECONDS = 5.0
_QUERY_TIMEOUT_SECONDS = 30.0
_CLOSE_TIMEOUT_SECONDS = 10.0
_REMOTE_ERROR_TYPES = {
    error_type.__name__: error_type
    for error_type in (
        SourceUnavailableError,
        RuntimeError,
        TimeoutError,
        ValueError,
        TypeError,
    )
}


def _error_payload(error: BaseException) -> tuple[str, str, str]:
    return type(error).__name__, str(error), "".join(traceback.format_exception(error))


def _remote_error(payload: tuple[str, str, str]) -> BaseException:
    name, message, remote_traceback = payload
    error = _REMOTE_ERROR_TYPES.get(name, RuntimeError)(message)
    error.add_note(f"Raised in the hand-tracking process:\n{remote_traceback.rstrip()}")
    return error


# --------------------------------------------------------------------------
# Child process


class _Outbox:
    """Ordered messages to the parent, sent by one thread so producers never block."""

    def __init__(self, connection) -> None:
        self._connection = connection
        self._items: deque = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="dex-teleop-tracking-sender", daemon=True
        )
        self._thread.start()

    def put(self, item: tuple) -> None:
        with self._condition:
            self._items.append(item)
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._thread.join(timeout=_CLOSE_TIMEOUT_SECONDS)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._items and not self._closed:
                    self._condition.wait()
                batch = list(self._items)
                self._items.clear()
                if not batch:
                    return
            try:
                self._connection.send(batch)
            except (BrokenPipeError, EOFError, OSError):
                return


class _ForwardingSession:
    """Child-side session that numbers rows like the parent's and forwards each append.

    The parent's real session validates every forwarded append and must return
    the same row, so this side keeps only identity keys, not samples.
    """

    closed = False

    def __init__(self, session_id: int, outbox: _Outbox) -> None:
        from dex_teleop.omnigibson.hand_tracking_recording import (
            _retargeted_key,
            _sample_key,
        )

        self._session_id = session_id
        self._outbox = outbox
        self._sample_key = _sample_key
        self._retargeted_key = _retargeted_key
        self._rows: dict[tuple[str, str], dict[tuple, int]] = {}

    def _append(self, kind: str, stream_id: str, key: tuple, sample: Any) -> int:
        rows = self._rows.setdefault((kind, stream_id), {})
        row = rows.setdefault(key, len(rows))
        self._outbox.put(("row", self._session_id, kind, stream_id, sample, row))
        return row

    def append_articulation(self, stream_id: str, sample) -> int:
        return self._append("articulation", stream_id, self._sample_key(sample), sample)

    def append_wrist(self, stream_id: str, sample) -> int:
        return self._append("wrist", stream_id, self._sample_key(sample), sample)

    def append_retargeted(self, stream_id: str, sample) -> int:
        return self._append("retargeted", stream_id, self._retargeted_key(sample), sample)


def _describe(worker: MultiSourceTrackingWorker) -> dict[str, Any]:
    return {
        "articulation_source": worker.articulation_source,
        "wrist_source": worker.wrist_source,
        "record_articulation_sources": worker.record_articulation_sources,
        "record_wrist_sources": worker.record_wrist_sources,
        "articulation_streams": dict(worker.articulation_streams),
        "wrist_streams": dict(worker.wrist_streams),
        "control_wrist_stream": worker.control_wrist_stream,
        "retargeting_stream": worker.retargeting_stream,
        "retargeter_name": worker.retargeter_name,
        "handedness": worker.handedness,
        "retarget_once_per_snapshot": worker.retarget_once_per_snapshot,
        "haptic_sources": tuple(worker.haptic_sources),
    }


def _status(worker: MultiSourceTrackingWorker) -> dict[str, Any]:
    status = worker._tracking_status()
    try:
        worker.check_health()
    except BaseException as error:
        status["health"] = _error_payload(error)
    else:
        status["health"] = None
    return status


def _apply(worker: MultiSourceTrackingWorker, operation: str, value: Any, outbox: _Outbox) -> Any:
    if operation == "reset":
        worker.reset(timeout=value)
        return None
    if operation == "attach":
        session_id, timeout = value
        worker.set_recording_session(_ForwardingSession(session_id, outbox), timeout=timeout)
        return None
    if operation == "detach":
        worker.set_recording_session(None, timeout=value)
        return None
    if operation == "query":
        function, args = value
        return function(worker, *args)
    if operation == "first_timeout_message":
        return worker._first_timeout_message(value)
    raise ValueError(f"Unknown hand-tracking process operation {operation!r}")


def _serve_controls(worker: MultiSourceTrackingWorker, controls, outbox: _Outbox) -> None:
    next_status = 0.0
    while True:
        if controls.poll(max(0.0, next_status - time.monotonic())):
            try:
                message = controls.recv()
            except EOFError:
                return
            if message[0] == "consumed":
                try:
                    worker.snapshot()
                except Exception:
                    pass  # health is reported through the status stream
                continue
            if message[0] == "haptics":
                _, handedness, amplitudes = message
                try:
                    worker.apply_haptic_feedback(handedness, amplitudes)
                except Exception:
                    pass  # a dead sidecar surfaces through the status stream
                continue
            _, request_id, operation, value = message
            if operation == "close":
                return
            try:
                outbox.put(("reply", request_id, True, _apply(worker, operation, value, outbox)))
            except BaseException as error:
                outbox.put(("reply", request_id, False, _error_payload(error)))
        if time.monotonic() >= next_status:
            outbox.put(("status", _status(worker)))
            next_status = time.monotonic() + _STATUS_PERIOD_SECONDS


def _ignore_interrupt(_signum, _frame) -> None:
    """Leave Ctrl+C to the parent, which stops this process after saving its recording.

    A Python handler, unlike SIG_IGN, is reset by exec, so helper processes such
    as the MANUS bridge keep their default Ctrl+C behavior.
    """


def _serve(factory: Callable[[], MultiSourceTrackingWorker], controls, events, log_level: int) -> None:
    signal.signal(signal.SIGINT, _ignore_interrupt)
    logging.basicConfig(level=log_level)
    outbox = _Outbox(events)
    try:
        worker = factory()
        if not isinstance(worker, MultiSourceTrackingWorker):
            raise TypeError(
                f"Hand-tracking process factory returned {type(worker).__name__}, "
                "not MultiSourceTrackingWorker"
            )
        outbox.put(("describe", _describe(worker)))
        sequence = itertools.count(1)
        # Stream receipts travel with each snapshot so freshness checks never
        # see a snapshot newer than the receipts it was fused from.
        worker.publish_listener = lambda snapshot: outbox.put(
            ("snapshot", next(sequence), snapshot, worker._tracking_status())
        )
        worker.start()
    except BaseException as error:
        outbox.put(("error", _error_payload(error)))
        outbox.close()
        return
    outbox.put(("started",))
    close_error = None
    try:
        _serve_controls(worker, controls, outbox)
    finally:
        try:
            worker.close()
        except BaseException as error:
            close_error = _error_payload(error)
        outbox.put(("closed", close_error))
        outbox.close()


# --------------------------------------------------------------------------
# Parent process


class ProcessTrackingWorker(_SelectedRoleChecks):
    """The ``MultiSourceTrackingWorker`` API, backed by a worker in a child process.

    ``factory`` builds the unstarted worker inside the child, so it must be
    picklable: for example ``functools.partial`` of a module-level function
    with plain arguments. Configuration attributes such as
    ``retargeting_stream`` are available once :meth:`start` returns.
    """

    def __init__(self, factory: Callable[[], MultiSourceTrackingWorker]) -> None:
        self._factory = factory
        self._process = None
        self._controls = None
        self._events = None
        self._receiver: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._state = threading.Condition()
        self._start_attempted = False
        self._started = False
        self._start_error: tuple[str, str, str] | None = None
        self._fatal_error: BaseException | None = None
        self._exited = False
        self._closing = False
        self._closed = False
        self._close_reported = False
        self._close_error: tuple[str, str, str] | None = None
        self._latest: tuple[int, RetargetingSnapshot] | None = None
        self._consumed_sequence: int | None = None
        self._status: dict[str, Any] = {"receipts": {}, "first": False, "health": None}
        self._replies: dict[int, tuple[bool, Any]] = {}
        self._boundary_requests: set[int] = set()
        self._request_ids = itertools.count(1)
        self._session_ids = itertools.count(1)
        self._sessions: dict[int, HandTrackingRecordingSession] = {}
        self._recording_session: HandTrackingRecordingSession | None = None
        self._recording_session_id: int | None = None
        self.haptic_sources: tuple[str, ...] = ()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._start_attempted:
            raise RuntimeError("ProcessTrackingWorker has already been started")
        self._start_attempted = True
        context = multiprocessing.get_context("spawn")
        child_controls, self._controls = context.Pipe(duplex=False)
        self._events, child_events = context.Pipe(duplex=False)
        self._process = context.Process(
            target=_serve,
            args=(self._factory, child_controls, child_events, logging.getLogger().getEffectiveLevel()),
            name="dex-teleop-tracking",
            daemon=True,
        )
        self._process.start()
        child_controls.close()
        child_events.close()
        self._receiver = threading.Thread(
            target=self._receive, name="dex-teleop-tracking-receiver", daemon=True
        )
        self._receiver.start()
        with self._state:
            while not self._started and self._start_error is None and not self._exited:
                self._state.wait()
            start_error = self._start_error
            failure = None if self._started else self._fatal_error
        if start_error is not None or failure is not None:
            self._closed = True
            self._stop_process()
            raise _remote_error(start_error) if start_error is not None else failure
        pending_session, self._recording_session = self._recording_session, None
        if pending_session is not None:
            self.set_recording_session(pending_session)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process is None:
            return
        with self._state:
            self._closing = True
        try:
            with self._send_lock:
                self._controls.send(("control", next(self._request_ids), "close", None))
        except (BrokenPipeError, EOFError, OSError):
            pass
        deadline = time.monotonic() + _CLOSE_TIMEOUT_SECONDS
        with self._state:
            while not self._close_reported and not self._exited:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._state.wait(remaining)
            close_error = self._close_error
        self._stop_process()
        if close_error is not None:
            raise _remote_error(close_error)

    def _stop_process(self) -> None:
        process = self._process
        process.join(timeout=_CLOSE_TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join()
        if self._receiver is not None:
            self._receiver.join(timeout=2.0)
        self._controls.close()
        self._events.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # -- control-loop API ----------------------------------------------------

    def snapshot(self, maximum_age: float | None = None) -> RetargetingSnapshot | None:
        if maximum_age is not None and maximum_age < 0:
            raise ValueError("maximum_age must be non-negative")
        self.check_health()
        with self._state:
            entry = self._latest
        if entry is None:
            return None
        sequence, latest = entry
        if self.retarget_once_per_snapshot and sequence != self._consumed_sequence:
            self._consumed_sequence = sequence
            self._send(("consumed", sequence))
        if maximum_age is not None:
            self._raise_if_snapshot_stale(latest, maximum_age)
        return latest

    def wait_for_first(self, timeout: float) -> RetargetingSnapshot:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = time.monotonic() + timeout
        with self._state:
            while not self._ready() and self._fatal_error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._state.wait(remaining)
            ready = self._ready()
        if not ready:
            self.check_health()
            raise SourceUnavailableError(
                self._request("first_timeout_message", timeout, timeout=_QUERY_TIMEOUT_SECONDS)
            )
        snapshot = self.snapshot()
        if snapshot is None:
            self.check_health()
            raise SourceUnavailableError("Worker signaled without publishing a fused observation")
        return snapshot

    def _ready(self) -> bool:
        return self._latest is not None and bool(self._status.get("first"))

    def check_health(self) -> None:
        with self._state:
            fatal = self._fatal_error
            health = self._status.get("health")
        if fatal is not None:
            raise RuntimeError(str(fatal)) from fatal
        if health is not None:
            raise _remote_error(health)

    def check_stream_freshness(self, maximum_age: float) -> None:
        """Require a recent sample from every selected and record-only role."""

        if not math.isfinite(maximum_age) or maximum_age < 0.0:
            raise ValueError("maximum_age must be finite and non-negative")
        self.check_health()
        with self._state:
            receipts = dict(self._status.get("receipts", {}))
        self._raise_if_streams_stale(receipts, maximum_age)

    @property
    def recording_session(self) -> HandTrackingRecordingSession | None:
        return self._recording_session

    def set_recording_session(
        self,
        session: HandTrackingRecordingSession | None,
        *,
        timeout: float = 2.0,
    ) -> None:
        """Attach or detach a per-task collector at a worker-thread boundary."""

        MultiSourceTrackingWorker._validate_recording_session(session)
        if not self._start_attempted:
            self._recording_session = session
            return
        if session is self._recording_session:
            return
        previous_id = self._recording_session_id
        if session is None:
            self._request("detach", timeout, timeout=timeout + _REPLY_MARGIN_SECONDS, boundary=True)
            session_id = None
        else:
            session_id = next(self._session_ids)
            with self._state:
                self._sessions[session_id] = session
            try:
                self._request(
                    "attach",
                    (session_id, timeout),
                    timeout=timeout + _REPLY_MARGIN_SECONDS,
                    boundary=True,
                )
            except BaseException:
                with self._state:
                    self._sessions.pop(session_id, None)
                raise
        with self._state:
            self._sessions.pop(previous_id, None)
        self._recording_session = session
        self._recording_session_id = session_id
        self._reopen_retargeting()

    def reset(self, *, timeout: float = 2.0) -> None:
        """Synchronously reset fusion and retargeter state in the tracking process."""

        self._request("reset", timeout, timeout=timeout + _REPLY_MARGIN_SECONDS, boundary=True)
        self._reopen_retargeting()

    def query(self, function: Callable[..., Any], *args: Any) -> Any:
        """Return ``function(worker, *args)`` evaluated in the tracking process."""

        return self._request("query", (function, args), timeout=_QUERY_TIMEOUT_SECONDS)

    def apply_haptic_feedback(self, handedness, amplitudes) -> bool:
        """Forward finger amplitudes to the tracking process without waiting for a reply."""

        if not self.haptic_sources:
            return False
        handedness = getattr(handedness, "value", handedness)
        self._send(("haptics", handedness, [float(value) for value in amplitudes]))
        return True

    # -- plumbing ------------------------------------------------------------

    def _reopen_retargeting(self) -> None:
        # A boundary drops any snapshot published before its reply; tell the
        # child that nothing is pending so a once-per-snapshot worker resumes.
        if self.retarget_once_per_snapshot:
            self._send(("consumed", None))

    def _send(self, message: tuple) -> None:
        try:
            with self._send_lock:
                self._controls.send(message)
        except (BrokenPipeError, EOFError, OSError) as error:
            with self._state:
                deadline = time.monotonic() + 2.0
                while not self._exited and time.monotonic() < deadline:
                    self._state.wait(deadline - time.monotonic())
            self.check_health()
            raise RuntimeError("Hand-tracking process stopped accepting commands") from error

    def _request(self, operation: str, value: Any, *, timeout: float, boundary: bool = False) -> Any:
        if self._process is None or self._closed:
            self.check_health()
            raise RuntimeError("ProcessTrackingWorker is not running")
        request_id = next(self._request_ids)
        if boundary:
            with self._state:
                self._boundary_requests.add(request_id)
        self._send(("control", request_id, operation, value))
        deadline = time.monotonic() + timeout
        with self._state:
            while request_id not in self._replies:
                if self._exited:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for the hand-tracking process to apply {operation!r}"
                    )
                self._state.wait(remaining)
            reply = self._replies.pop(request_id, None)
        if reply is None:
            self.check_health()
            raise RuntimeError("Hand-tracking process exited before replying")
        succeeded, result = reply
        if not succeeded:
            raise _remote_error(result)
        return result

    def _receive(self) -> None:
        failure = None
        try:
            while True:
                for item in self._events.recv():
                    self._handle(item)
        except (EOFError, OSError):
            pass
        except BaseException as error:
            failure = error
        self._process.join(timeout=2.0)
        with self._state:
            if failure is not None and self._fatal_error is None:
                self._fatal_error = RuntimeError(f"Hand-tracking receiver failed: {failure}")
            if not self._closing and self._fatal_error is None:
                self._fatal_error = RuntimeError(
                    "Hand-tracking process exited unexpectedly "
                    f"(exit code {self._process.exitcode})"
                )
            self._exited = True
            self._state.notify_all()

    def _handle(self, item: tuple) -> None:
        kind = item[0]
        if kind == "row":
            self._import_row(*item[1:])
        elif kind == "snapshot":
            _, sequence, snapshot, status = item
            with self._state:
                self._latest = (sequence, snapshot)
                self._status = {**self._status, **status}
                self._state.notify_all()
        elif kind == "status":
            with self._state:
                self._status = item[1]
                self._state.notify_all()
        elif kind == "reply":
            _, request_id, succeeded, result = item
            with self._state:
                if request_id in self._boundary_requests:
                    self._boundary_requests.discard(request_id)
                    if succeeded:
                        # Everything published before this point predates the boundary.
                        self._latest = None
                        self._status = {"receipts": {}, "first": False, "health": self._status.get("health")}
                self._replies[request_id] = (succeeded, result)
                self._state.notify_all()
        elif kind == "describe":
            self._apply_description(item[1])
        elif kind == "started":
            with self._state:
                self._started = True
                self._state.notify_all()
        elif kind == "error":
            with self._state:
                self._start_error = item[1]
                self._state.notify_all()
        elif kind == "closed":
            with self._state:
                self._close_reported = True
                self._close_error = item[1]
                self._state.notify_all()
        else:
            raise ValueError(f"Unknown hand-tracking process message {kind!r}")

    def _import_row(self, session_id: int, kind: str, stream_id: str, sample: Any, row: int) -> None:
        with self._state:
            session = self._sessions.get(session_id)
        if session is None:
            return
        try:
            recorded = getattr(session, f"append_{kind}")(stream_id, sample)
            if recorded != row:
                raise RuntimeError(
                    f"stream {stream_id!r} recorded row {recorded} in this process but row {row} "
                    "in the hand-tracking process"
                )
        except BaseException as error:
            with self._state:
                if self._fatal_error is None:
                    self._fatal_error = RuntimeError(
                        f"Multi-source hand retargeting worker failed: {error}"
                    )
                    self._fatal_error.__cause__ = error
                self._state.notify_all()

    def _apply_description(self, description: dict[str, Any]) -> None:
        self.articulation_source = description["articulation_source"]
        self.wrist_source = description["wrist_source"]
        self.record_articulation_sources = tuple(description["record_articulation_sources"])
        self.record_wrist_sources = tuple(description["record_wrist_sources"])
        self._articulation_keys = (self.articulation_source, *self.record_articulation_sources)
        self._wrist_keys = (self.wrist_source, *self.record_wrist_sources)
        self.articulation_streams = MappingProxyType(description["articulation_streams"])
        self.wrist_streams = MappingProxyType(description["wrist_streams"])
        self.control_wrist_stream = description["control_wrist_stream"]
        self.retargeting_stream = description["retargeting_stream"]
        self.retargeter_name = description["retargeter_name"]
        self.handedness = description["handedness"]
        self.retarget_once_per_snapshot = description["retarget_once_per_snapshot"]
        self.haptic_sources = tuple(description.get("haptic_sources", ()))
