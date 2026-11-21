"""EMG recorder fed by the dexemg bridge (the launcher's ``--emg-bridge``).

``dex_teleop.emg.sidecar`` connects to the band itself. While the dexemg bridge runs, the bridge owns the
band (one process per device), so this sidecar takes the raw samples from the bridge instead and writes the
same file with the same writer: the launcher merges it into the recording and aligns it with the actions
exactly as for ``--emg``. It speaks the same preview protocol to the launcher (``status`` / ``samples`` /
``ready`` / ``error`` over UDP), so ``EmgSession`` and the EMG window work unchanged.

Differences from the SDK sidecar, all recorded in the EMG group's attributes:

* ``samples/raw_adc`` is zero and ``samples/sdk_timestamp_ms`` is -1: the bridge keeps the SDK's microvolt
  values only (what HandSync records);
* saturation is NaN; impedance (``batches/impedance_ohm``) is the latest impedance packet the bridge received
  (NaN until the first one; ``impedance_available`` says whether any arrived);
* ``samples/sample_index`` is made continuous across a band reconnect (the band's counter restarts): the gap
  is filled from the host clock and counted in ``sample_index_resets``;
* ``samples/estimated_monotonic_ns`` uses the band's measured sample period, not the nominal one (the band
  delivers about 501.4 samples per host second, so the nominal period drifts 2.7 ms per second);
* every neutral recording of the bridge session is stored under ``emg/neutral_<i>`` with its channel scale,
  so the recording carries the normalization it was decoded with.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
import traceback

import numpy as np

from dex_teleop.emg.dexemg_protocol import DEFAULT_HOST, DEFAULT_PORT, PROTOCOL, decode, encode
from dex_teleop.emg.sidecar import PREVIEW_SAMPLE_LIMIT, EmgHdf5Writer, _UdpPublisher
from dex_teleop.emg.types import EmgBatch

RESET_GAP_SAMPLES = 2_000_000  # a jump this large (or backwards) is a counter restart, not lost packets


class ContinuousIndex:
    """The band's sample counter made strictly increasing across restarts, using the host clock for the gap."""

    def __init__(self, sample_rate_hz: float) -> None:
        self._period_ns = 1.0e9 / float(sample_rate_hz)
        self._offset = 0
        self._last_index: int | None = None
        self._last_time_ns: int | None = None
        self.resets = 0

    def __call__(self, device_index: np.ndarray, t_ns: np.ndarray) -> np.ndarray:
        device_index = np.asarray(device_index, dtype=np.int64)
        index = device_index + self._offset
        if self._last_index is not None and not (0 < int(index[0]) - self._last_index < RESET_GAP_SAMPLES):
            elapsed = max(1, int(round((int(t_ns[0]) - self._last_time_ns) / self._period_ns)))
            self._offset += self._last_index + elapsed - int(index[0])
            index = device_index + self._offset
            self.resets += 1
        self._last_index, self._last_time_ns = int(index[-1]), int(t_ns[-1])
        return index


def batch_from_message(message: dict, index: ContinuousIndex, impedance_ohm=None) -> EmgBatch:
    values = np.asarray(message["values"], dtype=np.float32)
    t_ns = np.asarray(message["t_ns"], dtype=np.int64)
    lost = message.get("is_lost")
    return EmgBatch(
        receive_monotonic_ns=int(message.get("received_ns", t_ns[-1])),
        sample_index=index(np.asarray(message["sample_index"], dtype=np.int64), t_ns),
        sdk_timestamp_ms=np.full(len(values), -1, dtype=np.int64),
        signal_uv=values,
        raw_adc=np.zeros(values.shape, dtype=np.int32),
        is_lost=np.zeros(len(values), dtype=np.bool_) if lost is None else np.asarray(lost, dtype=np.bool_),
        impedance_ohm=(np.full(values.shape[1], np.nan, dtype=np.float32) if impedance_ohm is None
                       else np.asarray(impedance_ohm, dtype=np.float32).reshape(values.shape[1])),
        saturation=np.full(values.shape[1], np.nan, dtype=np.float32),
    )


def write_neutral(writer: EmgHdf5Writer, message: dict) -> None:
    """Store one neutral of the bridge session next to the samples (replacing an earlier copy of it)."""

    group = writer._group  # same package: the writer owns the file, the neutral belongs to the EMG group
    name = f"neutral_{int(message['index'])}"
    if name in group:
        del group[name]
    neutral = group.create_group(name)
    samples = message.get("samples")
    if samples is not None:
        neutral.create_dataset("signal_uv", data=np.asarray(samples, dtype=np.float32), compression="gzip")
    if message.get("scale") is not None:
        neutral.create_dataset("channel_scale_uv", data=np.asarray(message["scale"], dtype=np.float32))
    for key in ("time", "origin", "rejected", "active", "trim_seconds", "sample_rate_hz"):
        if message.get(key) is not None:
            neutral.attrs[key] = message[key]
    if message.get("ratio_to_training") is not None:
        neutral.attrs["ratio_to_training_json"] = json.dumps(message["ratio_to_training"])
    neutral.attrs["scale_semantics"] = "per-channel standard deviation of signal_uv after trim_seconds (dexemg)"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preview-host", default="127.0.0.1")
    parser.add_argument("--preview-port", required=True, type=int)
    parser.add_argument("--bridge-host", default=DEFAULT_HOST)
    parser.add_argument("--bridge-port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    return parser


def run(args) -> int:
    publisher = _UdpPublisher(args.preview_host, args.preview_port)
    stopping = {"flag": False}

    def request_stop(_signum, _frame):
        stopping["flag"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    writer = None
    connection = None
    try:
        publisher.send("status", status=f"connecting to the dexemg bridge at {args.bridge_host}:{args.bridge_port}")
        try:
            connection = socket.create_connection((args.bridge_host, args.bridge_port), timeout=args.connect_timeout)
        except OSError as error:
            raise RuntimeError(
                f"No dexemg bridge at {args.bridge_host}:{args.bridge_port} ({error}). Start it first "
                "(dex_teleop/scripts/dexemg_gui.py or dexemg_bridge.py); the bridge keeps the band."
            ) from error
        connection.settimeout(0.2)
        connection.sendall(encode({"type": "cmd", "action": "subscribe_raw"}))
        connection.sendall(encode({"type": "cmd", "action": "get_neutral"}))
        hello = None
        status = {}
        pending_neutrals: list[dict] = []
        impedance = np.full(8, np.nan, dtype=np.float32)  # latest electrode impedance (ohm) from the bridge
        impedance_seen = False
        index = None
        first_batch = True
        buffer = b""
        last_data = time.monotonic()
        parent = os.getppid()
        while not stopping["flag"]:
            if os.getppid() != parent:
                break  # the launcher is gone without stopping us: close the file instead of recording forever
            try:
                chunk = connection.recv(1 << 16)
            except socket.timeout:
                if writer is not None and time.monotonic() - last_data > 5.0:
                    publisher.send("status", status="no EMG from the bridge for 5 s (band disconnected?)")
                    last_data = time.monotonic()
                continue
            if not chunk:
                raise RuntimeError("The dexemg bridge closed the connection")
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                message = decode(line)
                kind = message["type"]
                if kind == "hello":
                    hello = message
                    if message.get("protocol") != PROTOCOL:
                        raise RuntimeError(f"Unexpected bridge protocol {message.get('protocol')!r}, expected {PROTOCOL!r}")
                elif kind == "status":
                    status = message
                    if writer is None:
                        publisher.send(
                            "status",
                            status=f"dexemg bridge is {message.get('state')!r}; waiting for EMG samples "
                            f"({message.get('message') or 'band not streaming yet'})",
                        )
                elif kind == "neutral":
                    if writer is None:
                        pending_neutrals.append(message)
                    else:
                        write_neutral(writer, message)
                elif kind == "impedance":
                    impedance = np.asarray(message["values"], dtype=np.float32)
                    impedance_seen = True
                elif kind == "error":
                    raise RuntimeError(f"dexemg bridge error: {message.get('message')}")
                elif kind == "emg":
                    if hello is None:
                        raise RuntimeError("The bridge sent EMG before its hello message")
                    sample_rate_hz = float(hello.get("sample_rate_hz", 500))
                    if writer is None:
                        channel_count = len(message["values"][0])
                        band = status.get("band") or {}
                        metadata = {
                            "source": "dexemg bridge raw stream (dex_teleop.emg.bridge_sidecar)",
                            "bridge_protocol": PROTOCOL,
                            "bridge_started": str(hello.get("started")),
                            "bridge_checkpoint": str(hello.get("checkpoint")),
                            "device_name": str(band.get("device_name") or "unknown (dexemg bridge)"),
                            "device_address": str(band.get("device_address") or "unknown"),
                            "band_json": json.dumps(band, default=str),
                            "filter_configuration": json.dumps(hello.get("firmware_filters"), default=str),
                            "signal_units": "microvolts as scaled by the Synchroni SDK (sample.data), as HandSync records",
                            "raw_adc_available": False,
                            "sdk_timestamp_available": False,

                            "decoder_enabled": False,
                            "bridge_channel_scale_json": json.dumps(status.get("channel_scale")),
                            "bridge_neutral_recorded": str(status.get("neutral_recorded")),
                        }
                        writer = EmgHdf5Writer(
                            args.output, sample_rate_hz=sample_rate_hz, channel_count=channel_count, metadata=metadata,
                            fit_sample_period=True,
                        )
                        index = ContinuousIndex(sample_rate_hz)
                        for neutral in pending_neutrals:
                            write_neutral(writer, neutral)
                        pending_neutrals = []
                        publisher.send("status", status="streaming", sample_rate_hz=sample_rate_hz, channel_count=channel_count, **metadata)
                    resets = index.resets
                    batch = batch_from_message(message, index, impedance)
                    if index.resets != resets:
                        writer.mark_segment()
                    writer.append(batch)
                    last_data = time.monotonic()
                    start = max(0, batch.sample_count - PREVIEW_SAMPLE_LIMIT)
                    publisher.send(
                        "samples",
                        receive_monotonic_ns=batch.receive_monotonic_ns,
                        sample_index=batch.sample_index[start:].tolist(),
                        signal_uv=batch.signal_uv[start:].tolist(),
                        is_lost=batch.is_lost[start:].tolist(),
                        impedance_ohm=batch.impedance_ohm.tolist(),
                    )
                    if first_batch:
                        first_batch = False
                        publisher.send("ready", sample_rate_hz=sample_rate_hz, channel_count=channel_count, **metadata)
        publisher.send("status", status="stopping")
        return 0
    except Exception as error:  # noqa: BLE001 - reported to the launcher, which decides
        publisher.send("error", message=str(error))
        traceback.print_exc()
        return 1
    finally:
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if writer is not None:
            try:
                if index is not None:
                    writer._group.attrs["sample_index_resets"] = int(index.resets)
                writer._group.attrs["impedance_available"] = bool(impedance_seen)
                writer.close()
            except Exception:
                traceback.print_exc()
        publisher.close()


def main(argv: list[str] | None = None) -> None:
    sys.exit(run(_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
