"""Recording the raw EMG through the dexemg bridge (``--emg-bridge``): bridge stream, sidecar, launcher flag."""

from __future__ import annotations

import json
import socket
from pathlib import Path
import time

import h5py
import numpy as np
import pytest

from dex_teleop.arat import AratTaskCatalog
from dex_teleop.emg.bridge_sidecar import ContinuousIndex, batch_from_message
from dex_teleop.emg.dexemg_bridge import BridgeEngine, FrameServer, SampleBlock
from dex_teleop.emg.dexemg_protocol import PROTOCOL, decode, encode
from dex_teleop.emg.session import EmgSession
from dex_teleop.emg.sidecar import estimate_sample_times
from dex_teleop.omnigibson.launch_config import parse_launch_arguments
from dex_teleop.omnigibson.launcher import _parser, _resolve_tracking_selection, _validate_manus_selection
from dex_teleop.tracking.dexemg import probe_dexemg_bridge

RATE = 500
PERIOD_NS = 2_000_000


class FakeDecoder:
    chunk_steps = 50
    reference_scale = None

    def __init__(self) -> None:
        self.scale = None

    def set_scale(self, scale) -> None:
        self.scale = np.asarray(scale)

    def feed(self, samples):  # never reached: the tests record without decoding
        raise AssertionError("the decoder must not run before a neutral is set")


def _engine() -> BridgeEngine:
    return BridgeEngine(FakeDecoder(), None, emit_hz=50.0, neutral_seconds=1.0, log=False)


def _block(first_index: int, count: int = 10, t0_ns: int = 1_000_000_000, lost=None) -> SampleBlock:
    index = np.arange(first_index, first_index + count, dtype=np.int64)
    values = (index[:, None] + np.arange(8)[None, :] * 0.5).astype(np.float32)
    t_ns = t0_ns + index * PERIOD_NS
    mask = None if lost is None else np.asarray(lost, dtype=np.bool_)
    return SampleBlock(values, index, t_ns, 0, mask, None if lost is None else int(t_ns[-1]) + 300_000)


def test_continuous_index_survives_a_counter_restart():
    index = ContinuousIndex(RATE)
    t = lambda n, start: start + np.arange(n, dtype=np.int64) * PERIOD_NS  # noqa: E731
    first = index(np.arange(1000, 1010), t(10, 0))
    np.testing.assert_array_equal(first, np.arange(1000, 1010))
    # three lost samples: the device counter jumps, the index follows it (no reset)
    second = index(np.arange(1013, 1023), t(10, 26_000_000))
    np.testing.assert_array_equal(second, np.arange(1013, 1023))
    assert index.resets == 0
    # the band reconnects 1 s after the last sample and counts from 0 again
    restart_ns = int(t(10, 26_000_000)[-1]) + 1_000_000_000
    third = index(np.arange(0, 10), t(10, restart_ns))
    assert index.resets == 1
    assert third[0] == second[-1] + RATE  # 1 s at 500 Hz, from the host clock
    np.testing.assert_array_equal(np.diff(third), 1)
    fourth = index(np.arange(10, 20), t(10, restart_ns + 20_000_000))
    assert fourth[0] == third[-1] + 1 and index.resets == 1


def _callbacks(rate_hz: float, seconds: float, rng, first_index: int = 0, start_ns: int = 10**12):
    """8-sample callbacks of a band whose clock gives ``rate_hz`` samples per host second, with receive jitter."""

    count = int(seconds * rate_hz) // 8 * 8
    indices = first_index + np.arange(count, dtype=np.int64)
    true_ns = start_ns + np.round(np.arange(count) * 1e9 / rate_hz).astype(np.int64)
    last_rows = np.arange(7, count, 8)
    delay = rng.uniform(5e6, 90e6, len(last_rows)).astype(np.int64)
    delay[::37] = 5_000_000  # the least-delayed callbacks
    return indices, true_ns, last_rows, true_ns[last_rows] + delay


def test_fitted_sample_period_removes_the_drift_of_the_nominal_one():
    rng = np.random.default_rng(3)
    indices, true_ns, last_rows, receive = _callbacks(501.37, 600.0, rng)
    nominal, _ = estimate_sample_times(indices, last_rows, receive, 2_000_000)
    fitted, info = estimate_sample_times(indices, last_rows, receive, 2_000_000, fit_period=True)
    # nominal: anchored at the end, 1.6 s early at the start of a 10 min recording
    assert abs((nominal - true_ns)[-1]) < 10e6 and (nominal - true_ns)[0] < -1.5e9
    # fitted: the minimum-delay anchor (5 ms here) everywhere
    error_ms = (fitted - true_ns) / 1e6
    assert info["fitted"] == [True] and abs(info["period_ns"][0] - 1e9 / 501.37) < 2.0
    assert 3.0 < error_ms.min() and error_ms.max() < 7.0
    assert np.all(np.diff(fitted) > 0)


def test_fitted_sample_period_handles_a_counter_restart_and_short_recordings():
    rng = np.random.default_rng(4)
    first = _callbacks(501.37, 120.0, rng)
    gap_ns = 7_000_000_000
    second = _callbacks(501.37, 60.0, rng, start_ns=int(first[1][-1]) + gap_ns)
    continuous = ContinuousIndex(RATE)
    indices = np.concatenate([continuous(first[0], first[1]), continuous(second[0], second[1])])
    assert continuous.resets == 1
    true_ns = np.concatenate([first[1], second[1]])
    last_rows = np.concatenate([first[2], second[2] + len(first[0])])
    receive = np.concatenate([first[3], second[3]])
    fitted, info = estimate_sample_times(
        indices, last_rows, receive, 2_000_000, fit_period=True, segment_first_rows=(0, len(first[0]))
    )
    error_ms = (fitted - true_ns) / 1e6
    assert info["fitted"] == [True, True] and 3.0 < error_ms.min() and error_ms.max() < 7.0
    assert np.all(np.diff(fitted) > 0)
    # 4 s of data cannot measure a period: the nominal one is used, as without the fit
    short = _callbacks(500.0, 4.0, rng)
    fitted, info = estimate_sample_times(short[0], short[2], short[3], 2_000_000, fit_period=True)
    nominal, _ = estimate_sample_times(short[0], short[2], short[3], 2_000_000)
    assert info["fitted"] == [False] and np.array_equal(fitted, nominal)


def test_raw_message_round_trips_into_a_writer_batch():
    engine = _engine()
    sent = []
    engine.emit_raw = lambda block: sent.append(decode(encode(engine.raw_message(block))))
    lost = [False] * 9 + [True]
    block = _block(40, lost=lost)
    engine.on_block(block)
    assert engine.state == "waiting_for_neutral" and len(sent) == 1  # recorded although nothing is decoded yet
    batch = batch_from_message(sent[0], ContinuousIndex(RATE))
    np.testing.assert_array_equal(batch.signal_uv, block.values)
    np.testing.assert_array_equal(batch.sample_index, block.sample_index)
    np.testing.assert_array_equal(batch.is_lost, lost)
    assert batch.receive_monotonic_ns == block.received_ns
    assert batch.signal_uv.dtype == np.float32 and not batch.raw_adc.any() and (batch.sdk_timestamp_ms == -1).all()
    # a replayed block knows neither the lost flags nor the callback time
    replayed = decode(encode(engine.raw_message(_block(50))))
    batch = batch_from_message(replayed, ContinuousIndex(RATE))
    assert not batch.is_lost.any() and batch.receive_monotonic_ns == int(_block(50).t_ns[-1])


def test_neutral_messages_carry_the_samples_and_mark_the_active_one():
    engine = _engine()
    emitted = []
    engine.emit_neutral = lambda: emitted.append(engine.neutral_messages(latest_only=True))
    rng = np.random.default_rng(0)
    first = rng.normal(0.0, 1.0, (6000, 8)).astype(np.float32)
    second = rng.normal(0.0, 2.0, (6000, 8)).astype(np.float32)
    engine.load_neutral(first, "live neutral recording")
    engine.load_neutral(second, "live neutral recording")
    assert [len(batch) for batch in emitted] == [1, 1] and emitted[1][0]["index"] == 1
    messages = [decode(encode(message)) for message in engine.neutral_messages()]
    assert [message["active"] for message in messages] == [False, True]
    np.testing.assert_allclose(np.asarray(messages[0]["samples"], np.float32), first)
    np.testing.assert_allclose(messages[1]["scale"], engine.scale, rtol=1e-6)
    assert messages[0]["origin"] == "live neutral recording" and messages[0]["rejected"] is False


def _read_lines(connection: socket.socket, want: int, timeout: float = 3.0) -> list[dict]:
    connection.settimeout(0.1)
    buffer, messages, deadline = b"", [], time.monotonic() + timeout
    while len(messages) < want and time.monotonic() < deadline:
        try:
            chunk = connection.recv(1 << 16)
        except socket.timeout:
            continue
        if not chunk:
            break
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            messages.append(decode(line))
    return messages


def test_only_subscribed_clients_receive_the_raw_stream():
    commands = []
    server = FrameServer(
        "127.0.0.1", 0, commands.append, lambda: [{"type": "hello", "protocol": PROTOCOL}],
        on_query=lambda _message: [{"type": "neutral", "index": 0}],
    )
    port = server._server.getsockname()[1]
    recorder = socket.create_connection(("127.0.0.1", port))
    finger_client = socket.create_connection(("127.0.0.1", port))
    try:
        recorder.sendall(encode({"type": "cmd", "action": "subscribe_raw"}) + encode({"type": "cmd", "action": "get_neutral"}))
        finger_client.sendall(encode({"type": "cmd", "action": "calibrate_neutral"}))
        deadline = time.monotonic() + 3.0
        while (server.raw_client_count != 1 or not commands) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.raw_client_count == 1 and server.client_count == 2
        assert commands == [{"type": "cmd", "action": "calibrate_neutral"}]  # recorder commands stay in the server
        server.broadcast_raw({"type": "emg", "sample_index": [1]})
        server.broadcast({"type": "frame", "seq": 1})
        assert [m["type"] for m in _read_lines(recorder, 4)] == ["hello", "neutral", "emg", "frame"]
        assert [m["type"] for m in _read_lines(finger_client, 2)] == ["hello", "frame"]
    finally:
        recorder.close()
        finger_client.close()
        server.close()


class RawBridge:
    """A FrameServer + engine fed by hand: the bridge without band, decoder or main loop."""

    def __init__(self, state: str = "waiting_for_neutral", neutral=None) -> None:
        self.engine = _engine()
        if neutral is not None:
            self.engine.load_neutral(neutral, "live neutral recording")
        self.engine.state = state
        self.engine.connected = state != "connecting"
        self.engine.band = {"device_name": "OB-test", "device_address": "AA:BB"}

        def greeting():
            return [{"type": "hello", "protocol": PROTOCOL, "sample_rate_hz": RATE, "started": "t0",
                     "checkpoint": "fake.pt", "firmware_filters": {"hpf": True}}, self.engine.status()]

        self.server = FrameServer("127.0.0.1", 0, lambda _m: None, greeting,
                                  on_query=lambda _m: self.engine.neutral_messages())
        self.port = self.server._server.getsockname()[1]
        self.engine.recorders = lambda: self.server.raw_client_count
        self.engine.emit_raw = lambda block: self.server.broadcast_raw(self.engine.raw_message(block))
        self.engine.emit_neutral = lambda: [
            self.server.broadcast_raw(m) for m in self.engine.neutral_messages(latest_only=True)
        ]

    def close(self) -> None:
        self.server.close()


def test_bridge_sidecar_writes_the_stream_and_the_neutrals(tmp_path):
    rng = np.random.default_rng(1)
    neutral = rng.normal(0.0, 1.0, (6000, 8)).astype(np.float32)
    bridge = RawBridge(state="streaming", neutral=neutral)
    bridge.engine.decoder.feed = lambda samples: np.zeros((0, 20), np.float32)
    bridge.engine._decode = lambda *args: None
    output = tmp_path / "emg.hdf5"
    session = EmgSession(output, bridge_address=("127.0.0.1", bridge.port))
    try:
        started = time.monotonic_ns()
        feeding = {"stop": False}

        def feed():  # 10-sample packets at the band's 500 Hz; the device counter restarts once
            sent, device_index, clock_start = 0, 5000, None
            while not feeding["stop"]:
                if bridge.server.raw_client_count:
                    clock_start = clock_start or time.monotonic_ns()
                    while (sent + 10) * PERIOD_NS <= time.monotonic_ns() - clock_start:
                        if device_index == 5200:
                            device_index = 0
                        block = _block(device_index)
                        block.t_ns = clock_start + (sent + 1 + np.arange(10, dtype=np.int64)) * PERIOD_NS
                        bridge.engine.on_block(block)
                        sent += 10
                        device_index += 10
                time.sleep(0.002)

        import threading

        thread = threading.Thread(target=feed, daemon=True)
        thread.start()
        session.start(timeout=20.0)
        assert session.sample_rate_hz == RATE and session.channel_count == 8
        assert session.metadata["device_name"] == "OB-test"
        time.sleep(0.6)
        second = rng.normal(0.0, 1.5, (6000, 8)).astype(np.float32)
        bridge.engine.load_neutral(second, "live neutral recording")  # a neutral recorded during the session
        time.sleep(0.4)
        session.check_health()
        feeding["stop"] = True
        thread.join(timeout=2.0)
        session.close()
    finally:
        session.close()
        bridge.close()
    with h5py.File(output, "r") as recording:
        emg = recording["emg"]
        index = emg["samples/sample_index"][:]
        estimated = emg["samples/estimated_monotonic_ns"][:]
        signal = emg["samples/signal_uv"][:]
        assert len(index) >= 200 and signal.shape == (len(index), 8)
        assert np.all(np.diff(index) > 0) and np.all(np.diff(estimated) > 0)
        assert emg.attrs["sample_index_resets"] == 1
        assert len(emg.attrs["fitted_segment_first_row"]) == 2 and emg.attrs["fitted_segment_first_row"][1] == 200
        assert estimated[0] >= started - 100_000_000 and estimated[-1] <= time.monotonic_ns()
        assert emg.attrs["raw_adc_available"] == False  # noqa: E712
        np.testing.assert_allclose(emg["neutral_0/signal_uv"][:], neutral)
        np.testing.assert_allclose(emg["neutral_1/signal_uv"][:], second)
        assert emg["neutral_0"].attrs["active"] == True  # noqa: E712 - as sent when it was the latest one
        assert emg["neutral_1/channel_scale_uv"].shape == (8,)


def test_bridge_sidecar_reports_a_missing_bridge(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    session = EmgSession(tmp_path / "emg.hdf5", bridge_address=("127.0.0.1", port))
    with pytest.raises(Exception, match="No dexemg bridge|exited"):
        session.start(timeout=20.0)


def test_recording_probe_needs_a_connected_band_but_not_a_neutral():
    bridge = RawBridge(state="waiting_for_neutral")
    try:
        with pytest.raises(SystemExit, match="Record the neutral first"):
            probe_dexemg_bridge("127.0.0.1", bridge.port, timeout=1.0)
        message = probe_dexemg_bridge("127.0.0.1", bridge.port, timeout=1.0, require_decoder=False)
        assert message.startswith("dexemg bridge OK for EMG recording") and "NO OWN NEUTRAL YET" in message
        bridge.engine.state = "connecting"
        with pytest.raises(SystemExit, match="'connecting'"):
            probe_dexemg_bridge("127.0.0.1", bridge.port, timeout=1.0, require_decoder=False)
    finally:
        bridge.close()


def _parse(*argv):
    return parse_launch_arguments(_parser(AratTaskCatalog()), list(argv), default_config=None)[0]


def test_launcher_exposes_emg_bridge_recording():
    task = "arat_grasp_block_10cm"
    args = _parse("--task", task, "--emg-bridge", "--dexemg-port", "8791")
    assert args.emg_bridge and args.dexemg_port == 8791
    _validate_manus_selection(args, _resolve_tracking_selection(args))  # the bridge address is allowed without EMG fingers
    assert not _parse("--task", task).emg_bridge
    # EMG fingers + EMG recording: forbidden through the band (--emg), allowed through the bridge
    args = _parse("--task", task, "--hand-source", "dexemg", "--wrist-source", "quest", "--emg")
    with pytest.raises(SystemExit, match="--emg-bridge"):
        _validate_manus_selection(args, _resolve_tracking_selection(args))
    args = _parse("--task", task, "--hand-source", "dexemg", "--wrist-source", "quest", "--emg-bridge")
    args.emg = True  # what main() does for --emg-bridge
    _validate_manus_selection(args, _resolve_tracking_selection(args))


def test_hand_bench_exposes_emg_bridge_recording():
    from dex_teleop.hand_bench import launcher as hand_bench

    parser = hand_bench.create_parser() if hasattr(hand_bench, "create_parser") else hand_bench._parser()
    glove = parser.parse_args(["--emg-bridge", "--dexemg-port", "8791"])
    hand_bench._validate_args(glove)
    assert glove.emg and glove.record and glove.emg_bridge
    session = hand_bench.create_emg_session(glove, Path("/tmp/does-not-matter.hdf5"))
    assert session.bridge_address == ("127.0.0.1", 8791)
    emg_fingers = parser.parse_args(["--hand-source", "dexemg", "--emg-bridge"])
    hand_bench._validate_args(emg_fingers)
    with pytest.raises(SystemExit, match="--emg-bridge"):
        hand_bench._validate_args(parser.parse_args(["--hand-source", "dexemg", "--emg"]))
    with pytest.raises(SystemExit, match="do not apply"):
        hand_bench._validate_args(parser.parse_args(["--emg-bridge", "--emg-device", "x"]))


def test_band_direct_recording_defaults_to_every_firmware_filter(tmp_path):
    from dex_teleop.emg import sidecar
    from dex_teleop.hand_bench import launcher as hand_bench

    session = EmgSession(tmp_path / "emg.hdf5")
    assert (session.filter_hpf, session.filter_lpf, session.filter_notch) == (True, True, "both")
    assert sidecar._parser().parse_args(["--output", "x", "--preview-port", "1"]).filter_notch == "both"
    args = hand_bench._parser().parse_args(["--emg"])
    hand_bench._validate_args(args)
    assert hand_bench.create_emg_session(args, tmp_path / "rec.hdf5").filter_notch == "both"


def test_record_only_bridge_passes_raw_emg_on_and_never_decodes():
    from dex_teleop.emg.dexemg_bridge import RecordOnlyDecoder

    engine = BridgeEngine(RecordOnlyDecoder(50), None, emit_hz=50.0, neutral_seconds=1.0, log=False, record_only=True)
    sent, frames = [], []
    engine.emit_raw = sent.append
    engine.emit = frames.append
    engine.on_block(_block(0))
    assert engine.state == "recording_only" and engine.status()["record_only"] is True
    # a neutral can still be recorded: it is kept for the recorder, and still nothing is decoded
    rng = np.random.default_rng(5)
    engine.load_neutral(rng.normal(0.0, 1.0, (6000, 8)).astype(np.float32), "live neutral recording")
    assert engine.state == "recording_only" and len(engine.neutral_messages()) == 1
    for start in range(10, 2010, 10):
        engine.on_block(_block(start))
    assert len(sent) == 201 and frames == [] and engine.seq == 0


def test_probes_tell_a_record_only_bridge_apart():
    bridge = RawBridge(state="recording_only")
    bridge.engine.record_only = True
    try:
        message = probe_dexemg_bridge("127.0.0.1", bridge.port, timeout=1.0, require_decoder=False)
        assert message.startswith("dexemg bridge OK for EMG recording (record-only bridge)")
        with pytest.raises(SystemExit, match="--record-only"):
            probe_dexemg_bridge("127.0.0.1", bridge.port, timeout=1.0)
    finally:
        bridge.close()


def _trim_fixture(tmp_path, episodes):
    """A 4 s EMG stream (8-sample callbacks, 2 channels) and a recording whose episodes are (start s, actions)."""
    from dex_teleop.emg import ActionTimingSample, EmgBatch, write_action_timing_episodes
    from dex_teleop.emg.sidecar import EmgHdf5Writer

    emg_path, recording_path = tmp_path / "emg.hdf5", tmp_path / "recording.hdf5"
    writer = EmgHdf5Writer(emg_path, sample_rate_hz=500.0, channel_count=2, metadata={}, fit_sample_period=True)
    origin = 10_000_000_000
    for first in range(0, 2000, 8):
        index = np.arange(first, first + 8, dtype=np.int64)
        values = np.stack([index, -index], axis=1).astype(np.float32)
        writer.append(EmgBatch(
            receive_monotonic_ns=origin + int(index[-1]) * PERIOD_NS, sample_index=index + 700, sdk_timestamp_ms=index,
            signal_uv=values, raw_adc=values.astype(np.int32), is_lost=np.zeros(8, dtype=np.bool_),
            impedance_ohm=np.full(2, float(first), dtype=np.float32), saturation=np.zeros(2, dtype=np.float32)))
    writer._group.create_group("neutral_0").create_dataset("signal_uv", data=np.ones((10, 2), dtype=np.float32))
    writer.close()
    timings = []
    with h5py.File(recording_path, "w") as recording:
        data = recording.create_group("data")
        for episode_id, (start_s, actions) in enumerate(episodes):
            episode = data.create_group(f"demo_{episode_id}")
            episode.attrs["num_samples"] = actions
            episode.create_dataset("action", data=np.zeros((actions, 3), dtype=np.float32))
            start = origin + int(start_s * 1e9)
            timings.append([ActionTimingSample(start + k * 40_000_000, start + (k + 1) * 40_000_000, 0.0, 0.0) for k in range(actions)])
    write_action_timing_episodes(recording_path, timings)
    return emg_path, recording_path, origin


def test_merged_emg_starts_at_the_first_recorded_action_and_ends_with_the_last(tmp_path):
    from dex_teleop.emg import merge_emg_recording

    # SPACE 1.0 s into the stream: 10 actions (0.4 s); reset; SPACE again at 2.5 s: 5 actions (0.2 s)
    emg_path, recording_path, origin = _trim_fixture(tmp_path, [(1.0, 10), (2.5, 5)])
    merge_emg_recording(recording_path, emg_path, trim_to_recording=True)
    with h5py.File(recording_path, "r") as recording, h5py.File(emg_path, "r") as staging:
        emg = recording["emg"]
        times = emg["samples/estimated_monotonic_ns"][:]
        signal = emg["samples/signal_uv"][:]
        assert times[0] >= origin + 1_000_000_000 and times[0] - (origin + 1_000_000_000) < PERIOD_NS
        assert times[-1] <= origin + 2_700_000_000 and (origin + 2_700_000_000) - times[-1] < PERIOD_NS
        rows_before, rows_after = int(emg.attrs["trimmed_rows_before"]), int(emg.attrs["trimmed_rows_after"])
        assert rows_before + len(times) + rows_after == emg.attrs["untrimmed_sample_count"] == 2000
        assert emg.attrs["sample_count"] == len(times) and emg.attrs["trimmed_to_recording"]
        # the kept samples are the same samples, in order
        np.testing.assert_array_equal(signal, staging["emg/samples/signal_uv"][rows_before:rows_before + len(times)])
        np.testing.assert_array_equal(np.diff(emg["samples/sample_index"][:]), 1)
        assert emg.attrs["first_sample_index"] == emg["samples/sample_index"][0] == 700 + rows_before
        # callbacks: every kept row belongs to exactly one kept batch, and the row bookkeeping is rebased
        first, count = emg["batches/first_sample_row"][:], emg["batches/sample_count"][:]
        assert first[0] == 0 and int(count.sum()) == len(times) and np.array_equal(first[1:], np.cumsum(count)[:-1])
        batch_id = emg["samples/batch_id"][:]
        assert batch_id[0] == 0 and batch_id[-1] == len(first) - 1 == emg.attrs["batch_count"] - 1
        assert np.array_equal(np.repeat(np.arange(len(first)), count), batch_id)
        assert emg["batches/impedance_ohm"][0, 0] == (rows_before // 8) * 8
        assert "neutral_0" in emg and emg["neutral_0/signal_uv"].shape == (10, 2)
        # episode / action ranges index the trimmed rows
        first_episode, second_episode = recording["synchronization/demo_0"], recording["synchronization/demo_1"]
        assert first_episode["episode_emg_row_start"][()] == 0
        assert second_episode["episode_emg_row_end"][()] == len(times)
        assert first_episode["action_emg_row_start"][0] == 0 and len(first_episode["action_emg_row_start"]) == 10
        np.testing.assert_array_equal(np.diff(first_episode["action_emg_row_start"][:]), 20)  # 40 ms at 500 Hz
        assert emg.attrs["fitted_segment_first_row"].tolist() == [0]


def test_merged_emg_is_empty_when_nothing_was_recorded_and_whole_without_the_trim(tmp_path):
    from dex_teleop.emg import merge_emg_recording

    emg_path, recording_path, _origin = _trim_fixture(tmp_path, [])
    merge_emg_recording(recording_path, emg_path, trim_to_recording=True)
    with h5py.File(recording_path, "r") as recording:
        assert recording["emg/samples/signal_uv"].shape == (0, 2) and recording["emg/batches/sample_count"].shape == (0,)
        assert recording["emg"].attrs["trimmed_rows_before"] == 0 and recording["emg"].attrs["trimmed_rows_after"] == 2000
    emg_path, recording_path, _origin = _trim_fixture(tmp_path, [(1.0, 10)])
    merge_emg_recording(recording_path, emg_path)
    with h5py.File(recording_path, "r") as recording:
        assert recording["emg/samples/signal_uv"].shape == (2000, 2) and "trimmed_to_recording" not in recording["emg"].attrs


def test_ctrl_c_of_the_terminal_does_not_reach_the_emg_recorder(tmp_path):
    """The recorder runs in its own session: only the launcher gets the terminal's SIGINT and stops it itself."""
    import os

    bridge = RawBridge(state="recording_only")
    session = EmgSession(tmp_path / "emg.hdf5", bridge_address=("127.0.0.1", bridge.port))
    stop = {"flag": False}

    def feed():
        index = 0
        while not stop["flag"]:
            if bridge.server.raw_client_count:
                block = _block(index)
                block.t_ns = time.monotonic_ns() - (9 - np.arange(10, dtype=np.int64)) * PERIOD_NS
                bridge.engine.on_block(block)
                index += 10
            time.sleep(0.02)

    import threading

    thread = threading.Thread(target=feed, daemon=True)
    thread.start()
    try:
        session.start(timeout=20.0)
        assert os.getsid(session._process.pid) != os.getsid(0)
        assert os.getpgid(session._process.pid) != os.getpgrp()
        session.check_health()
    finally:
        stop["flag"] = True
        session.close()
        bridge.close()
    assert session._process is None or session._process.poll() is not None


def test_glove_ergonomics_are_read_back_from_a_teleoperation_recording(tmp_path):
    from dex_teleop.emg.recording import read_glove_ergonomics

    path = tmp_path / "rec.hdf5"
    provenance = [{"glove_id": 1}] + [
        {"glove_id": 1, "ergonomics_deg": json.dumps([float(k + i) for k in range(20)]), "ergonomics_publish_time": 100 + i,
         "ergonomics_arrival_monotonic_ns": 5_000 + i, "ergonomics_order_version": "manus-sdk-3.2-ergonomics-side20-degrees"}
        for i in range(3)
    ]
    with h5py.File(path, "w") as f:
        g = f.create_group("hand_tracking/articulation_streams/articulation.manus")
        g.create_dataset("capture_monotonic_ns", data=np.arange(4, dtype=np.int64) * 10)
        g.create_dataset("provenance_json", data=[json.dumps(p) for p in provenance], dtype=h5py.string_dtype())
    out = read_glove_ergonomics(path)
    assert out["capture_monotonic_ns"].tolist() == [10, 20, 30]
    assert out["angles_deg"].shape == (3, 20) and out["angles_deg"][2, 0] == 2.0
    assert out["publish_time"].tolist() == [100, 101, 102]


def test_impedance_packets_reach_the_status_the_recorder_and_the_preview(tmp_path):
    from types import SimpleNamespace

    from dex_teleop.emg.dexemg_bridge import impedance_from_packet

    packet = SimpleNamespace(channelSamples=[[SimpleNamespace(impedance=1.0), SimpleNamespace(impedance=200_000.0 + 1000 * c)]
                                             for c in range(8)])
    values = impedance_from_packet(packet)
    assert values == [200_000.0 + 1000 * c for c in range(8)]
    assert impedance_from_packet(SimpleNamespace(channelSamples=[[SimpleNamespace(impedance=float("nan"))]] * 8)) is None

    bridge = RawBridge(state="recording_only")
    session = EmgSession(tmp_path / "emg.hdf5", bridge_address=("127.0.0.1", bridge.port))
    bridge.engine.emit_impedance = lambda message: bridge.server.broadcast_raw(message)
    stop = {"flag": False}

    def feed():
        index = 0
        while not stop["flag"]:
            if bridge.server.raw_client_count:
                if index == 100:
                    bridge.engine.set_impedance(values, time.monotonic_ns())
                block = _block(index)
                block.t_ns = time.monotonic_ns() - (9 - np.arange(10, dtype=np.int64)) * PERIOD_NS
                bridge.engine.on_block(block)
                index += 10
            time.sleep(0.02)

    import threading

    thread = threading.Thread(target=feed, daemon=True)
    thread.start()
    try:
        session.start(timeout=20.0)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not np.isfinite(session.preview().impedance_ohm).all():
            time.sleep(0.05)
        assert session.preview().impedance_ohm.tolist() == pytest.approx(values)
        status = bridge.engine.status()
        assert status["impedance_ohm"] == values and status["impedance_age_s"] < 10
    finally:
        stop["flag"] = True
        session.close()
        bridge.close()
    with h5py.File(tmp_path / "emg.hdf5", "r") as f:
        z = f["emg/batches/impedance_ohm"][:]
        assert np.isnan(z[0]).all() and np.allclose(z[-1], values)
        assert f["emg"].attrs["impedance_available"] == True  # noqa: E712


def test_bridge_window_impedance_colours():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("dexemg_gui", Path(__file__).resolve().parents[1] / "scripts/dexemg_gui.py")
    gui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gui)
    from dex_teleop.emg.ui import IMPEDANCE_FAIR_MAX_OHM, IMPEDANCE_GOOD_MAX_OHM

    assert (gui.IMPEDANCE_GOOD_MAX_OHM, gui.IMPEDANCE_FAIR_MAX_OHM) == (IMPEDANCE_GOOD_MAX_OHM, IMPEDANCE_FAIR_MAX_OHM)
    assert gui.impedance_text(None) == ("Z -", gui.GREY)
    assert gui.impedance_text(300_000.0) == ("Z 300 kOhm", gui.GREEN)
    assert gui.impedance_text(800_000.0)[1] == gui.AMBER and gui.impedance_text(2_000_000.0)[1] == gui.RED
    assert gui.impedance_text(300_000.0, stale=True)[1] == gui.GREY


def test_deadband_holds_still_within_the_band_and_follows_without_jumps():
    from dex_teleop.emg.dexemg_bridge import _parser, apply_deadband

    rng = np.random.default_rng(7)
    still = 0.5 + rng.uniform(-0.02, 0.02, (200, 3))          # wander of +-0.02 around 0.5
    out, last = apply_deadband(still, None, 0.05)
    assert np.ptp(out, axis=0).max() == 0.0                    # completely still
    ramp = np.linspace(0.5, 1.5, 200)[:, None] * np.ones((1, 3))
    out2, last2 = apply_deadband(ramp, last, 0.05)
    assert np.all(np.abs(out2 - ramp) <= 0.05 + 1e-12)         # never further than the band from the input
    assert np.abs(np.diff(out2, axis=0)).max() <= np.abs(np.diff(ramp, axis=0)).max() + 1e-12   # no jumps
    assert out2[-1, 0] == pytest.approx(1.45)                  # follows from the band's width behind
    # state carries across chunks: two halves give the same result as one call
    a, mid = apply_deadband(ramp[:100], last, 0.05); b, _ = apply_deadband(ramp[100:], mid, 0.05)
    np.testing.assert_allclose(np.concatenate([a, b]), out2)
    args = _parser().parse_args(["--checkpoint", "x.pt", "--deadband-deg", "4"])
    assert args.deadband_deg == 4.0 and _parser().parse_args(["--checkpoint", "x.pt"]).deadband_deg == 0.0


def test_bridge_limits_its_decoder_threads_by_default():
    from dex_teleop.emg.dexemg_bridge import DEFAULT_DECODER_THREADS, _parser

    assert DEFAULT_DECODER_THREADS == 2
    assert _parser().parse_args(["--checkpoint", "x.pt"]).threads == 2
    assert _parser().parse_args(["--checkpoint", "x.pt", "--threads", "0"]).threads == 0  # PyTorch's own default


def test_bridge_stops_the_sdk_ble_host_on_every_retry():
    """sensor-sdk's controller is one object per process and its terminate() works once; the bridge re-arms it."""

    import queue

    from dex_teleop.emg.dexemg_bridge import BandAcquisition

    class Controller:
        stopped_hosts = 0

        def terminate(self):  # as in sensor-sdk 0.9.11: a second call returns without stopping the new host
            if getattr(self, "_terminated", False):
                return
            self._terminated = True
            self.stopped_hosts += 1

    controller = Controller()
    acquisition = BandAcquisition(queue.Queue(), queue.Queue(), device="x", adapter=None, scan_ms=1, package_count=8)
    for _ in range(3):
        acquisition.controller = controller
        acquisition._terminate_controller()
        assert acquisition.controller is None
    assert controller.stopped_hosts == 3


def test_bridge_turns_the_sdk_log_files_off_unless_asked():
    """The SDK writes a log file per connection attempt; a bridge waiting for the band filled the disk with them."""

    import queue

    from dex_teleop.emg.dexemg_bridge import BandAcquisition, _parser

    class Controller:
        def __init__(self):
            self.calls = []

        def setLogPath(self, enabled=True, path=None):  # sensor-sdk 0.9.11 signature
            self.calls.append((enabled, path))

    def acquisition(**kwargs):
        return BandAcquisition(queue.Queue(), queue.Queue(), device="x", adapter=None, scan_ms=1, package_count=8, **kwargs)

    quiet = acquisition()
    quiet.controller = Controller()
    quiet._quiet_sdk_log()
    assert quiet.controller.calls == [(False, None)]

    kept = acquisition(sdk_log=True)
    kept.controller = Controller()
    kept._quiet_sdk_log()
    assert kept.controller.calls == []

    # an SDK build without the switch: reported once, never fatal
    class Old:
        def setLogPath(self, *args, **kwargs):
            raise TypeError("unexpected keyword")

    old = acquisition()
    old.controller = Old()
    old._quiet_sdk_log()
    old._quiet_sdk_log()
    assert old.events.qsize() == 1 and "log files" in old.events.get()[1]

    assert _parser().parse_args(["--checkpoint", "x.pt"]).sdk_log is False
    assert _parser().parse_args(["--checkpoint", "x.pt", "--sdk-log"]).sdk_log is True
