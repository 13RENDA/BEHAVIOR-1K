#!/usr/bin/env python3
"""EMG teleop control window: starts the dexemg bridge and shows what it is doing.

    conda activate emg2pose
    python dex_teleop/scripts/dexemg_gui.py --checkpoint ~/claude_work/emg_teleop/deploy_wrist_s0.pt

One window, top to bottom: connection state of the wrist band (colour-coded), the live signal level of
its 8 channels, the neutral recording (button, countdown, progress, result against the training
neutrals), the decoder output rate, whether the simulator is attached, and the bridge's last log lines.
Closing the window stops the bridge. Every option this script does not know is passed to the bridge
(``--adapter hci1``, ``--neutral-seconds 12`` ...). ``--attach`` only monitors a bridge that is
already running.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dex_teleop.emg.dexemg_protocol import DEFAULT_HOST, DEFAULT_PORT, decode, encode  # noqa: E402

BRIDGE = Path(__file__).resolve().parent / "dexemg_bridge.py"
GREEN, AMBER, RED, GREY = "#2e9d57", "#d9941a", "#cc3b3b", "#8a8f98"
# Electrode contact, the thresholds of the SDK example GUI (dex_teleop.emg.ui, shared with the simulator's EMG panel).
IMPEDANCE_GOOD_MAX_OHM, IMPEDANCE_FAIR_MAX_OHM = 500_000.0, 999_000.0
IMPEDANCE_STALE_S = 5.0


def impedance_text(ohm, stale: bool = False) -> tuple[str, str]:
    """(label, colour) of one channel's electrode impedance; grey when unknown or older than IMPEDANCE_STALE_S."""

    if ohm is None or not ohm > 0.0:
        return "Z -", GREY
    colour = GREEN if ohm <= IMPEDANCE_GOOD_MAX_OHM else AMBER if ohm <= IMPEDANCE_FAIR_MAX_OHM else RED
    return f"Z {ohm / 1000.0:.0f} kOhm", GREY if stale else colour
LEVEL_FULL_SCALE_UV = 60.0


def headline(status: dict | None, bridge_alive: bool, socket_up: bool) -> tuple[str, str]:
    """(text, colour) of the big status line, from the bridge's latest status message."""

    if not bridge_alive:
        return "BRIDGE STOPPED", RED
    if status is None or not socket_up:
        return "STARTING BRIDGE...", GREY
    state = status.get("state")
    if state == "error":
        return "ERROR", RED
    if not status.get("connected"):
        return "BAND NOT CONNECTED - searching", RED if "not connected" in (status.get("message") or "") else AMBER
    calibration = status.get("calibration")
    if calibration is not None:
        wait = calibration.get("starts_in_s") or 0.0
        if wait > 0:
            return f"NEUTRAL IN {wait:0.0f} s - get into the neutral pose", AMBER
        return "RECORDING NEUTRAL - hold still", AMBER
    if status.get("record_only"):
        return "CONNECTED - RECORD ONLY (raw EMG to the simulator, no decoding)", GREEN
    if state == "streaming":
        return "CONNECTED - DECODING", GREEN
    return "CONNECTED - neutral not recorded yet", AMBER


class Monitor:
    """Socket client thread: latest status, frame counter, hello, connection flag."""

    def __init__(self, host: str, port: int) -> None:
        self.host, self.port = host, port
        self.status: dict | None = None
        self.hello: dict | None = None
        self.frames = 0
        self.up = False
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sock = socket.create_connection((self.host, self.port), timeout=1.0)
            except OSError:
                self.up = False
                time.sleep(0.5)
                continue
            sock.settimeout(1.0)
            self._sock, self.up, buffer = sock, True, b""
            while not self._stop.is_set():
                try:
                    chunk = sock.recv(1 << 16)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        message = decode(line)
                    except ValueError:
                        continue
                    kind = message["type"]
                    if kind == "status":
                        self.status = message
                    elif kind == "frame":
                        self.frames += 1
                    elif kind == "hello":
                        self.hello = message
            self.up, self._sock = False, None
            sock.close()

    def send(self, action: str, **values) -> bool:
        sock = self._sock
        if sock is None:
            return False
        try:
            sock.sendall(encode({"type": "cmd", "action": action, **values}))
            return True
        except OSError:
            return False

    def close(self) -> None:
        self._stop.set()


class Window:
    def __init__(self, root: tk.Tk, monitor: Monitor, bridge: subprocess.Popen | None, log: deque, neutral_seconds: float) -> None:
        self.root, self.monitor, self.bridge, self.log = root, monitor, bridge, log
        self.neutral_seconds = neutral_seconds
        self._last_frames, self._last_time, self._frame_rate = 0, time.monotonic(), 0.0
        root.title("dexemg - EMG teleop bridge")
        root.geometry("760x720")
        pad = {"padx": 10, "pady": 4}

        self.head = tk.Label(root, text="", font=("TkDefaultFont", 17, "bold"), fg="white", bg=GREY, pady=10)
        self.head.pack(fill="x")
        self.message = tk.Label(root, text="", anchor="w", justify="left", wraplength=730)
        self.message.pack(fill="x", **pad)

        band = ttk.LabelFrame(root, text="Wrist band")
        band.pack(fill="x", **pad)
        self.band = tk.Label(band, text="-", anchor="w", justify="left")
        self.band.pack(fill="x", padx=8, pady=4)

        levels = ttk.LabelFrame(root, text="Signal per channel (RMS of the last 0.5 s, microvolts) and electrode impedance Z")
        levels.pack(fill="x", **pad)
        self.bars, self.bar_labels, self.impedance_labels = [], [], []
        for channel in range(8):
            row = tk.Frame(levels)
            row.pack(fill="x", padx=8, pady=1)
            tk.Label(row, text=f"CH{channel}", width=4, anchor="w").pack(side="left")
            bar = ttk.Progressbar(row, maximum=100.0, length=520)
            bar.pack(side="left", padx=4)
            label = tk.Label(row, text="-", width=22, anchor="w")
            label.pack(side="left")
            impedance = tk.Label(row, text="", width=14, anchor="w")
            impedance.pack(side="left")
            self.bars.append(bar)
            self.bar_labels.append(label)
            self.impedance_labels.append(impedance)

        neutral = ttk.LabelFrame(root, text="Neutral (normalization) - record after every time the band is put on")
        neutral.pack(fill="x", **pad)
        row = tk.Frame(neutral)
        row.pack(fill="x", padx=8, pady=4)
        self.neutral_button = ttk.Button(row, text=f"Record neutral ({neutral_seconds:g} s)", command=self.record_neutral)
        self.neutral_button.pack(side="left")
        self.neutral_progress = ttk.Progressbar(row, maximum=100.0, length=300)
        self.neutral_progress.pack(side="left", padx=10)
        ttk.Button(row, text="Use training scale (fallback)", command=lambda: self.monitor.send("use_reference_scale")).pack(side="left")
        self.relaxed = tk.Label(neutral, text="", anchor="w", font=("TkDefaultFont", 10, "bold"))
        self.relaxed.pack(fill="x", padx=8)
        self.neutral_text = tk.Label(neutral, text="-", anchor="w", justify="left", wraplength=720)
        self.neutral_text.pack(fill="x", padx=8, pady=2)

        decoder = ttk.LabelFrame(root, text="Decoder and simulator")
        decoder.pack(fill="x", **pad)
        self.decoder = tk.Label(decoder, text="-", anchor="w", justify="left")
        self.decoder.pack(fill="x", padx=8, pady=4)

        logs = ttk.LabelFrame(root, text="Bridge log")
        logs.pack(fill="both", expand=True, **pad)
        self.log_text = tk.Text(logs, height=8, state="disabled", wrap="word", font=("TkFixedFont", 9))
        self.log_text.pack(fill="both", expand=True, padx=6, pady=4)

        ttk.Button(root, text="Quit (stops the bridge)" if bridge is not None else "Close", command=self.quit).pack(pady=6)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.refresh()

    def record_neutral(self) -> None:
        if not self.monitor.send("calibrate_neutral", seconds=self.neutral_seconds):
            self.message.config(text="The bridge is not reachable yet.")

    def refresh(self) -> None:
        status = self.monitor.status if self.monitor.up else None
        alive = self.bridge is None or self.bridge.poll() is None
        text, colour = headline(status, alive, self.monitor.up)
        self.head.config(text=text, bg=colour)
        if not alive:
            self.message.config(text=f"The bridge exited with code {self.bridge.returncode}; see the log below.")
        elif status is not None:
            self.message.config(text=status.get("message") or "")
        now = time.monotonic()
        if now - self._last_time >= 1.0:
            self._frame_rate = (self.monitor.frames - self._last_frames) / (now - self._last_time)
            self._last_frames, self._last_time = self.monitor.frames, now
        if status is not None:
            band = status.get("band") or {}
            rate = status.get("rate_hz")
            self.band.config(
                text=(
                    f"{band.get('device_name', '?')}  {band.get('device_address', '')}  on {band.get('ble_adapter', '?')}"
                    f"   firmware {band.get('firmware_version', '?')}\n"
                    f"filters {band.get('firmware_filters', '?')}\n"
                    f"sample rate {rate:.0f} Hz (expected 500)   lost samples {status.get('lost_samples')}   "
                    f"index gaps {status.get('index_gaps')}"
                    if status.get("connected") and rate
                    else "not connected"
                )
            )
            impedance = status.get("impedance_ohm") or [None] * 8
            stale = (status.get("impedance_age_s") or 0.0) > IMPEDANCE_STALE_S
            for channel in range(8):
                text, colour = impedance_text(impedance[channel], stale)
                self.impedance_labels[channel].config(text=text, fg=colour)
            rms = status.get("channel_rms_uv") or [None] * 8
            ratio = status.get("scale_ratio_to_training") or [None] * 8
            reference = status.get("reference_scale") or [None] * 8
            levels_now: list[float] = []
            for channel in range(8):
                value = rms[channel]
                self.bars[channel]["value"] = 0.0 if value is None else min(100.0, 100.0 * value / LEVEL_FULL_SCALE_UV)
                parts = ["-" if value is None else f"{value:6.1f} uV"]
                level = None if value is None or reference[channel] is None else value / reference[channel]
                if level is not None:
                    levels_now.append(level)
                    parts.append("relaxed" if level < 1.5 else f"{level:.1f}x rest")
                self.bar_labels[channel].config(text="   ".join(parts), fg="black" if level is None or level < 1.5 else RED)
            if levels_now and status.get("calibration") is None:
                calm = max(levels_now) < 1.5
                self.relaxed.config(
                    text="All channels at rest level: good moment to record the neutral" if calm
                    else "Not at rest (red channels): relax the forearm before recording the neutral",
                    fg=GREEN if calm else RED,
                )
            calibration = status.get("calibration")
            if calibration is not None:
                self.neutral_progress["value"] = 100.0 * calibration["collected"] / max(1, calibration["target"])
                self.neutral_button.state(["disabled"])
            else:
                self.neutral_progress["value"] = 100.0 if status.get("channel_scale") else 0.0
                self.neutral_button.state(["!disabled"] if status.get("connected") else ["disabled"])
            if status.get("neutral_rejection"):
                self.neutral_text.config(text=status["neutral_rejection"], fg=RED)
            elif status.get("channel_scale") and status.get("record_only"):
                self.neutral_text.config(
                    text=f"recorded {status.get('neutral_recorded') or ''}; saved with the recording "
                    "(record-only: not checked against the training neutrals)", fg="black")
            elif status.get("channel_scale"):
                warning = status.get("scale_warning")
                self.neutral_text.config(
                    text=f"recorded {status.get('neutral_recorded') or ''}; ratio to the training neutrals: "
                    + "  ".join("-" if r is None else f"{r:.2f}" for r in ratio)
                    + (f"\nWARNING: {warning}" if warning else "\nall channels within 1.5x of the training neutrals"),
                    fg=RED if warning else "black",
                )
            else:
                self.neutral_text.config(
                    text="not recorded (optional in record-only mode; a neutral recorded here is saved with the recording)"
                    if status.get("record_only") else "not recorded yet: the decoder sends no hand until it is", fg="black")
            recorders = int(status.get("recorders") or 0)  # launchers recording the raw EMG (--emg-bridge)
            simulators = max(0, int(status.get("clients") or 0) - 1 - recorders)
            decode_ms = status.get("decode_ms_median")
            self.decoder.config(
                text=("record-only: no hand is decoded\n" if status.get("record_only") else
                      f"hands sent: {self._frame_rate:.0f} per second (expected 50)   total {status.get('frames')}   "
                      f"decode {'-' if decode_ms is None else f'{decode_ms:.1f} ms'} per 100 ms chunk\n")
                + f"simulator attached: {'YES' if simulators else 'no'}"
                + (f" ({simulators} clients)" if simulators > 1 else "")
                + f"   raw EMG being recorded by the simulator: {'YES' if recorders else 'no'}"
            )
        lines = list(self.log)[-40:]
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("end", "\n".join(lines))
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.root.after(200, self.refresh)

    def quit(self) -> None:
        self.monitor.send("quit")
        self.monitor.close()
        if self.bridge is not None and self.bridge.poll() is None:
            try:
                self.bridge.wait(timeout=12)
            except subprocess.TimeoutExpired:
                self.bridge.kill()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, help="wrist-band dexemg checkpoint (required unless --attach or --record-only)")
    parser.add_argument("--record-only", action="store_true",
                        help="no decoding: the bridge only owns the band and passes its raw EMG on to a launcher "
                             "started with --emg-bridge (glove teleoperation + EMG recording); no checkpoint needed")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--neutral-seconds", type=float, default=12.0)
    parser.add_argument("--attach", action="store_true", help="monitor a bridge that is already running")
    parser.add_argument("--auto-neutral", action="store_true",
                        help="start the neutral recording as soon as the band connects (default: wait for the button)")
    args, bridge_args = parser.parse_known_args()
    log: deque = deque(maxlen=400)
    bridge = None
    if not args.attach:
        if args.record_only and args.checkpoint is not None:
            parser.error("--record-only does not decode: drop --checkpoint")
        if args.checkpoint is None and not args.record_only:
            parser.error("--checkpoint is required (or use --record-only / --attach)")
        command = [sys.executable, "-u", str(BRIDGE),
                   *(("--record-only",) if args.record_only else ("--checkpoint", str(args.checkpoint.expanduser()))),
                   "--host", args.host, "--port", str(args.port), "--neutral-seconds", str(args.neutral_seconds)]
        if not args.auto_neutral:
            command.append("--no-neutral-on-start")
        command += bridge_args
        bridge = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True)

        def pump() -> None:
            for line in bridge.stdout:
                line = line.rstrip()
                if line and "Warning" not in line and "ref-compat" not in line:
                    log.append(line)
                    print(line, flush=True)

        threading.Thread(target=pump, daemon=True).start()
    monitor = Monitor(args.host, args.port)
    root = tk.Tk()
    Window(root, monitor, bridge, log, args.neutral_seconds)
    try:
        root.mainloop()
    finally:
        monitor.close()
        if bridge is not None and bridge.poll() is None:
            bridge.terminate()


if __name__ == "__main__":
    main()
