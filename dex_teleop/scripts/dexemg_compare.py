#!/usr/bin/env python3
"""Live comparison of the EMG-decoded hand with the MANUS glove, as Sharpa finger commands.

Wear the wrist EMG band and the right MANUS glove together, start the EMG bridge
(``dexemg_gui.py`` / ``dexemg_bridge.py``), then, in the ``behavior_dex`` environment:

    python dex_teleop/scripts/dexemg_compare.py --manus-calibration ~/manus_setup/calibration/right.mcal

The tool reads both hands, retargets each with its own copy of the teleoperation retargeter and shows
how far apart the 22 Sharpa joint commands are: mean absolute error in degrees next to a constant
baseline (the glove command's own mean this session), their ratio, the correlation, per finger. It owns
the glove dongle, so run the simulator with ``--hand-source dexemg`` (hand bench, or
``--config dexemg`` on the wrist bench / ARAT launcher), never with the glove, at the same time. It does
not need the simulator at all.

What the number contains: decoder error plus the difference between the decoder's generic hand model
and the glove skeleton, after retargeting. It is the difference between glove teleoperation and EMG
teleoperation of the fingers, not the human joint-angle error of the offline studies.

On exit (Ctrl+C or closing the window) it writes ``summary.json`` (overall and per joint, lag, open and
closed extremes) and ``samples.npz`` (every paired command and both raw landmark streams) to
``--output-dir`` (default ``~/dexemg_teleop_logs/compare_<time>``).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dex_teleop.emg.compare import FINGERS, CommandComparison, format_snapshot  # noqa: E402
from dex_teleop.emg.dexemg_protocol import DEFAULT_HOST, DEFAULT_PORT  # noqa: E402
from dex_teleop.retargeting.retargeter import LandmarkRetargeter  # noqa: E402
from dex_teleop.tracking.dexemg import DexemgArticulationSource  # noqa: E402
from dex_teleop.tracking.openxr import articulation_to_mediapipe21  # noqa: E402
from dex_teleop.types import MEDIAPIPE_JOINT_NAMES, Handedness  # noqa: E402

DEFAULT_SDK_ROOT = Path.home() / "manus_setup" / "vendor" / "ManusSDK_v3.2.1"
RIGHT = Handedness.RIGHT


def landmarks_of(sample) -> np.ndarray:
    """Wrist-local MediaPipe-21 landmarks (21, 3) of any articulation sample."""

    return articulation_to_mediapipe21(sample).positions(MEDIAPIPE_JOINT_NAMES)


class ManusGlove:
    """The right MANUS glove through the teleoperation sidecar (Integrated mode, no tracker)."""

    def __init__(self, args) -> None:
        from dex_teleop.tracking.manus import ManusIntegratedSource

        self.source = ManusIntegratedSource(
            sdk_root=str(args.manus_sdk_root),
            backend="python",
            mode="integrated",
            startup_timeout=args.manus_startup_timeout,
            required_handedness=RIGHT,
            right_calibration=None if args.manus_calibration is None else str(args.manus_calibration),
            require_tracker_for_wrist=False,
        )

    def start(self) -> None:
        self.source.start()

    def poll(self, _emg_samples) -> list[tuple[float, np.ndarray]]:
        self.source.check_health()
        return [(sample.timestamp, landmarks_of(sample)) for sample in self.source.drain_articulations(RIGHT)]

    def close(self) -> None:
        self.source.close()


class ReplayedGlove:
    """Testing: glove landmarks looked up by the EMG frame's sample index (a perfectly synchronous glove).

    ``path`` holds ``sample_index`` (n,) and ``landmarks`` (n, 21, 3) of the take the bridge replays.
    """

    def __init__(self, path: Path, period: int) -> None:
        with np.load(path) as data:
            self.index = np.asarray(data["sample_index"], dtype=np.int64)
            self.landmarks = np.asarray(data["landmarks"], dtype=np.float64)
        self.period = int(period)

    def start(self) -> None:
        pass

    def poll(self, emg_samples) -> list[tuple[float, np.ndarray]]:
        out = []
        for sample in emg_samples:
            position = int(sample.provenance["sample_index"]) % self.period
            row = min(int(np.searchsorted(self.index, position)), len(self.index) - 1)
            out.append((sample.provenance["emg_sample_monotonic_ns"] / 1e9, self.landmarks[row]))
        return out

    def close(self) -> None:
        pass


class Session:
    """Reads both hands, retargets the newest of each, accumulates the comparison and the raw streams."""

    def __init__(self, emg: DexemgArticulationSource, glove, window_s: float) -> None:
        self.emg, self.glove, self.window_s = emg, glove, window_s
        self.emg_retargeter = LandmarkRetargeter.from_hand_model("sharpa", hand_side="right")
        self.glove_retargeter = LandmarkRetargeter.from_hand_model("sharpa", hand_side="right")
        self.comparison = CommandComparison(tuple(self.emg_retargeter.joint_names))
        self.lock = threading.Lock()
        self.raw_emg: list[tuple[float, float, np.ndarray, np.ndarray]] = []  # control t, sample t, landmarks, angles
        self.raw_glove: list[tuple[float, np.ndarray]] = []
        self.latest_emg = None
        self.latest_glove: tuple[float, np.ndarray] | None = None
        self.error: BaseException | None = None
        self.started = time.monotonic()
        self.emg_seen = self.glove_seen = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="dexemg-compare")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def reset(self) -> None:
        with self.lock:
            self.comparison.reset()
            self.started = time.monotonic()

    def _run(self) -> None:
        last_emg_id = last_glove_time = None
        try:
            while not self._stop.is_set():
                self.emg.check_health()
                emg_samples = self.emg.drain_articulations(RIGHT)
                glove_samples = self.glove.poll(emg_samples)
                for sample in emg_samples:
                    angles = np.asarray(json.loads(sample.provenance.get("angles_rad_json") or "[]") or np.full(20, np.nan),
                                        dtype=np.float32)
                    self.raw_emg.append((sample.timestamp, sample.provenance["emg_sample_monotonic_ns"] / 1e9,
                                         landmarks_of(sample).astype(np.float32), angles))
                    self.latest_emg = sample
                for timestamp, landmarks in glove_samples:
                    self.raw_glove.append((timestamp, landmarks.astype(np.float32)))
                    self.latest_glove = (timestamp, landmarks)
                self.emg_seen += len(emg_samples)
                self.glove_seen += len(glove_samples)
                emg, glove = self.latest_emg, self.latest_glove
                if emg is None or glove is None or (emg.source_frame_id == last_emg_id and glove[0] == last_glove_time):
                    self._stop.wait(0.005)
                    continue
                # One pair per pass: the newest hand of each interface, i.e. what each would command now.
                emg_command = self.emg_retargeter.retarget_landmarks(landmarks_of(emg), apply_filter=True, timestamp=emg.timestamp)
                glove_command = self.glove_retargeter.retarget_landmarks(glove[1], apply_filter=True, timestamp=glove[0])
                last_emg_id, last_glove_time = emg.source_frame_id, glove[0]
                with self.lock:
                    self.comparison.add(time.monotonic(), emg.timestamp, emg.provenance["emg_sample_monotonic_ns"] / 1e9,
                                        glove[0], emg_command, glove_command)
        except BaseException as error:  # noqa: BLE001 - surfaced by the foreground loop
            self.error = error

    def snapshots(self) -> tuple[dict | None, dict | None, int]:
        with self.lock:
            return self.comparison.snapshot(self.window_s), self.comparison.snapshot(None), len(self.comparison)

    def save(self, directory: Path, meta: dict) -> dict | None:
        directory.mkdir(parents=True, exist_ok=True)
        with self.lock:
            summary = self.comparison.summary()
            comparison = self.comparison
            np.savez_compressed(
                directory / "samples.npz",
                joint_names=np.asarray(comparison.joint_names),
                pair_time=np.asarray(comparison.times), emg_control_time=np.asarray(comparison.emg_times),
                emg_sample_time=np.asarray(comparison.emg_sample_times), glove_time=np.asarray(comparison.glove_times),
                emg_command_rad=np.asarray(comparison.emg, dtype=np.float32),
                glove_command_rad=np.asarray(comparison.glove, dtype=np.float32),
                raw_emg_control_time=np.asarray([r[0] for r in self.raw_emg]),
                raw_emg_sample_time=np.asarray([r[1] for r in self.raw_emg]),
                raw_emg_landmarks_m=np.asarray([r[2] for r in self.raw_emg], dtype=np.float32),
                raw_emg_angles_rad=np.asarray([r[3] for r in self.raw_emg], dtype=np.float32),
                raw_glove_time=np.asarray([r[0] for r in self.raw_glove]),
                raw_glove_landmarks_m=np.asarray([r[1] for r in self.raw_glove], dtype=np.float32),
            )
        (directory / "summary.json").write_text(json.dumps({**meta, "summary": summary}, indent=1, default=str))
        return summary


def run_gui(session: Session, window_s: float, duration_s: float = 0.0) -> None:
    import tkinter as tk
    from tkinter import ttk

    green, amber, red = "#2e9d57", "#d9941a", "#cc3b3b"
    root = tk.Tk()
    root.title("dexemg - EMG vs glove")
    root.geometry("640x470")
    head = tk.Label(root, text="waiting for both hands...", font=("TkDefaultFont", 16, "bold"), fg="white", bg="#8a8f98", pady=10)
    head.pack(fill="x")
    sub = tk.Label(root, text="", anchor="w", justify="left")
    sub.pack(fill="x", padx=10, pady=4)
    frame = ttk.LabelFrame(root, text=f"Per finger, last {window_s:g} s: EMG vs glove command error (bar: 0-30 deg)")
    frame.pack(fill="x", padx=10, pady=4)
    rows = {}
    for finger in FINGERS:
        row = tk.Frame(frame)
        row.pack(fill="x", padx=8, pady=2)
        tk.Label(row, text=finger, width=7, anchor="w").pack(side="left")
        bar = ttk.Progressbar(row, maximum=30.0, length=330)
        bar.pack(side="left", padx=4)
        label = tk.Label(row, text="-", anchor="w", width=34)
        label.pack(side="left")
        rows[finger] = (bar, label)
    session_label = tk.Label(root, text="", anchor="w", justify="left")
    session_label.pack(fill="x", padx=10, pady=6)
    note = tk.Label(root, anchor="w", justify="left", wraplength=610, fg="#555",
                    text="Sharpa joint commands after retargeting both hands. Constant = the glove command's own mean "
                         "this session. Ratio below 1 means the EMG hand follows the glove better than not moving at all.")
    note.pack(fill="x", padx=10)
    buttons = tk.Frame(root)
    buttons.pack(pady=8)
    ttk.Button(buttons, text="Reset statistics", command=session.reset).pack(side="left", padx=6)
    ttk.Button(buttons, text="Save and quit", command=root.destroy).pack(side="left", padx=6)

    def refresh() -> None:
        if session.error is not None:
            head.config(text=f"STOPPED: {str(session.error)[:70]}", bg=red)
            return
        recent, overall, pairs = session.snapshots()
        if recent is None:
            head.config(text=f"waiting for both hands... (EMG {session.emg_seen}, glove {session.glove_seen} samples)")
        else:
            ratio = recent["ratio"]
            if not recent["glove_moving"]:
                head.config(text="glove hand is not moving - move your fingers", bg="#8a8f98")
            else:
                head.config(
                    text=f"EMG vs glove  {recent['mae_deg']:.1f} deg   (constant {recent['const_deg']:.1f}, ratio {ratio:.2f})",
                    bg=green if ratio < 0.7 else amber if ratio < 1.0 else red,
                )
            sub.config(text=f"last {window_s:g} s: mean correlation {recent['mean_r']:.2f} over {recent['samples']} pairs")
            for finger, (bar, label) in rows.items():
                row = recent["fingers"].get(finger)
                if row is not None:
                    bar["value"] = min(30.0, row["mae_deg"])
                    label.config(text=f"{row['mae_deg']:5.1f} deg   constant {row['const_deg']:4.1f}   r {row['r']:+.2f}")
        if overall is not None:
            session_label.config(text=f"whole session ({overall['seconds']:.0f} s, {pairs} pairs): "
                                      f"{overall['mae_deg']:.2f} deg, constant {overall['const_deg']:.2f}, "
                                      + (f"ratio {overall['ratio']:.2f}, " if overall["glove_moving"] else "")
                                      + f"mean r {overall['mean_r']:.2f}")
        root.after(300, refresh)

    refresh()
    if duration_s > 0:
        root.after(int(duration_s * 1000), root.destroy)
    root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manus-calibration", type=Path, help=".mcal of the wearer's right glove")
    parser.add_argument("--manus-sdk-root", type=Path, default=DEFAULT_SDK_ROOT)
    parser.add_argument("--manus-startup-timeout", type=float, default=35.0)
    parser.add_argument("--dexemg-host", default=DEFAULT_HOST)
    parser.add_argument("--dexemg-port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dexemg-startup-timeout", type=float, default=60.0)
    parser.add_argument("--window", type=float, default=10.0, help="seconds of the 'recent' numbers (default 10)")
    parser.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds (0 = until closed)")
    parser.add_argument("--no-gui", action="store_true", help="print a line every 2 s instead of opening a window")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--glove-replay-npz", type=Path, default=None,
                        help="testing: take the glove from recorded landmarks indexed by the EMG sample index")
    parser.add_argument("--glove-replay-period", type=int, default=0, help="testing: samples in the replayed take")
    args = parser.parse_args()
    started = _dt.datetime.now()
    output = args.output_dir or Path.home() / "dexemg_teleop_logs" / f"compare_{started:%Y%m%d_%H%M%S}"

    emg = DexemgArticulationSource(args.dexemg_host, args.dexemg_port, startup_timeout=args.dexemg_startup_timeout,
                                   stale_timeout=5.0)
    glove = (ReplayedGlove(args.glove_replay_npz, args.glove_replay_period)
             if args.glove_replay_npz is not None else ManusGlove(args))
    print("Connecting to the EMG bridge (it must be streaming: neutral recorded)...", flush=True)
    emg.start()
    print("EMG bridge OK. Starting the MANUS glove (up to 35 s)..." if args.glove_replay_npz is None else "EMG bridge OK.", flush=True)
    try:
        glove.start()
    except BaseException:
        emg.close()
        raise
    session = Session(emg, glove, args.window)
    session.start()
    print(f"Comparing. Results will be saved to {output}", flush=True)
    try:
        if args.no_gui:
            end = None if args.duration <= 0 else time.monotonic() + args.duration
            while session.error is None and (end is None or time.monotonic() < end):
                time.sleep(2.0)
                recent, _overall, _pairs = session.snapshots()
                print(format_snapshot(f"last {args.window:g} s", recent), flush=True)
        else:
            run_gui(session, args.window, args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
        meta = {"started": started.isoformat(timespec="seconds"), "bridge": emg.recording_metadata(),
                "glove": "replay " + str(args.glove_replay_npz) if args.glove_replay_npz is not None else
                {"manus_calibration": None if args.manus_calibration is None else str(args.manus_calibration),
                 "manus_sdk_root": str(args.manus_sdk_root)},
                "window_s": args.window,
                "space": "Sharpa joint commands (rad) from LandmarkRetargeter('sharpa', 'right'), one instance per hand",
                "constant_baseline": "the glove command's mean over the session"}
        summary = session.save(output, meta)
        glove.close()
        emg.close()
    if session.error is not None:
        print(f"Stopped by an error: {type(session.error).__name__}: {session.error}", flush=True)
    if summary is None:
        print("Too few paired samples for a summary.", flush=True)
        return 1
    print(format_snapshot("whole session", summary), flush=True)
    for name, row in summary.get("extremes", {}).items():
        print(f"  {name:6s} 10% of glove flexion: {row['mae_deg']:.2f} deg (constant {row['const_deg']:.2f}), "
              f"EMG - glove flexion {row['flexion_bias_deg']:+.1f} deg", flush=True)
    lag = summary.get("lag", {}).get("control_time")
    if lag:
        print(f"  lag (control time): EMG {lag['emg_behind_glove_ms']:+.0f} ms behind the glove (peak r {lag['peak_r']:.2f})", flush=True)
    print(f"Saved {output / 'summary.json'} and samples.npz", flush=True)
    return 0 if session.error is None else 1


if __name__ == "__main__":
    sys.exit(main())
