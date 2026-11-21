"""Wire protocol between the dexemg bridge and :class:`dex_teleop.tracking.dexemg.DexemgArticulationSource`.

The bridge (``dex_teleop.emg.dexemg_bridge``) runs in the ``emg2pose`` conda environment: it reads the
wrist EMG band, normalizes it with a neutral recording of the same wearing, runs dexemg's streaming
wrist-band decoder and turns the 20 predicted joint angles into a right hand's MediaPipe-21 landmarks
(meters, OpenXR anatomical wrist frame).  It serves them as newline-delimited JSON on a local TCP socket.
The teleoperation process (``behavior_dex``) connects as a client.  This module has no torch/SDK
dependencies so both sides import it.

Messages, bridge -> client (one JSON object per line):

``hello``   once per connection: protocol, landmark order, frame, handedness, model metadata.
``status``  about twice a second: ``state`` (``connecting``, ``waiting_for_neutral``, ``calibrating``,
            ``streaming``, ``error``), calibration progress, the active channel scale, stream health.
``frame``   one decoded hand: ``seq``, ``timestamp_ns`` (Linux CLOCK_MONOTONIC of the EMG sample the pose
            belongs to), ``emit_ns``, ``sample_index``, ``angles`` (20, rad), ``landmarks`` (21 x 3, m).
``error``   a fatal bridge error; the bridge exits afterwards.

Client -> bridge: ``{"type": "cmd", "action": "calibrate_neutral", "seconds": 12.0}``.

Recording the raw EMG while teleoperating (``dex_teleop.emg.bridge_sidecar``, ``launch_og.py --emg-bridge``):

``{"type": "cmd", "action": "subscribe_raw"}``  this client then also receives every packet as
``emg``     ``sample_index`` (n, the band's counter), ``t_ns`` (n, estimated CLOCK_MONOTONIC per sample),
            ``received_ns`` (CLOCK_MONOTONIC of the SDK callback), ``values`` (n x 8, microvolts as the SDK
            scales them, firmware filters on), ``is_lost`` (n, or null);
``neutral`` whenever a neutral is recorded, and for every neutral so far in reply to
            ``{"type": "cmd", "action": "get_neutral"}``: ``index``, ``time``, ``origin``, ``rejected``,
            ``active``, ``scale`` (8), ``ratio_to_training``, ``samples`` (m x 8 microvolts, or null).
"""

from __future__ import annotations

import json
from typing import Any

from dex_teleop.types import MEDIAPIPE_JOINT_NAMES, OPENXR_ANATOMICAL_WRIST_FRAME

PROTOCOL = "dex_teleop.dexemg/1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
LANDMARK_NAMES = MEDIAPIPE_JOINT_NAMES
LANDMARK_SCHEMA = "mediapipe21"
COORDINATE_FRAME = OPENXR_ANATOMICAL_WRIST_FRAME
# "recording_only": a --record-only bridge with the band streaming (raw EMG to recorders, no decoded hand)
STATES = ("connecting", "waiting_for_neutral", "calibrating", "streaming", "recording_only", "error")
SOURCE_NAME = "dexemg"


def encode(message: dict[str, Any]) -> bytes:
    """One message as a newline-terminated compact JSON line."""

    return (json.dumps(message, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def decode(line: bytes | str) -> dict[str, Any]:
    """Parse one line; raise ``ValueError`` for anything that is not a typed JSON object."""

    value = json.loads(line)
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        raise ValueError("dexemg message must be a JSON object with a string 'type'")
    return value


__all__ = [
    "COORDINATE_FRAME",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "LANDMARK_NAMES",
    "LANDMARK_SCHEMA",
    "PROTOCOL",
    "SOURCE_NAME",
    "STATES",
    "decode",
    "encode",
]
