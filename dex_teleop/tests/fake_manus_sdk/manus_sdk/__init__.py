"""Offline stand-in for the MANUS Core 3.2 Python SDK (``manus_sdk``).

The bridge under test imports ``lib``, ``ffi``, ``ManusSDK`` and
``manus_sdk.generated._enums`` exactly like the real package. This module
mimics the CFFI surface the bridge touches (``ffi.new`` for the handful of SDK
structs, ``ffi.string``, ``ffi.memmove``, ``ffi.sizeof``, ``ffi.NULL`` and
``ffi.def_extern``) and runs a small scripted "embedded Core": connect
callback, landscape with one right glove owned by user 7, raw-skeleton frames
at a fixed rate. Every mutating SDK call is appended to the JSONL file named by
``DEX_TELEOP_FAKE_MANUS_LOG`` so tests can assert on tracker injection,
offsets and haptics.

Scenarios (``DEX_TELEOP_FAKE_MANUS_SCENARIO``): ``normal`` (default),
``no_license``, ``no_glove``, ``send_trackers_fail``, ``disconnect_after_frames``.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Any, Callable

from . import generated  # noqa: F401 - mirrors the real package layout
from .generated import _enums as E

SCENARIO = os.environ.get("DEX_TELEOP_FAKE_MANUS_SCENARIO", "normal")
LOG_PATH = os.environ.get("DEX_TELEOP_FAKE_MANUS_LOG")
FRAME_HZ = float(os.environ.get("DEX_TELEOP_FAKE_MANUS_FRAME_HZ", "100"))
GLOVE_ID = 1417281806
USER_ID = 7
NODE_COUNT = 25

_log_lock = threading.Lock()


def _log(call: str, **values: Any) -> None:
    if not LOG_PATH:
        return
    with _log_lock, open(LOG_PATH, "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"call": call, **values}) + "\n")


# ---------------------------------------------------------------------------
# Minimal cdata model


class _Char:
    """A fixed-size ``char[N]`` buffer."""

    def __init__(self, size: int) -> None:
        self._data = bytearray(size)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> bytes:
        return bytes(self._data[index : index + 1])

    def __setitem__(self, index: int, value: bytes) -> None:
        self._data[index] = value[0] if isinstance(value, (bytes, bytearray)) else int(value)

    def write(self, data: bytes) -> None:
        data = bytes(data)[: len(self._data)]
        self._data[: len(data)] = data

    def text(self) -> bytes:
        return bytes(self._data).split(b"\0", 1)[0]


SCHEMAS: dict[str, dict[str, str]] = {
    "ManusVec3": {"x": "float", "y": "float", "z": "float"},
    "ManusQuaternion": {"w": "float", "x": "float", "y": "float", "z": "float"},
    "ManusTransform": {"position": "ManusVec3", "rotation": "ManusQuaternion", "scale": "ManusVec3"},
    "ManusTimestamp": {"time": "int"},
    "Version": {"major": "int", "minor": "int", "patch": "int", "label": "char[16]", "sha": "char[16]", "tag": "char[16]"},
    "ManusVersion": {"versionInfo": "char[16]"},
    "ManusHost": {"hostName": "char[256]", "ipAddress": "char[40]", "manusCoreVersion": "Version"},
    "TrackerId": {"id": "char[256]"},
    "TrackerData": {
        "lastUpdateTime": "ManusTimestamp",
        "trackerId": "TrackerId",
        "userId": "int",
        "isHmd": "bool",
        "trackerType": "int",
        "rotation": "ManusQuaternion",
        "position": "ManusVec3",
        "quality": "int",
        "trackedPoint": "int",
    },
    "TrackerStreamInfo": {"publishTime": "ManusTimestamp", "trackerCount": "int"},
    "SkeletonNode": {"id": "int", "transform": "ManusTransform"},
    "RawSkeletonInfo": {"gloveId": "int", "nodesCount": "int", "publishTime": "ManusTimestamp"},
    "SkeletonStreamInfo": {"publishTime": "ManusTimestamp", "skeletonsCount": "int"},
    "NodeInfo": {"nodeId": "int", "parentId": "int", "chainType": "int", "side": "int", "fingerJointType": "int"},
    "CoordinateSystemVUH": {"view": "int", "up": "int", "handedness": "int", "unitScale": "float"},
    "TrackerOffset": {"entryType": "int", "translation": "ManusVec3", "rotation": "ManusQuaternion"},
    "DongleLandscapeData": {"id": "int"},
    "GloveLandscapeData": {"id": "int", "side": "int", "isHaptics": "bool", "dongleID": "int", "pairedState": "int"},
    "DeviceLandscape": {"dongles": "DongleLandscapeData[16]", "dongleCount": "int", "gloves": "GloveLandscapeData[32]", "gloveCount": "int"},
    "UserLandscapeData": {"id": "int", "name": "char[64]", "dongleID": "int", "leftGloveID": "int", "rightGloveID": "int", "userIndex": "int"},
    "UserLandscape": {"users": "UserLandscapeData[16]", "userCount": "int"},
    "SkeletonLandscape": {"skeletonCount": "int"},
    "TrackerLandscapeData": {
        "id": "char[256]",
        "type": "int",
        "systemType": "int",
        "user": "int",
        "isHMD": "bool",
        "manufacturer": "char[256]",
        "productName": "char[256]",
        "trackedPoint": "int",
    },
    "TrackerLandscape": {"trackers": "TrackerLandscapeData[128]", "trackerCount": "int"},
    "LicenseInfo": {"sdk": "bool", "integrated": "bool", "linuxTarget": "bool", "raw": "bool"},
    "SettingsLandscape": {"manusCoreVersion": "Version", "license": "LicenseInfo"},
    "Landscape": {
        "gloveDevices": "DeviceLandscape",
        "users": "UserLandscape",
        "skeletons": "SkeletonLandscape",
        "trackers": "TrackerLandscape",
        "settings": "SettingsLandscape",
    },
}


def _default(type_name: str):
    if type_name == "float":
        return 0.0
    if type_name == "int":
        return 0
    if type_name == "bool":
        return False
    if type_name.startswith("char["):
        return _Char(int(type_name[5:-1]))
    if type_name.endswith("]"):
        element, _, count = type_name[:-1].partition("[")
        return _Array(element, int(count))
    return _Struct(type_name)


class _Struct:
    def __init__(self, type_name: str) -> None:
        object.__setattr__(self, "_type", type_name)
        object.__setattr__(self, "_fields", {})

    def __getattr__(self, name: str):
        schema = SCHEMAS[object.__getattribute__(self, "_type")]
        if name not in schema:
            raise AttributeError(f"{self._type} has no field {name!r}")
        fields = object.__getattribute__(self, "_fields")
        if name not in fields:
            fields[name] = _default(schema[name])
        return fields[name]

    def __setattr__(self, name: str, value) -> None:
        schema = SCHEMAS[object.__getattribute__(self, "_type")]
        if name not in schema:
            raise AttributeError(f"{self._type} has no field {name!r}")
        kind = schema[name]
        if kind == "float":
            value = float(value)
        elif kind == "int":
            value = int(value)
            if value < 0:
                raise OverflowError("unsigned field assigned a negative value")
        elif kind == "bool":
            value = bool(value)
        elif isinstance(value, _Struct) and kind == value._type:
            value = value.copy()
        else:
            raise TypeError(f"cannot assign {value!r} to {self._type}.{name} ({kind})")
        object.__getattribute__(self, "_fields")[name] = value

    def copy(self) -> "_Struct":
        clone = _Struct(self._type)
        for name, value in object.__getattribute__(self, "_fields").items():
            if isinstance(value, _Struct):
                value = value.copy()
            elif isinstance(value, _Char):
                fresh = _Char(len(value))
                fresh.write(bytes(value._data))
                value = fresh
            object.__getattribute__(clone, "_fields")[name] = value
        return clone

    # Pointer-like access: ``p[0]`` dereferences.
    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self


class _Array:
    def __init__(self, element: str, count: int) -> None:
        self.element = element
        self.count = count
        self._items: dict[int, Any] = {}

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        if not 0 <= index < self.count:
            raise IndexError(index)
        if index not in self._items:
            self._items[index] = _default(self.element)
        return self._items[index]

    def __setitem__(self, index: int, value) -> None:
        if not 0 <= index < self.count:
            raise IndexError(index)
        self._items[index] = value


class _Scalar:
    def __init__(self, value=0) -> None:
        self.value = value

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self.value

    def __setitem__(self, index: int, value) -> None:
        if index != 0:
            raise IndexError(index)
        self.value = value


class _Null:
    def __repr__(self) -> str:
        return "<cdata NULL>"


class _FFI:
    NULL = _Null()

    def new(self, ctype: str, init=None):
        ctype = ctype.strip()
        if ctype.endswith("*"):
            base = ctype[:-1].strip()
            if base in SCHEMAS:
                return _Struct(base)
            if base in ("bool", "_Bool"):
                return _Scalar(False)
            return _Scalar(0)
        if ctype.endswith("[]"):
            base = ctype[:-2].strip()
            if base == "unsigned char":
                return bytes(init)
            if isinstance(init, int):
                return _Array(base, init)
            return list(init)
        if "[" in ctype and ctype.endswith("]"):
            base, _, count = ctype[:-1].partition("[")
            base = base.strip()
            count = int(count)
            if base == "float":
                values = [0.0] * count if init is None else [float(value) for value in init]
                if len(values) != count:
                    raise ValueError("initializer length mismatch")
                return values
            return _Array(base, count)
        raise TypeError(f"fake ffi cannot allocate {ctype!r}")

    def string(self, value, maxlen=None) -> bytes:
        if isinstance(value, _Char):
            return value.text()
        if isinstance(value, (bytes, bytearray)):
            data = bytes(value)
            if maxlen is not None:
                data = data[:maxlen]
            return data.split(b"\0", 1)[0]
        raise TypeError(f"fake ffi cannot stringify {value!r}")

    def memmove(self, destination, source, size: int) -> None:
        if not isinstance(destination, _Char):
            raise TypeError("fake ffi.memmove supports char arrays only")
        destination.write(bytes(source)[:size])

    def sizeof(self, value) -> int:
        if isinstance(value, _Char):
            return len(value)
        raise TypeError("fake ffi.sizeof supports char arrays only")

    def def_extern(self):
        def decorate(function: Callable):
            setattr(lib, function.__name__, function)
            return function

        return decorate


ffi = _FFI()


# ---------------------------------------------------------------------------
# Scripted embedded Core


def _topology():
    nodes = [(0, 0, int(E.ChainType.Hand), int(E.FingerJointType.Invalid))]
    node_id = 1
    chains = (
        (E.ChainType.FingerThumb, True),
        (E.ChainType.FingerIndex, False),
        (E.ChainType.FingerMiddle, False),
        (E.ChainType.FingerRing, False),
        (E.ChainType.FingerPinky, False),
    )
    for chain, is_thumb in chains:
        if is_thumb:
            joints = (
                (E.FingerJointType.Metacarpal, 0),
                (E.FingerJointType.Proximal, node_id),
                (E.FingerJointType.Distal, node_id + 1),
                (E.FingerJointType.Tip, node_id + 2),
            )
        else:
            joints = (
                (E.FingerJointType.Metacarpal, 0),
                (E.FingerJointType.Proximal, node_id),
                (E.FingerJointType.Intermediate, node_id + 1),
                (E.FingerJointType.Distal, node_id + 2),
                (E.FingerJointType.Tip, node_id + 3),
            )
        for joint, parent in joints:
            nodes.append((node_id, parent, int(chain), int(joint)))
            node_id += 1
    assert len(nodes) == NODE_COUNT
    return nodes


TOPOLOGY = _topology()
ROOT_POSITION = (1.0, 2.0, 3.0)
ROOT_QUATERNION_XYZW = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))


def _sample_nodes():
    nodes = []
    for node_id in range(NODE_COUNT):
        local = (node_id * 0.001, node_id * 0.002, -node_id * 0.001)
        # Global = root + local rotated by +90 degrees about Z (same as fake_manus_bridge).
        position = (
            ROOT_POSITION[0] - local[1],
            ROOT_POSITION[1] + local[0],
            ROOT_POSITION[2] + local[2],
        )
        nodes.append((node_id, position, ROOT_QUATERNION_XYZW))
    return nodes


SAMPLE_NODES = _sample_nodes()


class _Lib:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._callbacks: dict[str, Callable | None] = {}
        self._hand_motion = int(E.HandMotion.Auto)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._initialized = False
        self._connected = False
        self._frame = 0
        self._tracker_sends = 0
        self.python_raw_skeleton_callback = None
        self.python_skeleton_callback = None
        self.python_landscape_callback = None
        self.python_ergonomics_callback = None
        self.python_raw_device_data_callback = None
        self.python_system_callback = None
        self.python_connected_callback = None
        self.python_disconnected_callback = None
        self.python_log_callback = None
        self.python_tracker_callback = None

    # -- registration -------------------------------------------------------

    def _register(self, name: str, callback) -> int:
        self._callbacks[name] = None if isinstance(callback, _Null) or callback is None else callback
        return int(E.SDKReturnCode.Success)

    def CoreSdk_RegisterCallbackForRawSkeletonStream(self, callback):
        return self._register("raw_skeleton", callback)

    def CoreSdk_RegisterCallbackForSkeletonStream(self, callback):
        return self._register("skeleton", callback)

    def CoreSdk_RegisterCallbackForLandscapeStream(self, callback):
        return self._register("landscape", callback)

    def CoreSdk_RegisterCallbackForErgonomicsStream(self, callback):
        return self._register("ergonomics", callback)

    def CoreSdk_RegisterCallbackForRawDeviceDataStream(self, callback):
        return self._register("raw_device", callback)

    def CoreSdk_RegisterCallbackForSystemStream(self, callback):
        return self._register("system", callback)

    def CoreSdk_RegisterCallbackForOnConnect(self, callback):
        return self._register("connect", callback)

    def CoreSdk_RegisterCallbackForOnDisconnect(self, callback):
        return self._register("disconnect", callback)

    def CoreSdk_RegisterCallbackForOnLog(self, callback):
        return self._register("log", callback)

    def CoreSdk_RegisterCallbackForTrackerStream(self, callback):
        return self._register("tracker", callback)

    # -- lifecycle ----------------------------------------------------------

    def CoreSdk_InitializeIntegrated(self):
        _log("CoreSdk_InitializeIntegrated")
        self._initialized = True
        return int(E.SDKReturnCode.Success)

    def CoreSdk_InitializeCore(self):
        _log("CoreSdk_InitializeCore")
        self._initialized = True
        return int(E.SDKReturnCode.Success)

    def CoreSdk_ShutDown(self):
        _log("CoreSdk_ShutDown")
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None
        self._initialized = False
        self._connected = False
        return int(E.SDKReturnCode.Success)

    def CoreSdk_SetSessionType(self, session_type):
        return int(E.SDKReturnCode.Success)

    def CoreSdk_SetSettingsLocation(self, path):
        return int(E.SDKReturnCode.Success)

    def CoreSdk_SetLogLocation(self, path):
        return int(E.SDKReturnCode.Success)

    def CoreSdk_InitializeCoordinateSystemWithVUH(self, coordinates, use_world):
        _log(
            "CoreSdk_InitializeCoordinateSystemWithVUH",
            view=int(coordinates.view),
            up=int(coordinates.up),
            handedness=int(coordinates.handedness),
        )
        return int(E.SDKReturnCode.Success)

    def CoreSdk_LookForHosts(self, wait_seconds, loopback_only):
        _log("CoreSdk_LookForHosts", wait_seconds=int(wait_seconds), loopback_only=bool(loopback_only))
        return int(E.SDKReturnCode.Success)

    def CoreSdk_GetNumberOfAvailableHostsFound(self, count):
        count[0] = 1
        return int(E.SDKReturnCode.Success)

    def CoreSdk_GetAvailableHostsFound(self, hosts, count):
        host = hosts[0]
        host.hostName.write(b"fake-integrated")
        host.ipAddress.write(b"127.0.0.1")
        host.manusCoreVersion.major = 3
        host.manusCoreVersion.minor = 2
        host.manusCoreVersion.patch = 1
        return int(E.SDKReturnCode.Success)

    def CoreSdk_ConnectToHost(self, host):
        _log("CoreSdk_ConnectToHost", host=host.hostName.text().decode())
        if self._thread is not None:
            return int(E.SDKReturnCode.Success)
        self._stop.clear()
        self._connected = True
        self._thread = threading.Thread(target=self._run_core, name="fake-manus-core", daemon=True)
        self._thread.start()
        return int(E.SDKReturnCode.Success)

    def CoreSdk_Disconnect(self):
        _log("CoreSdk_Disconnect")
        self._stop.set()
        self._connected = False
        return int(E.SDKReturnCode.Success)

    def CoreSdk_GetVersionsAndCheckCompatibility(self, sdk_version, core_version, compatible):
        sdk_version.versionInfo.write(b"3.2.1")
        core_version.versionInfo.write(b"3.2.1")
        compatible[0] = True
        return int(E.SDKReturnCode.Success)

    def CoreSdk_GetSessionId(self, session):
        session[0] = 42
        return int(E.SDKReturnCode.Success)

    def CoreSdk_SetRawSkeletonHandMotion(self, motion):
        _log("CoreSdk_SetRawSkeletonHandMotion", motion=int(motion))
        self._hand_motion = int(motion)
        return int(E.SDKReturnCode.Success)

    def CoreSdk_GetRawSkeletonHandMotion(self, motion):
        motion[0] = self._hand_motion
        return int(E.SDKReturnCode.Success)

    # -- raw skeleton ---------------------------------------------------------

    def CoreSdk_GetRawSkeletonNodeCount(self, glove_id, count):
        if int(glove_id) != GLOVE_ID:
            return int(E.SDKReturnCode.InvalidID)
        count[0] = NODE_COUNT
        return int(E.SDKReturnCode.Success)

    def CoreSdk_GetRawSkeletonNodeInfoArray(self, glove_id, array, size):
        if int(glove_id) != GLOVE_ID or int(size) != NODE_COUNT:
            return int(E.SDKReturnCode.ArgumentSizeMismatch)
        for index, (node_id, parent, chain, joint) in enumerate(TOPOLOGY):
            info = array[index]
            info.nodeId = node_id
            info.parentId = parent
            info.chainType = chain
            info.side = int(E.Side.Right)
            info.fingerJointType = joint
        return int(E.SDKReturnCode.Success)

    def CoreSdk_GetRawSkeletonInfo(self, index, info):
        if int(index) != 0:
            return int(E.SDKReturnCode.InvalidArgument)
        info.gloveId = GLOVE_ID
        info.nodesCount = NODE_COUNT
        info.publishTime.time = 987654321 + self._frame
        return int(E.SDKReturnCode.Success)

    def CoreSdk_GetRawSkeletonData(self, index, nodes, count):
        if int(index) != 0 or int(count) != NODE_COUNT:
            return int(E.SDKReturnCode.ArgumentSizeMismatch)
        for position_index, (node_id, position, quaternion) in enumerate(SAMPLE_NODES):
            node = nodes[position_index]
            node.id = node_id
            node.transform.position.x, node.transform.position.y, node.transform.position.z = position
            x, y, z, w = quaternion
            node.transform.rotation.x = x
            node.transform.rotation.y = y
            node.transform.rotation.z = z
            node.transform.rotation.w = w
            node.transform.scale.x = node.transform.scale.y = node.transform.scale.z = 1.0
        return int(E.SDKReturnCode.Success)

    # -- trackers -------------------------------------------------------------

    def CoreSdk_GetTrackerData(self, index, data):
        return int(E.SDKReturnCode.DataNotAvailable)

    def CoreSdk_SendDataForTrackers(self, data, count):
        self._tracker_sends += 1
        record = {
            "id": data.trackerId.id.text().decode(),
            "user_id": int(data.userId),
            "is_hmd": bool(data.isHmd),
            "tracker_type": int(data.trackerType),
            "position": [data.position.x, data.position.y, data.position.z],
            "rotation_xyzw": [data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w],
            "quality": int(data.quality),
            "tracked_point": int(data.trackedPoint),
            "last_update_time": int(data.lastUpdateTime.time),
        }
        if SCENARIO == "send_trackers_fail":
            _log("CoreSdk_SendDataForTrackers", result="InternalError", **record)
            return int(E.SDKReturnCode.InternalError)
        if self._tracker_sends <= 3 or self._tracker_sends % 50 == 0:
            _log("CoreSdk_SendDataForTrackers", result="Success", **record)
        return int(E.SDKReturnCode.Success)

    def CoreSdk_AssignTrackerToUser(self, tracker_id, user_id):
        _log("CoreSdk_AssignTrackerToUser", id=tracker_id.id.text().decode(), user_id=int(user_id))
        return int(E.SDKReturnCode.Success)

    def CoreSdk_AssignRoleToTracker(self, tracker_id, role):
        _log("CoreSdk_AssignRoleToTracker", id=tracker_id.id.text().decode(), role=int(role))
        return int(E.SDKReturnCode.Success)

    def CoreSdk_SetTrackerOffset(self, user_id, offset):
        _log(
            "CoreSdk_SetTrackerOffset",
            user_id=int(user_id),
            entry_type=int(offset.entryType),
            translation=[offset.translation.x, offset.translation.y, offset.translation.z],
            rotation_xyzw=[offset.rotation.x, offset.rotation.y, offset.rotation.z, offset.rotation.w],
        )
        return int(E.SDKReturnCode.Success)

    # -- haptics / calibration ---------------------------------------------------

    def CoreSdk_VibrateFingersForGlove(self, glove_id, powers):
        _log("CoreSdk_VibrateFingersForGlove", glove_id=int(glove_id), powers=[float(value) for value in powers])
        return int(E.SDKReturnCode.Success)

    def CoreSdk_SetGloveCalibration(self, glove_id, data, length, result):
        _log("CoreSdk_SetGloveCalibration", glove_id=int(glove_id), length=int(length))
        result[0] = int(E.SetGloveCalibrationReturnCode.Success)
        return int(E.SDKReturnCode.Success)

    # -- scripted core thread --------------------------------------------------------

    def _landscape(self):
        landscape = _Struct("Landscape")
        settings = landscape.settings
        settings.manusCoreVersion.major = 3
        settings.manusCoreVersion.minor = 2
        settings.manusCoreVersion.patch = 1
        settings.license.sdk = True
        settings.license.integrated = SCENARIO != "no_license"
        settings.license.linuxTarget = True
        if SCENARIO != "no_glove":
            devices = landscape.gloveDevices
            devices.gloveCount = 1
            glove = devices.gloves[0]
            glove.id = GLOVE_ID
            glove.side = int(E.Side.Right)
            glove.isHaptics = True
            glove.dongleID = 5
            devices.dongleCount = 1
            devices.dongles[0].id = 5
            users = landscape.users
            users.userCount = 1
            user = users.users[0]
            user.id = USER_ID
            user.name.write(b"fake-user")
            user.rightGloveID = GLOVE_ID
            user.dongleID = 5
        return landscape

    def _run_core(self) -> None:
        time.sleep(0.01)
        host = _Struct("ManusHost")
        host.hostName.write(b"fake-integrated")
        host.ipAddress.write(b"127.0.0.1")
        host.manusCoreVersion.major, host.manusCoreVersion.minor, host.manusCoreVersion.patch = 3, 2, 1
        connect = self._callbacks.get("connect")
        if connect is not None:
            connect(host)
        time.sleep(0.01)
        landscape = self._callbacks.get("landscape")
        if landscape is not None:
            landscape(self._landscape())
        if SCENARIO in {"no_license", "no_glove"}:
            return
        period = 1.0 / FRAME_HZ
        disconnect_after = 20 if SCENARIO == "disconnect_after_frames" else None
        while not self._stop.is_set():
            if disconnect_after is not None and self._frame >= disconnect_after:
                callback = self._callbacks.get("disconnect")
                if callback is not None:
                    callback(host)
                return
            stream = _Struct("SkeletonStreamInfo")
            stream.skeletonsCount = 1
            stream.publishTime.time = 987654321 + self._frame
            callback = self._callbacks.get("raw_skeleton")
            if callback is not None:
                callback(stream)
            self._frame += 1
            self._stop.wait(period)


lib = _Lib()


class ManusSDK:
    """Mirror of the real wrapper: register Python callbacks through def_extern."""

    _callback_storage: list = []

    def _register(self, extern_name: str, register, callback: Callable) -> int:
        def trampoline(*args):
            callback(*args)

        trampoline.__name__ = extern_name
        ffi.def_extern()(trampoline)
        ManusSDK._callback_storage.append(trampoline)
        return register(trampoline)

    def register_raw_skeleton_callback(self, callback):
        return self._register("python_raw_skeleton_callback", lib.CoreSdk_RegisterCallbackForRawSkeletonStream, callback)

    def register_skeleton_callback(self, callback):
        return self._register("python_skeleton_callback", lib.CoreSdk_RegisterCallbackForSkeletonStream, callback)

    def register_landscape_callback(self, callback):
        return self._register("python_landscape_callback", lib.CoreSdk_RegisterCallbackForLandscapeStream, callback)

    def register_ergonomics_callback(self, callback):
        return self._register("python_ergonomics_callback", lib.CoreSdk_RegisterCallbackForErgonomicsStream, callback)

    def register_raw_device_data_callback(self, callback):
        return self._register("python_raw_device_data_callback", lib.CoreSdk_RegisterCallbackForRawDeviceDataStream, callback)

    def register_system_callback(self, callback):
        return self._register("python_system_callback", lib.CoreSdk_RegisterCallbackForSystemStream, callback)

    def register_connected_callback(self, callback):
        return self._register("python_connected_callback", lib.CoreSdk_RegisterCallbackForOnConnect, callback)

    def register_disconnected_callback(self, callback):
        return self._register("python_disconnected_callback", lib.CoreSdk_RegisterCallbackForOnDisconnect, callback)

    def register_log_callback(self, callback):
        return self._register("python_log_callback", lib.CoreSdk_RegisterCallbackForOnLog, callback)

    def register_tracker_callback(self, callback):
        return self._register("python_tracker_callback", lib.CoreSdk_RegisterCallbackForTrackerStream, callback)


__all__ = ["ManusSDK", "ffi", "lib"]
