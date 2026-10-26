"""The subset of MANUS SDK 3.2.1 enums the bridge uses, with the real values."""

from enum import IntEnum


class SDKReturnCode(IntEnum):
    Success = 0
    InvalidArgument = 1
    ArgumentSizeMismatch = 2
    UnsupportedStringSizeEncountered = 3
    SdkNotAvailable = 4
    HostFinderNotAvailable = 5
    DataNotAvailable = 6
    MemoryError = 7
    InternalError = 8
    FunctionCalledAtWrongTime = 9
    NotConnected = 10
    ConnectionTimeout = 11
    InvalidID = 12
    NullPointer = 13
    InvalidSequence = 14
    NoCoordinateSystemSet = 15
    SdkIsTerminating = 16


class Side(IntEnum):
    Invalid = 0
    Left = 1
    Right = 2
    Center = 3


class HandMotion(IntEnum):
    None_ = 0
    IMU = 1
    Tracker = 2
    Tracker_RotationOnly = 3
    Auto = 4


class ChainType(IntEnum):
    Invalid = 0
    Arm = 1
    Leg = 2
    Neck = 3
    Spine = 4
    FingerThumb = 5
    FingerIndex = 6
    FingerMiddle = 7
    FingerRing = 8
    FingerPinky = 9
    Pelvis = 10
    Head = 11
    Shoulder = 12
    Hand = 13
    Foot = 14
    Toe = 15


class FingerJointType(IntEnum):
    Invalid = 0
    Metacarpal = 1
    Proximal = 2
    Intermediate = 3
    Distal = 4
    Tip = 5


class TrackerType(IntEnum):
    Unknown = 0
    Head = 1
    Waist = 2
    LeftHand = 3
    RightHand = 4
    LeftFoot = 5
    RightFoot = 6
    LeftUpperArm = 7
    RightUpperArm = 8
    LeftUpperLeg = 9
    RightUpperLeg = 10
    Controller = 11
    Camera = 12
    MAX_SIZE = 13


class TrackingQuality(IntEnum):
    Untrackable = 0
    BadTracking = 1
    Trackable = 2


class TrackerSystemType(IntEnum):
    Unknown = 0
    Antilatency = 1
    ART = 2
    OpenVR = 3
    Optitrack = 4
    Vicon = 5
    OpenXR = 6


class TrackedPoint(IntEnum):
    Unknown = 0
    Casing = 1
    Joint = 2


class TrackerOffsetType(IntEnum):
    Unknown = 0
    HeadTrackerToHead = 1
    HeadTrackerToTopOfHead = 2
    LeftHandTrackerToWrist = 3
    RightHandTrackerToWrist = 4
    LeftFootTrackerToAnkle = 5
    RightFootTrackerToAnkle = 6
    HipTrackerToHip = 7
    HipTrackerToLeftLeg = 8
    HipTrackerToRightLeg = 9
    LeftUpperArmTrackerToElbow = 10
    RightUpperArmTrackerToElbow = 11
    LeftUpperArmTrackerToShoulder = 12
    RightUpperArmTrackerToShoulder = 13
    LeftHandTrackerToCasing = 14
    RightHandTrackerToCasing = 15
    MAX_SIZE = 16


class SessionType(IntEnum):
    Unknown = 0
    UnityPlugin = 1
    UnrealPlugin = 2
    CoreSDK = 3


class SetGloveCalibrationReturnCode(IntEnum):
    Error = 0
    Success = 1
    VersionError = 2
    WrongSideError = 3
    GloveNotFoundError = 4
    UserServiceError = 5
    DeserializationError = 6
    MAX_SIZE = 7


class AxisView(IntEnum):
    Invalid = 0
    ZFromViewer = 1
    YFromViewer = 2
    XFromViewer = 3
    XToViewer = 4
    YToViewer = 5
    ZToViewer = 6


class AxisPolarity(IntEnum):
    Invalid = 0
    NegativeZ = 1
    NegativeY = 2
    NegativeX = 3
    PositiveX = 4
    PositiveY = 5
    PositiveZ = 6


class LogSeverity(IntEnum):
    Debug = 0
    Info = 1
    Warn = 2
    Error = 3
