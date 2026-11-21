"""Per-finger haptic feedback from simulated contact, Isaac Teleop style.

NVIDIA Isaac Teleop models haptics as a graph phase symmetric to tracking
input: a *retargeter* maps sim-side tactile data (``TactileVector``) to a
device-side schema, a vendor-neutral *haptic sink* stores one value per
endpoint and flushes it after the step, and an ``IHapticDevice`` adapter
delivers it to the hardware (for MANUS Metagloves Pro Haptic: five finger
amplitudes ordered thumb to pinky). This module follows that contract for the
ARAT launcher so the same pieces could later be wired into an Isaac Teleop
``TeleopSessionConfig(sinks=[...])``:

- :class:`FingerContactForces` is the tactile vector: the net contact-force
  magnitude on each digit of the simulated Sharpa hand, read from
  ``robot.get_finger_contact_data`` and classified with ``HandSemantics``.
- :class:`ContactForceToFingerAmplitudes` is the tactile retargeter: force to
  a 0..1 amplitude per finger with a contact threshold, saturation force, a
  minimum perceptible amplitude and exponential smoothing.
- :class:`ManusGloveHapticDevice` is the ``IHapticDevice``: it forwards the
  amplitudes to the tracking worker, which relays them to the MANUS sidecar
  (``CoreSdk_VibrateFingersForGlove``).
- :class:`HapticSink` stores the frame's amplitudes and flushes them at a
  bounded rate, resending only when they changed or a heartbeat is due
  (the glove keeps the last command until told otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

HAPTIC_FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
"""Actuator order of a MANUS Metaglove Pro Haptic (thumb first), as in Isaac Teleop."""


class TactileVector(Protocol):
    """Isaac Teleop's sim-side tactile schema: one scalar per channel."""

    @property
    def values(self) -> np.ndarray: ...


@dataclass(frozen=True)
class FingerContactForces:
    """Net contact-force magnitude per digit (newtons), thumb..pinky."""

    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if values.shape != (len(HAPTIC_FINGER_ORDER),):
            raise ValueError(f"expected {len(HAPTIC_FINGER_ORDER)} finger forces, got {values.shape}")
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError("finger contact forces must be finite and non-negative")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    @classmethod
    def zero(cls) -> "FingerContactForces":
        return cls(np.zeros(len(HAPTIC_FINGER_ORDER)))

    @classmethod
    def from_contacts(
        cls,
        contacts: Iterable[tuple],
        classify_link: Callable[[str], tuple[str, str] | None],
        *,
        pad_weight: float = 1.0,
        dorsal_weight: float = 0.5,
    ) -> "FingerContactForces":
        """Sum contact-force magnitudes per digit.

        ``contacts`` are the 6-tuples of ``robot.get_finger_contact_data``
        (hand link path, other body path, position, normal, force, separation);
        ``classify_link`` maps a hand link *name* to (digit, part). Pad (fingertip
        elastomer) contacts count fully, other finger links count ``dorsal_weight``;
        palm contacts are not fed back (no actuator).
        """

        totals = np.zeros(len(HAPTIC_FINGER_ORDER))
        for hand_path, _other, _position, _normal, force, _separation in contacts:
            classified = classify_link(str(hand_path).rsplit("/", 1)[-1])
            if classified is None:
                continue
            digit, part = classified
            if digit not in HAPTIC_FINGER_ORDER:
                continue
            magnitude = float(np.linalg.norm(np.asarray(_as_float_array(force), dtype=np.float64)))
            if not math.isfinite(magnitude):
                continue
            weight = pad_weight if part == "pad" else dorsal_weight
            totals[HAPTIC_FINGER_ORDER.index(digit)] += weight * magnitude
        return cls(totals)

    def as_mapping(self) -> dict[str, float]:
        return {finger: float(value) for finger, value in zip(HAPTIC_FINGER_ORDER, self.values)}


def _as_float_array(value):
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        detached = getattr(value, "detach", None)
        if callable(detached):
            value = detached()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        return value.tolist()
    return value


@dataclass(frozen=True)
class HapticAmplitudes:
    """Device-side schema for the glove: five actuator amplitudes in 0..1."""

    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if values.shape != (len(HAPTIC_FINGER_ORDER),):
            raise ValueError(f"expected {len(HAPTIC_FINGER_ORDER)} amplitudes, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("haptic amplitudes must be finite")
        values = np.clip(values, 0.0, 1.0)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    @classmethod
    def silent(cls) -> "HapticAmplitudes":
        return cls(np.zeros(len(HAPTIC_FINGER_ORDER)))

    @property
    def active(self) -> bool:
        return bool((self.values > 0.0).any())

    def as_list(self) -> list[float]:
        return [float(value) for value in self.values]


@dataclass(frozen=True)
class HapticConfig:
    """Tactile-to-amplitude mapping and delivery policy."""

    # Contact force (N) below which a finger is considered not touching.
    minimum_force_n: float = 0.5
    # Contact force (N) at which the actuator saturates at full amplitude.
    maximum_force_n: float = 6.0
    # Amplitude floor applied as soon as a finger touches, so light contact is felt.
    minimum_amplitude: float = 0.15
    # Exponential smoothing factor per update (1.0 = no smoothing); rising
    # edges are applied immediately so contact onset is never delayed.
    smoothing: float = 0.5
    # Maximum command rate towards the glove.
    update_hz: float = 30.0
    # Resend unchanged non-zero amplitudes at least this often.
    heartbeat_s: float = 0.5
    # Amplitude change that counts as "changed" for resend purposes.
    change_epsilon: float = 0.02
    # Assisted grasp: while an object is held, the grasping digits vibrate at least this much (the hand
    # no longer collides with an object on a compliant weld, so contact alone would go silent), after a
    # full-amplitude pulse of grasp_pulse_s when the weld forms. 0 switches the grasp cue off.
    grasp_hold_amplitude: float = 0.35
    grasp_pulse_s: float = 0.15

    def __post_init__(self) -> None:
        if not (0.0 <= self.minimum_force_n < self.maximum_force_n) or not math.isfinite(self.maximum_force_n):
            raise ValueError("haptics require 0 <= minimum_force_n < maximum_force_n (finite)")
        if not 0.0 <= self.minimum_amplitude <= 1.0:
            raise ValueError("minimum_amplitude must be within [0, 1]")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("smoothing must be within (0, 1]")
        if not math.isfinite(self.update_hz) or self.update_hz <= 0.0:
            raise ValueError("update_hz must be positive and finite")
        if not math.isfinite(self.heartbeat_s) or self.heartbeat_s <= 0.0:
            raise ValueError("heartbeat_s must be positive and finite")
        if not 0.0 <= self.change_epsilon < 1.0:
            raise ValueError("change_epsilon must be within [0, 1)")
        if not 0.0 <= self.grasp_hold_amplitude <= 1.0:
            raise ValueError("grasp_hold_amplitude must be within [0, 1]")
        if not math.isfinite(self.grasp_pulse_s) or self.grasp_pulse_s < 0.0:
            raise ValueError("grasp_pulse_s must be non-negative and finite")

    def as_mapping(self) -> dict[str, float]:
        return {
            "minimum_force_n": self.minimum_force_n,
            "maximum_force_n": self.maximum_force_n,
            "minimum_amplitude": self.minimum_amplitude,
            "smoothing": self.smoothing,
            "update_hz": self.update_hz,
            "heartbeat_s": self.heartbeat_s,
            "change_epsilon": self.change_epsilon,
            "grasp_hold_amplitude": self.grasp_hold_amplitude,
            "grasp_pulse_s": self.grasp_pulse_s,
        }


class ContactForceToFingerAmplitudes:
    """Tactile retargeter: finger contact forces to glove amplitudes."""

    def __init__(self, config: HapticConfig | None = None) -> None:
        self.config = config or HapticConfig()
        self._state = np.zeros(len(HAPTIC_FINGER_ORDER))

    def reset(self) -> None:
        self._state[:] = 0.0

    def __call__(self, forces: FingerContactForces) -> HapticAmplitudes:
        config = self.config
        span = config.maximum_force_n - config.minimum_force_n
        normalized = np.clip((forces.values - config.minimum_force_n) / span, 0.0, 1.0)
        touching = forces.values > config.minimum_force_n
        target = np.where(
            touching,
            config.minimum_amplitude + (1.0 - config.minimum_amplitude) * normalized,
            0.0,
        )
        rising = target > self._state
        smoothed = self._state + config.smoothing * (target - self._state)
        self._state = np.where(rising, target, smoothed)
        # Let decays settle to exactly zero instead of lingering asymptotically.
        self._state[self._state < 0.01] = 0.0
        return HapticAmplitudes(self._state.copy())


class IHapticDevice(Protocol):
    """Vendor adapter contract (mirrors Isaac Teleop's ``IHapticDevice``)."""

    @property
    def endpoints(self) -> tuple[str, ...]: ...

    def apply(self, endpoint: str, amplitudes: HapticAmplitudes) -> bool: ...

    def flush(self) -> None: ...


class ManusGloveHapticDevice:
    """``IHapticDevice`` that relays amplitudes to the MANUS sidecar via the worker."""

    def __init__(self, worker, *, endpoint: str = "right") -> None:
        apply = getattr(worker, "apply_haptic_feedback", None)
        if not callable(apply):
            raise TypeError("worker must provide apply_haptic_feedback(handedness, amplitudes)")
        self._worker = worker
        self._endpoint = endpoint
        self.sent_commands = 0

    @property
    def endpoints(self) -> tuple[str, ...]:
        return (self._endpoint,)

    def apply(self, endpoint: str, amplitudes: HapticAmplitudes) -> bool:
        if endpoint != self._endpoint:
            raise KeyError(endpoint)
        accepted = bool(self._worker.apply_haptic_feedback(endpoint, amplitudes.as_list()))
        if accepted:
            self.sent_commands += 1
        return accepted

    def flush(self) -> None:
        return None


@dataclass
class HapticSink:
    """Stores the frame's amplitudes per endpoint and flushes them to the device."""

    device: IHapticDevice
    config: HapticConfig = field(default_factory=HapticConfig)
    clock: Callable[[], float] = time.monotonic
    _pending: dict[str, HapticAmplitudes] = field(default_factory=dict, init=False)
    _last_sent: dict[str, HapticAmplitudes] = field(default_factory=dict, init=False)
    _last_sent_time: dict[str, float] = field(default_factory=dict, init=False)
    _next_allowed_time: float = field(default=0.0, init=False)

    def write(self, endpoint: str, amplitudes: HapticAmplitudes) -> None:
        if endpoint not in self.device.endpoints:
            raise KeyError(endpoint)
        self._pending[endpoint] = amplitudes

    def flush(self) -> list[str]:
        """Send pending values that changed (or are due a heartbeat). Returns endpoints sent."""

        now = self.clock()
        sent: list[str] = []
        if not self._pending or now < self._next_allowed_time:
            return sent
        for endpoint, amplitudes in list(self._pending.items()):
            previous = self._last_sent.get(endpoint)
            changed = previous is None or bool(
                (np.abs(amplitudes.values - previous.values) > self.config.change_epsilon).any()
            )
            heartbeat_due = (
                amplitudes.active
                and now - self._last_sent_time.get(endpoint, -math.inf) >= self.config.heartbeat_s
            )
            silence_edge = previous is not None and previous.active and not amplitudes.active
            if not (changed or heartbeat_due or silence_edge):
                continue
            if self.device.apply(endpoint, amplitudes):
                self._last_sent[endpoint] = amplitudes
                self._last_sent_time[endpoint] = now
                sent.append(endpoint)
        self._pending.clear()
        if sent:
            self._next_allowed_time = now + 1.0 / self.config.update_hz
        self.device.flush()
        return sent

    def silence(self) -> None:
        """Stop every actuator immediately, bypassing the rate limit."""

        for endpoint in self.device.endpoints:
            self.device.apply(endpoint, HapticAmplitudes.silent())
            self._last_sent[endpoint] = HapticAmplitudes.silent()
            self._last_sent_time[endpoint] = self.clock()
        self._pending.clear()
        self.device.flush()


class ContactHapticFeedback:
    """End-to-end controller: robot contacts -> amplitudes -> glove, once per step."""

    def __init__(
        self,
        robot,
        classify_link: Callable[[str], tuple[str, str] | None],
        device: IHapticDevice,
        *,
        arm: str = "default",
        endpoint: str = "right",
        config: HapticConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.robot = robot
        self.classify_link = classify_link
        self.config = config or HapticConfig()
        self.retargeter = ContactForceToFingerAmplitudes(self.config)
        self.sink = HapticSink(device, config=self.config, clock=clock)
        self.arm = arm
        self.endpoint = endpoint
        self.last_forces = FingerContactForces.zero()
        self.last_amplitudes = HapticAmplitudes.silent()
        self._clock = clock
        self._held_digits: frozenset[str] = frozenset()
        self._held_since = 0.0

    def grasp_cue(self, grasp_digits: Iterable[str] | None) -> np.ndarray:
        """Amplitude floor per finger from the assisted grasp: a pulse when the weld forms, then a steady hold."""

        digits = frozenset(grasp_digits or ()) & frozenset(HAPTIC_FINGER_ORDER)
        if digits and not self._held_digits:
            self._held_since = self._clock()
        self._held_digits = digits
        cue = np.zeros(len(HAPTIC_FINGER_ORDER))
        if not digits or self.config.grasp_hold_amplitude <= 0.0:
            return cue
        level = 1.0 if self._clock() - self._held_since < self.config.grasp_pulse_s else self.config.grasp_hold_amplitude
        for index, finger in enumerate(HAPTIC_FINGER_ORDER):
            if finger in digits:
                cue[index] = level
        return cue

    def step(self, contacts: Sequence[tuple] | None = None, grasp_digits: Iterable[str] | None = None) -> HapticAmplitudes:
        """Contact forces to amplitudes; ``grasp_digits`` are the digits holding a welded object, if any."""

        if contacts is None:
            contacts = self.robot.get_finger_contact_data(self.arm)
        self.last_forces = FingerContactForces.from_contacts(contacts, self.classify_link)
        contact = self.retargeter(self.last_forces)
        self.last_amplitudes = HapticAmplitudes(np.maximum(contact.values, self.grasp_cue(grasp_digits)))
        self.sink.write(self.endpoint, self.last_amplitudes)
        self.sink.flush()
        return self.last_amplitudes

    def reset(self) -> None:
        self._held_digits = frozenset()
        self.retargeter.reset()
        self.last_forces = FingerContactForces.zero()
        self.last_amplitudes = HapticAmplitudes.silent()
        self.sink.silence()

    def metadata(self) -> Mapping[str, object]:
        return {
            "provider": "contact-force finger haptics (Isaac Teleop HapticSink contract)",
            "finger_order": list(HAPTIC_FINGER_ORDER),
            "endpoint": self.endpoint,
            **self.config.as_mapping(),
        }


__all__ = [
    "HAPTIC_FINGER_ORDER",
    "ContactForceToFingerAmplitudes",
    "ContactHapticFeedback",
    "FingerContactForces",
    "HapticAmplitudes",
    "HapticConfig",
    "HapticSink",
    "IHapticDevice",
    "ManusGloveHapticDevice",
    "TactileVector",
]
