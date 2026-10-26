"""YAML launch presets for ``launch_og.py``.

A preset is a flat YAML mapping from ``launch_og.py`` option names to values.
Keys are the option names with the leading dashes removed; ``-`` and ``_`` are
interchangeable (``manus-core-host`` and ``manus_core_host`` both address
``--manus-core-host``). Every key, choice, and value type is validated against
the live argparse parser, so a preset can only express what the command line
can express. The validated values are installed as parser *defaults*: any
option given explicitly on the command line still wins.

Value rules:

- ``null`` (or ``~``) keeps the built-in default; it never sets ``None``.
- ``store_true``/``--[no-]flag`` options take ``true``/``false``.
- repeatable options (``--record-hand-source``) take a list; command-line
  values *replace* the preset list rather than appending to it.
- other scalars follow the option's ``type`` and ``choices`` exactly as the
  command line does (``assisted_grasp_break_force: none`` disables the weld
  break force just like ``--assisted-grasp-break-force none``).

Presets shipped with the repository live in ``dex_teleop/configs/<name>.yaml``
and are selected by bare name (``--config hts``); any other YAML file is
selected by path. ``--no-config`` disables presets entirely.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import difflib
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

LAUNCH_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"
DEFAULT_LAUNCH_CONFIG = "manus_core"
CONFIG_OPTION_DESTS = frozenset({"config", "no_config"})
_YAML_SUFFIXES = frozenset({".yaml", ".yml"})


@dataclass(frozen=True)
class LaunchConfig:
    """A validated preset: option destinations mapped to parser-ready values."""

    path: Path
    values: dict[str, Any]
    sha256: str

    @property
    def name(self) -> str:
        return self.path.stem

    def fingerprint(self) -> dict[str, str]:
        return {"file": self.path.name, "sha256": self.sha256}


def available_launch_configs(root: Path = LAUNCH_CONFIG_ROOT) -> tuple[str, ...]:
    """Names of the presets shipped in ``dex_teleop/configs``."""

    if not root.is_dir():
        return ()
    return tuple(sorted(path.stem for path in root.glob("*.yaml") if path.is_file()))


def resolve_launch_config_path(value: str | Path, root: Path = LAUNCH_CONFIG_ROOT) -> Path:
    """Map a preset name (``configs/<name>.yaml``) or an explicit path to a file."""

    text = str(value).strip()
    if not text:
        raise SystemExit("--config requires a preset name or a YAML file path")
    candidate = Path(text).expanduser()
    looks_like_path = (
        candidate.suffix.lower() in _YAML_SUFFIXES
        or candidate.is_absolute()
        or len(candidate.parts) > 1
    )
    if looks_like_path:
        if candidate.is_file():
            return candidate.resolve()
        raise SystemExit(f"Launch config file not found: {candidate}")
    preset = root / f"{text}.yaml"
    if preset.is_file():
        return preset.resolve()
    names = ", ".join(available_launch_configs(root)) or "none found"
    raise SystemExit(
        f"Unknown launch config preset {text!r}. Presets in {root}: {names}. "
        "Pass a .yaml path to use a file elsewhere, or --no-config for the built-in defaults."
    )


def _option_actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    actions: dict[str, argparse.Action] = {}
    for action in parser._actions:  # noqa: SLF001 - argparse has no public action registry
        if not action.option_strings or isinstance(action, argparse._HelpAction):  # noqa: SLF001
            continue
        actions.setdefault(action.dest, action)
    return actions


def _normalize_key(raw_key: object, path: Path) -> str:
    if not isinstance(raw_key, str) or not raw_key.strip():
        raise SystemExit(f"{path}: option names must be non-empty strings, got {raw_key!r}")
    return raw_key.strip().lstrip("-").replace("-", "_")


def _display_flag(action: argparse.Action) -> str:
    long_options = [option for option in action.option_strings if option.startswith("--")]
    return (long_options or action.option_strings)[0]


def _coerce_scalar(action: argparse.Action, raw: Any, fail) -> Any:
    if isinstance(raw, (list, tuple, dict)):
        fail("expected a single scalar value")
    if isinstance(raw, bool):
        fail("expected a value, not true/false")
    text = raw if isinstance(raw, str) else str(raw)
    if action.type is None:
        value: Any = text
    else:
        try:
            value = action.type(text)
        except (ValueError, TypeError, argparse.ArgumentTypeError) as error:
            detail = str(error) or error.__class__.__name__
            fail(f"invalid value {raw!r} ({detail})")
    if action.choices is not None and value not in action.choices:
        choices = ", ".join(str(choice) for choice in action.choices)
        fail(f"must be one of {choices}; got {raw!r}")
    return value


def _coerce(action: argparse.Action, raw: Any, *, path: Path, key: str) -> Any:
    def fail(detail: str) -> None:
        raise SystemExit(f"{path}: {key} ({_display_flag(action)}): {detail}")

    if isinstance(
        action,
        (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction),  # noqa: SLF001
    ):
        if not isinstance(raw, bool):
            fail("expected true or false")
        return raw
    if isinstance(action, argparse._AppendAction):  # noqa: SLF001
        items = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        return [_coerce_scalar(action, item, fail) for item in items]
    if isinstance(action, argparse._StoreAction) and action.nargs in (None, "?"):  # noqa: SLF001
        return _coerce_scalar(action, raw, fail)
    fail("this option cannot be set from a launch config; pass it on the command line")
    return None  # unreachable; keeps type checkers quiet


def load_launch_config(path: str | Path, parser: argparse.ArgumentParser) -> LaunchConfig:
    """Read and validate a preset against ``parser`` without applying it."""

    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise SystemExit(f"Could not read launch config {path}: {error}") from error
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise SystemExit(f"Launch config {path} is not valid YAML: {error}") from error
    if document is None:
        document = {}
    if not isinstance(document, Mapping):
        raise SystemExit(f"Launch config {path} must be a mapping of option names to values")

    actions = _option_actions(parser)
    values: dict[str, Any] = {}
    for raw_key, raw_value in document.items():
        key = _normalize_key(raw_key, path)
        if key in CONFIG_OPTION_DESTS:
            raise SystemExit(f"{path}: {raw_key}: a launch config cannot select another launch config")
        action = actions.get(key)
        if action is None:
            suggestion = difflib.get_close_matches(key, actions, n=1)
            hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
            raise SystemExit(f"{path}: unknown launch_og.py option {raw_key!r}{hint}")
        if key in values:
            raise SystemExit(f"{path}: option {key!r} is set more than once")
        if raw_value is None:
            continue
        values[key] = _coerce(action, raw_value, path=path, key=str(raw_key))

    for group in parser._mutually_exclusive_groups:  # noqa: SLF001
        chosen = [
            _display_flag(action)
            for action in group._group_actions  # noqa: SLF001
            if action.dest in values
        ]
        if len(chosen) > 1:
            raise SystemExit(f"{path}: {' and '.join(chosen)} cannot both be set")

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return LaunchConfig(path=path, values=values, sha256=digest)


def parse_launch_arguments(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
    *,
    default_config: str | None = DEFAULT_LAUNCH_CONFIG,
    skip_for_flags: Sequence[str] = (),
) -> tuple[argparse.Namespace, LaunchConfig | None]:
    """Parse ``argv`` with the selected preset installed as parser defaults.

    ``parser`` must already declare ``--config`` and ``--no-config``. The preset
    is discovered with a preliminary pass, validated, applied through
    ``parser.set_defaults``, and then the full parse runs so command-line values
    override it. ``default_config`` names the preset used when ``--config`` is
    absent; ``None`` means no preset unless one is requested. Presets describe
    teleoperation runs, so when any boolean flag in ``skip_for_flags`` (for
    example ``--view-only``) is present the preset is not applied at all and the
    returned config is ``None``; explicit command-line options still parse.
    """

    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", default=None)
    preliminary.add_argument("--no-config", action="store_true")
    skip_dests = []
    for index, flag in enumerate(skip_for_flags):
        dest = f"_skip_preset_{index}"
        preliminary.add_argument(flag, action="store_true", dest=dest)
        skip_dests.append(dest)
    selected, _ = preliminary.parse_known_args(argv)
    if selected.no_config and selected.config is not None:
        parser.error("--no-config cannot be combined with --config")
    skipped = any(getattr(selected, dest) for dest in skip_dests)

    config: LaunchConfig | None = None
    requested = selected.config if selected.config is not None else default_config
    if not selected.no_config and not skipped and requested is not None:
        config = load_launch_config(resolve_launch_config_path(requested), parser)
        parser.set_defaults(**config.values)

    args = parser.parse_args(argv)
    if getattr(args, "config", None) != selected.config or bool(
        getattr(args, "no_config", False)
    ) != selected.no_config:
        parser.error("--config and --no-config must be spelled out in full")

    if config is not None:
        # argparse appends command-line values to a list default; an explicit
        # command-line list replaces the preset list instead of extending it.
        for dest, preset_value in config.values.items():
            if not isinstance(preset_value, list):
                continue
            parsed = getattr(args, dest, None)
            if (
                parsed is not preset_value
                and isinstance(parsed, list)
                and parsed[: len(preset_value)] == preset_value
            ):
                setattr(args, dest, parsed[len(preset_value) :])
    return args, config


def add_launch_config_arguments(
    parser: argparse.ArgumentParser, *, default_config: str | None = DEFAULT_LAUNCH_CONFIG
) -> argparse.ArgumentParser:
    """Declare ``--config``/``--no-config`` so they appear in ``--help``."""

    presets = ", ".join(available_launch_configs()) or "none found"
    default_text = (
        f"default: {default_config}" if default_config is not None else "default: none"
    )
    parser.add_argument(
        "--config",
        metavar="NAME|PATH",
        default=None,
        help=(
            "YAML launch preset applied as defaults; explicit flags override it. "
            f"A bare name selects {LAUNCH_CONFIG_ROOT}/<name>.yaml (presets: {presets}); "
            f"otherwise a YAML file path ({default_text})"
        ),
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Ignore launch presets and start from the built-in option defaults",
    )
    return parser
