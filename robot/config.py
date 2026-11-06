"""Load config/stack/*.yaml into typed objects.

Every tunable lives in yaml. ``load_config()`` reads all files in the config
directory; each file becomes a section (``cfg.section("control")``). The joint
table and the actuator model get typed dataclasses because the HAL, safety layer
and RL env all depend on them.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_DIR = os.path.join(REPO_ROOT, "config", "stack")


class ConfigError(ValueError):
    """Raised when a config file is missing a key or has an invalid value."""


@dataclass(frozen=True)
class Joint:
    """One actuated joint. Angles in radians, joint (sim) convention."""

    name: str
    board: int
    channel: int
    sign: int
    min_rad: float
    max_rad: float
    default_rad: float
    offset_deg: float


@dataclass
class JointTable:
    """Ordered joint list with lookups. The order is the order of the yaml file."""

    joints: List[Joint]

    def __post_init__(self) -> None:
        self.by_name: Dict[str, Joint] = {j.name: j for j in self.joints}
        seen = set()
        for j in self.joints:
            key = (j.board, j.channel)
            if key in seen:
                raise ConfigError(f"board 0x{j.board:02X} channel {j.channel} used twice")
            seen.add(key)

    @property
    def names(self) -> List[str]:
        return [j.name for j in self.joints]

    def __len__(self) -> int:
        return len(self.joints)

    def __iter__(self):
        return iter(self.joints)

    def __getitem__(self, name: str) -> Joint:
        return self.by_name[name]

    def default_pose(self) -> Dict[str, float]:
        """Default (stand) pose, joint name -> rad."""
        return {j.name: j.default_rad for j in self.joints}

    def mirror(self, name: str) -> Optional[str]:
        """Name of the left/right partner joint (R_knee <-> L_knee), if any."""
        if name.startswith("R_"):
            other = "L_" + name[2:]
        elif name.startswith("L_"):
            other = "R_" + name[2:]
        else:
            return None
        return other if other in self.by_name else None


@dataclass(frozen=True)
class ActuatorParams:
    """MG996R model parameters (SI units)."""

    kp: float
    kv: float
    torque_limit: float
    max_speed: float
    deadband: float
    backlash: float
    latency_range: Tuple[float, float]
    lag_tau: float
    armature: float
    damping: float
    frictionloss: float

    @property
    def latency_mean(self) -> float:
        return 0.5 * (self.latency_range[0] + self.latency_range[1])


@dataclass
class Config:
    """All config sections plus the typed joint table and actuator model."""

    config_dir: str
    sections: Dict[str, Dict[str, Any]]
    joints: JointTable
    actuator: ActuatorParams
    root: str = REPO_ROOT
    offsets_deg: Dict[int, float] = field(default_factory=dict)

    def section(self, name: str) -> Dict[str, Any]:
        """A whole yaml file as a dict ({} if the file does not exist)."""
        return self.sections.get(name, {})

    def get(self, dotted: str, default: Any = None) -> Any:
        """Look up ``"control.loop.rate_hz"`` style keys."""
        node: Any = self.sections
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        """Like get() but raises ConfigError when the key is missing."""
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise ConfigError(f"missing config key: {dotted}")
        return value

    def path(self, rel: str) -> str:
        """Resolve a repo-relative path from the config."""
        return rel if os.path.isabs(rel) else os.path.join(self.root, rel)


def load_offsets(path: str) -> Dict[int, float]:
    """Read offsets.json ({"6": -3.0, ...}) as channel -> degrees. Missing file = {}."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {int(k): float(v) for k, v in raw.items()}


def _parse_joints(robot: Dict[str, Any], offsets: Dict[int, float]) -> JointTable:
    if "joints" not in robot:
        raise ConfigError("robot.yaml has no 'joints' table")
    primary = int(robot["boards"][0]["address"])
    joints = []
    for name, j in robot["joints"].items():
        try:
            board = int(j["board"])
            channel = int(j["channel"])
            sign = int(j.get("sign", 1))
            lo, hi = float(j["min_deg"]), float(j["max_deg"])
        except KeyError as e:
            raise ConfigError(f"joint {name}: missing {e}") from None
        if sign not in (1, -1):
            raise ConfigError(f"joint {name}: sign must be +1 or -1, got {sign}")
        if lo >= hi:
            raise ConfigError(f"joint {name}: min_deg must be < max_deg")
        if "offset_deg" in j:
            offset = float(j["offset_deg"])
        elif board == primary:
            offset = offsets.get(channel, 0.0)
        else:
            offset = 0.0
        joints.append(Joint(
            name=name, board=board, channel=channel, sign=sign,
            min_rad=math.radians(lo), max_rad=math.radians(hi),
            default_rad=math.radians(float(j.get("default_deg", 0.0))),
            offset_deg=offset,
        ))
    return JointTable(joints)


def _parse_actuator(act: Dict[str, Any]) -> ActuatorParams:
    m = act.get("mg996r")
    if m is None:
        raise ConfigError("actuator.yaml has no 'mg996r' section")
    lat = m["latency_ms"]
    return ActuatorParams(
        kp=float(m["kp"]), kv=float(m["kv"]), torque_limit=float(m["torque_limit"]),
        max_speed=float(m["max_speed"]),
        deadband=math.radians(float(m["deadband_deg"])),
        backlash=math.radians(float(m["backlash_deg"])),
        latency_range=(float(lat[0]) / 1000.0, float(lat[1]) / 1000.0),
        lag_tau=float(m["lag_tau_ms"]) / 1000.0,
        armature=float(m["armature"]), damping=float(m["damping"]),
        frictionloss=float(m["frictionloss"]),
    )


def load_config(config_dir: Optional[str] = None) -> Config:
    """Load every ``*.yaml`` in ``config_dir`` (default config/stack)."""
    config_dir = config_dir or DEFAULT_CONFIG_DIR
    if not os.path.isdir(config_dir):
        raise ConfigError(f"config dir not found: {config_dir}")
    sections: Dict[str, Dict[str, Any]] = {}
    for fname in sorted(os.listdir(config_dir)):
        if fname.endswith((".yaml", ".yml")):
            with open(os.path.join(config_dir, fname)) as f:
                sections[os.path.splitext(fname)[0]] = yaml.safe_load(f) or {}
    for required in ("robot", "control", "actuator"):
        if required not in sections:
            raise ConfigError(f"missing {required}.yaml in {config_dir}")
    robot = sections["robot"]
    offsets_path = robot.get("offsets_file", "data/calibration/offsets.json")
    if not os.path.isabs(offsets_path):
        offsets_path = os.path.join(REPO_ROOT, offsets_path)
    offsets = load_offsets(offsets_path)
    return Config(
        config_dir=config_dir,
        sections=sections,
        joints=_parse_joints(robot, offsets),
        actuator=_parse_actuator(sections["actuator"]),
        offsets_deg=offsets,
    )
