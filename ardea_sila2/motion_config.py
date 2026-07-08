"""Motion / calibration configuration for the Ardea SiLA2 server.

Kept **separate** from the operational ``config.toml`` (server/IP/ports) on purpose
(see orchestration_design.md §4.4): these values describe physical motion and
calibration, so a wrong value risks a physical collision. They live in their own
``motion.toml`` supplied via ``--motion-config`` and are edited only by developers.

Step 1 covers the named robot poses used by ``RobotPoseService``:

- ``[base_pose]``    — the base pose (home/origin), reference for ``IsAtBasePose``
- ``[retract_pose]`` — the retract pose, reference for ``IsAtRetractPose``

Carriage movement is permitted when the robot is at either pose. Later steps add
``[carriage]`` / ``[stations]`` / ``[pacscripts]`` / ``[hand]`` sections here.
"""

from __future__ import annotations

import dataclasses
import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Number of leading joint axes compared for pose matching (J1..J6). CurJnt also
# reports trailing auxiliary axes (0.0 on this robot); those are ignored.
COMPARED_AXES = 6


class MotionConfigError(Exception):
    """Raised when the motion configuration is missing, invalid, or incomplete."""


@dataclass
class PoseConfig:
    """A named robot pose: reference joint angles [deg] + match tolerance [deg]."""

    joint_angles_deg: list[float]
    tolerance_deg: float = 0.01

    def matches(self, curjnt: list[float]) -> bool:
        """True if ``curjnt`` matches this pose within tolerance on the first 6 axes.

        Raises ``MotionConfigError`` if ``curjnt`` has fewer than 6 elements.
        """
        if len(curjnt) < COMPARED_AXES:
            raise MotionConfigError(
                f"CurJnt has {len(curjnt)} axes; need at least {COMPARED_AXES} to compare."
            )
        return all(
            abs(curjnt[i] - self.joint_angles_deg[i]) <= self.tolerance_deg
            for i in range(COMPARED_AXES)
        )


@dataclass
class CarriageConfig:
    """Travel-carriage motion parameters (fixed; not exposed as command args)."""

    default_speed_mm_s: int = 50     # positioning speed [mm/s]
    accel_mm_s_ms: int = 1           # accel/decel [mm/s/ms]
    range_min_mm: int = 0            # lower travel bound [mm]
    range_max_mm: int = 2600         # upper travel bound [mm]
    move_timeout_s: float = 60.0     # max wait for a move to complete [s]
    poll_interval_s: float = 0.2     # status/position poll interval [s]


@dataclass
class HandConfig:
    """Hand (gripper) motion parameters. Grip force is fixed per the hand doc."""

    closed_position: int = 0     # D5050 fully closed
    open_position: int = 140     # D5050 fully open
    speed: int = 0               # D5060 0=slowest .. 255=fastest
    grip_force: int = 1          # D5070 fixed
    move_timeout_s: float = 30.0


@dataclass
class PacScriptConfig:
    """DENSO task (PAC program) names run via b-CAP TaskService.RunTask."""

    pick_approach: str = "PickUp0"   # approach / grab position
    pick_retract: str = "PickUp1"    # retract -> ends at the retract pose
    put_approach: str = "PickUp0"    # approach (Put currently reuses the same tasks)
    put_retract: str = "PickUp1"     # retract -> ends at the retract pose


@dataclass
class MotionConfig:
    base_pose: PoseConfig     # home/origin pose
    retract_pose: PoseConfig  # retract pose
    carriage: CarriageConfig = field(default_factory=CarriageConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    pacscripts: PacScriptConfig = field(default_factory=PacScriptConfig)


def _build_pose(data: Any, section: str) -> PoseConfig:
    if not isinstance(data, dict):
        raise MotionConfigError(f"Missing required [{section}] section in motion config.")
    known = {f.name for f in dataclasses.fields(PoseConfig)}
    unknown = set(data) - known
    if unknown:
        raise MotionConfigError(f"Unknown key(s) in [{section}]: {', '.join(sorted(unknown))}")
    if "joint_angles_deg" not in data:
        raise MotionConfigError(f"Missing required [{section}].joint_angles_deg.")
    angles = data["joint_angles_deg"]
    if (not isinstance(angles, list) or len(angles) != COMPARED_AXES
            or not all(isinstance(x, (int, float)) for x in angles)):
        raise MotionConfigError(
            f"[{section}].joint_angles_deg must be a list of {COMPARED_AXES} numbers (J1..J6)."
        )
    tol = data.get("tolerance_deg", 0.01)
    if not isinstance(tol, (int, float)) or tol <= 0:
        raise MotionConfigError(f"[{section}].tolerance_deg must be a number > 0.")
    return PoseConfig(joint_angles_deg=[float(x) for x in angles], tolerance_deg=float(tol))


def _build_carriage(data: Any) -> CarriageConfig:
    if data is None:
        return CarriageConfig()
    if not isinstance(data, dict):
        raise MotionConfigError("[carriage] must be a table.")
    known = {f.name for f in dataclasses.fields(CarriageConfig)}
    unknown = set(data) - known
    if unknown:
        raise MotionConfigError(f"Unknown key(s) in [carriage]: {', '.join(sorted(unknown))}")
    c = CarriageConfig(**{**dataclasses.asdict(CarriageConfig()), **data})
    if not (0 <= c.range_min_mm < c.range_max_mm):
        raise MotionConfigError(
            f"[carriage] must satisfy 0 <= range_min_mm ({c.range_min_mm}) < range_max_mm ({c.range_max_mm})."
        )
    if c.default_speed_mm_s <= 0 or c.accel_mm_s_ms <= 0:
        raise MotionConfigError("[carriage] default_speed_mm_s and accel_mm_s_ms must be > 0.")
    if c.move_timeout_s <= 0 or c.poll_interval_s <= 0:
        raise MotionConfigError("[carriage] move_timeout_s and poll_interval_s must be > 0.")
    return c


def _build_hand(data: Any) -> HandConfig:
    if data is None:
        return HandConfig()
    if not isinstance(data, dict):
        raise MotionConfigError("[hand] must be a table.")
    known = {f.name for f in dataclasses.fields(HandConfig)}
    unknown = set(data) - known
    if unknown:
        raise MotionConfigError(f"Unknown key(s) in [hand]: {', '.join(sorted(unknown))}")
    h = HandConfig(**{**dataclasses.asdict(HandConfig()), **data})
    if not (0 <= h.closed_position <= h.open_position):
        raise MotionConfigError("[hand] must satisfy 0 <= closed_position <= open_position.")
    if not (0 <= h.speed <= 255):
        raise MotionConfigError("[hand] speed must be 0..255.")
    if h.move_timeout_s <= 0:
        raise MotionConfigError("[hand] move_timeout_s must be > 0.")
    return h


def _build_pacscripts(data: Any) -> PacScriptConfig:
    if data is None:
        return PacScriptConfig()
    if not isinstance(data, dict):
        raise MotionConfigError("[pacscripts] must be a table.")
    known = {f.name for f in dataclasses.fields(PacScriptConfig)}
    unknown = set(data) - known
    if unknown:
        raise MotionConfigError(f"Unknown key(s) in [pacscripts]: {', '.join(sorted(unknown))}")
    p = PacScriptConfig(**{**dataclasses.asdict(PacScriptConfig()), **data})
    if not all((p.pick_approach, p.pick_retract, p.put_approach, p.put_retract)):
        raise MotionConfigError(
            "[pacscripts] pick_approach/pick_retract/put_approach/put_retract must be non-empty."
        )
    return p


def load_motion_config(path: str | Path) -> MotionConfig:
    """Load and validate the motion configuration from a TOML file.

    Logs the loaded values at INFO level so a mistaken calibration is easy to spot
    at startup (orchestration_design.md §4.4).
    """
    path = Path(path)
    if not path.is_file():
        raise MotionConfigError(f"Motion configuration file not found: {path}")

    with path.open("rb") as f:
        data: dict[str, Any] = tomllib.load(f)

    base = _build_pose(data.get("base_pose"), "base_pose")
    retract = _build_pose(data.get("retract_pose"), "retract_pose")
    carriage = _build_carriage(data.get("carriage"))
    hand = _build_hand(data.get("hand"))
    pacscripts = _build_pacscripts(data.get("pacscripts"))
    cfg = MotionConfig(
        base_pose=base, retract_pose=retract, carriage=carriage, hand=hand, pacscripts=pacscripts
    )

    logger.info(
        "Motion config loaded from %s: base_pose=%s (tol %.4f deg), retract_pose=%s (tol %.4f deg), "
        "carriage(speed=%d mm/s, accel=%d, range=%d..%d mm, move_timeout=%.1fs, poll=%.2fs), "
        "hand(closed=%d, open=%d, speed=%d, force=%d), pacscripts(pick=%s/%s)",
        path, base.joint_angles_deg, base.tolerance_deg,
        retract.joint_angles_deg, retract.tolerance_deg,
        carriage.default_speed_mm_s, carriage.accel_mm_s_ms,
        carriage.range_min_mm, carriage.range_max_mm,
        carriage.move_timeout_s, carriage.poll_interval_s,
        hand.closed_position, hand.open_position, hand.speed, hand.grip_force,
        pacscripts.pick_approach, pacscripts.pick_retract,
    )
    return cfg
