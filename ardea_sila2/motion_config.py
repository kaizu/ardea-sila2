"""Motion / calibration configuration for the Ardea SiLA2 server.

Kept **separate** from the operational ``config.toml`` (server/IP/ports) on purpose
(see orchestration_design.md §4.4): these values describe physical motion and
calibration, so a wrong value risks a physical collision. They live in their own
``motion.toml`` supplied via ``--motion-config`` and are edited only by developers.

Step 1 covers the named robot poses used by ``RobotPoseService``:

- ``[base_pose]``    — the base pose (home/origin), reference for ``IsAtBasePose``
- ``[retract_pose]`` — the retract pose, reference for ``IsAtRetractPose``
- ``[inverse_base_pose]``    — the base pose turned 180° (J1 flipped); the arm faces
  the opposite direction. Reference for ``IsAtInverseBasePose``.
- ``[inverse_retract_pose]`` — the retract pose turned 180°. Reference for
  ``IsAtInverseRetractPose``.

The carriage may move when the robot is at **any** of these four poses (base,
retract, or either inverse pose); see ``MotionConfig.at_movable_pose``.

Other sections: ``[carriage]`` (travel params), ``[hand]`` (gripper params),
``[stations.<id>]`` (labware stations: position + approach/retract task pair), and
``[common].return_home`` (the shared retract->base task).
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


# Allowed values for the per-station orientation settings.
STATION_DIRECTIONS = ("forward", "reverse")  # arm facing at the station
STATION_GRIPS = ("short", "long")            # plate grip: short-edge / long-edge


@dataclass
class StationConfig:
    """A labware station: carriage position [mm] + its approach/retract task pair.

    Roles are common to all stations: script_a = approach, script_b = retract. The
    same pair serves both Pick (hand closes) and Put (hand opens).

    ``direction`` records which way the arm faces to work this station: ``forward``
    uses the normal poses (base/retract), ``reverse`` the 180°-turned inverse poses
    (inverse_base/inverse_retract). ``grip`` records the plate grip orientation
    (``short``-edge / ``long``-edge). Both are stored for now and not yet wired into
    Pick/Put behaviour (all current stations are forward + short).
    """

    position_mm: int
    script_a: str              # approach task (RunTask)
    script_b: str              # retract task (RunTask)
    direction: str = "forward"  # "forward" | "reverse" (see STATION_DIRECTIONS)
    grip: str = "short"         # "short" | "long" (see STATION_GRIPS)


@dataclass
class MotionConfig:
    base_pose: PoseConfig             # home/origin pose
    retract_pose: PoseConfig          # retract pose
    inverse_base_pose: PoseConfig     # base pose turned 180° (J1 flipped)
    inverse_retract_pose: PoseConfig  # retract pose turned 180°
    carriage: CarriageConfig = field(default_factory=CarriageConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    stations: dict[str, StationConfig] = field(default_factory=dict)
    return_home: str = "BasePosition"  # common task: retract -> base pose (requires hand open)
    # reverse counterpart: inverse retract -> inverse base pose (requires hand open)
    return_home_reverse: str = "InverseBasePosition"
    # Turn tasks used by RobotOrientationService.SetOrientation. Each ends at the
    # (inverse) retract pose regardless of the (known) starting pose.
    to_forward: str = "RetractPosition"          # turn to forward: ends at retract_pose
    to_reverse: str = "InverseRetractPosition"   # turn to reverse: ends at inverse_retract_pose

    def poses_for(self, direction: str) -> "tuple[PoseConfig, PoseConfig]":
        """Return the (base-like, retract-like) poses for a station ``direction``.

        forward -> (base_pose, retract_pose); reverse -> (inverse_base_pose,
        inverse_retract_pose). Pick/Put use these so a reverse station starts/ends at
        the 180°-turned poses instead of the normal ones.
        """
        if direction == "reverse":
            return self.inverse_base_pose, self.inverse_retract_pose
        return self.base_pose, self.retract_pose

    def return_home_for(self, direction: str) -> str:
        """Return the return-home task for a station ``direction`` (retract-like -> base-like)."""
        return self.return_home_reverse if direction == "reverse" else self.return_home

    def at_movable_pose(self, curjnt: list[float]) -> bool:
        """True if ``curjnt`` matches any pose from which the carriage may move.

        The carriage-move interlock permits motion at the base or retract pose and
        at their 180°-turned counterparts (the arm facing the opposite direction);
        at all four the arm is tucked clear of the travel envelope. Raises
        ``MotionConfigError`` (via ``PoseConfig.matches``) if ``curjnt`` is too short.
        """
        return (
            self.base_pose.matches(curjnt)
            or self.retract_pose.matches(curjnt)
            or self.inverse_base_pose.matches(curjnt)
            or self.inverse_retract_pose.matches(curjnt)
        )

    def station_at(self, position_mm: int) -> "tuple[str, StationConfig] | None":
        """Return (station_id, station) whose position matches ``position_mm``, else None.

        Station positions are unique (enforced at load), so at most one matches. This
        is how Pick/Put resolve the station from the current carriage position while
        they take no StationId argument yet.
        """
        for sid, st in self.stations.items():
            if st.position_mm == position_mm:
                return sid, st
        return None


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


def _build_stations(data: Any, carriage: CarriageConfig) -> dict[str, StationConfig]:
    if not isinstance(data, dict) or not data:
        raise MotionConfigError("At least one [stations.<id>] must be defined.")
    known = {f.name for f in dataclasses.fields(StationConfig)}
    stations: dict[str, StationConfig] = {}
    positions: dict[int, str] = {}
    for sid, sdata in data.items():
        if not isinstance(sdata, dict):
            raise MotionConfigError(f"[stations.{sid}] must be a table.")
        unknown = set(sdata) - known
        if unknown:
            raise MotionConfigError(f"Unknown key(s) in [stations.{sid}]: {', '.join(sorted(unknown))}")
        for key in ("position_mm", "script_a", "script_b"):
            if key not in sdata:
                raise MotionConfigError(f"Missing required [stations.{sid}].{key}.")
        pos = sdata["position_mm"]
        if not isinstance(pos, int) or not (carriage.range_min_mm <= pos <= carriage.range_max_mm):
            raise MotionConfigError(
                f"[stations.{sid}].position_mm must be an int within "
                f"{carriage.range_min_mm}..{carriage.range_max_mm}."
            )
        if not sdata["script_a"] or not sdata["script_b"]:
            raise MotionConfigError(f"[stations.{sid}].script_a/script_b must be non-empty.")
        direction = sdata.get("direction", "forward")
        if direction not in STATION_DIRECTIONS:
            raise MotionConfigError(
                f"[stations.{sid}].direction must be one of {STATION_DIRECTIONS} (got {direction!r})."
            )
        grip = sdata.get("grip", "short")
        if grip not in STATION_GRIPS:
            raise MotionConfigError(
                f"[stations.{sid}].grip must be one of {STATION_GRIPS} (got {grip!r})."
            )
        if pos in positions:
            raise MotionConfigError(
                f"[stations.{sid}].position_mm {pos} duplicates [stations.{positions[pos]}]; "
                "station positions must be unique (Pick/Put resolve by current carriage position)."
            )
        positions[pos] = sid
        stations[sid] = StationConfig(
            position_mm=pos, script_a=str(sdata["script_a"]), script_b=str(sdata["script_b"]),
            direction=direction, grip=grip,
        )
    return stations


def _build_common(data: Any) -> "tuple[str, str, str, str]":
    """Return (return_home, return_home_reverse, to_forward, to_reverse) from [common]."""
    data = data or {}
    if not isinstance(data, dict):
        raise MotionConfigError("[common] must be a table.")
    defaults = {"return_home": "BasePosition", "return_home_reverse": "InverseBasePosition",
                "to_forward": "RetractPosition", "to_reverse": "InverseRetractPosition"}
    unknown = set(data) - set(defaults)
    if unknown:
        raise MotionConfigError(f"Unknown key(s) in [common]: {', '.join(sorted(unknown))}")
    values = {k: str(data.get(k, v)) for k, v in defaults.items()}
    for k in defaults:
        if not values[k]:
            raise MotionConfigError(f"[common].{k} must be non-empty.")
    return (values["return_home"], values["return_home_reverse"],
            values["to_forward"], values["to_reverse"])


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
    inverse_base = _build_pose(data.get("inverse_base_pose"), "inverse_base_pose")
    inverse_retract = _build_pose(data.get("inverse_retract_pose"), "inverse_retract_pose")
    carriage = _build_carriage(data.get("carriage"))
    hand = _build_hand(data.get("hand"))
    stations = _build_stations(data.get("stations"), carriage)
    return_home, return_home_reverse, to_forward, to_reverse = _build_common(data.get("common"))
    cfg = MotionConfig(
        base_pose=base, retract_pose=retract,
        inverse_base_pose=inverse_base, inverse_retract_pose=inverse_retract,
        carriage=carriage, hand=hand, stations=stations, return_home=return_home,
        return_home_reverse=return_home_reverse, to_forward=to_forward, to_reverse=to_reverse,
    )

    logger.info(
        "Motion config loaded from %s: base_pose=%s (tol %.4f), retract_pose=%s (tol %.4f), "
        "inverse_base_pose=%s, inverse_retract_pose=%s, "
        "carriage(range=%d..%d mm), hand(closed=%d, open=%d), "
        "return_home=%s, return_home_reverse=%s, to_forward=%s, to_reverse=%s, stations=%s",
        path, base.joint_angles_deg, base.tolerance_deg,
        retract.joint_angles_deg, retract.tolerance_deg,
        inverse_base.joint_angles_deg, inverse_retract.joint_angles_deg,
        carriage.range_min_mm, carriage.range_max_mm,
        hand.closed_position, hand.open_position,
        return_home, return_home_reverse, to_forward, to_reverse,
        {sid: (s.position_mm, s.script_a, s.script_b, s.direction, s.grip)
         for sid, s in stations.items()},
    )
    return cfg
