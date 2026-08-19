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
``[stations.<id>]`` (labware stations: position + approach/retract task pair; the rail is
served from both sides, so two stations may share a position when their ``direction``
differs -- ``(position_mm, direction)`` is the unique key), and
``[common].return_home`` (the shared retract->base task). Since PutLabware stops at the
retract pose, ``return_home`` is no longer used by Pick/Put: it is what
RobotOrientationService.ReturnHome runs (see ``home_path``). ``return_home_reverse`` is
currently unused -- no path home goes through the inverse base pose.
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
    """A named robot pose: reference joint angles [deg], match tolerance [deg], and
    the PacScript (``task``) that drives the arm to this pose from any known pose.

    ``task`` is used by RobotOrientationService.SetOrientation to reach this pose.
    """

    joint_angles_deg: list[float]
    tolerance_deg: float = 0.01
    task: str = ""  # PacScript that moves the arm to this pose (RunTask)

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
    """Hand (gripper) motion parameters. Grip force is fixed per the hand doc.

    The close target depends on the station's grip orientation: ``closed_position``
    for a short-edge grip (fully closed) and ``closed_position_long`` for a long-edge
    grip (the jaws need not close as far). See ``closed_position_for``.
    """

    closed_position: int = 0         # D5050 close target for short-edge grip (fully closed)
    closed_position_long: int = 110  # D5050 close target for long-edge grip
    open_position: int = 140         # D5050 fully open
    speed: int = 0                   # D5060 0=slowest .. 255=fastest
    grip_force: int = 1              # D5070 fixed
    move_timeout_s: float = 30.0

    def closed_position_for(self, grip: str) -> int:
        """Close (chuck) target for a station's grip: long -> closed_position_long, else closed_position."""
        return self.closed_position_long if grip == "long" else self.closed_position


# Allowed values for the per-station orientation settings.
STATION_DIRECTIONS = ("forward", "reverse")  # arm facing at the station
STATION_GRIPS = ("short", "long")            # plate grip: short-edge / long-edge
# Per-station task pairs, required in every [stations.<id>]. Pick and Put are kept
# separate so a station can place along a different trajectory than it takes.
_STATION_SCRIPT_KEYS = ("pick_script_a", "pick_script_b", "put_script_a", "put_script_b")


@dataclass
class StationConfig:
    """A labware station: carriage position [mm] + its approach/retract task pairs.

    Pick and Put take **separate** task pairs (``pick_script_a``/``pick_script_b`` and
    ``put_script_a``/``put_script_b``; ``_a`` = approach, ``_b`` = retract). They may
    name the same tasks -- and every station currently does -- but keeping them apart
    lets a station place a plate along a different trajectory than it takes one,
    configured in the TOML without touching code.

    ``direction`` is which way the arm faces to work this station: ``forward`` uses the
    normal poses (base/retract), ``reverse`` the 180°-turned inverse poses
    (inverse_base/inverse_retract). ``grip`` is the plate grip orientation
    (``short``-edge / ``long``-edge) and selects Pick's chuck target.
    """

    position_mm: int
    pick_script_a: str          # Pick: approach task (RunTask)
    pick_script_b: str          # Pick: retract task (RunTask)
    put_script_a: str           # Put: approach task (RunTask)
    put_script_b: str           # Put: retract task (RunTask)
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
    # Common task: retract -> base pose (requires hand open). Not used by Pick/Put any
    # more -- PutLabware stops at the retract pose -- but used by ReturnHome (home_path).
    return_home: str = "BasePosition"
    # reverse counterpart: inverse retract -> inverse base pose (requires hand open)
    return_home_reverse: str = "InverseBasePosition"

    def orientation_target(self, curjnt: list[float], direction: str) -> "PoseConfig | None":
        """Return the pose SetOrientation should move to, or None if not at a known pose.

        The pose *family* (base vs retract) is preserved and only the facing flips:
        from a base-family pose (base/inverse-base) -> base_pose (forward) or
        inverse_base_pose (reverse); from a retract-family pose -> retract_pose or
        inverse_retract_pose. Raises ``MotionConfigError`` via ``matches`` if curjnt
        is too short.
        """
        if self.base_pose.matches(curjnt) or self.inverse_base_pose.matches(curjnt):
            return self.base_pose if direction == "forward" else self.inverse_base_pose
        if self.retract_pose.matches(curjnt) or self.inverse_retract_pose.matches(curjnt):
            return self.retract_pose if direction == "forward" else self.inverse_retract_pose
        return None

    def home_path(self, curjnt: list[float]) -> "list[tuple[str, PoseConfig, str]] | None":
        """Steps that park the arm at the base pose: (task, pose reached, its name).

        ``[]`` means it is already there; ``None`` that it is at no known pose.
        Used by RobotOrientationService.ReturnHome. Every leg is a transition that has
        been run on the real machine:
        - retract -> base is the old return-home task (``[common].return_home``),
        - inverse base -> base is the same task SetOrientation's forward turn uses,
        - inverse retract goes home **via the retract pose** rather than in one move:
          both of its legs are proven, whereas a direct inverse-retract -> base task
          would change pose family and facing at once and has never been run.
        (``return_home_reverse`` is therefore still unused; reaching the inverse base
        pose is not on any path home.)
        """
        if self.base_pose.matches(curjnt):
            return []
        if self.retract_pose.matches(curjnt):
            return [(self.return_home, self.base_pose, "base")]
        if self.inverse_base_pose.matches(curjnt):
            return [(self.base_pose.task, self.base_pose, "base")]
        if self.inverse_retract_pose.matches(curjnt):
            return [
                (self.retract_pose.task, self.retract_pose, "retract"),
                (self.return_home, self.base_pose, "base"),
            ]
        return None

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
        """Return the return-home task for a station ``direction`` (retract-like -> base-like).

        No caller left in Pick/Put (both end at the retract pose); kept for the explicit
        park/return-home command still to be written.
        """
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

    def facing_of(self, curjnt: list[float]) -> "str | None":
        """Which way the arm faces at its current pose: "forward", "reverse", or None.

        None means it is at no known pose. Both pose families count: the base and retract
        poses face forward, their 180°-turned counterparts reverse.
        """
        if self.base_pose.matches(curjnt) or self.retract_pose.matches(curjnt):
            return "forward"
        if self.inverse_base_pose.matches(curjnt) or self.inverse_retract_pose.matches(curjnt):
            return "reverse"
        return None

    def stations_at(self, position_mm: int) -> "list[tuple[str, StationConfig]]":
        """Every station at ``position_mm``, in configuration order.

        The rail has stations on both sides ("駆動側" / "従動側"), so one carriage
        position can serve two of them -- one per facing.
        """
        return [(sid, st) for sid, st in self.stations.items() if st.position_mm == position_mm]

    def station_at(
        self, position_mm: int, direction: "str | None" = None
    ) -> "tuple[str, StationConfig] | None":
        """Return the station at ``position_mm`` facing ``direction``, else None.

        ``(position_mm, direction)`` is unique (enforced at load), so at most one matches.
        This is how Pick/Put resolve the station while they take no StationId argument:
        the carriage position says where, and the arm's current facing says which side.
        With ``direction`` omitted, only an unambiguous position resolves -- two stations
        there return None rather than a guess.
        """
        here = self.stations_at(position_mm)
        if direction is None:
            return here[0] if len(here) == 1 else None
        for sid, st in here:
            if st.direction == direction:
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
    task = data.get("task", "")
    if not isinstance(task, str) or not task:
        raise MotionConfigError(
            f"[{section}].task must be a non-empty string (the PacScript that reaches this pose)."
        )
    return PoseConfig(joint_angles_deg=[float(x) for x in angles], tolerance_deg=float(tol), task=task)


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
    if not (0 <= h.closed_position_long <= h.open_position):
        raise MotionConfigError("[hand] must satisfy 0 <= closed_position_long <= open_position.")
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
    positions: dict[tuple[int, str], str] = {}   # (position_mm, direction) -> station id
    for sid, sdata in data.items():
        if not isinstance(sdata, dict):
            raise MotionConfigError(f"[stations.{sid}] must be a table.")
        # Pick/Put used to share one script_a/script_b pair. Name the split explicitly
        # rather than letting it fall through as a generic "unknown key".
        legacy = {"script_a", "script_b"} & set(sdata)
        if legacy:
            raise MotionConfigError(
                f"[stations.{sid}] uses the old shared key(s) {', '.join(sorted(legacy))}; "
                "Pick and Put now take separate pairs -- use pick_script_a/pick_script_b "
                "and put_script_a/put_script_b (they may name the same tasks)."
            )
        unknown = set(sdata) - known
        if unknown:
            raise MotionConfigError(f"Unknown key(s) in [stations.{sid}]: {', '.join(sorted(unknown))}")
        for key in ("position_mm", *_STATION_SCRIPT_KEYS):
            if key not in sdata:
                raise MotionConfigError(f"Missing required [stations.{sid}].{key}.")
        pos = sdata["position_mm"]
        if not isinstance(pos, int) or not (carriage.range_min_mm <= pos <= carriage.range_max_mm):
            raise MotionConfigError(
                f"[stations.{sid}].position_mm must be an int within "
                f"{carriage.range_min_mm}..{carriage.range_max_mm}."
            )
        empty = [k for k in _STATION_SCRIPT_KEYS if not sdata[k]]
        if empty:
            raise MotionConfigError(f"[stations.{sid}].{'/'.join(empty)} must be non-empty.")
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
        # The rail is served from both sides, so a position may repeat as long as the two
        # stations face opposite ways: Pick/Put resolve one by position **and** the arm's
        # current facing, so (position, direction) is what has to be unique.
        if (pos, direction) in positions:
            raise MotionConfigError(
                f"[stations.{sid}] duplicates [stations.{positions[(pos, direction)]}]: both are "
                f"at {pos} mm facing {direction}. A position may be shared only by stations "
                "facing opposite ways (Pick/Put resolve by position plus the arm's facing)."
            )
        positions[(pos, direction)] = sid
        stations[sid] = StationConfig(
            position_mm=pos,
            pick_script_a=str(sdata["pick_script_a"]),
            pick_script_b=str(sdata["pick_script_b"]),
            put_script_a=str(sdata["put_script_a"]),
            put_script_b=str(sdata["put_script_b"]),
            direction=direction, grip=grip,
        )
    return stations


def _build_common(data: Any) -> "tuple[str, str]":
    """Return (return_home, return_home_reverse) task names from [common]."""
    data = data or {}
    if not isinstance(data, dict):
        raise MotionConfigError("[common] must be a table.")
    defaults = {"return_home": "BasePosition", "return_home_reverse": "InverseBasePosition"}
    unknown = set(data) - set(defaults)
    if unknown:
        raise MotionConfigError(f"Unknown key(s) in [common]: {', '.join(sorted(unknown))}")
    values = {k: str(data.get(k, v)) for k, v in defaults.items()}
    for k in defaults:
        if not values[k]:
            raise MotionConfigError(f"[common].{k} must be non-empty.")
    return values["return_home"], values["return_home_reverse"]


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
    return_home, return_home_reverse = _build_common(data.get("common"))
    cfg = MotionConfig(
        base_pose=base, retract_pose=retract,
        inverse_base_pose=inverse_base, inverse_retract_pose=inverse_retract,
        carriage=carriage, hand=hand, stations=stations, return_home=return_home,
        return_home_reverse=return_home_reverse,
    )

    logger.info(
        "Motion config loaded from %s: base_pose=%s (tol %.4f, task=%s), "
        "retract_pose=%s (tol %.4f, task=%s), inverse_base_pose=%s (task=%s), "
        "inverse_retract_pose=%s (task=%s), carriage(range=%d..%d mm), hand(closed=%d, open=%d), "
        "return_home=%s, return_home_reverse=%s, stations=%s",
        path, base.joint_angles_deg, base.tolerance_deg, base.task,
        retract.joint_angles_deg, retract.tolerance_deg, retract.task,
        inverse_base.joint_angles_deg, inverse_base.task,
        inverse_retract.joint_angles_deg, inverse_retract.task,
        carriage.range_min_mm, carriage.range_max_mm,
        hand.closed_position, hand.open_position,
        return_home, return_home_reverse,
        {sid: (s.position_mm, (s.pick_script_a, s.pick_script_b),
               (s.put_script_a, s.put_script_b), s.direction, s.grip)
         for sid, s in stations.items()},
    )
    return cfg
