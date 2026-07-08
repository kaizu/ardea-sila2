"""Motion / calibration configuration for the Ardea SiLA2 server.

Kept **separate** from the operational ``config.toml`` (server/IP/ports) on purpose
(see orchestration_design.md §4.4): these values describe physical motion and
calibration, so a wrong value risks a physical collision. They live in their own
``motion.toml`` supplied via ``--motion-config`` and are edited only by developers.

Step 1 covers the named robot poses used by ``RobotPoseService``:

- ``[base_pose]``    — the base pose (= 原位置), reference for ``IsAtBasePose``
- ``[retract_pose]`` — the retract pose (= 退避位置), reference for ``IsAtRetractPose``

Carriage movement is permitted when the robot is at either pose. Later steps add
``[carriage]`` / ``[stations]`` / ``[pacscripts]`` / ``[hand]`` sections here.
"""

from __future__ import annotations

import dataclasses
import logging
import tomllib
from dataclasses import dataclass
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
class MotionConfig:
    base_pose: PoseConfig     # = 原位置
    retract_pose: PoseConfig  # = 退避位置


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
    cfg = MotionConfig(base_pose=base, retract_pose=retract)

    logger.info(
        "Motion config loaded from %s: base_pose=%s (tol %.4f deg), retract_pose=%s (tol %.4f deg)",
        path, base.joint_angles_deg, base.tolerance_deg,
        retract.joint_angles_deg, retract.tolerance_deg,
    )
    return cfg
