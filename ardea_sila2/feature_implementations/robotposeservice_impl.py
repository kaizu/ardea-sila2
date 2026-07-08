"""RobotPoseService implementation (Ardea-specific).

Reads the robot's current joint angles over b-CAP (reusing ``bcap_sila2.bcap``)
and reports whether the arm is at a configured named pose — the base pose (原位置)
or the retract pose (退避位置) — by comparing the first six axes within tolerance.
Read-only: it never moves the robot. Reference poses and tolerance come from the
motion configuration (``self.parent_server.motion``); see ``motion_config``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bcap_sila2.bcap import RobotUnavailableError, get_joint_angles
from orinexception import ORiNException
from sila2.server import MetadataDict

from ..generated.robotposeservice import (
    ControllerConnectionError,
    IsAtBasePose_Responses,
    IsAtRetractPose_Responses,
    PoseDimensionMismatch,
    RobotAccessError,
    RobotPoseServiceBase,
)
from ..motion_config import MotionConfigError

if TYPE_CHECKING:
    from ..server import Server


class RobotPoseServiceImpl(RobotPoseServiceBase):
    def __init__(self, parent_server: Server) -> None:
        super().__init__(parent_server=parent_server)

    def _current_joint_angles(self) -> list[float]:
        """Read CurJnt over b-CAP, mapping failures to SiLA errors (like RobotServiceImpl)."""
        cfg = self.parent_server.config.controller
        try:
            return get_joint_angles(cfg)
        except OSError as e:
            raise ControllerConnectionError(str(e))
        except (ORiNException, RobotUnavailableError) as e:
            raise RobotAccessError(str(e))

    def IsAtBasePose(self, *, metadata: MetadataDict) -> IsAtBasePose_Responses:
        angles = self._current_joint_angles()
        try:
            ok = self.parent_server.motion.base_pose.matches(angles)
        except MotionConfigError as e:
            raise PoseDimensionMismatch(str(e))
        return IsAtBasePose_Responses(IsAtBasePose=ok)

    def IsAtRetractPose(self, *, metadata: MetadataDict) -> IsAtRetractPose_Responses:
        angles = self._current_joint_angles()
        try:
            ok = self.parent_server.motion.retract_pose.matches(angles)
        except MotionConfigError as e:
            raise PoseDimensionMismatch(str(e))
        return IsAtRetractPose_Responses(IsAtRetractPose=ok)
