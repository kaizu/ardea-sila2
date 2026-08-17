"""RobotOrientationService implementation (Ardea-specific).

Moves the DENSO arm between its four known poses by running PacScripts over b-CAP.

``SetOrientation`` turns the arm to face forward or reverse, keeping the pose family:

- ``forward`` -> RunTask(motion.to_forward)  -> ends at the retract pose
- ``reverse`` -> RunTask(motion.to_reverse)  -> ends at the inverse retract pose

``ReturnHome`` parks the arm at the base pose from any of the four, walking the steps
``MotionConfig.home_path`` returns (empty at the base pose; one task from the retract or
inverse base pose; two -- via the retract pose -- from the inverse retract pose, so that
no leg is a transition the machine has never run). It additionally requires the hand to
be fully open, which the home task assumes; that is the one PLC read in this feature.

Both commands may only run while the arm is at one of the four known poses so the motion
starts from a safe, known posture. Both hold the server OperationCoordinator for the
whole sequence so no carriage move or pick/put runs concurrently, and verify the arm
reached the expected pose afterwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bcap_sila2.bcap import (
    RobotUnavailableError,
    TaskAbnormalStopError,
    TaskTimeoutError,
    get_joint_angles,
    run_task,
)
from orinexception import ORiNException
from sila2.server import MetadataDict, ObservableCommandInstanceWithIntermediateResponses

from kvcomplus_sila2 import kvcomplus

from ..generated.robotorientationservice import (
    ControllerConnectionError,
    HandNotOpen,
    InvalidDirection,
    PlcAccessError,
    PlcConnectionError,
    PoseNotRestored,
    ReturnHome_IntermediateResponses,
    ReturnHome_Responses,
    RobotAccessError,
    RobotNotAtKnownPose,
    RobotOrientationServiceBase,
    SetOrientation_IntermediateResponses,
    SetOrientation_Responses,
    TaskAccessError,
    TaskExecutionTimeout,
)

if TYPE_CHECKING:
    from ..server import Server

_ONE_CYCLE = 1  # RunTask mode: run once and stop
DM = 18                  # device type for DM/D devices
HAND_CUR_POS = 6060      # D6060 hand current position
_HAND_OPEN_TOL = 3       # tolerance [units] for "hand fully open" (matches LabwareService)


class RobotOrientationServiceImpl(RobotOrientationServiceBase):
    def __init__(self, parent_server: Server) -> None:
        super().__init__(parent_server=parent_server)

    def _joint_angles(self) -> list[float]:
        try:
            return get_joint_angles(self.parent_server.config.controller)
        except OSError as e:
            raise ControllerConnectionError(str(e))
        except (ORiNException, RobotUnavailableError) as e:
            raise RobotAccessError(str(e))

    def _run_task(self, name: str) -> None:
        cfg = self.parent_server.config
        tcfg = cfg.task
        try:
            run_task(
                cfg.controller, name, mode=_ONE_CYCLE,
                poll_interval=tcfg.poll_interval_seconds,
                start_timeout=tcfg.start_timeout_seconds,
                completion_timeout=tcfg.completion_timeout_seconds,
            )
        except OSError as e:
            raise ControllerConnectionError(str(e))
        except TaskTimeoutError as e:
            raise TaskExecutionTimeout(str(e))
        except (ORiNException, TaskAbnormalStopError) as e:
            raise TaskAccessError(str(e))

    def _hand_position(self) -> int:
        """Read D6060 over KV COM+, mapping failures to this feature's SiLA errors."""
        plc = self.parent_server.config.plc
        try:
            return kvcomplus.read_word(plc, DM, HAND_CUR_POS)
        except kvcomplus.KvComError as e:
            msg = str(e).lower()
            if "bridge" in msg or "connect" in msg or "timed out" in msg:
                raise PlcConnectionError(str(e))
            raise PlcAccessError(str(e))

    def SetOrientation(
        self,
        Direction: str,
        *,
        metadata: MetadataDict,
        instance: ObservableCommandInstanceWithIntermediateResponses[SetOrientation_IntermediateResponses],
    ) -> SetOrientation_Responses:
        motion = self.parent_server.motion

        direction = str(Direction).strip().lower()
        if direction not in ("forward", "reverse"):
            raise InvalidDirection(f"Direction {Direction!r} is not 'forward' or 'reverse'.")

        def phase(name: str) -> None:
            instance.send_intermediate_response(SetOrientation_IntermediateResponses(Phase=name))

        # One motion at a time (shared robot/carriage OperationCoordinator).
        with self.parent_server.operation_lock:
            # Pick the target pose from the current pose family + requested facing:
            # a base-family pose stays base (base/inverse-base), a retract-family pose
            # stays retract, and only the forward/reverse facing flips.
            target = motion.orientation_target(self._joint_angles(), direction)
            if target is None:
                raise RobotNotAtKnownPose(
                    "Robot is at none of the base/retract/inverse-base/inverse-retract "
                    "poses; orientation change refused."
                )

            instance.begin_execution()
            phase(f"turning {direction}: RunTask({target.task})")
            self._run_task(target.task)

            phase("verify target pose")
            if not target.matches(self._joint_angles()):
                raise PoseNotRestored(
                    f"Robot did not reach the target pose after RunTask({target.task})."
                )

            instance.progress = 1.0
            return SetOrientation_Responses(Orientation=direction)

    def ReturnHome(
        self,
        *,
        metadata: MetadataDict,
        instance: ObservableCommandInstanceWithIntermediateResponses[ReturnHome_IntermediateResponses],
    ) -> ReturnHome_Responses:
        motion = self.parent_server.motion

        def phase(name: str) -> None:
            instance.send_intermediate_response(ReturnHome_IntermediateResponses(Phase=name))

        # One motion at a time (shared robot/carriage OperationCoordinator).
        with self.parent_server.operation_lock:
            steps = motion.home_path(self._joint_angles())
            if steps is None:
                raise RobotNotAtKnownPose(
                    "Robot is at none of the base/retract/inverse-base/inverse-retract "
                    "poses; return-home refused."
                )

            # The home task assumes an open hand, and parking with a labware held is not
            # intended -- checked even when already home, so the contract is the same
            # whichever pose the arm is at.
            hand_pos = self._hand_position()
            open_pos = motion.hand.open_position
            if abs(hand_pos - open_pos) > _HAND_OPEN_TOL:
                raise HandNotOpen(
                    f"Hand is at {hand_pos} (open={open_pos}); must be fully open to return home."
                )

            instance.begin_execution()
            if not steps:
                phase("already at the base pose")
                instance.progress = 1.0
                return ReturnHome_Responses(AtBasePose=True)

            for task, target, name in steps:
                phase(f"to {name}: RunTask({task})")
                self._run_task(task)
                phase(f"verify {name} pose")
                if not target.matches(self._joint_angles()):
                    raise PoseNotRestored(
                        f"Robot did not reach the {name} pose after RunTask({task})."
                    )

            instance.progress = 1.0
            return ReturnHome_Responses(AtBasePose=True)
