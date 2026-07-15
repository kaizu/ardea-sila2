"""RobotOrientationService implementation (Ardea-specific): SetOrientation.

Turns the DENSO arm to face forward or reverse by running a dedicated turn
PacScript over b-CAP:

- ``forward`` -> RunTask(motion.to_forward)  -> ends at the retract pose
- ``reverse`` -> RunTask(motion.to_reverse)  -> ends at the inverse retract pose

The command may only run while the arm is at one of the four known poses (base,
retract, inverse base, inverse retract) so the 180° turn starts from a safe,
known posture. It holds the server OperationCoordinator for the whole turn so no
carriage move or pick/put runs concurrently, and verifies the arm reached the
expected pose afterwards.
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

from ..generated.robotorientationservice import (
    ControllerConnectionError,
    InvalidDirection,
    PoseNotRestored,
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
