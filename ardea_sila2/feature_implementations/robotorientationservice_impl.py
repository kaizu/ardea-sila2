"""RobotOrientationService implementation (Ardea-specific).

Moves the DENSO arm between its four known poses.

``SetOrientation`` turns the arm to face forward or reverse, keeping the pose family:

- ``forward`` -> the retract pose (or the base pose, from a base-family pose)
- ``reverse`` -> the 180°-turned counterpart of whichever family the arm is in

``ReturnHome`` parks the arm at the base pose from any of the four, and additionally
requires the hand to be fully open, which the home task assumes.

Both may only run while the arm is at one of the four known poses so the motion starts
from a safe, known posture, and both hold the server OperationCoordinator so no carriage
move or pick/put runs concurrently. The motions themselves live in
:mod:`ardea_sila2.ops` (shared with LabwareService.Transfer, which turns the arm as part
of a route); this module maps that module's neutral exceptions onto the errors this
feature declares.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Type

from sila2.server import MetadataDict, ObservableCommandInstanceWithIntermediateResponses

from .. import ops
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

_OP_ERRORS: dict[Type[ops.OpError], Type[Exception]] = {
    ops.PlcConnectionFailed: PlcConnectionError,
    ops.PlcAccessFailed: PlcAccessError,
    ops.ControllerConnectionFailed: ControllerConnectionError,
    ops.RobotAccessFailed: RobotAccessError,
    ops.TaskTimedOut: TaskExecutionTimeout,
    ops.TaskFailed: TaskAccessError,
    ops.PoseNotReached: PoseNotRestored,
    ops.NotAtKnownPose: RobotNotAtKnownPose,
    ops.HandNotOpenError: HandNotOpen,
}


@contextmanager
def _sila_errors() -> Iterator[None]:
    """Translate ops exceptions into this feature's SiLA errors."""
    try:
        yield
    except ops.OpError as e:
        cls = _OP_ERRORS.get(type(e))
        if cls is None:
            raise
        raise cls(str(e)) from e


class RobotOrientationServiceImpl(RobotOrientationServiceBase):
    def __init__(self, parent_server: Server) -> None:
        super().__init__(parent_server=parent_server)

    @property
    def _ops(self) -> ops.MotionOps:
        return self.parent_server.ops

    def SetOrientation(
        self,
        Direction: str,
        *,
        metadata: MetadataDict,
        instance: ObservableCommandInstanceWithIntermediateResponses[SetOrientation_IntermediateResponses],
    ) -> SetOrientation_Responses:
        direction = str(Direction).strip().lower()
        if direction not in ("forward", "reverse"):
            raise InvalidDirection(f"Direction {Direction!r} is not 'forward' or 'reverse'.")

        def phase(name: str) -> None:
            instance.send_intermediate_response(SetOrientation_IntermediateResponses(Phase=name))

        # One motion at a time (shared robot/carriage OperationCoordinator).
        with self.parent_server.operation_lock, _sila_errors():
            instance.begin_execution()
            # The pose family is preserved and only the facing flips: a base-family pose
            # stays base, a retract-family pose stays retract.
            self._ops.set_orientation(direction, phase)
            instance.progress = 1.0
            return SetOrientation_Responses(Orientation=direction)

    def ReturnHome(
        self,
        *,
        metadata: MetadataDict,
        instance: ObservableCommandInstanceWithIntermediateResponses[ReturnHome_IntermediateResponses],
    ) -> ReturnHome_Responses:
        def phase(name: str) -> None:
            instance.send_intermediate_response(ReturnHome_IntermediateResponses(Phase=name))

        with self.parent_server.operation_lock, _sila_errors():
            instance.begin_execution()
            # The home task assumes an open hand, and parking with a labware held is not
            # intended -- checked even when already home, so the contract is the same
            # whichever pose the arm is at.
            self._ops.require_hand_open("must be fully open to return home")
            self._ops.return_home(phase)
            instance.progress = 1.0
            return ReturnHome_Responses(AtBasePose=True)
