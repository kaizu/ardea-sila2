"""LabwareService implementation (Ardea-specific).

Labware handling plus the two actuator utilities that had no home of their own:

  PickLabware  starts at the station direction's base **or** retract pose (from the base
               pose the arm moves to the retract pose first, which is where the approach
               task runs from) and ends at the retract pose.
  PutLabware   starts and ends at the retract pose. It deliberately does **not** return
               home: a pick following a put would only have to undo that move.
  Transfer     drives a whole route -- carriage to the source, face it, pick, carriage to
               the destination, face it, put -- so the carriage need not start at the
               source station and the client needs one call instead of four.
  MoveHand / ActivateHand   drive the gripper on its own (device units; speed and force
               come from the motion config).
  ToggleLight / LightIsOn   switch the machine light, which is a boolean variable on the
               **robot controller** (b-CAP), not a PLC signal.

**The physical work lives in** :mod:`ardea_sila2.ops`. This module holds the server's
OperationCoordinator for the whole command, calls the operations, and translates their
neutral exceptions into the errors this feature declares. That split is what lets
Transfer perform a carriage move, a turn, a pick and a put in one command: calling the
other features' commands would deadlock on the non-reentrant lock.

Grasp verification (only meaningful when gripping, i.e. at Pick time):
- Pick uses the grip bit D6002.6 sampled at the chuck's completion instant: 0 = the jaws
  stopped short (an object is held), 1 = they reached the commanded close (empty). This
  is robust for both short- and long-edge grips and does not depend on the hand springing
  back to open, which on the real hand it does not reliably do.
- Put only sanity-checks, before opening, that the hand is not (near) fully open. That
  stays position-based: at Put time there is no fresh chuck to read a grip bit from, and
  no state is carried over from a previous Pick.
Either raises GraspFailed. Both checks are skipped entirely when the server is started
with --skip-grasp-check (``parent_server.verify_grasp`` is False).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Optional, Type

from bcap_sila2.bcap import RobotUnavailableError, read_variable, write_variable
from orinexception import ORiNException
from sila2.server import MetadataDict, ObservableCommandInstanceWithIntermediateResponses

from .. import ops
from ..generated.labwareservice import (
    ActivateHand_IntermediateResponses,
    ActivateHand_Responses,
    CarriageFault,
    CarriageNotReady,
    ControllerConnectionError,
    GraspFailed,
    HandError,
    HandNotOpen,
    InvalidHandPosition,
    InvalidStation,
    LabwareServiceBase,
    MoveHand_IntermediateResponses,
    MoveHand_Responses,
    MoveTimeout,
    NoStationAtPosition,
    PickLabware_IntermediateResponses,
    PickLabware_Responses,
    PlcAccessError,
    PlcConnectionError,
    PoseNotRestored,
    PutLabware_IntermediateResponses,
    PutLabware_Responses,
    RobotAccessError,
    RobotNotAtBasePose,
    RobotNotAtKnownPose,
    RobotNotAtRetractPose,
    TaskAccessError,
    TaskExecutionTimeout,
    ToggleLight_Responses,
    Transfer_IntermediateResponses,
    Transfer_Responses,
    VariableAccessError,
)

if TYPE_CHECKING:
    from ..server import Server

# ops raises feature-agnostic exceptions; each one maps to an error this feature declares.
_OP_ERRORS: dict[Type[ops.OpError], Type[Exception]] = {
    ops.PlcConnectionFailed: PlcConnectionError,
    ops.PlcAccessFailed: PlcAccessError,
    ops.ControllerConnectionFailed: ControllerConnectionError,
    ops.RobotAccessFailed: RobotAccessError,
    ops.TaskTimedOut: TaskExecutionTimeout,
    ops.TaskFailed: TaskAccessError,
    ops.PoseNotReached: PoseNotRestored,
    ops.WrongStartPose: PoseNotRestored,   # commands with a pose gate override this
    ops.NotAtKnownPose: RobotNotAtKnownPose,
    ops.NotInMovablePose: RobotNotAtKnownPose,
    ops.HandNotOpenError: HandNotOpen,
    ops.HandFailed: HandError,
    ops.HandPositionInvalid: InvalidHandPosition,
    ops.GraspFailedError: GraspFailed,
    ops.CarriageNotReadyError: CarriageNotReady,
    ops.CarriageFaultError: CarriageFault,
    ops.CarriageMoveTimedOut: MoveTimeout,
    ops.StationUnknown: InvalidStation,
    ops.NoStationHere: NoStationAtPosition,
}


@contextmanager
def _sila_errors(start_pose_error: Optional[Type[Exception]] = None) -> Iterator[None]:
    """Translate ops exceptions into this feature's SiLA errors.

    ``start_pose_error`` names the error for "the arm is not where this command starts",
    which differs per command (Pick says base, Put says retract) though ops reports one
    condition. Anything unmapped propagates, and the framework reports it as undefined.
    """
    try:
        yield
    except ops.OpError as e:
        cls = _OP_ERRORS.get(type(e))
        if start_pose_error is not None and isinstance(e, ops.WrongStartPose):
            cls = start_pose_error
        if cls is None:
            raise
        raise cls(str(e)) from e


class LabwareServiceImpl(LabwareServiceBase):
    def __init__(self, parent_server: Server) -> None:
        super().__init__(parent_server=parent_server)
        # Guards only the light's read-modify-write toggle. Deliberately not the server's
        # operation_lock: switching a light is not motion and must not wait for a pick.
        self._light_lock = threading.Lock()

    @property
    def _ops(self) -> ops.MotionOps:
        return self.parent_server.ops

    # ---- observable command: PickLabware ----
    def PickLabware(
        self,
        *,
        metadata: MetadataDict,
        instance: ObservableCommandInstanceWithIntermediateResponses[PickLabware_IntermediateResponses],
    ) -> PickLabware_Responses:
        def phase(name: str) -> None:
            instance.send_intermediate_response(PickLabware_IntermediateResponses(Phase=name))

        # Hold the OperationCoordinator for the whole pick.
        with self.parent_server.operation_lock, _sila_errors(RobotNotAtBasePose):
            instance.begin_execution()
            # No StationId argument yet: the station is the one at the current position.
            station_id, station = self._ops.station_here()
            at_retract = self._ops.pick(
                station_id, station, phase, self.parent_server.verify_grasp
            )
            instance.progress = 1.0
            return PickLabware_Responses(AtRetractPose=at_retract)

    # ---- observable command: PutLabware ----
    def PutLabware(
        self,
        *,
        metadata: MetadataDict,
        instance: ObservableCommandInstanceWithIntermediateResponses[PutLabware_IntermediateResponses],
    ) -> PutLabware_Responses:
        def phase(name: str) -> None:
            instance.send_intermediate_response(PutLabware_IntermediateResponses(Phase=name))

        with self.parent_server.operation_lock, _sila_errors(RobotNotAtRetractPose):
            instance.begin_execution()
            station_id, station = self._ops.station_here()
            at_retract = self._ops.put(
                station_id, station, phase, self.parent_server.verify_grasp
            )
            instance.progress = 1.0
            return PutLabware_Responses(AtRetractPose=at_retract)

    # ---- observable command: Transfer ----
    def Transfer(
        self,
        SourceStation: str,
        DestinationStation: str,
        *,
        metadata: MetadataDict,
        instance: ObservableCommandInstanceWithIntermediateResponses[Transfer_IntermediateResponses],
    ) -> Transfer_Responses:
        def phase(name: str) -> None:
            instance.send_intermediate_response(Transfer_IntermediateResponses(Phase=name))

        # One lock for the whole route: no carriage move or pick may interleave with it.
        with self.parent_server.operation_lock, _sila_errors():
            instance.begin_execution()
            final, at_retract = self._ops.transfer(
                str(SourceStation).strip(),
                str(DestinationStation).strip(),
                phase,
                self.parent_server.verify_grasp,
            )
            instance.progress = 1.0
            return Transfer_Responses(CarriagePosition=final, AtRetractPose=at_retract)

    # ---- observable command: MoveHand (gripper on its own) ----
    def MoveHand(
        self,
        Position: int,
        *,
        metadata: MetadataDict,
        instance: ObservableCommandInstanceWithIntermediateResponses[MoveHand_IntermediateResponses],
    ) -> MoveHand_Responses:
        h = self.parent_server.motion.hand
        target = int(Position)
        if not (0 <= target <= h.open_position):
            raise InvalidHandPosition(
                f"Position {target} is outside 0..{h.open_position} (device units)."
            )

        def phase(name: str) -> None:
            instance.send_intermediate_response(MoveHand_IntermediateResponses(Phase=name))

        # Same lock as Pick/Put: the hand must not move while a pick/put owns it.
        with self.parent_server.operation_lock, _sila_errors():
            instance.begin_execution()
            phase(f"moving hand to {target} (speed {h.speed}, force {h.grip_force})")
            grip_bit = self._ops.hand_move(target)
            reached = self._ops.hand_position()
            phase(f"done at {reached}")
            instance.progress = 1.0
            return MoveHand_Responses(Position=reached, StoppedShort=grip_bit == 0)

    # ---- observable command: ActivateHand ----
    def ActivateHand(
        self,
        *,
        metadata: MetadataDict,
        instance: ObservableCommandInstanceWithIntermediateResponses[ActivateHand_IntermediateResponses],
    ) -> ActivateHand_Responses:
        def phase(name: str) -> None:
            instance.send_intermediate_response(ActivateHand_IntermediateResponses(Phase=name))

        with self.parent_server.operation_lock, _sila_errors():
            instance.begin_execution()
            # Activation strokes the jaws, so refuse unless they are empty and open.
            self._ops.require_hand_open(
                "activation strokes the jaws, so it is refused unless they are fully open"
            )
            phase("toggling D5002.0 OFF -> ON (the jaws will stroke)")
            self._ops.activate_hand()
            phase("activated")
            instance.progress = 1.0
            return ActivateHand_Responses(Activated=True)

    # ---- machine light (a robot-controller variable, not a PLC signal) ----
    def _read_light(self) -> bool:
        cfg = self.parent_server.config
        try:
            return bool(read_variable(cfg.controller, cfg.light.variable))
        except OSError as e:
            raise ControllerConnectionError(str(e))
        except (ORiNException, RobotUnavailableError) as e:
            raise VariableAccessError(str(e))

    def get_LightIsOn(self, *, metadata: MetadataDict) -> bool:
        return self._read_light()

    def ToggleLight(self, *, metadata: MetadataDict) -> ToggleLight_Responses:
        cfg = self.parent_server.config
        # Read-modify-write, so serialise it: two clients toggling at once would
        # otherwise both read the same state and one write would be lost.
        with self._light_lock:
            new_state = not self._read_light()
            try:
                write_variable(cfg.controller, cfg.light.variable, new_state)
            except OSError as e:
                raise ControllerConnectionError(str(e))
            except (ORiNException, RobotUnavailableError) as e:
                raise VariableAccessError(str(e))
        return ToggleLight_Responses(IsOn=new_state)
