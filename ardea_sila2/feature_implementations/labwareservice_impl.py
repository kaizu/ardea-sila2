"""LabwareService implementation (Ardea-specific): PickLabware / PutLabware.

Both commands resolve a station from the current carriage position (motion
``station_at``; no StationId argument yet), then run that station's task pair
around a hand action. The poses used depend on the station's ``direction``:
forward uses the base/retract poses, reverse uses the 180°-turned inverse
base/retract poses (motion ``poses_for`` / ``return_home_for``):
  PickLabware (robot must start at the direction's base pose):
    approach (script_a) -> close hand (chuck) -> retract (script_b) -> confirm retract.
  PutLabware (robot must start at the direction's retract pose):
    approach (script_a) -> open hand (unchuck) -> retract (script_b) -> confirm retract
    -> return_home (direction's common task, hand must be open) -> confirm base.

Reuses bcap task/pose helpers and the kvcomplus atomic primitives; holds the
server OperationCoordinator for the whole sequence so no carriage move runs
concurrently. The D5000.1 carriage-lockout is NOT used (it breaks b-CAP RunTask;
see note below).

Grasp verification (by position, not the grip bit): an empty hand springs back to
the open position when the move bit clears, while a held labware keeps the jaws
closed at the object's width (which can be small, e.g. 4, or larger, e.g. 79). So
"holding" == the current hand position is NOT near open (<= open_position -
_GRASP_MARGIN). PickLabware checks this after the chuck (with a short settle for the
empty case to reopen); PutLabware checks it before opening. Either raises GraspFailed.
(D6002.6/.7 fall back to 0 at rest and don't distinguish held-small-object from
empty, so position is used instead.)
"""

from __future__ import annotations

import time
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

from ..generated.labwareservice import (
    ControllerConnectionError,
    GraspFailed,
    HandError,
    HandNotOpen,
    LabwareServiceBase,
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
    RobotNotAtRetractPose,
    TaskAccessError,
    TaskExecutionTimeout,
)

if TYPE_CHECKING:
    from ..server import Server

DM = 18
# NOTE: the design's D5000.1 "robot-operating" carriage-lockout interlock is NOT
# used here. Empirically, setting D5000.1 ON makes b-CAP RunTask fail
# (TaskAccessError 0x81501078), so it is incompatible with running robot tasks and
# is omitted until a different lockout mechanism is worked out (design §4.3 / Q8).
# Hand signals.
HAND_WORD = 5002
BIT_HAND_ACTIVE = 0            # D5002.0
BIT_HAND_MOVE = 3             # D5002.03
HAND_STATUS = 6002            # D6002 (.0 active, .4/.5 activation done, .7 done)
HAND_TGT = 5050
HAND_SPEED = 5060
HAND_FORCE = 5070
HAND_CUR_POS = 6060           # D6060 hand current position
CARRIAGE_CUR_POS = 6010       # D6010 carriage current position [mm] (2 words)
_HAND_START_TIMEOUT_S = 5.0
_HAND_OPEN_TOL = 3            # tolerance [units] for "hand fully open" check
# "Holding a labware" = the jaws did NOT spring back to the open position after the
# move bit cleared (an empty hand reopens to ~open_position; a held one stays closed
# at the object's width, which can be anywhere well below open, e.g. 4 or 79). So
# holding <=> current position <= open_position - _GRASP_MARGIN (no lower bound).
_GRASP_MARGIN = 10           # margin [units] below open_position that still counts as "holding"
_GRASP_SETTLE_S = 1.5        # wait after a chuck for an empty hand to reopen before checking
_ONE_CYCLE = 1                 # RunTask mode: run once and stop


class LabwareServiceImpl(LabwareServiceBase):
    def __init__(self, parent_server: Server) -> None:
        super().__init__(parent_server=parent_server)

    # ---- helpers ----
    def _plc(self):
        return self.parent_server.config.plc

    @staticmethod
    def _kv(fn):
        try:
            return fn()
        except kvcomplus.KvComError as e:
            msg = str(e).lower()
            if "bridge" in msg or "connect" in msg or "timed out" in msg:
                raise PlcConnectionError(str(e))
            raise PlcAccessError(str(e))

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

    def _hand_status(self) -> int:
        return self._kv(lambda: kvcomplus.read_word(self._plc(), DM, HAND_STATUS))

    def _hand_move(self, target: int) -> int:
        """Move the hand to ``target`` (activate if needed, write params, wait done).

        Used for both chuck (target=closed) and unchuck (target=open). Returns the
        in-position/grip bit D6002.6 sampled at the completion instant (bit7->1):
        0 = stopped short (gripping an object), 1 = reached the commanded position.
        """
        plc = self._plc()
        h = self.parent_server.motion.hand

        # activate (idempotent): D5002.0 ON, wait D6002.0/.4/.5
        st = self._hand_status()
        if not (st & 1 and (st >> 4) & 1 and (st >> 5) & 1):
            self._kv(lambda: kvcomplus.write_bit(plc, DM, HAND_WORD, BIT_HAND_ACTIVE, True))
            t0 = time.time()
            while True:
                st = self._hand_status()
                if st & 1 and (st >> 4) & 1 and (st >> 5) & 1:
                    break
                if time.time() - t0 > h.move_timeout_s:
                    raise HandError("hand activation did not complete (D6002.0/.4/.5)")
                time.sleep(0.3)

        # write target/speed/force, then raise the move trigger
        self._kv(lambda: kvcomplus.write_word(plc, DM, HAND_TGT, int(target)))
        self._kv(lambda: kvcomplus.write_word(plc, DM, HAND_SPEED, h.speed))
        self._kv(lambda: kvcomplus.write_word(plc, DM, HAND_FORCE, h.grip_force))
        self._kv(lambda: kvcomplus.write_bit(plc, DM, HAND_WORD, BIT_HAND_MOVE, True))
        try:
            # phase A: wait for motion to start (done-bit7 -> 0)
            t0 = time.time()
            while time.time() - t0 <= _HAND_START_TIMEOUT_S:
                if (self._hand_status() >> 7) & 1 == 0:
                    break
                time.sleep(0.3)
            # phase B: wait for completion (done-bit7 -> 1); sample grip bit there
            grip_bit = 1
            t1 = time.time()
            while True:
                st = self._hand_status()
                if (st >> 7) & 1 == 1:
                    grip_bit = (st >> 6) & 1
                    break
                if time.time() - t1 > h.move_timeout_s:
                    raise HandError("hand move did not complete (D6002.7)")
                time.sleep(0.3)
        finally:
            self._kv(lambda: kvcomplus.write_bit(plc, DM, HAND_WORD, BIT_HAND_MOVE, False))
        return grip_bit

    # ---- observable command: PickLabware ----
    def PickLabware(
        self,
        *,
        metadata: MetadataDict,
        instance: ObservableCommandInstanceWithIntermediateResponses[PickLabware_IntermediateResponses],
    ) -> PickLabware_Responses:
        motion = self.parent_server.motion

        def phase(name: str) -> None:
            instance.send_intermediate_response(PickLabware_IntermediateResponses(Phase=name))

        # Hold the OperationCoordinator for the whole pick.
        with self.parent_server.operation_lock:
            # Resolve the station from the current carriage position (no StationId arg yet).
            carriage_pos = self._kv(lambda: kvcomplus.read_dword(self._plc(), DM, CARRIAGE_CUR_POS))
            resolved = motion.station_at(carriage_pos)
            if resolved is None:
                raise NoStationAtPosition(f"No station defined at carriage position {carriage_pos} mm.")
            station_id, station = resolved

            # Poses depend on the station's facing: forward -> base/retract,
            # reverse -> inverse base/retract (arm turned 180°).
            base_like, retract_like = motion.poses_for(station.direction)
            base_name = "inverse base" if station.direction == "reverse" else "base"
            retract_name = "inverse retract" if station.direction == "reverse" else "retract"

            # Precondition: the hand must be fully open.
            hand_pos = self._kv(lambda: kvcomplus.read_word(self._plc(), DM, HAND_CUR_POS))
            open_pos = motion.hand.open_position
            if abs(hand_pos - open_pos) > _HAND_OPEN_TOL:
                raise HandNotOpen(f"Hand is at {hand_pos} (open={open_pos}); must be fully open to pick.")

            # Pose gate: PickLabware requires the direction's start pose only.
            angles = self._joint_angles()
            if not base_like.matches(angles):
                raise RobotNotAtBasePose(
                    f"Robot is not at the {base_name} pose ({station.direction} station); pick refused."
                )

            instance.begin_execution()
            phase(f"start (station {station_id}, {station.direction})")

            phase(f"approach: RunTask({station.script_a})")
            self._run_task(station.script_a)

            phase("chuck: closing hand")
            # Close target depends on the station's grip orientation (long -> not fully closed).
            self._hand_move(motion.hand.closed_position_for(station.grip))
            # Grasp check by position: an empty hand springs back to the open position
            # once the move bit clears, while a held labware keeps the jaws closed
            # (at the object's width). After a short settle, "grasped" = not reopened.
            time.sleep(_GRASP_SETTLE_S)
            hand_pos = self._kv(lambda: kvcomplus.read_word(self._plc(), DM, HAND_CUR_POS))
            if hand_pos > motion.hand.open_position - _GRASP_MARGIN:
                raise GraspFailed(f"No labware grasped (hand reopened to {hand_pos}).")

            phase(f"retract: RunTask({station.script_b})")
            self._run_task(station.script_b)

            # Confirm the robot returned to the direction's retract pose.
            phase(f"verify {retract_name} pose")
            at_retract = retract_like.matches(self._joint_angles())
            if not at_retract:
                raise PoseNotRestored(
                    f"Robot did not return to the {retract_name} pose after pick-retract."
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
        motion = self.parent_server.motion

        def phase(name: str) -> None:
            instance.send_intermediate_response(PutLabware_IntermediateResponses(Phase=name))

        with self.parent_server.operation_lock:
            # Resolve the station from the current carriage position (no StationId arg yet).
            carriage_pos = self._kv(lambda: kvcomplus.read_dword(self._plc(), DM, CARRIAGE_CUR_POS))
            resolved = motion.station_at(carriage_pos)
            if resolved is None:
                raise NoStationAtPosition(f"No station defined at carriage position {carriage_pos} mm.")
            station_id, station = resolved

            # Poses depend on the station's facing (forward -> base/retract,
            # reverse -> inverse base/retract).
            base_like, retract_like = motion.poses_for(station.direction)
            base_name = "inverse base" if station.direction == "reverse" else "base"
            retract_name = "inverse retract" if station.direction == "reverse" else "retract"

            # Precondition: the robot must be at the direction's retract pose (start pose).
            angles = self._joint_angles()
            if not retract_like.matches(angles):
                raise RobotNotAtRetractPose(
                    f"Robot is not at the {retract_name} pose ({station.direction} station); put refused."
                )

            # Precondition: the hand must be holding a labware, i.e. the jaws are not
            # (near) fully open. An empty hand rests at ~open_position; a held one
            # stays closed at the object's width (which may be small, e.g. 4).
            h = motion.hand
            hand_pos = self._kv(lambda: kvcomplus.read_word(self._plc(), DM, HAND_CUR_POS))
            if hand_pos > h.open_position - _GRASP_MARGIN:
                raise GraspFailed(
                    f"Hand is not holding a labware (D6060={hand_pos} ~ open); nothing to put."
                )

            instance.begin_execution()
            phase(f"start (station {station_id}, {station.direction})")

            phase(f"approach: RunTask({station.script_a})")
            self._run_task(station.script_a)

            phase("unchuck: opening hand")
            self._hand_move(motion.hand.open_position)

            phase(f"retract: RunTask({station.script_b})")
            self._run_task(station.script_b)

            phase(f"verify {retract_name} pose")
            if not retract_like.matches(self._joint_angles()):
                raise PoseNotRestored(f"Robot did not return to the {retract_name} pose after put-retract.")

            # Return to the direction's base pose. The return-home task only runs with
            # the hand open, which it is after the unchuck above; verify defensively.
            return_home = motion.return_home_for(station.direction)
            hand_pos = self._kv(lambda: kvcomplus.read_word(self._plc(), DM, HAND_CUR_POS))
            if abs(hand_pos - motion.hand.open_position) > _HAND_OPEN_TOL:
                raise HandNotOpen(
                    f"Hand is at {hand_pos} (open={motion.hand.open_position}); "
                    "return-home requires the hand open."
                )

            phase(f"return home: RunTask({return_home})")
            self._run_task(return_home)

            phase(f"verify {base_name} pose")
            at_base = base_like.matches(self._joint_angles())
            if not at_base:
                raise PoseNotRestored(f"Robot did not return to the {base_name} pose after return-home.")

            instance.progress = 1.0
            return PutLabware_Responses(AtBasePose=at_base)
