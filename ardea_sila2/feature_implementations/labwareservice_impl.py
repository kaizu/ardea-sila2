"""LabwareService implementation (Ardea-specific) — PickLabware.

Test sequence (user-defined 2026-07-08):
  1. verify the robot is at a movable pose (base or retract);
  2. set the robot-operating signal D5000.1 ON (carriage lockout);
  3. RunTask(pick_approach)   -- PickUp0, approach/grab;
  4. close the hand (chuck);
  5. RunTask(pick_retract)    -- PickUp1, ends at the retract pose;
  6. clear D5000.1 (finally);
  7. confirm the robot is at the retract pose.

Reuses bcap task/pose helpers and the kvcomplus atomic primitives; holds the
server OperationCoordinator for the whole sequence so no carriage move runs
concurrently.

Grasp verification: PickLabware confirms a labware was grasped after closing
(D6002.6==0 at chuck completion = stopped short on an object); PutLabware confirms
the hand is holding a labware before opening (hand position between fully-closed
and fully-open). Both raise GraspFailed otherwise. D6002.6/.7 fall back to 0 at
rest, so the grip bit is sampled at the move's completion instant (see _hand_move).
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
    CarriageNotAtOrigin,
    ControllerConnectionError,
    GraspFailed,
    HandError,
    HandNotOpen,
    LabwareServiceBase,
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
_GRASP_MARGIN = 5            # margin [units] from full open/close for "holding a labware"
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
        pac = motion.pacscripts

        def phase(name: str) -> None:
            instance.send_intermediate_response(PickLabware_IntermediateResponses(Phase=name))

        # Hold the OperationCoordinator for the whole pick.
        with self.parent_server.operation_lock:
            # Precondition: the carriage must be at the origin (0 mm).
            carriage_pos = self._kv(lambda: kvcomplus.read_dword(self._plc(), DM, CARRIAGE_CUR_POS))
            if carriage_pos != 0:
                raise CarriageNotAtOrigin(f"Carriage is at {carriage_pos} mm; must be 0 mm to pick.")

            # Precondition: the hand must be fully open.
            hand_pos = self._kv(lambda: kvcomplus.read_word(self._plc(), DM, HAND_CUR_POS))
            open_pos = motion.hand.open_position
            if abs(hand_pos - open_pos) > _HAND_OPEN_TOL:
                raise HandNotOpen(f"Hand is at {hand_pos} (open={open_pos}); must be fully open to pick.")

            # Pose gate: PickLabware requires the base pose only (retract not allowed).
            angles = self._joint_angles()
            if not motion.base_pose.matches(angles):
                raise RobotNotAtBasePose("Robot is not at the base pose; pick refused.")

            instance.begin_execution()
            phase("start")

            # (D5000.1 carriage-lockout intentionally omitted — see note at top.)
            phase(f"approach: RunTask({pac.pick_approach})")
            self._run_task(pac.pick_approach)

            phase("chuck: closing hand")
            grip_bit = self._hand_move(motion.hand.closed_position)
            # Grasp check: at chuck completion, D6002.6==0 means the hand stopped
            # short on an object (grasped); ==1 means it reached full close (empty).
            if grip_bit != 0:
                raise GraspFailed("No labware grasped (hand reached full close = empty grip).")

            phase(f"retract: RunTask({pac.pick_retract})")
            self._run_task(pac.pick_retract)

            # Confirm the robot returned to the retract pose.
            phase("verify retract pose")
            at_retract = motion.retract_pose.matches(self._joint_angles())
            if not at_retract:
                raise PoseNotRestored("Robot did not return to the retract pose after pick-retract.")

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
        pac = motion.pacscripts

        def phase(name: str) -> None:
            instance.send_intermediate_response(PutLabware_IntermediateResponses(Phase=name))

        with self.parent_server.operation_lock:
            # Precondition: the carriage must be at the origin (0 mm).
            carriage_pos = self._kv(lambda: kvcomplus.read_dword(self._plc(), DM, CARRIAGE_CUR_POS))
            if carriage_pos != 0:
                raise CarriageNotAtOrigin(f"Carriage is at {carriage_pos} mm; must be 0 mm to put.")

            # Precondition: the robot must be at the retract pose (base pose is NOT allowed).
            angles = self._joint_angles()
            if not motion.retract_pose.matches(angles):
                raise RobotNotAtRetractPose("Robot is not at the retract pose; put refused.")

            # Precondition: the hand must be holding a labware (position between fully
            # closed and fully open). At rest the grip bit is lost, so use position.
            h = motion.hand
            hand_pos = self._kv(lambda: kvcomplus.read_word(self._plc(), DM, HAND_CUR_POS))
            if not (h.closed_position + _GRASP_MARGIN <= hand_pos <= h.open_position - _GRASP_MARGIN):
                raise GraspFailed(
                    f"Hand is not holding a labware (D6060={hand_pos}); nothing to put."
                )

            instance.begin_execution()
            phase("start")

            phase(f"approach: RunTask({pac.put_approach})")
            self._run_task(pac.put_approach)

            phase("unchuck: opening hand")
            self._hand_move(motion.hand.open_position)

            phase(f"retract: RunTask({pac.put_retract})")
            self._run_task(pac.put_retract)

            phase("verify retract pose")
            if not motion.retract_pose.matches(self._joint_angles()):
                raise PoseNotRestored("Robot did not return to the retract pose after put-retract.")

            # Return to the base pose. PickUp2 (return_home) only runs with the hand
            # open, which it is after the unchuck above; verify defensively.
            hand_pos = self._kv(lambda: kvcomplus.read_word(self._plc(), DM, HAND_CUR_POS))
            if abs(hand_pos - motion.hand.open_position) > _HAND_OPEN_TOL:
                raise HandNotOpen(
                    f"Hand is at {hand_pos} (open={motion.hand.open_position}); "
                    "return-home requires the hand open."
                )

            phase(f"return home: RunTask({pac.return_home})")
            self._run_task(pac.return_home)

            phase("verify base pose")
            at_base = motion.base_pose.matches(self._joint_angles())
            if not at_base:
                raise PoseNotRestored("Robot did not return to the base pose after return-home.")

            instance.progress = 1.0
            return PutLabware_Responses(AtBasePose=at_base)
