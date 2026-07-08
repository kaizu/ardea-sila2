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

NOTE: grasp-success verification (chuck ending with D6002.6==0 = holding an
object) is intentionally NOT enforced here, because the test runs without labware
(the hand closes fully = empty grip). The formal version will use D6002.6.
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
    HandError,
    HandNotOpen,
    LabwareServiceBase,
    PickLabware_IntermediateResponses,
    PickLabware_Responses,
    PlcAccessError,
    PlcConnectionError,
    PoseNotRestored,
    RobotAccessError,
    RobotNotInMovablePose,
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

    def _chuck(self) -> None:
        """Close the hand (activate if needed, move to closed position, wait done).

        Grasp success (D6002.6) is deliberately not checked here (test without labware).
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
        self._kv(lambda: kvcomplus.write_word(plc, DM, HAND_TGT, h.closed_position))
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
            # phase B: wait for completion (done-bit7 -> 1)
            t1 = time.time()
            while True:
                if (self._hand_status() >> 7) & 1 == 1:
                    break
                if time.time() - t1 > h.move_timeout_s:
                    raise HandError("hand chuck did not complete (D6002.7)")
                time.sleep(0.3)
        finally:
            self._kv(lambda: kvcomplus.write_bit(plc, DM, HAND_WORD, BIT_HAND_MOVE, False))

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

            # Pose gate: robot must start at the base or retract pose.
            angles = self._joint_angles()
            if not (motion.base_pose.matches(angles) or motion.retract_pose.matches(angles)):
                raise RobotNotInMovablePose(
                    "Robot is at neither the base nor the retract pose; pick refused."
                )

            instance.begin_execution()
            phase("start")

            # (D5000.1 carriage-lockout intentionally omitted — see note at top.)
            phase(f"approach: RunTask({pac.pick_approach})")
            self._run_task(pac.pick_approach)

            phase("chuck: closing hand")
            self._chuck()

            phase(f"retract: RunTask({pac.pick_retract})")
            self._run_task(pac.pick_retract)

            # Confirm the robot returned to the retract pose.
            phase("verify retract pose")
            at_retract = motion.retract_pose.matches(self._joint_angles())
            if not at_retract:
                raise PoseNotRestored("Robot did not return to the retract pose after pick-retract.")

            instance.progress = 1.0
            return PickLabware_Responses(AtRetractPose=at_retract)
