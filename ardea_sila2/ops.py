"""The machine's physical operations, independent of any SiLA feature.

Every motion the Ardea server can perform lives here exactly once: carriage moves, the
named-pose transitions, pick, put, and the hand. The feature implementations are thin
wrappers that take the server's OperationCoordinator and translate this module's
exceptions into their own SiLA errors.

Why the split: ``LabwareService.Transfer`` performs a carriage move, a turn, a pick and a
put in one command. It cannot call ``CarriageService.MoveCarriage`` and friends to do
that -- each of those acquires ``parent_server.operation_lock``, which is a plain
``threading.Lock``, so a command calling another would deadlock against itself, and each
also expects a SiLA command ``instance`` of its own. So the physical work moved down here,
where it takes **no lock** and reports progress through a plain ``phase`` callback. The
public commands keep the locking, the pose gates they advertise, and their error sets.

Two rules for anything added here:

- **Take no lock.** The caller holds ``operation_lock`` for the whole operation.
- **Raise only** :class:`OpError` **subclasses.** They carry no SiLA identity; each
  feature maps them onto the errors its own feature definition declares (the same
  condition is a different generated class in each feature, e.g. LabwareService's
  ``PlcAccessError`` is not CarriageService's).

Device addresses are the Ardea signal proposal's (orchestration_design.md §3): all DM
(device type 18), 2-word values signed 32-bit (low word at the address, high at +1).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Optional

from bcap_sila2.bcap import (
    RobotUnavailableError,
    TaskAbnormalStopError,
    TaskTimeoutError,
    get_joint_angles,
    run_task,
)
from orinexception import ORiNException

from kvcomplus_sila2 import kvcomplus

from .motion_config import PoseConfig, StationConfig

if TYPE_CHECKING:
    from .server import Server

# --- device addresses (KV-8000, DM = type 18) ---
DM = 18
# carriage
ADDR_REQ_WORD = 5000       # D5000 : holds the move-request bit
BIT_MOVE_REQ = 0           #   .0  : travel-carriage move-request
ADDR_DONE_WORD = 6000      # D6000 : complete (.0) / moving (.1)
BIT_DONE = 0
BIT_MOVING = 1
ADDR_TGT = 5010            # D5010 : target position [mm]      (2 words)
ADDR_SPEED = 5020          # D5020 : positioning speed [mm/s]  (2 words)
ADDR_ACCEL = 5030          # D5030 : accel/decel [mm/s/ms]     (2 words)
ADDR_CUR_POS = 6010        # D6010 : current position [mm]     (2 words)
ADDR_FAULT = 6005          # D6005 : fault/alarm bits (any set = fault)
# hand
HAND_WORD = 5002           # D5002 : hand write bits
BIT_HAND_ACTIVE = 0        #   .0  : activation enable (level, but recovery needs an edge)
BIT_HAND_MOVE = 3          #   .03 : move trigger (needs a 0 -> 1 edge)
HAND_STATUS = 6002         # D6002 : .0 active, .4/.5 activation done, .6 in-position, .7 done
HAND_TGT = 5050            # D5050 : target position
HAND_SPEED = 5060          # D5060 : speed
HAND_FORCE = 5070          # D5070 : grip force
HAND_CUR_POS = 6060        # D6060 : current position

# NOTE: the design's D5000.1 "robot-operating" carriage lockout is NOT used. Setting it
# ON makes b-CAP RunTask fail (TaskAccessError 0x81501078), so it is incompatible with
# running robot tasks (design §4.3 / Q8).

_CARRIAGE_START_TIMEOUT_S = 10.0   # max wait for the PLC to acknowledge a move
_HAND_START_TIMEOUT_S = 5.0        # max wait for the hand's done bit to clear
_HAND_ACTIVATE_OFF_S = 1.0         # how long D5002.0 stays OFF before the recovering edge
HAND_OPEN_TOL = 3                  # tolerance [units] for "hand fully open"
GRASP_MARGIN = 10                  # below open_position, still counts as "holding"
_ONE_CYCLE = 1                     # RunTask mode: run once and stop

Phase = Callable[[str], None]


def no_phase(_: str) -> None:
    """Default ``phase`` callback: report nothing."""


# --- exceptions -------------------------------------------------------------------
# Deliberately not SiLA errors: see the module docstring.

class OpError(Exception):
    """Base class for every failure this module reports."""


class PlcConnectionFailed(OpError): ...
class PlcAccessFailed(OpError): ...
class ControllerConnectionFailed(OpError): ...
class RobotAccessFailed(OpError): ...
class TaskTimedOut(OpError): ...
class TaskFailed(OpError): ...
class PoseNotReached(OpError): ...     # a task ran but the arm is not where it should be
class WrongStartPose(OpError): ...     # precondition: the arm is not where the op starts
class NotAtKnownPose(OpError): ...
class NotInMovablePose(OpError): ...
class HandNotOpenError(OpError): ...
class HandFailed(OpError): ...
class HandPositionInvalid(OpError): ...
class GraspFailedError(OpError): ...
class CarriageNotReadyError(OpError): ...
class CarriageFaultError(OpError): ...
class CarriageMoveTimedOut(OpError): ...
class StationUnknown(OpError): ...
class NoStationHere(OpError): ...


class MotionOps:
    """Physical operations on one Ardea machine. Holds no state of its own."""

    def __init__(self, server: Server) -> None:
        self.server = server

    # ---- plumbing ----
    @property
    def motion(self):
        return self.server.motion

    def _plc(self):
        return self.server.config.plc

    @staticmethod
    def _kv(fn):
        """Run a kvcomplus call, classifying KvComError as connection vs access."""
        try:
            return fn()
        except kvcomplus.KvComError as e:
            msg = str(e).lower()
            if "bridge" in msg or "connect" in msg or "timed out" in msg:
                raise PlcConnectionFailed(str(e))
            raise PlcAccessFailed(str(e))

    def joint_angles(self) -> list[float]:
        try:
            return get_joint_angles(self.server.config.controller)
        except OSError as e:
            raise ControllerConnectionFailed(str(e))
        except (ORiNException, RobotUnavailableError) as e:
            raise RobotAccessFailed(str(e))

    def run_task(self, name: str) -> None:
        cfg = self.server.config
        tcfg = cfg.task
        try:
            run_task(
                cfg.controller, name, mode=_ONE_CYCLE,
                poll_interval=tcfg.poll_interval_seconds,
                start_timeout=tcfg.start_timeout_seconds,
                completion_timeout=tcfg.completion_timeout_seconds,
            )
        except OSError as e:
            raise ControllerConnectionFailed(str(e))
        except TaskTimeoutError as e:
            raise TaskTimedOut(str(e))
        except (ORiNException, TaskAbnormalStopError) as e:
            raise TaskFailed(str(e))

    # ---- reads ----
    def carriage_position(self) -> int:
        return self._kv(lambda: kvcomplus.read_dword(self._plc(), DM, ADDR_CUR_POS))

    def hand_position(self) -> int:
        return self._kv(lambda: kvcomplus.read_word(self._plc(), DM, HAND_CUR_POS))

    def hand_status(self) -> int:
        return self._kv(lambda: kvcomplus.read_word(self._plc(), DM, HAND_STATUS))

    def hand_is_open(self) -> bool:
        return abs(self.hand_position() - self.motion.hand.open_position) <= HAND_OPEN_TOL

    def at_movable_pose(self) -> bool:
        return self.motion.at_movable_pose(self.joint_angles())

    def station_here(self) -> "tuple[str, StationConfig]":
        """Resolve the station from the current carriage position, or raise."""
        pos = self.carriage_position()
        resolved = self.motion.station_at(pos)
        if resolved is None:
            raise NoStationHere(f"No station defined at carriage position {pos} mm.")
        return resolved

    def station_by_name(self, name: str) -> "tuple[str, StationConfig]":
        station = self.motion.stations.get(name)
        if station is None:
            raise StationUnknown(
                f"{name!r} is not a station; known stations: "
                f"{', '.join(sorted(self.motion.stations))}."
            )
        return name, station

    @staticmethod
    def pose_names(direction: str) -> "tuple[str, str]":
        """Human-readable (base-like, retract-like) pose names for a station direction."""
        if direction == "reverse":
            return "inverse base", "inverse retract"
        return "base", "retract"

    # ---- hand ----
    @staticmethod
    def _is_activated(status: int) -> bool:
        """True if D6002 shows the activated state (.0 and .4 and .5, i.e. 0x0031)."""
        return bool(status & 1 and (status >> 4) & 1 and (status >> 5) & 1)

    def activate_hand(self) -> None:
        """Drive D5002.0 OFF then ON and wait for the activated state.

        The falling edge is the point: with D5002.0 already ON but the hand stuck at
        D6002=0x0000/0x0001, writing ON again does nothing (2026-07-24, again 2026-08-17).
        **This strokes the jaws** (140 -> ~10 -> 140 measured 2026-08-14), so every caller
        must have checked that they are open -- a held labware would be dropped.
        """
        plc = self._plc()
        h = self.motion.hand
        self._kv(lambda: kvcomplus.write_bit(plc, DM, HAND_WORD, BIT_HAND_ACTIVE, False))
        time.sleep(_HAND_ACTIVATE_OFF_S)
        self._kv(lambda: kvcomplus.write_bit(plc, DM, HAND_WORD, BIT_HAND_ACTIVE, True))
        t0 = time.time()
        while not self._is_activated(self.hand_status()):
            if time.time() - t0 > h.move_timeout_s:
                raise HandFailed("hand activation did not complete (D6002.0/.4/.5)")
            time.sleep(0.3)

    def hand_move(self, target: int) -> int:
        """Move the hand to ``target``; return the grip bit D6002.6 sampled at completion.

        0 = the jaws stopped short (an object is held), 1 = they reached the commanded
        position. Sampling at the completion instant (bit7 -> 1) is what makes this
        reliable for both short- and long-edge grips; read later, bit 6 falls again.

        A deactivated hand is recovered **only with open jaws**: recovery strokes them, so
        a Put (which starts holding a labware) is refused instead of dropping it.
        """
        plc = self._plc()
        h = self.motion.hand

        if not self._is_activated(self.hand_status()):
            if not self.hand_is_open():
                raise HandNotOpenError(
                    "Hand is deactivated (D6002=0x%04X) and not fully open; re-activating "
                    "would stroke the jaws and drop what is held. Put the labware down "
                    "first, then run ActivateHand." % self.hand_status()
                )
            self.activate_hand()

        self._kv(lambda: kvcomplus.write_word(plc, DM, HAND_TGT, int(target)))
        self._kv(lambda: kvcomplus.write_word(plc, DM, HAND_SPEED, h.speed))
        self._kv(lambda: kvcomplus.write_word(plc, DM, HAND_FORCE, h.grip_force))
        self._kv(lambda: kvcomplus.write_bit(plc, DM, HAND_WORD, BIT_HAND_MOVE, True))
        try:
            # phase A: wait for motion to start (done bit 7 -> 0)
            t0 = time.time()
            while time.time() - t0 <= _HAND_START_TIMEOUT_S:
                if (self.hand_status() >> 7) & 1 == 0:
                    break
                time.sleep(0.3)
            # phase B: wait for completion (done bit 7 -> 1); sample the grip bit there
            grip_bit = 1
            t1 = time.time()
            while True:
                st = self.hand_status()
                if (st >> 7) & 1 == 1:
                    grip_bit = (st >> 6) & 1
                    break
                if time.time() - t1 > h.move_timeout_s:
                    raise HandFailed("hand move did not complete (D6002.7)")
                time.sleep(0.3)
        finally:
            # The trigger needs a 0 -> 1 edge next time, and a stale 1 makes the following
            # move report the previous completion. Always drop it.
            self._kv(lambda: kvcomplus.write_bit(plc, DM, HAND_WORD, BIT_HAND_MOVE, False))
        return grip_bit

    def require_hand_open(self, why: str) -> int:
        """Raise unless the jaws are fully open; return the position read."""
        pos = self.hand_position()
        open_pos = self.motion.hand.open_position
        if abs(pos - open_pos) > HAND_OPEN_TOL:
            raise HandNotOpenError(f"Hand is at {pos} (open={open_pos}); {why}.")
        return pos

    # ---- carriage ----
    def move_carriage(
        self,
        target_mm: int,
        phase: Phase = no_phase,
        report: Optional[Callable[[int], None]] = None,
    ) -> int:
        """Move the carriage to ``target_mm`` and return the position reached.

        Gated on the robot being at a movable pose and the carriage being ready and
        fault-free. The move-request bit is always cleared afterwards.
        """
        plc = self._plc()
        car = self.motion.carriage
        if not (car.range_min_mm <= target_mm <= car.range_max_mm):
            raise StationUnknown(
                f"target {target_mm} mm is outside the travel range "
                f"{car.range_min_mm}..{car.range_max_mm}."
            )
        if not self.at_movable_pose():
            raise NotInMovablePose(
                "Robot is at none of the base, retract, inverse-base, or "
                "inverse-retract poses; carriage move refused."
            )

        fault = self._kv(lambda: kvcomplus.read_word(plc, DM, ADDR_FAULT))
        if fault != 0:
            raise CarriageFaultError(f"D6005 fault bits = 0x{fault:04X}.")
        status = self._kv(lambda: kvcomplus.read_word(plc, DM, ADDR_DONE_WORD))
        if (status >> BIT_DONE) & 1 != 1 or (status >> BIT_MOVING) & 1 == 1:
            raise CarriageNotReadyError("Carriage not ready (D6000.0 off or D6000.1 on).")

        self._kv(lambda: kvcomplus.write_dword(plc, DM, ADDR_TGT, target_mm))
        self._kv(lambda: kvcomplus.write_dword(plc, DM, ADDR_SPEED, car.default_speed_mm_s))
        self._kv(lambda: kvcomplus.write_dword(plc, DM, ADDR_ACCEL, car.accel_mm_s_ms))
        self._kv(lambda: kvcomplus.write_bit(plc, DM, ADDR_REQ_WORD, BIT_MOVE_REQ, True))
        phase(f"carriage -> {target_mm} mm")

        def tick() -> int:
            pos = self.carriage_position()
            if report is not None:
                report(pos)
            return pos

        try:
            # Phase A: wait for the PLC to acknowledge the start (moving on, or complete
            # off) -- until then the status still reads "at rest".
            started = False
            t0 = time.time()
            while time.time() - t0 <= _CARRIAGE_START_TIMEOUT_S:
                st = self._kv(lambda: kvcomplus.read_word(plc, DM, ADDR_DONE_WORD))
                tick()
                if (st >> BIT_MOVING) & 1 == 1 or (st >> BIT_DONE) & 1 == 0:
                    started = True
                    break
                time.sleep(car.poll_interval_s)
            if not started:
                raise CarriageMoveTimedOut(
                    f"PLC did not acknowledge the move within {_CARRIAGE_START_TIMEOUT_S:g}s."
                )

            # Phase B: wait for completion (moving off and complete on).
            t1 = time.time()
            while True:
                st = self._kv(lambda: kvcomplus.read_word(plc, DM, ADDR_DONE_WORD))
                tick()
                if (st >> BIT_MOVING) & 1 == 0 and (st >> BIT_DONE) & 1 == 1:
                    break
                if time.time() - t1 > car.move_timeout_s:
                    raise CarriageMoveTimedOut(
                        f"Move did not complete within {car.move_timeout_s:g}s."
                    )
                time.sleep(car.poll_interval_s)
        finally:
            self._kv(lambda: kvcomplus.write_bit(plc, DM, ADDR_REQ_WORD, BIT_MOVE_REQ, False))

        return self.carriage_position()

    # ---- named poses ----
    def run_to_pose(self, pose: PoseConfig, name: str, phase: Phase = no_phase) -> None:
        """Run the task that reaches ``pose`` and verify the arm got there."""
        phase(f"to {name}: RunTask({pose.task})")
        self.run_task(pose.task)
        phase(f"verify {name} pose")
        if not pose.matches(self.joint_angles()):
            raise PoseNotReached(f"Robot did not reach the {name} pose after RunTask({pose.task}).")

    def set_orientation(self, direction: str, phase: Phase = no_phase) -> PoseConfig:
        """Turn the arm to face ``direction``, keeping the pose family. Returns the target.

        A no-op (still verified) when the arm already faces that way.
        """
        target = self.motion.orientation_target(self.joint_angles(), direction)
        if target is None:
            raise NotAtKnownPose(
                "Robot is at none of the base/retract/inverse-base/inverse-retract poses."
            )
        name = ("inverse " if direction == "reverse" else "") + (
            "base" if target in (self.motion.base_pose, self.motion.inverse_base_pose) else "retract"
        )
        phase(f"turning {direction}: RunTask({target.task})")
        self.run_task(target.task)
        phase("verify target pose")
        if not target.matches(self.joint_angles()):
            raise PoseNotReached(f"Robot did not reach the {name} pose after RunTask({target.task}).")
        return target

    def face(self, direction: str, phase: Phase = no_phase) -> None:
        """Turn the arm to face ``direction`` only if it does not already."""
        angles = self.joint_angles()
        if self.motion.orientation_target(angles, direction) is None:
            raise NotAtKnownPose(
                "Robot is at none of the base/retract/inverse-base/inverse-retract poses."
            )
        facing_reverse = (
            self.motion.inverse_base_pose.matches(angles)
            or self.motion.inverse_retract_pose.matches(angles)
        )
        if facing_reverse == (direction == "reverse"):
            return
        self.set_orientation(direction, phase)

    def return_home(self, phase: Phase = no_phase) -> None:
        """Park the arm at the base pose from any of the four known poses."""
        steps = self.motion.home_path(self.joint_angles())
        if steps is None:
            raise NotAtKnownPose(
                "Robot is at none of the base/retract/inverse-base/inverse-retract poses."
            )
        if not steps:
            phase("already at the base pose")
            return
        for task, target, name in steps:
            phase(f"to {name}: RunTask({task})")
            self.run_task(task)
            phase(f"verify {name} pose")
            if not target.matches(self.joint_angles()):
                raise PoseNotReached(f"Robot did not reach the {name} pose after RunTask({task}).")

    # ---- pick / put ----
    def pick(
        self,
        station_id: str,
        station: StationConfig,
        phase: Phase = no_phase,
        verify_grasp: bool = True,
    ) -> bool:
        """Pick the labware at ``station``; return True if the arm ended at retract.

        Starts at the direction's base **or** retract pose -- from the base pose the arm
        moves to the retract pose first, because that is where the approach task runs
        from. Requires the hand fully open.
        """
        base_like, retract_like = self.motion.poses_for(station.direction)
        base_name, retract_name = self.pose_names(station.direction)

        self.require_hand_open("must be fully open to pick")

        angles = self.joint_angles()
        start_at_retract = retract_like.matches(angles)
        if not start_at_retract and not base_like.matches(angles):
            raise WrongStartPose(
                f"Robot is at neither the {base_name} nor the {retract_name} pose "
                f"({station.direction} station); pick refused."
            )

        phase(f"start (station {station_id}, {station.direction})")
        if not start_at_retract:
            self.run_to_pose(retract_like, retract_name, phase)

        phase(f"approach: RunTask({station.pick_script_a})")
        self.run_task(station.pick_script_a)

        phase("chuck: closing hand")
        # The close target depends on the grip orientation (long -> not fully closed).
        grip_bit = self.hand_move(self.motion.hand.closed_position_for(station.grip))
        if verify_grasp and grip_bit == 1:
            raise GraspFailedError(
                "No labware grasped (hand reached the commanded close position; "
                "grip bit D6002.6=1)."
            )

        phase(f"retract: RunTask({station.pick_script_b})")
        self.run_task(station.pick_script_b)

        phase(f"verify {retract_name} pose")
        at_retract = retract_like.matches(self.joint_angles())
        if not at_retract:
            raise PoseNotReached(
                f"Robot did not return to the {retract_name} pose after pick-retract."
            )
        return at_retract

    def put(
        self,
        station_id: str,
        station: StationConfig,
        phase: Phase = no_phase,
        verify_grasp: bool = True,
    ) -> bool:
        """Place the held labware at ``station``; return True if the arm ended at retract.

        Starts and ends at the direction's retract pose -- no return home, so a pick at
        this station can follow immediately.
        """
        retract_like = self.motion.poses_for(station.direction)[1]
        _, retract_name = self.pose_names(station.direction)

        if not retract_like.matches(self.joint_angles()):
            raise WrongStartPose(
                f"Robot is not at the {retract_name} pose ({station.direction} station); "
                "put refused."
            )

        # Position-based, and only meaningful for a hand already closed on something:
        # there is no fresh chuck here to read a grip bit from, and no state is carried
        # over from the pick.
        if verify_grasp:
            hand_pos = self.hand_position()
            if hand_pos > self.motion.hand.open_position - GRASP_MARGIN:
                raise GraspFailedError(
                    f"Hand is not holding a labware (D6060={hand_pos} ~ open); nothing to put."
                )

        phase(f"start (station {station_id}, {station.direction})")
        phase(f"approach: RunTask({station.put_script_a})")
        self.run_task(station.put_script_a)

        phase("unchuck: opening hand")
        self.hand_move(self.motion.hand.open_position)

        phase(f"retract: RunTask({station.put_script_b})")
        self.run_task(station.put_script_b)

        phase(f"verify {retract_name} pose")
        at_retract = retract_like.matches(self.joint_angles())
        if not at_retract:
            raise PoseNotReached(
                f"Robot did not return to the {retract_name} pose after put-retract."
            )
        return at_retract

    # ---- transfer ----
    def transfer(
        self,
        source_id: str,
        destination_id: str,
        phase: Phase = no_phase,
        verify_grasp: bool = True,
    ) -> "tuple[int, bool]":
        """Carry a labware between stations; return (carriage position, at retract pose).

        Drives the whole route, so the carriage need not start at the source: move there,
        face the station, pick, move to the destination, face it, put. The arm must start
        at one of the four known poses and the hand must be open. Each pick and put is
        given its station explicitly rather than resolving one from the carriage position,
        so a carriage that has not settled to the exact millimetre cannot make this fail.
        """
        source_id, source = self.station_by_name(source_id)
        destination_id, destination = self.station_by_name(destination_id)

        if not self.at_movable_pose():
            raise NotAtKnownPose(
                "Robot is at none of the base/retract/inverse-base/inverse-retract poses; "
                "transfer refused."
            )
        self.require_hand_open("must be fully open to start a transfer")

        phase(f"transfer {source_id} -> {destination_id}")
        if self.carriage_position() != source.position_mm:
            self.move_carriage(source.position_mm, phase)
        self.face(source.direction, phase)
        self.pick(source_id, source, phase, verify_grasp)

        # The turn happens at the destination, holding the labware: verified on the real
        # machine in both directions, and it keeps every turn at a station position.
        self.move_carriage(destination.position_mm, phase)
        self.face(destination.direction, phase)
        at_retract = self.put(destination_id, destination, phase, verify_grasp)
        return self.carriage_position(), at_retract
