"""LabwareService implementation (Ardea-specific): PickLabware / PutLabware.

Both commands resolve a station from the current carriage position (motion
``station_at``; no StationId argument yet), then run that station's task pair
around a hand action. The poses used depend on the station's ``direction``:
forward uses the base/retract poses, reverse uses the 180°-turned inverse
base/retract poses (motion ``poses_for``):
Each operation uses its own task pair from the station (Pick: pick_script_a/_b,
Put: put_script_a/_b), so placing and taking can follow different trajectories:
  PickLabware (robot must start at the direction's base OR retract pose):
    [base pose only: RunTask(retract pose task) -> confirm retract]
    -> approach (pick_script_a) -> close hand (chuck) -> retract (pick_script_b)
    -> confirm retract.
  PutLabware (robot must start at the direction's retract pose):
    approach (put_script_a) -> open hand (unchuck) -> retract (put_script_b)
    -> confirm retract.

**Both commands end at the direction's retract pose, and both approach tasks run from
it.** Put deliberately does NOT return home: a Pick following a Put would only have to
undo that move, so the pair used to swing base->retract->base once per transport for
nothing. The base pose is now only the power-on/parking pose; ``[common].return_home``
and ``return_home_reverse`` stay in the motion config for an explicit park command.
Consequently a Pick that starts at the base pose moves to the retract pose first.

Reuses bcap task/pose helpers and the kvcomplus atomic primitives; holds the
server OperationCoordinator for the whole sequence so no carriage move runs
concurrently. The D5000.1 carriage-lockout is NOT used (it breaks b-CAP RunTask;
see note below).

Grasp verification (only meaningful when gripping, i.e. at Pick time):
- PickLabware uses the grip bit D6002.6 sampled at the chuck's completion instant
  (returned by ``_hand_move``): 0 = the jaws stopped short of the commanded close
  (an object is held), 1 = the jaws reached the commanded position (empty). This is
  robust for both short- and long-edge grips and does not depend on the hand
  springing back to open (which, on the real hand, it does not reliably do).
- PutLabware only sanity-checks, before opening, that the hand is not (near) fully
  open, i.e. it is closed on something. This stays position-based (<= open_position -
  _GRASP_MARGIN): at Put time there is no fresh chuck to read a grip bit from, and no
  per-command state is carried over from a previous Pick.
Either raises GraspFailed. Both checks are skipped entirely when the server is
started with --skip-grasp-check (``parent_server.verify_grasp`` is False).

Three utilities for setup and debugging live here too, rather than in features of their
own: ``MoveHand`` (drive the gripper to a position in device units, speed and force from
the motion config), ``ActivateHand`` (the D5002.0 OFF->ON recovery toggle), and
``ToggleLight``/``LightIsOn`` (the machine light -- a boolean variable on the **robot
controller**, read and written over b-CAP; it is not a PLC signal).
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from bcap_sila2.bcap import (
    RobotUnavailableError,
    TaskAbnormalStopError,
    TaskTimeoutError,
    get_joint_angles,
    read_variable,
    run_task,
    write_variable,
)
from orinexception import ORiNException
from sila2.server import MetadataDict, ObservableCommandInstanceWithIntermediateResponses

from kvcomplus_sila2 import kvcomplus

from ..generated.labwareservice import (
    ActivateHand_IntermediateResponses,
    ActivateHand_Responses,
    ControllerConnectionError,
    GraspFailed,
    HandError,
    HandNotOpen,
    InvalidHandPosition,
    LabwareServiceBase,
    MoveHand_IntermediateResponses,
    MoveHand_Responses,
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
    ToggleLight_Responses,
    VariableAccessError,
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
_HAND_ACTIVATE_OFF_S = 1.0     # how long D5002.0 stays OFF before the recovering edge
_HAND_OPEN_TOL = 3            # tolerance [units] for "hand fully open" check
# PutLabware precondition only: the hand is "closed on something" if its position is
# not (near) fully open, i.e. current position <= open_position - _GRASP_MARGIN.
# (PickLabware no longer uses position; it uses the grip bit from _hand_move.)
_GRASP_MARGIN = 10           # margin [units] below open_position that still counts as "holding"
_ONE_CYCLE = 1                 # RunTask mode: run once and stop


class LabwareServiceImpl(LabwareServiceBase):
    def __init__(self, parent_server: Server) -> None:
        super().__init__(parent_server=parent_server)
        # Guards only the light's read-modify-write toggle. Deliberately not the server's
        # operation_lock: switching a light is not motion and must not wait for a pick.
        self._light_lock = threading.Lock()

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

    @staticmethod
    def _is_activated(status: int) -> bool:
        """True if D6002 shows the activated state (.0 and .4 and .5, i.e. 0x0031)."""
        return bool(status & 1 and (status >> 4) & 1 and (status >> 5) & 1)

    def _hand_open(self) -> bool:
        """True if the jaws read as fully open (nothing held)."""
        pos = self._kv(lambda: kvcomplus.read_word(self._plc(), DM, HAND_CUR_POS))
        return abs(pos - self.parent_server.motion.hand.open_position) <= _HAND_OPEN_TOL

    def _activate_hand(self) -> None:
        """Drive D5002.0 OFF then ON and wait for the activated state.

        The falling edge is the point: with D5002.0 already ON but the hand stuck at
        D6002=0x0000/0x0001, writing ON again does nothing (2026-07-24, again 2026-08-17).
        **This strokes the jaws** (140 -> ~10 -> 140 measured 2026-08-14), so every caller
        must have checked that they are open -- a held labware would be dropped.
        """
        plc = self._plc()
        h = self.parent_server.motion.hand
        self._kv(lambda: kvcomplus.write_bit(plc, DM, HAND_WORD, BIT_HAND_ACTIVE, False))
        time.sleep(_HAND_ACTIVATE_OFF_S)
        self._kv(lambda: kvcomplus.write_bit(plc, DM, HAND_WORD, BIT_HAND_ACTIVE, True))
        t0 = time.time()
        while not self._is_activated(self._hand_status()):
            if time.time() - t0 > h.move_timeout_s:
                raise HandError("hand activation did not complete (D6002.0/.4/.5)")
            time.sleep(0.3)

    def _hand_move(self, target: int) -> int:
        """Move the hand to ``target`` (activate if needed, write params, wait done).

        Used for both chuck (target=closed) and unchuck (target=open). Returns the
        in-position/grip bit D6002.6 sampled at the completion instant (bit7->1):
        0 = stopped short (gripping an object), 1 = reached the commanded position.

        A deactivated hand is recovered **only with open jaws**: recovery strokes them,
        so a Put (which starts holding a labware) is refused instead of dropping it.
        """
        plc = self._plc()
        h = self.parent_server.motion.hand

        if not self._is_activated(self._hand_status()):
            if not self._hand_open():
                raise HandNotOpen(
                    "Hand is deactivated (D6002=0x%04X) and not fully open; re-activating "
                    "would stroke the jaws and drop what is held. Put the labware down "
                    "first, then run ActivateHand." % self._hand_status()
                )
            self._activate_hand()

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

            # Pose gate: PickLabware accepts either of the direction's two known poses.
            # The approach task runs from the retract pose, so a base-pose start is moved
            # there first; a retract-pose start (where a preceding Put left the arm) goes
            # straight in.
            angles = self._joint_angles()
            start_at_retract = retract_like.matches(angles)
            if not start_at_retract and not base_like.matches(angles):
                raise RobotNotAtBasePose(
                    f"Robot is at neither the {base_name} nor the {retract_name} pose "
                    f"({station.direction} station); pick refused."
                )

            instance.begin_execution()
            phase(f"start (station {station_id}, {station.direction})")

            if not start_at_retract:
                # Base pose -> retract pose. PoseConfig.task is the PacScript that reaches
                # the pose; it needs the hand open, which the precondition above checked.
                phase(f"to {retract_name}: RunTask({retract_like.task})")
                self._run_task(retract_like.task)
                if not retract_like.matches(self._joint_angles()):
                    raise PoseNotRestored(
                        f"Robot did not reach the {retract_name} pose before the pick approach."
                    )

            phase(f"approach: RunTask({station.pick_script_a})")
            self._run_task(station.pick_script_a)

            phase("chuck: closing hand")
            # Close target depends on the station's grip orientation (long -> not fully closed).
            # _hand_move returns the grip bit D6002.6 sampled at completion: 0 = jaws
            # stopped short (holding an object), 1 = reached the commanded close (empty).
            grip_bit = self._hand_move(motion.hand.closed_position_for(station.grip))
            if self.parent_server.verify_grasp and grip_bit == 1:
                raise GraspFailed(
                    "No labware grasped (hand reached the commanded close position; "
                    "grip bit D6002.6=1)."
                )

            phase(f"retract: RunTask({station.pick_script_b})")
            self._run_task(station.pick_script_b)

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

            # Poses depend on the station's facing (forward -> retract, reverse ->
            # inverse retract). Put starts and ends at the retract-like pose; the
            # base-like pose plays no part in it.
            retract_like = motion.poses_for(station.direction)[1]
            retract_name = "inverse retract" if station.direction == "reverse" else "retract"

            # Precondition: the robot must be at the direction's retract pose (start pose).
            angles = self._joint_angles()
            if not retract_like.matches(angles):
                raise RobotNotAtRetractPose(
                    f"Robot is not at the {retract_name} pose ({station.direction} station); put refused."
                )

            # Precondition (position-based; only meaningful for a hand already closed
            # on something — there is no fresh chuck here to read a grip bit from, and
            # no state is carried from a previous Pick): the hand must not be (near)
            # fully open. Skipped when grasp verification is disabled.
            h = motion.hand
            if self.parent_server.verify_grasp:
                hand_pos = self._kv(lambda: kvcomplus.read_word(self._plc(), DM, HAND_CUR_POS))
                if hand_pos > h.open_position - _GRASP_MARGIN:
                    raise GraspFailed(
                        f"Hand is not holding a labware (D6060={hand_pos} ~ open); nothing to put."
                    )

            instance.begin_execution()
            phase(f"start (station {station_id}, {station.direction})")

            phase(f"approach: RunTask({station.put_script_a})")
            self._run_task(station.put_script_a)

            phase("unchuck: opening hand")
            self._hand_move(motion.hand.open_position)

            phase(f"retract: RunTask({station.put_script_b})")
            self._run_task(station.put_script_b)

            # The put ends here, at the retract pose. No return-home: the arm stays where
            # the next PickLabware at this station wants it (see the module docstring).
            phase(f"verify {retract_name} pose")
            at_retract = retract_like.matches(self._joint_angles())
            if not at_retract:
                raise PoseNotRestored(f"Robot did not return to the {retract_name} pose after put-retract.")

            instance.progress = 1.0
            return PutLabware_Responses(AtRetractPose=at_retract)

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
        with self.parent_server.operation_lock:
            instance.begin_execution()
            phase(f"moving hand to {target} (speed {h.speed}, force {h.grip_force})")
            grip_bit = self._hand_move(target)
            reached = self._kv(lambda: kvcomplus.read_word(self._plc(), DM, HAND_CUR_POS))
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

        with self.parent_server.operation_lock:
            # Activation strokes the jaws, so refuse unless they are empty and open.
            hand_pos = self._kv(lambda: kvcomplus.read_word(self._plc(), DM, HAND_CUR_POS))
            open_pos = self.parent_server.motion.hand.open_position
            if abs(hand_pos - open_pos) > _HAND_OPEN_TOL:
                raise HandNotOpen(
                    f"Hand is at {hand_pos} (open={open_pos}); activation strokes the jaws, "
                    "so it is refused unless they are fully open."
                )

            instance.begin_execution()
            phase("toggling D5002.0 OFF -> ON (the jaws will stroke)")
            self._activate_hand()
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
