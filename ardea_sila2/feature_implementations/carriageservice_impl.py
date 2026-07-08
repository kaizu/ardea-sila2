"""CarriageService implementation (Ardea-specific).

Moves the travel carriage over the KEYENCE PLC (reusing the kvcomplus atomic
primitives) and streams its position. A move is gated on the robot being at a
movable pose (base or retract) and the carriage being ready, mirrors the proven
move sequence (write target/speed/accel -> raise move-request -> wait -> always
clear the request bit), and reports the live position via intermediate responses
and the observable CarriagePosition property.

Signal addresses are from the Ardea signal proposal (orchestration_design.md §3):
all DM (device type 18); 2-word values are signed 32-bit (low @ addr, high @ addr+1).
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from bcap_sila2.bcap import RobotUnavailableError, get_joint_angles
from orinexception import ORiNException
from sila2.server import MetadataDict, ObservableCommandInstanceWithIntermediateResponses

from kvcomplus_sila2 import kvcomplus

from ..generated.carriageservice import (
    CarriageFault,
    CarriageNotReady,
    CarriageServiceBase,
    ControllerConnectionError,
    InvalidStation,
    MoveCarriage_IntermediateResponses,
    MoveCarriage_Responses,
    MoveTimeout,
    PlcAccessError,
    PlcConnectionError,
    RobotAccessError,
    RobotNotInMovablePose,
)

if TYPE_CHECKING:
    from ..server import Server

# --- device addresses (KV-8000, DM = type 18) ---
DM = 18
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

_START_TIMEOUT_S = 10.0    # max wait for the PLC to acknowledge the move start


class CarriageServiceImpl(CarriageServiceBase):
    def __init__(self, parent_server: Server) -> None:
        super().__init__(parent_server=parent_server)
        self._poller_started = False
        self._poller_lock = threading.Lock()

    # ---- helpers ----
    def _plc(self):
        return self.parent_server.config.plc

    @staticmethod
    def _kv(fn):
        """Run a kvcomplus call, mapping KvComError to the right SiLA error."""
        try:
            return fn()
        except kvcomplus.KvComError as e:
            msg = str(e).lower()
            if "bridge" in msg or "connect" in msg or "timed out" in msg:
                raise PlcConnectionError(str(e))
            raise PlcAccessError(str(e))

    def _robot_in_movable_pose(self) -> bool:
        cfg = self.parent_server.config.controller
        try:
            angles = get_joint_angles(cfg)
        except OSError as e:
            raise ControllerConnectionError(str(e))
        except (ORiNException, RobotUnavailableError) as e:
            raise RobotAccessError(str(e))
        motion = self.parent_server.motion
        return motion.base_pose.matches(angles) or motion.retract_pose.matches(angles)

    # ---- observable property: CarriagePosition ----
    def CarriagePosition_on_subscription(self, *, metadata: MetadataDict):
        self._ensure_poller()
        return None  # use the default producer queue

    def _ensure_poller(self) -> None:
        with self._poller_lock:
            if self._poller_started:
                return
            self._poller_started = True
        threading.Thread(target=self._poll_loop, name="carriage-pos-poll", daemon=True).start()

    def _poll_loop(self) -> None:
        plc = self._plc()
        poll = self.parent_server.motion.carriage.poll_interval_s
        while True:
            try:
                pos = kvcomplus.read_dword(plc, DM, ADDR_CUR_POS)
                self.update_CarriagePosition(float(pos))
            except Exception:
                # transient (PLC blip / bridge restart); try again next tick
                pass
            time.sleep(poll)

    # ---- observable command: MoveCarriage ----
    def MoveCarriage(
        self,
        StationId: str,
        *,
        metadata: MetadataDict,
        instance: ObservableCommandInstanceWithIntermediateResponses[MoveCarriage_IntermediateResponses],
    ) -> MoveCarriage_Responses:
        plc = self._plc()
        car = self.parent_server.motion.carriage

        # StationId -> target mm (temporary: parse as int mm directly; see design Q6)
        try:
            target = int(str(StationId).strip())
        except (TypeError, ValueError):
            raise InvalidStation(f"StationId {StationId!r} is not an integer.")
        if not (car.range_min_mm <= target <= car.range_max_mm):
            raise InvalidStation(
                f"target {target} mm is outside the travel range {car.range_min_mm}..{car.range_max_mm}."
            )

        # One motion at a time (shared robot/carriage OperationCoordinator).
        with self.parent_server.operation_lock:
            # Pose gate: robot must be at the base or retract pose.
            if not self._robot_in_movable_pose():
                raise RobotNotInMovablePose(
                    "Robot is at neither the base nor the retract pose; carriage move refused."
                )

            # Carriage must be ready (complete on, not moving) and fault-free.
            fault = self._kv(lambda: kvcomplus.read_word(plc, DM, ADDR_FAULT))
            if fault != 0:
                raise CarriageFault(f"D6005 fault bits = 0x{fault:04X}.")
            status = self._kv(lambda: kvcomplus.read_word(plc, DM, ADDR_DONE_WORD))
            if (status >> BIT_DONE) & 1 != 1 or (status >> BIT_MOVING) & 1 == 1:
                raise CarriageNotReady("Carriage not ready (D6000.0 off or D6000.1 on).")

            start_pos = self._kv(lambda: kvcomplus.read_dword(plc, DM, ADDR_CUR_POS))

            # Write target / speed / accel (each an atomic 2-word write).
            self._kv(lambda: kvcomplus.write_dword(plc, DM, ADDR_TGT, target))
            self._kv(lambda: kvcomplus.write_dword(plc, DM, ADDR_SPEED, car.default_speed_mm_s))
            self._kv(lambda: kvcomplus.write_dword(plc, DM, ADDR_ACCEL, car.accel_mm_s_ms))

            # Raise the move-request bit (atomic RMW; preserves D5000.1 etc.).
            self._kv(lambda: kvcomplus.write_bit(plc, DM, ADDR_REQ_WORD, BIT_MOVE_REQ, True))
            instance.begin_execution()

            def report(pos: int) -> None:
                span = target - start_pos
                if span:
                    instance.progress = max(0.0, min(1.0, (pos - start_pos) / span))
                instance.send_intermediate_response(
                    MoveCarriage_IntermediateResponses(CurrentPosition=float(pos))
                )

            try:
                # Phase A: wait for the PLC to acknowledge the start (moving on, or
                # complete off) — until then the status still reads "at rest".
                started = False
                t0 = time.time()
                while time.time() - t0 <= _START_TIMEOUT_S:
                    st = self._kv(lambda: kvcomplus.read_word(plc, DM, ADDR_DONE_WORD))
                    report(self._kv(lambda: kvcomplus.read_dword(plc, DM, ADDR_CUR_POS)))
                    if (st >> BIT_MOVING) & 1 == 1 or (st >> BIT_DONE) & 1 == 0:
                        started = True
                        break
                    time.sleep(car.poll_interval_s)
                if not started:
                    raise MoveTimeout(
                        f"PLC did not acknowledge the move within {_START_TIMEOUT_S:g}s."
                    )

                # Phase B: wait for completion (moving off and complete on).
                t1 = time.time()
                while True:
                    st = self._kv(lambda: kvcomplus.read_word(plc, DM, ADDR_DONE_WORD))
                    pos = self._kv(lambda: kvcomplus.read_dword(plc, DM, ADDR_CUR_POS))
                    report(pos)
                    if (st >> BIT_MOVING) & 1 == 0 and (st >> BIT_DONE) & 1 == 1:
                        break
                    if time.time() - t1 > car.move_timeout_s:
                        raise MoveTimeout(f"Move did not complete within {car.move_timeout_s:g}s.")
                    time.sleep(car.poll_interval_s)
            finally:
                # Always clear the move-request bit.
                self._kv(lambda: kvcomplus.write_bit(plc, DM, ADDR_REQ_WORD, BIT_MOVE_REQ, False))

            final = self._kv(lambda: kvcomplus.read_dword(plc, DM, ADDR_CUR_POS))
            instance.progress = 1.0
            return MoveCarriage_Responses(FinalPosition=float(final))
