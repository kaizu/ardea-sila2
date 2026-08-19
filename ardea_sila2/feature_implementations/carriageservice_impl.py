"""CarriageService implementation (Ardea-specific).

Moves the travel carriage and streams its position. The move itself -- the pose gate, the
readiness checks and the proven write/wait/always-clear sequence -- lives in
:mod:`ardea_sila2.ops` so that LabwareService.Transfer can perform the same move as part
of a larger command; see that module's docstring. This one adds what is SiLA-specific:
the OperationCoordinator, the intermediate position reports, and the mapping from the
operations' neutral exceptions to this feature's errors.

Signal addresses are from the Ardea signal proposal (orchestration_design.md §3):
all DM (device type 18); 2-word values are signed 32-bit (low @ addr, high @ addr+1).
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Type

from sila2.server import MetadataDict, ObservableCommandInstanceWithIntermediateResponses

from kvcomplus_sila2 import kvcomplus

from .. import ops
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

DM = 18                    # device type for DM/D devices
ADDR_CUR_POS = ops.ADDR_CUR_POS   # D6010 : current position [mm] (2 words), for the poller

_OP_ERRORS: dict[Type[ops.OpError], Type[Exception]] = {
    ops.PlcConnectionFailed: PlcConnectionError,
    ops.PlcAccessFailed: PlcAccessError,
    ops.ControllerConnectionFailed: ControllerConnectionError,
    ops.RobotAccessFailed: RobotAccessError,
    ops.NotInMovablePose: RobotNotInMovablePose,
    ops.CarriageNotReadyError: CarriageNotReady,
    ops.CarriageFaultError: CarriageFault,
    ops.CarriageMoveTimedOut: MoveTimeout,
    ops.StationUnknown: InvalidStation,
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


class CarriageServiceImpl(CarriageServiceBase):
    def __init__(self, parent_server: Server) -> None:
        super().__init__(parent_server=parent_server)
        self._poller_started = False
        self._poller_lock = threading.Lock()

    # ---- helpers ----
    def _plc(self):
        return self.parent_server.config.plc

    # ---- unobservable property: StationNames ----
    def get_StationNames(self, *, metadata: MetadataDict) -> list[str]:
        # The station table comes from the motion config loaded at startup; the set
        # of names is fixed for the server lifetime. Ordering is not significant.
        return list(self.parent_server.motion.stations.keys())

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
        # StationId -> target mm (temporary: parse as int mm directly; see design Q6)
        try:
            target = int(str(StationId).strip())
        except (TypeError, ValueError):
            raise InvalidStation(f"StationId {StationId!r} is not an integer.")

        # One motion at a time (shared robot/carriage OperationCoordinator).
        with self.parent_server.operation_lock, _sila_errors():
            start_pos = self.parent_server.ops.carriage_position()
            instance.begin_execution()

            def report(pos: int) -> None:
                span = target - start_pos
                if span:
                    instance.progress = max(0.0, min(1.0, (pos - start_pos) / span))
                instance.send_intermediate_response(
                    MoveCarriage_IntermediateResponses(CurrentPosition=float(pos))
                )

            final = self.parent_server.ops.move_carriage(target, report=report)
            instance.progress = 1.0
            return MoveCarriage_Responses(FinalPosition=float(final))
