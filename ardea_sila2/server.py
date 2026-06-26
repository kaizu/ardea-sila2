"""Ardea SiLA2 server.

Exposes both providers at once by reusing the feature implementations from the
``bcap_sila2`` and ``kvcomplus_sila2`` packages (vendored as git submodules):

- b-CAP / DENSO robot: VariableService, TaskService, RobotService
- KV COM+ / KEYENCE PLC: DeviceService, ConnectionService

Ardea-specific orchestration features (e.g. the travel-carriage move ⇔ robot
interlock) will be added on top in a later step. The reused implementations read
their settings from ``self.parent_server.config`` (``.controller`` / ``.task``
for b-CAP, ``.plc`` for KV COM+), which the Ardea :class:`Config` provides.
"""

import logging
from typing import Optional
from uuid import UUID, uuid4

from sila2.server import SilaServer

# Reused b-CAP (DENSO robot) provider
from bcap_sila2.feature_implementations.robotservice_impl import RobotServiceImpl
from bcap_sila2.feature_implementations.taskservice_impl import TaskServiceImpl
from bcap_sila2.feature_implementations.variableservice_impl import VariableServiceImpl
from bcap_sila2.generated.robotservice import RobotServiceFeature
from bcap_sila2.generated.taskservice import TaskServiceFeature
from bcap_sila2.generated.variableservice import VariableServiceFeature

# Reused KV COM+ (KEYENCE PLC) provider
from kvcomplus_sila2 import kvcomplus
from kvcomplus_sila2.feature_implementations.connectionservice_impl import ConnectionServiceImpl
from kvcomplus_sila2.feature_implementations.deviceservice_impl import DeviceServiceImpl
from kvcomplus_sila2.generated.connectionservice import ConnectionServiceFeature
from kvcomplus_sila2.generated.deviceservice import DeviceServiceFeature

from .config import Config

logger = logging.getLogger(__name__)


class Server(SilaServer):
    def __init__(
        self,
        config: Config,
        server_uuid: Optional[UUID] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        # Settings read by the reused feature implementations via
        # ``self.parent_server.config`` (.controller/.task for b-CAP, .plc for KV COM+).
        self.config = config

        if name is None:
            name = "Ardea SiLA2 Server"
        if description is None:
            description = (
                "SiLA 2 server for the Ardea device. It drives the device through "
                "two providers at once: the DENSO robot controller over ORiN b-CAP "
                "(VariableService, TaskService) and the KEYENCE PLC over the KV COM+ "
                "Library (DeviceService, ConnectionService). The provider features "
                "are reused from the bcap_sila2 and kvcomplus_sila2 packages. "
                "Connection parameters are supplied at startup via a TOML "
                "configuration file ([controller], [task], [plc], [server])."
            )
        super().__init__(
            server_name=name,
            server_description=description,
            server_type="ArdeaSila2Server",
            server_version="0.1.0",
            server_vendor_url="https://example.com",
            server_uuid=server_uuid if server_uuid is not None else uuid4(),
        )

        # --- b-CAP / DENSO robot provider ---
        self.variableservice = VariableServiceImpl(self)
        self.set_feature_implementation(VariableServiceFeature, self.variableservice)

        self.taskservice = TaskServiceImpl(self)
        self.set_feature_implementation(TaskServiceFeature, self.taskservice)

        self.robotservice = RobotServiceImpl(self)
        self.set_feature_implementation(RobotServiceFeature, self.robotservice)

        # --- KV COM+ / KEYENCE PLC provider ---
        self.deviceservice = DeviceServiceImpl(self)
        self.set_feature_implementation(DeviceServiceFeature, self.deviceservice)

        self.connectionservice = ConnectionServiceImpl(self)
        self.set_feature_implementation(ConnectionServiceFeature, self.connectionservice)

        # Pre-warm the persistent KV COM+ connection so its cost is paid at
        # startup rather than on the first command. Non-fatal if the PLC is
        # unreachable now — operations reconnect lazily. (The KV COM+ side is
        # held open for the server lifetime; see kvcomplus_sila2.kvcomplus.)
        try:
            info = kvcomplus.connect(self.config.plc)
            logger.info("KV COM+ connected at startup: %s", info)
        except kvcomplus.KvComError as e:
            logger.warning(
                "KV COM+ not connected at startup (will retry on first command): %s", e
            )

    def stop(self, *args, **kwargs):
        # Shut down the 32-bit KV COM+ bridge subprocess (if it was ever started)
        # and release the held PLC connection, after the gRPC server has stopped
        # accepting commands. b-CAP needs no teardown (pure-Python, per-op socket).
        try:
            return super().stop(*args, **kwargs)
        finally:
            kvcomplus.shutdown()
