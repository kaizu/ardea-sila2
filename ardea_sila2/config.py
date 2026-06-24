"""Server configuration for the Ardea SiLA2 server.

The Ardea device is driven through two providers at once:

- the DENSO robot controller over b-CAP (reusing ``bcap_sila2``), configured by
  the ``[controller]`` and ``[task]`` sections;
- the KEYENCE PLC over KV COM+ (reusing ``kvcomplus_sila2``), configured by the
  ``[plc]`` section.

The connection dataclasses are reused verbatim from the provider packages so
their feature implementations — which read ``self.parent_server.config.controller``,
``.task`` and ``.plc`` — work unchanged when registered on the Ardea server.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bcap_sila2.config import ControllerConfig, TaskConfig
from kvcomplus_sila2.config import PlcConfig


class ConfigError(Exception):
    """Raised when the configuration file is missing, invalid, or incomplete."""


@dataclass
class ServerConfig:
    """SiLA server listening settings."""

    host: str = "0.0.0.0"
    port: int = 50053


@dataclass
class Config:
    controller: ControllerConfig  # b-CAP / DENSO robot
    plc: PlcConfig                 # KV COM+ / KEYENCE PLC
    task: TaskConfig = field(default_factory=TaskConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def _build(cls: type, data: dict[str, Any], section: str) -> Any:
    """Construct a config dataclass, rejecting unknown keys to catch typos."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in [{section}]: {', '.join(sorted(unknown))}"
        )
    return cls(**data)


def load_config(path: str | Path) -> Config:
    """Load and validate the server configuration from a TOML file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    with path.open("rb") as f:
        data: dict[str, Any] = tomllib.load(f)

    # [controller] — b-CAP robot (host/port required, like bcap-sila2)
    controller_data = data.get("controller")
    if not isinstance(controller_data, dict):
        raise ConfigError("Missing required [controller] section in configuration file.")
    for key in ("host", "port"):
        if key not in controller_data:
            raise ConfigError(f"Missing required [controller].{key} in configuration file.")

    # [plc] — KV COM+ PLC (peer/plc_id required, like kvcomplus-sila2)
    plc_data = data.get("plc")
    if not isinstance(plc_data, dict):
        raise ConfigError("Missing required [plc] section in configuration file.")
    for key in ("peer", "plc_id"):
        if key not in plc_data:
            raise ConfigError(f"Missing required [plc].{key} in configuration file.")

    controller = _build(ControllerConfig, controller_data, "controller")
    plc = _build(PlcConfig, plc_data, "plc")
    task = _build(TaskConfig, data.get("task", {}), "task")
    server = _build(ServerConfig, data.get("server", {}), "server")
    return Config(controller=controller, plc=plc, task=task, server=server)
