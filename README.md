# ardea-sila2

A **SiLA 2 server** for the **Ardea** device. Ardea is driven through two
providers at once, and this server exposes both:

- **DENSO robot controller** over ORiN b-CAP — reuses
  [bcap-sila2](https://github.com/kaizu/bcap-sila2): `VariableService`,
  `TaskService`
- **KEYENCE PLC** over the KV COM+ Library — reuses
  [kvcomplus-sila2](https://github.com/kaizu/kvcomplus-sila2): `DeviceService`,
  `ConnectionService`

Both provider packages are vendored as **git submodules** under `third_party/`
and installed editable, so their wrappers and features are reused directly
rather than duplicated. Ardea-specific orchestration features (e.g. the
travel-carriage move ⇔ robot-operation interlock) will be layered on top.

> ⚠️ **Disclaimer**: This software is under active development. It comes with
> **no warranty** and the authors accept **no liability** for any results or
> damages. Use at your own risk, and verify safety before connecting to real
> hardware.

## Requirements

Windows with the **KV COM+ Library (KV-DH1L)** installed (for the PLC provider),
Python >= 3.11, and [uv](https://docs.astral.sh/uv/).

## Setup

The provider packages (and b-CAP's own `orin_bcap` submodule) are nested
submodules, so clone recursively:

```bash
git clone --recurse-submodules git@github.com:kaizu/ardea-sila2.git
cd ardea-sila2
uv sync
```

Already cloned: `git submodule update --init --recursive && uv sync`.

## Configuration

```bash
cp config.example.toml config.toml
```

The Ardea config merges both providers: `[controller]` / `[task]` for the b-CAP
robot and `[plc]` for the KV COM+ PLC, plus `[server]` for the listen
address/port. See [config.example.toml](config.example.toml).

## Run the server

```bash
uv run python -m ardea_sila2 --config config.toml --insecure
```

Listen address/port come from `[server]`; override with `-a/--ip-address` and
`-p/--port`. See `--help` for TLS options.

## License

[MIT](LICENSE) © 2026 Kazunari Kaizu
