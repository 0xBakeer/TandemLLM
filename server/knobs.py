"""The knobs the conductor and the proxy read (#108), with or without the engine installed.

A box that runs the engine reads them through engine/settings.py's registry, as every other read does.
The standalone proxy (`tandem-dash`, its image) ships `proxy` and `server` only: there the same names
are read from the environment (prefix QWEN38_) with the defaults below, which
tests/test_knobs_standalone.py holds equal to the registry's.
"""

from __future__ import annotations

import os

#: name -> default, as engine/settings.py KNOBS has them: the conductor's and the proxy's reads
DEFAULTS: dict[str, str] = {
    "CLIENT_CWD": "note", "CONDUCTOR_FIT": "model", "MODEL_NAME": "", "NARRATOR_DEVICE": "auto",
    "NARRATOR_DIR": "", "NARRATOR_MODEL": "", "NOTE_ANCHOR": "on", "NOTE_DEDUP": "parts",
    "SESSION_GENERATIONS": "on", "SHADOW_DIR": "", "SHADOW_DISK_GB": "20", "SHADOW_DISK": "",
    "SHADOW_GB": "4", "SHADOW_GRID": "512", "SHADOW_MODEL": "", "SHADOW_POOL_SLACK_GIB": "2",
    "STATUS_CLIENTS": "open-webui", "TANDEM_SCRATCH": "", "TANDEM_URL": "", "TOOL_FIT": "on",
}

_MISSING = object()


class StandaloneSettings:
    """`EngineSettings.get` over the names above, for a process without the engine."""

    prefix = "QWEN38_"

    def get(self, name: str, default=_MISSING):
        if default is _MISSING:
            if name not in DEFAULTS:
                raise KeyError(f"{self.prefix}{name} is not a knob the conductor or the proxy reads (server/knobs.py)")
            default = DEFAULTS[name]
        return os.environ.get(self.prefix + name, default)

    def is_set(self, name: str) -> bool:
        return (self.prefix + name) in os.environ


try:
    from engine.settings import SETTINGS
except ModuleNotFoundError as exc:          # the standalone proxy: no engine package
    if exc.name not in ("engine", "engine.settings"):
        raise
    SETTINGS = StandaloneSettings()
