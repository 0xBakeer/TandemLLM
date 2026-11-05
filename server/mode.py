"""OPS-260: the mode a server runs in and what runs beside the engine, for /health and /v1/models.

    engine  the engine alone: no conductor config directory (ops/serve.env's QSE_MODE=engine, a new
            install's default; every benchmark server)
    chat    the conductor is on (QSE_MODE=chat, the chat (beta))

`loaded` names what is loaded now: the conductor and its session monitor's mode, the decision models,
the narrator, every small-model role whose state is `loaded`, and whether chatd (the Chat tab's
service, QSE_CHAT_URL) takes a connection: a connect at most once a CHATD_TTL_S, never a request.
Nothing here imports the engine.
"""

from __future__ import annotations

import socket
import threading
import time

from server import chat_forward

CHATD_TTL_S = 5.0
CHATD_CONNECT_S = 0.25

_lock = threading.Lock()
_chatd = {"at": None, "up": False}


def chatd_up(clock=time.monotonic, connect=None) -> bool:
    """Whether chatd's port takes a connection, cached CHATD_TTL_S."""
    now = clock()
    with _lock:
        if _chatd["at"] is not None and now - _chatd["at"] < CHATD_TTL_S:
            return _chatd["up"]
    try:
        if connect is not None:
            up = bool(connect(chat_forward.target()))
        else:
            with socket.create_connection(chat_forward.target(), timeout=CHATD_CONNECT_S):
                up = True
    except (OSError, ValueError):          # ValueError: a QSE_CHAT_URL whose port is not a port
        up = False
    with _lock:
        _chatd.update(at=now, up=up)
    return up


def report(state: dict, roles: dict | None, chatd=chatd_up) -> dict:
    """`{"mode": engine|chat, "loaded": {...}}` from the server's STATE and its roles report."""
    cond, mon, reg, nar = (dict.get(state, k) for k in ("conductor", "monitor", "deciders", "narrator"))
    slot = getattr(nar, "slot", None)
    return {"mode": "chat" if cond is not None else "engine",
            "loaded": {"conductor": cond is not None,
                       "monitor": getattr(mon, "mode", None) or "raw",
                       "decision_models": list(reg.loaded()) if reg is not None else [],
                       "narrator": (slot.model.name if slot is not None and slot.state == "loaded"
                                    else None),
                       "roles": sorted(k for k, v in ((roles or {}).get("roles") or {}).items()
                                       if isinstance(v, dict) and v.get("state") == "loaded"),
                       "chatd": "up" if chatd() else "down"}}
