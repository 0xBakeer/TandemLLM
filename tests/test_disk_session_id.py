"""The disk tier's name for a request (`app._disk_session_id`): a request's own id, else for a Hermes
main request under an active plan the router's head key; never the shared side session."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)

from server import app  # noqa: E402
from server.conductor.router import Plan, Session  # noqa: E402


def _plan(active=True, side=None, key="h:abc"):
    p = Plan(None)
    p.active = active
    p.side = side
    p.session = Session(key) if key is not None else None
    return p


def test_hermes_with_an_active_plan_takes_the_head_key():
    assert app._disk_session_id("hermes", _plan(), None) == "h:abc"


def test_a_hermes_side_request_keeps_no_name():
    side = _plan(side={"kind": "title"}, key="side")
    assert app._disk_session_id("hermes", side, None) is None


def test_a_raw_request_keeps_no_name():
    assert app._disk_session_id("hermes", None, None) is None
    assert app._disk_session_id("hermes", _plan(active=False), None) is None


def test_a_request_with_its_own_session_id_keeps_it():
    assert app._disk_session_id("hermes", _plan(), "conv-7") == "conv-7"
    assert app._disk_session_id("openwebui", _plan(), "conv-7") == "conv-7"


def test_another_client_kind_gets_no_head_key():
    assert app._disk_session_id("openwebui", _plan(), None) is None
    assert app._disk_session_id(None, _plan(), None) is None


if __name__ == "__main__":
    n = 0
    for name, f in sorted(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print(f"  {name:<68} ok")
            n += 1
    print(f"{n} passed")
