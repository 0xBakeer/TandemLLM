"""GET /v1/models/{id} is keyed like the list route above it (deep audit 0.11.3).

With users configured, /v1/models answers 401 to an unkeyed client but
/v1/models/<id> answered 200 to everyone -- the card is the same private
routing slice the list guards: backend ids, labels, upstream ids, prices.

The handler is driven directly: the route's own branches, no server up.

Run: python -m pytest -q tests/test_proxy_models_card_auth.py
"""

from __future__ import annotations

import email.message
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proxy import app  # noqa: E402


class _Table:
    def card(self, name, started):
        return {"id": name, "object": "model"}


class _Svc:
    started = "1700000000"

    def __init__(self, users):
        self.users = users
        self.asked = 0
        self.table = _Table()

    def authorize(self, handler):
        self.asked += 1
        if not self.users:
            return True, None
        if handler.headers.get("Authorization") == "Bearer right":
            return True, "u"
        return False, None


class _Json:
    def __init__(self):
        self.calls = []

    def __call__(self, code, payload, extra_headers=()):
        self.calls.append((code, payload))


def _hit(path: str, svc: _Svc, auth_header: str | None = None):
    got = _Json()
    h = app.Handler.__new__(app.Handler)
    h.path = path
    h.command = "GET"
    h.client_address = ("198.51.100.7", 5555)
    h.headers = email.message.Message()
    if auth_header:
        h.headers["Authorization"] = auth_header
    h.close_connection = True
    h._json = got
    h._strip_session = lambda: False
    saved = dict(app.STATE)
    saved_forward = app.chat_forward.route
    app.STATE.update({"svc": svc, "auth": None})
    app.chat_forward.route = lambda handler, verb, auth: False
    try:
        h.do_GET()
    finally:
        app.STATE.clear()
        app.STATE.update(saved)
        app.chat_forward.route = saved_forward
    return got.calls, svc.asked


def test_the_card_route_asks_like_the_list_route():
    svc = _Svc(users={"u": "..."})
    calls, asked = _hit("/v1/models/private-model", svc)
    assert asked == 1, "the card route never consulted authorize()"
    assert not any(c == 200 for c, _ in calls), \
        f"an unkeyed client read a card: {calls}"


def test_a_keyed_client_still_gets_the_card():
    svc = _Svc(users={"u": "..."})
    calls, _ = _hit("/v1/models/private-model", svc, auth_header="Bearer right")
    assert calls == [(200, {"id": "private-model", "object": "model"})], calls


def test_no_users_configured_changes_nothing():
    svc = _Svc(users={})
    calls, _ = _hit("/v1/models/any-model", svc)
    assert calls == [(200, {"id": "any-model", "object": "model"})], calls
