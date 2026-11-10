"""Access control, handler-level with fake peers and headers, on a CPU.

the Gherkin is the matrix: through the proxy (X-Forwarded-For set) the admin routes need a
token, the OpenAI API does not, /health says only its status; the admin token and the session
cookie open the dashboard, the metrics token opens /metrics and nothing else; the cookie cannot
clear the cache; the box's own tools need nothing; a sixth failed login in a minute is a 429; and
no token ever appears in the log, the log stream, /v1/dashboard/system or /metrics. With the
dashboard login off (the server's default) the dashboard's read API answers without a token and the
page is told there is nothing to sign in to; the cache routes, /metrics and the full /health do not
change.

AU1 (2026-10-06, the owner: "remove the password prompt from the ui from everywhere; we will define
later when we set the password"): the admin token is the admin password and the one switch. With
none set every admin route answers anyone, login on or off (until then: 404, fail closed), and
/metrics keeps the metrics token when one is set; with one set, everything above holds unchanged.

Run: python tests/test_auth.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_app_loop import Req, serve  # noqa: E402  (first: the CPU environment)
from test_usage import UTok, _ids, _script  # noqa: E402

from http.server import ThreadingHTTPServer  # noqa: E402

from server import app, auth, dashboard_api, logbuf, metrics  # noqa: E402
from tools import contract_check  # noqa: E402

ADMIN = "adm-" + "a1b2c3d4" * 6
METRICS = "met-" + "9f8e7d6c" * 6
PROXY = {"X-Forwarded-For": "203.0.113.44", "X-Real-IP": "203.0.113.44"}


def _setup(admin=ADMIN, metrics_token=METRICS, **kw):
    serve()
    app.STATE.update(tok=UTok(), auth=auth.Auth(admin, metrics_token, **kw),
                     gpu_sampler=dashboard_api.GpuSampler(cmd="/nonexistent"), args={"x": 1},
                     started=int(time.time()))


def call(method, path, headers=None, body=None, peer="127.0.0.1"):
    req = Req(path, body if body is not None else {})
    req.command, req.client_address = method, (peer, 0)
    req.headers.update(headers or {})
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        getattr(req, f"do_{method}")()
    head, _, raw = req.wfile.getvalue().partition(b"\r\n\r\n")
    head = head.decode()
    code = int(head.split(" ", 2)[1])
    try:
        payload = json.loads(raw) if raw and not raw.startswith(b"#") else raw.decode()
    except ValueError:
        payload = raw.decode()
    return code, head, payload


def bearer(tok):
    return {"Authorization": f"Bearer {tok}"}


def test_admin_routes_need_a_token_through_the_proxy():
    _setup()
    for method, path in (("GET", "/v1/dashboard/summary"), ("GET", "/v1/dashboard/logs"),
                         ("GET", "/v1/dashboard/logs/export"),
                         ("GET", "/metrics"), ("GET", "/v1/cache/stats"),
                         ("POST", "/v1/cache/clear")):
        code, head, body = call(method, path, PROXY)
        assert code == 401, (method, path, code)
        assert "WWW-Authenticate: Bearer" in head, head
        assert not contract_check.check(body, "error"), body


def test_the_public_api_is_unchanged():
    _setup()
    real = app.generate_stream
    app.generate_stream = _script(_ids("hi"))
    try:
        code, _, body = call("POST", "/v1/chat/completions", PROXY,
                             {"messages": [{"role": "user", "content": "q"}], "stream": True})
        assert code == 200 and "data: [DONE]" in body, (code, body[-200:])
        code, _, body = call("POST", "/v1/chat/completions", PROXY,
                             {"messages": [{"role": "user", "content": "q"}]})
        assert code == 200 and body["choices"], body
    finally:
        app.generate_stream = real
    code, _, body = call("GET", "/v1/models", PROXY)
    assert code == 200 and body["data"][0]["id"] == "t", body


def test_public_health_is_minimal():
    _setup()
    code, _, body = call("GET", "/health", PROXY)
    assert code == 200 and body == {"status": "ok"}, body
    app.STATE["draining"] = True
    code, _, body = call("GET", "/health", PROXY)
    assert code == 503 and body == {"status": "draining"}, body
    app.STATE["draining"] = False
    code, _, body = call("GET", "/health", dict(PROXY, **bearer(METRICS)))
    assert code == 200 and "inflight" in body and "cache" in body


def test_the_bearer_token():
    _setup()
    code, _, body = call("GET", "/v1/dashboard/summary", dict(PROXY, **bearer(ADMIN)))
    assert code == 200 and body["contract_version"] == dashboard_api.CONTRACT, body
    code, _, _ = call("GET", "/v1/dashboard/summary", dict(PROXY, **bearer(ADMIN[:-1] + "x")))
    assert code == 401
    code, _, _ = call("GET", "/v1/cache/stats", dict(PROXY, **bearer(ADMIN)))
    assert code == 200
    code, _, _ = call("POST", "/v1/cache/clear", dict(PROXY, **bearer(ADMIN)))
    assert code == 200


def test_the_metrics_token_is_scoped():
    _setup()
    code, head, text = call("GET", "/metrics", dict(PROXY, **bearer(METRICS)))
    assert code == 200 and "text/plain" in head
    for method, path in (("GET", "/v1/dashboard/summary"), ("GET", "/v1/cache/stats"),
                         ("POST", "/v1/cache/clear"), ("GET", "/v1/dashboard/session")):
        code, _, _ = call(method, path, dict(PROXY, **bearer(METRICS)))
        assert code == 401, (path, code)
    code, _, _ = call("GET", "/metrics", dict(PROXY, **bearer(ADMIN)))
    assert code == 200, "the admin token opens /metrics too"


def _login(headers=None, token=ADMIN):
    code, head, _ = call("POST", "/v1/dashboard/session", dict(PROXY, **(headers or {})),
                         {"token": token})
    cookie = None
    for line in head.split("\r\n"):
        if line.startswith("Set-Cookie: qse_dash="):
            cookie = line[len("Set-Cookie: "):].split(";")[0]
    return code, head, cookie


def test_the_session_cookie():
    _setup()
    code, head, cookie = _login()
    assert code == 204 and cookie, head
    for attr in ("HttpOnly", "SameSite=Strict", "Path=/", f"Max-Age={auth.SESSION_S}"):
        assert attr in head, (attr, head)
    assert "Secure" not in head
    code, head, _ = _login({"X-Forwarded-Proto": "https"})
    assert "; Secure" in head
    code, _, body = call("GET", "/v1/dashboard/session", dict(PROXY, Cookie=cookie))
    assert code == 200 and body["authenticated"] is True and body["expires_at"].endswith("Z")
    assert body["login"] is True
    assert not contract_check.check(body, "session")
    # the browser's EventSource: the log stream with the cookie only, over a real socket
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        s = socket.create_connection(("127.0.0.1", httpd.server_address[1]), timeout=5)
        s.sendall(f"GET /v1/dashboard/logs?follow=1 HTTP/1.1\r\nHost: x\r\nX-Forwarded-For: "
                  f"203.0.113.44\r\nCookie: {cookie}\r\n\r\n".encode())
        got = s.recv(4096)
        assert got.startswith(b"HTTP/1.1 200") and b"text/event-stream" in got, got[:200]
        s.close()
    finally:
        httpd.shutdown()
    code, head, _ = call("DELETE", "/v1/dashboard/session", dict(PROXY, Cookie=cookie))
    assert code == 204 and "qse_dash=;" in head and "Max-Age=0" in head
    code, _, _ = call("GET", "/v1/dashboard/logs?follow=0", dict(PROXY, Cookie=cookie))
    assert code == 401, "a logged-out session is dead on the server too"
    code, _, _ = call("GET", "/v1/dashboard/session", dict(PROXY, Cookie=cookie))
    assert code == 401


def test_two_logins_in_one_second_are_two_sessions():
    """A session cookie was `<expiry second>.<mac of it>`, so every login in the same second
    got the SAME cookie: a sign-out on one device killed the other device's session, and a login
    right after a sign-out got the value just revoked -- the e2e's next test landed on the token
    screen (the fake-engine tier, 2026-09-25). Each login is its own session now."""
    _setup()
    now = [1_790_000_000.0]
    app.STATE["auth"] = a = auth.Auth(ADMIN, METRICS, clock=lambda: now[0])
    _, _, one = _login()
    _, _, two = _login()                                   # the same second
    assert one and two and one != two, (one, two)
    code, _, _ = call("DELETE", "/v1/dashboard/session", dict(PROXY, Cookie=one))
    assert code == 204
    code, _, _ = call("GET", "/v1/dashboard/summary", dict(PROXY, Cookie=one))
    assert code == 401, "the signed-out session is dead"
    code, _, _ = call("GET", "/v1/dashboard/summary", dict(PROXY, Cookie=two))
    assert code == 200, "the other device's session is not"
    _, _, three = _login()                                 # a login right after the sign-out
    code, _, _ = call("GET", "/v1/dashboard/summary", dict(PROXY, Cookie=three))
    assert code == 200, "a new login is never born revoked"


def test_a_forged_or_expired_cookie_is_refused():
    _setup()
    now = [1_790_000_000.0]
    app.STATE["auth"] = a = auth.Auth(ADMIN, METRICS, clock=lambda: now[0])
    value, exp = a.make_cookie()
    exp_s, nonce, mac = value.split(".")
    for bad in (f"{int(exp_s) + 999}.{nonce}.{mac}", f"{exp_s}.{nonce}.{'0' * 64}",
                f"{exp_s}.{'f' * 16}.{mac}", f"{exp_s}.{mac}", "garbage", f"{exp_s}"):
        code, _, _ = call("GET", "/v1/dashboard/summary", dict(PROXY, Cookie=f"qse_dash={bad}"))
        assert code == 401, bad
    code, _, _ = call("GET", "/v1/dashboard/summary", dict(PROXY, Cookie=f"qse_dash={value}"))
    assert code == 200
    now[0] += auth.SESSION_S + 1
    code, _, _ = call("GET", "/v1/dashboard/summary", dict(PROXY, Cookie=f"qse_dash={value}"))
    assert code == 401, "expired"
    # rotating the admin token ends every session
    now[0] -= auth.SESSION_S
    app.STATE["auth"] = auth.Auth("adm-" + "z" * 40, METRICS, clock=lambda: now[0])
    code, _, _ = call("GET", "/v1/dashboard/summary", dict(PROXY, Cookie=f"qse_dash={value}"))
    assert code == 401


def test_the_cookie_cannot_clear_the_cache():
    _setup()
    _, _, cookie = _login()
    code, _, _ = call("POST", "/v1/cache/clear", dict(PROXY, Cookie=cookie))
    assert code == 401
    code, _, _ = call("GET", "/v1/cache/stats", dict(PROXY, Cookie=cookie))
    assert code == 200, "reading is fine"


def test_trusted_local_tools():
    _setup()
    code, _, body = call("GET", "/health")                         # the watchdog
    assert code == 200 and "inflight" in body and "memory" in body
    code, _, text = call("GET", "/metrics")                        # row3
    assert code == 200 and "qse_" in text
    code, _, _ = call("POST", "/v1/cache/clear")                   # gate.sh's cold rows
    assert code == 200
    code, _, _ = call("GET", "/v1/cache/stats")
    assert code == 200
    # not the dashboard, and not a LAN client talking to :8000 directly
    code, _, _ = call("GET", "/v1/dashboard/summary")
    assert code == 401
    code, _, body = call("GET", "/health", peer="203.0.113.44")
    assert body == {"status": "ok"}
    code, _, _ = call("GET", "/metrics", peer="203.0.113.44")
    assert code == 401
    # and --no-trust-loopback removes the exemption
    _setup(trust_loopback=False)
    code, _, _ = call("GET", "/metrics")
    assert code == 401


def test_the_public_liveness_page():
    """/metrics/up answers anyone, with one number, so a scrape can tell a down engine
    from a wrong metrics token."""
    _setup()
    code, head, text = call("GET", "/metrics/up", PROXY)
    assert code == 200 and "text/plain" in head and text.endswith("qse_up 1\n"), text
    assert "qse_requests" not in text
    app.STATE["draining"] = True
    code, _, text = call("GET", "/metrics/up", PROXY)
    assert code == 200 and text.endswith("qse_up 0\n")
    app.STATE["draining"] = False
    code, _, _ = call("GET", "/metrics", PROXY)
    assert code == 401


ADMIN_ROUTES = (("GET", "/v1/dashboard/summary"), ("GET", "/v1/dashboard/metrics"),
                ("GET", "/v1/dashboard/logs/export"),
                ("GET", "/v1/cache/stats"), ("POST", "/v1/cache/clear"), ("GET", "/metrics"))


def test_no_password_opens_every_admin_route():
    """AU1: no admin token (the password) configured -- through the proxy, from the LAN, with no
    header or with any bearer, login on or off -- every admin route answers, the full /health
    too; the session check says there is no password and nothing to sign in to."""
    for login in (True, False):
        _setup(admin=None, metrics_token=None, login=login)
        assert auth.Auth(None, None, login=login).password_set is False
        for headers, peer in (({}, "127.0.0.1"), (PROXY, "127.0.0.1"), ({}, "203.0.113.44"),
                              (bearer("anything-" + "x" * 40), "203.0.113.44")):
            for method, path in ADMIN_ROUTES:
                code, _, body = call(method, path, headers, peer=peer)
                assert code == 200, (login, method, path, headers, peer, code, body)
            code, _, body = call("GET", "/health", headers, peer=peer)
            assert code == 200 and set(body) > {"status"}, "the full body, no token"
        code, head, body = call("GET", "/v1/dashboard/session", PROXY)
        assert code == 200 and body == {"contract_version": "1.1", "authenticated": True,
                                        "login": False, "password_set": False,
                                        "expires_at": None}, body
        assert not contract_check.check(body, "session")
        assert "Set-Cookie" not in head
        for method, b in (("POST", {"token": ADMIN}), ("DELETE", None)):
            code, head, body = call(method, "/v1/dashboard/session", PROXY, b)
            assert code == 404 and "no admin password" in body["error"]["message"], (method, body)
            assert "Set-Cookie" not in head
    d = auth.Auth().describe()
    assert "admin password: not set" in d and "open" in d and "trusted network" in d, d


def test_no_password_keeps_the_metrics_token_for_scrapers():
    """AU1: the metrics token is not the password. With only it set, /metrics still wants it
    (or the box itself); the dashboard's own copy and every admin route stay open."""
    _setup(admin=None, metrics_token=METRICS)
    code, _, _ = call("GET", "/metrics", PROXY)
    assert code == 401
    code, _, _ = call("GET", "/metrics", bearer("wrong-" + "w" * 40), peer="203.0.113.44")
    assert code == 401
    code, _, text = call("GET", "/metrics", dict(PROXY, **bearer(METRICS)))
    assert code == 200 and "qse_" in text
    code, _, _ = call("GET", "/metrics")
    assert code == 200, "the box's own tools"
    code, _, text = call("GET", "/v1/dashboard/metrics", PROXY)
    assert code == 200 and "qse_" in text, "the page needs no metrics token"
    for method, path in ADMIN_ROUTES[:4]:
        code, _, _ = call(method, path, PROXY)
        assert code == 200, path
    assert "metrics token: set" in auth.Auth(None, METRICS).describe()


def test_a_password_brings_the_rule_back():
    """AU1: setting the admin token is the one switch back: the same routes refuse without it
    through the proxy, and take it as a bearer (the dashboard session too)."""
    _setup(admin=ADMIN, metrics_token=None, login=True)
    assert app.STATE["auth"].password_set is True
    for method, path in ADMIN_ROUTES:
        code, _, _ = call(method, path, PROXY)
        assert code == 401, (path, code)
        code, _, _ = call(method, path, dict(PROXY, **bearer(ADMIN)))
        assert code == 200, (path, code)
    code, _, body = call("GET", "/health", PROXY)
    assert body == {"status": "ok"}
    code, _, body = call("GET", "/v1/dashboard/session", dict(PROXY, **bearer(ADMIN)))
    assert code == 200 and body["login"] is True and body["password_set"] is True, body
    assert not contract_check.check(body, "session")
    assert "admin password: set" in auth.Auth(ADMIN).describe()


def test_brute_force():
    _setup()
    other = {"X-Forwarded-For": "203.0.113.45", "X-Real-IP": "203.0.113.45"}
    for i in range(5):
        code, _, _ = _login(token=f"wrong-{i}")
        assert code == 401
    code, head, _ = _login()
    assert code == 429 and "Retry-After" in head, "the sixth, even with the right token"
    code, _, _ = call("POST", "/v1/dashboard/session", other, {"token": ADMIN})
    assert code == 204, "another address is not blocked"


def test_short_tokens_are_refused():
    for kw in ({"admin_token": "tiny-tok"}, {"metrics_token": "q" * 31}):
        try:
            auth.Auth(**kw)
            raise AssertionError(f"{kw} accepted")
        except SystemExit as exc:
            assert "32" in str(exc) and list(kw.values())[0] not in str(exc), "never echoed"


def test_tokens_never_leak():
    _setup()
    buf = logbuf.LogBuffer()
    out, err = io.StringIO(), io.StringIO()
    old = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = logbuf.Tee(out, buf, "stdout"), logbuf.Tee(err, buf, "stderr")
    os.environ.update(QSE_ADMIN_TOKEN=ADMIN, QSE_METRICS_TOKEN=METRICS)
    try:
        app.STATE.update(verbose=True, log_buffer=buf)
        print(f"[server] {app.STATE['auth'].describe()}", flush=True)
        _, _, cookie = _login()
        seen = []
        for method, path, h in (("GET", "/v1/dashboard/system", bearer(ADMIN)),
                                ("GET", "/metrics", bearer(METRICS)),
                                ("GET", "/v1/dashboard/logs?follow=0&level=debug", bearer(ADMIN)),
                                ("GET", "/health", bearer(ADMIN)),
                                ("GET", "/v1/dashboard/session", {"Cookie": cookie}),
                                ("GET", "/v1/dashboard/logs/export", bearer(ADMIN))):
            req = Req(path, {})
            req.command = method
            req.headers.update(dict(PROXY, **h))
            getattr(req, f"do_{method}")()
            seen.append(req.wfile.getvalue().decode())
            assert req.wfile.getvalue().startswith(b"HTTP/1.1 200"), (path, seen[-1][:120])
    finally:
        sys.stdout, sys.stderr = old
        os.environ.pop("QSE_ADMIN_TOKEN", None)
        os.environ.pop("QSE_METRICS_TOKEN", None)
        app.STATE["verbose"] = False
    everything = "".join(seen) + out.getvalue() + err.getvalue() + json.dumps(list(buf.ring))
    assert "<redacted>" in seen[0]
    for secret in (ADMIN, METRICS):
        assert secret not in everything, "a token leaked"
    assert "dashboard login: on" in out.getvalue()


# -------------------------------------------------------------- the dashboard login off
DASH_READS = ("/v1/dashboard/summary", "/v1/dashboard/usage?bucket=day",
              "/v1/dashboard/requests?limit=5", "/v1/dashboard/system",
              "/v1/dashboard/logs?follow=0", "/v1/dashboard/live?follow=0",
              "/v1/dashboard/metrics", "/v1/dashboard/logs/export")


def test_login_off_opens_the_read_api_without_a_token():
    """Off is the server's default: anyone who reaches the port reads the dashboard, through the
    proxy and from the LAN alike, with no admin token configured at all (no secrets.env)."""
    for admin, metrics_token in ((ADMIN, METRICS), (None, None)):
        _setup(admin=admin, metrics_token=metrics_token, login=False)
        for headers, peer in ((PROXY, "127.0.0.1"), ({}, "203.0.113.44")):
            for path in DASH_READS:
                code, _, body = call("GET", path, headers, peer=peer)
                assert code == 200, (admin is not None, path, code, body)
            code, _, text = call("GET", "/v1/dashboard/metrics", headers, peer=peer)
            assert "qse_" in text, "the same page as /metrics"
        code, head, body = call("GET", "/v1/dashboard/session", PROXY)
        assert code == 200 and body == {"contract_version": "1.1", "authenticated": True,
                                        "login": False, "password_set": admin is not None,
                                        "expires_at": None}, body
        assert "Set-Cookie" not in head
        assert not contract_check.check(body, "session")
        # nothing to sign in to or out of
        for method, b in (("POST", {"token": ADMIN}), ("DELETE", None)):
            code, head, body = call(method, "/v1/dashboard/session", PROXY, b)
            assert code == 404 and body["error"]["type"] == "not_found", (method, code)
            assert "Set-Cookie" not in head


def test_login_off_leaves_the_other_routes_alone():
    """The login decides /v1/dashboard/* only: the cache routes, /metrics and the full /health
    need what they needed before, and the cache clear still takes only the bearer token."""
    _setup(login=False)
    code, _, _ = call("POST", "/v1/cache/clear", PROXY)
    assert code == 401
    code, _, _ = call("POST", "/v1/cache/clear", PROXY, peer="203.0.113.44")
    assert code == 401
    code, _, _ = call("POST", "/v1/cache/clear", dict(PROXY, **bearer(ADMIN)))
    assert code == 200
    for path in ("/metrics", "/v1/cache/stats"):
        code, _, _ = call("GET", path, PROXY)
        assert code == 401, path
    code, _, body = call("GET", "/health", PROXY)
    assert body == {"status": "ok"}
    code, _, _ = call("GET", "/metrics", dict(PROXY, **bearer(METRICS)))
    assert code == 200
    code, _, _ = call("POST", "/v1/cache/clear")                   # the box's own gate.sh
    assert code == 200
    # AU1: with no tokens at all the admin routes answer through the proxy (until 2026-10-06: 404)
    _setup(admin=None, metrics_token=None, login=False)
    for method, path in (("POST", "/v1/cache/clear"), ("GET", "/metrics"),
                         ("GET", "/v1/cache/stats")):
        code, _, _ = call(method, path, PROXY)
        assert code == 200, (path, code)


def test_login_on_guards_the_dashboards_metrics_copy():
    _setup()
    code, _, _ = call("GET", "/v1/dashboard/metrics", PROXY)
    assert code == 401
    code, _, _ = call("GET", "/v1/dashboard/metrics", dict(PROXY, **bearer(METRICS)))
    assert code == 401, "the metrics token opens /metrics, not the dashboard"
    _, _, cookie = _login()
    code, _, text = call("GET", "/v1/dashboard/metrics", dict(PROXY, Cookie=cookie))
    assert code == 200 and "qse_" in text


# ------------------------------------------------------------- the log export (SRV-804)
EXPORT = "/v1/dashboard/logs/export"
FILE = 'Content-Disposition: attachment; filename="tandemllm-'


def test_export_follows_the_rule_of_the_log_route():
    """`/v1/dashboard/logs/export` answers whoever `/v1/dashboard/logs` answers and nobody else:
    a 401 and no file without a token or a session, with the metrics token and with a wrong
    token; the file with the admin token and with the session cookie; a 401 again once the
    session is logged out."""
    _setup()
    for headers in (PROXY, dict(PROXY, **bearer(METRICS)),
                    dict(PROXY, **bearer(ADMIN[:-1] + "x")), dict(PROXY, Cookie="qse_dash=nope")):
        codes = []
        for path in ("/v1/dashboard/logs?follow=0", EXPORT):
            code, head, body = call("GET", path, headers)
            codes.append(code)
            assert "WWW-Authenticate: Bearer" in head, (path, head)
            assert not contract_check.check(body, "error"), body
            assert "Content-Disposition" not in head, "a refusal is not a file"
        assert codes == [401, 401], (headers, codes)
    code, head, text = call("GET", EXPORT, dict(PROXY, **bearer(ADMIN)))
    assert code == 200 and "Content-Type: text/plain; charset=utf-8" in head, (code, head)
    assert FILE in head and text.startswith(logbuf.EXPORT_TITLE + "\n"), (head, text[:80])
    _, _, cookie = _login()
    code, head, text = call("GET", EXPORT, dict(PROXY, Cookie=cookie))
    assert code == 200 and FILE in head and text.startswith(logbuf.EXPORT_TITLE + "\n")
    code, _, _ = call("DELETE", "/v1/dashboard/session", dict(PROXY, Cookie=cookie))
    assert code == 204
    code, head, _ = call("GET", EXPORT, dict(PROXY, Cookie=cookie))
    assert code == 401 and "Content-Disposition" not in head, "a logged-out session gets no file"


def test_export_is_no_route_when_the_dashboard_is_off():
    """The policy's `absent` verdict (the old fail-closed rule; `decide` no longer returns it, the
    handler still honours it) is a 404 for the export exactly as for the log route, admin token
    or not."""
    _setup()

    class Off(auth.Auth):
        def decide(self, route, peer, headers):
            return "absent" if route == "dashboard" else super().decide(route, peer, headers)

    app.STATE["auth"] = Off(ADMIN, METRICS)
    for path in ("/v1/dashboard/logs?follow=0", EXPORT):
        code, head, body = call("GET", path, dict(PROXY, **bearer(ADMIN)))
        assert code == 404 and body["error"]["type"] == "not_found", (path, code, body)
        assert "Content-Disposition" not in head


def _get(port, path, token=ADMIN):
    """One GET over a real socket, read to the end of its body: (status line, head, body)."""
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    s.sendall(f"GET {path} HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {token}\r\n"
              f"Connection: close\r\n\r\n".encode())
    raw = b""
    while True:
        head, sep, body = raw.partition(b"\r\n\r\n")
        if sep and len(body) >= int(head.split(b"Content-Length: ")[1].split(b"\r\n")[0]):
            break
        chunk = s.recv(65536)
        if not chunk:
            break
        raw += chunk
    s.close()
    head, _, body = raw.partition(b"\r\n\r\n")
    return head.split(b"\r\n")[0].decode(), head.decode(), body


def test_export_does_not_use_a_stream_place():
    """Four log streams are open, the cap: a fifth stream is refused with a 429, the export is
    not. It is answered in full, it leaves the four places as they were, and the ring is neither
    cleared nor moved by it."""
    _setup()
    buf = logbuf.LogBuffer()
    for i in range(30):
        buf.append(f"[server] line {i}")
    app.STATE["log_buffer"] = buf
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port, streams = httpd.server_address[1], []
    try:
        for _ in range(logbuf.MAX_SUBSCRIBERS):
            s = socket.create_connection(("127.0.0.1", port), timeout=10)
            s.sendall(f"GET /v1/dashboard/logs?follow=1 HTTP/1.1\r\nHost: x\r\n"
                      f"Authorization: Bearer {ADMIN}\r\n\r\n".encode())
            streams.append(s)
        end = time.time() + 5
        while buf.subscribers < logbuf.MAX_SUBSCRIBERS and time.time() < end:
            time.sleep(0.05)
        assert buf.subscribers == logbuf.MAX_SUBSCRIBERS, buf.subscribers
        status, _, _ = _get(port, "/v1/dashboard/logs?follow=1")
        assert " 429 " in status, "the cap holds for a fifth stream"
        before = buf.snapshot()
        for _ in range(3):
            status, head, body = _get(port, EXPORT)
            assert " 200 " in status and FILE in head, (status, head)
            text = body.decode()
            assert "count: 30 lines" in text and text.count("[server] line ") == 30, text[:300]
        assert buf.subscribers == logbuf.MAX_SUBSCRIBERS, "the export took or freed no place"
        assert buf.snapshot() == before, "the export cleared or moved the ring"
    finally:
        for s in streams:
            s.close()
        end = time.time() + 5
        while buf.subscribers and time.time() < end:
            buf.append("[server] tick")
            time.sleep(0.1)
        httpd.shutdown()
    assert buf.subscribers == 0, "the closed streams gave their places back"


def test_export_does_not_take_the_engine_lock():
    """A generation holds the engine lock from its prefill to its last token. The export is
    answered while the lock is held, and it never asks for the lock: a stand-in lock that fails
    the test on any acquire sees none."""
    _setup()
    done = {}

    def ask():
        done["answer"] = call("GET", EXPORT, dict(PROXY, **bearer(ADMIN)))

    with app.LOCK:
        t = threading.Thread(target=ask, daemon=True)
        t.start()
        t.join(10)
        assert not t.is_alive(), "the export waited for the engine lock"
    code, head, text = done["answer"]
    assert code == 200 and FILE in head and text.startswith(logbuf.EXPORT_TITLE + "\n")

    class Spy:
        asked = 0

        def acquire(self, *a, **kw):
            Spy.asked += 1
            return True

        def release(self):
            pass

        def locked(self):
            return False

        def __enter__(self):
            return self.acquire() and self

        def __exit__(self, *exc):
            return False

    real, app.LOCK = app.LOCK, Spy()
    try:
        code, _, _ = call("GET", EXPORT, dict(PROXY, **bearer(ADMIN)))
    finally:
        app.LOCK = real
    assert code == 200 and Spy.asked == 0, (code, Spy.asked)


def test_the_login_setting():
    """`--dashboard-login`, default from QSE_DASHBOARD_LOGIN, default off; the policy object read
    from the environment agrees, and the startup line says which."""
    old = os.environ.pop(auth.LOGIN_ENV, None)
    try:
        assert app.parser().parse_args([]).dashboard_login == "off"
        assert auth.Auth.from_env({}).login is False
        assert "admin password: not set" in auth.Auth.from_env({}).describe()
        assert "dashboard login: off" in auth.Auth.from_env({"QSE_ADMIN_TOKEN": ADMIN}).describe()
        os.environ[auth.LOGIN_ENV] = "on"
        assert app.parser().parse_args([]).dashboard_login == "on"
        assert app.parser().parse_args(["--dashboard-login", "off"]).dashboard_login == "off"
        assert auth.Auth.from_env().login is True
        assert auth.Auth.from_env(login=False).login is False, "the flag wins"
    finally:
        os.environ.pop(auth.LOGIN_ENV, None)
        if old is not None:
            os.environ[auth.LOGIN_ENV] = old


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
