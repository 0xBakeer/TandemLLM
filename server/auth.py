"""Who may read what: the OpenAI API stays open, the rest needs a token.

your-host.example proxies every path to this server, and the reverse proxy reaches it through an ssh tunnel,
so a request from any device on the LAN arrives from 127.0.0.1 -- the source address cannot tell
the box's own tools from anyone else. So the mechanism is a bearer token, with one narrow
exemption, and a session cookie so the dashboard's EventSource needs no token in its URL:

    route                                   who
    POST /v1/chat/completions, /v1/completions, GET /v1/models      anyone (per-key API auth is)
    GET /health                             anyone gets {"status"}; the full body needs a token or trusted-local
    GET /metrics                            the metrics token, the admin token/session, or trusted-local
    GET /v1/cache/stats                     the admin token/session, or trusted-local
    POST /v1/cache/clear                    the admin BEARER token (never the cookie), or trusted-local
    GET /dashboard, /dashboard/*            anyone: the static shell, no data in it
    /v1/dashboard/*                         the admin token or session only (POST /v1/dashboard/session is the login)

TRUSTED-LOCAL (`--trust-loopback`, on by default): the TCP peer is loopback AND the request has
neither `X-Forwarded-For` nor `X-Real-IP`. The reverse proxy (nginx) sets both on everything it proxies, so
tunnelled traffic never qualifies, and the watchdog, row3, gate.sh and soak on the box keep working
with no token.

FAIL CLOSED: with no admin token configured, the admin routes answer 404 to everyone -- the
dashboard API does not exist. Tokens come from `~/.qwen38-spark-engine/secrets.env` (made by
`ops/make-secrets.sh`, mode 600, sourced by `ops/start.sh`), are at least 32 characters, are
compared with `hmac.compare_digest`, and are never printed, logged or returned.

THE SESSION: `POST /v1/dashboard/session {"token"}` sets `qse_dash=<expiry>.<nonce>.<HMAC-SHA256>` (a
key derived from the admin token, so rotating the token ends every session; the nonce makes each
login its own session, so signing out ends that one only), HttpOnly, SameSite=Strict,
Path=/, 12 hours, Secure behind https. Five failed logins a minute from one address (X-Real-IP, or
the peer) and the sixth gets a 429. The dashboard API only reads, and the one route that writes
(`/v1/cache/clear`) does not take the cookie, so SameSite=Strict is the whole CSRF story.
"""

from __future__ import annotations

import collections
import hashlib
import hmac
import os
import secrets
import threading
import time

COOKIE = "qse_dash"
SESSION_S = int(os.environ.get("QSE_SESSION_S", 400 * 86400))  # a LAN-only dashboard: one login lasts 400 days (the cookie cap browsers keep)
MIN_TOKEN = 32
LOGIN_FAILS = 5
LOGIN_WINDOW_S = 60.0
LOOPBACK = ("127.0.0.1", "::1", "::ffff:127.0.0.1")


def _eq(a: str | None, b: str | None) -> bool:
    return bool(a) and bool(b) and hmac.compare_digest(a.encode(), b.encode())


class Auth:
    def __init__(self, admin_token: str | None = None, metrics_token: str | None = None, *,
                 trust_loopback: bool = True, clock=time.time):
        for name, tok in (("QSE_ADMIN_TOKEN", admin_token), ("QSE_METRICS_TOKEN", metrics_token)):
            if tok is not None and len(tok) < MIN_TOKEN:
                raise SystemExit(f"[auth] {name} is shorter than {MIN_TOKEN} characters; make a "
                                 f"new one with ops/make-secrets.sh --rotate")
        self.admin = admin_token or None
        self.metrics = metrics_token or None
        self.trust_loopback = bool(trust_loopback)
        self.clock = clock
        self._key = (hmac.new(self.admin.encode(), b"qse-dash-session-v1", hashlib.sha256).digest()
                     if self.admin else None)
        self._fails: dict = collections.defaultdict(collections.deque)
        self._revoked: dict[str, int] = {}          # logged-out cookie values, until they expire
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, env=None, **kw) -> "Auth":
        env = os.environ if env is None else env
        return cls(env.get("QSE_ADMIN_TOKEN") or None, env.get("QSE_METRICS_TOKEN") or None, **kw)

    def describe(self) -> str:
        """The startup line: whether auth is on, never a token."""
        return (f"dashboard auth: {'on' if self.admin else 'off (no admin token)'}, "
                f"metrics token: {'set' if self.metrics else 'not set'}, "
                f"trusted loopback: {'on' if self.trust_loopback else 'off'}")

    # ----------------------------------------------------------------- who is asking
    @staticmethod
    def _header(headers, name: str) -> str:
        try:
            return headers.get(name) or ""
        except Exception:                                          # noqa: BLE001
            return ""

    def trusted_local(self, peer: str, headers) -> bool:
        if not self.trust_loopback or peer not in LOOPBACK:
            return False
        return not (self._header(headers, "X-Forwarded-For") or self._header(headers, "X-Real-IP"))

    def bearer(self, headers) -> str | None:
        auth = self._header(headers, "Authorization").strip()
        if auth[:7].lower() != "bearer ":
            return None
        return auth[7:].strip() or None

    def admin_bearer(self, headers) -> bool:
        return _eq(self.bearer(headers), self.admin)

    def metrics_bearer(self, headers) -> bool:
        return _eq(self.bearer(headers), self.metrics)

    # ----------------------------------------------------------------- the session cookie
    def make_cookie(self, now: float | None = None) -> tuple[str, int]:
        """`<expiry>.<nonce>.<mac>`. The nonce makes each login its own session: the value
        was the expiry second and its mac alone, so two logins in one second shared a cookie and a
        sign-out revoked both, or a login right after a sign-out was born revoked."""
        exp = int((self.clock() if now is None else now) + SESSION_S)
        nonce = secrets.token_hex(8)
        mac = hmac.new(self._key, f"{exp}.{nonce}".encode(), hashlib.sha256).hexdigest()
        return f"{exp}.{nonce}.{mac}", exp

    def cookie(self, headers) -> str | None:
        val = None
        for part in self._header(headers, "Cookie").split(";"):
            k, _, v = part.strip().partition("=")
            if k == COOKIE:
                val = v
        return val or None

    def revoke(self, headers) -> None:
        """Logout: the presented session stops working here too, not only in the browser."""
        exp = self.session(headers)
        if exp is not None:
            now = self.clock()
            with self._lock:
                self._revoked = {v: e for v, e in self._revoked.items() if e > now}
                self._revoked[self.cookie(headers)] = exp

    def session(self, headers, now: float | None = None) -> int | None:
        """The session's expiry if the request carries a valid, unexpired one, else None."""
        if self._key is None:
            return None
        val = self.cookie(headers)
        if not val or val in self._revoked:
            return None
        parts = val.split(".")
        if len(parts) != 3 or not parts[0].isdigit():
            return None
        exp_s, nonce, mac = parts
        want = hmac.new(self._key, f"{exp_s}.{nonce}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(want, mac):
            return None
        exp = int(exp_s)
        return exp if exp > (self.clock() if now is None else now) else None

    def admin_any(self, headers) -> bool:
        return self.admin_bearer(headers) or self.session(headers) is not None

    # ----------------------------------------------------------------- login rate limit
    def login_blocked(self, who: str) -> bool:
        now = self.clock()
        with self._lock:
            q = self._fails[who]
            while q and now - q[0] > LOGIN_WINDOW_S:
                q.popleft()
            return len(q) >= LOGIN_FAILS

    def login_failed(self, who: str) -> None:
        with self._lock:
            self._fails[who].append(self.clock())
            if len(self._fails) > 10_000:
                self._fails.clear()

    # ----------------------------------------------------------------- the policy
    def decide(self, route: str, peer: str, headers) -> str:
        """`ok`, `unauthorized` (401) or `absent` (404, fail closed) for one guarded route kind:
        `metrics`, `cache_read`, `cache_clear`, `dashboard`, `health_full`."""
        local = self.trusted_local(peer, headers)
        if route == "dashboard":
            if not self.admin:
                return "absent"
            return "ok" if self.admin_any(headers) else "unauthorized"
        if local:
            return "ok"
        if route == "metrics":
            if not (self.admin or self.metrics):
                return "absent"
            return ("ok" if self.metrics_bearer(headers) or self.admin_any(headers)
                    else "unauthorized")
        if route == "health_full":
            return ("ok" if self.metrics_bearer(headers) or self.admin_any(headers)
                    else "unauthorized")
        if not self.admin:
            return "absent"
        if route == "cache_read":
            return "ok" if self.admin_any(headers) else "unauthorized"
        if route == "cache_clear":
            return "ok" if self.admin_bearer(headers) else "unauthorized"
        return "unauthorized"
