"""The proxy as one object, shared by its two entry points.

`python -m proxy` (proxy/app.py) serves it alone, without torch and without an engine.
`server/app.py --backends PATH` serves it inside the engine's process: a request whose model routes
to a backend goes to `Service.serve` on the handler's own thread and never touches the engine's
lock, queue or `max_queue`; a request for `TandemLLM` runs as before. One port, one ledger, one
dashboard.

    svc = Service(cfg, ledger=..., live=..., auth=..., reserved=("TandemLLM", raw_name))
    svc.start()                                  the health poller
    svc.authorize(handler) -> (ok, user)         the users' keys on the OpenAI routes
    svc.route(path, body) -> Route | None        None: not a backend's model (the engine's, or 404)
    svc.serve(handler, path, raw, body, route, user)
    svc.models(created) -> [...]                 the backends' part of /v1/models

THE USERS' KEYS. With at least one user in the config (or `TANDEM_API_KEYS` set), a request to the
OpenAI routes without a known key is a 401, for a backend's model and for the engine's alike. The
box's own tools stay open the way the admin routes treat them: a request from loopback with
neither `X-Forwarded-For` nor `X-Real-IP` (trusted-local, server/auth.py) needs no key. The
label of the key fills the ledger's `user` column; the client id stays the token's hash.
"""

from __future__ import annotations

import datetime as _dt
import threading
import time
import uuid

from proxy import adapters as adapters_mod
from proxy import config as config_mod
from proxy import health as health_mod
from proxy import relay as relay_mod
from proxy import route as route_mod
from proxy.measure import RelayRecord
from server import ledger as ledger_mod

PROXY_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/responses")

MS_BUCKETS = (0.25, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, 250, 1000)
TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 60, 120)


class Metrics:
    """The proxy's families, added to a `server.metrics.Registry` (the engine's own in the
    embedded mode, so one /metrics page carries both). The engine's families keep their labels:
    a relayed request is never counted in them."""

    def __init__(self, registry, svc: "Service"):
        from server import metrics as m
        self.svc = svc
        self.requests = registry.add(m.Counter(
            "qse_proxy_requests_total", "requests relayed to a backend, by its id, its kind and "
            "the finish reason", ("backend", "kind", "finish_reason")))
        self.tokens = registry.add(m.Counter(
            "qse_proxy_tokens_total", "tokens a backend reported in `usage`, by type (prompt, "
            "cached, completion, reasoning)", ("backend", "type")))
        self.cost = registry.add(m.Counter(
            "qse_proxy_cost_micros_total", "cost at the model's price, micro-dollars",
            ("backend",)))
        self.retries = registry.add(m.Counter(
            "qse_upstream_retries_total", "tries again before any byte reached the client, by "
            "backend and reason (429, 502, 503, connect)", ("backend", "reason")))
        self.proxy_ms = registry.add(m.Histogram(
            "qse_proxy_overhead_ms", "the proxy's own time per request: arrival to the upstream "
            "send, plus the first frame's delay through the proxy", MS_BUCKETS, ("backend",)))
        self.ttft = registry.add(m.Histogram(
            "qse_proxy_ttft_seconds", "arrival at the proxy to the first token frame (the "
            "network and the backend's queue inside)", TTFT_BUCKETS, ("backend",)))
        registry.add(m.Gauge("qse_backend_up", "1 when the backend's last health check passed",
                             ("backend",), collect=self._up))
        registry.add(m.Gauge("qse_backend_inflight", "requests relayed to the backend now",
                             ("backend",), collect=self._inflight))

    def _up(self):
        return {(bid,): 1.0 if s.get("state") == "up" else 0.0
                for bid, s in self.svc.health.state.items() if s.get("state") != "disabled"}

    def _inflight(self):
        return {(b["id"],): float(self.svc.relay.inflight.count(b["id"]))
                for b in self.svc.cfg.backends()}

    def record(self, rec: RelayRecord) -> None:
        b = rec.backend or "unknown"
        self.requests.inc(backend=b, kind=rec.backend_kind or "unknown",
                          finish_reason=rec.finish_reason or "unknown")
        for t, v in (("prompt", rec.prompt_tokens), ("cached", rec.cached_tokens),
                     ("completion", rec.completion_tokens), ("reasoning", rec.reasoning_tokens)):
            if v:
                self.tokens.inc(float(v), backend=b, type=t)
        if rec.cost_micros:
            self.cost.inc(float(rec.cost_micros), backend=b)
        if rec.proxy_ms is not None:
            self.proxy_ms.observe(rec.proxy_ms, backend=b)
        if rec.ttft_ms is not None:
            self.ttft.observe(rec.ttft_ms / 1e3, backend=b)


class Service:
    def __init__(self, cfg: "config_mod.ConfigDB", *, ledger=None, live=None, auth=None,
                 reserved=(), registry=None, log=None, engine_version: str = "",
                 code_sha: str = "", health_kw: dict | None = None):
        self.cfg = cfg
        self.ledger = ledger
        self.live = live
        self.auth = auth
        self.log = log or (lambda msg: print(msg, flush=True))
        self.engine_version, self.code_sha = engine_version, code_sha
        self.table = route_mod.Table(cfg, reserved)
        self.relay = relay_mod.Relay(self)
        self.health = health_mod.Health(self, **(health_kw or {}))
        self._adapters: dict = {}
        self._lock = threading.Lock()
        self.metrics = Metrics(registry, self) if registry is not None else None
        self.started = time.time()

    def start(self) -> "Service":
        self.health.start()
        return self

    def stop(self) -> None:
        self.health.stop()
        self.cfg.flush_seen(force=True)

    # ----------------------------------------------------------------- pieces the relay asks for
    def adapter(self, backend: dict):
        key = (backend["id"], backend["updated_at"], backend["kind"])
        with self._lock:
            a = self._adapters.get(key)
            if a is None:
                a = adapters_mod.make(backend, log=self.log)
                self._adapters = {k: v for k, v in self._adapters.items() if k[0] != key[0]}
                self._adapters[key] = a
            return a

    def price_of(self, bid: str, model: str | None) -> dict | None:
        if not model:
            return None
        rows = self.table.snapshot()["by_backend"].get(bid) or {}
        m = rows.get(model)
        if m is None:
            for r in rows.values():
                if r["public_id"] == model:
                    m = r
                    break
        return m["price"] if m is not None else None

    def count_retry(self, bid: str, reason: str) -> None:
        if self.metrics is not None:
            self.metrics.retries.inc(backend=bid, reason=reason)

    # ----------------------------------------------------------------- the OpenAI routes
    def authorize(self, handler) -> tuple[bool, str | None]:
        """(may it go on, the user's label). On False a 401 is already written."""
        try:
            active = self.cfg.users_active()
        except Exception:                                          # noqa: BLE001
            active = False
        if not active:
            return True, None
        token = None
        auth = ""
        try:
            auth = (handler.headers.get("Authorization") or "").strip()
        except Exception:                                          # noqa: BLE001
            pass
        if auth[:7].lower() == "bearer ":
            token = auth[7:].strip()
        label = self.cfg.user_for_key(token)
        if label is not None:
            return True, label
        peer = (getattr(handler, "client_address", None) or ("",))[0]
        if self.auth is not None and self.auth.trusted_local(peer, handler.headers):
            return True, None
        body = (b'{"error": {"message": "a proxy API key is needed (Authorization: Bearer '
                b'<key>); the admin makes one under Backends, Users", "type": '
                b'"invalid_request_error", "code": "invalid_api_key"}}')
        try:
            handler.send_response(401)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("WWW-Authenticate", 'Bearer realm="TandemLLM"')
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
        except OSError:
            pass
        return False, None

    def route(self, path: str, body: dict):
        """The route of a request, or None when no backend's model is asked for. NotFound for a
        model the table knows is ambiguous."""
        if path not in PROXY_PATHS or not isinstance(body, dict):
            return None
        model = body.get("model")
        if not isinstance(model, str) or not model:
            return None
        try:
            return self.table.resolve(model)
        except route_mod.NotFound as exc:
            if "served by" in exc.message:
                raise
            return None

    def serve(self, handler, path: str, raw: bytes, body: dict, route, user: str | None,
              t_arrival: float | None = None) -> RelayRecord:
        rid = ("chatcmpl-px-" if path.endswith("chat/completions") else "px-") + \
            uuid.uuid4().hex[:24]
        rec = RelayRecord(rid, relay_mod.endpoint_of(path),
                          bool(body.get("stream")) and path != "/v1/embeddings",
                          t_arrival=t_arrival)
        rec.model = route.public
        rec.model_served = route.public
        rec.user = user
        rec.client_id, rec.client_kind = ledger_mod.client_of(handler.headers, body)
        rec.sock = getattr(handler, "connection", None)
        mt = body.get("max_tokens", body.get("max_completion_tokens",
                                             body.get("max_output_tokens")))
        rec.max_tokens = mt if isinstance(mt, int) and not isinstance(mt, bool) else None
        rec.backend = route.backend
        b = self.table.backend_for(route.backend)
        rec.backend_kind = b["kind"] if b else None
        if self.live is not None:
            try:
                self.live.register(rec)
            except Exception:                                      # noqa: BLE001
                pass
        try:
            self.relay.serve(handler, path, raw, body, route, rec)
        except Exception as exc:                                   # noqa: BLE001
            rec.finish_reason = rec.finish_reason or "error"
            rec.error_type = rec.error_type or type(exc).__name__
            self.log(f"[proxy] relay failed: {type(exc).__name__}: {exc}")
            if rec.status in (None, 200) and rec.t_sent is None:
                rec.status = 500
        finally:
            rec.end()
            if self.live is not None:
                try:
                    self.live.finish(rec)
                except Exception:                                  # noqa: BLE001
                    pass
            self.account(rec)
        return rec

    def account(self, rec) -> None:
        """One ledger row and the proxy's metrics. Never fails a request."""
        try:
            row = rec.row(self.engine_version, self.code_sha)
        except Exception as exc:                                   # noqa: BLE001
            self.log(f"[proxy] row failed: {type(exc).__name__}: {exc}")
            return
        if self.metrics is not None:
            try:
                self.metrics.record(rec)
            except Exception:                                      # noqa: BLE001
                pass
        if self.ledger is not None:
            self.ledger.submit(row)
        self.log(f"[proxy] {rec.request_id} {rec.backend}/{rec.backend_model} "
                 f"status={rec.status} finish={rec.finish_reason} "
                 f"tokens={rec.prompt_tokens}/{rec.completion_tokens} "
                 f"ttft_ms={_f(rec.ttft_ms)} decode_tps={_f(rec.decode_tps)} "
                 f"proxy_ms={_f(rec.proxy_ms)} retries={rec.retries}"
                 + (f" fallback_from={rec.fallback_from}" if rec.fallback_from else ""))

    def account_refusal(self, handler, path: str, body, status: int, user=None,
                        finish="error", error_type=None, model=None) -> None:
        """A ledger row for a request answered before any backend (a 401, a 404 model)."""
        rec = RelayRecord("px-" + uuid.uuid4().hex[:24], relay_mod.endpoint_of(path),
                          bool(isinstance(body, dict) and body.get("stream")))
        rec.model = model or (body.get("model") if isinstance(body, dict) else "") or ""
        rec.status, rec.finish_reason, rec.error_type, rec.user = status, finish, error_type, user
        rec.client_id, rec.client_kind = ledger_mod.client_of(handler.headers,
                                                              body if isinstance(body, dict)
                                                              else None)
        rec.end()
        self.account(rec)

    def models(self, created: int) -> list[dict]:
        return self.table.listing(created)

    # ----------------------------------------------------------------- views
    def cost_today(self, tz=None) -> dict:
        """backend id -> micro-dollars since local midnight, from the ledger (None without one)."""
        if self.ledger is None:
            return {}
        now = _dt.datetime.now().astimezone() if tz is None else _dt.datetime.now(tz)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            rows = self.ledger.reader().execute(
                "SELECT backend, SUM(cost_micros), COUNT(cost_micros) FROM requests WHERE "
                "ts_ms >= ? AND backend IS NOT NULL GROUP BY backend",
                (int(midnight.timestamp() * 1000),)).fetchall()
        except Exception:                                          # noqa: BLE001
            return {}
        return {b: (int(s) if n else None) for b, s, n in rows}

    def backends_view(self) -> list[dict]:
        """One row per backend for the summary, the system snapshot and the Backends screen."""
        cost = self.cost_today()
        out = []
        for b in self.cfg.backends():
            h = self.health.view(b["id"])
            n_models = sum(1 for m in self.cfg.models(b["id"]) if m["present"] and m["enabled"])
            out.append({"id": b["id"], "kind": b["kind"], "label": b["label"],
                        "enabled": b["enabled"], "cloud": b["cloud"],
                        "state": h["state"] if b["enabled"] else "disabled",
                        "since": _iso(h["since"]), "last_error": h["last_error"],
                        "latency_ms": h["latency_ms"], "in_flight": self.relay.inflight.count(
                            b["id"]), "max_inflight": b["max_inflight"], "models": n_models,
                        "harness_mode": b["harness"]["mode"],
                        "cost_today_micros": cost.get(b["id"]),
                        "scrape_age_s": None, "plugin_age_s": None})
        return out


def _f(x):
    return "-" if x is None else f"{x:.1f}"


def _iso(ts):
    if ts is None:
        return None
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
