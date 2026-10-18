"""autodetect — see the inference engines already running on this box, no wiring required.

An observer route, not a config system: it lists what LISTENs on localhost, fingerprints each
port as an inference engine (metrics-family → OpenAI-shape → engine-specific endpoint →
known-port hint, in that order of evidence), and peeks one stats snapshot per engine. It
NEVER registers, relays to, or configures anything — wiring a detected engine stays the
operator's explicit act in the Backends view.

Off by default: the routes answer 404 'no route' unless TANDEM_AUTODETECT=1.
Stdlib-only, like every proxy module. All environment, proc and network seams are
injectable so the tests can run against a fabricated /proc and fake engines.

    TANDEM_AUTODETECT=1        enable the routes
    TANDEM_AUTODETECT_PORTS    comma list replacing the hint table's reach (probe ONLY these)
    TANDEM_AUTODETECT_SELF     this proxy's own port, skipped when set
"""

import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from proxy import kinds as K        # T10: kind detection on top of the fingerprint

# --------------------------------------------------------------- configuration

TRUE = ("1", "on", "true", "yes")

def enabled(env) -> bool:
    return str(env.get("TANDEM_AUTODETECT", "")).lower() in TRUE


# best-effort family hints by port — used ONLY when live evidence cannot decide, and
# stamped into `how` so the dashboard can say it guessed. (tensorfold ALSO has a
# real fingerprint now — the fork's adapter reads its `tensorfold:` metrics
# families + /health live rates; the hint is only the fallback for a server
# scraped before its first metric sample exists.)
PORT_HINTS = {
    8000: "vllm", 30000: "sglang", 30002: "tandemllm", 30001: "tandemllm",
    8122: "tensorfold", 8874: "tabby", 11434: "ollama", 8080: "llama.cpp",
    5001: "koboldcpp",
}

# metrics-name families: first metric-name prefix seen (per line) that matches wins.
# vLLM/SGLang ship official Prometheus clients, so a colon in the engine prefix
# is a client default, not a convention we have to trust: an unprefixed
# vLLM-shaped server is a misconfiguration (documented, not hunted).
METRIC_FAMILIES = (
    ("vllm:", "vllm"),
    ("sglang:", "sglang"),
    ("tandem_", "tandemllm"),
    ("tabby_", "tabby"),
    ("exl3_", "exl3"),
    # tensorfold exports its stats ONLY under the literal `tensorfold:` prefix —
    # verified live 2026-10-18 (:8122 parity run): NO unprefixed vLLM-named
    # mirror exists (the 0.3.0 fork's header comment was wrong). The four
    # STATS_KEYS still resolve because the suffix scan is endswith-based and
    # happily reads through the colon prefix; richer families the engine
    # exports today: kv_cache_usage_perc/ratio, preemptions_total,
    # mtp_accepted_total/drafted_total, and a /health {live:{...}} rate source.
    ("tensorfold:", "tensorfold"),
)

# one stats snapshot per family: metric-name suffix -> output key. Names are the
# intersection of what each engine actually exports (vLLM/SGLang <v0.10 and ≥v0.10
# naming both; the parse is a plain prefix walk, first hit per key wins).
STATS_KEYS = {
    "requests_running": ("num_requests_running", "requests_running", "running_requests",
                         "num_running_reqs"),          # sglang (0.3.0 fork's live table)
    "requests_waiting": ("num_requests_waiting", "requests_waiting", "waiting_requests",
                         "queue_size", "num_queue_reqs"),
    "prompt_tokens": ("prompt_tokens_total", "prompt_tokens"),
    "generation_tokens": ("generation_tokens_total", "generation_tokens"),
}

LABELS = {
    "tandemllm": "TandemLLM", "vllm": "vLLM", "sglang": "SGLang", "tensorfold": "TensorFold",
    "openai": "OpenAI-compatible", "ollama": "Ollama", "llama.cpp": "llama.cpp",
    "tabby": "TabbyAPI", "exl3": "EXL3 server", "koboldcpp": "KoboldCpp",
    "http": "HTTP service", "open": "open port",
}

SCAN_WORKERS = 8
PROBE_TIMEOUT = 0.8          # per request
MAX_PROBED_PORTS = 40        # a laptop with 200 listeners is not our workload
CACHE_S = 5.0

# ------------------------------------------------------------------ scanning


def listen_sockets(proc: str) -> list[dict]:
    """LISTEN sockets from <proc>/net/tcp{,6}: {port, bind, ip, inode}. Local addresses only
    on loopback or the wildcard — a service bound to a routable address is still visible but
    we probe it as 127.0.0.1 when the box serves it there too (we only ever probe loopback)."""
    out = []
    for name in ("net/tcp", "net/tcp6"):
        try:
            with open(os.path.join(proc, name), "r", encoding="ascii") as fh:
                lines = fh.read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            p = line.split()
            if len(p) < 10 or p[3] != "0A":          # 0A = TCP_LISTEN
                continue
            host_hex, port_hex = p[1].rsplit(":", 1)
            try:
                port = int(port_hex, 16)
            except ValueError:
                continue
            out.append({"port": port, "bind": _unhex_host(host_hex), "ip": name[-3:],
                        "inode": p[9]})
    return out


def _unhex_host(h: str) -> str:
    """0100007F -> 127.0.0.1 (little-endian v4); v6 32-hex -> '::1' or '*' best effort."""
    if len(h) == 8:
        b = bytes.fromhex(h)
        return ".".join(str(x) for x in reversed(b))
    if h == "00000000000000000000000001000000":
        return "::1"
    return "*"


def socket_owner(proc: str, inodes: set[str]) -> dict | None:
    """pid + shortened cmdline owning any of these socket inodes. /proc scan, cheap."""
    try:
        pids = sorted((int(x) for x in os.listdir(proc) if x.isdigit()), reverse=True)
    except OSError:
        return None
    for pid in pids:
        d = os.path.join(proc, str(pid), "fd")
        try:
            fds = os.listdir(d)
        except OSError:
            continue
        for fd in fds:
            try:
                t = os.readlink(os.path.join(d, fd))
            except OSError:
                continue
            if t.startswith("socket:[") and t[8:-1] in inodes:
                try:
                    with open(os.path.join(proc, str(pid), "cmdline"), "rb") as fh:
                        cmd = fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
                except OSError:
                    cmd = ""
                return {"pid": pid, "cmd": cmd[:160]}
    return None


def _get(url: str, timeout: float) -> tuple[int, str] | None:
    """(status, body-head) or None on any network failure. GET only, 64 KB head."""
    req = urllib.request.Request(url, headers={"User-Agent": "tandemllm-proxy/autodetect"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:       # noqa: S310 (fixed http://127.0.0.1)
            return r.status, r.read(65536).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read(4096).decode("utf-8", "replace")
        except Exception:                                             # noqa: BLE001
            return e.code, ""
    except Exception:                                                 # noqa: BLE001 — timeout/refused
        return None


def _get_json(url: str, timeout: float):
    got = _get(url, timeout)
    if not got or got[0] >= 400:
        return None
    try:
        return json.loads(got[1])
    except ValueError:
        return None


# ------------------------------------------------------------------ fingerprint


def _post_json(url: str, body: str, timeout: float) -> tuple[int, str] | None:
    """One POST with a tiny JSON body; None on any socket trouble — the
    autodetect house rule: an unreachable port answers nothing, never a lie."""
    import urllib.error
    import urllib.request
    try:
        req = urllib.request.Request(
            url, data=body.encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(2048).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:
        return None


def _post(url: str, timeout: float) -> tuple[int, str] | None:
    return _post_json(url, "{}", timeout)


def fingerprint(url: str, port: int, *, fetch=_get, fetch_json=_get_json,
                timeout: float = PROBE_TIMEOUT) -> dict:
    """Classify one loopback port. Evidence order (strongest first), early exit on match:

    1. /metrics name-table          vllm: / sglang: / tandem_ / tabby_ / exl3_
    2. OpenAI shape (/v1/models)    + engine-specific tie-breaks
       /api/version -> ollama, /props -> llama.cpp
    3. /health json {mode,version}  -> tandemllm (0.9.1 answer)
    4. port hint                    only if nothing answered decisively
    5. any HTTP 2xx/404             -> 'http' (something serves HTTP, not an engine)
    """
    r: dict = {"family": None, "how": None, "model": None, "version": None,
               "metrics_url": None, "endpoints": {}}

    models = fetch_json(f"{url}/v1/models", timeout)
    if isinstance(models, dict) and isinstance(models.get("data"), list):
        ids = [m.get("id") for m in models["data"] if isinstance(m, dict) and m.get("id")]
        r["model"] = ids[0] if ids else None
        r["endpoints"]["models"] = True

    metrics = fetch(f"{url}/metrics", timeout)
    if metrics and metrics[0] < 400:
        for prefix, fam in METRIC_FAMILIES:
            if re.search(rf"(?m)^{re.escape(prefix)}", metrics[1]):
                r["family"], r["how"] = fam, f"metrics:{prefix}"
                r["metrics_url"] = f"{url}/metrics"
                return r
        if models:
            r["family"], r["how"] = "openai", "models+metrics(no known table)"

    health = fetch(f"{url}/health", timeout)
    if health:
        r["endpoints"]["health"] = health[0]
        body = health[1][:2000]
        if r["family"] is None:
            try:
                hj = json.loads(body)
                if isinstance(hj, dict) and "version" in hj:
                    r["version"] = str(hj.get("version"))
                    if "mode" in hj:
                        r["family"], r["how"] = "tandemllm", "health:{mode,version}"
            except ValueError:
                pass

    if r["family"] is None and models:
        ver = fetch_json(f"{url}/api/version", timeout)
        props = fetch_json(f"{url}/props", timeout)
        if isinstance(ver, dict) and "version" in ver:
            r["family"], r["how"] = "ollama", "api/version"
            r["version"] = str(ver["version"])
        elif isinstance(props, dict) and ("model_path" in props or "model_type" in props):
            r["family"], r["how"] = "llama.cpp", "props"
            r["model"] = r["model"] or str(props.get("model_path") or props.get("model_type"))
        else:
            r["family"], r["how"] = "openai", "v1/models shape"

    if r["family"] is None and port in PORT_HINTS and (health or models):
        r["family"], r["how"] = PORT_HINTS[port], f"port-hint:{port} (weak)"

    if r["family"] is None and (health or models or (metrics and metrics[0] < 500)):
        r["family"], r["how"] = "http", "answers HTTP, no engine fingerprint"

    return r


def _stats_snapshot(url: str, family: str, *, fetch=_get, timeout: float = PROBE_TIMEOUT) -> dict:
    """One read of /metrics: first hit per known key, label-bearing series summed across
    engines/shards (vLLM v0.10 emits `name{engine="0"} 3.0`). No counters diffing —
    absolute values, the dashboard shows them as such."""
    out: dict = {}
    if family not in ("vllm", "sglang", "tandemllm", "tabby", "exl3", "openai",
                      "tensorfold", "llama.cpp"):
        return out
    got = fetch(f"{url}/metrics", timeout)
    if not got or got[0] >= 400:
        return out
    seen: set[str] = set()
    for line in got[1].splitlines():
        if not line or line[0] == "#":
            continue
        name, _, val = line.partition(" ")
        base = name.split("{", 1)[0].rsplit("_created", 1)[0]
        for key, suffixes in STATS_KEYS.items():
            for suf in suffixes:
                if base.endswith(suf):
                    try:
                        v = float(val.strip())
                    except ValueError:
                        break
                    seen.add(key)
                    out[key] = out.get(key, 0.0) + v
                    break
    return out


# ------------------------------------------------------------------- cache


class _Cache:
    def __init__(self):
        self.at = 0.0
        self.val: dict | None = None
        self.lock = threading.Lock()

    def get_or(self, ttl, make):
        with self.lock:
            now = time.time()
            if self.val is None or now - self.at >= ttl:
                self.at, self.val = now, make()
            return self.val if self.val is not None else {}


_CACHES: dict[tuple, _Cache] = {}
_C_LOCK = threading.Lock()


def _cache(key: tuple) -> _Cache:
    with _C_LOCK:
        return _CACHES.setdefault(key, _Cache())


# ---------------------------------------------------------------- the scan


def scan(env, *, proc: str = "/proc", self_port: int | None = None,
         fetch=_get, fetch_json=_get_json, now: float | None = None) -> dict:
    """One pass. Bounded: loopback listeners only, MAX_PROBED_PORTS, 800 ms probes,
    5 s cache. Everything else on the box stays invisible."""
    override = str(env.get("TANDEM_AUTODETECT_PORTS", "") or "").strip()
    wanted = [int(x) for x in re.split(r"[,\s]+", override) if x.strip().isdigit()] if override else None

    socks = listen_sockets(proc)
    by_port: dict[int, list[str]] = {}
    for s in socks:
        by_port.setdefault(s["port"], []).append(s["bind"])
    ports = sorted(wanted or by_port)
    try:                                                   # env wins over the kwarg when set
        own = int(str(env.get("TANDEM_AUTODETECT_SELF", "")).strip() or "0")
        if own:
            self_port = own
    except ValueError:
        pass
    ports = [p for p in ports if p != self_port][:MAX_PROBED_PORTS]

    def one(port: int) -> dict:
        url = f"http://127.0.0.1:{port}"
        fp = fingerprint(url, port, fetch=fetch, fetch_json=fetch_json)
        # T10: every engine gains a `kind`. Known families map directly; an
        # 'openai'-shaped leftover gets one refined probe pass (proxy/kinds.py)
        # — detection only, relay behavior untouched.
        if fp["family"] in ("openai", "http", "open", None):
            # _get_json(url, timeout) takes the timeout explicitly; kinds calls
            # fetch_json(url) — close that gap here, once
            refined = (lambda u, _fj=fetch_json: _fj(u, PROBE_TIMEOUT))
            kind = K.detect(url, fetch_json=refined) \
                if fp["family"] == "openai" else "generic"
        else:
            kind = K.kind_for_family(fp["family"])
        bind = sorted(set(by_port.get(port) or ["?"]))
        entry = {"port": port, "bind": ", ".join(bind), "kind": kind, **fp}
        if fp["family"] in ("open", "http") or fp["family"] is None:
            entry["family"] = fp["family"] or "open"
        if fp["family"] in LABELS and fp["family"] not in ("http", "open"):
            entry["label"] = LABELS[fp["family"]]
            entry["stats_url"] = fp.get("metrics_url") or f"{url}/metrics"
        return entry

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
        entries = list(ex.map(one, ports))
    engines = [e for e in entries if e.get("label")]
    return {"generated": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now or time.time())),
            "scanned": len(ports), "took_ms": int((time.time() - t0) * 1000),
            "engines": [e for e in engines if e["family"] != "http"],
            "http": [e for e in engines if e["family"] == "http"],
            "other": [{"port": e["port"], "bind": e["bind"]}
                      for e in entries if not e.get("label")]}


def scan_cached(env, **kw) -> dict:
    return _cache(("scan",)).get_or(CACHE_S, lambda: scan(env, **kw))


def stats_for(env, port: int, family: str, *, fetch=_get) -> dict:
    url = f"http://127.0.0.1:{port}"
    return _cache(("stats", port)).get_or(2.0, lambda: _stats_snapshot(url, family, fetch=fetch))


# ----------------------------------------------------------------- routing
# Same shape as proxy/traces: a handle() the app calls for GET /v1/autodetect*.


def handle(req, method: str, path: str, raw: str, env, **scan_kw):
    from urllib.parse import parse_qs, urlsplit            # request-local import cost: nil
    if method != "GET":
        return req._json(405, {"error": {"type": "method", "message": "GET only"}})
    q = {k: v[0] for k, v in parse_qs(urlsplit(raw).query).items()}
    if path == "/v1/autodetect":
        return req._json(200, scan_cached(env, **scan_kw))
    if path == "/v1/autodetect/stats":
        try:
            port = int(q.get("port", ""))
        except ValueError:
            return req._json(400, {"error": {"type": "bad", "message": "port must be an int"}})
        fam = q.get("family", "openai")
        return req._json(200, {"port": port, "stats": stats_for(env, port, fam, fetch=_get)})
    return req._json(404, {"error": {"type": "not_found", "message": "no route"}})
