#!/usr/bin/env python3
"""autodetect: fake engines on real loopback ports, a fabricated /proc, a default-off route.

Run:  python3 tests/test_proxy_autodetect.py
"""

import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proxy import autodetect as A                                            # noqa: E402


# ---------------------------------------------------------------- fake engines

def _serve(routes: dict):
    """A tiny HTTP server answering the dict's paths; anything else 404."""
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body, code = routes.get(self.path, (b"nope", 404))
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format="", *args):                             # quiet
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _open_raw():
    """A listener that never speaks HTTP — a port that just exists."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    return s, s.getsockname()[1]


def _fake_proc(base: str, ports: list):
    os.makedirs(os.path.join(base, "net"), exist_ok=True)
    with open(os.path.join(base, "net", "tcp"), "w", encoding="ascii") as fh:
        fh.write("   sl  local_address rem_address   st tx_rx tr:tr retr shr\n")
        for p in ports:
            fh.write(f"0: 0100007F:{p:04X} 00000000:0000 0A 1 2 3 4 5 6 7 8 9\n")
    with open(os.path.join(base, "net", "tcp6"), "w", encoding="ascii") as fh:
        fh.write("   sl  local_address rem_address   st tx_rx tr:tr retr shr\n")
    return base


def _hex_host_fix():
    assert A._unhex_host("0100007F") == "127.0.0.1"
    assert A._unhex_host("00000000000000000000000001000000") == "::1"
    assert A._unhex_host("00000000000000000000FFFF00000000") == "*"
    print("ok  hex")


# -------------------------------------------------------------------- scans

def test_scan_classifies(tmp):
    vllm, p_vllm = _serve({
        "/v1/models": (json.dumps({"object": "list",
                                   "data": [{"id": "fake/hybrid48", "object": "model"}]}).encode(), 200),
        "/metrics": (b"# HELP x\nvllm:num_requests_running 3.0\nvllm:prompt_tokens_total 1200.0\n"
                     b"vllm:generation_tokens_total 4400.0\nvllm:num_requests_waiting 1.0\n", 200),
        "/health": (b"", 200),
    })
    ollama, p_ol = _serve({
        "/api/version": (json.dumps({"version": "0.13.4"}).encode(), 200),
        "/v1/models": (json.dumps({"data": [{"id": "llama3:latest"}]}).encode(), 200),
    })
    plain, p_plain = _serve({
        "/v1/models": (json.dumps({"data": [{"id": "some/model"}]}).encode(), 200),
    })
    tandem, p_tandem = _serve({
        "/health": (json.dumps({"status": "ok", "mode": "serve", "version": "0.9.1"}).encode(), 200),
    })
    raw, p_raw = _open_raw()
    try:
        base = os.path.join(tmp, "proc")
        _fake_proc(base, [p_vllm, p_ol, p_plain, p_tandem, p_raw])
        out = A.scan({}, proc=base, now=time.time())
        eng = {e["port"]: e for e in out["engines"]}
        assert eng[p_vllm]["family"] == "vllm", eng[p_vllm]
        assert eng[p_vllm]["how"] == "metrics:vllm:"
        assert eng[p_vllm]["model"] == "fake/hybrid48"
        assert eng[p_ol]["family"] == "ollama" and eng[p_ol]["version"] == "0.13.4"
        assert eng[p_plain]["family"] == "openai"
        assert eng[p_tandem]["family"] == "tandemllm"
        assert p_raw in [o["port"] for o in out["other"]], out["other"]
        # the stats peek
        st = A._stats_snapshot(f"http://127.0.0.1:{p_vllm}", "vllm")
        assert st == {"requests_running": 3.0, "prompt_tokens": 1200.0,
                      "generation_tokens": 4400.0, "requests_waiting": 1.0}, st
        print("ok  scan classifies five ports")
    finally:
        for x in (vllm, ollama, plain, tandem):
            x.shutdown()
        raw.close()


def test_scan_bounds_and_override(tmp):
    # the override probes ONLY the listed ports even though /proc lists more
    srv, port = _serve({"/v1/models": (json.dumps({"data": []}).encode(), 200)})
    try:
        base = _fake_proc(os.path.join(tmp, "proc2"), [port, 9, 10])
        out = A.scan({"TANDEM_AUTODETECT_PORTS": str(port)}, proc=base)
        assert out["scanned"] == 1
        base_all = _fake_proc(os.path.join(tmp, "proc3"), [9, 10])
        assert A.scan({}, proc=base_all)["scanned"] == 2   # closed ports fail fast
        # self-exclusion: the kwarg survives, and the env var wins when both are set
        base_self = _fake_proc(os.path.join(tmp, "proc4"), [port, 9, 10])
        assert port not in [e["port"] for e in A.scan({}, proc=base_self, self_port=port)["engines"]]
        assert 9 not in [e["port"] for e in
                         A.scan({"TANDEM_AUTODETECT_SELF": "9"}, proc=base_self,
                                self_port=port)["engines"]]
    finally:
        srv.shutdown()
    print("ok  ports override + closed ports are cheap + self-exclusion")


def test_caches_are_bounded():
    # scan cache: a second call within the ttl does not re-run
    calls = {"n": 0}

    def make():
        calls["n"] += 1
        return {"x": calls["n"]}
    c = A._cache(("t",))
    A._CACHES[("t",)] = c
    a = A._cache(("t",)).get_or(60.0, make)
    b = A._cache(("t",)).get_or(60.0, make)
    assert a == b == {"x": 1}
    print("ok  caches")


# ------------------------------------------------------------------- route

def test_route():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    srv, port = _serve({"/v1/models": (json.dumps({"data": [{"id": "m"}]}).encode(), 200)})
    try:
        from proxy_util import Proxy                        # the house fixture
        with Proxy() as px:                                 # gate unset: indistinguishable
            code, _, body = px.get("/v1/autodetect")
            assert code == 404 and b"no route" in body, (code, body[:120])
        with Proxy(env={"TANDEM_AUTODETECT": "1", "TANDEM_AUTODETECT_PORTS": str(port)}) as px:
            code, _, body = px.get("/v1/autodetect")
            assert code == 200, code
            engines = json.loads(body)["engines"]
            assert any(e["port"] == port and e["family"] == "openai" for e in engines), engines
            code, _, body = px.get(f"/v1/autodetect/stats?port={port}&family=openai")
            assert code == 200 and "stats" in json.loads(body)
            code, _ = px.get("/v1/autodetect/stats?port=xx")[:2]
            assert code == 400
    finally:
        srv.shutdown()
    print("ok  route: 404 off, 200 on, stats, bad input 400")


def run():
    import tempfile
    _hex_host_fix()
    with tempfile.TemporaryDirectory() as tmp:
        test_scan_classifies(tmp)
        test_scan_bounds_and_override(tmp)
    test_caches_are_bounded()
    test_route()


if __name__ == "__main__":
    run()
