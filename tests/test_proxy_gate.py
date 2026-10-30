"""tools/proxy_gate.py on LAN jitter (NP1's issue 4): the decode check is relative, on the median,
and a request that misses the per-request bound is sent again.

A fake engine streams 40 tokens 10 ms apart and reports its own `predicted_per_second` on the
finish chunk, as the real one does. "Jitter" holds the last token frame back by 80 ms after the
engine's clock has stopped -- what a late frame on a LAN looks like to the proxy, which times the
frames as they arrive (about -17 % on this request's rate).

On a loaded box (FX14, 2026-10-08: the CPU suite at nice 19 next to a night A/B) the scheduler
adds a few ms between the engine's clock and the proxy's, which moved requests that were NOT made
late past 2 % and the proxy's own p50 past 2 ms; with 5 ms gaps and a 20 ms late frame that was
enough to fail the PASS case and blur the FAIL cases. So: the stream is twice as long and the late
frame four times as late (the same signal, a quarter of the relative jitter), and a run whose
failure comes with load seen in the run itself -- a request that was not made late off by more
than 2 % or retried, or the proxy's p50 at 2 ms or more -- is run again, up to 3 times. A failure
on a run without that signature fails at once, so a gate that stops retrying, keeps the wrong try
or judges wrong is never retried into a pass.

The gate's bound on one request is --tol-one (5 %); 2 % is its bound on the median. A request
that is retried keeps its best try, and the test holds that try to the gate's own 5 %, not to 2 %:
on the loaded box a clean retry landed at -4.7 % (2026-10-10), -4.4 % (10-22) and +2.6 % (10-23,
the proxy's own cost on that request 16 ms against about 0.3 ms on a quiet run, its first frame read
39 ms after the request), the gate passed each time, and the test failed on the 2 %. The proxy's
cost on any one request at 2 ms or more is also load seen in the run, the late request included.
Which try the gate keeps and when it stops is checked on scripted rates (best_of), with no clock.

    one late frame, first try only     the old gate (2 % on every request, one try) fails; the new
                                       one passes on the retry and counts it
    the retry's first frame read late  the clean retry lands past 2 % and within 5 % (the 10-23
                                       run): the gate passes it
    a request late on every try        fails on any setting (a real proxy fault is not hidden)
    best_of()                          the kept try, the stop, a strict failure, on scripted rates
    judge()                            the median, the per-request bound, the strict checks

CPU only, loopback (the fake engine and the proxy on ephemeral ports), about 20 s.

Run: python tests/test_proxy_gate.py
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile
import time
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import proxy_gate as G  # noqa: E402

N_TOK, GAP_S, LATE_S, HOLD_S = 40, 0.010, 0.080, 0.010
ATTEMPTS = 3
TOL_ONE = 0.05      # tools/proxy_gate.py's --tol-one default: the bound on one request's best try


def _serve(late: dict, ports, held: dict | None = None) -> None:
    """The fake engine's process: frames at GAP_S apart, the last one held back LATE_S on the tries `late`
    names ({request index: tries}, None: every try). On the tries `held` names the first frame leaves
    HOLD_S after the engine's clock stamped it and the second on time, as when the proxy reads the first
    frame late (2026-10-23): the engine's rate is unchanged and the proxy's comes out about 2.5 % high."""
    held = held or {}
    tries: dict[int, int] = {}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"object": "list", "data": [{"id": "fake-38",
                                                           "object": "model"}]}).encode()
            self.send_response(200 if self.path.endswith("/models") else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            t0 = time.perf_counter()
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            m = re.search(r"\(run (\d+)\)", req["messages"][-1]["content"])
            i = int(m.group(1)) if m else -1
            k = tries[i] = tries.get(i, 0) + 1
            which = late.get(i, ())
            is_late = i in late and (which is None or k in which)
            is_held = i in held and (held[i] is None or k in held[i])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()

            def frame(obj):
                self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                self.wfile.flush()

            sent = []
            for n in range(N_TOK):
                if n:
                    time.sleep(GAP_S - (HOLD_S if n == 1 and is_held else 0))
                if n == 0 and is_held:
                    sent.append(time.perf_counter())     # the engine's clock: the first token
                    time.sleep(HOLD_S)                    # read by the proxy HOLD_S later
                elif n == N_TOK - 1 and is_late:
                    sent.append(time.perf_counter())     # the engine's clock stops here
                    time.sleep(LATE_S)                    # the network holds the frame back
                else:
                    sent.append(time.perf_counter())
                frame({"id": "c", "object": "chat.completion.chunk", "model": "fake-38",
                       "choices": [{"index": 0, "delta": {"content": f"w{n} "}}]})
            tps = (N_TOK - 1) / (sent[-1] - sent[0])
            frame({"id": "c", "object": "chat.completion.chunk", "model": "fake-38",
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 31, "completion_tokens": N_TOK,
                             "total_tokens": 31 + N_TOK},
                   "timings": {"predicted_per_second": tps,
                               "ttft_ms": (sent[0] - t0) * 1e3, "cache_n": 0}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    ports.put(srv.server_address[1])
    srv.serve_forever()


class Engine:
    """The fake engine in a process of its own (OPS-231): in the test's process its frames shared the GIL
    with the gate's reader, and under load the reader's stalls showed as a slower decode on a clean try
    (request 2's retry at -4.4 %, 2026-10-22). late: {request index: tries that get the late frame}
    (None: every try)."""

    def __init__(self, late: dict, held: dict | None = None):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        ports = ctx.Queue()
        self.proc = ctx.Process(target=_serve, args=(late, ports, held), daemon=True)
        self.proc.start()
        self.port = ports.get(timeout=60)

    def close(self):
        self.proc.terminate()
        self.proc.join(10)


def _gate(late: dict, *args, held: dict | None = None) -> tuple[int, dict, str]:
    eng = Engine(late, held)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "gate.json")
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                rc = G.main(["--url", f"http://127.0.0.1:{eng.port}/v1", "--n", "6",
                             "--max-tokens", str(N_TOK), "--out", out, *args])
            rep = json.load(open(out))
        finally:
            eng.close()
    return rc, rep, buf.getvalue()


def _load_seen(rep: dict, late) -> str:
    """Why a run's numbers are the box's and not the gate's ('' when the run is clean): a request
    that was not made late off by more than 2 % or retried, or the proxy's own cost at 2 ms+ on any
    request, the late ones included (about 0.3 ms on a quiet run; 16 ms on the late request's kept retry,
    10-23)."""
    why = []
    s = rep["summary"]
    if s.get("proxy_ms_p50") is not None and s["proxy_ms_p50"] >= 2.0:
        why.append(f"proxy_ms p50 {s['proxy_ms_p50']:.2f}")
    for r in rep["rows"]:
        pms = (r.get("proxy") or {}).get("proxy_ms")
        if pms is not None and pms >= 2.0:
            why.append(f"request {r['i']} proxy_ms {pms:.2f}")
    for r in rep["rows"]:
        if r["i"] in late or "decode_rel" not in r:
            continue
        if r.get("tries", 1) > 1 or abs(r["decode_rel"]) > 0.02:
            why.append(f"request {r['i']} (not late) {r['decode_rel']:+.1%} in {r.get('tries', 1)} tries")
    return "; ".join(why)


def _settled(late: dict, check, *args, held: dict | None = None) -> str:
    """`check(rc, rep, log)` on a gate run; again (up to ATTEMPTS) only when it failed on a run
    that shows load, at once when the run was clean."""
    seen = []
    for k in range(ATTEMPTS):
        rc, rep, log = _gate(late, *args, held=held)
        try:
            out = check(rc, rep, log)
        except AssertionError:
            why = _load_seen(rep, late)
            if not why:
                raise
            seen.append(why)
            print(f"    run {k + 1} on a loaded box ({why}); again", flush=True)
            continue
        return out + (f" [run {k + 1}; load before: {' | '.join(seen)}]" if seen else "")
    raise AssertionError(f"{ATTEMPTS} runs, each with load seen: " + " | ".join(seen))


def test_one_late_frame_old_gate_fails_new_gate_retries():
    def old(rc, rep, log):
        s = rep["summary"]
        assert rc == 1 and s["verdict"] == "FAIL", (s, log[-1500:])
        r2 = [r for r in rep["rows"] if r["i"] == 2][0]
        assert r2["decode_rel"] < -0.05, r2
        return f"old gate FAIL (request 2 at {r2['decode_rel']:+.1%}, one try)"

    def new(rc, rep, log):
        s = rep["summary"]
        assert rc == 0 and s["verdict"] == "PASS", (s, log[-1500:])
        # request 2 is retried; on a loaded runner a request that was not made late may be as well, and
        # the gate passes it on its retry too (OPS-231): the check is the late one's retry and the pass
        assert s["retried"] >= 1 and s["loopback"] is True, s
        return _late_once_retried(rep, log) + f", median {s['decode_rel_median']:.2%}"

    return (_settled({2: (1,)}, old, "--tol-one", "0.02", "--repeats", "1") + "; "
            + _settled({2: (1,)}, new))


def _late_once_retried(rep: dict, log: str) -> str:
    """Request 2, late on try 1 only: retried, and the try kept is a later one. The late try reads
    about -16 %, past TOL_ONE, so a kept try within TOL_ONE is not it; the kept try is held to the
    gate's own per-request bound, not to the 2 % of its median (a clean try on a loaded box lands a
    few percent off)."""
    r2 = [r for r in rep["rows"] if r["i"] == 2][0]
    first = re.search(r"\[gate\]\s+2 try 1: decode ([-+][0-9.]+)%", log)
    assert first and float(first.group(1)) / 100 < -TOL_ONE, log[-1500:]
    assert r2["tries"] >= 2 and abs(r2["decode_rel"]) <= TOL_ONE, r2
    return (f"new gate PASS on the retry (request 2 at {first.group(1)}% on try 1, "
            f"{r2['decode_rel']:+.1%} kept of {r2['tries']})")


def test_a_retry_read_late_passes_within_the_request_bound():
    """The 10-23 run: request 2's late try is retried and its clean retry comes out past 2 % because the
    proxy read its first frame late. Within 5 %, so the gate keeps it and passes."""
    def check(rc, rep, log):
        s = rep["summary"]
        assert rc == 0 and s["verdict"] == "PASS", (s, log[-1500:])
        return _late_once_retried(rep, log)

    return _settled({2: (1,)}, check, held={2: (2,)})


def test_best_of_keeps_the_best_try_and_stops():
    """The gate's tries on scripted rates: no clock, so the same answer on any box."""
    def script(*rels, fail_at=None):
        it = iter(range(len(rels)))

        def send():
            k = next(it)
            return {"decode_rel": rels[k], "fail": ["status 500"] if k == fail_at else []}
        return send

    seen = []
    r = G.best_of(script(-0.16, 0.026), 0.05, 3, lambda k, rel: seen.append((k, rel)))
    assert (r["tries"], r["decode_rel"], seen) == (2, 0.026, [(1, -0.16)]), (r, seen)
    r = G.best_of(script(-0.16, -0.07, -0.09), 0.05, 3)       # never within: the best of 3 is kept
    assert (r["tries"], r["decode_rel"]) == (3, -0.07), r
    assert G.judge([_row(0, 0.0), _row(1, r["decode_rel"], tries=3)])["verdict"] == "FAIL"
    r = G.best_of(script(0.01, -0.2), 0.05, 3)                # within on the first try: one try
    assert (r["tries"], r["decode_rel"]) == (1, 0.01), r
    r = G.best_of(script(-0.16, 0.0, fail_at=1), 0.05, 3)     # a strict failure counts, no more tries
    assert (r["tries"], r["fail"]) == (2, ["status 500"]), r
    r = G.best_of(script(-0.16, 0.0), 0.05, 1)                # --repeats 1: one try, kept as it is
    assert (r["tries"], r["decode_rel"]) == (1, -0.16), r
    return "kept try, stop within the bound, best of 3, strict failure, one try"


def test_a_request_late_every_try_fails():
    def check(rc, rep, log):
        s = rep["summary"]
        assert rc == 1 and s["verdict"] == "FAIL", (s, log[-1500:])
        r4 = [r for r in rep["rows"] if r["i"] == 4][0]
        assert r4["tries"] == 3 and r4["decode_rel"] < -TOL_ONE, r4
        assert any(i == 4 for i, _ in s["fail_list"]), s["fail_list"]
        return f"request 4 late on 3 of 3 tries: FAIL ({r4['decode_rel']:+.1%})"

    return _settled({4: None}, check)


def _row(i, rel, fail=(), pms=0.4, tries=1):
    return {"i": i, "decode_rel": rel, "fail": list(fail), "tries": tries,
            "ttft_delta_ms": 1.0, "proxy": {"proxy_ms": pms}}


def test_judge():
    ok = [_row(i, 0.004 * (-1) ** i) for i in range(20)]
    assert G.judge(ok)["verdict"] == "PASS"
    # four requests at 3 % (the LAN failures of 2026-10-04) pass: the median is what counts
    lan = ok[:16] + [_row(16 + k, -0.03) for k in range(4)]
    assert G.judge(lan)["verdict"] == "PASS", G.judge(lan)
    # past the per-request bound fails, and so does a median past 2 %
    assert G.judge(ok[:19] + [_row(19, -0.08, tries=3)])["verdict"] == "FAIL"
    assert G.judge([_row(i, 0.025) for i in range(20)])["verdict"] == "FAIL"
    # a strict check fails whatever the rates say, and so does the proxy's own cost
    assert G.judge(ok[:19] + [_row(19, 0.0, fail=["token counts differ"])])["verdict"] == "FAIL"
    assert G.judge([_row(i, 0.0, pms=2.5) for i in range(20)])["verdict"] == "FAIL"
    assert G.loopback("http://127.0.0.1:8000/v1") and G.loopback("http://localhost:8000/v1")
    assert not G.loopback("http://192.168.1.10:8000/v1")
    return "median, per-request bound, strict checks, proxy_ms, loopback"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<56} ok   {fn() or ''}", flush=True)
        except AssertionError as e:
            fails += 1
            print(f"  {name:<56} FAIL {e}", flush=True)
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
