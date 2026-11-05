"""SPD-249: /health's body timed by part and reused for a TTL (server/health_cache.py)."""

import json
import threading

from server.health_cache import HealthCache, Parts


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _builder(calls):
    def build(part):
        calls.append(1)
        return {"cache": part("cache", lambda: {"n": len(calls)}), "status": "stale", "x": 1}
    return build


def test_without_a_ttl_every_call_builds():
    calls = []
    h = HealthCache(ttl_ms=0)
    for i in range(3):
        d = json.loads(h.answer({"status": "ok", "uptime_s": i}, _builder(calls)))
        assert d["cache"] == {"n": i + 1} and d["uptime_s"] == i and d["status"] == "ok"
    assert len(calls) == 3
    assert h.report()["builds"] == 3 and h.report()["reused"] == 0


def test_a_ttl_reuses_the_body_and_keeps_the_fresh_fields_fresh():
    calls, clk = [], Clock()
    h = HealthCache(ttl_ms=500, clock=clk)
    d1 = json.loads(h.answer({"status": "ok", "uptime_s": 1}, _builder(calls)))
    clk.t += 0.2
    d2 = json.loads(h.answer({"status": "failed", "uptime_s": 2}, _builder(calls)))
    assert len(calls) == 1
    assert d2["cache"] == d1["cache"] == {"n": 1}
    assert d2["status"] == "failed" and d2["uptime_s"] == 2          # fresh, never the body's
    assert list(d2).count("status") == 1
    clk.t += 0.4                                                      # 600 ms after the build
    d3 = json.loads(h.answer({"status": "ok", "uptime_s": 3}, _builder(calls)))
    assert len(calls) == 2 and d3["cache"] == {"n": 2}
    rep = d3["health_cost"]
    assert rep["builds"] == 2 and rep["reused"] == 1 and rep["ttl_ms"] == 500
    assert set(rep["last_build_ms"]) >= {"cache", "encode", "total"}


def test_a_call_never_waits_for_another_builder():
    calls, clk = [], Clock()
    h = HealthCache(ttl_ms=10, clock=clk)
    h.answer({"status": "ok"}, _builder(calls))                       # the first body
    clk.t += 1.0
    entered, release = threading.Event(), threading.Event()

    def slow(part):
        entered.set()
        release.wait(5)
        return {"cache": {"n": "slow"}}

    th = threading.Thread(target=h.answer, args=({"status": "ok"}, slow))
    th.start()
    assert entered.wait(5)
    d = json.loads(h.answer({"status": "ok"}, _builder(calls)))      # answers at once, the last body
    assert d["cache"] == {"n": 1} and len(calls) == 1
    release.set()
    th.join(5)
    assert json.loads(h.answer({"status": "ok"}, _builder(calls)))["cache"] == {"n": "slow"}


def test_an_empty_body_is_valid_json():
    h = HealthCache(ttl_ms=0)
    assert json.loads(h.answer({"status": "ok"}, lambda part: {}))["status"] == "ok"


def test_parts_add_up_and_pass_arguments():
    p = Parts()
    assert p("a", lambda x, y: x + y, 2, 3) == 5
    p("a", lambda: None)
    assert set(p.ms) == {"a"} and p.ms["a"] >= 0


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
