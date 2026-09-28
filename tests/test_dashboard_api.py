"""The dashboard API, contract v1, on a CPU over seeded temp ledgers.

the Gherkin is the matrix, and every response in it is validated against
docs/contract/dashboard-v1 with tools/contract_check.py -- the same schemas the dashboard's mocks
are held to. Also: DST days, bad parameters, the ledger switched off, the example files, the
routes through the handler, and the 500,000-row budget.

Run: python tests/test_dashboard_api.py
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
import sqlite3
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_app_loop import Req, serve  # noqa: E402  (first: the CPU environment)

from server import app, auth, dashboard_api as D, ledger, usage  # noqa: E402
from tools import contract_check  # noqa: E402

BERLIN = D.zone("Europe/Berlin")
ADMIN = "adm-" + "7" * 40
A, B = "k:3f9a1c0e2b7d", "k:91c2e4aa0b11"


def _ms(y, m, d, hh=12, mm=0, tz=BERLIN) -> int:
    return int(dt.datetime(y, m, d, hh, mm, tzinfo=tz).timestamp() * 1000)


def _ledger(rows: list[dict]):
    led = ledger.Ledger(os.path.join(tempfile.mkdtemp(prefix="qse-dash-"), "l.sqlite3")).open()
    base = usage.RequestRecord("x", "chat", True).row("0.1.0", "c0de")
    for i, r in enumerate(rows):
        row = dict(base, request_id=f"r{i}", model="qwen38-spark-engine", client_id=A,
                   client_kind="open-webui", status=200, finish_reason="stop", prompt_tokens=100,
                   cached_tokens=0, completion_tokens=50, reasoning_tokens=0, queue_ms=0.5,
                   prompt_ms=100.0, ttft_ms=100.5, decode_ms=1000.0, total_ms=1101.0,
                   decode_tps=49.0, prefill_tps=1000.0, blocks=10, draft_tokens=150,
                   draft_accepted=39, cache_source="none")
        row.update(r)
        led.submit(row)
    assert led.flush(5.0)
    return led


def _api(led, now_ms: int, **kw):
    return D.DashboardAPI(led, clock=lambda: now_ms / 1000, cache_ttl=0, **kw)


def _valid(payload, name):
    errs = contract_check.check(json.loads(json.dumps(payload)), name)
    assert not errs, errs[:10]
    return payload


NOW = _ms(2026, 9, 24, 18, 0)


def test_summary_windows():
    days = {0: 1, 5: 2, 20: 3, 200: 4}                   # days ago -> rows
    rows = [{"ts_ms": NOW - d * 86_400_000 - k * 1000} for d, n in days.items() for k in range(n)]
    s = _valid(_api(_ledger(rows), NOW).summary({"tz": "Europe/Berlin"}), "summary")
    w = s["windows"]
    assert [w[k]["requests"] for k in ("today", "7d", "30d", "365d", "all")] == [1, 3, 6, 10, 10], w
    assert w["all"]["active_days"] == 4 and w["7d"]["prompt_tokens"] == 300
    assert s["streak"] == {"current_days": 1, "longest_days": 1}, s["streak"]
    assert s["ledger"]["enabled"] and s["ledger"]["rows"] == 10
    assert s["tz"] == "Europe/Berlin"


def test_streaks():
    rows = [{"ts_ms": NOW - d * 86_400_000} for d in (1, 2, 3, 10, 11, 12, 13, 14)]
    s = _api(_ledger(rows), NOW).summary({})
    # today has nothing yet: the current streak runs through yesterday
    assert s["streak"] == {"current_days": 3, "longest_days": 5}, s["streak"]


def test_dense_day_buckets_for_the_heatmap():
    rows = [{"ts_ms": NOW - d * 86_400_000} for d in (0, 40, 300)]
    u = _valid(_api(_ledger(rows), NOW).usage({"bucket": "day"}), "usage")
    assert len(u["buckets"]) == 365, len(u["buckets"])
    empty = [b for b in u["buckets"] if b["requests"] == 0]
    assert len(empty) == 362
    assert all(b["decode_tps_p50"] is None and b["ttft_ms_p90"] is None for b in empty)
    assert u["to"] == "2026-09-24" and u["from"] == "2025-09-25"
    assert u["buckets"][-1]["start"] == "2026-09-24T00:00:00+02:00"
    assert [t["date"] for t in u["top_days"]] == ["2025-11-28", "2026-08-15", "2026-09-24"]


def test_local_days():
    t = int(dt.datetime(2026, 3, 28, 23, 30, tzinfo=dt.timezone.utc).timestamp() * 1000)
    api = _api(_ledger([{"ts_ms": t}]), NOW)
    u = api.usage({"from": "2026-03-27", "to": "2026-03-30", "tz": "Europe/Berlin"})
    got = {b["start"][:10]: b["requests"] for b in u["buckets"]}
    assert got == {"2026-03-27": 0, "2026-03-28": 0, "2026-03-29": 1, "2026-03-30": 0}, got
    u = api.usage({"from": "2026-03-27", "to": "2026-03-30", "tz": "UTC"})
    assert {b["start"][:10]: b["requests"] for b in u["buckets"]}["2026-03-28"] == 1


def test_dst_days_have_23_and_25_hours():
    api = _api(_ledger([{"ts_ms": _ms(2026, 10, 25, 2, 30)}]), NOW)
    for day, hours in (("2026-03-29", 23), ("2026-10-25", 25), ("2026-09-24", 24)):
        u = _valid(api.usage({"bucket": "hour", "from": day, "to": day}), "usage")
        assert len(u["buckets"]) == hours, (day, len(u["buckets"]))
    u = api.usage({"bucket": "hour", "from": "2026-10-25", "to": "2026-10-25"})
    assert sum(b["requests"] for b in u["buckets"]) == 1
    starts = [b["start"] for b in u["buckets"]]
    assert "2026-10-25T02:00:00+02:00" in starts and "2026-10-25T02:00:00+01:00" in starts
    # and the day buckets on either side of the change are whole local days
    u = api.usage({"from": "2026-10-24", "to": "2026-10-26"})
    assert [b["start"] for b in u["buckets"]] == ["2026-10-24T00:00:00+02:00",
                                                  "2026-10-25T00:00:00+02:00",
                                                  "2026-10-26T00:00:00+01:00"]
    assert [b["requests"] for b in u["buckets"]] == [0, 1, 0]


def test_hour_range_limit_and_bad_parameters():
    api = _api(_ledger([]), NOW)
    for q, needle in (({"bucket": "hour", "from": "2026-08-01", "to": "2026-09-09"}, "31"),
                      ({"bucket": "week"}, "bucket"), ({"tz": "Mars/Olympus"}, "time zone"),
                      ({"from": "yesterday"}, "from"), ({"from": "2026-09-25", "to": "2026-09-01"},
                                                        "after")):
        try:
            api.usage(q)
            raise AssertionError(f"{q} was accepted")
        except D.ApiError as exc:
            assert exc.status == 400 and needle in exc.message, (q, exc.message)
            _valid(exc.body(), "error")
    for q in ({"limit": "0"}, {"limit": "501"}, {"before": "x"}):
        try:
            api.requests(q)
            raise AssertionError(f"{q} was accepted")
        except D.ApiError as exc:
            assert exc.status == 400
    api.usage({"bucket": "hour", "from": "2026-08-10", "to": "2026-09-09"})     # 31 days is fine


def test_filters():
    rows = [{"ts_ms": NOW - k * 60_000, "client_id": A, "prompt_tokens": 10} for k in range(3)]
    rows += [{"ts_ms": NOW - k * 60_000, "client_id": B, "client_kind": "curl",
              "prompt_tokens": 1000} for k in range(2)]
    home = tempfile.mkdtemp()
    labels = os.path.join(home, "clients.json")
    json.dump({A: "open-webui"}, open(labels, "w"))
    api = _api(_ledger(rows), NOW, clients_path=labels)
    u = _valid(api.usage({"client": A}), "usage")
    assert u["totals"]["requests"] == 3 and u["totals"]["prompt_tokens"] == 30, u["totals"]
    assert u["filters"] == {"model": None, "client": A}
    assert u["dimensions"]["clients"] == [
        {"id": A, "label": "open-webui", "kind": "open-webui"},
        {"id": B, "label": None, "kind": "curl"}], u["dimensions"]
    assert u["dimensions"]["models"] == ["qwen38-spark-engine"]
    assert api.usage({"model": "other"})["totals"]["requests"] == 0
    assert api.usage({})["top_days"][0]["top_client"] == B          # 2,000 tokens against 30


def test_percentiles_exclude_replays():
    rows = [{"ts_ms": NOW - k * 1000, "decode_tps": float(10 * (k + 1)), "ttft_ms": float(k + 1)}
            for k in range(10)]
    rows += [{"ts_ms": NOW - (20 + k) * 1000, "cache_source": "response", "decode_tps": 99999.0,
              "ttft_ms": 0.1, "prompt_tokens": 7} for k in range(5)]
    rows += [{"ts_ms": NOW - 40_000, "finish_reason": "refused", "status": 503,
              "prompt_tokens": None, "completion_tokens": None, "decode_tps": None}]
    t = _api(_ledger(rows), NOW).usage({})["totals"]
    assert t["decode_tps_p50"] == 50.0 and t["decode_tps_p90"] == 90.0, t   # nearest rank of 10
    assert t["ttft_ms_p50"] == 5.0
    assert t["requests"] == 16 and t["refused"] == 1 and t["errors"] == 0
    assert t["prompt_tokens"] == 10 * 100 + 5 * 7, "token sums include every row"
    assert t["tokens_per_block_mean"] == round(49 * 10 / 100, 2)


def test_the_same_numbers_without_numpy():
    rows = [{"ts_ms": NOW - k * 1000, "decode_tps": float((k * 37) % 101), "ttft_ms": float(k)}
            for k in range(57)]
    led = _ledger(rows)
    with_np = _api(led, NOW).usage({})
    real = D.np
    D.np = None
    try:
        without = _api(led, NOW).usage({})
    finally:
        D.np = real
    assert with_np["totals"] == without["totals"] and with_np["buckets"] == without["buckets"]
    vals = sorted(float((k * 37) % 101) for k in range(57))
    assert without["totals"]["decode_tps_p90"] == vals[51]            # ceil(0.9 * 57) = 52nd


def test_request_paging():
    api = _api(_ledger([{"ts_ms": NOW - k * 1000} for k in range(120)]), NOW)
    pages, before = [], None
    for _ in range(3):
        p = _valid(api.requests({"limit": "50", **({"before": str(before)} if before else {})}),
                   "requests")
        pages.append(p)
        before = p["next_before"]
    assert [len(p["requests"]) for p in pages] == [50, 50, 20]
    assert pages[-1]["next_before"] is None
    ids = [r["id"] for p in pages for r in p["requests"]]
    assert ids == sorted(ids, reverse=True) and len(set(ids)) == 120
    r = pages[0]["requests"][0]
    assert r["tokens_per_block"] == 4.9 and r["client"]["kind"] == "open-webui"
    assert r["ts"].endswith("Z") and r["stream"] is True
    fin = api.requests({"finish": "length"})
    assert fin["requests"] == [] and fin["next_before"] is None


def test_the_ledger_switched_off():
    api = D.DashboardAPI(None, clock=lambda: NOW / 1000)
    s = _valid(api.summary({}), "summary")
    assert s["ledger"]["enabled"] is False and s["windows"]["all"]["requests"] == 0
    u = _valid(api.usage({}), "usage")
    assert len(u["buckets"]) == 365 and u["dimensions"] == {"models": [], "clients": []}
    _valid(api.requests({}), "requests")


def _get(path, headers=None, peer="127.0.0.1", token=ADMIN):
    app.STATE.setdefault("auth", auth.Auth(ADMIN))
    req = Req(path, {})
    if token:
        req.headers["Authorization"] = f"Bearer {token}"
    req.command, req.client_address = "GET", (peer, 0)
    req.headers.update(headers or {})
    with contextlib.redirect_stdout(io.StringIO()):
        req.do_GET()
    head, _, body = req.wfile.getvalue().partition(b"\r\n\r\n")
    return head.decode(), json.loads(body)


def test_the_system_snapshot_survives_a_failing_source():
    serve()
    app.STATE["gpu_sampler"] = D.GpuSampler(cmd="/nonexistent/nvidia-smi")
    app.STATE["args"] = {"max_len": 262144, "corpus": os.path.expanduser("~/x/corpus")}
    app.STATE.pop("dashboard_api", None)
    head, body = _get("/v1/dashboard/system")
    assert head.startswith("HTTP/1.1 200"), head
    _valid(body, "system")
    assert body["gpu"]["name"] is None and body["gpu"]["temperature_c"] is None
    assert body["flags"]["args"]["corpus"] == "~/x/corpus"
    assert body["ledger"]["enabled"] is False
    assert body["engine"]["status"] == "ok"


def test_secrets_are_redacted():
    serve()
    app.STATE["gpu_sampler"] = D.GpuSampler(cmd="/nonexistent/nvidia-smi")
    app.STATE["args"] = {"max_len": 1, "default_max_tokens": 32768, "api_key": "k-123"}
    os.environ.update(QSE_ADMIN_TOKEN="adm-7e1c-secret-value", QSE_METRICS_TOKEN="met-99ab-value",
                      QWEN38_DEEP="32")
    try:
        head, body = _get("/v1/dashboard/system")
    finally:
        for k in ("QSE_ADMIN_TOKEN", "QSE_METRICS_TOKEN", "QWEN38_DEEP"):
            os.environ.pop(k, None)
    env = body["flags"]["env"]
    assert env["QSE_ADMIN_TOKEN"] == "<redacted>" and env["QSE_METRICS_TOKEN"] == "<redacted>"
    assert env["QWEN38_DEEP"] == "32"
    args = body["flags"]["args"]
    assert args["default_max_tokens"] == 32768 and args["api_key"] == "<redacted>", args
    raw = json.dumps(body)
    assert "adm-7e1c" not in raw and "met-99ab" not in raw


def test_routes_through_the_handler():
    serve()
    app.STATE["ledger"] = _ledger([{"ts_ms": int(time.time() * 1000)}])
    app.STATE.pop("dashboard_api", None)
    try:
        for path, name in (("/v1/dashboard/summary", "summary"),
                           ("/v1/dashboard/usage?bucket=hour", "usage"),
                           ("/v1/dashboard/requests?limit=5", "requests")):
            head, body = _get(path)
            assert head.startswith("HTTP/1.1 200") and "Cache-Control: no-store" in head, head
            _valid(body, name)
        head, body = _get("/v1/dashboard/usage?bucket=week")
        assert head.startswith("HTTP/1.1 400") and body["error"]["type"] == "bad_request"
        head, body = _get("/v1/dashboard/nothing")
        assert head.startswith("HTTP/1.1 404")
        # the admin token, from anywhere; without it, not even from the box itself
        head, _ = _get("/v1/dashboard/summary", {"X-Forwarded-For": "192.168.178.20"})
        assert head.startswith("HTTP/1.1 200"), head
        head, _ = _get("/v1/dashboard/summary", {"X-Forwarded-For": "192.168.178.20"}, token=None)
        assert head.startswith("HTTP/1.1 401"), head
        head, _ = _get("/v1/dashboard/summary", token=None)
        assert head.startswith("HTTP/1.1 401"), head
    finally:
        app.STATE.pop("ledger", None)


def test_the_example_files_validate():
    root = contract_check.ROOT
    names = sorted(f[:-len(".example.json")] for f in os.listdir(root)
                   if f.endswith(".example.json"))
    assert set(names) >= {"summary", "usage", "requests", "system", "logs-json", "logs-line", "gap", "error", "session"}
    for n in names:
        errs = contract_check.check(json.load(open(os.path.join(root, f"{n}.example.json"))), n)
        assert not errs, (n, errs[:5])


def test_the_validator_catches_drift():
    good = json.load(open(os.path.join(contract_check.ROOT, "requests.example.json")))
    bad = json.loads(json.dumps(good))
    bad["requests"][0]["stream"] = "yes"
    del bad["requests"][0]["ttft_ms"]
    bad["requests"][0]["prompt_text"] = "no"
    errs = contract_check.check(bad, "requests")
    assert any("stream" in e for e in errs) and any("ttft_ms" in e for e in errs) \
        and any("prompt_text" in e for e in errs), errs


def test_final_days_are_memoised_and_today_is_not():
    rows = [{"ts_ms": NOW - d * 86_400_000} for d in (0, 3)]
    led = _ledger(rows)
    api = _api(led, NOW)
    assert api.usage({})["totals"]["requests"] == 2
    # a late row for a final day is not seen again (the memo), one for today is
    led.submit(dict(usage.RequestRecord("x", "chat", True).row(), ts_ms=NOW - 3 * 86_400_000,
                    model="qwen38-spark-engine", client_id=A, client_kind="other", status=200))
    led.submit(dict(usage.RequestRecord("y", "chat", True).row(), ts_ms=NOW - 60_000,
                    model="qwen38-spark-engine", client_id=A, client_kind="other", status=200))
    led.flush(5.0)
    u = api.usage({})
    assert u["totals"]["requests"] == 3, u["totals"]
    assert u["buckets"][-1]["requests"] == 2 and u["buckets"][-4]["requests"] == 1
    # a prune invalidates the memo
    led.stats["pruned"] += 1
    assert api.usage({})["totals"]["requests"] == 4
    # yesterday is final only two hours after it ended
    early = _ms(2026, 9, 24, 1, 30)
    api2 = _api(led, early)
    api2.usage({})
    assert _dt_date(2026, 9, 23) not in api2._memo[("Europe/Berlin", None, None)]
    assert _dt_date(2026, 9, 22) in api2._memo[("Europe/Berlin", None, None)]


def _dt_date(y, m, d):
    return dt.date(y, m, d)


def test_fast_enough():
    """500,000 synthetic rows over a year: the 365-day usage query under 1 s (a laptop CPU here;
    the board's CPU is in the same range)."""
    from tools import ledger_synth
    path = os.path.join(tempfile.mkdtemp(prefix="qse-synth-"), "l.sqlite3")
    n = ledger_synth.write(path, 500_000, 365, seed=7, end=NOW / 1000)
    assert 490_000 <= n <= 510_000, n
    led = ledger.Ledger(path).open()
    api = _api(led, NOW)
    t0 = time.perf_counter()
    u = api.usage({"bucket": "day"})
    dt_usage = time.perf_counter() - t0
    t0 = time.perf_counter()
    s = api.summary({})
    dt_summary = time.perf_counter() - t0
    _valid(u, "usage")
    _valid(s, "summary")
    lo = _ms(2025, 9, 25, 0, 0)
    inside = sqlite3.connect(path).execute(
        "SELECT COUNT(*) FROM requests WHERE ts_ms >= ?", (lo,)).fetchone()[0]
    assert u["totals"]["requests"] == inside and inside > 0.99 * n, (u["totals"]["requests"], n)
    assert sum(b["requests"] for b in u["buckets"]) == inside
    t0 = time.perf_counter()
    api.usage({"bucket": "day"})
    api._cache.clear()
    dt_warm = time.perf_counter() - t0
    print(f"    500k rows: usage(365 d) cold {dt_usage * 1000:.0f} ms, summary after it "
          f"{dt_summary * 1000:.0f} ms, usage again {dt_warm * 1000:.0f} ms")
    assert dt_usage < 1.0, dt_usage
    led.close()


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
