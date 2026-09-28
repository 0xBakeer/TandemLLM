"""Prove `/v1/dashboard/live` is live, and that its numbers are the response's.

    python tools/live_check.py --base http://127.0.0.1:8011 --token "$QSE_ADMIN_TOKEN" \\
        --out results/live/live-check-<date>.json

Fires streamed chat requests at a server -- two at once, so one queues behind the other -- and
polls `GET /v1/dashboard/live?follow=0` every 0.5 s from a thread while they run. Then it checks:

  1. LIVE: for every request that decoded for more than two seconds, the polled `tokens` grew
     between snapshots while its phase was `decode`, and `decode_tps_now` was set;
  2. EQUAL: every request's last `done` row equals its own response `timings` -- `ttft_ms`,
     `queue_ms`, `prompt_per_second`, `predicted_per_second`, `predicted_n`, `predicted_ms`,
     `tokens_per_block`, `total_ms` -- to the cent (they are the same record);
  3. COUNTS: `counts.in_flight` reached 2 (one decoding, one queued or prefilling) and
     `counts.served` grew by the number of requests sent;
  4. QUEUE: the second request of a pair was seen as `queued` at least once.

Standard library only, so it runs from the Mac against the box. Exit 1 on any failed check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request

PROMPTS = [
    "Explain how speculative decoding verifies a block of drafted tokens, in about 200 words.",
    "Write a short story of about 250 words about a lighthouse keeper who counts ships.",
    "List twelve rivers of Europe with their length in kilometres and one city on each.",
    "Describe the difference between prefill and decode in an inference engine, 150 words.",
]


def _req(base, path, token=None, data=None, timeout=300):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(base + path, data=json.dumps(data).encode() if data else None,
                                  headers=headers)


def snapshot(base, token):
    with urllib.request.urlopen(_req(base, "/v1/dashboard/live?follow=0", token),
                                timeout=10) as r:
        return json.loads(r.read())


def chat(base, model, prompt, max_tokens, think):
    """One streamed chat; returns (request id, timings, usage, wall seconds)."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": True,
            "chat_template_kwargs": {"enable_thinking": think}}
    t0 = time.perf_counter()
    rid, timings, usage = None, None, None
    with urllib.request.urlopen(_req(base, "/v1/chat/completions", data=body),
                                timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            rid = chunk.get("id", rid)
            if "timings" in chunk:
                timings, usage = chunk["timings"], chunk.get("usage")
    return rid, timings, usage, time.perf_counter() - t0


class Poller(threading.Thread):
    def __init__(self, base, token, period):
        super().__init__(daemon=True)
        self.base, self.token, self.period = base, token, period
        self.snaps: list[dict] = []
        self.errors: list[str] = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                self.snaps.append(snapshot(self.base, self.token))
            except Exception as exc:                                   # noqa: BLE001
                self.errors.append(f"{type(exc).__name__}: {exc}")
            self.stop.wait(self.period)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8011")
    ap.add_argument("--token", default=os.environ.get("QSE_ADMIN_TOKEN"))
    ap.add_argument("--model", default="qwen38-spark-engine")
    ap.add_argument("--pairs", type=int, default=2, help="pairs of concurrent requests")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--period", type=float, default=0.5)
    ap.add_argument("--out")
    a = ap.parse_args()
    if not a.token:
        print("a token is needed (--token or QSE_ADMIN_TOKEN)")
        return 2

    before = snapshot(a.base, a.token)
    poller = Poller(a.base, a.token, a.period)
    poller.start()
    results: dict[str, dict] = {}
    lock = threading.Lock()

    def one(prompt):
        rid, timings, usage, wall = chat(a.base, a.model, prompt, a.max_tokens, a.think)
        with lock:
            results[rid] = {"timings": timings, "usage": usage, "wall_s": round(wall, 3)}

    sent = 0
    for i in range(a.pairs):
        ts = [threading.Thread(target=one, args=(PROMPTS[(2 * i + j) % len(PROMPTS)],))
              for j in range(2)]
        for t in ts:
            t.start()
            time.sleep(0.3)                      # the second lands while the first prefills
        for t in ts:
            t.join()
        sent += 2
    time.sleep(2.5)                              # two more snapshots with every row done
    poller.stop.set()
    poller.join(timeout=5)
    after = snapshot(a.base, a.token)

    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {detail}")

    # per request: the polled rows, in order
    rows: dict[str, list[dict]] = {}
    for s in poller.snaps:
        for r in s["requests"]:
            rows.setdefault(r["request_id"], []).append(r)
    missing = [rid for rid in results if rid not in rows]
    check("every request was seen by the poller", not missing, f"missing {missing}")

    for rid, res in results.items():
        seen = rows.get(rid, [])
        dec = [r for r in seen if r["phase"] == "decode"]
        t = res["timings"] or {}
        if (t.get("predicted_ms") or 0) > 2000:
            grew = len(dec) >= 2 and dec[-1]["tokens"] > dec[0]["tokens"]
            now = any(r["decode_tps_now"] is not None for r in dec)
            check(f"{rid[:20]} tokens grew while decoding",
                  grew and now, f"{[r['tokens'] for r in dec]} now={[r['decode_tps_now'] for r in dec]}")
        done = [r for r in seen if r["phase"] == "done"]
        if not done:
            check(f"{rid[:20]} final row present", False, "no done row polled")
            continue
        d = done[-1]
        pairs = [("ttft_ms", "ttft_ms"), ("queue_ms", "queue_ms"),
                 ("prefill_tps", "prompt_per_second"), ("decode_tps", "predicted_per_second"),
                 ("tokens", "predicted_n"), ("decode_ms", "predicted_ms"),
                 ("tokens_per_block", "tokens_per_block"), ("elapsed_ms", "total_ms")]
        diffs = {}
        for lk, tk in pairs:
            lv, tv = d.get(lk), t.get(tk)
            if lk in ("prefill_tps", "decode_tps", "tokens_per_block") and lv is None and tv == 0.0:
                continue                         # the response prints 0.0 where the row says null
            if lv != tv and not (isinstance(lv, (int, float)) and isinstance(tv, (int, float))
                                 and abs(lv - tv) <= 0.011):
                diffs[lk] = (lv, tv)
        check(f"{rid[:20]} final row == timings", not diffs,
              f"{diffs}" if diffs else f"tps {d['decode_tps']} ttft {d['ttft_ms']} tok/blk {d['tokens_per_block']}")

    peak = max((s["counts"]["in_flight"] for s in poller.snaps), default=0)
    check("in_flight reached 2", peak >= 2, f"peak {peak}")
    queued = any(r["phase"] == "queued" for rs in rows.values() for r in rs)
    check("a request was seen queued", queued, "")
    check("served grew by the requests sent",
          after["counts"]["served"] - before["counts"]["served"] == sent,
          f"{before['counts']['served']} -> {after['counts']['served']} ({sent} sent)")
    nowv = [s["now"]["decode_tps"] for s in poller.snaps if s["now"]["decode_tps"]]
    check("now.decode_tps was set while decoding", bool(nowv),
          f"{len(nowv)} of {len(poller.snaps)} snapshots, max {max(nowv) if nowv else None}")
    check("poller saw no errors", not poller.errors, f"{poller.errors[:3]}")

    ok = all(c["ok"] for c in checks)
    out = {"base": a.base, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sent": sent,
           "snapshots": len(poller.snaps), "period_s": a.period, "checks": checks,
           "results": results, "before": before["counts"], "after": after["counts"],
           "polled_rows": rows, "ok": ok}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {a.out}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
