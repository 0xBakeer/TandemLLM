"""Hours of mixed traffic against a running server, watching for the things a bench cannot see.

The atlas row sends fifty identical-shaped requests to a freshly started process with the caches
off, and it answers one question very well: how fast is a generation. It cannot answer any of the
questions that decide whether this is a thing the operator can leave running:

  * does memory climb -- the state cache, the suffix store, the response cache, the allocator;
  * does the cache budget actually evict, or does it grow until the board is full;
  * do the rates drift as the process ages, and does time-to-first-token drift with them;
  * what happens to a client that arrives while a long generation is in flight;
  * what happens to a client that hangs up half way through a stream;
  * does anything raise, and if it does, does the caller find out.

So the mix here is deliberately unlike the row: five languages of workload, thinking on and off,
single-shot and multi-turn sessions, prompts from forty tokens to twenty-four thousand, several
connections at once, and a fraction of requests that disconnect on purpose. Every request's
outcome is recorded, `/health` is sampled on a timer, and both are reported as a table per
interval so a drift shows up as a trend rather than as a final average.

    python tools/soak.py --base-url http://127.0.0.1:8000/v1 --minutes 120 --concurrency 3
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request

FILLER = (
    "The engine holds one sequence at a time, so the server is honest about that and serialises "
    "requests behind a lock. A block of sixteen rows verifies for about a hundred milliseconds, "
    "and the drafter's own forward costs a quarter of that again. "
)

GERMAN = (
    "Schreibe eine kurze, sachliche Zusammenfassung der folgenden Idee: ein Sprachmodell, das "
    "seine eigenen naechsten Tokens vorschlaegt und sie in einem einzigen Durchlauf pruefen "
    "laesst, spart Bandbreite und nicht Rechenzeit. Erklaere, warum das auf dieser Hardware der "
    "entscheidende Unterschied ist."
)

#: (name, prompt, thinking, max_tokens, turns). `turns` > 1 continues the conversation, which is
#: what exercises the session cache: the second turn's prompt is the first turn's prefix plus glue.
WORKLOADS = [
    ("prose", "Write three paragraphs about why a slow river is a bad metaphor for time.",
     False, 320, 1),
    ("chat", "I have two hours and a tired brain. What should I do with them? Be concrete.",
     False, 256, 3),
    ("code", "Write a Python function that merges overlapping intervals, with tests. Explain the "
     "invariant it maintains.", False, 700, 1),
    ("edit", "Rewrite this to be shorter and less pompous, keeping every fact: 'It is incumbent "
     "upon us to acknowledge that the aforementioned methodology, in its current instantiation, "
     "may be characterised as suboptimal with respect to the efficiency criteria.'", False, 220, 1),
    ("german", GERMAN, False, 400, 2),
    ("think", "A train leaves at 09:14 and arrives at 11:02, having stopped twice for four "
     "minutes each. What was its moving time, and what is the trap in this question?", True, 900, 1),
    ("long", None, False, 300, 1),          # prompt built to ~24k tokens at run time
    ("emoji", "Reply with one line: three emoji with skin-tone modifiers, two flags, and the "
     "word 'Grüße'. Nothing else.", False, 80, 1),
]


def post_stream(url: str, body: dict, timeout: float, abandon_after: float = 0.0) -> dict:
    """One streamed chat completion, measured. Returns a record; never raises for a bad answer."""
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    rec = {"t0": time.time(), "ttft": None, "tokens": 0, "chars": 0, "finish": None,
           "status": None, "error": None, "usage": None, "abandoned": False}
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        rec["status"] = resp.status
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                rec["usage"] = d["usage"]
            if d.get("error"):
                rec["error"] = str(d["error"])[:200]
            for c in d.get("choices", []):
                piece = c["delta"].get("content") or c["delta"].get("reasoning_content")
                if piece:
                    if rec["ttft"] is None:
                        rec["ttft"] = time.time() - rec["t0"]
                    rec["chars"] += len(piece)
                if c.get("finish_reason"):
                    rec["finish"] = c["finish_reason"]
            if abandon_after and time.time() - rec["t0"] > abandon_after:
                # A reader that goes away mid-stream. The server should log it as `abandoned`,
                # release the engine, and be ready for the next request -- not leak the lock.
                rec["abandoned"] = True
                resp.close()
                break
    except urllib.error.HTTPError as exc:
        rec["status"] = exc.code
        try:
            rec["error"] = exc.read().decode("utf-8", "replace")[:200]
        except Exception:
            rec["error"] = f"HTTP {exc.code}"
    except Exception as exc:                                       # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["wall"] = time.time() - rec["t0"]
    if rec["usage"]:
        rec["tokens"] = rec["usage"].get("completion_tokens", 0)
    if rec["tokens"] > 1 and rec["ttft"] is not None and rec["wall"] > rec["ttft"]:
        rec["tok_s"] = (rec["tokens"] - 1) / (rec["wall"] - rec["ttft"])
    else:
        rec["tok_s"] = 0.0
    return rec


class Soak:
    def __init__(self, a):
        self.a = a
        self.recs: list[dict] = []
        self.health: list[dict] = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.long_prompt = ("Read this and then answer the question at the end.\n\n"
                            + FILLER * a.long_repeat
                            + "\n\nQuestion: in two sentences, what is the passage claiming?")

    def one(self, rng: random.Random) -> None:
        name, prompt, think, max_tok, turns = rng.choice(WORKLOADS)
        if name == "long":
            prompt = self.long_prompt
        messages = []
        for turn in range(turns):
            messages = messages + [{"role": "user", "content": prompt if turn == 0
                                    else "Now say the same thing in one sentence."}]
            body = {"model": self.a.model, "stream": True, "temperature": 0,
                    "max_tokens": max_tok, "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": bool(think)},
                    "conversation_id": f"soak-{name}-{rng.randrange(10**6)}"}
            body["messages"] = messages
            abandon = self.a.abandon_after if rng.random() < self.a.abandon_rate else 0.0
            rec = post_stream(self.a.base_url, body, self.a.timeout, abandon)
            rec["workload"] = name
            rec["turn"] = turn + 1
            with self.lock:
                self.recs.append(rec)
            if rec["error"] or rec["abandoned"] or not rec["chars"]:
                return
            messages = messages + [{"role": "assistant", "content": "(previous answer)"}]

    def worker(self, seed: int) -> None:
        rng = random.Random(seed)
        while not self.stop.is_set():
            try:
                self.one(rng)
            except Exception as exc:                               # noqa: BLE001
                with self.lock:
                    self.recs.append({"workload": "?", "error": f"driver: {exc}", "tok_s": 0.0,
                                      "wall": 0.0, "tokens": 0, "ttft": None, "finish": None,
                                      "status": None, "abandoned": False, "turn": 0})
            time.sleep(self.a.gap)

    def sampler(self) -> None:
        while not self.stop.is_set():
            try:
                with urllib.request.urlopen(self.a.base_url.rsplit("/v1", 1)[0] + "/health",
                                            timeout=10) as r:
                    h = json.loads(r.read())
                h["t"] = time.time()
                with self.lock:
                    self.health.append(h)
            except Exception:
                pass
            self.stop.wait(self.a.health_every)

    def report(self, since: float, label: str) -> None:
        with self.lock:
            recs = [r for r in self.recs if r.get("t0", 0) >= since]
            health = list(self.health)
        if not recs:
            print(f"[{label}] no requests"); return
        ok = [r for r in recs if not r["error"] and not r["abandoned"] and r["tokens"]]
        rates = sorted(r["tok_s"] for r in ok) or [0.0]
        ttfts = sorted(r["ttft"] for r in ok if r["ttft"] is not None) or [0.0]
        fins: dict = {}
        for r in recs:
            key = ("abandoned" if r["abandoned"] else r["error"] and "error" or r["finish"])
            fins[key] = fins.get(key, 0) + 1
        mem = health[-1].get("memory", {}) if health else {}
        cache = (health[-1].get("cache") or {}) if health else {}
        store = (cache.get("state_store") or {}) if isinstance(cache, dict) else {}
        print(f"[{label}] n={len(recs)} ok={len(ok)} "
              f"tok/s p50={statistics.median(rates):.2f} p10={rates[len(rates) // 10]:.2f} "
              f"ttft p50={statistics.median(ttfts) * 1e3:.0f} ms "
              f"finish={fins} "
              f"alloc={mem.get('allocated_gb')} reserved={mem.get('reserved_gb')} "
              f"rss={mem.get('rss_gb')} store={store}", flush=True)

    def run(self) -> int:
        t0 = time.time()
        threads = [threading.Thread(target=self.worker, args=(i,), daemon=True)
                   for i in range(self.a.concurrency)]
        threads.append(threading.Thread(target=self.sampler, daemon=True))
        for t in threads:
            t.start()
        last = t0
        try:
            while time.time() - t0 < self.a.minutes * 60:
                time.sleep(min(self.a.report_every, 5))
                if time.time() - last >= self.a.report_every:
                    self.report(last, time.strftime("%H:%M:%S"))
                    last = time.time()
        except KeyboardInterrupt:
            pass
        self.stop.set()
        for t in threads:
            t.join(timeout=self.a.timeout + 10)
        print("\n" + "=" * 96)
        self.report(0.0, "TOTAL")
        with self.lock:
            recs, health = list(self.recs), list(self.health)
        bad = [r for r in recs if r["error"]]
        if bad:
            print(f"\n{len(bad)} requests with an error:")
            seen: dict = {}
            for r in bad:
                seen[r["error"][:120]] = seen.get(r["error"][:120], 0) + 1
            for msg, n in sorted(seen.items(), key=lambda kv: -kv[1]):
                print(f"  {n:4d}  {msg}")
        if health:
            first, last_h = health[0].get("memory", {}), health[-1].get("memory", {})
            print(f"\nmemory first -> last: allocated {first.get('allocated_gb')} -> "
                  f"{last_h.get('allocated_gb')} GB, reserved {first.get('reserved_gb')} -> "
                  f"{last_h.get('reserved_gb')} GB, rss {first.get('rss_gb')} -> "
                  f"{last_h.get('rss_gb')} GB")
        if self.a.json_out:
            json.dump({"requests": recs, "health": health}, open(self.a.json_out, "w"))
            print(f"[soak] -> {self.a.json_out}")
        return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen38-spark-engine")
    ap.add_argument("--minutes", type=float, default=120.0)
    ap.add_argument("--concurrency", type=int, default=3,
                    help="connections at once. The engine serves one; the rest queue, which is "
                         "the point -- a queued client must get an answer or an honest refusal")
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--report-every", type=float, default=300.0)
    ap.add_argument("--health-every", type=float, default=30.0)
    ap.add_argument("--abandon-rate", type=float, default=0.08,
                    help="fraction of requests that hang up mid-stream on purpose")
    ap.add_argument("--abandon-after", type=float, default=3.0)
    ap.add_argument("--long-repeat", type=int, default=900,
                    help="copies of the filler paragraph in the 'long' workload; 900 is ~24k tokens")
    ap.add_argument("--json-out", default="")
    return Soak(ap.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
