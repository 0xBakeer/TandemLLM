#!/usr/bin/env python3
"""A weight card's speed row (SPD-245): decode, prefill and 1/2/4/8 requests at once, against the
server that runs now, with the proof that N requests decoded together.

    python tools/parallel_card.py --model-id qwen3.8-27b --profile exl3-4.0 \\
        --env-name live8-cold --env-file run/setup.env --commit "$(git rev-parse --short HEAD)" \\
        --out models/qwen3.8-27b/measurements/parallel-2026-10-26-exl3-4.0.json

The server is started beforehand on the profile (ops/use-profile.sh); the tool only sends requests
and reads /health. It writes one raw file: every request's record, the /health samples by level,
the server's limit knobs, and `row`, the entry for the `parallel.rows` list of the model's
model.yaml (the registry holds the two equal, server/setup/registry.py). `--row-out` also writes the
row alone as YAML.

WHAT IT RUNS (bench/card-prompts-v1.json; thinking off, `X-Tandem: raw`, cache_prompt false, a nonce
line at the start of every request, the sampling of `--temperature/--top-k/--top-p`, a seed of its
own for every request):

  decode   the code and the prose workload alone, `--decode-runs` each after `--warmups` warm-ups;
           decode tok/s = (completion tokens - 1) / (last token - first token); median per workload.
           A workload whose text arrives in one event has no decode window, and the row leaves it out
           (LFM2.5's code answer, 2026-10-26: one stream event a run, one run ending as tool_calls).
  prefill  prompts of 2,048, 8,192 and 32,768 tokens (counted with the server's /v1/tokenize and
           checked against the request's prompt_tokens), `--prefill-runs` each, 1 token out;
           TTFT = send to first token; prefill tok/s = prompt tokens / TTFT; medians.
  levels   N = 1, 2, 4, 8 prose requests released together, `--warm-rounds` discarded and
           `--rounds` measured. Per round: wall tok/s = all completion tokens / (last end - send),
           TTFT included; decode tok/s = all completion tokens after the first of each request /
           (last token - first first token); the medians over the rounds.
  proof    /health every 0.2 s during every round: the slots in the decoding set and the requests
           running. A round whose most slots decoding at once is not N, or that saw more than N
           running (a request from elsewhere), is invalid and run again (`--retries`); a level
           without `--rounds` valid rounds makes the row invalid, and the registry refuses it.

Before the levels the server's live slot limit (/health parallel.max_running) must be at least the
largest N; the tool refuses to start otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time
import urllib.request
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOL = "tools/parallel_card.py"
SCHEMA = 1


# --- requests ---------------------------------------------------------------------------------

def _get(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _post_json(url: str, body: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Client:
    def __init__(self, base: str, model: str, sampling: dict, headers: dict):
        self.base, self.model, self.sampling, self.headers = base.rstrip("/"), model, sampling, headers
        self.extra = {"chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}

    def messages(self, system: str, user: str) -> list:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def stream(self, messages: list, max_tokens: int, seed: int, t_send: float | None = None) -> dict:
        """One streamed chat request: its times (monotonic), usage and finish reason."""
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "stream": True,
                "stream_options": {"include_usage": True}, "seed": seed, **self.sampling, **self.extra}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **self.headers})
        t0 = time.monotonic() if t_send is None else t_send
        first = last = None
        usage, finish, events = {}, None, 0
        digest = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if isinstance(ev.get("usage"), dict):
                    usage = ev["usage"]
                for c in ev.get("choices") or []:
                    if c.get("finish_reason") is not None:
                        finish = c["finish_reason"]
                    d = c.get("delta") or {}
                    piece = "".join(v for k in ("content", "reasoning_content", "reasoning")
                                    if isinstance((v := d.get(k)), str))
                    if piece:
                        now = time.monotonic()
                        first = first or now
                        last = now
                        events += 1
                        digest.update(piece.encode())
        end = time.monotonic()
        comp = int(usage.get("completion_tokens") or 0)
        win = (last - first) if first and last else 0.0
        return {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": comp,
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "finish_reason": finish, "events": events, "seed": seed,
                "ttft_s": round(first - t0, 4) if first else None,
                "decode_tok_s": round((comp - 1) / win, 3) if comp > 1 and win > 0 else None,
                "e2e_s": round(end - t0, 4), "t_send": t0, "t_first": first, "t_last": last, "t_end": end,
                "text_sha256": digest.hexdigest()[:16]}


def nonce(*parts) -> str:
    return hashlib.sha256(("card|" + "|".join(map(str, parts)) + "|" + str(time.time_ns())).encode()).hexdigest()[:32]


# --- /health ----------------------------------------------------------------------------------

def health(base: str) -> dict:
    try:
        return _get(base.rstrip("/") + "/health", timeout=5)
    except Exception as exc:                                         # noqa: BLE001
        return {"error": str(exc)}


def health_record(h: dict) -> dict:
    """The fields of /health a card names (no paths)."""
    p = h.get("parallel") or {}
    c = h.get("cache") or {}
    w = h.get("weights") or {}
    return {"version": h.get("version"), "weights": w.get("name"), "hf_repo": w.get("hf_repo"),
            "hf_revision": w.get("hf_revision"), "format": w.get("format"), "max_len": h.get("max_len"),
            "lanes": p.get("n"), "lanes_requested": p.get("requested"), "max_running": p.get("max_running"),
            "max_running_reason": p.get("max_running_reason"),
            "windows": sorted({s.get("max_len") for s in p.get("slots") or [] if s.get("max_len")}),
            "max_queue": h.get("max_queue"),
            "suffix_store": bool(c.get("suffix_store")), "response_cache": bool(c.get("response_cache")),
            "prefix_cache": c.get("prefix_cache"), "prefix_chunk": c.get("prefix_chunk")}


class Poller:
    """/health every 0.2 s: the slots decoding at once and the requests running."""

    def __init__(self, base: str):
        self.base, self.samples, self.stop = base, [], threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop.is_set():
            p = health(self.base).get("parallel") or {}
            d = p.get("decoding")
            self.samples.append({"decoding": d if isinstance(d, int) else len(d or []) if d is not None else None,
                                 "running": p.get("running")})
            self.stop.wait(0.2)

    def __enter__(self):
        self.th.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.th.join()

    def summary(self) -> dict:
        dec = [s["decoding"] for s in self.samples if s["decoding"] is not None]
        run = [s["running"] for s in self.samples if s["running"] is not None]
        hist: dict = {}
        for d in dec:
            hist[str(d)] = hist.get(str(d), 0) + 1
        return {"samples": len(self.samples), "max_decoding_at_once": max(dec) if dec else 0,
                "max_running": max(run) if run else 0, "decoding_hist": hist}


def wait_idle(base: str, limit_s: float = 120.0) -> bool:
    t = time.monotonic()
    while time.monotonic() - t < limit_s:
        p = health(base).get("parallel") or {}
        if not p.get("running") and not p.get("decoding"):
            return True
        time.sleep(1.0)
    return False


# --- the prompts ------------------------------------------------------------------------------

def load_prompts(path: Path) -> dict:
    raw = path.read_bytes()
    pj = json.loads(raw)
    filler = (REPO / pj["filler"]).read_text()
    pj["_sha256"] = hashlib.sha256(raw).hexdigest()
    pj["_filler"] = " ".join(filler.split())
    pj["_filler_sha256"] = hashlib.sha256(filler.encode()).hexdigest()
    return pj


def count_tokens(base: str, messages: list) -> int | None:
    try:
        r = _post_json(base.rstrip("/") + "/v1/tokenize",
                       {"messages": messages, "chat_template_kwargs": {"enable_thinking": False}})
        return int(r["tokens"])
    except Exception:                                                 # noqa: BLE001
        return None


def filler_words(pj: dict, depth: int) -> list[str]:
    words = pj["_filler"].split(" ")
    while len(words) < depth * 2:
        words = words + words
    return words


def prefill_words(base: str, pj: dict, depth: int, head: str, offset: int) -> tuple[int, int | None]:
    """How many filler words after `head` make a request of `depth` prompt tokens (by /v1/tokenize,
    + `offset`), and the count it plans; without /v1/tokenize, 0.75 words a token and None."""
    words = filler_words(pj, depth)

    def n_of(k):
        n = count_tokens(base, [{"role": "user", "content": head + " ".join(words[:k])}])
        return None if n is None else n + offset
    lo, hi = 0, depth * 2
    if n_of(hi) is None:
        return int(depth * 0.75), None
    while lo < hi:                                     # the longest word prefix that fits
        mid = (lo + hi + 1) // 2
        if n_of(mid) <= depth:
            lo = mid
        else:
            hi = mid - 1
    return lo, n_of(lo)


# --- the phases -------------------------------------------------------------------------------

def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def run_decode(cl: Client, pj: dict, a, log) -> dict:
    out = {"warmups": [], "runs": {"code": [], "prose": []}}
    for i in range(a.warmups):
        for w in ("code", "prose"):
            r = cl.stream(cl.messages(pj["system"], pj["nonce_line"].format(nonce=nonce("w", w, i)) + pj["workloads"][w]),
                          256, 9000 + i)
            out["warmups"].append(r)
    for i in range(a.decode_runs):
        for w in ("code", "prose"):
            r = cl.stream(cl.messages(pj["system"], pj["nonce_line"].format(nonce=nonce("d", w, i)) + pj["workloads"][w]),
                          a.max_tokens, 100 + 10 * i + (w == "prose"))
            out["runs"][w].append(r)
            log(f"decode {w} {i + 1}: {r['decode_tok_s']} tok/s, {r['completion_tokens']} tokens, "
                f"TTFT {r['ttft_s']} s, prompt {r['prompt_tokens']}, {r['finish_reason']}")
    out["median"] = {w: med([r["decode_tok_s"] for r in out["runs"][w]]) for w in ("code", "prose")}
    out["prompt_tokens"] = {w: med([r["prompt_tokens"] for r in out["runs"][w]]) for w in ("code", "prose")}
    return out


def run_prefill(cl: Client, pj: dict, a, log) -> dict:
    # the offset between /v1/tokenize and the request's prompt_tokens, from one probe
    probe = [{"role": "user", "content": f"Probe {nonce('probe')}. " + pj["_filler"][:2000]}]
    n_tok = count_tokens(cl.base, probe)
    pr = cl.stream(probe, 1, 7)
    offset = (pr["prompt_tokens"] - n_tok) if (n_tok is not None and pr["prompt_tokens"]) else 0
    out = {"tokenize_offset": offset, "probe": pr, "depths": {}}
    log(f"prefill: /v1/tokenize offset {offset}")
    for depth in [int(x) for x in a.prefill_depths.split(",") if x]:
        runs = []
        k, planned = prefill_words(cl.base, pj, depth, f"Cold prefill {'0' * 32}.\n", offset)
        words = filler_words(pj, depth)
        for i in range(a.prefill_runs):
            text = f"Cold prefill {nonce('p', depth, i)}.\n" + " ".join(words[:k])
            r = cl.stream([{"role": "user", "content": text}], 1, 200 + i)
            r["planned_tokens"] = planned
            r["prefill_tok_s"] = round(r["prompt_tokens"] / r["ttft_s"], 1) if r["ttft_s"] and r["prompt_tokens"] else None
            runs.append(r)
            log(f"prefill {depth} {i + 1}: {r['prompt_tokens']} tokens, TTFT {r['ttft_s']} s, "
                f"{r['prefill_tok_s']} tok/s, cached {r['cached_tokens']}")
        out["depths"][str(depth)] = {"runs": runs, "ttft_s": med([r["ttft_s"] for r in runs]),
                                     "tok_s": med([r["prefill_tok_s"] for r in runs]),
                                     "prompt_tokens": med([r["prompt_tokens"] for r in runs])}
    return out


def one_round(cl: Client, pj: dict, n: int, tag: str, max_tokens: int) -> dict:
    user = [pj["nonce_line"].format(nonce=nonce("c", n, tag, i)) + pj["workloads"]["prose"] for i in range(n)]
    gate = threading.Barrier(n + 1)
    res: dict = {}
    errs: list = []

    def go(i):
        try:
            gate.wait()
            res[i] = cl.stream(cl.messages(pj["system"], user[i]), max_tokens,
                               zlib.crc32(f"{n}|{tag}|{i}".encode()) & 0x7FFFFFFF)
        except BaseException as exc:                                  # noqa: BLE001
            errs.append(repr(exc))

    ths = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    for t in ths:
        t.start()
    with Poller(cl.base) as pol:
        t0 = time.monotonic()
        gate.wait()
        for t in ths:
            t.join()
    streams = [res[i] for i in sorted(res)]
    proof = pol.summary()
    if errs or len(streams) != n:
        return {"tag": tag, "errors": errs, "streams": streams, "proof": proof, "valid": False}
    tokens = sum(s["completion_tokens"] for s in streams)
    end = max(s["t_end"] for s in streams)
    first = min(s["t_first"] for s in streams if s["t_first"])
    last = max(s["t_last"] for s in streams if s["t_last"])
    after_first = sum(max(0, s["completion_tokens"] - 1) for s in streams)
    valid = proof["max_decoding_at_once"] == n and proof["max_running"] <= n and \
        all(s["completion_tokens"] > 0 for s in streams)
    for s in streams:
        for k in ("t_send", "t_first", "t_last", "t_end"):
            s[k] = round(s[k] - t0, 4) if s[k] is not None else None
    return {"tag": tag, "wall_s": round(end - t0, 4), "tokens": tokens,
            "wall_tok_s": round(tokens / (end - t0), 2),
            "decode_tok_s": round(after_first / (last - first), 2) if last > first else None,
            "ttft_s": med([s["ttft_s"] for s in streams]), "proof": proof, "valid": valid, "streams": streams}


def run_levels(cl: Client, pj: dict, a, log) -> dict:
    out = {}
    for n in [int(x) for x in a.levels.split(",") if x]:
        lv = {"warm": [], "rounds": [], "invalid": []}
        for w in range(a.warm_rounds):
            wait_idle(cl.base)
            lv["warm"].append(one_round(cl, pj, n, f"w{w}", a.max_tokens))
        tries = 0
        while len(lv["rounds"]) < a.rounds and tries < a.rounds + a.retries:
            tries += 1
            if not wait_idle(cl.base):
                log(f"C{n}: the server did not go idle in 120 s (traffic from elsewhere)")
            r = one_round(cl, pj, n, f"r{tries}", a.max_tokens)
            (lv["rounds"] if r["valid"] else lv["invalid"]).append(r)
            log(f"C{n} round {tries}: wall {r.get('wall_tok_s')} tok/s, decode {r.get('decode_tok_s')}, "
                f"TTFT {r.get('ttft_s')} s, at once {r['proof']['max_decoding_at_once']} "
                f"running {r['proof']['max_running']} hist {r['proof']['decoding_hist']} valid={r['valid']}")
        rs = lv["rounds"]
        lv["valid"] = len(rs) == a.rounds
        lv["wall_tok_s"] = med([r["wall_tok_s"] for r in rs])
        lv["decode_tok_s"] = med([r["decode_tok_s"] for r in rs])
        lv["ttft_s"] = med([s["ttft_s"] for r in rs for s in r["streams"]])
        lv["at_once"] = min((r["proof"]["max_decoding_at_once"] for r in rs), default=0)
        lv["tokens_per_request"] = sorted({s["completion_tokens"] for r in rs for s in r["streams"]})
        out[str(n)] = lv
    return out


# --- the row ----------------------------------------------------------------------------------

def _r(x, nd):
    return None if x is None else round(float(x), nd)


def build_row(raw: dict) -> dict:
    """The model.yaml entry of a raw file (the registry holds model.yaml's row equal to raw['row'])."""
    s = raw["setup"]
    row = {"profile": raw["profile"], "build": raw["build"], "commit": raw["commit"], "date": raw["date"],
           "raw": raw["raw_path"],
           "setup": {"env": s["env"], "lanes": s["lanes"], "window": s["window"], "max_tokens": s["max_tokens"],
                     "suffix_store": s["suffix_store"], "response_cache": s["response_cache"],
                     "thinking": False, "sampling": s["sampling"],
                     "prompts": {"set": s["prompts"]["set"], "sha256": s["prompts"]["sha256"][:16],
                                 "prompt_tokens": s["prompts"]["prompt_tokens"]}},
           "rounds": raw["args"]["rounds"]}
    if raw.get("decode"):
        # a workload whose stream gave no decode window (its text arrived in one event) is left out
        row["decode"] = {w: _r(raw["decode"]["median"][w], 1) for w in ("code", "prose")
                         if raw["decode"]["median"].get(w) is not None}
    if raw.get("prefill"):
        row["prefill"] = {d: {"ttft_s": _r(v["ttft_s"], 2), "tok_s": _r(v["tok_s"], 0),
                              "prompt_tokens": int(v["prompt_tokens"]) if v["prompt_tokens"] else None}
                          for d, v in raw["prefill"]["depths"].items()}
    if raw.get("levels"):
        row["levels"] = {n: {"wall_tok_s": _r(v["wall_tok_s"], 1), "decode_tok_s": _r(v["decode_tok_s"], 1),
                             "ttft_s": _r(v["ttft_s"], 2), "at_once": int(v["at_once"]),
                             "rounds": len(v["rounds"])}
                         for n, v in raw["levels"].items()}
    if raw.get("note"):
        row["note"] = raw["note"]
    return row


def row_yaml(row: dict) -> str:
    """The row as a YAML list item (flow maps for the inner objects, as the measured table writes them)."""
    def flow(v):
        if isinstance(v, dict):
            return "{" + ", ".join(f"{_k(k)}: {flow(x)}" for k, x in v.items()) + "}"
        if isinstance(v, bool):
            return "true" if v else "false"
        if v is None:
            return "null"
        if isinstance(v, str):
            return json.dumps(v)
        return repr(v)

    def _k(k):
        return json.dumps(k) if str(k).isdigit() else k
    lines = []
    for i, (k, v) in enumerate(row.items()):
        lines.append(("    - " if i == 0 else "      ") + f"{k}: {flow(v)}")
    return "\n".join(lines) + "\n"


def envfile(path: str) -> dict:
    out = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k] = v.strip("'\"")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="TandemLLM")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--env-name", required=True, help="the setup the server runs (a name for the row)")
    ap.add_argument("--env-file", required=True, help="the server's run/setup.env (recorded as is)")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--build", default=None, help="default: the VERSION file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--row-out", default=None)
    ap.add_argument("--prompts", default=str(REPO / "bench" / "card-prompts-v1.json"))
    ap.add_argument("--parts", default="decode,prefill,levels")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warm-rounds", type=int, default=1)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--decode-runs", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--prefill-depths", default="2048,8192,32768")
    ap.add_argument("--prefill-runs", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--note", default=None)
    a = ap.parse_args(argv)

    log = lambda *x: print(f"[card {time.strftime('%H:%M:%S')}]", *x, flush=True)   # noqa: E731
    pj = load_prompts(Path(a.prompts))
    sampling = {"temperature": a.temperature, "top_k": a.top_k, "top_p": a.top_p, "min_p": 0.0}
    cl = Client(a.base, a.model, sampling, {"X-Tandem": "raw"})
    h0 = health(a.base)
    if "error" in h0:
        print(f"no server at {a.base}: {h0['error']}", file=sys.stderr)
        return 2
    hr = health_record(h0)
    parts = set(a.parts.split(","))
    top = max(int(x) for x in a.levels.split(",") if x)
    if "levels" in parts and (hr["max_running"] or 1) < top:
        print(f"refused: the server's live slot limit is {hr['max_running']} ({hr['max_running_reason']}), "
              f"under {top} requests at once", file=sys.stderr)
        return 2
    env = envfile(a.env_file)
    log(f"server {hr['version']} lanes {hr['lanes']} max_running {hr['max_running']} windows {hr['windows']} "
        f"store {hr['suffix_store']} response cache {hr['response_cache']}")
    if not wait_idle(a.base, 300):
        print("refused: the server is not idle", file=sys.stderr)
        return 2
    raw: dict = {"schema": SCHEMA, "tool": TOOL, "model": a.model_id, "profile": a.profile,
                 "build": a.build or (REPO / "VERSION").read_text().strip(), "commit": a.commit,
                 "date": time.strftime("%Y-%m-%d"), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "args": {k: (os.path.basename(v) if k in ("env_file", "out", "row_out", "prompts") and v else v)
                          for k, v in vars(a).items()},
                 "health_start": hr,
                 "env": {k: v for k, v in env.items() if "${HOME}" not in v and not v.startswith("/")}}
    raw["raw_path"] = os.path.relpath(os.path.abspath(a.out), REPO) if os.path.abspath(a.out).startswith(str(REPO)) \
        else f"models/{a.model_id}/measurements/{os.path.basename(a.out)}"
    if "decode" in parts:
        raw["decode"] = run_decode(cl, pj, a, log)
    if "prefill" in parts:
        raw["prefill"] = run_prefill(cl, pj, a, log)
    if "levels" in parts:
        raw["levels"] = run_levels(cl, pj, a, log)
    h1 = health(a.base)
    raw["health_end"] = health_record(h1)
    raw["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    pt = (raw.get("decode") or {}).get("prompt_tokens") or {}
    raw["setup"] = {"env": a.env_name, "lanes": hr["lanes"], "window": max(hr["windows"] or [hr["max_len"] or 0]),
                    "max_tokens": a.max_tokens, "suffix_store": hr["suffix_store"],
                    "response_cache": hr["response_cache"], "sampling": {**sampling, "seed": "one per request"},
                    "prompts": {"set": pj["version"], "sha256": pj["_sha256"], "filler_sha256": pj["_filler_sha256"],
                                "prompt_tokens": {w: int(v) for w, v in pt.items() if v}}}
    raw["valid"] = all(v["valid"] for v in (raw.get("levels") or {}).values())
    if a.note:
        raw["note"] = a.note
    raw["row"] = build_row(raw)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(raw, fh, indent=1)
        fh.write("\n")
    if a.row_out:
        Path(a.row_out).write_text(row_yaml(raw["row"]))
    log("wrote", a.out, "valid" if raw["valid"] else "INVALID")
    return 0 if raw["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
