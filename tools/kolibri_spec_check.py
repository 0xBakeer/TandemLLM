"""Kolibri speculation on the GPU: verify parity, StairCut's prices, the lossless gate, tok/s.

    python tools/kolibri_spec_check.py [parity] [prices] [gate] --set DIR \
        [--tok DIR] [--max-len 40000] [--ctx 1024,8192,32768] [--max-new 384] \
        [--corpus ~/.kolibri-engine/corpus] [--stair-out ops/kolibri-stair.json] [--out res.json]

One engine load for every step named (run it with the server stopped: one engine on the GPU).

parity: each verify twin against the decode op it mirrors, row by row, bit for bit (FP8 one-row
  projections, NVFP4 at up to 16 rows, the routed experts with the shared one, the head), then
  `Verifier.selfcheck` (a 12-row chain, a tree and a commit against the decode step), then random
  trees of 2 to 32 nodes at three contexts against `ReplayVerifier` (each node decoded alone).
prices: the decode step and a verify of R rows (R in engine/kolibri/spec.SIZES) at each context,
  rows taken from real text (a chain is consecutive tokens; a tree is two branches), median of
  repeats with the host work the loop does; the commit and a lookup's cost. Written as the
  StairCut table.
gate: six prompts through the chat template (prose, code, an edit of a file, a verbatim quote,
  German, a chat with tools), greedy, decoded with speculation off and on through the loop the
  server uses (`SpecRows`); the token ids must be equal. tok/s of each, n = 1 per class.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.kolibri import kernels as KK  # noqa: E402
from engine.kolibri import spec as S  # noqa: E402
from engine.kolibri.model import KolibriEngine  # noqa: E402
from engine.kolibri.verify import ReplayVerifier, Verifier  # noqa: E402


def sync():
    torch.cuda.synchronize()


def text_ids(tok, n: int) -> list[int]:
    """n tokens of real text: the held-out files, repeated as needed."""
    parts = []
    for f in ("heldout_prose.txt", "heldout_code.txt", "heldout_de.txt", "calib.txt"):
        p = os.path.join(ROOT, "bench", f)
        if os.path.isfile(p):
            parts.append(open(p, encoding="utf-8").read())
    ids = tok("\n\n".join(parts), add_special_tokens=False).input_ids
    out = []
    while len(out) < n:
        out.extend(ids)
    return out[:n]


# ------------------------------------------------------------------------------ parity
@torch.inference_mode()
def parity(eng, v: Verifier, tok, ctxs) -> dict:
    res = {"ops": {}, "selfcheck": None, "random_trees": {}}
    torch.manual_seed(0)
    dev = eng.device
    ops = res["ops"]

    def rows_equal(name, got, ref_fn, x):
        bad = [i for i in range(x.shape[0]) if not torch.equal(got[i:i + 1], ref_fn(x[i:i + 1]))]
        ops[name] = "equal" if not bad else f"rows {bad[:6]} differ"
        return not bad

    H = eng.cfg.hidden
    for lw in eng.layers[:5]:                       # 4 sliding layers and a full one
        tag = "sliding" if lw.sliding else "full"
        for M in (2, 7, 16, 32):
            x = (torch.randn(M, H, device=dev) * 0.5).to(torch.bfloat16)
            rows_equal(f"qkv/{tag}/M{M}", v._lin(lw.qkv, x), lambda r: lw.qkv.matmul(r), x)
            xo = (torch.randn(M, eng.cfg.q_size, device=dev) * 0.5).to(torch.bfloat16)
            rows_equal(f"o/{tag}/M{M}", v._lin(lw.o, xo), lambda r: lw.o.matmul(r), xo)
            # a decode row's MoE with its combine done (the fused combine hands back parts)
            fc = getattr(KK, "FUSE_COMBINE", None)
            if fc is not None:
                KK.FUSE_COMBINE = False
            try:
                rows_equal(f"moe/{tag}/M{M}", v._moe(x, lw), lambda r: eng._moe(r, lw), x)
            finally:
                if fc is not None:
                    KK.FUSE_COMBINE = fc
    for M in (2, 9, 32):
        x = torch.randn(M, H, device=dev)
        rows_equal(f"head/M{M}", v._head(x), lambda r: eng.head.logits(r), x)
    # the drafter's taps (the residual after layers 1, 13, 25, 37, 49): each verify row's equal
    # to the decode step's for the same token
    tl = [L for L in (1, 13, 25, 37, 49) if L < eng.cfg.layers]
    eng.set_taps(tl)
    v.set_taps()
    ids = text_ids(tok, 300)
    eng.reset()
    eng.prefill(ids[:256])
    eng.truncate(255)
    dec = []
    for t in ids[255:263]:
        eng.decode(t)
        dec.append(eng.dec_taps.clone())
    eng.truncate(255)
    v.verify(ids[255:263], [-1] + list(range(7)))
    bad = [i for i in range(8) if not torch.equal(v.taps[:, i], dec[i])]
    ops["taps"] = "equal" if not bad else f"rows {bad} differ"
    eng.set_taps([])
    v.set_taps()
    eng.reset()
    ok, why = v.selfcheck(text_ids(tok, 720), log=print)
    res["selfcheck"] = "pass" if ok else why
    rnd = random.Random(1)
    long = text_ids(tok, max(ctxs) + 64)
    rep = ReplayVerifier(eng)
    for c in ctxs:
        eng.reset()
        eng.prefill(long[:c - 1])
        bad = []
        for trial in range(6):
            n = rnd.choice([2, 3, 5, 8, 13, 16, 24, 32])
            parents = [-1]
            for i in range(1, n):
                # DFS pre-order: a node's parent is its predecessor or one of its ancestors
                cand, j = [], i - 1
                while j >= 0:
                    cand.append(j)
                    j = parents[j]
                parents.append(rnd.choice(cand[:3]))
            toks = [long[c - 1]] + [rnd.randrange(eng.cfg.vocab) if rnd.random() < 0.4
                                    else long[c + i] for i in range(n - 1)]
            a = v.verify(toks, parents).clone()
            b = rep.verify(toks, parents)
            diff = [i for i in range(n) if not torch.equal(a[i], b[i])]
            if diff:
                bad.append({"n": n, "rows": diff[:5],
                            "max_abs": float((a - b).abs().max())})
        res["random_trees"][c] = "equal" if not bad else bad
    eng.reset()
    res["pass"] = (all(s == "equal" for s in ops.values()) and res["selfcheck"] == "pass"
                   and all(s == "equal" for s in res["random_trees"].values()))
    return res


# ------------------------------------------------------------------------------ prices
@torch.inference_mode()
def prices(eng, v: Verifier, tok, ctxs, reps: int = 7) -> dict:
    long = text_ids(tok, max(ctxs) + 4096)
    out = {"decode": {}, "chain": {}, "tree": {}}
    for c in ctxs:
        eng.reset()
        eng.prefill(long[:c - 1])
        p = eng.kv.length
        anchor = long[c - 1]
        ts = []
        for _ in range(reps + 2):
            eng.truncate(p)
            sync()
            t0 = time.perf_counter()
            eng.decode(anchor)
            sync()
            ts.append((time.perf_counter() - t0) * 1e3)
        eng.truncate(p)
        out["decode"][str(c)] = round(statistics.median(ts[2:]), 3)
        ch, tr = {}, {}
        for R in S.SIZES:
            chain = long[c - 1:c - 1 + R]
            v.verify(chain, [-1] + list(range(R - 1)))       # capture outside the timing
            ts = []
            for _ in range(reps):
                sync()
                t0 = time.perf_counter()
                lg = v.verify(chain, [-1] + list(range(R - 1)))
                float(lg[0, 0])
                ts.append((time.perf_counter() - t0) * 1e3)
            ch[str(R)] = round(statistics.median(ts), 3)
            # a tree: the chain's first half, a second branch from node 0 out of other text
            h = (R + 1) // 2
            toks = long[c - 1:c - 1 + h] + long[c + 2000:c + 2000 + (R - h)]
            parents = [-1] + list(range(h - 1)) + [0] + list(range(h, R - 1))
            ts = []
            for _ in range(reps):
                sync()
                t0 = time.perf_counter()
                lg = v.verify(toks, parents)
                float(lg[0, 0])
                ts.append((time.perf_counter() - t0) * 1e3)
            tr[str(R)] = round(statistics.median(ts), 3)
        out["chain"][str(c)] = ch
        out["tree"][str(c)] = tr
        print(f"[prices] ctx {c}: decode {out['decode'][str(c)]} ms; chain {ch}; tree {tr}",
              flush=True)
    # the commit of an 8-node path and the loop's host work around a verify
    eng.reset()
    eng.prefill(long[:1023])
    ts = []
    for _ in range(reps):
        v.verify(long[1023:1031], [-1] + list(range(7)))
        sync()
        t0 = time.perf_counter()
        v.commit(list(range(8)))
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)
        eng.truncate(1023)
    out["commit_ms"] = round(statistics.median(ts), 3)
    eng.reset()
    out["overhead_ms"] = round(out["commit_ms"] + 0.15, 3)
    out["measured"] = (f"{time.strftime('%Y-%m-%d %H:%M')} tools/kolibri_spec_check.py prices, "
                       f"median of {reps}, host work included")
    return out


# ------------------------------------------------------------------------------ A/B and profile
@torch.inference_mode()
def ab(eng, tok, ctx: int = 1024, sizes=(2, 4, 8, 16), reps: int = 7) -> dict:
    """Verify ms per row count with the rows on a grid axis (PAR 1) and in a loop (PAR 0)."""
    from engine.kolibri import verify as VM
    long = text_ids(tok, ctx + 64)
    eng.reset()
    eng.prefill(long[:ctx - 1])
    out = {}
    for par in (1, 0):
        VM.PAR = bool(par)
        v = Verifier(eng)
        row = {}
        for R in sizes:
            chain = long[ctx - 1:ctx - 1 + R]
            v.verify(chain, [-1] + list(range(R - 1)))
            ts = []
            for _ in range(reps):
                sync()
                t0 = time.perf_counter()
                float(v.verify(chain, [-1] + list(range(R - 1)))[0, 0])
                ts.append((time.perf_counter() - t0) * 1e3)
            row[str(R)] = round(statistics.median(ts), 3)
        out[f"par{par}"] = row
        del v
        torch.cuda.empty_cache()
        print(f"[ab] PAR={par}: {row}", flush=True)
    VM.PAR = True
    eng.reset()
    return out


@torch.inference_mode()
def profile(eng, v: Verifier, tok, sizes=(1, 2, 8)) -> dict:
    """GPU time per kernel name for the decode step (1) and an eager verify of R rows."""
    from torch.profiler import ProfilerActivity, profile as tprof
    from engine.kolibri.verify import _Bufs, tree_tables
    long = text_ids(tok, 1100)
    eng.reset()
    eng.prefill(long[:1023])
    out = {}
    for R in sizes:
        with tprof(activities=[ProfilerActivity.CUDA]) as pr:
            for _ in range(3):
                if R == 1:
                    h = eng._hidden(torch.tensor([long[1023]], device=eng.device), 1023, True)
                    eng.head.logits(h)
                else:
                    b = _Bufs(R, eng.device)
                    d, a = tree_tables([-1] + list(range(R - 1)))
                    b.load(1023, long[1023:1023 + R], d, a)
                    v._forward(b)
            sync()
        agg: dict = {}
        for e in pr.key_averages():
            t = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
            if t:
                agg[e.key[:60]] = agg.get(e.key[:60], 0) + t / 3 / 1000.0
        top = sorted(agg.items(), key=lambda kv: -kv[1])[:14]
        out[str(R)] = {"total_ms": round(sum(agg.values()), 2),
                       "top": {k: round(t, 3) for k, t in top}}
        print(f"[profile] R={R}: {out[str(R)]}", flush=True)
        eng.truncate(1023)
    eng.reset()
    return out


# ------------------------------------------------------------------------------ the gate
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search_web", "description": "Search the web",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}}]


def gate_messages() -> dict:
    src = open(os.path.join(ROOT, "engine", "prices.py"), encoding="utf-8").read()
    prose = open(os.path.join(ROOT, "bench", "heldout_prose.txt"), encoding="utf-8").read()[:2400]
    off = {"enable_thinking": False}
    P = {
        "prose": ([{"role": "user", "content": "Write a short story (about 300 words) about a "
                    "lighthouse keeper who finds a message in a bottle."}], None, off),
        "code": ([{"role": "user", "content": "Write a Python function that parses an ISO-8601 "
                   "duration such as P3DT4H5M6S into seconds, with docstring and three pytest "
                   "tests."}], None, off),
        "edit": ([{"role": "user", "content": "Rename the function `load` to `load_table` and "
                   "the module constant `KEYS` to `TABLE_KEYS` everywhere in this file, change "
                   "nothing else, and return the complete file in one code block.\n\n```python\n"
                   + src + "\n```"}], None, off),
        "copy": ([{"role": "user", "content": "Repeat the following text exactly, word for word, "
                   "then quote verbatim every sentence in it that contains a number.\n\n"
                   + prose}], None, off),
        "german": ([{"role": "user", "content": "Erkläre in acht Sätzen, wie ein Kühlschrank "
                     "funktioniert, und nenne zwei typische Fehler bei der Benutzung."}], None, off),
        "chat_tools": ([{"role": "system", "content": "You are a helpful assistant. Use the tools "
                         "when they help."},
                        {"role": "user", "content": "Wie ist das Wetter gerade in Berlin und in "
                         "Hamburg? Und such mir bitte die Öffnungszeiten vom Pergamonmuseum."}],
                       TOOLS, {"reasoning_effort": "low"}),
    }
    # the edit again behind 7k tokens of other text: the full layers' rows and the ring at 8k
    long_doc = "\n\n".join(open(os.path.join(ROOT, "bench", f), encoding="utf-8").read()
                            for f in ("heldout_prose.txt", "heldout_de.txt", "heldout_code.txt")
                            if os.path.isfile(os.path.join(ROOT, "bench", f)))
    P["edit_8k"] = ([{"role": "user", "content": "Background material, not part of the task:\n\n"
                      + long_doc[:26000] + "\n\nThe task: " + P["edit"][0][0]["content"]}],
                    None, off)
    return P


def gate_prompts(tok) -> dict:
    out = {}
    for k, (msgs, tools, kw) in gate_messages().items():
        ids = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=True,
                                      **kw)
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        out[k] = [int(t) for t in ids]
    return out


@torch.inference_mode()
def generate(eng, rows: S.SpecRows, prompt: list[int], max_new: int, stops: set) -> tuple:
    eng.reset()
    row = eng.prefill(prompt)
    ctx = list(prompt)
    rows.start(ctx)
    out = []
    sync()
    t0 = time.perf_counter()
    for i in range(max_new):
        t = int(row.argmax())
        out.append(t)
        ctx.append(t)
        if t in stops or i + 1 == max_new:
            break
        row = rows.next(t, ctx)
    rows.settle()
    sync()
    dt = time.perf_counter() - t0
    return out, dt


def gate(eng, v: Verifier, spec: S.KolibriSpec, tok, max_new: int, stops: set) -> dict:
    res = {"classes": {}}
    allok = True
    for name, prompt in gate_prompts(tok).items():
        off, t_off = generate(eng, S.SpecRows(eng, None, None), prompt, max_new, stops)
        # the calibration carries over from class to class (as in the server); the counts are
        # this class's own
        cls_spec = S.KolibriSpec(spec.sources, spec.cut, max_rows=spec.max_rows)
        rows = S.SpecRows(eng, v, cls_spec)
        on, t_on = generate(eng, rows, prompt, max_new, stops)
        same = on == off
        allok &= same
        first = next((i for i, (a, b) in enumerate(zip(on, off)) if a != b), None)
        r = {"prompt_tokens": len(prompt), "tokens": len(off), "equal": same,
             "first_diff": first if not same else None,
             "tok_s_off": round(len(off) / t_off, 1), "tok_s_on": round(len(on) / t_on, 1),
             "speedup": round((len(on) / t_on) / (len(off) / t_off), 3),
             "spec": rows.report()}
        res["classes"][name] = r
        print(f"[gate] {name}: {len(prompt)} prompt, {len(off)} tokens, equal={same}, "
              f"{r['tok_s_off']} -> {r['tok_s_on']} tok/s ({r['speedup']}x), "
              f"rounds {r['spec'].get('verify_rounds')}, accepted/round "
              f"{r['spec'].get('accepted_per_round')}", flush=True)
        res["classes"][name]["text_head"] = tok.decode(off[:40])
        res["classes"][name]["off_ids"] = off
    res["pass"] = allok
    sp = [c["speedup"] for c in res["classes"].values()]
    res["mean_speedup"] = round(sum(sp) / len(sp), 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["parity", "prices", "gate"])
    ap.add_argument("--set", required=True, help="the NVFP4 set directory")
    ap.add_argument("--fp8", default="", help="the FP8 release, for attention (empty: from the set)")
    ap.add_argument("--tok", default="", help="tokenizer directory (default: --fp8, then --set)")
    ap.add_argument("--max-len", type=int, default=40000)
    ap.add_argument("--ctx", default="1024,8192,32768")
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--stop-at-eos", action="store_true",
                    help="end a gate answer at its end token (default: decode --max-new tokens "
                         "regardless, so every class compares the same length past its answer)")
    ap.add_argument("--corpus", default="~/.kolibri-engine/corpus")
    ap.add_argument("--stair", default="", help="price table for the gate (default: the one "
                                                 "measured in this run, else ops/kolibri-stair.json)")
    ap.add_argument("--stair-out", default=os.path.join(ROOT, "ops", "kolibri-stair.json"))
    ap.add_argument("--out", default="")
    ap.add_argument("--profile-rows", default="1,2,8", help="rows of the profiled verify")
    a = ap.parse_args()
    from engine.kolibri import chat
    from engine.tokfp import fingerprint
    tdir = os.path.expanduser(a.tok or a.fp8 or a.set)
    tok = chat.load_tokenizer(tdir)
    stops = set(chat.stop_ids(tdir))
    ctxs = [int(x) for x in a.ctx.split(",") if x]
    t0 = time.time()
    eng = KolibriEngine.load(os.path.expanduser(a.set), os.path.expanduser(a.fp8) or None,
                             device="cuda", max_len=a.max_len, graphs=True)
    eng.prefill(tok("warm up", add_special_tokens=False).input_ids)
    eng.decode(1)
    eng.reset()
    print(f"[spec-check] loaded in {time.time() - t0:.0f}s", flush=True)
    v = Verifier(eng)
    res = {"supported": v.ok, "why": v.why, "moe_br": v.moe_br}
    if not v.ok:
        print(json.dumps(res))
        return
    if "parity" in a.steps:
        res["parity"] = parity(eng, v, tok, ctxs)
        print(json.dumps(res["parity"], indent=1), flush=True)
    if "ab" in a.steps:
        res["ab"] = ab(eng, tok)
    if "profile" in a.steps:
        res["profile"] = profile(eng, v, tok, tuple(int(r) for r in a.profile_rows.split(",")))
    table = None
    if "prices" in a.steps:
        res["prices"] = prices(eng, v, tok, ctxs)
        table = S.PriceTable(res["prices"])
        if a.stair_out:
            with open(a.stair_out, "w") as f:
                json.dump(res["prices"], f, indent=1)
    if "gate" in a.steps:
        if table is None:
            path = a.stair or os.path.join(ROOT, "ops", "kolibri-stair.json")
            table = S.PriceTable.load(path) if os.path.isfile(path) else S.default_prices()
        corpus = os.path.expanduser(a.corpus)
        look = S.LookupSource(corpus=corpus if os.path.isdir(corpus) else "",
                              tokenizer_sha=fingerprint(tdir))
        spec = S.KolibriSpec([look], S.StairCut(table, max_rows=v.max_rows), max_rows=v.max_rows)
        res["gate"] = gate(eng, v, spec, tok, a.max_new, stops if a.stop_at_eos else set())
        res["gate"]["ignore_eos"] = not a.stop_at_eos
        res["gate"]["corpus"] = corpus if look.d.corpus is not None else None
    res["graphs"] = v.stats
    s = json.dumps(res, indent=1)
    print(s)
    if a.out:
        with open(a.out, "w") as f:
            f.write(s)


if __name__ == "__main__":
    main()
