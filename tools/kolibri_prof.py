"""Decode-step A/B and per-kernel breakdown on the GPU, one load for every variant. A variant sets module switches, the decode graph is captured again, and the graph's
decode rate is timed at each context (median of `--reps` runs of 64 steps after 8 warm-up steps).
The last variant also gets a per-kernel profile of the decode step's ops run eagerly.

    python tools/kolibri_prof.py --variants 'base:NV1ROW=0,MOE2=0,HEAD2=0;new:' --ctx 1024,8192,32768
    (switches name globals of engine.kolibri.kernels; `attn.X=1` sets engine.kolibri.attn.X)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.kolibri import attn as KA  # noqa: E402
from engine.kolibri import kernels as KK  # noqa: E402
from engine.kolibri.model import KolibriEngine  # noqa: E402
from tools import kolibri_attn_kernels as AK  # noqa: E402

MODS = {"kernels": KK, "attn": KA, "ak": AK}


def _shape_like(cur, flat):
    """Ints `flat` in the nesting of `cur` (a tuple, or a tuple of tuples)."""
    if isinstance(cur, tuple) and cur and isinstance(cur[0], tuple):
        out, i = [], 0
        for c in cur:
            out.append(tuple(flat[i:i + len(c)]))
            i += len(c)
        return tuple(out)
    return tuple(flat)


def parse(spec: str):
    """'name:A=1,ak.B=2,NV_1ROW[6144x2560]=16/512/8/3/0;name2:...' -> [(name, [(obj, key, value)])].
    obj is a module (setattr) or a dict (an entry of a launch-shape table)."""
    out = []
    for part in [p for p in spec.split(";") if p.strip()]:
        name, _, sets = part.partition(":")
        kv = []
        for s in [x for x in sets.split(",") if x.strip()]:
            k, _, v = s.partition("=")
            k = k.strip()
            sub = None
            if "[" in k:
                k, sub = k[:-1].split("[")
                sub = tuple(int(x) for x in sub.split("x"))
            mod, _, attr = k.rpartition(".")
            m = MODS[mod or "kernels"]
            cur = getattr(m, attr)
            if sub is not None:
                kv.append((cur, sub, tuple(int(x) for x in v.split("/"))))
            elif isinstance(cur, tuple):
                kv.append((m, attr, _shape_like(cur, [int(x) for x in v.split("/")])))
            else:
                kv.append((m, attr, type(cur)(int(v))))
        out.append((name.strip(), kv))
    return out


def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def _put(obj, key, val):
    if isinstance(obj, dict):
        obj[key] = val
    else:
        setattr(obj, key, val)


def rate(eng, ids, ctx, reps, snap):
    """Decode rate from the prefilled context `snap` (the ring at ctx; full-layer rows below ctx
    are never written by decoding past it)."""
    eng.kv.restore(snap)
    lg = eng.decode(int(ids[ctx]))
    out = []
    for _ in range(reps):
        for _ in range(8):
            lg = eng.decode(int(lg.argmax()))
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(64):
            lg = eng.decode(int(lg.argmax()))
        torch.cuda.synchronize()
        out.append(64 / (time.perf_counter() - t))
    return statistics.median(out), out


@torch.inference_mode()
def host_gap(eng, ids, ctx, snap, n=64):
    """The host's share of a step: the usual loop (argmax read back, then fill and replay) against a
    loop that feeds the argmax to the next step on the device and never waits (the bound a
    pipelined server loop could reach), and the CPU time of the replay call alone."""
    out = {}
    eng.kv.restore(snap)
    lg = eng.decode(int(ids[ctx]))
    for _ in range(8):
        lg = eng.decode(int(lg.argmax()))
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        lg = eng.decode(int(lg.argmax()))
    torch.cuda.synchronize()
    out["sync_loop_tok_s"] = n / (time.perf_counter() - t)
    eng.kv.restore(snap)
    lg = eng.decode(int(ids[ctx]))
    pos = eng.kv.length
    torch.cuda.synchronize()
    rep_s = 0.0
    t = time.perf_counter()
    for i in range(n):
        eng._tok.copy_(eng._logits[0].argmax().reshape(1))
        eng._pos.fill_(pos + i)
        t1 = time.perf_counter()
        eng._graph.replay()
        rep_s += time.perf_counter() - t1
    torch.cuda.synchronize()
    out["device_fed_tok_s"] = n / (time.perf_counter() - t)
    out["replay_call_us"] = rep_s / n * 1e6
    eng.kv.length = pos + n
    return out


def forced_logits(eng, ids, ctx, snap, n=48):
    """Logits of n teacher-forced decode steps from the context `snap` (the captured graph)."""
    eng.kv.restore(snap)
    out = []
    for t in ids[ctx:ctx + n]:
        out.append(eng.decode(int(t)).float().clone())
    return torch.stack(out)


def profile(eng, ids, ctx=1024):
    from torch.profiler import ProfilerActivity, profile as tprof
    eng.reset()
    eng.prefill(ids[:ctx])
    lg = eng.decode(5)
    for _ in range(3):
        lg = eng.decode(int(lg.argmax()))
    torch.cuda.synchronize()

    def step():
        with torch.inference_mode():
            eng._tok.fill_(int(lg.argmax()))
            eng._pos.fill_(eng.kv.length)
            eng.head.logits(eng._hidden(eng._tok, eng._pos, True))
    for _ in range(2):
        step()
    torch.cuda.synchronize()
    with tprof(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(5):
            step()
        torch.cuda.synchronize()
    rows, tot = [], 0.0
    for e in p.key_averages():
        if e.device_time_total > 0 and e.device_type.name == "CUDA":
            rows.append((e.key[:60], e.count // 5, e.device_time_total / 5, e.device_time_total / max(1, e.count)))
            tot += e.device_time_total / 5
    rows.sort(key=lambda r: -r[2])
    print(f"kernels per token {sum(r[1] for r in rows)}, GPU time per token {tot / 1e3:.2f} ms", flush=True)
    for r in rows[:24]:
        print(f"{r[0]:60s} n/tok {r[1]:4d}  {r[2] / 1e3:7.3f} ms/tok  {r[3]:7.1f} us each", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="the NVFP4 set directory")
    ap.add_argument("--fp8", default="", help="the FP8 release, for attention (empty: from the set)")
    ap.add_argument("--variants", default="new:")
    ap.add_argument("--ctx", default="1024,8192,32768")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=1, help="variants interleaved this many times")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--hostgap", action="store_true", help="host share of a step (see host_gap)")
    ap.add_argument("--bitcheck", default="", help="variants whose decode logits must be bit-equal, "
                    "e.g. 'kern2,pdl' (48 teacher-forced steps at each context)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.fp8 or a.set, "tokenizer.json"))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = "".join(open(os.path.join(root, "bench", f)).read() for f in
                   ("heldout_prose.txt", "heldout_code.txt", "heldout_de.txt")
                   if os.path.isfile(os.path.join(root, "bench", f)))
    ids = tok.encode(text, add_special_tokens=False).ids
    ctxs = [int(c) for c in a.ctx.split(",") if c]
    while len(ids) < max(ctxs) + 200:
        ids = ids + ids
    eng = KolibriEngine.load(a.set, a.fp8 or None, max_len=max(ctxs) + 1024, graphs=True)
    variants = parse(a.variants)
    defaults = {}
    for _, kv in variants:
        for m, at, _ in kv:
            defaults[(id(m), at)] = (m, at, _get(m, at))
    res = {}
    for ctx in ctxs:
        for m, at, v in defaults.values():
            _put(m, at, v)
        eng._graph = None
        eng.reset()
        eng.prefill(ids[:ctx])
        snap = eng.kv.snapshot()
        if a.hostgap:
            eng._graph = None
            eng.kv.restore(snap)
            eng.decode(int(ids[ctx]))                     # capture
            print(json.dumps({"hostgap": host_gap(eng, ids, ctx, snap), "ctx": ctx}), flush=True)
        if a.bitcheck:
            lgs = {}
            for name, kv in variants:
                if name not in a.bitcheck.split(","):
                    continue
                for m, at, v in defaults.values():
                    _put(m, at, v)
                for m, at, v in kv:
                    _put(m, at, v)
                eng._graph = None
                lgs[name] = forced_logits(eng, ids, ctx, snap)
            names = list(lgs)
            for n2 in names[1:]:
                d = (lgs[n2] - lgs[names[0]]).abs().max().item()
                print(json.dumps({"bitcheck": [names[0], n2], "ctx": ctx, "equal": d == 0.0,
                                  "max_abs_diff": d}), flush=True)
            del lgs
        for _ in range(a.rounds):
            for name, kv in variants:
                for m, at, v in defaults.values():
                    _put(m, at, v)
                for m, at, v in kv:
                    _put(m, at, v)
                eng._graph = None
                med, runs = rate(eng, ids, ctx, a.reps, snap)
                res.setdefault(name, {}).setdefault(str(ctx), []).extend(runs)
                print(json.dumps({"variant": name, "ctx": ctx, "tok_s": round(med, 2),
                                  "runs": [round(r, 2) for r in runs]}), flush=True)
        del snap
    for m, at, v in defaults.values():
        _put(m, at, v)
    for name in res:
        print(json.dumps({"variant": name, "median_tok_s": {c: round(statistics.median(v), 2)
                                                             for c, v in res[name].items()}}), flush=True)
    if not a.no_profile:
        profile(eng, ids)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
