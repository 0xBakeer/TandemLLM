"""SPD-249 (T1) on the board: the packed MTP draft's steps after the feed from the lane count's CUDA graph
(QWEN38_FN_PACK_GRAPH) against the eager packed pass, over a trace of real texts.

    PYTHONPATH=<repo> python tools/fn_pack_graph_check.py --pack ~/flashnext-packs/3.05 \\
        --texts ~/spd249-1029/ab2-1e5/finAS-c8.json --rounds 200 --heads 32768,0 --json out.json

The texts: the bench's own requests (bare_bench JSON: the first measured round's prompts and served texts, as
tools/fn_mtp_accept.py's G2 reads them). Each lane prefills its prompt and text (the trunk's streams of every
position) and gets two MTP heads primed alike on its prompt: twin A drafts packed eager, twin B packed from the graph.
Each round a set of lanes (all of them in lane order, most rounds; else a random subset of 2..all lanes in a random
order) feeds both twins of each lane the same next 1..depth + 1 rows of its text with their next tokens; then A's heads
run packed eager (QWEN38_FN_PACK_GRAPH off), B's packed with the graph (on, captures allowed). Compared every round,
lane and depth: the top-2 token ids and both probabilities, bit for bit (the float32 bits), and the lanes' MTP caches
(K, V, indexer keys, pooled keys) past the round. Reported: the comparisons, the graph's captures / replays / entry
refills / eager drafts, and ms a round of each (offline, after a warm round; not a served measurement).
--keyed (coupled drafts, QWEN38_FN_MTP_COUPLED on): each lane drafts as a seeded sampled request (its seed, the
corrector's sampling where a corrector file is loaded), both twins staged with the same keys and parameters each round;
every fifth round a random half of the lanes drafts argmax (a pass with keyed and argmax lanes).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.methods import draft_meet  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bits(rows: list) -> list:
    return [[struct.unpack("<I", struct.pack("<f", p))[0] for p in r] for r in rows]


def _packed(heads: list, feeds: list) -> list:
    got = []
    draft_meet.MEET = lambda head: (got.append(head) or True)
    try:
        for h, (rw, nx) in zip(heads, feeds):
            h.launch(rw, nx)
    finally:
        draft_meet.MEET = None
    type(got[0]).run_packed(got)
    return [h.read() for h in heads]


def _caches_equal(a, b, upto: int) -> bool:
    from engine.models.flashnext.ref import MTP
    sa, sb = a.st, b.st
    ok = torch.equal(sa.k[MTP][:, :upto], sb.k[MTP][:, :upto]) and torch.equal(sa.v[MTP][:, :upto], sb.v[MTP][:, :upto])
    if MTP in sa.ik:
        ok = ok and torch.equal(sa.ik[MTP][:upto], sb.ik[MTP][:upto])
    if MTP in sa.pool:
        ok = ok and torch.equal(sa.pool[MTP], sb.pool[MTP])
    return bool(ok)


def _samplers(n: int) -> list:
    """--keyed: a seeded sampled request a lane, sampled as the corrector's file was trained (else T 1, top-k 20,
    top-p 0.95)."""
    import types
    from engine.models.flashnext import corrector as CR
    c = CR.get() if CR.ON else None
    sm = dict(getattr(c, "sampling", None) or {})
    return [types.SimpleNamespace(coupled=True, seed=1000 + i, temperature=float(sm.get("temperature") or 1.0),
                                  top_k=int(sm.get("top_k") or 20), top_p=float(sm.get("top_p") or 0.95), min_p=0.0)
            for i in range(n)]


def run_head(fn, lanes: list, a, hot: int, log) -> dict:
    """Both twin sets of every lane over the trace with draft head `hot` (0: the full head)."""
    from engine.models.flashnext import engine as E
    from engine.models.flashnext.mtp import FlashNextMTPHead
    if hot:
        fn.set_draft_head(np.load(os.path.join(ROOT, "ops", "fn-hot-vocab.npy"))[:hot])
    else:
        fn.set_draft_head(None)
    rng = random.Random(a.seed)
    cap = max(len(x["ids"]) for x in lanes) + 64
    twins = []
    for x in lanes:
        P = len(x["prompt"])
        k = P - 1                                       # prime the prompt's rows; the text's rows ride the rounds
        pair = (FlashNextMTPHead(fn, a.depth, cap), FlashNextMTPHead(fn, a.depth, cap))
        for h in pair:
            h.prime(x["rows"][:k], x["ids"][1:k + 1])
        twins.append({"A": pair[0], "B": pair[1], "k": k})
    torch.cuda.synchronize()
    res = {"hot": hot, "rounds": 0, "lane_rounds": 0, "picks": 0, "tok_diff": 0, "prob_diff": 0, "cache_diff": 0,
           "by_lanes": {}, "ms_eager": {}, "ms_graph": {}, "first_diff": None, "keyed_lane_rounds": 0,
           "mixed_rounds": 0}
    sms = _samplers(len(lanes)) if a.keyed else None
    stats0 = fn.pack_graph_stats()
    for r in range(a.rounds):
        live = [i for i, t in enumerate(twins) if t["k"] + a.depth + 2 < len(lanes[i]["ids"])]
        if len(live) < 2:
            break
        if r % 4 == 3:                                  # a random subset in a random order
            n = rng.randint(2, len(live))
            sel = rng.sample(live, n)
        else:
            sel = live
        feeds = []
        for i in sel:
            t, x = twins[i], lanes[i]
            n = min(1 + rng.randrange(a.depth + 1), len(x["ids"]) - t["k"] - a.depth - 2)
            k = t["k"]
            feeds.append((x["rows"][k:k + n], x["ids"][k + 1:k + n + 1]))
            t["k"] = k + n
        if sms is not None:
            argmax = set(rng.sample(sel, len(sel) // 2)) if r % 5 == 4 else set()
            res["mixed_rounds"] += bool(argmax)
            for i in sel:
                t = twins[i]
                kd = None if i in argmax else t["A"].keyed_for(sms[i], t["k"] + 1)
                t["A"].keyed = t["B"].keyed = kd
                res["keyed_lane_rounds"] += kd is not None
        torch.cuda.synchronize()
        E.PACK_GRAPH = False
        t0 = time.perf_counter()
        outs_a = _packed([twins[i]["A"] for i in sel], feeds)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        E.PACK_GRAPH = True
        fn.pack_capture_ok = True
        c0 = fn.pack_graph_stats()["captures"]
        t2 = time.perf_counter()
        outs_b = _packed([twins[i]["B"] for i in sel], feeds)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        E.PACK_GRAPH = False
        fn.pack_capture_ok = False
        captured = fn.pack_graph_stats()["captures"] != c0
        L = len(sel)
        bl = res["by_lanes"].setdefault(L, {"rounds": 0, "equal": 0})
        bl["rounds"] += 1
        same_round = True
        for j, i in enumerate(sel):
            (ta, pa), (tb, pb) = outs_a[j], outs_b[j]
            res["picks"] += len(ta)
            if ta != tb:
                res["tok_diff"] += 1
                same_round = False
            if _bits(pa) != _bits(pb):
                res["prob_diff"] += 1
                same_round = False
            ha, hb = twins[i]["A"], twins[i]["B"]
            if ha.st.length != hb.st.length or not _caches_equal(ha, hb, ha.st.length + a.depth):
                res["cache_diff"] += 1
                same_round = False
            if not same_round and res["first_diff"] is None:
                res["first_diff"] = {"round": r, "lane": i, "lanes": sel, "A": [ta, pa], "B": [tb, pb]}
        res["rounds"] += 1
        res["lane_rounds"] += L
        bl["equal"] += same_round
        if r and not captured:
            res["ms_eager"].setdefault(L, []).append(1e3 * (t1 - t0))
            res["ms_graph"].setdefault(L, []).append(1e3 * (t3 - t2))
        if r < 3 or not same_round or r % 50 == 0:
            log(f"[pgraph] head {hot or 'full'} round {r}: lanes {sel}, rows {[len(f[1]) for f in feeds]}, "
                f"equal {same_round}{' (captured)' if captured else ''}")
    st = fn.pack_graph_stats()
    res["graph"] = {k: (v - stats0[k] if isinstance(v, (int, float)) and k in stats0 and not isinstance(v, bool)
                        else v) for k, v in st.items()}
    res["ms_eager"] = {L: round(float(np.median(v)), 3) for L, v in sorted(res["ms_eager"].items())}
    res["ms_graph"] = {L: round(float(np.median(v)), 3) for L, v in sorted(res["ms_graph"].items())}
    res["all_equal"] = res["tok_diff"] == 0 and res["prob_diff"] == 0 and res["cache_diff"] == 0
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--texts", required=True, help="bare_bench JSON (comma): its prompts and texts are the trace")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", default="32768,0", help="draft heads to check: hot vocabulary sizes, 0 = the full head")
    ap.add_argument("--seed", type=int, default=249)
    ap.add_argument("--keyed", action="store_true", help="coupled drafts: every lane seeded and sampled")
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from engine.models.flashnext.serving import build
    from tools.fn_mtp_accept import g2_bench_texts
    if a.keyed:
        from engine.models.flashnext import coupled as CP
        assert CP.ON, "--keyed needs QWEN38_FN_MTP_COUPLED=1"
    tok = AutoTokenizer.from_pretrained(a.pack)
    texts = g2_bench_texts(a.texts.split(","), tok, 1000)
    L = max(len(t["ids"]) + len(t["out"]) for t in texts) + 256
    t0 = time.time()
    eng, _, _ = build(a.pack, L, tok, spec=True, depth=a.depth, log=lambda s: print(s, flush=True))
    fn = eng.fn
    print(f"[pgraph] loaded in {time.time() - t0:.0f} s; {len(texts)} lanes; keyed {a.keyed}", flush=True)
    lanes = []
    for t in texts:
        ids = list(t["ids"]) + list(t["out"])
        st = fn.new_state(L)
        fn.prefill(torch.tensor(ids, device="cuda"), st, keep_stack=True)
        lanes.append({"name": t["name"], "prompt": t["ids"], "ids": ids, "rows": fn.prompt_stack.clone()})
        del st
        torch.cuda.empty_cache()
    out = {"pack": a.pack, "texts": a.texts, "lanes": [x["name"] for x in lanes],
           "lens": [[len(x["prompt"]), len(x["ids"])] for x in lanes], "depth": a.depth, "rounds_asked": a.rounds,
           "knobs": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(("QWEN38_FN_", "QSE_"))},
           "heads": []}
    for hot in [int(x) for x in a.heads.split(",") if x != ""]:
        res = run_head(fn, lanes, a, hot, lambda s: print(s, flush=True))
        out["heads"].append(res)
        print(f"[pgraph] head {hot or 'full'}: ALL EQUAL {res['all_equal']} over {res['rounds']} rounds, "
              f"{res['lane_rounds']} lane-rounds ({res['keyed_lane_rounds']} keyed, {res['mixed_rounds']} mixed rounds), {res['picks']} picks (top-2 ids + probability bits), caches; "
              f"diffs tok {res['tok_diff']} prob {res['prob_diff']} cache {res['cache_diff']}; by lanes "
              f"{ {k: (v['equal'], v['rounds']) for k, v in sorted(res['by_lanes'].items())} }; graph {res['graph']}; "
              f"ms eager {res['ms_eager']} graph {res['ms_graph']}", flush=True)
    out["all_equal"] = all(h["all_equal"] for h in out["heads"])
    print(f"[pgraph] {'ALL EQUAL' if out['all_equal'] else 'DIFFERENT'}", flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)
    sys.exit(0 if out["all_equal"] else 1)


if __name__ == "__main__":
    with torch.no_grad():
        main()
