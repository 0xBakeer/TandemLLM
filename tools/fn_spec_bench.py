"""Flash-Next single-stream decode with MTP speculation on the board (ENG-153): what a round costs.

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_spec_bench.py --pack ~/fn2/packs/3.05 \
        --classes prose,code,chat --ctx 1024 --new 256 --depths 1,2,3,4 --json out.json

Per class (tools/fn_generate.py's prompts at `--ctx`), on one state reused by every run (so the
captured graphs are reused): plain greedy decode through `DecodeGraph`, then greedy speculation with
the MTP chain drafter (`MTPGraphDrafter`) at each depth. Each configuration runs once to capture its
graphs (not timed) and once timed. Recorded: tok/s = (tokens - 1) / the time from the first round's
start to the last round's end; the round's parts measured on the host after a device sync (draft =
the MTP graph incl. its heads, verify = staging + the verify graph, commit = the GDN re-walk and the
cache moves); tokens a round; the p50 round; whether the text equals the plain text. A verify of N
rows reads the union of the experts its rows route to, so its cost is measured, never assumed.
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
from engine.models.flashnext.engine import DecodeGraph, FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402
from engine.models.flashnext.spec import MTPGraphDrafter, SpecDecoder  # noqa: E402
from tools.fn_generate import class_prompt, mem_available_gib  # noqa: E402


class Bench:
    def __init__(self, eng: FlashNextEngine, eos: set):
        self.eng, self.eos = eng, eos
        self.st = eng.new_state(eng.max_len)
        self.dg = None
        self.drafters: dict = {}
        self.decoders: dict = {}

    def prefill(self, ids: list[int]) -> int:
        eng = self.eng
        eng.reset(self.st)
        return int(eng.prefill(torch.tensor(ids), self.st)[0].argmax())

    def plain(self, ids: list[int], new: int) -> tuple[list[int], dict]:
        first = self.prefill(ids)
        if self.dg is None:
            self.dg = DecodeGraph(self.eng, self.st)
            self.dg.capture()
        out, per = [first], []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while len(out) < new and out[-1] not in self.eos:
            ts = time.perf_counter()
            out.append(self.dg.greedy(out[-1]))
            per.append(time.perf_counter() - ts)
        dt = time.perf_counter() - t0
        return out, {"tok_s": round((len(out) - 1) / dt, 2), "ms_p50": round(1e3 * statistics.median(per), 2)}

    def spec(self, ids: list[int], new: int, depth: int, hot: int) -> tuple[list[int], dict]:
        eng = self.eng
        first = self.prefill(ids)
        stack = eng.stack.clone()
        dr = self.drafters.get((depth, hot))
        if dr is None:
            dr = self.drafters[(depth, hot)] = MTPGraphDrafter(eng, depth, eng.max_len)
        sd = self.decoders.get(depth)
        if sd is None:
            sd = self.decoders[depth] = SpecDecoder(eng, self.st, dr, graphs=True)
        sd.drafter = dr
        dr.st.length = 0
        dr.st.pool_n = {}
        dr.prime(stack, ids, first)
        dr.capture_all()
        sd.stats = {k: 0 if isinstance(v, int) else 0.0 for k, v in sd.stats.items()}
        rounds: list = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = sd.generate(first, new, self.eos, round_ms=rounds)
        dt = time.perf_counter() - t0
        s = sd.stats
        R = max(s["rounds"], 1)
        return out, {"tok_s": round((len(out) - 1) / dt, 2), "rounds": s["rounds"],
                     "tokens_a_round": round((len(out) - 1) / R, 3),
                     "accepted_a_round": round(s["accepted"] / R, 3),
                     "draft_ms": round(1e3 * s["draft_s"] / R, 2), "verify_ms": round(1e3 * s["verify_s"] / R, 2),
                     "commit_ms": round(1e3 * s["commit_s"] / R, 2),
                     "round_ms_p50": round(statistics.median(r[0] for r in rounds), 2) if rounds else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--classes", default="prose,code,chat")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--depths", default="1,2,3,4")
    ap.add_argument("--hot", default="0", help="draft-head vocabularies: 0 = the full head, K = the first K "
                    "ids of ops/fn-hot-vocab.npy (int8 rows)")
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    idx, W, banks, tables, timing = load(a.pack, "cuda", mtp=True)
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=4096)
    tok = AutoTokenizer.from_pretrained(a.pack)
    eos = set(idx.cfg.eos_token_ids)
    depths = [int(x) for x in a.depths.split(",")]
    hots = [int(x) for x in a.hot.split(",")]
    import numpy as np
    hot_ids = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ops", "fn-hot-vocab.npy"))
    from engine.settings import SETTINGS
    res = {"pack": a.pack, "load": timing, "mem_after_load_gib": mem_available_gib(),
           "knobs": {"ctx": a.ctx, "new": a.new, "depths": depths, "hot": hots, "greedy": True, "thinking": False, "batch": 1,
                     "kv_dtype": "bf16", "max_len": 4096, "drafter": "MTP chain, graphs; head: hot 0 = full EXL3 (248,320), else int8 rows of the first K hot ids",
                     "verify": "graphs per row count", "QWEN38_FN_PDL": SETTINGS.get("FN_PDL"),
                     "QWEN38_FN_DENSE": SETTINGS.get("FN_DENSE"), "QWEN38_FN_MIX": SETTINGS.get("FN_MIX"),
                     "env": {k: v for k, v in os.environ.items() if k.startswith("QWEN38_")}},
           "classes": {}}
    b = Bench(eng, eos)
    for c in a.classes.split(","):
        ids = class_prompt(tok, c, a.ctx)
        b.plain(ids, 32)                                  # warm: the decode graph
        P, pm = b.plain(ids, a.new)
        row = {"prompt_tokens": len(ids), "plain": pm, "spec": {}}
        print(f"[bench] {c} plain {json.dumps(pm)}", flush=True)
        for hot in hots:
            eng.set_draft_head(hot_ids[:hot] if hot else None)
            for d in depths:
                b.spec(ids, a.new, d, hot)                # warm: every graph this depth uses
                S, sm = b.spec(ids, a.new, d, hot)
                sm["equal_plain"] = S == P
                tag = str(d) if not hot else f"{d}/hot{hot}"
                row["spec"][tag] = sm
                print(f"[bench] {c} depth {tag} {json.dumps(sm)}", flush=True)
        res["classes"][c] = row
    res["mem_end_gib"] = mem_available_gib()
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
