"""On the board (SRV-309): Flash-Next's chunk-major prefill with the window cap equals the uncapped one.

Flash-Next has no layer-major window (`window_ok` is False), so the cap splits a chunk into shorter
forwards (engine/cache.py `_prefill`). One engine, one copy of the weights, a fresh state per run:
`cache.prefill` of the same random prompt with no cap and with each cap of `--caps` (4096: the chunk itself, no split -- the control), at each length of
`--lens` (the default ones end on a short last chunk). Per run against the uncapped one: the last-row
logits, every section (GDN states, conv tails, PLE conv and context, mtp.h_last), the K, V and
indexer-key rows below the length, the pooled indexer keys and the length -- `torch.equal`.

Run on the box, the box alone:
    python tests/gpu/test_window_cap_fn_gpu.py --pack ~/flashnext-packs/3.05 [--lens 9000,20000] [--caps 1024,1500]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine import cache  # noqa: E402
from tools.fn_joint_check import _twin_diff  # noqa: E402


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--lens", default="9000,20000")
    ap.add_argument("--caps", default="4096,2048,1024,1500")
    ap.add_argument("--chunk", type=int, default=4096)
    a = ap.parse_args()
    from engine.models.flashnext.serving import FlashNextServing, build
    from transformers import AutoTokenizer
    lens = [int(x) for x in a.lens.split(",") if x]
    caps = [int(x) for x in a.caps.split(",") if x]
    L = max(lens) + 1024
    tok = AutoTokenizer.from_pretrained(a.pack)
    eng, _, _ = build(a.pack, L, tok, spec=False, log=lambda s: print(s, flush=True))
    assert not hasattr(eng, "prefill_window") or not eng.window_ok(0), "Flash-Next is expected to prefill chunk-major"
    bad = []
    for n in lens:
        g = torch.Generator().manual_seed(n + 1)
        ids = torch.randint(1000, 100000, (n,), generator=g).tolist()
        ref = FlashNextServing(eng.fn, L)
        cache.WINDOW_CAP = None
        t0 = time.time()
        lg_ref = cache.prefill(ref, None, ids, "cuda", chunk=a.chunk)[0].float().clone()
        torch.cuda.synchronize()
        print(f"  {n}: uncapped {time.time() - t0:.1f}s", flush=True)
        for cap in caps:
            s = FlashNextServing(eng.fn, L)
            cache.WINDOW_CAP = lambda c=cap: c
            t0 = time.time()
            lg = cache.prefill(s, None, ids, "cuda", chunk=a.chunk)[0].float().clone()
            torch.cuda.synchronize()
            cache.WINDOW_CAP = None
            d = _twin_diff(ref, s)
            ok = torch.equal(lg_ref, lg) and d["equal"]
            print(f"  {n}: cap {cap} {time.time() - t0:.1f}s logits {'=' if torch.equal(lg_ref, lg) else '!='} "
                  f"(max|d| {(lg_ref - lg).abs().max().item():.3e}, argmax {'=' if lg_ref.argmax() == lg.argmax() else '!='}) "
                  f"state {'=' if d['equal'] else str(len(d['sections_differ'])) + ' sections, views ' + str(d['views_differ'])}",
                  flush=True)
            if not ok:
                bad.append(f"{n} cap {cap}")
            del s
        del ref
    print("FAIL: " + "; ".join(bad) if bad else "OK: every capped prefill equals the uncapped one", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
