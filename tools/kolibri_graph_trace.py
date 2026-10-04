"""Where the decode step's time goes as the CUDA graph runs it. With programmatic dependent
launch the kernels overlap, so an eager profile no longer adds up; this reads an Nsight Systems
trace of graph replays instead and charges each kernel the time from the previous kernel's end to
its own end (the part of the step it alone holds up), next to its own duration.

    nsys profile --cuda-graph-trace=node --capture-range=cudaProfilerApi -o /tmp/k \
        python tools/kolibri_graph_trace.py run [--ctx 1024] [--steps 16] [KOLIBRI_* switches in env]
    nsys export --type sqlite -o /tmp/k.sqlite /tmp/k.nsys-rep
    python tools/kolibri_graph_trace.py report /tmp/k.sqlite
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import defaultdict


def run(a):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine.kolibri.model import KolibriEngine
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.fp8 or a.set, "tokenizer.json"))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ids = tok.encode(open(os.path.join(root, "bench", "heldout_prose.txt")).read(), add_special_tokens=False).ids
    while len(ids) < a.ctx + 64:
        ids = ids + ids
    eng = KolibriEngine.load(a.set, a.fp8 or None, max_len=a.ctx + 1024, graphs=True)
    eng.prefill(ids[:a.ctx])
    lg = eng.decode(int(ids[a.ctx]))
    for _ in range(8):
        lg = eng.decode(int(lg.argmax()))
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(a.steps):
        lg = eng.decode(int(lg.argmax()))
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def report(a):
    db = sqlite3.connect(a.sqlite)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    rows = db.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
    # a step = the kernels between two host gaps; group by the head GEMV, the step's last kernel
    steps, cur = [], []
    for s, e, n in rows:
        cur.append((s, e, names.get(n, str(n))))
        if "head" in names.get(n, ""):
            steps.append(cur)
            cur = []
    steps = steps[1:]                                       # the first may be cut by the capture range
    dur, crit, gap, cnt = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(int)
    span = 0.0
    for st in steps:
        span += (st[-1][1] - st[0][0]) / 1e3
        prev_end = st[0][0]
        for s, e, n in st:
            dur[n] += (e - s) / 1e3
            crit[n] += max(0, e - prev_end) / 1e3          # includes any idle gap before it
            gap[n] += max(0, s - prev_end) / 1e3
            cnt[n] += 1
            prev_end = max(prev_end, e)
    k = max(1, len(steps))
    print(f"{len(steps)} steps, {span / k / 1e3:.3f} ms a step from first start to last end, "
          f"{sum(cnt.values()) // k} kernels a step")
    print(f"{'kernel':48s} {'n':>4s} {'own us':>8s} {'held us':>8s} {'gap us':>7s} {'held ms/step':>12s}")
    for n in sorted(crit, key=lambda x: -crit[x]):
        c = cnt[n] / k
        print(f"{n[:48]:48s} {c:4.0f} {dur[n] / cnt[n]:8.1f} {crit[n] / cnt[n]:8.1f} {gap[n] / cnt[n]:7.1f} "
              f"{crit[n] / k / 1e3:12.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", required=True, help="the NVFP4 set directory")
    r.add_argument("--fp8", default="", help="the FP8 release, for attention (empty: from the set)")
    r.add_argument("--ctx", type=int, default=1024)
    r.add_argument("--steps", type=int, default=16)
    p = sub.add_parser("report")
    p.add_argument("sqlite")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    main()
