"""RND-37 gate replay (CPU only): committed tokens a block at the served cut, and the true token's
rank at slots 0..2, for lattice recordings of the same traces under several drafters.

Usage: rank_gate.py [--json OUT] LABEL=DIR[@BUDGET] ...

Each DIR holds `record_lattice.py` output (`*.json` traces + `*.lattice.npz`). BUDGET is the node
budget of the served cut (default 16; the b16 arm is served at 32). Per class (the traces' `klass`):
committed tokens a block of `engine.tree.lattice_tree` at that budget (nodes mode, temperature 1.0,
exactly `tools/lattice_oracle.py`'s replay), and following the true path the share of positions
whose true token is the row's first choice, second, third, 4-16th, or absent.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.lattice_oracle import replay  # noqa: E402
from tools.tree_sweep import load  # noqa: E402


def ranks(tr: dict) -> np.ndarray:
    """[3, 5] counts: slot x (rank1, rank2, rank3, rank4-16, absent), following the true path."""
    out = np.zeros((3, 5), dtype=np.int64)
    target = tr["target"]
    for a, i in tr["index"].items():
        cand, sc = tr["cand"][i], tr["scores"][i]
        row = 0
        for l in range(3):
            if a + 1 + l >= len(target):
                break
            tt = int(target[a + 1 + l])
            rc = [int(x) for x in cand[l]]
            if tt not in rc:
                out[l, 4] += 1
                break
            c = rc.index(tt)
            order = np.argsort(-sc[l, row], kind="stable")
            r = int(np.where(order == c)[0][0]) + 1
            out[l, 0 if r == 1 else 1 if r == 2 else 2 if r == 3 else 3] += 1
            row = c
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    res: dict = {}
    for spec in a.runs:
        label, _, rest = spec.partition("=")
        d, _, b = rest.partition("@")
        budget = int(b or 16)
        traces = load(os.path.expanduser(d))
        by = collections.defaultdict(lambda: [0, 0])
        rk = collections.defaultdict(lambda: np.zeros((3, 5), dtype=np.int64))
        for tr in traces:
            bl, cm, _, _ = replay(tr, "tree", budget)
            for k in (tr["klass"], "all"):
                by[k][0] += bl
                by[k][1] += cm
                rk[k] += ranks(tr)
        res[label] = {"dir": d, "budget": budget, "traces": len(traces),
                      "committed": {k: v[1] / max(v[0], 1) for k, v in by.items()},
                      "blocks": {k: v[0] for k, v in by.items()},
                      "rank": {k: (v / np.maximum(v.sum(1, keepdims=True), 1)).round(4).tolist()
                               for k, v in rk.items()}}
    classes = sorted({k for r in res.values() for k in r["committed"]} - {"all"}) + ["all"]
    print(f"{'committed tokens a block':34s} " + " ".join(f"{c:>8s}" for c in classes))
    for label, r in res.items():
        print(f"{label + ' @' + str(r['budget']):34s} "
              + " ".join(f"{r['committed'].get(c, float('nan')):8.3f}" for c in classes))
    print("\nrank of the true token, slots 0/1/2: rank-1 share (rank-2 share, absent)")
    for label, r in res.items():
        for c in classes:
            if c not in r["rank"]:
                continue
            s = r["rank"][c]
            print(f"{label:24s} {c:8s} " + "  ".join(
                f"s{l}: {s[l][0]:.3f} ({s[l][1]:.3f}, {s[l][4]:.3f})" for l in range(3)))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
