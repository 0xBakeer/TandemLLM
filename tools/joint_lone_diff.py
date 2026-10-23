#!/usr/bin/env python3
"""SPD-E5 exactness: where do a block's bits in the joint pass part from the same block verified alone?

E engines over one copy of the weights verify one tree each, three ways, from the same prefill:

  A. the joint pass (`engine.model.forward_tree_multi` over all E blocks: each request for a weight is one
     call over every engine's rows);
  B. each block alone through the same step generators (`forward_tree_multi` with one item);
  C. each block's lone tree verify (`forward_tree`: the verify graphs or the eager tree path, a chain's own
     kernel for a chain).

Every engine's step generator is wrapped, so each request it yields (kind, weight, its rows) and each answer
it receives are kept, for A and for B. The first step at which an engine's request differs between A and B
is a step of the engine's own code that saw different inputs; the first answer that differs while its
request was the same is the joined call that is not row-invariant. A against C and B against C tell whether
the lone path is another computation altogether. Prints, per engine, the first parting step (index, kind,
weight, the max |difference|), and the logits' status.

    python tools/joint_lone_diff.py --slots 4 --rows 8 [--chain]

The served environment (ops/serve.env sourced), the server stopped or enough memory beside it.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _clone(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, (tuple, list)):
        return type(x)(_clone(v) for v in x)
    return x


def _same(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return False, float("inf")
        if torch.equal(a, b):
            return True, 0.0
        return False, float((a.float() - b.float()).abs().max())
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and len(a) == len(b):
        worst, ok = 0.0, True
        for x, y in zip(a, b):
            s, d = _same(x, y)
            ok &= s
            worst = max(worst, d)
        return ok, worst
    return a == b, 0.0 if a == b else float("nan")


def _recorded(gen, log):
    """The step generator `gen`, its requests and answers appended to `log`."""
    try:
        req = next(gen)
    except StopIteration as fin:
        return fin.value
    while True:
        log.append(("req", req[0], req[1], _clone(req[2])))
        ans = yield req
        log.append(("ans", req[0], req[1], _clone(ans)))
        try:
            req = gen.send(ans)
        except StopIteration as fin:
            return fin.value


def _tree(rng, anchor, n, chain):
    from engine.tree import DraftTree
    parents, path = [-1], [0]
    for i in range(1, n):
        if not chain:
            path = path[:rng.randint(1, len(path))] if rng.random() < 0.6 else path
        parents.append(path[-1])
        path.append(i)
    return DraftTree([anchor] + [rng.randint(1000, 100000) for _ in range(n - 1)], parents)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--chain", action="store_true", help="chain-shaped blocks (the lone path's chain kernel)")
    ap.add_argument("--max-len", type=int, default=4096)
    a = ap.parse_args()
    from engine.config import load_config
    from engine.loader import Weights
    from engine.models.qwen3_5.adapter import Qwen38Engine
    import engine.model as M
    cfg = load_config(None)
    w = Weights(cfg.path, skip_mtp=True)
    rng = random.Random(31)
    g = torch.Generator().manual_seed(37)
    E = a.slots
    sets = {k: [Qwen38Engine(cfg, w, max_len=a.max_len) for _ in range(E)] for k in "ABC"}
    pos, blocks, trees = [], [], []
    with torch.no_grad():
        for i in range(E):
            p = torch.randint(1000, 100000, (rng.randint(150, 900),), generator=g).cuda()
            for k in "ABC":
                sets[k][i].forward(p, start=0, last_only=True)
            pos.append(p.numel())
            t = _tree(rng, int(p[-1]), a.rows, a.chain)
            trees.append(t)
            blocks.append(torch.tensor(t.tokens).cuda())
        logs = {"A": [[] for _ in range(E)], "B": [[] for _ in range(E)]}
        real = M.Qwen38Engine._forward_steps

        def wrapped(run, i):
            def steps(self, tokens, start):
                return _recorded(real(self, tokens, start), logs[run][i])
            return steps

        # A: the joint pass, each engine's generator recorded
        gens_patch = []
        for i in range(E):
            eng = sets["A"][i]
            eng._forward_steps = wrapped("A", i).__get__(eng)
        lA = M.forward_tree_multi([(sets["A"][i], blocks[i], trees[i].parents, pos[i]) for i in range(E)])
        # B: each block alone through the same generators
        lB = []
        for i in range(E):
            eng = sets["B"][i]
            eng._forward_steps = wrapped("B", i).__get__(eng)
            lB.append(M.forward_tree_multi([(eng, blocks[i], trees[i].parents, pos[i])])[0])
        # C: the lone tree verify
        lC = [sets["C"][i].forward_tree(blocks[i], trees[i].parents, start=pos[i]) for i in range(E)]
    for i in range(E):
        la, lb = logs["A"][i], logs["B"][i]
        first = None
        for k, (x, y) in enumerate(zip(la, lb)):
            ok, d = _same(x[3], y[3])
            if not ok:
                first = (k, x[0], x[1], x[2], d)
                break
        ab = _same(lA[i], lB[i])
        bc = _same(lB[i], lC[i])
        ac = _same(lA[i], lC[i])
        where = ("none" if first is None else
                 f"step {first[0]} ({first[1]} of {first[2]!s} {first[3]!s}), max |d| {first[4]:.3g}")
        print(f"DIFF engine {i} rows {a.rows}{' chain' if a.chain else ''}: A vs B logits "
              f"{'identical' if ab[0] else f'differ {ab[1]:.3g}'}; B vs C {'identical' if bc[0] else f'differ {bc[1]:.3g}'}; "
              f"A vs C {'identical' if ac[0] else f'differ {ac[1]:.3g}'}; first parting A vs B: {where} "
              f"(steps {len(la)} / {len(lb)})", flush=True)


if __name__ == "__main__":
    main()
