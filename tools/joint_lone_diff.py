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
    if isinstance(x, dict):
        return {k: _clone(v) for k, v in x.items()}
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
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False, float("nan")
        return _same([a[k] for k in sorted(a, key=str)], [b[k] for k in sorted(b, key=str)])
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return False, float("nan")
    if hasattr(a, "__dict__") and hasattr(b, "__dict__") and type(a) is type(b):
        return _same(vars(a), vars(b))
    try:
        eq = bool(a == b)
    except (RuntimeError, ValueError, TypeError):
        eq = a is b
    return eq, 0.0 if eq else float("nan")


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


def _joint_steps_nojoin(M, nojoin):
    """engine.model.joint_steps with the request kinds in `nojoin` run per engine (each engine's rows alone), every
    other kind joined as served: the brute-force half of the bisect (the kind whose un-joining makes the joint pass
    the lone one's bits is the parting op)."""
    def steps(eng0, gens):
        reqs = [next(g) for g in gens]
        gdn_pass: dict = {}
        done: list = [None] * len(gens)
        live = list(range(len(gens)))
        while live:
            groups: dict = {}
            for j in live:
                kind, name, _ = reqs[j]
                key = (kind, name, j) if kind in nojoin else (kind, name)
                groups.setdefault(key, []).append(j)
            answers: dict = {}
            for key, js in groups.items():
                kind, name = key[0], key[1]
                xs = [reqs[j][2] for j in js]
                if kind == "gdn":
                    for j, r in zip(js, M._run_gdn(eng0, name, xs, gdn_pass.setdefault(tuple(js), {}))):
                        answers[j] = r
                    continue
                if len(xs) == 1:
                    answers[js[0]] = M._run_request(eng0, kind, name, xs[0])
                    continue
                dim = 1 if kind in ("mlp", "head", "addnorm") else 0
                if isinstance(xs[0], tuple):
                    joined = tuple(torch.cat([x[k] for x in xs], dim=dim) for k in range(len(xs[0])))
                    sizes = [x[0].shape[dim] for x in xs]
                else:
                    joined = torch.cat(xs, dim=dim)
                    sizes = [x.shape[dim] for x in xs]
                if kind == "gated":
                    heads = eng0.cfg.linear_num_value_heads
                    sizes = [n // heads for n in sizes]
                out = M._run_request(eng0, kind, name, joined)
                if isinstance(out, tuple):
                    parts = list(zip(*(o.split(sizes, dim=dim) for o in out)))
                else:
                    parts = out.split(sizes, dim=dim)
                for j, part in zip(js, parts):
                    answers[j] = part
            nxt = []
            for j in live:
                try:
                    reqs[j] = gens[j].send(answers[j])
                    nxt.append(j)
                except StopIteration as fin:
                    done[j] = fin.value
            live = nxt
        return done
    return steps


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
    ap.add_argument("--nojoin", default="", help="request kinds run per engine in the joint pass (comma list)")
    ap.add_argument("--sweep", action="store_true", help="each kind un-joined in turn: joint vs lone per kind")
    a = ap.parse_args()
    from engine.config import load_config
    from engine.loader import Weights
    from engine.models.qwen3_5.adapter import Qwen38Engine
    import engine.model as M
    cfg = load_config(None)
    w = Weights(cfg.path, skip_mtp=True)
    if a.sweep:
        return _sweep(a, cfg, w)
    if a.nojoin:
        M.joint_steps = _joint_steps_nojoin(M, set(a.nojoin.split(",")))
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


def _sweep(a, cfg, w) -> None:
    """For each request kind (and all of them): the joint pass with that kind un-joined vs the lone verify, over
    `--rounds`-free single blocks on fresh engines: how many of E blocks come out lone-identical."""
    from engine.models.qwen3_5.adapter import Qwen38Engine
    import engine.model as M
    real = M.joint_steps
    kinds = ["grp", "lin", "gate", "gdn", "gated", "addnorm", "mlp", "head"]
    for nj in [[]] + [[k] for k in kinds] + [kinds]:
        M.joint_steps = _joint_steps_nojoin(M, set(nj)) if nj else real
        rng = random.Random(31)
        g = torch.Generator().manual_seed(37)
        E = a.slots
        ok = 0
        with torch.no_grad():
            J = [Qwen38Engine(cfg, w, max_len=a.max_len) for _ in range(E)]
            L = [Qwen38Engine(cfg, w, max_len=a.max_len) for _ in range(E)]
            items, lone = [], []
            for i in range(E):
                p = torch.randint(1000, 100000, (rng.randint(150, 900),), generator=g).cuda()
                J[i].forward(p, start=0, last_only=True)
                L[i].forward(p, start=0, last_only=True)
                t = _tree(rng, int(p[-1]), a.rows, a.chain)
                b = torch.tensor(t.tokens).cuda()
                items.append((J[i], b, t.parents, p.numel()))
                lone.append(L[i].forward_tree(b, t.parents, start=p.numel()))
            joint = M.forward_tree_multi(items)
            for x, y in zip(joint, lone):
                ok += int(torch.equal(x, y))
        M.joint_steps = real
        print(f"NOJOIN {','.join(nj) if nj else '(none: served)'}: {ok} of {E} blocks lone-identical", flush=True)
        del J, L
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
