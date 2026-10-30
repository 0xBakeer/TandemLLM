"""SPD-201: the greedy walk of every slot's verified tree on the device (the runner-owned round).

The host walk (`Qwen38Engine.accept_tree`, then `server/app.py _end_cut`) reads each slot's picks back
and walks its tree in Python. Here one program a slot does the same on the device, so the round's next
steps -- the commit prep, the drafter's sync, the packed draft -- can be queued behind it with no host in
between:

  node = 0; loop: want = picks[node]; the next node is the lowest-index child of `node` whose token is
  `want` (DFS order: `accept_tree`'s `next(c for c in range(node + 1, n) ...)`); stop where there is none.
  new = the path's tokens after the anchor, then picks[last]; the end cut keeps new through its first
  end token and at most `room` tokens (at least one), and the path to match.

Outputs per slot: `path` [T] (node indices, -1 past `plen`), `plen`, `new` [T] (-1 past `nnew`), `nnew`,
and `anchor` (the last kept token) and `adv` (rows committed: plen) for the next round's buffers.
`reference` is the host statements on the same tensors; `check()` holds the kernel to it.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _walk(PICKS, TOK, PAR, N, EOS, NEOS, ROOM, PATH, PLEN, NEW, NNEW, ANCH,
              s_p, s_t, s_r, s_e, s_path, s_new, T: tl.constexpr, KE: tl.constexpr):
        e = tl.program_id(0)
        idx = tl.arange(0, T)
        n = tl.load(N + e)
        tok = tl.load(TOK + e * s_t + idx, mask=idx < n, other=-1)
        par = tl.load(PAR + e * s_r + idx, mask=idx < n, other=-2)
        picks = tl.load(PICKS + e * s_p + idx, mask=idx < n, other=-1)
        # the walk: path[k] = node after k steps
        path = tl.where(idx == 0, 0, -1)
        node = 0
        alive = 1
        plen = 1
        for step in tl.static_range(T - 1):
            want = tl.sum(tl.where(idx == node, picks, 0))
            cand = (par == node) & (tok == want) & (idx > node) & (idx < n)
            c = tl.min(tl.where(cand, idx, T))
            go = (alive == 1) & (c < T)
            path = tl.where((idx == step + 1) & go, c, path)
            node = tl.where(go, c, node)
            plen = tl.where(go, plen + 1, plen)
            alive = tl.where(go, 1, 0)
        # new tokens: the path's tokens after the anchor, then the pick at the last node
        last_pick = tl.sum(tl.where(idx == node, picks, 0))
        # tok at path[k + 1] for k < plen - 1
        newv = tl.full([T], -1, tl.int64)
        for k in tl.static_range(T - 1):
            pk = tl.sum(tl.where(idx == k + 1, path, 0))
            tk = tl.sum(tl.where(idx == pk, tok, 0))
            newv = tl.where((idx == k) & (k + 1 < plen), tk, newv)
        newv = tl.where(idx == plen - 1, last_pick, newv)
        nnew = plen
        # the end cut: through the first end token, at most `room`, at least one
        ke = tl.arange(0, KE)
        neos = tl.load(NEOS + e)
        eos = tl.load(EOS + e * s_e + ke, mask=ke < neos, other=-3)
        hit = tl.sum(tl.where(newv[:, None] == eos[None, :], 1, 0), axis=1) > 0
        hit = hit & (idx < nnew)
        first = tl.min(tl.where(hit, idx, T))
        keep = tl.where(first < T, first + 1, nnew)
        room = tl.load(ROOM + e)
        keep = tl.maximum(1, tl.minimum(keep, room))
        nnew = keep
        plen = tl.minimum(plen, keep + 1)
        path = tl.where(idx < plen, path, -1)
        newv = tl.where(idx < nnew, newv, -1)
        tl.store(PATH + e * s_path + idx, path.to(tl.int32))
        tl.store(NEW + e * s_new + idx, newv)
        tl.store(PLEN + e, plen.to(tl.int32))
        tl.store(NNEW + e, nnew.to(tl.int32))
        anchor = tl.sum(tl.where(idx == nnew - 1, newv, 0))
        tl.store(ANCH + e, anchor)


def walk_greedy(picks, tokens, parents, n, eos, neos, room):
    """picks / tokens [E, T] int64, parents [E, T] int32 (or int64), n [E] int32, eos [E, KE] int64 (padded),
    neos [E] int32, room [E] int32 -> (path [E, T] int32, plen [E] int32, new [E, T] int64, nnew [E] int32,
    anchor [E] int64), all on the device."""
    E, T = tokens.shape
    dev = tokens.device
    path = torch.empty(E, T, dtype=torch.int32, device=dev)
    new = torch.empty(E, T, dtype=torch.int64, device=dev)
    plen = torch.empty(E, dtype=torch.int32, device=dev)
    nnew = torch.empty(E, dtype=torch.int32, device=dev)
    anchor = torch.empty(E, dtype=torch.int64, device=dev)
    if not dev.type == "cuda" or not HAVE_TRITON:
        return reference(picks, tokens, parents, n, eos, neos, room)
    TP = triton.next_power_of_2(max(2, T))
    KE = triton.next_power_of_2(max(1, eos.shape[1]))
    assert TP == T, "pad the slots' rows to a power of two"
    _walk[(E,)](picks, tokens, parents, n, eos, neos, room, path, plen, new, nnew, anchor,
                picks.stride(0), tokens.stride(0), parents.stride(0), eos.stride(0), path.stride(0),
                new.stride(0), T=TP, KE=KE, num_warps=1)
    return path, plen, new, nnew, anchor


def reference(picks, tokens, parents, n, eos, neos, room):
    """The host statements (`accept_tree`, `_end_cut`) on the same tensors, slot by slot."""
    E, T = tokens.shape
    path_o = torch.full((E, T), -1, dtype=torch.int32)
    new_o = torch.full((E, T), -1, dtype=torch.int64)
    plen_o = torch.zeros(E, dtype=torch.int32)
    nnew_o = torch.zeros(E, dtype=torch.int32)
    anch_o = torch.zeros(E, dtype=torch.int64)
    P, K, R = picks.cpu().tolist(), tokens.cpu().tolist(), parents.cpu().tolist()
    for e in range(E):
        m = int(n[e])
        tok, par, pk = K[e][:m], R[e][:m], P[e][:m]
        node, path = 0, [0]
        while True:
            want = pk[node]
            nxt = next((c for c in range(node + 1, m) if par[c] == node and tok[c] == want), None)
            if nxt is None:
                break
            path.append(nxt)
            node = nxt
        new = [tok[i] for i in path[1:]] + [pk[node]]
        es = set(int(x) for x in eos[e][:int(neos[e])].tolist())
        j = next((i for i, t in enumerate(new) if t in es), None)
        keep = max(1, min(len(new) if j is None else j + 1, int(room[e])))
        path, new = path[:keep + 1], new[:keep]
        path_o[e, :len(path)] = torch.tensor(path, dtype=torch.int32)
        new_o[e, :len(new)] = torch.tensor(new)
        plen_o[e], nnew_o[e], anch_o[e] = len(path), len(new), new[-1]
    dev = tokens.device
    return (path_o.to(dev), plen_o.to(dev), new_o.to(dev), nnew_o.to(dev), anch_o.to(dev))


def _battery(E: int, T: int, seed: int, dev):
    import random
    rng = random.Random(seed)
    V = 50
    picks = torch.zeros(E, T, dtype=torch.int64)
    tokens = torch.zeros(E, T, dtype=torch.int64)
    parents = torch.full((E, T), -1, dtype=torch.int32)
    n = torch.zeros(E, dtype=torch.int32)
    KE = 4
    eos = torch.full((E, KE), -1, dtype=torch.int64)
    neos = torch.zeros(E, dtype=torch.int32)
    room = torch.zeros(E, dtype=torch.int32)
    for e in range(E):
        m = rng.randint(1, T)
        n[e] = m
        path = [0]
        for i in range(1, m):
            path = path[:rng.randint(1, len(path))]
            parents[e, i] = path[-1]
            path.append(i)
        for i in range(m):
            tokens[e, i] = rng.randrange(V)
        for i in range(m):
            kids = [c for c in range(i + 1, m) if int(parents[e, c]) == i]
            picks[e, i] = int(tokens[e, rng.choice(kids)]) if kids and rng.random() < 0.8 else rng.randrange(V)
        k = rng.randint(0, KE)
        neos[e] = k
        for j in range(k):
            eos[e, j] = rng.randrange(V)
        room[e] = rng.randint(1, T + 2)
    return [x.to(dev) for x in (picks, tokens, parents, n, eos, neos, room)]


def check(rounds: int = 300) -> list[str]:
    dev = torch.device("cuda")
    deep = 0
    for r in range(rounds):
        T = (2, 4, 8, 16, 32)[r % 5]
        args = _battery(8, T, r, dev)
        got = walk_greedy(*args)
        ref = reference(*args)
        for a, b, name in zip(got, ref, ("path", "plen", "new", "nnew", "anchor")):
            assert torch.equal(a, b), (r, T, name, a, b)
        deep += int((got[1] > 2).sum())
    return [f"{rounds} rounds of 8 slots (T 2..32, EOS sets, budgets): path, lengths, new tokens and anchors "
            f"equal the host walk ({deep} slots past depth 1)"]


if __name__ == "__main__":
    print("\n".join(check()))
