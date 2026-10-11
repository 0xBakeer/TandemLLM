"""ENG-227: the host side of Flash-Next's verify blocks, without a GPU.

  * `tree_ctl` (the GDN tile's load/save per row): replayed symbolically over random DFS pre-order
    trees, every row starts from its parent's state (or the entry state for a root) and the scratch
    slots never exceed a row count's worth;
  * `window_src` (where a row's conv taps live): equal to a brute-force walk over positions, for the
    GDN conv (distances 3, 2, 1, a state of 3) and the PLE conv (9, 6, 3, a state of 9);
  * `walk` follows the target's token through the children and stops where none carries it;
  * `FlashNextEngine._tail_cols` (the conv state after a commit) is the last positions of the path;
  * `FlashNextEngine._stage_qsa` (ENG-147): dense rows see every position, sparse rows 2,048 selected
    positions and their tail; a row's own blocks map positions before the block start to cache slots
    and later ones to the row's ancestors; `qsa_class` widens by powers of two past the budget;
  * `StairCut` cuts a draft into a valid DFS block, keeps a confident chain and drops an unsure one,
    adds a sibling only when it pays, and learns its calibration from the walk.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.tree import DraftTree  # noqa: E402
from tools.fn_seq_kernels import tree_ctl, window_src  # noqa: E402


def _random_tree(rng: random.Random, n: int) -> list[int]:
    """A random DFS pre-order block of n rows (row 0 the anchor)."""
    parents = [-1]
    stack = [0]                                     # the path to the last row
    for i in range(1, n):
        stack = stack[:rng.randint(1, len(stack))]  # climb to a random ancestor of the last row
        parents.append(stack[-1])
        stack.append(i)
    DraftTree(tokens=list(range(n)), parents=parents).check()
    return parents


def test_tree_ctl_replays_every_parent_state():
    rng = random.Random(1)
    for n in list(range(1, 33)) * 8:
        parents = _random_tree(rng, n)
        ctl, slots = tree_ctl(parents)
        regs, stk = None, {}
        for i, (ld, sv) in enumerate(ctl):
            if ld == -2:
                start = "entry"
            elif ld == -1:
                start = regs
            else:
                start = stk[ld]
            want = "entry" if parents[i] < 0 else ("after", parents[i])
            assert start == want, (parents, i, start, want)
            regs = ("after", i)
            if sv >= 0:
                assert 0 <= sv < slots
                stk[sv] = regs
        assert slots <= max(n - 1, 0)
        if parents == list(range(-1, n - 1)):
            assert slots == 0 and all(c == (-1, -1) for c in ctl[1:])
    return "256 random trees of 1..32 rows; a chain uses no slot"


def _brute(parents: list[int], dists, state_len: int) -> list[list[int]]:
    depth = []
    for p in parents:
        depth.append(0 if p < 0 else depth[p] + 1)
    out = []
    for i in range(len(parents)):
        row = []
        for d in dists:
            a, k = i, d
            while k and a >= 0:
                a, k = parents[a], k - 1
            if a >= 0:
                row.append(a)
            else:
                # position depth[i] - d before the block start: state column state_len - (d - depth[i])
                row.append(-1 - (state_len - (d - depth[i])))
        out.append(row)
    return out


def test_window_src_equals_brute_force():
    rng = random.Random(2)
    for n in list(range(1, 33)) * 4:
        parents = _random_tree(rng, n)
        assert window_src(parents, (3, 2, 1), 3) == _brute(parents, (3, 2, 1), 3), parents
        assert window_src(parents, (9, 6, 3), 9) == _brute(parents, (9, 6, 3), 9), parents
    assert window_src([-1], (3, 2, 1), 3) == [[-1, -2, -3]]
    assert window_src([-1], (9, 6, 3), 9) == [[-1, -4, -7]]
    return "128 random trees, GDN and PLE windows; the one-row windows are the decode step's"


def test_walk():
    from engine.models.flashnext.spec import walk
    tokens = [5, 10, 11, 20, 21, 30, 12, 22]
    parents = [-1, 0, 1, 1, 3, 4, 0, 6]
    assert walk(tokens, parents, [12, 0, 0, 0, 0, 0, 22, 9]) == [0, 6, 7]
    assert walk(tokens, parents, [10, 20, 0, 21, 30, 1, 0, 0]) == [0, 1, 3, 4, 5]
    assert walk(tokens, parents, [99] * 8) == [0]
    return "paths through first and later children, and the empty path"


def test_tail_cols():
    from engine.models.flashnext.engine import FlashNextEngine
    tc = FlashNextEngine._tail_cols
    assert tc([0], 3) == [1, 2, 3]
    assert tc([0, 2], 3) == [2, 3, 5]
    assert tc([0, 2, 5, 6], 3) == [5, 8, 9]
    assert tc([0, 1], 9) == [2, 3, 4, 5, 6, 7, 8, 9, 10]
    return "the conv state after a path: its last positions, the state's columns before it"


class _Cfg:
    has_indexer = True
    indexer_compress_ratio = 4


class _H(dict):
    pass


def _qsa_buffers(M):
    import torch
    from tools.fn_seq_kernels import ANC, OWN
    z = lambda *s: torch.zeros(*s, dtype=torch.int32)          # noqa: E731
    return {"actl": z(M, 4), "oslot": z(M, OWN * 4), "opos": z(M, OWN), "qctl": z(M, 4), "tail": z(M, 3),
            "pidx": torch.zeros(1, dtype=torch.int64), "anc": z(M, ANC)}


def test_stage_qsa_rows():
    import numpy as np
    from engine.models.flashnext.engine import FlashNextEngine
    from tools.fn_seq_kernels import OWN
    rng = random.Random(3)
    eng = type("E", (), {"cfg": _Cfg()})()
    for L in (100, 2045, 2050, 3001, 8190):
        parents = _random_tree(rng, 12)
        depth = []
        for p in parents:
            depth.append(0 if p < 0 else depth[p] + 1)
        paths = []
        for i, p in enumerate(parents):
            paths.append([i] if p < 0 else paths[p] + [i])
        h = _qsa_buffers(12)
        anc = h["anc"].numpy()
        for i, pa in enumerate(paths):
            anc[i, :len(pa)] = L + np.asarray(pa)
        pos = L + np.asarray(depth)
        FlashNextEngine._stage_qsa(eng, h, L, L // 4, pos, anc, 12)
        for i in range(12):
            p, nblk = int(pos[i]), (int(pos[i]) + 1) // 4
            n, Lb, dense = h["actl"][i, :3].tolist()
            assert Lb == L and dense == int(nblk <= 512)
            assert n == (p + 1 if nblk <= 512 else 2048 + p + 1 - 4 * nblk), (L, i, n)
            assert h["qctl"][i, :2].tolist() == [nblk, L // 4]
            assert h["tail"][i].tolist() == [4 * nblk + k for k in range(3)]
            for j in range(OWN):
                assert int(h["opos"][i, j]) == 4 * (L // 4 + j)
                for k in range(4):
                    q = 4 * (L // 4 + j) + k
                    want = q if q < L else (L + paths[i][q - L] if q <= p else 0)
                    assert int(h["oslot"][i, 4 * j + k]) == want, (L, i, j, k)
    cls = FlashNextEngine.qsa_class
    eng2 = type("E", (), {"cfg": _Cfg()})()
    assert cls(eng2, 2050) == 0 and cls(eng2, 2051) == 1024 and cls(eng2, 4095) == 1024
    assert cls(eng2, 4096) == 2048 and cls(eng2, 32767) == 8192
    return "dense/sparse counts, own-block slots through the ancestors, tails, classes"


def test_staircut():
    from engine.models.flashnext.spec import StairCut
    sc = StairCut(siblings=True)
    tok = [[10 + k, 20 + k] for k in range(5)]
    toks, par = sc.cut(1, tok, [[0.99, 0.005]] * 5)
    assert toks == [1, 10, 11, 12, 13, 14] and par == [-1, 0, 1, 2, 3, 4], (toks, par)
    toks, par = sc.cut(1, tok, [[0.05, 0.02]] * 5)
    assert toks == [1], toks
    toks, par = sc.cut(1, tok, [[0.3, 0.65]] + [[0.97, 0.01]] * 4)
    DraftTree(tokens=toks, parents=par).check()
    assert 20 in toks and toks.index(20) < toks.index(10), (toks, par)          # sibling before the spine
    # learning: a spine token at p 0.95 that is always rejected pulls its bin down
    before = sc.c("spine", 0.95)
    for _ in range(50):
        sc.cut(1, tok, [[0.95, 0.01]] * 5)
        sc.observe([0], 30.0, 2.0)
    assert sc.c("spine", 0.95) < 0.2 < before
    return "confident chain kept, unsure dropped, a paying sibling first, calibration learned"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<44} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<44} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
