"""ENG-227: the host side of Flash-Next's verify blocks, without a GPU.

  * `tree_ctl` (the GDN tile's load/save per row): replayed symbolically over random DFS pre-order
    trees, every row starts from its parent's state (or the entry state for a root) and the scratch
    slots never exceed a row count's worth;
  * `window_src` (where a row's conv taps live): equal to a brute-force walk over positions, for the
    GDN conv (distances 3, 2, 1, a state of 3) and the PLE conv (9, 6, 3, a state of 9);
  * `walk` follows the target's token through the children and stops where none carries it;
  * `FlashNextEngine._tail_cols` (the conv state after a commit) is the last positions of the path.
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
