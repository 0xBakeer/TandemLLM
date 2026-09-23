"""SPD-22: the block's commit with no state copies, against the engine without the flag.

With `QWEN38_FUSED_COMMIT` a chain verify does not clone the recurrent state: it reads the entry
state and writes its final one into a spare buffer, and the commit rebuilds the live state from
the entry for all layers at once. None of that may change a number. On a CPU the commit runs the
torch reference of the same identity (`tools/gdn_commit_kernels.commit_reference`), so these tests
check the plumbing -- which buffer is read, which is written, what a full accept leaves, what a
partial accept and a tree path leave -- on the random four-layer model of test_forward_tree.
The kernel's own arithmetic is `tools/gdn_commit_kernels.py check()` on the board.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402  (sets the CPU environment before the engine imports)

import torch  # noqa: E402

import engine.model as M  # noqa: E402
from engine.tree import DraftTree  # noqa: E402

TOKS = [7, 11, 23, 5, 31, 2]


def _run(flag: bool, keep: int | None, seed: int = 3):
    M.FUSED_COMMIT = flag
    try:
        eng, pos = T.build(seed=seed)
        before = eng.state.S.clone()
        with torch.no_grad():
            lg = eng.forward_block(torch.tensor(TOKS), start=pos)
            entry_ok = torch.equal(eng._trace.S_entry, before)
            if keep is not None:
                eng.rollback_to(keep)
            nxt = eng.forward_block(torch.tensor([3, 9]), start=pos + (keep or len(TOKS)))
        return lg, nxt, T.snapshot(eng), entry_ok
    finally:
        M.FUSED_COMMIT = False


def test_a_partial_accept_leaves_the_same_state():
    out = []
    for keep in (1, 3, 5):
        lg0, n0, s0, _ = _run(False, keep)
        lg1, n1, s1, ok = _run(True, keep)
        assert ok, "the entry state the trace keeps is not the state before the block"
        assert torch.equal(lg0, lg1), "the verify's logits moved"
        dS, dc = T.diff(s0[0], s1[0]), T.diff(s0[1], s1[1])
        assert dS < 1e-6 and dc == 0.0, (keep, dS, dc)
        assert T.diff(n0, n1) < 1e-5, "the next block reads a different state"
        out.append(f"keep {keep}: S {dS:.1e}")
    return "; ".join(out) + "; conv exact; next block's logits agree"


def test_a_full_accept_without_a_commit_leaves_the_walked_state():
    _, n0, s0, _ = _run(False, None)
    _, n1, s1, _ = _run(True, None)
    assert torch.equal(s0[0], s1[0]) and torch.equal(s0[1], s1[1]), "full accept state differs"
    assert torch.equal(n0, n1)
    return "state and next logits bit-identical when the caller does not roll back"


def test_a_tree_path_commits_to_the_same_state():
    tree = T.BRANCHY
    out = []
    for leaf in tree.leaves():
        path = tree.path(leaf)
        snaps = []
        for flag in (False, True):
            M.FUSED_COMMIT = flag
            try:
                eng, pos = T.build(seed=2)
                with torch.no_grad():
                    eng.forward_tree(torch.tensor(tree.tokens), tree.parents, start=pos)
                    eng.commit_tree(path)
                snaps.append(T.snapshot(eng))
            finally:
                M.FUSED_COMMIT = False
        dS = T.diff(snaps[0][0], snaps[1][0])
        assert dS < 1e-6 and torch.equal(snaps[0][1], snaps[1][1]), (leaf, dS)
        n = len(path)
        assert torch.equal(snaps[0][2][..., :pos + n, :], snaps[1][2][..., :pos + n, :])
        assert snaps[0][4] == snaps[1][4] == pos + n
        out.append(f"leaf {leaf}: S {dS:.1e}")
    return "; ".join(out)


def test_the_two_buffers_trade_places():
    M.FUSED_COMMIT = True
    try:
        eng, pos = T.build(seed=4)
        a = eng.state.S
        with torch.no_grad():
            eng.forward_block(torch.tensor(TOKS[:3]), start=pos)
        b = eng.state.S
        assert b.data_ptr() != a.data_ptr() and eng._trace.S_entry.data_ptr() == a.data_ptr()
        with torch.no_grad():
            eng.rollback_to(2)
            eng.forward_block(torch.tensor(TOKS[:3]), start=pos + 2)
        assert eng.state.S.data_ptr() == a.data_ptr(), "the spare was not reused"
    finally:
        M.FUSED_COMMIT = False
    return "entry kept in place, final in the spare, the pair alternates"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:60s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
