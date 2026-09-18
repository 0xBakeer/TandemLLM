"""The anti-repetition penalties, unit-tested on a CPU (ENG-17).

What is being tested is the property the whole design rests on: the penalties are a deterministic
function of the token history, applied to the target's row before the argmax -- so the four claims
below are about *which history each row sees* and *what the transform does to a row*, both of which
are pure tensor logic and need no model at all.

  1. the three formulas (multiplicative rep on negative/positive logits, flat presence,
     per-occurrence frequency) applied only to tokens the history contains;
  2. chain rows see committed history PLUS the drafts above them (vLLM's per-position history);
  3. tree rows see their ANCESTOR PATH, and a sibling leaves nothing behind;
  4. an off spec is a byte-identical no-op, and `seed` resets (the gate tools reuse one state).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from engine.penalty import PenaltySpec, PenaltyState  # noqa: E402
from engine.tree import DraftTree  # noqa: E402

V = 50


def state(rep=1.0, presence=0.0, freq=0.0):
    return PenaltyState(PenaltySpec(rep=rep, presence=presence, freq=freq), V, "cpu")


def row():
    # logits centred on zero so rep's two branches are both exercised
    r = torch.linspace(-2.0, 2.0, V, dtype=torch.float32)
    return r


def test_off_spec_is_a_no_op():
    ps = state()
    assert not ps.spec.on
    r = row()
    before = r.clone()
    ps.seed([3, 5, 3])
    ps.commit([7])
    ps.apply_single(r)
    ps.apply_chain(r.unsqueeze(0), [1, 2])
    assert torch.equal(r, before), "an off spec must not touch a row"
    assert ps.counts is None, "an off spec must not allocate the count vector"


def test_seed_resets_between_runs():
    ps = state(rep=1.05)
    ps.seed([1, 2, 3])
    ps.commit([4, 5])
    assert int(ps.counts.sum()) == 5
    ps.seed([9])
    assert int(ps.counts.sum()) == 1, "seed must reset, or a reused state double-counts"


def test_rep_formula_only_on_history_tokens():
    ps = state(rep=2.0)
    ps.seed([10])                       # token 10 occurs once
    r = row()
    ps.apply_single(r)
    ref = row()
    i = 10
    if ref[i] < 0:
        assert torch.isclose(r[i], ref[i] * 2.0)
    else:
        assert torch.isclose(r[i], ref[i] / 2.0)
    others = torch.cat([r[:i], r[i + 1:]])
    ref_others = torch.cat([ref[:i], ref[i + 1:]])
    assert torch.equal(others, ref_others), "fresh vocabulary must be untouched"


def test_presence_and_frequency():
    ps = state(presence=0.5, freq=0.1)
    ps.seed([10, 10, 11])                # 10 occurs twice, 11 once
    r = row()
    ps.apply_single(r)
    ref = row()
    assert torch.isclose(r[10], ref[10] - 0.5 - 0.1 * 2)
    assert torch.isclose(r[11], ref[11] - 0.5 - 0.1 * 1)
    assert torch.isclose(r[0], ref[0]), "absent tokens pay nothing"


def test_chain_rows_see_the_drafts_above_them():
    ps = state(presence=1.0)
    ps.seed([40])                        # committed history: token 40
    d0, d1 = 41, 42
    lg = torch.stack([row(), row(), row()])     # rows: anchor, d0's row, d1's row
    ps.apply_chain(lg, [d0, d1])
    ref = row()
    # row 0: history {40}
    assert torch.isclose(lg[0][40], ref[40] - 1.0)
    assert torch.isclose(lg[0][41], ref[41]), "a draft below is NOT in row 0's history"
    # row 1: history {40, d0}
    assert torch.isclose(lg[1][40], ref[40] - 1.0)
    assert torch.isclose(lg[1][41], ref[41] - 1.0), "d0 entered after row 0 was penalized"
    # row 2: history {40, d0, d1}
    assert torch.isclose(lg[2][42], ref[42] - 1.0)
    assert torch.isclose(lg[1][42], ref[42]), "d1 is not in row 1's history yet"


def test_tree_rows_see_their_ancestor_path_only():
    # anchor 30 (committed, in the base) with two children and one grandchild:
    #   0:anchor(30)  1:b(31)  2:c(32)  3:d(33, child of 1)
    tree = DraftTree(tokens=[30, 31, 32, 33], parents=[-1, 0, 0, 1])
    ps = state(presence=1.0)
    ps.seed([30])                        # the anchor is committed and already counted
    lg = torch.stack([row(), row(), row(), row()])
    ps.apply_tree(lg, tree)
    ref = row()
    # node 1 (b): history {30}
    assert torch.isclose(lg[1][30], ref[30] - 1.0)
    assert torch.isclose(lg[1][31], ref[31]), "a node's own token is not in its history"
    # node 2 (c, SIBLING of b): history {30} -- b must not have leaked
    assert torch.isclose(lg[2][31], ref[31]), "a sibling's token must not be in node 2's history"
    assert torch.isclose(lg[2][30], ref[30] - 1.0)
    # node 3 (d, child of b): history {30, 31}
    assert torch.isclose(lg[3][31], ref[31] - 1.0), "the ancestor b IS in d's history"
    assert torch.isclose(lg[3][32], ref[32]), "the cousin c must not be in d's history"


def test_deep_tree_walk_pops_correctly():
    # a two-branch tree: DFS visits 1,2 (chain under b) then 3 (second child of anchor)
    tree = DraftTree(tokens=[10, 11, 12, 13], parents=[-1, 0, 1, 0])
    ps = state(presence=1.0)
    ps.seed([])
    lg = torch.stack([row(), row(), row(), row()])
    ps.apply_tree(lg, tree)
    ref = row()
    assert torch.isclose(lg[2][11], ref[11] - 1.0), "node 2's parent b is in its history"
    assert torch.isclose(lg[3][11], ref[11]), "after popping back, b must be gone for node 3"


def test_ranges_are_validated():
    for bad in (lambda: PenaltySpec(rep=0.0), lambda: PenaltySpec(presence=2.5),
                lambda: PenaltySpec(freq=-3.0)):
        try:
            bad()
            raise AssertionError("out-of-range penalty must raise")
        except ValueError:
            pass
