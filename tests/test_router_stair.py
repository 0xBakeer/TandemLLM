"""The per-block node count on the verify staircase (MergedRouter's `stair` cut) and the length
router's `wide` mode that uses it (`QWEN38_LEN_SWITCH=1`, `QWEN38_LEN_MODE=wide`)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.drafters.ngram import NgramDrafter  # noqa: E402
from engine.lenrouter import STAIR_MS, LengthRouter  # noqa: E402
from engine.router import MergedRouter  # noqa: E402
from engine.tree import DraftTree  # noqa: E402
from tests.test_lenrouter import FakeDrafter, FakeEng  # noqa: E402


def _arm(budget=31):
    ng = NgramDrafter(corpus_path="", min_order=3)
    head = FakeDrafter(FakeEng(), 16)
    r = MergedRouter(ng, head, mtp_depth=15, node_budget=budget, mtp_ms_per_token=0.0,
                     head_fixed_ms=15.0, adaptive_depth=False, verify_ms_table=dict(STAIR_MS),
                     tree_ms_table=dict(STAIR_MS))
    r.stair, r.stair_table = True, dict(STAIR_MS)
    return r


def _bush(width, depth, p_first, p_rest):
    """`width` branches under the anchor, each a chain `depth` deep. Branch 0's first node has
    probability `p_first`, the others `p_rest`; each deeper node keeps 0.8 of its parent's."""
    tokens, parents, scores, source = [0], [-1], [1.0], ["root"]
    for b in range(width):
        parent, p = 0, (p_first if b == 0 else p_rest)
        for d in range(depth):
            tokens.append(100 * b + d + 1)
            parents.append(parent)
            scores.append(p)
            source.append("df2")
            parent = len(tokens) - 1
            p *= 0.8
    return DraftTree(tokens, parents, scores, source)


def test_a_confident_line_is_cut_below_the_second_tile():
    """16 rows cost 78.9 ms and 17 cost 83.2: the 16th node has to carry 4.3 ms of verify."""
    arm = _arm()
    tree = _bush(width=4, depth=8, p_first=0.9, p_rest=0.05)
    cut, v = arm._stair_cut(tree, 15.0)
    assert cut.n_draft <= 15, cut.n_draft
    assert v > 0


def test_weak_far_nodes_are_dropped_and_strong_ones_kept():
    arm = _arm()
    weak = _bush(width=4, depth=7, p_first=0.6, p_rest=0.01)
    strong = _bush(width=4, depth=7, p_first=0.99, p_rest=0.6)
    cw, _ = arm._stair_cut(weak, 15.0)
    cs, _ = arm._stair_cut(strong, 15.0)
    assert cw.n_draft < cs.n_draft, (cw.n_draft, cs.n_draft)
    assert cs.n_draft > 16, "nodes worth the second tile are bought"


def test_the_cut_never_exceeds_the_budget_and_keeps_ancestors():
    arm = _arm(budget=10)
    tree = _bush(width=5, depth=6, p_first=0.99, p_rest=0.9)
    cut, _ = arm._stair_cut(tree, 15.0)
    assert cut.n_draft <= 10
    cut.check()


def test_the_cut_is_off_unless_asked():
    arm = _arm()
    arm.stair = False
    assert arm.stair is False


# --- the wide mode ------------------------------------------------------------------------------

class StairArm:
    wants_rows = True

    def __init__(self, head, budget):
        self.mtp = head
        self.node_budget = self.head_budget = budget
        self.stair = False
        self.stair_table = None
        self.head_fixed_ms = 27.0

    def propose_tree(self, context, k):
        n = self.node_budget
        return DraftTree([context[-1]] + [9000 + i for i in range(n)], [-1] + list(range(n)))

    def observe(self, tokens):
        pass

    def reset(self):
        pass

    def sync(self, tokens, hidden, first_pos, rows=None):
        self.mtp.sync(tokens, hidden, first_pos, rows=rows)

    def state_snapshot(self):
        return self.mtp.state_snapshot()


def _wide_router(**kw):
    eng = FakeEng()
    small, large = FakeDrafter(eng, 8), FakeDrafter(eng, 16)
    s, l = StairArm(small, 15), StairArm(large, 23)
    r = LengthRouter(s, l, tree=True, latch=True, drop_idle=True, learn_cost=False,
                     switch=True, switch_mode="wide", tree_wide_after=32, **kw)
    return r, s, l, small, large


def test_the_wide_mode_configures_both_arms_for_the_cut():
    r, s, l, _, _ = _wide_router()
    assert s.stair and l.stair and l.node_budget == 31 and l.head_fixed_ms == 15.0
    assert l.stair_table == STAIR_MS


def test_the_wide_mode_drafts_wide_and_releases_the_narrow_arm():
    r, s, l, small, large = _wide_router()
    for b in range(10):
        r.propose_tree(list(range(50 + b)), 31)
        assert r.last_key == "l"
        r.observe([1, 2, 3])
    assert r.idle == "s" and small.released and not large.released
    assert l.node_budget == 31, "the served budget override does not touch the cut's cap"
    snap = r.state_snapshot()
    assert snap[1] is None and snap[3] == "s"
    r.reset()
    assert r.idle is None
    r.propose_tree(list(range(80)), 31)
    assert r.idle == "s", "released again on the next request"


def test_the_mode_comes_from_the_environment():
    old = os.environ.pop("QWEN38_LEN_MODE", None)
    try:
        os.environ["QWEN38_LEN_MODE"] = "wide"
        eng = FakeEng()
        r = LengthRouter(StairArm(FakeDrafter(eng, 8), 15), StairArm(FakeDrafter(eng, 16), 23),
                         tree=True, latch=True, learn_cost=False, switch=True)
        assert r.switch_mode == "wide"
        os.environ["QWEN38_LEN_MODE"] = "nonsense"
        try:
            LengthRouter(StairArm(FakeDrafter(eng, 8), 15), StairArm(FakeDrafter(eng, 16), 23),
                         tree=True, latch=True, learn_cost=False, switch=True)
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown mode was accepted")
    finally:
        os.environ.pop("QWEN38_LEN_MODE", None)
        if old is not None:
            os.environ["QWEN38_LEN_MODE"] = old


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except Exception:
                fails += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    print(f"\n{fails} failed")
    sys.exit(1 if fails else 0)
