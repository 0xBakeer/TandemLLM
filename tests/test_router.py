"""The router's arithmetic, with a stand-in for the prediction head.

The router is the part of this track that runs on the board, so what matters is not that it picks
something but that what it picks follows the measured cost curve. Each test here fixes one drafter's
behaviour and checks the choice the curve implies.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.drafters.ngram import NgramDrafter  # noqa: E402
from engine.router import MergedRouter, verify_ms  # noqa: E402


class FakeHead:
    """A prediction head that always proposes the same chain, and counts how often it is asked."""

    name = "fake"

    def __init__(self, chain=(901, 902, 903)):
        self.chain = list(chain)
        self.calls = 0

    def propose(self, context, k):
        self.calls += 1
        return self.chain[:k]

    def observe(self, tokens):
        pass

    def reset(self):
        pass

    def sync(self, *args):
        pass


def _router(head=None, **kw):
    ng = NgramDrafter(corpus_path="", min_order=kw.pop("min_order", 3),
                      min_expected=kw.pop("min_expected", 0.5),
                      alpha=kw.pop("alpha", 0.2))
    return MergedRouter(ng, head or FakeHead(), **kw), ng


def test_verify_curve_is_monotone_and_matches_the_measured_points():
    assert verify_ms(1) == 151.0
    assert verify_ms(32) == 209.8
    assert verify_ms(8) < verify_ms(16) < verify_ms(32)
    # between measured points it interpolates rather than jumping
    assert 169.7 < verify_ms(10) < 179.4


def test_the_head_carries_a_context_the_lookup_drafter_knows_nothing_about():
    r, _ = _router()
    r.prime([1, 2, 3, 4, 5, 6, 7, 8])
    draft = r.propose([1, 2, 3, 4, 5, 6, 7, 8], 16)
    assert draft == [901, 902, 903], "with no match, the head is the only option"
    assert r.last == "mtp"


def test_a_long_exact_repeat_beats_the_head():
    seq = list(range(200, 260)) * 3
    r, _ = _router()
    r.prime(seq)
    draft = r.propose(seq, 16)
    assert draft and draft[0] != 901, "a certain lookup should win against a 3-token neural draft"
    assert r.last == "ngram"


def test_an_expensive_head_shifts_the_choice_to_the_lookup_drafter():
    """The head costs bytes; the lookup drafter costs nothing. Price decides, not preference."""
    seq = [7, 7, 8, 9] * 12 + [5] * 40
    cheap, _ = _router(mtp_ms_per_token=0.5, min_order=2)
    dear, _ = _router(mtp_ms_per_token=60.0, min_order=2)
    for r in (cheap, dear):
        r.prime(seq)
    ctx = seq + [7, 7]
    cheap.propose(ctx, 16)
    dear.propose(ctx, 16)
    assert dear.last == "ngram", "at 60 ms per drafted token the head has to be much better"


def test_calibration_falls_when_the_lookup_drafter_is_optimistic():
    seq = list(range(300, 340)) * 2
    r, _ = _router()
    r.prime(seq)
    before = r.calib.value
    for _ in range(10):
        r.propose(seq, 16)
        r.last_expected = 8.0                      # it claimed eight
        r.observe([-9, -8])                        # the target wrote something else entirely
    assert r.calib.value < before
    assert r.calib.value > 0.0


def test_calibration_learns_from_the_block_it_did_not_write():
    """The counterfactual is the whole point: a policy that only learns about what it chose
    will keep choosing it. On a quote workload the drafter alone beat the block drafter by
    42 % and the router picked it zero times out of nineteen blocks."""
    seq = list(range(600, 660)) * 3
    # alpha 1.5 makes the drafter's own estimate pessimistic, which is the case that matters:
    # it will not be chosen and it has to find out it was wrong anyway
    r, _ = _router(head=FakeHead(chain=[1, 2, 3]), alpha=1.5, min_expected=0.1)
    r.prime(seq)
    before = r.calib.value
    for _ in range(12):
        r.propose(seq, 16)
        r.last = "mtp"                    # pretend the head won the price every time
        r.observe(list(seq[:6]))          # and the lookup drafter would have been right anyway
    assert r.calib.value > before, "it must learn it was too modest without being chosen"


def test_the_tree_mode_merges_instead_of_choosing():
    seq = list(range(400, 460)) * 3
    head = FakeHead(chain=[777, 778])
    r, _ = _router(head=head)
    r.prime(seq)
    tree = r.propose_tree(seq, 16)
    assert tree is not None
    tree.check()
    sources = set(tree.source[1:])
    assert "mtp" in sources and "ngram" in sources, "both drafters belong in one block"
    assert tree.accepted_against([777, 778]) == 2, "the head's chain survives the merge"
    assert r.last == "merged"


def test_tree_mode_falls_back_to_the_head_alone_when_there_is_no_match():
    r, _ = _router()
    r.prime([1, 2, 3, 4, 5, 6, 7, 8])
    tree = r.propose_tree([1, 2, 3, 4, 5, 6, 7, 8], 16)
    assert tree is not None
    assert [tree.tokens[i] for i in range(1, len(tree))] == [901, 902, 903]


def test_a_declining_head_and_a_declining_lookup_produce_no_block():
    head = FakeHead(chain=[])
    r, _ = _router(head=head)
    r.prime([1, 2, 3, 4, 5, 6, 7, 8])
    assert r.propose([1, 2, 3, 4, 5, 6, 7, 8], 16) == []
    assert r.propose_tree([1, 2, 3, 4, 5, 6, 7, 8], 16) is None


def test_the_router_never_proposes_more_than_it_was_asked_for():
    seq = list(range(500, 560)) * 3
    r, _ = _router()
    r.prime(seq)
    for k in (1, 2, 4, 8, 16):
        assert len(r.propose(seq, k)) <= k


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()
