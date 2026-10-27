"""SPD-225: the round runner's priced cut (server/slots.py `priced_cut`, `RowPrices`): the joint round keeps
the rows where its expected tokens a ms are the most, on the pass prices learned from its own rounds."""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.tree import DraftTree  # noqa: E402
from server import slots  # noqa: E402


def _chain(scores):
    """An anchor and a chain of drafts with these path probabilities."""
    n = len(scores) + 1
    t = DraftTree([0] * n, [-1] + list(range(n - 1)))
    t.scores = [1.0] + list(scores)
    return t


def _prices(pass_ms: dict, lanes: int, rest: float = 10.0):
    pr = slots.RowPrices(alpha=1.0)
    for rows, ms in pass_ms.items():
        pr.note(lanes, rows, ms, ms + rest)
    return pr


def test_cost_interpolates_and_never_guesses_a_cheaper_pass_below_what_was_measured():
    pr = _prices({16: 40.0, 32: 80.0}, 8)
    assert pr.cost(8, 16) == 40.0 and pr.cost(8, 24) == 60.0
    assert pr.cost(8, 40) == 100.0                     # the line of the two largest
    assert pr.cost(8, 8) == 40.0                       # below the smallest: its own ms
    assert pr.cost(4, 16) is None                      # other lanes: not priced
    assert pr.rest(8) == 10.0
    return "a line between and past the measured rows; the smallest's ms below them; none at other lanes"


def test_a_pass_that_grows_with_its_rows_keeps_only_the_drafts_that_pay():
    # 8 lanes, each a chain of 3 drafts at 0.9 / 0.5 / 0.1: a pass of 8 rows 20 ms, every 8 more rows +20 ms
    pr = _prices({8: 20.0, 16: 40.0, 24: 60.0, 32: 80.0}, 8)
    trees = [_chain([0.9, 0.5, 0.1]) for _ in range(8)]
    keeps, exp, offered = slots.priced_cut(trees, pr, 32)
    # 8 rows: 8 / 30 = 0.27 a ms; 16: 15.2 / 50 = 0.30; 24: 19.2 / 70 = 0.27 -> the first draft of each lane
    assert [len(k) for k in keeps] == [2] * 8, keeps
    assert abs(exp - 8 * 1.9) < 1e-9 and offered == 32
    return "steep prices: one draft a lane kept (16 rows of 32)"


def test_a_flat_pass_keeps_every_row_and_the_cut_is_the_joint_cut():
    pr = _prices({8: 50.0, 16: 50.5, 24: 51.0, 32: 51.5}, 8)
    rng = random.Random(7)
    for _ in range(30):
        trees = [_chain(sorted((rng.random() for _ in range(rng.randint(1, 5))), reverse=True)) for _ in range(8)]
        keeps, _, _ = slots.priced_cut(trees, pr, 32)
        total = sum(len(t.tokens) for t in trees)
        assert sum(len(k) for k in keeps) >= min(32, total) - 4
        for t, k in zip(trees, keeps):
            assert 0 in k and all(t.parents[i] in k for i in k if i)
    return "flat prices: (nearly) every row the joint cut would keep, ancestor-closed"


def test_a_budget_keeps_the_longest_prefix_within_it_and_unpriced_lanes_return_none():
    pr = slots.RowPrices()
    trees = [_chain([0.9, 0.8, 0.7]) for _ in range(4)]
    assert slots.priced_cut(trees, pr, 16) is None
    keeps, _, _ = slots.priced_cut(trees, pr, 16, budget=9)
    assert sum(len(k) for k in keeps) <= 9 and all(0 in k for k in keeps)
    assert sum(len(k) for k in keeps) == 9             # the anchors, the four 0.9 drafts, one 0.8
    return "unpriced: None; an exploring budget: the best prefix within it"


def test_the_calibration_scales_the_scores_by_what_the_walks_accepted():
    pr = slots.RowPrices(alpha=1.0)
    pr.note_accept(3.0, 6.0)
    assert pr.cal == 0.5
    pr.note_accept(30.0, 1.0)
    assert pr.cal == 2.0                               # bounded
    return "cal = accepted / expected (alpha 1), at most 2"


def test_a_family_without_lone_anchors_keeps_each_lanes_best_draft():
    pr = _prices({8: 20.0, 16: 40.0, 24: 60.0, 32: 80.0}, 8)
    trees = [_chain([0.3, 0.1]) for _ in range(8)]
    keeps, _, _ = slots.priced_cut(trees, pr, 32)
    assert [len(k) for k in keeps] == [1] * 8, keeps        # 0.3 a draft does not pay a row
    keeps, _, _ = slots.priced_cut(trees, pr, 32, anchor_alone=False)
    assert [len(k) for k in keeps] == [2] * 8, keeps
    return "weak drafts on steep prices: anchors alone; a family that needs a draft keeps its best one"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
