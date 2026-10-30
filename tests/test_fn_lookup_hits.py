"""SPD-249: the offline draft-headroom counters (CPU, no weights).

  1  tools/fn_lookup_hits.py scan: a completion that repeats its prompt is found by the request's own index,
     every hit counted where the best candidate matches; a fresh text never fires
                                                        test_lookup_hits_on_a_repeat
  2  scan with MTP chains: beyond_mtp counts the positions the lookup matches 2+ and the chain fewer, and the
     rounds advance by 1 + the chain's accepted tokens  test_lookup_hits_against_mtp
  3  tools/fn_mtp_accept.py score: accept per depth, top-k coverage on the path and the first miss's
     top-k share                                        test_mtp_topk_coverage

Run: python tests/test_fn_lookup_hits.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.drafters.ngram import NgramDrafter  # noqa: E402
from tools import fn_lookup_hits as H  # noqa: E402


def _drafter():
    return NgramDrafter(corpus_path="", min_order=3, max_depth=16, min_corpus_order=8)


def test_lookup_hits_on_a_repeat():
    seq = list(range(100, 140))
    st = H.scan(_drafter(), seq, seq)                 # the completion is the prompt again
    s = H.summary(st, False)
    assert st["positions"] == 38
    # from the 3rd completion token on (order 3 needs 3 tokens of context) every position has the prompt's match
    assert st["fired"] >= 36, st
    assert st["hit_top"][2] == st["fired"], st
    assert s["source"]["local"] == 1.0
    st2 = H.scan(_drafter(), list(range(1, 30)), list(range(500, 540)))
    assert st2["fired"] == 0 and st2["hit_top"][2] == 0


def test_lookup_hits_against_mtp():
    seq = list(range(100, 120))
    comp = seq[:]
    good = [comp[i + 1: i + 3] for i in range(len(comp))]          # the chain right at depth 1 and 2
    bad = [[7, 7] for _ in comp]                                     # the chain always wrong
    st_good = H.scan(_drafter(), seq, comp, good, mtp_depth=2)
    st_bad = H.scan(_drafter(), seq, comp, bad, mtp_depth=2)
    assert st_good["beyond_mtp"] == 0
    assert st_bad["beyond_mtp"] == st_bad["hit_top"][2] > 0
    # a good chain accepts 2 a round: rounds every 3rd position
    assert st_good["rounds"] == (st_good["positions"] + 2) // 3
    assert st_bad["rounds"] == st_bad["positions"]


def test_mtp_topk_coverage():
    try:
        from tools.fn_mtp_accept import score
    except Exception as e:  # noqa: BLE001  (a CPU venv without the engine's GPU modules)
        print(f"skip test_mtp_topk_coverage: {e!r}")
        return
    toks = [0, 1, 2, 3, 4, 5]
    drafts = [[1, 9], [2, 3], [9, 4], [4, 5]]
    tops = [[[1, 5], [9, 2]], [[2, 6], [3, 7]], [[9, 3], [4, 8]], [[4, 0], [5, 0]]]
    r = score(drafts, tops, toks, 2, (1, 2), hot=[1, 2, 3])
    assert r["rounds"] == 4
    assert r["accept_at_depth"] == [0.75, 0.5]
    # depth 1 on the path in every round: top-1 3/4, top-2 4/4; depth 2 on the path in the 3 rounds right at 1
    assert r["on_path"] == [4, 3]
    assert r["coverage"][1] == [0.75, round(2 / 3, 4)]
    assert r["coverage"][2] == [1.0, 1.0]
    # first misses: round 0 at depth 2 (truth 2 in top-2), round 2 at depth 1 (truth 3 in top-2)
    assert r["miss_in_topk"][2] == [0.25, 0.25]
    assert r["miss_in_topk"][1] == [0.0, 0.0]
    assert r["hot_share"] == 0.6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
