"""The suffix memory: does it find what is there, decline what is not, and stay cheap.

Token ids here are arbitrary small integers; nothing in the drafter knows or cares what they mean.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.drafters.engram import EngramDrafter  # noqa: E402
from engine.drafters.ngram import (CorpusSuffixStore, LocalSuffixIndex,  # noqa: E402
                                   NgramDrafter)


def test_local_index_finds_the_longest_match():
    ix = LocalSuffixIndex()
    ix.prime([1, 2, 3, 4, 5, 1, 2, 3, 9, 9])
    n, pos = ix.lookup([1, 2, 3], min_order=2)
    assert n == 3
    assert sorted(ix.continuation(p, 1)[0] for p in pos) == [4, 9]


def test_local_index_positions_are_capped():
    ix = LocalSuffixIndex(orders=(3,), max_positions=4)
    ix.prime([7, 7, 7] * 20)
    _, pos = ix.lookup([7, 7, 7], min_order=3)
    assert len(pos) <= 4


def test_drafter_declines_on_a_context_it_has_never_seen():
    d = NgramDrafter(corpus_path="")
    d.prime([1, 2, 3, 4, 5, 6, 7, 8])
    assert d.propose_tree([1, 2, 3, 4, 5, 6, 7, 8]) is None
    assert d.propose([1, 2, 3, 4, 5, 6, 7, 8], 8) == []


def test_drafter_proposes_a_repeat():
    seq = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] * 3
    d = NgramDrafter(corpus_path="", min_expected=0.3)
    d.prime(seq)
    tree = d.propose_tree(seq, depth=8)
    assert tree is not None
    tree.check()
    # the sequence is perfectly periodic, so the continuation is known exactly
    assert tree.accepted_against([10, 11, 12, 13]) == 4


def test_ambiguity_becomes_branches_not_a_wrong_chain():
    # "a b" is followed by c twice and by d once
    seq = []
    for nxt in (30, 30, 31):
        seq += [1, 2, nxt, 99, 98, 97, 96, 95]
    d = NgramDrafter(corpus_path="", min_order=2, min_expected=0.1, branch_top_k=3)
    d.prime(seq)
    tree = d.propose_tree(seq[:-6] + [1, 2], depth=4)
    assert tree is not None
    first = [tree.tokens[c] for c in tree.children_of(0)]
    assert set(first) >= {30, 31}, "both observed continuations belong in the tree"
    assert tree.accepted_against([30]) == 1
    assert tree.accepted_against([31]) == 1


def test_scores_are_path_probabilities():
    seq = ([1, 2, 40, 50] + [77] * 6) * 4 + [1, 2, 40, 60] + [77] * 6
    d = NgramDrafter(corpus_path="", min_order=2, min_expected=0.1)
    d.prime(seq)
    tree = d.propose_tree(seq + [1, 2], depth=3)
    assert tree is not None
    for i in range(1, len(tree)):
        assert 0.0 < tree.scores[i] <= 1.0
        assert tree.scores[i] <= tree.scores[tree.parents[i]] + 1e-9


def test_expected_accepted_tracks_reality_on_a_repetitive_stream():
    rng = random.Random(7)
    vocab = [rng.randrange(1000) for _ in range(40)]
    stream = []
    for _ in range(60):
        start = rng.randrange(0, 30)
        stream += vocab[start:start + 8]
    d = NgramDrafter(corpus_path="", min_expected=0.3)
    d.prime(stream[:200])
    got, want = 0.0, 0
    fired = 0
    for i in range(200, len(stream) - 16):
        tree = d.propose_tree(stream[:i], depth=8)
        if tree is not None:
            fired += 1
            got += tree.expected_accepted()
            want += tree.accepted_against(stream[i:i + 16])
        d.observe([stream[i]])
    assert fired > 0
    # the estimate should be the right order of magnitude, not exact
    assert 0.4 < (got / max(want, 1)) < 2.5, f"expected {got:.1f} vs actual {want}"


def test_corpus_store_round_trip():
    import numpy as np
    from tools.build_corpus import build_suffix_array
    toks = np.array([5, 6, 7, 5, 6, 8, 5, 6, 7, 9], dtype=np.int32)
    sa = build_suffix_array(toks, max_order=4)
    store = CorpusSuffixStore(toks, sa, max_order=4)
    n, pos = store.lookup([1, 1, 5, 6, 7], min_order=2)
    assert n == 3
    assert sorted(store.continuation(p, 1)[0] for p in pos) == [5, 9]
    n2, pos2 = store.lookup([4, 4, 4], min_order=2)
    assert n2 == 0 and pos2 == []


def test_corpus_store_loads_from_disk():
    import numpy as np
    from tools.build_corpus import build_suffix_array, write_store
    with tempfile.TemporaryDirectory() as d:
        toks = np.array([1, 2, 3, 1, 2, 4, 1, 2, 3, 5], dtype=np.int32)
        write_store(d, toks, build_suffix_array(toks, 4), {"max_order": 4, "sources": []})
        store = CorpusSuffixStore.load(d)
        assert store is not None and store.n == 10
        n, pos = store.lookup([9, 1, 2], min_order=2)
        assert n == 2 and len(pos) == 3
        assert CorpusSuffixStore.load(os.path.join(d, "nope")) is None


def test_corpus_raises_coverage_over_the_sequence_alone():
    import numpy as np
    from tools.build_corpus import build_suffix_array
    rng = random.Random(3)
    corpus = [rng.randrange(500) for _ in range(4000)]
    store = CorpusSuffixStore(np.array(corpus, dtype=np.int32),
                              build_suffix_array(np.array(corpus, dtype=np.int32), 8),
                              max_order=8)
    # a fresh sequence that happens to quote the corpus
    quote = corpus[1000:1060]
    local_only = NgramDrafter(corpus_path="", min_expected=0.2, min_corpus_order=4)
    with_corpus = NgramDrafter(corpus_path="", min_expected=0.2, min_corpus_order=4)
    with_corpus.corpus = store
    hits = [0, 0]
    for d, slot in ((local_only, 0), (with_corpus, 1)):
        d.prime(quote[:10])
        for i in range(10, 40):
            if d.propose_tree(quote[:i], depth=8) is not None:
                hits[slot] += 1
            d.observe([quote[i]])
    assert hits[1] > hits[0], f"corpus should fire more often: {hits}"


def test_v2_never_costs_more_per_call_than_a_verify_step():
    rng = random.Random(11)
    stream = [rng.randrange(2000) for _ in range(4000)]
    d = NgramDrafter(corpus_path="")
    d.prime(stream)
    t0 = time.perf_counter()
    for i in range(500):
        d.propose_tree(stream[:3000 + i], depth=16)
    per_call_ms = (time.perf_counter() - t0) / 500 * 1000
    # the whole point of a lookup drafter is that it is free against a 151 ms verify
    assert per_call_ms < 5.0, f"{per_call_ms:.2f} ms per proposal is not free"


def test_v1_and_v2_agree_where_v1_is_confident():
    """v2 must not lose what v1 found: a unique long match gives the same chain."""
    seq = list(range(100, 160)) + list(range(100, 130))
    v1 = EngramDrafter()
    v1.prime(seq)
    v2 = NgramDrafter(corpus_path="", min_expected=0.3)
    v2.prime(seq)
    a = v1.propose(seq, 8)
    b = v2.propose(seq, 8)
    assert a and b
    assert a[:4] == b[:4]


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()
