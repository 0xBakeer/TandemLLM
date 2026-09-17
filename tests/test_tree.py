"""The draft-tree invariants, which the GDN verify path will depend on and cannot check itself.

Run: `python -m pytest tests/ -q`, or `python tests/test_tree.py` with no pytest installed.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.tree import DraftTree, TreeBuilder  # noqa: E402


def test_chain_is_a_tree():
    t = DraftTree.chain(5, [1, 2, 3])
    t.check()
    assert t.n_draft == 3
    assert t.depths() == [0, 1, 2, 3]
    assert t.leaves() == [3]


def test_builder_emits_dfs_preorder():
    b = TreeBuilder(9)
    a = b.add(0, 1, 0.9)
    b.add(a, 11, 0.8)
    c = b.add(0, 2, 0.5)
    b.add(c, 22, 0.4)
    t = b.build()
    t.check()
    # depth-first: the whole subtree of the first child comes before the second child
    assert t.tokens == [9, 1, 11, 2, 22]
    assert t.parents == [-1, 0, 1, 0, 3]


def test_shared_prefixes_collapse():
    t = DraftTree.from_sequences(7, [([1, 2, 3], 0.6), ([1, 2, 9], 0.3), ([4], 0.1)])
    t.check()
    # 1 and 2 are shared, so five nodes rather than seven
    assert t.n_draft == 5
    assert t.tokens.count(1) == 1


def test_ancestor_mask_is_lower_triangular():
    t = DraftTree.from_sequences(0, [([1, 2, 3], 1.0), ([1, 5], 0.5), ([8, 9], 0.4)])
    t.check()
    m = t.ancestor_mask()
    n = len(t)
    for i in range(n):
        assert m[i][i]
        for j in range(i + 1, n):
            assert not m[i][j], "a node may never attend to a later one"
    # every node attends to exactly its own path
    for i in range(n):
        assert [j for j in range(n) if m[i][j]] == sorted(t.path(i))


def test_accepted_against_picks_the_matching_branch():
    t = DraftTree.from_sequences(0, [([1, 2, 3], 0.6), ([1, 5, 6], 0.3)])
    assert t.accepted_against([1, 5, 6, 7]) == 3
    assert t.accepted_against([1, 2, 9]) == 2
    assert t.accepted_against([4]) == 0
    assert t.accepted_against([]) == 0


def test_expected_accepted_is_the_sum_of_path_probabilities():
    t = DraftTree(tokens=[0, 1, 2, 3], parents=[-1, 0, 1, 0], scores=[1.0, 0.7, 0.5, 0.3])
    assert abs(t.expected_accepted() - 1.5) < 1e-9


def test_prune_keeps_ancestors_and_respects_the_budget():
    seqs = [([i, i + 100, i + 200], 1.0 / (i + 1)) for i in range(10)]
    t = DraftTree.from_sequences(0, seqs)
    p = t.prune(budget=6)
    p.check()
    assert p.n_draft <= 6
    # anything kept has its whole path kept, which `check` proves structurally
    assert p.expected_accepted() <= t.expected_accepted() + 1e-9


def test_prune_refuses_nodes_that_do_not_pay():
    # one strong branch and a long tail of near-zero candidates
    seqs = [([1, 2, 3, 4], 100.0)] + [([9, 9, i], 0.001) for i in range(20)]
    t = DraftTree.from_sequences(0, seqs)
    p = t.prune(budget=48)
    p.check()
    assert p.n_draft < t.n_draft, "the tail should not survive the marginal-value test"
    assert p.accepted_against([1, 2, 3, 4]) == 4


def test_merge_shares_common_nodes():
    a = DraftTree.from_sequences(0, [([1, 2, 3], 0.9)], source="ngram")
    b = DraftTree.chain(0, [1, 2, 7], source="mtp")
    m = a.merge(b)
    m.check()
    # 1 and 2 are shared: five nodes, not six
    assert m.n_draft == 4
    assert m.accepted_against([1, 2, 7]) == 3
    assert m.accepted_against([1, 2, 3]) == 3


def test_merge_requires_the_same_anchor():
    try:
        DraftTree.chain(1, [2]).merge(DraftTree.chain(3, [4]))
    except AssertionError:
        return
    raise AssertionError("merging trees with different anchors must fail")


def test_value_follows_the_measured_cost_curve():
    t = DraftTree.chain(0, [1, 2, 3, 4, 5, 6, 7, 8])
    # eight nodes all certain: 9 tokens for 149.1 + 8 * 1.896 ms
    expected = 9.0 / ((149.1 + 8 * 1.896) / 1000.0)
    assert abs(t.value() - expected) < 1e-6


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()
