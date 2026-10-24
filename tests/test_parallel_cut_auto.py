"""SPD-155: `--parallel-cut auto` cuts the trees together from `auto_n` decoding slots on, and the
round never runs a joint group over the row cap uncut."""

from __future__ import annotations

from server import slots


class R:
    def __init__(self, rows, tree=object(), batchable=True):
        self.rows, self.tree, self.batchable = rows, tree, batchable


def _meet(cut, n=3):
    m = slots.Meeting(32, 0.002, cut=cut)
    m.auto_n = n
    return m


def test_joint_for():
    assert not _meet("share").joint_for(8)
    assert _meet("joint").joint_for(2)
    m = _meet("auto", 3)
    assert not m.joint_for(2) and m.joint_for(3) and m.joint_for(8)


def test_auto_below_n_packs_as_share():
    m = _meet("auto", 3)
    m.pending = {0: R(20), 1: R(20)}
    assert [len(g) for g in m.batches()] == [1, 1]


def test_auto_from_n_puts_every_tree_in_one_group():
    m = _meet("auto", 3)
    m.pending = {i: R(12) for i in range(4)}
    g = m.batches()
    assert len(g) == 1 and len(g[0]) == 4


def test_a_block_without_its_tree_is_packed_beside_the_joint_group():
    m = _meet("auto", 3)
    m.pending = {0: R(10), 1: R(10), 2: R(10), 3: R(8, tree=None)}
    g = m.batches()
    assert sorted(len(x) for x in g) == [1, 3]


def test_an_unbatchable_block_stays_alone():
    m = _meet("joint")
    m.pending = {0: R(4), 1: R(4), 2: R(1, batchable=False)}
    g = m.batches()
    assert sorted(len(x) for x in g) == [1, 2]


def test_the_27b_family_takes_64_rows_on_exl3_and_32_otherwise():
    """SPD-159: the row cap is the family's, by the weights it serves; no other family has a policy."""
    from engine.models.qwen3_5 import adapter

    class W:
        def __init__(self, src):
            self.exl3_source = src

    class E:
        def __init__(self, src):
            self.w = W(src)
    from engine.models.qwen3_5 import joint_graphs as JG
    # SPD-163: with the joint graphs on (the default on a GPU since SPD-201) a cut keeps each tree in the graphs' bucket
    graphs = {"tree_cap": JG.t_pad} if JG.ON else {}
    assert adapter.joint_policy(E("/packs/exl3")) == {"max_rows": 64, "cut": "auto", **graphs}
    assert adapter.joint_policy(E(None)) == {"max_rows": 32, "cut": "auto", **graphs}
    assert adapter.Qwen38Engine.joint_policy is adapter.joint_policy



def test_spd163_the_joint_cut_keeps_each_tree_within_a_cap():
    """A family's tree cap (`joint_policy` "tree_cap", the joint graphs' row bucket): no tree keeps more
    than the cap, every kept set stays ancestor-closed, and without a cap the cut is what it was."""
    from engine.tree import DraftTree
    import random
    rng = random.Random(3)

    def tree(n):
        parents, path = [-1], [0]
        for i in range(1, n):
            path = path[:rng.randint(1, len(path))]
            parents.append(path[-1])
            path.append(i)
        t = DraftTree([0] * n, parents)
        t.scores = [0.0] + [rng.random() for _ in range(n - 1)]
        return t
    for _ in range(50):
        trees = [tree(rng.randint(2, 30)) for _ in range(rng.randint(2, 8))]
        cap = rng.choice((4, 8, 12, 16))
        keeps = slots.joint_cut(trees, 64, cap)
        assert all(len(k) <= cap for k in keeps), (cap, [len(k) for k in keeps])
        assert sum(len(k) for k in keeps) <= 64
        for t, k in zip(trees, keeps):
            assert 0 in k and all(t.parents[i] in k for i in k if i)
        assert slots.joint_cut(trees, 64) == slots.joint_cut(trees, 64, None)
    from engine.models.qwen3_5 import joint_graphs as JG
    assert [JG.t_pad(e, 64) for e in (2, 3, 4, 5, 8)] == [16, 16, 16, 12, 8]
    assert JG.t_pad(8, 32) == 4


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
