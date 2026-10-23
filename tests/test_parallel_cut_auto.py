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
