"""SRV-87: the host side of Flash-Next in the server loop (engine/models/flashnext/serving.py and the
shared MTP tree drafter, engine/methods/mtp_drafter.py), no GPU.

  * the drafter pairs every synced row with the token after it -- the next synced row's token, or for
    the last row the anchor the proposal names -- across prompt chunks, a forced phrase, verified paths
    and single decode steps; the backlog past depth + 1 rows is primed in one call, in position order,
    and the last rows ride the draft's own feed;
  * the walk down a block (the shared engine/methods/walk.py) follows the target's picks and adds the
    target's token at the stop;
  * `kv.length` reads the state's length after a pending chain commit; zero resets; another length
    keeps the rows by length and clamps the pooled keys;
  * a forward that does not start at the state's length is refused;
  * the adapter's cold verbs: `cost()` from the state's sections, `settle()` lands a pending chain and
    the parked GDN rows, `discard_pending()` drops them, `trim_pending(n)` cuts a pending chain and
    refuses a cut inside rows the state absorbed.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.methods import walk  # noqa: E402
from engine.models.flashnext.ref import FNState  # noqa: E402
from engine.methods.mtp_drafter import MTPTreeDrafter  # noqa: E402
from engine.methods.staircut import PriceModel, StairCut  # noqa: E402
from engine.models.flashnext.serving import FlashNextServing  # noqa: E402
from engine.tree import DraftTree  # noqa: E402


class _Cfg:
    attention_layers = (3,)
    num_key_value_heads, head_dim, indexer_head_dim, has_indexer = 2, 8, 4, True


class _FN:
    """What the adapters read of FlashNextEngine, on the host."""

    def __init__(self):
        self.cfg = _Cfg()
        self.primed: list = []
        self.resets = 0
        self.commits: list = []
        self._pend_owner = None
        self._pend_n = torch.ones(2, dtype=torch.int32)
        self.flushes = 0

    def flush(self):
        self.flushes += 1
        self._pend_n.zero_()
        self._pend_owner = None

    def new_state(self, cap=None):
        st = FNState(cap=cap or 64)
        st.S[0] = torch.zeros(1, 2, 4, 4)
        st.conv[0] = torch.zeros(1, 3, 3)
        return st

    def new_mtp_state(self, cap=None):
        return FNState(cap=cap or 64)

    def mtp_prime(self, stack, nexts, st_m):
        self.primed.append((stack[:, 0].tolist(), list(nexts), st_m.length))
        st_m.length += len(nexts)

    def reset(self, st):
        self.resets += 1
        st.reset_inplace()

    def commit(self, st, path, graphs=None):
        self.commits.append(list(path))
        st.length += len(path)


def _rows(pos, n):
    """Streams whose first value is the row's position (so a test reads where a row went)."""
    return torch.arange(pos, pos + n, dtype=torch.float32)[:, None].repeat(1, 3)


def test_drafter_pairs_rows_with_their_next_token():
    fn = _FN()
    s = FlashNextServing(fn, 64)
    d = MTPTreeDrafter(s, s.add_mtp_head(2), StairCut(PriceModel()))
    assert s.draft_heads()["mtp"] is d.head
    prompt = list(range(100, 110))
    d.sync(prompt[:6], _rows(0, 6), 0)                 # a prompt in two chunks
    d.sync(prompt[6:], _rows(6, 4), 6)
    assert d._feed(500)                                # the first token after the prompt
    # 10 rows: 7 primed at once in position order, the last 3 (depth + 1) for the draft's feed
    assert fn.primed == [([0.0, 1, 2, 3, 4, 5, 6], prompt[1:8], 0)]
    rows, nexts = d.pending
    assert rows[:, 0].tolist() == [7, 8, 9] and nexts == prompt[8:] + [500]
    d.pending = None
    d.head.st.length += 3                               # what the draft's own feed does
    # a verified path (anchor 500 and two drafts), then a forced phrase, then one decode step
    d.sync([500, 501, 502], _rows(10, 3), 10)
    d.sync([503, 504], _rows(13, 2), 13)
    d.sync([505], _rows(15, 1), 15)
    assert d._feed(506)
    assert fn.primed[-1] == ([10.0, 11, 12], [501, 502, 503], 10)
    rows, nexts = d.pending
    assert rows[:, 0].tolist() == [13, 14, 15] and nexts == [504, 505, 506]
    d.reset()
    assert d.head.st.length == 0 and d.held is None and d.queue == [] and not d._feed(1)
    return "prompt chunks, forced phrase, decode steps: each row with its next token; backlog primed in order"


def test_walk_and_kv_length():
    fn = _FN()
    s = FlashNextServing(fn, 64)
    tree = DraftTree(tokens=[7, 8, 9, 4, 5], parents=[-1, 0, 1, 0, 3])
    assert walk.accept_tree(tree, [8, 9, 1, 0, 0]) == ([0, 1, 2], [8, 9, 1])
    assert walk.accept_tree(tree, [4, 0, 0, 5, 6]) == ([0, 3, 4], [4, 5, 6])
    assert walk.accept_tree(tree, [2, 0, 0, 0, 0]) == ([0], [2])
    s._chain = 3                                       # a chain block whose commit waits
    assert s.kv.length == 3 and fn.commits == [[0, 1, 2]] and s._chain is None
    s.st.pool_n = {3: 5}
    s.kv.length = 2
    assert s.st.length == 2 and s.st.pool_n[3] == 0
    s.kv.length = 0
    assert fn.resets == 1 and s.st.length == 0
    try:
        s.forward(torch.tensor([1]), start=5)
        raise AssertionError("a forward off the state's length must be refused")
    except ValueError:
        pass
    cost = s.cost()
    sc = cost.snapshot_cost()
    assert sc["recurrent_bytes"] == 2 * 4 * 4 * 4 and sc["kv_bytes_per_token"] == (2 * 8 * 2 + 4) * 2
    assert sc["conv_bytes"] == 9 * 4 and cost.section_bytes == sc["recurrent_bytes"] + sc["conv_bytes"]
    return "walk with stops at the root and deep; pending chain committed on read; reset; pool clamp; refusal; cost"


def test_pending_verbs():
    fn = _FN()
    s = FlashNextServing(fn, 64)
    s.st.length = 10
    s._chain = 4                                       # a chain block at 10..13 whose commit waits
    assert s.trim_pending(12) and fn.commits == [[0, 1]] and s.st.length == 12 and s._chain is None
    s._chain = 3
    assert s.trim_pending(12) and fn.commits == [[0, 1]] and s.st.length == 12   # none kept: dropped
    assert s.trim_pending(12) and not s.trim_pending(11) and s.st.length == 12   # absorbed: refused
    s._chain = 2
    assert not s.trim_pending(20) and s.st.length == 12 and s._chain == 2       # past the chain: no
    fn._pend_owner = s.st                              # parked GDN rows of this state
    s.settle()
    assert fn.commits[-1] == [0, 1] and fn.flushes == 1 and fn._pend_owner is None and s.st.length == 14
    fn._pend_owner, fn._pend_n[:] = s.st, 3
    s._chain, s.picks = 2, [1, 2]
    s.discard_pending()
    assert s._chain is None and s.picks is None and fn._pend_owner is None and int(fn._pend_n.sum()) == 0
    other = FNState(cap=8)
    fn._pend_owner = other                             # another state's parked rows are not ours
    s.discard_pending()
    s.settle()
    assert fn._pend_owner is other and fn.flushes == 1
    return "trim inside a pending chain, drop it, refuse absorbed rows; settle lands chain and parked rows; discard"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<52} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<52} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
