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

import types

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
    ple_layers = {1: 0}
    eos_token_id, hc_count, hidden_size = 3, 1, 3


class _FN:
    """What the adapters read of FlashNextEngine, on the host."""

    device, dtype = "cpu", torch.float32

    def __init__(self):
        self.cfg = _Cfg()
        self.hash = {1: types.SimpleNamespace(context_len=2)}
        self.pools: list = []
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
        cap = cap or 64
        st = FNState(cap=cap)
        st.S[0] = torch.zeros(1, 2, 4, 4)
        st.conv[0] = torch.zeros(1, 3, 3)
        st.ple_conv[1] = torch.zeros(1, 3, 9)
        st.k[3], st.v[3], st.ik[3] = torch.zeros(2, cap, 8), torch.zeros(2, cap, 8), torch.zeros(cap, 4)
        return st

    def new_mtp_state(self, cap=None):
        cap = cap or 64
        st = FNState(cap=cap)
        st.k[-1], st.v[-1], st.ik[-1] = torch.zeros(2, cap, 8), torch.zeros(2, cap, 8), torch.zeros(cap, 4)
        return st

    def build_pool(self, st, layers=None):
        self.pools.append((st.length, layers))

    def mtp_prime(self, stack, nexts, st_m):
        self.primed.append((stack[:, 0].tolist(), list(nexts), st_m.length))
        st_m.length += len(nexts)

    def reset(self, st):
        self.resets += 1
        st.reset_inplace()

    def commit(self, st, path, graphs=None):
        self.commits.append(list(path))
        st.length += len(path)
        return torch.full((len(path), 3), float(st.length))


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


class _StubHead:
    """An MTP head that always drafts tokens 10 + k (p 0.5) and 20 + k (p 0.1) at depth k."""

    depth = 3

    def __init__(self):
        self.st = types.SimpleNamespace(length=0)

    def reset(self):
        self.st.length = 0

    def prime(self, rows, nexts):
        self.st.length += len(nexts)

    def launch(self, rows, nexts):
        self.st.length += len(nexts)

    def read(self):
        return [[10 + k, 20 + k] for k in range(self.depth)], [[0.5, 0.1]] * self.depth


def test_the_lookup_joins_the_mtp_round():
    """ENG-229: with a lookup the round's tree carries the head's chain and the lookup's copy of the
    request's own text (one verify); the lookup is primed with the prompt, follows what is committed
    and is reset with the drafter; without a match the round is the head's alone."""
    from engine.drafters.ngram import NgramDrafter
    eng = types.SimpleNamespace(captured=False, commit_ms=0.0)
    ng = NgramDrafter(corpus_path="", min_order=3, max_depth=16)
    d = MTPTreeDrafter(eng, _StubHead(), StairCut(PriceModel(), siblings=False), ng)
    text = [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 600, 601]
    prompt = text + [700] + text[:4]                    # the request repeats its own text
    d.reset()
    d.prime(prompt)
    d.sync(prompt, _rows(0, len(prompt)), 0)
    tree = d.propose_tree(prompt, 32)
    tree.check()
    copy = text[4:] + [700] + text[:4]                 # what followed the last match, the lookup's draft
    assert tree.tokens[1:1 + len(copy)] == copy and tree.parents[1:1 + len(copy)] == list(range(len(copy))), tree.tokens
    assert tree.tokens[1 + len(copy):] == [10, 11, 12] and tree.parents[1 + len(copy)] == 0  # the head's chain
    assert d.stats["looked"] == 1
    d.observe([504, 505])                               # committed: the lookup's index follows
    assert ng.local.lookup(prompt + [504, 505], 3)[0] >= 3
    d.reset()
    assert ng.local.lookup(prompt, 3) == (0, [])
    d.prime([1, 2, 3, 4])
    d.sync([1, 2, 3, 4], _rows(0, 4), 0)
    tree = d.propose_tree([1, 2, 3, 4], 32)
    assert tree.tokens == [4, 10, 11, 12] and d.stats["looked"] == 1, tree.tokens
    return "the copy and the head's chain in one tree; primed, observed, reset; no match: the head alone"


def test_counting_the_state_issues_no_copy():
    """2026-10-12: /health and the dashboard (other threads) count the state's bytes through `cost()`;
    inside a graph capture a copy from another thread invalidates the capture, so counting reads the
    buffers as they are, and only `sections()` (the caches, on the engine's thread) brings the PLE
    context up to date."""
    fn = _FN()
    s = FlashNextServing(fn, 64)
    s.st.ple_ctx = {1: torch.tensor([7, 8])}
    before = s._ctx[1].clone()
    s.cost()
    assert torch.equal(s._ctx[1], before)
    assert s.state.sections()["ple.ctx.1"].tolist() == [[7, 8]]
    return "cost() leaves the buffers alone; sections() refreshes the PLE context"


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
    # the GDN conv tails and the PLE conv state; the sections add the PLE context and the MTP input
    assert sc["conv_bytes"] == (9 + 27) * 4
    assert cost.section_bytes == sc["recurrent_bytes"] + sc["conv_bytes"] + 2 * 8 + 3 * 4
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


def test_the_cache_state_and_a_resume():
    fn = _FN()
    s = FlashNextServing(fn, 64)
    sec = s.cache_state().sections()
    assert list(sec) == ["gdn.S.0", "gdn.conv.0", "ple.conv.1", "ple.ctx.1", "mtp.h_last"], list(sec)
    assert sec["ple.ctx.1"].tolist() == [[3, 3]]                     # no context yet: the end-of-text id
    assert [n for n, _, _ in s.cache_state().views()] == ["k.3", "v.3", "ik.3"]
    s.st.ple_ctx[1] = torch.tensor([[5, 6]])
    s.st.length = 4
    s.commit_tree([0, 1])                                             # h_last: the path's last row
    assert s.last_streams().tolist() == [[6.0, 6.0, 6.0]]
    kept = tuple(t.clone() for t in s.cache_state().sections().values())
    assert kept[3].tolist() == [[5, 6]]
    s.st.ple_ctx[1] = torch.tensor([[9, 9]])
    s.st.S[0].fill_(7.0)
    for t, x in zip(s.cache_state().sections().values(), kept):      # what `state_load` does
        t.copy_(x)
    s.resume_at(6)
    assert s.st.ple_ctx[1].tolist() == [[5, 6]] and float(s.st.S[0].abs().sum()) == 0.0
    assert s.st.length == 6 and fn.pools[-1] == (6, None)
    assert s.cost().section_bytes == sum(t.numel() * t.element_size() for t in kept)
    return "5 sections (the PLE context and the MTP input among them), 3 views; a resume restores them"


def test_the_drafter_resumes_and_snapshots_its_head():
    fn = _FN()
    s = FlashNextServing(fn, 64)
    d = MTPTreeDrafter(s, s.add_mtp_head(2), StairCut(PriceModel()))
    assert [t.shape for t, _ in d.kv_views()] == [(2, 64, 8), (2, 64, 8), (64, 4)]
    d.sync(list(range(100, 110)), _rows(0, 10), 0)
    d.sync([110, 111], _rows(10, 2), 10)                              # rows 0..9 now know their next token
    snap = d.state_snapshot()
    assert snap["len"] == 10 and fn.primed[-1][2] == 0 and len(fn.primed[-1][1]) == 10
    assert d.held[0] == [110, 111] and d.queue == []
    s.st.length = 12
    s._h_last.fill_(11.0)
    d.state_resume(12)                                                # a resume in place at 12
    assert d.head.st.length == 11 and fn.pools[-1] == (11, [-1])
    assert d.held[1].tolist() == [[11.0, 11.0, 11.0]] and d.pending is None and d.queue == []
    d.state_restore(snap)
    assert d.head.st.length == 10 and fn.pools[-1] == (10, [-1])
    d.state_resume(0)
    assert d.held is None and d.head.st.length == 0
    return "the head's views; a snapshot feeds the queue first; resume and restore set its length and the waiting row"


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
