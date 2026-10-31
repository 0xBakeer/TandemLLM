"""SPD-249 (T1): the host side of Flash-Next's packed MTP draft graph (QWEN38_FN_PACK_GRAPH), no GPU.

  * the knobs are registered, off by default, and read where the registry says;
  * a draft runs eager where a graph would not hold it (a measurement on, one lane, mixed depths, a sparse class,
    caches of another layout), and while its key has no graph and may not capture one (before its
    QWEN38_FN_PACK_GRAPH_AFTER-th draft, or without the server's word); a key captures once, then replays;
  * a kept graph rewrites its entry rows only for other lanes or another lane order, never for the same lanes;
  * the head reads its own lane of the replay it was in; the ring of pinned copies keeps a replay's picks for
    PACK_RING replays, and a read past that is counted, not hidden;
  * `run_packed` hands each head its lane, and a capture round counts as one on every head (its price is skipped).

The bits of the graph against the eager pass are the board's: tools/fn_pack_graph_check.py,
tests/gpu/test_fn_pack_graph_gpu.py.
"""

from __future__ import annotations

import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.models.flashnext import engine as E  # noqa: E402
from engine.models.flashnext.mtp import FlashNextMTPHead  # noqa: E402
from engine.models.flashnext.ref import MTP  # noqa: E402


def _state(cap=64, hk=2, d=8, contiguous=True):
    st = types.SimpleNamespace(k={}, v={})
    k = torch.zeros(hk, cap, d)
    v = torch.zeros(hk, cap, d)
    if not contiguous:
        k = torch.zeros(hk, d, cap).transpose(1, 2)
    st.k[MTP], st.v[MTP] = k, v
    return st


def _items(n=3, depth=4, cls=0, states=None):
    states = states or [_state() for _ in range(n)]
    return [(i, st, 2, depth, cls) for i, st in enumerate(states)]


class _Graph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1

    def reset(self):
        pass


class _Event:
    def __init__(self):
        self.recorded = 0
        self.waits = 0

    def record(self):
        self.recorded += 1

    def synchronize(self):
        self.waits += 1


class _PG:
    """`PackGraph` on the host: its key, inputs and picks, no pinned memory, no event."""

    made: list = []

    def __init__(self, eng, key, xw, hw, xdt, hdt):
        self.key, self.L, self.depth = key, key[0], key[1]
        self.x, self.h = torch.zeros(key[0], xw), torch.zeros(key[0], hw)
        self.graph, self.sig, self.done = None, None, _Event()
        self.rows = [slice(i, i + 1) for i in range(key[0])]
        self.aent = {"k": 0}
        self.stale = 0
        self.published = 0
        _PG.made.append(self)

    def publish(self, captured):
        self.published += 1
        return types.SimpleNamespace(G=self, captured=captured)

    def drop(self):
        self.graph = None


class _Eng:
    """What `_pack_graph_run` reads of FlashNextEngine; the capture and the entry fill are recorded."""

    _pg_state = E.FlashNextEngine._pg_state
    _pack_why = E.FlashNextEngine._pack_why
    _pack_graph_run = E.FlashNextEngine._pack_graph_run
    pack_graph_stats = E.FlashNextEngine.pack_graph_stats
    _pack_use = E.FlashNextEngine._pack_use

    def __init__(self):
        self.draft_head = None
        self.pack_capture_ok = False
        self.route_trace = None
        self._moe_record = None
        self.calls: list = []

    def _pack_lanes(self, items):
        return [[("lanes", tuple(it[0] for it in items), j)] for j in range(1, items[0][3])]

    def _attn_multi_fill(self, cx, rows, li, use, ent, k):
        assert li == MTP
        self.calls.append(("fill", k, cx[0][1]))

    def _pack_capture(self, G, cxs):
        self.calls.append(("capture", G.key[0]))
        G.graph = _Graph()
        ps = self._pg_state()
        ps["graphs"][G.key] = G
        ps["stats"]["captures"] += 1


def _run(eng, items, rows=6, H=4):
    x, h = torch.arange(rows * H, dtype=torch.float32).view(rows, H), torch.zeros(rows, 2 * H)
    last = torch.tensor([1 + 2 * i for i in range(len(items))])
    return eng._pack_graph_run(items, last, x, h)


def test_t1_the_knobs_are_registered_off():
    from engine.settings import KNOBS
    assert KNOBS["FN_PACK_GRAPH"] == ("0", "engine/models/flashnext/engine.py"), KNOBS["FN_PACK_GRAPH"]
    assert KNOBS["FN_PACK_GRAPH_AFTER"] == ("2", "engine/models/flashnext/engine.py"), KNOBS["FN_PACK_GRAPH_AFTER"]
    if "QWEN38_FN_PACK_GRAPH" not in os.environ:
        assert E.PACK_GRAPH is False
    return "FN_PACK_GRAPH 0, FN_PACK_GRAPH_AFTER 2, read in engine.py"


def test_t1_why_a_draft_runs_eager():
    eng = _Eng()
    assert eng._pack_why(_items()) is None
    assert eng._pack_why(_items(n=1)) == "lanes"
    mixed = _items()
    mixed[1] = (1, mixed[1][1], 2, 3, 0)
    assert eng._pack_why(mixed) == "depth"
    assert eng._pack_why(_items(cls=4096)) == "sparse"
    assert eng._pack_why(_items(states=[_state(), _state(cap=128), _state()])) == "layout"
    assert eng._pack_why(_items(states=[_state(), _state(contiguous=False), _state()])) == "layout"
    eng.route_trace = []
    assert eng._pack_why(_items()) == "measuring"
    return "lanes, depth, sparse, layout (cap, strides), measuring"


def test_t1_a_key_captures_once_at_its_nth_draft_with_the_servers_word():
    _PG.made.clear()
    real = E.PackGraph
    E.PackGraph = _PG
    try:
        eng = _Eng()
        items = _items(n=3)
        assert _run(eng, items) is None                      # 1st draft of the key: eager
        eng.pack_capture_ok = False
        assert _run(eng, items) is None                      # 2nd, but no word from the server: eager
        eng.pack_capture_ok = True
        out = _run(eng, items)                               # the server allows: captured, then replayed
        assert out is not None and out.captured and [c for c in eng.calls if c[0] == "capture"] == [("capture", 3)]
        G = _PG.made[-1]
        assert G.graph.replays == 1 and G.published == 1
        assert torch.equal(G.x, torch.arange(24, dtype=torch.float32).view(6, 4)[[1, 3, 5]])
        out2 = _run(eng, items)
        assert out2 is not None and not out2.captured and G.graph.replays == 2
        assert [c for c in eng.calls if c[0] == "capture"] == [("capture", 3)], eng.calls
        st = eng.pack_graph_stats()
        assert st["eager"] == {"no graph": 2} and st["captures"] == 1 and st["replays"] == 2, st
        assert st["kept"] == [[3, 4]], st
    finally:
        E.PackGraph = real
    return "eager, eager (no word), capture + replay, replay; one capture"


def test_t1_a_kept_graph_rewrites_its_entries_only_for_other_lanes():
    _PG.made.clear()
    real, after = E.PackGraph, E.PACK_GRAPH_AFTER
    E.PackGraph, E.PACK_GRAPH_AFTER = _PG, 1
    try:
        eng = _Eng()
        eng.pack_capture_ok = True
        sts = [_state() for _ in range(4)]
        a = [(0, sts[0], 2, 4, 0), (1, sts[1], 1, 4, 0), (2, sts[2], 3, 4, 0)]
        _run(eng, a)
        G = _PG.made[-1]
        fills0 = [c for c in eng.calls if c[0] == "fill"]
        assert fills0 == [], "a capture's own run writes its entry rows"
        _run(eng, a)                                          # the same lanes: no rewrite
        assert [c for c in eng.calls if c[0] == "fill"] == []
        b = [(0, sts[0], 2, 4, 0), (3, sts[3], 1, 4, 0), (2, sts[2], 3, 4, 0)]    # another lane, same count
        _run(eng, b)
        fills = [c for c in eng.calls if c[0] == "fill"]
        assert [f[1] for f in fills] == [0, 1, 2] and G.done.waits == 1, (fills, G.done.waits)
        assert all(f[2] == (0, 3, 2) for f in fills), fills
        c = [b[2], b[0], b[1]]                                # the same lanes in another order
        _run(eng, c)
        assert len([x for x in eng.calls if x[0] == "fill"]) == 6 and G.done.waits == 2
        _run(eng, c)
        assert len([x for x in eng.calls if x[0] == "fill"]) == 6
        assert len(_PG.made) == 1 and G.graph.replays == 5
        assert eng.pack_graph_stats()["refills"] == 2
        # another lane count is another key: a graph of its own
        _run(eng, a[:2])
        assert len(_PG.made) == 2 and _PG.made[-1].L == 2
    finally:
        E.PackGraph, E.PACK_GRAPH_AFTER = real, after
    return "same lanes: no rewrite; a new lane or order: one rewrite after the last replay; 2 lanes: own graph"


def _ring_graph(L=3, depth=2):
    G = object.__new__(E.PackGraph)
    G.L, G.depth, G.n, G.stale = L, depth, 0, 0
    G.gen = [0] * E.PACK_RING
    G.ring = [(torch.zeros(depth, L, 2, dtype=torch.long), torch.zeros(depth, L, 2)) for _ in range(E.PACK_RING)]
    G.tok = torch.zeros(depth, L, 2, dtype=torch.long)
    G.prob = torch.zeros(depth, L, 2)
    G.done = _Event()
    return G


def test_t1_a_head_reads_its_lane_of_its_replay():
    G = _ring_graph()
    outs = []
    for r in range(3):
        G.tok.copy_(torch.arange(12).view(2, 3, 2) + 100 * r)
        G.prob.copy_(torch.full((2, 3, 2), 0.25 * r))
        outs.append(G.publish(captured=(r == 0)))
    assert G.done.recorded == 3 and outs[0].captured and not outs[1].captured
    tok, prob = outs[1].read(2, 2)
    assert tok == [[104, 105], [110, 111]] and prob == [[0.25, 0.25], [0.25, 0.25]], (tok, prob)
    assert outs[0].read(0, 1)[0] == [[0, 1]] and G.stale == 0
    for _ in range(E.PACK_RING):                              # PACK_RING more replays take replay 0's slot
        G.publish(captured=False)
    outs[0].read(0, 1)
    assert G.stale == 1
    return f"lane rows of the replay read; its slot kept for {E.PACK_RING} replays, a later read counted"


def test_t1_run_packed_hands_each_head_its_lane():
    G = _ring_graph(L=2, depth=3)
    G.tok.copy_(torch.arange(12).view(3, 2, 2))
    G.prob.copy_(torch.arange(12, dtype=torch.float32).view(3, 2, 2) / 16)
    res = {"out": G.publish(captured=True)}

    class _F:
        _mtp_key = 0

        def mtp_chain_multi(self, items):
            res["items"] = items
            return res["out"]

    fn = _F()
    heads = []
    for i in range(2):
        h = object.__new__(FlashNextMTPHead)
        h.fn, h.depth, h.key, h._n, h._cls, h._pk, h.captures = fn, 3, 10 + i, 2 + i, 0, None, 5
        h.st = types.SimpleNamespace(length=40)
        heads.append(h)
    FlashNextMTPHead.run_packed(heads)
    assert [it[0] for it in res["items"]] == [10, 11] and [it[2] for it in res["items"]] == [2, 3]
    assert [h.captures for h in heads] == [6, 6], "a capture round is every head's"
    real = torch.cuda.current_stream
    torch.cuda.current_stream = lambda: types.SimpleNamespace(synchronize=lambda: None)
    try:
        tok1, prob1 = heads[1].read()
        assert tok1 == [[2, 3], [6, 7], [10, 11]] and prob1 == [[0.125, 0.1875], [0.375, 0.4375], [0.625, 0.6875]]
        assert heads[1].st.length == 43 and heads[1]._pk is None
        res["out"] = None                                     # eager: the heads read their own buffers
        FlashNextMTPHead.run_packed(heads)
        assert heads[0]._pk is None and [h.captures for h in heads] == [6, 6]
    finally:
        torch.cuda.current_stream = real
    return "lane i to head i, a capture counted on every head; eager leaves the heads' buffers"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<64} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<64} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
