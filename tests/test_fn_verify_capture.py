"""SPD-261: Flash-Next's verify, commit and draft graphs at a context-class edge and after a failed capture, no GPU.

SRV-765 (fd627ac3, tool-eval-bench): a 13-row tree whose deepest row sat at 4,095 (class 1,024) found no graph for
(13, 1,024); `verify` keyed a new one and captured it over a chain of 13 rows, whose last row (4,097) is class 2,048:
the warm-up's `class_check` raised `ClassError` and the request failed. The uncaptured graph stayed in `vgraphs`, and
the next two requests on that slot (new conversations, class 1,024, 13 rows) replayed it: `'NoneType' object has no
attribute 'replay'`.

  * a block whose chain would cross its class edge verifies eagerly (the block's own class, the same launch), and so
    does its commit (a commit graph reads the verify graph's own intermediates); no graph is kept for it, and the
    round is not a price of its width;
  * the next conversation captures that key cleanly and replays it;
  * a verify, commit or draft capture that fails keeps no graph, wherever it failed (a warm-up, the capture, its
    end); the state its warm-ups wrote is put back (parked GDN rows folded into part of the layers, a commit applied
    once), the call runs eagerly on the thread's own stream, the round completes, and the next one captures the key;
  * an idle capture that fails leaves the state as it was;
  * a graph a failed capture dropped is kept on the adapter and never reset (its reset would give back the cuBLAS
    workspace of its capture stream, SPD-223).

`FlashNextEngine.verify`, `commit`, `qsa_class`, `class_check`, the `VerifyGraph` / `CommitGraph` / `MTPChainGraph`
classes and the MTP head are the real ones; the launches and the CUDA graph and stream calls are host stand-ins.
"""

from __future__ import annotations

import contextlib
import gc
import os
import sys
import weakref

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext import engine as E  # noqa: E402
from engine.models.flashnext.ref import FNState  # noqa: E402
from engine.models.flashnext.serving import FlashNextServing  # noqa: E402

V = 5
#: a 13-row tree of depth 10 (a chain of 11 rows, two more children of row 9), as SRV-765's failed block
TREE13 = list(range(-1, 10)) + [9, 9]
CHAIN13 = list(range(-1, 12))


class _Cfg:
    has_indexer, indexer_compress_ratio = True, 4
    ple_layers: dict = {}
    linear_layers = ()
    eos_token_id, hc_count, hidden_size = 3, 1, 3


class _Stream:
    def __init__(self, name: str):
        self.name = name

    def wait_stream(self, s) -> None:
        pass

    def synchronize(self) -> None:
        pass


class _Cuda:
    """The CUDA calls a capture and a replay make, on the host. `where` says where a launch runs ("warm" on a
    capture's side stream, "graph" inside a capture, else "eager"); `cur` is the thread's stream. A graph replays only
    once captured. `fail` makes the next capture section raise before it begins; `fail_end` makes the next capture's
    end raise, which leaves the thread on the capture stream (torch 2.13's `torch.cuda.graph.__exit__` ends the
    capture before it leaves the stream)."""

    def __init__(self):
        self.where = "eager"
        self.fail = self.fail_end = None
        self.replays: list = []
        self.resets: list = []
        self.default, self.capture = _Stream("default"), _Stream("capture")
        self.cur = self.default
        cuda = self

        class Graph:
            def __init__(self):
                self.done = False

            def replay(self):
                if not self.done:
                    raise RuntimeError("replay of a graph that was never captured")
                cuda.replays.append(self)

            def reset(self):
                cuda.resets.append(self)
                self.done = False

        @contextlib.contextmanager
        def graph(g, **kw):
            if cuda.fail is not None:
                exc, cuda.fail = cuda.fail, None
                raise exc
            prev, cuda.cur, cuda.where = cuda.cur, cuda.capture, "graph"
            try:
                yield
            except BaseException:
                cuda.cur, cuda.where = prev, "eager"
                raise
            cuda.where = "eager"
            if cuda.fail_end is not None:
                exc, cuda.fail_end = cuda.fail_end, None
                raise exc                       # still on the capture stream
            cuda.cur = prev
            g.done = True
        graph.default_capture_stream = self.capture

        @contextlib.contextmanager
        def side(s):
            prev, cuda.cur, cuda.where = cuda.cur, s, "warm"
            try:
                yield
            finally:
                cuda.cur, cuda.where = prev, "eager"

        self.calls = {"Stream": lambda *a, **k: _Stream("side"), "stream": side,
                      "current_stream": lambda *a, **k: cuda.cur, "default_stream": lambda *a, **k: cuda.default,
                      "set_stream": lambda s: setattr(cuda, "cur", s), "CUDAGraph": Graph, "graph": graph,
                      "synchronize": lambda *a, **k: None}

    @contextlib.contextmanager
    def on(self):
        real = {k: getattr(torch.cuda, k) for k in self.calls}
        try:
            for k, v in self.calls.items():
                setattr(torch.cuda, k, v)
            yield self
        finally:
            for k, v in real.items():
                setattr(torch.cuda, k, v)


class _Eng:
    """FlashNextEngine's own verify, commit, class and class guard; `rows_step`, `commit_dev` and the MTP chain stand
    in for the launches, each recorded after the real guard passed (rows, class, the block's last position, where),
    with the stream it ran on. With `fold` the state is two GDN layers (`st.S`) and a conv window (`st.conv`): a
    verify launch folds the parked rows' count into each layer and zeroes it, a commit advances the window by one and
    parks its path. `fail_rows` raises at the start of the next launches, `fail_layer` (layer, where, exception) in
    the middle of one, `fail_commit` / `fail_mtp` (where, exception) in a commit or a draft launch."""

    device, dtype = "cpu", torch.float32
    verify = E.FlashNextEngine.verify
    commit = E.FlashNextEngine.commit
    qsa_class = E.FlashNextEngine.qsa_class
    class_check = E.FlashNextEngine.class_check

    def __init__(self, cuda: _Cuda, fold: bool = False):
        self.cuda = cuda
        self.cfg = _Cfg()
        self.hash: dict = {}
        self._rb, self._joint, self._pending, self.host_ms = {}, {}, {}, {}
        self.fold, self.rec, self._vcls = fold, {}, 0
        self.launches: list = []
        self.streams: list = []
        self.commits: list = []
        self.drafts: list = []
        self.fail_rows: list = []
        self.fail_layer = self.fail_commit = self.fail_mtp = None
        self._n = torch.zeros(1)

    def new_state(self, cap=None, paged=False):
        st = FNState(cap=cap or 64)
        if self.fold:
            st.S = {0: torch.zeros(1, 2), 1: torch.zeros(1, 2)}
            st.conv = {0: torch.zeros(1, 2)}
        return st

    def settle(self, st):
        pass

    def _pend_of(self, st):
        return None, self._n

    def _rows_buffers(self, M, key=None):
        rb = self._rb.get(M)
        if rb is None:
            rb = self._rb[M] = {"logits": torch.zeros(M, V), "next": torch.zeros(M, dtype=torch.long),
                                "next_h": torch.zeros(M, dtype=torch.long)}
        return rb

    def stage_rows(self, tokens, parents, st, key=None, ple=None, rb=None):
        depth = [0] * len(parents)
        for i, p in enumerate(parents):
            depth[i] = 0 if p < 0 else depth[p] + 1
        self._rows_buffers(len(tokens))["last"] = st.length + max(depth)
        self._block = (list(tokens), list(parents), None)

    def _fail(self, name: str, where: str):
        f = getattr(self, name)
        if f is not None and f[0] == where:
            setattr(self, name, None)
            raise f[1]

    def rows_step(self, st, M, write=False, cls=0):
        rb = self._rows_buffers(M)
        self.class_check(rb.get("last", st.length + M - 1), cls, "rows_step")       # the real step's first check
        if self.fail_rows:
            raise self.fail_rows.pop(0)
        if self.fold and self.cuda.where != "graph":         # a capture records; a replay runs what it recorded
            for li in sorted(st.S):
                if self.fail_layer is not None and self.fail_layer[:2] == (li, self.cuda.where):
                    exc, self.fail_layer = self.fail_layer[2], None
                    raise exc
                st.S[li] += self._n                          # the parked rows walked into this layer
            self._n.zero_()
        self.rec = {"stack": torch.zeros(M, 3)}
        self.stack = self.rec["stack"]
        self.launches.append((M, cls, rb["last"], self.cuda.where))
        if self.cuda.where == "eager":
            self.streams.append(self.cuda.cur.name)
        return rb["logits"]

    def stage_commit(self, st, path):
        return (len(self._block[0]), len(path), any(p != j for j, p in enumerate(path)), self._vcls)

    def commit_dev(self, st, key):
        if self.fold and self.cuda.where != "graph":
            for t in st.conv.values():
                t += 1                                       # the path's columns into the window: twice is wrong
            self._n.fill_(key[1])
        self.commits.append((key, self.cuda.where))
        self._fail("fail_commit", self.cuda.where)

    def _commit_bufs(self, key):
        return {"stack": torch.zeros(key[1], 3)}

    # the MTP head's chain
    def new_mtp_state(self, cap, pages=None):
        return FNState(cap=cap)

    def _mtp_bufs(self, depth):
        return self.__dict__.setdefault("_mb", {"stack": torch.zeros(33, 3), "tok_h": torch.zeros(32, 2),
                                                "prob_h": torch.zeros(32, 2)})

    def stage_mtp(self, st_m, nexts, depth, keyed=None):
        return self.qsa_class(st_m.length + len(nexts) + depth - 1)

    def mtp_chain_dev(self, st_m, n, depth, cls=0, keyed=False):
        self.class_check(st_m.length + n + depth - 1, cls, "mtp_chain_dev")
        self._fail("fail_mtp", self.cuda.where)
        self.drafts.append((n, cls, self.cuda.where, self.cuda.cur.name))

    def keyed_head(self, lanes):
        return None, None


def _serving(cuda: _Cuda, length: int, fold: bool = False):
    fn = _Eng(cuda, fold)
    s = FlashNextServing(fn, 8192)
    s.st.length = length
    return fn, s


def _eager(fn):
    return [x for x in fn.launches if x[3] == "eager"]


def test_spd261_a_tree_under_a_class_edge_verifies_eagerly_and_commits_eagerly():
    """SRV-765's block: 13 rows from 4,085, deepest row at 4,095 (class 1,024); the chain a capture stages ends at
    4,097 (class 2,048). The block verifies eagerly at class 1,024, keeps no graph, and its commit runs eagerly. The
    round is not a price of its width (an eager verify), and no capture time is taken from it."""
    cuda = _Cuda()
    fn, s = _serving(cuda, 4085)
    assert fn.qsa_class(4085 + 10) == 1024 and fn.qsa_class(4085 + 12) == 2048
    with cuda.on():
        s.forward_tree(torch.arange(13), TREE13, 4085)
        assert fn.launches == [(13, 1024, 4095, "eager")], fn.launches
        assert (13, 1024) not in s.vgraphs and s.stats["verify_eager"] == 1
        assert s.captured and s.capture_ms is None, (s.captured, s.capture_ms)
        s.commit_tree([0, 1, 2])
    assert fn.commits == [((13, 3, False, 1024), "eager")] and not s.cgraphs, (fn.commits, s.cgraphs)
    assert s.st.length == 4088
    return "13 rows at 4,085, depth 10: one eager launch at class 1,024, no graph kept, an eager commit, not priced"


def test_spd261_the_next_conversation_captures_and_replays_that_key():
    """After SRV-765's block the slot's next requests (new conversations at 2,066 and 2,153 tokens, class 1,024,
    13-row blocks) failed on the graph left uncaptured. Now the first captures (13, 1,024) and the next replays it."""
    cuda = _Cuda()
    fn, s = _serving(cuda, 4085)
    with cuda.on():
        try:
            s.forward_tree(torch.arange(13), TREE13, 4085)
        except E.ClassError:
            pass                                 # fd627ac3: the capture's ClassError; the asserts below then fail
        s.commit_tree([0])
        for n in (2066, 2153):
            s.st.length = n
            s.forward_tree(torch.arange(13), CHAIN13, n)
            s.commit_tree([0, 1])
    g = s.vgraphs[(13, 1024)]
    assert g.graph is not None and g.graph.done and len(cuda.replays) >= 2, (g.graph, cuda.replays)
    caps = [x for x in fn.launches if x[3] == "graph"]
    assert caps == [(13, 1024, 2078, "graph")], caps
    assert not s.captured, "a replayed graph's round is a price"
    return "the next conversation captured (13, 1,024) over its own chain and both rounds replayed it"


#: where a verify capture fails: the warm-up's class check (before any launch), the middle of the first warm-up
#: (layer 0 folded, layer 1 not), the capture section, the capture's end (the thread left on the capture stream)
VERIFY_FAILS = (("class", lambda fn, c: fn.fail_rows.append(E.ClassError("rows_step: class 1024 below 2048"))),
                ("warm-up", lambda fn, c: setattr(fn, "fail_layer", (1, "warm", RuntimeError("out of memory")))),
                ("capture", lambda fn, c: setattr(c, "fail", RuntimeError("capture failed"))),
                ("end", lambda fn, c: setattr(c, "fail_end", RuntimeError("operation not permitted when stream is "
                                                                          "capturing"))))


def test_spd261_a_failed_verify_capture_keeps_no_graph_and_the_round_completes():
    """A verify capture that fails keeps no graph, wherever it failed. The state is put back first: 3 parked GDN rows
    are folded into every layer exactly once (a warm-up that broke after layer 0 folded them there and not into layer
    1, with the count not zeroed). The block verifies eagerly on the thread's own stream (after a failed capture end,
    torch leaves the thread on the capture stream), the round completes and is not a price, and its commit runs
    eagerly. The next round captures the key again and replays it."""
    out = []
    for name, arm in VERIFY_FAILS:
        cuda = _Cuda()
        fn, s = _serving(cuda, 2300, fold=True)
        fn._n.fill_(3)
        with cuda.on():
            arm(fn, cuda)
            s.forward_tree(torch.arange(5), [-1, 0, 1, 1, 3], 2300)
            assert (5, 1024) not in s.vgraphs, (name, s.vgraphs)
            assert _eager(fn) == [(5, 1024, 2303, "eager")] and fn.streams == ["default"], (name, fn.launches,
                                                                                             fn.streams)
            assert cuda.cur is cuda.default, (name, cuda.cur.name)
            assert all(torch.equal(t, torch.full((1, 2), 3.0)) for t in s.st.S.values()) and int(fn._n) == 0, \
                (name, s.st.S, fn._n)
            assert s._eager_block and s.stats["verify_eager"] == 1 and s.captured and s.capture_ms is not None, name
            s.commit_tree([0, 1])
            assert fn.commits[-1] == ((5, 2, False, 1024), "eager") and not s.cgraphs, (name, fn.commits)
            assert torch.equal(s.st.conv[0], torch.ones(1, 2)) and int(fn._n) == 2 and s.st.length == 2302, name
            s.st.length = 2310
            n = len(cuda.replays)
            s.forward_tree(torch.arange(5), [-1, 0, 1, 2, 3], 2310)
            g = s.vgraphs.get((5, 1024))
            assert g is not None and g.graph is not None and g.graph.done and len(cuda.replays) == n + 1, (name, g)
            s.commit_tree([0, 1, 2])
            assert s.cgraphs and fn.commits[-1][1] == "graph", (name, fn.commits)
        out.append(name)
    return "no graph kept after a failed verify capture (" + ", ".join(out) + "); the state folded once, the block " \
           "verified eagerly on the default stream; the next round captured and replayed the key"


def test_spd261_a_failed_commit_capture_keeps_no_graph_and_the_commit_completes():
    """`commit` keys a new commit graph before it captures it: a failed capture left it, and the next commit of that
    key replayed None. `CommitGraph.capture`'s warm-up applies the commit and is undone only after a capture that
    returns. Now the key is dropped, the state is put back and the commit runs eagerly: the window advanced once, the
    path parked, the length moved. The next commit of that key captures it again."""
    out = []
    for name, arm in (("warm-up", lambda fn, c: setattr(fn, "fail_commit", ("warm", RuntimeError("out of memory")))),
                      ("capture", lambda fn, c: setattr(c, "fail", RuntimeError("capture failed"))),
                      ("end", lambda fn, c: setattr(c, "fail_end", RuntimeError("capture invalidated")))):
        cuda = _Cuda()
        fn, s = _serving(cuda, 2300, fold=True)
        with cuda.on():
            s.forward_tree(torch.arange(3), [-1, 0, 1], 2300)
            arm(fn, cuda)
            s.commit_tree([0, 1])
            assert not s.cgraphs and fn.commits[-1] == ((3, 2, False, 1024), "eager"), (name, s.cgraphs, fn.commits)
            assert torch.equal(s.st.conv[0], torch.ones(1, 2)) and int(fn._n) == 2 and s.st.length == 2302, \
                (name, s.st.conv, fn._n, s.st.length)
            assert cuda.cur is cuda.default and s.captured, (name, cuda.cur.name, s.captured)
            s.forward_tree(torch.arange(3), [-1, 0, 1], 2302)
            s.commit_tree([0, 1])
        g = s.cgraphs.get((3, 2, False, 1024))
        assert g is not None and g.graph is not None and g.graph.done and s.st.length == 2304, (name, g, s.st.length)
        out.append(name)
    return "a failed commit capture (" + ", ".join(out) + ") kept nothing; the commit applied once, eagerly; the " \
           "next commit captured and replayed its key"


def test_spd261_a_failed_mtp_chain_capture_keeps_no_graph_and_drafts_eagerly():
    """`FlashNextMTPHead.launch` keys a new chain graph before it captures it: a capture that raised left it, and
    every later draft of that key replayed it. The head `add_mtp_head` builds drops the key and runs that draft's chain
    eagerly on the thread's own stream (the drafter does not price the round); the next draft of the key captures it
    and replays it. `capture_all` drops every graph it keyed before it raises."""
    out = []
    for name, arm in (("warm-up", lambda fn, c: setattr(fn, "fail_mtp", ("warm", RuntimeError("cannot be accessed "
                                                                                              "from Triton")))),
                      ("capture", lambda fn, c: setattr(c, "fail", RuntimeError("out of memory"))),
                      ("end", lambda fn, c: setattr(c, "fail_end", RuntimeError("capture invalidated")))):
        cuda = _Cuda()
        fn, s = _serving(cuda, 2300)
        with cuda.on():
            head = s.add_mtp_head(3)
            head.st.length = 2300
            key = (2, 1024)                     # an argmax draft's chain (`keyed` None)
            arm(fn, cuda)
            head.launch(torch.zeros(2, 3), [7, 8])
            assert key not in head.graphs, (name, list(head.graphs))
            assert fn.drafts[-1] == (2, 1024, "eager", "default") and cuda.cur is cuda.default, (name, fn.drafts)
            assert head.captures == 1, (name, head.captures)
            head.read()
            assert head.st.length == 2302, name
            head.launch(torch.zeros(2, 3), [7, 8])
            g = head.graphs.get(key)
            assert g is not None and g.graph.done and cuda.replays and head.captures == 2, (name, g)
        out.append(name)
    cuda = _Cuda()
    fn, s = _serving(cuda, 2300)
    with cuda.on():
        head = s.add_mtp_head(3)
        head.st.length = 2300
        real, n = fn.mtp_chain_dev, {"graph": 0}

        def third_capture_fails(st_m, n_, depth, cls=0, keyed=False):
            if cuda.where == "graph":
                n["graph"] += 1
                if n["graph"] == 3:
                    raise RuntimeError("out of memory")
            return real(st_m, n_, depth, cls, keyed)
        fn.mtp_chain_dev = third_capture_fails
        try:
            head.capture_all()
        except RuntimeError:
            pass
        else:
            raise AssertionError("capture_all swallowed its failure")
        assert head.graphs == {} and cuda.cur is cuda.default, list(head.graphs)
        fn.mtp_chain_dev = real
        head.capture_all()
        assert head.graphs and all(g.graph.done for g in head.graphs.values()), head.graphs
    out.append("capture_all")
    return "no MTP chain graph kept after a failed capture (" + ", ".join(out) + "); the draft ran eagerly on the " \
           "default stream, the next draft captured and replayed its key"


def test_spd261_a_failed_idle_capture_leaves_the_state_as_it_was():
    """An idle window's verify capture whose warm-up broke after one GDN layer: the parked rows' count and the layers
    as they were, no graph kept, the failure raised to the idle loop (which logs it)."""
    cuda = _Cuda()
    fn, s = _serving(cuda, 2300, fold=True)
    fn._n.fill_(3)
    s.uses = {5: 1}
    with cuda.on():
        fn.fail_layer = (1, "warm", RuntimeError("out of memory"))
        steps = s.idle_captures(ahead=False)
        try:
            for _ in steps:
                pass
        except RuntimeError:
            pass
        else:
            raise AssertionError("the idle capture's failure was swallowed")
    assert (5, 1024) not in s.vgraphs and int(fn._n) == 3, (s.vgraphs, fn._n)
    assert all(torch.equal(t, torch.zeros(1, 2)) for t in s.st.S.values()), s.st.S
    return "a failed idle verify capture left 3 parked rows unfolded and no graph"


def test_spd261_a_dropped_graph_is_kept_and_never_reset():
    """A verify, commit, draft or idle capture that fails drops its graph from its key, and the graph object stays alive
    in the adapter's `dead_graphs` (the MTP head's `dead` is that list), never reset: torch's `CUDAGraph.reset`, also
    run by its destructor, gives back the cuBLAS workspace of the graph's capture stream, which a kept graph captured on
    the same stream may read (engine/capture_lock.py, SPD-223). fd627ac3 kept such a graph under its key."""
    out = []

    def kept(name, s, cuda, kind):
        assert len(s.dead_graphs) == 1 and isinstance(s.dead_graphs[0], kind), (name, s.dead_graphs)
        refs = [weakref.ref(s.dead_graphs[0])] + ([weakref.ref(s.dead_graphs[0].graph)]
                                                   if s.dead_graphs[0].graph is not None else [])
        gc.collect()
        assert all(r() is not None for r in refs) and not cuda.resets, (name, refs, cuda.resets)
        out.append(name)

    cuda = _Cuda()
    fn, s = _serving(cuda, 2300, fold=True)
    with cuda.on():
        cuda.fail_end = RuntimeError("capture invalidated")
        s.forward_tree(torch.arange(5), [-1, 0, 1, 1, 3], 2300)
        assert (5, 1024) not in s.vgraphs
    kept("verify", s, cuda, E.VerifyGraph)

    cuda = _Cuda()
    fn, s = _serving(cuda, 2300, fold=True)
    with cuda.on():
        s.forward_tree(torch.arange(3), [-1, 0, 1], 2300)
        cuda.fail = RuntimeError("capture failed")
        s.commit_tree([0, 1])
        assert not s.cgraphs and (3, 1024) in s.vgraphs
    kept("commit", s, cuda, E.CommitGraph)

    cuda = _Cuda()
    fn, s = _serving(cuda, 2300)
    with cuda.on():
        head = s.add_mtp_head(3)
        assert head.dead is s.dead_graphs
        head.st.length = 2300
        cuda.fail_end = RuntimeError("capture invalidated")
        head.launch(torch.zeros(2, 3), [7, 8])
        assert (2, 1024) not in head.graphs
    kept("draft", s, cuda, E.MTPChainGraph)

    cuda = _Cuda()
    fn, s = _serving(cuda, 2300, fold=True)
    s.uses = {5: 1}
    with cuda.on():
        cuda.fail = RuntimeError("capture failed")
        try:
            for _ in s.idle_captures(ahead=False):
                pass
        except RuntimeError:
            pass
        else:
            raise AssertionError("the idle capture's failure was swallowed")
        assert (5, 1024) not in s.vgraphs
    kept("idle", s, cuda, E.VerifyGraph)
    return "a dropped graph (" + ", ".join(out) + ") stayed alive in dead_graphs and was never reset"
