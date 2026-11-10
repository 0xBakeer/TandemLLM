"""SRV-795: a slot's CUDA graphs on a capture stream of its own, dropped all at once (engine/models/graph_set.py,
engine/models/flashnext/serving.py `drop_graphs`, server/app.py `_drop_slot_graphs`). CPU only, fake graphs, streams
and allocator pools.

THE INCIDENT (2026-11-08 19:34): live :8000, up 29 h, held 3,905 graphs' pools, 33.0 GB reserved (21.2 GB free and
unreleasable), every capture since the start still alive; MemAvailable was 9 GiB while idle and chats ended with
`memory is full`. Nothing ever reset a slot's lazily captured graphs.

  * a graph set leases its capture stream at its first capture and keeps it for its life; a drop resets every graph
    in it and leaves the counting of the reset to its caller, after the allocator gave the pools back;
  * every capture in engine/models/flashnext/engine.py names a real stream (`capture_stream`); a state without a set
    captures on torch's stream as before; a decode capture that fails leaves the capture stream and the state as
    they were, its graph kept for its slot's drop;
  * a Flash-Next slot's drop resets its decode, verify, commit, draft and failed captures' graphs, once each, after a
    pending chain's commit, and clears them; its idle captures wait for its next round; an idle capture step holds
    the set's lock, never across a step's yield, and a drop between two steps ends that slot's window;
  * the server drops idle slots' sets the least recently used first, never the asking slot's, a locked one's, one a
    request still works on (a phase, a round runner task; slot 0 also the GPU turn), one an idle capture step holds
    or while a capture runs; each drop counts its own pools' bytes, empties the allocator, then counts the reset;
    the reclaim stops at the KV pool's own reading and after two drops that gave nothing;
  * the KV pool's ladder runs `graphs` after the caches and the small models, before any conversation;
  * the cap counts the growth past the load's pools: over it an idle window drops sets down to 3/4 of it, never one
    that served in the last 5 minutes, and stops at a drop that gave nothing; it captures only while one more graph
    stays under the cap, so the next window drops nothing; `QSE_RECLAIM_GRAPHS=off` drops nothing.

Run: python tests/test_graph_set.py
"""

from __future__ import annotations

import ast
import contextlib
import io
import os
import sys
import threading
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_forward_tree  # noqa: E402,F401  (the CPU environment before the engine imports)

import torch  # noqa: E402

from engine import capture_lock  # noqa: E402
from engine.models.graph_set import GraphSet, lru  # noqa: E402

GiB = 1 << 30


class _Graph:
    """A CUDAGraph: its resets, in the shared log."""

    def __init__(self, log: list, name: str):
        self.log, self.name = log, name

    def reset(self):
        self.log.append(("reset", self.name))


@contextlib.contextmanager
def _streams():
    """capture_lock's streams made on the host; the free list and counts as they were after."""
    made = []

    def new():
        made.append(types.SimpleNamespace(n=len(made)))
        return made[-1]
    real, free, leases = capture_lock._new_stream, list(capture_lock._FREE), dict(capture_lock.LEASES)
    capture_lock._new_stream = new
    capture_lock._FREE.clear()
    try:
        yield made
    finally:
        capture_lock._new_stream = real
        capture_lock._FREE[:] = free
        capture_lock.LEASES.update(leases)


def test_a_set_keeps_its_one_stream_and_a_drop_resets_every_graph():
    log = []
    with _streams() as made:
        gs = GraphSet()
        assert not gs.holding and capture_lock.LEASES["out"] == 0
        s = gs.stream()
        assert gs.stream() is s and len(made) == 1 and capture_lock.LEASES["out"] == 1 and gs.holding
        gen = capture_lock.GEN["captures"]
        assert gs.drop([_Graph(log, "a"), None, _Graph(log, "b")]) == 2
        assert log == [("reset", "a"), ("reset", "b")]
        assert not gs.holding and gs.dropped and gs.stats == {"drops": 1, "graphs_dropped": 2}
        assert capture_lock.LEASES["out"] == 1 and not capture_lock._FREE, "the stream stays the slot's"
        assert capture_lock.GEN["captures"] == gen, "the caller counts the reset, after its empty"
        gs.used()
        assert not gs.dropped and gs.t_used > 0
        assert gs.stream() is s and len(made) == 1 and gs.holding
    return "one stream for the slot's life; a drop resets each graph once and counts nothing itself"


def test_lru_puts_the_oldest_round_first_and_skips_sets_without_graphs():
    with _streams():
        a, b, c, d = GraphSet(), GraphSet(), GraphSet(), GraphSet()
        for gs, t in ((a, 30.0), (b, 10.0), (c, 20.0), (d, 1.0)):
            gs.stream()
            gs.t_used = t
        d.drop([])                                          # dropped, nothing captured since
        got = [k for k, _ in lru([("a", a), ("b", b), ("c", c), ("d", d), ("e", None)])]
    assert got == ["b", "c", "a"], got
    return "b (10 s), c, a; a set holding nothing is no victim"


def test_every_flash_next_capture_names_a_real_stream():
    """Every `torch.cuda.graph(...)` of the Flash-Next engine passes `stream=` something other than None: a graph
    captured on torch's shared stream beside a slot's would have its cuBLAS workspace dropped by another owner's reset
    (SPD-223)."""
    from engine.models.flashnext import engine as E
    from engine.models.flashnext.ref import FNState
    src = open(os.path.join(ROOT, "engine", "models", "flashnext", "engine.py")).read()
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and ast.unparse(n.func) == "torch.cuda.graph"]
    assert len(calls) == 6, len(calls)                    # SPD-249: T1's packed draft graph, on a leased stream
    for c in calls:
        kw = [k for k in c.keywords if k.arg == "stream"]
        assert kw and not (isinstance(kw[0].value, ast.Constant) and kw[0].value.value is None), ast.unparse(c)
    st = FNState(cap=8)
    assert E.capture_stream(st) is None                    # no owner: torch's stream, as before
    with _streams() as made:
        st.graphs = GraphSet()
        assert E.capture_stream(st) is made[0] and made[0] is not None and st.graphs.holding
    return "6 captures (decode, joint, verify, commit, MTP chain, packed draft) name a stream; an owner's is a leased one"


class _DEng:
    """What `DecodeGraph` reads of the engine: one GDN state and a parked count each step writes."""

    cfg = types.SimpleNamespace(has_indexer=False, indexer_compress_ratio=4)

    def __init__(self):
        self.cnt, self.stack = torch.zeros(1, dtype=torch.int32), None

    def qsa_class(self, n):
        return 0

    def settle(self, st):
        pass

    def _pend_of(self, st):
        return {}, self.cnt

    def _rows_buffers(self, M):
        return {}

    def stage(self, token, st):
        pass

    def step(self, st, cls=0):
        st.S[0].add_(1.0)
        self.cnt += 1
        st.ple_ctx[1] = "written"
        self.stack = torch.zeros(1, 3)


def test_a_failed_decode_capture_leaves_the_stream_and_the_state_as_they_were():
    """Review #5: a decode capture whose end raised left the thread on the slot's stream and the state its warm-ups
    wrote (two steps and the capture's); now the thread is back on its own stream, the GDN state, the parked count,
    the PLE context and the length are put back, and the graph is kept for the slot's drop."""
    from test_fn_verify_capture import _Cuda
    from engine.models.flashnext import engine as E
    from engine.models.flashnext.ref import FNState
    cuda = _Cuda()
    eng, st = _DEng(), FNState(cap=8)
    st.S[0] = torch.zeros(1, 2, 2, 2)
    st.length, st.ple_ctx = 5, {1: "before"}
    st.graphs = GraphSet()
    with cuda.on():
        dg = E.DecodeGraph(eng, st)
        cuda.fail_end = RuntimeError("capture invalidated")
        try:
            dg.capture(0)
            raise AssertionError("the failed capture was not raised")
        except RuntimeError as exc:
            assert "invalidated" in str(exc)
        assert cuda.streams == ["leased"], cuda.streams
        assert cuda.cur is cuda.default, cuda.cur.name
        dg.capture(0)                                       # the next one captures cleanly
    assert float(st.S[0].abs().sum()) == 0 and int(eng.cnt) == 0, (st.S[0], eng.cnt)
    assert st.ple_ctx == {1: "before"} and st.length == 5
    assert len(dg.dead) == 1 and list(dg.graphs) == [0] and dg.dead[0] is not dg.graphs[0]
    return "the thread on its own stream, state, count, PLE context and length back; the graph kept in dead"


def _slot():
    """A Flash-Next slot over test_fn_serving's host engine, with its MTP head."""
    from test_fn_serving import _FN
    from engine.models.flashnext.serving import FlashNextServing
    fn = _FN()
    fn.qsa_class = lambda n: 0

    def prefill(ids, st, keep_stack=True):
        st.length += ids.numel()
        fn.prompt_stack = torch.zeros(ids.numel(), 3)
        return torch.zeros(1, 5)
    fn.prefill = prefill
    s = FlashNextServing(fn, 64)
    s.add_mtp_head(2)
    return fn, s


def test_a_slot_drops_every_graph_it_holds_at_once():
    """The decode step's graphs by class and a failed one, the verify and commit graphs, the head's chains and the
    graphs failed captures kept: each reset once, after the pending chain's commit (its commit graph reads its verify
    graph's records), every dict cleared (the head's `dead` is the adapter's list still), the stream kept."""
    log = []
    with _streams():
        fn, s = _slot()
        head = s.heads["mtp"]
        assert s.st.graphs is s.graph_set and head.st.graphs is s.graph_set and head.dead is s.dead_graphs
        s.graph_set.stream()
        G = lambda name: types.SimpleNamespace(graph=_Graph(log, name))      # noqa: E731
        s.dg = types.SimpleNamespace(graphs={0: _Graph(log, "d0"), 1024: _Graph(log, "d1024")},
                                     dead=[_Graph(log, "d-dead")])
        s.vgraphs = {(3, 0): G("v3")}
        s.cgraphs = {(3, 2, False, 0): G("c3")}
        head.graphs = {(1, 0): G("m1"), (2, 0, "k"): G("m2k")}
        head.fresh = {(1, 0)}
        s.dead_graphs.extend([G("v-dead"), types.SimpleNamespace(graph=None)])
        s._fresh = {(3, 0)}
        assert s.graph_count() == 9
        s.st.length, s._chain, s._width = 10, 2, 3
        commit = fn.commit
        fn.commit = lambda st, path, graphs=None: log.append(("commit", list(path))) or commit(st, path, graphs)
        assert s.drop_graphs() == 8
        assert log[0] == ("commit", [0, 1]), "the pending chain commits first"
        assert sorted(n for k, n in log[1:]) == sorted(["d0", "d1024", "d-dead", "v3", "c3", "m1", "m2k", "v-dead"])
        assert all(k == "reset" for k, _ in log[1:])
        assert s.dg is None and not s.vgraphs and not s.cgraphs and not s.dead_graphs and not s._fresh
        assert not head.graphs and not head.fresh and head.dead is s.dead_graphs
        assert not s.graph_set.holding and capture_lock.LEASES["out"] == 1 and s.graph_count() == 0
        assert s.idle_captures() is None, "a dropped slot waits for its next round"
        s.st.length = 0
        s.forward(torch.tensor([5, 6]), 0)                  # the slot's next turn
        assert not s.graph_set.dropped and s.graph_set.t_used > 0
    return "8 graphs reset once each after the chain's commit; stream kept; idle captures wait for a round"


def test_an_idle_capture_step_holds_the_set_never_across_its_yield():
    from engine.models.flashnext import serving as S
    held = []
    with _streams():
        fn, s = _slot()
        head = s.heads["mtp"]

        class VG:
            def __init__(self, fn_, st, M, cls):
                pass

            def capture(self):
                held.append(s.graph_set.lock.locked())

        head.capture_class = lambda n, cls: held.append(s.graph_set.lock.locked())
        real, S.VerifyGraph = S.VerifyGraph, VG
        sync, torch.cuda.synchronize = torch.cuda.synchronize, lambda: None
        try:
            steps = s._capture_steps(0, [("v", 2), ("m", 1)])
            next(steps)
            assert held == [True] and not s.graph_set.lock.locked()
            assert s.graph_set.lock.acquire(blocking=False)      # a drop between two steps gets it
            s.graph_set.lock.release()
            next(steps)
            assert held == [True, True] and not s.graph_set.lock.locked()
        finally:
            S.VerifyGraph = real
            torch.cuda.synchronize = sync
    return "held inside each step, free at each yield"


def test_a_drop_between_two_idle_steps_ends_the_slots_window():
    """Review #6: the idle window's generator checked `dropped` once, when it was planned; a reclaim on another thread
    between two steps dropped the set and the next step captured into it again. Now each step looks first."""
    from engine.models.flashnext import serving as S
    made = []
    with _streams():
        fn, s = _slot()

        class VG:
            def __init__(self, fn_, st, M, cls):
                self.M, self.graph = M, None

            def capture(self):
                made.append(self.M)
        real, S.VerifyGraph = S.VerifyGraph, VG
        sync, torch.cuda.synchronize = torch.cuda.synchronize, lambda: None
        try:
            steps = s._capture_steps(0, [("v", 2), ("v", 3), ("v", 4)])
            next(steps)
            assert made == [2]
            s.drop_graphs()                                 # a reclaim between the two steps
            assert next(steps, "end") == "end" and made == [2] and not s.vgraphs, made
        finally:
            S.VerifyGraph = real
            torch.cuda.synchronize = sync
    return "1 of 3 captured, then a drop: the window ends, nothing captured into the dropped set"


# ------------------------------------------------------------------ the server's drops

def _app():
    from server import app
    return app


class _Alloc:
    """The allocator's private pools as the server reads them: bytes by pool id. A dropped set's pools are given back
    at the next `empty` unless a live tensor pins them; the walk (`_Mem.pools`) is taken again only after
    `capture_lock.GEN` moved, as `cache.TorchPools`'s; `log` the order of drops, empties and counts."""

    def __init__(self, base: float = 0.0, n: int = 1):
        self.pools = {("load", i): int(base * GiB / n) for i in range(n)} if base else {}
        self.freeable, self.pinned, self.log, self.freed = set(), set(), [], 0

    def total(self) -> int:
        return sum(self.pools.values())

    def empty(self) -> None:
        self.log.append("empty")
        for p in sorted(self.freeable - self.pinned):
            self.freed += self.pools.pop(p, 0)
        self.freeable = set()


class _Mem:
    """The memory policy as the idle window and the cap read it: no peak bound, the graphs' pools from `alloc`'s walk."""

    on, ws_floor, peak = True, 0, 1 << 60

    def __init__(self, alloc: _Alloc):
        self.alloc, self.empties, self._gen, self._walk = alloc, 0, None, None

    def memory(self):
        return 0, 0

    def pools(self):
        gen = capture_lock.GEN["captures"]
        if self._walk is None or gen != self._gen:
            self._gen = gen
            self._walk = {"graphs_reserved": self.alloc.total(), "graphs_pools": len(self.alloc.pools)}
        return self._walk

    def empty(self, why):
        self.empties += 1
        self.alloc.empty()
        return 0


class _SlotEng:
    """A slot's engine whose graph set holds pools of `gib` in `alloc`; it owes `owed` idle captures of 0.25 GiB."""

    def __init__(self, alloc: _Alloc, idx: int, gib: float, t_used: float, owed: int = 0, pools: int = 1):
        self.alloc, self.idx, self.owed, self.k = alloc, idx, owed, 0
        self.pids: list = []
        for _ in range(pools):
            self._pool(gib / pools)
        self.graph_set = GraphSet()
        self.graph_set.stream()
        self.graph_set.t_used = t_used
        self.kv = types.SimpleNamespace(length=1000)

    def _pool(self, gib: float) -> None:
        pid = ("slot", self.idx, self.k)
        self.k += 1
        self.alloc.pools[pid] = int(gib * GiB)
        self.pids.append(pid)

    def graph_pools(self) -> set:
        return set(self.pids)

    def drop_graphs(self):
        self.alloc.log.append(f"drop{self.idx}")
        self.alloc.freeable |= set(self.pids)
        n, self.pids = len(self.pids), []
        return self.graph_set.drop([_Graph([], str(i)) for i in range(n)])

    def graph_count(self):
        return len(self.pids)

    def idle_captures(self, ahead=True, wants=None):
        if not self.owed or self.graph_set.dropped:
            return None
        return self._steps()

    def _steps(self):
        while self.owed:
            self.owed -= 1
            self.graph_set.stream()
            self._pool(0.25)
            capture_lock.graphs_reset()                     # a capture's count (capture_lock.LOCK's)
            yield


@contextlib.contextmanager
def _server(engines: dict, alloc: _Alloc, mem=None, avail=None):
    """STATE with slot 0's engine and side slots 1.. (`engines` index -> engine), the sets' bytes read from `alloc`,
    the KV pool's MemAvailable from `avail()` (None: no pool)."""
    from engine import kvpool
    app = _app()
    slots = [types.SimpleNamespace(index=i, lock=threading.Lock(), phase="", values={"engine": e, "drafter": None})
             for i, e in sorted(engines.items())]
    real = (app._pools_bytes, capture_lock.graphs_reset, kvpool.POOL, app._round_runner.RUNNER, app.LOCK.turn)
    keep = {k: dict.get(app.STATE, k) for k in ("engine", "drafter", "pool", "mem", "last_end_mono", "mem_load")}
    reset = capture_lock.graphs_reset
    try:
        app._pools_bytes = lambda pids: sum(alloc.pools.get(p, 0) for p in pids)
        capture_lock.graphs_reset = lambda: (alloc.log.append("gen"), reset())[1]
        kvpool.POOL = types.SimpleNamespace(meminfo=avail) if avail is not None else None
        app._round_runner.RUNNER = None
        app.LOCK.turn = None
        app.STATE["engine"], app.STATE["drafter"] = engines.get(0), None
        app.STATE["pool"] = types.SimpleNamespace(side=lambda: slots[1:], slots=slots)
        app.STATE["mem"] = mem
        app.STATE["last_end_mono"] = time.monotonic() - 5.0
        app.STATE["mem_load"] = {"pools": {"graphs": {"reserved": sum(v for k, v in alloc.pools.items()
                                                                     if k[0] == "load")}}}
        app.STATE.pop("draining", None)
        app.STATE.pop("graphs_cap_said", None)
        yield app, {s.index: s for s in slots}
    finally:
        (app._pools_bytes, capture_lock.graphs_reset, kvpool.POOL, app._round_runner.RUNNER, app.LOCK.turn) = real
        for k, v in keep.items():
            if v is None:
                app.STATE.pop(k, None)
            else:
                app.STATE[k] = v
        app.STATE.pop("graphs_cap_said", None)


def test_the_reclaim_drops_idle_slots_least_recently_used_first():
    """Slots 0-4 hold 1 GiB of graphs each; their last rounds: slot 2 the oldest, then 0, 4, 3, 1. Slot 1 is busy
    and slot 3 asks: a reclaim of 1.5 GiB takes slot 2's and slot 0's (one line each, the bytes their own pools gave
    back at the allocator's empty, the reset counted after it), a large one then slot 4's and never slots 1 or 3. A
    set an idle capture step holds, and every set while a capture runs on another thread, is skipped."""
    with _streams():
        alloc = _Alloc()
        e = {i: _SlotEng(alloc, i, 1.0, t) for i, t in ((0, 20.0), (1, 50.0), (2, 10.0), (3, 40.0), (4, 30.0))}
        with _server(e, alloc, _Mem(alloc)) as (app, slots):
            slots[1].lock.acquire()                             # slot 1 decodes
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                assert app._kv_reclaim_graphs(int(1.5 * GiB), "slot3") == 2 * GiB
            assert out.getvalue().splitlines() == ["[graphs] dropped slot2: 1 graphs, 1.00 GiB back (why=reclaim)",
                                                   "[graphs] dropped slot0: 1 graphs, 1.00 GiB back (why=reclaim)"]
            assert alloc.log == ["drop2", "empty", "gen", "drop0", "empty", "gen"], alloc.log
            with contextlib.redirect_stdout(io.StringIO()):
                assert app._kv_reclaim_graphs(100 * GiB, "slot3") == GiB
            assert [i for i, x in e.items() if x.graph_set.dropped] == [0, 2, 4]
            assert not app.LOCK.locked() and not slots[4].lock.locked() and slots[1].lock.locked()
            slots[1].lock.release()
            e[1].graph_set.lock.acquire()                       # an idle capture step on slot 1
            assert app._kv_reclaim_graphs(100 * GiB, "slot3") == 0 and not e[1].graph_set.dropped
            e[1].graph_set.lock.release()
            got = []
            with capture_lock.GUARD:                            # a capture on another thread
                t = threading.Thread(target=lambda: got.append(app._kv_reclaim_graphs(100 * GiB, "slot3")))
                t.start()
                t.join()
            assert got == [0] and not e[1].graph_set.dropped and not e[3].graph_set.dropped
            rep = app._graphs_report()
            by = {r["slot"]: r for r in rep["slots"]}
            assert by[2]["graphs"] == 0 and by[2]["dropped"] and not by[2]["holding"] and by[1]["graphs"] == 1, by
            assert rep["cap_gib"] == app.GRAPH_POOL_GIB and rep["reclaim"] is app.RECLAIM_GRAPHS
    return "LRU 2, 0, then 4 (drop, empty, count each); never busy 1 or asking 3; skipped under a step or a capture"


def test_a_drop_waits_for_a_request_still_on_the_slot():
    """Review #4: `_runner_decode`'s request may give its slot back (`_par_end`) before the round runner saw its task
    cancelled, so the slot's lock is free while a round may still replay its graphs. No drop while the slot is in a
    phase or the runner holds an unfinished task over its engine; slot 0 also only while the GPU turn is free."""
    with _streams():
        alloc = _Alloc()
        e = {i: _SlotEng(alloc, i, 1.0, float(i)) for i in range(3)}
        with _server(e, alloc, _Mem(alloc)) as (app, slots):
            slots[1].phase = "runner"
            task = types.SimpleNamespace(eng=e[2], done=False)
            app._round_runner.RUNNER = types.SimpleNamespace(tasks=[task])

            class Turn:
                free, taken = False, []

                def try_acquire(self, who):
                    self.taken.append(who)
                    return self.free

                def release(self, who):
                    self.taken.append(-who - 1)
            app.LOCK.turn = turn = Turn()
            with contextlib.redirect_stdout(io.StringIO()):
                assert app._kv_reclaim_graphs(100 * GiB, "slot9") == 0
                assert not any(x.graph_set.dropped for x in e.values()) and turn.taken == [0]
                slots[1].phase, task.done, turn.free = "", True, True
                assert app._kv_reclaim_graphs(100 * GiB, "slot9") == 3 * GiB
            assert all(x.graph_set.dropped for x in e.values()) and turn.taken == [0, 0, -1], turn.taken
            assert not app.LOCK.locked()
    return "a phase, a runner task or a held turn (slot 0): no drop; all clear: dropped, the turn given back"


def test_the_reclaim_stops_at_the_pools_reading_and_after_two_empty_drops():
    """Review #3: the reserve's global delta read 0 while other slots allocated, so the loop dropped every slot. Now
    a drop counts its own pools, the loop stops once the KV pool's own reading (MemAvailable) is up by `want`, and
    after two drops that gave nothing (their pools pinned) it drops no more."""
    with _streams():
        alloc = _Alloc()
        e = {i: _SlotEng(alloc, i, 1.0, float(i)) for i in range(4)}
        extra = [0]
        with _server(e, alloc, _Mem(alloc), avail=lambda: 10 * GiB + alloc.freed + extra[0]) as (app, _):
            with contextlib.redirect_stdout(io.StringIO()):
                extra[0] = 0
                real = app._drop_slot_graphs

                def drop(*h):
                    n = real(*h)
                    extra[0] = 5 * GiB                      # another slot's memory came back meanwhile
                    return n
                app._drop_slot_graphs = drop
                try:
                    assert app._kv_reclaim_graphs(3 * GiB, "slot9") == GiB
                finally:
                    app._drop_slot_graphs = real
                assert [i for i, x in e.items() if x.graph_set.dropped] == [0], "the pool's reading was enough"
                extra[0] = 0
                alloc.pinned = set(e[1].pids) | set(e[2].pids)     # a live tensor holds their pools
                assert app._kv_reclaim_graphs(3 * GiB, "slot9") == 0
        assert [i for i, x in e.items() if x.graph_set.dropped] == [0, 1, 2], "two drops gave nothing: slot 3 kept"
    return "stopped at the pool's reading after 1 of 3 GiB; two pinned sets gave nothing: the third kept"


def test_the_ladder_drops_graphs_before_any_conversation():
    from engine import kvpool
    app = _app()
    kp = kvpool.KVPool(hard_floor=0, meminfo=lambda: 100 * GiB)
    keep = app.RECLAIM_GRAPHS
    try:
        app.RECLAIM_GRAPHS = True
        app._add_kv_reclaimers(kp)
        on = [n for n, _ in kp.reclaimers]
        app.RECLAIM_GRAPHS = False
        kp.reclaimers = []
        app._add_kv_reclaimers(kp)
        off = [n for n, _ in kp.reclaimers]
    finally:
        app.RECLAIM_GRAPHS = keep
    assert on == ["caches", "roles", "graphs", "written_side", "idle_side", "idle_slot0", "unwritten_side",
                  "kept_park"], on
    assert off == [n for n in on if n != "graphs"]
    return "caches, roles, graphs, then the conversations; QSE_RECLAIM_GRAPHS=off leaves it out"


def test_the_cap_counts_growth_drops_to_its_low_water_and_does_not_churn():
    """Review #1, #2: the load's pools (1 GiB) are not growth. Growth 2.5 GiB over a 2 GiB cap: the window drops the
    oldest idle set (slot 1, 1 GiB) and is at 1.5, the low water (3/4 of the cap); slot 0 keeps its set. It then
    captures while one more graph (the walk's average, 0.25 GiB) stays under the cap: two of slot 0's four, one
    "paused" line. The next window drops nothing and captures nothing: no churn."""
    with _streams():
        now = time.monotonic()
        alloc = _Alloc(base=1.0, n=4)                      # every pool here 0.25 GiB: the walk's average
        e = {0: _SlotEng(alloc, 0, 1.0, now - 1000, owed=4, pools=4), 1: _SlotEng(alloc, 1, 1.0, now - 2000, pools=4),
             2: _SlotEng(alloc, 2, 0.5, 0.0, pools=2)}
        e[2].graph_set.drop([])                             # dropped and nothing since: 0.5 GiB that stays pinned
        alloc.pinned = set(e[2].pids)
        mem = _Mem(alloc)
        app = _app()
        keep = app.GRAPH_POOL_GIB
        out = io.StringIO()
        try:
            app.GRAPH_POOL_GIB = 2.0
            with _server(e, alloc, mem) as (app, _), contextlib.redirect_stdout(out):
                assert app._graphs_growth()[0] == int(2.5 * GiB)
                got = app.graphs_once()
                lines = out.getvalue().splitlines()
                assert [ln for ln in lines if "dropped" in ln] == [
                    "[graphs] dropped slot1: 4 graphs, 1.00 GiB back (why=cap)"], lines
                assert got == "cap" and e[0].owed == 2 and not e[0].graph_set.dropped, (got, e[0].owed, lines)
                assert app._graphs_growth()[0] == 2 * GiB
                assert sum(ln.startswith("[graphs] idle captures paused") for ln in lines) == 1, lines
                assert app.graphs_once() == "cap" and e[0].owed == 2
                assert sum("dropped" in ln for ln in out.getvalue().splitlines()) == 1, "the next window drops nothing"
                assert not app.LOCK.locked()
        finally:
            app.GRAPH_POOL_GIB = keep
    return "2.5 GiB growth over 2: slot 1 dropped to 1.5; 2 captures up to the cap; the next window: nothing"


def test_the_cap_never_takes_a_recent_set_and_stops_at_an_empty_drop():
    """A set that served in the last GRAPH_SET_MIN_IDLE_S stays whatever the growth (the reclaim may take it under
    pressure); a drop that gives nothing back ends the loop (its pools pinned); `QSE_RECLAIM_GRAPHS=off` drops
    nothing at the cap."""
    with _streams():
        now = time.monotonic()
        alloc = _Alloc()
        e = {0: _SlotEng(alloc, 0, 3.0, now - 30), 1: _SlotEng(alloc, 1, 3.0, now - 1000),
             2: _SlotEng(alloc, 2, 3.0, now - 2000)}
        app = _app()
        keep = app.GRAPH_POOL_GIB, app.RECLAIM_GRAPHS
        try:
            app.GRAPH_POOL_GIB = 2.0
            with _server(e, alloc, _Mem(alloc)) as (app, _), contextlib.redirect_stdout(io.StringIO()):
                app.RECLAIM_GRAPHS = False
                assert app._graphs_cap() == 0 and not any(x.graph_set.dropped for x in e.values())
                app.RECLAIM_GRAPHS = True
                alloc.pinned = set(e[2].pids)
                assert app._graphs_cap() == 0
                assert e[2].graph_set.dropped and not e[1].graph_set.dropped, "an empty drop ends the loop"
                alloc.pinned = set()
                assert app._graphs_cap() == 3 * GiB and e[1].graph_set.dropped
                assert not e[0].graph_set.dropped, "slot 0 served 30 s ago"
                assert app._kv_reclaim_graphs(GiB, "slot9") == 3 * GiB and e[0].graph_set.dropped, \
                    "the reclaim takes it"
        finally:
            app.GRAPH_POOL_GIB, app.RECLAIM_GRAPHS = keep
    return "off: nothing; a pinned drop ends the loop; a set used 30 s ago stays at the cap, the reclaim takes it"


def test_no_cap_and_no_reading_drop_nothing():
    with _streams():
        alloc = _Alloc()
        e = {0: _SlotEng(alloc, 0, 30.0, 1.0)}
        app = _app()
        keep = app.GRAPH_POOL_GIB
        try:
            with _server(e, alloc, _Mem(alloc)) as (app, _):
                app.GRAPH_POOL_GIB = 0.0
                assert app._graphs_over_cap() == 0 and app.graphs_once() == "nothing"
                app.GRAPH_POOL_GIB = 8.0
                app.STATE["mem"] = types.SimpleNamespace(on=True, pools=lambda: None, ws_floor=0, peak=1 << 60,
                                                         memory=lambda: (0, 0))
                assert app._graphs_over_cap() == 0 and app.graphs_once() == "nothing"
                assert not e[0].graph_set.dropped
        finally:
            app.GRAPH_POOL_GIB = keep
    return "cap 0, or no pools probe: no drop"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<78} ok   {fn() or ''}")
        except Exception as exc:                                   # noqa: BLE001
            fails += 1
            import traceback
            traceback.print_exc()
            print(f"  {name:<78} FAIL ({exc})")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
