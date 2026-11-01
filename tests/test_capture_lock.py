"""The lock between CUDA graph captures and the disk tier's writer, and a head whose chunks failed.

THE INCIDENT (2026-10-02 21:09, the first request after the restart; engine log and
results/disk/live-20261002210930.json). The disk tier's writer thread copied the session head's
rows out of the engine's buffers while the drafter captured its CUDA graph on the engine thread.
The copy failed (`cudaErrorStreamCaptureUnsupported`) after 1 of the head's 15 chunks, the
capture failed with it (`info.status != cudaStreamCaptureStatusInvalidated` in the caching
allocator) and the request ended in an error. The tier held 4 anchors and 1 chunk: no anchor's
chunk chain was whole, so the live check's next session found nothing on disk and prefilled its
17,074 tokens cold (22.5 s). And `plan` counted those anchors as written, so no later request was
ever admitted to write the missing chunks.

  1  every CUDA graph capture in engine/ and server/ holds `engine.capture_lock.LOCK` (the source,
     every `with torch.cuda.graph(...)`), and the drafter's capture does when it runs
  2  the writer copies nothing while a capture holds the lock, and finishes the job after it
  3  a head whose chunks failed part way is admitted again by the next session's first request,
     the missing chunks are written, and the session after it starts from the disk, bit for bit
     a cold prefill

  4  SPD-257: under expandable segments a capture runs with the allocator on plain segments (the rest of the
     setting kept) and the setting goes back when the capture lets go, once for nested holds; `GUARD`,
     `off_thread`, the live view's marks and the GPU health probe take the same lock without the switch; a
     setting torch refuses turns the switch off; without expandable segments nothing is switched
                                                  test_a_capture_runs_on_plain_segments
     a capture counts once (`GEN`) when its outermost hold lets go, a reset outside a capture once
                                                  test_a_capture_counts_once_when_its_outermost_hold_lets_go

The GPU side is tools/capture_race_check.py (a hold step).

Run: python tests/test_capture_lock.py
"""

from __future__ import annotations

import ast
import os
import sys
import tempfile
import types

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")

import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from engine import cache, capture_lock, disk_cache  # noqa: E402
from test_cache import DEV, fresh, tokens  # noqa: E402
from test_disk_cache import CH, Pos, exact, run, tier  # noqa: E402

TAG = {"client": "opencode", "head_end": 100}



# ------------------------------------------------------------------ SPD-257: a capture's pool on plain segments

def test_a_capture_runs_on_plain_segments():
    """Each expandable segment reserves ~130 GiB of address space and each graph captured without a shared pool owns a
    pool: the address space grew from 59 to 109 TiB in 20 minutes on a test server. A capture's hold switches the
    allocator to plain segments and back; nothing else that takes the lock switches it."""
    import threading
    calls = []
    keep = (dict(capture_lock.ALLOC), capture_lock._set_allocator, os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))

    def fresh_alloc(conf):
        capture_lock.ALLOC.update(on=None, conf="", plain="", toggles=0, errors=0, error=None)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf
    try:
        capture_lock._set_allocator = calls.append
        fresh_alloc("max_split_size_mb:512,expandable_segments:True")
        with capture_lock.LOCK:
            assert calls == ["max_split_size_mb:512,expandable_segments:False"], calls
            with capture_lock.LOCK:                    # nested: switched once
                assert capture_lock.held()
            assert len(calls) == 1
        assert calls[-1] == "max_split_size_mb:512,expandable_segments:True" and len(calls) == 2, calls
        with capture_lock.GUARD:
            pass
        with capture_lock.off_thread("test.plain"):
            pass
        t = threading.Thread(target=lambda: capture_lock.GUARD.acquire(blocking=False)
                             and capture_lock.GUARD.release())
        t.start()
        t.join()
        assert len(calls) == 2 and capture_lock.ALLOC["toggles"] == 2, calls
        # a capture on another thread holds the lock: GUARD waits for it as before
        got = []
        with capture_lock.LOCK:
            t = threading.Thread(target=lambda: got.append(capture_lock.GUARD.acquire(blocking=False)))
            t.start()
            t.join()
        assert got == [False], got

        def refuse(setting):
            raise RuntimeError("expandable_segments cannot change now")
        capture_lock._set_allocator = refuse
        fresh_alloc("expandable_segments:True")
        with capture_lock.LOCK:
            pass
        assert capture_lock.ALLOC["on"] is False and capture_lock.ALLOC["errors"] == 1, capture_lock.ALLOC
        capture_lock._set_allocator = calls.append
        n = len(calls)
        fresh_alloc("max_split_size_mb:512")           # no expandable segments: nothing to switch
        with capture_lock.LOCK:
            pass
        assert len(calls) == n and capture_lock.ALLOC["on"] is False
    finally:
        capture_lock.ALLOC.clear()
        capture_lock.ALLOC.update(keep[0])
        capture_lock._set_allocator = keep[1]
        if keep[2] is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = keep[2]
    return "plain while a capture holds the lock, back after; GUARD and off_thread switch nothing"


def test_a_capture_counts_once_when_its_outermost_hold_lets_go():
    """SPD-257: the memory policy walks the allocator's segments again only when `GEN` moved. A capture counts once,
    when its outermost hold of LOCK lets go (a nested hold does not count, and nothing counts while it holds);
    GUARD and off_thread count nothing; a reset outside a capture counts (`graphs_reset`: a joint graph cache's
    eviction, the 27B's flush, each of which calls it)."""
    from engine.models.graph_lru import GraphLRU
    gen = capture_lock.GEN
    g0 = gen["captures"]
    with capture_lock.LOCK:
        with capture_lock.LOCK:
            pass
        assert gen["captures"] == g0, "counted while the capture still holds the lock"
    assert gen["captures"] == g0 + 1, gen
    with capture_lock.GUARD:
        pass
    with capture_lock.off_thread("test.gen"):
        pass
    assert gen["captures"] == g0 + 1, "GUARD or off_thread counted"
    capture_lock.graphs_reset()
    assert gen["captures"] == g0 + 2

    class _G:
        def __init__(self):
            self.graph = types.SimpleNamespace(reset=lambda: None)
    lru = GraphLRU(cap=1)
    lru.take("a", _G)
    assert gen["captures"] == g0 + 2, "a take that evicts nothing"
    lru.take("b", _G)
    assert gen["captures"] == g0 + 3 and lru.evicted == 1, gen
    tree = ast.parse(open(os.path.join(ROOT, "engine/models/qwen3_5/joint_graphs.py")).read())
    flush = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_flush")
    assert any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "graphs_reset"
               for c in ast.walk(flush)), "the 27B's joint graphs' flush does not count its resets"
    return "a capture once (nested too), GUARD none, a reset and an eviction once each"


# ------------------------------------------------------------------ SPD-249: the guard for CUDA work off the engine's thread

def test_off_thread_waits_for_an_open_capture():
    """A thread entering `off_thread` while another holds LOCK (a capture) runs only after the capture let go."""
    import threading
    import time
    order = []
    opened = threading.Event()

    def capture():
        with capture_lock.LOCK:
            opened.set()
            time.sleep(0.3)
            order.append("capture done")

    t = threading.Thread(target=capture)
    t.start()
    opened.wait()
    with capture_lock.off_thread("test.reader"):
        order.append("reader ran")
    t.join()
    assert order == ["capture done", "reader ran"], order
    assert capture_lock.SITES.get("test.reader", 0) >= 1
    return "the reader waited for the capture"


def test_the_debug_check_fails_off_thread_without_the_guard():
    """QSE_CAPTURE_GUARD_DEBUG: `check` raises outside `off_thread` and passes inside it or inside a capture's LOCK;
    off the switch it does nothing."""
    real = capture_lock.DEBUG
    capture_lock.DEBUG = True
    try:
        try:
            capture_lock.check("test.site")
            raise AssertionError("no CaptureGuardError outside the guard")
        except capture_lock.CaptureGuardError:
            pass
        with capture_lock.off_thread("test.site"):
            capture_lock.check("test.site")
        with capture_lock.LOCK:
            capture_lock.check("test.site")
        assert not capture_lock.held()
    finally:
        capture_lock.DEBUG = real
    capture_lock.DEBUG = False
    try:
        capture_lock.check("test.site")
    finally:
        capture_lock.DEBUG = real
    return "raises unguarded (debug), passes under off_thread and under a capture's LOCK; silent off debug"


#: SPD-249: CUDA calls that run on a thread that is not the engine's (file, function, site): each holds
#: `off_thread(site)` around its CUDA call. A spy over the served engine (every CUDA entry point called from another
#: thread while a capture was open, 561 captures under a 5 Hz /health and 1 Hz /metrics poll) found these.
OFF_THREAD_SITES = [
    ("server/metrics.py", "_unified_free", "metrics.unified_free"),
]


def test_every_off_thread_cuda_site_holds_the_guard():
    """Each listed site's CUDA call sits inside `with capture_lock.off_thread("<site>")`, with `check("<site>")`
    inside it (the debug run names a site entered without it)."""
    for path, func, site in OFF_THREAD_SITES:
        tree = ast.parse(open(os.path.join(ROOT, path)).read())
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
        guarded = [w for w in ast.walk(fn) if isinstance(w, ast.With) and any(
            isinstance(i.context_expr, ast.Call) and isinstance(i.context_expr.func, ast.Attribute)
            and i.context_expr.func.attr == "off_thread" and i.context_expr.args
            and getattr(i.context_expr.args[0], "value", None) == site for i in w.items)]
        assert guarded, f"{path}:{func} has no off_thread({site!r})"
        calls = [c for w in guarded for c in ast.walk(w) if isinstance(c, ast.Call)
                 and isinstance(c.func, ast.Attribute) and c.func.attr == "check"]
        assert any(getattr(c.args[0], "value", None) == site for c in calls if c.args), f"{path}:{func}: no check"
        cuda = [c for c in ast.walk(fn) if isinstance(c, ast.Attribute) and isinstance(c.value, ast.Attribute)
                and c.value.attr == "cuda"]
        inside = {id(c) for w in guarded for c in ast.walk(w)}
        assert cuda and all(id(c) in inside for c in cuda), f"{path}:{func}: a CUDA call outside the guard"
    return f"{len(OFF_THREAD_SITES)} site(s): the CUDA call inside off_thread, checked"


def test_a_metrics_scrape_waits_for_an_open_capture():
    """/metrics' free-memory reading (`_unified_free`, a driver call) runs only after a capture let go."""
    import threading
    import time
    from server import metrics
    order, opened = [], threading.Event()
    real = torch.cuda.mem_get_info
    torch.cuda.mem_get_info = lambda *a: (order.append("driver call"), (7.0, 9.0))[1]

    def capture():
        with capture_lock.LOCK:
            opened.set()
            time.sleep(0.3)
            order.append("capture done")
    try:
        t = threading.Thread(target=capture)
        t.start()
        opened.wait()
        got = metrics._unified_free()
        t.join()
    finally:
        torch.cuda.mem_get_info = real
    assert got == 7.0 and order == ["capture done", "driver call"], (got, order)
    return "the scrape's driver call came after the capture"


# ------------------------------------------------------------------ 1. every capture holds it

def _is_graph_call(node) -> bool:
    f = node.func if isinstance(node, ast.Call) else None
    return (isinstance(f, ast.Attribute) and f.attr == "graph"
            and isinstance(f.value, ast.Attribute) and f.value.attr == "cuda")


def _is_lock(node) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "LOCK"
            and isinstance(node.value, ast.Name) and node.value.id == "capture_lock")


def test_every_capture_in_the_engine_and_the_server_holds_the_lock():
    found = []
    for top in ("engine", "server"):
        for dirpath, _, files in os.walk(os.path.join(ROOT, top)):
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(dirpath, f)
                tree = ast.parse(open(path).read(), path)
                for node in ast.walk(tree):
                    if not isinstance(node, ast.With):
                        continue
                    exprs = [it.context_expr for it in node.items]
                    if any(_is_graph_call(e) for e in exprs):
                        rel = os.path.relpath(path, ROOT)
                        found.append(rel)
                        assert any(_is_lock(e) for e in exprs), (
                            f"{rel}:{node.lineno}: a CUDA graph capture without "
                            f"capture_lock.LOCK in the same `with`")
    # N2: the narrator's decode step (engine/generative/qwen35.py) is the third
    # draft_graph.py twice: the lone draft graph and the packed draft of parallel requests
    # ENG-221: Flash-Next's decode step (engine/models/flashnext/engine.py DecodeGraph); ENG-227/153:
    # its verify block (VerifyGraph), its commit (CommitGraph) and its MTP chain draft (MTPChainGraph)
    # ENG-287: LFM2's decode step, joint decode step, verify block and commit (engine/models/lfm2_moe/engine.py)
    # ENG-319: NemotronH's decode step, joint decode step, verify block and commit; SPD-123: the LiLiCorr draft;
    # SPD-233: the LiLiCorr draft of several lanes in one pass (lilicorr.py _PackedGraph)
    # ENG-399: Flint's decode step, joint decode step, verify block and commit (engine/models/flint/engine.py)
    # SPD-163: the 27B's joint verify pass (engine/models/qwen3_5/joint_graphs.py JointGraphs)
    # SPD-249: Flash-Next's joint verify pass (engine/models/flashnext/engine.py `_jg_capture`)
    assert sorted(found) == ["engine/drafters/draft_graph.py", "engine/drafters/draft_graph.py",
                             "engine/drafters/lilicorr.py", "engine/drafters/lilicorr.py",
                             "engine/generative/qwen35.py"] + [
                             "engine/models/flashnext/engine.py"] * 5 + ["engine/models/flint/engine.py"] * 4 + [
                             "engine/models/lfm2_moe/engine.py"] * 4 + [
                             "engine/models/nemotron_h/engine.py"] * 4 + [
                             "engine/models/qwen3_5/joint_graphs.py", "engine/verify_graph.py"], found
    return f"{len(found)} capture sites, each under the lock"


def test_the_drafters_capture_runs_with_the_lock_held():
    from engine.drafters import draft_graph as dg
    seen = []

    class Graph:
        def replay(self):
            seen.append(("replay", capture_lock.LOCK._is_owned()))

    class Capture:
        def __init__(self, g, stream=None, pool=None):
            pass

        def __enter__(self):
            seen.append(("capture", capture_lock.LOCK._is_owned()))

        def __exit__(self, *exc):
            return False

    class Stream:
        def wait_stream(self, other):
            pass

    class On:
        def __init__(self, stream):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    saved = {k: getattr(torch.cuda, k) for k in ("graph", "CUDAGraph", "stream", "current_stream")}
    torch.cuda.graph, torch.cuda.CUDAGraph = Capture, Graph
    torch.cuda.stream, torch.cuda.current_stream = On, Stream
    try:
        g = dg.DraftGraph.__new__(dg.DraftGraph)
        g.d = types.SimpleNamespace(ctx_len=10, tree_temp=1.0)
        g.host, g.scal = torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long)
        g.win, g.graphs, g.stream = 2048, {}, Stream()
        g.stats = {"captured": 0, "replayed": 0}
        g._body = lambda span, temp: ("cand", "scores")
        g.run(5, 9)
        g.run(6, 10)                                  # the same class: a replay only
    finally:
        for k, v in saved.items():
            setattr(torch.cuda, k, v)
    assert seen == [("capture", True), ("replay", False), ("replay", False)], seen
    assert not capture_lock.LOCK._is_owned()


def test_a_capture_stream_has_one_owner_at_a_time():
    """SPD-223: torch drops a capture stream's cuBLAS workspace when any graph captured on it is reset, so two
    graph owners must never hold one stream (a shared pooled stream left a kept draft graph bound to a freed
    workspace). The lease hands a stream to one owner until it is given back, then hands it on."""
    made = []
    saved = capture_lock._new_stream, list(capture_lock._FREE), dict(capture_lock.LEASES)
    capture_lock._new_stream = lambda: made.append(object()) or made[-1]
    capture_lock._FREE.clear()
    try:
        a, b, c = capture_lock.lease_stream(), capture_lock.lease_stream(), capture_lock.lease_stream()
        assert len({id(a), id(b), id(c)}) == 3 and len(made) == 3
        capture_lock.give_back(b)
        d = capture_lock.lease_stream()
        assert d is b and len(made) == 3, "a given-back stream serves the next owner"
        e = capture_lock.lease_stream()
        assert e not in (a, c, d) and len(made) == 4
        capture_lock.give_back(None)
    finally:
        capture_lock._new_stream = saved[0]
        capture_lock._FREE[:] = saved[1]
        capture_lock.LEASES.clear()
        capture_lock.LEASES.update(saved[2])


# ------------------------------------------------------------------ 2. the writer waits

def test_the_writer_copies_nothing_while_a_capture_holds_the_lock():
    head = tokens(100, seed=31)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d)
        with capture_lock.LOCK:                       # the engine thread captures a graph
            cache.prefill(fresh(), Pos(), head, DEV, chunk=CH, disk=t, conv_id="ses_c",
                          disk_tag=TAG)
            assert not t.flush(timeout=0.5), "the writer went on inside the capture"
            assert not t.chunks
        assert t.flush()
        rows = t.report()["top"]
        assert rows and all(r["complete"] for r in rows) and len(t.chunks) == 6, rows
        t.close()


# ------------------------------------------------------------------ 3. a head that failed part way

def test_a_head_whose_chunks_failed_is_admitted_again_and_then_read_bit_for_bit():
    head = tokens(100, seed=21)
    s1, s2, s3 = (head + tokens(30, seed=k) for k in (22, 23, 24))
    with tempfile.TemporaryDirectory() as d:
        t = tier(d)
        real, n = disk_cache.write_file, {"chunks": 0}

        def failing(path, parts):
            if path.endswith(".chk"):
                n["chunks"] += 1
                if n["chunks"] > 1:
                    raise RuntimeError("CUDA error: operation not permitted when stream is "
                                       "capturing")
            return real(path, parts)

        disk_cache.write_file = failing
        try:
            run(fresh(), s1, t, Pos(), conv_id="ses_1", disk_tag=TAG)
        finally:
            disk_cache.write_file = real
        rows = t.report()["top"]
        assert rows and not any(r["complete"] for r in rows) and len(t.chunks) == 1, rows
        assert any(e["kind"] == "error" for e in t.report()["events"])
        # the next session's first request: nothing whole to read, so it prefills cold, and its
        # head is admitted again although its anchors are on disk; the chunks they lack go too
        _, reused, kind = run(fresh(), s2, t, Pos(), conv_id="ses_2", disk_tag=TAG)
        assert kind is None and reused == 0, kind
        rows = t.report()["top"]
        assert all(r["complete"] for r in rows) and len(t.chunks) == 6, rows
        eng, dr = fresh(), Pos()
        lg, reused, kind = run(eng, s3, t, dr, conv_id="ses_3", disk_tag=TAG)
        assert (kind, reused) == ("disk", 96), (kind, reused)
        exact(eng, lg, s3, dr)
        t.close()


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:75s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
