"""The roles' pool sizes without torch 2.13.0's reference leak (the bool_dealloc crash of 2026-10-05).

:8000 died at 13:12, three hours after the start: "Fatal Python error: bool_dealloc: deallocating
True or False". torch 2.13.0's `torch._C._cuda_memorySnapshot` takes one reference too few on every
item of the tuple it is handed, so each `MemPool.snapshot()` takes one reference off `True` and off
each of the pool's id ints. `Placement.pool_bytes` called it for the narrator's and the shadow's
pools on every `Budget.memory()`, which every `/metrics` collector reaches through `cache_stats()`
(30 a render since 0.4.0's `role_exclude`). The matrix:

  1  the upstream call as measured: N calls of `MemPool.snapshot()` take N references off `True`
     and off the id ints; the call with `None` takes none (passes, saying so, on a torch that no
     longer leaks)                                                test_the_upstream_snapshot_leaks
  2  `Placement.pool_bytes` many times over keeps every reference (failed before the fix: -N on
     `True`, -N on each id int)                                   test_pool_bytes_keeps_the_references
  3  the pool's segments are the full snapshot's with its `segment_pool_id`, summed; one snapshot
     serves the calls inside `SNAPSHOT_TTL_S`, never a tuple      test_pool_bytes_from_the_full_snapshot
                                                                  test_pool_bytes_sums_its_own_segments
  4  the watch: a count under half its high is logged once per interval with its fall rate and
     the time to 0; a steady count is not; 3.12+ starts nothing   test_the_refwatch
  5  `/health`'s memory part carries the counts      test_health_memory_has_the_counts
  6  SPD-257: the memory policy's walk of the allocator (`cache.TorchPools`, after each capture) and
     `/v1/cache/memory?pools=1` (`server.app._allocator_pools`) keep every reference over N walks (with
     `torch.cuda.memory_snapshot()`, which hands the call `(0, 0, True)`: -N on `True`, -2N on 0)
                                                  test_the_memory_snapshot_call_leaks_too
                                                  test_the_policys_walk_keeps_the_references
                                                  test_the_pools_route_keeps_the_references
  7  no call in engine/ or server/ hands the snapshot a tuple: `torch.cuda.memory_snapshot()`,
     a pool's `.snapshot()` and `_cuda_memorySnapshot` with anything but `None` are refused
                                                  test_no_snapshot_call_that_drops_references

CPU only: a CUDA build of torch answers `_cuda_memorySnapshot` without a device (empty segments);
run with CUDA_VISIBLE_DEVICES= on the box. Run: python tests/test_pool_snapshot_refcount.py
"""

from __future__ import annotations

import ast
import gc
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from engine.decision import device  # noqa: E402
from server import refwatch  # noqa: E402

N = 2000
POOL_ID = (0, 7)        # a MemPool's id is (0, k): small ints, singletons like True on 3.11


class _Pool:
    """A `torch.cuda.MemPool` without a device: its id, and its own Python `snapshot` method (the
    call the engine made before the fix)."""
    id = POOL_ID
    snapshot = getattr(getattr(torch.cuda, "MemPool", None), "snapshot", None)


def _has_snapshot() -> bool:
    return hasattr(torch._C, "_cuda_memorySnapshot")


def _refs() -> dict:
    gc.collect()
    return {"True": sys.getrefcount(True), "False": sys.getrefcount(False),
            "0": sys.getrefcount(POOL_ID[0]), "7": sys.getrefcount(POOL_ID[1])}


def _delta(fn, n: int = N) -> dict:
    before = _refs()
    for _ in range(n):
        fn()
    after = _refs()
    return {k: after[k] - before[k] for k in before}


def test_the_upstream_snapshot_leaks():
    if not _has_snapshot():
        print("    (torch without CUDA: no _cuda_memorySnapshot, skipped)")
        return
    d = _delta(lambda: _Pool().snapshot())
    if d["True"] == 0:
        print(f"    (this torch {torch.__version__} keeps the references: fixed upstream)")
    else:
        # torch 2.13.0+cu130 on the box: exactly one reference a call off True and each id int
        assert d["True"] == -N and d["7"] == -N, d
    assert all(v == 0 for v in _delta(lambda: torch._C._cuda_memorySnapshot(None)).values())


def test_pool_bytes_keeps_the_references():
    if not _has_snapshot():
        print("    (torch without CUDA: no _cuda_memorySnapshot, skipped)")
        return
    p = object.__new__(device.Placement)
    p.pool = _Pool()
    keep = getattr(device, "SNAPSHOT_TTL_S", None)
    device.SNAPSHOT_TTL_S = 0.0          # every call reaches torch
    try:
        assert p.pool_bytes() == (0, 0)  # no device: no segments
        d = _delta(p.pool_bytes)
    finally:
        device.SNAPSHOT_TTL_S = keep
    assert all(v == 0 for v in d.values()), f"pool_bytes dropped references: {d}"


def test_pool_bytes_from_the_full_snapshot():
    calls = []
    segs = [{"segment_pool_id": (0, 7), "total_size": 4 << 20, "allocated_size": 1 << 20},
            {"segment_pool_id": (0, 7), "total_size": 2 << 20, "allocated_size": 2 << 20},
            {"segment_pool_id": (0, 8), "total_size": 64 << 20, "allocated_size": 9 << 20},
            {"segment_pool_id": (0, 0), "total_size": 1 << 30, "allocated_size": 1 << 30}]

    def fake(arg):
        calls.append(arg)
        return {"segments": segs}

    real = getattr(torch._C, "_cuda_memorySnapshot", None)
    torch._C._cuda_memorySnapshot = fake
    t = [100.0]
    try:
        with device._SNAP_LOCK:
            device._SNAP.update(t=None, segments=None)
        p = object.__new__(device.Placement)
        p.pool = _Pool()
        clock = lambda: t[0]                                       # noqa: E731
        assert [len(device.pool_segments(p.pool, clock))] == [2]
        assert device.pool_segments(types.SimpleNamespace(id=(0, 8)), clock)[0]["total_size"] \
            == 64 << 20
        assert calls == [None], calls                    # never a tuple; one snapshot for both
        t[0] += device.SNAPSHOT_TTL_S + 0.01
        assert len(device.pool_segments(p.pool, clock)) == 2
        assert calls == [None, None], calls              # a new one after the TTL
        # no pool (the CPU): (0, 0), nothing asked
        p.pool = None
        n = len(calls)
        assert p.pool_bytes() == (0, 0) and len(calls) == n
    finally:
        if real is None:
            del torch._C._cuda_memorySnapshot
        else:
            torch._C._cuda_memorySnapshot = real
        with device._SNAP_LOCK:
            device._SNAP.update(t=None, segments=None)


def test_pool_bytes_sums_its_own_segments():
    segs = [{"segment_pool_id": (0, 7), "total_size": 4 << 20, "allocated_size": 1 << 20},
            {"segment_pool_id": [0, 7], "total_size": 2 << 20, "allocated_size": 2 << 20},
            {"segment_pool_id": (0, 8), "total_size": 64 << 20, "allocated_size": 9 << 20}]
    real = getattr(torch._C, "_cuda_memorySnapshot", None)
    torch._C._cuda_memorySnapshot = lambda arg: {"segments": segs}
    try:
        with device._SNAP_LOCK:
            device._SNAP.update(t=None, segments=None)
        p = object.__new__(device.Placement)
        p.pool = _Pool()
        assert p.pool_bytes() == (6 << 20, 3 << 20)
    finally:
        if real is None:
            del torch._C._cuda_memorySnapshot
        else:
            torch._C._cuda_memorySnapshot = real
        with device._SNAP_LOCK:
            device._SNAP.update(t=None, segments=None)


def test_the_refwatch():
    counts = {"True": 10000}
    t = [0.0]
    lines = []
    w = refwatch.RefWatch(log=lines.append, getref=lambda o: counts["True"] if o is True else 500,
                          clock=lambda: t[0], window_s=600.0, warn_every_s=600.0)
    for _ in range(10):                                  # steady: nothing said
        w.sample()
        t[0] += 30.0
    assert lines == [] and w.high["True"] == 10000 and w.high["None"] == 500
    for _ in range(40):                                  # 20 references a minute... then faster
        counts["True"] -= 200
        w.sample()
        t[0] += 30.0
    assert len(lines) == 1, lines                        # under half once, then once per interval
    assert lines[0].startswith("[refwatch] True: ") and "falling" in lines[0] \
        and "min to 0" in lines[0], lines[0]
    assert abs(w.rate("True") - 200 / 30.0) < 0.01, w.rate("True")
    t[0] += 600.0
    counts["True"] -= 200
    w.sample()
    assert len(lines) == 2 and w.warnings == 2
    assert w.stats()["now"]["True"] == counts["True"]
    keep = refwatch.immortal
    refwatch.immortal = lambda: True
    try:
        assert refwatch.start(log=lines.append) is None
    finally:
        refwatch.immortal = keep


def test_health_memory_has_the_counts():
    """`/health`'s memory part (what a soak watches) carries the singletons' counts now."""
    from server import app
    out = app._memory()
    assert set(out["python_refs"]) == {name for name, _ in refwatch.WATCH}, out
    assert out["python_refs"]["True"] > 0 and out["python_refs"]["None"] > 0
    if os.path.exists("/proc/self/status"):           # SPD-257: the address space, beside the resident set
        assert out["vm_size_gb"] >= out["rss_gb"] > 0, out


def test_the_memory_snapshot_call_leaks_too():
    """`torch.cuda.memory_snapshot()` hands the call `(0, 0, True)`: on torch 2.13.0 every call takes one
    reference off `True` and two off the int 0 (passes, saying so, on a torch that no longer leaks)."""
    if not _has_snapshot():
        print("    (torch without CUDA: no _cuda_memorySnapshot, skipped)")
        return
    d = _delta(torch.cuda.memory_snapshot)
    if d["True"] == 0:
        print(f"    (this torch {torch.__version__} keeps the references: fixed upstream)")
    else:
        assert d["True"] == -N and d["0"] == -2 * N, d


def test_the_policys_walk_keeps_the_references():
    """SPD-257: `cache.TorchPools` walks the allocator after each graph capture (`capture_lock.GEN`); over N walks no
    reference goes (failed before the fix on torch 2.13.0: -N on `True`, -2N on 0)."""
    if not _has_snapshot():
        print("    (torch without CUDA: no _cuda_memorySnapshot, skipped)")
        return
    from engine import cache, capture_lock
    gen = capture_lock.GEN
    # the count past the watched ints (0, 7): its own steps move no reference of theirs
    gen["captures"] = max(gen["captures"], 1000)

    def captured_then_read():
        gen["captures"] += 1
        return tp()
    keep = torch.cuda.memory_stats
    torch.cuda.memory_stats = lambda *a, **k: {"reserved_bytes.all.current": 64 << 30,
                                               "allocated_bytes.all.current": 60 << 30}
    try:
        tp = cache.TorchPools()
        tp()
        d = _delta(captured_then_read)
        assert tp.walks == N + 1, tp.walks
    finally:
        torch.cuda.memory_stats = keep
    assert all(v == 0 for v in d.values()), f"the policy's walk dropped references: {d}"
    return f"{tp.walks} walks, every count kept"


def test_the_pools_route_keeps_the_references():
    """SPD-257: `/v1/cache/memory?pools=1` walks the allocator once more (`_allocator_pools`): no reference goes."""
    if not _has_snapshot():
        print("    (torch without CUDA: no _cuda_memorySnapshot, skipped)")
        return
    from server import app
    assert set(app._allocator_pools()) >= {"default", "graphs"}
    d = _delta(app._allocator_pools, n=500)
    assert all(v == 0 for v in d.values()), f"_allocator_pools dropped references: {d}"


def _leaky_snapshot_calls(path: str) -> list:
    """The calls in one file that hand torch's snapshot a tuple (see the module's head)."""
    tree = ast.parse(open(path).read(), path)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        attr = node.func.attr
        recv = ast.unparse(node.func.value)
        if attr == "memory_snapshot":
            out.append((node.lineno, f"{recv}.memory_snapshot(...)"))
        elif attr == "snapshot" and ("pool" in recv.lower() or "MemPool" in recv):
            out.append((node.lineno, f"{recv}.snapshot(...)"))
        elif attr == "_cuda_memorySnapshot" and not (
                len(node.args) == 1 and not node.keywords and isinstance(node.args[0], ast.Constant)
                and node.args[0].value is None):
            out.append((node.lineno, f"{recv}._cuda_memorySnapshot({ast.unparse(node.args[0]) if node.args else ''})"))
    return out


def test_no_snapshot_call_that_drops_references():
    """SPD-257: the engine and the server read the allocator's segments only through the call that is handed
    `None` (`engine.decision.device._all_segments`, `engine.cache.allocator_segments`). The check itself is
    checked on the calls it must refuse."""
    bad = ("import torch\n"
           "a = torch.cuda.memory_snapshot()\n"
           "b = self.pool.snapshot()\n"
           "c = torch.cuda.MemPool().snapshot()\n"
           "d = torch._C._cuda_memorySnapshot((0, 0, True))\n"
           "e = torch._C._cuda_memorySnapshot(pool.id)\n"
           "f = torch._C._cuda_memorySnapshot(None)\n"
           "g = store.snapshot()\n")
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(bad)
    try:
        assert [ln for ln, _ in _leaky_snapshot_calls(f.name)] == [2, 3, 4, 5, 6], _leaky_snapshot_calls(f.name)
    finally:
        os.remove(f.name)
    found, seen = [], 0
    for top in ("engine", "server"):
        for dirpath, _, files in os.walk(os.path.join(ROOT, top)):
            for name in files:
                if not name.endswith(".py") or name.startswith("._"):   # a copy's macOS resource forks
                    continue
                path = os.path.join(dirpath, name)
                seen += 1
                found += [f"{os.path.relpath(path, ROOT)}:{ln}: {what}" for ln, what in _leaky_snapshot_calls(path)]
    assert seen > 100, f"only {seen} files under engine/ and server/"
    assert not found, "snapshot calls that drop references (use engine.cache.allocator_segments):\n" + \
        "\n".join(found)
    return f"{seen} files, no such call"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
