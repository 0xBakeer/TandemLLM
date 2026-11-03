"""SPD-257 (QA SPD-258): a chat's follow-up turn resumes at its end anchor, also at the memory ceiling.

THE MEASUREMENT (2026-10-31, live, three turns of one Chat tab chat): every turn resumed at the
4,096-token grid line from the disk and read 7.0 to 7.6k tokens again (prefill 7.9 to 9.3 s); after
each turn the engine logged `end anchor not kept (memory, no policy)`, 128 times in two hours. The
engine held 62.3 GiB allocated against a 64.36 GiB ceiling less 2 GiB of slack: the caches' caps add
up to more than the room above the load, so the end anchor, asked last, was refused at every end.
The reserve over what was allocated was the allocator's: the CUDA graphs' private pools and free
blocks in default-pool segments that still held a live block, which no shed lowers.

THE FIX:
  1  `ResidentPrefix.reserve_end`: each resident prefix holds its end anchor's buffers from the
     load on (QWEN38_END_RESERVE); no gate sheds them, no admission refuses them
  2  `Budget`: the peak stage stops once the stale rule holds and the stale rule counts only the
     default pool's free blocks (`Budget.pools`: the graphs' pools never count); no state is kept
     between gates; the allocator's expandable segments let `empty` hand back a fragmented
     segment's free pages, so the ceiling stage lowers the reserve
  3  a side slot's idle stash goes at `stash_idle`, never while a guest's rows are being copied into
     it; a side slot's guest chat becomes the slot's conversation at its end (QWEN38_SIDE_GUEST_END),
     never over a parent or worker that comes back (ENG-371) and only when its end can be kept; a new
     chat takes the slot served longest ago (QWEN38_SIDE_LRU); the log line reads the resident's own
     policy
  4  the memory report by owner (`memory_owners`, GET /v1/cache/memory, /health `memory.owners`)

The policy runs here on `SegAlloc`, a fake caching allocator with segments, blocks, owners, pools and
streams whose `snapshot()` has torch's shape, so `cache.TorchPools` reads it as it reads the device.

Gherkin rows of SPD-257 here: the follow-up at the ceiling resumes at its end (1, with the regression:
the same policy without the reservation refuses it), the park moves the pair and the next end has one
again (1), the graphs' pools' free bytes are no room for a workspace (2), a reserve over the peak that
only the graphs' pools hold sheds nothing (2), the ceiling under expandable segments sheds only what
it needs (2), a gate decides from the reading alone (2), an admission over the peak gives its shed back
before it asks again (2), a walk taken during a capture is taken again after it (2), a shed from another slot during a guest's
stash copy gives nothing back and a lost stash cuts the guest's rows (3), a guest chat on a side slot
keeps its end, its follow-up is routed to that slot and writes the same greedy tokens (3, regression:
off), a waiting parent or worker keeps the slot (3), an end that cannot be kept cuts nothing (3), the
router's question never takes the slot (3), the side slot's log line (3), the report by owner with the
graphs' pools and the policy's own free reserve (4).

Run: python tests/test_end_anchor_memory.py
"""

from __future__ import annotations

import io
import os
import sys
import threading
import time
from contextlib import redirect_stdout
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)

import torch  # noqa: E402

from engine import cache, capture_lock  # noqa: E402
from engine.cache import GIB, Budget, MemoryRefused  # noqa: E402
from server import app, endalign  # noqa: E402
from server import slots as S  # noqa: E402
from test_cache import fresh, tokens  # noqa: E402
from test_end_align import tb_of  # noqa: E402
from test_end_anchor import BIG, CH, TOOL, _state, turn  # noqa: E402
from test_side_carry import HEAD, S_PROMPT  # noqa: E402

MIB = 1 << 20
PAGE = 2 * MIB


# ------------------------------------------------------------------ a fake caching allocator

class SegAlloc:
    """The CUDA caching allocator in miniature. Segments of blocks, each segment in one pool (`(0, 0)` the
    default pool, any other a graph's private pool) and on one stream; a block is live (an owner's
    tensor) or free. `malloc` takes the first free block of its pool and stream that fits (best fit is
    not needed here) and splits it, else a new segment; with `reuse=False` it always takes a new segment
    (an allocator whose free blocks never fit the next buffer). `empty` gives back every default-pool
    segment with no live block; with `expandable=True` it also unmaps the free pages of a segment that
    still holds a live block (2 MiB granularity, exact here). A graph's pool never gives anything back.
    `snapshot()` has the shape of the segments `torch._C._cuda_memorySnapshot(None)` names."""

    def __init__(self, reuse: bool = True, expandable: bool = False):
        self.reuse = reuse
        self.expandable = expandable
        self.segs: list[dict] = []
        self.handles: dict = {}
        self._n = 0

    # --- the allocator's operations
    def malloc(self, n: int, owner: str = "", pool=(0, 0), stream: int = 0) -> int:
        n = int(n)
        if self.reuse:
            for seg in self.segs:
                if seg["pool"] != tuple(pool) or seg["stream"] != stream:
                    continue
                for i, b in enumerate(seg["blocks"]):
                    if not b["live"] and b["size"] >= n:
                        rest = b["size"] - n
                        b.update(size=n, live=True, owner=owner)
                        if rest:
                            seg["blocks"].insert(i + 1, {"size": rest, "live": False, "owner": ""})
                        return self._handle(seg, b)
        seg = {"pool": tuple(pool), "stream": stream, "blocks": []}
        size = -(-n // PAGE) * PAGE if self.expandable else n
        b = {"size": n, "live": True, "owner": owner}
        seg["blocks"].append(b)
        if size > n:
            seg["blocks"].append({"size": size - n, "live": False, "owner": ""})
        self.segs.append(seg)
        return self._handle(seg, b)

    def shared(self, n: int, m: int, owner: str, pool=(0, 0)) -> int:
        """A new segment holding `owner`'s block of `n` bytes and another owner's live block of `m` after it:
        the handle of the first."""
        size = n + m
        if self.expandable:
            size = -(-size // PAGE) * PAGE
        b = {"size": n, "live": True, "owner": owner}
        seg = {"pool": tuple(pool), "stream": 0,
               "blocks": [b, {"size": m, "live": True, "owner": "other"}]}
        if size > n + m:
            seg["blocks"].append({"size": size - n - m, "live": False, "owner": ""})
        self.segs.append(seg)
        return self._handle(seg, b)

    def _handle(self, seg, b) -> int:
        self._n += 1
        self.handles[self._n] = (seg, b)
        return self._n

    def free(self, h: int) -> int:
        seg, b = self.handles.pop(h)
        b["live"], b["owner"] = False, ""
        # neighbouring free blocks merge, as the allocator's do
        merged = []
        for x in seg["blocks"]:
            if merged and not merged[-1]["live"] and not x["live"]:
                merged[-1]["size"] += x["size"]
            else:
                merged.append(x)
        seg["blocks"] = merged
        return b["size"]

    def empty(self) -> None:
        keep = []
        for seg in self.segs:
            if seg["pool"] != (0, 0):
                keep.append(seg)                      # a graph's pool keeps its segments
                continue
            if not any(b["live"] for b in seg["blocks"]):
                continue
            if self.expandable:
                seg["blocks"] = [b for b in seg["blocks"] if b["live"]]
            keep.append(seg)
        self.segs = keep

    # --- what torch reads
    def probe(self) -> tuple[int, int]:
        res = sum(b["size"] for s in self.segs for b in s["blocks"])
        alloc = sum(b["size"] for s in self.segs for b in s["blocks"] if b["live"])
        return alloc, res

    def snapshot(self) -> list:
        out = []
        for s in self.segs:
            out.append({"segment_pool_id": s["pool"], "stream": s["stream"],
                        "total_size": sum(b["size"] for b in s["blocks"]),
                        "allocated_size": sum(b["size"] for b in s["blocks"] if b["live"]),
                        "is_expandable": self.expandable, "segment_type": "large",
                        "blocks": [{"size": b["size"], "state": "active_allocated" if b["live"] else "inactive"}
                                   for b in s["blocks"]]})
        return out

    def stats(self) -> dict:
        alloc, res = self.probe()
        return {"reserved_bytes.all.current": res, "allocated_bytes.all.current": alloc,
                "segment.all.current": len(self.segs)}

    # --- a cache that holds some of it
    def cache(self, order: list, level: str, gib: float, pieces: int = 1, safe_ok: bool = True,
              between: float = 0.0):
        """A level of `gib` in `pieces` blocks; with `between` GiB, a live block of another owner after each
        piece in the piece's own segment (the fragmentation that keeps a freed piece's segment reserved)."""
        handles = []
        for _ in range(pieces):
            n, m = int(gib * GIB / pieces), int(between * GIB)
            if m:
                handles.append(self.shared(n, m, level))
            else:
                handles.append(self.malloc(n, owner=level))

        def fn(safe=False):
            if safe and not safe_ok:
                return 0
            freed = sum(self.free(h) for h in handles)
            handles.clear()
            if freed:
                order.append(level)
            return freed
        return fn


class FakeTorchCuda:
    """The three readings `cache.TorchPools` takes (`torch.cuda.memory_stats`, the snapshot call with `None`
    and the calling stream), over a `SegAlloc`, for a `with` block. The snapshot is the one call the engine
    makes, `torch._C._cuda_memorySnapshot(None)`: `torch.cuda.memory_snapshot()` drops references on torch
    2.13.0 (tests/test_pool_snapshot_refcount.py)."""

    def __init__(self, sa: SegAlloc):
        self.sa = sa
        self.saved = None
        self.snapshots = 0
        self.args: list = []

    def _snapshot(self, arg):
        self.snapshots += 1
        self.args.append(arg)
        return {"segments": self.sa.snapshot()}

    def __enter__(self):
        self.saved = (torch.cuda.memory_stats, getattr(torch._C, "_cuda_memorySnapshot", None),
                      torch.cuda.current_stream)
        torch.cuda.memory_stats = self.sa.stats
        torch._C._cuda_memorySnapshot = self._snapshot
        torch.cuda.current_stream = lambda *a: SimpleNamespace(cuda_stream=0)
        return self

    def __exit__(self, *a):
        torch.cuda.memory_stats, snap, torch.cuda.current_stream = self.saved
        if snap is None:
            del torch._C._cuda_memorySnapshot
        else:
            torch._C._cuda_memorySnapshot = snap
        assert all(x is None for x in self.args), f"a snapshot handed a tuple: {self.args[:3]}"


def live_budget(sa: SegAlloc, ceiling: float = 64.36, peak: float = 75.36, **kw) -> Budget:
    """The live numbers of 2026-10-31: ceiling 64.36, peak 75.36, slack 2 GiB, a 2.75 GiB workspace floor."""
    m = Budget(int(ceiling * GIB), int(peak * GIB), 2 * GIB, probe=sa.probe, empty=sa.empty,
               peak_probe=lambda reset=False: 0, ws_floor=int(2.75 * GIB), **kw)
    m.pools_probe = cache.TorchPools()
    return m


def _engine(sa: SegAlloc, gib: float) -> int:
    return sa.malloc(int(gib * GIB), owner="engine")


def _graphs(sa: SegAlloc, reserved: float, live: float) -> None:
    """A graph's private pool: `reserved` GiB of segments, `live` of them held by the graph."""
    pool = (1, 7)
    h = sa.malloc(int(reserved * GIB), owner="graph", pool=pool)
    if live < reserved:
        sa.free(h)
        sa.malloc(int(live * GIB), owner="graph", pool=pool)


# ------------------------------------------------------------------ the allocator's pools

def test_spd257_torch_pools_reads_the_graphs_pools_and_walks_only_after_a_capture():
    """The graphs' figures are the walk's until a graph is captured; the default pool's free bytes are the
    allocator's counters now: reserved less allocated less the graphs' free bytes."""
    sa = SegAlloc()
    _engine(sa, 50.0)
    _graphs(sa, 4.33, 1.30)
    h = sa.malloc(int(1.0 * GIB), owner="store")
    sa.free(h)                                         # 1 GiB free in the default pool
    tp = cache.TorchPools()
    with FakeTorchCuda(sa) as fc:
        p = tp()
        assert p["graphs_reserved"] == int(4.33 * GIB) and p["graphs_free"] == int(4.33 * GIB) - int(1.30 * GIB)
        assert p["default_free"] == int(1.0 * GIB) and p["graphs_pools"] == 1, p
        h2 = sa.malloc(int(0.5 * GIB), owner="anchor")  # out of the free block
        p2 = tp()
        assert p2["default_free"] == int(1.0 * GIB) - int(0.5 * GIB), p2
        sa.free(h2)
        sa.empty()                                     # the free segment goes back: the counters say so
        assert tp()["default_free"] == 0 and fc.snapshots == 1, "no capture: no walk"
        with capture_lock.LOCK:                        # a capture into the graph's pool
            sa.malloc(int(0.5 * GIB), owner="graph", pool=(1, 7))
        p3 = tp()
        assert fc.snapshots == 2 and p3["graphs_free"] == p["graphs_free"] - int(0.5 * GIB), p3
        assert p3["walks"] == 2 and p3["captures"] == capture_lock.GEN["captures"], p3
    return f"graphs {p['graphs_reserved'] / GIB:.2f} GiB ({p['graphs_free'] / GIB:.2f} free), 2 walks"


def test_spd257_the_graphs_pools_free_bytes_are_no_room_for_a_workspace():
    """The reserve over what is allocated is a graph's pool's free blocks. Before, the stale rule
    counted them as room and admitted the prefill over the peak; now the levels go first, and with
    nothing left the prefill is refused. Free blocks in the default pool still admit it."""
    sa = SegAlloc()
    _engine(sa, 62.3)
    _graphs(sa, 14.0, 1.0)                             # 13 GiB free in the graph's pool
    order = []
    m = live_budget(sa, ceiling=80.0)
    with FakeTorchCuda(sa):
        try:
            m.gate(8000)
            raise AssertionError("admitted on the graphs' free blocks")
        except MemoryRefused:
            pass
        assert m.stats["stale_admits"] == 0
        m.pools_probe = None                           # the rule before: the whole free reserve
        g = m.gate(8000)
        assert g.get("stale") and m.stats["stale_admits"] == 1, g
    sa2 = SegAlloc()
    _engine(sa2, 62.3)
    _graphs(sa2, 10.0, 1.0)
    m2 = live_budget(sa2, ceiling=80.0)
    m2.register("store", sa2.cache(order, "store", 0.5))
    sa2.free(sa2.shared(int(4.0 * GIB), int(0.1 * GIB), "old"))   # 4 GiB free beside a live block
    with FakeTorchCuda(sa2):
        g2 = m2.gate(8000)
    assert g2.get("stale") and order == [], (g2, order)
    return "graphs' free blocks: refused; the default pool's: admitted, nothing shed"


class _FailingPools:
    """A pools probe whose reading fails, with the last reading it took (or none)."""

    def __init__(self, last):
        self.last = last

    def __call__(self):
        raise RuntimeError("the allocator's snapshot failed")


def test_spd257_a_failed_pools_reading_never_admits_a_workspace_on_the_scratch():
    """The reading of the pools fails (the snapshot raises). Before, the stale rule then counted the whole free
    reserve, the graphs' scratch with it, and admitted the prefill over the peak. Now the last reading's graphs'
    free bytes stay out, and with no reading there is no such room: refused either way. Free bytes in the default
    pool still admit it on the last reading; no probe at all (an allocator without expandable segments) is the
    rule before."""
    sa = SegAlloc()
    _engine(sa, 62.3)
    _graphs(sa, 14.0, 1.0)                             # 13 GiB free in the graph's pool
    m = live_budget(sa, ceiling=80.0)
    with FakeTorchCuda(sa):
        good = dict(m.pools())
    for last in (good, None):
        m.pools_probe = _FailingPools(last)
        alloc, res = m.memory()
        assert m.default_free(alloc, res) == 0, (last is None, m.default_free(alloc, res) / GIB)
        try:
            m.gate(8000)
            raise AssertionError(f"admitted on the graphs' scratch (last reading: {last is not None})")
        except MemoryRefused:
            pass
    assert m.stats["stale_admits"] == 0 and m.stats["refused"] == 2, m.stats
    sa.free(sa.shared(int(4.0 * GIB), int(0.1 * GIB), "old"))     # 4 GiB free in the default pool
    m.pools_probe = _FailingPools(good)
    assert m.gate(8000).get("stale"), "the default pool's free bytes on the last reading"
    m.pools_probe = _FailingPools(None)
    try:
        m.gate(8000)
        raise AssertionError("admitted with no reading")
    except MemoryRefused:
        pass
    m.pools_probe = None
    assert m.gate(8000).get("stale") and m.stats["stale_admits"] == 2, m.stats
    return "refused on a failed reading with or without the last one; the default pool's free bytes admitted"


def test_spd257_the_peak_stage_sheds_nothing_while_the_stale_rule_holds():
    """Two readings of 2026-10-31 (63.5 GiB allocated, 65.66 reserved, a 2.75 GiB workspace; 66.57
    reserved, a 7.29 GiB workspace, peak 76.18) are under the peak: nothing is shed for the peak. Over
    the peak with the workspace inside the default pool's free blocks: nothing either. Over the peak
    with a workspace the free blocks cannot hold: the levels go until one of the two holds."""
    cases = []
    for alloc, res, n, peak in ((63.5, 65.66, 100, 75.36), (64.0, 66.57, 100_000, 76.18)):
        sa = SegAlloc()
        _engine(sa, alloc - 2.0)                       # the four levels below hold 2 GiB of it
        _graphs(sa, res - alloc, 0.0)
        order = []
        m = live_budget(sa, ceiling=70.0, peak=peak)
        m.ws_per_token = int(7.29 * GIB / 100_000)
        for lv in ("store", "anchors", "park", "pin"):
            m.register(lv, sa.cache(order, lv, 0.5))
        with FakeTorchCuda(sa):
            g = m.gate(n)
        assert order == [] and not any(e.get("why") == "peak" for e in m.events), (alloc, order)
        cases.append(round(g["workspace"] / GIB, 2))
    # over the peak, the workspace inside the default pool's free blocks: the stale rule, no shed
    sa = SegAlloc()
    _engine(sa, 62.0)
    _graphs(sa, 9.0, 1.0)
    order = []
    m = live_budget(sa, ceiling=80.0)
    for lv in ("store", "anchors", "park"):
        m.register(lv, sa.cache(order, lv, 0.5))
    sa.free(sa.shared(int(4.0 * GIB), int(0.1 * GIB), "old"))     # 4 GiB free in a segment with a live block
    with FakeTorchCuda(sa):
        g = m.gate(100)
    assert order == [] and g.get("stale"), (order, g)
    # over the peak, the default pool's free blocks short of the workspace: levels go until it fits
    sa = SegAlloc()
    _engine(sa, 63.0)
    _graphs(sa, 9.0, 1.0)
    order = []
    m = live_budget(sa, ceiling=80.0)
    for lv in ("store", "anchors", "park", "pin"):
        m.register(lv, sa.cache(order, lv, 1.0, between=0.01))
    with FakeTorchCuda(sa):
        g = m.gate(100)
        assert order and order != ["store", "anchors", "park", "pin"], order
        assert m._peak_over(g["workspace"]) <= 0
    assert any(e["kind"] == "peak_over" for e in m.events)
    return f"workspaces {cases} GiB: nothing shed; short: shed {order}"


def test_spd257_the_ceiling_under_expandable_segments_sheds_only_what_it_needs():
    """Caches whose blocks sit between live ones. Without expandable segments a shed frees blocks that
    stay reserved, every level goes and the reserve stays over the ceiling. With them `empty` hands the
    free pages back: the first levels bring the reserve under the ceiling and the rest stay."""
    out = {}
    for exp in (False, True):
        sa = SegAlloc(expandable=exp)
        _engine(sa, 60.0)
        _graphs(sa, 1.4, 1.0)
        order = []
        m = live_budget(sa)
        for lv in ("store", "stash_idle", "anchors", "stash", "park", "pin"):
            m.register(lv, sa.cache(order, lv, 0.8, pieces=4, between=0.01))
        with FakeTorchCuda(sa):
            m.gate(100)
        alloc, res = sa.probe()
        out[exp] = (list(order), res)
        if exp:
            assert res <= m.ceiling and len(order) < 6, (order, res / GIB)
        else:
            assert len(order) == 6 and res > m.ceiling, (order, res / GIB)
            assert m.stats["over_ceiling"] == 1 and any(e["kind"] == "over_ceiling" for e in m.events)
    return (f"plain: shed {len(out[False][0])}, reserved {out[False][1] / GIB:.2f} GiB; expandable: shed "
            f"{out[True][0]}, reserved {out[True][1] / GIB:.2f} GiB")


def test_spd257_a_gate_decides_from_the_reading_alone():
    """No stuck or stale record. Two gates on the same reading decide the same; the policy keeps no
    knob or field for a reserve it gave up on."""
    from engine import settings
    assert "MEM_STUCK" not in settings.KNOBS and "MEM_STALE_ADMIT" not in settings.KNOBS
    sa = SegAlloc()
    _engine(sa, 62.3)
    _graphs(sa, 14.0, 1.0)
    m = live_budget(sa, ceiling=80.0)
    for attr in ("stuck", "stuck_on", "stale_admit_on", "reserve_floor", "graph_free", "stale_blocked"):
        assert not hasattr(m, attr), attr
    with FakeTorchCuda(sa):
        for _ in range(2):
            try:
                m.gate(8000)
                raise AssertionError("admitted")
            except MemoryRefused:
                pass
    assert m.stats["refused"] == 2


def test_spd257_the_ceiling_leaves_the_graphs_scratch_out():
    """The hold of 2026-10-31: the graphs' pools grew to 4.57 GiB (3.19 GiB free, their replays' scratch)
    while what was allocated stayed under the ceiling, and the ceiling stage gave the store and a 3.63 GiB
    park away into it. Now the ceiling leaves those free bytes out; with nothing shed no over-ceiling line
    is logged (the report has the pools); with what is allocated itself over the ceiling the levels still
    go, and without a pools reading as before."""
    def setup(extra=0.0):
        sa = SegAlloc(expandable=True)
        _engine(sa, 58.0 + extra)
        _graphs(sa, 4.57, 1.38)
        order = []
        m = live_budget(sa)
        m.register("store", sa.cache(order, "store", 0.5))
        m.register("park", sa.cache(order, "park", 3.63))
        return sa, m, order
    sa, m, order = setup()
    with FakeTorchCuda(sa):
        m.gate(100)
    alloc, res = sa.probe()
    assert order == [] and res > m.ceiling and alloc <= m.ceiling, (order, res / GIB)
    assert not [e for e in m.events if e["kind"] == "over_ceiling"] and m.stats["over_ceiling"] == 0, m.events
    gfree = cache.allocator_pools(sa.snapshot())["graphs"]["free"]
    assert m.report()["pools"]["graphs_free"] == gfree and gfree > 3 * GIB, m.report()["pools"]
    sa, m, order = setup(extra=1.5)                    # allocated itself over the ceiling
    with FakeTorchCuda(sa):
        m.gate(100)
    assert order == ["store", "park"], order
    sa, m, order = setup()
    m.pools_probe = None                               # no reading: the whole reserve counts, as before
    m.gate(100)
    assert order == ["store", "park"], order
    return f"reserved {res / GIB:.2f} GiB over {m.ceiling / GIB:.2f}, allocated {alloc / GIB:.2f}: nothing shed"


def test_spd257_the_scratch_the_ceiling_leaves_out_stops_short_of_the_peak():
    """The scratch the ceiling leaves out has a bound: the room between the ceiling and the peak less the
    smallest prefill's workspace (8.25 GiB at the live limits). 9.5 GiB of scratch beside 64.0 GiB allocated
    holds 73.5 GiB between requests, over the 72.61 the peak leaves after a 2.75 GiB workspace: the levels go
    until the reserve is back under that bound, and the line says what was shed and what holds the rest."""
    sa = SegAlloc(expandable=True)
    _engine(sa, 62.0)
    _graphs(sa, 10.0, 0.5)
    order = []
    m = live_budget(sa)
    m.register("store", sa.cache(order, "store", 0.5))
    m.register("park", sa.cache(order, "park", 1.0))
    bound = m.peak - m.ws_floor
    alloc0, res0 = sa.probe()
    assert alloc0 <= m.ceiling and res0 > bound, (alloc0 / GIB, res0 / GIB)
    with FakeTorchCuda(sa):
        m.gate(100)
        exempt = m._exempt()
        assert exempt == m.peak - m.ceiling - m.ws_floor < m._scratch(), (exempt, m._scratch())
    alloc, res = sa.probe()
    assert order == ["store", "park"] and res <= bound, (order, res / GIB)
    ev = [e for e in m.events if e["kind"] == "over_ceiling"]
    assert len(ev) == 1 and ev[0]["shed"] == ["store", "park"] and ev[0]["exempt"] == exempt, ev
    lines = io.StringIO()
    with redirect_stdout(lines):
        app._OVER_CEILING_LOGGED[0] = 0.0
        app._memory_event(ev[0])
    assert lines.getvalue().startswith("[memory] over the ceiling by ") and "after shedding store,park" in \
        lines.getvalue(), lines.getvalue()
    return f"reserved {res0 / GIB:.2f} -> {res / GIB:.2f} GiB (bound {bound / GIB:.2f}), shed {order}"


def test_spd257_under_the_ceiling_a_gate_reads_no_pools():
    """A gate under the ceiling and the peak, and a request's end with the reserve under the slack, walk nothing:
    the walk holds the interpreter lock (10 to 140 ms on the box), and the scratch can only lower the ceiling's
    figure."""
    sa = SegAlloc(expandable=True)
    _engine(sa, 55.0)
    _graphs(sa, 1.4, 1.0)
    m = live_budget(sa)
    with FakeTorchCuda(sa) as fc:
        for _ in range(5):
            m.gate(8000)
            m.done()
        assert fc.snapshots == 0 and m.pools_probe.walks == 0, fc.snapshots
        _graphs(sa, 12.0, 0.5)                         # over the ceiling: now the pools are read
        m.gate(100)
        assert fc.snapshots >= 1
    return "5 gates and ends under the limits: 0 walks"


def test_spd257_a_roles_own_pool_is_no_graphs_scratch():
    """A role's `torch.cuda.MemPool` (id (0, k): the shadow reader's, the narrator's, a decision model's) beside
    a graph's pool (id (k, 0)). The policy's figures leave the role's pool out up to its claim (`exclude`); its
    free blocks are no graph's scratch, so the ceiling sheds for the engine's own reserve as it does without
    the role. Before, every pool but the default one counted as the graphs', the role's 2 GiB of free blocks
    came off the reserve a second time and nothing was shed. The report names the role's pool apart."""
    def run(role: bool):
        sa = SegAlloc()
        _engine(sa, 59.5)
        order = []
        store = sa.cache(order, "store", 0.5)                         # a segment of its own
        sa.free(sa.shared(int(2.5 * GIB), int(0.01 * GIB), "old"))   # 2.5 GiB free beside a live block
        sa.free(sa.malloc(int(1.0 * GIB), owner="graph", pool=(5, 0)))   # a graph's scratch, all free
        if role:
            sa.free(sa.malloc(int(3.0 * GIB), owner="role", pool=(0, 7)))
            sa.malloc(int(1.0 * GIB), owner="role", pool=(0, 7))      # 3 GiB reserved, 1 GiB live
        m = live_budget(sa, ceiling=62.0, exclude=(lambda: (int(1.0 * GIB), int(3.0 * GIB))) if role else None)
        m.register("store", store)
        with FakeTorchCuda(sa):
            p = dict(m.pools())
            alloc, res = m.memory()
            free = m.default_free(alloc, res)
            m.gate(100)
        return order, p, free
    plain, p0, f0 = run(False)
    with_role, p1, f1 = run(True)
    assert plain == ["store"] and with_role == ["store"], (plain, with_role)
    assert p1["graphs_free"] == p0["graphs_free"] == int(1.0 * GIB) and p1["graphs_pools"] == 1, p1
    assert p1["mempools_free"] == int(2.0 * GIB) and p1["mempools_pools"] == 1 and p0["mempools_pools"] == 0, p1
    assert f0 == f1 == int(2.5 * GIB), (f0 / GIB, f1 / GIB)        # the role's free blocks are no room either
    return f"the store shed with and without the role's pool ({p1['mempools_free'] / GIB:.1f} GiB free in it)"


def test_spd257_the_idle_empty_leaves_the_graphs_scratch_out():
    """After a request the reserve over the slack goes back; the graphs' scratch is no such reserve. With
    3.19 GiB of it and nothing free in the default pool no `empty` runs (before: one after every request);
    with 2.5 GiB free in the default pool it does."""
    sa = SegAlloc(expandable=True)
    _engine(sa, 58.0)
    _graphs(sa, 4.57, 1.38)
    m = live_budget(sa)
    with FakeTorchCuda(sa):
        assert m.done() == 0 and m.stats["empties"] == 0
        sa.free(sa.malloc(int(2.5 * GIB), owner="workspace"))
        sa.malloc(PAGE, owner="live")
        assert m.done() > 0 and m.stats["empties"] == 1
    m.pools_probe = None
    m.done()
    assert m.stats["empties"] == 2, "without a reading the whole reserve counts, as before"


def test_spd257_a_cache_near_the_peak_counts_its_whole_buffer():
    """With the allocator's pools read, a buffer counts all its bytes on top of the reserve against the peak, also
    when a free block of the default pool could take it (no anchor or park ever fit one under expandable
    segments); without them a reserve over what is allocated counts as room, as before. No code path admits a
    cache by a fit."""
    sa = SegAlloc()
    _engine(sa, 60.0)
    _graphs(sa, 9.0, 1.0)                              # 8 GiB free, the graphs' scratch
    sa.free(sa.shared(int(1.0 * GIB), int(0.1 * GIB), "old"))     # 1 GiB free in the default pool
    m = live_budget(sa, ceiling=70.0, peak=70.5)       # reserved 70.1: 0.4 GiB under the peak
    with FakeTorchCuda(sa):
        assert not m.admit("anchors", int(0.9 * GIB)), "0.9 GiB more over the peak, free block or not"
        assert m.admit("store", int(0.3 * GIB)), "under the peak"
        m.ws_now = int(0.2 * GIB)
        assert not m.admit("store", int(0.3 * GIB)), "the running workspace counts"
        m.ws_now = 0
    m.pools_probe = None
    assert m.admit("anchors", int(0.9 * GIB)), "no reading: the free reserve counts as room, as before"
    assert not hasattr(m, "_fits") and not hasattr(m, "_fit_settle") and not hasattr(m, "_fit_pending")
    assert not hasattr(cache.TorchPools, "fits") and "admit_fit" not in m.stats
    assert not [e for e in m.events if e["kind"] == "admit_fit"]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in ("engine/cache.py", "server/app.py"):
        with open(os.path.join(root, rel)) as f:
            assert "admit_fit" not in f.read(), rel
    return "refused over the peak with a free block, admitted under it; no fit path"


def _band(pools: bool, store: float = 3.0):
    """A reading of 2026-11-01: 61 GiB allocated, 70.3 reserved (the graphs' scratch), a 4.5 GiB
    workspace running, a `store` GiB state store. A 1 GiB buffer is under the ceiling less the slack and under the
    peak on the reserve (74.8 of 75.36), over it with its whole bytes on top (75.8)."""
    sa = SegAlloc(expandable=True)
    _engine(sa, 61.0 - store - 1.0)
    _graphs(sa, 10.3, 1.0)
    order = []
    m = live_budget(sa)
    if not pools:
        m.pools_probe = None
    if store:
        m.register("store", sa.cache(order, "store", store))
    m.ws_now = int(4.5 * GIB)
    alloc, res = sa.probe()
    assert abs(alloc / GIB - 61.0) < 0.01 and abs(res / GIB - 70.3) < 0.01, (alloc / GIB, res / GIB)
    return sa, m, order


def test_spd257_an_admission_over_the_peak_gives_its_shed_back_before_it_asks_again():
    """A guest's stash (1 GiB) asked for while a prefill's workspace runs, the reserve over what is allocated the
    graphs' scratch. Counted whole on top of the reserve it is over the peak; the store below it goes, and a shed
    lowers what is allocated, not the reserve: before, the question was asked again on the same reserve, refused,
    and the store was gone for nothing (the stash refused cuts the guest's conversation). Now the shed's free
    blocks go back to the driver first, and the stash is admitted on the lower reserve. With no level below to
    give anything, nothing is emptied; with no pools read (the default allocator) the reserve is room, as before."""
    sa, m, order = _band(pools=True)
    with FakeTorchCuda(sa):
        res0 = sa.probe()[1]
        assert m.admit("stash", GIB), m.events
        res1 = sa.probe()[1]
    empties = [e for e in m.events if e["kind"] == "empty"]
    assert order == ["store"] and [e["why"] for e in empties] == ["admit:stash"], (order, list(m.events))
    assert res0 - res1 >= int(2.9 * GIB) and res1 + GIB + m.ws_now <= m.peak, (res0 / GIB, res1 / GIB)
    assert m.stats["admit_refused"] == 0 and m.stats["shed"]["store"] == 1
    sa, m, order = _band(pools=True, store=0.0)
    with FakeTorchCuda(sa):
        assert not m.admit("stash", GIB)
    assert m.stats["empties"] == 0 and m.stats["admit_refused"] == 1, m.stats
    sa, m, order = _band(pools=False)
    assert m.admit("stash", GIB) and order == [] and m.stats["empties"] == 0
    return f"store shed and emptied: the reserve fell {(res0 - res1) / GIB:.2f} GiB, the stash admitted"


def test_spd257_a_walk_during_a_capture_is_taken_again_after_it():
    """Another thread (an HTTP reader, the gate of another slot) reads the pools while a capture holds the lock and
    is half way through its allocations into the graph's pool. The capture counts when it lets go (`GEN`), so the
    next reading walks again and sees the whole capture; counted when it took the lock, the walk taken during it
    would have stood for good."""
    sa = SegAlloc(expandable=True)
    _engine(sa, 50.0)
    _graphs(sa, 2.0, 1.0)                              # 1 GiB free in the graph's pool
    tp = cache.TorchPools()
    inside, go, err = threading.Event(), threading.Event(), []

    def capture():
        try:
            with capture_lock.LOCK:
                sa.malloc(int(0.25 * GIB), owner="graph", pool=(1, 7))
                inside.set()
                go.wait(10)
                sa.malloc(int(0.25 * GIB), owner="graph", pool=(1, 7))
        except Exception as exc:                                   # noqa: BLE001
            err.append(exc)
    with FakeTorchCuda(sa) as fc:
        t = threading.Thread(target=capture)
        t.start()
        assert inside.wait(10)
        during = dict(tp())
        go.set()
        t.join(10)
        after = dict(tp())
        again = dict(tp())
    assert not err and not t.is_alive(), err
    assert during["graphs_free"] == int(0.75 * GIB) and during["walks"] == 1, during
    assert after["graphs_free"] == int(0.5 * GIB) and after["walks"] == 2 and fc.snapshots == 2, after
    assert again["walks"] == 2, again
    return "a walk during the capture, one after it, none more"


def test_spd257_the_policys_walk_reads_no_block():
    """The walk the policy takes after each capture counts segments only: the largest free block on each stream is
    the on-demand report's (GET /v1/cache/memory?pools=1, the load line), which no policy figure uses."""
    sa = SegAlloc()
    _engine(sa, 50.0)
    sa.free(sa.malloc(int(0.3 * GIB), owner="old"))

    class _Seg(dict):
        def get(self, k, d=None):
            if k == "blocks":
                raise AssertionError("the policy's walk read the blocks")
            return super().get(k, d)
    segs = [_Seg(s) for s in sa.snapshot()]
    p = cache.allocator_pools(segs)
    assert "largest_free" not in p["default"] and p["default"]["segments"] == 2, p["default"]
    full = cache.allocator_pools(sa.snapshot(), blocks=True)
    assert full["default"]["largest_free"] == {0: int(0.3 * GIB)}, full["default"]
    with FakeTorchCuda(sa):
        assert app._allocator_pools()["default"]["largest_free"] == {0: int(0.3 * GIB)}
        tp = cache.TorchPools()
        tp()
    assert "largest_free" not in tp.walk["default"]
    return "the walk: segments only; the report on demand: the largest free block by stream"


def test_spd257_the_report_names_the_policys_free_reserve():
    """`pools.default_free` is the free reserve less the graphs' free bytes and still holds the roles' own pools'
    free bytes; the policy leaves those out (`foreign_free`), and the report's `default_free` is its figure."""
    sa = SegAlloc()
    _engine(sa, 55.0)
    _graphs(sa, 2.0, 1.0)
    sa.free(sa.malloc(int(1.5 * GIB), owner="old"))    # 1.5 GiB free in the default pool
    m = live_budget(sa)
    m.foreign_free = lambda: int(0.5 * GIB)            # of it a role's pool's
    assert m.report()["default_free"] is None, "no reading yet"
    with FakeTorchCuda(sa):
        p = m.pools()
        alloc, res = m.memory()
        rep = m.report()
        policy = m.default_free(alloc, res)
    assert p["default_free"] == int(1.5 * GIB) and rep["pools"]["default_free"] == p["default_free"], p
    assert rep["default_free"] == policy == int(1.0 * GIB), (rep["default_free"], policy)
    m.pools_probe = None
    assert m.report()["default_free"] == int(1.0 * GIB) + int(1.0 * GIB), "no pools read: the free reserve"
    return "pools 1.50 GiB, the policy's 1.00 GiB"


def test_spd257_the_load_line_names_the_captures_switch_only_where_it_runs():
    """Every family runs `capture_lock.alloc_expandable()` at the load, so `ALLOC["on"]` is False for those without
    the setting: the load line says nothing of the captures' switch then, and says it where the setting asks for
    expandable segments or a switch failed."""
    sa = SegAlloc()
    _engine(sa, 55.0)
    _graphs(sa, 1.41, 1.0)
    m = live_budget(sa)
    keep = dict(capture_lock.ALLOC)
    old = dict.get(app.STATE, "mem")
    dict.__setitem__(app.STATE, "mem", m)
    lines = {}
    try:
        for on, errors in ((False, 0), (None, 0), (True, 0), (False, 1)):
            capture_lock.ALLOC.update(on=on, errors=errors, toggles=4, error="refused" if errors else None)
            buf = io.StringIO()
            with FakeTorchCuda(sa), redirect_stdout(buf):
                app._mem_load_mark()
            lines[(on, errors)] = buf.getvalue()
    finally:
        capture_lock.ALLOC.clear()
        capture_lock.ALLOC.update(keep)
        if old is None:
            dict.pop(app.STATE, "mem", None)
        else:
            dict.__setitem__(app.STATE, "mem", old)
        app.STATE.pop("mem_load", None)
    assert all("[memory] after the load" in v for v in lines.values()), lines
    assert "captures on plain segments" not in lines[(False, 0)] + lines[(None, 0)], lines
    assert "captures on plain segments (4 switches)" in lines[(True, 0)], lines[(True, 0)]
    assert "captures on plain segments (4 switches, off: refused)" in lines[(False, 1)], lines[(False, 1)]
    return "no clause without the setting; switches and a failure named"


class _CountedPools(cache.TorchPools):
    def __init__(self):
        super().__init__()
        self.reads = 0

    def __call__(self):
        self.reads += 1
        return super().__call__()


def test_spd257_gates_and_ends_walk_once_until_a_capture():
    """The reserve over the ceiling only by the graphs' scratch (the steady state of a long run): every gate and
    every request's end reads the pools, and each request makes a buffer in a new segment, so the reserve moves
    at every one (a walk keyed on the reserve walked at every request). N gate/done cycles with no capture take
    one walk; a capture (its lock let go) takes one more, a graph owner's reset (`give_back`) one more."""
    N = 20
    sa = SegAlloc(expandable=True)
    _engine(sa, 58.0)
    _graphs(sa, 4.57, 1.38)
    order = []
    m = live_budget(sa)
    m.pools_probe = _CountedPools()
    m.register("store", sa.cache(order, "store", 0.5))
    m.register("park", sa.cache(order, "park", 3.63))
    stream = object()
    walks = []
    try:
        with FakeTorchCuda(sa) as fc:
            for step in ("none", "capture", "reset"):
                if step == "capture":
                    with capture_lock.LOCK:
                        sa.malloc(int(0.2 * GIB), owner="graph", pool=(1, 7))
                elif step == "reset":
                    capture_lock.give_back(stream)
                for _ in range(N):
                    m.gate(100)
                    sa.malloc(PAGE, owner="anchors")   # a new segment: the reserve moves
                    m.done()
                walks.append(fc.snapshots)
    finally:
        if stream in capture_lock._FREE:              # the lease pool as it was
            capture_lock._FREE.remove(stream)
            capture_lock.LEASES["out"] += 1
    assert walks == [1, 2, 3] and m.pools_probe.walks == 3, walks
    assert m.pools_probe.reads >= 3 * N * 2, m.pools_probe.reads
    assert order == [] and m.stats["empties"] == 0, (order, m.stats["empties"])
    p = m.report()["pools"]
    assert p["graphs_free"] == cache.allocator_pools(sa.snapshot())["graphs"]["free"] > 2 * GIB, p
    assert p["walks"] == 3 and p["captures"] == capture_lock.GEN["captures"], p
    return f"{3 * N} gates and ends, {m.pools_probe.reads} readings, {walks[-1]} walks"


def test_spd257_random_admissions_keep_the_reserve_under_the_peak():
    """200 random admissions, buffers made and freed, on an allocator that reuses its free blocks and on one that
    never does: after each buffer is made, the reserve plus the running workspace stays under the peak."""
    import random
    out = {}
    for reuse in (True, False):
        rnd = random.Random(257)
        sa = SegAlloc(reuse=reuse)
        _engine(sa, 60.0)
        _graphs(sa, 4.0, 1.0)
        m = live_budget(sa, ceiling=68.0, peak=72.0)
        live, worst, admitted = [], None, 0
        with FakeTorchCuda(sa):
            for i in range(200):
                m.ws_now = rnd.choice((0, int(2.75 * GIB), int(4.0 * GIB)))
                n = rnd.randint(1, 600) * MIB
                if m.admit(rnd.choice(("store", "anchors", "park", "stash")), n):
                    admitted += 1
                    live.append(sa.malloc(n, owner="cache"))
                    alloc, res = sa.probe()
                    over = res + m.ws_now - m.peak
                    worst = over if worst is None else max(worst, over)
                if live and rnd.random() < 0.45:
                    sa.free(live.pop(rnd.randrange(len(live))))
                if rnd.random() < 0.1:
                    m.ws_now = 0
                    m.done()
            m.done()
        out[reuse] = (admitted, worst)
        assert admitted >= 20 and m.stats["admit_refused"] > 0, (reuse, admitted, m.stats["admit_refused"])
        assert worst <= 0, (reuse, worst / GIB)
    return (f"reuse: {out[True][0]} admitted, worst {out[True][1] / GIB:.2f} GiB to the peak; "
            f"no reuse: {out[False][0]} admitted, worst {out[False][1] / GIB:.2f} GiB")


def test_spd257_the_report_names_what_no_shed_gives_back():
    sa = SegAlloc()
    _engine(sa, 55.0)
    _graphs(sa, 4.33, 1.30)
    sa.free(sa.shared(int(0.6 * GIB), int(0.01 * GIB), "old"))   # free beside a live block, not expandable
    m = live_budget(sa)
    with FakeTorchCuda(sa):
        p = m.pools()
        rep = m.report()
    assert p["graphs_free"] == int(4.33 * GIB) - int(1.30 * GIB)
    assert p["unreleasable"] == p["graphs_free"] + int(0.6 * GIB), p
    assert rep["pools"]["unreleasable"] == p["unreleasable"] and p["default_free"] == int(0.6 * GIB)
    return f"unreleasable {p['unreleasable'] / GIB:.2f} GiB"


# ------------------------------------------------------------------ the reserved end, on the model

class Full:
    """An allocator at the ceiling: every admission is refused."""

    def __init__(self, alloc):
        self.alloc = alloc

    def __call__(self):
        return self.alloc, self.alloc


def _full():
    return cache.Budget(10 * BIG, 12 * BIG, 0, probe=Full(10 * BIG), empty=lambda: None)


def test_spd257_the_follow_up_resumes_at_its_end_at_the_ceiling():
    prompt = tokens(100, seed=60)
    res = _state(mem=_full())
    turn(prompt, 20)
    assert res.end is None and res.stats["end_refused"] == 1, "the regression: refused at the ceiling"
    res = _state(mem=_full())
    assert res.reserve_end(app.STATE["engine"]) and res.end_bytes > 0
    out1, _, _ = turn(prompt, 20)                    # every admission refused: the end needs none
    assert res.end is not None and res.stats["end_refused"] == 0, res.stats
    out2, pre2, _ = turn(prompt + out1 + TOOL, 20)
    assert pre2["kind"] == "end" and pre2["forwarded"] == len(TOOL) + 1, pre2
    nbytes = res.end_bytes
    assert res.shed_end() == 0 and res.end is not None and res.end_bytes == nbytes, "the gate keeps it"
    return f"resumed at {pre2['reused']}, forwarded {pre2['forwarded']}"


def test_spd257_the_park_takes_the_end_and_the_reservation_comes_back():
    res = _state(park_bytes=BIG)
    eng = app.STATE["engine"]
    res.reserve_end(eng)
    pair = res.end_bytes
    head = tokens(64, seed=70)
    a = head + tokens(60, seed=71)
    out_a, _, _ = turn(a, 30, session="A")
    b = head + tokens(70, seed=73)
    turn(b, 10, session="B")                          # A parked with its end: B's end re-reserved
    assert res.park_end is not None and res.end is not None, (res.park_end, res.end)
    assert res.stats["end_rereserved"] == 1 and res.end_bytes == pair, res.stats
    _, pre, _ = turn(a + out_a + TOOL, 4, session="A")  # the swap: A back at its end
    assert pre["kind"] == "end" and res.end_bytes == pair, pre
    assert res.park_owner == "B" and res.park_end is not None
    return f"one pair in place, one parked; A resumed at {pre['reused']}"


# ------------------------------------------------------------------ side slots and the stash

def _pool_home(**side_kw):
    res = _state(park_bytes=BIG)
    vals = {k: None for k in S.OVERLAY}
    side_kw.setdefault("guest_max", 200)
    side_res = cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, guest_frac=1.0, end=True, **side_kw)
    vals.update(engine=fresh(), drafter=None, max_len=512, resident=side_res,
                end_align=endalign.EndAlign(tb_of()))
    side = S.Slot(1, 512, values=vals)
    pool = S.Pool([S.Slot(0, 512), side])
    app.STATE["pool"] = pool
    side_res.reserve_end(vals["engine"])
    return res, pool, side, side_res


def on_side(pool, side, ids, n, session, no_carry=False):
    s = pool.take_slot(side.index, len(ids) + n, owner=session)
    assert s is side
    s.no_carry = no_carry
    S.bind(s)
    try:
        return turn(ids, n, session)
    finally:
        S.bind(None)
        pool.release_side(s)


def test_spd257_a_side_slots_idle_stash_goes_before_the_ends():
    saved = {k: app.STATE.get(k) for k in ("resident", "mem")}
    m = Budget(10 * BIG, 12 * BIG, 0, probe=lambda: (0, 0), empty=lambda: None)
    res0 = cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, end=True, mem=m)
    try:
        dict.__setitem__(app.STATE, "resident", res0)
        dict.__setitem__(app.STATE, "mem", m)
        m.register("stash_idle", lambda safe=False: res0.shed_stash(True))
        a = SimpleNamespace(prefix_chunk=CH, resident_tail=4, resident_guest_max=200)
        old = app.SIDE_RESIDENT_GB
        app.SIDE_RESIDENT_GB = 0.5
        try:
            side = app._side_resident(a, object())         # no tokenizer: no glue ids
        finally:
            app.SIDE_RESIDENT_GB = old
        side._stash_bufs = [torch.empty(1000, dtype=torch.float32)]
        side._stash_rows = 1000
        assert m.shedders["stash_idle"](True) == 4000 and side._stash_bufs is None
        side._stash_bufs = [torch.empty(1000, dtype=torch.float32)]
        side._stash = {"start": 0, "rows": 10, "views": []}       # a guest's rows wait in it
        assert m.shedders["stash_idle"](True) == 0 and side._stash_bufs is not None
    finally:
        for k, v in saved.items():
            dict.__setitem__(app.STATE, k, v)


GUEST = HEAD + tokens(40, seed=98)                    # a short request: a guest of S by size


def test_spd257_a_shed_during_the_guests_stash_copy_gives_nothing_back():
    """The race the side slot's `stash_idle` opens: another slot's gate sheds the stash while a guest's rows
    are being copied into it (`_stash_for`, before `_stash` names the buffers). Now it gives back nothing,
    the guest's turn ends with the slot's conversation put back, and its next turn resumes at its end.
    Without the guard (simulated: the buffers go mid-copy) the stash is lost and `settle_guest` cuts the
    rows the guest wrote (`stash_lost`) instead of failing."""
    res, pool, side, side_res = _pool_home()
    out_s, _, _ = on_side(pool, side, S_PROMPT, 40, "S")
    on_side(pool, side, GUEST, 8, "G")                # a first guest: the stash buffers exist now
    assert side_res._stash_bufs is not None
    real = cache._kv_views
    got = {}

    def views_with_a_gate(eng, drafter):
        # the other slot's gate, inside the copy: both the idle level and the stash level
        if side_res._stash_busy and not got:
            got["idle"] = side_res.shed_stash(True)
            got["stash"] = side_res.shed_stash(False)
            got["bufs"] = side_res._stash_bufs is not None
        return real(eng, drafter)
    cache._kv_views = views_with_a_gate
    try:
        on_side(pool, side, GUEST, 8, "G2")
    finally:
        cache._kv_views = real
    assert got == {"idle": 0, "stash": 0, "bufs": True}, got
    assert side_res.stats["stash_held"] == 2 and side_res.stats["stash_lost"] == 0, side_res.stats
    _, pre, _ = on_side(pool, side, S_PROMPT + out_s + TOOL, 8, "S")
    assert pre["kind"] == "end" and pre["forwarded"] == len(TOOL) + 1, pre
    # the buffers gone while the guest ran (what the guard prevents): the rows it wrote are cut
    on_side(pool, side, GUEST, 8, "G3")
    side_res._stash = {"start": 0, "rows": 4, "views": []}
    side_res._stash_bufs = None
    valid = side_res.valid
    side_res.settle_guest(side.values["engine"])
    assert side_res.stats["stash_lost"] == 1 and side_res.valid == 0 < valid, (side_res.valid, valid)
    return f"held {side_res.stats['stash_held']}, S resumed at {pre['reused']}"


def test_spd257_a_shed_where_the_guests_stash_is_cleared_gives_nothing_back():
    """The same race at the other end: the guest's rows go back at the next request (`settle_guest`), and another
    slot's gate sheds the idle stash (`shed_stash(True)`) at the point where the stash is cleared. Cleared before
    the copy, the shed found no stash and gave the buffers back: the stash counted as lost and the slot's
    conversation was cut. Now the buffers stay in use until the rows are back: nothing given back, nothing
    lost, and the conversation's next turn resumes at its end."""
    res, pool, side, side_res = _pool_home()
    out_s, _, _ = on_side(pool, side, S_PROMPT, 40, "S")
    on_side(pool, side, GUEST, 8, "G")
    assert side_res._stash is not None and side_res._stash_bufs is not None, "the guest's rows wait in the stash"
    got = []

    class AtTheClear(cache.ResidentPrefix):
        @property
        def _stash(self):
            return self.__dict__["_stash"]

        @_stash.setter
        def _stash(self, v):
            was = self.__dict__.get("_stash")
            self.__dict__["_stash"] = v
            if (v is None and was is not None and not got
                    and sys._getframe(1).f_code.co_name == "settle_guest"):
                got.append(self.shed_stash(True))      # another slot's gate, right after the stash is cleared
    assert type(side_res) is cache.ResidentPrefix
    held = side_res.stats["stash_held"]
    side_res.__class__ = AtTheClear
    try:
        _, pre, _ = on_side(pool, side, S_PROMPT + out_s + TOOL, 8, "S")
    finally:
        side_res.__class__ = cache.ResidentPrefix
    assert got == [0], got
    assert side_res.stats["stash_lost"] == 0 and side_res.stats["stash_held"] == held + 1, side_res.stats
    assert side_res._stash_bufs is not None and not side_res._stash_busy
    assert pre["kind"] == "end" and pre["forwarded"] == len(TOOL) + 1, pre
    return f"shed at the clear gave back {got[0]}, S resumed at {pre['reused']}"


def test_spd257_a_guest_stash_copy_and_a_gate_on_two_threads():
    """The same race on two threads: a gate's idle-stash shed loops while guests copy their rows; every
    stash the guests made is put back whole (no `stash_lost`) and every shed between them freed buffers."""
    res, pool, side, side_res = _pool_home()
    out_s, _, _ = on_side(pool, side, S_PROMPT, 40, "S")
    stop = threading.Event()
    freed = []

    def gate():
        while not stop.is_set():
            freed.append(side_res.shed_stash(True))
            time.sleep(0.0002)
    th = threading.Thread(target=gate)
    th.start()
    try:
        for i in range(6):
            on_side(pool, side, GUEST, 4, f"G{i}")
    finally:
        stop.set()
        th.join()
    assert side_res.stats["stash_lost"] == 0, side_res.stats
    _, pre, _ = on_side(pool, side, S_PROMPT + out_s + TOOL, 8, "S")
    assert pre["kind"] == "end", pre
    return f"{len(freed)} sheds, {sum(1 for f in freed if f)} gave buffers back, none lost"


def test_spd257_the_side_slots_log_line_names_its_policy():
    m = Budget(10 * BIG, 12 * BIG, 0, probe=Full(10 * BIG), empty=lambda: None)
    res, pool, side, side_res = _pool_home()
    side_res.end_reserved = False
    side_res._end_bufs = None
    side_res.mem = m
    app.STATE["verbose"] = True
    log = io.StringIO()
    try:
        with redirect_stdout(log):
            on_side(pool, side, S_PROMPT, 20, "S")
    finally:
        app.STATE["verbose"] = False
    assert "end anchor not kept (memory, allocated " in log.getvalue(), log.getvalue()[-400:]
    assert "no policy" not in log.getvalue()


# ------------------------------------------------------------------ a guest chat on a side slot keeps its end

def _pool_sides(k: int):
    """Slot 0 and `k` side slots, each with a resident prefix that keeps its end."""
    _state(park_bytes=BIG)
    sides = []
    for i in range(1, k + 1):
        vals = {k_: None for k_ in S.OVERLAY}
        r = cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, guest_frac=1.0, end=True, guest_max=200)
        vals.update(engine=fresh(), drafter=None, max_len=512, resident=r,
                    end_align=endalign.EndAlign(tb_of()))
        r.reserve_end(vals["engine"])
        sides.append(S.Slot(i, 512, values=vals))
    pool = S.Pool([S.Slot(0, 512)] + sides)
    pool.lru = True
    app.STATE["pool"] = pool
    return pool, sides


def _guest_reference():
    _state()
    out1, _, _ = turn(GUEST, 30, session="G")
    out2, pre2, _ = turn(GUEST + out1 + TOOL, 24, session="G")
    assert pre2["kind"] == "end"
    return out1, out2


def _guest_then_follow_up(mode: str, idle: bool = True, no_carry: bool = False):
    old = app.SIDE_GUEST_END
    try:
        app.SIDE_GUEST_END = mode
        res, pool, side, side_res = _pool_home()
        on_side(pool, side, S_PROMPT, 60, "S")
        s_end = side_res.end["at"]
        if idle:
            side_res.t_used -= app.SIDE_GUEST_IDLE_S + 1
        out1, _, _ = on_side(pool, side, GUEST, 30, "G", no_carry=no_carry)
        promoted = side_res.stats["guests_promoted"]
        owner, end = side_res.owner, (side_res.end or {}).get("at")
        out2, pre2, _ = on_side(pool, side, GUEST + out1 + TOOL, 24, "G")
        return SimpleNamespace(out1=out1, out2=out2, pre2=pre2, promoted=promoted, owner=owner,
                               end=end, s_end=s_end, side_res=side_res)
    finally:
        app.SIDE_GUEST_END = old


def test_spd257_a_guest_chat_on_a_side_slot_keeps_its_end_and_writes_the_same_tokens():
    ref1, ref2 = _guest_reference()
    r = _guest_then_follow_up("promote")
    assert r.promoted == 1 and r.owner == "G" and r.end == len(GUEST) + 29, (r.promoted, r.owner, r.end)
    assert r.out1 == ref1
    assert r.pre2["kind"] == "end" and r.pre2["forwarded"] == len(TOOL) + 1, r.pre2
    assert r.out2 == ref2, "the gate: the same greedy tokens as the chat run alone"
    return f"promoted; follow-up resumed at {r.pre2['reused']}, forwarded {r.pre2['forwarded']}"


def test_spd257_regression_off_the_guests_follow_up_reads_it_again():
    r = _guest_then_follow_up("off")
    assert r.promoted == 0 and r.owner == "S", (r.promoted, r.owner)
    assert r.pre2["kind"] != "end" and r.pre2["forwarded"] > len(TOOL) + 1, r.pre2
    return f"off: the follow-up forwarded {r.pre2['forwarded']}"


def test_spd257_a_recent_longer_conversation_keeps_the_slot():
    r = _guest_then_follow_up("promote", idle=False)
    assert r.promoted == 0, "S was served just now and holds more than the guest: it stays"


def test_spd257_the_routers_question_never_takes_the_slot():
    r = _guest_then_follow_up("promote", no_carry=True)
    assert r.promoted == 0


def test_spd257_a_waiting_parent_or_worker_keeps_its_side_slot():
    """ENG-371 against the promotion: S on a side slot waits for its worker (or is the worker),
    idle past SIDE_GUEST_IDLE_S, and a short new chat G runs there as a guest. G is not promoted, the
    stash puts S's rows back, nothing is cut without the keeper, and S's return resumes at its end."""
    old = app.SIDE_GUEST_END
    for role in ("parent", "worker"):
        calls = []
        try:
            app.SIDE_GUEST_END = "promote"
            res, pool, side, side_res = _pool_home()
            side_res.keeper = lambda r, e, d, stay=False: calls.append((r.owner, r.valid))
            out_s, _, _ = on_side(pool, side, S_PROMPT, 60, "S")
            s_end = side_res.end["at"]
            if role == "parent":
                app._note_parent("S")
            else:
                app._note_parent(None, worker="S")
            assert app._comes_back("S")
            side_res.t_used -= app.SIDE_GUEST_IDLE_S + 1
            on_side(pool, side, GUEST, 30, "G")
            assert side_res.stats["guests_promoted"] == 0 and side_res.owner == "S", (role, side_res.owner)
            assert calls == [] and side_res.end["at"] == s_end, (role, calls, side_res.end)
            _, pre, _ = on_side(pool, side, S_PROMPT + out_s + TOOL, 8, "S")
            assert pre["kind"] == "end" and pre["forwarded"] == len(TOOL) + 1, (role, pre)
        finally:
            app.SIDE_GUEST_END = old
            app.PARENTS.pop("S", None)
            app.WORKERS.pop("S", None)
    return f"not promoted; S resumed at {pre['reused']}, forwarded {pre['forwarded']}"


def test_spd257_an_end_that_cannot_be_kept_cuts_nothing():
    """A promotion checks what `keep_end` needs before it cuts the old conversation: a pending
    block past the committed tokens that cannot be cut back ('ahead'), a drafter that cannot resume in
    place ('off'), no end buffers and no room ('memory'). Each leaves S in place and the stash waiting."""
    old = app.SIDE_GUEST_END
    try:
        app.SIDE_GUEST_END = "off"                    # G's turn ends as a guest; promoted by hand below
        res, pool, side, side_res = _pool_home()
        on_side(pool, side, S_PROMPT, 60, "S")
        out_g, _, _ = on_side(pool, side, GUEST, 30, "G")
        eng = side.values["engine"]
        n = len(GUEST)
        ctx = GUEST + out_g[:29]                      # the context `_remember` hands over
        assert side_res._stash is not None and int(eng.kv.length) == len(ctx)
        valid, owner = side_res.valid, side_res.owner
        real_trim = eng.trim_pending
        eng.trim_pending = lambda k: False
        try:
            assert not side_res.promote_guest(eng, None, ctx[:-1], n, "G"), "ahead"
        finally:
            eng.trim_pending = real_trim
        no_resume = SimpleNamespace(can_resume=lambda: False)
        assert not side_res.promote_guest(eng, no_resume, ctx, n, "G"), "off"
        side_res.end_reserved, bufs, side_res._end_bufs = False, side_res._end_bufs, None
        side_res.mem = _full()
        assert not side_res.promote_guest(eng, None, ctx, n, "G"), "memory"
        assert side_res.stats["guest_end_unkept"] == 3
        assert side_res.valid == valid and side_res.owner == owner and side_res._stash is not None
        side_res.mem, side_res._end_bufs, side_res.end_reserved = None, bufs, True
        assert side_res.promote_guest(eng, None, ctx, n, "G") and side_res.owner == "G"
        assert side_res.keep_end(eng, None, ctx, n) == "kept"
    finally:
        app.SIDE_GUEST_END = old


def test_spd257_side_home_routes_the_promoted_guests_follow_up_to_its_slot():
    """The follow-up is routed (`_side_home`, then LRU `take_side`), not forced onto the slot: promoted, G's
    next turn names slot 1 its home and resumes at its end there; off, it names no home, LRU sends it to
    the slot never served (slot 2), and it reads the whole chat again."""
    old = app.SIDE_GUEST_END
    got = {}
    for mode in ("promote", "off"):
        try:
            app.SIDE_GUEST_END = mode
            pool, (s1, s2) = _pool_sides(2)
            on_side(pool, s1, S_PROMPT, 60, "S")
            s1.values["resident"].t_used -= app.SIDE_GUEST_IDLE_S + 1
            out1, _, _ = on_side(pool, s1, GUEST, 30, "G")
            ids = GUEST + out1 + TOOL
            need = len(ids) + 24
            home = app._side_home(pool, ids, need, CH, None)
            if home is not None:
                slot = pool.take_slot(home[0], need, owner="G")
            else:
                slot = pool.take_side(need)
            assert slot is not None
            pool.release_side(slot)
            _, pre, _ = on_side(pool, slot, ids, 24, "G")
            got[mode] = (home, slot.index, pre["kind"], pre["forwarded"])
        finally:
            app.SIDE_GUEST_END = old
    assert got["promote"][0] is not None and got["promote"][0][0] == 1 and got["promote"][1] == 1, got
    assert got["promote"][2] == "end" and got["promote"][3] == len(TOOL) + 1, got
    assert got["off"][0] is None and got["off"][1] == 2 and got["off"][3] == len(ids), got
    return f"promote: home {got['promote'][0]}, forwarded {got['promote'][3]}; off: slot 2, forwarded {got['off'][3]}"


def test_spd257_take_side_gives_the_slot_served_longest_ago():
    def slot(i, t):
        r = SimpleNamespace(t_used=t)
        return S.Slot(i, 1000, values={"resident": r})
    pool = S.Pool([S.Slot(0, 1000), slot(1, 300.0), slot(2, 100.0), slot(3, 200.0)])
    assert pool.take_side(10).index == 1, "off: the lowest free slot"
    pool.release_side(pool.slots[1])
    pool.lru = True
    assert pool.take_side(10).index == 2
    assert pool.take_side(10).index == 3
    assert pool.take_side(10).index == 1


# ------------------------------------------------------------------ the report by owner

def test_spd257_the_memory_report_names_the_owners_and_the_graphs_pools():
    sa = SegAlloc()
    _engine(sa, 55.0)
    _graphs(sa, 1.41, 1.0)
    for _ in range(5):
        sa.malloc(GIB, owner="caches")
    m = live_budget(sa)
    res, pool, side, side_res = _pool_home()
    res.reserve_end(app.STATE["engine"])
    dict.__setitem__(app.STATE, "mem", m)
    try:
        app.STATE["mem_load"] = {"allocated": int(55 * GIB), "reserved": int(57 * GIB),
                                 "caches": res.end_bytes + side_res.end_bytes}
        on_side(pool, side, S_PROMPT, 20, "S")
        with FakeTorchCuda(sa):
            o = app.memory_owners(pools=True)
            rep = m.report()
        assert set(o["by_owner"]) == {"slot0", "slot1"}, o["by_owner"].keys()
        s1 = o["by_owner"]["slot1"]
        assert s1["end"] == side_res.end_bytes and s1["end_reserved"] and s1["end_at"] == side_res.end["at"]
        assert s1["anchors"] == len(side_res.anchors) * side_res._anchor_bytes
        assert o["caches"] == sum(v["total"] for v in o["by_owner"].values())
        alloc = sa.probe()[0]
        assert o["above_load"] == alloc - int(55 * GIB)
        assert o["unattributed"] == o["above_load"] - o["caches_since_load"]
        assert o["limits"]["room"] == m.ceiling - m.slack - alloc
        g = o["allocator_pools"]["graphs"]
        assert g["pools"] == 1 and g["reserved"] == int(1.41 * GIB) and g["free"] == int(1.41 * GIB) - GIB, g
        assert o["limits"]["pools"]["graphs_free"] == g["free"] and o["limits"]["default_free"] == 0
        assert rep["pools"] is not None and rep["pools"]["graphs_reserved"] == g["reserved"], rep["pools"]
        h = app._owners_gib(o)
        assert set(h["by_part_gb"]) >= {"anchors", "end", "stash", "pin", "park"} and "unattributed_gb" in h
        return f"caches {o['caches']} B, graphs' pools {g['reserved'] / GIB:.2f} GiB ({g['free'] / GIB:.2f} free)"
    finally:
        app.STATE.pop("mem_load", None)


def test_spd257_health_builds_the_owner_summary_at_most_every_ttl():
    """`/health` is built once a second under polling; the summary by owner walks every slot's buffers, so
    it is rebuilt at most every OWNERS_TTL_S and `/v1/cache/stats` does not carry it."""
    calls = []
    real = app.memory_owners
    app.memory_owners = lambda pools=False: calls.append(pools) or {"by_owner": {}, "caches": 0}
    old = (list(app._OWNERS_LAST), app.OWNERS_TTL_S)
    try:
        app._OWNERS_LAST[:] = [0.0, None]
        a = app._owners_cached()
        b = app._owners_cached()
        assert len(calls) == 1 and a["age_s"] == 0.0 and b["age_s"] >= 0.0, (calls, a, b)
        app.OWNERS_TTL_S = 0.0
        app._owners_cached()
        assert len(calls) == 2
        assert "memory_owners" not in app.cache_stats()
    finally:
        app.memory_owners = real
        app._OWNERS_LAST[:], app.OWNERS_TTL_S = old


def test_spd257_the_req_line_counts_the_policys_time_by_part():
    """The [req] line's gate/done/remember/settle parts (`cache.req_ms_fields`): a request's gate is its own
    `gate` part, the `done` the engine ran before it is its `done` part, a guest's rows put back are its `settle`
    part; each request counts from 0, another thread's work counts in none of them, and a request with no parts
    prints nothing. A fake clock moves 3 ms at each reading of the allocator's counters and 5 ms for a guest's
    rows."""
    import threading

    clock = {"t": 0.0, "reads": 0}
    real = cache.time.perf_counter
    cache.time.perf_counter = lambda: clock["t"]

    def probe():
        clock["t"] += 0.003
        clock["reads"] += 1
        return (0, 0)

    def reads(fn, *a, **kw):
        n = clock["reads"]
        fn(*a, **kw)
        return 3.0 * (clock["reads"] - n)

    m = Budget(10 * BIG, 12 * BIG, 0, probe=probe, empty=lambda: None)
    res = cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, end=True, mem=m)
    res._settle = lambda eng, st: clock.__setitem__("t", clock["t"] + 0.005)
    try:
        r1 = cache.req_ms_start()
        g1 = reads(m.gate, 100)
        res.settle_guest(None)                    # no stash: nothing put back
        res._stash = {"rows": 8}
        res.settle_guest(None)                    # a guest's rows go back: 5 ms
        cache.req_ms_add("remember", 2.0)
        first = cache.req_ms_fields(r1)
        d1 = reads(m.done)                        # after the [req] line: the next request's `done` part
        r2 = cache.req_ms_start()
        g2 = reads(m.gate, 100) + reads(m.gate, 100, count=False)    # a second pass adds to the part
        t = threading.Thread(target=lambda: (m.gate(100), cache.req_ms_add("remember", 9.0)))
        t.start()                                 # another slot's request, on a thread no request started
        t.join()
        second = cache.req_ms_fields(r2)
    finally:
        cache.time.perf_counter = real
        cache._REQ.__dict__.pop("ms", None)
    assert g1 > 0 and d1 > 0, (g1, d1)
    assert first == f" gate_ms={g1:.1f} done_ms=0.0 remember_ms=2.0 settle_ms=5.0", first
    assert m.stats["last_done_ms"] == d1, (m.stats["last_done_ms"], d1)
    assert second == f" gate_ms={g2:.1f} done_ms={d1:.1f} remember_ms=0.0 settle_ms=0.0", second
    assert cache.req_ms_fields(None) == ""
    assert res._stash is None and not res._stash_busy
    return f"first{first}; second{second}"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:80s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
