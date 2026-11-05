"""The engine's memory limits (OPS-47, QA OPS-48) on a fake allocator: the policy, not the model.

`engine/cache.py::Budget` decides three things -- may a cache grow, must caches go back before a
prefill, may a prefill run at all -- from the allocator's two numbers (allocated, reserved). These
tests give it an allocator whose numbers the test sets, and check each decision against the limits
The owner set on 2026-10-01: a 70 GiB ceiling between requests, a 77 GiB declared peak, a 2 GiB
reserve slack, and the shed order store, body anchors, stash, park, pin.

The scenarios are OPS-47's Gherkin (rows of OPS-48): 2 evict before clone, 3 the ordered shed,
4 the 503 instead of a wedge, 5 the reserve back at idle, 7 compaction drops a session's entries,
8 the planner counts the engine's peak. The model-level rows (1, 6, 9) are tests/test_resident_park.py.

Run: python tests/test_budget.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import cache  # noqa: E402
from engine.cache import GIB, Budget, MemoryRefused, StateStore  # noqa: E402


class FakeAlloc:
    """The CUDA caching allocator's two counters, and the peak since a reset."""

    def __init__(self, allocated_gib: float, reserve_gib: float = 0.0):
        self.alloc = int(allocated_gib * GIB)
        self.res = self.alloc + int(reserve_gib * GIB)
        self.peak = self.alloc
        self.empties = 0

    def probe(self):
        return self.alloc, self.res

    def empty(self):
        self.empties += 1
        self.res = self.alloc

    def peak_probe(self, reset=False):
        if reset:
            self.peak = self.alloc
            return 0
        return self.peak

    def take(self, n):
        self.alloc += n
        self.res = max(self.res, self.alloc)
        self.peak = max(self.peak, self.alloc)

    def give(self, n):
        self.alloc -= n                       # freed blocks stay in the reserve


def budget(fa: FakeAlloc, ceiling=70, peak=77, slack=2, **kw) -> Budget:
    return Budget(int(ceiling * GIB), int(peak * GIB), int(slack * GIB), probe=fa.probe,
                  empty=fa.empty, peak_probe=fa.peak_probe, **kw)


def shedder(fa: FakeAlloc, order: list, level: str, gib: float, safe_ok: bool = True):
    """A cache holding `gib` that gives it all back when asked (and records the order)."""
    held = {"b": int(gib * GIB)}

    def fn(safe):
        if safe and not safe_ok:
            return 0
        b, held["b"] = held["b"], 0
        if b:
            order.append(level)
            fa.give(b)
        return b
    return fn


# ------------------------------------------------------------------ 3 and 4: the gate

def test_under_both_limits_nothing_is_shed():
    fa = FakeAlloc(60, 1)
    m = budget(fa)
    order = []
    for lv in Budget.LEVELS:
        m.register(lv, shedder(fa, order, lv, 1))
    g = m.gate(1000)
    assert order == [] and g["shed"] == [] and m.ws_now == cache.WORKSPACE_FLOOR


def test_over_the_ceiling_the_reserve_goes_first_then_the_caches_cheapest_first():
    fa = FakeAlloc(69.5, 3)          # reserved 72.5 over the 70 ceiling, 3 of it reserve
    m = budget(fa)
    order = []
    for lv, gib in (("store", 1), ("anchors", 1), ("stash", 1), ("park", 2), ("pin", 4)):
        m.register(lv, shedder(fa, order, lv, gib))
    m.gate(1000)
    # the reserve went back (72.5 -> 69.5): no cache had to go
    assert order == [] and fa.empties == 1 and fa.res <= 70 * GIB
    fa.take(int(1.5 * GIB))           # 71.0 allocated, no reserve: now caches go
    m.gate(1000)
    assert order == ["store"], order
    assert fa.res <= 70 * GIB


def test_a_capped_workspace_stops_growing_and_a_larger_prefill_raises_the_cap():
    """SPD-99: Flash-Next's prefill workspace grows to about 140k tokens and then stops (10.46 GiB at
    147,149, 10.47 at 261,549): the adapter gives a cap, the estimate stops there, a long prefill that
    took more raises it; without a cap (the 27B) the line goes on as before."""
    fa = FakeAlloc(50)
    m = budget(fa, peak=90, ws_floor=int(2.75 * GIB), ws_per_token=76309, ws_cap=int(10.75 * GIB))
    assert m.workspace(8192) == int(2.75 * GIB)
    assert abs(m.workspace(65_536) - 76309 * 65_536) == 0
    assert m.workspace(262_144) == int(10.75 * GIB)
    m.mark()
    fa.take(int(11.5 * GIB))                           # a prefill that took 11.5 GiB above the engine
    fa.give(int(11.5 * GIB))
    m.prefilled(200_000)
    assert m.ws_cap == int(11.5 * GIB) and any(e["kind"] == "estimate_raised" and "cap" in e for e in m.events)
    plain = budget(FakeAlloc(50))
    assert plain.ws_cap is None and plain.workspace(262_144) == cache.WORKSPACE_PER_TOKEN * 262_144
    return "the floor, the line, the cap; a larger prefill raises it; no cap for the 27B"


def test_nemotron_h_workspace_is_flat_so_a_million_token_prefill_runs():
    """ENG-363: NemotronH's prefill workspace does not grow with the prompt (3.34 GiB above the engine at
    131,054 tokens, 3.18 at 249,983 on Super VL): its adapter says so, and a 1,000,000-token prefill with
    the engine at the live numbers (80.6 GiB allocated, 82.4 reserved, peak 94.8 with slot 0's window
    and three side slots) is admitted. The 27B's line refused it on 2026-10-16 ("needs 107.3 GiB")."""
    from types import SimpleNamespace

    from engine.models.nemotron_h.serving import NemotronHServing
    srv = NemotronHServing.__new__(NemotronHServing)
    srv.st = SimpleNamespace(conv={}, ssm={})
    srv.cfg = SimpleNamespace(kv_bytes_per_token=lambda dtype_bytes: 8192)
    c = srv.cost()
    assert c.view_bytes_per_token == 8192
    fa = FakeAlloc(80.6, reserve_gib=1.8)
    m = budget(fa, ceiling=90.8, peak=94.8, ws_floor=c.workspace_floor, ws_per_token=c.workspace_per_token,
               ws_cap=c.workspace_cap)
    assert m.workspace(8192) == int(3.5 * GIB) and m.workspace(1_000_000) == int(4.0 * GIB)
    g = m.gate(1_000_000)
    assert g["workspace"] == int(4.0 * GIB)
    m.done()
    old = budget(FakeAlloc(80.6, reserve_gib=1.8), ceiling=90.8, peak=94.8)
    try:
        old.gate(1_000_000)
        raise AssertionError("the 27B's line admitted a 1,000,000-token prefill")
    except MemoryRefused:
        pass
    # what a long prefill teaches stays under the cap (3.34 GiB at 131,054 tokens)
    m.mark()
    fa.take(int(3.34 * GIB))
    fa.give(int(3.34 * GIB))
    m.prefilled(131_054)
    assert m.workspace(1_000_000) == int(4.0 * GIB)
    return "4.0 GiB at 1,000,000 tokens: admitted; the 27B's 28 GiB: refused"


def test_slot_0s_window_above_the_limits_window_raises_ceiling_and_peak():
    """ENG-363: MEM_CEILING_GIB and MEM_PEAK_GIB are set at a 262,144-token window; slot 0 at 1,048,576
    holds 786,432 more rows of KV from the load on (pool off), so both limits go up by them (6 GiB on
    Super VL), kept in `fixed` like the side slots'. At 262,144, with limits off or with no engine
    cost, nothing moves."""
    from types import SimpleNamespace

    from server import app as A
    from server import memlimits
    eng = SimpleNamespace(cost=lambda: SimpleNamespace(view_bytes_per_token=8192))
    old = A.STATE.get("mem")
    try:
        m = A.STATE["mem"] = budget(FakeAlloc(80), ceiling=76, peak=80)
        A._window_above_limits(SimpleNamespace(max_len=1_048_576), eng)
        extra = 8192 * (1_048_576 - memlimits.LIMITS_WINDOW)
        assert extra == 6 * GIB
        assert (m.ceiling, m.peak, m.fixed) == (int(76 * GIB) + extra, int(80 * GIB) + extra, extra)
        m = A.STATE["mem"] = budget(FakeAlloc(80), ceiling=76, peak=80)
        A._window_above_limits(SimpleNamespace(max_len=262_144), eng)
        assert (m.ceiling, m.fixed) == (int(76 * GIB), 0)
        m = A.STATE["mem"] = budget(FakeAlloc(80), ceiling=0, peak=0)      # limits off
        A._window_above_limits(SimpleNamespace(max_len=1_048_576), eng)
        assert m.fixed == 0
    finally:
        A.STATE["mem"] = old
    return "+6 GiB at 1,048,576; nothing at 262,144 or with the limits off"


def test_a_long_prefill_sheds_store_anchors_stash_park_pin_in_that_order_until_it_fits():
    fa = FakeAlloc(70)
    m = budget(fa)
    order = []
    for lv, gib in (("store", 2), ("anchors", 1.5), ("stash", 0.9), ("park", 3), ("pin", 4)):
        m.register(lv, shedder(fa, order, lv, gib))
    n = 262_144                       # the workspace at the full window: ~7.3 GiB
    ws = m.workspace(n)
    assert 7.0 * GIB < ws < 7.6 * GIB, ws / GIB
    g = m.gate(n)
    # 70 + 7.33 = 77.33 > 77: the store alone (2 GiB) brings it to 75.33
    assert order == ["store"] and g["shed"] == ["store"], order
    fa.take(int(4.5 * GIB))           # caches grew back and more: 72.5 + 7.33 needs three levels
    order.clear()
    g = m.gate(n)
    assert order == ["anchors", "stash", "park"], order
    assert fa.res + ws <= 77 * GIB
    assert m.stats["shed"]["park"] == 1 and [e["level"] for e in m.events if e["kind"] == "shed"]


def _partial(fa, asked, gib, margin=0.25, chunk=None, stuck=None):
    """A store of `gib` that gives back what the gate needs plus `margin` (SPD-249 `StateStore.shed_to`), at most
    `chunk` GiB a call; `stuck` (a dict) keeps the freed bytes in the reserve (the allocator's cache)."""
    held = {"b": int(gib * GIB), "heads": []}

    def fn(safe, need, heads=True):
        asked.append(need)
        held["heads"].append(heads)
        b = min(held["b"], need + int(margin * GIB), int(chunk * GIB) if chunk else 1 << 62)
        held["b"] -= b
        fa.give(b)
        if stuck is not None:
            stuck["b"] += b
        return {"freed": b, "dropped": 1, "heads_kept": 1 if held["b"] else 0}
    return fn, held


class StoreCacheAlloc(FakeAlloc):
    """An allocator whose cache keeps what the store frees: `empty` gives back everything else."""

    def __init__(self, allocated_gib):
        super().__init__(allocated_gib)
        self.stuck = {"b": 0}

    def empty(self):
        self.empties += 1
        self.res = self.alloc + self.stuck["b"]


def test_spd249_a_chunk_that_lowers_the_reserve_lets_the_next_chunk_go():
    """Store chunks of 0.5 GiB, over by 1.2 GiB: each chunk lowers the reserve, so three go; the rest of the store
    and every other level stay; the first chunk is plain entries only, the later ones may reach the heads."""
    fa = FakeAlloc(71.2)
    m = budget(fa)
    asked, order = [], []
    fn, held = _partial(fa, asked, 5, chunk=0.5)
    m.register("store", shedder(fa, order, "store", 5))
    m.register_partial("store", fn)
    m.register("anchors", shedder(fa, order, "anchors", 1))
    m.gate(1000)
    chunks = [e for e in m.events if e["kind"] == "shed_chunk"]
    assert len(chunks) == 3 and all(c["decision"] == "continue" for c in chunks), chunks
    assert held["heads"] == [False, True, True] and order == [] and fa.res <= 70 * GIB, (held, order)
    assert all(c["reserved_before"] - c["reserved_after"] == c["freed"] for c in chunks), chunks
    return "3 chunks of 0.5 GiB, each lowering the reserve; heads open from the second; nothing else shed"


def test_spd249_a_chunk_the_reserve_keeps_stops_the_store_and_the_next_level_goes():
    """The store's frees stay in the allocator's cache: the first chunk (plain entries only) leaves the reserve
    where it was, the store's shed stops there (its heads untouched) and the anchors go."""
    fa = StoreCacheAlloc(71)
    m = budget(fa)
    asked, order = [], []
    fn, held = _partial(fa, asked, 5, stuck=fa.stuck)
    m.register("store", shedder(fa, order, "store", 5))
    m.register_partial("store", fn)
    m.register("anchors", shedder(fa, order, "anchors", 1.5))
    g = m.gate(1000)
    chunks = [e for e in m.events if e["kind"] == "shed_chunk"]
    assert len(chunks) == 1 and chunks[0]["decision"].startswith("stop"), chunks
    assert held["heads"] == [False] and held["b"] == int(3.75 * GIB), held
    assert order == ["anchors"] and g["shed"] == ["store", "anchors"] and fa.res <= 70 * GIB, (order, g, fa.res)
    return "one chunk the reserve kept: the store stopped (3.75 GiB and its heads kept), the anchors went"


def test_spd249_chunks_that_keep_lowering_the_reserve_still_empty_the_store():
    """Over by more than the store holds, every chunk lowering the reserve: the store empties, then the next level."""
    fa = FakeAlloc(73)
    m = budget(fa)
    asked, order = [], []
    fn, held = _partial(fa, asked, 2, chunk=0.75)
    m.register("store", shedder(fa, order, "store", 2))
    m.register_partial("store", fn)
    m.register("anchors", shedder(fa, order, "anchors", 1.5))
    m.gate(1000)
    chunks = [e for e in m.events if e["kind"] == "shed_chunk"]
    assert held["b"] == 0 and len(chunks) == 3 and all(c["decision"] == "continue" for c in chunks), (held, chunks)
    assert order == ["anchors"] and fa.res <= 70 * GIB, order
    return "three chunks, each lowering the reserve: the store empty, then the anchors"


def test_spd249_the_gate_sheds_only_the_store_bytes_it_needs():
    """Over the ceiling by 1 GiB with a 5 GiB store: the gate asks the store for 1 GiB (its partial shedder adds
    the margin), the rest of the store stays, no other level is touched; the shed's log line has the need."""
    fa = FakeAlloc(71)
    m = budget(fa)
    asked, order = [], []
    fn, held = _partial(fa, asked, 5)
    m.register("store", shedder(fa, order, "store", 5))
    m.register_partial("store", fn)
    m.register("anchors", shedder(fa, order, "anchors", 1))
    g = m.gate(1000)
    assert asked == [1 * GIB] and held["b"] == int(3.75 * GIB), (asked, held)
    assert order == [] and g["shed"] == ["store"] and fa.res <= 70 * GIB, (order, g)
    ev = [e for e in m.events if e["kind"] == "shed"][-1]
    assert ev["level"] == "store" and ev["need"] == 1 * GIB and ev["freed"] == int(1.25 * GIB), ev
    return "1 GiB over: 1.25 GiB of a 5 GiB store given back, the other levels kept"


def test_spd249_a_store_too_small_for_the_need_empties_and_the_next_level_follows():
    """Over by 3 GiB with a 2 GiB store: the store goes whole (as before), then the next level."""
    fa = FakeAlloc(73)
    m = budget(fa)
    asked, order = [], []
    fn, held = _partial(fa, asked, 2)
    m.register("store", shedder(fa, order, "store", 2))
    m.register_partial("store", fn)
    m.register("anchors", shedder(fa, order, "anchors", 1.5))
    g = m.gate(1000)
    assert held["b"] == 0 and order == ["anchors"] and g["shed"] == ["store", "anchors"], (held, order, g)
    assert fa.res <= 70 * GIB
    return "the store emptied, then the anchors: the order of the levels unchanged"


def test_with_nothing_left_to_shed_the_prefill_is_refused_not_run():
    fa = FakeAlloc(75)                # e.g. a decision model in the same allocator, not excluded
    m = budget(fa)
    order = []
    m.register("store", shedder(fa, order, "store", 0.5))
    try:
        m.gate(262_144)
        raise AssertionError("a prefill over the peak must be refused")
    except MemoryRefused as exc:
        assert exc.peak == 77 * GIB and exc.need > exc.peak
    assert order == ["store"] and m.stats["refused"] == 1
    assert any(e["kind"] == "refused" for e in m.events)
    assert m.ws_now == 0, "a refused prefill reserves no workspace"


class StuckAlloc(FakeAlloc):
    """2026-10-04 16:41: a reserve `empty` cannot give back (a role's own pool keeps its free
    segments while it lives); `release()` drops the pool, and the next empty returns it."""

    def __init__(self, allocated_gib, stuck_gib, reserve_gib=0.0):
        super().__init__(allocated_gib, reserve_gib)
        self.stuck = int(stuck_gib * GIB)
        self.res += self.stuck

    def empty(self):
        self.empties += 1
        self.res = self.alloc + self.stuck

    def release(self, safe=False):
        b, self.stuck = self.stuck, 0
        return b


def test_a_roles_pool_full_of_free_segments_is_given_back_before_a_refusal():
    # the incident's numbers: 63 GiB in use, 17.6 GiB free inside the shadow's pool, peak 77
    fa = StuckAlloc(63.2, 17.6)
    m = budget(fa)
    order = []
    for lv in ("store", "anchors", "end", "stash", "park", "pin"):
        m.register(lv, shedder(fa, order, lv, 0))
    m.register("roles", fa.release)
    g = m.gate(587)
    assert g["shed"] == ["roles"] and fa.stuck == 0 and fa.res <= 77 * GIB, g
    assert m.stats["refused"] == 0 and m.refused_in_a_row == 0


def test_a_tiny_prefill_is_never_refused_on_a_stale_reserve_the_engine_can_use():
    fa = StuckAlloc(63.2, 17.6)       # stuck in the engine's own allocator (fragmentation)
    m = budget(fa)
    g = m.gate(587)
    assert g.get("stale") and m.stats["stale_admits"] == 1, g
    assert any(e["kind"] == "stale_admit" for e in m.events)
    # a prefill whose workspace does not fit in the free reserve is still refused
    try:
        m.gate(262_144 * 8)
        raise AssertionError("refused expected")
    except MemoryRefused:
        pass
    assert m.refused_in_a_row == 1
    m.gate(12)
    assert m.refused_in_a_row == 0


def test_the_free_reserve_inside_a_roles_pool_does_not_count_as_the_engines():
    fa = StuckAlloc(63.2, 17.6)
    m = budget(fa)
    m.foreign_free = lambda: fa.stuck   # all of it is the role's, nothing releases it
    for _ in range(3):
        try:
            m.gate(587)
            raise AssertionError("refused expected")
        except MemoryRefused:
            pass
    assert m.refused_in_a_row == 3 and m.events[-1]["in_a_row"] == 3
    assert m.events[-1]["allocated"] < m.events[-1]["reserved"]


def test_srv775_the_workspace_spare_is_the_default_pools_free_bytes_less_the_roles_pools():
    """SRV-775: `workspace_spare`, the admission's credit for a prefill admitted over the peak: the free default-pool
    bytes (`default_free`: the free reserve less the roles' pools by `foreign_free` and the graphs' scratch) less the
    roles' pools' free bytes of the same reading. The saved state of 2026-11-02 in round figures: 64.7 allocated,
    15.0 free in the reserve, 12.7 of it the graphs' scratch, 1.1 free in a role's pool: 2.3 free default-pool bytes,
    1.2 credited. A credit only on a live reading: a walk that fails, with or without a reading before it (the
    production probe keeps its last one in `.last`, which `default_free` falls back to), and no pools probe at all
    give none."""
    fa = FakeAlloc(64.7, 15.0)
    m = budget(fa, ceiling=64.8, peak=75.8)
    reading = {"graphs_free": int(12.7 * GIB), "mempools_free": int(1.1 * GIB)}
    m.pools_probe = lambda: reading
    alloc, res = m.memory()
    assert m.default_free(alloc, res) == int(15.0 * GIB) - int(12.7 * GIB)
    assert m.workspace_spare() == int(15.0 * GIB) - int(12.7 * GIB) - int(1.1 * GIB), m.workspace_spare() / GIB
    reading["mempools_free"] = int(3 * GIB)                     # more than the default pool's free bytes
    assert m.workspace_spare() == 0
    m.foreign_free = lambda: int(0.5 * GIB)
    reading["mempools_free"] = 0
    assert m.workspace_spare() == int(15.0 * GIB) - int(0.5 * GIB) - int(12.7 * GIB)

    def failing():
        raise RuntimeError("walk failed")
    m.pools_probe = failing
    assert m.workspace_spare() == 0, "no reading: no credit on the graphs' scratch"

    class LastThenFails:
        """`TorchPools`' shape: the last reading in `.last`, and a walk that now fails."""

        def __init__(self, last):
            self.last = last

        def __call__(self):
            raise RuntimeError("walk failed")
    m.pools_probe = LastThenFails({"graphs_free": int(12.7 * GIB), "mempools_free": int(1.1 * GIB)})
    stale = m.default_free(alloc, res)
    assert stale == int(15.0 * GIB) - int(0.5 * GIB) - int(12.7 * GIB), stale / GIB
    assert m.workspace_spare() == 0, f"a failed walk credited the last one's {stale / GIB:.2f} GiB"
    m.pools_probe = None
    assert m.workspace_spare() == 0, "no pools probe: no reading, no credit"
    return (f"2.3 free default-pool bytes, 1.2 credited; a failed walk 0 (the last walk's {stale / GIB:.1f} not "
            f"credited), no probe 0")


def test_the_decision_models_are_not_the_engine():
    """Laya's checkpoints live in the same allocator but are roles with claims of their own."""
    fa = FakeAlloc(78)
    m = budget(fa, exclude=lambda: 9 * GIB)
    assert m.memory() == (69 * GIB, 69 * GIB)
    m.gate(1000)                      # 69 + 1.15 is under the peak and 69 under the ceiling


def test_limits_off_is_the_old_engine():
    fa = FakeAlloc(100)
    m = budget(fa, ceiling=0, peak=0)
    assert m.gate(262_144) == {} and m.admit("park", 10 * GIB) and m.done() == 0


# ------------------------------------------------------------------ admission

def test_a_cache_grows_only_under_the_ceiling_less_the_slack():
    fa = FakeAlloc(66)
    m = budget(fa)
    assert m.admit("park", int(1.5 * GIB))          # 67.5 <= 68
    assert not m.admit("park", int(2.5 * GIB))      # 68.5 > 68, nothing below to give back
    assert m.stats["admit_refused"] == 1


def test_a_more_valuable_cache_takes_the_room_of_the_cheaper_ones():
    fa = FakeAlloc(67)
    m = budget(fa)
    order = []
    m.register("store", shedder(fa, order, "store", 2))
    m.register("anchors", shedder(fa, order, "anchors", 0.3))
    m.register("pin", shedder(fa, order, "pin", 4))
    assert m.admit("park", int(2.5 * GIB))          # 69.5 > 68: the store and the free anchors go
    assert order == ["store", "anchors"], order
    order.clear()
    fa.alloc = int(67 * GIB)
    assert not m.admit("store", int(2 * GIB)), "the cheapest level takes nobody's room"
    assert order == []


def test_while_a_prefill_waits_its_workspace_counts():
    fa = FakeAlloc(60)
    m = budget(fa)
    m.gate(262_144)                                  # 60 + 7.33 <= 77: it runs
    assert m.ws_now > 7 * GIB
    assert not m.admit("pin", 10 * GIB)              # 70 + 7.33 > 77 before its forwards
    m.prefilled(262_144)
    assert m.ws_now == 0
    assert m.admit("pin", 4 * GIB)


# ------------------------------------------------------------------ SRV-773, SRV-775: the rows a prefill forwards

def test_srv773_a_continuation_is_charged_its_rows_not_its_context():
    """g8's shape (2026-11-02): 21 new rows on a 62,886-token context were charged the whole context's workspace and
    refused ten times. With the adapter's model (`ws_rows`) the gate charges the rows the prefill forwards; a cold
    prompt (no rows said) is charged as before, and a refused continuation says what it was."""
    fa = FakeAlloc(16)
    m = budget(fa, ceiling=18, peak=20, slack=1, ws_rows=lambda n, r: 1 * GIB if r < n else 5 * GIB)
    g = m.gate(62_886, rows=21)
    assert g["workspace"] == GIB and g["rows"] == 21 and m.ws_now == GIB, g
    try:
        m.gate(62_886)
        raise AssertionError("the whole context's workspace is over the peak")
    except MemoryRefused as exc:
        assert exc.rows is None and exc.need == 21 * GIB and "a 62886-token prefill" in str(exc), exc
    assert m.ws_now == 0, "a refused gate leaves no workspace in flight"
    m.peak = int(16.5 * GIB)
    try:
        m.gate(62_886, rows=21)
        raise AssertionError("16 + 1 is over a 16.5 GiB peak")
    except MemoryRefused as exc:
        assert exc.rows == 21 and "a 21-row continuation on a 62886-token context needs 17.0 GiB" in str(exc), exc
    plain = budget(FakeAlloc(16), ceiling=18, peak=20, slack=1)
    assert plain.workspace_rows(62_886, 21) == plain.workspace(62_886), "no model: the line, whatever the rows"
    return "21 rows: 1 GiB admitted; the context: 21 GiB refused; at 16.5 the message names the 21 rows"


#: SRV-773, measured 2026-11-02 on Flash-Next 3.05 (one DGX Spark, raw requests alone): a resumed
#: prefill's (served rows n, forwarded rows, the larger of its prefill's peak and its request end's peak, GiB)
FN_RESUMED = [(58_020, 20, 0.10), (8_319, 20, 0.41), (8_575, 256, 0.22), (9_598, 1_023, 0.60), (13_693, 4_095, 1.59),
              (26_019, 20, 0.10), (26_275, 256, 0.22), (27_299, 1_024, 0.69), (31_395, 4_096, 1.62),
              (58_020, 20, 0.10), (58_276, 256, 0.25), (59_300, 1_024, 0.98), (63_395, 4_095, 2.02),
              (100_019, 20, 0.10), (100_274, 255, 0.40), (101_298, 1_024, 1.55), (105_394, 4_096, 4.48),
              (125_020, 20, 0.10), (125_276, 256, 0.40), (126_300, 1_024, 1.55), (130_396, 4_096, 5.10),
              (133_019, 20, 0.10), (133_275, 256, 0.70), (134_298, 1_023, 2.68), (138_393, 4_095, 8.36),
              (142_489, 4_096, 8.27), (146_202, 3_713, 8.63), (146_222, 20, 0.11)]
#: and the cold or seeded ones: (n, rows, slot, the allocated delta or the request end's when larger, the policy's
#: workspace plus the rows' streams, GiB); slot 0's delta holds its anchors, admitted apart (`Budget.admit`)
FN_COLD = [(58_000, 58_000, 0, 6.71, 4.85), (8_299, 4_151, 1, 2.66, 2.28), (25_999, 17_630, 2, 3.05, 2.71),
           (58_000, 49_641, 3, 5.13, 4.80), (99_999, 91_611, 4, 8.16, 7.81), (125_000, 116_637, 5, 8.63, 8.27),
           (132_999, 124_630, 6, 8.78, 8.60)]


def test_srv773_the_flashnext_model_covers_the_measured_rows():
    """Flash-Next's workspace by forwarded rows (`FlashNextServing._workspace_rows`, the adapter's
    `CostModel.workspace_rows`) on every measured row: never under what the row took, 21 rows at g8's 62,886-token
    context under 1 GiB (the line charged 4.47), g2's 3,714 rows at 135,083 at most the cap, and a cold prompt never
    under the line, whatever its length."""
    import types
    from engine.models.flashnext import serving as FS
    from test_fn_serving import _qsa_class
    stub = types.SimpleNamespace(fn=types.SimpleNamespace(qsa_class=_qsa_class))

    def charge(n, rows):
        return FS.FlashNextServing._workspace_rows(stub, n, rows) / GIB
    worst = min(charge(n, r) - took for n, r, took in FN_RESUMED)
    assert worst >= 0.38, [(n, r, took, round(charge(n, r), 2)) for n, r, took in FN_RESUMED]
    for n, r, slot, delta, ws_stack in FN_COLD:
        assert charge(n, r) >= ws_stack + 0.45, (n, r, charge(n, r))
        assert slot == 0 or charge(n, r) >= delta, (n, r, charge(n, r), delta)
    assert charge(62_886, 21) <= 1.0 and charge(135_083, 3_714) <= 10.75
    line = budget(FakeAlloc(50), peak=90, ws_floor=FS.WS_LINE_FLOOR, ws_per_token=FS.WS_LINE_ROW, ws_cap=FS.WS_CAP)
    for n in (2_000, 30_000, 85_549, 262_144):
        assert charge(n, n) >= line.workspace(n) / GIB, (n, charge(n, n), line.workspace(n) / GIB)
    # the served cost carries it: the policy charges a continuation its rows, a cold prompt at least its line
    model = lambda n, r: FS.FlashNextServing._workspace_rows(stub, n, r)     # noqa: E731
    m = budget(FakeAlloc(50), peak=90, ws_floor=FS.WS_LINE_FLOOR, ws_per_token=FS.WS_LINE_ROW, ws_cap=FS.WS_CAP,
               ws_rows=model)
    assert m.workspace_rows(62_886, 21) == model(62_886, 21) and m.workspace_rows(2_000) == m.workspace(2_000)
    return (f"28 resumed rows covered (least +{worst:.2f} GiB), 7 cold; g8 {charge(62_886, 21):.2f} GiB, "
            f"g2 {charge(135_083, 3_714):.2f}, 262k cold {charge(262_144, 262_144):.2f}")


def test_srv775_the_admission_counts_a_cold_prompt_by_its_rows():
    """The admission's count of a prefill in flight (`workspace_rows(line=False)`): by the rows alone, a cold prompt
    without the line's 2.75 GiB floor, which the gate keeps. Lone cold prompts of 205-215 rows peaked at 0.20-0.67 GiB
    on 2026-11-02 (`[req] peak_gib`); counted at 2.75 each, eight of them kept the eighth waiting 12 s for the box's
    floor. A model without the keyword, and a family without a model, count as the gate charges."""
    import types
    from engine.models.flashnext import serving as FS
    from test_fn_serving import _qsa_class
    stub = types.SimpleNamespace(fn=types.SimpleNamespace(qsa_class=_qsa_class))
    model = lambda n, r, line=True: FS.FlashNextServing._workspace_rows(stub, n, r, line=line)     # noqa: E731
    kw = dict(ws_floor=FS.WS_LINE_FLOOR, ws_per_token=FS.WS_LINE_ROW, ws_cap=FS.WS_CAP)
    m = budget(FakeAlloc(50), peak=90, ws_rows=model, **kw)
    for n, took in ((205, 0.67), (215, 0.33), (1_640, None), (4_096, None)):
        bare = m.workspace_rows(n, n, line=False) / GIB
        assert m.workspace_rows(n, n) == FS.WS_LINE_FLOOR, "the gate: the line's floor"
        assert bare < FS.WS_LINE_FLOOR / GIB and (took is None or bare >= took + 0.1), (n, bare)
    assert m.workspace_rows(5_767, 5_767, line=False) < m.workspace_rows(5_767, 5_767)
    for n, r in ((62_886, 21), (135_083, 3_714), (102_614, 102_614)):
        assert m.workspace_rows(n, r, line=False) <= m.workspace_rows(n, r), (n, r)
    assert m.workspace_rows(62_886, 21, line=False) == m.workspace_rows(62_886, 21), "a continuation: no line"
    two = budget(FakeAlloc(50), peak=90, ws_rows=lambda n, r: GIB, **kw)
    assert two.workspace_rows(205, 205, line=False) == two.workspace_rows(205, 205) == FS.WS_LINE_FLOOR
    plain = budget(FakeAlloc(50), peak=90, **kw)
    assert plain.workspace_rows(205, 205, line=False) == plain.workspace(205)
    return (f"205 rows cold: gate {m.workspace_rows(205, 205) / GIB:.2f} GiB, admission "
            f"{m.workspace_rows(205, 205, line=False) / GIB:.2f}; 1,640: {m.workspace_rows(1_640, 1_640, line=False) / GIB:.2f}")


def test_srv775_the_strict_gate_counts_the_prefills_in_flight():
    """QSE_PEAK_STRICT's gate adds the other prefills in flight to the reserve; the plain gate does not. `ws_now` is
    the largest admitted prefill's workspace, kept by thread: one prefill's end leaves the others' (it once held one
    figure, which slot 0's request end zeroed under a side slot's prefill). The largest, not the sum: summed, eight
    cold prompts at the gate's 2.75 GiB floor made `admit` refuse the stash during a burst (2026-11-02)."""
    import threading
    fa = FakeAlloc(62)
    m = budget(fa, ceiling=66, peak=70, ws_rows=lambda n, r: GIB)
    assert m.gate(10_000, rows=100)["workspace"] == GIB                 # 62 + 1 <= 70
    try:
        m.gate(10_000, rows=100, others=8 * GIB)                        # 62 + 1 + 8 > 70
        raise AssertionError("the strict gate counts the others")
    except MemoryRefused as exc:
        assert exc.need == 71 * GIB, exc
    m.prefill_done()
    steps = {k: (threading.Event(), threading.Event()) for k in ("a", "b")}
    ws = {"a": 1 * GIB, "b": 2 * GIB}
    two = budget(FakeAlloc(50), ceiling=66, peak=70, ws_rows=lambda n, r: ws["a"] if r == 10 else ws["b"])

    def run(k):
        two.gate(1_000, rows=10 if k == "a" else 20)
        steps[k][0].set()
        steps[k][1].wait(10)
        two.prefill_done()
    ths = {k: threading.Thread(target=run, args=(k,)) for k in steps}
    for k in ("a", "b"):
        ths[k].start()
        assert steps[k][0].wait(10)
    try:
        assert two.ws_now == 2 * GIB and two.report()["ws_now"] == 2 * GIB, two.ws_now     # the largest of 1 and 2
        two.done()                                   # another thread's request end: its own entry only
        assert two.ws_now == 2 * GIB
        steps["b"][1].set()
        ths["b"].join(10)
        assert two.ws_now == 1 * GIB, "b's prefill has run: a's workspace stays"
    finally:
        for k in steps:
            steps[k][1].set()
        for th in ths.values():
            th.join(10)
    assert two.ws_now == 0
    return "strict: 62 + 1 + 8 > 70 refused; two threads 1 and 2 GiB: 2 (the largest), b done: 1 left"


def test_srv775_a_side_slots_gate_runs_the_peak_stage_only():
    """A side slot's gate (`ceiling=False`) skips the ceiling stage: over the ceiling and under the peak it sheds
    nothing and empties nothing (no allocator walk either), where slot 0's gate gives the reserve back and sheds the
    store as before. Over the peak both shed for the workspace (2026-11-02: every side gate ran the ceiling stage,
    57 ms median against 0 before, and a c8 round's first tokens came 0.5 s later)."""
    walks = []
    for ceiling in (False, True):
        fa = FakeAlloc(69.5, 3)                      # reserved 72.5: over the 70 ceiling, under the 77 peak
        m = budget(fa)
        m.pools_probe = lambda: walks.append(ceiling) or {"graphs_free": 0}
        order = []
        m.register("store", shedder(fa, order, "store", 2))
        g = m.gate(1_000, ceiling=ceiling)
        if not ceiling:
            assert order == [] and fa.empties == 0 and g["shed"] == [] and m.stats["empties"] == 0, (order, g)
            assert walks == [], "no pool walk under the peak"
        else:
            assert fa.empties == 1 and fa.res <= 70 * GIB and walks, (fa.empties, walks)
        m.prefill_done()
    fa = FakeAlloc(76)                               # 76 allocated, no free reserve: + 1.15 over the 77 peak
    m = budget(fa)
    order = []
    m.register("store", shedder(fa, order, "store", 2))
    g = m.gate(1_000, ceiling=False)
    assert g["shed"] == ["store"] and order == ["store"], (g, order)
    return "side: 72.5 reserved, nothing shed, no empty, no walk; slot 0: the reserve given back; over the peak: store"


def _gate_in_thread(m: Budget, n: int, **kw) -> dict:
    """`m.gate(n, **kw)` on a thread of its own, which stays alive (its `ws_now` entry with it) until `out["go"]` is
    set: `out["done"]` is set once the gate has returned (`out["g"]`, `out["own"]`) or raised (`out["exc"]`)."""
    import threading
    out = {"done": threading.Event(), "go": threading.Event()}

    def run():
        try:
            out["g"] = m.gate(n, **kw)
            out["own"] = m.ws_own()
        except Exception as exc:                                   # noqa: BLE001
            out["exc"] = exc
        out["done"].set()
        out["go"].wait(10)
        m.prefill_done()
    out["th"] = threading.Thread(target=run, daemon=True)
    out["th"].start()
    return out


def test_srv775_a_side_gate_far_under_the_peak_admits_without_the_lock():
    """2026-11-02, hold 4: at a c8 round's start the eight side gates queued on the policy's lock behind slot 0's gate
    (side gate_ms 49.3 ms median, 88.5 max, the reserve under the ceiling all along). A side slot's gate whose reserve
    plus workspace stays `FAST_MARGIN` (0.5 GiB) under the peak now admits on one reading without the lock: it
    returns while another thread holds the lock, keeps its workspace in `ws_now` and counts `fast_gates`, with the
    decision the locked path takes there (no shed, no empty)."""
    fa = FakeAlloc(60, 2)                            # reserved 62 + 1 GiB of workspace: 14 under the 77 peak
    m = budget(fa, ws_rows=lambda n, r: GIB)
    reads = []
    probe = m._probe
    m._probe = lambda: reads.append(1) or probe()
    m.lock.acquire()                                 # slot 0's gate, or a request's `done`, holds it
    try:
        out = _gate_in_thread(m, 10_000, rows=100, ceiling=False)
        assert out["done"].wait(5), "the side gate waited for the lock"
        assert "exc" not in out, out
        g = out["g"]
        assert g["fast"] is True and g["workspace"] == GIB and g["shed"] == [] and g["rows"] == 100, g
        assert g["reserved"] == 62 * GIB and g["allocated"] == 60 * GIB, g
        assert len(reads) == 1, f"one reading of the reserve, not {len(reads)}"
        assert out["own"] == GIB and m.ws_now == GIB and m.report()["ws_now"] == GIB, (out["own"], m.ws_now)
        assert m.stats["gates"] == 1 and m.stats["fast_gates"] == 1 and m.stats["last_gate"] is g, m.stats
        assert fa.empties == 0 and m.stats["empties"] == 0 and m.refused_in_a_row == 0
    finally:
        m.lock.release()
        out["go"].set()
        out["th"].join(10)
    assert m.ws_now == 0, "its prefill has run: the workspace leaves `ws_now`"
    ref = budget(FakeAlloc(60, 2), ws_rows=lambda n, r: GIB)
    ref.strict = True                                # the locked path on the same reserve
    want = ref.gate(10_000, rows=100, ceiling=False)
    ref.prefill_done()
    same = ("n", "workspace", "allocated", "reserved", "shed", "rows")
    assert {k: g[k] for k in same} == {k: want[k] for k in same} and "fast" not in want, (g, want)
    assert ref.stats["fast_gates"] == 0 and ref.stats["gates"] == 1
    return "62 reserved + 1 GiB: admitted with the lock held elsewhere, 1 reading, ws_now 1 GiB, fast_gates 1"


def test_srv775_a_side_gate_near_or_over_the_peak_takes_the_lock_and_decides_as_before():
    """Within `FAST_MARGIN` of the peak, or over it, a side slot's gate takes the lock as before: it waits while
    another thread holds it, then admits under the peak (76.2 + 0.5 <= 77), sheds the store over it, or refuses
    with nothing left to shed. None of them is a fast gate."""
    margin = Budget.FAST_MARGIN / GIB
    assert margin == 0.5
    fa = FakeAlloc(75.5, 0.7)                        # reserved 76.2 + 0.5 GiB = 76.7: under the peak, in the margin
    m = budget(fa, ws_rows=lambda n, r: GIB // 2)
    m.lock.acquire()
    try:
        out = _gate_in_thread(m, 10_000, rows=100, ceiling=False)
        assert not out["done"].wait(0.3), "within the margin the side gate takes the lock"
    finally:
        m.lock.release()
    try:
        assert out["done"].wait(5) and "exc" not in out, out
        g = out["g"]
        assert "fast" not in g and g["shed"] == [] and g["workspace"] == GIB // 2, g
        assert out["own"] == GIB // 2 and m.stats["fast_gates"] == 0 and m.stats["gates"] == 1, m.stats
    finally:
        out["go"].set()
        out["th"].join(10)
    fa = FakeAlloc(76)                               # 76 + 1.15 over the 77 peak: the store goes, as before
    m = budget(fa)
    order = []
    m.register("store", shedder(fa, order, "store", 2))
    g = m.gate(1_000, ceiling=False)
    assert g["shed"] == ["store"] and order == ["store"] and "fast" not in g, (g, order)
    assert m.stats["fast_gates"] == 0
    m.prefill_done()
    m = budget(FakeAlloc(76))                        # nothing to shed: refused, as before
    try:
        m.gate(1_000, ceiling=False)
        raise AssertionError("over the peak with nothing to shed")
    except MemoryRefused as exc:
        assert exc.peak == 77 * GIB, exc
    assert m.stats["fast_gates"] == 0 and m.stats["refused"] == 1 and m.ws_now == 0
    return f"76.7 of 77 (margin {margin} GiB): waited for the lock, admitted; over the peak: store shed, then refused"


def test_srv775_slot0_and_the_strict_peak_keep_the_lock():
    """Slot 0's gate (`ceiling=True`), a gate that counts other prefills (`others`) and every gate under the strict
    peak (`Budget.strict`, QSE_PEAK_STRICT) take the lock as before, however far under the peak."""
    cases = (({"ceiling": True}, False), ({"ceiling": False, "others": GIB}, False), ({"ceiling": False}, True))
    for kw, strict in cases:
        m = budget(FakeAlloc(40), ws_rows=lambda n, r: GIB)
        m.strict = strict
        m.lock.acquire()
        try:
            out = _gate_in_thread(m, 10_000, rows=100, **kw)
            assert not out["done"].wait(0.3), (kw, strict)
        finally:
            m.lock.release()
        try:
            assert out["done"].wait(5) and "exc" not in out, out
            assert "fast" not in out["g"] and m.stats["fast_gates"] == 0 and m.stats["gates"] == 1, (kw, m.stats)
            assert out["own"] == GIB, (kw, out["own"])
        finally:
            out["go"].set()
            out["th"].join(10)
    return "slot 0, others > 0 and strict: each waited for the lock, fast_gates 0"


def test_srv773_a_continuation_teaches_the_line_nothing():
    """`prefilled` learns the line from a cold prefill of 64k rows or more; a continuation's few rows (or a re-read
    from a resume point) say nothing of a whole prompt's workspace, and `mark(rows=...)` keeps them out."""
    def flashnext_like():
        fa = FakeAlloc(50)
        return fa, budget(fa, peak=90, ws_floor=int(2.75 * GIB), ws_per_token=76309, ws_cap=int(10.75 * GIB))
    fa, m = flashnext_like()
    m.mark(rows=21)
    fa.take(9 * GIB)
    fa.give(9 * GIB)
    m.prefilled(100_000)
    assert m.ws_per_token == 76309 and m.ws_cap == int(10.75 * GIB) and m.stats["observed"] is None
    for rows in (None, 100_000):
        fa, m = flashnext_like()
        m.mark(rows=rows)
        fa.take(9 * GIB)
        fa.give(9 * GIB)
        m.prefilled(100_000)
        assert m.ws_per_token == int(9 * GIB / 100_000) and m.stats["observed"]["workspace"] == 9 * GIB, rows
    return "mark(rows=21): nothing learned; mark() and mark(rows=n): the line raised to 96,636 B a token"


# ------------------------------------------------------------------ 5: the reserve at idle

def test_the_reserve_over_its_slack_goes_back_after_the_request():
    fa = FakeAlloc(65, 7)
    m = budget(fa)
    freed = m.done()
    assert freed == 7 * GIB and fa.empties == 1 and m.stats["empties"] == 1
    fa.res += int(1.5 * GIB)                         # under the slack: kept
    assert m.done() == 0 and fa.empties == 1


def test_the_workspace_estimate_learns_upward_only():
    fa = FakeAlloc(60)
    m = budget(fa)
    was = m.ws_per_token
    m.mark()
    fa.take(int(6 * GIB))                            # 6 GiB at 100k is more than the line says
    fa.give(int(6 * GIB))
    m.prefilled(100_000)
    assert m.ws_per_token == int(6 * GIB / 100_000) > was
    assert m.stats["observed"]["workspace"] == 6 * GIB
    m.mark()
    fa.take(GIB)
    fa.give(GIB)
    m.prefilled(100_000)
    assert m.ws_per_token == int(6 * GIB / 100_000), "a cheaper prefill does not lower it"
    assert m.stats["prefill_peak"] == 66 * GIB and m.lifetime() >= 66 * GIB


def test_what_a_prefill_keeps_is_not_its_workspace():
    """The anchors and the store snapshot a prefill takes are still allocated at its end; only
    what it gave back is workspace (the 2026-10-01 gate learned 113 kB a token without this)."""
    fa = FakeAlloc(60)
    m = budget(fa)
    was = m.ws_per_token
    m.mark()
    fa.take(3 * GIB)                  # anchors and a snapshot, kept
    fa.take(2 * GIB)                  # the chunk's scores and activations
    fa.give(2 * GIB)
    m.prefilled(100_000)
    assert m.stats["observed"]["workspace"] == 2 * GIB and m.ws_per_token == was
    m.mark()
    fa.take(9 * GIB)
    fa.give(9 * GIB)
    m.prefilled(40_000)               # short: the fixed part dominates, nothing is learned
    assert m.ws_per_token == was and m.stats["observed"]["n"] == 100_000


# ------------------------------------------------------------------ 2: the store, room before clone

class Snap:
    def __init__(self, n, nbytes):
        self.length, self.nbytes = n, nbytes

    def parts(self):
        return {"kv": self.nbytes}


def test_the_store_makes_room_before_the_clone():
    st = StateStore(10, chunk=4, max_entry_bytes=6)
    toks = list(range(40))
    st.put(toks, Snap(4, 4), "a")
    st.put(toks, Snap(8, 4), "a")
    assert st.bytes == 8
    assert st.reserve(5)                             # 8 + 5 > 10: the oldest goes first
    assert st.bytes == 4 and st.stats["evictions"] == 1
    st.put(toks, Snap(12, 5), "b")
    assert st.bytes == 9 <= st.budget
    assert not st.reserve(7), "over the entry cap: refused, nothing evicted"
    assert st.bytes == 9


def test_a_compaction_drops_its_sessions_entries_only():
    st = StateStore(100, chunk=4)
    toks = list(range(40))
    st.put(toks, Snap(4, 3), "A")
    st.put(toks, Snap(8, 3), "B")
    st.put(toks, Snap(12, 3), "A")
    assert st.drop_session("A") == 6
    assert st.bytes == 3 and st.report()["entries"] == 1 and st.stats["dropped_session"] == 2
    assert st.drop_session(None) == 0 and st.drop_session("nobody") == 0


def test_the_store_sheds_and_takes_new_limits():
    st = StateStore(100, chunk=4)
    toks = list(range(40))
    for n in (4, 8, 12):
        st.put(toks, Snap(n, 10), "A")
    st.set_limits(budget_bytes=15, max_entry_bytes=9)
    assert st.bytes == 10 and st.max_entry == 9
    assert st.shed() == 10 and st.bytes == 0


# ------------------------------------------------------------------ 8: the planner counts the peak

def _residency_doc():
    from server.conductor import schema
    return schema.seed("residency")


def test_the_planner_refuses_a_12_gib_role_at_37_9_free_with_the_engine_at_70_of_77():
    from server.conductor import residency
    doc = _residency_doc()
    d = residency.check("reader", 37.9, doc, engine_peak_gib=77, engine_now_gib=70)
    assert not d["allowed"] and d["reason"] == "refused_memory"
    assert d["engine_growth_gib"] == 7 and d["after_gib"] == 18.9
    # the old rule (no peak) admitted it -- the bug
    old = residency.check("reader", 37.9, doc)
    assert old["allowed"] and old["after_gib"] == 25.9
    # the decider (3.0) still fits: 37.9 - 7 - 3 = 27.9 >= 21
    assert residency.check("decider", 37.9, doc, engine_peak_gib=77, engine_now_gib=70)["allowed"]
    # with no reading of the engine the whole peak counts
    assert residency.check("decider", 60, doc, engine_peak_gib=77)["engine_growth_gib"] == 77


def test_the_plan_takes_the_growth_once():
    from server.conductor import residency
    p = residency.plan(37.9, _residency_doc(), engine_peak_gib=77, engine_now_gib=70)
    assert p["engine_growth_gib"] == 7 and p["spendable_gib"] == 9.9
    rows = {r["role"]: r for r in p["rows"]}
    assert rows["decider"]["allowed"] and rows["decider"]["after_gib"] == 27.9
    assert not rows["reader"]["allowed"]


def test_the_command_line_takes_the_engine_numbers():
    with tempfile.TemporaryDirectory() as root:
        env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env.pop("MEM_PEAK_GIB", None)
        r = subprocess.run([sys.executable, "-m", "server.conductor.residency", "--root", root,
                            "check", "reader", "--mem-available-gib", "37.9",
                            "--engine-peak-gib", "77", "--engine-now-gib", "70"],
                           capture_output=True, text=True, env=env, cwd=env["PYTHONPATH"])
        assert r.returncode == 3, r.stdout + r.stderr
        d = json.loads(r.stdout)
        assert d["engine_growth_gib"] == 7 and d["reason"] == "refused_memory"
        env["MEM_PEAK_GIB"] = "77"                   # serve.env's, when /health does not answer
        r = subprocess.run([sys.executable, "-m", "server.conductor.residency", "--root", root,
                            "check", "decider", "--mem-available-gib", "100"],
                           capture_output=True, text=True, env=env, cwd=env["PYTHONPATH"])
        assert json.loads(r.stdout)["engine_growth_gib"] == 77


# ------------------------------------------------------------------ the limits table

class _Store:
    def __init__(self, v, doc):
        self.v, self.doc = v, doc

    def peek(self, kind):
        assert kind == "residency"
        return self.v, self.doc


class _Cond:
    def __init__(self, v, doc):
        self.store = _Store(v, doc)


def _args(**over):
    a = {"mem_ceiling_gib": 70.0, "mem_peak_gib": 77.0, "reserve_slack_gib": 2.0,
         "resident_gb": 2.5, "resident_stash_gb": 1.0, "resident_guest_max": 8192,
         "resident_pin_gb": 4.0, "resident_park_gb": 4.0, "cache_budget_gb": 2.0,
         "store_entry_gb": 1.0, "image_cache_mb": 512.0, "suffix_store_mb": 192.0,
         "prefix_chunk": 1024, "narrow_arm": "on", "max_len": 262144}
    a.update(over)
    return a


def test_the_view_has_every_key_and_max_len_is_locked():
    from server import memlimits
    st = {"args": _args(), "serve_flags": _args(), "max_len": 262144}
    v = memlimits.view(st)
    keys = [r["key"] for r in v["rows"]]
    assert keys == list(memlimits.KEYS)
    row = {r["key"]: r for r in v["rows"]}
    assert row["MAX_LEN"]["applies"] == "locked" and row["MAX_LEN"]["value"] == 262144
    assert row["PREFIX_CHUNK"]["applies"] == "restart" and not row["PREFIX_CHUNK"]["pending_restart"]
    assert row["RESIDENT_PARK_GB"]["applies"] == "next request"
    assert row["NARROW_ARM"]["value"] == 1 and row["CACHE_GB"]["source"] == "serve.env"


def test_a_models_own_grid_is_shown_locked_with_no_restart_pending():
    """rc-0.8.0 live: on Flash-Next the memory plan said PREFIX_CHUNK 1,024 "needs a restart; the
    engine runs 4,096 until it restarts", which no restart changes: the model's caches keep its own
    grid (`caps.anchor_grid`). The row shows that grid, locked, with nothing pending."""
    from server import memlimits
    st = {"args": _args(prefix_chunk=4096), "serve_flags": _args(), "max_len": 262144,
          "anchor_grid": 4096}
    row = {r["key"]: r for r in memlimits.view(st)["rows"]}["PREFIX_CHUNK"]
    assert row["value"] == row["running"] == 4096 and row["applies"] == "locked", row
    assert not row["pending_restart"] and "own grid: 4,096 tokens" in row["locked"], row
    # a dashboard override changes nothing for such a model either
    st["conductor"] = _Cond(5, {"engine": {"PREFIX_CHUNK": 512}})
    row = {r["key"]: r for r in memlimits.view(st)["rows"]}["PREFIX_CHUNK"]
    assert row["value"] == 4096 and not row["pending_restart"]


def test_a_dashboard_override_applies_live_or_says_it_needs_a_restart():
    from server import memlimits
    fa = FakeAlloc(60)
    mem = budget(fa)
    store = StateStore(2 * GIB, chunk=1024)
    doc = {"engine": {"RESIDENT_PARK_GB": 2, "CACHE_GB": 1, "MEM_CEILING_GIB": 68,
                      "PREFIX_CHUNK": 512, "MAX_LEN": 131072}}
    st = {"args": _args(), "serve_flags": _args(), "mem": mem, "state_store": store,
          "conductor": _Cond(3, doc), "max_len": 262144}
    done = memlimits.refresh(st)
    assert "MEM_CEILING_GIB=68" in done and "CACHE_GB=1" in done
    assert mem.ceiling == 68 * GIB and store.budget == GIB
    assert memlimits.refresh(st) == [], "the same version applies once"
    row = {r["key"]: r for r in memlimits.view(st)["rows"]}
    assert row["PREFIX_CHUNK"]["pending_restart"] and row["PREFIX_CHUNK"]["source"] == "dashboard"
    assert row["MAX_LEN"]["value"] == 262144, "the hard rule is not overridable"
    assert row["MEM_CEILING_GIB"]["source"] == "dashboard" and row["MEM_CEILING_GIB"]["value"] == 68
    # the key leaves the document: back to the flag
    st["conductor"] = _Cond(4, {"engine": {}})
    memlimits.refresh(st)
    assert mem.ceiling == 70 * GIB and store.budget == 2 * GIB


def test_overrides_at_start_never_stop_the_engine():
    import argparse
    from server import memlimits
    a = argparse.Namespace(**_args())
    lines = memlimits.overrides_at_start(
        a, {"engine": {"NARROW_ARM": 0, "PREFIX_CHUNK": 512, "MAX_LEN": 1024}}, len_mode="b")
    assert a.narrow_arm == "on" and any("skipped" in x for x in lines)
    assert a.prefix_chunk == 512 and a.max_len == 262144
    memlimits.overrides_at_start(a, {"engine": {"NARROW_ARM": 0}}, len_mode="wide")
    assert a.narrow_arm == "off"


def test_the_schema_takes_the_engine_section_and_checks_peak_over_ceiling():
    from server.conductor import schema
    doc = schema.seed("residency")
    assert schema.check("residency", dict(doc, engine={"RESIDENT_PARK_GB": 4})) == []
    errs = schema.check("residency", dict(doc, engine={"MEM_CEILING_GIB": 70, "MEM_PEAK_GIB": 60}))
    assert any("MEM_PEAK_GIB is under" in e for e in errs), errs
    assert schema.check("residency", dict(doc, engine={"MAX_LEN": 131072})), "MAX_LEN is refused"


# ------------------------------------------------------------------ Prometheus

def test_the_memory_metrics_render_from_the_cache_stats():
    from server import metrics as M
    rep = budget(FakeAlloc(60, 1)).report()
    rep["shed"]["park"] = 2
    rep["refused"] = 1
    stats = {"memory": rep, "resident": {"anchor_pool_bytes": 3, "stash_bytes": 0, "pin_bytes": 5,
                                         "park_bytes": 7}, "state_store": {"bytes": 11}}
    M.bind(cache_stats=lambda: stats)
    text = M.render()
    assert 'qse_memory_bytes{part="ceiling"} 75161927680' in text, text
    assert 'qse_memory_bytes{part="resident_park"} 7' in text
    assert 'qse_memory_shed_total{level="park"} 2' in text
    assert 'qse_requests_refused_total{reason="memory"} 1' in text


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()
