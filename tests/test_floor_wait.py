"""SRV-797, CPU only: no request ends because memory is short; it waits for memory and goes on by itself.

The two failures of 2026-11-08, on fake MemAvailable, and the review of the first fix:

  * a 49k-row chat was admitted alone at 9.8 GiB (6.9 >= the admission's floor 5.4: chunk, KV commit, margin, its own
    chunk charged as its workspace), and the KV pool's gate, whose hard floor counts the whole chunk while that prefill
    runs (8.4), refused its first page; it gave way, was admitted again, 16 times, then `memory_full`. The admission
    now also checks what the gate will see at the prefill's pages, in the order they commit (`check`'s "prefill":
    the prompt's pages under the pool's floor, the workspace only when some of them come after it); the rest of the
    request (its decode pages) is checked under the admission's floor as before, so nothing is counted twice: what
    it admits the gate takes, and 0.1 GiB less the gate refuses. A 142k turn with a 10.8 GiB workspace is admitted
    at 19.9 GiB, as on 2026-11-06. Short, the pool's reclaim levels run (any level in its list), then it waits;
  * a turn short only by the caches' room (what the state store and the resident MAY still take) is let in on it
    (`relieve`: the policy holds that room from the caches until its prefill has run); one short by memory a lane
    holds waits;
  * a lone streamed chat whose prefill the gate refused got `memory_full` after 0 tokens. It gives way, waits in the
    admission's line and goes on; its thinking budget starts as it would have (refused before it was armed) or goes
    on from where it was (after a token);
  * a streamed lane gives way past `KV_RECOMPUTE_MAX` (its waits are bounded by `MEM_WAIT_S`, then the wait's own
    refusal, `memory_wait_timeout`), backs off longer at each give-way with little written between them, out of the
    line (requests that fit go meanwhile); the line is by who gave way first; a re-admission is priced as the first;
    the request's deadline moves on by its waits; a planned stop during the wait is the restart's error;
  * one not streamed keeps SRV-709's rule (alone, or after 16 give-ways, `memory_full`; a wait at most
    `--queue-timeout`), and a wait that reaches its bound names the cause and says how to wait longer
    (`wait_for_memory`).

Run: python tests/test_floor_wait.py
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from server import slots  # noqa: E402

GiB = slots.GiB
MiB = slots.MiB

# the live numbers of 2026-11-08 (chatcmpl-ea7d..., slot 5): a 4,096-row chunk's workspace at the class of 49k rows,
# one KV commit, the margin; the request's own workspace (1,521 rows forwarded) and its free blocks; its KV pages (the
# prompt's 49,174 rows and the notes' room, 57,430 rows at 31 KB, and 8,192 rows of answer)
STEP, KV_STEP, MARGIN = int(3.02 * GiB), int(0.4 * GiB), 5 * GiB
WS, SPARE, KV, KV_PROMPT = int(1.9 * GiB), int(0.9 * GiB), int(1.9 * GiB), int(1.66 * GiB)
ROWS = 49_174


def _guard(rows: int, cap: int, step: int = STEP):
    """server/app.py `_guard` on the live state: the chunk at the class of `rows`, a KV commit and the margin."""
    return slots.guard(cap, step=step if rows > 0 else 0, kv=KV_STEP, sandbox=0, margin=MARGIN)


def _box(avail_gib: float, room=lambda: 0, spare=lambda: SPARE, step=STEP):
    box = {"avail": int(avail_gib * GiB)}
    adm = slots.Admission(15 * GiB, meminfo=lambda: {"MemAvailable": box["avail"]}, growth=lambda: 0,
                          room=room, spare=spare, guard=lambda rows: _guard(rows, 15 * GiB, step))
    pool = slots.GuardedPool(hard_floor=10 * GiB, margin_rows=0, meminfo=lambda: box["avail"])
    # as server/app.py wires them (`_build_side_slots`)
    pool.guard = lambda: (_guard(adm.ctx_rows(), pool.hard_floor_cfg, step) or (None,))[0]
    pool.guard_at = lambda rows: (_guard(max(int(rows), adm.ctx_rows()), pool.hard_floor_cfg, step) or (None,))[0]
    adm.pool_floor = pool.floor_at
    return box, adm, pool


def _serve(box, adm, pool, key, ws_new, kv, kv_prompt, per_chunk, owner="slot5", chunks=10):
    """The request's pages through the gate, in the order the engine commits them, 4 MiB a commit. Flash-Next
    (`per_chunk`): each chunk's pages before that chunk, its workspace, then the allocator's blocks back to the driver
    (engine/models/flashnext/engine.py `prefill` empties after every chunk once the state holds more than 4,096
    rows, as a restored 47k-row conversation does from its first chunk). The 27B: the prompt's pages before the
    prefill, then its workspace, which stays in the allocator's reserve. Then its prefill has run (`prefilled`: the
    floor drops to the decode's) and its answer's pages commit. Raises `PoolFull` as the gate does."""
    def commit(n):
        for _ in range(int(n) // (4 * MiB)):
            pool.gate(owner, 4 * MiB)
            box["avail"] -= 4 * MiB
    if per_chunk:
        for c in range(chunks):
            commit(kv_prompt // chunks)
            box["avail"] -= ws_new
            box["avail"] += ws_new              # `torch.cuda.empty_cache()` after the chunk
    else:
        commit(kv_prompt)
        box["avail"] -= ws_new
    adm.prefilled(key)
    commit(kv - kv_prompt)


def _admit_49k(adm, key="ea7d"):
    return adm.reserve(key, WS + KV, kv=KV, owner="slot5", over_peak=True, rows=ROWS, kv_prompt=KV_PROMPT)


def test_the_pools_floor_is_the_guard_at_the_prefills_class():
    box, adm, pool = _box(20.0)
    assert pool.floor_at(0) == pool.hard_floor == KV_STEP + MARGIN, "no prefill in flight: no chunk"
    assert pool.floor_at(ROWS) == STEP + KV_STEP + MARGIN
    assert adm.reserve("a", WS + KV, kv=KV, owner="slot5", over_peak=True, rows=ROWS)[0]
    assert pool.hard_floor == pool.floor_at(0) == pool.floor_at(10) == STEP + KV_STEP + MARGIN, \
        "while the prefill runs the gate holds its chunk"
    pool.guard_at = lambda rows: 1 / 0
    assert pool.floor_at(ROWS) == 10 * GiB, "no answer: the configured floor"
    pool.guard_at = None
    assert pool.floor_at(ROWS) == pool.hard_floor
    return f"{(KV_STEP + MARGIN) / GiB:.2f} GiB idle, {(STEP + KV_STEP + MARGIN) / GiB:.2f} while a 49k prefill runs"


def test_failure_1_without_the_pools_check_the_admission_admits_what_the_gate_refuses():
    """The incident, as it was: admitted alone at 6.9 >= 5.4, and the gate refuses its prompt's pages at 8.42."""
    from engine import kvpool
    box, adm, pool = _box(9.8)
    adm.pool_floor = None
    ok, d = _admit_49k(adm)
    assert ok and d["alone"] and abs(d["after"] / GiB - 6.9) < 0.01 and abs(d["floor"] / GiB - 5.4) < 0.01, d
    try:
        _serve(box, adm, pool, "ea7d", WS - SPARE, KV, KV_PROMPT, False)
        raise AssertionError("PoolFull expected")
    except kvpool.PoolFull:
        pass
    return "6.9 >= 5.4 admitted; the gate's 8.42 refused its prompt's pages"


def test_failure_1_the_admission_checks_the_prefills_pages_as_the_gate_will_and_says_so():
    box, adm, pool = _box(9.8)
    ok, d = _admit_49k(adm)
    assert not ok and d["alone"] and "ea7d" not in adm.held, d
    p = d["prefill"]
    assert p["floor"] == STEP + KV_STEP + MARGIN and not p["ok"] and p["kv"] == KV_PROMPT, p
    assert abs(d["floor"] / GiB - 5.4) < 0.01 and abs(d["after"] / GiB - 6.9) < 0.01, "the admission's own check"
    after, floor = slots.Admission.binding(d)
    assert (after, floor) == (p["after"], p["floor"]) and abs((floor - after) / GiB - 0.28) < 0.01
    words = slots.Admission.describe(d)
    assert words.startswith("alone: MemAvailable 9.8 GiB") and "= 6.9 GiB (>= floor 5.4" in words, words
    assert ("; its prefill's pages: MemAvailable 9.8 - the other workspaces - KV pages 1.7 = 8.1 GiB (< the KV "
            "pool's floor 8.4 while its prefill runs)") in words, words
    ok0, d0 = adm.check(WS, kv=0, over_peak=True, rows=ROWS)
    assert ok0 and "prefill" not in d0, "no KV pages (a static KV, the fake engine): nothing for the gate"
    return words


def test_what_the_admission_admits_the_gate_takes_and_no_more():
    """The invariant, over MemAvailable 8 to 16 GiB, for both commit orders (Flash-Next's chunks, the 27B's prompt
    first): every request admitted passes the gate with every page, and at 0.1 GiB under the first one admitted the
    gate refuses (nothing counted twice)."""
    from engine import kvpool
    out = []
    for per_chunk in (True, False):
        seen = []
        for tenth in range(80, 161):
            box, adm, pool = _box(tenth / 10)
            if not _admit_49k(adm)[0]:
                continue
            seen.append(tenth / 10)
            _serve(box, adm, pool, "ea7d", WS - SPARE, KV, KV_PROMPT, per_chunk)
        assert seen and seen[0] == 10.1 and seen[-1] == 16.0, (per_chunk, seen)
        box, adm, pool = _box(10.0)
        adm.pool_floor = None
        assert _admit_49k(adm)[0], "the admission's own check alone lets it in"
        try:
            _serve(box, adm, pool, "ea7d", WS - SPARE, KV, KV_PROMPT, per_chunk)
            raise AssertionError("PoolFull expected")
        except kvpool.PoolFull:
            pass
        out.append(f"{'chunk by chunk' if per_chunk else 'prompt first'}: from 10.1 GiB")
    return "; ".join(out) + "; 0.1 under, the gate refuses"


def test_a_142k_turn_with_a_10_8_gib_workspace_is_admitted_at_19_9_gib():
    """2026-11-06 (SRV-785): a 142k turn resumed in place, its 10.8 GiB workspace counted as the floor's chunk too,
    waited at MemAvailable 19.9 GiB; admitted once the chunk counted once. Its prompt's pages commit before that
    workspace (one chunk), under the pool's floor (10 GiB, its configured value); its workspace and its answer's pages
    under the admission's 5.4. The first fix (the pool's floor over the whole request: 19.9 - 10.8 - 0.5 = 8.6 < 10)
    held it again. The gate takes every page of it."""
    ws, kv, kvp = int(10.8 * GiB), int(0.5 * GiB), int(0.25 * GiB)
    box, adm, pool = _box(19.9, room=lambda: 30 * GiB, step=ws)
    ok, d = adm.reserve("t142", ws + kv, kv=kv, owner="slot1", rows=142_000, kv_prompt=kvp)
    assert ok and abs(d["after"] / GiB - 8.6) < 0.01 and abs(d["floor"] / GiB - 5.4) < 0.01, d
    assert d["prefill"]["ok"] and d["prefill"]["floor"] == 10 * GiB and abs(d["prefill"]["after"] / GiB - 19.65) < 0.01
    _serve(box, adm, pool, "t142", ws, kv, kvp, False, owner="slot1")
    assert box["avail"] >= pool.hard_floor
    return f"8.6 >= 5.4, its prompt's pages 19.65 >= 10: admitted, {box['avail'] / GiB:.2f} GiB left after it"


def test_a_cold_142k_turn_is_admitted_where_main_admitted_it():
    """A cold 142k turn on Flash-Next (every row forwarded, 35 chunks): main admitted it at MemAvailable = workspace +
    pages + the admission's 5.4 (25.9 GiB). Its prompt's pages commit chunk by chunk, each chunk's workspace given
    back after it, so the gate never sees the workspace beside them: admitted at 25.9, held at 25.8 by the admission's
    own check (as on main), and the gate takes every page. (68ec1e04 counted the workspace on top of the pool's floor
    here too: 28.4 GiB.)"""
    ws, kv, kvp = int(10.8 * GiB), int(9.7 * GiB), int(9.45 * GiB)
    out = []
    for avail in (25.85, 25.95):
        box, adm, pool = _box(avail, room=lambda: 30 * GiB, step=ws)
        ok, d = adm.reserve("cold", ws + kv, kv=kv, owner="slot2", rows=142_000, kv_prompt=kvp)
        assert d["prefill"]["ok"], d["prefill"]
        out.append(ok)
        if ok:
            _serve(box, adm, pool, "cold", ws, kv, kvp, True, owner="slot2", chunks=35)
            assert box["avail"] >= pool.hard_floor
    assert out == [False, True], out
    return "held at 25.85 (- 10.8 - 9.7 < 5.4, main's check), admitted at 25.95; every page passed the gate"


def test_a_turn_short_only_by_the_caches_room_is_let_in_and_one_short_by_a_lanes_kv_waits():
    """(13) A turn short by the caches' growth room alone (what the store and the resident MAY take, nothing they
    hold) is admitted at once, the room held from the caches until its prefill has run; one short by another lane's
    KV still to commit waits, and no room stays held for it."""
    room = {"g": 3 * GiB, "held": {}}

    def relieve(key, cut):
        got = min(int(cut), room["g"])
        room["held"][key] = got
        return got

    def unrelieve(key):
        room["held"].pop(key, None)
    adm = slots.Admission(10 * GiB, meminfo=lambda: {"MemAvailable": int(13.0 * GiB)},
                          growth=lambda: room["g"] - sum(room["held"].values()), relieve=relieve, unrelieve=unrelieve)
    # 13 - 3 growth - 1 workspace = 9 < 10: short 1 GiB, all of it growth
    ok, d = adm.reserve("turn", GiB)
    assert ok and d["relieved"] == GiB and room["held"] == {"turn": GiB} and abs(d["after"] / GiB - 10.0) < 0.01, d
    assert "let in on 1.0 GiB of the caches' room (held from them until it ends)" in slots.Admission.describe(d)
    adm.prefilled("turn")
    assert room["held"] == {"turn": GiB}, "the whole request needed the room: its answer's pages too"
    adm.release("turn")
    assert room["held"] == {}
    # short in its prefill's pages only (the pool's floor), not as a whole: the room is the caches' again once prefilled
    adm.pool_floor = lambda rows: 10 * GiB
    adm.cap = 6 * GiB
    ok, d = adm.reserve("pre", GiB + GiB // 2, kv=GiB // 2, kv_prompt=GiB // 2)
    assert ok and d["relieved"] == GiB // 2 and not d["relieved_life"] and d["prefill"]["ok"], d
    adm.prefilled("pre")
    assert room["held"] == {}, "its prefill has run: the room is the caches' again"
    adm.release("pre")
    adm.pool_floor, adm.cap = None, 10 * GiB
    # a lane in flight with 6 GiB of KV still to commit: 13 - 3 - 1 - 6 = 3, and 6 with all the room: still short
    adm.held["lane"], adm.kv["lane"] = 6 * GiB, (6 * GiB, None, 0)
    ok, d = adm.reserve("next", GiB)
    assert not ok and "relieved" not in d and room["held"] == {}, (d, room)
    return "short by the room: admitted at once; short by a lane's KV: waits, nothing held"


def test_the_policy_keeps_the_held_room_from_the_caches():
    """`_par_relieve`: the growth the admission counts falls by what it asked, the policy's `admit` keeps a cache out
    of the held room, and the room is the caches' again after `_par_unrelieve`."""
    from engine import cache
    app = _app()
    mem = cache.Budget(70 * GiB, 77 * GiB, 2 * GiB, probe=lambda: (40 * GiB, 50 * GiB), empty=lambda: 0,
                       peak_probe=lambda **kw: 0)
    store = types.SimpleNamespace(budget=20 * GiB, bytes=4 * GiB)
    saved = dict(app.STATE)
    try:
        app.STATE.clear()
        app.STATE.update(mem=mem, state_store=store)
        before = app._par_growth()
        assert before == 16 * GiB, before                 # min(20 - 4 by the budget, 70 - 2 - 40 = 28)
        assert app._par_relieve("k", 3 * GiB) == 3 * GiB and app._par_growth() == 13 * GiB
        assert mem.growth_held == 15 * GiB, "room 28 over budgets 16: the hold takes the gap and the cut"
        assert mem.admit("store", 13 * GiB) and not mem.admit("store", 13 * GiB + MiB)
        app._par_unrelieve("k")
        assert mem.growth_held == 0 and app._par_growth() == before and mem.admit("store", 20 * GiB)
        assert app._par_relieve("k", 50 * GiB) == 16 * GiB, "at most the growth it counts"
        app._par_unrelieve("k")
    finally:
        app.STATE.clear()
        app.STATE.update(saved)
    return "growth 16 -> 13 GiB, a cache admitted up to 13 GiB only, all back after"


def test_a_lane_backing_off_holds_no_one_back_and_the_line_is_by_who_gave_way_first():
    adm = slots.Admission(10 * GiB, meminfo=lambda: None)
    adm.first("old", at=100.0)
    adm.first("young", at=200.0)
    adm.wait("new")
    assert [adm.wait(k) for k in ("old", "young", "new")] == [1, 2, 3], "the one that gave way first, first"
    adm.first("old", at=100.0)                    # it gave way again: its place stays
    assert adm.wait("old") == 1 and adm.outranked(2)
    now = time.monotonic()
    adm.hold("old", now + 30)
    adm.hold("young", now + 30)
    assert adm.wait("new") == 1 and not adm.outranked(1) and adm.ahead_higher("new") == 0, \
        "while they back off, requests that fit go"
    adm.hold("old", now - 1)
    assert adm.wait("new") == 2 and adm.wait("old") == 1
    adm.release("old")
    assert "old" not in adm.holds
    return "oldest give-way first; a backing-off lane holds no one back"


def test_the_backoff_doubles_and_never_overflows():
    app = _app()
    lo, hi = app.KV_GIVE_WAY_BACKOFF_S
    assert [app._give_way_hold(k) for k in (1, 2, 3, 7, 8, 5000)] == [lo, 2 * lo, 4 * lo, hi, hi, hi]
    assert app._give_way_hold(9, long=False) == lo
    return f"{lo:g}, {2 * lo:g}, {4 * lo:g} ... {hi:g} s"


def test_a_wait_for_memory_moves_the_requests_deadline_on():
    import torch
    app = _app()
    box, adm, pool = _wired(10.5)
    d0 = app.Deadline(60.0)
    end0 = d0.t_end
    with _served(app, pool):
        threading.Timer(1.2, lambda: box.__setitem__("avail", 14 * GiB)).start()
        pr = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 1)
        with contextlib.redirect_stdout(io.StringIO()):
            assert app.Handler._par_admit(_H(app), pr, "late", torch.zeros(64, dtype=torch.long), stream=True,
                                          deadline=d0)
    assert 1.0 < d0.t_end - end0 < 3.5, d0.t_end - end0
    return f"moved on by {d0.t_end - end0:.1f} s"


def test_the_opt_in_to_wait_long_without_a_stream():
    import torch
    app = _app()
    assert app._wait_for_memory({"wait_for_memory": True}, {}) and app._wait_for_memory(
        {"tandem": {"wait_for_memory": True}}, {}) and app._wait_for_memory({}, {"X-Tandem-Wait-For-Memory": "1"})
    assert not app._wait_for_memory({}, {}) and not app._wait_for_memory({"wait_for_memory": "yes"}, None)
    box, adm, pool = _wired(10.5)
    with _served(app, pool):
        app.STATE["queue_timeout"] = 0.5
        threading.Timer(1.5, lambda: box.__setitem__("avail", 14 * GiB)).start()
        pr = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 1)
        with contextlib.redirect_stdout(io.StringIO()):
            assert app.Handler._par_admit(_H(app), pr, "agent", torch.zeros(64, dtype=torch.long), stream=False,
                                          wait_long=True)
    return "body, tandem object or header; waited past --queue-timeout"


# ------------------------------------------------------------------ the handler
def _app():
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_forward_tree  # noqa: F401  (the CPU environment before the engine imports)
    from server import app
    return app


#: a row of the fake paged KV: a 512-token prompt and the notes' room (8,768 rows) take about `KV`
PER_ROW = KV // 8_768


class _Pages:
    """A slot's paged KV as `_par_admit` and `_kv_give_back_own` read it, by rows; with a pool and a box, `ensure`
    commits through the pool's gate and takes the bytes from the box, `trim(0)` gives them back."""

    def __init__(self, per_row=PER_ROW, pool=None, box=None):
        self.per_row, self.pool, self.box = int(per_row), pool, box
        self.owner, self.bufs, self.trims, self.rows, self.commits = "slot0", [], 0, 0, 0

    def need_bytes(self, rows):
        return max(0, int(rows) - self.rows) * self.per_row

    def ensure(self, rows):
        n = self.need_bytes(rows)
        if n and self.pool is not None:
            self.pool.gate(self.owner, n)
        if n and self.box is not None:
            self.box["avail"] -= n
        self.rows = max(self.rows, int(rows))
        self.commits += 1 if n else 0

    def margin(self):
        return 0

    def trim(self, rows=0):
        self.trims += 1
        if self.box is not None:
            self.box["avail"] += self.rows * self.per_row
        self.rows = 0
        return 0


class _H:
    """The handler's surface `_par_admit` and `_recomputing` use."""

    def __init__(self, app):
        self.sent = None
        self.wfile = io.BytesIO()
        self._mem_refusal = None
        self._par_admit = types.MethodType(app.Handler._par_admit, self)

    def _client_gone(self):
        return False

    def _send_bytes(self, code, raw, extra_headers=()):
        self.sent = (code, raw)

    def _busy(self, code, message, retry=5):
        self.sent = (code, message)


@contextlib.contextmanager
def _served(app, pool):
    """`pool` as the process's KV pool and a slot-0 engine with paged KV, for one test."""
    from engine import kvpool
    saved, old = dict(app.STATE), kvpool.POOL
    app.STATE.clear()
    pages = _Pages()
    app.STATE.update(engine=types.SimpleNamespace(kv=types.SimpleNamespace(pages=pages, max_len=262_144, length=0)),
                     queue_timeout=120.0, request_timeout=3600.0)
    kvpool.POOL = pool
    try:
        yield pages
    finally:
        kvpool.POOL = old
        app.STATE.clear()
        app.STATE.update(saved)


def _wired(avail_gib: float, cap_gib: float = 10.0):
    """An admission and a pool on a MemAvailable the test moves; no memory policy, so the workspace is the cold
    prompt's (`cache.WORKSPACE_FLOOR` for a short one) and the guards are the configured floors. The fake pages answer
    the same bytes for the prompt as for the whole request."""
    box = {"avail": int(avail_gib * GiB)}
    adm = slots.Admission(6 * GiB, meminfo=lambda: {"MemAvailable": box["avail"]}, growth=lambda: 0)
    pool = slots.GuardedPool(hard_floor=int(cap_gib * GiB), margin_rows=0, meminfo=lambda: box["avail"])
    adm.pool_floor = pool.floor_at
    return box, adm, pool


def test_alone_and_short_the_pools_reclaim_levels_run_before_it_waits():
    """A lone request the pool's floor holds back asks the pool's reclaim list, in its order, for what it is short,
    any level in it (`graphs` stands for a level registered later), and is admitted at once when they gave enough."""
    import torch
    app = _app()
    box, adm, pool = _wired(11.0)
    calls = []

    def level(name, gives):
        def fn(want, owner):
            calls.append((name, want, owner))
            box["avail"] += gives
            return gives
        return fn
    pool.add_reclaimer("caches", level("caches", 0))
    pool.add_reclaimer("graphs", level("graphs", 2 * GiB))
    pool.add_reclaimer("kept_park", level("kept_park", GiB))
    with _served(app, pool):
        pr = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 0)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ok = app.Handler._par_admit(_H(app), pr, "lone", torch.zeros(512, dtype=torch.long), stream=True)
    short = 10 * GiB - (11 * GiB - (512 + 8_256) * PER_ROW)      # its prompt's pages under the pool's floor
    assert ok is True and "lone" in adm.held and adm.stats["waits"] == 0, out.getvalue()
    assert [c[0] for c in calls] == ["caches", "graphs"] and calls[0][1] == short and calls[0][2] == "slot0", calls
    assert "level=graphs bytes=2147483648" in out.getvalue(), out.getvalue()
    return f"short {short / GiB:.2f} GiB: caches 0, graphs 2.00, admitted alone without a wait"


def _lane(app, pool, make, stream=True, max_new=8, think=None, prompt=None, h=None, rec=None, **kw):
    """`_recomputing` on a lane of `pool`, `make` its generation: tokens, error, the client's bytes, rec, log."""
    import torch
    from server import usage
    h = h or _H(app)
    rec = rec or usage.RequestRecord("chatcmpl-lane", "chat", stream)
    got, err = [], None
    out = io.StringIO()
    p = prompt if prompt is not None else torch.zeros(64, dtype=torch.long)
    with contextlib.redirect_stdout(out):
        try:
            for t in app.Handler._recomputing(h, pool, "lane", rec, stream, p, max_new, think, make, **kw):
                got.append(int(t))
        except Exception as exc:                                   # noqa: BLE001
            err = exc
    return got, err, h.wfile.getvalue().decode(), rec, out.getvalue()


def _full():
    from engine import kvpool
    return kvpool.PoolFull("KV pool: committing 4 MiB for slot0 would leave 8.2 GiB available, under the hard floor")


def test_failure_2_a_lone_prefill_the_pool_refuses_waits_for_memory_and_goes_on():
    """A lone streamed lane whose prefill the gate refuses gives its pages back, waits in the admission's line while
    the pool's floor holds it back (one refusal, no re-admission until memory is there), and goes on by itself once
    memory frees: the whole answer, no error."""
    app = _app()
    box, adm, pool = _wired(10.5)
    calls = []

    def make(cur, left):
        calls.append(box["avail"])
        if box["avail"] < 12 * GiB:
            raise _full()
        yield from range(1, left + 1)
    was = app.KV_GIVE_WAY_BACKOFF_S
    app.KV_GIVE_WAY_BACKOFF_S = (0.2, 1.0)
    try:
        with _served(app, pool) as pages:
            threading.Timer(2.5, lambda: box.__setitem__("avail", 14 * GiB)).start()
            pr = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 1)
            t0 = time.monotonic()
            got, err, raw, rec, log = _lane(app, pr, make)
    finally:
        app.KV_GIVE_WAY_BACKOFF_S = was
    assert err is None and got == list(range(1, 9)), (err, got, log)
    assert len(calls) == 2 and rec.gave_way == 1 and pages.trims == 1, (calls, log)
    assert ": paused: waiting for memory (" in raw and time.monotonic() - t0 >= 2.4, raw
    assert "gives way after 0 tokens" in log and "0 lane(s) go on" in log and "admitted after" in log, log
    return f"refused once, waited {time.monotonic() - t0:.1f} s, went on: {len(got)} tokens"


def test_thinking_on_a_refusal_at_the_first_commit_starts_the_budget_as_it_would_have():
    """(2) A family that maps the prompt's rows before it arms the thinking budget (`_generate_stream`: `kv.ensure`,
    then `think.start`) is refused at its first commit: the budget was never started. The continuation arms it as an
    unrefused run does (open block, stall arms, its count), not with the never-started state. Refused after three
    reasoning tokens, the count goes on from three."""
    import torch
    from engine import spec
    from server import fake_engine
    app = _app()
    tok = fake_engine.FakeTokenizer()
    arms = {"loop": {"min_size": 2, "max_size": 4, "min_count": 3, "min_span": 8}}
    prompt = torch.tensor(tok.encode_text("<|im_start|>assistant\n<think>"), dtype=torch.long)

    def run(refuse_after):
        think = spec.ThinkBudget(tok, budget=100, arms=arms)
        tries = []

        def make(cur, left):
            tries.append(1)
            if len(tries) == 1 and refuse_after is None:
                raise _full()                          # before `think.start`, as the first commit
            think.start(cur.tolist())
            for j, t in enumerate(tok.encode_text("abcdefgh")[:left]):
                if len(tries) == 1 and j == refuse_after:
                    raise _full()
                think.observe([t])
                yield t
        box, adm, pool = _wired(50.0)
        was = app.KV_GIVE_WAY_BACKOFF_S
        app.KV_GIVE_WAY_BACKOFF_S = (0.01, 0.02)
        try:
            with _served(app, pool):
                pr = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 1)
                got, err, raw, rec, log = _lane(app, pr, make, think=think, prompt=prompt)
        finally:
            app.KV_GIVE_WAY_BACKOFF_S = was
        assert err is None and len(tries) == 2, (err, log[-400:])
        return think, got
    think, got = run(None)
    assert think.inside and think._arms is not None and think.n == 8 and len(got) == 8, vars(think)
    think, got = run(3)
    assert think.inside and think._arms is not None and think.n == 8 and len(got) == 8, (think.n, len(got))
    return "refused at the first commit: armed, 8 counted; after 3 tokens: 3 + 5"


def test_a_streamed_lane_gives_way_past_the_old_count_and_backs_off():
    """A streamed lane the pool refuses 20 times (more than `KV_RECOMPUTE_MAX`) is not stopped: it gives way each
    time, backs off longer each time with fewer than `KV_GIVE_WAY_PROGRESS` tokens written between them, and
    finishes; with that many written between them it starts from the first step again. With memory there the log and
    the stream say it backs off, not that it waits for memory."""
    app = _app()
    box, adm, pool = _wired(50.0)

    def run(between):
        times = []

        def make(cur, left):
            times.append(time.monotonic())
            if len(times) <= 20:
                yield from [7] * between
                raise _full()
            yield from range(left)
        with _served(app, pool):
            pr = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 2)
            got, err, raw, rec, log = _lane(app, pr, make, max_new=20 * between + 20)
        assert err is None and len(got) == 20 * between + 20 and rec.gave_way == 20, (err, len(got), log[-500:])
        return [b - a for a, b in zip(times, times[1:])], raw, log
    was = app.KV_GIVE_WAY_BACKOFF_S
    app.KV_GIVE_WAY_BACKOFF_S = (0.01, 0.08)
    try:
        gaps, raw, log = run(1)
        assert gaps[0] < gaps[4] and min(gaps[5:]) >= 0.075, gaps
        assert "backs off 0.01 s after giving way" in log and "waits for memory" not in log, log[:600]
        assert ": paused: backing off after giving way, memory is there" in raw, raw[:300]
        steady, _, _ = run(app.KV_GIVE_WAY_PROGRESS)
        assert max(steady) < 0.06, steady
    finally:
        app.KV_GIVE_WAY_BACKOFF_S = was
    assert app.KV_RECOMPUTE_MAX < 20
    return (f"20 give-ways, 1 token between them: gaps {gaps[0] * 1e3:.0f} to {max(gaps) * 1e3:.0f} ms; "
            f"{app.KV_GIVE_WAY_PROGRESS} tokens between them: at most {max(steady) * 1e3:.0f} ms")


def test_a_streamed_lane_the_pool_always_refuses_ends_with_the_waits_refusal():
    app = _app()
    box, adm, pool = _wired(50.0)

    def make(cur, left):
        raise _full()
        yield
    was = app.KV_GIVE_WAY_BACKOFF_S, app.MEM_WAIT_S
    app.KV_GIVE_WAY_BACKOFF_S, app.MEM_WAIT_S = (0.1, 0.2), 0.7
    try:
        with _served(app, pool):
            pr = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 1)
            t0 = time.monotonic()
            got, err, raw, rec, log = _lane(app, pr, make)
            took = time.monotonic() - t0
    finally:
        app.KV_GIVE_WAY_BACKOFF_S, app.MEM_WAIT_S = was
    assert isinstance(err, app.MemoryWaitTimeout), err
    body = app.memory_full_error(err)
    assert body["code"] == "memory_wait_timeout" and body["retry_after"] == 30, body
    assert 0.6 < took < 2.0 and 3 <= rec.gave_way <= 6, (took, rec.gave_way)
    return f"ended after {took:.1f} s of waits, {rec.gave_way} give-ways: {body['code']}"


def test_a_re_admission_is_priced_as_the_first_one():
    """(5) The continuation's admission gets its image rows (for the prompt with its text), `cold`, `home`, its priority,
    the request's deadline and its long wait, as the first one did."""
    import torch
    app = _app()
    box, adm, pool = _wired(50.0)
    seen = []
    h = _H(app)

    def admit(pool_, cid, cur, **kw):
        seen.append(dict(kw, n=int(cur.numel())))
        return True
    h._par_admit = admit
    tries = []

    def make(cur, left):
        tries.append(1)
        if len(tries) == 1:
            yield 5
            raise _full()
        yield from range(left)
    dl = app.Deadline(30.0)
    with _served(app, pool):
        pr = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 2)
        got, err, raw, rec, log = _lane(app, pr, make, h=h, deadline=dl,
                                        admit={"rows_of": lambda n: n + 1000, "cold": True, "home": 300,
                                               "priority": 2, "wait_long": False})
    assert err is None and len(seen) == 1, (err, seen)
    kw = seen[0]
    assert (kw["n"], kw["rows"], kw["cold"], kw["home"], kw["priority"], kw["deadline"]) == (65, 1065, True, 300, 2, dl)
    assert kw["stream"] is True and kw["wait_long"] is True and kw["respond"] is False
    return "rows 1,065 (64 + 1 + 1,000 image rows), cold, home 300, priority 2, deadline"


def test_a_lane_not_streamed_keeps_srv_709s_rule():
    """(7) Not streamed: alone, the refusal stands at once (`memory_full`, a 503); with another lane running it gives
    way at most `KV_RECOMPUTE_MAX` times, each wait at most `--queue-timeout`, backing off the first step each time."""
    app = _app()
    box, adm, pool = _wired(50.0)
    calls = []

    def make(cur, left):
        calls.append(time.monotonic())
        raise _full()
        yield
    with _served(app, pool):
        alone = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 1)
        got, err, raw, rec, log = _lane(app, alone, make, stream=False)
    from engine import kvpool
    assert isinstance(err, kvpool.PoolFull) and len(calls) == 1 and not getattr(rec, "gave_way", 0), (err, calls)
    assert app.memory_full_error(err)["code"] == "memory_full"
    calls.clear()
    was = app.KV_GIVE_WAY_BACKOFF_S
    app.KV_GIVE_WAY_BACKOFF_S = (0.001, 5.0)
    try:
        with _served(app, pool):
            two = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 2)
            got, err, raw, rec, log = _lane(app, two, make, stream=False)
    finally:
        app.KV_GIVE_WAY_BACKOFF_S = was
    assert isinstance(err, kvpool.PoolFull) and len(calls) == app.KV_RECOMPUTE_MAX + 1, (err, len(calls))
    assert rec.gave_way == app.KV_RECOMPUTE_MAX and raw == "" and calls[-1] - calls[0] < 3.0, raw
    return f"alone: 1 try; beside a lane: {len(calls)} tries in {calls[-1] - calls[0]:.2f} s, then memory_full"


def test_a_lane_not_streamed_that_waits_its_queue_timeout_gets_the_waits_readable_refusal():
    """(14) Not streamed, the wait after a give-way ends at `--queue-timeout` with the wait's refusal: 503, Retry-After,
    `memory_wait_timeout`, what it needed and how to wait longer; not `memory_full`."""
    app = _app()
    box, adm, pool = _wired(10.5)

    def make(cur, left):
        raise _full()
        yield
    with _served(app, pool):
        app.STATE["queue_timeout"] = 1.0
        two = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 2)
        t0 = time.monotonic()
        got, err, raw, rec, log = _lane(app, two, make, stream=False)
        took = time.monotonic() - t0
    assert isinstance(err, app.MemoryWaitTimeout) and 0.9 < took < 4.0, (err, took)
    body = app.memory_full_error(err)
    assert body["code"] == "memory_wait_timeout" and body["type"] == "server_busy" and body["retry_after"] == 30
    msg = body["message"]
    assert msg.startswith("Memory stayed full for ") and " free above the guard): this turn did not start; " \
        "send it again, or stream it (or send `wait_for_memory: true`) to wait longer." in msg, msg
    return msg


def test_a_planned_stop_during_the_wait_is_the_restarts_error():
    """(8) A lane waiting for memory after a give-way when the engine starts to stop: the restart's retryable error
    (`engine_restarting`), not `memory_full`."""
    app = _app()
    box, adm, pool = _wired(10.5)

    def make(cur, left):
        raise _full()
        yield
    with _served(app, pool):
        threading.Timer(0.8, lambda: app.STATE.__setitem__("draining", True)).start()
        pr = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 1)
        got, err, raw, rec, log = _lane(app, pr, make)
    assert isinstance(err, app.EngineRestarting), err
    body = app.memory_full_error(err)
    assert body["code"] == "engine_restarting" and body["retry_after"] == 30 and "restarting" in body["message"]
    return body["message"]


def test_through_the_handler_a_turn_waits_at_the_pools_floor_then_a_refused_prefill_goes_on():
    """(11) The whole path in process on the fake engine (`--parallel 3`) with a KV pool: a streamed chat whose prompt's
    pages the pool's floor holds back waits for memory (its stream open, its comments), is admitted when memory frees,
    its prefill is refused once (`FAKE_PREFILL_POOL_FULL_ONCE`), it gives way, backs off and finishes on the same
    stream: no error, one give-way, one resume. Every page went through the pool's gate (the fake commits them through
    `kv.ensure`), none refused, and the answer is the one the same request gets unrefused."""
    import json
    from engine import kvpool
    app = _app()
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from test_slot0_wait import _fake_parallel
    from test_ui_states import Req
    from server import fake_engine
    par = _fake_parallel()
    box = {"avail": int(10.5 * GiB)}
    adm = par.admission
    adm.cap = 6 * GiB
    adm.meminfo = lambda: {"MemAvailable": box["avail"]}
    kp = slots.GuardedPool(hard_floor=10 * GiB, margin_rows=0, meminfo=lambda: box["avail"])
    adm.pool_floor = kp.floor_at
    kv = fake_engine.FakeEngine.kv                  # the fakes' one `kv`, slot 0's and the side slots'
    old = kvpool.POOL
    kvpool.POOL = kp
    pages = _Pages(pool=kp, box=box)
    kv.pages, kv.max_len, kv.ensure = pages, 65_536, pages.ensure
    was = app.MEM_LINE_EVERY_S, app.MEM_WAIT_S
    app.MEM_LINE_EVERY_S, app.MEM_WAIT_S = 0.5, 60.0
    out: dict = {}
    body = {"model": "TandemLLM", "stream": True, "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "FAKE_PREFILL_POOL_FULL_ONCE through the handler"}]}

    def chat(key="raw"):
        head, raw = Req(body, {"User-Agent": "chatd/0.1 (qse-dashboard)"}).response()
        out.update({key: raw})

    def text(raw):
        got = []
        for line in raw.splitlines():
            if line.startswith("data: {"):
                for c in json.loads(line[6:]).get("choices") or []:
                    got.append((c.get("delta") or {}).get("content") or "")
        return "".join(got)
    th = threading.Thread(target=chat, daemon=True)
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            th.start()
            time.sleep(2.0)
            box["avail"] = 14 * GiB
            th.join(30)
            chat("again")                           # met the full pool already: not refused
    finally:
        app.MEM_LINE_EVERY_S, app.MEM_WAIT_S = was
        kvpool.POOL = old
        del kv.pages, kv.max_len, kv.ensure
    assert not th.is_alive()
    raw = out["raw"]
    assert ": waiting for memory" in raw and '"error"' not in raw and raw.rstrip().endswith("data: [DONE]"), raw[-600:]
    assert text(raw) and text(raw) == text(out["again"]), (text(raw), text(out["again"]))
    assert adm.stats["waits"] >= 1 and pages.commits >= 2 and kp.stats["refused"] == 0, (adm.stats, kp.stats)
    assert kp.stats["min_avail_after"] >= 10 * GiB, kp.stats
    rc = app.STATE.get("kv_recompute") or {}
    assert rc.get("gave_way") == 1 and rc.get("resumed") == 1 and rc.get("gave_up") == 0, rc
    return f"waited at the pool's floor, refused once in the prefill, finished; recompute {rc}"


def test_the_backoff_hold_starts_at_the_give_way_and_the_first_admissions_wait_counts():
    """(4) The lane is out of the line from its give-way on, before its re-admission prices and gates it (the gate's
    peak wait too). (5) Its first admission's wait counts toward `MEM_WAIT_S`: a lane that already waited the bound is
    not given another one."""
    from server import usage
    app = _app()
    box, adm, pool = _wired(50.0)
    h = _H(app)
    seen = []

    def admit(pool_, cid, cur, **kw):
        seen.append((dict(adm.holds), time.monotonic(), kw["hold_s"]))
        return True
    h._par_admit = admit
    tries = []

    def make(cur, left):
        tries.append(1)
        if len(tries) == 1:
            raise _full()
        yield from range(left)
    with _served(app, pool):
        pr = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 2)
        got, err, raw, rec, log = _lane(app, pr, make, h=h)
    holds, t, hold = seen[0]
    assert err is None and "lane" in holds and 0 < holds["lane"] - t <= hold + 0.01, (err, seen)
    was = app.MEM_WAIT_S
    app.MEM_WAIT_S = 0.5
    try:
        r0 = usage.RequestRecord("chatcmpl-lane", "chat", True)
        r0.wait_mem_ms = 600.0                    # its first admission waited 0.6 s of the 0.5
        tries.clear()
        with _served(app, pool):
            pr = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 2)
            got, err, raw, rec, log = _lane(app, pr, make, rec=r0)
    finally:
        app.MEM_WAIT_S = was
    assert isinstance(err, app.MemoryWaitTimeout) and len(tries) == 1 and not getattr(rec, "gave_way", 0), err
    return "held from the give-way on; a lane that waited its bound at admission ends at once"


def test_a_backoff_says_so_and_ends_admitted_at_its_bound():
    """(6) While a lane backs off with memory there it is not a memory wait (`wait_reason` "backoff", not
    `waiting_memory`). (7) A backoff longer than what is left of the bound stops at the bound, and the lane is admitted
    there, not refused with "Memory stayed full"."""
    from server import usage
    app = _app()
    box, adm, pool = _wired(50.0)
    tries, during = [], []
    rec = usage.RequestRecord("chatcmpl-lane", "chat", True)

    def make(cur, left):
        tries.append(1)
        if len(tries) == 1:
            raise _full()
        yield from range(left)

    def sample():
        time.sleep(0.5)
        during.append((rec.wait_reason, rec.waiting_memory))
    was = app.KV_GIVE_WAY_BACKOFF_S, app.MEM_WAIT_S
    app.KV_GIVE_WAY_BACKOFF_S, app.MEM_WAIT_S = (5.0, 5.0), 1.2
    try:
        with _served(app, pool):
            threading.Thread(target=sample, daemon=True).start()
            pr = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 2)
            t0 = time.monotonic()
            got, err, raw, rec, log = _lane(app, pr, make, rec=rec)
            took = time.monotonic() - t0
    finally:
        app.KV_GIVE_WAY_BACKOFF_S, app.MEM_WAIT_S = was
    assert err is None and len(got) == 8 and len(tries) == 2 and 1.0 < took < 2.5, (err, took, log[-400:])
    assert during == [("backoff", False)], during
    assert "backs off 5 s after giving way" in log and "admitted after" in log and "REFUSED" not in log, log
    return f"backing off, not waiting; admitted at its bound after {took:.1f} s"


def test_a_continuation_over_the_peak_is_refused_as_such():
    """(8) A continuation no reserve could take (the engine's load and its workspace over the peak) is refused at once
    on re-admission with that reason (`memory_refused`), not a "memory stayed full" wait refusal."""
    app = _app()
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from test_kvpool_admission import _Mem
    box, adm, pool = _wired(50.0)

    def make(cur, left):
        raise _full()
        yield
    with _served(app, pool):
        app.STATE.update(mem=_Mem(peak=10 * GiB, ws=11 * GiB), mem_load={"allocated": 0})
        pr = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 2)
        got, err, raw, rec, log = _lane(app, pr, make)
    assert isinstance(err, app.PeakRefused), err
    body = app.memory_full_error(err)
    assert body["code"] == "memory_refused" and "retry when the engine is idle" in body["message"], body
    return body["message"]


def test_the_backoff_counts_the_tokens_since_the_last_give_way():
    """(9) A lane writing 200 tokens between give-ways (fewer than `KV_GIVE_WAY_PROGRESS`) backs off longer each time;
    the count starts again at every give-way, not where the streak began."""
    app = _app()
    box, adm, pool = _wired(50.0)
    tries = []

    def make(cur, left):
        tries.append(1)
        if len(tries) <= 3:
            yield from [7] * 200
            raise _full()
        yield from range(left)
    was = app.KV_GIVE_WAY_BACKOFF_S
    app.KV_GIVE_WAY_BACKOFF_S = (0.01, 1.0)
    try:
        with _served(app, pool):
            pr = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 2)
            got, err, raw, rec, log = _lane(app, pr, make, max_new=700)
    finally:
        app.KV_GIVE_WAY_BACKOFF_S = was
    assert err is None and len(got) == 700, err
    holds = [x.split(" s at the soonest")[0].rsplit(" ", 1)[1] for x in log.splitlines() if "at the soonest" in x]
    assert holds == ["0.01", "0.02", "0.04"], holds
    return "200 tokens between give-ways: 0.01, 0.02, 0.04 s"


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            r = fn()
            n += 1
            print(f"ok  {name}" + (f"  {r}" if r else ""))
    print(f"{n} passed")
