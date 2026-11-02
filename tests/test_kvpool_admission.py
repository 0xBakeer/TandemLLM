"""SRV-307 + SRV-305, CPU only: the parallel admission with the shared KV pool, and the wait for
memory as the dashboard sees it.

  * a request's KV part (the pages its prompt and max_tokens may still commit) is reserved; what
    its slot has committed since it was admitted is credited, never counted twice;
  * the KV part is never capped by the engine's room to its peak (the pages are outside the
    memory policy);
  * a request alone is admitted, as before; a release frees its bytes and its credit record;
  * waiters for memory keep their arrival order (`wait` gives each its place), `unwait` takes one
    out, and the report counts them (`waiting_memory`);
  * the parallel view shows a held request's lane as `waiting_memory` with the label "waiting for
    memory", counts it in `totals.waiting_memory` and `waiting`, never in `decoding`;
  * the contract schema knows the new state and the new total.

Run: python tests/test_kvpool_admission.py
"""

from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from server import slots  # noqa: E402

GiB = slots.GiB


def _adm(avail_gib: float, floor_gib: float = 15, committed=None, room=None):
    return slots.Admission(int(floor_gib * GiB), meminfo=lambda: {"MemAvailable": int(avail_gib * GiB)},
                           growth=lambda: 0, room=room, committed=committed)


def test_kv_part_is_reserved_and_credited_as_it_commits():
    done = {"slot1": 0, "slot2": 0}
    adm = _adm(40, committed=lambda o: done[o])
    ok, _ = adm.reserve("a", 2 * GiB + 10 * GiB, kv=10 * GiB, owner="slot1")
    assert ok                                      # alone
    ok, d = adm.reserve("b", 1 * GiB + 14 * GiB, kv=14 * GiB, owner="slot2")
    # 40 - (2 + 10 held) - (1 + 14) = 13 < 15
    assert not ok and d["kv_reserved"] == 10 * GiB and d["after"] == 13 * GiB
    # slot 1 committed 6 GiB: MemAvailable fell by those 6, the reservation by the same 6
    done["slot1"] = 6 * GiB
    adm.meminfo = lambda: {"MemAvailable": 34 * GiB}
    ok, d = adm.reserve("b", 1 * GiB + 14 * GiB, kv=14 * GiB, owner="slot2")
    assert not ok and d["after"] == 13 * GiB, "the same answer: nothing counted twice"
    adm.release("a")
    ok, d = adm.reserve("b", 15 * GiB, kv=14 * GiB, owner="slot2")
    assert ok                                      # alone again
    assert adm.report()["kv_to_commit_gib"] == 14.0


def test_credit_never_goes_negative_or_past_the_kv_part():
    c = {"slot1": 5 * GiB}
    adm = _adm(100, committed=lambda o: c[o])
    adm.reserve("a", 3 * GiB, kv=2 * GiB, owner="slot1")
    c["slot1"] = 1 * GiB                           # trimmed below its baseline
    ws, kv = adm._held()
    assert (ws, kv) == (1 * GiB, 2 * GiB)
    c["slot1"] = 50 * GiB
    ws, kv = adm._held()
    assert (ws, kv) == (1 * GiB, 0)


def test_kv_is_not_capped_by_the_engine_room():
    adm = _adm(60, room=lambda: 1 * GiB)
    adm.reserve("x", GiB, owner=None)              # alone, workspace only
    ok, d = adm.check(2 * GiB + 30 * GiB, kv=30 * GiB)
    # workspace parts capped at the 1 GiB room (SRV-775: as before, the gate keeps each prefill under the
    # peak); the 30 GiB of pages counted in full
    assert d["after"] == 60 * GiB - 1 * GiB - 30 * GiB and ok


def test_plain_reservations_behave_as_before():
    adm = _adm(30)
    assert adm.reserve("a", 4 * GiB)[0]
    ok, d = adm.reserve("b", 12 * GiB)             # 30 - 4 - 12 = 14 < 15
    assert not ok and d["kv"] == 0
    assert adm.reserve("c", 10 * GiB)[0]           # 16 >= 15
    adm.release("a")
    adm.release("c")
    assert adm.reserved() == 0


def test_waiters_keep_their_order_and_are_counted():
    adm = _adm(10)
    assert adm.wait("w1") == 1
    assert adm.wait("w2") == 2
    assert adm.wait("w1") == 1                     # asking again keeps its place
    assert adm.report()["waiting_memory"] == 2
    adm.unwait("w1")
    assert adm.wait("w2") == 1
    adm.release("w2")
    assert adm.report()["waiting_memory"] == 0


def test_describe_names_the_kv_pages():
    adm = _adm(30)
    adm.reserve("a", 4 * GiB, kv=3 * GiB, owner="s1")
    ok, d = adm.check(20 * GiB, kv=18 * GiB)
    text = slots.Admission.describe(d)
    assert "KV pages: this request 18.0, in flight 3.0" in text and "< floor 15.0" in text


def test_parallel_lane_waiting_for_memory():
    from server import parallel

    class Rec:
        waiting_memory = True

    assert parallel.ParallelBoard._state(None, Rec(), None) == "waiting_memory"
    Rec.waiting_memory = False
    assert parallel.ParallelBoard._state(None, Rec(), 1.0) == "done"


def test_contract_knows_the_state_and_the_total():
    p = os.path.join(ROOT, "docs", "contract", "dashboard-v1", "parallel.schema.json")
    s = json.load(open(p))
    txt = json.dumps(s)
    assert '"waiting_memory"' in txt
    assert "waiting_memory" in s["properties"]["totals"]["properties"] if "totals" in s.get(
        "properties", {}) else "waiting_memory" in txt


def _app():
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_forward_tree  # noqa: F401  (the CPU environment before the engine imports)
    from server import app
    return app


class _H:
    """The handler's surface `_par_admit` uses."""

    def __init__(self):
        self.sent = None

    def _client_gone(self):
        return False

    def _send_bytes(self, code, raw, extra_headers=()):
        self.sent = (code, json.loads(raw), dict(extra_headers))

    def _busy(self, code, message, retry=5):
        self.sent = (code, {"error": {"message": message}}, {"Retry-After": str(retry)})


def test_a_non_streamed_wait_ends_in_a_readable_503():
    import types
    import torch
    app = _app()
    from server import usage
    app.STATE.clear()
    app.STATE["queue_timeout"] = 1.5
    app.STATE["request_timeout"] = 3600.0
    adm = _adm(18)                                 # SRV-775: a lone request is checked too: 18 - 2 >= 15
    assert adm.reserve("big", 2 * GiB)[0]
    pool = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 1)
    rec = usage.RequestRecord("chatcmpl-y", "chat", False)
    said = []
    h = _H()
    t0 = time.monotonic()
    assert app.Handler._par_admit(h, pool, "small", torch.zeros(576), rec=rec,
                                  say=lambda short, place=1: said.append((short, place)),
                                  stream=False) is False
    assert 1.4 < time.monotonic() - t0 < 4.0
    code, body, hdr = h.sent
    assert code == 503 and body["error"]["code"] == "memory_wait_timeout", body
    assert "memory is almost full" in body["error"]["message"] and hdr.get("Retry-After") == "30"
    assert rec.wait_reason == "memory" and rec.waiting_memory is False
    assert said and said[0][1] == 1, said              # the line, with its place in line
    assert adm.stats["timeouts"] == 1 and adm.report()["waiting_memory"] == 0
    assert h._mem_wait is True


def test_a_streamed_wait_outlasts_the_queue_timeout():
    import threading
    import types
    import torch
    app = _app()
    from server import usage
    app.STATE.clear()
    app.STATE["queue_timeout"] = 0.5
    app.STATE["request_timeout"] = 30.0
    box = {"avail": 18 * GiB}                     # SRV-775: a lone request is checked too: 18 - 2 >= 15
    adm = slots.Admission(15 * GiB, meminfo=lambda: {"MemAvailable": box["avail"]},
                          growth=lambda: 0)
    assert adm.reserve("big", 2 * GiB)[0]
    pool = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 1)
    rec = usage.RequestRecord("chatcmpl-z", "chat", True)
    rec.lock_acquired()

    def free_later():
        time.sleep(2.5)
        adm.release("big")
    threading.Thread(target=free_later, daemon=True).start()
    h = _H()
    assert app.Handler._par_admit(h, pool, "small", torch.zeros(576), rec=rec, stream=True) is True
    assert rec.wait_mem_ms >= 2000 and h.sent is None and rec.waiting_memory is False


def test_admission_takes_back_idle_conversations_before_it_waits():
    app = _app()
    from engine import kvpool
    calls = []
    pool = kvpool.KVPool(hard_floor=0, meminfo=lambda: 100 * GiB)
    pool.add_reclaimer("caches", lambda want, o: calls.append(("caches", want)) or GiB)
    pool.add_reclaimer("idle_side", lambda want, o: calls.append(("idle_side", want, o)) or 3 * GiB)
    pool.add_reclaimer("idle_slot0", lambda want, o: calls.append(("idle_slot0", want, o)) or 4 * GiB)
    old = kvpool.POOL
    kvpool.POOL = pool
    try:
        got = app._kv_reclaim_idle({"floor": 15 * GiB, "after": 10 * GiB}, "slot2")
        assert got == 7 * GiB, got
        assert calls == [("idle_side", 5 * GiB, "slot2"), ("idle_slot0", 2 * GiB, "slot2")], calls
        calls.clear()
        assert app._kv_reclaim_idle({"floor": 15 * GiB, "after": 13 * GiB}, "slot1") == 3 * GiB
        assert [c[0] for c in calls] == ["idle_side"], "stops once enough came back"
        assert app._kv_reclaim_idle({"floor": 15 * GiB, "after": None}, "slot1") == 0
    finally:
        kvpool.POOL = old


def test_a_release_for_a_sandbox_gives_back_idle_memory_in_the_pools_order():
    """SRV-483: chatd's sandbox has priority over what the engine holds idle."""
    app = _app()
    from engine import kvpool
    calls = []
    pool = kvpool.KVPool(hard_floor=0, meminfo=lambda: 100 * GiB)
    pool.add_reclaimer("caches", lambda want, o: calls.append(("pool caches", want)) or GiB)
    pool.add_reclaimer("roles", lambda want, o: calls.append(("roles", want, o)) or GiB)
    pool.add_reclaimer("idle_side", lambda want, o: calls.append(("idle_side", want, o)) or 2 * GiB)
    pool.add_reclaimer("idle_slot0", lambda want, o: calls.append(("idle_slot0", want, o)) or GiB)
    old = kvpool.POOL, app._kv_reclaim_caches, app._lock_when_idle
    kvpool.POOL = pool
    app._kv_reclaim_caches = lambda want, o: calls.append(("caches", want, o)) or GiB
    try:
        out = app._memory_release(3 * GiB)
        assert calls == [("caches", 3 * GiB, "release"), ("roles", 2 * GiB, "release"),
                         ("idle_side", GiB, "release")], calls
        assert out["released_bytes"] == 4 * GiB and out["asked_bytes"] == 3 * GiB
        assert out["levels"] == {"caches": GiB, "roles": GiB, "idle_side": 2 * GiB}
        assert pool.stats["reclaims"] == 2 and pool.stats["reclaimed_bytes"] == 3 * GiB
        # a request runs: the caches stay, the idle conversations still give way
        app._lock_when_idle = lambda: False
        calls.clear()
        assert app._memory_release(GiB)["levels"] == {"roles": GiB}
        assert calls == [("roles", GiB, "release")]
        calls.clear()
        assert app._memory_release(0)["released_bytes"] == 0 and calls == []
    finally:
        kvpool.POOL, app._kv_reclaim_caches, app._lock_when_idle = old


def test_spd271_every_kv_reclaim_resets_the_heads():
    """SPD-271: each KV reclaim that empties a slot (an idle side slot's, slot 0's idle conversation, a lane that gives
    its memory back) sets the engine's length to 0; on Flash-Next that restart now resets the MTP head with the
    trunk and forgets the failed idle captures (before, slot 1's head stayed at 85,010 over a trunk at 0)."""
    import types
    from engine.models.flashnext import serving as FS
    app = _app()

    def fake_engine():
        trims = []
        pages = types.SimpleNamespace(committed_bytes=lambda: 3 * GiB, margin=256, rows=0,
                                      trim=lambda n: trims.append(n) or 3 * GiB)
        st = types.SimpleNamespace(length=85_010, pages=pages)
        st.reset_inplace = lambda: setattr(st, "length", 0)
        head = types.SimpleNamespace(st=types.SimpleNamespace(length=85_010, pool_n={1: 1}), depth=3)
        head.reset = lambda: (setattr(head.st, "length", 0), setattr(head.st, "pool_n", {}))
        srv = types.SimpleNamespace(_chain=None, picks=None, fn=types.SimpleNamespace(drop_parked=lambda st: None),
                                    st=st, heads={"mtp": head}, _capture_failed={16384: 1}, max_len=262_144,
                                    _settle_chain=lambda: None)
        srv.restart = types.MethodType(FS.FlashNextServing.restart, srv)
        return types.SimpleNamespace(kv=FS._KV(srv)), srv, head, trims

    saved = {k: app.STATE.get(k) for k in ("pool", "engine", "resident")}
    try:
        for name in ("idle_side", "idle_slot0", "give_back_own"):
            eng, srv, head, trims = fake_engine()
            if name == "idle_side":
                app.STATE["pool"] = slots.Pool([slots.Slot(0, 262_144), slots.Slot(
                    1, 262_144, values={"engine": eng, "resident": None})])
                got = app._kv_reclaim_side(GiB, "slot0")
            else:
                app.STATE["pool"], app.STATE["engine"], app.STATE["resident"] = None, eng, None
                got = app._kv_reclaim_slot0(GiB, "slot3") if name == "idle_slot0" else app._kv_give_back_own()
            assert got == 3 * GiB and trims == [0], (name, got, trims)
            assert srv.st.length == 0 and head.st.length == 0 and head.st.pool_n == {}, (name, head.st)
            assert srv._capture_failed == {}, (name, srv._capture_failed)
    finally:
        for k, v in saved.items():
            if v is None:
                app.STATE.pop(k, None)
            else:
                app.STATE[k] = v
    return "idle side slot, slot 0's idle conversation, a lane's give-back: the head reset each time"


class _Mem:
    """The memory policy's surface `_par_admit` uses (engine/cache.py `Budget`), recording what it is asked: a
    workspace of `ws` whatever the rows, a gate that refuses while `refuse()` says so."""
    on = True

    def __init__(self, peak, ws, refuse=None, res=None):
        self.peak, self.ws, self.refuse, self.res = peak, ws, refuse, res
        self.gates, self.refusals, self.over = [], [], []

    def workspace_rows(self, n, rows=None, line=True):
        return self.ws

    def gate(self, n, count=True, rows=None, others=0, ceiling=True):
        self.gates.append({"n": n, "rows": rows, "others": others, "count": count, "ceiling": ceiling,
                           "held": getattr(self.res, "end_hold", None)})
        if self.refuse is not None and self.refuse():
            from engine import cache
            exc = cache.MemoryRefused(self.peak + GiB, self.peak, n, rows=rows)
            exc.detail = {"allocated": self.peak - GiB, "reserved": self.peak}
            raise exc
        return {}

    def refused(self, exc):
        self.refusals.append(exc)

    def admit_over_peak(self, n, rows, ws, slot, **info):
        self.over.append((n, rows, ws, slot, info))

    def prefill_done(self):
        pass


def _admit(app, pool, cid, n, rec=None, stream=True):
    import torch
    h = _H()
    return app.Handler._par_admit(h, pool, cid, torch.zeros(n, dtype=torch.long), rec=rec, stream=stream), h


def test_srv773_no_reclaim_for_a_request_the_gate_refuses():
    """A prefill no reserve could ever take (the engine's load plus its workspace over the peak) gets its 503 at once:
    no gate, no shed, and no idle conversation reclaimed for it (live 2026-11-02: 7 of 23 reclaims served retries that
    slot 0's gate then refused)."""
    import types
    app = _app()
    from engine import kvpool
    from server import usage
    calls = []
    kp = kvpool.KVPool(hard_floor=0, meminfo=lambda: 100 * GiB)
    kp.add_reclaimer("idle_side", lambda want, o: calls.append(("idle_side", want)) or GiB)
    kp.add_reclaimer("idle_slot0", lambda want, o: calls.append(("idle_slot0", want)) or GiB)
    old = kvpool.POOL
    app.STATE.clear()
    mem = _Mem(peak=10 * GiB, ws=11 * GiB)
    app.STATE.update(mem=mem, mem_load={"allocated": 0}, queue_timeout=1.0, request_timeout=30.0)
    kvpool.POOL = kp
    try:
        adm = _adm(16)
        pool = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 0)
        rec = usage.RequestRecord("chatcmpl-r3", "chat", True)
        ok, h = _admit(app, pool, "r3", 262_144, rec=rec)
        assert ok is False and rec.stop_detail == "memory", (ok, rec.stop_detail)
        code, body, hdr = h.sent
        assert code == 503 and hdr.get("Retry-After") == "30", (code, hdr)
        assert "a 262144-token prefill needs 11.0 GiB" in body["error"]["message"], body
        assert calls == [] and mem.gates == [] and len(mem.refusals) == 1 and not adm.held, (calls, mem.gates)
    finally:
        kvpool.POOL = old
        app.STATE.clear()
    return "262,144 cold rows, 11 GiB over a 10 GiB peak: 503 at once, no gate, no reclaim"


class _Res:
    """A slot's resident prefix as the admission reads it: g8's conversation with its end kept at 62,865."""

    def __init__(self):
        self.end, self.end_hold, self.asked = {"at": 62_865}, False, []

    def resume_estimate(self, keys, n, mm_ok=False):
        self.asked.append((len(keys), n, mm_ok))
        return 62_865, "end"

    def common(self, keys):
        return min(len(keys), 62_865)

    def end_offer(self, c, max_len, mm_ok=False):
        return 62_865 if c >= 62_865 and max_len >= 62_865 else 0


def test_srv773_a_resumed_turn_is_admitted_with_the_end_held():
    """g8's shape: a 62,886-token follow-up whose slot keeps its end at 62,865 is charged its 21 rows, gated with
    that end held (the gate's levels never give it back for this request), and admitted."""
    import types
    app = _app()
    from server import usage
    app.STATE.clear()
    res = _Res()
    mem = _Mem(peak=int(75.8 * GiB), ws=int(0.8 * GiB), res=res)
    app.STATE.update(mem=mem, resident=res, mem_load={"allocated": int(57.8 * GiB)}, queue_timeout=1.0,
                     request_timeout=30.0)
    try:
        adm = _adm(100)
        pool = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 0)
        rec = usage.RequestRecord("chatcmpl-g8", "chat", True)
        ok, h = _admit(app, pool, "g8", 62_886, rec=rec)
        assert ok is True and h.sent is None
        assert res.asked and set(res.asked) == {(62_886, 62_886, False)}, res.asked   # priced at every pass
        assert mem.gates == [{"n": 62_886, "rows": 21, "others": 0, "count": False, "ceiling": True,
                              "held": True}], mem.gates
        assert rec.ws_rows == 21 and rec.ws_kind == "end" and abs(rec.ws_gib - 0.8) < 1e-6, vars(rec)
        assert adm.held["g8"] == int(0.8 * GiB) and mem.over == []
    finally:
        app.STATE.clear()
    return "rows 21 at the gate with the end held; 0.80 GiB reserved"


def test_srv775_a_lone_refusal_is_admitted_over_the_peak_and_counted():
    """A prefill the gate refuses runs over the peak when no other prefill is in flight, counted once with its slot;
    with another one in flight it waits, the gate run again when it is first in line, and is admitted once that
    prefill has run and the gate admits it."""
    import threading
    import types
    app = _app()
    from server import usage
    app.STATE.clear()
    refuse = {"on": True}
    mem = _Mem(peak=75 * GiB, ws=int(9.9 * GiB), refuse=lambda: refuse["on"])
    app.STATE.update(mem=mem, mem_load={"allocated": 58 * GiB}, queue_timeout=1.0, request_timeout=30.0)
    try:
        adm = _adm(100)
        asked = []
        reserve = adm.reserve
        adm.reserve = lambda key, need, **kw: asked.append((key, kw.get("over_peak"))) or reserve(key, need, **kw)
        pool = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 1)
        slots.bind(slots.Slot(3, 262_144))
        try:
            ok, h = _admit(app, pool, "g2", 135_083, rec=usage.RequestRecord("chatcmpl-g2", "chat", True))
        finally:
            slots.bind(None)
        assert ok is True and h.sent is None and adm.stats["peak_waits"] == 0
        assert mem.over == [(135_083, 135_083, int(9.9 * GiB), 3, {"cid": "g2", "resume": None})], mem.over
        assert [g["ceiling"] for g in mem.gates] == [False], "a side slot's gate: the peak stage only"
        assert asked == [("g2", True)], "the over-peak admit's workspace counts in full at the floor"
        adm.release("g2")
        assert adm.reserve("other", 4 * GiB)[0]       # another prefill in flight
        rec = usage.RequestRecord("chatcmpl-w", "chat", True)
        rec.lock_acquired()

        def other_done():
            time.sleep(1.5)
            refuse["on"] = False
            adm.prefilled("other")
        threading.Thread(target=other_done, daemon=True).start()
        t0 = time.monotonic()
        ok, h = _admit(app, pool, "w", 135_083, rec=rec)
        assert ok is True and h.sent is None and time.monotonic() - t0 >= 1.4, time.monotonic() - t0
        assert len(mem.over) == 1 and adm.stats["peak_waits"] == 1 and rec.wait_mem_ms >= 1000, adm.stats
        assert len(mem.gates) >= 3 and all(g["count"] is False for g in mem.gates), mem.gates
        assert mem.gates[-1]["ceiling"] is True and asked[-1] == ("w", False), (mem.gates[-1], asked)
    finally:
        app.STATE.clear()
    return "alone: over the peak, counted on slot 3; beside a prefill: waited, then admitted"


def test_srv775_strict_keeps_a_lone_request_waiting():
    """QSE_PEAK_STRICT: a lone request the gate refuses waits too; a non-streamed one gets its 503 after the queue
    timeout, and nothing ran over the peak."""
    import types
    app = _app()
    from server import usage
    app.STATE.clear()
    mem = _Mem(peak=75 * GiB, ws=int(9.9 * GiB), refuse=lambda: True)
    app.STATE.update(mem=mem, mem_load={"allocated": 58 * GiB}, queue_timeout=1.0, request_timeout=30.0)
    was = app.PEAK_STRICT
    app.PEAK_STRICT = True
    try:
        adm = _adm(100)
        pool = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 0)
        rec = usage.RequestRecord("chatcmpl-s", "chat", False)
        ok, h = _admit(app, pool, "s", 135_083, rec=rec, stream=False)
        assert ok is False and h.sent[0] == 503 and h.sent[1]["error"]["code"] == "memory_wait_timeout", h.sent
        assert mem.over == [] and adm.stats["peak_waits"] == 1 and rec.stop_detail == "memory_wait_timeout"
        assert mem.gates and all(g["others"] == 0 and g["count"] is False for g in mem.gates), mem.gates
    finally:
        app.PEAK_STRICT = was
        app.STATE.clear()
    return "strict: waited, 503 after the queue timeout, nothing over the peak"


class _SpareMem(_Mem):
    """`_Mem` with the saved state of 2026-11-02 (allocated 64.71, reserved 79.70 over a 75.82 peak) and the
    allocator's free default-pool bytes the policy reads (`workspace_spare`)."""

    def __init__(self, *a, spare=0, **kw):
        super().__init__(*a, **kw)
        self.spare, self.spares = spare, 0

    def memory(self):
        return int(64.71 * GiB), int(79.70 * GiB)

    def workspace_spare(self):
        self.spares += 1
        return self.spare


def test_srv775_a_lone_over_peak_admit_is_credited_the_free_blocks_and_starts_at_once():
    """SRV-775 (2026-11-02): a lone 31,053-row chat on a server whose reserve sat 3.9 GiB over the peak was gate-refused,
    admitted over the peak, then charged its whole 3.89 GiB workspace at the box's floor: MemAvailable 13.8 - 3.89 =
    9.9 < 10, and it waited. The workspace takes the allocator's 2.28 GiB of free default-pool blocks first, which
    MemAvailable already leaves out: 13.8 - 1.61 = 12.2, admitted on the first reservation, no wait."""
    import contextlib
    import io
    import types
    app = _app()
    from server import usage
    app.STATE.clear()
    ws = int(3.89 * GiB)
    mem = _SpareMem(peak=int(75.82 * GiB), ws=ws, refuse=lambda: True, spare=int(2.28 * GiB))
    app.STATE.update(mem=mem, mem_load={"allocated": int(54.4 * GiB)}, queue_timeout=1.0, request_timeout=30.0)
    try:
        adm = slots.Admission(10 * GiB, meminfo=lambda: {"MemAvailable": int(13.8 * GiB)},
                              growth=lambda: int(13.6 * GiB), room=app._par_room, spare=app._par_spare)
        pool = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 0)
        rec = usage.RequestRecord("chatcmpl-98df742", "chat", True)
        rec.lock_acquired()
        out = io.StringIO()
        t0 = time.monotonic()
        with contextlib.redirect_stdout(out):
            ok, h = _admit(app, pool, "98df742", 31_053, rec=rec)
        assert ok is True and h.sent is None and time.monotonic() - t0 < 0.9, (ok, h.sent)
        assert adm.stats["waits"] == 0 and rec.wait_mem_ms is None and "waits for memory" not in out.getvalue()
        assert len(mem.over) == 1 and mem.spares == 1 and adm.held["98df742"] == ws, (mem.over, adm.held)
        adm.release("98df742")
        mem.spare = 0                             # no free blocks: the whole workspace is new, as before
        ok, d = adm.check(ws, count=False, over_peak=True)
        assert not ok and abs(d["after"] / GiB - 9.91) < 0.01, d
    finally:
        app.STATE.clear()
    return "13.8 - (3.89 - 2.28) = 12.2 >= 10: admitted at once; without the credit 9.9 < 10"


def test_srv777_a_client_that_leaves_during_the_memory_wait_is_booked_abandoned():
    """SRV-777: a client that closed its socket while its request waited for memory (chatd's Stop, before any header)
    ended with no `[req]` line, uncounted, its live row "error: internal error after 0 tokens". It is booked as the
    queue books one that left: `INFLIGHT["abandoned"]` + 1, finish `abandoned` with its detail, the `[req]` line,
    and no waiter left in the admission. A lane resuming inside its own answer (`respond=False`) is the stream's to
    book: nothing is counted here."""
    import contextlib
    import io
    import types
    import torch
    app = _app()
    from server import activity, usage
    app.STATE.clear()
    app.STATE["queue_timeout"] = 30.0
    app.STATE["request_timeout"] = 3600.0

    class _Leaves(_H):
        def __init__(self):
            super().__init__()
            self.polls = 0

        def _client_gone(self):
            self.polls += 1
            return self.polls >= 2                 # gone at the second poll, one second into the wait

    for respond in (True, False):
        adm = _adm(18)                             # 18 - 2 (in flight) - its workspace < 15: it waits
        assert adm.reserve("big", 2 * GiB)[0]
        pool = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 1)
        rec = usage.RequestRecord("chatcmpl-left", "chat", True)
        rec.lock_acquired()
        h = _Leaves()
        before = app.INFLIGHT["abandoned"]
        out = io.StringIO()
        t0 = time.monotonic()
        try:
            with contextlib.redirect_stdout(out):
                app.Handler._par_admit(h, pool, "w1", torch.zeros(576, dtype=torch.long), rec=rec, stream=True,
                                       respond=respond)
            raise AssertionError("ClientGone expected")
        except app.ClientGone:
            pass
        assert 0.9 < time.monotonic() - t0 < 3.0 and h.polls == 2 and h.sent is None, (time.monotonic() - t0, h.polls)
        assert adm.report()["waiting_memory"] == 0 and rec.waiting_memory is False and "w1" not in adm.held
        lines = [ln for ln in out.getvalue().splitlines() if ln.startswith("[req] w1 left while waiting")]
        if not respond:
            assert app.INFLIGHT["abandoned"] == before and rec.finish_reason is None and not lines, out.getvalue()
            continue
        assert app.INFLIGHT["abandoned"] == before + 1 and len(lines) == 1, out.getvalue()
        assert lines[0].startswith("[req] w1 left while waiting for memory (") and lines[0].endswith(" ms)"), lines
        assert rec.finish_reason == "abandoned" and rec.stop_detail == "left_while_waiting_memory", vars(rec)
        assert rec.client_gone_at is not None and rec.wait_mem_ms >= 900, rec.wait_mem_ms
        stop = activity.stop_of(rec, time.perf_counter())
        assert stop["reason"] == "abandoned" and stop["detail"] == "left_while_waiting_memory", stop
        assert stop["sentence"].startswith("abandoned by the client after ") and \
            stop["sentence"].endswith(" waiting for memory, 0 tokens sent"), stop["sentence"]
    return "left at the second poll: abandoned +1, one [req] line, no waiter; a resuming lane: not booked here"


def test_srv775_the_workspace_is_released_when_the_prefill_has_run():
    """`_par_prefilled`: the admission keeps the request's KV part only, and the memory policy's prefills in flight
    lose its workspace (a side slot prefills with no policy, so `Budget.prefilled` never ran there)."""
    app = _app()
    from engine import cache
    app.STATE.clear()
    m = cache.Budget(70 * GiB, 77 * GiB, 2 * GiB, probe=lambda: (60 * GiB, 60 * GiB), empty=lambda: None,
                     peak_probe=lambda reset=False: 0, ws_rows=lambda n, r: GiB)
    adm = _adm(100)
    app.STATE.update(mem=m, pool=slots.Pool([slots.Slot(0, 262_144)], admission=adm))
    try:
        m.gate(62_886, rows=21)
        assert adm.reserve("c1", GiB + 3 * GiB, kv=3 * GiB)[0]
        assert m.ws_now == GiB and adm.ws_in_flight() == GiB
        app._par_prefilled("c1")
        assert adm.held["c1"] == 3 * GiB and adm.ws_in_flight() == 0 and m.ws_now == 0, adm.held
    finally:
        app.STATE.clear()
    return "held 4 -> 3 GiB (its KV part), ws_now 1 -> 0"


class _LineMem(_Mem):
    """`_Mem` whose gate charge is the line's floor (2.75 GiB) and whose count by rows alone is 0.83 GiB."""

    def workspace_rows(self, n, rows=None, line=True):
        return int(2.75 * GiB) if line else int(0.83 * GiB)


def test_srv775_eight_short_cold_prompts_are_admitted_together():
    """The box's floor counts a prefill in flight by its rows: eight cold prompts of 205 tokens (bare_bench c8) are
    admitted at once with MemAvailable 44.5 GiB, cache growth 13.4 and the floor at 10 (8 x 0.83 GiB in flight).
    Counted at the gate's 2.75 GiB each, the eighth found 44.5 - 13.4 - 7 x 2.75 - 2.75 = 9.1 GiB and waited: on
    2026-11-02 it waited 12 s a round, and the round's first tokens came after 10.5-13.7 s (before SRV-775: 2.0-2.3 s).
    With the engine's room to its peak (13.5 GiB at that reserve) the growth and the workspaces in flight are
    capped at it, as before SRV-775: a family without a rows model (2.75 GiB each, by rows too) starts all eight."""
    import types
    app = _app()
    from server import usage
    app.STATE.clear()
    mem = _LineMem(peak=200 * GiB, ws=0)
    app.STATE.update(mem=mem, mem_load={"allocated": 0}, queue_timeout=0.5, request_timeout=30.0)
    try:
        for cls_, want in ((_LineMem, 8), (_Mem, 7)):
            app.STATE["mem"] = cls_(peak=200 * GiB, ws=int(2.75 * GiB))
            adm = slots.Admission(10 * GiB, meminfo=lambda: {"MemAvailable": int(44.5 * GiB)},
                                  growth=lambda: int(13.4 * GiB))
            pool = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 7)
            got = []
            for i in range(8):
                ok, h = _admit(app, pool, f"c{i}", 205, rec=usage.RequestRecord(f"cmpl-{i}", "chat", False),
                               stream=False)
                got.append(ok)
            assert sum(got) == want, (cls_.__name__, got, adm.stats)
            assert adm.stats["waits"] == 8 - want, adm.stats
        assert adm.held and all(v == int(2.75 * GiB) for v in adm.held.values())
        app.STATE["mem"] = _Mem(peak=200 * GiB, ws=int(2.75 * GiB))     # no rows model: 2.75 by rows too
        adm = slots.Admission(10 * GiB, meminfo=lambda: {"MemAvailable": int(44.5 * GiB)},
                              growth=lambda: int(13.4 * GiB), room=lambda: int(13.5 * GiB))
        pool = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 7)
        got = [_admit(app, pool, f"r{i}", 205, rec=usage.RequestRecord(f"cmpl-r{i}", "chat", False),
                      stream=False)[0] for i in range(8)]
        ok, d = adm.check(int(2.75 * GiB), count=False)
        assert all(got) and adm.stats["waits"] == 0 and d["after"] == int(31.0 * GiB), (got, adm.stats, d)
        assert "under the room" in slots.Admission.describe(d), slots.Admission.describe(d)
    finally:
        app.STATE.clear()
    return ("by rows: 8 of 8 admitted at once; by the gate's line: the eighth waits (and gets its 503 here); "
            "under the room: 8 of 8 at 2.75 GiB each")


class _Clock:
    """server/app.py's `time` for one `_par_admit` call: `sleep` moves the clock by its seconds and runs the steps of
    `at` (second -> callable) the clock has reached; everything else is the real module."""

    def __init__(self, at=None):
        self.t0 = self.t = 1000.0
        self.at = dict(at or {})

    def monotonic(self):
        return self.t

    perf_counter = monotonic

    def sleep(self, s):
        self.t += s
        for k in sorted(k for k in self.at if k <= self.t - self.t0):
            self.at.pop(k)()

    def __getattr__(self, k):
        return getattr(time, k)


def test_srv775_a_peak_wait_runs_the_gate_again_only_when_a_prefill_has_run_or_every_5_s():
    """A request the gate refuses beside another prefill waits for it. A gate pass sheds, collects and empties the
    allocator while the other lanes decode; it ran every second of the wait. Now it runs again once the prefills in
    flight dropped, the resume point moved, or `PEAK_REGATE_S` (5 s) passed. On a clock the test moves: gated at
    0 s, not at 1-4 s, again at 5 s (still refused), and at 7 s, the second the other prefill ran; admitted then."""
    import types
    app = _app()
    clock = _Clock()

    class _Timed(_Mem):
        def gate(self, n, count=True, rows=None, others=0, ceiling=True):
            self.at.append(round(clock.t - clock.t0, 1))
            return super().gate(n, count=count, rows=rows, others=others, ceiling=ceiling)

    refuse = {"on": True}
    mem = _Timed(peak=75 * GiB, ws=int(2 * GiB), refuse=lambda: refuse["on"])
    mem.at = []
    app.STATE.clear()
    app.STATE.update(mem=mem, mem_load={"allocated": 58 * GiB}, queue_timeout=1.0, request_timeout=30.0)
    adm = _adm(100)
    assert adm.reserve("other", 4 * GiB)[0]          # another prefill in flight

    def other_ran():
        refuse["on"] = False
        adm.prefilled("other")
    clock.at[7] = other_ran
    old = app.time
    app.time = clock
    try:
        pool = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 1)
        ok, h = _admit(app, pool, "pw", 62_886)
    finally:
        app.time = old
        app.STATE.clear()
    assert ok is True and h.sent is None, h.sent
    assert mem.at == [0.0, 5.0, 7.0], mem.at
    assert adm.stats["peak_waits"] == 1 and mem.over == [] and adm.held["pw"] == 2 * GiB, (adm.stats, mem.over)
    return f"gated at {mem.at} s of the wait"


def test_srv775_a_member_that_waits_for_memory_leaves_the_prefill_burst():
    """A member of the prefill burst that waits for memory at its admission leaves the burst: the others close it
    without it, where they held their first tokens for it up to the burst's wait (2026-11-02: one member's 12 s wait
    kept the seven others' first tokens at 10.5-13.7 s). Admitted later, it is no member (it prefills alone), and
    `_par_end`'s leave is a no-op."""
    import threading
    import types
    app = _app()
    from server.prefill_burst import Burst
    app.STATE.clear()
    app.STATE.update(queue_timeout=10.0, request_timeout=30.0)
    burst = Burst(wait_s=10)
    for who in (1, 2, 3):
        assert burst.arrive(who, idle=(who == 1))
    box = {"avail": 18 * GiB}                          # 18 - 2 - 1.15 < 15: member 3 waits at the box's floor
    adm = slots.Admission(15 * GiB, meminfo=lambda: {"MemAvailable": box["avail"]}, growth=lambda: 0)
    assert adm.reserve("big", 2 * GiB)[0]
    pool = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 2, burst=burst)
    got = {}

    def member3():
        slots.bind(slots.Slot(3, 262_144))
        try:
            got["ok"] = _admit(app, pool, "m3", 576)[0]
        finally:
            slots.bind(None)
    th = threading.Thread(target=member3, daemon=True)
    try:
        th.start()
        t_end = time.monotonic() + 5
        while (burst.member(3) or "m3" not in adm.waiting) and time.monotonic() < t_end:
            time.sleep(0.02)
        assert not burst.member(3) and "m3" in adm.waiting, (burst.pending, adm.waiting)
        waited = {}
        t1 = threading.Thread(target=lambda: waited.update(
            one=burst.prefilled(1, release=lambda: None, acquire=lambda: None)), daemon=True)
        t0 = time.monotonic()
        t1.start()
        time.sleep(0.1)
        assert burst.prefilled(2) is False                  # the last member left: it closes the burst
        t1.join(5)
        assert waited.get("one") is True and time.monotonic() - t0 < 2.0, (waited, time.monotonic() - t0)
        assert burst.stats["bursts"] == 1 and burst.stats["max_members"] == 3 and not burst.is_open()
        box["avail"] = 100 * GiB
        th.join(5)
        assert got.get("ok") is True and not burst.member(3), got
        before = dict(burst.stats)
        burst.leave(3)                                      # `_par_end`'s
        assert burst.stats == before and not burst.is_open()
    finally:
        box["avail"] = 100 * GiB
        th.join(5)
        app.STATE.clear()
    return "member 3 waited for memory and left; 1 and 2 closed the burst without it; admitted, it is no member"


class _Branch:
    """A branch's resident prefix as the admission reads it: it resumes at the anchor `points[0]` (no end, no prompt
    end); a shed of the anchors (`shed`) gives that one back, and it resumes at the next."""

    def __init__(self, points):
        self.points, self.end, self.end_hold = list(points), None, False

    def resume_estimate(self, keys, n, mm_ok=False):
        return self.points[0], "resident"

    def shed(self):
        if len(self.points) > 1:
            self.points.pop(0)


class _RowsMem(_Mem):
    """`_Mem` with a workspace of 1 MiB a forwarded row, and `on_gate(i)` run inside the i-th pass, before its
    verdict (where the real gate sheds)."""

    def __init__(self, peak, refuse=None, res=None, on_gate=None):
        super().__init__(peak, 0, refuse=refuse, res=res)
        self.on_gate = on_gate

    def workspace_rows(self, n, rows=None, line=True):
        return (n if rows is None else rows) << 20

    def gate(self, n, count=True, rows=None, others=0, ceiling=True):
        if self.on_gate is not None:
            self.on_gate(len(self.gates))
        return super().gate(n, count=count, rows=rows, others=others, ceiling=ceiling)


def test_srv773_a_pass_that_sheds_the_anchor_a_branch_resumes_at_runs_again_on_the_new_rows():
    """The gate's anchors level keeps the newest anchors and the head lines: a branch resumes at one in the middle,
    and the pass that admitted its 4,886 rows gave that anchor back. Priced again after the pass, the prefill
    forwards 58,790 rows from the head line at 4,096: the gate runs again on those, and the reservation and the
    `[req]` fields carry them, not the 4,886."""
    import types
    app = _app()
    from server import usage
    app.STATE.clear()
    res = _Branch([58_000, 4_096])
    mem = _RowsMem(peak=200 * GiB, res=res, on_gate=lambda i: res.shed() if i == 0 else None)
    app.STATE.update(mem=mem, resident=res, mem_load={"allocated": 0}, queue_timeout=1.0, request_timeout=30.0)
    try:
        adm = _adm(1000)
        pool = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 0)
        rec = usage.RequestRecord("chatcmpl-br", "chat", True)
        ok, h = _admit(app, pool, "br", 62_886, rec=rec)
        assert ok is True and h.sent is None
        assert [g["rows"] for g in mem.gates] == [4_886, 58_790], mem.gates
        assert rec.ws_rows == 58_790 and rec.ws_kind == "resident" and adm.held["br"] == 58_790 << 20, vars(rec)
        assert mem.over == []
    finally:
        app.STATE.clear()
    return "gate rows 4,886 then 58,790; reserved 58,790 MiB"


def test_srv773_a_shed_during_the_wait_is_charged_and_an_over_peak_admit_counted_once():
    """Alone and refused, the refused pass's shed counts in the over-peak admit's charge (32,886 rows, not 4,886).
    While it waits at the box's floor another slot's gate sheds the anchor it resumes at: the gate runs again on the
    58,790 rows before the reservation, refused again alone, and the admit is not counted a second time."""
    import threading
    import types
    app = _app()
    from server import usage
    app.STATE.clear()
    res = _Branch([58_000, 30_000, 4_096])
    mem = _RowsMem(peak=200 * GiB, refuse=lambda: True, res=res, on_gate=lambda i: res.shed() if i == 0 else None)
    app.STATE.update(mem=mem, resident=res, mem_load={"allocated": 0}, queue_timeout=1.0, request_timeout=30.0)
    box = {"avail": 40 * GiB}                    # 40 - 32.1 under the 15 GiB floor: it waits
    adm = slots.Admission(15 * GiB, meminfo=lambda: {"MemAvailable": box["avail"]}, growth=lambda: 0)
    try:
        pool = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 1)
        rec = usage.RequestRecord("chatcmpl-wt", "chat", True)
        rec.lock_acquired()

        def other_slot():
            time.sleep(1.2)
            res.shed()
            box["avail"] = 1000 * GiB
        threading.Thread(target=other_slot, daemon=True).start()
        ok, h = _admit(app, pool, "wt", 62_886, rec=rec)
        assert ok is True and h.sent is None and rec.wait_mem_ms >= 1000, rec.wait_mem_ms
        assert [g["rows"] for g in mem.gates] == [4_886, 58_790], mem.gates
        assert [(o[1], o[2]) for o in mem.over] == [(32_886, 32_886 << 20)], mem.over
        assert mem.ws_now == 58_790 << 20 and adm.held["wt"] == 58_790 << 20 and rec.ws_rows == 58_790
    finally:
        app.STATE.clear()
    return "over the peak once at 32,886 rows; re-gated and reserved at 58,790 after the wait's shed"


def test_srv773_the_served_prompts_end_is_held_after_the_routers_notes():
    """A conductor turn on a parallel server: the slot's rows are the served prompt, with the router's note in turn 1.
    `_par_admit` gates on the client's tokens, which part from the rows at that note, so its gate holds no end; the
    handler holds the end on the served prompt once the notes and images are in (`_hold_served_end`), and the
    prefill resumes at it."""
    import types
    import torch
    app = _app()
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from test_cache import tokens
    from test_end_anchor import _state, turn
    from server import usage
    res = _state()
    note = tokens(9, seed=93)
    client1 = tokens(60, seed=94)
    served1 = client1[:30] + note + client1[30:]
    out, _, _ = turn(served1, 6, "chat-n")
    new = tokens(7, seed=95)
    client2, served2 = client1 + out + new, served1 + out + new
    assert res.end is not None and not app._end_held(res, client2, len(client2), False)
    mem = _Mem(peak=200 * GiB, ws=GiB, res=res)
    app.STATE.update(mem=mem, mem_load={"allocated": 0}, queue_timeout=1.0, request_timeout=30.0)
    try:
        adm = _adm(100)
        pool = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 0)
        rec = usage.RequestRecord("chatcmpl-n", "chat", True)
        ok, h = _admit(app, pool, "n", len(client2), rec=rec)
        assert ok is True and [g["held"] for g in mem.gates] == [False] and not res.end_hold, mem.gates
        app._hold_served_end(res, torch.tensor(served2), None, None, len(served2))
        assert res.end_hold and res.shed_end(False) == 0 and res.end is not None
        e = int(res.end["at"])
        app.STATE["mem"] = None
        _, pre, _ = turn(served2, 4, "chat-n")
        assert pre["kind"] == "end" and pre["reused"] == e, pre
        assert not res.end_hold, "the prefill's begin lets it go"
    finally:
        app.STATE.clear()
    src = open(os.path.join(ROOT, "server", "app.py")).read()
    tail = src[src.index("prompt, in_think = plan.apply_prompt(prompt, STATE[\"tok\"], in_think)"):]
    at = tail.index("_hold_served_end(res_, prompt, mm, comp, n_prompt)")
    assert tail.index("prompt, mm = _mm_prompt(prompt, images, rec)") < at, "after the notes and the images"
    assert tail[:at].rstrip().endswith("if par is not None:"), "on a parallel server"
    return "gate on the client's tokens: no hold; on the served prompt: held until the prefill"


def test_the_release_route_answers_trusted_local_and_the_admin_only():
    import contextlib
    import io
    app = _app()
    from server import auth

    class Req(app.Handler):
        def __init__(self, body, headers):
            raw = json.dumps(body).encode()
            self.rfile, self.wfile = io.BytesIO(raw), io.BytesIO()
            self.headers = {"Content-Length": str(len(raw)), "Content-Type": "application/json",
                            **headers}
            self.path, self.command, self.request_version = ("/v1/cache/release", "POST",
                                                             "HTTP/1.1")
            self.requestline, self.client_address = "POST /v1/cache/release", ("127.0.0.1", 0)
            self.close_connection = True

    def call(body, headers=None):
        r = Req(body, headers or {})
        with contextlib.redirect_stdout(io.StringIO()):
            r.do_POST()
        head, _, raw = r.wfile.getvalue().partition(b"\r\n\r\n")
        return int(head.split()[1]), json.loads(raw)

    asked = []
    old = app._memory_release, dict(app.STATE)
    app._memory_release = lambda want, why="sandbox": asked.append((want, why)) or {
        "asked_bytes": want, "released_bytes": 0, "levels": {}, "mem_available_bytes": None}
    try:
        app.STATE["auth"] = auth.Auth("adm-" + "5e7u9" * 10, None)
        code, out = call({"bytes": 2 * GiB, "why": "sandbox"})          # chatd on the box
        assert code == 200 and out["asked_bytes"] == 2 * GiB and asked == [(2 * GiB, "sandbox")]
        code, _ = call({"bytes": GiB}, {"X-Forwarded-For": "203.0.113.4",
                                        "X-Real-IP": "203.0.113.4"})
        assert code == 401 and len(asked) == 1, "through the reverse proxy: the admin token"
        for bad in ({"bytes": -1}, {"bytes": "x"}, {"bytes": 1 << 41}):
            assert call(bad)[0] == 400
        assert len(asked) == 1
    finally:
        app._memory_release = old[0]
        app.STATE.clear()
        app.STATE.update(old[1])


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"ok  {name}")
    print(f"{n} passed")
