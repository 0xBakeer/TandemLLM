"""SRV-801, CPU only: a turn waits for the memory its own prefill takes, and says so in the numbers of its check.

A user's box on 0.15.3 (2026-11-09) held turns of a few hundred new tokens for minutes with 11 to 15 GiB free. The
state is replayed here on a fake MemAvailable, with Flash-Next's workspace model and the KV row size of that log
(6,515,851,264 bytes for 214,280 rows), through the server's own admission (`Handler._par_admit`) and guard:

  * a turn of the conversation slot 0 holds (212,938 client tokens, 449 rows forwarded) was priced as a cold prefill
    of the whole prompt (`ws_rows=212938 ws_gib=10.75`) and waited 148.6 s. Slot 0 had no share of its own
    conversation in the price (`_slot_share`); the client's tokens part from its rows at the first span the server
    put there, so where it resumes in place said nothing;
  * the floor a prefill's pages commit under held a whole 4,096-row chunk's workspace (4.4 GiB) for every turn. A
    turn that read 180 new tokens after its chat came back from the NVMe was held under 9.7 GiB with 11 free, where
    its own chunk is 1.2 GiB (`Admission.live_guard`). A prefill found to forward more than its price is counted
    at what it forwards from its first chunk on (`Admission.forwards`);
  * the wait retried the pool's reclaim every second and printed each hand-back that freed nothing, eight lines a
    second (`_empty_line`);
  * the live view named the pages of that turn as its need without saying so, beside a wait line with other numbers
    (`Admission.terms`, server/activity.py `memory_lines`).

The floor of the whole request (chunk, KV commit, sandbox, margin) is as it was: a turn that maps 5.7 GiB of pages and
forwards 3,300 rows at 15.3 GiB free is 0.2 GiB short of it and still waits.

Slot 0's share is counted on the client's copy of the last prompt. The rows hold the served prompt, with the spans the
server put there, so a regenerate or an edit shares the copy further than it shares the rows: priced on the copy it was
charged 1 row and 488 rows and forwarded 4,197 and 3,968 from the last chunk line below the place the rows part. Those
turns are priced on the rows (`_rows_forward`), and before any prefill's first forward its real rows are held against
its price: the reserve is raised, or the lane gives way before the forward allocates (`_first_forward`).

Run: python tests/test_memory_wait.py
"""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import sys
import threading
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from server import slots  # noqa: E402
from test_kvpool_admission import _Clock, _FNMem, _H, _HomeRes, _app  # noqa: E402

GiB, MiB = slots.GiB, slots.MiB

# the turn that waited 148.6 s: its prompt's client tokens and the rows its prefill forwarded
N, FWD = 212_938, 449
# a KV row, from the log's reclaim of slot 0's idle conversation
PER_ROW = 6_515_851_264 // 214_280
# MemAvailable and the KV pages of the lanes in flight on the wait line of that minute
AVAIL, IN_FLIGHT = 15.3, 0.4
# the turn of the screenshot: 180 new tokens of a 143,590-token chat read back from the NVMe, 11 GiB free
N_SHOT, FWD_SHOT = 143_590, 180
MARGIN, KV_STEP = 5 * GiB, 384 * MiB
# a chunk of the prefill, the row the rows part from a regenerate's copy at (the first span the client's copy lacks
# near the prompt's end) and the last chunk line below it, where its prefill resumes
CHUNK = 4_096
PART, LINE = 203_414, 49 * CHUNK


class _Pages:
    """A slot's paged KV as the admission and the guard read it (engine/kvpool.py `PagedSet`), `rows` mapped."""

    def __init__(self, rows: int = 0, owner: str = "slot0"):
        self.rows, self.owner = int(rows), owner
        self.bufs = [types.SimpleNamespace(outer=48, row_bytes=1024, page=2 << 20)]

    def margin(self) -> int:
        return 256

    def need_bytes(self, rows: int) -> int:
        return max(0, int(rows) - self.rows) * PER_ROW

    def trim(self, rows: int = 0) -> int:
        self.rows = min(self.rows, int(rows))
        return 0


def _engine(rows: int = 0, owner: str = "slot0"):
    return types.SimpleNamespace(kv=types.SimpleNamespace(pages=_Pages(rows, owner), max_len=262_144, length=rows))


def _wire(app, adm, kp) -> None:
    """The guards as the server wires them (`_wire_guards`). 0.15.3 wired them in its start-up, with these lines:
    kept here so a test of the charge fails there on the charge."""
    if hasattr(app, "_wire_guards"):
        return app._wire_guards(adm, kp)
    adm.guard = lambda rows: app._guard(rows, adm.cap)
    kp.guard = lambda: (app._guard(adm.ctx_rows(), kp.hard_floor_cfg) or (None,))[0]
    kp.guard_at = lambda rows: (app._guard(max(int(rows), adm.ctx_rows()), kp.hard_floor_cfg) or (None,))[0]
    adm.pool_floor = getattr(kp, "floor_at", None) or (lambda rows: kp.hard_floor)


@contextlib.contextmanager
def _box(app, avail_gib: float, in_flight_gib: float = IN_FLIGHT, clock=None, **state):
    """The server's state for one admission: MemAvailable `avail_gib`, a 15 GiB admission floor and a 10 GiB pool
    floor as configured, both reading the live guard (`_wire_guards`), Flash-Next's workspace model, the engine at
    54.4 GiB under a 75.82 GiB peak, and the KV pages of the lanes in flight. The clock is the test's, or `clock`."""
    box = {"avail": int(avail_gib * GiB)}
    adm = slots.Admission(15 * GiB, meminfo=lambda: {"MemAvailable": box["avail"]}, growth=lambda: 0)
    kp = slots.GuardedPool(hard_floor=10 * GiB, margin_rows=0, meminfo=lambda: box["avail"])
    mem = _FNMem(peak=int(75.82 * GiB), ws=0)
    pool = types.SimpleNamespace(admission=adm, turn=None, burst=None, running=lambda: 1,
                                 slots=[types.SimpleNamespace(phase="", prefill_left=None) for _ in range(8)])
    st = dict(mem=mem, mem_load={"allocated": int(54.4 * GiB)}, prefix_chunk=4096, engine=_engine(),
              queue_timeout=1.0, request_timeout=30.0)
    st.update(state)
    old_time, old_pool = app.time, app.kvpool.POOL
    app.STATE.clear()
    app.STATE.update(st)
    app.kvpool.POOL = kp
    app.time = clock or _Clock()
    _wire(app, adm, kp)
    if in_flight_gib:
        kv = int(in_flight_gib * GiB)
        assert adm.reserve("in-flight", kv, kv=kv, owner="slot3")[0]
    try:
        yield types.SimpleNamespace(box=box, adm=adm, kp=kp, mem=mem, pool=pool)
    finally:
        slots.bind(None)
        app.time, app.kvpool.POOL = old_time, old_pool
        app.STATE.clear()


def _admit(app, b, cid: str, n, home: int = 0, max_new: int = 8192, slot=None, limit_s: float = 2.0,
           home_rows: bool = False, stream: bool = True):
    """`_par_admit` of a prompt (`n` tokens, or its ids) on slot 0 or `slot`, its wait bounded at `limit_s` on the
    test's clock: (admitted, its record, the engine log's lines). `home_rows`: priced on the rows, where the server
    has that (`_slot_share`'s answer for a regenerate or an edit)."""
    import torch
    from server import usage
    rec = usage.RequestRecord(f"chatcmpl-{cid}", "chat", True)
    out = io.StringIO()
    prompt = torch.zeros(n, dtype=torch.long) if isinstance(n, int) else torch.tensor(n, dtype=torch.long)
    kw = {"home_rows": True} if home_rows and "home_rows" in inspect.signature(app.Handler._par_admit).parameters \
        else {}
    slots.bind(slot)
    try:
        with contextlib.redirect_stdout(out):
            ok = app.Handler._par_admit(_H(), b.pool, cid, prompt, rec=rec, stream=stream,
                                        max_new=max_new, home=home, limit_s=limit_s, **kw)
    finally:
        slots.bind(None)
    return ok, rec, out.getvalue()


def _side(res, index: int = 1):
    return slots.Slot(index, 262_144, values={"resident": res, "engine": _engine(0, f"slot{index}")})


def _from_nvme(n: int, fwd: int):
    """A side slot's resident whose conversation the pool's reclaim wrote to the NVMe: the entry serves all but the
    last `fwd` rows of an `n`-token prompt (`SessionSpill.candidate`), and no page of it is mapped."""
    res = _HomeRes(0)
    res.spiller = types.SimpleNamespace(candidate=lambda r, keys, max_len, replay, mm_ok=False: ("entry", n - fwd, 0))
    return res


def _slot0(view_end: int = N - FWD):
    """Slot 0 holding the conversation: the client's tokens of this turn, the last prompt whose rows the slot kept
    (all of them up to `view_end`, then its generation prompt's four tokens) and the rows in place."""
    ids = [1] * N
    view = ids[:view_end] + [2] * 4
    res = _HomeRes(view_end + 31)
    return ids, dict(resident=res, slot0_view=view, engine=_engine(res.valid))


def _rows(tokens: list, end_at: int, hidden: list, lines: int = LINE):
    """Slot 0's rows as the admission asks them: `ResidentPrefix`'s own host-side answers (`common`, `best`,
    `end_offer`, `resume_estimate`, `replay`) on a served prompt of `tokens`, its end anchor at `end_at` with the
    spans the server put there, and an anchor at every chunk line up to `lines`. No engine and no buffers."""
    from engine import cache

    class Rows(cache.ResidentPrefix):
        def __init__(self):
            pass

        def clear(self):
            self.tokens, self.valid, self.anchors, self.end = [], 0, {}, None

    res = Rows()
    res.tokens, res.valid, res.chunk = list(tokens), len(tokens), CHUNK
    res.anchors = {k: () for k in range(CHUNK, lines + 1, CHUNK)}
    res.end = {"at": int(end_at), "hidden": list(hidden), "mark": None}
    res.pend = res.grid = res.spiller = res._stash = res.end_lost = None
    res.glue, res.owner, res.end_hold, res.stats = set(), "chat", False, {"end_replayed": 0}
    return res


def _chat(n: int, spans=((PART, 455),), answer: int = 16):
    """Slot 0 after a turn of `n` client tokens: the client's copy of that prompt (the view), and the rows of the
    prompt as it was served: the copy with the server's `spans` (row, length) in it, then the answer's rows. The end
    anchor is the served prompt's end. (the view, the rows, the server's state)"""
    view = [1] * (n - 4) + [2] * 4
    rows, hidden, at = [], [], 0
    for p, k in spans:
        take = p - len(rows)
        rows += view[at:at + take]
        at += take
        hidden.append((p, [7] * k))
        rows += [7] * k
    rows += view[at:]
    res = _rows(rows + [3] * answer, len(rows), hidden)
    return view, res, dict(resident=res, slot0_view=view, engine=_engine(res.valid))


def _priced(app, b, cid: str, ids: list, conv: int = 5_000):
    """A turn of `ids` on slot 0 as the handler admits it: its share (`_slot_share`), then `_par_admit`.
    (admitted, its record, the workspace its reserve holds, the log) In 0.15.3 slot 0 had no share."""
    why = {"ids": ids, "conv": conv}
    home = app._slot_share(why, None) if hasattr(app, "_slot_share") else 0
    ok, rec, log = _admit(app, b, cid, ids, home=home, home_rows=bool(why.get("home_rows")))
    ws = b.adm.held[cid] - b.adm.kv[cid][0] if cid in b.adm.held else None
    return ok, rec, ws, log


def test_a_turn_of_the_chat_slot_0_holds_is_charged_the_rows_it_reads():
    """The 148.6 s wait. Priced cold, as 0.15.3 priced it, the turn takes 9.4 GiB of workspace at the floor and waits
    at 15.3 GiB free; with slot 0's share it is charged the 449 rows past the slot's (1.75 GiB) and is admitted, and
    from 8.0 GiB free. The floor is not lower: at 7.9 GiB it waits."""
    app = _app()
    ids, st = _slot0()
    with _box(app, AVAIL, **st) as b:
        home = app._slot_share({"ids": ids, "conv": 5_000}, None)
        assert home == N - FWD, f"slot 0's share of its own conversation: {home:,} of {N:,} tokens"
        ok, rec, log = _admit(app, b, "9555e", N, home=home)
        assert (rec.ws_rows, rec.ws_kind) == (FWD, "home") and abs(rec.ws_gib - 1.75) < 0.01, \
            f"charged {rec.ws_rows:,} rows ({rec.ws_gib:.2f} GiB, {rec.ws_kind}), forwards {FWD}"
        assert ok is True and "waits for memory" not in log, log
        held = b.adm.held["9555e"]
        assert abs(held / GiB - (1.746 + 0.478)) < 0.01, held / GiB
    # the same turn priced cold: what 0.15.3 charged it, and why it waited
    with _box(app, AVAIL, **st) as b:
        ok, rec, log = _admit(app, b, "cold", N, home=0)
        assert rec.ws_rows == N and rec.ws_kind is None and abs(rec.ws_gib - 10.75) < 0.01, vars(rec)
        assert ok is False and rec.stop_detail == "memory_wait_timeout", (ok, rec.stop_detail)
        cold_line = ("cold waits for memory: MemAvailable 15.3 GiB - cache growth 0.0 - in flight 0.4 - this request "
                     "9.9 = 5.0 GiB (< floor 5.4 = chunk 0.0 (this request's 4.4 counted above) + KV commit 0.4 + "
                     "sandbox 0.0 + margin 5.0, at most 15.0); KV pages: this request 0.5, in flight 0.4")
        assert cold_line in log, log
    for avail, want in ((8.1, True), (7.9, False)):
        with _box(app, avail, **st) as b:
            ok, rec, log = _admit(app, b, "edge", N, home=home)
            assert ok is want, (avail, log)
            if not want:
                assert rec.mem_wait["check"] == "request" and rec.mem_wait["guard"] == MARGIN + KV_STEP, rec.mem_wait
    return f"449 rows: {held / GiB:.2f} GiB at the floor, admitted at 15.3 and from 8.0 GiB; cold 9.9, waits"


def test_slot_0s_share_counts_the_last_answer_as_the_client_sends_it_back():
    """The view is the last prompt's client tokens, so the share ends before the last answer: a turn whose client
    sends 2,000 tokens of answer back is charged 2,449 rows (3.6 GiB), more than it forwards, never the prompt."""
    app = _app()
    ids, st = _slot0(N - FWD - 2_000)
    with _box(app, AVAIL, **st) as b:
        home = app._slot_share({"ids": ids, "conv": 5_000}, None)
        ok, rec, log = _admit(app, b, "answer", N, home=home)
        assert ok is True and rec.ws_rows == FWD + 2_000 and rec.ws_kind == "home", (vars(rec), log)
        assert abs(rec.ws_gib - 3.615) < 0.01, rec.ws_gib
    return f"{rec.ws_rows:,} rows, {rec.ws_gib:.2f} GiB"


def test_slot_0_has_a_share_only_of_the_conversation_it_holds():
    """No view, another conversation, a view under one chunk, a share that ends in the client's head, the rows' test
    alone (`SIDE_CONTINUE=resident`): no share, a cold price as before. A block the client moved to its last message
    is read past (`end_align`). The share is at most the rows in place: a slot a reclaim emptied is cold."""
    app = _app()
    ids, st = _slot0()
    view = st["slot0_view"]
    why = {"ids": ids, "conv": 5_000}
    with _box(app, AVAIL, **st) as b:
        assert app._slot_share(why, None) == N - FWD
        assert app._slot_share({"conv": 5_000}, None) == 0, "no client tokens"
        assert app._slot_share({"ids": [1] * 3 + [7] * 1_000, "conv": 3}, None) == 0, "another conversation"
        assert app._slot_share({"ids": ids, "conv": N}, None) == 0, "the shared part ends in the client's head"
        old = app.SIDE_CONTINUE
        app.SIDE_CONTINUE = "resident"
        try:
            assert app._slot_share(why, None) == 0
        finally:
            app.SIDE_CONTINUE = old
        app.STATE["slot0_view"] = view[:2_000]
        assert app._slot_share({"ids": view[:2_000] + [1] * 50, "conv": 100}, None) == 0, "a view under one chunk"
        app.STATE["slot0_view"] = None
        assert app._slot_share(why, None) == 0, "no view"
        # a block that moved to the client's last message: the tokens part at 6,000 and meet again past it
        app.STATE["slot0_view"] = [1] * 6_000 + [3] * 146_000
        assert app._slot_share(why, None) == 0
        app.STATE["end_align"] = types.SimpleNamespace(moved_common=lambda v, i: 150_000)
        assert app._slot_share(why, None) == 150_000
        # a side slot's share is `_side_home`'s, on that slot only (SRV-787)
        side = types.SimpleNamespace(index=2)
        assert app._slot_share({"home": 2, "home_share": 57_823, "ids": ids}, side) == 57_823
        assert app._slot_share({"home": 1, "home_share": 57_823, "ids": ids}, side) == 0
    for valid, rows, kind in ((100_000, N - 100_000, "home"), (0, N, None)):
        with _box(app, 40.0, **dict(st, resident=_HomeRes(valid), engine=_engine(valid))) as b:
            ok, rec, _ = _admit(app, b, "valid", N, home=N - FWD)
            assert ok is True and (rec.ws_rows, rec.ws_kind) == (rows, kind), (valid, vars(rec))
    return "no view, another chat, a short view, the client's head: 0; never past the rows in place"


def test_the_pools_floor_holds_the_chunk_the_turn_forwards():
    """The screenshot's turn: 180 new tokens of a 143,590-token chat read back from the NVMe, so its prompt's pages
    (4.3 GiB) are all to map. 0.15.3 held a whole 4,096-row chunk in the floor those pages commit under (9.7 GiB): it
    needed 14.0 GiB free. Its own chunk is 1.2 GiB: the floor is 6.5, it is admitted at 11.2 GiB free, and the pool's
    gate takes every page it was admitted for with MemAvailable never under the margin. A prefill whose forwarded
    rows are not known, a cold prompt, or a whole chunk in flight beside it: 9.7 as before."""
    app = _app()
    from engine import kvpool
    n, fwd = N_SHOT, FWD_SHOT
    kv_prompt, kv = (n + 8_256) * PER_ROW, (n + 16_448) * PER_ROW
    with _box(app, 11.2, in_flight_gib=0) as b:
        ws = b.mem.model(n, fwd, line=False)
        chunk = b.mem.model(n, 4_096, line=False)
        assert abs(ws / GiB - 1.172) < 0.01 and abs(chunk / GiB - 4.357) < 0.01 and abs(kv_prompt / GiB - 4.3) < 0.01
        assert b.kp.floor_at(n, fwd) == ws + KV_STEP + MARGIN, b.kp.floor_at(n, fwd) / GiB
        whole = chunk + KV_STEP + MARGIN
        assert b.kp.floor_at(n) == b.kp.floor_at(n, n) == b.kp.floor_at(n, 4_096) == whole, "unknown or cold: whole"
        assert abs(whole / GiB - 9.73) < 0.01 and b.kp.hard_floor == KV_STEP + MARGIN, "idle: no chunk"
        ok, rec, log = _admit(app, b, "shot", n, slot=_side(_from_nvme(n, fwd)))
        assert (rec.ws_rows, rec.ws_kind) == (fwd, "spill"), vars(rec)
        assert ok is True and "waits for memory" not in log, \
            f"waits at 11.2 GiB free under a floor of {whole / GiB:.1f}: {log}"
        # its prefill runs: the gate's floor is the admission's, and every page it was admitted for commits
        assert b.kp.hard_floor == ws + KV_STEP + MARGIN
        low = b.box["avail"]
        try:
            for _ in range(kv_prompt // (4 * MiB)):
                b.kp.gate("slot1", 4 * MiB)
                b.box["avail"] -= 4 * MiB
            low = b.box["avail"] - ws                      # its one chunk's workspace, mapped and given back
            b.adm.prefilled("shot")
            for _ in range((kv - kv_prompt) // (4 * MiB)):
                b.kp.gate("slot1", 4 * MiB)
                b.box["avail"] -= 4 * MiB
        except kvpool.PoolFull as exc:
            raise AssertionError(f"the gate refused pages the admission counted: {exc}") from None
        assert low >= MARGIN and b.box["avail"] >= MARGIN + KV_STEP, (low / GiB, b.box["avail"] / GiB)
    # under 11.08 GiB the whole request is short of the admission's floor, which is as it was
    with _box(app, 10.9, in_flight_gib=0) as b:
        ok, rec, log = _admit(app, b, "shot", n, slot=_side(_from_nvme(n, fwd)))
        m = rec.mem_wait
        assert ok is False and m["check"] == "request" and m["guard"] == MARGIN + KV_STEP, (ok, m)
        assert abs(m["need"] / GiB - 5.70) < 0.01 and abs(m["short"] / GiB - 0.18) < 0.01, m
    # a whole chunk in flight on another slot: the floor holds it, for this turn too
    with _box(app, 30.0, in_flight_gib=0) as b:
        assert b.adm.reserve("long", 8 * GiB, rows=200_000)[0]
        assert b.kp.floor_at(n, fwd) == b.kp.hard_floor == b.mem.model(200_000, 4_096, line=False) + KV_STEP + MARGIN
    return (f"floor {(ws + KV_STEP + MARGIN) / GiB:.2f} GiB (was {whole / GiB:.2f}); admitted at 11.2 GiB, lowest "
            f"MemAvailable in the prefill {low / GiB:.2f}")


def test_a_prefill_that_forwards_more_than_its_price_is_counted_at_what_it_forwards():
    """The price is the admission's estimate. A read-back that fails forwards the whole prompt: at its first chunk
    the prefill says where it started (`_par_chunks`), and the floor holds a whole chunk from there. One that
    forwards fewer rows than its price keeps the price; a request whose prefill has run is not counted."""
    app = _app()
    n, fwd = N_SHOT, FWD_SHOT
    with _box(app, 30.0, in_flight_gib=0) as b:
        ws, chunk = b.mem.model(n, fwd, line=False), b.mem.model(n, 4_096, line=False)
        cid = "chatcmpl-shot"
        assert b.adm.reserve(cid, ws, rows=n, fwd=fwd)[0] and b.kp.hard_floor == ws + KV_STEP + MARGIN
        app.STATE["pool"] = b.pool
        where: dict = {}
        hook = app._par_chunks(None, where, cid)
        where["start"] = 0                              # the prefill found nothing to resume
        hook(4_096, n)
        assert b.adm.fwd[cid] == n and b.kp.hard_floor == chunk + KV_STEP + MARGIN, b.kp.hard_floor / GiB
        assert b.adm.floor_at()[0] == chunk + KV_STEP + MARGIN
        where["start"] = n - 100
        hook(8_192, n)
        assert b.adm.fwd[cid] == n, "told once"
        b.adm.forwards(cid, 50)
        assert b.adm.fwd[cid] == n, "never under what it was counted at"
        b.adm.prefilled(cid)
        b.adm.forwards(cid, n)
        assert cid not in b.adm.fwd and b.kp.hard_floor == KV_STEP + MARGIN
        assert b.adm.reserve("whole", ws, rows=n)[0]
        b.adm.forwards("whole", 10)
        assert "whole" not in b.adm.fwd and b.kp.hard_floor == chunk + KV_STEP + MARGIN, "unknown stays a whole chunk"
    priced, real = (ws + KV_STEP + MARGIN) / GiB, (chunk + KV_STEP + MARGIN) / GiB
    return f"{priced:.2f} GiB as priced, {real:.2f} once it reads all"


def test_the_floor_of_the_whole_request_is_as_it_was():
    """The wait line of that minute: a turn that maps 5.7 GiB of pages and forwards about 3,300 rows (4.0 GiB) at
    15.3 GiB free and 0.4 in flight is 0.2 GiB short of 5.4 (chunk, KV commit, margin; no constant changed) and
    waits. Its pages' check, 0.3 short of 9.7 on 0.15.3, holds its own chunk now and is not what holds it."""
    app = _app()
    n, fwd = 186_000, 3_300
    with _box(app, AVAIL) as b:
        ok, rec, log = _admit(app, b, "541dc", n, slot=_side(_from_nvme(n, fwd)))
        m = rec.mem_wait
        assert ok is False and m["check"] == "request", (ok, m)
        assert "541dc waits for memory: MemAvailable 15.3 GiB - cache growth 0.0 - in flight 0.4 - this request 9.7 " \
               "= 5.2 GiB (< floor 5.4 = chunk 0.0 (this request's 4.0 counted above) + KV commit 0.4 + sandbox 0.0 " \
               "+ margin 5.0, at most 15.0); KV pages: this request 5.7, in flight 0.4" in log, log
        assert m["guard"] == MARGIN + KV_STEP and abs(m["short"] / GiB - 0.21) < 0.02, m
    with _box(app, 15.6) as b:
        assert _admit(app, b, "541dc", n, slot=_side(_from_nvme(n, fwd)))[0] is True
    return f"short {m['short'] / GiB:.2f} GiB of {m['guard'] / GiB:.2f} at 15.3 GiB free; admitted at 15.6"


def test_a_wait_says_its_own_checks_numbers():
    """What the user reads is the check that holds the turn, term by term: free - the others - this turn against
    what must stay free, the numbers of the engine log's wait line. The screenshot's sentence ("needs about 4.3 GiB
    to read 180 new tokens; 11 GiB is free and 9.7 must stay free for the system") was the pages' check named as
    the turn's need; it now says so. The block is the live contract's `memory`."""
    app = _app()
    from server import activity
    n, fwd = 186_000, 3_300
    with _box(app, AVAIL) as b:
        ok, rec, log = _admit(app, b, "541dc", n, slot=_side(_from_nvme(n, fwd)))
        m, d = rec.mem_wait, b.adm.last_refused
    t = slots.Admission.terms(d)
    assert (m["free"], m["others"], m["need"], m["guard"]) == (t["avail"], t["others"], t["mine"], t["floor"])
    assert m["free"] - m["others"] - m["need"] == d["after"] and m["guard"] - d["after"] == m["short"], m
    assert m["others"] == int(IN_FLIGHT * GiB) and m["need"] == d["need"], (m, d)
    out = activity.memory_of(m, m["t0"] + 1.0)
    assert out["lines"][0] == ("This turn needs about 9.7 GiB to read 3,300 new tokens; 15 GiB is free, other turns "
                               "and the caches hold 0.4 of it, and 5.4 must stay free for the system"), out["lines"]
    assert (out["need_gib"], out["others_gib"], out["check"]) == (9.73, 0.4, "request"), out
    # the pages' check, as it held the screenshot's turn on 0.15.3 (a whole chunk in its floor)
    with _box(app, 11.0, in_flight_gib=0) as b:
        ws, kv = b.mem.model(N_SHOT, FWD_SHOT, line=False), (N_SHOT + 16_448) * PER_ROW
        ok, d = b.adm.check(ws + kv, kv=kv, rows=N_SHOT, kv_prompt=(N_SHOT + 8_256) * PER_ROW)
        t = slots.Admission.terms(d)
        assert not ok and t["check"] == "prefill" and t["others"] == 0, t
        m2 = app._mem_wait_of(d, FWD_SHOT, 0.0, 60.0, 1, 0)
    line = activity.memory_lines(m2, 1.0, 59.0)[0]
    assert line == ("This turn needs about 4.3 GiB for its pages to read 180 new tokens; 11 GiB is free and 9.7 must "
                    "stay free while it reads"), line
    assert "its prefill's pages: MemAvailable 11.0 - growth and in flight 0.0 - its KV pages 4.3 = 6.7 GiB (< the " \
           "KV pool's floor 9.7 while its prefill runs)" in slots.Admission.describe(d)
    words = activity.refusal_words(m)
    assert "(needed 9.7 GiB, 9.5 free above the guard)" in words, words

    def block(node):
        if isinstance(node, dict):
            p = node.get("properties")
            if isinstance(p, dict) and "need_gib" in p and "lines" in p:
                return node
            node = list(node.values())
        for v in node if isinstance(node, list) else ():
            got = block(v)
            if got is not None:
                return got
        return None
    schema = block(json.load(open(os.path.join(ROOT, "docs", "contract", "dashboard-v1", "live.schema.json"))))
    assert schema["additionalProperties"] is False and set(out) <= set(schema["properties"]), \
        sorted(set(out) - set(schema["properties"]))
    try:
        import jsonschema
    except ImportError:
        return f"{out['lines'][0]} (jsonschema not installed: keys only)"
    jsonschema.validate(out, schema)
    return out["lines"][0]


def test_a_retry_that_frees_nothing_prints_once_then_a_count():
    """The wait retried the pool's reclaim every second, eight hand-backs each, and every one printed
    `[memory] reserve back to the driver: 0.00 GiB in 0.37 ms (kv_pool)`: 88 s, about 700 lines. Of a run that frees
    nothing the first has its line, the next are counted and named once a minute, and one that frees memory always
    has its line, with the count before it. Each cause has its own run."""
    app = _app()
    app._EMPTY_RUN.clear()
    ev = {"kind": "empty", "why": "kv_pool", "freed": 0, "ms": 0.37}
    lines = [app._empty_line(ev, now=1000.0 + i / 8) for i in range(88 * 8)]
    said = [x for x in lines if x]
    assert lines[0] == "[memory] reserve back to the driver: 0.00 GiB in 0.37 ms (kv_pool)", lines[0]
    assert said[1:] == ["[memory] reserve back to the driver: 0.00 GiB, 480 more times in 177.6 ms (kv_pool)"], said
    assert app._empty_line(dict(ev, why="gate"), now=1088.0) == \
        "[memory] reserve back to the driver: 0.00 GiB in 0.37 ms (gate)", "another cause: its own run"
    got = app._empty_line(dict(ev, freed=3 * GiB, ms=41.0), now=1088.0)
    assert got == ("[memory] reserve back to the driver: 3.00 GiB in 41.0 ms (kv_pool); 223 more before it freed "
                   "nothing (82.5 ms)"), got
    assert app._empty_line(dict(ev, freed=GiB, ms=9.0), now=1089.0) == \
        "[memory] reserve back to the driver: 1.00 GiB in 9.0 ms (kv_pool)"
    assert app._empty_line(ev, now=1090.0) == lines[0], "a new run starts with its line"
    app._EMPTY_RUN.clear()
    return f"{len(lines)} hand-backs that freed nothing in 88 s: {len(said)} lines"


def test_a_regenerate_is_charged_the_rows_past_the_line_the_rows_part_under():
    """`t3r`: the last prompt sent again. The view is that prompt, so the share was all of it and the price 1 row
    (0.79 GiB); the rows hold a span of the server's 1,487 rows before the prompt's end, the prefill resumes at the
    chunk line below it (row 200,704) and forwards 4,197. Priced on the rows it is charged those, and its reserve
    holds their workspace by the price's formula."""
    app = _app()
    n = LINE + 4_197
    view, res, st = _chat(n)
    with _box(app, AVAIL, **st) as b:
        ok, rec, ws, log = _priced(app, b, "t3r", list(view))
        assert rec.ws_rows == 4_197, \
            f"charged {rec.ws_rows:,} rows ({rec.ws_gib:.2f} GiB, {rec.ws_kind}) for a regenerate that forwards 4,197"
        assert ok is True and rec.ws_kind == "resident" and ws == b.mem.model(n, 4_197, line=False), (vars(rec), log)
        assert res.stats["end_replayed"] == 0, "asking the rows counts nothing"
    one = b.mem.model(n, 1, line=False) / GiB
    return f"4,197 rows: {ws / GiB:.2f} GiB of workspace at the floor (1 row: {one:.2f})"


def test_a_regenerate_whose_copy_the_alignment_serves_is_charged_from_the_line_it_serves_to():
    """The same turn as the box served it: a span of the server's early in the prompt too, so the client's tokens
    part from the rows at row 3,000 and resume nowhere by themselves. The alignment serves the rows up to the last
    line under the place they part again and the copy from there (`EndAlign.served`): 4,197 rows forwarded, and the
    price is the client's tokens less those. With no alignment to ask, the rows serve nothing: priced cold."""
    app = _app()
    n = LINE + 4_197 - 303
    view, res, st = _chat(n, spans=((3_000, 303), (PART, 455)))
    align = types.SimpleNamespace(moved_common=lambda v, i: i,
                                  served=lambda r, ids: r.tokens[:LINE] + list(ids[LINE - 303:]))
    with _box(app, AVAIL, end_align=align, **st) as b:
        ok, rec, ws, log = _priced(app, b, "t3r", list(view))
        assert rec.ws_rows == 4_197, \
            f"charged {rec.ws_rows:,} rows ({rec.ws_gib:.2f} GiB, {rec.ws_kind}) for a regenerate that forwards 4,197"
        assert ok is True and rec.ws_kind == "rows" and ws == b.mem.model(n, 4_197, line=False), (vars(rec), log)
    with _box(app, 40.0, **st) as b:
        ok, rec, _, log = _priced(app, b, "none", list(view))
        assert ok is True and rec.ws_rows == n and rec.ws_kind is None, (vars(rec), log)
    return f"4,197 rows by the alignment's answer ({ws / GiB:.2f} GiB); the whole prompt with none"


def test_an_edit_of_the_last_message_is_charged_the_rows_past_that_line():
    """`t3e`: the last message edited. The copy shares the view up to that message, 488 rows before the end (1.83
    GiB); the rows part from it at the same span as a regenerate's, and the prefill forwards 3,968 from the line."""
    app = _app()
    n = LINE + 3_968
    view, res, st = _chat(n)
    ids = view[:n - 488] + [5] * 484 + [2] * 4
    with _box(app, AVAIL, **st) as b:
        ok, rec, ws, log = _priced(app, b, "t3e", ids)
        assert rec.ws_rows == 3_968, \
            f"charged {rec.ws_rows:,} rows ({rec.ws_gib:.2f} GiB, {rec.ws_kind}) for an edit that forwards 3,968"
        assert ok is True and ws == b.mem.model(n, 3_968, line=False), (vars(rec), log)
    return f"3,968 rows: {ws / GiB:.2f} GiB (488 rows: {b.mem.model(n, 488, line=False) / GiB:.2f})"


def test_a_regenerate_after_several_turns_is_charged_more_than_a_chunk():
    """`t5`: five turns on, each with its span in the rows and no anchor past row 200,704: the regenerate forwards
    7,281 rows, more than one chunk. Its price is those rows and the pool's floor holds a whole chunk for it."""
    app = _app()
    n = LINE + 7_281
    view, res, st = _chat(n, spans=((PART, 455), (204_950, 420), (206_010, 380), (207_100, 300)))
    with _box(app, AVAIL, **st) as b:
        ok, rec, ws, log = _priced(app, b, "t5", list(view))
        assert rec.ws_rows == 7_281, \
            f"charged {rec.ws_rows:,} rows ({rec.ws_gib:.2f} GiB, {rec.ws_kind}) for a regenerate that forwards 7,281"
        assert ok is True and ws == b.mem.model(n, 7_281, line=False), (vars(rec), log)
        assert b.adm.fwd["t5"] == 7_281 and b.kp.hard_floor == b.mem.model(n, CHUNK, line=False) + KV_STEP + MARGIN
    return f"7,281 rows: {ws / GiB:.2f} GiB, a whole chunk in the pool's floor"


def test_the_next_turn_of_slot_0s_chat_is_charged_its_new_rows_not_the_prompt():
    """`t3`, the short follow-up: 199,651 client tokens, 508 of them past the last prompt. The client's tokens part
    from the rows at the first span the server put there (row 3,000), so 0.15.3 priced the whole prompt: 199,651 rows,
    10.75 GiB, for a turn that forwarded 998. It is charged the 508 rows past its share, as before this change: the
    next turn is served from the end anchor and is not priced on the rows again."""
    app = _app()
    n, new = 199_651, 508
    view, res, st = _chat(n - new, spans=((3_000, 305), (150_000, 4_000)), answer=300)
    assert res.end["at"] == 203_448
    ids = view + [4] * (new - 4) + [2] * 4
    with _box(app, AVAIL, **st) as b:
        ok, rec, ws, log = _priced(app, b, "t3", ids)
        assert rec.ws_rows == new, \
            f"charged {rec.ws_rows:,} rows ({rec.ws_gib:.2f} GiB) for a turn that forwards 998"
        assert ok is True and rec.ws_kind == "home" and "waits for memory" not in log, (vars(rec), log)
        assert ws == b.mem.model(n, new, line=False)
        # its prefill, as the box served it: 204,446 rows with the server's spans, resumed at the end anchor
        app.STATE["pool"] = b.pool
        out = io.StringIO()
        with contextlib.redirect_stdout(out), app._forward_ask("t3", 204_446):
            assert app._first_forward(None, None, 203_448, False) is None
        up = b.mem.model(204_446, 998, line=False)
        assert b.adm.held["t3"] - b.adm.kv["t3"][0] == up > ws, "the reserve holds the rows it forwards"
        assert f"[memory] t3 forwards 998 rows, priced at 508 rows: its workspace {ws / GiB:.2f} -> {up / GiB:.2f} " \
               f"GiB, reserved" in out.getvalue(), out.getvalue()
    return f"508 rows, {ws / GiB:.2f} GiB at the gate; {up / GiB:.2f} GiB once it forwards 998"


def test_a_request_goes_on_past_the_view_only_from_its_generation_prompt():
    """What tells the next turn from a regenerate or an edit: it is longer than the view and shares it up to the
    view's generation prompt (the template's marks), or all of it when the marks are not known."""
    app = _app()
    view = [1] * 5_000 + [90, 93, 2, 2]
    v = len(view)
    old = app._CONV_MARKS
    try:
        app._CONV_MARKS = (90, [91], 92)
        assert app._goes_on(view[:-2] + [4] * 300, view, v - 2), "the next turn"
        assert not app._goes_on(view, view, v), "a regenerate"
        assert not app._goes_on(view[:4_000] + [5] * 2_000, view, 4_000), "an edit, longer than the view"
        assert not app._goes_on(view[:-4] + [5] * 300, view, v - 4), "parts before the generation prompt"
        app._CONV_MARKS = False
        assert not app._goes_on(view[:-2] + [4] * 300, view, v - 2), "marks unknown: the whole view"
        assert app._goes_on(view + [4] * 300, view, v)
    finally:
        app._CONV_MARKS = old
    return "longer than the view and shared up to its generation prompt"


def test_a_reserve_is_raised_to_the_rows_the_prefill_forwards_or_refused_there():
    """The backstop, at the admission: a request admitted at 1 row whose prefill forwards 4,197. `forwards` with their
    workspace raises its reserve when the floor holds the difference (to the byte: margin + KV commit above the
    workspace and the pages), and refuses with the reserve as it was when it does not. Nothing to raise when the
    reserve holds it already, for a key it does not hold, or once the prefill has run."""
    app = _app()
    n, kv = LINE + 4_197, 300 * MiB
    for spare, want in ((16 * MiB, True), (-16 * MiB, False)):
        with _box(app, 30.0, in_flight_gib=0) as b:
            ws1, ws = b.mem.model(n, 1, line=False), b.mem.model(n, 4_197, line=False)
            b.box["avail"] = MARGIN + KV_STEP + ws + kv + spare
            assert b.adm.reserve("r", ws1 + kv, kv=kv, rows=n, fwd=1)[0]
            assert b.adm.forwards("r", 1, ws=ws1) == (True, None), "nothing to raise"
            ok, d = b.adm.forwards("r", 4_197, ws=ws)
            assert ok is want, (spare, slots.Admission.describe(d))
            assert d["raise"] == {"rows": 4_197, "priced": 1, "have": ws1, "ws": ws}, d["raise"]
            assert b.adm.held["r"] == (ws if want else ws1) + kv, "raised, or as it was"
            assert b.adm.fwd["r"] == 4_197, "the floor holds the chunk it forwards either way"
            assert b.kp.hard_floor == b.mem.model(n, CHUNK, line=False) + KV_STEP + MARGIN
            assert (b.adm.stats.get("raised", 0), b.adm.stats.get("raise_refused", 0)) == ((1, 0) if want else (0, 1))
            if not want:
                assert d["after"] < d["floor"] and d["floor"] == MARGIN + KV_STEP, d
            assert b.adm.forwards("other", 4_197, ws=ws) == (True, None)
            b.adm.prefilled("r")
            assert b.adm.forwards("r", 9_000, ws=2 * ws) == (True, None) and b.adm.held["r"] == kv
    return f"{ws1 / GiB:.2f} -> {ws / GiB:.2f} GiB with the floor held; refused 16 MiB under it, the reserve kept"


def test_a_prefill_asks_before_its_first_forward_and_gives_way_when_its_rows_are_not_there():
    """The backstop, at the prefill (`cache.FIRST_FORWARD`): before the first forward the request's real rows are
    held against its price. With room the reserve is raised and the prefill goes on; without, `ForwardShort` (a
    refusal of the pool's kind, so the lane gives way as for pages) is raised before anything is forwarded. In a
    prefill burst the member joins with no job, so the others' joint forward runs, and raises after the join. A
    prefill that is no request's, or one that forwards what it was priced at, asks nothing."""
    app = _app()
    from engine import kvpool
    n = LINE + 4_197

    def first(b, cid, start, burst=None):
        b.pool.burst = burst
        app.STATE["pool"] = b.pool
        out, err = io.StringIO(), None
        with contextlib.redirect_stdout(out):
            try:
                with app._forward_ask(cid, n):
                    app._first_forward(types.SimpleNamespace(), None, start, False)
            except Exception as exc:                               # noqa: BLE001
                err = exc
        return err, out.getvalue()

    for avail, want in ((30.0, True), (8.0, False)):
        with _box(app, avail, in_flight_gib=0) as b:
            ws1, ws = b.mem.model(n, 1, line=False), b.mem.model(n, 4_197, line=False)
            assert b.adm.reserve("r", ws1, rows=n, fwd=1)[0]
            assert first(b, None, LINE) == (None, ""), "the engine's own turn"
            assert first(b, "r", n - 1) == (None, ""), "forwards what it was priced at"
            err, log = first(b, "r", LINE)
            head = (f"[memory] r forwards 4,197 rows, priced at 1 row: its workspace {ws1 / GiB:.2f} -> "
                    f"{ws / GiB:.2f} GiB")
            if want:
                assert err is None and b.adm.held["r"] == ws and f"{head}, reserved" in log, (err, log)
                continue
            assert isinstance(err, app.ForwardShort) and isinstance(err, kvpool.PoolFull), err
            assert app.memory_full_error(err)["code"] == "memory_full"
            assert b.adm.held["r"] == ws1 and f"{head}, not there (" in log and \
                "it gives way before its first forward" in log, log
            assert not app._FWD_ASK, "asked once a prefill"
            # in a burst: it joins with no job and no rows, then raises
            joined = []
            burst = types.SimpleNamespace(member=lambda i: True, join=lambda i, job, run, release=None, acquire=None,
                                          rows=0: joined.append((job, rows)) or (False, None))
            old = app.PREFILL_BURST
            app.PREFILL_BURST = "joint"
            try:
                err, log = first(b, "r", LINE, burst)
            finally:
                app.PREFILL_BURST = old
            assert isinstance(err, app.ForwardShort) and joined == [(None, 0)], (err, joined)
    return f"1 row -> 4,197: raised to {ws / GiB:.2f} GiB at 30 GiB free; at 8 GiB it gives way before the forward"


def test_a_lane_short_at_its_first_forward_gives_way_and_is_admitted_on_what_it_reads():
    """The backstop through the lane (`_recomputing`): a streamed turn admitted at 1 row at 8 GiB free finds 4,197
    rows to forward. It gives its pages back before the forward, waits in the admission's line priced on what it
    now reads, and answers once the memory is there. Not streamed and alone, the refusal stands (`memory_full`)."""
    app = _app()
    import torch
    from test_floor_wait import _H as _LaneH, _lane
    n = LINE + 4_197
    was = app.KV_GIVE_WAY_BACKOFF_S
    app.KV_GIVE_WAY_BACKOFF_S = (0.2, 1.0)
    try:
        for stream in (True, False):
            view, res, st = _chat(n)
            starts = []

            def make(cur, left):
                rows = int(cur.numel())
                start = int(app.STATE["resident"].resume_estimate(cur.tolist(), rows)[0])
                starts.append(start)
                with app._forward_ask("lane", rows):
                    app._first_forward(types.SimpleNamespace(), cur, start, False)
                yield from range(1, left + 1)
            with _box(app, 8.0, in_flight_gib=0, clock=time, **dict(st, queue_timeout=20.0)) as b:
                app.STATE["pool"] = b.pool
                ok, rec0, _ = _admit(app, b, "lane", list(view), home=n, stream=stream)
                assert ok is True and rec0.ws_rows == 1, vars(rec0)
                timer = threading.Timer(1.0, lambda: b.box.__setitem__("avail", 40 * GiB))
                timer.start()
                t0 = time.monotonic()
                try:
                    got, err, raw, rec, log = _lane(app, b.pool, make, stream=stream, prompt=torch.tensor(view),
                                                    h=_LaneH(app), admit={"home": n, "home_rows": True})
                finally:
                    timer.cancel()
                took = time.monotonic() - t0
            if not stream:
                assert isinstance(err, app.ForwardShort) and starts == [LINE] and not getattr(rec, "gave_way", 0), \
                    (err, starts, log)
                assert app.memory_full_error(err)["code"] == "memory_full"
                continue
            assert err is None and got == list(range(1, 9)), (err, got, log)
            assert starts == [LINE, 0] and rec.gave_way == 1, (starts, log)
            assert "it gives way before its first forward" in log and "gives way after 0 tokens" in log, log
            assert rec.ws_rows == n and took >= 0.9, (rec.ws_rows, took)
            said = f"gave way before its forward, admitted again on {rec.ws_rows:,} rows after {took:.1f} s"
    finally:
        app.KV_GIVE_WAY_BACKOFF_S = was
    return said


def test_rows_that_changed_hands_leave_no_view():
    """The share is the view's length, so a view left behind rows that hold another prompt priced a later request at
    home on rows that are not its own. At the end of a request on slot 0 that kept no view, the view goes when the
    rows no longer hold its prompt. It stays while they do, while a guest's stash waits to put them back, and when
    the rows are gone (the price takes no more than the rows hold)."""
    app = _app()
    n = LINE + 4_197
    view, res, st = _chat(n)
    served = res.tokens[:res.end["at"]]
    with _box(app, AVAIL, **st):
        why = {"ids": list(view), "conv": 5_000}
        app.STATE.update(slot0_rows=served, slot0_seen=(0.0, True))
        assert app._par_view_check() is False and app._slot_share(dict(why), None) == n, "the rows hold it"
        res.valid = 0
        assert app._par_view_check() is False and app.STATE["slot0_view"] is view, "rows gone: the view stays"
        res.tokens, res.valid, res._stash = [9] * 60_000, 60_000, object()
        assert app._par_view_check() is False, "a guest's stash puts them back"
        res._stash = None
        assert app._par_view_check() is True, "another prompt's rows"
        assert not any(k in app.STATE for k in ("slot0_view", "slot0_rows", "slot0_seen"))
        assert app._slot_share(dict(why), None) == 0 and app._par_view_check() is False
    return "the view goes with the rows it stood for"


def test_a_quiet_run_says_its_last_count_when_the_wait_ends():
    """Of a run of hand-backs that freed nothing, only the first had its line when the wait ended inside a minute:
    the count was lost. The wait's end prints what each run counted since its last line, once, and the next
    hand-back that frees nothing starts a new run."""
    app = _app()
    app._EMPTY_RUN.clear()
    ev = {"kind": "empty", "why": "kv_pool", "freed": 0, "ms": 0.37}
    lines = [app._empty_line(ev, now=1000.0 + i / 8) for i in range(12)]
    assert lines[0] and not any(lines[1:]), lines
    assert app._empty_line(dict(ev, why="gate"), now=1001.0), "a run of one: no count to say"
    assert app._empty_last() == ["[memory] reserve back to the driver: 0.00 GiB, 11 more times in 4.1 ms (kv_pool), "
                                 "the last of the run"]
    assert app._empty_last() == [] and not app._EMPTY_RUN
    assert app._empty_line(ev, now=1002.0) == lines[0], "a new run starts with its line"
    app._EMPTY_RUN.clear()
    return "12 hand-backs in 1.5 s: the first line, then `11 more times` at the wait's end"


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        out = fn()
        print(f"ok  {fn.__name__}" + (f"  {out}" if out else ""))
    print(f"{len(fns)} passed")
