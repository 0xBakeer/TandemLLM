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

Run: python tests/test_memory_wait.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
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


class _Pages:
    """A slot's paged KV as the admission and the guard read it (engine/kvpool.py `PagedSet`), `rows` mapped."""

    def __init__(self, rows: int = 0, owner: str = "slot0"):
        self.rows, self.owner = int(rows), owner
        self.bufs = [types.SimpleNamespace(outer=48, row_bytes=1024, page=2 << 20)]

    def margin(self) -> int:
        return 256

    def need_bytes(self, rows: int) -> int:
        return max(0, int(rows) - self.rows) * PER_ROW


def _engine(rows: int = 0, owner: str = "slot0"):
    return types.SimpleNamespace(kv=types.SimpleNamespace(pages=_Pages(rows, owner), max_len=262_144, length=rows))


@contextlib.contextmanager
def _box(app, avail_gib: float, in_flight_gib: float = IN_FLIGHT, **state):
    """The server's state for one admission: MemAvailable `avail_gib`, a 15 GiB admission floor and a 10 GiB pool
    floor as configured, both reading the live guard (`_wire_guards`), Flash-Next's workspace model, the engine at
    54.4 GiB under a 75.82 GiB peak, and the KV pages of the lanes in flight. The clock is the test's."""
    box = {"avail": int(avail_gib * GiB)}
    adm = slots.Admission(15 * GiB, meminfo=lambda: {"MemAvailable": box["avail"]}, growth=lambda: 0)
    kp = slots.GuardedPool(hard_floor=10 * GiB, margin_rows=0, meminfo=lambda: box["avail"])
    mem = _FNMem(peak=int(75.82 * GiB), ws=0)
    pool = types.SimpleNamespace(admission=adm, turn=None, running=lambda: 1,
                                 slots=[types.SimpleNamespace(phase="", prefill_left=None) for _ in range(8)])
    st = dict(mem=mem, mem_load={"allocated": int(54.4 * GiB)}, prefix_chunk=4096, engine=_engine(),
              queue_timeout=1.0, request_timeout=30.0)
    st.update(state)
    old_time, old_pool = app.time, app.kvpool.POOL
    app.STATE.clear()
    app.STATE.update(st)
    app.kvpool.POOL = kp
    app.time = _Clock()
    app._wire_guards(adm, kp)
    if in_flight_gib:
        kv = int(in_flight_gib * GiB)
        assert adm.reserve("in-flight", kv, kv=kv, owner="slot3")[0]
    try:
        yield types.SimpleNamespace(box=box, adm=adm, kp=kp, mem=mem, pool=pool)
    finally:
        slots.bind(None)
        app.time, app.kvpool.POOL = old_time, old_pool
        app.STATE.clear()


def _admit(app, b, cid: str, n: int, home: int = 0, max_new: int = 8192, slot=None, limit_s: float = 2.0):
    """`_par_admit` of an `n`-token prompt on slot 0 or `slot`, its wait bounded at `limit_s` on the test's clock:
    (admitted, its record, the engine log's lines)."""
    import torch
    from server import usage
    rec = usage.RequestRecord(f"chatcmpl-{cid}", "chat", True)
    out = io.StringIO()
    slots.bind(slot)
    try:
        with contextlib.redirect_stdout(out):
            ok = app.Handler._par_admit(_H(), b.pool, cid, torch.zeros(n, dtype=torch.long), rec=rec, stream=True,
                                        max_new=max_new, home=home, limit_s=limit_s)
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


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        out = fn()
        print(f"ok  {fn.__name__}" + (f"  {out}" if out else ""))
    print(f"{len(fns)} passed")
