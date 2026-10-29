"""The side lane (server/preempt.py): a short request served in the middle of another
one's generation, and the generation given back exactly as it was.

2026-10-04 16:43-16:48: one opencode turn thought for minutes; opencode's title request of a new
session waited 256 s and got a 429, and the router's own questions to Qwen (`anon`, raw, 24 tokens
under a choice grammar) gave up after 9 s each.

The test matrix:

  the engine (the random 4-layer model, fp32 on the CPU):
    1  a guest served between two verify blocks -- after a full accept, after a partial one
       (`rollback_to`), twice in a row -- leaves the next block's logits, the recurrent state, the
       convolution tails and every KV row of the generation bit for bit as an uninterrupted run
                                                              test_engine_*_bit_for_bit
    2  the guest's own tokens are plain greedy decoding of its prompt, whatever it interrupted,
       its rows reaching past the generation's length or not   test_guest_is_plain_greedy
    3  a guest between two prefill chunks: the prefill ends in the same logits and state
                                                              test_engine_prefill_chunk_*
  the lane (no engine):
    4  a pending job a holder serves; a job no holder takes is run by its own thread under the lock;
       a client that leaves withdraws a pending job; a failing job is the guest's error, never the
       holder's                                               test_lane_*
  the disk tier:
    5  an open write that has not copied the rows the guest writes stops where it is; one that has,
       goes on                                                test_disk_guard_*
  the handler (the fake engine in process, the lane on):
    6  opencode's title request during a long turn is served at once, the turn's text is the text
       it writes alone; the ledger row says `lane`; the journal row carries `extra.lane`
                                                              test_served_title_*
    7  the router's raw question shape (24 tokens, thinking off) is served in the lane; a raw
       request with thinking on, one over the output cap, and a TandemLLM turn wait in the queue
                                                              test_raw_question_*, test_not_*
  side slots (SPD-249, QWEN38_OWN_SIDE):
    8  TandemLLM's own raw question short enough for the lane gets a side window and takes an idle
       side slot (no carry) while slot 0 is busy, never one whose last end waits for its
       conversation; a client's raw request keeps the lane; a longer own question does not wait
       15 s behind other own work; off: the lane and the wait   test_spd249_*

Run: python tests/test_side_lane.py
"""

from __future__ import annotations

import os
import sys
import threading
import time

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from engine import cache  # noqa: E402
from engine.model import h2d  # noqa: E402
from server import preempt  # noqa: E402
from test_cache import DEV, fresh, tokens  # noqa: E402

CH = 16


# ------------------------------------------------------------------ 1-3. the engine

def _guest(eng, ids, n=6):
    job = preempt.Job(ids, n, eos=set())
    preempt.decode(eng, job, device=DEV, chunk=CH, h2d=h2d)
    out = []
    while not job.out.empty():
        out.append(job.out.get())
    return out


def _generation(guests: dict, prompt_n=70, blocks=((5, 5), (4, 2), (6, 6), (3, 1)), seed=1):
    """A prefill and four verify blocks (tokens, kept rows), with guests served before the blocks
    named in `guests` (index -> list of guest prompts). Returns the engine, every block's logits
    and the final length."""
    eng = fresh()
    prompt = tokens(prompt_n, seed=seed)
    cache.prefill(eng, None, prompt, DEV, chunk=CH)
    pos, outs = prompt_n, []
    for i, (n, keep) in enumerate(blocks):
        for g in guests.get(i, ()):
            _guest(eng, g)
        lg = eng.forward_block(torch.tensor(tokens(n, seed=10 + i)), start=pos)
        outs.append(lg.clone())
        if keep < n:
            eng.rollback_to(keep)
        pos += keep
    return eng, outs, pos


def _same(a, b, outs_a, outs_b, n):
    for x, y in zip(outs_a, outs_b):
        assert torch.equal(x, y), (x - y).abs().max().item()
    assert torch.equal(a.state.S, b.state.S), (a.state.S - b.state.S).abs().max().item()
    assert torch.equal(a.state.conv, b.state.conv)
    assert torch.equal(a.kv.k[:, :, :, :n], b.kv.k[:, :, :, :n])
    assert torch.equal(a.kv.v[:, :, :, :n], b.kv.v[:, :, :, :n])
    assert a.kv.length == b.kv.length == n


def test_engine_after_a_full_and_a_partial_accept_bit_for_bit():
    ref, outs, n = _generation({})
    # before block 1 (after block 0's full accept) and before block 2 (after a rollback)
    got, outs2, n2 = _generation({1: [tokens(30, seed=40)], 2: [tokens(25, seed=41)]})
    _same(ref, got, outs, outs2, n)


def test_engine_two_guests_in_a_row_and_one_past_the_generation_bit_for_bit():
    ref, outs, n = _generation({})
    # a guest longer than the whole generation writes rows past its length
    got, outs2, _ = _generation({0: [tokens(20, seed=50), tokens(120, seed=51)],
                                 3: [tokens(9, seed=52)]})
    _same(ref, got, outs, outs2, n)


def test_guest_is_plain_greedy():
    ids = tokens(40, seed=60)
    cold = fresh()
    lg, _, _ = cache.prefill(cold, None, ids, DEV, chunk=CH)
    want, pos, row = [], len(ids), lg[0, -1]
    for _ in range(6):
        t = int(row.argmax())
        want.append(t)
        row = cold.forward(torch.tensor([t]), start=pos, last_only=True)[0, -1]
        pos += 1
    eng = fresh()
    cache.prefill(eng, None, tokens(90, seed=61), DEV, chunk=CH)
    assert _guest(eng, ids) == want
    eng2 = fresh()
    cache.prefill(eng2, None, tokens(10, seed=62), DEV, chunk=CH)     # rows past its length
    assert _guest(eng2, ids) == want


def test_engine_prefill_chunk_guest_bit_for_bit():
    prompt = tokens(100, seed=70)
    ref = fresh()
    lg_ref, _, _ = cache.prefill(ref, None, prompt, DEV, chunk=CH)
    eng = fresh()
    served = []

    def hook(done, total):
        if done in (32, 64):
            served.append(_guest(eng, tokens(20, seed=71 + done)))

    lg, _, _ = cache.prefill(eng, None, prompt, DEV, chunk=CH, on_chunk=hook)
    assert len(served) == 2
    assert torch.equal(lg, lg_ref)
    _same(ref, eng, [], [], len(prompt))


# ------------------------------------------------------------------ 4. the lane

def _ran(job):
    out = []
    while True:
        t = job.out.get(timeout=5)
        if t is preempt.DONE:
            return out
        out.append(t)


def _writer(job):
    for t in job.ids[:job.max_new]:
        job.n_out += 1
        job.out.put(t)


def test_lane_a_holder_serves_a_pending_job():
    lane = preempt.Lane(_writer)
    job = preempt.Job([5, 6, 7], 2, eos=set())
    lane.submit(job)
    assert lane.serve("decode") == 1
    assert _ran(job) == [5, 6] and job.where == "decode" and job.state == "done"
    assert job.done.is_set() and job.lent_ms is not None
    assert lane.serve("decode") == 0
    assert lane.stats["served_decode"] == 1


def test_lane_a_job_nobody_takes_runs_under_the_lock_on_its_own_thread():
    lane = preempt.Lane(_writer)
    lock = threading.Lock()
    job = preempt.Job([1, 2, 3], 3, eos=set())
    got = list(lane.tokens(job, take=lambda: lock.acquire(blocking=False), release=lock.release))
    assert got == [1, 2, 3] and job.where == "self" and not lock.locked()


def test_lane_a_job_waits_while_the_lock_is_held_and_a_holder_serves_it():
    lane = preempt.Lane(_writer)
    lock = threading.Lock()
    lock.acquire()                                  # a generation holds the engine
    job = preempt.Job([4, 4], 2, eos=set())
    box = {}
    t = threading.Thread(target=lambda: box.update(out=list(lane.tokens(
        job, take=lambda: lock.acquire(blocking=False), release=lock.release))))
    t.start()
    while not lane.jobs:
        time.sleep(0.01)
    time.sleep(0.2)
    assert job.state == "pending"                   # nobody served it, the lock is held
    lane.serve("decode")                            # the holder's safe point
    t.join(5)
    lock.release()
    assert box["out"] == [4, 4] and job.where == "decode"


def test_lane_a_client_that_leaves_withdraws_its_pending_job():
    lane = preempt.Lane(_writer)
    lane.gone_exc = TimeoutError
    lock = threading.Lock()
    lock.acquire()
    job = preempt.Job([1], 1, eos=set())
    try:
        list(lane.tokens(job, take=lambda: False, release=lock.release, gone=lambda: True))
        raise AssertionError("no exception")
    except TimeoutError:
        pass
    assert job.state == "cancelled" and not lane.jobs and lane.serve("decode") == 0


def test_lane_a_failing_job_is_the_guests_error_not_the_holders():
    def bad(job):
        job.out.put(9)
        raise RuntimeError("boom")

    lane = preempt.Lane(bad)
    job = preempt.Job([1], 4, eos=set())
    box = {}

    def guest():
        try:
            box["out"] = list(lane.tokens(job, take=lambda: False, release=lambda: None))
        except RuntimeError as exc:
            box["err"] = str(exc)

    t = threading.Thread(target=guest)
    t.start()
    while not lane.jobs:
        time.sleep(0.01)
    assert lane.serve("decode") == 1                # did not raise
    t.join(5)
    assert box == {"err": "boom"} and lane.stats["errors"] == 1


def test_lane_a_job_without_memory_room_waits_for_a_later_safe_point():
    rooms = [False, True]
    lane = preempt.Lane(_writer, room=lambda job, where: rooms.pop(0))
    job = preempt.Job([3], 1, eos=set())
    lane.submit(job)
    assert lane.serve("prefill") == 0 and job.state == "pending"     # the holder at its peak
    assert lane.serve("decode") == 1 and _ran(job) == [3]
    assert lane.stats["no_room"] == 1


def test_lane_caps():
    lane = preempt.Lane(_writer, prompt_max=2048, new_max=64)
    assert lane.eligible(587, 64) and lane.eligible(409, 24)
    assert not lane.eligible(2049, 24) and not lane.eligible(500, 65)
    assert not preempt.Lane(_writer, where="off").eligible(10, 10)


# ------------------------------------------------------------------ 5. the disk tier

class _Job:
    def __init__(self, chunks, copied, limit):
        self.chunks, self.copied, self.limit = chunks, copied, limit


def _tier(*jobs):
    from engine.disk_cache import DiskTier
    t = DiskTier.__new__(DiskTier)
    t._fence = threading.Lock()
    t._open_jobs = list(jobs)
    t.chunk = 1024
    return t


def test_disk_guard_stops_a_write_that_has_not_copied_the_guests_rows():
    j = _Job([1024, 2048, 3072, 4096], copied=0, limit=4096)
    assert _tier(j).guard_rows(700) == 1 and j.limit == 0


def test_disk_guard_leaves_a_write_past_the_guests_rows():
    j = _Job([1024, 2048, 3072, 4096], copied=2048, limit=4096)
    assert _tier(j).guard_rows(700) == 0 and j.limit == 4096
    # chunks already on the disk (the job's list starts higher): nothing below the guest's rows
    k = _Job([5120, 6144], copied=0, limit=6144)
    assert _tier(k).guard_rows(2000) == 0 and k.limit == 6144


# ------------------------------------------------------------------ 6-7. the handler

def _lane_on():
    from server import app
    # the fake's tokenizer is a character a token: opencode's 587-token title is 2,302 here
    app.STATE["lane"] = preempt.Lane(app.STATE["lane_run"], prompt_max=4096)
    app.STATE["lane"].gone_exc = app.ClientGone
    return app.STATE["lane"]


def test_served_title_during_a_long_turn_is_at_once_and_the_turn_is_unchanged():
    import test_side_requests as ts
    import test_ui_states as tu
    from server import app
    cond, order = ts.serve(None)
    body = ts.tu_body("FAKE_SLOW FAKE_LONG one", tu.TOOLS, 120)
    with ts.threads_quiet():
        alone = ts.go(body, {"x-session-id": "ses_alone"})
    lane = _lane_on()
    with ts.threads_quiet():
        busy, box = ts._bg(body, {"x-session-id": "ses_alone"})
        ts._wait_until(lambda: app.LOCK.locked())
        time.sleep(0.5)                              # the turn decodes at 10 tokens a second
        t0 = time.monotonic()
        side = ts.go(ts.opencode_title())
        side_s = time.monotonic() - t0
        assert busy.is_alive(), ("the turn should still be running", side_s)
        busy.join(60)
    assert side_s < 2.0, side_s
    assert side["usage"]["completion_tokens"] <= 64 and side["content"], side
    assert box["out"]["content"] == alone["content"]
    assert box["out"]["reasoning"] == alone["reasoning"]
    assert lane.stats["served_decode"] == 1, lane.report()
    rows = [r for r in cond.journal.ring.values() if r.get("kind") == "side"]
    assert rows and rows[-1]["extra"]["lane"]["where"] == "decode", rows[-1]["extra"]
    app.STATE["lane"] = None


def _raw_question(**extra):
    return {"model": "qwen38-spark-engine", "stream": False, "temperature": 0, "max_tokens": 24,
            "messages": [{"role": "system", "content": "Pick one id."},
                         {"role": "user", "content": "the user wants a voxel pagoda"}],
            "chat_template_kwargs": {"enable_thinking": False}, **extra}


def _raw_during_turn(body, headers=None):
    import test_side_requests as ts
    import test_ui_states as tu
    from server import app
    ts.serve(None)
    lane = _lane_on()
    with ts.threads_quiet():
        busy, _ = ts._bg(ts.tu_body("FAKE_SLOW one", tu.TOOLS, 40), {"x-session-id": "ses_b"})
        ts._wait_until(lambda: app.LOCK.locked())
        time.sleep(0.3)
        t0 = time.monotonic()
        out = ts.go(body, headers if headers is not None else
                    {"X-Tandem": "raw", "User-Agent": "tandem-conductor/1"})
        took = time.monotonic() - t0
        busy.join(60)
    app.STATE["lane"] = None
    return out, took, lane


def test_raw_question_is_served_in_the_lane():
    out, took, lane = _raw_during_turn(_raw_question())
    assert took < 2.0 and lane.stats["served_decode"] == 1, (took, lane.report())
    assert out["usage"]["completion_tokens"] <= 24


def test_not_a_raw_request_that_thinks():
    out, took, lane = _raw_during_turn(_raw_question(chat_template_kwargs={}))
    assert lane.stats["jobs"] == 0 and took > 2.0, (took, lane.report())


def test_not_a_raw_request_over_the_output_cap():
    out, took, lane = _raw_during_turn(_raw_question(max_tokens=200))
    assert lane.stats["jobs"] == 0 and took > 2.0, (took, lane.report())


def test_not_a_tandem_turn():
    import test_ui_states as tu
    body = {"model": "TandemLLM", "stream": False, "max_tokens": 20,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": tu.SYSTEM},
                         {"role": "user", "content": "hello"}]}
    out, took, lane = _raw_during_turn(body, {})
    assert lane.stats["jobs"] == 0 and took > 2.0, (took, lane.report())


def test_a_long_side_request_waits_past_the_queue_timeout_and_is_served():
    import test_side_requests as ts
    import test_ui_states as tu
    from server import app
    ts.serve(None)
    _lane_on()
    app.STATE["queue_timeout"] = 1.0
    try:
        with ts.threads_quiet():
            busy, _ = ts._bg(ts.tu_body("FAKE_SLOW one", tu.TOOLS, 40), {"x-session-id": "ses_q"})
            ts._wait_until(lambda: app.LOCK.locked())
            t0 = time.monotonic()
            long_side = ts.go(ts.opencode_title(user="x" * 5000))     # over the lane's cap
            took = time.monotonic() - t0
            busy.join(60)
            raw = ts.go(_raw_question(max_tokens=200), {"X-Tandem": "raw"})   # idle: served
    finally:
        app.STATE["lane"] = None
        app.STATE["queue_timeout"] = 120.0
    assert took > 1.5 and long_side["usage"] and long_side["usage"]["completion_tokens"] > 0, \
        (took, long_side["head"][:80])
    assert raw["usage"]["completion_tokens"] > 0



# ------------------------------------------------------------------ 8. side slots (SPD-249)

def _own_need(n_rows=300, max_new=24, own=True, lane_kind="raw", side_len=8192):
    """`_par_side_need` for a raw question of `n_rows` rendered tokens beside a busy slot 0."""
    from server import app as A
    from server import slots as S
    pool = S.Pool([S.Slot(0, 65_536), S.Slot(1, side_len), S.Slot(2, side_len)])
    saved = {k: A.STATE.get(k) for k in ("pool", "resident", "slot0_view")}
    A.STATE["pool"], A.STATE["resident"] = pool, None
    A.STATE.pop("slot0_view", None)
    real = A.build_prompt
    A.build_prompt = lambda body, client=None, *, count=False, left_out=0: list(range(n_rows))
    try:
        why: dict = {}
        need = A._par_side_need({"messages": []}, None, 0, max_new, None, [], lane_kind, why,
                                own=own)
        return need, why, pool
    finally:
        A.build_prompt = real
        for k, v in saved.items():
            if v is None:
                A.STATE.pop(k, None)
            else:
                A.STATE[k] = v


def test_spd249_an_own_lane_question_gets_a_side_window():
    from server import app as A
    need, why, _ = _own_need()
    assert need == 300 + 24 + A.SIDE_MARGIN and why["lane_side"], (need, why)
    need, why, _ = _own_need(own=False)
    assert need is None and "lane_side" not in why, "a client's raw request keeps the lane"
    need, why, _ = _own_need(lane_kind=None)
    assert "lane_side" not in why, "not a lane job: the usual window rules"
    need, why, _ = _own_need(side_len=256)
    assert need is None and why["lane_side"], "no side slot holds it: the lane"
    was = A.OWN_SIDE
    A.OWN_SIDE = "off"
    try:
        need, why, _ = _own_need()
        assert need is None and "lane_side" not in why
    finally:
        A.OWN_SIDE = was


def test_spd249_the_routers_estimate_counts_an_idle_side_slot():
    """`_own_side_free` (the router's `own_eta`) as admission: a lane-sized own question has an idle
    side slot whose last end is not kept; a longer one any idle slot; off: the lane, no slot."""
    import time as _t
    from server import app as A
    from server import slots as S
    pool = S.Pool([S.Slot(0, 65_536), S.Slot(1, 8192)])
    was_pool, was = A.STATE.get("pool"), A.OWN_SIDE
    A.STATE["pool"] = pool
    try:
        assert A._own_side_free(300, 24, True, None) and A._own_side_free(3000, 24, False, -1)
        assert not A._own_side_free(9000, 24, False, -1)
        pool.slots[1].carry = {"gen": pool.slots[1].gen, "t": _t.monotonic()}
        assert not A._own_side_free(300, 24, True, None) and A._own_side_free(3000, 24, False, -1)
        pool.slots[1].carry = None
        A.OWN_SIDE = "off"
        assert not A._own_side_free(300, 24, True, None) and not A._own_side_free(3000, 24, False, -1)
    finally:
        A.OWN_SIDE = was
        A.STATE["pool"] = was_pool


def test_spd249_side_ok_lets_own_work_take_a_side_slot_at_once():
    from server import app as A
    assert A._side_ok(0.0, None) and not A._side_ok(0.0, -1)
    assert A._side_ok(0.0, -1, own=True), "an own question needs no slot 0 caches"
    assert A._side_ok(A.SIDE_AFTER_OWN_S, -1)
    was = A.OWN_SIDE
    A.OWN_SIDE = "off"
    try:
        assert not A._side_ok(0.0, -1, own=True)
    finally:
        A.OWN_SIDE = was


def test_spd249_side_free_reads_busy_slots_and_kept_ends():
    import time as _t
    from server import slots as S
    pool = S.Pool([S.Slot(0, 65_536), S.Slot(1, 8192)])
    assert pool.side_free(1000) and not pool.side_free(9000)
    s1 = pool.slots[1]
    s1.carry = {"gen": s1.gen, "t": _t.monotonic()}
    assert not pool.side_free(1000, keep_carry_s=600.0) and pool.side_free(1000)
    assert pool.take_side(1000, keep_carry_s=600.0) is None
    s1.carry = None
    got = pool.take_side(1000)
    assert got is s1 and not pool.side_free(1000)
    s1.lock.release()


def _fake_par():
    import contextlib
    import io
    from server import app as A
    from server import fake_engine
    a = A.parser().parse_args(["--fake-engine", "--fake-tps", "2000", "--max-len", "65536",
                               "--default-max-tokens", "64", "--parallel", "3",
                               "--side-max-len", "8192"])
    A.STATE.clear()
    A.CLOCK = A.qwait.TurnClock()
    with contextlib.redirect_stdout(io.StringIO()):
        fake_engine.load(A, a)
    lane = A.STATE["lane"] = preempt.Lane(A.STATE["lane_run"], prompt_max=2048)
    lane.gone_exc = A.ClientGone
    return A.STATE["pool"], lane


def _held_then(body, holder, wait=1.5):
    """Slot 0 held by `holder` (the lock held, no generation in it), then `body` from the router's
    agent: (the response or None while it still waits, seconds, pool, lane)."""
    from server import app as A
    from test_ui_states import Req
    pool, lane = _fake_par()
    A.LOCK.acquire()
    A.LOCK.turn.release(0)
    with A.QUEUE:
        A.HOLDER["who"] = dict(holder, t=time.monotonic(), t_hold=time.monotonic())
    out: dict = {}

    def send():
        t0 = time.monotonic()
        head, raw = Req(body, {"User-Agent": "tandem-conductor/1", "X-Tandem": "raw"}).response()
        out.update(raw=raw, t=time.monotonic() - t0)
    th = threading.Thread(target=send, daemon=True)
    try:
        th.start()
        th.join(wait)
        early = dict(out) if not th.is_alive() else None
    finally:
        with A.QUEUE:
            A.HOLDER["who"] = None
        A.LOCK._lk.release()
        th.join(30)
    A.STATE["lane"] = None
    return early, out, pool, lane


def test_spd249_an_own_question_takes_an_idle_side_slot_not_the_lane():
    """Slot 0 busy with a client's turn: the router's question runs on an idle side slot at once (no
    carry), not in slot 0's lane behind the questions ahead of it; off: the lane, served only when the
    lock comes free (here: no generation serves it meanwhile)."""
    from server import app as A
    early, out, pool, lane = _held_then(_raw_question(), {"rank": A.RANK_TURN})
    assert early is not None and '"choices"' in early["raw"], out
    assert pool.stats.get("lane_side") == 1 and pool.stats["side_grants"] == 1, pool.stats
    assert [getattr(s, "no_carry", False) for s in pool.side()].count(True) == 1, "nobody carries it"
    assert lane.stats["jobs"] == 0, lane.report()
    was = A.OWN_SIDE
    A.OWN_SIDE = "off"
    try:
        early, out, pool, lane = _held_then(_raw_question(), {"rank": A.RANK_TURN}, wait=1.0)
    finally:
        A.OWN_SIDE = was
    assert early is None and '"choices"' in out["raw"], out
    assert lane.stats["jobs"] == 1 and pool.stats["side_grants"] == 0, (lane.report(), pool.stats)


def test_spd249_a_long_own_question_does_not_wait_behind_own_work():
    """A question too long for the lane (the package question at 2.8k tokens) while slot 0 runs another
    of the router's questions: an idle side slot at once; off: it waits SIDE_AFTER_OWN_S."""
    from server import app as A
    body = _raw_question()
    body["messages"][1]["content"] = "the user wants a voxel pagoda " * 100     # 3,000 tokens here
    early, out, pool, _ = _held_then(body, {"rank": A.RANK_TURN, "own": True})
    assert early is not None and pool.stats["side_grants"] == 1, (out, pool.stats)
    assert "lane_side" not in pool.stats
    was = A.OWN_SIDE
    A.OWN_SIDE = "off"
    try:
        early, out, pool, _ = _held_then(body, {"rank": A.RANK_TURN, "own": True}, wait=1.0)
    finally:
        A.OWN_SIDE = was
    assert early is None and '"choices"' in out["raw"] and pool.stats["side_grants"] == 0, pool.stats

if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
