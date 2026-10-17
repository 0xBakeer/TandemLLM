"""ENG-371 (QA ENG-372): an agent's worker keeps its own rows between its calls, and its parent's
rows are kept while it waits. CPU, the random 4-layer model and the slot choice on fakes.

THE INCIDENT (2026-10-17 23:38, Qwen3.8-27B EXL3 6.0, `--parallel 2`, `--resident-guest-max 8192`;
chat 01M55WQ92F8D8BKXMKCKWCNXM9, engine-20261017-220207.log). Slot 0 held another chat (24,200
rows, with images), the parent ran on side slot 1. The worker's calls 1 to 3 (2,532, 5,764 and 7,925
tokens) each started from 0:

  * call 1 ran on slot 0 as a GUEST of the other chat (shorter than 8,192 and than a quarter of
    24,200): its rows were stashed over and put back after it, `end anchor not kept (guest)`;
  * call 2 found slot 0 held by call 1's tail, took side slot 1 as a guest of slot 0 and wrote over
    the parent's 15,066 rows there (the stash could not hold its reach: `not_resident`), keeping
    nothing of its own either; the parent's return re-read 11,039 of 16,159 tokens;
  * call 3 (over a quarter of 24,200) was no guest and read all 7,925 again.

And the run before it (chat 01M55TKMFSS, 23:05): the worker took slot 0 from its parent, the park
held the parent, and the memory gate gave the park back at the ceiling unwritten
(`shed level=park why=ceiling freed=1898774528`): the parent's return re-read 11,895 of 17,015.

THE FIX: a request with `X-Tandem-Worker: 1` / `X-Tandem-Parent-Session` is never a guest
(`ResidentPrefix.begin(stay=True)`, `_par_side_need`); a side slot's rows of a waiting parent go to
the NVMe before another session's prefill cuts them (`ResidentPrefix.keeper`, `_keep_parent`); a
parked parent goes to the NVMe before the memory policy takes the park (`_shed_park`). Each test
runs the old path too and shows it re-reading.

Run: python tests/test_worker_cache.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)
import torch  # noqa: E402

from engine import cache  # noqa: E402
from server import app  # noqa: E402
from server import slots as slots_mod  # noqa: E402
from test_cache import fresh, tokens  # noqa: E402
from test_end_anchor import BIG, CH, TOOL, _state  # noqa: E402
from test_session_spill import _spill  # noqa: E402

WHEAD = tokens(40, seed=60)                  # the worker's system block and tools, every worker's
PHEAD = tokens(40, seed=61)                  # the chat's


def wturn(ids, n, session, stay=False):
    """One request as the handler runs it (`turn` of test_end_anchor), `stay` for a worker."""
    out = list(app.generate_stream(torch.tensor(ids), n, set(), None, session, stay=stay))
    pre = dict(app.STATE["last_prefill"])
    app._remember(list(ids), out, session)
    return out, pre


def cold(ids, n, **kw):
    """The same request with nothing cached: the tokens every resume must give."""
    _state(**kw)
    out, _ = wturn(ids, n, "cold")
    return out


def _tail(pre, ids):
    return len(ids) - int(pre["reused"])


# ------------------------------------------------------------------ 1: the incident on slot 0

def _calls_beside_a_long_chat(stay):
    """Another chat on slot 0 (408 rows), then a worker's call 1 (80 tokens: a guest by length),
    the router's question and the parent's context peek between, then call 2."""
    _state(guest_max=8192)
    other = tokens(400, seed=70)
    wturn(other, 8, "chat-other")
    w1 = WHEAD + tokens(40, seed=71)
    out1, pre1 = wturn(w1, 8, "chat-p-task-1", stay)
    q, _ = wturn(tokens(12, seed=72), 2, None)           # the router's question: a guest
    app._peek_find(PHEAD + tokens(300, seed=73))         # the parent's Context panel peek
    w2 = w1 + out1 + TOOL
    out2, pre2 = wturn(w2, 8, "chat-p-task-1", stay)
    return w1, out1, w2, out2, pre2


def test_a_workers_next_call_resumes_at_its_own_end_beside_a_long_chat():
    w1, out1, w2, out2, pre2 = _calls_beside_a_long_chat(stay=True)
    assert pre2["reused"] >= len(w1) + len(out1) - 1, pre2
    assert _tail(pre2, w2) <= len(TOOL) + 1, pre2               # only the new tail is prefilled
    assert out2 == cold(w2, 8), "the resume gives the cold prefill's tokens"
    # the old path: a guest, its rows put back under the other chat's after each call
    _, _, w2_, _, pre_old = _calls_beside_a_long_chat(stay=False)
    assert pre_old["reused"] == 0 and _tail(pre_old, w2_) == len(w2_), pre_old


def test_a_short_request_without_the_worker_header_is_still_a_guest():
    res = _state(guest_max=8192)
    other = tokens(400, seed=70)
    out, _ = wturn(other, 8, "chat-other")
    wturn(tokens(60, seed=74), 4, "title")
    assert res.owner == "chat-other" and res.stats["guests"] == 1, res.stats
    _, pre = wturn(other + out + tokens(9, seed=75), 4, "chat-other")
    assert pre["reused"] >= len(other) + len(out) - 1, pre


# ------------------------------------------------------------------ 2: two chats' workers at once

def _two_chats(stay, d):
    """Two parents (348 rows each, the same head) and their workers (80 tokens, the same worker
    head), taking turns on slot 0 with a park that holds one conversation and the NVMe spill.
    Returns the resident prefix, each worker's (tail prefilled, prompt) of calls 2 and 3, and
    each parent's return."""
    per = sum(t.numel() // t.shape[ax] * t.element_size()
              for t, ax in cache._kv_views(fresh(), None))
    res = _state(guest_max=8192, park_bytes=900 * per + 8 * fresh().state.nbytes,
                 park_anchors=4)
    _spill(d, res, min_tokens=64)
    app.PARENTS.clear()
    p = {k: PHEAD + tokens(300, seed=s) for k, s in (("a", 80), ("b", 81))}
    w = {k: WHEAD + tokens(40, seed=s) for k, s in (("a", 82), ("b", 83))}
    pout = {}
    for k in "ab":
        pout[k], _ = wturn(p[k], 8, f"chat-{k}")
    tails = {}
    for step in range(3):
        for k in "ab":
            app._note_parent(f"chat-{k}")
            out, pre = wturn(w[k], 6, f"chat-{k}-task-1", stay)
            if step:
                tails.setdefault(k, []).append((_tail(pre, w[k]), len(w[k])))
            w[k] = w[k] + out + tokens(12, seed=90 + 2 * step + (k == "b"))
    back = {}
    for k in "ab":
        ids = p[k] + pout[k] + tokens(15, seed=95 + (k == "b"))
        out, pre = wturn(ids, 6, f"chat-{k}")
        back[k] = (ids, out, pre)
    return res, tails, back


def test_two_chats_workers_each_keep_their_prefix_and_the_parents_come_back():
    d = tempfile.mkdtemp()
    try:
        res, tails, back = _two_chats(True, d)
        for k in "ab":
            assert all(t <= 12 + 1 for t, _ in tails[k]), (k, tails)   # each call: its tail only
            ids, out, pre = back[k]
            assert _tail(pre, ids) <= 15 + 1, (k, pre)              # the parent: its tail only
        assert res.stats["park_swaps"] >= 4 and res.stats["park_spills"] >= 2, res.stats
        ids, out, _ = back["b"]
        assert out == cold(ids, 6), "the parent's return gives the cold prefill's tokens"
    finally:
        shutil.rmtree(d, ignore_errors=True)
    d = tempfile.mkdtemp()
    try:
        _, tails, _ = _two_chats(False, d)
        # the old path: call 1, shorter than a quarter of the parent, was a guest, and call 2
        # read again all of its prompt past the worker head the two workers share
        assert all(t >= n - len(WHEAD) for k in "ab" for t, n in tails[k][:1]), tails
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 3: the parent on a side slot

def _side_parent(keeper, d):
    """The parent ran on a side slot (no park, no spill of its own); its worker's call takes that
    slot; the parent's return runs on slot 0."""
    res0 = _state(guest_max=8192)
    _spill(d, res0, min_tokens=64)
    eng0 = app.STATE["engine"]
    side = cache.ResidentPrefix(BIG, CH, end=True, stash_bytes=BIG, guest_max=8192,
                                guest_frac=1.0, park_bytes=0)
    side.keeper = keeper
    eng1 = fresh()
    app.PARENTS.clear()
    p1 = PHEAD + tokens(300, seed=84)
    app.STATE.update(engine=eng1, resident=side)              # the side slot's thread
    pout, _ = wturn(p1, 8, "chat-p")
    app._note_parent("chat-p")
    wturn(WHEAD + tokens(40, seed=85), 6, "chat-p-task-1", stay=True)
    assert side.owner == "chat-p-task-1"
    app.STATE.update(engine=eng0, resident=res0)              # slot 0
    ids = p1 + pout + tokens(15, seed=86)
    out, pre = wturn(ids, 6, "chat-p")
    return ids, out, pre


def test_a_parent_on_a_side_slot_goes_to_the_nvme_before_its_worker_takes_the_slot():
    d = tempfile.mkdtemp()
    try:
        ids, out, pre = _side_parent(app._keep_parent, d)
        assert _tail(pre, ids) <= 15 + 1, pre
        assert out == cold(ids, 6), "read back from the NVMe: the cold prefill's tokens"
    finally:
        shutil.rmtree(d, ignore_errors=True)
    d = tempfile.mkdtemp()
    try:
        ids, _, pre = _side_parent(None, d)                     # the old side slot
        assert pre["reused"] == 0, pre
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_side_slots_rows_of_a_session_that_is_no_parent_are_not_written():
    d = tempfile.mkdtemp()
    try:
        _state()
        sp, _ = _spill(d, app.STATE["resident"], min_tokens=64)
        side = cache.ResidentPrefix(BIG, CH, end=True, stash_bytes=BIG, guest_frac=1.0)
        side.keeper = app._keep_parent
        app.PARENTS.clear()
        app.STATE.update(engine=fresh(), resident=side)
        wturn(PHEAD + tokens(400, seed=87), 8, "chat-q")
        wturn(PHEAD + tokens(420, seed=88), 6, "chat-r")
        assert not sp.entries, sp.entries
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 4: the park and the memory policy

def _parked_parent(shed, d):
    per = sum(t.numel() // t.shape[ax] * t.element_size()
              for t, ax in cache._kv_views(fresh(), None))
    res = _state(guest_max=8192, park_bytes=900 * per + 8 * fresh().state.nbytes)
    _spill(d, res, min_tokens=64)
    app.PARENTS.clear()
    p1 = PHEAD + tokens(300, seed=89)
    pout, _ = wturn(p1, 8, "chat-p")
    app._note_parent("chat-p")
    wturn(WHEAD + tokens(40, seed=90), 6, "chat-p-task-1", stay=True)
    assert res.park_owner == "chat-p" and res.park_rows, (res.park_owner, res.park_rows)
    shed(res)                                                  # the gate at the ceiling
    assert not res.park_rows
    ids = p1 + pout + tokens(15, seed=91)
    out, pre = wturn(ids, 6, "chat-p")
    return ids, out, pre


def test_a_parked_parent_goes_to_the_nvme_before_the_memory_policy_takes_the_park():
    d = tempfile.mkdtemp()
    try:
        ids, out, pre = _parked_parent(lambda r: app._shed_park(r), d)
        assert _tail(pre, ids) <= 15 + 1, pre
        assert out == cold(ids, 6)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    d = tempfile.mkdtemp()
    try:
        ids, _, pre = _parked_parent(lambda r: r.shed_park(), d)     # the old shedder
        assert _tail(pre, ids) > 300, pre
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_on_a_side_slots_thread_the_park_goes_as_before():
    class Res:
        park_rows, park_base, park_owner, _stash = 300, 40, "chat-p", None
        spilled = 0

        def _spill_park(self, eng, drafter):
            self.spilled += 1
            return True

        def shed_park(self, safe=False):
            self.park_rows = 0
            return 1

    app.PARENTS.clear()
    app._note_parent("chat-p")
    r = Res()
    slots_mod.bind(slots_mod.Slot(1, 1024))
    try:
        assert app._shed_park(r) == 1 and r.spilled == 0
    finally:
        slots_mod.bind(None)
    r = Res()
    assert app._shed_park(r) == 1 and r.spilled == 1


# ------------------------------------------------------------------ 5: the slot choice

class FakeRes:
    """Slot 0's resident as `_par_side_need` reads it."""

    def __init__(self, toks, owner=None, chunk=CH, park=0):
        self.tokens, self.valid, self.chunk, self.owner = list(toks), len(toks), chunk, owner
        self.guest_max, self.end, self.park_rows, self._park = 8192, None, park, park

    def common(self, ids):
        return app._shared_head(ids, self.tokens[:self.valid])

    def park_offer(self, ids, c, max_len, align=True):
        return self._park


def _need(res, ids, **kw):
    pool = slots_mod.Pool([slots_mod.Slot(0, 262_144), slots_mod.Slot(1, 262_144)])
    saved = {k: app.STATE.get(k) for k in ("pool", "resident", "slot0_view", "prefix_chunk",
                                           "spill")}
    app.STATE.update(pool=pool, resident=res, prefix_chunk=CH, spill=None)
    app.STATE.pop("slot0_view", None)
    real = app.build_prompt
    app.build_prompt = lambda body, client=None, *, count=False, left_out=0: list(ids)

    class Plan:
        active = True
    try:
        why: dict = {}
        need = app._par_side_need({"messages": []}, None, 0, 4096, Plan(), [], None, why, **kw)
        return need, why
    finally:
        app.build_prompt = real
        for k, v in saved.items():
            if v is None:
                app.STATE.pop(k, None)
            else:
                app.STATE[k] = v


def test_the_worker_headers():
    assert app._worker_of({"X-Tandem-Worker": "1", "X-Tandem-Parent-Session": "chat-a"}) == \
        (True, "chat-a")
    assert app._worker_of({"X-Tandem-Parent-Session": "chat-a"}) == (True, "chat-a")
    assert app._worker_of({"X-Tandem-Worker": "1"}) == (True, None)
    assert app._worker_of({"x-session-id": "chat-a"}) == (False, None)


def test_a_worker_is_no_guest_of_slot_0_and_never_waits_for_a_side_slot():
    other = FakeRes(tokens(24_200, seed=1), owner="chat-other")
    ids = tokens(2_532, seed=2)
    _, why = _need(other, ids)
    assert why.get("guest"), why                       # the old rule: a guest of the other chat
    _, why = _need(other, ids, stay=True, parent="chat-p")
    assert not why.get("guest") and not why.get("prefer_side_wait"), why


def test_a_worker_beside_its_parent_on_slot_0_takes_a_side_slot_first():
    parent = FakeRes(tokens(9_000, seed=3), owner="chat-p")
    _, why = _need(parent, tokens(2_532, seed=4), stay=True, parent="chat-p")
    assert why.get("prefer_side") == 9_000 and not why.get("prefer_side_wait"), why
    _, why = _need(parent, tokens(2_532, seed=4), stay=True, parent="chat-q")
    assert not why.get("prefer_side"), why


def test_a_conversation_slot_0_has_parked_tries_slot_0_first():
    _, why = _need(FakeRes(tokens(12_000, seed=5), park=11_000), tokens(11_500, seed=6))
    assert why.get("wait") == "park" and why.get("slot0_first") and why["park"] == 11_000, why
    _, why = _need(FakeRes(tokens(12_000, seed=5)), tokens(11_500, seed=6))
    assert why.get("wait") is None, why


def test_a_parent_waits_for_its_worker_for_parent_keep_s():
    app.PARENTS.clear()
    app._note_parent("chat-p", now=100.0)
    assert app._waiting_parent("chat-p", now=100.0 + app.PARENT_KEEP_S - 1)
    assert not app._waiting_parent("chat-p", now=100.0 + app.PARENT_KEEP_S)
    assert not app._waiting_parent(None) and not app._waiting_parent("chat-q", now=100.0)
    app._note_parent("chat-q", now=100.0 + app.PARENT_KEEP_S + 1)
    assert "chat-p" not in app.PARENTS                 # a stale entry goes


if __name__ == "__main__":
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            with torch.no_grad():
                fn()
            print(f"  {name:90s} ok")
            passed += 1
    print(f"{passed} passed")
