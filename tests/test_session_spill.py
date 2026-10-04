"""Live sessions on the NVMe (engine/spill.py, 0.3.22-test), on the random 4-layer model.

THE CLAIM. A conversation written to the disk and read back is the conversation that was in the
engine, byte for byte: every row of every position-indexed buffer, the recurrent state and the
convolution tails of the end anchor and of the saved grid anchors, the tokens, the `taint` line,
the hidden spans and the session. The next turn after a restore therefore resumes where it would
have resumed had the rows never left (the end anchor), and writes the same greedy tokens.

  1  a planned stop: the drain writes the resident conversation, a new process (a new engine on the
     same weights, a reopened spill) reads it back before its first request; rows, anchors and the
     end anchor equal the old engine's bit for bit, and the next turn resumes at the end with the
     same tokens as without the restart
  2  an eviction: another session's request that is not a guest writes the conversation to the disk
     first (it is not parked); the conversation's next turn reads it back (`spill`) and writes the
     same tokens as with no other request in between; the other session is written in turn when
     it is long enough
  3  incremental by content: written again a turn later, only the blocks that changed are written;
     read back over rows that hold the file's bytes, only the others are read
  4  the hidden spans: the handler's hook reads the conversation back before the replay, so a copy
     without the hidden phrase resumes at the end anchor
  5  a file that does not hold what was written fails the restore (its sums), the entry goes, and
     the prefill is a cold one; an entry of another fingerprint is swept at open
  6  the budget: least recently used first; a conversation that does not fit is not written
  7  the parked conversation is written at the drain with its head from the rows in place

Run: python tests/test_session_spill.py
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

from engine import cache, disk_cache, spill  # noqa: E402
from server import app  # noqa: E402
from test_cache import DEV, fresh, tokens  # noqa: E402
from test_end_anchor import CH, BIG, TOOL, TOOL2, _state, turn, _hidden_turn, PHRASE  # noqa: E402

HUGE = 10 ** 15


def _spill(d, res, *, fp="t", budget_gb=1.0, min_tokens=64, free=None):
    t = disk_cache.DiskTier(os.path.join(d, "tier"), chunk=CH, fingerprint="tier", min_tokens=CH,
                            free_bytes=lambda p: HUGE, settings={"mode": "off"}).open()
    sp = spill.SessionSpill(os.path.join(d, "tier", "sessions"), chunk=CH, fingerprint=fp,
                            tier=t, settings={"budget_gb": budget_gb, "min_tokens": min_tokens,
                                              "drain_min_tokens": CH},
                            free_bytes=free or (lambda p: HUGE), floor_bytes=0).open()
    res.spiller = sp
    app.STATE["spill"] = sp
    return sp, t


def _rows(eng, n):
    return [t.narrow(ax, 0, n).clone() for t, ax in cache._kv_views(eng, None)]


def _same_rows(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def _restart(d, fp="t", **kw):
    """A new process: a new engine on the same weights, a new resident prefix, the spill reopened
    from its directory; the drained conversation read back as `spill_startup` does."""
    res = _state(**kw)
    sp, t = _spill(d, res, fp=fp)
    app.spill_startup()
    return res, sp, t


# ------------------------------------------------------------------ 1. a planned stop

def test_a_drained_conversation_comes_back_after_a_restart_bit_for_bit():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        prompt = tokens(150, seed=70)
        # the reference: no restart between the two turns
        _state()
        out1, _, _ = turn(prompt, 60)
        out2_ref, pre_ref, _ = turn(prompt + out1 + TOOL, 24)
        assert pre_ref["kind"] == "end"
        # the run: turn 1, the drain, a restart, turn 2
        res = _state()
        sp, t = _spill(d, res)
        out1b, _, ctx1 = turn(prompt, 60)
        assert out1b == out1
        e, taint = res.end["at"], res.taint
        eng0 = app.STATE["engine"]
        rows0 = _rows(eng0, e)
        end0 = tuple(x.clone() for x in res._end_bufs)
        anchors0 = {b: (S.clone(), c.clone()) for b, (S, c) in res.anchors.items()}
        app.spill_drain()
        assert sp.stats["saves"] == 1 and sp.drained() is not None
        t.close()
        res2, sp2, t2 = _restart(d)
        eng1 = app.STATE["engine"]
        assert eng1 is not eng0
        assert res2.valid == e and res2.end["at"] == e and res2.taint == taint
        assert res2.tokens == res.tokens[:e]
        _same_rows(_rows(eng1, e), rows0)
        assert torch.equal(res2._end_bufs[0], end0[0]) and torch.equal(res2._end_bufs[1], end0[1])
        kept = sorted(anchors0)[-4:]
        assert sorted(res2.anchors) == kept
        for b in kept:
            assert torch.equal(res2.anchors[b][0], anchors0[b][0])
            assert torch.equal(res2.anchors[b][1], anchors0[b][1])
        out2, pre2, _ = turn(prompt + out1 + TOOL, 24)
        assert pre2["kind"] == "end" and pre2["reused"] == e, pre2
        assert out2 == out2_ref
        t2.close()
        return f"{e} tokens drained and read back bit for bit; turn 2 resumed at the end, same 24 tokens"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 2. an eviction

def _evict_run(with_spill: bool, other_len=200, d=None):
    res = _state(park_bytes=0, guest_max=40)
    sp = t = None
    if with_spill:
        sp, t = _spill(d, res)
    prompt = tokens(220, seed=71)
    out1, _, _ = turn(prompt, 40, session="A")
    e = res.end["at"]
    other = tokens(other_len, seed=72)
    turn(other, 8, session="B")                     # not a guest: longer than guest_max
    out2, pre2, _ = turn(prompt + out1 + TOOL, 16, session="A")
    return res, sp, t, e, out2, pre2


def test_another_sessions_request_writes_the_conversation_first_and_it_comes_back():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        _, _, _, e0, ref, pre_ref = _evict_run(False)
        assert pre_ref["reused"] < e0, "without the spill the conversation is re-read"
        res, sp, t, e, out2, pre2 = _evict_run(True, d=d)
        assert e == e0
        assert res.stats["spills"] >= 1 and sp.stats["saves"] >= 2, (res.stats, sp.stats)
        assert pre2["kind"] == "spill" and pre2["reused"] == e, pre2
        assert out2 == ref
        assert res.stats["parks"] == 0
        # B (208 tokens, over min_tokens) was written in turn when A came back
        owners = sorted(x.man["owner"] for x in sp.entries.values())
        assert owners == ["A", "B"], owners
        t.close()
        return f"evicted at {e}, read back by its next turn ({pre2['reused']} reused, was {pre_ref['reused']})"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_session_snapshot_does_not_drop_a_long_conversation():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        store = cache.StateStore(BIG, chunk=CH)
        res = _state(store=store)
        sp, t = _spill(d, res)
        prompt = tokens(220, seed=73)
        out1, _, _ = turn(prompt, 30, session="A")
        e = res.end["at"]
        res._spill(app.STATE["engine"], None, "evict")
        assert sp.stats["saves"] == 1
        res.clear()                                   # what a session snapshot's restore does
        out2, pre2, _ = turn(prompt + out1 + TOOL, 8, session="A")
        assert pre2["kind"] == "spill" and pre2["reused"] == e
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 3. incremental

def test_written_again_only_the_changed_blocks_are_written_and_read():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res = _state()
        sp, t = _spill(d, res)
        prompt = tokens(200, seed=74)
        out1, _, _ = turn(prompt, 30, session="A")
        eng = app.STATE["engine"]
        first = sp.save_resident(res, eng, None, "evict")
        n1 = first["blocks_written"]
        out2, _, _ = turn(prompt + out1 + TOOL, 20, session="A")
        second = sp.save_resident(res, eng, None, "evict")
        assert second["blocks_kept"] >= (len(prompt) + len(out1)) // CH - 2, second
        assert second["blocks_written"] < n1, (first, second)
        # read back over rows that hold the same bytes: nothing of them is read
        info = sp.restore(res, eng, None, sp.entries[second["eid"]])
        assert info["blocks_read"] == 0 and info["blocks_in_place"] == len(
            sp.entries[second["eid"]].man["blocks"]), info
        t.close()
        return f"{n1} blocks, then {second['blocks_written']} written and {second['blocks_kept']} kept"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 4. the hidden spans

def test_the_handlers_hook_reads_it_back_before_the_replay():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res = _state()
        sp, t = _spill(d, res)
        prompt = tokens(100, seed=62)
        out, ctx = _hidden_turn(prompt, session="A")
        e = res.end["at"]
        copy = prompt + out + TOOL                    # the client's copy: no phrase
        sp.save_resident(res, app.STATE["engine"], None, "evict")
        res.clear()
        for tt, ax in cache._kv_views(app.STATE["engine"], None):
            tt.fill_(0)
        app._spill_before_replay(copy)
        assert res.end is not None and res.end["at"] == e and res.end["hidden"]
        ids, spans = res.replay(copy)
        assert spans and ids == ctx + TOOL
        _, pre, _ = turn(ids, 8, session="A", hidden=spans)
        assert pre["kind"] == "spill" and pre["reused"] == e, pre
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 5. a bad file

def test_a_file_that_fails_its_sums_fails_the_restore_and_the_prefill_is_cold():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res = _state()
        sp, t = _spill(d, res)
        prompt = tokens(200, seed=75)
        out1, _, _ = turn(prompt, 20, session="A")
        info = sp.save_resident(res, app.STATE["engine"], None, "evict")
        e = sp.entries[info["eid"]]
        blk = os.path.join(e.path, "r0000032.blk")
        with open(blk, "r+b") as f:
            f.seek(spill.ALIGN + 100)
            b = f.read(1)
            f.seek(spill.ALIGN + 100)
            f.write(bytes([b[0] ^ 0xFF]))
        res.clear()
        _, pre, _ = turn(prompt + out1 + TOOL, 4, session="A")
        assert pre["kind"] is None and pre["reused"] == 0, pre
        assert sp.stats["failed"].get("sums") == 1 and not sp.entries, (sp.stats, sp.events)
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_another_fingerprint_is_swept_and_an_incomplete_entry_too():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res = _state()
        sp, t = _spill(d, res)
        out1, _, _ = turn(tokens(200, seed=76), 20, session="A")
        sp.save_resident(res, app.STATE["engine"], None, "drain")
        os.makedirs(os.path.join(sp.root, "deadbeef"), mode=0o700)   # a crash before its manifest
        t.close()
        res2 = _state()
        sp2, t2 = _spill(d, res2, fp="other")
        assert not sp2.entries and sp2.stats["swept"] == 2
        assert os.listdir(sp2.root) == []
        t2.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 6. the budget

def test_least_recently_used_goes_first_and_what_does_not_fit_is_not_written():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res = _state()
        eng = app.STATE["engine"]
        sp, t = _spill(d, res)
        out_a, _, _ = turn(tokens(200, seed=77), 20, session="A")
        first = sp.save_resident(res, eng, None, "evict")
        sp.cfg["budget_gb"] = 1.5 * first["bytes"] / 1e9
        out_b, _, _ = turn(tokens(200, seed=78), 20, session="B")
        sp.save_resident(res, eng, None, "evict")
        owners = sorted(x.man["owner"] for x in sp.entries.values())
        assert owners == ["B"], owners
        assert sp.stats["evicted"].get("budget") == 1
        sp.cfg["budget_gb"] = 1e-6
        _, _, _ = turn(tokens(200, seed=79), 20, session="C")
        before = sp.stats["refused"].get("room", 0)        # B's own eviction was refused too
        assert sp.save_resident(res, eng, None, "evict") is None
        assert sp.stats["refused"].get("room") == before + 1
        assert all(x.man["owner"] != "C" for x in sp.entries.values())
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 7. the park

def test_the_parked_conversation_is_written_with_its_head_from_the_rows_in_place():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res = _state(park_bytes=BIG, guest_max=10)
        sp, t = _spill(d, res, min_tokens=10 ** 6)     # evictions park in RAM here
        head = tokens(64, seed=80)
        a = head + tokens(150, seed=81)
        out_a, _, _ = turn(a, 10, session="A")
        b = head + tokens(140, seed=82)
        turn(b, 10, session="B")
        assert res.park_rows and res.park_owner == "A"
        eng = app.STATE["engine"]
        top = res.park_base + res.park_rows
        want = []
        for (tt, ax), buf in zip(cache._kv_views(eng, None), res._park_bufs):
            want.append(torch.cat([tt.narrow(ax, 0, res.park_base),
                                   buf.narrow(ax, 0, res.park_rows)], dim=ax).clone())
        info = sp.save_park(res, eng, None, "drain")
        assert info and info["tokens"] == top
        res.clear()
        res.shed_park()
        for tt, ax in cache._kv_views(eng, None):
            tt.fill_(0)
        got = sp.restore(res, eng, None, sp.entries[info["eid"]])
        assert got and res.valid == top and res.owner == "A"
        for (tt, ax), w in zip(cache._kv_views(eng, None), want):
            assert torch.equal(tt.narrow(ax, 0, top), w)
        _, pre, _ = turn(a + out_a + TOOL2, 4, session="A")
        assert pre["kind"] in ("resident", "end"), pre
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 0.3.23-test: its own I/O path

def test_its_own_lock_and_buffers_the_tier_waits_and_unchanged_anchors_stay():
    """0.3.23-test: a save takes neither the tier's lock nor its chunk slots (the tier's writer
    holding its lock does not stop it), the tier's writer waits while it runs, a block's sums over
    every buffer are `block_sums`' integers, and a second save of an unchanged conversation writes
    no anchor again."""
    import threading
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res = _state()
        sp, t = _spill(d, res)
        prompt = tokens(200, seed=76)
        turn(prompt, 30, session="A")
        eng = app.STATE["engine"]
        views = cache._kv_views(eng, None)
        assert spill.block_sums_views(views, 16, 16) == [spill.block_sums(x, ax, 16, 16)
                                                         for x, ax in views]
        seen = []
        real = spill.SessionSpill._tier_hold

        def hold(self, dd):
            real(self, dd)
            seen.append(t.spill_active)
        spill.SessionSpill._tier_hold = hold
        held = threading.Event()
        try:
            with t._io:                      # the tier's writer in the middle of a file
                done = []
                th = threading.Thread(target=lambda: done.append(
                    sp.save_resident(res, eng, None, "evict")))
                th.start()
                th.join(30)
                held.set()
            assert done and done[0] is not None, "the save waited for the tier's lock"
        finally:
            spill.SessionSpill._tier_hold = real
        assert seen == [1, 0] and t.spill_active == 0, seen
        assert not t.buf_chunk, "the tier's chunk slots were not used"
        first = done[0]
        again = sp.save_resident(res, eng, None, "drain")
        assert again["blocks_written"] == 0 and again["bytes_written"] == 0, again
        assert again["bytes"] == first["bytes"], (first, again)
        info = sp.restore(res, eng, None, sp.entries[again["eid"]])
        assert info is not None and info["end"], info
        t.close()
        return f"{first['bytes_written']} bytes, then {again['bytes_written']}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _main():
    import inspect
    fails = 0
    for name, fn in sorted(inspect.getmembers(sys.modules[__name__], inspect.isfunction),
                           key=lambda x: x[1].__code__.co_firstlineno):
        if not name.startswith("test_"):
            continue
        try:
            msg = fn()
            print(f"PASS {name}" + (f": {msg}" if msg else ""))
        except Exception:                                              # noqa: BLE001
            import traceback
            fails += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_main())
