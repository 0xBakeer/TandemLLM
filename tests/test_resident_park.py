"""The memory limits on the random 4-layer model (OPS-47, QA OPS-48): the parked session, the caps,
the shed, and the claim that none of it changes a token.

Every resume here lands on the chunk grid, so a resumed request is held to bit equality with a cold
prefill of its whole prompt, as in tests/test_resident.py and tests/test_resident_pin.py.

  * A, B, A: two sessions that share their head (opencode's tool block) and nothing after it. B
    replaces A's rows; A's next turn swaps them back from the park; B's next turn swaps again.
  * the caps: a long run under small budgets never holds more than each cap.
  * shed then resume: every level given back, and the next turn is the same bytes as the run that
    gave nothing back, and as a cold prefill.
  * a refused allocation (the memory policy says no) leaves the cold path, never a wrong state.
  * the trace replay: one opencode-shaped trace with the caches off, small and large; the token
    streams are identical, only where each prefill started differs (the plan's gate a).

Run: python tests/test_resident_park.py
"""

from __future__ import annotations

import os
import sys

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import cache  # noqa: E402
from test_cache import DEV, fresh, tokens  # noqa: E402
from test_resident import cold, decode, same_state  # noqa: E402

CH = 16
BIG = 1 << 30
HEAD = tokens(130, seed=1)                   # the tool block and system text both sessions share


def per_row(eng) -> int:
    return sum(t.numel() // t.shape[ax] * t.element_size() for t, ax in cache._kv_views(eng, None))


def resident(eng, park_rows: int = 400, park_anchors: int = 4, **kw):
    park = park_rows * per_row(eng) + park_anchors * eng.state.nbytes if park_rows else 0
    kw.setdefault("stash_bytes", BIG)
    return cache.ResidentPrefix(kw.pop("budget", BIG), CH, park_bytes=park,
                                park_anchors=park_anchors, **kw)


def turn(eng, res, ids, session, **kw):
    info = {}
    lg, reused, fwd = cache.prefill(eng, None, ids, DEV, chunk=CH, resident=res, conv_id=session,
                                    info=info, **kw)
    return lg, reused, info


def grow(ids, seed):
    """The next turn: the prompt, an answer, a tool result (the rows the answer's decode wrote
    are a `decode` call of the test's own, where it ran)."""
    return ids + tokens(7, seed=seed) + tokens(11, seed=seed + 1)


# ------------------------------------------------------------------ 6: A, B, A

def test_a_session_switch_back_swaps_the_parked_rows_in_bit_for_bit():
    eng = fresh()
    res = resident(eng)
    a = HEAD + tokens(150, seed=2)
    turn(eng, res, a, "A")
    a = grow(a, 10)
    turn(eng, res, a, "A")
    decode(eng, len(a), k=5, seed=12)
    b = HEAD + tokens(140, seed=3)                   # another session, the same head
    _, reused, info = turn(eng, res, b, "B")
    assert reused == 128 and info["kind"] == "resident", (reused, info)
    assert res.stats["parks"] == 1 and res.park_owner == "A" and res.park_base == 128
    decode(eng, len(b), k=5, seed=13)
    a2 = grow(a, 14)                            # A again: from the park, not re-read
    warm, reused, info = turn(eng, res, a2, "A")
    top = max(x for x in range(CH, len(a), CH))      # A's newest anchor below its prompt
    assert reused == top and info["kind"] == "resident", (reused, top)
    assert res.stats["park_hits"] == 1 and res.stats["park_swaps"] == 1
    assert res.park_owner == "B", "B was parked in A's place"
    ref, lg = cold(a2)
    assert torch.equal(warm, lg), (warm - lg).abs().max().item()
    same_state(eng, ref, len(a2))
    decode(eng, len(a2), k=5, seed=15)
    b2 = grow(b, 16)                            # and B again, swapped back too
    warm, reused, _ = turn(eng, res, b2, "B")
    assert reused == max(x for x in range(CH, len(b), CH)) and res.stats["park_hits"] == 2
    ref, lg = cold(b2)
    assert torch.equal(warm, lg)
    same_state(eng, ref, len(b2))
    sw = res.switches[-1]
    assert sw["from"] == "A" and sw["to"] == "B" and sw["via"] == "park" and sw["reread"] > 0


def test_a_park_smaller_than_the_session_holds_what_fits_and_still_resumes_exactly():
    eng = fresh()
    res = resident(eng, park_rows=70, park_anchors=2)
    a = HEAD + tokens(200, seed=4)
    turn(eng, res, a, "A")
    decode(eng, len(a), k=5, seed=20)
    turn(eng, res, HEAD + tokens(100, seed=5), "B")
    assert res.park_base == 128 and res.park_rows == 64, res.report()     # 192 is the top that fits
    assert res.park_bytes <= res.park_budget
    warm, reused, _ = turn(eng, res, grow(a, 21), "A")
    assert reused == 192
    ref, lg = cold(grow(a, 21))
    assert torch.equal(warm, lg)


def test_a_parked_prompt_is_not_pinned_too():
    """The park keeps the rows above the shared head and the head stays in place: pinning it as
    well held the same session twice (3.8 GiB beside 4 GiB in the 2026-10-01 gate)."""
    eng = fresh()
    res = resident(eng, pin_bytes=BIG)
    turn(eng, res, HEAD + tokens(150, seed=34), "A")
    turn(eng, res, HEAD + tokens(140, seed=35), "B")
    assert res.stats["parks"] == 1 and res.stats["pins"] == 0 and res.pin_rows == 0
    turn(eng, res, [0] + tokens(260, seed=36), "B", compacting=True)      # B's compaction: pinned
    assert res.stats["pins"] == 1 and res.stats["parks"] == 1


def test_the_same_session_parting_from_itself_is_not_parked():
    eng = fresh()
    res = resident(eng)
    a = HEAD + tokens(150, seed=6)
    turn(eng, res, a, "A")
    turn(eng, res, HEAD + tokens(150, seed=7), "A")  # an edited earlier message: A again
    assert res.stats["parks"] == 0


def test_a_compaction_does_not_park_and_drops_its_parked_rows():
    eng = fresh()
    res = resident(eng)
    a = HEAD + tokens(150, seed=8)
    turn(eng, res, a, "A")
    turn(eng, res, [0] + tokens(260, seed=9), "A", compacting=True)
    assert res.stats["parks"] == 0
    turn(eng, res, a, "A")
    turn(eng, res, HEAD + tokens(120, seed=30), "B")            # A parked
    assert res.park_owner == "A"
    assert res.drop_session("A") > 0 and res.park_rows == 0 and res.stats["park_dropped"] == 1
    assert res.drop_session("B") == 0


def test_without_session_ids_a_prompt_that_parts_from_the_rows_is_another_session():
    eng = fresh()
    res = resident(eng)
    a = HEAD + tokens(150, seed=31)
    turn(eng, res, a, None)
    turn(eng, res, HEAD + tokens(120, seed=32), None)
    assert res.stats["parks"] == 1 and res.stats["switches"] == 1
    warm, reused, _ = turn(eng, res, grow(a, 33), None)
    assert res.stats["park_hits"] == 1
    ref, lg = cold(grow(a, 33))
    assert torch.equal(warm, lg)


# ------------------------------------------------------------------ 1: the caps hold

def test_every_cache_stays_under_its_cap_over_a_long_interleaved_run():
    eng = fresh()
    ab = eng.state.nbytes
    pr = per_row(eng)
    res = cache.ResidentPrefix(5 * ab, CH, stash_bytes=40 * pr, guest_max=40,
                               pin_bytes=100 * pr + 3 * ab, pin_anchors=3,
                               park_bytes=90 * pr + 2 * ab, park_anchors=2)
    store = cache.StateStore(3 * ab + 200 * pr, chunk=CH, max_entry_bytes=ab + 150 * pr,
                             per_prefill=1)
    sess = {"A": HEAD + tokens(60, seed=40), "B": HEAD + tokens(70, seed=41)}
    for t in range(6):
        for s in ("A", "B"):
            sess[s] = sess[s] + tokens(25, seed=50 + t * 2 + (s == "B"))
            cache.prefill(eng, None, sess[s], DEV, store=store, chunk=CH, checkpoint=True,
                          resident=res, conv_id=s)
            decode(eng, len(sess[s]), k=4, seed=70 + t)
            cache.prefill(eng, None, tokens(20, seed=90 + t), DEV, store=store, chunk=CH,
                          checkpoint=True, resident=res, conv_id="title")   # a guest
            assert res.anchor_bytes <= 5 * ab, res.report()
            assert res.stash_bytes <= 40 * pr and res.pin_bytes <= 100 * pr + 3 * ab
            assert res.park_bytes <= 90 * pr + 2 * ab and store.bytes <= store.budget


# ------------------------------------------------------------------ shed then resume

def _filled(eng):
    """Caches holding something at every level, from the same requests: A's compaction pins its
    head, B parks A's next prompt, a title call stashes B's rows, the store has prefix entries."""
    res = resident(eng, pin_bytes=200 * per_row(eng) + 8 * eng.state.nbytes)
    store = cache.StateStore(BIG, chunk=CH, per_prefill=1)

    def run(ids, s, **kw):
        cache.prefill(eng, None, ids, DEV, store=store, chunk=CH, checkpoint=True, resident=res,
                      conv_id=s, **kw)
    a = HEAD + tokens(150, seed=60)
    run(a, "A")
    a = grow(a, 61)
    run(a, "A")
    decode(eng, len(a), k=4, seed=62)
    run([0] + tokens(250, seed=63), "A", compacting=True)               # pins A's head
    decode(eng, 251, k=4, seed=64)
    a = HEAD + tokens(60, seed=65)                                       # the turn after it
    run(a, "A")
    run(HEAD + tokens(140, seed=66), "B")                                # parks A
    decode(eng, 270, k=4, seed=67)
    run(tokens(20, seed=68), "title")                                    # a guest: a stash
    return res, store, a


def test_shed_every_level_then_the_next_turn_is_the_same_bytes():
    eng1, eng2 = fresh(), fresh()
    res1, store1, a1 = _filled(eng1)
    res2, store2, a2 = _filled(eng2)
    assert a1 == a2 and res1.pin_rows and res1.park_rows and res1._stash is not None
    assert res1.pin_bytes <= res1.pin_budget and res1.park_bytes <= res1.park_budget
    assert store1.bytes > 0
    freed = store2.shed() + res2.shed_anchors() + res2.shed_stash() + res2.shed_park() \
        + res2.shed_pin()
    # what is left is the anchors a level never takes: the newest few and the head lines
    assert freed > 0 and store2.bytes == 0
    assert res2.stash_bytes == res2.pin_bytes == res2.park_bytes == 0
    assert res2.bytes == len(res2.anchors) * eng2.state.nbytes and not res2._free
    assert len(res2.anchors) <= res2.tail + res2.pin_anchor_max
    nxt = grow(a1, 66)
    w1, r1, _ = cache.prefill(eng1, None, nxt, DEV, store=store1, chunk=CH, checkpoint=True,
                              resident=res1, conv_id="A")
    w2, r2, _ = cache.prefill(eng2, None, nxt, DEV, store=store2, chunk=CH, checkpoint=True,
                              resident=res2, conv_id="A")
    assert r1 > 0 and r2 == 0, (r1, r2)              # warm from the park against cold
    assert torch.equal(w1, w2), (w1 - w2).abs().max().item()
    same_state(eng1, eng2, len(nxt))
    ref, lg = cold(nxt)
    assert torch.equal(w1, lg)


def test_a_safe_shed_never_touches_rows_in_use():
    eng = fresh()
    res, store, a = _filled(eng)
    pending = res._stash is not None
    assert pending and res.shed_stash(safe=True) == 0 and res._stash is not None
    anchors = dict(res.anchors)
    res.shed_anchors(safe=True)
    assert res.anchors == anchors, "a safe shed frees only the free list"


# ------------------------------------------------------------------ the memory policy says no

class CacheProbe:
    """An allocator that is the caches and nothing else, with a ceiling the test sets."""

    def __init__(self, res, store, ceiling):
        self.res, self.store, self.ceiling = res, store, ceiling

    def probe(self):
        n = self.res.bytes + (self.store.bytes if self.store is not None else 0)
        return n, n


def _policy(res, store, ceiling):
    p = CacheProbe(res, store, ceiling)
    mem = cache.Budget(ceiling, ceiling * 4, 0, probe=p.probe, empty=lambda: None,
                       peak_probe=lambda reset=False: 0)
    res.mem = mem
    if store is not None:
        mem.register("store", store.shed)
    for lv in ("anchors", "stash", "park", "pin"):
        mem.register(lv, getattr(res, f"shed_{lv}"))
    return mem


def test_refused_allocations_take_the_cold_path_and_stay_exact():
    eng = fresh()
    ab = eng.state.nbytes
    res = resident(eng)
    store = cache.StateStore(BIG, chunk=CH, per_prefill=1)
    mem = _policy(res, store, 4 * ab)                # room for four anchors and nothing else
    a = HEAD + tokens(150, seed=70)
    cache.prefill(eng, None, a, DEV, store=store, chunk=CH, checkpoint=True, resident=res,
                  conv_id="A", mem=mem)
    assert res.stats["refused_anchor"] > 0 and res._allocated <= 4
    assert mem.stats["admit_refused"] > 0
    decode(eng, len(a), k=4, seed=71)
    b = HEAD + tokens(140, seed=72)
    cache.prefill(eng, None, b, DEV, store=store, chunk=CH, checkpoint=True, resident=res,
                  conv_id="B", mem=mem)
    assert res.park_rows == 0 and res.stats["refused_park"] >= 1
    decode(eng, len(b), k=4, seed=73)
    g = tokens(20, seed=74)                          # a guest whose stash is refused: rows cut
    cache.prefill(eng, None, g, DEV, store=store, chunk=CH, checkpoint=True, resident=res,
                  conv_id="t", mem=mem)
    assert res.stats["refused_stash"] >= 1
    for ids, s in ((grow(a, 75), "A"), (b + tokens(9, seed=76), "B")):
        warm, _, _ = cache.prefill(eng, None, ids, DEV, store=store, chunk=CH, checkpoint=True,
                                   resident=res, conv_id=s, mem=mem)
        ref, lg = cold(ids)
        assert torch.equal(warm, lg)
        same_state(eng, ref, len(ids))


# ------------------------------------------------------------------ 9: the trace replay

def _greedy(eng, logits, n_prompt, k):
    out, tok, pos = [], int(logits[0, -1].argmax()), n_prompt
    for _ in range(k):
        out.append(tok)
        lg = eng.forward(torch.tensor([tok], device=DEV), start=pos, last_only=True)
        pos += 1
        tok = int(lg[0, -1].argmax())
    return out


def _trace():
    """The night of 2026-09-30, scaled down: one session growing turn by turn with a short title
    call between, its compaction and the turn after it, and a second session in between."""
    a = HEAD + tokens(90, seed=80)
    b = HEAD + tokens(70, seed=81)
    return [("A", a, 6, False), ("title", tokens(18, seed=82), 3, False),
            ("A", None, 6, False), ("B", b, 6, False), ("A", None, 6, False),
            ("B", None, 5, False), ("A", None, 5, False),
            ("A", [0] + tokens(200, seed=83), 4, True),          # the compaction
            ("A", HEAD + tokens(25, seed=84), 6, False), ("B", None, 4, False)]


def _replay(make):
    eng = fresh()
    res, store = make(eng)
    sess, streams, starts = {}, [], []
    for i, (s, ids, k, comp) in enumerate(_trace()):
        if ids is None:
            ids = sess[s] + streams[sess[s + ":last"]] + tokens(9, seed=100 + i)
        if comp and store is not None:
            store.drop_session(s)
        if comp and res is not None:
            res.drop_session(s)
        lg, reused, _ = cache.prefill(eng, None, ids, DEV, store=store, chunk=CH,
                                      checkpoint=store is not None, resident=res, conv_id=s,
                                      compacting=comp)
        streams.append(_greedy(eng, lg, len(ids), k))
        starts.append(reused)
        if s != "title":
            sess[s], sess[s + ":last"] = ids, len(streams) - 1
    return streams, starts


def test_the_trace_gives_the_same_tokens_with_the_caches_off_small_and_large():
    def off(eng):
        return None, None

    def small(eng):                  # the "2 GiB" run: everything capped, the park too short
        ab, pr = eng.state.nbytes, per_row(eng)
        return (cache.ResidentPrefix(5 * ab, CH, stash_bytes=30 * pr, guest_max=40,
                                     pin_bytes=80 * pr + 2 * ab, pin_anchors=2,
                                     park_bytes=40 * pr + 2 * ab, park_anchors=2),
                cache.StateStore(2 * ab, chunk=CH, max_entry_bytes=ab + 60 * pr, per_prefill=1))

    def large(eng):                  # the "8 GiB" run: nothing is ever short of room
        return (cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, pin_bytes=BIG, park_bytes=BIG),
                cache.StateStore(BIG, chunk=CH, per_prefill=1))

    s0, r0 = _replay(off)
    s2, r2 = _replay(small)
    s8, r8 = _replay(large)
    assert s0 == s2 == s8, "a cache limit changed a token"
    assert sum(r0) == 0 and sum(r8) > sum(r2) > 0, (r0, r2, r8)


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()
