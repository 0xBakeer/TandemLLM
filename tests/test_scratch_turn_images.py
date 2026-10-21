"""SRV-669: a scratch turn that continues a record after images never forwards the images' key ids.

THE CRASH (Ornith 1.5 122B, 0.13.0, 2026-10-21 about 15:00). A Chat-tab conversation of 174,166
tokens with 8 images (8,720 rows) had its end anchor kept after them (`keep_end`, mm). Idle, the
memory writer took its turn on that record: its own memory gate (a plain `Budget.gate`, no hold)
gave the end anchor back at the ceiling (`shed level=end`), and the writer's prefill resumed at an
anchor below the images and forwarded the record's tokens from there. Those tokens are the rows'
keys: an image's rows are key ids at 2**40 and above. The embedding's gather asserted on the device
(`vectorized_gather_kernel index out of bounds`), the CUDA context was lost and every request after
it answered 500. On the CPU the same forward raises `IndexError` in the embedding.

THE FIX. The preparer and the writer hold the end anchor through their gate as a request does
(`_gate_holding_end`); when the gate takes it all the same, a record after images is not continued
(`end_lost`). The compaction served as the continuation holds its end too. And `cache.prefill`
refuses to forward any id outside the vocabulary (`KeyIdsForwarded`) before a row or a state
changes, whoever the caller.

  1  the mechanism: the record after images with its end gone, prefilled as the writer did, reached
     the embedding with a key id; now it raises `KeyIdsForwarded` and the rows stay as they were
  2  the writer with a gate that sheds the end unless it is held: the end is held, the turn resumes
     at it and writes; the rows, anchors and end are as they were
  3  a gate that refuses with the end held: run again without the hold, it takes the end; the
     writer stops (`end_lost`), forwards nothing, and the next look finds no record after images
  4  the preparer, the same gate: `end_lost`, nothing forwarded
  5  a text record whose end went at the gate is still continued from the anchors (no key ids)

Run: python tests/test_scratch_turn_images.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_compaction_continue as C  # noqa: E402  (sets the CPU environment first)
import test_memory_writer as W  # noqa: E402

import torch  # noqa: E402

from engine import cache  # noqa: E402
from server import app, compaction  # noqa: E402
from test_cache import tokens  # noqa: E402

GIB = 1 << 30
KEY = (1 << 40) + 12345                    # an image's key id (engine/vision.py KEY_BASE and up)
D = -7                                     # the rotary offset the images left


def _mm_turn(res, seed=71):
    """A conversation whose end anchor was kept after images: the turn at a rotary offset (the
    end says mm), and an image's key id in the rows between the last anchor below the end and the
    end, as the box's record held its images' 8,720 key ids above its anchors."""
    prompt = tokens(110, seed=seed)
    C._turn(prompt, 40, pos_delta=D, delta=D)
    assert res.end is not None and res.end.get("mm"), res.end
    e = res.end["at"]
    below = [b for b in res.anchors if b < e]
    p = (max(below) if below else 0) + 3
    assert p < e
    res.tokens[p] = KEY
    return p


class _Gate:
    """A `Budget` whose allocator is over its ceiling until the end anchor goes: the gate sheds
    the end (level `end`) unless the request holds it. `refuse_held`: the peak is short by the
    end's bytes too, so a gate with the end held refuses and runs once more without the hold."""

    def __init__(self, res, refuse_held=False):
        self.res, self.st = res, {"reserved": 8 * GIB}
        peak = 8 * GIB if refuse_held else 64 * GIB
        self.mem = cache.Budget(GIB, peak, GIB, probe=lambda: (self.st["reserved"],) * 2,
                                empty=lambda: None, peak_probe=lambda: 0)
        self.mem.register("end", self._shed_end)

    def _shed_end(self, safe):
        freed = self.res.shed_end(safe)
        if freed:
            self.st["reserved"] = 0
        return freed


def _spy():
    """Every prefill's forwarded ids, through `cache.prefill`."""
    seen, real = [], cache.prefill

    def prefill(eng, drafter, ids, *a, **kw):
        out = real(eng, drafter, ids, *a, **kw)
        seen.append({"n": len(ids), "reused": out[1], "forwarded": out[2]})
        return out
    cache.prefill = prefill
    return seen, real


# ------------------------------------------------------------------ 1. the mechanism

def test_a_record_after_images_without_its_end_is_never_forwarded():
    res = C._state(end=True)
    p = _mm_turn(res)
    r = compaction.find_record(res, None, "ses")
    assert r is not None and r.mm and r.delta == D and r.tokens[p] == KEY
    tail = tokens(12, seed=72)
    # what the writer's plain gate did on the box: the end goes, then the turn prefills
    assert res.shed_end() > 0 and res.end is None
    before = (list(res.tokens), res.valid, sorted(res.anchors))
    ids = r.tokens + tail
    try:
        list(app.generate_stream(torch.tensor(ids), 4, set(), None, "ses", pos_delta=r.delta,
                                 end_mm=True, scratch=True, compacting=True))
    except cache.KeyIdsForwarded as exc:
        assert f"row {p}" in str(exc), exc
    else:
        raise AssertionError("the key id was forwarded")
    assert (list(res.tokens), res.valid, sorted(res.anchors)) == before
    # before SRV-669 the same call reached the embedding: on the CPU an IndexError there, on the
    # GPU the device-side assert of the box
    real = cache._no_key_ids
    cache._no_key_ids = lambda *a, **k: None
    try:
        list(app.generate_stream(torch.tensor(ids), 4, set(), None, "ses", pos_delta=r.delta,
                                 end_mm=True, scratch=True, compacting=True))
    except IndexError as exc:
        old = str(exc)
    else:
        old = None
    finally:
        cache._no_key_ids = real
    assert old is not None and "index out of range" in old, old
    return f"refused at row {p} with the rows unchanged; without the guard: IndexError ({old})"


# ------------------------------------------------------------------ 2. the writer holds its end

def _writer(refuse_held):
    res, cond = W._setup()
    _mm_turn(res)
    g = _Gate(res, refuse_held=refuse_held)
    app.STATE["mem"] = g.mem
    app.STATE["tok"] = W._Tok()
    W._idle()
    return res, cond, g


def test_the_writer_holds_the_end_after_images_through_its_gate():
    res, cond, g = _writer(refuse_held=False)
    before = (list(res.tokens), res.valid, dict(res.end), sorted(res.anchors), res.taint)
    seen, real = _spy()
    try:
        got = app.memory_once()
    finally:
        cache.prefill = real
    assert got == "wrote", got
    assert seen and seen[0]["reused"] == before[2]["at"], (seen, before[2])
    assert (list(res.tokens), res.valid, dict(res.end), sorted(res.anchors), res.taint) == before
    assert not res.end_hold
    assert g.mem.stats["shed"]["end"] == 0
    return f"resumed at the end ({seen[0]['reused']}), {seen[0]['forwarded']} rows forwarded"


def test_a_gate_that_takes_the_end_stops_the_writer_after_images():
    res, cond, g = _writer(refuse_held=True)
    seen, real = _spy()
    try:
        got = app.memory_once()
    finally:
        cache.prefill = real
    assert got == "end_lost", got
    assert seen == [], seen                              # nothing was prefilled
    assert g.mem.stats["shed"]["end"] == 1 and res.end is None and not res.end_hold
    st = cond.memory.writer.state("ses")
    assert cond.notes.notes("owner") == [], st
    # the next look: the rows' last prompt holds a key id, which is no record either
    assert app.memory_once() == "nothing"
    return "end_lost, nothing forwarded; the next look: nothing"


# ------------------------------------------------------------------ 4. the preparer

def test_a_gate_that_takes_the_end_stops_the_preparer_after_images():
    res = C._prep_state()
    _mm_turn(res)
    prep = app.STATE["prepared"]
    prep.seen("ses", "opencode")
    app.STATE["tok"] = C._Tok()
    g = _Gate(res, refuse_held=True)
    app.STATE["mem"] = g.mem
    seen, real = _spy()
    try:
        got = app.prepare_once()
    finally:
        cache.prefill = real
    assert got == "end_lost", got
    assert seen == [] and res.end is None and not res.end_hold
    assert prep.stats["prepared"] == 0
    # held and not refused: the summary is written from the end
    res = C._prep_state()
    _mm_turn(res)
    prep = app.STATE["prepared"]
    prep.seen("ses", "opencode")
    app.STATE["tok"] = C._Tok()
    app.STATE["mem"] = _Gate(res).mem
    e = res.end["at"]
    seen, real = _spy()
    try:
        got = app.prepare_once()
    finally:
        cache.prefill = real
    assert got in ("prepared", "length"), got
    assert seen and seen[0]["reused"] == e and res.end is not None, (seen, e)
    return f"end_lost with the end taken; held: {got} from the end ({e})"


# ------------------------------------------------------------------ 5. text records

def test_a_text_record_whose_end_went_is_continued_from_the_anchors():
    res, cond = W._setup()
    C._turn(tokens(110, seed=73), 40)
    assert res.end is not None and not res.end.get("mm")
    g = _Gate(res, refuse_held=True)
    app.STATE["mem"] = g.mem
    app.STATE["tok"] = W._Tok()
    W._idle()
    seen, real = _spy()
    try:
        got = app.memory_once()
    finally:
        cache.prefill = real
    assert got == "wrote", got
    assert seen and seen[0]["reused"] < seen[0]["n"] - 20, seen
    return f"wrote, resumed at {seen[0]['reused']} of {seen[0]['n']}"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            got = fn()
            print(f"  {name:66s} ok  {got or ''}")
            passed += 1
    print(f"{passed} passed")
