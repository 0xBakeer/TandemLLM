"""The run monitor's stall rule (SRV-677, chatd/stall.py).

THE CASE (live 2026-10-21, chat 01M5F3ARQQ4XM4QF4YZTHXKZYB): after a render at step 96 the run
spent steps 97-133 editing exposure, lights and fog in two or three files and rendering again,
each render with no console error and the voxels still near black; the cause was elsewhere. No
call repeated, so the same-call guard never fired.

  1  the screenshot thumbnail: an 8-bit PNG of every filter type decodes to the thumbnail of its
     pixels; anything else is None
  2  the rule: checks whose outcome changes are progress and set the run's pace; a streak of the
     same outcome on a few files steers once, then stops; a picture that changes is progress;
     edits spread over many files never fire
  3  through chatd (tools/chat_trace.py's replay): the note rides in the stored result (every
     request starts with the one before it), the stop ends the run `interrupted` (`stall`), the
     chat reads idle, and Continue goes on with the same streak

The replay of the recorded chat (tools/chat_trace.py with --store on the box) fires at steps 112
(steer) and 133 (stop), and on none of ten other recorded chats.

Run: python tests/test_chatd_stall.py
"""

from __future__ import annotations

import json
import os
import random
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from chatd import stall  # noqa: E402


# ---------------------------------------------------------------------- 1. the thumbnail

def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else b if pb <= pc else c


def png(pixels, w, h, bpp=3, filters=(0, 1, 2, 3, 4)):
    """An 8-bit PNG of `pixels` (rows of w * bpp bytes), each row encoded with the next filter."""
    ctype = {1: 0, 2: 4, 3: 2, 4: 6}[bpp]
    out, prev = bytearray(), bytearray(w * bpp)
    for y in range(h):
        row = pixels[y]
        f = filters[y % len(filters)]
        enc = bytearray()
        for x in range(w * bpp):
            a = row[x - bpp] if x >= bpp else 0
            b, c = prev[x], (prev[x - bpp] if x >= bpp else 0)
            pred = (0, a, b, (a + b) >> 1, _paeth(a, b, c))[f]
            enc.append((row[x] - pred) & 255)
        out += bytes([f]) + enc
        prev = row

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(out))) + chunk(b"IEND", b""))


def _thumb_of(pixels, w, h, bpp, gw=stall.THUMB[0], gh=stall.THUMB[1]):
    sums, cnt = [0] * (gw * gh), [0] * (gw * gh)
    ystep, xstep = max(1, h // (gh * 8)), max(1, w // (gw * 4))
    for y in range(0, h, ystep):
        gy = y * gh // h
        for x in range(0, w, xstep):
            o = x * bpp
            r = pixels[y]
            v = r[o] if bpp < 3 else (r[o] * 3 + r[o + 1] * 6 + r[o + 2]) // 10
            k = gy * gw + x * gw // w
            sums[k] += v
            cnt[k] += 1
    return [s // max(1, c) for s, c in zip(sums, cnt)]


def test_the_thumbnail_of_every_filter_and_colour_type():
    rnd = random.Random(5)
    w, h = 96, 54
    for bpp in (1, 2, 3, 4):
        pixels = [bytearray(rnd.randrange(256) for _ in range(w * bpp)) for _ in range(h)]
        got = stall.thumb(png(pixels, w, h, bpp))
        assert got == _thumb_of(pixels, w, h, bpp), bpp
    assert stall.thumb(b"GIF89a....") is None and stall.thumb(b"") is None
    # flat pictures are the same picture; a picture and its inverse are not
    flat = [bytearray([9] * (w * 3)) for _ in range(h)]
    t = stall.thumb(png(flat, w, h))
    assert stall.correlation(t, t) == 1.0
    grad = [bytearray([(x * 255 // w) for x in range(w) for _ in range(3)]) for _ in range(h)]
    inv = [bytearray(255 - v for v in r) for r in grad]
    g, i = stall.thumb(png(grad, w, h)), stall.thumb(png(inv, w, h))
    assert stall.correlation(g, g) > 0.99 and stall.correlation(g, i) < -0.9
    # brighter is the same picture in kind
    bright = [bytearray(min(255, v // 2 + 100) for v in r) for r in grad]
    assert stall.correlation(g, stall.thumb(png(bright, w, h))) > 0.99


# ---------------------------------------------------------------------- 2. the rule

def _render(w, step, text="no console errors", pic=None):
    return w.observe(step, "render_page", {"port": 8090}, "rendered http://x at 12:00:0" +
                     str(step % 10) + "\n" + text, images=[("image/png", pic)] if pic else ())


def _edit(w, step, path):
    return w.observe(step, "edit_file", {"path": path, "old": "a", "new": "b"}, "edited")


def _pics():
    w, h = 64, 36
    a = png([bytearray([(x * 4) % 256 for x in range(w) for _ in range(3)]) for _ in range(h)], w, h)
    b = png([bytearray([(y * 7) % 256 for _ in range(w * 3)]) for y in range(h)], w, h)
    return a, b


def test_progress_sets_the_pace_and_a_streak_on_few_files_steers_then_stops():
    w = stall.Watch()
    step = 0
    # building: every check fixes an error (progress), every 3 steps
    for k, err in enumerate(("TypeError", "ReferenceError", "SyntaxError", "RangeError",
                             "URIError", "EvalError")):
        step += 2
        _edit(w, step, f"src/f{k}.js")
        step += 1
        assert _render(w, step, f"exception: {err} at line {k}") is None
    assert sorted(w.gaps) == [3] * 5 and w.threshold() == stall.MIN_STEPS
    # the stall: the same outcome, the same two files
    fired = []
    first = step + 1
    assert _render(w, first) is None                      # a new outcome: progress
    for k in range(12):
        for path in ("src/App.js", "src/Atmosphere.js"):
            step += 1
            _edit(w, step, path)
        step += 1
        note = _render(w, step)
        if note:
            fired.append((step, w.events[-1]["level"], note))
    (s1, l1, n1), (s2, l2, n2) = fired
    assert l1 == "steer" and s1 - first >= stall.MIN_STEPS and w.events[0]["checks"] >= 4
    assert "step back" in n1 and "App.js and Atmosphere.js" in n1, n1
    assert l2 == "stop" and s2 - s1 >= stall.MIN_STEPS and w.stop and "stops here" in n2
    # a progress check after it starts over
    step += 1
    assert _render(w, step, "exception: new") is None and w.level == 0 and not fired[2:]


def test_a_picture_that_changes_is_progress_and_many_files_never_fire():
    a, b = _pics()
    w = stall.Watch()
    step = 1
    _render(w, step, pic=a)
    for k in range(20):
        step += 2
        _edit(w, step - 1, "src/App.js")
        assert _render(w, step, pic=a if k % 2 else b) is None    # it looks different each time
    assert len(w.gaps) == 20 and not w.events
    w = stall.Watch()
    _render(w, 1)
    for k in range(30):                                   # one file each: working through a list
        _edit(w, 2 + 2 * k, f"src/part{k}.js")
        assert _render(w, 3 + 2 * k) is None
    assert not w.events
    # a bash command run again is a check; a different command is not
    w = stall.Watch()
    for k in range(3):
        assert w.observe(1 + k, "bash", {"command": f"node --check f{k}.js"}, "ok") is None
    assert w.checks == 0 and not w.gaps
    w.observe(10, "bash", {"command": "npm test"}, "3 failed: parse")
    w.observe(11, "bash", {"command": "npm test"}, "5 failed: parse")    # numbers are no change
    assert w.progress_step is None and w.checks == 1
    w.observe(12, "bash", {"command": "npm test"}, "all passed")
    assert w.progress_step == 12


# ---------------------------------------------------------------------- 3. through chatd

def _call(k, step, name, args, out, images=None):
    return {"type": "tool_call", "id": f"c{k}", "name": name, "step": step, "status": "done",
            "input": json.dumps(args),
            "output": {"content": out, "is_error": False, "images": images or []}}


def _stalled_trace(cycles=14):
    parts, step, k = [], 0, 0
    for c in range(cycles):
        for path in ("/workspace/reef/App.js", "/workspace/reef/Atmosphere.js"):
            step += 1
            k += 1
            parts.append(_call(k, step, "write_file", {"path": path, "content": f"v{c}"},
                               f"wrote {path}"))
        step += 1
        k += 1
        parts.append(_call(k, step, "render_page", {"port": 8093},
                           f"rendered http://localhost:8093/ at 1280x720 just now (14:{c:02d})\n"
                           f"no console errors", [{"id": f"img{c}", "mime": "image/png"}]))
    parts.append({"type": "text", "text": "Done.", "step": step + 1})
    return {"settings": {"effort": "auto", "thinking": True, "permission_mode": "yolo",
                         "tools": ["bash", "read_file", "write_file", "edit_file",
                                   "render_page"], "max_steps": 100},
            "messages": [{"role": "user", "parts": [{"type": "text", "text": "Make it brighter"}]},
                         {"role": "assistant", "parts": parts}]}


def test_through_chatd_the_note_is_stored_and_the_stop_is_resumable():
    import chat_trace as T
    res = T.replay(_stalled_trace())
    notes = T.monitor_notes(res["snapshot"])
    assert [n[2] for n in notes] == ["steer", "stop"], notes
    (_, s1, _), (_, s2, _) = notes
    assert s1 >= stall.MIN_STEPS and s2 - s1 >= stall.MIN_STEPS, notes
    # the replay pressed Continue after the stop (the record goes on), the streak carried on
    assert res["continued"] == [(1, "stall")], res["continued"]
    # every request starts with the one before it: the note is in the stored result
    assert not T.check(res), T.check(res)
    steer = next(m for b in res["requests"] for m in b["messages"]
                 if m["role"] == "tool" and "[run monitor:" in json.dumps(m))
    assert "step back" in json.dumps(steer)
    assert res["snapshot"]["messages"][-1]["status"] == "done"


def test_the_stop_reads_idle_with_its_notice():
    """Without the record's Continue: the run ends at the stop, interrupted, the chat idle."""
    import chat_trace as T
    trace = _stalled_trace()
    # the record ends right after the stop's step: nothing to continue with
    res = T.replay(trace)
    stop_step = T.monitor_notes(res["snapshot"])[-1][1]
    trace["messages"][1]["parts"] = [p for p in trace["messages"][1]["parts"]
                                     if p["step"] <= stop_step]
    res = T.replay(trace)
    last = res["snapshot"]["messages"][-1]
    assert last["status"] == "interrupted" and last["stats"]["interrupted"] == "stall", last["stats"]
    assert res["snapshot"]["status"] == "idle"
    notice = [p for p in last["parts"] if p.get("type") in ("notice", "error")]
    assert notice and "the run monitor" in json.dumps(notice), last["parts"][-2:]


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
