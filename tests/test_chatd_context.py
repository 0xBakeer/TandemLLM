"""chatd's context on a long tool run (rc-0.8.0, the prompt "Create a clock in a single html").

THE CASE (live, 2026-10-13): a 21-step run on Flash-Next failed with "This conversation is about
319,136 tokens and the model's window is 262,144" while the engine had 16,980 tokens. Re-rendered
from the store: 2,122,473 request characters, 2,077,842 of them (97.9 %) the base64 of five 300 KB
render_page screenshots, counted as text at the learned 0.15 tokens a character. Each screenshot
also came as its own user message, six user turns in all, and the model wrote "The user is
repeating the same message for the 4th time". No compaction: the one user turn left nothing before
the last two turns to summarize.

  1  a tool's image rides in its tool message (one user turn). Images are append-only (SRV-546):
     all of them up to TOOL_IMAGES_KEPT, then the first ones and the newest, so a new screenshot
     never changes the prompt before the newest-but-one; that one becomes a line saying a newer
     screenshot follows. Other backends get the images in a user message after the results
  2  the estimate counts an image by its pixels (32 x 32 a token), never by its base64, and the
     tokens-per-character ratio is learned on the text alone
  3  a long run compacts its earlier steps between two steps, keeps its user message and goes on

CPU only. Run: python -m pytest -q tests/test_chatd_context.py
"""

from __future__ import annotations

import base64
import json
import os
import struct
import zlib

from chatd_util import Chat
from test_chatd_agent import calls, requests, small_window

from chatd import compaction as comp, render


def png(w: int, h: int, pad: int = 0) -> bytes:
    """A real PNG of w x h (one colour), with `pad` bytes of noise in an ancillary chunk: a 300 KB
    screenshot's bytes without drawing one."""
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    raw = b"".join(b"\x00" + b"\x20" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"tEXt", b"x\x00" + os.urandom(pad))
            + chunk(b"IEND", b""))


def run_with_screenshots(n: int = 5, pad: int = 300_000):
    """The live case's shape: one user message, then n steps that each rendered the page."""
    shots = {f"img{k}": png(800, 900, pad) for k in range(n)}
    parts = []
    for k in range(n):
        parts += [{"type": "thinking", "text": f"check {k}", "step": 2 * k + 1},
                  {"type": "tool_call", "id": f"call_{k}", "name": "render_page", "step": 2 * k + 1,
                   "input": json.dumps({"path": "/workspace/clock.html"}), "status": "done",
                   "output": {"content": f"rendered file:///workspace/clock.html; screenshot "
                                         f"/workspace/.shots/call_{k}.png\nno console errors",
                              "images": [{"id": f"img{k}", "mime": "image/png"}]}},
                  {"type": "tool_call", "id": f"edit_{k}", "name": "edit_file", "step": 2 * k + 2,
                   "input": "{}", "status": "done", "output": {"content": "edited clock.html"}}]
    branch = [{"id": "u", "role": "user", "parts": [{"type": "text",
                                                      "text": "Create a clock in a single html"}]},
              {"id": "a", "role": "assistant", "parts": parts}]
    return branch, (lambda a: shots.get(a["id"]))


# ---------------------------------------------------------------------- 1. the images in a request

def _shots(body):
    msgs = body["messages"]
    return [m for m in msgs if m["role"] == "tool" and m["tool_call_id"].startswith("call_")]


def _images(shots):
    return [p for m in shots if isinstance(m["content"], list) for p in m["content"]
            if p["type"] == "image_url"]


def test_tool_screenshots_ride_in_their_tool_message_and_stay_where_they_are():
    branch, blob = run_with_screenshots()
    body = render.body("m", {}, branch, system="s", vision=True, blob=blob)
    msgs = body["messages"]
    users = [m for m in msgs if m["role"] == "user"]
    assert len(users) == 1 and users[0]["content"] == "Create a clock in a single html"
    shots = _shots(body)
    assert len(_images(shots)) == 5 and all(isinstance(m["content"], list) for m in shots)
    assert shots[-1]["content"][0]["text"].endswith("no console errors")
    # the same branch renders to the same bytes (the engine's cache finds its prefix)
    assert render.encode(body) == render.encode(render.body("m", {}, branch, system="s",
                                                            vision=True, blob=blob))
    # another backend: the images in a user message after the results
    other = render.body("m", {}, branch, system="s", vision=True, blob=blob, inline_images=False)
    after = [m for m in other["messages"] if m["role"] == "user" and isinstance(m["content"], list)]
    assert len(after) == 5 and all(len(m["content"]) == 2 for m in after)
    # a model without vision: no image at all
    text = render.body("m", {}, branch, system="s", vision=False, blob=blob)
    assert not any(isinstance(m["content"], list) for m in text["messages"])


def test_srv546_a_new_screenshot_only_appends_to_the_prompt():
    """SRV-546 (live 2026-10-17 10:18:36): with only the newest screenshot kept, each new one turned
    the previous into a line and the engine read 90,186 tokens again. Every step's request now
    starts with the previous step's request, byte for byte, up to the cap; past it only the
    newest-but-one changes, and everything before it stays."""
    # the pagoda run's seven screenshots: every request starts with the one before it
    branch7, blob7 = run_with_screenshots(7, pad=10)
    msgs = [render.body("m", {}, [branch7[0], {"id": "a", "role": "assistant", "parts": [
        p for p in branch7[1]["parts"] if p["step"] <= step]}], system="s", vision=True,
        blob=blob7)["messages"] for step in range(1, 15)]
    for step in range(1, 14):
        assert msgs[step][:len(msgs[step - 1])] == msgs[step - 1], f"step {step + 1} rewrote it"
    k = render.TOOL_IMAGES_KEPT
    n = k + 3
    branch, blob = run_with_screenshots(n, pad=10)
    full = branch[1]["parts"]

    def at(step):                       # the request after `step` model calls
        b = [branch[0], {"id": "a", "role": "assistant",
                         "parts": [p for p in full if p["step"] <= step]}]
        return render.body("m", {}, b, system="s", vision=True, blob=blob)["messages"]

    prev = None
    for step in range(1, 2 * n + 1):
        cur = at(step)
        if prev is not None:
            shots_now = (step + 1) // 2         # odd steps add a screenshot
            if shots_now <= k or step % 2 == 0:
                assert cur[:len(prev)] == prev, f"step {step} rewrote the prompt"
            else:
                # past the cap: the messages before the newest-but-one's result stay as they were
                ids = [m.get("tool_call_id") for m in prev]
                cut = ids.index(f"call_{shots_now - 2}")
                assert cur[:cut] == prev[:cut], f"step {step} changed an earlier image"
                assert cur[cut] != prev[cut] and render.LEFT_OUT in cur[cut]["content"]
        prev = cur
    shots = _shots({"messages": at(2 * n)})
    kept = [m["tool_call_id"] for m in shots if isinstance(m["content"], list)]
    assert kept == [f"call_{i}" for i in range(k - 1)] + [f"call_{n - 1}"]
    assert all(render.LEFT_OUT in m["content"] for m in shots if not isinstance(m["content"], list))


def test_srv546_the_now_line_names_a_re_read_and_its_reason():
    """SRV-546: the card said "Prefilling 114,336 tokens" beside the run's 81,843 written tokens and
    its 38 min, which read as 36 tok/s; the engine's live row says what the cache held and why
    the rest is read again."""
    from chatd.service import prefill_words
    why = "the chat's text differs from what the engine read, at token 88,640 of 112,642"
    assert prefill_words({"total": 114_336, "cached": 29_696, "why": why}) == \
        f"Re-reading 84,640 of 114,336 tokens: {why}"
    assert prefill_words({"total": 114_336, "cached": 112_642, "why": None}) == \
        "Reading 1,694 new tokens (112,642 cached)"
    assert prefill_words({"total": 30_443, "cached": 0}) == "Prefilling 30,443 tokens"


# ---------------------------------------------------------------------- 2. the estimate

def test_the_estimate_counts_an_image_by_its_pixels_not_its_base64():
    branch, blob = run_with_screenshots()
    url = "data:image/png;base64," + base64.b64encode(blob({"id": "img0"})).decode()
    assert len(url) > 400_000
    assert comp.image_tokens(url) == 25 * 29 + 2                     # 800 x 900 at 32 pixels
    body = render.body("m", {}, branch, system="s", vision=True, blob=blob)
    chars, img = comp.measure(body)
    assert img == 5 * (25 * 29 + 2) and chars < 10_000, (chars, img)
    est = comp.estimate(body, 0.15)
    assert est < 6_000, est
    # the old count: every image's base64 as text, all five kept
    real = render.TOOL_IMAGES_KEPT
    render.TOOL_IMAGES_KEPT = 99
    try:
        old = render.body("m", {}, branch, system="s", vision=True, blob=blob, inline_images=False)
    finally:
        render.TOOL_IMAGES_KEPT = real
    old_chars = len(json.dumps(old["messages"], ensure_ascii=False))
    assert old_chars * 0.15 > 50 * est
    # the ratio is learned on text: the engine's count less the images' tokens
    assert comp.learn(1000 + img, chars, img) == max(0.15, min(0.8, 1000 / chars))
    assert comp.learn(img, chars, img) is None
    # JPEG and GIF sizes; an unknown format counts as IMAGE_UNKNOWN
    jpg = b"\xff\xd8\xff\xe0\x00\x10" + b"JFIF\x00" + b"\x00" * 9 + \
        b"\xff\xc0\x00\x11\x08" + struct.pack(">HH", 640, 480) + b"\x03" + b"\x00" * 9
    assert comp.image_tokens("data:image/jpeg;base64," + base64.b64encode(jpg).decode()) == \
        20 * 15 + 2
    gif = b"GIF89a" + struct.pack("<HH", 64, 64) + b"\x00" * 8
    assert comp.image_tokens("data:image/gif;base64," + base64.b64encode(gif).decode()) == 4 + 2
    assert comp.image_tokens("data:image/webp;base64,AAAA") == comp.IMAGE_UNKNOWN


# ---------------------------------------------------------------------- 3. a long run compacts

def test_step_boundary():
    def run(steps):
        return [{"role": "user", "parts": []},
                {"role": "assistant", "parts": [{"type": "text", "step": s} for s in steps]}]
    assert comp.step_boundary(run([1, 2, 3])) is None
    assert comp.step_boundary(run([1, 2, 3, 4, 5])) == (1, 3)
    assert comp.step_boundary(run([1, 2, 3, 4, 5]), prior_step=3) is None
    assert comp.step_boundary(run([1, 2, 3, 4, 5, 6, 7]), prior_step=3) == (1, 5)
    assert comp.step_boundary([{"role": "user", "parts": []}]) is None


def test_a_long_run_compacts_between_steps_and_keeps_its_user_message():
    c = Chat()
    try:
        small_window(c, window=6000)
        c.svc.learn_ratio = lambda *a: None                 # the fake counts words, not tokens
        cid = c.new()
        c.svc.conv(cid).tok_ratio = 0.5
        ws = c.svc.workspace(cid)
        ws.write("a.txt", ("alpha line\n" * 20).encode())
        ws.write("b.txt", ("beta line\n" * 20).encode())
        c.ok("PATCH", f"/conversations/{cid}/settings", {"max_steps": 16})
        c.send(cid, calls(("read_file", {"path": "a.txt"}), ("read_file", {"path": "b.txt"}))
               + " FAKE_CALLS_ALWAYS")
        s = c.wait_idle(cid, 60)
        assert s["compactions"], "compacted inside the run"
        cp = s["compactions"][0]
        assert "@" in cp["boundary_id"] and cp["boundary_id"].split("@")[0] == s["messages"][1]["id"]
        reqs = requests(c)
        after = [b for b in reqs if "Summary of the earlier part" in
                 (b["messages"][0].get("content") or "")]
        assert after, "the next steps carry the summary"
        for b in after:
            users = [m for m in b["messages"] if m["role"] == "user"]
            assert len(users) == 1 and "FAKE_CALLS" in users[0]["content"]
            assert comp.estimate(b, 0.5) + comp.MIN_ANSWER <= 6000
        assert len(reqs) > 8, "the run went on past the window"
        assert "no room left" not in json.dumps(s["messages"][-1])
    finally:
        c.close()
