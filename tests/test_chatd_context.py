"""chatd's context on a long tool run (rc-0.8.0, the prompt "Create a clock in a single html").

THE CASE (live, 2026-10-13): a 21-step run on Flash-Next failed with "This conversation is about
319,136 tokens and the model's window is 262,144" while the engine had 16,980 tokens. Re-rendered
from the store: 2,122,473 request characters, 2,077,842 of them (97.9 %) the base64 of five 300 KB
render_page screenshots, counted as text at the learned 0.15 tokens a character. Each screenshot
also came as its own user message, six user turns in all, and the model wrote "The user is
repeating the same message for the 4th time". No compaction: the one user turn left nothing before
the last two turns to summarize.

  1  a tool's image rides in its tool message (one user turn), and only the newest stays; an older
     one is a line saying a newer screenshot follows. Other backends get the images in a user
     message after the results, the newest only
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

def test_tool_screenshots_ride_in_their_tool_message_and_only_the_newest_stays():
    branch, blob = run_with_screenshots()
    body = render.body("m", {}, branch, system="s", vision=True, blob=blob)
    msgs = body["messages"]
    users = [m for m in msgs if m["role"] == "user"]
    assert len(users) == 1 and users[0]["content"] == "Create a clock in a single html"
    shots = [m for m in msgs if m["role"] == "tool" and m["tool_call_id"].startswith("call_")]
    images = [p for m in shots if isinstance(m["content"], list) for p in m["content"]
              if p["type"] == "image_url"]
    assert len(images) == 1 and isinstance(shots[-1]["content"], list), "the newest only"
    assert all(render.LEFT_OUT in m["content"] for m in shots[:-1])
    assert shots[-1]["content"][0]["text"].endswith("no console errors")
    # the same branch renders to the same bytes (the engine's cache finds its prefix)
    assert render.encode(body) == render.encode(render.body("m", {}, branch, system="s",
                                                            vision=True, blob=blob))
    # another backend: the newest image in a user message after the results
    other = render.body("m", {}, branch, system="s", vision=True, blob=blob, inline_images=False)
    after = [m for m in other["messages"] if m["role"] == "user" and isinstance(m["content"], list)]
    assert len(after) == 1 and len(after[0]["content"]) == 2
    # a model without vision: no image at all
    text = render.body("m", {}, branch, system="s", vision=False, blob=blob)
    assert not any(isinstance(m["content"], list) for m in text["messages"])


# ---------------------------------------------------------------------- 2. the estimate

def test_the_estimate_counts_an_image_by_its_pixels_not_its_base64():
    branch, blob = run_with_screenshots()
    url = "data:image/png;base64," + base64.b64encode(blob({"id": "img0"})).decode()
    assert len(url) > 400_000
    assert comp.image_tokens(url) == 25 * 29 + 2                     # 800 x 900 at 32 pixels
    body = render.body("m", {}, branch, system="s", vision=True, blob=blob)
    chars, img = comp.measure(body)
    assert img == 25 * 29 + 2 and chars < 10_000, (chars, img)
    est = comp.estimate(body, 0.15)
    assert est < 3_000, est
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
