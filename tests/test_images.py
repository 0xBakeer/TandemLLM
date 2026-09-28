"""A chat request's images at the server's door -- found, fetched, decoded,
or refused with a 400 that names the field. No model, no GPU: server/images.py and the live label.

Run: python tests/test_images.py
"""

from __future__ import annotations

import base64
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from server import images as I  # noqa: E402


def _png(w=8, h=6, color=(200, 10, 10), fmt="PNG") -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format=fmt)
    return buf.getvalue()


def _data(raw: bytes, mime="image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def _msg(*parts, role="user"):
    return {"role": role, "content": list(parts)}


def _img(url):
    return {"type": "image_url", "image_url": {"url": url}}


class FakePre:
    """Stands in for the checkpoint's processor: a grid from the image size, patches of zeros."""

    def __init__(self):
        self.calls = 0

    def __call__(self, pil, source=""):
        import torch
        from engine.vision import Image, digest_of
        self.calls += 1
        w, h = pil.size
        grid = (1, 2 * max(1, h // 8), 2 * max(1, w // 8))
        pv = torch.tensor(list(pil.tobytes()), dtype=torch.float32)[:1].repeat(
            grid[0] * grid[1] * grid[2], 1536)
        return Image(pv, grid, digest_of(pv, grid), source)


def _refused(fn, param_part: str, words: str):
    try:
        fn()
    except I.ImageRefusal as exc:
        body = exc.body()["error"]
        assert param_part in body["param"], (body["param"], param_part)
        assert words in body["message"], (body["message"], words)
        assert body["type"] == "invalid_request_error"
        return body
    raise AssertionError(f"accepted; expected a 400 naming {param_part}")


LIM = I.Limits(max_bytes=4096, max_decode_pixels=10_000, max_images=2)


def test_text_requests_cost_nothing():
    body = {"messages": [{"role": "user", "content": "hi"},
                         _msg({"type": "text", "text": "hello"})]}
    assert I.collect(body, LIM, None) == []            # no processor needed, nothing fetched
    return "no image parts: [] without a processor"


def test_data_url_is_decoded_and_preprocessed():
    pre = FakePre()
    body = {"messages": [_msg({"type": "text", "text": "what is this"}, _img(_data(_png())))]}
    ims = I.collect(body, LIM, pre)
    assert len(ims) == 1 and pre.calls == 1 and ims[0].source == "data"
    # the plain-string form some clients send
    body2 = {"messages": [_msg({"type": "image_url", "image_url": _data(_png())})]}
    assert len(I.collect(body2, LIM, pre)) == 1
    return "data: URL, object and string forms"


def test_the_order_is_the_template_order():
    pre = FakePre()
    a, b = _data(_png(8, 8, (1, 2, 3))), _data(_png(16, 8, (4, 5, 6)))
    body = {"messages": [_msg(_img(a)), {"role": "assistant", "content": "ok"}, _msg(_img(b))]}
    ims = I.collect(body, LIM, pre)
    assert [im.grid for im in ims] == [(1, 2, 2), (1, 2, 4)], [im.grid for im in ims]
    return "message order kept"


def test_refusals_name_the_field():
    pre = FakePre()
    m = lambda *p, role="user": {"messages": [_msg({"type": "text", "text": "x"}, *p, role=role)]}  # noqa: E731
    _refused(lambda: I.collect(m(_img("http://example.com/a.png")), LIM, pre),
             "messages[0].content[1].image_url", "https:// or a data: URL")
    _refused(lambda: I.collect(m(_img("ftp://x/y")), LIM, pre), "content[1]", "https://")
    _refused(lambda: I.collect(m(_img("data:image/png;base64,@@@")), LIM, pre), "content[1]",
             "not valid base64")
    _refused(lambda: I.collect(m(_img("data:text/plain;base64,aGk=")), LIM, pre), "content[1]",
             "not an image")
    _refused(lambda: I.collect(m(_img("data:image/png,rawbytes")), LIM, pre), "content[1]",
             "base64")
    _refused(lambda: I.collect(m(_img(_data(b"GIF89a-not-really"))), LIM, pre), "content[1]",
             "could not decode")
    _refused(lambda: I.collect(m(_img(_data(_png(fmt="TIFF"), "image/tiff"))), LIM, pre),
             "content[1]", "unsupported image format")
    _refused(lambda: I.collect(m(_img(_data(_png(200, 100)))), LIM, pre), "content[1]",
             "the limit is 10000 pixels")
    big = I.Limits(max_bytes=64)
    _refused(lambda: I.collect(m(_img(_data(_png(64, 64, (9, 99, 199))))), big, pre),
             "content[1]", "larger than 64 bytes")
    _refused(lambda: I.collect(m(_img(_data(_png())), role="system"), LIM, pre), "content[1]",
             "system message")
    _refused(lambda: I.collect(m({"type": "video_url", "video_url": {"url": "x"}}), LIM, pre),
             "content[1]", "video")
    _refused(lambda: I.collect(m({"type": "image_url", "image_url": {}}), LIM, pre),
             "content[1].image_url", "needs a url")
    three = m(_img(_data(_png())), _img(_data(_png())), _img(_data(_png())))
    _refused(lambda: I.collect(three, LIM, pre), "content[3]", "the limit is 2")
    _refused(lambda: I.collect(m(_img(_data(_png()))), LIM, None), "content[1]",
             "without the vision tower")
    off = I.Limits(https=False)
    _refused(lambda: I.collect(m(_img("https://example.com/a.png")), off, pre), "content[1]",
             "disabled")
    return "15 refusals, each a 400 naming its part"


def test_admission_refuses_an_image_too_big_to_encode():
    pre = FakePre()
    body = {"messages": [_msg(_img(_data(_png(32, 32))))]}
    _refused(lambda: I.collect(body, LIM, pre, act_budget=1 << 20,
                               act_bytes=lambda n: n * (1 << 20)),
             "content[0]", "GiB to encode")
    assert len(I.collect(body, LIM, pre, act_budget=1 << 30, act_bytes=lambda n: n)) == 1
    return "the tower's byte math decides before the queue"


def test_https_fetch_failure_is_a_400():
    pre = FakePre()
    lim = I.Limits(timeout_s=2.0)
    # a port nothing listens on: a refused connection, not a hang and not a 500
    body = {"messages": [_msg(_img("https://127.0.0.1:9/a.png"))]}
    _refused(lambda: I.collect(body, lim, pre), "content[0]", "could not fetch")
    return "connection refused -> 400"


def test_live_label_while_encoding():
    from server import activity

    class Rec:
        encoding = (1, 2)
    assert activity.encoding_label(Rec()) == "Encoding image 1 of 2"
    Rec.encoding = (1, 1)
    assert activity.encoding_label(Rec()) == "Encoding image"
    Rec.encoding = None
    assert activity.encoding_label(Rec()) is None
    return "Encoding image 1 of 2"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<52} ok   {fn() or ''}", flush=True)
        except AssertionError as e:
            fails += 1
            print(f"  {name:<52} FAIL {e}", flush=True)
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
