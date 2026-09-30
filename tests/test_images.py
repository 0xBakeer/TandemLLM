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

    merge = 2

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


def test_image_rows_are_capped_per_request():
    pre = FakePre()
    # 32x32 -> grid (1, 8, 8) -> 16 rows each
    body = {"messages": [_msg(_img(_data(_png(32, 32))), _img(_data(_png(32, 32, (1, 1, 1)))))]}
    assert len(I.collect(body, I.Limits(max_image_rows=32), pre)) == 2
    _refused(lambda: I.collect(body, I.Limits(max_image_rows=31), pre), "content[1]",
             "32 prompt rows")
    return "16 + 16 rows: 32 fits, 31 is a 400 naming the second image"


def test_https_fetch_failure_is_a_400():
    pre = FakePre()
    # a port nothing listens on: a refused connection, not a hang and not a 500 (loopback
    # needs the operator's switch, which is what the next test is about)
    lim = I.Limits(timeout_s=2.0, allow_private=True)
    body = {"messages": [_msg(_img("https://127.0.0.1:9/a.png"))]}
    _refused(lambda: I.collect(body, lim, pre), "content[0]", "could not fetch")
    return "connection refused -> 400"


def _resolver(table):
    """A getaddrinfo that answers from `table` (host -> [ip, ...])."""
    def resolve(host, port, type=None):
        if host not in table:
            raise OSError("no such host")
        return [(2, 1, 6, "", (ip, port)) for ip in table[host]]
    return resolve


class FakeResp:
    def __init__(self, status=200, headers=None, body=b"", drip=0.0):
        self.status, self.headers, self.body, self.drip = status, headers or {}, body, drip

    def getheader(self, k, default=None):
        return self.headers.get(k, default)

    def read(self, n):
        import time
        if self.drip:
            time.sleep(self.drip)
            n = min(n, 1)
        b, self.body = self.body[:n], self.body[n:]
        return b


class FakeConn:
    """Scripted https: `routes[(host, path)] -> FakeResp`; records every (host, ip, path)."""
    log: list = []
    routes: dict = {}

    def __init__(self, host, port, ip, timeout):
        self.host, self.ip, self.sock = host, ip, None

    def request(self, method, path, headers=None):
        self.path = path
        FakeConn.log.append((self.host, self.ip, path))

    def getresponse(self):
        return FakeConn.routes[(self.host, self.path)]

    def close(self):
        pass


def test_https_fence_refuses_non_public_hosts():
    pub = _resolver({"img.example": ["93.184.215.14"], "lan.example": ["192.168.1.20"],
                     "mixed.example": ["93.184.215.14", "10.0.0.5"],
                     "v6loop.example": ["::1"], "mapped.example": ["::ffff:127.0.0.1"],
                     "nat64.example": ["64:ff9b::a00:1"], "cgnat.example": ["100.64.0.1"],
                     "meta.example": ["169.254.169.254"]})
    lim = I.Limits()
    for host in ("lan.example", "mixed.example", "v6loop.example", "mapped.example",
                 "nat64.example", "cgnat.example", "meta.example"):
        _refused(lambda: I.fetch_https(f"https://{host}/a.png", "p", lim, resolve=pub,
                                       connect=FakeConn), "p", "non-public address")
    for url in ("https://127.0.0.1/a.png", "https://[::1]/a.png", "https://10.1.2.3/x"):
        _refused(lambda: I.fetch_https(url, "p", lim, resolve=_resolver(
            {u: [u] for u in ("127.0.0.1", "::1", "10.1.2.3")}), connect=FakeConn),
            "p", "non-public address")
    _refused(lambda: I.fetch_https("https://user:pw@img.example/a.png", "p", lim, resolve=pub,
                                   connect=FakeConn), "p", "credentials")
    _refused(lambda: I.fetch_https("https://nowhere.example/a.png", "p", lim, resolve=pub,
                                   connect=FakeConn), "p", "does not resolve")
    # the operator's switch lets the LAN through, and the connection goes to the checked address
    FakeConn.log, FakeConn.routes = [], {("lan.example", "/a.png?x=1"): FakeResp(body=b"img")}
    assert I.fetch_https("https://lan.example/a.png?x=1", "p", I.Limits(allow_private=True),
                         resolve=pub, connect=FakeConn) == b"img"
    assert FakeConn.log == [("lan.example", "192.168.1.20", "/a.png?x=1")], FakeConn.log
    return "LAN, loopback, mapped, NAT64, CGNAT, metadata, credentials: 400; switch lets LAN in"


def test_https_redirects_are_fenced_and_counted():
    pub = _resolver({"a.example": ["93.184.215.14"], "b.example": ["93.184.215.15"],
                     "lan.example": ["10.0.0.9"]})
    lim = I.Limits(max_redirects=2)
    R = FakeResp
    FakeConn.log = []
    FakeConn.routes = {("a.example", "/1"): R(302, {"Location": "https://b.example/2"}),
                       ("b.example", "/2"): R(301, {"Location": "/3"}),
                       ("b.example", "/3"): R(200, body=b"png")}
    assert I.fetch_https("https://a.example/1", "p", lim, resolve=pub, connect=FakeConn) == b"png"
    assert [x[2] for x in FakeConn.log] == ["/1", "/2", "/3"]
    FakeConn.routes = {("a.example", "/1"): R(302, {"Location": "http://a.example/2"})}
    _refused(lambda: I.fetch_https("https://a.example/1", "p", lim, resolve=pub,
                                   connect=FakeConn), "p", "not https")
    FakeConn.routes = {("a.example", "/1"): R(307, {"Location": "https://lan.example/x"})}
    _refused(lambda: I.fetch_https("https://a.example/1", "p", lim, resolve=pub,
                                   connect=FakeConn), "p", "non-public address")
    FakeConn.routes = {("a.example", "/1"): R(302, {"Location": "/1"})}
    _refused(lambda: I.fetch_https("https://a.example/1", "p", lim, resolve=pub,
                                   connect=FakeConn), "p", "more than 2 times")
    FakeConn.routes = {("a.example", "/1"): R(404)}
    _refused(lambda: I.fetch_https("https://a.example/1", "p", lim, resolve=pub,
                                   connect=FakeConn), "p", "HTTP 404")
    return "redirect chain followed; to http, to LAN, a loop and a 404 refused"


def test_https_size_and_deadline():
    pub = _resolver({"a.example": ["93.184.215.14"]})
    R = FakeResp
    FakeConn.routes = {("a.example", "/big"): R(200, {"Content-Length": "5000"}, b"x" * 10),
                       ("a.example", "/liar"): R(200, {}, b"x" * 5000),
                       ("a.example", "/slow"): R(200, {}, b"x" * 100, drip=0.05)}
    lim = I.Limits(max_bytes=4096, timeout_s=0.5)
    _refused(lambda: I.fetch_https("https://a.example/big", "p", lim, resolve=pub,
                                   connect=FakeConn), "p", "larger than 4096")
    _refused(lambda: I.fetch_https("https://a.example/liar", "p", lim, resolve=pub,
                                   connect=FakeConn), "p", "larger than 4096")
    import time
    t0 = time.monotonic()
    _refused(lambda: I.fetch_https("https://a.example/slow", "p", lim, resolve=pub,
                                   connect=FakeConn), "p", "longer than 0.5 s")
    assert time.monotonic() - t0 < 1.5
    return "Content-Length, streamed size and a dripping server: 400 inside the deadline"


def test_public_address():
    ok = ["93.184.215.14", "2606:2800:21f:cb07:6820:80da:af6b:8b2c", "8.8.8.8"]
    bad = ["127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.0.1", "169.254.169.254", "0.0.0.0",
           "100.64.0.1", "224.0.0.1", "::1", "fe80::1", "fc00::1", "::ffff:10.0.0.1",
           "64:ff9b::7f00:1", "2002:7f00:1::1", "255.255.255.255"]
    assert all(I.public_address(a) for a in ok), [a for a in ok if not I.public_address(a)]
    assert not any(I.public_address(a) for a in bad), [a for a in bad if I.public_address(a)]
    return f"{len(ok)} public, {len(bad)} not"


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
