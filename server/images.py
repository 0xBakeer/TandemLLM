"""The images of a chat request -- found, fetched, decoded and preprocessed -- or a 400.

Everything here runs on the handler's thread BEFORE the request queues for the engine: a download,
a decode and the checkpoint's own image processor are host work, and a bad image is the client's
400, not a queue slot. What comes out is a list of `engine.vision.Image` (the processor's patches,
the grid, the content digest the caches key on) in the order the chat template will write their
placeholders.

Accepted: `{"type": "image_url", "image_url": {"url": ...}}` (and `"image_url": "<url>"`) in a
user, assistant or tool message, where the URL is `https://...` or `data:image/...;base64,...`.
Refused with a 400 that names the field: any other URL scheme, a download that fails or is larger
than the byte cap, a body that is not an image in one of the formats below, an image with more
pixels than the decode cap, more images than the count cap, an image in a system message (the chat
template forbids it), video parts, and any image when the server runs without the vision tower.

An https fetch is the server making a request on a client's behalf, so it is fenced: the host must
resolve to public addresses only (no loopback, private, link-local, multicast or reserved ranges,
unless the operator allows them), the connection goes to the address that was checked (a second
DNS answer cannot move it), a redirect is followed only to another https URL that passes the same
check and at most `max_redirects` times, a URL with credentials in it is refused, and the whole
download -- connect, redirects and body -- has one deadline.

Privacy: a URL or an image's bytes never reach a log line; the log says "data" or "https" and the
grid.
"""

from __future__ import annotations

import base64
import binascii
import http.client
import io
import ipaddress
import socket
import ssl
import time
import urllib.parse

FORMATS = ("PNG", "JPEG", "WEBP", "GIF", "BMP")


class ImageRefusal(ValueError):
    """A 400 naming the field (same shape as server/compat.py `Refusal`)."""

    def __init__(self, param: str, message: str):
        super().__init__(message)
        self.param = param

    def body(self) -> dict:
        return {"error": {"message": str(self), "type": "invalid_request_error",
                          "param": self.param}}


class Limits:
    def __init__(self, max_bytes: int = 20 << 20, max_decode_pixels: int = 64_000_000,
                 max_images: int = 16, timeout_s: float = 15.0, https: bool = True,
                 allow_private: bool = False, max_redirects: int = 3,
                 max_image_rows: int = 65536):
        self.max_bytes = int(max_bytes)
        self.max_decode_pixels = int(max_decode_pixels)
        self.max_images = int(max_images)
        self.timeout_s = float(timeout_s)       # the whole https download, redirects included
        self.https = bool(https)
        self.allow_private = bool(allow_private)
        self.max_redirects = int(max_redirects)
        # prompt rows of all a request's images together. Also what bounds the host memory a
        # queued request holds: a row is four patches of 1536 values, 12 KiB in bf16
        self.max_image_rows = int(max_image_rows)


def _url_of(part: dict) -> str | None:
    v = part.get("image_url")
    if isinstance(v, dict):
        v = v.get("url")
    if v is None and isinstance(part.get("image"), str):
        v = part["image"]
    return v if isinstance(v, str) else None


def image_parts(messages) -> list[tuple[str, str]]:
    """`[(param, url)]` for every image part, in message order -- the order the template writes
    the placeholders in. Raises `ImageRefusal` for parts this server does not take."""
    out: list[tuple[str, str]] = []
    if not isinstance(messages, list):
        return out
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for j, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            param = f"messages[{i}].content[{j}]"
            if kind in ("video", "video_url", "input_video") or "video" in part:
                raise ImageRefusal(param, "video input is not supported by this engine")
            if kind in ("image_url", "image", "input_image") or "image_url" in part \
                    or "image" in part:
                if m.get("role") == "system":
                    raise ImageRefusal(param, "a system message cannot contain images")
                url = _url_of(part)
                if not url:
                    raise ImageRefusal(param + ".image_url", "image_url needs a url")
                out.append((param + ".image_url", url))
    return out


def fetch(url: str, param: str, lim: Limits) -> tuple[bytes, str]:
    """The image's bytes and where they came from ("data" or "https")."""
    if url.startswith("data:"):
        head, sep, payload = url.partition(",")
        if not sep:
            raise ImageRefusal(param, "malformed data: URL")
        meta = head[5:].split(";")
        mime = meta[0].lower() if meta and meta[0] else ""
        if mime and not mime.startswith("image/"):
            raise ImageRefusal(param, f"data: URL of type {mime!r} is not an image")
        if "base64" not in [x.lower() for x in meta[1:]]:
            raise ImageRefusal(param, "data: URL images must be base64-encoded")
        if len(payload) > (lim.max_bytes * 4) // 3 + 8:
            raise ImageRefusal(param, f"image is larger than {lim.max_bytes} bytes")
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            raise ImageRefusal(param, "data: URL is not valid base64") from None
        if len(raw) > lim.max_bytes:
            raise ImageRefusal(param, f"image is larger than {lim.max_bytes} bytes")
        return raw, "data"
    if url.startswith("https://"):
        if not lim.https:
            raise ImageRefusal(param, "https image URLs are disabled on this server; "
                                      "send the image as a data: URL")
        return fetch_https(url, param, lim), "https"
    raise ImageRefusal(param, "image_url must be an https:// or a data: URL")


# ------------------------------------------------------------------------------------ https
REDIRECTS = (301, 302, 303, 307, 308)


def public_address(ip: str) -> bool:
    """Whether an address is on the public internet: not loopback, private, link-local, shared
    (100.64/10), multicast, reserved or unspecified. An IPv4 address mapped or embedded in IPv6 is
    judged as the IPv4 address it carries."""
    a = ipaddress.ip_address(ip.split("%", 1)[0])
    if isinstance(a, ipaddress.IPv6Address):
        inner = a.ipv4_mapped or a.sixtofour or (a.teredo[1] if a.teredo else None)
        if inner is None and a in ipaddress.ip_network("64:ff9b::/96"):
            inner = ipaddress.IPv4Address(int(a) & 0xFFFFFFFF)          # NAT64
        if inner is not None:
            a = inner
    return a.is_global and not a.is_multicast


def _target(url: str, param: str, lim: Limits, resolve) -> tuple[str, int, str, str]:
    """`(host, port, path, ip)` for an https URL that passes the fence, or a 400."""
    try:
        u = urllib.parse.urlsplit(url)
        port = u.port or 443
    except ValueError:
        raise ImageRefusal(param, "image_url is not a valid URL") from None
    if u.scheme.lower() != "https":
        raise ImageRefusal(param, "image_url must be an https:// or a data: URL")
    if u.username is not None or u.password is not None:
        raise ImageRefusal(param, "image_url must not carry credentials")
    host = u.hostname
    if not host:
        raise ImageRefusal(param, "image_url has no host")
    try:
        infos = resolve(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        raise ImageRefusal(param, "could not fetch the image (the host does not resolve)") from None
    ips = [info[4][0] for info in infos]
    if not ips:
        raise ImageRefusal(param, "could not fetch the image (the host does not resolve)")
    if not lim.allow_private and not all(public_address(ip) for ip in ips):
        # every answer, not just the first: a name with one public and one private address
        # would otherwise be a coin toss away from the operator's own network
        raise ImageRefusal(param, "image_url resolves to a non-public address; this server only "
                                  "fetches images from public hosts")
    path = u.path or "/"
    if u.query:
        path += "?" + u.query
    return host, port, path, ips[0]


class _PinnedHTTPS(http.client.HTTPSConnection):
    """An https connection to an address checked beforehand: the socket goes to `ip`, the TLS
    handshake and the certificate check use the host name."""

    def __init__(self, host: str, port: int, ip: str, timeout: float):
        super().__init__(host, port, timeout=timeout, context=ssl.create_default_context())
        self._ip = ip

    def connect(self):
        sock = socket.create_connection((self._ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def fetch_https(url: str, param: str, lim: Limits, *, resolve=None, connect=None) -> bytes:
    """The body of an https image URL, inside the fence the module docstring describes.
    `resolve` and `connect(host, port, ip, timeout)` are for the tests."""
    resolve = resolve or socket.getaddrinfo
    connect = connect or _PinnedHTTPS
    deadline = time.monotonic() + lim.timeout_s

    def left() -> float:
        t = deadline - time.monotonic()
        if t <= 0:
            raise ImageRefusal(param, f"could not fetch the image (took longer than "
                                      f"{lim.timeout_s:g} s)")
        return t

    for _hop in range(lim.max_redirects + 1):
        host, port, path, ip = _target(url, param, lim, resolve)
        conn = connect(host, port, ip, left())
        try:
            try:
                conn.request("GET", path, headers={"User-Agent": "tandem-engine/vision",
                                                   "Accept": "image/*"})
                r = conn.getresponse()
            except ImageRefusal:
                raise
            except (OSError, http.client.HTTPException, ValueError) as exc:
                raise ImageRefusal(param, f"could not fetch the image "
                                          f"({type(exc).__name__})") from None
            if r.status in REDIRECTS:
                loc = r.getheader("Location")
                if not loc:
                    raise ImageRefusal(param, f"could not fetch the image (HTTP {r.status} "
                                              f"without a Location)")
                url = urllib.parse.urljoin(url, loc)
                if not url.lower().startswith("https://"):
                    raise ImageRefusal(param, "image_url redirects to a URL that is not https")
                continue
            if r.status != 200:
                raise ImageRefusal(param, f"could not fetch the image (HTTP {r.status})")
            size = r.getheader("Content-Length")
            if size and size.strip().isdigit() and int(size) > lim.max_bytes:
                raise ImageRefusal(param, f"image is larger than {lim.max_bytes} bytes")
            chunks, got = [], 0
            while True:
                # a server that sends one byte a second must not hold the handler past the
                # deadline: the socket timeout is re-armed with what is left before every read
                t = left()
                if conn.sock is not None:
                    conn.sock.settimeout(t)
                try:
                    b = r.read(min(65536, lim.max_bytes + 1 - got))
                except (OSError, http.client.HTTPException) as exc:
                    raise ImageRefusal(param, f"could not fetch the image "
                                              f"({type(exc).__name__})") from None
                if not b:
                    break
                chunks.append(b)
                got += len(b)
                if got > lim.max_bytes:
                    raise ImageRefusal(param, f"image is larger than {lim.max_bytes} bytes")
            return b"".join(chunks)
        finally:
            conn.close()
    raise ImageRefusal(param, f"image_url redirects more than {lim.max_redirects} times")


def decode(raw: bytes, param: str, lim: Limits):
    """A PIL RGB image, or a 400. The pixel count is checked from the header, before the pixels
    are decoded (a small file can claim a huge canvas)."""
    try:
        from PIL import Image as PILImage
    except ImportError:            # pragma: no cover - the server's own environment has PIL
        raise ImageRefusal(param, "this server cannot decode images (no PIL)") from None
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PILImage.DecompressionBombWarning)
            img = PILImage.open(io.BytesIO(raw))
            fmt = (img.format or "").upper()
            if fmt not in FORMATS:
                raise ImageRefusal(param, f"unsupported image format {fmt or 'unknown'!r} "
                                          f"(accepted: {', '.join(FORMATS)})")
            w, h = img.size
            if w <= 0 or h <= 0:
                raise ImageRefusal(param, "image has no pixels")
            if w * h > lim.max_decode_pixels:
                raise ImageRefusal(param, f"image is {w}x{h} pixels; the limit is "
                                          f"{lim.max_decode_pixels} pixels")
            if getattr(img, "is_animated", False):
                img.seek(0)
            img.load()
            return img.convert("RGB")
    except ImageRefusal:
        raise
    except Exception as exc:
        raise ImageRefusal(param, f"could not decode the image ({type(exc).__name__})") from None


class Preprocessor:
    """The checkpoint's own image processor (preprocessor_config.json), and the digest.

    `dtype` is what the patches are kept in until the tower reads them. The tower's first op
    casts its input to the tower's dtype, so patches kept in that dtype give the tower the same
    numbers and take half the host memory of the processor's float32 (a 16-megapixel image is
    400 MB of float32 patches)."""

    def __init__(self, snapshot: str, merge: int, max_pixels: int = 0, dtype=None):
        from transformers import AutoImageProcessor
        self.proc = AutoImageProcessor.from_pretrained(snapshot)
        self.merge = int(merge)
        self.dtype = dtype
        self.kwargs = {}
        if max_pixels:
            size = dict(getattr(self.proc, "size", {}) or {})
            size["longest_edge"] = int(max_pixels)
            if size.get("shortest_edge", 0) > int(max_pixels):
                size["shortest_edge"] = int(max_pixels)
            self.kwargs["size"] = size

    def __call__(self, pil, source: str = ""):
        from engine.vision import Image, digest_of
        out = self.proc(images=[pil], return_tensors="pt", **self.kwargs)
        pv = out["pixel_values"]
        grid = tuple(int(x) for x in out["image_grid_thw"][0].tolist())
        pv = pv.float() if self.dtype is None else pv.to(self.dtype)
        pv = pv.contiguous()
        return Image(pixel_values=pv, grid=grid, digest=digest_of(pv, grid), source=source)


def collect(body: dict, lim: Limits, pre: "Preprocessor | None", *, act_budget: int = 0,
            act_bytes=None) -> list:
    """The request's images, preprocessed, or `ImageRefusal`. `[]` for a request without images
    (the common case, and the only one that costs nothing more than a walk over the messages).

    `act_bytes(n_patches)` is the tower's per-image peak (engine/vision.py), checked against
    `act_budget` here, before the request queues: the admission check for the image prefill."""
    parts = image_parts(body.get("messages"))
    if not parts:
        return []
    if pre is None:
        raise ImageRefusal(parts[0][0], "this server runs without the vision tower; image "
                                        "input is not available")
    if len(parts) > lim.max_images:
        raise ImageRefusal(parts[lim.max_images][0],
                           f"{len(parts)} images; the limit is {lim.max_images} a request")
    images = []
    rows = 0
    for param, url in parts:
        raw, source = fetch(url, param, lim)
        pil = decode(raw, param, lim)
        try:
            img = pre(pil, source)
        except Exception as exc:
            raise ImageRefusal(param, f"the image processor refused the image "
                                      f"({type(exc).__name__})") from None
        if act_budget and act_bytes is not None:
            need = act_bytes(img.pixel_values.shape[0])
            if need > act_budget:
                raise ImageRefusal(param, f"image needs ~{need / 2**30:.1f} GiB to encode; the "
                                          f"limit is {act_budget / 2**30:.1f} GiB "
                                          f"(send a smaller image)")
        rows += img.tokens(pre.merge)
        if rows > lim.max_image_rows:
            raise ImageRefusal(param, f"the request's images take {rows} prompt rows up to this "
                                      f"one; the limit is {lim.max_image_rows} (send smaller or "
                                      f"fewer images)")
        images.append(img)
    return images
