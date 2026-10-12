"""The sandbox's way out: the internet, never the LAN or this box (SRV-355, Chat 6).

A sandbox container runs with `--network none`: it has a loopback and nothing else, so no packet of
its own reaches the LAN, the router, the engine on :8000 or chatd. Its way out is an HTTP proxy:
inside the container `relay.py` listens on 127.0.0.1:3128 (HTTP_PROXY and HTTPS_PROXY point
there) and passes every connection to a unix socket that chatd serves for that chat only
(`<data>/egress/<conversation>/proxy.sock`, mounted read-only at /run/tandem). Here each request
is checked before anything is connected:

- the host is resolved here, and the connection goes to the address that was checked (a name that
  resolves to a public address once and a private one later cannot slip through);
- every address must be global: private (10/8, 172.16/12, 192.168/16, fc00::/7), loopback,
  link-local, CGNAT (100.64/10), multicast, reserved and the unspecified ranges are refused, IPv4
  mapped into IPv6 included; one refused address refuses the name;
- `CONNECT host:port` tunnels TLS (pip, npm, git over https, curl); an absolute `http://` request
  is passed on with `Connection: close`;
- a chat whose network is `off` gets 403 for everything.

This replaces the plan's nftables rule for the `tandem-sbx` user (that needs root, which this box's
chatd does not have): the container cannot open a socket to anything but its own loopback, which is
stricter than a rule that drops what it sends. `fetch()` is the same check for chatd's own
web_fetch tool. Logs carry counts, never a host or a path.
"""

from __future__ import annotations

import http.client
import ipaddress
import os
import socket
import ssl
import threading
import urllib.parse

CHUNK = 65536
CONNECT_TIMEOUT_S = 15.0
MAX_FETCH = 4 * 1024 * 1024


class Refused(Exception):
    pass


def allowed(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr.split("%")[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast and not ip.is_reserved


def resolve(host: str, port: int) -> list[tuple]:
    """The addresses to connect to, every one checked; Refused when any is not public."""
    if not host or port <= 0 or port > 65535:
        raise Refused("bad host or port")
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise Refused(f"cannot resolve {host}") from None
    if not infos:
        raise Refused(f"cannot resolve {host}")
    for fam, _, _, _, sa in infos:
        if not allowed(sa[0]):
            raise Refused(f"{host} is a local network address; only the public internet is "
                          f"reachable")
    return infos


def connect(host: str, port: int) -> socket.socket:
    last = None
    for fam, typ, proto, _, sa in resolve(host, port):
        s = socket.socket(fam, typ, proto)
        s.settimeout(CONNECT_TIMEOUT_S)
        try:
            s.connect(sa)
            s.settimeout(None)
            return s
        except OSError as exc:
            last = exc
            s.close()
    raise Refused(f"cannot connect to {host}:{port} ({type(last).__name__})")


def fetch(url: str, max_bytes: int = MAX_FETCH, redirects: int = 5) -> tuple[int, str, bytes]:
    """GET a public URL with the same check (web_fetch); redirects are checked again."""
    for _ in range(redirects + 1):
        u = urllib.parse.urlsplit(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise Refused("only http and https URLs")
        port = u.port or (443 if u.scheme == "https" else 80)
        sock = connect(u.hostname, port)
        if u.scheme == "https":
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=u.hostname)
        sock.settimeout(30)
        conn = http.client.HTTPConnection(u.hostname, port)
        conn.sock = sock
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        try:
            conn.request("GET", path, headers={"User-Agent": "TandemLLM-chat/0.1 (web_fetch)",
                                               "Accept": "text/html,text/plain,*/*;q=0.8"})
            r = conn.getresponse()
            if r.status in (301, 302, 303, 307, 308) and r.getheader("Location"):
                url = urllib.parse.urljoin(url, r.getheader("Location"))
                continue
            body = r.read(max_bytes)
            return r.status, (r.getheader("Content-Type") or "").split(";")[0], body
        finally:
            conn.close()
    raise Refused("too many redirects")


class Proxy:
    """One chat's proxy on its unix socket; `mode()` is asked per connection (on or off)."""

    def __init__(self, path: str, mode, log=None):
        self.path = path
        self.mode = mode
        self.log = log or (lambda line: None)
        self.sock: socket.socket | None = None
        self.stats = {"connections": 0, "refused": 0}

    def start(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(self.path)
        os.chmod(self.path, 0o666)
        s.listen(64)
        self.sock = s
        threading.Thread(target=self._accept, name="chatd-egress", daemon=True).start()

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def _accept(self) -> None:
        while self.sock is not None:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    def _serve(self, c: socket.socket) -> None:
        self.stats["connections"] += 1
        up = None
        try:
            c.settimeout(30)
            head = b""
            while b"\r\n\r\n" not in head:
                d = c.recv(CHUNK)
                if not d:
                    return
                head += d
                if len(head) > 65536:
                    return
            line, _, rest = head.partition(b"\r\n")
            method, target, _ = (line.decode("latin-1").split(" ") + ["", "", ""])[:3]
            if self.mode() != "on":
                raise Refused("the network is off for this chat")
            if method == "CONNECT":
                host, _, port = target.rpartition(":")
                up = connect(host.strip("[]"), int(port or 443))
                c.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                extra = rest.split(b"\r\n\r\n", 1)[1] if b"\r\n\r\n" in rest else b""
                if extra:
                    up.sendall(extra)
            else:
                u = urllib.parse.urlsplit(target)
                if u.scheme != "http" or not u.hostname:
                    raise Refused("send absolute http:// URLs or CONNECT")
                up = connect(u.hostname, u.port or 80)
                hdrs, _, body = rest.partition(b"\r\n\r\n")
                keep = [h for h in hdrs.split(b"\r\n") if h and not h.lower().startswith(
                    (b"proxy-", b"connection:", b"keep-alive:"))]
                path = (u.path or "/") + (f"?{u.query}" if u.query else "")
                up.sendall(f"{method} {path} HTTP/1.1\r\n".encode() + b"\r\n".join(keep) +
                           b"\r\nConnection: close\r\n\r\n" + body)
            c.settimeout(None)
            _splice(c, up)
        except (Refused, ValueError) as exc:
            self.stats["refused"] += 1
            msg = str(exc).encode()
            try:
                c.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nContent-Length: "
                          + str(len(msg) + 1).encode() + b"\r\nConnection: close\r\n\r\n" + msg +
                          b"\n")
            except OSError:
                pass
        except OSError:
            pass
        finally:
            for s in (c, up):
                if s is not None:
                    try:
                        s.close()
                    except OSError:
                        pass


def _splice(a: socket.socket, b: socket.socket) -> None:
    def pump(src, dst):
        try:
            while True:
                d = src.recv(CHUNK)
                if not d:
                    break
                dst.sendall(d)
        except OSError:
            pass
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass
    t = threading.Thread(target=pump, args=(b, a), daemon=True)
    t.start()
    pump(a, b)
    t.join(600)
