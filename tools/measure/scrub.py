"""What a shared page must not carry: home paths and other absolute paths (reduced to the file's name),
host names, user names, IP addresses and URLs of the box. A run records its own private terms (the
host name, the user name, the home directory) in data.json under `private`; the page never shows them
and `leaks(html, data)` finds any that slipped through."""

from __future__ import annotations

import getpass
import os
import re
import socket

PATH = re.compile(r"(?<![\w.])(?:~|/home|/Users|/root|/tmp|/private|/mnt|/opt|/srv|/data|/var|/media|/nvme\w*)"
                  r"(?:/[^\s\"'<>,;)\]}]+)+")
URL = re.compile(r"\bhttps?://[^\s\"'<>]+")
IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
LAN = re.compile(r"\b[\w-]+\.(?:local|lan|internal|home|localdomain|corp)\b", re.I)


def private_terms() -> dict:
    """This box's own names, recorded at run time for the scrub (never shown)."""
    out = {"hosts": [], "users": [], "homes": []}
    try:
        h = socket.gethostname()
        out["hosts"] = sorted({h, h.split(".")[0]})
    except OSError:
        pass
    try:
        out["users"] = [getpass.getuser()]
    except Exception:                                                # noqa: BLE001
        pass
    out["homes"] = [os.path.expanduser("~")]
    return out


def _terms(private: dict | None) -> list[str]:
    """The private names to remove; a name that is a word of the page's own hardware label (a host
    called after its board, as "dgx") names nothing and is left to that label."""
    allow = {w.lower() for w in re.findall(r"[\w-]+", " ".join((private or {}).get("allow") or []))}
    t = []
    for k in ("homes", "hosts", "users"):
        t += [x for x in (private or {}).get(k) or [] if x and len(x) >= 3 and x.lower() not in allow]
    return sorted(set(t), key=len, reverse=True)


def text(s: str, private: dict | None = None) -> str:
    """One string, public: paths to their last part, URLs and addresses out, private names out."""
    if not isinstance(s, str):
        return s
    for home in (private or {}).get("homes") or []:
        if home and len(home) > 1:
            s = s.replace(home, "~")
    s = URL.sub("[local server]", s)
    s = PATH.sub(lambda m: os.path.basename(m.group(0).rstrip("/")) or "[path]", s)
    s = IP.sub("[address]", s)
    s = LAN.sub("[host]", s)
    for t in _terms(private):
        s = re.sub(rf"(?<![\w-]){re.escape(t)}(?![\w-])", "[box]", s, flags=re.I)
    return s


def deep(x, private: dict | None = None):
    """A JSON value with every string (keys too) made public."""
    if isinstance(x, dict):
        return {text(k, private) if isinstance(k, str) else k: deep(v, private) for k, v in x.items()}
    if isinstance(x, list):
        return [deep(v, private) for v in x]
    if isinstance(x, str):
        return text(x, private)
    return x


def leaks(html: str, private: dict | None = None) -> list[str]:
    """What in a rendered page still names the box: private terms, home paths, addresses, URLs other
    than the allowed CDNs."""
    found = []
    for t in _terms(private):
        if re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", html, re.I):
            found.append(f"private name {t!r}")
    for m in PATH.finditer(html):
        found.append(f"path {m.group(0)[:60]!r}")
    for m in IP.finditer(html):
        found.append(f"address {m.group(0)!r}")
    for m in URL.finditer(html):
        if not m.group(0).startswith(("https://cdnjs.cloudflare.com/", "https://fonts.googleapis.com/",
                                      "https://fonts.gstatic.com/", "http://www.w3.org/2000/svg",
                                      "https://github.com/0xBakeer/TandemLLM")):
            found.append(f"url {m.group(0)[:60]!r}")
    for m in LAN.finditer(html):
        found.append(f"host {m.group(0)!r}")
    return found
