"""Scanned PDFs and images, read once when their message is sent.

An OCR pass turns a PDF or an image into markdown that is saved next to the upload as
`uploads/<name>.md` and stored as a text attachment of its own (`Service._ocr_ws`,
chatd/service.py). From there the existing flow takes over unchanged (chatd/attach.py,
chatd/render.py): a small text goes into the prompt whole, a big one stays on disk and the
model pages through it with read_file — which is what a scanned PDF could never do before,
since the bytes are a picture of text (`a binary file, not shown`, chatd/tools.py).

The engine is Indra OCR (https://huggingface.co/datalab-to/indra-ocr): a 1M-parameter
pipeline that runs on CPU, `pip install indra-ocr`, ~18 MB of models on first use. Without
that package this module is inert — `wanted` still says which files it would read, but
`engine` finds nothing, no message is delayed, and nothing changes.

    wanted(name, mime)    an OCR candidate: a PDF or an image
    convert(host_path)    the pages' markdown, joined; None with no engine, or no text

Environment: `TANDEM_OCR=off` forces the whole path off; `TANDEM_OCR_THREADS` (4) is the
CPU budget of one conversion; `TANDEM_OCR_MAX_MB` (64) skips absurd inputs. The pass runs
on the sending request's thread, so a long PDF delays its own message and nothing else.
"""

from __future__ import annotations

import os
import stat
import threading
from typing import Any

OFF = os.environ.get("TANDEM_OCR", "").strip().lower() in ("off", "0", "no")


def _num(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


THREADS = max(1, int(_num("TANDEM_OCR_THREADS", 4)))
MAX_MB = _num("TANDEM_OCR_MAX_MB", 64.0)

#: the file names OCR reads; a PDF by name too, because a client that guesses mime types
#: sends `application/octet-stream` and the bytes are the same pages
EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff", ".bmp")


def wanted(name: str, mime: str) -> bool:
    """Whether an attachment is an OCR candidate: a PDF or an image, any of their spellings."""
    if OFF:
        return False
    n = (name or "").lower()
    m = (mime or "").lower()
    return n.endswith(EXTENSIONS) or m == "application/pdf" or m.startswith("image/")


_conv: "Any" = None                  # None: not tried yet; False: tried, no engine; else the load
_lock = threading.Lock()


def engine() -> "Any":
    """The one Converter for this process, or None: loaded once, lazily, off the store's locks.

    Indra downloads its models on the first load, so this can block on the network — only an
    OCR candidate's send pays for it, once. An ImportError (the package is not installed), a
    failed download or any other load error leaves the engine off for the process's life.
    """
    global _conv
    if _conv is None:
        with _lock:
            if _conv is None:
                try:
                    import indra_ocr
                    _conv = indra_ocr.load(threads=THREADS)
                except Exception:
                    _conv = False
    return _conv if _conv is not False else None


def engine_key() -> str:
    """The engine's identity for a cache key (empty with none): a memo keyed on the file's bytes
    alone would serve one engine's text after another engine had replaced it."""
    eng = engine()
    return "" if eng is None else str(id(eng))


def convert(host_path: str) -> str | None:
    """The file's pages as markdown, page breaks joined; None when it will not or cannot."""
    # the sandbox owns uploads/, but between the file's placement and this call the directory's
    # owner can swap the placed file for a symlink to any host file — and the OCR text would go
    # into the model's prompt. So: lstat, no following, and only a regular file is read. A
    # refusal is None, like any unreadable scan: the message keeps the file itself, unscanned.
    try:
        st = os.stat(host_path, follow_symlinks=False)
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    eng = engine()
    if eng is None:
        return None
    try:
        if MAX_MB and st.st_size > MAX_MB * 1_000_000:
            return None
        parts = [page.markdown for _i, page in eng.convert(host_path)]
    except Exception:
        # a broken PDF, a decoder's failure: the message keeps the file itself, unscanned
        return None
    text = "\n\n".join(p.strip() for p in parts if p and p.strip())
    return text or None
