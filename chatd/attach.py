"""A safety net for attachments: none is ever inlined past the window (SRV-487).

Live (2026-10-15, chat 01M4ZSH3DEEX2G3X11YK1RND30): a 9.5 MB Medium article saved as HTML, its
figures embedded as base64 `data:` URIs, went into the prompt whole (2,363,997 tokens), and the run
stopped with "This conversation is about 2,875,364 tokens ... Start a new chat", which blamed the
conversation for one file.

    save_upload(ws, name, data)  every attachment of a message goes to the chat's workspace under
                                 uploads/ when the message is sent (a free name); the sandbox's
                                 tools work on it from there
    budget(window, used)         the most tokens one message's text attachments may put in the
                                 prompt: a quarter of the window's free room, at most
                                 INLINE_MAX_TOKENS
    line(a)                      what the prompt gets instead of a text file over the budget: one
                                 line with its name, size, tokens and path

The placement is decided once, at the send (`inline`, `path` on the attachment), so every later turn
renders the same bytes (chatd/render.py).
"""

from __future__ import annotations

import os

INLINE_MAX_TOKENS = 32_000
INLINE_SHARE = 0.25
LINE_TOKENS = 64                    # what the line costs in the prompt, about
UPLOADS = "uploads"


def budget(window: int, used: int) -> int:
    """The tokens one message's text attachments may take: a quarter of the free room, at most
    INLINE_MAX_TOKENS; INLINE_MAX_TOKENS when the window is unknown."""
    if not window:
        return INLINE_MAX_TOKENS
    return max(0, min(INLINE_MAX_TOKENS, int((window - used) * INLINE_SHARE)))


def size_words(n: int) -> str:
    return (f"{n / 1024 / 1024:.1f} MB" if n >= 1024 * 1024 else f"{n / 1024:.0f} KB"
            if n >= 1024 else f"{n} bytes")


def line(a: dict) -> str:
    """The prompt's line for a text attachment over the budget."""
    where = (f"It is saved at {a['path']}." if a.get("path") else
             "It was attached before attachments were saved to the workspace.")
    return (f"\n\n[Attached file {a.get('name') or 'file'}: "
            f"{size_words(int(a.get('bytes') or 0))}, about {int(a.get('tokens') or 0):,} tokens: "
            f"too long to include. {where}]")


def save_upload(ws_dir: str, name: str, data: bytes) -> str:
    """The file under the workspace's uploads/ with a free name; its path as the sandbox sees it."""
    up = os.path.join(ws_dir, UPLOADS)
    os.makedirs(up, exist_ok=True)
    base = os.path.basename(name) or "file"
    stem, ext = os.path.splitext(base)
    i, pick = 1, base
    while os.path.exists(os.path.join(up, pick)):
        i += 1
        pick = f"{stem}-{i}{ext}"
    with open(os.path.join(up, pick), "wb") as fh:
        fh.write(data)
    return f"/workspace/{UPLOADS}/{pick}"
