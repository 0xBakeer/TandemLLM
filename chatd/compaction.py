"""The context guard and compaction (SRV-351; plan note Chat 7).

Before every model call chatd estimates the request in tokens: the request's characters times the
chat's own tokens-per-character, learned from the last usage frame (the engine's real count over
the characters chatd sent), 0.3 until there is one, plus each image by its pixel size (an image's
base64 is never counted as text). Then:

1. over 80 % of the window (`compact_at`, from the model card's context length), the conversation
   is compacted first: one call writes a summary of everything before the last two user turns;
   from then on the model gets the summary in its system block plus those turns. The database
   keeps every message; the snapshot lists the compaction (`Conversation.compactions`) and a
   `compaction` event tells every device. The request carries `tandem.compaction: true`, so our
   engine does not guess (SRV-351) and serves it as a continuation of the conversation it holds;
   Inside one long run (one user message, many steps: nothing before the last two user turns)
   the run's earlier steps are summarized instead, all but the last KEEP_STEPS, between two steps
   (`step_boundary`); the user's message stays. Live (2026-10-13) a 21-step run hit the guard
   with no compaction, as there was no earlier user turn to summarize. A chat the engine refused
   for memory compacts at 80 % of the refused request as well (`compact_at`; chatd/runs.py).
   The request after a compaction lands under TARGET_SHARE of that limit: the long read_file
   and bash results of what it keeps become stubs (chatd/render.py), and a long run keeps fewer
   steps until it fits;
2. still too long (one huge turn, nothing before the last two to summarize): the reasoning of
   earlier, finished turns is left out of the request. Templates such as Qwen's drop it anyway;
   others keep it, and 28,000 tokens of thinking then fill a 40,960 window (seen live on
   2026-10-12). Inside the running turn it is never dropped;
3. still too long: the run fails before the engine is asked, with the numbers, instead of the
   engine's 400 ("prompt is 40,982 tokens and the context is 40,960");
4. `max_tokens` is set to what the window has left (and the model card's output limit), so a
   long thought ends at the window instead of past it.
"""

from __future__ import annotations

import base64
import binascii
import json
import struct

COMPACT_SHARE = 0.8
DEFAULT_RATIO = 0.3                 # tokens per character until the engine has told us
MIN_ANSWER = 1024                   # the room an answer needs, at least
KEEP_USER_TURNS = 2
SUMMARY_TOKENS = 2048
INSTRUCTIONS = (
    "Write a summary of the conversation above, for yourself, so that you can continue it without "
    "the earlier messages. Keep: what the user wants and every decision they made; facts, names, "
    "numbers, file paths and commands that matter; what was done, what worked and what did not; "
    "the open questions and the next steps. Leave out pleasantries. Plain prose and short lists, "
    "at most 600 words. Do not continue the conversation and do not call tools.")


#: an image's tokens: one per 32 x 32 pixels (Qwen-VL: 16-pixel patches merged 2 x 2) and the two
#: markers around them, as many as the engine's tower can take; a format whose size is not read here
#: counts as IMAGE_UNKNOWN. Live (2026-10-13) five 300 KB screenshots counted as their base64,
#: 2,077,842 characters: the guard said 319,136 tokens where the engine had 16,980
VISION_PX = 32
IMAGE_MAX = 16384
IMAGE_UNKNOWN = 1024


def _size(data: bytes) -> tuple[int, int] | None:
    """(width, height) of a PNG, JPEG or GIF from its header."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        return struct.unpack("<HH", data[6:10])
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
            i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    return None


def image_tokens(url: str) -> int:
    """The tokens of one `image_url`: by its pixel size, never by the length of its base64."""
    try:
        head, b64 = url.split(",", 1)
        data = base64.b64decode(b64 if ";base64" in head else "")
    except (ValueError, binascii.Error):
        return IMAGE_UNKNOWN
    wh = _size(data)
    if not wh:
        return IMAGE_UNKNOWN
    w, h = wh
    return min(IMAGE_MAX, max(1, -(-w // VISION_PX)) * max(1, -(-h // VISION_PX))) + 2


def _split(messages: list) -> tuple[list, int]:
    """The messages with every image's data left out, and the images' tokens."""
    out, tokens = [], 0
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list):
            out.append(m)
            continue
        parts = []
        for p in c:
            url = ((p.get("image_url") or {}).get("url") if isinstance(p, dict) and
                   p.get("type") == "image_url" else None)
            if url:
                tokens += image_tokens(url)
                parts.append({"type": "image_url"})
            else:
                parts.append(p)
        out.append(dict(m, content=parts))
    return out, tokens


def measure(body: dict) -> tuple[int, int]:
    """(the request's characters without image data, its images' tokens)."""
    msgs, img = _split(body.get("messages") or [])
    chars = len(json.dumps(msgs, ensure_ascii=False))
    chars += len(json.dumps(body.get("tools") or [], ensure_ascii=False))
    return chars, img


def estimate(body: dict, ratio: float | None) -> int:
    chars, img = measure(body)
    return int(chars * (ratio or DEFAULT_RATIO)) + img + 16


def learn(prompt_tokens: int, request_chars: int, image_tokens: int = 0) -> float | None:
    """Tokens per character of text: the engine's count less the images' tokens, over the
    characters chatd sent without the image data (`measure`)."""
    text = prompt_tokens - image_tokens
    if text > 0 and request_chars > 200:
        return max(0.15, min(0.8, text / request_chars))
    return None


def compact_at(window: int, memory: int | None = None) -> int:
    """The estimate over which the guard compacts: 80 % of the window, or of the smallest request
    the engine refused for memory in this chat (`Conv.memory_limit`) when that is less; 0 when
    neither is known. Live (2026-10-17) a run grew back to 215,247 tokens after a compaction, under
    80 % of its 262,144 window, and the engine refused every call: the prefill needed 53.3 GiB
    of a 52.6 GiB peak while another chat ran on its other slot."""
    at = [int(x * COMPACT_SHARE) for x in (window, memory) if x]
    return min(at) if at else 0


def boundary(branch: list[dict]) -> int | None:
    """The index of the first message kept after a compaction (the second-to-last user turn),
    or None when there is nothing before it to summarize."""
    users = [i for i, m in enumerate(branch) if m["role"] == "user"]
    if len(users) < KEEP_USER_TURNS + 1:
        return None
    i = users[-KEEP_USER_TURNS]
    return i if i > 0 else None


#: a long run's steps kept whole after its earlier ones are compacted, at most
KEEP_STEPS = 3
#: the request after a compaction, at most, as a share of the limit that set it off (`compact_at`):
#: fewer steps are kept until it fits (one at the least)
TARGET_SHARE = 0.5


def step_boundary(branch: list[dict], prior_step: int = 0,
                  keep: int = KEEP_STEPS) -> tuple[int, int] | None:
    """A long run's compaction: (the index of the last assistant message, the first of its steps
    kept), when it has more than `keep` steps after the ones a compaction already took
    (`prior_step`); None otherwise. Its user message stays, so the request keeps one user turn."""
    if not branch or branch[-1].get("role") != "assistant" or len(branch) < 2:
        return None
    done = sorted({int(p.get("step") or 1) for p in branch[-1].get("parts") or []})
    done = [x for x in done if x >= prior_step]
    if len(done) <= keep:
        return None
    return len(branch) - 1, done[-keep]


def shrink(messages: list[dict], budget_chars: int) -> list[dict]:
    """A compaction request's history that fits: reasoning dropped, then each long message cut
    in its middle, oldest first."""
    out = [dict(m) for m in messages]
    for m in out:
        m.pop("reasoning_content", None)
    total = len(json.dumps(out, ensure_ascii=False))
    for m in out:
        if total <= budget_chars:
            break
        c = m.get("content")
        if isinstance(c, str) and len(c) > 2000:
            keep = max(1000, len(c) - (total - budget_chars))
            cut = c[:keep // 2] + "\n[... cut for the summary ...]\n" + c[-keep // 2:]
            total -= len(c) - len(cut)
            m["content"] = cut
    return out
