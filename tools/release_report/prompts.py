"""The report's prompt text: real prose, a fresh nonce line first, sliced to a token count.

`--prose` picks the text, read from local files only (nothing is downloaded):

  repo            bench/heldout_prose.txt (1,224 words), repeated to the length: always there, but a
                  long prompt repeats the same text, which the page says
  wikitext2       WikiText-2's train split from the local Hugging Face cache (Salesforce/wikitext,
                  wikitext-2-raw-v1), its rows from 3,000 on (rows 0-2,554 seeded the lookup store,
                  tools/longprompts.py): about 2 M tokens of prose that does not repeat
  FILE            any UTF-8 text file

Each prompt takes its text from its own offset, so two prompts of a run share no text, and starts
with its own nonce line, so no cache holds it."""

from __future__ import annotations

import glob
import hashlib
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
NONCE_LINE = "Request nonce: {nonce}\n\n"


def nonce(*parts) -> str:
    return hashlib.sha256(("rr|" + "|".join(map(str, parts)) + "|" + str(time.time_ns())).encode()).hexdigest()[:24]


def wikitext2_text() -> str | None:
    pat = os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/"
                             "wikitext-2-raw-v1/train-*.parquet")
    files = sorted(glob.glob(pat))
    if not files:
        return None
    import pyarrow.parquet as pq
    rows = pq.read_table(files[0], columns=["text"]).column("text").to_pylist()
    return "".join(r for r in rows[3000:] if r)


class Prose:
    """The prose and its tokens (with the model's tokenizer; characters for a server without one)."""

    def __init__(self, spec: str = "repo", tok=None):
        self.spec = spec
        if spec == "repo":
            text = (REPO / "bench" / "heldout_prose.txt").read_text()
            self.name, self.repeats = "bench/heldout_prose.txt", True
        elif spec == "wikitext2":
            text = wikitext2_text()
            if text is None:
                raise SystemExit("--prose wikitext2: no WikiText-2 train file in the local Hugging Face cache "
                                 "(Salesforce/wikitext, wikitext-2-raw-v1); pass --prose repo or a text file")
            self.name, self.repeats = "WikiText-2 train, rows 3000 on", False
        else:
            text = Path(spec).read_text(encoding="utf-8")
            self.name, self.repeats = os.path.basename(spec), False
        self.text = " ".join(text.split()) if spec == "repo" else text
        self.sha256 = hashlib.sha256(self.text.encode()).hexdigest()
        self.tok = tok
        if tok is not None:
            enc = tok(self.text, add_special_tokens=False, return_offsets_mapping=True)
            self.ids = enc["input_ids"]
            self.offsets = enc["offset_mapping"]
        else:
            self.ids, self.offsets = None, None
        self.size = len(self.ids) if self.ids is not None else len(self.text)
        self.cursor = 0

    def describe(self) -> dict:
        return {"source": self.name, "sha256": self.sha256, "tokens": self.size if self.tok else None,
                "chars": len(self.text), "repeats": self.repeats}

    def take(self, n: int) -> str:
        """The next `n` tokens of prose (characters without a tokenizer), wrapping around; a repeat of
        the text is counted in `self.repeats` when a slice wraps."""
        if n <= 0:
            return ""
        out, left = [], n
        while left > 0:
            k = min(left, self.size - self.cursor)
            if self.offsets is not None:
                a = self.offsets[self.cursor][0]
                b = self.offsets[self.cursor + k - 1][1]
                out.append(self.text[a:b])
            else:
                out.append(self.text[self.cursor:self.cursor + k])
            self.cursor += k
            left -= k
            if self.cursor >= self.size:
                self.cursor = 0
                if n > k or left > 0:
                    self.repeats = True
        return " ".join(out)


def sized(cl, prose: Prose, tokens: int, tail: str, system: str | None = None, tag: str = "p") -> tuple[str, int]:
    """A rendered raw prompt of about `tokens` tokens: a nonce line, prose, then `tail` (the task), in one
    user turn (after `system`, when given). Returns (prompt text, its token count)."""
    head = NONCE_LINE.format(nonce=nonce(tag, tokens))

    def msgs(body: str) -> list:
        m = [{"role": "user", "content": head + body + tail}]
        return ([{"role": "system", "content": system}] + m) if system else m
    memo = cl.__dict__.setdefault("_base_tokens", {})
    key = (system, tail)
    if key not in memo:                    # the template and the task without prose, once a shape
        memo[key] = cl.count(cl.render(msgs(""))) if cl.tok is not None else len(cl.render(msgs("")))
    base = memo[key]
    body = prose.take(max(0, tokens - base))
    prompt = cl.render(msgs(body))
    n = cl.count(prompt) if cl.tok is not None else len(prompt)
    return prompt, n


def chat(cl, messages: list) -> str:
    """A conversation rendered for `/v1/completions` (the template with thinking off)."""
    return cl.render(messages)
