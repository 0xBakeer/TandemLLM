"""A fake generative model for the tests and the fake engine: no weights, no torch.

Every word of its reply is one token, each taking `delay_s`, so the token cap and the time budget
are exact in a test. The reply is `reply(messages)`; by default it is "now " and the first five
words of the event in the narrator's prompt ("Event: ..."), visibly not a template. `fail` makes
`generate` raise; `loads` counts the loads.
"""

from __future__ import annotations

import re
import time

from engine.generative.base import GenerativeModel


def _default(messages: list[dict]) -> str:
    text = str((messages[-1] or {}).get("content") or "") if messages else ""
    m = re.search(r"^Event: (.+)$", text, re.M)
    return "now " + (" ".join(m.group(1).split()[:5]) if m else "working")


class FakeGenerative(GenerativeModel):
    family = "fake"
    owned_by = "tests"
    memory_claim_gib = 0.0

    def __init__(self, name: str = "fake-narrator", *, delay_s: float = 0.0, reply=None,
                 clock=time.monotonic, sleep=time.sleep):
        self.name = name
        self.delay_s = float(delay_s)
        self.reply = reply or _default
        self.clock, self.sleep = clock, sleep
        self.fail = False
        self.loaded = False
        self.loads = 0
        self.calls: list[dict] = []
        self._words: list[str] = []

    def load(self, where) -> None:
        self.loaded = True
        self.loads += 1

    def unload(self) -> None:
        self.loaded = False

    def encode(self, messages: list[dict]) -> list[int]:
        self._words = str(self.reply(messages)).split(" ")
        self.calls.append({"messages": messages})
        return [ord(c) for c in str(messages[-1].get("content") or "")][:512] or [0]

    def generate(self, ids: list[int], max_new: int, deadline: float | None) -> dict:
        if self.fail:
            raise RuntimeError("the fake generative model failed on purpose")
        out, finish = [], "stop"
        if deadline is not None and self.clock() >= deadline:
            return {"ids": [], "finish": "time", "prefilled": 0, "reused": 0}
        for i, _ in enumerate(self._words):
            if self.delay_s:
                self.sleep(self.delay_s)
            out.append(i)
            if len(out) >= max_new and len(out) < len(self._words):
                finish = "length"
                break
            if deadline is not None and self.clock() >= deadline and len(out) < len(self._words):
                finish = "time"
                break
        self.calls[-1]["finish"] = finish
        return {"ids": out, "finish": finish, "prefilled": len(ids), "reused": 0}

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._words[i] for i in ids)

    def precision(self) -> str | None:
        return "none (a test double)" if self.loaded else None
