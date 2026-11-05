"""The server a measurement talks to: the checks before anything is measured (status ok, no response
cache, idle, answers raw), its tokenizer, its process (the knobs /health does not show), and the
names a page gives it. Shared by tools/measure and tools/release_report."""

from __future__ import annotations

import os
import re
import time

from tools import parallel_card as pc
from tools.measure import client as cmod
from tools.measure.prompts import sized


class Refused(SystemExit):
    def __init__(self, why: str):
        super().__init__(f"[report] refused: {why}")
        self.why = why


def bare_refusal(h: dict) -> str | None:
    """Why this server's raw requests would not reach the bare engine alone, or None."""
    if h.get("status") != "ok":
        return f"/health says status {h.get('status')!r}" + (f" ({h['error']})" if h.get("error") else "")
    if (h.get("cache") or {}).get("response_cache"):
        return ("the response cache is on: an identical greedy request could be answered from it. Start the "
                "server with RESPONSE_CACHE=off, or use launch mode")
    return None


def model_card(base: str, h: dict) -> dict:
    d = cmod.get_json(base.rstrip("/") + "/v1/models")
    for m in d.get("data") or []:
        if m.get("id") == h.get("model") or (m.get("tandem") or {}).get("kind") == "engine":
            return m
    return {}


def local_pid(base: str) -> int | None:
    """The pid listening on a local server's port (Linux, a process of this user), read from /proc; None
    elsewhere. Its environment and command line give the knobs /health does not show."""
    from urllib.parse import urlsplit
    u = urlsplit(base)
    if u.hostname not in ("127.0.0.1", "localhost", "::1") or not os.path.isdir("/proc/net"):
        return None
    port = f"{u.port or 80:04X}"
    inodes = set()
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            for line in open(f).read().splitlines()[1:]:
                x = line.split()
                if x[1].rsplit(":", 1)[1] == port and x[3] == "0A":
                    inodes.add(x[9])
        except (OSError, IndexError):
            continue
    if not inodes:
        return None
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            for fd in os.listdir(f"/proc/{d}/fd"):
                link = os.readlink(f"/proc/{d}/fd/{fd}")
                if link.startswith("socket:[") and link[8:-1] in inodes:
                    return int(d)
        except OSError:
            continue
    return None


def tokenizer_for(a, h: dict):
    path = a.tokenizer or (h.get("weights") or {}).get("weights")
    if path and os.path.isdir(path):
        try:
            return pc.load_tokenizer(path)
        except Exception as exc:                                     # noqa: BLE001
            raise Refused(f"the model's tokenizer did not load from the weights folder ({type(exc).__name__}); "
                          "run with the engine's Python, or pass --tokenizer DIR")
    if is_fake(h):
        return None
    raise Refused("the model's tokenizer is needed to render prompts from its chat template, and the weights "
                  "folder /health names is not on this machine: pass --tokenizer DIR")


def is_fake(h: dict) -> bool:
    """A server that names no weights folder: the fake engine (server/fake_engine.py), whose prompts are
    plain text."""
    return not (h.get("weights") or {}).get("weights")


def wait_idle(base: str, wait_s: float) -> dict:
    t0 = time.monotonic()
    while True:
        h = cmod.health(base)
        why = cmod.busy(h)
        if not why:
            return h
        if time.monotonic() - t0 >= wait_s:
            raise Refused(f"the server is busy ({why}); a report measures an idle server. Try again when it is "
                          "idle, or pass --wait-idle SECONDS")
        time.sleep(2.0)


def probe(cl, prose) -> dict:
    """One ~2k request, 64 tokens: the raw check and the rates the estimate uses."""
    p, _ = sized(cl, prose, 2048, "\n\nSummarise the text above.", tag="probe")
    r = cl.stream_rec(p, 64, seed=1)
    if r.get("error"):
        raise Refused(f"the probe request failed: {r['error']}")
    if not r.get("raw"):
        raise Refused(f"the probe was not served raw ({r['raw_why']}): the chat layer is in front of "
                      "/v1/completions on this server")
    return {"prefill_tok_s": r.get("prefill_tok_s") or 1000.0, "decode_tok_s": r.get("decode_tok_s") or 30.0,
            "ttft_s": r.get("ttft_s"), "prompt_tokens": r.get("prompt_tokens")}


def label_of(card: dict, h: dict) -> str:
    """The model's name for the page: the card's label, unless it is only the served name (as
    "TandemLLM"), then the card's model id, then the weights' name."""
    t = card.get("tandem") or {}
    served = {str(h.get("model") or ""), "TandemLLM"}
    for x in (t.get("label"), t.get("model"), (h.get("weights") or {}).get("hf_repo"),
              (h.get("weights") or {}).get("name")):
        if x and str(x) not in served:
            return re.sub(r"[-_](exl3|gguf|nvfp4|fp8|awq|gptq|bf16)$", "", str(x).split("/")[-1], flags=re.I)
    return str(h.get("model") or "model")


def hardware_label(prof) -> str:
    fp = prof.fp
    if prof.id == "gb10":
        return "one DGX Spark (GB10)"
    if prof.id == "fake":
        return "no GPU (the fake engine)"
    n = f"{fp.count} x " if fp.count and fp.count > 1 else "one "
    return f"{n}{fp.name}" + (f" ({fp.memory_gib:g} GiB)" if fp.memory_gib else "")
