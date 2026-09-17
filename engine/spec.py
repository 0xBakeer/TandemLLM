"""Greedy decoding, with and without speculation, and the accounting that compares them.

The speculative loop is the ordinary one: a drafter proposes m tokens, the engine verifies the
block `[last] + draft` in a single pass, the longest prefix of the draft the model agrees with is
accepted, and the model's own token at the first disagreement is accepted as well -- so a block
always yields at least one token and at most m + 1.

What is specific to this model is the rollback. Eleven twelfths of the layers keep a recurrent state
instead of a cache, and the state after a partial accept has to be reconstructed rather than
truncated. `Qwen38Engine.rollback_to` does that from the block trace; the loop here only decides how
many tokens to keep.

The invariant that makes any of this trustworthy: **greedy output must not depend on the drafter.**
`verify_lossless` checks exactly that, with a drafter built to be wrong on purpose.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from engine.drafters import Drafter


@dataclass
class DecodeStats:
    tokens: int = 0
    blocks: int = 0
    drafted: int = 0
    accepted: int = 0
    rollbacks: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    rollback_s: float = 0.0
    draft_s: float = 0.0
    per_block: list[int] = field(default_factory=list)

    @property
    def tok_s(self) -> float:
        return self.tokens / self.decode_s if self.decode_s else 0.0

    @property
    def accept_len(self) -> float:
        return self.tokens / self.blocks if self.blocks else 0.0

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    def line(self, label: str) -> str:
        return (f"{label:22s} {self.tokens:5d} tok  {self.tok_s:6.2f} tok/s  "
                f"blocks {self.blocks:4d}  accepted/block {self.accept_len:5.2f}  "
                f"draft acceptance {self.accept_rate * 100:5.1f} %  "
                f"rollbacks {self.rollbacks:4d}  "
                f"[prefill {self.prefill_s * 1e3:6.0f} ms  decode {self.decode_s:6.2f} s  "
                f"draft {self.draft_s * 1e3:5.0f} ms  rollback {self.rollback_s * 1e3:6.0f} ms]")


def _stop(tok: int, eos: list[int]) -> bool:
    return tok in eos


def generate_greedy(eng, prompt: torch.Tensor, max_new: int,
                    eos: list[int] | None = None) -> tuple[list[int], DecodeStats]:
    """One token per forward pass. The baseline every speculative run must reproduce exactly."""
    eos = eos or []
    st = DecodeStats()
    eng.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = eng.forward(prompt, start=0, last_only=True)
    torch.cuda.synchronize()
    st.prefill_s = time.perf_counter() - t0
    pos = prompt.numel()
    tok = int(logits[0, -1].argmax())
    out = [tok]
    t0 = time.perf_counter()
    with torch.no_grad():
        while len(out) < max_new and not _stop(tok, eos):
            logits = eng.forward(torch.tensor([tok], device=prompt.device), start=pos,
                                 last_only=True)
            pos += 1
            tok = int(logits[0, -1].argmax())
            out.append(tok)
            st.blocks += 1
    torch.cuda.synchronize()
    st.decode_s = time.perf_counter() - t0
    st.tokens = len(out) - 1
    return out, st


def generate_spec(eng, prompt: torch.Tensor, max_new: int, drafter: Drafter, k: int,
                  eos: list[int] | None = None) -> tuple[list[int], DecodeStats]:
    eos = eos or []
    st = DecodeStats()
    eng.reset()
    drafter.reset()
    prompt_list = prompt.tolist()
    if hasattr(drafter, "prime"):
        drafter.prime(prompt_list)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = eng.forward(prompt, start=0, last_only=True)
    torch.cuda.synchronize()
    st.prefill_s = time.perf_counter() - t0
    pos = prompt.numel()
    tok = int(logits[0, -1].argmax())
    out = [tok]
    drafter.observe([tok])
    ctx = prompt_list + [tok]

    torch.cuda.synchronize()
    t_dec = time.perf_counter()
    with torch.no_grad():
        while len(out) < max_new and not _stop(tok, eos):
            td = time.perf_counter()
            draft = drafter.propose(ctx, min(k, max_new - len(out)))
            st.draft_s += time.perf_counter() - td
            if not draft:
                logits = eng.forward(torch.tensor([tok], device=prompt.device), start=pos,
                                     last_only=True)
                pos += 1
                tok = int(logits[0, -1].argmax())
                out.append(tok)
                ctx.append(tok)
                drafter.observe([tok])
                st.blocks += 1
                continue
            block = torch.tensor([tok] + draft, device=prompt.device)
            lg = eng.forward_block(block, start=pos)
            picks = lg.argmax(-1).tolist()
            n = 0
            for i, d in enumerate(draft):
                if picks[i] != d:
                    break
                n += 1
            new = draft[:n] + [picks[n]]
            st.blocks += 1
            st.drafted += len(draft)
            st.accepted += n
            st.per_block.append(len(new))
            if n < len(draft):
                tr = time.perf_counter()
                eng.rollback_to(n + 1)
                torch.cuda.synchronize()
                st.rollback_s += time.perf_counter() - tr
                st.rollbacks += 1
            pos += n + 1
            for t in new:
                out.append(t)
                ctx.append(t)
                if _stop(t, eos):
                    break
            drafter.observe(new)
            tok = out[-1]
    torch.cuda.synchronize()
    st.decode_s = time.perf_counter() - t_dec
    st.tokens = len(out) - 1
    return out, st
