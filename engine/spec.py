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

import math
import time
from dataclasses import dataclass, field

import torch

from engine.drafters import Drafter


@dataclass
class Relax:
    """The relaxed accept rule. Off by default, and it is the only thing in this file that can
    change what the engine writes.

    Greedy verification accepts a drafted token only when it is the target's argmax, and that is
    what makes the speculative path's output identical to the unspeculated one. Two knobs loosen it:

        tau    accept `t` when `p(t) >= tau * p(argmax)`; `tau = 1` is the greedy rule
        rank   accept `t` when it is among the target's `rank` most likely; `rank = 1` is greedy

    Both are evaluated on logits and neither needs a softmax: `p(t) >= tau * p1` is
    `logit(t) >= logit_max + log(tau)`, and a rank test is a `topk`. A block whose tokens were
    accepted this way is still verified in one pass and still costs one pass; what it loses is the
    guarantee, and `LIMITATIONS.md` says so.
    """

    tau: float = 1.0
    rank: int = 1

    @property
    def on(self) -> bool:
        return self.tau < 1.0 or self.rank > 1

    def accepts(self, row: torch.Tensor, token: int, argmax: int) -> bool:
        if token == argmax:
            return True
        if not self.on:
            return False
        if self.tau < 1.0:
            if float(row[token]) >= float(row[argmax]) + math.log(self.tau):
                return True
        if self.rank > 1:
            top = torch.topk(row, self.rank).indices
            if bool((top == token).any()):
                return True
        return False


@dataclass
class DecodeStats:
    tokens: int = 0
    blocks: int = 0
    drafted: int = 0
    accepted: int = 0
    rollbacks: int = 0
    nodes: int = 0
    relaxed: int = 0
    # For every drafted token the target did not agree with: where that token ranked in the
    # target's own distribution, and its probability as a fraction of the argmax's. This is what
    # says whether a tolerance can ever recover a rejection, and it costs one topk on a block that
    # was going to be rolled back anyway.
    miss_rank: list[int] = field(default_factory=list)
    miss_ratio: list[float] = field(default_factory=list)
    prefill_s: float = 0.0
    decode_s: float = 0.0
    rollback_s: float = 0.0
    draft_s: float = 0.0
    per_block: list[int] = field(default_factory=list)
    # Top-1 minus top-2 logit at each greedy step. A losslessness gate that fails at a position
    # where this is near zero is measuring the prompt, not the engine -- the 07:30 entry in the
    # ledger is that mistake made once already.
    gaps: list[float] = field(default_factory=list)
    tops: list[float] = field(default_factory=list)

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
                + (f"nodes/block {self.nodes / self.blocks:5.1f}  " if self.nodes else "") +
                f"[prefill {self.prefill_s * 1e3:6.0f} ms  decode {self.decode_s:6.2f} s  "
                f"draft {self.draft_s * 1e3:5.0f} ms  rollback {self.rollback_s * 1e3:6.0f} ms]")


def _stop(tok: int, eos: list[int]) -> bool:
    return tok in eos


def generate_greedy(eng, prompt: torch.Tensor, max_new: int,
                    eos: list[int] | None = None,
                    record_gaps: bool = False) -> tuple[list[int], DecodeStats]:
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

    def gap(lg):
        if not record_gaps:
            return
        two = lg[0, -1].float().topk(2).values
        st.gaps.append(float(two[0] - two[1]))
        st.tops.append(float(two[0]))

    gap(logits)
    tok = int(logits[0, -1].argmax())
    out = [tok]
    t0 = time.perf_counter()
    with torch.no_grad():
        while len(out) < max_new and not _stop(tok, eos):
            logits = eng.forward(torch.tensor([tok], device=prompt.device), start=pos,
                                 last_only=True)
            pos += 1
            gap(logits)
            tok = int(logits[0, -1].argmax())
            out.append(tok)
            st.blocks += 1
    torch.cuda.synchronize()
    st.decode_s = time.perf_counter() - t0
    st.tokens = len(out) - 1
    return out, st


def generate_spec(eng, prompt: torch.Tensor, max_new: int, drafter: Drafter, k: int,
                  eos: list[int] | None = None,
                  relax: Relax | None = None,
                  profile_misses: bool = False) -> tuple[list[int], DecodeStats]:
    eos = eos or []
    relax = relax or Relax()
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
        if hasattr(drafter, "sync"):
            drafter.sync(prompt_list, eng.hidden_post_norm[0], 0)
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
                # A drafter may decline, and the lookup drafter declines on most steps. A drafter
                # that holds a prediction head must not, and the constraint is not a preference:
                # the head conditions its next draft on the target's hidden state at the last
                # committed position, and a single-token step never computes that row. It computes
                # the hidden state of the token it was given, which produced the new token; the new
                # token's own hidden state does not exist until the next forward pass. So a head
                # cannot be brought current here, and `MergedRouter` is built never to reach this
                # line: the head always proposes something, and the router prices it rather than
                # silencing it.
                #
                # A drafter whose own cache is indexed by ABSOLUTE POSITION -- the block drafter's
                # is -- is the opposite case. Its conditioning for position `pos` is the target's
                # hidden state of the token AT `pos`, which this forward does compute, so it can
                # be brought current, and it must be: leave the gap and it falls permanently one
                # position behind the target and declines for ever after.
                prev = tok
                logits = eng.forward(torch.tensor([tok], device=prompt.device), start=pos,
                                     last_only=True)
                if hasattr(drafter, "sync"):
                    drafter.sync([prev], eng.hidden_post_norm[0], pos)
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
                if picks[i] == d:
                    n += 1
                    continue
                if profile_misses:
                    # Two reductions over a 248,320-wide row and a sync, on a block that was about
                    # to be rolled back. Small, but not free, so it is off while anything is timed.
                    row = lg[i].float()
                    st.miss_rank.append(int((row > row[d]).sum()) + 1)
                    st.miss_ratio.append(float(torch.exp(row[d] - row[picks[i]])))
                if relax.on and relax.accepts(lg[i], d, picks[i]):
                    # The draft token stands, and every logit after it in this block was already
                    # computed conditioned on it, so the rest of the block needs no recomputation.
                    st.relaxed += 1
                    n += 1
                    continue
                break
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
            if hasattr(drafter, "sync"):
                drafter.sync([int(x) for x in block[:n + 1]], eng.hidden_post_norm[0, :n + 1], pos)
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


def generate_spec_tree(eng, prompt: torch.Tensor, max_new: int, drafter, k: int,
                       eos: list[int] | None = None) -> tuple[list[int], DecodeStats]:
    """The same loop, with a draft TREE instead of a chain.

    The only structural difference is what a rejection costs. A chain that is wrong at slot 2 throws
    away slots 3..k with it; a tree keeps whatever branch the target did take. On this board that
    trade is close to free -- the verify step reads the same 19.4 GB of weights whether it carries
    two rows or sixteen -- so the drafter is asked for width and the node budget, not the depth, is
    what the router prices.

    `drafter.propose_tree(context, k)` returns an `engine.tree.DraftTree` or None. None is the same
    decline the chain loop handles: one ordinary step, with the drafter brought current afterwards.
    """
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
        if hasattr(drafter, "sync"):
            drafter.sync(prompt_list, eng.hidden_post_norm[0], 0)
    torch.cuda.synchronize()
    st.prefill_s = time.perf_counter() - t0
    pos = prompt.numel()
    tok = int(logits[0, -1].argmax())
    out = [tok]
    drafter.observe([tok])
    ctx = prompt_list + [tok]
    st.nodes = 0

    torch.cuda.synchronize()
    t_dec = time.perf_counter()
    with torch.no_grad():
        while len(out) < max_new and not _stop(tok, eos):
            td = time.perf_counter()
            tree = drafter.propose_tree(ctx, min(k, max_new - len(out)))
            st.draft_s += time.perf_counter() - td
            if tree is None or tree.n_draft == 0:
                prev = tok
                logits = eng.forward(torch.tensor([tok], device=prompt.device), start=pos,
                                     last_only=True)
                if hasattr(drafter, "sync"):
                    drafter.sync([prev], eng.hidden_post_norm[0], pos)
                pos += 1
                tok = int(logits[0, -1].argmax())
                out.append(tok)
                ctx.append(tok)
                drafter.observe([tok])
                st.blocks += 1
                continue
            block = torch.tensor(tree.tokens, device=prompt.device)
            lg = eng.forward_tree(block, tree.parents, start=pos)
            picks = lg.argmax(-1).tolist()
            path, new = eng.accept_tree(tree, picks)
            n = len(path) - 1
            st.blocks += 1
            st.nodes += tree.n_draft
            st.drafted += tree.n_draft
            st.accepted += n
            st.per_block.append(len(new))
            tr = time.perf_counter()
            eng.commit_tree(path)
            torch.cuda.synchronize()
            st.rollback_s += time.perf_counter() - tr
            if n < tree.n_draft:
                st.rollbacks += 1
            if hasattr(drafter, "sync"):
                hid = eng.hidden_post_norm[0, torch.tensor(path, device=prompt.device)]
                _sync(drafter, [int(tree.tokens[i]) for i in path], hid, pos, path)
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


def _sync(drafter, tokens, hidden, first_pos, rows) -> None:
    """Bring a drafter current after a tree block.

    A drafter that reads the hidden states this loop hands it needs no more than the rows of the
    accepted path, which is what `hidden` already is. One does not: the block drafter takes five
    mid-stack residual streams off the engine's tap, and the tap holds one row per NODE, in DFS
    order. Which of those rows the target committed to is `rows`, so it gets them.
    """
    if getattr(drafter, "wants_rows", False):
        drafter.sync(tokens, hidden, first_pos, rows=rows)
    else:
        drafter.sync(tokens, hidden, first_pos)


class ThinkBudget:
    """A cap on how long the model is allowed to reason, enforced by the engine.

    The template opens the reasoning block in the generation prompt itself -- with thinking on it
    ends in `<think>\n` -- so the model is inside the block from the first token it writes, and the
    only thing that ends the block is the model deciding to write `</think>`. On a hard problem it
    can decide that very late, and the tokens in between are paid for at the same rate as the
    answer.

    Budget forcing is the documented way to stop it: when the budget is spent, the block is closed
    FOR the model, by appending a short sentence that says why and then the closing tag, and
    generation continues from there into the answer. The sentence matters -- an abrupt `</think>`
    leaves the model mid-thought and the answers get worse -- and this is the phrasing the vendor's
    own examples use.

    **This changes the output.** It is not a speed trick that preserves what the model would have
    said; it is an instruction to stop thinking and answer, and on a problem that needed the
    thinking the answer will be worse. Every number measured under a budget has to say which budget
    it was measured under.
    """

    PHRASE = ("\n\nConsidering the limited time by the user, I have to give the solution based on "
              "the thinking directly now.\n</think>\n\n")

    def __init__(self, tokenizer, budget: int = 0, phrase: str | None = None):
        self.budget = int(budget or 0)
        self.open_id = _special_id(tokenizer, "<think>")
        self.end_id = _special_id(tokenizer, "</think>")
        self.close_ids = tokenizer(phrase or self.PHRASE,
                                   add_special_tokens=False).input_ids
        self.n = 0
        self.inside = False
        self.done = False

    def start(self, prompt_ids: list[int]) -> "ThinkBudget":
        """Arm for one request. The prompt's own tail says whether the block is already open."""
        self.n, self.inside, self.done = 0, False, False
        tail = list(prompt_ids[-6:])
        if self.open_id in tail:
            self.inside = self.end_id not in tail[tail.index(self.open_id):]
        return self

    def observe(self, ids) -> None:
        for t in ids:
            if self.done:
                return
            if not self.inside:
                if t == self.open_id:
                    self.inside = True
                continue
            if t == self.end_id:
                self.inside, self.done = False, True
            else:
                self.n += 1

    @property
    def hit(self) -> bool:
        return bool(self.budget) and self.inside and not self.done and self.n >= self.budget


def _special_id(tokenizer, text: str) -> int:
    i = tokenizer.convert_tokens_to_ids(text)
    if isinstance(i, int) and i >= 0 and i != getattr(tokenizer, "unk_token_id", None):
        return i
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return int(ids[-1])
