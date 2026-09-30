"""An OpenAI-compatible server over this engine, standard library only.

The engine holds one sequence: one recurrent state per linear-attention layer, one KV buffer, one
drafter cache. So the server is honest about that -- requests are serialised behind a lock, and
`/v1/models` and the usage counts are the only places it pretends to be a fleet. Concurrency
numbers from this server are queueing numbers, and the docs say so.

What it exists for: the bench harness this engine is measured against speaks
`POST /v1/chat/completions` with `stream: true` and `stream_options: {"include_usage": true}`,
reads time-to-first-token from the first content delta and inter-token latency from the gaps
between deltas, and takes the token counts from the usage frame. Any of those missing turns a
measured row into an estimated one, so all three are produced exactly.

Streaming emits one chunk per token rather than per buffer flush, because the gaps between chunks
are the measurement.

    python server/app.py --port 8000 --drafter mtp --depth 3
    QWEN38_NVFP4=~/nvfp4/mlp-clip.safetensors python server/app.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import signal
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import cache  # noqa: E402
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)
from engine.drafters import tree_steps  # noqa: E402
from engine.spec import Relax, ThinkBudget  # noqa: E402
from engine.penalty import PatternStop, PenaltySpec, PenaltyState  # noqa: E402
from engine.sample import Sampler  # noqa: E402
from server.stream import (  # noqa: E402
    OPEN_THINK, Detokenizer, Reasoning, StopStrings, opens_think, split_full,
)
from server.toolcall import (JSON_ANY, ToolCallBuffer, parse_tool_calls,  # noqa: E402
                             schema_types)
from server import compat  # noqa: E402
from engine import grammar as grammar_mod  # noqa: E402
from server import logprobs as lp_mod  # noqa: E402
from server import metrics  # noqa: E402
from server import usage as usage_mod  # noqa: E402
from server import ledger as ledger_mod  # noqa: E402
from server import dashboard_api  # noqa: E402
from server import live as live_mod  # noqa: E402
from server import activity as activity_mod  # noqa: E402
from server import logbuf  # noqa: E402
from server import auth as auth_mod  # noqa: E402
from server import static  # noqa: E402
from server import images as images_mod  # noqa: E402

STATE: dict = {}
# the routers' prices for this weight set (engine/prices.py); empty = the code's constants
PRICES: dict = {}
LOCK = threading.Lock()

# The tree loop launches the next draft BEFORE it streams the block it just
# accepted: accept, commit, drafter sync, the drafter's bookkeeping, the next proposal up to its
# draft launch -- then the tokens go to the client (detokenizer, SSE writes, socket flushes) while
# the draft runs, and only then does the host wait for the draft. Until now the GPU idled through
# the streaming. The same calls with the same arguments in the same order, except that the yields
# move behind the launch, so the output is the same token for token. Measured and left off:
# with the window detokenizer the whole consumer costs 6.9 us
# a token on the box, so this can move at most ~0.03 ms of a ~96 ms round.
LAUNCH_FIRST = _S.get("LAUNCH_FIRST") == "1"


def _launch(steps):
    """Run a `*_steps` proposal to its first stop (its draft is on the device) or to its end.
    Returns (the generator if it stopped, else None; its result if it ended)."""
    try:
        next(steps)
    except StopIteration as done:
        return None, done.value
    return steps, None


def _collect(steps, done):
    """The proposal `_launch` started: drive it to the end, or hand back what it already returned."""
    if steps is None:
        return done
    while True:
        try:
            next(steps)
        except StopIteration as end:
            return end.value

# A chunk's keyword arguments when the request did not ask for log-probabilities.
NO_LP: dict = {}

# What the client is told when the repetition guard ends a stream. The finish reason is
# "stop" because that is the whole OpenAI vocabulary; this marker is the engine's own voice and
# says plainly that the cut was the engine's, so an incomplete answer is never presented as a
# finished one. Token accounting and the caches never see it.
GUARD_MARKER = "\n\n[engine: repetition guard stopped the output here]"


def _guard_headers(pstop) -> tuple:
    """`X-Engine-Stop` for a NON-streamed response only.

    A stream's headers are on the wire before its first token, and the guard fires at the end, so
    a streamed response can never carry this header. The streamed client gets the same fact in
    band instead: `GUARD_MARKER` is the last content delta before the finish chunk.
    """
    if pstop is not None and pstop.hit:
        return (("X-Engine-Stop", f"pattern-stop({pstop.label})"),)
    return ()

# The engine holds ONE sequence, so requests are serialised behind `LOCK` and everything else here
# is about being honest to a caller who arrives while it is held. An unbounded wait is not honesty:
# a client that queued behind four long generations gets an answer some minutes after it stopped
# caring, having held a socket the whole time. So the depth is bounded and the wait has a limit,
# and both refusals say `Retry-After`.
QUEUE = threading.Lock()


#: seconds between two prefill heartbeat comments, at least
HEARTBEAT_EVERY_S = 1.0


class ClientGone(ConnectionResetError):
    """The client closed its connection while its request waited or prefilled.

    A ConnectionResetError, so every path that already treats a reset reader as an abandoned
    request -- the stream's handler, `_complete_logged` -- treats this one the same way."""
INFLIGHT = {"waiting": 0, "running": 0, "served": 0, "refused": 0, "errors": 0,
            "timeouts": 0, "abandoned": 0}


class Deadline:
    """A wall-clock cap on one generation, checked between blocks.

    Between blocks and not inside them: a block is one verify and is not interruptible, so the
    granularity of this is about 130 ms, which is the granularity the engine has. `hit` is what the
    stream reports as `finish_reason`, because a generation that ran out of time and one that ran
    out of tokens are different things to the caller and both used to be "length".
    """

    __slots__ = ("t_end", "hit")

    def __init__(self, seconds: float):
        self.t_end = time.monotonic() + float(seconds) if seconds else 0.0
        self.hit = False

    def expired(self) -> bool:
        if self.t_end and time.monotonic() > self.t_end:
            self.hit = True
        return self.hit


class BlockStats:
    """The two factors of one generation's speed, and where in each block the draft went wrong.

    tok/s is committed tokens a block over block time, and the row used to report only the
    quotient: a change that trades one factor for the other was invisible in it. So every
    forward the decode loop pays -- a verify block, a declined single step, a forced reasoning
    close -- is one block here, and the `[req]` line carries the count and the decode time beside
    the tokens, from which `tools/row3.py` takes tokens/block and ms/block.

    `accept` is the first-miss histogram: per block that had a draft, how many draft
    tokens it could have accepted (a chain's length, a tree's depth) and how many it did, so
    `tools/accept_hist.py --curve` can rebuild P(slot i accepted | the slots before it were),
    censored where a block had no slot i. It rides on `[req]` rather than on the `[drafter]` line,
    which is printed at the start of the NEXT request and so never reaches the row's last one.
    """

    __slots__ = ("blocks", "t_first", "t_last", "accept")

    def __init__(self):
        self.blocks = 0
        self.t_first = self.t_last = None
        self.accept: dict[int, dict[int, int]] = {}

    def first(self) -> None:
        self.t_first = self.t_last = time.perf_counter()

    def block(self, depth: int = 0, accepted: int = 0) -> None:
        self.blocks += 1
        self.t_last = time.perf_counter()
        if depth > 0:
            h = self.accept.setdefault(depth, {})
            h[accepted] = h.get(accepted, 0) + 1

    def fields(self, n_out: int) -> str:
        """` blocks=.. committed=.. decode_ms=.. accept=..`; the prefill's token is not decode's."""
        if self.t_first is None:
            return ""
        ms = (self.t_last - self.t_first) * 1e3
        acc = "|".join(f"{d}:" + ",".join(f"{a}x{n}" for a, n in sorted(h.items()))
                       for d, h in sorted(self.accept.items())) or "-"
        return (f" blocks={self.blocks} committed={max(0, n_out - 1)} decode_ms={ms:.1f}"
                f" accept={acc}")


# ------------------------------------------------------------------ generation
def generate_stream(prompt: torch.Tensor, max_new: int, eos: set[int], think=None,
                    conv_id: str | None = None, deadline: "Deadline | None" = None,
                    pen: "PenaltyState | None" = None, pstop: "PatternStop | None" = None,
                    sampler: "Sampler | None" = None, lpr: "lp_mod.Recorder | None" = None,
                    on_prefill=None, mm=None):
    """Yield token ids as they are decided, speculation included.

    Same loop as `engine.spec.generate_spec`, rewritten as a generator so a token reaches the
    socket at the moment it is accepted rather than at the end of the block. A verified block
    produces several tokens at once and they go out together; that is what the engine does, and
    smoothing it would make the inter-token latency a fiction.

    The prefill is `engine.cache.prefill`, which may resume from a state the store already holds.
    Nothing downstream of it knows or cares: it restores the same bytes a forward would have
    written and returns the same logits.

    `lpr` (`logprobs`) is handed each decided token's row where the token is decided --
    after the penalties and `logit_bias`, before the sampling filter -- in the order the tokens are
    yielded; a request that does not ask passes None and the loop only checks for it.

    `on_prefill(done, total)` is called between prefill chunks: the handler's check for a
    client that left, which raises and ends the prefill there, and its stream heartbeat.

    `mm` (engine/vision.py MMContext): the request's images. The engine reads it while it
    prefills the prompt, and every row after the prompt takes its rotary offset `mm.delta`; the
    caches key on `mm.key_ids`. None for text, and set (to None) on every request, so a request
    never inherits the previous one's images.
    """
    from engine.model import h2d           # not at import: the fake engine's server has no triton
    eng = STATE["engine"]
    eng.mm = mm
    eng.pos_delta = mm.delta if mm is not None else 0
    drafter = STATE["drafter"]
    k = STATE["k"]
    ctx = prompt.tolist()
    if pen is not None:
        pen.seed(ctx)
    # The list the engine's own positions index into, published for `_remember`. It is the SAME
    # object, appended to as tokens are committed, so `ctx[:eng.kv.length]` is by construction the
    # prefix the engine really forwarded -- including when a caller abandons this generator half
    # way through a verified block, where `pos` has already advanced past what has been appended
    # and the slice comes out short. Reconstructing it from the tokens the caller collected would
    # be an argument about that invariant instead of a use of it.
    STATE["last_ctx"] = ctx
    bs = STATE["blocks"] = BlockStats()
    if think is not None:
        think.start(ctx)
    with torch.no_grad():
        if drafter is not None:
            # A drafter that keeps per-request policy state clears it in `reset()`, so this is the
            # last moment the PREVIOUS request's choices can be read. Printed here rather than at
            # the end of the generation because the stream returns from four places inside its
            # loop and none of them is an exit worth wrapping for a log line.
            if STATE.get("verbose") and hasattr(drafter, "report"):
                print(f"[drafter] {drafter.report()}", flush=True)
            drafter.reset()
            if hasattr(drafter, "set_sampling"):
                # a sampler-carrying drafter draws its proposals from its own
                # distribution under the request's profile and carries q for the verify.
                # `--sampled-tree det`: the tree is the greedy request's, built without
                # the sample, and walked by drawing the target's token at each node.
                det = (STATE.get("sampled_tree") == "det" and STATE.get("tree")
                       and sampler is not None and sampler.on)
                drafter.set_sampling(None if det else sampler)
            if hasattr(drafter, "prime"):
                drafter.prime(ctx)
        t_pre = time.perf_counter()
        # the handler's chunk hook carries the dict the prefill describes itself in
        # (start, chunk, t0, kind), so the live view reads this request's own prefill
        where = getattr(on_prefill, "info", None)
        if where is None:
            where = {}
        logits, reused, forwarded = cache.prefill(
            eng, drafter, ctx, prompt.device, store=STATE.get("state_store"),
            chunk=STATE.get("prefix_chunk", 0), conv_id=conv_id,
            checkpoint=bool(STATE.get("prefix_cache")), resident=STATE.get("resident"),
            on_chunk=on_prefill, info=where,
            key_ids=mm.key_ids if mm is not None else None)
        if mm is not None:
            mm.release()
        STATE["last_prefill"] = {"reused": reused, "forwarded": forwarded,
                                 "ms": (time.perf_counter() - t_pre) * 1e3,
                                 "kind": where.get("kind") if reused else None}
        pos = prompt.numel()
        if pen is not None:
            pen.mask = bool(think is not None and think.inside)
            pen.apply_single(logits[0, -1])
        tok = sampler(logits[0, -1], index=len(ctx)) if sampler is not None and sampler.on \
            else int(logits[0, -1].argmax())
        if lpr is not None:
            lpr.rows(logits[0, -1:], [tok])
        n_out = 1
        ctx.append(tok)
        bs.first()
        if pen is not None:
            pen.commit([tok])
        if drafter is not None:
            drafter.observe([tok])
        yield tok
        if tok in eos:
            return
        # Sampled requests keep their drafter (rejection sampling under speculation --
        # see engine/sample.py; the output follows the target's sampled distribution either way).
        # v1: a sampled request takes the q-aware CHAIN. `--sampled-tree`: the
        # tree instead -- `det`, the greedy request's tree walked by drawing at each node, or
        # `mixed`, the sampled chain as the tree's spine (its q rows accepted by rejection
        # sampling) with the lattice's siblings beside it (engine/tree.py::spine_tree).
        tree_mode = (STATE.get("tree") and drafter is not None
                     and hasattr(drafter, "propose_tree")
                     and (not (sampler is not None and sampler.on)
                          or STATE.get("sampled_tree", False)))
        ahead = None                  # the next proposal, launched before the last stream
        while n_out < max_new:
            if deadline is not None and deadline.expired():
                return
            if tree_mode:
                # The tree path. It is the same loop with three lines changed: the drafter hands
                # back a shape rather than a list, the accept is a walk down that shape instead of
                # a prefix comparison, and the commit takes the path rather than a length. The
                # reason it is worth the branch is in docs/speculative-decoding.md under "tree verify":
                # the step costs the same for two rows as for sixteen.
                if ahead is not None:
                    tree, ahead = _collect(*ahead), None
                else:
                    tree = drafter.propose_tree(ctx, min(k, max_new - n_out))
                # the KV write counts NODES (anchor included) while the budget above
                # counts output tokens, and a drafter may return more nodes than it was handed.
                # A DFS pre-order prefix is a valid tree, so cutting at the row bound only
                # drops candidates.
                if tree is not None:
                    tree = tree.truncate(eng.max_len - pos)
                if tree is None or tree.n_draft == 0:
                    draft = []
                else:
                    block = h2d(tree.tokens, torch.long, prompt.device)
                    tvt = time.perf_counter()
                    lg = eng.forward_tree(block, tree.parents, start=pos)
                    if pen is not None:
                        pen.mask = bool(think is not None and think.inside)
                        pen.apply_tree(lg, tree)
                    on_verify = getattr(drafter, "on_verify", None)
                    if on_verify is not None:
                        on_verify(tree.n_draft + 1, (time.perf_counter() - tvt) * 1e3)
                    if sampler is not None and sampler.on:
                        # Rejection accept down the tree: the target's own token is
                        # sampled at every node and the walk follows the child carrying it; where
                        # no child carries it, the draw is the token and the walk stops.
                        path, new = sampler.tree_walk(sampler.probs_rows(lg), tree.tokens,
                                                      tree.parents, start=len(ctx), q=tree.q)
                    else:
                        # a graphed verify took the argmax itself; a penalty changed the
                        # logits after it, so then the loop takes it
                        picks_d = eng.picks if eng.picks is not None and pen is None else None
                        picks_t = (picks_d if picks_d is not None else lg.argmax(-1)).tolist()
                        path, new = eng.accept_tree(tree, picks_t)
                    if lpr is not None:
                        lpr.rows(lg[h2d(path, torch.long, prompt.device)], new)
                    eng.commit_tree(path)
                    if hasattr(drafter, "sync"):
                        sel = h2d(path, torch.long, prompt.device)
                        toks = [int(tree.tokens[i]) for i in path]
                        if getattr(drafter, "wants_rows", False):
                            drafter.sync(toks, eng.hidden_post_norm[0, sel], pos, rows=path)
                        else:
                            drafter.sync(toks, eng.hidden_post_norm[0, sel], pos)
                    pos += len(path)
                    bs.block(max(tree.depths()), len(path) - 1)
                    drafter.observe(new)
                    if pen is not None:
                        pen.commit(new)
                    stop_now = pstop is not None and pstop.observe(new)
                    if (LAUNCH_FIRST and not stop_now and n_out + len(new) < max_new
                            and not any(t in eos for t in new)):
                        # the stream goes on after this block, so the next proposal is
                        # started first and the block is streamed while its draft runs. What the
                        # old order did between the yields and the proposal -- the think budget's
                        # look at the block -- is done before it, as it only reads the tokens.
                        n0 = len(ctx)
                        ctx.extend(new)
                        if think is not None:
                            think.observe(new)
                        if think is None or not think.hit:
                            ahead = _launch(tree_steps(drafter, ctx,
                                                       min(k, max_new - n_out - len(new))))
                        sent = 0
                        try:
                            for t in new:
                                n_out += 1
                                sent += 1
                                yield t
                        finally:
                            # a reader that stops here leaves `ctx` as the old order did: through
                            # the token it was handed, since `_remember` publishes this list
                            del ctx[n0 + sent:]
                        tok = ctx[-1]
                        if think is None or not think.hit:
                            continue
                    else:
                        for t in new:
                            ctx.append(t)
                            n_out += 1
                            yield t
                            if t in eos or n_out >= max_new:
                                return
                        if stop_now:
                            # The repeating block was yielded first: the client sees what was
                            # written, then the stream ends with finish_reason stop and a [req]
                            # annotation.
                            return
                        tok = ctx[-1]
                        if think is not None:
                            think.observe(new)
                    if think is not None:
                        if think.hit:
                            for t in _force_close(eng, drafter, think, ctx, pos, prompt.device,
                                                  pen=pen, pstop=pstop, sampler=sampler, lpr=lpr):
                                n_out += 1
                                yield t
                                if n_out >= max_new:
                                    return
                            if pstop is not None and pstop.hit:
                                return                 # the guard fired on the phrase
                            pos += 1 + len(think.close_ids)
                            tok = ctx[-1]
                    continue
            else:
                draft = (drafter.propose(ctx, min(k, max_new - n_out))
                         if drafter is not None else [])
                # the block's KV write is `1 + len(draft)` rows (the anchor's own row is
                # one of them) while every clamp above counts output tokens, and a drafter may
                # return more than it was handed. Cap on rows here, where the forward is paid.
                draft = draft[:max(0, eng.max_len - pos - 1)]
            if not draft:
                logits = eng.forward(h2d([tok], torch.long, prompt.device), start=pos,
                                     last_only=True)
                # Bring a position-indexed drafter current, as engine/spec.py's loop does.
                # The block drafter's cache must cover every committed position; skip this one and
                # it is one behind for good -- it declines every later step, and the request
                # finishes one token a forward.
                if drafter is not None and hasattr(drafter, "sync"):
                    drafter.sync([tok], eng.hidden_post_norm[0], pos)
                pos += 1
                bs.block()
                if pen is not None:
                    pen.mask = bool(think is not None and think.inside)
                    pen.apply_single(logits[0, -1])
                tok = sampler(logits[0, -1], index=len(ctx)) if sampler is not None and sampler.on \
                    else int(logits[0, -1].argmax())
                if lpr is not None:
                    lpr.rows(logits[0, -1:], [tok])
                ctx.append(tok)
                n_out += 1
                if pen is not None:
                    pen.commit([tok])
                if pstop is not None and pstop.observe([tok]):
                    yield tok
                    return
                if drafter is not None:
                    drafter.observe([tok])
                yield tok
                if tok in eos:
                    return
                if think is not None:
                    think.observe([tok])
                    # The same close the verified paths make. A drafter-less server --
                    # and any declined step -- used to observe the token and never look, so the
                    # budget and the stall signal could not close a block on this path at all.
                    if think.hit:
                        for t in _force_close(eng, drafter, think, ctx, pos, prompt.device,
                                              pen=pen, pstop=pstop, sampler=sampler, lpr=lpr):
                            n_out += 1
                            yield t
                            if n_out >= max_new:
                                return
                        if pstop is not None and pstop.hit:
                            return
                        pos += 1 + len(think.close_ids)
                        tok = ctx[-1]
                continue
            block = h2d([tok] + draft, torch.long, prompt.device)
            tv = time.perf_counter()
            lg = eng.forward_block(block, start=pos)
            if pen is not None:
                pen.mask = bool(think is not None and think.inside)
                pen.apply_chain(lg, draft)
            # the same hook the bench loop has: a drafter that prices block widths learns what a
            # width costs from the loop that pays for it (engine/lenrouter.py)
            on_verify = getattr(drafter, "on_verify", None)
            if on_verify is not None:
                on_verify(len(draft) + 1, (time.perf_counter() - tv) * 1e3)
            if sampler is not None and sampler.on:
                # Rejection accept of the chain. A drafter that sampled its proposal carries a q
                # row per token (min(1, p(d)/q(d)), residual (p - q)+); a deterministic
                # arm carries none and gets the shortcut -- draw the target's own token,
                # it matches the draft or it is the token.
                qrows = getattr(drafter, "last_q", None) if drafter is not None else None
                n, x = sampler.chain_accept(sampler.probs_rows(lg), draft, qrows, start=len(ctx))
                new = draft[:n] + [x]
            else:
                picks = lg.argmax(-1).tolist()
                relax = STATE["relax"]
                n = 0
                for i, d in enumerate(draft):
                    if picks[i] == d:
                        n += 1
                        continue
                    if relax.on and relax.accepts(lg[i], d, picks[i]):
                        n += 1
                        continue
                    break
                new = draft[:n] + [picks[n]]
            if lpr is not None:
                lpr.rows(lg, new)
            if pen is not None:
                pen.commit(new)
            stop_now = pstop is not None and pstop.observe(new)
            if n < len(draft):
                eng.rollback_to(n + 1)
            if drafter is not None and hasattr(drafter, "sync"):
                drafter.sync([int(x) for x in block[:n + 1]], eng.hidden_post_norm[0, :n + 1], pos)
            pos += n + 1
            bs.block(len(draft), n)
            if drafter is not None:
                drafter.observe(new)
            for t in new:
                ctx.append(t)
                n_out += 1
                yield t
                if t in eos or n_out >= max_new:
                    return
            if stop_now:
                return
            tok = ctx[-1]
            if think is not None:
                think.observe(new)
                if think.hit:
                    for t in _force_close(eng, drafter, think, ctx, pos, prompt.device,
                                          pen=pen, pstop=pstop, sampler=sampler, lpr=lpr):
                        n_out += 1
                        yield t
                        if n_out >= max_new:
                            return
                    if pstop is not None and pstop.hit:
                        return                         # the guard fired on the phrase
                    pos += 1 + len(think.close_ids)
                    tok = ctx[-1]


def _force_close(eng, drafter, think, ctx, pos, device, pen=None, pstop=None, sampler=None,
                 lpr=None):
    """Close the reasoning block for the model and take the first token of its answer.

    The forced tokens are run through the engine exactly as generated ones are -- one forward at
    `pos` -- so the KV, the recurrent state and the drafter's context all carry them, and the
    answer that follows is conditioned on a block that really does end where it appears to.
    """
    # `ctx[-1]` is the last committed token and its own forward has not happened yet -- that is
    # the loop's invariant, `len(ctx) == pos + 1`, and it is why the next verify block starts with
    # it. The forced pass has to carry it, or the closing phrase would be written over its position.
    print(f"[think] closed the reasoning block: reason={think.reason or 'budget'} "
          f"at {think.n} tokens", flush=True)
    think.t_forced = time.perf_counter()               # once per forced close
    from engine.model import h2d
    closing = list(think.close_ids)
    forced = [int(ctx[-1])] + closing
    lg = eng.forward(h2d(forced, torch.long, device), start=pos, last_only=True)
    if STATE.get("blocks") is not None:
        STATE["blocks"].block()
    if drafter is not None and hasattr(drafter, "sync"):
        drafter.sync(forced, eng.hidden_post_norm[0], pos)
    if drafter is not None:
        drafter.observe(closing)
    ctx.extend(closing)
    if pen is not None:
        # The forced tokens join the history -- they are real context -- but no penalty was
        # consulted in choosing them, which is the point of forcing. The token AFTER them is a
        # genuine decision, so it is penalized against a history that includes them.
        pen.commit(closing)
        pen.apply_single(lg[0, -1])
    think.observe(closing)
    if lpr is not None:
        lpr.forced(closing)
    if pstop is not None and pstop.observe(closing):
        # The guard fired on the phrase itself, and a guard hit ENDS the generation: no answer
        # token is drawn, and the caller returns on `pstop.hit`. It used to carry on
        # decoding instead, and its next step forwarded `ctx[-1]` -- the phrase's last token,
        # already written by the forward above -- a second time, one row further on. Ending here
        # leaves the KV holding exactly `ctx`, every token once.
        for t in closing:
            yield t
        return
    for t in closing:
        yield t
    nxt = (sampler(lg[0, -1], index=len(ctx)) if sampler is not None and sampler.on
           else int(lg[0, -1].argmax()))
    if lpr is not None:
        lpr.rows(lg[0, -1:], [nxt])
    ctx.append(nxt)
    if pen is not None:
        pen.commit([nxt])
    if drafter is not None:
        drafter.observe([nxt])
    yield nxt


def _remember(prompt_ids: list[int], out_ids: list[int], conv_id: str | None,
              mm=None) -> None:
    """After a turn: keep the state it ended in, and add its tokens to the suffix store.

    The loop's invariant at the end of a generation is `kv.length == len(ctx) - 1` -- the last
    token has been decided and not forwarded -- so what is snapshotted is the prefix that really
    was forwarded, and `generate_stream` publishes that very list rather than one rebuilt here. The next turn's prompt begins with all of it plus the chat template's own glue,
    so it resumes here and pays for the glue and the new message rather than for the conversation.
    """
    eng, drafter = STATE["engine"], STATE["drafter"]
    store = STATE.get("state_store")
    committed = STATE.get("last_ctx") or []
    if mm is not None and len(committed) >= mm.n:
        # the state is keyed by the images' content, not by their placeholders
        committed = list(mm.key_ids) + list(committed[mm.n:])
    if store is not None and STATE.get("session_cache") and eng.kv.length:
        # `StateStore.put` declines when the snapshot is longer than the tokens it is given, which
        # is exactly the abandoned-mid-block case above -- and, before phase 9, was also the
        # ordinary case, because the loop advances `pos` by the whole accepted path and appends to
        # `ctx` one token at a time as it yields them. Every generation that ends inside a block
        # ends with `kv.length` ahead of `len(ctx)`, and every generation ends inside a block, so
        # the store was declining EVERY put and the session cache had never stored anything.
        # `puts: 0, hits: 0, misses: 40` after forty requests is what that looks like from
        # outside, and nothing else in the server complains.
        before = store.stats["puts"]
        snap = cache.capture(eng, drafter, max_bytes=store.max_entry)
        if snap is not None:
            store.put(committed, snap, conv_id, kind="session")
        if STATE.get("verbose") and store.stats["puts"] == before:
            print(f"[cache] put declined: kv.length={eng.kv.length} ctx={len(committed)}",
                  flush=True)
        # `declined_short` in /v1/cache/stats is the same event counted, for when nobody is
        # reading the log.
    suffix = STATE.get("suffix_store")
    if suffix is not None:
        if mm is not None:
            # the lookup corpus is text: an image's placeholder rows are not written to it
            pad = mm.tower.cfg.image_token_id
            prompt_ids = [t for t in prompt_ids if t != pad]
        full = list(prompt_ids) + list(out_ids)
        suffix.append(full if STATE.get("suffix_scope") == "all" else out_ids)


def _session_header(headers) -> str | None:
    """Opencode's session id (`x-session-id`, sent with every request of a session): a label for the
    live view's `continues` link only -- the caches never see it."""
    try:
        v = headers.get("x-session-id") or headers.get("X-Session-Id")
    except Exception:                                              # noqa: BLE001
        v = None
    return v[:128] if isinstance(v, str) and v else None


def _finishing(rec, step: str) -> None:
    """One of the three steps after the token loop (flush, saving_state, final_chunk).
    The first also records the state the loop ended in. Three assignments a request."""
    if rec.step is None:
        rec.end_state = activity_mod.state_of(rec, ignore_step=True)
        rec.t_finishing = time.perf_counter()
    rec.step = step


def _stop_detail(finish: str, ids: list, eos, cut: bool, pstop) -> str | None:
    """What refines a `stop`: the end token, a stop string or the repetition guard."""
    if finish != "stop":
        return None
    if pstop is not None and pstop.hit:
        return "pattern_guard"
    if cut:
        return "stop_string"
    return "eos" if ids and ids[-1] in eos else None


def conversation_id(body: dict, headers) -> str | None:
    """A label for the conversation, if the client offers one. It is never load-bearing.

    The state store matches on the token prefix and checks it element for element, so a wrong or
    missing conversation id costs a cache hit and can never produce a wrong one. What the id buys
    is a readable `/v1/cache/stats` and, when two conversations share a prefix, a way to tell which
    entry belongs to which.
    """
    for key in ("conversation_id", "session_id", "user"):
        v = body.get(key)
        if isinstance(v, str) and v:
            return v[:128]
    meta = body.get("metadata")
    if isinstance(meta, dict):
        v = meta.get("conversation_id") or meta.get("session_id")
        if isinstance(v, str) and v:
            return v[:128]
    try:
        v = headers.get("X-Conversation-Id")
    except Exception:
        v = None
    return v[:128] if isinstance(v, str) and v else None


def _normalize_tool_arguments(messages: list) -> list:
    """Assistant tool_calls arrive with `arguments` as a JSON STRING (OpenAI shape); the chat
    template iterates it as a mapping, so it must be a dict before rendering. Unparseable strings
    are wrapped rather than raising a 500 mid-prompt."""
    out = []
    for m in messages:
        calls = m.get("tool_calls") if isinstance(m, dict) else None
        if calls:
            m = dict(m)
            fixed = []
            for c in calls:
                c = dict(c)
                fn = dict(c.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except ValueError:
                        fn["arguments"] = {"value": args}
                c["function"] = fn
                fixed.append(c)
            m["tool_calls"] = fixed
        out.append(m)
    return out


def _with_directive(messages: list, note: str) -> list:
    """`messages` with `note` as the last paragraph of the system message (one is added if the
    conversation has none; the template renders it after the tools either way)."""
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        first = dict(messages[0])
        c = first.get("content")
        if isinstance(c, list):
            first["content"] = list(c) + [{"type": "text", "text": "\n\n" + note}]
        else:
            first["content"] = (c.rstrip() + "\n\n" + note) if c else note
        return [first] + list(messages[1:])
    return [{"role": "system", "content": note}] + list(messages)


def build_prompt(body: dict) -> tuple[torch.Tensor, str, bool]:
    """The prompt ids, what kind of request it was, and whether it ends inside `<think>`.

    The third value is the whole of bug 1. `add_generation_prompt=True` with thinking enabled ends
    the rendered prompt with `<|im_start|>assistant\n<think>\n`, so the OPENING tag is part of the
    prompt and the model never generates it: the only tag in the output is `</think>`, and a client
    that folds a reasoning block on a matched pair sees an unmatched closer and folds nothing.
    Whoever renders the prompt is the only code that can know this, so it is answered here and the
    server puts the tag back.
    """
    tok = STATE["tok"]
    if "messages" in body:
        kwargs = dict(body.get("chat_template_kwargs") or {})
        kwargs.setdefault("enable_thinking", True)
        # Tool calling, request side. The Qwen template has native tool support and
        # renders the official <tool_call><function=...> protocol when `tools` is passed; without
        # this the model never sees the client's tool schemas (chat 53d7ca38: it announced a web
        # search and stopped; chat efa916ed: it invented write_file from its priors).
        tools = body.get("tools")
        mode, name = compat.tool_choice(body)
        if tools and mode == "none":
            tools = None
        if tools:
            kwargs["tools"] = tools
        if body.get("tool_choice") is not None:
            kwargs["tool_choice"] = body["tool_choice"]
        messages = _normalize_tool_arguments(body["messages"])
        # the template has no notion of `tool_choice`, so `required` and a named function
        # are asked for in words -- one sentence at the end of the system turn, after the tool
        # definitions and the client's own system prompt. `auto` and `none` add nothing.
        note = compat.directive(mode, name) if tools else None
        if note:
            messages = _with_directive(messages, note)
        # The template resolves `reasoning_effort` to xhigh unless told otherwise, and xhigh is a
        # paragraph of instructions telling the model to check its assumptions and consider
        # alternatives. A server default is the cheapest way to make it think less, because it
        # changes what the model is asked for rather than cutting it off part-way.
        effort = body.get("reasoning_effort") or STATE.get("reasoning_effort")
        if effort:
            kwargs.setdefault("reasoning_effort", effort)
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True, **kwargs)
        ids = _as_ids(enc)
        # Decoded from the ids rather than rendered a second time, so what is inspected is exactly
        # the token sequence the engine is about to be given.
        return ids, "chat", opens_think(tok.decode(ids.tolist(), skip_special_tokens=False))
    text = body.get("prompt")
    if isinstance(text, list):
        text = text[0]
    return _as_ids(tok(text or "", return_tensors="pt")), "text", False


def _mm_prompt(prompt: torch.Tensor, images: list, rec=None):
    """The prompt with each image's one placeholder expanded to its rows, and the
    request's `MMContext`. `rec.encoding` is the live view's "encoding image i of n"."""
    from engine.vision import MMContext, expand
    tower = STATE["vision"]
    ids, spans = expand(prompt.tolist(), images, tower.cfg.image_token_id,
                        tower.cfg.spatial_merge_size)

    def on_encode(i, n):
        if rec is not None:
            rec.encoding = (i, n) if i is not None else None

    mm = MMContext(ids, spans, images, tower, STATE["engine"].cfg,
                   cache=STATE.get("image_cache"), on_encode=on_encode)
    if STATE.get("verbose"):
        grids = ",".join("x".join(str(g) for g in im.grid) for im in images)
        print(f"[vision] {len(images)} image(s) grids={grids} rows={mm.image_tokens} "
              f"delta={mm.delta} sources={','.join(im.source for im in images)}", flush=True)
    return torch.tensor(ids, dtype=torch.long, device=prompt.device), mm


def _as_ids(enc) -> torch.Tensor:
    """One 1-D int tensor of token ids, whatever shape the tokeniser handed back.

    The chat-template call returns a mapping, a plain tensor or a fast-tokeniser Encoding
    depending on the version and the arguments, and getting this wrong is a 500 on every request.
    """
    obj = enc
    for key in ("input_ids",):
        if hasattr(obj, key):
            obj = getattr(obj, key)
            break
        if isinstance(obj, dict) or hasattr(obj, "keys"):
            try:
                obj = obj[key]
                break
            except Exception:
                pass
    if not torch.is_tensor(obj):
        if hasattr(obj, "ids"):
            obj = torch.tensor(obj.ids, dtype=torch.long)
        else:
            obj = torch.tensor(obj, dtype=torch.long)
    while obj.dim() > 1:
        obj = obj[0]
    return obj.to(STATE["device"])


def eos_ids(body: dict) -> set[int]:
    tok = STATE["tok"]
    out = set()
    for t in (tok.eos_token_id, STATE["cfg_eos"]):
        if isinstance(t, int):
            out.add(t)
        elif isinstance(t, (list, tuple)):
            out.update(int(x) for x in t)
    return out


# ------------------------------------------------------------------ HTTP
def _chunk(cid: str, model: str, created: int, delta: dict, finish=None, usage=None,
           error=None, extra: dict | None = None, logprobs: dict | None = None) -> str:
    body = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [] if delta is None and finish is None else
            [{"index": 0, "delta": delta or {}, "finish_reason": finish, "logprobs": logprobs}]}
    if usage is not None:
        body["usage"] = usage
    if extra:
        # `usage`, `timings` and `metrics` of the whole request, on the ONE chunk of the
        # stream that carries them -- see server/usage.py's `placement`.
        body.update(extra)
    if error is not None:
        # Not in the OpenAI schema, and deliberately alongside a real `finish_reason` rather than
        # instead of one: a client that only knows the schema still sees the stream end, and a
        # client that looks can find out why.
        body["error"] = error
    return "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"


def _log_request(cid: str, n_prompt: int, n_out: int, finish: str, t0: float, *,
                 stream: bool, exc: BaseException | None = None,
                 pen: PenaltySpec | None = None, pattern: str | None = None,
                 rec: "usage_mod.RequestRecord | None" = None, temp: float = 0.0,
                 tools: str | None = None) -> None:
    """One line per generation, always, whatever happened to it.

    The server used to log the HTTP status and nothing else, so an answer that stopped at the
    default token limit and an answer that stopped because the engine raised looked the same from
    the outside -- a 200 and a short reply. Everything needed to tell those apart is here.

    `rec` is the request's record: it takes the loop's block counts from here, where they
    are popped, and its end time is this line's, so `timings.total_ms` is the `ms` printed.

    `tools` is `parsed:N` -- the calls read out of the answer -- on a request that carried
    tools or produced a call; a tool-free line is what it was.
    """
    now = time.perf_counter()
    ms = (now - t0) * 1e3
    rate = (n_out - 1) / (ms / 1e3) if n_out > 1 and ms > 0 else 0.0
    with QUEUE:
        if finish == "error":
            INFLIGHT["errors"] += 1
        elif finish == "timeout":
            INFLIGHT["timeouts"] += 1
        elif finish == "abandoned":
            INFLIGHT["abandoned"] += 1
    # an exception's message, capped and withheld if it quotes the request
    tail = f"  !! {type(exc).__name__}: {logbuf.safe_message(exc)}" if exc is not None else ""
    pen_s = (f" pen=({pen.rep:g},{pen.presence:g},{pen.freq:g},n={pen.no_repeat})"
             if pen is not None and pen.penalizes else "")
    if pen is not None and pen.bias:
        pen_s += f" bias={len(pen.bias)}"
    tools_s = f" tools={tools}" if tools else ""
    pat_s = f" pattern-stop({pattern})" if pattern else ""
    # which requests sample. Only a sampled request says so, as `pen=` only says it when
    # on, so a greedy line is what it was.
    temp_s = f" temp={temp:g}" if temp > 0 else ""
    bs = STATE.pop("blocks", None)
    blk_s = bs.fields(n_out) if bs is not None else ""
    if rec is not None:
        rec.absorb_blocks(bs)
        rec.t_end = now
    print(f"[req] {cid} {'stream' if stream else 'json'} prompt={n_prompt} "
          f"completion={n_out} finish={finish} {ms:.0f} ms {rate:.2f} tok/s{temp_s}{pen_s}{pat_s}"
          f"{tools_s}{blk_s}{tail}", flush=True)


def _tools_note(body: dict, parsed: int, dropped: int = 0) -> str | None:
    """The `[req]` line's `tools=` field: None for a request without tools that called nothing."""
    if not body.get("tools") and not parsed:
        return None
    return f"parsed:{parsed}" + (f",dropped:{dropped}" if dropped else "")


def _account(rec: "usage_mod.RequestRecord") -> None:
    """Every request that reached a completion route, whatever became of it: one ledger row.

    Called once, from `Handler._complete`'s `finally` -- served, refused at the queue, rejected
    with a 400, failed or abandoned. The ledger's `submit` never blocks.
    """
    STATE["last_request_ts"] = rec.ts
    try:
        metrics.on_record(rec)
    except Exception:                                              # noqa: BLE001
        pass                             # a metric must never fail a request
    led = STATE.get("ledger")
    if led is not None:
        led.submit(rec.row(STATE.get("version", ""), STATE.get("code_sha", "")))


def _grammar_vocab() -> "grammar_mod.Vocab":
    """Every token as bytes, for the constraint masks: built on the first constrained
    request (half a second over 248k tokens) and kept."""
    v = STATE.get("grammar_vocab")
    if v is None:
        v = STATE["grammar_vocab"] = grammar_mod.Vocab.from_tokenizer(
            STATE["tok"], STATE["engine"].cfg.vocab_size)
    return v


def _auth() -> "auth_mod.Auth":
    """The access policy: from the environment at startup; tests set their own."""
    a = STATE.get("auth")
    if a is None:
        a = STATE["auth"] = auth_mod.Auth.from_env()
    return a


def _live() -> dict:
    """The dashboard's status pill: never cached, never takes the engine lock."""
    last = STATE.get("last_request_ts")
    return {"status": "draining" if STATE.get("draining") else ("busy" if LOCK.locked() else "ok"),
            "running": INFLIGHT["running"], "waiting": INFLIGHT["waiting"],
            "uptime_s": round(time.time() - STATE.get("started", time.time()), 1),
            "last_request_at": dashboard_api.iso_utc(last * 1000) if last else None}


def _system() -> dict:
    """`/v1/dashboard/system`: what is running, how it is configured, what it holds."""
    mem = {"gpu_allocated_bytes": None, "gpu_reserved_bytes": None,
           "gpu_max_allocated_bytes": None}
    try:
        if torch.cuda.is_available():
            mem = {"gpu_allocated_bytes": int(torch.cuda.memory_allocated()),
                   "gpu_reserved_bytes": int(torch.cuda.memory_reserved()),
                   "gpu_max_allocated_bytes": int(torch.cuda.max_memory_allocated())}
    except Exception:                                              # noqa: BLE001
        pass
    mem.update(dashboard_api.meminfo())
    led = STATE.get("ledger")
    li = led.info() if led is not None else None
    sampler = STATE.setdefault("gpu_sampler", dashboard_api.GpuSampler())
    try:
        caches = cache_stats()
    except Exception:                                              # noqa: BLE001
        caches = None
    return {
        "engine": {"version": STATE.get("version", ""), "git_sha": STATE.get("git_sha", ""),
                   "code_sha256": STATE.get("code_sha256", ""),
                   "started_at": dashboard_api.iso_utc_s(STATE.get("started", time.time())),
                   "uptime_s": round(time.time() - STATE.get("started", time.time()), 1),
                   "pid": os.getpid(), "model": STATE.get("model"), "max_len": STATE.get("max_len"),
                   "drafter": type(STATE.get("drafter")).__name__
                   if STATE.get("drafter") is not None else None,
                   "tree": bool(STATE.get("tree")),
                   "reasoning_format": STATE.get("reasoning_format"),
                   "reasoning_effort": STATE.get("reasoning_effort"),
                   "status": _live()["status"]},
        "flags": {"args": dashboard_api.redact_args(STATE.get("args") or {}),
                  "env": dashboard_api.redact_env(dict(os.environ))},
        "memory": mem,
        "gpu": sampler.sample(),
        "caches": caches,
        "queue": {"running": INFLIGHT["running"], "waiting": INFLIGHT["waiting"],
                  "max_queue": STATE.get("max_queue"),
                  "queue_timeout_s": STATE.get("queue_timeout"),
                  "request_timeout_s": STATE.get("request_timeout")},
        "inflight": {k: INFLIGHT[k] for k in ("served", "refused", "errors", "timeouts",
                                             "abandoned")},
        "ledger": {"enabled": li is not None,
                   "rows": (li or {}).get("rows") or 0, "bytes": (li or {}).get("bytes") or 0,
                   "oldest": dashboard_api.iso_utc((li or {}).get("oldest_ms")),
                   "queue": (li or {}).get("queue") or 0, "dropped": (li or {}).get("dropped") or 0},
        "disk": {"state_dir_free_bytes": dashboard_api.disk_free()},
    }


def dashboard() -> "dashboard_api.DashboardAPI":
    api = STATE.get("dashboard_api")
    if api is None or api.ledger is not STATE.get("ledger"):
        api = STATE["dashboard_api"] = dashboard_api.DashboardAPI(
            STATE.get("ledger"), live=_live, system=_system)
    return api


_MEM_CACHE: dict = {"t": 0.0, "v": None}


def _mem_gib() -> dict:
    """MemAvailable and this process's RSS in GiB, read at most every 5 s; nulls where
    /proc does not exist."""
    now = time.monotonic()
    if _MEM_CACHE["v"] is not None and now - _MEM_CACHE["t"] < 5.0:
        return _MEM_CACHE["v"]
    out = {"mem_available_gib": None, "rss_gib": None}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    out["mem_available_gib"] = round(int(line.split()[1]) / (1 << 20), 1)
                    break
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        out["rss_gib"] = round(pages * os.sysconf("SC_PAGE_SIZE") / (1 << 30), 1)
    except (OSError, ValueError, IndexError):
        pass
    _MEM_CACHE.update(t=now, v=out)
    return out


def _live_engine() -> dict:
    """The engine block of the live stream: plain attribute reads off STATE, no lock."""
    eng = STATE.get("engine")
    store = STATE.get("state_store")
    kv = getattr(eng, "kv", None)
    try:
        st = ({"entries": len(store._d), "bytes_gib": round(store.bytes / (1 << 30), 2)}
              if store is not None else None)
    except (AttributeError, RuntimeError, TypeError):
        st = None
    return {"model": STATE.get("model", ""), "version": STATE.get("version", ""),
            "draining": bool(STATE.get("draining")),
            "kv": {"length": int(getattr(kv, "length", 0) or 0),
                   "max_len": int(STATE.get("max_len") or 0)},
            "memory": _mem_gib(), "store": st}


def live_registry() -> "live_mod.LiveRegistry":
    """The in-flight view behind `/v1/dashboard/live`. Reads the running
    `BlockStats`, `last_prefill` and the engine block off STATE; never the engine lock."""
    reg = STATE.get("live")
    if reg is None:
        hz = int(STATE.get("live_hz", live_mod.DEFAULT_HZ))
        reg = STATE["live"] = live_mod.LiveRegistry(
            blocks=lambda: STATE.get("blocks"), last_prefill=lambda: STATE.get("last_prefill"),
            engine=_live_engine, hz=hz,
            activity=bool(STATE.get("live_activity", True)) and hz > 0,
            queue_timeout=lambda: float(STATE.get("queue_timeout", 120.0)))
    return reg


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "TandemLLM"
    _status: int | None = None

    def send_response(self, code, message=None):
        # The status of this request as it went out -- what the ledger row records.
        self._status = code
        super().send_response(code, message)

    def log_message(self, fmt, *args):
        if STATE.get("verbose"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -------------------------------------------------------------- helpers
    def _send_bytes(self, code: int, raw: bytes, extra_headers: tuple = ()) -> None:
        """Write one response, tolerating a client that hung up first.

        A health checker that disconnects mid-write used to raise BrokenPipeError out of here
        and into socketserver's handle_error, which prints a full traceback that reads as a
        server crash. The request is gone either way; swallow it and mark the connection closed.
        """
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for k, v in extra_headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _json(self, code: int, payload: dict, extra_headers: tuple = ()) -> None:
        self._send_bytes(code, json.dumps(payload, ensure_ascii=False).encode(), extra_headers)

    def _busy(self, code: int, message: str, retry: int = 5) -> None:
        """A refusal the caller can act on: a status, a reason, and when to come back."""
        self._send_bytes(code, json.dumps({"error": {"message": message, "type": "server_busy",
                                                     "code": code}}).encode(),
                         extra_headers=(("Retry-After", str(retry)),))

    def _read(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # -------------------------------------------------------------- access
    def _peer(self) -> str:
        return (self.client_address or ("",))[0]

    def _allowed(self, route: str) -> bool:
        """The route's policy (server/auth.py); on a refusal the answer is already written."""
        verdict = _auth().decide(route, self._peer(), self.headers)
        if verdict == "ok":
            return True
        path = self.path.split("?")[0]
        if verdict == "absent":
            self._json(404, {"error": {"type": "not_found", "message": f"no route {path}"}})
        else:
            self._json(401, {"error": {"type": "unauthorized",
                                       "message": "a bearer token or the dashboard session is "
                                                  "needed"}},
                       extra_headers=(("WWW-Authenticate", 'Bearer realm="TandemLLM"'),))
        return False

    def _session(self, method: str, body: dict | None = None) -> None:
        """`/v1/dashboard/session`: POST logs in, GET says whether, DELETE logs out.

        With the dashboard login off there is nothing to log in to: GET says so (`login: false`,
        which the page reads to skip its sign-in screen), and POST and DELETE do not exist."""
        a = _auth()
        if not a.login:
            if method == "GET":
                return self._json(200, {"contract_version": dashboard_api.SESSION_CONTRACT,
                                        "authenticated": True, "login": False,
                                        "expires_at": None},
                                  extra_headers=(("Cache-Control", "no-store"),))
            return self._json(404, {"error": {"type": "not_found",
                                              "message": "the dashboard login is off"}})
        if not a.admin:
            return self._json(404, {"error": {"type": "not_found",
                                              "message": "no route /v1/dashboard/session"}})
        secure = "; Secure" if (self.headers.get("X-Forwarded-Proto") or "") == "https" else ""
        if method == "GET":
            exp = a.session(self.headers)
            if exp is None and not a.admin_bearer(self.headers):
                return self._allowed("dashboard")               # the 401
            exp = exp or int(time.time()) + auth_mod.SESSION_S
            return self._json(200, {"contract_version": dashboard_api.SESSION_CONTRACT,
                                    "authenticated": True, "login": True,
                                    "expires_at": dashboard_api.iso_utc_s(exp)})
        if method == "DELETE":
            a.revoke(self.headers)
            return self._send_empty(204, (("Set-Cookie", f"{auth_mod.COOKIE}=; HttpOnly; "
                                                         f"SameSite=Strict; Path=/; Max-Age=0"
                                                         f"{secure}"),))
        who = self.headers.get("X-Real-IP") or self._peer()
        if a.login_blocked(who):
            return self._json(429, {"error": {"type": "too_many",
                                              "message": "too many failed logins; wait a minute"}},
                              extra_headers=(("Retry-After", "60"),))
        token = (body or {}).get("token") if isinstance(body, dict) else None
        if not isinstance(token, str) or not auth_mod._eq(token, a.admin):
            a.login_failed(who)
            return self._json(401, {"error": {"type": "unauthorized", "message": "wrong token"}},
                              extra_headers=(("WWW-Authenticate", 'Bearer realm="TandemLLM"'),))
        value, _ = a.make_cookie()
        return self._send_empty(204, (("Set-Cookie", f"{auth_mod.COOKIE}={value}; HttpOnly; "
                                                     f"SameSite=Strict; Path=/; "
                                                     f"Max-Age={auth_mod.SESSION_S}{secure}"),))

    def _send_empty(self, code: int, headers: tuple = ()) -> None:
        try:
            self.send_response(code)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Length", "0")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _dashboard_get(self, path: str) -> None:
        """`GET /v1/dashboard/{summary,usage,requests,system,logs,live,metrics}`."""
        from urllib.parse import parse_qs, urlsplit
        if path == "/v1/dashboard/session":
            return self._session("GET")
        if not self._allowed("dashboard"):
            return
        if path == "/v1/dashboard/metrics":
            # The /metrics page under the dashboard's rule: the Performance view reads its
            # 5-minute figures from it, and with the login off a browser has no session that
            # /metrics would take. /metrics itself keeps its own rule for the scrapers.
            return metrics.serve(self)
        q = {k: v[-1] for k, v in parse_qs(urlsplit(self.path).query).items()}
        if path in ("/v1/dashboard/logs", "/v1/dashboard/live"):
            try:
                return (self._dashboard_logs if path.endswith("logs") else self._dashboard_live)(q)
            except dashboard_api.ApiError as exc:
                return self._json(exc.status, exc.body())
        api = dashboard()
        fn = {"/v1/dashboard/summary": api.summary, "/v1/dashboard/usage": api.usage,
              "/v1/dashboard/requests": api.requests,
              "/v1/dashboard/system": lambda _q: api.system()}.get(path)
        if fn is None:
            return self._json(404, {"error": {"type": "not_found", "message": f"no route {path}"}})
        try:
            return self._json(200, fn(q), extra_headers=(("Cache-Control", "no-store"),))
        except dashboard_api.ApiError as exc:
            return self._json(exc.status, exc.body())

    def _dashboard_logs(self, q: dict) -> None:
        """`GET /v1/dashboard/logs`: the backlog, then live lines, as server-sent events.

        Never takes the engine lock. A reader that disconnects is cleaned up without a traceback
        (rule); one that does not read loses its oldest lines and gets `event: gap`.

        A reader that CLOSED is noticed within a second, not at the next write: on a
        quiet log that was the heartbeat, up to `log_ping_s` (15 s) later -- and a write to a
        closed socket only fails the time after -- so a closed tab kept its place under the
        four-stream cap and reopening the Dev tab a few times in a row got 429. And a new stream
        that finds the cap full takes the place of a closed one at once (`LogBuffer.subscribe`).
        """
        buf = STATE.get("log_buffer") or logbuf.BUFFER
        level = q.get("level") or "info"
        if level not in logbuf.LEVELS:
            raise dashboard_api.ApiError(400, "bad_request",
                                         f"level must be one of {', '.join(logbuf.LEVELS)}")
        grep = q.get("grep") or None
        if grep is not None and len(grep) > 128:
            raise dashboard_api.ApiError(400, "bad_request", "grep is at most 128 characters")
        try:
            backlog = int(q.get("backlog") or 500)
            lei = self.headers.get("Last-Event-ID")
            since = q.get("since")
            after = int(lei) if lei else (int(since) if since and since.isdigit() else None)
        except ValueError:
            raise dashboard_api.ApiError(400, "bad_request", "backlog and Last-Event-ID are integers")
        if not 0 <= backlog <= 5000:
            raise dashboard_api.ApiError(400, "bad_request", "backlog must be 0..5000")
        if after is None and since:
            try:
                t = dashboard_api._dt.datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError:
                raise dashboard_api.ApiError(400, "bad_request",
                                             "since is a sequence number or RFC 3339")
            after = buf.after_time(dashboard_api.iso_utc(t.timestamp() * 1000))
        if q.get("follow", "1") in ("0", "false", "no"):
            return self._json(200, {"contract_version": dashboard_api.CONTRACT,
                                    "lines": buf.lines(level=level, after=after, grep=grep,
                                                       limit=backlog),
                                    "last_seq": buf.seq}, extra_headers=(("Cache-Control",
                                                                          "no-store"),))
        sub = buf.subscribe(level, grep, self._reader_gone)  # before the backlog: nothing falls between
        if sub is None:
            raise dashboard_api.ApiError(429, "too_many",
                                         f"at most {logbuf.MAX_SUBSCRIBERS} log streams at once")
        ping = float(STATE.get("log_ping_s", 15.0))
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            w = self.wfile
            last = after or 0

            def send(entries):
                nonlocal last
                for e in entries:
                    if e["seq"] > last:
                        w.write(f"event: log\nid: {e['seq']}\ndata: "
                                f"{json.dumps(e, ensure_ascii=False)}\n\n".encode())
                        last = e["seq"]

            send(buf.lines(level=level, after=after, grep=grep, limit=backlog))
            w.flush()
            quiet = time.monotonic()
            while not STATE.get("draining"):
                items, dropped = sub.take(min(ping, 1.0))
                if sub.closed or self._reader_gone():
                    break
                if dropped:
                    w.write(f"event: gap\ndata: {json.dumps({'dropped': dropped})}\n\n".encode())
                if items:
                    send(items)
                elif not dropped:
                    if time.monotonic() - quiet < ping:
                        continue
                    w.write(b": ping\n\n")
                w.flush()
                quiet = time.monotonic()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_connection = True
        finally:
            buf.unsubscribe(sub)

    def _dashboard_live(self, q: dict) -> None:
        """`GET /v1/dashboard/live` (contract 1.1): the requests in flight.

        `follow=0` is one JSON snapshot. Otherwise server-sent events: first the full snapshot
        with the 5-minute `history` and `recent` (a reconnect gets the same: the snapshot is the
        state, ids are not replayed), then the event the sampler encoded for each tick -- the same
        bytes for every open stream -- whenever its `seq` moved on: `QSE_LIVE_HZ` a second while
        a request is in flight, once a second otherwise. `: ping` every 15 s. Never takes the
        engine lock; a closed reader is noticed within 250 ms; at most `live.MAX_STREAMS` at once.
        """
        reg = live_registry()
        if q.get("follow", "1") in ("0", "false", "no"):
            return self._json(200, reg.snapshot(), extra_headers=(("Cache-Control", "no-store"),))
        if not reg.subscribe():
            raise dashboard_api.ApiError(429, "too_many",
                                         f"at most {live_mod.MAX_STREAMS} live streams at once")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            w = self.wfile
            sent, data = reg.first_event()
            w.write(data)
            w.flush()
            pinged = time.monotonic()
            while not STATE.get("draining"):
                reg.wait_event(sent, 0.25)
                ev = reg.event()
                if ev is not None and ev[0] > sent:
                    sent = ev[0]
                    w.write(ev[1])
                    w.flush()
                if time.monotonic() - pinged >= live_mod.PING_S:
                    pinged = time.monotonic()
                    w.write(b": ping\n\n")
                    w.flush()
                if self._reader_gone():
                    break
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_connection = True
        finally:
            reg.unsubscribe()

    def _reader_gone(self) -> bool:
        """The client closed its end: the socket reads as end-of-file. A stream's reader sends
        nothing after its request, so a readable socket here is the close (or a reset)."""
        try:
            ready, _, _ = select.select([self.connection], [], [], 0)
            return bool(ready) and not self.connection.recv(1, socket.MSG_PEEK)
        except (OSError, ValueError):
            return True

    def _client_gone(self) -> bool:
        """`_reader_gone` for a completion: the request body is read, so the client sends nothing
        more, and a socket that reads as end-of-file is a client that left. No socket (a test's
        handler) is a client that stays."""
        if getattr(self, "connection", None) is None:
            return False
        return self._reader_gone()

    def _prefill_watch(self, stream: bool, rec=None):
        """`on_prefill` for this request. After every prefill chunk: raise `ClientGone` if
        the client left -- the lock is released at once and the rows prefilled so far stay good
        for its retry -- and, once the prefill has run `--prefill-heartbeat-s`, send a
        streamed client an SSE comment at most once a second. A comment is not an event: no
        client reads it as a token, and no first-token clock starts on it."""
        hb = float(STATE.get("prefill_heartbeat") or 0.0)
        t0 = time.perf_counter()
        last = [t0]

        def watch(done: int, total: int) -> None:
            if self._client_gone():
                if rec is not None and rec.client_gone_at is None:
                    rec.client_gone_at = time.perf_counter()     # seen by the handler first
                raise ClientGone(f"client left during the prefill at {done}/{total}")
            now = time.perf_counter()
            if rec is not None:
                # the prefill's progress for the live view, one tuple a chunk (a chunk is
                # a forward of hundreds to thousands of rows); no sync, so `done` is what the host
                # has issued
                p = rec.pf
                rec.pf = ((done, total, now, p[0], p[2], p[5] + 1) if p is not None
                          else (done, total, now, None, None, 1))
            if stream and hb > 0 and now - t0 >= hb and now - last[0] >= HEARTBEAT_EVERY_S:
                last[0] = now
                self.wfile.write(f": prefill {done}/{total}\n\n".encode())
                self.wfile.flush()
        watch.info = {}                  # the prefill fills it: start, chunk, t0, kind
        return watch

    # -------------------------------------------------------------- routes
    def do_GET(self):
        raw = self.path.split("?")[0]
        if raw == "/dashboard" or raw.startswith("/dashboard/"):
            # the dashboard's static shell: public, no data in it
            return static.serve(self, self.path, STATE.get("dashboard_dir"))
        path = raw.rstrip("/") or "/"
        if path.startswith("/v1/dashboard/"):
            return self._dashboard_get(path)
        if path in ("/health", "/healthz", "/v1/health"):
            # What a watchdog needs to decide whether to restart, and what a person needs to
            # decide whether it is wedged or merely busy. `draining` is the difference between
            # "not taking work" and "broken", and a restarter that cannot tell them apart will
            # kill a server in the middle of a graceful shutdown.
            code = 503 if STATE.get("draining") else 200
            if _auth().decide("health_full", self._peer(), self.headers) != "ok":
                # through the proxy, the status and nothing about the configuration
                return self._json(code, {"status": "draining" if STATE.get("draining") else "ok"})
            return self._json(code, {
                "status": "draining" if STATE.get("draining") else "ok",
                "model": STATE.get("model"),
                "uptime_s": round(time.time() - STATE.get("started", time.time()), 1),
                "engine_busy": LOCK.locked(),
                "inflight": dict(INFLIGHT),
                "max_queue": STATE.get("max_queue"),
                "request_timeout_s": STATE.get("request_timeout"),
                "max_len": STATE.get("max_len"),
                "sampling": {"temperature": STATE.get("temperature", 0.0),
                             "top_p": STATE.get("top_p", 1.0),
                             "top_k": STATE.get("top_k", 0)},
                "penalty": {"rep": STATE["pen_spec"].rep,
                            "presence": STATE["pen_spec"].presence,
                            "freq": STATE["pen_spec"].freq,
                            "no_repeat": STATE["pen_spec"].no_repeat},
                "default_max_tokens": STATE.get("default_max_tokens"),
                "cache": cache_stats(),
                "memory": _memory(),
                **({"price_table": STATE["price_table"]} if STATE.get("price_table") else {}),
            })
        if path == "/metrics/up":
            # a public page with one number and nothing else, so the scrape can tell an
            # engine that is down from a metrics token that is wrong (the real page's 401).
            raw = (b"# HELP qse_up 1 while the server takes work, 0 while it drains\n"
                   b"# TYPE qse_up gauge\nqse_up " + (b"0" if STATE.get("draining") else b"1")
                   + b"\n")
            try:
                self.send_response(200)
                self.send_header("Content-Type", metrics.CONTENT_TYPE)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            return
        if path == "/metrics":
            # Prometheus, in its own text format. Everything it reports is collected in
            # server/metrics.py, which wraps this module rather than editing it; see its header.
            if not self._allowed("metrics"):
                return
            return metrics.serve(self)
        if path == "/v1/cache/stats":
            if not self._allowed("cache_read"):
                return
            return self._json(200, cache_stats())
        if path == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": STATE["model"], "object": "model", "created": STATE["started"],
                 "owned_by": "local"}]})
        return self._json(404, {"error": {"message": f"no route {path}", "type": "not_found"}})

    def do_DELETE(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/v1/dashboard/session":
            return self._session("DELETE")
        return self._json(404, {"error": {"message": f"no route {path}", "type": "not_found"}})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        try:
            body = self._read()
        except Exception as exc:
            if path in ("/v1/chat/completions", "/v1/completions"):
                rec = usage_mod.RequestRecord("-", "chat" if path.endswith("chat/completions")
                                              else "completions", False)
                rec.status, rec.model = 400, str(STATE.get("model", ""))
                rec.client_id, rec.client_kind = ledger_mod.client_of(self.headers)
                _account(rec)
            return self._json(400, {"error": {"message": f"bad json: {exc}",
                                              "type": "invalid_request_error"}})
        if path == "/v1/dashboard/session":
            return self._session("POST", body)
        if path == "/v1/cache/clear":
            if not self._allowed("cache_clear"):
                return
            # Measuring a warm number against a cold one needs a way back to cold that is not a
            # server restart, because a restart also throws away the Triton autotuning and the
            # first row would pay for the compiler -- the trap the phase-6 table was thrown away
            # for. This clears the caches and nothing else.
            with LOCK:
                for name in ("state_store", "response_cache", "resident"):
                    obj = STATE.get(name)
                    if obj is not None:
                        obj.clear() if hasattr(obj, "clear") else None
            return self._json(200, cache_stats())
        if path not in ("/v1/chat/completions", "/v1/completions"):
            return self._json(404, {"error": {"message": f"no route {path}",
                                              "type": "not_found"}})
        with logbuf.serving(body):
            return self._complete_logged(body, path)

    def _complete_logged(self, body: dict, path: str) -> None:
        try:
            return self._complete(body, chat=path.endswith("chat/completions"))
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:                                  # noqa: BLE001
            logbuf.print_exc()
            if self._streamed:
                # The response is an event stream whose headers are long gone. A JSON error body
                # written here would be appended to it as garbage, and the client would see the
                # truncation without the reason -- which is precisely bug 2's symptom.
                return
            try:
                if isinstance(exc, grammar_mod.GrammarError):
                    # the constraint left no legal token -- the request's own doing
                    return self._json(400, compat.Refusal(compat.constraint_field(body),
                                                          str(exc)).body())
                return self._json(500, {"error": {"message": str(exc),
                                                  "type": "internal_error"}})
            except Exception:
                return

    _streamed = False

    def _complete(self, body: dict, chat: bool) -> None:
        """One completion request, and its record accounted for on every way out."""
        t_req = time.perf_counter()
        cid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        rec = usage_mod.RequestRecord(cid, "chat" if chat else "completions",
                                      bool(body.get("stream")), t_arrival=t_req)
        rec.model = str(body.get("model") or STATE.get("model", ""))
        rec.client_id, rec.client_kind = ledger_mod.client_of(self.headers)
        # what the live view links and watches -- the conversation (opencode sends its
        # session as `x-session-id`) and the socket the sampler checks for a client that left
        rec.conv = conversation_id(body, self.headers) or _session_header(self.headers)
        rec.sock = getattr(self, "connection", None)
        live_registry().register(rec)                  # visible from now until 30 s after
        try:
            return self._serve(body, chat, rec)
        except (BrokenPipeError, ConnectionResetError):
            raise
        except BaseException:
            if not self._streamed:
                rec.status = 500                   # `do_POST` answers it after this unwinds
            raise
        finally:
            if rec.status != 500:
                rec.status = self._status or rec.status
            if rec.finish_reason is None and rec.status in (429, 503):
                rec.finish_reason = "refused"
            rec.end()
            live_registry().finish(rec)
            _account(rec)

    def _serve(self, body: dict, chat: bool, rec: "usage_mod.RequestRecord") -> None:
        t_req, cid = rec.t_arrival, rec.request_id
        if STATE.get("log_request_keys"):
            print(logbuf.request_keys_line(cid, body), flush=True)
        # a field this engine cannot serve as asked is a 400 that names it, never a silent
        # ignore (server/compat.py holds the disposition of every OpenAI field).
        structured = bool(STATE.get("structured_outputs", True))
        try:
            compat.check(body, chat, structured=structured)
            pattern = compat.structured_pattern(body) if structured else None
            rec.constrained = "response_format" if pattern is not None else None
            # tool_choice required or named is enforced with the same machinery -- the
            # answer as calls to the allowed functions, their values typed by the schemas
            tool_types = None
            if pattern is None and structured and chat:
                forced = compat.tool_constraint(body)
                if forced is not None:
                    pattern, tool_types = forced
                    rec.constrained = "tool_choice"
            # compiled before the lock -- a bad constraint is the client's 400, not a
            # queue slot -- and cached, so a client that sends one schema pays for it once
            gram = grammar_mod.grammar_for(pattern, _grammar_vocab()) if pattern else None
        except compat.Refusal as exc:
            return self._json(400, exc.body())
        except grammar_mod.GrammarError as exc:
            return self._json(400, compat.Refusal(compat.constraint_field(body), str(exc)).body())
        # a request's images are fetched, decoded and preprocessed here, before it
        # queues -- host work, and a bad image is the client's 400, not a queue slot
        images: list = []
        if chat:
            try:
                images = images_mod.collect(
                    body, STATE.get("image_limits") or images_mod.Limits(), STATE.get("image_pre"),
                    act_budget=int(STATE.get("image_act_budget") or 0),
                    act_bytes=(STATE["vision"].activation_bytes if STATE.get("vision") else None))
            except images_mod.ImageRefusal as exc:
                return self._json(400, exc.body())
            rec.images = len(images)
        # Sampling. Greedy is the exact path and stays the default; a request that asks
        # for sampling gets real sampling from engine/sample.py, on the single-token path (no
        # drafter) until rejection sampling lands. Seeded requests reproduce exactly.
        try:
            sampler = Sampler(
                temperature=float(body["temperature"]) if body.get("temperature") is not None
                else float(STATE.get("temperature", 0.0)),
                top_p=float(body["top_p"]) if body.get("top_p") is not None
                else float(STATE.get("top_p", 1.0)),
                top_k=int(body["top_k"]) if body.get("top_k") is not None
                else int(STATE.get("top_k", 0)),
                seed=body.get("seed"),
                draft_temperature=body.get("draft_temperature"),
                min_p=float(body["min_p"]) if body.get("min_p") is not None else 0.0)
        except (TypeError, ValueError) as exc:
            return self._json(400, {"error": {"message": f"bad sampling parameter: {exc}",
                                              "type": "invalid_request_error",
                                              "param": "temperature"}})
        if sampler.temperature > 2.0:
            return self._json(400, {"error": {
                "message": f"temperature must be in [0, 2], got {sampler.temperature}",
                "type": "invalid_request_error", "param": "temperature"}})
        rec.temperature = sampler.temperature          # shown on the live row
        # BUG 2, the first half. This used to default to 256 tokens, and a client that does not
        # send `max_tokens` -- Open WebUI does not -- got an answer that stopped in the middle of
        # a sentence with `finish_reason: "length"` and no other sign that anything had happened.
        # Two hundred and fifty-six tokens is about forty lines of code, which is exactly where
        # the Mario page stopped. A serving default belongs at the context length, not at a
        # number small enough to truncate an ordinary answer.
        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens")
                      or STATE.get("default_max_tokens") or 8192)
        budget = body.get("max_reasoning_tokens")
        if budget is None:
            budget = body.get("thinking_budget")
        if budget is None:
            budget = STATE.get("think_budget") or 0
        budget = int(budget or 0)
        # the budget and the stall close work on BOTH verify paths now -- the forced close
        # runs a chain-shaped forward of the closing phrase, which the tree path writes like any
        # other block. `think` is constructed even without a budget, because the stall
        # detector is the signal that replaces a cut.
        stream = bool(body.get("stream"))
        try:
            fmt = str(body.get("reasoning_format") or STATE.get("reasoning_format") or "tags")
            Reasoning(fmt)                       # validate before anything is generated
        except ValueError as exc:
            return self._json(400, {"error": {"message": str(exc),
                                              "type": "invalid_request_error",
                                              "param": "reasoning_format"}})
        # `body`, `finish` (the default, which is what Open WebUI's base models get),
        # `separate` (the client asked with include_usage) or `none` -- one place, never two.
        where = usage_mod.placement(body, stream, bool(STATE.get("usage_default", True)))
        # Anti-repetition penalties. Deterministic on the target's logits, so greedy and
        # speculative decoding stay identical under the rule -- see engine/penalty.py. The
        # per-request names are the OpenAI ones plus the HF one; defaults come from the server
        # flags. NOTE: clients that already send presence_penalty (Open WebUI's roster rows do)
        # change from ignored to honoured the day this ships -- that is the fix, and it is said
        # out loud in docs/server.md.
        base: PenaltySpec = STATE["pen_spec"]

        def _p(name: str, default: float) -> float:
            v = body.get(name)
            return float(v) if v is not None else default

        try:
            pen_spec = PenaltySpec(rep=_p("repetition_penalty", base.rep),
                                   presence=_p("presence_penalty", base.presence),
                                   freq=_p("frequency_penalty", base.freq),
                                   no_repeat=int(_p("no_repeat_ngram_size", base.no_repeat)),
                                   bias=compat.logit_bias(body, STATE["engine"].cfg.vocab_size))
        except compat.Refusal as exc:
            return self._json(400, exc.body())
        except (TypeError, ValueError) as exc:
            return self._json(400, {"error": {"message": f"bad penalty parameter: {exc}",
                                              "type": "invalid_request_error",
                                              "param": "repetition_penalty"}})
        pen = (PenaltyState(pen_spec, STATE["engine"].cfg.vocab_size, STATE.get("device", "cuda"))
               if pen_spec.on else None)
        pstop = PatternStop(*STATE["pattern_stop"]) if STATE.get("pattern_stop") else None
        stops = body.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        model = body.get("model") or STATE["model"]
        created = int(time.time())
        tok = STATE["tok"]

        conv_id = conversation_id(body, self.headers)
        # choices, log-probabilities. /14: which answers are read for calls -- every
        # chat answer unless `tool_choice` is none -- the request's tool names for the JSON gate,
        # and `parallel_tool_calls: false` as a cap of one.
        n_choices = int(body.get("n") or 1)
        lp_top = compat.top_logprobs(body, chat)
        tool_mode = compat.tool_choice(body)[0] if chat else "none"
        parse_calls = chat and tool_mode != "none"
        tool_names = compat.tool_names(body) if parse_calls else []
        max_calls = 1 if body.get("parallel_tool_calls") is False else None
        # a value the model writes as text goes back as the JSON type its schema asks for
        # (`"offset": 150`, not `"150"`) -- clients validate the arguments against the schema and
        # refuse the string. the constrained parameters keep their JSON-literal reading.
        if parse_calls and body.get("tools"):
            typed = schema_types(body.get("tools"))
            for fname, keys in (tool_types or {}).items():
                typed.setdefault(fname, {}).update({k: JSON_ANY for k in keys})
            tool_types = typed or None

        if STATE.get("draining"):
            rec.stop_detail = "shutting_down"
            return self._busy(503, "the server is shutting down", retry=30)
        with QUEUE:
            if INFLIGHT["waiting"] >= int(STATE.get("max_queue", 8)):
                INFLIGHT["refused"] += 1
                rec.stop_detail = "queue_full"
                return self._busy(503, f"{INFLIGHT['waiting']} requests are already queued and "
                                       f"this engine serves one at a time", retry=5)
            INFLIGHT["waiting"] += 1
        left = False
        try:
            # in one-second slices, so a client that gave up while queued -- opencode
            # cancels and re-sends -- does not get a generation nobody reads ahead of its retry
            wait_until = time.monotonic() + float(STATE.get("queue_timeout", 120.0))
            got = LOCK.acquire(blocking=False)
            while not got:
                if self._client_gone():
                    left = True
                    rec.client_gone_at = time.perf_counter()
                    break
                slice_s = min(1.0, wait_until - time.monotonic())
                if slice_s <= 0:
                    break
                got = LOCK.acquire(timeout=slice_s)
        finally:
            # Off the waiting list whatever happened, including an exception in `acquire` itself.
            # A counter that leaks on the error path turns the queue bound into a slowly closing
            # door, and the symptom -- 503s on an idle server, hours later -- would be read as a
            # leak somewhere else entirely.
            with QUEUE:
                INFLIGHT["waiting"] -= 1
        if left:
            with QUEUE:
                INFLIGHT["abandoned"] += 1
            rec.finish_reason = "abandoned"
            print(f"[req] {cid} left while queued ({(time.perf_counter() - t_req) * 1e3:.0f} ms)",
                  flush=True)
            raise ClientGone("client left while queued")
        if not got:
            with QUEUE:
                INFLIGHT["refused"] += 1
            rec.stop_detail = "queue_timeout"
            return self._busy(429, "timed out waiting for the engine", retry=10)
        with QUEUE:
            INFLIGHT["running"] += 1
        rec.lock_acquired()
        deadline = Deadline(float(STATE.get("request_timeout", 0.0)))
        try:
            prompt, _, in_think = build_prompt(body)
            mm = None
            if images:
                try:
                    prompt, mm = _mm_prompt(prompt, images, rec)
                except ValueError as exc:
                    return self._json(400, {"error": {"message": str(exc),
                                                      "type": "invalid_request_error",
                                                      "param": "messages"}})
            eos = eos_ids(body)
            n_prompt = int(prompt.numel())
            rec.prompt_tokens, rec.thinking = n_prompt, bool(in_think)
            # BUG 2, the second half. The KV buffer is `--max-len` long and the recurrent state is
            # indexed by absolute position, so a generation that runs past the end of it does not
            # degrade -- it raises, part way through a stream whose headers have already gone out.
            # The request is clamped to what the engine can hold, and refused outright when the
            # prompt alone does not fit, which is a 400 the client can read rather than a truncated
            # answer it cannot.
            room = int(STATE["max_len"]) - n_prompt - 1
            if room <= 0:
                rec.stop_detail = "prompt_too_long"
                return self._json(400, {"error": {
                    "message": f"prompt is {n_prompt} tokens and the context is "
                               f"{STATE['max_len']}; nothing is left to generate",
                    "type": "invalid_request_error", "param": "messages"}})
            max_new = max(1, min(max_new, room))
            rec.max_tokens = max_new
            think = ThinkBudget(tok, budget, stall=bool(STATE.get("think_stall", True)))
            rec.live_refs = (think, None)              # read by the live sampler
            prompt_ids = prompt.tolist()
            if gram is not None:
                # the constraint joins the penalties as the last logit processor: every decision
                # site already calls `pen`, so the loop is unchanged
                cons = grammar_mod.Constraint(gram, eos, STATE.get("device", "cuda"),
                                              think_end=think.end_id, in_think=in_think)
                pen = cons if pen is None else grammar_mod.LogitChain([pen, cons])

            # The exact-prompt response cache. Greedy decoding is a function of (prompt, params),
            # so an identical request has an identical answer and this is memoisation rather than
            # an approximation. Under a relaxed accept rule the engine is not answering the
            # greedy question at all, and that is the one setting where the key would be lying
            # about what produced the value -- so the cache is not consulted.
            rcache = STATE.get("response_cache")
            rkey, cached_ids = None, None
            if rcache is not None and not STATE["relax"].on and not sampler.on and lp_top is None:
                # A sampled answer is not a function of (prompt, params) in any replayable sense
                # without the RNG stream; do not memoise it. A request for log-probabilities needs
                # the rows, and a replay has none.
                # The penalty values are part of the question being memoised: greedy under
                # penalties is a different function, and a key without them would replay an
                # answer a different setting produced (cache-key fix).
                tools_key = hashlib.sha256(json.dumps(
                    [body.get("tools"), body.get("tool_choice")], sort_keys=True).encode()
                    ).hexdigest()[:16] if body.get("tools") else ""
                rkey = cache.ResponseCache.key(
                    mm.key_ids if mm is not None else prompt_ids, max_new=max_new, budget=budget, stops=tuple(stops),
                    eos=tuple(sorted(eos)), tree=bool(STATE.get("tree")), pen=pen_spec.key(),
                    tools=tools_key, **({"grammar": gram.pattern} if gram is not None else {}))
                cached_ids = rcache.get(rkey)
            # This request's own prefill publishes a NEW dict here; a replay publishes none.
            prefill_before = STATE.get("last_prefill")
            lpr = lp_mod.Recorder(lp_top) if lp_top is not None else None

            def open_source(samp, budget_state, guard, recorder, cached):
                """One generation's token source: the replay, or the engine."""
                if cached is not None:
                    rec.absorb_response_cache()
                    return rec.track(iter(list(cached)))
                kw = {"lpr": recorder} if recorder is not None else {}
                if mm is not None:
                    kw["mm"] = mm
                return rec.track(generate_stream(prompt, max_new, eos, budget_state, conv_id,
                                                 deadline, pen=pen, pstop=guard, sampler=samp,
                                                 on_prefill=watch, **kw))

            watch = self._prefill_watch(stream, rec)
            rec.prefill_info = watch.info
            source = open_source(sampler, think, pstop, lpr, cached_ids)

            def settle(ids: list[int], finish: str, exc: BaseException | None = None,
                       calls: int = 0, more: tuple = (), prefill: bool = True) -> None:
                """The record's counts, from the ids the engine committed. `more`: the
                earlier choices' ids of an `n > 1` request, whose tokens count too."""
                rec.completion_tokens = len(ids) + sum(len(m) for m in more)
                rec.finish_reason, rec.tool_calls = finish, calls
                if exc is not None:
                    rec.error_type = type(exc).__name__
                if in_think:
                    rec.reasoning_tokens = sum(usage_mod.reasoning_count(
                        x, usage_mod.special_id(tok, "</think>"), think.end_text)
                        for x in (*more, ids))
                if (prefill and cached_ids is None
                        and STATE.get("last_prefill") is not prefill_before):
                    rec.absorb_prefill(STATE.get("last_prefill"))

            if not stream:
                done: list[dict] = []
                for ci in range(n_choices):
                    if ci:
                        # `n`: the next choice has its own draws (a seeded request, its own
                        # seed, so every choice reproduces), guard, budget and recorder; the same
                        # prompt, rules and deadline. The engine holds one sequence, so choices
                        # are generated one after another, and never replayed from the cache.
                        pstop = (PatternStop(*STATE["pattern_stop"])
                                 if STATE.get("pattern_stop") else None)
                        think = ThinkBudget(tok, budget, stall=bool(STATE.get("think_stall", True)))
                        rec.live_refs = (think, None)
                        lpr = lp_mod.Recorder(lp_top) if lp_top is not None else None
                        source = open_source(sampler.for_choice(ci), think, pstop, lpr, None)
                    earlier = tuple(d["ids"] for d in done)
                    ids = []
                    try:
                        for t in source:
                            ids.append(t)
                    except ClientGone:
                        settle(ids, "abandoned", more=earlier, prefill=not ci)
                        _log_request(cid, n_prompt, len(ids), "abandoned", t_req, stream=False,
                                     pen=pen_spec, rec=rec, temp=sampler.temperature)
                        raise
                    except Exception as exc:                          # noqa: BLE001
                        # The [req] line and the `errors` count, as the streamed path has them
                        #; `do_POST` still answers the 500 and prints the traceback.
                        settle(ids, "error", exc, more=earlier, prefill=not ci)
                        _log_request(cid, n_prompt, len(ids), "error", t_req, stream=False,
                                     exc=exc, pen=pen_spec, rec=rec, temp=sampler.temperature)
                        raise
                    last_choice = ci == n_choices - 1
                    if last_choice:
                        _finishing(rec, "saving_state")
                    if cached_ids is None or ci:
                        _remember(prompt_ids, ids, conv_id, mm=mm)
                        if rkey is not None and not ci:
                            rcache.put(rkey, ids, prompt_ids)
                    text = tok.decode(ids, skip_special_tokens=True)
                    if pstop is not None and pstop.hit:
                        finish = "stop"
                    else:
                        finish = "stop" if (ids and ids[-1] in eos) else (
                            "timeout" if deadline.hit else "length")
                    text, cut = _apply_stops(text, stops)
                    if cut:
                        finish = "stop"
                    calls, dropped = [], 0
                    if parse_calls:
                        # Only the answer can call a tool: a call the model writes inside
                        # its reasoning is a thought about calling, not a call.
                        head, answer = _reasoning_head(text, in_think)
                        answer, calls = parse_tool_calls(answer, names=tool_names or None,
                                                         eos=bool(ids) and ids[-1] in eos,
                                                         types=tool_types)
                        if max_calls is not None and len(calls) > max_calls:
                            calls, dropped = calls[:max_calls], len(calls) - max_calls
                        text = head + answer
                        if calls and finish == "stop":
                            finish = "tool_calls"
                    if pstop is not None and pstop.hit:
                        text += GUARD_MARKER
                    settle(ids, finish, calls=len(calls) + sum(len(d["calls"]) for d in done),
                           more=earlier, prefill=not ci)
                    rec.stop_detail = _stop_detail(finish, ids, eos, cut, pstop)
                    if calls:
                        rec.tool_names = (rec.tool_names or []) + [
                            c["function"]["name"] for c in calls]
                    _log_request(cid, n_prompt, len(ids), finish, t_req, stream=False,
                                 pen=pen_spec,
                                 pattern=(pstop.label if pstop is not None and pstop.hit else None),
                                 rec=rec, temp=sampler.temperature,
                                 tools=_tools_note(body, len(calls) + dropped, dropped))
                    done.append({"ids": ids, "text": text, "finish": finish, "calls": calls,
                                 "lpr": lpr, "pstop": pstop})
                # `usage` with its details, and the top-level `timings` and `metrics`.
                # Open WebUI reads only `usage` on this path; the other two are for the rest.
                fields = rec.fields()
                fmt_lp = lp_mod.Formatter(tok) if lp_top is not None else None
                choices = []
                for ci, d in enumerate(done):
                    entries = (lp_mod.emitted(d["lpr"].entries, d["ids"], eos)
                               if d["lpr"] is not None else None)
                    if chat:
                        content, reasoning = split_full(d["text"], fmt, in_think=in_think)
                        message = {"role": "assistant", "content": content}
                        if d["calls"]:
                            message["tool_calls"] = d["calls"]
                        if reasoning is not None:
                            message["reasoning_content"] = reasoning
                        choices.append({"index": ci, "finish_reason": d["finish"],
                                        "logprobs": ({"content": fmt_lp.chat(entries)}
                                                     if entries is not None else None),
                                        "message": message})
                    else:
                        choices.append({"index": ci, "finish_reason": d["finish"],
                                        "logprobs": (fmt_lp.legacy(entries)
                                                     if entries is not None else None),
                                        "text": d["text"]})
                payload = {"id": cid, "object": "chat.completion" if chat else "text_completion",
                           "created": created, "model": model, "usage": fields["usage"],
                           "choices": choices}
                payload["timings"], payload["metrics"] = fields["timings"], fields["metrics"]
                hit = next((d["pstop"] for d in done
                            if d["pstop"] is not None and d["pstop"].hit), None)
                _finishing(rec, "final_chunk")
                return self._json(200, payload, extra_headers=_guard_headers(hit))

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            w = self.wfile
            self._streamed = True
            # One piece of text at a time, and every character in it final: `Detokenizer` holds a
            # code point back until all of its bytes have arrived, which is bug 3. `Reasoning` says
            # which field each piece belongs in, which is the second half of bug 1.
            det = Detokenizer(lambda seq: tok.decode(seq, skip_special_tokens=True))
            split = Reasoning(fmt, in_think=in_think)
            # a tail that could still become a stop string waits until it does or cannot;
            # the stop string is never sent, not even the part of it that arrives first.
            stopper = StopStrings(stops)

            tbuf = (ToolCallBuffer(names=tool_names or None, max_calls=max_calls, types=tool_types)
                    if parse_calls else None)
            rec.live_refs = (think, tbuf)
            # BUG 1. The prompt ended inside `<think>`, so the opening tag is already spent and the
            # model will only ever write the closing one. Put it back as the start of `content`.
            # NOT ahead of the loop. `source` is a generator and the prefill runs inside
            # its first `next()`, so a tag sent before the loop reached the client ahead of the
            # model and every client timing its first content delta read the round trip (3 ms on
            # the box, against a 551 ms request). It rides on the first piece of text instead:
            # the client's first content delta is the model's first token, and still opens with
            # the tag.
            opener = [OPEN_THINK + "\n"] if in_think and fmt in ("tags", "both") else []
            #, the same mistake with the other chunk: the role chunk went out before the
            # loop, and a client that stamps TTFT on the first chunk with a `choices` array
            # (vLLM's bench client does, whatever the chunk holds) read the HTTP round trip. It
            # now goes out in the same write as the first chunk after it -- the first text, or the
            # finish chunk of a stream that has none -- and is still the first chunk, unchanged.
            role = ([_chunk(cid, model, created, {"role": "assistant", "content": ""})]
                    if chat else [])

            def put(data: str) -> None:
                w.write(((role.pop() if role else "") + data).encode())

            # a chunk carries the log-probabilities of the tokens decided since the last
            # chunk that carried some -- text can lag its tokens (a held character, a stop-string
            # prefix, a tool block), so a chunk may carry none or several; the finish chunk the
            # rest.
            fmt_lp = lp_mod.Formatter(tok) if lpr is not None else None
            lp_at = [0, 0]                       # entries sent, characters of text they covered

            def lp_take() -> dict:
                """`logprobs=` for the next chunk; a request that did not ask passes NO_LP."""
                got = lp_mod.emitted(lpr.entries, ids, eos)[lp_at[0]:]
                lp_at[0] += len(got)
                if chat:
                    return {"logprobs": {"content": fmt_lp.chat(got)}}
                out = fmt_lp.legacy(got, lp_at[1])
                lp_at[1] += sum(len(t) for t in out["tokens"])
                return {"logprobs": out}

            def tools_note() -> str | None:
                if tbuf is None:
                    return _tools_note(body, 0)
                return _tools_note(body, len(tbuf.calls) + tbuf.dropped, tbuf.dropped)

            def send(pairs) -> None:
                for field, piece in pairs:
                    if not piece:
                        continue
                    if opener and field != "reasoning":
                        piece = opener.pop() + piece
                    if tbuf is not None and field == "content":
                        # Content is routed through the buffer so a tool-call block is held back
                        # instead of being shown as raw XML; a recognised call streams
                        # its arguments live as OpenAI deltas through the same feed (the rolex_svg
                        # fix: a whole-file call used to arrive in one lump at the very end).
                        for out_piece in tbuf.feed(piece):
                            put(_chunk(cid, model, created, {"content": out_piece},
                                       **(lp_take() if lpr is not None else NO_LP)))
                        for d in tbuf.drain_deltas():
                            put(_chunk(cid, model, created, {"tool_calls": [d]},
                                       **(lp_take() if lpr is not None else NO_LP)))
                        w.flush()
                        continue
                    if chat:
                        key = "reasoning_content" if field == "reasoning" else "content"
                        put(_chunk(cid, model, created, {key: piece},
                                   **(lp_take() if lpr is not None else NO_LP)))
                    else:
                        put(_text_chunk(cid, model, created, piece,
                                        **(lp_take() if lpr is not None else NO_LP)))
                    w.flush()

            ids: list[int] = []
            finish = "length"
            failed: BaseException | None = None
            # reset per stream; the non-streamed branch returns before this point
            cut = False
            try:
                for t in source:
                    ids.append(t)
                    if t in eos:
                        finish = "stop"
                        break
                    piece = det.push(ids)
                    if not piece:
                        continue                   # a byte-level token that is not a character yet
                    send(split.push(stopper.push(piece)))
                    if stopper.hit:
                        finish, cut = "stop", True
                        break
                _finishing(rec, "flush")                 # the token loop has ended
                if not cut:
                    send(split.push(stopper.push(det.flush(ids)) + stopper.finish()))
                    if stopper.hit:
                        finish = "stop"
                if tbuf is not None:
                    # The held text goes out RAW, not through `send()`. `send()` feeds content
                    # back into the same buffer, and what is held still contains the opener: the
                    # block re-opened, the arguments went out a second time at a new index, and
                    # the fallback text was swallowed -- for a bare partial opener ("hello
                    # <tool"), all of it. The sweep that follows carries the calls that were not
                    # streamed live.
                    left, sweep = tbuf.finish(eos=bool(ids) and ids[-1] in eos)
                    if left:
                        put(_chunk(cid, model, created, {"content": left},
                                   **(lp_take() if lpr is not None else NO_LP)))
                    for delta in sweep:
                        put(_chunk(cid, model, created, {"tool_calls": [delta]},
                                   **(lp_take() if lpr is not None else NO_LP)))
                    if left or sweep:
                        w.flush()
                    if tbuf.calls and finish == "stop":
                        finish = "tool_calls"
                    if tbuf.calls:
                        rec.tool_names = [c["function"]["name"] for c in tbuf.calls]
                if pstop is not None and pstop.hit:
                    send([("content", GUARD_MARKER)])
                send(split.finish())
                if pstop is not None and pstop.hit and finish == "length":
                    finish = "stop"
                elif deadline.hit and finish == "length":
                    finish = "timeout"
            except (BrokenPipeError, ConnectionResetError):
                # The reader hung up -- a closed pipe and a reset connection are the same event
                # seen from two kernels, and neither is this server's fault. There is nothing to
                # report and nowhere to report it.
                if rec.client_gone_at is None:
                    rec.client_gone_at = time.perf_counter()
                settle(ids, "abandoned", calls=len(tbuf.calls) if tbuf is not None else 0)
                _log_request(cid, n_prompt, len(ids), "abandoned", t_req, stream=True, pen=pen_spec,
                             pattern=(pstop.label if pstop is not None and pstop.hit else None),
                             rec=rec, temp=sampler.temperature, tools=tools_note())
                raise
            except Exception as exc:                                  # noqa: BLE001
                # BUG 2, the third half. The headers of a stream go out before the first token, so
                # an exception raised half way through used to unwind into `do_POST`, which tried
                # to send a 500 -- a JSON body, with its own status line, appended to a live
                # event stream. Every client on earth reads that as a stream that simply stopped.
                # It is logged here, and the stream is CLOSED PROPERLY: the partial text the reader
                # already has, then a finish reason that says what happened.
                failed = exc
                finish = "error"
                logbuf.print_exc()
            if cached_ids is None and failed is None:
                _finishing(rec, "saving_state")
                _remember(prompt_ids, ids, conv_id, mm=mm)
                if rkey is not None:
                    rcache.put(rkey, ids, prompt_ids)
            settle(ids, finish, failed, calls=len(tbuf.calls) if tbuf is not None else 0)
            rec.stop_detail = _stop_detail(finish, ids, eos, stopper.hit, pstop)
            _log_request(cid, n_prompt, len(ids), finish, t_req, stream=True, exc=failed, pen=pen_spec,
                         pattern=(pstop.label if pstop is not None and pstop.hit else None),
                         rec=rec, temp=sampler.temperature, tools=tools_note())
            # usage, timings and metrics on exactly one chunk -- the finish chunk by
            # default, the separate `choices: []` chunk when the client asked for include_usage.
            fields = rec.fields() if where in ("finish", "separate") else None
            on_finish = fields if where == "finish" else None
            _finishing(rec, "final_chunk")
            try:
                if opener:
                    # No text at all -- a stop string at the first character, or a failure before
                    # the first token. The block still opens, as the non-streamed answer's does.
                    put(_chunk(cid, model, created, {"content": opener.pop()}))
                if failed is not None and chat:
                    put(_chunk(cid, model, created, {}, finish=finish,
                               error={"message": str(failed), "type": type(failed).__name__},
                               extra=on_finish, **(lp_take() if lpr is not None else NO_LP)))
                else:
                    put(_chunk(cid, model, created, {}, finish=finish, extra=on_finish,
                               **(lp_take() if lpr is not None else NO_LP)) if chat
                        else _text_chunk(cid, model, created, "", finish=finish, extra=on_finish,
                                         **(lp_take() if lpr is not None else NO_LP)))
                if where == "separate":
                    w.write(_chunk(cid, model, created, None, extra=fields).encode())
                w.write(b"data: [DONE]\n\n")
                w.flush()
            except BrokenPipeError:
                pass
        finally:
            LOCK.release()
            with QUEUE:
                INFLIGHT["running"] -= 1
                INFLIGHT["served"] += 1


def _memory() -> dict:
    """What a soak run watches for a leak.

    On this board the GPU and the host share one pool and `nvidia-smi` reports N/A for used
    memory, so the numbers that mean anything are the allocator's own -- `allocated` is what the
    engine is holding and `reserved` is what it has taken from the driver and not given back. A
    leak shows as `allocated` climbing across requests; fragmentation shows as `reserved` climbing
    while `allocated` does not.
    """
    out = {}
    try:
        out["allocated_gb"] = round(torch.cuda.memory_allocated() / 2**30, 3)
        out["reserved_gb"] = round(torch.cuda.memory_reserved() / 2**30, 3)
        out["max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    except Exception as exc:                                      # noqa: BLE001
        out["error"] = str(exc)
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["rss_gb"] = round(int(line.split()[1]) / 2**20, 3)
                    break
    except Exception:
        pass
    return out


def cache_stats() -> dict:
    """What `/v1/cache/stats` answers: what is held, what it costs, and what it bought.

    The byte figures are the real ones -- every snapshot is asked for its own `nbytes` and the
    parts are broken out -- because the interesting question about a state cache on a 121 GiB board
    is not whether it hits, it is what a hit costs to keep. On this model the recurrent state is
    ~150 MB per snapshot whatever the prefix length, and the KV is ~0.8 MB a token; a budget is a
    number of conversations long before it is a number of tokens long.
    """
    eng = STATE.get("engine")
    out = {"model": STATE.get("model"),
           "session_cache": bool(STATE.get("session_cache")),
           "prefix_cache": bool(STATE.get("prefix_cache")),
           "prefix_chunk": STATE.get("prefix_chunk", 0),
           "last_prefill": STATE.get("last_prefill")}
    store = STATE.get("state_store")
    out["state_store"] = store.report() if store is not None else None
    res = STATE.get("resident")
    out["resident"] = res.report() if res is not None else None
    rcache = STATE.get("response_cache")
    out["response_cache"] = rcache.report() if rcache is not None else None
    suffix = STATE.get("suffix_store")
    out["suffix_store"] = suffix.report() if suffix is not None else None
    tower = STATE.get("vision")
    if tower is not None:
        ic = STATE.get("image_cache")
        out["vision"] = {"tower_bytes": tower.nbytes, **tower.stats,
                         "embed_cache": ic.report() if ic is not None else None}
    if eng is not None:
        cfg = eng.cfg
        kv_per_token = (len(cfg.attention_layers) * cfg.num_key_value_heads * cfg.head_dim * 2 * 2)
        out["snapshot_cost"] = {
            "recurrent_bytes": eng.state.S.numel() * 4,
            "conv_bytes": eng.state.conv.numel() * eng.state.conv.element_size(),
            "kv_bytes_per_token": kv_per_token,
        }
    return out


def _text_chunk(cid, model, created, piece, finish=None, extra: dict | None = None,
                logprobs: dict | None = None) -> str:
    body = {"id": cid, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": piece, "finish_reason": finish,
                         "logprobs": logprobs}]}
    if extra:
        body.update(extra)
    return "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"


def _reasoning_head(text: str, in_think: bool) -> tuple[str, str]:
    """`(reasoning block, answer)`: the text through the first `</think>` when the prompt opened
    the block, the rest after it. A block that never closed has no answer yet."""
    if not in_think:
        return "", text
    i = text.find("</think>")
    return (text, "") if i < 0 else (text[:i + len("</think>")], text[i + len("</think>"):])


def _stop_index(text: str, stops: list[str]) -> int | None:
    hits = [text.find(s) for s in stops if s and text.find(s) >= 0]
    return min(hits) if hits else None


def _apply_stops(text: str, stops: list[str]) -> tuple[str, bool]:
    i = _stop_index(text, stops)
    return (text[:i], True) if i is not None else (text, False)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--served-model", default="qwen3.8-27b-spark-engine")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=8192,
                    help="context length. The KV buffer is allocated for all of it up front, so "
                         "this is a memory decision as much as a capability one: 64 KiB a token "
                         "of engine KV plus 20 KiB a token per drafter arm — 27.9 GB of KV and "
                         "53.4 GB of whole stack at 262,144 (measured 2026-09-18, "
                         "tools/mem_audit.py)")
    ap.add_argument("--rep-penalty", type=float, default=1.0,
                    help="deterministic repetition penalty on the target's logits, before the "
                         "argmax (1.0 = off, today's output). Exact under speculation because "
                         "acceptance is argmax equality; see engine/penalty.py. Per request: "
                         "send `repetition_penalty` in the body")
    ap.add_argument("--presence-penalty", type=float, default=0.0,
                    help="flat subtraction from tokens already in the history (0 = off). "
                         "Per request: `presence_penalty` in the body — OpenAI clients and "
                         "Open WebUI already send it, so the day this ships those requests "
                         "change from ignored to honoured")
    ap.add_argument("--frequency-penalty", type=float, default=0.0,
                    help="subtraction per occurrence, on tokens already in the history (0 = off). "
                         "Per request: `frequency_penalty` in the body")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="server default sampling temperature (0 = greedy, the exact path). "
                         "A request that sends temperature > 0 samples -- via engine/sample.py -- "
                         "and decodes without a drafter until rejection sampling exists")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--no-repeat-ngram", type=int, default=0,
                    help="forbid the token that would complete an n-gram already seen "
                         "(HF's no_repeat_ngram_size; 0 = off, must be >= 2). Deterministic, so "
                         "speculation stays exact; per request: `no_repeat_ngram_size`")
    ap.add_argument("--pattern-stop", default="",
                    help="end a generation that repeats one short pattern forever, as MAX:MIN:COUNT "
                         "(e.g. 64:1:8 -- a block of 1..64 tokens repeated 8 times ends the "
                         "generation with finish_reason stop). The backstop for a greedy loop the "
                         "penalties cannot break, vLLM's RepetitionDetectionParams; empty = off")
    ap.add_argument("--default-max-tokens", type=int, default=8192,
                    help="what a request that does not send max_tokens gets. It used to be 256 "
                         "and a long answer stopped in the middle of a line")
    ap.add_argument("--reasoning-format", default="tags",
                    choices=("tags", "reasoning_content", "both"),
                    help="how the reasoning block reaches the client: inside <think></think> in "
                         "content (default, and what Open WebUI folds on), as OpenAI-style "
                         "reasoning_content deltas, or both. Overridable per request")
    ap.add_argument("--drafter", default="mtp",
                    choices=("mtp", "router", "merged", "dflash2", "lenrouter", "none"))
    ap.add_argument("--tree", action="store_true",
                    help="verify a draft TREE per step instead of a chain, where the drafter "
                         "builds one; see docs/speculative-decoding.md")
    ap.add_argument("--sampled-tree", nargs="?", const="det", default="",
                    choices=("", "det", "mixed"),
                    help="let a SAMPLED request keep the tree verify. `det` (the bare "
                         "flag): the greedy request's tree, walked by drawing the target's token "
                         "at each node; `mixed`: the drafter's sampled chain as the spine, accepted "
                         "against its q by rejection sampling, with the lattice's siblings. Off by "
                         "default: sampled requests take the q-aware chain")
    ap.add_argument("--budget", type=int, default=16, help="nodes per tree, anchor included")
    ap.add_argument("--df2-temp", type=float, default=1.0)
    ap.add_argument("--corpus", default=_S.get("CORPUS"),
                    help="suffix store for the lookup drafter, used by --drafter merged")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--k", type=int, default=0, help="verify block size; 0 = the drafter's depth")
    ap.add_argument("--draft-head", default=None)
    ap.add_argument("--nvfp4", default=None)
    ap.add_argument("--price-table", default=os.environ.get("QSE_PRICE_TABLE", ""),
                    help="FILE[:NAME], the routers' verify/draft/rollback prices for this "
                         "weight set (engine/prices.py); empty = the NVFP4 constants in the code. "
                         "QWEN38_TREE_MS still overrides the tree curve")
    ap.add_argument("--fp8-head", default=None,
                    help="e4m3 lm_head from tools/quant_head.py; halves the head's 2.54 GB in "
                         "both the verify pass and the block drafter's top-k read. `build` (or "
                         "`build:r1,r2,...`) quantises it at load from the checkpoint's bf16 head "
                         "with the served file's ratios; empty = the bf16 head")
    ap.add_argument("--dflash2-blocks", type=int, default=1,
                    help="chained 8-wide draft blocks; 1 proposes 7 tokens, 2 proposes 14")
    ap.add_argument("--dflash2-path", default="greedy", choices=("greedy", "viterbi"))
    ap.add_argument("--dflash2-ckpt", default=None)
    ap.add_argument("--dflash2-ckpt16", default=None,
                    help="the sixteen-wide drafter, for --drafter lenrouter. The router holds both "
                         "checkpoints and picks the block length per step; see engine/lenrouter.py")
    ap.add_argument("--len-fixed", type=int, default=0,
                    help="pin --drafter lenrouter to one block width (8 or 16). 0 lets the "
                         "policy choose, which since phase 9 means --len-latch: one decision a "
                         "request. 8 and 16 measure the fixed baselines "
                         "through the same code")
    ap.add_argument("--len-latch", action=argparse.BooleanOptionalAction, default=True,
                    help="decide the block width ONCE a request -- four wide blocks, up to four "
                         "narrow probes, then no more switching. ON by default since phase 9, "
                         "because switching arms costs 3-18 %% of acceptance depending on the "
                         "workload whether or not the switch was a good idea. "
                         "--no-len-latch restores the per-block router; needs --len-fixed 0 to do "
                         "anything either way")
    ap.add_argument("--len-explore", type=int, default=32,
                    help="blocks between forced wide probes when nothing suggests one")
    ap.add_argument("--deep", type=int, default=int(_S.get("DEEP")),
                    help="with --drafter lenrouter --tree: after --deep-after wide blocks in a row "
                         "commit their whole width, propose the lookup drafter's long exact "
                         "continuation of this request's text as one chain of up to this many rows "
                         ". 0 = off. Default from QWEN38_DEEP")
    ap.add_argument("--deep-after", type=int,
                    default=int(_S.get("DEEP_AFTER")),
                    help="full wide blocks in a row before a deep chain (2 keeps it off new "
                         "text). Default from QWEN38_DEEP_AFTER")
    ap.add_argument("--drop-idle", action=argparse.BooleanOptionalAction, default=False,
                    help="release the arm that LOSES the latch for the rest of the request: no "
                         "tap, no sync, and nothing of it in the state snapshot. It stops about "
                         "1.5 ms a block of keeping a cache current for a drafter that will not "
                         "draft again, and takes a third off every snapshot, which is a third "
                         "more entries inside the same cache budget. REFUSES to start with "
                         "--no-len-latch, since the release only happens when the latch closes. "
                         "Off by default until the phase-10 measurement says otherwise")
    ap.add_argument("--relax-tau", type=float, default=1.0,
                    help="LOSSY. Accept a drafted token whose probability is at least this fraction "
                         "of the argmax's. 1.0 is the lossless rule and the default")
    ap.add_argument("--relax-rank", type=int, default=1,
                    help="LOSSY. Accept a drafted token among the target's top-r. 1 is lossless")
    ap.add_argument("--think-stall", dest="think_stall", action="store_true", default=True,
                    help="while the reasoning block is open, watch for looping (a short "
                         "pattern repeated) or novelty collapse and SIGNAL the model by closing "
                         "the block with the vendor's phrase -- it concludes and answers instead "
                         "of the stream being cut. --no-think-stall disables it")
    ap.add_argument("--no-think-stall", dest="think_stall", action="store_false",
                    help="disable the stall close (the budget still applies)")
    ap.add_argument("--think-budget", type=int, default=0,
                    help="default cap on reasoning tokens per request, 0 = uncapped. A request may "
                         "override it with `max_reasoning_tokens`. When the cap is reached the "
                         "engine closes the reasoning block itself -- see engine/spec.py's "
                         "ThinkBudget -- which CHANGES THE ANSWER and is not a speed trick")
    ap.add_argument("--reasoning-effort", default=None, choices=("low", "medium", "xhigh"),
                    help="default `reasoning_effort` for the chat template. The template's own "
                         "default is xhigh, which is a paragraph asking the model to validate "
                         "assumptions and weigh alternatives before answering")
    ap.add_argument("--cache-budget-gb", type=float, default=40.0,
                    help="RAM budget for the state cache, in GiB. One snapshot is the 48 recurrent "
                         "states (~150 MB on this model, whatever the prefix length) plus the KV "
                         "of the prefix, so this is a number of CONVERSATIONS, not of tokens. "
                         "0 turns the state cache off entirely")
    ap.add_argument("--no-session-cache", action="store_true",
                    help="do not keep a conversation's state after its turn. On by default: the "
                         "next turn then re-reads the whole conversation through 64 layers")
    ap.add_argument("--no-prefix-cache", action="store_true",
                    help="do not checkpoint a prefill at chunk boundaries. On by default, which "
                         "is what makes a shared system prompt free from the second request on")
    ap.add_argument("--resident-gb", type=float, default=4.0,
                    help="GiB for the resident prefix's anchors -- the recurrent state "
                         "(146.8 MiB) at chunk boundaries of the prompt the KV buffer holds, so the "
                         "next turn of a growing conversation resumes in place, KV not copied, at "
                         "any length (4 GiB = 27 anchors). 0 turns it off")
    ap.add_argument("--resident-stash-gb", type=float, default=2.0,
                    help="GiB for the rows a short unrelated request between two turns may "
                         "overwrite, copied aside and back (~104 kB a row: 2 GiB = ~20k rows)")
    ap.add_argument("--resident-tail", type=int, default=4,
                    help="the newest anchors never evicted (the next turn resumes near the end)")
    ap.add_argument("--prefill-heartbeat-s", type=float, default=5.0,
                    help="a streamed request whose prefill has run this long gets an SSE "
                         "comment (': prefill done/total') after every chunk, at most once a "
                         "second, so a client and every proxy between see the stream is alive. "
                         "Clients ignore comments; 0 = never. The engine checks for a client "
                         "that left after every chunk either way")
    ap.add_argument("--live-activity", choices=("on", "off"), default="on",
                    help="/the live activity on /v1/dashboard/live (states, "
                         "prefill progress, timeline, the last 20 stops) at QSE_LIVE_HZ events a "
                         "second while busy. off = contract 1.1 with those fields null at one "
                         "event a second: the kill switch, and the A/B that isolates its cost")
    ap.add_argument("--max-prefill-rows", type=int, default=8192,
                    help="with the prefix cache off, forward a long prompt in chunks of this many "
                         "rows instead of one call; a 131k-token single forward exhausted the "
                         "board. 0 = one call. With the prefix cache on its chunk applies")
    ap.add_argument("--prefix-chunk", type=int, default=1024,
                    help="tokens between prefill checkpoints, and the forward size of EVERY "
                         "prefill while the prefix cache is on -- the two have to agree or a warm "
                         "prefill is not the same arithmetic as a cold one. A CHUNK COSTS A WHOLE "
                         "16.35 GB WEIGHT READ: a 1,724-token prompt is 1.14x at 1024 and 1.57x "
                         "at 256. 1024 is the default because a prompt "
                         "nobody shares pays that and gets nothing. Drop it to 256 if you serve "
                         "one system prompt to many different tails, where the finer grid wins "
                         "back far more than it costs")
    ap.add_argument("--response-cache", action="store_true",
                    help="OPT-IN. Answer an identical (prompt, params) request from memory. "
                         "Greedy decoding is a function so this is exact, but a server that "
                         "answers from a dictionary must never be what a benchmark measures")
    ap.add_argument("--response-cache-mb", type=float, default=256.0)
    ap.add_argument("--response-cache-ttl", type=float, default=3600.0)
    ap.add_argument("--suffix-store", default=_S.get("SUFFIX_STORE"),
        help="directory for the persistent suffix store of what this engine has read and written, "
             "which the lookup drafter reads as a second corpus. Token ids only, never text, "
             "outside this repository, mode 0700. Empty string turns it off")
    ap.add_argument("--suffix-store-readonly", action="store_true",
                    help="read the suffix store and never append to it: a fixed benchmark measured "
                         "against a store of real traffic must not write itself into it")
    ap.add_argument("--suffix-store-mb", type=float, default=192.0,
                    help="cap on the store, in MiB of int32 token ids (192 MiB = 48 M tokens). "
                         "Over the cap the oldest half is forgotten at the next document boundary")
    ap.add_argument("--suffix-store-scope", default="all", choices=("all", "outputs"),
                    help="`all` remembers prompts and answers, `outputs` only what the engine "
                         "wrote. Both stay on the box")
    ap.add_argument("--request-timeout", type=float, default=900.0,
                    help="wall-clock cap on one generation, checked between blocks; 0 disables. "
                         "A request that hits it finishes with reason 'timeout', which is a "
                         "different thing from 'length' and the caller should be told which")
    ap.add_argument("--max-queue", type=int, default=8,
                    help="requests allowed to wait for the engine before new ones get a 503 with "
                         "Retry-After. This engine serves one sequence at a time")
    ap.add_argument("--queue-timeout", type=float, default=120.0,
                    help="how long a request waits for the engine before a 429")
    # image input
    ap.add_argument("--vision", default="on", choices=("on", "off"),
                    help="load the checkpoint's vision tower (0.92 GB bf16) and accept image_url "
                         "parts in chat requests; off refuses them with a 400. A checkpoint "
                         "without a tower serves text either way")
    ap.add_argument("--image-max-mb", type=float, default=20.0,
                    help="largest image a request may send or link, after base64 decoding")
    ap.add_argument("--image-max-decode-pixels", type=int, default=64_000_000,
                    help="largest canvas an image may declare, checked before its pixels are "
                         "decoded; the processor then resizes to its own limits")
    ap.add_argument("--image-max-pixels", type=int, default=0,
                    help="the processor's resize target cap in pixels (0: the checkpoint's "
                         "preprocessor_config.json, 16,777,216 for Qwen3.8 = 16,384 rows an image)")
    ap.add_argument("--max-images", type=int, default=16, help="images one request may send")
    ap.add_argument("--max-image-rows", type=int, default=65536,
                    help="prompt rows all of a request's images may take together (one row per "
                         "2x2 block of 16-pixel patches; a 1024x1024 image is 1024 rows)")
    ap.add_argument("--image-https", default="on", choices=("on", "off"),
                    help="fetch https:// image URLs (off: data: URLs only)")
    ap.add_argument("--image-https-private", default="off", choices=("on", "off"),
                    help="let https image URLs reach loopback, private and link-local "
                         "addresses (off: public hosts only)")
    ap.add_argument("--image-fetch-timeout", type=float, default=15.0,
                    help="seconds for a whole https image download, redirects included")
    ap.add_argument("--image-cache-mb", type=float, default=512.0,
                    help="encoded images kept on the device by content, so a conversation that "
                         "sends an image again does not encode it again")
    ap.add_argument("--image-act-gb", type=float, default=6.0,
                    help="admission: an image whose encode would need more than this (the "
                         "tower's byte math, engine/vision.py) is refused with a 400")
    ap.add_argument("--usage-default", default=os.environ.get("QSE_USAGE_DEFAULT", "on"),
                    choices=("on", "off"),
                    help="a streamed request that sends no stream_options gets usage, "
                         "timings and metrics on its finish chunk (on, the default: Open WebUI asks "
                         "for include_usage only when a model's Usage capability is ticked), or "
                         "nothing, as before (off). include_usage true/false is honoured either "
                         "way. Default from QSE_USAGE_DEFAULT")
    ap.add_argument("--usage-ledger", default="off",
                    help="the SQLite usage ledger's path, or off (the default). One row per "
                         "request, counts and times only, no text. NOT read from the environment: "
                         "ops/start.sh passes serve.env's QSE_USAGE_LEDGER for the :8000 service, "
                         "and every benchmark server stays off")
    ap.add_argument("--usage-retention-days", type=int, default=ledger_mod.MIN_RETENTION_DAYS,
                    help="rows older than this are pruned daily at 04:00; below 400 only with "
                         "QSE_TEST=1")
    ap.add_argument("--trust-loopback", action=argparse.BooleanOptionalAction, default=True,
                    help="a request from loopback with no X-Forwarded-For / X-Real-IP "
                         "header (the box's own watchdog, row3, gate, soak) needs no token for "
                         "/metrics, the full /health and the cache routes. The reverse proxy sets both "
                         "headers, so tunnelled traffic never qualifies")
    ap.add_argument("--dashboard-login", default=os.environ.get(auth_mod.LOGIN_ENV,
                                                                 auth_mod.LOGIN_DEFAULT),
                    choices=("on", "off"),
                    help="on: /v1/dashboard/* needs the admin token or a session, and the page "
                         "shows its sign-in screen (404 without an admin token). off, the default: "
                         "the dashboard's read API answers without a token and the page opens "
                         "straight into its views. The cache routes, /metrics and the full /health "
                         "keep their own rules either way. Default from QSE_DASHBOARD_LOGIN")
    ap.add_argument("--dashboard-dir", default=None,
                    help="where the dashboard's built files are (default: dashboard/dist in this "
                         "repository); /dashboard/ serves them, or a placeholder when missing")
    ap.add_argument("--fake-engine", action="store_true",
                    help="the e2e: no weights, no GPU -- a deterministic CPU token source "
                         "(server/fake_engine.py) behind the real handler, auth, ledger, metrics "
                         "and logs. Refuses the production ledger")
    ap.add_argument("--fake-tps", type=float, default=200.0,
                    help="with --fake-engine: tokens a second it writes")
    ap.add_argument("--fake-prefill-tps", type=float, default=0.0,
                    help="with --fake-engine: the prefill rate of a FAKE_SLOW_PREFILL prompt "
                         "(0 = 2,000 tokens a second)")
    ap.add_argument("--structured-outputs", action=argparse.BooleanOptionalAction, default=True,
                    help="serve response_format json_object / json_schema and "
                         "structured_outputs (regex, choice, json) as masks over the target's "
                         "rows, exact under speculation. A request without them is untouched; "
                         "--no-structured-outputs refuses them with a 400 as before")
    ap.add_argument("--log-content", action="store_true",
                    help="let exception messages that quote a request into the log. Off by "
                         "default: no line carries prompt, message, tool or answer text")
    ap.add_argument("--log-request-keys", action="store_true",
                    help="/one [body] line per request with the parameter names and "
                         "the scalar values of the non-content ones (model, stream, max_tokens, "
                         "temperature, ...); never messages, prompt, tools or stop strings")
    ap.add_argument("--verbose", action="store_true")
    return ap


def open_ledger(a, *, test: bool | None = None) -> "ledger_mod.Ledger | None":
    """The usage ledger the flags ask for, or None. Refuses a bad configuration."""
    if test is None:
        test = os.environ.get("QSE_TEST") == "1"
    path = None if a.usage_ledger in ("", "off") else a.usage_ledger
    ledger_mod.check_config(path, a.usage_retention_days, test=test)
    if path is None:
        return None
    return ledger_mod.Ledger(path, retention_days=a.usage_retention_days)


def main() -> None:
    ap = parser()
    a = ap.parse_args()
    # argparse does not check a default against `choices`, and a typo in QSE_DASHBOARD_LOGIN
    # ("true", "ON") must not quietly decide who reads the dashboard
    if a.dashboard_login not in ("on", "off"):
        ap.error(f"--dashboard-login / {auth_mod.LOGIN_ENV} is on or off, not "
                 f"{a.dashboard_login!r}")
    # the tokens come from the environment (ops/start.sh sources secrets.env); a token that
    # is too short stops the start here, before the minutes of loading
    STATE["auth"] = auth_mod.Auth.from_env(trust_loopback=a.trust_loopback,
                                           login=a.dashboard_login == "on")
    # every line from here on also reaches /v1/dashboard/logs; the log file is unchanged
    logbuf.install()
    logbuf.CONFIG["log_content"] = bool(a.log_content)
    if a.log_content:
        print("[server] --log-content is ON: exception messages may quote requests in the log",
              flush=True)
    # before the minutes of loading: a ledger configuration that must not start stops here (a
    # fake engine is a test run: it may never write the production ledger)
    led = open_ledger(a, test=True if a.fake_engine else None)
    if a.fake_engine:
        from server import fake_engine
        fake_engine.load(sys.modules[__name__], a)
    else:
        _load(a)
    _serve(a, led)


def _blocking_sync() -> None:
    """Hypothesis 3: the host waits on the GPU by sleeping instead of spinning.

    The loop synchronises twice a round and spends most of a ~95 ms round waiting there; by default
    the CUDA runtime spins a CPU core for it, on a SoC whose CPU and GPU share one power budget. With
    QWEN38_BLOCKING_SYNC=1 the device's primary context -- the one torch uses -- is created with
    CU_CTX_SCHED_BLOCKING_SYNC, set through the driver before torch touches the device. Only how the
    host waits changes, never a value; each wait then costs a wake-up."""
    import ctypes
    cu = ctypes.CDLL("libcuda.so.1")
    dev = ctypes.c_int()
    rc = (cu.cuInit(0), cu.cuDeviceGet(ctypes.byref(dev), 0),
          cu.cuDevicePrimaryCtxSetFlags(dev, 0x04))                  # CU_CTX_SCHED_BLOCKING_SYNC
    flags, active = ctypes.c_uint(), ctypes.c_int()
    cu.cuDevicePrimaryCtxGetState(dev, ctypes.byref(flags), ctypes.byref(active))
    print(f"[server] blocking sync: driver calls {rc}, primary context flags 0x{flags.value:x} "
          f"(active {active.value})", flush=True)
    if any(rc) or not flags.value & 0x04:
        raise SystemExit("[server] QWEN38_BLOCKING_SYNC=1 but the context flag did not take")


def _load(a) -> None:
    """The real engine: weights, drafters, caches, the warm-up and the verify graphs."""
    if _S.get("BLOCKING_SYNC") == "1":
        _blocking_sync()
    from engine.config import load_config
    from engine.loader import Weights
    from engine.model import Qwen38Engine
    from transformers import AutoTokenizer
    t0 = time.time()
    # the price table first: a bad file must stop the start before minutes of weight loading
    from engine import prices as _prices
    _pname, _pentry = _prices.load(getattr(a, "price_table", ""))
    PRICES.clear()
    PRICES.update(_pentry)
    if _pname:
        STATE["price_table"] = _pname
        print(f"[prices] {_pname}: " + ", ".join(f"{k}={v}" for k, v in _pentry.items()))
    cfg = load_config(a.model)
    from engine.tokfp import fingerprint as _tok_fp
    STATE["tokenizer_sha"] = _tok_fp(cfg.path)      # the suffix stores check it
    w = Weights(cfg.path, skip_mtp=a.drafter in ("none", "dflash2", "merged", "lenrouter"),
                nvfp4=a.nvfp4,
                fp8_head=a.fp8_head)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len)
    tok = AutoTokenizer.from_pretrained(cfg.path)
    _load_vision(a, cfg)
    drafter = None
    if a.drafter == "mtp":
        from engine.drafters.mtp import MTPDrafter
        drafter = MTPDrafter(eng, max_len=a.max_len, depth=a.depth, draft_head=a.draft_head)
    elif a.drafter == "router":
        from engine.router import RouterDrafter
        drafter = RouterDrafter(eng, max_len=a.max_len, depth=a.depth)
    elif a.drafter == "merged":
        # The configuration the 11:18 gate passed on, plus the tree: the block drafter priced
        # against the lookup drafter every step, both putting candidates in one verify call.
        from engine.drafters.dflash2 import DFlash2Drafter
        from engine.drafters.ngram import NgramDrafter
        from engine.router import MergedRouter, VERIFY_MS, VERIFY_MS_NVFP4
        table = VERIFY_MS_NVFP4 if a.nvfp4 or _S.get("NVFP4") else VERIFY_MS
        head = DFlash2Drafter(eng, a.dflash2_ckpt, blocks=1, max_len=a.max_len,
                              path=a.dflash2_path, draft_head=a.draft_head)
        head.tree_temp = a.df2_temp
        head._build()
        # `--budget` counts nodes INCLUDING the anchor, because that is what the measured curve is
        # keyed by and where its cliff is: 16 nodes cost 164.4 ms and 17 cost 172. So the drafters
        # get one fewer.
        ng = NgramDrafter(corpus_path=a.corpus, tokenizer_sha=STATE.get("tokenizer_sha"), min_order=3, max_depth=16,
                          node_budget=a.budget - 1, branch_top_k=3, min_expected=0.2,
                          alpha=0.6, corpus_weight=0.5, min_corpus_order=8,
                          verify_base_ms=table[min(table)],
                          verify_per_node_ms=(table[max(table)] - table[min(table)])
                          / (max(table) - min(table)))
        drafter = MergedRouter(ng, head, mtp_depth=head.cfg.block_size - 1,
                               node_budget=a.budget - 1, mtp_ms_per_token=0.0,
                               head_fixed_ms=35.0, adaptive_depth=False,
                               rollback_ms=6.2, verify_ms_table=table)
        a.depth = a.budget - 1
    elif a.drafter == "dflash2":
        from engine.drafters.dflash2 import DFlash2Drafter
        drafter = DFlash2Drafter(eng, a.dflash2_ckpt, blocks=a.dflash2_blocks,
                                 path=a.dflash2_path, draft_head=a.draft_head,
                                 max_len=a.max_len)
        drafter._build()
        # The block width, not `--depth`, is what this drafter proposes per verify pass.
        a.depth = (drafter.cfg.block_size - 1) * a.dflash2_blocks
    elif a.drafter == "lenrouter":
        from engine.drafters.dflash2 import DFlash2Drafter
        from engine.lenrouter import LengthRouter
        if not a.dflash2_ckpt16:
            raise SystemExit("--drafter lenrouter needs --dflash2-ckpt16")
        small = DFlash2Drafter(eng, a.dflash2_ckpt, blocks=1, path=a.dflash2_path,
                               draft_head=a.draft_head, max_len=a.max_len, block=8)
        small._build()
        large = DFlash2Drafter(eng, a.dflash2_ckpt16, blocks=1, path=a.dflash2_path,
                               draft_head=a.draft_head, max_len=a.max_len, block=16)
        large._build()
        if a.tree:
            # The combined configuration: each arm is the lookup drafter's tree merged with that
            # arm's own lattice, and the length router chooses the node budget. One NgramDrafter
            # for both arms -- its suffix index is updated in `observe`, and two arms each
            # observing every block would index every token twice.
            from engine.drafters.ngram import NgramDrafter
            from engine.router import MergedRouter, served_tree_table, tree_nodes
            tree_table = served_tree_table(PRICES.get("tree_ms"))
            ng = NgramDrafter(corpus_path=a.corpus, tokenizer_sha=STATE.get("tokenizer_sha"), min_order=3, max_depth=16,
                              node_budget=large.cfg.block_size - 1, branch_top_k=3,
                              min_expected=0.2, alpha=0.6, corpus_weight=0.5, min_corpus_order=8,
                              verify_base_ms=tree_table[8],
                              verify_per_node_ms=(tree_table[16] - tree_table[8]) / 8)
            arms = [MergedRouter(ng, head, mtp_depth=head.cfg.block_size - 1,
                                 node_budget=tree_nodes(head.cfg.block_size) - 1, mtp_ms_per_token=0.0,
                                 head_fixed_ms=PRICES.get("head_fixed_ms", 27.0), adaptive_depth=False,
                                 rollback_ms=PRICES.get("rollback_ms", 6.4),
                                 verify_ms_table=dict(tree_table), tree_ms_table=dict(tree_table))
                    for head in (small, large)]
            drafter = LengthRouter(arms[0], arms[1], fixed=a.len_fixed,
                                   explore_period=a.len_explore, tree=True, ngram=ng,
                                   verify_table=PRICES.get("lenrouter_tree_ms"),
                                   draft_table=PRICES.get("lenrouter_draft_ms"),
                                   latch=a.len_latch, drop_idle=a.drop_idle, deep=a.deep,
                                   deep_after=a.deep_after, latch_table=dict(tree_table))
        else:
            drafter = LengthRouter(small, large, fixed=a.len_fixed,
                                   explore_period=a.len_explore, latch=a.len_latch,
                                   width_trim=not a.len_latch, drop_idle=a.drop_idle)
        # The router may propose the wide block on any step, so the loop's cap has to be the wide
        # one; asking it for fewer would silently pin it to the narrow length. With the deep chain
        # it is the deep width, for the same reason.
        a.depth = max(large.cfg.block_size - 1, a.deep - 1 if a.tree else 0)
    gen_cfg = os.path.join(cfg.path, "generation_config.json")
    cfg_eos = None
    if os.path.isfile(gen_cfg):
        cfg_eos = json.load(open(gen_cfg)).get("eos_token_id")
    relax = Relax(a.relax_tau, a.relax_rank)

    # --- the serving-time caches (engine/cache.py) ---------------------------------------------
    session_on = not a.no_session_cache
    prefix_on = not a.no_prefix_cache
    budget = int(a.cache_budget_gb * (1 << 30))
    if (session_on or prefix_on) and not cache.drafter_is_cacheable(drafter):
        # This drafter's cache is indexed by absolute position and it cannot hand it over. A
        # restored prefix would leave a permanent hole in it and the engine would decode at one
        # token a block, which is a far worse trade than a cold prefill.
        print(f"[cache] drafter {a.drafter} cannot snapshot its own cache; state cache OFF")
        session_on = prefix_on = False
    store = cache.StateStore(budget, chunk=a.prefix_chunk,
                             max_entry_bytes=budget // 4) if (budget and
                                                              (session_on or prefix_on)) else None
    # the resident prefix needs the prefill's chunk grid, which only the prefix cache has
    resident = (cache.ResidentPrefix(int(a.resident_gb * (1 << 30)), a.prefix_chunk,
                                     tail=a.resident_tail,
                                     stash_bytes=int(a.resident_stash_gb * (1 << 30)))
                if (prefix_on and a.resident_gb > 0 and a.prefix_chunk > 0
                    and cache.resident_capable(drafter)) else None)
    rcache = (cache.ResponseCache(int(a.response_cache_mb * (1 << 20)), a.response_cache_ttl)
              if a.response_cache else None)
    suffix = None
    if a.suffix_store:
        suffix = cache.PersistentSuffixStore(
            a.suffix_store, max_tokens=int(a.suffix_store_mb * (1 << 20)) // 4,
            readonly=a.suffix_store_readonly, tokenizer_sha=STATE.get("tokenizer_sha")).open()
        reader = next((d for d in (drafter, getattr(drafter, "ngram", None),
                                   getattr(drafter, "engram", None))
                       if hasattr(d, "add_store")), None)
        if reader is not None:
            reader.add_store(suffix)
        else:
            print(f"[cache] drafter {a.drafter} has no lookup store; the suffix store is being "
                  f"written but nothing reads it")

    STATE.update(engine=eng, tok=tok, drafter=drafter, k=a.k or a.depth, device="cuda",
                 tree=bool(a.tree), sampled_tree=a.sampled_tree, relax=relax,
                 model=a.served_model, started=int(time.time()), verbose=a.verbose,
                 max_len=int(a.max_len), default_max_tokens=int(a.default_max_tokens),
                 reasoning_format=a.reasoning_format, request_timeout=float(a.request_timeout),
                 max_queue=int(a.max_queue), queue_timeout=float(a.queue_timeout),
                 draining=False,
                 cfg_eos=cfg_eos, think_budget=a.think_budget, think_stall=a.think_stall,
                 reasoning_effort=a.reasoning_effort,
                 pen_spec=PenaltySpec(a.rep_penalty, a.presence_penalty,
                                     a.frequency_penalty, a.no_repeat_ngram),
                 temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                 pattern_stop=(tuple(int(x) for x in a.pattern_stop.split(":"))
                               if a.pattern_stop else None),
                 state_store=store, session_cache=session_on, prefix_cache=prefix_on,
                 resident=resident, prefill_heartbeat=float(a.prefill_heartbeat_s),
                 prefix_chunk=cache.prefill_chunk(prefix_on, a.prefix_chunk, a.max_prefill_rows),
                 response_cache=rcache, suffix_store=suffix, suffix_scope=a.suffix_store_scope,
                 usage_default=(a.usage_default == "on"))
    print(f"[server] {w.report()}")
    if relax.on:
        print(f"[server] LOSSY ACCEPT RULE ON: tau={relax.tau} rank={relax.rank}. Output is not "
              f"greedy and not reproducible against any lossless run. See docs/exactness.md")
    print(f"[cache] session={'on' if session_on else 'off'} "
          f"prefix={'on' if prefix_on else 'off'}/{a.prefix_chunk} "
          f"budget={a.cache_budget_gb:.0f} GiB "
          f"response={'on' if rcache else 'off'} "
          f"resident={(f'{a.resident_gb:g}+{a.resident_stash_gb:g} GiB' if resident else 'off')} "
          f"suffix={(suffix.report()['tokens'] if suffix else 0)} tokens")
    print(f"[server] drafter={a.drafter} depth={a.depth} k={STATE['k']} "
          f"nvfp4={w.nvfp4_source or 'off'} fp8_head={(w.fp8_head_source if str(w.fp8_head_source).startswith('build') else 'on') if w.fp8_head_source else 'off'} "
          f" loaded in {time.time() - t0:.1f}s")
    # one warm request, so the first measured one is not paying for Triton autotuning
    with torch.no_grad():
        list(generate_stream(tok("warm up the kernels", return_tensors="pt").input_ids[0].cuda(),
                             4, set()))
    # the verify graphs for every width at the first context class, here rather than
    # inside the first requests (one capture is a fraction of a second, thirty are several)
    if eng._graphs_for(2, 0) is not None:
        t_g = time.time()
        with torch.no_grad():
            # a router that sizes every block on the staircase (QWEN38_LEN_MODE calc/wide) submits
            # any row count up to 32; the others stay at 2..16 and capture the rest lazily
            n_g = eng._graphs.precapture(widths=range(2, 33) if getattr(drafter, "calc", False)
                                         else range(2, 17))
        print(f"[server] verify graphs: {n_g} captured in {time.time() - t_g:.1f}s", flush=True)


def _load_vision(a, cfg) -> None:
    """The checkpoint's vision tower (bf16, as stored), its image processor and the
    embedding cache -- when the checkpoint has a tower and `--vision on`."""
    from engine.vision import EmbedCache, VisionTower, load_vision_config
    vcfg = load_vision_config(cfg.path) if getattr(a, "vision", "on") == "on" else None
    if vcfg is None:
        print("[vision] off" + (" (the checkpoint has no vision tower)"
                                if getattr(a, "vision", "on") == "on" else ""), flush=True)
        return
    t0 = time.time()
    tower = VisionTower.from_checkpoint(cfg.path, vcfg, device="cuda")
    pre = images_mod.Preprocessor(cfg.path, vcfg.spatial_merge_size,
                                  max_pixels=int(a.image_max_pixels), dtype=tower.dtype)
    STATE.update(vision=tower, image_pre=pre,
                 image_cache=EmbedCache(int(a.image_cache_mb * (1 << 20))),
                 image_limits=images_mod.Limits(
                     max_bytes=int(a.image_max_mb * (1 << 20)),
                     max_decode_pixels=int(a.image_max_decode_pixels),
                     max_images=int(a.max_images), timeout_s=float(a.image_fetch_timeout),
                     https=a.image_https == "on",
                     allow_private=a.image_https_private == "on",
                     max_image_rows=int(a.max_image_rows)),
                 image_act_budget=int(a.image_act_gb * (1 << 30)))
    # the kernels' first launch here, not in the first image request: a 4x4-patch image
    with torch.no_grad():
        m = vcfg.spatial_merge_size * 2
        tower.encode(torch.zeros(m * m, vcfg.patch_dim), (1, m, m))
    tower.stats.update(images=0, patches=0, ms=0.0)
    print(f"[vision] tower {tower.nbytes / 1e9:.2f} GB bf16, {vcfg.depth} blocks, "
          f"{len(tower.t)} tensors; processor {type(pre.proc).__name__}; "
          f"loaded in {time.time() - t0:.1f}s", flush=True)


def _serve(a, led) -> None:
    """What every server does once its engine is loaded, the real one or the fake one."""
    STATE.update(identity())
    STATE["args"] = vars(a)
    print(f"[server] {STATE['auth'].describe()}", flush=True)
    STATE["log_request_keys"] = bool(a.log_request_keys)
    # the fake engine's writer applies no logit processor, so it cannot honour a constraint
    STATE["structured_outputs"] = bool(getattr(a, "structured_outputs", True)) and not a.fake_engine
    STATE["dashboard_dir"] = a.dashboard_dir
    STATE["live_hz"] = live_mod.hz_from_env(os.environ.get("QSE_LIVE_HZ"))
    STATE["live_activity"] = getattr(a, "live_activity", "on") == "on" and STATE["live_hz"] > 0
    STATE.pop("live", None)                      # built with the settings above
    print(f"[server] live activity {'on' if STATE['live_activity'] else 'off'}, "
          f"{STATE['live_hz'] if STATE['live_activity'] else 1} Hz while busy", flush=True)
    if led is not None:
        STATE["ledger"] = led.open()
        print(f"[ledger] on: {led.path}, retention {led.retention_days} days", flush=True)
    else:
        print("[ledger] off", flush=True)
    # After the warm-up, so the request that pays for Triton autotuning is not in the histograms.
    metrics.install(sys.modules[__name__])
    live_registry().start()                      # samples only while something is in flight
    print(f"[server] listening on http://{a.host}:{a.port}  model {a.served_model}", flush=True)
    print(f"[server] queue max {a.max_queue} wait {a.queue_timeout:.0f}s "
          f"request timeout {a.request_timeout:.0f}s", flush=True)
    serve_until_drained(ThreadingHTTPServer((a.host, a.port), Handler))


def identity() -> dict:
    """What this process is: the version file, the code hash row3 records, the git commit.

    The served directory is an rsync copy with no `.git`, so the commit comes from `.git-sha`
    when the deploy wrote one.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {"version": "", "code_sha256": "", "code_sha": "", "git_sha": ""}
    try:
        out["version"] = open(os.path.join(root, "VERSION")).read().strip()
    except OSError:
        pass
    try:
        from pathlib import Path
        from tools.row3 import code_hash, git_head
        out["code_sha256"] = code_hash(Path(root))
        out["code_sha"] = out["code_sha256"][:16]
        out["git_sha"] = (git_head(Path(root)) or "")[:12]
    except Exception:                                              # noqa: BLE001
        pass
    if not out["git_sha"]:
        try:
            out["git_sha"] = open(os.path.join(root, ".git-sha")).read().strip()[:12]
        except OSError:
            pass
    return out


def serve_until_drained(httpd) -> None:
    """Serve until SIGTERM/SIGINT, then drain: refuse new work, finish what is running, exit."""

    def _drain(signum, _frame):
        """SIGTERM/SIGINT: stop taking work, let the generation in flight finish, then exit.

        A hard kill in the middle of a verify leaves the caller with a truncated stream and the
        watchdog with no way to tell a crash from a deploy. `draining` makes new requests a 503
        with `Retry-After`, `/health` says `draining` rather than `ok`, and `shutdown()` stops the
        listener -- it has to run off the serving thread or it deadlocks against the loop it is
        stopping. The wait for the work in flight is below, after `serve_forever` returns.
        """
        if STATE.get("draining"):
            return
        STATE["draining"] = True
        print(f"[server] signal {signum}: draining, {INFLIGHT['running']} running, "
              f"{INFLIGHT['waiting']} queued", flush=True)
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _drain)
    signal.signal(signal.SIGINT, _drain)
    try:
        httpd.serve_forever()
    finally:
        # `shutdown()` stops the LISTENER; the handler threads are ThreadingHTTPServer's, and
        # those are daemons. Returning from here used to end the interpreter under the stream the
        # drain had promised to finish -- every graceful stop, every hold's stop, every deploy
        # restart cut the generation in flight. So wait for the work that was admitted:
        # the one running and any already queued for the lock. stop.sh's grace period is the cap
        # on how long a stop may take; this is not a second one.
        while True:
            with QUEUE:
                busy = INFLIGHT["running"] + INFLIGHT["waiting"]
            if not busy:
                break
            time.sleep(0.2)
        # The usage ledger is the one thing that does persist: what is queued is flushed now,
        # after the last request's row was handed over.
        led = STATE.get("ledger")
        if led is not None:
            led.close()
        # The engine holds one sequence and the drafters hold caches indexed by absolute position;
        # neither survives the process, so there is nothing else to persist. What is worth printing is
        # the tally, because a soak run reads it from the last line of the log.
        print(f"[server] stopped. {INFLIGHT}", flush=True)


if __name__ == "__main__":
    main()
