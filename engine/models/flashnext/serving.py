"""Flash-Next behind the server's own generation loop (SRV-87): an engine and a drafter with the shapes
`server/app.py` and `engine/cache.py` drive the 27B through.

The server's loop (`_generate_stream`) speaks a small interface: `forward` (a prompt chunk, a forced
phrase or one decode token at the state's length), `forward_tree` (a draft block; logits of every
row), `commit_tree` (keep a path), `hidden_post_norm` (the rows a drafter syncs from), `picks` (the
verify's own greedy tokens), `kv.length`, `reset`; the walk down a block is the shared one
(engine/methods/walk.py). The caches and the dashboard add the adapter's cold verbs (ENG-E8): `cost`,
`settle`, `discard_pending`, `trim_pending`. `FlashNextServing` maps it onto `FlashNextEngine`:

  forward, one token      the captured decode step (`DecodeGraph`), streams of that row
  forward, more tokens    the prefill kernels (`FlashNextEngine.prefill`), every row's streams
  forward_tree            `verify` from captured graphs: logits [M, vocab], the greedy picks, streams
  commit_tree             `commit` from captured graphs (the folded GDN commit included)
  forward_block           a chain verify whose commit waits for `rollback_to` or the next call

so everything above it -- penalties, the think budget and its forced close, pattern stops, logprobs,
sampled requests walked down the greedy tree (`--sampled-tree det`), tool calls, the live view -- is
the 27B's code. A verified row is the decode step's row bit for bit (ENG-227), so greedy speculation
writes the plain greedy text and a sampled walk draws exactly what plain sampled decoding draws.

`FlashNextTreeDrafter` is the MTP layer as the loop's tree drafter: the loop syncs the rows it
committed (`sync(tokens, streams, pos)`); a row's MTP input pairs its streams with the token after it,
known when the next row arrives or, for the last row, when the next proposal names its anchor. The
backlog is fed in one call (`FlashNextEngine.mtp_prime`) except the last rows, which ride the draft's
graph; StairCut cuts the chain and learns from the verify the loop timed (`on_verify`).
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from engine.drafters import run_steps
from engine.methods.manifest import Manifest
from engine.models.adapter import CostModel
from engine.models.flashnext.engine import DecodeGraph, FlashNextEngine, MTPChainGraph
from engine.models.flashnext.ref import FNModelState
from engine.models.flashnext.spec import MTPGraphDrafter, StairCut
from engine.settings import SETTINGS as _S
from engine.tree import DraftTree


#: fingerprint v3 (`FlashNextServing.manifest`): the package's files that compute rows, the tables
MODULES = ("engine/models/flashnext/__init__.py", "engine/models/flashnext/config.py",
           "engine/models/flashnext/engine.py", "engine/models/flashnext/load.py",
           "engine/models/flashnext/pack.py", "engine/models/flashnext/ple.py",
           "engine/models/flashnext/ref.py", "engine/models/flashnext/spec.py", "engine/settings.py")
TABLES = ("ops/fn-exl3-tiles.json", "ops/fn-dense-splits.json")


def manifest() -> Manifest:
    """What Flash-Next's stored rows depend on (fingerprint v3, ENG-243): its package's arithmetic
    (not this adapter), the knob registry, its own tile tables; the `tools/` kernels follow."""
    return Manifest(arch="qwen4_exp", modules=MODULES, data=TABLES)


class _KV:
    """The `kv` the server reads: the length of the sequence the state holds (a pending chain commit
    lands first), its capacity. Setting it to zero is a reset; to any other length it keeps the KV
    rows by length and clamps the pooled keys -- the recurrent state is the caller's to restore first
    (`state.restore`), as on the 27B."""

    pages = None

    def __init__(self, serving: "FlashNextServing"):
        self._s = serving

    @property
    def length(self) -> int:
        self._s._settle_chain()
        return self._s.st.length

    @length.setter
    def length(self, v: int) -> None:
        v = int(v)
        if v == 0:
            self._s.reset()
            return
        if not 0 < v <= self.max_len:
            raise ValueError(f"Flash-Next's state cannot hold {v} rows (capacity {self.max_len})")
        self._s.st.length = v
        self._s.st.clamp_pool()

    @property
    def max_len(self) -> int:
        return self._s.max_len


class FlashNextServing:
    """`FlashNextEngine` with one state and its graphs, in the server's engine interface."""

    def __init__(self, fn: FlashNextEngine, max_len: int):
        self.fn, self.cfg, self.max_len = fn, fn.cfg, max_len
        self.st = fn.new_state(max_len)
        self.state = FNModelState(self.st)  # recurrent, conv and PLE sections (engine/models/state.py)
        self.kv = _KV(self)
        self.dg = None                      # the decode step's graphs, made on the first decode token
        self.vgraphs: dict = {}
        self.cgraphs: dict = {}
        self.picks = None
        self.mm, self.pos_delta, self.tap = None, 0, None
        self._hidden = None                 # [1, rows, hc H]: the last forward's streams
        self._chain = None                  # a forward_block's rows awaiting their commit
        self.commit_ms = 0.0
        self.captured = False               # the last block captured a verify or commit graph
        self.stats = {"prefill_rows": 0, "decode_steps": 0, "verify_blocks": 0, "verify_rows": 0}

    # --- the loop's interface -------------------------------------------------------------------
    @property
    def hidden_post_norm(self) -> torch.Tensor:
        return self._hidden

    def cost(self) -> CostModel:
        """What a state snapshot holds: its sections whole (the recurrent, conv and PLE ones) and the
        cache rows a token adds (K, V and the indexer key of every attention layer, bf16)."""
        sec = {k: t.numel() * t.element_size() for k, t in self.state.sections().items()}
        c = self.cfg
        per_tok = len(c.attention_layers) * (c.num_key_value_heads * c.head_dim * 2
                                             + (c.indexer_head_dim if c.has_indexer else 0)) * 2
        return CostModel(section_bytes=sum(sec.values()), view_bytes_per_token=per_tok,
                         parts={"recurrent": sum(b for k, b in sec.items() if k.startswith("gdn.S")),
                                "conv": sum(b for k, b in sec.items() if k.startswith(("gdn.conv", "ple.conv")))})

    def manifest(self) -> Manifest:
        return manifest()

    def settle(self) -> None:
        """Land what is pending, so the state's sections are the sequence's whole (before a
        snapshot): a chain block's commit, and the GDN rows the folded commit parked for the next call."""
        self._settle_chain()
        if self.fn._pend_owner is self.st:
            self.fn.flush()

    def discard_pending(self) -> None:
        """Forget what is pending after the state was overwritten (a restore): the chain block, the
        last verify's picks and the GDN rows parked for the state that was there."""
        self._chain = None
        self.picks = None
        if self.fn._pend_owner is self.st:
            self.fn._pend_n.zero_()
            self.fn._pend_owner = None

    def trim_pending(self, n: int) -> bool:
        """Bring the state back to length `n` inside work not yet committed: a chain block
        (`forward_block`) whose commit waits keeps only its rows below `n` (none: the verify moved
        nothing but its own KV rows). False, and nothing changed, when the state already absorbed the
        rows past `n` -- a verified path is committed whole, conv tails included (cutting inside it is
        ENG-235's)."""
        n, L = int(n), self.st.length
        if self._chain is None:
            return n == L
        if not L <= n <= L + self._chain:
            return False
        self._chain = None
        if n > L:
            self.commit_tree(list(range(n - L)))
        return True

    def window_ok(self, i: int) -> bool:
        return False                        # no layer-major prefill window (engine/cache.py)

    def reset(self) -> None:
        self._chain = None
        self.picks = None
        self.fn.reset(self.st)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, start: int, last_only: bool = True) -> torch.Tensor:
        """Rows `ids` at positions start..: one token through the decode step, more through the prefill
        kernels. Logits [1, 1, vocab] of the last row (fp32)."""
        self._settle_chain()
        ids = ids.view(-1)
        self._at(start)
        T = ids.numel()
        if T == 1 and start > 0:
            if self.dg is None:
                self.dg = DecodeGraph(self.fn, self.st)
            cls = self.fn.qsa_class(start)
            lg = self.dg(int(ids[0]))
            self._hidden = self.dg.stacks[cls].view(1, 1, -1)
            self.stats["decode_steps"] += 1
            return lg.view(1, 1, -1)
        lg = self.fn.prefill(ids, self.st, keep_stack=True)
        self._hidden = self.fn.prompt_stack.unsqueeze(0)
        self.stats["prefill_rows"] += T
        return lg.view(1, 1, -1)

    @torch.no_grad()
    def forward_tree(self, block: torch.Tensor, parents: list[int], start: int) -> torch.Tensor:
        """A draft block in DFS pre-order (row 0 the anchor at `start`): every row's logits [M, vocab]
        fp32 (a static buffer), its greedy pick in `picks`, its streams in `hidden_post_norm`. Nothing
        but the block's own KV rows is written until `commit_tree`."""
        self._settle_chain()
        self._at(start)
        toks = [int(t) for t in block.view(-1).tolist()]
        M = len(toks)
        n_graphs = len(self.vgraphs)
        self.fn.verify(toks, [int(p) for p in parents], self.st, self.vgraphs)
        self.captured = len(self.vgraphs) > n_graphs
        rb = self.fn._rb[M]
        self.picks = rb["next"][:M]
        self._hidden = self.fn.rec["stack"][:M].unsqueeze(0)
        self.stats["verify_blocks"] += 1
        self.stats["verify_rows"] += M
        return rb["logits"][:M]

    @torch.no_grad()
    def commit_tree(self, path: list[int]) -> None:
        t0 = time.perf_counter()
        n_graphs = len(self.cgraphs)
        self.fn.commit(self.st, [int(p) for p in path], self.cgraphs)
        self.commit_ms = (time.perf_counter() - t0) * 1e3
        self.captured |= len(self.cgraphs) > n_graphs

    @torch.no_grad()
    def forward_block(self, block: torch.Tensor, start: int) -> torch.Tensor:
        """A chain block: its rows' logits; the rows are kept by `rollback_to(keep)`, or all of them
        by the next call on the engine."""
        n = block.numel()
        lg = self.forward_tree(block, list(range(-1, n - 1)), start)
        self._chain = n
        return lg

    @torch.no_grad()
    def rollback_to(self, keep: int) -> None:
        self._chain = None
        self.commit_tree(list(range(int(keep))))

    # --- internals -------------------------------------------------------------------------------
    def _settle_chain(self) -> None:
        if self._chain is not None:
            n, self._chain = self._chain, None
            self.commit_tree(list(range(n)))

    def _at(self, start: int) -> None:
        if int(start) != self.st.length:
            raise ValueError(f"Flash-Next forward at {start}, the state holds {self.st.length} rows")


class FlashNextTreeDrafter:
    """The MTP layer as the server loop's tree drafter (see the module's docstring)."""

    name = "mtp"
    last_q = None
    wants_rows = False

    def __init__(self, serving: FlashNextServing, depth: int = 4, siblings: bool = False):
        self.s, self.fn, self.depth = serving, serving.fn, depth
        self.cut = StairCut(siblings=siblings)
        self.mtp = MTPGraphDrafter(self.fn, depth, serving.max_len, cut=self.cut)
        self.held = None                    # (tokens, streams) of the last rows synced
        self.queue: list = []               # (streams, next tokens) not fed yet
        self.round = None                   # the draft ms of the block awaiting its verify
        self.verify_ms = None
        self.stats = {"rounds": 0, "draft_ms": 0.0, "fed_rows": 0}

    def report(self) -> str:
        """The verbose server's line at each request: rounds, draft ms, StairCut's prices and offset."""
        c = self.cut
        return (f"mtp rounds={self.stats['rounds']} fed={self.stats['fed_rows']} "
                f"draft_ms={c.draft_ms if c.draft_ms is None else round(c.draft_ms, 2)} delta={round(c.delta, 2)} "
                f"V={ {r: round(v, 1) for r, v in sorted(c.V.items())} } "
                f"spine_cal={[round(c.c('spine', (b + 0.5) / 10), 2) for b in range(10)]}")

    def capture_all(self) -> None:
        self.mtp.capture_all()

    # --- the loop's drafter interface -------------------------------------------------------------
    def reset(self) -> None:
        self.mtp.st.length = 0
        self.mtp.st.pool_n = {}
        self.mtp.pending = None
        self.held, self.queue, self.round, self.verify_ms = None, [], None, None

    def prime(self, tokens: list[int]) -> None:
        pass                                # the prompt's rows come through `sync`

    def set_sampling(self, sampler) -> None:
        pass                                # the draft is the greedy tree; a sampled walk draws on it

    def sync(self, tokens: list[int], streams: torch.Tensor, pos: int, rows=None) -> None:
        toks = [int(t) for t in tokens]
        h = streams.reshape(len(toks), -1)
        if h.shape[0] <= 64:
            h = h.clone()                   # a static buffer the next forward writes again
        if self.held is not None:
            ht, hs = self.held
            self.queue.append((hs, ht[1:] + [toks[0]]))
        self.held = (toks, h)

    def observe(self, tokens: list[int]) -> None:
        # a round that captured a graph took the capture's time: StairCut learns nothing from it
        # (its price for that row count would start at ~170 ms and the cut would never ask again)
        if self.round is not None and self.verify_ms is not None and not self.s.captured:
            self.mtp.observe(list(range(len(tokens))), self.verify_ms + self.s.commit_ms, self.round)
        self.round, self.verify_ms = None, None

    def on_verify(self, rows: int, ms: float) -> None:
        self.verify_ms = ms

    def propose_tree(self, context: list[int], k: int) -> DraftTree | None:
        return run_steps(self.propose_tree_steps(context, k))

    def propose_tree_steps(self, context: list[int], k: int):
        """The block for the round anchored at context[-1]: the backlog fed, the draft graph launched
        (the generator stops there so the loop streams the last block meanwhile), the cut."""
        t0 = time.perf_counter()
        if not self._feed(int(context[-1])) or k < 1:
            return None
        fn, st_m, D = self.fn, self.mtp.st, self.depth
        rows, nexts = self.mtp.pending
        self.mtp.pending = None
        n = len(nexts)
        cls = fn.qsa_class(st_m.length + n + D - 1)
        g = self.mtp.graphs.get((n, cls))
        if g is None:
            g = self.mtp.graphs[(n, cls)] = MTPChainGraph(fn, st_m, n, D, cls)
            g.capture()
        b = fn._mtp_bufs(D)
        b["stack"][:n].copy_(rows)
        fn.stage_mtp(st_m, nexts, D)
        g.graph.replay()
        host = time.perf_counter() - t0
        yield                               # the loop streams the last block while the draft runs
        t1 = time.perf_counter()
        torch.cuda.current_stream().synchronize()
        st_m.length += n
        toks, parents = self.cut.cut(int(context[-1]), b["tok_h"][:D].tolist(), b["prob_h"][:D].tolist())
        tree = DraftTree(tokens=toks, parents=parents).truncate(int(k) + 1)
        # the draft's own cost: its host work on both sides of the stop, not the streaming between
        # (StairCut priced the gap as drafting and stopped drafting, 2026-10-11)
        self.round = (host + time.perf_counter() - t1) * 1e3
        self.stats["rounds"] += 1
        self.stats["draft_ms"] += self.round
        return tree

    # --- internals -------------------------------------------------------------------------------
    def _feed(self, anchor: int) -> bool:
        """Every synced row paired with its next token (the last with the anchor); all but the last
        depth + 1 into the MTP cache now, those into the draft's own feed. False: nothing synced."""
        if self.held is not None:
            ht, hs = self.held
            self.queue.append((hs, ht[1:] + [anchor]))
            self.held = None
        if not self.queue:
            return self.mtp.pending is not None
        rows = torch.cat([q[0] for q in self.queue]) if len(self.queue) > 1 else self.queue[0][0]
        nexts = [t for q in self.queue for t in q[1]]
        self.queue = []
        n, keep = len(nexts), min(len(nexts), self.depth + 1)
        if n > keep:
            self.fn.mtp_prime(rows[:n - keep], nexts[:n - keep], self.mtp.st)
        self.mtp.pending = (rows[n - keep:], nexts[n - keep:])
        self.stats["fed_rows"] += n
        return True


#: the repository root (the hot draft vocabulary is ops/fn-hot-vocab.npy)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def build(pack: str, max_len: int, tok, spec: bool = True, depth: int = 4, mark=lambda stage: None, log=print):
    """Flash-Next for the server (`server/app.py` `_load_flashnext`): the pack's weights, the engine in
    the loop's interface, and with `spec` the MTP drafter over the hot draft head (`FN_DRAFT_HOT` ids of
    ops/fn-hot-vocab.npy, 0 = the whole vocabulary), its graphs captured -- gated by the load-time
    self-check (a failed check serves plain decode). `mark(stage)` at "weights", "engine", "drafters"
    (the server's load ledger). Returns (serving, drafter or None, info)."""
    from engine.models.flashnext.load import load as fn_load
    idx, W, banks, tables, timing = fn_load(pack, "cuda", mtp=spec)
    mark("weights")
    fn = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=max_len)
    eng = FlashNextServing(fn, max_len)
    mark("engine")
    drafter, check, hot = None, None, int(_S.get("FN_DRAFT_HOT"))
    if spec:
        if hot:
            fn.set_draft_head(np.load(os.path.join(ROOT, "ops", "fn-hot-vocab.npy"))[:hot])
        text = tok.apply_chat_template([{"role": "user", "content": "Count from one to twenty in words."}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        check = fn.self_check(tok(text, add_special_tokens=False)["input_ids"], log=log)
        if check["ok"]:
            drafter = FlashNextTreeDrafter(eng, depth=depth)
            drafter.capture_all()
        else:
            log("[server] Flash-Next self-check FAILED: speculation off, plain decode only")
    mark("drafters")
    info = {"pack": pack, "load": timing, "self_check": check,
            "draft_head": f"hot {hot}" if spec else None, "depth": depth if drafter is not None else 0}
    return eng, drafter, info
