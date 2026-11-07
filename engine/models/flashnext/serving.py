"""Flash-Next behind the server's own generation loop (SRV-87): an engine and a drafter with the shapes
`server/app.py` and `engine/cache.py` drive the 27B through.

The server's loop (`_generate_stream`) speaks a small interface: `forward` (a prompt chunk, a forced
phrase or one decode token at the state's length), `forward_tree` (a draft block; logits of every
row), `commit_tree` (keep a path), `hidden_post_norm` (the rows a drafter syncs from), `picks` (the
verify's own greedy tokens), `kv.length`, `reset`; the walk down a block is the shared one
(engine/methods/walk.py). The caches and the dashboard add the adapter's cold verbs (ENG-E8): `cost`,
`cache_state` (`FlashNextCacheState`: what the caches keep, ENG-157), `resume_at`, `settle`,
`discard_pending`, `trim_pending`. `FlashNextServing` maps it onto `FlashNextEngine`:

  forward, one token      the captured decode step (`DecodeGraph`), streams of that row
  forward, more tokens    the prefill kernels (`FlashNextEngine.prefill`), every row's streams
  forward_tree            `verify` from captured graphs: logits [M, vocab], the greedy picks, streams
  commit_tree             `commit` from captured graphs (the folded GDN commit included)
  forward_block           a chain verify whose commit waits for `rollback_to` or the next call

so everything above it -- penalties, the think budget and its forced close, pattern stops, logprobs,
sampled requests walked down the greedy tree (`--sampled-tree det`), tool calls, the live view -- is
the 27B's code. A verified row is the decode step's row bit for bit (ENG-227), so greedy speculation
writes the plain greedy text and a sampled walk draws exactly what plain sampled decoding draws.

The drafter is the shared MTP tree drafter (engine/methods/mtp_drafter.py) over Flash-Next's MTP head
(engine/models/flashnext/mtp.py, `draft_heads()["mtp"]`), cut by the shared StairCut whose prior is the
price file of this pack on this device (ops/prices/qwen3.8-flash-next/<profile>.<device>.json).
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from engine.methods import busy
from engine.methods.device import device_id
from engine.methods.manifest import Manifest
from engine.methods.mtp_drafter import MTPTreeDrafter
from engine.methods.staircut import PriceModel, StairCut
from engine.models.adapter import Capabilities, CostModel
from engine.settings import SETTINGS as _S
from engine.models.flashnext import coupled as CP
from engine.models.flashnext.engine import (CAPTURE_ROWS, PREFILL_CHUNK, QSA_TILE, CommitGraph, DecodeGraph, FlashNextEngine,
                                            VerifyGraph, _state_restore, _state_saved)
from engine.models.flashnext.mtp import FlashNextMTPHead
from engine.models.state import SectionState
from engine.settings import SETTINGS as _S


#: fingerprint v3 (`FlashNextServing.manifest`): the package's files that compute rows, the tables
MODULES = ("engine/models/flashnext/__init__.py", "engine/models/flashnext/config.py",
           "engine/models/flashnext/engine.py", "engine/models/flashnext/load.py",
           "engine/models/flashnext/mtp.py", "engine/models/flashnext/pack.py",
           "engine/models/flashnext/ple.py", "engine/models/flashnext/ref.py",
           "engine/models/flashnext/vision.py", "engine/models/flashnext/knobs.py",
           "engine/models/flashnext/joint_prefill.py", "engine/settings.py")
TABLES = ("ops/fn-exl3-tiles.json", "ops/fn-dense-splits.json")


def manifest() -> Manifest:
    """What Flash-Next's stored rows depend on (fingerprint v3, ENG-243): its package's arithmetic
    (not this adapter), the knob registry, its own tile tables; the `tools/` kernels follow."""
    return Manifest(arch="qwen4_exp", modules=MODULES, data=TABLES)


class _DraftReadWait:
    """SPD-249: the draft read's stream wait as a call of its own, for the runner's timer (`host_timers`)."""

    @staticmethod
    def wait() -> None:
        torch.cuda.current_stream().synchronize()


def _leave_capture_stream() -> None:
    """SPD-261: after a capture that raised, the engine thread on its own stream again. `torch.cuda.graph` ends the
    capture before it leaves the capture stream (torch 2.13), so a capture whose end raised (an invalidated one) left
    the thread on torch's capture stream, where every later launch of the thread would queue. The verify, commit and
    draft graphs are captured on that stream; the engine's threads run on the default stream."""
    cap = getattr(torch.cuda.graph, "default_capture_stream", None)
    if cap is not None and torch.cuda.current_stream() == cap:
        torch.cuda.set_stream(torch.cuda.default_stream())


def _after_failed_capture() -> None:
    """SPD-261: before a call runs again eagerly after its capture raised: the thread's own stream, and every launch of
    the capture's warm-ups landed (they were queued on a stream of their own)."""
    _leave_capture_stream()
    torch.cuda.synchronize()


class IdleCaptureFailed(RuntimeError):
    """SPD-271: an idle capture of class `cls` raised `exc`; the engine recorded the class at its state
    (`FlashNextServing._capture_failed`) and the idle looks skip it until that state changes."""

    def __init__(self, cls: int, exc: BaseException):
        self.cls, self.exc = int(cls), exc
        super().__init__(f"class {cls}: {type(exc).__name__}: {exc}")


class _MTPHead(FlashNextMTPHead):
    """SPD-261: Flash-Next's MTP head as `add_mtp_head` builds it: a chain capture that fails keeps no graph.

    `FlashNextMTPHead.launch` and `capture_all` key a new chain graph before they capture it, so a capture that raised
    (an allocation in the graph pool, a Triton launch the KV pool refused) left an uncaptured graph under its key and
    every later draft of that key replayed it. Here a call that raises drops the graphs it keyed. A draft whose capture
    failed then runs its chain eagerly: `mtp_chain_dev` from the same staging, the graph's own launches (the capture's
    warm-ups wrote only cache rows past the length and pool blocks the feed writes again). The round completes, and
    the next draft of that key captures it again. A draft only proposes tokens: the verify decides what is written.
    `capture_all` and `capture_class` raise as before. A dropped graph goes to `dead` (the adapter's `dead_graphs`):
    kept, never reset."""

    def __init__(self, *a, dead: list | None = None, **kw):
        self.dead = dead if dead is not None else []
        super().__init__(*a, **kw)

    def _drop(self, n0: int) -> list:
        made = list(self.graphs)[n0:]
        for k in made:
            self.dead.append(self.graphs.pop(k))
            self.fresh.discard(k)
        return made

    def capture_all(self) -> None:
        n0 = len(self.graphs)
        try:
            super().capture_all()
        except Exception:
            self._drop(n0)
            _leave_capture_stream()
            raise

    def capture_class(self, n: int, cls: int) -> None:
        n0 = len(self.graphs)
        try:
            super().capture_class(n, cls)
        except Exception:
            self._drop(n0)
            _leave_capture_stream()
            raise

    def launch(self, rows: torch.Tensor, nexts: list[int]) -> None:
        n0 = len(self.graphs)
        try:
            super().launch(rows, nexts)
            return
        except Exception as exc:
            made = self._drop(n0)
            if not made:
                raise
            _after_failed_capture()
            print(f"[fn-graphs] MTP chain capture {made[0]} failed ({type(exc).__name__}: {exc}): the draft runs "
                  f"eagerly", flush=True)
        fn, n = self.fn, len(nexts)
        cls = fn.qsa_class(self.st.length + n + self.depth - 1)
        kd = self.keyed if CP.ON else None
        fn._mtp_key, self._keyed = self.key, kd is not None
        fn._mtp_bufs(self.depth)["stack"][:n].copy_(rows)
        staged = (fn.stage_mtp(self.st, nexts, self.depth, keyed=kd) if kd is not None
                  else fn.stage_mtp(self.st, nexts, self.depth))
        assert staged == cls
        fn.mtp_chain_dev(self.st, n, self.depth, cls, keyed=kd is not None)
        self._khot = self._hot_of(1)
        self._n = n
        self.captures += 1                  # the draft's time holds the failed capture: the drafter skips the round


class _KV:
    """The `kv` the server reads: the length of the sequence the state holds (a pending chain commit
    lands first), its capacity. Setting it to zero starts a new sequence in place (`restart`: the rows
    stay, as the 27B's do); to any other length it keeps the KV rows by length and clamps the pooled
    keys -- the recurrent state is the caller's to restore first (`state.restore`), as on the 27B."""

    def __init__(self, serving: "FlashNextServing"):
        self._s = serving

    @property
    def pages(self):
        """The KV pool's paged set of the state's rows (the trunk's and the MTP head's), or None."""
        return self._s.st.pages

    @property
    def length(self) -> int:
        self._s._settle_chain()
        return self._s.st.length

    @length.setter
    def length(self, v: int) -> None:
        v = int(v)
        if v == 0:
            self._s.restart()
            return
        if not 0 < v <= self.max_len:
            raise ValueError(f"Flash-Next's state cannot hold {v} rows (capacity {self.max_len})")
        self._s.st.length = v
        self._s.st.clamp_pool()

    @property
    def max_len(self) -> int:
        return self._s.max_len


def _tensors(obj) -> list:
    """The tensors an object of the weight dict holds: itself, or its attributes (an EXL3 block, an
    int8 row set, an expert bank's kept images)."""
    if torch.is_tensor(obj):
        return [obj]
    out = []
    for v in vars(obj).values() if hasattr(obj, "__dict__") else ():
        if torch.is_tensor(v):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            out += [x for x in v if torch.is_tensor(x)]
    return out


class FlashNextCacheState(SectionState):
    """What the caches keep of a Flash-Next sequence (engine/models/state.py, ENG-157). Sections: the
    GDN states and conv tails, the PLE conv state and its token context (a fixed buffer, the end-of-
    text id where the sequence has none yet) and `mtp.h_last`, the last committed row's streams (the
    MTP layer's next input, which waits for its next token). Views: K, V and the indexer keys of every
    attention layer. The pooled indexer keys are derived and built again after a restore
    (`FlashNextServing.resume_at`)."""

    def __init__(self, serving: "FlashNextServing"):
        self.s = serving

    def sections(self) -> dict:
        s, st = self.s, self.s.st
        for li, buf in s._ctx.items():
            ctx = st.ple_ctx.get(li)
            if ctx is None:
                buf.fill_(s.cfg.eos_token_id)
            else:
                buf.copy_(ctx.view(1, -1))
        return self.buffers()

    def buffers(self) -> dict:
        """The sections' buffers as they are, without bringing the PLE context up to date: for what
        only counts their bytes (`cost`, `memory_parts`), which other threads call -- /health and the
        dashboard -- and which must issue no CUDA work: inside a graph capture (CUDA's global mode) a
        copy from another thread invalidates the capture and fails the request (2026-10-12, the MTP
        chain's capture at the 262,144-token class while /health was polled)."""
        s, st = self.s, self.s.st
        out = {}
        for li in sorted(st.S):
            out[f"gdn.S.{li}"] = st.S[li]
        for li in sorted(st.conv):
            out[f"gdn.conv.{li}"] = st.conv[li]
        for li in sorted(st.ple_conv):
            out[f"ple.conv.{li}"] = st.ple_conv[li]
        for li in sorted(s._ctx):
            out[f"ple.ctx.{li}"] = s._ctx[li]
        out["mtp.h_last"] = s._h_last
        return out

    def views(self) -> list:
        st = self.s.st
        out = []
        for li in sorted(st.k):
            out += [(f"k.{li}", st.k[li], 1), (f"v.{li}", st.v[li], 1)]
            if li in st.ik:
                out.append((f"ik.{li}", st.ik[li], 0))
        return out


#: The prefill workspace (the memory policy's `workspace`, engine/cache.py `Budget`): the line of 2026-10-12, a floor,
#: the bytes a token, the cap where it stops growing (`cost`)
WS_LINE_FLOOR, WS_LINE_ROW, WS_CAP = int(2.75 * (1 << 30)), 76_309, int(10.75 * (1 << 30))
#: SRV-773: the workspace by forwarded rows and context class (`_workspace_rows`), measured 2026-11-02 (`cost`): one
#: chunk of FN_PREFILL_CHUNK rows takes `A + B x T + D x T x class` for its T rows, every forwarded row its streams
#: 2026-11-06, after the indexer's scores and the PLE decode went into row tiles (engine.py `QSA_TILE`): raw requests
#: alone on the Spark, 20 rows from 6.7k to 255k of context (`FN_RESUMED_1106`, tests/test_budget.py). A tail of
#: 4,406 to 4,652 rows peaks at 2.18 to 2.65 GiB at every context from 112k to 255k (8.36 GiB at 138k before), one of
#: 1,094 to 1,156 rows at 0.95 to 1.51, 266 to 282 rows at 0.26 to 0.47, 14 rows at 0.10 to 0.13. The class's term
#: counts a tile's rows only: its scores are [tile, class] fp32 and each 1,024-row block's products [4 x 1,024, blocks]
WS_CHUNK_A = int(0.25 * (1 << 30))
WS_ROW_B = 400_000                     # a row: the chunk's activations, q/k/v, MoE buffers, slabs, the PLE rows
WS_ROW_CLASS_D = 24                    # a tile's row and a class block: the indexer's scores and products (19 measured)
WS_STACK_ROW = 20_480                  # a row's streams: `prompt_stack`, hc_count 4 x hidden 2,560 bf16 (the config)
WS_FLOOR_1, WS_ADD_1 = int(0.75 * (1 << 30)), int(0.5 * (1 << 30))    # a resumed chunk or less: its floor, its margin
WS_ADD_N = 1 << 30                     # more than a chunk: the margin (the streams' cat copy)
WS_MARGIN = 1.15


class FlashNextServing:
    """`FlashNextEngine` with one state and its graphs, in the server's engine interface."""

    # the caches' grid: the prefill kernels' chunk (FN_PREFILL_CHUNK), so a resumed prefill runs the
    # chunks a cold one runs; side slots: ENG-241; the shadow reader: validated on the 27B's text only
    # side_lane: Borrow keeps the 27B's pending commit, not this engine's chain and folded commit
    # (lending it raised `'FlashNextCacheState' object has no attribute 'primed'`, rc-0.8.0 live)
    # SPD-215: a lane's plain step joins the round's joint pass as a one-row block (`joint_plain`): the MTP
    # cut skips drafts often at C8, and each plain step alone held the other lanes' round
    caps = Capabilities(caches=True, side_slots=True, anchor_grid=4096, shadow=False, side_lane=False,
                        joint_plain=True)
    #: SPD-249: `forward_tree_multi` reads each block's tokens on the host (the runner may hand host tensors:
    #: QSE_RUNNER_HOST_BLOCKS)
    host_blocks = True
    #: SPD-249: the round runner sets it before a joint pass when a joint graph may be captured in it
    joint_capture_ok = False

    def __init__(self, fn: FlashNextEngine, max_len: int):
        self.fn, self.cfg, self.max_len = fn, fn.cfg, max_len
        self.st = fn.new_state(max_len, paged=True)
        c, dev = fn.cfg, fn.device
        self._ctx = {li: torch.full((1, fn.hash[li].context_len), c.eos_token_id, dtype=torch.long, device=dev)
                     for li in c.ple_layers}
        self._h_last = torch.zeros(1, c.hc_count * c.hidden_size, dtype=fn.dtype, device=dev)
        self.state = FlashNextCacheState(self)
        self.kv = _KV(self)
        self.dg = None                      # the decode step's graphs, made on the first decode token
        self.vgraphs: dict = {}
        self.cgraphs: dict = {}
        self._eager_block = False           # SPD-261: the last block verified eagerly, so its commit runs eagerly too
        # SPD-261: every verify, commit or draft graph a failed capture dropped, kept for the adapter's life and never
        # reset or cleared: a reset gives back torch's cuBLAS workspace of the graph's capture stream
        # (engine/capture_lock.py, SPD-223), which a kept graph captured on the same stream may read
        self.dead_graphs: list = []
        # verify width M / commit key (M, n, moved) -> rounds that used it, from a prior of one round
        # each for chains of 2-6 rows (a class's idle captures need shapes before the first rounds)
        self.uses: dict = dict(PRIOR_USES)
        self._fresh: set = set()            # graphs captured while idle and not replayed yet
        self._owed: dict = {}               # class -> a length a turn found it at without its graphs
        self._capture_failed: dict = {}     # SPD-271: class -> the state (`_state_key`) its idle capture failed at
        self.last_owed = None              # the last `capture_cost`'s class and missing graphs (the turn's line)
        self.capture_ms = None              # one verify or commit graph's capture, ms (EMA 0.2)
        self._width = 0                     # the last verify block's rows (its commit's key)
        self.picks = None
        self.tap = None
        self._hidden = None                 # [1, rows, hc H]: the last forward's streams
        self._chain = None                  # a forward_block's rows awaiting their commit
        self.commit_ms = 0.0
        self.captured = False               # the last block captured a verify or commit graph
        self.plain_captured = False         # the last plain step captured its decode graph
        self.heads: dict = {}               # draft heads by name (`draft_heads`)
        self._weights = None                # the weights' bytes by part (`memory_parts`), once
        self.stats = {"prefill_rows": 0, "decode_steps": 0, "verify_blocks": 0, "verify_rows": 0, "verify_eager": 0,
                      "idle_capture_failed": 0}

    # --- the loop's interface -------------------------------------------------------------------
    @property
    def hidden_post_norm(self) -> torch.Tensor:
        return self._hidden

    @property
    def mm(self):
        """VIS-97: the request's images (engine/vision.py `MMContext`), set by the server as on the 27B;
        kept by the state and the draft heads' states (`FNState.mm`), where the engine reads them."""
        return self.st.mm

    @mm.setter
    def mm(self, mm) -> None:
        for st in self._states():
            st.mm = mm

    @property
    def pos_delta(self) -> int:
        """The rotary offset of the rows after the prompt (position = slot + delta, VIS-97)."""
        return self.st.delta

    @pos_delta.setter
    def pos_delta(self, d: int) -> None:
        for st in self._states():
            st.delta = int(d)

    #: SPD-233: the rows a joint pass of 8 lanes keeps (QWEN38_FN_JOINT_ROWS8; 0: the cap, 32), the best
    #: lanes' nodes by score: served 2026-10-25, Flash-Next 3.05 at C8, 8 / 16 / 24 / 32 rows a pass:
    #: 63.1 / 73.4 / 75.3 / 72.2 tok/s
    JOINT_ROWS_AT = {8: int(_S.get("FN_JOINT_ROWS8") or 0)}

    def joint_policy(self) -> dict:
        """The joint pass of this family: 32 rows (the row-invariant widths of its kernels, `tree_batchable`)
        and, by lanes, the rows a pass keeps (`JOINT_ROWS_AT`, a class's measured table)."""
        return {"max_rows": 32, "rows_at": {e: n for e, n in self.JOINT_ROWS_AT.items() if n}}

    def price_class(self, n: int) -> int:
        """The draft prices' context class of a round at context length `n` (engine/methods/staircut.py):
        0 while the sparse attention is dense, else the next of 4,096 x 2^k and 6,144 x 2^k at or above
        n, up to 262,144. A verify's score rows grow with the context itself, not only with the sparse
        attention's class: on the P0 runs a block cost about 41 ms at 140,000 tokens and 53 at 260,800,
        both in the 262,144 class."""
        if not self.fn.qsa_class(n):
            return 0
        c = 4096
        while c < n and c < 262_144:
            c = c * 3 // 2 if (c & (c - 1)) == 0 else c * 4 // 3
        return min(c, 262_144)

    def cost(self) -> CostModel:
        """What a state snapshot holds: its sections whole (the recurrent, conv and PLE ones) and the
        cache rows a token adds (K, V and the indexer key of every attention layer, bf16)."""
        sec = {k: t.numel() * t.element_size() for k, t in self.state.buffers().items()}
        c = self.cfg
        per_tok = len(c.attention_layers) * (c.num_key_value_heads * c.head_dim * 2
                                             + (c.indexer_head_dim if c.has_indexer else 0)) * 2
        # the prefill workspace, measured 2026-10-12 on 3.05: 2.51 GiB at 8k and 2.40 GiB at 32k
        # (tools/fn_memory_check.py; the 4,096-row chunk's activations), 10.46 GiB above the engine at a
        # 147,149-token prefill and 10.47 GiB at 261,549 (SPD-99): it grows to about 140k tokens and then
        # stops -- a floor of 2.75 GiB, the line through the 147,149-token point (76,309 bytes a token),
        # capped at 10.75 GiB (the memory policy raises the cap if a prefill ever needs more).
        # SRV-773, measured 2026-11-02 on 3.05 (the Spark, tree 5e4bb7c6 with the request-peak build, raw requests
        # alone, 35 rows from 8k to 146k): the line is the whole context's, while a prefill runs in chunks of
        # FN_PREFILL_CHUNK rows (4,096) and nothing but the prompt's streams grows with its rows past one chunk. A
        # resumed turn of 21 rows took 0.10 GiB at every context (the line: 2.75 to 10.39 GiB). The chunk's
        # `a + b x T + d x T x class` fitted on 26 continuation rows: a 0.180 GiB, b 170,274 B, d 29.9 B (max |resid|
        # 0.68 GiB, rms 0.23); `WS_*` round them up, the streams' 20,480 B a row are the config's. Charged against
        # the measured peak (the prefill's and the request end's), 28 of 28 continuations are covered (+0.38 GiB
        # the least) and 7 of 7 cold or seeded prefills against their workspace and streams. A cold forward under
        # 4,151 rows and the streams' life past the first draft round were not measured (`_workspace_rows`)
        return CostModel(section_bytes=sum(sec.values()), view_bytes_per_token=per_tok,
                         workspace_floor=WS_LINE_FLOOR, workspace_per_token=WS_LINE_ROW, workspace_cap=WS_CAP,
                         workspace_rows=self._workspace_rows,
                         parts={"recurrent": sum(b for k, b in sec.items() if k.startswith("gdn.S")),
                                "conv": sum(b for k, b in sec.items() if k.startswith(("gdn.conv", "ple.conv")))})

    def _workspace_rows(self, n_ctx: int, rows: int, line: bool = True) -> int:
        """SRV-773 (`CostModel.workspace_rows`): the workspace of forwarding `rows` rows onto `n_ctx` served rows, at
        the class of the last served row. A resumed prefill of one chunk or less: its chunk and its rows' streams
        with a margin, at least 0.75 GiB. More than a chunk (a re-read), and every cold prompt whatever its rows: a
        whole chunk and every row's streams with a margin, never under the line (a cold forward under 4,151 rows is
        not measured: the line's 2.75 GiB floor stays its charge). Both at most the cap.

        SRV-775: `line` False, the admission's count of a prefill in flight: by its rows alone, a cold prompt too,
        without the line's floor (2026-11-02: lone cold prompts of 205-215 rows peaked at 0.20-0.67 GiB, charged 2.75;
        eight such prompts counted 22 GiB against the box's floor and the eighth waited 12 s)."""
        n = int(n_ctx)
        r = max(1, min(int(rows), n))
        cls = int(self.fn.qsa_class(n - 1))

        def chunk(t: int) -> int:
            t = min(int(t), PREFILL_CHUNK)
            return WS_CHUNK_A + WS_ROW_B * t + WS_ROW_CLASS_D * min(t, QSA_TILE) * cls
        if r <= PREFILL_CHUNK and (r < n or not line):
            ws = max(WS_FLOOR_1, int(WS_MARGIN * (chunk(r) + WS_STACK_ROW * r)) + WS_ADD_1)
        else:
            ws = int(WS_MARGIN * (chunk(r) + WS_STACK_ROW * r)) + WS_ADD_N
            if line:
                # a cold prompt: the whole context's line. A resumed forward of more than a chunk: the line of its
                # rows (2026-11-06: 22,262 rows at 134k took 2.56 GiB, 61,050 at 201k 3.77, 41,417 at 249k 3.46,
                # each charged the context's 10.75)
                ws = max(min(max(WS_LINE_FLOOR, WS_LINE_ROW * (n if r == n else r)), WS_CAP), ws)
        return min(ws, WS_CAP)

    def manifest(self) -> Manifest:
        return manifest()

    def precapture(self, drafter) -> int | None:
        """After the server's warm-up: the warm-up's rows are nobody's conversation, so the state starts
        over and class 0's graphs (the shapes the rounds use, from the prior and the warm-up) are
        captured now, not in the first request (2026-10-12: the first short request after a restart
        drafted at 38.5 tok/s while it captured them, 58 after). Returns the graphs captured."""
        if drafter is None:
            return None
        self.kv.length = 0
        steps = self.idle_captures(ahead=False)       # the classes ahead are the idle windows' (SRV-713)
        return sum(1 for _ in steps) if steps is not None else 0

    def host_timers(self, timed) -> None:
        """SPD-249 (the round runner's QSE_RUNNER_COUNTERS): this engine's host calls under the runner's timer
        (`timed(obj, attr, name)`): the MTP staging, the draft read's wait apart from the read, the joint pass's PLE
        ids (the hashing) and rows (the table's read), the joint pass's issue and the commits' issue. The same calls
        in the same order (the read's wait is the stream sync the read starts with), so nothing computed changes."""
        fn = self.fn
        timed(fn, "stage_mtp", "stage_mtp")
        timed(fn, "path_ple_ids", "ple_hash")
        timed(fn, "_ple_batch", "ple_batch")
        timed(fn, "verify_multi", "verify_issue")
        timed(fn, "commit_multi", "commit_issue")
        for tab in {id(t): t for t in (getattr(fn, "tables", None) or {}).values()}.values():
            timed(tab, "rows_of", "ple_rows")
        for h in self.heads.values():
            if getattr(h.read, "_hc", False):
                continue
            w = _DraftReadWait()
            timed(w, "wait", "read_wait")

            def read(_orig=h.read, _w=w):
                _w.wait()
                return _orig()
            h.read = read
            timed(h, "read", "read")

    def joint_stats(self) -> dict | None:
        """SPD-249: the joint passes' measured shape for /health and the runner's line: the distinct routed experts
        a layer (`union.ExpertUnion`), None when not counted."""
        un = getattr(self.fn, "union", None)
        rep = un.report() if un is not None else None
        out = {"experts_union": rep} if rep is not None else {}
        from engine.models.flashnext import engine as E
        if E.JOINT_GRAPH:
            out["joint_graphs"] = self.fn.joint_graph_stats()          # SPD-249
        return out or None

    def draft_heads(self) -> dict:
        """The draft heads this pack carries, by name: its MTP layer ("mtp", made by `add_mtp_head`)."""
        return self.heads

    def side_slot(self, max_len: int, drafter0=None) -> tuple:
        """ENG-241 step 2: a side slot's engine and drafter over this engine's weights (`fn`, the one
        copy): its own state (paged as slot 0's when the pool is on), its own graphs and MTP head; the
        drafter's StairCut from the same price file, its lookup over slot 0's corpus and suffix store
        (read-only). The slots share the engine's scratch buffers: a slot's work runs on its GPU turn,
        a call on one state applies another's parked GDN rows first (`FlashNextEngine.settle`), each
        MTP head drafts into its own buffers, and decoding slots verify their blocks in one pass
        (`forward_tree_multi`)."""
        e2 = type(self)(self.fn, max_len)      # SPD-217: a subclass's slot is its own class (Ornith's)
        d2 = None
        if drafter0 is not None and hasattr(drafter0, "cut"):
            ng = None
            if drafter0.ngram is not None:
                from engine.drafters.ngram import NgramDrafter
                ng = NgramDrafter(corpus_path="", min_order=3, max_depth=16, min_corpus_order=8)
                ng.corpus = drafter0.ngram.corpus
            c0 = drafter0.cut
            d2 = MTPTreeDrafter(e2, e2.add_mtp_head(drafter0.depth),
                                StairCut(PriceModel(c0.prices.table, c0.prices.step), siblings=c0.siblings), ng,
                                freeze=getattr(drafter0, "freeze", False))
            d2.mtp_look_backoff = getattr(drafter0, "mtp_look_backoff", 0)      # SPD-249
            d2.capture_all()
        return e2, d2

    def side_slot_bytes(self, max_len: int, paged: bool) -> tuple:
        """(bytes a token, constant bytes) a side slot takes before it serves anything: its KV and the
        MTP head's when the window is not paged; the recurrent sections, and the graphs' pools a slot
        captures (SIDE_SLOT_CONST, measured)."""
        cost = self.cost()
        per_token = 0 if paged else cost.view_bytes_per_token * 13 // 12      # the MTP layer's 13th
        return per_token, cost.section_bytes + SIDE_SLOT_CONST

    def add_mtp_head(self, depth: int) -> FlashNextMTPHead:
        self.heads["mtp"] = _MTPHead(self.fn, depth, self.max_len, pages=self.st.pages,     # SPD-261
                                     dead=self.dead_graphs)
        # the head's graphs are captured now (`capture_all`): with the KV pool its first rows must be
        # mapped -- a Triton launch refuses a buffer whose first page is not ("cannot be accessed from
        # Triton", 2026-10-12)
        self._ensure(CAPTURE_ROWS)
        return self.heads["mtp"]

    def settle(self) -> None:
        """Land what is pending, so the state's sections are the sequence's whole (before a
        snapshot): a chain block's commit, and the GDN rows the folded commit parked for the next call."""
        self._settle_chain()
        if self.fn.parked(self.st):
            self.fn.flush(self.st)

    def cache_state(self) -> FlashNextCacheState:
        return self.state

    def resume_at(self, n: int) -> None:
        """The engine continues at length `n` from the state the caller just wrote into it: the PLE
        context from its section, the length, the pooled indexer keys built from the keys in place."""
        self.discard_pending()
        for li, buf in self._ctx.items():
            self.st.ple_ctx[li] = buf.view(1, -1).cpu().clone()
        self.kv.length = n
        if n > 0:
            self.fn.build_pool(self.st)

    def memory_parts(self) -> dict:
        """The dashboard's memory split (server/memsplit.py): the trunk's and the MTP layer's weights
        (each storage once), the trunk's KV, indexer keys and pooled keys, the drafter's MTP KV, the
        recurrent sections."""
        if self._weights is None:
            seen, by = set(), {"weights": 0, "mtp_weights": 0}
            items = [(k, v) for k, v in self.fn.W.items()] + [(f"bank.{li}", b) for li, b in self.fn.banks.items()]
            for name, obj in items:
                part = "mtp_weights" if name.startswith("mtp.") or name == "bank.-1" else "weights"
                for t in _tensors(obj):
                    s = t.untyped_storage()
                    if s.data_ptr() not in seen:
                        seen.add(s.data_ptr())
                        by[part] += s.nbytes()
            self._weights = by
        st = self.st

        from engine import kvpool as _kvp

        def nb(*ds):
            # a paged buffer's `nbytes` is its reserved address range: its committed pages are
            # the pool's figure (server/memsplit.py adds them as `kv_pool`)
            return sum(int(t.numel() * t.element_size()) for d in ds for t in d.values()
                       if not _kvp.paged(t))
        head = self.heads.get("mtp")
        mtp_kv = 0
        if head is not None:
            mtp_kv = nb(head.st.k, head.st.v, head.st.ik, head.st.pool)
        return {**self._weights, "kv_pool": nb(st.k, st.v, st.ik, st.pool), "drafter_kv": mtp_kv,
                "gdn_state": sum(int(t.numel() * t.element_size()) for t in self.state.buffers().values())}

    def resident_files(self) -> list:
        """The n-gram table: its policy, its file and page-cache bytes, the cold cache's counts."""
        return [t.report() for t in {id(t): t for t in self.fn.tables.values()}.values()]

    def last_streams(self) -> torch.Tensor:
        """The last committed row's streams [1, hc H] (`mtp.h_last`)."""
        return self._h_last

    def discard_pending(self) -> None:
        """Forget what is pending after the state was overwritten (a restore): the chain block, the
        last verify's picks and the GDN rows parked for the state that was there."""
        self._chain = None
        self.picks = None
        self.fn.drop_parked(self.st)

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
        """A new sequence (`restart`), as the 27B's reset: its KV rows stay."""
        self.restart()

    def restart(self) -> None:
        """A new sequence in the same buffers: the recurrent sections cleared and the length 0, the KV,
        indexer-key and pooled-key rows left in place (nothing reads a row past the length). The
        resident prefix keeps a conversation's rows there while a short request runs from row 0 (a
        guest, engine/cache.py, whose cold prefill calls `reset`); zeroing every row there lost the
        rows above the guest's stash (ENG-157, 2026-10-12: rows 2,144-4,095 of a 40,000-token
        conversation came back as zeros, and a 6,000-token prompt resumed at 4,096 over them answered
        differently from a cold prefill, 67 of 75 sections wrong)."""
        self._chain = None
        self.picks = None
        self.fn.drop_parked(self.st)
        self.st.reset_inplace()
        # SPD-271: a head's length goes with the trunk's: a KV reclaim left a head at 85,010 over a trunk at 0 and
        # every idle look planned a class below the head's (ClassError, 2026-11-02)
        for h in self.heads.values():
            h.reset()
        self._capture_failed.clear()

    # --- a context class's graphs, captured while the server is idle -----------------------------
    #: the shapes a class captures ahead: the verify widths and commit shapes that make up this share
    #: of the rounds so far, at most this many of each
    IDLE_SHARE, IDLE_MAX = 0.99, 24

    def _capture_took(self, t0: float) -> None:
        ms = (time.perf_counter() - t0) * 1e3
        self.capture_ms = ms if self.capture_ms is None else 0.8 * self.capture_ms + 0.2 * ms

    def _top(self, commits: bool) -> list:
        keys = sorted(((k, n) for k, n in self.uses.items() if isinstance(k, tuple) == commits),
                      key=lambda kn: -kn[1])
        total, out, got = sum(n for _, n in keys), [], 0
        for k, n in keys:
            if got >= self.IDLE_SHARE * total or len(out) >= self.IDLE_MAX:
                break
            out.append(k)
            got += n
        return out

    def _missing(self, cls: int) -> list:
        """What the class lacks of the shapes the rounds use (`_top`): ("v", M) verify widths,
        ("c", (M, n, moved)) commits of a captured or missing width, ("m", n) the MTP head's feeds."""
        widths = self._top(False)
        out = [("v", M) for M in widths if (M, cls) not in self.vgraphs]
        out += [("c", k) for k in self._top(True) if k[0] in widths and k + (cls,) not in self.cgraphs
                and (not k[2] or 2 <= k[1] < k[0])]
        for head in self.heads.values():
            out += [("m", n) for n in range(1, head.depth + 2) if (n, cls) not in head.graphs]
        return out

    def capture_cost(self, n: int) -> float:
        """What a drafted turn at context length `n` would spend capturing the graphs its class lacks,
        ms: the first drafted request in each class paid 3.4-5.1 s of captures (P0, 2026-10-12)
        until the idle captures below took them (`StairCut.turn` weighs it with the backlog). A class
        that lacks graphs here is owed them: the next idle window captures it whatever length the state
        is left at (a conductor box leaves it at a side question's).

        SPD-249: commit graphs are not charged. The idle window leaves them to their first use
        (`idle_captures`), where a commit captures the one shape it takes."""
        cls = self.fn.qsa_class(n)
        miss = self._missing(cls)
        charged = [m for m in miss if m[0] != "c"]
        if charged:
            self._owed.setdefault(cls, int(n))
        # what the turn's line names: the class and its missing verify, commit and draft graphs
        self.last_owed = f"{cls}:" + "".join(f"{k}{sum(1 for m in miss if m[0] == k)}" for k in "vcm")
        return len(charged) * (self.capture_ms if self.capture_ms is not None else CAPTURE_MS)

    def _next_class_len(self, n: int) -> int | None:
        """The shortest context past `n` in another class (`qsa_class` grows with the length), or None
        when no context under `max_len` is."""
        q, cls, top = self.fn.qsa_class, self.fn.qsa_class(n), int(self.max_len) - 1
        lo, hi = int(n), max(int(n) + 1, 1)
        while hi < top and q(hi) == cls:
            lo, hi = hi, min(top, hi * 2)
        if hi <= lo or q(hi) == cls:
            return None
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if q(mid) == cls else (lo, mid)
        return hi

    #: SRV-713: the span past the state's length whose classes an idle window captures ahead: a request
    #: through the conductor starts 4.1-4.8k tokens long (its plan's prompt, 2026-10-26), two classes past an
    #: empty state's (1,024 from 2,051 tokens, 2,048 from 4,096)
    AHEAD_TOKENS = 6144

    def _capture_lengths(self, ahead: bool) -> list:
        """The lengths whose classes the idle captures serve, in order: the state's own; with `ahead`,
        those turns found without graphs (`capture_cost`), then the next class past the state's length
        (the one a conversation grows into) and every further one that starts within `AHEAD_TOKENS` of
        it. SRV-713: every request through the conductor starts in the 2,048 class (its plan's prompt
        is 4.1-4.8k tokens), the warm-up captures class 0 only and the class of a short state's length
        is class 0, so each such turn took the plain step: 39.5 tok/s on code against 81.4 raw."""
        n = int(self.st.length)
        out = [n]
        if ahead:
            out += sorted(self._owed.values())
            x, first = self._next_class_len(n), True
            while x is not None and (first or x <= n + self.AHEAD_TOKENS):
                out.append(x)
                x, first = self._next_class_len(x), False
        return out

    def idle_captures(self, ahead: bool = True, wants=None):
        """The graphs the class of the state's length lacks, one capture a step (a generator: the
        server stops between two when a request waits), or None when it lacks none. An empty state
        (a side slot after its warm-up) captures class 0's: its first drafted turn would pay them, so
        it would never draft (2026-10-12: 3 requests at once, the side slots drafted nothing). Every
        capture leaves the state as it was: a verify writes only rows past the length, a commit's and the
        MTP chain's warm-ups are undone (engine.py).

        With `ahead`, then the classes of `_capture_lengths` past the state's own, each where `wants(n)`
        (the drafter's `wants_graphs`) says drafting can pay. A class's graphs fit any length up to the class's
        own: the rows' positions, the visible keys and the dense flag of each row are the replay's inputs (SPD-249:
        never a class below the state's, whose sparse rows would read `sel` no launch wrote). A capture's
        rows read no position the state has not committed: a dense row (a short state's) reads up to
        its own, a sparse one only complete blocks (the scores past them are -inf).

        SPD-249: with `ahead` (the server's idle window) no commit graphs: the verify and draft graphs only. Each
        lane's commit captures there (24 shapes a class, 8 lanes) slowed every later joint round of the process
        (Flash-Next 3.05, 8 lanes of 189-token prompts after one idle window, and 8 lanes of 2k prompts with the
        commit graphs of the lanes' own class).
        A commit captures its own shape at first use (`commit`); the warm-up's call (`precapture`, without
        `ahead`) still takes class 0's."""
        todo, seen = [], set()
        # SPD-249: never a class below the state's own (its rows' positions from the length on): an owed turn
        # that started under it and grew past it keeps its entry for a shorter state (`class_check`).
        # SPD-271: nor below a head's (a chain's last row is the head's length + its feed + depth - 1, the feed
        # at most depth + 1 rows)
        floor = max([self.fn.qsa_class(int(self.st.length) + CAPTURE_ROWS)]
                    + [self.fn.qsa_class(int(h.st.length) + 2 * h.depth) for h in self.heads.values()])
        key = self._state_key()
        for i, n in enumerate(self._capture_lengths(ahead)):
            cls = self.fn.qsa_class(n)
            if cls in seen:
                continue
            seen.add(cls)
            if cls < floor:
                continue
            if self._capture_failed.get(cls) == key:
                continue                    # SPD-271: failed at this state; retried once the state changes
            if i and wants is not None and not wants(n):
                continue
            miss = self._missing(cls)
            if ahead:
                miss = [m for m in miss if m[0] != "c"]
            if miss:
                todo.append((cls, miss))
            else:
                self._owed.pop(cls, None)
        return self._capture_classes(todo) if todo else None

    def _state_key(self) -> tuple:
        """SPD-271: the state an idle capture is planned on: the trunk's length and each head's."""
        return (int(self.st.length), tuple(int(h.st.length) for h in self.heads.values()))

    def _capture_classes(self, todo: list):
        key = self._state_key()
        for cls, miss in todo:
            try:
                yield from self._capture_steps(cls, miss)
            except Exception as exc:
                # SPD-271: recorded per class at this state; the next looks skip it until the state changes
                self._capture_failed[cls] = key
                self.stats["idle_capture_failed"] += 1
                raise IdleCaptureFailed(cls, exc) from exc
            self._owed.pop(cls, None)

    @torch.no_grad()
    def _capture_steps(self, cls: int, miss: list):
        self._settle_chain()
        fn, st = self.fn, self.st
        self._ensure(st.length + CAPTURE_ROWS)
        for kind, key in sorted(miss, key=lambda m: "vcm".index(m[0])):     # widths before their commits
            t0 = time.perf_counter()
            if kind == "v":
                g = VerifyGraph(fn, st, key, cls)
                self._capture_undone(g)
                self.vgraphs[(key, cls)] = g
                self._fresh.add((key, cls))
            elif kind == "c":
                M, n, moved = key
                vg = self.vgraphs.get((M, cls))
                if vg is None:
                    continue
                parents, path = list(range(-1, M - 1)), list(range(n))
                if moved:
                    # the path leaves the chain at its last row: row M - 1, a second child of row n - 2
                    parents[M - 1], path[-1] = n - 2, M - 1
                fn.stage_rows([0] * M, parents, st)
                fn._vcls = cls
                vg.replay()
                fn.stage_commit(st, path)
                g = CommitGraph(fn, st, (M, n, moved, cls))
                self._capture_undone(g)
                self.cgraphs[(M, n, moved, cls)] = g
                self._fresh.add((M, n, moved, cls))
            else:
                for head in self.heads.values():
                    head.capture_class(key, cls)
            torch.cuda.synchronize()
            self._capture_took(t0)
            yield

    def _state_undo(self):
        """SPD-261: before a capture, what its warm-ups may write (`_state_saved`: a verify's warm-up folds the parked
        GDN rows into each layer it reaches, a commit's applies the commit) and the parked rows' count, saved; returns
        the call that puts them back. `VerifyGraph` and `CommitGraph` leave the state consistent only after a capture
        that returns."""
        fn, st = self.fn, self.st
        saved, cnt = _state_saved(fn, st), fn._pend_of(st)[1].clone()

        def undo() -> None:
            _state_restore(saved)
            fn._pend_of(st)[1].copy_(cnt)
        return undo

    def _capture_undone(self, g) -> None:
        """SPD-261: an idle capture whose failure leaves the slot's state as it was (the caller keys the graph only
        after `g.capture()` returns); a graph whose capture failed goes to `dead_graphs`."""
        undo = self._state_undo()
        try:
            g.capture()
        except Exception:
            self.dead_graphs.append(g)
            _after_failed_capture()
            undo()
            raise

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, start: int, last_only: bool = True) -> torch.Tensor:
        """Rows `ids` at positions start..: one token through the decode step, more through the prefill
        kernels. Logits [1, 1, vocab] of the last row (fp32)."""
        self._settle_chain()
        ids = ids.view(-1)
        self._at(start)
        T = ids.numel()
        self._ensure(start + T)
        if T == 1 and start > 0:
            if self.dg is None:
                self.dg = DecodeGraph(self.fn, self.st)
            cls = self.fn.qsa_class(start)
            n0 = self.dg.captures
            lg = self.dg(int(ids[0]))
            self.plain_captured = self.dg.captures != n0       # this step's time is a capture's
            self._hidden = self.dg.stacks[cls].view(1, 1, -1)
            self._h_last.copy_(self._hidden[0, -1:])
            self.stats["decode_steps"] += 1
            return lg.view(1, 1, -1)
        lg = self.fn.prefill(ids, self.st, keep_stack=True)
        self._hidden = self.fn.prompt_stack.unsqueeze(0)
        self._h_last.copy_(self._hidden[0, T - 1:T])
        self.stats["prefill_rows"] += T
        return lg.view(1, 1, -1)

    def prefill_joint_ok(self, T: int, start: int) -> bool:
        """SPD-249 (QWEN38_FN_JOINT_PREFILL): whether a prefill burst's joint forward (`prefill_joint`) takes this
        slot's prompt of `T` rows from `start`: its `forward` would run them as one chunk on the prefill kernels (not a
        decode step, no images). Off: never (each prompt prefills alone). A family built on this adapter (Ornith's
        Qwen35MoeServing) runs its own trunk, which this joint forward does not hold: never."""
        if type(self) is not FlashNextServing:
            return False
        from engine.models.flashnext import joint_prefill as JP
        return JP.ON and JP.ok(self.fn, T, start, self.st)

    @torch.no_grad()
    def prefill_joint_prepare(self, T: int, start: int) -> None:
        """SPD-249: this slot's own part of its prefill before it joins a burst's joint forward, the steps its
        `forward(tokens, start)` takes first, in its order: its open chain committed, the state at `start`, the pages of
        rows [0, start + T) committed (`_ensure`). The server's burst hook runs it on the member's own thread before
        `Burst.join`, so what fails here (the KV pool full, a start off the state's length) fails this member alone, as
        its own `forward` would, and it stays out of the joint forward; the leader's thread runs only the forward.
        Again on a prepared slot (`prefill_joint` runs it for a caller that did not): no work (no open chain, one compare
        each)."""
        self._settle_chain()
        self._at(start)
        self._ensure(int(start) + int(T))

    @staticmethod
    @torch.no_grad()
    def prefill_joint(items: list) -> list:
        """SPD-249: [(serving, tokens, start)] of a prefill burst's slots over one engine, each `prefill_joint_ok`
        and prepared (`prefill_joint_prepare`, on its own thread): their prompts in one forward
        (engine/models/flashnext/joint_prefill.py: each layer's experts once for every prompt's rows, each prompt's rows
        its own prefill's bits). Each slot is left as its `forward(tokens, start)` leaves it; returns each one's logits
        [1, 1, vocab] (fp32), in order. A failure inside the forward fails every slot of it: their states are left
        part-way (the next prefill on each slot starts from a reset or an anchor, engine/cache.py)."""
        from engine.models.flashnext import joint_prefill as JP
        fn = items[0][0].fn
        assert all(s.fn is fn for s, _, _ in items), "a joint prefill over one engine's slots"
        jobs = []
        for s, toks, start in items:
            s.prefill_joint_prepare(int(toks.numel()), int(start))
            jobs.append((toks, s.st))
        outs = JP.prefill_joint(fn, jobs)
        res = []
        for (s, toks, start), (lg, stack) in zip(items, outs):
            T = int(toks.numel())
            s._hidden = stack.unsqueeze(0)
            s._h_last.copy_(stack[T - 1:T])
            s.stats["prefill_rows"] += T
            res.append(lg.view(1, 1, -1))
        return res

    @torch.no_grad()
    def forward_tree(self, block: torch.Tensor, parents: list[int], start: int) -> torch.Tensor:
        """A draft block in DFS pre-order (row 0 the anchor at `start`): every row's logits [M, vocab]
        fp32 (a static buffer), its greedy pick in `picks`, its streams in `hidden_post_norm`. Nothing
        but the block's own KV rows is written until `commit_tree`."""
        self._settle_chain()
        self._at(start)
        toks = [int(t) for t in block.view(-1).tolist()]
        M = len(toks)
        self._ensure(start + M)
        t0 = time.perf_counter()
        how = self._verify(toks, [int(p) for p in parents])
        # SPD-261: a round that captured its graph, or tried to, or verified eagerly is not a price of its width
        self.captured = how != "graph"
        self.uses[M] = self.uses.get(M, 0) + 1
        self._width = M
        cls = getattr(self.fn, "_vcls", 0)
        if (M, cls) in self._fresh:
            # an idle capture's first replay uploads the graph: not a price either
            self._fresh.discard((M, cls))
            self.captured, how = True, "capture"
        if how == "capture":
            self._capture_took(t0)
        rb = self.fn._rb[M]
        self.picks = rb["next"][:M]
        self._hidden = self.fn.rec["stack"][:M].unsqueeze(0)
        self.stats["verify_blocks"] += 1
        self.stats["verify_rows"] += M
        return rb["logits"][:M]

    def _verify(self, toks: list[int], parents: list[int]) -> str:
        """SPD-261: `verify` of a block at the state's length, from its graph or eagerly. Returns "graph" (a kept graph
        replayed), "capture" (the round captured the block's graph, or tried to) or "eager" (a block under a class
        edge). After an eager verify the block's commit is eager too (`commit_tree`: a commit graph reads the verify
        graph's own intermediates).

        `verify` keys a block's graph by M and the class of its deepest row, and captures a missing one over a chain of
        M rows from the length. With the indexer, a chain whose last row is in a higher class than the block's deepest
        row fails its own warm-up's `class_check` (a shallow tree just under a class edge: rows to 4,095 are class
        1,024, the chain's to 4,097 class 2,048). Such a block verifies eagerly, at its own class with the same
        kernels; the class's graph comes from an idle window or a round whose chain fits. A capture that fails keeps
        no graph and leaves the state as it was; the block then verifies eagerly (`_graphs_call`)."""
        fn, st, M = self.fn, self.st, len(toks)
        q, L = fn.qsa_class, int(st.length)
        top = cls = q(L + M - 1)            # the class of the chain a capture stages
        if top != q(L):
            depth = [0] * M
            for i, p in enumerate(parents):
                depth[i] = 0 if p < 0 else depth[p] + 1
            cls = q(L + max(depth))         # `verify`'s class
        capture = (M, cls) not in self.vgraphs
        if capture and cls != top:
            print(f"[fn-graphs] {M}-row block at {L}: no graph of class {cls}, and a capture's chain would end in class "
                  f"{top}: verified eagerly", flush=True)
            kept = False
            fn.verify(toks, parents, st, None)
        else:
            kept = self._graphs_call("verify", lambda g: fn.verify(toks, parents, st, g), self.vgraphs, capture)[1]
        self._eager_block = not kept
        if not kept:
            self.stats["verify_eager"] += 1
        return "graph" if not capture else "capture" if cls == top else "eager"

    def _graphs_call(self, what: str, call, graphs: dict, capture: bool) -> tuple:
        """SPD-261: `call(graphs)`, a `verify` or `commit` from captured graphs; (its result, True), or (the eager call's
        result, False) after a failed capture. `verify` and `commit` key a new graph before they capture it, so a
        capture that raised left a graph that was never captured, and the next rounds of that key replayed it
        (`graph` None). Here the graphs a failed call keyed are dropped. With `capture` (the call's key is missing)
        the state its warm-ups may write is saved first (`_state_undo`); a failed call then puts it back and runs
        eagerly (`call(None)`), so the round completes and the next one captures the key again. A dropped graph goes
        to `dead_graphs` and is never reset: a reset gives back torch's cuBLAS workspace of its capture stream
        (engine/capture_lock.py, SPD-223), which a kept graph captured on the same stream may read."""
        n0 = len(graphs)
        undo = self._state_undo() if capture else None
        try:
            return call(graphs), True
        except Exception as exc:
            made = list(graphs)[n0:]
            for k in made:
                self.dead_graphs.append(graphs.pop(k))
            if not made or undo is None:
                raise
            _after_failed_capture()
            undo()
            print(f"[fn-graphs] {what} capture {made[0]} failed ({type(exc).__name__}: {exc}): eager instead",
                  flush=True)
        return call(None), False

    def tree_batchable(self, parents) -> bool:
        """A block a joint pass may take (`forward_tree_multi`): its rows within the widths whose
        projections and experts give a row its lone bits (the self-check's, up to 32)."""
        return len(parents) <= 32

    @staticmethod
    def prefetch_rows(items: list) -> None:
        """SPD-249: [(serving, tokens, parents)] of trees whose tokens the round runner knows before it stages them:
        their PLE rows read early into the table's cache (`FlashNextEngine.ple_prefetch`, QWEN38_FN_PLE_PREFETCH)."""
        by: dict = {}
        for s, tokens, parents in items:
            by.setdefault(id(s.fn), (s.fn, []))[1].append(([int(t) for t in tokens], [int(p) for p in parents], s.st))
        for fn, jobs in by.values():
            fn.ple_prefetch(jobs)

    @staticmethod
    @torch.no_grad()
    def forward_tree_multi(items: list) -> list:
        """ENG-241 step 3: [(serving, block, parents, start)] of sequence slots over one engine, verified
        in one pass (`FlashNextEngine.verify_multi`: the weights read once); each slot's logits, picks
        and streams as its `forward_tree` leaves them, and its block committed by its `commit_tree`."""
        fn = items[0][0].fn
        # SPD-249: a joint graph may be captured in this pass only when the server's round runner says so (its
        # thread, no request waiting for the GPU: `joint_capture_ok`, set a round at a time)
        fn.joint_capture_ok = bool(getattr(items[0][0], "joint_capture_ok", False))
        items[0][0].joint_capture_ok = False
        jobs = []
        t0 = time.perf_counter()
        for s, block, parents, start in items:
            s._settle_chain()
            s._at(start)
            s._ensure(start + block.numel())
            jobs.append(([int(t) for t in block.view(-1).tolist()], [int(p) for p in parents], s.st))
        jt = getattr(fn, "joint_timing", None)
        if jt is not None:
            # SPD-239: the host before the pass (a chain's commit, the window, the blocks to the host)
            jt["host_pre"] = jt.get("host_pre", 0.0) + 1e3 * (time.perf_counter() - t0)
        try:
            outs = fn.verify_multi(jobs)
        finally:
            fn.joint_capture_ok = False
        # SPD-249: the pass is queued: the round runner's host work no round waits for runs in the wait
        # (engine/methods/busy.py; nothing without a runner hook)
        busy.busy()
        torch.cuda.current_stream().synchronize()
        res = []
        for (s, block, parents, start), (logits, picks, stack) in zip(items, outs):
            M = block.numel()
            s.picks, s._hidden = picks, stack.unsqueeze(0)
            s.captured, s._width = False, M
            s.uses[M] = s.uses.get(M, 0) + 1
            s.stats["verify_blocks"] += 1
            s.stats["verify_rows"] += M
            res.append(logits)
        return res

    @torch.no_grad()
    def commit_tree(self, path: list[int]) -> None:
        t0 = time.perf_counter()
        fn, st, path = self.fn, self.st, [int(p) for p in path]
        n_graphs, kept = len(self.cgraphs), True
        if self._eager_block:
            # SPD-261: after an eager verify (a commit graph reads the verify graph's own intermediates)
            self._eager_block = False
            stack = fn.commit(st, path, None)
        else:
            # SPD-261: `commit`'s key (`stage_commit`: the verified block's rows, the path's, moved, the class); a block
            # of a joint pass commits eagerly whatever it is given
            key = (self._width, len(path), any(p != j for j, p in enumerate(path)), getattr(fn, "_vcls", 0))
            capture = key not in self.cgraphs and id(st) not in fn._joint
            stack, kept = self._graphs_call("commit", lambda g: fn.commit(st, path, g), self.cgraphs, capture)
        self.commit_ms = (time.perf_counter() - t0) * 1e3
        if len(self.cgraphs) > n_graphs or not kept:
            self.captured = True
            self._capture_took(t0)
        k = (self._width, len(path), any(int(p) != j for j, p in enumerate(path)))
        self.uses[k] = self.uses.get(k, 0) + 1
        if k + (getattr(self.fn, "_vcls", 0),) in self._fresh:
            self._fresh.discard(k + (getattr(self.fn, "_vcls", 0),))
            self.captured = True
        if path:
            self._h_last.copy_(stack[len(path) - 1:len(path)])

    @staticmethod
    @torch.no_grad()
    def commit_tree_multi(items: list) -> None:
        """SPD-239: [(serving, path)] of one joint pass's slots committed at once (`FlashNextEngine.commit_multi`:
        every lane's copies in one launch), each slot left as its `commit_tree` leaves it; a slot whose block was
        not in the joint pass takes its own `commit_tree`."""
        fn = items[0][0].fn
        mine = [(s, path) for s, path in items if id(s.st) in fn._joint]
        for s, path in items:
            if id(s.st) not in fn._joint:
                s.commit_tree(path)
        if not mine:
            return
        t0 = time.perf_counter()
        stacks = fn.commit_multi([(s.st, [int(p) for p in path]) for s, path in mine])
        ms = (time.perf_counter() - t0) * 1e3 / len(mine)
        for (s, path), stack in zip(mine, stacks):
            s.commit_ms = ms
            k = (s._width, len(path), any(int(p) != j for j, p in enumerate(path)))
            s.uses[k] = s.uses.get(k, 0) + 1
            if k + (getattr(fn, "_vcls", 0),) in s._fresh:
                s._fresh.discard(k + (getattr(fn, "_vcls", 0),))
                s.captured = True
            if path:
                s._h_last.copy_(stack[len(path) - 1:len(path)])

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
    def _states(self) -> list:
        return [self.st] + [h.st for h in self.heads.values()]

    def _settle_chain(self) -> None:
        if self._chain is not None:
            n, self._chain = self._chain, None
            self.commit_tree(list(range(n)))

    def _ensure(self, rows: int) -> None:
        """ENG-241: commit the pages that hold rows [0, rows) and the pool's margin before a write
        reaches them (one int compare when they are); never inside a graph capture, whose caller
        committed the rows before it captured."""
        ps = self.st.pages
        if ps is None or rows + ps.margin() <= ps.rows:
            return
        if ps.on_device() and torch.cuda.is_current_stream_capturing():
            return
        ps.ensure(rows)

    def _at(self, start: int) -> None:
        if int(start) != self.st.length:
            raise ValueError(f"Flash-Next forward at {start}, the state holds {self.st.length} rows")


#: one graph's capture before the server has timed its own, ms (P0, 2026-10-12: 3,391 ms for 9 at the
#: 32,768 class, 5,121 for 12, 4,981 for 15 at 131,072)
CAPTURE_MS = 380.0
#: a side slot's graph pools and scratch beyond its state, bytes (ENG-241: an estimate until measured)
SIDE_SLOT_CONST = int(1.5 * (1 << 30))
#: the shapes a class captures ahead before any round was counted: chain blocks of 2-6 rows, each
#: path length (one round each, so the server's own rounds outweigh them at once)
PRIOR_USES = {**{M: 1 for M in range(2, 7)}, **{(M, n, False): 1 for M in range(2, 7) for n in range(1, M + 1)}}

#: the repository root (the hot draft vocabulary is ops/fn-hot-vocab.npy, the price files ops/prices/)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#: the model's name in ops/prices/ and models/
MODEL_ID = "qwen3.8-flash-next"


def profile_of(pack: str) -> str:
    """The weight profile a pack is served as: the running one (QSE_WEIGHTS), else flashnext-<pack dir>."""
    return os.environ.get("QSE_WEIGHTS") or "flashnext-" + os.path.basename(os.path.normpath(pack))


def prices(pack: str, log=print) -> PriceModel:
    """StairCut's prior for this pack on this device (ops/prices/<model>/<profile>.<device>.json)."""
    return PriceModel.load(os.path.join(ROOT, "ops", "prices", MODEL_ID, f"{profile_of(pack)}.{device_id()}.json"),
                           log=log)


def draft_freeze(pm: PriceModel, log=print) -> bool:
    """SPD-249: whether the MTP drafters keep their adaptive state at its start (FN_DRAFT_FREEZE): only over a price
    file, whose prices a frozen cut keeps. With no price file for this pack on this device the cut has no prices but
    the ones it learns, so it learns them as served."""
    if _S.get("FN_DRAFT_FREEZE") != "1":
        return False
    if not pm.table:
        log("[server] FN_DRAFT_FREEZE: no price file for this pack on this device: the drafters learn their prices")
        return False
    log("[server] Flash-Next drafter state frozen (FN_DRAFT_FREEZE): the cut keeps the price file's prices and "
        "learns nothing")
    return True


def corrector_heads(fn, rows: int) -> dict:
    """SPD-249: the keyed heads' hot ids -> whether a corrector table of `rows` hot rows covers that head (its table is
    by hot index, the heads are prefixes of ops/fn-hot-vocab.npy). The keyed heads are `keyed_head`'s: the coupled
    head and its one-lane rows, else the draft head; {}: keyed drafts read the full head."""
    ids = [fn.coupled_ids, fn.lone_ids] if fn.coupled_head is not None else [fn.draft_ids]
    return {int(i.numel()): int(i.numel()) <= rows for i in ids if i is not None}


def build(pack: str, max_len: int, tok, spec: bool = True, depth: int = 4, mark=lambda stage: None, log=print,
          corpus: str | None = None, kv_pool=None):
    """Flash-Next for the server (`serve` below): the pack's weights, the engine in the loop's
    interface, and with `spec` the shared MTP drafter over the pack's MTP head with the hot draft head
    (`FN_DRAFT_HOT` ids of ops/fn-hot-vocab.npy, 0 = the whole vocabulary), its graphs captured and
    StairCut's prior from the price file -- gated by the load-time self-check (a failed check serves
    plain decode); with FN_LOOKUP the lookup over the request's text and `corpus` joins the cut (ENG-229;
    the tokenizer is the 27B's, so its corpus and suffix store read as they are, checked by fingerprint).
    `mark(stage)` at "weights", "engine", "drafters" (the server's load ledger). `kv_pool()`: opens the
    KV pool after the weights, before the state is allocated, as the 27B's builder does (ENG-241: opened
    before the weights, the self-check's first Triton launch refused a pointer, "cannot be accessed
    from Triton", 2026-10-12).
    Returns (serving, drafter or None, info)."""
    from engine.models.flashnext.load import load as fn_load
    idx, W, banks, tables, timing = fn_load(pack, "cuda", mtp=spec)
    mark("weights")
    fn = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=max_len)
    if kv_pool is not None:
        kv_pool()
    eng = FlashNextServing(fn, max_len)
    mark("engine")
    drafter, check, hot, corrector = None, None, int(_S.get("FN_DRAFT_HOT")), None
    if spec:
        if hot:
            fn.set_draft_head(np.load(os.path.join(ROOT, "ops", "fn-hot-vocab.npy"))[:hot])
        from engine.models.flashnext import coupled as CP
        if CP.ON:
            # SPD-249: seeded sampled requests draft coupled; their own hot head (FN_MTP_COUPLED_HOT), their cut
            # inputs (the measured file next to the price file)
            if CP.HOT:
                # FN_MTP_COUPLED_HOT_LONE: a one-lane keyed chain reads the coupled head's first rows (a view)
                fn.set_coupled_head(np.load(os.path.join(ROOT, "ops", "fn-hot-vocab.npy"))[:CP.HOT], lone=CP.HOT_LONE)
            if CP.HOT_LONE and fn.lone_head is None:
                log(f"[server] FN_MTP_COUPLED_HOT_LONE={CP.HOT_LONE} ignored: it needs FN_MTP_COUPLED_HOT above it "
                    f"(now {CP.HOT})")
            ci = CP.CalInputs.get(pack)
            lone = f", one lane hot {fn.lone_ids.numel()}" if fn.lone_head is not None else ""
            log(f"[server] Flash-Next coupled drafts for seeded sampled requests: tau {CP.TAU} x the request's, "
                f"filter {'the request' if CP.FILTER else 'none'}, head "
                f"{'hot ' + str(CP.HOT) if CP.HOT else 'the draft head'}{lone}, sibling "
                f"{'margin <= ' + str(CP.SIB_MARGIN) if CP.SIB else 'off'}, cut inputs "
                f"{'from ' + ci.file if ci is not None else 'the cut own'}")
            from engine.models.flashnext import corrector as CR
            heads = None
            if CR.ON:
                # SPD-249: the corrector on the device before any draft graph is captured (its launch is in them),
                # its file checked against this model's MTP output width
                cr = CR.get(pack=pack, hidden=idx.cfg.hidden_size)
                if cr is not None:
                    heads = corrector_heads(fn, int(cr.T.shape[0]))
                    if not heads:
                        log("[server] FN_MTP_CORRECTOR needs a hot draft head (its table is by hot index): keyed "
                            "drafts over the full head keep the plain coupled pick")
                    for n, ok in heads.items():
                        if not ok:
                            log(f"[server] FN_MTP_CORRECTOR: its table has {cr.T.shape[0]} hot rows, the keyed head "
                                f"{n}: that head's drafts keep the plain coupled pick")
            corrector = CR.status(heads)
        text = tok.apply_chat_template([{"role": "user", "content": "Count from one to twenty in words."}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        check = fn.self_check(tok(text, add_special_tokens=False)["input_ids"], log=log)
        if check["ok"]:
            ngram = None
            if _S.get("FN_LOOKUP") == "1":
                from engine.drafters.ngram import NgramDrafter
                from engine.tokfp import fingerprint
                ngram = NgramDrafter(corpus_path=corpus or "", tokenizer_sha=fingerprint(pack), min_order=3,
                                     max_depth=16, min_corpus_order=8)
                log(f"[server] Flash-Next lookup in the cut: corpus "
                    f"{(corpus or 'none') if ngram.corpus is not None else 'none'}")
            pm = prices(pack, log)
            drafter = MTPTreeDrafter(eng, eng.add_mtp_head(depth), StairCut(pm, siblings=False), ngram,
                                     freeze=draft_freeze(pm, log))
            # SPD-249: the lookup's backoff on rounds other lanes share (0: off)
            drafter.mtp_look_backoff = int(_S.get("FN_LOOK_BACKOFF") or 0)
            if drafter.mtp_look_backoff and ngram is not None:
                log(f"[server] Flash-Next lookup backoff on shared rounds: up to {drafter.mtp_look_backoff} rounds "
                    f"(FN_LOOK_BACKOFF)")
            drafter.capture_all()
        else:
            log("[server] Flash-Next self-check FAILED: speculation off, plain decode only")
    mark("drafters")
    info = {"pack": pack, "load": timing, "self_check": check,
            "draft_head": f"hot {hot}" if spec else None, "depth": depth if drafter is not None else 0}
    if spec and fn.coupled_head is not None:
        info["draft_head"] += f", coupled hot {fn.coupled_ids.numel()}"
        if fn.lone_head is not None:
            info["draft_head"] += f" (one lane {fn.lone_ids.numel()})"
    if corrector is not None:
        # SPD-249: whether the coupled drafts' corrector loaded, its file and why not (/health weights.corrector)
        info["corrector"] = corrector
    return eng, drafter, info


def serve(a, ctx) -> dict:
    """Qwen3.8-Flash-Next from an EXL3 pack for the server (`--model <pack>`, the registry's `serve`;
    server/app.py `_load`): the engine and its MTP drafter in the server loop's interface, so the
    request path is the 27B's. The draft is the MTP chain from graphs cut by StairCut over the hot
    draft head, verified as a tree that a sampled request walks (`det`); the load-time self-check gates
    it. `ctx` is the server's: `keys(path, weights)` (the tokenizer's and the weights' fingerprints),
    `mark(stage)` (the load ledger), `log(line)`, `vision(path, tower_of)` (the server's tower, processor and
    image caches, with the pack's EXL3 tower: VIS-97). What it cannot hold yet is in `FlashNextServing.caps`."""
    from transformers import AutoTokenizer

    from engine.spec import Relax
    ctx.keys(a.model, [a.model])
    spec = a.drafter != "none"
    if a.drafter not in ("mtp", "none"):
        ctx.log(f"[server] Flash-Next drafts with its own MTP layer: --drafter {a.drafter} taken as mtp")
    tok = AutoTokenizer.from_pretrained(a.model)
    depth = a.depth if a.depth > 3 else 4
    pool = ctx.kv_pool if getattr(a, "kv_pool", "off") == "on" and hasattr(ctx, "kv_pool") else None
    eng, drafter, info = build(a.model, a.max_len, tok, spec=spec, depth=depth, mark=ctx.mark, log=ctx.log,
                               corpus=a.corpus, kv_pool=pool)

    def tower_of(vcfg):
        from engine.models.flashnext.pack import PackIndex
        from engine.models.flashnext.vision import load_tower
        return load_tower(PackIndex.open(a.model, table=False), a.model)
    ctx.vision(a.model, tower_of)
    ctx.mark("vision")
    if drafter is not None:
        a.depth = depth
        if drafter.ngram is not None:
            # the loop hands a round k rows past the anchor: the lookup's copy may run past the head's
            # depth, up to what the cut admits
            a.k = max(int(a.k or 0), drafter.cut.max_rows - 1)
    describe = (f"Flash-Next {a.model}: load {info['load'].get('total_s')} s, drafter "
                f"{'MTP chain + StairCut, depth ' + str(a.depth) if drafter is not None else 'none'}"
                f"{' + lookup' if getattr(drafter, 'ngram', None) is not None else ''}, "
                f"max_len {a.max_len:,},")
    return {"engine": eng, "tok": tok, "drafter": drafter, "path": a.model, "device": "cuda",
            "cfg": None, "w": None, "describe": describe,
            "serving": {"tree": drafter is not None, "sampled_tree": "det", "relax": Relax(1.0, 1),
                        "weights_info": {"format": "exl3", "weights": a.model,
                                         "drafters": "mtp (the pack's own layer)" if drafter is not None else None,
                                         "draft_head": info["draft_head"], "load_s": info["load"].get("total_s"),
                                         "self_check": info["self_check"],
                                         **({"corrector": info["corrector"]} if "corrector" in info else {})}}}
