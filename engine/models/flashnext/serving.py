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

from engine.methods.device import device_id
from engine.methods.manifest import Manifest
from engine.methods.mtp_drafter import MTPTreeDrafter
from engine.methods.staircut import PriceModel, StairCut
from engine.models.adapter import Capabilities, CostModel
from engine.models.flashnext.engine import DecodeGraph, FlashNextEngine
from engine.models.flashnext.mtp import FlashNextMTPHead
from engine.models.state import SectionState
from engine.settings import SETTINGS as _S


#: fingerprint v3 (`FlashNextServing.manifest`): the package's files that compute rows, the tables
MODULES = ("engine/models/flashnext/__init__.py", "engine/models/flashnext/config.py",
           "engine/models/flashnext/engine.py", "engine/models/flashnext/load.py",
           "engine/models/flashnext/mtp.py", "engine/models/flashnext/pack.py",
           "engine/models/flashnext/ple.py", "engine/models/flashnext/ref.py", "engine/settings.py")
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


class FlashNextServing:
    """`FlashNextEngine` with one state and its graphs, in the server's engine interface."""

    # the caches' grid: the prefill kernels' chunk (FN_PREFILL_CHUNK), so a resumed prefill runs the
    # chunks a cold one runs; side slots: ENG-241
    caps = Capabilities(caches=True, side_slots=False, anchor_grid=4096)

    def __init__(self, fn: FlashNextEngine, max_len: int):
        self.fn, self.cfg, self.max_len = fn, fn.cfg, max_len
        self.st = fn.new_state(max_len)
        c, dev = fn.cfg, fn.device
        self._ctx = {li: torch.full((1, fn.hash[li].context_len), c.eos_token_id, dtype=torch.long, device=dev)
                     for li in c.ple_layers}
        self._h_last = torch.zeros(1, c.hc_count * c.hidden_size, dtype=fn.dtype, device=dev)
        self.state = FlashNextCacheState(self)
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
        self.heads: dict = {}               # draft heads by name (`draft_heads`)
        self._weights = None                # the weights' bytes by part (`memory_parts`), once
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
        # the prefill workspace, measured 2026-10-12 on 3.05 (tools/fn_memory_check.py): 2.51 GiB at 8k
        # and 2.40 GiB at 32k -- the 4,096-row chunk's activations, flat in the context -- so a floor of
        # 2.75 GiB; the line is the QSA score buffer's bound (9,216 bytes a context token) twice, 4.5 GiB at
        # 262,144, until the memory policy has observed peaks above 65,536 tokens
        return CostModel(section_bytes=sum(sec.values()), view_bytes_per_token=per_tok,
                         workspace_floor=int(2.75 * (1 << 30)), workspace_per_token=2 * 9216,
                         parts={"recurrent": sum(b for k, b in sec.items() if k.startswith("gdn.S")),
                                "conv": sum(b for k, b in sec.items() if k.startswith(("gdn.conv", "ple.conv")))})

    def manifest(self) -> Manifest:
        return manifest()

    def precapture(self, drafter) -> None:
        return None                         # the drafter captured every graph at load (`capture_all`)

    def draft_heads(self) -> dict:
        """The draft heads this pack carries, by name: its MTP layer ("mtp", made by `add_mtp_head`)."""
        return self.heads

    def add_mtp_head(self, depth: int) -> FlashNextMTPHead:
        self.heads["mtp"] = FlashNextMTPHead(self.fn, depth, self.max_len)
        return self.heads["mtp"]

    def settle(self) -> None:
        """Land what is pending, so the state's sections are the sequence's whole (before a
        snapshot): a chain block's commit, and the GDN rows the folded commit parked for the next call."""
        self._settle_chain()
        if self.fn._pend_owner is self.st:
            self.fn.flush()

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

        def nb(*ds):
            return sum(int(t.numel() * t.element_size()) for d in ds for t in d.values())
        head = self.heads.get("mtp")
        mtp_kv = 0
        if head is not None:
            mtp_kv = nb(head.st.k, head.st.v, head.st.ik, head.st.pool)
        return {**self._weights, "kv_pool": nb(st.k, st.v, st.ik, st.pool), "drafter_kv": mtp_kv,
                "gdn_state": sum(int(t.numel() * t.element_size()) for t in self.state.sections().values())}

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
            self._h_last.copy_(self._hidden[0, -1:])
            self.stats["decode_steps"] += 1
            return lg.view(1, 1, -1)
        lg = self.fn.prefill(ids, self.st, keep_stack=True)
        self._hidden = self.fn.prompt_stack.unsqueeze(0)
        self._h_last.copy_(self._hidden[0, T - 1:T])
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
        stack = self.fn.commit(self.st, [int(p) for p in path], self.cgraphs)
        self.commit_ms = (time.perf_counter() - t0) * 1e3
        self.captured |= len(self.cgraphs) > n_graphs
        if path:
            self._h_last.copy_(stack[len(path) - 1:len(path)])

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


def build(pack: str, max_len: int, tok, spec: bool = True, depth: int = 4, mark=lambda stage: None, log=print,
          corpus: str | None = None):
    """Flash-Next for the server (`serve` below): the pack's weights, the engine in the loop's
    interface, and with `spec` the shared MTP drafter over the pack's MTP head with the hot draft head
    (`FN_DRAFT_HOT` ids of ops/fn-hot-vocab.npy, 0 = the whole vocabulary), its graphs captured and
    StairCut's prior from the price file -- gated by the load-time self-check (a failed check serves
    plain decode); with FN_LOOKUP the lookup over the request's text and `corpus` joins the cut (ENG-229;
    the tokenizer is the 27B's, so its corpus and suffix store read as they are, checked by fingerprint).
    `mark(stage)` at "weights", "engine", "drafters" (the server's load ledger).
    Returns (serving, drafter or None, info)."""
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
            ngram = None
            if _S.get("FN_LOOKUP") == "1":
                from engine.drafters.ngram import NgramDrafter
                from engine.tokfp import fingerprint
                ngram = NgramDrafter(corpus_path=corpus or "", tokenizer_sha=fingerprint(pack), min_order=3,
                                     max_depth=16, min_corpus_order=8)
                log(f"[server] Flash-Next lookup in the cut: corpus "
                    f"{(corpus or 'none') if ngram.corpus is not None else 'none'}")
            drafter = MTPTreeDrafter(eng, eng.add_mtp_head(depth), StairCut(prices(pack, log), siblings=False), ngram)
            drafter.capture_all()
        else:
            log("[server] Flash-Next self-check FAILED: speculation off, plain decode only")
    mark("drafters")
    info = {"pack": pack, "load": timing, "self_check": check,
            "draft_head": f"hot {hot}" if spec else None, "depth": depth if drafter is not None else 0}
    return eng, drafter, info


def serve(a, ctx) -> dict:
    """Qwen3.8-Flash-Next from an EXL3 pack for the server (`--model <pack>`, the registry's `serve`;
    server/app.py `_load`): the engine and its MTP drafter in the server loop's interface, so the
    request path is the 27B's. The draft is the MTP chain from graphs cut by StairCut over the hot
    draft head, verified as a tree that a sampled request walks (`det`); the load-time self-check gates
    it. `ctx` is the server's: `keys(path, weights)` (the tokenizer's and the weights' fingerprints),
    `mark(stage)` (the load ledger), `log(line)`. What it cannot hold yet is in `FlashNextServing.caps`."""
    from transformers import AutoTokenizer

    from engine.spec import Relax
    ctx.keys(a.model, [a.model])
    spec = a.drafter != "none"
    if a.drafter not in ("mtp", "none"):
        ctx.log(f"[server] Flash-Next drafts with its own MTP layer: --drafter {a.drafter} taken as mtp")
    tok = AutoTokenizer.from_pretrained(a.model)
    depth = a.depth if a.depth > 3 else 4
    eng, drafter, info = build(a.model, a.max_len, tok, spec=spec, depth=depth, mark=ctx.mark, log=ctx.log,
                               corpus=a.corpus)
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
                                         "self_check": info["self_check"]}}}
