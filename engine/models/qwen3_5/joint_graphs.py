"""The 27B's joint verify pass served from CUDA graphs (SPD-163, QWEN38_JOINT_GRAPHS).

A joint pass (`engine.model.forward_tree_multi`) issues about 5,300 kernels for 8 blocks, and in the server
its host side ran 113-131 ms against about 103 ms of GPU work: the slots' Python threads share one GIL with
it. Captured, a pass is one replay. What a pass varies is made graph-safe the way the lone verify graphs
make it (engine/verify_graph.py), per engine:

  * every engine's block is padded to one row count, `T_pad = min(16, max_rows // E)`: the pad rows are leaf
    children of the anchor after the real nodes in DFS order, so no real row sees them (the ancestor mask,
    the convolution windows) and their tokens are the anchor's. The joint cut keeps each tree within
    T_pad (server/app.py `_run_round`, the family's `joint_policy` "tree_cap");
  * the tokens, positions and KV slots of a block, its `[start, start + T]` and its tree are static device
    buffers, filled before a replay; attention writes the KV at the slots and reads the whole cache with the
    length from the device (`decode_attention_dev`, the lone graph's), so the pad rows are never read;
  * the GDN recurrence is the grouped WY launch (QWEN38_JOINT_GDN), whose offset table is this graph's own:
    a pinned host buffer the graph copies as its first node. Before a replay the host writes each engine's
    real row count (the kernels mask the pad rows: their loads read zeros, as the unpadded launch's do) and
    the offsets of the static factor buffers of its fold parity -- so the parity is not in the key.

The key is (the engines in order, T_pad, each engine's context class, its fold, the kernel signature).
Each real row is computed by the same statements as in the eager joint pass, at the same row count of its
own block: `tools/joint_graph_check.py` holds the logits, the state and the KV of every engine to its
eager verify, over rounds with commits. Everything else (a block past T_pad, a context past the graphs'
class, a chain on its chain kernel, fp8 KV) runs the eager joint pass, unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from engine import capture_lock
from engine.models.graph_lru import GraphLRU
from engine.settings import SETTINGS as _S

#: on (1): a joint pass of the 27B from a CUDA graph where it qualifies; 0: the eager joint pass
ON = _S.get("JOINT_GRAPHS") == "1"
#: joint graphs kept (one per engine set, row bucket and context classes); the least recently replayed go
CAP = int(_S.get("JOINT_GRAPHS_MAX") or 12)
#: a pass shape is captured once it has come this many times (an engine set that only exists while the
#: slots' streams end one by one runs eager instead of paying a capture each)
AFTER = int(_S.get("JOINT_GRAPHS_AFTER") or 3)
#: the most rows a block keeps in a graphed pass (the grouped WY tile)
MAX_T = 16
#: SPD-201: a pass of one block from a graph too (the round runner's QSE_RUNNER_JOINT1)
ONE = __import__("os").environ.get("QSE_RUNNER_JOINT1", "0") == "1"
#: SPD-187: set by the round planner for the pass about to run (server/app.py `_run_round`): its bucket
FORCE_T = None
#: the factor columns of the GDN offset table a replay rewrites for an engine's parity (gc, kk, delta, and
#: the pending commit's kk, u, gc: tools/gdn_wy_kernels.py NOFF)
_PAR_COLS = [1, 3, 10, 12, 13, 14]


#: SPD-199: rows a block keeps in a joint pass at any E (0: the share of the row cap, `max_rows // E`); the
#: joint cut's tree cap and the graphs' padding both follow it
ROWS_A_SLOT = int(_S.get("JOINT_ROWS_A_SLOT") or 0)


def t_pad(E: int, max_rows: int) -> int:
    """The rows every block of an E-block graphed pass is padded to (the joint cut's tree cap)."""
    t = min(MAX_T, max_rows // max(1, E))
    if ROWS_A_SLOT:
        t = min(t, ROWS_A_SLOT)
    return max(2, t)


class _Bufs:
    """One engine's static inputs at one row count: tokens, positions and KV slots in one device tensor
    (one copy a replay), the start and length, the padded tree, the context class."""

    def __init__(self, eng, T: int):
        from engine.verify_graph import StaticTree
        dev = eng.device
        self.T = T
        self.ids = torch.zeros(3, T, dtype=torch.long, device=dev)
        self.ids_host = torch.zeros(3, T, dtype=torch.long).pin_memory()
        self.tok, self.pos, self.slots = self.ids[0], self.ids[1], self.ids[2]
        self.lenp = torch.zeros(2, dtype=torch.int32, device=dev)
        self.lenp_host = torch.zeros(2, dtype=torch.int32).pin_memory()
        self.tree = StaticTree(T, eng.cfg.linear_conv_kernel_dim, dev)
        self.max_lc = 0

    def fill(self, eng, tokens: list, depths: list, start: int, n_real: int, ctx) -> None:
        T = self.T
        h = self.ids_host
        h[0, :].copy_(torch.tensor(tokens, dtype=torch.long))
        s0 = start + eng.pos_delta
        h[1, :].copy_(torch.tensor([s0 + d for d in depths], dtype=torch.long))
        h[2, :].copy_(torch.arange(start, start + T, dtype=torch.long))
        self.ids.copy_(h, non_blocking=True)
        self.lenp_host[0], self.lenp_host[1] = start, start + n_real
        self.lenp.copy_(self.lenp_host, non_blocking=True)
        self.tree.load(ctx)


class _Recorder:
    """The GDN offset table of one graph (tools/gdn_wy_kernels.py TABLE_REC): one row of E x NOFF a
    layer, in pinned memory the graph copies to the device as its first node."""

    def __init__(self, E: int, dev, layers: int):
        from tools.gdn_wy_kernels import NOFF
        self.E, self.K = E, 0
        self.host = torch.zeros(layers, E, NOFF, dtype=torch.int64).pin_memory()
        self.np = self.host.numpy()
        self.dev = torch.zeros(layers, E, NOFF, dtype=torch.int64, device=dev)
        self.calls: list = []

    def begin(self) -> None:
        self.dev.copy_(self.host, non_blocking=True)

    def record(self, off: list, items: list, facs: list):
        k = self.K
        self.K += 1
        self.np[k].reshape(-1)[:] = off
        p0 = items[0]["pend"]
        bases = ([f.data_ptr() for f in facs[0]],
                 [x.data_ptr() for x in p0[:3]] if p0 is not None else None)
        self.calls.append(([(it.get("eng"), it.get("slot")) for it in items], bases))
        return self.dev[k]

    def parity_table(self) -> np.ndarray | None:
        """[K, E, 2, 6]: per layer, engine and fold parity the offsets of the factor columns (None
        without the fold: the factors are then the launch's own buffers, which do not move)."""
        out = np.zeros((self.K, self.E, 2, len(_PAR_COLS)), dtype=np.int64)
        for k, (engs, (fb, pb)) in enumerate(self.calls):
            if pb is None:
                return None
            for e, (eng, i) in enumerate(engs):
                for p in (0, 1):
                    fac, other = eng._fac[p], eng._fac[1 - p]
                    ptrs = [fac[2][i].data_ptr(), fac[0][i].data_ptr(), fac[1][i].data_ptr(),
                            other[0][i].data_ptr(), other[1][i].data_ptr(), other[2][i].data_ptr()]
                    base = [fb[2], fb[0], fb[1], pb[0], pb[1], pb[2]]
                    for c, (a, b) in enumerate(zip(ptrs, base)):
                        d = a - b
                        if d % 64:                      # fp32, and the kernels' 16-element hint
                            raise ValueError("a factor buffer is not 16-element aligned")
                        out[k, e, p, c] = d // 4
        return out

    def patch(self, ns: list, pars: list, var) -> None:
        K = self.K
        self.np[:K, :, 0] = np.asarray(ns, dtype=np.int64)[None, :]
        if var is not None:
            for e, p in enumerate(pars):
                self.np[:K, e, _PAR_COLS] = var[:, e, p, :]


class _Captured:
    __slots__ = ("graph", "logits", "hidden_pre", "hidden_post", "taps", "layers", "factors",
                 "conv_entry", "rec", "var", "done", "fold")

    def __init__(self):
        self.graph = torch.cuda.CUDAGraph()
        self.done = None


class JointGraphs:
    """The joint graphs of the engines over one set of weights."""

    def __init__(self):
        self.lru = GraphLRU(CAP)
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream()
        self.bufs: dict = {}
        self.bad: set = set()                # keys whose capture was refused: eager from then on
        self.seen: dict = {}                 # passes of a shape not captured yet
        self.stats = {"replayed": 0, "captured": 0, "eager": 0, "why": {}, "capture_ms": 0.0}

    def _no(self, why: str):
        self.stats["eager"] += 1
        self.stats["why"][why] = self.stats["why"].get(why, 0) + 1
        return None

    # ------------------------------------------------------------------ eligibility
    def plan(self, items: list):
        """(T_pad, per engine (padded tokens, padded parents, n real)) when this pass can be graphed,
        else None (and why, in `stats`)."""
        import engine.model as M
        from engine.verify_graph import MAX_CTX
        if not (M.JOINT_GDN and M.GDN_AB and M.FUSED_COMMIT and M.FUSED_GDNVERIFY and M.DECODE_ATTN
                and M.FUSED_ADDNORM and M.FUSED["norm"] and M.JOINT_ROWWISE):
            return self._no("flags")
        E = len(items)
        ns = [len(p) for _, _, p, _ in items]
        cap_rows = int(items[0][0].joint_policy()["max_rows"])
        T = t_pad(E, cap_rows)
        if FORCE_T is not None and 2 <= FORCE_T <= MAX_T and E * FORCE_T <= cap_rows:
            T = FORCE_T                     # SPD-187: the round planner's bucket
        if max(ns) > T:
            # the joint cut keeps trees within T_pad (`tree_cap`); a pass made otherwise runs eager
            return self._no("rows")
        folds = []
        for eng, _, _, start in items:
            if eng.kv.fp8 or eng._rope_rows is not None or eng.mm is not None:
                return self._no("engine")
            if start + T > MAX_CTX or start + T > eng.max_len:
                return self._no("context")
            folds.append(eng._folds(T, tree=True))
        if len(set(folds)) != 1:
            return self._no("fold")
        if not (M.GQA_FROM > T):
            return self._no("rows")
        pads = []
        for (eng, tokens, parents, start), n in zip(items, ns):
            toks = [int(t) for t in tokens.view(-1).tolist()] if torch.is_tensor(tokens) else list(tokens)
            par = [int(p) for p in parents]
            toks += [toks[0]] * (T - n)
            par += [0] * (T - n)
            pads.append((toks, par, n))
        return T, pads, folds[0]

    # ------------------------------------------------------------------ the pass
    def run(self, items: list):
        """`forward_tree_multi(items)` from a graph; None when the pass does not qualify or its graph could
        not be captured (the caller runs it eager, from the state it was handed)."""
        import engine.model as M
        from engine.verify_graph import VerifyGraphs
        plan = self.plan(items)
        if plan is None:
            return None
        T, pads, fold = plan
        dev = items[0][0].device
        bufs, ctxs = [], []
        for (eng, tokens, parents, start), (toks, par, n) in zip(items, pads):
            eng.kv.ensure(start + T)             # SRV-307: the padded rows, before the graph writes them
            ctx = M.TreeCtx.get(tuple(par), dev, eng.cfg.linear_conv_kernel_dim)
            b = self.bufs.get((id(eng), T))
            if b is None:
                b = self.bufs[(id(eng), T)] = _Bufs(eng, T)
            b.fill(eng, toks, ctx.depth_list, start, n, ctx)
            b.max_lc = VerifyGraphs.ctx_class(start + T)
            bufs.append(b)
            ctxs.append(M.TreeCtx.get(tuple(int(p) for p in parents), dev, eng.cfg.linear_conv_kernel_dim))
        key = (tuple(id(e) for e, *_ in items), T, tuple(b.max_lc for b in bufs), fold,
               VerifyGraphs.signature())
        if key in self.bad:
            return self._no("refused")
        if key not in self.lru:
            n_seen = self.seen[key] = self.seen.get(key, 0) + 1
            if n_seen < AFTER:
                return self._no("warming")
            if len(self.seen) > 4096:
                self.seen.clear()
            # captured before `_fold_begin` hands this round's pending commit to the device, so a refusal
            # leaves the engines as the eager pass expects them; the capture writes the parity `_par`
            keep = [eng._fold_par for eng, *_ in items]
            try:
                for eng, *_ in items:
                    if fold:
                        eng._fold_buffers()
                        eng._fold_par = eng._par
                import time
                t0 = time.perf_counter()
                made = self._capture(items, bufs, fold)
                self.stats["capture_ms"] += (time.perf_counter() - t0) * 1e3
            except Exception as exc:                                    # noqa: BLE001
                self.bad.add(key)
                print(f"[joint-graphs] capture refused ({type(exc).__name__}: {exc}); this pass shape runs "
                      f"eager", flush=True)
                return self._no("capture")
            finally:
                for (eng, *_), fp in zip(items, keep):
                    eng._fold_par = fp
            self.lru.take(key, lambda: made)
        cap = self.lru.take(key, None)
        for eng, *_ in items:
            # `_tree_begin`'s state steps: the fold's pending commit handed to the device, the parity fixed
            eng.picks = None
            if fold:
                eng._fold_begin()
            else:
                eng._settle()
            eng._tree_is_chain = False
        ns = [p[2] for p in pads]
        if cap.done is not None:
            cap.done.synchronize()          # the last replay has read the table; it may be rewritten
        cap.rec.patch(ns, [eng._fold_par for eng, *_ in items] if fold else [0] * len(items), cap.var)
        cap.graph.replay()
        if cap.done is None:
            cap.done = torch.cuda.Event()
        cap.done.record()
        self.stats["replayed"] += 1
        return [self._restore(eng, cap, e, start, n, ctx, fold)
                for e, ((eng, _, _, start), n, ctx) in enumerate(zip(items, ns, ctxs))]

    def _restore(self, eng, cap: "_Captured", e: int, start: int, n: int, ctx, fold: bool):
        """What `_tree_end` leaves after the eager joint pass: the trace (the factors of this parity's
        buffers, or the graph's own), the hidden rows, the tree, the length, the drafter's taps."""
        import engine.model as M
        tr = M.BlockTrace()
        tr.S_entry = eng.state.S
        tr.conv_entry = cap.conv_entry[e]
        tr.layers = cap.layers[e]
        if fold:
            par = eng._fold_par
            # the factors live in this parity's static buffers; the fold's commit names them by parity
            # and never reads these views, so they are made only if something asks (`_Factors`)
            tr.factors = _Factors(eng, eng._fac[par], cap.layers[e], n)
            tr.fold_par = par
        else:
            tr.factors = {layer: tuple(f[..., :n, :] if f.dim() == 4 else f[..., :n] for f in fs)
                          for layer, fs in cap.factors[e].items()}
        tr.start = start
        eng.tree = None
        eng.trace = None
        if fold:
            eng._fold_end(None)
        eng._trace = tr
        eng._tree = ctx
        eng._tree_start = start
        eng.hidden_pre_norm = cap.hidden_pre[e][:, :n]
        eng.hidden_post_norm = cap.hidden_post[e][:, :n]
        eng.state.primed = True
        eng.kv.length = start + n
        if eng.tap is not None:
            for h in cap.taps[e]:
                eng.tap(h)
        return cap.logits[e][0, :n]

    # ------------------------------------------------------------------ capture
    def _body(self, items, bufs, fold: bool, rec):
        import engine.model as M
        from tools import gdn_wy_kernels as G
        engs = [it[0] for it in items]
        gens = []
        for eng, b in zip(engs, bufs):
            eng.trace = M.BlockTrace()
            eng.trace.S_entry = eng.state.S
            eng.trace.conv_entry = eng.state.conv.clone()
            eng.trace.fold_par = eng._fold_par if fold else None
            eng.tree = b.tree
            eng._jg = b
            gens.append(eng._forward_steps(b.tok, 0))
        G.TABLE_REC = rec
        try:
            if rec is not None:
                rec.begin()
            done = M.joint_steps(engs[0], gens)
        finally:
            G.TABLE_REC = None
            traces = []
            for eng in engs:
                eng._jg = None
                eng.tree = None
                traces.append(eng.trace)
                eng.trace = None
        if rec is not None and rec.K != len(engs[0].cfg.linear_layers):
            raise _NotGraphable(f"{rec.K} grouped GDN launches for {len(engs[0].cfg.linear_layers)} layers")
        return done, traces

    def _capture(self, items, bufs, fold: bool) -> "_Captured":
        engs = [it[0] for it in items]
        cap = _Captured()
        conv_keep = [eng.state.conv.clone() for eng in engs]
        pn_keep = [eng._pn.clone() if fold else None for eng in engs]
        for eng in engs:
            if fold:
                eng._pn.zero_()            # the warm-up must not apply the pending commit (the replay does)
        taps0 = [eng.tap for eng in engs]
        for eng in engs:
            eng.tap = None
        try:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self._body(items, bufs, fold, None)
            torch.cuda.current_stream().wait_stream(self.stream)
            for eng, c in zip(engs, conv_keep):
                eng.state.conv.copy_(c)
            taps = [[] for _ in engs]
            for eng, t0, tl in zip(engs, taps0, taps):
                eng.tap = tl.append if t0 is not None else None
            rec = _Recorder(len(engs), engs[0].device, len(engs[0].cfg.linear_layers))
            with capture_lock.LOCK, torch.cuda.graph(cap.graph, pool=self.pool, stream=self.stream):
                done, traces = self._body(items, bufs, fold, rec)
        finally:
            for eng, t0 in zip(engs, taps0):
                eng.tap = t0
        for eng, c, pk in zip(engs, conv_keep, pn_keep):
            eng.state.conv.copy_(c)
            if pk is not None:
                eng._pn.copy_(pk)
        cap.rec = rec
        cap.var = rec.parity_table() if fold else None
        if fold and cap.var is None:
            raise _NotGraphable("a folding pass without a pending commit's buffers")
        cap.logits = done
        cap.taps = taps
        cap.layers = [tr.layers for tr in traces]
        cap.factors = [tr.factors for tr in traces]
        cap.conv_entry = [tr.conv_entry for tr in traces]
        cap.hidden_pre = [eng.hidden_pre_norm for eng in engs]
        cap.hidden_post = [eng.hidden_post_norm for eng in engs]
        self.stats["captured"] += 1
        return cap

    def report(self) -> dict:
        return dict(self.lru.report(), **{k: v for k, v in self.stats.items()})


class _NotGraphable(Exception):
    pass


class _Factors(Mapping):
    """A folding pass's factors by layer, as views of the parity's static buffers, made on access.

    A view is cut from the slot map at access time; the pass's own commit reads the buffers by parity, so
    the views only serve a caller that reads them in the same round."""

    __slots__ = ("eng", "fac", "layers", "n")

    def __init__(self, eng, fac, layers, n):
        self.eng, self.fac, self.layers, self.n = eng, fac, layers, n

    def __getitem__(self, layer):
        if layer not in self.layers:
            raise KeyError(layer)
        i, n, f = self.eng.state.slot[layer], self.n, self.fac
        return (f[0][i][None, :, :n], f[1][i][None, :, :n], f[2][i][None, :, :n])

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)


_GRAPHS: dict = {}


def graphs_for(eng) -> JointGraphs:
    """The joint graphs of the engines over `eng`'s weights (one set for every slot)."""
    key = id(eng.w)
    g = _GRAPHS.get(key)
    if g is None:
        g = _GRAPHS[key] = JointGraphs()
    return g


def forward_tree_multi(items: list) -> list:
    """The 27B's joint pass: from a graph where it qualifies (`ON`), else `engine.model.forward_tree_multi`."""
    import engine.model as M
    if ON and (len(items) > 1 or ONE) and torch.cuda.is_available():
        out = graphs_for(items[0][0]).run(items)
        if out is not None:
            return out
    return M.forward_tree_multi(items)
