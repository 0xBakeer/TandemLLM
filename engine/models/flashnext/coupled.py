"""SPD-249: coupled (keyed) MTP drafts for a seeded sampled request, over Flash-Next's draft head.

A seeded request's target emits x_t = argmax(log p_f,t + G_t) (engine/sample.py `pick_at`): p_f the filtered target
row, G_t Gumbel noise from `torch.rand(V)` after `manual_seed(_key(seed, t))`, V the target row's width. The argmax
draft proposes argmax(q), so its acceptance at a position is p_f(argmax q), never above the target's own top-1 mass.
A coupled draft proposes argmax(log q_f,t + G_t) with the SAME G_t (q_f: the draft head's row at temperature
tau x the request's, with the request's top-k and top-p), so it lands on the target's draw whenever the two
distributions put the same token on top of that noise; its acceptance goes to 1 as q_f goes to p_f
(2026-10-30, prose teacher forced: the first draft's acceptance 0.606 -> 0.675 at 32k, 0.704 at 64k). The output does not change: every emitted
token is still the target's keyed draw, so a seeded text is bit-identical to plain seeded decoding (the
`engine/sample.py` contract; the 27B's DFlash drafts this way).

The pick runs on the device inside the draft chain (the lone chain's captured graph, `MTPChainGraph` with
`keyed`, and the packed chain, `mtp_chain_multi`): the head's row (hot ids), its top `KP` logits, the request's
filter on them, the noise at their token ids (Philox at the ids, element j of the target's `torch.rand(V)`, as
`engine/keyed_noise.py` `_sparse_keyed` reads it), the best and second score. It overwrites the plain chain's
(token, probability) pair of a keyed row; the next depth is fed the coupled token. A row whose `on` flag is 0 keeps
the plain pair bit for bit. What a keyed round's host stages (`keyed_for`): the noise keys of positions
start .. start + depth - 1 (start = len(context), the walk's own `start`) and the row's parameters.

The pair's probabilities are the statistic the cut reads (StairCut: per depth the spine's and the sibling's):
`STAT_Q` the head's own probability of the coupled token at T 1 (the argmax draft's statistic when the coupled
token is the argmax) and, for the rank-2 candidate, its probability given the rank-1 missed; `STAT_MARGIN`
1 - exp(-m / 2) and exp(-m / 2) with m the score margin (rank 1 - rank 2; a lone candidate: 1 and 0). The rank-2
candidate (the depth-1 sibling) is offered only while m <= the row's sibling margin (else its probability is 0 and the cut skips it).
The head: FN_MTP_COUPLED_HOT gives keyed drafts their own hot head; with FN_MTP_COUPLED_HOT_LONE a keyed chain of one
lane (its draft on the round's critical path) reads that head's first rows and a packed chain of several lanes the
whole head (`FlashNextEngine.keyed_head`). With FN_MTP_COUPLED_CAL the round's cut inputs come from the measured file (`CalInputs`): the statistic it was
measured for, its per-depth calibration and, at one lane, its class-0 prices and draft cost.
"""

from __future__ import annotations

import json
import math
import os

import torch

from engine.settings import SETTINGS as _S

ON = _S.get("FN_MTP_COUPLED") == "1"
TAU = float(_S.get("FN_MTP_COUPLED_TAU"))
FILTER = _S.get("FN_MTP_COUPLED_FILTER") == "1"
HOT = int(_S.get("FN_MTP_COUPLED_HOT") or 0)
HOT_LONE = int(_S.get("FN_MTP_COUPLED_HOT_LONE") or 0)
SIB = _S.get("FN_MTP_COUPLED_SIB") == "1"
SIB_MARGIN = float(_S.get("FN_MTP_COUPLED_SIB_MARGIN"))
CAL = _S.get("FN_MTP_COUPLED_CAL") == "1"

#: candidates a draft row brings to the pick (the request's top-k is cut to them; a nucleus wider than them is
#: taken over them: the draft's q, not the target's, so only acceptance moves)
KP = 64
#: the row parameters' layout (`keyed_for`, the kernel): tau, top_k (0: none), top_p (1: none), sibling margin
#: (< 0: no sibling), on (0: the plain pair stays)
NPAR = 8
STAT_Q, STAT_MARGIN = 0, 1

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as _ld
    from triton.language.random import philox_impl

    @triton.jit
    def _pick(VALS, IDS, KEYS, s_key, PAR, s_par, Z, s_z, TOK, s_tok, PROB, s_prob, S,
              STAT: tl.constexpr, KP: tl.constexpr):
        r = tl.program_id(0)
        o = tl.arange(0, KP)
        tau = tl.load(PAR + r * s_par)
        top_k = tl.load(PAR + r * s_par + 1).to(tl.int32)
        top_p = tl.load(PAR + r * s_par + 2)
        sib_m = tl.load(PAR + r * s_par + 3)
        on = tl.load(PAR + r * s_par + 4)
        v = tl.load(VALS + r * KP + o)                       # the head's logits, descending
        ids = tl.load(IDS + r * KP + o)                      # their token ids
        ninf = float("-inf")
        x = _ld.div_rn(v, tau)
        mx = tl.max(x, axis=0)
        keep = x > ninf
        nk = tl.sum(keep.to(tl.int32), axis=0)
        kth = tl.sum(tl.where(o == top_k - 1, x, 0.0), axis=0)
        cut = tl.where((top_k > 0) & (top_k - 1 < nk), kth, ninf)     # the request's top-k (ties at the cut kept)
        keep = keep & (x >= cut)
        xm = tl.where(keep, x, ninf)
        e = tl.where(keep, _ld.exp(xm - mx), 0.0)
        lse = mx + _ld.log(tl.sum(e, axis=0))
        cum = tl.cumsum(tl.where(keep, _ld.exp(xm - lse), 0.0), axis=0)
        drop = (cum > top_p) & (top_p < 1.0)
        prev = tl.cumsum(drop.to(tl.int32), axis=0) - drop.to(tl.int32)
        keep = keep & ((prev == 0) | (o == 0))               # the request's top-p: the first token that crosses stays
        xm = tl.where(keep, x, ninf)
        e2 = tl.where(keep, _ld.exp(xm - mx), 0.0)
        logq = xm - mx - _ld.log(tl.sum(e2, axis=0))
        # the target's noise at the candidates' ids: element j of torch.rand(V) after manual_seed(key)
        key = tl.load(KEYS + r * s_key).to(tl.uint64)
        k0 = (key & 0xFFFFFFFF).to(tl.uint32)
        k1 = ((key >> 32) & 0xFFFFFFFF).to(tl.uint32)
        j = ids.to(tl.int64)
        it = j // (4 * S)
        rem = j - it * (4 * S)
        ii = rem // S
        idx = rem - ii * S
        z = (o * 0).to(tl.uint32)
        o0, o1, o2, o3 = philox_impl(it.to(tl.uint32), z, idx.to(tl.uint32), z, k0, k1, 10)
        w = tl.where(ii == 0, o0, tl.where(ii == 1, o1, tl.where(ii == 2, o2, o3)))
        u = w.to(tl.float32) * 2.3283064365386963e-10 + 1.1641532182693481e-10
        u = tl.where(u == 1.0, 0.0, u)
        g = -_ld.log(-_ld.log(tl.maximum(u, 1e-20)))
        sc = tl.where(keep, logq + g, ninf)
        best = tl.max(sc, axis=0)
        i1 = tl.min(tl.where(sc == best, o, KP), axis=0)
        sc2 = tl.where(o == i1, ninf, sc)
        sec = tl.max(sc2, axis=0)
        has2 = sec > ninf
        i2 = tl.min(tl.where(sc2 == sec, o, KP), axis=0)
        i2 = tl.where(has2, i2, tl.where(i1 == 0, 1, 0))     # no rank 2: the plain row's other top token
        t1 = tl.sum(tl.where(o == i1, ids, 0), axis=0)
        t2 = tl.sum(tl.where(o == i2, ids, 0), axis=0)
        margin = tl.where(has2, best - sec, float("inf"))
        if STAT == 0:
            # the head's own probability at T 1 (softmax over its row: Z, the plain chain's sum, holds the rest)
            zz = tl.load(Z + r * s_z)
            v0 = tl.max(v, axis=0)
            q1 = _ld.exp(tl.sum(tl.where(o == i1, v, 0.0), axis=0) - v0) / zz
            q2 = _ld.exp(tl.sum(tl.where(o == i2, v, 0.0), axis=0) - v0) / zz
            s1 = q1
            s2 = tl.minimum(q2 / tl.maximum(1.0 - q1, 1e-6), 1.0)
        else:
            s1 = tl.where(has2, 1.0 - _ld.exp(-0.5 * margin), 1.0)
            s2 = tl.where(has2, _ld.exp(-0.5 * margin), 0.0)
        s2 = tl.where(has2 & (sib_m >= 0.0) & (margin <= sib_m), s2, 0.0)
        live = on != 0.0
        tl.store(TOK + r * s_tok, t1.to(tl.int64), mask=live)
        tl.store(TOK + r * s_tok + 1, t2.to(tl.int64), mask=live)
        tl.store(PROB + r * s_prob, s1, mask=live)
        tl.store(PROB + r * s_prob + 1, s2, mask=live)

    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False


def stat() -> int:
    """The statistic the kernel reports: the calibration file names the one it was measured for (`CalInputs.stat`)."""
    ci = CalInputs.get()
    return ci.stat if ci is not None else STAT_Q


def pick(lg: torch.Tensor, hids, keys: torch.Tensor, par: torch.Tensor, z: torch.Tensor, tok: torch.Tensor,
         prob: torch.Tensor, V: int, kp: int = KP) -> None:
    """The coupled pick of each keyed row of `lg` [R, Vh] (the draft head's fp32 row; `hids` [Vh] its token ids, None:
    the full vocabulary), written over `tok` / `prob` [R, 2] (the plain chain's pairs) where the row's `on` is set.
    `keys` [R] int64 (a strided view is fine), `par` [R, NPAR] fp32, `z` [R] or [R, 1]: the plain chain's softmax
    sum at T 1 (sum of exp(lg - max)). Graph-capturable: no host reads."""
    from engine import keyed_noise as KN
    R = lg.shape[0]
    v, i = torch.topk(lg, kp, dim=-1, sorted=True)
    t = hids.index_select(0, i.reshape(-1)).view(i.shape) if hids is not None else i
    S, _ = KN.policy(V, lg.device)
    zf = z.reshape(R, -1)
    _pick[(R,)](v, t, keys, keys.stride(0), par, par.stride(0), zf, zf.stride(0), tok, tok.stride(0), prob,
                prob.stride(0), S, STAT=stat(), KP=kp, num_warps=2)


@torch.no_grad()
def pick_ref(lg: torch.Tensor, hids, keys: list, par: torch.Tensor, V: int, statistic: int = STAT_Q,
             kp: int = KP) -> tuple:
    """`pick` in torch ops, for the tests: the noise rows from `torch.rand` itself after `manual_seed(key)` (what
    `pick_at` draws from), the filter as `Sampler._filter` over the row's top `kp`. Returns (tok [R, 2],
    prob [R, 2], margin [R]); rows with `on` 0 are left as zeros."""
    R = lg.shape[0]
    tok = torch.zeros(R, 2, dtype=torch.long)
    prob = torch.zeros(R, 2)
    marg = torch.full((R,), float("nan"))
    for r in range(R):
        tau, top_k, top_p, sib_m, on = [float(x) for x in par[r, :5]]
        if on == 0.0:
            continue
        row = lg[r].float()
        v, i = torch.topk(row, kp, sorted=True)
        ids = hids[i] if hids is not None else i
        x = v / tau
        keep = torch.isfinite(x)
        k = int(top_k)
        if 0 < k <= int(keep.sum()):
            keep &= x >= x[k - 1]
        if top_p < 1.0:
            # `Sampler._filter`'s top-p: keep the shortest prefix whose mass exceeds top_p (the crossing token stays)
            lse = torch.logsumexp(x.masked_fill(~keep, float("-inf")), 0)
            cum = torch.cumsum(torch.exp(x - lse).masked_fill(~keep, 0.0), 0)
            drop = cum > top_p
            shifted = torch.zeros_like(drop)
            shifted[1:] = drop[:-1]
            keep &= ~shifted
        logq = torch.log_softmax(x.masked_fill(~keep, float("-inf")), 0)
        g = torch.Generator(device=lg.device)
        g.manual_seed(int(keys[r]))
        u = torch.rand(V, generator=g, device=lg.device)[ids.to(lg.device)].cpu()
        gum = -torch.log(-torch.log(u.clamp_min(1e-20)))
        sc = (logq.cpu() + gum).masked_fill(~keep.cpu(), float("-inf"))
        s2, o2 = torch.topk(sc, 2)
        has2 = bool(torch.isfinite(s2[1]))
        i1 = int(o2[0])
        i2 = int(o2[1]) if has2 else (1 if i1 == 0 else 0)
        m = float(s2[0] - s2[1]) if has2 else math.inf
        tok[r, 0], tok[r, 1] = int(ids[i1]), int(ids[i2])
        if statistic == STAT_Q:
            p = torch.softmax(row, 0)
            q1 = float(p[i[i1]])
            q2 = float(p[i[i2]])
            sp1, sp2 = q1, min(q2 / max(1.0 - q1, 1e-6), 1.0)
        else:
            sp1, sp2 = (1.0 - math.exp(-0.5 * m), math.exp(-0.5 * m)) if has2 else (1.0, 0.0)
        prob[r, 0] = sp1
        prob[r, 1] = sp2 if (has2 and sib_m >= 0.0 and m <= sib_m) else 0.0
        marg[r] = m
    return tok, prob, marg


def check_cal(tab) -> None:
    """Raise ValueError unless `tab` is a calibration table `DepthDeciles` reads in a round: kinds ("spine", "sib")
    to rows a depth, each row 10 rates in [0, 1] or None (a calibration file's `cal` entries)."""
    if not isinstance(tab, dict):
        raise ValueError(f"a table is a map of kinds, not {type(tab).__name__}")
    for kind, rows in tab.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{kind}: no rows")
        for d, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != 10:
                raise ValueError(f"{kind} depth {d}: not 10 bins")
            for v in row:
                if v is not None and not (isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0):
                    raise ValueError(f"{kind} depth {d}: {v!r} is not a rate")


def _check_doc(doc) -> None:
    """Raise ValueError unless `doc` is a calibration file `CalInputs` serves from (its docstring's layout)."""
    if not isinstance(doc, dict):
        raise ValueError(f"not a map but {type(doc).__name__}")
    if doc.get("stat", "q") not in ("q", "margin"):
        raise ValueError(f"stat {doc.get('stat')!r}: q or margin")
    cal = doc.get("cal") or {}
    if not isinstance(cal, dict):
        raise ValueError("cal is not a map")
    for hot, tab in cal.items():
        try:
            int(hot)
            check_cal(tab)
        except (TypeError, ValueError) as e:
            raise ValueError(f"cal {hot!r}: {e}") from None
    try:
        {int(k): float(v) for k, v in (doc.get("lone") or {}).items()}
        dm = doc.get("draft_ms")
        if isinstance(dm, dict):
            {str(k): float(v) for k, v in dm.items()}
        elif dm is not None:
            float(dm)
    except (AttributeError, TypeError, ValueError) as e:
        raise ValueError(f"lone or draft_ms: {e!r}") from None


def eligible(sampler) -> bool:
    """A request whose drafts couple: the knob on and the request seeded and sampling (`Sampler.coupled`)."""
    return ON and sampler is not None and bool(getattr(sampler, "coupled", False))


def keyed_for(sampler, start: int, depth: int) -> tuple | None:
    """What a keyed round stages for its chain: the noise keys of output positions start .. start + depth - 1 (the
    walk's `start` = len(context): the anchor's row draws at `start`, so draft depth d + 1 meets the target's draw at
    start + d) and the row's parameters. None: the round drafts argmax."""
    if not eligible(sampler):
        return None
    from engine.sample import _key
    seed = sampler.seed
    keys = [_key(seed, start + d) for d in range(depth)]
    par = [TAU * sampler.temperature,
           float(sampler.top_k) if FILTER else 0.0,
           float(sampler.top_p) if FILTER else 1.0,
           SIB_MARGIN if SIB else -1.0, 1.0, 0.0, 0.0, 0.0]
    return keys, par


class CalInputs:
    """SPD-249: a coupled round's cut inputs, measured for the coupled policy under the bench's sampling and
    stored next to the price file (`<profile>.<device>.coupled.json`):

        {"stat": "q" | "margin",
         "cal": {"<hot size>": {"spine": [[10 bins] a depth], "sib": [[10 bins] a depth]}},
         "lone": {"<rows>": ms}, "draft_ms": ms or {"<hot size>": ms}, "measured": "..."}

    `cal` maps the reported statistic's tenth to the measured acceptance at each depth (spine: the depth's draft
    given every draft above it right; sib: the rank-2 candidate given the rank-1 missed); `lone` and `draft_ms` the
    one-lane class-0 prices (verify + commit by rows) and the draft's cost. At more than one lane only the
    calibration is used (the per-lane joint prices lowered the rate of 8 requests at once)."""

    _cache: dict = {}

    def __init__(self, doc: dict, path: str):
        self.file = path
        self.stat = STAT_MARGIN if doc.get("stat") == "margin" else STAT_Q
        self.cal = {int(k): v for k, v in (doc.get("cal") or {}).items()}
        self.lone = {int(k): float(v) for k, v in (doc.get("lone") or {}).items()}
        self.draft_ms = doc.get("draft_ms")
        self._models: dict = {}

    @classmethod
    def path(cls, pack: str | None = None) -> str:
        """Next to the price file of the served profile (engine/models/flashnext/serving.py `prices`)."""
        from engine.methods.device import device_id
        from engine.models.flashnext.serving import MODEL_ID, ROOT, profile_of
        prof = profile_of(pack) if pack else (os.environ.get("QSE_WEIGHTS") or "flashnext-3.05")
        return os.path.join(ROOT, "ops", "prices", MODEL_ID, f"{prof}.{device_id()}.coupled.json")

    @classmethod
    def get(cls, pack: str | None = None) -> "CalInputs | None":
        """The served file's inputs: None with the knob off, no file or a file that does not read as the layout above
        (a line says which once; the cut keeps its own inputs)."""
        if not CAL:
            return None
        if "doc" not in cls._cache:
            p = cls.path(pack)
            cls._cache["doc"] = None
            if not os.path.isfile(p):
                print(f"[coupled] FN_MTP_COUPLED_CAL: no file {p}: the cut keeps its own inputs", flush=True)
            else:
                try:
                    with open(p) as fh:
                        doc = json.load(fh)
                    _check_doc(doc)
                    cls._cache["doc"] = cls(doc, p)
                except (OSError, ValueError) as e:
                    print(f"[coupled] FN_MTP_COUPLED_CAL: {p} unusable ({e}): the cut keeps its own inputs", flush=True)
        return cls._cache["doc"]

    def inputs(self, hot: int, lanes: int, cls: int = 0):
        """(calibrator, prices or None, draft ms or None) for a keyed round of a head with `hot` ids at `lanes`, context
        class `cls`: the prices and draft cost only where they were measured (one lane, class 0)."""
        from engine.methods.staircut import DepthDeciles, PriceModel
        key = (hot, lanes <= 1 and cls == 0)
        got = self._models.get(key)
        if got is None:
            tab = self.cal.get(hot) or {}
            cal = DepthDeciles(tab)
            one = key[1]
            prices = PriceModel({0: {"chain": dict(self.lone)}}, None) if one and self.lone else None
            dm = self.draft_ms.get(str(hot)) if isinstance(self.draft_ms, dict) else self.draft_ms
            got = self._models[key] = (cal, prices, dm if one else None)
        return got
