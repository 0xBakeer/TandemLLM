"""SPD-249: a learned corrector of Flash-Next's coupled MTP drafts (FN_MTP_CORRECTOR, default off).

The coupled pick (engine/models/flashnext/coupled.py) drafts a seeded sampled request's depth d as argmax_c(log q_f(c) + G_t(c)) over
the draft head's top candidates, G_t the target's own keyed noise of that output position: the draft lands on the
target's draw whenever q_f and the target's filtered row p_f put the same token on top of that noise. Re-ranked with
p_f itself, a pick inside the head's top k is right exactly when the target's token is in that top k (that oracle,
2026-10-31, prose at one request: tokens a round x1.24 / 1.33 / 1.38 over the argmax draft at k 4 / 8 / 16 on the 32k
head; the coupled pick reaches x1.07).

The corrector re-scores the head's top k (k = `kc`, 4 or 8) of a keyed row with a score of the DCOCO form (draft
logit plus a learned correction; arXiv 2610.12570), on the MTP chain's own inputs only (the drafter is frozen):

    s_c = beta_d (v_c - v_0) / tau  +  kappa_d (keep_c - 1)  +  rho_d[rank c]  +  w2 . gelu(r + T[c] + C f_c)
    r   = A x + a_d                    (once a row: x the MTP layer's output, the head's input, d the draft depth)

v_c the head's logit of candidate c (T 1, descending; v_0 the top), tau = FN_MTP_COUPLED_TAU x the request's
temperature, keep_c the request's filter on the coupled row (the coupled pick's top-k and top-p), T a table over the hot ids (a
candidate's frozen input embedding and head row through a learned projection, folded into one row a token when the
file is written), f_c five scalars of the head's confidence in c (log-probability, gap to the top, rank, top flag,
keep; the LiLiCorr set, arXiv 2608.20530). w2 starts at 0, beta at 1, kappa high: an untrained corrector is the
plain coupled pick over the top k. The pick is argmax_c (s_c + G_t(c)) with the same keyed noise, so the draft is still a guess of the
target's draw; every emitted token stays the target's draw, and a seeded text is bit-identical with the corrector on
or off (only proposals change). Unlike DCOCO's and LiLiCorr's argmax targets (point-mass proposals checked by
residual sampling), the pick here is Gumbel-coupled: its acceptance is
sum_i p_i / (p_i + sum_{v != i} max(p_i q_v / q_i, p_v) + P_out) for q = softmax(s) over the k and P_out the target's
mass outside them, which the training maximises; its optimum is q = p_f on the k.

On the device: the row part (one GEMV a row) in torch, then `_corr` after the coupled `_pick`, over the rows whose
parameters set the corrector flag (`coupled.keyed_for`: the knob on and the request's sampling the one the file was
trained for). Rows without the flag keep the plain coupled pair bit for bit; with the knob off nothing is launched. The reported
statistic is the file's (`stat`: "q" the head's own probability of the pick at T 1, as the coupled pick; "c" the corrector's
softmax), with its own per-depth calibration for the cut (`cal`, the format of the coupled calibration file) where the file has one.

File (safetensors, `FN_MTP_CORRECTOR_FILE`; auto: ops/correctors/<model>/<profile>.corrector.safetensors, the profile
as the price files resolve it; ops/use-profile.sh fetches a profile's file to QSE_CORRECTOR_PATH from QSE_CORRECTOR_HF at
the commit QSE_CORRECTOR_REVISION and checks it against QSE_CORRECTOR_SHA256): A [hh, H] bf16, a [D, hh], T [hot, hh]
bf16, C [5, hh], w2 [hh], cp [D, 2 + kc] (beta, kappa, rho) fp32; metadata "corrector" (JSON): kc, hh, D, tau, sampling
{temperature, top_k, top_p}, stat, cal, the training record. `check` reads the file against the kernel before anything
goes to the device; a file that is missing, fails it, or was trained at another tau than FN_MTP_COUPLED_TAU leaves the
coupled drafts plain, with one log line, and `status` (the build's info, `/health` `weights.corrector`) says which and
why.
"""

from __future__ import annotations

import hashlib
import json
import os

import torch
import torch.nn.functional as F

from engine.settings import SETTINGS as _S

ON = _S.get("FN_MTP_CORRECTOR") == "1"
FILE = _S.get("FN_MTP_CORRECTOR_FILE") or "auto"
NF = 5
#: the row parameters' slot of the corrector flag (coupled.py NPAR layout: tau, top_k, top_p, sibling margin, on, here)
FLAG = 5
STAT_Q, STAT_MARGIN, STAT_C = 0, 1, 2
STATS = {"q": STAT_Q, "margin": STAT_MARGIN, "c": STAT_C}
#: the file's tensors (the module docstring's shapes)
TENSORS = ("A", "a", "T", "C", "w2", "cp")
#: `_corr` holds a [kc, hh] fp32 tile a row: the widest hidden layer and the largest tile `check` takes
HH_MAX, TILE_MAX = 1024, 16384

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as _ld
    from triton.language.random import philox_impl

    @triton.jit
    def _corr(VALS, HI, IDS, KEYS, s_key, PAR, s_par, Z, s_z, RROW, TAB, CF, W2, CPD, TOK, s_tok, PROB, s_prob, S,
              STAT: tl.constexpr, KP: tl.constexpr, KC: tl.constexpr, HH: tl.constexpr):
        r = tl.program_id(0)
        o = tl.arange(0, KP)
        tau = tl.load(PAR + r * s_par)
        top_k = tl.load(PAR + r * s_par + 1).to(tl.int32)
        top_p = tl.load(PAR + r * s_par + 2)
        sib_m = tl.load(PAR + r * s_par + 3)
        on = tl.load(PAR + r * s_par + 4)
        cflag = tl.load(PAR + r * s_par + 5)
        v = tl.load(VALS + r * KP + o)                       # the head's logits, descending
        ninf = float("-inf")
        # the request's filter on the coupled row, as coupled._pick computes it (keep_c)
        x = _ld.div_rn(v, tau)
        mx = tl.max(x, axis=0)
        keep = x > ninf
        nk = tl.sum(keep.to(tl.int32), axis=0)
        kth = tl.sum(tl.where(o == top_k - 1, x, 0.0), axis=0)
        cut = tl.where((top_k > 0) & (top_k - 1 < nk), kth, ninf)
        keep = keep & (x >= cut)
        xm = tl.where(keep, x, ninf)
        e = tl.where(keep, _ld.exp(xm - mx), 0.0)
        lse_t = mx + _ld.log(tl.sum(e, axis=0))
        cum = tl.cumsum(tl.where(keep, _ld.exp(xm - lse_t), 0.0), axis=0)
        drop = (cum > top_p) & (top_p < 1.0)
        prev = tl.cumsum(drop.to(tl.int32), axis=0) - drop.to(tl.int32)
        keep = keep & ((prev == 0) | (o == 0))
        # the k candidates and their features
        oc = tl.arange(0, KC)
        vc = tl.load(VALS + r * KP + oc)
        hi = tl.load(HI + r * KP + oc)
        idc = tl.load(IDS + r * KP + oc)
        keepc = tl.sum(tl.where(o[None, :] == oc[:, None], keep[None, :].to(tl.float32), 0.0), axis=1)
        v0 = tl.max(v, axis=0)
        zz = tl.load(Z + r * s_z)
        lse = v0 + _ld.log(zz)
        hh = tl.arange(0, HH)
        pre = tl.load(RROW + r * HH + hh)[None, :] + tl.load(TAB + hi.to(tl.int64)[:, None] * HH + hh[None, :]).to(tl.float32)
        rk = oc.to(tl.float32) / (KC - 1)
        pre += (vc - lse)[:, None] * tl.load(CF + hh)[None, :]
        pre += (vc - v0)[:, None] * tl.load(CF + HH + hh)[None, :]
        pre += rk[:, None] * tl.load(CF + 2 * HH + hh)[None, :]
        pre += tl.where(oc == 0, 1.0, 0.0)[:, None] * tl.load(CF + 3 * HH + hh)[None, :]
        pre += keepc[:, None] * tl.load(CF + 4 * HH + hh)[None, :]
        hid = 0.5 * pre * (1.0 + _ld.erf(pre * 0.7071067811865476))
        corr = tl.sum(hid * tl.load(W2 + hh)[None, :], axis=1)
        s = tl.load(CPD) * _ld.div_rn(vc - v0, tau) + tl.load(CPD + 1) * (keepc - 1.0) + tl.load(CPD + 2 + oc) + corr
        s = tl.where(vc > ninf, s, ninf)
        # the target's noise at the candidates' ids (coupled._pick)
        key = tl.load(KEYS + r * s_key).to(tl.uint64)
        k0 = (key & 0xFFFFFFFF).to(tl.uint32)
        k1 = ((key >> 32) & 0xFFFFFFFF).to(tl.uint32)
        j = idc.to(tl.int64)
        it = j // (4 * S)
        rem = j - it * (4 * S)
        ii = rem // S
        idx = rem - ii * S
        z = (oc * 0).to(tl.uint32)
        o0, o1, o2, o3 = philox_impl(it.to(tl.uint32), z, idx.to(tl.uint32), z, k0, k1, 10)
        w = tl.where(ii == 0, o0, tl.where(ii == 1, o1, tl.where(ii == 2, o2, o3)))
        u = w.to(tl.float32) * 2.3283064365386963e-10 + 1.1641532182693481e-10
        u = tl.where(u == 1.0, 0.0, u)
        g = -_ld.log(-_ld.log(tl.maximum(u, 1e-20)))
        sc = s + g
        best = tl.max(sc, axis=0)
        i1 = tl.min(tl.where(sc == best, oc, KC), axis=0)
        sc2 = tl.where(oc == i1, ninf, sc)
        sec = tl.max(sc2, axis=0)
        has2 = sec > ninf
        i2 = tl.min(tl.where(sc2 == sec, oc, KC), axis=0)
        i2 = tl.where(has2, i2, tl.where(i1 == 0, 1, 0))
        t1 = tl.sum(tl.where(oc == i1, idc, 0), axis=0)
        t2 = tl.sum(tl.where(oc == i2, idc, 0), axis=0)
        margin = tl.where(has2, best - sec, float("inf"))
        if STAT == 0:
            q1 = _ld.exp(tl.sum(tl.where(oc == i1, vc, 0.0), axis=0) - v0) / zz
            q2 = _ld.exp(tl.sum(tl.where(oc == i2, vc, 0.0), axis=0) - v0) / zz
            s1 = q1
            s2 = tl.minimum(q2 / tl.maximum(1.0 - q1, 1e-6), 1.0)
        elif STAT == 1:
            s1 = tl.where(has2, 1.0 - _ld.exp(-0.5 * margin), 1.0)
            s2 = tl.where(has2, _ld.exp(-0.5 * margin), 0.0)
        else:
            sm = tl.max(s, axis=0)
            es = tl.where(s > ninf, _ld.exp(s - sm), 0.0)
            ps = es / tl.sum(es, axis=0)
            c1 = tl.sum(tl.where(oc == i1, ps, 0.0), axis=0)
            c2 = tl.sum(tl.where(oc == i2, ps, 0.0), axis=0)
            s1 = c1
            s2 = tl.minimum(c2 / tl.maximum(1.0 - c1, 1e-6), 1.0)
        s2 = tl.where(has2 & (sib_m >= 0.0) & (margin <= sib_m), s2, 0.0)
        live = (on != 0.0) & (cflag != 0.0)
        tl.store(TOK + r * s_tok, t1.to(tl.int64), mask=live)
        tl.store(TOK + r * s_tok + 1, t2.to(tl.int64), mask=live)
        tl.store(PROB + r * s_prob, s1, mask=live)
        tl.store(PROB + r * s_prob + 1, s2, mask=live)

    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False


def keep_mask(v: torch.Tensor, tau: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """The request's filter on the coupled row over the head's top values `v` [B, K] (descending), as `_pick` (and
    `_corr`) compute it: x = v / tau, the top-k cut (ties kept), the top-p over the kept (the first that crosses
    stays). [B, K] bool."""
    x = v / tau.view(-1, 1)
    keep = torch.isfinite(x)
    nk = keep.sum(1, keepdim=True)
    if top_k > 0:
        kth = x[:, top_k - 1:top_k] if x.shape[1] >= top_k else x[:, -1:]
        keep &= (x >= torch.where(nk > top_k - 1, kth, torch.full_like(kth, float("-inf"))))
    if top_p < 1.0:
        xm = x.masked_fill(~keep, float("-inf"))
        lse = torch.logsumexp(xm, 1, keepdim=True)
        cum = torch.cumsum(torch.exp(xm - lse).masked_fill(~keep, 0.0), 1)
        drop = cum > top_p
        prev = torch.cumsum(drop.int(), 1) - drop.int()
        first = torch.zeros_like(keep)
        first[:, 0] = True
        keep &= (prev == 0) | first
    return keep


def features(v: torch.Tensor, lse: torch.Tensor, keep: torch.Tensor, kc: int) -> torch.Tensor:
    """f_c [B, kc, 5]: log-probability under the head at T 1, gap to the top, rank / (kc - 1), top flag, keep."""
    vc = v[:, :kc]
    rk = (torch.arange(kc, device=v.device, dtype=torch.float32) / (kc - 1)).expand_as(vc)
    top = (torch.arange(kc, device=v.device) == 0).float().expand_as(vc)
    return torch.stack([vc - lse.view(-1, 1), vc - v[:, :1], rk, top, keep[:, :kc].float()], -1)


def scores_ref(p: dict, x: torch.Tensor, v: torch.Tensor, hi: torch.Tensor, lse: torch.Tensor, keep: torch.Tensor,
               tau: torch.Tensor, depth: torch.Tensor, kc: int, table: torch.Tensor | None = None,
               drop: float = 0.0) -> torch.Tensor:
    """s [B, kc] in torch ops (the trainer's forward and the kernel's reference): `p` the parameters (A, a, T, C, w2,
    cp), x [B, H], v [B, >= kc] descending, hi [B, >= kc] hot indices, lse [B], keep [B, >= kc], tau [B], depth [B].
    `drop`: dropout on the hidden layer (training only)."""
    T = p["T"] if table is None else table
    r = F.linear(x.to(p["A"].dtype), p["A"]).float() + p["a"][depth]
    f = features(v, lse, keep, kc)
    pre = r[:, None, :] + T[hi[:, :kc].long()].float() + f @ p["C"]
    hid = F.gelu(pre)
    if drop > 0.0:
        hid = F.dropout(hid, p=drop, training=True)
    corr = hid @ p["w2"]
    cp = p["cp"][depth]
    vc = v[:, :kc]
    s = cp[:, :1] * (vc - v[:, :1]) / tau.view(-1, 1) + cp[:, 1:2] * (keep[:, :kc].float() - 1.0) + cp[:, 2:2 + kc] + corr
    return s.masked_fill(~torch.isfinite(vc), float("-inf"))


def default_path(profile: str | None = None, pack: str | None = None) -> str:
    """ops/correctors/<model>/<profile>.corrector.safetensors: `profile`, else the served pack's as the price files
    resolve it (serving.py `profile_of`: QSE_WEIGHTS, else flashnext-<pack dir>), as `coupled.CalInputs.path`."""
    from engine.models.flashnext.serving import MODEL_ID, ROOT, profile_of
    prof = profile or (profile_of(pack) if pack else (os.environ.get("QSE_WEIGHTS") or "flashnext-3.05"))
    return os.path.join(ROOT, "ops", "correctors", MODEL_ID, f"{prof}.corrector.safetensors")


def _pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def check(t: dict, meta: dict, hidden: int | None = None, tau: float | None = None) -> None:
    """Raise ValueError naming the first thing in a corrector file the kernel cannot take: a missing tensor, one that
    is not floating point or holds a value that is not finite, kc not a power of 2 from 2 to the pick's KP, hh not a
    power of 2 up to HH_MAX, a [kc, hh] tile above TILE_MAX, a shape other than the metadata's (A [hh, H] with H = `hidden`, the MTP layer's output width, when
    given; a [D, hh], T [rows, hh], C [5, hh], w2 [hh], cp [D, 2 + kc]), an unknown statistic, a sampling or tau
    that is not a number, a calibration table `coupled.check_cal` refuses; with `tau` (FN_MTP_COUPLED_TAU), a file
    trained at another tau. `_corr` reads every one of these sizes from its launch arguments, so a file that passes
    is read inside its tensors."""
    from engine.models.flashnext.coupled import KP, check_cal
    miss = [k for k in TENSORS if k not in t]
    if miss:
        raise ValueError(f"no tensor {', '.join(miss)}")
    for k in TENSORS:
        if not t[k].is_floating_point():
            raise ValueError(f"{k} is {t[k].dtype}, not floating point")
        if not bool(torch.isfinite(t[k]).all()):
            raise ValueError(f"{k} holds a value that is not finite")
    try:
        kc, hh, D = int(meta["kc"]), int(meta["hh"]), int(meta["D"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"metadata kc, hh, D: {e!r}") from None
    if not (2 <= kc <= KP and _pow2(kc)):
        raise ValueError(f"kc {kc}: a power of 2 from 2 to {KP}")
    if not _pow2(hh) or hh > HH_MAX:
        raise ValueError(f"hh {hh}: a power of 2 up to {HH_MAX}")
    if kc * hh > TILE_MAX:
        raise ValueError(f"kc {kc} x hh {hh}: the kernel's tile holds at most {TILE_MAX} values")
    if D < 1:
        raise ValueError(f"D {D}: no depth")
    want = {"A": (hh, int(hidden) if hidden else (t["A"].shape[-1] if t["A"].dim() else 0)), "a": (D, hh),
            "C": (NF, hh), "w2": (hh,), "cp": (D, 2 + kc)}
    for k, shape in want.items():
        if tuple(t[k].shape) != shape:
            raise ValueError(f"{k} {list(t[k].shape)}, the metadata and the model say {list(shape)}")
    if t["T"].dim() != 2 or t["T"].shape[0] < 1 or t["T"].shape[1] != hh:
        raise ValueError(f"T {list(t['T'].shape)}, the metadata says [rows, {hh}]")
    if meta.get("stat", "q") not in STATS:
        raise ValueError(f"stat {meta.get('stat')!r}: one of {', '.join(STATS)}")
    sm = meta.get("sampling") or {}
    if not isinstance(sm, dict):
        raise ValueError(f"sampling {sm!r}: not a map")
    try:
        float(sm.get("temperature", 1.0)), int(sm.get("top_k", 20)), float(sm.get("top_p", 0.95))
        if "tau" in meta:
            float(meta["tau"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"sampling or tau: {e!r}") from None
    if tau is not None and "tau" in meta and abs(float(meta["tau"]) - float(tau)) > 1e-6:
        raise ValueError(f"trained at tau {float(meta['tau'])}, FN_MTP_COUPLED_TAU is {float(tau)}")
    cal = meta.get("cal") or {}
    if not isinstance(cal, dict):
        raise ValueError(f"cal {type(cal).__name__}: not a map")
    for hot, tab in cal.items():
        try:
            int(hot)
            check_cal(tab)
        except (TypeError, ValueError) as e:
            raise ValueError(f"cal {hot!r}: {e}") from None


class Corrector:
    """The corrector's parameters on the device and its launch (`apply`)."""

    _cache: dict = {}

    def __init__(self, t: dict, meta: dict, path: str, device="cuda"):
        self.file, self.meta = path, meta
        self.kc, self.hh, self.D = int(meta["kc"]), int(meta["hh"]), int(meta["D"])
        self.A = t["A"].to(device=device, dtype=torch.bfloat16).contiguous()
        self.a = t["a"].to(device=device, dtype=torch.float32).contiguous()
        self.T = t["T"].to(device=device, dtype=torch.bfloat16).contiguous()
        self.C = t["C"].to(device=device, dtype=torch.float32).contiguous()
        self.w2 = t["w2"].to(device=device, dtype=torch.float32).contiguous()
        self.cp = t["cp"].to(device=device, dtype=torch.float32).contiguous()
        self.sampling = meta.get("sampling") or {}
        self.stat = STATS[meta.get("stat", "q")]
        self.cal = {int(k): v for k, v in (meta.get("cal") or {}).items()}
        self._cals: dict = {}

    @classmethod
    def load(cls, path: str, device="cuda", hidden: int | None = None, tau: float | None = None) -> "Corrector":
        """The file at `path` on `device`, after `check` (ValueError: what it gets wrong; nothing on the device)."""
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            raw = (f.metadata() or {}).get("corrector")
            if raw is None:
                raise ValueError("no corrector metadata")
            meta = json.loads(raw)
            if not isinstance(meta, dict):
                raise ValueError("the corrector metadata is not a map")
            t = {k: f.get_tensor(k) for k in f.keys()}
        check(t, meta, hidden, tau)
        return cls(t, meta, path, device)

    @classmethod
    def get(cls, pack: str | None = None, hidden: int | None = None) -> "Corrector | None":
        """The served corrector: None with the knob off, or with no file, a file `check` refuses (`hidden`: the MTP
        layer's output width, the build passes it) or one trained at another tau than FN_MTP_COUPLED_TAU; a line
        says which once, and every keyed row keeps the plain coupled pick. `pack`: the served pack (the auto path's
        profile). The first call decides; the build makes it before any draft graph is captured."""
        if not ON:
            return None
        if "c" not in cls._cache:
            p = default_path(pack=pack) if FILE in ("", "auto") else FILE
            c, why, digest = None, None, None
            if not os.path.isfile(p):
                why = "no file"
                print(f"[corrector] FN_MTP_CORRECTOR: no file {p}: the coupled drafts stay plain", flush=True)
            else:
                try:
                    from engine.models.flashnext import coupled as CP
                    digest = _sha256(p)
                    c = cls.load(p, hidden=hidden, tau=CP.TAU)
                    print(f"[corrector] FN_MTP_CORRECTOR: {p}: top {c.kc}, width {c.hh}, depths {c.D}, hot rows "
                          f"{c.T.shape[0]}, stat {c.meta.get('stat', 'q')}, trained for {c.sampling}", flush=True)
                except Exception as e:                                  # noqa: BLE001 - any fault: the plain pick
                    c, why = None, "unusable: " + str(e).replace(p, os.path.basename(p))     # no paths
                    print(f"[corrector] FN_MTP_CORRECTOR: {p} unusable ({e}): the coupled drafts stay plain",
                          flush=True)
            cls._cache["c"] = c
            cls._cache["why"] = why
            cls._cache["file"] = (os.path.basename(p), digest)
        return cls._cache["c"]

    @classmethod
    def status(cls, heads: dict | None = None) -> dict:
        """What the build's info and `/health` (`weights.corrector`) say of the corrector, after `get`: loaded or not
        and why not, the file's name and sha256, its sizes and the sampling and tau it was trained for, and
        `heads` (hot ids -> whether its table covers that keyed head). No paths."""
        if not ON:
            return {"loaded": False, "why": "QWEN38_FN_MTP_CORRECTOR is off"}
        cls.get()
        c = cls._cache.get("c")
        name, digest = cls._cache.get("file") or (None, None)
        out = {"loaded": c is not None, "why": cls._cache.get("why"), "file": name, "sha256": digest}
        if c is not None:
            out.update({"kc": c.kc, "hh": c.hh, "D": c.D, "hot_rows": int(c.T.shape[0]),
                        "stat": c.meta.get("stat", "q"), "tau": c.meta.get("tau"),
                        "sampling": {k: c.sampling.get(k) for k in ("temperature", "top_k", "top_p")}})
            if heads is not None:
                out["heads"] = {str(n): bool(ok) for n, ok in heads.items()}
        return out

    def matches(self, sampler) -> bool:
        """The request samples as the file was trained for, and the coupled row is filtered like the request
        (FN_MTP_COUPLED_FILTER), as in training; else its rows keep the plain coupled pick."""
        from engine.models.flashnext import coupled as CP
        sm = self.sampling
        return (CP.FILTER and abs(float(sampler.temperature) - float(sm.get("temperature", 1.0))) < 1e-6
                and int(sampler.top_k) == int(sm.get("top_k", 20))
                and abs(float(sampler.top_p) - float(sm.get("top_p", 0.95))) < 1e-6
                and float(getattr(sampler, "min_p", 0.0) or 0.0) == 0.0)

    def calibration(self, hot: int):
        """The cut's calibrator for the corrector's statistic at a head of `hot` ids (None: the file has none)."""
        if hot not in self._cals:
            from engine.methods.staircut import DepthDeciles
            tab = self.cal.get(hot)
            self._cals[hot] = DepthDeciles(tab) if tab else None
        return self._cals[hot]

    def apply(self, v, hi, ids, keys, par, zf, tok, prob, S: int, x: torch.Tensor, depth: int) -> None:
        """The corrected pick over rows of `v` [R, KP] / `hi` (hot indices) / `ids` (token ids) whose flag is set, over
        the plain coupled pairs in `tok` / `prob`. Graph-capturable."""
        d = min(int(depth), self.D - 1)
        rrow = F.linear(x.to(torch.bfloat16), self.A).float() + self.a[d]
        R, KP = v.shape
        _corr[(R,)](v, hi, ids, keys, keys.stride(0), par, par.stride(0), zf, zf.stride(0), rrow.contiguous(), self.T,
                    self.C, self.w2, self.cp[d], tok, tok.stride(0), prob, prob.stride(0), S, STAT=self.stat, KP=KP,
                    KC=self.kc, HH=self.hh, num_warps=4)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def get(pack: str | None = None, hidden: int | None = None) -> Corrector | None:
    return Corrector.get(pack, hidden)


def status(heads: dict | None = None) -> dict:
    return Corrector.status(heads)


def flag_for(sampler) -> float:
    """The corrector flag of a keyed row's parameters (`coupled.keyed_for`)."""
    c = Corrector.get()
    return 1.0 if c is not None and c.matches(sampler) else 0.0
