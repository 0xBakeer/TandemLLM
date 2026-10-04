"""Kolibri-1 to NVFP4 from the BF16 checkpoint, one layer at a time, with its quality gate.

No card holds the 156 GB BF16 checkpoint and none has to: GPTQ needs one layer's weights and that
layer's input statistics, and a Kolibri-1 layer is 3.1 GB. So the driver keeps the calibration
corpus as a residual stream on the device (600k tokens by 2,560 in BF16, 3.1 GB) and walks the 50
layers: read the layer, run the stream through it while summing every projection's second moment,
quantise, write the layer's file, advance. The forward is `tools/kolibri_ref.py`'s, the arithmetic
of Aleph Alpha's vLLM plugin.

What a mixture of experts changes against `tools/quant_nvfp4.py`:

  * an expert sees only the rows the router sends it, so each expert has its own second moment:
    `gate_proj` and `up_proj` share one (same input), `down_proj` has its own (`silu(g) * u`);
  * GPTQ runs on all experts at once (`tools/kolibri_nvfp4.py`, a leading expert axis);
  * the floor rule picks the method from an expert's row count n: GPTQ at n >= 1024; the clip search
    weighted by the expert's own channel statistics at 64 <= n < 1024; below 64 the clip search
    weighted by the layer's pooled statistics for gate and up (every expert reads the same normed
    hidden state) and the unweighted clip search for down. `report/L.json` lists every expert's n,
    method and errors.

What stays BF16: the router gate and its bias, every norm, the embedding. The head becomes e4m3
with one fp32 scale per vocabulary row (logits in fp32). Attention and the shared expert go to
NVFP4 with GPTQ like the experts.

The gate runs in the same job, without an engine: the held-out texts travel through every layer as
three more residual streams in fp32, one through the BF16 weights, one through the FP8 release
(dequantised with its fp32 block scales) and one through our NVFP4 weights (dequantised exactly), so
the three sets of logits are on identical tokens. Then a short greedy check decodes five prompts
from the packed NVFP4 set and scores the generations against BF16.

    python tools/kolibri_quant.py build --bf16-repo Aleph-Alpha/Kolibri-1-BF16 --local /tmp/bf16 \
        --fp8-repo Aleph-Alpha/Kolibri-1 --fp8-local /tmp/fp8 --corpus /work/<prefix>/corpus \
        --out /work/<prefix>/set [--layers 0-1] [--calib-tokens 600000] [--greedy]
    python tools/kolibri_quant.py ref --fp8-repo Aleph-Alpha/Kolibri-1 --fp8-local /tmp/fp8 \
        --seqs seqs.json --out ours.pt
    python tools/kolibri_quant.py compare --vllm vllm.pt --ours ours.pt --out crosscheck.json
    python tools/kolibri_quant.py selftest       # batched GPTQ against tools/quant_nvfp4.py (GPU)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import kolibri_nvfp4 as kn  # noqa: E402
from tools.kolibri_ref import (Ckpt, LayerW, attn_block, layer_forward, load_layer,  # noqa: E402
                               moe_block, rms)

EOS_IDS = (127906, 127901)       # <|im_end|>, <|endoftext|>
PROJ = ("gate_proj", "up_proj", "down_proj")
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


# ------------------------------------------------------------------ the checkpoints
class Fetcher:
    """Shards of a Hub repo copied to local disk in the order the layers need them, in the
    background, so reading layer L never waits on the network once the copy is ahead."""

    def __init__(self, repo: str, local: str, pool: ThreadPoolExecutor, revision: str | None = None):
        from huggingface_hub import hf_hub_download
        self.repo, self.local, self.pool, self.rev = repo, local, pool, revision
        self.dl = hf_hub_download
        os.makedirs(local, exist_ok=True)
        self.futs: dict[str, object] = {}
        self.lock = threading.Lock()
        self.bytes = 0
        self.t0 = time.time()
        for f in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
            self.dl(repo, f, local_dir=local, revision=revision)

    def _get(self, fname: str) -> str:
        p = self.dl(self.repo, fname, local_dir=self.local, revision=self.rev)
        with self.lock:
            self.bytes += os.path.getsize(p)
        return p

    def want(self, fname: str):
        with self.lock:
            if fname not in self.futs:
                self.futs[fname] = self.pool.submit(self._get, fname)
            return self.futs[fname]

    def wait(self, fname: str) -> None:
        self.want(fname).result()

    def rate(self) -> str:
        dt = max(1e-3, time.time() - self.t0)
        return f"{self.bytes / 1e9:.1f} GB in {dt:.0f} s ({self.bytes / dt / 1e6:.0f} MB/s)"


class Source(Ckpt):
    """`kolibri_ref.Ckpt` over a directory, optionally filled by a `Fetcher`."""

    def __init__(self, d: str, fetcher: Fetcher | None = None):
        super().__init__(d)
        self.fetcher = fetcher

    def _h(self, f: str):
        if f not in self.handles and self.fetcher is not None:
            self.fetcher.wait(f)
        return super()._h(f)

    def files_of_layer(self, L: int) -> list[str]:
        pre = f"model.layers.{L}."
        return sorted({f for k, f in self.map.items() if k.startswith(pre)})

    def files_outside(self) -> list[str]:
        return sorted({f for k, f in self.map.items() if not k.startswith("model.layers.")})

    def close_layer(self) -> None:
        self.handles.clear()
        self.drop_cache()


def open_source(path: str | None, repo: str | None, local: str | None, pool) -> Source | None:
    if repo:
        return Source(local, Fetcher(repo, local, pool))
    if path:
        return Source(path)
    return None


def prefetch_order(srcs: list[Source], layers: list[int]) -> None:
    """Queue every shard the run needs: the outside tensors first, then layer by layer, the
    sources interleaved so the FP8 layer arrives with the BF16 one."""
    for s in srcs:
        if s.fetcher:
            for f in s.files_outside():
                s.fetcher.want(f)
    for L in layers:
        for s in srcs:
            if s.fetcher:
                for f in s.files_of_layer(L):
                    s.fetcher.want(f)


# ------------------------------------------------------------------ helpers
def to_dtype(lw: LayerW, dtype) -> LayerW:
    t = {}
    for k, v in vars(lw).items():
        t[k] = v if k in ("gate", "bias") or not torch.is_tensor(v) else v.to(dtype)
    return LayerW(**t)


def remainder(ck: Ckpt, L: int) -> dict[str, torch.Tensor]:
    """The BF16 tensors of layer L the NVFP4 set keeps as they are (names without `model.`)."""
    p = f"model.layers.{L}."
    names = ["input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm",
             "self_attn.q_norm", "self_attn.k_norm", "mlp.gate"]
    out = {f"layers.{L}.{n}.weight": ck.raw(p + n + ".weight").to(torch.bfloat16) for n in names}
    out[f"layers.{L}.moe.router.expert_bias"] = ck.raw(p + "moe.router.expert_bias")
    return out


def nv_dense(codes, scale, s2) -> torch.Tensor:
    return kn.dequant(codes, scale, s2, torch.float32)


class PackedLayer(LayerW):
    """A layer of the written NVFP4 set on the device: experts stay packed and are dequantised
    when the router picks them (the greedy check); everything else is dense fp32."""

    def expert(self, e: int):
        return tuple(nv_dense(self.pc[i][e], self.ps[i][e], self.p2[i][e]) for i in range(3))


def read_layer_file(path: str, L: int, c: dict, device, packed: bool) -> LayerW:
    """Layer L of the written set as fp32 dense weights (the gate stream) or packed experts."""
    E = c["num_experts"]
    with safe_open(path, framework="pt", device=str(device)) as f:
        g = f.get_tensor
        keys = set(f.keys())

        def nv(base):
            """Dense fp32 of one projection: NVFP4, or FP8 with fp32 128x128 block scales (the final
            set keeps some attention and shared-expert projections in the release's format)."""
            if base + ".weight_scale_inv" in keys:
                w, sc = g(base + ".weight"), g(base + ".weight_scale_inv").float()
                n, k = w.shape
                sc = sc.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
                return w.float() * sc
            return nv_dense(g(base + ".weight"), g(base + ".weight_scale"), g(base + ".weight_scale_2"))
        t = {}
        rem = {"n_in": "input_layernorm", "n_pa": "post_attn_norm", "n_pal": "post_attention_layernorm",
               "n_pf": "post_ffn_norm", "qn": "self_attn.q_norm", "kn": "self_attn.k_norm"}
        for k, n in rem.items():
            t[k] = g(f"layers.{L}.{n}.weight").float()
        t["gate"] = g(f"layers.{L}.mlp.gate.weight").float()
        t["bias"] = g(f"layers.{L}.moe.router.expert_bias").float()
        for k, n in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj")):
            t[k] = nv(f"layers.{L}.self_attn.{n}")
        for k, n in (("sg", "gate_proj"), ("su", "up_proj"), ("sd", "down_proj")):
            t[k] = nv(f"layers.{L}.mlp.shared_experts.{n}")
        if packed:
            pc, ps, p2 = [], [], []
            for n in PROJ:
                pc.append(torch.stack([g(f"layers.{L}.mlp.experts.{e}.{n}.weight") for e in range(E)]))
                ps.append(torch.stack([g(f"layers.{L}.mlp.experts.{e}.{n}.weight_scale") for e in range(E)]))
                p2.append(torch.stack([g(f"layers.{L}.mlp.experts.{e}.{n}.weight_scale_2") for e in range(E)]))
            return PackedLayer(pc=pc, ps=ps, p2=p2, **t)
        for k, n in (("eg", "gate_proj"), ("eu", "up_proj"), ("ed", "down_proj")):
            first = nv(f"layers.{L}.mlp.experts.0.{n}")
            st = torch.empty((E,) + tuple(first.shape), dtype=torch.float32, device=device)
            st[0] = first
            for e in range(1, E):
                st[e] = nv(f"layers.{L}.mlp.experts.{e}.{n}")
            t[k] = st
    return LayerW(**t)


def chunked_rms(x, w, eps, rows: int = 65536):
    out = torch.empty_like(x)
    for i in range(0, x.shape[0], rows):
        out[i:i + rows] = rms(x[i:i + rows], w, eps)
    return out


def load_corpus(d: str, total: int, mix: dict[str, float], long_every: int = 7):
    """The calibration tokens: each split cut to its share of `total`, then into sequences of
    2,048 tokens with every `long_every`-th one 4,096 (a quarter of the tokens at the default), so
    the sliding layers see full windows and the full layers long range."""
    seqs = []
    for name, frac in mix.items():
        arr = np.load(os.path.join(d, f"calib-{name}.npy")).astype(np.int64)
        arr = arr[: int(total * frac)]
        pos, i = 0, 0
        while pos < arr.size:
            n = 4096 if (i % long_every) == long_every - 1 else 2048
            piece = arr[pos:pos + n]
            if piece.size >= 64:
                seqs.append((name, torch.from_numpy(piece.copy())))
            pos += n
            i += 1
    return seqs


def gate_texts(d: str, gate_tokens: int, eval_tokens: int) -> list[tuple[str, torch.Tensor]]:
    out = []
    for n in ("prose", "code", "de"):
        p = os.path.join(d, f"heldout-{n}.npy")
        if os.path.isfile(p):
            out.append((f"heldout-{n}", torch.from_numpy(np.load(p).astype(np.int64)[:gate_tokens])))
    if eval_tokens:
        for n in ("code", "en", "de"):
            p = os.path.join(d, f"eval-{n}.npy")
            if os.path.isfile(p):
                out.append((f"eval-{n}", torch.from_numpy(np.load(p).astype(np.int64)[:eval_tokens])))
    return out


# ------------------------------------------------------------------ the calibration pass
class Stats:
    """Second moments of every projection input of one layer, over the calibration stream."""

    def __init__(self, c, dev, experts: bool = True):
        H, E, Fi = c["hidden_size"], c["num_experts"], c["moe_intermediate_size"]
        self.experts = experts
        if not experts:      # attention only (`cmd_attn`): no 10 GB of expert second moments
            E = Fi = 1
        A = c["num_attention_heads"] * c["head_dim"]
        z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=dev)  # noqa: E731
        self.attn, self.o, self.pool = z(H, H), z(A, A), z(H, H)
        self.sd = z(c["shared_expert_intermediate_size"], c["shared_expert_intermediate_size"])
        self.gu, self.d = z(E, H, H), z(E, Fi, Fi)
        self.n = torch.zeros(E, dtype=torch.long)
        self.tokens = 0


def calib_layer(r, segs, lw: LayerW, c, L, st: Stats | None, rows: int = 131072):
    """Advance the calibration stream r [N, H] (bf16, in place) through layer L with BF16
    weights, summing the second moments into `st` (None: advance only, for a resumed layer)."""
    eps = c["rms_norm_eps"]
    sliding = c["layer_types"][L] == "sliding_attention"

    def hook(kind, *a):
        if st is None:
            return
        if kind == "attn_in":
            xf = a[0].float()
            st.attn.addmm_(xf.T, xf)
        elif kind == "o_in":
            xf = a[0].float()
            st.o.addmm_(xf.T, xf)

    for s, n in segs:
        x = rms(r[s:s + n], lw.n_in, eps)
        r[s:s + n] += rms(attn_block(x, lw, c, sliding, hook), lw.n_pa, eps)
    N = r.shape[0]
    h2 = chunked_rms(r, lw.n_pal, eps)
    E, k = c["num_experts"], c["num_experts_per_tok"]
    ids = torch.empty(N, k, dtype=torch.long, device=r.device)
    w = torch.empty(N, k, dtype=torch.float32, device=r.device)
    for i in range(0, N, rows):
        lg = h2[i:i + rows].float() @ lw.gate.T
        ids[i:i + rows] = torch.topk(lg + lw.bias, k=k, dim=-1)[1]
        w[i:i + rows] = torch.sigmoid(lg.gather(1, ids[i:i + rows]))
    if c.get("norm_topk_prob"):
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    flat = ids.flatten()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E).cpu()
    wflat = w.flatten()
    out = torch.zeros(N, r.shape[1], dtype=torch.float32, device=r.device)
    off = 0
    for e in range(E):
        n = int(counts[e])
        if n == 0:
            continue
        sel = order[off:off + n]
        off += n
        for j in range(0, n, rows):
            sj = sel[j:j + rows]
            tok = sj // k
            x = h2[tok]
            a = F.silu(x @ lw.eg[e].T) * (x @ lw.eu[e].T)
            if st is not None and st.experts:
                xf = x.float()
                st.gu[e].addmm_(xf.T, xf)
                af = a.float()
                st.d[e].addmm_(af.T, af)
                del xf, af
            out.index_add_(0, tok, (a @ lw.ed[e].T).float() * wflat[sj, None])
    for i in range(0, N, rows):
        hc = h2[i:i + rows]
        a = F.silu(hc @ lw.sg.T) * (hc @ lw.su.T)
        if st is not None and st.experts:
            hf = hc.float()
            st.pool.addmm_(hf.T, hf)
            af = a.float()
            st.sd.addmm_(af.T, af)
            del hf, af
        out[i:i + rows] += (a @ lw.sd.T).float()
    del h2
    for i in range(0, N, rows):
        r[i:i + rows] += rms(out[i:i + rows].to(r.dtype), lw.n_pf, eps)
    if st is not None:
        if st.experts:
            st.n += counts
        st.tokens += N
    return counts


# ------------------------------------------------------------------ quantising one layer
def _s2_rows(*ws) -> torch.Tensor:
    """Per-row weight_scale_2 for matrices stacked along N: [E, sum N]."""
    return torch.cat([kn.scale_2_of(w)[..., None].expand(*w.shape[:-1]) for w in ws], dim=-1)


def _split(codes, scale, s2r, sizes):
    out, o = [], 0
    for n in sizes:
        out.append((codes[..., o:o + n, :], scale[..., o:o + n, :], s2r[..., o, 0]))
        o += n
    return out


def quant_attn(lw, Hattn, Ho, L, put, *, ratios=kn.CLIP_RATIOS, damp=0.01, act_order=False) -> dict:
    """GPTQ for the four attention projections of `lw` (any object with fp q, k, v, o): q, k and v
    share their input and are one call; o has its own. `put(base, codes, scale, s2)` receives
    each tensor. Returns the relative output error of each on the calibration rows."""
    rep: dict = {}
    Wqkv = torch.cat([lw.q, lw.k, lw.v]).float()[None]
    s2 = _s2_rows(lw.q[None].float(), lw.k[None].float(), lw.v[None].float())
    cq, sq, s2r, att = kn.gptq_batched(Wqkv, Hattn[None], s2, ratios=ratios, damp=damp, act_order=act_order)
    sizes = [lw.q.shape[0], lw.k.shape[0], lw.v.shape[0]]
    deq = nv_dense(cq[0], sq[0], s2r[0])
    o = 0
    for name, n, (cc, ss, s2v) in zip(ATTN[:3], sizes, _split(cq[0], sq[0], s2r[0], sizes)):
        put(f"layers.{L}.self_attn.{name}", cc, ss, s2v)
        rep[name] = {"calib": float(kn.rel_output_error(Wqkv[:, o:o + n], deq[None, o:o + n], Hattn[None])[0])}
        o += n
    Wo = lw.o.float()[None]
    co, so, s2o, att_o = kn.gptq_batched(Wo, Ho[None], ratios=ratios, damp=damp, act_order=act_order)
    put(f"layers.{L}.self_attn.o_proj", co[0], so[0], s2o[0, 0, 0])
    rep["o_proj"] = {"calib": float(kn.rel_output_error(Wo, nv_dense(co, so, s2o), Ho[None])[0])}
    rep["damping_attempts"] = att + att_o
    return rep


def quantise_layer(lw: LayerW, st: Stats, c, L, args, log) -> tuple[dict, dict]:
    """GPTQ/clip for every projection of layer L. Returns (tensors to write, report)."""
    dev = lw.q.device
    E = c["num_experts"]
    ratios = kn.CLIP_RATIOS
    out: dict[str, torch.Tensor] = {}
    rep: dict = {"layer": L, "calib_tokens": st.tokens}

    def put(base, codes, scale, s2):
        out[base + ".weight"] = codes.contiguous().cpu()
        out[base + ".weight_scale"] = scale.contiguous().cpu()
        out[base + ".weight_scale_2"] = torch.as_tensor(float(s2), dtype=torch.float32).reshape(())

    t0 = time.time()
    rep["attn"] = quant_attn(lw, st.attn, st.o, L, put, ratios=ratios, damp=args.damp)
    # the shared expert: gate/up on the pooled second moment (every token), down on its own
    Wsgu = torch.cat([lw.sg, lw.su]).float()[None]
    s2 = _s2_rows(lw.sg[None].float(), lw.su[None].float())
    cs, ss_, s2s, _ = kn.gptq_batched(Wsgu, st.pool[None], s2, ratios=ratios, damp=args.damp)
    Fs = lw.sg.shape[0]
    for name, (cc, sc, s2v) in zip(PROJ[:2], _split(cs[0], ss_[0], s2s[0], [Fs, Fs])):
        put(f"layers.{L}.mlp.shared_experts.{name}", cc, sc, s2v)
    cd, sd, s2d, _ = kn.gptq_batched(lw.sd.float()[None], st.sd[None], ratios=ratios, damp=args.damp)
    put(f"layers.{L}.mlp.shared_experts.down_proj", cd[0], sd[0], s2d[0, 0, 0])
    dsg = nv_dense(cs, ss_, s2s)
    rep["shared"] = {
        "gate_proj": float(kn.rel_output_error(Wsgu[:, :Fs], dsg[:, :Fs], st.pool[None])[0]),
        "up_proj": float(kn.rel_output_error(Wsgu[:, Fs:], dsg[:, Fs:], st.pool[None])[0]),
        "down_proj": float(kn.rel_output_error(lw.sd.float()[None], nv_dense(cd, sd, s2d), st.sd[None])[0])}
    t_dense = time.time() - t0

    # the experts, by the floor rule
    n = st.n.tolist()
    method = ["gptq" if x >= args.gptq_min else "clip-own" if x >= args.clip_min else "clip-pool" for x in n]
    pool_act = torch.diagonal(st.pool).clamp_min(0) / max(1, st.tokens)
    errs = [dict(n=n[e], method=method[e]) for e in range(E)]
    attempts_hist: dict[int, int] = {}
    t1 = time.time()
    Fi = c["moe_intermediate_size"]
    for m in ("gptq", "clip-own", "clip-pool"):
        sel_all = [e for e in range(E) if method[e] == m]
        for b0 in range(0, len(sel_all), args.expert_batch):
            sel = sel_all[b0:b0 + args.expert_batch]
            si = torch.tensor(sel, device=dev)
            Wg, Wu, Wd = lw.eg[si].float(), lw.eu[si].float(), lw.ed[si].float()
            Wgu = torch.cat([Wg, Wu], dim=1)
            s2gu = _s2_rows(Wg, Wu)
            Hgu, Hd = st.gu[si], st.d[si]
            if m == "gptq":
                cgu, sgu, s2gu_r, att = kn.gptq_batched(Wgu, Hgu, s2gu, ratios=ratios, damp=args.damp)
                cdn, sdn, s2dn_r, attd = kn.gptq_batched(Wd, Hd, ratios=ratios, damp=args.damp)
                for a_ in att + attd:
                    attempts_hist[a_] = attempts_hist.get(a_, 0) + 1
            elif m == "clip-own":
                nn_ = st.n[sel].to(dev).float()[:, None]
                act_gu = torch.diagonal(Hgu, dim1=1, dim2=2).clamp_min(0) / nn_
                act_d = torch.diagonal(Hd, dim1=1, dim2=2).clamp_min(0) / nn_
                cgu, sgu, s2gu_r = kn.clip_batched(Wgu, act_gu, s2gu, ratios=ratios)
                cdn, sdn, s2dn_r = kn.clip_batched(Wd, act_d, ratios=ratios)
            else:
                cgu, sgu, s2gu_r = kn.clip_batched(Wgu, pool_act[None].expand(len(sel), -1), s2gu,
                                                   ratios=ratios)
                cdn, sdn, s2dn_r = kn.clip_batched(Wd, None, ratios=ratios)
            dgu = nv_dense(cgu, sgu, s2gu_r)
            ddn = nv_dense(cdn, sdn, s2dn_r)
            # errors on the calibration inputs: the expert's own H where it has rows, the pooled
            # one for gate/up where it has none (down then has no input to measure on)
            Hm = torch.where((st.n[sel] > 0).to(dev)[:, None, None], Hgu, st.pool[None])
            eg_ = kn.rel_output_error(Wg, dgu[:, :Fi], Hm)
            eu_ = kn.rel_output_error(Wu, dgu[:, Fi:], Hm)
            ed_ = kn.rel_output_error(Wd, ddn, Hd)
            for j, e in enumerate(sel):
                for pi, name in enumerate(PROJ):
                    if pi < 2:
                        cc = cgu[j, pi * Fi:(pi + 1) * Fi]
                        sc = sgu[j, pi * Fi:(pi + 1) * Fi]
                        s2v = s2gu_r[j, pi * Fi, 0]
                    else:
                        cc, sc, s2v = cdn[j], sdn[j], s2dn_r[j, 0, 0]
                    put(f"layers.{L}.mlp.experts.{e}.{name}", cc, sc, s2v)
                errs[e]["calib"] = [float(eg_[j]), float(eu_[j]), float(ed_[j]) if n[e] > 0 else None]
            del Wg, Wu, Wd, Wgu, Hgu, Hd, dgu, ddn
    t_exp = time.time() - t1
    rep["experts"] = errs
    rep["expert_damping_attempts"] = attempts_hist
    cnt = np.array(n)
    rep["experts_summary"] = {
        "gptq": int((cnt >= args.gptq_min).sum()),
        "clip_own": int(((cnt >= args.clip_min) & (cnt < args.gptq_min)).sum()),
        "clip_pool": int((cnt < args.clip_min).sum()), "zero_rows": int((cnt == 0).sum()),
        "rows_median": float(np.median(cnt)), "rows_max_share": float(cnt.max() / max(1, cnt.sum())),
        "share_under_gptq_floor": float((cnt < args.gptq_min).mean()),
    }
    rep["seconds_quant"] = {"dense": t_dense, "experts": t_exp}
    log(f"[L{L}] quantised: attention+shared {t_dense:.1f} s, experts {t_exp:.1f} s; "
        f"{json.dumps(rep['experts_summary'])}")
    return out, rep


# ------------------------------------------------------------------ the gate streams
class HeldoutErr:
    """||X (W - Wq)^T||^2 and ||X W^T||^2 per projection, summed over the held-out rows that pass
    through the BF16 stream: the layer-local error on text the quantiser never saw."""

    def __init__(self, lw32: LayerW, lnv: LayerW):
        self.w, self.q = lw32, lnv
        self.acc: dict[str, list[float]] = {}

    def _add(self, key, x, W, Wq):
        y = x @ W.T
        e = x @ (W - Wq).T
        a = self.acc.setdefault(key, [0.0, 0.0, 0])
        a[0] += float(e.pow(2).sum())
        a[1] += float(y.pow(2).sum())
        a[2] += x.shape[0]

    def __call__(self, kind, *a):
        w, q = self.w, self.q
        if kind == "attn_in":
            for n in ("q", "k", "v"):
                self._add(n + "_proj", a[0], getattr(w, n), getattr(q, n))
        elif kind == "o_in":
            self._add("o_proj", a[0], w.o, q.o)
        elif kind == "expert":
            e, rows, x, act = a
            ge, ue, de = w.expert(e)
            gq, uq, dq = q.expert(e)
            self._add(f"e{e}.gate_proj", x, ge, gq)
            self._add(f"e{e}.up_proj", x, ue, uq)
            self._add(f"e{e}.down_proj", act, de, dq)
        elif kind == "shared":
            h, act = a
            self._add("shared.gate_proj", h, w.sg, q.sg)
            self._add("shared.up_proj", h, w.su, q.su)
            self._add("shared.down_proj", act, w.sd, q.sd)

    def summary(self) -> dict:
        rel = {k: v[0] / max(v[1], 1e-30) for k, v in self.acc.items()}
        ex = [v for k, v in rel.items() if k.startswith("e")]
        at = {k: v for k, v in rel.items() if k in ("q_proj", "k_proj", "v_proj", "o_proj")}
        allv = list(rel.values())
        return {"median_all": float(np.median(allv)) if allv else None,
                "median_experts": float(np.median(ex)) if ex else None,
                "p90_experts": float(np.percentile(ex, 90)) if ex else None,
                "max_experts": float(max(ex)) if ex else None,
                "experts_measured": len(ex) // 3,
                "attention": at, "attention_max": max(at.values()) if at else None,
                "shared": {k[7:]: v for k, v in rel.items() if k.startswith("shared.")},
                "per_projection": rel}


def final_logits(h, norm_w, head, eps):
    """fp32 logits of a residual stream; `head` is a fp32 [V, H] matrix or (e4m3 codes, row scale)."""
    x = rms(h, norm_w, eps).float()
    if isinstance(head, tuple):
        codes, scale = head
        out = torch.empty(x.shape[0], codes.shape[0], dtype=torch.float32, device=x.device)
        for v0 in range(0, codes.shape[0], 16384):
            Wv = codes[v0:v0 + 16384].float() * scale[v0:v0 + 16384, None]
            out[:, v0:v0 + 16384] = x @ Wv.T
        return out
    return x @ head.T


def text_metrics(logits: dict[str, torch.Tensor], ids: torch.Tensor) -> dict:
    """NLL per stream, deltas against bf16 and fp8, argmax agreement overall and on the positions
    where the reference is confident (top 1 ahead of top 2 by at least 1.0 in logits)."""
    tgt = ids[1:].to(next(iter(logits.values())).device)
    m: dict = {"tokens": int(tgt.numel())}
    top = {}
    for k, lg in logits.items():
        lp = torch.log_softmax(lg[:-1], -1)
        m[f"nll_{k}"] = float(-lp.gather(1, tgt[:, None]).mean())
        t2 = lg[:-1].topk(2, -1)
        top[k] = (t2.indices[:, 0], t2.values[:, 0] - t2.values[:, 1])
    for ref in ("bf16", "fp8"):
        if ref not in logits:
            continue
        conf = top[ref][1] >= 1.0
        m[f"confident_share_{ref}"] = float(conf.float().mean())
        for k in logits:
            if k == ref:
                continue
            same = (top[k][0] == top[ref][0]).float()
            m[f"delta_{k}_vs_{ref}"] = m[f"nll_{k}"] - m[f"nll_{ref}"]
            m[f"argmax_{k}_vs_{ref}"] = float(same.mean())
            m[f"argmax_conf_{k}_vs_{ref}"] = float(same[conf].mean()) if conf.any() else None
    return m


def topk_save(lg: torch.Tensor, k: int = 256) -> dict:
    lp = torch.log_softmax(lg.float(), -1)
    t = lp.topk(min(k, lp.shape[-1]), -1)
    return {"ids": t.indices.int().cpu(), "logprobs": t.values.half().cpu()}


# ------------------------------------------------------------------ the greedy check
def render_prompt(user: str, effort: str) -> str:
    from tools.kolibri_corpus import EFFORT
    s = ("<|im_start|>system\n# Reasoning effort\n\n" + EFFORT[effort] + "<|im_end|>\n"
         f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
    if effort == "none":
        s += "<think>\n\n</think>\n\n"
    return s


GREEDY_PROMPTS = [
    ("Erkläre in drei Sätzen, warum der Himmel blau ist.", "none"),
    ("Schreibe eine kurze E-Mail an einen Kollegen, in der du ein Treffen am Donnerstag um 14 Uhr "
     "vorschlägst.", "none"),
    ("Was ist die Hauptstadt von Deutschland, und wie viele Einwohner hat sie ungefähr?", "none"),
    ("Write a Python function that returns the n-th Fibonacci number iteratively, with a short "
     "docstring.", "none"),
    ("Wie viele Minuten hat eine Woche? Rechne kurz nach.", "low"),
]


@torch.no_grad()
def greedy_decode(layers: list[LayerW], c, emb, norm_w, head, prompts: list[list[int]], max_new: int,
                  log) -> list[list[int]]:
    """Batched greedy decoding with a KV cache per sequence and layer, eager torch."""
    eps = c["rms_norm_eps"]
    B = len(prompts)
    kv = [[{} for _ in layers] for _ in range(B)]
    nxt, pos = [], []
    for i, p in enumerate(prompts):
        r = emb[torch.tensor(p, device=emb.device)].float()
        for L, lw in enumerate(layers):
            r, _ = layer_forward(r, lw, c, L, kv=kv[i][L], q_start=0)
        nxt.append(int(final_logits(r[-1:], norm_w, head, eps).argmax(-1)))
        pos.append(len(p))
    gen = [[t] for t in nxt]
    done = [t in EOS_IDS for t in nxt]
    t0 = time.time()
    for step in range(1, max_new):
        act = [i for i in range(B) if not done[i]]
        if not act:
            break
        r = emb[torch.tensor([gen[i][-1] for i in act], device=emb.device)].float()
        for L, lw in enumerate(layers):
            sliding = c["layer_types"][L] == "sliding_attention"
            xs = rms(r, lw.n_in, eps)
            o = torch.cat([attn_block(xs[j:j + 1], lw, c, sliding, kv=kv[i][L], q_start=pos[i])
                           for j, i in enumerate(act)])
            r = r + rms(o, lw.n_pa, eps)
            y, _ = moe_block(rms(r, lw.n_pal, eps), lw, c)
            r = r + rms(y, lw.n_pf, eps)
        tok = final_logits(r, norm_w, head, eps).argmax(-1).tolist()
        for j, i in enumerate(act):
            gen[i].append(tok[j])
            pos[i] += 1
            if tok[j] in EOS_IDS:
                done[i] = True
    log(f"[greedy] {sum(len(g) for g in gen)} tokens in {time.time() - t0:.0f} s")
    return gen


def four_gram_unique(ids: list[int]) -> float:
    g = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return len(set(g)) / len(g) if g else 1.0


# ------------------------------------------------------------------ build
def parse_layers(spec: str, n: int) -> list[int]:
    if spec in ("", "all"):
        return list(range(n))
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b or a) + 1))


@torch.no_grad()
def cmd_build(args) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "layers"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "report"), exist_ok=True)
    work = args.work
    os.makedirs(os.path.join(work, "layers"), exist_ok=True)
    # the log is written locally and copied to --out after every layer: a bucket mount takes
    # whole files well and many small appends badly
    logp = os.path.join(work, "build.log")
    logf = open(logp, "a")

    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    pool = ThreadPoolExecutor(args.fetch_workers)
    bf = open_source(args.bf16, args.bf16_repo, args.local, pool)
    f8 = open_source(args.fp8, args.fp8_repo, args.fp8_local, pool)
    # the target: the weights GPTQ rounds toward and the stream the second moments come from.
    # `fp8` = the FP8 release dequantised with its fp32 block scales: the policy as trained (FP8
    # quantisation-aware RL, tech report App. I.4.1) and served, and the gate's primary reference.
    target = getattr(args, "target", "bf16")
    refs = {k: s for k, s in (("bf16", bf), ("fp8", f8)) if s is not None}
    if target not in refs:
        raise SystemExit(f"--target {target} needs that checkpoint (--{target} or --{target}-repo)")
    tgt = refs[target]
    c = json.load(open(os.path.join(tgt.dir, "config.json")))
    eps = c["rms_norm_eps"]
    layers = parse_layers(args.layers, c["num_hidden_layers"])
    full = layers == list(range(c["num_hidden_layers"]))
    srcs = list(refs.values())
    prefetch_order(srcs, layers)
    log(f"[build] layers {layers[0]}-{layers[-1]} on {dev}; target {target}; bf16 {bf.dir if bf else '-'}; "
        f"fp8 {f8.dir if f8 else '-'}; git {os.environ.get('GIT_SHA', '?')}")

    mix = {k: float(v) for k, v in (x.split(":") for x in args.mix.split(","))}
    seqs = load_corpus(args.corpus, args.calib_tokens, mix)
    segs, s0 = [], 0
    for _, ids in seqs:
        segs.append((s0, ids.numel()))
        s0 += ids.numel()
    gtexts = gate_texts(args.corpus, args.gate_tokens, args.eval_tokens)
    log(f"[corpus] {len(seqs)} calibration sequences, {s0} tokens "
        f"({', '.join(f'{k} {sum(i.numel() for n, i in seqs if n == k)}' for k in mix)}); "
        f"gate texts {[(n, int(i.numel())) for n, i in gtexts]}")

    t0 = time.time()
    emb = tgt.get("model.embed_tokens.weight", torch.bfloat16).to(dev)
    r = emb[torch.cat([i for _, i in seqs]).to(dev)]
    streams = {}
    for k, s in refs.items():
        ek = s.get("model.embed_tokens.weight", torch.float32).to(dev)
        streams[k] = [ek[i.to(dev)] for _, i in gtexts]
        del ek
    streams["nvfp4"] = [h.clone() for h in streams[target]]
    for s in srcs:
        s.close_layer()
    log(f"[embed] calibration stream {tuple(r.shape)} {r.dtype}, {r.numel() * 2 / 1e9:.2f} GB, "
        f"{time.time() - t0:.0f} s; fetch {tgt.fetcher.rate() if tgt.fetcher else 'mount'}")
    heldout_idx = [j for j, (n, _) in enumerate(gtexts) if n.startswith("heldout")]
    summary_rows = []

    for L in layers:
        tl = time.time()
        lpath_out = os.path.join(args.out, "layers", f"{L}.safetensors")
        rpath_out = os.path.join(args.out, "report", f"{L}.json")
        lpath = os.path.join(work, "layers", f"{L}.safetensors")
        done = args.resume and os.path.isfile(lpath_out) and os.path.isfile(rpath_out)
        lw = load_layer(tgt, L, c, dev, torch.bfloat16)
        t_read = time.time() - tl
        t1 = time.time()
        st = None if done else Stats(c, dev)
        calib_layer(r, segs, lw, c, L, st)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        t_fwd = time.time() - t1
        t2 = time.time()
        if done:
            shutil.copyfile(lpath_out, lpath)
            rep = json.load(open(rpath_out))
            log(f"[L{L}] resumed from {lpath_out}")
        else:
            tensors, rep = quantise_layer(lw, st, c, L, args, log)
            tensors.update(remainder(tgt, L))
            del st
            save_file(tensors, lpath, metadata={"format": "nvfp4", "group": "16", "layer": str(L),
                                                 "target": target, "source": os.path.basename(tgt.dir)})
            del tensors
        t_quant = time.time() - t2
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        # the gate streams through this layer, and the held-out layer-local error
        t3 = time.time()
        # the target's reference stream runs on its exact fp32 weights (an FP8 target dequantised
        # and cast to bf16 for the calibration would perturb the reference by 2^-9 a weight)
        lw32 = to_dtype(lw, torch.float32) if target == "bf16" else load_layer(tgt, L, c, dev, torch.float32)
        del lw
        lnv = read_layer_file(lpath, L, c, dev, packed=False)
        her = HeldoutErr(lw32, lnv)       # the layer-local error against the target's weights
        for j in range(len(gtexts)):
            streams[target][j], _ = layer_forward(streams[target][j], lw32, c, L,
                                                  hook=her if j in heldout_idx else None)
        del lw32
        for j in range(len(gtexts)):
            streams["nvfp4"][j], _ = layer_forward(streams["nvfp4"][j], lnv, c, L)
        del lnv
        for k, s in refs.items():
            if k == target:
                continue
            lo = load_layer(s, L, c, dev, torch.float32)
            for j in range(len(gtexts)):
                streams[k][j], _ = layer_forward(streams[k][j], lo, c, L)
            del lo
        for s in srcs:
            s.close_layer()
        t_gate = time.time() - t3
        rep["heldout"] = her.summary()
        t4 = time.time()
        if not done:
            shutil.copyfile(lpath, lpath_out)
        t_write = time.time() - t4
        rep["seconds"] = {"read": t_read, "calib_forward_stats": t_fwd, "quantise_and_save": t_quant,
                          "gate_streams": t_gate, "copy_to_out": t_write, "total": time.time() - tl}
        if not done:
            with open(rpath_out, "w") as f:
                json.dump(rep, f)
        ho = rep["heldout"]
        calib_med = float(np.median([x for e in rep["experts"] for x in e.get("calib", []) if x is not None]))
        summary_rows.append({"layer": L, "seconds": rep["seconds"]["total"], "heldout_median": ho["median_all"],
                             "heldout_attention_max": ho["attention_max"], "calib_median_experts": calib_med})
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        shutil.copyfile(logp, os.path.join(args.out, "build.log"))
        log(f"[L{L}] {rep['seconds']['total']:.0f} s (read {t_read:.0f}, forward+stats {t_fwd:.0f}, "
            f"quant+save {t_quant:.0f}, gate {t_gate:.0f}, copy {t_write:.0f}); held-out rel. error "
            f"median {ho['median_all']:.5f}, experts median {ho['median_experts']}, attention max "
            f"{ho['attention_max']:.5f}; calib median (experts) {calib_med:.5f}; "
            f"peak {torch.cuda.max_memory_allocated() / 2**30 if dev.type == 'cuda' else 0:.1f} GiB; "
            f"fetch {tgt.fetcher.rate() if tgt.fetcher else 'mount'}")
    del r
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    summary: dict = {"layers": summary_rows, "calib_tokens": s0, "mix": mix, "complete": full}

    if full or args.force_head:
        # the head and the final logits of the three streams
        t5 = time.time()
        wh = tgt.get("lm_head.weight", torch.bfloat16).to(dev)
        hc, hs = kn.quant_head_e4m3(wh)
        del wh
        heads = {k: (s.get("model.norm.weight", torch.float32).to(dev), s.get("lm_head.weight", torch.float32).to(dev))
                 for k, s in refs.items()}
        heads["nvfp4"] = (heads[target][0], (hc, hs))
        outside = {"embed_tokens.weight": emb.cpu(), "norm.weight": tgt.raw("model.norm.weight"),
                   "lm_head.weight": hc.cpu(), "lm_head.weight_scale": hs.cpu()}
        opath = os.path.join(work, "outside.safetensors")
        save_file(outside, opath, metadata={"format": "kolibri1-nvfp4-outside", "head": "e4m3, fp32 row scale",
                                            "target": target, "source": os.path.basename(tgt.dir)})
        shutil.copyfile(opath, os.path.join(args.out, "outside.safetensors"))
        del outside
        gate = {}
        os.makedirs(os.path.join(args.out, "gate"), exist_ok=True)
        for j, (name, ids) in enumerate(gtexts):
            lg = {k: final_logits(streams[k][j], heads[k][0], heads[k][1], eps) for k in streams}
            gate[name] = text_metrics(lg, ids)
            torch.save({"ids": ids, **{k: topk_save(v) for k, v in lg.items()}},
                       os.path.join(args.out, "gate", f"{name}.pt"))
            log(f"[gate] {name}: " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                                                for k, v in gate[name].items()}))
            del lg
        summary["gate"] = gate
        ho = [g for n, g in gate.items() if n.startswith("heldout")]
        if ho:
            for k in refs:
                summary[f"gate_worst_delta_vs_{k}"] = max(g[f"delta_nvfp4_vs_{k}"] for g in ho)
            worst = summary[f"gate_worst_delta_vs_{target}"]
            summary["gate_target"] = target
            summary["gate_pass"] = bool(worst <= 0.05)
            log(f"[gate] worst held-out delta against {target} {worst:+.4f} nats (bar +0.05): "
                f"{'PASS' if worst <= 0.05 else 'FAIL'}; "
                + ", ".join(f"vs {k} {summary[f'gate_worst_delta_vs_{k}']:+.4f}" for k in refs))
        log(f"[gate] head + logits {time.time() - t5:.0f} s")
        del streams, heads

        if args.greedy:
            summary["greedy"] = greedy_check(args, c, layers, work, emb, (hc, hs), tgt, dev, log)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"[build] done; summary -> {os.path.join(args.out, 'summary.json')}")
    shutil.copyfile(logp, os.path.join(args.out, "build.log"))
    pool.shutdown(wait=False, cancel_futures=True)


@torch.no_grad()
def greedy_check(args, c, layers, work, emb, head_q, bf: Source, dev, log) -> dict:
    """Decode the five prompts from the packed NVFP4 set; then run the BF16 model teacher-forced
    over prompt + generation and count the generated tokens that are also BF16's argmax."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(bf.dir, "tokenizer.json"))
    eps = c["rms_norm_eps"]
    t0 = time.time()
    packed = [read_layer_file(os.path.join(work, "layers", f"{L}.safetensors"), L, c, dev, packed=True)
              for L in layers]
    norm_w = bf.get("model.norm.weight", torch.float32).to(dev)
    log(f"[greedy] packed set resident in {time.time() - t0:.0f} s, "
        f"{torch.cuda.memory_allocated() / 2**30 if dev.type == 'cuda' else 0:.1f} GiB allocated")
    prompts = [tok.encode(render_prompt(u, e), add_special_tokens=False).ids for u, e in GREEDY_PROMPTS]
    gen = greedy_decode(packed, c, emb, norm_w, head_q, prompts, args.greedy_tokens, log)
    del packed
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    # BF16, teacher-forced over each prompt + generation, streamed a layer at a time
    t1 = time.time()
    full = [torch.tensor(p + g, device=dev) for p, g in zip(prompts, gen)]
    hs = [emb[x].float() for x in full]
    for L in layers:
        lw = load_layer(bf, L, c, dev, torch.float32)
        for i in range(len(hs)):
            hs[i], _ = layer_forward(hs[i], lw, c, L)
        del lw
        bf.close_layer()
    head = bf.get("lm_head.weight", torch.float32).to(dev)
    rows = []
    for i, (p, g) in enumerate(zip(prompts, gen)):
        lg = final_logits(hs[i], norm_w, head, eps)
        am = lg.argmax(-1).tolist()
        # position len(p) - 1 + j predicts generated token j
        agree = [am[len(p) - 1 + j] == g[j] for j in range(len(g))]
        rows.append({"prompt": GREEDY_PROMPTS[i][0], "effort": GREEDY_PROMPTS[i][1],
                     "tokens": len(g), "ended": g[-1] in EOS_IDS, "unique_4gram": four_gram_unique(g),
                     "bf16_argmax_share": sum(agree) / len(agree), "text": tok.decode(g)})
        log(f"[greedy] {i}: {len(g)} tokens, ended {rows[-1]['ended']}, 4-gram {rows[-1]['unique_4gram']:.3f}, "
            f"reference argmax share {rows[-1]['bf16_argmax_share']:.3f}: {rows[-1]['text'][:300]!r}")
    log(f"[greedy] reference ({os.path.basename(bf.dir)}) teacher-forced pass {time.time() - t1:.0f} s")
    ok = all(r["ended"] for r in rows) and all(r["unique_4gram"] >= 0.9 for r in rows)
    return {"rows": rows, "pass": ok}


# ------------------------------------------------------------------ which part costs the most
def stream_stats(lg: torch.Tensor, ids: torch.Tensor) -> dict:
    tgt = ids[1:].to(lg.device)
    lp = torch.log_softmax(lg[:-1].float(), -1)
    t2 = lg[:-1].topk(2, -1)
    return {"nll": float(-lp.gather(1, tgt[:, None]).mean()), "top": t2.indices[:, 0].cpu(),
            "gap": (t2.values[:, 0] - t2.values[:, 1]).cpu()}


def compare_stats(st: dict[str, dict]) -> dict:
    out: dict = {}
    for k, v in st.items():
        out[f"nll_{k}"] = v["nll"]
    for ref in ("bf16", "fp8"):
        conf = st[ref]["gap"] >= 1.0
        for k, v in st.items():
            if k == ref:
                continue
            same = (v["top"] == st[ref]["top"]).float()
            out[f"delta_{k}_vs_{ref}"] = v["nll"] - st[ref]["nll"]
            out[f"argmax_conf_{k}_vs_{ref}"] = float(same[conf].mean())
    return out


@torch.no_grad()
def cmd_mixgate(args) -> None:
    """The gate texts through the written NVFP4 set with one part at a time put back in BF16 (or
    FP8): attention, the shared expert, both, the head. The same streams as `build`'s gate, so the
    deltas say which part of the set costs how much on identical tokens."""
    dev = torch.device(args.device)
    pool = ThreadPoolExecutor(args.fetch_workers)
    bf = open_source(args.bf16, args.bf16_repo, args.local, pool)
    f8 = open_source(args.fp8, args.fp8_repo, args.fp8_local, pool)
    c = json.load(open(os.path.join(bf.dir, "config.json")))
    eps = c["rms_norm_eps"]
    nl = c["num_hidden_layers"]
    prefetch_order([bf, f8], list(range(nl)))
    gtexts = gate_texts(args.corpus, args.gate_tokens, args.eval_tokens)
    emb = bf.get("model.embed_tokens.weight", torch.float32, dev)
    emb8 = f8.get("model.embed_tokens.weight", torch.float32, dev)
    variants = ["bf16", "fp8", "nv", "nv_attn_bf16", "nv_attn_fp8", "nv_shared_bf16", "nv_experts_only"]
    hs = {v: [(emb8 if v == "fp8" else emb)[i.to(dev)] for _, i in gtexts] for v in variants}
    del emb8
    ATT = ("q", "k", "v", "o")
    SH = ("sg", "su", "sd")
    t0 = time.time()
    for L in range(nl):
        lb = load_layer(bf, L, c, dev, torch.float32)
        l8 = load_layer(f8, L, c, dev, torch.float32)
        loc = os.path.join(args.work, f"{L}.safetensors")
        os.makedirs(args.work, exist_ok=True)
        shutil.copyfile(os.path.join(args.set, "layers", f"{L}.safetensors"), loc)
        ln = read_layer_file(loc, L, c, dev, packed=False)

        def mix(attn_from=None, shared_from=None, ln=ln):
            t = dict(vars(ln))
            if attn_from is not None:
                t.update({k: getattr(attn_from, k) for k in ATT})
            if shared_from is not None:
                t.update({k: getattr(shared_from, k) for k in SH})
            return LayerW(**t)
        lws = {"bf16": lb, "fp8": l8, "nv": ln, "nv_attn_bf16": mix(lb), "nv_attn_fp8": mix(l8),
               "nv_shared_bf16": mix(None, lb), "nv_experts_only": mix(lb, lb)}
        for v in variants:
            for j in range(len(gtexts)):
                hs[v][j], _ = layer_forward(hs[v][j], lws[v], c, L)
        del lb, l8, ln, lws
        os.remove(loc)
        bf.close_layer()
        f8.close_layer()
        if L % 10 == 9:
            print(f"[mix] layer {L} {time.time() - t0:.0f} s", flush=True)
    norm = bf.get("model.norm.weight", torch.float32, dev)
    norm8 = f8.get("model.norm.weight", torch.float32, dev)
    hb = bf.get("lm_head.weight", torch.float32, dev)
    h8 = f8.get("lm_head.weight", torch.float32, dev)
    with safe_open(os.path.join(args.set, "outside.safetensors"), framework="pt", device=str(dev)) as f:
        hq = (f.get_tensor("lm_head.weight"), f.get_tensor("lm_head.weight_scale"))
    res = {}
    for j, (name, ids) in enumerate(gtexts):
        st = {}
        for v in variants:
            head = hb if v == "bf16" else h8 if v == "fp8" else hq
            st[v] = stream_stats(final_logits(hs[v][j], norm8 if v == "fp8" else norm, head, eps), ids)
            if v == "nv":
                st["nv_head_bf16"] = stream_stats(final_logits(hs[v][j], norm, hb, eps), ids)
            if v == "nv_experts_only":
                st["nv_experts_only_head_bf16"] = stream_stats(final_logits(hs[v][j], norm, hb, eps), ids)
        res[name] = compare_stats(st)
        print(f"[mix] {name}: " + "  ".join(f"{k[6:-8]} {v:+.4f}" for k, v in res[name].items()
                                           if k.startswith("delta_") and k.endswith("_vs_bf16")), flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[mix] done in {time.time() - t0:.0f} s -> {args.out}", flush=True)


# ------------------------------------------------------------------ attention again
FINE_RATIOS = tuple(round(1.0 - 0.05 * i, 2) for i in range(9))       # 1.0 .. 0.60
ATTN_VARIANTS = {
    # name: (target weights, act order, clip ratios)
    "base": ("bf16", False, kn.CLIP_RATIOS),
    "fp8t": ("fp8", False, kn.CLIP_RATIOS),            # GPTQ toward the FP8 release's weights
    "act": ("bf16", True, kn.CLIP_RATIOS),
    "fp8t-act": ("fp8", True, kn.CLIP_RATIOS),
    "fp8t-fine": ("fp8", False, FINE_RATIOS),
}


@torch.no_grad()
def cmd_attn(args) -> None:
    """Requantise attention only, in several recipes, keeping the experts and shared experts of
    an existing set (`--set`), and gate every recipe on the same streams as `build`.

    The calibration stream runs through the BF16 layers again for the attention second moments
    (all 600k tokens see every attention projection). Each recipe's attention goes to
    `<out>/attn/<recipe>/L.safetensors`. If the best recipe (lowest worst delta against BF16 on
    the eval texts) passes the bar on every gate text, its full layer files are written to
    `<out>/layers/` with the rest copied from `--set`, which is never modified."""
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    work = args.work
    os.makedirs(os.path.join(work, "layers"), exist_ok=True)
    logp = os.path.join(work, "attn.log")
    logf = open(logp, "a")

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    pool = ThreadPoolExecutor(args.fetch_workers)
    bf = open_source(args.bf16, args.bf16_repo, args.local, pool)
    f8 = open_source(args.fp8, args.fp8_repo, args.fp8_local, pool)
    c = json.load(open(os.path.join(bf.dir, "config.json")))
    eps = c["rms_norm_eps"]
    nl = c["num_hidden_layers"]
    layers = list(range(nl))
    prefetch_order([bf, f8], layers)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    log(f"[attn] recipes {variants}; set {args.set}; git {os.environ.get('GIT_SHA', '?')}")
    mix = {k: float(v) for k, v in (x.split(":") for x in args.mix.split(","))}
    seqs = load_corpus(args.corpus, args.calib_tokens, mix)
    segs, s0 = [], 0
    for _, ids in seqs:
        segs.append((s0, ids.numel()))
        s0 += ids.numel()
    gtexts = gate_texts(args.corpus, args.gate_tokens, args.eval_tokens)
    emb = bf.get("model.embed_tokens.weight", torch.bfloat16, dev)
    r = emb[torch.cat([i for _, i in seqs]).to(dev)]
    emb8 = f8.get("model.embed_tokens.weight", torch.float32, dev)
    hs = {"bf16": [emb[i.to(dev)].float() for _, i in gtexts], "fp8": [emb8[i.to(dev)] for _, i in gtexts]}
    for v in variants:
        hs[v] = [h.clone() for h in hs["bf16"]]
    del emb8
    reps: dict = {v: {} for v in variants}
    t_all = time.time()
    for L in layers:
        tl = time.time()
        lw = load_layer(bf, L, c, dev, torch.bfloat16)
        st = Stats(c, dev, experts=False)
        calib_layer(r, segs, lw, c, L, st)
        t_fwd = time.time() - tl
        l8 = load_layer(f8, L, c, dev, torch.float32)
        loc = os.path.join(work, "layers", f"{L}.safetensors")
        shutil.copyfile(os.path.join(args.set, "layers", f"{L}.safetensors"), loc)
        ln = read_layer_file(loc, L, c, dev, packed=False)
        lw32 = to_dtype(lw, torch.float32)
        del lw
        tq = time.time()
        lws = {}
        for v in variants:
            tgt, act, ratios = ATTN_VARIANTS[v]
            src = lw32 if tgt == "bf16" else l8
            tens: dict[str, torch.Tensor] = {}

            def put(base, codes, scale, s2, tens=tens):
                tens[base + ".weight"] = codes.contiguous().cpu()
                tens[base + ".weight_scale"] = scale.contiguous().cpu()
                tens[base + ".weight_scale_2"] = torch.as_tensor(float(s2), dtype=torch.float32).reshape(())
            reps[v][L] = quant_attn(src, st.attn, st.o, L, put, ratios=ratios, damp=args.damp, act_order=act)
            d = os.path.join(work, "attn", v)
            os.makedirs(d, exist_ok=True)
            save_file(tens, os.path.join(d, f"{L}.safetensors"), metadata={"format": "nvfp4", "recipe": v})
            t = dict(vars(ln))
            for k, n in zip(("q", "k", "v", "o"), ATTN):
                b = f"layers.{L}.self_attn.{n}"
                t[k] = nv_dense(tens[b + ".weight"].to(dev), tens[b + ".weight_scale"].to(dev),
                                tens[b + ".weight_scale_2"])
            lws[v] = LayerW(**t)
        t_q = time.time() - tq
        tg = time.time()
        for j in range(len(gtexts)):
            hs["bf16"][j], _ = layer_forward(hs["bf16"][j], lw32, c, L)
            hs["fp8"][j], _ = layer_forward(hs["fp8"][j], l8, c, L)
            for v in variants:
                hs[v][j], _ = layer_forward(hs[v][j], lws[v], c, L)
        del lw32, l8, ln, lws, st
        bf.close_layer()
        f8.close_layer()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        log(f"[L{L}] {time.time() - tl:.0f} s (forward+stats {t_fwd:.0f}, quant {t_q:.0f}, gate "
            f"{time.time() - tg:.0f}); calib error q/k/v/o " + "  ".join(
                f"{v} " + "/".join(f"{reps[v][L][n]['calib']:.4f}" for n in ATTN) for v in variants))
        shutil.copyfile(logp, os.path.join(args.out, "attn.log"))
    del r
    for v in variants:
        dst = os.path.join(args.out, "attn", v)
        os.makedirs(dst, exist_ok=True)
        for L in layers:
            shutil.copyfile(os.path.join(work, "attn", v, f"{L}.safetensors"), os.path.join(dst, f"{L}.safetensors"))
    norm = bf.get("model.norm.weight", torch.float32, dev)
    norm8 = f8.get("model.norm.weight", torch.float32, dev)
    h8 = f8.get("lm_head.weight", torch.float32, dev)
    hb = bf.get("lm_head.weight", torch.float32, dev)
    with safe_open(os.path.join(args.set, "outside.safetensors"), framework="pt", device=str(dev)) as f:
        hq = (f.get_tensor("lm_head.weight"), f.get_tensor("lm_head.weight_scale"))
    gate: dict = {}
    for j, (name, ids) in enumerate(gtexts):
        stt = {}
        for v in ["bf16", "fp8"] + variants:
            head = hb if v == "bf16" else h8 if v == "fp8" else hq
            stt[v] = stream_stats(final_logits(hs[v][j], norm8 if v == "fp8" else norm, head, eps), ids)
        gate[name] = compare_stats(stt)
        log(f"[gate] {name}: " + "  ".join(f"{v} {gate[name][f'delta_{v}_vs_bf16']:+.4f}/"
                                          f"{gate[name][f'delta_{v}_vs_fp8']:+.4f}" for v in variants))
    worst = {v: {"eval": max(g[f"delta_{v}_vs_bf16"] for n, g in gate.items() if n.startswith("eval")),
                 "all": max(g[f"delta_{v}_vs_bf16"] for g in gate.values()),
                 "all_vs_fp8": max(g[f"delta_{v}_vs_fp8"] for g in gate.values())} for v in variants}
    best = min(variants, key=lambda v: worst[v]["eval"])
    passed = worst[best]["all"] <= 0.05
    summary = {"recipes": {v: ATTN_VARIANTS[v][0] + (" act-order" if ATTN_VARIANTS[v][1] else "")
                           + f" ratios {list(ATTN_VARIANTS[v][2])}" for v in variants},
               "gate": gate, "worst": worst, "best": best, "pass": passed, "attn_reports": reps,
               "seconds": time.time() - t_all}
    log(f"[attn] best recipe {best} (worst eval delta {worst[best]['eval']:+.4f}); worst over all six texts "
        f"{worst[best]['all']:+.4f} against BF16, {worst[best]['all_vs_fp8']:+.4f} against FP8: "
        f"{'PASS' if passed else 'FAIL'} (bar +0.05)")
    if passed:
        os.makedirs(os.path.join(args.out, "layers"), exist_ok=True)
        from safetensors.torch import load_file
        for L in layers:
            loc = os.path.join(work, "layers", f"{L}.safetensors")
            t = load_file(loc)
            t.update(load_file(os.path.join(work, "attn", best, f"{L}.safetensors")))
            save_file(t, loc, metadata={"format": "nvfp4", "group": "16", "layer": str(L), "attention": best,
                                        "source": "Aleph-Alpha/Kolibri-1-BF16"})
            shutil.copyfile(loc, os.path.join(args.out, "layers", f"{L}.safetensors"))
        shutil.copyfile(os.path.join(args.set, "outside.safetensors"), os.path.join(args.out, "outside.safetensors"))
        log(f"[attn] set with attention {best} -> {args.out}/layers")
        if args.greedy:
            hc, hs_ = hq
            summary["greedy"] = greedy_check(args, c, layers, work, emb, (hc, hs_), bf, dev, log)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    shutil.copyfile(logp, os.path.join(args.out, "attn.log"))
    pool.shutdown(wait=False, cancel_futures=True)


# ------------------------------------------------------------------ the cross-check against vLLM
@torch.no_grad()
def cmd_ref(args) -> None:
    """Our streamed forward (fp32, FP8 release dequantised with its fp32 scales) over the token
    sequences vLLM scored, top-k per position."""
    dev = torch.device(args.device)
    if args.act_fp8:
        import tools.kolibri_ref as kr
        kr.ACT_FP8 = True
    pool = ThreadPoolExecutor(args.fetch_workers)
    src = open_source(args.fp8, args.fp8_repo, args.fp8_local, pool)
    c = json.load(open(os.path.join(src.dir, "config.json")))
    seqs = json.load(open(args.seqs))
    t0 = time.time()
    prefetch_order([src], list(range(c["num_hidden_layers"])))
    emb = src.get("model.embed_tokens.weight", torch.float32).to(dev)
    hs = [emb[torch.tensor(s["ids"], device=dev)] for s in seqs]
    del emb
    for L in range(c["num_hidden_layers"]):
        lw = load_layer(src, L, c, dev, torch.float32)
        for i in range(len(hs)):
            hs[i], _ = layer_forward(hs[i], lw, c, L)
        del lw
        src.close_layer()
        if L % 10 == 9:
            print(f"[ref] layer {L} {time.time() - t0:.0f} s", flush=True)
    norm_w = src.get("model.norm.weight", torch.float32).to(dev)
    head = src.get("lm_head.weight", torch.float32).to(dev)
    out = []
    for s, h in zip(seqs, hs):
        lg = final_logits(h, norm_w, head, c["rms_norm_eps"])
        lp = torch.log_softmax(lg, -1)
        t = lp.topk(args.k, -1)
        out.append({"name": s["name"], "ids": s["ids"], "top_ids": t.indices.cpu(),
                    "top_lp": t.values.cpu(), "gap": (lg.topk(2, -1).values.diff(dim=-1).neg()[:, 0]).cpu()})
    torch.save(out, args.out)
    print(f"[ref] {len(seqs)} sequences in {time.time() - t0:.0f} s -> {args.out}", flush=True)


def cmd_compare(args) -> None:
    """Agreement between vLLM's prompt log-probs and ours, on the confident positions (our top 1
    ahead of our top 2 by at least 1.0) and overall."""
    v = {s["name"]: s for s in torch.load(args.vllm, weights_only=False)}
    tok = None
    if args.tokenizer:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer)
    o = {s["name"]: s for s in torch.load(args.ours, weights_only=False)}
    res, tot_c, agree_c = {}, 0, 0
    for name, so in o.items():
        sv = v[name]
        T = len(so["ids"])
        ours_top = so["top_ids"][:, 0]
        vt = sv["top_ids"]           # [T, k], position t predicts token t+1, -1 where vLLM gave none
        n = min(T - 1, vt.shape[0])
        mo, mv = ours_top[:n], vt[:n, 0]
        valid = mv >= 0
        conf = (so["gap"][:n] >= 1.0) & valid
        same = (mo == mv) & valid
        row = {"positions": int(valid.sum()), "argmax_agree": float(same.sum() / max(1, valid.sum())),
               "confident": int(conf.sum()), "argmax_agree_confident": float((same & conf).sum() / max(1, conf.sum()))}
        if sv.get("gen_start") is not None:
            g0 = sv["gen_start"]
            gen_pos = torch.arange(n) >= g0 - 1
            row["generated_tokens"] = int((gen_pos & valid).sum())
            row["generated_agree"] = float((same & gen_pos).sum() / max(1, (gen_pos & valid).sum()))
        # the token after the last one: vLLM's top 5 (with its text) and ours
        row["next_top5_vllm"] = sv.get("next_top5")
        ours5 = so["top_ids"][T - 1].tolist()
        row["next_top5_ours"] = [(t, tok.decode([t]) if tok else None) for t in ours5]
        row["next_top1_same"] = bool(sv["top_ids"][T - 1, 0] == so["top_ids"][T - 1, 0])
        res[name] = row
        if name.startswith("heldout"):
            tot_c += int(conf.sum())
            agree_c += int((same & conf).sum())
    summary = {"per_seq": res, "heldout_confident": tot_c,
               "heldout_argmax_agree_confident": agree_c / max(1, tot_c)}
    summary["pass"] = summary["heldout_argmax_agree_confident"] >= 0.99
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_seq"}, indent=1))
    for k, r in res.items():
        print(k, json.dumps(r, ensure_ascii=False))


def cmd_selftest(args) -> None:
    """The batched GPTQ and clip search against `tools/quant_nvfp4.py` on real-shaped random
    matrices (needs its Triton import, so a GPU job): codes and output error per matrix."""
    from tools import quant_nvfp4 as qn
    torch.manual_seed(0)
    dev = args.device
    E, N, K, n = 4, 1024, 2560, 3000
    W = (torch.randn(E, N, K, device=dev) * 0.02).bfloat16().float()
    X = torch.randn(E, n, K, device=dev) * torch.linspace(0.2, 3, K, device=dev)
    H = torch.bmm(X.transpose(1, 2), X)
    t0 = time.time()
    cb, sb, s2b, _ = kn.gptq_batched(W, H)
    tb = time.time() - t0
    res = []
    for e in range(E):
        t1 = time.time()
        blk = qn.gptq_nvfp4(W[e], H[e])
        ts = time.time() - t1
        same = float((blk.w == cb[e]).float().mean())
        es = qn.output_error(W[e], blk, H[e])
        eb = float(kn.rel_output_error(W[e:e + 1], kn.dequant(cb[e:e + 1], sb[e:e + 1], s2b[e:e + 1]), H[e:e + 1])[0])
        res.append({"codes_same": same, "err_serial": es, "err_batched": eb, "s_serial": ts})
    clip_b = kn.clip_batched(W, torch.diagonal(H, dim1=1, dim2=2))
    clip_same = [float((qn.quantize_clipped(W[e].bfloat16(), torch.diagonal(H[e])).w == clip_b[0][e]).float().mean())
                 for e in range(E)]
    # the full build's workload shape: one batch of 64 experts' stacked gate/up and their down matrices
    del W, X, H
    B, rows = 64, 4096
    Wb = torch.randn(B, 1024, 2560, device=dev) * 0.02
    Xb = torch.randn(rows, 2560, device=dev)
    Hb = (Xb.T @ Xb)[None].repeat(B, 1, 1)
    torch.cuda.synchronize()
    t0 = time.time()
    kn.gptq_batched(Wb, Hb, inplace_h=True)
    torch.cuda.synchronize()
    t_gu = time.time() - t0
    del Wb, Hb
    Wd = torch.randn(B, 2560, 512, device=dev) * 0.02
    Xd = torch.randn(rows, 512, device=dev)
    Hd = (Xd.T @ Xd)[None].repeat(B, 1, 1)
    t0 = time.time()
    kn.gptq_batched(Wd, Hd, inplace_h=True)
    torch.cuda.synchronize()
    t_d = time.time() - t0
    print(json.dumps({"batched_seconds": tb, "per_matrix": res, "clip_codes_same": clip_same,
                      "batch64_gate_up_s": t_gu, "batch64_down_s": t_d,
                      "layer_all_gptq_estimate_s": 6 * (t_gu + t_d),
                      "peak_gib": torch.cuda.max_memory_allocated() / 2**30}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def src_args(p):
        p.add_argument("--fp8", default=None, help="FP8 release directory (mount or local)")
        p.add_argument("--fp8-repo", default=None, help="or: copy it from this repo ...")
        p.add_argument("--fp8-local", default="/tmp/fp8", help="... into this directory")
        p.add_argument("--fetch-workers", type=int, default=2)
        p.add_argument("--device", default="cuda")

    b = sub.add_parser("build")
    src_args(b)
    b.add_argument("--bf16", default=None)
    b.add_argument("--bf16-repo", default=None)
    b.add_argument("--local", default="/tmp/bf16")
    b.add_argument("--corpus", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--work", default="/tmp/kq")
    b.add_argument("--layers", default="all", help="all, or A-B")
    b.add_argument("--calib-tokens", type=int, default=600_000)
    b.add_argument("--mix", default="code:0.25,en:0.30,de:0.30,chat:0.15")
    b.add_argument("--gate-tokens", type=int, default=2048)
    b.add_argument("--eval-tokens", type=int, default=8192)
    b.add_argument("--gptq-min", type=int, default=1024)
    b.add_argument("--clip-min", type=int, default=64)
    b.add_argument("--expert-batch", type=int, default=64)
    b.add_argument("--damp", type=float, default=0.01)
    b.add_argument("--target", default="bf16", choices=["bf16", "fp8"],
                   help="the weights GPTQ rounds toward and the stream the second moments come from "
                        "(fp8: the release dequantised with its fp32 block scales, the QAT policy)")
    b.add_argument("--resume", action="store_true")
    b.add_argument("--force-head", action="store_true", help="head + gate logits even for a partial run")
    b.add_argument("--greedy", action="store_true")
    b.add_argument("--greedy-tokens", type=int, default=300)
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("ref")
    src_args(r)
    r.add_argument("--seqs", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--act-fp8", action="store_true", help="emulate vLLM's W8A8 activation rounding")
    r.set_defaults(fn=cmd_ref)

    cp = sub.add_parser("compare")
    cp.add_argument("--vllm", required=True)
    cp.add_argument("--ours", required=True)
    cp.add_argument("--out", required=True)
    cp.add_argument("--tokenizer", default=None)
    cp.set_defaults(fn=cmd_compare)

    m = sub.add_parser("mixgate", help="the gate with one part of the set back in BF16/FP8")
    src_args(m)
    m.add_argument("--bf16", default=None)
    m.add_argument("--bf16-repo", default=None)
    m.add_argument("--local", default="/tmp/bf16")
    m.add_argument("--set", required=True, help="the build's --out (layers/, outside.safetensors)")
    m.add_argument("--corpus", required=True)
    m.add_argument("--work", default="/tmp/mix")
    m.add_argument("--gate-tokens", type=int, default=2048)
    m.add_argument("--eval-tokens", type=int, default=8192)
    m.add_argument("--out", required=True)
    m.set_defaults(fn=cmd_mixgate)

    at = sub.add_parser("attn", help="requantise attention only, several recipes, gate each")
    src_args(at)
    at.add_argument("--bf16", default=None)
    at.add_argument("--bf16-repo", default=None)
    at.add_argument("--local", default="/tmp/bf16")
    at.add_argument("--set", required=True, help="the existing set (its experts are kept)")
    at.add_argument("--corpus", required=True)
    at.add_argument("--out", required=True, help="a NEW directory; --set is never written")
    at.add_argument("--work", default="/tmp/ka")
    at.add_argument("--variants", default="fp8t,act,fp8t-act,fp8t-fine")
    at.add_argument("--calib-tokens", type=int, default=600_000)
    at.add_argument("--mix", default="code:0.25,en:0.30,de:0.30,chat:0.15")
    at.add_argument("--gate-tokens", type=int, default=2048)
    at.add_argument("--eval-tokens", type=int, default=8192)
    at.add_argument("--damp", type=float, default=0.01)
    at.add_argument("--greedy", action="store_true")
    at.add_argument("--greedy-tokens", type=int, default=300)
    at.set_defaults(fn=cmd_attn)

    s = sub.add_parser("selftest")
    s.add_argument("--device", default="cuda")
    s.set_defaults(fn=cmd_selftest)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
