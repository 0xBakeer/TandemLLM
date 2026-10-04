"""A plain PyTorch forward of Kolibri-1, one layer at a time, on the CPU or one GPU.

Kolibri-1 has no modeling code in transformers; its only implementation is Aleph Alpha's vLLM plugin
(`aleph-alpha-inference`, `aleph_alpha_inference/kolibri1.py`, v1.0.0). This file is that forward
written out in tensors, so that the engine has a reference it can run beside it, and so that a
checkpoint too large for the board (BF16, 156 GB) can still be read: weights are loaded one layer at
a time, every sequence of the run passes that layer, and the layer is freed before the next one.

What the plugin does, and what this file does the same way:

- each layer: r += post_attn_norm(attn(input_layernorm(r))); r += post_ffn_norm(moe(post_attention_layernorm(r)))
  (sandwich norms; RMSNorm is x * rsqrt(mean(x^2) + eps) * w);
- attention: GQA 48/4, head 128, per-head RMSNorm of q and k, then RoPE (neox, base 10,000) on the
  sliding-window layers only; full-attention layers have no position encoding at all; a sliding layer
  sees the 512 preceding tokens and the current one (`sliding_window` 513); scale 1/sqrt(128);
- the router: fp32 logits; the top 6 by logits + expert_bias; each chosen expert weighted by
  sigmoid(logit), unbiased, not renormalised (`norm_topk_prob: false`), no scaling factor;
- the shared expert (SwiGLU, width 512) is ungated and added to the routed sum;
- the head in fp32 (`head_dtype: float32`).

The checkpoint may be the BF16 release or the FP8 one (e4m3 codes with an fp32 `weight_scale_inv`
per 128 by 128 block); FP8 is dequantised on load. Page cache is dropped after each layer's shard
reads (posix_fadvise), because on a DGX Spark the page cache and the GPU share one memory pool.

    python tools/kolibri_ref.py --model DIR --text a.txt --text b.txt [--max-tokens 1024] \
        [--device cpu] [--dtype float32] [--union] [--out result.json] [--save-logits DIR]

It reports, per text: the teacher-forced loss in nats a token, the argmax agreement with the next
token, the top-5 at the last position, and with `--union` the routed experts a layer touches for
windows of 1 to 32 consecutive tokens (the bytes a speculative verify reads) against the uniform
expectation, and the tokens each expert received (the calibration coverage a quantiser gets).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from safetensors import safe_open


class Ckpt:
    """Tensors by name over a sharded safetensors checkpoint, FP8 pairs dequantised."""

    def __init__(self, d: str):
        self.dir = d
        idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
        self.map = idx["weight_map"]
        self.handles: dict[str, object] = {}
        self.touched: set[str] = set()

    def _h(self, f: str):
        if f not in self.handles:
            self.handles[f] = safe_open(os.path.join(self.dir, f), framework="pt")
        self.touched.add(f)
        return self.handles[f]

    def has(self, name: str) -> bool:
        return name in self.map

    def raw(self, name: str) -> torch.Tensor:
        return self._h(self.map[name]).get_tensor(name)

    def get(self, name: str, dtype: torch.dtype, device=None) -> torch.Tensor:
        """The tensor as `dtype`; FP8 pairs dequantised with their fp32 block scales (on `device`
        when given, which is much faster on a GPU than on the CPU)."""
        w = self.raw(name)
        if device is not None:
            w = w.to(device)
        s = name[: -len("weight")] + "weight_scale_inv" if name.endswith("weight") else None
        if w.dtype == torch.float8_e4m3fn and s and self.has(s):
            sc = self.raw(s).to(w.device).float()
            n, k = w.shape
            sc = sc.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
            return (w.float() * sc).to(dtype)
        return w.to(dtype)

    def drop_cache(self) -> None:
        """Forget the pages read so far, so the reads do not crowd the GPU's memory."""
        for f in self.touched:
            try:
                fd = os.open(os.path.join(self.dir, f), os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except (OSError, AttributeError):
                pass
        self.touched.clear()


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * w.float()).to(x.dtype)


def rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """Neox-style rotary over the whole head (vLLM's get_rope default)."""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d))
    f = pos.float()[:, None] * inv[None, :]
    cos, sin = torch.cat([f.cos()] * 2, -1), torch.cat([f.sin()] * 2, -1)
    xf = x.float()
    x1, x2 = xf[..., : d // 2], xf[..., d // 2 :]
    rot = torch.cat([-x2, x1], -1)
    return (xf * cos[:, None, :] + rot * sin[:, None, :]).to(x.dtype)


def attention(h, L, c, ck, dt, sliding: bool):
    T = h.shape[0]
    nq, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    p = f"model.layers.{L}.self_attn."
    q = (h @ ck.get(p + "q_proj.weight", dt).T).view(T, nq, hd)
    k = (h @ ck.get(p + "k_proj.weight", dt).T).view(T, nkv, hd)
    v = (h @ ck.get(p + "v_proj.weight", dt).T).view(T, nkv, hd)
    q = rms(q, ck.get(p + "q_norm.weight", dt), c["rms_norm_eps"])
    k = rms(k, ck.get(p + "k_norm.weight", dt), c["rms_norm_eps"])
    pos = torch.arange(T, device=h.device)
    if sliding:
        q, k = rope(q, pos, c["rope_theta"]), rope(k, pos, c["rope_theta"])
    i, j = pos[:, None], pos[None, :]
    allowed = j <= i
    if sliding:
        allowed &= j >= i - (c["sliding_window"] - 1)
    rep = nq // nkv
    k = k.repeat_interleave(rep, 1)
    v = v.repeat_interleave(rep, 1)
    o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                                       attn_mask=allowed[None], scale=hd ** -0.5)
    o = o.transpose(0, 1).reshape(T, nq * hd)
    return o @ ck.get(p + "o_proj.weight", dt).T


def swiglu(x, pre, ck, dt):
    g = x @ ck.get(pre + "gate_proj.weight", dt).T
    u = x @ ck.get(pre + "up_proj.weight", dt).T
    return (F.silu(g) * u) @ ck.get(pre + "down_proj.weight", dt).T


def moe(h, L, c, ck, dt):
    p = f"model.layers.{L}.mlp."
    gate = ck.get(p + "gate.weight", torch.float32)
    bias = ck.raw(f"model.layers.{L}.moe.router.expert_bias").float()
    logits = h.float() @ gate.T
    ids = torch.topk(logits + bias, k=c["num_experts_per_tok"], dim=-1)[1]
    w = torch.sigmoid(logits.gather(1, ids))
    if c.get("norm_topk_prob"):
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    out = torch.zeros(h.shape, dtype=torch.float32, device=h.device)
    for e in torch.unique(ids).tolist():
        rows, slot = (ids == e).nonzero(as_tuple=True)
        y = swiglu(h[rows], p + f"experts.{e}.", ck, dt)
        out.index_add_(0, rows, y.float() * w[rows, slot, None])
    out += swiglu(h, p + "shared_experts.", ck, dt).float()
    return out.to(h.dtype), ids, logits


# ------------------------------------------------------------------ one layer, weights resident
# The functions above read every tensor from the checkpoint as they go. A quantiser holds a
# layer on the device instead, in three versions at once (BF16,
# FP8 dequantised, NVFP4 dequantised), and needs to see each projection's input. `LayerW` is one
# layer's weights in one dtype; `layer_forward` is the same forward as `attention` + `moe` above,
# written over it, with a hook that is handed every projection's input and the router's choices.

# vLLM serves the FP8 release as W8A8: `activation_scheme: dynamic` quantises the input of every
# FP8 linear and expert to e4m3 per token and per group of 128 (scale amax / 448, fp32), as the
# QAT trainer did. The forward below keeps activations in full precision unless ACT_FP8 is set,
# which emulates that rounding (the router gate, not converted in the release, stays exact).
ACT_FP8 = False


def act_q(x: torch.Tensor, group: int = 128) -> torch.Tensor:
    """Per-token, per-128-group dynamic e4m3 quantise-dequantise (vLLM's per_token_group_quant_fp8)."""
    if not ACT_FP8:
        return x
    T, K = x.shape
    g = x.float().reshape(T, K // group, group)
    s = (g.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-10)
    return ((g / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s).reshape(T, K).to(x.dtype)


class LayerW:
    """One layer's weights, resident. Experts are stacked: eg/eu [E, F, H], ed [E, H, F], unless
    a subclass overrides `expert(e)` (the packed NVFP4 set at decode)."""

    NAMES = {"n_in": "input_layernorm", "n_pa": "post_attn_norm", "n_pal": "post_attention_layernorm",
             "n_pf": "post_ffn_norm", "q": "self_attn.q_proj", "k": "self_attn.k_proj",
             "v": "self_attn.v_proj", "o": "self_attn.o_proj", "qn": "self_attn.q_norm",
             "kn": "self_attn.k_norm", "gate": "mlp.gate", "sg": "mlp.shared_experts.gate_proj",
             "su": "mlp.shared_experts.up_proj", "sd": "mlp.shared_experts.down_proj"}

    def __init__(self, **t):
        for k, v in t.items():
            setattr(self, k, v)

    def expert(self, e: int):
        return self.eg[e], self.eu[e], self.ed[e]


def load_layer(ck: Ckpt, L: int, c: dict, device, dtype=torch.float32, experts: bool = True) -> LayerW:
    """Layer L of a checkpoint (BF16, or FP8 dequantised with its fp32 block scales) as `dtype` on
    `device`. The router gate and its bias are fp32 whatever `dtype` is, as in the plugin."""
    p = f"model.layers.{L}."
    t = {k: ck.get(p + n + ".weight", dtype, device) for k, n in LayerW.NAMES.items() if k != "gate"}
    t["gate"] = ck.get(p + "mlp.gate.weight", torch.float32, device)
    t["bias"] = ck.raw(p + "moe.router.expert_bias").float().to(device)
    if experts:
        E = c["num_experts"]
        for k, proj in (("eg", "gate_proj"), ("eu", "up_proj"), ("ed", "down_proj")):
            first = ck.get(p + f"mlp.experts.0.{proj}.weight", dtype, device)
            st = torch.empty((E,) + tuple(first.shape), dtype=dtype, device=device)
            st[0] = first
            for e in range(1, E):
                st[e] = ck.get(p + f"mlp.experts.{e}.{proj}.weight", dtype, device)
            t[k] = st
    return LayerW(**t)


def attn_core(q, k, v, sliding: bool, window: int, q_start: int = 0, qblock: int = 1024):
    """Causal GQA over keys at positions 0..Tk-1 for queries at q_start..q_start+Tq-1, a sliding
    layer seeing its `window` rows (the query and the window - 1 before it). Query blocks of
    `qblock` keep the mask and the scores small on long texts. q [Tq, nq, hd], k/v [Tk, nkv, hd]."""
    Tq, nq, hd = q.shape
    rep = nq // k.shape[1]
    out = torch.empty_like(q)
    for i0 in range(0, Tq, qblock):
        i1 = min(Tq, i0 + qblock)
        a0, a1 = q_start + i0, q_start + i1
        k0 = max(0, a0 - (window - 1)) if sliding else 0
        pq = torch.arange(a0, a1, device=q.device)[:, None]
        pk = torch.arange(k0, a1, device=q.device)[None, :]
        allowed = pk <= pq
        if sliding:
            allowed &= pk >= pq - (window - 1)
        kk = k[k0:a1].transpose(0, 1).repeat_interleave(rep, 0)
        vv = v[k0:a1].transpose(0, 1).repeat_interleave(rep, 0)
        o = F.scaled_dot_product_attention(q[i0:i1].transpose(0, 1), kk, vv, attn_mask=allowed[None],
                                           scale=hd ** -0.5)
        out[i0:i1] = o.transpose(0, 1)
    return out


def attn_block(x, lw: LayerW, c: dict, sliding: bool, hook=None, kv: dict | None = None,
               q_start: int = 0):
    """x: the normed input [T, H]. With `kv` (a dict per layer and sequence) the keys and values
    are appended to it and the queries attend over everything in it (decode)."""
    T = x.shape[0]
    nq, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    eps = c["rms_norm_eps"]
    if hook:
        hook("attn_in", x)
    xq = act_q(x)
    q = rms((xq @ lw.q.T).view(T, nq, hd), lw.qn, eps)
    k = rms((xq @ lw.k.T).view(T, nkv, hd), lw.kn, eps)
    v = (xq @ lw.v.T).view(T, nkv, hd)
    if sliding:
        pos = torch.arange(q_start, q_start + T, device=x.device)
        q, k = rope(q, pos, c["rope_theta"]), rope(k, pos, c["rope_theta"])
    if kv is not None:
        if "k" in kv:
            k = torch.cat([kv["k"], k])
            v = torch.cat([kv["v"], v])
        kv["k"], kv["v"] = k, v
    o = attn_core(q, k, v, sliding, c["sliding_window"], q_start=q_start).reshape(T, nq * hd)
    if hook:
        hook("o_in", o)
    return act_q(o) @ lw.o.T


def route(h, lw: LayerW, c: dict):
    """The plugin's `sigmoid_logit_add_routing`: top k on fp32 logits + bias, unbiased sigmoid
    weights, renormalised only if the config says so."""
    logits = h.float() @ lw.gate.T
    ids = torch.topk(logits + lw.bias, k=c["num_experts_per_tok"], dim=-1)[1]
    w = torch.sigmoid(logits.gather(1, ids))
    if c.get("norm_topk_prob"):
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return ids, w, logits


def moe_block(h, lw: LayerW, c: dict, hook=None):
    """h: the normed input [T, H] -> (routed sum + shared expert, in h's dtype; ids)."""
    ids, w, _ = route(h, lw, c)
    if hook:
        hook("router", h, ids, w)
    out = torch.zeros(h.shape, dtype=torch.float32, device=h.device)
    for e in torch.unique(ids).tolist():
        rows, slot = (ids == e).nonzero(as_tuple=True)
        g, u, d = lw.expert(e)
        x = h[rows]
        xq = act_q(x)
        a = F.silu(xq @ g.T) * (xq @ u.T)
        if hook:
            hook("expert", e, rows, x, a)
        out.index_add_(0, rows, (act_q(a) @ d.T).float() * w[rows, slot, None])
    hq = act_q(h)
    a = F.silu(hq @ lw.sg.T) * (hq @ lw.su.T)
    if hook:
        hook("shared", h, a)
    out += (act_q(a) @ lw.sd.T).float()
    return out.to(h.dtype), ids


def layer_forward(r, lw: LayerW, c: dict, L: int, hook=None, kv: dict | None = None,
                  q_start: int = 0):
    """One decoder layer on the residual stream r [T, H]:
    r += post_attn_norm(attn(input_layernorm(r))); r += post_ffn_norm(moe(post_attention_layernorm(r)))."""
    eps = c["rms_norm_eps"]
    sliding = c["layer_types"][L] == "sliding_attention"
    r = r + rms(attn_block(rms(r, lw.n_in, eps), lw, c, sliding, hook, kv, q_start), lw.n_pa, eps)
    y, ids = moe_block(rms(r, lw.n_pal, eps), lw, c, hook)
    return r + rms(y, lw.n_pf, eps), ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", action="append", default=[], help="a text file; repeat for more")
    ap.add_argument("--prompt", action="append", default=[], help="a literal prompt; its top-5 next tokens are reported")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=0, help="stop after this many layers (0 = all); no loss then")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--union", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--save-logits", default="", help="directory for each sequence's fp32 logits (.pt)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dt = getattr(torch, a.dtype)
    dev = torch.device(a.device)
    c = json.load(open(os.path.join(a.model, "config.json")))
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.environ.get("KOLIBRI_TOKENIZER") or os.path.join(a.model, "tokenizer.json"))

    seqs = []
    for f in a.text:
        ids = tok.encode(open(f).read(), add_special_tokens=False).ids[: a.max_tokens]
        seqs.append({"name": os.path.basename(f), "ids": ids})
    for i, s in enumerate(a.prompt):
        seqs.append({"name": f"prompt{i}", "ids": tok.encode(s, add_special_tokens=False).ids, "prompt": s})
    ck = Ckpt(a.model)
    t0 = time.time()
    emb = ck.get("model.embed_tokens.weight", dt)
    hs = [emb[torch.tensor(s["ids"])].to(dev) for s in seqs]
    del emb
    ck.drop_cache()
    routes = [[] for _ in seqs]
    nl = a.layers or c["num_hidden_layers"]
    eps = c["rms_norm_eps"]
    for L in range(nl):
        p = f"model.layers.{L}."
        sliding = c["layer_types"][L] == "sliding_attention"
        n_in = ck.get(p + "input_layernorm.weight", dt).to(dev)
        n_pa = ck.get(p + "post_attn_norm.weight", dt).to(dev)
        n_pal = ck.get(p + "post_attention_layernorm.weight", dt).to(dev)
        n_pf = ck.get(p + "post_ffn_norm.weight", dt).to(dev)
        cd = _Dev(ck, dev)
        for i, r in enumerate(hs):
            hs[i] = r + rms(attention(rms(r, n_in, eps), L, c, cd, dt, sliding), n_pa, eps)
        # The MoE is per token: every sequence's rows go through it at once, so each expert is
        # read and dequantised once a layer.
        lens = [r.shape[0] for r in hs]
        y, ids, _ = moe(rms(torch.cat(hs), n_pal, eps), L, c, cd, dt)
        for i, (yi, ii) in enumerate(zip(y.split(lens), ids.split(lens))):
            hs[i] = hs[i] + rms(yi, n_pf, eps)
            routes[i].append(ii.cpu())
        ck.drop_cache()
        print(f"[ref] layer {L:2d} {'swa ' if sliding else 'full'} done {time.time() - t0:7.1f}s", flush=True)
    res = {"model": a.model, "dtype": a.dtype, "layers": nl, "seconds": None, "seqs": []}
    if nl == c["num_hidden_layers"]:
        fn = ck.get("model.norm.weight", dt).to(dev)
        head = ck.get("lm_head.weight", torch.float32).to(dev)
        for i, s in enumerate(seqs):
            logits = rms(hs[i], fn, eps).float() @ head.T
            ids = torch.tensor(s["ids"], device=dev)
            row = {"name": s["name"], "tokens": len(s["ids"])}
            if len(s["ids"]) > 1:
                lp = torch.log_softmax(logits[:-1], -1)
                nll = -lp.gather(1, ids[1:, None]).squeeze(1)
                row["nll_mean"] = nll.mean().item()
                row["argmax_next_agree"] = (logits[:-1].argmax(-1) == ids[1:]).float().mean().item()
                top2 = logits[:-1].topk(2, -1).values
                row["confident_share"] = ((top2[:, 0] - top2[:, 1]) >= 1.0).float().mean().item()
            top = torch.softmax(logits[-1], -1).topk(5)
            row["last_top5"] = [(tok.decode([t]), round(p, 4)) for t, p in zip(top.indices.tolist(), top.values.tolist())]
            if a.save_logits:
                os.makedirs(a.save_logits, exist_ok=True)
                torch.save({"ids": s["ids"], "logits": logits.cpu()}, os.path.join(a.save_logits, s["name"] + ".pt"))
            res["seqs"].append(row)
            print(f"[ref] {s['name']}: {json.dumps(row, ensure_ascii=False)}", flush=True)
    if a.union:
        E, k = c["num_experts"], c["num_experts_per_tok"]
        un = {}
        load = torch.zeros(nl, E)
        for i, s in enumerate(seqs):
            for L, ids in enumerate(routes[i]):
                load[L] += torch.bincount(ids.flatten(), minlength=E).float()
        for n in (1, 2, 4, 8, 16, 24, 32):
            vals = []
            for i, s in enumerate(seqs):
                T = len(s["ids"])
                if T < n:
                    continue
                for L, ids in enumerate(routes[i]):
                    for st in range(0, T - n + 1, n):
                        vals.append(len(torch.unique(ids[st:st + n])))
            if vals:
                un[n] = {"mean": sum(vals) / len(vals), "uniform": E * (1 - (1 - k / E) ** n), "windows": len(vals)}
        per_layer_n16 = defaultdict(list)
        for i, s in enumerate(seqs):
            T = len(s["ids"])
            for L, ids in enumerate(routes[i]):
                for st in range(0, T - 16 + 1, 16):
                    per_layer_n16[L].append(len(torch.unique(ids[st:st + 16])))
        tot = load.sum(1, keepdim=True)
        res["union"] = un
        res["union16_by_layer"] = {L: sum(v) / len(v) for L, v in per_layer_n16.items() if v}
        res["expert_load"] = {
            "tokens_routed_per_layer": tot[0].item(),
            "min_tokens_an_expert": load.min().item(),
            "experts_with_zero": int((load == 0).sum().item()),
            "experts_under_10": int((load < 10).sum().item()),
            "max_share": (load / tot).max().item(),
            "median_tokens_an_expert": load.median().item(),
        }
        print("[ref] union " + json.dumps(un), flush=True)
        print("[ref] load " + json.dumps(res["expert_load"]), flush=True)
    res["seconds"] = time.time() - t0
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)


class _Dev:
    """The checkpoint, with every tensor moved to the run's device as it is read."""

    def __init__(self, ck: Ckpt, dev: torch.device):
        self.ck, self.dev = ck, dev

    def get(self, name, dtype):
        return self.ck.get(name, dtype).to(self.dev)

    def raw(self, name):
        return self.ck.raw(name).to(self.dev)


if __name__ == "__main__":
    main()
