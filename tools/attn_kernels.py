"""Attention for a decode step or a verify block, reading each cached key and value ONCE.

A verify block is at most seventeen query rows against a context that is a prompt long. Below
`GQA_FROM` rows the engine used to hand SDPA a `repeat_interleave`d cache: the four key/value heads
widened to the twenty-four query heads, which writes the whole context six times over and reads it
back, per layer, per step. At a few hundred tokens of context that is a 3.7 MB copy the SDPA kernel
liked having (SPEED-LEDGER, phase 3). At 32k it is 384 MB a tensor a layer, and the step reads
about 26 GB of it against 15 GB of weights.

This kernel indexes instead. One program owns one key/value head, one slice of the context and a
group of query rows, where a "row" is a (query head, block position) pair: the six query heads that
share a key/value head and the T positions of the block are flattened into up to 6 x 17 = 102 rows,
so each key and value tile is loaded once for all of them and fed to `tl.dot`. The context is split
across programs (flash-decoding) and a second, small kernel combines the slices.

The mask is the engine's: every column before `start` is visible to every row, and inside the block
row t sees column start + j where `block_mask[t, j]` -- the causal triangle for a chain, the
ancestor relation for a tree.

Optionally the cache is e4m3 with one fp32 scale per (head, token) for keys and one for values
(VIS-5): half the bytes of bf16. The scales fold into the kernel for free -- the key scale
multiplies a score column, the value scale multiplies a probability column -- so the dot products
run on the unscaled codes.

The arithmetic order differs from SDPA's, so the output is not bit-identical to the path it
replaces. It IS row-independent: a row's result does not depend on how many other rows share the
launch, which is what keeps the decode step and the verify block on the same numbers and the
losslessness gate meaningful. `check()` compares against a float32 reference.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False

FP8_MAX = 448.0


if HAVE_TRITON:

    @triton.jit
    def _attn_split(Q, K, V, KS, VS, BMASK, OUT, PM, PL, PACC,
                    T, R, NKV, START, LC, CHUNK,
                    s_qh, s_qt, s_kh, s_kn, s_vh, s_vn, s_sh, s_oh, s_ot,
                    SCALE: tl.constexpr, REP: tl.constexpr, D: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, FP8: tl.constexpr, DIRECT: tl.constexpr):
        rg = tl.program_id(0)
        kvh = tl.program_id(1)
        sp = tl.program_id(2)
        rows = rg * BM + tl.arange(0, BM)                    # flattened (group, position)
        rmask = rows < R
        g = rows // T
        t = rows % T
        qh = kvh * REP + g
        od = tl.arange(0, D)
        q = tl.load(Q + qh[:, None] * s_qh + t[:, None] * s_qt + od[None, :],
                    mask=rmask[:, None], other=0.0)
        c0 = sp * CHUNK
        c1 = tl.minimum(c0 + CHUNK, LC)
        m_i = tl.full([BM], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BM], dtype=tl.float32)
        acc = tl.zeros([BM, D], dtype=tl.float32)
        for n0 in range(c0, c1, BN):
            cols = n0 + tl.arange(0, BN)
            cmask = cols < c1
            k = tl.load(K + kvh * s_kh + cols[:, None] * s_kn + od[None, :],
                        mask=cmask[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k.to(tl.bfloat16)))
            if FP8:
                ks = tl.load(KS + kvh * s_sh + cols, mask=cmask, other=0.0)
                s = s * ks[None, :]
            s = s * SCALE
            inblk = cols - START
            bm = tl.load(BMASK + t[:, None] * T + inblk[None, :],
                         mask=rmask[:, None] & (inblk[None, :] >= 0) & cmask[None, :], other=0)
            vis = (cols[None, :] < START) | (bm != 0)
            vis = vis & cmask[None, :] & rmask[:, None]
            s = tl.where(vis, s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(V + kvh * s_vh + cols[:, None] * s_vn + od[None, :],
                        mask=cmask[:, None], other=0.0)
            if FP8:
                vs = tl.load(VS + kvh * s_sh + cols, mask=cmask, other=0.0)
                p = p * vs[None, :]
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
            m_i = m_new
        if DIRECT:
            o = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
            tl.store(OUT + t[:, None] * s_ot + qh[:, None] * s_oh + od[None, :],
                     o.to(tl.bfloat16), mask=rmask[:, None])
        else:
            base = sp * (NKV * R) + kvh * R + rows
            tl.store(PM + base, m_i, mask=rmask)
            tl.store(PL + base, l_i, mask=rmask)
            tl.store(PACC + base[:, None] * D + od[None, :], acc, mask=rmask[:, None])

    @triton.jit
    def _attn_combine(PM, PL, PACC, OUT, T, R, NKV, NS, s_oh, s_ot,
                      REP: tl.constexpr, D: tl.constexpr, NSP: tl.constexpr):
        kvh = tl.program_id(0)
        r = tl.program_id(1)
        g = r // T
        t = r % T
        qh = kvh * REP + g
        stride = NKV * R
        sps = tl.arange(0, NSP)
        smask = sps < NS
        idx = sps * stride + kvh * R + r
        m = tl.load(PM + idx, mask=smask, other=-float("inf"))
        l = tl.load(PL + idx, mask=smask, other=0.0)
        mx = tl.max(m, 0)
        mx = tl.where(mx == -float("inf"), 0.0, mx)
        w = tl.where(smask, tl.exp(m - mx), 0.0)
        od = tl.arange(0, D)
        acc = tl.load(PACC + idx[:, None] * D + od[None, :], mask=smask[:, None], other=0.0)
        num = tl.sum(acc * w[:, None], 0)
        den = tl.sum(l * w, 0)
        o = num / tl.where(den == 0.0, 1.0, den)
        tl.store(OUT + t * s_ot + qh * s_oh + od, o.to(tl.bfloat16))


def pick_launch(T: int, rep: int, lc: int, *, min_chunk: int = 512, max_splits: int = 64):
    """Row-group size, number of row groups, the context chunk and the number of chunks.

    The chunk is a function of the context length alone, and a power of two times `min_chunk`, so
    the single-token step at context p and the verify block whose row sees p columns cut the context
    at the same places. Columns a row may not see add exact zeros, so a row's arithmetic is then the
    same whether it was decoded alone or verified in a block: the step and the verify agree bit for
    bit except where the chunk doubles (at 32k, 64k and 128k of context).
    """
    R = rep * T
    bm = 16 if R <= 16 else 32
    groups = -(-R // bm)
    per = -(-lc // max_splits)
    chunk = max(min_chunk, 1 << (per - 1).bit_length())
    return bm, groups, -(-lc // chunk), chunk


def decode_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int,
                     block_mask: torch.Tensor, *, ks: torch.Tensor | None = None,
                     vs: torch.Tensor | None = None, scale: float | None = None,
                     bn: int = 32, num_warps: int = 4, num_stages: int = 2) -> torch.Tensor:
    """softmax(q k^T * scale, masked) v, with q [1, Hq, T, D] and the cache k, v [1, Hkv, L, D].

    `L` is the context INCLUDING the block (start + T). `block_mask` is [T, T] bool: row t may see
    block column j. Returns [1, Hq, T, D] bf16, laid out so `.transpose(1, 2)` is contiguous.
    With `ks`/`vs` ([1, Hkv, >=L] fp32) the cache is e4m3 codes and these are its scales.
    """
    _, hq, T, D = q.shape
    _, hkv, lc, _ = k.shape
    rep = hq // hkv
    assert hq == 24 or hq == rep * hkv, (hq, hkv)
    fp8 = ks is not None
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    out = torch.empty(T, hq, D, dtype=torch.bfloat16, device=q.device)
    bm, groups, ns, chunk = pick_launch(T, rep, lc)
    R = rep * T
    bmask = block_mask.to(torch.int8).contiguous()
    q0, k0, v0 = q[0], k[0], v[0]
    ks0 = ks[0] if fp8 else k0
    vs0 = vs[0] if fp8 else k0
    s_sh = ks0.stride(0) if fp8 else 0
    direct = ns == 1
    if direct:
        pm = pl = pacc = out
    else:
        pm = torch.empty(ns * R * hkv, dtype=torch.float32, device=q.device)
        pl = torch.empty_like(pm)
        pacc = torch.empty(ns * R * hkv, D, dtype=torch.float32, device=q.device)
    _attn_split[(groups, hkv, ns)](
        q0, k0, v0, ks0, vs0, bmask, out, pm, pl, pacc,
        T, R, hkv, start, lc, chunk,
        q0.stride(0), q0.stride(1), k0.stride(0), k0.stride(1), v0.stride(0), v0.stride(1),
        s_sh, out.stride(1), out.stride(0),
        SCALE=scale, REP=rep, D=D, BM=bm, BN=bn, FP8=fp8, DIRECT=direct,
        num_warps=num_warps, num_stages=num_stages)
    if not direct:
        _attn_combine[(hkv, R)](pm, pl, pacc, out, T, R, hkv, ns, out.stride(1), out.stride(0),
                                REP=rep, D=D, NSP=triton.next_power_of_2(ns), num_warps=4)
    return out.unsqueeze(0).transpose(1, 2)


def quantize_kv(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[..., D] bf16 -> e4m3 codes and one fp32 scale per vector (amax / 448)."""
    amax = x.float().abs().amax(-1).clamp_min(1e-12)
    s = amax / FP8_MAX
    codes = (x.float() / s[..., None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return codes, s


def reference(q, k, v, start, block_mask, ks=None, vs=None, scale=None):
    """The same attention in float32 torch, with the cache dequantised when it is e4m3."""
    _, hq, T, D = q.shape
    hkv, lc = k.shape[1], k.shape[2]
    kf = k.float() * (ks[..., :lc, None] if ks is not None else 1.0)
    vf = v.float() * (vs[..., :lc, None] if vs is not None else 1.0)
    rep = hq // hkv
    kf = kf.repeat_interleave(rep, dim=1)
    vf = vf.repeat_interleave(rep, dim=1)
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    s = torch.einsum("bhtd,bhnd->bhtn", q.float(), kf) * scale
    vis = torch.zeros(T, lc, dtype=torch.bool, device=q.device)
    vis[:, :start] = True
    vis[:, start:start + T] = block_mask
    s = s.masked_fill(~vis, float("-inf"))
    return torch.einsum("bhtn,bhnd->bhtd", s.softmax(-1), vf)


def check(ctx=(1, 300, 4096, 33000), Ts=(1, 8, 16, 17), *, fp8: bool = False, seed: int = 0,
          device: str = "cuda") -> list[dict]:
    """Max |d| against the float32 reference, beside the bf16 SDPA path's own distance from it."""
    import torch.nn.functional as F
    torch.manual_seed(seed)
    out = []
    for L in ctx:
        for T in Ts:
            start = L
            lc = start + T
            q = torch.randn(1, 24, T, 256, device=device, dtype=torch.bfloat16)
            k = torch.randn(1, 4, lc, 256, device=device, dtype=torch.bfloat16)
            v = torch.randn(1, 4, lc, 256, device=device, dtype=torch.bfloat16)
            # a tree mask with a branch, and the causal triangle on the diagonal
            bm = torch.ones(T, T, dtype=torch.bool, device=device).tril()
            if T >= 4:
                bm[T // 2:, 1:T // 2] = False
                bm[T // 2:, 0] = True
            ks = vs = None
            if fp8:
                k, ks = quantize_kv(k)
                v, vs = quantize_kv(v)
            ref = reference(q, k, v, start, bm, ks, vs)
            got = decode_attention(q, k, v, start, bm, ks=ks, vs=vs).float()
            row = {"ctx": L, "T": T, "max_abs": (got - ref).abs().max().item(),
                   "ref_absmax": ref.abs().max().item()}
            if not fp8:
                vis = torch.zeros(T, lc, dtype=torch.bool, device=device)
                vis[:, :start] = True
                vis[:, start:] = bm
                sd = F.scaled_dot_product_attention(q, k.repeat_interleave(6, 1),
                                                    v.repeat_interleave(6, 1), attn_mask=vis)
                row["sdpa_abs"] = (sd.float() - ref).abs().max().item()
            # row independence, on the chain triangle: row t verified in the block against the
            # same row decoded alone at context start + t + 1. Bit-identical by construction.
            if T > 1:
                tri = torch.ones(T, T, dtype=torch.bool, device=device).tril()
                blk = decode_attention(q, k, v, start, tri, ks=ks, vs=vs)
                one = torch.cat([decode_attention(
                    q[:, :, t:t + 1], k[:, :, :start + t + 1], v[:, :, :start + t + 1], start + t,
                    torch.ones(1, 1, dtype=torch.bool, device=device), ks=ks, vs=vs)
                    for t in range(T)], dim=2)
                row["row_indep_abs"] = (one.float() - blk.float()).abs().max().item()
            out.append(row)
    return out


def bench(ctx=(4096, 8192, 32768, 131072), Ts=(1, 16), *, layers: int = 16, reps: int = 5,
          device: str = "cuda") -> list[dict]:
    """Milliseconds for the sixteen attention layers of one step, three ways, on distinct buffers.

    Each layer gets its own cache, so the loop reads 16 layers' worth of KV cold, as a step does:
    the SDPA path as the engine ships it below 64 rows (`repeat_interleave`, then a boolean mask),
    this kernel on the bf16 cache, and this kernel on the e4m3 one.
    """
    import time as _t
    import torch.nn.functional as F
    out = []
    for L in ctx:
        for T in Ts:
            start = L - T
            q = torch.randn(1, 24, T, 256, device=device, dtype=torch.bfloat16)
            ks_ = [torch.randn(1, 4, L, 256, device=device, dtype=torch.bfloat16)
                   for _ in range(layers)]
            vs_ = [torch.randn(1, 4, L, 256, device=device, dtype=torch.bfloat16)
                   for _ in range(layers)]
            tri = torch.ones(T, T, dtype=torch.bool, device=device).tril()
            vis = torch.zeros(T, L, dtype=torch.bool, device=device)
            vis[:, :start] = True
            vis[:, start:] = tri

            def sdpa():
                for k, v in zip(ks_, vs_):
                    F.scaled_dot_product_attention(q, k.repeat_interleave(6, 1),
                                                   v.repeat_interleave(6, 1),
                                                   attn_mask=None if T == 1 else vis)

            def mine():
                for k, v in zip(ks_, vs_):
                    decode_attention(q, k, v, start, tri)

            q8 = [quantize_kv(k) for k in ks_]
            v8 = [quantize_kv(v) for v in vs_]

            def mine8():
                for (k, sk), (v, sv) in zip(q8, v8):
                    decode_attention(q, k, v, start, tri, ks=sk, vs=sv)

            row = {"ctx": L, "T": T}
            for name, fn in (("sdpa", sdpa), ("kernel", mine), ("kernel_fp8", mine8)):
                try:
                    fn()
                    torch.cuda.synchronize()
                    best = 1e9
                    for _ in range(reps):
                        t0 = _t.perf_counter()
                        fn()
                        torch.cuda.synchronize()
                        best = min(best, _t.perf_counter() - t0)
                    row[name] = best * 1e3
                except torch.OutOfMemoryError:
                    row[name] = float("nan")
                    torch.cuda.empty_cache()
            bytes_bf16 = layers * 2 * 4 * L * 256 * 2
            row["kv_gb"] = bytes_bf16 / 1e9
            print(f"ctx {L:>7} T {T:>2}  kv {row['kv_gb']:6.2f} GB   sdpa {row['sdpa']:8.2f} ms   "
                  f"kernel {row['kernel']:7.2f} ms ({bytes_bf16 / row['kernel'] / 1e6:6.1f} GB/s)   "
                  f"fp8 {row['kernel_fp8']:7.2f} ms", flush=True)
            out.append(row)
            del ks_, vs_, q8, v8
            torch.cuda.empty_cache()
    return out
