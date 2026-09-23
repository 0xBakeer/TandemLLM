"""The non-projection half of a linear-attention layer over a verify block, in four kernels.

At one token `tools/gdn_kernels.py::decode_pre` already does this: the convolution, its SiLU, the
state shift and the gates in two kernels, and a `REP` index in the recurrence instead of widening
sixteen key heads to forty-eight. A verify block never took that path -- it needs the factors for
the commit and a tree has no single convolution successor -- so over a block the same mixer is the
general torch path: `cat`, `copy_`, `conv1d`, `silu`, the nine small ops of the gates, two
`repeat_interleave`s, the `contiguous` copies the recurrence asks for, six trace clones, and the
factors (`l2norm` of the keys, two `transpose().contiguous()`, a `cumsum`): about forty launches
a layer, forty-eight layers, every block.

Here it is four:

  _verify_conv   the depthwise convolution and its SiLU for all T rows, a chain's sliding window or
                 a tree's ancestor windows (`TreeCtx.conv_idx`), and for a chain the state shift.
                 Accumulated in fp32 and rounded once, as `decode_pre` does it -- the torch path
                 rounds the convolution to bf16 before the SiLU, the decode step does not.
  _verify_gate   `g`, `beta` for all T rows, and for a chain the cumulative gate the commit reads.
                 `beta` in fp32 as `decode_pre` has it (the torch path rounds the sigmoid to bf16).
  _block_step    the chain recurrence (as `_gdn_block_step`), reading key head `h // REP`, writing
                 the commit's factors in the layout the commit reads: the normalised keys, the
                 per-token updates `[H, T, Dv]`.
  _tree_step     the ancestor-aware recurrence (as `_gdn_tree_step`), same additions, and the
                 cumulative gate along each path.

So the arithmetic of a verified row moves toward the decode step's -- the non-speculative greedy
run the losslessness gate compares against decodes one token at a time through `decode_pre`.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _verify_conv(X, s_xt, STATE, s_state, W, WIN, OUT, C, T,
                     WIDTH: tl.constexpr, BLOCK: tl.constexpr, TREE: tl.constexpr):
        """OUT[t, c] = silu(sum_w joined[win[t, w], c] * W[c, w]), joined = [state | x rows].

        A chain's window of row t is joined columns t .. t+WIDTH-1, and its new state is the last
        WIDTH-1 entries of the last row's window -- held in registers from the loop, stored after
        every load, as `_gdn_conv_silu` does it. A tree's windows come from `WIN` and its state is
        not advanced here (the commit writes the accepted path's).
        """
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < C
        tw = tl.arange(0, WIDTH)
        wv = tl.load(W + off[:, None] * WIDTH + tw[None, :], mask=mask[:, None],
                     other=0.0).to(tl.float32)
        joined = tl.zeros([BLOCK, WIDTH], dtype=tl.float32)
        for t in range(T):
            if TREE:
                idx = tl.load(WIN + t * WIDTH + tw).to(tl.int32)
            else:
                idx = t + tw
            fs = idx < WIDTH - 1
            sv = tl.load(STATE + off[:, None] * s_state + idx[None, :],
                         mask=mask[:, None] & fs[None, :], other=0.0)
            xv = tl.load(X + (idx - (WIDTH - 1))[None, :] * s_xt + off[:, None],
                         mask=mask[:, None] & (~fs)[None, :], other=0.0)
            joined = tl.where(fs[None, :], sv, xv).to(tl.float32)
            acc = tl.sum(joined * wv, axis=1)
            acc = acc * tl.sigmoid(acc)
            tl.store(OUT + t * C + off, acc.to(OUT.dtype.element_ty), mask=mask)
        if not TREE:
            tl.store(STATE + off[:, None] * s_state + (tw[None, :] - 1),
                     joined.to(STATE.dtype.element_ty),
                     mask=mask[:, None] & (tw[None, :] >= 1))

    @triton.jit
    def _verify_gate(A, B, s_a, s_b, ALOG, DTB, G, BETA, GC, T, H,
                     BLOCK: tl.constexpr, CUM: tl.constexpr):
        """`g = -exp(A_log) softplus(a + dt_bias)`, `beta = sigmoid(b)` for T rows, and (chain) the
        cumulative gate along the block, `GC[h, t]`."""
        off = tl.arange(0, BLOCK)
        mask = off < H
        dt = tl.load(DTB + off, mask=mask, other=0.0).to(tl.float32)
        al = tl.load(ALOG + off, mask=mask, other=0.0).to(tl.float32)
        cum = tl.zeros([BLOCK], dtype=tl.float32)
        for t in range(T):
            a = tl.load(A + t * s_a + off, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(B + t * s_b + off, mask=mask, other=0.0).to(tl.float32)
            x = a + dt
            sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
            g = -tl.exp(al) * sp
            tl.store(G + t * H + off, g, mask=mask)
            tl.store(BETA + t * H + off, 1.0 / (1.0 + tl.exp(-b)), mask=mask)
            if CUM:
                cum += g
                tl.store(GC + off * T + t, cum, mask=mask)

    @triton.jit
    def _block_step(Q, K, V, s_t, G, BETA, S, S_OUT, OUT, DELTA, KK, T,
                    s_h, s_k, s_v,
                    DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr, REP: tl.constexpr,
                    EPS: tl.constexpr, SCALE: tl.constexpr):
        """`_gdn_block_step`, reading key head h // REP from the convolution's own output rows
        (row stride `s_t`), writing DELTA [H, T, DV] and the normalised keys KK [H, T, DK]."""
        h = tl.program_id(0)
        vb = tl.program_id(1)
        H = tl.num_programs(0)
        hk = h // REP
        ok = tl.arange(0, DK)
        ov = vb * BV + tl.arange(0, BV)
        s = tl.load(S + h * s_h + ok[:, None] * s_k + ov[None, :] * s_v)
        for t in range(T):
            q = tl.load(Q + t * s_t + hk * DK + ok).to(tl.float32)
            k = tl.load(K + t * s_t + hk * DK + ok).to(tl.float32)
            q = q * tl.rsqrt(tl.sum(q * q) + EPS) * SCALE
            k = k * tl.rsqrt(tl.sum(k * k) + EPS)
            v = tl.load(V + t * s_t + h * DV + ov).to(tl.float32)
            g = tl.load(G + t * H + h)
            beta = tl.load(BETA + t * H + h)
            s = s * tl.exp(g)
            kv = tl.sum(s * k[:, None], axis=0)
            delta = (v - kv) * beta
            s = s + k[:, None] * delta[None, :]
            tl.store(DELTA + (h * T + t) * DV + ov, delta)
            if vb == 0:
                tl.store(KK + (h * T + t) * DK + ok, k)
            out = tl.sum(s * q[:, None], axis=0)
            tl.store(OUT + (t * H + h) * DV + ov, out.to(OUT.dtype.element_ty))
        tl.store(S_OUT + h * s_h + ok[:, None] * s_k + ov[None, :] * s_v, s)

    @triton.jit
    def _tree_step(Q, K, V, s_t, G, BETA, DEPTH, S, OUT, DELTA, KK, GC, T,
                   s_h, s_k, s_v,
                   DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr, REP: tl.constexpr,
                   MAXD: tl.constexpr, EPS: tl.constexpr, SCALE: tl.constexpr):
        """`_gdn_tree_step` with the same additions; GC [H, T] is the gate along each path."""
        h = tl.program_id(0)
        vb = tl.program_id(1)
        H = tl.num_programs(0)
        hk = h // REP
        ok = tl.arange(0, DK)
        ov = vb * BV + tl.arange(0, BV)
        s0 = tl.load(S + h * s_h + ok[:, None] * s_k + ov[None, :] * s_v)
        d_idx = tl.arange(0, MAXD)
        st_k = tl.zeros([MAXD, DK], dtype=tl.float32)
        st_u = tl.zeros([MAXD, BV], dtype=tl.float32)
        st_g = tl.zeros([MAXD], dtype=tl.float32)
        for t in range(T):
            d = tl.load(DEPTH + t)
            q = tl.load(Q + t * s_t + hk * DK + ok).to(tl.float32)
            k = tl.load(K + t * s_t + hk * DK + ok).to(tl.float32)
            q = q * tl.rsqrt(tl.sum(q * q) + EPS) * SCALE
            k = k * tl.rsqrt(tl.sum(k * k) + EPS)
            v = tl.load(V + t * s_t + h * DV + ov).to(tl.float32)
            g = tl.load(G + t * H + h)
            beta = tl.load(BETA + t * H + h)
            anc = d_idx < d
            gp = tl.sum(tl.where(d_idx == d - 1, st_g, 0.0))
            w = tl.where(anc, tl.exp(gp - st_g), 0.0)
            kw = st_k * w[:, None]
            s = tl.exp(gp) * s0 + tl.dot(tl.trans(kw), st_u)
            s = s * tl.exp(g)
            kv = tl.sum(s * k[:, None], axis=0)
            delta = (v - kv) * beta
            s = s + k[:, None] * delta[None, :]
            gc = gp + g
            out = tl.sum(s * q[:, None], axis=0)
            tl.store(OUT + (t * H + h) * DV + ov, out.to(OUT.dtype.element_ty))
            tl.store(DELTA + (h * T + t) * DV + ov, delta)
            if vb == 0:
                tl.store(GC + h * T + t, gc)
                tl.store(KK + (h * T + t) * DK + ok, k)
            here = (d_idx == d)
            st_k = tl.where(here[:, None], k[None, :], st_k)
            st_u = tl.where(here[:, None], delta[None, :], st_u)
            st_g = tl.where(here, gc, st_g)


MAXD = 16
# The recurrences' value block and warps (SPD-31 sweeps them; the shipped pair is what K1 measured).
import os as _os
BV = int(_os.environ.get("QWEN38_GDNV_BV", "16"))
WARPS = int(_os.environ.get("QWEN38_GDNV_WARPS", "4"))


def verify_mixer(mixed: torch.Tensor, conv_state: torch.Tensor, conv_w: torch.Tensor,
                 a_raw: torch.Tensor, b_raw: torch.Tensor, a_log: torch.Tensor,
                 dt_bias: torch.Tensor, state: torch.Tensor, *, key_dim: int, key_heads: int,
                 value_heads: int, head_k: int, head_v: int, window: torch.Tensor | None = None,
                 depths: torch.Tensor | None = None, max_depth: int = 0,
                 out_state: torch.Tensor | None = None, bv: int | None = None,
                 warps: int | None = None):
    """The whole recurrent half of a linear-attention layer over a verify block.

    mixed       [T, C] the qkv projection rows (any row stride; C = 2 key_dim + value_dim)
    conv_state  [1, C, W-1]  advanced in place for a chain, untouched for a tree
    a_raw/b_raw [T, H]
    state       [1, H, Dk, Dv] the entry state; a chain writes its walked state to `out_state`
                (default: in place), a tree writes none
    window      the tree's conv windows [T, W] (None for a chain); `depths` [T] its node depths

    Returns (o [1, T, H, Dv] in mixed's dtype, factors (kk [1, H, T, Dk], u [1, H, T, Dv],
    gc [1, H, T]) as the commit reads them).
    """
    T, C = mixed.shape
    W = conv_w.shape[-1]
    H = value_heads
    dev = mixed.device
    tree = window is not None
    assert mixed.stride(1) == 1, mixed.stride()
    st = conv_state.reshape(C, W - 1)
    qkv = torch.empty(T, C, dtype=mixed.dtype, device=dev)
    _verify_conv[(triton.cdiv(C, 256),)](
        mixed, mixed.stride(0), st, st.stride(0), conv_w, window if tree else mixed, qkv, C, T,
        WIDTH=W, BLOCK=256, TREE=tree, num_warps=4)
    g = torch.empty(T, H, dtype=torch.float32, device=dev)
    beta = torch.empty(T, H, dtype=torch.float32, device=dev)
    gc = torch.empty(H, T, dtype=torch.float32, device=dev)
    nb = triton.next_power_of_2(H)
    _verify_gate[(1,)](a_raw, b_raw, a_raw.stride(0), b_raw.stride(0), a_log, dt_bias,
                       g, beta, gc, T, H, BLOCK=nb, CUM=not tree, num_warps=1)
    q, k, v = qkv[:, :key_dim], qkv[:, key_dim:2 * key_dim], qkv[:, 2 * key_dim:]
    S = state.reshape(H, head_k, head_v)
    out = torch.empty(T, H, head_v, dtype=mixed.dtype, device=dev)
    delta = torch.empty(H, T, head_v, dtype=torch.float32, device=dev)
    kk = torch.empty(H, T, head_k, dtype=torch.float32, device=dev)
    rep = H // key_heads
    bv = BV if bv is None else bv
    warps = WARPS if warps is None else warps
    if tree:
        if max_depth >= MAXD:
            raise ValueError(f"tree is {max_depth + 1} deep, kernel carries {MAXD}")
        _tree_step[(H, head_v // bv)](
            q, k, v, C, g, beta, depths, S, out, delta, kk, gc, T,
            S.stride(0), S.stride(1), S.stride(2),
            DK=head_k, DV=head_v, BV=bv, REP=rep, MAXD=MAXD, EPS=1e-6, SCALE=head_k ** -0.5,
            num_warps=warps)
    else:
        So = S if out_state is None else out_state.reshape(H, head_k, head_v)
        _block_step[(H, head_v // bv)](
            q, k, v, C, g, beta, S, So, out, delta, kk, T,
            S.stride(0), S.stride(1), S.stride(2),
            DK=head_k, DV=head_v, BV=bv, REP=rep, EPS=1e-6, SCALE=head_k ** -0.5,
            num_warps=warps)
    return out.view(1, T, H, head_v), (kk[None], delta[None], gc[None]), qkv


def reference(mixed, conv_state, conv_w, a_raw, b_raw, a_log, dt_bias, state, *, key_dim,
              key_heads, value_heads, head_k, head_v, window=None, depths=None):
    """The engine's general path for the same block, for `check()`: conv_update / conv_tree, the
    torch gates, `repeat_interleave`, the existing recurrence kernels and the torch factors."""
    import torch.nn.functional as F
    from engine import gdn
    from tools.gdn_kernels import fused_block_step
    from tools.gdn_tree_kernels import fused_tree_step
    T, C = mixed.shape
    raw = mixed.t()[None]                                      # [1, C, T]
    if window is not None:
        x = gdn.conv_tree(raw, conv_state, conv_w, window)
    else:
        x = gdn.conv_update(raw, conv_state, conv_w)
    x = x.transpose(1, 2)
    q, k, v = x.split([key_dim, key_dim, value_heads * head_v], dim=-1)
    q = q.reshape(1, T, key_heads, head_k)
    k = k.reshape(1, T, key_heads, head_k)
    v = v.reshape(1, T, value_heads, head_v)
    beta = b_raw[None].sigmoid()
    g = -a_log.float().exp() * F.softplus(a_raw[None].float() + dt_bias.float())
    rep = value_heads // key_heads
    q = q.repeat_interleave(rep, dim=2)
    k = k.repeat_interleave(rep, dim=2)
    kk = gdn.l2norm(k.float(), dim=-1).transpose(1, 2).contiguous()
    if window is not None:
        o, delta, gc = fused_tree_step(q, k, v, g, beta, depths, state)
        return o, (kk, delta.transpose(1, 2).contiguous(), gc.transpose(1, 2).contiguous())
    o, delta = fused_block_step(q, k, v, g, beta, state)
    return o, (kk, delta.transpose(1, 2).contiguous(),
               g.float().cumsum(dim=1).transpose(1, 2).contiguous())


def check(T: int = 16, seed: int = 0) -> list[str]:
    """The fused mixer against the engine's general path, chain and tree, on the model's shapes."""
    from engine.tree import DraftTree
    kd, kh, vh, dk, dv, W = 2048, 16, 48, 128, 128, 4
    C = 2 * kd + vh * dv
    g_ = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    tree = DraftTree(tokens=[0] * 9, parents=[-1, 0, 1, 2, 1, 4, 0, 6, 6])
    for kind in ("chain", "tree"):
        n = T if kind == "chain" else len(tree.parents)
        mixed = (torch.randn(n, C, device="cuda", generator=g_) * 0.5).to(torch.bfloat16)
        cs = (torch.randn(1, C, W - 1, device="cuda", generator=g_) * 0.5).to(torch.bfloat16)
        cw = (torch.randn(C, W, device="cuda", generator=g_) * 0.3).to(torch.bfloat16)
        a = (torch.randn(n, vh, device="cuda", generator=g_)).to(torch.bfloat16)
        b = (torch.randn(n, vh, device="cuda", generator=g_)).to(torch.bfloat16)
        alog = (torch.randn(vh, device="cuda", generator=g_) * 0.5).to(torch.bfloat16)
        dtb = (torch.randn(vh, device="cuda", generator=g_) * 0.5).to(torch.bfloat16)
        S0 = torch.randn(1, vh, dk, dv, device="cuda", generator=g_) * 0.05
        win = dep = None
        if kind == "tree":
            win = torch.tensor(tree.conv_windows(W), dtype=torch.long, device="cuda")
            dep = torch.tensor(tree.depths(), dtype=torch.long, device="cuda")
        kw = dict(key_dim=kd, key_heads=kh, value_heads=vh, head_k=dk, head_v=dv)
        cs_r, S_r = cs.clone(), S0.clone()
        o_r, f_r = reference(mixed, cs_r, cw, a, b, alog, dtb, S_r, window=win, depths=dep, **kw)
        cs_f, S_f = cs.clone(), S0.clone()
        o_f, f_f, _ = verify_mixer(mixed, cs_f, cw, a, b, alog, dtb, S_f, window=win, depths=dep,
                                   max_depth=max(tree.depths()) if kind == "tree" else 0, **kw)
        rel = lambda x, y: ((x.float() - y.float()).abs().max() / y.float().abs().max()).item()
        r = {"o": rel(o_f, o_r), "kk": rel(f_f[0], f_r[0]), "u": rel(f_f[1], f_r[1]),
             "gc": rel(f_f[2], f_r[2]), "S": rel(S_f, S_r), "conv": rel(cs_f, cs_r)}
        out.append(f"{kind} T={n}: " + "  ".join(f"{k} {v:.2e}" for k, v in r.items()))
        assert max(r.values()) < 2e-2, r
    return out


def bench(T: int = 16, layers: int = 48, reps: int = 10,
          grid=((16, 4), (16, 2), (16, 1), (8, 1), (8, 2), (32, 4), (32, 2))) -> list[str]:
    """The mixer over `layers` distinct layers' worth of state, chain and tree, per (BV, warps):
    the recurrence's serial T loop has two reductions over the key dimension a step, and how many
    warps share them decides whether those are shuffles or shared-memory round trips."""
    from engine.tree import DraftTree
    kw = dict(key_dim=2048, key_heads=16, value_heads=48, head_k=128, head_v=128)
    ins = [_mixer_bench_inputs(T) for _ in range(layers)]
    tree = DraftTree(tokens=[0] * T, parents=[-1, 0, 0] + list(range(2, T - 1)))
    win = torch.tensor(tree.conv_windows(4), dtype=torch.long, device="cuda")
    dep = torch.tensor(tree.depths(), dtype=torch.long, device="cuda")
    scratch = [torch.empty_like(x["state"]) for x in ins]
    out = []
    for bv, wp in grid:
        row = []
        for kind in ("chain", "tree"):
            def one():
                for x, sc in zip(ins, scratch):
                    verify_mixer(**x, window=win if kind == "tree" else None,
                                 depths=dep if kind == "tree" else None,
                                 max_depth=max(tree.depths()) if kind == "tree" else 0,
                                 out_state=sc if kind == "chain" else None, bv=bv, warps=wp, **kw)
            one()
            torch.cuda.synchronize()
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(reps):
                one()
            t1.record()
            t1.synchronize()
            row.append(t0.elapsed_time(t1) / reps)
        out.append(f"BV {bv:2d} warps {wp}: chain {row[0]:.3f} ms  tree {row[1]:.3f} ms  "
                   f"({layers} layers, T={T})")
    return out


def _mixer_bench_inputs(n):
    kd, vh, dv, W = 2048, 48, 128, 4
    C = 2 * kd + vh * dv
    return dict(mixed=torch.randn(n, C, device="cuda").to(torch.bfloat16) * 0.5,
                conv_state=torch.randn(1, C, W - 1, device="cuda").to(torch.bfloat16) * 0.5,
                conv_w=torch.randn(C, W, device="cuda").to(torch.bfloat16) * 0.3,
                a_raw=torch.randn(n, vh, device="cuda").to(torch.bfloat16),
                b_raw=torch.randn(n, vh, device="cuda").to(torch.bfloat16),
                a_log=torch.randn(vh, device="cuda").to(torch.bfloat16) * 0.5,
                dt_bias=torch.randn(vh, device="cuda").to(torch.bfloat16) * 0.5,
                state=torch.randn(1, vh, 128, dv, device="cuda") * 0.05)


if __name__ == "__main__":
    import sys as _s
    for line in (bench() if "--bench" in _s.argv else check()):
        print(line, flush=True)
