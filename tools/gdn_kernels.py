"""Fused decode-step kernels for the Gated DeltaNet layers.

Forty-eight of this model's sixty-four layers are linear-attention mixers, and at one token per
step their recurrence is not a bandwidth problem -- the whole recurrent state is 3.1 MB per layer,
6.3 MB read and written, about 31 us at this board's measured rate. It is a *launch and latency*
problem: `engine/gdn.py`'s reference step issues an exponential, two einsums, a multiply-accumulate
and several elementwise ops as separate kernels over tensors small enough that each one is almost
entirely launch overhead, and it does that forty-eight times per token.

This file writes the whole step as one kernel. Per (head, value-block) program:

    S <- S * exp(g)
    kv <- S^T k            (reduction over the key dimension)
    delta <- (v - kv) * beta
    S <- S + k (x) delta
    out <- S^T q

Every one of those touches the same [128, BV] tile of the state, so the tile is loaded once, kept
in registers, and stored once. The l2 normalisation and the 1/sqrt(d) scale of q and k are folded
in as well, because they are two more launches over 128 floats.

The block over the VALUE dimension is what makes this parallel: the recurrence is sequential in
time and independent across (head, value column), so a grid of 48 heads x 4 value blocks of 32 is
192 programs of 16 KB each -- the shape that other implementations of this same geometry have
converged on, and enough blocks to fill this board's 48 SMs without each one being trivial.

NUMERICS. The arithmetic is fp32 throughout and in the reference's order, and the output is rounded
to bf16 exactly where the reference rounds it. It is not bit-identical: the two reductions are tree
reductions where `torch.einsum` may use a different order. `check()` measures the difference against
the reference on real tensors, and the engine-level gates (argmax agreement, greedy losslessness)
are what decide whether it is allowed to ship.
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
    def _gdn_decode_step(Q, K, V, G, BETA, S, OUT,
                         s_h, s_k, s_v,
                         DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
                         EPS: tl.constexpr, SCALE: tl.constexpr):
        h = tl.program_id(0)
        vb = tl.program_id(1)
        ok = tl.arange(0, DK)
        ov = vb * BV + tl.arange(0, BV)

        q = tl.load(Q + h * DK + ok).to(tl.float32)
        k = tl.load(K + h * DK + ok).to(tl.float32)
        q = q * tl.rsqrt(tl.sum(q * q) + EPS) * SCALE
        k = k * tl.rsqrt(tl.sum(k * k) + EPS)
        v = tl.load(V + h * DV + ov).to(tl.float32)
        g = tl.load(G + h).to(tl.float32)
        beta = tl.load(BETA + h).to(tl.float32)

        sp = S + h * s_h + ok[:, None] * s_k + ov[None, :] * s_v
        s = tl.load(sp) * tl.exp(g)                      # [DK, BV] fp32
        kv = tl.sum(s * k[:, None], axis=0)              # [BV]
        delta = (v - kv) * beta
        s = s + k[:, None] * delta[None, :]
        tl.store(sp, s)
        out = tl.sum(s * q[:, None], axis=0)
        tl.store(OUT + h * DV + ov, out.to(OUT.dtype.element_ty))


def fused_decode_step(query, key, value, g, beta, state, *, bv: int = 16):
    """One token of the gated delta rule, fused. Same signature as the reference's T = 1 case.

    query/key  [1, 1, H, Dk]   pre-normalisation, as the reference takes them
    value      [1, 1, H, Dv]
    g, beta    [1, 1, H]
    state      [1, H, Dk, Dv]  fp32, advanced IN PLACE, exactly as the reference does
    returns    out [1, 1, H, Dv] in query's dtype
    """
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    B, T, H, Dk = query.shape
    Dv = value.shape[-1]
    if B != 1 or T != 1:
        raise ValueError(f"fused_decode_step is the T = 1 path only, got B={B} T={T}")
    if Dv % bv:
        raise ValueError(f"value head dim {Dv} is not a multiple of bv={bv}")
    q = query.reshape(H, Dk).contiguous()
    k = key.reshape(H, Dk).contiguous()
    v = value.reshape(H, Dv).contiguous()
    gg = g.reshape(H).contiguous().float()
    bb = beta.reshape(H).contiguous().float()
    S = state.reshape(H, Dk, Dv)
    out = torch.empty(H, Dv, dtype=query.dtype, device=query.device)
    _gdn_decode_step[(H, Dv // bv)](
        q, k, v, gg, bb, S, out,
        S.stride(0), S.stride(1), S.stride(2),
        DK=Dk, DV=Dv, BV=bv, EPS=1e-6, SCALE=Dk ** -0.5,
        num_warps=4,
    )
    return out.view(1, 1, H, Dv)


if HAVE_TRITON:

    @triton.jit
    def _gdn_block_step(Q, K, V, G, BETA, S, OUT, DELTA, T,
                        s_qt, s_vt, s_gt, s_h, s_k, s_v,
                        DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
                        EPS: tl.constexpr, SCALE: tl.constexpr):
        """The same recurrence, T tokens deep, with the state tile never leaving registers."""
        h = tl.program_id(0)
        vb = tl.program_id(1)
        ok = tl.arange(0, DK)
        ov = vb * BV + tl.arange(0, BV)
        sp = S + h * s_h + ok[:, None] * s_k + ov[None, :] * s_v
        s = tl.load(sp)
        for t in range(T):
            q = tl.load(Q + t * s_qt + h * DK + ok).to(tl.float32)
            k = tl.load(K + t * s_qt + h * DK + ok).to(tl.float32)
            q = q * tl.rsqrt(tl.sum(q * q) + EPS) * SCALE
            k = k * tl.rsqrt(tl.sum(k * k) + EPS)
            v = tl.load(V + t * s_vt + h * DV + ov).to(tl.float32)
            g = tl.load(G + t * s_gt + h).to(tl.float32)
            beta = tl.load(BETA + t * s_gt + h).to(tl.float32)
            s = s * tl.exp(g)
            kv = tl.sum(s * k[:, None], axis=0)
            delta = (v - kv) * beta
            s = s + k[:, None] * delta[None, :]
            tl.store(DELTA + t * s_vt + h * DV + ov, delta)
            out = tl.sum(s * q[:, None], axis=0)
            tl.store(OUT + t * s_vt + h * DV + ov, out.to(OUT.dtype.element_ty))
        tl.store(sp, s)


def fused_block_step(query, key, value, g, beta, state, *, bv: int = 16):
    """The recurrence over a whole verify block, one kernel per layer.

    The chunked form exists because a long sequence cannot hold its state in registers and has to be
    blocked into matrix work. A speculative block is eight tokens. Eight tokens fit: the [128, BV]
    tile of the state is loaded once, walked forward eight times, and stored once, so the state
    traffic of verifying eight tokens is the state traffic of verifying one.

    Returns `out` in query's dtype and `delta`, the per-token rank-1 update vectors, in fp32.
    `delta` is what a partial accept needs: with the normalised keys and the cumulative gate it
    gives the state after any prefix without touching the recurrence again. See
    `Qwen38Engine.rollback_to`.
    """
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    B, T, H, Dk = query.shape
    Dv = value.shape[-1]
    if B != 1:
        raise ValueError(f"fused_block_step is a single sequence, got B={B}")
    if Dv % bv:
        raise ValueError(f"value head dim {Dv} is not a multiple of bv={bv}")
    q = query.reshape(T, H, Dk).contiguous()
    k = key.reshape(T, H, Dk).contiguous()
    v = value.reshape(T, H, Dv).contiguous()
    gg = g.reshape(T, H).contiguous().float()
    bb = beta.reshape(T, H).contiguous().float()
    S = state.reshape(H, Dk, Dv)
    out = torch.empty(T, H, Dv, dtype=query.dtype, device=query.device)
    delta = torch.empty(T, H, Dv, dtype=torch.float32, device=query.device)
    _gdn_block_step[(H, Dv // bv)](
        q, k, v, gg, bb, S, out, delta, T,
        H * Dk, H * Dv, H,
        S.stride(0), S.stride(1), S.stride(2),
        DK=Dk, DV=Dv, BV=bv, EPS=1e-6, SCALE=Dk ** -0.5,
        num_warps=4,
    )
    return out.view(1, T, H, Dv), delta.view(1, T, H, Dv)


def check_block(H: int = 48, Dk: int = 128, Dv: int = 128, T: int = 8, *, bv: int = 16,
                seed: int = 0, device: str = "cuda") -> dict:
    """The block kernel against the chunked form the engine uses for a verify pass today."""
    from engine.gdn import chunk_gated_delta_rule

    torch.manual_seed(seed)
    S0 = torch.randn(1, H, Dk, Dv, device=device, dtype=torch.float32) * 0.1
    q = torch.randn(1, T, H, Dk, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, Dk, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, Dv, device=device, dtype=torch.bfloat16)
    g = -torch.rand(1, T, H, device=device, dtype=torch.float32) * 0.5
    beta = torch.rand(1, T, H, device=device, dtype=torch.float32)
    o_ref, S_ref = chunk_gated_delta_rule(q, k, v, g, beta, S0.clone(), chunk_size=max(2, T))
    S_fus = S0.clone()
    o_fus, _ = fused_block_step(q, k, v, g, beta, S_fus, bv=bv)
    return {"out_absmax": float((o_ref.float() - o_fus.float()).abs().max()),
            "out_scale": float(o_ref.float().abs().max()),
            "state_absmax": float((S_ref - S_fus).abs().max()),
            "state_scale": float(S_ref.abs().max())}


def bench_block(H: int = 48, Dk: int = 128, Dv: int = 128, *, bv: int = 16, iters: int = 100,
                device: str = "cuda", lengths=(4, 8, 12, 16)) -> list[str]:
    import time
    from engine.gdn import chunk_gated_delta_rule
    out = []
    for T in lengths:
        S = torch.randn(1, H, Dk, Dv, device=device, dtype=torch.float32) * 0.1
        q = torch.randn(1, T, H, Dk, device=device, dtype=torch.bfloat16)
        k = torch.randn(1, T, H, Dk, device=device, dtype=torch.bfloat16)
        v = torch.randn(1, T, H, Dv, device=device, dtype=torch.bfloat16)
        g = -torch.rand(1, T, H, device=device, dtype=torch.float32) * 0.5
        beta = torch.rand(1, T, H, device=device, dtype=torch.float32)
        row = []
        for name, fn in (("chunked", lambda: chunk_gated_delta_rule(q, k, v, g, beta, S,
                                                                    chunk_size=max(2, T))),
                         ("fused", lambda: fused_block_step(q, k, v, g, beta, S, bv=bv))):
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            row.append((time.perf_counter() - t0) / iters * 1e6)
        c = check_block(T=T, bv=bv)
        out.append(f"T={T:3d}  chunked {row[0]:7.1f} us  fused {row[1]:6.1f} us  "
                   f"x{row[0] / row[1]:4.2f}   over 48 layers {row[0] * 48 / 1000:6.2f} -> "
                   f"{row[1] * 48 / 1000:5.2f} ms   out |d| {c['out_absmax']:.2e} "
                   f"(scale {c['out_scale']:.2f})  state |d| {c['state_absmax']:.2e} "
                   f"(scale {c['state_scale']:.2f})")
    return out


# --------------------------------------------------------------------------- the gate

def check(H: int = 48, Dk: int = 128, Dv: int = 128, *, bv: int = 32, seed: int = 0,
          device: str = "cuda", steps: int = 8) -> dict:
    """Run the fused step and the reference step from the same state, `steps` times in a row.

    One step would hide a state-update error behind the output comparison; running a few in a row
    lets any divergence in the state accumulate, which is the failure that matters -- the state is
    what the next token reads.
    """
    from engine.gdn import recurrent_gated_delta_rule

    torch.manual_seed(seed)
    S_ref = torch.randn(1, H, Dk, Dv, device=device, dtype=torch.float32) * 0.1
    S_fus = S_ref.clone()
    worst_out, worst_state = 0.0, 0.0
    for _ in range(steps):
        q = torch.randn(1, 1, H, Dk, device=device, dtype=torch.bfloat16)
        k = torch.randn(1, 1, H, Dk, device=device, dtype=torch.bfloat16)
        v = torch.randn(1, 1, H, Dv, device=device, dtype=torch.bfloat16)
        g = -torch.rand(1, 1, H, device=device, dtype=torch.float32) * 0.5
        beta = torch.rand(1, 1, H, device=device, dtype=torch.float32)
        o_ref, _ = recurrent_gated_delta_rule(q, k, v, g, beta, S_ref)
        o_fus = fused_decode_step(q, k, v, g, beta, S_fus, bv=bv)
        worst_out = max(worst_out, float((o_ref.float() - o_fus.float()).abs().max()))
        worst_state = max(worst_state, float((S_ref - S_fus).abs().max()))
    return {"out_absmax": worst_out, "state_absmax": worst_state,
            "state_rms": float(S_ref.pow(2).mean().sqrt())}


def bench(H: int = 48, Dk: int = 128, Dv: int = 128, *, bv: int = 32, iters: int = 200,
          device: str = "cuda") -> dict:
    """Per-layer microseconds, both ways. A standalone number -- read the ledger's 13:40 entry
    before believing it as a budget; the in-engine breakdown is the one that counts."""
    import time
    from engine.gdn import recurrent_gated_delta_rule

    S = torch.randn(1, H, Dk, Dv, device=device, dtype=torch.float32) * 0.1
    q = torch.randn(1, 1, H, Dk, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 1, H, Dk, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, 1, H, Dv, device=device, dtype=torch.bfloat16)
    g = -torch.rand(1, 1, H, device=device, dtype=torch.float32) * 0.5
    beta = torch.rand(1, 1, H, device=device, dtype=torch.float32)
    out = {}
    for name, fn in (("reference", lambda: recurrent_gated_delta_rule(q, k, v, g, beta, S)),
                     ("fused", lambda: fused_decode_step(q, k, v, g, beta, S, bv=bv))):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        out[name + "_us"] = (time.perf_counter() - t0) / iters * 1e6
    bytes_moved = H * Dk * Dv * 4 * 2
    out["fused_GBs"] = bytes_moved / (out["fused_us"] * 1e-6) / 1e9
    return out


def _main() -> None:
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--bv", type=int, default=16)
    ap.add_argument("--sweep", action="store_true", help="try every legal value block")
    ap.add_argument("--block", action="store_true",
                    help="the whole-block kernel against the chunked form, by block length")
    a = ap.parse_args()
    if a.block:
        for line in bench_block(bv=a.bv):
            print(line)
        return
    for bv in ([16, 32, 64, 128] if a.sweep else [a.bv]):
        c = check(bv=bv)
        b = bench(bv=bv)
        print(f"bv={bv:3d}  out |d| {c['out_absmax']:.3e}  state |d| {c['state_absmax']:.3e} "
              f"(state rms {c['state_rms']:.3f})   "
              f"reference {b['reference_us']:7.1f} us  fused {b['fused_us']:6.1f} us  "
              f"{b['fused_GBs']:6.1f} GB/s  x{b['reference_us'] / b['fused_us']:.2f}")


if __name__ == "__main__":
    _main()
