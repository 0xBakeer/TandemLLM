"""SPD-201: every keyed draw's uniforms of a round in one kernel, bit for bit `torch.rand`.

A seeded request's tree walk (`engine.sample._keyed_draws`) draws one noise row a depth: it seeds the
slot's generator with `_key(seed, position)` and takes `torch.rand(V, generator=g)`. That is a
manual_seed, a launch and four elementwise launches a depth, about 30 a slot and 240 a round at 8 slots.
Here one kernel writes every (slot, depth) row's uniforms, each the row `torch.rand` writes after that
manual_seed, and the noise and argmax are torch's own elementwise ops over all rows at once.

`torch.rand` on CUDA (ATen `distribution_elementwise_grid_stride_kernel`, `curand_uniform4`): S =
256 x min(SMs x maxThreadsPerSM / 256, ceil(V / 256)) threads; thread `idx` at iteration `it` takes the
four outputs of Philox4x32-10 at counter (offset / 4 + it, 0, idx, 0) under the 64-bit key `seed`, and
element `it * 4S + ii * S + idx` is output `ii` as `float(r) * 2^-32 + 2^-33` (curand's (0, 1]), 1.0
folded to 0 (ATen's [0, 1)). The generator's offset moves by `4 * ceil(V / 4S)`. `check` holds the
kernel to `torch.rand` on the device it runs on, and tests/test_keyed_noise.py pins it per torch
version: the launch policy is ATen's, and a torch that changes it fails the test, not the bits.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.random import philox_impl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False

_BLOCK_SIZE = 256           # ATen's block_size_bound
_UNROLL = 4                 # curand_uniform4
_POLICY: dict = {}


def policy(V: int, device) -> tuple[int, int]:
    """(threads S, the generator's offset step) of `torch.rand(V)` on `device`, ATen's
    `calc_execution_policy`."""
    dev = torch.device(device)
    key = (V, dev.index)
    got = _POLICY.get(key)
    if got is None:
        p = torch.cuda.get_device_properties(dev)
        grid = min(p.multi_processor_count * (p.max_threads_per_multi_processor // _BLOCK_SIZE),
                   (V + _BLOCK_SIZE - 1) // _BLOCK_SIZE)
        S = _BLOCK_SIZE * grid
        step = ((V - 1) // (S * _UNROLL) + 1) * 4
        got = _POLICY[key] = (S, step)
    return got


if HAVE_TRITON:

    @triton.jit
    def _put(ROW, j, o, live, V):
        # curand's _curand_uniform: (0, 1]; the product is exact, so a fused multiply-add rounds alike
        u = o.to(tl.float32) * 2.3283064365386963e-10 + 1.1641532182693481e-10
        u = tl.where(u == 1.0, 0.0, u)                  # ATen's uniform: [0, 1)
        tl.store(ROW + j, u, mask=live & (j < V))

    @triton.jit
    def _uniforms(KEYS, OUT, V, S, ITERS, s_out, BLOCK: tl.constexpr):
        r = tl.program_id(0)
        idx = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        live = idx < S
        key = tl.load(KEYS + r).to(tl.uint64)
        k0 = (key & 0xFFFFFFFF).to(tl.uint32)
        k1 = ((key >> 32) & 0xFFFFFFFF).to(tl.uint32)
        z = (idx * 0).to(tl.uint32)
        c2 = idx.to(tl.uint32)
        for it in range(ITERS):
            c0 = z + it.to(tl.uint32)
            o0, o1, o2, o3 = philox_impl(c0, z, c2, z, k0, k1, 10)
            base = it * 4 * S + idx
            _put(OUT + r * s_out, base, o0, live, V)
            _put(OUT + r * s_out, base + S, o1, live, V)
            _put(OUT + r * s_out, base + 2 * S, o2, live, V)
            _put(OUT + r * s_out, base + 3 * S, o3, live, V)


def uniforms(keys: list[int], V: int, device) -> torch.Tensor:
    """[len(keys), V] float32: row r is `torch.rand(V, generator=g)` after `g.manual_seed(keys[r])`."""
    dev = torch.device(device)
    R = len(keys)
    out = torch.empty(R, V, dtype=torch.float32, device=dev)
    if R == 0:
        return out
    S, step = policy(V, dev)
    kt = torch.tensor(keys, dtype=torch.int64).to(dev, non_blocking=True)
    BLOCK = 256
    _uniforms[(R, triton.cdiv(S, BLOCK))](kt, out, V, S, step // 4, out.stride(0), BLOCK=BLOCK,
                                          num_warps=4)
    return out


def reference(keys: list[int], V: int, device) -> torch.Tensor:
    g = torch.Generator(device=device)
    rows = []
    for k in keys:
        g.manual_seed(k)
        rows.append(torch.rand(V, generator=g, device=device))
    return torch.stack(rows) if rows else torch.empty(0, V, device=device)


def check(n_keys: int = 64, sizes=(1, 7, 256, 4096, 151936, 248320, 300000)) -> list[str]:
    """The kernel against `torch.rand` on this device: every row bit for bit, and the offset step."""
    import random
    dev = torch.device("cuda")
    rng = random.Random(7)
    out = []
    for V in sizes:
        keys = [rng.getrandbits(63) for _ in range(n_keys)] + [0, 1, (1 << 63) - 1]
        got, ref = uniforms(keys, V, dev), reference(keys, V, dev)
        bad = int((got.view(torch.int32) != ref.view(torch.int32)).any(-1).sum())
        g = torch.Generator(device=dev)
        g.manual_seed(keys[0])
        torch.rand(V, generator=g, device=dev)
        step = policy(V, dev)[1]
        assert bad == 0, f"V={V}: {bad} of {len(keys)} rows differ from torch.rand"
        assert g.get_offset() == step, (V, g.get_offset(), step)
        out.append(f"V={V}: {len(keys)} rows bit-identical, offset step {step}")
    return out


# ---- SPD-201: a seeded walk's draws over each row's kept set only ------------------------------------------
# `_filter` keeps at most top_k tokens of a row (ties at the cut included); outside them the distribution is 0
# and a keyed draw (`argmax(log p + G)`) can never land there. So the draw needs the row's top candidates, the
# filter's arithmetic on them and the noise at their ids, not the [n, V] tensors. Element j of a keyed noise
# row is Philox output (j mod 4S) // S at counter ((j // 4S), 0, j mod S, 0), as `_uniforms` writes it; the
# elementwise steps use libdevice's IEEE division, exp and log (torch's own); only the sums of the softmax and
# the top-p cumulative mass are taken over the kept values in another order than torch's [V]-wide reductions.
if HAVE_TRITON:
    from triton.language.extra.cuda import libdevice as _ld

    @triton.jit
    def _sparse_keyed(VALS, IDS, KEYS, OUT, s_r, S, TEMP, TOP_K, TOP_P, LOG_MIN_P,
                      HAS_MINP: tl.constexpr, HAS_TOPP: tl.constexpr, KP: tl.constexpr):
        r = tl.program_id(0)
        o = tl.arange(0, KP)
        x = _ld.div_rn(tl.load(VALS + r * s_r + o), TEMP)          # logits / temperature, descending
        ids = tl.load(IDS + r * s_r + o)
        ninf = float("-inf")
        mx = tl.max(x, axis=0)
        keep = x > ninf
        if HAS_MINP:
            keep = keep & (x >= mx + LOG_MIN_P)                     # `logits < floor` is dropped
        nk = tl.sum(keep.to(tl.int32), axis=0)
        kth = tl.sum(tl.where(o == TOP_K - 1, x, 0.0), axis=0)
        cut = tl.where(TOP_K - 1 < nk, kth, ninf)                  # the top-k cutoff of the kept row
        keep = keep & (x >= cut)
        xm = tl.where(keep, x, ninf)
        if HAS_TOPP:
            e = tl.where(keep, _ld.exp(xm - mx), 0.0)
            lse = mx + _ld.log(tl.sum(e, axis=0))
            cum = tl.cumsum(tl.where(keep, _ld.exp(xm - lse), 0.0), axis=0)
            drop = cum > TOP_P
            prev = tl.cumsum(drop.to(tl.int32), axis=0) - drop.to(tl.int32)   # a drop before this one
            keep = keep & ((prev == 0) | (o == 0))                  # the first token that crosses stays
            xm = tl.where(keep, x, ninf)
        e2 = tl.where(keep, _ld.exp(xm - mx), 0.0)
        logp = _ld.log(_ld.div_rn(e2, tl.sum(e2, axis=0)))
        key = tl.load(KEYS + r).to(tl.uint64)
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
        sc = tl.where(keep, logp + g, ninf)
        best = tl.max(sc, axis=0)
        pick = tl.min(tl.where(sc == best, o, KP), axis=0)
        tl.store(OUT + r, tl.sum(tl.where(o == pick, ids, 0), axis=0))


#: candidates a row brings to the sparse keyed draw; the request's top_k must leave room for ties at its cut
SPARSE_KP = 64
SPARSE_TOPK_MAX = 48


def _consecutive(rows: list):
    """`rows` as one [sum n, V] view when they are consecutive row slices of one fp32 tensor, else None."""
    x0 = rows[0]
    if x0.dtype != torch.float32 or x0.dim() != 2 or x0.stride(1) != 1:
        return None
    n, nxt = int(x0.shape[0]), x0.data_ptr() + x0.shape[0] * x0.stride(0) * 4
    for x in rows[1:]:
        if (x.dtype != x0.dtype or x.dim() != 2 or x.stride() != x0.stride() or x.shape[1] != x0.shape[1]
                or x.data_ptr() != nxt or x.untyped_storage().data_ptr() != x0.untyped_storage().data_ptr()):
            return None
        n += int(x.shape[0])
        nxt = x.data_ptr() + x.shape[0] * x.stride(0) * 4
    return x0.as_strided((n, x0.shape[1]), x0.stride())


def sparse_keyed_draws(rows: list, keys: list[int], V: int, temperature: float, top_k: int, top_p: float,
                       min_p: float, device) -> torch.Tensor:
    """One draw per row of `rows` (each [n_i, V] logits, penalties applied), row r under noise key keys[r]:
    `argmax(log _filter(row) + G_key)` evaluated on the row's top SPARSE_KP logits. int64 [sum n_i]."""
    import math
    dev = torch.device(device)
    n = sum(int(x.shape[0]) for x in rows)
    vals = torch.empty(n, SPARSE_KP, dtype=torch.float32, device=dev)
    ids = torch.empty(n, SPARSE_KP, dtype=torch.long, device=dev)
    whole = _consecutive(rows)
    if whole is not None:
        # the slots' rows are one tensor's consecutive rows (the joint pass's head): one top-k for all
        # (a row's top-k is its own)
        torch.topk(whole, SPARSE_KP, dim=-1, sorted=True, out=(vals, ids))
    else:
        r0 = 0
        for x in rows:
            r1 = r0 + int(x.shape[0])
            torch.topk(x.float() if x.dtype != torch.float32 else x, SPARSE_KP, dim=-1, sorted=True,
                       out=(vals[r0:r1], ids[r0:r1]))
            r0 = r1
    out = torch.empty(n, dtype=torch.long, device=dev)
    S, _ = policy(V, dev)
    kt = torch.tensor(keys, dtype=torch.int64).to(dev, non_blocking=True)
    _sparse_keyed[(n,)](vals, ids, kt, out, vals.stride(0), S, float(temperature), int(top_k), float(top_p),
                        math.log(min_p) if min_p > 0 else 0.0, HAS_MINP=min_p > 0.0, HAS_TOPP=top_p < 1.0,
                        KP=SPARSE_KP, num_warps=2)
    return out


if __name__ == "__main__":
    print(f"torch {torch.__version__}")
    print("\n".join(check()))
