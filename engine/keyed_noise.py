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


if __name__ == "__main__":
    print(f"torch {torch.__version__}")
    print("\n".join(check()))
