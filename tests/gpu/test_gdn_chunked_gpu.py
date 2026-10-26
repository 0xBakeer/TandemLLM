"""On the board: the GDN prefill recurrence at a 32-row chunk (SPD-245, tools/gdn_chunked_kernels.py,
QWEN38_GDN_CHUNKED) against the shipped 64-row path (tools/gdn_prep_kernels.py chunk_prefill_normed) and an
fp64 token-serial reference, on the 27B's head shapes with correlated keys and a non-zero entry state:
outputs within a few bf16 ulps of the shipped path, the state within fp32 tolerance, the error against the
fp64 reference no larger than the shipped path's, and a window split at a chunk boundary (two calls, the
state carried) bit for bit the one call. Run on the box under the lock:
python tests/gpu/test_gdn_chunked_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine import gdn  # noqa: E402
from tools.gdn_chunked_kernels import chunked_prefill_normed  # noqa: E402
from tools.gdn_prep_kernels import chunk_prefill_normed  # noqa: E402

H, HK, D = 48, 16, 128                 # the 27B's value heads, key heads, head width


def inputs(T: int, seed: int = 0, corr: float = 0.7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(1, T, HK, D, device="cuda", generator=g) * 0.5
    k = torch.randn(1, T, HK, D, device="cuda", generator=g) * 0.5
    common = torch.randn(1, 1, HK, D, device="cuda", generator=g)
    q, k = q * (1 - corr) + common * corr, k * (1 - corr) + common * corr
    q = gdn.l2norm(q.bfloat16(), dim=-1, eps=1e-6)
    k = gdn.l2norm(k.bfloat16(), dim=-1, eps=1e-6)
    v = (torch.randn(1, T, H, D, device="cuda", generator=g) * 0.5).bfloat16()
    gate = -torch.rand(1, T, H, device="cuda", generator=g) * 0.3
    beta = torch.rand(1, T, H, device="cuda", generator=g).bfloat16()
    S0 = torch.randn(1, H, D, D, device="cuda", generator=g) * 0.05
    return q, k, v, gate, beta, S0


def truth(q, k, v, gate, beta, S0):
    """The gated delta rule one token at a time in fp64."""
    rep = H // HK
    qd = q[0].double().repeat_interleave(rep, 1) * D ** -0.5
    kd = k[0].double().repeat_interleave(rep, 1)
    vd, gd, bd = v[0].double(), gate[0].double(), beta[0].double()
    S = S0[0].double().clone()
    o = torch.empty(q.shape[1], H, D, dtype=torch.float64, device="cuda")
    for t in range(q.shape[1]):
        S = S * gd[t].exp()[:, None, None]
        pred = torch.einsum("hk,hkv->hv", kd[t], S)
        S = S + torch.einsum("hk,hv->hkv", kd[t], (vd[t] - pred) * bd[t][:, None])
        o[t] = torch.einsum("hk,hkv->hv", qd[t], S)
    return o, S


def test_close_to_the_shipped_path_and_the_fp64_reference():
    for T, seed in ((1024, 0), (1000, 1), (96, 2)):
        args = inputs(T, seed)
        with torch.no_grad():
            o, S = chunked_prefill_normed(*args)
            ro, rS = chunk_prefill_normed(*args, chunk_size=64, bv=16, warps=4)
            to, tS = truth(*args)
        scale = ro.float().abs().max().item()
        assert (o.float() - ro.float()).abs().max().item() <= 4 * scale * 2 ** -8, T
        assert ((S - rS).abs().max() / rS.abs().max()).item() < 1e-4, T
        err = lambda x, y: ((x.double() - y).abs().max() / y.abs().max()).item()   # noqa: E731
        assert err(o[0], to) <= 1.05 * err(ro[0], to) + 1e-4, T
        assert err(S[0], tS) <= 1.5 * err(rS[0], tS), T
        assert o.dtype == ro.dtype and S.dtype == torch.float32 and S.shape == rS.shape


def test_a_window_split_at_a_chunk_boundary_is_the_one_call():
    q, k, v, gate, beta, S0 = inputs(2048, 3)
    with torch.no_grad():
        o, S = chunked_prefill_normed(q, k, v, gate, beta, S0)
        o1, S1 = chunked_prefill_normed(q[:, :1024], k[:, :1024], v[:, :1024], gate[:, :1024],
                                        beta[:, :1024], S0)
        o2, S2 = chunked_prefill_normed(q[:, 1024:], k[:, 1024:], v[:, 1024:], gate[:, 1024:],
                                        beta[:, 1024:], S1)
    assert torch.equal(torch.cat([o1, o2], dim=1), o) and torch.equal(S2, S)


def test_a_fresh_sequence_is_a_zero_state():
    q, k, v, gate, beta, _ = inputs(256, 4)
    with torch.no_grad():
        o, S = chunked_prefill_normed(q, k, v, gate, beta, None)
        oz, Sz = chunked_prefill_normed(q, k, v, gate, beta, torch.zeros(1, H, D, D, device="cuda"))
    assert torch.equal(o, oz) and torch.equal(S, Sz)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)
