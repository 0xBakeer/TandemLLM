"""On the board: the Kolibri-1 kernels against their torch paths, at the served shapes.

  * FP8 block linear with fp32 scales (q|k|v fused 7168x2560, o 2560x6144) at 1, 7 and 300 rows;
  * the NVFP4 MoE (gate/up/down 512x2560, 2560x512, 33 experts) at 1 row (decode), 16 and 700 rows,
    with a shared column, against `moe_experts_torch`; a row alone equals the row inside a batch
    (row invariance) bit for bit for every row count up to SMALL_M (decode and verify share one program shape);
  * the e4m3 head GEMV at one row against the chunked fp32 product;
  * the tiny random Kolibri on cuda against the same model on the CPU.

    python -m pytest -q tests/gpu/test_kolibri_gpu.py     (or run the file: it calls every test)
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from engine.kolibri import kernels as K  # noqa: E402
from tools import kolibri_nvfp4 as kn  # noqa: E402

G = torch.Generator(device="cuda").manual_seed(3)


def _fp8(N, Kd):
    w = torch.randn(N, Kd, device="cuda", generator=G) * 0.02
    s = w.reshape(N // 128, 128, Kd // 128, 128).abs().amax((1, 3)) / 448.0
    codes = (w / s.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(torch.float8_e4m3fn)
    return K.FP8Linear(codes, s)


def _close(a, b, tol):
    err = (a.float() - b.float()).abs().max().item()
    sc = b.float().abs().max().item()
    assert err <= tol * sc, (err, sc)


def test_fp8_linear():
    for N, Kd in ((7168, 2560), (2560, 6144)):
        lin = _fp8(N, Kd)
        W = lin.dense()
        for M in (1, 7, 300):
            x = (torch.randn(M, Kd, device="cuda", generator=G)).to(torch.bfloat16)
            _close(lin.matmul(x), x.float() @ W.T, 0.01)


def _bank(E, N, Kd):
    w = torch.randn(E, N, Kd, device="cuda", generator=G) * 0.03
    c, s, s2 = kn.clip_batched(w, None)
    return K.ExpertBank(c, s, s2[:, 0, 0])


def test_moe_and_row_invariance():
    E, H, I = 33, 2560, 512
    Gb, Ub, Db = _bank(E, I, H), _bank(E, I, H), _bank(E, H, I)
    for M in (1, 16, 700):
        x = torch.randn(M, H, device="cuda", generator=G).to(torch.bfloat16)
        logits = torch.randn(M, E - 1, device="cuda", generator=G)
        ids, w = K.route(x, torch.randn(E - 1, H, device="cuda", generator=G) * 0.02,
                         torch.zeros(E - 1, device="cuda"), 6, E - 1)
        y = K.moe_experts(x, ids, w, Gb, Ub, Db)
        yr = K.moe_experts_torch(x, ids, w, Gb, Ub, Db)
        _close(y, yr, 0.01)
        if 1 < M <= K.SMALL_M:
            y0 = K.moe_experts(x[3:4], ids[3:4], w[3:4], Gb, Ub, Db)
            assert torch.equal(y0[0], y[3]), "row invariance"


def test_nvfp4_dense():
    w = torch.randn(7168, 2560, device="cuda", generator=G) * 0.02
    c, s, s2 = kn.clip_batched(w[None], None)
    lin = K.NVFP4Linear(c[0], s[0], s2[0, :, 0])
    for M in (1, 40):
        x = torch.randn(M, 2560, device="cuda", generator=G).to(torch.bfloat16)
        _close(lin.matmul(x), x.float() @ lin.dense().T, 0.01)


def test_head():
    w = torch.randn(128000, 2560, device="cuda", generator=G) * 0.02
    hc, hs = kn.quant_head_e4m3(w)
    h = K.E4M3Head(hc, hs)
    x = torch.randn(1, 2560, device="cuda", generator=G)
    a = h.logits(x)
    b = h.logits(torch.cat([x, x]))[0:1]
    _close(a, b, 1e-4)


def test_tiny_model_cuda_equals_cpu():
    from engine.kolibri.model import KolibriEngine
    from tools import kolibri_tiny as tiny
    ids = [5, 17, 3, 88, 41, 41, 2, 60, 7, 19, 33, 71, 12, 9]
    with tempfile.TemporaryDirectory() as d:
        st, rel = tiny.write(d)
        cpu = KolibriEngine.load(st, rel, device="cpu", max_len=64, attention="torch", log=lambda s: None)
        gpu = KolibriEngine.load(st, rel, device="cuda", max_len=64, attention="torch", log=lambda s: None)
        a, b = cpu.forward(ids), gpu.forward(ids).cpu()
        sc = a.abs().max().item()
        row_err = (a - b).abs().amax(-1)
        assert (row_err <= 0.03 * sc).float().mean() >= 0.8, row_err / sc
        gpu.reset()
        gpu.prefill(ids[:5])
        lg = [gpu.decode(t).cpu() for t in ids[5:]]
        assert (torch.stack(lg).argmax(-1) == b[5:].argmax(-1)).float().mean() >= 0.8


def test_fused_glue():
    H = 2560
    for M in (1, 300):
        r = torch.randn(M, H, device="cuda", generator=G) * 3
        y = torch.randn(M, H, device="cuda", generator=G).to(torch.bfloat16)
        w1 = (1 + 0.1 * torch.randn(H, device="cuda", generator=G)).to(torch.bfloat16)
        w2 = (1 + 0.1 * torch.randn(H, device="cuda", generator=G)).to(torch.bfloat16)
        ro, xo = K.add_rms2(r, y, w1, w2, 1e-6)
        r_ref = r + K.rms(y, w1, 1e-6, torch.float32)
        assert torch.allclose(ro, r_ref, atol=1e-5, rtol=1e-5)
        _close(xo, K.rms(r_ref, w2, 1e-6), 0.01)
        _close(K.rms_fused(r, w2, 1e-6), K.rms(r, w2, 1e-6), 0.01)
        gate = torch.randn(384, H, device="cuda", generator=G) * 0.02
        bias = torch.randn(384, device="cuda", generator=G) * 0.01
        x = torch.randn(M, H, device="cuda", generator=G).to(torch.bfloat16)
        i1, w1_ = K.route_fused(x, gate, bias, 6, 384)
        gb = gate.to(torch.bfloat16)
        lg_k = K.router_logits(x, gb)
        assert torch.allclose(lg_k, x.float() @ gb.float().T, atol=1e-4, rtol=1e-5)
        i2, w2_ = K.route_fused(x, gb, bias, 6, 384)
        i3, w3_ = K.route(x, gb, bias, 6, 384)
        if M <= 32:
            assert (i2 == i3).float().mean() > 0.99 and torch.allclose(w2_, w3_, atol=1e-5)
        i0, w0 = K.route(x, gate, bias, 6, 384)
        assert torch.equal(i1, i0)
        assert torch.allclose(w1_, w0, atol=1e-6)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n, flush=True)
