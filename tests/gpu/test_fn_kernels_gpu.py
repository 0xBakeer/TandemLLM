"""On the board (ENG-223, ENG-141, ENG-143, ENG-149): the Flash-Next kernels of tools/fn_kernels.py.

  * gemv (int8 rows, fp16) and mix_up equal their torch arithmetic within fp32 summation order, and
    every row of an M-row call (M = 2..32) is the M = 1 call's row bit for bit;
  * the router equals its torch arithmetic (ids; weights within one bf16 step), resolves an exact tie
    to the lower index, and puts the shared expert in the last slot;
  * the MoE kernels on random EXL3 experts at the real shapes (hidden 2560, expert 640): the one-row
    output equals the fixed-order torch reference within the BF16 bound below; every row of an M-row
    call equals its M = 1 call bit for bit, with rows sharing experts; experts no row selects are
    never read (their trellis is overwritten and nothing changes); every K the packs use passes.

Run on the box: python tests/gpu/test_fn_kernels_gpu.py   (under flock ~/.qwen38-box.flock)
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine.models.flashnext.pack import Int8Rows  # noqa: E402
from tools import exl3_linear as X  # noqa: E402
from tools import fn_kernels as K  # noqa: E402

H, I = 2560, 640
DEV = "cuda"
# the MoE one-row bound: the kernels and the reference make the same bf16 roundings (gate/up out,
# silu, the product, the final sum); only fp32 summation order differs, which can flip a bf16
# rounding of an intermediate. Measured max |diff| / max |ref| stays below 1e-2 (two bf16 steps
# of the largest output); a wrong expert, order or rotation is O(1).
MOE_REL = 1e-2


def _x(M, K_, seed, scale=1.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(M, K_, generator=g, device=DEV) * scale).to(torch.bfloat16)


def test_gemv_reference_and_rows():
    out = []
    for N, Kk, kind in ((324, 10240, "int8"), (513, 2560, "fp16"), (96, 2560, "fp16"), (10240, 2560, "fp16")):
        g = torch.Generator(device=DEV).manual_seed(N)
        w16 = (torch.randn(N, Kk, generator=g, device=DEV) * 0.02).half()
        w = Int8Rows.of(w16) if kind == "int8" else w16
        wf = w.dequant() if kind == "int8" else w16.float()
        x = _x(32, Kk, 1)
        y = K.gemv(x, w)
        ref = x.float() @ wf.T
        rel = ((y - ref).abs().max() / ref.abs().max()).item()
        assert rel < 1e-5, (N, Kk, kind, rel)
        one = torch.cat([K.gemv(x[m:m + 1], w) for m in range(32)])
        for M in (2, 5, 16, 32):
            assert torch.equal(K.gemv(x[:M], w), one[:M]), (N, Kk, kind, M)
        out.append(f"{N}x{Kk} {kind} rel {rel:.0e}")
    return "; ".join(out) + "; rows 1..32 bit-equal"


def test_mix_up_reference_and_rows():
    R, HC = 320, 4
    for kind in ("int8", "fp16"):
        g = torch.Generator(device=DEV).manual_seed(5)
        w16 = (torch.randn(HC * H, R, generator=g, device=DEV) * 0.05).half()
        w = Int8Rows.of(w16) if kind == "int8" else w16
        wf = w.dequant() if kind == "int8" else w16.float()
        d = torch.randn(32, R, generator=g, device=DEV) * 2
        n = _x(32, HC * H, 6)
        x = K.mix_up(d, n, w, HC)
        a = torch.nn.functional.silu((d.to(torch.bfloat16).float() / HC).to(torch.bfloat16).float())
        a = a.to(torch.bfloat16).float()
        u = torch.sigmoid((a @ wf.T).to(torch.bfloat16).float()).to(torch.bfloat16).float()
        p = (u * n.float()).to(torch.bfloat16).float().view(32, HC, H)
        ref = ((((p[:, 0] + p[:, 1]) + p[:, 2]) + p[:, 3]) / HC).to(torch.bfloat16)
        diff = (x.float() - ref.float()).abs().max().item()
        assert diff <= 2 * ref.float().abs().max().item() * 2 ** -7, (kind, diff)
        one = torch.cat([K.mix_up(d[m:m + 1], n[m:m + 1], w, HC) for m in range(32)])
        for M in (2, 7, 32):
            assert torch.equal(K.mix_up(d[:M], n[:M], w, HC), one[:M]), (kind, M)
    return "int8 and fp16 up weights: reference within one bf16 step, rows 1..32 bit-equal"


def test_router():
    g = torch.Generator(device=DEV).manual_seed(7)
    L = torch.randn(64, 513, generator=g, device=DEV) * 3
    L[0, 10] = L[0, 20] = 50.0                      # an exact tie at the top: id 10 before 20
    ids, wts = K.router(L, 512, 10)
    rid, rw = K.router_reference(L, 512, 10)
    assert torch.equal(ids, rid), (ids[:2], rid[:2])
    assert (wts - rw).abs().max().item() <= 2 ** -8
    assert ids[0, 0].item() == 10 and ids[0, 1].item() == 20 and (ids[:, 10] == 512).all()
    one = torch.cat([torch.cat(K.router(L[m:m + 1], 512, 10), 1) for m in range(64)])
    both = torch.cat([ids.float(), wts], 1)
    assert torch.equal(one, both)
    return "64 rows: ids equal, weights within one bf16 step, tie -> lower index, shared slot 512"


def _bf_steps(a: torch.Tensor, b: torch.Tensor) -> float:
    """max |a - b| in units of the bf16 step at b's magnitude (0 = equal)."""
    step = torch.where(b == 0, torch.full_like(b, 2 ** -133), b.abs() * 2 ** -7)
    return ((a.float() - b.float()).abs() / step.float()).max().item()


def test_hc_norm_and_inject_norm():
    from engine.models.flashnext.ref import rms
    g = torch.Generator(device=DEV).manual_seed(31)
    M, hc = 32, 4
    h0 = (torch.randn(M, hc * H, generator=g, device=DEV) * 2).to(torch.bfloat16)
    out = torch.randn(M, H, generator=g, device=DEV).to(torch.bfloat16)
    d = torch.randn(M, 324, generator=g, device=DEV) * 3
    for wdt in (torch.float16, torch.bfloat16):
        w = (torch.randn(hc * H, generator=g, device=DEV) * 0.1).to(wdt)
        n = K.hc_norm(h0, w, H, 1e-6)
        assert _bf_steps(n, rms(h0, w, 1e-6, group=H)) <= 1.0
        hn, nn = K.inject_norm(h0, out, d, 320, w, hc, 1e-6)
        inj = 2 * torch.sigmoid(d[:, 320:].to(torch.bfloat16) / hc)
        want_h = h0 + (out.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
        assert torch.equal(hn, want_h)
        assert _bf_steps(nn, rms(want_h, w, 1e-6, group=H)) <= 1.0
        h1, n1 = K.inject_norm(h0, out, d, 320, None, hc, 1e-6)
        assert torch.equal(h1, want_h) and n1 is None
        one = torch.cat([K.inject_norm(h0[m:m + 1], out[m:m + 1], d[m:m + 1], 320, w, hc, 1e-6)[1] for m in range(M)])
        assert torch.equal(one, nn)
        assert torch.equal(torch.cat([K.hc_norm(h0[m:m + 1], w, H, 1e-6) for m in range(M)]), n)
    return "norm within one bf16 step of torch's, inject bit-equal to torch's expression, rows bit-equal"


def test_gated_norm():
    g = torch.Generator(device=DEV).manual_seed(32)
    o = torch.randn(48 * 3, 128, generator=g, device=DEV).to(torch.bfloat16)
    z = torch.randn(48 * 3, 128, generator=g, device=DEV).to(torch.bfloat16)
    w = (torch.randn(128, generator=g, device=DEV) * 0.3 + 1).to(torch.bfloat16)
    for sig in (True, False):
        y = K.gated_norm(o, z, w, 1e-6, sig)
        hh = o.float()
        hh = hh * torch.rsqrt(hh.pow(2).mean(-1, keepdim=True) + 1e-6)
        hh = w * hh.to(torch.bfloat16)
        act = torch.sigmoid if sig else torch.nn.functional.silu
        want = (hh * act(z.float())).to(torch.bfloat16)
        assert _bf_steps(y, want) <= 2.0, sig
        assert torch.equal(K.gated_norm(o[:48], z[:48], w, 1e-6, sig), y[:48])
    return "sigmoid and silu gates within two bf16 steps of torch's expression; rows independent"


def test_gemv_bf16_out():
    g = torch.Generator(device=DEV).manual_seed(33)
    w = (torch.randn(96, H, generator=g, device=DEV) * 0.02).half()
    x = _x(5, H, 34)
    assert torch.equal(K.gemv_bf16(x, w), K.gemv(x, w).to(torch.bfloat16))
    return "the bf16 output is the fp32 output rounded"


def _bank(n_exp: int, kbs, shared_k: int, seed: int = 0):
    blocks = []
    for e in range(n_exp + 1):
        kb = shared_k if e == n_exp else kbs[e % len(kbs)]
        gt = X.random_block(I, H, kb, seed=seed + 3 * e)
        up = X.random_block(I, H, kb, seed=seed + 3 * e + 1)
        dn = X.random_block(H, I, kb, seed=seed + 3 * e + 2)
        blocks.append((gt, up, dn))
    trip = [tuple((b.t, b.suh[0].contiguous(), b.svh) for b in eb) for eb in blocks]
    return K.MoEBank(trip, DEV), blocks


def _ids(M, n_exp, k, seed, share=True):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for m in range(M):
        if share and m % 3 == 1 and rows:
            r = rows[-1][:]                                  # the same experts as the previous row
            r[0], r[1] = r[1], r[0]
        else:
            r = torch.randperm(n_exp, generator=g)[:k].tolist()
        rows.append(r)
    ids = torch.tensor([r + [n_exp] for r in rows], dtype=torch.int32, device=DEV)
    w = torch.rand(M, k + 1, generator=g).to(torch.bfloat16).float().to(DEV)
    return ids, w


def test_moe_reference_one_row():
    bank, blocks = _bank(24, (3,), 5)
    worst = 0.0
    for seed in range(4):
        x = _x(1, H, 10 + seed, 0.5)
        ids, w = _ids(1, 24, 10, seed)
        got = K.moe(x, ids, w, bank)
        ref = K.moe_reference(x, ids.cpu(), w.cpu(), blocks)
        rel = ((got.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
        worst = max(worst, rel)
        assert rel < MOE_REL, (seed, rel)
    return f"10 routed (K3) + shared (K5) vs fixed-order torch: worst max|diff|/max|ref| {worst:.1e}"


def test_moe_rows_equal_one_row_call():
    bank, _ = _bank(24, (3,), 5)
    x = _x(32, H, 3, 0.5)
    ids, w = _ids(32, 24, 10, 9)
    one = torch.cat([K.moe(x[m:m + 1], ids[m:m + 1], w[m:m + 1], bank) for m in range(32)])
    for M in range(2, 33):
        got = K.moe(x[:M], ids[:M], w[:M], bank)
        assert torch.equal(got, one[:M]), M
    return "M = 2..32 (every third row shares the previous row's experts): bit-equal to M = 1"


def test_moe_reads_only_selected_experts():
    bank, blocks = _bank(64, (3,), 5)
    x = _x(2, H, 4, 0.5)
    ids, w = _ids(2, 64, 10, 11, share=False)
    want = K.moe(x, ids, w, bank)
    used = set(ids.flatten().tolist())
    for e in range(64):
        if e not in used:
            for b in blocks[e]:
                b.t.copy_(torch.randint(-32768, 32767, b.t.shape, dtype=torch.int32, device=DEV).to(torch.int16))
                b.svh.fill_(float("nan"))
    got = K.moe(x, ids, w, bank)
    assert torch.equal(got, want)
    return f"{64 - len(used & set(range(64)))} of 64 experts unselected and overwritten (trellis random, svh NaN): output unchanged"


def test_moe_every_k():
    out = []
    for kb in (2, 3, 4, 5, 6, 8):
        bank, blocks = _bank(12, (kb,), 6 if kb != 6 else 5, seed=100 * kb)
        x = _x(8, H, kb, 0.5)
        ids, w = _ids(8, 12, 10, kb)
        got = K.moe(x, ids, w, bank)
        ref = K.moe_reference(x[:1], ids[:1].cpu(), w[:1].cpu(), blocks)
        rel = ((got[:1].float() - ref.float()).abs().max() / ref.float().abs().max()).item()
        assert rel < MOE_REL, (kb, rel)
        one = torch.cat([K.moe(x[m:m + 1], ids[m:m + 1], w[m:m + 1], bank) for m in range(8)])
        assert torch.equal(got, one), kb
        out.append(f"K{kb} {rel:.0e}")
    return ", ".join(out)


def test_moe_graph_capture():
    bank, _ = _bank(24, (3,), 5)
    x = _x(1, H, 21, 0.5)
    ids, w = _ids(1, 24, 10, 21)
    bufs = {}
    out = torch.empty(1, H, dtype=torch.bfloat16, device=DEV)
    order = torch.empty(11, dtype=torch.int32, device=DEV)
    K.moe(x, ids, w, bank, bufs=bufs, out=out)          # build + warm
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        K.moe(x, ids, w, bank, bufs=bufs, out=out)
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        order.copy_(K.expert_order(ids))
        K.moe(x, ids, w, bank, order=order, bufs=bufs, out=out)
    want = K.moe(x, ids, w, bank)
    ids2, w2 = _ids(1, 24, 10, 22)
    ids.copy_(ids2)
    w.copy_(w2)
    gr.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, K.moe(x, ids2, w2, bank)) and not torch.equal(out, want)
    return "captured once, replayed with new expert ids in the device buffer: equal to eager"


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                print(f"  {name:<40} ok   {fn() or ''}", flush=True)
                passed += 1
            except AssertionError as e:
                print(f"  {name:<40} FAIL {e}", flush=True)
                failed += 1
    print(f"{passed} passed" + (f", {failed} FAILED" if failed else ""))
    sys.exit(1 if failed else 0)
