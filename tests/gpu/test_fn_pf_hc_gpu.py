"""On the board (SPD-249 P2): tools/fn_pf_hc.py equals the prefill mixer and inject it replaces, bit for bit.

  * mix: (x, di) equal `_mix_pre`'s replaced path (fn_prefill_kernels `mm_rows`, the torch silu, `mm_rows`,
    `mix_tail`) in each `mm_rows` mode (QWEN38_FN_PREFILL_MM fp32, tc, tcn, fp16) at the mixer shapes for every row
    count in ROWS (`--all-rows`: 1 .. 4,096 each for fp32 and tc), with n's fp32 form given (as the inject's launch
    makes it) and without it, and x written into rows of a larger buffer (a window's `out`);
  * mix_a, fp32 and bf16 output, over every bf16 value of bf16(di) (a NaN aside: both NaN);
  * inject_norm: (h, n) equal tools/fn_kernels.inject_norm's at stream widths 2,560 (the model's), 512, 1,024,
    2,048, 3,000 and 4,096, norm weights fp16, bf16 and none, every row count in ROWS; its n32 is n.float();
  * the times of both paths at 4,096 rows (printed, not asserted).

Run on the box: python tests/gpu/test_fn_pf_hc_gpu.py [--all-rows]
"""

from __future__ import annotations

import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine.models.flashnext.pack import Int8Rows  # noqa: E402
from tools import fn_kernels as K  # noqa: E402
from tools import fn_pf_hc as HC  # noqa: E402
from tools import fn_prefill_kernels as PK  # noqa: E402

DEV = "cuda"
H, HCN, R = 2560, 4, 320
ROWS = sorted(set(list(range(1, 70)) + [96, 127, 128, 129, 255, 256, 500, 512, 1000, 1023, 1024, 1025, 1500, 2047,
                                         2048, 2049, 3000, 3333, 4000, 4095, 4096]))


def _w(N, K_, G, seed, scale=1e-3):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return Int8Rows.of((torch.randn(N, K_, generator=g, device=DEV) * scale).half(), G)


def _n(M, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(M, HCN * H, generator=g, device=DEV).to(torch.bfloat16)


def _same(x, y) -> bool:
    """Bit equality."""
    return x.shape == y.shape and x.dtype == y.dtype and bool(torch.equal(x.contiguous().view(torch.uint8),
                                                                           y.contiguous().view(torch.uint8)))


def _inj_inputs(M, G, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    h0 = (torch.randn(M, 4 * G, generator=g, device=DEV) * 3).to(torch.bfloat16)
    out = (torch.randn(M, G, generator=g, device=DEV) * 2).to(torch.bfloat16)
    d = torch.randn(M, R + 4, generator=g, device=DEV) * 4
    return h0, out, d


def _mix_mode(mode, rows):
    wd, wu = _w(R + HCN, HCN * H, 128, 31), _w(HCN * H, R, 64, 32)
    assert HC.mix_ok(_n(4, 0), HCN)
    bad = []
    saved = PK.MM
    PK.MM = mode
    try:
        for M in rows:
            n = _n(M, 700 + M)
            x, di = HC.mix_reference(n, wd, wu, R, HCN)
            for n32 in (None, n.float()) if HC.wants_n32() else (None,):
                x2, di2 = HC.mix(n, n32, wd, wu, R, HCN)
                if not (_same(x, x2) and _same(di, di2)):
                    bad.append((M, n32 is not None, int((x != x2).sum()), int((di != di2).sum())))
            buf = torch.zeros(M + 2, H, dtype=torch.bfloat16, device=DEV)
            x3, _ = HC.mix(n, None, wd, wu, R, HCN, buf[1:M + 1])
            if not (x3.data_ptr() == buf[1].data_ptr() and _same(x, x3) and not buf[0].any() and not buf[-1].any()):
                bad.append((M, "out", int((x != x3).sum()), 0))
    finally:
        PK.MM = saved
    assert not bad, f"mix ({mode}) differs at (rows, n32, x diffs, di diffs) {bad[:12]}"
    return f"{len(rows)} row counts"


def test_mix_equals_the_replaced_path(rows=None):
    return _mix_mode("fp32", rows or ROWS)


def test_mix_equals_the_replaced_path_tc(rows=None):
    return _mix_mode("tc", rows or ROWS)


def test_mix_equals_the_replaced_path_tcn():
    return _mix_mode("tcn", ROWS)


def test_mix_equals_the_replaced_path_fp16():
    return _mix_mode("fp16", ROWS)


def test_mix_a_every_bf16():
    bits = torch.arange(-32768, 32768, dtype=torch.int32, device=DEV).to(torch.int16)
    v = bits.view(torch.bfloat16)
    d = torch.zeros(v.numel(), 8, device=DEV)
    d[:, 3] = v.float()
    ref = torch.nn.functional.silu(d[:, :4].to(torch.bfloat16).float() / HCN).to(torch.bfloat16)
    for fp32 in (True, False):
        want = ref.float() if fp32 else ref
        got = HC.mix_a(d, 4, HCN, fp32=fp32)
        assert got.dtype == want.dtype
        it = torch.int32 if fp32 else torch.int16
        ok = (want.view(it) == got.view(it)) | (torch.isnan(want) & torch.isnan(got))
        assert bool(ok.all()), f"mix_a (fp32 {fp32}) differs at {int((~ok).sum())} values"
    return f"{v.numel()} bf16 values, fp32 and bf16 out"


def test_inject_norm_equals_fn_kernels(rows=None):
    bad = []
    n_cases = 0
    for G in (2560, 512, 1024, 2048, 3000, 4096):
        g = torch.Generator(device=DEV).manual_seed(G)
        wts = {"fp16": (torch.randn(4 * G, generator=g, device=DEV) * 0.1).half(),
               "bf16": (torch.randn(4 * G, generator=g, device=DEV) * 0.1).to(torch.bfloat16), "none": None}
        for M in (rows or ROWS) if G == 2560 else (1, 3, 64, 1000):
            h0, out, d = _inj_inputs(M, G, M * 7 + G)
            for wn, w in wts.items():
                h, n = K.inject_norm(h0, out, d, R, w, 4, 1e-6)
                h2, n2, _ = HC.inject_norm(h0, out, d, R, w, 4, 1e-6)
                n_cases += 1
                if not (_same(h, h2) and (n is None and n2 is None or _same(n, n2))):
                    bad.append((G, M, wn))
    assert not bad, f"inject_norm differs at (G, rows, weight) {bad[:12]}"
    return f"{n_cases} cases"


def test_inject_n32_is_n_float():
    h0, out, d = _inj_inputs(77, H, 3)
    w = (torch.randn(4 * H, device=DEV) * 0.1).half()
    h, n, n32 = HC.inject_norm(h0, out, d, R, w, 4, 1e-6, n32=True)
    assert _same(n32, n.float())
    assert HC.inject_norm(h0, out, d, R, None, 4, 1e-6, n32=True)[2] is None


def _ms(f, reps=20):
    f()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def test_times_at_4096_rows():
    wd, wu = _w(R + HCN, HCN * H, 128, 51), _w(HCN * H, R, 64, 52)
    h0, o, d = _inj_inputs(4096, H, 5)
    w = (torch.randn(4 * H, device=DEV) * 0.1).half()

    def old():
        _, n = K.inject_norm(h0, o, d, R, w, 4, 1e-6)
        return HC.mix_reference(n, wd, wu, R, HCN)

    def new():
        _, n, n32 = HC.inject_norm(h0, o, d, R, w, 4, 1e-6, n32=True)
        return HC.mix(n, n32, wd, wu, R, HCN)

    _, n = K.inject_norm(h0, o, d, R, w, 4, 1e-6)
    out = {"site replaced (inject + mixer)": _ms(old), "site fused": _ms(new),
           "inject_norm fn_kernels": _ms(lambda: K.inject_norm(h0, o, d, R, w, 4, 1e-6)),
           "inject_norm fused + n32": _ms(lambda: HC.inject_norm(h0, o, d, R, w, 4, 1e-6, n32=True)),
           "mixer replaced": _ms(lambda: HC.mix_reference(n, wd, wu, R, HCN)),
           "n.float()": _ms(lambda: n.float()),
           "torch silu chain": _ms(lambda: torch.nn.functional.silu(d[:, :R].to(torch.bfloat16).float() / HCN).to(
               torch.bfloat16).float()),
           "mix_a": _ms(lambda: HC.mix_a(d, R, HCN))}
    return " ".join(f"{k} {v:.3f} ms;" for k, v in out.items())


if __name__ == "__main__":
    all_rows = "--all-rows" in sys.argv
    fails = 0
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        try:
            if all_rows and name in ("test_mix_equals_the_replaced_path", "test_mix_equals_the_replaced_path_tc",
                                     "test_inject_norm_equals_fn_kernels"):
                got = fn(rows=list(range(1, 4097)))
            else:
                got = fn()
            print(f"  {name:60s} ok{'  ' + got if isinstance(got, str) else ''}", flush=True)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  {name:60s} FAIL {type(e).__name__}: {str(e)[:600]}", flush=True)
    print(f"{len(tests) - fails} passed, {fails} failed")
    sys.exit(1 if fails else 0)
