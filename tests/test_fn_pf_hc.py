"""SPD-249 P2: the fused prefill mixer's and inject's switches and dispatch (CPU; the bits are
tests/gpu/test_fn_pf_hc_gpu.py's and tools/fn_pf_hc_check.py's).

  * QWEN38_FN_PF_HC_FUSED and QWEN38_FN_PF_INJ_FUSED are Flash-Next knobs, off by default, owned by the engine
  * `fn_pf_hc.mix_ok` / `inject_ok` refuse CPU tensors
  * `_mix_pre` takes the replaced path when its switch is off and `fn_pf_hc.mix` otherwise, whatever `mm_rows` mode
    runs; `_inject_pre` likewise with `fn_kernels.inject_norm` / `fn_pf_hc.inject_norm`
  * `fn_pf_hc.mix` gives the fp32 GEMMs fp32 inputs and every other mode `mm_rows` itself with bf16 inputs
  * n's fp32 form goes from an inject to the next mixer of that same n only (by identity), and `_n32_clear` drops
    what no mixer took (stand-in modules record the calls)

Run: python tests/test_fn_pf_hc.py
"""

from __future__ import annotations

import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_knobs_are_registered_off():
    from engine.settings import KNOBS
    from engine.models.flashnext import engine as E
    for k in ("FN_PF_HC_FUSED", "FN_PF_INJ_FUSED"):
        assert KNOBS[k] == ("0", "engine/models/flashnext/engine.py"), k
    assert E.PF_HC_FUSED is (os.environ.get("QWEN38_FN_PF_HC_FUSED") == "1")
    assert E.PF_INJ_FUSED is (os.environ.get("QWEN38_FN_PF_INJ_FUSED") == "1")


def test_cpu_tensors_are_refused():
    from tools import fn_pf_hc as HC
    n = torch.zeros(3, 10240, dtype=torch.bfloat16)
    assert not HC.mix_ok(n, 4)
    assert not HC.inject_ok(n, torch.zeros(3, 2560, dtype=torch.bfloat16), 4)


class _Stubs:
    """fn_pf_hc, fn_prefill_kernels' mm_rows / mix_tail / MM and fn_kernels.inject_norm replaced by recorders."""

    def __init__(self):
        import tools
        from tools import fn_kernels as K
        from tools import fn_prefill_kernels as PK
        self.tools, self.K, self.PK = tools, K, PK
        self.calls = []
        c = self.calls
        self.hc = types.SimpleNamespace(
            mix_ok=lambda n, hc: True,
            wants_n32=lambda: PK.MM == "fp32",
            mix=lambda n, n32, wd, wu, R, hc: c.append(("mix", n32)) or ("x", "d"),
            inject_ok=lambda h, out, hc: True,
            inject_norm=lambda h, out, d, R, w, hc, eps, n32=False: c.append(("inject", n32)) or (
                h, torch.zeros(1), torch.ones(1) if n32 else None))
        self.saved = (sys.modules.get("tools.fn_pf_hc"), getattr(tools, "fn_pf_hc", None), PK.MM, PK.mm_rows,
                      PK.mix_tail, K.inject_norm)

    def __enter__(self):
        sys.modules["tools.fn_pf_hc"] = self.hc
        self.tools.fn_pf_hc = self.hc
        c = self.calls
        self.PK.mm_rows = lambda x, w: c.append(("mm_rows", w)) or torch.zeros(2, 324)
        self.PK.mix_tail = lambda u, n, hc: c.append(("mix_tail",)) or "x0"
        self.K.inject_norm = lambda h, out, d, R, w, hc, eps: c.append(("K.inject",)) or (h, torch.zeros(1))
        return self

    def __exit__(self, *a):
        m, attr, self.PK.MM, self.PK.mm_rows, self.PK.mix_tail, self.K.inject_norm = self.saved
        if m is None:
            sys.modules.pop("tools.fn_pf_hc", None)
        else:
            sys.modules["tools.fn_pf_hc"] = m
        if attr is None:
            if hasattr(self.tools, "fn_pf_hc"):
                delattr(self.tools, "fn_pf_hc")
        else:
            self.tools.fn_pf_hc = attr


def _eng():
    from engine.models.flashnext import engine as E
    eng = E.FlashNextEngine.__new__(E.FlashNextEngine)
    eng.cfg = types.SimpleNamespace(hc_count=4)
    eng.dtype = torch.bfloat16
    eng.W = {"s.down_inject": "wd", "s.input_mix_weight_up.weight": "wu", "s.rank": 320}
    return eng


def test_dispatch():
    from engine.models.flashnext import engine as E
    saved = (E.PF_HC_FUSED, E.PF_INJ_FUSED)
    try:
        with _Stubs() as s:
            eng = _eng()
            n = torch.zeros(2, 16, dtype=torch.bfloat16)
            for hc_on, mm, fused in ((False, "fp32", False), (True, "fp32", True), (True, "tc", True), (True, "fp16", True),
                                     (False, "tc", False)):
                s.calls.clear()
                E.PF_HC_FUSED, s.PK.MM = hc_on, mm
                out = eng._mix_pre(n, "s")
                kinds = [c[0] for c in s.calls]
                if fused:
                    assert out == ("x", "d") and kinds == ["mix"], (hc_on, mm, s.calls)
                else:
                    assert kinds == ["mm_rows", "mm_rows", "mix_tail"], (hc_on, mm, s.calls)
            for inj_on, hc_on, w, want in ((False, True, "w", ["K.inject"]), (True, False, "w", [("inject", False)]),
                                           (True, True, "w", [("inject", True)]), (True, True, None, [("inject", False)])):
                s.calls.clear()
                E.PF_INJ_FUSED, E.PF_HC_FUSED, s.PK.MM = inj_on, hc_on, "fp32"
                eng._inject_pre(torch.zeros(1), torch.zeros(1), torch.zeros(1), 320, w, 4, 1e-6)
                assert [c if c[0] == "inject" else c[0] for c in s.calls] == want, (inj_on, hc_on, w, s.calls)
                eng._n32_clear()
            for mm in ("tc", "fp16"):                                         # no fp32 form for a GEMM that reads bf16
                s.calls.clear()
                E.PF_INJ_FUSED, E.PF_HC_FUSED, s.PK.MM = True, True, mm
                eng._inject_pre(torch.zeros(1), torch.zeros(1), torch.zeros(1), 320, "w", 4, 1e-6)
                assert s.calls == [("inject", False)], (mm, s.calls)
    finally:
        E.PF_HC_FUSED, E.PF_INJ_FUSED = saved


def test_mix_follows_the_mm_rows_mode():
    from tools import fn_pf_hc as HC
    from tools import fn_prefill_kernels as PK
    saved = (PK.MM, PK.mm_rows, PK._wf, PK.mix_tail, HC.mix_a)
    calls = []
    q8 = types.SimpleNamespace(q=1)                                           # an int8 weight
    f16 = torch.zeros(3, 4, dtype=torch.float16)                              # an fp16 weight
    try:
        PK.mm_rows = lambda x, w: calls.append(("mm_rows", x.dtype)) or torch.zeros(2, 3)
        PK._wf = lambda w: calls.append(("wf",)) or torch.zeros(3, 4)
        PK.mix_tail = lambda u, n, hc: "x"
        HC.mix_a = lambda d, R, hc, fp32=True: calls.append(("mix_a", fp32)) or torch.zeros(
            2, 4, dtype=torch.float32 if fp32 else torch.bfloat16)
        n = torch.zeros(2, 4, dtype=torch.bfloat16)
        for mm, wd, wu, want in (
                ("fp32", q8, q8, [("wf",), ("mix_a", True), ("wf",)]),
                ("fp32", f16, f16, [("wf",), ("mix_a", True), ("wf",)]),
                ("tc", q8, q8, [("mm_rows", torch.bfloat16), ("mix_a", False), ("mm_rows", torch.bfloat16)]),
                ("tc", f16, q8, [("wf",), ("mix_a", False), ("mm_rows", torch.bfloat16)]),
                ("fp16", q8, f16, [("mm_rows", torch.bfloat16), ("mix_a", False), ("mm_rows", torch.bfloat16)])):
            calls.clear()
            PK.MM = mm
            assert HC.lib32(wd) is (mm == "fp32" or (mm == "tc" and wd is f16))
            assert HC.wants_n32() is (mm == "fp32")
            x, di = HC.mix(n, None, wd, wu, 3, 4)
            assert x == "x" and calls == want, (mm, calls)
    finally:
        PK.MM, PK.mm_rows, PK._wf, PK.mix_tail, HC.mix_a = saved


def test_n32_goes_to_the_next_mixer_of_the_same_n_only():
    from engine.models.flashnext import engine as E
    saved = (E.PF_HC_FUSED, E.PF_INJ_FUSED)
    try:
        with _Stubs() as s:
            E.PF_HC_FUSED = E.PF_INJ_FUSED = True
            s.PK.MM = "fp32"
            eng = _eng()
            _, n = eng._inject_pre(torch.zeros(1), torch.zeros(1), torch.zeros(1), 320, "w", 4, 1e-6)
            s.calls.clear()
            eng._mix_pre(torch.zeros(1, dtype=torch.bfloat16), "s")          # another n: no fp32 form
            eng._mix_pre(n, "s")                                              # this n: its fp32 form
            eng._mix_pre(n, "s")                                              # taken once
            assert [c[1] is not None for c in s.calls] == [False, True, False], s.calls
            _, n2 = eng._inject_pre(torch.zeros(1), torch.zeros(1), torch.zeros(1), 320, "w", 4, 1e-6)
            assert len(eng._n32) == 1
            eng._n32_clear()
            assert not eng._n32
    finally:
        E.PF_HC_FUSED, E.PF_INJ_FUSED = saved


if __name__ == "__main__":
    fails = 0
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        try:
            fn()
            print(f"  {name:55s} ok")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  {name:55s} FAIL {type(e).__name__}: {e}")
    print(f"{len(tests) - fails} passed, {fails} failed")
    sys.exit(1 if fails else 0)
