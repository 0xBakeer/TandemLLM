"""The Ada kernels' load-probe contract (the PR #77 review): verdicts are recorded once by
`exl3_ada.probe` / `moe_ada.probe` at weight load; a dispatch that finds no verdict raises
rather than probe mid-call, and a kernel error after a True verdict propagates -- it never
silently changes the arithmetic a call uses. The GPU half runs the real probes on whatever
CUDA device is present (on the 4080 the verdicts must all come back True).
"""
import sys
import unittest.mock as mock

sys.path.insert(0, __file__.rsplit("/", 2)[0])

import torch

import tools.exl3_ada as EA
import tools.exl3_linear as X


def test_unprobed_dispatch_raises_not_probes():
    """rot_ok/decode_ok/gemv_ok/moe_ada.ok on a device and rate nothing probed raise --
    the caller asked for a kernel whose load probe never ran."""
    from tools import moe_ada
    dev = torch.device("cuda:7")
    for fn in (lambda: EA.rot_ok(dev), lambda: EA.decode_ok(dev, 2),
               lambda: EA.gemv_ok(dev, 2), lambda: moe_ada.ok(dev)):
        try:
            fn()
        except RuntimeError:
            pass
        else:
            raise AssertionError("an unprobed dispatch did not raise")
    return "RuntimeError for every unprobed (device, rate)"


def test_cpu_decode_never_needs_a_probe():
    """A CPU trellis decodes on the torch reference and never asks for a verdict."""
    t = torch.randint(-32768, 32768, (4, 2, 32), dtype=torch.int16)
    assert torch.equal(EA.decode(t), X.decode_trellis(t))
    return "cpu tensor -> decode_trellis, no probe, no verdict"


def _gpu() -> bool:
    return torch.cuda.is_available()


def test_probe_then_kernels_without_a_safety_net():
    """The real probe on this device, then the contract: True verdict -> the kernel's own
    error propagates out of `decode`; False -> the reference serves and the kernel is never
    called."""
    if not _gpu():
        return "skipped: no CUDA device"
    dev = torch.device("cuda", torch.cuda.current_device())
    dk, kb = str(dev), 5
    EA.probe(dev, kb)
    assert EA.rot_ok(dev) is True and EA.decode_ok(dev, kb) is True \
        and EA.gemv_ok(dev, kb) is True, "this build's Ada kernels must probe True"
    t = torch.randint(-32768, 32768, (4, 2, X.tile_width(kb)), dtype=torch.int16, device=dev)

    real = EA.decode_trellis_triton
    EA.decode_trellis_triton = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaboom"))
    try:
        try:
            EA.decode(t)
        except RuntimeError as e:
            assert "kaboom" in str(e)
        else:
            raise AssertionError("a True verdict swallowed a kernel error")
    finally:
        EA.decode_trellis_triton = real

    EA._DEC_OK[(dk, kb)] = False                     # a failed verdict: the reference serves
    try:
        EA.decode_trellis_triton = lambda *a, **k: (_ for _ in ()).throw(AssertionError("called"))
        assert torch.equal(EA.decode(t), X.decode_trellis(t))
    finally:
        EA.decode_trellis_triton = real
        EA._DEC_OK[(dk, kb)] = True
    return "True verdict propagates, False serves the reference, kernel untouched"


def test_moe_probe_and_error_propagation():
    """MoEBank's load probes the device; a True verdict then lets a kernel error out."""
    if not _gpu():
        return "skipped: no CUDA device"
    from tools import moe_ada
    from tools import fn_kernels as FK
    dev = torch.device("cuda", torch.cuda.current_device())
    moe_ada.probe(dev)
    assert moe_ada.ok(dev) is True, "this build's MoE kernels must probe True"
    assert torch.cuda.current_device() >= 0

    g = torch.Generator().manual_seed(7)
    def tre(kb):
        return torch.randint(-32768, 32768, (8, 8, X.tile_width(kb)), dtype=torch.int16,
                             generator=g).to(dev)
    def su(n):
        return torch.randn(n, dtype=torch.float16, generator=g).to(dev)
    experts = [((tre(25), su(128), su(128)), (tre(25), su(128), su(128)),
                (tre(25), su(128), su(128))) for _ in range(4)]
    bank = FK.MoEBank(experts, dev)
    x = torch.randn(4, 128, dtype=torch.bfloat16, generator=g).to(dev)
    ids = torch.randint(0, bank.n, (4, 2), generator=g).to(dev)

    real = moe_ada.moe_experts
    moe_ada.moe_experts = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaboom"))
    try:
        try:
            FK.moe_experts(x, ids, bank)
        except RuntimeError as e:
            assert "kaboom" in str(e)
        else:
            raise AssertionError("a True verdict swallowed a MoE kernel error")
    finally:
        moe_ada.moe_experts = real
    return "MoEBank probed at load; the True verdict propagates a kernel error"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<44} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<44} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
