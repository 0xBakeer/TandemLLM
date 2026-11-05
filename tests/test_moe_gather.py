"""The masked-load MoE split kernels (tools/moe_ada.py). GPU only.

Dispatch contract: the masked variants are the default wherever they are bitwise-identical
to the addressed-load kernels -- banks whose widest tile is <= 32 u32 words (kb 2,3,4,15,25,35)
on devices whose load probe latched `_MASKED_EQ`; wider banks (kb 5..8) keep the addressed
kernels and `QWEN38_MOE_GATHER=1` remains the opt-in that serves the masked form there, whose
fp32 partials re-associate at ~ulp scale (a numerics class, like FN_GEMV_SPLIT's order change).
Grouped and ungrouped variants decode identically, so a pair's bits still do not depend on
how many partners share its expert, and a row's bits still do not depend on M.

Run: python tests/test_moe_gather.py
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])

import torch

from tools import fn_kernels as FK
from tools import gpu as _gpu
from tools import moe_ada as MA
from tools import fn_prefill_kernels as PK
from tools.exl3_linear import tile_width

DEV = "cuda"


def _bank(kb_list, h=128, i=128, seed=0xADA5):
    g = torch.Generator().manual_seed(seed)
    experts = []
    for kb in kb_list:
        def tre():
            return torch.randint(-32768, 32768, (h // 16, i // 16, tile_width(kb)),
                                 dtype=torch.int16, generator=g).to(DEV)

        def su(n):
            return torch.randn(n, dtype=torch.float16, generator=g).to(DEV)
        experts.append(((tre(), su(128), su(128)),
                        (tre(), su(128), su(128)),
                        (tre(), su(128), su(128))))
    return FK.MoEBank(experts, DEV)


def _run(x, ids, bank, gather):
    orig = MA.MOE_GATHER
    MA.MOE_GATHER = gather
    try:
        return MA.moe_experts(x, ids, bank)
    finally:
        MA.MOE_GATHER = orig


def _run_forced(x, ids, bank, masked):
    orig = MA._FORCE_MASKED
    MA._FORCE_MASKED = masked
    try:
        return MA.moe_experts(x, ids, bank)
    finally:
        MA._FORCE_MASKED = orig


def test_flag_off_is_the_unchanged_path():
    # flag-off on a wmax<=32 bank serves the masked kernels -- whose bits equal the addressed
    # path's (asserted per rate below) -- so the path's bits are unchanged either way; against
    # the reference it keeps the probe's tolerance, and repeated runs are bit-identical.
    bank = _bank([2, 25] * 4)
    x = (torch.randn(8, 128, generator=torch.Generator().manual_seed(1)) * .5).to(DEV, torch.bfloat16)
    ids = torch.randint(0, bank.n, (8, 2), generator=torch.Generator().manual_seed(2)).to(DEV)
    assert torch.allclose(PK.moe_grouped_reference(x, ids, bank),
                          _run(x, ids, bank, False), atol=4e-2, rtol=4e-2)
    assert torch.equal(_run(x, ids, bank, False), _run(x, ids, bank, False))


def test_gather_vs_reference():
    # flag-on drift stays inside the probe's tolerance band at every tested rate mix,
    # including kb=8 (mask all-true) and kb=15 (W2v=12, the heaviest masking + wrap).
    for kbs in ([25] * 8, [2, 25] * 4, [8, 15] * 4):
        bank = _bank(kbs)
        g = torch.Generator().manual_seed(3)
        for m in (1, 4, 8, 16):
            x = (torch.randn(m, 128, generator=g) * .5).to(DEV, torch.bfloat16)
            ids = torch.randint(0, bank.n, (m, 2), generator=g).to(DEV)
            assert torch.allclose(PK.moe_grouped_reference(x, ids, bank),
                                  _run(x, ids, bank, True), atol=4e-2, rtol=4e-2), (kbs, m)


def test_masked_default_is_bitwise():
    # the wmax<=32 default's contract, run as the persistent regression net: forced addressed
    # vs forced masked through `moe_experts`'s own dispatch, per rate and per arm -- P<16 goes
    # ungrouped, 16<=P<=128 grouped.
    for kbs in ([2] * 4, [3] * 4, [4] * 4, [15] * 4, [25] * 4, [35] * 4, [2, 25] * 2):
        bank = _bank(kbs)
        g = torch.Generator().manual_seed(7)
        for m in (1, 4, 8, 16):
            x = (torch.randn(m, 128, generator=g) * .5).to(DEV, torch.bfloat16)
            ids = torch.randint(0, bank.n, (m, 2), generator=g).to(DEV)
            assert torch.equal(_run_forced(x, ids, bank, False),
                               _run_forced(x, ids, bank, True)), (kbs, m)


def test_wide_rates_stay_direct():
    # wmax>32 is the known re-association: flag-off dispatch keeps the addressed kernels, the
    # masked form is genuinely different there, and only the flag puts a wide bank on it.
    dev = torch.device(DEV)
    MA.probe(dev)
    bank = _bank([8, 15] * 4)
    g = torch.Generator().manual_seed(9)
    x = (torch.randn(8, 128, generator=g) * .5).to(DEV, torch.bfloat16)
    ids = torch.randint(0, bank.n, (8, 2), generator=g).to(DEV)
    assert not torch.equal(_run_forced(x, ids, bank, False),
                           _run_forced(x, ids, bank, True))
    wmax = MA._bank_wmax(bank)
    assert wmax > 32
    assert MA._use_masked(wmax, dev) is False
    orig = MA.MOE_GATHER
    try:
        MA.MOE_GATHER = True
        assert MA._use_masked(wmax, dev) is True
    finally:
        MA.MOE_GATHER = orig
    if MA._MASKED_EQ.get(MA._devkey(dev)):
        assert MA._use_masked(32, dev) is True      # the probed wmax<=32 default


def test_drift_is_ulp_scale():
    # the flag-on/flag-off delta on a wmax>32 bank is bounded and ~ulp class, not a
    # wrong-value bug. (A wmax<=32 bank has no delta at all -- the masked default is bitwise.)
    for kbs in ([8, 15] * 4,):
        bank = _bank(kbs)
        g = torch.Generator().manual_seed(4)
        x = (torch.randn(16, 128, generator=g) * .5).to(DEV, torch.bfloat16)
        ids = torch.randint(0, bank.n, (16, 2), generator=g).to(DEV)
        off, on = _run(x, ids, bank, False), _run(x, ids, bank, True)
        # rounding-boundary flips only: compare as bf16 codes -- a diff is at most one ulp,
        # and the flipped fraction is tiny (measured ~0.008% at deployed dims)
        bo = off.to(torch.bfloat16).view(torch.int16)
        bn = on.to(torch.bfloat16).view(torch.int16)
        ulps = (bo - bn).abs()
        assert ulps.max() <= 1, (kbs, ulps.max().item())
        assert (ulps > 0).float().mean() < 5e-3, (kbs, (ulps > 0).float().mean().item())


def test_row_invariance_under_gather():
    # a pair's bits do not depend on M under flag-on either: row-for-row bitwise vs batched.
    bank = _bank([2, 25] * 4)
    g = torch.Generator().manual_seed(5)
    x = (torch.randn(16, 128, generator=g) * .5).to(DEV, torch.bfloat16)
    ids = torch.randint(0, bank.n, (16, 2), generator=g).to(DEV)
    full = _run(x, ids, bank, True)
    for m in (0, 5, 15):
        alone = _run(x[m:m + 1], ids[m:m + 1], bank, True)
        assert torch.equal(full[m * 2:m * 2 + 2], alone), m


def test_parity_across_the_grouped_boundary():
    # the same pair alone (ungrouped dispatch) and inside a grouped call produce the same
    # bits under flag-on -- the property the self-check enforces at load.
    bank = _bank([2, 25] * 4)
    g = torch.Generator().manual_seed(6)
    x = (torch.randn(16, 128, generator=g) * .5).to(DEV, torch.bfloat16)
    ids = torch.randint(0, bank.n, (16, 2), generator=g).to(DEV)
    grp = _run(x, ids, bank, True)
    for m in (0, 9):
        assert torch.equal(grp[m * 2:m * 2 + 2], _run(x[m:m + 1], ids[m:m + 1], bank, True)), m


def test_split_kernels_parity():
    # kernel level: masked grouped == masked ungrouped for the same pair, bitwise.
    from tools import exl3_ada as EA
    bank = _bank([25] * 8)
    ki, kbias = EA._k_inv(), EA._k_bias()
    K = N = 128
    KT, NT = K // 16, N // 16
    NBS, KS = 1, 2
    dev = torch.device(DEV)
    pinv = MA._pinv(dev)
    xh = torch.randn(2, K, dtype=torch.float16, generator=torch.Generator().manual_seed(8)).to(dev)
    ids = torch.tensor([3], dtype=torch.int32, device=dev)
    yu = torch.empty(2 * KS * NT * 16, dtype=torch.float32, device=dev)
    MA._moe_trellis_split_m[(1, 2, NBS * KS)](
        xh, ids, bank.tr, bank.kb, pinv, yu, K, KT, NT, NBS, KS, J=2, TJ=-1,
        K_INV=ki, K_BIAS=kbias, W2P=32, num_stages=1)
    z = dict(dtype=torch.int32, device=dev)
    yg = torch.zeros(2 * KS * NT * 16, dtype=torch.float32, device=dev)
    MA._moe_trellis_split_g_m[(1, 2, NBS * KS)](
        xh, torch.tensor([0], **z), torch.tensor([3], **z), torch.tensor([0], **z),
        torch.tensor([1], **z), torch.tensor([1], **z), bank.tr, bank.kb, pinv, yg,
        K, KT, NT, NBS, KS, J=2, TJ=-1, K_INV=ki, K_BIAS=kbias, G=2, W2P=32,
        num_stages=1)
    torch.cuda.synchronize()
    assert torch.equal(yu, yg)


def _fk_run(x, ids, bank, group):
    orig = FK.MOE_GROUP
    FK.MOE_GROUP = group
    try:
        return FK.moe_experts(x, ids, bank)
    finally:
        FK.MOE_GROUP = orig


def test_moe_group_bitwise():
    # ENG-241: at M>=4 `moe_experts` walks same-expert pairs through <=8-row CTAs whose tile
    # decode amortizes across the group. Every pair's yd equals the ungrouped path's bitwise,
    # whatever the routing -- including the real pack's shared-expert column (last slot = E).
    if not _gpu.profile(torch.device(DEV)).inline("fn_kernels"):
        return "skipped: the group kernels are a CUDA path"
    for kbs in ([25] * 8, [2, 25] * 4, [4] * 4):
        bank = _bank(kbs)
        g = torch.Generator().manual_seed(11)
        for m in (4, 8, 16, 32):
            x = (torch.randn(m, 128, generator=g) * .5).to(DEV, torch.bfloat16)
            ids = torch.randint(0, bank.n, (m, 2), generator=g).to(DEV, torch.int32)
            ids[:, -1] = bank.n - 1          # the shared-expert slot, as the router appends it
            assert torch.equal(_fk_run(x, ids, bank, True),
                               _fk_run(x, ids, bank, False)), (kbs, m)


def test_moe_group_companion_permutation():
    # a row's bits cannot depend on which rows share its CTA: hold row 0 fixed, permute its
    # companions, its pairs' yd must not change.
    if not _gpu.profile(torch.device(DEV)).inline("fn_kernels"):
        return "skipped: the group kernels are a CUDA path"
    bank = _bank([25] * 8)
    g = torch.Generator().manual_seed(13)
    m, S = 16, 3
    x = (torch.randn(m, 128, generator=g) * .5).to(DEV, torch.bfloat16)
    ids = torch.randint(0, bank.n, (m, S), generator=g).to(DEV, torch.int32)
    base = _fk_run(x, ids, bank, True)
    for _ in range(8):
        p = torch.cat([torch.arange(1), 1 + torch.randperm(m - 1, generator=g)]).to(DEV)
        got = _fk_run(x[p], ids[p], bank, True)
        # pair p sits at flat index p*S+s in both calls
        want = base.reshape(m, S, -1)[p].reshape(m * S, -1)
        assert torch.equal(got, want)
        assert torch.equal(got[:S], base[:S])     # row 0 unmoved: identical bits


def test_moe_group_boundary_runs():
    # runs of one expert around the MOE_GR=8 split: exactly 8, 9 (8+1), 16 (8+8), 17 (8+8+1);
    # plus every pair on one expert (the all-same adversarial case). All bitwise vs ungrouped.
    if not _gpu.profile(torch.device(DEV)).inline("fn_kernels"):
        return "skipped: the group kernels are a CUDA path"
    bank = _bank([25] * 8)
    g = torch.Generator().manual_seed(17)
    for run in (8, 9, 16, 17):
        m = (run + 3) // 4
        x = (torch.randn(m, 128, generator=g) * .5).to(DEV, torch.bfloat16)
        ids = torch.randint(0, bank.n, (m, 4), generator=g).to(DEV, torch.int32)
        ids.reshape(-1)[:run] = 3                # pairs 0..run-1 all route to expert 3
        assert torch.equal(_fk_run(x, ids, bank, True),
                           _fk_run(x, ids, bank, False)), run
    x = (torch.randn(8, 128, generator=g) * .5).to(DEV, torch.bfloat16)
    ids = torch.full((8, 4), 5, dtype=torch.int32, device=DEV)   # 32 pairs, one expert
    assert torch.equal(_fk_run(x, ids, bank, True), _fk_run(x, ids, bank, False))


if __name__ == "__main__":
    assert torch.cuda.is_available(), "cuda required"
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"  {name:<44} ok")
        except Exception as e:
            fails += 1
            print(f"  {name:<44} FAIL {e!r}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
