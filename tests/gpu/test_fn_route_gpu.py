"""On the board (SPD-249, T8): the fused routing launch.

  * `route` is the router, the pad mask, `expert_order` and `fn_moe_groups` bit for bit: ids, weights, the order and the
    group table (walked or scanned, caps 8/6/4/1), 1..32 rows, pad rows, logits with exact ties, 512 and 256 experts
    (the kernel's two widths);
  * the adapter's `_moe_routed` (on a stand-in engine with random EXL3 experts at the real shapes) gives today's ids,
    weights and every real pair's expert output: per-pair kernels at 1..32 rows (the lone verify and the decode step),
    grouped kernels in every variant (FN_MOE_AS 0/1/2, FN_MOE_STAGE, caps, scanned groups) with and without pad rows;
  * the same calls captured in a CUDA graph on a stream of their own, replayed with new rows and a new pad mask, equal
    the eager calls of those rows.

Run on the box: python tests/gpu/test_fn_route_gpu.py   (under the GPU lock)
"""

from __future__ import annotations

import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine.models.flashnext import engine as EM  # noqa: E402
from tools import exl3_linear as X  # noqa: E402
from tools import fn_kernels as K  # noqa: E402

H, I = 2560, 640
DEV = "cuda"
NE, KT = 24, 10


def _logits(M, E, seed, ties=True):
    g = torch.Generator(device=DEV).manual_seed(seed)
    lg = torch.randn(M, E + 1, generator=g, device=DEV) * 2
    if ties:
        lg[:, 1::7] = lg[:, 0:1]                              # exact ties at bf16 and above: the lower index wins
        lg[::3, 5:40] = lg[::3, 4:5]
    return lg


def _table(G, NG):
    return [G[r, :2 + int(G[r, 1])].tolist() for r in range(int(NG))]


def test_route_is_router_mask_sort_groups():
    mod = K._module()
    n = 0
    try:
        for E in (512, 256):
            for M in (1, 2, 3, 5, 8, 13, 16, 24, 31, 32):
                for npad in (0, 1, 3):
                    if npad >= M:
                        continue
                    lg = _logits(M, E, 1000 * E + 10 * M + npad, ties=M % 2 == 0)
                    pad = torch.zeros(M, 1, dtype=torch.bool, device=DEV)
                    if npad:
                        pad[torch.randperm(M, generator=torch.Generator().manual_seed(M))[:npad].to(DEV)] = True
                    ids0, w0 = K.router(lg, E, KT)
                    ids0.masked_fill_(pad, -1)
                    order0 = K.expert_order(ids0)
                    ids1, w1, order1 = K.route(lg, E, KT, pad=pad)
                    assert torch.equal(ids0, ids1) and torch.equal(w0, w1), (E, M, npad)
                    assert torch.equal(order0, order1), (E, M, npad)
                    ref_o, ref_g = K.route_index_reference(ids0.cpu(), 6)
                    assert torch.equal(ref_o.to(DEV), order1)
                    for gr in (8, 6, 4, 1):
                        mod.fn_set_moe_gr(gr)
                        for gp in (0, 1):
                            mod.fn_set_moe_gpar(gp)
                            P = ids0.numel()
                            G0 = torch.full((P, K.MOE_G_COLS), -7, dtype=torch.int32, device=DEV)
                            NG0 = torch.zeros(1, dtype=torch.int32, device=DEV)
                            mod.fn_moe_groups(ids0.reshape(-1), order0, G0, NG0)
                            ids2, w2, G1, NG1 = K.route(lg, E, KT, pad=pad, groups=True)
                            assert torch.equal(ids2, ids0) and torch.equal(w2, w0)
                            assert _table(G1, NG1) == _table(G0, NG0), (E, M, gr, gp)
                            n += 1
    finally:
        mod.fn_set_moe_gr(8)
        mod.fn_set_moe_gpar(0)
    return f"{n} cases (512/256 experts, 1..32 rows, pad rows, ties, caps 8/6/4/1, walked/scanned)"


def _bank(seed=0):
    blocks = []
    for e in range(NE + 1):
        kb = 5 if e == NE else 3
        blocks.append(tuple(X.random_block(n_, k_, kb, seed=seed + 3 * e + j)
                            for j, (n_, k_) in enumerate(((I, H), (I, H), (H, I)))))
    trip = [tuple((b.t, b.suh[0].contiguous(), b.svh) for b in eb) for eb in blocks]
    return K.MoEBank(trip, DEV)


def _engine(bank):
    g = torch.Generator(device=DEV).manual_seed(77)
    cfg = types.SimpleNamespace(num_experts=NE, num_experts_per_tok=KT, num_hidden_layers=1)
    return types.SimpleNamespace(cfg=cfg, banks={0: bank}, pre=lambda li: "l0.",
                                 W={"l0.mlp.router": (torch.randn(NE + 1, H, generator=g, device=DEV) * 0.02).half()},
                                 _order1=torch.empty(0, dtype=torch.int32, device=DEV))


def _today(fake, x, pad):
    lg = K.gemv(x, fake.W["l0.mlp.router"])
    ids, wts = K.router(lg, NE, KT)
    if pad is not None:
        ids.masked_fill_(pad, -1)
    yd = K.moe_experts(x, ids, fake.banks[0], order=fake._order1 if x.shape[0] == 1 else None)
    return yd.clone(), ids.clone(), wts.clone()


def _routed(fake, x, pad):
    yd, ids, wts = EM.FlashNextEngine._moe_routed(fake, x, 0, pad)
    torch.cuda.synchronize()
    return yd.clone(), ids.clone(), wts.clone()


def _same(a, b):
    real = a[1].reshape(-1) >= 0
    return torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]) and torch.equal(a[0][real], b[0][real])


def _x(M, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(M, H, generator=g, device=DEV) * 0.5).to(torch.bfloat16)


def test_moe_routed_per_pair():
    bank = _bank()
    fake = _engine(bank)
    old = K.MOE_GROUP
    K.MOE_GROUP = False
    try:
        for M in (1, 2, 3, 4, 5, 6, 8, 16, 32):
            x = _x(M, 10 + M)
            want = _today(fake, x, None)
            assert _same(_routed(fake, x, None), want), M
    finally:
        K.MOE_GROUP = old
    return "rows 1..32: ids, weights, every pair's output = today's"


def test_moe_routed_grouped():
    bank = _bank(1)
    fake = _engine(bank)
    mod = K._module()
    old = (K.MOE_GROUP, K.MOE_AS, K.MOE_STAGE)
    n = 0
    try:
        K.MOE_GROUP = True
        for M in (2, 5, 16, 24, 32):
            for npad in (0, 3):
                x = _x(M, 40 + M)
                pad = torch.zeros(M, 1, dtype=torch.bool, device=DEV)
                if npad:
                    pad[M - npad:] = True
                for asv, stage in ((0, False), (0, True), (1, False), (2, False)):
                    K.MOE_AS, K.MOE_STAGE = asv, stage
                    for gr in (8, 6):
                        mod.fn_set_moe_gr(gr)
                        for gp in (0, 1):
                            mod.fn_set_moe_gpar(gp)
                            want = _today(fake, x, pad)
                            assert _same(_routed(fake, x, pad), want), (M, npad, asv, stage, gr, gp)
                            n += 1
    finally:
        K.MOE_GROUP, K.MOE_AS, K.MOE_STAGE = old
        mod.fn_set_moe_gr(8)
        mod.fn_set_moe_gpar(0)
    return f"{n} cases (rows 2..32, pad rows, AS 0/1/2, staged, caps 8/6, walked/scanned) = today's"


def test_routed_in_a_graph():
    from engine import capture_lock
    bank = _bank(2)
    fake = _engine(bank)
    mod = K._module()
    old = (K.MOE_GROUP, K.MOE_AS)
    out = []
    try:
        mod.fn_set_moe_gr(6)
        mod.fn_set_moe_gpar(1)
        K.MOE_AS = 1
        for group, M in ((True, 24), (False, 4), (False, 1)):
            K.MOE_GROUP = group
            xb = _x(M, 60 + M).clone()
            pb = torch.zeros(M, 1, dtype=torch.bool, device=DEV) if group else None
            EM.FlashNextEngine._moe_routed(fake, xb, 0, pb)            # warm
            torch.cuda.synchronize()
            s = capture_lock.lease_stream()
            s.wait_stream(torch.cuda.current_stream())
            g = torch.cuda.CUDAGraph()
            with capture_lock.LOCK, torch.cuda.graph(g, pool=torch.cuda.graph_pool_handle(), stream=s):
                yd, ids, wts = EM.FlashNextEngine._moe_routed(fake, xb, 0, pb)
            torch.cuda.current_stream().wait_stream(s)
            for rep in range(3):
                xb.copy_(_x(M, 70 + rep))
                if pb is not None:
                    pb.zero_()
                    pb[M - rep:] = True if rep else False
                g.replay()
                torch.cuda.synchronize()
                want = _today(fake, xb.clone(), None if pb is None else pb.clone())
                assert _same((yd, ids, wts), want), (group, M, rep)
            out.append(f"{'grouped' if group else 'per-pair'} {M}")
            del g
    finally:
        K.MOE_GROUP, K.MOE_AS = old
        mod.fn_set_moe_gr(8)
        mod.fn_set_moe_gpar(0)
    return "captured (" + ", ".join(out) + "), replayed with new rows and pad masks = eager today's"


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
