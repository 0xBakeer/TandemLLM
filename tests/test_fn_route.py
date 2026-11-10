"""SPD-249 (T8): the fused routing launch's index, the host side (CPU).

  1  route_kernel's index step (an expert's rows as a bit mask, one scan over the experts, a pair's place from
     popcounts), modelled by `fn_kernels.route_index_reference`, is `expert_order`'s stable sort and
     moe_groups_kernel's walk over it (`groups_reference`): random routings of 1..32 rows, pad rows, caps 1..8
                                                                     test_index_is_the_sort_and_the_walk
  2  the knob is registered, off by default, and the adapter reads it
                                                                     test_knobs_off_by_default
  3  `_moe_routed` asks for one route launch (the order, or the groups on the grouped kernels; one row keeps the router
     launch) and the experts from that index; `moe_parts` takes it only with the knob on, for 1..32 rows, where the
     CUDA kernels run (a fake kernel module records the calls)       test_moe_routed_dispatch
  4  the load-time self-check with FN_ROUTE_FUSED runs every call through `moe_parts` too (per-pair, and grouped with
     pad rows) and a difference there leaves it failing; knob off: today's arm only (fake engine and kernels)
                                                                     test_self_check_routed_arms
  5  the knob acts for Flash-Next only: Qwen3.5-MoE (a subclass of the engine) keeps the router, mask and experts
     launches in `moe_parts` and today's arm in the self-check with the knob on, the same launches and values as with
     it off; Flash-Next itself takes the route launch                test_qwen35_moe_ignores_route_knobs

GPU twins: tests/gpu/test_fn_route_gpu.py. Run: python tests/test_fn_route.py
"""

from __future__ import annotations

import os
import random
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import fn_kernels as K  # noqa: E402

E, KTOP = 512, 10
S = KTOP + 1


def _ids(rng: random.Random, M: int, pad_p: float = 0.2) -> torch.Tensor:
    rows = []
    for _ in range(M):
        if rng.random() < pad_p:
            rows.append([-1] * S)
        else:
            pool = rng.sample(range(E), KTOP) if rng.random() < 0.7 else rng.sample(range(24), KTOP)
            rows.append(pool + [E])
    return torch.tensor(rows, dtype=torch.int32)


def test_index_is_the_sort_and_the_walk():
    rng = random.Random(3)
    n = 0
    for M in list(range(1, 33)) * 3:
        ids = _ids(rng, M)
        want_order = torch.sort(ids.reshape(-1), stable=True).indices.to(torch.int32)
        for gr in (1, 3, 6, 8):
            order, groups = K.route_index_reference(ids, gr)
            assert torch.equal(order, want_order), (M, gr)
            assert groups == K.groups_reference(ids, want_order, gr), (M, gr)
            n += 1
    return f"{n} routings: order = stable sort, groups = the walk"


def test_knobs_off_by_default():
    from engine.models.flashnext.knobs import KNOBS
    from engine.settings import KNOBS as ALL
    assert KNOBS["FN_ROUTE_FUSED"] == ("0", "engine/models/flashnext/engine.py") and "FN_ROUTE_FUSED" in ALL
    for k in ("FN_SHARED_SIDE", "FN_SHARED_MODE"):
        assert k not in KNOBS, f"{k}: M4 is not on this branch"
    src = open(os.path.join(ROOT, "engine/models/flashnext/engine.py")).read()
    assert 'ROUTE_FUSED = _S.get("FN_ROUTE_FUSED") == "1"' in src
    if "QWEN38_FN_ROUTE_FUSED" not in os.environ:
        from engine.models.flashnext import engine as EM
        assert EM.ROUTE_FUSED is False
    return "FN_ROUTE_FUSED registered, '0', read in engine.py; no M4 knob"


class _Rec:
    """A stand-in for tools/fn_kernels: records the launches `moe_parts` / `_moe_routed` ask for, returns tensors of the
    shapes."""

    def __init__(self, group=False, ok=True):
        self.calls = []
        self.MOE_GROUP, self.ROUTE_ROWS, self.ok = group, 32, ok

    def route_ok(self, dev):
        return self.ok

    def gemv(self, x, w):
        self.calls.append(("gemv",))
        return torch.zeros(x.shape[0], E + 1)

    def route(self, logits, E_, k, pad=None, groups=False):
        self.calls.append(("route", groups, None if pad is None else int(pad.sum())))
        M = logits.shape[0]
        ids = torch.zeros(M, k + 1, dtype=torch.int32)
        if groups:
            return ids, ids.float(), "g", "ng"
        return ids, ids.float(), torch.arange(M * (k + 1), dtype=torch.int32)

    def router(self, logits, E_, k):
        self.calls.append(("router",))
        return torch.zeros(logits.shape[0], k + 1, dtype=torch.int32), torch.zeros(logits.shape[0], k + 1)

    def moe_experts(self, x, ids, bank, order=None):
        self.calls.append(("moe_experts",))
        return torch.zeros(1)

    def moe_experts_ix(self, x, ids, bank, ix):
        self.calls.append(("experts", ix[0], len(ix[1]) if ix[0] == "order" else None))
        return torch.zeros(1)


def _fake_engine(cls=None):
    from engine.models.flashnext import engine as EM
    cls = cls or EM.FlashNextEngine
    e = object.__new__(cls)
    e.cfg = types.SimpleNamespace(num_experts=E, num_experts_per_tok=KTOP)
    e.banks, e.W, e.pre = {0: "bank"}, {"l0.mlp.router": None}, (lambda li: "l0.")
    e.route_trace, e._moe_record = None, None
    e._order1 = torch.empty(0, dtype=torch.int32)
    e.device = "cpu"
    return e


def _with_kernels(rec):
    import tools
    real = sys.modules["tools.fn_kernels"]
    sys.modules["tools.fn_kernels"] = rec
    tools.fn_kernels = rec
    return real


def _restore_kernels(real):
    import tools
    sys.modules["tools.fn_kernels"] = real
    tools.fn_kernels = real


def test_moe_routed_dispatch():
    from engine.models.flashnext import engine as EM
    FN = EM.FlashNextEngine
    rec = _Rec()
    real = _with_kernels(rec)
    old = EM.ROUTE_FUSED
    try:
        fake = _fake_engine()
        # per-pair, 4 rows: one route launch for the full order
        FN._moe_routed(fake, torch.zeros(4, 8), 0, None)
        assert rec.calls == [("gemv",), ("route", False, None), ("experts", "order", 4 * S)], rec.calls
        # one row: today's router launch and the empty order
        rec.calls.clear()
        FN._moe_routed(fake, torch.zeros(1, 8), 0, None)
        assert rec.calls == [("gemv",), ("router",), ("experts", "order", 0)], rec.calls
        # grouped with pad rows: the groups from the route launch, the pad mask handed in
        rec.calls.clear()
        rec.MOE_GROUP = True
        pad = (torch.arange(24) % 3 == 2).view(-1, 1)
        FN._moe_routed(fake, torch.zeros(24, 8), 0, pad)
        assert rec.calls == [("gemv",), ("route", True, 8), ("experts", "groups", None)], rec.calls
        # moe_parts: knob off -> today's launches; on -> the route launch for 1..32 rows only, where the kernels run
        rec.MOE_GROUP = False
        for fused, M, ok, want in ((False, 4, True, "moe_experts"), (True, 4, True, "experts"),
                                   (True, 32, True, "experts"), (True, 33, True, "moe_experts"),
                                   (True, 4, False, "moe_experts")):
            EM.ROUTE_FUSED, rec.ok = fused, ok
            rec.calls.clear()
            FN.moe_parts(fake, torch.zeros(M, 8), 0)
            assert rec.calls[-1][0] == want, (fused, M, ok, rec.calls)
    finally:
        EM.ROUTE_FUSED = old
        _restore_kernels(real)
    return ("one route launch (order or groups), one row the router; moe_parts takes it with the knob on for 1..32 "
            "rows where the CUDA kernels run")


def _twins(fault=None, rows=32, H=3):
    """A fake engine + kernel module for `_check_twins`: x row t carries t, the router, the experts and `moe_parts` look
    up the decode step's records by it; `fault` breaks one arm."""
    g = torch.Generator().manual_seed(9)
    I_all = torch.randint(0, E, (rows, S), generator=g, dtype=torch.int32)
    W_all = torch.rand(rows, S, generator=g)
    Y_all = torch.randn(rows, S, H, generator=g)
    by_layer = {li: [(torch.full((1, 4), float(t)), I_all[t:t + 1].clone(), W_all[t:t + 1].clone(), Y_all[t].clone())
                     for t in range(rows)] for li in (0, 1)}
    seen: list = []

    class TK:
        ROUTE_ROWS, MOE_GROUP = 32, False

        def route_ok(self, dev):
            return True

        def gemv(self, x, w):
            return x

        def router(self, lg, E_, k):
            t = lg[:, 0].long()
            return I_all[t].clone(), W_all[t].clone()

        def moe_experts(self, x, ids, bank):
            return Y_all[x[:, 0].long()].reshape(-1, H).clone()
    tk = TK()
    cfg = types.SimpleNamespace(num_experts=E, num_experts_per_tok=KTOP)
    from engine.models.flashnext import engine as EM
    fake = types.SimpleNamespace(cfg=cfg, banks={0: "b0", 1: "b1"}, W={"l0.mlp.router": None}, pre=lambda li: "l0.",
                                 CHAIN_WIDTHS=EM.FlashNextEngine.CHAIN_WIDTHS, TREE_ROWS=EM.FlashNextEngine.TREE_ROWS,
                                 route_knobs=EM.FlashNextEngine.route_knobs, device="cpu")
    fake.route_trace = trace = []

    def moe_parts(x, li, pad=None):
        assert fake.route_trace is None, "the check's calls stay out of the route trace"
        seen.append((li, tk.MOE_GROUP, x.shape[0], None if pad is None else pad.view(-1).tolist()))
        t = x[:, 0].long()
        ids, wts, yd = I_all[t].clone(), W_all[t].clone(), Y_all[t].clone()
        if pad is not None and fault != "pad":
            ids.masked_fill_(pad, -1)
        if fault == "routed" and not tk.MOE_GROUP and li == 1 and x.shape[0] == 12:
            yd[5, 3, 1] += 1                                    # one pair's output, row 5 of the 12-row call
        if fault == "grouped" and tk.MOE_GROUP and x.shape[0] == 8 and pad is not None:
            wts[1, 0] += 1
        return yd.reshape(-1, H), ids, wts
    fake.moe_parts = moe_parts
    return fake, tk, by_layer, seen, trace


def test_self_check_routed_arms():
    from engine.models.flashnext import engine as EM
    real = sys.modules["tools.fn_kernels"]
    old = EM.ROUTE_FUSED
    import tools
    res = []
    try:
        for fused, fault, want in ((False, None, None), (False, "routed", None), (True, None, None),
                                   (True, "routed", {"layer": 1, "row": 5, "call": "routed:chain12"}),
                                   (True, "grouped", {"layer": 0, "row": 1, "call": "routed_grouped:chain8"}),
                                   (True, "pad", {"layer": 0, "row": 2, "call": "routed_grouped:chain4"})):
            fake, tk, by_layer, seen, trace = _twins(fault)
            sys.modules["tools.fn_kernels"] = tk
            tools.fn_kernels = tk
            EM.ROUTE_FUSED = fused
            bad, arms = EM.FlashNextEngine._check_twins(fake, by_layer)
            assert bad == want, (fused, fault, bad)
            assert fake.route_trace is trace and trace == [], "route trace restored, untouched"
            assert tk.MOE_GROUP is False, "MOE_GROUP restored"
            if not fused:
                assert arms == {"experts": 164} and not seen, (arms, len(seen))
                continue
            # 2 layers x 82 rows (chain 2, 4, 8, 12, 16, 32 + tree 8); grouped: every third row a pad row (24 a layer)
            assert arms == {"experts": 164, "routed": 164, "routed_grouped": 116}, arms
            assert len(seen) == 2 * 7 * 2, len(seen)
            assert [s[1:3] for s in seen[:4]] == [(False, 2), (True, 2), (False, 4), (True, 4)], seen[:4]
            assert seen[3][3] == [False, False, True, False] and seen[2][3] is None, seen[2:4]
            res.append(fault or "clean")
    finally:
        EM.ROUTE_FUSED = old
        sys.modules["tools.fn_kernels"] = real
        tools.fn_kernels = real
    return ("knobs off: today's arm only (164 rows, no moe_parts call); on: + routed 164 + routed_grouped 116 rows; "
            "a fault in the per-pair, the grouped arm or a pad row's ids fails the check (" + ", ".join(res) + ")")


class _RecToday(_Rec):
    """`_Rec` plus today's experts launch; the router's ids carry the row's place, so a pad row's -1 shows."""

    def router(self, logits, E_, k):
        self.calls.append(("router",))
        M = logits.shape[0]
        ids = torch.arange(M, dtype=torch.int32).view(-1, 1).repeat(1, k + 1)
        return ids, ids.float() / 8

    def moe_experts(self, x, ids, bank, order=None):
        self.calls.append(("moe_experts", bank, None if order is None else int(order.numel())))
        return x.repeat_interleave(ids.shape[1], 0) + 1


def test_qwen35_moe_ignores_route_knobs():
    from engine.models.flashnext import engine as EM
    from engine.models.qwen3_5_moe.engine import Qwen35MoeEngine
    FN = EM.FlashNextEngine
    assert FN.route_knobs is True and Qwen35MoeEngine.route_knobs is False
    # the gate lives in the base class's methods: the subclass must not bring its own MoE dispatch or check
    for name in ("moe_parts", "_moe_routed", "_check_twins", "self_check"):
        assert name not in Qwen35MoeEngine.__dict__, f"Qwen35MoeEngine overrides {name}: re-check the knob gate"
    old = EM.ROUTE_FUSED
    shapes = ((1, None), (4, None), (24, None), (24, "pad"), (32, "pad"))

    def run(cls, group):
        rec = _RecToday(group=group)
        prev = _with_kernels(rec)
        try:
            out = []
            for M, p in shapes:
                x = torch.arange(M * 8, dtype=torch.float32).view(M, 8)
                pad = (torch.arange(M) % 3 == 2).view(-1, 1) if p else None
                rec.calls.clear()
                yd, ids, wts = cls.moe_parts(_fake_engine(cls), x, 0, pad=pad)
                out.append((M, p, list(rec.calls), yd.clone(), ids.clone(), wts.clone()))
            return out
        finally:
            _restore_kernels(prev)

    n = 0
    try:
        for group in (False, True):
            EM.ROUTE_FUSED = False
            off = run(Qwen35MoeEngine, group)
            for M, p, calls, yd, ids, wts in off:
                assert [c[0] for c in calls] == ["gemv", "router", "moe_experts"], calls
            EM.ROUTE_FUSED = True
            on = run(Qwen35MoeEngine, group)
            for a, b in zip(off, on):
                assert a[:3] == b[:3], (group, a[:3], b[:3])
                assert all(torch.equal(u, v) for u, v in zip(a[3:], b[3:])), (group, a[:2])
                n += 1
            # a pad row's ids are -1 on this path too (the mask after the router)
            M, p, calls, yd, ids, wts = on[3]
            assert bool((ids[2] == -1).all()) and bool((ids[0] >= 0).all()), ids
            # the family the knob is for takes the route launch (calls of 2..32 rows)
            fn = run(FN, group)
            assert any(c[0] == "route" for c in fn[1][2]), fn[1][2]
        # the self-check: today's arm only for the subclass with FN_ROUTE_FUSED on
        EM.ROUTE_FUSED = True
        fake, tk, by_layer, seen, trace = _twins()
        fake.route_knobs = Qwen35MoeEngine.route_knobs
        real = _with_kernels(tk)
        try:
            bad, arms = Qwen35MoeEngine._check_twins(fake, by_layer)
            assert bad is None and arms == {"experts": 164} and not seen, (bad, arms, len(seen))
        finally:
            _restore_kernels(real)
        fake, tk, by_layer, seen, trace = _twins()
        real = _with_kernels(tk)
        try:
            bad, arms = FN._check_twins(fake, by_layer)
            assert arms == {"experts": 164, "routed": 164, "routed_grouped": 116} and seen, arms
        finally:
            _restore_kernels(real)
    finally:
        EM.ROUTE_FUSED = old
    return (f"Qwen3.5-MoE with FN_ROUTE_FUSED on (per-pair and grouped): {n} calls of 1..32 rows = the knob-off launches "
            "(gemv, router, mask, moe_experts) and values; self-check arm 'experts' only; Flash-Next takes the route "
            "launch")


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
