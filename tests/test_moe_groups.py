"""The fused pair->group builder (tools/moe_ada.py::_moe_groups_fused). GPU only.

Exact-output contract: ORD/ES/LEAD/LEN/GC are integer index tables, so equality is
torch.equal -- no drift class. The oracle is the in-file reference path the docstring
cites: torch.argsort(stable=True) + _moe_groups_build. Sweep covers every grouped P
(16..128, the dispatch domain; the kernel itself serves P <= 128), and the edge
patterns named in review: all-same expert, all-distinct (P=128 x n=256 -> GC=128, the
no-sentinel-padding boundary), runs of exactly G, runs of G+-1, expert-0 and expert
n-1 clustering, plus random cells at several expert-domain sizes. Replay equality is
checked too since the launch is CUDA-graph captured inside moe_experts.

Run: python tests/test_moe_groups.py
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])

import torch

from tools import moe_ada as MA

DEV = "cuda"
G = 2


def oracle(flat, P):
    """argsort(stable)+index_select+_moe_groups_build, the reference the kernel replaces."""
    order = torch.argsort(flat, stable=True).to(torch.int32)
    es = flat[order.long()].to(torch.int32).contiguous()
    lead = torch.zeros(P, dtype=torch.int32, device=DEV)
    glen = torch.zeros(P, dtype=torch.int32, device=DEV)
    gc = torch.zeros(1, dtype=torch.int32, device=DEV)
    MA._moe_groups_build[(1,)](es, lead, glen, gc, P, G=G, num_warps=1)
    return order, es, lead, glen, gc


def fused(flat, P):
    order = torch.empty(P, dtype=torch.int32, device=DEV)
    es = torch.empty(P, dtype=torch.int32, device=DEV)
    lead = torch.empty(P, dtype=torch.int32, device=DEV)
    glen = torch.empty(P, dtype=torch.int32, device=DEV)
    gc = torch.empty(1, dtype=torch.int32, device=DEV)
    MA._moe_groups_fused[(1,)](flat, order, es, lead, glen, gc, P, G=G, num_warps=1)
    return order, es, lead, glen, gc


def check(flat, P, tag):
    o_ref, e_ref, l_ref, n_ref, g_ref = oracle(flat, P)
    o, e, l, n, g = fused(flat, P)
    assert torch.equal(o, o_ref), f"{tag}: ORD differs"
    assert torch.equal(e, e_ref), f"{tag}: ES differs"
    gc_v = g_ref.item()
    assert g.item() == gc_v, f"{tag}: GC {g.item()} != {gc_v}"
    assert torch.equal(l[:gc_v], l_ref[:gc_v]), f"{tag}: LEAD differs"
    assert torch.equal(n[:gc_v], n_ref[:gc_v]), f"{tag}: LEN differs"


def patterns(P, n, gen):
    """Named edge patterns over an n-expert id domain."""
    out = {}
    out["all_same"] = torch.full((P,), n // 2, dtype=torch.int32, device=DEV)
    # all-distinct needs n >= P; below that the maximally-distinct prefix plus fill
    if n >= P:
        out["all_distinct"] = torch.randperm(n, generator=gen,
                                             device=DEV)[:P].to(torch.int32)
    else:
        tail = torch.randint(0, n, (P - n,), generator=gen, device=DEV,
                             dtype=torch.int32)
        out["all_distinct"] = torch.cat(
            [torch.randperm(n, generator=gen, device=DEV).to(torch.int32), tail])
    # runs of exactly G then a new expert
    out["runs_G"] = (torch.arange(P, device=DEV) // G % n).to(torch.int32)
    # runs of G+1 / G-1 alternating
    r = torch.empty(P, dtype=torch.int32, device=DEV)
    lens = torch.tensor([G + 1 if i % 2 == 0 else G - 1 for i in range(P)], device=DEV)
    idx = torch.repeat_interleave(torch.arange(P, device=DEV) % n, lens)[:P]
    out["runs_Gpm1"] = idx.to(torch.int32)
    out["expert0"] = torch.zeros(P, dtype=torch.int32, device=DEV)
    out["expert0"][: max(1, P // 4)] = 0
    out["expert0"][P // 4:] = torch.randint(0, n, (P - P // 4,), generator=gen,
                                            device=DEV, dtype=torch.int32)
    out["expertNm1"] = torch.randint(0, n, (P,), generator=gen, device=DEV,
                                     dtype=torch.int32)
    out["expertNm1"][: max(1, P // 4)] = n - 1
    out["random"] = torch.randint(0, n, (P,), generator=gen, device=DEV,
                                  dtype=torch.int32)
    return out


def test_sweep_against_oracle():
    gen = torch.Generator(device=DEV).manual_seed(0)
    cells = 0
    for P in range(16, 129):
        for n in (9, 64, 256):
            for name, flat in patterns(P, n, gen).items():
                check(flat, P, f"P={P} n={n} {name}")
                cells += 1
    # the GC=128 boundary needs P=128 all-distinct on n>=128 -- covered above at
    # n=256; make the failing configuration explicit so a revert reports it by name
    check(torch.randperm(256, generator=gen, device=DEV)[:128].to(torch.int32),
          128, "P=128 n=256 all_distinct (GC=128)")
    print(f"  sweep: {cells} cells + GC=128 boundary, all torch.equal")


def test_small_p_below_grouped_floor():
    # kernel serves P < 16 too (probe path uses P=8/16)
    gen = torch.Generator(device=DEV).manual_seed(1)
    for P in (1, 4, 8, 15):
        flat = torch.randint(0, 64, (P,), generator=gen, device=DEV, dtype=torch.int32)
        check(flat, P, f"P={P}")


def test_replay_equality():
    P, n = 96, 64
    gen = torch.Generator(device=DEV).manual_seed(2)
    flat = torch.randint(0, n, (P,), generator=gen, device=DEV, dtype=torch.int32)
    ref = fused(flat, P)
    # capture once, replay twice: integer tables must come out identical
    order = torch.empty(P, dtype=torch.int32, device=DEV)
    es = torch.empty(P, dtype=torch.int32, device=DEV)
    lead = torch.empty(P, dtype=torch.int32, device=DEV)
    glen = torch.empty(P, dtype=torch.int32, device=DEV)
    gc = torch.empty(1, dtype=torch.int32, device=DEV)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        MA._moe_groups_fused[(1,)](flat, order, es, lead, glen, gc, P, G=G, num_warps=1)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        MA._moe_groups_fused[(1,)](flat, order, es, lead, glen, gc, P, G=G, num_warps=1)
    outs = []
    for _ in range(2):
        g.replay()
        outs.append((order.clone(), es.clone(), lead.clone(), glen.clone(), gc.clone()))
    for t in outs:
        for a, b in zip(t, ref):
            assert torch.equal(a, b), "graph-replayed tables differ from direct launch"
    assert all(torch.equal(outs[0][i], outs[1][i]) for i in range(5))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  {t.__name__:<44} ok")
    print(f"{len(tests)} passed")
