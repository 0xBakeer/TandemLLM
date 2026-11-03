"""On the board (SPD-249): the kernel parts a joint verify graph adds.

  * `fn_seq_kernels.ple_conv_multi` over 3 lanes' rows (each its own state, window rows and row count) inside a
    padded rows buffer equals `ple_conv` called a lane at a time, bit for bit, and writes nothing past a lane's rows;
  * the grouped expert kernels with pad rows (every slot's id -1, as a joint graph's router output is masked) give the
    real rows the unpadded call's bits, and the groups hold the real pairs only (no expert a pad row would pull); the
    combine over the padded rows gives the real rows' bits (a pad row's tied ids rank by slot);
  * `attn_prep_multi` over a rows buffer padded past the last lane: the real rows' queries and cache writes equal the
    unpadded launch's, the pad rows write no cache row;
  * a graph of `ple_conv_multi` + the grouped experts captured once and replayed after the host entries were rewritten
    for other lanes and row counts equals the eager calls.

Run on the box: python tests/gpu/test_fn_joint_graph_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools import exl3_linear as X  # noqa: E402
from tools import fn_kernels as K  # noqa: E402
from tools import fn_seq_kernels as SK  # noqa: E402

DEV = "cuda"
H, I = 2560, 640


def _g(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def _ple_lanes(g, MS, C, R):
    """3 lanes' PLE conv inputs inside one R-row buffer: (gvn, gv, y) [R, C], each lane's state, window rows."""
    gvn = torch.randn(R, C, generator=g, device=DEV).to(torch.bfloat16)
    gv = torch.randn(R, C, generator=g, device=DEV).to(torch.bfloat16)
    wc = (torch.randn(C, 4, generator=g, device=DEV) * 0.3).to(torch.bfloat16)
    lanes, s = [], 0
    for j, M in enumerate(MS):
        par = list(range(-1, M - 1)) if j != 1 else [-1, 0, 0, 2][:M]
        src = torch.tensor(SK.window_src(par, (9, 6, 3), 9), dtype=torch.int32, device=DEV)
        ps = torch.randn(1, C, 9, generator=g, device=DEV).to(torch.bfloat16)
        lanes.append({"r": slice(s, s + M), "src": src, "ps": ps})
        s += M
    return gvn, gv, wc, lanes


def _ple_fill(ent_h, gvn, gv, y, wc, lanes):
    for j, ln in enumerate(lanes):
        r = ln["r"]
        ent_h[j] = torch.tensor(SK.ple_conv_entry(gvn[r], gv[r], ln["ps"], wc, ln["src"], y[r]), dtype=torch.int64)


def test_ple_conv_multi_equals_a_lane_at_a_time():
    C, R, MS = 4 * 2560, 16, [3, 4, 2]
    gvn, gv, wc, lanes = _ple_lanes(_g(11), MS, C, R)
    y = torch.full((R, C), 7.0, dtype=torch.bfloat16, device=DEV)
    ent_h = torch.zeros(len(MS), 16, dtype=torch.int64).pin_memory()
    _ple_fill(ent_h, gvn, gv, y, wc, lanes)
    SK.ple_conv_multi(ent_h.to(DEV), C, R)
    for ln in lanes:
        r = ln["r"]
        want = SK.ple_conv(gvn[r], gv[r], ln["ps"], wc, ln["src"])
        assert torch.equal(y[r], want), r
    n = sum(MS)
    assert bool((y[n:] == 7.0).all()), "a pad row was written"
    return f"lanes of {MS} rows in a {R}-row buffer: = ple_conv a lane at a time; pad rows untouched"


def _bank(n_exp: int, seed: int = 0):
    trip = []
    for e in range(n_exp + 1):
        kb = 5 if e == n_exp else 3
        eb = [X.random_block(I, H, kb, seed=seed + 3 * e), X.random_block(I, H, kb, seed=seed + 3 * e + 1),
              X.random_block(H, I, kb, seed=seed + 3 * e + 2)]
        trip.append(tuple((b.t, b.suh[0].contiguous(), b.svh) for b in eb))
    return K.MoEBank(trip, DEV)


def _ids(M, n_exp, k, seed):
    g = torch.Generator().manual_seed(seed)
    rows = [torch.randperm(n_exp, generator=g)[:k].tolist() for _ in range(M)]
    return torch.tensor([r + [n_exp] for r in rows], dtype=torch.int32, device=DEV)


def test_grouped_experts_skip_pad_rows():
    bank = _bank(24, 5)
    out = []
    old = K.MOE_GROUP
    try:
        K.MOE_GROUP = True
        for n, R in ((13, 16), (23, 24), (24, 24), (30, 32)):
            x = (torch.randn(R, H, generator=_g(40 + n), device=DEV) * 0.5).to(torch.bfloat16)
            ids = _ids(R, 24, 10, 40 + n)
            want = K.moe_experts(x[:n].contiguous(), ids[:n].contiguous(), bank)
            ids_p = ids.clone()
            ids_p[n:] = -1
            got = K.moe_experts(x, ids_p, bank)
            S = ids.shape[1]
            assert torch.equal(got[:n * S], want), (n, R)
            # the groups: the real pairs only, every expert's pairs those of the unpadded call
            mod = K._module()
            flat_p, flat = ids_p.reshape(-1), ids[:n].reshape(-1)
            gp = torch.zeros(R * S, 2 + 8, dtype=torch.int32, device=DEV)
            g0 = torch.zeros(n * S, 2 + 8, dtype=torch.int32, device=DEV)
            ngp = torch.zeros(1, dtype=torch.int32, device=DEV)
            ng0 = torch.zeros(1, dtype=torch.int32, device=DEV)
            mod.fn_moe_groups(flat_p, K.expert_order(ids_p), gp, ngp, bank.n)
            mod.fn_moe_groups(flat, K.expert_order(ids[:n]), g0, ng0, bank.n)
            assert int(ngp) == int(ng0), (n, int(ngp), int(ng0))
            assert torch.equal(gp[:int(ngp)], g0[:int(ng0)]), n
            # the combine over the padded rows: the real rows' bits, the pad rows' slots all ranked (no stray read)
            w = torch.rand(R, S, generator=torch.Generator().manual_seed(n)).to(DEV)
            cp = K.moe_combine(got, ids_p, w)
            torch.cuda.synchronize()
            assert torch.equal(cp[:n], K.moe_combine(want, ids[:n].contiguous(), w[:n].contiguous())), n
            u_p = int(torch.unique(ids_p[:, :-1][ids_p[:, :-1] >= 0]).numel())
            u_0 = int(torch.unique(ids[:n, :-1]).numel())
            assert u_p == u_0, (u_p, u_0)
            out.append(f"{n}/{R}")
    finally:
        K.MOE_GROUP = old
    return f"real/padded rows {', '.join(out)}: real rows = unpadded bits (experts, combine), groups and union = unpadded"


def test_attn_prep_multi_pad_rows_write_nothing():
    HQ, HK, D, ROT, CAP = 16, 2, 256, 64, 2048
    g = _g(9)
    MS, L0, R = [3, 2], [300, 700], 8
    n = sum(MS)
    qkv = torch.randn(R, HQ * 2 * D + 2 * HK * D, generator=g, device=DEV).bfloat16()
    qw = torch.randn(D, generator=g, device=DEV).bfloat16() * 0.1
    kw = torch.randn(D, generator=g, device=DEV).bfloat16() * 0.1
    ang = torch.arange(CAP, device=DEV).float()[:, None] * torch.linspace(0.001, 0.5, ROT, device=DEV)[None, :]
    cos, sin = ang.cos().bfloat16(), ang.sin().bfloat16()

    def run(rows_total):
        caches, ent, s = [], [], 0
        pos_all = torch.zeros(R, dtype=torch.int64, device=DEV)            # the pad rows: position and slot 0
        for M, L in zip(MS, L0):
            kc = torch.zeros(HK, CAP, D, dtype=torch.bfloat16, device=DEV)
            vc = torch.zeros(HK, CAP, D, dtype=torch.bfloat16, device=DEV)
            pos_all[s:s + M] = torch.arange(L, L + M)
            caches.append((kc, vc))
            ent.append(K.attn_prep_entry(s, M, kc, vc, None, pos_all[s:s + M], pos_all[s:s + M]))
            s += M
        e = torch.tensor(ent, dtype=torch.int64, device=DEV)
        q = K.attn_prep_multi(qkv[:rows_total], None, e, qw, kw, cos, sin, HQ, HK, 0, 0, 0, 1e-6)
        return q, caches
    q0, c0 = run(n)
    q1, c1 = run(R)
    assert torch.equal(q1[:n], q0), "real rows"
    for (k0, v0), (k1, v1) in zip(c0, c1):
        assert torch.equal(k0, k1) and torch.equal(v0, v1), "a pad row wrote a cache row"
    return f"lanes of {MS} rows in an {R}-row buffer: real queries and cache rows = unpadded; slot 0 never written"


def test_graph_replay_after_new_entries():
    C, R = 4 * 2560, 24
    bank = _bank(24, 7)
    g = _g(21)
    gvn, gv, wc, lanes = _ple_lanes(g, [3, 4, 2], C, R)
    x = (torch.randn(R, H, generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    ids = _ids(R, 24, 10, 21)
    pad = torch.zeros(R, 1, dtype=torch.bool, device=DEV)
    ent_h = torch.zeros(3, 16, dtype=torch.int64).pin_memory()
    ent_d = torch.zeros(3, 16, dtype=torch.int64, device=DEV)
    y = torch.zeros(R, C, dtype=torch.bfloat16, device=DEV)
    old = K.MOE_GROUP
    K.MOE_GROUP = True
    try:
        def body():
            ent_d.copy_(ent_h, non_blocking=True)
            SK.ple_conv_multi(ent_d, C, R)
            i2 = ids.clone()
            i2.masked_fill_(pad, -1)
            return K.moe_experts(x, i2, bank)
        _ple_fill(ent_h, gvn, gv, y, wc, lanes)
        pad[9:] = True
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            yd = body()
        # other lanes and row counts: new entries in the pinned host rows, new pad rows, new ids
        lanes2 = [{"r": slice(0, 5), "src": torch.tensor(SK.window_src(list(range(-1, 4)), (9, 6, 3), 9),
                                                           dtype=torch.int32, device=DEV),
                   "ps": torch.randn(1, C, 9, generator=g, device=DEV).to(torch.bfloat16)},
                  {"r": slice(5, 7), "src": torch.tensor(SK.window_src([-1, 0], (9, 6, 3), 9), dtype=torch.int32,
                                                         device=DEV),
                   "ps": torch.randn(1, C, 9, generator=g, device=DEV).to(torch.bfloat16)},
                  {"r": slice(7, 17), "src": torch.tensor(SK.window_src(list(range(-1, 9)), (9, 6, 3), 9),
                                                          dtype=torch.int32, device=DEV),
                   "ps": torch.randn(1, C, 9, generator=g, device=DEV).to(torch.bfloat16)}]
        _ple_fill(ent_h, gvn, gv, y, wc, lanes2)
        pad.zero_()
        pad[17:] = True
        ids.copy_(_ids(R, 24, 10, 22))
        gr.replay()
        torch.cuda.synchronize()
        for ln in lanes2:
            r = ln["r"]
            assert torch.equal(y[r], SK.ple_conv(gvn[r], gv[r], ln["ps"], wc, ln["src"])), r
        S = ids.shape[1]
        want = K.moe_experts(x[:17].contiguous(), ids[:17].contiguous(), bank)
        assert torch.equal(yd[:17 * S], want)
    finally:
        K.MOE_GROUP = old
    return "captured once; replayed with other lanes' entries, rows and pad rows: = the eager calls"


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                print(f"  {name:<44} ok   {fn() or ''}", flush=True)
                passed += 1
            except AssertionError as e:
                print(f"  {name:<44} FAIL {e}", flush=True)
                failed += 1
    print(f"{passed} passed" + (f", {failed} FAILED" if failed else ""))
    sys.exit(1 if failed else 0)
