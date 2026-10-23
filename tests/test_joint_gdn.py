"""SPD-163, QWEN38_JOINT_GDN: the joint pass's grouped GDN recurrence, its plumbing on the CPU.

The kernels are Triton; their bits are `tools/gdn_wy_kernels.py check_grouped()` on the board. Here:
  1. `grouped_ok` takes tree blocks of 2..16 rows laid out alike, and refuses a chain, a 17-row block,
     mixed pending commits and a stride that differs;
  2. the offset table puts every engine's tensors at their own addresses: base + offset is the
     engine's first element, for each column, and T is each engine's row count;
  3. `_run_gdn` takes the grouped launch when the blocks fit it, else each engine's own
     `verify_mixer` call with the arguments the lone statement passes.
"""

from __future__ import annotations

import os
import sys

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from tools import gdn_wy_kernels as G  # noqa: E402

H, DK, DV, C, W = 4, 8, 8, 40, 4


def _items(sizes, pend=True, joined=True):
    mixed = torch.zeros(sum(sizes), C + 7)
    ab = torch.zeros(sum(sizes), 2 * H)
    out, r = [], 0
    for T in sizes:
        P = 3
        pd = ((torch.zeros(H, P, DK), torch.zeros(H, P, DV), torch.zeros(H, P),
               torch.zeros(P, dtype=torch.int32), torch.zeros(1, dtype=torch.int32))
              if pend else None)
        m = mixed[r:r + T, :C] if joined else torch.zeros(T, C + 7)[:, :C]
        out.append(dict(mixed=m, conv_state=torch.zeros(C, W - 1), window=torch.zeros(T, W, dtype=torch.long),
                        anc=torch.zeros(T, T, dtype=torch.bool), a_raw=ab[r:r + T, :H],
                        b_raw=ab[r:r + T, H:], state=torch.zeros(H, DK, DV), pend=pd,
                        fac_out=(torch.zeros(H, 32, DK), torch.zeros(H, 32, DV), torch.zeros(H, 32))))
        r += T
    return out


def test_grouped_ok_takes_trees_of_up_to_16_rows_laid_out_alike():
    assert G.grouped_ok(_items([2, 16, 5]))
    assert G.grouped_ok(_items([4, 4], pend=False))
    assert not G.grouped_ok(_items([4])), "one engine is the lone launch"
    assert not G.grouped_ok(_items([4, 17])), "a 17-row block is another tile"
    chain = _items([4, 4])
    chain[1]["window"] = None
    assert not G.grouped_ok(chain), "a chain"
    mixed = _items([4, 4])
    mixed[1]["pend"] = None
    assert not G.grouped_ok(mixed), "one pending commit and one none"
    odd = _items([4, 4])
    odd[1]["mixed"] = torch.zeros(4, C)
    assert not G.grouped_ok(odd), "another row stride"
    return "trees 2..16 alike: yes; one engine, 17 rows, a chain, mixed pend, other strides: no"


def test_the_offset_table_puts_each_engine_at_its_own_tensors():
    sizes = [3, 7, 2]
    items = _items(sizes)
    facs = [it["fac_out"] for it in items]
    rows = [0, 3, 10, 12]
    off = G.grouped_offsets(items, facs, rows, H * DV)
    assert len(off) == G.NOFF * 3
    cols = [("gc", lambda it, f: f[2]), ("anc", lambda it, f: it["anc"]), ("kk", lambda it, f: f[0]),
            ("mixed", lambda it, f: it["mixed"]), ("conv", lambda it, f: it["conv_state"]),
            ("window", lambda it, f: it["window"]), ("a", lambda it, f: it["a_raw"]),
            ("b", lambda it, f: it["b_raw"]), ("state", lambda it, f: it["state"]),
            ("delta", lambda it, f: f[1])]
    for e, (it, f) in enumerate(zip(items, facs)):
        row = off[e * G.NOFF:(e + 1) * G.NOFF]
        assert row[0] == sizes[e]
        for c, (name, get) in enumerate(cols, start=1):
            t, t0 = get(it, f), get(items[0], facs[0])
            assert t0.data_ptr() + row[c] * t.element_size() == t.data_ptr(), (e, name)
        assert row[11] == rows[e] * H * DV
        for c, (x, x0) in enumerate(zip(it["pend"], items[0]["pend"]), start=12):
            assert x0.data_ptr() + row[c] * x.element_size() == x.data_ptr(), (e, "pend", c)
        assert row[17] == sizes[e], "the ancestor mask's row stride"
    nopend = G.grouped_offsets(_items([2, 2], pend=False), facs[:2], rows[:3], H * DV)
    assert nopend[G.NOFF + 12:G.NOFF + 17] == [0] * 5
    return "every column of every engine: base + offset = its own first element; T its rows"


def test_run_gdn_groups_when_the_blocks_fit_and_calls_each_alone_otherwise():
    import engine.model as M
    from tools import gdn_verify_kernels as V
    import test_forward_tree as T
    eng, _ = T.build(seed=3)
    seen = {"grouped": 0, "alone": []}
    keep = (G.grouped_ok, G.wy_recurrence_grouped, V.verify_mixer, V.WY, V.WY_FUSED, V.WY_MAXT,
            V.WY_FUSED_MAXT)

    def grouped(items, **kw):
        seen["grouped"] += 1
        assert kw["value_heads"] == eng.cfg.linear_num_value_heads
        return [("o", it["mixed"].shape[0]) for it in items]

    def alone(mixed, conv_state, conv_w, a, b, a_log, dt_bias, state, **kw):
        seen["alone"].append((mixed.shape[0], kw["pend"], kw["store_state"], kw["window"] is not None))
        return "o", mixed.shape[0], None

    G.wy_recurrence_grouped, V.verify_mixer = grouped, alone
    V.WY, V.WY_FUSED, V.WY_MAXT, V.WY_FUSED_MAXT = True, True, 16, 16
    try:
        cfg = eng.cfg
        Hv, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim

        def x(T):
            return dict(mixed=torch.zeros(T, 3), conv_state=torch.zeros(1, 5, 3),
                        a_raw=torch.zeros(T, 2), b_raw=torch.zeros(T, 2),
                        state=torch.zeros(1, Hv, dk, dv), window=torch.zeros(T, 4), depths=None,
                        max_depth=1, anc=torch.zeros(T, T), pend=None, fac_out=None, store=True)
        G.grouped_ok = lambda items: True
        r = M._run_gdn(eng, "layers.0", [x(3), x(5)])
        assert seen["grouped"] == 1 and r == [("o", 3), ("o", 5)] and not seen["alone"]
        G.grouped_ok = lambda items: False
        r = M._run_gdn(eng, "layers.0", [x(3), x(5)])
        assert seen["grouped"] == 1 and seen["alone"] == [(3, None, True, True), (5, None, True, True)]
        assert r == [("o", 3), ("o", 5)]
        G.grouped_ok = lambda items: True
        M._run_gdn(eng, "layers.0", [x(4)])
        assert seen["grouped"] == 1, "a single engine takes its own launches"
        V.WY_FUSED_MAXT = 8
        M._run_gdn(eng, "layers.0", [x(3), x(5)])
        assert seen["grouped"] == 1, "a served fused tile under 16 rows is not the grouped one"
    finally:
        (G.grouped_ok, G.wy_recurrence_grouped, V.verify_mixer, V.WY, V.WY_FUSED, V.WY_MAXT,
         V.WY_FUSED_MAXT) = keep
    return "fits: one grouped call; else each engine's verify_mixer with the lone arguments"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:72s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
