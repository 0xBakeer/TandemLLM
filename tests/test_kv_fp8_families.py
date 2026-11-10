"""SPD-150: the e4m3 KV cache of LFM2 (LFM2_KV_FP8) and NemotronH (NH_KV_FP8), both off by default.

  1  each knob is in its family's knobs.py (default 0, read in its engine) and that file in the
     family's manifest                                                    test_1_knobs_default_off
  2  the manifest's arch is lfm2_moe-kvfp8 / nemotron_h-kvfp8 with the knob on, the arch without
                                                                          test_2_manifest_arch
  3  the KV bytes a token, bf16 and e4m3 (codes and one fp32 scale a KV head), at the served models'
     shapes; cost() and side_slot_bytes of the tiny models with the knob off and on
                                                                          test_3_bytes_a_token
  4  the state with the knob on: K and V uint8 [Hk, cap, D] and their scales fp32 [Hk, cap]; off: bf16
     and no scales; the cache views carry the scales, a capture's warm-up saves them, memory_parts
     counts them                                                          test_4_state_views_saved
  5  tools/kv_fp8: kv_quant2's CPU path codes the rows `idx` at `slot` and nothing else, within
     e4m3's half step; kv_dequant_cat is the decoded rows, then the chunk's   test_5_coding
  6  the row attention's torch twin on the decoded cache within e4m3's error of the bf16 cache's
                                                                          test_6_reference_attention
  7  off path: attn_rows without scales launches the bf16 kernel with main's arguments; the kernel's
     FP8 branch is a constexpr off by default, its scale arguments trailing and None by default
                                                                          test_7_off_launch
  8  LFM2's tiny engine with the e4m3 cache (Triton on a GPU, or TRITON_INTERPRET=1): the first
     prompt chunk is the bf16 engine's bit for bit and codes its rows as quantize_kv does; a second
     chunk and decode steps follow the bf16 engine closely; a verify block's rows are the decode
     steps' bit for bit; a moved path's codes and scales commit; a joint decode step and a joint
     verify pass over two coded states give each state's lone bits; the self-check passes
                                                                          test_8_lfm2_engine
  9  the same on NemotronH's tiny engine                                  test_9_nemotron_engine

Run: python tests/test_kv_fp8_families.py   (TRITON_INTERPRET=1 for 8 and 9 on the CPU)
"""

from __future__ import annotations

import contextlib
import inspect
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.models.lfm2_moe import kernels as K  # noqa: E402
from tools import kv_fp8 as KV  # noqa: E402

SKIP = None if (K.HAVE_TRITON and (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET"))) else \
    "skipped: Triton on a GPU, or TRITON_INTERPRET=1 (the CPU interpreter)"
DEV = "cuda" if torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET") else "cpu"


@contextlib.contextmanager
def knob(name: str, value: str | None):
    """QWEN38_<name> set to `value` (None: unset) inside the block."""
    key = "QWEN38_" + name
    old = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _lfm(fp8: bool):
    """(cfg, engine, bf16 engine): tests/test_lfm2_moe_engine.py's tiny model, a second engine over its
    weights built with LFM2_KV_FP8 as asked."""
    from engine.models.lfm2_moe.engine import Lfm2MoeEngine
    from tests import test_lfm2_moe_engine as E
    cfg, eng0, _ = E.tiny()
    with knob("LFM2_KV_FP8", "1" if fp8 else None):
        eng = Lfm2MoeEngine(cfg, eng0.W, eng0.experts, device=eng0.device, max_len=eng0.max_len)
    return cfg, eng, eng0


def _nh(fp8: bool):
    from engine.models.nemotron_h.engine import NemotronHEngine
    from tests import test_nemotron_h_engine as E
    cfg, eng0, _ = E.tiny()
    with knob("NH_KV_FP8", "1" if fp8 else None):
        eng = NemotronHEngine(cfg, eng0.W, eng0.experts, device=eng0.device, max_len=eng0.max_len)
    return cfg, eng, eng0


# ------------------------------------------------------------------ 1
def test_1_knobs_default_off():
    from engine import settings
    from engine.methods.manifest import _knob_table
    from engine.models.lfm2_moe import serving as LS
    from engine.models.nemotron_h import serving as NS
    for name, pkg, mods in (("LFM2_KV_FP8", "lfm2_moe", LS.MODULES), ("NH_KV_FP8", "nemotron_h", NS.MODULES)):
        own = f"engine/models/{pkg}/knobs.py"
        assert _knob_table(os.path.join(ROOT, own))[name] == ("0", f"engine/models/{pkg}/engine.py"), name
        assert settings.KNOBS[name] == ("0", f"engine/models/{pkg}/engine.py"), name
        with knob(name, None):
            assert settings.SETTINGS.get(name) == "0"
        assert own in mods, f"{own} not in its manifest"
    return "LFM2_KV_FP8 and NH_KV_FP8 default 0"


# ------------------------------------------------------------------ 2
def test_2_manifest_arch():
    from engine.models.lfm2_moe import serving as LS
    from engine.models.nemotron_h import serving as NS
    for name, mod, arch in (("LFM2_KV_FP8", LS, "lfm2_moe"), ("NH_KV_FP8", NS, "nemotron_h")):
        with knob(name, None):
            off = mod.manifest()
        with knob(name, "0"):
            assert mod.manifest().arch == arch
        with knob(name, "1"):
            on = mod.manifest()
        assert off.arch == arch and on.arch == arch + "-kvfp8", (off.arch, on.arch)
        assert off.modules == on.modules and off.data == on.data
    return "lfm2_moe / lfm2_moe-kvfp8, nemotron_h / nemotron_h-kvfp8"


# ------------------------------------------------------------------ 3
def test_3_bytes_a_token():
    from engine.models.lfm2_moe.serving import Lfm2MoeServing
    from engine.models.nemotron_h.serving import NemotronHServing
    # the served shapes: LFM2.5-24B-A2B 10 attention layers of 8 KV heads of 64; NemotronH 2 KV heads of 128,
    # Lightning 6 attention layers, Bolt and Super VL 8
    want = {"lfm2.5-24b-a2b": (KV.bytes_per_token(8, 64, False) * 10, KV.bytes_per_token(8, 64, True) * 10),
            "lightning": (KV.bytes_per_token(2, 128, False) * 6, KV.bytes_per_token(2, 128, True) * 6),
            "bolt": (KV.bytes_per_token(2, 128, False) * 8, KV.bytes_per_token(2, 128, True) * 8),
            "super-vl": (KV.bytes_per_token(2, 128, False) * 8, KV.bytes_per_token(2, 128, True) * 8)}
    assert want == {"lfm2.5-24b-a2b": (20480, 10880), "lightning": (6144, 3168), "bolt": (8192, 4224),
                    "super-vl": (8192, 4224)}, want
    for build, cls in ((_lfm, Lfm2MoeServing), (_nh, NemotronHServing)):
        cfg, eng, eng0 = build(True)
        off, on = cls(eng0, 64), cls(eng, 64)
        n = len(cfg.attention_layers)
        assert off.cost().view_bytes_per_token == cfg.kv_bytes_per_token(2) == n * 2 * cfg.kv_dim * 2
        assert on.cost().view_bytes_per_token == n * 2 * (cfg.kv_dim + 4 * cfg.num_key_value_heads)
        assert on.cost().section_bytes == off.cost().section_bytes
        for s in (off, on):
            per, const = s.side_slot_bytes(64, paged=False)
            assert per == s.cost().view_bytes_per_token and s.side_slot_bytes(64, paged=True)[0] == 0
    return ", ".join(f"{k} {a:,} -> {b:,}" for k, (a, b) in want.items())


# ------------------------------------------------------------------ 4
def test_4_state_views_saved():
    from engine.models.lfm2_moe import engine as LE
    from engine.models.lfm2_moe.serving import Lfm2MoeServing
    from engine.models.nemotron_h import engine as NE
    from engine.models.nemotron_h.serving import NemotronHServing
    out = []
    for build, cls, mod in ((_lfm, Lfm2MoeServing, LE), (_nh, NemotronHServing, NE)):
        cfg, eng, eng0 = build(True)
        Hk, D = cfg.num_key_value_heads, cfg.head_dim
        st, st0 = eng.new_state(96), eng0.new_state(96)
        assert eng.kv_fp8 and not eng0.kv_fp8
        assert sorted(st.ks) == sorted(st.vs) == sorted(st.k) == list(cfg.attention_layers)
        for li in cfg.attention_layers:
            assert st.k[li].dtype == st.v[li].dtype == torch.uint8 and st.k[li].shape == (Hk, 96, D)
            assert st.ks[li].dtype == st.vs[li].dtype == torch.float32 and st.ks[li].shape == (Hk, 96)
            assert st0.k[li].dtype == torch.bfloat16 and st0.k[li].shape == (Hk, 96, D)
        assert not st0.ks and not st0.vs
        srv, srv0 = cls(eng, 64), cls(eng0, 64)
        names = [n for n, _, _ in srv.cache_state().views()]
        assert names == [f"{x}.{li}" for li in cfg.attention_layers for x in ("k", "v", "ks", "vs")], names
        assert [n for n, _, _ in srv0.cache_state().views()] == [f"{x}.{li}" for li in cfg.attention_layers
                                                                 for x in ("k", "v")]
        assert all(ax == 1 for _, _, ax in srv.cache_state().views())
        kv = srv.memory_parts()["kv_pool"]
        assert kv == len(cfg.attention_layers) * 64 * 2 * (Hk * D + 4 * Hk), kv
        assert srv0.memory_parts()["kv_pool"] == len(cfg.attention_layers) * 64 * 2 * Hk * D * 2
        st.length = 10
        saved = mod._state_saved(eng, st)
        held = {id(t) for t, *_ in saved}
        assert all(id(st.ks[li]) in held and id(st.vs[li]) in held for li in cfg.attention_layers)
        t, ax, lo, n, x = next(e for e in saved if e[0] is st.ks[cfg.attention_layers[0]])
        assert (ax, lo, n) == (1, 10, mod.CAPTURE_ROWS) and x.shape == (Hk, mod.CAPTURE_ROWS)
        out.append(f"{type(eng).__name__} {len(cfg.attention_layers)} layers")
    return ", ".join(out)


# ------------------------------------------------------------------ 5
def test_5_coding():
    g = torch.Generator().manual_seed(3)
    Hk, R, D, cap = 2, 6, 64, 40
    src = (torch.randn(Hk, R, D, generator=g) * 3).to(torch.bfloat16)
    vsrc = (torch.randn(Hk, R, D, generator=g) * 0.2).to(torch.bfloat16)
    kd, vd = torch.full((Hk, cap, D), 7, dtype=torch.uint8), torch.full((Hk, cap, D), 7, dtype=torch.uint8)
    ks, vs = torch.full((Hk, cap), -1.0), torch.full((Hk, cap), -1.0)
    idx, slot = torch.tensor([4, 0, 2]), torch.tensor([30, 11, 12])
    KV.kv_quant2(src, vsrc, idx, kd, ks, vd, vs, slot)
    other = [s for s in range(cap) if s not in slot.tolist()]
    assert bool((kd[:, other] == 7).all()) and bool((ks[:, other] == -1).all()), "a row outside `slot` was written"
    for x, d, s in ((src, kd, ks), (vsrc, vd, vs)):
        codes, sc = KV.quantize_kv(x[:, idx])
        assert torch.equal(d[:, slot], codes) and torch.equal(s[:, slot], sc)
        back = KV.kv_dequant(d[:, slot], s[:, slot]).float()
        ref = x[:, idx].float()
        # e4m3: 3 mantissa bits (a half step 2^-4 of the value), the subnormal floor, and the bf16 rounding
        bound = ref.abs() * (2 ** -4 + 2 ** -8) + s[:, slot, None] * 2 ** -9
        assert bool(((back - ref).abs() <= bound + 1e-6).all()), float(((back - ref).abs() - bound).max())
    L, T = 20, 5
    rows = torch.randn(Hk, T, D, generator=g).to(torch.bfloat16)
    codes = torch.randint(0, 120, (Hk, cap, D), dtype=torch.uint8, generator=g)
    scales = torch.rand(Hk, cap, generator=g)
    cat = KV.kv_dequant_cat(codes, scales, L, rows, block=7)
    assert torch.equal(cat, torch.cat([KV.kv_dequant(codes[:, :L], scales[:, :L]), rows], 1))
    assert torch.equal(KV.kv_dequant_cat(codes, scales, 0, rows), rows)
    return "codes and scales at the slots only, within e4m3's half step"


# ------------------------------------------------------------------ 6
def test_6_reference_attention():
    g = torch.Generator().manual_seed(4)
    M, Hq, Hk, D, cap = 3, 8, 2, 64, 48
    q = torch.randn(M, Hq, D, generator=g).to(torch.bfloat16)
    kc = torch.randn(Hk, cap, D, generator=g).to(torch.bfloat16)
    vc = torch.randn(Hk, cap, D, generator=g).to(torch.bfloat16)
    ctl = torch.tensor([[30, 29], [31, 29], [31, 29]], dtype=torch.int32)
    anc = torch.zeros(M, K.ANC, dtype=torch.int32)
    anc[:, 0] = 29
    anc[1, 1], anc[2, 1] = 30, 31
    k8, ks = KV.quantize_kv(kc)
    v8, vs = KV.quantize_kv(vc)
    exact = K.attn_rows_reference(q, kc, vc, ctl, anc)
    coded = K.attn_rows_reference(q, k8, v8, ctl, anc, ks=ks, vs=vs)
    same = K.attn_rows_reference(q, KV.kv_dequant(k8, ks), KV.kv_dequant(v8, vs), ctl, anc)
    assert torch.equal(coded, same), "the twin does not read the decoded values"
    err = float((coded - exact).abs().max() / exact.abs().max())
    assert err < 0.08, err
    return f"max error {err:.3f} of the largest output"


# ------------------------------------------------------------------ 7
class _Launch:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *a, **kw: self.calls.append((grid, a, kw))


def test_7_off_launch():
    w = inspect.signature(K.attn_rows).parameters
    assert list(w)[-2:] == ["ks", "vs"] and w["ks"].default is None and w["vs"].default is None
    assert list(inspect.signature(K.qk_prep).parameters)[-2:] == ["q", "pdl"]
    if K.HAVE_TRITON:
        p = inspect.signature(K._attn_split.fn).parameters
        assert list(p)[-4:] == ["KS", "VS", "s_sh", "FP8"], list(p)
        assert p["FP8"].default is False and all(p[x].default is None for x in ("KS", "VS", "s_sh"))
    split, comb = _Launch(), _Launch()
    keep = {n: getattr(K, n) for n in ("_attn_split", "_attn_combine") if hasattr(K, n)}
    K._attn_split, K._attn_combine = split, comb
    try:
        M, Hq, Hk, D, cap = 2, 4, 2, 64, 32
        q = torch.zeros(M, Hq, D, dtype=torch.bfloat16)
        ctl, anc = torch.zeros(M, 2, dtype=torch.int32), torch.zeros(M, K.ANC, dtype=torch.int32)
        kc = torch.zeros(Hk, cap, D, dtype=torch.bfloat16)
        K.attn_rows(q, kc, kc, ctl, anc, 4)
        k8 = torch.zeros(Hk, cap, D, dtype=torch.uint8)
        sc = torch.zeros(Hk, cap)
        K.attn_rows(q, k8, k8, ctl, anc, 4, ks=sc, vs=sc)
    finally:
        for n in ("_attn_split", "_attn_combine"):
            if n in keep:
                setattr(K, n, keep[n])
            else:
                delattr(K, n)
    (g0, a0, kw0), (g1, a1, kw1) = split.calls
    assert g0 == g1 == (M, Hk, 4) and len(a0) == len(a1) == 12
    assert set(kw0) == {"SCALE", "REP", "D", "HQ", "HK", "BM", "BN", "CHUNK_", "NA", "NS", "DOT32_", "num_warps",
                        "num_stages"}, sorted(kw0)
    assert set(kw1) - set(kw0) == {"KS", "VS", "s_sh", "FP8"} and kw1["FP8"] is True and kw1["s_sh"] == cap
    assert {k: v for k, v in kw1.items() if k in kw0} == kw0
    assert len(comb.calls) == 2 and comb.calls[0][2] == comb.calls[1][2]
    return "the bf16 launch takes main's 12 arguments and 13 keywords"


# ------------------------------------------------------------------ 8, 9
def _decode(eng, st, toks):
    out = []
    for t in toks:
        eng.stage(t, st)
        out.append(eng.step(st)[0].clone())
        st.length += 1
    return torch.stack(out)


def _rel(a, b) -> float:
    return float((a.float() - b.float()).abs().max() / b.float().abs().max())


def _engine_check(build, prompt, cap: int):
    """What test 8 and 9 check on one family's tiny engine (see the module docstring)."""
    cfg, eng, eng0 = build(True)
    st, s0 = eng.new_state(cap), eng0.new_state(cap)
    li = cfg.attention_layers[0]
    a = 10
    # the first chunk attends to its own rows in bf16: the bf16 engine's logits; its rows coded as quantize_kv codes
    l8 = eng.prefill(prompt[:a], st, head="all")
    l0 = eng0.prefill(prompt[:a], s0, head="all")
    assert torch.equal(l8, l0), "the first chunk differs from the bf16 engine's"
    codes, sc = KV.quantize_kv(s0.k[li][:, :a])
    same = float((st.k[li][:, :a] == codes).float().mean())
    assert same >= (1.0 if DEV == "cpu" else 0.999) and torch.allclose(st.ks[li][:, :a], sc), same
    # a second chunk reads the cached rows decoded
    l8 = eng.prefill(prompt[a:], st, head="all")
    l0 = eng0.prefill(prompt[a:], s0, head="all")
    e_chunk = _rel(l8, l0)
    assert e_chunk < 0.1, e_chunk
    L = st.length
    rec = [t for d in (st.conv, getattr(st, "ssm", {})) for t in d.values()]
    snap = [t.clone() for t in rec]
    toks = [11, 50, 3, 7, 19, 44]
    dec = _decode(eng, st, toks)
    e_dec = _rel(dec, _decode(eng0, s0, toks))
    assert e_dec < 0.1, e_dec
    # a verify block from the state at L is the decode steps bit for bit on the coded cache
    st.length = L
    for t, x in zip(rec, snap):
        t.copy_(x)
    eng.stage_rows(toks, list(range(-1, len(toks) - 1)), st)
    got = eng.rows_step(st, len(toks))
    assert torch.equal(got, dec), "a verify row differs from the decode step's on the e4m3 cache"
    # a moved path: its codes and scales go to their positions, and the chain continues from them
    eng.stage_rows([toks[0], 9, toks[1]], [-1, 0, 0], st)
    eng.rows_step(st, 3)
    k2, s2 = st.k[li][:, L + 2].clone(), st.ks[li][:, L + 2].clone()
    eng.commit(st, [0, 2])
    assert st.length == L + 2 and torch.equal(st.k[li][:, L + 1], k2) and torch.equal(st.ks[li][:, L + 1], s2)
    rest = _decode(eng, st, toks[2:])
    assert torch.equal(rest, dec[2:]), "the committed moved path does not continue the chain"
    # joint passes over two coded states: each state's rows are its lone call's
    a2, b2, a3, b3 = (eng.new_state(cap) for _ in range(4))
    for s_, n_ in ((a2, len(prompt)), (b2, 12), (a3, len(prompt)), (b3, 12)):
        eng.prefill(prompt[:n_], s_)
    rbs = eng.decode_multi([(5, a2, "j0"), (9, b2, "j1")])
    la, lb = rbs[0]["logits"][0].clone(), rbs[1]["logits"][0].clone()
    a2.length += 1
    b2.length += 1
    assert torch.equal(la, _decode(eng, a3, [5])[0]) and torch.equal(lb, _decode(eng, b3, [9])[0]), "joint decode"
    rbs = eng.verify_multi([([7, 8, 9], [-1, 0, 1], a2), ([4, 6], [-1, 0], b2)])
    ja, jb = rbs[0]["logits"][:3].clone(), rbs[1]["logits"][:2].clone()
    eng.stage_rows([7, 8, 9], [-1, 0, 1], a3)
    assert torch.equal(ja, eng.rows_step(a3, 3)), "joint verify"
    eng.stage_rows([4, 6], [-1, 0], b3)
    assert torch.equal(jb, eng.rows_step(b3, 2)), "joint verify"
    res = eng.self_check([int(t) for t in prompt], steps=6, log=None)
    assert res["ok"], res
    return f"second chunk {e_chunk:.3f}, decode {e_dec:.3f} of the bf16 engine's largest logit ({DEV})"


def test_8_lfm2_engine():
    if SKIP:
        return SKIP
    from tests.test_lfm2_moe_engine import PROMPT
    return _engine_check(_lfm, PROMPT, 128)


def test_9_nemotron_engine():
    if SKIP:
        return SKIP
    from tests.test_nemotron_h_engine import PROMPT
    return _engine_check(_nh, PROMPT, 256)


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<36} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            import traceback
            traceback.print_exc()
            print(f"  {name:<36} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
