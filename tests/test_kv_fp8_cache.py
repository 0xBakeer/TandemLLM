"""SPD-150: the e4m3 KV cache of Flash-Next (QWEN38_FN_KV_FP8), Qwen3.5 MoE (QWEN38_Q35M_KV_FP8) and the 27B
(QWEN38_KV_FP8), on the host.

The test matrix:

  1  every knob is off by default                                        test_the_knobs_are_off
  2  tools/kv_fp8: a row's codes and scale (amax / 448), the decoded row within e4m3's step, kv_quant2's CPU
     twin writes the rows at their slots and nothing else, the bytes a token   test_the_coding
  3  Flash-Next's attention reference on an e4m3 cache is its reference on the decoded bf16 cache, and
     close to the bf16 cache's                                           test_flash_next_attention_reads_codes
  4  the same for Qwen3.5 MoE's attention reference                      test_qwen35_moe_attention_reads_codes
  5  a Flash-Next state with the cache on: uint8 codes and fp32 scales [Hk, cap] on the trunk's attention
     layers, bf16 indexer keys; off: main's bf16 buffers and no scales; the MTP layer's cache stays bf16;
     a reset zeroes the scales                                           test_flash_next_state
  6  a call's rows staged in bf16 are coded at their slots, the indexer keys copied as they are; with the cache
     off the prep writes the caches at the slots (no staging)            test_flash_next_staging
  7  a capture's warm-up saves the scales' rows with the codes', and puts them back
                                                                         test_a_capture_saves_the_scales
  8  the reference forward and a joint pass's one-launch attention refuse an e4m3 cache (the reference keeps
     bf16; the lanes go one at a time)                                   test_paths_that_refuse
  9  the cost a token, a side slot's bytes and the cache views: Flash-Next 27,648 -> 15,552 (side slot 29,952
     -> 17,856, the MTP layer bf16); the views carry ks./vs.; Qwen3.5 MoE's prompt workspace counts the
     decoded cache                                                       test_flash_next_bytes_and_views
 10  the manifests name the e4m3 cache as another arch (qwen4_exp-kvfp8, qwen3_5_moe-kvfp8, qwen3_5-kvfp8),
     off as before                                                       test_the_manifests
 11  the 27B's cost a token with QWEN38_KV_FP8: 33,280 instead of 65,536  test_the_27b_cost

Run: python tests/test_kv_fp8_cache.py
"""

from __future__ import annotations

import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import kv_fp8 as KQ  # noqa: E402


class _Env:
    """Set environment keys for a block, then put them back."""

    def __init__(self, **kv):
        self.kv, self.old = kv, {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            os.environ[k] = v
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _engine(fp8: bool):
    """A Flash-Next engine object with only what the state and staging code reads (tests/fn_tiny.py's config)."""
    from engine.models.flashnext.engine import KV_STAGE_ROWS, FlashNextEngine
    from tests.fn_tiny import TINY
    from engine.models.flashnext.config import config_from_dict
    cfg = config_from_dict(dict(TINY))
    e = FlashNextEngine.__new__(FlashNextEngine)
    e.cfg, e.device, e.dtype, e.max_len, e.kv_fp8 = cfg, "cpu", torch.bfloat16, 64, fp8
    e._kst = torch.zeros(cfg.num_key_value_heads, KV_STAGE_ROWS, cfg.head_dim, dtype=torch.bfloat16)
    e._vst = torch.zeros_like(e._kst)
    e._ist = torch.zeros(KV_STAGE_ROWS, cfg.indexer_head_dim, dtype=torch.bfloat16)
    e._kidx = torch.arange(KV_STAGE_ROWS)
    e._pending = {}
    return e


def test_the_knobs_are_off():
    from engine.settings import KNOBS
    for k in ("FN_KV_FP8", "Q35M_KV_FP8", "KV_FP8"):
        assert KNOBS[k][0] == "0", (k, KNOBS[k])
    return "FN_KV_FP8, Q35M_KV_FP8, KV_FP8: 0"


def test_the_coding():
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(2, 9, 64, generator=g) * torch.linspace(0.01, 30, 9)[None, :, None]).to(torch.bfloat16)
    codes, sc = KQ.quantize_kv(x)
    assert codes.dtype == torch.uint8 and sc.dtype == torch.float32 and sc.shape == (2, 9)
    assert torch.allclose(sc, x.float().abs().amax(-1) / KQ.FP8_MAX)
    y = KQ.kv_dequant(codes, sc)
    # e4m3 keeps 3 mantissa bits: a value's error at most half its step, 1/16 of it, plus bf16's rounding
    err = ((y.float() - x.float()).abs() / x.float().abs().clamp_min(1e-30))[x.float().abs() > sc[..., None] * 2 ** -6]
    assert err.max() <= 1 / 16 + 1 / 128, err.max()
    kd, vd = torch.zeros(2, 32, 64, dtype=torch.uint8), torch.zeros(2, 32, 64, dtype=torch.uint8)
    ks, vs = torch.zeros(2, 32), torch.zeros(2, 32)
    idx, slot = torch.tensor([3, 0, 5]), torch.tensor([10, 11, 30])
    KQ.kv_quant2(x, -x, idx, kd, ks, vd, vs, slot)
    c3, s3 = KQ.quantize_kv(x[:, [3, 0, 5]])
    cv, sv = KQ.quantize_kv(-x[:, [3, 0, 5]])
    assert torch.equal(kd[:, [10, 11, 30]], c3) and torch.equal(ks[:, [10, 11, 30]], s3)
    assert torch.equal(vd[:, [10, 11, 30]], cv) and torch.equal(vs[:, [10, 11, 30]], sv)
    others = [i for i in range(32) if i not in (10, 11, 30)]
    assert not kd[:, others].any() and not ks[:, others].any()
    assert KQ.bytes_per_token(2, 256, False) == 2048 and KQ.bytes_per_token(2, 256, True) == 1040
    return f"max relative error {err.max():.4f}; 2 heads of 256: 2,048 -> 1,040 bytes a token"


def _sparse_case(seed=0, M=3, cap=40, HQ=4, HK=2, D=16):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(M, HQ, D, generator=g).to(torch.bfloat16)
    kc = torch.randn(HK, cap, D, generator=g).to(torch.bfloat16)
    vc = torch.randn(HK, cap, D, generator=g).to(torch.bfloat16)
    return q, kc, vc


def _close(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), 0).item()


def test_flash_next_attention_reads_codes():
    from tools import fn_seq_kernels as SK
    q, kc, vc = _sparse_case()
    M = q.shape[0]
    ctl = torch.zeros(M, 4, dtype=torch.int32)
    ctl[:, 0], ctl[:, 1], ctl[:, 2] = torch.tensor([30, 31, 32]), 30, 0
    sel = torch.zeros(M, SK.NSEL, dtype=torch.int32)
    for t in range(M):
        n = int(ctl[t, 0])
        sel[t, :n] = torch.cat([torch.arange(0, 28, dtype=torch.int32)[:n - (t + 2)],
                                torch.arange(28, 30, dtype=torch.int32), 30 + torch.arange(t, dtype=torch.int32)])[:n]
    anc = torch.zeros(M, SK.ANC, dtype=torch.int32)
    anc[:, :3] = torch.tensor([30, 31, 32], dtype=torch.int32)
    k8, ks = KQ.quantize_kv(kc)
    v8, vs = KQ.quantize_kv(vc)
    o8 = SK.attn_rows_reference(q, k8, v8, ctl, anc, sel, ks=ks, vs=vs)
    od = SK.attn_rows_reference(q, KQ.kv_dequant(k8, ks), KQ.kv_dequant(v8, vs), ctl, anc, sel)
    ob = SK.attn_rows_reference(q, kc, vc, ctl, anc, sel)
    assert torch.equal(o8, od)
    c = _close(o8, ob)
    assert c > 0.995, c
    return f"the decoded cache's own result; cosine to the bf16 cache's {c:.5f}"


def test_qwen35_moe_attention_reads_codes():
    from engine.models.qwen3_5_moe import kernels as AK
    q, kc, vc = _sparse_case(seed=2)
    M = q.shape[0]
    ctl = torch.tensor([[31, 30], [32, 30], [33, 30]], dtype=torch.int32)
    anc = torch.tensor([[30, 0, 0], [30, 31, 0], [30, 31, 32]], dtype=torch.int32)
    k8, ks = KQ.quantize_kv(kc)
    v8, vs = KQ.quantize_kv(vc)
    o8 = AK.attn_rows_reference(q, k8, v8, ctl, anc, ks=ks, vs=vs)
    od = AK.attn_rows_reference(q, KQ.kv_dequant(k8, ks), KQ.kv_dequant(v8, vs), ctl, anc)
    ob = AK.attn_rows_reference(q, kc, vc, ctl, anc)
    assert torch.equal(o8, od) and o8.shape == (M, 4, 16)
    c = _close(o8, ob)
    assert c > 0.995, c
    return f"cosine to the bf16 cache's {c:.5f}"


def test_flash_next_state():
    from engine.models.flashnext.ref import MTP
    on, off = _engine(True), _engine(False)
    c = on.cfg
    li = c.attention_layers[0]
    so, sf = on.new_state(48), off.new_state(48)
    Hk, D = c.num_key_value_heads, c.head_dim
    assert so.k[li].dtype == so.v[li].dtype == torch.uint8 and so.k[li].shape == (Hk, 48, D)
    assert so.ks[li].dtype == so.vs[li].dtype == torch.float32 and so.ks[li].shape == (Hk, 48)
    assert so.ik[li].dtype == torch.bfloat16
    assert sf.k[li].dtype == torch.bfloat16 and not sf.ks and not sf.vs
    m = on.new_mtp_state(48) if hasattr(on, "new_mtp_state") else None
    if m is not None:
        assert m.k[MTP].dtype == torch.bfloat16 and not m.ks, "the MTP layer's cache stays bf16"
    so.ks[li].fill_(1.0)
    on.reset(so)
    assert not so.ks[li].any()
    cl = so.clone()
    assert cl.ks[li] is not so.ks[li] and torch.equal(cl.ks[li], so.ks[li])
    return f"layer {li}: codes {tuple(so.k[li].shape)} uint8, scales {tuple(so.ks[li].shape)} fp32"


def test_flash_next_staging():
    on, off = _engine(True), _engine(False)
    c = on.cfg
    li = c.attention_layers[0]
    st = on.new_state(48)
    k, v, ik, idx = on._kv_targets(st, li, 3)
    assert k is on._kst and v is on._vst and ik is on._ist and torch.equal(idx, torch.arange(3))
    g = torch.Generator().manual_seed(4)
    k[:, :3] = torch.randn(c.num_key_value_heads, 3, c.head_dim, generator=g).to(torch.bfloat16)
    v[:, :3] = torch.randn(c.num_key_value_heads, 3, c.head_dim, generator=g).to(torch.bfloat16)
    ik[:3] = torch.randn(3, c.indexer_head_dim, generator=g).to(torch.bfloat16)
    slot = torch.tensor([20, 21, 22])
    on._kv_code(st, li, k, v, ik, idx, slot)
    ck, sk = KQ.quantize_kv(k[:, :3])
    assert torch.equal(st.k[li][:, 20:23], ck) and torch.equal(st.ks[li][:, 20:23], sk)
    assert torch.equal(st.ik[li][20:23], ik[:3]), "the indexer keys are copied as they are"
    sf = off.new_state(48)
    k2, v2, ik2, idx2 = off._kv_targets(sf, li, 3)
    assert k2 is sf.k[li] and v2 is sf.v[li] and ik2 is sf.ik[li] and idx2 is None
    kt = torch.zeros(c.num_key_value_heads, 5, c.head_dim, dtype=torch.bfloat16)
    k3, _, _, idx3 = on._kv_targets(st, li, 5, kt, kt.clone(), torch.zeros(5, c.indexer_head_dim))
    assert k3 is kt and torch.equal(idx3, torch.arange(5)), "a prompt chunk stages in its own scratch"
    return "3 rows coded at slots 20..22; off: the caches themselves"


def test_a_capture_saves_the_scales():
    from engine.models.flashnext import engine as E
    on = _engine(True)
    li = on.cfg.attention_layers[0]
    st = on.new_state(200)
    for t in E._state_tensors(st).values():
        if t.is_floating_point():
            t.copy_(torch.randn_like(t.float()).to(t.dtype))
        else:
            t.copy_(torch.randint(0, 255, t.shape, dtype=t.dtype))
    st.length = 40
    before = {k: v.clone() for k, v in E._state_tensors(st).items()}
    saved = E._state_saved(on, st)
    names = {id(t): k for k, t in E._state_tensors(st).items()}
    got = {names[id(t)]: (lo, n) for t, ax, lo, n, x in saved}
    assert got[f"ks.{li}"] == (40, min(E.CAPTURE_ROWS, 160)) and got[f"vs.{li}"] == got[f"ks.{li}"]
    st.ks[li][:, 40:40 + E.CAPTURE_ROWS].add_(1.0)          # what a warm-up writes: rows from the length on
    st.k[li][:, 40:40 + E.CAPTURE_ROWS].add_(1)
    E._state_restore(saved)
    assert all(torch.equal(before[k], v) for k, v in E._state_tensors(st).items())
    return f"ks.{li} / vs.{li} rows {got[f'ks.{li}']} saved and restored"


def test_paths_that_refuse():
    on = _engine(True)
    li = on.cfg.attention_layers[0]
    st = on.new_state(32)
    try:
        on.forward(torch.tensor([5, 6]), st)
        raise AssertionError("the reference forward ran on an e4m3 cache")
    except NotImplementedError as e:
        msg = str(e)
    assert on._attn_multi([{"st": st}], [slice(0, 1)], None, None, li, None) is None
    return msg[:60]


class _Fn:
    """What FlashNextServing reads of its engine for the bytes and views: Flash-Next's real attention shape."""

    def __init__(self, fp8: bool):
        self.kv_fp8 = fp8
        self.cfg = types.SimpleNamespace(attention_layers=list(range(3, 48, 4)), num_key_value_heads=2, head_dim=256,
                                         indexer_head_dim=128, has_indexer=True, num_attention_heads=24)


def _serving(fp8: bool):
    from engine.models.flashnext.ref import FNState
    from engine.models.flashnext.serving import FlashNextCacheState, FlashNextServing
    s = FlashNextServing.__new__(FlashNextServing)
    s.fn, s.cfg = _Fn(fp8), _Fn(fp8).cfg
    st = FNState(cap=8)
    for li in (3, 7):
        st.k[li], st.v[li], st.ik[li] = (torch.zeros(2, 8, 4, dtype=torch.uint8 if fp8 else torch.bfloat16),
                                         torch.zeros(2, 8, 4, dtype=torch.uint8 if fp8 else torch.bfloat16),
                                         torch.zeros(8, 2))
        if fp8:
            st.ks[li], st.vs[li] = torch.zeros(2, 8), torch.zeros(2, 8)
    s.st = st
    s.state = types.SimpleNamespace(buffers=lambda: {"gdn.S.0": torch.zeros(4)})
    cs = FlashNextCacheState.__new__(FlashNextCacheState)
    cs.s = s
    return s, cs


def test_flash_next_bytes_and_views():
    out = {}
    for fp8 in (False, True):
        s, cs = _serving(fp8)
        cost = s.cost()
        side = s.side_slot_bytes(1024, paged=False)[0]
        names = [n for n, _, _ in cs.views()]
        out[fp8] = (cost.view_bytes_per_token, side, names)
        assert s.side_slot_bytes(1024, paged=True)[0] == 0
    assert out[False][:2] == (27648, 29952), out[False]
    assert out[True][:2] == (15552, 17856), out[True]
    assert out[False][2] == ["k.3", "v.3", "ik.3", "k.7", "v.7", "ik.7"]
    assert out[True][2] == ["k.3", "v.3", "ks.3", "vs.3", "ik.3", "k.7", "v.7", "ks.7", "vs.7", "ik.7"]
    from engine.models.qwen3_5_moe.serving import Qwen35MoeServing
    ws = {}
    for fp8 in (False, True):
        s, _ = _serving(fp8)
        s.__class__ = Qwen35MoeServing
        s.cfg.has_indexer = False
        s.max_len = 1024
        ws[fp8] = s.cost().workspace_per_token
    assert ws[True] - ws[False] == 2 * 256 * 2 * 2, ws
    return f"a token 27,648 -> 15,552, a side slot 29,952 -> 17,856; Qwen3.5 MoE's prompt workspace +{ws[True] - ws[False]} B"


def test_the_manifests():
    from engine.models.flashnext import serving as FS
    from engine.models.qwen3_5_moe import serving as QS
    from engine.models.qwen3_5 import adapter as QA
    off = (FS.manifest().arch, QS.manifest().arch, QA.manifest().arch)
    with _Env(QWEN38_FN_KV_FP8="1", QWEN38_Q35M_KV_FP8="1", QWEN38_KV_FP8="1"):
        on = (FS.manifest().arch, QS.manifest().arch, QA.manifest().arch)
    assert off == ("qwen4_exp", "qwen3_5_moe", "qwen3_5"), off
    assert on == ("qwen4_exp-kvfp8", "qwen3_5_moe-kvfp8", "qwen3_5-kvfp8"), on
    return ", ".join(on)


def test_the_27b_cost():
    from engine.models.qwen3_5 import adapter as QA
    cfg = types.SimpleNamespace(attention_layers=list(range(16)), num_key_value_heads=4, head_dim=256)
    st = types.SimpleNamespace(S=torch.zeros(2), conv=torch.zeros(2), nbytes=16)
    got = [QA.cost(types.SimpleNamespace(cfg=cfg, state=st, kv=types.SimpleNamespace(fp8=f))).view_bytes_per_token
           for f in (False, True)]
    assert got == [65536, 33280], got
    return f"{got[0]:,} -> {got[1]:,} bytes a token"


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
