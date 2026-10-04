"""The Kolibri-1 engine core on a tiny random Kolibri, CPU only.

  * the loader reads the set + the FP8 release (the mix) and the set alone, formats from the tensors;
  * `forward` equals `tools/kolibri_ref.layer_forward` over the same weights dequantised (the torch
    paths of the kernels round the projection inputs to bf16, so the bar is a tolerance, plus the
    argmax on every confident row);
  * prefill + decode one token at a time equals the full forward (the KV cache path), and
    `truncate` then re-decoding gives the same logits (prefix reuse);
  * the shared expert rides as expert E with weight 1.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine.kolibri.model import KolibriEngine  # noqa: E402
from tools import kolibri_tiny as tiny  # noqa: E402

IDS = [5, 17, 3, 88, 41, 41, 2, 60, 7, 19, 33, 71, 12, 9]


def _check(eng_logits, ref):
    """A random tiny model has near-ties in its router, and one flipped expert moves a row a lot;
    so most rows (not all) must be within 3 % of the logit scale, and confident rows agree."""
    scale = ref.abs().max().item()
    row_err = (eng_logits - ref).abs().amax(-1)
    assert (row_err <= 0.03 * scale).float().mean().item() >= 0.8, (row_err / scale)
    top2 = ref.topk(2, -1).values
    conf = (top2[:, 0] - top2[:, 1]) >= 0.2 * scale
    if conf.any():      # a random tiny model may have no row that confident
        agree = (eng_logits.argmax(-1)[conf] == ref.argmax(-1)[conf]).float().mean().item()
        assert agree >= 0.9, agree


def test_mix_forward_matches_reference():
    with tempfile.TemporaryDirectory() as d:
        st, rel = tiny.write(d)
        eng = KolibriEngine.load(st, rel, device="cpu", max_len=64, attention="torch", log=lambda s: None)
        assert eng.info["attention"] == "fp8"
        assert eng.layers[0].G.E == tiny.CFG["num_experts"] + 1     # the shared expert appended
        lg = eng.forward(IDS)
        _check(lg, tiny.reference_logits(st, rel, IDS, attn="fp8"))


def test_set_attention_forward_matches_reference():
    with tempfile.TemporaryDirectory() as d:
        st, rel = tiny.write(d)
        eng = KolibriEngine.load(st, None, device="cpu", max_len=64, attention="torch", log=lambda s: None)
        assert eng.info["attention"] == "set"
        _check(eng.forward(IDS), tiny.reference_logits(st, rel, IDS, attn="set"))


def test_manifest_picks_attention():
    import json
    with tempfile.TemporaryDirectory() as d:
        st, rel = tiny.write(d)
        json.dump({"attention": "set"}, open(os.path.join(st, "manifest.json"), "w"))
        eng = KolibriEngine.load(st, rel, device="cpu", max_len=64, attention="torch", log=lambda s: None)
        assert eng.info["attention"] == "set"


def test_decode_equals_forward_and_truncate():
    with tempfile.TemporaryDirectory() as d:
        st, rel = tiny.write(d)
        eng = KolibriEngine.load(st, rel, device="cpu", max_len=64, attention="torch", log=lambda s: None)
        full = eng.forward(IDS)
        eng.reset()
        last = eng.prefill(IDS[:4])
        assert torch.allclose(last, full[3], atol=2e-2, rtol=0)
        for i, t in enumerate(IDS[4:], start=4):
            lg = eng.decode(t)
            assert torch.allclose(lg, full[i], atol=2e-2, rtol=0), i
        assert eng.kv.length == len(IDS)
        # prefix reuse: back to 8 rows, the same continuation again (sliding window 5 < 8)
        eng.truncate(8)
        lg = eng.prefill(IDS[8:11], start=8)
        assert torch.allclose(lg, full[10], atol=2e-2, rtol=0)
        assert eng.kv.length == 11
        # chunked prefill (chunk < length) equals one chunk
        eng.reset()
        eng.chunk = 3
        assert torch.allclose(eng.prefill(IDS), full[-1], atol=2e-2, rtol=0)


def test_sample_row_matches_sampler_distribution():
    """The top-k fast path draws from the sampler's own filtered distribution."""
    from engine.sample import Sampler
    from server.kolibri_serve import sample_row
    torch.manual_seed(0)
    row = torch.randn(500) * 3
    s = Sampler(temperature=0.7, top_p=0.9, top_k=20, min_p=0.0, seed=None)
    want = s._filter(row.float())
    counts = torch.zeros(500)
    for _ in range(4000):
        counts[sample_row(s, row, 0)] += 1
    got = counts / counts.sum()
    assert (got[want == 0] == 0).all()
    assert (got - want).abs().max() < 0.03


def test_kvfullsh8_layout():
    """The published layout: q/o NVFP4 on sliding layers, k/v FP8 everywhere, full layers' attention and
    the shared expert FP8 (fp32 tables). Read from the set alone (its manifest), formats per tensor."""
    from engine.kolibri.weights import Concat, SharedExpert
    from engine.kolibri.kernels import FP8Linear
    with tempfile.TemporaryDirectory() as d:
        st, rel = tiny.write(d, layout="kvfullsh8")
        eng = KolibriEngine.load(st, rel, device="cpu", max_len=64, attention="torch", log=lambda s: None)
        assert eng.info["attention"] == "set"
        l0, l2 = eng.layers[0], eng.layers[2]
        assert isinstance(l0.qkv, Concat) and isinstance(l0.qkv.parts[1], FP8Linear)
        assert isinstance(l2.qkv, FP8Linear) and isinstance(l2.o, FP8Linear)
        assert isinstance(l0.shared, SharedExpert) and l0.G.E == tiny.CFG["num_experts"]
        _check(eng.forward(IDS), tiny.reference_logits(st, rel, IDS, attn="set"))
        eng.reset()
        full = eng.forward(IDS)
        eng.reset()
        eng.prefill(IDS[:6])
        for i, t in enumerate(IDS[6:], start=6):
            assert torch.allclose(eng.decode(t), full[i], atol=2e-2, rtol=0), i
