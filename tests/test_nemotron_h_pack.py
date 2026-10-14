"""The NemotronH checkpoint reader: a tiny BF16 checkpoint written on the CPU reads back to the
reference's canonical names (and loads to the same forward); Lightning 30B-A3B's real headers (6,513
tensors, 270 of them the MTP head's) audit clean; a missing tensor, a wrong shape, a truncated shard and
a wrong sha256 stop with the name."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models.lfm2_moe.pack import TensorInfo  # noqa: E402
from engine.models.nemotron_h.config import config_from_dict  # noqa: E402
from engine.models.nemotron_h.load import load  # noqa: E402
from engine.models.nemotron_h.pack import PackIndex, canonical, expected  # noqa: E402
from engine.models.nemotron_h.ref import NemotronHRef  # noqa: E402
from tests.test_nemotron_h_ref import PROMPT, PUZZLE, TINY, random_weights  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "nemotron_h")


def _raises(fn, *needles):
    try:
        fn()
    except (ValueError, FileNotFoundError) as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return
    raise AssertionError(f"no error naming {needles}")


def _write(d: str, raw: dict, W: dict, shards: int = 2) -> None:
    from safetensors.torch import save_file
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(raw, f)
    keys = sorted(W)
    wm = {}
    for s in range(shards):
        part = {("backbone." + k if not k.startswith(("lm_head", "mtp.")) else k): W[k].to(
            torch.float32 if k.endswith("e_score_correction_bias") else torch.bfloat16).contiguous()
            for k in keys[s::shards]}
        fn = f"model-{s + 1:05d}-of-{shards:05d}.safetensors"
        save_file(part, os.path.join(d, fn))
        wm.update({k: fn for k in part})
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": wm}, f)


def test_canonical_names():
    assert canonical("backbone.layers.3.mixer.in_proj.weight") == "layers.3.mixer.in_proj.weight"
    assert canonical("language_model.backbone.norm_f.weight") == "norm_f.weight"
    assert canonical("language_model.lm_head.weight") == "lm_head.weight"
    assert canonical("mtp.layers.0.eh_proj.weight") == "mtp.layers.0.eh_proj.weight"
    assert canonical("vision_model.radio_model.blocks.0.attn.qkv.weight") is None


def test_tiny_checkpoint_round_trip():
    for raw in (TINY, PUZZLE):
        cfg = config_from_dict(raw, "tiny")
        W = random_weights(cfg)
        with tempfile.TemporaryDirectory() as d:
            _write(d, raw, W)
            idx, W2, experts, timing = load(d, "cpu", dtype=torch.float32, log=None)
            assert not idx.has_mtp and len(idx.tensors) == len(W)
            a = NemotronHRef(cfg, {k: v.to(torch.bfloat16).float() if not k.endswith("e_score_correction_bias")
                                   else v for k, v in W.items()})
            b = NemotronHRef(idx.cfg, W2, experts=experts)
            la = a.prefill(PROMPT, a.new_state(32))
            lb = b.prefill(PROMPT, b.new_state(32))
            assert torch.equal(la, lb), (la - lb).abs().max()
            db = idx.decode_bytes(context=1000)
            assert db["parts"]["kv_read"] == 1000 * cfg.kv_bytes_per_token()


def test_audit_stops_by_name():
    cfg = config_from_dict(TINY, "tiny")
    W = random_weights(cfg)
    with tempfile.TemporaryDirectory() as d:
        W2 = dict(W)
        del W2["layers.3.mixer.dt_bias"]
        _write(d, TINY, W2)
        _raises(lambda: PackIndex.open(d).audit(), "layers.3.mixer.dt_bias")
    with tempfile.TemporaryDirectory() as d:
        W2 = dict(W)
        W2["layers.2.mixer.k_proj.weight"] = torch.zeros(3, cfg.hidden_size)
        _write(d, TINY, W2)
        _raises(lambda: PackIndex.open(d).audit(), "layers.2.mixer.k_proj.weight")
    with tempfile.TemporaryDirectory() as d:
        _write(d, TINY, W)
        fn = os.path.join(d, "model-00002-of-00002.safetensors")
        with open(fn, "r+b") as f:
            f.truncate(os.path.getsize(fn) - 16)
        _raises(lambda: PackIndex.open(d), "model-00002-of-00002.safetensors", "truncated")
    with tempfile.TemporaryDirectory() as d:
        _write(d, TINY, W)
        idx = PackIndex.open(d)
        man = {}
        for fn in idx.files:
            man[fn] = {"sha256": hashlib.sha256(open(os.path.join(d, fn), "rb").read()).hexdigest()}
        assert set(idx.verify_sha256(man).values()) == {"ok"}
        man["model-00001-of-00002.safetensors"]["sha256"] = "0" * 64
        _raises(lambda: idx.verify_sha256(man), "model-00001-of-00002.safetensors")


def test_lightning_real_headers_audit():
    with open(os.path.join(FIX, "lightning-30b-a3b.json")) as f:
        cfg = config_from_dict(json.load(f), "lightning")
    with gzip.open(os.path.join(FIX, "lightning-30b-a3b-headers.json.gz"), "rt") as f:
        hdr = json.load(f)
    assert len(hdr) == 6513 and sum(k.startswith("mtp.") for k in hdr) == 270
    DS = {"BF16": 2, "F32": 4}
    tensors = {}
    for k, (dt, shape) in hdr.items():
        n = canonical(k)
        nb = DS[dt]
        for s in shape:
            nb *= s
        tensors[n] = TensorInfo(n, k, "", dt, tuple(shape), 0, nb)
    idx = PackIndex("lightning", cfg, tensors, {})
    idx.has_mtp = True
    idx.audit()
    assert {n for n, _ in expected(cfg)} == set(tensors)
    db = idx.decode_bytes()
    # 23 MoE layers x 6 experts x (1856 x 2688 x 2 x 2 bytes) of routed experts a token
    assert db["parts"]["experts"] == 23 * 6 * 1856 * 2688 * 4
    assert 6.0e9 < db["total"] < 7.5e9, db["total"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all nemotron_h pack tests passed")
