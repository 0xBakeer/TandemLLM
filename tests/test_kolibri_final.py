"""`tools/kolibri_final.py` on the tiny random Kolibri: the pooled gate's streams agree with the
build's own gate on the all-NVFP4 variant, the candidates differ where they should, the bytes a
token reads order the candidates, the written final set re-reads to the gated variant exactly
(sha256 in the manifest, both formats in one layer file), and the greedy check decodes from it."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from tests.test_kolibri_quant import CFG, _corpus, fp8_of, tiny_tensors, write_ckpt  # noqa: E402
from tools import kolibri_final as kf  # noqa: E402
from tools import kolibri_quant as kq  # noqa: E402
from tools import kolibri_ref as kr  # noqa: E402


def _build(d, target="fp8"):
    t = tiny_tensors()
    write_ckpt(os.path.join(d, "bf16"), t)
    write_ckpt(os.path.join(d, "fp8"), fp8_of(t))
    _corpus(os.path.join(d, "corpus"))
    ap = argparse.Namespace(
        bf16=os.path.join(d, "bf16"), bf16_repo=None, local=None, fp8=os.path.join(d, "fp8"), fp8_repo=None,
        fp8_local=None, fetch_workers=1, device="cpu", corpus=os.path.join(d, "corpus"),
        out=os.path.join(d, "set"), work=os.path.join(d, "work"), layers="all", calib_tokens=20000,
        mix="code:0.25,en:0.30,de:0.30,chat:0.15", gate_tokens=300, eval_tokens=0, gptq_min=1024,
        clip_min=64, expert_batch=4, damp=0.01, target=target, resume=False, force_head=False, greedy=False,
        greedy_tokens=4)
    kq.cmd_build(ap)
    return ap


def _chat(d):
    rng = np.random.default_rng(3)
    seqs, vl = [], []
    for i in range(3):
        ids = rng.integers(0, CFG["vocab_size"], 40).tolist()
        seqs.append({"name": f"gen-p{i:02d}-de-off", "ids": ids})
        vl.append({"name": f"gen-p{i:02d}-de-off", "gen_start": 12})
    seqs.append({"name": "raw-de", "ids": [1, 2, 3]})
    os.makedirs(d, exist_ok=True)
    json.dump(seqs, open(os.path.join(d, "seqs.json"), "w"))
    torch.save(vl, os.path.join(d, "vllm.pt"))


def test_final_gate_write_and_reread():
    torch.manual_seed(11)
    with tempfile.TemporaryDirectory() as d:
        ap = _build(d, target="bf16")   # the build gate's nvfp4 stream is then bit for bit this gate's nv stream
        _chat(os.path.join(d, "chat"))
        # eval texts: 2 chunks of 128 for en only (code/de have the held-out text alone)
        np.save(os.path.join(d, "corpus", "eval-en.npy"), np.random.default_rng(4).integers(0, CFG["vocab_size"], 260).astype(np.int32))
        out = os.path.join(d, "final")
        cmd = [sys.executable, os.path.join(os.path.dirname(HERE), "tools", "kolibri_final.py"),
               "--bf16", ap.bf16, "--fp8", ap.fp8, "--device", "cpu", "--corpus", ap.corpus,
               "--chat", os.path.join(d, "chat"), "--eval-tokens", "128", "--eval-chunks", "2",
               "--set", ap.out, "--out", out, "--work", os.path.join(d, "kfw"),
               "--variants", "mix,nv,kv8,full8,kvfull8,nvsh8", "--choose", "kvfull8", "--greedy", "--greedy-tokens", "3"]
        subprocess.run(cmd, check=True)
        s = json.load(open(os.path.join(out, "summary.json")))
        g = s["gate"]
        # the all-NVFP4 stream is the build's own gate stream: same NLL on the held-out text
        b = json.load(open(os.path.join(ap.out, "summary.json")))["gate"]["heldout-prose"]
        assert abs(g["nv"]["domains"]["en"]["nll"] - g["nv"]["domains"]["en"]["nll"]) == 0
        assert abs(g["nv"]["texts"]["heldout-prose"]["delta_vs_fp8"] - b["delta_nvfp4_vs_fp8"]) < 1e-4
        assert abs(g["nv"]["texts"]["heldout-prose"]["delta_vs_bf16"] - b["delta_nvfp4_vs_bf16"]) < 1e-4
        # domains, tokens, SE present; chat scored on generated positions only
        assert set(g["mix"]["domains"]) == {"en", "code", "de", "chat"}
        assert g["mix"]["domains"]["chat"]["tokens"] == 3 * (40 - 12)
        assert g["mix"]["domains"]["en"]["tokens"] == 299 + 127 + 127
        assert g["mix"]["domains"]["en"]["se_vs_fp8"] > 0
        # the mix with everything of attention in FP8 differs from all-NVFP4; kv8 sits between in bytes
        assert g["mix"]["domains"]["en"]["nll"] != g["nv"]["domains"]["en"]["nll"]
        bp = s["bytes_per_token"]
        assert bp["nv"]["total"] < bp["kv8"]["total"] < bp["kvfull8"]["total"] < bp["mix"]["total"]
        assert bp["nv"]["total"] < bp["nvsh8"]["total"]
        for v in s["variants"]:
            assert set(s["verdict"][v]) >= {"pass", "swap_ok", "worst_domain_delta", "crash"}
        # the written set: manifest, sha256, both formats in a file, and it re-reads to the gated stream
        m = json.load(open(os.path.join(out, "manifest.json")))
        assert m["variant"] == "kvfull8"
        c = CFG
        for L in range(c["num_hidden_layers"]):
            lm = m["layers"][str(L)]
            p = os.path.join(out, lm["file"])
            assert kf.sha256_file(p) == lm["sha256"]
            full = c["layer_types"][L] == "full_attention"
            assert lm["formats"]["self_attn.k_proj"] == "fp8_e4m3_block128"
            assert lm["formats"]["self_attn.q_proj"] == ("fp8_e4m3_block128" if full else "nvfp4")
            assert lm["formats"]["mlp.shared_experts.gate_proj"] == "nvfp4"
        assert kf.sha256_file(os.path.join(out, "outside.safetensors")) == m["outside"]["sha256"]
        # re-stream the held-out text through the written files and compare with the gate's NLL
        name = "heldout-prose"
        ids = torch.from_numpy(np.load(os.path.join(ap.corpus, "heldout-prose.npy")).astype(np.int64))
        from safetensors import safe_open
        with safe_open(os.path.join(out, "outside.safetensors"), framework="pt") as f:
            emb, nw = f.get_tensor("embed_tokens.weight").float(), f.get_tensor("norm.weight").float()
            hq = (f.get_tensor("lm_head.weight"), f.get_tensor("lm_head.weight_scale"))
        h = emb[ids]
        for L in range(c["num_hidden_layers"]):
            lw = kq.read_layer_file(os.path.join(out, f"layers/{L}.safetensors"), L, c, "cpu", packed=False)
            h, _ = kr.layer_forward(h, lw, c, L)
        nll = kf.per_token(kq.final_logits(h, nw, hq, c["rms_norm_eps"]), ids)["nll"]
        saved = torch.load(os.path.join(out, "gate", name + ".pt"))["kvfull8"]["nll"]
        assert torch.allclose(nll, saved, atol=1e-4), (nll.mean(), saved.mean())
        assert len(s["greedy"]["rows"]) == len(kq.GREEDY_PROMPTS) and s["greedy"]["reference"] == "fp8"


def test_block_se_and_verdict():
    d = [torch.linspace(0, 1, 640)]
    se, nb = kf.block_se(d, 64)
    assert nb == 10 and se > 0
    ds = {"domains": {"en": {"delta_vs_fp8": 0.03}, "de": {"delta_vs_fp8": 0.045}},
          "texts": {"a": {"delta_vs_fp8": 0.06}, "b": {"delta_vs_fp8": 0.02}}}
    v = kf.verdict(ds)
    assert v["pass"] and not v["swap_ok"] and v["worst_domain"] == "de"
    ds["texts"]["a"]["delta_vs_fp8"] = 0.2
    assert kf.verdict(ds)["crash"] and not kf.verdict(ds)["pass"]
