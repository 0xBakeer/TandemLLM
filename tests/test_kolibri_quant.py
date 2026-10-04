"""The Kolibri-1 NVFP4 build on a tiny random Kolibri, CPU only, in seconds.

  * `kolibri_ref.layer_forward` (weights resident, the quantiser's forward) gives the same residual
    stream as the checkpoint-reading `attention` + `moe`.
  * the batched GPTQ: a matrix gets the same codes alone as inside a batch; with an identity second
    moment it is the clip search byte for byte; with a correlated one it leaves less output error.
  * the floor rule picks GPTQ, the own-statistics clip and the pooled clip at its boundaries.
  * `build` end to end from a BF16 and an FP8 checkpoint: every layer file, the outside tensors,
    the three-stream gate, the greedy check, and a resumed run that reproduces the summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from tools import kolibri_nvfp4 as kn  # noqa: E402
from tools import kolibri_quant as kq  # noqa: E402
from tools import kolibri_ref as kr  # noqa: E402

CFG = {
    "hidden_size": 64, "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
    "head_dim": 32, "rms_norm_eps": 1e-6, "vocab_size": 96, "rope_theta": 10000.0,
    "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32, "norm_topk_prob": False, "sliding_window": 5,
    "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
}


def tiny_tensors(c=CFG, seed=0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    H, A, KV = c["hidden_size"], c["num_attention_heads"] * c["head_dim"], c["num_key_value_heads"] * c["head_dim"]
    Fi, Fs, E, V = c["moe_intermediate_size"], c["shared_expert_intermediate_size"], c["num_experts"], c["vocab_size"]

    def r(*s, scale=0.08):
        return (torch.randn(*s, generator=g) * scale).to(torch.bfloat16)

    def nrm(n):
        return (1.0 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)
    t = {"model.embed_tokens.weight": r(V, H, scale=1.0), "lm_head.weight": r(V, H, scale=0.2),
         "model.norm.weight": nrm(H)}
    for L in range(c["num_hidden_layers"]):
        p = f"model.layers.{L}."
        for n in ("input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm"):
            t[p + n + ".weight"] = nrm(H)
        t[p + "self_attn.q_proj.weight"] = r(A, H)
        t[p + "self_attn.k_proj.weight"] = r(KV, H)
        t[p + "self_attn.v_proj.weight"] = r(KV, H)
        t[p + "self_attn.o_proj.weight"] = r(H, A)
        t[p + "self_attn.q_norm.weight"] = nrm(c["head_dim"])
        t[p + "self_attn.k_norm.weight"] = nrm(c["head_dim"])
        t[p + "mlp.gate.weight"] = r(E, H, scale=0.3)
        t[p + "moe.router.expert_bias"] = r(E, scale=0.1)
        for e in range(E):
            t[p + f"mlp.experts.{e}.gate_proj.weight"] = r(Fi, H)
            t[p + f"mlp.experts.{e}.up_proj.weight"] = r(Fi, H)
            t[p + f"mlp.experts.{e}.down_proj.weight"] = r(H, Fi)
        t[p + "mlp.shared_experts.gate_proj.weight"] = r(Fs, H)
        t[p + "mlp.shared_experts.up_proj.weight"] = r(Fs, H)
        t[p + "mlp.shared_experts.down_proj.weight"] = r(H, Fs)
    return t


def fp8_of(t: dict) -> dict:
    """The FP8 release's layout: every projection as e4m3 codes + an fp32 scale per 128x128 block."""
    out = {}
    for k, v in t.items():
        if k.endswith("_proj.weight") and v.dim() == 2:
            N, K = v.shape
            nb, kb = -(-N // 128), -(-K // 128)
            s = torch.empty(nb, kb)
            codes = torch.empty(N, K, dtype=torch.float8_e4m3fn)
            for i in range(nb):
                for j in range(kb):
                    blk = v[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128].float()
                    s[i, j] = blk.abs().max().clamp_min(1e-12) / 448.0
                    codes[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = (blk / s[i, j]).to(torch.float8_e4m3fn)
            out[k] = codes
            out[k[: -len("weight")] + "weight_scale_inv"] = s
        else:
            out[k] = v
    return out


def write_ckpt(d: str, t: dict, c=CFG) -> None:
    os.makedirs(d, exist_ok=True)
    keys = sorted(t)
    half = len(keys) // 2
    wm = {}
    for i, part in enumerate((keys[:half], keys[half:])):
        f = f"model-{i + 1:05d}-of-00002.safetensors"
        save_file({k: t[k].contiguous() for k in part}, os.path.join(d, f))
        wm.update({k: f for k in part})
    json.dump({"weight_map": wm}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    json.dump(c, open(os.path.join(d, "config.json"), "w"))
    from tokenizers import Tokenizer, models, pre_tokenizers
    tk = Tokenizer(models.WordLevel({"[UNK]": 0, "die": 1, "der": 2, "ist": 3, "the": 4}, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.save(os.path.join(d, "tokenizer.json"))


def test_layer_forward_equals_reading_forward():
    torch.manual_seed(1)
    with tempfile.TemporaryDirectory() as d:
        write_ckpt(d, tiny_tensors())
        ck = kr.Ckpt(d)
        c = CFG
        eps = c["rms_norm_eps"]
        ids = torch.randint(0, c["vocab_size"], (13,))
        r0 = ck.get("model.embed_tokens.weight", torch.float32)[ids]
        r1 = r0.clone()
        for L in range(c["num_hidden_layers"]):
            p = f"model.layers.{L}."
            sliding = c["layer_types"][L] == "sliding_attention"
            g = lambda n: ck.get(p + n + ".weight", torch.float32)  # noqa: E731
            r0 = r0 + kr.rms(kr.attention(kr.rms(r0, g("input_layernorm"), eps), L, c, ck, torch.float32, sliding),
                             g("post_attn_norm"), eps)
            y, _, _ = kr.moe(kr.rms(r0, g("post_attention_layernorm"), eps), L, c, ck, torch.float32)
            r0 = r0 + kr.rms(y, g("post_ffn_norm"), eps)
            lw = kr.load_layer(ck, L, c, "cpu", torch.float32)
            r1, _ = kr.layer_forward(r1, lw, c, L)
        assert torch.allclose(r0, r1, atol=1e-5, rtol=1e-5), (r0 - r1).abs().max()


def test_kv_decode_equals_full_forward():
    torch.manual_seed(2)
    with tempfile.TemporaryDirectory() as d:
        write_ckpt(d, tiny_tensors())
        ck = kr.Ckpt(d)
        c = CFG
        lws = [kr.load_layer(ck, L, c, "cpu", torch.float32) for L in range(3)]
        emb = ck.get("model.embed_tokens.weight", torch.float32)
        ids = torch.randint(0, c["vocab_size"], (11,))
        full = emb[ids]
        for L, lw in enumerate(lws):
            full, _ = kr.layer_forward(full, lw, c, L)
        kv = [{} for _ in lws]
        rows = []
        for t in range(11):
            r = emb[ids[t:t + 1]]
            for L, lw in enumerate(lws):
                r, _ = kr.layer_forward(r, lw, c, L, kv=kv[L], q_start=t)
            rows.append(r)
        assert torch.allclose(full, torch.cat(rows), atol=1e-5), (full - torch.cat(rows)).abs().max()


def test_gptq_batch_independent_and_identity_is_clip():
    torch.manual_seed(3)
    E, N, K = 3, 24, 64
    W = torch.randn(E, N, K) * 0.05
    X = torch.randn(E, 400, K) * torch.linspace(0.1, 2.0, K)
    H = torch.bmm(X.transpose(1, 2), X)
    cb, sb, s2b, _ = kn.gptq_batched(W, H, block=32)
    for e in range(E):
        c1, s1, s21, _ = kn.gptq_batched(W[e:e + 1], H[e:e + 1], block=32)
        assert torch.equal(c1[0], cb[e]) and torch.equal(s1[0].float(), sb[e].float())
    # identity second moment: no correlation, nothing to propagate -> the plain clip search
    I = torch.eye(K).expand(E, K, K).clone()
    cg, sg, _, _ = kn.gptq_batched(W, I, damp=0.0, block=32)
    cc, sc, _ = kn.clip_batched(W, torch.ones(E, K))
    assert torch.equal(cg, cc) and torch.equal(sg.float(), sc.float())
    # correlated inputs: GPTQ leaves less output error than the diagonal-weighted clip search
    Xc = torch.randn(E, 400, 8) @ torch.randn(E, 8, K) + 0.05 * torch.randn(E, 400, K)
    Hc = torch.bmm(Xc.transpose(1, 2), Xc)
    cq, sq, s2q, _ = kn.gptq_batched(W, Hc, block=32)
    ccl, scl, s2c = kn.clip_batched(W, torch.diagonal(Hc, dim1=1, dim2=2))
    eq = kn.rel_output_error(W, kn.dequant(cq, sq, s2q), Hc)
    ec = kn.rel_output_error(W, kn.dequant(ccl, scl, s2c), Hc)
    assert (eq < ec).all(), (eq, ec)


def test_stacked_rows_keep_their_own_scale_2():
    torch.manual_seed(4)
    a = torch.randn(1, 16, 32) * 0.01
    b = torch.randn(1, 16, 32) * 1.0
    s2 = kq._s2_rows(a, b)
    codes, sc, s2r = kn.clip_batched(torch.cat([a, b], 1), None, s2)
    ca, sa, s2a = kn.clip_batched(a, None)
    assert torch.equal(codes[:, :16], ca) and torch.equal(sc[:, :16].float(), sa.float())
    assert float(s2r[0, 0, 0]) == float(s2a[0, 0, 0]) and float(s2r[0, 16, 0]) != float(s2a[0, 0, 0])


def test_dequant_round_trip_and_head():
    torch.manual_seed(5)
    W = torch.randn(2, 8, 64)
    codes, sc, s2 = kn.clip_batched(W, None)
    D = kn.dequant(codes, sc, s2)
    rel = float((D - W).pow(2).sum() / W.pow(2).sum())
    assert rel < 0.02, rel
    w = torch.randn(40, 64).to(torch.bfloat16)
    hc, hs = kn.quant_head_e4m3(w)
    assert hc.dtype == torch.float8_e4m3fn and hs.shape == (40,)
    assert float((hc.float() * hs[:, None] - w.float()).abs().max()) < 0.1 * float(w.float().abs().max())


def _args(**kw):
    a = argparse.Namespace(damp=0.01, gptq_min=1024, clip_min=64, expert_batch=3)
    vars(a).update(kw)
    return a


def test_floor_rule_boundaries():
    torch.manual_seed(6)
    with tempfile.TemporaryDirectory() as d:
        write_ckpt(d, tiny_tensors())
        c = CFG
        lw = kr.load_layer(kr.Ckpt(d), 0, c, "cpu", torch.bfloat16)
        st = kq.Stats(c, "cpu")
        H = c["hidden_size"]
        st.n = torch.tensor([0, 63, 64, 1023, 1024, 5000, 1, 2])
        for e in range(c["num_experts"]):
            n = int(st.n[e])
            X = torch.randn(n, H)
            st.gu[e] = X.T @ X
            A = torch.randn(n, c["moe_intermediate_size"])
            st.d[e] = A.T @ A
        X = torch.randn(8000, H)
        st.attn = st.pool = X.T @ X
        O = torch.randn(8000, c["num_attention_heads"] * c["head_dim"])
        st.o = O.T @ O
        S = torch.randn(8000, c["shared_expert_intermediate_size"])
        st.sd = S.T @ S
        st.tokens = 8000
        tensors, rep = kq.quantise_layer(lw, st, c, 0, _args(), lambda s: None)
        assert [e["method"] for e in rep["experts"]] == ["clip-pool", "clip-pool", "clip-own", "clip-own",
                                                           "gptq", "gptq", "clip-pool", "clip-pool"]
        assert rep["experts"][0]["calib"][2] is None
        for e in range(c["num_experts"]):
            for p in kq.PROJ:
                for s in ("weight", "weight_scale", "weight_scale_2"):
                    assert f"layers.0.mlp.experts.{e}.{p}.{s}" in tensors
        assert rep["experts_summary"]["share_under_gptq_floor"] == 6 / 8


def _corpus(d, c=CFG, n=6000):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)
    for name in ("code", "en", "de", "chat"):
        np.save(os.path.join(d, f"calib-{name}.npy"), rng.integers(0, c["vocab_size"], n).astype(np.int32))
    for name in ("prose", "code", "de"):
        np.save(os.path.join(d, f"heldout-{name}.npy"), rng.integers(0, c["vocab_size"], 300).astype(np.int32))
    np.save(os.path.join(d, "eval-en.npy"), rng.integers(0, c["vocab_size"], 500).astype(np.int32))


def test_build_end_to_end_and_resume():
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory() as d:
        t = tiny_tensors()
        write_ckpt(os.path.join(d, "bf16"), t)
        write_ckpt(os.path.join(d, "fp8"), fp8_of(t))
        _corpus(os.path.join(d, "corpus"))
        ap = argparse.Namespace(
            bf16=os.path.join(d, "bf16"), bf16_repo=None, local=None, fp8=os.path.join(d, "fp8"), fp8_repo=None,
            fp8_local=None, fetch_workers=1, device="cpu", corpus=os.path.join(d, "corpus"),
            out=os.path.join(d, "out"), work=os.path.join(d, "work"), layers="all", calib_tokens=20000,
            mix="code:0.25,en:0.30,de:0.30,chat:0.15", gate_tokens=256, eval_tokens=400, gptq_min=1024,
            clip_min=64, expert_batch=4, damp=0.01, resume=False, force_head=False, greedy=True, greedy_tokens=5)
        kq.cmd_build(ap)
        s = json.load(open(os.path.join(d, "out", "summary.json")))
        for L in range(3):
            assert os.path.isfile(os.path.join(d, "out", "layers", f"{L}.safetensors"))
            rep = json.load(open(os.path.join(d, "out", "report", f"{L}.json")))
            assert rep["heldout"]["median_all"] < 0.05, rep["heldout"]["median_all"]
            assert sum(e["n"] for e in rep["experts"]) == 2 * rep["calib_tokens"]
        assert os.path.isfile(os.path.join(d, "out", "outside.safetensors"))
        g = s["gate"]["heldout-prose"]
        for k in ("nll_bf16", "nll_fp8", "nll_nvfp4", "delta_nvfp4_vs_bf16", "delta_nvfp4_vs_fp8",
                  "argmax_conf_nvfp4_vs_bf16", "delta_fp8_vs_bf16"):
            assert k in g, k
        assert abs(g["delta_nvfp4_vs_bf16"]) < 0.2
        assert len(s["greedy"]["rows"]) == len(kq.GREEDY_PROMPTS)
        assert all(r["tokens"] == 5 for r in s["greedy"]["rows"])
        # the greedy tokens are NVFP4's argmax; on a tiny random model BF16 agrees on most of them
        assert np.mean([r["bf16_argmax_share"] for r in s["greedy"]["rows"]]) > 0.5
        ap.resume = True
        kq.cmd_build(ap)
        s2 = json.load(open(os.path.join(d, "out", "summary.json")))
        assert s2["gate"] == s["gate"]


def test_mixgate_runs_and_nv_matches_build():
    torch.manual_seed(8)
    with tempfile.TemporaryDirectory() as d:
        t = tiny_tensors()
        write_ckpt(os.path.join(d, "bf16"), t)
        write_ckpt(os.path.join(d, "fp8"), fp8_of(t))
        _corpus(os.path.join(d, "corpus"))
        ap = argparse.Namespace(
            bf16=os.path.join(d, "bf16"), bf16_repo=None, local=None, fp8=os.path.join(d, "fp8"), fp8_repo=None,
            fp8_local=None, fetch_workers=1, device="cpu", corpus=os.path.join(d, "corpus"),
            out=os.path.join(d, "out"), work=os.path.join(d, "work"), layers="all", calib_tokens=20000,
            mix="code:0.25,en:0.30,de:0.30,chat:0.15", gate_tokens=256, eval_tokens=400, gptq_min=1024,
            clip_min=64, expert_batch=4, damp=0.01, resume=False, force_head=False, greedy=False, greedy_tokens=5)
        kq.cmd_build(ap)
        s = json.load(open(os.path.join(d, "out", "summary.json")))
        mp = os.path.join(d, "mix.json")
        kq.cmd_mixgate(argparse.Namespace(
            bf16=ap.bf16, bf16_repo=None, local=None, fp8=ap.fp8, fp8_repo=None, fp8_local=None, fetch_workers=1,
            device="cpu", set=ap.out, corpus=ap.corpus, work=os.path.join(d, "mixw"), gate_tokens=256,
            eval_tokens=400, out=mp))
        m = json.load(open(mp))
        for name, g in s["gate"].items():
            assert abs(m[name]["delta_nv_vs_bf16"] - g["delta_nvfp4_vs_bf16"]) < 1e-5
            assert abs(m[name]["delta_fp8_vs_bf16"] - g["delta_fp8_vs_bf16"]) < 1e-5
            assert "delta_nv_experts_only_head_bf16_vs_bf16" in m[name]


def test_gptq_act_order():
    torch.manual_seed(9)
    E, N, K = 2, 24, 64
    W = torch.randn(E, N, K) * 0.05
    # identity second moment: static groups and nothing to propagate -> the clip search's codes,
    # put back in stored order whatever the permutation was
    I = torch.eye(K).expand(E, K, K).clone() * torch.linspace(1, 2, K)
    ca, sa, _, _ = kn.gptq_batched(W, I, damp=0.0, block=32, act_order=True)
    cc, sc, _ = kn.clip_batched(W, torch.diagonal(I, dim1=1, dim2=2))
    assert torch.equal(ca, cc) and torch.equal(sa.float(), sc.float())
    # correlated inputs: act order still beats the clip search on output error
    X = torch.randn(E, 400, 8) @ torch.randn(E, 8, K) + 0.05 * torch.randn(E, 400, K) * torch.linspace(0.1, 3, K)
    Hc = torch.bmm(X.transpose(1, 2), X)
    cq, sq, s2q, _ = kn.gptq_batched(W, Hc, block=32, act_order=True)
    ccl, scl, s2c = kn.clip_batched(W, torch.diagonal(Hc, dim1=1, dim2=2))
    eq = kn.rel_output_error(W, kn.dequant(cq, sq, s2q), Hc)
    ec = kn.rel_output_error(W, kn.dequant(ccl, scl, s2c), Hc)
    assert (eq < ec).all(), (eq, ec)


def test_attn_requantise_recipes():
    torch.manual_seed(10)
    with tempfile.TemporaryDirectory() as d:
        t = tiny_tensors()
        write_ckpt(os.path.join(d, "bf16"), t)
        write_ckpt(os.path.join(d, "fp8"), fp8_of(t))
        _corpus(os.path.join(d, "corpus"))
        ap = argparse.Namespace(
            bf16=os.path.join(d, "bf16"), bf16_repo=None, local=None, fp8=os.path.join(d, "fp8"), fp8_repo=None,
            fp8_local=None, fetch_workers=1, device="cpu", corpus=os.path.join(d, "corpus"),
            out=os.path.join(d, "out"), work=os.path.join(d, "work"), layers="all", calib_tokens=20000,
            mix="code:0.25,en:0.30,de:0.30,chat:0.15", gate_tokens=256, eval_tokens=400, gptq_min=1024,
            clip_min=64, expert_batch=4, damp=0.01, resume=False, force_head=False, greedy=False, greedy_tokens=5)
        kq.cmd_build(ap)
        s = json.load(open(os.path.join(d, "out", "summary.json")))
        before = {L: open(os.path.join(d, "out", "layers", f"{L}.safetensors"), "rb").read() for L in range(3)}
        out2 = os.path.join(d, "out2")
        kq.cmd_attn(argparse.Namespace(
            bf16=ap.bf16, bf16_repo=None, local=None, fp8=ap.fp8, fp8_repo=None, fp8_local=None, fetch_workers=1,
            device="cpu", set=ap.out, corpus=ap.corpus, out=out2, work=os.path.join(d, "w2"),
            variants="base,fp8t,act,fp8t-act,fp8t-fine", calib_tokens=20000, mix=ap.mix, gate_tokens=256,
            eval_tokens=400, damp=0.01, greedy=True, greedy_tokens=4))
        a = json.load(open(os.path.join(out2, "summary.json")))
        # the base recipe is the build's own attention: same deltas on the same streams
        for name, g in s["gate"].items():
            assert abs(a["gate"][name]["delta_base_vs_bf16"] - g["delta_nvfp4_vs_bf16"]) < 1e-5, name
        for v in ("fp8t", "act", "fp8t-act", "fp8t-fine"):
            assert os.path.isfile(os.path.join(out2, "attn", v, "2.safetensors"))
        # the old set is untouched
        for L in range(3):
            assert open(os.path.join(d, "out", "layers", f"{L}.safetensors"), "rb").read() == before[L]
        if a["pass"]:
            assert os.path.isfile(os.path.join(out2, "layers", "0.safetensors")) and "greedy" in a


def test_build_toward_fp8_target():
    """`--target fp8`: the quantiser rounds toward the FP8 release's weights, the second moments
    come from its stream, the remainder/embedding/head are its tensors, and the gate still reports
    both references on identical tokens."""
    torch.manual_seed(9)
    with tempfile.TemporaryDirectory() as d:
        t = tiny_tensors()
        f = fp8_of(t)
        write_ckpt(os.path.join(d, "bf16"), t)
        write_ckpt(os.path.join(d, "fp8"), f)
        _corpus(os.path.join(d, "corpus"))
        ap = argparse.Namespace(
            bf16=os.path.join(d, "bf16"), bf16_repo=None, local=None, fp8=os.path.join(d, "fp8"), fp8_repo=None,
            fp8_local=None, fetch_workers=1, device="cpu", corpus=os.path.join(d, "corpus"),
            out=os.path.join(d, "out"), work=os.path.join(d, "work"), layers="all", calib_tokens=20000,
            mix="code:0.25,en:0.30,de:0.30,chat:0.15", gate_tokens=256, eval_tokens=400, gptq_min=1024,
            clip_min=64, expert_batch=4, damp=0.01, target="fp8", resume=False, force_head=False, greedy=True,
            greedy_tokens=4)
        kq.cmd_build(ap)
        s = json.load(open(os.path.join(d, "out", "summary.json")))
        assert s["gate_target"] == "fp8" and "gate_worst_delta_vs_bf16" in s and "gate_worst_delta_vs_fp8" in s
        g = s["gate"]["heldout-prose"]
        assert abs(g["delta_nvfp4_vs_fp8"]) < 0.2 and abs(g["delta_fp8_vs_bf16"]) < 0.2
        from safetensors import safe_open
        with safe_open(os.path.join(d, "out", "layers", "0.safetensors"), framework="pt") as sf:
            assert sf.metadata()["target"] == "fp8"
            # the remainder is the FP8 release's (bit-identical to BF16's here, as in the real release)
            assert torch.equal(sf.get_tensor("layers.0.mlp.gate.weight"), f["model.layers.0.mlp.gate.weight"])
            # the attention codes differ from a BF16-targeted build: the targets differ by the FP8 rounding
            q = sf.get_tensor("layers.0.self_attn.q_proj.weight")
        ap2 = argparse.Namespace(**{**vars(ap), "target": "bf16", "out": os.path.join(d, "out2"),
                                    "work": os.path.join(d, "work2"), "greedy": False})
        kq.cmd_build(ap2)
        with safe_open(os.path.join(d, "out2", "layers", "0.safetensors"), framework="pt") as sf:
            q2 = sf.get_tensor("layers.0.self_attn.q_proj.weight")
        assert not torch.equal(q, q2)
        assert len(s["greedy"]["rows"]) == len(kq.GREEDY_PROMPTS)
