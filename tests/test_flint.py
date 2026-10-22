"""Flint on a tiny random model: the config reader, the reference forward against a plain fp32 implementation
of the published description, the kernels against their torch twins (attention with sinks, the sliding
window on a ring, the router, the prompt attention), and the board engine: its rows against the reference,
a verify block's rows against the decode step's bit for bit, commit of a moved path, the chunked prefill,
a context past the ring's size. Kernel and engine tests need Triton on a GPU, or TRITON_INTERPRET=1."""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flint import kernels as K  # noqa: E402
from engine.models.flint.config import config_from_dict  # noqa: E402
from engine.models.flint.ref import FlintRef, chain  # noqa: E402

GPU = torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET")
TRITON = K.HAVE_TRITON and (GPU or bool(os.environ.get("TRITON_INTERPRET")))
DEV = "cuda" if GPU else "cpu"
TYPES = ["full_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention",
         "sliding_attention"]
TINY = {
    "architectures": ["FlintForCausalLM"], "model_type": "flint", "attention_bias": False, "bos_token_id": 1,
    "eos_token_id": [2, 3], "pad_token_id": 2, "decoder_sparse_step": 1, "head_dim": 64, "hidden_act": "silu",
    "hidden_size": 256, "intermediate_size": 512, "layer_types": TYPES, "max_position_embeddings": 4096,
    "mlp_only_layers": [0, 1], "moe_intermediate_size": 128, "norm_topk_prob": True, "num_attention_heads": 4,
    "num_experts": 8, "num_experts_per_tok": 3, "num_hidden_layers": 6, "num_key_value_heads": 2,
    "no_rope_layer_indices": [0, 1, 4], "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
    "router_score_function": "sigmoid", "shared_expert_intermediate_size": 256, "sliding_window": 9,
    "softmax_type": "learnable", "tie_word_embeddings": False, "torch_dtype": "bfloat16", "use_sliding_window": True,
    "vocab_size": 256, "output_router_logits": False, "router_aux_loss_coef": 0.0005, "use_cache": True,
    "initializer_range": 0.02, "attention_dropout": 0.0, "transformers_version": "5.12.1",
}


def _random_weights(cfg, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    H, D, Hq, Hk = cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads

    def r(*s, scale=None):
        return torch.randn(*s, generator=g) * (scale if scale is not None else 1 / math.sqrt(s[-1]))

    def nw(n):
        return 1 + 0.1 * torch.randn(n, generator=g)
    W = {"embed_tokens.weight": r(cfg.vocab_size, H, scale=1.0), "lm_head.weight": r(cfg.vocab_size, H),
         "norm.weight": nw(H)}
    for li in range(cfg.num_hidden_layers):
        p = f"layers.{li}."
        W[p + "input_layernorm.weight"], W[p + "post_attention_layernorm.weight"] = nw(H), nw(H)
        a = p + "self_attn."
        W[a + "q_proj.weight"], W[a + "k_proj.weight"] = r(Hq * D, H), r(Hk * D, H)
        W[a + "v_proj.weight"], W[a + "o_proj.weight"] = r(Hk * D, H), r(H, Hq * D)
        W[a + "q_norm.weight"], W[a + "k_norm.weight"] = nw(D), nw(D)
        W[a + "softmax_offset"] = torch.randn(Hq, generator=g) * 2
        m = p + "mlp."
        if cfg.is_moe(li):
            I, S = cfg.moe_intermediate_size, cfg.shared_expert_intermediate_size
            W[m + "gate.weight"] = r(cfg.num_experts, H, scale=0.5 / math.sqrt(H))
            for e in range(cfg.num_experts):
                q = m + f"experts.{e}."
                W[q + "gate_proj.weight"], W[q + "up_proj.weight"], W[q + "down_proj.weight"] = r(I, H), r(I, H), r(H, I)
            q = m + "shared_expert."
            W[q + "gate_proj.weight"], W[q + "up_proj.weight"], W[q + "down_proj.weight"] = r(S, H), r(S, H), r(H, S)
        else:
            I = cfg.intermediate_size
            W[m + "gate_proj.weight"], W[m + "up_proj.weight"], W[m + "down_proj.weight"] = r(I, H), r(I, H), r(H, I)
    return W


def plain_forward(raw: dict, W: dict, ids: torch.Tensor) -> torch.Tensor:
    """The published description in plain fp32, a whole sequence at once (masks, no caches)."""
    c = raw
    T, D, Hq, Hk = ids.numel(), c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
    eps = c["rms_norm_eps"]

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    def glu(x, p):
        return (F.silu(x @ W[p + "gate_proj.weight"].T) * (x @ W[p + "up_proj.weight"].T)) @ W[p + "down_proj.weight"].T
    pos = torch.arange(T)
    inv = 1.0 / (c["rope_theta"] ** (torch.arange(0, D, 2).float() / D))
    f = pos[:, None].float() * inv[None]
    cos, sin = torch.cat([f, f], -1).cos(), torch.cat([f, f], -1).sin()

    def rot(x):
        return torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1)
    h = W["embed_tokens.weight"][ids]
    for li in range(c["num_hidden_layers"]):
        p = f"layers.{li}."
        a = p + "self_attn."
        x = rms(h, W[p + "input_layernorm.weight"])
        q = rms((x @ W[a + "q_proj.weight"].T).view(T, Hq, D), W[a + "q_norm.weight"])
        k = rms((x @ W[a + "k_proj.weight"].T).view(T, Hk, D), W[a + "k_norm.weight"])
        v = (x @ W[a + "v_proj.weight"].T).view(T, Hk, D)
        if li not in c["no_rope_layer_indices"]:
            q, k = q * cos[:, None] + rot(q) * sin[:, None], k * cos[:, None] + rot(k) * sin[:, None]
        k, v = k.repeat_interleave(Hq // Hk, 1), v.repeat_interleave(Hq // Hk, 1)
        s = torch.einsum("thd,shd->hts", q, k) / math.sqrt(D)
        allow = pos[None, :] <= pos[:, None]
        if c["layer_types"][li] == "sliding_attention":
            allow = allow & (pos[:, None] - pos[None, :] < c["sliding_window"])
        s = s.masked_fill(~allow[None], float("-inf"))
        s = torch.cat([s, W[a + "softmax_offset"].view(Hq, 1, 1).expand(Hq, T, 1)], -1)
        o = torch.einsum("hts,shd->thd", torch.softmax(s, -1)[..., :T], v).reshape(T, Hq * D)
        h = h + o @ W[a + "o_proj.weight"].T
        x = rms(h, W[p + "post_attention_layernorm.weight"])
        if li in c["mlp_only_layers"]:
            h = h + glu(x, p + "mlp.")
        else:
            sc = torch.sigmoid(x @ W[p + "mlp.gate.weight"].T)
            ids_ = torch.topk(sc, c["num_experts_per_tok"], -1).indices
            wt = sc.gather(1, ids_)
            wt = wt / (wt.sum(-1, keepdim=True) + 1e-20)
            y = torch.zeros_like(x)
            for t in range(T):
                for j in range(ids_.shape[1]):
                    y[t] += wt[t, j] * glu(x[t:t + 1], p + f"mlp.experts.{int(ids_[t, j])}.")[0]
            h = h + y + glu(x, p + "mlp.shared_expert.")
    return rms(h, W["norm.weight"]) @ W["lm_head.weight"].T


def test_config():
    cfg = config_from_dict(TINY, "tiny")
    assert cfg.sliding_layers == [2, 3, 5] and cfg.full_layers == [0, 1, 4] and cfg.moe_layers == [2, 3, 4, 5]
    assert not cfg.has_rope(4) and cfg.has_rope(2) and cfg.eos_token_ids == [2, 3] and cfg.route_scale == 1.0
    assert config_from_dict({**TINY, "quantization_config": {"expert_interm_div": 128}}, "tiny").route_scale == 128.0
    for bad in ({"softmax_type": "off"}, {"router_score_function": "softmax"}, {"new_field": 1},
                {"layer_types": TYPES[:-1]}):
        try:
            config_from_dict({**TINY, **bad}, "tiny")
        except ValueError:
            continue
        raise AssertionError(f"{bad} was accepted")


def test_ref_equals_plain_description():
    """The reference (blocks, caches, a tree) equals the plain fp32 forward, past the window."""
    cfg = config_from_dict(TINY, "tiny")
    W = _random_weights(cfg)
    ids = torch.randint(4, 256, (40,), generator=torch.Generator().manual_seed(1))
    want = plain_forward(TINY, W, ids)
    ref = FlintRef(cfg, W)
    st = ref.new_state(64)
    got = torch.cat([ref.prefill(ids[:23], st), ref.prefill(ids[23:30], st)] +
                    [ref.decode(int(t), st) for t in ids[30:]])
    assert (got - want).abs().max() < 1e-4, (got - want).abs().max()
    # a tree block: rows 1 and 2 both children of row 0; row 2's logits equal the chain [0, 2]'s
    st2 = ref.new_state(64)
    ref.prefill(ids[:30], st2)
    lt = ref.rows(torch.tensor([int(ids[30]), 7, int(ids[31])]), [-1, 0, 0], st2)
    assert (lt[2] - want[31]).abs().max() < 1e-4
    ref.commit(st2, [0, 2])
    assert st2.length == 32
    assert (ref.decode(int(ids[32]), st2)[0] - want[32]).abs().max() < 1e-4


def test_kernels_twins():
    if not TRITON:
        print("skipped: Triton on a GPU, or TRITON_INTERPRET=1")
        return
    g = torch.Generator().manual_seed(3)
    M, Hq, Hk, D, cap = 5, 4, 2, 64, 1200
    sink = torch.randn(Hq, generator=g).to(DEV)
    for window, ring, L in ((0, 0, 700), (9, 128, 700), (513, 640, 1100)):
        rows = ring if window else cap
        kc = torch.randn(Hk, rows, D, generator=g).to(torch.bfloat16).to(DEV)
        vc = torch.randn(Hk, rows, D, generator=g).to(torch.bfloat16).to(DEV)
        q = torch.randn(M, Hq, D, generator=g).to(torch.bfloat16).to(DEV)
        parents = [-1, 0, 1, 0, 3]
        paths = []
        for i, p in enumerate(parents):
            paths.append([i] if p < 0 else paths[p] + [i])
        ctl = torch.tensor([[L + len(pa), L] for pa in paths], dtype=torch.int32)
        anc = torch.zeros(M, K.ANC, dtype=torch.int32)
        for i, pa in enumerate(paths):
            anc[i, :len(pa)] = torch.tensor([L + r for r in pa], dtype=torch.int32)
        ns = K.class_of(L + M, cap)
        got = K.attn_rows(q, kc, vc, ctl.to(DEV), anc.to(DEV), ns, sink, window=window, ring=ring).float()
        want = K.attn_rows_reference(q, kc, vc, ctl.to(DEV), anc.to(DEV), sink, window=window, ring=ring)
        assert (got - want).abs().max() < 2e-2, (window, (got - want).abs().max())
        one = K.attn_rows(q[2:3].contiguous(), kc, vc, ctl[2:3].to(DEV), anc[2:3].to(DEV), ns, sink, window=window,
                          ring=ring)
        assert torch.equal(one[0], got[2].to(torch.bfloat16)), "a row of a block differs from its lone call"
    logits = torch.randn(6, 8, generator=g).to(DEV)
    ids, w = K.router(logits, None, 3, 1e-20)
    ids2, w2 = K.router_reference(logits, None, 3, 1e-20)
    assert torch.equal(ids.cpu(), ids2.cpu()) and (w - w2).abs().max() < 1e-6
    ids3, w3 = K.router(logits, None, 3, 1e-20, scale=128.0)          # an EXL3 pack's expert scale
    assert torch.equal(ids3, ids) and torch.equal(w3, w * 128.0)


def test_prompt_attention_reference():
    g = torch.Generator().manual_seed(4)
    T, S, Hq, Hk, D = 7, 20, 4, 2, 64
    q, k, v = torch.randn(T, Hq, D, generator=g), torch.randn(Hk, S, D, generator=g), torch.randn(Hk, S, D, generator=g)
    sink = torch.randn(Hq, generator=g)
    for window in (0, 5):
        got = K.prompt_attention_reference(q, k, v, sink, window)
        for t in range(T):
            pos = S - T + t
            lo = max(0, pos - window + 1) if window else 0
            kk, vv = k[:, lo:pos + 1].repeat_interleave(2, 0), v[:, lo:pos + 1].repeat_interleave(2, 0)
            s = torch.einsum("hd,hnd->hn", q[t], kk) / math.sqrt(D)
            a = torch.softmax(torch.cat([s, sink[:, None]], -1), -1)[:, :-1]
            assert (torch.einsum("hn,hnd->hd", a, vv) - got[t]).abs().max() < 1e-5
        if GPU:
            o = K.prompt_attention(q.to(DEV, torch.bfloat16), k.to(DEV, torch.bfloat16), v.to(DEV, torch.bfloat16), 0,
                                   sink.to(DEV), window).float().cpu()
            assert (o - got).abs().max() < 3e-2, (window, (o - got).abs().max())


def _tiny_engine(max_len=512):
    from engine.models.flint.engine import FlintEngine
    from engine.models.lfm2_moe.load import ExpertsBF16
    cfg = config_from_dict(TINY, "tiny")
    W0 = _random_weights(cfg)
    W = {k: (v.float() if k.endswith("softmax_offset") else v.to(torch.bfloat16)).to(DEV) for k, v in W0.items()
         if ".experts." not in k}
    experts = {}
    for li in cfg.moe_layers:
        f = f"layers.{li}.mlp.experts."
        w13 = torch.stack([torch.cat([W0[f + f"{e}.gate_proj.weight"], W0[f + f"{e}.up_proj.weight"]])
                           for e in range(cfg.num_experts)])
        w2 = torch.stack([W0[f + f"{e}.down_proj.weight"] for e in range(cfg.num_experts)])
        experts[li] = ExpertsBF16(w13.to(torch.bfloat16).to(DEV).contiguous(), w2.to(torch.bfloat16).to(DEV).contiguous())
    for li in range(cfg.num_hidden_layers):
        p = f"layers.{li}."
        a = p + "self_attn."
        W[a + "qkv"] = torch.cat([W[a + f"{x}_proj.weight"] for x in ("q", "k", "v")]).contiguous()
        m = p + ("mlp.shared_expert." if cfg.is_moe(li) else "mlp.")
        W[m + "gate_up"] = torch.cat([W[m + "gate_proj.weight"], W[m + "up_proj.weight"]]).contiguous()
    return cfg, W0, FlintEngine(cfg, W, experts, device=DEV, max_len=max_len)


def _close(got, want):
    """bf16 against fp32: most rows within bf16 noise; a near-tie in the router (a gap of 2e-4 between the
    3rd and 4th sigmoid on this model) picks another expert for a row and its error spreads to the rows after
    it, so the bound is on the median row and on the greedy agreement, not on every row."""
    d = (got.float().cpu() - want).abs().max(-1).values
    agree = (got.float().cpu().argmax(-1) == want.argmax(-1)).float().mean().item()
    assert d.median() < 0.1 and agree >= 0.9, (d.median().item(), agree)


def test_engine_rows_against_reference_and_itself():
    """Decode past the ring (128 rows on W = 9): the engine's logits follow the plain forward; a verify block
    of the same tokens equals the decode steps bit for bit; a tree's moved path commits."""
    if not TRITON:
        print("skipped: Triton on a GPU, or TRITON_INTERPRET=1")
        return
    cfg, W0, eng = _tiny_engine()
    assert eng.ring == 128
    ids = torch.randint(4, 256, (170,), generator=torch.Generator().manual_seed(5))
    want = plain_forward(TINY, W0, ids)
    st = eng.new_state(512)
    lg = eng.prefill(ids[:150], st, head="all")
    _close(lg, want[:150])
    L = st.length
    rows = []
    for t in ids[150:166]:
        eng.stage(int(t), st)
        rows.append(eng.step(st)[0].clone())
        st.length += 1
    dec = torch.stack(rows)
    _close(dec, want[150:166])
    # the same 16 tokens as one chain block from length L
    st.length = L
    eng.stage_rows([int(t) for t in ids[150:166]], chain(16), st)
    blk = eng.rows_step(st, 16)
    assert torch.equal(blk, dec), "a verify row differs from the decode step's"
    # a tree: rows 1 and 2 children of row 0; keep [0, 2] (moved), then decode on
    eng.stage_rows([int(ids[150]), 9, int(ids[151])], [-1, 0, 0], st)
    tl3 = eng.rows_step(st, 3)
    assert torch.equal(tl3[2], dec[1])
    eng.commit(st, [0, 2])
    assert st.length == L + 2
    rows = []
    for t in ids[152:160]:
        eng.stage(int(t), st)
        rows.append(eng.step(st)[0].clone())
        st.length += 1
    assert torch.equal(torch.stack(rows), dec[2:10]), "the committed moved path does not continue the chain"


def test_fp8_kv_cache():
    """FLINT_KV_FP8: the full layers' e4m3 cache. Prefill, decode and verify follow the bf16 cache's logits
    closely (e4m3 has 3 mantissa bits); a verify block equals the decode steps bit for bit on the coded cache;
    a moved path's codes and scales commit; the codes and scales are what kernels.quantize_kv makes."""
    if not GPU:
        print("skipped: the e4m3 cache runs on a GPU")
        return
    from engine.settings import SETTINGS
    import os as _os
    old = _os.environ.get("QWEN38_FLINT_KV_FP8")
    _os.environ["QWEN38_FLINT_KV_FP8"] = "1"
    try:
        assert SETTINGS.get("FLINT_KV_FP8") == "1"
        cfg, W0, eng = _tiny_engine()
    finally:
        if old is None:
            _os.environ.pop("QWEN38_FLINT_KV_FP8", None)
        else:
            _os.environ["QWEN38_FLINT_KV_FP8"] = old
    assert eng.kv_fp8
    _, _, ref = _tiny_engine()
    assert not ref.kv_fp8
    ids = torch.randint(4, 256, (170,), generator=torch.Generator().manual_seed(5))
    st, sr = eng.new_state(512), ref.new_state(512)
    full = [li for li in range(cfg.num_hidden_layers) if not cfg.is_sliding(li)]
    assert full and all(st.k[li].dtype == torch.uint8 and li in st.ks for li in full)
    lg = eng.prefill(ids[:150], st, head="all")
    lr = ref.prefill(ids[:150], sr, head="all")
    agree = (lg.argmax(-1) == lr.argmax(-1)).float().mean().item()
    assert agree >= 0.9, agree
    li = full[0]
    codes, sc = K.quantize_kv(sr.k[li][:, :150])
    same = (st.k[li][:, :150] == codes).float().mean().item()
    assert same >= 0.999 and torch.allclose(st.ks[li][:, :150], sc), same
    L = st.length
    rows = []
    for t in ids[150:166]:
        eng.stage(int(t), st)
        rows.append(eng.step(st)[0].clone())
        st.length += 1
    dec = torch.stack(rows)
    st.length = L
    eng.stage_rows([int(t) for t in ids[150:166]], chain(16), st)
    assert torch.equal(eng.rows_step(st, 16), dec), "a verify row differs from the decode step's on the e4m3 cache"
    eng.stage_rows([int(ids[150]), 9, int(ids[151])], [-1, 0, 0], st)
    eng.rows_step(st, 3)
    eng.commit(st, [0, 2])
    rows = []
    for t in ids[152:160]:
        eng.stage(int(t), st)
        rows.append(eng.step(st)[0].clone())
        st.length += 1
    assert torch.equal(torch.stack(rows), dec[2:10]), "the committed moved path does not continue the chain"
    return f"prefill top-1 agreement with the bf16 cache {agree:.3f}"


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
