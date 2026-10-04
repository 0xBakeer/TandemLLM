"""The pooled gate over the candidate splits of Kolibri-1's NVFP4 set, and the final set for the engine.

The first gate (`kolibri_quant.py build`) took the worst of six texts, three of them 1.5k to 2k
tokens, against BF16. On the saved log-probs the paired block standard error of a delta on such a
text is 0.01 nats and the spread between passages of one 8k text is 0.02 to 0.025, so a pass or
fail decided by 0.01 on a short text was a coin flip; and the texts had no chat or thinking,
although the served traffic is ChatML with `<think>` on. This gate pools per domain:

  * English prose, code, German prose: the held-out text plus three 8,192-token cuts of the wider
    eval text (`corpus2/`), about 26k tokens a domain;
  * chat: the 34 ChatML sequences of the vLLM cross-check (`tools/kolibri_vllm_check.py`) (vLLM's own generations through the
    template, thinking on and off, English, German, code), scored on the generated tokens only;

against the FP8 release as the primary reference (the policy as trained with FP8 QAT and served,
tech report App. I.4.1) with BF16 beside it, and prints the paired block standard error (64-token
blocks) next to every delta. The bar is +0.05 nats a token per domain against FP8; a candidate
replaces the safe choice (the mix) only with a margin of 0.01 in every domain; any single text
over +0.10 is a crash and fails the candidate whatever the pool says.

The candidates recombine files that exist: the routed experts of one NVFP4 set, its shared expert
or the FP8 release's, and attention per layer and projection from the set (NVFP4) or the FP8
release. Nothing is requantised here. The fastest candidate that passes with the margin is written
as `<out>/layers/L.safetensors` + `outside.safetensors` + `manifest.json`, with the format of every
projection in the manifest (the engine reads it), one sha256 per file, and the bytes a decode token
reads. A greedy check then decodes five prompts from the written files and scores them against the
FP8 reference.

    python tools/kolibri_final.py --bf16-repo Aleph-Alpha/Kolibri-1-BF16 --fp8-repo Aleph-Alpha/Kolibri-1 \
        --corpus /work/<prefix>/corpus2 --chat /work/<prefix>/crosscheck \
        --set /work/<prefix>/set-fp8t [--set-old /work/<prefix>/set --attn-old .../set-attn2/attn/fp8t] \
        --out /work/<prefix>/final [--variants mix,nv,...] [--choose NAME] [--greedy]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import kolibri_nvfp4 as kn  # noqa: E402
from tools.kolibri_quant import (ATTN, PROJ, final_logits, greedy_check, nv_dense, open_source,  # noqa: E402
                                 prefetch_order, read_layer_file)
from tools.kolibri_ref import LayerW, layer_forward, load_layer  # noqa: E402

ATT_KEYS = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}
SH_KEYS = {"sg": "gate_proj", "su": "up_proj", "sd": "down_proj"}
BAR, MARGIN, CRASH = 0.05, 0.01, 0.10


# ------------------------------------------------------------------ the candidates
def rule_all_nv(L, p, c):
    return "nv"


def rule_all_fp8(L, p, c):
    return "fp8"


def rule_kv8(L, p, c):
    return "fp8" if p in ("k_proj", "v_proj") else "nv"


def rule_v8(L, p, c):
    return "fp8" if p == "v_proj" else "nv"


def rule_full8(L, p, c):
    return "fp8" if c["layer_types"][L] == "full_attention" else "nv"


def rule_kvfull8(L, p, c):
    return "fp8" if (p in ("k_proj", "v_proj") or c["layer_types"][L] == "full_attention") else "nv"


def rule_tail8(L, p, c):
    return "fp8" if L >= c["num_hidden_layers"] - 10 else "nv"


# name: (which set's experts, attention rule, shared expert source)
VARIANTS = {
    "mix": ("new", rule_all_fp8, "nv"),          # the safe choice: NVFP4 experts, FP8 attention
    "nv": ("new", rule_all_nv, "nv"),            # all NVFP4
    "kv8": ("new", rule_kv8, "nv"),              # q, o NVFP4; k, v FP8
    "full8": ("new", rule_full8, "nv"),          # the 10 full-attention layers FP8, sliding NVFP4
    "kvfull8": ("new", rule_kvfull8, "nv"),      # both splits
    "nvsh8": ("new", rule_all_nv, "fp8"),        # all NVFP4 attention, shared expert FP8
    "kvfullsh8": ("new", rule_kvfull8, "fp8"),   # both splits + shared expert FP8
    "tail8": ("new", rule_tail8, "nv"),          # the last 10 layers' attention FP8
    "v8": ("new", rule_v8, "nv"),                # v alone FP8
    "old-mix": ("old", rule_all_fp8, "nv"),      # the first build (BF16 target), as a candidate
    "old-nv": ("old", rule_all_nv, "nv"),        # its all-NVFP4 form with the fp8t attention
}
DEFAULT_VARIANTS = "mix,nv,kv8,full8,kvfull8,nvsh8,kvfullsh8,tail8,v8,old-mix,old-nv"


# ------------------------------------------------------------------ the gate texts
def load_domains(corpus: str, chat: str | None, eval_tokens: int, eval_chunks: int):
    """{domain: [(name, ids LongTensor, mask BoolTensor over positions 0..T-2)]}."""
    dom: dict[str, list] = {}

    def add(d, name, ids, mask=None):
        ids = torch.as_tensor(np.asarray(ids).astype(np.int64))
        if mask is None:
            mask = torch.ones(ids.numel() - 1, dtype=torch.bool)
        dom.setdefault(d, []).append((name, ids, mask))

    for d, held, ev in (("en", "heldout-prose", "eval-en"), ("code", "heldout-code", "eval-code"),
                        ("de", "heldout-de", "eval-de")):
        p = os.path.join(corpus, held + ".npy")
        if os.path.isfile(p):
            add(d, held, np.load(p))
        p = os.path.join(corpus, ev + ".npy")
        if os.path.isfile(p):
            arr = np.load(p)
            for i in range(eval_chunks):
                piece = arr[i * eval_tokens:(i + 1) * eval_tokens]
                if piece.size >= min(256, eval_tokens // 2):
                    add(d, f"{ev}-{i}", piece)
    if chat:
        seqs = json.load(open(os.path.join(chat, "seqs.json")))
        gen = {}
        vp = os.path.join(chat, "vllm.pt")
        if os.path.isfile(vp):
            for s in torch.load(vp, weights_only=False):
                if s.get("gen_start") is not None:
                    gen[s["name"]] = int(s["gen_start"])
        for s in seqs:
            if not s["name"].startswith("gen"):
                continue
            ids = np.asarray(s["ids"])
            g0 = gen.get(s["name"], 0)
            mask = torch.zeros(ids.size - 1, dtype=torch.bool)
            mask[max(0, g0 - 1):] = True        # position t predicts token t+1; generated tokens are ids[g0:]
            add("chat", s["name"], ids, mask)
    return dom


# ------------------------------------------------------------------ statistics
def per_token(lg: torch.Tensor, ids: torch.Tensor) -> dict:
    """Per-position NLL, top-1 and the top-1/top-2 gap of one stream on one text."""
    tgt = ids[1:].to(lg.device)
    lp = torch.log_softmax(lg[:-1].float(), -1)
    nll = -lp.gather(1, tgt[:, None])[:, 0]
    t2 = lg[:-1].topk(2, -1)
    return {"nll": nll.cpu(), "top": t2.indices[:, 0].int().cpu(), "gap": (t2.values[:, 0] - t2.values[:, 1]).cpu()}


def block_se(deltas: list[torch.Tensor], block: int = 64) -> tuple[float, int]:
    """Standard error of the pooled mean from the means of consecutive `block`-token blocks (each
    text cut on its own; a block of fewer than block/2 tokens is dropped)."""
    means = []
    for d in deltas:
        for i in range(0, d.numel(), block):
            b = d[i:i + block]
            if b.numel() >= block // 2:
                means.append(float(b.mean()))
    if len(means) < 2:
        return float("nan"), len(means)
    m = torch.tensor(means)
    return float(m.std(unbiased=True) / math.sqrt(len(means))), len(means)


def domain_stats(stats: dict, domains: dict, streams: list[str], refs=("fp8", "bf16")) -> dict:
    """stats[name][stream] = per_token(...). Returns {stream: {domain: {...}, "texts": {...}}}."""
    out: dict = {}
    for v in streams:
        out[v] = {"domains": {}, "texts": {}}
        for d, texts in domains.items():
            row: dict = {"tokens": 0, "texts": len(texts)}
            for ref in refs:
                if ref == v or ref not in streams:
                    continue
                deltas, agree, nconf = [], 0, 0
                for name, ids, mask in texts:
                    a, b = stats[name][v], stats[name][ref]
                    dl = (a["nll"] - b["nll"])[mask]
                    deltas.append(dl)
                    conf = (b["gap"] >= 1.0) & mask
                    agree += int(((a["top"] == b["top"]) & conf).sum())
                    nconf += int(conf.sum())
                    out[v]["texts"].setdefault(name, {})[f"delta_vs_{ref}"] = float(dl.mean())
                    out[v]["texts"][name]["tokens"] = int(mask.sum())
                    out[v]["texts"][name][f"se_vs_{ref}"] = block_se([dl])[0]
                cat = torch.cat(deltas)
                se, nb = block_se(deltas)
                row[f"delta_vs_{ref}"] = float(cat.mean())
                row[f"se_vs_{ref}"] = se
                row[f"blocks_vs_{ref}"] = nb
                row[f"argmax_conf_vs_{ref}"] = agree / max(1, nconf)
                row["tokens"] = int(cat.numel())
            row["nll"] = float(torch.cat([stats[n][v]["nll"][m] for n, _, m in texts]).mean())
            out[v]["domains"][d] = row
    return out


def verdict(ds: dict, ref: str = "fp8", bar=BAR, margin=MARGIN, crash=CRASH) -> dict:
    doms = ds["domains"]
    worst_d = max(doms.values(), key=lambda r: r.get(f"delta_vs_{ref}", -9))
    worst = max(r[f"delta_vs_{ref}"] for r in doms.values())
    worst_text = max(t[f"delta_vs_{ref}"] for t in ds["texts"].values())
    return {"worst_domain_delta": worst, "worst_domain": [d for d, r in doms.items() if r is worst_d][0],
            "worst_text_delta": worst_text, "crash": worst_text > crash,
            "pass": worst <= bar and worst_text <= crash,
            "swap_ok": worst <= bar - margin and worst_text <= crash}


# ------------------------------------------------------------------ bytes a token reads
def nv_bytes(N, K) -> int:
    return N * K // 2 + N * K // 16 + 4


def fp8_bytes(N, K) -> int:
    return N * K + (-(-N // 128)) * (-(-K // 128)) * 4


def bytes_per_token(c: dict, rule, shared: str) -> dict:
    H, A = c["hidden_size"], c["num_attention_heads"] * c["head_dim"]
    KV = c["num_key_value_heads"] * c["head_dim"]
    Fi, Fs, E, k = c["moe_intermediate_size"], c["shared_expert_intermediate_size"], c["num_experts"], c["num_experts_per_tok"]
    shapes = {"q_proj": (A, H), "k_proj": (KV, H), "v_proj": (KV, H), "o_proj": (H, A)}
    attn = experts = sh = rest = 0
    for L in range(c["num_hidden_layers"]):
        for p, (N, K) in shapes.items():
            attn += nv_bytes(N, K) if rule(L, p, c) == "nv" else fp8_bytes(N, K)
        experts += k * (2 * nv_bytes(Fi, H) + nv_bytes(H, Fi))
        f = nv_bytes if shared == "nv" else fp8_bytes
        sh += 2 * f(Fs, H) + f(H, Fs)
        rest += E * H * 2 + E * 2 + 4 * H * 2 + 2 * c["head_dim"] * 2     # router, bias, norms (BF16)
    head = c["vocab_size"] * H + c["vocab_size"] * 4                        # e4m3 + fp32 row scale
    outside = head + H * 2 + H * 2                                           # one embedding row, final norm
    tot = attn + experts + sh + rest + outside
    return {"attention": attn, "experts": experts, "shared": sh, "router_norms": rest, "head_and_outside": outside,
            "total": tot, "total_gb": tot / 1e9, "ceiling_tok_s_at_273GBps": 273e9 / tot}


# ------------------------------------------------------------------ reading the sets
def attn_from_file(path: str, L: int, dev) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device=str(dev)) as f:
        out = {}
        for key, n in ATT_KEYS.items():
            b = f"layers.{L}.self_attn.{n}"
            out[key] = nv_dense(f.get_tensor(b + ".weight"), f.get_tensor(b + ".weight_scale"),
                                f.get_tensor(b + ".weight_scale_2"))
        return out


def local_copy(src: str, work: str) -> str:
    os.makedirs(work, exist_ok=True)
    dst = os.path.join(work, os.path.basename(src))
    shutil.copyfile(src, dst)
    return dst


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ writing the final set
def write_layer(L: int, c: dict, src_set: str, attn_nv_src: str | None, f8, rule, shared: str,
                dst: str) -> tuple[dict, str]:
    """One final layer file: everything of the source set's layer file except the projections the
    rule puts in FP8, which come from the release as e4m3 codes + fp32 `weight_scale_inv`.
    Returns (formats per projection, sha256)."""
    from safetensors.torch import load_file
    t = load_file(os.path.join(src_set, "layers", f"{L}.safetensors"))
    fmt: dict = {}
    if attn_nv_src is not None:
        t.update(load_file(os.path.join(attn_nv_src, f"{L}.safetensors")))
    p = f"model.layers.{L}."
    for n in ATTN:
        base = f"layers.{L}.self_attn.{n}"
        if rule(L, n, c) == "fp8":
            for s in ("weight", "weight_scale", "weight_scale_2"):
                t.pop(f"{base}.{s}", None)
            t[base + ".weight"] = f8.raw(p + f"self_attn.{n}.weight")
            t[base + ".weight_scale_inv"] = f8.raw(p + f"self_attn.{n}.weight_scale_inv").float()
            assert t[base + ".weight"].dtype == torch.float8_e4m3fn
            fmt[f"self_attn.{n}"] = "fp8_e4m3_block128"
        else:
            fmt[f"self_attn.{n}"] = "nvfp4"
    for n in PROJ:
        base = f"layers.{L}.mlp.shared_experts.{n}"
        if shared == "fp8":
            for s in ("weight", "weight_scale", "weight_scale_2"):
                t.pop(f"{base}.{s}", None)
            t[base + ".weight"] = f8.raw(p + f"mlp.shared_experts.{n}.weight")
            t[base + ".weight_scale_inv"] = f8.raw(p + f"mlp.shared_experts.{n}.weight_scale_inv").float()
            fmt[f"mlp.shared_experts.{n}"] = "fp8_e4m3_block128"
        else:
            fmt[f"mlp.shared_experts.{n}"] = "nvfp4"
    fmt["mlp.experts.*"] = "nvfp4"
    fmt["mlp.gate"] = "bf16"
    fmt["moe.router.expert_bias"] = "bf16"
    fmt["norms"] = "bf16"
    meta = {"format": "kolibri1-mixed", "layer": str(L), "nvfp4_group": "16",
            "fp8_block": "128x128, fp32 weight_scale_inv", "attention": ",".join(fmt[f"self_attn.{n}"] for n in ATTN),
            "shared_experts": fmt["mlp.shared_experts.gate_proj"]}
    save_file({k: v.contiguous() for k, v in t.items()}, dst, metadata=meta)
    return fmt, sha256_file(dst)


# ------------------------------------------------------------------ main
@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bf16", default=None)
    ap.add_argument("--bf16-repo", default=None)
    ap.add_argument("--local", default="/tmp/bf16")
    ap.add_argument("--fp8", default=None)
    ap.add_argument("--fp8-repo", default=None)
    ap.add_argument("--fp8-local", default="/tmp/fp8")
    ap.add_argument("--fetch-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--corpus", required=True, help="corpus2: heldout-*.npy and eval-*.npy")
    ap.add_argument("--chat", default=None, help="the cross-check directory (seqs.json, vllm.pt)")
    ap.add_argument("--eval-tokens", type=int, default=8192)
    ap.add_argument("--eval-chunks", type=int, default=3)
    ap.add_argument("--set", required=True, help="the NVFP4 set whose experts the candidates use")
    ap.add_argument("--set-old", default=None, help="a second set, for the old-* controls")
    ap.add_argument("--attn-old", default=None, help="per-layer NVFP4 attention files for old-nv")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/tmp/kf")
    ap.add_argument("--bar", type=float, default=BAR)
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--crash", type=float, default=CRASH)
    ap.add_argument("--choose", default=None, help="write this variant whatever the gate says")
    ap.add_argument("--no-gate", action="store_true", help="skip the streams (with --choose)")
    ap.add_argument("--no-write", action="store_true", help="gate only")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--greedy-tokens", type=int, default=300)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    logp = os.path.join(args.work, "final.log")
    logf = open(logp, "a")

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    pool = ThreadPoolExecutor(args.fetch_workers)
    bf = open_source(args.bf16, args.bf16_repo, args.local, pool)
    f8 = open_source(args.fp8, args.fp8_repo, args.fp8_local, pool)
    c = json.load(open(os.path.join(f8.dir, "config.json")))
    eps = c["rms_norm_eps"]
    nl = c["num_hidden_layers"]
    refs = {k: s for k, s in (("bf16", bf), ("fp8", f8)) if s is not None}
    prefetch_order(list(refs.values()), list(range(nl)))
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if args.set_old is None:
        variants = [v for v in variants if VARIANTS[v][0] != "old"]
    for v in variants:
        assert v in VARIANTS, v
    sets = {"new": args.set, "old": args.set_old}
    attn_nv = {"new": None, "old": args.attn_old}
    domains = load_domains(args.corpus, args.chat, args.eval_tokens, args.eval_chunks)
    texts = [(n, ids) for d in domains.values() for n, ids, _ in d]
    log(f"[final] set {args.set}; old {args.set_old}; variants {variants}; git {os.environ.get('GIT_SHA', '?')}")
    log("[final] domains " + json.dumps({d: [(n, int(i.numel()), int(m.sum())) for n, i, m in t] for d, t in domains.items()}))
    bpt = {v: bytes_per_token(c, VARIANTS[v][1], VARIANTS[v][2]) for v in variants}
    for v in variants:
        log(f"[bytes] {v}: {bpt[v]['total_gb']:.3f} GB a token, ceiling {bpt[v]['ceiling_tok_s_at_273GBps']:.0f} tok/s")

    summary: dict = {"variants": variants, "bytes_per_token": bpt, "bar": args.bar, "margin": args.margin,
                     "crash": args.crash, "domains": {d: [(n, int(i.numel()), int(m.sum())) for n, i, m in t]
                                                     for d, t in domains.items()}}
    chosen = args.choose
    if not args.no_gate:
        t0 = time.time()
        embs = {k: s.get("model.embed_tokens.weight", torch.float32, dev) for k, s in refs.items()}
        hs = {k: [embs[k][ids.to(dev)] for _, ids in texts] for k in refs}
        set_emb = {}
        for which, sd in sets.items():
            if sd is None:
                continue
            with safe_open(os.path.join(sd, "outside.safetensors"), framework="pt", device=str(dev)) as f:
                set_emb[which] = f.get_tensor("embed_tokens.weight").float()
        for v in variants:
            hs[v] = [set_emb[VARIANTS[v][0]][ids.to(dev)] for _, ids in texts]
        del embs, set_emb
        for s in refs.values():
            s.close_layer()
        for L in range(nl):
            tl = time.time()
            lw_ref = {k: load_layer(s, L, c, dev, torch.float32) for k, s in refs.items()}
            l8 = lw_ref["fp8"]
            ln = {}
            for which, sd in sets.items():
                if sd is None or not any(VARIANTS[v][0] == which for v in variants):
                    continue
                loc = local_copy(os.path.join(sd, "layers", f"{L}.safetensors"), os.path.join(args.work, which))
                ln[which] = read_layer_file(loc, L, c, dev, packed=False)
                if attn_nv[which]:
                    for key, val in attn_from_file(os.path.join(attn_nv[which], f"{L}.safetensors"), L, dev).items():
                        setattr(ln[which], key, val)
                os.remove(loc)
            lws = {}
            for v in variants:
                which, rule, shared = VARIANTS[v]
                t = dict(vars(ln[which]))
                for key, n in ATT_KEYS.items():
                    if rule(L, n, c) == "fp8":
                        t[key] = getattr(l8, key)
                if shared == "fp8":
                    for key in SH_KEYS:
                        t[key] = getattr(l8, key)
                lws[v] = LayerW(**t)
            for k in list(refs) + variants:
                lw = lw_ref[k] if k in refs else lws[k]
                for j in range(len(texts)):
                    hs[k][j], _ = layer_forward(hs[k][j], lw, c, L)
            del lw_ref, l8, ln, lws
            for s in refs.values():
                s.close_layer()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            if L % 5 == 4 or L == nl - 1:
                log(f"[gate] layer {L} done, {time.time() - tl:.0f} s this layer, {time.time() - t0:.0f} s in all; "
                    f"peak {torch.cuda.max_memory_allocated() / 2**30 if dev.type == 'cuda' else 0:.1f} GiB")
        # heads
        heads = {k: (s.get("model.norm.weight", torch.float32, dev), s.get("lm_head.weight", torch.float32, dev))
                 for k, s in refs.items()}
        for which, sd in sets.items():
            if sd is None or not any(VARIANTS[v][0] == which for v in variants):
                continue
            with safe_open(os.path.join(sd, "outside.safetensors"), framework="pt", device=str(dev)) as f:
                heads[which] = (f.get_tensor("norm.weight").float(),
                                (f.get_tensor("lm_head.weight"), f.get_tensor("lm_head.weight_scale")))
        stats: dict = {}
        os.makedirs(os.path.join(args.out, "gate"), exist_ok=True)
        for j, (name, ids) in enumerate(texts):
            stats[name] = {}
            for k in list(refs) + variants:
                nw, head = heads[k] if k in refs else heads[VARIANTS[k][0]]
                stats[name][k] = per_token(final_logits(hs[k][j], nw, head, eps), ids)
            torch.save({"ids": ids, **stats[name]}, os.path.join(args.out, "gate", f"{name}.pt"))
        del hs, heads
        ds = domain_stats(stats, domains, list(refs) + variants)
        ver = {v: verdict(ds[v], "fp8", args.bar, args.margin, args.crash) for v in variants}
        summary["gate"] = ds
        summary["verdict"] = ver
        summary["gate_seconds"] = time.time() - t0
        # the table
        doms = list(domains)
        log("[gate] delta vs FP8 (block SE) | vs BF16, per domain; tokens " + json.dumps({d: ds["fp8"]["domains"][d]["tokens"] for d in doms}))
        log("[gate] " + " ".join(f"{d:>22s}" for d in doms) + "   worst  pass swap  GB/tok")
        def cell(r, ref):
            a = r.get(f"delta_vs_{ref}")
            return "   ref   " if a is None else f"{a:+.4f}({r.get(f'se_vs_{ref}', float('nan')):.4f})"

        for v in [k for k in refs if k != "fp8"] + variants:
            cells = [cell(ds[v]["domains"][d], "fp8") + "|" + cell(ds[v]["domains"][d], "bf16")[:7] for d in doms]
            tail = (f"  {ver[v]['worst_domain_delta']:+.4f} {'yes' if ver[v]['pass'] else 'NO '}  "
                    f"{'yes' if ver[v]['swap_ok'] else 'no '}  {bpt[v]['total_gb']:.2f}") if v in ver else ""
            log(f"[gate] {v:>10s} " + " ".join(f"{x:>22s}" for x in cells) + tail)
        if chosen is None:
            ok = [v for v in variants if ver[v]["swap_ok"]]
            if ok:
                chosen = min(ok, key=lambda v: (bpt[v]["total"], v))
                log(f"[final] chosen {chosen}: the fewest bytes a token among those passing with the margin {ok}")
            else:
                passing = [v for v in variants if ver[v]["pass"] and VARIANTS[v][1] is rule_all_fp8]
                if passing:
                    chosen = min(passing, key=lambda v: ver[v]["worst_domain_delta"])
                    log(f"[final] no split passes with the margin; chosen the better mix {chosen} of {passing}")
                else:
                    chosen = "old-mix" if "old-mix" in variants else "mix"
                    log(f"[final] NOTHING passes the bar; writing {chosen} anyway, flagged")
        summary["chosen"] = chosen
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        shutil.copyfile(logp, os.path.join(args.out, "final.log"))
    if args.no_write or chosen is None:
        log("[final] gate only, nothing written")
        return

    # the final set
    which, rule, shared = VARIANTS[chosen]
    src_set = sets[which]
    t1 = time.time()
    os.makedirs(os.path.join(args.out, "layers"), exist_ok=True)
    wl = os.path.join(args.work, "final-layers")
    os.makedirs(wl, exist_ok=True)
    layers_m = {}
    for L in range(nl):
        loc = os.path.join(wl, f"{L}.safetensors")
        fmt, sha = write_layer(L, c, src_set, attn_nv[which], f8, rule, shared, loc)
        f8.close_layer()
        shutil.copyfile(loc, os.path.join(args.out, "layers", f"{L}.safetensors"))
        layers_m[str(L)] = {"file": f"layers/{L}.safetensors", "sha256": sha, "bytes": os.path.getsize(loc),
                            "layer_type": c["layer_types"][L], "formats": fmt}
        if L % 10 == 9:
            log(f"[write] layer {L}, {time.time() - t1:.0f} s")
    oloc = os.path.join(args.work, "outside.safetensors")
    shutil.copyfile(os.path.join(src_set, "outside.safetensors"), oloc)
    shutil.copyfile(oloc, os.path.join(args.out, "outside.safetensors"))
    with safe_open(oloc, framework="pt") as f:
        outside_m = {k: {"shape": list(f.get_slice(k).get_shape()), "dtype": str(f.get_slice(k).get_dtype())}
                     for k in f.keys()}
        outside_meta = f.metadata()
    manifest = {
        "model": "Aleph-Alpha/Kolibri-1", "written": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "code": os.environ.get("GIT_SHA", "?"), "variant": chosen,
        "variant_description": {"experts_set": src_set, "attention_rule": rule.__name__, "shared_experts": shared},
        "sources": {"experts_and_nvfp4": src_set, "nvfp4_attention_files": attn_nv[which],
                    "fp8": "Aleph-Alpha/Kolibri-1 (the FP8 release; e4m3 codes + fp32 weight_scale_inv copied as they are)"},
        "formats": {
            "nvfp4": {"weight": "uint8 [N, K/2], two e2m1 codes a byte, low nibble = even column; code = sign bit (8) | "
                                "index into [0, 0.5, 1, 1.5, 2, 3, 4, 6]",
                      "weight_scale": "float8_e4m3fn [N, K/16], one per group of 16 consecutive columns",
                      "weight_scale_2": "fp32 scalar per tensor",
                      "value": "e2m1(code) * float(weight_scale) * weight_scale_2, in fp32"},
            "fp8_e4m3_block128": {"weight": "float8_e4m3fn [N, K]",
                                  "weight_scale_inv": "fp32 [ceil(N/128), ceil(K/128)], one per 128x128 block "
                                                      "(row block i covers rows 128i..128i+127, same for columns)",
                                  "value": "float(weight) * weight_scale_inv[i, j], in fp32; the scales must stay fp32 "
                                           "(a BF16 cast changes the weights the gate measured)"},
            "e4m3_head": {"lm_head.weight": "float8_e4m3fn [V, H]", "lm_head.weight_scale": "fp32 [V], one per vocabulary row",
                          "logits": "fp32: x @ (float(weight) * scale[:, None]).T"},
            "bf16": "as stored",
        },
        "tensor_names": {
            "layer": "layers.{L}.self_attn.{q,k,v,o}_proj.*, layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.*, "
                     "layers.{L}.mlp.shared_experts.{gate,up,down}_proj.*, layers.{L}.mlp.gate.weight (BF16 router), "
                     "layers.{L}.moe.router.expert_bias (BF16), layers.{L}.{input_layernorm,post_attn_norm,"
                     "post_attention_layernorm,post_ffn_norm}.weight, layers.{L}.self_attn.{q_norm,k_norm}.weight",
            "outside": "embed_tokens.weight (BF16), norm.weight (BF16), lm_head.weight + lm_head.weight_scale (e4m3 head)"},
        "forward": "tools/kolibri_ref.py: sandwich norms; RoPE (neox, base 10000) on sliding layers only, window 513; "
                   "no position encoding on full layers; router top-6 on fp32 logits + expert_bias, weights "
                   "sigmoid(logit) unbiased, not renormalised; shared expert added; fp32 head",
        "layers": layers_m,
        "outside": {"file": "outside.safetensors", "sha256": sha256_file(oloc), "bytes": os.path.getsize(oloc),
                    "tensors": outside_m, "metadata": outside_meta},
        "bytes_per_token": bpt[chosen],
        "total_bytes": sum(m["bytes"] for m in layers_m.values()) + os.path.getsize(oloc),
        "gate": {"bar": args.bar, "margin": args.margin, "crash": args.crash,
                 "verdict": summary.get("verdict", {}).get(chosen),
                 "domains": summary.get("gate", {}).get(chosen, {}).get("domains")},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    log(f"[write] {chosen}: {manifest['total_bytes'] / 1e9:.2f} GB in {time.time() - t1:.0f} s -> {args.out}")
    summary["manifest"] = {"variant": chosen, "total_bytes": manifest["total_bytes"]}

    if args.greedy:
        with safe_open(oloc, framework="pt", device=str(dev)) as f:
            emb = f.get_tensor("embed_tokens.weight")
            hq = (f.get_tensor("lm_head.weight"), f.get_tensor("lm_head.weight_scale"))
        ga = argparse.Namespace(greedy_tokens=args.greedy_tokens)
        # `greedy_check` reads `<work>/layers/L.safetensors` and scores against the checkpoint it is
        # given: the FP8 release, the reference the gate used
        gw = os.path.join(args.work, "greedy")
        os.makedirs(gw, exist_ok=True)
        if os.path.lexists(os.path.join(gw, "layers")):
            os.remove(os.path.join(gw, "layers"))
        os.symlink(wl, os.path.join(gw, "layers"))
        summary["greedy"] = greedy_check(ga, c, list(range(nl)), gw, emb, hq, f8, dev, log)
        summary["greedy"]["reference"] = "fp8"
        manifest["greedy"] = summary["greedy"]
        with open(os.path.join(args.out, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    shutil.copyfile(logp, os.path.join(args.out, "final.log"))
    pool.shutdown(wait=False, cancel_futures=True)
    log("[final] done")


if __name__ == "__main__":
    main()
