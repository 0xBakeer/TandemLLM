"""The Kolibri-1 engine on the GPU: correctness against the reference forward on the same weights,
a greedy answer through the chat template, and the single-stream decode rate at several contexts.

    python tools/kolibri_engine_check.py --set DIR [--fp8 DIR] \
        [--texts bench/heldout_prose.txt,bench/heldout_code.txt] [--tokens 1024] \
        [--speed 1024,8192,32768] [--max-len 40000] [--attention auto|torch|kernel] [--no-ref] \
        [--out result.json]

Correctness. The reference is `tools/kolibri_ref.layer_forward` (the plugin's forward written out,
fp32 activations) over the engine's OWN weights, dequantised to fp32 one layer at a time from the
loaded `KLayer`s, so the comparison has no quantisation in it: what is left is bf16 activations
into the projections, bf16 KV, the kernels' sum order, and routing flips that follow from those.
Reported per text: the NLL of both, argmax agreement overall and on the positions where the
reference's top 1 leads its top 2 by at least 1.0 logit (the bar: >= 0.99), and the largest logit
gap. Texts: the held-out files and a few prompts through the chat template.

Greedy: the capital question in German and English through the template (effort none), decoded by
the engine's prefill + decode; "Berlin" must be in the German answer.

Speed: prefill N tokens of text, then 64 greedy decode steps, timed after 8 warm-up steps; tok/s
per context, against the byte ceiling of the weights a token reads (+ the KV it reads).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.kolibri.model import KolibriEngine  # noqa: E402
from tools import kolibri_ref as kr  # noqa: E402

BW = 273e9


def render(user: str, effort: str = "none", system: str | None = None) -> str:
    from tools.kolibri_corpus import EFFORT
    sys_txt = "# Reasoning effort\n\n" + EFFORT[effort]
    if system:
        sys_txt = system + "\n\n" + sys_txt
    s = f"<|im_start|>system\n{sys_txt}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    if effort == "none":
        s += "<think>\n\n</think>\n\n"
    return s


def ref_layer(lw, cfg) -> kr.LayerW:
    """The engine's layer L as fp32 dense weights (the reference's LayerW)."""
    q, k, v = lw.qkv.dense().split([cfg.q_size, cfg.kv_size, cfg.kv_size])
    E = cfg.experts
    G = torch.stack([lw.G.dense(e) for e in range(lw.G.E)])
    U = torch.stack([lw.U.dense(e) for e in range(lw.U.E)])
    D = torch.stack([lw.D.dense(e) for e in range(lw.D.E)])
    if lw.shared is None:
        sg, su, sd = G[E], U[E], D[E]
    else:
        sg, su = lw.shared.gu.dense().split([lw.shared.F, lw.shared.F])
        sd = lw.shared.d.dense()
    return kr.LayerW(n_in=lw.n_in.float(), n_pa=lw.n_pa.float(), n_pal=lw.n_pal.float(), n_pf=lw.n_pf.float(),
                     q=q, k=k, v=v, o=lw.o.dense(), qn=lw.q_norm.float(), kn=lw.k_norm.float(),
                     gate=lw.gate.float(), bias=lw.bias, sg=sg, su=su, sd=sd, eg=G[:E], eu=U[:E], ed=D[:E])


@torch.inference_mode()
def reference_logits(eng: KolibriEngine, seqs: list[list[int]]) -> list[torch.Tensor]:
    cfg = eng.cfg
    c = cfg.raw
    rs = [eng.emb[torch.tensor(s, device=eng.device)].float() for s in seqs]
    for lw in eng.layers:
        w = ref_layer(lw, cfg)
        for i, r in enumerate(rs):
            rs[i], _ = kr.layer_forward(r, w, c, lw.index)
        del w
        torch.cuda.empty_cache()
    out = []
    for r in rs:
        h = kr.rms(r, eng.final_norm.float(), cfg.eps)
        out.append(eng.head.logits(h))
    return out


def metrics(lg_e: torch.Tensor, lg_r: torch.Tensor, ids: list[int]) -> dict:
    t = torch.tensor(ids[1:], device=lg_e.device)
    m = {"tokens": len(ids)}
    for k, lg in (("engine", lg_e), ("ref", lg_r)):
        lp = torch.log_softmax(lg[:-1].float(), -1)
        m[f"nll_{k}"] = float(-lp.gather(1, t[:, None]).mean())
    t2 = lg_r.topk(2, -1)
    conf = (t2.values[:, 0] - t2.values[:, 1]) >= 1.0
    same = lg_e.argmax(-1) == t2.indices[:, 0]
    m["argmax_agree"] = float(same.float().mean())
    m["confident"] = int(conf.sum())
    m["argmax_agree_confident"] = float(same[conf].float().mean()) if conf.any() else None
    m["max_abs_logit_diff"] = float((lg_e - lg_r).abs().max())
    m["mean_abs_logit_diff"] = float((lg_e - lg_r).abs().mean())
    return m


@torch.inference_mode()
def greedy(eng: KolibriEngine, tok, prompt: str, n: int = 48) -> tuple[str, float]:
    ids = tok.encode(prompt, add_special_tokens=False).ids
    eng.reset()
    lg = eng.prefill(ids)
    out = []
    t0 = time.perf_counter()
    for _ in range(n):
        t = int(lg.argmax())
        out.append(t)
        if t == eng.cfg.eos:
            break
        lg = eng.decode(t)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return tok.decode(out, skip_special_tokens=False), len(out) / dt


@torch.inference_mode()
def speed(eng: KolibriEngine, ids: list[int], ctx: int, steps: int = 64, warm: int = 8) -> dict:
    src = (ids * (ctx // max(1, len(ids)) + 1))[:ctx]
    eng.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    lg = eng.prefill(src)
    torch.cuda.synchronize()
    tp = time.perf_counter() - t0
    for _ in range(warm):
        lg = eng.decode(int(lg.argmax()))
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(steps):
        lg = eng.decode(int(lg.argmax()))
    torch.cuda.synchronize()
    td = time.perf_counter() - t1
    cfg = eng.cfg
    full = sum(1 for L in range(cfg.layers) if not cfg.sliding(L))
    kv_rows = full * (ctx + warm + steps // 2) + (cfg.layers - full) * min(cfg.window, ctx)
    kv_bytes = kv_rows * cfg.kv_size * 2 * 2
    b = eng.decode_bytes() + kv_bytes
    return {"ctx": ctx, "prefill_s": tp, "prefill_tok_s": ctx / tp, "decode_tok_s": steps / td,
            "ms_per_token": td / steps * 1e3, "bytes_per_token_gb": b / 1e9, "ceiling_tok_s": BW / b,
            "share_of_ceiling": (steps / td) / (BW / b)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True)
    ap.add_argument("--fp8", default="")
    ap.add_argument("--attn-from", default=None, choices=[None, "fp8", "set"])
    ap.add_argument("--texts", default="bench/heldout_prose.txt,bench/heldout_code.txt")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--speed", default="1024,8192,32768")
    ap.add_argument("--max-len", type=int, default=40000)
    ap.add_argument("--attention", default="auto", choices=["auto", "torch", "kernel"])
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--decode-check", action="store_true",
                    help="also teacher-force every text through prefill(1 token) + decode steps "
                         "(the captured decode graph) and score those logits against the reference")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.fp8 or a.set, "tokenizer.json"))
    t0 = time.time()
    eng = KolibriEngine.load(a.set, a.fp8 or None, device="cuda", max_len=a.max_len, attn=a.attn_from,
                             attention=a.attention, graphs=not a.no_graphs)
    res = {"load_s": time.time() - t0, "info": eng.info, "attention": type(eng.attn).__name__,
           "gpu_alloc_gb": torch.cuda.memory_allocated() / 1e9, "decode_weight_bytes_gb": eng.decode_bytes() / 1e9}
    print(json.dumps(res), flush=True)

    seqs, names = [], []
    for f in [x for x in a.texts.split(",") if x]:
        seqs.append(tok.encode(open(f).read(), add_special_tokens=False).ids[: a.tokens])
        names.append(os.path.basename(f))
    chats = [("Was ist die Hauptstadt von Deutschland?", "none"),
             ("What is the capital of France? Answer in one sentence.", "none"),
             ("Schreibe eine Python-Funktion, die prüft, ob eine Zahl eine Primzahl ist.", "low")]
    for i, (u, eff) in enumerate(chats):
        p = render(u, eff)
        seqs.append(tok.encode(p, add_special_tokens=False).ids)
        names.append(f"chat{i}")
    # the engine's teacher-forced logits
    res["texts"] = {}
    eng_lg = []
    for s in seqs:
        eng.reset()
        eng_lg.append(eng.forward(s))
    dec_lg = []
    if a.decode_check:
        t1 = time.time()
        for s in seqs:
            eng.reset()
            rows = [eng.prefill(s[:1]).float()]
            for t in s[1:]:
                rows.append(eng.decode(t).float().clone())
            dec_lg.append(torch.stack(rows))
        res["decode_check_s"] = time.time() - t1
    if not a.no_ref:
        t1 = time.time()
        ref_lg = reference_logits(eng, seqs)
        res["ref_s"] = time.time() - t1
        tot_c = tot_a = 0
        for n, s, le, lr in zip(names, seqs, eng_lg, ref_lg):
            m = metrics(le, lr, s)
            res["texts"][n] = m
            tot_c += m["confident"]
            tot_a += round((m["argmax_agree_confident"] or 0) * m["confident"])
            print(f"[check] {n}: {json.dumps(m)}", flush=True)
        res["argmax_agree_confident_all"] = tot_a / max(1, tot_c)
        res["pass"] = res["argmax_agree_confident_all"] >= 0.99
        print(f"[check] confident agreement {res['argmax_agree_confident_all']:.4f} over {tot_c} "
              f"positions: {'PASS' if res['pass'] else 'FAIL'}", flush=True)
        if dec_lg:
            res["texts_decode"] = {}
            tot_c = tot_a = 0
            for n, s, le, lr in zip(names, seqs, dec_lg, ref_lg):
                m = metrics(le, lr, s)
                res["texts_decode"][n] = m
                tot_c += m["confident"]
                tot_a += round((m["argmax_agree_confident"] or 0) * m["confident"])
                print(f"[decode-check] {n}: {json.dumps(m)}", flush=True)
            res["decode_agree_confident_all"] = tot_a / max(1, tot_c)
            res["decode_pass"] = res["decode_agree_confident_all"] >= 0.99
            print(f"[decode-check] confident agreement {res['decode_agree_confident_all']:.4f} over "
                  f"{tot_c} positions: {'PASS' if res['decode_pass'] else 'FAIL'}", flush=True)
        del ref_lg
    del eng_lg, dec_lg
    torch.cuda.empty_cache()
    # greedy through the template
    res["greedy"] = {}
    for u, eff in chats[:2] + [("Erkläre in zwei Sätzen, warum der Himmel blau ist.", "none")]:
        txt, tps = greedy(eng, tok, render(u, eff), 64)
        res["greedy"][u] = {"answer": txt, "tok_s": tps}
        print(f"[greedy] {u!r} -> {txt!r} ({tps:.1f} tok/s)", flush=True)
    res["berlin"] = "Berlin" in res["greedy"][chats[0][0]]["answer"]
    # speed
    res["speed"] = []
    base = seqs[0] + seqs[1] + seqs[2]
    for ctx in [int(x) for x in a.speed.split(",") if x]:
        if ctx + 100 > a.max_len:
            continue
        s = speed(eng, base, ctx)
        res["speed"].append(s)
        print(f"[speed] {json.dumps(s)}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
