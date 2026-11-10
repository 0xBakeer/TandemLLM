"""SPD-249 M1b: the prefill's mixer GEMM arms (QWEN38_FN_PREFILL_MM) against a prefill whose mixer GEMMs are
computed in fp64. Test tool; the fp64 arm is not a knob.

    python tools/fn_prefill_mm_ref.py texts --pack PACK --src DIR --out prompts.json [--ctx 3072,8192] [--at 0.25,0.55]
    python tools/fn_prefill_mm_ref.py gate --pack PACK --prompts prompts.json --json out.json [--arms fp32,tcn,tc]
        [--noise 6e-7~0,1.2e-6~0,...] [--deadline 1500]

texts (CPU, tokenizer only): the gate's prose prompts (tools/fn_generate.py `class_prompt`, with the row where its
prose ends and the filler starts) and, for every Project Gutenberg file in DIR, one passage per context, from the
paragraph at fraction `--at` of the book, cut so the templated prompt (passage, then the prose request) is `ctx`
tokens.

gate: per prompt the reference trunk (`forward`), the fp64-mixer prefill (`forward_pre` with every mixer GEMM of
`mm_rows` as x times q s in fp64, the sum rounded once to fp32; everything else as the fp32 arm), then each arm's
`forward_pre`. Every arm is scored against both yardsticks the way tools/fn_prefill_check.py scores the kernels
(teacher-forced top-1 by the yardstick's margin), plus where the rows that differ sit and the streams' distance
per row. Then (`--noise`, until `--deadline` s) the fp32 arm with its mixer GEMM outputs times 1 + eps z, a seed a
run, against the same two yardsticks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SECRET = ("KEY", "TOKEN", "SECRET", "PASS", "AUTH", "CRED")


def gutenberg_body(path: str) -> tuple[str, list[str]]:
    """A Gutenberg plain-text file's title line and its paragraphs (licence header and footer cut, lines joined)."""
    raw = open(path, encoding="utf-8-sig").read().replace("\r\n", "\n")
    title = raw.split("\n", 1)[0].strip()
    a = raw.find("*** START OF")
    a = raw.find("\n", a) + 1 if a >= 0 else 0
    b = raw.find("*** END OF")
    body = raw[a:b if b > 0 else len(raw)]
    paras = [" ".join(x.split()) for x in re.split(r"\n\s*\n", body)]
    return title, [p for p in paras if p]


def fit(tok, text: str, ctx: int, ask: str) -> list[int]:
    """The longest head of `text` whose templated prompt (text, then `ask`) is at most `ctx` tokens."""
    from tools.fn_generate import chat_ids
    fid = tok(text, add_special_tokens=False)["input_ids"]
    lo, hi = 0, len(fid)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(chat_ids(tok, tok.decode(fid[:mid]) + "\n\n" + ask)) <= ctx:
            lo = mid
        else:
            hi = mid - 1
    return chat_ids(tok, tok.decode(fid[:lo]) + "\n\n" + ask)


def cmd_texts(a) -> None:
    from transformers import AutoTokenizer

    from tools import fn_generate as G
    tok = AutoTokenizer.from_pretrained(a.pack)
    ctxs = [int(x) for x in a.ctx.split(",")]
    ats = [float(x) for x in a.at.split(",")]
    out = []
    prose_only = G.chat_ids(tok, open(os.path.join(G.ROOT, "bench", G.FILL["prose"])).read())
    for n in ctxs:
        ids = G.class_prompt(tok, "prose", n)
        split = next((i for i in range(min(len(ids), len(prose_only))) if ids[i] != prose_only[i]), None)
        out.append({"name": f"gate-prose-{n}", "source": "bench/heldout_prose.txt + filler", "ctx": n, "ids": ids,
                    "split": split, "head": tok.decode(ids[:40])})
    for fn in sorted(os.listdir(a.src)):
        if not re.fullmatch(r"pg\d+\.txt", fn):
            continue
        title, paras = gutenberg_body(os.path.join(a.src, fn))
        total = sum(len(p) for p in paras)
        for n, at in zip(ctxs, ats):
            acc, i0 = 0, 0
            while i0 < len(paras) and acc < at * total:
                acc += len(paras[i0])
                i0 += 1
            text, j = "", i0
            while j < len(paras) and len(text) < n * 8:
                text += ("\n\n" if text else "") + paras[j]
                j += 1
            ids = fit(tok, text, n, G.ASK["prose"])
            out.append({"name": f"{fn[:-4]}-{n}", "source": fn, "title": title, "ctx": n, "at": at, "paragraph": i0,
                        "ids": ids, "split": None, "head": text[:120]})
    for r in out:
        print(f"[texts] {r['name']}: {len(r['ids'])} ids, split {r['split']} | {r['head'][:90]!r}", flush=True)
    json.dump(out, open(a.out, "w"))
    print("M1BTEXTSDONE", flush=True)


def mm_f64(x: torch.Tensor, w) -> torch.Tensor:
    """x @ (q s)^T with x and the weight exact in fp64 (an int8 times an fp32 scale, a bf16 x), the sum in fp64,
    rounded once to fp32."""
    if hasattr(w, "q"):
        q, s = w.q, w.s
        N, K = q.shape
        sv = s.reshape(N, -1, 1).double()
        wd = (q.double().view(N, sv.shape[1], -1) * sv).view(N, K)
    else:
        wd = w.double()
    return (x.double() @ wd.T).float()


def cmd_gate(a) -> None:
    from engine.settings import SETTINGS as S
    from engine.models.flashnext.engine import FlashNextEngine
    from engine.models.flashnext.load import load
    from tools import fn_prefill_check as PC
    from tools import fn_prefill_kernels as PK
    from tools import fn_prefill_profile as PP
    from tools.fn_generate import mem_available_gib

    t_start = time.time()
    prompts = json.load(open(a.prompts))
    if a.only:
        keep = set(a.only.split(","))
        prompts = [p for p in prompts if p["name"] in keep]
    arms = a.arms.split(",")
    noise = [(float(e), int(s)) for e, s in (x.split("~") for x in a.noise.split(","))] if a.noise else []
    idx, W, banks, tables, timing = load(a.pack, "cuda")
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=max(len(p["ids"]) for p in prompts) + 128)
    dev = eng.device
    fam = PP.weight_families(eng)
    mm0, MM0 = PK.mm_rows, PK.MM
    env = {k: v for k, v in sorted(os.environ.items())
           if k.startswith(("QWEN38_", "QSE_", "FN_")) and not any(s in k.upper() for s in SECRET)}
    res = {"tree": os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pack": a.pack, "load": timing,
           "mem_after_load_gib": mem_available_gib(),
           "knobs": {"chunk": a.chunk, "arms": arms, "noise": a.noise, "deadline_s": a.deadline,
                     "prefill_mm_default": MM0, "settings_FN_PREFILL_MM": S.get("FN_PREFILL_MM"),
                     "f64_arm": "mixer GEMMs of mm_rows as x @ (q s)^T in fp64, rounded once to fp32; rest as fp32",
                     "noise_form": "fp32 arm, mixer GEMM outputs of mm_rows times 1 + eps z, z normal, one generator a run",
                     "engine_dtype": str(getattr(eng, "dtype", None)), "kv_dtype": "bf16 (engine default)", "mtp": False, "continuation": False,
                     "env": env},
           "prompts": [{k: v for k, v in p.items() if k != "ids"} | {"tokens": len(p["ids"])} for p in prompts],
           "rows": [], "noise_rows": []}

    def dump():
        json.dump(res, open(a.json, "w"), indent=1)

    @torch.no_grad()
    def ref(ids):
        st, out = eng.new_state(eng.max_len), []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, len(ids), a.chunk):
            eng.forward(torch.tensor(ids[i:i + a.chunk], device=dev), st, head="none")
            out.append(eng.stack)
            if st.length > 4096:
                torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        del st
        torch.cuda.empty_cache()
        return torch.cat(out), round(t, 2)

    @torch.no_grad()
    def pre(ids, arm, f64=False, eps=0.0, seed=0):
        calls = [0]
        g = torch.Generator(device=dev).manual_seed(seed) if eps else None

        def mm(x, w):
            if fam.get(id(w)) != "mixers":
                return mm0(x, w)
            calls[0] += 1
            y = mm_f64(x, w) if f64 else mm0(x, w)
            if eps:
                y = y * (1.0 + eps * torch.randn(y.shape, device=y.device, generator=g))
            return y
        PK.MM = arm
        if f64 or eps:
            PK.mm_rows = mm
        try:
            st, out = eng.new_state(eng.max_len), []
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(0, len(ids), a.chunk):
                eng.forward_pre(torch.tensor(ids[i:i + a.chunk], device=dev), st, head="none")
                out.append(eng.stack)
                if st.length > 4096:
                    torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t = time.perf_counter() - t0
        finally:
            PK.mm_rows, PK.MM = mm0, MM0
        del st
        torch.cuda.empty_cache()
        return torch.cat(out), calls[0], round(t, 2)

    @torch.no_grad()
    def score(sx, sy, split):
        r = PC.pair_rows(eng, sx, sy)
        out = PC.compare(r)
        bad = r["ix"] != r["iy"]
        n = int(bad.numel())
        out["disagree_hist_512"] = torch.bincount(bad.nonzero().view(-1) // 512, minlength=(n + 511) // 512).tolist()
        if split:
            out["split"] = {"rows_before": split, "agree_before": int((~bad[:split]).sum()),
                            "rows_after": n - split, "agree_after": int((~bad[split:]).sum())}
        d = []
        for r0 in range(0, n, 1024):
            x, y = sx[r0:r0 + 1024].float(), sy[r0:r0 + 1024].float()
            d.append(((y - x).norm(dim=1) / x.norm(dim=1).clamp_min(1e-30)).cpu())
        d = torch.sort(torch.cat(d)).values
        out["stream_dist"] = {"median": float(d[n // 2]), "mean": float(d.mean()), "p90": float(d[int(n * 0.9)])}
        return out

    def line(tag, name, arm, sc):
        sp = sc.get("split")
        spl = f" before/after split {sp['agree_before']}/{sp['rows_before']} {sp['agree_after']}/{sp['rows_after']}" if sp else ""
        return (f"[m1b] {tag:8} {name:18} {arm:16} agree {sc['agree']}/{sc['positions']} ({sc['agree_frac']:.4f}) sure "
                f"{sc['both_margins_over_5']} by margin {[v['agree'] for v in sc['by_margin'].values()]}{spl} dist "
                f"{sc['stream_dist']['median']:.3e} logit p50 {sc['max_logit_diff']['p50']}")

    keep = {}
    for k, p in enumerate(prompts):
        ids, name, split = p["ids"], p["name"], p.get("split")
        s_ref, t_ref = ref(ids)
        s64, n64, t64 = pre(ids, "fp32", f64=True)
        row = {"name": name, "ctx": p["ctx"], "tokens": len(ids), "split": split, "ref_s": t_ref, "f64_s": t64,
               "f64_mixer_gemms": n64, "f64_vs_ref": score(s_ref, s64, split), "arms": {}}
        print(line("f64/ref", name, "f64", row["f64_vs_ref"]) + f" | ref {t_ref} s f64 {t64} s gemms {n64}", flush=True)
        if k == 0:
            again, _, _ = pre(ids, "fp32", f64=True)
            row["f64_repeat_bit_equal"] = bool(torch.equal(again, s64))
            print(f"[m1b] f64 arm run twice on {name}: bit-equal {row['f64_repeat_bit_equal']}", flush=True)
            del again
        for arm in arms:
            sy, _, t = pre(ids, arm)
            row["arms"][arm] = {"s": t, "vs_ref": score(s_ref, sy, split), "vs_f64": score(s64, sy, split)}
            print(line("vs ref", name, arm, row["arms"][arm]["vs_ref"]) + f" | {t} s", flush=True)
            print(line("vs f64", name, arm, row["arms"][arm]["vs_f64"]), flush=True)
            del sy
        row["mem_gib"] = mem_available_gib()
        res["rows"].append(row)
        dump()
        keep[name] = (s_ref.cpu(), s64.cpu(), split)
        del s_ref, s64
        torch.cuda.empty_cache()
    print(f"[m1b] main arms done at {time.time() - t_start:.0f} s, MemAvailable {mem_available_gib()} GiB", flush=True)
    for p in prompts:
        if not noise:
            break
        name = p["name"]
        s_ref, s64, split = (x.to(dev) if torch.is_tensor(x) else x for x in keep[name])
        for eps, seed in noise:
            if time.time() - t_start > a.deadline:
                res["noise_rows"].append({"name": name, "eps": eps, "seed": seed, "skipped": "deadline"})
                continue
            sy, calls, t = pre(p["ids"], "fp32", eps=eps, seed=seed)
            r = {"name": name, "ctx": p["ctx"], "eps": eps, "seed": seed, "noisy_mixer_gemms": calls, "s": t,
                 "vs_ref": score(s_ref, sy, split), "vs_f64": score(s64, sy, split)}
            res["noise_rows"].append(r)
            arm = f"fp32~{eps:g}~{seed}"
            print(line("vs ref", name, arm, r["vs_ref"]) + f" | {t} s gemms {calls}", flush=True)
            print(line("vs f64", name, arm, r["vs_f64"]), flush=True)
            del sy
        dump()
        del s_ref, s64
        torch.cuda.empty_cache()
    res["wall_s"] = round(time.time() - t_start, 1)
    dump()
    print("M1BGATEDONE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("texts")
    t.add_argument("--pack", required=True)
    t.add_argument("--src", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--ctx", default="3072,8192")
    t.add_argument("--at", default="0.25,0.55")
    g = sub.add_parser("gate")
    g.add_argument("--pack", required=True)
    g.add_argument("--prompts", required=True)
    g.add_argument("--json", required=True)
    g.add_argument("--only", default="")
    g.add_argument("--arms", default="fp32,tcn,tc")
    g.add_argument("--noise", default="")
    g.add_argument("--chunk", type=int, default=2048)
    g.add_argument("--deadline", type=float, default=1500.0)
    a = ap.parse_args()
    (cmd_texts if a.cmd == "texts" else cmd_gate)(a)


if __name__ == "__main__":
    main()
