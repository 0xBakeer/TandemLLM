"""Flint's correctness ladder and its speed numbers, one model a process (ENG-399, SPD-142).

    PY=<engine python>; PYTHONPATH=<repo>:$PYTHONPATH
    $PY tools/flint_check.py engine --model <dir> --out run.pt     greedy decode of the prompts below on the
                                                                   engine (prefill, then the captured decode
                                                                   step), every step's logits kept
    $PY tools/flint_check.py cmp --a run.pt --b ref.pt             two runs on the same tokens (`engine --force`
                                                                   feeds one run's tokens to another model)
    $PY tools/flint_check.py bundle --model <dir> --bundle <ref>   the engine teacher-forced on a reference
                                                                   bundle (input ids, BF16 logits and the
                                                                   converter's own logits on the pack): KL and
                                                                   top-1 agreement against each
    $PY tools/flint_check.py mkbundle --model <bf16 dir> --out <ref>   such a bundle from this engine on BF16
    $PY tools/flint_check.py bench --model <dir>                   greedy decode tok/s on prose, code and chat
    $PY tools/flint_check.py staircase --model <dir>               verify ms for 1..16 rows, the experts a block
                                                                   reads, prefill tok/s
    $PY tools/flint_check.py prices --stair s.json --bench b.json --out <price file>
    $PY tools/flint_check.py kvfp8 --model <dir> --bundle <dir>    the e4m3 cache (FLINT_KV_FP8) against the bf16 one
    $PY tools/flint_check.py audit --pack <dir>                    the pack's audit record (CPU)

A comparison reports, over every position: the argmax agreement, the positions where the two disagree and
whether each is a near tie (the reference's margin under 8 bf16 steps of its largest logit), the logit
difference, and the KL from the reference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BENCH = {
    "prose": "Write three paragraphs about why the memory system, and not the arithmetic units, sets the speed "
             "of a language model that generates one token at a time.",
    "code": "Write a Python function that parses an ISO-8601 duration such as P3DT4H5M6S into a number of "
            "seconds. Validate the input, raise ValueError with a useful message, and include three doctest examples.",
    "chat": "I have a machine with 121 GB of unified memory and about 273 GB/s of bandwidth. Explain in plain "
            "language what that means for running a mixture-of-experts model with 2.5 billion active parameters.",
}
PROMPTS = [
    "Explain in three short paragraphs why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Give me a packing list for a weekend hiking trip in the Alps in October.",
    "Translate into German: The meeting is moved to Thursday at ten, please bring the report.",
    "A train leaves at 14:35 and arrives at 17:10. How long is the trip? Show the steps.",
    "Summarise the plot of Romeo and Juliet in five sentences.",
    "List five differences between TCP and UDP as a table.",
]


def _ids(tok, prompt: str) -> list[int]:
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


def _engine(model: str, max_len: int):
    from engine.models.flint.engine import FlintEngine
    from engine.models.flint.load import load
    t0 = time.time()
    idx, W, experts, timing = load(model, "cuda", log=None)
    eng = FlintEngine(idx.cfg, W, experts, device="cuda", max_len=max_len)
    torch.cuda.synchronize()
    return idx, eng, {"load_s": round(time.time() - t0, 1), "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                      **timing}


def cmd_engine(a) -> None:
    from transformers import AutoTokenizer

    from engine.models.flint.engine import DecodeGraph
    tok = AutoTokenizer.from_pretrained(a.model)
    idx, eng, info = _engine(a.model, a.max_len)
    print(json.dumps({"load": info, "notes": idx.notes}), flush=True)
    out = {"model": a.model, "tokens": a.tokens, "runs": [], "info": info}
    forced = torch.load(a.force, weights_only=False)["runs"] if a.force else None
    for j, p in enumerate(PROMPTS[:a.prompts]):
        ids = forced[j]["ids"] if forced else _ids(tok, p)
        st = eng.new_state(a.max_len)
        lg = eng.prefill(torch.tensor(ids), st)
        rows = [lg[0].to(torch.bfloat16).cpu()]
        gen = [forced[j]["gen"][0] if forced else int(lg[0].argmax())]
        dg = DecodeGraph(eng, st) if a.graph == "on" else None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.tokens - 1):
            if dg is not None:
                lg = dg(gen[-1])
            else:
                eng.stage(gen[-1], st)
                lg = eng.step(st)
                st.length += 1
            rows.append(lg[0].to(torch.bfloat16).cpu())
            gen.append(int(lg[0].argmax()))
            if forced:
                gen[-1] = forced[j]["gen"][len(gen) - 1]           # teacher-forced: the other run's tokens
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        r = {"prompt": p, "ids": ids, "gen": gen, "decode_tok_s": round((a.tokens - 1) / dt, 2),
             "text": tok.decode(gen)}
        out["runs"].append({**r, "logits": torch.stack(rows)})
        print(json.dumps({k: r[k] for k in ("prompt", "decode_tok_s")} | {"text": r["text"][:160]}), flush=True)
    torch.save(out, a.out)


def _compare(run: dict, ref_logits: list) -> dict:
    agree = tot = near = other = 0
    maxrel, mean = 0.0, []
    bad = []
    for r, ref in zip(run["runs"], ref_logits):
        eng = r["logits"].float()
        ref = ref.float()
        scale = ref.abs().max(-1).values
        d = (eng - ref).abs()
        maxrel = max(maxrel, float((d.max(-1).values / scale).max()))
        mean.append(float((d.mean(-1) / scale).mean()))
        ea, ra = eng.argmax(-1), ref.argmax(-1)
        for i in range(eng.shape[0]):
            tot += 1
            if ea[i] == ra[i]:
                agree += 1
                continue
            margin = float(ref[i, ra[i]] - ref[i, ea[i]])
            tie = margin < 8 * 2 ** -7 * float(scale[i])
            near += tie
            other += not tie
            if len(bad) < 12:
                bad.append({"prompt": r["prompt"][:40], "pos": i, "margin": round(margin, 4), "near_tie": bool(tie)})
    return {"positions": tot, "argmax_agree": agree, "agree_pct": round(100 * agree / max(tot, 1), 2),
            "near_tie": near, "other": other, "max_rel_diff": round(maxrel, 4),
            "mean_rel_diff": round(sum(mean) / max(len(mean), 1), 5), "disagreements": bad}


def cmd_cmp(a) -> None:
    """Two runs on the same tokens: `--a` an engine run, `--b` an engine run or a reference's saved logits
    (hf/exl3 --save): the comparison of `_compare` with b as the reference (a quality figure when b is
    BF16 and a is a pack)."""
    run = torch.load(a.a, weights_only=False)
    b = torch.load(a.b, weights_only=False)
    refs = [r["logits"] for r in b["runs"]] if isinstance(b, dict) else b
    res = _compare(run, refs)
    kls = []
    for r, ref in zip(run["runs"], refs):
        p = torch.log_softmax(ref.float(), -1)
        q = torch.log_softmax(r["logits"].float(), -1)
        kls.append((p.exp() * (p - q)).sum(-1))
    k = torch.cat(kls)
    res["kl_ref_to_a"] = {"mean": round(float(k.mean()), 5), "median": round(float(k.median()), 5),
                          "p90": round(float(k.quantile(0.9)), 4)}
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def cmd_bench(a) -> None:
    """Single-stream greedy decode: `tokens` tokens after each of the prose, code and chat prompts (one fresh
    state each, the captured decode step, a sync every token: the server's loop), ms a token from the host's
    wall clock; then the prefill of `prefill` tokens of held-out text (bench/heldout_prose.txt, heldout_code.txt, calib.txt)."""
    from transformers import AutoTokenizer

    from engine.models.flint.engine import DecodeGraph
    tok = AutoTokenizer.from_pretrained(a.model)
    idx, eng, info = _engine(a.model, a.max_len)
    res = {"model": a.model, "load": info, "tokens": a.tokens, "rows": {}}
    eos = set(idx.cfg.eos_token_ids)
    for kind, p in BENCH.items():
        ids = _ids(tok, p)
        st = eng.new_state(a.max_len)
        lg = eng.prefill(torch.tensor(ids), st)
        t = int(lg[0].argmax())
        dg = DecodeGraph(eng, st)
        t = dg.greedy(t)                         # the capture
        torch.cuda.synchronize()
        n, ended = 0, None
        t0 = time.perf_counter()
        for _ in range(a.tokens):
            t = dg.greedy(t)
            n += 1
            if t in eos and ended is None:
                ended = n                        # greedy past the end token: the bytes a token are the same
        dt = time.perf_counter() - t0
        res["rows"][kind] = {"tok_s": round(n / dt, 2), "ms_token": round(1e3 * dt / n, 3), "eos_at": ended}
        print(json.dumps({kind: res["rows"][kind]}), flush=True)
    res["graph"] = {"ms_token": round(sum(r["ms_token"] for r in res["rows"].values()) / len(res["rows"]), 3)}
    res["decode_bytes"] = idx.decode_bytes(context=512)["total"]
    res["GBps"] = round(res["decode_bytes"] / (res["graph"]["ms_token"] / 1e3) / 1e9, 1)
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench")
    text = "\n\n".join(open(os.path.join(root, f)).read() for f in ("heldout_prose.txt", "heldout_code.txt", "calib.txt"))
    pids = tok(text, add_special_tokens=False)["input_ids"][:a.prefill]
    for rep in range(2):
        st = eng.new_state(a.max_len)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        eng.prefill(torch.tensor(pids), st)
        torch.cuda.synchronize()
        res["prefill"] = {"tokens": len(pids), "tok_s": round(len(pids) / (time.perf_counter() - t0), 1), "rep": rep}
    print(json.dumps({"prefill": res["prefill"]}), flush=True)
    res["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(json.dumps(res), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def cmd_mkbundle(a) -> None:
    """A reference bundle from this engine on the BF16 checkpoint: `rows` rows of `length` tokens of held-out
    text (bench/heldout_prose.txt, heldout_code.txt, calib.txt, cut where each row starts) and every position's
    BF16 logits -- what `bundle` compares a pack against, without the converter."""
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    idx, eng, info = _engine(a.model, a.max_len)
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench")
    ids = []
    for f in ("heldout_prose.txt", "heldout_code.txt", "calib.txt"):
        t = tok(open(os.path.join(root, f)).read(), add_special_tokens=False)["input_ids"]
        ids += [t[i:i + a.length] for i in range(0, len(t) - a.length + 1, a.length)]
    step = max(1, len(ids) // a.rows)
    ids = torch.tensor(ids[::step][:a.rows])
    logits = []
    for r in range(ids.shape[0]):
        st = eng.new_state(a.max_len)
        logits.append(eng.prefill(ids[r], st, head="all").to(torch.bfloat16).cpu())
    os.makedirs(a.out, exist_ok=True)
    save_file({"input_ids": ids.contiguous()}, os.path.join(a.out, "input_ids.safetensors"))
    save_file({"logits": torch.stack(logits).contiguous()}, os.path.join(a.out, "logits_bf16.safetensors"))
    print(json.dumps({"rows": ids.shape[0], "length": ids.shape[1], "out": a.out}))


def cmd_bundle(a) -> None:
    """The engine teacher-forced on a reference bundle (the conversion job's: input ids, the BF16 logits and
    the converter's logits on the pack, 8 rows): our rows chunk by chunk through the prefill path's head
    (head="all"), then KL and top-1 agreement against the BF16 and against the converter."""
    from safetensors.torch import load_file

    def one(path, key):
        # this tool's bundles keep one stacked tensor; the converter's model_diff keeps `tensor.N` rows
        d = load_file(path)
        if key in d:
            return d[key]
        return torch.cat([d[f"tensor.{i}"] for i in range(len(d))])
    idx, eng, info = _engine(a.model, a.max_len)
    ids = one(os.path.join(a.bundle, "input_ids.safetensors"), "input_ids")
    refs = {}
    for name in sorted(os.listdir(a.bundle)):
        if name.startswith("logits_") and name.endswith(".safetensors"):
            refs[name[len("logits_"):-len(".safetensors")]] = one(os.path.join(a.bundle, name), "logits")
    ids = ids.view(-1, ids.shape[-1])
    res = {"model": a.model, "rows": ids.shape[0], "length": ids.shape[1], "against": {}}
    ours = []
    for r in range(ids.shape[0]):
        st = eng.new_state(a.max_len)
        ours.append(eng.prefill(ids[r], st, head="all").float().cpu())
    V = idx.cfg.vocab_size
    for name, ref in refs.items():
        ref = ref.view(ids.shape[0], ids.shape[1], -1)[..., :V].float()
        kl, agree, tot, nll_a, nll_b = [], 0, 0, 0.0, 0.0
        for r in range(ids.shape[0]):
            p = torch.log_softmax(ref[r], -1)
            q = torch.log_softmax(ours[r], -1)
            kl.append((p.exp() * (p - q)).sum(-1))
            agree += int((ref[r].argmax(-1) == ours[r].argmax(-1)).sum())
            tot += ids.shape[1]
            tgt = ids[r, 1:].long()
            nll_a += float(-q[:-1].gather(1, tgt[:, None]).sum())
            nll_b += float(-p[:-1].gather(1, tgt[:, None]).sum())
        k = torch.cat(kl)
        n = ids.shape[0] * (ids.shape[1] - 1)
        res["against"][name] = {"kl_mean": round(float(k.mean()), 5), "kl_median": round(float(k.median()), 5),
                                "kl_p90": round(float(k.quantile(0.9)), 4), "top1_agree_pct": round(100 * agree / tot, 2),
                                "ppl_engine": round(float(torch.exp(torch.tensor(nll_a / n))), 4),
                                "ppl_ref": round(float(torch.exp(torch.tensor(nll_b / n))), 4)}
        print(json.dumps({name: res["against"][name]}), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)

def _teacher(eng, st, ids: torch.Tensor, start: int, block: int) -> torch.Tensor:
    """Logits of rows `start`.. of `ids` teacher-forced through verify blocks of `block` rows after a prefill
    of the first `start` (each block reads the cache the rows before it left), fp32 on the host."""
    from engine.models.flint.ref import chain
    eng.prefill(ids[:start], st, head="none")
    out = []
    for a in range(start, ids.numel(), block):
        toks = [int(t) for t in ids[a:a + block]]
        eng.stage_rows(toks, chain(len(toks)), st)
        out.append(eng.rows_step(st, len(toks)).float().cpu().clone())
        eng.commit(st, list(range(len(toks))))
    return torch.cat(out)


def _kl_rows(p_logits: torch.Tensor, q_logits: torch.Tensor, tgt: torch.Tensor) -> dict:
    p, q = torch.log_softmax(p_logits, -1), torch.log_softmax(q_logits, -1)
    kl = (p.exp() * (p - q)).sum(-1)
    n = tgt.numel()
    return {"kl_mean": round(float(kl.mean()), 5), "kl_median": round(float(kl.median()), 5),
            "kl_p90": round(float(kl.quantile(0.9)), 4),
            "top1_agree_pct": round(100 * float((p_logits.argmax(-1) == q_logits.argmax(-1)).float().mean()), 2),
            "ppl_exact": round(float(torch.exp(-p[:n].gather(1, tgt[:, None]).mean())), 4),
            "ppl_fp8": round(float(torch.exp(-q[:n].gather(1, tgt[:, None]).mean())), 4)}


def cmd_kvfp8(a) -> None:
    """FLINT_KV_FP8 against the exact cache, one engine with both states: each bundle row prefilled to
    `--start` tokens, the rest teacher-forced through `--block`-row verify blocks that read the cache (the
    path decode and drafting take); then a long prompt at each `--long` length (comma separated), its last
    `--tail` tokens teacher-forced the same way: held-out text repeated, or with `--long-file` the first tokens of
    one text that does not repeat. KL, top-1 and perplexity of the e4m3 cache against the bf16, each over the
    rows that have a next token."""
    from safetensors.torch import load_file
    from engine.models.flint.engine import ROWS_MAX
    from transformers import AutoTokenizer
    longest = max(int(x) for x in str(a.long).split(","))
    idx, eng, info = _engine(a.model, max(a.max_len, longest + 64))
    c = eng.cfg

    def state(fp8: bool):
        eng.kv_fp8 = fp8
        if fp8 and eng._kst is None:
            eng._kst = torch.zeros(c.num_key_value_heads, ROWS_MAX + 1, c.head_dim, dtype=eng.dtype, device=eng.device)
            eng._vst = torch.zeros_like(eng._kst)
        return eng.new_state(max(a.max_len, longest + 64))
    d = load_file(os.path.join(a.bundle, "input_ids.safetensors"))
    ids = d["input_ids"] if "input_ids" in d else torch.cat([d[f"tensor.{i}"] for i in range(len(d))])
    ids = ids.view(-1, ids.shape[-1])
    res = {"model": a.model, "block": a.block, "start": a.start, "rows": ids.shape[0], "length": ids.shape[1]}
    P, Q, T = [], [], []
    for r in range(ids.shape[0]):
        # a row's last logit has no target in the row: dropped, so the rows line up with T when joined
        P.append(_teacher(eng, state(False), ids[r], a.start, a.block)[:-1])
        Q.append(_teacher(eng, state(True), ids[r], a.start, a.block)[:-1])
        T.append(ids[r, a.start + 1:].long())
    res["bundle"] = _kl_rows(torch.cat(P), torch.cat(Q), torch.cat(T))
    print(json.dumps({"bundle": res["bundle"]}), flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench")
    if a.long_file:
        # one long text that does not repeat (a source file): the prompt is its first `long` tokens
        base = tok(open(a.long_file).read(), add_special_tokens=False)["input_ids"]
    else:
        text = "\n\n".join(open(os.path.join(root, f)).read() for f in ("heldout_prose.txt", "heldout_code.txt",
                                                                          "calib.txt"))
        base = tok(text, add_special_tokens=False)["input_ids"]
    for n in [int(x) for x in str(a.long).split(",")]:
        if a.long_file and n > len(base):
            raise SystemExit(f"{a.long_file}: {len(base)} tokens, fewer than {n}")
        lid = torch.tensor((base * (n // len(base) + 1))[:n])
        s0 = n - a.tail
        p = _teacher(eng, state(False), lid, s0, a.block)
        q = _teacher(eng, state(True), lid, s0, a.block)
        res.setdefault("long", {})[str(n)] = {"tokens": n, "tail": a.tail, **_kl_rows(p[:-1], q[:-1], lid[s0 + 1:].long())}
        print(json.dumps({"long": {str(n): res["long"][str(n)]}}), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def cmd_staircase(a) -> None:
    """Verify blocks of 1..16 rows (chains) from captured graphs at two context lengths: ms a block
    (CUDA events, the median of `reps`), and the distinct experts the block's rows route to, summed
    over the MoE layers (from an eager run of the same block)."""
    from engine.models.flint.engine import VerifyGraph
    from engine.models.flint import kernels as K
    idx, eng, info = _engine(a.model, a.max_len)
    g = torch.Generator().manual_seed(0)
    res = {"model": a.model, "load": info, "rows": {}}
    res["prefill_tok_s"] = {}
    for ctx in a.contexts:
        st = eng.new_state(a.max_len)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        eng.prefill(torch.randint(0, 150_000, (ctx,), generator=g), st)
        torch.cuda.synchronize()
        res["prefill_tok_s"][ctx] = round(ctx / (time.perf_counter() - t0), 1)
        print(json.dumps({"ctx": ctx, "prefill_tok_s": res["prefill_tok_s"][ctx]}), flush=True)
        toks = torch.randint(0, 150_000, (16,), generator=g).tolist()
        out = {}
        for M in range(1, 17):
            ns = K.class_of(st.length + M, st.cap)
            eng.route_trace = []
            eng.stage_rows(toks[:M], list(range(-1, M - 1)), st)
            eng.rows_step(st, M, ns=ns)
            experts = sum(len(set(ids.view(-1).tolist())) for _, ids in eng.route_trace)
            eng.route_trace = None
            vg = VerifyGraph(eng, st, M, ns)
            vg.capture()
            eng.stage_rows(toks[:M], list(range(-1, M - 1)), st)
            times = []
            for _ in range(a.reps):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                vg.replay()
                e1.record()
                torch.cuda.synchronize()
                times.append(e0.elapsed_time(e1))
            times.sort()
            out[M] = {"ms": round(times[len(times) // 2], 3), "experts_read": experts}
            print(json.dumps({"ctx": ctx, "rows": M, **out[M]}), flush=True)
            del vg
        res["rows"][ctx] = out
    print(json.dumps(res), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def cmd_audit(a) -> None:
    """The pack's audit record from its headers alone (CPU, nothing loaded): kinds, formats, the K
    histogram, resident and decode bytes; a tensor the kernels cannot decode is named."""
    from engine.models.flint.pack import PackIndex, summary
    idx = PackIndex.open(a.pack)
    try:
        idx.audit()
        ok = None
    except ValueError as e:
        ok = str(e)
    s = summary(idx)
    s.pop("files")
    s["refused"] = ok
    s["safetensors_gb"] = round(sum(os.path.getsize(os.path.join(idx.path, f)) for f in os.listdir(idx.path)
                                    if f.endswith(".safetensors")) / 1e9, 2)
    print(json.dumps(s, indent=1))


def cmd_prices(a) -> None:
    """StairCut's price file from a staircase run and a bench run: per attention class (the adapter's
    price_class of the staircase's contexts) the chain verify ms for 1..16 rows, the plain step's ms."""
    from engine.models.flint import kernels as K
    stair = json.load(open(a.stair))
    bench = json.load(open(a.bench))
    plain = bench["graph"]["ms_token"]
    meas = {}
    for ctx, rows in stair["rows"].items():
        cls = K.CHUNK * K.class_of(int(ctx) + 16, int(a.max_len))
        meas[cls] = {int(r): v["ms"] for r, v in rows.items()}
    # the plain step of a class: its one-row verify scaled by the bench's plain/one-row ratio at the lowest class
    # (the decode step reads the same KV as a one-row block; one plain price for every class put a 128k step at
    # the 2k step's 7.4 ms against 23.9); every class from the lowest measured one up gets a row, the ones between
    # two measured classes interpolated in log context, past the last the last
    lo = min(meas)
    ratio = plain / meas[lo][1]
    ks = sorted(meas)
    out = {}
    import math
    for n in K.classes(int(a.max_len)):
        cls = K.CHUNK * n
        if cls < ks[0]:
            continue
        if cls in meas:
            tbl = meas[cls]
        elif cls > ks[-1]:
            # past the last measured class: its prices plus the extra KV a row block reads once (the 14 full
            # layers' bytes a token at the measured bus rate)
            extra = (cls - ks[-1]) / 2 * a.kv_bytes / (a.bus_gbs * 1e9) * 1e3
            tbl = {r: round(v + extra, 3) for r, v in meas[ks[-1]].items()}
        else:
            a0 = max(k for k in ks if k < cls)
            b0 = min(k for k in ks if k > cls)
            w = (math.log(cls) - math.log(a0)) / (math.log(b0) - math.log(a0))
            tbl = {r: round(meas[a0][r] + w * (meas[b0][r] - meas[a0][r]), 3) for r in meas[a0]}
        out["0" if cls == lo else str(cls)] = {"chain": {str(r): v for r, v in sorted(tbl.items())},
                                               "plain": {"1": round(tbl[1] * ratio, 3)}}
    out["step_ms"] = round(max(0.1, (stair["rows"][max(stair["rows"], key=int)]["16"]["ms"]
                                     - stair["rows"][max(stair["rows"], key=int)]["15"]["ms"])), 2)
    out["measured"] = (f"tools/flint_check.py staircase and bench on {os.path.basename(a.stair)}, "
                       f"{os.path.basename(a.bench)}: chain = a VerifyGraph replay of 1..16 rows (CUDA events, the "
                       "median), at the staircase's contexts (the lowest as class 0); plain = the captured decode "
                       "step's ms a token, host wall clock with a sync a token. No spec or prime prior: the cut "
                       "drafts and learns them online.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("engine")
    e.add_argument("--model", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--tokens", type=int, default=128)
    e.add_argument("--prompts", type=int, default=8)
    e.add_argument("--graph", default="on", choices=("on", "off"))
    e.add_argument("--max-len", type=int, default=8192)
    e.add_argument("--force", help="a run whose prompts and tokens are fed instead of the greedy ones (teacher-forced)")
    cp = sub.add_parser("cmp")
    cp.add_argument("--a", required=True)
    cp.add_argument("--b", required=True)
    cp.add_argument("--json")
    mb = sub.add_parser("mkbundle")
    mb.add_argument("--model", required=True)
    mb.add_argument("--out", required=True)
    mb.add_argument("--rows", type=int, default=8)
    mb.add_argument("--length", type=int, default=1024)
    mb.add_argument("--max-len", type=int, default=4096)
    bu = sub.add_parser("bundle")
    bu.add_argument("--model", required=True)
    bu.add_argument("--bundle", required=True)
    bu.add_argument("--max-len", type=int, default=4096)
    bu.add_argument("--json")
    b = sub.add_parser("bench")
    b.add_argument("--model", required=True)
    b.add_argument("--tokens", type=int, default=512)
    b.add_argument("--prefill", type=int, default=4096)
    b.add_argument("--max-len", type=int, default=8192)
    b.add_argument("--json")
    kv = sub.add_parser("kvfp8")
    kv.add_argument("--model", required=True)
    kv.add_argument("--bundle", required=True)
    kv.add_argument("--start", type=int, default=128)
    kv.add_argument("--block", type=int, default=16)
    kv.add_argument("--long", default="65536", help="prompt lengths, comma separated")
    kv.add_argument("--long-file", default="", help="a long text that does not repeat (the prompt's source)")
    kv.add_argument("--tail", type=int, default=512)
    kv.add_argument("--max-len", type=int, default=8192)
    kv.add_argument("--json")
    s = sub.add_parser("staircase")
    s.add_argument("--model", required=True)
    s.add_argument("--contexts", type=int, nargs="+", default=[2048, 16384])
    s.add_argument("--reps", type=int, default=20)
    s.add_argument("--max-len", type=int, default=16384)
    s.add_argument("--json")
    au = sub.add_parser("audit")
    au.add_argument("--pack", required=True)
    pr = sub.add_parser("prices")
    pr.add_argument("--stair", required=True)
    pr.add_argument("--bench", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--max-len", type=int, default=131072)
    pr.add_argument("--kv-bytes", type=int, default=28672, help="KV bytes a token (past the last measured class)")
    pr.add_argument("--bus-gbs", type=float, default=210.0, help="the KV read rate (128k decode on 4.0: 211 GB/s)")
    a = ap.parse_args()
    {"cmp": cmd_cmp, "audit": cmd_audit, "prices": cmd_prices, "engine": cmd_engine, "bundle": cmd_bundle,
     "mkbundle": cmd_mkbundle,
     "bench": cmd_bench, "staircase": cmd_staircase, "kvfp8": cmd_kvfp8}[a.cmd](a)


if __name__ == "__main__":
    main()
