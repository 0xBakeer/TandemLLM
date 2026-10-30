"""SPD-233 on the board: where a NemotronH round of N lanes goes, in process (no server).

    python tools/nh_round_profile.py --model ~/models/nemotron-3.5-bolt-74b-a9b-exl3-4.0 \\
        --lilicorr ~/models/nemotron-3.5-bolt-lilicorr --lanes 8 --rows 4 --rounds 12 --out prof.json

N serving slots over one engine (slot 0 built as the server builds it: taps, self-check, LiLiCorr gate), each
prefilled with its own prose prompt. A round is the served round's GPU work in its order: every lane's draft
(LiLiCorr, one graph a lane, or `--pack on`: one packed draft), the joint verify of every lane's anchor and its
first `--rows` - 1 drafted tokens (`forward_tree_multi`), the greedy walk, every lane's commit. Per phase: host
ms, GPU ms (CUDA events), and over the `--profile` last rounds the kernels the phase ran (torch profiler, the
kernels bucketed by the record_function range their GPU time falls in): busy ms, launches, the top kernels and
a roll-up by part (experts, EXL3 projections, Mamba2, attention, router/combine, norms, other).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPTS = [
    "Write a detailed essay on the history of the printing press and its effect on literacy in Europe.",
    "Explain how a refrigerator works, step by step, to a curious teenager.",
    "Describe a long walk through Kyoto in autumn, street by street.",
    "Write a letter to an old friend about learning to cook during a hard winter.",
    "Explain why the moon has phases and why eclipses are rare.",
    "Describe how bread is made, from the wheat field to the table.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain the rules of chess to a beginner, with an example game.",
]

PARTS = [("experts", r"relu2|moe_groups|moe_gu|moe_down"), ("exl3", r"dense_rows|dense_kernel|dense_rot|gemm_kernel|recon|rot_in|rot_out"),
         ("mamba", r"_ssm_walk|_ssm_rows|_conv_rows|_gated_norm"), ("attn", r"attn"), ("router_combine", r"_router|_combine|sort|radix"),
         ("norm", r"_add_rms|rms"), ("gemm_other", r"gemm|gemv|cutlass|nvjet|sm\d+_xmma|ampere|hopper|w4a16|nvfp4"),
         ("copy_index", r"copy|index|cat|gather|scatter|fill|elementwise|vectorized|reduce|argmax|arange")]


def part_of(name: str) -> str:
    for p, rx in PARTS:
        if re.search(rx, name, re.I):
            return p
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lilicorr", required=True)
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--rows", type=int, default=4, help="rows a lane in the joint verify (anchor + drafts)")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--profile", type=int, default=4, help="the last rounds under the torch profiler")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--draft", default="on", choices=("on", "off"),
                    help="off: no draft (with --rows 1 a round is the joint plain decode step alone)")
    ap.add_argument("--pack", default="off", choices=("on", "off"))
    ap.add_argument("--check", default="off", choices=("on", "off"),
                    help="with --pack on: a lone twin head a lane drafts from the same rows first; per round whether "
                         "every lane's drafted tokens are its twin's (a draft only proposes: a speed matter)")
    ap.add_argument("--joint-check", default="on", choices=("on", "off"),
                    help="at the second round (untimed) each lane's block verified alone first (its lone pass, a captured graph): "
                         "the joint pass's logits of every lane must equal them bit for bit")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from engine.models.nemotron_h.serving import NemotronHServing, build
    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    srv, info = build(a.model, a.max_len, tok, lilicorr=a.lilicorr, log=lambda s: print(s, flush=True))
    print(f"[prof] built in {time.time() - t0:.1f}s, lilicorr {info['lilicorr']}", flush=True)
    eng = srv.fn
    slots = [srv]
    for i in range(1, a.lanes):
        s = NemotronHServing(eng, a.max_len, slot=f"p{i}")
        s.add_lilicorr(model=srv.heads["lilicorr"].m)
        slots.append(s)
    anchors = []
    for s, p in zip(slots, PROMPTS * 4):
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"])
        lg = s.forward(ids, 0)
        anchors.append(int(lg.view(-1).argmax()))
    torch.cuda.synchronize()
    heads = [s.heads["lilicorr"] for s in slots]
    from engine.drafters import lilicorr as LC
    from engine.methods import draft_meet
    LC.PACK = a.pack == "on"
    twins = [LC.LiLiCorrHead(s, srv.heads["lilicorr"].m) for s in slots] if a.check == "on" else None
    agree = {"rounds": 0, "lanes_top1_equal": 0, "lanes": 0, "slots_equal": 0, "slots": 0, "max_prob_diff": 0.0,
             "lone_ms": [], "packed_ms": []}

    phases = ("draft", "verify", "walk", "commit")
    host = {p: [] for p in phases}
    gpu = {p: [] for p in phases}
    period, accepted = [], []
    prof = None
    joint_check = None
    tmp = tempfile.mkdtemp()
    for r in range(a.rounds):
        if r == a.rounds - a.profile:
            torch.cuda.synchronize()
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                      torch.profiler.ProfilerActivity.CUDA])
            prof.__enter__()
        ev = [torch.cuda.Event(enable_timing=True)]
        ev[0].record()
        hs = {}
        if twins is not None:
            torch.cuda.synchronize()
            tl = time.perf_counter()
            for h, x in zip(twins, anchors):
                h.launch(None, [x])
            lone = [h.read() for h in twins]
            torch.cuda.synchronize()
            agree["lone_ms"].append(1e3 * (time.perf_counter() - tl))
        # draft
        t = time.perf_counter()
        with torch.profiler.record_function("ph:draft"):
            if a.draft == "off":
                drafts = [[] for _ in heads]
            elif a.pack == "on":
                got = []
                draft_meet.MEET = lambda h: (got.append(h) or True)
                try:
                    for h, s, x in zip(heads, slots, anchors):
                        h.launch(None, [x])
                finally:
                    draft_meet.MEET = None
                type(got[0]).run_packed(got)
            else:
                for h, x in zip(heads, anchors):
                    h.launch(None, [x])
            if a.draft == "on":
                drafts = [h.read()[0] for h in heads]
        hs["draft"] = time.perf_counter() - t
        if twins is not None:
            torch.cuda.synchronize()
            agree["packed_ms"].append(1e3 * (time.perf_counter() - t))
            got = [h._out.tolist() for h in heads]
            for (lt, lp), d, g in zip(lone, drafts, got):
                agree["lanes"] += 1
                agree["lanes_top1_equal"] += int([q[0] for q in lt] == [q[0] for q in d])
                agree["slots"] += len(lt)
                agree["slots_equal"] += sum(int(q[0] == w[0]) for q, w in zip(lt, d))
                agree["max_prob_diff"] = max(agree["max_prob_diff"], max(abs(p[0] - r[1]) for p, r in zip(lp, g)))
            agree["rounds"] += 1
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        ev.append(e)
        # verify
        t = time.perf_counter()
        with torch.profiler.record_function("ph:verify"):
            items = []
            for s, x, d in zip(slots, anchors, drafts):
                toks = [x] + [int(q[0]) for q in d[:a.rows - 1]]
                items.append((s, torch.tensor(toks, device="cuda"), list(range(-1, len(toks) - 1)), s.st.length))
            if r == 1 and a.joint_check == "on":
                lone_lg = [s.forward_tree(blk, par, st0).clone() for s, blk, par, st0 in items]
            outs = NemotronHServing.forward_tree_multi(items)
            if r == 1 and a.joint_check == "on":
                jc = [bool(torch.equal(x, y)) for x, y in zip(lone_lg, outs)]
                joint_check = {"lanes_equal": sum(jc), "lanes": len(jc), "rows": [len(i[2]) for i in items],
                               "max_abs_diff": max(float((x - y).abs().max()) for x, y in zip(lone_lg, outs))}
                print(f"[prof] joint check: {joint_check}", flush=True)
        hs["verify"] = time.perf_counter() - t
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        ev.append(e)
        # walk (greedy: the longest prefix the picks agree with)
        t = time.perf_counter()
        with torch.profiler.record_function("ph:walk"):
            paths = []
            for s, (_, blk, _, _) in zip(slots, items):
                picks = s.picks.tolist()
                b = blk.tolist()
                n = 0
                while n + 1 < len(b) and b[n + 1] == picks[n]:
                    n += 1
                paths.append((list(range(n + 1)), picks[n]))
        hs["walk"] = time.perf_counter() - t
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        ev.append(e)
        t = time.perf_counter()
        with torch.profiler.record_function("ph:commit"):
            for s, (path, _) in zip(slots, paths):
                s.commit_tree(path)
        hs["commit"] = time.perf_counter() - t
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        ev.append(e)
        torch.cuda.synchronize()
        anchors = [nx for _, nx in paths]
        accepted.append(sum(len(p) for p, _ in paths))
        if r >= 2:
            for p, x, (a0, a1) in zip(phases, (hs[p] for p in phases), zip(ev, ev[1:])):
                host[p].append(1e3 * x)
                gpu[p].append(a0.elapsed_time(a1))
            period.append(ev[0].elapsed_time(ev[-1]))
    trace = os.path.join(tmp, "trace.json")
    prof.__exit__(None, None, None)
    prof.export_chrome_trace(trace)
    tr = json.load(open(trace))["traceEvents"]
    ann = [x for x in tr if x.get("cat") == "gpu_user_annotation" and str(x.get("name", "")).startswith("ph:")]
    kern = [x for x in tr if x.get("cat") == "kernel"]
    cpu_ann = [x for x in tr if x.get("cat") == "user_annotation" and str(x.get("name", "")).startswith("ph:")]
    launches = [x for x in tr if x.get("cat") == "cuda_runtime" and re.search(r"LaunchKernel|GraphLaunch|launchKernel",
                                                                             str(x.get("name", "")))]
    per = {}
    for k in kern:
        mid = k["ts"] + k.get("dur", 0) / 2
        ph = next((x["name"][3:] for x in ann if x["ts"] <= mid <= x["ts"] + x.get("dur", 0)), "outside")
        d = per.setdefault(ph, {"busy_ms": 0.0, "kernels": 0, "parts": {}, "top": {}})
        ms = k.get("dur", 0) / 1e3
        d["busy_ms"] += ms
        d["kernels"] += 1
        pa = part_of(k["name"])
        pp = d["parts"].setdefault(pa, [0.0, 0])
        pp[0] += ms
        pp[1] += 1
        tp = d["top"].setdefault(k["name"][:90], [0.0, 0])
        tp[0] += ms
        tp[1] += 1
    for x in launches:
        ph = next((c["name"][3:] for c in cpu_ann if c["ts"] <= x["ts"] <= c["ts"] + c.get("dur", 0)), "outside")
        d = per.setdefault(ph, {"busy_ms": 0.0, "kernels": 0, "parts": {}, "top": {}})
        d["launch_calls"] = d.get("launch_calls", 0) + 1
    n = max(1, a.profile)
    for d in per.values():
        d["busy_ms"] = round(d["busy_ms"] / n, 2)
        d["kernels"] = round(d["kernels"] / n, 1)
        d["launch_calls"] = round(d.get("launch_calls", 0) / n, 1)
        d["parts"] = {k: [round(v[0] / n, 2), round(v[1] / n, 1)] for k, v in sorted(d["parts"].items(), key=lambda kv: -kv[1][0])}
        d["top"] = {k: [round(v[0] / n, 2), round(v[1] / n, 1)]
                    for k, v in sorted(d["top"].items(), key=lambda kv: -kv[1][0])[:14]}

    def med(xs):
        xs = sorted(xs)
        return round(xs[len(xs) // 2], 2) if xs else None
    res = {"lanes": a.lanes, "rows_a_lane": a.rows, "pack": a.pack, "rounds_timed": len(period),
           "period_ms": med(period), "tokens_a_round": round(sum(accepted[2:]) / max(1, len(accepted) - 2), 2),
           "host_ms": {p: med(v) for p, v in host.items()}, "gpu_ms": {p: med(v) for p, v in gpu.items()},
           "profiled": per, "joint_check": joint_check, "check": ({k: (round(sorted(v)[len(v) // 2], 2) if isinstance(v, list) and v else v)
                                       for k, v in agree.items()} if twins is not None else None), "lilicorr": info["lilicorr"] and {k: v for k, v in info["lilicorr"].items() if k != "gate"}}
    res["tok_s"] = round(1e3 * res["tokens_a_round"] / res["period_ms"], 1)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "profiled"}), flush=True)
    for ph in phases:
        d = per.get(ph) or {}
        print(f"[prof] {ph}: busy {d.get('busy_ms')} ms, {d.get('kernels')} kernels, {d.get('launch_calls')} launch calls; "
              f"parts {d.get('parts')}", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
