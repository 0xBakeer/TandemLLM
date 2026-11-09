"""SPD-249 P1: Flash-Next's layer-major prefill windows against the chunk-major prefill, in one process.

    PYTHONPATH=<repo> python tools/fn_window_check.py --pack ~/flashnext-packs/3.05 \
        --ctx 2048,8192,32768 --windows 8192,32768 --json out.json

One engine (the weights once), twin slots (`FlashNextServing`) a prompt: one prefills chunk-major (QWEN38_FN_PF_WINDOW
0: `forward_pre` a chunk at a time), the other with windows of each `--windows` rows (`layer_major.window`). The
prompt is a class prompt of `--ctx` tokens (prose, code and chat in turn). Per prompt and window: the logits, every
row's streams (`hidden_post_norm`), every section (GDN states, conv tails, PLE conv and context, mtp.h_last), the K, V
and indexer-key rows, the pooled indexer keys and the length, each `torch.equal`; then `--steps` greedy decode steps
and a verify block with its commit on both twins, logits equal at each. With `--lines`, the slot verb the server's
cache loop calls (`FlashNextServing.prefill_window` with a state buffer a chunk line, then each line replayed as the
loop does) against the chunk-major twin's sections, length and streams at every line.

`--rot-rows`: the window arm also takes the pair rotation in row order (QWEN38_FN_PF_ROT_ROWS), the chunk-major arm
not, so the check holds both changes to the live prefill at once.

`--timing R`: per context and arm of `--timing-arms` ("window:rot", rot 1 = the row-order pair rotation; default the
chunk-major prefill and each window with it), R + 1 prefills on a fresh state (the first warms), a device sync
around each, the median; the allocator's peak above the slot's resident bytes for each.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.fn_generate import class_prompt  # noqa: E402
from tools.fn_joint_check import _twin_diff  # noqa: E402


def _decode_after(s_l, s_j, lg_l, steps: int) -> tuple:
    """`steps` greedy decode steps and a verify block with its commit on both twins: (steps equal, verify equal)."""
    nxt = int(lg_l.view(-1).argmax())
    eq, toks = [], [nxt]
    for _ in range(steps):
        x_l = s_l.forward(torch.tensor([nxt], device="cuda"), s_l.st.length)
        x_j = s_j.forward(torch.tensor([nxt], device="cuda"), s_j.st.length)
        eq.append(bool(torch.equal(x_l, x_j)))
        nxt = int(x_l.view(-1).argmax())
        toks.append(nxt)
    blk = [nxt, toks[1], toks[2], toks[3]]
    v_l = s_l.forward_tree(torch.tensor(blk, device="cuda"), [-1, 0, 1, 2], s_l.st.length).clone()
    v_j = s_j.forward_tree(torch.tensor(blk, device="cuda"), [-1, 0, 1, 2], s_j.st.length).clone()
    s_l.commit_tree([0, 1])
    s_j.commit_tree([0, 1])
    return eq, bool(torch.equal(v_l, v_j))


def _sections(s) -> dict:
    s._settle_chain()
    s.fn.flush(s.st)
    return {k: v.clone() for k, v in s.state.sections().items()}


@torch.no_grad()
def lines_check(fn, tok, L, ids: list, win: int, rot: bool = False) -> dict:
    """The slot verb with a state buffer at every chunk line, replayed as engine/cache.py's window loop replays it,
    against the chunk-major twin at each line."""
    from engine import cache
    from engine.models.flashnext import engine as E
    from engine.models.flashnext.serving import FlashNextServing
    from tools import fn_prefill_kernels as PK
    chunk = E.PREFILL_CHUNK
    s_l, s_j = FlashNextServing(fn, L), FlashNextServing(fn, L)
    t = torch.tensor(ids, device="cuda")
    plan = [(a, min(chunk, len(ids) - a)) for a in range(0, len(ids), chunk)]
    plan = plan[: max(2, win // chunk)]
    ref = []
    PK.ROT_ROWS = False
    for a, n in plan:                                          # chunk-major: what each line holds
        lg = s_l.forward(t[a:a + n], a).clone()
        ref.append({"secs": _sections(s_l), "length": s_l.st.length, "hidden": s_l.hidden_post_norm.clone(),
                    "logits": lg})
    assert s_j.window_ok(0), "window_ok refused a text prompt at 0"
    toks = [t[a:a + n] for a, n in plan]
    PK.ROT_ROWS = rot
    bufs = [cache.state_empty(s_j) for _ in plan]
    lg, hid, taps = s_j.prefill_window(toks, [a for a, _ in plan], bufs)
    cache.state_save(bufs[-1], s_j)
    out = []
    for c, (a, n) in enumerate(plan):                          # engine/cache.py's replay of the window's lines
        cache.state_load(s_j, bufs[c])
        s_j.kv.length = a + n
        s_j.replay_taps(taps[c])
        got = _sections(s_j)
        bad = [k for k in ref[c]["secs"] if not torch.equal(ref[c]["secs"][k], got[k])]
        out.append({"line": a + n, "sections_differ": bad, "length_equal": s_j.st.length == ref[c]["length"],
                    "hidden_equal": bool(torch.equal(hid[c], ref[c]["hidden"]))})
    r = {"window_rows": sum(n for _, n in plan), "chunks": len(plan), "lines": out,
         "logits_equal": bool(torch.equal(lg.view(-1), ref[-1]["logits"].view(-1))), "end": _twin_diff(s_l, s_j)}
    steps, ver = _decode_after(s_l, s_j, ref[-1]["logits"], 4)
    r.update(steps_equal=steps, verify_equal=ver, after=_twin_diff(s_l, s_j))
    r["all_equal"] = bool(r["logits_equal"] and r["end"]["equal"] and all(steps) and ver and r["after"]["equal"]
                          and all(not x["sections_differ"] and x["length_equal"] and x["hidden_equal"] for x in out))
    del s_l, s_j, bufs, ref
    torch.cuda.empty_cache()
    return r


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ctx", default="2048,8192,32768")
    ap.add_argument("--windows", default="8192,32768")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--timing", type=int, default=0)
    ap.add_argument("--lines", action="store_true")
    ap.add_argument("--rot-rows", action="store_true")
    ap.add_argument("--timing-arms", default="")
    ap.add_argument("--skip-check", action="store_true", help="timing only")
    ap.add_argument("--json")
    a = ap.parse_args()
    from engine.models.flashnext import engine as E
    from engine.models.flashnext import layer_major as LM
    from engine.models.flashnext.serving import FlashNextServing, build
    from tools import fn_prefill_kernels as PK
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.pack)
    rot0 = PK.ROT_ROWS
    ctxs = [int(x) for x in a.ctx.split(",") if x.strip()]
    wins = [int(x) for x in a.windows.split(",") if x.strip()]
    L = max(ctxs) + a.steps + 64 + 16
    eng, _, _ = build(a.pack, L, tok, spec=False, log=lambda s: print(s, flush=True))
    fn = eng.fn
    knob = LM.ROWS
    print(f"[win] chunk {E.PREFILL_CHUNK}, path {fn.prefill_path}, knob FN_PF_WINDOW {knob} FN_PF_ROT_ROWS {int(rot0)} "
          f"(the tool sets both itself), group rows {PK.GROUP_ROWS}, rot rows in the window arm {a.rot_rows}, "
          f"max_len {L}", flush=True)
    rep = {"chunk": E.PREFILL_CHUNK, "knob": knob, "rot_rows_knob": rot0, "group_rows": PK.GROUP_ROWS,
           "rot_rows_window_arm": a.rot_rows, "windows": wins, "rows": [], "lines": [], "timing": []}
    ok_all = True
    for k, n in enumerate(ctxs):
        ids = class_prompt(tok, ("prose", "code", "chat")[k % 3], n)[:n]
        t = torch.tensor(ids, device="cuda")
        for w in ([] if a.skip_check else wins):
            s_l, s_j = FlashNextServing(fn, L), FlashNextServing(fn, L)
            LM.ROWS, PK.ROT_ROWS = 0, False
            lg_l = s_l.forward(t, 0).clone()
            LM.ROWS, PK.ROT_ROWS = w, a.rot_rows
            lg_j = s_j.forward(t, 0).clone()
            LM.ROWS, PK.ROT_ROWS = knob, rot0
            r = {"ctx": len(ids), "window": w, "logits_equal": bool(torch.equal(lg_l, lg_j)),
                 "logits_max_diff": float((lg_l - lg_j).abs().max()),
                 "hidden_equal": bool(torch.equal(s_l.hidden_post_norm, s_j.hidden_post_norm)),
                 "argmax": [int(lg_l.view(-1).argmax()), int(lg_j.view(-1).argmax())]}
            r.update(_twin_diff(s_l, s_j))
            steps, ver = _decode_after(s_l, s_j, lg_l, a.steps)
            r.update(steps_equal=steps, verify_equal=ver, after=_twin_diff(s_l, s_j))
            r["all_equal"] = bool(r["logits_equal"] and r["hidden_equal"] and r["equal"] and all(steps) and ver
                                  and r["after"]["equal"])
            ok_all = ok_all and r["all_equal"]
            rep["rows"].append(r)
            print(f"[win] ctx {len(ids)} window {w}: logits {r['logits_equal']} hidden {r['hidden_equal']} state "
                  f"{r['equal']} (sections {r['sections_differ']} views {r['views_differ']} pool {r['pool_differ']} "
                  f"ctx {r['ple_ctx_differ']}) steps {steps} verify {ver} after {r['after']['equal']} "
                  f"max diff {r['logits_max_diff']}", flush=True)
            del s_l, s_j
            torch.cuda.empty_cache()
            if a.lines and len(ids) > E.PREFILL_CHUNK:
                LM.ROWS = w
                lr = lines_check(fn, tok, L, ids, w, a.rot_rows)
                LM.ROWS, PK.ROT_ROWS = knob, rot0
                lr.update(ctx=len(ids), window=w)
                ok_all = ok_all and lr["all_equal"]
                rep["lines"].append(lr)
                print(f"[win] lines ctx {len(ids)} window {w}: {lr['chunks']} chunks, all equal {lr['all_equal']} "
                      f"(logits {lr['logits_equal']} end {lr['end']['equal']} steps {lr['steps_equal']} verify "
                      f"{lr['verify_equal']} lines {[(x['line'], x['sections_differ'], x['length_equal'], x['hidden_equal']) for x in lr['lines']]})",
                      flush=True)
        if a.timing:
            s = FlashNextServing(fn, L)
            row = {"ctx": len(ids)}
            arms = ([tuple(int(v) for v in x.split(":")) for x in a.timing_arms.split(",") if x.strip()]
                    or [(0, 0)] + [(w, int(a.rot_rows)) for w in wins])
            for w, rot in arms:
                if w and len(ids) <= E.PREFILL_CHUNK:
                    continue
                LM.ROWS, PK.ROT_ROWS = w, bool(rot)
                ts, peaks = [], []
                for i in range(a.timing + 1):
                    s.kv.length = 0
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    s.forward(t, 0)
                    torch.cuda.synchronize()
                    if i:
                        ts.append(time.perf_counter() - t0)
                        peaks.append((torch.cuda.max_memory_allocated() - base) / (1 << 30))
                LM.ROWS, PK.ROT_ROWS = knob, rot0
                med = sorted(ts)[len(ts) // 2]
                row[f"w{w}r{rot}"] = {"s": [round(x, 3) for x in ts], "median_s": round(med, 3),
                                      "tok_s": round(len(ids) / med, 1), "peak_gib": round(max(peaks), 3)}
                print(f"[win] timing ctx {len(ids)} window {w} rot rows {rot}: {row[f'w{w}r{rot}']}", flush=True)
            rep["timing"].append(row)
            del s
            torch.cuda.empty_cache()
    rep["all_equal"] = ok_all
    print(f"[win] ALL EQUAL {ok_all}", flush=True)
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
