"""Flash-Next's joint verify against each state's lone verify, in one process (ENG-241 step 3).

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_joint_check.py --pack ~/flashnext-packs/3.05 \
        --tokens 3000,7000,11000 --rounds 12 --json out.json

One engine (the weights once) and two states a prompt: one runs every round alone (`forward_tree`,
with its graphs), the other runs the same rounds together with the other prompts' states
(`forward_tree_multi`, one eager pass). Each round every state gets the same block: its own greedy
continuation as a chain of `--block` rows on even rounds, a tree with a second branch on odd rounds,
and commits the same path (the walk of its lone verify). Per round: whether every state's logits in
the joint pass are its lone logits bit for bit; at the end, each state's recurrent sections and KV
rows against its lone twin.

`--timing R`: then R more rounds of chain blocks, timed with a device sync around each: every state's
lone verify and commit from its graphs (what the slots do alone, summed), the same eagerly (no graphs),
and the joint pass with its commits; and the joint pass's GPU time by part (`FlashNextEngine.
joint_timing`: staging, PLE/norms/mixers, projections, GDN recurrence, attention, experts, head).
`--kernels N`: then one more joint pass with its commits under torch.profiler: the wall time, the GPU's
kernel time (busy) and the top N kernels by summed time. `--lone-kernels N`: the same for the first
state's lone verify and commit (its graphs, as a slot alone runs them), at every block width 2..`--block`.
`--dump-moe F` (SPD-249): one more joint pass, every layer's expert rows and ids saved to F (real routing for
tools/fn_moe_group_check.py --replay).
`--commit-multi` (SPD-239): the joint states commit in one call (`FlashNextServing.commit_tree_multi`, every
lane's copies in one launch); their next rounds and end states are then held to the lone twins' own commits.
`--joint-prefill` (SPD-249): instead of the verify rounds, the joint prefill of a burst against each prompt's lone
prefill (`FlashNextServing.prefill_joint` vs `forward`), on twin states. Bursts: the prompts of `--prompts-json` (a
bench JSON's first row's `prompts`, tokenized as the server does) if given, then `--tokens` (class prompts of those
lengths, with a one-token prompt and a prompt resumed past a prefix), then three prompts of `--long` rows (more than
FN_PREFILL_CHUNK rows: cut into runs). Per prompt: logits, streams (`hidden_post_norm`), every section (GDN states,
conv tails, PLE conv and context, mtp.h_last), the K, V and indexer-key rows, the pooled indexer keys and the length,
each `torch.equal`; then `--steps` greedy decode steps and a verify block with its commit on both twins, logits
equal at each. `--timing R`: the first burst's lone prefills (summed) and its joint prefill, R times each on fresh
states, a device sync around each (median s).
`--graph` (SPD-249): the joint passes from the joint graphs (QWEN38_FN_JOINT_GRAPH, captures allowed every pass):
each lane's block width varies by round (1..`--block` rows, a tree on odd rounds) and every fourth round only the
first five lanes run, so the keys' padding, the lanes' row cuts and other lane sets are replayed; per round whether it
ran from a graph, its pad rows' expert ids (all -1) and its real rows' (none -1). The timing then alternates eager and
graph passes (`joint_eager_ms`, `joint_graph_ms`, without the part marks).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.methods import walk  # noqa: E402
from engine.tree import DraftTree  # noqa: E402
from tools.fn_generate import class_prompt  # noqa: E402


def block_of(cont: list[int], at: int, n: int, tree: bool) -> tuple[list[int], list[int]]:
    """Row 0 the next token, then the continuation as a chain; a tree adds a second child of row 1
    (a token that is not the continuation) before the chain goes on."""
    toks = cont[at:at + n]
    parents = list(range(-1, len(toks) - 1))
    if tree and len(toks) >= 3:
        toks = toks[:2] + [(toks[2] + 17) % 200000] + toks[2:]
        parents = [-1, 0, 1, 1] + list(range(3, len(toks) - 1))
    return toks, parents


def _twin_diff(s_l, s_j) -> dict:
    """SPD-249: where two slots' states differ: the sections (GDN states, conv tails, PLE conv and context,
    mtp.h_last), the K / V / indexer-key rows below the length, the pooled indexer keys below the length's blocks and
    their counts, the length."""
    s_l._settle_chain()
    s_j._settle_chain()
    a, b = s_l.st, s_j.st
    s_l.fn.flush(a)                                           # parked GDN rows (the folded commit) into the states
    s_j.fn.flush(b)
    secs_l, secs_j = s_l.state.sections(), s_j.state.sections()
    bad = [k for k in secs_l if not torch.equal(secs_l[k], secs_j[k])]
    views = [n for (n, t, ax), (_, u, _) in zip(s_l.state.views(), s_j.state.views())
             if not torch.equal(t.narrow(ax, 0, a.length), u.narrow(ax, 0, b.length))]
    nb = a.length // a.pool_ratio
    pool = [li for li in sorted(a.pool) if not torch.equal(a.pool[li][:nb], b.pool[li][:nb])]
    ctx = [li for li in sorted(set(a.ple_ctx) | set(b.ple_ctx))
           if li not in a.ple_ctx or li not in b.ple_ctx or not torch.equal(a.ple_ctx[li], b.ple_ctx[li])]
    return {"length": [a.length, b.length], "sections_differ": bad, "views_differ": views, "pool_differ": pool,
            "pool_n_equal": a.pool_n == b.pool_n, "ple_ctx_differ": ctx,
            "equal": a.length == b.length and not bad and not views and not pool and a.pool_n == b.pool_n and not ctx}


@torch.no_grad()
def joint_prefill_check(a) -> None:
    """SPD-249: `--joint-prefill` (the module docstring)."""
    from engine.models.flashnext import engine as E
    from engine.models.flashnext import joint_prefill as JP
    from engine.models.flashnext.serving import FlashNextServing, build
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.pack)
    bursts = []
    if a.prompts_json:
        texts = json.load(open(a.prompts_json))["rows"][0]["prompts"]
        bursts.append(("bench", [(tok(t, add_special_tokens=False)["input_ids"], 0) for t in texts]))
    lens = [int(x) for x in a.tokens.split(",") if x.strip()]
    if lens:
        mixed = [(class_prompt(tok, ("prose", "code", "chat")[i % 3], n)[:n], 0) for i, n in enumerate(lens)]
        mixed.append((class_prompt(tok, "chat", 3)[:1], 0))                       # a one-token prompt (start 0)
        body = class_prompt(tok, "prose", 700)[:700]
        mixed.append((body, 500))                                                # rows 500.. after a 500-row prefix
        bursts.append(("mixed", mixed))
    if a.long:
        bursts.append(("long", [(class_prompt(tok, ("code", "prose", "chat")[i], a.long)[:a.long], 0) for i in range(3)]))
    L = max(len(ids) for _, b in bursts for ids, _ in b) + a.steps + 64 + 16
    eng, _, _ = build(a.pack, L, tok, spec=False, log=lambda s: print(s, flush=True))
    fn = eng.fn
    print(f"[jp] chunk {E.PREFILL_CHUNK}, path {fn.prefill_path}, knob FN_JOINT_PREFILL {int(JP.ON)} (the tool calls the "
          f"joint prefill itself), max_len {L}", flush=True)
    report = {"bursts": [], "timing": None, "chunk": E.PREFILL_CHUNK}
    ok_all = True
    for name, burst in bursts:
        lone = [FlashNextServing(fn, L) for _ in burst]
        joint = [FlashNextServing(fn, L) for _ in burst]
        items, lone_lg = [], []
        for (ids, start), s_l, s_j in zip(burst, lone, joint):
            t = torch.tensor(ids, device="cuda")
            if start:                                     # the prefix alone on both twins, then the rest
                s_l.forward(t[:start], 0)
                s_j.forward(t[:start], 0)
            assert s_j.prefill_joint_ok(len(ids) - start, start) or not JP.ON
            assert JP.ok(fn, len(ids) - start, start, s_j.st), (name, len(ids), start)
            lone_lg.append(s_l.forward(t[start:], start).clone())
            items.append((s_j, t[start:], start))
        out = FlashNextServing.prefill_joint(items)
        torch.cuda.synchronize()
        rows = []
        for k, ((ids, start), s_l, s_j, lg_l, lg_j) in enumerate(zip(burst, lone, joint, lone_lg, out)):
            r = {"rows": len(ids) - start, "start": start, "logits_equal": bool(torch.equal(lg_l, lg_j)),
                 "logits_max_diff": float((lg_l - lg_j).abs().max()),
                 "hidden_equal": bool(torch.equal(s_l.hidden_post_norm, s_j.hidden_post_norm)),
                 "argmax": [int(lg_l.view(-1).argmax()), int(lg_j.view(-1).argmax())]}
            r.update(_twin_diff(s_l, s_j))
            # then the same greedy decode steps and one verify block + commit on both twins
            nxt = int(lg_l.view(-1).argmax())
            steps_eq, toks = [], [nxt]
            for _ in range(a.steps):
                x_l = s_l.forward(torch.tensor([nxt], device="cuda"), s_l.st.length)
                x_j = s_j.forward(torch.tensor([nxt], device="cuda"), s_j.st.length)
                steps_eq.append(bool(torch.equal(x_l, x_j)))
                nxt = int(x_l.view(-1).argmax())
                toks.append(nxt)
            blk = [nxt, toks[1], toks[2], toks[3]]                    # a chain block (its tokens need not be greedy)
            v_l = s_l.forward_tree(torch.tensor(blk, device="cuda"), [-1, 0, 1, 2], s_l.st.length).clone()
            v_j = s_j.forward_tree(torch.tensor(blk, device="cuda"), [-1, 0, 1, 2], s_j.st.length).clone()
            s_l.commit_tree([0, 1])
            s_j.commit_tree([0, 1])
            r.update(steps_equal=steps_eq, verify_equal=bool(torch.equal(v_l, v_j)), after=_twin_diff(s_l, s_j))
            r["all_equal"] = bool(r["logits_equal"] and r["hidden_equal"] and r["equal"] and all(steps_eq)
                                  and r["verify_equal"] and r["after"]["equal"])
            ok_all = ok_all and r["all_equal"]
            rows.append(r)
            print(f"[jp] {name} {k}: rows {r['rows']} start {start}: logits {r['logits_equal']} hidden "
                  f"{r['hidden_equal']} state {r['equal']} (sections {r['sections_differ']} views {r['views_differ']} "
                  f"pool {r['pool_differ']} ctx {r['ple_ctx_differ']}) steps {steps_eq} verify {r['verify_equal']} "
                  f"after {r['after']['equal']} max diff {r['logits_max_diff']}", flush=True)
        report["bursts"].append({"name": name, "prompts": len(burst), "rows": sum(len(i) - s for i, s in burst),
                                 "runs": JP.runs([len(i) - s for i, s in burst], E.PREFILL_CHUNK), "per_prompt": rows,
                                 "all_equal": all(r["all_equal"] for r in rows)})
        del lone, joint, items, out
        torch.cuda.empty_cache()
    if a.timing and bursts:
        # the first burst's lone prefills (one after another, summed) against its joint prefill, on fresh states
        name, burst = bursts[0]
        ss = [FlashNextServing(fn, L) for _ in burst]
        ids_ = [torch.tensor(i, device="cuda") for i, _ in burst]
        tl, tj = [], []
        for rep in range(a.timing + 1):
            for s in ss:
                s.kv.length = 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for s, t in zip(ss, ids_):
                s.forward(t, 0)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for s in ss:
                s.kv.length = 0
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            FlashNextServing.prefill_joint([(s, t, 0) for s, t in zip(ss, ids_)])
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            if rep:                                          # the first pair warms
                tl.append(t1 - t0)
                tj.append(t3 - t2)
        med = lambda v: sorted(v)[len(v) // 2]               # noqa: E731
        report["timing"] = {"burst": name, "prompts": len(burst), "rows": sum(i.numel() for i in ids_),
                            "lone_sum_s": [round(x, 4) for x in tl], "joint_s": [round(x, 4) for x in tj],
                            "lone_sum_median_s": round(med(tl), 4), "joint_median_s": round(med(tj), 4)}
        print(f"[jp] timing {report['timing']}", flush=True)
    report["all_equal"] = ok_all
    print(f"[jp] ALL EQUAL {ok_all}", flush=True)
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokens", default="3000,7000,11000")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--timing", type=int, default=0)
    ap.add_argument("--moe-group", action="store_true", help="the grouped expert kernels (tools/fn_kernels.py MOE_GROUP)")
    ap.add_argument("--arch", default="flashnext", choices=("flashnext", "qwen3_5_moe"),
                    help="the family whose joint pass is checked (qwen3_5_moe: Ornith)")
    ap.add_argument("--kernels", type=int, default=0, help="one more joint pass under torch.profiler: top N kernels")
    ap.add_argument("--lone-kernels", type=int, default=0, help="the first state's lone verify + commit by width: top N")
    ap.add_argument("--commit-multi", action="store_true",
                    help="SPD-239: the joint states commit in one call (FlashNextServing.commit_tree_multi)")
    ap.add_argument("--dump-moe", help="SPD-249: one more joint pass; every layer's expert input rows and ids to this "
                    "file (torch.save), for tools/fn_moe_group_check.py --replay")
    ap.add_argument("--graph", action="store_true", help="SPD-249: the joint passes from the joint graphs")
    ap.add_argument("--joint-prefill", action="store_true", help="SPD-249: joint prefill vs each prompt's lone prefill")
    ap.add_argument("--prompts-json", help="SPD-249: a bench JSON whose first row's prompts are the first burst")
    ap.add_argument("--long", type=int, default=1800, help="SPD-249: the rows of the last burst's three prompts")
    ap.add_argument("--steps", type=int, default=6, help="SPD-249: greedy decode steps after the prefill")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.joint_prefill:
        return joint_prefill_check(a)
    global FlashNextServing, build
    if a.arch == "qwen3_5_moe":
        # SPD-217: Ornith's trunk (Qwen3.5 MoE), the same checks over `Qwen35MoeEngine.verify_multi`
        from engine.models.qwen3_5_moe.serving import Qwen35MoeServing as FlashNextServing, build
    else:
        from engine.models.flashnext.serving import FlashNextServing, build
    if a.moe_group:
        from tools import fn_kernels
        fn_kernels.MOE_GROUP = True
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.pack)
    lens = [int(x) for x in a.tokens.split(",")]
    L = max(lens) + (a.rounds + a.timing + (3 * a.block if a.lone_kernels else 0) + (1 if a.dump_moe else 0)) * (a.block + 1) + 1024
    eng, _, _ = build(a.pack, L, tok, spec=False, log=lambda s: print(s, flush=True))
    fn = eng.fn
    # one state a length: the kinds in turn (prose, code, chat, prose, ...), so 8 lengths are 8 states
    prompts = [torch.tensor(class_prompt(tok, ("prose", "code", "chat")[i % 3], n), device="cuda")
               for i, n in enumerate(lens)]
    lone = [eng] + [FlashNextServing(fn, L) for _ in prompts[1:]]
    joint = [FlashNextServing(fn, L) for _ in prompts]
    eager = [FlashNextServing(fn, L) for _ in prompts] if a.timing else []
    conts, at = [], []
    for s_l, s_j, ids in zip(lone, joint, prompts):
        lg = s_l.forward(ids, 0)
        s_j.forward(ids, 0)
        if eager:
            eager[len(conts)].forward(ids, 0)
        # the continuation the blocks are made of: greedy from a third state
        tmp = FlashNextServing(fn, L)
        x = tmp.forward(ids, 0)
        cont = [int(x.view(-1).argmax())]
        for i in range((a.rounds + a.timing + (3 * a.block if a.lone_kernels else 0) + (1 if a.dump_moe else 0)) * (a.block + 1)):
            x = tmp.forward(torch.tensor([cont[-1]], device="cuda"), ids.numel() + i)
            cont.append(int(x.view(-1).argmax()))
        del tmp
        conts.append(cont)
        at.append(0)
        assert int(lg.view(-1).argmax()) == cont[0]
    def commit_joint(states, paths_):
        # the joint pass's states keep their paths: each its own commit_tree, or all in one call
        if a.commit_multi:
            FlashNextServing.commit_tree_multi(list(zip(states, paths_)))
        else:
            for s_, p_ in zip(states, paths_):
                s_.commit_tree(p_)

    E_mod = None
    if a.graph:
        from engine.models.flashnext import engine as E_mod
        E_mod.JOINT_GRAPH = True

    def jstats():
        return json.loads(json.dumps(fn.joint_graph_stats())) if a.graph else {}

    rounds = []
    for r in range(a.rounds):
        on = list(range(len(prompts)))
        if a.graph:
            if r % 4 == 3:
                on = on[:5]
            blocks = {j: block_of(conts[j], at[j], 1 + (r + j) % a.block, r % 2 == 1) for j in on}
            joint[0].joint_capture_ok = True
        else:
            blocks = {j: block_of(conts[j], at[j], a.block, r % 2 == 1) for j in on}
        g0 = jstats()
        lone_out, paths = {}, {}
        for j in on:
            s = lone[j]
            toks, parents = blocks[j]
            lg = s.forward_tree(torch.tensor(toks, device="cuda"), parents, s.st.length).clone()
            path, new = walk.accept_tree(DraftTree(tokens=toks, parents=parents), lg.argmax(-1).tolist())
            s.commit_tree(path)
            lone_out[j] = lg
            paths[j] = path
        outs = FlashNextServing.forward_tree_multi(
            [(joint[j], torch.tensor(blocks[j][0], device="cuda"), blocks[j][1], joint[j].st.length) for j in on])
        same = [bool(torch.equal(lone_out[j], o)) for j, o in zip(on, outs)]
        diff = [float((lone_out[j] - o).abs().max()) for j, o in zip(on, outs)]
        rows = sum(len(blocks[j][0]) for j in on)
        info = {}
        if a.graph:
            g1 = jstats()
            info["graph"] = g1["replays"] > g0["replays"]
            info["captured"] = g1["captures"] > g0["captures"]
            info["eager"] = {k: v - g0["eager"].get(k, 0) for k, v in g1["eager"].items() if v != g0["eager"].get(k, 0)}
            if info["graph"]:
                # the pad rows route to no expert, the real rows to their own (SPD-249: U and every expert's rows are
                # the unpadded pass's)
                R = next(x for x in E_mod.JG_ROWS if x >= rows)
                graphs = fn._jg_state()["graphs"]                  # the one this pass replayed: used last
                key = next(k for k in reversed(list(graphs)) if k[:3] == (len(on), R, 0))
                G = graphs[key]
                ids = [G.t[("ids", li)] for li in range(fn.cfg.num_hidden_layers)]
                info["pad_ids_masked"] = all(bool((t[rows:] == -1).all()) for t in ids)
                info["real_ids_routed"] = all(bool((t[:rows] >= 0).all()) for t in ids)
                info["pad_rows"] = key[1] - rows
        commit_joint([joint[j] for j in on], [paths[j] for j in on])
        for j in on:
            at[j] += len(paths[j])
        rounds.append({"round": r, "lanes": len(on), "rows": rows, "tree": r % 2 == 1, "equal": same, "max_diff": diff,
                       **info})
        print(f"[joint] round {r}: {len(on)} lanes, {rows} rows, equal {same}, max diff {diff} {info}", flush=True)
    timing = None
    if a.timing:
        tl = te = tj = 0.0
        tg, ng_ = [0.0, 0.0], [0, 0]
        fn.joint_timing = None if a.graph else {}
        cap0 = jstats().get("captures", 0)
        for r in range(a.timing):
            if a.graph:
                E_mod.JOINT_GRAPH = bool(r % 2)                 # eager and graph passes in turn
                joint[0].joint_capture_ok = True
            blocks = [block_of(conts[j], at[j], a.block, False) for j in range(len(prompts))]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for j, s in enumerate(lone):
                s.forward_tree(torch.tensor(blocks[j][0], device="cuda"), blocks[j][1], s.st.length)
                s.commit_tree(list(range(len(blocks[j][0]))))
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for j, s in enumerate(eager):                     # the engine's verify and commit, no graphs
                fn.verify(blocks[j][0], blocks[j][1], s.st)
                fn.commit(s.st, list(range(len(blocks[j][0]))))
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            FlashNextServing.forward_tree_multi(
                [(s, torch.tensor(blocks[j][0], device="cuda"), blocks[j][1], s.st.length) for j, s in enumerate(joint)])
            commit_joint(joint, [list(range(len(blocks[j][0]))) for j in range(len(joint))])
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            if r:                                               # the first round warms
                tl, te, tj = tl + t1 - t0, te + t2 - t1, tj + t3 - t2
            if a.graph and r >= 2 and jstats()["captures"] == cap0:
                tg[r % 2] += t3 - t2                            # no pass that captured
                ng_[r % 2] += 1
            cap0 = jstats().get("captures", 0)
            for j in range(len(prompts)):
                at[j] += len(blocks[j][0])
        n = max(1, a.timing - 1)
        npass = fn.joint_timing.get("passes") if fn.joint_timing is not None else None
        jt = {k: round(v / npass, 2) for k, v in fn.joint_timing.items() if k != "passes"} if npass else {}
        timing = {"states": len(prompts), "rows": len(prompts) * a.block, "lone_graphs_ms": round(tl / n * 1e3, 2),
                  "lone_eager_ms": round(te / n * 1e3, 2), "joint_ms": round(tj / n * 1e3, 2), "joint_parts_ms": jt}
        if a.graph:
            E_mod.JOINT_GRAPH = True
            timing.update(joint_eager_ms=round(tg[0] / max(1, ng_[0]) * 1e3, 2),
                          joint_graph_ms=round(tg[1] / max(1, ng_[1]) * 1e3, 2), graphs=jstats())
        fn.joint_timing = None
        print(f"[joint] timing {timing}", flush=True)
    kern = None
    if a.kernels:
        from torch.profiler import ProfilerActivity, profile
        blocks = [block_of(conts[j], at[j], a.block, False) for j in range(len(prompts))]
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            FlashNextServing.forward_tree_multi(
                [(s, torch.tensor(blocks[j][0], device="cuda"), blocks[j][1], s.st.length) for j, s in enumerate(joint)])
            for j, s in enumerate(joint):
                s.commit_tree(list(range(len(blocks[j][0]))))
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        for j in range(len(prompts)):
            at[j] += len(blocks[j][0])
        by, n_k = {}, 0
        for ev in prof.events():
            if ev.device_type == torch.autograd.DeviceType.CUDA:
                by[ev.name] = by.get(ev.name, 0.0) + ev.device_time
                n_k += 1
        busy = sum(by.values()) / 1e3
        top = sorted(by.items(), key=lambda kv: -kv[1])[:a.kernels]
        kern = {"wall_ms": round(wall * 1e3, 2), "busy_ms": round(busy, 2), "kernels": n_k,
                "top": [(k[:90], round(v / 1e3, 3)) for k, v in top]}
        print(f"[joint] kernels: wall {kern['wall_ms']} ms (profiled), GPU busy {kern['busy_ms']} ms, "
              f"{n_k} kernels", flush=True)
        for k, v in kern["top"]:
            print(f"[joint]   {v:8.3f} ms  {k}", flush=True)
    if a.dump_moe:
        # SPD-249: the experts' real routing of a joint pass (each layer's x and ids), to replay in the kernel bench
        from tools import fn_kernels as FK
        layer_of = {id(b): li for li, b in fn.banks.items()} if isinstance(fn.banks, dict) else \
            {id(b): li for li, b in enumerate(fn.banks)}
        rec, real = [], FK.moe_experts

        def spy(x, ids, bank, order=None):
            if x.shape[0] > 1:
                rec.append({"layer": layer_of.get(id(bank)), "x": x.detach().cpu().clone(), "ids": ids.detach().cpu().clone()})
            return real(x, ids, bank, order)
        blocks = [block_of(conts[j], at[j], a.block, False) for j in range(len(prompts))]
        FK.moe_experts = spy
        try:
            FlashNextServing.forward_tree_multi(
                [(s, torch.tensor(blocks[j][0], device="cuda"), blocks[j][1], s.st.length) for j, s in enumerate(joint)])
        finally:
            FK.moe_experts = real
        commit_joint(joint, [list(range(len(blocks[j][0]))) for j in range(len(joint))])
        for j in range(len(prompts)):
            at[j] += len(blocks[j][0])
        torch.save({"rows": len(prompts) * a.block, "records": rec}, a.dump_moe)
        u = [int(torch.unique(r["ids"][:, :-1]).numel()) for r in rec]
        print(f"[joint] dumped {len(rec)} expert calls ({rec[0]['x'].shape[0] if rec else 0} rows): distinct routed "
              f"experts a layer mean {sum(u) / max(1, len(u)):.1f}", flush=True)
    lone_k = None
    if a.lone_kernels:
        from torch.profiler import ProfilerActivity, profile
        lone_k = {}
        s = lone[0]
        for w in range(2, a.block + 1):
            for rep_ in range(3):                               # two warm (the graphs' captures), the third is profiled
                blk = block_of(conts[0], at[0], w, False)
                torch.cuda.synchronize()
                if rep_ < 2:
                    s.forward_tree(torch.tensor(blk[0], device="cuda"), blk[1], s.st.length)
                    s.commit_tree(list(range(len(blk[0]))))
                    at[0] += len(blk[0])
                    continue
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    t0 = time.perf_counter()
                    s.forward_tree(torch.tensor(blk[0], device="cuda"), blk[1], s.st.length)
                    s.commit_tree(list(range(len(blk[0]))))
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - t0
                at[0] += len(blk[0])
            by = {}
            for ev in prof.events():
                if ev.device_type == torch.autograd.DeviceType.CUDA:
                    by[ev.name] = by.get(ev.name, 0.0) + ev.device_time
            top = sorted(by.items(), key=lambda kv: -kv[1])[:a.lone_kernels]
            lone_k[w] = {"wall_ms": round(wall * 1e3, 2), "busy_ms": round(sum(by.values()) / 1e3, 2),
                         "top": [(k[:90], round(v / 1e3, 3)) for k, v in top]}
            print(f"[joint] lone width {w}: wall {lone_k[w]['wall_ms']} ms, GPU busy {lone_k[w]['busy_ms']} ms", flush=True)
            for k, v in lone_k[w]["top"]:
                print(f"[joint]   {v:8.3f} ms  {k}", flush=True)
    end = []
    for s_l, s_j in zip(lone, joint):
        s_l._settle_chain()
        s_j._settle_chain()
        fn.flush()
        secs_l, secs_j = s_l.state.sections(), s_j.state.sections()
        bad = [k for k in secs_l if not torch.equal(secs_l[k], secs_j[k])]
        views = [n for (n, t, ax), (_, u, _) in zip(s_l.state.views(), s_j.state.views())
                 if not torch.equal(t.narrow(ax, 0, s_l.st.length), u.narrow(ax, 0, s_j.st.length))]
        end.append({"length": [s_l.st.length, s_j.st.length], "sections_differ": bad, "views_differ": views})
        print(f"[joint] end: {end[-1]}", flush=True)
    if a.graph:
        print(f"[joint] graphs: {jstats()}", flush=True)
    if a.json:
        json.dump({"rounds": rounds, "end": end, "timing": timing, "kernels": kern, "lone_kernels": lone_k}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
