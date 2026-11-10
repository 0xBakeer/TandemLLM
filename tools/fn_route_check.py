"""SPD-249 (T8) bit check on the pack: the fused routing launch against today's launches, on real routing, in one
process.

    PYTHONPATH=<repo> python tools/fn_route_check.py --pack ~/flashnext-packs/3.05 \
        --replay moe-ids.pt,c1-moe.pt --json out.json
    python tools/fn_route_check.py --compare off.pt on.pt      (two tools/fn_joint_check.py --dump-logits files)

Replay: each record of a `fn_joint_check --dump-moe` file (a layer's MoE input rows of a real joint pass) goes through
the engine's `moe_parts` twice: knob off (today's router, mask, sort and groups) and knob on (QWEN38_FN_ROUTE_FUSED:
`fn_kernels.route`), the serving variant of the grouped kernels from the environment (FN_MOE_AS / GR / GPAR, the profile's). Per
call: the ids, the weights, every real pair's expert output (yd) and the layer's output after the combine
(`combine_inject_norm` with the layer's own rank and next-site norm over fixed h0 and d: the streams and their norm, the
"layer output"), each `torch.equal`. Shapes:

  joint    the record's rows through the grouped kernels (the joint graph's call), the last 0, 3 and 8 rows padded (a
           joint graph's pad rows: ids -1; their rows are not compared);
  lone     the record's first M rows, M = 1..6, 8, 16, 32 (rows repeated past the record's), through the per-pair
           kernels (the lone verify's and the decode step's call);
  index    route's order (lone) and groups (joint) against `expert_order` and `fn_moe_groups` on the same ids;
  graph    the on-call of every layer captured in one CUDA graph on a leased stream with a graph pool (the joint graph's
           capture), replayed with another record's rows and pad mask copied in, against the eager off-call of those
           rows.

Self-check: the load-time self-check with the knob on (its routed arms run the verify calls through `moe_parts`), then
with a fault injected into the route launch and into the grouped experts: those runs must FAIL (`self_checks`).

`--timing R`: the 48 layers' off-calls and on-calls each captured as one graph (MoE calls only: router GEMV, routing,
experts; no combine), replayed R times, median ms a pass, for the joint and the lone shapes. An offline kernel time, not
the bare A/B.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _set(E, K, fused: bool, group: bool) -> None:
    E.ROUTE_FUSED = fused
    K.MOE_GROUP = group


def _layer_out(fn, K, li, yd, ids, wts, h0, d):
    """The layer's streams and their norm after the MoE (the trunk's `combine_inject_norm` of layer li)."""
    c = fn.cfg
    p = fn.pre(li)
    nxt = None if li + 1 in c.ple_layers else fn._site_norm(li + 1)
    return K.combine_inject_norm(h0, yd, ids, wts, d, fn.W[p + "mlp_hyper_connection.rank"], nxt, c.hc_count,
                                 c.rms_norm_eps)


def _real_pairs(ids):
    return ids.reshape(-1) >= 0


def _cmp(a, b) -> bool:
    return a is None and b is None or (a is not None and b is not None and bool(torch.equal(a, b)))


def call_pair(fn, E, K, x, li, pad, h0, d, group: bool):
    """(off, on) of one moe_parts call: each (yd, ids, wts, h, n)."""
    out = []
    for fused in (False, True):
        _set(E, K, fused, group)
        yd, ids, wts = fn.moe_parts(x, li, pad=pad)
        h, n = _layer_out(fn, K, li, yd, ids, wts, h0, d)
        out.append((yd.clone(), ids.clone(), wts.clone(), h.clone(), None if n is None else n.clone()))
    torch.cuda.synchronize()
    return out


def equal_calls(off, on) -> dict:
    yd0, i0, w0, h0, n0 = off
    yd1, i1, w1, h1, n1 = on
    real = _real_pairs(i0)
    rows = (i0 >= 0).all(dim=1)
    r = {"ids": bool(torch.equal(i0, i1)), "wts": bool(torch.equal(w0, w1)),
         "yd": bool(torch.equal(yd0[real], yd1[real])),
         "h": bool(torch.equal(h0[rows], h1[rows])), "n": _cmp(None if n0 is None else n0[rows], None if n1 is None else n1[rows])}
    r["all"] = all(r.values())
    return r


def index_check(fn, K, x, li, pad) -> dict:
    """route's ids, weights, order and groups against router + mask + expert_order + fn_moe_groups."""
    c = fn.cfg
    E, k = c.num_experts, c.num_experts_per_tok
    mod = K._module()
    logits = K.gemv(x, fn.W[fn.pre(li) + "mlp.router"])
    ids0, w0 = K.router(logits, E, k)
    if pad is not None:
        ids0.masked_fill_(pad, -1)
    order0 = K.expert_order(ids0)
    P = ids0.numel()
    G0 = torch.full((P, K.MOE_G_COLS), -7, dtype=torch.int32, device="cuda")
    NG0 = torch.zeros(1, dtype=torch.int32, device="cuda")
    mod.fn_moe_groups(ids0.reshape(-1), order0, G0, NG0)
    ids1, w1, order1 = K.route(logits, E, k, pad=pad)
    ok = {"ids": bool(torch.equal(ids0, ids1)), "wts": bool(torch.equal(w0, w1)), "order": bool(torch.equal(order0, order1))}

    def table(G, NG):
        return [G[r, :2 + int(G[r, 1])].tolist() for r in range(int(NG))]
    ids2, w2, G1, NG1 = K.route(logits, E, k, pad=pad, groups=True)
    ok["groups"] = bool(torch.equal(ids2, ids0)) and bool(torch.equal(w2, w0)) and table(G1, NG1) == table(G0, NG0)
    ok["all"] = all(ok.values())
    return ok


def capture_layers(fn, E, K, X, pads, fused, group, with_out, h0, d):
    """Every layer's moe_parts (and, with_out, its combine) captured in one graph on a leased stream with a graph pool,
    the joint graph's capture; returns (graph, outputs by layer)."""
    from engine import capture_lock
    _set(E, K, fused, group)
    for li in range(fn.cfg.num_hidden_layers):                    # warm: every launch's attributes set eagerly
        fn.moe_parts(X[li], li, pad=pads[li])
    torch.cuda.synchronize()
    s = capture_lock.lease_stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    outs = {}
    with capture_lock.LOCK, torch.cuda.graph(g, pool=torch.cuda.graph_pool_handle(), stream=s):
        for li in range(fn.cfg.num_hidden_layers):
            yd, ids, wts = fn.moe_parts(X[li], li, pad=pads[li])
            outs[li] = (yd, ids, wts) + (_layer_out(fn, K, li, yd, ids, wts, h0, d) if with_out else ())
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    return g, outs


def timed_replay(g, reps: int) -> float:
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


@torch.no_grad()
def replay_check(a) -> dict:
    from engine.models.flashnext import engine as E
    from engine.models.flashnext.serving import build
    from tools import fn_kernels as K
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.pack)
    eng, _, _ = build(a.pack, 4096, tok, spec=False, log=lambda s: None)
    fn = eng.fn
    c = fn.cfg
    knobs = {k: os.environ.get("QWEN38_" + k) for k in ("FN_MOE_AS", "FN_MOE_GR", "FN_MOE_GPAR", "FN_MOE_ST",
                                                        "FN_MOE_WARPS", "FN_PDL", "FN_ROUTE_FUSED")}
    print(f"[route] knobs from the environment {knobs}; MOE_AS {K.MOE_AS}; the tool sets ROUTE_FUSED per arm",
          flush=True)
    g = torch.Generator(device="cuda").manual_seed(11)
    HC = c.hc_count * c.hidden_size
    res = {"knobs": knobs, "files": {}}
    ok_all = True
    for path in [p for p in a.replay.split(",") if p.strip()]:
        d_ = torch.load(path)
        recs = [r for r in d_["records"] if r["layer"] is not None]
        R0 = recs[0]["x"].shape[0]
        fr = {"rows": R0, "layers": len(recs), "joint": {}, "lone": {}, "index": {}, "graph": None}
        h0 = (torch.randn(32, HC, generator=g, device="cuda")).to(torch.bfloat16)
        dd = torch.randn(32, 512, generator=g, device="cuda") * 3      # d [M, R + hc]: the kernel reads R + hc <= 512
        # joint: the record's rows, grouped, pad rows 0 / 3 / 8 (rows left >= 2)
        for npad in (0, 3, 8):
            if R0 - npad < 2:
                continue
            bad, n = [], 0
            for r in recs:
                li, x = r["layer"], r["x"].cuda()
                pad = torch.zeros(R0, 1, dtype=torch.bool, device="cuda")
                if npad:
                    pad[R0 - npad:] = True
                off, on = call_pair(fn, E, K, x, li, pad, h0[:R0], dd[:R0], True)
                e = equal_calls(off, on)
                n += 1
                if not e["all"]:
                    bad.append({"layer": li, **e})
                if npad == 0 or npad == 3:
                    ix = index_check(fn, K, x, li, pad)
                    if not ix["all"]:
                        bad.append({"layer": li, "index": ix})
            fr["joint"][npad] = {"calls": n, "bad": bad[:6], "equal": not bad}
            ok_all = ok_all and not bad
            print(f"[route] {os.path.basename(path)} joint rows {R0} pad {npad}: {n} layers, equal {not bad} "
                  f"{bad[:2]}", flush=True)
        # lone: the first M rows (repeated past the record), per-pair kernels
        for M in (1, 2, 3, 4, 5, 6, 8, 16, 32):
            bad, n = [], 0
            for r in recs:
                li = r["layer"]
                xr = r["x"].cuda()
                x = xr.repeat((M + R0 - 1) // R0, 1)[:M].contiguous()
                off, on = call_pair(fn, E, K, x, li, None, h0[:M], dd[:M], False)
                e = equal_calls(off, on)
                n += 1
                if not e["all"]:
                    bad.append({"layer": li, **e})
                if M > 1 and M in (2, 5, 32):
                    ix = index_check(fn, K, x, li, None)
                    if not ix["all"]:
                        bad.append({"layer": li, "index": ix})
            fr["lone"][M] = {"calls": n, "bad": bad[:6], "equal": not bad}
            ok_all = ok_all and not bad
            print(f"[route] {os.path.basename(path)} lone rows {M}: {n} layers, equal {not bad} {bad[:2]}", flush=True)
        # graph: every layer's on-call captured once, replayed with another record set's rows and pad mask
        if len(recs) == c.num_hidden_layers and R0 >= 4:
            Xb = {li: torch.empty(R0, c.hidden_size, dtype=torch.bfloat16, device="cuda") for li in range(len(recs))}
            Pb = {li: torch.zeros(R0, 1, dtype=torch.bool, device="cuda") for li in range(len(recs))}
            by = {r["layer"]: r for r in recs}
            for li in Xb:
                Xb[li].copy_(by[li]["x"])
            dl = dd[:R0]
            gr, outs = capture_layers(fn, E, K, Xb, Pb, True, True, True, h0[:R0], dl)
            # new inputs: each layer's rows from the layer below (another real routing), 3 pad rows
            for li in Xb:
                Xb[li].copy_(by[(li + 1) % len(recs)]["x"])
                Pb[li][R0 - 3:] = True
            gr.replay()
            torch.cuda.synchronize()
            bad = []
            for li in Xb:
                off = call_pair(fn, E, K, Xb[li].clone(), li, Pb[li].clone(), h0[:R0], dl, True)[0]
                got = tuple(t.clone() if t is not None else None for t in outs[li])
                e = equal_calls(off, got)
                if not e["all"]:
                    bad.append({"layer": li, **e})
            fr["graph"] = {"layers": len(Xb), "bad": bad[:6], "equal": not bad}
            ok_all = ok_all and not bad
            print(f"[route] {os.path.basename(path)} graph (48 layers, replayed with new rows + 3 pad rows): equal "
                  f"{not bad} {bad[:2]}", flush=True)
            del gr, outs
        # timing: the MoE calls of a pass, off and on, each one graph
        if a.timing and len(recs) == c.num_hidden_layers:
            by = {r["layer"]: r for r in recs}
            tm = {}
            for shape, M, group in (("joint", R0, True), ("lone", min(R0, 4), False)):
                Xb = {li: by[li]["x"][:M].cuda().contiguous() for li in by}
                Pb = {li: (torch.zeros(M, 1, dtype=torch.bool, device="cuda") if group else None) for li in by}
                for lab, fused in (("off", False), ("t8", True)):
                    gr, _ = capture_layers(fn, E, K, Xb, Pb, fused, group, False, None, None)
                    gr.replay()
                    torch.cuda.synchronize()
                    tm[f"{shape}{M}_{lab}"] = round(timed_replay(gr, a.timing), 3)
                    del gr
                print(f"[route] timing {shape} rows {M} (48 MoE calls a graph, median of {a.timing}): "
                      f"{ {k: v for k, v in tm.items() if k.startswith(shape)} }", flush=True)
            fr["timing_ms"] = tm
        res["files"][os.path.basename(path)] = fr
    _set(E, K, False, False)
    res["self_check_on"], res["self_check_faults"] = self_checks(fn, E, K, tok)
    ok_all = ok_all and all(c.get("ok") and "arms" in c for c in res["self_check_on"].values())
    ok_all = ok_all and not any(c.get("ok") for c in res["self_check_faults"].values())
    res["all_equal"] = ok_all
    print(f"[route] ALL EQUAL {ok_all}", flush=True)
    return res


def self_checks(fn, E, K, tok) -> tuple:
    """The load-time self-check with the knob on. Its routed arms send every verify-shaped call through `moe_parts`
    (route_kernel, moe_experts_ix). Then two faults: the multi-row route launch's weights (row 0, slot 0, + 1) and the grouped experts' output
    (pair 0, + 1); the decode step's single row takes neither (router_kernel, the per-pair order), so only the routed
    arms can see them: the check must FAIL with a routed arm's call in first_diff. Returns (by arm, by fault)."""
    text = tok.apply_chat_template([{"role": "user", "content": "Count from one to twenty in words."}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt = tok(text, add_special_tokens=False)["input_ids"]
    log = lambda s: print(s, flush=True)                           # noqa: E731
    on, faults = {}, {}
    try:
        E.ROUTE_FUSED = True
        t0 = time.time()
        on["t8"] = fn.self_check(prompt, log=log)
        on["t8"]["seconds"] = round(time.time() - t0, 1)
        real_route, real_ix = K.route, K.moe_experts_ix

        def bad_route(*a, **k):
            r = real_route(*a, **k)
            r[1][0, 0] += 1.0
            return r

        def bad_ix(x, ids, bank, ix):
            yd = real_ix(x, ids, bank, ix)
            if ix[0] == "groups":
                yd[0] += 1.0
            return yd
        for lab, attr, f, real in (("route_wts", "route", bad_route, real_route),
                                   ("grouped_yd", "moe_experts_ix", bad_ix, real_ix)):
            E.ROUTE_FUSED = True
            setattr(K, attr, f)
            try:
                faults[lab] = fn.self_check(prompt, log=log)
            finally:
                setattr(K, attr, real)
            print(f"[route] self-check fault {lab}: ok {faults[lab]['ok']} (want False), first_diff "
                  f"{faults[lab]['first_diff']}", flush=True)
    finally:
        E.ROUTE_FUSED = False
    return on, faults


def compare(a_path: str, b_path: str) -> dict:
    A, B = torch.load(a_path), torch.load(b_path)
    r = {"conts": A["conts"] == B["conts"], "rounds": len(A["lone"]), "lone_bad": [], "joint_bad": [], "end_bad": []}
    for i, (la, lb) in enumerate(zip(A["lone"], B["lone"])):         # sha256 digests of the logits' bytes
        for j in la:
            if lb.get(j) != la[j]:
                r["lone_bad"].append((i, j))
    for i, (ja, jb) in enumerate(zip(A["joint"], B["joint"])):
        for j in ja:
            if jb.get(j) != ja[j]:
                r["joint_bad"].append((i, j))
    r["logits_compared"] = sum(len(x) for x in A["lone"]) + sum(len(x) for x in A["joint"])
    for i, (ea, eb) in enumerate(zip(A["end"], B["end"])):
        if ea != eb:
            r["end_bad"].append(i)
    r["same_rounds"] = len(A["lone"]) == len(B["lone"]) and len(A["joint"]) == len(B["joint"])
    r["equal"] = bool(r["conts"] and r["same_rounds"] and not r["lone_bad"] and not r["joint_bad"] and not r["end_bad"]
                      and len(A["end"]) == len(B["end"]))
    print(f"[route] compare {os.path.basename(a_path)} vs {os.path.basename(b_path)}: {r}", flush=True)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack")
    ap.add_argument("--replay", default="", help="fn_joint_check --dump-moe files, comma separated")
    ap.add_argument("--timing", type=int, default=0, help="graph replays a timed arm (0: no timing)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="two fn_joint_check --dump-logits files")
    ap.add_argument("--json")
    a = ap.parse_args()
    t0 = time.time()
    res = compare(*a.compare) if a.compare else replay_check(a)
    res["seconds"] = round(time.time() - t0, 1)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=str)
    sys.exit(0 if res.get("all_equal", res.get("equal")) else 1)


if __name__ == "__main__":
    main()
