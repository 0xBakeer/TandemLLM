"""SPD-249: the prompts of a prefill burst in one forward on the prefill kernels, each prompt's rows bit for bit
its own prefill's (`FlashNextEngine.prefill`, one chunk: `forward_pre` with head "last").

Eight requests sent at once used to prefill one after another; each prefill read every layer's experts for its own
rows. Here the prompts go through the layers together and each layer's experts run once over every prompt's rows
(`fn_prefill_kernels.moe_grouped`: the pairs grouped by expert, a pair's output its own row's whatever group or item
it lands in), the burst reading the expert bytes once instead of once a prompt.

Everything else of a layer runs a prompt at a time, as the prompt's own prefill runs it, at its own row count: the
library GEMMs pick their algorithm by the row count M (tools/exl3_linear's slab GEMM past EXL3_CHUNK_UNTIL rows and
`fn_prefill_kernels.mm_rows`' fp32 GEMMs: mixers, router, GDN a|b, PLE projections), and the PLE, GDN and attention
layers read and write each prompt's own state. A layer's EXL3 slabs and fp32 weight forms are decoded once for all
of its prompts (`slab_cache_*`, `wf_cache_*`: the same tensors, the same GEMM shapes, bit for bit the uncached calls)
and cleared after the layer. The head reads each prompt's last row alone.

`ok` says which prompt chunks can join (the kernels path, one chunk, no decode step, no images); `prefill_joint` cuts
the jobs into runs of at most FN_PREFILL_CHUNK rows (the prefill workspace the memory policy holds) and runs each run
of two or more jointly, a lone job through `fn.prefill` itself. tools/fn_joint_check.py --joint-prefill holds every
prompt's logits, streams, KV, indexer keys, pooled keys, GDN and conv states and PLE context to its lone prefill's.
"""

from __future__ import annotations

import torch

from engine.settings import SETTINGS as _S

#: SPD-249: the server's prefill burst runs its Flash-Next prompts through `prefill_joint`; 0: each alone
ON = _S.get("FN_JOINT_PREFILL") == "1"


def ok(fn, T: int, start: int, st) -> bool:
    """Whether a prompt chunk of T rows at `start` on state `st` can join: its own forward would be one chunk on the
    prefill kernels (not the decode step's one token, not the reference path), without images."""
    from engine.models.flashnext import engine as E
    T, start = int(T), int(start)
    return (fn.prefill_path == "kernels" and 1 <= T <= E.PREFILL_CHUNK and not (T == 1 and start > 0)
            and st.mm is None and not st.delta)


def runs(lens: list, cap: int) -> list:
    """The jobs' indices in order, cut into runs whose rows add up to at most `cap` (a job alone may fill one)."""
    out, cur, n = [], [], 0
    for i, T in enumerate(lens):
        if cur and n + int(T) > cap:
            out.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += int(T)
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def prefill_joint(fn, jobs: list) -> list:
    """jobs [(tokens [T], state)], each `ok`, on distinct states -> [(logits fp32 [1, vocab], streams [T, hc H])] in
    order, every state left as `fn.prefill(tokens, state, keep_stack=True)` leaves it (that call's `out[-1:]` and
    `prompt_stack`)."""
    from engine.models.flashnext import engine as E
    assert len({id(st) for _, st in jobs}) == len(jobs), "a state twice in one joint prefill"
    out = [None] * len(jobs)
    for run in runs([int(t.numel()) for t, _ in jobs], E.PREFILL_CHUNK):
        if len(run) == 1:
            t, st = jobs[run[0]]
            lg = fn.prefill(t.view(-1), st, keep_stack=True)
            out[run[0]] = (lg, fn.prompt_stack)
            continue
        for i, r in zip(run, _joint(fn, [jobs[i] for i in run])):
            out[i] = r
    return out


def _joint(fn, jobs: list) -> list:
    """One run of jobs through every layer (see the module docstring)."""
    from tools import exl3_linear as X
    from tools import fn_kernels as K
    from tools import fn_prefill_kernels as PK
    c, W = fn.cfg, fn.W
    H, hc, eps = c.hidden_size, c.hc_count, c.rms_norm_eps
    seqs = []
    for t, st in jobs:
        fn.flush(st)                                         # `prefill`'s first step
        t = t.view(-1).to(fn.device)
        pre = fn._pre_ctl(st.length, t.numel(), st)         # `forward_pre`'s
        seqs.append({"t": t, "st": st, "pre": pre, "h": fn.embed(t).repeat(1, hc), "n": None})
    X.slab_cache_on()
    PK.wf_cache_on()
    try:
        for li in range(c.num_hidden_layers):
            p = fn.pre(li)
            xs, dis = [], []
            for s in seqs:                                    # `trunk(pre=True)` up to the MoE, a prompt at a time
                fn._pre = s["pre"]
                h, st = s["h"], s["st"]
                if li in c.ple_layers:
                    h = h + fn._ple_pre(h, s["t"], li, st)
                    s["n"] = None
                n = s["n"] if s["n"] is not None else K.hc_norm(h, fn._site_norm(li), H, eps)
                x, di = fn._mix_pre(n, p + "attn_hyper_connection")
                out = fn._gdn_pre(x, li, st) if c.is_linear(li) else fn._attn_pre(x, li, st)
                h, n = fn._inject_pre(h, out, di, W[p + "attn_hyper_connection.rank"],
                                      W[p + "mlp_hyper_connection.hc_norm.weight"], hc, eps)
                x, di = fn._mix_pre(n, p + "mlp_hyper_connection")
                s["h"] = h
                xs.append(x)
                dis.append(di)
            # `_moe_pre`: the router a prompt at a time (its GEMM picks by M), the experts of every prompt's rows at once,
            # the combine a prompt at a time
            ids, wts = [], []
            for s, x in zip(seqs, xs):
                i_, w_ = K.router(PK.mm_rows(x, W[p + "mlp.router"]), c.num_experts, c.num_experts_per_tok)
                if fn.route_trace is not None:
                    fn.route_trace.append((li, i_[:, :-1].long()))
                ids.append(i_)
                wts.append(w_)
            yd = PK.moe_grouped(torch.cat(xs), torch.cat(ids), fn.banks[li])
            del xs
            nxt = None if li + 1 in c.ple_layers else fn._site_norm(li + 1)
            R = W[p + "mlp_hyper_connection.rank"]
            r0 = 0
            for s, i_, w_, di in zip(seqs, ids, wts, dis):
                r1 = r0 + i_.numel()
                s["h"], s["n"] = fn._inject_pre(s["h"], K.moe_combine(yd[r0:r1], i_, w_), di, R, nxt, hc, eps)
                r0 = r1
            del yd, ids, wts, dis
            X.slab_cache_clear()
            PK.wf_cache_clear()
    finally:
        X.slab_cache_off()
        PK.wf_cache_off()
        fn._n32_clear()
        fn._pre = None
    res = []
    for s in seqs:                                            # `forward_pre`'s end and `prefill`'s
        st, h = s["st"], s["h"]
        n = s["n"] if s["n"] is not None else K.hc_norm(h, fn._site_norm(c.num_hidden_layers), H, eps)
        st.length += s["t"].numel()
        lg = fn.final(n[-1:])
        stack = torch.cat([h])
        res.append((lg[-1:], stack))
        if st.length > 4096:
            torch.cuda.empty_cache()
    fn.stack = seqs[-1]["h"]
    fn.prompt_stack = res[-1][1]
    return res
