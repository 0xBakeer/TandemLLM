"""SPD-249 P1: consecutive prompt chunks of one sequence LAYER-major on the prefill kernels, each chunk's rows bit for
bit its own chunk-major forward's (`FlashNextEngine.forward_pre`, chunk after chunk).

A chunk-major prefill runs a 4,096-row chunk through all 48 layers before the next chunk starts, so every layer's
experts run once a chunk, on that chunk's pairs alone. Here every chunk of a window passes layer l before any chunk
passes layer l + 1, and the layer's experts run once over every chunk's rows (`fn_prefill_kernels.moe_grouped`: the
pairs grouped by expert, a pair's output its own row's whatever group or item it lands in, as the joint prefill of a
burst, engine/models/flashnext/joint_prefill.py). An expert's rows fill its 64-row items across the window, so the
items that hold a few pairs of a chunk merge.

Everything else of a layer runs a chunk at a time, in chunk order, at the chunk's own row count and on the state as
the chunk-major order leaves it at that layer: the GDN, conv and PLE states of layer l and the KV, indexer keys and
pooled blocks of layer l are written by chunk c before chunk c + 1 reads them, as before; no layer reads another
layer's state. The library GEMMs pick their algorithm by the row count M (the EXL3 slab GEMM past EXL3_CHUNK_UNTIL
rows, `mm_rows`' fp32 GEMMs), so each chunk keeps its own call. A layer's EXL3 slabs and fp32 weight forms are
decoded once for all of its chunks (`slab_cache_*`, `wf_cache_*`: the same tensors, the same GEMM shapes) and dropped
after the layer.

The window holds every chunk's streams between layers and the experts' buffers of all its rows at once: about
190 KB a row more than one chunk (the grouped inputs, gate|up, the down products in fp32, the streams and the
mixer inputs), which the memory policy charges (serving.py `_workspace_rows`). `ROWS` caps a window
(QWEN38_FN_PF_WINDOW; 0: chunk-major as before).

tools/fn_window_check.py holds the window's logits, streams, sections, KV, indexer and pooled keys, length and PLE
context to the chunk-major prefill's, then decode steps on both.
"""

from __future__ import annotations

import torch

from engine.settings import SETTINGS as _S

#: SPD-249 P1: the most rows a layer-major window takes (a multiple of the prefill chunk); 0: chunk-major
ROWS = int(_S.get("FN_PF_WINDOW") or 0)


def groups(lens: list, cap: int) -> list:
    """The chunks' indices in order, cut into windows whose rows add up to at most `cap` (a chunk alone may fill
    one); a window of one chunk runs chunk-major."""
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


def ok(fn, st) -> bool:
    """Whether a window can run on `st`: the prefill kernels, text only (images go chunk-major)."""
    return fn.prefill_path == "kernels" and st.mm is None and not st.delta


@torch.no_grad()
def window(fn, toks: list, st, head: bool = True, snap=None) -> tuple:
    """toks [T_c] the window's chunks in order, from `st.length` on -> (logits fp32 [1, vocab] of the last chunk's
    last row or None, [streams [T_c, hc H] of each chunk]); `st` left as the chunks' forwards leave it.

    `snap(c, li)`: called after chunk c (every chunk but the last) has passed layer li, so a caller can keep the
    recurrent sections a chunk-major prefill holds at the end of chunk c (`FlashNextServing.prefill_window`)."""
    from tools import exl3_linear as X
    from tools import fn_kernels as K
    from tools import fn_prefill_kernels as PK
    c, W = fn.cfg, fn.W
    H, hc, eps = c.hidden_size, c.hc_count, c.rms_norm_eps
    C = len(toks)
    fn.flush(st)                                             # `prefill`'s first step
    L0 = st.length
    seqs, a, shared = [], L0, {}
    for t in toks:
        t = t.view(-1).to(fn.device)
        T = t.numel()
        pre = fn._pre_ctl(a, T, st)                          # `forward_pre`'s, at the chunk's own start
        # a chunk's attention scratch (`attn_rows`' partials, the indexer's queries) is a chunk-major forward's
        # own; the window's chunks of one row count share one set, as consecutive chunk-major forwards reuse
        # the allocator's blocks (every launch writes what it reads)
        pre["bufs"] = shared.setdefault(T, {})
        seqs.append({"t": t, "L0": a, "pre": pre, "h": fn.embed(t).repeat(1, hc), "n": None, "trace": []})
        a += T
    rows = fn.moe_group_rows or PK.GROUP_ROWS
    trace = fn.route_trace
    X.slab_cache_on()
    PK.wf_cache_on()
    try:
        for li in range(c.num_hidden_layers):
            p = fn.pre(li)
            xs, dis = [], []
            for ci, s in enumerate(seqs):                     # `trunk(pre=True)` up to the MoE, a chunk at a time
                fn._pre = s["pre"]
                st.length = s["L0"]                           # the chunk's forward sees its own start
                h = s["h"]
                if li in c.ple_layers:
                    h = h + fn._ple_pre(h, s["t"], li, st)
                    s["n"] = None
                n = s["n"] if s["n"] is not None else K.hc_norm(h, fn._site_norm(li), H, eps)
                x, di = fn._mix_pre(n, p + "attn_hyper_connection")
                out = fn._gdn_pre(x, li, st) if c.is_linear(li) else fn._attn_pre(x, li, st)
                h, n = K.inject_norm(h, out, di, W[p + "attn_hyper_connection.rank"],
                                     W[p + "mlp_hyper_connection.hc_norm.weight"], hc, eps)
                x, di = fn._mix_pre(n, p + "mlp_hyper_connection")
                s["h"] = h
                xs.append(x)
                dis.append(di)
                if snap is not None and ci < C - 1:
                    snap(ci, li)
            # `_moe_pre`: the router a chunk at a time (its GEMM picks by M), the experts of every chunk's rows at
            # once, the combine a chunk at a time
            ids, wts = [], []
            for s, x in zip(seqs, xs):
                i_, w_ = K.router(PK.mm_rows(x, W[p + "mlp.router"]), c.num_experts, c.num_experts_per_tok)
                if trace is not None:
                    s["trace"].append((li, i_[:, :-1].long()))
                ids.append(i_)
                wts.append(w_)
            yd = PK.moe_grouped(torch.cat(xs) if C > 1 else xs[0], torch.cat(ids) if C > 1 else ids[0],
                                fn.banks[li], rows=rows)
            del xs
            nxt = None if li + 1 in c.ple_layers else fn._site_norm(li + 1)
            R = W[p + "mlp_hyper_connection.rank"]
            r0 = 0
            for s, i_, w_, di in zip(seqs, ids, wts, dis):
                r1 = r0 + i_.numel()
                s["h"], s["n"] = K.inject_norm(s["h"], K.moe_combine(yd[r0:r1], i_, w_), di, R, nxt, hc, eps)
                r0 = r1
            del yd, ids, wts, dis
            X.slab_cache_clear()
            PK.wf_cache_clear()
    finally:
        X.slab_cache_off()
        PK.wf_cache_off()
        fn._pre = None
        st.length = L0
    st.length = a
    if trace is not None:                                     # in the chunk-major order
        for s in seqs:
            trace.extend(s["trace"])
    last = seqs[-1]
    fn.stack = last["h"]
    lg = None
    if head:
        n = last["n"] if last["n"] is not None else K.hc_norm(last["h"], fn._site_norm(c.num_hidden_layers), H, eps)
        lg = fn.final(n[-1:])
    return lg, [s["h"] for s in seqs]
