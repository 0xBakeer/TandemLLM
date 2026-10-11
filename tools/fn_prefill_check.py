"""ENG-225 on the board: the prefill kernels against the reference forward.

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_prefill_check.py --pack ~/fn2/packs/3.05 \
        --ctx 1024,8192,32768 --classes prose,code --new 64 --json out.json

Per context and class (tools/fn_generate.py's prompts filled to `--ctx` tokens), two states read the
same prompt in the same chunks, one through the reference forward (`forward`), one through the prefill
kernels (`forward_pre`):

  teacher-forced   after each chunk, the next-token logits of every row of both (the same final mixer
                   and head on each path's own streams, in blocks of 256 rows): argmax agreement over
                   every position, and the reference's top-2 margin at each disagreement; a position
                   is confident when that margin is at least `--confident` (logits). With `--rows`, a
                   third state reads the prompt through the decode path's own kernels (verify blocks of
                   32 chain rows, each committed: the row-by-row arithmetic, bit-equal to decode steps),
                   and every pair of the three paths is compared the same way.
  continuation     greedy decode of `--new` tokens from each state (the decode graph): the two texts,
                   and at the first difference the reference state's own top-2 margin (a near tie or not).
  MTP prime        with `--mtp`: the MTP layer fed the prompt's streams and next tokens on each path
                   (`mtp_feed` / `mtp_pre`), its draft argmax per position compared the same way; and
                   the MTP layer alone (`mtp_same_streams`): both fed the reference's streams.

Also the wall time of each path's prompt (the kernel path's tok/s is ENG-225's number when the reference
does not run beside it: tools/fn_prefill_profile.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import DecodeGraph, FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402
from tools.fn_generate import class_prompt, mem_available_gib  # noqa: E402


@torch.no_grad()
def head_top2(eng: FlashNextEngine, stack: torch.Tensor, block: int = 256):
    """(top-1 ids, top-1 - top-2 margin) of every row's next-token logits from its streams."""
    from tools import fn_kernels as K
    c = eng.cfg
    ids, margin = [], []
    for r0 in range(0, stack.shape[0], block):
        n = K.hc_norm(stack[r0:r0 + block].contiguous(), eng._site_norm(c.num_hidden_layers), c.hidden_size,
                      c.rms_norm_eps)
        v, i = torch.topk(eng.final(n), 2, dim=-1)
        ids.append(i[:, 0])
        margin.append(v[:, 0] - v[:, 1])
    return torch.cat(ids), torch.cat(margin)


@torch.no_grad()
def mtp_top2(eng: FlashNextEngine, h: torch.Tensor, block: int = 256):
    """The same for the MTP layer's own streams (its final mixer, the shared head)."""
    ids, margin = [], []
    for r0 in range(0, h.shape[0], block):
        x = eng.mixer(h[r0:r0 + block], "mtp.hyper_connection_mixer", combine=False)
        v, i = torch.topk(eng.lin(x, eng.W["lm_head.weight"]).float(), 2, dim=-1)
        ids.append(i[:, 0])
        margin.append(v[:, 0] - v[:, 1])
    return torch.cat(ids), torch.cat(margin)


def compare(ia, ma, ib, confident: float) -> dict:
    eq = ia == ib
    bad = (~eq).nonzero().view(-1).tolist()
    mg = ma[~eq].tolist()
    return {"positions": int(eq.numel()), "agree": int(eq.sum()), "agree_frac": round(float(eq.float().mean()), 5),
            "confident_disagree": sum(m >= confident for m in mg),
            "disagree_ref_margins": [round(m, 4) for m in sorted(mg, reverse=True)[:12]],
            "disagree_positions": bad[:12]}


@torch.no_grad()
def rows_path(eng: FlashNextEngine, ids: list[int], st, block: int = 32):
    """The prompt through verify blocks of `block` chain rows, each committed: (top-1, margin) per row."""
    i1, m1 = [], []
    for i in range(0, len(ids), block):
        toks = ids[i:i + block]
        eng.verify(toks, list(range(-1, len(toks) - 1)), st)
        v, j = torch.topk(eng._rb[len(toks)]["logits"], 2, dim=-1)
        i1.append(j[:, 0].clone())
        m1.append((v[:, 0] - v[:, 1]).clone())
        eng.commit(st, list(range(len(toks))))
    eng.flush()
    return torch.cat(i1), torch.cat(m1)


@torch.no_grad()
def check(eng: FlashNextEngine, ids: list[int], chunk: int, new: int, eos: set, confident: float, mtp: bool,
          rows: bool = False) -> dict:
    dev = eng.device
    A, B = eng.new_state(eng.max_len), eng.new_state(eng.max_len)
    ia, ma, ib = [], [], []
    sa, sb = [], []
    ta = tb = 0.0
    for i in range(0, len(ids), chunk):
        t = torch.tensor(ids[i:i + chunk], device=dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        eng.forward(t, A, head="none")
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        st_a = eng.stack
        eng.forward_pre(t, B, head="none")
        torch.cuda.synchronize()
        tb += time.perf_counter() - t1
        ta += t1 - t0
        st_b = eng.stack
        x, m = head_top2(eng, st_a)
        y, _ = head_top2(eng, st_b)
        ia.append(x), ma.append(m), ib.append(y)
        if mtp:
            sa.append(st_a)
            sb.append(st_b)
        if A.length > 4096:
            torch.cuda.empty_cache()
    eng.build_pool(A)
    ia, ma, ib = torch.cat(ia), torch.cat(ma), torch.cat(ib)
    res = {"prompt_tokens": len(ids), "ref_s": round(ta, 2), "kernels_s": round(tb, 2),
           "teacher_forced": compare(ia, ma, ib, confident)}
    states = [(A, int(ia[-1])), (B, int(ib[-1]))]
    if rows:
        C = eng.new_state(eng.max_len)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ic, mc = rows_path(eng, ids, C)
        res["rows_s"] = round(time.perf_counter() - t0, 2)
        res["rows_vs_ref"] = compare(ia, ma, ic, confident)
        res["kernels_vs_rows"] = compare(ic, mc, ib, confident)
        states.append((C, int(ic[-1])))
    # continuation: greedy from each state (the first token is each path's own last-row argmax)
    outs = []
    for st, first in states:
        g = DecodeGraph(eng, st)
        toks, margins = [first], []
        while len(toks) < new and toks[-1] not in eos:
            lg = g(toks[-1])[0]
            v, i = torch.topk(lg, 2)
            toks.append(int(i[0]))
            margins.append(float(v[0] - v[1]))
        outs.append((toks, margins))
        del g
    def cont(x, y, m0):
        (tx, mg), (ty, _) = x, y
        d = next((j for j in range(min(len(tx), len(ty))) if tx[j] != ty[j]), None)
        return {"tokens": len(tx), "equal": tx == ty, "first_diff": d,
                "margin_at_diff": None if d is None else round(float(m0) if d == 0 else mg[d - 1], 4)}
    res["continuation"] = cont(outs[0], outs[1], ma[-1])
    if rows:
        res["continuation_rows_vs_ref"] = cont(outs[0], outs[2], ma[-1])
        res["continuation_kernels_vs_rows"] = cont(outs[2], outs[1], mc[-1])
    if mtp:
        sa, sb = torch.cat(sa), torch.cat(sb)
        L = len(ids)
        ma_st, mb_st, mc_st = (eng.new_mtp_state(eng.max_len) for _ in range(3))
        xa, xm, xb, xc = [], [], [], []
        for i in range(0, L - 1, chunk):
            j = min(i + chunk, L - 1)
            nt = torch.tensor(ids[i + 1:j + 1])
            _, ha = eng.mtp(sa[i:j], nt.to(dev), ma_st, head="x")
            hb = eng.mtp_pre(sb[i:j], nt, mb_st)
            hc = eng.mtp_pre(sa[i:j], nt, mc_st)
            a1, a2 = mtp_top2(eng, ha)
            xa.append(a1), xm.append(a2), xb.append(mtp_top2(eng, hb)[0]), xc.append(mtp_top2(eng, hc)[0])
        xa, xm = torch.cat(xa), torch.cat(xm)
        res["mtp_prime"] = compare(xa, xm, torch.cat(xb), confident)
        res["mtp_same_streams"] = compare(xa, xm, torch.cat(xc), confident)
    del A, B, states
    torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ctx", default="1024,8192")
    ap.add_argument("--classes", default="prose,code")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--confident", type=float, default=1.0)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--rows", action="store_true", help="also the decode path's own kernels (verify blocks)")
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    ctxs = [int(x) for x in a.ctx.split(",")]
    idx, W, banks, tables, timing = load(a.pack, "cuda", mtp=a.mtp)
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=max(ctxs) + a.new + 64)
    tok = AutoTokenizer.from_pretrained(a.pack)
    eos = set(idx.cfg.eos_token_ids)
    res = {"pack": a.pack, "load": timing, "mem_after_load_gib": mem_available_gib(),
           "knobs": {"ctx": ctxs, "chunk": a.chunk, "new": a.new, "confident_margin": a.confident, "greedy": True,
                     "kv_dtype": "bf16", "mtp": a.mtp, "rows_path": a.rows}, "rows": []}
    for n in ctxs:
        for c in a.classes.split(","):
            r = {"ctx": n, "class": c, **check(eng, class_prompt(tok, c, n), a.chunk, a.new, eos, a.confident, a.mtp, a.rows)}
            r["mem_gib"] = mem_available_gib()
            res["rows"].append(r)
            print(f"[prefill-check] {json.dumps(r)}", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
