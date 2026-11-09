"""SPD-249 P2 on the board: the prefill with QWEN38_FN_PF_HC_FUSED off and on, bit for bit, on a real pack.

    PYTHONPATH=<repo> python tools/fn_pf_hc_check.py --pack ~/flashnext-packs/3.05 --ctx 2048,8192,32768 \
        --sites --json out.json

One load, one process. Per context (tools/fn_generate.py's prose prompt filled to `--ctx` tokens) the prompt is
prefilled on a fresh state with the switches off, then on (engine `PF_HC_FUSED` and `PF_INJ_FUSED`, read at each
`_mix_pre` / `_inject_pre` call):

  logits      the last row's fp32 logits (`prefill`'s output), bit for bit
  streams     every row's pre-final-mixer streams (`prompt_stack`, keep_stack), bit for bit
  state       every cache and recurrent tensor of the state (KV, indexer keys, pooled keys, GDN state and conv,
              PLE conv and context) by a bit checksum on the device
  times       each arm's prefill wall time (synchronized; the first context also runs once untimed per arm, so
              both arms' kernels are built and warm)

`--sites`: one more pass with the switches on in which every `_mix_pre` and `_inject_pre` call also runs the
replaced path on the same inputs and compares the results (x and di; h and n) bit for bit: the fused launches
against the unfused chain on every real layer input of the prompt (every layer's two sites, every chunk).

`--joint N`: also N prompts of 2,048 tokens in one joint prefill (engine/models/flashnext/joint_prefill.py), off
and on, each prompt's logits and streams bit for bit.

Exit 1 on any difference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext import engine as E  # noqa: E402
from engine.models.flashnext.engine import FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402
from tools.fn_generate import class_prompt, mem_available_gib  # noqa: E402


def checksum(t: torch.Tensor) -> list:
    """Two int64 sums over a tensor's bits (plain and position-weighted): equal tensors give equal sums, and a single
    flipped bit moves both."""
    t = t.detach().contiguous()
    v = t.view(torch.uint8).reshape(-1)
    pad = (-v.numel()) % 4
    if pad:
        v = torch.cat([v, v.new_zeros(pad)])
    w = v.view(torch.int32).to(torch.int64)
    idx = torch.arange(w.numel(), device=w.device, dtype=torch.int64) % 1000003 + 1
    return [int(w.sum()), int((w * idx).sum())]


def state_sums(st) -> dict:
    out = {}
    for name in ("k", "v", "ik", "conv", "S", "ple_conv", "pool"):
        for li, t in sorted(getattr(st, name).items(), key=lambda kv: str(kv[0])):
            if isinstance(t, torch.Tensor):
                out[f"{name}{li}"] = checksum(t)
    out["ple_ctx"] = {str(k): (v.tolist() if isinstance(v, torch.Tensor) else v) for k, v in st.ple_ctx.items()}
    out["pool_n"] = {str(k): int(v) for k, v in st.pool_n.items()}
    out["length"] = st.length
    return out


def same_bits(x: torch.Tensor, y: torch.Tensor) -> bool:
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    return bool(torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)))


def switch(on: bool) -> None:
    E.PF_HC_FUSED = E.PF_INJ_FUSED = on


def arm(eng: FlashNextEngine, ids: list[int], on: bool) -> dict:
    switch(on)
    st = eng.new_state(eng.max_len)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    lg = eng.prefill(torch.tensor(ids), st, keep_stack=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    out = {"logits": lg.clone(), "stack": eng.prompt_stack, "state": state_sums(st), "s": dt}
    eng.prompt_stack = None
    del st
    torch.cuda.empty_cache()
    return out


class SiteCheck:
    """`_mix_pre` and `_inject_pre` wrapped: the fused results, and the replaced path's on the same inputs, compared bit
    for bit."""

    def __init__(self, eng: FlashNextEngine):
        self.eng, self.mix0, self.inj0 = eng, eng._mix_pre, eng._inject_pre
        self.st = {"mix": [0, 0, 0, 0], "inject": [0, 0, 0, 0]}     # calls, rows, differing calls, fused calls
        self.first_bad = None
        eng._mix_pre, eng._inject_pre = self.mix, self.inject

    def _cmp(self, kind, rows, a, b, what):
        s = self.st[kind]
        s[0] += 1
        s[1] += rows
        if not all(x is None and y is None or same_bits(x, y) for x, y in zip(a, b)):
            s[2] += 1
            if self.first_bad is None:
                self.first_bad = {"kind": kind, "where": what, "rows": rows,
                                  "diffs": [None if x is None else int((x != y).sum()) for x, y in zip(a, b)]}

    def mix(self, n, p):
        switch(False)
        r0 = self.mix0(n, p)
        switch(True)
        r1 = self.mix0(n, p)
        self._cmp("mix", n.shape[0], r0, r1, p)
        from tools import fn_pf_hc as HC
        W = self.eng.W
        self.st["mix"][3] += int(HC.ok(n, W[p + ".down_inject"], W[p + ".input_mix_weight_up.weight"], self.eng.cfg.hc_count))
        return r1

    def inject(self, h, out, di, R, w, hc, eps):
        switch(False)
        r0 = self.inj0(h, out, di, R, w, hc, eps)
        switch(True)
        r1 = self.inj0(h, out, di, R, w, hc, eps)
        self._cmp("inject", h.shape[0], r0, r1, "w None" if w is None else "w")
        from tools import fn_pf_hc as HC
        self.st["inject"][3] += int(HC.inject_ok(h, out, hc))
        return r1

    def restore(self):
        self.eng._mix_pre, self.eng._inject_pre = self.mix0, self.inj0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ctx", default="2048,8192,32768")
    ap.add_argument("--sites", action="store_true")
    ap.add_argument("--joint", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from tools import fn_prefill_kernels as PK
    ctxs = [int(x) for x in a.ctx.split(",")]
    idx, W, banks, tables, timing = load(a.pack, "cuda")
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=max(ctxs) + 64)
    tok = AutoTokenizer.from_pretrained(a.pack)
    res = {"pack": a.pack, "load": timing, "mem_after_load_gib": mem_available_gib(),
           "knobs": {"FN_PREFILL_MM": PK.MM, "FN_PREFILL_CHUNK": E.PREFILL_CHUNK, "FN_MIX": os.environ.get("QWEN38_FN_MIX", "int8"),
                     "kv_dtype": "bf16", "prefix_cache": "off", "env": {k: v for k, v in os.environ.items()
                                                                        if k.startswith(("QWEN38_FN_", "QSE_"))}},
           "rows": [], "ok": True}
    warm = class_prompt(tok, "prose", 256)
    for on in (False, True):
        arm(eng, warm, on)
    for n in ctxs:
        ids = class_prompt(tok, "prose", n)
        if n == ctxs[0]:
            for on in (False, True):
                arm(eng, ids, on)
        off = arm(eng, ids, False)
        on = arm(eng, ids, True)
        row = {"ctx": len(ids), "logits_equal": same_bits(off["logits"], on["logits"]),
               "streams_equal": same_bits(off["stack"], on["stack"]), "state_equal": off["state"] == on["state"],
               "argmax": [int(off["logits"].argmax()), int(on["logits"].argmax())],
               "max_abs_logit_diff": float((off["logits"] - on["logits"]).abs().max()),
               "off_s": round(off["s"], 3), "on_s": round(on["s"], 3),
               "off_tok_s": round(len(ids) / off["s"], 1), "on_tok_s": round(len(ids) / on["s"], 1)}
        if not row["state_equal"]:
            row["state_diff"] = [k for k in off["state"] if off["state"][k] != on["state"].get(k)][:20]
        del off, on
        torch.cuda.empty_cache()
        if a.sites:
            sc = SiteCheck(eng)
            try:
                st = eng.new_state(eng.max_len)
                eng.prefill(torch.tensor(ids), st)
                del st
            finally:
                sc.restore()
            row["sites"] = {k: {"calls": v[0], "rows": v[1], "differ": v[2], "fused": v[3]} for k, v in sc.st.items()}
            row["sites"]["first_bad"] = sc.first_bad
            torch.cuda.empty_cache()
        row["ok"] = bool(row["logits_equal"] and row["streams_equal"] and row["state_equal"]
                         and (not a.sites or (sc.first_bad is None and all(v[0] > 0 and v[3] == v[0] for v in sc.st.values()))))
        res["ok"] &= row["ok"]
        res["rows"].append(row)
        print("[pf-hc] " + json.dumps(row), flush=True)
    if a.joint:
        from engine.models.flashnext import joint_prefill as JP
        prompts = [class_prompt(tok, cls, 2048 - 37 * i) for i, cls in
                   zip(range(a.joint), ["prose", "code"] * a.joint)]
        got = {}
        for on in (False, True):
            switch(on)
            jobs = [(torch.tensor(t), eng.new_state(eng.max_len)) for t in prompts]
            got[on] = [(lg.clone(), stack.clone()) for lg, stack in JP.prefill_joint(eng, jobs)]
            del jobs
            torch.cuda.empty_cache()
        jr = {"prompts": len(prompts), "rows": [len(t) for t in prompts],
              "logits_equal": all(same_bits(x[0], y[0]) for x, y in zip(got[False], got[True])),
              "streams_equal": all(same_bits(x[1], y[1]) for x, y in zip(got[False], got[True]))}
        jr["ok"] = jr["logits_equal"] and jr["streams_equal"]
        res["ok"] &= jr["ok"]
        res["joint"] = jr
        print("[pf-hc] joint " + json.dumps(jr), flush=True)
    switch(False)
    print(f"[pf-hc] {'ALL EQUAL' if res['ok'] else 'DIFFERENT'}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
