"""SPD-249 P2 on the board: the prefill with QWEN38_FN_PF_INJ_FUSED / QWEN38_FN_PF_HC_FUSED off and on, bit for bit,
on a real pack, and both arms' times.

    PYTHONPATH=<repo> python tools/fn_pf_hc_check.py --pack ~/flashnext-packs/3.05 --ctx 2048,8192,32768 \
        --modes fp32,tc --reps 2 --sites --joint 3 --json out.json

One load, one process. Per `mm_rows` mode of `--modes` (QWEN38_FN_PREFILL_MM, set in the process; the first is the
timed one) and context (tools/fn_generate.py's prose prompt filled to `--ctx` tokens) the prompt is prefilled on a
fresh state by each arm in turn, `--reps` times round robin (once in the later modes):

  base        both switches off
  inj         QWEN38_FN_PF_INJ_FUSED
  fused       both
  st1         QWEN38_FN_PF_ATTN_STAGES=1 (SPD-249 P6: the attention split launch's stages), the switches off
  fused_st1   both switches and QWEN38_FN_PF_ATTN_STAGES=1
  win8k       QWEN38_FN_PF_WINDOW=8192 (SPD-249 P1: layer-major windows), the rest off
  win32k      QWEN38_FN_PF_WINDOW=32768 and QWEN38_FN_PF_ROT_ROWS, the rest off
  all8k       both switches, QWEN38_FN_PF_ATTN_STAGES=1 and QWEN38_FN_PF_WINDOW=8192
  all32k      both switches, QWEN38_FN_PF_ATTN_STAGES=1, QWEN38_FN_PF_WINDOW=32768 and QWEN38_FN_PF_ROT_ROWS
  nc          QWEN38_FN_PF_NOCOPY (SPD-249 P4: a chunk's copies left out), the rest off
  all8knc     all8k and QWEN38_FN_PF_NOCOPY
  all32knc    all32k and QWEN38_FN_PF_NOCOPY

(the tool sets every one of these itself, whatever the process's environment says) and every run is compared with the first base run (so the base's second run shows that a prefill repeats its bits):

  logits      the last row's fp32 logits (`prefill`'s output), bit for bit
  streams     every row's pre-final-mixer streams (`prompt_stack`, keep_stack), bit for bit
  state       every cache and recurrent tensor of the state (KV, indexer keys, pooled keys, GDN state and conv,
              PLE conv and context) by a bit checksum on the device
  times       each run's prefill wall time (device synchronized), an arm's median, its gain on the base's median;
              a mode's first context runs every arm once untimed first, so each arm's kernels are built and warm;
              `--warm-each`: every context does (the arms in order, the base first; the dropped times are `warm_s`)

`--sites`: one more pass with the switches on (the window, rotation order and stages of `--sites-arm`) in which every `_mix_pre` and `_inject_pre` call also runs the
replaced path on the same inputs and compares the results (x and di; h and n) bit for bit: the fused launches
against the unfused chain on every real layer input of the prompt (every layer's two sites, every chunk), with the
count of calls the fused path took and of mixers that got n's fp32 form from an inject.

`--joint N`: also N prompts in one joint prefill (engine/models/flashnext/joint_prefill.py) per mode and arm, each
prompt's logits and streams bit for bit.

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
from engine.models.flashnext import layer_major as LM  # noqa: E402
from engine.models.flashnext.engine import FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402
from tools.fn_generate import class_prompt, mem_available_gib  # noqa: E402

#: an environment variable whose name holds one of these never goes into the record
SECRET = ("KEY", "TOKEN", "SECRET", "PASS", "AUTH", "CRED")


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


def state_sums(eng, st) -> dict:
    eng.flush(st)
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


#: (PF_HC_FUSED, PF_INJ_FUSED, PF_ATTN_STAGES[, layer_major.ROWS, fn_prefill_kernels.ROT_ROWS[, PF_NOCOPY]])
ARMS = {"base": (False, False, 2), "inj": (False, True, 2), "fused": (True, True, 2), "st1": (False, False, 1),
        "fused_st1": (True, True, 1), "win8k": (False, False, 2, 8192, False), "win32k": (False, False, 2, 32768, True),
        "all8k": (True, True, 1, 8192, False), "all32k": (True, True, 1, 32768, True),
        "nc": (False, False, 2, 0, False, True), "all8knc": (True, True, 1, 8192, False, True),
        "all32knc": (True, True, 1, 32768, True, True)}


def fuse(hc: bool, inj: bool | None = None) -> None:
    E.PF_HC_FUSED, E.PF_INJ_FUSED = hc, hc if inj is None else inj


def switch(hc: bool, inj: bool | None = None, stages: int = 2, window: int = 0, rot: bool = False,
           nocopy: bool = False) -> None:
    from tools import fn_prefill_kernels as PK
    fuse(hc, inj)
    E.PF_ATTN_STAGES, LM.ROWS, PK.ROT_ROWS, E.PF_NOCOPY = stages, window, rot, nocopy


#: the layer-major windows a prefill ran (`layer_major.window` counted)
WINDOWS = [0]


def _count_windows() -> None:
    f = LM.window

    def window(*a, **kw):
        WINDOWS[0] += 1
        return f(*a, **kw)
    LM.window = window


def arm(eng: FlashNextEngine, ids: list[int], name: str) -> dict:
    switch(*ARMS[name])
    st = eng.new_state(eng.max_len)
    torch.cuda.synchronize()
    WINDOWS[0] = 0
    held = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    lg = eng.prefill(torch.tensor(ids), st, keep_stack=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    out = {"logits": lg.clone(), "stack": eng.prompt_stack, "state": state_sums(eng, st), "s": dt,
           "peak_gib": (torch.cuda.max_memory_allocated() - held) / (1 << 30), "windows": WINDOWS[0]}
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
        self.n32 = 0                                                  # mixers that got n's fp32 form
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
        from tools import fn_pf_hc as HC
        held = getattr(self.eng, "_n32", None)
        self.n32 += int(bool(held) and id(n) in held)
        fuse(True)
        r1 = self.mix0(n, p)                    # first: it takes n's fp32 form when an inject left one
        fuse(False)
        r0 = self.mix0(n, p)
        fuse(True)
        self._cmp("mix", n.shape[0], r0, r1, p)
        self.st["mix"][3] += int(HC.mix_ok(n, self.eng.cfg.hc_count))
        return r1

    def inject(self, h, out, di, R, w, hc, eps):
        from tools import fn_pf_hc as HC
        fuse(False)
        r0 = self.inj0(h, out, di, R, w, hc, eps)
        fuse(True)
        r1 = self.inj0(h, out, di, R, w, hc, eps)
        self._cmp("inject", h.shape[0], r0, r1, "w None" if w is None else "w")
        self.st["inject"][3] += int(HC.inject_ok(h, out, hc))
        return r1

    def restore(self):
        self.eng._mix_pre, self.eng._inject_pre = self.mix0, self.inj0


def med(v: list) -> float:
    v = sorted(v)
    return (v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ctx", default="2048,8192,32768")
    ap.add_argument("--modes", default="", help="mm_rows modes, the timed one first (default: the process's own)")
    ap.add_argument("--arms", default="base,inj,fused")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--sites", action="store_true")
    ap.add_argument("--sites-arm", default="fused", help="the arm whose window, rotation order and stages the pass takes")
    ap.add_argument("--warm-each", action="store_true", help="every context runs every arm once untimed first")
    ap.add_argument("--joint", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from tools import fn_prefill_kernels as PK
    ctxs = [int(x) for x in a.ctx.split(",")]
    arms = [x for x in a.arms.split(",") if x]
    assert arms[0] == "base" and all(x in ARMS for x in arms) and a.sites_arm in ARMS, arms
    mode0, own = PK.MM, (E.PF_HC_FUSED, E.PF_INJ_FUSED, E.PF_ATTN_STAGES, LM.ROWS, PK.ROT_ROWS, E.PF_NOCOPY)
    modes = [x for x in a.modes.split(",") if x] or [mode0]
    idx, W, banks, tables, timing = load(a.pack, "cuda")
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=max(ctxs) + 64)
    tok = AutoTokenizer.from_pretrained(a.pack)
    res = {"pack": a.pack, "load": timing, "mem_after_load_gib": mem_available_gib(),
           "knobs": {"FN_PREFILL_MM": mode0, "modes": modes, "FN_PREFILL_CHUNK": E.PREFILL_CHUNK,
                     "prefill_path": eng.prefill_path, "group_rows": PK.GROUP_ROWS, "reps": a.reps,
                     "arms": {k: ARMS[k] for k in arms}, "warm_each": a.warm_each, "sites_arm": a.sites_arm,
                     "window_margin_gib": LM.MARGIN / (1 << 30), "window_row_bytes": LM.ROW_BYTES,
                     "FN_MIX": os.environ.get("QWEN38_FN_MIX", "int8"), "kv_dtype": str(eng.dtype),
                     "prefix_cache": "off", "max_len": eng.max_len,
                     "env": {k: v for k, v in os.environ.items() if k.startswith(("QWEN38_", "QSE_"))
                             and not any(w in k.upper() for w in SECRET)}},
           "rows": [], "joint": [], "ok": True}
    print("[pf-hc] knobs " + json.dumps({k: v for k, v in res["knobs"].items() if k != "env"}), flush=True)
    _count_windows()
    for name in arms:
        arm(eng, class_prompt(tok, "prose", 256), name)
    for mode in modes:
        PK.MM = mode
        for n in ctxs:
            ids = class_prompt(tok, "prose", n)
            warm = {}
            if n == ctxs[0] or a.warm_each:
                for name in arms:
                    warm[name] = round(arm(eng, ids, name)["s"], 3)
            ref = None
            times = {k: [] for k in arms}
            peak, wins = {k: 0.0 for k in arms}, {}
            eq = {k: {"logits": True, "streams": True, "state": True} for k in arms}
            diff = {}
            for _ in range(a.reps if mode == modes[0] else 1):
                for name in arms:
                    got = arm(eng, ids, name)
                    times[name].append(got["s"])
                    peak[name], wins[name] = max(peak[name], got["peak_gib"]), got["windows"]
                    if ref is None:
                        ref = got
                        continue
                    eq[name]["logits"] &= same_bits(ref["logits"], got["logits"])
                    eq[name]["streams"] &= same_bits(ref["stack"], got["stack"])
                    eq[name]["state"] &= ref["state"] == got["state"]
                    if not all(eq[name].values()) and name not in diff:
                        diff[name] = {"max_abs_logit_diff": float((ref["logits"] - got["logits"]).abs().max()),
                                      "argmax": [int(ref["logits"].argmax()), int(got["logits"].argmax())],
                                      "state": [k for k in ref["state"] if ref["state"][k] != got["state"].get(k)][:20]}
                    del got
                    torch.cuda.empty_cache()
            del ref
            torch.cuda.empty_cache()
            m = {k: med(v) for k, v in times.items()}
            row = {"mode": mode, "ctx": len(ids), "equal": eq, "diff": diff, "warm_s": warm,
                   "s": {k: [round(x, 3) for x in v] for k, v in times.items()},
                   "median_s": {k: round(v, 3) for k, v in m.items()},
                   "gain_s": {k: round(m["base"] - v, 3) for k, v in m.items() if k != "base"},
                   "tok_s": {k: round(len(ids) / v, 1) for k, v in m.items()},
                   "peak_gib": {k: round(v, 3) for k, v in peak.items()}, "windows": wins}
            row["ok"] = all(all(v.values()) for v in eq.values())
            if a.sites:
                switch(True, True, *ARMS[a.sites_arm][2:])
                sc = SiteCheck(eng)
                try:
                    st = eng.new_state(eng.max_len)
                    eng.prefill(torch.tensor(ids), st)
                    del st
                finally:
                    sc.restore()
                    switch(False)
                row["sites"] = {k: {"calls": v[0], "rows": v[1], "differ": v[2], "fused": v[3]} for k, v in sc.st.items()}
                row["sites"]["n32_mixers"] = sc.n32
                row["sites"]["first_bad"] = sc.first_bad
                row["ok"] &= sc.first_bad is None and all(v[0] > 0 and v[3] == v[0] for v in sc.st.values())
                torch.cuda.empty_cache()
            res["ok"] &= row["ok"]
            res["rows"].append(row)
            print("[pf-hc] " + json.dumps(row), flush=True)
        if a.joint:
            from engine.models.flashnext import joint_prefill as JP
            prompts = [class_prompt(tok, cls, 1300 - 37 * i) for i, cls in zip(range(a.joint), ["prose", "code"] * a.joint)]
            got = {}
            for name in arms:
                switch(*ARMS[name])
                jobs = [(torch.tensor(t), eng.new_state(eng.max_len)) for t in prompts]
                got[name] = [(lg.clone(), stack.clone()) for lg, stack in JP.prefill_joint(eng, jobs)]
                del jobs
                torch.cuda.empty_cache()
            jr = {"mode": mode, "prompts": len(prompts), "rows": [len(t) for t in prompts],
                  "one_run": len(JP.runs([len(t) for t in prompts], E.PREFILL_CHUNK)) == 1,
                  "logits_equal": {k: all(same_bits(x[0], y[0]) for x, y in zip(got["base"], got[k])) for k in arms[1:]},
                  "streams_equal": {k: all(same_bits(x[1], y[1]) for x, y in zip(got["base"], got[k])) for k in arms[1:]}}
            jr["ok"] = all(jr["logits_equal"].values()) and all(jr["streams_equal"].values())
            res["ok"] &= jr["ok"]
            res["joint"].append(jr)
            del got
            torch.cuda.empty_cache()
            print("[pf-hc] joint " + json.dumps(jr), flush=True)
    PK.MM = mode0
    E.PF_HC_FUSED, E.PF_INJ_FUSED, E.PF_ATTN_STAGES, LM.ROWS, PK.ROT_ROWS, E.PF_NOCOPY = own
    print(f"[pf-hc] {'ALL EQUAL' if res['ok'] else 'DIFFERENT'}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
