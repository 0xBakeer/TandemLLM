"""Where a Flash-Next prefill spends its time, by part (ENG-225's profile).

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_prefill_profile.py --pack ~/fn2/packs/3.05 \
        --ctx 2048,8192,32768 --chunk 2048 --json out.json

Per context (tools/fn_generate.py's prose prompt filled to `--ctx` tokens): one prefill with every part
timed (a device sync before and after each call, time EXCLUSIVE of the parts it calls), then one
prefill untimed for the honest wall time and tok/s. Parts: the PLE branch, the hyper-connection norms,
mixers (down GEMV, mix_up) and injects, EXL3 projections, other projections, the GDN conv and gates, the
delta rule, attention (the core, the QSA indexer), the router, the experts, the final mixer and head.
`--path kernel` profiles `FlashNextEngine.prefill_kernels` (ENG-225); `ref` the reference forward.

`--kernels N`: one more run under torch.profiler (device activity only): the top N kernels and every kernel's time by
stage (`STAGES`). `--sites N` (SPD-249 P4): one more run under torch.profiler with the host's ops and their Python
stacks, and the launches of the "copies and elementwise" stage by the line that issued them: a kernel belongs to the
op that launched it (the profiler's correlation), the op to the innermost frame of this repo on its stack. Launches,
device seconds, the op and the kernel's kind a line, the top N. The stacks slow the host, not the kernels: the
seconds are device time, to lay beside the stage table, and the run's wall time is not a prefill's.
`--sync-sites N`: the same question asked another way, one more run with every torch call timed on the host with a
device sync before and after it (`SyncSites`), by the innermost repo line that made the call: calls and wall seconds
of each (line, function), the top N. A kernel launched outside torch (the Triton kernels) is not a torch call: its
time is drained by the sync before the next call and counted nowhere.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402
from tools.fn_generate import class_prompt, mem_available_gib  # noqa: E402


class PartTimer:
    """Exclusive wall time of wrapped calls, each bracketed by device syncs."""

    def __init__(self):
        self.t: dict = defaultdict(float)
        self.n: dict = defaultdict(int)
        self.stack: list = []
        self.undo: list = []

    def wrap(self, obj, name: str, label):
        f = getattr(obj, name)

        def g(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.stack.append(0.0)
            try:
                return f(*a, **kw)
            finally:
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                child = self.stack.pop()
                lab = label(*a) if callable(label) else label
                self.t[lab] += dt - child
                self.n[lab] += 1
                if self.stack:
                    self.stack[-1] += dt
        self.undo.append((obj, name, obj.__dict__.get(name) if hasattr(obj, "__dict__") else None, f))
        setattr(obj, name, g)

    def restore(self) -> None:
        for obj, name, own, f in reversed(self.undo):
            if own is None and not isinstance(obj, type(sys)):
                delattr(obj, name)                       # back to the class's method
            else:
                setattr(obj, name, f)
        self.undo = []


def instrument(eng: FlashNextEngine) -> PartTimer:
    from tools import fn_kernels as K
    from tools import fn_seq_kernels as SK
    from tools.exl3_linear import EXL3Block
    T = PartTimer()

    def lin_label(x, w):
        return "exl3 projections" if isinstance(w, EXL3Block) else "other projections"
    T.wrap(eng, "lin", lin_label)
    for name, lab in (("ple", "ple"), ("gdn", "gdn other"), ("attention", "attention other"),
                      ("delta_rule", "gdn delta rule"), ("qsa_mask", "attention qsa"), ("final", "final mixer + head"),
                      ("moe", "moe (reference)"), ("moe_parts", "moe other"), ("build_pool", "qsa pool"),
                      ("_ple_pre", "ple"), ("_gdn_pre", "gdn other"), ("_attn_pre", "attention other"),
                      ("_moe_pre", "moe other"), ("_mix_pre", "mixers (prefill)")):
        if hasattr(eng, name):
            T.wrap(eng, name, lab)
    for name, lab in (("gemv", "mixer down / router gemv"), ("mix_up", "mixer up"), ("hc_norm", "norms + injects"),
                      ("inject_norm", "norms + injects"), ("combine_inject_norm", "norms + injects"),
                      ("router", "router"), ("moe_experts", "moe experts"), ("gemv_bf16", "other projections")):
        T.wrap(K, name, lab)
    for name, lab in (("attn_rows", "attention core"), ("qsa_scores", "attention qsa"), ("qsa_select", "attention qsa"),
                      ("qsa_pool", "qsa pool"), ("gdn_pre", "gdn conv + gates")):
        T.wrap(SK, name, lab)
    from tools import fn_prefill_kernels as PK
    from tools import gdn_prefill_kernels as GK
    fam = weight_families(eng)

    def mm_label(x, w):
        return "library GEMM rows: " + fam.get(id(w), "other")
    T.wrap(PK, "mm_rows", mm_label)
    for name, lab in (("moe_grouped", "moe experts"), ("mix_tail", "mixer up"), ("qsa_scores_rows", "attention qsa")):
        T.wrap(PK, name, lab)
    T.wrap(GK, "fused_chunk_prefill", "gdn delta rule")
    return T


def weight_families(eng: FlashNextEngine) -> dict:
    """SPD-249 P0: id(weight) -> the family `mm_rows` serves it to (mixers, router, GDN a|b, PLE)."""
    out = {}
    for k, w in eng.W.items():
        if k.endswith((".down_inject", ".input_mix_weight_up.weight")):
            out[id(w)] = "mixers"
        elif k.endswith("mlp.router"):
            out[id(w)] = "router"
        elif k.endswith("linear_attn.ab"):
            out[id(w)] = "gdn a|b"
        elif ".ple." in k:
            out[id(w)] = "ple"
    return out


#: SPD-249 P0: a kernel's stage by its name (first match), for the stage table of a prefill profile
STAGES = (
    ("expert grouped GEMMs", ("group_gemm_kernel",)),
    ("expert rotations + combine + routing", ("pair_rot_kernel", "pair_silu_rot_kernel", "moe_combine_kernel",
                                              "router_kernel", "RadixSort", "radixSort", "DeviceScan",
                                              "kernelHistogram", "histogram")),
    ("hyper-connection norms + mix tails", ("inject_norm_kernel", "hc_norm_kernel", "mix_tail_kernel",
                                           "mix_up", "combine_inject", "pf_mix_a_kernel")),
    ("EXL3 projections: rotations", ("rot_in_kernel", "rot_out_kernel")),
    ("EXL3 projections: slab decode", ("recon_kernel",)),
    ("library fp32 GEMM (mm_rows)", ("magma_sgemm", "sgemm", "gemmSN", "gemv2T", "gemvx")),
    ("library fp16 GEMM (EXL3 slabs)", ("s16816gemm", "s1688gemm", "nvjet", "cutlass_80_tensorop", "xmma")),
    ("DeltaNet kernels", ("_scan", "_intra", "gdn_pre_kernel", "gated_norm_kernel", "fused_chunk", "_prep",
                          "chunk_", "_ut", "_wy")),
    ("attention core + QSA", ("_attn_rows", "attn_prep_kernel", "qsa_", "gate_mul_kernel")),
    ("PLE kernels", ("ple_", "decode_rows", "_decode")),
    ("copies and elementwise", ("direct_copy", "copy_kernel", "Memcpy", "Memset", "elementwise_kernel",
                                "FillFunctor", "reduce_kernel", "index_elementwise", "index_copy", "index_select",
                                "indexSelect", "cat_", "CatArray", "fill_", "scatter", "gather")),
)


def stage_of(name: str) -> str:
    for stage, keys in STAGES:
        if any(k in name for k in keys):
            return stage
    return "other kernels"


COPIES = "copies and elementwise"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def site_of(stack: list) -> str:
    """The innermost frame of this repo on a profiler stack ("file(line): function"), the repo's path dropped."""
    for fr in stack:
        if fr.startswith(ROOT + os.sep) and "fn_prefill_profile.py" not in fr:
            return fr[len(ROOT) + 1:]
    return "(no repo frame)"


def copy_sites(prof, top: int) -> dict:
    """SPD-249 P4: the launches of the "copies and elementwise" stage by the line that issued them."""
    from torch.autograd import DeviceType
    keys = dict(STAGES)[COPIES]
    by, line, linked = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
    for fe in prof.events():
        ks = getattr(fe, "kernels", None)
        if not ks or fe.device_type != DeviceType.CPU:
            continue
        site = outer = None
        for k in ks:
            stage = stage_of(k.name)
            us = getattr(k, "duration", 0)
            linked[stage][0] += us / 1e6
            linked[stage][1] += 1
            if stage != COPIES:
                continue
            if site is None:
                # the kernel hangs on the event that launched it: the op itself or a runtime call under it
                op = fe
                while not op.name.startswith("aten::") and op.cpu_parent is not None:
                    op = op.cpu_parent
                outer, src = op, fe
                while outer.cpu_parent is not None and outer.cpu_parent.name.startswith("aten::"):
                    outer = outer.cpu_parent
                while not src.stack and src.cpu_parent is not None:
                    src = src.cpu_parent
                site = site_of(src.stack or [])
            kind = next(x for x in keys if x in k.name)
            for d, key in ((by, (site, outer.name, op.name, kind)), (line, site)):
                d[key][0] += us / 1e6
                d[key][1] += 1
    tot = sum(v[0] for v in by.values())
    return {"method": "torch.profiler, host ops with Python stacks + device kernels; a kernel's line is the innermost "
                      "repo frame of the op that launched it; device seconds",
            "copies_s": round(tot, 3), "copies_launches": sum(v[1] for v in by.values()),
            "linked_by_stage": {k: (round(v[0], 3), v[1]) for k, v in sorted(linked.items(), key=lambda kv: -kv[1][0])},
            "lines": [(k, round(v[0], 4), v[1]) for k, v in sorted(line.items(), key=lambda kv: -kv[1][0])[:top]],
            "sites": [(*k, round(v[0], 4), v[1]) for k, v in sorted(by.items(), key=lambda kv: -kv[1][0])[:top]]}


class SyncSites(torch.overrides.TorchFunctionMode):
    """Every torch call under the mode timed alone: a device sync, the call, a sync; by the innermost repo frame."""

    def __init__(self):
        super().__init__()
        self.by = defaultdict(lambda: [0.0, 0])

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", None) or str(func)
        if name in ("__get__", "__set__", "size", "dim", "numel", "data_ptr", "element_size", "is_contiguous", "stride"):
            return func(*args, **(kwargs or {}))
        f = sys._getframe(1)
        while f is not None and not (f.f_code.co_filename.startswith(ROOT + os.sep)
                                     and not f.f_code.co_filename.endswith("fn_prefill_profile.py")):
            f = f.f_back
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = func(*args, **(kwargs or {}))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        site = "(no repo frame)" if f is None else f"{f.f_code.co_filename[len(ROOT) + 1:]}({f.f_lineno}): {f.f_code.co_name}"
        g = self.by[(site, name)]
        g[0] += dt
        g[1] += 1
        return out

    def table(self, top: int) -> dict:
        rows = sorted(self.by.items(), key=lambda kv: -kv[1][0])
        return {"method": "TorchFunctionMode: each torch call between two device syncs, host wall seconds, by the "
                          "innermost repo frame; kernels launched outside torch are not counted",
                "torch_calls": sum(v[1] for v in self.by.values()), "torch_s": round(sum(v[0] for v in self.by.values()), 3),
                "sites": [(k[0], k[1], round(v[0], 4), v[1]) for k, v in rows[:top]]}


def run(eng: FlashNextEngine, ids: list[int], path: str, chunk: int) -> float:
    st = eng.new_state(eng.max_len)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    eng.prefill(torch.tensor(ids), st, chunk=chunk or None, path="kernels" if path == "kernel" else "ref")
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    del st
    torch.cuda.empty_cache()
    return dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ctx", default="2048,8192")
    ap.add_argument("--chunk", type=int, default=0, help="rows a chunk (0: the path's default)")
    ap.add_argument("--path", default="ref", choices=("ref", "kernel"))
    ap.add_argument("--timed", default="1", help="0: wall time only (no per-part profile)")
    ap.add_argument("--wall-max", type=int, default=1 << 30, help="no untimed run past this context")
    ap.add_argument("--kernels", type=int, default=0, help="also the top N kernels of one untimed run (torch.profiler)")
    ap.add_argument("--repeat", type=int, default=1, help="untimed runs (wall_s is the last; wall_runs all of them)")
    ap.add_argument("--sites", type=int, default=0, help="also the top N lines that launch copies and elementwise kernels")
    ap.add_argument("--sites-min", type=int, default=0, help="no --sites / --sync-sites run below this context")
    ap.add_argument("--sync-sites", type=int, default=0, help="also the top N (line, torch function) by sync-timed seconds")
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    ctxs = [int(x) for x in a.ctx.split(",")]
    idx, W, banks, tables, timing = load(a.pack, "cuda")
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=max(ctxs) + 64)
    tok = AutoTokenizer.from_pretrained(a.pack)
    res = {"pack": a.pack, "path": a.path, "chunk": a.chunk, "load": timing, "mem_after_load_gib": mem_available_gib(),
           "knobs": {"kv_dtype": "bf16", "prefix_cache": "off", "head": "last row of the last chunk",
                     "env": {k: v for k, v in os.environ.items() if k.startswith("QWEN38_FN_P")}}, "rows": []}
    from engine.models.flashnext import engine as E
    from engine.models.flashnext import layer_major as LM
    from tools import fn_prefill_kernels as PK
    res["knobs"]["effective"] = {
        "FN_PF_WINDOW": LM.ROWS, "FN_PF_ROT_ROWS": int(PK.ROT_ROWS), "FN_PF_HC_FUSED": int(E.PF_HC_FUSED),
        "FN_PF_INJ_FUSED": int(E.PF_INJ_FUSED), "FN_PF_ATTN_STAGES": E.PF_ATTN_STAGES, "FN_PREFILL_MM": PK.MM,
        "FN_PREFILL_CHUNK": E.PREFILL_CHUNK, "FN_GROUP_ROWS": PK.GROUP_ROWS, "prefill_path": eng.prefill_path}
    print(f"[prefill-profile] knobs {json.dumps(res['knobs']['effective'])}", flush=True)
    nwin, window = [0], LM.window

    def counted(*args, **kw):
        nwin[0] += 1
        return window(*args, **kw)
    LM.window = counted
    run(eng, class_prompt(tok, "prose", 256), a.path, a.chunk)          # build the kernels
    for n in ctxs:
        ids = class_prompt(tok, "prose", n)
        row = {"ctx": len(ids)}
        if a.timed == "1":
            T = instrument(eng)
            dt = run(eng, ids, a.path, a.chunk)
            T.restore()
            parts = sorted(T.t.items(), key=lambda kv: -kv[1])
            row["timed_s"] = round(dt, 2)
            row["parts_s"] = {k: round(v, 3) for k, v in parts}
            row["parts_share"] = {k: round(v / dt, 3) for k, v in parts}
            row["calls"] = dict(T.n)
        if len(ids) <= a.wall_max:
            walls = [run(eng, ids, a.path, a.chunk) for _ in range(max(1, a.repeat))]
            dt = walls[-1]
            row["wall_s"] = round(dt, 2)
            row["tok_s"] = round(len(ids) / dt, 1)
            row["wall_runs"] = [round(w, 3) for w in walls]
        if a.kernels:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                run(eng, ids, a.path, a.chunk)
            per = {}
            for e in prof.key_averages():
                us = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
                if us:
                    per[e.key] = (per.get(e.key, (0.0, 0))[0] + us / 1e6, e.count)
            row["gpu_s"] = round(sum(v[0] for v in per.values()), 2)
            row["kernels"] = [(k[:160], round(v[0], 3), v[1]) for k, v in sorted(per.items(), key=lambda kv: -kv[1][0])[:a.kernels]]
            stages: dict = defaultdict(lambda: [0.0, 0, 0])
            for k, (s_, n_) in per.items():
                g = stages[stage_of(k)]
                g[0] += s_
                g[1] += n_
                g[2] += 1
            row["stages_s"] = {k: (round(v[0], 3), v[1], v[2]) for k, v in sorted(stages.items(), key=lambda kv: -kv[1][0])}
            row["kernel_kinds"] = len(per)
            del prof
        if a.sync_sites and len(ids) >= a.sites_min:
            m = SyncSites()
            with m:
                row["sync_sites_run_s"] = round(run(eng, ids, a.path, a.chunk), 2)
            row["sync_sites"] = m.table(a.sync_sites)
        if a.json:                                                         # the stacks' run may not fit: keep the rest
            json.dump({**res, "rows": res["rows"] + [row]}, open(a.json, "w"), indent=1)
        if a.sites and len(ids) >= a.sites_min:
            from torch.profiler import ProfilerActivity, profile
            try:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
                    row["sites_run_s"] = round(run(eng, ids, a.path, a.chunk), 2)
                row["copy_sites"] = copy_sites(prof, a.sites)
                del prof
            except Exception as e:  # noqa: BLE001
                row["copy_sites"] = {"error": f"{type(e).__name__}: {e}"[:400]}
        row["mem_gib"] = mem_available_gib()
        row["windows"], nwin[0] = nwin[0], 0                              # layer-major windows over the row's runs
        res["rows"].append(row)
        print(f"[prefill-profile] {json.dumps(row)}", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
