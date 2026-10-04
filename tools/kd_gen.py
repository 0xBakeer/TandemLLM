"""Kolibri-1's own answers to the drafter prompts, generated with vLLM + Aleph Alpha's plugin.

Runs inside the vLLM image on an HF Jobs GPU (ops/hfjobs-kdspark.sh gen). The model is the FP8
release, the policy Aleph Alpha trained with FP8 quantisation-aware RL. Every prompt goes through
the release's own chat template (`apply_chat_template` with `reasoning_effort` and `tools`), so the
answers carry Kolibri's thinking blocks, its effort sentence and its Hermes tool calls exactly as a
served request would.

Sampling: the release defaults (T 1.0, top_p 0.97, top_k 128) for prompts marked `sample`, T 0 for
prompts marked `greedy` (kd_prompts.py assigns a quarter of them). The seed is per prompt, so a
re-run of a shard gives the same answers.

Output: shards `gen-XXXXX.pt` of 512 prompts each under --out, each a list of
  {id, src, kind, lang, split, effort, mode, prompt_ids, gen_ids, finish}
written atomically; a shard that exists is skipped, so a job killed by its timeout resumes where it
stopped. A `stats.json` beside them records tokens and wall time per shard (the throughput the
pilot prices the full run with).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train", help="train | heldout | all")
    ap.add_argument("--start", type=int, default=0, help="first prompt (in file order of the split)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--budget-min", type=float, default=0.0, help="stop starting shards after this")
    ap.add_argument("--mirror", default=None, help="copy every finished shard here (the bucket), so a "
                    "trainer can pick it up while this job runs")
    a = ap.parse_args()
    t_start = time.time()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in open(a.prompts)]
    if a.split != "all":
        rows = [r for r in rows if r["split"] == a.split]
    rows = rows[a.start:]
    if a.limit:
        rows = rows[: a.limit]
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              max_num_seqs=a.max_num_seqs, seed=0, enable_prefix_caching=True)
    print(f"[gen] loaded in {time.time() - t_start:.0f} s; {len(rows)} prompts", flush=True)
    stops = [127906, 127901]
    stats_path = os.path.join(a.out, f"stats-{a.split}-{a.start}.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else []
    for s0 in range(0, len(rows), a.shard):
        name = os.path.join(a.out, f"gen-{a.split}-{a.start + s0:06d}.pt")
        if os.path.exists(name) or (a.mirror and os.path.exists(os.path.join(a.mirror, os.path.basename(name)))):
            continue
        if a.budget_min and (time.time() - t_start) / 60 > a.budget_min:
            print(f"[gen] budget reached, stopping before {name}", flush=True)
            break
        chunk = rows[s0: s0 + a.shard]
        prompts, params, keep = [], [], []
        for i, r in enumerate(chunk):
            kw = {"reasoning_effort": r["effort"]}
            if r.get("tools"):
                kw["tools"] = r["tools"]
            try:
                text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, **kw)
            except Exception as e:  # noqa: BLE001  (a malformed tool schema: skip the prompt)
                print(f"[gen] template failed for {r['id']}: {e}", flush=True)
                continue
            ids = tok.encode(text, add_special_tokens=False)
            room = a.max_model_len - len(ids) - 8
            if room < 64:
                continue
            seed = int(r["id"][:8], 16)
            if r["mode"] == "greedy":
                sp = SamplingParams(temperature=0.0, max_tokens=min(r["max_new"], room), stop_token_ids=stops)
            else:
                sp = SamplingParams(temperature=1.0, top_p=0.97, top_k=128, max_tokens=min(r["max_new"], room),
                                    stop_token_ids=stops, seed=seed)
            prompts.append({"prompt_token_ids": ids})
            params.append(sp)
            keep.append(r)
        t0 = time.time()
        outs = llm.generate(prompts, params, use_tqdm=False)
        dt = time.time() - t0
        recs, ntok, nprompt = [], 0, 0
        for r, p, o in zip(keep, prompts, outs):
            g = list(o.outputs[0].token_ids)
            ntok += len(g)
            nprompt += len(p["prompt_token_ids"])
            recs.append({k: r[k] for k in ("id", "src", "kind", "lang", "split", "effort", "mode")} |
                        {"prompt_ids": p["prompt_token_ids"], "gen_ids": g,
                         "finish": o.outputs[0].finish_reason})
        torch.save(recs, name + ".part")
        os.replace(name + ".part", name)
        stats.append({"shard": os.path.basename(name), "prompts": len(recs), "gen_tokens": ntok,
                      "prompt_tokens": nprompt, "seconds": round(dt, 1), "gen_tok_s": round(ntok / dt, 1)})
        json.dump(stats, open(stats_path, "w"), indent=1)
        if a.mirror:
            import shutil
            os.makedirs(a.mirror, exist_ok=True)
            dst = os.path.join(a.mirror, os.path.basename(name))
            shutil.copyfile(name, dst + ".part")
            os.replace(dst + ".part", dst)
            shutil.copyfile(stats_path, os.path.join(a.mirror, os.path.basename(stats_path)))
        print(f"[gen] {os.path.basename(name)}: {len(recs)} prompts, {ntok} gen tokens "
              f"({ntok / max(1, len(recs)):.0f} each), {nprompt} prompt tokens, {dt:.0f} s, "
              f"{ntok / dt:.0f} gen tok/s; total {(time.time() - t_start) / 60:.1f} min", flush=True)
        if s0 == 0:
            r, o = keep[0], outs[0]
            print(f"[gen] sample ({r['src']}, {r['effort']}, {r['mode']}): "
                  f"{tok.decode(o.outputs[0].token_ids)[:600]!r}", flush=True)


if __name__ == "__main__":
    main()
