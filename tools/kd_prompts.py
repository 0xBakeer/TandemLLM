"""Prompts for the Kolibri-1 block drafter: a broad training mix and a held-out set from OTHER sources.

The drafter is trained on Kolibri's own answers to these prompts (self-distillation), so what the
prompts decide is which text the drafter sees: chat, code, German, reasoning, tool calls, with and
without thinking.

THE LEAKAGE TRAP, AND HOW THIS FILE AVOIDS IT
  Templated prompts with no variable part put byte copies of the same text on both sides of a
  train/held-out split, and the held-out score then measures memory. Here:
  * every prompt is a real dataset row, never a template;
  * the held-out set comes from sources the training mix never reads (SPEED-Bench, MT-Bench,
    GermanRAG, the oasst2 VALIDATION trees), not from a split of the same files;
  * every prompt is keyed by the sha256 of its normalised rendered messages, duplicates are dropped
    across both sides, and a held-out prompt whose first 200 normalised characters match a training
    prompt is dropped from the held-out side.

Sources are read from the Hub's parquet conversion (shard 0 of each split; downloaded once into
--cache). Output: one JSON line per prompt with id, src, kind, lang, split, messages, tools,
effort, mode (sample | greedy) and max_new, shuffled with a fixed seed so any prefix of the training
lines is itself a balanced mix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import urllib.request

PARQUET = "https://huggingface.co/api/datasets/{ds}/parquet/{cfg}/{split}/0.parquet"

# (name, dataset, config, split, kind, lang, count) -- training pool
TRAIN = [
    ("ultrachat", "HuggingFaceH4/ultrachat_200k", "default", "train_sft", "chat", "en", 7000),
    ("oasst2-en", "OpenAssistant/oasst2", "default", "train", "chat", "en", 2500),
    ("oasst2-de", "OpenAssistant/oasst2", "default", "train", "chat", "de", 4000),
    ("alpaca-de", "FreedomIntelligence/alpaca-gpt4-deutsch", "default", "train", "chat", "de", 6000),
    ("dolly-de", "argilla/databricks-dolly-15k-curated-multilingual", "default", "de", "chat", "de", 3500),
    ("oss-instruct", "bigcode/self-oss-instruct-sc2-exec-filter-50k", "default", "train", "code", "en", 4000),
    ("codefeedback", "m-a-p/CodeFeedback-Filtered-Instruction", "default", "train", "code", "en", 4000),
    ("gsm8k", "openai/gsm8k", "main", "train", "math", "en", 2500),
    ("numina", "AI-MO/NuminaMath-CoT", "default", "train", "math", "en", 3000),
    ("glaive", "glaiveai/glaive-function-calling-v2", "default", "train", "tool", "en", 3000),
    ("hermes-fc", "NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train", "tool", "en", 1500),
]
# held-out: different sources
HELD = [
    ("speed", "nvidia/SPEED-Bench", "qualitative", "test", "mixed", "mixed", 10_000),
    ("mtbench", "HuggingFaceH4/mt_bench_prompts", "default", "train", "chat", "en", 10_000),
    ("germanrag", "DiscoResearch/germanrag", "default", "train", "rag", "de", 150),
    ("oasst2-val-de", "OpenAssistant/oasst2", "default", "validation", "chat", "de", 150),
]

EFFORTS = [("none", 0.45), ("low", 0.20), ("medium", 0.15), ("high", 0.20)]
MAX_NEW = {"none": 1024, "low": 1536, "medium": 2048, "high": 2048}
GERMAN_NOTE = "Antworte auf Deutsch."


def fetch(cache: str, ds: str, cfg: str, split: str) -> str:
    os.makedirs(cache, exist_ok=True)
    out = os.path.join(cache, f"{ds}-{cfg}-{split}".replace("/", "_") + ".parquet")
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        req = urllib.request.Request(PARQUET.format(ds=ds, cfg=cfg, split=split))
        tok = os.environ.get("HF_TOKEN")
        if tok:
            req.add_header("Authorization", f"Bearer {tok}")
        with urllib.request.urlopen(req) as r, open(out + ".part", "wb") as f:
            f.write(r.read())
        os.replace(out + ".part", out)
    return out


def rows(path: str) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def clip(s: str, n: int = 6000) -> str:
    return s if len(s) <= n else s[:n]


# ----------------------------------------------------------------------------- per source

def parse_glaive_tools(system: str) -> list[dict] | None:
    """The function definitions in a glaive SYSTEM text, as OpenAI-shaped tools; None if none."""
    i = system.find("{")
    if i < 0:
        return None
    body = system[i:]
    dec = json.JSONDecoder()
    tools, j = [], 0
    while j < len(body):
        while j < len(body) and body[j] in " \n\r\t,-":
            j += 1
        if j >= len(body) or body[j] != "{":
            break
        try:
            obj, j = dec.raw_decode(body, j)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict) and "name" in obj:
            tools.append({"type": "function", "function": obj})
    return tools or None


def extract(name: str, r: dict, all_rows=None) -> list[tuple[list[dict], list | None]]:
    """[(messages, tools)] from one row; [] to skip it."""
    u = lambda t: [{"role": "user", "content": clip(t)}]  # noqa: E731
    if name == "ultrachat":
        return [(u(r["prompt"]), None)] if r.get("prompt") else []
    if name.startswith("oasst2"):
        return [(u(r["text"]), None)] if r.get("parent_id") is None and r.get("role") == "prompter" else []
    if name == "alpaca-de":
        c = r.get("conversations") or []
        return [(u(c[0]["value"]), None)] if c and c[0].get("from") == "human" else []
    if name == "dolly-de":
        t = r["instruction"] + (("\n\n" + r["context"]) if r.get("context") else "")
        return [(u(t), None)]
    if name == "oss-instruct":
        return [(u(r["instruction"]), None)]
    if name == "codefeedback":
        return [(u(r["query"]), None)]
    if name == "gsm8k":
        return [(u(r["question"]), None)]
    if name == "numina":
        return [(u(r["problem"]), None)]
    if name == "glaive":
        tools = parse_glaive_tools(r.get("system") or "")
        m = re.search(r"USER:\s*(.*?)(?:\n\s*\n\s*(?:ASSISTANT|FUNCTION RESPONSE):|$)", r.get("chat") or "", re.S)
        if not m or not m.group(1).strip():
            return []
        return [(u(m.group(1).strip()), tools)]
    if name == "hermes-fc":
        conv = r.get("conversations") or []
        user = next((c["value"] for c in conv if c.get("from") == "human"), None)
        try:
            tools = json.loads(r["tools"]) if isinstance(r.get("tools"), str) else r.get("tools")
        except json.JSONDecodeError:
            tools = None
        return [(u(user), tools or None)] if user else []
    if name == "speed":
        turns = r.get("turns")
        if isinstance(turns, str):
            try:
                import ast
                turns = ast.literal_eval(turns)
            except (ValueError, SyntaxError):
                turns = [turns]
        t = (turns or [""])[0]
        if not t or "SHOULD BE FETCHED FROM THE SOURCE" in t:
            return []
        return [(u(t), None)]
    if name == "mtbench":
        p = r.get("prompt") or []
        return [(u(p[0]), None)] if p else []
    if name == "germanrag":
        ctx = "\n\n".join(r.get("contexts") or [])
        return [([{"role": "system", "content": clip("Beantworte die Frage anhand der folgenden Texte.\n\n" + ctx, 8000)},
                  {"role": "user", "content": r["question"]}], None)]
    raise KeyError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every training count")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    seen: dict[str, str] = {}
    heads: set[str] = set()
    out: list[dict] = []
    stats: dict[str, int] = {}

    def add(name, kind, lang, split, msgs, tools):
        key = json.dumps([msgs, tools], sort_keys=True, ensure_ascii=False)
        h = hashlib.sha256(norm(key).encode()).hexdigest()[:20]
        head = norm(msgs[-1]["content"])[:200]
        if h in seen:
            return False
        if split == "heldout" and head in heads:
            return False
        seen[h] = split
        if split == "train":
            heads.add(head)
        k = kind if kind != "mixed" else "mixed"
        effort = rng.choices([e for e, _ in EFFORTS], [w for _, w in EFFORTS])[0]
        if kind in ("math",) and effort == "none" and rng.random() < 0.5:
            effort = "low"
        system_de = (kind in ("code", "math", "tool") and split == "train" and rng.random() < 0.15)
        if system_de:
            lang = "de"
            if msgs[0]["role"] == "system":
                msgs = [{"role": "system", "content": msgs[0]["content"] + "\n\n" + GERMAN_NOTE}] + msgs[1:]
            else:
                msgs = [{"role": "system", "content": GERMAN_NOTE}] + msgs
        mode = "greedy" if rng.random() < 0.25 else "sample"
        out.append({"id": h, "src": name, "kind": k, "lang": lang, "split": split, "messages": msgs,
                    "tools": tools, "effort": effort, "mode": mode, "max_new": MAX_NEW[effort]})
        stats[f"{split}/{name}"] = stats.get(f"{split}/{name}", 0) + 1
        return True

    # training first, so the held-out filter knows every training head
    for name, ds, cfg, split, kind, lang, n in TRAIN:
        rs = rows(fetch(a.cache, ds, cfg, split))
        if name.startswith("oasst2"):
            rs = [r for r in rs if r.get("lang") == lang]
        rng.shuffle(rs)
        want, got = int(n * a.scale), 0
        for r in rs:
            for msgs, tools in extract(name, r):
                if got < want and add(name, kind, lang, "train", msgs, tools):
                    got += 1
            if got >= want:
                break
        print(f"train {name:14s} {got:6d} of {want}")
    for name, ds, cfg, split, kind, lang, n in HELD:
        rs = rows(fetch(a.cache, ds, cfg, split))
        if name.startswith("oasst2"):
            rs = [r for r in rs if r.get("lang") == "de"]
        rng.shuffle(rs)
        got = 0
        for r in rs:
            for msgs, tools in extract(name, r):
                if got < n and add(name, kind if name != "speed" else r.get("category", "mixed"),
                                   lang, "heldout", msgs, tools):
                    got += 1
            if got >= n:
                break
        print(f"held  {name:14s} {got:6d}")
    train = [x for x in out if x["split"] == "train"]
    held = [x for x in out if x["split"] == "heldout"]
    rng.shuffle(train)
    rng.shuffle(held)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for x in held + train:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    de = sum(1 for x in train if x["lang"] == "de")
    print(f"train {len(train)} (German {de / max(1, len(train)):.1%}), held-out {len(held)}"
          f" (German {sum(1 for x in held if x['lang'] == 'de')}); greedy share "
          f"{sum(1 for x in out if x['mode'] == 'greedy') / len(out):.1%}")


if __name__ == "__main__":
    main()
