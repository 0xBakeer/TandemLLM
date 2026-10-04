"""Kolibri-1 through Aleph Alpha's own vLLM plugin: the reference our plain-torch forward is checked
against, before any Hessian is built from that forward.

Runs inside `vllm/vllm-openai:v0.29.0` with `pip install --no-deps aleph-alpha-inference==1.0.0`,
on the FP8 release. It writes two files:

  * `seqs.json`: every token sequence it scored, for `kolibri_quant.py ref` to score again;
  * `vllm.pt`: per sequence, the top-k next-token ids at every position (prompt log-probs), and
    for prompts the top 5 of the token after the last one.

The sequences: the three held-out gate texts (2,048 tokens each, the same ids the build's gate
uses), the two raw capital prompts of the first CPU run of `tools/kolibri_ref.py`, the capital questions through Kolibri's chat
template (thinking off and on), and 30 prompts in English, German and code, thinking on and off,
whose greedy answers are appended and scored too. The chat prompts are rendered with the
tokenizer's own template (transformers' `apply_chat_template`), which also checks the hand
renderer in `tools/kolibri_corpus.py` on the same messages.

    python tools/kolibri_vllm_check.py --model /models/fp8 --corpus DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

PROMPTS = [
    # (text, language, thinking)
    ("Explain in two sentences what a hash table is.", "en", False),
    ("What causes the seasons on Earth?", "en", False),
    ("Give me three tips for writing a clear email.", "en", False),
    ("Summarise the plot of Romeo and Juliet in three sentences.", "en", False),
    ("What is the difference between weather and climate?", "en", False),
    ("Name the planets of the solar system in order.", "en", False),
    ("How does a refrigerator keep food cold?", "en", True),
    ("Is 391 a prime number?", "en", True),
    ("Why do we have leap years?", "en", False),
    ("What is the boiling point of water at sea level in Fahrenheit?", "en", False),
    ("Erkläre in zwei Sätzen, was eine Hashtabelle ist.", "de", False),
    ("Warum gibt es Jahreszeiten auf der Erde?", "de", False),
    ("Gib mir drei Tipps für eine klare E-Mail.", "de", False),
    ("Fasse die Handlung von Faust in drei Sätzen zusammen.", "de", False),
    ("Was ist der Unterschied zwischen Wetter und Klima?", "de", False),
    ("Nenne die Bundesländer Deutschlands.", "de", False),
    ("Wie funktioniert ein Kühlschrank?", "de", True),
    ("Ist 391 eine Primzahl?", "de", True),
    ("Warum gibt es Schaltjahre?", "de", False),
    ("Schreibe ein kurzes Gedicht über den Herbst.", "de", False),
    ("Write a Python function that checks whether a string is a palindrome.", "code", False),
    ("Write a SQL query that returns the ten most recent orders per customer.", "code", False),
    ("Explain what this does: `sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:3]`", "code", False),
    ("Schreibe eine Bash-Zeile, die alle .log-Dateien älter als 7 Tage löscht.", "code", False),
    ("Write a JavaScript function that debounces another function.", "code", True),
    ("What is 17 * 23?", "en", True),
    ("Wie viele Sekunden hat ein Tag?", "de", True),
    ("Translate into German: The meeting has been moved to Thursday afternoon.", "en", False),
    ("Übersetze ins Englische: Der Zug hat leider zwanzig Minuten Verspätung.", "de", False),
    ("Write a haiku about the sea.", "en", False),
]

CAPITALS = [
    ("chat-cap-de-off", "Was ist die Hauptstadt von Deutschland?", False),
    ("chat-cap-de-on", "Was ist die Hauptstadt von Deutschland?", True),
    ("chat-cap-en-off", "What is the capital of Germany?", False),
    ("chat-cap-fr-de-off", "Was ist die Hauptstadt von Frankreich?", False),
]
RAW = [("raw-de", "Die Hauptstadt von Deutschland ist"), ("raw-en", "The capital of France is")]


def top_table(plp, ids, k):
    """vLLM prompt log-probs -> [T, k] top ids for the token after each position (-1 = none)."""
    T = len(ids)
    out = torch.full((T, k), -1, dtype=torch.long)
    lps = torch.full((T, k), float("nan"))
    for t in range(1, T):
        d = plp[t]
        if not d:
            continue
        ranked = sorted(((v.rank, tid, v.logprob) for tid, v in d.items() if v.rank is not None and v.rank <= k))
        for rnk, tid, lp in ranked:
            out[t - 1, rnk - 1] = tid
            lps[t - 1, rnk - 1] = lp
    return out, lps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.94)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.kolibri_corpus import render

    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              enforce_eager=True, max_num_seqs=16, seed=0)
    print(f"[vllm] loaded in {time.time() - t0:.0f} s", flush=True)

    def chat(text, thinking):
        msgs = [{"role": "user", "content": text}]
        kw = {} if thinking else {"enable_thinking": False}
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        mine = render(msgs + [{"role": "assistant", "content": ""}], "high" if thinking else "none")
        mine = mine[: mine.rfind("<|im_start|>assistant\n")] + "<|im_start|>assistant\n"
        if not thinking:
            mine += "<think>\n\n</think>\n\n"
        return s, s == mine

    seqs, template_same = [], []
    for n in ("prose", "code", "de"):
        if not os.path.isfile(os.path.join(a.corpus, f"heldout-{n}.npy")):
            continue
        ids = np.load(os.path.join(a.corpus, f"heldout-{n}.npy")).astype(np.int64)[: a.gate_tokens].tolist()
        seqs.append({"name": f"heldout-{n}", "ids": ids})
    for name, s in RAW:
        seqs.append({"name": name, "ids": tok.encode(s, add_special_tokens=False), "text": s})
    gen_prompts = [(f"cap:{n}", t, th) for n, t, th in CAPITALS] + \
                  [(f"p{i:02d}-{lang}-{'on' if th else 'off'}", t, th) for i, (t, lang, th) in enumerate(PROMPTS)]
    rendered = []
    for name, text, th in gen_prompts:
        s, same = chat(text, th)
        template_same.append(same)
        rendered.append((name, s))
    print(f"[template] hand renderer equals apply_chat_template on {sum(template_same)}/{len(template_same)}",
          flush=True)
    sp_gen = SamplingParams(temperature=0.0, max_tokens=a.max_new, logprobs=a.k)
    outs = llm.generate([{"prompt_token_ids": tok.encode(s, add_special_tokens=False)} for _, s in rendered], sp_gen)
    answers = {}
    for (name, s), o in zip(rendered, outs):
        p = list(o.prompt_token_ids)
        g = list(o.outputs[0].token_ids)
        first = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        top5 = sorted(((v.rank, tid, v.logprob, v.decoded_token) for tid, v in first.items()))[: a.k]
        answers[name] = {"prompt": s, "answer": o.outputs[0].text, "tokens": len(g),
                         "finish": o.outputs[0].finish_reason, "first_top5": top5}
        seqs.append({"name": f"gen-{name}", "ids": p + g, "gen_start": len(p)})
    with open(os.path.join(a.out, "answers.json"), "w") as f:
        json.dump(answers, f, indent=1, ensure_ascii=False)
    for name in list(answers)[:6]:
        print(f"[gen] {name}: {answers[name]['answer'][:400]!r}", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=a.k, logprobs=a.k)
    t1 = time.time()
    res = llm.generate([{"prompt_token_ids": s["ids"]} for s in seqs], sp)
    out = []
    for s, o in zip(seqs, res):
        top, lps = top_table(o.prompt_logprobs, s["ids"], a.k)
        nxt = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        n5 = sorted(((v.rank, tid, v.logprob, v.decoded_token) for tid, v in nxt.items()))[: a.k]
        top[len(s["ids"]) - 1] = torch.tensor([t for _, t, _, _ in n5] + [-1] * (a.k - len(n5)))
        out.append({"name": s["name"], "ids": s["ids"], "top_ids": top, "top_lp": lps,
                    "gen_start": s.get("gen_start"), "next_top5": [(t, round(lp, 4), d) for _, t, lp, d in n5]})
        if s["name"].startswith(("raw", "gen-cap")):
            print(f"[next] {s['name']}: {out[-1]['next_top5']}", flush=True)
    print(f"[vllm] scored {len(seqs)} sequences in {time.time() - t1:.0f} s", flush=True)
    torch.save(out, os.path.join(a.out, "vllm.pt"))
    with open(os.path.join(a.out, "seqs.json"), "w") as f:
        json.dump([{"name": s["name"], "ids": s["ids"]} for s in seqs], f)
    with open(os.path.join(a.out, "template.json"), "w") as f:
        json.dump({"hand_renderer_equal": template_same, "names": [n for n, _, _ in gen_prompts]}, f)


if __name__ == "__main__":
    main()
