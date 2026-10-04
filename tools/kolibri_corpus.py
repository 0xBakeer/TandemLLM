"""The calibration and evaluation texts for Kolibri-1's NVFP4 build, cut in Kolibri's own tokens.

`tools/calib_corpus.py` was written for Qwen3.8: its budgets are characters, its German share is
60k of 560k characters, and it has no chat. Kolibri-1 has its own tokenizer (UniBPE, 128,000 ids,
`tokenizer.json`, read with the `tokenizers` package), is sold on German, and is a post-trained
reasoning model that reads raw text as its own thinking. So the split here is, by tokens:

  * code 25 %: `bench/calib.txt` and modules of the interpreter's standard library;
  * English 30 % and German 30 %: Wikipedia leads (`wikimedia/wikipedia`, shard 0 of each);
  * chat 15 %: conversations rendered through Kolibri's ChatML template by hand (the template's
    system block with the effort sentence, `<think>` blocks, Hermes tool calls): OpenAssistant
    (English and German), the German Dolly translation, UltraChat, a few reasoning traces from
    s1K-1.1 (thinking on), and synthetic tool calls.

Every document passes the guard of `calib_corpus.py` (no 12-word run shared with
`bench/heldout_{prose,code}.txt`, and `bench/heldout_de.txt` when you add a German held-out text;
the repository does not ship one). After the calibration splits, `eval-{code,en,de}` take the
next documents in the same order for the wider gate table. Writes `<split>.txt`, `<split>.npy`
(int32 token ids, documents joined by "\n\n") and `manifest.json`. CPU only, no torch.

    python tools/kolibri_corpus.py --tokenizer DIR/tokenizer.json --out /work/<prefix>/corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.calib_corpus import HELDOUT, Guard, code_docs, wiki_docs  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EFFORT = {
    "none": "Reasoning is disabled. Proceed straight to answering according to the user's instructions.",
    "low": "Reasoning effort is set to low. Think briefly through only the essential steps in the user's language, then proceed directly to the answer.",
    "medium": "Reasoning effort is set to medium. Think through the task methodically in the user's language, check key assumptions, and provide a well-supported answer.",
    "high": "Reasoning effort is set to high. Think carefully through the task in the user's language, validate key assumptions, consider plausible alternatives, and prioritize correctness and clarity.",
}

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_train", "description": "Find train connections between two stations.",
     "parameters": {"type": "object", "properties": {"from": {"type": "string"}, "to": {"type": "string"}, "date": {"type": "string"}}, "required": ["from", "to"]}}},
    {"type": "function", "function": {"name": "run_sql", "description": "Run a read-only SQL query.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


def render(messages: list[dict], effort: str = "high", tools: list | None = None) -> str:
    """Kolibri-1's chat template (tokenizer_config.json of both releases), for a finished
    conversation: system block with the effort sentence, user turns, assistant turns with a
    `<think>` block after the last user turn only, Hermes tool calls, tool responses as user turns."""
    out = []
    sys_parts = []
    if messages and messages[0]["role"] == "system":
        sys_parts.append(messages[0]["content"])
        messages = messages[1:]
    sys_parts.append("# Reasoning effort\n\n" + EFFORT[effort])
    if tools:
        sys_parts.append(
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided "
            "with function signatures within <tools></tools> XML tags:\n<tools>"
            + "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools)
            + "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
            "<args-json-object>}\n</tool_call>")
    out.append("<|im_start|>system\n" + "\n\n".join(sys_parts) + "<|im_end|>\n")
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"
                     and not m["content"].startswith("<tool_response>")), default=-1)
    i = 0
    while i < len(messages):
        m = messages[i]
        if m["role"] == "user":
            out.append(f"<|im_start|>user\n{m['content']}<|im_end|>\n")
        elif m["role"] == "assistant":
            s = "<|im_start|>assistant\n"
            if i > last_user:
                r = (m.get("reasoning") or "").strip("\n")
                s += "<think>\n" + r + "\n</think>\n\n"
            content = (m.get("content") or "").lstrip("\n")
            s += content
            for j, tc in enumerate(m.get("tool_calls") or []):
                if (j == 0 and content) or j > 0:
                    s += "\n"
                s += ('<tool_call>\n{"name": "' + tc["name"] + '", "arguments": '
                      + json.dumps(tc["arguments"], ensure_ascii=False) + "}\n</tool_call>")
            out.append(s + "<|im_end|>\n")
        elif m["role"] == "tool":
            s = "<|im_start|>user" if i == 0 or messages[i - 1]["role"] != "tool" else ""
            s += "\n<tool_response>\n" + m["content"] + "\n</tool_response>"
            if i == len(messages) - 1 or messages[i + 1]["role"] != "tool":
                s += "<|im_end|>\n"
            out.append(s)
        i += 1
    return "".join(out)


def _dl(repo: str, fname: str, cache: str | None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fname, repo_type="dataset", cache_dir=cache)


def oasst(cache, langs=("de", "en")):
    """OpenAssistant threads, best-ranked reply at each step, up to three exchanges."""
    import pyarrow.parquet as pq
    t = pq.read_table(_dl("OpenAssistant/oasst1", "data/train-00000-of-00001-b42a775f407cee45.parquet", cache),
                      columns=["message_id", "parent_id", "text", "role", "lang", "rank", "deleted"]).to_pydict()
    kids: dict[str, list[int]] = {}
    roots = []
    for i, (pid, dele) in enumerate(zip(t["parent_id"], t["deleted"])):
        if dele:
            continue
        if pid is None:
            roots.append(i)
        else:
            kids.setdefault(pid, []).append(i)
    for r in roots:
        if t["lang"][r] not in langs:
            continue
        conv, cur = [], r
        while cur is not None and len(conv) < 6:
            conv.append({"role": "user" if t["role"][cur] == "prompter" else "assistant", "content": t["text"][cur]})
            ch = kids.get(t["message_id"][cur], [])
            if not ch:
                break
            cur = min(ch, key=lambda k: (t["rank"][k] if t["rank"][k] is not None else 99, k))
        if conv[-1]["role"] == "user":
            conv = conv[:-1]
        if len(conv) >= 2:
            yield f"oasst:{t['message_id'][r]}:{t['lang'][r]}", conv


def dolly_de(cache):
    import pyarrow.parquet as pq
    t = pq.read_table(_dl("argilla/databricks-dolly-15k-curated-multilingual",
                          "data/de-00000-of-00001-b043739d21afba4d.parquet", cache),
                      columns=["id", "instruction", "context", "response"]).to_pydict()
    for i, ins, ctx, resp in zip(t["id"], t["instruction"], t["context"], t["response"]):
        u = ins if not ctx else f"{ins}\n\n{ctx}"
        yield f"dolly-de:{i}", [{"role": "user", "content": u}, {"role": "assistant", "content": resp}]


def ultrachat(cache):
    import pyarrow.parquet as pq
    t = pq.read_table(_dl("HuggingFaceH4/ultrachat_200k", "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
                          cache), columns=["prompt_id", "messages"]).to_pydict()
    for pid, msgs in zip(t["prompt_id"], t["messages"]):
        conv = [{"role": m["role"], "content": m["content"]} for m in msgs[:6] if m["role"] in ("user", "assistant")]
        if len(conv) >= 2:
            yield f"ultrachat:{pid}", conv


def s1k(cache):
    import pyarrow.parquet as pq
    t = pq.read_table(_dl("simplescaling/s1K-1.1", "data/train-00000-of-00001.parquet", cache),
                      columns=["question", "deepseek_thinking_trajectory", "deepseek_attempt"]).to_pydict()
    for i, (q, th, a) in enumerate(zip(t["question"], t["deepseek_thinking_trajectory"], t["deepseek_attempt"])):
        # the first ~6k characters of a trace: the opening and the working, not its 30k-character tail
        th = th[:6000]
        th = th[: th.rfind("\n") if th.rfind("\n") > 3000 else len(th)]
        yield f"s1k:{i}", [{"role": "user", "content": q}, {"role": "assistant", "reasoning": th, "content": a}]


def toolcalls(n: int, seed: int = 11):
    """Short synthetic Hermes tool-call exchanges, German and English, with the tool's answer."""
    rnd = random.Random(seed)
    cities = ["Berlin", "München", "Hamburg", "Köln", "Leipzig", "Freiburg", "Zürich", "Wien", "Lyon", "Boston"]
    for i in range(n):
        kind = i % 3
        de = rnd.random() < 0.5
        if kind == 0:
            c = rnd.choice(cities)
            q = f"Wie ist das Wetter gerade in {c}?" if de else f"What's the weather like in {c} right now?"
            call = {"name": "get_weather", "arguments": {"city": c, "unit": "celsius"}}
            temp = rnd.randint(-5, 31)
            resp = json.dumps({"city": c, "temp_c": temp, "condition": rnd.choice(["sunny", "rain", "cloudy", "snow"])})
            ans = (f"In {c} sind es gerade {temp} Grad." if de else f"It is {temp} °C in {c} right now.")
        elif kind == 1:
            a, b = rnd.sample(cities[:8], 2)
            q = (f"Such mir bitte eine Zugverbindung von {a} nach {b} für morgen früh." if de
                 else f"Find me a train from {a} to {b} tomorrow morning.")
            call = {"name": "search_train", "arguments": {"from": a, "to": b, "date": "2026-10-05"}}
            h = rnd.randint(5, 9)
            resp = json.dumps({"connections": [{"dep": f"0{h}:12", "arr": f"{h + 3:02d}:47", "changes": rnd.randint(0, 2)}]})
            ans = (f"Es gibt eine Verbindung um 0{h}:12 Uhr, Ankunft {h + 3:02d}:47 Uhr." if de
                   else f"There is a train at 0{h}:12, arriving at {h + 3:02d}:47.")
        else:
            y = rnd.randint(2019, 2025)
            q = (f"Wie viele Bestellungen hatten wir {y}?" if de else f"How many orders did we have in {y}?")
            call = {"name": "run_sql", "arguments": {"query": f"SELECT COUNT(*) FROM orders WHERE year = {y};"}}
            cnt = rnd.randint(1000, 90000)
            resp = json.dumps({"rows": [[cnt]]})
            ans = (f"{y} gab es {cnt} Bestellungen." if de else f"There were {cnt} orders in {y}.")
        msgs = [{"role": "user", "content": q},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "content": resp},
                {"role": "assistant", "content": ans}]
        yield f"tool:{i}", msgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="Kolibri-1's tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=640_000, help="calibration tokens over all four splits")
    ap.add_argument("--mix", default="code:0.25,en:0.30,de:0.30,chat:0.15")
    ap.add_argument("--eval-tokens", type=int, default=12_000, help="each of eval-{code,en,de}")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--ban", action="append", default=["pagoda garden"])
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)

    def n_tok(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False).ids)

    os.makedirs(args.out, exist_ok=True)
    guard = Guard(HELDOUT, args.ban)
    for p in HELDOUT:
        assert os.path.isfile(p), p
    mix = {k: float(v) for k, v in (x.split(":") for x in args.mix.split(","))}
    budget = {k: int(args.tokens * v) for k, v in mix.items()}
    manifest: dict = {"python": sys.version.split()[0], "tokenizer_sha256":
                      hashlib.sha256(open(args.tokenizer, "rb").read()).hexdigest(),
                      "mix": mix, "shingle_guard": [os.path.basename(p) for p in HELDOUT], "files": {}}

    def take(docs, tokens: int, min_chars: int = 400):
        texts, ids, n = [], [], 0
        for doc_id, text in docs:
            if n >= tokens:
                break
            if len(text) < min_chars or not guard.ok(text):
                continue
            texts.append(text)
            ids.append(doc_id)
            n += n_tok(text) + 1
        return texts, ids

    def write(name: str, texts: list[str], ids: list[str]) -> None:
        body = "\n\n".join(texts)
        with open(os.path.join(args.out, name + ".txt"), "w") as f:
            f.write(body)
        arr = np.asarray(tok.encode(body, add_special_tokens=False).ids, dtype=np.int32)
        np.save(os.path.join(args.out, name + ".npy"), arr)
        manifest["files"][name] = {"docs": len(ids), "chars": len(body), "tokens": int(arr.size),
                                   "sha256": hashlib.sha256(body.encode()).hexdigest(), "ids": ids}
        print(f"  {name:12s} {len(ids):5d} docs {len(body):9d} chars {arr.size:8d} tokens", flush=True)

    code = code_docs(["triton"])
    base = open(os.path.join(HERE, "bench", "calib.txt")).read()
    assert guard.ok(base)
    t, i = take(code, budget["code"] - n_tok(base))
    write("calib-code", [base] + t, ["bench/calib.txt"] + i)
    t, i = take(code, args.eval_tokens)
    write("eval-code", t, i)
    for lang in ("en", "de"):
        docs = wiki_docs(lang, 0, args.cache)
        t, i = take(docs, budget[lang])
        write(f"calib-{lang}", t, i)
        t, i = take(docs, args.eval_tokens)
        write(f"eval-{lang}", t, i)

    # chat: a fixed interleave of the sources, thinking on for half (effort high/medium/low and a
    # <think> block that is empty unless the source has a trace), off for the other half
    rnd = random.Random(5)
    srcs = {"oasst": oasst(args.cache), "dolly": dolly_de(args.cache), "ultra": ultrachat(args.cache),
            "s1k": s1k(args.cache), "tool": toolcalls(400)}
    share = {"oasst": 0.40, "dolly": 0.20, "ultra": 0.20, "s1k": 0.12, "tool": 0.08}
    chat_t, chat_i = [], []
    for name, frac in share.items():
        n = 0
        for doc_id, conv in srcs[name]:
            if n >= budget["chat"] * frac:
                break
            effort = "high" if name == "s1k" else rnd.choice(["none", "none", "low", "medium", "high", "none"])
            text = render(conv, effort, TOOLS if name == "tool" else None)
            if not guard.ok(text):
                continue
            chat_t.append(text)
            chat_i.append(f"{doc_id}:{effort}")
            n += n_tok(text)
    order = list(range(len(chat_t)))
    rnd.shuffle(order)
    # conversations are joined without a separator: each already ends on "<|im_end|>\n"
    body = "".join(chat_t[k] for k in order)
    with open(os.path.join(args.out, "calib-chat.txt"), "w") as f:
        f.write(body)
    arr = np.asarray(tok.encode(body, add_special_tokens=False).ids, dtype=np.int32)
    np.save(os.path.join(args.out, "calib-chat.npy"), arr)
    manifest["files"]["calib-chat"] = {"docs": len(order), "chars": len(body), "tokens": int(arr.size),
                                       "sha256": hashlib.sha256(body.encode()).hexdigest(),
                                       "ids": [chat_i[k] for k in order]}
    print(f"  {'calib-chat':12s} {len(order):5d} docs {len(body):9d} chars {arr.size:8d} tokens", flush=True)

    for p in HELDOUT:
        name = "heldout-" + os.path.basename(p)[len("heldout_"):-4]
        arr = np.asarray(tok.encode(open(p).read(), add_special_tokens=False).ids, dtype=np.int32)
        np.save(os.path.join(args.out, name + ".npy"), arr)
        manifest["files"][name] = {"tokens": int(arr.size), "source": os.path.relpath(p, HERE)}
        print(f"  {name:12s} {arr.size:8d} tokens", flush=True)
    manifest["dropped_by_guard"] = guard.dropped
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    tot = sum(v["tokens"] for k, v in manifest["files"].items() if k.startswith("calib-"))
    print(f"[corpus] calibration {tot} tokens; {guard.dropped} documents dropped by the guard -> {args.out}")


if __name__ == "__main__":
    main()
