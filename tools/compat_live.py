"""SRV-17 on a served engine, through the official `openai` client: each field works or is refused.

    ssh -N -L 8011:127.0.0.1:8011 dgx &
    python tools/compat_live.py --base http://127.0.0.1:8011/v1 --tokenizer <model dir> \\
        --json results/api/compat-live-<label>.json

The tokenizer is only needed for `logit_bias`, which takes token ids. Every check prints PASS or
FAIL with what it saw; the exit code is the number of failures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

OFF = {"chat_template_kwargs": {"enable_thinking": False}}
ASK = "Name three rivers in Europe, one line each, nothing else."
CAPITAL = "What is the capital of France? Answer with one word."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    import openai
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    client = openai.OpenAI(base_url=a.base, api_key="none", timeout=600, max_retries=0)
    model = client.models.list().data[0].id
    results = []

    def check(name, ok, **seen):
        results.append({"check": name, "pass": bool(ok), **seen})
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {json.dumps(seen, ensure_ascii=False)[:300]}",
              flush=True)

    def chat(content=ASK, **kw):
        extra = dict(OFF)
        extra.update(kw.pop("extra_body", {}))
        for k in ("min_p", "top_k"):             # not OpenAI fields: the client refuses them
            if k in kw:
                extra[k] = kw.pop(k)
        return client.chat.completions.create(model=model, max_tokens=kw.pop("max_tokens", 48),
                                              messages=[{"role": "user", "content": content}],
                                              extra_body=extra, **kw)

    def refused(param, **kw):
        try:
            chat(**kw)
        except openai.BadRequestError as exc:
            got = (exc.body or {}).get("param") if isinstance(exc.body, dict) else None
            return got == param, got
        return False, "200"

    # --- logprobs, both transports and both endpoints
    r = chat(temperature=0, logprobs=True, top_logprobs=3)
    c = r.choices[0]
    ent = c.logprobs.content
    want_n = r.usage.completion_tokens - (1 if c.finish_reason == "stop" else 0)
    rebuilt = bytes(b for e in ent for b in e.bytes).decode("utf-8", errors="replace")
    check("logprobs json: one entry a token, the bytes spell the answer",
          len(ent) == want_n and rebuilt == c.message.content and
          all(len(e.top_logprobs) == 3 and e.logprob <= 0 for e in ent),
          entries=len(ent), tokens=want_n)
    check("logprobs greedy: the chosen token carries the top log-probability",
          all(abs(e.logprob - e.top_logprobs[0].logprob) < 1e-6 for e in ent),
          worst=max(e.top_logprobs[0].logprob - e.logprob for e in ent))
    streamed = []
    text = ""
    for ch in chat(temperature=0, logprobs=True, top_logprobs=3, stream=True):
        for cc in ch.choices:
            text += cc.delta.content or ""
            if cc.logprobs and cc.logprobs.content:
                streamed += cc.logprobs.content
    check("logprobs stream: the same tokens as the JSON answer, the same values to 1e-2",
          text == c.message.content and [e.token for e in streamed] == [e.token for e in ent]
          and all(abs(x.logprob - y.logprob) < 1e-2 for x, y in zip(streamed, ent)),
          entries=len(streamed))
    cr = client.completions.create(model=model, prompt="The three largest rivers in Europe are",
                                   max_tokens=24, temperature=0, logprobs=2)
    lp = cr.choices[0].logprobs
    check("completions logprobs: the legacy shape, offsets into the text",
          "".join(lp.tokens) == cr.choices[0].text and len(lp.token_logprobs) == len(lp.tokens)
          and lp.text_offset == [sum(len(t) for t in lp.tokens[:i]) for i in range(len(lp.tokens))],
          tokens=len(lp.tokens))

    # --- n
    r = chat(temperature=0, n=3)
    texts = [x.message.content for x in r.choices]
    one = chat(temperature=0)
    check("n=3 greedy: three identical choices, usage counts all three",
          len(set(texts)) == 1 and [x.index for x in r.choices] == [0, 1, 2]
          and r.usage.completion_tokens == 3 * one.usage.completion_tokens,
          usage=r.usage.completion_tokens, one=one.usage.completion_tokens)
    s = dict(temperature=0.8, seed=1234, n=3)
    ra, rb = chat(**s), chat(**s)
    ta = [x.message.content for x in ra.choices]
    first = chat(temperature=0.8, seed=1234)
    check("n=3 seeded: three different choices, reproduced, choice 0 is the n=1 answer",
          len(set(ta)) == 3 and ta == [x.message.content for x in rb.choices]
          and ta[0] == first.choices[0].message.content, choices=ta)

    # --- min_p
    g = chat(temperature=0).choices[0].message.content
    m1 = chat(temperature=1.0, min_p=1.0, seed=3).choices[0].message.content
    check("min_p=1.0 at temperature 1: only the top token survives, so the answer is the greedy one",
          m1 == g, greedy=g, sampled=m1)
    ma, mb = (chat(temperature=1.0, min_p=0.2, seed=9).choices[0].message.content for _ in (0, 1))
    check("min_p=0.2 seeded: reproduced", ma == mb, answer=ma)

    # --- logit_bias
    base = chat(CAPITAL, temperature=0, max_tokens=8).choices[0].message.content
    ban = {str(i): -100 for w in ("Paris", " Paris", "PAR", "Par") for i in
           tok(w, add_special_tokens=False).input_ids[:1]}
    biased = [chat(CAPITAL, temperature=0, max_tokens=8, logit_bias=ban).choices[0].message.content
              for _ in (0, 1)]
    check("logit_bias -100 on Paris's first tokens: the answer changes, and repeats exactly",
          "Paris" in base and "Paris" not in biased[0] and biased[0] == biased[1],
          plain=base, biased=biased[0], ids=sorted(ban))
    plus = {str(tok(" Berlin", add_special_tokens=False).input_ids[0]): 30,
            str(tok("Berlin", add_special_tokens=False).input_ids[0]): 30}
    got = chat(CAPITAL, temperature=0, max_tokens=8, logit_bias=plus).choices[0].message.content
    check("logit_bias +30 on Berlin: the answer is Berlin", "Berlin" in got, answer=got)

    # --- neutral fields change nothing
    plain = chat(temperature=0).choices[0].message.content
    same = all(chat(temperature=0, extra_body=e).choices[0].message.content == plain
               for e in ({"store": True}, {"metadata": {"conversation_id": "x"}},
                         {"prediction": {"type": "content", "content": "the Rhine"}},
                         {"service_tier": "auto"}, {"response_format": {"type": "text"}}))
    check("store, metadata, prediction, service_tier, response_format text: the same answer", same)

    # --- refusals name their field
    for label, param, kw in (
            ("response_format json_object", "response_format",
             {"extra_body": {"response_format": {"type": "json_object"}}}),
            ("n=2 streamed", "n", {"n": 2, "stream": True}),
            ("audio", "audio", {"extra_body": {"audio": {"voice": "alloy", "format": "wav"}}}),
            ("functions", "functions", {"extra_body": {"functions": [{"name": "f"}]}}),
            ("top_logprobs without logprobs", "top_logprobs", {"top_logprobs": 2}),
            ("logit_bias outside the vocabulary", "logit_bias",
             {"logit_bias": {"999999999": 5}})):
        ok, got = refused(param, **kw)
        check(f"refused: {label} -> 400 naming {param}", ok, got=got)

    bad = sum(not r["pass"] for r in results)
    print(f"{len(results) - bad}/{len(results)} checks pass")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"base": a.base, "model": model,
                                            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                            "checks": results}, indent=1))
    return bad


if __name__ == "__main__":
    sys.exit(main())
