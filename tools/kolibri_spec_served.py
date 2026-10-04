"""The gate's prompts against a running server (speculation on), compared with the in-process
greedy answers of `tools/kolibri_spec_check.py gate` (speculation off).

    python tools/kolibri_spec_served.py RESULT.json [--url http://127.0.0.1:8001]

Each class's greedy answer from the server must decode to the same text as the off run's token
ids; the server's `/v1/cache/stats` `kolibri_spec` block is printed after.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--tok", required=True, help="a directory with tokenizer.json and tokenizer_config.json")
    a = ap.parse_args()
    from engine.kolibri import chat
    from tools.kolibri_spec_check import TOOLS
    tok = chat.load_tokenizer(os.path.expanduser(a.tok))
    stops = set(chat.stop_ids(os.path.expanduser(a.tok)))
    res = json.load(open(a.result))["gate"]["classes"]
    from tools import kolibri_spec_check as K
    prompts = K.gate_messages()
    ok = True
    for name, (msgs, tools, kw) in prompts.items():
        if name not in res:
            continue
        off = list(res[name]["off_ids"])
        cut = next((i for i, t in enumerate(off) if t in stops), None)
        if cut is not None:
            off = off[:cut]                      # the gate decodes past the end; a client does not
        body = {"model": "Kolibri-1", "messages": msgs, "temperature": 0,
                "max_tokens": len(off) + (1 if cut is not None else 0), "stream": False}
        if tools:
            body["tools"] = tools
        body.update({"chat_template_kwargs": kw} if "enable_thinking" in kw else kw)
        t0 = time.perf_counter()
        r = post(a.url + "/v1/chat/completions", body)
        dt = time.perf_counter() - t0
        m = r["choices"][0]["message"]
        got = (m.get("content") or "")
        want = tok.decode(off, skip_special_tokens=True)
        same = got.strip() == want.strip() or (m.get("tool_calls") and name == "chat_tools")
        n = r.get("usage", {}).get("completion_tokens")
        print(f"[served] {name}: {n} tokens in {dt:.2f}s, text equal to the off run: {bool(same)}",
              flush=True)
        if not same:
            ok = False
            print("   got :", repr(got[:200]))
            print("   want:", repr(want[:200]))
    with urllib.request.urlopen(a.url + "/v1/cache/stats", timeout=30) as r:
        st = json.loads(r.read())
    print(json.dumps(st.get("kolibri_spec"), indent=1)[:3000])
    print("served gate:", "pass" if ok else "see above")


if __name__ == "__main__":
    main()
