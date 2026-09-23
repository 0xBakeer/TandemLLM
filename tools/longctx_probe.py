"""Decode speed at long context: one request per length through the served stack. A probe, not a row.

The row is 256 tokens in and 256 out, and it says nothing about what the engine does once a prompt
is an agent's 30k-token conversation, where every verify also reads the whole KV of sixteen
attention layers. This starts a test server in the served configuration (caches off, as the row
runs it), sends ONE request per length -- a real document from `tools/longprompts.py`'s set,
thinking off, greedy, 256 tokens out, "continue the text" -- and reports time to first token and the
decode rate by the row's own formula, (completion - 1) / (e2e - ttft).

    # inside ops/hold.sh: it starts an engine
    python tools/longctx_probe.py --lens 8192,32768,131072 --label before
    python tools/longctx_probe.py --lens 8192,32768,131072 --label kvfp8 --env QWEN38_KV_FP8=1

One request a length decides nothing by itself; before and after on the same prompt, in the same
hold, is the comparison, and a difference inside a few per cent is not one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import row3  # noqa: E402


def stream(port: int, text: str, max_tokens: int) -> dict:
    body = {"model": "x", "stream": True, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": text}],
            "chat_template_kwargs": {"enable_thinking": False},
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, usage, pieces = None, {}, []
    with urllib.request.urlopen(req, timeout=3600) as fh:
        for raw in fh:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    pieces.append(piece)
    e2e = time.perf_counter() - t0
    n = int(usage.get("completion_tokens") or 0)
    return {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": n,
            "ttft_s": ttft, "e2e_s": e2e,
            "tok_s": (n - 1) / (e2e - ttft) if ttft is not None and n > 1 and e2e > ttft else 0.0,
            "head": "".join(pieces)[:120]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--lens", default="8192,32768,131072")
    ap.add_argument("--data", default="bench/longprompts")
    ap.add_argument("--domain", default="prose")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--max-len", type=int, default=262144)
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--server-arg", action="append", default=[])
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--out", default="results/longctx")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    man = json.load(open(os.path.join(a.data, "manifest.json")))
    prompts = {}
    for length in [int(x) for x in a.lens.split(",")]:
        ids = np.load(os.path.join(a.data, f"ids-{length}.npy"))
        i = next(j for j, m in enumerate(man["prompts"][str(length)]) if m["domain"] == a.domain)
        prompts[length] = (tok.decode(ids[i].tolist()) + "\n\nContinue the text above from where "
                           "it stops, in the same style.")

    ns = argparse.Namespace(port=a.port, python=row3.DEFAULT_PY, pythonpath=row3.HOME / "pylibs",
                            repo=row3.REPO, max_len=a.max_len, len_fixed=0, budget=16,
                            nvfp4=row3.DEFAULT_NV, head=row3.DEFAULT_HEAD,
                            server_arg=["--drop-idle"] + a.server_arg, len_latch=True,
                            start_timeout=600)
    env = dict(kv.split("=", 1) for kv in a.env)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = row3.start_server(ns, env, out_dir / f"{a.label}-server.log")
    res = {"label": a.label, "env": env, "server_arg": a.server_arg, "max_len": a.max_len,
           "domain": a.domain, "runs": {}}
    try:
        stream(a.port, "Say hello.", 16)                  # the autotuning, outside the numbers
        for length, text in prompts.items():
            r = stream(a.port, text, a.max_tokens)
            res["runs"][str(length)] = r
            print(f"[longctx] {a.label} {length:>7}  prompt {r['prompt_tokens']}  "
                  f"ttft {r['ttft_s']:.1f} s  decode {r['tok_s']:.2f} tok/s  "
                  f"({r['completion_tokens']} tok)  {r['head'][:60]!r}", flush=True)
    finally:
        row3.stop_server(proc)
    path = out_dir / f"{a.label}.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"[longctx] {path}")


if __name__ == "__main__":
    main()
