"""The proxy's integration gate: a live TandemLLM engine behind the standalone proxy.

    python tools/proxy_gate.py --url http://<engine host>:8000/v1 [--n 20] [--key ...] \
        [--out report.json]

Starts the proxy in this process (a temp config, ledger and key file), registers the engine as a
`tandemllm` backend, and sends `--n` small streamed requests with `X-Tandem: raw` (the engine's
plain path, no conductor), one at a time, greedy, thinking off. Each answer carries the engine's
own `usage` and `timings` on its finish chunk; the proxy's ledger row for the same request is
compared with them:

    prompt_tokens, completion_tokens   equal
    decode_tps                         within 2 % of the engine's predicted_per_second
    ttft_ms                            at or above the engine's ttft_ms (the network and the proxy
                                       are inside it); the difference is reported
    proxy_ms                           p50 under 2 ms

Exit 0 when every check holds, 1 otherwise. The requests are small (a 60-token prompt, at most
96 tokens out) and read-only; nothing on the engine changes.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import statistics
import sys
import time
from urllib.parse import urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from proxy_util import Proxy  # noqa: E402

PROMPT = ("Write the numbers from one to thirty as English words, separated by commas, "
          "and nothing else.")


def _models(url: str, key: str | None) -> list[str]:
    u = urlsplit(url)
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=10)
    c.request("GET", u.path.rstrip("/") + "/models",
              headers={"Authorization": f"Bearer {key}"} if key else {})
    r = c.getresponse()
    return [m["id"] for m in json.loads(r.read())["data"]]


def _finish_fields(stream: bytes) -> dict:
    """The engine's usage and timings from the chunk that carries them."""
    out = {}
    for ev in stream.split(b"\n\n"):
        if not ev.startswith(b"data: {"):
            continue
        obj = json.loads(ev[6:])
        if obj.get("usage"):
            out["usage"] = obj["usage"]
        if obj.get("timings"):
            out["timings"] = obj["timings"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="the engine's base URL, ending in /v1")
    ap.add_argument("--key", default=None, help="the engine's bearer, if it needs one")
    ap.add_argument("--model", default=None, help="default: the first id the engine lists")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    model = a.model or _models(a.url, a.key)[0]
    env = {"TANDEM_BACKENDS": "engine1", "TANDEM_BACKEND_ENGINE1_KIND": "tandemllm",
           "TANDEM_BACKEND_ENGINE1_URL": a.url}
    if a.key:
        env["TANDEM_BACKEND_ENGINE1_KEY"] = a.key
    rows, fails = [], []
    with Proxy(env) as px:
        px.sync()
        listed = [m["id"] for m in json.loads(px.get("/v1/models")[2])["data"]]
        public = model if model in listed else f"engine1/{model}"
        print(f"[gate] engine {a.url} model {model} (public id {public}); {a.n} requests, raw, "
              f"greedy, thinking off", flush=True)
        for i in range(a.n):
            body = {"model": public, "stream": True, "max_tokens": a.max_tokens,
                    "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": f"{PROMPT} (run {i})"}]}
            st, hdr, data = px.post("/v1/chat/completions", body, headers={"X-Tandem": "raw"},
                                    timeout=300)
            eng = _finish_fields(data)
            row = px.rows()[-1]
            t, u = eng.get("timings") or {}, eng.get("usage") or {}
            rec = {"i": i, "status": st, "x_tandem": hdr.get("X-Tandem"),
                   "engine": {"prompt_tokens": u.get("prompt_tokens"),
                              "completion_tokens": u.get("completion_tokens"),
                              "decode_tps": t.get("predicted_per_second"),
                              "ttft_ms": t.get("ttft_ms"), "cache_n": t.get("cache_n")},
                   "proxy": {k: row[k] for k in ("prompt_tokens", "completion_tokens",
                                                 "decode_tps", "ttft_ms", "proxy_ms",
                                                 "usage_source", "backend")}}
            e, p = rec["engine"], rec["proxy"]
            why = []
            if st != 200:
                why.append(f"status {st}")
            if hdr.get("X-Tandem") not in (None, "raw"):
                why.append(f"X-Tandem {hdr.get('X-Tandem')}")
            if (e["prompt_tokens"], e["completion_tokens"]) != (p["prompt_tokens"],
                                                                 p["completion_tokens"]):
                why.append("token counts differ")
            if e["decode_tps"] and p["decode_tps"]:
                rec["decode_rel"] = p["decode_tps"] / e["decode_tps"] - 1
                if abs(rec["decode_rel"]) > a.tol:
                    why.append(f"decode {p['decode_tps']:.2f} vs {e['decode_tps']:.2f}")
            else:
                why.append("no decode rate")
            if e["ttft_ms"] is not None and p["ttft_ms"] is not None:
                rec["ttft_delta_ms"] = p["ttft_ms"] - e["ttft_ms"]
                if rec["ttft_delta_ms"] < 0:
                    why.append("proxy TTFT below the engine's")
            rec["fail"] = why
            if why:
                fails.append((i, why))
            rows.append(rec)
            print(f"[gate] {i:2d} tokens {p['prompt_tokens']}/{p['completion_tokens']} decode "
                  f"engine {e['decode_tps']} proxy {p['decode_tps']} ({rec.get('decode_rel', 0):+.2%})"
                  f" ttft engine {e['ttft_ms']} proxy {p['ttft_ms']} proxy_ms {p['proxy_ms']}"
                  + (f"  FAIL {why}" if why else ""), flush=True)
    pms = [r["proxy"]["proxy_ms"] for r in rows if r["proxy"]["proxy_ms"] is not None]
    rel = [abs(r["decode_rel"]) for r in rows if "decode_rel" in r]
    dt = [r["ttft_delta_ms"] for r in rows if "ttft_delta_ms" in r]
    summary = {"n": len(rows), "fails": len(fails),
               "proxy_ms_p50": statistics.median(pms) if pms else None,
               "proxy_ms_max": max(pms) if pms else None,
               "decode_rel_max": max(rel) if rel else None,
               "ttft_delta_ms_p50": statistics.median(dt) if dt else None,
               "ttft_delta_ms_max": max(dt) if dt else None}
    ok = not fails and summary["proxy_ms_p50"] is not None and summary["proxy_ms_p50"] < 2.0
    summary["verdict"] = "PASS" if ok else "FAIL"
    print(f"[gate] {json.dumps(summary)}", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "rows": rows, "at": time.time()}, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
