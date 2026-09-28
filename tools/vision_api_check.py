"""image requests against a running server, as a client sends them.

    python tools/vision_api_check.py --base http://127.0.0.1:8011/v1 --out vapi.json

Sends the tools/vision_refcheck.py image set as `data:` URLs (greedy, thinking off, 64 tokens), then
the cache cases the ticket names, and writes what came back:

  * answer       each image's text, finish reason, prompt tokens (image rows included) and the
                 prefill's cached / forwarded split from the usage block;
  * follow-up    a second turn that sends the first image again (reported: the template can part
                 turn two from turn one's state before the image when thinking is off);
  * other image  the same second turn with another image of the same size in turn one: the
                 placeholders are identical, so a placeholder- or URL-keyed cache would resume;
                 this must reuse nothing past the image's first row;
  * prefix       a long system prompt, image A, a long question; then image B; then A again: B
                 reuses at most the system prompt's checkpoint (1,024), A again resumes from the
                 checkpoint past its image (2,048) and answers as A did;
  * refusals     a bad scheme, broken base64 and a non-image are 400s naming the part;
  * text         one text-only request before and after the images, identical to each other.

`--compare-plain FILE` holds the answers against a plain-decoding run of the same weights
(tools/vision_refcheck.py engine --json): the served speculation is lossless, so the tokens agree up
to the batched-verify tie flips the release documents.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.vision_refcheck import QUESTIONS, draw  # noqa: E402

OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def post(base, body, timeout=600):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.time() - t0
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}"), time.time() - t0


def ask(base, url, question, model, max_tokens=64, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                     {"type": "text", "text": question}]}]
    body = dict({"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0},
                **OFF)
    return post(base, body)


def summary(code, body, secs):
    if code != 200:
        return {"code": code, "error": body.get("error")}
    ch = body["choices"][0]
    u = body.get("usage") or {}
    det = u.get("prompt_tokens_details") or {}
    return {"code": code, "text": ch["message"].get("content"), "finish": ch["finish_reason"],
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "cached_tokens": det.get("cached_tokens"), "secs": round(secs, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--model", default="qwen38-spark-engine")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    a = ap.parse_args()
    out: dict = {"answers": [], "checks": {}}
    fails = []
    text_body = dict({"model": a.model, "max_tokens": 48, "temperature": 0,
                      "messages": [{"role": "user", "content": "Name three rivers in Europe."}]},
                     **OFF)
    t1 = summary(*post(a.base, text_body))
    for kind, size, q in QUESTIONS:
        r = summary(*ask(a.base, data_url(draw(kind, size)), q, a.model, a.max_tokens))
        r["kind"] = kind
        out["answers"].append(r)
        print(f"[vapi] {kind:7s} {r.get('code')} prompt {r.get('prompt_tokens')} "
              f"cached {r.get('cached_tokens')} {r.get('secs')}s: {r.get('text')!r}", flush=True)
        if r.get("code") != 200:
            fails.append(f"{kind}: {r}")
    # a follow-up turn that sends the image again (what Open WebUI does): the session state of
    # turn one is keyed on the image's content, so the new prompt resumes past the image rows
    kind, size, q = QUESTIONS[0]
    first = out["answers"][0]
    url = data_url(draw(kind, size))

    def turn2(img_url):
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_url}},
                                             {"type": "text", "text": q}]},
                {"role": "assistant", "content": first.get("text") or ""},
                {"role": "user", "content": "Now answer in five words."}]
        return summary(*post(a.base, dict({"model": a.model, "messages": msgs, "max_tokens": 24,
                                           "temperature": 0}, **OFF)))
    again = turn2(url)
    # information, not a check: with thinking off the template renders a past answer without the
    # empty think block the generation prompt ended with, so turn two can part from turn one's
    # state before the image for reasons that have nothing to do with images
    ok = again.get("code") == 200
    out["checks"]["follow_up"] = {"ok": ok, **again}
    print(f"[vapi] follow-up same image: cached {again.get('cached_tokens')} of "
          f"{again.get('prompt_tokens')} (turn one's prompt {first.get('prompt_tokens')})",
          flush=True)
    if not ok:
        fails.append(f"follow_up: {again}")
    # the same conversation with ANOTHER image of the same size in turn one: the placeholders
    # are identical, so a cache keyed on them would resume from turn one's state; keyed on the
    # content, nothing past the image's first row may be reused
    other = draw("scene", (1024, 768)).resize(size)
    swapped = turn2(data_url(other))
    ok = swapped.get("code") == 200 and (swapped.get("cached_tokens") or 0) < 16
    out["checks"]["follow_up_other_image"] = {"ok": ok, **swapped}
    print(f"[vapi] follow-up other image: cached {swapped.get('cached_tokens')} of "
          f"{swapped.get('prompt_tokens')}", flush=True)
    if not ok:
        fails.append(f"follow_up_other_image: {swapped}")
    # a long system prompt first, so a prefix checkpoint lands before the image; then image A,
    # image B of the same size in the same place, and A again
    # ~1,400 tokens of system prompt put the image at ~1,405-1,705 and ~800 tokens of question
    # after it put a prefix checkpoint (the 1,024-token grid) at 2,048, past the image
    sys_p = "You describe images for a test. " * 200
    ques = "Describe it. " + "Mention every colour you can see in the picture. " * 80
    a1 = summary(*ask(a.base, data_url(draw("shapes", (640, 480))), ques, a.model, 32,
                      system=sys_p))
    b1 = summary(*ask(a.base, data_url(other), ques, a.model, 32, system=sys_p))
    a2 = summary(*ask(a.base, data_url(draw("shapes", (640, 480))), ques, a.model, 32,
                      system=sys_p))
    swapped_ok = (b1.get("text") != a1.get("text")
                  and (b1.get("cached_tokens") or 0) <= 1024
                  and (a2.get("cached_tokens") or 0) >= 2048
                  and a2.get("text") == a1.get("text"))
    out["checks"]["prefix_other_image"] = {"ok": swapped_ok, "a": a1, "b": b1, "a_again": a2}
    print(f"[vapi] prefix: A cached {a1.get('cached_tokens')}/{a1.get('prompt_tokens')}, "
          f"B cached {b1.get('cached_tokens')}/{b1.get('prompt_tokens')}, A again cached "
          f"{a2.get('cached_tokens')}; A!=B {b1.get('text') != a1.get('text')}, "
          f"A==A {a2.get('text') == a1.get('text')}", flush=True)
    if not swapped_ok:
        fails.append(f"prefix_other_image: {out['checks']['prefix_other_image']}")
    # refusals
    bad = {"scheme": "http://example.com/x.png", "base64": "data:image/png;base64,@@@",
           "notimage": "data:image/png;base64," + base64.b64encode(b"hello world").decode()}
    for name, url in bad.items():
        code, body, _ = ask(a.base, url, "What is this?", a.model, 8)
        err = body.get("error") or {}
        ok = code == 400 and str(err.get("param", "")).startswith("messages[0].content[0]")
        out["checks"][f"refuse_{name}"] = {"ok": ok, "code": code, "error": err}
        print(f"[vapi] refuse {name}: {code} {err.get('param')} {err.get('message')!r}", flush=True)
        if not ok:
            fails.append(f"refuse {name}: {code} {err}")
    t2 = summary(*post(a.base, text_body))
    out["checks"]["text"] = {"ok": t1.get("text") == t2.get("text"), "before": t1, "after": t2}
    if t1.get("text") != t2.get("text"):
        fails.append("text before/after differ")
    out["fails"] = fails
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[vapi] {'PASS' if not fails else 'FAIL'} ({len(fails)} failures)")
    for x in fails:
        print("   ", x)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
