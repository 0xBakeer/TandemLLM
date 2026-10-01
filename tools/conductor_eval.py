"""How well the conductor's classifiers pick the task class: accuracy, confusion and latency per stage.

    python tools/conductor_eval.py --arbiter http://127.0.0.1:8010 --qwen http://127.0.0.1:8000 \\
        [--set tests/fixtures/conductor/task_prompts.jsonl] [--config ~/.qwen38-spark-engine/conductor] \\
        [--out ~/.qwen38-spark-engine/conductor/evals/classifier-<time>.json] [--limit N]

Every prompt of the labelled set goes to each stage once, with the class descriptions, question
and thresholds of the router document the server loads (`--config`, else the built-in one). The
cascade is then derived the way `Router.cascade` runs it: the first stage whose answer reaches its
threshold decides; nobody sure is "no answer" (the router keeps the session's class then).

Per stage: `accuracy` (right answers over all prompts; an answer below the threshold counts as
none), `answered` (reached the threshold), `accuracy_answered`, `top1` (the label with the highest
probability, threshold ignored), p50 and p95 latency, and the confusion table (true class ->
answer, "—" for none). The JSON is what the dashboard's Compare screen reads from the box's
`evals/` directory. The labelled set is written from task types, not from anyone's chats.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server.conductor import schema, signals  # noqa: E402
from server.conductor.classify import Arbiter, Qwen  # noqa: E402
from server.conductor.store import Store  # noqa: E402

NONE = "—"


def pct(vals, q):
    if not vals:
        return None
    v = sorted(vals)
    return round(v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))], 1)


def summarise(rows: list[dict], key: str, labels: list[str]) -> dict:
    """One stage's figures from the per-prompt records."""
    conf = {t: {} for t in labels}
    correct = answered = top1 = 0
    ms = []
    for r in rows:
        a = r[key]
        if a is None:
            continue
        pred = a["label"] if a.get("accepted") else NONE
        conf[r["label"]][pred] = conf[r["label"]].get(pred, 0) + 1
        correct += pred == r["label"]
        answered += pred != NONE
        top1 += a.get("label") == r["label"]
        ms.append(a["ms"])
    n = len(rows)
    return {"n": n, "answered": answered, "correct": correct,
            "accuracy": round(correct / n, 4) if n else 0.0,
            "accuracy_answered": round(correct / answered, 4) if answered else None,
            "top1": round(top1 / n, 4) if n else 0.0,
            "p50_ms": pct(ms, 0.5), "p95_ms": pct(ms, 0.95), "confusion": conf}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default=os.path.join(ROOT, "tests", "fixtures", "conductor",
                                                  "task_prompts.jsonl"))
    ap.add_argument("--arbiter", default=None, help="arbiter base URL (the decider stage)")
    ap.add_argument("--arbiter-model", default="auto")
    ap.add_argument("--qwen", default=None, help="engine base URL (the Qwen stage, raw model)")
    ap.add_argument("--qwen-model", default="qwen38-spark-engine")
    ap.add_argument("--config", default=None, help="the conductor directory whose router to use")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    router = Store(a.config).peek("router")[1] if a.config else schema.seed("router")
    clf = router["classifier"]
    stages = {s["kind"]: s for s in clf["stages"]}
    options = {c["id"]: c["description"] for c in router["classes"]}
    labels = list(options)
    items = [json.loads(line) for line in open(a.set) if line.strip()]
    if a.limit:
        items = items[:a.limit]
    arb = Arbiter(a.arbiter, a.arbiter_model, stages.get("arbiter", {}).get("timeout_ms", 400)) \
        if a.arbiter else None
    qwen = Qwen(a.qwen, a.qwen_model, stages.get("qwen", {}).get("timeout_ms", 8000)) \
        if a.qwen else None
    listing = "\n".join(f"- {k}: {v}" for k, v in options.items())
    prompt = clf["qwen_prompt"].replace("{classes}", listing)
    rows = []
    t0 = time.time()
    for i, it in enumerate(items):
        text = signals.digest(it["text"], int(clf["digest_chars"]))
        rec = {"text": it["text"], "label": it["label"], "arbiter": None, "qwen": None}
        if arb is not None:
            ans = arb.choice(text, clf["question"], options)
            rec["arbiter"] = dict(ans.as_dict(), accepted=ans.error is None and ans.label in options
                                  and ans.confidence >= stages["arbiter"]["accept"])
        if qwen is not None:
            ans = qwen.choice(text, prompt, options, max_chars=int(clf["digest_chars"]))
            rec["qwen"] = dict(ans.as_dict(), accepted=ans.error is None and ans.label in options
                               and ans.confidence >= stages["qwen"]["accept"])
        # the cascade in the router's order: the first stage that is sure decides
        casc, ms = None, 0.0
        for st in clf["stages"]:
            r = rec.get(st["kind"])
            if not st.get("enabled", True) or r is None:
                continue
            ms += r["ms"]
            if r["accepted"]:
                casc = r["label"]
                break
        rec["cascade"] = {"label": casc, "accepted": casc is not None, "ms": round(ms, 2)}
        rows.append(rec)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(items)} in {time.time() - t0:.0f} s", file=sys.stderr, flush=True)

    out = {"kind": "classifier",
           "created_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "set": {"name": os.path.relpath(a.set, ROOT), "n": len(rows), "classes": labels},
           "config": {"router_from": a.config or "built-in", "thresholds": {
               k: v["accept"] for k, v in stages.items()}},
           "models": {"arbiter": a.arbiter_model if arb else None,
                      "qwen": a.qwen_model if qwen else None},
           "stages": {}, "rows": rows}
    for key in ("arbiter", "qwen", "cascade"):
        if any(r[key] is not None for r in rows) and (key != "cascade" or arb or qwen):
            out["stages"][key] = summarise(rows, key, labels)
    print(f"{'stage':8} {'accuracy':>9} {'answered':>9} {'acc|ans':>8} {'top1':>6} {'p50':>7} "
          f"{'p95':>7}")
    for k, s in out["stages"].items():
        aa = f"{s['accuracy_answered']:.3f}" if s["accuracy_answered"] is not None else "—"
        print(f"{k:8} {s['accuracy']:9.3f} {s['answered']:5}/{s['n']:<3} {aa:>8} "
              f"{s['top1']:6.3f} {s['p50_ms'] or 0:7.1f} {s['p95_ms'] or 0:7.1f}")
    path = a.out or os.path.join(os.path.expanduser(a.config or "~/.qwen38-spark-engine/conductor"),
                                 "evals", f"classifier-{time.strftime('%Y%m%dT%H%M%S')}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
