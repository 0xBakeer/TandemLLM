"""What a verified block commits, as a distribution, per workload and over the row.

The row's mean is a statement about its reproduction workloads and its median about fresh text
(docs/09-measurement.md). Neither says whether a block ever runs out of WIDTH, which is the one
question a deeper draft answers: a block that commits all sixteen tokens its arm could hold would
have committed more from a longer draft, and a block that commits four would not. Since 2026-09-23
the length router counts that per request -- `commits 16:4x5,16x5 cap arm 5 depth 5` on the
`[drafter]` line the server prints at the start of the NEXT request -- and this reads it back.

    # the row: every [drafter] line of a row3 server log, summed
    python tools/accept_hist.py results/row3/<label>/server.log

    # the five bench workloads through the served stack, one request each, on a test server
    # (inside ops/hold.sh: it starts an engine)
    python tools/accept_hist.py --serve --port 8001 --json results/accept-five.json

`cap arm` is a block that committed its arm's whole width (8 or 16). `cap depth` is a block that
committed everything it was handed: the whole width, or a tree the prune stopped short of the arm,
or the budget left at the end of a generation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

LINE = re.compile(r"\[drafter\] .*?commits (\S+) cap arm (\d+) depth (\d+)")


def parse_hist(token: str) -> dict[int, dict[int, int]]:
    """`8:2x5,8x1|16:4x2` -> {8: {2: 5, 8: 1}, 16: {4: 2}}; `-` -> {}."""
    out: dict[int, dict[int, int]] = {}
    if token == "-":
        return out
    for part in token.split("|"):
        arm, cells = part.split(":", 1)
        out[int(arm)] = {int(c): int(n) for c, n in (x.split("x") for x in cells.split(","))}
    return out


def parse_log(text: str) -> list[dict]:
    """One record per `[drafter]` line that carries the histogram, in log order."""
    recs = []
    for m in LINE.finditer(text):
        recs.append({"commits": parse_hist(m.group(1)), "cap_arm": int(m.group(2)),
                     "cap_depth": int(m.group(3))})
    return recs


def summarize(recs: list[dict]) -> dict:
    """Blocks, the mean commit and the two cap rates, per arm and over both."""
    arms: dict[int, dict[int, int]] = {}
    cap_arm = cap_depth = 0
    for r in recs:
        cap_arm += r["cap_arm"]
        cap_depth += r["cap_depth"]
        for arm, h in r["commits"].items():
            dst = arms.setdefault(arm, {})
            for c, n in h.items():
                dst[c] = dst.get(c, 0) + n
    out = {"arms": {}, "cap_arm": cap_arm, "cap_depth": cap_depth}
    total_blocks = total_tokens = 0
    for arm, h in sorted(arms.items()):
        blocks = sum(h.values())
        tokens = sum(c * n for c, n in h.items())
        total_blocks += blocks
        total_tokens += tokens
        out["arms"][arm] = {"blocks": blocks, "mean": tokens / blocks if blocks else 0.0,
                            "full": h.get(arm, 0), "full_pct": 100.0 * h.get(arm, 0) / blocks
                            if blocks else 0.0, "hist": dict(sorted(h.items()))}
    out["blocks"] = total_blocks
    out["mean"] = total_tokens / total_blocks if total_blocks else 0.0
    out["cap_arm_pct"] = 100.0 * cap_arm / total_blocks if total_blocks else 0.0
    out["cap_depth_pct"] = 100.0 * cap_depth / total_blocks if total_blocks else 0.0
    return out


def show(label: str, s: dict) -> str:
    rows = [f"{label}: {s['blocks']} blocks, {s['mean']:.2f} committed a block, "
            f"arm cap {s['cap_arm']} ({s['cap_arm_pct']:.1f} %), "
            f"depth cap {s['cap_depth']} ({s['cap_depth_pct']:.1f} %)"]
    for arm, a in s["arms"].items():
        hist = " ".join(f"{c}:{n}" for c, n in a["hist"].items())
        rows.append(f"    arm {arm:2d}: {a['blocks']:5d} blocks  mean {a['mean']:5.2f}  "
                    f"full {a['full']:4d} ({a['full_pct']:5.1f} %)   {hist}")
    return "\n".join(rows)


# ---------------------------------------------------------------- the five workloads, served


def _post(port: int, prompt: str, max_tokens: int) -> dict:
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as fh:
        out = json.load(fh)
    out["_wall_s"] = time.perf_counter() - t0
    return out


def serve(a) -> dict:
    """Start the served configuration on a test port, one request per workload, and read each
    request's histogram off the `[drafter]` line the NEXT request prints."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import row3
    from tools.bench_decode import PROMPTS
    ns = argparse.Namespace(port=a.port, python=Path(a.python), pythonpath=Path(a.pythonpath),
                            repo=Path(a.repo), max_len=a.max_len, len_fixed=0, budget=16,
                            nvfp4=row3.DEFAULT_NV, head=row3.DEFAULT_HEAD,
                            server_arg=["--drop-idle"] + a.server_arg, len_latch=True,
                            start_timeout=600)
    env = dict(kv.split("=", 1) for kv in a.env)
    log = Path(a.log)
    proc = row3.start_server(ns, env, log)
    order = ["warm"] + [n for n in PROMPTS for _ in range(a.repeat)] + ["flush"]
    walls = []
    try:
        for name in order:
            text = PROMPTS["chat"] if name in ("warm", "flush") else PROMPTS[name]
            r = _post(a.port, text, 16 if name == "flush" else a.max_tokens)
            walls.append((name, r["_wall_s"], r.get("usage", {}).get("completion_tokens")))
    finally:
        row3.stop_server(proc)
    recs = parse_log(log.read_text())
    # line k reports request k-1; line 0 reports nothing (the router's own initial state)
    per_req = recs[1:]
    names = order[:-1]
    if len(per_req) != len(names):
        raise SystemExit(f"{len(per_req)} [drafter] lines for {len(names)} requests; see {log}")
    by: dict[str, list[dict]] = {}
    for name, rec in zip(names, per_req):
        if name != "warm":
            by.setdefault(name, []).append(rec)
    out = {name: summarize(rs) for name, rs in by.items()}
    out["ALL"] = summarize([r for rs in by.values() for r in rs])
    out["_walls"] = walls
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", help="server logs to sum")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=262144)
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--server-arg", action="append", default=[])
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--python", default=os.path.expanduser(
        "~/recipes/ling3-flash-dgx-spark/.venv/bin/python"))
    ap.add_argument("--pythonpath", default=os.path.expanduser("~/pylibs"))
    ap.add_argument("--log", default="results/accept-serve.log")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    if a.serve:
        out = serve(a)
        for name, s in out.items():
            if not name.startswith("_"):
                print(show(name, s))
    else:
        recs = [r for p in a.logs for r in parse_log(Path(p).read_text())]
        out = {"ALL": summarize(recs), "requests": len(recs)}
        print(show(f"{len(recs)} requests", out["ALL"]))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
