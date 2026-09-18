"""The atlas row, N times a configuration, reported as a median and a spread.

Phase 9 compared two configurations by running the row once on each, read a 7 % difference in the
mean, and drew a conclusion that evaporated when the *reverted* configuration was run a third time
and read higher than the one that had just been blamed for the difference. Two runs of the same
engine differ by 10.4 % on the mean and 6.7 % on the median, because the row is fifty requests of
an ADAPTIVE policy: the latch decides each request's block width from timing taken inside that
request, so two runs make different decisions on the same prompts.

So a single row cannot decide anything smaller than about a tenth of itself, and the fix is not a
better formula -- it is more rows. This script runs the row `--runs` times a configuration,
recomputes every row from its own per-request records with the one formula

    tok/s = (completion_tokens - 1) / (e2e_s - ttft_s)

and reports, per configuration, the MEDIAN of the runs beside the spread of the runs, so that a
difference between two configurations can be read against the noise that produced it. A difference
smaller than the spread is not a result, and the script says so rather than leaving the arithmetic
to the reader.

**Use one server for all the runs of a configuration, which is the default.** Phase 9's three rows
of the release candidate came from three separate server processes and spread 10.4 % on the mean
and 6.7 % on the median; three rows of the same configuration through ONE process spread 5.9 % and
3.1 % (phase 10, 03:51). Most of the row's run-to-run noise is between processes rather than
between rows, so `--restart-each` -- which reproduces phase 9's procedure exactly -- roughly
doubles the noise the comparison has to beat. The one number it costs is `wall`: run 1 of a shared
server carries the Triton autotuning in its warm-ups and reads 20 % longer, while the fifty
measured requests do not, so compare on `mean` and `p50` and not on `wall`.

Two things it does that a shell loop does not. The atlas runner names its output file from a hash
of the configuration, so a second run of the same spec OVERWRITES the first: each run here gets its
own `--out` directory and its record is copied out under the run's own name. And the server is
started and torn down by PID rather than by pattern, on a port given on the command line, because
the release candidate serves on :8000 from another directory and must not be touched.

    # three rows of the shipped configuration, from the speed checkout, on :8001
    python tools/row3.py --label rc --runs 3 --port 8001

    # three rows of the two flags the row could not resolve in phase 9
    python tools/row3.py --label nodes-alias --runs 3 --port 8001 \
        --env QWEN38_DF2_TREE_MODE=nodes --env QWEN38_TREE_ALIAS_STATE=1

    # and the comparison, off the two reports, without touching the board
    python tools/row3.py --compare results/row3/rc.json results/row3/nodes-alias.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
REPO = Path(__file__).resolve().parent.parent

DEFAULT_PY = HOME / "recipes/ling3-flash-dgx-spark/.venv/bin/python"
DEFAULT_ATLAS = HOME / "inf-atlas/bench"
DEFAULT_SPEC = "packets/ownengine.serve-single-i256-o256-v1.json"
DEFAULT_NV = ",".join(
    str(HOME / "nvfp4" / n) for n in ("mlp-clip.safetensors", "gdn-clip.safetensors", "attn-clip.safetensors")
)
DEFAULT_HEAD = str(HOME / "nvfp4/head-fp8.safetensors")


# ---------------------------------------------------------------- the row's arithmetic


def row_stats(record: dict) -> dict:
    """Recompute one atlas run from its own per-request records.

    The first token is the prefill's, so it is subtracted, and the decode time is the request's
    wall clock past its first token. Warm-ups and failures are dropped. This is the formula every
    row in RESULTS.md since phase 6 is computed with, and it reproduces phase 6's published
    32.13 / 23.26 / 99.09 / 768 ms / 358.851 s exactly off the stored run.
    """
    reqs = record["raw"]["payload"]["requests"]
    rates, ttfts = [], []
    for r in reqs:
        if r.get("warmup") or r.get("status") != "ok":
            continue
        n = r.get("completion_tokens") or 0
        decode_s = (r["e2e_ms"] - r["ttft_ms"]) / 1000.0
        if n < 2 or decode_s <= 0:
            continue
        rates.append((n - 1) / decode_s)
        ttfts.append(r["ttft_ms"])
    rates.sort()
    it = record["raw"]["payload"]["iterations"][record["raw"]["payload"].get("selected_iteration", 0)]
    return {
        "n": len(rates),
        "mean": statistics.fmean(rates),
        "p50": statistics.median(rates),
        "p90": rates[min(len(rates) - 1, int(round(0.9 * (len(rates) - 1))))],
        "max": rates[-1],
        "ttft_p50_ms": statistics.median(ttfts),
        "wall_s": it.get("duration_s"),
    }


def across(runs: list[dict], key: str) -> dict:
    """Median of the runs, and the spread that says what the median is worth."""
    xs = sorted(r[key] for r in runs if r.get(key) is not None)
    if not xs:
        return {}
    med = statistics.median(xs)
    return {
        "median": med,
        "min": xs[0],
        "max": xs[-1],
        "spread_pct": 100.0 * (xs[-1] - xs[0]) / med if med else 0.0,
    }


# ---------------------------------------------------------------- driving the board


def health_ok(port: int, path: str = "/v1/models") -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3) as fh:
            return fh.status == 200
    except Exception:
        return False


def start_server(a, env_extra: dict[str, str], log: Path) -> subprocess.Popen:
    if health_ok(a.port):
        raise SystemExit(f"[row3] REFUSING: something already answers on :{a.port}")
    cmd = [
        str(a.python), "-u", "server/app.py",
        "--host", "127.0.0.1", "--port", str(a.port),
        "--max-len", str(a.max_len),
        "--drafter", "lenrouter", "--dflash2-path", "greedy",
        "--len-fixed", str(a.len_fixed),
        "--tree", "--budget", str(a.budget),
        "--corpus", str(a.repo / "corpus"),
        "--dflash2-ckpt", str(a.repo / "train/ft-b8-v2"),
        "--dflash2-ckpt16", str(a.repo / "train/ft-b16"),
        "--nvfp4", a.nvfp4, "--fp8-head", a.head,
        "--no-session-cache", "--no-prefix-cache", "--cache-budget-gb", "0",
        "--verbose",
    ]
    if a.len_latch:
        cmd.append("--len-latch")
    cmd += a.server_arg
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(a.pythonpath), "TZ": "Europe/Berlin"})
    env.update(env_extra)
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "w")
    print(f"[row3] start :{a.port}  {' '.join(f'{k}={v}' for k, v in sorted(env_extra.items()))}")
    proc = subprocess.Popen(cmd, cwd=a.repo, env=env, stdout=fh, stderr=subprocess.STDOUT,
                            start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < a.start_timeout:
        if proc.poll() is not None:
            raise SystemExit(f"[row3] server exited with {proc.returncode}; see {log}")
        if health_ok(a.port):
            print(f"[row3] healthy after {time.time() - t0:.0f}s")
            return proc
        time.sleep(3)
    stop_server(proc)
    raise SystemExit(f"[row3] server never became healthy; see {log}")


def stop_server(proc: subprocess.Popen | None, grace: int = 60) -> None:
    """SIGTERM the process group we started, by PID. Never a pattern: the release candidate is
    another `server/app.py` on this board and a pattern kill would take the operator's service down."""
    if proc is None or proc.poll() is not None:
        return
    print(f"[row3] stop pid {proc.pid}")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    t0 = time.time()
    while time.time() - t0 < grace:
        if proc.poll() is not None:
            return
        time.sleep(1)
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait(timeout=30)


def run_row(a, out_dir: Path, tag: str) -> dict:
    """One atlas row into a directory of its own, so the runner's hashed filename cannot collide
    with the previous run's."""
    run_out = out_dir / f"raw-{tag}"
    if run_out.exists():
        shutil.rmtree(run_out)
    run_out.mkdir(parents=True)
    cmd = [
        str(a.atlas / ".venv/bin/atlas-bench"), "run",
        "--spec", a.spec,
        "--base-url", f"http://127.0.0.1:{a.port}/v1",
        "--out", str(run_out),
        "--login", a.login, "--dedicated", "--tokenizer", a.tokenizer,
    ]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=a.atlas, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], sep="\n")
        raise SystemExit(f"[row3] atlas-bench failed ({r.returncode}) for {tag}")
    # The runner does not write into `--out`: it writes into a registry-shaped tree underneath it,
    # `results/<engine>/<org>/<model>/<hardware>/<hash>--<workload>--<hash>.json`. So the record is
    # found by walking, and the walk is what makes a per-run directory worth having -- the leaf
    # name is a hash of the spec and a second run of the same spec overwrites the first.
    found = sorted(run_out.rglob("*.json"))
    found = [p for p in found if "serve-single" in p.name] or found
    if not found:
        raise SystemExit(f"[row3] no run record in {run_out}")
    record = json.loads(found[-1].read_text())
    st = row_stats(record)
    st["tag"] = tag
    st["record"] = str(found[-1])
    st["bench_wall_s"] = round(time.time() - t0, 1)
    print(f"[row3] {tag:22s} mean {st['mean']:6.2f}  p50 {st['p50']:6.2f}  p90 {st['p90']:6.2f}  "
          f"max {st['max']:7.2f}  ttft p50 {st['ttft_p50_ms']:.0f} ms  wall {st['wall_s']:.1f} s")
    return st


# ---------------------------------------------------------------- reporting


def report(label: str, runs: list[dict]) -> dict:
    keys = ("mean", "p50", "p90", "max", "ttft_p50_ms", "wall_s")
    summary = {k: across(runs, k) for k in keys}
    print()
    print(f"=== {label}: {len(runs)} rows ===")
    print(f"{'run':<10}{'mean':>9}{'p50':>9}{'p90':>9}{'max':>9}{'ttft p50':>11}{'wall':>9}")
    for r in runs:
        print(f"{r['tag']:<10}{r['mean']:>9.2f}{r['p50']:>9.2f}{r['p90']:>9.2f}{r['max']:>9.2f}"
              f"{r['ttft_p50_ms']:>10.0f}m{r['wall_s']:>8.1f}")
    print(f"{'MEDIAN':<10}{summary['mean']['median']:>9.2f}{summary['p50']['median']:>9.2f}"
          f"{summary['p90']['median']:>9.2f}{summary['max']['median']:>9.2f}"
          f"{summary['ttft_p50_ms']['median']:>10.0f}m{summary['wall_s']['median']:>8.1f}")
    print(f"{'spread %':<10}{summary['mean']['spread_pct']:>9.1f}{summary['p50']['spread_pct']:>9.1f}"
          f"{summary['p90']['spread_pct']:>9.1f}{summary['max']['spread_pct']:>9.1f}"
          f"{summary['ttft_p50_ms']['spread_pct']:>11.1f}{summary['wall_s']['spread_pct']:>9.1f}")
    return summary


def compare(paths: list[str]) -> None:
    """Read two or more reports and say, for each statistic, whether the difference between their
    medians is bigger than the spread that produced them."""
    reports = [json.loads(Path(p).read_text()) for p in paths]
    base = reports[0]
    print(f"baseline: {base['label']}  ({len(base['runs'])} rows)")
    for other in reports[1:]:
        print(f"\nagainst:  {other['label']}  ({len(other['runs'])} rows)")
        print(f"{'stat':<12}{'baseline':>10}{'other':>10}{'delta %':>10}{'noise %':>10}{'disjoint':>10}   verdict")
        for k in ("mean", "p50", "p90", "max", "ttft_p50_ms", "wall_s"):
            b, o = base["summary"][k], other["summary"][k]
            if not b or not o:
                continue
            delta = 100.0 * (o["median"] - b["median"]) / b["median"]
            noise = max(b["spread_pct"], o["spread_pct"])
            # The ranges are disjoint when every row of one configuration beat every row of the
            # other. With three rows a side that is the strongest thing this row can say, and it
            # is the claim phase 9 believed it had after one row each.
            disjoint = o["min"] > b["max"] or o["max"] < b["min"]
            verdict = "RESOLVED" if abs(delta) > noise and disjoint else "not resolved"
            print(f"{k:<12}{b['median']:>10.2f}{o['median']:>10.2f}{delta:>+10.1f}{noise:>10.1f}"
                  f"{('yes' if disjoint else 'no'):>10}   {verdict}")
        print("\n  RESOLVED needs both: a difference bigger than the run-to-run noise, and run ranges\n"
              "  that do not overlap. 'not resolved' means the row did not decide it -- more rows or a\n"
              "  different instrument would. It does NOT mean the two configurations are equal.")


# ---------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare", nargs="+", help="read saved reports and compare them; touches no board")
    p.add_argument("--label", default="row3")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--env", action="append", default=[], metavar="K=V",
                   help="environment for the server process, repeatable")
    p.add_argument("--server-arg", action="append", default=[], help="extra server/app.py argument, repeatable")
    p.add_argument("--restart-each", action="store_true",
                   help="a fresh server process per row, which is what phase 9 did and what roughly "
                        "doubles the spread: 10.4 %% on the mean across processes against 5.9 %% "
                        "within one. The default keeps one server, and then `wall` is not "
                        "comparable across runs because run 1 pays for the autotuning")
    p.add_argument("--no-server", action="store_true", help="bench a server that is already up on --port")
    p.add_argument("--repo", type=Path, default=REPO)
    p.add_argument("--python", type=Path, default=DEFAULT_PY)
    p.add_argument("--pythonpath", type=Path, default=HOME / "pylibs")
    p.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    p.add_argument("--spec", default=DEFAULT_SPEC)
    p.add_argument("--login", default="0xBakeer")
    p.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B")
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--budget", type=int, default=16)
    p.add_argument("--len-fixed", type=int, default=0)
    p.add_argument("--len-latch", action="store_true", default=True)
    p.add_argument("--no-len-latch", dest="len_latch", action="store_false")
    p.add_argument("--nvfp4", default=DEFAULT_NV)
    p.add_argument("--head", default=DEFAULT_HEAD)
    p.add_argument("--start-timeout", type=int, default=600)
    p.add_argument("--out-dir", type=Path, default=REPO / "results/row3")
    a = p.parse_args()

    if a.compare:
        compare(a.compare)
        return

    env_extra = dict(kv.split("=", 1) for kv in a.env)
    out_dir = a.out_dir / a.label
    out_dir.mkdir(parents=True, exist_ok=True)

    runs, proc = [], None
    try:
        if not a.no_server and not a.restart_each:
            proc = start_server(a, env_extra, out_dir / "server.log")
        for i in range(1, a.runs + 1):
            if a.restart_each and not a.no_server:
                proc = start_server(a, env_extra, out_dir / f"server-{i}.log")
            runs.append(run_row(a, out_dir, f"{i}"))
            if a.restart_each and not a.no_server:
                stop_server(proc)
                proc = None
    finally:
        stop_server(proc)

    summary = report(a.label, runs)
    doc = {
        "label": a.label,
        "env": env_extra,
        "server_arg": a.server_arg,
        "spec": a.spec,
        "restart_each": bool(a.restart_each),
        "runs": runs,
        "summary": summary,
        "taken": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = a.out_dir / f"{a.label}.json"
    path.write_text(json.dumps(doc, indent=2))
    print(f"\n[row3] {path}")


if __name__ == "__main__":
    main()
